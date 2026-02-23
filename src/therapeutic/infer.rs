//! ML inference, e.g. for Therapeutic properties. Shares the model and relevant
//! properties with `train.rs`.

use std::{collections::HashMap, fs, io, time::Instant};

use bio_files::md_params::ForceFieldParams;
use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    module::Module,
    prelude::Int,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, NamedMpkFileRecorder, Recorder},
    tensor::{Tensor, TensorData, backend::Backend},
};

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc,
        gnn::{
            GraphData, GraphDataComponent, GraphDataSpacial, PER_COMP_SCALARS, PER_EDGE_COMP_FEATS,
            PER_EDGE_FEATS, PER_PHARM_SCALARS, PER_SPACIAL_EDGE_FEATS, pad_adj_and_mask,
            pad_edge_feats,
        },
        train::{
            MAX_ATOMS, MAX_COMPS, MAX_PHARM, Model, ModelConfig, StandardScaler, mlp_feats_from_mol,
        },
    },
};

// todo: Stack overflow with Burn CPU
// CPU (i.e.NdArray) seems to be much faster for inference than GPU.
// type InferBackend = Wgpu;
type InferBackend = NdArray;
// type InferBackend = Cpu;

pub struct Infer {
    model: Model<InferBackend>,
    scaler: StandardScaler,
    device: <InferBackend as Backend>::Device,
}

impl Infer {
    /// Helper used by both loading approaches.
    fn load(cfg_bytes: &[u8], scaler_bytes: &[u8]) -> io::Result<(Self, NdArrayDevice)> {
        let config: ModelConfig = serde_json::from_slice(cfg_bytes)?;
        let scaler: StandardScaler = serde_json::from_slice(scaler_bytes)?;

        let device = Default::default();

        // Prevent randomness in results.
        const SEED: u64 = 42;
        InferBackend::seed(&device, SEED);

        let model = config.init::<InferBackend>(&device);

        Ok((
            Self {
                model,
                scaler,
                device,
            },
            device,
        ))
    }

    /// Load the model and related data from file. Use this for the training and eval executable. Since we
    /// evaluate from the same run as training, the training  data does not get embedded there, so
    /// we load from disk.
    pub fn load_from_file(data_set: DatasetTdc) -> io::Result<Self> {
        let (model_path, scaler_path, cfg_path) = data_set.model_paths();

        let scaler_bytes = fs::read(scaler_path)?;
        let cfg_bytes = fs::read(&cfg_path)?;

        let (mut model, device) = Self::load(&cfg_bytes, &scaler_bytes)?;

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.model = model
            .model
            .load_file(model_path, &recorder, &device)
            .map_err(|e| io::Error::other(e))?;

        Ok(model)
    }

    /// Load the model from our include_bytes, i.e. part of the binary. Use this for inference, i.e.
    /// for the main application
    #[cfg(not(feature = "train"))]
    pub fn load_from_embedded(data_set: DatasetTdc) -> io::Result<Self> {
        let (model_bytes, scaler_bytes, cfg_bytes) = data_set.data()?;

        let (mut model, device) = Self::load(cfg_bytes, scaler_bytes)?;

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();

        let record: <Model<InferBackend> as Module<InferBackend>>::Record = recorder
            .load(model_bytes.to_vec(), &device)
            .map_err(io::Error::other)?;

        model.model = model.model.load_record(record);

        Ok(model)
    }

    pub fn infer(
        &self,
        mol: &MoleculeSmall,
        mut feat_params: Vec<f32>,
        ff_params: &ForceFieldParams,
    ) -> io::Result<f32> {
        let start = Instant::now();

        // Prepare Globals
        let n_feat_params = feat_params.len();
        self.scaler.apply_in_place(&mut feat_params);

        // Extract Graph Data (New Return Signature)
        let graph_atom_bond = GraphData::new(&mol, ff_params)?;

        let Some(comps) = &mol.components else {
            return Err(io::Error::other("Missing components in ML inference"));
        };

        let graph_comp = GraphDataComponent::new(&comps)?;
        let graph_spacial =
            GraphDataSpacial::new(mol).unwrap_or_else(|_| GraphDataSpacial::empty());

        // 3. Pad Data (Replicating Batcher Logic for BatchSize=1)
        let num_atoms = graph_atom_bond.num_atoms;
        let n = num_atoms.min(MAX_ATOMS);

        let num_comps = graph_comp.num_comps;
        let n_comps = num_comps.min(MAX_COMPS);

        // Pad Indices (Int) --
        let p_el_ids = {
            let mut v = Vec::with_capacity(MAX_ATOMS);
            v.extend_from_slice(&graph_atom_bond.elem_indices[0..n]);
            v.extend(std::iter::repeat_n(0, MAX_ATOMS - n));

            v
        };

        let p_ff_ids = {
            let mut v = Vec::with_capacity(MAX_ATOMS);
            v.extend_from_slice(&graph_atom_bond.ff_indices[0..n]);
            v.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

            v
        };

        let p_comp_type_ids = {
            let mut v = Vec::with_capacity(MAX_COMPS);
            v.extend_from_slice(&graph_comp.comp_type_indices[0..n_comps]);
            v.extend(std::iter::repeat_n(0, MAX_COMPS - n_comps));

            v
        };

        // -- Pad Comp Scalars (Float) --
        let mut p_comp_scalars = Vec::with_capacity(MAX_COMPS * PER_COMP_SCALARS);
        p_comp_scalars.extend_from_slice(&graph_comp.scalars[0..n_comps * PER_COMP_SCALARS]);
        p_comp_scalars.extend(std::iter::repeat_n(
            0.0f32,
            (MAX_COMPS - n_comps) * PER_COMP_SCALARS,
        ));

        // -- Pad Scalars (Float) --
        // Calculate dimensionality (should be 2: charge + degree)
        let n_scalars_per_atom = if num_atoms > 0 {
            graph_atom_bond.scalars.len() / num_atoms
        } else {
            2
        };
        let mut p_scalars = Vec::with_capacity(MAX_ATOMS * n_scalars_per_atom);
        p_scalars.extend_from_slice(&graph_atom_bond.scalars[0..n * n_scalars_per_atom]);
        p_scalars.extend(std::iter::repeat_n(
            0.0,
            (MAX_ATOMS - n) * n_scalars_per_atom,
        ));

        // -- Pad Adj & Mask (Float) --
        // Use the helper from train.rs
        let (padded_adj, padded_mask) =
            pad_adj_and_mask(&graph_atom_bond.adj, num_atoms, MAX_ATOMS);
        let p_edge_feats = pad_edge_feats(
            &graph_atom_bond.edge_feats,
            num_atoms,
            PER_EDGE_FEATS,
            MAX_ATOMS,
        );

        let (padded_adj_comp, padded_mask_comp) =
            pad_adj_and_mask(&graph_comp.adj, num_comps, MAX_COMPS);
        let p_edge_comp_feats = pad_edge_feats(
            &graph_comp.edge_feats,
            num_comps,
            PER_EDGE_COMP_FEATS,
            MAX_COMPS,
        );

        // Spatial (pharmacophore) graph padding
        let num_pharm = graph_spacial.num_nodes;
        let n_pharm = num_pharm.min(MAX_PHARM);

        let p_pharm_ids = {
            let mut v = Vec::with_capacity(MAX_PHARM);
            v.extend_from_slice(&graph_spacial.pharm_type_indices[0..n_pharm]);
            v.extend(std::iter::repeat_n(0_i32, MAX_PHARM - n_pharm));
            v
        };

        let mut p_pharm_scalars = Vec::with_capacity(MAX_PHARM * PER_PHARM_SCALARS);
        p_pharm_scalars.extend_from_slice(&graph_spacial.scalars[0..n_pharm * PER_PHARM_SCALARS]);
        p_pharm_scalars.extend(std::iter::repeat_n(
            0.0_f32,
            (MAX_PHARM - n_pharm) * PER_PHARM_SCALARS,
        ));

        let (padded_adj_pharm, padded_mask_pharm) =
            pad_adj_and_mask(&graph_spacial.adj, num_pharm, MAX_PHARM);
        let p_edge_pharm_feats = pad_edge_feats(
            &graph_spacial.edge_feats,
            num_pharm,
            PER_SPACIAL_EDGE_FEATS,
            MAX_PHARM,
        );

        // 4. Create Tensors
        let t_param_feats = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(feat_params, [1, n_feat_params]),
            &self.device,
        );

        // Int Tensors for Embeddings
        let t_elem_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(p_el_ids, [1, MAX_ATOMS]),
            &self.device,
        );

        let t_ff_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(p_ff_ids, [1, MAX_ATOMS]),
            &self.device,
        );

        let t_comp_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(p_comp_type_ids, [1, MAX_COMPS]),
            &self.device,
        );

        // Float Tensors for the rest
        let t_scalars = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(p_scalars, [1, MAX_ATOMS, n_scalars_per_atom]),
            &self.device,
        );

        let t_adj = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_adj, [1, MAX_ATOMS, MAX_ATOMS]),
            &self.device,
        );

        let t_edge_feats = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(p_edge_feats, [1, MAX_ATOMS, MAX_ATOMS, PER_EDGE_FEATS]),
            &self.device,
        );

        let t_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_mask, [1, MAX_ATOMS, 1]),
            &self.device,
        );

        // Comp tensors
        let t_comp_scalars = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(p_comp_scalars, [1, MAX_COMPS, PER_COMP_SCALARS]),
            &self.device,
        );

        let t_comp_adj = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_adj_comp, [1, MAX_COMPS, MAX_COMPS]),
            &self.device,
        );

        let t_comp_edge_feats = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(
                p_edge_comp_feats,
                [1, MAX_COMPS, MAX_COMPS, PER_EDGE_COMP_FEATS],
            ),
            &self.device,
        );

        let t_comp_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_mask_comp, [1, MAX_COMPS, 1]),
            &self.device,
        );

        // Spatial (pharmacophore) tensors
        let t_pharm_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(p_pharm_ids, [1, MAX_PHARM]),
            &self.device,
        );

        let t_pharm_scalars = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(p_pharm_scalars, [1, MAX_PHARM, PER_PHARM_SCALARS]),
            &self.device,
        );

        let t_pharm_adj = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_adj_pharm, [1, MAX_PHARM, MAX_PHARM]),
            &self.device,
        );

        let t_pharm_edge_feats = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(
                p_edge_pharm_feats,
                [1, MAX_PHARM, MAX_PHARM, PER_SPACIAL_EDGE_FEATS],
            ),
            &self.device,
        );

        let t_pharm_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_mask_pharm, [1, MAX_PHARM, 1]),
            &self.device,
        );

        // 5. Forward Pass
        let forward_tensor = self.model.forward(
            t_elem_ids,
            t_ff_ids,
            t_scalars,
            t_adj,
            t_edge_feats,
            t_mask,
            t_comp_ids,
            t_comp_scalars,
            t_comp_adj,
            t_comp_edge_feats,
            t_comp_mask,
            t_pharm_ids,
            t_pharm_scalars,
            t_pharm_adj,
            t_pharm_edge_feats,
            t_pharm_mask,
            t_param_feats,
        );

        let val = forward_tensor
            .into_data()
            .to_vec::<f32>()
            .map_err(|_| io::Error::other("Tensor error"))?[0];

        let real_val = self.scaler.denormalize_target(val);

        Ok(real_val)
    }
}

/// Convenience function that may apply to many properties. Assumes a standard feature set.
/// We cache any loaded models.
///
/// For normal application use: Load from the embedded models. For train/eval pipelines, load
/// from file.
pub fn infer_general(
    mol: &MoleculeSmall,
    dataset: DatasetTdc,
    models: &mut HashMap<DatasetTdc, Infer>,
    ff_params: &ForceFieldParams,
    load_from_file: bool,
) -> io::Result<f32> {
    let feat_params = mlp_feats_from_mol(mol)?;

    let infer = match models.get_mut(&dataset) {
        Some(inf) => inf,
        None => {
            // let infer =
            let infer = if load_from_file {
                Infer::load_from_file(dataset)?
            } else {
                #[cfg(not(feature = "train"))]
                {
                    Infer::load_from_embedded(dataset)?
                }
                #[cfg(feature = "train")]
                {
                    return Err(io::Error::other(
                        "load_from_embedded not available in train build",
                    ));
                }
            };

            models.insert(dataset, infer);
            models.get_mut(&dataset).unwrap()
        }
    };

    infer.infer(mol, feat_params, ff_params)
}
