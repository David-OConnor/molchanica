//! ML inference, e.g. for Therapeutic properties. Shares the model and relevant
//! properties with `train.rs`.

use bio_files::md_params::ForceFieldParams;
use burn::{
    backend::NdArray,
    module::Module,
    prelude::Int,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::Backend},
};
use std::{collections::HashMap, fs, io, time::Instant};

use crate::therapeutic::gnn::{PER_EDGE_FEATS, pad_edge_feats};
use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc,
        gnn::{mol_to_graph_data, pad_adj_and_mask},
        train::{MAX_ATOMS, Model, ModelConfig, StandardScaler, model_paths, param_feats_from_mol},
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
    pub fn load(data_set: DatasetTdc) -> io::Result<Self> {
        let (model_path, scaler_path, cfg_path) = model_paths(data_set);

        // Model extension is inferred automatically.
        let cfg_bytes = fs::read(&cfg_path)?;
        let scaler_bytes = fs::read(scaler_path)?;

        let config: ModelConfig = serde_json::from_slice(&cfg_bytes)?;
        let scaler: StandardScaler = serde_json::from_slice(&scaler_bytes)?;

        let device = Default::default();

        // Prevent randomness in results.
        const SEED: u64 = 42;
        InferBackend::seed(&device, SEED);

        let mut model = config.init::<InferBackend>(&device);

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model = model
            .load_file(model_path, &recorder, &device)
            .map_err(|e| io::Error::other(e))?;

        Ok(Self {
            model,
            scaler,
            device,
        })
    }

    pub fn infer(
        &self,
        mol: &MoleculeSmall,
        mut feat_params: Vec<f32>,
        ff_params: &ForceFieldParams,
    ) -> io::Result<f32> {
        let start = Instant::now();

        // 1. Prepare Globals
        let n_feat_params = feat_params.len();
        self.scaler.apply_in_place(&mut feat_params);

        // 2. Extract Graph Data (New Return Signature)
        let graph = mol_to_graph_data(&mol, ff_params)?;

        // 3. Pad Data (Replicating Batcher Logic for BatchSize=1)
        let num_atoms = graph.num_atoms;
        let n = num_atoms.min(MAX_ATOMS);

        // -- Pad Indices (Int) --
        let mut p_el_ids = Vec::with_capacity(MAX_ATOMS);
        p_el_ids.extend_from_slice(&graph.elem_indices[0..n]);
        p_el_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

        let mut p_ff_ids = Vec::with_capacity(MAX_ATOMS);
        p_ff_ids.extend_from_slice(&graph.ff_indices[0..n]);
        p_ff_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

        // -- Pad Scalars (Float) --
        // Calculate dimensionality (should be 2: charge + degree)
        let n_scalars_per_atom = if num_atoms > 0 {
            graph.scalars.len() / num_atoms
        } else {
            2
        };
        let mut p_scalars = Vec::with_capacity(MAX_ATOMS * n_scalars_per_atom);
        p_scalars.extend_from_slice(&graph.scalars[0..n * n_scalars_per_atom]);
        p_scalars.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * n_scalars_per_atom));

        // -- Pad Adj & Mask (Float) --
        // Use the helper from train.rs
        let (padded_adj, padded_mask) = pad_adj_and_mask(&graph.adj, num_atoms);
        let p_edge_feats = pad_edge_feats(&graph.edge_feats, num_atoms);

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

        // 5. Forward Pass
        let forward_tensor = self.model.forward(
            t_elem_ids,
            t_ff_ids,
            t_scalars,
            t_adj,
            t_edge_feats,
            t_mask,
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
pub fn infer_general(
    mol: &MoleculeSmall,
    dataset: DatasetTdc,
    models: &mut HashMap<DatasetTdc, Infer>,
    ff_params: &ForceFieldParams,
) -> io::Result<f32> {
    let feat_params = param_feats_from_mol(mol)?;

    let infer = match models.get_mut(&dataset) {
        Some(inf) => inf,
        None => {
            let infer = Infer::load(dataset)?;
            models.insert(dataset, infer);
            models.get_mut(&dataset).unwrap()
        }
    };

    infer.infer(mol, feat_params, ff_params)
}
