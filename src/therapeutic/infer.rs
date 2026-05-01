//! ML inference, e.g. for Therapeutic properties. Shares the model and relevant
//! properties with `train.rs`.

use std::{collections::HashMap, fs, io, time::Instant};

use bio_files::md_params::ForceFieldParams;
#[cfg(not(feature = "train"))]
use burn::record::{NamedMpkBytesRecorder, Recorder};
use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    module::Module,
    prelude::Int,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::Backend},
};

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc,
        gnn::{
            ATOM_GNN_EDGE_LAYERS, ATOM_GNN_PER_EDGE_FEATS_LAYER_0, GRAPH_ANALYSIS_FEATURE_VERSION,
            GraphDataComponent, PER_ATOM_SCALARS, PER_COMP_SCALARS, PER_EDGE_COMP_FEATS,
            PER_PHARM_SCALARS, PER_SPACIAL_EDGE_FEATS,
            atom_bond::{GraphDataAtom, pad_atom_adj_and_mask, pad_atom_edge_feats},
            pad_adj_and_mask, pad_edge_feats, pad_indices, pad_scalars,
            spacial::GraphDataSpacial,
        },
        mlp::mlp_feats_from_mol,
        non_nn_ml::GnnAnalysisTools,
        train::{MAX_ATOMS, MAX_COMPS, MAX_PHARM, Model, ModelConfig, StandardScaler},
    },
};
// todo: Stack overflow with Burn CPU
// CPU (i.e.NdArray) seems to be much faster for inference than GPU.
// The `Cpu` backend is a newer one: "Burn CPU", but we currently get a stack overflow when using it.
// `Wgpu` is a good default for GPU inference, if that turns out to be faster once we have updated
// our algorithm. `Cuda` is another option; it uses `Cudarc`, and limits inference to Nvidia GPUs.
// We may have to make sure its Cudarc linking settings match ours.

// type InferBackend = Wgpu;
type InferBackend = NdArray;
// type InferBackend = Cpu;
// todo: Also look at Vulkan.

pub struct Infer {
    model: Model<InferBackend>,
    scaler: StandardScaler,
    device: <InferBackend as Backend>::Device,
    atom_graph_analysis: GnnAnalysisTools,
    comp_graph_analysis: GnnAnalysisTools,
    spacial_graph_analysis: GnnAnalysisTools,
}

impl Infer {
    /// Helper used by both loading approaches.
    fn load(cfg_bytes: &[u8], scaler_bytes: &[u8]) -> io::Result<(Self, NdArrayDevice)> {
        let mut config_json: serde_json::Value = serde_json::from_slice(cfg_bytes)?;

        if let Some(obj) = config_json.as_object_mut() {
            obj.entry("atom_graph_analysis")
                .or_insert_with(|| serde_json::json!(GnnAnalysisTools::default()));
            obj.entry("comp_graph_analysis")
                .or_insert_with(|| serde_json::json!(GnnAnalysisTools::default()));
            obj.entry("spacial_graph_analysis")
                .or_insert_with(|| serde_json::json!(GnnAnalysisTools::default()));
            let feature_version = obj
                .entry("graph_analysis_feature_version")
                .or_insert_with(|| serde_json::json!(1))
                .as_u64()
                .unwrap_or(1) as u8;

            if feature_version < GRAPH_ANALYSIS_FEATURE_VERSION {
                for key in [
                    "atom_graph_analysis",
                    "comp_graph_analysis",
                    "spacial_graph_analysis",
                ] {
                    if let Some(analysis) = obj.get_mut(key).and_then(|v| v.as_object_mut()) {
                        analysis.insert("lhn_similarity".to_string(), serde_json::json!(false));
                    }
                }
            }
        }
        let config: ModelConfig = serde_json::from_value(config_json)?;
        if config.edge_feat_dim != ATOM_GNN_PER_EDGE_FEATS_LAYER_0
            || config.comp_edge_feat_dim != PER_EDGE_COMP_FEATS
        {
            return Err(io::Error::other(format!(
                "Therapeutic model edge feature dims are incompatible with the current graph \
                 layout. Model has atom/component edge dims {}/{} but code expects {}/{}. \
                 Retrain or regenerate the model artifacts.",
                config.edge_feat_dim,
                config.comp_edge_feat_dim,
                ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
                PER_EDGE_COMP_FEATS,
            )));
        }
        let scaler: StandardScaler = serde_json::from_slice(scaler_bytes)?;
        let atom_graph_analysis = config.atom_graph_analysis.clone();
        let comp_graph_analysis = config.comp_graph_analysis.clone();
        let spacial_graph_analysis = config.spacial_graph_analysis.clone();

        let device = Default::default();

        // Prevent randomness in results.
        const SEED: u64 = 69;
        InferBackend::seed(&device, SEED);

        let model = config.init::<InferBackend>(&device);

        Ok((
            Self {
                model,
                scaler,
                device,
                atom_graph_analysis,
                comp_graph_analysis,
                spacial_graph_analysis,
            },
            device,
        ))
    }

    /// Load the model and related data from file. Use this for both training and inference pipelines
    /// executable. Since we evaluate from the same run as training, the training data does not get embedded there, so
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
            .map_err(io::Error::other)?;

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

    /// Entry point for inference. Used by the main program to estimate properties,
    /// and in the evaluation pipeline.
    pub fn infer(
        &self,
        mol: &MoleculeSmall,
        mut feat_params: Vec<f32>,
        ff_params: &ForceFieldParams,
    ) -> io::Result<f32> {
        #[allow(unused)]
        let start = Instant::now();

        let n_feat_params = feat_params.len();
        self.scaler.apply_in_place(&mut feat_params);

        let graph_atom_bond = GraphDataAtom::new(mol, ff_params, &self.atom_graph_analysis)?;

        let Some(comps) = &mol.components else {
            return Err(io::Error::other("Missing components in ML inference"));
        };

        let graph_comp = GraphDataComponent::new(mol, comps, ff_params, &self.comp_graph_analysis)?;
        // GraphDataSpacial::new returns Ok(empty) when characterization is missing
        // or no pharmacophore sites are present, so this never errors.
        let graph_spacial = GraphDataSpacial::new(mol, &self.spacial_graph_analysis)?;

        // 3. Pad Data (Replicating Batcher Logic for BatchSize=1)
        let num_atoms = graph_atom_bond.num_atoms;
        let num_comps = graph_comp.num_comps;
        let num_pharm = graph_spacial.num_nodes;

        // Atom graph
        let p_el_ids = pad_indices(&graph_atom_bond.elem_indices, num_atoms, MAX_ATOMS);
        let p_ff_ids = pad_indices(&graph_atom_bond.ff_indices, num_atoms, MAX_ATOMS);
        let p_scalars = pad_scalars(
            &graph_atom_bond.scalars,
            num_atoms,
            PER_ATOM_SCALARS,
            MAX_ATOMS,
        );
        let (padded_adj, padded_mask) =
            pad_atom_adj_and_mask(&graph_atom_bond.adj, num_atoms, MAX_ATOMS);
        let p_edge_feats = pad_atom_edge_feats(&graph_atom_bond.edge_feats, num_atoms, MAX_ATOMS);

        // Component graph
        let p_comp_type_ids = pad_indices(&graph_comp.comp_type_indices, num_comps, MAX_COMPS);
        let p_comp_scalars =
            pad_scalars(&graph_comp.scalars, num_comps, PER_COMP_SCALARS, MAX_COMPS);
        let (padded_adj_comp, padded_mask_comp) =
            pad_adj_and_mask(&graph_comp.adj, num_comps, MAX_COMPS);
        let p_edge_comp_feats = pad_edge_feats(
            &graph_comp.edge_feats,
            num_comps,
            PER_EDGE_COMP_FEATS,
            MAX_COMPS,
        );

        // Spatial (pharmacophore) graph
        let p_pharm_ids = pad_indices(&graph_spacial.pharm_type_indices, num_pharm, MAX_PHARM);
        let p_pharm_scalars = pad_scalars(
            &graph_spacial.scalars,
            num_pharm,
            PER_PHARM_SCALARS,
            MAX_PHARM,
        );
        let (padded_adj_pharm, padded_mask_pharm) =
            pad_adj_and_mask(&graph_spacial.adj, num_pharm, MAX_PHARM);
        let p_edge_pharm_feats = pad_edge_feats(
            &graph_spacial.edge_feats,
            num_pharm,
            PER_SPACIAL_EDGE_FEATS,
            MAX_PHARM,
        );

        // Create Tensors
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
            TensorData::new(p_scalars, [1, MAX_ATOMS, PER_ATOM_SCALARS]),
            &self.device,
        );

        let t_adj = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(padded_adj, [1, ATOM_GNN_EDGE_LAYERS, MAX_ATOMS, MAX_ATOMS]),
            &self.device,
        );

        let t_edge_feats = Tensor::<InferBackend, 5>::from_data(
            TensorData::new(
                p_edge_feats,
                [
                    1,
                    ATOM_GNN_EDGE_LAYERS,
                    MAX_ATOMS,
                    MAX_ATOMS,
                    ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
                ],
            ),
            &self.device,
        );

        let t_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_mask, [1, MAX_ATOMS, 1]),
            &self.device,
        );
        let atom_graph_analysis = if graph_atom_bond.analysis_features.is_empty() {
            vec![0.0]
        } else {
            graph_atom_bond.analysis_features.clone()
        };
        let t_atom_graph_analysis = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(atom_graph_analysis.clone(), [1, atom_graph_analysis.len()]),
            &self.device,
        );
        let comp_graph_analysis_dim = self.comp_graph_analysis.feature_dim().max(1);
        let comp_graph_analysis = if graph_comp.analysis_features.is_empty() {
            vec![0.0; comp_graph_analysis_dim]
        } else {
            debug_assert_eq!(graph_comp.analysis_features.len(), comp_graph_analysis_dim);
            graph_comp.analysis_features.clone()
        };
        let t_comp_graph_analysis = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(comp_graph_analysis, [1, comp_graph_analysis_dim]),
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
        let spacial_graph_analysis_dim = self.spacial_graph_analysis.feature_dim().max(1);
        let spacial_graph_analysis = if graph_spacial.analysis_features.is_empty() {
            vec![0.0; spacial_graph_analysis_dim]
        } else {
            debug_assert_eq!(
                graph_spacial.analysis_features.len(),
                spacial_graph_analysis_dim
            );
            graph_spacial.analysis_features.clone()
        };
        let t_spacial_graph_analysis = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(spacial_graph_analysis, [1, spacial_graph_analysis_dim]),
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
            t_comp_graph_analysis,
            t_pharm_ids,
            t_pharm_scalars,
            t_pharm_adj,
            t_pharm_edge_feats,
            t_pharm_mask,
            t_spacial_graph_analysis,
            t_atom_graph_analysis,
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

impl Infer {
    /// Batched inference: pack all molecules into a single forward pass. Much faster
    /// than calling `infer` per-molecule when you have many to evaluate (e.g. eval).
    /// Errors out if any single mol fails feature extraction or graph construction.
    pub fn infer_batch(
        &self,
        mols: &[&MoleculeSmall],
        ff_params: &ForceFieldParams,
    ) -> io::Result<Vec<f32>> {
        if mols.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = mols.len();

        let mut all_elem = Vec::with_capacity(batch_size * MAX_ATOMS);
        let mut all_ff = Vec::with_capacity(batch_size * MAX_ATOMS);
        let mut all_scalars = Vec::with_capacity(batch_size * MAX_ATOMS * PER_ATOM_SCALARS);
        let mut all_adj =
            Vec::with_capacity(batch_size * ATOM_GNN_EDGE_LAYERS * MAX_ATOMS * MAX_ATOMS);
        let mut all_edge = Vec::with_capacity(
            batch_size
                * ATOM_GNN_EDGE_LAYERS
                * MAX_ATOMS
                * MAX_ATOMS
                * ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
        );
        let mut all_mask = Vec::with_capacity(batch_size * MAX_ATOMS);

        let mut all_comp_ids = Vec::with_capacity(batch_size * MAX_COMPS);
        let mut all_comp_scalars = Vec::with_capacity(batch_size * MAX_COMPS * PER_COMP_SCALARS);
        let mut all_comp_adj = Vec::with_capacity(batch_size * MAX_COMPS * MAX_COMPS);
        let mut all_comp_edge =
            Vec::with_capacity(batch_size * MAX_COMPS * MAX_COMPS * PER_EDGE_COMP_FEATS);
        let mut all_comp_mask = Vec::with_capacity(batch_size * MAX_COMPS);
        let comp_graph_analysis_dim = self.comp_graph_analysis.feature_dim().max(1);
        let mut all_comp_graph_analysis = Vec::with_capacity(batch_size * comp_graph_analysis_dim);

        let mut all_pharm_ids = Vec::with_capacity(batch_size * MAX_PHARM);
        let mut all_pharm_scalars = Vec::with_capacity(batch_size * MAX_PHARM * PER_PHARM_SCALARS);
        let mut all_pharm_adj = Vec::with_capacity(batch_size * MAX_PHARM * MAX_PHARM);
        let mut all_pharm_edge =
            Vec::with_capacity(batch_size * MAX_PHARM * MAX_PHARM * PER_SPACIAL_EDGE_FEATS);
        let mut all_pharm_mask = Vec::with_capacity(batch_size * MAX_PHARM);
        let spacial_graph_analysis_dim = self.spacial_graph_analysis.feature_dim().max(1);
        let mut all_spacial_graph_analysis =
            Vec::with_capacity(batch_size * spacial_graph_analysis_dim);
        let atom_graph_analysis_dim = self.atom_graph_analysis.feature_dim().max(1);
        let mut all_atom_graph_analysis = Vec::with_capacity(batch_size * atom_graph_analysis_dim);

        let mut all_globals: Vec<f32> = Vec::new();
        let mut feat_dim = 0;

        for mol in mols {
            let mut feat_params = mlp_feats_from_mol(mol)?;
            if feat_dim == 0 {
                feat_dim = feat_params.len();
                all_globals.reserve(batch_size * feat_dim);
            }
            self.scaler.apply_in_place(&mut feat_params);
            all_globals.extend(feat_params);

            let g = GraphDataAtom::new(mol, ff_params, &self.atom_graph_analysis)?;
            let comps = mol
                .components
                .as_ref()
                .ok_or_else(|| io::Error::other("Missing components in ML inference"))?;
            let gc = GraphDataComponent::new(mol, comps, ff_params, &self.comp_graph_analysis)?;
            let gs = GraphDataSpacial::new(mol, &self.spacial_graph_analysis)?;

            // Atom graph
            all_elem.extend(pad_indices(&g.elem_indices, g.num_atoms, MAX_ATOMS));
            all_ff.extend(pad_indices(&g.ff_indices, g.num_atoms, MAX_ATOMS));
            all_scalars.extend(pad_scalars(
                &g.scalars,
                g.num_atoms,
                PER_ATOM_SCALARS,
                MAX_ATOMS,
            ));
            let (a, m) = pad_atom_adj_and_mask(&g.adj, g.num_atoms, MAX_ATOMS);
            all_adj.extend(a);
            all_mask.extend(m);
            all_edge.extend(pad_atom_edge_feats(&g.edge_feats, g.num_atoms, MAX_ATOMS));
            if g.analysis_features.is_empty() {
                all_atom_graph_analysis
                    .resize(all_atom_graph_analysis.len() + atom_graph_analysis_dim, 0.0);
            } else {
                debug_assert_eq!(g.analysis_features.len(), atom_graph_analysis_dim);
                all_atom_graph_analysis.extend_from_slice(&g.analysis_features);
            }

            // Component graph
            all_comp_ids.extend(pad_indices(&gc.comp_type_indices, gc.num_comps, MAX_COMPS));
            all_comp_scalars.extend(pad_scalars(
                &gc.scalars,
                gc.num_comps,
                PER_COMP_SCALARS,
                MAX_COMPS,
            ));
            let (ca, cm) = pad_adj_and_mask(&gc.adj, gc.num_comps, MAX_COMPS);
            all_comp_adj.extend(ca);
            all_comp_mask.extend(cm);
            all_comp_edge.extend(pad_edge_feats(
                &gc.edge_feats,
                gc.num_comps,
                PER_EDGE_COMP_FEATS,
                MAX_COMPS,
            ));
            if gc.analysis_features.is_empty() {
                all_comp_graph_analysis
                    .resize(all_comp_graph_analysis.len() + comp_graph_analysis_dim, 0.0);
            } else {
                debug_assert_eq!(gc.analysis_features.len(), comp_graph_analysis_dim);
                all_comp_graph_analysis.extend_from_slice(&gc.analysis_features);
            }

            // Spatial (pharmacophore) graph
            all_pharm_ids.extend(pad_indices(&gs.pharm_type_indices, gs.num_nodes, MAX_PHARM));
            all_pharm_scalars.extend(pad_scalars(
                &gs.scalars,
                gs.num_nodes,
                PER_PHARM_SCALARS,
                MAX_PHARM,
            ));
            let (pa, pm) = pad_adj_and_mask(&gs.adj, gs.num_nodes, MAX_PHARM);
            all_pharm_adj.extend(pa);
            all_pharm_mask.extend(pm);
            all_pharm_edge.extend(pad_edge_feats(
                &gs.edge_feats,
                gs.num_nodes,
                PER_SPACIAL_EDGE_FEATS,
                MAX_PHARM,
            ));
            if gs.analysis_features.is_empty() {
                all_spacial_graph_analysis.resize(
                    all_spacial_graph_analysis.len() + spacial_graph_analysis_dim,
                    0.0,
                );
            } else {
                debug_assert_eq!(gs.analysis_features.len(), spacial_graph_analysis_dim);
                all_spacial_graph_analysis.extend_from_slice(&gs.analysis_features);
            }
        }

        let dev = &self.device;

        let t_elem = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(all_elem, [batch_size, MAX_ATOMS]),
            dev,
        );
        let t_ff = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(all_ff, [batch_size, MAX_ATOMS]),
            dev,
        );
        let t_scalars = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_scalars, [batch_size, MAX_ATOMS, PER_ATOM_SCALARS]),
            dev,
        );
        let t_adj = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(
                all_adj,
                [batch_size, ATOM_GNN_EDGE_LAYERS, MAX_ATOMS, MAX_ATOMS],
            ),
            dev,
        );
        let t_edge = Tensor::<InferBackend, 5>::from_data(
            TensorData::new(
                all_edge,
                [
                    batch_size,
                    ATOM_GNN_EDGE_LAYERS,
                    MAX_ATOMS,
                    MAX_ATOMS,
                    ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
                ],
            ),
            dev,
        );
        let t_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_mask, [batch_size, MAX_ATOMS, 1]),
            dev,
        );
        let t_atom_graph_analysis = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(
                all_atom_graph_analysis,
                [batch_size, atom_graph_analysis_dim],
            ),
            dev,
        );

        let t_comp_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(all_comp_ids, [batch_size, MAX_COMPS]),
            dev,
        );
        let t_comp_scalars = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_comp_scalars, [batch_size, MAX_COMPS, PER_COMP_SCALARS]),
            dev,
        );
        let t_comp_adj = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_comp_adj, [batch_size, MAX_COMPS, MAX_COMPS]),
            dev,
        );
        let t_comp_edge = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(
                all_comp_edge,
                [batch_size, MAX_COMPS, MAX_COMPS, PER_EDGE_COMP_FEATS],
            ),
            dev,
        );
        let t_comp_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_comp_mask, [batch_size, MAX_COMPS, 1]),
            dev,
        );
        let t_comp_graph_analysis = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(
                all_comp_graph_analysis,
                [batch_size, comp_graph_analysis_dim],
            ),
            dev,
        );

        let t_pharm_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(all_pharm_ids, [batch_size, MAX_PHARM]),
            dev,
        );
        let t_pharm_scalars = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(
                all_pharm_scalars,
                [batch_size, MAX_PHARM, PER_PHARM_SCALARS],
            ),
            dev,
        );
        let t_pharm_adj = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_pharm_adj, [batch_size, MAX_PHARM, MAX_PHARM]),
            dev,
        );
        let t_pharm_edge = Tensor::<InferBackend, 4>::from_data(
            TensorData::new(
                all_pharm_edge,
                [batch_size, MAX_PHARM, MAX_PHARM, PER_SPACIAL_EDGE_FEATS],
            ),
            dev,
        );
        let t_pharm_mask = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(all_pharm_mask, [batch_size, MAX_PHARM, 1]),
            dev,
        );
        let t_spacial_graph_analysis = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(
                all_spacial_graph_analysis,
                [batch_size, spacial_graph_analysis_dim],
            ),
            dev,
        );

        let t_params = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(all_globals, [batch_size, feat_dim]),
            dev,
        );

        let preds = self.model.forward(
            t_elem,
            t_ff,
            t_scalars,
            t_adj,
            t_edge,
            t_mask,
            t_comp_ids,
            t_comp_scalars,
            t_comp_adj,
            t_comp_edge,
            t_comp_mask,
            t_comp_graph_analysis,
            t_pharm_ids,
            t_pharm_scalars,
            t_pharm_adj,
            t_pharm_edge,
            t_pharm_mask,
            t_spacial_graph_analysis,
            t_atom_graph_analysis,
            t_params,
        );

        let preds_vec = preds
            .into_data()
            .to_vec::<f32>()
            .map_err(|_| io::Error::other("Tensor error"))?;

        Ok(preds_vec
            .into_iter()
            .map(|p| self.scaler.denormalize_target(p))
            .collect())
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
