//! ML inference, e.g. for Therapeutic properties. Shares the model and relevant
//! properties with `train.rs`.

use std::{collections::HashMap, fs, io, time::Instant};

use crate::therapeutic::train::pad_adj_and_mask;
use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc,
        train::{
            MAX_ATOMS, Model, ModelConfig, StandardScaler, model_paths, mol_to_graph_data,
            param_feats_from_mol,
        },
    },
};
use burn::backend::NdArray;
use burn::prelude::Int;
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_cpu::Cpu;

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

    pub fn infer(&self, mol: &MoleculeSmall, mut feat_params: Vec<f32>) -> io::Result<f32> {
        let start = Instant::now();

        // 1. Prepare Globals
        let n_feat_params = feat_params.len();
        self.scaler.apply_in_place(&mut feat_params);

        // 2. Extract Graph Data (New Return Signature)
        let (elem_ids, ff_ids, scalars, adj_vec, num_atoms) = mol_to_graph_data(&mol)?;

        // 3. Pad Data (Replicating Batcher Logic for BatchSize=1)
        let n = num_atoms.min(MAX_ATOMS);

        // -- Pad Indices (Int) --
        let mut p_elem_ids = Vec::with_capacity(MAX_ATOMS);
        p_elem_ids.extend_from_slice(&elem_ids[0..n]);
        p_elem_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

        let mut p_ff_ids = Vec::with_capacity(MAX_ATOMS);
        p_ff_ids.extend_from_slice(&ff_ids[0..n]);
        p_ff_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

        // -- Pad Scalars (Float) --
        // Calculate dimensionality (should be 2: charge + degree)
        let n_scalars_per_atom = if num_atoms > 0 {
            scalars.len() / num_atoms
        } else {
            2
        };
        let mut p_scalars = Vec::with_capacity(MAX_ATOMS * n_scalars_per_atom);
        p_scalars.extend_from_slice(&scalars[0..n * n_scalars_per_atom]);
        p_scalars.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * n_scalars_per_atom));

        // -- Pad Adj & Mask (Float) --
        // Use the helper from train.rs
        let (padded_adj, padded_mask) = pad_adj_and_mask(&adj_vec, num_atoms);

        // 4. Create Tensors
        let t_param_feats = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(feat_params, [1, n_feat_params]),
            &self.device,
        );

        // Int Tensors for Embeddings
        let t_elem_ids = Tensor::<InferBackend, 2, Int>::from_data(
            TensorData::new(p_elem_ids, [1, MAX_ATOMS]),
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

    infer.infer(mol, feat_params)
}
