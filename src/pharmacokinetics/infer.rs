//! ML inference, e.g. for Therapeutic properties. Shares the model and relevant
//! properties with `train.rs`.

use std::{fs, io, time::Instant};

use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::Backend},
};

use crate::pharmacokinetics::train::model_paths;
use crate::{
    molecules::small::MoleculeSmall,
    pharmacokinetics::train::{
        FEAT_DIM_ATOMS, MAX_ATOMS, Model, ModelConfig, StandardScaler, mol_to_graph_data,
        pad_graph_data, param_feats_from_mol,
    },
};

type InferBackend = NdArray;

pub struct Infer {
    model: Model<InferBackend>,
    scaler: StandardScaler,
    device: <InferBackend as Backend>::Device,
}

impl Infer {
    pub fn load(target_name: &str) -> io::Result<Self> {
        let (model_path, scaler_path, cfg_path) = model_paths(target_name);

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
        println!("Starting inference...");
        let start = Instant::now();

        // Feature parameters are from inferred properties of molecules, e.g.
        // data in our `MolCharacterization` struct.
        let n_feat_params = feat_params.len();
        self.scaler.apply_in_place(&mut feat_params);

        // Calculate graph features; these are related to atoms and bonds.
        let num_atoms = mol.common.atoms.len();
        if num_atoms == 0 {
            return Ok(0.0); // Or handle error
        }

        let (node_vec, adj_vec, _) = mol_to_graph_data(&mol);
        let (padded_nodes, padded_adj, padded_mask) =
            pad_graph_data(&node_vec, &adj_vec, num_atoms);

        // Create Tensors
        // Note: Batch size is 1
        let t_globals = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(feat_params, [1, n_feat_params]),
            &self.device,
        );
        let t_nodes = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_nodes, [1, MAX_ATOMS, FEAT_DIM_ATOMS]),
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

        let forward_tensor = self.model.forward(t_nodes, t_adj, t_mask, t_globals);

        let val = forward_tensor
            .into_data()
            .to_vec::<f32>()
            .map_err(|_| io::Error::other("Tensor error"))?[0];

        let elapsed = start.elapsed();
        println!("Inference complete in {:?}", elapsed);

        Ok(val)
    }
}

fn param_err() -> io::Error {
    io::Error::new(
        io::ErrorKind::Other,
        "Missing molecule characterization; can't infer.",
    )
}

/// Convenience function that may apply to many properties. Assumes a standard feature set.
pub fn infer_general(mol: &MoleculeSmall, target_name: &str) -> io::Result<f32> {
    let feat_params = param_feats_from_mol(mol)?;

    // todo: Store models to state! Don't load from disk each time.
    let infer = Infer::load(target_name)?;
    infer.infer(mol, feat_params)
}
