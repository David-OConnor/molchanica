use std::{fs, io, path::Path, time::Instant};

use bio_files::{AtomGeneric, BondGeneric};
use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::Backend},
};

use crate::{
    molecules::small::MoleculeSmall,
    pharmacokinetics::sol_train::{
        AQ_SOL_FEATURE_DIM, ATOM_FEATURE_DIM, AqSolModel, AqSolModelConfig, MAX_ATOMS,
        MODEL_CFG_FILE, MODEL_DIR, MODEL_FILE, SCALER_FILE, StandardScaler, features_from_molecule,
        mol_to_graph_data, pad_graph_data,
    },
};

type InferBackend = NdArray;

pub struct AqSolInfer {
    model: AqSolModel<InferBackend>,
    scaler: StandardScaler,
    device: <InferBackend as Backend>::Device,
}

impl AqSolInfer {
    pub fn load() -> io::Result<Self> {
        let model_dir = Path::new(MODEL_DIR);

        let cfg_bytes = fs::read(model_dir.join(MODEL_CFG_FILE))?;
        let scaler_bytes = fs::read(model_dir.join(SCALER_FILE))?;

        let config: AqSolModelConfig = serde_json::from_slice(&cfg_bytes)?;
        let scaler: StandardScaler = serde_json::from_slice(&scaler_bytes)?;

        let device = Default::default();
        let mut model = config.init::<InferBackend>(&device);

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model = model
            .load_file(model_dir.join(MODEL_FILE), &recorder, &device)
            .map_err(|e| io::Error::other(e))?;

        Ok(Self {
            model,
            scaler,
            device,
        })
    }

    pub fn infer(&self, mol: &MoleculeSmall) -> io::Result<f32> {
        println!("Starting solubility inference...");
        let start = Instant::now();

        // 1. Calculate Global Features
        let Some(char) = &mol.characterization else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Missing molecule characterization; can't infer solubility",
            ));
        };
        let mut global_raw = features_from_molecule(char)?;

        // println!("INFERENCE CALCULATED FEATURES: {:?}", global_raw); // <--- Add this

        self.scaler.apply_in_place(&mut global_raw);

        // 2. Calculate Graph Features

        let num_atoms = mol.common.atoms.len();
        if num_atoms == 0 {
            return Ok(0.0); // Or handle error
        }

        let (node_vec, adj_vec, _) = mol_to_graph_data(&mol);
        let (padded_nodes, padded_adj, padded_mask) =
            pad_graph_data(&node_vec, &adj_vec, num_atoms);

        // 4. Create Tensors
        // Note: Batch size is 1
        let t_globals = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(global_raw.to_vec(), [1, AQ_SOL_FEATURE_DIM]),
            &self.device,
        );
        let t_nodes = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(padded_nodes, [1, MAX_ATOMS, ATOM_FEATURE_DIM]),
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
        let y = self.model.forward(t_nodes, t_adj, t_mask, t_globals);

        // Extract result
        let val = y
            .into_data()
            .to_vec::<f32>()
            .map_err(|_| io::Error::other("Tensor error"))?[0];

        let elapsed = start.elapsed();
        println!("Inference complete in {:?}", elapsed);

        Ok(val)
    }
}

pub fn infer_solubility(mol: &MoleculeSmall) -> io::Result<f32> {
    AqSolInfer::load()?.infer(mol)
}
