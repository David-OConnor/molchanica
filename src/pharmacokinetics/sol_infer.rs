use std::{fs, io, path::Path};

use bio_files::{AtomGeneric, BondGeneric};
use burn::{
    backend::NdArray,
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, TensorData, backend::Backend},
};

use crate::molecules::small::MoleculeSmall;
// Import everything needed from sol_train
use crate::pharmacokinetics::sol_train::{
    AQ_SOL_FEATURE_DIM, ATOM_FEATURE_DIM, AqSolModel, AqSolModelConfig, MAX_ATOMS, MODEL_CFG_FILE,
    MODEL_FILE, SCALER_FILE, StandardScaler, mol_to_graph_data,
};

type InferBackend = NdArray;

pub struct AqSolInfer {
    model: AqSolModel<InferBackend>,
    scaler: StandardScaler,
    device: <InferBackend as Backend>::Device,
}

impl AqSolInfer {
    pub fn load(model_dir: &Path) -> io::Result<Self> {
        let cfg_bytes = fs::read(model_dir.join(MODEL_CFG_FILE))?;
        let scaler_bytes = fs::read(model_dir.join(SCALER_FILE))?;

        let config: AqSolModelConfig = serde_json::from_slice(&cfg_bytes)?;
        let scaler: StandardScaler = serde_json::from_slice(&scaler_bytes)?;

        let device = Default::default();
        let mut model = config.init::<InferBackend>(&device);

        let recorder = CompactRecorder::new();
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
        // 1. Calculate Global Features
        let mut global_raw = features_from_molecule(mol);
        self.scaler.apply_in_place(&mut global_raw);

        // 2. Calculate Graph Features
        // Convert MoleculeSmall atoms to AtomGeneric for the shared function
        let atoms_gen: Vec<AtomGeneric> = mol.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<BondGeneric> = mol.common.bonds.iter().map(|a| a.to_generic()).collect();

        let num_atoms = atoms_gen.len();
        if num_atoms == 0 {
            return Ok(0.0); // Or handle error
        }

        let (node_vec, adj_vec, _) = mol_to_graph_data(&atoms_gen, &bonds_gen);

        // 3. Manual Padding (Match the Batcher logic)
        let mut padded_nodes = node_vec;
        padded_nodes
            .extend(std::iter::repeat(0.0).take((MAX_ATOMS - num_atoms) * ATOM_FEATURE_DIM));

        let mut padded_adj = Vec::with_capacity(MAX_ATOMS * MAX_ATOMS);
        for r in 0..num_atoms {
            let start = r * num_atoms;
            // Copy row
            padded_adj.extend_from_slice(&adj_vec[start..start + num_atoms]);
            // Pad row
            padded_adj.extend(std::iter::repeat(0.0).take(MAX_ATOMS - num_atoms));
        }
        // Pad remaining rows
        padded_adj.extend(std::iter::repeat(0.0).take((MAX_ATOMS - num_atoms) * MAX_ATOMS));

        let mut padded_mask = vec![1.0; num_atoms];
        padded_mask.extend(std::iter::repeat(0.0).take(MAX_ATOMS - num_atoms));

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

        Ok(val)
    }
}

pub fn infer(mol: &MoleculeSmall, model_dir: &Path) -> io::Result<f32> {
    AqSolInfer::load(model_dir)?.infer(mol)
}

// Logic to extract the 17 global features from your MoleculeSmall struct
fn features_from_molecule(mol: &MoleculeSmall) -> [f32; AQ_SOL_FEATURE_DIM] {
    let c = match mol.characterization.as_ref() {
        Some(c) => c,
        None => return [0.0; AQ_SOL_FEATURE_DIM],
    };

    [
        c.mol_weight,
        c.calc_log_p,
        c.molar_refractivity,
        c.num_heavy_atoms as f32,
        c.h_bond_acceptor.len() as f32,
        c.h_bond_donor.len() as f32,
        c.num_hetero_atoms as f32,
        c.rotatable_bonds.len() as f32,
        c.num_valence_elecs as f32,
        // Placeholders if you don't calculate these yet:
        0.0, // NumAromaticRings
        0.0, // NumSaturatedRings
        0.0, // NumAliphaticRings
        c.rings.len() as f32,
        c.tpsa_ertl,
        c.asa_labute,
        c.balaban_j,
        c.bertz_ct,
    ]
}
