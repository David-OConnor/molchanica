use std::{fs, io, path::Path};

use burn::{
    backend::NdArray,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, TensorData, activation, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::molecules::small::MoleculeSmall;

pub const AQ_SOL_FEATURE_DIM: usize = 17;

pub const MODEL_CFG_FILE: &str = "aqsol_model_config.json";
pub const MODEL_FILE: &str = "aqsol_model";
pub const SCALER_FILE: &str = "aqsol_scaler.json";

type InferBackend = NdArray;
type InferDevice = <InferBackend as Backend>::Device;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl StandardScaler {
    pub fn apply_in_place(&self, x: &mut [f32; AQ_SOL_FEATURE_DIM]) {
        for i in 0..AQ_SOL_FEATURE_DIM {
            let s = if self.std[i].abs() < 1.0e-12 {
                1.0
            } else {
                self.std[i]
            };
            x[i] = (x[i] - self.mean[i]) / s;
        }
    }
}

#[derive(Config, Debug)]
pub struct AqSolModelConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub hidden_dim2: usize,
}

#[derive(Module, Debug)]
pub struct AqSolModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl AqSolModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AqSolModel<B> {
        AqSolModel {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim2).init(device),
            fc3: LinearConfig::new(self.hidden_dim2, 1).init(device),
        }
    }
}

impl<B: Backend> AqSolModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x); // Burn 0.19: relu is a free function, not a method. :contentReference[oaicite:2]{index=2}
        let x = self.fc2.forward(x);
        let x = activation::relu(x);
        self.fc3.forward(x)
    }
}

pub struct AqSolInfer {
    model_cfg: AqSolModelConfig,
    scaler: StandardScaler,
    model: AqSolModel<InferBackend>,
    device: InferDevice,
}

impl AqSolInfer {
    pub fn load(model_dir: &Path) -> io::Result<Self> {
        let cfg_bytes =
            fs::read(model_dir.join(MODEL_CFG_FILE)).map_err(|e| io::Error::other(e))?;
        let scaler_bytes =
            fs::read(model_dir.join(SCALER_FILE)).map_err(|e| io::Error::other(e))?;

        let model_cfg: AqSolModelConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| io::Error::other(e))?;
        let scaler: StandardScaler =
            serde_json::from_slice(&scaler_bytes).map_err(|e| io::Error::other(e))?;

        let device = InferDevice::default();
        let mut model = model_cfg.init::<InferBackend>(&device);

        let recorder = CompactRecorder::new();
        model = model
            .load_file(model_dir.join(MODEL_FILE), &recorder, &device)
            .map_err(|e| io::Error::other(e))?;

        Ok(Self {
            model_cfg,
            scaler,
            model,
            device,
        })
    }

    pub fn infer(&self, mol: &MoleculeSmall) -> io::Result<f32> {
        let mut feats = features_from_molecule(mol);

        self.scaler.apply_in_place(&mut feats);

        let x_data = TensorData::new(feats.to_vec(), [1usize, self.model_cfg.input_dim]);
        let x = Tensor::<InferBackend, 2>::from_data(x_data, &self.device);

        let y = self.model.forward(x);
        let y_vec = y
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| io::Error::other(""))?;

        Ok(y_vec[0])
    }
}

pub fn infer(mol: &MoleculeSmall, model_dir: &Path) -> io::Result<f32> {
    AqSolInfer::load(model_dir)?.infer(mol)
}

fn features_from_molecule(mol: &MoleculeSmall) -> [f32; AQ_SOL_FEATURE_DIM] {
    let c = match mol.characterization.as_ref() {
        Some(c) => c,
        None => {
            return [0.0; AQ_SOL_FEATURE_DIM];
        }
    };

    let mol_weight = c.mol_weight;
    let mol_log_p = c.calc_log_p.unwrap_or(0.0);
    let mol_mr = c.m_r;

    let heavy_atom_count = c.num_heavy_atoms as f32;

    let num_h_acceptors = c.h_bond_acceptor.len() as f32;
    let num_h_donors = c.h_bond_donor.len() as f32;

    let num_het_atoms = c.num_hetero_atoms as f32;
    let num_rotatable_bonds = c.rotatable_bonds.len() as f32;

    let num_valence_elec = c.num_valence_elecs as f32;

    // If you don’t already classify rings by aromatic/saturated/aliphatic, these remain 0.
    // You can wire these to your ring classifier when available.
    let num_aromatic_rings = 0.0;
    let num_saturated_rings = 0.0;
    let num_aliphatic_rings = 0.0;

    let ring_count = c.rings.len() as f32;

    let tpsa = c.topological_polar_surface_area.unwrap_or(0.0);

    // If you later compute these, map them here.
    let labute_asa = 0.0;
    let balaban_j = c.balaban_j;
    let bertz_ct = c.bertz_ct;

    [
        mol_weight,
        mol_log_p,
        mol_mr,
        heavy_atom_count,
        num_h_acceptors,
        num_h_donors,
        num_het_atoms,
        num_rotatable_bonds,
        num_valence_elec,
        num_aromatic_rings,
        num_saturated_rings,
        num_aliphatic_rings,
        ring_count,
        tpsa,
        labute_asa,
        balaban_j,
        bertz_ct,
    ]
}
