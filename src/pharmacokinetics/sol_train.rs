#![allow(unused)] // Required to prevent false positives.

//! To run: `cargo r --release --features train-sol --bin train_sol`

use std::{fs, io, path::Path, time::Instant};

use bio_files::{AtomGeneric, BondGeneric};
use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    lr_scheduler::constant::ConstantLr,
    module::Module,
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        Tensor, TensorData, activation,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        ClassificationOutput, InferenceStep, Learner, RegressionOutput, SupervisedTraining,
        TrainOutput, TrainStep, TrainingStrategy, metric::LossMetric,
    },
};
use na_seq::Element::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

// ==============================================================================================
// 1. CONSTANTS & CONFIGURATION
// ==============================================================================================

pub const AQ_SOL_FEATURE_DIM: usize = 17;
pub const ATOM_FEATURE_DIM: usize = 10; // One-hot encoding dimension
pub const MAX_ATOMS: usize = 60; // Max atoms for padding

pub const MODEL_CFG_FILE: &str = "qsol_model_config.json";
pub const MODEL_FILE: &str = "aqsol_model.model";
pub const SCALER_FILE: &str = "aqsol_scaler.json";

const MODEL_DIR: &str = ".ml_models/aqsol";
const AQ_SOL_DB_CSV_PATH: &str = "C:/Users/the_a/Desktop/bio_misc/AqSolDB/results/data_curated.csv";
const AQ_SOL_DB_SDF_PATH: &str = "C:/Users/the_a/Desktop/bio_misc/AqSolDb_mols";

type TrainBackend = Autodiff<NdArray>;
// type TrainBackend = Wgpu<f32, i32>; // todo?
type ValidBackend = NdArray;

#[derive(Config, Debug)]
pub struct AqSolModelConfig {
    pub global_input_dim: usize,
    pub atom_input_dim: usize,
    pub gnn_hidden_dim: usize,
    pub mlp_hidden_dim: usize,
}

impl AqSolModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AqSolModel<B> {
        AqSolModel {
            gnn_proj: LinearConfig::new(self.atom_input_dim, self.gnn_hidden_dim).init(device),
            global_fc: LinearConfig::new(self.global_input_dim, self.mlp_hidden_dim).init(device),
            head: LinearConfig::new(self.gnn_hidden_dim + self.mlp_hidden_dim, 1).init(device),
        }
    }
}

// ==============================================================================================
// 2. MODEL DEFINITION (HYBRID GNN + MLP)
// ==============================================================================================

#[derive(Module, Debug)]
pub struct AqSolModel<B: Backend> {
    /// Graph Branch: Simple projection (GCN-like)
    gnn_proj: Linear<B>,
    /// Global Branch: MLP for CSV features
    global_fc: Linear<B>,
    /// Joint Branch: Combine (Graph Sum + Global) -> Output
    head: Linear<B>,
}

impl<B: Backend> AqSolModel<B> {
    pub fn forward(
        &self,
        nodes: Tensor<B, 3>,   // [Batch, MaxAtoms, AtomDim]
        adj: Tensor<B, 3>,     // [Batch, MaxAtoms, MaxAtoms]
        mask: Tensor<B, 3>,    // [Batch, MaxAtoms, 1]
        globals: Tensor<B, 2>, // [Batch, GlobalDim]
    ) -> Tensor<B, 2> {
        // --- Graph Branch ---
        // 1. Aggregation: A * X (Sum neighbors)
        let agg = adj.matmul(nodes);
        // 2. Projection: (A*X)W
        let graph_h = activation::relu(self.gnn_proj.forward(agg));
        // 3. Masking: Zero out padding atoms so they don't contribute to sum
        let graph_h = graph_h * mask;
        // 4. Pooling: Sum over atoms -> [Batch, GnnHidden]
        let graph_embedding = graph_h.sum_dim(1).squeeze();

        // --- Global Branch ---
        let global_embedding = activation::relu(self.global_fc.forward(globals));

        // --- Fusion ---
        let combined = Tensor::cat(vec![graph_embedding, global_embedding], 1);

        // --- Readout ---
        self.head.forward(combined)
    }
}

// ==============================================================================================
// 3. DATA STRUCTURES & BATCHER
// ==============================================================================================

#[derive(Clone, Debug)]
pub struct Sample {
    pub global_feats: [f32; AQ_SOL_FEATURE_DIM],
    pub node_feats: Vec<f32>,
    pub adj: Vec<f32>,
    pub num_atoms: usize,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub struct AqSolBatch<B: Backend> {
    pub nodes: Tensor<B, 3>,
    pub adj: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
    pub globals: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct AqSolBatcher {
    pub scaler: StandardScaler,
}

impl<B: Backend> Batcher<B, Sample, AqSolBatch<B>> for AqSolBatcher {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> AqSolBatch<B> {
        let batch_size = items.len();

        let mut batch_nodes = Vec::new();
        let mut batch_adj = Vec::new();
        let mut batch_mask = Vec::new();
        let mut batch_globals = Vec::new();
        let mut batch_y = Vec::new();

        for mut item in items {
            // Apply Scaler
            self.scaler.apply_in_place(&mut item.global_feats);
            batch_globals.extend_from_slice(&item.global_feats);
            batch_y.push(item.target);

            // Graph Padding
            let n = item.num_atoms.min(MAX_ATOMS);

            // Nodes (Pad with 0)
            batch_nodes.extend(item.node_feats.iter().take(n * ATOM_FEATURE_DIM));
            batch_nodes.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * ATOM_FEATURE_DIM));

            // Mask (1 for atom, 0 for pad)
            batch_mask.extend(std::iter::repeat(1.0).take(n));
            batch_mask.extend(std::iter::repeat(0.0).take(MAX_ATOMS - n));

            // Adj (Reconstruct from flat to padded row-by-row)
            for r in 0..n {
                let row_start = r * item.num_atoms;
                batch_adj.extend(item.adj.iter().skip(row_start).take(n));
                batch_adj.extend(std::iter::repeat(0.0).take(MAX_ATOMS - n));
            }
            // Pad remaining rows
            batch_adj.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * MAX_ATOMS));
        }

        let nodes = TensorData::new(batch_nodes, [batch_size, MAX_ATOMS, ATOM_FEATURE_DIM]);
        let adj = TensorData::new(batch_adj, [batch_size, MAX_ATOMS, MAX_ATOMS]);
        let mask = TensorData::new(batch_mask, [batch_size, MAX_ATOMS, 1]);
        let globals = TensorData::new(batch_globals, [batch_size, AQ_SOL_FEATURE_DIM]);
        let y = TensorData::new(batch_y, [batch_size, 1]);

        AqSolBatch {
            nodes: Tensor::from_data(nodes, device),
            adj: Tensor::from_data(adj, device),
            mask: Tensor::from_data(mask, device),
            globals: Tensor::from_data(globals, device),
            targets: Tensor::from_data(y, device),
        }
    }
}

// ==============================================================================================
// 4. UTILITIES (Scaler & Graph Conversion)
// ==============================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl StandardScaler {
    pub fn apply_in_place(&self, x: &mut [f32; AQ_SOL_FEATURE_DIM]) {
        for i in 0..AQ_SOL_FEATURE_DIM {
            let s = if self.std[i].abs() < 1e-9 {
                1.0
            } else {
                self.std[i]
            };
            x[i] = (x[i] - self.mean[i]) / s;
        }
    }
}

/// Helper: Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
pub fn mol_to_graph_data(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_atoms = atoms.len();

    // 1. Build Adjacency List
    let mut adj_list = vec![Vec::new(); num_atoms];
    for bond in bonds {
        // SDF is usually 1-based, AtomGeneric might carry that.
        // Assuming AtomGeneric serial numbers are 1-based:
        let a1 = (bond.atom_0_sn as usize).saturating_sub(1);
        let a2 = (bond.atom_1_sn as usize).saturating_sub(1);
        if a1 < num_atoms && a2 < num_atoms {
            adj_list[a1].push(a2);
            adj_list[a2].push(a1);
        }
    }

    // 2. Build Node Features
    let mut node_feats = Vec::new();
    for atom in atoms {
        let mut f = vec![0.0; ATOM_FEATURE_DIM];
        // One-hot mapping
        let idx = match atom.element {
            Carbon => 0,
            Nitrogen => 1,
            Oxygen => 2,
            Sulfur => 3,
            Fluorine => 4,
            Chlorine => 5,
            Bromine => 6,
            Iodine => 7,
            Hydrogen => 8,
            _ => 9,
        };
        f[idx] = 1.0;
        node_feats.extend(f);
    }

    // 3. Build Adjacency Matrix (Flat)
    let mut adj = vec![0.0; num_atoms * num_atoms];
    for (i, neighbors) in adj_list.iter().enumerate() {
        for &n in neighbors {
            adj[i * num_atoms + n] = 1.0;
        }
        // Self loop
        adj[i * num_atoms + i] = 1.0;
    }

    // 4. Mask (All 1.0 for real atoms)
    let mask = vec![1.0; num_atoms];
    (node_feats, adj, mask)
}

fn fit_scaler(train: &[Sample]) -> StandardScaler {
    let n = train.len().max(1) as f32;
    let mut mean = vec![0.0; AQ_SOL_FEATURE_DIM];
    let mut var = vec![0.0; AQ_SOL_FEATURE_DIM];

    for s in train {
        for i in 0..AQ_SOL_FEATURE_DIM {
            mean[i] += s.global_feats[i];
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    for s in train {
        for i in 0..AQ_SOL_FEATURE_DIM {
            let d = s.global_feats[i] - mean[i];
            var[i] += d * d;
        }
    }
    let std = var.iter().map(|v| (v / n).sqrt()).collect();
    StandardScaler { mean, std }
}

// ==============================================================================================
// 5. TRAINING IMPLEMENTATION
// ==============================================================================================

impl TrainStep for AqSolModel<TrainBackend> {
    type Input = AqSolBatch<TrainBackend>;
    type Output = RegressionOutput<TrainBackend>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pred = self.forward(batch.nodes, batch.adj, batch.mask, batch.globals);
        let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);

        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            RegressionOutput::new(loss, pred, batch.targets),
        )
    }
}

impl InferenceStep for AqSolModel<ValidBackend> {
    type Input = AqSolBatch<ValidBackend>;
    type Output = RegressionOutput<ValidBackend>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        // This is exactly what your ValidStep does
        let pred = self.forward(batch.nodes, batch.adj, batch.mask, batch.globals);
        let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);

        RegressionOutput::new(loss, pred, batch.targets)
    }
}

// impl<B: Backend> ValidStep<AqSolBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: AqSolBatch<B>) -> ClassificationOutput<B> {
//         self.forward_classification(batch.adj, batch.targets)
//     }
// }

// impl TrainStep<AqSolBatch<TrainBackend>, RegressionOutput<TrainBackend>>
//     for AqSolModel<TrainBackend>
// {
//     fn step(&self, batch: AqSolBatch<TrainBackend>) -> TrainOutput<RegressionOutput<TrainBackend>> {
//         // The logic remains largely the same, but ensure your Backend
//         // matches the specific requirements of the generalized TrainStep.
//         let pred = self.forward(batch.nodes, batch.adj, batch.mask, batch.globals);
//         let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);
//         let grads = loss.backward();
//
//         TrainOutput::new(
//             self,
//             grads,
//             RegressionOutput::new(loss, pred, batch.targets),
//         )
//     }
// }
//
// impl ValidStep<AqSolBatch<ValidBackend>, RegressionOutput<ValidBackend>>
//     for AqSolModel<ValidBackend>
// {
//     fn step(&self, batch: AqSolBatch<ValidBackend>) -> RegressionOutput<ValidBackend> {
//         let pred = self.forward(batch.nodes, batch.adj, batch.mask, batch.globals);
//         let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);
//         RegressionOutput::new(loss, pred, batch.targets)
//     }
// }

// --- CSV Deserialization Helpers ---

/// This corresponds to `AqSolDb`'s `data_curated`.csv.
#[derive(Clone, Debug, Deserialize)]
struct CsvRow {
    #[serde(rename = "ID")]
    id: String,
    #[serde(rename = "Solubility")]
    solubility: f32,
    #[serde(rename = "MolWt")]
    mol_weight: f32,
    #[serde(rename = "MolLogP")]
    mol_log_p: f32,
    #[serde(rename = "MolMR")]
    mol_mr: f32,
    #[serde(rename = "HeavyAtomCount")]
    heavy_atom_count: u16,
    #[serde(rename = "NumHAcceptors")]
    num_h_acceptors: u8,
    #[serde(rename = "NumHDonors")]
    num_h_donors: u8,
    #[serde(rename = "NumHeteroatoms")]
    num_het_atoms: u8,
    #[serde(rename = "NumRotatableBonds")]
    num_rotatable_bonds: u8,
    #[serde(rename = "NumValenceElectrons")]
    num_valence_elec: u16,
    #[serde(rename = "NumAromaticRings")]
    num_aromatic_rings: u8,
    #[serde(rename = "NumSaturatedRings")]
    num_saturated_rings: u8,
    #[serde(rename = "NumAliphaticRings")]
    num_aliphatic_rings: u8,
    #[serde(rename = "RingCount")]
    ring_count: u8,
    #[serde(rename = "TPSA")]
    tpsa: f32,
    #[serde(rename = "LabuteASA")]
    labute_asa: f32,
    #[serde(rename = "BalabanJ")]
    balaban_j: f32,
    #[serde(rename = "BertzCT")]
    bertz_ct: f32,
}

fn features_from_csv_row(r: &CsvRow) -> [f32; AQ_SOL_FEATURE_DIM] {
    [
        r.mol_weight,
        r.mol_log_p,
        r.mol_mr,
        r.heavy_atom_count as f32,
        r.num_h_acceptors as f32,
        r.num_h_donors as f32,
        r.num_het_atoms as f32,
        r.num_rotatable_bonds as f32,
        r.num_valence_elec as f32,
        r.num_aromatic_rings as f32,
        r.num_saturated_rings as f32,
        r.num_aliphatic_rings as f32,
        r.ring_count as f32,
        r.tpsa,
        r.labute_asa,
        r.balaban_j,
        r.bertz_ct,
    ]
}

// --- Data Loader Logic ---

fn read_data(csv_path: &Path, sdf_folder: &Path) -> io::Result<Vec<Sample>> {
    let mut samples = Vec::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    for result in rdr.deserialize::<CsvRow>() {
        let row = result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // 1. Load SDF
        let sdf_path = sdf_folder.join(format!("{}.sdf", row.id));

        // --- TODO: REPLACE THIS WITH YOUR REAL SDF PARSER ---
        // e.g., let mol = crate::io::read_sdf(&sdf_path)?;
        // let atoms = mol.atoms; let bonds = mol.bonds;
        let (atoms, bonds) = mock_load_sdf_atoms(&sdf_path);
        // ---------------------------------------------------

        let num_atoms = atoms.len();
        if num_atoms == 0 || num_atoms > MAX_ATOMS {
            continue;
        }

        let (n_feats, adj, _) = mol_to_graph_data(&atoms, &bonds);

        samples.push(Sample {
            global_feats: features_from_csv_row(&row),
            node_feats: n_feats,
            adj: adj,
            num_atoms: num_atoms,
            target: row.solubility,
        });
    }

    Ok(samples)
}

// Temporary Mock to allow compilation.
fn mock_load_sdf_atoms(_p: &Path) -> (Vec<AtomGeneric>, Vec<BondGeneric>) {
    // Just returning a dummy carbon so compilation succeeds.
    // Logic: In real code, parse the SDF file here.
    let mut a = AtomGeneric::default();
    a.element = Carbon;
    (vec![a], vec![])
}

// ==============================================================================================
// 6. PUBLIC ENTRY POINT
// ==============================================================================================

pub fn main() {
    let start = Instant::now();
    println!("Started training");

    let model_dir = Path::new(MODEL_DIR);
    fs::create_dir_all(model_dir).unwrap();

    // Data loading
    let mut data_csv_sdf =
        read_data(Path::new(AQ_SOL_DB_CSV_PATH), Path::new(AQ_SOL_DB_SDF_PATH)).unwrap();

    println!("Sample len: {}", data_csv_sdf.len());

    let mut rng = StdRng::seed_from_u64(42);
    data_csv_sdf.shuffle(&mut rng);

    let split = ((data_csv_sdf.len() as f32) * 0.9).round() as usize;
    let (train_raw, valid_raw) = data_csv_sdf.split_at(split);

    let scaler = fit_scaler(train_raw);
    // let device = <TrainBackend as Backend>::Device::default();
    let device = Default::default();

    // 2. Data Loaders
    let train_loader = DataLoaderBuilder::new(AqSolBatcher {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .shuffle(42)
    .build(InMemDataset::new(train_raw.to_vec()));

    let valid_loader = DataLoaderBuilder::new(AqSolBatcher { scaler })
        .batch_size(128)
        .build(InMemDataset::new(valid_raw.to_vec()));

    let model_cfg = AqSolModelConfig {
        global_input_dim: AQ_SOL_FEATURE_DIM,
        atom_input_dim: ATOM_FEATURE_DIM,
        gnn_hidden_dim: 64,
        mlp_hidden_dim: 128,
    };

    let model = model_cfg.init::<TrainBackend>(&device);
    let optim = AdamConfig::new().init();
    let lr_scheduler = ConstantLr::new(3e-4);

    // 4. The New 0.20 SupervisedTraining Flow
    let training = SupervisedTraining::new(
        model_dir.to_str().unwrap(), // Artifact directory
        train_loader,
        valid_loader,
    )
    .metrics((LossMetric::new(),)) // Note the tuple format for metrics
    .num_epochs(80)
    .with_training_strategy(TrainingStrategy::SingleDevice(device.clone()))
    .summary(); // Provides the TUI/CLI output

    // 5. Launch the Learner
    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    // 6. Save using the result
    let recorder = CompactRecorder::new();
    result
        .model
        .save_file(model_dir.join(MODEL_FILE), &recorder)
        .unwrap();

    let elapsed = start.elapsed();
    println!(
        "Training complete in {:?}. Saved model to {MODEL_FILE}",
        elapsed
    );
}
