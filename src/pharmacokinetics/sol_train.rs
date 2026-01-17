#![allow(unused)] // Required to prevent false positives.

//! To run: `cargo r --release --features train-sol --bin train_sol`

use std::{fs, io, path::Path, time::Instant};

use bio_files::{AtomGeneric, BondGeneric, Sdf};
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
    record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder},
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

// use molchanica::molecules;

// Todo: Problem importing from Molchanica, but we'd like to use these characteristics.
// use crate::{mol_characterization::MolCharacterization, molecules::small::MoleculeSmall};
// ==============================================================================================
// 1. CONSTANTS & CONFIGURATION
// ==============================================================================================

pub const AQ_SOL_FEATURE_DIM: usize = 14; // Update this A/R as you add or remove features.

pub const ATOM_FEATURE_DIM: usize = 10; // One-hot encoding dimension
pub const MAX_ATOMS: usize = 60; // Max atoms for padding

pub const MODEL_CFG_FILE: &str = "qsol_model_config.json";
// Extension handled automatically
// pub const MODEL_FILE: &str = "aqsol_model.mpk";
pub const MODEL_FILE: &str = "aqsol_model";
pub const SCALER_FILE: &str = "aqsol_scaler.json";

pub const MODEL_DIR: &str = "ml_models/aqsol";
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
        // let graph_embedding = graph_h.sum_dim(1).squeeze(1);
        let graph_embedding = graph_h.sum_dim(1).flatten(0, 1);

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
    /// From computed properties of the molecule.
    pub features_property: [f32; AQ_SOL_FEATURE_DIM],
    /// From the atom/bond graph
    pub features_node: Vec<f32>,
    pub adj_list: Vec<f32>,
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
            self.scaler.apply_in_place(&mut item.features_property);
            batch_globals.extend_from_slice(&item.features_property);
            batch_y.push(item.target);

            let (p_nodes, p_adj, p_mask) =
                pad_graph_data(&item.features_node, &item.adj_list, item.num_atoms);

            batch_nodes.extend(p_nodes);
            batch_adj.extend(p_adj);
            batch_mask.extend(p_mask);
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

/// Helper: Pads a single graph to MAX_ATOMS.
/// Returns (PaddedNodes, PaddedAdj, PaddedMask) as flat vectors.
pub fn pad_graph_data(
    raw_nodes: &[f32],
    raw_adj: &[f32],
    num_atoms: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(MAX_ATOMS);

    // 1. Nodes: Copy n, pad rest
    let mut p_nodes = Vec::with_capacity(MAX_ATOMS * ATOM_FEATURE_DIM);
    p_nodes.extend_from_slice(&raw_nodes[0..n * ATOM_FEATURE_DIM]);
    p_nodes.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * ATOM_FEATURE_DIM));

    // 2. Mask: 1.0 for atoms, 0.0 for pad
    let mut p_mask = Vec::with_capacity(MAX_ATOMS);
    p_mask.extend(std::iter::repeat(1.0).take(n));
    p_mask.extend(std::iter::repeat(0.0).take(MAX_ATOMS - n));

    // 3. Adj: Reconstruct row-by-row to handle 2D padding
    let mut p_adj = Vec::with_capacity(MAX_ATOMS * MAX_ATOMS);
    for r in 0..n {
        let row_start = r * num_atoms; // Input is flat [num_atoms * num_atoms]
        // Copy valid columns
        p_adj.extend_from_slice(&raw_adj[row_start..row_start + n]);
        // Pad columns (right side of matrix)
        p_adj.extend(std::iter::repeat(0.0).take(MAX_ATOMS - n));
    }
    // Pad rows (bottom of matrix)
    let remaining_rows = MAX_ATOMS - n;
    p_adj.extend(std::iter::repeat(0.0).take(remaining_rows * MAX_ATOMS));

    (p_nodes, p_adj, p_mask)
}

/// Helper: Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
pub fn mol_to_graph_data(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_atoms = atoms.len();

    // 1. Build Adjacency List (Same as before)
    let mut adj_list = vec![Vec::new(); num_atoms];
    for bond in bonds {
        let a1 = (bond.atom_0_sn as usize).saturating_sub(1);
        let a2 = (bond.atom_1_sn as usize).saturating_sub(1);
        if a1 < num_atoms && a2 < num_atoms {
            adj_list[a1].push(a2);
            adj_list[a2].push(a1);
        }
    }

    // 2. Build Node Features (Same as before)
    let mut node_feats = Vec::new();
    for atom in atoms {
        // ... (Keep your existing One-hot logic here) ...
        let mut f = vec![0.0; ATOM_FEATURE_DIM];
        let idx = match atom.element {
            /*...*/ _ => 9,
        }; // keep your match
        f[idx] = 1.0;
        node_feats.extend(f);
    }

    // --- 3. Build Adjacency Matrix (NORMALIZED) ---
    let mut adj = vec![0.0; num_atoms * num_atoms];

    for (i, neighbors) in adj_list.iter().enumerate() {
        // Degree = neighbors + self-loop
        let degree = (neighbors.len() as f32) + 1.0;

        // Simple Average Rule: weight = 1.0 / degree
        // (Alternative: Kipf & Welling use 1 / sqrt(deg_i * deg_j))
        let norm = 1.0 / degree;

        for &n in neighbors {
            adj[i * num_atoms + n] = norm;
        }
        // Self loop
        adj[i * num_atoms + i] = norm;
    }

    // 4. Mask (Same as before)
    let mask = vec![1.0; num_atoms];
    (node_feats, adj, mask)
}

fn fit_scaler(train: &[Sample]) -> StandardScaler {
    let n = train.len().max(1) as f32;
    let mut mean = vec![0.0; AQ_SOL_FEATURE_DIM];
    let mut var = vec![0.0; AQ_SOL_FEATURE_DIM];

    for s in train {
        for i in 0..AQ_SOL_FEATURE_DIM {
            mean[i] += s.features_property[i];
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    for s in train {
        for i in 0..AQ_SOL_FEATURE_DIM {
            let d = s.features_property[i] - mean[i];
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

/// This must match the fields in `sol_infer::features_from_molecule`.
/// This contains features from the CSV only; it doesn't have atom or bond data.
fn csv_to_features(row: &[String]) -> [f32; AQ_SOL_FEATURE_DIM] {
    [
        row[9].parse().unwrap_or(0.0),  // MolWt
        row[10].parse().unwrap_or(0.0), // MolLogP
        row[11].parse().unwrap_or(0.0), // MolMR
        row[12].parse().unwrap_or(0.0), // HeavyAtomCount
        row[13].parse().unwrap_or(0.0), // NumHAcceptors
        row[14].parse().unwrap_or(0.0), // NumHDonors
        row[15].parse().unwrap_or(0.0), // NumHeteroatoms
        row[16].parse().unwrap_or(0.0), // NumRotatableBonds
        row[17].parse().unwrap_or(0.0), // NumValenceElectrons
        row[18].parse().unwrap_or(0.0), // NumAromaticRings
        row[19].parse().unwrap_or(0.0), // NumSaturatedRings
        row[20].parse().unwrap_or(0.0), // NumAliphaticRings
        row[21].parse().unwrap_or(0.0), // RingCount
        row[22].parse().unwrap_or(0.0), // TPSA
                                        // ASA is not matching, although I'm not sure why.
                                        // row[23].parse().unwrap_or(0.0), // LabuteASA
                                        // Balaban J and BertCT: We are unable to accurate calculate them, so skip for ML.
                                        // row[24].parse().unwrap_or(0.0), // BalabanJ
                                        // row[25].parse().unwrap_or(0.0), // BertzCT
    ]
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;

    let bytes = line.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                if in_quotes {
                    if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                        cur.push('"');
                        i += 2;
                        continue;
                    } else {
                        in_quotes = false;
                        i += 1;
                        continue;
                    }
                } else {
                    in_quotes = true;
                    i += 1;
                    continue;
                }
            }
            b',' => {
                if !in_quotes {
                    out.push(cur);
                    cur = String::new();
                    i += 1;
                    continue;
                }
            }
            _ => {}
        }

        cur.push(bytes[i] as char);
        i += 1;
    }

    out.push(cur);
    out
}

fn read_data(csv_path: &Path, sdf_folder: &Path) -> io::Result<Vec<Sample>> {
    let mut samples = Vec::new();

    let csv_data = fs::read_to_string(csv_path)?;

    // Skip the header.
    for line in csv_data.lines().skip(1) {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }

        let cols: Vec<String> = split_csv_line(line);

        // for col in &cols {
        //     println!("-col: {col}");
        // }

        let filename = &cols[0];
        let features_property = csv_to_features(&cols);

        let solubility: f32 = cols[5].parse().unwrap();

        let sdf_path = sdf_folder.join(format!("{filename}.sdf"));

        // println!("Loading SDF at path {sdf_path:?}"); //  todo temp
        let sdf = match Sdf::load(&sdf_path) {
            Ok(s) => s,
            Err(e) => {
                println!("Error loading SDF at path {sdf_path:?}: {:?}", e);
                continue;
            }
        };

        // We are experimenting with using our internally-derived characteristics
        // instead of those in the CSV; it may be more consistent.
        // todo: Unable to import from molchanica.
        // let features_property = {
        //     let mol: MoleculeSmall = sdf.clone().try_into()?;
        //     let char = MolCharacterization::new(&mol.common);
        //     features_from_molecule(&char)?
        // };

        let atoms = sdf.atoms.clone();
        let bonds = sdf.bonds.clone();

        if bonds.is_empty() && atoms.len() > 20 {
            println!("\n\nNo bonds found in SDF at path (Likely you RMed them) {sdf_path:?}\n\n");
        }

        if bonds.is_empty() {
            eprintln!("No bonds found in SDF at path {sdf_path:?}. Skipping.");
            continue;
        }

        // ---------------------------------------------------

        let num_atoms = atoms.len();
        if num_atoms == 0 || num_atoms > MAX_ATOMS {
            continue;
        }

        let (features_node, adj_list, _) = mol_to_graph_data(&atoms, &bonds);

        samples.push(Sample {
            features_property,
            features_node,
            adj_list,
            num_atoms,
            target: solubility,
        });
    }

    Ok(samples)
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

    let valid_loader = DataLoaderBuilder::new(AqSolBatcher {
        scaler: scaler.clone(),
    })
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
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    result
        .model
        .save_file(model_dir.join(MODEL_FILE), &recorder)
        .unwrap();

    //  Save the Model Config
    let config_path = model_dir.join(MODEL_CFG_FILE);
    let config_file = fs::File::create(&config_path).expect("Could not create config file");
    serde_json::to_writer_pretty(config_file, &model_cfg).expect("Could not write config");
    println!("Saved config to {:?}", config_path);

    //  Save the Scaler
    // You need this for inference to know the means/stds to normalize new data.
    let scaler_path = model_dir.join(SCALER_FILE);
    let scaler_file = fs::File::create(&scaler_path).expect("Could not create scaler file");
    serde_json::to_writer_pretty(scaler_file, &scaler).expect("Could not write scaler");
    println!("Saved scaler to {:?}", scaler_path);

    let elapsed = start.elapsed().as_secs();
    println!(
        "Training complete in {:?} s. Saved model to {MODEL_FILE}",
        elapsed
    );
}
