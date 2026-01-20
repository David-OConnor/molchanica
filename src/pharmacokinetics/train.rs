#![allow(unused)] // Required to prevent false positives.

//! Entry point for training of therapeutic properties. (Via a thin wrapper in `/src/train.rs` required
//! by Rust's system)
//!
//! This is tailored towards data from Therapeutic Data Commons (TDC).

//! To run: `cargo r --release --features train --bin train --`
//!
//! Add these CLI params (Separated for clarity with the long paths.
//! --csv C:/Users/the_a/Desktop/bio_misc/tdc_data/bbb_martins.csv`
//! --sdf C:/Users/the_a/Desktop/bio_misc/tdc_data/bbb_martins

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
use std::path::PathBuf;
use std::{env, fs, io, path::Path, time::Instant};

use crate::{
    mol_characterization::MolCharacterization,
    molecules::{Atom, Bond, small::MoleculeSmall},
};

// todo: What should this be?
pub const FEAT_DIM_ATOMS: usize = 10; // One-hot encoding dimension

// todo: How should this be set up
pub const MAX_ATOMS: usize = 100; // Max atoms for padding

pub const MODEL_DIR: &str = "ml_models";

const TGT_COL_TDC: usize = 2;

type TrainBackend = Autodiff<NdArray>;
// type TrainBackend = Wgpu<f32, i32>; // todo?
type ValidBackend = NdArray;

/// Given a target (e.g. pharamaceutical property) name, get standardized filenames
/// for the (model, scalar, config).
pub(in crate::pharmacokinetics) fn model_paths(target_name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let model_dir = Path::new(MODEL_DIR);

    // Extension is implicit in the model, for Burn.
    let model = model_dir.join(format!("{target_name}_model"));
    let scaler = model_dir.join(format!("{target_name}_scaler.json"));
    let cfg = model_dir.join(format!("{target_name}_model_config.json"));

    (model, scaler, cfg)
}

#[derive(Config, Debug)]
pub(in crate::pharmacokinetics) struct ModelConfig {
    pub global_input_dim: usize,
    pub atom_input_dim: usize,
    pub gnn_hidden_dim: usize,
    pub mlp_hidden_dim: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            gnn_proj: LinearConfig::new(self.atom_input_dim, self.gnn_hidden_dim).init(device),
            global_fc: LinearConfig::new(self.global_input_dim, self.mlp_hidden_dim).init(device),
            // todo: For now with no graph
            // head: LinearConfig::new(self.gnn_hidden_dim + self.mlp_hidden_dim, 1).init(device),
            head: LinearConfig::new(self.mlp_hidden_dim, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(in crate::pharmacokinetics) struct Model<B: Backend> {
    /// Graph Branch: Simple projection (GCN-like)
    gnn_proj: Linear<B>,
    /// Global Branch: MLP for CSV features
    global_fc: Linear<B>,
    /// Joint Branch: Combine (Graph Sum + Global) -> Output
    head: Linear<B>,
}

impl<B: Backend> Model<B> {
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
        // let graph_embedding = graph_h.sum_dim(1).flatten(0, 1);

        // --- Global Branch ---
        let global_embedding = activation::relu(self.global_fc.forward(globals));

        // --- Fusion ---
        // let combined = Tensor::cat(vec![graph_embedding, global_embedding], 1);
        // todo temp removed the graph.
        let combined = Tensor::cat(vec![global_embedding], 1);

        // --- Readout ---
        self.head.forward(combined)
    }
}

#[derive(Clone, Debug)]
pub(in crate::pharmacokinetics) struct Sample {
    /// From computed properties of the molecule.
    // pub features_property: [f32; FEAT_DIM_PARAM],
    pub features_property: Vec<f32>,
    /// From the atom/bond graph
    pub features_node: Vec<f32>,
    pub adj_list: Vec<f32>,
    pub num_atoms: usize,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub(in crate::pharmacokinetics) struct Batch<B: Backend> {
    pub nodes: Tensor<B, 3>,
    pub adj: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
    pub globals: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone)]
pub(in crate::pharmacokinetics) struct Batcher_ {
    pub scaler: StandardScaler,
}

impl<B: Backend> Batcher<B, Sample, Batch<B>> for Batcher_ {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> Batch<B> {
        let batch_size = items.len();

        let mut batch_nodes = Vec::new();
        let mut batch_adj = Vec::new();
        let mut batch_mask = Vec::new();
        let mut batch_globals = Vec::new();
        let mut batch_y = Vec::new();

        let n_feat_params = items[0].features_property.len();

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

        let nodes = TensorData::new(batch_nodes, [batch_size, MAX_ATOMS, FEAT_DIM_ATOMS]);
        let adj = TensorData::new(batch_adj, [batch_size, MAX_ATOMS, MAX_ATOMS]);
        let mask = TensorData::new(batch_mask, [batch_size, MAX_ATOMS, 1]);
        let globals = TensorData::new(batch_globals, [batch_size, n_feat_params]);
        let y = TensorData::new(batch_y, [batch_size, 1]);

        Batch {
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
    // pub fn apply_in_place(&self, x: &mut [f32; FEAT_DIM_PARAM]) {
    pub fn apply_in_place(&self, x: &mut Vec<f32>) {
        for i in 0..x.len() {
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
    let mut p_nodes = Vec::with_capacity(MAX_ATOMS * FEAT_DIM_ATOMS);
    p_nodes.extend_from_slice(&raw_nodes[0..n * FEAT_DIM_ATOMS]);
    p_nodes.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * FEAT_DIM_ATOMS));

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
pub fn mol_to_graph_data(mol: &MoleculeSmall) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_atoms = mol.common.atoms.len();

    // 1. Node Features (Unchanged)
    let mut node_feats = Vec::new();
    for atom in &mol.common.atoms {
        let mut f = vec![0.0; FEAT_DIM_ATOMS];
        // Ensure this match is IDENTICAL to your training expectations
        let idx = match atom.element {
            Hydrogen => 0,
            Carbon => 1,
            Nitrogen => 2,
            Oxygen => 3,
            Fluorine => 4,
            Phosphorus => 5,
            Sulfur => 6,
            Chlorine => 7,
            Bromine => 8,
            _ => 9,
        };
        f[idx] = 1.0;
        node_feats.extend(f);
    }

    // 2. Build Raw Adjacency (Force Symmetry)
    // We use a flat vector to represent the matrix temporarily
    let mut raw_adj = vec![0.0; num_atoms * num_atoms];
    // Also track degree for normalization
    let mut degrees = vec![0.0; num_atoms];

    for (i, neighbors) in mol.common.adjacency_list.iter().enumerate() {
        for &n in neighbors {
            // Force Symmetry: If i connects to n, set both [i,n] and [n,i]
            if i < num_atoms && n < num_atoms {
                raw_adj[i * num_atoms + n] = 1.0;
                raw_adj[n * num_atoms + i] = 1.0;
            }
        }
        // Self-loops are added in the normalization step typically,
        // but let's add them explicitly to the raw matrix first for clarity
        raw_adj[i * num_atoms + i] = 1.0;
    }

    // 3. Recalculate Degrees (inclusive of self-loops)
    for i in 0..num_atoms {
        let mut d = 0.0;
        for j in 0..num_atoms {
            if raw_adj[i * num_atoms + j] > 0.0 {
                d += 1.0;
            }
        }
        degrees[i] = d;
    }

    // 4. Apply Symmetric Normalization: D^-0.5 * A * D^-0.5
    // Entry[i][j] = Entry[i][j] / sqrt(deg[i] * deg[j])
    let mut final_adj = vec![0.0; num_atoms * num_atoms];
    for i in 0..num_atoms {
        for j in 0..num_atoms {
            if raw_adj[i * num_atoms + j] > 0.0 {
                let d_i = degrees[i];
                let d_j = degrees[j];

                let v: f32 = (d_i * d_j);
                let norm_factor = v.sqrt();
                final_adj[i * num_atoms + j] = 1.0 / norm_factor;
            }
        }
    }

    let mask = vec![1.0; num_atoms];
    (node_feats, final_adj, mask)
}

fn fit_scaler(train: &[Sample]) -> StandardScaler {
    let n = train.len().max(1) as f32;

    let num_params = train[0].features_property.len();

    let mut mean = vec![0.0; num_params];
    let mut var = vec![0.0; num_params];

    for s in train {
        for i in 0..num_params {
            mean[i] += s.features_property[i];
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    for s in train {
        for i in 0..num_params {
            let d = s.features_property[i] - mean[i];
            var[i] += d * d;
        }
    }
    let std = var.iter().map(|v| (v / n).sqrt()).collect();
    StandardScaler { mean, std }
}

impl TrainStep for Model<TrainBackend> {
    type Input = Batch<TrainBackend>;
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

impl InferenceStep for Model<ValidBackend> {
    type Input = Batch<ValidBackend>;
    type Output = RegressionOutput<ValidBackend>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        // This is exactly what your ValidStep does
        let pred = self.forward(batch.nodes, batch.adj, batch.mask, batch.globals);
        let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);

        RegressionOutput::new(loss, pred, batch.targets)
    }
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

fn read_data(csv_path: &Path, sdf_folder: &Path, tgt_col: usize) -> io::Result<Vec<Sample>> {
    let mut samples = Vec::new();

    let csv_data = fs::read_to_string(csv_path)?;

    // Skip the header.
    for (i, line) in csv_data.lines().skip(1).enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }

        let cols: Vec<String> = split_csv_line(line);

        // We determine which file to open based on our SDF-download script's convention,
        // using the CSV filename, and row index (0-based, skipping header).
        // let filename = &cols[0];
        let filename = csv_path.file_stem().unwrap().to_str().unwrap();

        let solubility: f32 = cols[tgt_col].parse().unwrap();

        let sdf_path = sdf_folder.join(format!("{filename}_id_{i}.sdf"));

        let mut mol: MoleculeSmall = {
            let sdf = match Sdf::load(&sdf_path) {
                Ok(s) => s,
                Err(e) => {
                    println!("Error loading SDF at path {sdf_path:?}: {:?}", e);
                    continue;
                }
            };
            sdf.clone().try_into()?
        };

        // We are experimenting with using our internally-derived characteristics
        // instead of those in the CSV; it may be more consistent.
        mol.characterization = Some(MolCharacterization::new(&mol.common));
        let features_property = param_feats_from_mol(&mol)?;

        if mol.common.bonds.is_empty() && mol.common.atoms.len() > 20 {
            println!("/n/nNo bonds found in SDF at path (Likely you RMed them) {sdf_path:?}/n/n");
        }

        if mol.common.bonds.is_empty() {
            eprintln!("No bonds found in SDF at path {sdf_path:?}. Skipping.");
            continue;
        }

        // ---------------------------------------------------

        let num_atoms = mol.common.atoms.len();
        if num_atoms == 0 || num_atoms > MAX_ATOMS {
            continue;
        }

        let (features_node, adj_list, _) = mol_to_graph_data(&mol);

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

// Note: We can make variants of this A/R tuned to specific inference items. For now, we are using
// a single  set of features for all targets.
/// Extract features from a molecule that are relevant for inferring the target parameter. We use this
/// in both training and inference workflows.
pub fn param_feats_from_mol(mol: &MoleculeSmall) -> io::Result<Vec<f32>> {
    let Some(c) = &mol.characterization else {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Missing mol characterization",
        ));
    };

    Ok(vec![
        c.num_atoms as f32,
        c.num_bonds as f32,
        c.mol_weight,
        c.num_heavy_atoms as f32,
        c.h_bond_acceptor.len() as f32,
        c.h_bond_donor.len() as f32,
        c.num_hetero_atoms as f32,
        c.halogen.len() as f32,
        c.rotatable_bonds.len() as f32,
        c.amines.len() as f32,
        c.amides.len() as f32,
        c.carbonyl.len() as f32,
        c.hydroxyl.len() as f32,
        c.num_valence_elecs as f32,
        c.num_rings_aromatic as f32,
        c.num_rings_saturated as f32,
        c.num_rings_aliphatic as f32,
        c.rings.len() as f32,
        c.log_p,
        c.molar_refractivity,
        c.psa_topo,
        c.asa_topo,
        c.volume,
        // Use the non-geometric TPSA and ASA values; they're more similar to the training data.
        // c.tpsa_ertl,
        // c.asa_labute,
        // c.balaban_j,
        // c.bertz_ct,
    ])
}

fn cli_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_owned())
}

pub fn main() {
    let args: Vec<String> = env::args().collect();

    let csv_path = cli_value(&args, "--csv").unwrap();
    let sdf_path = cli_value(&args, "--sdf").unwrap();
    let smiles_col = cli_value(&args, "--smiles");
    let smiles_col = cli_value(&args, "--target");

    // For now at least, the target name will always be the csv filename (Without extension)
    let target_name = Path::new(&csv_path).file_stem().unwrap().to_str().unwrap();

    let (model_path, scaler_path, config_path) = model_paths(target_name);

    let start = Instant::now();
    println!("Started training");

    let model_dir = Path::new(MODEL_DIR);
    fs::create_dir_all(model_dir).unwrap();

    // Data loading
    let mut data_csv_sdf =
        read_data(Path::new(&csv_path), Path::new(&sdf_path), TGT_COL_TDC).unwrap();

    println!("Sample len: {}", data_csv_sdf.len());

    let mut rng = StdRng::seed_from_u64(42);
    data_csv_sdf.shuffle(&mut rng);

    let split = ((data_csv_sdf.len() as f32) * 0.9).round() as usize;
    let (train_raw, valid_raw) = data_csv_sdf.split_at(split);

    let scaler = fit_scaler(train_raw);
    // let device = <TrainBackend as Backend>::Device::default();
    let device = Default::default();

    let train_loader = DataLoaderBuilder::new(Batcher_ {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .shuffle(42)
    .build(InMemDataset::new(train_raw.to_vec()));

    let valid_loader = DataLoaderBuilder::new(Batcher_ {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .build(InMemDataset::new(valid_raw.to_vec()));

    let num_params = train_raw[0].features_property.len();

    let model_cfg = ModelConfig {
        global_input_dim: num_params,
        atom_input_dim: FEAT_DIM_ATOMS,
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
    result.model.save_file(&model_path, &recorder).unwrap();

    //  Save the Model Config
    let config_file = fs::File::create(&config_path).expect("Could not create config file");
    serde_json::to_writer_pretty(config_file, &model_cfg).expect("Could not write config");
    println!("Saved config to {:?}", config_path);

    //  Save the Scaler
    // You need this for inference to know the means/stds to normalize new data.
    let scaler_file = fs::File::create(&scaler_path).expect("Could not create scaler file");
    serde_json::to_writer_pretty(scaler_file, &scaler).expect("Could not write scaler");
    println!("Saved scaler to {:?}", scaler_path);

    let elapsed = start.elapsed().as_secs();
    println!(
        "Training complete in {:?} s. Saved model to {model_path:?}",
        elapsed
    );
}
