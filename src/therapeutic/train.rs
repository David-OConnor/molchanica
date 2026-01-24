#![allow(unused)] // Required to prevent false positives.

//! Entry point for training of therapeutic properties. (Via a thin wrapper in `/src/train.rs` required
//! by Rust's system)
//!
//! This is tailored towards data from Therapeutic Data Commons (TDC).

//! To run: `cargo r --release --features train --bin train -- --path C:/Users/the_a/Desktop/bio_misc/tdc_data/bbb_martins.csv`
//!
//! Add the `tgt` param if training on a single file.
//! --tgt bbb_martins`

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    env, fs,
    hash::{Hash, Hasher},
    io,
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use bio_files::{AtomGeneric, BondGeneric, BondType, Sdf, md_params::ForceFieldParams};
#[cfg(feature = "train")]
use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::{
    backend::Autodiff,
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    lr_scheduler::constant::ConstantLr,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::AdamConfig,
    record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Tensor, TensorData, activation,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        ClassificationOutput, InferenceStep, Learner, LearnerSummary, RegressionOutput,
        SupervisedTraining, TrainOutput, TrainStep, TrainingStrategy,
        metric::{LossMetric, MetricDefinition},
        renderer::{
            EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
            MetricsRendererEvaluation, MetricsRendererTraining, TrainingProgress,
        },
    },
};
use dynamics::params::FfParamSet;
use na_seq::Element::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

#[cfg(feature = "train")]
use crate::therapeutic::model_eval::eval;
use crate::{
    mol_characterization::MolCharacterization,
    molecules::{Atom, Bond, MolType, small::MoleculeSmall},
    therapeutic::{DatasetTdc, train_test_split_indices::TrainTestSplit},
};

// Number of buckets for Force Field types (hashing trick)
// 20 is usually enough to capture major distinct atom types without too many collisions. (?)
pub(in crate::therapeutic) const FF_BUCKETS: usize = 20;

// 10 (Elements) + 1 (Degree) + 20 (FF Hashed) + 1 (Partial Charge)
pub(in crate::therapeutic) const FEAT_DIM_ATOMS: usize = 12 + FF_BUCKETS;

// todo: How should this be set up
pub(in crate::therapeutic) const MAX_ATOMS: usize = 100; // Max atoms for padding

pub(in crate::therapeutic) const MODEL_DIR: &str = "ml_models";
pub(in crate::therapeutic) const TGT_COL_TDC: usize = 2;

// Increasing layers may or may not improve model performance. It will slow down inference and training.
const NUM_GNN_LAYERS: usize = 6;
const NUM_MLP_LAYERS: usize = 6;
// It seems that low or no dropout significantly improves results, but perhaps it makes the
// model more likely to overfit, and makes it less general?
const DROPOUT: f64 = 0.1;

#[cfg(feature = "train")]
type TrainBackend = Autodiff<Wgpu>;
#[cfg(feature = "train")]
type ValidBackend = Wgpu;

/// Given a target (e.g. pharamaceutical property) name, get standardized filenames
/// for the (model, scalar, config).
pub(in crate::therapeutic) fn model_paths(data_set: DatasetTdc) -> (PathBuf, PathBuf, PathBuf) {
    let model_dir = Path::new(MODEL_DIR);

    // Extension is implicit in the model, for Burn.
    let model = model_dir.join(format!("{data_set}_model"));
    let scaler = model_dir.join(format!("{data_set}_scaler.json"));
    let cfg = model_dir.join(format!("{data_set}_model_config.json"));

    (model, scaler, cfg)
}

#[derive(Config, Debug)]
pub(in crate::therapeutic) struct ModelConfig {
    pub global_input_dim: usize,
    pub atom_input_dim: usize,
    pub gnn_hidden_dim: usize,
    pub mlp_hidden_dim: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let dim_gnn = self.gnn_hidden_dim;
        let dim_mlp = self.mlp_hidden_dim;
        let combined_dim = self.mlp_hidden_dim + self.gnn_hidden_dim;

        Model {
            gnn_layers: vec![LinearConfig::new(dim_gnn, dim_gnn).init(device); NUM_GNN_LAYERS],
            mlp_layers: vec![LinearConfig::new(dim_mlp, dim_mlp).init(device); NUM_MLP_LAYERS],
            fusion_norm: LayerNormConfig::new(combined_dim).init(device),
            head: LinearConfig::new(combined_dim, 1).init(device),
            dropout: DropoutConfig::new(DROPOUT).init(),
        }
    }
}

#[derive(Module, Debug)]
pub(in crate::therapeutic) struct Model<B: Backend> {
    /// GNN layers: Broadly graph and per-atom data.
    gnn_layers: Vec<Linear<B>>,
    /// Parameter features. These are for molecule-level parameters. (Atom count, weight, volume, PSA etc)
    mlp_layers: Vec<Linear<B>>,
    fusion_norm: LayerNorm<B>,
    /// Joint Branch
    head: Linear<B>,
    /// This dropout is useful if using more than 3 GNN layers.
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    /// Make a single middle GNN layer. (Use for the atom and bond graph)
    /// This can be used to construct any GNN layer.
    fn make_gnn_layer(
        &self,
        adj: &Tensor<B, 3>,
        mask: &Tensor<B, 3>,
        layer_in: Tensor<B, 3>,
        gnn_this_layer: &Linear<B>,
        dropout: bool,
        first_layer: bool,
    ) -> Tensor<B, 3> {
        let agg = adj.clone().matmul(layer_in.clone());
        let mut layer = activation::relu(gnn_this_layer.forward(agg));

        // We use dropout for all layers except the final.
        // Dropout randomly zeros out some values, so the model can't rely too heavily on a single
        // feature. It can reduce overfitting, and improve generalization. For example, this may
        // randomly remove a fraction ofn edges during training.
        if dropout {
            layer = self.dropout.forward(layer);
        }

        let term_0 = if first_layer { layer } else { layer + layer_in };

        term_0 * mask.clone()
    }

    pub fn forward(
        &self,
        nodes: Tensor<B, 3>,
        adj: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        params: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // The GNN layer: This relates to the bond graph, and per-atom features.
        let mut gnn_prev = nodes.clone();

        for (i, layer) in self.gnn_layers.iter().enumerate() {
            let dropout = i != self.gnn_layers.len() - 1;
            let first_layer = i == 0;

            gnn_prev = self.make_gnn_layer(&adj, &mask, gnn_prev, layer, dropout, first_layer);
        }

        // Pooling
        let graph_sum = gnn_prev.sum_dim(1);
        let atom_counts = mask.sum_dim(1);
        let graph_mean = graph_sum / (atom_counts + 1e-6);

        // todo: QC this.
        let graph_embedding = graph_mean.flatten(1, 2);

        // The MLP layers: These uses numerical parameters characteristic of the whole molecule.
        let mut mlp_prev = params;
        for (i, layer) in self.mlp_layers.iter().enumerate() {
            let x_mlp = activation::relu(layer.forward(mlp_prev.clone()));

            // Skip dropout on the final layer.
            if i != self.mlp_layers.len() - 1 {
                mlp_prev = self.dropout.forward(x_mlp);
            } else {
                mlp_prev = x_mlp;
            }
        }

        let combined = Tensor::cat(vec![graph_embedding, mlp_prev], 1);
        let combined = self.fusion_norm.forward(combined);

        self.head.forward(combined)
    }
}

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct Sample {
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
pub(in crate::therapeutic) struct Batch<B: Backend> {
    pub nodes: Tensor<B, 3>,
    pub adj: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
    pub globals: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone)]
pub(in crate::therapeutic) struct Batcher_ {
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

            let norm_y = self.scaler.normalize_target(item.target);
            batch_y.push(norm_y);

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(in crate::therapeutic) struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub y_mean: f32,
    pub y_std: f32,
}

impl StandardScaler {
    pub fn normalize_target(&self, y: f32) -> f32 {
        let s = if self.y_std.abs() < 1e-9 {
            1.0
        } else {
            self.y_std
        };
        (y - self.y_mean) / s
    }

    pub fn denormalize_target(&self, y_norm: f32) -> f32 {
        let s = if self.y_std.abs() < 1e-9 {
            1.0
        } else {
            self.y_std
        };
        y_norm * s + self.y_mean
    }

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
pub(in crate::therapeutic) fn pad_graph_data(
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

/// Helper to deterministically map a string to a bucket index [0..FF_BUCKETS)
fn hash_ff_type(ff_type: &str) -> usize {
    let mut s = DefaultHasher::new();
    ff_type.hash(&mut s);
    (s.finish() as usize) % FF_BUCKETS
}

/// Helper: Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
pub(in crate::therapeutic) fn mol_to_graph_data(
    mol: &MoleculeSmall,
) -> io::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let num_atoms = mol.common.atoms.len();

    // Atom (node) features (Element, Degree, FF Type, atom name, Charge)
    // Make sure FEAT_DIM_ATOMS is set to 32 (10 elements + 1 degree + 20 FF + 1 charge)
    let mut node_feats = Vec::with_capacity(num_atoms * FEAT_DIM_ATOMS);
    let degrees: Vec<usize> = mol.common.adjacency_list.iter().map(|n| n.len()).collect();

    for (i, atom) in mol.common.atoms.iter().enumerate() {
        let mut f = vec![0.; FEAT_DIM_ATOMS];

        // Element One-Hot (Indices 0-9)
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

        // Degree Feature
        // Normalized roughly to 0-1 range
        f[10] = degrees[i] as f32 * 0.16;

        // C. Force Field Type Hashed One-Hot (Indices 11-30)
        // We handle missing FF types gracefully, though ideally your data has them.
        if let Some(ff_type) = &atom.force_field_type {
            let bucket = hash_ff_type(ff_type);
            // Offset by 11 (previous features)
            if 11 + bucket < FEAT_DIM_ATOMS {
                f[11 + bucket] = 1.0;
            }
        } else {
            return Err(io::Error::new(
                ErrorKind::Other,
                "Atom missing force field type",
            ));
        }

        // if let Some(atom_name) = &atom.type_in_res_general {
        //     let bucket = hash_ff_type(atom_name);
        //     // Offset by 11 (previous features)
        //     if 12 + bucket < FEAT_DIM_ATOMS {
        //         f[12 + bucket] = 1.0;
        //     }
        // } else {
        //     return Err(io::Error::new(ErrorKind::Other, "Atom missing type-in-res"));
        // }

        // D. Partial Charge
        if let Some(partial_charge) = &atom.partial_charge {
            f[31] = *partial_charge;
        } else {
            return Err(io::Error::new(
                ErrorKind::Other,
                "Atom missing partial charge",
            ));
        }

        node_feats.extend(f);
    }

    // Weighted Adjacency
    let mut raw_adj = vec![0.0; num_atoms * num_atoms];

    for i in 0..num_atoms {
        raw_adj[i * num_atoms + i] = 1.0;
    }

    // Fill edges based on Bond Type
    for bond in &mol.common.bonds {
        let u = bond.atom_0;
        let v = bond.atom_1;

        if u >= num_atoms || v >= num_atoms {
            continue;
        }

        let weight = match bond.bond_type {
            BondType::Single => 1.0,
            BondType::Aromatic => 1.5, // Significant for solubility/planarity
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            _ => 1.0,
        };

        raw_adj[u * num_atoms + v] = weight;
        raw_adj[v * num_atoms + u] = weight;
    }

    // Symmetric Normalization (Using Weights)
    // D^(-0.5) * A * D^(-0.5)

    // Calculate weighted degrees (sum of weights in row)
    let mut degrees_vec = vec![0.0; num_atoms];
    for i in 0..num_atoms {
        let mut d = 0.0;
        for j in 0..num_atoms {
            d += raw_adj[i * num_atoms + j];
        }
        degrees_vec[i] = d;
    }

    let mut final_adj = vec![0.0; num_atoms * num_atoms];
    for i in 0..num_atoms {
        for j in 0..num_atoms {
            let val = raw_adj[i * num_atoms + j];
            if val > 0.0 {
                let d_i: f32 = degrees_vec[i];
                let d_j = degrees_vec[j];
                // Avoid division by zero if a node is somehow isolated (though self-loop prevents this)
                final_adj[i * num_atoms + j] = val / (d_i * d_j).sqrt();
            }
        }
    }

    // Mask is just 1s for valid atoms
    let mask = vec![1.0; num_atoms];

    Ok((node_feats, final_adj, mask))
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

    let mut y_sum = 0.0;
    for s in train {
        y_sum += s.target;
    }
    let y_mean = y_sum / n;

    let mut y_var = 0.0;
    for s in train {
        let diff = s.target - y_mean;
        y_var += diff * diff;
    }
    let y_std = (y_var / n).sqrt();

    let std = var.iter().map(|v| (v / n).sqrt()).collect();

    StandardScaler {
        mean,
        std,
        y_mean,
        y_std,
    }
}

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
/// Loads molecules from SDF files in a folder, and target data from a CSV. Used in both training and
/// evaluation workflows.
pub(in crate::therapeutic) fn load_training_data(
    csv_path: &Path,
    sdf_path: &Path,
    tgt_col: usize,
    // Optionally filter to only certain indices. This is useful in test/train splits for evaluation.
    indices: Option<&[usize]>,
    mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
    gaff2: &ForceFieldParams,
) -> io::Result<Vec<(MoleculeSmall, f32)>> {
    let csv_file = fs::File::open(csv_path)?;
    let mut rdr = csv::Reader::from_reader(csv_file);

    let csv_data = fs::read_to_string(csv_path)?;

    let mut result = Vec::new();

    // Iterate over records (automatically handles quotes and headers)
    for (i, record) in rdr.records().enumerate() {
        if let Some(ind) = indices {
            if !ind.contains(&i) {
                continue;
            }
        }

        let record = record?;

        // Robust float parsing
        let target_str = &record[tgt_col];
        let target: f32 = match target_str.parse() {
            Ok(v) => v,
            Err(_) => {
                // If we can't parse the target (e.g. "NaN"), skip this sample
                continue;
            }
        };

        // We determine which file to open based on our SDF-download script's convention,
        // using the CSV filename, and row index (0-based, skipping header).
        // let filename = &cols[0];
        let filename = csv_path.file_stem().unwrap().to_str().unwrap();

        let sdf_path = sdf_path.join(format!("{filename}_id_{i}.sdf"));

        let mut mol: MoleculeSmall = {
            let sdf = match Sdf::load(&sdf_path) {
                Ok(s) => s,
                Err(e) => {
                    // We accept that some SDF files are missing from not being able
                    // to download them from PubChem.
                    // println!("Error loading SDF at path {sdf_path:?}: {:?}", e);
                    continue;
                }
            };
            sdf.clone().try_into()?
        };

        mol.update_ff_related(mol_specific_param_set, gaff2, true);

        // We are experimenting with using our internally-derived characteristics
        // instead of those in the CSV; it may be more consistent.
        mol.characterization = Some(MolCharacterization::new(&mol.common));

        result.push((mol, target));
    }

    Ok(result)
}

/// For training: Load CSV and SDF molecule data for a training set.
/// If doing a train/test split (e.g. for eval), set indices to the training ones.
#[cfg(feature = "train")]
fn read_data(
    csv_path: &Path,
    sdf_path: &Path,
    tgt_col: usize,
    indices: Option<&[usize]>,
    mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
    gaff2: &ForceFieldParams,
) -> io::Result<Vec<Sample>> {
    let mut samples = Vec::new();

    let loaded = load_training_data(
        csv_path,
        sdf_path,
        tgt_col,
        indices,
        mol_specific_param_set,
        gaff2,
    )?;

    for (mol, target) in loaded {
        let feat_params = param_feats_from_mol(&mol)?;

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

        let (features_node, adj_list, _) = mol_to_graph_data(&mol)?;

        samples.push(Sample {
            features_property: feat_params,
            features_node,
            adj_list,
            num_atoms,
            target,
        });
    }

    Ok(samples)
}

// Note: We can make variants of this A/R tuned to specific inference items. For now, we are using
// a single  set of features for all targets.
/// Extract features from a molecule that are relevant for inferring the target parameter. We use this
/// in both training and inference workflows.
pub(in crate::therapeutic) fn param_feats_from_mol(mol: &MoleculeSmall) -> io::Result<Vec<f32>> {
    let Some(c) = &mol.characterization else {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Missing mol characterization",
        ));
    };

    // Helper to compress large ranges (Log1p)
    // We use abs() to handle potential negative LogP inputs safely if you apply it there,
    // though usually we only apply this to Counts and Weights.
    let ln = |x: f32| (x + 1.0).ln();

    // We are generally apply ln to values that can be "large".

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
        c.wiener_index.unwrap_or(0) as f32,
        //
        // ----
        //
        // ln(c.num_atoms as f32),
        // ln(c.num_bonds as f32),
        // ln(c.mol_weight),
        // ln(c.num_heavy_atoms as f32),
        // c.h_bond_acceptor.len() as f32,
        // c.h_bond_donor.len() as f32,
        // c.num_hetero_atoms as f32,
        // c.halogen.len() as f32,
        // c.rotatable_bonds.len() as f32,
        // c.amines.len() as f32,
        // c.amides.len() as f32,
        // c.carbonyl.len() as f32,
        // c.hydroxyl.len() as f32,
        // c.num_valence_elecs as f32,
        // c.num_rings_aromatic as f32,
        // c.num_rings_saturated as f32,
        // c.num_rings_aliphatic as f32,
        // c.rings.len() as f32,
        // c.log_p,
        // c.molar_refractivity,
        // ln(c.psa_topo),
        // ln(c.asa_topo),
        // ln(c.volume),
        // ln(c.wiener_index.unwrap_or(0) as f32),
    ])
}

fn cli_has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn cli_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_owned())
}

#[cfg(feature = "train")]
fn train(
    data_path: &Path,
    dataset: DatasetTdc,
    tgt_col: usize,
    mol_specific_params: &mut HashMap<String, ForceFieldParams>,
    gaff2: &ForceFieldParams,
) -> io::Result<()> {
    let (csv_path, mol_path) = dataset.csv_mol_paths(data_path)?;

    // For now at least, the target name will always be the csv filename (Without extension)
    let target_name = Path::new(&csv_path).file_stem().unwrap().to_str().unwrap();

    let (model_path, scaler_path, config_path) = model_paths(dataset);

    let start = Instant::now();
    println!("Started training on {csv_path:?}");

    let model_dir = Path::new(MODEL_DIR);
    fs::create_dir_all(model_dir)?;

    let mut data = read_data(
        Path::new(&csv_path),
        Path::new(&mol_path),
        tgt_col,
        None,
        mol_specific_params,
        &gaff2,
    )?;

    let data_len = data.len();

    let num_params = data[0].features_property.len();

    let (data_train, data_validation) = {
        let tts = TrainTestSplit::new(dataset);
        let mut data_tr = Vec::with_capacity(tts.train.len());
        let mut data_v = Vec::with_capacity(tts.test.len());

        for (i, v) in data.into_iter().enumerate() {
            if tts.train.contains(&i) {
                data_tr.push(v);
            } else if tts.test.contains(&i) {
                data_v.push(v);
            } else {
                eprintln!(
                    "Warning: Invalid split. Train len: {}, Test len: {} total: {}. missing: {}",
                    tts.train.len(),
                    tts.test.len(),
                    data_len,
                    data_len - (tts.train.len() + tts.test.len()),
                );
                // todo Hmmm. What could be causing the mismatch?
                // return Err(io::Error::new(
                //     ErrorKind::Other,
                //     format!(
                //         "Invalid split. Train len: {}, Test len: {} total: {}. missing: {}",
                //         tts.train.len(),
                //         tts.test.len(),
                //         data_len,
                //         data_len - (tts.train.len() + tts.test.len()),
                //     ),
                // ));
            }
        }

        (data_tr, data_v)
    };

    println!("Training on : {} samples", data_train.len());

    let scaler = fit_scaler(&data_train);
    let device = Default::default();

    // burn_wgpu::init_setup::<burn_wgpu::graphics::Dx12>(&device, RuntimeOptions::default());

    let train_loader = DataLoaderBuilder::new(Batcher_ {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .shuffle(42)
    .build(InMemDataset::new(data_train));

    let valid_loader = DataLoaderBuilder::new(Batcher_ {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .build(InMemDataset::new(data_validation.to_vec()));

    let model_cfg = ModelConfig {
        global_input_dim: num_params,
        atom_input_dim: FEAT_DIM_ATOMS,
        gnn_hidden_dim: 64,
        mlp_hidden_dim: 128,
    };

    let model = model_cfg.init::<TrainBackend>(&device);

    println!("Model parameter count: {}", model.num_params());

    let optim = AdamConfig::new().init();
    let lr_scheduler = ConstantLr::new(3e-4);

    let training = SupervisedTraining::new(model_dir.to_str().unwrap(), train_loader, valid_loader)
        .metrics((LossMetric::new(),)) // Note the tuple format for metrics
        .num_epochs(80)
        .with_training_strategy(TrainingStrategy::SingleDevice(device.clone()))
        .summary(); // Provides the TUI/CLI output

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

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

    Ok(())
}

#[cfg(feature = "train")]
pub fn main() {
    let args: Vec<String> = env::args().collect();

    // Assumption: In this path is both A: a CSV for each param we wish to train and B: a corresponding
    // folder filled with SDF files for each of these.
    let path = cli_value(&args, "--path").unwrap();
    let data_path = Path::new(&path);

    // Allow passing a single target name, vs the whole folder.
    let target = cli_value(&args, "--tgt");
    let eval_ = cli_has_flag(&args, "--eval");

    // Load force field data, which we need for FF type and partial charge.
    let gaff2 = FfParamSet::new_amber().unwrap().small_mol.unwrap();
    let mol_specific_params = &mut HashMap::new();

    let datasets = match target {
        Some(t) => vec![DatasetTdc::from_str(&t).unwrap()],
        None => DatasetTdc::all(),
    };

    for dataset in datasets {
        if eval_ {
            match eval(data_path, dataset, TGT_COL_TDC, mol_specific_params, &gaff2) {
                Ok(ev) => {
                    println!("Eval for {dataset}: {ev}");
                }
                Err(e) => {
                    eprintln!("Error evaluating {dataset}: {e}");
                }
            }
        } else {
            if let Err(e) = train(data_path, dataset, TGT_COL_TDC, mol_specific_params, &gaff2) {
                eprintln!("Error training {dataset}: {e}");
            }
        }
    }
}
