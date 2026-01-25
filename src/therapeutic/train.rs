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
    collections::{HashMap, HashSet, hash_map::DefaultHasher},
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
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::Int;
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

use crate::molecules::build_adjacency_list;
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
// pub(in crate::therapeutic) const FEAT_DIM_ATOMS: usize = 12 + FF_BUCKETS;

// todo: How should this be set up
pub(in crate::therapeutic) const MAX_ATOMS: usize = 100; // Max atoms for padding

pub(in crate::therapeutic) const MODEL_DIR: &str = "ml_models";
pub(in crate::therapeutic) const TGT_COL_TDC: usize = 2;

// Note: Excluding H or not appears not to make any notable difference at first;
// Experiment with this more later.
const EXCLUDE_HYDROGEN: bool = true;

// Increasing layers may or may not improve model performance. It will slow down inference and training.
const NUM_GNN_LAYERS: usize = 3;
const NUM_MLP_LAYERS: usize = 3;
// It seems that low or no dropout significantly improves results, but perhaps it makes the
// model more likely to overfit, and makes it less general? Maybe 0.1 or disabled.
// Perhaps skip dropout due to our small TDC data sets.
const DROPOUT: Option<f64> = None;

const BOND_SIGMA_SQ: f32 = 3.3; // Å. Try 1.5 - 2.2 for sigma, (Square it)

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

fn vocab_lookup_element(el: na_seq::Element) -> i32 {
    // 0 is reserved for Padding in the Batcher, so we start at 1.
    match el {
        Hydrogen => 1,
        Carbon => 2,
        Nitrogen => 3,
        Oxygen => 4,
        Fluorine => 5,
        Phosphorus => 6,
        Sulfur => 7,
        Chlorine => 8,
        Bromine => 9,
        Iodine => 10,
        _ => 11, // "Other" bucket
    }
}

fn vocab_lookup_ff(ff: Option<&String>) -> i32 {
    // 0 is Padding.
    match ff {
        Some(s) => {
            // Hash to range [1..20]
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            ((h.finish() % (FF_BUCKETS as u64)) + 1) as i32
        }
        None => (FF_BUCKETS as i32) + 1, // Unknown bucket
    }
}

#[derive(Config, Debug)]
pub(in crate::therapeutic) struct ModelConfig {
    pub global_input_dim: usize,
    // pub atom_input_dim: usize,
    pub gnn_hidden_dim: usize,
    pub mlp_hidden_dim: usize,
    // These covab and embeddings are used for per-atom GNN properties, e.g. element.
    pub vocab_size_elem: usize,
    pub vocab_size_ff: usize,
    pub embedding_dim: usize,
    /// E.g. 2, one for charge; one for degree, one for R, one for mean_nb_dist
    pub n_node_scalars: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let dim_gnn = self.gnn_hidden_dim;
        let dim_mlp = self.mlp_hidden_dim;
        let combined_dim = self.mlp_hidden_dim + self.gnn_hidden_dim;

        let emb_elem = EmbeddingConfig::new(self.vocab_size_elem, self.embedding_dim).init(device);
        let emb_ff = EmbeddingConfig::new(self.vocab_size_ff, self.embedding_dim).init(device);

        // The input to the GNN is now: embedding_dim + embedding_dim + scalar_features (degree, charge)
        let gnn_input_dim = self.embedding_dim * 2 + self.n_node_scalars;

        // GNN
        let mut gnn_layers = Vec::with_capacity(NUM_GNN_LAYERS);
        for layer_i in 0..NUM_GNN_LAYERS {
            let (in_dim, out_dim) = if layer_i == 0 {
                (gnn_input_dim, dim_gnn)
            } else {
                (dim_gnn, dim_gnn)
            };
            gnn_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
        }

        // MLP
        let mut mlp_layers = Vec::with_capacity(NUM_MLP_LAYERS);
        for layer_i in 0..NUM_MLP_LAYERS {
            let (in_dim, out_dim) = if layer_i == 0 {
                (self.global_input_dim, dim_mlp)
            } else {
                (dim_mlp, dim_mlp)
            };
            mlp_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
        }

        Model {
            emb_elem,
            emb_ff,
            gnn_layers,
            mlp_layers,
            fusion_norm: LayerNormConfig::new(combined_dim).init(device),
            head: LinearConfig::new(combined_dim, 1).init(device),
            dropout: DropoutConfig::new(DROPOUT.unwrap_or_default()).init(),
        }
    }
}

#[derive(Module, Debug)]
pub(in crate::therapeutic) struct Model<B: Backend> {
    emb_elem: Embedding<B>,
    emb_ff: Embedding<B>,
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
        if dropout && DROPOUT.is_some() {
            layer = self.dropout.forward(layer);
        }

        let term_0 = if first_layer { layer } else { layer + layer_in };

        term_0 * mask.clone()
    }

    pub fn forward(
        &self,
        elem_ids: Tensor<B, 2, Int>, // [Batch, MaxAtoms]
        ff_ids: Tensor<B, 2, Int>,   // [Batch, MaxAtoms]
        scalars: Tensor<B, 3>,       // [Batch, MaxAtoms, NumScalars] (Charge, Degree)
        adj: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        params: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Lookup Embeddings
        let x_elem = self.emb_elem.forward(elem_ids); // [Batch, MaxAtoms, EmbDim]
        let x_ff = self.emb_ff.forward(ff_ids); // [Batch, MaxAtoms, EmbDim]

        // Concatenate embeddings with scalar features (Charge, Degree)
        let nodes = Tensor::cat(vec![x_elem, x_ff, scalars], 2);

        // The GNN layer: This relates to the bond graph, and per-atom features.
        let mut gnn_prev = nodes;

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
        // let graph_embedding = graph_mean.flatten(1, 2);
        // let graph_embedding = graph_mean;
        // let graph_embedding =
        //     graph_mean.reshape([graph_mean.dims()[0], self.gnn_layers[0].out_features()]);

        // Graph_mean is Tensor<B, 3> with shape [B, 1, D]
        let [b, _one, d] = graph_mean.dims();
        let graph_embedding = graph_mean.reshape([b, d]); // Tensor<B, 2> [B, D]

        // The MLP layers: These uses numerical parameters characteristic of the whole molecule.
        let mut mlp_prev = params;
        for (i, layer) in self.mlp_layers.iter().enumerate() {
            mlp_prev = activation::relu(layer.forward(mlp_prev.clone()));

            // Skip dropout on the final layer.
            if i != self.mlp_layers.len() - 1 && DROPOUT.is_some() {
                mlp_prev = self.dropout.forward(mlp_prev);
            }
        }

        // let combined = Tensor::cat(vec![graph_embedding, mlp_prev], 1);
        let combined = Tensor::cat(vec![graph_embedding, mlp_prev], 1); // both Tensor<B,2>oth Tensor<B,2>
        let combined = self.fusion_norm.forward(combined);

        self.head.forward(combined)
    }
}

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct Sample {
    /// From computed properties of the molecule.
    pub features_property: Vec<f32>,

    pub elem_ids: Vec<i32>, // Ints for Embedding
    pub ff_ids: Vec<i32>,   // Ints for Embedding
    pub scalars: Vec<f32>,  // Charge, Degree

    /// From the atom/bond graph
    // pub features_node: Vec<f32>,
    pub adj_list: Vec<f32>,
    pub num_atoms: usize,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct Batch<B: Backend> {
    pub elem_ids: Tensor<B, 2, Int>,
    pub ff_ids: Tensor<B, 2, Int>,
    pub scalars: Tensor<B, 3>,
    // pub nodes: Tensor<B, 3>,
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

        let mut batch_elem_ids = Vec::new();
        let mut batch_ff_ids = Vec::new();
        let mut batch_scalars = Vec::new();
        let mut batch_adj = Vec::new();
        let mut batch_mask = Vec::new();
        let mut batch_globals = Vec::new();
        let mut batch_y = Vec::new();

        let n_feat_params = items[0].features_property.len();
        // Calculate num scalars per atom based on first item
        let n_scalars_per_atom = if items[0].num_atoms > 0 {
            items[0].scalars.len() / items[0].num_atoms
        } else {
            2
        };

        for mut item in items {
            // 1. Globals & Target
            self.scaler.apply_in_place(&mut item.features_property);
            batch_globals.extend_from_slice(&item.features_property);
            batch_y.push(self.scaler.normalize_target(item.target));

            // 2. Pad & Extend Graph Data
            let n = item.num_atoms.min(MAX_ATOMS);

            // -- Elem IDs (Int) --
            batch_elem_ids.extend_from_slice(&item.elem_ids[0..n]);
            batch_elem_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

            // -- FF IDs (Int) --
            batch_ff_ids.extend_from_slice(&item.ff_ids[0..n]);
            batch_ff_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

            // -- Scalars (Float) --
            batch_scalars.extend_from_slice(&item.scalars[0..n * n_scalars_per_atom]);
            batch_scalars.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * n_scalars_per_atom));

            // -- Adj & Mask (Float) --
            // FIXED: Using new helper function
            let (p_adj, p_mask) = pad_adj_and_mask(&item.adj_list, item.num_atoms);
            batch_adj.extend(p_adj);
            batch_mask.extend(p_mask);
        }

        let elem_ids = TensorData::new(batch_elem_ids, [batch_size, MAX_ATOMS]);
        let ff_ids = TensorData::new(batch_ff_ids, [batch_size, MAX_ATOMS]);
        let scalars = TensorData::new(batch_scalars, [batch_size, MAX_ATOMS, n_scalars_per_atom]);
        let adj = TensorData::new(batch_adj, [batch_size, MAX_ATOMS, MAX_ATOMS]);
        let mask = TensorData::new(batch_mask, [batch_size, MAX_ATOMS, 1]);
        let globals = TensorData::new(batch_globals, [batch_size, n_feat_params]);
        let y = TensorData::new(batch_y, [batch_size, 1]);

        Batch {
            elem_ids: Tensor::from_data(elem_ids, device),
            ff_ids: Tensor::from_data(ff_ids, device),
            scalars: Tensor::from_data(scalars, device),
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
pub(in crate::therapeutic) fn pad_adj_and_mask(
    raw_adj: &[f32],
    num_atoms: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(MAX_ATOMS);

    // 1. Mask: 1.0 for atoms, 0.0 for pad
    let mut p_mask = Vec::with_capacity(MAX_ATOMS);
    p_mask.extend(std::iter::repeat(1.0).take(n));
    p_mask.extend(std::iter::repeat(0.0).take(MAX_ATOMS - n));

    // 2. Adj: Reconstruct row-by-row to handle 2D padding
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

    (p_adj, p_mask)
}

/// Helper to deterministically map a string to a bucket index [0..FF_BUCKETS)
fn hash_ff_type(ff_type: &str) -> usize {
    let mut s = DefaultHasher::new();
    ff_type.hash(&mut s);
    (s.finish() as usize) % FF_BUCKETS
}

// todo: Experimental!
fn atom_geom_scalars(atoms: &[Atom], adj: &[Vec<usize>]) -> Vec<(f32, f32)> {
    let n = atoms.len().max(1);

    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;

    for a in atoms {
        cx += a.posit.x as f32;
        cy += a.posit.y as f32;
        cz += a.posit.z as f32;
    }

    let inv_n = 1.0 / (n as f32);
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    // Scale by RMS radius to keep numbers ~O(1)
    let mut r2_sum = 0.0f32;
    for a in atoms {
        let dx = a.posit.x as f32 - cx;
        let dy = a.posit.y as f32 - cy;
        let dz = a.posit.z as f32 - cz;
        r2_sum += dx * dx + dy * dy + dz * dz;
    }
    let rms = (r2_sum * inv_n).sqrt().max(1e-6);

    let mut out = Vec::with_capacity(atoms.len());

    for (i, a) in atoms.iter().enumerate() {
        let dx = (a.posit.x as f32 - cx) / rms;
        let dy = (a.posit.y as f32 - cy) / rms;
        let dz = (a.posit.z as f32 - cz) / rms;

        let r = (dx * dx + dy * dy + dz * dz).sqrt(); // invariant

        // mean neighbor distance (raw Å-ish; you can also divide by rms)
        let mut sum = 0.0f32;
        let mut cnt = 0.0f32;
        for &j in adj.get(i).unwrap_or(&Vec::new()).iter() {
            let b = &atoms[j];
            let ddx = (a.posit.x - b.posit.x) as f32;
            let ddy = (a.posit.y - b.posit.y) as f32;
            let ddz = (a.posit.z - b.posit.z) as f32;
            sum += (ddx * ddx + ddy * ddy + ddz * ddz).sqrt();
            cnt += 1.0;
        }
        let mean_nb_dist = if cnt > 0.0 { sum / cnt } else { 0.0 };

        out.push((r, mean_nb_dist));
    }

    out
}

/// Helper: Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
pub(in crate::therapeutic) fn mol_to_graph_data(
    mol: &MoleculeSmall,
) -> io::Result<(Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>, usize)> {
    let (atoms, bonds, adj) = if EXCLUDE_HYDROGEN {
        let a: Vec<_> = mol
            .common
            .atoms
            .iter()
            .filter(|a| a.element != Hydrogen)
            .cloned()
            .collect();
        let sns: Vec<_> = a.iter().map(|a| a.serial_number).collect();
        let mut sn_to_new = HashMap::with_capacity(a.len());
        for (new_i, a) in a.iter().enumerate() {
            sn_to_new.insert(a.serial_number, new_i);
        }

        let mut bonds_ = Vec::new();
        for b in mol.common.bonds.iter() {
            if let (Some(&u), Some(&v)) = (sn_to_new.get(&b.atom_0_sn), sn_to_new.get(&b.atom_1_sn))
            {
                let mut b2 = b.clone();
                b2.atom_0 = u;
                b2.atom_1 = v;
                bonds_.push(b2);
            }
        }
        let adj = build_adjacency_list(&bonds_, a.len());
        (a, bonds_, adj)
    } else {
        (
            mol.common.atoms.clone(),
            mol.common.bonds.clone(),
            mol.common.adjacency_list.clone(),
        )
    };

    let num_atoms = atoms.len();
    if num_atoms == 0 {
        return Err(io::Error::new(ErrorKind::Other, "Molecule has 0 atoms"));
    }

    // --- 1. NODE FEATURES (Indices & Scalars) ---
    let mut elem_indices = Vec::with_capacity(num_atoms);
    let mut ff_indices = Vec::with_capacity(num_atoms);
    let mut scalars = Vec::with_capacity(num_atoms * 2);

    let geom = atom_geom_scalars(&atoms, &adj);

    for (i, atom) in atoms.iter().enumerate() {
        elem_indices.push(vocab_lookup_element(atom.element));
        ff_indices.push(vocab_lookup_ff(atom.force_field_type.as_ref()));

        let degree = adj.get(i).map(|n| n.len()).unwrap_or(0);
        scalars.push(degree as f32 / 6.0);
        scalars.push(atom.partial_charge.unwrap_or(0.0));

        let (r, mean_nb_dist) = geom[i];
        scalars.push(r);
        scalars.push(mean_nb_dist);
    }

    // --- 2. EDGE FEATURES (Weighted Adjacency) ---
    let mut raw_adj = vec![0.0; num_atoms * num_atoms];
    // Self loops
    for i in 0..num_atoms {
        raw_adj[i * num_atoms + i] = 1.0;
    }

    for bond in &bonds {
        let u = bond.atom_0;
        let v = bond.atom_1;
        if u >= num_atoms || v >= num_atoms {
            continue;
        }

        // Euclidean Distance
        let p1 = atoms[u].posit;
        let p2 = atoms[v].posit;
        let dist =
            ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2) + (p1.z - p2.z).powi(2)).sqrt() as f32;

        let bond_strength = match bond.bond_type {
            BondType::Single => 1.0,
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            BondType::Aromatic => 1.5,
            _ => 1.0,
        };

        let k = (-(dist.powi(2)) / (2.0 * BOND_SIGMA_SQ)).exp();
        let weight = bond_strength * k;

        raw_adj[u * num_atoms + v] = weight;
        raw_adj[v * num_atoms + u] = weight;
    }

    // Symmetric Normalization: D^(-0.5) * A * D^(-0.5)
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
                let d_i = degrees_vec[i].max(1e-8);
                let d_j = degrees_vec[j].max(1e-8);
                final_adj[i * num_atoms + j] = val / (d_i * d_j).sqrt();
            }
        }
    }

    Ok((elem_indices, ff_indices, scalars, final_adj, num_atoms))
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
        let pred = self.forward(
            batch.elem_ids,
            batch.ff_ids,
            batch.scalars,
            batch.adj,
            batch.mask,
            batch.globals,
        );

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
        let pred = self.forward(
            batch.elem_ids,
            batch.ff_ids,
            batch.scalars,
            batch.adj,
            batch.mask,
            batch.globals,
        );

        let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);

        RegressionOutput::new(loss, pred, batch.targets)
    }
}

/// Each field is the molecule, and target value.
#[derive(Clone, Debug, Default)]
pub(in crate::therapeutic) struct TrainingData {
    pub train: Vec<(MoleculeSmall, f32)>,
    pub test: Vec<(MoleculeSmall, f32)>,
}

// impl TrainingData {
//     pub fn new(train: Vec<(MoleculeSmall, f32)>, test: Vec<(MoleculeSmall, f32)>) -> Self {
//         Self { train, test }
//     }
// }

#[cfg(feature = "train")]
/// Loads molecules from SDF files in a folder, and target data from a CSV. Used in both training and
/// evaluation workflows.
///
/// We run this split while loading, upstream of skipping molecules for any reason.
/// This ensures the indices remain correct after skipping molecules.
pub(in crate::therapeutic) fn load_training_data(
    csv_path: &Path,
    sdf_path: &Path,
    tgt_col: usize,
    tts: &TrainTestSplit,
    mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
    gaff2: &ForceFieldParams,
    test_only: bool,
) -> io::Result<TrainingData> {
    let csv_file = fs::File::open(csv_path)?;
    let mut rdr = csv::Reader::from_reader(csv_file);

    let mut result = TrainingData::default();

    // These Hash sets improve speed over  using the tts variables directly. (Double-nested loop)
    let train_set: HashSet<usize> = tts.train.iter().copied().collect();
    let test_set: HashSet<usize> = tts.test.iter().copied().collect();

    // Iterate over records (automatically handles quotes and headers)
    for (i, record) in rdr.records().enumerate() {
        if test_only && !test_set.contains(&i) {
            continue;
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

        if train_set.contains(&i) {
            result.train.push((mol, target));
        } else if test_set.contains(&i) {
            result.test.push((mol, target));
        } else {
            eprintln!("Warning: Record {i} not present in the train/test split for set {filename}");
        }
    }

    Ok(result)
}

/// For training: Load CSV and SDF molecule data for a training set.
/// Returns (train, test)
#[cfg(feature = "train")]
fn read_data(
    csv_path: &Path,
    sdf_path: &Path,
    tgt_col: usize,
    tts: &TrainTestSplit,
    mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
    gaff2: &ForceFieldParams,
) -> io::Result<(Vec<Sample>, Vec<Sample>)> {
    let loaded = load_training_data(
        csv_path,
        sdf_path,
        tgt_col,
        tts,
        mol_specific_param_set,
        gaff2,
        false,
    )?;

    let mut result_train = Vec::new();
    let mut result_test = Vec::new();

    for (result_set, data) in [
        (&mut result_train, &loaded.train),
        (&mut result_test, &loaded.test),
    ] {
        for (mol, target) in data {
            let feat_params = param_feats_from_mol(&mol)?;

            if mol.common.bonds.is_empty() {
                // eprintln!("No bonds found in SDF. Skipping.");
                continue;
            }

            // FIXED: Destructuring the new return tuple
            let (elem_ids, ff_ids, scalars, adj_list, num_atoms) = match mol_to_graph_data(&mol) {
                Ok(res) => res,
                Err(_) => continue, // Skip malformed molecules
            };

            if num_atoms == 0 || num_atoms > MAX_ATOMS {
                continue;
            }

            result_set.push(Sample {
                features_property: feat_params,
                elem_ids, // Defined now
                ff_ids,   // Defined now
                scalars,  // Defined now
                adj_list,
                num_atoms,
                target: *target,
            });
        }
    }
    Ok((result_train, result_test))
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
        // c.num_atoms as f32,
        // c.num_bonds as f32,
        // c.mol_weight,
        // c.num_heavy_atoms as f32,
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
        // c.psa_topo,
        // c.asa_topo,
        // c.volume,
        // c.wiener_index.unwrap_or(0) as f32,
        //
        // ----
        //
        ln(c.num_atoms as f32),
        ln(c.num_bonds as f32),
        ln(c.mol_weight),
        ln(c.num_heavy_atoms as f32),
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
        ln(c.psa_topo),
        ln(c.asa_topo),
        ln(c.volume),
        ln(c.wiener_index.unwrap_or(0) as f32),
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
pub(in crate::therapeutic) fn train(
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

    let tts = TrainTestSplit::new(dataset);

    let (data_train, data_valid) = read_data(
        Path::new(&csv_path),
        Path::new(&mol_path),
        tgt_col,
        &tts,
        mol_specific_params,
        &gaff2,
    )?;

    println!("Training on : {} samples", data_train.len());

    let scaler = fit_scaler(&data_train);
    let device = Default::default();

    // burn_wgpu::init_setup::<burn_wgpu::graphics::Dx12>(&device, RuntimeOptions::default());

    let train_loader = DataLoaderBuilder::new(Batcher_ {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .shuffle(42)
    .build(InMemDataset::new(data_train.clone()));

    let valid_loader = DataLoaderBuilder::new(Batcher_ {
        scaler: scaler.clone(),
    })
    .batch_size(128)
    .build(InMemDataset::new(data_valid.to_vec()));

    // Training
    let num_params = data_train[0].features_property.len();
    let model_cfg = ModelConfig {
        global_input_dim: num_params,
        gnn_hidden_dim: 64,
        mlp_hidden_dim: 128,
        vocab_size_elem: 12,           // matches vocab_lookup_element max + 1
        vocab_size_ff: FF_BUCKETS + 2, // 0 pad + 1..FF_BUCKETS + unknown.
        embedding_dim: 16,             // Tune this (8, 16, 32)
        n_node_scalars: 4,
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
                    println!("Eval results for {dataset}: {ev}");
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
