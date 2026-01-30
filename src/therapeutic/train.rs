#![allow(unused)] // Required to prevent false positives.

//! Entry point for training of therapeutic properties. (Via a thin wrapper in `/src/train.rs` required
//! by Rust's system)
//!
//! This is tailored towards data from Therapeutic Data Commons (TDC).

//! To run: `cargo r --release --features train --bin train -- --path C:/Users/the_a/Desktop/bio_misc/tdc_data/bbb_martins.csv`
//!
//! Add the `tgt` param if training on a single file. Can be a single target, or multiple.
//! --tgt bbb_martins`

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    hash::{Hash, Hasher},
    io,
    io::Write,
    path::Path,
    str::FromStr,
    time::Instant,
};

use bio_files::{Sdf, md_params::ForceFieldParams};
#[cfg(feature = "train")]
use burn::backend::Wgpu;
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
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::AdamConfig,
    prelude::Int,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Tensor, TensorData, activation,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
        TrainingStrategy, metric::LossMetric,
    },
};
use dynamics::params::FfParamSet;
use include_dir::{Dir, include_dir};
use na_seq::Element::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[cfg(feature = "train")]
use crate::therapeutic::model_eval::eval;
use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc, gnn,
        gnn::{GraphData, PER_ATOM_SCALARS, PER_EDGE_FEATS},
        train_test_split_indices::TrainTestSplit,
    },
};

// Number of buckets for Force Field types (hashing trick)
// 20 is usually enough to capture major distinct atom types without too many collisions. (?)
pub(in crate::therapeutic) const FF_BUCKETS: usize = 20;

// 10 (Elements) + 1 (Degree) + 20 (FF Hashed) + 1 (Partial Charge)
// pub(in crate::therapeutic) const FEAT_DIM_ATOMS: usize = 12 + FF_BUCKETS;

// todo: How should this be set up
pub(in crate::therapeutic) const MAX_ATOMS: usize = 100; // Max atoms for padding

pub(in crate::therapeutic) const MODEL_DIR: &str = "ml_models/models";

// We see the validation and training dirs separate from model_dir so `include_dir` doesn't place
// them in the executable.
pub(in crate::therapeutic) const TRAIN_VALID_DIR: &str = "ml_models";
pub(in crate::therapeutic) static MODEL_INCLUDE: Dir =
    include_dir!("$CARGO_MANIFEST_DIR/ml_models/models");

pub(in crate::therapeutic) const TGT_COL_TDC: usize = 2;

// Note: Excluding H or not appears not to make any notable difference at first;
// Experiment with this more later.
pub(in crate::therapeutic) const EXCLUDE_HYDROGEN: bool = true;

// Increasing layers may or may not improve model performance. It will slow down inference and training.
const NUM_GNN_LAYERS: usize = 3;
const NUM_MLP_LAYERS: usize = 3;
// It seems that low or no dropout significantly improves results, but perhaps it makes the
// model more likely to overfit, and makes it less general? Maybe 0.1 or disabled.
// Perhaps skip dropout due to our small TDC data sets.
const DROPOUT: Option<f64> = None;

pub(in crate::therapeutic) const BOND_SIGMA_SQ: f32 = 3.3; // Å. Try 1.5 - 2.2 for sigma, (Square it)

#[cfg(feature = "train")]
type TrainBackend = Autodiff<Wgpu>;
#[cfg(feature = "train")]
type ValidBackend = Wgpu;

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
    pub edge_feat_dim: usize,
}

impl ModelConfig {
    // pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
    //     let dim_gnn = self.gnn_hidden_dim;
    //     let dim_mlp = self.mlp_hidden_dim;
    //     let combined_dim = self.mlp_hidden_dim + self.gnn_hidden_dim;
    //
    //     let emb_elem = EmbeddingConfig::new(self.vocab_size_elem, self.embedding_dim).init(device);
    //     let emb_ff = EmbeddingConfig::new(self.vocab_size_ff, self.embedding_dim).init(device);
    //
    //     // The input to the GNN is now: embedding_dim + embedding_dim + scalar_features (degree, charge)
    //     let gnn_input_dim = self.embedding_dim * 2 + self.n_node_scalars;
    //
    //     let edge_encoder = LinearConfig::new(self.edge_feat_dim, dim_gnn).init(device);
    //
    //     // GNN
    //     let mut gnn_layers = Vec::with_capacity(NUM_GNN_LAYERS);
    //     for layer_i in 0..NUM_GNN_LAYERS {
    //         let (in_dim, out_dim) = if layer_i == 0 {
    //             (gnn_input_dim, dim_gnn)
    //         } else {
    //             (dim_gnn, dim_gnn)
    //         };
    //         gnn_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
    //     }
    //
    //     let edge_proj = LinearConfig::new(self.edge_feat_dim, 1).init(device);
    //
    //     // MLP
    //     let mut mlp_layers = Vec::with_capacity(NUM_MLP_LAYERS);
    //     for layer_i in 0..NUM_MLP_LAYERS {
    //         let (in_dim, out_dim) = if layer_i == 0 {
    //             (self.global_input_dim, dim_mlp)
    //         } else {
    //             (dim_mlp, dim_mlp)
    //         };
    //         mlp_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
    //     }
    //
    //     Model {
    //         emb_elem,
    //         emb_ff,
    //         edge_encoder,
    //         gnn_layers,
    //         edge_proj,
    //         mlp_layers,
    //         fusion_norm: LayerNormConfig::new(combined_dim).init(device),
    //         head: LinearConfig::new(combined_dim, 1).init(device),
    //         dropout: DropoutConfig::new(DROPOUT.unwrap_or_default()).init(),
    //     }
    // }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let dim_gnn = self.gnn_hidden_dim;
        let dim_mlp = self.mlp_hidden_dim;
        let combined_dim = self.mlp_hidden_dim + self.gnn_hidden_dim;

        let emb_elem = EmbeddingConfig::new(self.vocab_size_elem, self.embedding_dim).init(device);
        let emb_ff = EmbeddingConfig::new(self.vocab_size_ff, self.embedding_dim).init(device);

        let gnn_input_dim = self.embedding_dim * 2 + self.n_node_scalars;

        // [NEW] Project Nodes to match Hidden Dim (so we can add Edges to them)
        let node_encoder = LinearConfig::new(gnn_input_dim, dim_gnn).init(device);

        // [UNCHANGED] Project Edges to match Hidden Dim
        let edge_encoder = LinearConfig::new(self.edge_feat_dim, dim_gnn).init(device);

        // [CHANGED] All GNN layers now operate on dim_gnn -> dim_gnn
        let mut gnn_layers = Vec::with_capacity(NUM_GNN_LAYERS);
        for _ in 0..NUM_GNN_LAYERS {
            gnn_layers.push(LinearConfig::new(dim_gnn, dim_gnn).init(device));
        }

        let edge_proj = LinearConfig::new(self.edge_feat_dim, 1).init(device);

        // ... MLP Layers (Unchanged) ...
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
            node_encoder, // Add to struct
            edge_encoder,
            gnn_layers,
            edge_proj,
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
    node_encoder: Linear<B>,
    edge_encoder: Linear<B>,
    /// GNN layers: Broadly graph and per-atom data.
    gnn_layers: Vec<Linear<B>>,
    edge_proj: Linear<B>,
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
    /// GINE-style Layer:
    /// Aggregates neighbors, injecting edge features into the message.
    fn make_gnn_layer(
        &self,
        adj_weighted: &Tensor<B, 3>, // [Batch, N, N]
        mask: &Tensor<B, 3>,         // [Batch, N, 1]
        nodes: Tensor<B, 3>,         // [Batch, N, D_in]
        edge_emb: &Tensor<B, 4>,     // [Batch, N, N, D_hidden]
        gnn_linear: &Linear<B>,      // The update layer
        dropout: bool,
    ) -> Tensor<B, 3> {
        // 1. Broadcast Nodes to Neighbors: [B, N, D] -> [B, 1, N, D]
        let nodes_j = nodes.clone().unsqueeze_dim(1);

        // 2. Combine Node + Edge (GINE): [B, 1, N, D] + [B, N, N, D] -> [B, N, N, D]
        let message = activation::relu(nodes_j + edge_emb.clone());

        // 3. Aggregate: message * weights -> Sum(dim 2)
        let weights = adj_weighted.clone().unsqueeze_dim(3); // [B, N, N, 1]

        // sum_dim(2) produces [B, N, 1, D].
        // flatten(2, 3) merges the last two dims -> [B, N, D]
        let agg = (message * weights).sum_dim(2).flatten(2, 3);

        let mut layer_out = gnn_linear.forward(agg);

        if dropout && DROPOUT.is_some() {
            layer_out = self.dropout.forward(layer_out);
        }

        // 5. Residual Connection
        // If dims match (usually true except first layer), add residual.
        let out = if nodes.dims()[2] == layer_out.dims()[2] {
            layer_out + nodes
        } else {
            layer_out
        };

        out * mask.clone()
    }

    pub fn forward(
        &self,
        // These indexes are for mapping string values to integers for use in the model.
        elem_idx: Tensor<B, 2, Int>,
        ff_idx: Tensor<B, 2, Int>,
        scalars: Tensor<B, 3>,
        adj: Tensor<B, 3>,
        edge_feats: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        params: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, n, _n2, f] = edge_feats.dims();

        // 1. Edge Gating
        let ef_flat = edge_feats.clone().reshape([b * n * n, f]);
        let gate_flat = activation::sigmoid(self.edge_proj.forward(ef_flat.clone()));
        let gate = gate_flat.reshape([b, n, n]);
        let adj_eff = adj * gate;

        // Edge Embedding
        let edge_emb_flat = self.edge_encoder.forward(ef_flat);

        // Get d_hidden dynamically. `Linear` struct does not have public `d_output`.
        let [_, d_hidden] = edge_emb_flat.dims();
        let edge_emb = edge_emb_flat.reshape([b, n, n, d_hidden]);

        // 3. Prepare Nodes
        let x_elem = self.emb_elem.forward(elem_idx);
        let x_ff = self.emb_ff.forward(ff_idx);
        let raw_nodes = Tensor::cat(vec![x_elem, x_ff, scalars], 2);

        // Project Nodes to Hidden Dim
        let nodes = self.node_encoder.forward(raw_nodes);

        let mut gnn_prev = nodes;

        for (i, layer) in self.gnn_layers.iter().enumerate() {
            let dropout = i != self.gnn_layers.len() - 1;
            gnn_prev = self.make_gnn_layer(&adj_eff, &mask, gnn_prev, &edge_emb, layer, dropout);
        }

        let graph_sum = gnn_prev.sum_dim(1);
        let atom_counts = mask.sum_dim(1);
        let graph_mean = graph_sum / (atom_counts + 1e-6);
        let [b_g, _one, d] = graph_mean.dims();
        let graph_embedding = graph_mean.reshape([b_g, d]);

        let mut mlp_prev = params;
        for (i, layer) in self.mlp_layers.iter().enumerate() {
            mlp_prev = activation::relu(layer.forward(mlp_prev.clone()));
            if i != self.mlp_layers.len() - 1 && DROPOUT.is_some() {
                mlp_prev = self.dropout.forward(mlp_prev);
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
    pub features_property: Vec<f32>,
    pub graph: GraphData,
    pub target: f32,
}

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct Batch<B: Backend> {
    pub el_indices: Tensor<B, 2, Int>,
    pub ff_indices: Tensor<B, 2, Int>,
    /// Atom-specific properties, e.g. partial charge.
    pub scalars: Tensor<B, 3>,
    pub adj_list: Tensor<B, 3>,
    pub edge_feats: Tensor<B, 4>,
    pub mask: Tensor<B, 3>,
    pub mol_params: Tensor<B, 2>,
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
        let mut batch_edge_feats = Vec::new();
        let mut batch_mask = Vec::new();
        let mut batch_globals = Vec::new();
        let mut batch_y = Vec::new();

        let n_feat_params = items[0].features_property.len();

        // Calculate num scalars per atom based on first item
        let n_scalars_per_atom = if items[0].graph.num_atoms > 0 {
            items[0].graph.scalars.len() / items[0].graph.num_atoms
        } else {
            2
        };

        for mut item in items {
            // Mol parameters, and the target value.
            self.scaler.apply_in_place(&mut item.features_property);
            batch_globals.extend_from_slice(&item.features_property);
            batch_y.push(self.scaler.normalize_target(item.target));

            // Pad and Extend Graph Data
            let n = item.graph.num_atoms.min(MAX_ATOMS);

            batch_elem_ids.extend_from_slice(&item.graph.elem_indices[0..n]);
            batch_elem_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

            batch_ff_ids.extend_from_slice(&item.graph.ff_indices[0..n]);
            batch_ff_ids.extend(std::iter::repeat(0).take(MAX_ATOMS - n));

            batch_scalars.extend_from_slice(&item.graph.scalars[0..n * n_scalars_per_atom]);
            batch_scalars.extend(std::iter::repeat(0.0).take((MAX_ATOMS - n) * n_scalars_per_atom));

            // Adjacency list and  mask
            let (p_adj, p_mask) = gnn::pad_adj_and_mask(&item.graph.adj, item.graph.num_atoms);
            batch_adj.extend(p_adj);
            batch_mask.extend(p_mask);

            let p_edge = gnn::pad_edge_feats(&item.graph.edge_feats, item.graph.num_atoms);
            batch_edge_feats.extend(p_edge);
        }

        let elem_ids = TensorData::new(batch_elem_ids, [batch_size, MAX_ATOMS]);
        let ff_ids = TensorData::new(batch_ff_ids, [batch_size, MAX_ATOMS]);
        let scalars = TensorData::new(batch_scalars, [batch_size, MAX_ATOMS, n_scalars_per_atom]);
        let adj = TensorData::new(batch_adj, [batch_size, MAX_ATOMS, MAX_ATOMS]);
        let edge_feats = TensorData::new(
            batch_edge_feats,
            [batch_size, MAX_ATOMS, MAX_ATOMS, PER_EDGE_FEATS],
        );
        let mask = TensorData::new(batch_mask, [batch_size, MAX_ATOMS, 1]);
        let globals = TensorData::new(batch_globals, [batch_size, n_feat_params]);
        let y = TensorData::new(batch_y, [batch_size, 1]);

        Batch {
            el_indices: Tensor::from_data(elem_ids, device),
            ff_indices: Tensor::from_data(ff_ids, device),
            scalars: Tensor::from_data(scalars, device),
            adj_list: Tensor::from_data(adj, device),
            edge_feats: Tensor::from_data(edge_feats, device),
            mask: Tensor::from_data(mask, device),
            mol_params: Tensor::from_data(globals, device),
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

#[cfg(feature = "train")]
impl TrainStep for Model<TrainBackend> {
    type Input = Batch<TrainBackend>;
    type Output = RegressionOutput<TrainBackend>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pred = self.forward(
            batch.el_indices,
            batch.ff_indices,
            batch.scalars,
            batch.adj_list,
            batch.edge_feats,
            batch.mask,
            batch.mol_params,
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
            batch.el_indices,
            batch.ff_indices,
            batch.scalars,
            batch.adj_list,
            batch.edge_feats,
            batch.mask,
            batch.mol_params,
        );

        let loss = MseLoss::new().forward(pred.clone(), batch.targets.clone(), Reduction::Mean);

        RegressionOutput::new(loss, pred, batch.targets)
    }
}

/// Each field is the molecule, and target value.
#[derive(Clone, Debug, Default)]
pub(in crate::therapeutic) struct TrainingData {
    pub train: Vec<(MoleculeSmall, f32)>,
    pub validation: Vec<(MoleculeSmall, f32)>,
    pub test: Vec<(MoleculeSmall, f32)>,
}

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
    ff_params: &ForceFieldParams,
    test_only: bool,
) -> io::Result<TrainingData> {
    let csv_file = fs::File::open(csv_path)?;
    let mut rdr = csv::Reader::from_reader(csv_file);

    let mut result = TrainingData::default();

    // These Hash sets improve speed over  using the tts variables directly. (Double-nested loop)
    let train_set: HashSet<usize> = tts.train.iter().copied().collect();
    let validation_set: HashSet<usize> = tts.validation.iter().copied().collect();
    let test_set: HashSet<usize> = tts.test.iter().copied().collect();

    let mut record_count = 0;
    // Iterate over records (automatically handles quotes and headers)
    for (i, record) in rdr.records().enumerate() {
        record_count += 1;

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

        // Note: We are skipping populating mol-specific parameters. These are generally dihedrals,
        // but less commonly valence angles.
        // We are starting with bond-stretching params only in our model.
        mol.update_ff_related(mol_specific_param_set, ff_params, true);

        // We are experimenting with using our internally-derived characteristics
        // instead of those in the CSV; it may be more consistent.
        mol.update_characterization();

        if train_set.contains(&i) {
            result.train.push((mol, target));
        } else if validation_set.contains(&i) {
            result.validation.push((mol, target));
        } else if test_set.contains(&i) {
            result.test.push((mol, target));
        } else {
            eprintln!("Warning: Record {i} not present in the train/test split for set {filename}");
        }
    }

    assert_eq!(
        record_count,
        tts.train.len() + tts.validation.len() + tts.test.len()
    );

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
    ff_params: &ForceFieldParams,
) -> io::Result<(Vec<Sample>, Vec<Sample>)> {
    let loaded = load_training_data(
        csv_path,
        sdf_path,
        tgt_col,
        tts,
        mol_specific_param_set,
        ff_params,
        false,
    )?;

    let mut result_train = Vec::new();
    let mut result_test = Vec::new();

    for (result_set, data) in [
        (&mut result_train, &loaded.train),
        (&mut result_test, &loaded.validation),
    ] {
        for (mol, target) in data {
            let feat_params = param_feats_from_mol(&mol)?;

            if mol.common.bonds.is_empty() {
                // eprintln!("No bonds found in SDF. Skipping.");
                continue;
            }

            let graph = match gnn::mol_to_graph_data(&mol, ff_params) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("Error getting graph data: {:?}", e);
                    continue;
                }
            };

            // if num_atoms == 0 || num_atoms > MAX_ATOMS {
            //     continue;
            // }
            if graph.num_atoms == 0 {
                continue;
            }

            result_set.push(Sample {
                features_property: feat_params,
                graph,
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

    // ----

    // We are generally apply ln to values that can be "large".
    // Note: We do seem to get better results using ln values.

    // todo: Many of these are suspect.

    // Ring count: Pos
    // Function groups: Pos
    // Valence: Neg
    // c.rings.len() as f32 * 6. / c.num_atoms as f32: Pos
    // Ring count: Pos
    // Wiener index: Neg impact
    // Mol weight: neg impact
    // Num bonds: Positive impact
    // Rot bond count: Positive impact
    // ln(c.psa_topo / c.asa_topo): Pos
    // psa topo: Pos
    // SAS topo: Big pos
    // Num heavy: pos
    // Het: Pos
    // Halogen: Pos
    // Volume: Pos (big)

    // -----

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
        // // c.num_valence_elecs as f32,
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
        // ln(c.mol_weight),
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
        c.carboxylate.len() as f32,
        c.sulfonamide.len() as f32,
        c.sulfonimide.len() as f32,
        // c.num_valence_elecs as f32,
        // c.num_rings_aromatic as f32,
        // c.num_rings_saturated as f32,
        // c.num_rings_aliphatic as f32,
        c.rings.len() as f32,
        c.log_p,
        c.molar_refractivity,
        ln(c.psa_topo),
        ln(c.asa_topo),
        ln(c.volume),
        // ln(c.wiener_index.unwrap_or(0) as f32),
        c.rings.len() as f32 * 6. / c.num_atoms as f32, // todo temp
        ln(c.psa_topo / c.asa_topo),
        // ln(c.greasiness),
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

    let (model_path, scaler_path, config_path) = dataset.model_paths();

    let start = Instant::now();
    println!("Started training on {csv_path:?}");

    // let model_dir = Path::new(MODEL_DIR);
    let model_dir_train_val = Path::new(TRAIN_VALID_DIR);
    fs::create_dir_all(model_dir_train_val)?;

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

    // This seeding prevents random behavior.
    const SEED: u64 = 42;
    TrainBackend::seed(&device, SEED);
    ValidBackend::seed(&device, SEED);

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
        n_node_scalars: PER_ATOM_SCALARS,
        edge_feat_dim: PER_EDGE_FEATS,
    };

    let model = model_cfg.init::<TrainBackend>(&device);

    println!("Model parameter count: {}", model.num_params());

    let optim = AdamConfig::new().init();
    let lr_scheduler = ConstantLr::new(3e-4);

    let training = SupervisedTraining::new(
        model_dir_train_val.to_str().unwrap(),
        train_loader,
        valid_loader,
    )
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
    let ff_params = FfParamSet::new_amber().unwrap().small_mol.unwrap();
    let mol_specific_params = &mut HashMap::new();

    let datasets = match target {
        Some(t) => t
            .split_whitespace()
            .map(|set| DatasetTdc::from_str(&t).unwrap())
            .collect(),
        None => DatasetTdc::all(),
    };

    for dataset in datasets {
        if eval_ {
            match eval(
                data_path,
                dataset,
                TGT_COL_TDC,
                mol_specific_params,
                &ff_params,
            ) {
                Ok(ev) => {
                    println!("Eval results for {dataset}: {ev}");
                }
                Err(e) => {
                    eprintln!("Error evaluating {dataset}: {e}");
                }
            }
        } else {
            if let Err(e) = train(
                data_path,
                dataset,
                TGT_COL_TDC,
                mol_specific_params,
                &ff_params,
            ) {
                eprintln!("Error training {dataset}: {e}");
            }
        }
    }
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
