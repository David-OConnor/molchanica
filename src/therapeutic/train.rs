#![allow(unused)] // Required to prevent false positives.

//! Entry point for training of therapeutic properties. (Via a thin wrapper in `/src/train.rs` required
//! by Rust's system)
//!
//! This is tailored towards data from Therapeutic Data Commons (TDC).

//! To run: `cargo r --release --features train --bin train -- --path C:/Users/the_a/Desktop/bio_misc/tdc_data`
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
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
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
use crate::therapeutic::eval::eval;
use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc, gnn,
        gnn::{
            GraphData, GraphDataComponent, GraphDataSpacial, PER_ATOM_SCALARS, PER_COMP_SCALARS,
            PER_EDGE_COMP_FEATS, PER_EDGE_FEATS, PER_PHARM_SCALARS, PER_SPACIAL_EDGE_FEATS,
            PHARM_VOCAB_SIZE,
        },
        pharmacophore::Pharmacophore,
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
pub(in crate::therapeutic) const MAX_COMPS: usize = 30; // Max components for padding
pub(in crate::therapeutic) const MAX_PHARM: usize = 30; // Max pharmacophore nodes for padding

/// Configuation for the model; can be set at runtime, e.g. using the config file.
/// Loaded from `therapeutic_training_config.toml` at the project root.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(in crate::therapeutic) struct BranchConfig {
    pub gnn_atom_enabled: bool,
    pub gnn_comp_enabled: bool,
    pub gnn_spacial_enabled: bool,
    pub mlp_enabled: bool,
    pub gnn_atom_layers: u8,
    pub gnn_comp_layers: u8,
    pub gnn_spacial_layers: u8,
    pub mlp_layers: u8,
}

impl Default for BranchConfig {
    fn default() -> Self {
        Self {
            gnn_atom_enabled: true,
            gnn_comp_enabled: true,
            gnn_spacial_enabled: true,
            mlp_enabled: true,
            gnn_atom_layers: 3,
            gnn_comp_layers: 3,
            gnn_spacial_layers: 2,
            mlp_layers: 3,
        }
    }
}

/// Minimal parser for the subset of TOML used by `therapeutic_training_config.toml`:
/// top-level sections (`[name]`) containing `key = true/false` pairs.
/// Unknown keys are silently ignored; missing keys keep the default value.
#[cfg(feature = "train")]
fn load_branch_config(dataset_name: &str) -> BranchConfig {
    const CONFIG_PATH: &str = "therapeutic_training_config.toml";

    let text = match fs::read_to_string(CONFIG_PATH) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Warning: could not read {CONFIG_PATH}: {e}. Using defaults.");
            return BranchConfig::default();
        }
    };

    // Parse into HashMap<section, HashMap<key, String>>; handles both bool and integer values.
    let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            current = line[1..line.len() - 1].to_string();
        } else if let Some((k, v)) = line.split_once('=') {
            // Strip inline comments and whitespace.
            let v = v.split('#').next().unwrap_or("").trim().to_string();
            sections
                .entry(current.clone())
                .or_default()
                .insert(k.trim().to_string(), v);
        }
    }

    // Look up: specific dataset → [default] → first section in file → all-true fallback
    let (section_name, map) = if let Some(m) = sections.get(dataset_name) {
        (dataset_name.to_string(), Some(m))
    } else if let Some(m) = sections.get("default") {
        ("default".to_string(), Some(m))
    } else if let Some((name, m)) = sections.iter().next() {
        (name.clone(), Some(m))
    } else {
        ("(none — file empty or unparseable)".to_string(), None)
    };

    println!("  Branch config section used: [{section_name}]");

    let get = |map: Option<&HashMap<String, String>>, key: &str| -> bool {
        map.and_then(|m| m.get(key))
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(true)
    };

    let get_int = |map: Option<&HashMap<String, String>>, key: &str, default: u8| -> u8 {
        map.and_then(|m| m.get(key))
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(default)
    };

    BranchConfig {
        gnn_atom_enabled: get(map, "gnn_atom_enabled"),
        gnn_comp_enabled: get(map, "gnn_comp_enabled"),
        gnn_spacial_enabled: get(map, "gnn_spacial_enabled"),
        mlp_enabled: get(map, "mlp_enabled"),
        gnn_atom_layers: get_int(map, "gnn_atom_layers", 3),
        gnn_comp_layers: get_int(map, "gnn_comp_layers", 3),
        gnn_spacial_layers: get_int(map, "gnn_spacial_layers", 2),
        mlp_layers: get_int(map, "mlp_layers", 3),
    }
}

pub(in crate::therapeutic) const MODEL_DIR: &str = "ml_models/models";

pub(in crate::therapeutic) const TRAIN_VALID_DIR: &str = "ml_models";

// Only embed model files in the main app binary. The train binary always loads from disk,
// and embedding here would force a recompile every time a training run writes new model files.
#[cfg(not(feature = "train"))]
pub(in crate::therapeutic) static MODEL_INCLUDE: Dir =
    include_dir!("$CARGO_MANIFEST_DIR/ml_models/models");

pub(in crate::therapeutic) const TGT_COL_TDC: usize = 2;

// Note: Excluding H or not appears not to make any notable difference at first;
// Experiment with this more later.
pub(in crate::therapeutic) const EXCLUDE_HYDROGEN: bool = true;

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
    // These vocab and embeddings are used for per-atom GNN properties, e.g. element.
    pub vocab_size_elem: usize,
    pub vocab_size_ff: usize,
    pub embedding_dim: usize,
    /// E.g. 2, one for charge; one for degree, one for R, one for mean_nb_dist
    pub n_node_scalars: usize,
    pub edge_feat_dim: usize,
    // Component GNN fields
    pub vocab_size_comp: usize,
    pub comp_embedding_dim: usize,
    pub n_comp_scalars: usize,
    pub comp_edge_feat_dim: usize,
    // Spatial (pharmacophore) GNN fields
    pub vocab_size_pharm: usize,
    pub pharm_embedding_dim: usize,
    pub n_pharm_scalars: usize,
    pub spacial_edge_feat_dim: usize,
    // Which branches are active — saved so inference reloads correctly
    pub gnn_atom_enabled: bool,
    pub gnn_comp_enabled: bool,
    pub gnn_spacial_enabled: bool,
    pub mlp_enabled: bool,
    pub gnn_atom_layers: u8,
    pub gnn_comp_layers: u8,
    pub gnn_spacial_layers: u8,
    pub mlp_layers: u8,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let dim_gnn = self.gnn_hidden_dim;
        let dim_mlp = self.mlp_hidden_dim;

        // Each GNN branch contributes mean + max pooled vectors (hence * 2).
        let combined_dim = if self.mlp_enabled {
            self.mlp_hidden_dim
        } else {
            0
        } + if self.gnn_atom_enabled {
            self.gnn_hidden_dim * 2
        } else {
            0
        } + if self.gnn_comp_enabled {
            self.gnn_hidden_dim * 2
        } else {
            0
        } + if self.gnn_spacial_enabled {
            self.gnn_hidden_dim * 2
        } else {
            0
        };

        let emb_elem = EmbeddingConfig::new(self.vocab_size_elem, self.embedding_dim).init(device);
        let emb_ff = EmbeddingConfig::new(self.vocab_size_ff, self.embedding_dim).init(device);

        let gnn_input_dim = self.embedding_dim * 2 + self.n_node_scalars;

        // Project Nodes to match Hidden Dim (so we can add Edges to them)
        let node_encoder = LinearConfig::new(gnn_input_dim, dim_gnn).init(device);

        // Project Edges to match Hidden Dim
        let edge_encoder = LinearConfig::new(self.edge_feat_dim, dim_gnn).init(device);

        // All GNN layers now operate on dim_gnn -> dim_gnn
        let mut gnn_atom_layers = Vec::with_capacity(self.gnn_atom_layers as usize);
        for _ in 0..self.gnn_atom_layers {
            gnn_atom_layers.push(LinearConfig::new(dim_gnn, dim_gnn).init(device));
        }

        let edge_proj = LinearConfig::new(self.edge_feat_dim, 1).init(device);

        // Component GNN
        let emb_comp =
            EmbeddingConfig::new(self.vocab_size_comp, self.comp_embedding_dim).init(device);
        let comp_gnn_input_dim = self.comp_embedding_dim + self.n_comp_scalars;
        let comp_node_encoder = LinearConfig::new(comp_gnn_input_dim, dim_gnn).init(device);
        let comp_edge_encoder = LinearConfig::new(self.comp_edge_feat_dim, dim_gnn).init(device);

        let mut gnn_comp_layers = Vec::with_capacity(self.gnn_comp_layers as usize);
        for _ in 0..self.gnn_comp_layers {
            gnn_comp_layers.push(LinearConfig::new(dim_gnn, dim_gnn).init(device));
        }

        let comp_edge_proj = LinearConfig::new(self.comp_edge_feat_dim, 1).init(device);

        // Spatial (pharmacophore) GNN
        let emb_pharm =
            EmbeddingConfig::new(self.vocab_size_pharm, self.pharm_embedding_dim).init(device);
        let pharm_gnn_input_dim = self.pharm_embedding_dim + self.n_pharm_scalars;
        let pharm_node_encoder = LinearConfig::new(pharm_gnn_input_dim, dim_gnn).init(device);
        let pharm_edge_encoder =
            LinearConfig::new(self.spacial_edge_feat_dim, dim_gnn).init(device);

        let mut spacial_gnn_layers = Vec::with_capacity(self.gnn_spacial_layers as usize);
        for _ in 0..self.gnn_spacial_layers {
            spacial_gnn_layers.push(LinearConfig::new(dim_gnn, dim_gnn).init(device));
        }

        let pharm_edge_proj = LinearConfig::new(self.spacial_edge_feat_dim, 1).init(device);

        // MLP Layers
        let mut mlp_layers = Vec::with_capacity(self.mlp_layers as usize);
        for layer_i in 0..self.mlp_layers {
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
            node_encoder,
            edge_encoder,
            gnn_layers: gnn_atom_layers,
            edge_proj,
            emb_comp,
            comp_node_encoder,
            comp_edge_encoder,
            comp_gnn_layers: gnn_comp_layers,
            comp_edge_proj,
            emb_pharm,
            pharm_node_encoder,
            pharm_edge_encoder,
            spacial_gnn_layers,
            pharm_edge_proj,
            mlp_layers,
            fusion_norm: LayerNormConfig::new(combined_dim).init(device),
            head: LinearConfig::new(combined_dim, 1).init(device),
            dropout: DropoutConfig::new(DROPOUT.unwrap_or_default()).init(),
            atom_gnn_enabled: self.gnn_atom_enabled,
            comp_gnn_enabled: self.gnn_comp_enabled,
            spacial_gnn_enabled: self.gnn_spacial_enabled,
            mlp_enabled: self.mlp_enabled,
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
    /// Component GNN branch
    emb_comp: Embedding<B>,
    comp_node_encoder: Linear<B>,
    comp_edge_encoder: Linear<B>,
    comp_gnn_layers: Vec<Linear<B>>,
    comp_edge_proj: Linear<B>,
    /// Spatial (pharmacophore) GNN branch. Nodes = pharmacophore features;
    /// edges encode Euclidean distances via Gaussian RBF.
    emb_pharm: Embedding<B>,
    pharm_node_encoder: Linear<B>,
    pharm_edge_encoder: Linear<B>,
    spacial_gnn_layers: Vec<Linear<B>>,
    pharm_edge_proj: Linear<B>,
    /// Parameter features. These are for molecule-level parameters. (Atom count, weight, volume, PSA etc)
    mlp_layers: Vec<Linear<B>>,
    fusion_norm: LayerNorm<B>,
    /// Joint Branch
    head: Linear<B>,
    /// This dropout is useful if using more than 3 GNN layers.
    dropout: Dropout,
    // Active branches — not trained weights, restored from ModelConfig JSON on load.
    #[module(skip)]
    atom_gnn_enabled: bool,
    #[module(skip)]
    comp_gnn_enabled: bool,
    #[module(skip)]
    spacial_gnn_enabled: bool,
    #[module(skip)]
    mlp_enabled: bool,
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

        let mut layer_out = activation::relu(gnn_linear.forward(agg));

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
        // Component graph
        comp_idx: Tensor<B, 2, Int>,
        comp_scalars: Tensor<B, 3>,
        comp_adj: Tensor<B, 3>,
        comp_edge_feats: Tensor<B, 4>,
        comp_mask: Tensor<B, 3>,
        // Spatial (pharmacophore) graph
        pharm_idx: Tensor<B, 2, Int>,
        pharm_scalars: Tensor<B, 3>,
        pharm_adj: Tensor<B, 3>,
        pharm_edge_feats: Tensor<B, 4>,
        pharm_mask: Tensor<B, 3>,
        params: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut branches: Vec<Tensor<B, 2>> = Vec::new();

        // --- Atom GNN branch ---
        if self.atom_gnn_enabled {
            let [b, n, _n2, f] = edge_feats.dims();

            let ef_flat = edge_feats.clone().reshape([b * n * n, f]);
            let gate_flat = activation::sigmoid(self.edge_proj.forward(ef_flat.clone()));
            let gate = gate_flat.reshape([b, n, n]);
            let adj_eff = adj * gate;

            let edge_emb_flat = self.edge_encoder.forward(ef_flat);
            let [_, d_hidden] = edge_emb_flat.dims();
            let edge_emb = edge_emb_flat.reshape([b, n, n, d_hidden]);

            let x_elem = self.emb_elem.forward(elem_idx);
            let x_ff = self.emb_ff.forward(ff_idx);
            let raw_nodes = Tensor::cat(vec![x_elem, x_ff, scalars], 2);
            let nodes = self.node_encoder.forward(raw_nodes);

            let mut gnn_prev = nodes;
            for (i, layer) in self.gnn_layers.iter().enumerate() {
                let dropout = i != self.gnn_layers.len() - 1;
                gnn_prev =
                    self.make_gnn_layer(&adj_eff, &mask, gnn_prev, &edge_emb, layer, dropout);
            }

            let graph_sum = gnn_prev.clone().sum_dim(1);
            let atom_counts = mask.clone().sum_dim(1);
            let graph_mean = graph_sum / (atom_counts.clone() + 1e-6);
            // Max pooling: set padding positions to -inf so they are never selected.
            let fill = (mask - 1.0) * 1e9;
            let graph_max = (gnn_prev + fill).max_dim(1);
            let has_nodes = atom_counts.clamp(0.0, 1.0);
            let graph_max = graph_max * has_nodes;
            let [b_g, _one, d] = graph_mean.dims();
            branches.push(Tensor::cat(
                vec![graph_mean.reshape([b_g, d]), graph_max.reshape([b_g, d])],
                1,
            ));
        }

        // --- Component GNN branch ---
        if self.comp_gnn_enabled {
            let [bc, nc, _nc2, fc] = comp_edge_feats.dims();

            let cef_flat = comp_edge_feats.clone().reshape([bc * nc * nc, fc]);
            let comp_gate_flat = activation::sigmoid(self.comp_edge_proj.forward(cef_flat.clone()));
            let comp_gate = comp_gate_flat.reshape([bc, nc, nc]);
            let comp_adj_eff = comp_adj * comp_gate;

            let comp_edge_emb_flat = self.comp_edge_encoder.forward(cef_flat);
            let [_, d_comp_hidden] = comp_edge_emb_flat.dims();
            let comp_edge_emb = comp_edge_emb_flat.reshape([bc, nc, nc, d_comp_hidden]);

            let x_comp = self.emb_comp.forward(comp_idx);
            let raw_comp_nodes = Tensor::cat(vec![x_comp, comp_scalars], 2);
            let comp_nodes = self.comp_node_encoder.forward(raw_comp_nodes);

            let mut comp_gnn_prev = comp_nodes;
            for (i, layer) in self.comp_gnn_layers.iter().enumerate() {
                let dropout = i != self.comp_gnn_layers.len() - 1;
                comp_gnn_prev = self.make_gnn_layer(
                    &comp_adj_eff,
                    &comp_mask,
                    comp_gnn_prev,
                    &comp_edge_emb,
                    layer,
                    dropout,
                );
            }

            let comp_sum = comp_gnn_prev.clone().sum_dim(1);
            let comp_counts = comp_mask.clone().sum_dim(1);
            let comp_mean = comp_sum / (comp_counts.clone() + 1e-6);
            let comp_fill = (comp_mask - 1.0) * 1e9;
            let comp_max = (comp_gnn_prev + comp_fill).max_dim(1);
            let has_comp_nodes = comp_counts.clamp(0.0, 1.0);
            let comp_max = comp_max * has_comp_nodes;
            let [b_c, _one_c, d_c] = comp_mean.dims();
            branches.push(Tensor::cat(
                vec![comp_mean.reshape([b_c, d_c]), comp_max.reshape([b_c, d_c])],
                1,
            ));
        }

        // --- Spatial (pharmacophore) GNN branch ---
        // Nodes = pharmacophore sites (H-bond donors/acceptors, hydrophobics, ring centres).
        // Edge features are distance-based RBF encodings; the adjacency is Gaussian-weighted.
        if self.spacial_gnn_enabled {
            let [bp, np, _np2, fp] = pharm_edge_feats.dims();

            let pef_flat = pharm_edge_feats.clone().reshape([bp * np * np, fp]);
            let pharm_gate_flat =
                activation::sigmoid(self.pharm_edge_proj.forward(pef_flat.clone()));
            let pharm_gate = pharm_gate_flat.reshape([bp, np, np]);
            let pharm_adj_eff = pharm_adj * pharm_gate;

            let pharm_edge_emb_flat = self.pharm_edge_encoder.forward(pef_flat);
            let [_, d_pharm_hidden] = pharm_edge_emb_flat.dims();
            let pharm_edge_emb = pharm_edge_emb_flat.reshape([bp, np, np, d_pharm_hidden]);

            let x_pharm = self.emb_pharm.forward(pharm_idx);
            let raw_pharm_nodes = Tensor::cat(vec![x_pharm, pharm_scalars], 2);
            let pharm_nodes = self.pharm_node_encoder.forward(raw_pharm_nodes);

            let mut spacial_gnn_prev = pharm_nodes;
            for (i, layer) in self.spacial_gnn_layers.iter().enumerate() {
                let dropout = i != self.spacial_gnn_layers.len() - 1;
                spacial_gnn_prev = self.make_gnn_layer(
                    &pharm_adj_eff,
                    &pharm_mask,
                    spacial_gnn_prev,
                    &pharm_edge_emb,
                    layer,
                    dropout,
                );
            }

            let pharm_sum = spacial_gnn_prev.clone().sum_dim(1);
            let pharm_counts = pharm_mask.clone().sum_dim(1);
            let pharm_mean = pharm_sum / (pharm_counts.clone() + 1e-6);
            let pharm_fill = (pharm_mask - 1.0) * 1e9;
            let pharm_max = (spacial_gnn_prev + pharm_fill).max_dim(1);
            let has_pharm_nodes = pharm_counts.clamp(0.0, 1.0);
            let pharm_max = pharm_max * has_pharm_nodes;
            let [b_p, _one_p, d_p] = pharm_mean.dims();
            branches.push(Tensor::cat(
                vec![
                    pharm_mean.reshape([b_p, d_p]),
                    pharm_max.reshape([b_p, d_p]),
                ],
                1,
            ));
        }

        // --- MLP branch ---
        if self.mlp_enabled {
            let mut mlp_prev = params;
            for (i, layer) in self.mlp_layers.iter().enumerate() {
                mlp_prev = layer.forward(mlp_prev.clone());
                // Skip activation on the last layer so the MLP output is unconstrained
                // going into the fusion head.
                if i != self.mlp_layers.len() - 1 {
                    mlp_prev = activation::relu(mlp_prev);
                    if DROPOUT.is_some() {
                        mlp_prev = self.dropout.forward(mlp_prev);
                    }
                }
            }
            branches.push(mlp_prev);
        }

        let combined = Tensor::cat(branches, 1);
        let combined = self.fusion_norm.forward(combined);

        self.head.forward(combined)
    }
}

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct Sample {
    /// From computed properties of the molecule.
    pub features_property: Vec<f32>,
    pub graph: GraphData,
    pub graph_comp: GraphDataComponent,
    pub graph_spacial: GraphDataSpacial,
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
    // Component graph
    pub comp_indices: Tensor<B, 2, Int>,
    pub comp_scalars: Tensor<B, 3>,
    pub comp_adj_list: Tensor<B, 3>,
    pub comp_edge_feats: Tensor<B, 4>,
    pub comp_mask: Tensor<B, 3>,
    // Spatial (pharmacophore) graph
    pub pharm_indices: Tensor<B, 2, Int>,
    pub pharm_scalars: Tensor<B, 3>,
    pub pharm_adj_list: Tensor<B, 3>,
    pub pharm_edge_feats: Tensor<B, 4>,
    pub pharm_mask: Tensor<B, 3>,
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
        let mut batch_comp_ids = Vec::new();
        let mut batch_comp_scalars = Vec::new();
        let mut batch_comp_adj = Vec::new();
        let mut batch_comp_edge_feats = Vec::new();
        let mut batch_comp_mask = Vec::new();
        let mut batch_pharm_ids = Vec::new();
        let mut batch_pharm_scalars = Vec::new();
        let mut batch_pharm_adj = Vec::new();
        let mut batch_pharm_edge_feats = Vec::new();
        let mut batch_pharm_mask = Vec::new();
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

            // Pad and Extend Atom Graph Data
            let n = item.graph.num_atoms.min(MAX_ATOMS);

            batch_elem_ids.extend_from_slice(&item.graph.elem_indices[0..n]);
            batch_elem_ids.extend(std::iter::repeat_n(0, MAX_ATOMS - n));

            batch_ff_ids.extend_from_slice(&item.graph.ff_indices[0..n]);
            batch_ff_ids.extend(std::iter::repeat_n(0, MAX_ATOMS - n));

            batch_scalars.extend_from_slice(&item.graph.scalars[0..n * n_scalars_per_atom]);
            batch_scalars.extend(std::iter::repeat_n(
                0.0,
                (MAX_ATOMS - n) * n_scalars_per_atom,
            ));

            let (p_adj, p_mask) =
                gnn::pad_adj_and_mask(&item.graph.adj, item.graph.num_atoms, MAX_ATOMS);
            batch_adj.extend(p_adj);
            batch_mask.extend(p_mask);

            let p_edge = gnn::pad_edge_feats(
                &item.graph.edge_feats,
                item.graph.num_atoms,
                PER_EDGE_FEATS,
                MAX_ATOMS,
            );
            batch_edge_feats.extend(p_edge);

            // Pad and Extend Component Graph Data
            let n_comps = item.graph_comp.num_comps.min(MAX_COMPS);

            batch_comp_ids.extend_from_slice(&item.graph_comp.comp_type_indices[0..n_comps]);
            batch_comp_ids.extend(std::iter::repeat_n(0_i32, MAX_COMPS - n_comps));

            batch_comp_scalars
                .extend_from_slice(&item.graph_comp.scalars[0..n_comps * PER_COMP_SCALARS]);
            batch_comp_scalars.extend(std::iter::repeat_n(
                0.0_f32,
                (MAX_COMPS - n_comps) * PER_COMP_SCALARS,
            ));

            let (p_comp_adj, p_comp_mask) =
                gnn::pad_adj_and_mask(&item.graph_comp.adj, item.graph_comp.num_comps, MAX_COMPS);
            batch_comp_adj.extend(p_comp_adj);
            batch_comp_mask.extend(p_comp_mask);

            let p_comp_edge = gnn::pad_edge_feats(
                &item.graph_comp.edge_feats,
                item.graph_comp.num_comps,
                PER_EDGE_COMP_FEATS,
                MAX_COMPS,
            );
            batch_comp_edge_feats.extend(p_comp_edge);

            // Pad and Extend Spatial (Pharmacophore) Graph Data
            let n_pharm = item.graph_spacial.num_nodes.min(MAX_PHARM);

            batch_pharm_ids.extend_from_slice(&item.graph_spacial.pharm_type_indices[0..n_pharm]);
            batch_pharm_ids.extend(std::iter::repeat_n(0_i32, MAX_PHARM - n_pharm));

            batch_pharm_scalars
                .extend_from_slice(&item.graph_spacial.scalars[0..n_pharm * PER_PHARM_SCALARS]);
            batch_pharm_scalars.extend(std::iter::repeat_n(
                0.0_f32,
                (MAX_PHARM - n_pharm) * PER_PHARM_SCALARS,
            ));

            let (p_pharm_adj, p_pharm_mask) = gnn::pad_adj_and_mask(
                &item.graph_spacial.adj,
                item.graph_spacial.num_nodes,
                MAX_PHARM,
            );
            batch_pharm_adj.extend(p_pharm_adj);
            batch_pharm_mask.extend(p_pharm_mask);

            let p_pharm_edge = gnn::pad_edge_feats(
                &item.graph_spacial.edge_feats,
                item.graph_spacial.num_nodes,
                PER_SPACIAL_EDGE_FEATS,
                MAX_PHARM,
            );
            batch_pharm_edge_feats.extend(p_pharm_edge);
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
        let comp_ids = TensorData::new(batch_comp_ids, [batch_size, MAX_COMPS]);
        let comp_scalars = TensorData::new(
            batch_comp_scalars,
            [batch_size, MAX_COMPS, PER_COMP_SCALARS],
        );
        let comp_adj = TensorData::new(batch_comp_adj, [batch_size, MAX_COMPS, MAX_COMPS]);
        let comp_edge_feats = TensorData::new(
            batch_comp_edge_feats,
            [batch_size, MAX_COMPS, MAX_COMPS, PER_EDGE_COMP_FEATS],
        );
        let comp_mask = TensorData::new(batch_comp_mask, [batch_size, MAX_COMPS, 1]);
        let pharm_ids = TensorData::new(batch_pharm_ids, [batch_size, MAX_PHARM]);
        let pharm_scalars = TensorData::new(
            batch_pharm_scalars,
            [batch_size, MAX_PHARM, PER_PHARM_SCALARS],
        );
        let pharm_adj = TensorData::new(batch_pharm_adj, [batch_size, MAX_PHARM, MAX_PHARM]);
        let pharm_edge_feats = TensorData::new(
            batch_pharm_edge_feats,
            [batch_size, MAX_PHARM, MAX_PHARM, PER_SPACIAL_EDGE_FEATS],
        );
        let pharm_mask = TensorData::new(batch_pharm_mask, [batch_size, MAX_PHARM, 1]);
        let globals = TensorData::new(batch_globals, [batch_size, n_feat_params]);
        let y = TensorData::new(batch_y, [batch_size, 1]);

        Batch {
            el_indices: Tensor::from_data(elem_ids, device),
            ff_indices: Tensor::from_data(ff_ids, device),
            scalars: Tensor::from_data(scalars, device),
            adj_list: Tensor::from_data(adj, device),
            edge_feats: Tensor::from_data(edge_feats, device),
            mask: Tensor::from_data(mask, device),
            comp_indices: Tensor::from_data(comp_ids, device),
            comp_scalars: Tensor::from_data(comp_scalars, device),
            comp_adj_list: Tensor::from_data(comp_adj, device),
            comp_edge_feats: Tensor::from_data(comp_edge_feats, device),
            comp_mask: Tensor::from_data(comp_mask, device),
            pharm_indices: Tensor::from_data(pharm_ids, device),
            pharm_scalars: Tensor::from_data(pharm_scalars, device),
            pharm_adj_list: Tensor::from_data(pharm_adj, device),
            pharm_edge_feats: Tensor::from_data(pharm_edge_feats, device),
            pharm_mask: Tensor::from_data(pharm_mask, device),
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

    pub fn apply_in_place(&self, x: &mut [f32]) {
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
            batch.comp_indices,
            batch.comp_scalars,
            batch.comp_adj_list,
            batch.comp_edge_feats,
            batch.comp_mask,
            batch.pharm_indices,
            batch.pharm_scalars,
            batch.pharm_adj_list,
            batch.pharm_edge_feats,
            batch.pharm_mask,
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
        let pred = self.forward(
            batch.el_indices,
            batch.ff_indices,
            batch.scalars,
            batch.adj_list,
            batch.edge_feats,
            batch.mask,
            batch.comp_indices,
            batch.comp_scalars,
            batch.comp_adj_list,
            batch.comp_edge_feats,
            batch.comp_mask,
            batch.pharm_indices,
            batch.pharm_scalars,
            batch.pharm_adj_list,
            batch.pharm_edge_feats,
            batch.pharm_mask,
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

            match sdf.clone().try_into() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error loading SDF; skipping mol at {sdf_path:?}: {e:?}");
                    continue;
                }
            }
        };

        // Note: We are skipping populating mol-specific parameters. These are generally dihedrals,
        // but less commonly valence angles.
        // We are starting with bond-stretching params only in our model.
        mol.update_ff_related(mol_specific_param_set, ff_params, true);

        // We are experimenting with using our internally-derived characteristics
        // instead of those in the CSV; it may be more consistent.
        mol.update_characterization(); // also builds mol.components

        mol.pharmacophore = Pharmacophore::new_all_candidates(&mol);

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

    if record_count != tts.train.len() + tts.validation.len() + tts.test.len() {
        eprintln!(
            "\n\n Error: Train/test/split for {csv_path:?} counts do not match record count.\n\
        records: {record_count}, \ntrain: {}\nvalid: {}\ntest:{}",
            tts.train.len(),
            tts.validation.len(),
            tts.test.len()
        );
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
            let feat_params = mlp_feats_from_mol(&mol)?;

            if mol.common.bonds.is_empty() {
                // eprintln!("No bonds found in SDF. Skipping.");
                continue;
            }

            let graph = match GraphData::new(&mol, ff_params) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("Error getting graph data: {:?}", e);
                    continue;
                }
            };

            if graph.num_atoms == 0 {
                continue;
            }

            let graph_comp = match &mol.components {
                Some(comps) => match GraphDataComponent::new(comps) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("Error getting comp graph data: {e:?}");
                        continue;
                    }
                },
                None => {
                    eprintln!("Missing components for mol; skipping.");
                    continue;
                }
            };

            let graph_spacial = match GraphDataSpacial::new(&mol) {
                Ok(g) => g,
                Err(e) => {
                    // Should not happen: new() returns Ok(empty) for missing/no-feature cases.
                    eprintln!("Unexpected error building spatial graph: {e:?}; skipping sample.");
                    continue;
                }
            };

            result_set.push(Sample {
                features_property: feat_params,
                graph,
                graph_comp,
                graph_spacial,
                target: *target,
            });
        }
    }
    Ok((result_train, result_test))
}

// Note: We can make variants of this A/R tuned to specific inference items. For now, we are using
// a single  set of features for all targets.
/// Extract  molecule-level features from a molecule that are relevant for inferring the target parameter. We use this
/// in both training and inference workflows.
///
/// We avoid features that may be more robustly represented by GNNs. For example, the count of rings,
/// functional groups, and H bond donors/acceptors.
pub(in crate::therapeutic) fn mlp_feats_from_mol(mol: &MoleculeSmall) -> io::Result<Vec<f32>> {
    let Some(c) = &mol.characterization else {
        return Err(io::Error::other("Missing mol characterization"));
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
        // c.h_bond_acceptor.len() as f32,
        // c.h_bond_donor.len() as f32,
        c.num_hetero_atoms as f32,
        c.halogen.len() as f32,
        c.rotatable_bonds.len() as f32,
        // c.amines.len() as f32,
        // c.amides.len() as f32,
        // c.carbonyl.len() as f32,
        // c.hydroxyl.len() as f32,
        // c.carboxylate.len() as f32,
        // c.sulfonamide.len() as f32,
        // c.sulfonimide.len() as f32,
        // c.num_valence_elecs as f32,
        // c.num_rings_aromatic as f32,
        // c.num_rings_saturated as f32,
        // c.num_rings_aliphatic as f32,
        // c.rings.len() as f32,
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

    let branch_cfg = load_branch_config(&dataset.name());
    println!(
        "Branch config for '{}': atom_gnn={} comp_gnn={} spacial_gnn={} mlp={}",
        dataset.name(),
        branch_cfg.gnn_atom_enabled,
        branch_cfg.gnn_comp_enabled,
        branch_cfg.gnn_spacial_enabled,
        branch_cfg.mlp_enabled,
    );

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
        vocab_size_comp: 10, // 10 component types (see vocab_lookup_component)
        comp_embedding_dim: 8,
        n_comp_scalars: PER_COMP_SCALARS,
        comp_edge_feat_dim: PER_EDGE_COMP_FEATS,
        // Spatial (pharmacophore) GNN
        vocab_size_pharm: PHARM_VOCAB_SIZE, // 0=pad,1=donor,2=acc,3=hydrophobic,4=aromatic
        pharm_embedding_dim: 8,
        n_pharm_scalars: PER_PHARM_SCALARS,
        spacial_edge_feat_dim: PER_SPACIAL_EDGE_FEATS,
        // Read which branches to enable from the config file.
        gnn_atom_enabled: branch_cfg.gnn_atom_enabled,
        gnn_comp_enabled: branch_cfg.gnn_comp_enabled,
        gnn_spacial_enabled: branch_cfg.gnn_spacial_enabled,
        mlp_enabled: branch_cfg.mlp_enabled,
        gnn_atom_layers: branch_cfg.gnn_atom_layers,
        gnn_comp_layers: branch_cfg.gnn_comp_layers,
        gnn_spacial_layers: branch_cfg.gnn_spacial_layers,
        mlp_layers: branch_cfg.mlp_layers,
    };

    let model = model_cfg.init::<TrainBackend>(&device);

    println!("Model parameter count: {}", model.num_params());

    let optim = AdamConfig::new().init();
    // Cosine annealing from 3e-4 → 1e-5 over the full training run.
    let num_iters = ((data_train.len() + 127) / 128) * 80; // steps_per_epoch * num_epochs
    let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(3e-4, num_iters)
        .with_min_lr(1e-5)
        .init()
        .unwrap();

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
                    println!("\nEval results for {dataset}: {ev}");
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
