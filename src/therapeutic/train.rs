#![allow(unused)] // Required to prevent false positives.

//! Entry point for training of therapeutic properties. (Via a thin wrapper in `/src/train.rs` required
//! by Rust's system)
//!
//! This is tailored towards data from Therapeutic Data Commons (TDC).

//! To run: `cargo r --release --features train --bin train -- --path C:/Users/the_a/Desktop/bio_misc/tdc_data`
//!
//! Add the `tgt` param if training on a single file. Can be a single target, or multiple.
//! --tgt bbb_martins`
//!
//! Add the `--eval` tag to evaluate

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    hash::Hasher,
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
    screening::pharmacophore::Pharmacophore,
    therapeutic::{
        DatasetTdc, gnn,
        gnn::{
            ATOM_GNN_EDGE_LAYERS, ATOM_GNN_PER_EDGE_FEATS_LAYER_0, GRAPH_ANALYSIS_FEATURE_VERSION,
            PER_ATOM_SCALARS, PER_COMP_SCALARS, PER_EDGE_COMP_FEATS, PER_PHARM_SCALARS,
            PER_SPACIAL_EDGE_FEATS, SPACIAL_VOCAB_SIZE, atom_bond, atom_bond::GraphDataAtom,
            component::GraphDataComponent, spacial::GraphDataSpacial,
        },
        mlp, non_nn_ml,
        non_nn_ml::GnnAnalysisTools,
        train_test_split_indices::TrainTestSplit,
    },
};

// Number of buckets for Force Field types (hashing trick)
// 20 is usually enough to capture major distinct atom types without too many collisions. (?)
pub(in crate::therapeutic) const FF_BUCKETS: usize = 20;

// Note: Excluding H or not appears not to make any notable difference at first;
// Experiment with this more later.
pub(in crate::therapeutic) const EXCLUDE_HYDROGEN: bool = true;

// 10 (Elements) + 1 (Degree) + 20 (FF Hashed) + 1 (Partial Charge)
// pub(in crate::therapeutic) const FEAT_DIM_ATOMS: usize = 12 + FF_BUCKETS;

// todo: How should this be set up
pub(in crate::therapeutic) const MAX_ATOMS: usize = 100; // Max atoms for padding
pub(in crate::therapeutic) const MAX_COMPS: usize = 30; // Max components for padding
pub(in crate::therapeutic) const MAX_PHARM: usize = 30; // Max pharmacophore nodes for padding

/// Configuration, as loaded from our config file; can be set at runtime.
/// Loaded from `therapeutic_training_config.toml` at the project root. See that file for
/// field descriptions, and default values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(in crate::therapeutic) struct ParamConfig {
    pub gnn_atom_enabled: bool,
    pub gnn_comp_enabled: bool,
    pub gnn_spacial_enabled: bool,
    pub mlp_enabled: bool,
    pub gnn_atom_layers: u8,
    pub gnn_comp_layers: u8,
    pub gnn_spacial_layers: u8,
    pub mlp_layers: u8,
    pub atom_graph_analysis_weisfeiler_lehman: bool,
    pub atom_graph_analysis_graphlets: Option<Vec<u8>>,
    pub atom_graph_analysis_path_based_methods: bool,
    pub atom_graph_analysis_local_overlap_statistics: bool,
    pub atom_graph_analysis_katz_index: bool,
    pub atom_graph_analysis_lhn_similarity: bool,
    pub atom_graph_analysis_random_walk_methods: bool,
    pub comp_graph_analysis_weisfeiler_lehman: bool,
    pub comp_graph_analysis_graphlets: Option<Vec<u8>>,
    pub comp_graph_analysis_path_based_methods: bool,
    pub comp_graph_analysis_local_overlap_statistics: bool,
    pub comp_graph_analysis_katz_index: bool,
    pub comp_graph_analysis_lhn_similarity: bool,
    pub comp_graph_analysis_random_walk_methods: bool,
    pub spacial_graph_analysis_weisfeiler_lehman: bool,
    pub spacial_graph_analysis_graphlets: Option<Vec<u8>>,
    pub spacial_graph_analysis_path_based_methods: bool,
    pub spacial_graph_analysis_local_overlap_statistics: bool,
    pub spacial_graph_analysis_katz_index: bool,
    pub spacial_graph_analysis_lhn_similarity: bool,
    pub spacial_graph_analysis_random_walk_methods: bool,
    // pub exclude_hydrogens: bool,
}

// impl Default for ParamConfig {
//     fn default() -> Self {
//         Self {
//             gnn_atom_enabled: true,
//             gnn_comp_enabled: true,
//             gnn_spacial_enabled: true,
//             mlp_enabled: true,
//             gnn_atom_layers: 3,
//             gnn_comp_layers: 3,
//             gnn_spacial_layers: 2,
//             mlp_layers: 3,
//             // exclude_hydrogens: true,
//         }
//     }
// }

/// Minimal parser for the subset of TOML used by `therapeutic_training_config.toml`:
/// top-level sections (`[name]`) containing `key = bool/int/[int, ...]` pairs.
/// Unknown keys are silently ignored; missing keys keep the default value.
pub(in crate::therapeutic) fn load_param_cfg(dataset_name: &str) -> io::Result<ParamConfig> {
    const CONFIG_PATH: &str = "therapeutic_training_config.toml";

    let text = fs::read_to_string(CONFIG_PATH)?;

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

    let default_atom_graph_analysis = non_nn_ml::atom_graph_analysis_tools();
    let default_comp_graph_analysis = non_nn_ml::component_graph_analysis_tools();
    let default_spacial_graph_analysis = non_nn_ml::spacial_graph_analysis_tools();

    let get_bool = |map: Option<&HashMap<String, String>>, key: &str, default: bool| -> bool {
        map.and_then(|m| m.get(key))
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(default)
    };

    let get_int = |map: Option<&HashMap<String, String>>, key: &str, default: u8| -> u8 {
        map.and_then(|m| m.get(key))
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(default)
    };

    let get_u8_list_opt = |map: Option<&HashMap<String, String>>,
                           key: &str,
                           default: Option<Vec<u8>>|
     -> Option<Vec<u8>> {
        let Some(raw) = map.and_then(|m| m.get(key)) else {
            return default;
        };

        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") || trimmed == "[]" {
            return None;
        }

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = trimmed[1..trimmed.len() - 1].trim();
            if inner.is_empty() {
                return None;
            }

            let parsed = inner
                .split(',')
                .map(|part| part.trim().parse::<u8>())
                .collect::<Result<Vec<_>, _>>();

            return parsed
                .ok()
                .and_then(|vals| if vals.is_empty() { None } else { Some(vals) })
                .or(default);
        }

        trimmed.parse::<u8>().ok().map(|v| vec![v]).or(default)
    };

    Ok(ParamConfig {
        gnn_atom_enabled: get_bool(map, "gnn_atom_enabled", true),
        gnn_comp_enabled: get_bool(map, "gnn_comp_enabled", true),
        gnn_spacial_enabled: get_bool(map, "gnn_spacial_enabled", true),
        mlp_enabled: get_bool(map, "mlp_enabled", true),
        gnn_atom_layers: get_int(map, "gnn_atom_layers", 3),
        gnn_comp_layers: get_int(map, "gnn_comp_layers", 3),
        gnn_spacial_layers: get_int(map, "gnn_spacial_layers", 2),
        mlp_layers: get_int(map, "mlp_layers", 3),
        atom_graph_analysis_weisfeiler_lehman: get_bool(
            map,
            "atom_graph_analysis_weisfeiler_lehman",
            default_atom_graph_analysis.weisfeiler_lehman,
        ),
        atom_graph_analysis_graphlets: get_u8_list_opt(
            map,
            "atom_graph_analysis_graphlets",
            default_atom_graph_analysis.graphlets.clone(),
        ),
        atom_graph_analysis_path_based_methods: get_bool(
            map,
            "atom_graph_analysis_path_based_methods",
            default_atom_graph_analysis.path_based_methods,
        ),
        atom_graph_analysis_local_overlap_statistics: get_bool(
            map,
            "atom_graph_analysis_local_overlap_statistics",
            default_atom_graph_analysis.local_overlap_statistics,
        ),
        atom_graph_analysis_katz_index: get_bool(
            map,
            "atom_graph_analysis_katz_index",
            default_atom_graph_analysis.katz_index,
        ),
        atom_graph_analysis_lhn_similarity: get_bool(
            map,
            "atom_graph_analysis_lhn_similarity",
            default_atom_graph_analysis.lhn_similarity,
        ),
        atom_graph_analysis_random_walk_methods: get_bool(
            map,
            "atom_graph_analysis_random_walk_methods",
            default_atom_graph_analysis.random_walk_methods,
        ),
        comp_graph_analysis_weisfeiler_lehman: get_bool(
            map,
            "comp_graph_analysis_weisfeiler_lehman",
            default_comp_graph_analysis.weisfeiler_lehman,
        ),
        comp_graph_analysis_graphlets: get_u8_list_opt(
            map,
            "comp_graph_analysis_graphlets",
            default_comp_graph_analysis.graphlets.clone(),
        ),
        comp_graph_analysis_path_based_methods: get_bool(
            map,
            "comp_graph_analysis_path_based_methods",
            default_comp_graph_analysis.path_based_methods,
        ),
        comp_graph_analysis_local_overlap_statistics: get_bool(
            map,
            "comp_graph_analysis_local_overlap_statistics",
            default_comp_graph_analysis.local_overlap_statistics,
        ),
        comp_graph_analysis_katz_index: get_bool(
            map,
            "comp_graph_analysis_katz_index",
            default_comp_graph_analysis.katz_index,
        ),
        comp_graph_analysis_lhn_similarity: get_bool(
            map,
            "comp_graph_analysis_lhn_similarity",
            default_comp_graph_analysis.lhn_similarity,
        ),
        comp_graph_analysis_random_walk_methods: get_bool(
            map,
            "comp_graph_analysis_random_walk_methods",
            default_comp_graph_analysis.random_walk_methods,
        ),
        spacial_graph_analysis_weisfeiler_lehman: get_bool(
            map,
            "spacial_graph_analysis_weisfeiler_lehman",
            default_spacial_graph_analysis.weisfeiler_lehman,
        ),
        spacial_graph_analysis_graphlets: get_u8_list_opt(
            map,
            "spacial_graph_analysis_graphlets",
            default_spacial_graph_analysis.graphlets.clone(),
        ),
        spacial_graph_analysis_path_based_methods: get_bool(
            map,
            "spacial_graph_analysis_path_based_methods",
            default_spacial_graph_analysis.path_based_methods,
        ),
        spacial_graph_analysis_local_overlap_statistics: get_bool(
            map,
            "spacial_graph_analysis_local_overlap_statistics",
            default_spacial_graph_analysis.local_overlap_statistics,
        ),
        spacial_graph_analysis_katz_index: get_bool(
            map,
            "spacial_graph_analysis_katz_index",
            default_spacial_graph_analysis.katz_index,
        ),
        spacial_graph_analysis_lhn_similarity: get_bool(
            map,
            "spacial_graph_analysis_lhn_similarity",
            default_spacial_graph_analysis.lhn_similarity,
        ),
        spacial_graph_analysis_random_walk_methods: get_bool(
            map,
            "spacial_graph_analysis_random_walk_methods",
            default_spacial_graph_analysis.random_walk_methods,
        ),
    })
}

pub(in crate::therapeutic) fn atom_graph_analysis_from_param_cfg(
    param_cfg: &ParamConfig,
) -> GnnAnalysisTools {
    GnnAnalysisTools {
        weisfeiler_lehman: param_cfg.atom_graph_analysis_weisfeiler_lehman,
        graphlets: param_cfg.atom_graph_analysis_graphlets.clone(),
        path_based_methods: param_cfg.atom_graph_analysis_path_based_methods,
        local_overlap_statistics: param_cfg.atom_graph_analysis_local_overlap_statistics,
        katz_index: param_cfg.atom_graph_analysis_katz_index,
        lhn_similarity: param_cfg.atom_graph_analysis_lhn_similarity,
        random_walk_methods: param_cfg.atom_graph_analysis_random_walk_methods,
    }
}

pub(in crate::therapeutic) fn comp_graph_analysis_from_param_cfg(
    param_cfg: &ParamConfig,
) -> GnnAnalysisTools {
    GnnAnalysisTools {
        weisfeiler_lehman: param_cfg.comp_graph_analysis_weisfeiler_lehman,
        graphlets: param_cfg.comp_graph_analysis_graphlets.clone(),
        path_based_methods: param_cfg.comp_graph_analysis_path_based_methods,
        local_overlap_statistics: param_cfg.comp_graph_analysis_local_overlap_statistics,
        katz_index: param_cfg.comp_graph_analysis_katz_index,
        lhn_similarity: param_cfg.comp_graph_analysis_lhn_similarity,
        random_walk_methods: param_cfg.comp_graph_analysis_random_walk_methods,
    }
}

pub(in crate::therapeutic) fn spacial_graph_analysis_from_param_cfg(
    param_cfg: &ParamConfig,
) -> GnnAnalysisTools {
    GnnAnalysisTools {
        weisfeiler_lehman: param_cfg.spacial_graph_analysis_weisfeiler_lehman,
        graphlets: param_cfg.spacial_graph_analysis_graphlets.clone(),
        path_based_methods: param_cfg.spacial_graph_analysis_path_based_methods,
        local_overlap_statistics: param_cfg.spacial_graph_analysis_local_overlap_statistics,
        katz_index: param_cfg.spacial_graph_analysis_katz_index,
        lhn_similarity: param_cfg.spacial_graph_analysis_lhn_similarity,
        random_walk_methods: param_cfg.spacial_graph_analysis_random_walk_methods,
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

// It seems that low or no dropout significantly improves results, but perhaps it makes the
// model more likely to overfit, and makes it less general? Maybe 0.1 or disabled.
// Perhaps skip dropout due to our small TDC data sets.
const DROPOUT: Option<f64> = None;

pub(in crate::therapeutic) const BOND_SIGMA_SQ: f32 = 3.3; // Å. Try 1.5 - 2.2 for sigma, (Square it)

#[cfg(feature = "train")]
type TrainBackend = Autodiff<Wgpu>;
#[cfg(feature = "train")]
type ValidBackend = Wgpu;
// ALso look at Vulkan.

#[derive(Config, Debug)]
pub(in crate::therapeutic) struct ModelConfig {
    pub graph_analysis_feature_version: u8,
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
    /// Graph-level AtomGraph analyses persisted with the model so inference can
    /// rebuild the same features that training used. Older model configs default
    /// to no extra analyses.
    pub atom_graph_analysis: GnnAnalysisTools,
    pub comp_graph_analysis: GnnAnalysisTools,
    pub spacial_graph_analysis: GnnAnalysisTools,
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
        let atom_graph_analysis_dim = self.atom_graph_analysis.feature_dim();
        let comp_graph_analysis_dim = self.comp_graph_analysis.feature_dim();
        let spacial_graph_analysis_dim = self.spacial_graph_analysis.feature_dim();

        // Each GNN branch contributes mean + max pooled vectors (hence * 2).
        let combined_dim = if self.mlp_enabled {
            self.mlp_hidden_dim
        } else {
            0
        } + if self.gnn_atom_enabled {
            self.gnn_hidden_dim * 2
                + if atom_graph_analysis_dim > 0 {
                    self.gnn_hidden_dim
                } else {
                    0
                }
        } else {
            0
        } + if self.gnn_comp_enabled {
            self.gnn_hidden_dim * 2
                + if comp_graph_analysis_dim > 0 {
                    self.gnn_hidden_dim
                } else {
                    0
                }
        } else {
            0
        } + if self.gnn_spacial_enabled {
            self.gnn_hidden_dim * 2
                + if spacial_graph_analysis_dim > 0 {
                    self.gnn_hidden_dim
                } else {
                    0
                }
        } else {
            0
        };

        assert!(
            combined_dim > 0,
            "ModelConfig has all branches disabled — at least one of \
             mlp_enabled / gnn_atom_enabled / gnn_comp_enabled / gnn_spacial_enabled \
             must be true."
        );

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

        let atom_graph_analysis_encoder = if self.gnn_atom_enabled && atom_graph_analysis_dim > 0 {
            vec![LinearConfig::new(atom_graph_analysis_dim, dim_gnn).init(device)]
        } else {
            Vec::new()
        };

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

        let comp_graph_analysis_encoder = if self.gnn_comp_enabled && comp_graph_analysis_dim > 0 {
            vec![LinearConfig::new(comp_graph_analysis_dim, dim_gnn).init(device)]
        } else {
            Vec::new()
        };

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

        let spacial_graph_analysis_encoder =
            if self.gnn_spacial_enabled && spacial_graph_analysis_dim > 0 {
                vec![LinearConfig::new(spacial_graph_analysis_dim, dim_gnn).init(device)]
            } else {
                Vec::new()
            };

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
            atom_graph_analysis_encoder,
            edge_proj,
            emb_comp,
            comp_node_encoder,
            comp_edge_encoder,
            comp_gnn_layers: gnn_comp_layers,
            comp_graph_analysis_encoder,
            comp_edge_proj,
            emb_pharm,
            pharm_node_encoder,
            pharm_edge_encoder,
            spacial_gnn_layers,
            spacial_graph_analysis_encoder,
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
    /// Projects explicit AtomGraph analysis features into the same hidden space as
    /// the learned atom-branch graph embedding.
    atom_graph_analysis_encoder: Vec<Linear<B>>,
    edge_proj: Linear<B>,
    /// Component GNN branch
    emb_comp: Embedding<B>,
    comp_node_encoder: Linear<B>,
    comp_edge_encoder: Linear<B>,
    comp_gnn_layers: Vec<Linear<B>>,
    comp_graph_analysis_encoder: Vec<Linear<B>>,
    comp_edge_proj: Linear<B>,
    /// Spatial (pharmacophore) GNN branch. Nodes = pharmacophore features;
    /// edges encode Euclidean distances via Gaussian RBF.
    emb_pharm: Embedding<B>,
    pharm_node_encoder: Linear<B>,
    pharm_edge_encoder: Linear<B>,
    spacial_gnn_layers: Vec<Linear<B>>,
    spacial_graph_analysis_encoder: Vec<Linear<B>>,
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

/// Apply symmetric normalization D^(-1/2) A D^(-1/2) to a batched adjacency
/// matrix [B, N, N]. Padding rows/cols (which are all-zero in the raw adjacency)
/// remain zero thanks to the eps clamp.
fn sym_normalize<B: Backend>(adj: Tensor<B, 3>) -> Tensor<B, 3> {
    // Sum over j (the trailing N) — keepdim leaves shape [B, N, 1].
    let deg = adj.clone().sum_dim(2);
    let deg_inv_sqrt = deg.clamp_min(1e-9).sqrt().recip(); // [B, N, 1]
    let deg_inv_sqrt_t = deg_inv_sqrt.clone().swap_dims(1, 2); // [B, 1, N]
    adj * deg_inv_sqrt * deg_inv_sqrt_t
}

fn sym_normalize_layered<B: Backend>(adj: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, l, n, _] = adj.dims();
    sym_normalize(adj.reshape([b * l, n, n])).reshape([b, l, n, n])
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

    fn make_multiplex_gnn_layer(
        &self,
        adj_weighted: &Tensor<B, 4>, // [Batch, Layer, N, N]
        mask: &Tensor<B, 3>,         // [Batch, N, 1]
        nodes: Tensor<B, 3>,         // [Batch, N, D_in]
        edge_emb: &Tensor<B, 5>,     // [Batch, Layer, N, N, D_hidden]
        gnn_linear: &Linear<B>,
        dropout: bool,
    ) -> Tensor<B, 3> {
        let [b, l, n, _] = adj_weighted.dims();
        let d = nodes.dims()[2];

        let nodes_j = nodes.clone().unsqueeze_dim::<4>(1).unsqueeze_dim::<5>(1);
        // Note: no inner ReLU here. Several edge-feature scalars are *signed*
        // (signed angle deviation on L1, torsion alignment ∈ [-1, 1] on L2/L3) and the
        // FF training signal lives in those negative values too. The outer ReLU after
        // `gnn_linear` keeps the GNN nonlinear without clipping signed inputs.
        let message = nodes_j + edge_emb.clone();
        let weights = adj_weighted.clone().unsqueeze_dim(4);

        let agg = (message * weights)
            .sum_dim(3)
            .reshape([b, l, n, d])
            .sum_dim(1)
            .reshape([b, n, d]);

        let mut layer_out = activation::relu(gnn_linear.forward(agg));

        if dropout && DROPOUT.is_some() {
            layer_out = self.dropout.forward(layer_out);
        }

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
        adj: Tensor<B, 4>,
        edge_feats: Tensor<B, 5>,
        mask: Tensor<B, 3>,
        // Component graph
        comp_idx: Tensor<B, 2, Int>,
        comp_scalars: Tensor<B, 3>,
        comp_adj: Tensor<B, 3>,
        comp_edge_feats: Tensor<B, 4>,
        comp_mask: Tensor<B, 3>,
        comp_graph_analysis: Tensor<B, 2>,
        // Spatial (pharmacophore) graph
        pharm_idx: Tensor<B, 2, Int>,
        pharm_scalars: Tensor<B, 3>,
        pharm_adj: Tensor<B, 3>,
        pharm_edge_feats: Tensor<B, 4>,
        pharm_mask: Tensor<B, 3>,
        spacial_graph_analysis: Tensor<B, 2>,
        atom_graph_analysis: Tensor<B, 2>,
        params: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut branches: Vec<Tensor<B, 2>> = Vec::new();

        // --- Atom GNN branch ---
        if self.atom_gnn_enabled {
            let [b, l, n, _n2, f] = edge_feats.dims();

            let ef_flat = edge_feats.clone().reshape([b * l * n * n, f]);
            let gate_flat = activation::sigmoid(self.edge_proj.forward(ef_flat.clone()));
            let gate = gate_flat.reshape([b, l, n, n]);
            // Gate the raw adjacency, then symmetric-normalize so message
            // magnitudes stay scale-invariant w.r.t. the gate.
            let adj_eff = sym_normalize_layered(adj * gate);

            let edge_emb_flat = self.edge_encoder.forward(ef_flat);
            let [_, d_hidden] = edge_emb_flat.dims();
            let edge_emb = edge_emb_flat.reshape([b, l, n, n, d_hidden]);

            let x_elem = self.emb_elem.forward(elem_idx);
            let x_ff = self.emb_ff.forward(ff_idx);
            let raw_nodes = Tensor::cat(vec![x_elem, x_ff, scalars], 2);
            let nodes = self.node_encoder.forward(raw_nodes);

            let mut gnn_prev = nodes;
            for (i, layer) in self.gnn_layers.iter().enumerate() {
                let dropout = i != self.gnn_layers.len() - 1;
                gnn_prev = self
                    .make_multiplex_gnn_layer(&adj_eff, &mask, gnn_prev, &edge_emb, layer, dropout);
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
            let mut atom_branch = vec![graph_mean.reshape([b_g, d]), graph_max.reshape([b_g, d])];

            if let Some(analysis_encoder) = self.atom_graph_analysis_encoder.first() {
                atom_branch.push(activation::relu(
                    analysis_encoder.forward(atom_graph_analysis),
                ));
            }

            branches.push(Tensor::cat(atom_branch, 1));
        }

        // --- Component GNN branch ---
        if self.comp_gnn_enabled {
            let [bc, nc, _nc2, fc] = comp_edge_feats.dims();

            let cef_flat = comp_edge_feats.clone().reshape([bc * nc * nc, fc]);
            let comp_gate_flat = activation::sigmoid(self.comp_edge_proj.forward(cef_flat.clone()));
            let comp_gate = comp_gate_flat.reshape([bc, nc, nc]);
            let comp_adj_eff = sym_normalize(comp_adj * comp_gate);

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
            let mut comp_branch = vec![comp_mean.reshape([b_c, d_c]), comp_max.reshape([b_c, d_c])];

            if let Some(analysis_encoder) = self.comp_graph_analysis_encoder.first() {
                comp_branch.push(activation::relu(
                    analysis_encoder.forward(comp_graph_analysis),
                ));
            }

            branches.push(Tensor::cat(comp_branch, 1));
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
            let pharm_adj_eff = sym_normalize(pharm_adj * pharm_gate);

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
            let mut spacial_branch = vec![
                pharm_mean.reshape([b_p, d_p]),
                pharm_max.reshape([b_p, d_p]),
            ];

            if let Some(analysis_encoder) = self.spacial_graph_analysis_encoder.first() {
                spacial_branch.push(activation::relu(
                    analysis_encoder.forward(spacial_graph_analysis),
                ));
            }

            branches.push(Tensor::cat(spacial_branch, 1));
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
    pub graph: GraphDataAtom,
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
    pub adj_list: Tensor<B, 4>,
    pub edge_feats: Tensor<B, 5>,
    pub mask: Tensor<B, 3>,
    // Component graph
    pub comp_indices: Tensor<B, 2, Int>,
    pub comp_scalars: Tensor<B, 3>,
    pub comp_adj_list: Tensor<B, 3>,
    pub comp_edge_feats: Tensor<B, 4>,
    pub comp_mask: Tensor<B, 3>,
    pub comp_graph_analysis: Tensor<B, 2>,
    // Spatial (pharmacophore) graph
    pub pharm_indices: Tensor<B, 2, Int>,
    pub pharm_scalars: Tensor<B, 3>,
    pub pharm_adj_list: Tensor<B, 3>,
    pub pharm_edge_feats: Tensor<B, 4>,
    pub pharm_mask: Tensor<B, 3>,
    pub spacial_graph_analysis: Tensor<B, 2>,
    pub atom_graph_analysis: Tensor<B, 2>,
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
        let mut batch_comp_graph_analysis = Vec::new();
        let mut batch_spacial_graph_analysis = Vec::new();
        let mut batch_atom_graph_analysis = Vec::new();
        let mut batch_globals = Vec::new();
        let mut batch_y = Vec::new();

        let n_feat_params = items[0].features_property.len();

        // Per-atom scalar count is fixed by GraphDataAtom construction.
        let n_scalars_per_atom = PER_ATOM_SCALARS;
        let n_atom_graph_analysis = items[0].graph.analysis_features.len().max(1);
        let n_comp_graph_analysis = items[0].graph_comp.analysis_features.len().max(1);
        let n_spacial_graph_analysis = items[0].graph_spacial.analysis_features.len().max(1);

        for mut item in items {
            // Mol parameters, and the target value.
            self.scaler.apply_in_place(&mut item.features_property);
            batch_globals.extend_from_slice(&item.features_property);
            batch_y.push(self.scaler.normalize_target(item.target));

            // Atom graph
            let g = &item.graph;
            batch_elem_ids.extend(gnn::pad_indices(&g.elem_indices, g.num_atoms, MAX_ATOMS));
            batch_ff_ids.extend(gnn::pad_indices(&g.ff_indices, g.num_atoms, MAX_ATOMS));
            batch_scalars.extend(gnn::pad_scalars(
                &g.scalars,
                g.num_atoms,
                n_scalars_per_atom,
                MAX_ATOMS,
            ));
            let (p_adj, p_mask) = atom_bond::pad_atom_adj_and_mask(&g.adj, g.num_atoms, MAX_ATOMS);
            batch_adj.extend(p_adj);
            batch_mask.extend(p_mask);
            batch_edge_feats.extend(atom_bond::pad_atom_edge_feats(
                &g.edge_feats,
                g.num_atoms,
                MAX_ATOMS,
            ));
            if g.analysis_features.is_empty() {
                batch_atom_graph_analysis
                    .resize(batch_atom_graph_analysis.len() + n_atom_graph_analysis, 0.0);
            } else {
                debug_assert_eq!(g.analysis_features.len(), n_atom_graph_analysis);
                batch_atom_graph_analysis.extend_from_slice(&g.analysis_features);
            }

            // Component graph
            let gc = &item.graph_comp;
            batch_comp_ids.extend(gnn::pad_indices(
                &gc.comp_type_indices,
                gc.num_comps,
                MAX_COMPS,
            ));
            batch_comp_scalars.extend(gnn::pad_scalars(
                &gc.scalars,
                gc.num_comps,
                PER_COMP_SCALARS,
                MAX_COMPS,
            ));
            let (p_comp_adj, p_comp_mask) = gnn::pad_adj_and_mask(&gc.adj, gc.num_comps, MAX_COMPS);
            batch_comp_adj.extend(p_comp_adj);
            batch_comp_mask.extend(p_comp_mask);
            batch_comp_edge_feats.extend(gnn::pad_edge_feats(
                &gc.edge_feats,
                gc.num_comps,
                PER_EDGE_COMP_FEATS,
                MAX_COMPS,
            ));
            if gc.analysis_features.is_empty() {
                batch_comp_graph_analysis
                    .resize(batch_comp_graph_analysis.len() + n_comp_graph_analysis, 0.0);
            } else {
                debug_assert_eq!(gc.analysis_features.len(), n_comp_graph_analysis);
                batch_comp_graph_analysis.extend_from_slice(&gc.analysis_features);
            }

            // Spatial (pharmacophore) graph
            let gs = &item.graph_spacial;
            batch_pharm_ids.extend(gnn::pad_indices(
                &gs.pharm_type_indices,
                gs.num_nodes,
                MAX_PHARM,
            ));
            batch_pharm_scalars.extend(gnn::pad_scalars(
                &gs.scalars,
                gs.num_nodes,
                PER_PHARM_SCALARS,
                MAX_PHARM,
            ));
            let (p_pharm_adj, p_pharm_mask) =
                gnn::pad_adj_and_mask(&gs.adj, gs.num_nodes, MAX_PHARM);
            batch_pharm_adj.extend(p_pharm_adj);
            batch_pharm_mask.extend(p_pharm_mask);
            batch_pharm_edge_feats.extend(gnn::pad_edge_feats(
                &gs.edge_feats,
                gs.num_nodes,
                PER_SPACIAL_EDGE_FEATS,
                MAX_PHARM,
            ));
            if gs.analysis_features.is_empty() {
                batch_spacial_graph_analysis.resize(
                    batch_spacial_graph_analysis.len() + n_spacial_graph_analysis,
                    0.0,
                );
            } else {
                debug_assert_eq!(gs.analysis_features.len(), n_spacial_graph_analysis);
                batch_spacial_graph_analysis.extend_from_slice(&gs.analysis_features);
            }
        }

        let elem_ids = TensorData::new(batch_elem_ids, [batch_size, MAX_ATOMS]);
        let ff_ids = TensorData::new(batch_ff_ids, [batch_size, MAX_ATOMS]);
        let scalars = TensorData::new(batch_scalars, [batch_size, MAX_ATOMS, n_scalars_per_atom]);
        let adj = TensorData::new(
            batch_adj,
            [batch_size, ATOM_GNN_EDGE_LAYERS, MAX_ATOMS, MAX_ATOMS],
        );
        let edge_feats = TensorData::new(
            batch_edge_feats,
            [
                batch_size,
                ATOM_GNN_EDGE_LAYERS,
                MAX_ATOMS,
                MAX_ATOMS,
                ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
            ],
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
        let comp_graph_analysis = TensorData::new(
            batch_comp_graph_analysis,
            [batch_size, n_comp_graph_analysis],
        );
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
        let spacial_graph_analysis = TensorData::new(
            batch_spacial_graph_analysis,
            [batch_size, n_spacial_graph_analysis],
        );
        let atom_graph_analysis = TensorData::new(
            batch_atom_graph_analysis,
            [batch_size, n_atom_graph_analysis],
        );
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
            comp_graph_analysis: Tensor::from_data(comp_graph_analysis, device),
            pharm_indices: Tensor::from_data(pharm_ids, device),
            pharm_scalars: Tensor::from_data(pharm_scalars, device),
            pharm_adj_list: Tensor::from_data(pharm_adj, device),
            pharm_edge_feats: Tensor::from_data(pharm_edge_feats, device),
            pharm_mask: Tensor::from_data(pharm_mask, device),
            spacial_graph_analysis: Tensor::from_data(spacial_graph_analysis, device),
            atom_graph_analysis: Tensor::from_data(atom_graph_analysis, device),
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
            batch.comp_graph_analysis,
            batch.pharm_indices,
            batch.pharm_scalars,
            batch.pharm_adj_list,
            batch.pharm_edge_feats,
            batch.pharm_mask,
            batch.spacial_graph_analysis,
            batch.atom_graph_analysis,
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
            batch.comp_graph_analysis,
            batch.pharm_indices,
            batch.pharm_scalars,
            batch.pharm_adj_list,
            batch.pharm_edge_feats,
            batch.pharm_mask,
            batch.spacial_graph_analysis,
            batch.atom_graph_analysis,
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

/// Convert pre-loaded `(MoleculeSmall, target)` pairs into model `Sample`s. Skips
/// any molecules that fail feature extraction or graph construction.
#[cfg(feature = "train")]
pub(in crate::therapeutic) fn samples_from_mols(
    data: &[(MoleculeSmall, f32)],
    ff_params: &ForceFieldParams,
    atom_graph_analysis: &GnnAnalysisTools,
    comp_graph_analysis: &GnnAnalysisTools,
    spacial_graph_analysis: &GnnAnalysisTools,
) -> Vec<Sample> {
    let mut out = Vec::with_capacity(data.len());
    for (mol, target) in data {
        let feat_params = match mlp::mlp_feats_from_mol(mol) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error extracting MLP features: {e:?}; skipping mol.");
                continue;
            }
        };

        if mol.common.bonds.is_empty() {
            continue;
        }

        let graph = match GraphDataAtom::new(mol, ff_params, atom_graph_analysis) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Error getting graph data: {e:?}");
                continue;
            }
        };

        if graph.num_atoms == 0 {
            continue;
        }

        let graph_comp = match &mol.components {
            Some(comps) => {
                match GraphDataComponent::new(mol, comps, ff_params, comp_graph_analysis) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("Error getting comp graph data: {e:?}");
                        continue;
                    }
                }
            }
            None => {
                eprintln!("Missing components for mol; skipping.");
                continue;
            }
        };

        // GraphDataSpacial::new returns Ok(empty) when characterization is missing
        // or no pharmacophore sites are present, so this never errors.
        let graph_spacial = match GraphDataSpacial::new(mol, spacial_graph_analysis) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Error getting spatial graph data: {e:?}");
                continue;
            }
        };

        out.push(Sample {
            features_property: feat_params,
            graph,
            graph_comp,
            graph_spacial,
            target: *target,
        });
    }
    out
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

/// Train, save model/scaler/config given pre-built train and validation samples.
/// `train()` and `eval()` both go through this entry point.
#[cfg(feature = "train")]
pub(in crate::therapeutic) fn train_with_samples(
    dataset: DatasetTdc,
    param_cfg: &ParamConfig,
    data_train: Vec<Sample>,
    data_valid: Vec<Sample>,
) -> io::Result<()> {
    let atom_graph_analysis = atom_graph_analysis_from_param_cfg(param_cfg);
    let comp_graph_analysis = comp_graph_analysis_from_param_cfg(param_cfg);
    let spacial_graph_analysis = spacial_graph_analysis_from_param_cfg(param_cfg);
    println!(
        "Branch config for '{}': atom_gnn={} comp_gnn={} spacial_gnn={} mlp={}",
        dataset.name(),
        param_cfg.gnn_atom_enabled,
        param_cfg.gnn_comp_enabled,
        param_cfg.gnn_spacial_enabled,
        param_cfg.mlp_enabled,
    );
    if param_cfg.gnn_atom_enabled && atom_graph_analysis.feature_dim() > 0 {
        println!(
            "AtomGraph analysis features enabled: {:?}",
            atom_graph_analysis.feature_names_with_prefix("atom")
        );
    }
    if param_cfg.gnn_comp_enabled && comp_graph_analysis.feature_dim() > 0 {
        println!(
            "ComponentGraph analysis features enabled: {:?}",
            comp_graph_analysis.feature_names_with_prefix("comp")
        );
    }
    if param_cfg.gnn_spacial_enabled && spacial_graph_analysis.feature_dim() > 0 {
        println!(
            "SpacialGraph analysis features enabled: {:?}",
            spacial_graph_analysis.feature_names_with_prefix("spacial")
        );
    }

    let (model_path, scaler_path, config_path) = dataset.model_paths();

    let start = Instant::now();

    let model_dir_train_val = Path::new(TRAIN_VALID_DIR);
    fs::create_dir_all(model_dir_train_val)?;

    if data_train.is_empty() {
        return Err(io::Error::other("No training samples"));
    }

    println!("Training on : {} samples", data_train.len());

    let scaler = fit_scaler(&data_train);
    let device = Default::default();

    // This seeding prevents random behavior.
    const SEED: u64 = 42;
    TrainBackend::seed(&device, SEED);
    ValidBackend::seed(&device, SEED);

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
    .build(InMemDataset::new(data_valid));

    let num_params = data_train[0].features_property.len();
    let model_cfg = ModelConfig {
        graph_analysis_feature_version: GRAPH_ANALYSIS_FEATURE_VERSION,
        global_input_dim: num_params,
        gnn_hidden_dim: 64,
        mlp_hidden_dim: 128,
        vocab_size_elem: 12,           // matches vocab_lookup_element max + 1
        vocab_size_ff: FF_BUCKETS + 2, // 0 pad + 1..FF_BUCKETS + unknown.
        embedding_dim: 16,             // Tune this (8, 16, 32)
        n_node_scalars: PER_ATOM_SCALARS,
        edge_feat_dim: ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
        vocab_size_comp: 11, // 0=pad + 10 component types (see vocab_lookup_component)
        comp_embedding_dim: 8,
        n_comp_scalars: PER_COMP_SCALARS,
        comp_edge_feat_dim: PER_EDGE_COMP_FEATS,
        // Spatial (pharmacophore) GNN
        vocab_size_pharm: SPACIAL_VOCAB_SIZE, // 0=pad,1=donor,2=acc,3=hydrophobic,4=aromatic
        pharm_embedding_dim: 8,
        n_pharm_scalars: PER_PHARM_SCALARS,
        spacial_edge_feat_dim: PER_SPACIAL_EDGE_FEATS,
        atom_graph_analysis: atom_graph_analysis.clone(),
        comp_graph_analysis: comp_graph_analysis.clone(),
        spacial_graph_analysis: spacial_graph_analysis.clone(),
        gnn_atom_enabled: param_cfg.gnn_atom_enabled,
        gnn_comp_enabled: param_cfg.gnn_comp_enabled,
        gnn_spacial_enabled: param_cfg.gnn_spacial_enabled,
        mlp_enabled: param_cfg.mlp_enabled,
        gnn_atom_layers: param_cfg.gnn_atom_layers,
        gnn_comp_layers: param_cfg.gnn_comp_layers,
        gnn_spacial_layers: param_cfg.gnn_spacial_layers,
        mlp_layers: param_cfg.mlp_layers,
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
    .metrics((LossMetric::new(),))
    .num_epochs(80)
    .with_training_strategy(TrainingStrategy::SingleDevice(device.clone()))
    .summary();

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    result.model.save_file(&model_path, &recorder).unwrap();

    let config_file = fs::File::create(&config_path).expect("Could not create config file");
    serde_json::to_writer_pretty(config_file, &model_cfg).expect("Could not write config");
    println!("Saved config to {:?}", config_path);

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
pub(in crate::therapeutic) fn train(
    data_path: &Path,
    dataset: DatasetTdc,
    tgt_col: usize,
    mol_specific_params: &mut HashMap<String, ForceFieldParams>,
    ff_params: &ForceFieldParams,
) -> io::Result<()> {
    let (csv_path, mol_path) = dataset.csv_mol_paths(data_path)?;

    println!("Started training on {csv_path:?}");

    let tts = TrainTestSplit::new(dataset);
    let param_cfg = load_param_cfg(&dataset.name())?;
    let atom_graph_analysis = atom_graph_analysis_from_param_cfg(&param_cfg);
    let comp_graph_analysis = comp_graph_analysis_from_param_cfg(&param_cfg);
    let spacial_graph_analysis = spacial_graph_analysis_from_param_cfg(&param_cfg);

    let loaded = load_training_data(
        Path::new(&csv_path),
        Path::new(&mol_path),
        tgt_col,
        &tts,
        mol_specific_params,
        ff_params,
        false,
    )?;

    let data_train = samples_from_mols(
        &loaded.train,
        ff_params,
        &atom_graph_analysis,
        &comp_graph_analysis,
        &spacial_graph_analysis,
    );
    let data_valid = samples_from_mols(
        &loaded.validation,
        ff_params,
        &atom_graph_analysis,
        &comp_graph_analysis,
        &spacial_graph_analysis,
    );

    train_with_samples(dataset, &param_cfg, data_train, data_valid)
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
            .map(|set| DatasetTdc::from_str(set).unwrap())
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
