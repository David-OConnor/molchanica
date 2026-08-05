//! A native Rust implementation of the ProteinMPNN message-passing network.
//!
//! [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) ·
//! [ProteinMPNN-ddG](https://github.com/PeptoneLtd/proteinmpnn_ddg)
//!
//! This exists so Molchanica can produce a whole saturation-mutagenesis scan without shelling out
//! to Python. The upstream ProteinMPNN-ddG is JAX-on-CUDA and Linux-only, so it could never be one
//! of the tools in [`crate::external_tools`]; but the network itself is small — six message-passing
//! layers over a 128-dimensional hidden state — and the arithmetic is entirely ordinary, so there
//! is nothing about it that needs a framework.
//!
//! # What is computed
//!
//! ProteinMPNN's `unconditional_probs`: one forward pass giving, for every position, a
//! distribution over the twenty amino acids conditioned on the backbone alone and on no part of
//! the sequence. That is the single-pass structure-only quantity a ΔΔG scan is built from, and it
//! is why the whole scan costs one pass rather than one pass per position.
//!
//! # Numerical fidelity
//!
//! Every layer here mirrors a specific upstream module, named in the comments. The weights are the
//! published ProteinMPNN checkpoints, converted by `scripts/convert_mpnn_weights.py`. That script
//! also writes a reference forward pass, and [`super::verify`] replays it through this code, so
//! agreement with upstream is something the user can check on their own machine in one command
//! rather than something they have to take on trust.

use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use rayon::prelude::*;

/// The 21-letter alphabet ProteinMPNN's output layer is ordered by. `X` is the unknown/other
/// class, and is excluded from mutation scanning.
pub const ALPHABET: [char; 21] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y', 'X',
];

/// Hidden width of every layer.
pub const HIDDEN: usize = 128;
/// Neighbours each residue attends to.
pub const K_NEIGHBORS: usize = 48;
/// Radial basis functions per atom pair.
const NUM_RBF: usize = 16;
/// The 25 ordered backbone atom pairs the edge features are built from.
const NUM_ATOM_PAIRS: usize = 25;
/// Width of the learned positional embedding.
const POSITIONAL_EMBEDDING: usize = 16;
/// Largest sequence separation the positional encoding distinguishes.
const MAX_RELATIVE: i32 = 32;
/// Messages are summed and divided by this rather than averaged; upstream calls it `scale`.
const MESSAGE_SCALE: f32 = 30.0;

/// A dense matrix in row-major order.
#[derive(Clone, Debug, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn row(&self, index: usize) -> &[f32] {
        &self.data[index * self.cols..(index + 1) * self.cols]
    }

    fn row_mut(&mut self, index: usize) -> &mut [f32] {
        let cols = self.cols;
        &mut self.data[index * cols..(index + 1) * cols]
    }
}

/// A `nn.Linear`: `y = x Wᵀ + b`, with `weight` stored `[out, in]` as PyTorch does.
#[derive(Clone, Debug, Default)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    /// Apply to every row of `input`, in parallel over rows.
    ///
    /// This is where essentially all the time goes — the edge tensors are `L × K × 384` — so the
    /// inner loop is kept to a contiguous dot product the compiler can vectorize.
    pub fn forward(&self, input: &Matrix) -> Matrix {
        debug_assert_eq!(input.cols, self.in_features);
        let mut output = Matrix::zeros(input.rows, self.out_features);
        let in_features = self.in_features;

        output
            .data
            .par_chunks_mut(self.out_features)
            .enumerate()
            .for_each(|(row_index, out_row)| {
                let in_row = &input.data[row_index * in_features..(row_index + 1) * in_features];
                for (out_index, value) in out_row.iter_mut().enumerate() {
                    let weight_row =
                        &self.weight[out_index * in_features..(out_index + 1) * in_features];
                    let mut sum = match &self.bias {
                        Some(bias) => bias[out_index],
                        None => 0.0,
                    };
                    for (x, w) in in_row.iter().zip(weight_row) {
                        sum += x * w;
                    }
                    *value = sum;
                }
            });
        output
    }
}

/// A `nn.LayerNorm` over the last dimension.
#[derive(Clone, Debug, Default)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LayerNorm {
    /// PyTorch's default epsilon, applied inside the square root as PyTorch does.
    const EPSILON: f32 = 1e-5;

    pub fn forward_in_place(&self, matrix: &mut Matrix) {
        let cols = matrix.cols;
        matrix.data.par_chunks_mut(cols).for_each(|row| {
            let mean = row.iter().sum::<f32>() / cols as f32;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / cols as f32;
            let inverse = 1.0 / (variance + Self::EPSILON).sqrt();
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value - mean) * inverse * self.weight[index] + self.bias[index];
            }
        });
    }
}

/// `torch.nn.GELU()`: the exact (erf) form, not the tanh approximation, which is what PyTorch's
/// default `GELU()` uses and what the checkpoints were trained with.
fn gelu_in_place(matrix: &mut Matrix) {
    matrix.data.par_iter_mut().for_each(|value| {
        *value *= 0.5 * (1.0 + erf(*value * std::f32::consts::FRAC_1_SQRT_2));
    });
}

/// Abramowitz & Stegun 7.1.26. Maximum absolute error 1.5e-7, comfortably below f32 resolution at
/// the magnitudes GELU is evaluated over.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// `PositionWiseFeedForward`: two linears with a GELU between them.
#[derive(Clone, Debug, Default)]
pub struct FeedForward {
    pub w_in: Linear,
    pub w_out: Linear,
}

impl FeedForward {
    fn forward(&self, input: &Matrix) -> Matrix {
        let mut hidden = self.w_in.forward(input);
        gelu_in_place(&mut hidden);
        self.w_out.forward(&hidden)
    }
}

/// One `EncLayer`: a node update followed by an edge update.
#[derive(Clone, Debug, Default)]
pub struct EncoderLayer {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
    pub w11: Linear,
    pub w12: Linear,
    pub w13: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub dense: FeedForward,
}

/// One `DecLayer`: a node update only; edges are not revised in the decoder.
#[derive(Clone, Debug, Default)]
pub struct DecoderLayer {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dense: FeedForward,
}

/// Every weight the network needs.
#[derive(Clone, Debug, Default)]
pub struct ProteinMpnnWeights {
    pub edge_embedding: Linear,
    pub norm_edges: LayerNorm,
    pub positional_embedding: Linear,
    pub w_e: Linear,
    pub encoder: Vec<EncoderLayer>,
    pub decoder: Vec<DecoderLayer>,
    pub w_out: Linear,
}

/// The backbone one scan runs over.
#[derive(Clone, Debug, Default)]
pub struct Backbone {
    /// N, CA, C, O for each residue, in that order. Missing atoms are not permitted; callers
    /// filter incomplete residues out first.
    pub n: Vec<[f32; 3]>,
    pub ca: Vec<[f32; 3]>,
    pub c: Vec<[f32; 3]>,
    pub o: Vec<[f32; 3]>,
    /// Residue numbers, used by the positional encoding. Gaps in numbering are meaningful: they
    /// tell the model two residues are not adjacent in sequence even though they are in the array.
    pub residue_index: Vec<i32>,
    /// Chain index per residue. The positional encoding only relates residues within a chain.
    pub chain_index: Vec<i32>,
}

impl Backbone {
    pub fn len(&self) -> usize {
        self.ca.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ca.is_empty()
    }

    fn validate(&self) -> io::Result<()> {
        let length = self.ca.len();
        if length < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a ΔΔG scan needs at least two residues with complete backbones",
            ));
        }
        let consistent = self.n.len() == length
            && self.c.len() == length
            && self.o.len() == length
            && self.residue_index.len() == length
            && self.chain_index.len() == length;
        if !consistent {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "backbone arrays have inconsistent lengths",
            ));
        }
        Ok(())
    }

    /// The virtual Cβ ProteinMPNN builds from the backbone, so glycine has one too.
    ///
    /// The coefficients are upstream's, fitted to reproduce a real Cβ position from the N/CA/C
    /// frame. Reimplementing rather than approximating matters: Cβ–Cβ is one of the 25 atom pairs,
    /// and every edge feature would shift if this were off.
    fn virtual_cb(&self, index: usize) -> [f32; 3] {
        let ca = self.ca[index];
        let b = subtract(ca, self.n[index]);
        let c = subtract(self.c[index], ca);
        let a = cross(b, c);
        [
            -0.582_734_3 * a[0] + 0.568_028_3 * b[0] - 0.540_674_7 * c[0] + ca[0],
            -0.582_734_3 * a[1] + 0.568_028_3 * b[1] - 0.540_674_7 * c[1] + ca[1],
            -0.582_734_3 * a[2] + 0.568_028_3 * b[2] - 0.540_674_7 * c[2] + ca[2],
        ]
    }
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = subtract(a, b);
    // The 1e-6 inside the root is upstream's, and is kept because it perturbs short distances
    // enough to matter at f32 precision.
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1e-6).sqrt()
}

/// The neighbour graph: for each residue, the `K` nearest by Cα distance, itself included.
struct NeighborGraph {
    /// `[length * k]` residue indices.
    indices: Vec<usize>,
    k: usize,
}

impl NeighborGraph {
    fn build(backbone: &Backbone, k: usize) -> Self {
        let length = backbone.len();
        let k = k.min(length);
        let mut indices = vec![0usize; length * k];

        indices.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            let mut distances: Vec<(f32, usize)> = (0..length)
                .map(|j| (distance(backbone.ca[i], backbone.ca[j]), j))
                .collect();
            // Ties broken by index so the graph is deterministic, which upstream's `topk`
            // also is; without it two runs on the same structure could differ.
            distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (slot, (_, j)) in row.iter_mut().zip(distances) {
                *slot = j;
            }
        });

        Self { indices, k }
    }

    fn neighbors(&self, residue: usize) -> &[usize] {
        &self.indices[residue * self.k..(residue + 1) * self.k]
    }
}

/// `_rbf`: 16 Gaussians evenly spaced over 2–22 Å.
fn rbf(distance: f32, out: &mut [f32]) {
    const D_MIN: f32 = 2.0;
    const D_MAX: f32 = 22.0;
    let sigma = (D_MAX - D_MIN) / NUM_RBF as f32;
    for (index, value) in out.iter_mut().enumerate() {
        // linspace(2, 22, 16) — endpoints inclusive, which is what torch.linspace gives.
        let mu = D_MIN + (D_MAX - D_MIN) * index as f32 / (NUM_RBF - 1) as f32;
        *value = (-((distance - mu) / sigma).powi(2)).exp();
    }
}

/// The 25 ordered (from, to) atom-type pairs, in upstream's exact order. The order is part of the
/// learned weight layout, so it cannot be rearranged.
const ATOM_PAIRS: [(usize, usize); NUM_ATOM_PAIRS] = [
    (1, 1), // Ca-Ca
    (0, 0), // N-N
    (2, 2), // C-C
    (3, 3), // O-O
    (4, 4), // Cb-Cb
    (1, 0), // Ca-N
    (1, 2), // Ca-C
    (1, 3), // Ca-O
    (1, 4), // Ca-Cb
    (0, 2), // N-C
    (0, 3), // N-O
    (0, 4), // N-Cb
    (4, 2), // Cb-C
    (4, 3), // Cb-O
    (3, 2), // O-C
    (0, 1), // N-Ca
    (2, 1), // C-Ca
    (3, 1), // O-Ca
    (4, 1), // Cb-Ca
    (2, 0), // C-N
    (3, 0), // O-N
    (4, 0), // Cb-N
    (2, 4), // C-Cb
    (3, 4), // O-Cb
    (2, 3), // C-O
];

/// Build the edge feature matrix, `[length * k, 416]`, before the embedding.
fn edge_features(backbone: &Backbone, graph: &NeighborGraph) -> Matrix {
    let length = backbone.len();
    let k = graph.k;
    // Atom order within a residue: N, Ca, C, O, Cb — indexed by `ATOM_PAIRS`.
    let atoms: Vec<[[f32; 3]; 5]> = (0..length)
        .map(|i| {
            [
                backbone.n[i],
                backbone.ca[i],
                backbone.c[i],
                backbone.o[i],
                backbone.virtual_cb(i),
            ]
        })
        .collect();

    let feature_width = POSITIONAL_EMBEDDING_INPUT + NUM_ATOM_PAIRS * NUM_RBF;
    let mut features = Matrix::zeros(length * k, feature_width);

    features
        .data
        .par_chunks_mut(feature_width)
        .enumerate()
        .for_each(|(edge, row)| {
            let i = edge / k;
            let j = graph.neighbors(i)[edge % k];

            // Positional one-hot first, matching the `cat((E_positional, RBF_all))` order.
            let same_chain = backbone.chain_index[i] == backbone.chain_index[j];
            let bucket = if same_chain {
                let offset = backbone.residue_index[i] - backbone.residue_index[j];
                (offset + MAX_RELATIVE).clamp(0, 2 * MAX_RELATIVE) as usize
            } else {
                // Residues in different chains get their own bucket rather than a spurious
                // sequence separation.
                (2 * MAX_RELATIVE + 1) as usize
            };
            row[bucket] = 1.0;

            for (pair_index, (from, to)) in ATOM_PAIRS.into_iter().enumerate() {
                let d = distance(atoms[i][from], atoms[j][to]);
                let start = POSITIONAL_EMBEDDING_INPUT + pair_index * NUM_RBF;
                rbf(d, &mut row[start..start + NUM_RBF]);
            }
        });

    features
}

/// One-hot width of the positional encoding: `2 × 32 + 1` in-chain buckets plus one for
/// cross-chain pairs.
const POSITIONAL_EMBEDDING_INPUT: usize = (2 * MAX_RELATIVE + 2) as usize;

/// Gather node states for every edge: `[length * k, HIDDEN]`.
fn gather_nodes(nodes: &Matrix, graph: &NeighborGraph) -> Matrix {
    let k = graph.k;
    let mut gathered = Matrix::zeros(nodes.rows * k, nodes.cols);
    let cols = nodes.cols;
    gathered
        .data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(edge, row)| {
            let source = graph.neighbors(edge / k)[edge % k];
            row.copy_from_slice(nodes.row(source));
        });
    gathered
}

/// Repeat each node state across its `k` edges: `[length * k, HIDDEN]`.
fn expand_nodes(nodes: &Matrix, k: usize) -> Matrix {
    let mut expanded = Matrix::zeros(nodes.rows * k, nodes.cols);
    let cols = nodes.cols;
    expanded
        .data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(edge, row)| row.copy_from_slice(nodes.row(edge / k)));
    expanded
}

/// Concatenate matrices along the feature axis. All must share a row count.
fn concat(parts: &[&Matrix]) -> Matrix {
    let rows = parts[0].rows;
    let cols: usize = parts.iter().map(|part| part.cols).sum();
    let mut out = Matrix::zeros(rows, cols);
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, slot)| {
            let mut offset = 0;
            for part in parts {
                slot[offset..offset + part.cols].copy_from_slice(part.row(row));
                offset += part.cols;
            }
        });
    out
}

/// Sum each residue's incoming messages and divide by the fixed scale.
fn aggregate(messages: &Matrix, length: usize, k: usize) -> Matrix {
    let cols = messages.cols;
    let mut out = Matrix::zeros(length, cols);
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(residue, row)| {
            for neighbor in 0..k {
                let message = messages.row(residue * k + neighbor);
                for (slot, value) in row.iter_mut().zip(message) {
                    *slot += value;
                }
            }
            for slot in row.iter_mut() {
                *slot /= MESSAGE_SCALE;
            }
        });
    out
}

fn add_in_place(target: &mut Matrix, delta: &Matrix) {
    target
        .data
        .par_iter_mut()
        .zip(delta.data.par_iter())
        .for_each(|(slot, value)| *slot += value);
}

/// A three-linear message MLP with GELU between each stage.
fn message_mlp(input: &Matrix, w1: &Linear, w2: &Linear, w3: &Linear) -> Matrix {
    let mut hidden = w1.forward(input);
    gelu_in_place(&mut hidden);
    let mut hidden = w2.forward(&hidden);
    gelu_in_place(&mut hidden);
    w3.forward(&hidden)
}

/// Per-position log-probabilities over the 21-letter alphabet.
pub struct LogProbabilities {
    pub length: usize,
    /// `[length * 21]`, log-softmax over the alphabet.
    pub data: Vec<f32>,
}

impl LogProbabilities {
    pub fn position(&self, residue: usize) -> &[f32] {
        &self.data[residue * ALPHABET.len()..(residue + 1) * ALPHABET.len()]
    }
}

/// Run the network: encoder, then decoder with no sequence information.
///
/// This is upstream's `unconditional_probs`. Its decoding-order mask is all zeros, which makes the
/// backward mask vanish and the forward mask reduce to the residue mask; every gather of the
/// sequence embedding is therefore multiplied by zero. Rather than build those tensors and
/// multiply them away, the zeroed terms are simply not constructed — the arithmetic is identical
/// and the sequence never enters.
pub fn forward(weights: &ProteinMpnnWeights, backbone: &Backbone) -> io::Result<LogProbabilities> {
    backbone.validate()?;
    let length = backbone.len();
    let graph = NeighborGraph::build(backbone, K_NEIGHBORS);
    let k = graph.k;

    // Features → edge embedding → LayerNorm → W_e.
    let raw = edge_features(backbone, &graph);
    let positional =
        weights
            .positional_embedding
            .forward(&slice_columns(&raw, 0, POSITIONAL_EMBEDDING_INPUT));
    let rbf_part = slice_columns(&raw, POSITIONAL_EMBEDDING_INPUT, raw.cols);
    let embedded = weights
        .edge_embedding
        .forward(&concat(&[&positional, &rbf_part]));
    let mut h_e = embedded;
    weights.norm_edges.forward_in_place(&mut h_e);
    let mut h_e = weights.w_e.forward(&h_e);

    // Node states start at zero: ProteinMPNN carries no per-residue input features, only edges.
    let mut h_v = Matrix::zeros(length, HIDDEN);

    for layer in &weights.encoder {
        // Node update: [h_V_i, h_E, h_V_j].
        let input = concat(&[&expand_nodes(&h_v, k), &h_e, &gather_nodes(&h_v, &graph)]);
        let messages = message_mlp(&input, &layer.w1, &layer.w2, &layer.w3);
        let delta = aggregate(&messages, length, k);
        add_in_place(&mut h_v, &delta);
        layer.norm1.forward_in_place(&mut h_v);
        let dense = layer.dense.forward(&h_v);
        add_in_place(&mut h_v, &dense);
        layer.norm2.forward_in_place(&mut h_v);

        // Edge update, using the freshly updated node states.
        let input = concat(&[&expand_nodes(&h_v, k), &h_e, &gather_nodes(&h_v, &graph)]);
        let messages = message_mlp(&input, &layer.w11, &layer.w12, &layer.w13);
        add_in_place(&mut h_e, &messages);
        layer.norm3.forward_in_place(&mut h_e);
    }

    // Decoder input: [h_V_i, h_E, zeros (the absent sequence embedding), h_V_j].
    let zeros = Matrix::zeros(length * k, HIDDEN);
    for layer in &weights.decoder {
        let input = concat(&[
            &expand_nodes(&h_v, k),
            &h_e,
            &zeros,
            &gather_nodes(&h_v, &graph),
        ]);
        let messages = message_mlp(&input, &layer.w1, &layer.w2, &layer.w3);
        let delta = aggregate(&messages, length, k);
        add_in_place(&mut h_v, &delta);
        layer.norm1.forward_in_place(&mut h_v);
        let dense = layer.dense.forward(&h_v);
        add_in_place(&mut h_v, &dense);
        layer.norm2.forward_in_place(&mut h_v);
    }

    let logits = weights.w_out.forward(&h_v);
    Ok(LogProbabilities {
        length,
        data: log_softmax_rows(&logits),
    })
}

fn slice_columns(matrix: &Matrix, start: usize, end: usize) -> Matrix {
    let cols = end - start;
    let mut out = Matrix::zeros(matrix.rows, cols);
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, slot)| slot.copy_from_slice(&matrix.row(row)[start..end]));
    out
}

/// Log-softmax, computed via the max-shift so that large logits cannot overflow the exponential.
fn log_softmax_rows(logits: &Matrix) -> Vec<f32> {
    let cols = logits.cols;
    let mut out = logits.data.clone();
    out.par_chunks_mut(cols).for_each(|row| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|value| (value - max).exp()).sum();
        let log_sum = max + sum.ln();
        for value in row.iter_mut() {
            *value -= log_sum;
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// Weight file
// ---------------------------------------------------------------------------------------------

/// Magic bytes of the converted weight format.
const MAGIC: &[u8; 4] = b"MCNN";
const FORMAT_VERSION: u32 = 1;

/// A flat name → tensor map, as `scripts/convert_mpnn_weights.py` writes it.
///
/// A converted file rather than the `.pt` directly: a PyTorch checkpoint is a zip of pickled
/// objects, and a pickle interpreter is both a large thing to write and a hazardous thing to point
/// at a downloaded file. Converting once, in the environment that already has Torch for the
/// LigandMPNN adapter, keeps this side to a length-prefixed read.
pub struct TensorFile {
    tensors: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl TensorFile {
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} is not a Molchanica weight file", path.display()),
            ));
        }
        let version = read_u32(&mut reader)?;
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "weight file version {version} is not supported (expected {FORMAT_VERSION}); \
                     re-run scripts/convert_mpnn_weights.py"
                ),
            ));
        }

        let count = read_u32(&mut reader)? as usize;
        let mut tensors = std::collections::HashMap::with_capacity(count);
        for _ in 0..count {
            let name_length = read_u32(&mut reader)? as usize;
            let mut name_bytes = vec![0u8; name_length];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

            let rank = read_u32(&mut reader)? as usize;
            let mut shape = Vec::with_capacity(rank);
            let mut elements = 1usize;
            for _ in 0..rank {
                let dimension = read_u32(&mut reader)? as usize;
                elements = elements.checked_mul(dimension).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "tensor shape overflows")
                })?;
                shape.push(dimension);
            }

            let mut bytes = vec![0u8; elements * 4];
            reader.read_exact(&mut bytes)?;
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            tensors.insert(name, (shape, values));
        }

        Ok(Self { tensors })
    }

    /// The synthetic backbone the converter ran upstream ProteinMPNN on.
    ///
    /// Stored in the same file as the reference outputs so the two cannot drift apart.
    pub fn reference_backbone(&self) -> io::Result<Backbone> {
        let coordinates = |name: &str| -> io::Result<Vec<[f32; 3]>> {
            let (shape, values) = self.take(name)?;
            if shape.len() != 2 || shape[1] != 3 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("'{name}' should be [length, 3]"),
                ));
            }
            Ok(values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
        };
        let integers = |name: &str| -> io::Result<Vec<i32>> {
            Ok(self
                .take(name)?
                .1
                .iter()
                .map(|value| *value as i32)
                .collect())
        };

        Ok(Backbone {
            n: coordinates("reference.N")?,
            ca: coordinates("reference.CA")?,
            c: coordinates("reference.C")?,
            o: coordinates("reference.O")?,
            residue_index: integers("reference.residue_idx")?,
            chain_index: integers("reference.chain_idx")?,
        })
    }

    /// The log-probabilities upstream produced for [`Self::reference_backbone`].
    pub fn reference_log_probs(&self) -> io::Result<Vec<f32>> {
        Ok(self.take("reference.log_probs")?.1.clone())
    }

    fn take(&self, name: &str) -> io::Result<&(Vec<usize>, Vec<f32>)> {
        self.tensors.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the weight file has no tensor named '{name}'"),
            )
        })
    }

    fn linear(&self, prefix: &str, expect_bias: bool) -> io::Result<Linear> {
        let (shape, weight) = self.take(&format!("{prefix}.weight"))?;
        if shape.len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "'{prefix}.weight' should be a matrix, but has rank {}",
                    shape.len()
                ),
            ));
        }
        let bias = if expect_bias {
            Some(self.take(&format!("{prefix}.bias"))?.1.clone())
        } else {
            None
        };
        Ok(Linear {
            out_features: shape[0],
            in_features: shape[1],
            weight: weight.clone(),
            bias,
        })
    }

    fn layer_norm(&self, prefix: &str) -> io::Result<LayerNorm> {
        Ok(LayerNorm {
            weight: self.take(&format!("{prefix}.weight"))?.1.clone(),
            bias: self.take(&format!("{prefix}.bias"))?.1.clone(),
        })
    }

    fn feed_forward(&self, prefix: &str) -> io::Result<FeedForward> {
        Ok(FeedForward {
            w_in: self.linear(&format!("{prefix}.W_in"), true)?,
            w_out: self.linear(&format!("{prefix}.W_out"), true)?,
        })
    }
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Load a converted ProteinMPNN checkpoint.
///
/// Tensor names are upstream's `state_dict` keys, so a checkpoint that has been renamed or
/// restructured fails here with the missing key named rather than producing silent nonsense.
pub fn load_weights(path: &Path) -> io::Result<ProteinMpnnWeights> {
    let file = TensorFile::load(path)?;

    let mut encoder = Vec::new();
    for index in 0.. {
        let prefix = format!("encoder_layers.{index}");
        if file.tensors.get(&format!("{prefix}.W1.weight")).is_none() {
            break;
        }
        encoder.push(EncoderLayer {
            w1: file.linear(&format!("{prefix}.W1"), true)?,
            w2: file.linear(&format!("{prefix}.W2"), true)?,
            w3: file.linear(&format!("{prefix}.W3"), true)?,
            w11: file.linear(&format!("{prefix}.W11"), true)?,
            w12: file.linear(&format!("{prefix}.W12"), true)?,
            w13: file.linear(&format!("{prefix}.W13"), true)?,
            norm1: file.layer_norm(&format!("{prefix}.norm1"))?,
            norm2: file.layer_norm(&format!("{prefix}.norm2"))?,
            norm3: file.layer_norm(&format!("{prefix}.norm3"))?,
            dense: file.feed_forward(&format!("{prefix}.dense"))?,
        });
    }

    let mut decoder = Vec::new();
    for index in 0.. {
        let prefix = format!("decoder_layers.{index}");
        if file.tensors.get(&format!("{prefix}.W1.weight")).is_none() {
            break;
        }
        decoder.push(DecoderLayer {
            w1: file.linear(&format!("{prefix}.W1"), true)?,
            w2: file.linear(&format!("{prefix}.W2"), true)?,
            w3: file.linear(&format!("{prefix}.W3"), true)?,
            norm1: file.layer_norm(&format!("{prefix}.norm1"))?,
            norm2: file.layer_norm(&format!("{prefix}.norm2"))?,
            dense: file.feed_forward(&format!("{prefix}.dense"))?,
        });
    }

    if encoder.is_empty() || decoder.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the weight file contains no encoder or decoder layers",
        ));
    }

    let weights = ProteinMpnnWeights {
        // `edge_embedding` is the one linear upstream declares with `bias=False`.
        edge_embedding: file.linear("features.edge_embedding", false)?,
        norm_edges: file.layer_norm("features.norm_edges")?,
        positional_embedding: file.linear("features.embeddings.linear", true)?,
        w_e: file.linear("W_e", true)?,
        encoder,
        decoder,
        w_out: file.linear("W_out", true)?,
    };
    validate_shapes(&weights)?;
    Ok(weights)
}

/// Check the loaded weights are the architecture this code implements.
///
/// Worth doing eagerly: a mismatched checkpoint would otherwise produce a shape panic deep inside
/// a parallel loop, or — worse, where dimensions happen to line up — plausible numbers.
fn validate_shapes(weights: &ProteinMpnnWeights) -> io::Result<()> {
    let expect = |what: &str, actual: usize, wanted: usize| -> io::Result<()> {
        if actual == wanted {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{what} is {actual}, but this implementation expects {wanted}"),
            ))
        }
    };

    expect(
        "the edge-embedding input width",
        weights.edge_embedding.in_features,
        POSITIONAL_EMBEDDING + NUM_ATOM_PAIRS * NUM_RBF,
    )?;
    expect(
        "the positional-embedding input width",
        weights.positional_embedding.in_features,
        POSITIONAL_EMBEDDING_INPUT,
    )?;
    expect(
        "the positional-embedding output width",
        weights.positional_embedding.out_features,
        POSITIONAL_EMBEDDING,
    )?;
    expect("the hidden width", weights.w_e.out_features, HIDDEN)?;
    expect(
        "the output alphabet size",
        weights.w_out.out_features,
        ALPHABET.len(),
    )?;
    // The encoder concatenates [node, edge, neighbour node]; the decoder additionally leaves room
    // for the sequence embedding this pass does not use.
    expect(
        "the encoder message input width",
        weights.encoder[0].w1.in_features,
        3 * HIDDEN,
    )?;
    expect(
        "the decoder message input width",
        weights.decoder[0].w1.in_features,
        4 * HIDDEN,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(
        out_features: usize,
        in_features: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Linear {
        Linear {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    #[test]
    fn linear_matches_the_pytorch_convention() {
        // weight is [out, in], so y0 = 1*x0 + 2*x1 + 10, y1 = 3*x0 + 4*x1 + 20.
        let layer = linear(2, 2, vec![1.0, 2.0, 3.0, 4.0], Some(vec![10.0, 20.0]));
        let mut input = Matrix::zeros(1, 2);
        input.data = vec![1.0, 1.0];

        let output = layer.forward(&input);
        assert_eq!(output.data, vec![13.0, 27.0]);
    }

    #[test]
    fn layer_norm_standardizes_then_rescales() {
        let norm = LayerNorm {
            weight: vec![1.0, 1.0, 1.0, 1.0],
            bias: vec![0.0; 4],
        };
        let mut matrix = Matrix::zeros(1, 4);
        matrix.data = vec![1.0, 2.0, 3.0, 4.0];
        norm.forward_in_place(&mut matrix);

        let mean: f32 = matrix.data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        // Unit variance, up to the epsilon inside the root.
        let variance: f32 = matrix.data.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((variance - 1.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_matches_reference_values() {
        let mut matrix = Matrix::zeros(1, 4);
        matrix.data = vec![-2.0, 0.0, 1.0, 3.0];
        gelu_in_place(&mut matrix);

        // Reference values from torch.nn.GELU().
        let expected = [-0.045_500, 0.0, 0.841_345, 2.995_950];
        for (actual, wanted) in matrix.data.iter().zip(expected) {
            assert!(
                (actual - wanted).abs() < 1e-4,
                "GELU gave {actual}, expected {wanted}"
            );
        }
    }

    #[test]
    fn rbf_peaks_at_its_own_centre() {
        let mut out = [0.0; NUM_RBF];
        // The first centre is exactly D_MIN, so a distance of 2 Å saturates it.
        rbf(2.0, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[NUM_RBF - 1] < 1e-6);

        // The last centre is exactly D_MAX.
        rbf(22.0, &mut out);
        assert!((out[NUM_RBF - 1] - 1.0).abs() < 1e-6);
        assert!(out[0] < 1e-6);
    }

    #[test]
    fn virtual_cb_sits_where_a_real_one_would() {
        // An idealized residue frame: the constructed Cβ should be roughly 1.5 Å from Cα, which is
        // the real Cα–Cβ bond length, and should not be collinear with the backbone.
        let backbone = Backbone {
            n: vec![[0.0, 0.0, 0.0]],
            ca: vec![[1.458, 0.0, 0.0]],
            c: vec![[2.009, 1.420, 0.0]],
            o: vec![[1.251, 2.390, 0.0]],
            residue_index: vec![1],
            chain_index: vec![0],
        };
        let cb = backbone.virtual_cb(0);
        let bond = distance(cb, backbone.ca[0]);
        assert!(
            (bond - 1.53).abs() < 0.1,
            "virtual Cβ is {bond} Å from Cα, which is not a bond length"
        );
        // A real Cβ is out of the N-Cα-C plane; this frame is in the z=0 plane, so it must not be.
        assert!(cb[2].abs() > 0.5);
    }

    #[test]
    fn neighbor_graph_puts_each_residue_first_and_is_deterministic() {
        let backbone = helix(6);
        let graph = NeighborGraph::build(&backbone, 3);

        for residue in 0..backbone.len() {
            // Distance to self is zero, so it always sorts first.
            assert_eq!(graph.neighbors(residue)[0], residue);
        }
        // Rebuilding gives the same graph; ties are broken by index rather than by sort order.
        let again = NeighborGraph::build(&backbone, 3);
        assert_eq!(graph.indices, again.indices);
    }

    #[test]
    fn positional_bucket_separates_chains_from_sequence_offsets() {
        let mut backbone = helix(3);
        backbone.chain_index = vec![0, 0, 1];
        let graph = NeighborGraph::build(&backbone, 3);
        let features = edge_features(&backbone, &graph);

        let bucket_of = |i: usize, j_slot: usize| -> usize {
            let row = features.row(i * graph.k + j_slot);
            row[..POSITIONAL_EMBEDDING_INPUT]
                .iter()
                .position(|value| *value == 1.0)
                .expect("exactly one positional bucket is set")
        };

        // A residue against itself is offset zero, which is the centre bucket.
        let self_slot = graph.neighbors(0).iter().position(|j| *j == 0).unwrap();
        assert_eq!(bucket_of(0, self_slot), MAX_RELATIVE as usize);

        // Residue 2 is on another chain, so it lands in the dedicated cross-chain bucket.
        let cross_slot = graph.neighbors(0).iter().position(|j| *j == 2).unwrap();
        assert_eq!(bucket_of(0, cross_slot), (2 * MAX_RELATIVE + 1) as usize);
    }

    #[test]
    fn edge_features_have_the_width_the_checkpoint_expects() {
        let backbone = helix(4);
        let graph = NeighborGraph::build(&backbone, 4);
        let features = edge_features(&backbone, &graph);

        assert_eq!(features.rows, 4 * 4);
        assert_eq!(
            features.cols,
            POSITIONAL_EMBEDDING_INPUT + NUM_ATOM_PAIRS * NUM_RBF
        );
        // 66 + 400 = 466 raw, which the two embeddings reduce to 16 + 400 = 416.
        assert_eq!(features.cols, 466);
        assert_eq!(POSITIONAL_EMBEDDING + NUM_ATOM_PAIRS * NUM_RBF, 416);
    }

    #[test]
    fn log_softmax_rows_sum_to_one_in_probability_space() {
        let mut logits = Matrix::zeros(2, 3);
        logits.data = vec![1.0, 2.0, 3.0, 1000.0, 1000.0, 1000.0];
        let log_probs = log_softmax_rows(&logits);

        let total: f32 = log_probs[..3].iter().map(|value| value.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5);

        // The second row would overflow the exponential without the max shift. It does not, but
        // subtracting two numbers near 1000 in f32 cancels away most of the mantissa, so the
        // tolerance here is looser — that is a property of single precision, and PyTorch's own
        // f32 log_softmax loses the same digits.
        let total: f32 = log_probs[3..].iter().map(|value| value.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "large-logit row summed to {total}"
        );
        assert!((log_probs[3] - (1.0f32 / 3.0).ln()).abs() < 1e-3);
    }

    #[test]
    fn concat_and_gather_line_up_with_the_neighbour_graph() {
        let backbone = helix(3);
        let graph = NeighborGraph::build(&backbone, 2);
        let mut nodes = Matrix::zeros(3, 2);
        nodes.data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let gathered = gather_nodes(&nodes, &graph);
        let expanded = expand_nodes(&nodes, 2);
        assert_eq!(gathered.rows, 6);
        // Every edge of residue 1 carries residue 1's own state on the expanded side...
        assert_eq!(expanded.row(2), [1.0, 1.0]);
        assert_eq!(expanded.row(3), [1.0, 1.0]);
        // ...and its neighbour's on the gathered side.
        let neighbor = graph.neighbors(1)[1];
        assert_eq!(gathered.row(3), nodes.row(neighbor));

        let joined = concat(&[&expanded, &gathered]);
        assert_eq!(joined.cols, 4);
        assert_eq!(
            joined.row(3),
            [1.0, 1.0, nodes.row(neighbor)[0], nodes.row(neighbor)[1]]
        );
    }

    #[test]
    fn aggregate_sums_then_divides_by_the_fixed_scale() {
        let mut messages = Matrix::zeros(4, 1);
        messages.data = vec![1.0, 2.0, 3.0, 4.0];
        let aggregated = aggregate(&messages, 2, 2);

        assert_eq!(aggregated.rows, 2);
        assert!((aggregated.data[0] - 3.0 / MESSAGE_SCALE).abs() < 1e-6);
        assert!((aggregated.data[1] - 7.0 / MESSAGE_SCALE).abs() < 1e-6);
    }

    #[test]
    fn rejects_backbones_that_cannot_be_scanned() {
        let mut backbone = helix(1);
        assert!(backbone.validate().is_err());

        backbone = helix(3);
        backbone.o.pop();
        assert!(backbone.validate().is_err());
    }

    /// A crude but geometrically sane α-helix, enough to exercise the graph and featurization.
    fn helix(length: usize) -> Backbone {
        let mut backbone = Backbone::default();
        for index in 0..length {
            let angle = index as f32 * 100.0f32.to_radians();
            let rise = index as f32 * 1.5;
            let ca = [2.3 * angle.cos(), 2.3 * angle.sin(), rise];
            backbone.n.push([ca[0] - 1.0, ca[1], ca[2] - 0.5]);
            backbone.ca.push(ca);
            backbone.c.push([ca[0] + 0.9, ca[1] + 0.9, ca[2] + 0.4]);
            backbone.o.push([ca[0] + 1.2, ca[1] + 2.0, ca[2] + 0.4]);
            backbone.residue_index.push(index as i32 + 1);
            backbone.chain_index.push(0);
        }
        backbone
    }
}
