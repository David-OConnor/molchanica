//! Graph neural nets: Represent molecules as graphs using several organization schemes. This includes
//! atoms as nodes and covalent bonds as edges; functional groups (components) as nodes and connections
//! between them as edges, etc. We are experimenting with different atom and node features.
//!
//! This includes atom and bond networks, and per-atom, per-bond, etc features. We are also
//! attempting to construct graphs from functional groups, pharmacophore features, and other
//! concepts. With and without geometry. (Distance/angles in space)
//!
//! We use *Graph classification or regression* as our primary technique in the set of Graph ML tools:
//! We view molecules as graphs, and infer properties of the entire graph. We may also implement
//! graph clustering later, for other applications like finding molecules similar to a target.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    iter::repeat_n,
};

use bio_files::{
    BondType,
    md_params::{DihedralParams, ForceFieldParams},
};

use crate::{
    molecules::Atom,
    therapeutic::train::{BOND_SIGMA_SQ, FF_BUCKETS},
};

// Degree (Number of edges incident to a node), partial charge, geometry (radius from molecular centroid, mean neighbor distance),
// is H-bond acceptor, is H-bond donor, in aromatic ring.
// Keep this in sync with `GraphDataAtom::new`.
pub(in crate::therapeutic) const PER_ATOM_SCALARS: usize = 7;

// Edge feature description for the Atom-as-node GNN.
// All multiplex layers share the same per-edge feature layout so the encoder MLP can
// process them as one batch. Layout (16 floats): (todo: QC if you want this, or if there's a better way)
//   [0..4)  layer one-hot [L0, L1, L2, L3] — tells the encoder which relation this edge
//           belongs to so it can specialize the meaning of the trailing scalars.
//   [4..8)  bond-type one-hot [single, double, triple, aromatic] — the bond order most
//           relevant to this edge (see per-layer notes below). Zero when no real bond
//           is associated with the edge (L1/L2 outer pairs).
//   [8]     rotatable-bond flag. Only populated on layer 0 covalent-bond edges.
//   [9..11) two layer-specific scalars (see below).
//   [11..16) dihedral-parameter summary:
//            [log_raw_barrier_sum, phase_cos_mean, phase_sin_mean,
//             periodicity_mean_norm, divider_mean_norm].
//
// Per-layer semantics (kept in sync with `GraphDataAtom::new`):
// - Layer 0 (covalent bonds, edge between bonded atoms a0-a1):
//     bond_type = (a0, a1) bond order; rotatable = whether (a0, a1) is rotatable;
//     scalars = [dr_norm, log_kb]; dihedral summary = proper torsions whose MIDDLE bond is
//     (a0, a1).
// - Layer 1 (valence angles a0-ctr-a1, edge between the OUTER 1-3 pair (a0, a1)):
//     bond_type = zero (no direct bond); rotatable = 0;
//     scalars = [signed_angle_deviation, log_k_theta]; dihedral summary = zero.
// - Layer 2 (proper dihedrals i0-i1-i2-i3, edge between the OUTER 1-4 pair (i0, i3)):
//     bond_type = (i1, i2) central-bond order (rotatable-bond chemistry);
//     rotatable = 0;
//     scalars = [torsion_alignment, log_effective_barrier_sum]; dihedral summary = raw
//     proper-dihedral params for that torsion.
// - Layer 3 (improper dihedrals at hub `ctr` with satellites, edge on each (ctr, sat)):
//     bond_type = (ctr, sat) bond order; rotatable = 0;
//     scalars = [torsion_alignment, log_effective_barrier_sum]; dihedral summary = raw
//     improper-dihedral params for that torsion.
//
const DIHEDRAL_PARAM_SUMMARY_FEATS: usize = 5;
const ATOM_GNN_EDGE_SHARED_FEATS: usize = 9;
const ATOM_GNN_EDGE_REL_SCALARS: usize = 2;

pub(in crate::therapeutic) const ATOM_GNN_PER_EDGE_FEATS_LAYER_0: usize =
    ATOM_GNN_EDGE_SHARED_FEATS + ATOM_GNN_EDGE_REL_SCALARS + DIHEDRAL_PARAM_SUMMARY_FEATS;
pub(in crate::therapeutic) const ATOM_GNN_PER_EDGE_FEATS_LAYER_1: usize =
    ATOM_GNN_PER_EDGE_FEATS_LAYER_0;
pub(in crate::therapeutic) const ATOM_GNN_PER_EDGE_FEATS_LAYER_2: usize =
    ATOM_GNN_PER_EDGE_FEATS_LAYER_0;
pub(in crate::therapeutic) const ATOM_GNN_PER_EDGE_FEATS_LAYER_3: usize =
    ATOM_GNN_PER_EDGE_FEATS_LAYER_0;
pub(in crate::therapeutic) const ATOM_GNN_EDGE_LAYERS: usize = 4;

// Degree, component size
// Keep this in sync with `GraphDataComponent::new`
pub(in crate::therapeutic) const PER_COMP_SCALARS: usize = 2;
// Keep this in sync with `GraphDataComponent::new`.
// [shared_atoms, rotatable, log_raw_barrier_sum, phase_cos_mean, phase_sin_mean,
//  periodicity_mean_norm, divider_mean_norm]
pub(in crate::therapeutic) const PER_EDGE_COMP_FEATS: usize = 2 + DIHEDRAL_PARAM_SUMMARY_FEATS;
// For bond-stretching force-field params, used in the atom-based GNN.
const KB_REF: f32 = 300.0;
const ANGLE_K_REF: f32 = 80.0;
const ANGLE_DIST_SCALE: f32 = 0.5;
const DIHEDRAL_BARRIER_REF: f32 = 4.0;
const DIHEDRAL_PERIODICITY_REF: f32 = 6.0;
const DIHEDRAL_DIVIDER_REF: f32 = 6.0;
const DEFAULT_BOND_ONE_HOT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
// Used on multiplex layers whose edges connect non-bonded atom pairs (1-3, 1-4 outer pairs).
const NO_BOND_ONE_HOT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

// Spacial (pharmacophore) GNN constants. Keep this in sync with (Where?)
// Node scalar features: [r_from_pharm_centroid, mean_pairwise_dist]. Keep these in sync with
// `GraphDataSpacial::new`.
pub(in crate::therapeutic) const PER_PHARM_SCALARS: usize = 2;
// Edge features: [scaled_dist, rbf_0, rbf_1, rbf_2, rbf_3]
pub(in crate::therapeutic) const PER_SPACIAL_EDGE_FEATS: usize = 5;
// Node type vocab: 0=pad, 1=HBondDonor, 2=HBondAcceptor, 3=Hydrophobic, 4=Aromatic
pub(in crate::therapeutic) const SPACIAL_VOCAB_SIZE: usize = 5;

// Tunable parameters for the spacial/pharmacophore GNN.
const SPACIAL_ADJ_SIGMA_SQ: f32 = 16.0; // sigma=4 Å for adjacency Gaussian
const SPACIAL_DIST_SCALE: f32 = 10.0; // Normalise raw distances to ~O(1)
const SPACIAL_RBF_SIGMA_SQ: f32 = 2.25; // sigma=1.5 Å for RBF basis functions
const SPACIAL_RBF_CENTERS: [f32; 4] = [2.0, 4.0, 6.0, 8.0]; // Å

const BOND_DIST_SPACIAL_SCALE: f32 = 0.15;
pub(in crate::therapeutic) const GRAPH_ANALYSIS_FEATURE_VERSION: u8 = 2;

pub(in crate::therapeutic) mod atom_bond;
pub(in crate::therapeutic) mod component;
pub(in crate::therapeutic) mod spacial;

fn atom_adj_i(layer: usize, i: usize, j: usize, n: usize) -> usize {
    layer * n * n + i * n + j
}

fn atom_edge_feats_i(layer: usize, i: usize, j: usize, k: usize, n: usize) -> usize {
    ((layer * n * n) + i * n + j) * ATOM_GNN_PER_EDGE_FEATS_LAYER_0 + k
}

fn bond_pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn bond_sn_pair_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn bond_type_one_hot(bond_type: BondType) -> [f32; 4] {
    match bond_type {
        BondType::Single => [1.0, 0.0, 0.0, 0.0],
        BondType::Double => [0.0, 1.0, 0.0, 0.0],
        BondType::Triple => [0.0, 0.0, 1.0, 0.0],
        BondType::Aromatic => [0.0, 0.0, 0.0, 1.0],
        _ => DEFAULT_BOND_ONE_HOT,
    }
}

fn relation_edge_features(
    layer: usize,
    bond_type_one_hot: [f32; 4],
    rotatable: f32,
    scalar_0: f32,
    scalar_1: f32,
    dihedral_summary: [f32; DIHEDRAL_PARAM_SUMMARY_FEATS],
) -> [f32; ATOM_GNN_PER_EDGE_FEATS_LAYER_0] {
    debug_assert!(layer < ATOM_GNN_EDGE_LAYERS);
    let mut layer_one_hot = [0.0f32; ATOM_GNN_EDGE_LAYERS];
    layer_one_hot[layer] = 1.0;
    [
        layer_one_hot[0],
        layer_one_hot[1],
        layer_one_hot[2],
        layer_one_hot[3],
        bond_type_one_hot[0],
        bond_type_one_hot[1],
        bond_type_one_hot[2],
        bond_type_one_hot[3],
        rotatable,
        scalar_0,
        scalar_1,
        dihedral_summary[0],
        dihedral_summary[1],
        dihedral_summary[2],
        dihedral_summary[3],
        dihedral_summary[4],
    ]
}

fn append_relation_edge(
    adj_layers: &mut [f32],
    edge_feats: &mut [f32],
    edge_feat_counts: &mut [usize],
    layer: usize,
    i: usize,
    j: usize,
    num_atoms: usize,
    weight: f32,
    features: &[f32; ATOM_GNN_PER_EDGE_FEATS_LAYER_0],
) {
    adj_layers[atom_adj_i(layer, i, j, num_atoms)] += weight;
    edge_feat_counts[atom_adj_i(layer, i, j, num_atoms)] += 1;

    for (k, &value) in features.iter().enumerate() {
        edge_feats[atom_edge_feats_i(layer, i, j, k, num_atoms)] += value;
    }
}

/// Normalize per-edge accumulations by their multiplicity. Multiple angles or dihedrals
/// can share the same outer-pair edge (fused ring systems, multi-path 1-4 pairs), and
/// `append_relation_edge` sums both the adjacency weight and the edge features on each
/// hit. Averaging keeps the adjacency a stable Gaussian-distance kernel and keeps the
/// edge-feature scalars bounded; otherwise a fused-ring 1-3 corner would fire 2x or 3x
/// on layer 1 vs an isolated 1-3 corner. count == 0 (self-loops, padding) and count == 1
/// (single contribution) are left untouched.
fn finalize_relation_edges(
    adj_layers: &mut [f32],
    edge_feats: &mut [f32],
    edge_feat_counts: &[usize],
    num_atoms: usize,
) {
    for layer in 0..ATOM_GNN_EDGE_LAYERS {
        for i in 0..num_atoms {
            for j in 0..num_atoms {
                let count = edge_feat_counts[atom_adj_i(layer, i, j, num_atoms)];
                if count <= 1 {
                    continue;
                }

                let inv = 1.0 / count as f32;
                adj_layers[atom_adj_i(layer, i, j, num_atoms)] *= inv;
                for k in 0..ATOM_GNN_PER_EDGE_FEATS_LAYER_0 {
                    edge_feats[atom_edge_feats_i(layer, i, j, k, num_atoms)] *= inv;
                }
            }
        }
    }
}

fn bond_edge_weight(atoms: &[Atom], i: usize, j: usize) -> f32 {
    let diff = atoms[i].posit - atoms[j].posit;
    let dist_sq = diff.magnitude_squared() as f32;
    (-dist_sq / (2.0 * BOND_SIGMA_SQ)).exp()
}

fn valence_angle(atoms: &[Atom], a0: usize, ctr: usize, a1: usize) -> Option<f32> {
    let v0 = atoms[a0].posit - atoms[ctr].posit;
    let v1 = atoms[a1].posit - atoms[ctr].posit;
    let mag = v0.magnitude() * v1.magnitude();
    if mag <= 1.0e-12 {
        return None;
    }

    let cos_theta = (v0.dot(v1) / mag).clamp(-1.0, 1.0);
    Some(cos_theta.acos() as f32)
}

fn dihedral_angle(atoms: &[Atom], a0: usize, a1: usize, a2: usize, a3: usize) -> Option<f32> {
    let b0 = atoms[a1].posit - atoms[a0].posit;
    let b1 = atoms[a2].posit - atoms[a1].posit;
    let b2 = atoms[a3].posit - atoms[a2].posit;

    let n0 = b0.cross(b1);
    let n1 = b1.cross(b2);

    let n0_mag_sq = n0.magnitude_squared();
    let n1_mag_sq = n1.magnitude_squared();
    let b1_mag = b1.magnitude();

    if n0_mag_sq <= 1.0e-12 || n1_mag_sq <= 1.0e-12 || b1_mag <= 1.0e-12 {
        return None;
    }

    let b1_unit = b1 * (1.0 / b1_mag);
    let m1 = n0.cross(b1_unit);
    let x = n0.dot(n1);
    let y = m1.dot(n1);

    Some(y.atan2(x) as f32)
}

#[derive(Clone, Copy, Debug, Default)]
struct DihedralParamAccumulator {
    raw_barrier_sum: f32,
    effective_weight_sum: f32,
    phase_cos_sum: f32,
    phase_sin_sum: f32,
    periodicity_sum: f32,
    divider_sum: f32,
}

impl DihedralParamAccumulator {
    fn add_terms(&mut self, params: &[DihedralParams]) {
        for param in params {
            let divider = param.divider.max(1) as f32;
            let raw_barrier = param.barrier_height.abs();
            if raw_barrier <= 0.0 {
                continue;
            }

            let weight = raw_barrier / divider;
            self.raw_barrier_sum += raw_barrier;
            self.effective_weight_sum += weight;
            self.phase_cos_sum += weight * param.phase.cos();
            self.phase_sin_sum += weight * param.phase.sin();
            self.periodicity_sum += weight * param.periodicity as f32;
            self.divider_sum += weight * divider;
        }
    }

    fn summary(self) -> [f32; DIHEDRAL_PARAM_SUMMARY_FEATS] {
        if self.raw_barrier_sum <= 1.0e-6 || self.effective_weight_sum <= 1.0e-6 {
            [0.0; DIHEDRAL_PARAM_SUMMARY_FEATS]
        } else {
            [
                (self.raw_barrier_sum / DIHEDRAL_BARRIER_REF).ln_1p(),
                (self.phase_cos_sum / self.effective_weight_sum).clamp(-1.0, 1.0),
                (self.phase_sin_sum / self.effective_weight_sum).clamp(-1.0, 1.0),
                (self.periodicity_sum / self.effective_weight_sum) / DIHEDRAL_PERIODICITY_REF,
                (self.divider_sum / self.effective_weight_sum) / DIHEDRAL_DIVIDER_REF,
            ]
        }
    }
}

/// For atom/bond, and component GNNs.
fn proper_dihedral_summaries_by_central_bond(
    atoms: &[Atom],
    adj: &[Vec<usize>],
    ff_params: &ForceFieldParams,
) -> HashMap<(usize, usize), [f32; DIHEDRAL_PARAM_SUMMARY_FEATS]> {
    let mut by_bond: HashMap<(usize, usize), DihedralParamAccumulator> = HashMap::new();

    for (i1, neighbors) in adj.iter().enumerate() {
        for &i2 in neighbors {
            if i1 >= i2 {
                continue;
            }

            for &i0 in adj[i1].iter().filter(|&&x| x != i2) {
                for &i3 in adj[i2].iter().filter(|&&x| x != i1) {
                    if i0 == i3 {
                        continue;
                    }

                    let Some(params) = (match (
                        atoms[i0].force_field_type.clone(),
                        atoms[i1].force_field_type.clone(),
                        atoms[i2].force_field_type.clone(),
                        atoms[i3].force_field_type.clone(),
                    ) {
                        (Some(ff0), Some(ff1), Some(ff2), Some(ff3)) => {
                            ff_params.get_dihedral(&(ff0, ff1, ff2, ff3), true, true)
                        }
                        _ => None,
                    }) else {
                        continue;
                    };

                    by_bond
                        .entry(bond_pair_key(i1, i2))
                        .or_default()
                        .add_terms(params);
                }
            }
        }
    }

    by_bond
        .into_iter()
        .map(|(bond, acc)| (bond, acc.summary()))
        .collect()
}

pub(in crate::therapeutic) fn pad_adj_and_mask(
    raw_adj: &[f32],
    num_atoms: usize,
    max: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(max);

    // Mask: 1.0 for atoms, 0.0 for pad
    let mut p_mask = Vec::with_capacity(max);
    p_mask.extend(repeat_n(1.0, n));
    p_mask.extend(repeat_n(0.0, max - n));

    //Adj: Reconstruct row-by-row to handle 2D padding
    let mut p_adj = Vec::with_capacity(max * max);
    for r in 0..n {
        let row_start = r * num_atoms; // Input is flat [num_atoms * num_atoms]
        // Copy valid columns
        p_adj.extend_from_slice(&raw_adj[row_start..row_start + n]);
        // Pad columns (right side of matrix)
        p_adj.extend(repeat_n(0.0, max - n));
    }
    // Pad rows (bottom of matrix)
    let remaining_rows = max - n;
    p_adj.extend(repeat_n(0.0, remaining_rows * max));

    (p_adj, p_mask)
}

/// For atom/bond, and component GNNs.
pub(in crate::therapeutic) fn pad_edge_feats(
    edge_feats: &[f32], // [num_atoms*num_atoms*PER_EDGE_FEATS]
    num_atoms: usize,
    num_feats: usize,
    max: usize,
) -> Vec<f32> {
    let n = num_atoms.min(max);

    let mut out = vec![0.0f32; max.pow(2) * num_feats];

    for i in 0..n {
        for j in 0..n {
            let src_base = (i * num_atoms + j) * num_feats;
            let dst_base = (i * max + j) * num_feats;
            out[dst_base..dst_base + num_feats]
                .copy_from_slice(&edge_feats[src_base..src_base + num_feats]);
        }
    }

    out
}

/// Pad a 1-D index vector to `max`, filling the tail with `0` (the padding token).
pub(in crate::therapeutic) fn pad_indices(src: &[i32], num: usize, max: usize) -> Vec<i32> {
    let n = num.min(max);
    let mut v = Vec::with_capacity(max);
    v.extend_from_slice(&src[0..n]);
    v.extend(repeat_n(0_i32, max - n));
    v
}

/// Pad a per-node scalar block (`num × n_per` flat) to `max × n_per`.
pub(in crate::therapeutic) fn pad_scalars(
    src: &[f32],
    num: usize,
    n_per: usize,
    max: usize,
) -> Vec<f32> {
    let n = num.min(max);
    let mut v = Vec::with_capacity(max * n_per);
    v.extend_from_slice(&src[0..n * n_per]);
    v.extend(repeat_n(0.0_f32, (max - n) * n_per));
    v
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

#[cfg(test)]
mod tests {
    use crate::therapeutic::non_nn_ml::*;

    #[test]
    fn atom_graph_analysis_feature_count_matches_names() {
        let tools = atom_graph_analysis_tools();
        assert_eq!(tools.feature_dim(), tools.feature_names().len());
    }

    #[test]
    fn graphlets_detect_triangle() {
        let triangle = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let feats = graphlet_size_3_features(&triangle);

        assert_eq!(feats[0], 0.0);
        assert_eq!(feats[1], 1.0);
        assert_eq!(feats[2], 1.0);
    }

    #[test]
    fn path_and_overlap_features_detect_chain_structure() {
        let chain = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];

        let path_feats = path_based_features(&chain);
        assert_eq!(path_feats[0], 1.0);
        assert!(path_feats[2] >= 0.75);

        let overlap_feats = local_overlap_features(&chain);
        assert_eq!(overlap_feats[0], 0.0);
        assert_eq!(overlap_feats[1], 0.0);
    }

    #[test]
    fn lhn_features_capture_degree_normalized_shared_neighbors() {
        let triangle = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let chain = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];

        let triangle_feats = lhn_similarity_features(&triangle);
        let chain_feats = lhn_similarity_features(&chain);

        assert!(triangle_feats[0] > chain_feats[0]);
        assert!(triangle_feats[2] > 0.0);
        assert_eq!(chain_feats[2], 0.0);
    }

    #[test]
    fn spacial_analysis_graph_is_not_complete_when_distance_structure_is_sparse() {
        let dist_mat = vec![
            0.0, 2.0, 9.0, 9.0, //
            2.0, 0.0, 2.0, 9.0, //
            9.0, 2.0, 0.0, 2.0, //
            9.0, 9.0, 2.0, 0.0, //
        ];
        let adj = build_spacial_analysis_adj(&dist_mat, 4);
        let edge_count = adj.iter().map(Vec::len).sum::<usize>() / 2;

        assert!(edge_count < 6);
        assert!(adj[1].contains(&2));
        assert_eq!(
            spacial_graph_analysis_tools().feature_dim(),
            atom_graph_analysis_tools().feature_dim()
        );
    }
}
