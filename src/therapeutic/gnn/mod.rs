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

use std::{collections::HashMap, iter::repeat_n};

use bio_files::md_params::{DihedralParams, ForceFieldParams};
use na_seq::Element::{self, *};

use crate::{molecules::Atom, therapeutic::gnn::component::NUM_COMP_EDGE_GEOM_FEATS};

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

// Barrier height, cos(phase), sin(phase), periodicity sum, divider sum
const DIHEDRAL_PARAM_SUMMARY_FEATS: usize = 4;

// These are used to normalize dihedral properties.
const DIHEDRAL_BARRIER_REF: f32 = 4.0;
const DIHEDRAL_PERIODICITY_REF: f32 = 6.0;
const DIHEDRAL_DIVIDER_REF: f32 = 6.0;

pub(in crate::therapeutic) const GRAPH_ANALYSIS_FEATURE_VERSION: u8 = 2;

pub(in crate::therapeutic) mod atom_bond;
pub(in crate::therapeutic) mod component;
pub(in crate::therapeutic) mod spacial;

fn bond_pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
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
                // (self.divider_sum / self.effective_weight_sum) / DIHEDRAL_DIVIDER_REF,
            ]
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DihedralGeometryAccumulator {
    alignment_sum: f32,
    effective_weight_sum: f32,
}

impl DihedralGeometryAccumulator {
    fn add_terms(&mut self, phi: f32, params: &[DihedralParams]) {
        for param in params {
            let divider = param.divider.max(1) as f32;
            let weight = (param.barrier_height / divider).abs();
            if weight <= 0.0 {
                continue;
            }

            self.alignment_sum += weight * (param.periodicity as f32 * phi - param.phase).cos();
            self.effective_weight_sum += weight;
        }
    }

    fn summary(self) -> [f32; NUM_COMP_EDGE_GEOM_FEATS] {
        if self.effective_weight_sum <= 1.0e-6 {
            [0.0; NUM_COMP_EDGE_GEOM_FEATS]
        } else {
            [
                (self.alignment_sum / self.effective_weight_sum).clamp(-1.0, 1.0),
                (self.effective_weight_sum / DIHEDRAL_BARRIER_REF).ln_1p(),
            ]
        }
    }
}

pub(in crate::therapeutic) fn dihedral_angle(
    atoms: &[Atom],
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
) -> Option<f32> {
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

/// For component-edge features: aggregate actual torsion geometry for each central bond.
fn proper_dihedral_stats_by_central_bond(
    atoms: &[Atom],
    adj: &[Vec<usize>],
    ff_params: &ForceFieldParams,
) -> HashMap<(usize, usize), [f32; NUM_COMP_EDGE_GEOM_FEATS]> {
    let mut by_bond: HashMap<(usize, usize), DihedralGeometryAccumulator> = HashMap::new();

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

                    let Some(phi) = dihedral_angle(atoms, i0, i1, i2, i3) else {
                        continue;
                    };

                    by_bond
                        .entry(bond_pair_key(i1, i2))
                        .or_default()
                        .add_terms(phi, params);
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

    let mut out = vec![0.; max.pow(2) * num_feats];

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
    v.extend(repeat_n(0, max - n));
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
    v.extend(repeat_n(0., (max - n) * n_per));

    v
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

/// Maps element to values the neural net can use.
pub(in crate::therapeutic) fn vocab_lookup_element(el: Element) -> i32 {
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
