//! This is for code specific to our atom-bond GNN. Layer 0 is a traditional atom/bond molecule
//! representation. (and one of the first applications of GNNs!) Layers 1-3 include bond
//! relations of multiple covalent bonds, using principles from MD.

use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    io,
    iter::repeat_n,
};

use bio_files::{
    BondType,
    md_params::{DihedralParams, ForceFieldParams},
};
use na_seq::{
    Element,
    Element::{
        Bromine, Carbon, Chlorine, Fluorine, Hydrogen, Iodine, Nitrogen, Oxygen, Phosphorus, Sulfur,
    },
};

use crate::{
    molecules::{Atom, build_adjacency_list, small::MoleculeSmall},
    therapeutic::{
        gnn,
        gnn::{
            ATOM_GNN_EDGE_LAYERS, ATOM_GNN_PER_EDGE_FEATS_LAYER_0, DIHEDRAL_BARRIER_REF,
            DIHEDRAL_PARAM_SUMMARY_FEATS, DihedralParamAccumulator, PER_ATOM_SCALARS,
        },
        non_nn_ml,
        non_nn_ml::GnnAnalysisTools,
        train::FF_BUCKETS,
    },
};

const ATOM_GNN_PER_EDGE_FEATS_LAYER_1: usize = ATOM_GNN_PER_EDGE_FEATS_LAYER_0;
const ATOM_GNN_PER_EDGE_FEATS_LAYER_2: usize = ATOM_GNN_PER_EDGE_FEATS_LAYER_0;
const ATOM_GNN_PER_EDGE_FEATS_LAYER_3: usize = ATOM_GNN_PER_EDGE_FEATS_LAYER_0;

// For bond-stretching force-field params.
const KB_REF: f32 = 300.0;
const ANGLE_K_REF: f32 = 80.0;
const ANGLE_DIST_SCALE: f32 = 0.5;
const DEFAULT_BOND_ONE_HOT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
// Used on multiplex layers whose edges connect non-bonded atom pairs (1-3, 1-4 outer pairs).
const NO_BOND_ONE_HOT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const BOND_DIST_SPACIAL_SCALE: f32 = 0.15;

// Note: Excluding H or not appears not to make any notable difference at first;
// Experiment with this more later.
const EXCLUDE_HYDROGEN: bool = true;

const BOND_SIGMA_SQ: f32 = 3.3; // Å. Try 1.5 - 2.2 for sigma, (Square it)

fn atom_adj_i(layer: usize, i: usize, j: usize, n: usize) -> usize {
    layer * n * n + i * n + j
}

fn atom_edge_feats_i(layer: usize, i: usize, j: usize, k: usize, n: usize) -> usize {
    ((layer * n * n) + i * n + j) * ATOM_GNN_PER_EDGE_FEATS_LAYER_0 + k
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

/// State for our atom-and-bond-based neural network. Atoms are nodes. 4 edge layers:
/// - covalent bonds (1 edge connects 2 nodes)
/// - Valence angles (2 edges connect 3 nodes)
/// - Proper dihedral angles (3 edges connect 4 nodes in a line)
/// - Improper dihedral angles (3 edges connect 4 nodes in a hub-and-spoke pattern)
///
/// Graph properties
/// ----------------
/// Edges: Multiplex atom graph with 4 undirected edge layers over the same node set:
///   covalent bonds, valence angles, proper dihedrals, and improper dihedrals. Each layer uses
///   symmetric adjacency weights from a Gaussian distance kernel over the participating atom pairs.
/// Underlying chemical graph has self-connections: No
/// GNN message-passing adjacency has self-loops: Yes (`adj[layer0, i, i] = 1.0` is added explicitly)
/// Nodes have multiple types (heterogeneous): No in the formal hetero-GNN sense; this is a
///   homogeneous atom graph, with element and FF bucket encoded as node embeddings/features
/// Edges have multiple types: Yes (single/double/triple/aromatic), represented as categorical
///   edge features; the adjacency only marks bonded neighbors and geometric proximity
/// Multipartite (edges can connect only to nodes of a diff type): No
/// Multiplex (Edges exist in layers; nodes are on all layers): Yes: Layers include for each node:
/// - Covalent bonds
/// - "Valence angles"/"angle bending" of two covalent bonds the node is a member of
/// - Proper dihedral angle of 3 linear covalent bonds the node is a member of
/// - Improper dihedral angle of 3 hub+spoke covalent bonds the node is a member of
/// Heterophily (Nodes are preferentially connected to others which have diff labels): Not fixed
///   by the construction; molecules can exhibit both homophilic and heterophilic local chemistry
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataAtom {
    /// Assign each element (Which we reasonably expect to encounter) an integer
    /// assignment to comply with the neural net's input requirements.
    pub elem_indices: Vec<i32>,
    /// An integer assignment for each force field type.
    pub ff_indices: Vec<i32>,
    /// Per-atom scalars:
    /// [degree, partial_charge, r_from_mol_centroid, mean_neighbor_dist,
    ///  is_h_bond_acceptor, is_h_bond_donor, in_aromatic_ring]
    pub scalars: Vec<f32>,
    /// Graph-level analysis features assembled according to
    /// `GnnAnalysisTools::feature_names()`.
    pub analysis_features: Vec<f32>,
    /// Flattened as `[layer, atom_i, atom_j]`, layer-major.
    pub adj: Vec<f32>,
    /// Flattened as `[layer, atom_i, atom_j, feat_k]`, layer-major.
    pub edge_feats: Vec<f32>,
    pub num_atoms: usize,
}

impl GraphDataAtom {
    /// Converts raw Atoms and Bonds into Flat vectors for Tensors.
    /// Used by both Training and Inference.
    pub(in crate::therapeutic) fn new(
        mol: &MoleculeSmall,
        ff_params: &ForceFieldParams,
        analysis_tools: &GnnAnalysisTools,
    ) -> io::Result<Self> {
        let (atoms, bonds, adj, old_to_new) = if EXCLUDE_HYDROGEN {
            let a: Vec<_> = mol
                .common
                .atoms
                .iter()
                .filter(|a| a.element != Hydrogen)
                .cloned()
                .collect();

            let mut sn_to_new = HashMap::with_capacity(a.len());
            let mut old_to_new = vec![None; mol.common.atoms.len()];
            for (new_i, a) in a.iter().enumerate() {
                sn_to_new.insert(a.serial_number, new_i);
            }
            for (old_i, atom) in mol.common.atoms.iter().enumerate() {
                old_to_new[old_i] = sn_to_new.get(&atom.serial_number).copied();
            }

            let mut bonds_ = Vec::new();
            for b in mol.common.bonds.iter() {
                if let (Some(&u), Some(&v)) =
                    (sn_to_new.get(&b.atom_0_sn), sn_to_new.get(&b.atom_1_sn))
                {
                    let mut b2 = b.clone();
                    b2.atom_0 = u;
                    b2.atom_1 = v;
                    bonds_.push(b2);
                }
            }
            let adj = build_adjacency_list(&bonds_, a.len());
            (a, bonds_, adj, old_to_new)
        } else {
            (
                mol.common.atoms.clone(),
                mol.common.bonds.clone(),
                mol.common.adjacency_list.clone(),
                (0..mol.common.atoms.len()).map(Some).collect(),
            )
        };

        let num_atoms = atoms.len();
        if num_atoms == 0 {
            return Err(io::Error::other("Molecule has 0 atoms"));
        }

        let Some(char) = &mol.characterization else {
            eprintln!("Missing char");
            return Err(io::Error::other("Missing characterization"));
        };
        let rotatable_bond_keys: HashSet<_> = char
            .rotatable_bonds
            .iter()
            .filter_map(|rot_bond| {
                mol.common
                    .bonds
                    .get(rot_bond.bond_i)
                    .map(|bond| bond_sn_pair_key(bond.atom_0_sn, bond.atom_1_sn))
            })
            .collect();

        let mut is_h_bond_acceptor = vec![false; num_atoms];
        for &old_i in &char.h_bond_acceptor {
            if let Some(Some(new_i)) = old_to_new.get(old_i) {
                is_h_bond_acceptor[*new_i] = true;
            }
        }

        let mut is_h_bond_donor = vec![false; num_atoms];
        for &old_i in &char.h_bond_donor {
            if let Some(Some(new_i)) = old_to_new.get(old_i) {
                is_h_bond_donor[*new_i] = true;
            }
        }

        let mut is_aromatic_ring = vec![false; num_atoms];
        for ring in &char.rings {
            for &old_i in &ring.atoms {
                if let Some(Some(new_i)) = old_to_new.get(old_i) {
                    is_aromatic_ring[*new_i] = true;
                }
            }
        }

        // Node features (Indices and Scalars)
        let mut elem_indices = Vec::with_capacity(num_atoms);
        let mut ff_indices = Vec::with_capacity(num_atoms);

        let mut scalars = Vec::with_capacity(num_atoms * PER_ATOM_SCALARS);

        let geom = atom_geom_scalars(&atoms, &adj);

        for (i, atom) in atoms.iter().enumerate() {
            elem_indices.push(vocab_lookup_element(atom.element));
            ff_indices.push(vocab_lookup_ff(atom.force_field_type.as_ref()));

            // Degree is the number of edges incident to a node.
            let degree = adj.get(i).map(|n| n.len()).unwrap_or(0);
            scalars.push(degree as f32 / 6.0);

            // Note: Including partial charge and FF type appears to be beneficial.
            scalars.push(atom.partial_charge.unwrap_or(0.0));

            // if let Some(lj_data) = ff_params
            //     .lennard_jones
            //     .get(atom.force_field_type.as_ref().unwrap())
            // {
            //     scalars.push(lj_data.sigma);
            //     scalars.push(lj_data.eps);
            // } else {
            //     eprintln!("Missing LJ for FF type {:?}", atom.force_field_type);
            //
            //     scalars.push(0.);
            //     scalars.push(0.);
            // }

            // scalars.push(ff_params.lennard_jones[atom.force_field_type.as_ref().unwrap()].sigma);
            // scalars.push(ff_params.lennard_jones[atom.force_field_type.as_ref().unwrap()].eps);

            let (r, mean_nb_dist) = geom[i];
            scalars.push(r);
            scalars.push(mean_nb_dist);

            let h_bond_acc = if is_h_bond_acceptor[i] { 1. } else { 0. };
            let h_bond_donor = if is_h_bond_donor[i] { 1. } else { 0. };
            scalars.push(h_bond_acc);
            scalars.push(h_bond_donor);

            let in_aromatic_ring = if is_aromatic_ring[i] { 1. } else { 0. };
            scalars.push(in_aromatic_ring);
        }

        let analysis_features = non_nn_ml::atom_graph_analysis_features(
            analysis_tools,
            &atoms,
            &adj,
            &is_h_bond_acceptor,
            &is_h_bond_donor,
            &is_aromatic_ring,
        );

        // Layered edge features and weighted adjacency.
        debug_assert_eq!(
            ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
            ATOM_GNN_PER_EDGE_FEATS_LAYER_1
        );
        debug_assert_eq!(
            ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
            ATOM_GNN_PER_EDGE_FEATS_LAYER_2
        );
        debug_assert_eq!(
            ATOM_GNN_PER_EDGE_FEATS_LAYER_0,
            ATOM_GNN_PER_EDGE_FEATS_LAYER_3
        );

        let proper_dihedral_summary_by_central_bond =
            gnn::proper_dihedral_summaries_by_central_bond(&atoms, &adj, ff_params);

        let n_atoms_sq = num_atoms.pow(2);
        let mut adj_layers = vec![0.; ATOM_GNN_EDGE_LAYERS * n_atoms_sq];
        let mut edge_feats =
            vec![0.; ATOM_GNN_EDGE_LAYERS * n_atoms_sq * ATOM_GNN_PER_EDGE_FEATS_LAYER_0];
        let mut edge_feat_counts = vec![0usize; ATOM_GNN_EDGE_LAYERS * n_atoms_sq];
        let mut bond_type_by_pair = HashMap::with_capacity(bonds.len());

        // Add explicit self-loops on every multiplex layer. Edge features at the self-loop
        // stay zero (the layer one-hot is also zero, signaling "self / no relation"), so
        // the encoder treats the self-message as a pure identity contribution. Without
        // self-loops on L1-L3, ablating L0 would leave those layers without any
        // self-message during message passing.
        for layer in 0..ATOM_GNN_EDGE_LAYERS {
            for i in 0..num_atoms {
                adj_layers[atom_adj_i(layer, i, i, num_atoms)] = 1.0;
            }
        }

        for bond in &bonds {
            let a0 = bond.atom_0;
            let a1 = bond.atom_1;
            if a0 >= num_atoms || a1 >= num_atoms {
                continue;
            }

            let bond_type_one_hot = bond_type_one_hot(bond.bond_type);
            bond_type_by_pair.insert(gnn::bond_pair_key(a0, a1), bond_type_one_hot);
            let is_rotatable =
                rotatable_bond_keys.contains(&bond_sn_pair_key(bond.atom_0_sn, bond.atom_1_sn));

            let p1 = atoms[a0].posit;
            let p2 = atoms[a1].posit;
            let dist_sq = (p1 - p2).magnitude_squared() as f32;
            let dist = dist_sq.sqrt();

            let ff0 = atoms[a0].force_field_type.clone().unwrap();
            let ff1 = atoms[a1].force_field_type.clone().unwrap();
            let bond_stretching = ff_params.get_bond(&(ff0.clone(), ff1.clone()), true);

            let (dr_norm, log_kb) = if let Some(v) = bond_stretching {
                let dr = ((dist - v.r_0) / BOND_DIST_SPACIAL_SCALE).clamp(-5.0, 5.0);
                let log_kb = (v.k_b / KB_REF).ln_1p();
                (dr, log_kb)
            } else {
                // Try a small set of coarse fallbacks; if none hit, leave the edge
                // features as zero rather than panicking.
                let candidates: &[(&str, &str)] = if (&ff0 == "cc" && ff1.starts_with("n"))
                    || (&ff1 == "cc" && ff0.starts_with("n"))
                {
                    &[("cc", "n4"), ("cc", "n")]
                } else if (&ff0 == "cg" && ff1.starts_with("c"))
                    || (&ff1 == "cg" && ff0.starts_with("c"))
                {
                    &[("cg", "cg"), ("cc", "cc")]
                } else if atoms[a0].element == Carbon && atoms[a1].element == Carbon {
                    &[("cc", "cc")]
                } else if atoms[a0].element == Nitrogen && atoms[a1].element == Nitrogen {
                    &[("n", "n")]
                } else {
                    &[]
                };

                let mut found = None;
                for (a, b) in candidates {
                    if let Some(v) = ff_params.get_bond(&((*a).to_owned(), (*b).to_owned()), false)
                    {
                        found = Some(v);
                        break;
                    }
                }

                if let Some(v) = found {
                    let dr = ((dist - v.r_0) / BOND_DIST_SPACIAL_SCALE).clamp(-5.0, 5.0);
                    let log_kb = (v.k_b / KB_REF).ln_1p();
                    (dr, log_kb)
                } else {
                    eprintln!(
                        "Missing bond stretching and no fallback for bond {bond:?}. \nAtoms {ff0} | {ff1}. Using zero bond-stretch edge features.",
                    );
                    (0.0, 0.0)
                }
            };

            let features = relation_edge_features(
                0,
                bond_type_one_hot,
                if is_rotatable { 1.0 } else { 0.0 },
                dr_norm,
                log_kb,
                proper_dihedral_summary_by_central_bond
                    .get(&gnn::bond_pair_key(a0, a1))
                    .copied()
                    .unwrap_or([0.0; DIHEDRAL_PARAM_SUMMARY_FEATS]),
            );
            let weight = bond_edge_weight(&atoms, a0, a1);

            append_relation_edge(
                &mut adj_layers,
                &mut edge_feats,
                &mut edge_feat_counts,
                0,
                a0,
                a1,
                num_atoms,
                weight,
                &features,
            );
            append_relation_edge(
                &mut adj_layers,
                &mut edge_feats,
                &mut edge_feat_counts,
                0,
                a1,
                a0,
                num_atoms,
                weight,
                &features,
            );
        }

        // Layer 1: valence-angle layer. For each angle a0-ctr-a1 we deposit ONE edge on
        // the OUTER 1-3 pair (a0, a1), not on the participating bonds. This makes layer 1
        // genuinely add new connectivity (1-3 reach) instead of relabeling layer-0 edges,
        // which is the actual point of multiplex.
        for (ctr, neighbors) in adj.iter().enumerate() {
            if neighbors.len() < 2 {
                continue;
            }

            for left in 0..neighbors.len() - 1 {
                for right in left + 1..neighbors.len() {
                    let a0 = neighbors[left];
                    let a1 = neighbors[right];

                    let Some(theta) = valence_angle(&atoms, a0, ctr, a1) else {
                        continue;
                    };

                    let angle_stats = match (
                        atoms[a0].force_field_type.clone(),
                        atoms[ctr].force_field_type.clone(),
                        atoms[a1].force_field_type.clone(),
                    ) {
                        (Some(ff0), Some(ff_ctr), Some(ff1)) => ff_params
                            .get_valence_angle(&(ff0, ff_ctr, ff1), true)
                            .map(|params| {
                                (
                                    ((theta - params.theta_0) / ANGLE_DIST_SCALE).clamp(-5.0, 5.0),
                                    (params.k / ANGLE_K_REF).ln_1p(),
                                )
                            })
                            .unwrap_or((0.0, 0.0)),
                        _ => (0.0, 0.0),
                    };

                    let features = relation_edge_features(
                        1,
                        NO_BOND_ONE_HOT,
                        0.0,
                        angle_stats.0,
                        angle_stats.1,
                        [0.0; DIHEDRAL_PARAM_SUMMARY_FEATS],
                    );
                    let weight = bond_edge_weight(&atoms, a0, a1);

                    append_relation_edge(
                        &mut adj_layers,
                        &mut edge_feats,
                        &mut edge_feat_counts,
                        1,
                        a0,
                        a1,
                        num_atoms,
                        weight,
                        &features,
                    );
                    append_relation_edge(
                        &mut adj_layers,
                        &mut edge_feats,
                        &mut edge_feat_counts,
                        1,
                        a1,
                        a0,
                        num_atoms,
                        weight,
                        &features,
                    );
                }
            }
        }

        // Layer 2: proper-dihedral layer. For each torsion i0-i1-i2-i3 we deposit ONE edge
        // on the OUTER 1-4 pair (i0, i3). The bond-type slot encodes the central (i1, i2)
        // bond order, since rotational chemistry hinges on the central-bond order
        // (single = freely rotating, double = restricted, etc.). Iterating with i1 < i2 over
        // the central bond enumerates each proper torsion exactly once, so no seen-set is
        // needed.
        for (i1, neighbors) in adj.iter().enumerate() {
            for &i2 in neighbors {
                if i1 >= i2 {
                    continue;
                }

                let central_bond_type = bond_type_by_pair
                    .get(&gnn::bond_pair_key(i1, i2))
                    .copied()
                    .unwrap_or(DEFAULT_BOND_ONE_HOT);

                for &i0 in adj[i1].iter().filter(|&&x| x != i2) {
                    for &i3 in adj[i2].iter().filter(|&&x| x != i1) {
                        if i0 == i3 {
                            continue;
                        }

                        let Some(phi) = dihedral_angle(&atoms, i0, i1, i2, i3) else {
                            continue;
                        };

                        let (proper_stats, proper_param_summary) = match (
                            atoms[i0].force_field_type.clone(),
                            atoms[i1].force_field_type.clone(),
                            atoms[i2].force_field_type.clone(),
                            atoms[i3].force_field_type.clone(),
                        ) {
                            (Some(ff0), Some(ff1), Some(ff2), Some(ff3)) => ff_params
                                .get_dihedral(&(ff0, ff1, ff2, ff3), true, true)
                                .map(|params| {
                                    (
                                        dihedral_edge_stats(phi, params),
                                        dihedral_param_summary(params),
                                    )
                                })
                                .unwrap_or(((0.0, 0.0), [0.0; DIHEDRAL_PARAM_SUMMARY_FEATS])),
                            _ => ((0.0, 0.0), [0.0; DIHEDRAL_PARAM_SUMMARY_FEATS]),
                        };

                        let features = relation_edge_features(
                            2,
                            central_bond_type,
                            0.0,
                            proper_stats.0,
                            proper_stats.1,
                            proper_param_summary,
                        );
                        let weight = bond_edge_weight(&atoms, i0, i3);

                        append_relation_edge(
                            &mut adj_layers,
                            &mut edge_feats,
                            &mut edge_feat_counts,
                            2,
                            i0,
                            i3,
                            num_atoms,
                            weight,
                            &features,
                        );
                        append_relation_edge(
                            &mut adj_layers,
                            &mut edge_feats,
                            &mut edge_feat_counts,
                            2,
                            i3,
                            i0,
                            num_atoms,
                            weight,
                            &features,
                        );
                    }
                }
            }
        }

        // Layer 3: improper-dihedral layer. Improper torsions have no natural "outer" pair
        // (the geometry is hub-and-spoke), so edges stay on (ctr, sat) bonds — the bond
        // chemistry is preserved by the bond_type slot. The triple-nested iteration over
        // satellite indices a < b < d already enumerates each (ctr, sat0, sat1, sat2)
        // configuration exactly once, so no seen-set is needed.
        for (ctr, satellites) in adj.iter().enumerate() {
            if satellites.len() < 3 {
                continue;
            }

            for a in 0..satellites.len() - 2 {
                for b in a + 1..satellites.len() - 1 {
                    for d in b + 1..satellites.len() {
                        let sat0 = satellites[a];
                        let sat1 = satellites[b];
                        let sat2 = satellites[d];

                        let (Some(ff0), Some(ff1), Some(ff_ctr), Some(ff2)) = (
                            atoms[sat0].force_field_type.clone(),
                            atoms[sat1].force_field_type.clone(),
                            atoms[ctr].force_field_type.clone(),
                            atoms[sat2].force_field_type.clone(),
                        ) else {
                            continue;
                        };

                        let mut lookup_satellites = [(ff0, sat0), (ff1, sat1), (ff2, sat2)];
                        lookup_satellites
                            .sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

                        let key = (
                            lookup_satellites[0].0.clone(),
                            lookup_satellites[1].0.clone(),
                            ff_ctr,
                            lookup_satellites[2].0.clone(),
                        );

                        let Some(params) = ff_params.get_dihedral(&key, false, true) else {
                            continue;
                        };

                        let ordered_satellites = [
                            lookup_satellites[0].1,
                            lookup_satellites[1].1,
                            lookup_satellites[2].1,
                        ];

                        let Some(phi) = dihedral_angle(
                            &atoms,
                            ordered_satellites[0],
                            ordered_satellites[1],
                            ctr,
                            ordered_satellites[2],
                        ) else {
                            continue;
                        };

                        let improper_stats = dihedral_edge_stats(phi, params);
                        let improper_param_summary = dihedral_param_summary(params);

                        for &sat in &ordered_satellites {
                            let bond_type = bond_type_by_pair
                                .get(&gnn::bond_pair_key(ctr, sat))
                                .copied()
                                .unwrap_or(DEFAULT_BOND_ONE_HOT);
                            let features = relation_edge_features(
                                3,
                                bond_type,
                                0.0,
                                improper_stats.0,
                                improper_stats.1,
                                improper_param_summary,
                            );
                            let weight = bond_edge_weight(&atoms, ctr, sat);

                            append_relation_edge(
                                &mut adj_layers,
                                &mut edge_feats,
                                &mut edge_feat_counts,
                                3,
                                ctr,
                                sat,
                                num_atoms,
                                weight,
                                &features,
                            );
                            append_relation_edge(
                                &mut adj_layers,
                                &mut edge_feats,
                                &mut edge_feat_counts,
                                3,
                                sat,
                                ctr,
                                num_atoms,
                                weight,
                                &features,
                            );
                        }
                    }
                }
            }
        }

        finalize_relation_edges(
            &mut adj_layers,
            &mut edge_feats,
            &edge_feat_counts,
            num_atoms,
        );

        // Note: symmetric normalization (D^-1/2 A D^-1/2) is applied in the model's
        // forward pass *after* the learned edge gate, so we ship the raw weighted
        // multiplex adjacency here. Relation-specific chemistry lives in `edge_feats`.

        Ok(Self {
            elem_indices,
            ff_indices,
            scalars,
            analysis_features,
            adj: adj_layers,
            edge_feats,
            num_atoms,
        })
    }
}

/// These are numerical properties of individual atoms. Partial charge, FF type etc.
fn atom_geom_scalars(atoms: &[Atom], adj: &[Vec<usize>]) -> Vec<(f32, f32)> {
    let n = atoms.len().max(1);

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for a in atoms {
        cx += a.posit.x;
        cy += a.posit.y;
        cz += a.posit.z;
    }

    let inv_n = 1.0 / (n as f64);
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    // Scale by RMS radius to keep numbers ~O(1)
    let mut r2_sum = 0.0f64;
    for a in atoms {
        let dx = a.posit.x - cx;
        let dy = a.posit.y - cy;
        let dz = a.posit.z - cz;
        r2_sum += dx * dx + dy * dy + dz * dz;
    }

    let rms = (r2_sum * inv_n).sqrt().max(1e-6);

    let mut out = Vec::with_capacity(atoms.len());

    for (i, a) in atoms.iter().enumerate() {
        let dx = (a.posit.x - cx) / rms;
        let dy = (a.posit.y - cy) / rms;
        let dz = (a.posit.z - cz) / rms;

        let r = (dx * dx + dy * dy + dz * dz).sqrt(); // invariant

        // Mean neighbor distance
        let mut dist_sum = 0.0_f64;
        let mut count = 0_u32;
        for &j in adj.get(i).unwrap_or(&Vec::new()).iter() {
            let b = &atoms[j];
            dist_sum += (a.posit - b.posit).magnitude();
            count += 1;
        }
        let mean_nb_dist = if count > 0 {
            dist_sum as f32 / count as f32
        } else {
            0.0
        };

        out.push((r as f32, mean_nb_dist));
    }

    out
}

/// Helper: Pads a single graph to MAX_ATOMS.
/// Returns (PaddedNodes, PaddedAdj, PaddedMask) as flat vectors.
///
/// For comps too.
pub(in crate::therapeutic) fn pad_atom_adj_and_mask(
    raw_adj: &[f32],
    num_atoms: usize,
    max: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(max);

    let mut p_mask = Vec::with_capacity(max);
    p_mask.extend(repeat_n(1.0, n));
    p_mask.extend(repeat_n(0.0, max - n));

    let mut p_adj = vec![0.0f32; ATOM_GNN_EDGE_LAYERS * max * max];

    for layer in 0..ATOM_GNN_EDGE_LAYERS {
        for i in 0..n {
            let src_base = atom_adj_i(layer, i, 0, num_atoms);
            let dst_base = layer * max * max + i * max;
            p_adj[dst_base..dst_base + n].copy_from_slice(&raw_adj[src_base..src_base + n]);
        }
    }

    (p_adj, p_mask)
}

pub(in crate::therapeutic) fn pad_atom_edge_feats(
    edge_feats: &[f32], // [layers*num_atoms*num_atoms*PER_EDGE_FEATS]
    num_atoms: usize,
    max: usize,
) -> Vec<f32> {
    let n = num_atoms.min(max);

    let mut out = vec![0.0f32; ATOM_GNN_EDGE_LAYERS * max.pow(2) * ATOM_GNN_PER_EDGE_FEATS_LAYER_0];

    for layer in 0..ATOM_GNN_EDGE_LAYERS {
        for i in 0..n {
            for j in 0..n {
                let src_base = atom_edge_feats_i(layer, i, j, 0, num_atoms);
                let dst_base =
                    ((layer * max * max) + i * max + j) * ATOM_GNN_PER_EDGE_FEATS_LAYER_0;
                out[dst_base..dst_base + ATOM_GNN_PER_EDGE_FEATS_LAYER_0].copy_from_slice(
                    &edge_feats[src_base..src_base + ATOM_GNN_PER_EDGE_FEATS_LAYER_0],
                );
            }
        }
    }

    out
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

fn dihedral_edge_stats(phi: f32, params: &[DihedralParams]) -> (f32, f32) {
    let mut barrier_sum = 0.0;
    let mut alignment_sum = 0.0;

    for param in params {
        let divider = param.divider.max(1) as f32;
        let weight = (param.barrier_height / divider).abs();
        if weight <= 0.0 {
            continue;
        }

        let phase = param.phase;
        let periodicity = param.periodicity as f32;
        alignment_sum += weight * (periodicity * phi - phase).cos();
        barrier_sum += weight;
    }

    if barrier_sum <= 1.0e-6 {
        (0.0, 0.0)
    } else {
        (
            (alignment_sum / barrier_sum).clamp(-1.0, 1.0),
            (barrier_sum / DIHEDRAL_BARRIER_REF).ln_1p(),
        )
    }
}

fn dihedral_param_summary(params: &[DihedralParams]) -> [f32; DIHEDRAL_PARAM_SUMMARY_FEATS] {
    let mut acc = DihedralParamAccumulator::default();
    acc.add_terms(params);
    acc.summary()
}
