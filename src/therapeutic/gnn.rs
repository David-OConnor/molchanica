//! Related to the graph neural net. This includes atom and bond networks,
//! and per-atom, per-bond etc features.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    io,
    io::ErrorKind,
};

use bio_files::{BondType, md_params::ForceFieldParams};
use na_seq::{
    Element,
    Element::{
        Bromine, Carbon, Chlorine, Fluorine, Hydrogen, Iodine, Nitrogen, Oxygen, Phosphorus, Sulfur,
    },
};

use crate::{
    mol_components::{ComponentType, MolComponents, build_adjacency_list_conn},
    molecules::{Atom, build_adjacency_list, small::MoleculeSmall},
    therapeutic::train::{BOND_SIGMA_SQ, EXCLUDE_HYDROGEN, FF_BUCKETS, MAX_ATOMS},
};

// Degree, partial charge, FF name, element, is H-bond acceptor, is H-bond donor, in aromatic ring,
pub(in crate::therapeutic) const PER_ATOM_SCALARS: usize = 7;

// Degree
pub(in crate::therapeutic) const PER_COMP_SCALARS: usize = 2;

// Scaled/modified proxies for r_0, k_b
pub(in crate::therapeutic) const PER_EDGE_FEATS: usize = 2;

// Shared.
pub(in crate::therapeutic) const PER_EDGE_COMP_FEATS: usize = 1;

// Spatial (pharmacophore) GNN constants.
// Node scalar features: [r_from_pharm_centroid, mean_pairwise_dist]
pub(in crate::therapeutic) const PER_PHARM_SCALARS: usize = 2;
// Edge features: [scaled_dist, rbf_0, rbf_1, rbf_2, rbf_3]
pub(in crate::therapeutic) const PER_SPACIAL_EDGE_FEATS: usize = 5;
// Node type vocab: 0=pad, 1=HBondDonor, 2=HBondAcceptor, 3=Hydrophobic, 4=Aromatic
pub(in crate::therapeutic) const PHARM_VOCAB_SIZE: usize = 5;

const SPACIAL_ADJ_SIGMA_SQ: f32 = 16.0; // sigma=4 Å for adjacency Gaussian
const SPACIAL_DIST_SCALE: f32 = 10.0; // Normalise raw distances to ~O(1)
const SPACIAL_RBF_SIGMA_SQ: f32 = 2.25; // sigma=1.5 Å for RBF basis functions
const SPACIAL_RBF_CENTERS: [f32; 4] = [2.0, 4.0, 6.0, 8.0]; // Å

const BOND_DIST_PARAM_SCALE: f32 = 0.15;
const KB_REF: f32 = 300.0;

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphData {
    /// Assign each element (Which we reasonably expect to encounter) an integer
    /// assignment, to comply with the neural net's input requirements.
    pub elem_indices: Vec<i32>,
    /// An integer assignment for each force field type.
    pub ff_indices: Vec<i32>,
    /// Partial charge, element, is an H bond donors/acceptor etc.
    pub scalars: Vec<f32>,
    pub adj: Vec<f32>,
    pub edge_feats: Vec<f32>,
    pub num_atoms: usize,
}

/// Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
impl GraphData {
    pub fn new(mol: &MoleculeSmall, ff_params: &ForceFieldParams) -> io::Result<Self> {
        let (atoms, bonds, adj) = if EXCLUDE_HYDROGEN {
            let a: Vec<_> = mol
                .common
                .atoms
                .iter()
                .filter(|a| a.element != Hydrogen)
                .cloned()
                .collect();

            let mut sn_to_new = HashMap::with_capacity(a.len());
            for (new_i, a) in a.iter().enumerate() {
                sn_to_new.insert(a.serial_number, new_i);
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
            return Err(io::Error::other("Molecule has 0 atoms"));
        }

        // Node features (Indices and Scalars)
        let mut elem_indices = Vec::with_capacity(num_atoms);
        let mut ff_indices = Vec::with_capacity(num_atoms);

        let mut scalars = Vec::with_capacity(num_atoms * PER_ATOM_SCALARS);

        let geom = atom_geom_scalars(&atoms, &adj);

        for (i, atom) in atoms.iter().enumerate() {
            elem_indices.push(vocab_lookup_element(atom.element));
            ff_indices.push(vocab_lookup_ff(atom.force_field_type.as_ref()));

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

            let Some(char) = &mol.characterization else {
                eprintln!("Missing char");
                return Err(io::Error::new(ErrorKind::Other, "Missing characterization"));
            };

            let h_bond_acc = if char.h_bond_acceptor.contains(&i) {
                1.
            } else {
                0.
            };
            let h_bond_donor = if char.h_bond_donor.contains(&i) {
                1.
            } else {
                0.
            };
            scalars.push(h_bond_acc);
            scalars.push(h_bond_donor);

            let mut in_aromatic_ring = 0.;
            for ring in &char.rings {
                if ring.atoms.contains(&i) {
                    in_aromatic_ring = 1.;
                    break;
                }
            }

            scalars.push(in_aromatic_ring);
        }

        // Edge features (Weighted Adjacency)
        let n_atoms_sq = num_atoms.pow(2);
        let mut adj_list = vec![0.; n_atoms_sq];
        let mut edge_feats = vec![0.; n_atoms_sq * PER_EDGE_FEATS];

        let edge_feats_i =
            |i: usize, j: usize, k: usize, n: usize| -> usize { (i * n + j) * PER_EDGE_FEATS + k };

        // Self loops
        for i in 0..num_atoms {
            adj_list[i * num_atoms + i] = 1.0;
        }

        for bond in &bonds {
            let a0 = bond.atom_0;
            let a1 = bond.atom_1;
            if a0 >= num_atoms || a1 >= num_atoms {
                continue;
            }

            let bond_strength = match bond.bond_type {
                BondType::Single => 1.0,
                BondType::Double => 2.0,
                BondType::Triple => 3.0,
                BondType::Aromatic => 1.5,
                _ => 1.0,
            };

            // todo: Lennard Jones?

            let p1 = atoms[a0].posit;
            let p2 = atoms[a1].posit;
            let dist_sq = (p1 - p2).magnitude_squared() as f32;
            let dist = dist_sq.sqrt();

            {
                let ff0 = atoms[a0].force_field_type.clone().unwrap();
                let ff1 = atoms[a1].force_field_type.clone().unwrap();
                let bond_stretching = ff_params.get_bond(&(ff0.clone(), ff1.clone()), true);

                let bond_stretching = if let Some(v) = bond_stretching {
                    v
                } else {
                    // A coarse fallback.
                    let mut safe_fallback = ("cc".to_owned(), "n".to_owned());

                    if (&ff0 == "cc" && ff1.starts_with("n"))
                        || (&ff1 == "cc" && ff0.starts_with("n"))
                    {
                        safe_fallback = ("cc".to_owned(), "n4".to_owned());
                    } else if (&ff0 == "cg" && ff1.starts_with("c"))
                        || (&ff1 == "cg" && ff0.starts_with("c"))
                    {
                        safe_fallback = ("cg".to_owned(), "cg".to_owned());
                    } else if atoms[a0].element == Carbon && atoms[a1].element == Carbon {
                        safe_fallback = ("cc".to_owned(), "cc".to_owned());
                        eprintln!(
                            "Missing bond stretching for bond {bond:?}. \nAtoms {ff0} | {ff1}\n. Using a substitute.",
                        );
                    } else if atoms[a0].element == Nitrogen && atoms[a1].element == Nitrogen {
                        safe_fallback = ("n".to_owned(), "n".to_owned());

                        eprintln!(
                            "Missing bond stretching for bond {bond:?}. \nAtoms {ff0} | {ff1}\n. Using a substitute.",
                        );
                    }

                    ff_params.get_bond(&safe_fallback, false).unwrap()
                };

                let dr_norm =
                    ((dist - bond_stretching.r_0) / BOND_DIST_PARAM_SCALE).clamp(-5.0, 5.0);
                let log_kb = (bond_stretching.k_b / KB_REF).ln_1p();

                edge_feats[edge_feats_i(a0, a1, 0, num_atoms)] = dr_norm;
                edge_feats[edge_feats_i(a0, a1, 1, num_atoms)] = log_kb;

                edge_feats[edge_feats_i(a1, a0, 0, num_atoms)] = dr_norm;
                edge_feats[edge_feats_i(a1, a0, 1, num_atoms)] = log_kb;
            }

            let k = (-dist_sq / (2.0 * BOND_SIGMA_SQ)).exp();
            let weight = bond_strength * k;

            adj_list[a0 * num_atoms + a1] = weight;
            adj_list[a1 * num_atoms + a0] = weight;
        }

        // Symmetric Normalization: D^(-0.5) * A * D^(-0.5)
        let mut degrees_vec = vec![0.0f32; num_atoms];
        for i in 0..num_atoms {
            let mut d = 0.0;
            for j in 0..num_atoms {
                d += adj_list[i * num_atoms + j];
            }
            degrees_vec[i] = d;
        }
        for i in 0..num_atoms {
            let di_inv_sqrt = 1.0 / degrees_vec[i].max(1e-9).sqrt();
            for j in 0..num_atoms {
                let dj_inv_sqrt = 1.0 / degrees_vec[j].max(1e-9).sqrt();
                adj_list[i * num_atoms + j] *= di_inv_sqrt * dj_inv_sqrt;
            }
        }

        Ok(Self {
            elem_indices,
            ff_indices,
            scalars,
            adj: adj_list,
            edge_feats,
            num_atoms,
        })
    }
}

/// Instead of atoms and bonds, this operates on components. (Fgs etc)
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataComponent {
    pub comp_type_indices: Vec<i32>,
    pub scalars: Vec<f32>,
    pub adj: Vec<f32>,
    pub edge_feats: Vec<f32>,
    pub num_comps: usize,
}

/// Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
impl GraphDataComponent {
    pub fn new(mol_comps: &MolComponents) -> io::Result<Self> {
        let comps = &mol_comps.components;
        let conns = &mol_comps.connections;

        let adj = build_adjacency_list_conn(conns, comps.len());

        let num_comps = comps.len();
        if num_comps == 0 {
            return Err(io::Error::other("Molecule has 0 components"));
        }

        // Node features (Indices and Scalars)
        let mut comp_type_indices = Vec::with_capacity(num_comps);

        let mut scalars = Vec::with_capacity(num_comps * PER_COMP_SCALARS);

        for (i, comp) in comps.iter().enumerate() {
            comp_type_indices.push(vocab_lookup_component(&comp.comp_type));

            let degree = adj.get(i).map(|n| n.len()).unwrap_or(0);
            scalars.push(degree as f32 / 6.0);
            // Number of atoms owned by this component, normalised.
            scalars.push(comp.atoms.len() as f32 / 10.0);
        }

        // Conn features (Weighted Adjacency)
        let n_atoms_sq = num_comps.pow(2);
        let mut adj_list = vec![0.; n_atoms_sq];
        let mut edge_feats = vec![0.; n_atoms_sq * PER_EDGE_COMP_FEATS];

        let edge_feats_i = |i: usize, j: usize, k: usize, n: usize| -> usize {
            (i * n + j) * PER_EDGE_COMP_FEATS + k
        };

        // Self loops
        for i in 0..num_comps {
            adj_list[i * num_comps + i] = 1.0;
        }

        for conn in conns {
            let a0 = conn.comp_0;
            let a1 = conn.comp_1;
            if a0 >= num_comps || a1 >= num_comps {
                continue;
            }

            adj_list[a0 * num_comps + a1] = 1.0;
            adj_list[a1 * num_comps + a0] = 1.0;

            // An edge feature for if the connection uses shared atoms or not.
            let shared = if conn.shared_atoms { 1. } else { 0. };
            edge_feats[edge_feats_i(a0, a1, 0, num_comps)] = shared;
            edge_feats[edge_feats_i(a1, a0, 0, num_comps)] = shared;
        }

        // Symmetric Normalization: D^(-0.5) * A * D^(-0.5)
        let mut degrees_vec = vec![0.0f32; num_comps];
        for i in 0..num_comps {
            let mut d = 0.0;
            for j in 0..num_comps {
                d += adj_list[i * num_comps + j];
            }
            degrees_vec[i] = d;
        }
        for i in 0..num_comps {
            let di_inv_sqrt = 1.0 / degrees_vec[i].max(1e-9).sqrt();
            for j in 0..num_comps {
                let dj_inv_sqrt = 1.0 / degrees_vec[j].max(1e-9).sqrt();
                adj_list[i * num_comps + j] *= di_inv_sqrt * dj_inv_sqrt;
            }
        }

        Ok(Self {
            comp_type_indices,
            scalars,
            adj: adj_list,
            edge_feats,
            num_comps,
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
pub(in crate::therapeutic) fn pad_adj_and_mask(
    raw_adj: &[f32],
    num_atoms: usize,
    max: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(max);

    // Mask: 1.0 for atoms, 0.0 for pad
    let mut p_mask = Vec::with_capacity(max);
    p_mask.extend(std::iter::repeat_n(1.0, n));
    p_mask.extend(std::iter::repeat_n(0.0, max - n));

    //Adj: Reconstruct row-by-row to handle 2D padding
    let mut p_adj = Vec::with_capacity(max * max);
    for r in 0..n {
        let row_start = r * num_atoms; // Input is flat [num_atoms * num_atoms]
        // Copy valid columns
        p_adj.extend_from_slice(&raw_adj[row_start..row_start + n]);
        // Pad columns (right side of matrix)
        p_adj.extend(std::iter::repeat(0.0).take(max - n));
    }
    // Pad rows (bottom of matrix)
    let remaining_rows = max - n;
    p_adj.extend(std::iter::repeat(0.0).take(remaining_rows * max));

    (p_adj, p_mask)
}

/// For comps too
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

/// Pharmacophore-feature graph for 3-D spatial/geometric structure.
///
/// Nodes = pharmacophore sites derived from the molecule characterization:
///   H-bond donors (type 1), H-bond acceptors (type 2),
///   hydrophobic carbons (type 3), aromatic ring centroids (type 4).
///
/// Edges are fully connected (pharmacophore graphs are small, ~5-30 nodes).
/// Edge weights use a Gaussian of Euclidean distance; edge features encode the
/// distance via a raw scaled value plus 4 RBF Gaussian basis functions, giving the
/// network explicit geometric information. Angles are captured implicitly by
/// multi-hop message passing (law-of-cosines encodes them after ≥2 GNN layers).
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataSpacial {
    /// Integer type index for each node: 1=Donor, 2=Acceptor, 3=Hydrophobic, 4=Aromatic.
    pub pharm_type_indices: Vec<i32>,
    /// Per-node scalar features: [r_from_pharm_centroid, mean_pairwise_dist] (both normalised).
    pub scalars: Vec<f32>,
    /// Gaussian distance-weighted adjacency (fully connected + self-loops). Flat [N²].
    pub adj: Vec<f32>,
    /// Per-edge geometric features: [scaled_dist, rbf0..rbf3]. Flat [N² × PER_SPACIAL_EDGE_FEATS].
    pub edge_feats: Vec<f32>,
    pub num_nodes: usize,
}

impl GraphDataSpacial {
    pub(in crate::therapeutic) fn empty() -> Self {
        Self {
            pharm_type_indices: Vec::new(),
            scalars: Vec::new(),
            adj: Vec::new(),
            edge_feats: Vec::new(),
            num_nodes: 0,
        }
    }

    pub fn new(mol: &MoleculeSmall) -> io::Result<Self> {
        let Some(char) = mol.characterization.as_ref() else {
            return Ok(Self::empty());
        };

        let atom_posits = &mol.common.atom_posits;

        // Collect (position as [f32;3], type_index) for each pharmacophore site.
        let mut nodes: Vec<([f32; 3], i32)> = Vec::new();

        for &i in &char.h_bond_donor {
            if i < atom_posits.len() {
                let p = atom_posits[i];
                nodes.push(([p.x as f32, p.y as f32, p.z as f32], 1));
            }
        }
        for &i in &char.h_bond_acceptor {
            if i < atom_posits.len() {
                let p = atom_posits[i];
                nodes.push(([p.x as f32, p.y as f32, p.z as f32], 2));
            }
        }
        for &i in &char.hydrophobic_carbon {
            if i < atom_posits.len() {
                let p = atom_posits[i];
                nodes.push(([p.x as f32, p.y as f32, p.z as f32], 3));
            }
        }
        for ring in &char.rings {
            let c = ring.center(atom_posits);
            nodes.push(([c.x as f32, c.y as f32, c.z as f32], 4));
        }

        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return Ok(Self::empty());
        }

        // Pharmacophore centroid.
        let (mut cx, mut cy, mut cz) = (0f32, 0f32, 0f32);
        for (p, _) in &nodes {
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        let inv_n = 1.0 / num_nodes as f32;
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        // Precompute all pairwise Euclidean distances (symmetric).
        let mut dist_mat = vec![0f32; num_nodes * num_nodes];
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                let pi = nodes[i].0;
                let pj = nodes[j].0;
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                dist_mat[i * num_nodes + j] = d;
                dist_mat[j * num_nodes + i] = d;
            }
        }

        // Node features.
        let mut pharm_type_indices = Vec::with_capacity(num_nodes);
        let mut scalars = Vec::with_capacity(num_nodes * PER_PHARM_SCALARS);

        for (i, (p, ty)) in nodes.iter().enumerate() {
            pharm_type_indices.push(*ty);

            // Rotation-invariant distance from pharmacophore centroid, normalised.
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            let r = (dx * dx + dy * dy + dz * dz).sqrt() / SPACIAL_DIST_SCALE;
            scalars.push(r);

            // Mean pairwise distance to all other pharmacophore nodes, normalised.
            let mean_d = if num_nodes > 1 {
                let mut sum = 0f32;
                for j in 0..num_nodes {
                    if i != j {
                        sum += dist_mat[i * num_nodes + j];
                    }
                }
                sum / ((num_nodes - 1) as f32 * SPACIAL_DIST_SCALE)
            } else {
                0.0
            };
            scalars.push(mean_d);
        }

        // Edge features and adjacency (fully connected; all pairs connected).
        let n2 = num_nodes * num_nodes;
        let mut adj = vec![0f32; n2];
        let mut edge_feats = vec![0f32; n2 * PER_SPACIAL_EDGE_FEATS];

        // Self-loops with weight 1; edge features at self-loop stay 0.
        for i in 0..num_nodes {
            adj[i * num_nodes + i] = 1.0;
        }

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    continue;
                }
                let d = dist_mat[i * num_nodes + j];
                let d_sq = d * d;

                // Gaussian-weighted adjacency (same spirit as the atom GNN bond kernel).
                adj[i * num_nodes + j] = (-d_sq / SPACIAL_ADJ_SIGMA_SQ).exp();

                // Edge feature vector: [scaled_dist, rbf_0 .. rbf_3].
                // RBF encoding explicitly gives the network multi-resolution distance info.
                let base = (i * num_nodes + j) * PER_SPACIAL_EDGE_FEATS;
                edge_feats[base] = (d / SPACIAL_DIST_SCALE).clamp(0.0, 1.5);
                for (k, &mu) in SPACIAL_RBF_CENTERS.iter().enumerate() {
                    let dmu = d - mu;
                    edge_feats[base + 1 + k] = (-(dmu * dmu) / SPACIAL_RBF_SIGMA_SQ).exp();
                }
            }
        }

        Ok(Self {
            pharm_type_indices,
            scalars,
            adj,
            edge_feats,
            num_nodes,
        })
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

/// Maps element to values the neural net can use.
fn vocab_lookup_element(el: Element) -> i32 {
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

/// Maps element to values the neural net can use.
fn vocab_lookup_component(comp_type: &ComponentType) -> i32 {
    use ComponentType::*;
    match comp_type {
        Atom(_) => 0,
        Ring(_) => 1,
        Chain(_) => 2,
        Hydroxyl => 3,
        Carbonyl => 4,
        Carboxylate => 5,
        Amine => 6,
        Amide => 7,
        Sulfonamide => 8,
        Sulfonimide => 9,
    }
}
