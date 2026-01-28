//! Related to the graph neural net. This includes atom and bond networks,
//! and per-atom, per-bond etc features.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    io,
    io::ErrorKind,
};

use bio_files::BondType;
use bio_files::md_params::ForceFieldParams;
use na_seq::Element::{
    Bromine, Carbon, Chlorine, Fluorine, Hydrogen, Iodine, Nitrogen, Oxygen, Phosphorus, Sulfur,
};

use crate::{
    molecules::{Atom, build_adjacency_list, small::MoleculeSmall},
    therapeutic::train::{
        BOND_SIGMA_SQ, EXCLUDE_HYDROGEN, FF_BUCKETS, MAX_ATOMS, Sample, StandardScaler,
    },
};

// Degree, partial charge, FF name, element, is H-bond acceptor, is H-bond donor, in aromatic ring
pub(in crate::therapeutic) const PER_ATOM_SCALARS: usize = 7;

// Scaled/modified proxies for r_0, k_b
pub(in crate::therapeutic) const PER_EDGE_FEATS: usize = 2;

const DR_SCALE: f32 = 0.15;
const KB_REF: f32 = 300.0;

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphData {
    pub elem_indices: Vec<i32>,
    pub ff_indices: Vec<i32>,
    pub scalars: Vec<f32>,
    pub adj: Vec<f32>,
    pub edge_feats: Vec<f32>,
    pub num_atoms: usize,
}

/// Helper: Converts raw Atoms and Bonds into Flat vectors for Tensors.
/// Used by both Training and Inference.
pub(in crate::therapeutic) fn mol_to_graph_data(
    mol: &MoleculeSmall,
    ff_params: &ForceFieldParams,
) -> io::Result<GraphData> {
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
            let ff1 = atoms[a0].force_field_type.clone().unwrap();
            let bond_stretching = ff_params.get_bond(&(ff0.clone(), ff1.clone()), true);

            let bond_stretching = if let Some(v) = bond_stretching {
                v
            } else {
                // A coarse fallback.
                let mut safe_fallback = ("cc".to_owned(), "n".to_owned());

                if (&ff0 == "cc" && ff1.starts_with("n")) || (&ff1 == "cc" && ff0.starts_with("n"))
                {
                    safe_fallback = ("cc".to_owned(), "n4".to_owned());
                } else if (&ff0 == "cg" && ff1.starts_with("c"))
                    || (&ff1 == "cg" && ff0.starts_with("c"))
                {
                    safe_fallback = ("cg".to_owned(), "cg".to_owned());
                } else if atoms[0].element == Carbon && atoms[1].element == Carbon {
                    safe_fallback = ("cc".to_owned(), "cc".to_owned());
                    eprintln!(
                        "Missing bond stretching for bond {bond:?}. \nAtoms {ff0} | {ff1}\n. Using a substitute.",
                    );
                } else if atoms[0].element == Nitrogen && atoms[1].element == Nitrogen {
                    safe_fallback = ("n".to_owned(), "n".to_owned());

                    eprintln!(
                        "Missing bond stretching for bond {bond:?}. \nAtoms {ff0} | {ff1}\n. Using a substitute.",
                    );
                }

                ff_params.get_bond(&safe_fallback, false).unwrap()
            };

            let dr_norm = ((dist - bond_stretching.r_0) / DR_SCALE).clamp(-5.0, 5.0);
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
    let mut degrees_vec = vec![0.0; num_atoms];
    for i in 0..num_atoms {
        let mut d = 0.0;
        for j in 0..num_atoms {
            d += adj_list[i * num_atoms + j];
        }
        degrees_vec[i] = d;
    }

    Ok(GraphData {
        elem_indices,
        ff_indices,
        scalars,
        adj: adj_list,
        edge_feats,
        num_atoms,
    })
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

pub fn fit_scaler(train: &[Sample]) -> StandardScaler {
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

/// Helper: Pads a single graph to MAX_ATOMS.
/// Returns (PaddedNodes, PaddedAdj, PaddedMask) as flat vectors.
pub(in crate::therapeutic) fn pad_adj_and_mask(
    raw_adj: &[f32],
    num_atoms: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(MAX_ATOMS);

    // Mask: 1.0 for atoms, 0.0 for pad
    let mut p_mask = Vec::with_capacity(MAX_ATOMS);
    p_mask.extend(std::iter::repeat(1.0).take(n));
    p_mask.extend(std::iter::repeat(0.0).take(MAX_ATOMS - n));

    //Adj: Reconstruct row-by-row to handle 2D padding
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

pub(in crate::therapeutic) fn pad_edge_feats(
    edge_feats: &[f32], // [num_atoms*num_atoms*PER_EDGE_FEATS]
    num_atoms: usize,
) -> Vec<f32> {
    let n = num_atoms.min(MAX_ATOMS);

    let mut out = vec![0.0f32; MAX_ATOMS.pow(2) * PER_EDGE_FEATS];

    for i in 0..n {
        for j in 0..n {
            let src_base = (i * num_atoms + j) * PER_EDGE_FEATS;
            let dst_base = (i * MAX_ATOMS + j) * PER_EDGE_FEATS;
            out[dst_base..dst_base + PER_EDGE_FEATS]
                .copy_from_slice(&edge_feats[src_base..src_base + PER_EDGE_FEATS]);
        }
    }

    out
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
