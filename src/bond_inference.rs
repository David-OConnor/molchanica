use std::collections::HashMap;

use lin_alg::f64::Vec3;
use na_seq::Element::{Fluorine, Hydrogen, Nitrogen, Oxygen, Sulfur};
use rayon::prelude::*;
use std::f64::consts::{PI, TAU};
use std::time::Instant;

use crate::molecules::{Atom, Bond, HydrogenBond, HydrogenBondTwoMols};

// Note: Chimera shows H bonds as ranging generally from 2.8 to 3.3.
// Note: These values all depend on which is the donor. Your code doesn't take this into account.
const H_BOND_O_O_DIST: f64 = 2.7;
const H_BOND_N_N_DIST: f64 = 3.05;
const H_BOND_O_N_DIST: f64 = 2.9;

const H_BOND_N_F_DIST: f64 = 2.75;
const H_BOND_N_S_DIST: f64 = 3.35;
const H_BOND_S_O_DIST: f64 = 3.35;
const H_BOND_S_F_DIST: f64 = 3.2; // rarate

const H_BOND_DIST_THRESH: f64 = 0.3;
const H_BOND_DIST_GRID: f64 = 3.6;

const H_BOND_ANGLE_THRESH: f64 = TAU / 3.;

// H-bond strength scoring: distance and angle ranges.
const H_BOND_STRENGTH_DIST_MIN: f64 = 2.4; // Å — strongest
const H_BOND_STRENGTH_DIST_MAX: f64 = 3.6; // Å — cutoff
const H_BOND_STRENGTH_ANGLE_MIN: f64 = PI * 2. / 3.; // 120° — weakest accepted

/// Spatial grid cell key for a 3D position.
fn cell_key(pos: Vec3) -> (i32, i32, i32) {
    (
        (pos.x / H_BOND_DIST_GRID).floor() as i32,
        (pos.y / H_BOND_DIST_GRID).floor() as i32,
        (pos.z / H_BOND_DIST_GRID).floor() as i32,
    )
}

/// Helper
fn h_bond_candidate_el(atom: &Atom) -> bool {
    matches!(atom.element, Nitrogen | Oxygen | Sulfur | Fluorine)
}

fn hydrogen_bond_inner(
    donor_heavy: &Atom,
    donor_heavy_posit: Vec3,
    donor_h_posit: Vec3,
    acc_candidate: &Atom,
    acc_candidate_posit: Vec3,
    donor_heavy_i: usize,
    donor_h_i: usize,
    acc_i: usize,
    relaxed_dist_thresh: bool,
) -> Option<HydrogenBond> {
    let d_e = donor_heavy.element; // Cleans up the verbose code below.
    let a_e = acc_candidate.element;
    // todo: Take into account typical lenghs of donor and receptor; here your order isn't used.
    let dist_thresh = if d_e == Oxygen && a_e == Oxygen {
        H_BOND_O_O_DIST
    } else if d_e == Nitrogen && a_e == Nitrogen {
        H_BOND_N_N_DIST
    } else if (d_e == Oxygen && a_e == Nitrogen) || (d_e == Nitrogen && a_e == Oxygen) {
        H_BOND_O_N_DIST
    } else if (d_e == Fluorine && a_e == Nitrogen) || (d_e == Nitrogen && a_e == Fluorine) {
        H_BOND_N_F_DIST
    } else {
        H_BOND_N_S_DIST // Good enough for other combos involving S and F, for now.
    };

    let modifier = if relaxed_dist_thresh {
        H_BOND_DIST_THRESH * 2.
    } else {
        H_BOND_DIST_THRESH
    };

    let dist_thresh_min = dist_thresh - modifier;
    let dist_thresh_max = dist_thresh + modifier;

    let dist = (acc_candidate_posit - donor_heavy_posit).magnitude();
    if dist < dist_thresh_min || dist > dist_thresh_max {
        return None;
    }

    let angle = {
        let donor_h = donor_h_posit - donor_heavy_posit;
        let donor_acceptor = donor_heavy_posit - acc_candidate_posit;

        donor_acceptor
            .to_normalized()
            .dot(donor_h.to_normalized())
            .acos()
    };

    if angle > H_BOND_ANGLE_THRESH {
        let strength = h_bond_strength(donor_heavy_posit, donor_h_posit, acc_candidate_posit);
        // Note: Assumes one way.
        Some(HydrogenBond::new(donor_heavy_i, acc_i, donor_h_i, strength))
    } else {
        None
    }
}

type AcceptorGrid<'a> = HashMap<(i32, i32, i32), Vec<(usize, &'a Atom, Vec3)>>;

/// Build a spatial hash grid from acceptor atoms for O(1) neighbor lookups.
fn build_acceptor_grid<'a>(
    atoms: &'a [Atom],
    posits: &[Vec3],
    indices: &[usize],
) -> AcceptorGrid<'a> {
    let mut grid: AcceptorGrid = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        if h_bond_candidate_el(atom) {
            let posit = posits[i];
            grid.entry(cell_key(posit))
                .or_default()
                .push((indices[i], atom, posit));
        }
    }
    grid
}

/// Create hydrogen bonds between all atoms in a single molecule. See `create_hydrogen_bonds_one_way` for the more
/// flexible fn it calls.
pub fn create_hydrogen_bonds_single_mol(
    atoms: &[Atom],
    posits: &[Vec3],
    bonds: &[Bond],
) -> Vec<HydrogenBond> {
    println!("Creating hydrogen bonds within a single molecule...");
    let start = Instant::now();

    let indices: Vec<_> = (0..atoms.len()).collect();
    let result = create_hydrogen_bonds_one_way(
        atoms, posits, &indices, bonds, atoms, posits, &indices, false,
    );

    let elapsed = start.elapsed().as_millis();
    println!("Hydrogen bonds created in {elapsed} ms");
    result
}

/// Create hydrogen bonds between two  molecules.
pub fn create_hydrogen_bonds_two_mols(
    atoms_mol0: &[Atom],
    posits_mol0: &[Vec3],
    bonds_mol0: &[Bond],
    atoms_mol1: &[Atom],
    posits_mol1: &[Vec3],
    bonds_mol1: &[Bond],
) -> Vec<HydrogenBondTwoMols> {
    let indices_0: Vec<_> = (0..atoms_mol0.len()).collect();
    let indices_1: Vec<_> = (0..atoms_mol1.len()).collect();

    // Run both directions in parallel.
    let (part_0, part_1) = rayon::join(
        || {
            create_hydrogen_bonds_one_way(
                atoms_mol0,
                posits_mol0,
                &indices_0,
                bonds_mol0,
                atoms_mol1,
                posits_mol1,
                &indices_1,
                false,
            )
        },
        || {
            create_hydrogen_bonds_one_way(
                atoms_mol1,
                posits_mol1,
                &indices_1,
                bonds_mol1,
                atoms_mol0,
                posits_mol0,
                &indices_0,
                false,
            )
        },
    );

    let mut res = Vec::with_capacity(part_0.len() + part_1.len());

    for bond in part_0 {
        res.push(HydrogenBondTwoMols {
            donor: (0, bond.donor),
            acceptor: (1, bond.acceptor),
            hydrogen: bond.hydrogen,
            strength: bond.strength,
        });
    }

    for bond in part_1 {
        res.push(HydrogenBondTwoMols {
            donor: (1, bond.donor),
            acceptor: (0, bond.acceptor),
            hydrogen: bond.hydrogen,
            strength: bond.strength,
        });
    }

    res
}

/// Infer hydrogen bonds from a list of atoms. This takes into account bond distance between suitable
/// atom types (generally N and O; sometimes S and F), and geometry regarding the hydrogen covalently
/// bonded to the donor.
///
/// Separates donor from acceptor inputs, for use in cases like bonds between targets and ligands.
/// We include indices, in the case where atoms are subsets of molecules; this allows bonds indices
/// to be preserved.
pub fn create_hydrogen_bonds_one_way(
    atoms_donor: &[Atom],
    posits_donor: &[Vec3],
    atoms_donor_i: &[usize],
    bonds_donor: &[Bond],
    atoms_acc: &[Atom],
    posits_acc: &[Vec3],
    atoms_acc_i: &[usize],
    relaxed_dist_thresh: bool,
) -> Vec<HydrogenBond> {
    // Build a map from global index -> local index for O(1) atom lookups.
    let donor_index_map: HashMap<usize, usize> = atoms_donor_i
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();

    // Build spatial grid for acceptor atoms.
    let grid = build_acceptor_grid(atoms_acc, posits_acc, atoms_acc_i);

    // Bonds between donor and H.
    let potential_donor_bonds: Vec<&Bond> = bonds_donor
        .iter()
        .filter(|b| {
            let atom_0 = donor_index_map.get(&b.atom_0).map(|&i| &atoms_donor[i]);
            let atom_1 = donor_index_map.get(&b.atom_1).map(|&i| &atoms_donor[i]);

            let (Some(atom_0), Some(atom_1)) = (atom_0, atom_1) else {
                return false;
            };

            let cfg_0_valid = h_bond_candidate_el(atom_0) && atom_1.element == Hydrogen;
            let cfg_1_valid = h_bond_candidate_el(atom_1) && atom_0.element == Hydrogen;

            cfg_0_valid || cfg_1_valid
        })
        .collect();

    // Process each donor bond in parallel; use the spatial grid for acceptor lookups.
    potential_donor_bonds
        .into_par_iter()
        .flat_map_iter(|donor_bond| {
            let local_0 = donor_index_map[&donor_bond.atom_0];
            let local_1 = donor_index_map[&donor_bond.atom_1];
            let atom_0 = &atoms_donor[local_0];
            let atom_1 = &atoms_donor[local_1];

            let (donor_heavy, posit_heavy, posit_h, donor_heavy_i, donor_h_i) =
                if atom_0.element == Hydrogen {
                    (
                        atom_1,
                        posits_donor[local_1],
                        posits_donor[local_0],
                        donor_bond.atom_1,
                        donor_bond.atom_0,
                    )
                } else {
                    (
                        atom_0,
                        posits_donor[local_0],
                        posits_donor[local_1],
                        donor_bond.atom_0,
                        donor_bond.atom_1,
                    )
                };

            let center = cell_key(posit_heavy);
            let mut local_bonds = Vec::new();

            for dx in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dz in -1i32..=1 {
                        let key = (center.0 + dx, center.1 + dy, center.2 + dz);
                        if let Some(acceptors) = grid.get(&key) {
                            for &(acc_i, acc_atom, acc_posit) in acceptors {
                                if let Some(bond) = hydrogen_bond_inner(
                                    donor_heavy,
                                    posit_heavy,
                                    posit_h,
                                    acc_atom,
                                    acc_posit,
                                    donor_heavy_i,
                                    donor_h_i,
                                    acc_i,
                                    relaxed_dist_thresh,
                                ) {
                                    local_bonds.push(bond);
                                }
                            }
                        }
                    }
                }
            }

            local_bonds
        })
        .collect()
}

/// Calculate hydrogen bond strength from donor heavy-atom, hydrogen, and acceptor positions.
/// Uses the D···A distance and the D-H···A angle (at H). Returns a value in [0, 1].
pub fn h_bond_strength(donor_posit: Vec3, h_posit: Vec3, acc_posit: Vec3) -> f32 {
    let dist = (donor_posit - acc_posit).magnitude();
    let dist_score = ((H_BOND_STRENGTH_DIST_MAX - dist)
        / (H_BOND_STRENGTH_DIST_MAX - H_BOND_STRENGTH_DIST_MIN))
        .clamp(0., 1.);

    // D-H···A angle measured at the hydrogen. 180° (π) is ideal / linear.
    let vec_hd = (donor_posit - h_posit).to_normalized();
    let vec_ha = (acc_posit - h_posit).to_normalized();
    let angle = vec_hd.dot(vec_ha).clamp(-1., 1.).acos();

    let angle_score =
        ((angle - H_BOND_STRENGTH_ANGLE_MIN) / (PI - H_BOND_STRENGTH_ANGLE_MIN)).clamp(0., 1.);

    (dist_score * angle_score) as f32
}
