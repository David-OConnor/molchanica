use std::f64::consts::{PI, TAU};

use lin_alg::f64::Vec3;
use na_seq::Element::{Fluorine, Hydrogen, Nitrogen, Oxygen, Sulfur};

use crate::{
    molecules::{Atom, Bond, HydrogenBond, HydrogenBondTwoMols},
    util::find_atom,
};

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

/// Helper
fn h_bond_candidate_el(atom: &Atom) -> bool {
    matches!(atom.element, Nitrogen | Oxygen | Sulfur | Fluorine)
}

fn hydrogen_bond_inner(
    bonds: &mut Vec<HydrogenBond>,
    donor_heavy: &Atom,
    donor_heavy_posit: Vec3,
    donor_h: &Atom,
    donor_h_posit: Vec3,
    acc_candidate: &Atom,
    acc_candidate_posit: Vec3,
    donor_heavy_i: usize,
    donor_h_i: usize,
    acc_i: usize,
    relaxed_dist_thresh: bool,
    // atoms_donor: &[Atom],
    // atoms_acc: &[Atom],
) {
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
        return;
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
        bonds.push(HydrogenBond::new(donor_heavy_i, acc_i, donor_h_i, strength));
    }
}

/// Create hydrogen bonds between all atoms in a single molecule. See `create_hydrogen_bonds_one_way` for the more
/// flexible fn it calls.
pub fn create_hydrogen_bonds_single_mol(
    atoms: &[Atom],
    posits: &[Vec3],
    bonds: &[Bond],
) -> Vec<HydrogenBond> {
    let indices: Vec<_> = (0..atoms.len()).collect();
    create_hydrogen_bonds_one_way(
        atoms, posits, &indices, bonds, atoms, posits, &indices, false,
    )
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
    // Mol 0: Donor. Mol 1: Acceptor.
    let part_0 = create_hydrogen_bonds_one_way(
        atoms_mol0,
        posits_mol0,
        &indices_0,
        bonds_mol0,
        atoms_mol1,
        posits_mol1,
        &indices_1,
        false,
    );

    // Mol 1: Donor. Mol 0: Acceptor.
    let part_1 = create_hydrogen_bonds_one_way(
        atoms_mol1,
        posits_mol1,
        &indices_1,
        bonds_mol1,
        atoms_mol0,
        posits_mol0,
        &indices_0,
        false,
    );

    let mut res = Vec::new();

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
    let mut result = Vec::new();

    // Bonds between donor and H.
    let potential_donor_bonds: Vec<&Bond> = bonds_donor
        .iter()
        .filter(|b| {
            // Maps from a global index, to a local atom from a subset.
            let atom_0 = find_atom(atoms_donor, atoms_donor_i, b.atom_0);
            let atom_1 = find_atom(atoms_donor, atoms_donor_i, b.atom_1);

            let (Some(atom_0), Some(atom_1)) = (atom_0, atom_1) else {
                eprintln!(
                    "Error! Can't find atoms from indices when making H bonds (Donor finding)"
                );
                return false;
            };

            let cfg_0_valid = h_bond_candidate_el(atom_0) && atom_1.element == Hydrogen;
            let cfg_1_valid = h_bond_candidate_el(atom_1) && atom_0.element == Hydrogen;

            cfg_0_valid || cfg_1_valid
        })
        .collect();

    let potential_acceptors: Vec<(usize, &Atom, Vec3)> = atoms_acc
        .iter()
        .enumerate()
        .filter(|(_, a)| h_bond_candidate_el(a))
        .map(|(i, a)| (atoms_acc_i[i], a, posits_acc[i]))
        .collect();

    for donor_bond in potential_donor_bonds {
        let donor_0 = find_atom(atoms_donor, atoms_donor_i, donor_bond.atom_0);
        let donor_1 = find_atom(atoms_donor, atoms_donor_i, donor_bond.atom_1);

        let (Some(donor_0), Some(donor_1)) = (donor_0, donor_1) else {
            eprintln!("Error! Can't find atoms from indices when making H bonds");
            continue;
        };

        let (donor_heavy, posit_heavy, donor_h, posit_h, donor_heavy_i, donor_h_i) =
            if donor_0.element == Hydrogen {
                (
                    donor_1,
                    posits_donor[donor_bond.atom_1],
                    donor_0,
                    posits_donor[donor_bond.atom_0],
                    donor_bond.atom_1,
                    donor_bond.atom_0,
                )
            } else {
                (
                    donor_0,
                    posits_donor[donor_bond.atom_0],
                    donor_1,
                    posits_donor[donor_bond.atom_1],
                    donor_bond.atom_0,
                    donor_bond.atom_1,
                )
            };

        for (acc_i, acc_candidate, acc_posit) in &potential_acceptors {
            hydrogen_bond_inner(
                &mut result,
                donor_heavy,
                posit_heavy,
                donor_h,
                posit_h,
                acc_candidate,
                *acc_posit,
                donor_heavy_i,
                donor_h_i,
                *acc_i,
                relaxed_dist_thresh,
                // atoms_donor,
                // atoms_acc,
            );
        }
    }

    result
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
