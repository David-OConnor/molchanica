//! For determining which bonds in a molecule allow free rotation, i.e. changing the dihedral
//! angle. For example, not rings.

use crate::molecules::{Bond, common::MoleculeCommon};
use bio_files::BondType;
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;

#[derive(Clone, Debug)]
pub struct RotatableBond {
    pub bond_i: usize,
    pub downstream_from_a1: Vec<usize>,
}

impl MoleculeCommon {
    /// Identify which bonds are rotatable; able to change the molecule's conformation. Note that we
    /// exclude methyl groups.
    ///
    /// Return the shorter downstream side.
    pub fn find_rotatable_bonds(&self) -> Vec<RotatableBond> {
        let mut out = Vec::new();

        for (bond_i, bond) in self.bonds.iter().enumerate() {
            if !matches!(bond.bond_type, BondType::Single) {
                continue;
            }

            let atom_0 = bond.atom_0;
            let atom_1 = bond.atom_1;

            if self.adjacency_list[atom_0].len() <= 1 || self.adjacency_list[atom_1].len() <= 1 {
                continue;
            }

            if bond.in_a_cycle(&self.adjacency_list) {
                continue;
            }

            let atom_count = self.atoms.len();

            let mut downstream = find_downstream_atoms(&self.adjacency_list, atom_0, atom_1);
            if downstream.is_empty() || downstream.len() == atom_count {
                continue;
            }

            // Make downstream the smaller side.
            if downstream.len() > atom_count / 2 {
                downstream = find_other_side(&downstream, atom_count)
            }

            // Exclude methyl, hydroxyl, etc groups by convention; they're rotatable, but don't notably
            // affect the conformation.
            let mut any_non_h = false;
            for i_down in &downstream {
                // Don't check for atoms part of the rotation bond.
                if *i_down == atom_0 || *i_down == atom_1 {
                    continue;
                }

                if self.atoms[*i_down].element != Element::Hydrogen {
                    any_non_h = true;
                    break;
                }
            }

            if !any_non_h {
                continue;
            }

            out.push(RotatableBond {
                bond_i,
                downstream_from_a1: downstream,
            });
        }

        out
    }

    /// Rotate part of the molecule around a bond. Rotates the *smaller* part of the molecule as divided
    /// by this bond: Each pivot rotation rotates the side of the flexible bond that
    /// has fewer atoms; the intent is to minimize the overall position changes for these flexible bond angle
    /// changes.
    ///
    /// For each rotatable bond, divide all atoms into two groups:
    /// those upstream of this bond, and those downstream. Note that not all bonds make sense as
    /// rotation centers. For example, bonds in rings.
    ///
    /// We assume this bond as been determined to be rotatable ahead of time.
    pub fn rotate_around_bond(
        &mut self,
        bond_pivot: usize,
        rot_amt: f64,
        downstream: Option<&[usize]>,
    ) {
        if let Some(posits) = rotate_around_bond(self, bond_pivot, rot_amt, downstream) {
            self.atom_posits = posits;
        }

        // todo: Do we want this?
        // We've updated atom positions in place; update internal coords.
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.posit = self.atom_posits[i];
        }
    }
}

/// Find atoms in the molecule that are not part of a set. We use this to find
/// the other downstream side, after calculating the first.
fn find_other_side(side0_downstream: &[usize], atom_count: usize) -> Vec<usize> {
    let mut res = Vec::with_capacity(atom_count - side0_downstream.len());

    let mut on_side0 = vec![false; atom_count];
    for &idx in side0_downstream {
        on_side0[idx] = true;
    }

    for i in 0..atom_count {
        if !on_side0[i] {
            res.push(i);
        }
    }
    res
}

/// See `MoleculeCommon::rotate_around_bond`; this allows us to do it without mutating.
pub fn rotate_around_bond(
    mol: &MoleculeCommon,
    bond_pivot: usize,
    rot_amt: f64,
    // Optionally provide as a cache; this can be calculated otherwise.
    downstream: Option<&[usize]>,
) -> Option<Vec<Vec3>> {
    if bond_pivot >= mol.bonds.len() {
        eprintln!("Error: Bond pivot out of bounds.");
        return None;
    }

    let pivot = &mol.bonds[bond_pivot];

    // Measure how many atoms would be "downstream" from each side
    let side0_downstream = if let Some(downstream) = downstream {
        downstream.to_vec()
    } else {
        find_downstream_atoms(&mol.adjacency_list, pivot.atom_1, pivot.atom_0) // atoms on atom_0 side
    };

    // This is a simple check for all atoms not in side0_downstream, but is faster.
    let side1_downstream = find_other_side(&side0_downstream, mol.atoms.len());

    // Rotate the smaller side; keep pivot_idx on the larger side
    let (pivot_idx, side_idx, downstream_atom_indices) =
        if side0_downstream.len() > side1_downstream.len() {
            (pivot.atom_0, pivot.atom_1, side1_downstream)
        } else {
            (pivot.atom_1, pivot.atom_0, side0_downstream)
        };

    // Pivot and side positions
    let pivot_pos = mol.atom_posits[pivot_idx];
    let side_pos = mol.atom_posits[side_idx];

    let axis_raw = side_pos - pivot_pos;
    let axis_len2 = axis_raw.dot(axis_raw);

    if axis_len2 <= 1.0e-24 {
        eprintln!("Error: bond axis is degenerate (zero length).");
        return None;
    }

    let axis_vec = axis_raw.to_normalized();

    // Build the Quaternion for this rotation (assumes rot_amt is radians)
    let rotator = Quaternion::from_axis_angle(axis_vec, rot_amt);

    // Now apply the rotation to each downstream atom:
    let mut result = mol.atom_posits.clone();

    // We're not using `rotate_about_axis` here due to only updating downstream atom indices.
    for &atom_idx in &downstream_atom_indices {
        let old_pos = mol.atom_posits[atom_idx];
        let relative = old_pos - pivot_pos;
        let new_pos = pivot_pos + rotator.rotate_vec(relative);

        result[atom_idx] = new_pos;
    }

    Some(result)
}

/// We use this to rotate molecules around a bond pivot. For example, by the user directly, or
/// in ligand conformation algorithms that try different poses.
///
/// `pivot` and `side` are atom indices.
fn find_downstream_atoms(adj_list: &[Vec<usize>], pivot: usize, side: usize) -> Vec<usize> {
    if pivot == side {
        return Vec::new();
    }

    let mut visited = vec![false; adj_list.len()];
    visited[pivot] = true;

    // A bit of pre-allocation.
    let mut stack = Vec::with_capacity(16);
    let mut result = Vec::with_capacity(16);

    visited[side] = true;
    stack.push(side);

    while let Some(u) = stack.pop() {
        result.push(u);

        for &v in &adj_list[u] {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }

    result
}

/// Determines if a bond is part of a cycle (ring). This has implications, for example, in determining
/// if it can be used as a rotation pivot.
impl Bond {
    pub fn in_a_cycle(&self, adj_list: &[Vec<usize>]) -> bool {
        let a0 = self.atom_0;
        let a1 = self.atom_1;

        if a0 == a1 {
            return false;
        }

        let mut visited = vec![false; adj_list.len()];
        let mut stack = Vec::with_capacity(16);

        visited[a0] = true;
        stack.push(a0);

        while let Some(cur) = stack.pop() {
            if cur == a1 {
                return true;
            }

            for &nbr in &adj_list[cur] {
                if (cur == a0 && nbr == a1) || (cur == a1 && nbr == a0) {
                    continue;
                }
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }

        false
    }
}
