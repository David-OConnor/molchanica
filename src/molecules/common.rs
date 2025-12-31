//! This defines a `MoleculeCommon` struct, which is shared by all molecule types. It includes
//! the most important features, like atoms, bonds, and metadata.
//!

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::atomic::Ordering,
};

use bio_files::BondType;
use lin_alg::f64::{Quaternion, Vec3};

use crate::{
    mol_editor::NEXT_ATOM_SN,
    molecules::{Atom, Bond, build_adjacency_list},
};
use crate::molecules::rotatable_bonds::find_downstream_atoms;

/// Contains fields shared by all molecule types.
#[derive(Debug, Clone)]
pub struct MoleculeCommon {
    pub ident: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// A fast lookup for finding atoms, by index, covalently bonded to each atom.
    pub adjacency_list: Vec<Vec<usize>>,
    /// For repositioning atoms, e.g. from dynamics or absolute positioning.
    ///
    /// Absolute atom positions. For absolute conformation type[s], these positions are set and accessed directly, e.g., by MD
    /// simulations. We leave the molecule atom positions as ingested directly from data files. (e.g., relative positions).
    /// For rigid and semi-rigid conformations, these are derivative of the pose, in conjunction with
    /// the molecule atoms' (relative) positions.
    pub atom_posits: Vec<Vec3>,
    pub metadata: HashMap<String, String>,
    /// This is a bit different, as it's for our UI only. Doesn't fit with the others,
    /// but is safer and easier than trying to sync Vec indices.
    pub visible: bool,
    pub path: Option<PathBuf>,
    pub selected_for_md: bool,
    pub entity_i_range: Option<(usize, usize)>,
}

impl Default for MoleculeCommon {
    /// Only so we can set visible: true.
    fn default() -> Self {
        Self {
            ident: String::new(),
            metadata: HashMap::new(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            adjacency_list: Vec::new(),
            atom_posits: Vec::new(),
            visible: true,
            path: None,
            selected_for_md: false,
            entity_i_range: None,
        }
    }
}

impl MoleculeCommon {
    /// If `bonds` is none, create it based on atom distances. Useful in the case of mmCIF files,
    /// which usually lack bond information.
    ///
    /// Hydrogens should have been added to the atom set, if required, prior to running this,
    /// so bonds are created.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let mut result = Self {
            ident,
            metadata,
            atoms,
            bonds,
            atom_posits,
            path,
            ..Self::default()
        };

        result.build_adjacency_list();
        result
    }

    /// Build a list of, for each atom, all atoms bonded to it.
    /// We use this as part of our flexible-bond conformation algorithm, and in setting up
    /// angles and dihedrals for molecular docking.
    ///
    /// Run this after populate hydrogens.
    pub fn build_adjacency_list(&mut self) {
        self.adjacency_list = build_adjacency_list(&self.bonds, self.atoms.len());
    }

    /// Reset atom positions to be at their internal values, e.g. as present in the Mol2 or SDF files.
    pub fn reset_posits(&mut self) {
        self.atom_posits = self.atoms.iter().map(|a| a.posit).collect();
    }

    /// Used for rotation and motion; the rough center of the molecule.
    pub fn centroid(&self) -> Vec3 {
        let n = self.atom_posits.len() as f64;
        let sum = self
            .atom_posits
            .iter()
            .fold(Vec3::new_zero(), |a, b| a + *b);
        sum / n
    }

    pub fn move_to(&mut self, pos: Vec3) {
        let delta = pos - self.centroid();
        for posit in &mut self.atom_posits {
            *posit += delta;
        }
    }

    pub fn rotate(&mut self, rot: Quaternion, pivot_: Option<usize>) {
        let pivot = match pivot_ {
            Some(i) => self.atom_posits[i],
            None => self.centroid(),
        };

        for posit in &mut self.atom_posits {
            let local = *posit - pivot;
            let rotated = rot.rotate_vec(local);
            let out = rotated + pivot;

            *posit = out;
        }
    }

    /// Removes an atom, and any bond to it. Re-index bonds due to this
    /// removal from likely the interior of the molecule's seq.
    pub fn remove_atom(&mut self, i: usize) {
        if i >= self.atoms.len() {
            eprintln!("Error removing atom: Out of range");
            return;
        }

        self.atoms.remove(i);
        self.atom_posits.remove(i);

        self.bonds.retain_mut(|bond| {
            if bond.atom_0 == i || bond.atom_1 == i {
                return false;
            }

            if bond.atom_0 > i {
                bond.atom_0 -= 1;
            }
            if bond.atom_1 > i {
                bond.atom_1 -= 1;
            }
            true
        });

        self.adjacency_list.remove(i);

        for adj in &mut self.adjacency_list {
            adj.retain(|&j| j != i);

            for j in adj.iter_mut() {
                if *j > i {
                    *j -= 1;
                }
            }
        }
    }

    /// Re-assign atom serial numbers as 1-ripple. Useful after or during editing, especially
    /// prior to saving in SDF format, which doesn't explicitly list SNs with the atom.
    pub fn reassign_sns(&mut self) {
        // todo: Be more clever about this.
        let mut updated_sns = Vec::with_capacity(self.atoms.len());

        for (i, atom) in self.atoms.iter_mut().enumerate() {
            let sn_new = i as u32 + 1;
            atom.serial_number = sn_new;
            updated_sns.push(sn_new);
        }

        for bond in &mut self.bonds {
            bond.atom_0_sn = updated_sns[bond.atom_0];
            bond.atom_1_sn = updated_sns[bond.atom_1];
        }

        NEXT_ATOM_SN.store(
            match updated_sns.last() {
                Some(l) => *l + 1,
                None => 1,
            },
            Ordering::Release,
        )
    }

    /// The sum of each atom's elemental atomic weight, in Daltons (amu).
    // Todo: Would be more generally useful in bio_files?
    pub fn atomic_weight(&self) -> f32 {
        let result: f64 = self
            .atoms
            .iter()
            .map(|a| a.element.atomic_weight() as f64)
            .sum();

        result as f32
    }

    /// Unweighted chemistry adjacency matrix: A N×N matrix with 1 where a bond exists
    /// (0 otherwise). N is the atom count.
    // Todo: Would be more generally useful in bio_files?
    pub fn adjacency_matrix(&self) -> Vec<Vec<u8>> {
        let n = self.adjacency_list.len();
        let mut result = vec![vec![0; n]; n];

        for (i, neighs) in self.adjacency_list.iter().enumerate() {
            for &j in neighs {
                if j < n && i != j {
                    result[i][j] = 1;
                    result[j][i] = 1;
                }
            }
        }

        result
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
    pub fn rotate_around_bond(&mut self, bond_pivot: usize, rot_amt: f64) {
        if bond_pivot >= self.bonds.len() {
            eprintln!("Error: Bond pivot out of bounds.");
            return;
        }

        let pivot = &self.bonds[bond_pivot];

        // Measure how many atoms would be "downstream" from each side
        let side0_downstream = find_downstream_atoms(&self.adjacency_list, pivot.atom_1, pivot.atom_0); // atoms on atom_0 side
        let side1_downstream = find_downstream_atoms(&self.adjacency_list, pivot.atom_0, pivot.atom_1); // atoms on atom_1 side

        // Rotate the smaller side; keep pivot_idx on the larger side
        let (pivot_idx, side_idx, downstream_atom_indices) =
            if side0_downstream.len() > side1_downstream.len() {
                (pivot.atom_0, pivot.atom_1, side1_downstream)
            } else {
                (pivot.atom_1, pivot.atom_0, side0_downstream)
            };

        // Pivot and side positions
        let pivot_pos = self.atom_posits[pivot_idx];
        let side_pos = self.atom_posits[side_idx];

        let axis_raw = side_pos - pivot_pos;
        let axis_len2 = axis_raw.dot(axis_raw);

        if axis_len2 <= 1.0e-24 {
            eprintln!("Error: bond axis is degenerate (zero length).");
            return;
        }

        let axis_vec = axis_raw.to_normalized();

        // Build the Quaternion for this rotation (assumes rot_amt is radians)
        let rotator = Quaternion::from_axis_angle(axis_vec, rot_amt);

        // Now apply the rotation to each downstream atom:
        for &atom_idx in &downstream_atom_indices {
            let old_pos = self.atom_posits[atom_idx];
            let relative = old_pos - pivot_pos;
            let new_pos = pivot_pos + rotator.rotate_vec(relative);
            self.atom_posits[atom_idx] = new_pos;
        }

        // We've updated atom positions in place; update internal coords.
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.posit = self.atom_posits[i];
        }
    }


    /// A helper used to ensure that there is a valid atom for each bond. (Checks both SN and index),
    /// and that checks if the adjacency list is up to date. This is used for debugging only.
    #[allow(unused)]
    pub fn validate_bonds(&self) {
        println!("\nValidating bonds... (This should not be in permanent code)\n");
        for bond in &self.bonds {
            assert!(bond.atom_0 < self.atoms.len());
            assert!(bond.atom_1 < self.atoms.len());
            assert_ne!(bond.atom_0, bond.atom_1);

            assert!(self.adjacency_list[bond.atom_0].contains(&bond.atom_1));
            assert!(self.adjacency_list[bond.atom_1].contains(&bond.atom_0));

            assert_eq!(self.adjacency_list.len(), self.atoms.len());

            assert_eq!(bond.atom_0_sn, self.atoms[bond.atom_0].serial_number);
            assert_eq!(bond.atom_1_sn, self.atoms[bond.atom_1].serial_number);
        }
    }
}
