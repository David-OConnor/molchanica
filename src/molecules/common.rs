//! This defines a `MoleculeCommon` struct, which is shared by all molecule types. It includes
//! the most important features, like atoms, bonds, and metadata.
//!

use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU32, Ordering},
};

use lin_alg::f64::{Quaternion, Vec3};

// Used by the mol editor, and alignment.
pub static NEXT_ATOM_SN: AtomicU32 = AtomicU32::new(0);

use crate::molecules::{Atom, Bond, build_adjacency_list};

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
    /// This is a cached derivative of `path`.
    pub filename: String,
    pub selected_for_md: bool,
    pub entity_i_range: Option<(usize, usize)>,
    // todo: Consider if we should move this to MoleculeSmall etc.
    // todo: This index only/always applies to small molecules.
    /// We have instantiated multiple copies of a molecule for MD simulations.
    /// If Some, that is the index of the parent.
    pub copy_for_md: Option<usize>,
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
            filename: String::new(),
            selected_for_md: false,
            entity_i_range: None,
            copy_for_md: None,
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

        let filename = match &path {
            Some(p) => p.file_stem().unwrap().to_string_lossy().to_string(),

            None => String::new(),
        };

        let mut result = Self {
            ident,
            metadata,
            atoms,
            bonds,
            atom_posits,
            path,
            filename,
            ..Self::default()
        };

        result.build_adjacency_list();
        result
    }

    pub fn update_path(&mut self, path: &Path) {
        self.path = Some(path.to_owned());
        self.filename = path.file_stem().unwrap().to_string_lossy().to_string();
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

    /// Update local positions so they're centered around the origin. Useful for molecule creation
    /// workflows.
    pub fn center_local_posits_around_origin(&mut self) {
        let center = {
            let mut c = Vec3::new_zero();
            for atom in &self.atoms {
                c += atom.posit;
            }
            c / self.atom_posits.len() as f64
        };

        for atom in &mut self.atoms {
            atom.posit -= center;
        }
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

    /// Uses atom internal positions.
    pub fn centroid_local(&self) -> Vec3 {
        let n = self.atoms.len() as f64;
        let mut sum = Vec3::new_zero();

        for a in &self.atoms {
            sum += a.posit;
        }

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
    /// We also  use it when assembling nucleic acids and other molecule generation.
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

    /// Uses both the indentifier and filename, if different.
    pub fn name(&self) -> String {
        let mut result = self.ident.to_string();
        if self.filename.to_lowercase() != result.to_lowercase() {
            result.push_str(format!(" | {}", self.filename).as_str());
        }

        result
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

/// Given stable atom serial numbers, reassign bond indices to match. Useful, for example, after
/// filtering a set of atoms and  bonds.
pub fn reassign_bond_indices(bonds: &mut [Bond], atoms: &[Atom]) {
    let sn_to_new_i: HashMap<_, _> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial_number, i)) // use usize if your bond fields are usize
        .collect();

    for b in bonds {
        b.atom_0 = sn_to_new_i[&b.atom_0_sn];
        b.atom_1 = sn_to_new_i[&b.atom_1_sn];
    }
}
