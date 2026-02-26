//! This defines a `MoleculeCommon` struct, which is shared by all molecule types. It includes
//! the most important features, like atoms, bonds, and metadata.
//!

use crate::mol_editor::add_atoms::{hydrogens_avail, remove_hydrogens};
use bio_files::BondType;
use dynamics::{find_planar_posit, find_tetra_posit_final, find_tetra_posits};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;
use na_seq::Element::{Carbon, Hydrogen, Nitrogen, Oxygen};
use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU32, Ordering},
};

// // Used by the mol editor, and alignment. Be careful with this!
// pub static NEXT_ATOM_SN: AtomicU32 = AtomicU32::new(0);

use crate::molecules::{Atom, Bond, build_adjacency_list};

#[derive(Clone, Copy, PartialEq)]
pub enum BondGeom {
    Linear,
    Planar,
    Tetrahedral,
}

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
    /// A cache
    pub next_atom_sn: u32,
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
            next_atom_sn: 1,
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

    pub fn get_atom(&self, i: usize) -> Option<&Atom> {
        if i < self.atoms.len() {
            Some(&self.atoms[i])
        } else {
            None
        }
    }

    pub fn get_atom_mut(&mut self, i: usize) -> Option<&mut Atom> {
        if i < self.atoms.len() {
            Some(&mut self.atoms[i])
        } else {
            None
        }
    }

    pub fn get_bond(&self, i: usize) -> Option<&Bond> {
        if i < self.bonds.len() {
            Some(&self.bonds[i])
        } else {
            None
        }
    }

    pub fn get_bond_mut(&mut self, i: usize) -> Option<&mut Bond> {
        if i < self.bonds.len() {
            Some(&mut self.bonds[i])
        } else {
            None
        }
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
        // Same logic as `centroid`, but for local positions.
        let center = {
            let mut c = Vec3::new_zero();
            for atom in &self.atoms {
                c += atom.posit;
            }
            c / self.atoms.len() as f64
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

        self.next_atom_sn = match updated_sns.last() {
            Some(l) => *l + 1,
            None => 1,
        };
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
            let filename = self.filename.as_str();

            let (truncated, did_truncate) = if filename.chars().count() > 12 {
                let mut s: String = filename.chars().take(12).collect();
                s.push_str("...");
                (s, true)
            } else {
                (filename.to_string(), false)
            };

            // (did_truncate is unused but kept to make intent obvious; remove if you want)
            let _ = did_truncate;

            result.push_str(" | ");
            result.push_str(&truncated);
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

    /// Adds an atom, and a bond between it and an existing one [parent]. Also adds Hydrogens on this atom.
    /// Returns (atom's new index, bond's new index).
    pub fn add_atom(
        &mut self,
        i_par: usize,
        element: Element,
        bond_type: BondType,
        ff_type: Option<String>,
        bond_len: Option<f64>,
        q: Option<f32>,
    ) -> Option<(usize, usize)> {
        let posit_parent = self.atom_posits[i_par];
        let el_parent = self.atoms[i_par].element;

        if el_parent == Hydrogen {
            return None;
        }

        // Delete hydrogens; we'll add back if required.
        if element != Hydrogen {
            remove_hydrogens(self, i_par);
        }

        let atoms_to_add = bonds_avail(i_par, self, el_parent);
        let currently_bound_count = self.adjacency_list[i_par].len();

        // let geom = match atoms_to_add {
        let geom = match atoms_to_add + currently_bound_count {
            4 => BondGeom::Tetrahedral,
            3 => BondGeom::Planar,
            2 => BondGeom::Linear,
            _ => {
                eprintln!("Error: Unexpected atoms to add count.");
                BondGeom::Tetrahedral
            }
        };

        // todo: Can't use `common` below here due to the delete_atom code and ownership.
        let posit = match find_appended_posit(
            posit_parent,
            &self.atoms,
            &self.adjacency_list[i_par],
            bond_len,
            element,
            geom,
        ) {
            Some(p) => p,
            // Can't add an atom; already too many atoms bonded.
            None => return None,
        };

        let new_sn = self.next_atom_sn;
        self.next_atom_sn += 1;

        let i_new_atom = self.atoms.len();
        let i_new_bond = self.bonds.len();

        if i_par >= self.atoms.len() {
            eprintln!("Index out of range when adding atoms: {i_par}");
            return None;
            // todo: This return and print are a workaround; find the root cause.
        }

        let atom_new = Atom {
            serial_number: new_sn,
            posit,
            element: element.clone(),
            type_in_res: None,
            force_field_type: ff_type.clone(),
            partial_charge: q,
            ..Default::default()
        };

        self.atoms.push(atom_new);

        self.atom_posits.push(posit);
        self.adjacency_list[i_par].push(i_new_atom);
        self.adjacency_list.push(vec![i_par]);

        self.bonds.push(Bond {
            bond_type,
            atom_0_sn: self.atoms[i_par].serial_number,
            atom_1_sn: new_sn,
            atom_0: i_par,
            atom_1: i_new_atom,
            is_backbone: false,
        });

        Some((i_new_atom, i_new_bond))
    }

    /// Populate  hydrogens like the standalone editor fn, but only update the mol; no drawing/state
    /// updates  etc. We use this, for example, when loading molecules that don't hav eH.
    pub fn populate_hydrogens_on_atom(&mut self, i: usize) {
        // todo: Dry with the other fn.
        if i >= self.atoms.len() {
            eprintln!("Error: Invalid atom index when populating Hydrogens.");
            return;
        }

        let el = self.atoms[i].element;
        if el == Hydrogen {
            return;
        }

        let h_to_add = bonds_avail(i, self, el);

        // let bonds_remaining = bonds_avail.saturating_sub(adj.len());

        for _ in 0..h_to_add {
            let atom = &self.atoms[i];
            let (ff_type, bond_len) = {
                let mut v = (None, 1.1);

                // Grabbing the first, arbitrarily.
                for (ff, bl) in hydrogens_avail(&atom.force_field_type) {
                    v.0 = Some(ff);
                    v.1 = bl;
                    break;
                }

                v
            };

            // I believe we populate partial charge after  and ff  type  after?
            self.add_atom(i, Hydrogen, BondType::Single, ff_type, Some(bond_len), None);
        }
    }

    pub fn populate_hydrogens(&mut self) {
        self.update_next_sn();
        for i in 0..self.atoms.len() {
            self.populate_hydrogens_on_atom(i);
        }
    }

    pub fn update_next_sn(&mut self) {
        let mut highest_sn = 0;
        for atom in &self.atoms {
            if atom.serial_number > highest_sn {
                highest_sn = atom.serial_number;
            }
        }

        self.next_atom_sn = highest_sn + 1;
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

pub fn find_appended_posit(
    posit_parent: Vec3,
    atoms: &[Atom],
    adj_to_par: &[usize],
    bond_len: Option<f64>,
    element: Element,
    geom: BondGeom,
) -> Option<Vec3> {
    let neighbor_count = adj_to_par.len();

    // Note on these computations: The parent atom is the "hub" of a tetrahedral or planar
    // hub-and-spoke config. Other spokes are existing atoms bound to this parent, and the atom
    // we're computing the position here to add.
    let result = match neighbor_count {
        // This 0 branch should only be called for disconnected parents.
        0 => Some(posit_parent + Vec3::new(1.3, 0., 0.)),
        1 => {
            // This neighbor is the *grandparent* to the one we're adding; what ties
            // the parent to it. We set up the atom we're adding so it's tau/3 to this
            // grandparent, rotated around the parent.
            let grandparent = atoms[adj_to_par[0]].posit;

            // For now, pick an arbitrary orientation of the 3 methyl atoms (relative to the rest of the system)
            // without regard for steric clashes, and let MD sort it out after.
            // todo: choose something explicitly that avoids steric clashes?

            const TETRA_ANGLE: f64 = 1.91063;

            let bond_par_gp = (grandparent - posit_parent).to_normalized();
            let ax_rot = bond_par_gp.any_perpendicular();
            let rotator = Quaternion::from_axis_angle(ax_rot, TETRA_ANGLE);

            // If H, shorten the bond.
            let mut relative_dir = rotator.rotate_vec(bond_par_gp);
            if element == Hydrogen {
                relative_dir = (relative_dir.to_normalized()) * 1.1;
            }
            Some(posit_parent + relative_dir)
        }
        2 => {
            let neighbor_0 = atoms[adj_to_par[0]].posit;
            let neighbor_1 = atoms[adj_to_par[1]].posit;

            match geom {
                BondGeom::Tetrahedral => {
                    // This function uses the distance between the first two params, so it's likely
                    // in the case of adding H, this is what we want. (?)
                    let (p0, p1) = find_tetra_posits(posit_parent, neighbor_1, neighbor_0);

                    // Score a candidate by its minimum distance to any existing neighbor; pick the larger score.
                    let neighbors: &[usize] = &adj_to_par;
                    let score = |p: Vec3| {
                        let mut best = f64::INFINITY;
                        for &ni in neighbors {
                            let q = atoms[ni].posit;
                            let d2 = (p - q).magnitude_squared();
                            if d2 < best {
                                best = d2;
                            }
                        }
                        best
                    };

                    Some(if score(p0) >= score(p1) { p0 } else { p1 })
                }
                BondGeom::Planar => Some(find_planar_posit(posit_parent, neighbor_0, neighbor_1)),
                BondGeom::Linear => {
                    return None;
                }
            }
        }
        3 => {
            if geom != BondGeom::Tetrahedral {
                return None;
            }

            // None
            let adj_0 = adj_to_par[0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_to_par[1];
            let neighbor_1 = atoms[adj_1].posit;
            let adj_2 = adj_to_par[2];
            let neighbor_2 = atoms[adj_2].posit;

            // todo. Check both angles?
            // If the incoming angles are ~τ/3, add in a planar config.
            // let bond_0 = neighbor_0 - posit_parent;
            // let bond_1 = neighbor_1 - posit_parent;
            // let angle = bond_1.to_normalized().dot(bond_0.to_normalized()).acos();

            // Planar; full.
            // if angle > 1.95 {
            // todo: Experiment. You may wish to use the character of neighboring bond count
            // todo and type instead of this angle.
            // if angle > 2.11 {
            //     println!("Planar abort!: {angle}"); // todo temp!!
            //     return None;
            // } else {
            Some(find_tetra_posit_final(
                posit_parent,
                neighbor_0,
                neighbor_1,
                neighbor_2,
            ))
            // }
        }
        _ => None,
    };

    // Set len, if applicable.
    // todo: Could be slightly more efficient to bake this length correction into the find_tetra
    // todo etc fns.
    match result {
        Some(p) => match bond_len {
            Some(l) => {
                let rel_pos = (p - posit_parent).to_normalized() * l;
                Some(posit_parent + rel_pos)
            }
            None => Some(p),
        },
        None => None,
    }
}

pub fn bonds_avail(i_atom: usize, mol: &MoleculeCommon, el: Element) -> usize {
    let mut bonds_avail: isize = match el {
        Carbon => 4,
        Oxygen => 2,
        Nitrogen => 3, // todo?
        Element::Chlorine => 0,
        _ => 0, // todo?
    };

    let mut ar_count = 0;
    for bond in &mol.bonds {
        if bond.atom_0 != i_atom && bond.atom_1 != i_atom {
            continue;
        }

        match bond.bond_type {
            BondType::Single => bonds_avail -= 1,
            BondType::Double => bonds_avail -= 2,
            BondType::Triple => bonds_avail -= 3,
            BondType::Aromatic => {
                ar_count += 1;
                // bonds_avail -= 2
            }
            _ => bonds_avail -= 1,
        }
    }

    // Special override for the non-integer case of Aromatic bonds (4 - 1.5 x 2 = 1)
    if ar_count == 2 {
        bonds_avail -= 3;
    }

    if bonds_avail < 0 {
        0
    } else {
        bonds_avail as usize
    }
}
