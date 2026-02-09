//! For determining which items the user has selected using the mouse cursor. Involves
//! mapping 2D to 3D coordinates, and choosing the right item from what's open and visible.

use std::{fmt, fmt::Display};

use bincode::{Decode, Encode};
use graphics::{ControlScheme, Scene};
use lin_alg::f32::Vec3 as Vec3F32;
use na_seq::{Element, Element::Hydrogen};

use crate::{
    drawing::MoleculeView,
    mol_editor::sync_md,
    mol_manip,
    mol_manip::ManipMode,
    molecules::{Atom, AtomRole, Bond, Chain, MolType, Residue, common::MoleculeCommon},
    state::{State, StateUi},
    util::{RedrawFlags, orbit_center},
};

#[derive(Clone, PartialEq, Debug, Default, Encode, Decode)]
pub enum Selection {
    #[default]
    None,
    /// Of the protein
    AtomPeptide(usize),
    /// Of the protein
    Residue(usize),
    /// Of the protein
    AtomsPeptide(Vec<usize>),
    /// Molecule index, atom index
    AtomLig((usize, usize)),
    /// Mol, set of atom indices
    AtomsLig((usize, Vec<usize>)),
    /// Molecule index, atom index
    AtomNucleicAcid((usize, usize)),
    /// Molecule index, atom index
    AtomLipid((usize, usize)),
    AtomPocket((usize, usize)),
    BondPeptide(usize),
    BondLig((usize, usize)),
    BondsLig((usize, Vec<usize>)),
    BondNucleicAcid((usize, usize)),
    BondLipid((usize, usize)),
    BondPocket((usize, usize)),
}

impl Selection {
    pub fn is_bond(&self) -> bool {
        matches!(
            self,
            Self::BondPeptide(_)
                | Self::BondLig(_)
                | Self::BondNucleicAcid(_)
                | Self::BondLipid(_)
                | Self::BondPocket(_)
        )
    }
}

// For hydrogens
const SELECTION_DIST_THRESH_H: f32 = 0.4; // e.g. ball + stick, or stick.
const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick, or stick.
const SELECTION_DIST_THRESH_BOND: f32 = 0.5; // e.g. ball + stick, or stick.
// Setting this high rel to `THRESH_SMALL` will cause more accidental selections of nearby atoms that
// the cursor is closer to the center of, but are behind the desired one.
// Setting it too low will cause the selector to "miss", even though the cursor is on an atom visual.
const SELECTION_DIST_THRESH_LARGE: f32 = 1.1; // e.g. VDW views like spheres.

const SEL_NEAR_PAD: f32 = 4.;

#[derive(Debug)]
struct Nearest {
    mol_type: MolType,
    mol_i: usize,
    atom_i: usize,
}

impl Nearest {
    pub fn indices(&self) -> (usize, usize) {
        (self.mol_i, self.atom_i)
    }
}

/// From under the cursor; pick the one near the ray, closest to the camera. This function is
/// run after the ray geometry is calculated, and is responsible for determining which atoms, residues, etc
/// are available for selection. It takes into account graphical filters, so only visible items
/// are selected.
///
/// Can select atoms from the protein, or ligand. Returns both the selection, and the distance;
/// we use this if running this for both atoms and bonds, so we can find the closest of the two.
///
/// todo: Refactor this to fix DRY among mol types.
pub fn find_selected_atom_or_bond(
    items_pep_along_ray: &[(usize, usize)],
    items_lig_along_ray: &[(usize, usize)],
    items_na_along_ray: &[(usize, usize)],
    items_lipid_along_ray: &[(usize, usize)],
    items_pocket_along_ray: &[(usize, usize)],
    atoms_pep: &[Atom],
    ress: &[Residue],
    atoms_lig: &[Vec<Atom>],
    atoms_na: &[Vec<Atom>],
    atoms_lipid: &[Vec<Atom>],
    atoms_pocket: &[Vec<Atom>],
    ray: &(Vec3F32, Vec3F32),
    ui: &StateUi,
    chains: &[Chain],
    // These aren't required if not in bond mode.
    bonds_pep: &[Bond],
    bonds_lig: &[Vec<Bond>],
    bonds_na: &[Vec<Bond>],
    bonds_lipid: &[Vec<Bond>],
    bonds_pocket: &[Vec<Bond>],
    bond_mode: bool,
    shift_held: bool,
) -> (Selection, f32) {
    if items_pep_along_ray.is_empty()
        && items_lig_along_ray.is_empty()
        && items_na_along_ray.is_empty()
        && items_lipid_along_ray.is_empty()
        && items_pocket_along_ray.is_empty()
    {
        return (Selection::None, 0.);
    }

    const INIT_DIST: f32 = f32::INFINITY;

    let ray_dir = (ray.1 - ray.0).to_normalized();
    let mut near_t = INIT_DIST;

    let mut nearest = Nearest {
        mol_type: MolType::Peptide,
        mol_i: 0,
        atom_i: 0,
    };
    let mut near_dist = INIT_DIST;

    for (i_mol, i_atom) in items_pep_along_ray {
        let chain_hidden = {
            let chains_this_atom: Vec<&Chain> =
                chains.iter().filter(|c| c.atoms.contains(i_atom)).collect();

            let mut hidden = false;
            for chain in &chains_this_atom {
                if !chain.visible {
                    hidden = true;
                    break;
                }
            }
            hidden
        };

        if chain_hidden {
            continue;
        }

        let atom = &atoms_pep[*i_atom];

        if ui.visibility.hide_sidechains || matches!(ui.mol_view, MoleculeView::Backbone) {
            if let Some(role) = atom.role
                && (role == AtomRole::Sidechain || role == AtomRole::H_Sidechain)
            {
                continue;
            }
        }

        if let Some(role) = atom.role {
            if ui.visibility.hide_sidechains && role == AtomRole::Sidechain {
                continue;
            }
            if role == AtomRole::Water
                && (ui.visibility.hide_water
                    || matches!(
                        ui.mol_view,
                        MoleculeView::SpaceFill | MoleculeView::Backbone
                    ))
            {
                continue;
            }
        }

        if ui.visibility.hide_hydrogen && atom.element == Hydrogen {
            continue;
        }

        if ui.visibility.hide_hetero && atom.hetero {
            continue;
        }

        if ui.visibility.hide_protein && !atom.hetero {
            continue;
        }

        let posit: Vec3F32 = if bond_mode {
            let bond = &bonds_pep[*i_atom];
            let atom_0 = &atoms_pep[bond.atom_0];
            let atom_1 = &atoms_pep[bond.atom_1];

            if ui.visibility.hide_hydrogen
                && (atom_0.element == Hydrogen || atom_1.element == Hydrogen)
            {
                continue;
            }

            ((atom_0.posit + atom_1.posit) / 2.).into()
        } else {
            let atom = &atoms_pep[*i_atom];

            if ui.visibility.hide_hydrogen && atom.element == Hydrogen {
                continue;
            }

            atom.posit.into()
        };

        let (dist_to_ray, t) = ray_metrics(ray.0, ray_dir, posit);
        if t < 0.0 {
            continue;
        }

        let eps = 1e-4;
        if dist_to_ray + eps < near_dist || ((dist_to_ray - near_dist).abs() <= eps && t < near_t) {
            nearest = Nearest {
                mol_type: MolType::Peptide,
                mol_i: *i_mol,
                atom_i: *i_atom,
            };
            near_dist = dist_to_ray;
            near_t = t;
        }
    }

    // Second pass for ligands; we skip most of the hidden checks here that apply to protein atoms.

    nearest_in_group(
        items_lig_along_ray,
        atoms_lig,
        bonds_lig,
        MolType::Ligand,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    nearest_in_group(
        items_na_along_ray,
        atoms_na,
        bonds_na,
        MolType::NucleicAcid,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    nearest_in_group(
        items_lipid_along_ray,
        atoms_lipid,
        bonds_lipid,
        MolType::Lipid,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    nearest_in_group(
        items_pocket_along_ray,
        atoms_pocket,
        bonds_pocket,
        MolType::Pocket,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    // This is equivalent to our empty check above, but catches the case of the atom count being
    // empty due to hidden chains.
    if near_dist == INIT_DIST {
        return (Selection::None, 0.);
    }

    let indices = nearest.indices();

    let sel = match ui.view_sel_level {
        // ViewSelLevel::Atom => match nearest.mol_type {
        ViewSelLevel::Atom | ViewSelLevel::Bond => {
            if bond_mode {
                match nearest.mol_type {
                    // todo: Rework this (with appropriate steps upstream). Get bonds along ray.
                    MolType::Peptide => Selection::BondPeptide(nearest.atom_i),
                    MolType::Ligand => {
                        if shift_held {
                            match &ui.selection {
                                Selection::BondLig((_mol_i_prev, bond_i_prev)) => {
                                    let updated = vec![*bond_i_prev];
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }
                                Selection::BondsLig((_mol_i_prev, atoms_i_prev)) => {
                                    let updated = atoms_i_prev.clone();
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }

                                _ => Selection::BondLig(indices),
                            }
                        } else {
                            Selection::BondLig(indices)
                        }
                    }
                    MolType::NucleicAcid => Selection::BondNucleicAcid(indices),
                    MolType::Lipid => Selection::BondLipid(indices),
                    MolType::Pocket => Selection::BondPocket(indices),
                    MolType::Water => unimplemented!(),
                }
            } else {
                match nearest.mol_type {
                    MolType::Peptide => {
                        if shift_held {
                            match &ui.selection {
                                Selection::AtomPeptide(atom_i) => {
                                    let updated = vec![*atom_i];
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Peptide,
                                        bond_mode,
                                    )
                                }
                                Selection::AtomsPeptide(atoms_i) => {
                                    let updated = atoms_i.clone();
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Peptide,
                                        bond_mode,
                                    )
                                }
                                _ => Selection::AtomPeptide(nearest.atom_i),
                            }
                        } else {
                            Selection::AtomPeptide(nearest.atom_i)
                        }
                    }
                    MolType::Ligand => {
                        if shift_held {
                            match &ui.selection {
                                Selection::AtomLig((_mol_i_prev, atom_i_prev)) => {
                                    let updated = vec![*atom_i_prev];
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }
                                Selection::AtomsLig((_mol_i_prev, atoms_i_prev)) => {
                                    let updated = atoms_i_prev.clone();
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }

                                _ => Selection::AtomLig(indices),
                            }
                        } else {
                            Selection::AtomLig(indices)
                        }
                    }
                    MolType::NucleicAcid => Selection::AtomNucleicAcid(indices),
                    MolType::Lipid => Selection::AtomLipid(indices),
                    _ => unreachable!(),
                }
            }
        }

        ViewSelLevel::Residue => {
            match nearest.mol_type {
                MolType::Peptide => {
                    for (i_res, _res) in ress.iter().enumerate() {
                        let atom_near = &atoms_pep[nearest.atom_i];
                        if let Some(i) = atom_near.residue
                            && i == i_res
                        {
                            return (Selection::Residue(i_res), near_dist);
                        }
                    }
                    Selection::None // Selected atom is not in a residue.
                }
                // These are the same as above.
                MolType::Ligand => Selection::AtomLig(indices),
                MolType::NucleicAcid => Selection::AtomNucleicAcid(indices),
                MolType::Lipid => Selection::AtomLipid(indices),
                _ => unreachable!(),
            }
        }
    };

    (sel, near_dist)
}

/// Helper
pub fn points_along_ray_inner(
    result: &mut Vec<(usize, usize)>,
    ray: &(Vec3F32, Vec3F32),
    ray_dir: Vec3F32,
    dist_thresh: f32,
    i_mol: usize,
    i_atom_or_bond: usize,
    posit: Vec3F32,
    el: Option<Element>,
) {
    // Compute the closest point on the ray to the atom position
    let to_atom: Vec3F32 = posit - ray.0;
    let t = to_atom.dot(ray_dir);
    let closest_point = ray.0 + ray_dir * t;

    // Compute the perpendicular distance to the ray
    let dist_to_ray = (posit - closest_point).magnitude();

    // todo: take atom radius into account. E.g. Hydrogens should required a smaller dist.
    // todo: This approach is a bit sloppy, but probably better than not including it.
    // if atom.element == Element::Hydrogen {
    //     // todo: This seems to prevent selecting at all; not sure why.
    //     // dist_thresh *= 0.9;
    // }

    // We render Hydrogens smaller; use a smaller thresh
    let mut thresh = dist_thresh;
    if el == Some(Hydrogen) && thresh == SELECTION_DIST_THRESH_SMALL {
        thresh = SELECTION_DIST_THRESH_H;
    }

    if dist_to_ray < thresh {
        result.push((i_mol, i_atom_or_bond));
    }
}

/// Used for cursor selection. Returns (atom indices prot, atom indices lig)
fn points_along_ray_atom(
    ray: (Vec3F32, Vec3F32),
    atoms: &[Vec<Atom>],
    dist_thresh: f32,
    // Each tuple is (mol i, atom i in that mol)
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i_mol, atoms_) in atoms.iter().enumerate() {
        for (i, atom) in atoms_.iter().enumerate() {
            points_along_ray_inner(
                &mut result,
                &ray,
                ray_dir,
                dist_thresh,
                i_mol,
                i,
                atom.posit.into(),
                Some(atom.element),
            );
        }
    }

    result
}

/// Special version, as peptides are currently only one molecule.
fn points_along_ray_atom_peptide(
    ray: (Vec3F32, Vec3F32),
    atoms: &[Atom],
    dist_thresh: f32,
    // Each tuple is (mol i, atom i in that mol)
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i, atom) in atoms.iter().enumerate() {
        points_along_ray_inner(
            &mut result,
            &ray,
            ray_dir,
            dist_thresh,
            0,
            i,
            atom.posit.into(),
            Some(atom.element),
        );
    }

    result
}

/// A helper
fn nearest_in_group(
    items: &[(usize, usize)],
    atoms: &[Vec<Atom>], // All atoms
    bonds: &[Vec<Bond>], // All bonds
    mol_type: MolType,
    bond_mode: bool,
    hide_h: bool,
    ray_origin: Vec3F32,
    ray_dir: Vec3F32,
    nearest: &mut Nearest,
    near_dist: &mut f32,
    near_t: &mut f32, // tie-break: depth along ray
) {
    let eps = 1e-4;

    for (i_mol, i_atom_bond) in items.iter() {
        let posit: Vec3F32 = if bond_mode {
            if *i_mol >= bonds.len() {
                eprintln!(
                    "Error: bond mode but i_mol {i_mol} >= bonds.len() : {:?}",
                    bonds
                );
                return;
            }

            let bond = &bonds[*i_mol][*i_atom_bond];
            let atom_0 = &atoms[*i_mol][bond.atom_0];
            let atom_1 = &atoms[*i_mol][bond.atom_1];

            if hide_h && (atom_0.element == Hydrogen || atom_1.element == Hydrogen) {
                continue;
            }

            ((atom_0.posit + atom_1.posit) / 2.).into()
        } else {
            let atom = &atoms[*i_mol][*i_atom_bond];
            if hide_h && atom.element == Hydrogen {
                continue;
            }
            atom.posit.into()
        };

        let (dist_to_ray, t) = ray_metrics(ray_origin, ray_dir, posit);
        if t < 0.0 {
            continue;
        }

        if dist_to_ray + eps < *near_dist
            || ((dist_to_ray - *near_dist).abs() <= eps && t < *near_t)
        {
            *nearest = Nearest {
                mol_type,
                mol_i: *i_mol,
                atom_i: *i_atom_bond,
            };
            *near_dist = dist_to_ray;
            *near_t = t;
        }
    }
}

/// Used for cursor selection. Returns (atom indices prot, atom indices lig)
pub fn points_along_ray_bond_(
    ray: (Vec3F32, Vec3F32),
    bonds_peptide: &[Bond],
    bonds_lig: &[Vec<Bond>],
    bonds_na: &[Vec<Bond>],
    bonds_lipid: &[Vec<Bond>],
    bonds_pocket: &[Vec<Bond>],
    // We need atoms to get positions
    atoms_peptide: &[Atom],
    atoms_lig: &[Vec<Atom>],
    atoms_na: &[Vec<Atom>],
    atoms_lipid: &[Vec<Atom>],
    atoms_pocket: &[Vec<Atom>],
    dist_thresh: f32,
    // Each tuple is (mol i, bondi in that mol)
) -> (
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
) {
    let mut result_prot = Vec::new();
    let mut result_lig = Vec::new();
    let mut result_na = Vec::new();
    let mut result_lipid = Vec::new();
    let mut result_pocket = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i, bond) in bonds_peptide.iter().enumerate() {
        let posit = (atoms_peptide[bond.atom_0].posit + atoms_peptide[bond.atom_1].posit) / 2.;
        points_along_ray_inner(
            &mut result_prot,
            &ray,
            ray_dir,
            dist_thresh,
            0,
            i,
            posit.into(),
            None,
        );
    }

    for (result, atoms_list, bonds_list) in [
        (&mut result_lig, &atoms_lig, &bonds_lig),
        (&mut result_na, &atoms_na, &bonds_na),
        (&mut result_lipid, &atoms_lipid, &bonds_lipid),
        (&mut result_pocket, &atoms_pocket, &bonds_pocket),
    ] {
        for (i_mol, bonds) in bonds_list.iter().enumerate() {
            for (i, bond) in bonds.iter().enumerate() {
                let a0 = atoms_list[i_mol][bond.atom_0].posit;
                let a1 = atoms_list[i_mol][bond.atom_1].posit;
                let posit = (a0 + a1) / 2.0;

                points_along_ray_inner(
                    result,
                    &ray,
                    ray_dir,
                    dist_thresh,
                    i_mol,
                    i,
                    posit.into(),
                    None,
                );
            }
        }
    }

    (
        result_prot,
        result_lig,
        result_na,
        result_lipid,
        result_pocket,
    )
}

/// Used for cursor selection. Returns (atom indices prot, atom indices lig)
pub fn points_along_ray_bond(
    ray: (Vec3F32, Vec3F32),
    bonds: &[Vec<Bond>],
    // We need atoms to get positions
    atoms: &[Vec<Atom>],
    dist_thresh: f32,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i_mol, bonds) in bonds.iter().enumerate() {
        for (i, bond) in bonds.iter().enumerate() {
            let a0 = atoms[i_mol][bond.atom_0].posit;
            let a1 = atoms[i_mol][bond.atom_1].posit;
            let posit = (a0 + a1) / 2.0;

            points_along_ray_inner(
                &mut result,
                &ray,
                ray_dir,
                dist_thresh,
                i_mol,
                i,
                posit.into(),
                None,
            );
        }
    }

    result
}

/// See note for atom version; slightly diff peptide structure.
pub fn points_along_ray_bond_peptide(
    ray: (Vec3F32, Vec3F32),
    bonds: &[Bond],
    // We need atoms to get positions
    atoms: &[Atom],
    dist_thresh: f32,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i, bond) in bonds.iter().enumerate() {
        let posit = (atoms[bond.atom_0].posit + atoms[bond.atom_1].posit) / 2.;
        points_along_ray_inner(
            &mut result,
            &ray,
            ray_dir,
            dist_thresh,
            0,
            i,
            posit.into(),
            None,
        );
    }

    result
}

// We're updating the position to use `atom_posits` instead of the internal `Atom` posit field,
// but this involves re-building.
// todo: I don't like this rebuilding.
fn get_atoms(mol: &MoleculeCommon) -> Vec<Atom> {
    // todo: I don't like this clone!
    mol.atoms
        .iter()
        .enumerate()
        .map(|(i, a)| Atom {
            posit: mol.atom_posits[i],
            element: a.element,
            residue: a.residue,
            ..Default::default()
        })
        .collect()
}

pub(crate) fn handle_selection_attempt(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
) {
    let Some(cursor) = state.ui.cursor_pos else {
        return;
    };

    let mut selected_ray = scene.screen_to_render(cursor);

    // Clip the near end of this to prevent false selections that seem to the user
    // to be behind the camera.
    let diff = selected_ray.1 - selected_ray.0;

    selected_ray.0 += diff.to_normalized() * SEL_NEAR_PAD;

    // todo: Lots of DRY here!

    let mut lig_atoms = Vec::new();
    for mol in &state.ligands {
        lig_atoms.push(get_atoms(&mol.common));
    }

    let mut na_atoms = Vec::new();
    for mol in &state.nucleic_acids {
        na_atoms.push(get_atoms(&mol.common));
    }
    let mut lipid_atoms = Vec::new();
    for mol in &state.lipids {
        lipid_atoms.push(get_atoms(&mol.common));
    }

    let mut pocket_atoms = Vec::new();
    for mol in &state.pockets {
        pocket_atoms.push(get_atoms(&mol.common));
    }

    let (pep_atoms, pep_res) = match &state.peptide {
        // Some(p) => (&p.common.atoms, &p.residues),
        Some(mol) => (&get_atoms(&mol.common), &mol.residues),
        None => (&Vec::new(), &Vec::new()),
    };

    // If we don't scale the selection distance appropriately, an atom etc
    // behind the desired one, but closer to the ray, may be selected; likely
    // this is undesired.
    let dist_thresh = match state.ui.mol_view {
        MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
        _ => match state.ui.view_sel_level {
            ViewSelLevel::Bond => SELECTION_DIST_THRESH_BOND,
            _ => SELECTION_DIST_THRESH_SMALL,
        },
    };

    let (sel_atoms, dist_atoms) = {
        let atoms_along_ray_pep =
            points_along_ray_atom_peptide(selected_ray, pep_atoms, dist_thresh);
        let atoms_along_ray_lig = points_along_ray_atom(selected_ray, &lig_atoms, dist_thresh);
        let atoms_along_ray_na = points_along_ray_atom(selected_ray, &na_atoms, dist_thresh);
        let atoms_along_ray_lipid = points_along_ray_atom(selected_ray, &lipid_atoms, dist_thresh);
        let atoms_along_ray_pocket =
            points_along_ray_atom(selected_ray, &pocket_atoms, dist_thresh);

        find_selected_atom_or_bond(
            &atoms_along_ray_pep,
            &atoms_along_ray_lig,
            &atoms_along_ray_na,
            &atoms_along_ray_lipid,
            &atoms_along_ray_pocket,
            pep_atoms,
            pep_res,
            &lig_atoms,
            &na_atoms,
            &lipid_atoms,
            &pocket_atoms,
            &selected_ray,
            &state.ui,
            &Vec::new(),
            &[],
            &[],
            &[],
            &[],
            &[],
            false,
            state.volatile.inputs_commanded.run,
        )
    };

    let (sel_bonds, dist_bonds) = {
        let mut pep_bonds = Vec::new();
        // todo: I don' tlike these clones.
        if let Some(mol) = &state.peptide {
            pep_bonds = mol.common.bonds.clone();
        }

        let mut lig_bonds = Vec::new();
        // todo: I don' tlike these clones.
        for mol in &state.ligands {
            lig_bonds.push(mol.common.bonds.clone());
        }

        let mut na_bonds = Vec::new();
        for mol in &state.nucleic_acids {
            na_bonds.push(mol.common.bonds.clone());
        }
        let mut lipid_bonds = Vec::new();
        for mol in &state.lipids {
            lipid_bonds.push(mol.common.bonds.clone());
        }

        let mut pocket_bonds = Vec::new();
        for mol in &state.pockets {
            pocket_bonds.push(mol.common.bonds.clone());
        }

        let atoms_along_ray_pep =
            points_along_ray_bond_peptide(selected_ray, &pep_bonds, pep_atoms, dist_thresh);
        let atoms_along_ray_lig =
            points_along_ray_bond(selected_ray, &lig_bonds, &lig_atoms, dist_thresh);
        let atoms_along_ray_lipid =
            points_along_ray_bond(selected_ray, &lipid_bonds, &lipid_atoms, dist_thresh);
        let atoms_along_ray_na =
            points_along_ray_bond(selected_ray, &na_bonds, &na_atoms, dist_thresh);
        let atoms_along_ray_pocket =
            points_along_ray_bond(selected_ray, &pocket_bonds, &pocket_atoms, dist_thresh);

        find_selected_atom_or_bond(
            &atoms_along_ray_pep,
            &atoms_along_ray_lig,
            &atoms_along_ray_na,
            &atoms_along_ray_lipid,
            &atoms_along_ray_pocket,
            pep_atoms,
            &Vec::new(), // todo: Peptide residues. once ready.
            &lig_atoms,
            &na_atoms,
            &lipid_atoms,
            &pocket_atoms,
            &selected_ray,
            &state.ui,
            &Vec::new(),
            &pep_bonds,
            &lig_bonds,
            &na_bonds,
            &pocket_bonds,
            &pocket_bonds,
            true,
            state.volatile.inputs_commanded.run,
        )
    };

    // Change the active molecule to the one of the selected atom or bond.
    match sel_atoms {
        Selection::AtomPeptide(_)
        | Selection::AtomsPeptide(_)
        | Selection::BondPeptide(_)
        | Selection::Residue(_) => {
            let mol_i = 0;
            state.volatile.active_mol = Some((MolType::Peptide, mol_i));
        }
        Selection::AtomLig((mol_i, _))
        | Selection::AtomsLig((mol_i, _))
        | Selection::BondLig((mol_i, _))
        | Selection::BondsLig((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::Ligand, mol_i));
        }
        Selection::AtomNucleicAcid((mol_i, _)) | Selection::BondNucleicAcid((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::NucleicAcid, mol_i));
        }
        Selection::AtomLipid((mol_i, _)) | Selection::BondLipid((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::Lipid, mol_i));
        }
        Selection::AtomPocket((mol_i, _)) | Selection::BondLipid((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::Lipid, mol_i));
        }
        _ => (),
    }

    let selection = if dist_atoms < dist_bonds {
        sel_atoms
    } else {
        sel_bonds
    };

    if selection == state.ui.selection {
        // Toggle.
        state.ui.selection = Selection::None;
    } else {
        state.ui.selection = selection;
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        state.volatile.orbit_center = state.volatile.active_mol;
        *center = orbit_center(state);
    }

    redraw.set_all();
}

/// A stripped-down version, for the mol editor. See notes there where applicable.
/// Note that this allows selecting *either atoms or bonds*, depending on where the click
/// occurs.
pub fn handle_selection_attempt_mol_editor(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut bool,
) {
    // todo: Allow a sel mode in the Primary mode that lets you pick either atoms or bonds, like this.

    let Some(cursor) = state.ui.cursor_pos else {
        return;
    };

    let mut selected_ray = scene.screen_to_render(cursor);

    let diff = selected_ray.1 - selected_ray.0;

    selected_ray.0 += diff.to_normalized() * SEL_NEAR_PAD;

    let dist_thresh = match state.ui.mol_view {
        MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
        _ => SELECTION_DIST_THRESH_SMALL,
    };

    let mol = &state.mol_editor.mol;

    let atoms: Vec<_> = mol
        .common
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| Atom {
            posit: mol.common.atom_posits[i],
            element: a.element,
            ..Default::default()
        })
        .collect();

    let (sel_atoms, dist_atoms) = {
        let atoms_along_ray_lig =
            points_along_ray_atom(selected_ray, &[atoms.clone()], dist_thresh);

        find_selected_atom_or_bond(
            &[],
            &atoms_along_ray_lig,
            &[],
            &[],
            &[],
            &Vec::new(),
            &Vec::new(),
            &[atoms.clone()], // todo: Don't like this.
            &[],
            &[],
            &[],
            &selected_ray,
            &state.ui,
            &Vec::new(),
            &Vec::new(),
            &[],
            &[],
            &[],
            &[],
            false,
            state.volatile.inputs_commanded.run,
        )
    };

    let (sel_bonds, dist_bonds) = {
        let bonds = mol.common.bonds.clone(); // todo: This clone...

        let bonds_along_ray_lig = points_along_ray_bond(
            selected_ray,
            &[bonds.clone()],
            &[atoms.clone()],
            dist_thresh,
        );

        find_selected_atom_or_bond(
            &[],
            &bonds_along_ray_lig,
            &[],
            &[],
            &[],
            &Vec::new(),
            &Vec::new(),
            &[atoms],
            &[],
            &[],
            &[],
            &selected_ray,
            &state.ui,
            &Vec::new(),
            &Vec::new(),
            &[bonds],
            &[],
            &[],
            &[],
            true,
            state.volatile.inputs_commanded.run,
        )
    };

    // todo: This fn is DRY with the non-editor version.

    let selection = if dist_atoms < dist_bonds {
        sel_atoms
    } else {
        sel_bonds
    };

    if selection == state.ui.selection {
        // Toggle.
        state.ui.selection = Selection::None;
    } else {
        state.ui.selection = selection;
    }

    // Adjust the atom or bond being manipulated, or deselect manipulation if appropriate.
    let manip_mode_new = match state.volatile.mol_manip.mode {
        ManipMode::Rotate((_mol_type, mol_i)) => {
            match &state.ui.selection {
                Selection::AtomLig(_) => ManipMode::None,
                Selection::BondLig(_) => ManipMode::Rotate((MolType::Ligand, mol_i)), // todo: QC
                _ => ManipMode::None,
            }
        }
        ManipMode::Move((_mol_type, mol_i)) => {
            match &state.ui.selection {
                Selection::AtomLig(_) => ManipMode::Move((MolType::Ligand, mol_i)), // todo: QC
                Selection::BondLig(_) => ManipMode::None,
                _ => ManipMode::None,
            }
        }
        _ => ManipMode::None,
    };

    let mut rebuild_md = false;
    let mut redraw_flags = RedrawFlags::default();
    redraw_flags.ligand = *redraw;

    mol_manip::set_manip(
        &mut state.volatile,
        &mut state.to_save.save_flag,
        scene,
        &mut redraw_flags,
        &mut rebuild_md,
        manip_mode_new,
        &state.ui.selection,
    );
    *redraw = redraw_flags.ligand;

    if rebuild_md {
        sync_md(state);
    }

    *redraw = true;
}

/// Handles logic regarding selection changes updating multi-atom lists, or reverting
/// to single-atom variants. Uses toggling behavior.
fn multi_sel_helper(
    mut updated: Vec<usize>,
    mol_i: usize,
    atom_or_bond_i: usize,
    mol_type: MolType,
    bond_mode: bool,
) -> Selection {
    if updated.contains(&atom_or_bond_i) {
        updated.retain(|idx| idx != &atom_or_bond_i);
    } else {
        updated.push(atom_or_bond_i);
    }

    // todo: We should handle the case where the mol i isn't from the previous mol.
    match updated.len() {
        0 => Selection::None,
        // Single-selection variants
        1 => {
            let result = (mol_i, updated[0]);
            match mol_type {
                MolType::Ligand => {
                    if bond_mode {
                        Selection::BondLig(result)
                    } else {
                        Selection::AtomLig(result)
                    }
                }
                MolType::Peptide => Selection::AtomPeptide(updated[0]),
                _ => unimplemented!(),
            }
        }
        // Multi-selection variants
        _ => match mol_type {
            MolType::Ligand => {
                if bond_mode {
                    Selection::BondsLig((mol_i, updated))
                } else {
                    Selection::AtomsLig((mol_i, updated))
                }
            }
            MolType::Peptide => Selection::AtomsPeptide(updated),
            _ => unimplemented!(),
        },
    }
}

// todo: Experimenting
fn ray_metrics(ray_origin: Vec3F32, ray_dir: Vec3F32, posit: Vec3F32) -> (f32, f32) {
    let to_p = posit - ray_origin;
    let t = to_p.dot(ray_dir);
    let closest = ray_origin + ray_dir * t;
    let dist_to_ray = (posit - closest).magnitude();
    (dist_to_ray, t)
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum ViewSelLevel {
    #[default]
    Atom,
    Bond,
    Residue,
}

impl ViewSelLevel {
    pub fn next(self) -> Self {
        match self {
            Self::Atom => Self::Bond,
            Self::Bond => Self::Residue,
            Self::Residue => Self::Atom,
        }
    }

    // todo: repetitive
    pub fn prev(self) -> Self {
        match self {
            Self::Atom => Self::Residue,
            Self::Bond => Self::Atom,
            Self::Residue => Self::Bond,
        }
    }
}

impl Display for ViewSelLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom => write!(f, "Atom"),
            Self::Residue => write!(f, "Residue"),
            Self::Bond => write!(f, "Bond"),
        }
    }
}
