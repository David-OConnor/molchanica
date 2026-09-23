//! For drawing peptides/proteins. This is a separate module from that used for
//! other molecule types due to differences in how we render it. It also includes code for
//! drawing modes which only apply to peptides, e.g. ribbon and solvent-accessible-surface.
//!
//! Note: It may be possible/desirable to consolidate the draw function with the other types, e.g. in the
//! `atoms_bonds` module.

use std::collections::HashMap;

use bio_files::{BondType, ResidueType};
use egui::FontFamily;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene, TextOverlay};
use lin_alg::f32::{Quaternion, Vec3};
use mol_defs::{
    molecules::{Atom, AtomRole, Chain, MolType, peptide::MoleculePeptide},
    reflection::DensityPt,
    sfc_mesh::{SOLVENT_RAD, make_sas_mesh},
};
use na_seq::Element;

use crate::{
    drawing,
    drawing::{
        BLEND_AMT_HETERO_RES, COLOR_HETERO_RES, COLOR_MOL_MOVING, COLOR_MOL_ROTATE,
        COLOR_SA_SURFACE, COLOR_SECONDARY_STRUCTURE, COLOR_SELECTED, COLOR_SFC_DOT,
        DENSITY_ISO_OPACITY, EntityClass, LABEL_COLOR_ATOM, LABEL_COLOR_MOL, LABEL_COLOR_MOL_SEL,
        LABEL_SIZE_CHAIN, LABEL_SIZE_MOL_LARGE, MESH_BALL_STICK_SPHERE, MESH_SPACEFILL_SPHERE,
        MESH_SURFACE_DOT, MESH_WATER_SPHERE, MoleculeView, SAS_ISO_OPACITY, SIZE_SFC_DOT,
        atoms_bonds,
        atoms_bonds::{
            ATOM_SHININESS, BALL_RADIUS_WATER_O, BALL_STICK_RADIUS, BALL_STICK_RADIUS_H,
            draw_hydrogen_bond,
        },
        effective_mol_view_peptide,
    },
    mol_manip::{ManipMode, PeptideMeshTransform},
    render::{
        MESH_CUBE, MESH_DENSITY_SURFACE, MESH_OTHER_RIBBONS, MESH_PEP_SOLVENT_SURFACE,
        MESH_SECONDARY_STRUCTURE, MESH_SPHERE_LOWRES,
    },
    selection::Selection,
    state::{DistFilter, OperatingMode, ResColoring, State, StateUi},
    util::{aromatic_ring_centroid, clear_mol_entity_indices, find_neighbor_posit, orbit_center},
};

/// A visual representation of volumetric electron density,
/// as loaded from .map files or similar. This is our point-based approach; not the isosurface.
/// We change size based on density, and not linearly, for visual effect.
pub fn draw_density_point_cloud(entities: &mut Vec<Entity>, density: &[DensityPt]) {
    entities.retain(|ent| ent.class != EntityClass::DensityPoint as u32);
    // clear_mol_entity_indices(state); // todo: Borrow mut problem.

    // const EPS: f64 = 0.0000001;

    for point in density {
        // For example, points we filter out for not being near the atoms; we set them to 0 density,
        // vice omitting them. Skipping them here makes rendering more efficient.
        // if point.density.abs() < EPS {
        //     continue;
        // }
        if point.density == 0. {
            continue;
        }

        // Todo: Sort out how you'll handle this. Currently, You discard these, or they'd go NaN
        // todo on the power computation.
        if point.density < 0.0 {
            continue;
        }

        let mut ent = Entity::new(
            MESH_SPHERE_LOWRES,
            point.coords.into(),
            Quaternion::new_identity(),
            0.03 * point.density.powf(1.3) as f32,
            (point.density as f32 * 2., 0.0, 0.2),
            ATOM_SHININESS,
        );
        ent.class = EntityClass::DensityPoint as u32;

        entities.push(ent);
    }
}

/// An isosurface of electron density,
/// as loaded from .map files or similar.
pub fn draw_density_surface(
    entities: &mut Vec<Entity>,
    state: &mut State,
    updates: &mut EngineUpdates,
) {
    entities.retain(|ent| ent.class != EntityClass::DensitySurface as u32);
    clear_mol_entity_indices(state, None);

    let mut ent = Entity::new(
        MESH_DENSITY_SURFACE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0., 1., 1.),
        ATOM_SHININESS,
    );
    ent.class = EntityClass::DensitySurface as u32;
    ent.opacity = DENSITY_ISO_OPACITY;
    entities.push(ent);

    updates.entities = EntityUpdate::All;
}

/// The dots view of solvent-accessible-surface
fn draw_dots(
    update_mesh: &mut bool,
    mesh_created: bool,
    transform: PeptideMeshTransform,
    scene: &mut Scene,
) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    if scene.meshes[MESH_PEP_SOLVENT_SURFACE].vertices.len() > 1_000_000 {
        eprintln!("Not drawing dots due to a large-mol rendering problem.");
        return;
    }

    for vertex in &scene.meshes[MESH_PEP_SOLVENT_SURFACE].vertices {
        let position = Vec3::from_slice(&vertex.position).unwrap();
        let mut entity = Entity::new(
            MESH_SURFACE_DOT,
            transform.transform_point(position),
            Quaternion::new_identity(),
            SIZE_SFC_DOT,
            COLOR_SFC_DOT,
            ATOM_SHININESS,
        );
        entity.class = EntityClass::SaSurfaceDots as u32;
        scene.entities.push(entity);
    }
}

/// The mesh view of solvent-accessible-surface
// fn draw_sa_surface(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene, color_by_vertex: Option<Vec<(u8, u8, u8)>>) {
fn draw_sa_surface(
    update_mesh: &mut bool,
    mesh_created: bool,
    transform: PeptideMeshTransform,
    scene: &mut Scene,
) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    let mut ent = Entity::new(
        MESH_PEP_SOLVENT_SURFACE,
        transform.translation,
        transform.rotation,
        1.,
        COLOR_SA_SURFACE,
        ATOM_SHININESS,
    );

    ent.class = EntityClass::SaSurface as u32;
    ent.opacity = SAS_ISO_OPACITY;
    // if let Some(color) = color_by_vertex {
    //     ent.color_by_vertex = Some(color)
    // }

    scene.entities.push(ent);
}

/// In ribbon/cartoon view, atoms and bonds are not drawn, so `text_overlay` never fires.
/// This function creates tiny invisible entities at the first atom of each chain (for chain
/// labels) and at atom 0 (for the molecule label), mirroring the logic in `text_overlay` but
/// without the atom-SN label that makes no sense for a ribbon.
fn ribbon_text_overlay_entities(
    mol: &MoleculePeptide,
    mol_active: bool,
    ui: &StateUi,
) -> Vec<Entity> {
    let mut result = Vec::new();

    if ui.visibility.labels.mol {
        if let Some(&posit) = mol.common.atom_posits.first() {
            let color = if mol_active {
                LABEL_COLOR_MOL_SEL
            } else {
                LABEL_COLOR_MOL
            };

            let mut ent = Entity::new(
                MESH_CUBE,
                posit.into(),
                Quaternion::new_identity(),
                0.01,
                (0., 0., 0.),
                ATOM_SHININESS,
            );

            ent.overlay_text = Some(TextOverlay {
                text: mol
                    .common
                    .name
                    .as_deref()
                    .unwrap_or(&mol.common.ident)
                    .to_owned(),
                size: LABEL_SIZE_MOL_LARGE,
                color,
                font_family: FontFamily::Proportional,
            });

            ent.class = EntityClass::Protein as u32;
            result.push(ent);
        }
    }

    if ui.visibility.labels.chain {
        for chain in &mol.chains {
            if !chain.visible {
                continue;
            }
            let Some(&first_atom_i) = chain.atoms.first() else {
                continue;
            };
            let Some(&posit) = mol.common.atom_posits.get(first_atom_i) else {
                continue;
            };
            let mut ent = Entity::new(
                MESH_CUBE,
                posit.into(),
                Quaternion::new_identity(),
                0.01,
                (0., 0., 0.),
                ATOM_SHININESS,
            );
            ent.overlay_text = Some(TextOverlay {
                text: chain.id.clone(),
                size: LABEL_SIZE_CHAIN,
                color: LABEL_COLOR_ATOM,
                font_family: FontFamily::Proportional,
            });
            ent.class = EntityClass::Protein as u32;
            result.push(ent);
        }
    }

    result
}

/// Secondary structure, e.g. cartoon view for proteins.
pub fn draw_secondary_structure(
    update_mesh: &mut bool,
    mesh_created: bool,
    transform: PeptideMeshTransform,
    mesh_i: usize,
    scene: &mut Scene,
) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
    }

    let mut ent = Entity::new(
        mesh_i,
        transform.translation,
        transform.rotation,
        1.,
        COLOR_SECONDARY_STRUCTURE,
        ATOM_SHININESS,
    );
    ent.class = EntityClass::SecondaryStructure as u32;
    scene.entities.push(ent);
}

/// For the surface filter, the distance threshold is scaled by this to get a depth below the surface,
/// in Å. Depths of interest are much shallower than distances for the selection and ligand filters:
/// For typical proteins, ~half of atoms are exposed, and nearly all are within 6Å of the surface.
pub const SFC_DIST_SCALE: f32 = 0.1;
/// Grid spacing of the surface filter's SAS mesh, in Å. Higher is faster, but cruder.
const SFC_MESH_PRECISION: f32 = 2.;

/// Buckets points into cubic cells, for quickly checking if a point is near any of them.
struct PointGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<Vec3>>,
}

impl PointGrid {
    /// `cell_size` must be at least the largest distance passed to `any_within`.
    fn new(pts: &[Vec3], cell_size: f32) -> Self {
        let mut result = Self {
            cell_size: cell_size.max(1.),
            cells: HashMap::new(),
        };

        for pt in pts {
            let key = result.key(*pt);
            result.cells.entry(key).or_default().push(*pt);
        }

        result
    }

    fn key(&self, pt: Vec3) -> (i32, i32, i32) {
        (
            (pt.x / self.cell_size).floor() as i32,
            (pt.y / self.cell_size).floor() as i32,
            (pt.z / self.cell_size).floor() as i32,
        )
    }

    /// If any point is within `dist` of `pt`.
    fn any_within(&self, pt: Vec3, dist: f32) -> bool {
        let dist_sq = dist.powi(2);
        let (x, y, z) = self.key(pt);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(cell) = self.cells.get(&(x + dx, y + dy, z + dz)) else {
                        continue;
                    };

                    if cell.iter().any(|p| (*p - pt).magnitude_squared() < dist_sq) {
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// Positions of the selected atoms, of any molecule type. For bonds, this is both of their atoms;
/// for residues, all of their atoms.
fn sel_posits(state: &State) -> Vec<Vec3> {
    // Peptide selections are of this one.
    let pep_i = state.peptide_for_tools_i();

    let res_atoms = |res_is: &[usize]| -> Vec<usize> {
        let Some(pep) = pep_i.and_then(|i| state.peptides.get(i)) else {
            return Vec::new();
        };

        res_is
            .iter()
            .filter_map(|i| pep.residues.get(*i))
            .flat_map(|res| res.atoms.iter().copied())
            .collect()
    };

    let none = Vec::new;

    // Molecule type, molecule index, atom indices, bond indices.
    let (mol_type, mol_i, atoms, bonds) = match &state.ui.selection {
        Selection::AtomPeptide(i) => (MolType::Peptide, pep_i, vec![*i], none()),
        Selection::AtomsPeptide(is) => (MolType::Peptide, pep_i, is.clone(), none()),
        Selection::BondPeptide(i) => (MolType::Peptide, pep_i, none(), vec![*i]),
        Selection::Residue(i) => (MolType::Peptide, pep_i, res_atoms(&[*i]), none()),
        Selection::Residues(is) => (MolType::Peptide, pep_i, res_atoms(is), none()),

        Selection::AtomLig((m, i)) => (MolType::Ligand, Some(*m), vec![*i], none()),
        Selection::AtomsLig((m, is)) => (MolType::Ligand, Some(*m), is.clone(), none()),
        Selection::BondLig((m, i)) => (MolType::Ligand, Some(*m), none(), vec![*i]),
        Selection::BondsLig((m, is)) => (MolType::Ligand, Some(*m), none(), is.clone()),

        Selection::AtomNucleicAcid((m, i)) => (MolType::NucleicAcid, Some(*m), vec![*i], none()),
        Selection::BondNucleicAcid((m, i)) => (MolType::NucleicAcid, Some(*m), none(), vec![*i]),

        Selection::AtomLipid((m, i)) => (MolType::Lipid, Some(*m), vec![*i], none()),
        Selection::BondLipid((m, i)) => (MolType::Lipid, Some(*m), none(), vec![*i]),

        Selection::AtomPocket((m, i)) => (MolType::Pocket, Some(*m), vec![*i], none()),
        Selection::BondPocket((m, i)) => (MolType::Pocket, Some(*m), none(), vec![*i]),

        Selection::None | Selection::ComponentEditor(_) => return Vec::new(),
    };

    let Some(mol) = mol_i.and_then(|i| state.get_mol(mol_type, i)) else {
        return Vec::new();
    };
    let common = mol.common();

    let bond_atoms = bonds
        .iter()
        .filter_map(|i| common.bonds.get(*i))
        .flat_map(|b| [b.atom_0, b.atom_1]);

    atoms
        .into_iter()
        .chain(bond_atoms)
        .filter_map(|i| common.atom_posits.get(i))
        .map(|p| (*p).into())
        .collect()
}

/// Hides each atom not within `thresh` of any of `targets`. If there are no targets, hides nothing.
fn hide_far_from(posits: &[Vec3], targets: &[Vec3], thresh: f32) -> Vec<bool> {
    if targets.is_empty() {
        return vec![false; posits.len()];
    }

    let grid = PointGrid::new(targets, thresh);

    posits
        .iter()
        .map(|p| !grid.any_within(*p, thresh))
        .collect()
}

/// Hides protein atoms deeper than `thresh` below the protein's surface. We measure depth as an atom's
/// distance to the solvent-accessible surface mesh, less its VdW radius and the probe radius;
/// exposed atoms have a depth near 0. Hetero atoms, e.g. ligands and water, aren't hidden.
fn hide_far_from_sfc(pep: &MoleculePeptide, posits: &[Vec3], thresh: f32) -> Vec<bool> {
    let mol = &pep.common;

    let atoms: Vec<(Vec3, f32)> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.hetero)
        .map(|(i, a)| (posits[i], a.element.vdw_radius()))
        .collect();

    // todo: Don't create this each drawing! Cache the atoms near the sfc pre-computed.
    let mesh = make_sas_mesh(&atoms, SOLVENT_RAD, SFC_MESH_PRECISION);

    let sfc_pts: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|v| Vec3::from_slice(&v.position).unwrap())
        .collect();

    let vdw_max = mol
        .atoms
        .iter()
        .map(|a| a.element.vdw_radius())
        .fold(0., f32::max);

    let grid = PointGrid::new(&sfc_pts, thresh + SOLVENT_RAD + vdw_max);

    mol.atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let dist = thresh + SOLVENT_RAD + atom.element.vdw_radius();
            !atom.hetero && !grid.any_within(posits[i], dist)
        })
        .collect()
}

// todo: Move this A/R. Util? Molecule? Method on peptide?
/// Filter by distance to the selection, active ligand, or protein surface; see `StateUi::dist_filter`.
/// Returns a flag for each of the peptide's atoms; `true` means it's filtered out.
pub fn filter_pep_atoms_by_dist(state: &State, pep: &MoleculePeptide) -> Vec<bool> {
    let posits: Vec<Vec3> = pep.common.atom_posits.iter().map(|p| (*p).into()).collect();
    let thresh = state.ui.nearby_dist_thresh as f32;

    match state.ui.dist_filter {
        DistFilter::None => vec![false; posits.len()],
        DistFilter::NearSel => hide_far_from(&posits, &sel_posits(state), thresh),
        DistFilter::NearLig => {
            // Don't treat the protein itself as the ligand; e.g. after clicking one of its atoms.
            let lig_posits: Vec<Vec3> = match state.volatile.active_mol {
                Some((mol_type, i)) if mol_type != MolType::Peptide => state
                    .get_mol(mol_type, i)
                    .map(|m| m.common().atom_posits.iter().map(|p| (*p).into()).collect())
                    .unwrap_or_default(),
                _ => Vec::new(),
            };

            hide_far_from(&posits, &lig_posits, thresh)
        }
        DistFilter::NearSfc => hide_far_from_sfc(pep, &posits, thresh * SFC_DIST_SCALE),
    }
}

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_peptide(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    if state.ui.mol_view_peptide == MoleculeView::Ribbon {
        if state.volatile.flags.ss_mesh_peptide != state.peptide_for_tools_i() {
            state.volatile.flags.ss_mesh_created = false;
            state.volatile.flags.update_ss_mesh = true;
        }
        if state.volatile.flags.ss_mesh_dirty {
            state.volatile.flags.update_ss_mesh = true;
            state.volatile.flags.ss_mesh_dirty = false;
        }
        let visible: Vec<bool> = state
            .peptides
            .iter()
            .map(|mol| mol.common.visible)
            .collect();
        if state.volatile.flags.ss_mesh_visible != visible {
            state.volatile.flags.update_ss_mesh = true;
        }
    }

    scene.entities.retain(|ent| {
        ent.class != EntityClass::Protein as u32
            && ent.class != EntityClass::SaSurface as u32
            && ent.class != EntityClass::SaSurfaceDots as u32
            && ent.class != EntityClass::SecondaryStructure as u32
    });
    for peptide in &mut state.peptides {
        peptide.common.entity_i_range = None;
    }

    // Edit small molecules only; not proteins.
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        return;
    }

    for mol_i in 0..state.peptides.len() {
        draw_peptide_one(state, scene, mol_i);
    }

    let selected_peptide = state.peptide_for_tools_i();
    let has_other_ribbons = state.peptides.iter().enumerate().any(|(mol_i, mol)| {
        Some(mol_i) != selected_peptide
            && mol.common.visible
            && effective_mol_view_peptide(state, mol_i) == MoleculeView::Ribbon
    });
    if has_other_ribbons {
        draw_secondary_structure(
            &mut state.volatile.flags.update_ss_mesh,
            state.volatile.flags.ss_mesh_created,
            PeptideMeshTransform::default(),
            MESH_OTHER_RIBBONS,
            scene,
        );
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }

    updates.entities = EntityUpdate::All;
    // Removing and re-appending protein entities can shift every later entity range even when the
    // total entity count stays unchanged.
    clear_mol_entity_indices(state, Some(MolType::Peptide));
}

fn draw_peptide_one(state: &mut State, scene: &mut Scene, mol_i: usize) {
    let mol_active = if let Some((active_mol_type, active_i)) = state.volatile.active_mol {
        MolType::Peptide == active_mol_type && mol_i == active_i
    } else {
        false
    };

    let Some(mol) = state.peptides.get(mol_i) else {
        return;
    };
    if !mol.common.visible {
        return;
    }

    let filtered_out_by_dist = filter_pep_atoms_by_dist(state, mol);

    let start_i = scene.entities.len();
    let mut entities = Vec::new();

    // todo:  Unless colored by res #, set to 0 to save teh computation.
    let aa_count = mol
        .residues
        .iter()
        .filter(|r| matches!(r.res_type, ResidueType::AminoAcid(_)))
        .count();

    let ui = &state.ui;
    let owns_shared_mesh = state.peptide_for_tools_i() == Some(mol_i);
    // Dots and solvent-surface views still use shared mesh slots. Ribbons for other peptides
    // are collected into a second mesh by the scene flag handler.
    let mol_view = effective_mol_view_peptide(state, mol_i);

    if owns_shared_mesh && mol_view == MoleculeView::Ribbon {
        let transform =
            if state.volatile.flags.ss_mesh_created && !state.volatile.flags.update_ss_mesh {
                state.volatile.mol_manip.ribbon_mesh_transform
            } else {
                PeptideMeshTransform::default()
            };

        draw_secondary_structure(
            &mut state.volatile.flags.update_ss_mesh,
            state.volatile.flags.ss_mesh_created,
            transform,
            MESH_SECONDARY_STRUCTURE,
            scene,
        );
    }
    if mol_view == MoleculeView::Ribbon {
        entities.extend(ribbon_text_overlay_entities(mol, mol_active, ui));
    }

    // Note that this renders over a sticks model.
    if owns_shared_mesh
        && !state.ui.visibility.hide_protein
        && mol_view == MoleculeView::Dots
        && !state.volatile.mol_manip.peptide_mesh_manip_pending
    {
        let transform = state.volatile.mol_manip.surface_mesh_transform;
        draw_dots(
            &mut state.volatile.flags.update_sas_mesh,
            state.volatile.flags.sas_mesh_created,
            transform,
            scene,
        );
    }

    // todo: Consider if you handle this here, or in a sep fn.
    if owns_shared_mesh && !state.ui.visibility.hide_protein && mol_view == MoleculeView::Surface {
        let transform = if state.volatile.flags.sas_mesh_created {
            state.volatile.mol_manip.surface_mesh_transform
        } else {
            PeptideMeshTransform::default()
        };
        draw_sa_surface(
            &mut state.volatile.flags.update_sas_mesh,
            state.volatile.flags.sas_mesh_created,
            transform,
            scene,
        );
    }

    let chains_invis: Vec<&Chain> = mol.chains.iter().filter(|c| !c.visible).collect();

    let sel = if !mol_active || ui.selection.is_bond() {
        &Selection::None
    } else {
        &ui.selection
    };

    // If sticks view, draw water molecules as balls.
    if matches!(mol_view, MoleculeView::Sticks | MoleculeView::BallAndStick)
        && !state.ui.visibility.hide_water
    {
        for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
            if atom.hetero {
                // todo: Excessive nesting.
                if let Some(role) = atom.role
                    && role == AtomRole::Water
                {
                    let color_atom = atoms_bonds::atom_color(
                        atom,
                        0,
                        i_atom,
                        &mol.residues,
                        mol.sifts_mapping.as_deref(),
                        aa_count,
                        mol.chains.len(),
                        &mol.common.atoms,
                        &mol.chains,
                        sel,
                        state.ui.view_sel_level,
                        false,
                        ResColoring::default(),
                        false,
                        MolType::Peptide,
                        &None,
                    );

                    let mut entity = Entity::new(
                        MESH_WATER_SPHERE,
                        mol.common.atom_posits[i_atom].into(),
                        Quaternion::new_identity(),
                        BALL_RADIUS_WATER_O,
                        color_atom,
                        ATOM_SHININESS,
                    );

                    entity.class = EntityClass::Protein as u32;
                    entities.push(entity);
                }
            }
        }
    }

    // Ligands, ions etc. that are part of the peptide are drawn using the small-molecule view,
    // rather than the peptide one. Water keeps the peptide view.
    let het_view = ui.mol_view.non_peptide_or_default();
    let is_lig_atom = |atom: &Atom| atom.hetero && atom.role != Some(AtomRole::Water);

    // Draw atoms.
    for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
        let lig_atom = is_lig_atom(atom);
        let view = if lig_atom { het_view } else { mol_view };

        if !matches!(view, MoleculeView::BallAndStick | MoleculeView::SpaceFill) {
            continue;
        }

        let mut chain_not_sel = false;
        for chain in &chains_invis {
            if chain.atoms.contains(&i_atom) {
                chain_not_sel = true;
                break;
            }
        }
        if chain_not_sel {
            continue;
        }

        if state.ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
        }

        if filtered_out_by_dist[i_atom] {
            continue;
        }

        if let Some(role) = atom.role {
            // Note: Ligand atoms have the sidechain role, from their atom names.
            if !lig_atom
                && (state.ui.visibility.hide_sidechains || mol_view == MoleculeView::Backbone)
                && matches!(role, AtomRole::Sidechain | AtomRole::H_Sidechain)
            {
                continue;
            }
            if (state.ui.visibility.hide_water || mol_view == MoleculeView::SpaceFill)
                && role == AtomRole::Water
            {
                continue;
            }
        }

        if (state.ui.visibility.hide_hetero && atom.hetero)
            || (state.ui.visibility.hide_protein && !atom.hetero)
        {
            continue;
        }

        let atom_posit = mol.common.atom_posits[i_atom];

        // todo: Use your new peptide field for filtered, instead of computing these each time.

        let (mut radius, mesh) = match view {
            MoleculeView::SpaceFill => (atom.element.vdw_radius(), MESH_SPACEFILL_SPHERE),
            _ => match atom.element {
                Element::Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
                _ => (BALL_STICK_RADIUS, MESH_BALL_STICK_SPHERE),
            },
        };

        if let Some(role) = atom.role
            && role == AtomRole::Water
        {
            radius = BALL_RADIUS_WATER_O
        }

        let dim_peptide = state.ui.visibility.dim_peptide && !atom.hetero;

        let mut color_atom = (0., 0., 0.);
        let mut manip_active = false;

        match state.volatile.mol_manip.mode {
            ManipMode::Move((mol_type, i)) => {
                if mol_type == MolType::Peptide && i == mol_i {
                    color_atom = COLOR_MOL_MOVING;
                    manip_active = true;
                }
            }
            ManipMode::Rotate((mol_type, i)) => {
                if mol_type == MolType::Peptide && i == mol_i {
                    color_atom = COLOR_MOL_ROTATE;
                    manip_active = true;
                }
            }
            ManipMode::None => (),
        }

        if !manip_active {
            color_atom = atoms_bonds::atom_color(
                atom,
                0,
                i_atom,
                &mol.residues,
                mol.sifts_mapping.as_deref(),
                aa_count,
                mol.chains.len(),
                &mol.common.atoms,
                &mol.chains,
                sel,
                state.ui.view_sel_level,
                dim_peptide,
                state.ui.res_coloring,
                state.ui.atom_color_by_charge,
                MolType::Peptide,
                &None,
            );
        }

        if atom.hetero && color_atom != COLOR_SELECTED {
            color_atom = drawing::blend_color(color_atom, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        // todo: Come back to this.
        // if state.volatile.md_local.mol_dynamics.is_some()
        //     && state.ui.md.peptide_only_near_ligs
        //     && mol.common.selected_for_md
        //     && state
        //         .ligands
        //         .iter()
        //         .filter(|l| l.common.selected_for_md)
        //         .count()
        //         != 0
        //     && state
        //         .volatile
        //         .md_local
        //         .viewer
        //         .peptide_selected
        //         .contains(&(0, i_atom))
        // {
        //     color_atom = blend_color(color_atom, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
        // }

        let mut entity = Entity::new(
            mesh,
            atom_posit.into(),
            Quaternion::new_identity(),
            radius,
            color_atom,
            ATOM_SHININESS,
        );

        // Note: We draw these on the bond entities if not in a view that shows atoms.
        drawing::text_overlay(
            &mut entity,
            mol.common.name.as_deref().unwrap_or(&mol.common.ident),
            i_atom,
            atom,
            mol_active,
            &mol.chains,
            mol.common.atoms.len(),
            ui,
        );

        entity.class = EntityClass::Protein as u32;
        entities.push(entity);
    }

    // For determining inside of rings.
    let mut hydrogen_is = Vec::with_capacity(mol.common.atoms.len());
    for atom in &mol.common.atoms {
        hydrogen_is.push(atom.element == Element::Hydrogen);
    }

    // Aromatic-only adjacency list for ring centroid BFS.
    let aromatic_adj = {
        let n = mol.common.atoms.len();
        let mut adj = vec![Vec::new(); n];
        for b in &mol.common.bonds {
            if b.bond_type == BondType::Aromatic {
                adj[b.atom_0].push(b.atom_1);
                adj[b.atom_1].push(b.atom_0);
            }
        }
        adj
    };

    let mut atoms_labeled = vec![false; mol.common.atoms.len()];

    // Draw bonds.
    for (i_bond, bond) in mol.common.bonds.iter().enumerate() {
        let atom_0 = &mol.common.atoms[bond.atom_0];
        let atom_1 = &mol.common.atoms[bond.atom_1];

        // Bonds to ligand atoms follow the small-molecule view, as those atoms do.
        let lig_bond = is_lig_atom(atom_0) || is_lig_atom(atom_1);
        let view = if lig_bond { het_view } else { mol_view };

        if lig_bond {
            if view == MoleculeView::SpaceFill {
                continue;
            }
        } else {
            if mol_view == MoleculeView::Backbone && !bond.is_backbone {
                continue;
            }

            // Don't draw bonds if on the ribbon or spacefill views, and the atoms aren't hetero.
            if matches!(mol_view, MoleculeView::Ribbon | MoleculeView::SpaceFill)
                && !atom_0.hetero
                && !atom_1.hetero
            {
                continue;
            }
        }

        let atom_0_posit = mol.common.atom_posits[bond.atom_0];
        let atom_1_posit = mol.common.atom_posits[bond.atom_1];

        if filtered_out_by_dist[bond.atom_0] {
            continue;
        }

        let mut chain_not_sel = false;
        for chain in &chains_invis {
            if chain.atoms.contains(&bond.atom_0) {
                chain_not_sel = true;
                break;
            }
        }
        if chain_not_sel {
            continue;
        }

        if state.ui.visibility.hide_hydrogen
            && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
        {
            continue;
        }

        // Assuming water won't be bonded to the main molecule.
        if !lig_bond
            && (state.ui.visibility.hide_sidechains || mol_view == MoleculeView::Backbone)
            && let Some(role_0) = atom_0.role
            && let Some(role_1) = atom_1.role
            && (role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain)
        {
            continue;
        }

        if (state.ui.visibility.hide_hetero && atom_0.hetero && atom_1.hetero)
            || (state.ui.visibility.hide_protein && !atom_0.hetero && !atom_1.hetero)
        {
            continue;
        }

        let posit_0: Vec3 = atom_0_posit.into();
        let posit_1: Vec3 = atom_1_posit.into();

        // For determining how to orient multiple-bonds.
        let neighbor_posit = if bond.bond_type == BondType::Aromatic {
            let centroid: Vec3 = aromatic_ring_centroid(
                &aromatic_adj,
                &mol.common.atom_posits,
                bond.atom_0,
                bond.atom_1,
                &hydrogen_is,
            )
            .map(|c| c.into())
            .unwrap_or_else(|| mol.common.atom_posits[0].into());
            (centroid, false)
        } else {
            let neighbor_i = find_neighbor_posit(
                &mol.common.adjacency_list,
                bond.atom_0,
                bond.atom_1,
                &hydrogen_is,
            );
            match neighbor_i {
                Some((i, p1)) => (mol.common.atom_posits[i].into(), p1),
                None => (mol.common.atom_posits[0].into(), false),
            }
        };

        let dim_peptide_0 =
            state.ui.visibility.dim_peptide && !mol.common.atoms[bond.atom_0].hetero;
        let dim_peptide_1 =
            state.ui.visibility.dim_peptide && !mol.common.atoms[bond.atom_1].hetero;

        let mut color_0 = (0., 0., 0.);
        let mut color_1 = (0., 0., 0.);

        let mut manip_active = false;

        match state.volatile.mol_manip.mode {
            ManipMode::Move((mol_type, i)) => {
                if mol_type == MolType::Peptide && i == mol_i {
                    color_0 = COLOR_MOL_MOVING;
                    color_1 = COLOR_MOL_MOVING;
                    manip_active = true;
                }
            }
            ManipMode::Rotate((mol_type, i)) => {
                if mol_type == MolType::Peptide && i == mol_i {
                    color_0 = COLOR_MOL_ROTATE;
                    color_1 = COLOR_MOL_ROTATE;
                    manip_active = true;
                }
            }
            ManipMode::None => (),
        }

        if !manip_active {
            color_0 = atoms_bonds::atom_color(
                atom_0,
                0,
                bond.atom_0,
                &mol.residues,
                mol.sifts_mapping.as_deref(),
                aa_count,
                mol.chains.len(),
                &mol.common.atoms,
                &mol.chains,
                sel,
                state.ui.view_sel_level,
                dim_peptide_0,
                state.ui.res_coloring,
                state.ui.atom_color_by_charge,
                MolType::Peptide,
                &None,
            );
            color_1 = atoms_bonds::atom_color(
                atom_1,
                0,
                bond.atom_1,
                &mol.residues,
                mol.sifts_mapping.as_deref(),
                aa_count,
                mol.chains.len(),
                &mol.common.atoms,
                &mol.chains,
                sel,
                state.ui.view_sel_level,
                dim_peptide_1,
                state.ui.res_coloring,
                state.ui.atom_color_by_charge,
                MolType::Peptide,
                &None,
            );
        }

        if mol_active
            && let Selection::BondPeptide(bond_i) = ui.selection
            && bond_i == i_bond
        {
            color_0 = COLOR_SELECTED;
            color_1 = COLOR_SELECTED;
        }

        if atom_0.hetero && color_0 != COLOR_SELECTED {
            color_0 = drawing::blend_color(color_0, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        if atom_1.hetero && color_1 != COLOR_SELECTED {
            color_1 = drawing::blend_color(color_1, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        if state.volatile.md_local.mol_dynamics.is_some()
            && state.ui.md.peptide_only_near_ligs
            && mol.common.selected_for_md.is_some()
            && state
                .ligands
                .iter()
                .filter(|l| l.common.selected_for_md.is_some())
                .count()
                != 0
        // todo: Come back to this.
        {
            // if state
            //     .volatile
            //     .md_local
            //     .viewer
            //     .peptide_selected
            //     .contains(&(0, bond.atom_0))
            // {
            //     color_0 = blend_color(color_0, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            // }
            // if state
            //     .volatile
            //     .md_local
            //     .viewer
            //     .peptide_selected
            //     .contains(&(0, bond.atom_1))
            // {
            //     color_1 = blend_color(color_1, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            // }
        }

        let to_hydrogen =
            atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen;

        let mut ents_new = atoms_bonds::bond_entities(
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            MolType::Peptide,
            &mol.common.ident,
            false,
            view != MoleculeView::BallAndStick,
            neighbor_posit,
            false,
            to_hydrogen,
        );

        if !matches!(view, MoleculeView::BallAndStick | MoleculeView::SpaceFill) {
            drawing::text_overlay_bond(
                &mut ents_new,
                bond,
                &mol.common.atoms,
                &mut atoms_labeled,
                mol.common.name.as_deref().unwrap_or(&mol.common.ident),
                mol_active,
                &mol.chains,
                mol.common.bonds.len(),
                ui,
            );
        }

        entities.extend(ents_new);
    }

    // Draw H bonds.
    // todo: DRY with Ligand
    // todo: This incorrectly hides hetero-only H bonds.
    if !state.ui.visibility.hide_h_bonds
        && !state.ui.visibility.hide_protein
        && !matches!(mol_view, MoleculeView::SpaceFill | MoleculeView::Ribbon)
    {
        for bond in &mol.bonds_hydrogen {
            let atom_donor = &mol.common.atoms[bond.donor];
            let atom_acceptor = &mol.common.atoms[bond.acceptor];

            // todo: DRY with above.
            if (state.ui.visibility.hide_sidechains || mol_view == MoleculeView::Backbone)
                && let Some(role_0) = atom_donor.role
                && let Some(role_1) = atom_acceptor.role
                && (role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain)
            {
                continue;
            }

            // todo: Should we pre-filter these atoms-to-disp by index? Would be faster, but
            // todo I don't wish to expend the effort on that here.

            if filtered_out_by_dist.get(bond.donor) == Some(&true) {
                continue;
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&bond.donor) || chain.atoms.contains(&bond.acceptor) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            if state.ui.visibility.hide_water {
                if let Some(role) = atom_donor.role
                    && role == AtomRole::Water
                {
                    continue;
                }
                if let Some(role) = atom_acceptor.role
                    && role == AtomRole::Water
                {
                    continue;
                }
            }
            entities.extend(draw_hydrogen_bond(
                mol.common.atom_posits[bond.donor].into(),
                mol.common.atom_posits[bond.acceptor].into(),
                MolType::Peptide,
                bond.strength,
                false,
            ));
        }
    }

    scene.entities.extend(entities);

    let end_i = scene.entities.len();
    if let Some(mol) = state.peptides.get_mut(mol_i) {
        mol.common.entity_i_range = Some((start_i, end_i));
    }
}
