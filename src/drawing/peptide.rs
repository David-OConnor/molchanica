//! For drawing peptides/proteins. This is a separate module from that used for
//! other molecule types due to differences in how we render it. It also includes code for
//! drawing modes which only apply to peptides, e.g. ribbon and solvent-accessible-surface.
//!
//! Note: It may be possible/desirable to consolidate the draw function with the other types, e.g. in the
//! `atoms_bonds` module.

use bio_files::{BondType, ResidueType};
use egui::FontFamily;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene, TextOverlay};
use lin_alg::f32::{Quaternion, Vec3};
use mol_defs::{
    molecules::{AtomRole, Chain, MolGenericRef, MolType, peptide::MoleculePeptide},
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
        MESH_CUBE, MESH_DENSITY_SURFACE, MESH_PEP_SOLVENT_SURFACE, MESH_SECONDARY_STRUCTURE,
        MESH_SPHERE_LOWRES,
    },
    selection::Selection,
    state::{OperatingMode, ResColoring, State, StateUi},
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
                text: mol.common.ident.clone(),
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
    scene: &mut Scene,
) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
    }

    let mut ent = Entity::new(
        MESH_SECONDARY_STRUCTURE,
        transform.translation,
        transform.rotation,
        1.,
        COLOR_SECONDARY_STRUCTURE,
        ATOM_SHININESS,
    );
    ent.class = EntityClass::SecondaryStructure as u32;
    scene.entities.push(ent);
}

// todo: Move this A/R. Util? Molecule? Method on peptide?
/// Filter by distance to various items. Has some computational complexity.
/// Indexes in the result are filtered out. So, an empty Vec means no restrictions.
// pub fn filter_pep_atoms_by_dist(mol: &MoleculeCommon, ui: &StateUi, lig: Option<&MoleculeCommon>) -> Vec<usize> {
// pub fn filter_pep_atoms_by_dist<'a>(mol: &MoleculeCommon, ui: &StateUi, active_mol: Option<MolGenericRef<'a>>) -> Vec<usize> {
pub fn filter_pep_atoms_by_dist<'a>(
    pep: &MoleculePeptide,
    ui: &StateUi,
    active_mol: Option<MolGenericRef<'a>>,
    mol_active: bool,
) -> Vec<usize> {
    let mut result = Vec::new();

    let mol = &pep.common;

    // Speed up computations by using magnitude squared.
    let nearby_dist_thresh_sq = ui.nearby_dist_thresh.pow(2) as f32;

    // todo: Experimenting. I'm not sure why we need this, but the results don't filter enough otherwise.
    let nearby_dist_thresh_sfc_sq = (ui.nearby_dist_thresh / 2).pow(2) as f32;

    if !ui.show_near_lig_only && !ui.show_near_sel_only && !ui.show_near_sfc_only {
        return Vec::new();
    }

    let sfc_pts = if ui.show_near_sfc_only {
        // Higher means faster, but cruder.
        const NEAR_SFC_MESH_PRECISION: f32 = 5.;

        let atoms: Vec<(Vec3, _)> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.hetero)
            .map(|(i, a)| (mol.atom_posits[i].into(), a.element.vdw_radius()))
            .collect();

        // todo: DOn't create this each drawing! Cache the atoms near the sfc pre-computed.
        let mesh = make_sas_mesh(&atoms, SOLVENT_RAD, NEAR_SFC_MESH_PRECISION);
        mesh.vertices
            .iter()
            .map(|v| Vec3::from_slice(&v.position).unwrap())
            .collect()
    } else {
        Vec::new()
    };

    // An optimization: Measure dist^2 once per residue, instead of per atom. This, for better or worse,
    // shows complete residues only.
    if ui.show_near_sfc_only {
        for res in &pep.residues {
            if res.atoms.is_empty() {
                break;
            }

            // Arbitrary; pick two atoms on either SN end for variety. If either is close to the surface,
            // pass all atoms in the residue.
            let res_atom_0 = &res.atoms[0];
            let res_atom_1 = res.atoms.last().unwrap_or(&res.atoms[0]);

            let p_atom_0: Vec3 = mol.atom_posits[*res_atom_0].into();
            let p_atom_1: Vec3 = mol.atom_posits[*res_atom_1].into();

            // Check if near any surface point.
            let mut passed = false;
            for pt in &sfc_pts {
                for posit in [p_atom_0, p_atom_1] {
                    if (*pt - posit).magnitude_squared() < nearby_dist_thresh_sfc_sq {
                        passed = true;
                        break;
                    }
                }
            }

            if !passed {
                for i_atom in &res.atoms {
                    result.push(*i_atom);
                }
                continue;
            }
        }
    }

    for (i_atom, _atom) in mol.atoms.iter().enumerate() {
        let posit = mol.atom_posits[i_atom];

        if ui.show_near_sel_only
            && mol_active
            && let Selection::AtomPeptide(i_sel) = &ui.selection
        {
            // todo: This will fail after moves and dynamics. You must pick the selected atom
            // todo posit correctly!

            if (posit - mol.atom_posits[*i_sel]).magnitude_squared() as f32 > nearby_dist_thresh_sq
            {
                result.push(i_atom);
                continue;
            }
        }

        if ui.show_near_lig_only
            && let Some(ref lig) = active_mol
        {
            let atom_sel = lig.common().atom_posits[0]; // todo: Centroid?

            if (posit - atom_sel).magnitude_squared() as f32 > nearby_dist_thresh_sq {
                result.push(i_atom);
                continue;
            }
        }
    }

    result
}

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_peptide(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
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

    let filtered_out_by_dist =
        filter_pep_atoms_by_dist(mol, &state.ui, state.active_mol(), mol_active);

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
    // Ribbon, dots and solvent-surface views each render from a mesh in a shared GPU slot, so
    // only one peptide can use them; the rest fall back to atoms and bonds.
    let mol_view = effective_mol_view_peptide(state, mol_i);

    if owns_shared_mesh && mol_view == MoleculeView::Ribbon {
        // Flush any deferred chain-visibility change into a full rebuild.
        if state.volatile.flags.ss_mesh_dirty {
            state.volatile.flags.update_ss_mesh = true;
            state.volatile.flags.ss_mesh_dirty = false;
        }
        let transform = if state.volatile.flags.ss_mesh_created {
            state.volatile.mol_manip.ribbon_mesh_transform
        } else {
            PeptideMeshTransform::default()
        };

        draw_secondary_structure(
            &mut state.volatile.flags.update_ss_mesh,
            state.volatile.flags.ss_mesh_created,
            transform,
            scene,
        );

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

    // Draw atoms.
    if matches!(
        mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
            if atom.hetero {
                let mut water = false;
                if let Some(role) = atom.role {
                    water = role == AtomRole::Water;
                }
                if !water && mol_view == MoleculeView::SpaceFill {
                    // Don't draw VDW spheres for hetero atoms; draw as sticks.
                    continue;
                }
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

            if filtered_out_by_dist.contains(&i_atom) {
                continue;
            }

            if let Some(role) = atom.role {
                if (state.ui.visibility.hide_sidechains || mol_view == MoleculeView::Backbone)
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

            let (mut radius, mesh) = match mol_view {
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
                color_atom =
                    drawing::blend_color(color_atom, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
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
                &mol.common.ident,
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

    // Draw bonds.
    for (i_bond, bond) in mol.common.bonds.iter().enumerate() {
        if mol_view == MoleculeView::Backbone && !bond.is_backbone {
            continue;
        }

        let atom_0 = &mol.common.atoms[bond.atom_0];
        let atom_1 = &mol.common.atoms[bond.atom_1];

        if mol_view == MoleculeView::Ribbon && !atom_0.hetero && !atom_1.hetero {
            continue;
        }

        let atom_0_posit = mol.common.atom_posits[bond.atom_0];
        let atom_1_posit = mol.common.atom_posits[bond.atom_1];

        // Don't draw bonds if on the spacefill view, and the atoms aren't hetero.
        if mol_view == MoleculeView::SpaceFill && !atom_0.hetero && !atom_1.hetero {
            continue;
        }

        if filtered_out_by_dist.contains(&bond.atom_0) {
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
        if (state.ui.visibility.hide_sidechains || mol_view == MoleculeView::Backbone)
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
            mol_view != MoleculeView::BallAndStick,
            neighbor_posit,
            false,
            to_hydrogen,
        );

        if !ents_new.is_empty()
            && !matches!(
                mol_view,
                MoleculeView::BallAndStick | MoleculeView::SpaceFill
            )
        {
            drawing::text_overlay(
                &mut ents_new[0],
                &mol.common.ident,
                bond.atom_0,
                atom_0,
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

            if filtered_out_by_dist.contains(&bond.donor) {
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
