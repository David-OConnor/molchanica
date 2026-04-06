use egui::Ui;
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use na_seq::Element;

use crate::{
    cam,
    molecules::{
        MolGenericRef, MolType, common::MoleculeCommon, lipid::MoleculeLipid,
        nucleic_acid::MoleculeNucleicAcid, peptide::MoleculePeptide, pocket::Pocket,
        small::MoleculeSmall,
    },
    render::{CAM_INIT_OFFSET, set_flashlight, set_static_light},
    selection::Selection,
    state::{State, StateUi},
};

// This control the clip planes in the camera frustum.
pub const RENDER_DIST_NEAR: f32 = 0.2;
pub const RENDER_DIST_FAR: f32 = 1_000.;

// These are Å multiplied by 10. Affects the user-setting near property.
// Near sets the camera frustum's near property.
pub const VIEW_DEPTH_NEAR_MIN: u16 = 2;
pub const VIEW_DEPTH_NEAR_MAX: u16 = 300;

// Distance between start and end of the fade. A smaller distance is a more aggressive fade.
pub const FOG_HALF_DEPTH_DEFAULT: u16 = 45;

// The range to start fading distance objects, and when the fade is complete.
pub const FOG_DIST_DEFAULT: u16 = 120;

// Affects the user-setting far property.
// Sets the fog center point in its fade.
pub const FOG_DIST_MIN: u16 = 1;
pub const FOG_DIST_MAX: u16 = 120;

const PEP_FOG_FAR_RATIO: usize = 20;

/// A wrapper which sets fog using either a manual, or automatic technique based on
/// configuration.
pub fn set_fog(state: &State, cam: &mut Camera) {
    if state.to_save.auto_fog {
        set_fog_dists_by_near_and_far_mols(state, cam);
    } else {
        set_fog_dist(cam, state.ui.view_depth.1, FOG_HALF_DEPTH_DEFAULT);
    }
}

/// From a fog-center distance and half-depth, compute where to place the fog start and end
/// distances from the camera.
pub fn calc_fog_dists(dist: u16, half_depth: u16) -> (f32, f32) {
    // Clamp.
    let min = dist.saturating_sub(half_depth);

    (min as f32, (dist + half_depth) as f32)
}

/// Set fog distances in the scene's Camera struct. Used when manually setting the dist
/// and half-depth.
pub fn set_fog_dist(cam: &mut Camera, dist: u16, half_depth: u16) {
    let (fog_start, fog_end) = if dist == FOG_DIST_MAX {
        (0., 0.) // No fog will render.
    } else {
        let val = dist;
        calc_fog_dists(val, half_depth)
    };

    cam.fog_start = fog_start;
    cam.fog_end = fog_end;
}

/// Returns `(nearest, farthest)` distances (from the camera) for in-view atoms of a molecule.
/// Returns `None` if no atoms are in the camera FOV.
fn find_mol_dist_range_inner(mol: MolGenericRef<'_>, cam: &Camera) -> Option<(f32, f32)> {
    let mut nearest = f32::INFINITY;
    let mut farthest = f32::NEG_INFINITY;

    if !mol.common().visible {
        return None;
    }

    for posit_f64 in &mol.common().atom_posits {
        let posit: Vec3 = (*posit_f64).into();
        if cam.in_view(posit).0 {
            let d = (cam.position - posit).magnitude();
            if d < nearest {
                nearest = d;
            }
            if d > farthest {
                farthest = d;
            }
        }
    }

    if nearest != f32::INFINITY {
        Some((nearest, farthest))
    } else {
        None
    }
}

/// Sets fog to be a linear ramp between the closest atom visible, and the farthest.
/// `fog_start` is placed at the nearest visible atom; `fog_end` at the farthest.
/// If nothing is in view, the fog values are left unchanged.
pub fn set_fog_dists_by_near_and_far_mols(state: &State, cam: &mut Camera) {
    let mut nearest = f32::INFINITY;
    let mut farthest = f32::NEG_INFINITY;

    let mut update = |range: Option<(f32, f32)>| {
        if let Some((n, f)) = range {
            if n < nearest {
                nearest = n;
            }
            if f > farthest {
                farthest = f;
            }
        }
    };

    let viewer = &state.volatile.md_local.viewer;

    if state.volatile.md_local.draw_md_mols {
        // for mol in &viewer.mols {

        // todo: Come back to this!
        // update(find_mol_dist_range_inner(MolGenericRef::Small(mol.mol), cam));
        // }

        //
        // for mol in &viewer.peptides {
        //     update(find_mol_dist_range_inner(MolGenericRef::Peptide(mol), cam));
        // }
        //
        // for mol in &viewer.small {
        //     update(find_mol_dist_range_inner(MolGenericRef::Small(mol), cam));
        // }
        //
        // for mol in &viewer.nucleic_acids {
        //     update(find_mol_dist_range_inner(
        //         MolGenericRef::NucleicAcid(mol),
        //         cam,
        //     ));
        // }
        // for mol in &viewer.lipids {
        //     update(find_mol_dist_range_inner(MolGenericRef::Lipid(mol), cam));
        // }
    } else {
        // For the peptide use the same sparse sampling used in `find_nearest_mol_dist_to_cam`
        // (every 20th carbon) so large proteins don't stall the update. This produces good-enough results,
        // and is faster. We handle peptides as a special case, as they're likely to be much larger than
        // small molecules. todo: Consider this for lipids and NAs etc A/R.
        if let Some(pep) = &state.peptide
            && pep.common.visible
        {
            let mut pep_nearest = f32::INFINITY;
            let mut pep_farthest = f32::NEG_INFINITY;

            for (i, _atom) in pep
                .common
                .atoms
                .iter()
                .filter(|a| a.element == Element::Carbon)
                .enumerate()
            {
                if !i.is_multiple_of(PEP_FOG_FAR_RATIO) {
                    continue;
                }

                let posit: Vec3 = pep.common.atom_posits[i].into();

                if cam.in_view(posit).0 {
                    let d = (cam.position - posit).magnitude();

                    if d < pep_nearest {
                        pep_nearest = d;
                    }
                    if d > pep_farthest {
                        pep_farthest = d;
                    }
                }
            }

            if pep_nearest != f32::INFINITY {
                update(Some((pep_nearest, pep_farthest)));
            }
        }

        for mol in &state.ligands {
            update(find_mol_dist_range_inner(MolGenericRef::Small(mol), cam));
        }
        for mol in &state.nucleic_acids {
            update(find_mol_dist_range_inner(
                MolGenericRef::NucleicAcid(mol),
                cam,
            ));
        }
        for mol in &state.lipids {
            update(find_mol_dist_range_inner(MolGenericRef::Lipid(mol), cam));
        }
    }

    if nearest != f32::INFINITY {
        cam.fog_start = nearest;
        cam.fog_end = farthest;
    }
}

pub fn cam_reset_controls(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    update: &mut EngineUpdates,
    changed: &mut bool,
) {
    ui.label("Cam:");

    // Preset buttons
    if ui
        .button("Front")
        .on_hover_text("Reset the camera to look at the \"front\" of the molecule. (Y axis)")
        .clicked()
    {
        reset_camera(state, scene, update, FWD_VEC);
        *changed = true;
    }

    if ui
        .button("Top")
        .on_hover_text("Reset the camera to look at the \"top\" of the molecule. (Z axis)")
        .clicked()
    {
        reset_camera(state, scene, update, -UP_VEC);
        *changed = true;
    }

    if ui
        .button("Left")
        .on_hover_text("Reset the camera to look at the \"left\" of the molecule. (X axis)")
        .clicked()
    {
        reset_camera(state, scene, update, RIGHT_VEC);
        *changed = true;
    }
}

pub fn move_cam_to_mol(
    mol: &MoleculeCommon,
    mol_type: MolType,
    mol_i: usize,
    cam_snapshot: &mut Option<usize>,
    scene: &mut Scene,
    orbit_center: &mut Option<(MolType, usize)>,
    look_to_beyond: lin_alg::f64::Vec3,
    engine_updates: &mut EngineUpdates,
) {
    let mol_pos: Vec3 = mol.centroid().into();
    let ctr: Vec3 = look_to_beyond.into();

    // A crude heuristic, but seems to work well. Could also incorporate the camera FOV;
    // we have that available here.
    let dist = (mol.atoms.len() as f32).cbrt() * 7.5;

    cam_look_at_outside(&mut scene.camera, mol_pos, ctr, dist);

    engine_updates.camera = true;

    set_flashlight(scene);
    engine_updates.lighting = true;

    *orbit_center = Some((mol_type, mol_i));
    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = mol.centroid().into();
    }

    *cam_snapshot = None;
}

// There are borrow-error reasons we have this separate wrapper, to prevent a double-borrow on state.
pub fn move_cam_to_active_mol(
    state: &mut State,
    scene: &mut Scene,
    look_to_beyond: lin_alg::f64::Vec3,
    engine_updates: &mut EngineUpdates,
) {
    // This avoids a double borrow.
    let mut cam_ss = state.ui.cam_snapshot;

    let mut oc_copy = state.volatile.orbit_center;
    let Some(mol) = &mut state.active_mol() else {
        return;
    };

    move_cam_to_mol(
        mol.common(),
        mol.mol_type(),
        state.volatile.active_mol.unwrap().1,
        &mut cam_ss,
        scene,
        &mut oc_copy,
        look_to_beyond,
        engine_updates,
    );
    set_fog(state, &mut scene.camera);

    state.ui.cam_snapshot = cam_ss;

    state.volatile.orbit_center = oc_copy;
}

const MOVE_TO_TARGET_DIST: f32 = 15.;
pub const MOVE_TO_CAM_DIST: f32 = 20.;

/// Move the camera to look at a point of interest. Takes the starting location into account.
/// todo: Smooth interpolated zoom.
pub fn cam_look_at(cam: &mut Camera, target: lin_alg::f64::Vec3) {
    let tgt: Vec3 = target.into();
    let diff = tgt - cam.position;
    let dir = diff.to_normalized();
    let dist = diff.magnitude();

    // Rotate the camera to look at the target.
    let cam_looking_at = cam.orientation.rotate_vec(FWD_VEC);
    let rotator = Quaternion::from_unit_vecs(cam_looking_at, dir);

    cam.orientation = rotator * cam.orientation;

    // Slide along the path between cam and target until close to it.
    let move_dist = dist - MOVE_TO_TARGET_DIST;
    cam.position += dir * move_dist;
}

pub fn cam_look_at_outside(cam: &mut Camera, target: Vec3, alignment: Vec3, dist: f32) {
    // Note: This is similar to `cam_look_at`, but we don't call that, as we're positioning
    // with an absolute orientation in mind, vice `cam_look_at`'s use of current cam LOS.

    // Look from the outside in, so our view is unobstructed by the protein. Do this after
    // the camera is positioned.
    let look_vec = (target - alignment).to_normalized();

    cam.position = target + look_vec * dist;
    cam.orientation = Quaternion::from_unit_vecs(FWD_VEC, -look_vec);
}

/// Resets the camera so that it's generally looking at most of the molecules. Its behavior depends
/// on the size and positions of open molecules. Used by various view preset buttons, and at init.
pub fn reset_camera(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    look_vec: Vec3, // unit vector the cam is pointing to.
) {
    let mut size = 8.; // E.g. for small organic molecules.

    // This is a rough way to do it.
    if state.volatile.md_local.draw_md_mols {
        size = 60.;
    }

    let mut center = if let Some(mol) = &state.peptide {
        // We cache center and size, due to the potential large number of molecules.
        let center = mol.center.into();
        size = mol.size;

        center
    } else if let Some(mol) = state.active_mol() {
        mol.common().centroid().into()
        // Leaving size at its default for now.
    } else {
        let mut n = 0;
        let mut centroid = Vec3::new_zero();

        for mol in &state.ligands[0..10.min(state.ligands.len())] {
            let c: Vec3 = mol.common.centroid().into();
            centroid += c;
            n += 1;
        }

        for mol in &state.lipids[0..10.min(state.lipids.len())] {
            let c: Vec3 = mol.common.centroid().into();
            centroid += c;
            n += 1;
        }

        for mol in &state.nucleic_acids[0..10.min(state.lipids.len())] {
            let c: Vec3 = mol.common.centroid().into();
            centroid += c;
            n += 1;
        }

        if n != 0 {
            centroid /= n as f32;
        }

        updates.camera = true;

        centroid
    };

    if state.volatile.md_local.draw_md_mols {
        // In lieu of having easy access to a sim box, we compute a sampled average.
        let mut c = Vec3::new_zero();

        if !state.volatile.md_local.viewer.snapshots.is_empty() {
            const SKIP: usize = 10;

            let mut count = 0;
            // note: Does not include water fields in snapshots.
            for at in state.volatile.md_local.viewer.snapshots[0]
                .atom_posits
                .iter()
                .skip(SKIP)
            {
                c = c + *at;
                count += 1;
            }

            if count != 0 {
                c /= count as f32;
            }
        }
        center = c;
    }

    let dist_fm_center = size + CAM_INIT_OFFSET;

    scene.camera.position = center - look_vec * dist_fm_center;
    scene.camera.orientation = Quaternion::from_unit_vecs(FWD_VEC, look_vec);

    set_static_light(scene, center, size);
    set_flashlight(scene);

    updates.camera = true;
    updates.lighting = true;

    set_fog(state, &mut scene.camera);

    // todo: A/R.
    state.ui.view_depth = (VIEW_DEPTH_NEAR_MIN, FOG_DIST_DEFAULT);
}

/// Move the camera to the selected atom or residue. If there is none, but there
/// is an active molecule, move the camera to that.
pub fn move_cam_to_sel(
    state_ui: &mut StateUi,
    mol_: &Option<MoleculePeptide>,
    ligs: &[MoleculeSmall],
    nucleic_acids: &[MoleculeNucleicAcid],
    lipids: &[MoleculeLipid],
    pockets: &[Pocket],
    cam: &mut Camera,
    updates: &mut EngineUpdates,
) {
    let mut selection_found = true;

    match &state_ui.selection {
        Selection::AtomPeptide(_i_atom) => {
            let Some(mol) = mol_ else {
                return;
            };
            let atom_sel = mol.get_sel_atom(&state_ui.selection);

            if let Some(atom) = atom_sel {
                cam_look_at(cam, atom.posit);
            }
        }
        Selection::AtomLig((i_mol, i_atom)) => {
            if *i_mol >= ligs.len() || *i_atom >= ligs[*i_mol].common.atom_posits.len() {
                eprintln!("Error: Sel atom index out of bounds when moving cam to sel");
                return;
            }
            cam_look_at(cam, ligs[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomNucleicAcid((i_mol, i_atom)) => {
            if *i_mol >= nucleic_acids.len()
                || *i_atom >= nucleic_acids[*i_mol].common.atom_posits.len()
            {
                eprintln!("Error: Sel atom index out of bounds when moving cam to sel");
                return;
            }
            cam_look_at(cam, nucleic_acids[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomLipid((i_mol, i_atom)) => {
            if *i_mol >= lipids.len() || *i_atom >= lipids[*i_mol].common.atom_posits.len() {
                eprintln!("Error: Sel atom index out of bounds when moving cam to sel");
                return;
            }
            cam_look_at(cam, lipids[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomPocket((i_mol, i_atom)) => {
            if *i_mol >= pockets.len() || *i_atom >= pockets[*i_mol].common.atom_posits.len() {
                eprintln!("Error: Sel atom index out of bounds when moving cam to sel");
                return;
            }
            cam_look_at(cam, pockets[*i_mol].common.atom_posits[*i_atom]);
        }
        _ => {
            selection_found = false;
        }
    }

    if !selection_found {
        // // todo: Get working; need to get active mol.
        // if let Some(mol) = state.active_mol() {
        //     cam_look_at(cam, mol.common().centroid());
        // }
    }

    updates.camera = true;
    state_ui.cam_snapshot = None;
}

pub fn move_mol_to_cam(mol: &mut MoleculeCommon, cam: &Camera) {
    let new_posit = cam.position + cam.orientation.rotate_vec(FWD_VEC) * MOVE_TO_CAM_DIST;
    mol.move_to(new_posit.into());
}
