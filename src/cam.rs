use egui::Ui;
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    molecules::{
        MoleculePeptide, common::MoleculeCommon, lipid::MoleculeLipid,
        nucleic_acid::MoleculeNucleicAcid, pocket::Pocket, small::MoleculeSmall,
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
pub const FOG_HALF_DEPTH: u16 = 40;

// The range to start fading distance objects, and when the fade is complete.
pub const FOG_DIST_DEFAULT: u16 = 70;

// Affects the user-setting far property.
// Sets the fog center point in its fade.
pub const FOG_DIST_MIN: u16 = 1;
pub const FOG_DIST_MAX: u16 = 120;

pub fn calc_fog_dists(dist: u16) -> (f32, f32) {
    // Clamp.
    let min = if dist > FOG_HALF_DEPTH {
        dist - FOG_HALF_DEPTH
    } else {
        0
    };

    (min as f32, (dist + FOG_HALF_DEPTH) as f32)
}

pub fn set_fog_dist(cam: &mut Camera, dist: u16) {
    let (fog_start, fog_end) = if dist == FOG_DIST_MAX {
        (0., 0.) // No fog will render.
    } else {
        let val = dist;
        calc_fog_dists(val)
    };

    cam.fog_start = fog_start;
    cam.fog_end = fog_end;
}

pub fn cam_reset_controls(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    changed: &mut bool,
) {
    ui.label("Cam:");

    // Preset buttons
    if ui
        .button("Front")
        .on_hover_text("Reset the camera to look at the \"front\" of the molecule. (Y axis)")
        .clicked()
    {
        reset_camera(state, scene, engine_updates, FWD_VEC);
        *changed = true;
    }

    if ui
        .button("Top")
        .on_hover_text("Reset the camera to look at the \"top\" of the molecule. (Z axis)")
        .clicked()
    {
        reset_camera(state, scene, engine_updates, -UP_VEC);
        *changed = true;
    }

    if ui
        .button("Left")
        .on_hover_text("Reset the camera to look at the \"left\" of the molecule. (X axis)")
        .clicked()
    {
        reset_camera(state, scene, engine_updates, RIGHT_VEC);
        *changed = true;
    }
}

pub fn move_cam_to_mol(
    mol: &MoleculeCommon,
    cam_snapshot: &mut Option<usize>,
    scene: &mut Scene,
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

    // todo: We likely need to set this too?
    // state.volatile.orbit_center = Some((MolType::Peptide, 0));
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
    let Some(mol) = &mut state.active_mol() else {
        return;
    };

    move_cam_to_mol(
        mol.common(),
        &mut cam_ss,
        scene,
        look_to_beyond,
        engine_updates,
    );

    state.ui.cam_snapshot = cam_ss;
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

/// Resets the camera to the *front* view, and related settings. Its behavior depends
/// on the size and positions of open molecules.
pub fn reset_camera(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    look_vec: Vec3, // unit vector the cam is pointing to.
) {
    let mut size = 8.; // E.g. for small organic molecules.

    let center = if let Some(mol) = &state.peptide {
        // We cache center and size, due to the potential large number of molecules.
        let center = mol.center.into();
        size = mol.size;

        center
    } else {
        if let Some(mol) = state.active_mol() {
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
            centroid /= n as f32;
            size = 40.; // A broad view.

            centroid
        }
    };

    let dist_fm_center = size + CAM_INIT_OFFSET;

    scene.camera.position = center - look_vec * dist_fm_center;
    scene.camera.orientation = Quaternion::from_unit_vecs(FWD_VEC, look_vec);

    set_static_light(scene, center, size);
    set_flashlight(scene);

    engine_updates.camera = true;
    engine_updates.lighting = true;

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
    engine_updates: &mut EngineUpdates,
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
            cam_look_at(cam, ligs[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomNucleicAcid((i_mol, i_atom)) => {
            // if *i_mol >= nucleic_acids.len() {
            //     return;
            // }
            cam_look_at(cam, nucleic_acids[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomLipid((i_mol, i_atom)) => {
            cam_look_at(cam, lipids[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomPocket((i_mol, i_atom)) => {
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

    engine_updates.camera = true;
    state_ui.cam_snapshot = None;
}

pub fn move_mol_to_cam(mol: &mut MoleculeCommon, cam: &Camera) {
    let new_posit = cam.position + cam.orientation.rotate_vec(FWD_VEC) * MOVE_TO_CAM_DIST;
    mol.move_to(new_posit.into());
}
