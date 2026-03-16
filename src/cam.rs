use egui::Ui;
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use na_seq::Element;

use crate::{
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

// How quickly to track the fog target: 0 = frozen, 1 = instant snap.
// At ~10 calls/sec (ratio-6 on typical mouse events) this converges in ~0.7 s,
// which is fast enough to feel responsive while suppressing frame-to-frame flicker.
const FOG_FADE_ALPHA: f32 = 0.1;
// Fog starts CLEAR_ZONE past the nearest atom so it is fully unobscured.
const FOG_CLEAR_ZONE: f32 = 5.0;
// Width of the fog gradient in the auto-fog mode (Å). Smaller = steeper/more aggressive fade.
// At 20 the gradient spans 40 Å; at 8 it spans 16 Å.
pub const FOG_AUTO_HALF_DEPTH: u16 = 15;

/// From a fog-center distance and half-depth, compute where to place the fog start and end
/// distances from the camera.
pub fn calc_fog_dists(dist: u16, half_depth: u16) -> (f32, f32) {
    // Clamp.
    let min = dist.saturating_sub(half_depth);

    (min as f32, (dist + half_depth) as f32)
}

/// Set fog distances in the scene's Camera struct.
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

/// Sets the fog distance automatically so the nearest visible atoms are clear, with an
/// aggressive fade behind them. Call this whenever the camera or molecule positions change.
///
/// Rather than snapping to the computed target, the current fog values are lerped toward
/// it each call, preventing the flicker/flash that would otherwise occur as the nearest
/// visible atom changes frame-to-frame.
///
/// Strategy:
/// - Find the closest atom in the camera FOV (`d_near`).
/// - Leave a small clear zone past `d_near` before fog begins.
/// - Half-depth (gradient width) is wider when a large protein is present, tighter otherwise.
/// - If nothing is in view, gently drift back toward the defaults.
pub fn set_fog_from_mols(state: &State, cam: &mut Camera) {
    let (target_start, target_end) = match find_nearest_mol_dist_to_cam(state, cam) {
        Some(d) => {
            let d_near = d.max(0.);

            let half_depth = FOG_AUTO_HALF_DEPTH;

            // Only enforce a floor so fog_start stays positive.
            // No upper clamp: when the camera is far away, dist grows with d_near,
            // ensuring the nearest visible atoms are never inside the fog zone.
            let dist =
                ((d_near + FOG_CLEAR_ZONE) as u16 + half_depth).max(FOG_DIST_MIN + half_depth);

            calc_fog_dists(dist, half_depth)
        }
        // Nothing in the FOV — drift toward the default (light) fog.
        None => calc_fog_dists(FOG_DIST_DEFAULT, FOG_HALF_DEPTH_DEFAULT),
    };

    // Snap upward instantly (nearest atoms must never be inside the fog zone),
    // but lerp downward slowly (prevents flicker when the nearest atom changes).
    if target_start >= cam.fog_start {
        cam.fog_start = target_start;
    } else {
        cam.fog_start += FOG_FADE_ALPHA * (target_start - cam.fog_start);
    }
    if target_end >= cam.fog_end {
        cam.fog_end = target_end;
    } else {
        cam.fog_end += FOG_FADE_ALPHA * (target_end - cam.fog_end);
    }
}

fn find_nearest_mol_inner(mol: MolGenericRef<'_>, cam: &Camera) -> Option<f32> {
    let mut nearest = f32::INFINITY;
    for posit_f64 in &mol.common().atom_posits {
        let posit: Vec3 = (*posit_f64).into();
        if cam.in_view(posit).0 {
            let d = (cam.position - posit).magnitude();
            if d < nearest {
                nearest = d;
            }
        }
    }
    if nearest != f32::INFINITY {
        Some(nearest)
    } else {
        None
    }
}

/// todo: Experimental
/// Find the distance of the closest molecule to the camera, in front of it. Run this regularly upon
/// camera movement, e.g. every x steps where camera position or orientation changes.
///
/// Returns None if there are no molecules in the camera FOV.
pub fn find_nearest_mol_dist_to_cam(state: &State, cam: &Camera) -> Option<f32> {
    let mut nearest = f32::INFINITY;

    // For the protein, rely on cached distances along a collection of radials.
    if let Some(pep) = &state.peptide {
        // todo: Very slow approach for now to demonstrate concept. Change this to use a cache!!
        for (i, _atom) in pep
            .common
            .atoms
            .iter()
            .filter(|a| a.element == Element::Carbon)
            .enumerate()
        {
            if !i.is_multiple_of(20) {
                continue;
            }

            let posit: Vec3 = pep.common.atom_posits[i].into();
            if cam.in_view(posit).0 {
                let dist = (cam.position - posit).magnitude() - 4.;
                if dist < nearest {
                    nearest = dist;
                }
            }
        }
    }

    for mol in &state.ligands {
        if let Some(v) = find_nearest_mol_inner(MolGenericRef::Small(mol), cam)
            && v < nearest
        {
            nearest = v;
        }
    }

    for mol in &state.nucleic_acids {
        if let Some(v) = find_nearest_mol_inner(MolGenericRef::NucleicAcid(mol), cam)
            && v < nearest
        {
            nearest = v;
        }
    }

    for mol in &state.lipids {
        if let Some(v) = find_nearest_mol_inner(MolGenericRef::Lipid(mol), cam)
            && v < nearest
        {
            nearest = v;
        }
    }

    if nearest != f32::INFINITY {
        return Some(nearest);
    }
    None
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
        centroid /= n as f32;
        size = 40.; // A broad view.

        centroid
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

    updates.camera = true;
    state_ui.cam_snapshot = None;
}

pub fn move_mol_to_cam(mol: &mut MoleculeCommon, cam: &Camera) {
    let new_posit = cam.position + cam.orientation.rotate_vec(FWD_VEC) * MOVE_TO_CAM_DIST;
    mol.move_to(new_posit.into());
}
