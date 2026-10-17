use std::f32::consts::{PI, TAU};

use egui::Ui;
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use mol_defs::molecules::{AtomRole, MolGenericRef, MolType, common::MoleculeCommon};
use na_seq::Element;

use crate::{
    prefs::DepthMode,
    render::{CAM_INIT_OFFSET, set_flashlight, set_static_light},
    selection::Selection,
    state::State,
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
pub const VIEW_DEPTH_DEFAULT: u16 = 120;

// Affects the user-setting far property.
// Sets the fog center point in its fade.
pub const FOG_DIST_MIN: u16 = 1;
pub const FOG_DIST_MAX: u16 = 120;

const PEP_FOG_FAR_RATIO: usize = 20;

/// The default viewing direction: looking down -Z, with +X right and +Y up. Our coordinates are
/// right-handed, as in PDB, mmCIF, SDF etc, so +Z points toward the viewer. This matches the
/// default view in PyMOL, Chimera etc.
pub const VIEW_DIR_FRONT: Vec3 = Vec3 {
    x: 0.,
    y: 0.,
    z: -1.,
};

/// The camera orientation that looks along `VIEW_DIR_FRONT`.
pub fn front_orientation() -> Quaternion {
    // Camera-local +Z is forward; turn it around the up axis.
    Quaternion::from_axis_angle(UP_VEC, TAU / 2.)
}

/// Apply the selected depth mode to the near clip plane and distant-object fade.
pub fn set_fog(state: &State, cam: &mut Camera) {
    let near = match state.to_save.depth_mode {
        DepthMode::Manual((near, _)) if near != VIEW_DEPTH_NEAR_MIN => near as f32 / 10.,
        _ => RENDER_DIST_NEAR,
    };
    if cam.near != near {
        cam.near = near;
        cam.update_proj_mat();
    }

    match state.to_save.depth_mode {
        DepthMode::Disabled => set_fog_dist(cam, FOG_DIST_MAX, FOG_HALF_DEPTH_DEFAULT),
        DepthMode::Auto if !state.volatile.md_local.draw_md_mols => {
            set_fog_dists_by_near_and_far_mols(state, cam);
        }
        DepthMode::Auto => set_fog_dist(cam, VIEW_DEPTH_DEFAULT, FOG_HALF_DEPTH_DEFAULT),
        DepthMode::Manual((_, far)) => set_fog_dist(cam, far, FOG_HALF_DEPTH_DEFAULT),
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
        calc_fog_dists(dist, half_depth)
    };

    cam.fog_start = fog_start;
    cam.fog_end = fog_end;
}

/// Returns the distance range of positions in the camera FOV, or `None` if none are visible.
fn visible_dist_range(positions: impl Iterator<Item = Vec3>, cam: &Camera) -> Option<(f32, f32)> {
    let mut nearest = f32::INFINITY;
    let mut farthest = f32::NEG_INFINITY;

    for posit in positions {
        if cam.in_view(posit).0 {
            let d = (cam.position - posit).magnitude();
            nearest = nearest.min(d);
            farthest = farthest.max(d);
        }
    }

    (nearest != f32::INFINITY).then_some((nearest, farthest))
}

/// Sets fog to be a linear ramp between the closest atom visible, and the farthest.
/// `fog_start` is placed at the nearest visible atom; `fog_end` at the farthest.
/// If nothing is in view, the fog values are left unchanged.
pub fn set_fog_dists_by_near_and_far_mols(state: &State, cam: &mut Camera) {
    // MD uses manual fog. This function is also called directly by the input handler.
    if state.volatile.md_local.draw_md_mols {
        return;
    }

    // Sample every 20th carbon per peptide. Pair atoms with their positions before filtering
    // so the carbon ordinal is never mistaken for an index into all atom positions.
    let peptide_positions = state
        .peptides
        .iter()
        .filter(|p| p.common.visible)
        .flat_map(|p| {
            p.common
                .atoms
                .iter()
                .zip(&p.common.atom_posits)
                .filter(|(atom, _)| atom.element == Element::Carbon)
                .step_by(PEP_FOG_FAR_RATIO)
                .map(|(_, posit)| (*posit).into())
        });

    let other_positions = state
        .ligands
        .iter()
        .map(|m| &m.common)
        .chain(state.nucleic_acids.iter().map(|m| &m.common))
        .chain(state.lipids.iter().map(|m| &m.common))
        .filter(|m| m.visible)
        .flat_map(|m| m.atom_posits.iter().map(|p| (*p).into()));

    if let Some((nearest, farthest)) =
        visible_dist_range(peptide_positions.chain(other_positions), cam)
    {
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

    for (label, axis, direction) in [
        ("Front", "Y", VIEW_DIR_FRONT),
        ("Top", "Z", -UP_VEC),
        ("Left", "X", RIGHT_VEC),
    ] {
        if ui
            .button(label)
            .on_hover_text(format!(
                "Reset the camera to look at the \"{}\" of the molecule. ({axis} axis)",
                label.to_lowercase(),
            ))
            .clicked()
        {
            reset_camera(state, scene, update, direction);
            *changed = true;
        }
    }
}

/// Owned camera framing data lets callers finish borrowing the molecule before updating state.
pub struct MolCameraTarget {
    molecule: (MolType, usize),
    center: Vec3,
    distance: f32,
}

impl MolCameraTarget {
    pub fn new(mol: &MoleculeCommon, molecule: (MolType, usize)) -> Self {
        Self {
            molecule,
            center: mol.centroid().into(),
            // A rough framing heuristic; a future version could incorporate the FOV.
            distance: (mol.atoms.len() as f32).cbrt() * 7.5,
        }
    }
}

pub fn move_cam_to_mol(
    target: MolCameraTarget,
    cam_snapshot: &mut Option<usize>,
    scene: &mut Scene,
    orbit_center: &mut Option<(MolType, usize)>,
    alignment: lin_alg::f64::Vec3,
    engine_updates: &mut EngineUpdates,
) {
    cam_look_at_outside(
        &mut scene.camera,
        target.center,
        alignment.into(),
        target.distance,
    );

    engine_updates.camera = true;

    set_flashlight(scene);
    engine_updates.lighting = true;

    *orbit_center = Some(target.molecule);
    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = target.center;
    }

    *cam_snapshot = None;
}

pub fn move_cam_to_active_mol(
    state: &mut State,
    scene: &mut Scene,
    alignment: lin_alg::f64::Vec3,
    engine_updates: &mut EngineUpdates,
) {
    let Some(molecule) = state.volatile.active_mol else {
        return;
    };
    let Some(mol) = state.active_mol() else {
        return;
    };
    let target = MolCameraTarget::new(mol.common(), molecule);

    move_cam_to_mol(
        target,
        &mut state.ui.cam_snapshot,
        scene,
        &mut state.volatile.orbit_center,
        alignment,
        engine_updates,
    );
    set_fog(state, &mut scene.camera);
}

const MOVE_TO_TARGET_DIST: f32 = 15.;
pub const MOVE_TO_CAM_DIST: f32 = 20.;

// Clearance kept between a newly opened molecule and existing ones, in Å.
const OPEN_CLEARANCE: f32 = 3.;
// Approximate drawn atom radius, used when testing whether atoms block the view.
const OCCLUDER_RADIUS: f32 = 1.5;
// Keeps small molecules from filling the view when framing them.
const FRAME_DIST_MIN: f32 = 12.;

// The search for free space to open a molecule into: candidate spacing, the number of shells
// around the starting point, and candidates per shell.
const PLACEMENT_STEP: f32 = 4.;
const PLACEMENT_SHELLS: usize = 12;
const PLACEMENT_DIRS: usize = 32;

// Candidate directions when looking for an unobscured view of a molecule.
const VIEW_DIRS: usize = 64;

/// Move the camera to look at a point of interest. Takes the starting location into account.
/// todo: Smooth interpolated zoom.
pub fn cam_look_at(cam: &mut Camera, target: lin_alg::f64::Vec3) {
    let tgt: Vec3 = target.into();
    let diff = tgt - cam.position;

    // Apply a relative rotation to preserve the camera's existing roll. If already at the
    // target, back away along the current viewing direction instead of normalizing zero.
    let cam_looking_at = cam.orientation.rotate_vec(FWD_VEC);
    let dir = direction_or(diff, cam_looking_at);
    let rotator = Quaternion::from_unit_vecs(cam_looking_at, dir);

    cam.orientation = rotator * cam.orientation;

    cam.position = tgt - dir * MOVE_TO_TARGET_DIST;
}

fn direction_or(vector: Vec3, fallback: Vec3) -> Vec3 {
    if vector.magnitude() > f32::EPSILON {
        vector.to_normalized()
    } else {
        fallback
    }
}

/// Place the camera a fixed distance behind a target, with an absolute viewing direction.
/// `direction` must be a unit vector.
fn place_camera(cam: &mut Camera, target: Vec3, direction: Vec3, distance: f32) {
    cam.position = target - direction * distance;

    // Rotate from the front view along the shortest arc. This keeps +X right, or +Y up
    // when possible, e.g. for the preset views.
    cam.orientation = Quaternion::from_unit_vecs(VIEW_DIR_FRONT, direction) * front_orientation();
}

pub fn cam_look_at_outside(cam: &mut Camera, target: Vec3, alignment: Vec3, dist: f32) {
    // Look from the outside toward the alignment point, through the target.
    let direction = direction_or(alignment - target, VIEW_DIR_FRONT);
    place_camera(cam, target, direction, dist);
}

fn mean_position(positions: impl Iterator<Item = Vec3>) -> Vec3 {
    let mut sum = Vec3::new_zero();
    let mut count = 0;

    for position in positions {
        sum += position;
        count += 1;
    }

    if count == 0 { sum } else { sum / count as f32 }
}

/// Determine the center and size used by view presets. Peptides are framed by their bounding radius.
fn reset_frame(state: &State) -> (Vec3, f32) {
    let md = &state.volatile.md_local;
    let default_size = if md.draw_md_mols { 60. } else { 8. };
    let mol = state
        .active_mol()
        .or_else(|| state.peptide_for_tools().map(MolGenericRef::Peptide));

    let (mut center, size) = if let Some(mol) = mol {
        match mol {
            MolGenericRef::Peptide(p) => (
                p.common.centroid().into(),
                p.common.bounding_radius() as f32,
            ),
            other => (other.common().centroid().into(), default_size),
        }
    } else {
        let mols = state
            .ligands
            .iter()
            .take(10)
            .map(|m| &m.common)
            .chain(state.lipids.iter().take(10).map(|m| &m.common))
            .chain(state.nucleic_acids.iter().take(10).map(|m| &m.common));

        (
            mean_position(mols.map(|m| m.centroid().into())),
            default_size,
        )
    };

    if md.draw_md_mols {
        // Sample the first frame; snapshot atom positions exclude water fields.
        center = md
            .viewer
            .snapshots
            .first()
            .map_or(Vec3::new_zero(), |snapshot| {
                mean_position(snapshot.atom_posits.iter().step_by(10).copied())
            });
    }

    (center, size)
}

/// Reset the view using the active molecule, a peptide, or a sample of open molecules.
/// `look_vec` is the unit viewing direction used by the preset buttons and at initialization.
pub fn reset_camera(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    look_vec: Vec3,
) {
    let (center, size) = reset_frame(state);
    place_camera(&mut scene.camera, center, look_vec, size + CAM_INIT_OFFSET);

    set_static_light(scene, center, size);
    set_flashlight(scene);

    updates.camera = true;
    updates.lighting = true;

    if let DepthMode::Manual(depth) = &mut state.to_save.depth_mode {
        *depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_DEFAULT);
    }
    set_fog(state, &mut scene.camera);
}

/// Move the camera to the selected atom or residue. If there is none, but there
/// is an active molecule, move the camera to that.
pub fn move_cam_to_sel(state: &mut State, cam: &mut Camera, updates: &mut EngineUpdates) {
    let target = if state.ui.selection == Selection::None {
        state.active_mol().map(|mol| mol.common().centroid())
    } else {
        state.selected_target()
    };
    let Some(target) = target else {
        return;
    };

    cam_look_at(cam, target);
    updates.camera = true;
    state.ui.cam_snapshot = None;
}

pub fn move_mol_to_cam(mol: &mut MoleculeCommon, cam: &Camera) {
    let new_posit = cam.position + cam.orientation.rotate_vec(FWD_VEC) * MOVE_TO_CAM_DIST;
    mol.move_to(new_posit.into());
}

/// Move a newly opened molecule in front of the camera, into free space where no other molecule
/// blocks the view of it. Use with [`frame_mol`], which then points the camera at it.
pub fn place_mol_in_view(mol: &mut MoleculeCommon, state: &State, cam: &Camera) {
    if mol.atom_posits.is_empty() {
        return;
    }

    let fwd = cam.orientation.rotate_vec(FWD_VEC);
    let start = cam.position + fwd * MOVE_TO_CAM_DIST;
    mol.move_to(start.into());

    let radius = mol.bounding_radius() as f32;
    let dist = frame_dist(cam, radius);

    // Only atoms this close to the start can collide with a candidate, or block the view of it.
    let reach = PLACEMENT_STEP * PLACEMENT_SHELLS as f32 + radius + dist + OCCLUDER_RADIUS;
    let obstacles: Vec<_> = obstacle_posits(state, None)
        .into_iter()
        .filter(|p| (*p - start).magnitude_squared() < reach * reach)
        .collect();

    if obstacles.is_empty() {
        return;
    }

    let min_sep = radius + OPEN_CLEARANCE;
    let is_free = |p: &Vec3| {
        obstacles
            .iter()
            .all(|o| (*o - *p).magnitude_squared() > min_sep * min_sep)
    };
    let is_clear = |p: &Vec3| count_occluders(&obstacles, *p, radius, fwd, dist) == 0;

    // First, step along the line of sight toward the camera; this is in front of whatever the
    // user is looking at. Then search outward from the start, preferring spots nearer the camera.
    let mut candidates: Vec<Vec3> = (0..)
        .map(|i| MOVE_TO_CAM_DIST - PLACEMENT_STEP * i as f32)
        .take_while(|t| *t >= PLACEMENT_STEP)
        .map(|t| cam.position + fwd * t)
        .collect();

    for i in 1..=PLACEMENT_SHELLS {
        let shell_radius = PLACEMENT_STEP * i as f32;

        let mut shell: Vec<_> = sphere_dirs(PLACEMENT_DIRS)
            .map(|dir| start + dir * shell_radius)
            .collect();

        shell.sort_by(|a, b| {
            let dist_a = (*a - cam.position).magnitude_squared();
            let dist_b = (*b - cam.position).magnitude_squared();
            dist_a.total_cmp(&dist_b)
        });

        candidates.extend(shell);
    }

    // If no spot has a clear view from the current camera direction, `frame_mol` turns the
    // camera to find one.
    let posit = candidates
        .iter()
        .find(|p| is_free(p) && is_clear(p))
        .or_else(|| candidates.iter().find(|p| is_free(p)));

    if let Some(p) = posit {
        mol.move_to((*p).into());
    }
}

/// Point the camera at a molecule, from far enough away to see all of it. E.g. for one that was
/// just opened. Keeps the current viewing direction, unless other molecules would block the view.
pub fn frame_mol(
    state: &mut State,
    scene: &mut Scene,
    molecule: (MolType, usize),
    updates: &mut EngineUpdates,
) {
    let Some(mol) = state.get_mol(molecule.0, molecule.1) else {
        return;
    };
    let mol = mol.common();

    if mol.atom_posits.is_empty() {
        return;
    }

    let center: Vec3 = mol.centroid().into();
    let radius = mol.bounding_radius() as f32;

    let cam = &mut scene.camera;
    let dist = frame_dist(cam, radius);
    let fwd = cam.orientation.rotate_vec(FWD_VEC);

    let obstacles = obstacle_posits(state, Some(molecule));
    let dir = clear_view_dir(&obstacles, center, radius, dist, fwd);

    // A relative rotation preserves the camera's roll.
    cam.orientation = Quaternion::from_unit_vecs(fwd, dir) * cam.orientation;
    cam.position = center - dir * dist;

    set_flashlight(scene);
    updates.camera = true;
    updates.lighting = true;

    // Manual depth was tuned for the previous view; it could clip or fog out this one.
    // As in `reset_camera`. Callers apply it with `set_fog`.
    if let DepthMode::Manual(depth) = &mut state.to_save.depth_mode {
        *depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_DEFAULT);
    }

    state.ui.cam_snapshot = None;
}

/// The camera distance from a sphere's center at which all of it fits in view.
fn frame_dist(cam: &Camera, radius: f32) -> f32 {
    let half_y = cam.fov_y / 2.;
    let half_x = if cam.aspect > 0. {
        (half_y.tan() * cam.aspect).atan()
    } else {
        half_y
    };

    (radius / half_y.min(half_x).sin()).max(FRAME_DIST_MIN)
}

/// Heavy-atom positions of visible molecules, other than `exclude`. Omits water, which is
/// often hidden.
fn obstacle_posits(state: &State, exclude: Option<(MolType, usize)>) -> Vec<Vec3> {
    fn tagged<'a>(
        mol_type: MolType,
        mols: impl Iterator<Item = &'a MoleculeCommon> + 'a,
    ) -> impl Iterator<Item = ((MolType, usize), &'a MoleculeCommon)> + 'a {
        mols.enumerate().map(move |(i, mol)| ((mol_type, i), mol))
    }

    tagged(MolType::Peptide, state.peptides.iter().map(|m| &m.common))
        .chain(tagged(
            MolType::Ligand,
            state.ligands.iter().map(|m| &m.common),
        ))
        .chain(tagged(
            MolType::NucleicAcid,
            state.nucleic_acids.iter().map(|m| &m.common),
        ))
        .chain(tagged(
            MolType::Lipid,
            state.lipids.iter().map(|m| &m.common),
        ))
        .chain(tagged(
            MolType::Pocket,
            state.pockets.iter().map(|m| &m.common),
        ))
        .filter(|(id, mol)| mol.visible && Some(*id) != exclude)
        .flat_map(|(_, mol)| mol.atoms.iter().zip(&mol.atom_posits))
        .filter(|(atom, _)| atom.element != Element::Hydrogen && atom.role != Some(AtomRole::Water))
        .map(|(_, posit)| (*posit).into())
        .collect()
}

/// The number of obstacle atoms blocking the view of a sphere, from a camera `dist` behind its
/// center along `dir`.
fn count_occluders(obstacles: &[Vec3], center: Vec3, radius: f32, dir: Vec3, dist: f32) -> usize {
    let cam_posit = center - dir * dist;

    obstacles
        .iter()
        .filter(|posit| {
            let rel = **posit - cam_posit;
            let t = rel.dot(dir);

            if t <= 0. || t >= dist {
                return false;
            }

            // The view cone narrows from the sphere's silhouette to a point at the camera.
            let cone_radius = radius * t / dist + OCCLUDER_RADIUS;
            rel.magnitude_squared() - t * t < cone_radius * cone_radius
        })
        .count()
}

/// A direction to view a sphere from, avoiding obstacles. Keeps the current direction unless
/// another is substantially less obscured; among equally obscured ones, prefers the smallest turn.
fn clear_view_dir(obstacles: &[Vec3], center: Vec3, radius: f32, dist: f32, current: Vec3) -> Vec3 {
    let occluded_current = count_occluders(obstacles, center, radius, current, dist);
    if occluded_current == 0 {
        return current;
    }

    let mut best = (occluded_current, current);
    for dir in sphere_dirs(VIEW_DIRS) {
        let occluded = count_occluders(obstacles, center, radius, dir, dist);

        if occluded < best.0 || (occluded == best.0 && dir.dot(current) > best.1.dot(current)) {
            best = (occluded, dir);
        }
    }

    // E.g. superposed proteins block each other from every direction. Don't swing the camera
    // around for a marginal improvement.
    if best.0 * 2 <= occluded_current {
        best.1
    } else {
        current
    }
}

/// Approximately evenly-spaced unit vectors, on a Fibonacci sphere.
fn sphere_dirs(n: usize) -> impl Iterator<Item = Vec3> {
    let golden_angle = PI * (3. - 5_f32.sqrt());

    (0..n).map(move |i| {
        let y = 1. - 2. * (i as f32 + 0.5) / n as f32;
        let r = (1. - y * y).sqrt();
        let theta = golden_angle * i as f32;

        Vec3::new(r * theta.cos(), y, r * theta.sin())
    })
}
