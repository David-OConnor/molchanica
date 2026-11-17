//! An optional floor below the scene, with X (red) and Z (blue) axis lines through the origin. Gives
//! an absolute spatial reference.
//!
//! The floor is a disc centered under the camera, fading out before the far clip plane, so it
//! appears infinite. It's transparent, so atoms below it remain visible.

use graphics::{Camera, EngineUpdates, Entity, EntityUpdate, Mesh, Scene, Vertex};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    cam::RENDER_DIST_FAR,
    drawing::EntityClass,
    render::{MESH_FLOOR, MESH_FLOOR_AXIS_X, MESH_FLOOR_AXIS_Z},
    state::{OperatingMode, State},
};

type Rgba = (u8, u8, u8, u8);

// Alpha is the opacity at the center; it fades toward the edge.
const COLOR_FLOOR: Rgba = (115, 115, 128, 64);
const COLOR_X_AXIS: Rgba = (255, 40, 40, 230);
const COLOR_Z_AXIS: Rgba = (50, 90, 255, 230);
/// Only puts entities in the transparent render pass; vertex alpha sets the actual opacity.
const ENTITY_OPACITY: f32 = 0.5;

// Å
/// Fully faded at this distance from the point below the camera. Inside the far clip plane, so
/// there's no hard edge.
const RADIUS: f32 = RENDER_DIST_FAR * 0.95;
/// Of the radius.
const FADE_START: f32 = 0.1;
const RINGS: usize = 32;
const SEGMENTS: usize = 96;
const AXIS_SEGMENTS: usize = 64;

/// Re-center when the camera moves this far.
const RECENTER_STEP: f32 = 10.;
/// Axis line width, as a ratio of camera height above the floor. Keeps lines a few pixels wide
/// below the camera.
const AXIS_WIDTH_RATIO: f32 = 0.0025;
const AXIS_WIDTH_MIN: f32 = 0.2;
const AXIS_WIDTH_STEP: f32 = 0.05;

const MARGIN_BELOW: f32 = 5.;
/// Height snaps to multiples of this.
const HEIGHT_SNAP: f32 = 10.;
/// Height used when no molecules are open.
const HEIGHT_DEFAULT: f32 = -30.;

const PEP_SAMPLE_RATIO: usize = 10;
const MD_SAMPLE_RATIO: usize = 10;

const CLASS: u32 = EntityClass::Floor as u32;

/// Where the floor entities are, relative to the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    center_x: f32,
    center_z: f32,
    axis_width: f32,
}

impl Placement {
    fn new(y: f32, cam: &Camera) -> Self {
        let snap = |v: f32, step: f32| (v / step).round() * step;

        let p = cam.position;
        let width = ((p.y - y).abs() * AXIS_WIDTH_RATIO).max(AXIS_WIDTH_MIN);

        Self {
            center_x: snap(p.x, RECENTER_STEP),
            center_z: snap(p.z, RECENTER_STEP),
            axis_width: snap(width, AXIS_WIDTH_STEP),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor {
    /// Absolute height, in Å.
    pub y: f32,
    /// The placement of the current floor entities.
    placed: Option<Placement>,
}

impl Floor {
    /// Place below the lowest atom.
    pub fn fit(state: &State) -> Self {
        let y = sample_posits(state)
            .map(|p| p.y)
            .reduce(f32::min)
            .map_or(HEIGHT_DEFAULT, |min| {
                ((min - MARGIN_BELOW) / HEIGHT_SNAP).floor() * HEIGHT_SNAP
            });

        Self { y, placed: None }
    }

    fn entities(&self, pl: Placement) -> Vec<Entity> {
        fn ent(mesh: usize, position: Vec3, scale: Vec3) -> Entity {
            let mut ent = Entity::new(
                mesh,
                position,
                Quaternion::new_identity(),
                1.,
                (1., 1., 1.),
                0.,
            );
            ent.scale_partial = Some(scale);
            ent.opacity = ENTITY_OPACITY;
            ent.class = CLASS;
            ent
        }

        // Axis half-length, so its fade ends at the disc's edge.
        let half_len = |offset: f32| (RADIUS.powi(2) - offset.powi(2)).max(0.).sqrt();

        // Mesh order draws the axes over the disc; transparent entities don't write depth.
        vec![
            ent(
                MESH_FLOOR,
                Vec3::new(pl.center_x, self.y, pl.center_z),
                Vec3::new(RADIUS, 1., RADIUS),
            ),
            ent(
                MESH_FLOOR_AXIS_X,
                Vec3::new(pl.center_x, self.y, 0.),
                Vec3::new(half_len(pl.center_z), 1., pl.axis_width),
            ),
            ent(
                MESH_FLOOR_AXIS_Z,
                Vec3::new(0., self.y, pl.center_z),
                Vec3::new(pl.axis_width, 1., half_len(pl.center_x)),
            ),
        ]
    }
}

/// 1 at the center, fading to 0 at `t = 1`.
fn fade(t: f32) -> f32 {
    let s = ((t - FADE_START) / (1. - FADE_START)).clamp(0., 1.);

    1. - s * s * (3. - 2. * s)
}

fn vertex(x: f32, z: f32, (r, g, b, a): Rgba, t: f32) -> Vertex {
    let mut v = Vertex::new([x, 0., z], Vec3::new(0., 1., 0.));
    // Vertex alpha 0 means "use the entity color".
    let alpha = ((a as f32 * fade(t)).round() as u8).max(1);
    v.color = Some((r, g, b, alpha));
    v
}

/// A unit-radius disc in the XZ plane, facing +Y.
pub fn mesh_disc() -> Mesh {
    let mut vertices = Vec::with_capacity((RINGS + 1) * (SEGMENTS + 1));
    for ring in 0..=RINGS {
        let r = ring as f32 / RINGS as f32;

        for seg in 0..=SEGMENTS {
            let θ = seg as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            vertices.push(vertex(r * θ.cos(), r * θ.sin(), COLOR_FLOOR, r));
        }
    }

    let row = SEGMENTS + 1;
    let mut indices = Vec::with_capacity(RINGS * SEGMENTS * 6);
    for ring in 0..RINGS {
        for seg in 0..SEGMENTS {
            let a = ring * row + seg;
            let (b, c, d) = (a + row, a + row + 1, a + 1);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    Mesh {
        vertices,
        indices,
        material: 0,
    }
}

/// A unit-width strip in the XZ plane, facing +Y, running from -1 to 1 along X or Z.
pub fn mesh_axis(along_x: bool) -> Mesh {
    let color = if along_x { COLOR_X_AXIS } else { COLOR_Z_AXIS };

    let mut vertices = Vec::with_capacity((AXIS_SEGMENTS + 1) * 2);
    for i in 0..=AXIS_SEGMENTS {
        let along = i as f32 / AXIS_SEGMENTS as f32 * 2. - 1.;

        for across in [-0.5, 0.5] {
            let (x, z) = if along_x {
                (along, across)
            } else {
                (across, along)
            };
            vertices.push(vertex(x, z, color, along.abs()));
        }
    }

    let mut indices = Vec::with_capacity(AXIS_SEGMENTS * 6);
    for i in 0..AXIS_SEGMENTS {
        let a = i * 2;
        let (b, c, d) = (a + 2, a + 3, a + 1);
        // Keep the winding facing +Y for both orientations.
        if along_x {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        } else {
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }

    Mesh {
        vertices,
        indices,
        material: 0,
    }
}

/// Positions of visible atoms, sampled for large molecules.
fn sample_posits(state: &State) -> Box<dyn Iterator<Item = Vec3> + '_> {
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        let posits = &state.mol_editor.mol.common.atom_posits;
        return Box::new(posits.iter().map(|p| (*p).into()));
    }

    let md = &state.volatile.md_local;
    if md.draw_md_mols {
        let viewer = &md.viewer;
        let snap = viewer
            .current_snapshot
            .and_then(|i| viewer.snapshots.get(i))
            .or_else(|| viewer.snapshots.first());

        return Box::new(
            snap.into_iter()
                .flat_map(|s| s.atom_posits.iter().step_by(MD_SAMPLE_RATIO).copied()),
        );
    }

    let peptides = state
        .peptides
        .iter()
        .filter(|m| m.common.visible())
        .flat_map(|m| m.common.atom_posits.iter().step_by(PEP_SAMPLE_RATIO));

    let others = state
        .ligands
        .iter()
        .map(|m| &m.common)
        .chain(state.nucleic_acids.iter().map(|m| &m.common))
        .chain(state.lipids.iter().map(|m| &m.common))
        .filter(|m| m.visible())
        .flat_map(|m| m.atom_posits.iter());

    Box::new(peptides.chain(others).map(|p| (*p).into()))
}

/// Show or hide the floor. When showing, fit its height to the current molecules.
pub fn toggle(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    if state.ui.floor.take().is_some() {
        let len_before = scene.entities.len();
        scene.entities.retain(|ent| ent.class != CLASS);

        // Removal shifts later entities.
        if scene.entities.len() != len_before {
            crate::util::clear_mol_entity_indices(state, None);
            updates.entities = EntityUpdate::All;
        }
        return;
    }

    state.ui.floor = Some(Floor::fit(state));
    update(state, scene, updates);
}

/// Follow the camera, and re-add the floor if a full rebuild cleared the scene's entities. Run
/// after all drawing for the frame.
pub fn update(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let Some(floor) = &mut state.ui.floor else {
        return;
    };

    let placement = Placement::new(floor.y, &scene.camera);
    let rebuilding = matches!(updates.entities, EntityUpdate::All);
    if !rebuilding && floor.placed == Some(placement) {
        return;
    }

    let new = floor.entities(placement);
    floor.placed = Some(placement);

    let mut found = false;
    let existing = scene.entities.iter_mut().filter(|e| e.class == CLASS);
    for (ent, new) in existing.zip(&new) {
        ent.position = new.position;
        ent.scale_partial = new.scale_partial;
        found = true;
    }

    if !found {
        // Appending doesn't shift existing entity indices.
        scene.entities.extend(new);
        updates.entities = EntityUpdate::All;
        return;
    }

    // Don't use `push_class`; it would drop a pending ID or index update.
    match &mut updates.entities {
        EntityUpdate::All => (),
        EntityUpdate::None => updates.entities = EntityUpdate::Classes(vec![CLASS]),
        EntityUpdate::Classes(classes) => classes.push(CLASS),
        _ => updates.entities = EntityUpdate::All,
    }
}
