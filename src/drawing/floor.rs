//! An optional floor below the scene, with an XZ grid and red X and blue Z axes.
//!
//! The floor is a disc centered under the camera, fading out before the far clip plane, so it
//! appears infinite. It's transparent, so atoms below it remain visible. Visible molecular
//! meshes cast shadows from the scene's point lights.

use graphics::{Camera, EngineUpdates, Entity, EntityUpdate, FloorShadows, Mesh, Scene, Vertex};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    cam::RENDER_DIST_FAR,
    drawing::EntityClass,
    render::{
        MESH_FLOOR, MESH_FLOOR_AXIS_X, MESH_FLOOR_AXIS_Z, MESH_FLOOR_GRID_X, MESH_FLOOR_GRID_Z,
    },
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
const GRID_SEGMENTS: usize = 16;

/// Re-center when the camera moves this far.
const RECENTER_STEP: f32 = 10.;
/// Axis line width, as a ratio of camera height above the floor. Keeps lines a few pixels wide
/// below the camera.
const AXIS_WIDTH_RATIO: f32 = 0.0025;
const AXIS_WIDTH_MIN: f32 = 0.2;
const AXIS_WIDTH_STEP: f32 = 0.05;

/// Grid line width as a fraction of the axis width.
const GRID_WIDTH_RATIO: f32 = 0.2;
/// Grid line opacity before fading, from 1 to 255.
const GRID_ALPHA: u8 = 50;

// Separate the depth-writing layers, including at the far end of the floor.
const LAYER_OFFSET_RATIO: f32 = 0.0025;

const MARGIN_BELOW: f32 = 5.;
/// Height snaps to multiples of this.
const HEIGHT_SNAP: f32 = 10.;
/// Height used when no molecules are open.
const HEIGHT_DEFAULT: f32 = -30.;

const PEP_SAMPLE_RATIO: usize = 10;
const MD_SAMPLE_RATIO: usize = 10;

const CLASS: u32 = EntityClass::Floor as u32;
const SHADOW_CASTERS: [EntityClass; 9] = [
    EntityClass::Protein,
    EntityClass::Ligand,
    EntityClass::NucleicAcid,
    EntityClass::Lipid,
    EntityClass::SecondaryStructure,
    EntityClass::SaSurface,
    EntityClass::SaSurfaceDots,
    EntityClass::Representation,
    EntityClass::WaterModel,
];

/// Where the floor entities are, relative to the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    center_x: f32,
    center_z: f32,
    axis_width: f32,
    layer_offset: f32,
    grid_spacing: u8,
}

impl Placement {
    fn new(y: f32, cam: &Camera, grid_spacing: u8) -> Self {
        let snap = |v: f32, step: f32| (v / step).round() * step;

        let p = cam.position;
        let height = p.y - y;
        let width = (height.abs() * AXIS_WIDTH_RATIO).max(AXIS_WIDTH_MIN);

        Self {
            center_x: snap(p.x, RECENTER_STEP),
            center_z: snap(p.z, RECENTER_STEP),
            axis_width: snap(width, AXIS_WIDTH_STEP),
            layer_offset: height * LAYER_OFFSET_RATIO,
            grid_spacing: grid_spacing.max(1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor {
    /// Absolute height, in Å.
    pub y: f32,
    /// The placement of the current floor entities.
    placed: Option<Placement>,
    had_molecules: bool,
}

impl Floor {
    /// Place below the lowest atom.
    pub fn fit(state: &State) -> Self {
        Self {
            y: fit_height(state),
            placed: None,
            had_molecules: has_open_molecules(state),
        }
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

        // Clip each line to the disc.
        let half_len = |offset: f32| (RADIUS.powi(2) - offset.powi(2)).max(0.).sqrt();
        let spacing = f32::from(pl.grid_spacing);
        let steps = (RADIUS / spacing).ceil() as i32;
        let grid_width = (pl.axis_width * GRID_WIDTH_RATIO).min(spacing * 0.25);

        // Each layer sits slightly closer to the camera than the one drawn before it.
        let mut entities = Vec::with_capacity(3 + 2 * (2 * steps + 1) as usize);
        entities.extend([
            ent(
                MESH_FLOOR,
                Vec3::new(pl.center_x, self.y, pl.center_z),
                Vec3::new(RADIUS, 1., RADIUS),
            ),
            ent(
                MESH_FLOOR_AXIS_X,
                Vec3::new(pl.center_x, self.y + 3. * pl.layer_offset, 0.),
                Vec3::new(half_len(pl.center_z), 1., pl.axis_width),
            ),
            ent(
                MESH_FLOOR_AXIS_Z,
                Vec3::new(0., self.y + 4. * pl.layer_offset, pl.center_z),
                Vec3::new(pl.axis_width, 1., half_len(pl.center_x)),
            ),
        ]);

        for (mesh, along_x, across_center, y) in [
            (
                MESH_FLOOR_GRID_X,
                true,
                pl.center_z,
                self.y + pl.layer_offset,
            ),
            (
                MESH_FLOOR_GRID_Z,
                false,
                pl.center_x,
                self.y + 2. * pl.layer_offset,
            ),
        ] {
            let nearest_step = (across_center / spacing).round();

            for step in -steps..=steps {
                let across = (nearest_step + step as f32) * spacing;
                let line_half_len = half_len(across - across_center);

                // Keep the instance count stable as the camera pans. The axis occupies zero.
                let width = if across == 0. { 0. } else { grid_width };
                let (position, scale) = if along_x {
                    (
                        Vec3::new(pl.center_x, y, across),
                        Vec3::new(line_half_len, 1., width),
                    )
                } else {
                    (
                        Vec3::new(across, y, pl.center_z),
                        Vec3::new(width, 1., line_half_len),
                    )
                };

                entities.push(ent(mesh, position, scale));
            }
        }

        entities
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
    mesh_line(along_x, axis_color(along_x), AXIS_SEGMENTS)
}

pub fn mesh_grid_line(along_x: bool) -> Mesh {
    let (r, g, b, _) = axis_color(along_x);
    let (fr, fg, fb, _) = COLOR_FLOOR;
    let color = (r.midpoint(fr), g.midpoint(fg), b.midpoint(fb), GRID_ALPHA);

    mesh_line(along_x, color, GRID_SEGMENTS)
}

fn axis_color(along_x: bool) -> Rgba {
    if along_x { COLOR_X_AXIS } else { COLOR_Z_AXIS }
}

fn mesh_line(along_x: bool, color: Rgba, segments: usize) -> Mesh {
    let mut vertices = Vec::with_capacity((segments + 1) * 2);
    for i in 0..=segments {
        let along = i as f32 / segments as f32 * 2. - 1.;

        for across in [-0.5, 0.5] {
            let (x, z) = if along_x {
                (along, across)
            } else {
                (across, along)
            };
            vertices.push(vertex(x, z, color, along.abs()));
        }
    }

    let mut indices = Vec::with_capacity(segments * 6);
    for i in 0..segments {
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

fn has_open_molecules(state: &State) -> bool {
    !state.peptides.is_empty()
        || !state.ligands.is_empty()
        || !state.nucleic_acids.is_empty()
        || !state.lipids.is_empty()
        || (state.volatile.operating_mode == OperatingMode::MolEditor
            && !state.mol_editor.mol.common.atom_posits.is_empty())
        || (state.volatile.md_local.draw_md_mols
            && !state.volatile.md_local.viewer.snapshots.is_empty())
}

fn fit_height(state: &State) -> f32 {
    sample_posits(state)
        .map(|p| p.y)
        .reduce(f32::min)
        .map_or(HEIGHT_DEFAULT, |min| {
            ((min - MARGIN_BELOW) / HEIGHT_SNAP).floor() * HEIGHT_SNAP
        })
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

fn remove_entities(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let len_before = scene.entities.len();
    scene.entities.retain(|ent| ent.class != CLASS);

    if scene.entities.len() != len_before {
        crate::util::clear_mol_entity_indices(state, None);
        updates.entities = EntityUpdate::All;
    }
}

/// Show or hide the floor. When showing, fit its height to the current molecules.
pub fn toggle(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let visible = !state.to_save.ui_prefs.floor_visible;
    state.to_save.ui_prefs.floor_visible = visible;
    state.to_save.save_flag = true;

    if !visible {
        state.ui.floor = None;
        scene.floor_shadows = None;
        remove_entities(state, scene, updates);

        return;
    }

    state.ui.floor = Some(Floor::fit(state));
    update(state, scene, updates);
}

/// Refit on the first molecule, follow the camera, and restore missing floor entities.
/// Run after all drawing for the frame.
pub fn update(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let Some(mut floor) = state.ui.floor else {
        return;
    };

    let has_molecules = has_open_molecules(state);
    let refit = has_molecules && !floor.had_molecules;
    if refit {
        floor.y = fit_height(state);
    }
    floor.had_molecules = has_molecules;
    let floor = state.ui.floor.insert(floor);

    let shadows = scene.floor_shadows.get_or_insert_with(|| FloorShadows {
        y: floor.y,
        receiver_mesh: MESH_FLOOR,
        caster_classes: SHADOW_CASTERS.map(|class| class as u32).to_vec(),
        strength: state.to_save.graphics.floor_shadow_strength,
    });
    shadows.y = floor.y;
    shadows.strength = state.to_save.graphics.floor_shadow_strength;

    let placement = Placement::new(
        floor.y,
        &scene.camera,
        state.to_save.ui_prefs.floor_grid_spacing,
    );
    let rebuilding = matches!(updates.entities, EntityUpdate::All);
    if !refit && !rebuilding && floor.placed == Some(placement) {
        return;
    }

    let spacing_changed = floor
        .placed
        .is_some_and(|pl| pl.grid_spacing != placement.grid_spacing);
    let new = floor.entities(placement);
    floor.placed = Some(placement);

    if spacing_changed {
        remove_entities(state, scene, updates);
    }

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
