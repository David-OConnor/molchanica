//! Volumetric fields and their molecular geometry.

mod cube;

use std::{
    io,
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, bond_inference::create_bonds};
pub(crate) use cube::is_cube;
use cube::{Cube, Kind};
use egui::{Color32, ComboBox, DragValue, RichText, Slider, SliderClamping, Ui};
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Mesh, Scene, UP_VEC, Vertex};
use lin_alg::f32::{Quaternion, Vec3};
use mcubes::{MarchingCubes, MeshSide};
use na_seq::Element;

use crate::{
    cam::{VIEW_DIR_FRONT, cam_look_at_outside},
    drawing::EntityClass,
    prefs::OpenType,
    render::{MESH_BOND, MESH_SPHERE_MEDRES, MESH_VOLUME, set_flashlight},
    state::{OperatingMode, State},
    util::clear_mol_entity_indices,
};

const DENSITY: [u8; 3] = [85, 200, 208];
const POSITIVE: [u8; 3] = [66, 145, 250];
const NEGATIVE: [u8; 3] = [240, 85, 89];
const NEUTRAL: [u8; 3] = [242, 242, 242];
pub(crate) const TYPE_COLOR: Color32 = Color32::from_rgb(85, 200, 208);

#[derive(Clone, Decode, Encode)]
pub(crate) struct SavedVolume {
    path: PathBuf,
    grid: Cube,
    potential: Option<Cube>,
    dataset: usize,
    contour: f32,
    opacity: f32,
    potential_range: f32,
    visible: bool,
    show_atoms: bool,
}

impl SavedVolume {
    pub fn load(path: &Path) -> io::Result<Self> {
        let source = Cube::load(path)?;
        let sibling_density = (source.kind == Kind::Potential && source.datasets.len() == 1)
            .then(|| matching_density(path, &source))
            .flatten();
        let (grid, potential) = match sibling_density {
            Some(density) => (density, Some(source)),
            None => (source, None),
        };
        let contour = default_contour(grid.kind, contour_peak(&grid, 0));

        Ok(Self {
            path: path.to_owned(),
            grid,
            potential,
            dataset: 0,
            contour,
            opacity: 0.68,
            potential_range: 0.05,
            visible: true,
            show_atoms: true,
        })
    }

    fn label(&self) -> &'static str {
        if self.potential.is_some() {
            "ESP on electron density"
        } else {
            self.grid.kind.label()
        }
    }

    fn name(&self) -> String {
        self.path
            .file_stem()
            .unwrap_or(self.path.as_os_str())
            .to_string_lossy()
            .into_owned()
    }

    fn validate(&self) -> io::Result<()> {
        self.grid.validate()?;
        if let Some(potential) = &self.potential {
            potential.validate()?;
            if potential.kind != Kind::Potential || !self.grid.matches_grid(potential) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Cube density and potential grids differ",
                ));
            }
        }
        if self.dataset >= self.grid.datasets.len()
            || !self.contour.is_finite()
            || self.contour <= 0.
            || !self.potential_range.is_finite()
            || self.potential_range <= 0.
            || !(0.0..=1.0).contains(&self.opacity)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Cube display settings",
            ));
        }

        Ok(())
    }

    fn bounds(&self) -> (Vec3, f32) {
        let grid = &self.grid;
        if grid.atoms.is_empty() {
            let edges = grid
                .axes
                .iter()
                .zip(grid.dims)
                .map(|(axis, size)| *axis * (size - 1) as f32);
            let (offset, lengths) = edges.fold((Vec3::new_zero(), 0.), |(sum, len), v| {
                (sum + v, len + v.magnitude())
            });

            return (grid.origin + offset * 0.5, lengths * 0.5);
        }

        let center = grid
            .atoms
            .iter()
            .fold(Vec3::new_zero(), |sum, atom| sum + atom.position)
            / grid.atoms.len() as f32;
        let radius = grid
            .atoms
            .iter()
            .map(|atom| (atom.position - center).magnitude())
            .fold(0., f32::max);

        (center, radius + 1.8)
    }
}

fn matching_density(path: &Path, potential: &Cube) -> Option<Cube> {
    let folder = path.parent()?;
    for name in ["Dt.cube", "electron-density.cube", "input.eldens.cube"] {
        if let Ok(density) = Cube::load(&folder.join(name))
            && density.kind == Kind::Density
            && density.datasets.len() == 1
            && density.matches_grid(potential)
        {
            return Some(density);
        }
    }

    None
}

fn contour_peak(grid: &Cube, dataset: usize) -> f32 {
    grid.datasets[dataset]
        .values
        .iter()
        .map(|v| v.abs())
        .fold(0., f32::max)
}

fn default_contour(kind: Kind, peak: f32) -> f32 {
    let recommended = match kind {
        Kind::Density => 0.001,
        Kind::Orbital | Kind::Potential => 0.03,
        Kind::Spin | Kind::Scalar => 0.005,
    };
    if peak > 0. && peak <= recommended {
        return (peak * 0.25).max(f32::MIN_POSITIVE);
    }

    recommended
}

struct Volume {
    view: SavedVolume,
    mesh: Option<Mesh>,
    atoms: Vec<Entity>,
    contour_peak: f32,
}

impl Volume {
    fn new(view: SavedVolume) -> Self {
        let atoms = atom_entities(&view.grid);
        let contour_peak = contour_peak(&view.grid, view.dataset);

        Self {
            view,
            mesh: None,
            atoms,
            contour_peak,
        }
    }
}

#[derive(Default)]
pub(crate) struct Volumes {
    items: Vec<Volume>,
    pub active: Option<usize>,
    dirty: bool,
    drawn: bool,
}

impl Volumes {
    pub fn push(&mut self, volume: SavedVolume) -> usize {
        self.items.push(Volume::new(volume));
        self.dirty = true;

        self.items.len() - 1
    }

    pub fn saved(&self) -> impl Iterator<Item = &SavedVolume> {
        self.items.iter().map(|volume| &volume.view)
    }

    pub fn restore(views: Vec<SavedVolume>, active: Option<usize>) -> io::Result<Self> {
        for view in &views {
            view.validate()?;
        }
        let mut volumes = Self::default();
        for view in views {
            volumes.push(view);
        }
        volumes.active = active.filter(|&i| i < volumes.items.len());

        Ok(volumes)
    }

    pub fn active_center(&self) -> Option<Vec3> {
        self.active_bounds().map(|(center, _)| center)
    }

    pub fn active_bounds(&self) -> Option<(Vec3, f32)> {
        self.items
            .get(self.active?)
            .map(|volume| volume.view.bounds())
    }

    pub fn visible_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.items.iter().filter(|v| v.view.visible).flat_map(|v| {
            let (center, radius) = v.view.bounds();
            let x = Vec3::new(radius, 0., 0.);
            let y = Vec3::new(0., radius, 0.);
            let z = Vec3::new(0., 0., radius);

            [
                center,
                center + x,
                center - x,
                center + y,
                center - y,
                center + z,
                center - z,
            ]
        })
    }
}

pub(crate) fn open(
    state: &mut State,
    path: &Path,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> io::Result<()> {
    let index = match state
        .volumes
        .items
        .iter()
        .position(|volume| volume.view.path == path)
    {
        Some(index) => {
            state.volumes.items[index].view.visible = true;
            state.volumes.dirty = true;
            index
        }
        None => state.volumes.push(SavedVolume::load(path)?),
    };
    let name = state.volumes.items[index].view.name();
    state.update_history(path, OpenType::Cube, Some(name));
    focus(state, scene, updates, index);

    Ok(())
}

fn focus(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates, index: usize) {
    let (center, radius) = state.volumes.items[index].view.bounds();
    state.volumes.active = Some(index);
    state.volatile.active_mol = None;
    state.volatile.active_seq = None;
    state.volatile.orbit_center = None;
    state.ui.selection = crate::selection::Selection::None;
    state.ui.seq_selection.clear();
    state.ui.cam_snapshot = None;
    let distance = crate::cam::frame_dist(&scene.camera, radius);
    cam_look_at_outside(&mut scene.camera, center, center + VIEW_DIR_FRONT, distance);
    if let ControlScheme::Arc { center: orbit } = &mut scene.input_settings.control_scheme {
        *orbit = center;
    }

    crate::cam::set_fog(state, &mut scene.camera);
    set_flashlight(scene);
    updates.camera = true;
    updates.lighting = true;
}

pub(crate) fn close(state: &mut State, scene: &mut Scene, index: usize) {
    let path = state.volumes.items.remove(index).view.path;
    state.volumes.dirty = true;
    for history in &mut state.to_save.open_history {
        if history.type_ == OpenType::Cube && history.path == path {
            history.last_session = false;
        }
    }
    state.volumes.active = match state.volumes.active {
        Some(active) if active == index => None,
        Some(active) if active > index => Some(active - 1),
        active => active,
    };
    crate::util::reset_orbit_center(state, scene);
    state.update_save_prefs();
}

pub(crate) fn sidebar(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.volumes.items.is_empty() {
        return;
    }
    if state.volatile.active_mol.is_some() || state.volatile.active_seq.is_some() {
        state.volumes.active = None;
    }

    ui.label(RichText::new("Cube maps").strong());
    let mut focus_index = None;
    let mut close_index = None;
    let mut changed = false;
    for (index, volume) in state.volumes.items.iter_mut().enumerate() {
        ui.push_id(("cube", index), |ui| {
            let active = state.volumes.active == Some(index);
            let view = &mut volume.view;
            ui.horizontal(|ui| {
                ui.label(RichText::new("C").color(TYPE_COLOR).strong())
                    .on_hover_text(view.label());
                if ui
                    .selectable_label(active, view.name())
                    .on_hover_text(view.path.display().to_string())
                    .clicked()
                {
                    focus_index = Some(index);
                }
                if ui
                    .selectable_label(view.visible, "👁")
                    .on_hover_text("Show or hide this Cube map")
                    .clicked()
                {
                    view.visible = !view.visible;
                    changed = true;
                }
                if ui
                    .small_button("×")
                    .on_hover_text("Close this Cube map")
                    .clicked()
                {
                    close_index = Some(index);
                }
            });
            if active {
                let (update, remesh) = settings_ui(view, &mut volume.contour_peak, ui);
                changed |= update || remesh;
                if remesh {
                    volume.mesh = None;
                }
            }
        });
    }
    state.volumes.dirty |= changed;
    if let Some(index) = focus_index {
        focus(state, scene, updates, index);
    }
    if let Some(index) = close_index {
        close(state, scene, index);
    }
}

fn settings_ui(view: &mut SavedVolume, peak: &mut f32, ui: &mut Ui) -> (bool, bool) {
    let mut update = false;
    let mut remesh = false;
    ui.indent("cube_settings", |ui| {
        ui.small(view.label());
        if view.grid.datasets.len() > 1 {
            let previous = view.dataset;
            ComboBox::from_id_salt("dataset")
                .selected_text(&view.grid.datasets[previous].label)
                .show_ui(ui, |ui| {
                    for (i, dataset) in view.grid.datasets.iter().enumerate() {
                        ui.selectable_value(&mut view.dataset, i, &dataset.label);
                    }
                });
            if view.dataset != previous {
                *peak = contour_peak(&view.grid, view.dataset);
                view.contour = default_contour(view.grid.kind, *peak);
                remesh = true;
            }
        }
        ui.horizontal(|ui| {
            let default = default_contour(view.grid.kind, *peak);
            let min = (default * 0.01).max(f32::MIN_POSITIVE);
            let max = peak.max(2. * default);
            let previous = view.contour;
            if ui
                .add(
                    Slider::new(&mut view.contour, min..=max)
                        .text("Contour")
                        .logarithmic(true)
                        .clamping(SliderClamping::Never)
                        .max_decimals(8),
                )
                .on_hover_text("Isovalue in the Cube file's units")
                .changed()
            {
                if view.contour.is_finite() && view.contour > 0. {
                    remesh = true;
                } else {
                    view.contour = previous;
                }
            }
            if ui.small_button("Reset").clicked() {
                view.contour = default;
                remesh = true;
            }
        });
        update |= ui
            .add(Slider::new(&mut view.opacity, 0.08..=1.).text("Opacity"))
            .changed();
        if !view.grid.atoms.is_empty() {
            update |= ui
                .checkbox(&mut view.show_atoms, "Show atoms and bonds")
                .changed();
        }
        if view.potential.is_some() {
            ui.horizontal(|ui| {
                ui.label("ESP scale");
                remesh |= ui
                    .add(
                        DragValue::new(&mut view.potential_range)
                            .speed(0.001)
                            .range(0.001..=10.),
                    )
                    .changed();
            });
        }
        if view.potential.is_some() || view.grid.kind != Kind::Density {
            ui.horizontal(|ui| {
                ui.colored_label(
                    Color32::from_rgb(NEGATIVE[0], NEGATIVE[1], NEGATIVE[2]),
                    "− Negative",
                );
                ui.colored_label(
                    Color32::from_rgb(POSITIVE[0], POSITIVE[1], POSITIVE[2]),
                    "+ Positive",
                );
            });
        }
    });

    (update, remesh)
}

fn atom_entities(grid: &Cube) -> Vec<Entity> {
    let mut entities = Vec::new();
    let mut atoms = Vec::new();
    for (i, atom) in grid.atoms.iter().enumerate() {
        if let Ok(element) = Element::from_atomic_number(atom.number) {
            atoms.push(AtomGeneric {
                serial_number: i as u32,
                element,
                posit: atom.position.into(),
                ..Default::default()
            });
            let radius = (element.covalent_radius() as f32 * 0.48).clamp(0.17, 0.44);
            entities.push(Entity::new(
                MESH_SPHERE_MEDRES,
                atom.position,
                Quaternion::new_identity(),
                radius,
                element.color(),
                0.3,
            ));
        }
    }
    for bond in create_bonds(&atoms) {
        let a = &grid.atoms[bond.atom_0_sn as usize];
        let b = &grid.atoms[bond.atom_1_sn as usize];
        let delta = b.position - a.position;
        let length = delta.magnitude();
        let orientation = Quaternion::from_unit_vecs(UP_VEC, delta / length);
        for (end, fraction) in [(a, 0.25), (b, 0.75)] {
            let color = Element::from_atomic_number(end.number).unwrap().color();
            let mut entity = Entity::new(
                MESH_BOND,
                a.position + delta * fraction,
                orientation,
                1.,
                color,
                0.3,
            );
            entity.scale_partial = Some(Vec3::new(0.85, length * 0.5, 0.85));
            entities.push(entity);
        }
    }
    for entity in &mut entities {
        entity.class = EntityClass::Volume as u32;
    }

    entities
}

fn potential_color(value: f32, range: f32) -> [u8; 3] {
    let sign_color = if value >= 0. { POSITIVE } else { NEGATIVE };
    let t = (value.abs() / range).min(1.);

    std::array::from_fn(|i| ((1. - t) * NEUTRAL[i] as f32 + t * sign_color[i] as f32).round() as u8)
}

fn sample(grid: &Cube, p: Vec3) -> f32 {
    let [nx, ny, nz] = grid.dims;
    let [x, y, z] = p.to_arr();
    let coords = [x, y, z];
    let bounds: [(usize, f32); 3] = std::array::from_fn(|i| {
        let coord = coords[i].clamp(0., (grid.dims[i] - 1) as f32);
        let lower = (coord as usize).min(grid.dims[i] - 2);

        (lower, coord - lower as f32)
    });
    let [(x, u), (y, v), (z, w)] = bounds;
    let mut value = 0.;
    let values = &grid.datasets[0].values;
    for (k, weight_z) in [(z, 1. - w), ((z + 1).min(nz - 1), w)] {
        for (j, weight_y) in [(y, 1. - v), (y + 1, v)] {
            value += (values[x + nx * (j + ny * k)] * (1. - u)
                + values[x + 1 + nx * (j + ny * k)] * u)
                * weight_y
                * weight_z;
        }
    }

    value
}

fn surface(view: &SavedVolume) -> Mesh {
    let grid = &view.grid;
    let [nx, ny, nz] = grid.dims;
    let [a, b, c] = grid.axes;
    let det = a.dot(b.cross(c));
    let [ra, rb, rc] = [b.cross(c) / det, c.cross(a) / det, a.cross(b) / det];
    let values = &grid.datasets[view.dataset].values;
    let mut mesh = Mesh::default();

    for sign in [1., -1.] {
        if (sign < 0. && grid.kind == Kind::Density)
            || !values.iter().any(|value| sign * value > view.contour)
        {
            continue;
        }

        let values = values.iter().map(|value| sign * value).collect();
        // Mesh in voxel coordinates, then apply the full affine transform.
        let dims = (nx, ny, nz);
        let size = (nx as f32, ny as f32, nz as f32);
        let mc = MarchingCubes::new(dims, size, size, Vec3::new_zero(), values, view.contour)
            .expect("validated Cube grid");
        let mut contour = mc.generate(MeshSide::OutsideOnly);
        let offset = mesh.vertices.len();
        mesh.vertices.extend(contour.vertices.into_iter().map(|v| {
            let p = v.posit;
            let world = grid.origin + a * p.x + b * p.y + c * p.z;
            let normal = v.normal;
            let normal = ra * normal.x + rb * normal.y + rc * normal.z;
            let length = normal.magnitude();
            let normal = if length.is_finite() && length > 0. {
                normal / length
            } else {
                UP_VEC
            };
            let [r, g, b] = match &view.potential {
                Some(potential) => potential_color(sample(potential, p), view.potential_range),
                None if grid.kind == Kind::Density => DENSITY,
                None if sign > 0. => POSITIVE,
                None => NEGATIVE,
            };
            let mut vertex = Vertex::new(world.to_arr(), normal);
            vertex.color = Some((r, g, b, 255));

            vertex
        }));
        if det < 0. {
            for triangle in contour.indices.chunks_exact_mut(3) {
                triangle.swap(1, 2);
            }
        }
        mesh.indices
            .extend(contour.indices.into_iter().map(|i| offset + i));
    }

    mesh
}

pub(crate) fn sync_scene(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let viewing = state.volatile.operating_mode == OperatingMode::Primary;
    let show = viewing && state.volumes.items.iter().any(|v| v.view.visible);
    if !state.volumes.dirty && !state.volumes.drawn && !show {
        return;
    }
    let lost = show
        && matches!(updates.entities, EntityUpdate::All)
        && !scene
            .entities
            .iter()
            .any(|entity| entity.class == EntityClass::Volume as u32);
    if !state.volumes.dirty && show == state.volumes.drawn && !lost {
        return;
    }

    scene
        .entities
        .retain(|entity| entity.class != EntityClass::Volume as u32);
    let mut combined = Mesh::default();
    if show {
        let mut geometry: Vec<&Cube> = Vec::new();
        for volume in &mut state.volumes.items {
            if !volume.view.visible {
                continue;
            }

            let mesh = volume.mesh.get_or_insert_with(|| surface(&volume.view));
            let alpha = (volume.view.opacity * 255.).round().clamp(1., 255.) as u8;
            let offset = combined.vertices.len();
            combined
                .vertices
                .extend(mesh.vertices.iter().copied().map(|mut vertex| {
                    if let Some((r, g, b, _)) = vertex.color {
                        vertex.color = Some((r, g, b, alpha));
                    }

                    vertex
                }));
            combined
                .indices
                .extend(mesh.indices.iter().map(|i| offset + i));
            if volume.view.show_atoms
                && !geometry
                    .iter()
                    .any(|grid| volume.view.grid.matches_grid(grid))
            {
                scene.entities.extend_from_slice(&volume.atoms);
                geometry.push(&volume.view.grid);
            }
        }
        if !combined.indices.is_empty() {
            let mut entity = Entity::new(
                MESH_VOLUME,
                Vec3::new_zero(),
                Quaternion::new_identity(),
                1.,
                (1., 1., 1.),
                0.25,
            );
            entity.class = EntityClass::Volume as u32;
            // Select the translucent pass. Surface alpha comes from each vertex.
            entity.opacity = 0.5;
            scene.entities.push(entity);
        }
    }

    state.volumes.dirty = false;
    state.volumes.drawn = show;
    scene.meshes[MESH_VOLUME] = combined;
    clear_mol_entity_indices(state, None);
    updates.entities = EntityUpdate::All;
    updates.meshes = true;
}
