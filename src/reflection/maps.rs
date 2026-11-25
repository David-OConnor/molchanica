use std::{
    io,
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};
use bio_files::DensityMap;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Mesh, Scene, UP_VEC, Vertex};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};
use mcubes::{MarchingCubes, MeshSide};
use mol_defs::reflection::{DENSITY_CELL_MARGIN, DENSITY_MAX_DIST, DensityRect};
use na_seq::Element;

use super::DensityRectExt;
use crate::{
    cam::{self, VIEW_DIR_FRONT},
    drawing::{EntityClass, append_translucent_mesh, atoms_bonds::ATOM_SHININESS},
    prefs::OpenType,
    render::{MESH_DENSITY_SURFACE, MESH_SPHERE_LOWRES, set_flashlight},
    state::{OperatingMode, State},
    util::clear_mol_entity_indices,
};

#[derive(Decode, Encode)]
pub(crate) struct DensityGrid {
    pub origin: Vec3F64,
    pub step: [f64; 3],
    pub dims: [usize; 3],
    pub values: Vec<f32>,
}

impl DensityGrid {
    pub fn from_map(map: &DensityMap) -> io::Result<Self> {
        let file_dims = [map.hdr.nx, map.hdr.ny, map.hdr.nz].map(|n| n as usize);
        let [nx, ny, nz] = map.perm_c2f.map(|axis| file_dims[axis]);
        let hdr = &map.hdr.inner;
        let cell = &hdr.cell;
        let step = [
            cell.a / f64::from(hdr.mx),
            cell.b / f64::from(hdr.my),
            cell.c / f64::from(hdr.mz),
        ];
        let mut values = Vec::with_capacity(map.data.len());
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let crystal = [x, y, z];
                    let [col, row, section] = map.perm_f2c.map(|axis| crystal[axis]);
                    let i = col + file_dims[0] * (row + file_dims[1] * section);
                    values.push(map.density_to_sig(map.data[i]));
                }
            }
        }
        let grid = Self {
            origin: cell.fractional_to_cartesian(map.origin_frac),
            step,
            dims: [nx, ny, nz],
            values,
        };
        grid.validate()?;

        Ok(grid)
    }

    pub fn validate(&self) -> io::Result<()> {
        if self.dims.contains(&0)
            || self.dims.iter().try_fold(1usize, |n, d| n.checked_mul(*d))
                != Some(self.values.len())
            || self.step.iter().any(|v| !v.is_finite() || *v <= 0.)
            || !self.origin.to_arr().iter().all(|v| v.is_finite())
            || self.values.iter().any(|v| !v.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid density grid",
            ));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Decode, Encode)]
pub(crate) struct DensityDisplay {
    pub surface: bool,
    pub dots: bool,
    pub contour: f32,
    pub opacity: f32,
}

impl Default for DensityDisplay {
    fn default() -> Self {
        Self {
            surface: true,
            dots: false,
            contour: 1.8,
            opacity: 0.5,
        }
    }
}

pub(crate) struct DensityMapView {
    pub name: String,
    pub path: Option<PathBuf>,
    pub map: DensityMap,
    pub grid: DensityGrid,
    pub display: DensityDisplay,
    mesh: Option<Mesh>,
}

impl DensityMapView {
    pub fn new(
        name: String,
        path: Option<PathBuf>,
        map: DensityMap,
        grid: DensityGrid,
        display: DensityDisplay,
    ) -> Self {
        Self {
            name,
            path,
            map,
            grid,
            display,
            mesh: None,
        }
    }

    pub fn invalidate_mesh(&mut self) {
        self.mesh = None;
    }

    fn axes(&self) -> [Vec3; 3] {
        let cell = &self.map.hdr.inner.cell;
        let (a, b, c) = cell.ortho.to_cols();
        let [dx, dy, dz] = self.grid.step;

        [
            (a * (dx / cell.a)).into(),
            (b * (dy / cell.b)).into(),
            (c * (dz / cell.c)).into(),
        ]
    }

    fn bounds(&self) -> (Vec3, f32) {
        let [a, b, c] = self.axes();
        let [nx, ny, nz] = self.grid.dims;
        let edges = [
            a * (nx - 1) as f32,
            b * (ny - 1) as f32,
            c * (nz - 1) as f32,
        ];
        let center = Vec3::from(self.grid.origin) + (edges[0] + edges[1] + edges[2]) * 0.5;
        let radius = 0.5 * edges.iter().map(|v| v.magnitude()).sum::<f32>();

        (center, radius.max(1.))
    }
}

#[derive(Default)]
pub(crate) struct DensityMaps {
    pub items: Vec<DensityMapView>,
    pub active: Option<usize>,
    pub surfaces_dirty: bool,
    pub dots_dirty: bool,
    surfaces_drawn: bool,
    dots_drawn: bool,
    dots_present: bool,
}

impl DensityMaps {
    pub fn push(&mut self, view: DensityMapView) -> usize {
        let index = self.items.len();
        self.surfaces_dirty |= view.display.surface;
        self.dots_dirty |= view.display.dots;
        self.items.push(view);

        index
    }

    pub fn reopen(&mut self, index: usize) {
        let display = &mut self.items[index].display;
        self.surfaces_dirty |= !display.surface;
        display.surface = true;
        self.active = Some(index);
    }

    pub fn active_map(&self) -> Option<&DensityMapView> {
        self.active.and_then(|i| self.items.get(i))
    }

    pub fn active_bounds(&self) -> Option<(Vec3, f32)> {
        self.active_map().map(DensityMapView::bounds)
    }

    pub fn visible_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.items
            .iter()
            .filter(|map| map.display.surface || map.display.dots)
            .flat_map(|map| {
                let (center, radius) = map.bounds();
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

impl State {
    pub(crate) fn load_density(
        &mut self,
        map: DensityMap,
        name: String,
        path: Option<PathBuf>,
        peptide_i: Option<usize>,
    ) -> io::Result<usize> {
        let existing = self.density_maps.items.iter().position(|view| match &path {
            Some(path) => view.path.as_ref() == Some(path),
            None => view.path.is_none() && view.name == name,
        });
        if let Some(i) = existing {
            self.density_maps.reopen(i);

            return Ok(i);
        }

        let atoms: Vec<_> = peptide_i
            .and_then(|i| self.peptides.get(i))
            .into_iter()
            .flat_map(|mol| &mol.common.atoms)
            .filter(|atom| atom.element != Element::Hydrogen)
            .map(|atom| atom.posit)
            .collect();
        let grid = if atoms.is_empty() {
            DensityGrid::from_map(&map)?
        } else {
            let resources = self.computation_resources();
            let rect = DensityRect::new(&atoms, &map, DENSITY_CELL_MARGIN);
            let points = rect.make_densities(
                &resources.device,
                #[cfg(feature = "cuda")]
                &resources.kernel_reflections,
                &atoms,
                &map.hdr.inner.cell,
                DENSITY_MAX_DIST,
            );
            let grid = DensityGrid {
                origin: rect.origin_cart,
                step: rect.step,
                dims: rect.dims,
                values: points.into_iter().map(|p| p.density as f32).collect(),
            };
            grid.validate()?;

            grid
        };
        let view = DensityMapView::new(name, path, map, grid, DensityDisplay::default());
        let i = self.density_maps.push(view);
        self.density_maps.active = Some(i);

        Ok(i)
    }

    pub(crate) fn close_density(&mut self, index: usize) {
        let maps = &mut self.density_maps;
        let view = maps.items.remove(index);
        maps.surfaces_dirty |= view.display.surface;
        maps.dots_dirty |= view.display.dots;
        maps.active = match maps.active {
            Some(active) if active == index => {
                maps.items.len().checked_sub(1).map(|last| index.min(last))
            }
            Some(active) if active > index => Some(active - 1),
            active => active,
        };

        if let Some(path) = view.path {
            for history in &mut self.to_save.open_history {
                if history.type_ == OpenType::Map && history.path == path {
                    history.last_session = false;
                }
            }
            self.update_save_prefs();
        }
    }
}

pub(crate) fn map_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn frame_map(
    state: &State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    index: usize,
) {
    let (center, radius) = state.density_maps.items[index].bounds();
    let distance = cam::frame_dist(&scene.camera, radius);
    cam::cam_look_at_outside(&mut scene.camera, center, center + VIEW_DIR_FRONT, distance);
    if let ControlScheme::Arc { center: orbit } = &mut scene.input_settings.control_scheme {
        *orbit = center;
    }
    cam::set_fog(state, &mut scene.camera);
    set_flashlight(scene);
    updates.camera = true;
    updates.lighting = true;
}

fn surface(view: &DensityMapView) -> Mesh {
    let [nx, ny, nz] = view.grid.dims;
    let contour = view.display.contour;
    if view.grid.dims.iter().any(|&n| n < 2) || !view.grid.values.iter().any(|&v| v > contour) {
        return Mesh::default();
    }

    let [a, b, c] = view.axes();
    let det = a.dot(b.cross(c));
    let [ra, rb, rc] = [b.cross(c) / det, c.cross(a) / det, a.cross(b) / det];
    let origin = Vec3::from(view.grid.origin);
    let size = (nx as f32, ny as f32, nz as f32);
    let mc = MarchingCubes::new(
        (nx, ny, nz),
        size,
        size,
        Vec3::new_zero(),
        view.grid.values.clone(),
        contour,
    )
    .expect("validated density grid");
    let mesh = mc.generate(MeshSide::OutsideOnly);
    let vertices = mesh
        .vertices
        .into_iter()
        .map(|v| {
            let p = v.posit;
            let world = origin + a * p.x + b * p.y + c * p.z;
            let n = v.normal;
            let normal = ra * n.x + rb * n.y + rc * n.z;
            let length = normal.magnitude();
            let normal = if length.is_finite() && length > 0. {
                normal / length
            } else {
                UP_VEC
            };
            let mut vertex = Vertex::new(world.to_arr(), normal);
            vertex.color = Some((0, 255, 255, 255));

            vertex
        })
        .collect();

    Mesh {
        vertices,
        indices: mesh.indices,
        material: 0,
    }
}

fn append_dots(entities: &mut Vec<Entity>, view: &DensityMapView) {
    let [nx, ny, _] = view.grid.dims;
    let [a, b, c] = view.axes();
    let origin = Vec3::from(view.grid.origin);

    for (i, &density) in view.grid.values.iter().enumerate() {
        if density <= 0. {
            continue;
        }

        let xyz =
            origin + a * (i % nx) as f32 + b * (i / nx % ny) as f32 + c * (i / nx / ny) as f32;
        let mut dot = Entity::new(
            MESH_SPHERE_LOWRES,
            xyz,
            Quaternion::new_identity(),
            0.03 * density.powf(1.3),
            (density * 2., 0., 0.2),
            ATOM_SHININESS,
        );
        dot.class = EntityClass::DensityPoint as u32;
        entities.push(dot);
    }
}

pub(crate) fn sync_scene(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let maps = &mut state.density_maps;
    let viewing = state.volatile.operating_mode == OperatingMode::Primary;
    let lost_entities = matches!(updates.entities, EntityUpdate::All);
    let show_surfaces = viewing && maps.items.iter().any(|view| view.display.surface);
    let redraw_surface = maps.surfaces_dirty
        || show_surfaces != maps.surfaces_drawn
        || (!scene.meshes[MESH_DENSITY_SURFACE].indices.is_empty()
            && lost_entities
            && !scene
                .entities
                .iter()
                .any(|e| e.class == EntityClass::DensitySurface as u32));
    let show_dots = viewing && maps.items.iter().any(|view| view.display.dots);
    let redraw_dots = maps.dots_dirty
        || show_dots != maps.dots_drawn
        || (maps.dots_present
            && lost_entities
            && !scene
                .entities
                .iter()
                .any(|e| e.class == EntityClass::DensityPoint as u32));
    if !redraw_surface && !redraw_dots {
        return;
    }

    scene.entities.retain(|e| {
        !(redraw_surface && e.class == EntityClass::DensitySurface as u32
            || redraw_dots && e.class == EntityClass::DensityPoint as u32)
    });

    if redraw_surface {
        let mut combined = Mesh::default();
        for view in maps
            .items
            .iter_mut()
            .filter(|view| show_surfaces && view.display.surface)
        {
            let mesh = std::mem::take(&mut view.mesh).unwrap_or_else(|| surface(view));
            append_translucent_mesh(&mut combined, &mesh, view.display.opacity);
            view.mesh = Some(mesh);
        }
        if !combined.indices.is_empty() {
            let mut entity = Entity::new(
                MESH_DENSITY_SURFACE,
                Vec3::new_zero(),
                Quaternion::new_identity(),
                1.,
                (1., 1., 1.),
                ATOM_SHININESS,
            );
            entity.class = EntityClass::DensitySurface as u32;
            // Vertex alpha holds each map's opacity.
            entity.opacity = 0.5;
            scene.entities.push(entity);
        }
        scene.meshes[MESH_DENSITY_SURFACE] = combined;
        updates.meshes = true;
        maps.surfaces_dirty = false;
        maps.surfaces_drawn = show_surfaces;
    }

    if redraw_dots {
        let before = scene.entities.len();
        for view in maps
            .items
            .iter()
            .filter(|view| show_dots && view.display.dots)
        {
            append_dots(&mut scene.entities, view);
        }
        maps.dots_present = scene.entities.len() > before;
        maps.dots_drawn = show_dots;
        maps.dots_dirty = false;
    }

    clear_mol_entity_indices(state, None);
    updates.entities = EntityUpdate::All;
}
