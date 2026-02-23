//! Protein pockets. Used for pharmacophore and other screening.

use std::{
    collections::{HashMap, HashSet},
    io,
    io::{Cursor, ErrorKind},
    path::Path,
};

use bincode::{
    BorrowDecode, Decode, Encode,
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
};
use bio_files::{Mol2, Sdf};
use graphics::{EngineUpdates, Mesh};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};

use crate::{
    drawing::EntityClass,
    molecules::{
        Atom, MolGenericRef, MolGenericTrait, MolType, MoleculePeptide,
        common::{MoleculeCommon, reassign_bond_indices},
        small::MoleculeSmall,
    },
    render::MESH_POCKET_START,
    sfc_mesh::{MeshColoring, apply_mesh_colors, get_mesh_colors, make_sas_mesh},
};

// A larger probe radius will make the pocket tighter and coarser. Tune this to be realistic.
// Note that this is added to VDW radius.
pub const PROBE_RADIUS_EXCLUDED_VOL: f32 = 0.8;
pub const POCKET_DIST_THRESH_DEFAULT: f64 = 11.;

// Larger values result in a smoother mesh.
pub const MESH_PROBE_RADIUS: f32 = 1.0;

// Lower is more expensive, but this pocket is relatively small compared to ones we display
// over whole proteins.
pub const POCKET_MESH_PRECISION: f32 = 0.5;

// Smaller cells reduce false candidates; bigger cells reduce grid size,
// and therefor make screening cheaper.
const CELL_SIZE_SPHERE: f32 = 0.5;

// 0.3 - 0.5 Angstroms is usually sufficient precision for screening.
// Smaller = more memory, smoother boundaries.
const VOXEL_RESOLUTION: f64 = 0.5;

/// For excluded volume.
#[derive(Clone, Copy, Debug, Default, Encode, Decode)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

// todo: How should we represent motion here?
#[derive(Clone, Debug)]
pub struct Pocket {
    /// Contains atoms around the pocket only.
    /// todo: Should this include enough atoms to perform basic MD, or just cover the surface?
    pub common: MoleculeCommon,
    /// Used to rotate the mesh, so we don't have to regenerate it when
    /// the user rotates the pocket.
    pub mesh_orientation: Quaternion, // todo: Unused
    /// This pivot must match the rotation we use for the inner
    /// molecules; this is the molecule's centroid.
    pub mesh_pivot: Vec3F32, // todo: Unused
    pub surface_mesh: Mesh,
    // todo: This excluded volume is duplicated with the pharmacophore. I think
    // todo having both here is fine for now, and we will settle out hwo the
    // todo state works organically.
    pub volume: PocketVolume,
    /// Index into the global meshes from the engine.
    /// Relative the to the base index for pockets. I.e, this value starts at 0, even though
    /// the meshes for pockets don't.
    pub mesh_i_rel: usize,
}

impl Encode for Pocket {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let mol = MoleculeSmall {
            common: self.common.clone(),
            ..Default::default()
        }
        .to_mol2();

        // write mol2 into bytes, then UTF-8
        let mut bytes = Vec::<u8>::new();
        mol.write_to(&mut bytes)
            .map_err(|e| EncodeError::OtherString(e.to_string()))?;

        let mol2_text =
            String::from_utf8(bytes).map_err(|e| EncodeError::OtherString(e.to_string()))?;

        mol2_text.encode(encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for Pocket {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mol2_text = String::decode(decoder)?;

        // If Mol2::new is fallible:
        let mol2 = Mol2::new(&mol2_text).map_err(|e| DecodeError::OtherString(e.to_string()))?;

        let mol: MoleculeSmall = mol2
            .try_into()
            .map_err(|_| DecodeError::OtherString(String::from("Problem loading from mol2")))?;

        Ok(mol.common.into())
    }
}

impl MolGenericTrait for Pocket {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::Pocket(self)
    }

    fn mol_type(&self) -> MolType {
        MolType::Lipid
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for Pocket {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let mesh_orientation = <Quaternion as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let mesh_pivot = <Vec3F32 as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let surface_mesh = <Mesh as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let volume = <PocketVolume as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;

        Ok(Self {
            common: MoleculeCommon::default(),
            mesh_orientation,
            mesh_pivot,
            surface_mesh,
            volume,
            mesh_i_rel: 0,
        })
    }
}

impl Pocket {
    /// Create a pocket from a protein. Uses a simple distance-based approach.
    pub fn new(mol: &MoleculePeptide, center: Vec3, dist_thresh: f64, ident: &str) -> Self {
        let dist_thresh_sq = dist_thresh.powi(2);

        let mol = {
            // For now at least, use use the atoms' original positions.
            let atoms: Vec<_> = mol
                .common
                .atoms
                .iter()
                .filter(|a| {
                    if a.hetero {
                        // E.g. ligands included with a mmCIF file, or water molecules.
                        return false;
                    }

                    let dist_sq = (a.posit - center).magnitude_squared();
                    dist_sq < dist_thresh_sq
                })
                .cloned()
                .collect();

            let atom_sns: HashSet<_> = atoms.iter().map(|a| a.serial_number).collect();

            let mut bonds: Vec<_> = mol
                .common
                .bonds
                .iter()
                .filter(|b| atom_sns.contains(&b.atom_0_sn) && atom_sns.contains(&b.atom_1_sn))
                .cloned()
                .collect();

            reassign_bond_indices(&mut bonds, &atoms);

            let mut mol = MoleculeCommon::new(ident.to_owned(), atoms, bonds, HashMap::new(), None);
            mol.center_local_posits_around_origin();

            mol
        };

        mol.into()
    }

    /// Can be used to quickly see visual changes, e.g. during manipulation, while being much
    /// cheaper than a full volume and mesh rebuild.
    pub fn rebuild_spheres(&mut self) {
        self.volume.rebuild_spheres(&self.common);
    }

    /// Run this, for example, after moving the molecule. Move the atoms in the same manner
    /// as with other molecule types, then run this to synchronize.
    ///
    /// Also rebuilds the mesh.
    pub fn regen_mesh_vol(&mut self, scene_meshes: &mut Vec<Mesh>, updates: &mut EngineUpdates) {
        self.volume = PocketVolume::new(&self.common);
        self.surface_mesh = make_mesh(&self.common.atoms, &self.common.atom_posits);

        let mesh_i = MESH_POCKET_START + self.mesh_i_rel;
        if mesh_i == scene_meshes.len() {
            scene_meshes.push(Mesh::default());
        } else if mesh_i > scene_meshes.len() {
            eprintln!(
                "Error: Unable to find the global mesh at {mesh_i} when assigning it for this pocket"
            );
            return;
        }

        scene_meshes[mesh_i] = self.surface_mesh.clone();

        updates.meshes = true;
        updates.entities.push_class(EntityClass::Pocket as u32);
    }

    /// Run this after a move. Resets local positions, and rebuilds everything else (volume, spheres,
    /// mesh etc, and updates the engine's meshes)
    pub fn reset_post_manip(
        &mut self,
        scene_meshes: &mut Vec<Mesh>,
        coloring: MeshColoring,
        updates: &mut EngineUpdates,
    ) {
        self.rebuild_spheres();
        self.regen_mesh_vol(scene_meshes, updates);

        let color = get_mesh_colors(&self.surface_mesh, &self.common, coloring, updates);
        apply_mesh_colors(&mut self.surface_mesh, &color);

        // We handle pushing this mesh in the regen method above.
        let mesh_i = MESH_POCKET_START + self.mesh_i_rel;
        if mesh_i >= scene_meshes.len() {
            eprintln!(
                "Error: Unable to find the global mesh at {mesh_i} when assigning it for this pocket"
            );
            return;
        }

        apply_mesh_colors(&mut scene_meshes[mesh_i], &color);
    }

    pub fn save_sdf(&self, path: &Path) -> io::Result<()> {
        MoleculeSmall {
            common: self.common.clone(),
            ..Default::default()
        }
        .to_sdf()
        .save(path)
    }

    pub fn save_mol2(&self, path: &Path) -> io::Result<()> {
        MoleculeSmall {
            common: self.common.clone(),
            ..Default::default()
        }
        .to_mol2()
        .save(path)
    }

    pub fn load(&self, path: &Path) -> io::Result<Self> {
        let extension = path
            .extension()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .to_str()
            .unwrap_or_default()
            .to_owned();

        let mut mol: MoleculeSmall = match extension.as_ref() {
            "sdf" => Sdf::load(path)?.try_into()?,
            "mol2" => Mol2::load(path)?.try_into()?,
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidFilename,
                    "Unknown file extension for pharmacophore load.",
                ));
            }
        };

        mol.common.update_path(path);

        let surface_mesh = make_mesh(&mol.common.atoms, &mol.common.atom_posits);
        let mesh_pivot = mol.common.centroid_local().into();

        Ok(Self {
            common: mol.common,
            // Note: Unused.
            mesh_orientation: Quaternion::new_identity(),
            mesh_pivot,
            surface_mesh,
            volume: Default::default(),
            mesh_i_rel: 0,
        })
    }
    // todo: mmCIF saving as well? Note that the input for these is generally mmCIF.
}

impl From<MoleculeCommon> for Pocket {
    /// Given the molecule, create the other features like volume, and surface mesh.
    /// This is an alternate constructor to `new`, which creates it from a macromolecule.
    fn from(mol: MoleculeCommon) -> Self {
        // Set up the mesh and volume after centering the local atom posits.
        let volume = PocketVolume::new(&mol);
        let surface_mesh = make_mesh(&mol.atoms, &mol.atom_posits);

        let mesh_pivot = mol.centroid_local().into();

        Self {
            common: mol,
            mesh_orientation: Quaternion::new_identity(),
            mesh_pivot,
            surface_mesh,
            volume,
            mesh_i_rel: 0,
        }
    }
}

/// For a voxel-based approach.
///
/// todo: Should this be something more like Barnes Hut, where you use various-sized  voxels?
///  e.g. big ones that take up most of the middle, then smaller ones  towards the edges.
#[derive(Clone, Debug, Default, Encode, Decode)]
pub struct PocketGrid {
    pub origin: Vec3,
    pub dims: (usize, usize, usize),
    pub resolution: f64,
    /// Linearized 3D grid. true = occupied (clash), false = free.
    /// You could use a BitVec crate to save 8x memory here, but Vec<bool> is faster/simpler.
    pub data: Vec<bool>,
}

impl PocketGrid {
    /// Create a grid from a list of spheres.
    /// This is the "expensive" step you only do once per protein pocket.
    pub fn new(spheres: &[Sphere]) -> Self {
        if spheres.is_empty() {
            return Self::default();
        }

        // 1. Calculate Bounds
        let mut min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
        let padding = 2.0; // Extra padding to catch sphere edges

        for s in spheres {
            let r = s.radius as f64;
            min = min.min(s.center - Vec3::splat(r));
            max = max.max(s.center + Vec3::splat(r));
        }

        // Pad the grid slightly so we don't panic on edges
        min -= Vec3::splat(padding);
        max += Vec3::splat(padding);

        let size = max - min;
        let dim_x = (size.x / VOXEL_RESOLUTION).ceil() as usize;
        let dim_y = (size.y / VOXEL_RESOLUTION).ceil() as usize;
        let dim_z = (size.z / VOXEL_RESOLUTION).ceil() as usize;

        let total_voxels = dim_x * dim_y * dim_z;
        let mut data = vec![false; total_voxels];

        let inv_res = 1.0 / VOXEL_RESOLUTION;

        // 2. Rasterize Spheres into Grid
        // Instead of checking every voxel against every sphere, we only check
        // voxels inside the bounding box of each sphere.
        for s in spheres {
            let r = s.radius as f64;
            let r_sq = r * r;

            // Determine sphere bounds in grid coordinates
            let s_min = (s.center - Vec3::splat(r) - min) * inv_res;
            let s_max = (s.center + Vec3::splat(r) - min) * inv_res;

            let start_x = s_min.x.floor().max(0.0) as usize;
            let end_x = s_max.x.ceil().min(dim_x as f64) as usize;

            let start_y = s_min.y.floor().max(0.0) as usize;
            let end_y = s_max.y.ceil().min(dim_y as f64) as usize;

            let start_z = s_min.z.floor().max(0.0) as usize;
            let end_z = s_max.z.ceil().min(dim_z as f64) as usize;

            for z in start_z..end_z {
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let idx = x + y * dim_x + z * dim_x * dim_y;

                        // Optimization: If already marked, skip math
                        if data[idx] {
                            continue;
                        }

                        // Calculate center of this voxel in world space
                        let voxel_pos = min
                            + Vec3::new(
                                x as f64 * VOXEL_RESOLUTION,
                                y as f64 * VOXEL_RESOLUTION,
                                z as f64 * VOXEL_RESOLUTION,
                            );

                        if (voxel_pos - s.center).magnitude_squared() <= r_sq {
                            data[idx] = true;
                        }
                    }
                }
            }
        }

        println!(
            "Generated PocketGrid: {}x{}x{} voxels.",
            dim_x, dim_y, dim_z
        );

        Self {
            origin: min,
            dims: (dim_x, dim_y, dim_z),
            resolution: VOXEL_RESOLUTION,
            data,
        }
    }

    /// O(1) check for collision.
    pub fn is_clashing(&self, point: Vec3) -> bool {
        let local = point - self.origin;

        // Negative checks (outside grid bounds = safe/empty space?)
        // Assuming the pocket is "solid" atoms and outside is "void".
        if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
            return false;
        }

        let inv_res = 1.0 / self.resolution;
        let x = (local.x * inv_res) as usize;
        let y = (local.y * inv_res) as usize;
        let z = (local.z * inv_res) as usize;

        let (dx, dy, dz) = self.dims;

        if x >= dx || y >= dy || z >= dz {
            return false;
        }

        // Linear index
        let idx = x + y * dx + z * dx * dy;

        // Use get in case logic fails, or unsafe get_unchecked for max speed
        // if you are confident in bounds checks above.
        self.data[idx]
    }
}

/// Generally the area taken up by protein atoms in the pocket + their VDW radius. The purpose
/// of this is to allow a fast comparison of if an atom is inside the pocket or in conflict with
/// the protein etc molecules that define it.
///
/// Note: We could take various approaches including voxels, spheres, gaussians etc.
/// Our goal is to represent a 3D space accurately, with fast determiniation if a point is
/// inside or outside the volume. We must be also be able to generate these easily from atom coordinates
/// in a protein.
///
/// We use meshes for visualization, but not membership determination, and we don't serialize these.
/// todo: Manual encode/decode, without serializing the meshes. Generate the mesh from the primary representation
/// the first time we display it in the UI.
#[derive(Clone, Debug, Default)]
pub struct PocketVolume {
    // todo: You likely won't keep both spheres and voxels.
    // Spheres are likely more accurate, while voxels are faster for checking for overlap.
    pub spheres: Vec<Sphere>,
    pub voxel_grid: PocketGrid,
    /// Hash grid acceleration: map cell -> indices into spheres.
    /// Cell size is chosen when building from the pocket.
    pub cell_size: f32,
    pub grid: HashMap<(i32, i32, i32), Vec<u32>>,
    //
    // /// Visualization cache only (not serialized))
    // pub mesh: Option<Mesh>,
}

// Manual bincode impl to skip Mesh
impl Encode for PocketVolume {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.spheres.encode(encoder)?;
        self.cell_size.encode(encoder)?;
        self.grid.encode(encoder)?;
        Ok(())
    }
}

// Owned decode
impl<Context> Decode<Context> for PocketVolume {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let spheres = Vec::<Sphere>::decode(decoder)?;
        let voxel_grid = PocketGrid::decode(decoder)?;
        let cell_size = f32::decode(decoder)?;
        let grid = HashMap::<(i32, i32, i32), Vec<u32>>::decode(decoder)?;

        Ok(Self {
            spheres,
            voxel_grid,
            cell_size,
            grid,
            // mesh: None,
        })
    }
}

// Borrow decode (required because your parent type derives Decode and bincode will try to
// generate BorrowDecode in many cases)
impl<'de, Context> BorrowDecode<'de, Context> for PocketVolume {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let spheres = <Vec<Sphere> as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let voxel_grid = <PocketGrid as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let cell_size = <f32 as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let grid =
            <HashMap<(i32, i32, i32), Vec<u32>> as BorrowDecode<'de, Context>>::borrow_decode(
                decoder,
            )?;

        Ok(Self {
            spheres,
            voxel_grid,
            cell_size,
            grid,
            // mesh: None,
        })
    }
}

impl PocketVolume {
    /// atoms_pocket is just from the atoms in the vicinity of the pocket. i.e,
    /// a subset of the protein.
    pub fn new(mol_pocket: &MoleculeCommon) -> Self {
        let mut res = Self::default();

        res.rebuild_spheres(mol_pocket);
        res.update_voxel_grid();

        println!(
            "Created {} pocket spheres from {} atoms.",
            res.spheres.len(),
            mol_pocket.atoms.len()
        );

        res
    }

    /// Separate so we can, for example, update this rapidly while manipulating a pocket.
    /// Bases it on relative atom positions. Also updates cell size and base grid.
    fn rebuild_spheres(&mut self, mol: &MoleculeCommon) {
        let mut spheres = Vec::with_capacity(mol.atoms.len());

        let mut max_r = 0.;

        for (i, a) in mol.atoms.iter().enumerate() {
            let center = mol.atom_posits[i];
            let r = a.element.vdw_radius() + PROBE_RADIUS_EXCLUDED_VOL;
            if r > max_r {
                max_r = r;
            }
            spheres.push(Sphere { center, radius: r });
        }

        // max_r is a decent default for "few spheres per cell".
        let cell_size = max_r.max(CELL_SIZE_SPHERE);

        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, s) in spheres.iter().enumerate() {
            let c = cell_of(s.center, cell_size);
            grid.entry(c).or_default().push(i as u32);
        }

        self.spheres = spheres;
        self.cell_size = cell_size;
        self.grid = grid;
    }

    /// E.g. update this after manipulation is complete.
    pub fn update_voxel_grid(&mut self) {
        self.voxel_grid = PocketGrid::new(&self.spheres);
    }

    /// Uses the voxel approach.
    pub fn inside(&self, point: Vec3) -> bool {
        self.voxel_grid.is_clashing(point)
    }

    // todo: Benchmark and parallelize this. Rayon, GPU etc.
    pub fn inside_spheres(&self, point: Vec3) -> bool {
        if self.spheres.is_empty() {
            return false;
        }

        let c = cell_of(point, self.cell_size);

        // Check this cell and neighbors (27 cells total).
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let k = (c.0 + dx, c.1 + dy, c.2 + dz);
                    let Some(ids) = self.grid.get(&k) else {
                        continue;
                    };

                    for &id in ids {
                        let s = &self.spheres[id as usize];
                        let dist_sq = (point - s.center).magnitude_squared();
                        let r = s.radius as f64;
                        if dist_sq <= r.powi(2) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Optional: positive value = how far *inside* excluded volume you are (0 = outside).
    /// Useful as a smooth-ish clash penalty ingredient.
    pub fn penetration_depth(&self, point: Vec3) -> f32 {
        if self.spheres.is_empty() {
            return 0.0;
        }

        let c = cell_of(point, self.cell_size);

        let mut best = 0.0f32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let k = (c.0 + dx, c.1 + dy, c.2 + dz);
                    let Some(ids) = self.grid.get(&k) else {
                        continue;
                    };

                    for &id in ids {
                        let s = &self.spheres[id as usize];
                        let d = (point - s.center).magnitude() as f32;
                        let pen = (s.radius - d).max(0.0);
                        if pen > best {
                            best = pen;
                        }
                    }
                }
            }
        }

        best
    }
}

fn cell_of(p: Vec3, cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / (cell_size as f64);
    (
        (p.x * inv).floor() as i32,
        (p.y * inv).floor() as i32,
        (p.z * inv).floor() as i32,
    )
}

/// We use local atom positions to make the mesh, then move the mesh position as required when
/// drawing.
// fn make_mesh(atoms: &[Atom]) -> Mesh {
fn make_mesh(atoms: &[Atom], posits: &[Vec3]) -> Mesh {
    // let atoms_for_mesh: Vec<_> = atoms
    //     .iter()
    //     .enumerate()
    //     .map(|(i, a)| (posits[i], a.element.vdw_radius()))
    //     .collect();

    let atoms_for_mesh: Vec<_> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (posits[i].into(), a.element.vdw_radius()))
        .collect();

    make_sas_mesh(&atoms_for_mesh, MESH_PROBE_RADIUS, POCKET_MESH_PRECISION)
}
