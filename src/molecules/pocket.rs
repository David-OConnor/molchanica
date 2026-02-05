//! Protein pockets. Used for pharmacophore and other screening.

use std::{
    collections::{HashMap, HashSet},
    io,
    io::ErrorKind,
    path::Path,
};

use crate::molecules::{
    Atom, MoleculePeptide,
    common::{MoleculeCommon, reassign_bond_indices},
    small::MoleculeSmall,
};
use crate::sa_surface::make_sas_mesh;
use bincode::{
    BorrowDecode, Decode, Encode,
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
};
use bio_files::{Mol2, Sdf};
use graphics::Mesh;
use lin_alg::f64::Vec3;

// A larger probe radius will make the pocket tighter and coarser. Tune this to be realistic.
// Note that this is added to VDW radius.
pub const PROBE_RADIUS_EXCLUDED_VOL: f32 = 0.8;
pub const POCKET_DIST_THRESH_DEFAULT: f64 = 11.;

// Larger values result in a smoother mesh.
pub const MESH_PROBE_RADIUS: f32 = 1.0;

// Lower is more expensive, but this pocket is relatively small compared to ones we display
// over whole proteins.
pub const POCKET_MESH_PRECISION: f32 = 0.5;

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
    pub mol: MoleculeCommon,
    pub surface_mesh: Mesh,
    // todo: This excluded volume is duplicated with the pharmacophore. I think
    // todo having both here is fine for now, and we will settle out hwo the
    // todo state works organically.
    pub volume: PocketVolume,
}

impl Pocket {
    /// Create a pocket from a protein. Uses a simple distance-based approach.
    pub fn new(mol: &MoleculePeptide, center: Vec3, dist_thresh: f64, ident: &str) -> Self {
        let dist_thresh_sq = dist_thresh.powi(2);

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

        let surface_mesh = make_mesh(&atoms);

        let mol = MoleculeCommon::new(ident.to_owned(), atoms, bonds, HashMap::new(), None);
        let volume = PocketVolume::new(&mol);

        Self {
            mol,
            surface_mesh,
            volume,
        }
    }

    pub fn save_sdf(&self, path: &Path) -> io::Result<()> {
        MoleculeSmall {
            common: self.mol.clone(),
            ..Default::default()
        }
        .to_sdf()
        .save(path)
    }

    pub fn save_mol2(&self, path: &Path) -> io::Result<()> {
        MoleculeSmall {
            common: self.mol.clone(),
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

        let surface_mesh = make_mesh(&mol.common.atoms);

        Ok(Self {
            mol: mol.common,
            surface_mesh,
            volume: Default::default(),
        })
    }
    // todo: mmCIF saving as well? Note that the input for these is generally mmCIF.
}

/// Generally the area taken up by protein atoms in the pocket + their VDW radius.
///
/// Note: We could take various approaches including voxels, spheres,  gaussians etc.
/// Our goal is to represent a 3D space accurately, with fast determiniation if a point is
/// inside or outside the volume. We must be also be able to generate these easily from atom coordinates
/// in a protein.
///
/// We use meshes for visualization, but not membership determination, and we don't serialize these.
/// todo: Manual encode/decode, without serializing the meshes. Generate the mesh from the primary representation
/// the first time we display it in the UI.
#[derive(Clone, Debug, Default)]
pub struct PocketVolume {
    pub spheres: Vec<Sphere>,
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
        let cell_size = f32::decode(decoder)?;
        let grid = HashMap::<(i32, i32, i32), Vec<u32>>::decode(decoder)?;

        Ok(Self {
            spheres,
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
        let cell_size = <f32 as BorrowDecode<'de, Context>>::borrow_decode(decoder)?;
        let grid =
            <HashMap<(i32, i32, i32), Vec<u32>> as BorrowDecode<'de, Context>>::borrow_decode(
                decoder,
            )?;

        Ok(Self {
            spheres,
            cell_size,
            grid,
            // mesh: None,
        })
    }
}

// todo: New pocket module
impl PocketVolume {
    /// atoms_pocket is just from the atoms in the vicinity of the pocket. i.e,
    /// a subset of the protein.
    pub fn new(atoms_pocket: &MoleculeCommon) -> Self {
        let mut spheres = Vec::with_capacity(atoms_pocket.atoms.len());
        let mut max_r = 0.;

        for (i, a) in atoms_pocket.atoms.iter().enumerate() {
            let center = atoms_pocket.atom_posits[i];
            let r = a.element.vdw_radius() + PROBE_RADIUS_EXCLUDED_VOL;
            if r > max_r {
                max_r = r;
            }
            spheres.push(Sphere { center, radius: r });
        }

        // Conservative: smaller cells reduce false candidates; bigger cells reduce grid size.
        // max_r is a decent default for "few spheres per cell".
        let cell_size = max_r.max(1.0);

        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, s) in spheres.iter().enumerate() {
            let c = cell_of(s.center, cell_size);
            grid.entry(c).or_default().push(i as u32);
        }

        Self {
            spheres,
            cell_size,
            grid,
            // mesh: None,
        }
    }

    pub fn inside(&self, point: Vec3) -> bool {
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

fn make_mesh(atoms: &[Atom]) -> Mesh {
    let atoms_for_mesh: Vec<(_)> = atoms
        .iter()
        .map(|a| (a.posit.into(), a.element.vdw_radius()))
        .collect();

    make_sas_mesh(&atoms_for_mesh, MESH_PROBE_RADIUS, POCKET_MESH_PRECISION)
}
