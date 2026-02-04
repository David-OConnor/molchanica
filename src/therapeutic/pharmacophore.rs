//! Used to characterize binding pockets of proteins with specific features,
//! then using these features to query databases for ligands that may fit.
//!
//! https://www.eyesopen.com/rocs

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    io,
    io::ErrorKind,
    path::Path,
};

use bincode::{
    BorrowDecode, Decode, Encode,
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
};
use bio_files::{Mol2, PharmacophoreTypeGeneric, Sdf};
use egui_file_dialog::FileDialog;
use graphics::Mesh;
use lin_alg::f64::Vec3;

use crate::{
    mol_characterization::{MolCharacterization, RingType},
    molecules::{MoleculeGeneric, MoleculePeptide, common::MoleculeCommon, small::MoleculeSmall},
    render::Color,
};

const PROBE_RADIUS_EXCLUDED_VOL: f32 = 1.4;

#[derive(Clone, Debug)]
pub struct Pocket {
    /// Contains atoms around the pocket only.
    /// todo: Should this include enough atoms to perform basic MD, or just cover the surface?
    pub mol: MoleculeCommon,
    // todo: How should we represent motion here?
    pub surface_mesh: Option<Mesh>,
    // todo: This excluded volume is duplicated with the pharmacophore. I think
    // todo having both here is fine for now, and we will settle out hwo the
    // todo state works organically.
    pub volume: PocketVolume,
}

impl Pocket {
    /// Create a pocket from a protein. Uses a simple distance-based approach.
    pub fn new(mol: MoleculePeptide, center: Vec3, dist_thresh: f64, ident: &str) -> Self {
        let dist_thresh_sq = dist_thresh.powi(2);

        // For now at least, use use the atoms' original positions.
        let atoms: Vec<_> = mol
            .common
            .atoms
            .iter()
            .filter(|a| {
                let dist_sq = (a.posit - center).magnitude_squared();
                dist_sq < dist_thresh_sq
            })
            .cloned()
            .collect();

        let atom_sns: HashSet<_> = atoms.iter().map(|a| a.serial_number).collect();

        let bonds: Vec<_> = mol
            .common
            .bonds
            .iter()
            .filter(|b| atom_sns.contains(&b.atom_0_sn) || atom_sns.contains(&b.atom_1_sn))
            .cloned()
            .collect();

        let mol = MoleculeCommon::new(ident.to_owned(), atoms, bonds, HashMap::new(), None);

        Self {
            mol,
            surface_mesh: None,
            volume: Default::default(),
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
        Ok(Self {
            mol: mol.common,
            surface_mesh: None,
            volume: Default::default(),
        })
    }
    // todo: mmCIF saving as well? Note that the input for these is generally mmCIF.
}

/// Hmm: https://www.youtube.com/watch?v=Z42UiJCRDYE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode, Hash, PartialOrd, Ord)] // Default is for the UI
pub enum PharmacophoreFeatType {
    Hydrophobic,
    Hydrophilic,
    Aromatic,
    #[default]
    Acceptor,
    AcceptorProjected,
    Donor,
    Cation,
    Anion,
    /// Directional.
    DonorProjected,
    // HeavyAtom,
    // Ring,
    // RingNonPlanar,
    // RingPlanarProjected,
    // Purine,
    // Pyrimidine,
    // Adenine,
    // Cytosine,
    // Guanine,
    // Thymine,
    // Uracil,
    // Deoxyribose,
    // Ribose,
    // ExitVector,
    // Halogen,
    // Bromine,
}

impl PharmacophoreFeatType {
    pub fn all() -> Vec<Self> {
        use PharmacophoreFeatType::*;
        vec![
            Hydrophobic,
            Hydrophilic,
            /// Has significance in Pi bonding, e.g. stacked rings.
            Aromatic,
            Acceptor,
            AcceptorProjected, // Directional
            Donor,
            Cation,
            Anion,
            DonorProjected, // Directional
            // HeavyAtom,
            // PlanarAtom,
            // NCNPlus,
            // Ring,
            // RingNonPlanar,
            // RingPlanarProjected,
            // Purine,
            // Pyrimidine,
            // Adenine,
            // Cytosine,
            // Guanine,
            // Thymine,
            // Uracil,
            // Deoxyribose,
            // Ribose,
            // ExitVector,
            // Halogen,
            // PiRingCenter,
            // AromaticOrPiRingNormal
            // MetalLigator,
            // MetalLigatorProjection
            // Link source
            // Link projection
            // VolumeConstraint,

        ]
    }

    /// List likely locations in a molecule to place this feature type.
    /// We can make this return more info than posit if required. We use this, for example,
    /// for display in the UI, allowing a user to select them.
    pub fn hint_sites(self, char: &MolCharacterization, atom_posits: &[Vec3]) -> Vec<Vec3> {
        use PharmacophoreFeatType::*;
        match self {
            Aromatic => {
                let mut sites = Vec::new();
                for ring in char
                    .rings
                    .iter()
                    .filter(|r| r.ring_type == RingType::Aromatic)
                {
                    sites.push(ring.center(atom_posits));
                }

                sites
            }
            Donor => {
                let mut sites = Vec::new();
                for v in &char.h_bond_donor {
                    sites.push(atom_posits[*v]);
                }

                sites
            }
            Acceptor => {
                let mut sites = Vec::new();
                for v in &char.h_bond_acceptor {
                    sites.push(atom_posits[*v]);
                }

                sites
            }
            Hydrophobic => {
                let mut sites = Vec::new();
                for v in &char.hydrophobic_carbon {
                    sites.push(atom_posits[*v]);
                }

                sites
            }
            _ => Vec::new(),
        }
    }

    pub fn disp_radius(self) -> f32 {
        use PharmacophoreFeatType::*;
        match self {
            // Fits inside the drawn ring bonds.
            Aromatic => 1.05,
            Hydrophobic => 1.0, // todo: Likkely depends on the region.
            _ => 0.6,
        }
    }

    pub fn color(self) -> Color {
        // todo: (u8 tuple instad of f32 tuple?)
        use PharmacophoreFeatType::*;

        match self {
            Hydrophobic => (0., 0.8, 0.),
            Hydrophilic => (1., 1., 1.),
            Aromatic => (0.4, 0.1, 0.8), // todo: Green?
            Acceptor => (1., 0.5, 0.2),
            // AcceptorProjected => (0., 1., 0.),
            Donor => (1., 1., 1.), // todo: Red?
            // DonorProjected => (1., 1., 1.),
            _ => (1., 0., 0.), // todo
        }
    }

    pub fn to_generic(self) -> PharmacophoreTypeGeneric {
        use PharmacophoreFeatType::*;
        match self {
            Hydrophobic => PharmacophoreTypeGeneric::Acceptor,
            Hydrophilic => PharmacophoreTypeGeneric::Hydrophobic,
            Aromatic => PharmacophoreTypeGeneric::Aromatic,
            Acceptor | AcceptorProjected => PharmacophoreTypeGeneric::Acceptor,
            Donor | DonorProjected => PharmacophoreTypeGeneric::Donor,
            Cation => PharmacophoreTypeGeneric::Cation,
            Anion => PharmacophoreTypeGeneric::Anion,
        }
    }
}

impl Display for PharmacophoreFeatType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // todo: Placeholder
        write!(f, "{:?}", self)
    }
}

impl From<PharmacophoreTypeGeneric> for PharmacophoreFeatType {
    fn from(value: PharmacophoreTypeGeneric) -> Self {
        use PharmacophoreFeatType::*;
        match &value {
            PharmacophoreTypeGeneric::Acceptor => Acceptor,
            PharmacophoreTypeGeneric::Donor => Donor,
            PharmacophoreTypeGeneric::Cation => Cation,
            PharmacophoreTypeGeneric::Rings => Aromatic, // todo?
            PharmacophoreTypeGeneric::Hydrophobic => Hydrophobic,
            PharmacophoreTypeGeneric::Hydrophilic => Hydrophilic,
            PharmacophoreTypeGeneric::Anion => Anion,
            PharmacophoreTypeGeneric::Aromatic => Aromatic,
            PharmacophoreTypeGeneric::Other(v) => {
                eprintln!("Unknown generic Pharmacophore type: {:?}", value);
                Acceptor
            }
        }
    }
}

//
// // todo: Unused for now in favor of absolute positions.
// #[derive(Clone, PartialEq, Debug, Encode, Decode)]
// pub enum Position {
//     /// Relative to what? Atom 0 of a target ligand? A reference atom in the pocket?
//     Posit(Vec3),
//     /// Index in molecule.
//     Atom(usize),
//     Atoms(Vec<usize>),
// }
//
// impl Position {
//     /// Get the absolute position of this feature; for example if it's based on atoms.
//     pub fn absolute(&self, atom_posits: Option<&[Vec3]>) -> io::Result<Vec3> {
//         use Position::*;
//
//         match self {
//             Atom(i) => {
//                 let Some(posits) = atom_posits else {
//                     return Err(io::Error::new(
//                         ErrorKind::Other,
//                         "Missing posits for relative query posit.",
//                     ));
//                 };
//
//                 if *i > posits.len() {
//                     return Err(io::Error::new(ErrorKind::Other, "Posit out of bound."));
//                 }
//
//                 Ok(posits[*i])
//             }
//             Atoms(idxs) => {
//                 let Some(posits) = atom_posits else {
//                     return Err(io::Error::new(
//                         ErrorKind::Other,
//                         "Missing posits for relative query posit.",
//                     ));
//                 };
//
//                 let mut result = Vec3::new_zero();
//                 for i in idxs {
//                     if *i > posits.len() {
//                         return Err(io::Error::new(ErrorKind::Other, "Posit out of bound."));
//                     }
//
//                     result += posits[*i];
//                 }
//                 Ok(result / idxs.len() as f64)
//             }
//             Posit(p) => Ok(*p),
//         }
//     }
// }

/// A simple harmonic oscillator representing the pharmacophore.
#[derive(Clone, Debug, Encode, Decode)]
pub struct Oscillator {
    pub k_b: f32,
    pub max_displacement: f32,
    pub orientation: Vec3,
}

#[derive(Clone, Debug, Encode, Decode)]
pub enum Motion {
    Oscillator(Oscillator),
    /// A, C; one or more overlapping gaussians.
    Gaussian(Vec<(f32, f32)>),
}

/// Relates two features, e.g. colocated ones.
#[derive(Clone, Copy, PartialEq, Debug, Encode, Decode)]
pub enum FeatureRelation {
    And((usize, usize)),
    Or((usize, usize)),
    // Not(PharmacophoreFeatType),
}

#[derive(Clone, Debug, Encode, Decode)]
pub struct PharmacophoreFeature {
    pub feature_type: PharmacophoreFeatType,
    // pub feature_type_additional: FeatureAdditional,
    pub posit: Vec3,
    // Note: For these projections, we can't easily add them as an inner value of FeatureType,
    // without adding a way to hash and sort them for certain uses.
    pub posit_projected: Option<Vec3>,
    /// Used when associating with a specific atom.
    pub atom_i: Option<Vec<usize>>,
    pub atom_i_projected: Option<usize>,
    pub strength: f32,
    pub tolerance: f32,
    // pub radius: f32,
    pub oscillation: Option<Motion>,
    pub ui_selected: bool,
}

impl Default for PharmacophoreFeature {
    fn default() -> Self {
        Self {
            feature_type: PharmacophoreFeatType::default(),
            posit: Vec3::new_zero(),
            posit_projected: None,
            atom_i: None,
            atom_i_projected: None,
            strength: 1.0, // todo?
            tolerance: 1.0,
            // radius: 1.0,
            oscillation: None,
            ui_selected: false,
        }
    }
}

impl Display for PharmacophoreFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} : Posit: {} Str: {:.2} Tol: {:.2}",
            self.feature_type, self.posit, self.strength, self.tolerance,
        )
    }
}

impl PharmacophoreFeature {
    /// Get the absolute position from atoms, if available.
    /// If multiple atoms, e.g. a ring, get the center.
    pub fn posit_from_atoms(&self, atom_posits: &[Vec3]) -> Option<Vec3> {
        let Some(atom_i) = &self.atom_i else {
            return None;
        };

        let mut result = Vec3::new_zero();
        for i in atom_i {
            if *i >= atom_posits.len() {
                eprintln!("Error: Atom index out of bounds when getting pharmacophore posit");
                return None;
            }

            result += atom_posits[*i];
        }

        Some(result / atom_i.len() as f64)
    }
}

/// For excluded volume.
#[derive(Clone, Copy, Debug, Default, Encode, Decode)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
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

    /// Visualization cache only (not serialized))
    pub mesh: Option<Mesh>,
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
            mesh: None,
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
            mesh: None,
        })
    }
}

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
            mesh: None,
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

    /// Call this if you want to display the excluded volume.
    /// Keep it as a cache: rebuild if mesh is None, or if spheres changed.
    pub fn mesh_cached_mut(&mut self) -> &mut Option<Mesh> {
        &mut self.mesh
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

#[derive(Clone, Debug, Default, Encode, Decode)]
pub struct Pharmacophore {
    pub name: String,
    pub features: Vec<PharmacophoreFeature>,
    pub feature_relations: Vec<FeatureRelation>,
    pub excluded_volume: Option<PocketVolume>,
}

impl Pharmacophore {
    pub fn create(mols: &[MoleculeSmall]) -> Vec<Self> {
        Vec::new()
    }

    /// Return (indices passed, atom posits, score).
    pub fn filter_ligs(&self, mols: &[MoleculeSmall], thresh: f32) -> Vec<(usize, Vec<Vec3>, f32)> {
        let mut res = Vec::new();
        for (i, mol) in mols.iter().enumerate() {
            let score = self.score(mol);

            if score > thresh {
                res.push((i, vec![], score));
            }
        }

        res
    }

    pub fn score(&self, mol: &MoleculeSmall) -> f32 {
        let char = match mol.characterization.as_ref() {
            Some(c) => c,
            None => return 0.0,
        };

        // let atoms: &[Atom] = &mol.common.atoms;
        let atom_posits = &mol.common.atom_posits;

        // Candidate sites on the ligand for a given feature type.
        let ligand_sites = |ft: PharmacophoreFeatType| -> Vec<Vec3> {
            match ft {
                PharmacophoreFeatType::Hydrophilic => {
                    // Reasonable default: polar sites = donors ∪ acceptors
                    let mut v = Vec::new();
                    for &i in &char.h_bond_donor {
                        v.push(atom_posits[i]);
                    }
                    for &i in &char.h_bond_acceptor {
                        v.push(atom_posits[i]);
                    }
                    v
                }
                _ => ft.hint_sites(char, atom_posits),
            }
        };

        let mut total_strength = 0.0f32;
        let mut total = 0.0f32;

        let mut matched = 0usize;
        let mut considered = 0usize;

        for feat in &self.features {
            // let qpos = match feat.posit.absolute(Some(&mol.common.atom_posits)) {
            //     Ok(p) => p,
            //     Err(_e) => {
            //         eprintln!("Failed to get absolute posit for feature: {:?}", feat);
            //         continue;
            //     }
            // };
            let qpos = feat.posit;

            considered += 1;
            total_strength += feat.strength.max(0.0);

            let sites = ligand_sites(feat.feature_type);
            if sites.is_empty() {
                continue;
            }

            let sigma = feat.tolerance.max(1e-6) as f64;

            let mut best = 0.0f32;
            for spos in sites {
                let dist_sq = (qpos - spos).magnitude_squared();

                let s = gaussian(dist_sq, sigma);
                if s > best {
                    best = s;
                }
            }

            if best > 0.2 {
                matched += 1;
            }

            total += feat.strength.max(0.0) * best;
        }

        if total_strength <= 0.0 || considered == 0 {
            return 0.0;
        }

        let mut score = total / total_strength;

        // Optional: require some minimum match fraction to avoid “one lucky feature” passing.
        let match_frac = matched as f32 / considered as f32;
        if match_frac < 0.4 {
            score *= match_frac / 0.4;
        }

        if let Some(ev) = &self.excluded_volume {
            // Penalize if any ligand atom is inside excluded volume.
            // For screening: cheap boolean check is often enough.
            let mut inside_count = 0usize;
            for &p in atom_posits {
                if ev.inside(p) {
                    inside_count += 1;
                }
            }

            if inside_count > 0 {
                // Simple penalty: scale down score by how many atoms clash.
                // Tune this to taste; for screening you often want to be harsh.
                let frac = inside_count as f32 / atom_posits.len().max(1) as f32;
                score *= (1.0 - frac).clamp(0.0, 1.0);
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Save to disk.
    pub fn save(&self, dialog: &mut FileDialog, name_default: &str) -> io::Result<()> {
        let fname_default = {
            let ext_default = "pmp"; // A custom format

            format!("{name_default}.{ext_default}")
        };

        dialog.config_mut().default_file_name = fname_default.to_string();
        dialog.config_mut().default_file_filter = Some("PMP (Pharmacophore)".to_owned());

        dialog.save_file();

        Ok(())
    }

    /// Terse
    pub fn summary(&self) -> String {
        let mut feat_counts = HashMap::new();
        for feat in &self.features {
            *feat_counts.entry(feat.feature_type).or_insert(0) += 1;
        }

        let mut items: Vec<_> = feat_counts.into_iter().collect();
        items.sort_by(|(a_ft, _), (b_ft, _)| a_ft.cmp(b_ft));

        let mut res = String::new();
        for (ft, count) in items {
            res += &format!("{ft}: {count} ");
        }

        res
    }
}

/// Handles adding the feature, the entity etc.
pub fn add_pharmacophore(
    mol: &mut MoleculeSmall,
    feat_type: PharmacophoreFeatType,
    atom_i: usize,
) -> io::Result<()> {
    // Ideally the user clicks a ring hint etc. Workaround for now.
    let posit = if feat_type == PharmacophoreFeatType::Aromatic {
        // todo: Move this logic (if you keep it)
        // todo: DOn't unwrap

        let mut val = None;
        for ring in &mol.characterization.as_ref().unwrap().rings {
            if ring.atoms.contains(&atom_i) {
                // val = Some(&ring.atoms);
                val = Some(ring.center(&mol.common.atom_posits));
                break;
            }
        }
        match val {
            // Some(v) => Position::Atoms(v.to_owned()),
            // Some(v) => Position::Atoms(v.to_owned()),
            Some(v) => v,
            None => return Err(io::Error::new(ErrorKind::Other, "No ring found for atom.")),
            // None => Position::Atom(atom_i),
        }
    } else {
        // Position::Atom(atom_i)

        if atom_i >= mol.common.atom_posits.len() {
            return Err(io::Error::new(
                ErrorKind::Other,
                "Atom index out of bounds.",
            ));
        }
        mol.common.atom_posits[atom_i]
    };

    mol.pharmacophore.features.push(PharmacophoreFeature {
        feature_type: feat_type,
        posit,
        atom_i: Some(vec![atom_i]),
        ..Default::default()
    });

    Ok(())
}

fn gaussian(dist_sq: f64, sigma: f64) -> f32 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let denom = 2.0 * sigma * sigma;
    (-(dist_sq / denom)).exp() as f32
}
