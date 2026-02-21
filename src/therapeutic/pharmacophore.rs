//! Used to characterize binding pockets of proteins with specific features,
//! then using these features to query databases for ligands that may fit.
//!
//! https://www.eyesopen.com/rocs

use std::{
    collections::HashMap,
    fmt::Display,
    io,
    io::ErrorKind,
    path::Path,
    sync::{mpsc, mpsc::Receiver},
    thread,
    time::Instant,
};

use bincode::{Decode, Encode, de::Decoder};
use bio_files::PharmacophoreTypeGeneric;
use egui_file_dialog::FileDialog;
use lin_alg::f64::Vec3;
use rayon::prelude::*;

use crate::{
    copy_le,
    mol_characterization::{MolCharacterization, RingType},
    mol_screening,
    molecules::{pocket::Pocket, small::MoleculeSmall},
    parse_le,
    render::Color,
    therapeutic::pharmacophore::PharmacophoreFeatType::{
        Acceptor, AcceptorProjected, Donor, DonorProjected, Hydrophilic, Hydrophobic,
    },
};
// #[derive(Clone, Debug)]
// pub struct PocketBinding {
//     /// Indices of these molecules.
//     pub pocket: usize,
//     pub ligand: usize,
//     // pub pocket: Pocket,
//     // pub ligand: MoleculeSmall
//     pub hydrogen_bonds: Vec<HydrogenBondTwoMols>,
// }

#[derive(Clone, Debug, Default)]
pub struct PharmacophoreState {
    pub screening_results: Vec<PhScreeningScore>,
    pub screening_in_progress: bool,
    pub ph_for_screening: Option<usize>,
}

pub const PHARMACOPHORE_SCREENING_THRESH_DEFAULT: f32 = 0.6;

pub type PhScreeningScore = (usize, String, Vec<Vec3>, f32);

/// Hmm: https://www.youtube.com/watch?v=Z42UiJCRDYE
/// The u8 rep is for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, PartialOrd, Ord)] // Default is for the UI
#[repr(u8)]
pub enum PharmacophoreFeatType {
    Hydrophobic = 0,
    Hydrophilic = 1,
    Aromatic = 3,
    #[default]
    Acceptor = 4,
    AcceptorProjected = 5,
    Donor = 6,
    Cation = 7,
    Anion = 8,
    /// Directional.
    DonorProjected = 9,
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

    # todo: Use TryFromPrimitive
    pub fn from_u8(v: u8) -> Option<Self> {
        use PharmacophoreFeatType::*;

        Some(match v {
            0 => Hydrophobic,
            1 => Hydrophilic,
            3 => Aromatic,
            4 => Acceptor,
            5 => AcceptorProjected,
            6 => Donor,
            7 => Cation,
            8 => Anion,
            9 => DonorProjected,
            _ => return None,
        })
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
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FeatureRelation {
    And((usize, usize)),
    Or((usize, usize)),
    // Not(PharmacophoreFeatType),
}

impl FeatureRelation {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = vec![0; 9];

        match self {
            Self::And((v0, v1)) => {
                res[0] = 0;
                copy_le!(res, (*v0 as u32), 1..5);
                copy_le!(res, (*v1 as u32), 5..9);
            }
            Self::Or((v0, v1)) => {
                res[0] = 1;
                copy_le!(res, (*v0 as u32), 1..5);
                copy_le!(res, (*v1 as u32), 5..9);
            }
        }

        res
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut res = vec![0; 9];

        let v0 = parse_le!(bytes, u32, 1..5) as usize;
        let v1 = parse_le!(bytes, u32, 5..9) as usize;

        match res[0] {
            0 => Self::And((v0, v1)),
            1 => Self::And((v0, v1)),
            _ => {
                eprintln!("Error parsing feat relation");
                Self::Or((v0, v1))
            } //
        }
    }
}

#[derive(Clone, Debug)]
pub struct PharmacophoreFeature {
    pub feature_type: PharmacophoreFeatType,
    // pub feature_type_additional: FeatureAdditional,
    pub posit: Vec3,
    // Note: For these projections, we can't easily add them as an inner value of FeatureType,
    // without adding a way to hash and sort them for certain uses.
    pub posit_projected: Option<Vec3>,
    /// Used when associating with a specific atom and molecule.
    pub atom_i: Vec<usize>,
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
            atom_i: Vec::new(),
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
            "{}:  Str: {:.2} Tol: {:.2}",
            self.feature_type, self.strength, self.tolerance,
        )
    }
}

impl PharmacophoreFeature {
    pub fn to_bytes(&self) -> Vec<u8> {
        let atom_len = self.atom_i.len();
        assert!(
            atom_len <= u8::MAX as usize,
            "atom_i too long to serialize as u8"
        );

        let atom_len = self.atom_i.len();
        assert!(
            atom_len <= u8::MAX as usize,
            "atom_i too long to serialize as u8"
        );

        // 1 + 24 + 1 + 4*atom_len + 4 + 4
        let total_size = 34 + 4 * atom_len;
        let mut result = vec![0; total_size];
        let mut i = 0usize;

        result[i] = self.feature_type as u8;
        i += 1;

        result[i..i + 24].copy_from_slice(&self.posit.to_le_bytes());
        i += 24;

        // todo: posit projected field?
        result[24] = self.atom_i.len() as u8;
        i += 1;

        for atom_i in &self.atom_i {
            copy_le!(result, *atom_i as u32, i..i + 4);
            i += 4;
        }

        // todo: atom_i projected field?

        copy_le!(result, self.strength, i..i + 4);
        i += 4;

        copy_le!(result, self.tolerance, i..i + 4);
        i += 4;

        // todo: Oscillation field?
        // ui_selected is not serialized.

        result
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut i = 0usize;

        assert!(bytes.len() >= 1 + 24 + 1 + 4 + 4, "bytes too short");

        let feature_type = PharmacophoreFeatType::from_u8(bytes[i]).unwrap_or_default();
        i += 1;

        let posit_bytes: [u8; 24] = bytes[i..i + 24].try_into().unwrap();
        let posit = Vec3::from_le_bytes(&posit_bytes);
        i += 24;

        let atom_len = bytes[i] as usize;
        i += 1;

        let needed = 1 + 24 + 1 + 4 * atom_len + 4 + 4;
        assert!(
            bytes.len() >= needed,
            "bytes too short for atom_i_len={atom_len}"
        );

        let mut atom_i = Vec::with_capacity(atom_len);
        for _ in 0..atom_len {
            let v = parse_le!(bytes, u32, i..i + 4);
            atom_i.push(v as usize);
            i += 4;
        }

        let strength = parse_le!(bytes, f32, i..i + 4);
        i += 4;

        let tolerance = parse_le!(bytes, f32, i..i + 4);
        // i += 4;

        Self {
            feature_type,
            posit,
            posit_projected: None,
            atom_i,
            atom_i_projected: None,
            strength,
            tolerance,
            oscillation: None,
            ui_selected: false,
        }
    }

    /// Get the absolute position from atoms, if available.
    /// If multiple atoms, e.g. a ring, get the center.
    pub fn posit_from_atoms(&self, atom_posits: &[Vec3]) -> Option<Vec3> {
        if self.atom_i.is_empty() {
            return None;
        };

        let mut result = Vec3::new_zero();
        for i in &self.atom_i {
            if *i >= atom_posits.len() {
                eprintln!("Error: Atom index out of bounds when getting pharmacophore posit");
                return None;
            }

            result += atom_posits[*i];
        }

        Some(result / self.atom_i.len() as f64)
    }
}

/// We don't have a Ligand field, as this pharmacophore may exist *as part of the ligand*.
#[derive(Clone, Debug, Default)]
pub struct Pharmacophore {
    pub name: String,
    /// Used for pairing with an open ligand.
    pub mol_ident: String,
    pub features: Vec<PharmacophoreFeature>,
    pub feature_relations: Vec<FeatureRelation>,
    // pub excluded_volume: Option<PocketVolume>,
    /// We mainly operate on the pocket's excluded volume, but associate with the whole pocket
    /// as its mesh is useful for visualzation, and atoms/bonds useful for moving and computing
    /// Hydrogen bonds with the ligand in the pharmacophore.
    pub pocket: Option<Pocket>,
}

impl Pharmacophore {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::new();
        // todo temp
        res
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::default()
    }

    /// Create a pharmacophore from all candidate sites in a molecule — one `PharmacophoreFeature`
    /// per site, not per type. For example, a molecule with two H-bond donors produces two
    /// Donor features. This is designed for use in the spatial ML pipeline rather than for
    /// screening queries (which use a curated subset of features).
    pub fn new_all_candidates(mol: &MoleculeSmall) -> Self {
        use PharmacophoreFeatType::*;

        let Some(char) = mol.characterization.as_ref() else {
            return Self::default();
        };

        let atom_posits = &mol.common.atom_posits;
        let mut features = Vec::new();

        // H-bond donors: atom position is the heavy atom bearing the H.
        for &i in &char.h_bond_donor {
            if i >= atom_posits.len() {
                continue;
            }
            features.push(PharmacophoreFeature {
                feature_type: Donor,
                posit: atom_posits[i],
                atom_i: vec![i],
                tolerance: 1.0,
                strength: 1.0,
                ..Default::default()
            });
        }

        // H-bond acceptors: atom position is the lone-pair-bearing heavy atom.
        for &i in &char.h_bond_acceptor {
            if i >= atom_posits.len() {
                continue;
            }
            features.push(PharmacophoreFeature {
                feature_type: Acceptor,
                posit: atom_posits[i],
                atom_i: vec![i],
                tolerance: 1.0,
                strength: 1.0,
                ..Default::default()
            });
        }

        // Cations: protonatable amines (consistent with how score() identifies cation sites).
        for &i in &char.amines {
            if i >= atom_posits.len() {
                continue;
            }
            features.push(PharmacophoreFeature {
                feature_type: Cation,
                posit: atom_posits[i],
                atom_i: vec![i],
                tolerance: 1.5,
                strength: 1.0,
                ..Default::default()
            });
        }

        // Anions: carboxylate oxygens (consistent with how score() identifies anion sites).
        for &i in &char.carboxylate {
            if i >= atom_posits.len() {
                continue;
            }
            features.push(PharmacophoreFeature {
                feature_type: Anion,
                posit: atom_posits[i],
                atom_i: vec![i],
                tolerance: 1.5,
                strength: 1.0,
                ..Default::default()
            });
        }

        // Aromatic rings: centroid position, all ring atoms stored in atom_i,
        // ring plane normal stored in oscillation so directional scoring in score() works.
        for ring in char
            .rings
            .iter()
            .filter(|r| r.ring_type == RingType::Aromatic)
        {
            // QC: all ring atom indices must be in bounds.
            if ring.atoms.iter().any(|&a| a >= atom_posits.len()) {
                eprintln!("Warning: aromatic ring has out-of-bounds atom index; skipping feature.");
                continue;
            }
            features.push(PharmacophoreFeature {
                feature_type: Aromatic,
                posit: ring.center(atom_posits),
                atom_i: ring.atoms.clone(),
                tolerance: 1.5,
                strength: 1.0,
                // Store ring normal so score() can apply directional modulation.
                oscillation: Some(Motion::Oscillator(Oscillator {
                    k_b: 0.0,
                    max_displacement: 0.0,
                    orientation: ring.plane_norm,
                })),
                ..Default::default()
            });
        }

        // Hydrophobic carbons.
        for &i in &char.hydrophobic_carbon {
            if i >= atom_posits.len() {
                continue;
            }
            features.push(PharmacophoreFeature {
                feature_type: Hydrophobic,
                posit: atom_posits[i],
                atom_i: vec![i],
                tolerance: 1.5,
                strength: 0.8, // Slightly down-weighted vs polar features.
                ..Default::default()
            });
        }

        Self {
            name: "All sites".to_string(),
            features,
            feature_relations: Vec::new(),
            pocket: None,
        }
    }

    /// Spawn a background thread that screens all molecules in `path` against this pharmacophore,
    /// sending results incrementally through the returned channel.
    ///
    /// Files are enumerated once, then loaded and scored in successive batches so only
    /// ~[`mol_screening::MOL_CACHE_SIZE_ATOM_COUNT`] atoms worth of molecules are held in
    /// memory at any one time.  Rayon parallelises scoring within each batch.
    ///
    /// Poll the returned [`Receiver`] each frame (see `threads::handle_thread_rx`).  Results
    /// arrive as `Vec<PhScreeningScore>` messages — one message per completed batch.  When the
    /// thread is done the channel closes and the receiver returns `Disconnected`.
    pub fn screen_ligs(
        &self,
        path: &Path,
        thresh: f32,
        ph_screening_in_progress: &mut bool,
    ) -> Receiver<Vec<PhScreeningScore>> {
        println!("Pharmacophore screening started");

        // Clone so the thread owns the pharmacophore, and convert the borrowed path to an
        // owned PathBuf — both are required for `thread::spawn`'s `'static` bound.
        let pharmacophore = self.clone();
        let path = path.to_path_buf();

        *ph_screening_in_progress = true;

        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let files = match mol_screening::collect_mol_files(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Error collecting molecule files from {path:?}: {e}");
                    return;
                }
            };

            if files.is_empty() {
                println!("No molecule files found in {path:?}");
                return;
            }

            let total_files = files.len();
            let mut file_offset = 0; // Files consumed (loaded or skipped).
            let mut mols_screened = 0; // Molecules scored so far.

            loop {
                let remaining = &files[file_offset..];
                if remaining.is_empty() {
                    break;
                }

                let (mols, files_consumed) = match mol_screening::load_mol_batch(remaining) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Error loading molecule batch: {e}");
                        break;
                    }
                };

                if mols.is_empty() {
                    break;
                }

                let batch_mol_offset = mols_screened;
                mols_screened += mols.len();
                file_offset += files_consumed;

                // Score in parallel within the batch — rayon handles the parallelism, so no
                // inner thread::spawn needed here.
                let batch_results: Vec<_> = mols
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, mol)| {
                        let score = pharmacophore.score(mol);
                        if score < thresh {
                            None
                        } else {
                            Some((
                                batch_mol_offset + i,
                                mol.common.ident.clone(),
                                vec![],
                                score,
                            ))
                        }
                    })
                    .collect();

                println!(
                    "Screening progress: {mols_screened} mols scored \
                     ({file_offset}/{total_files} files), {} passed this batch",
                    batch_results.len(),
                );

                // If the receiver was dropped (UI closed etc.), stop early.
                if tx.send(batch_results).is_err() {
                    break;
                }
            }

            println!("Pharmacophore screening complete.");
        });

        rx
    }

    pub fn score(&self, mol: &MoleculeSmall) -> f32 {
        let char = match mol.characterization.as_ref() {
            Some(c) => c,
            None => return 0.0,
        };

        if self.features.is_empty() {
            return 0.0;
        }

        let atoms = &mol.common.atoms;
        let atom_posits = &mol.common.atom_posits;
        let adj = &mol.common.adjacency_list;

        if atom_posits.is_empty() {
            return 0.0;
        }

        // H-bond donor direction: heavy atom toward attached H.
        let donor_dir = |i: usize| -> Option<Vec3> {
            if i >= adj.len() {
                return None;
            }
            for &j in &adj[i] {
                if j < atoms.len() && atoms[j].element == na_seq::Element::Hydrogen {
                    let d = atom_posits[j] - atom_posits[i];
                    let mag = d.magnitude();
                    if mag > 1e-8 {
                        return Some(d / mag);
                    }
                }
            }
            None
        };

        // H-bond acceptor direction: away from heavy-atom neighbors (lone-pair proxy).
        let acceptor_dir = |i: usize| -> Option<Vec3> {
            if i >= adj.len() {
                return None;
            }
            let mut centroid = Vec3::new_zero();
            let mut count = 0usize;
            for &j in &adj[i] {
                if j < atoms.len() && atoms[j].element != na_seq::Element::Hydrogen {
                    centroid += atom_posits[j];
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let c = centroid / count as f64;
            let d = atom_posits[i] - c;
            let mag = d.magnitude();
            if mag > 1e-8 { Some(d / mag) } else { None }
        };

        // Ligand candidate sites per feature type.
        // Each site: (position, claim_atom_indices, claim_ring_index, direction).
        // `claim_ring_index` is set for aromatic ring sites; `claim_atoms` for atom-based sites.
        // These are used for bijective matching to prevent the same ligand site from
        // satisfying multiple pharmacophore features.
        let ligand_sites =
            |ft: PharmacophoreFeatType| -> Vec<(Vec3, Vec<usize>, Option<usize>, Option<Vec3>)> {
                use PharmacophoreFeatType::*;
                match ft {
                    Hydrophobic => char
                        .hydrophobic_carbon
                        .iter()
                        .map(|&i| (atom_posits[i], vec![i], None, None))
                        .collect(),

                    Hydrophilic => {
                        let mut sites = Vec::new();
                        let mut seen = Vec::new();
                        for &i in &char.h_bond_donor {
                            sites.push((atom_posits[i], vec![i], None, None));
                            seen.push(i);
                        }
                        for &i in &char.h_bond_acceptor {
                            if !seen.contains(&i) {
                                sites.push((atom_posits[i], vec![i], None, None));
                            }
                        }
                        sites
                    }

                    Aromatic => char
                        .rings
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.ring_type == RingType::Aromatic)
                        .map(|(ri, ring)| {
                            (
                                ring.center(atom_posits),
                                Vec::new(),
                                Some(ri),
                                Some(ring.plane_norm),
                            )
                        })
                        .collect(),

                    Acceptor | AcceptorProjected => char
                        .h_bond_acceptor
                        .iter()
                        .map(|&i| (atom_posits[i], vec![i], None, acceptor_dir(i)))
                        .collect(),

                    Donor | DonorProjected => char
                        .h_bond_donor
                        .iter()
                        .map(|&i| (atom_posits[i], vec![i], None, donor_dir(i)))
                        .collect(),

                    Cation => char
                        .amines
                        .iter()
                        .map(|&i| (atom_posits[i], vec![i], None, None))
                        .collect(),

                    Anion => char
                        .carboxylate
                        .iter()
                        .map(|&i| (atom_posits[i], vec![i], None, None))
                        .collect(),
                }
            };

        // Greedy bijective matching: process high-strength features first so the most
        // important pharmacophore constraints claim their best ligand sites before weaker ones.
        let mut feat_order: Vec<usize> = (0..self.features.len()).collect();
        feat_order.sort_by(|&a, &b| {
            self.features[b]
                .strength
                .partial_cmp(&self.features[a].strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut claimed_atoms = vec![false; atom_posits.len()];
        let mut claimed_rings = vec![false; char.rings.len()];

        let mut feat_scores = vec![0.0f32; self.features.len()];
        let mut feat_matched = vec![false; self.features.len()];

        for &fi in &feat_order {
            let feat = &self.features[fi];
            let qpos = feat.posit;
            let sigma = feat.tolerance.max(1e-6) as f64;

            let sites = ligand_sites(feat.feature_type);
            if sites.is_empty() {
                continue;
            }

            // Pharmacophore feature direction: from projected position or oscillator orientation.
            let feat_dir: Option<Vec3> = if matches!(
                feat.feature_type,
                PharmacophoreFeatType::AcceptorProjected | PharmacophoreFeatType::DonorProjected
            ) {
                feat.posit_projected
                    .map(|proj| (proj - qpos).to_normalized())
            } else if feat.feature_type == PharmacophoreFeatType::Aromatic {
                feat.oscillation.as_ref().and_then(|m| match m {
                    Motion::Oscillator(o) => Some(o.orientation.to_normalized()),
                    _ => None,
                })
            } else {
                None
            };

            let mut best_score = 0.0f32;
            let mut best_idx: Option<usize> = None;

            for (si, (spos, claim_atoms, claim_ring, site_dir)) in sites.iter().enumerate() {
                // Bijective constraint: skip already-claimed sites.
                let already = if let Some(ri) = claim_ring {
                    *ri < claimed_rings.len() && claimed_rings[*ri]
                } else {
                    claim_atoms
                        .iter()
                        .any(|&a| a < claimed_atoms.len() && claimed_atoms[a])
                };
                if already {
                    continue;
                }

                let dist_sq = (qpos - *spos).magnitude_squared();
                let mut s = gaussian(dist_sq, sigma);

                // Directional modulation for projected/aromatic features.
                if let (Some(fd), Some(sd)) = (&feat_dir, site_dir) {
                    let cos_a = if feat.feature_type == PharmacophoreFeatType::Aromatic {
                        // Aromatic ring normals are valid in either orientation.
                        fd.dot(*sd).abs()
                    } else {
                        // H-bond projected features: direction matters.
                        fd.dot(*sd).max(0.0)
                    } as f32;
                    // 70% spatial, 30% directional.
                    s *= 0.7 + 0.3 * cos_a;
                }

                if s > best_score {
                    best_score = s;
                    best_idx = Some(si);
                }
            }

            if let Some(si) = best_idx {
                feat_scores[fi] = best_score;
                feat_matched[fi] = best_score > 0.2;

                // Claim the matched site.
                let (_, ref claim_atoms, claim_ring, _) = sites[si];
                if let Some(ri) = claim_ring {
                    if ri < claimed_rings.len() {
                        claimed_rings[ri] = true;
                    }
                }
                for &a in claim_atoms {
                    if a < claimed_atoms.len() {
                        claimed_atoms[a] = true;
                    }
                }
            }
        }

        // --- Feature relations (AND / OR) ---
        let mut or_suppressed = vec![false; self.features.len()];

        for rel in &self.feature_relations {
            match rel {
                FeatureRelation::Or((a, b)) => {
                    let (a, b) = (*a, *b);
                    if a < self.features.len() && b < self.features.len() {
                        // Keep the better-scoring alternative; suppress the other from the total.
                        if feat_scores[a] >= feat_scores[b] {
                            or_suppressed[b] = true;
                        } else {
                            or_suppressed[a] = true;
                        }
                    }
                }
                FeatureRelation::And((a, b)) => {
                    let (a, b) = (*a, *b);
                    if a < self.features.len() && b < self.features.len() {
                        // Both must match; penalize both if either fails.
                        if !feat_matched[a] || !feat_matched[b] {
                            feat_scores[a] *= 0.5;
                            feat_scores[b] *= 0.5;
                        }
                    }
                }
            }
        }

        // --- Weighted aggregation ---
        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f32;
        let mut matched_count = 0usize;
        let mut considered = 0usize;

        for (fi, feat) in self.features.iter().enumerate() {
            if or_suppressed[fi] {
                continue;
            }
            let w = feat.strength.max(0.0);
            considered += 1;
            total_weight += w;
            weighted_sum += w * feat_scores[fi];
            if feat_matched[fi] {
                matched_count += 1;
            }
        }

        if total_weight <= 0.0 || considered == 0 {
            return 0.0;
        }

        let mut score = weighted_sum / total_weight;

        // Coverage penalty: require a reasonable fraction of features to match.
        // Prevents a single strong match from passing screening.
        let match_frac = matched_count as f32 / considered as f32;
        if match_frac < 0.5 {
            score *= match_frac / 0.5;
        }

        // Excluded-volume steric clash penalty.
        if let Some(pocket) = &self.pocket {
            let mut clash_count = 0usize;
            for &p in atom_posits {
                if pocket.volume.inside(p) {
                    clash_count += 1;
                }
            }
            if clash_count > 0 {
                // Harsh: 2x multiplier makes even a few clashing atoms significantly reduce the
                // score. E.g. 10% atoms clashing → score *= 0.8; 25% → score *= 0.5.
                let clash_frac = clash_count as f32 / atom_posits.len().max(1) as f32;
                score *= (1.0 - 2.0 * clash_frac).clamp(0.0, 1.0);
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Save to disk.
    pub fn save_using_dialog(&self, dialog: &mut FileDialog, name_default: &str) -> io::Result<()> {
        let fname_default = {
            let ext_default = "pmp"; // A custom format

            format!("{name_default}.{ext_default}")
        };

        dialog.config_mut().default_file_name = fname_default.to_string();
        dialog.config_mut().default_file_filter = Some("PMP (Pharmacophore)".to_owned());

        dialog.save_file();

        Ok(())
    }

    // pub fn save(&self, path: &Path) -> io::Result<()> {
    //
    //     Ok(())
    // }

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
pub fn add_pharmacophore_feat(
    mol: &mut MoleculeSmall,
    feat_type: PharmacophoreFeatType,
    atom_i: usize,
) -> io::Result<()> {
    // Ideally the user clicks a ring hint etc. Workaround for now.

    let mut indices = vec![atom_i];

    let posit = if feat_type == PharmacophoreFeatType::Aromatic {
        // todo: Move this logic (if you keep it)
        // todo: DOn't unwrap

        let mut val = None;
        for ring in &mol.characterization.as_ref().unwrap().rings {
            if ring.atoms.contains(&atom_i) {
                val = Some(ring.center(&mol.common.atom_posits));
                indices = ring.atoms.clone();

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
        if atom_i >= mol.common.atom_posits.len() {
            return Err(io::Error::other("Atom index out of bounds."));
        }
        mol.common.atom_posits[atom_i]
    };

    mol.pharmacophore.features.push(PharmacophoreFeature {
        feature_type: feat_type,
        posit,
        atom_i: indices,
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
