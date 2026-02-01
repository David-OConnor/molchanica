//! Used to characterize binding pockets of proteins with specific features,
//! then using these features to query databases for ligands that may fit.
//!
//! https://www.eyesopen.com/rocs

use crate::therapeutic::pharmacophore;
use crate::{
    mol_characterization::{MolCharacterization, RingType},
    molecules::{Atom, common::MoleculeCommon, small::MoleculeSmall},
    render::Color,
};
use bincode::{Decode, Encode};
use egui_file_dialog::FileDialog;
use lin_alg::f64::Vec3;
use std::fmt::Display;
use std::io;

#[derive(Clone, Debug)]
pub struct Pocket {
    /// Contains atoms around the pocket only.
    pub mol: MoleculeCommon,
    // todo: How should we represent motion here?
}

/// Hmm: https://www.youtube.com/watch?v=Z42UiJCRDYE
#[derive(Debug, Clone, Copy, PartialEq, Default, Encode, Decode)] // Default is for the UI
pub enum PharmacophoreFeatureType {
    Hydrophobic,
    Hydrophilic,
    Aromatic,
    #[default]
    Acceptor,
    // AcceptorProjected,
    Donor,
    // DonorProjected,
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

impl PharmacophoreFeatureType {
    pub fn all() -> Vec<Self> {
        use PharmacophoreFeatureType::*;
        vec![
            Hydrophobic,
            Hydrophilic,
            /// Has significance in Pi bonding, e.g. stacked rings.
            Aromatic,
            Acceptor,
            // AcceptorProjected,
            Donor,
            // DonorProjected,
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
        ]
    }

    /// List likely locations in a molecule to place this feature type.
    /// We can make this return more info than posit if required.
    pub fn hint_sites(self, char: &MolCharacterization, atoms: &[Atom]) -> Vec<Vec3> {
        use PharmacophoreFeatureType::*;
        match self {
            Aromatic => {
                let mut sites = Vec::new();
                for ring in char
                    .rings
                    .iter()
                    .filter(|r| r.ring_type == RingType::Aromatic)
                {
                    sites.push(ring.center(atoms));
                }

                sites
            }
            Donor => {
                let mut sites = Vec::new();
                for v in &char.h_bond_donor {
                    sites.push(atoms[*v].posit);
                }

                sites
            }
            Acceptor => {
                let mut sites = Vec::new();
                for v in &char.h_bond_acceptor {
                    sites.push(atoms[*v].posit);
                }

                sites
            }
            Hydrophobic => {
                let mut sites = Vec::new();
                for v in &char.hydrophobic_carbon {
                    sites.push(atoms[*v].posit);
                }

                sites
            }
            _ => Vec::new(),
        }
    }

    pub fn disp_radius(self) -> f32 {
        use PharmacophoreFeatureType::*;
        match self {
            Aromatic => 1.5,    // I.e. encompassing the ring visually.
            Hydrophobic => 1.0, // todo: Likkely depends on the region.
            _ => 0.6,
        }
    }
}

impl Display for PharmacophoreFeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // todo: Placeholder
        write!(f, "{:?}", self)
    }
}

impl PharmacophoreFeatureType {
    pub fn color(self) -> Color {
        // todo: (u8 tuple instad of f32 tuple?)
        use PharmacophoreFeatureType::*;

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
}

#[derive(Clone, PartialEq, Debug, Encode, Decode)]
pub enum Position {
    /// Relative to what? Atom 0 of a target ligand? A reference atom in the pocket?
    Posit(Vec3),
    /// Index in molecule.
    Atom(usize),
    Atoms(Vec<usize>),
}

impl Position {
    pub fn instantaneous(&self, mol: &MoleculeCommon) -> Vec3 {
        use Position::*;
        match self {
            Atom(i) => mol.atom_posits[*i],
            Atoms(idxs) => {
                let mut result = Vec3::new_zero();
                for i in idxs {
                    result += mol.atom_posits[*i];
                }
                result / idxs.len() as f64
            }
            Posit(p) => *p,
        }
    }
}

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

#[derive(Clone, Debug, Encode, Decode)]
pub struct PharmacophoreFeature {
    pub feature_type: PharmacophoreFeatureType,
    pub posit: Position,
    pub strength: f32,
    pub tolerance: f32,
    // pub radius: f32,
    pub oscillation: Option<Motion>,
}

impl Default for PharmacophoreFeature {
    fn default() -> Self {
        Self {
            feature_type: PharmacophoreFeatureType::default(),
            posit: Position::Atom(0), // todo?
            strength: 1.0,            // todo?
            tolerance: 1.0,
            // radius: 1.0,
            oscillation: None,
        }
    }
}

impl Display for PharmacophoreFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} : Posit: {:?} Str: {:.2} Tol: {:.2}",
            self.feature_type, self.posit, self.strength, self.tolerance,
        )
    }
}

#[derive(Clone, Debug, Default, Encode, Decode)]
pub struct Pharmacophore {
    pub pocket_vol: f32,
    pub features: Vec<PharmacophoreFeature>,
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
        0.0
    }

    /// Save to disk.
    pub fn save(&self, dialog: &mut FileDialog, name_default: &str) -> io::Result<()> {
        let fname_default = {
            let ext_default = "pmp"; // A custom format

            format!("{name_default}.{ext_default}")
        };

        dialog.config_mut().default_file_name = fname_default.to_string();
        dialog.config_mut().default_file_filter = Some("PMP (Phormacophore)".to_owned());

        dialog.save_file();

        Ok(())
    }
}

/// Handles adding the feature, the entity etc.
pub fn add_pharmacophore(
    mol: &mut MoleculeSmall,
    feat_type: PharmacophoreFeatureType,
    atom_i: usize,
) {
    // Ideally the user clicks a ring hint etc. Workaround for now.
    let posit = if feat_type == PharmacophoreFeatureType::Aromatic {
        // todo: Move this logic (if you keep it)
        // todo: DOn't unwrap

        let mut val = None;
        for ring in &mol.characterization.as_ref().unwrap().rings {
            if ring.atoms.contains(&atom_i) {
                val = Some(&ring.atoms);
                break;
            }
        }
        match val {
            Some(v) => Position::Atoms(v.to_owned()),
            None => Position::Atom(atom_i),
        }
    } else {
        Position::Atom(atom_i)
    };

    mol.pharmacophore.features.push(PharmacophoreFeature {
        feature_type: feat_type,
        posit,
        ..Default::default()
    });
}
