//! Used to characterize binding pockets of proteins with specific features,
//! then using these features to query databases for ligands that may fit.

use lin_alg::f64::Vec3;
use std::fmt::Display;

use crate::molecules::small::MoleculeSmall;
// #[derive(Debug)]
// pub enum PharmacophoreFeature {
//     Hydrophobic(<(i8)>),
//     Hydrophilic(i8),
//     Aromatic(i8),
// }

/// Hmm: https://www.youtube.com/watch?v=Z42UiJCRDYE
#[derive(Debug, Clone, Copy, PartialEq, Default)] // Default is for the UI
pub enum PharmacophoreFeatureType {
    Hydrophobic,
    Hydrophilic,
    Aromatic,
    #[default]
    Acceptor,
    AcceptorProjected,
    Donor,
    DonorProjected,
    HeavyAtom,
    Hydrophobe,
    Ring,
    RingNonPlanar,
    RingPlanarProjected,
    Purine,
    Pyrimidine,
    Adenine,
    Cytosine,
    Guanine,
    Thymine,
    Uracil,
    Deoxyribose,
    Ribose,
    ExitVector,
    Halogen,
    Bromine,
}

impl PharmacophoreFeatureType {
    pub fn all() -> Vec<Self> {
        use PharmacophoreFeatureType::*;
        vec![
            Hydrophobic,
            Hydrophilic,
            Aromatic,
            Acceptor,
            AcceptorProjected,
            Donor,
            DonorProjected,
            HeavyAtom,
            Hydrophobe,
            Ring,
            RingNonPlanar,
            RingPlanarProjected,
            Purine,
            Pyrimidine,
            Adenine,
            Cytosine,
            Guanine,
            Thymine,
            Uracil,
            Deoxyribose,
            Ribose,
            ExitVector,
            Halogen,
        ]
    }
}

impl Display for PharmacophoreFeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // todo: Placeholder
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Position {
    /// Relative to what? Atom 0?
    Posit(Vec3),
    /// Index in molecule.
    Atom(usize),
}

#[derive(Clone, Debug)]
pub struct PharmacophoreFeature {
    pub feature_type: PharmacophoreFeatureType,
    pub posit: Position,
    pub strength: f32,
    pub tolerance: f32, // Default 1.0
    pub radius: f32,    // Default 1.0
}

impl Default for PharmacophoreFeature {
    fn default() -> Self {
        Self {
            feature_type: PharmacophoreFeatureType::default(),
            posit: Position::Atom(0), // todo?
            strength: 1.0,            // todo?
            tolerance: 1.0,
            radius: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Pharmacophore {
    pub pocket_vol: f32,
    pub features: Vec<PharmacophoreFeature>,
}

impl Pharmacophore {
    pub fn create(mols: &[MoleculeSmall]) -> Vec<Self> {
        Vec::new()
    }

    /// Return (indices passed, atom posits, score).
    pub fn filter_ligs(mols: &[MoleculeSmall]) -> Vec<(usize, Vec<Vec3>, f32)> {
        Vec::new()
    }
}
