//! Used to characterize binding pockets of proteins with specific features,
//! then using these features to query databases for ligands that may fit.

use lin_alg::f64::Vec3;

use crate::molecules::{HydrogenBond, small::MoleculeSmall};
// #[derive(Debug)]
// pub enum PharmacophoreFeature {
//     Hydrophobic(<(i8)>),
//     Hydrophilic(i8),
//     Aromatic(i8),
// }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PharmacophoreFeatureType {
    Hydrophobic,
    Hydrophilic,
    Aromatic,
}

#[derive(Debug)]
pub struct PharmacophoreFeature {
    feature_type: PharmacophoreFeatureType,
    posit: Vec3,
    strength: f32,
}

#[derive(Debug)]
pub struct Pharmacophore {
    pub pocket_vol: f32,
    pub features: Vec<PharmacophoreFeature>,
    hydrogen_bonds: Vec<HydrogenBond>,
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
