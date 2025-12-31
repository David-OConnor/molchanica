//! Performs flexible alignment of two small molecules.
//!
//! One application: Recovering native ligand binding poses.
//!
//! [Wang, 2023](https://www.biorxiv.org/content/10.1101/2023.12.17.572051v2.full.pdf)
//! [Brown, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6598199/)
//! [BCL on Github](https://github.com/BCLCommons/bcl)
//!
//!
//! [Web based BCL::MolAlign](http://servers.meilerlab.org/index.php/servers/molalign)
//! This one may be useful for validating your results.
//!

use lin_alg::f64::Vec3;

use crate::molecule::MoleculeCommon;

/// For scores, a higher negative value indicates more similarity.
pub struct MolAlignment {
    pub posits: Vec<Vec3>,
    pub avg_strain_energy: f32,

    pub similarity_measure: f32,
    /// Sum of avg_strain_energy and similarity_measure.
    pub alignment_score: f32,
    /// Grades chemical and/or shape similarity. Insufficient when the molecules are of sufficiently
    /// different sizes.
    pub tanimoto_coefficient: f32,
}

impl MolAlignment {
    /// Align two molecules. This is generally for small molecules. Scores by steric hindrance, and
    /// minimizing electrostatic interactions. The molecules being aligned should generally have some
    /// shared structure but will not be the same molecule.
    pub fn create(mol_0: &MoleculeCommon, mol_1: &MoleculeCommon) -> Vec<Self> {
        // Self {
        //     posits: Vec::new(),
        //     score_steric: 0.,
        //     score_electrostatic: 0.,
        // }

        Vec::new()
    }
}
