//! Measures properties of the molecules in the human body; relevant to drug development.
//!
//! Potential data sources and other resources
//! [Therapeutics Data Commons](https://tdcommons.ai/) [Arxiv from 2021](https://arxiv.org/pdf/2102.09548)
//! [TDC download without PyTDC](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2F21LKWG&utm_source=chatgpt.com)
//! [Aqueous Solubility Data Curation](https://github.com/mcsorkun/AqSolDB) [Paper, 2019](https://www.nature.com/articles/s41597-019-0151-1)
//! [PK-DB](https://pk-db.com/) [Paper, 2020](https://academic.oup.com/nar/article/49/D1/D1358/5957165?login=false)
//!
//! [TDC ADME](https://tdcommons.ai/single_pred_tasks/adme/)
//!
//! todo: [Look up QUPKAKE?](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00328)

pub mod infer;

mod solubility;
pub mod train;

// todo: Eval feature?
#[cfg(feature = "train")]
pub mod model_eval;
mod mol_gen;
#[cfg(feature = "train")]
mod train_test_split_indices;
// Pub to allow access from the training entry point.

use std::{collections::HashMap, io};

use crate::{
    mol_characterization::MolCharacterization,
    molecules::small::MoleculeSmall,
    therapeutic::infer::{Infer, infer_general},
};

/// Absorption, distribution, metabolism, and excretion (ADME) properties.
/// I believe this is broadly synonymous with Pharmacokinetics.
///
/// Ones marked "Binary" have training target data of either 0 or 1. We store as
/// floating point, for now, to assist with checking [confidence?].
#[derive(Clone, Debug, Default)]
pub struct Adme {
    // Absorption
    /// TDC.Caco2_Wang. cm/s
    pub intestinal_permeability: f32,
    /// TDC.HIA_Hou. Binary.
    pub intestinal_absorption: f32,
    /// TDC.Pgp_Broccatelli. Binary.
    pub pgp: f32,
    /// Bioavailability_Ma. Binary.
    pub oral_bioavailablity: f32,
    /// TDC.Lipophilicity_AstraZeneca. log-ratio.
    pub lipophilicity: f32,
    /// AqSolDB, or TDC.Solubility_AqSolDB. log mol/L
    /// LogS, where S is the aqueous solubility.
    pub solubility_water: f32,
    /// TDC.PAMPA_NCATS
    ///  PAMPA (parallel artificial membrane permeability assay) is a commonly employed assay
    /// to evaluate drug permeability across the cellular membrane. Binary.
    pub membrane_permeability: f32,
    /// TDC.hHydrationFreeEnergy_FreeSolv
    /// The Free Solvation Database, FreeSolv(SAMPL), provides experimental and calculated hydration
    /// free energy of small molecules in water. The calculated values are derived from alchemical
    /// free energy calculations using molecular dynamics simulations. todo: Units
    pub hydration_free_energy: f32,
    // Distribution
    /// TDC.BBB_Martins. Binary
    pub blood_brain_barrier: f32,
    /// TDC.PPBR_AZ. % binding value.
    pub plasma_protein_binding_rate: f32,
    /// Volume of Distribution at steady state.
    pub vdss: f32,
    // Metabolism
    /// CYP P450 2C19 Inhibition.
    ///  The CYP P450 genes are essential in the breakdown (metabolism) of various molecules and
    /// chemicals within cells. A drug that can inhibit these enzymes would mean poor metabolism
    /// to this drug and other drugs, which could lead to drug-drug interactions and adverse effects.
    ///
    /// CYP2C19 gene provides instructions for making an enzyme called the endoplasmic reticulum,
    /// which is involved in protein processing and transport.
    /// Binary.
    pub cyp_2c19_inhibition: f32,
    /// CYP2D6 is primarily expressed in the liver. Binary.
    pub cyp_2d6_inhibition: f32,
    // todo: More P450 inhibitions
    // Excretion.
    /// TDC.Half_Life_Obach. Todo: Units.
    pub half_life: f32,
    /// TDC.Clearance_Hepatocyte_AZ. todo: Units.
    pub clearance: f32,
}
#[derive(Clone, Debug, Default)]
pub struct Toxicity {
    /// TDC.LD50_Zhu. log(1/(mol/kg)).
    pub ld50: f32,
    /// TDC.hERG. Related to coordination of the heart's beating. Binary.
    pub ether_a_go_go: f32,
    /// TDC.AMES. Binary.
    pub mutagencity: f32,
    /// TDC.DILI. Binary.
    pub drug_induced_liver_injury: f32,
    /// TDC.Skin_Reaction. Binary.
    pub skin_reaction: f32,
    /// TDC.Carcinogens_lagunin. Binary.
    pub carcinogen: f32,
}

/// Estimates of how the molecule, in drug form, acts in the human body.
/// https://en.wikipedia.org/wiki/Pharmacokinetics
#[derive(Clone, Debug, Default)]
pub struct TherapeuticProperties {
    pub adme: Adme,
    pub toxicity: Toxicity,
    pub breakdown_products: Vec<MoleculeSmall>,
}

impl TherapeuticProperties {
    pub fn new(mol: &MoleculeSmall, models: &mut HashMap<String, Infer>) -> io::Result<Self> {
        // The target names here must match the CSV names, as downloaded from TDC.
        let adme = Adme {
            intestinal_permeability: infer_general(mol, "caco2_wang", models)?,
            intestinal_absorption: infer_general(mol, "hia_hou", models)?,
            pgp: infer_general(mol, "pgp_broccatelli", models)?,
            oral_bioavailablity: infer_general(mol, "bioavailability_ma", models)?,
            lipophilicity: infer_general(mol, "lipophilicity_astrazeneca", models)?,
            solubility_water: infer_general(mol, "solubility_aqsoldb", models)?,
            blood_brain_barrier: infer_general(mol, "bbb_martins", models)?,
            plasma_protein_binding_rate: infer_general(mol, "ppbr_az", models)?,
            membrane_permeability: infer_general(mol, "pampa_ncats", models)?,
            hydration_free_energy: infer_general(mol, "hydrationfreeenergy_freesolv", models)?,
            vdss: infer_general(mol, "vdss_lombardo", models)?,
            cyp_2c19_inhibition: infer_general(mol, "cyp2c19_veith", models)?,
            cyp_2d6_inhibition: infer_general(mol, "cyp2d6_veith", models)?,
            half_life: infer_general(mol, "clearance_hepatocyte_az", models)?,
            clearance: infer_general(mol, "half_life_obach", models)?,
        };

        let toxicity = Toxicity {
            ld50: infer_general(mol, "ld50_zhu", models)?,
            ether_a_go_go: infer_general(mol, "herg", models)?,
            mutagencity: infer_general(mol, "ames", models)?,
            drug_induced_liver_injury: infer_general(mol, "dili", models)?,
            skin_reaction: infer_general(mol, "skin_reaction", models)?,
            carcinogen: infer_general(mol, "carcinogens_lagunin", models)?,
        };

        Ok(Self {
            adme,
            toxicity,
            breakdown_products: Vec::new(),
        })
    }
}

/// PDB ID
fn get_common_drug_rel_proteins() -> Vec<String> {
    vec![]
}
