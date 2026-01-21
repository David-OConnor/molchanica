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
pub mod model_eval;
mod solubility;
pub mod train;
// Pub to allow access from the training entry point.

use std::{collections::HashMap, io};

use crate::{
    mol_characterization::MolCharacterization,
    molecules::small::MoleculeSmall,
    therapeutic::infer::{Infer, infer_general},
};

/// Absorption, distribution, metabolism, and excretion (ADME) properties.
/// I believe this is broadly synonymous with Pharmacokinetics.
#[derive(Clone, Debug, Default)]
pub struct Adme {
    pub breakdown_products: Vec<MoleculeSmall>,
    /// TDC.Caco2_Wang. cm/s
    pub intestinal_permeability: f32,
    /// TDC.HIA_Hou
    pub intestinal_absorption: f32,
    /// TDC.Pgp_Broccatelli
    pub pgp: f32,
    /// Bioavailability_Ma
    pub oral_bioavailablity: f32,
    /// TDC.Lipophilicity_AstraZeneca. log-ratio.
    pub lipophilicity: f32,
    /// AqSolDB, or TDC.Solubility_AqSolDB. log mol/L
    /// LogS, where S is the aqueous solubility.
    pub solubility_water: f32,
    /// TDC.BBB_Martins
    pub blood_brain_barrier: f32,
    /// TDC.PPBR_AZ. % binding value.
    pub plasma_protein_binding_rate: f32,
    // todo: More
}
#[derive(Clone, Debug, Default)]
pub struct Toxicity {
    /// TDC.LD50_Zhu. log(1/(mol/kg)).
    pub ld50: f32,
    /// TDC.hERG. Related to coordination of the heart's beating.
    pub ether_a_go_go: f32,
    /// TDC.AMES
    pub mutagencity: f32,
    /// TDC.DILI
    pub drug_induced_liver_injury: f32,
    /// TDC.Skin_Reaction
    pub skin_reaction: f32,
    /// TDC.Carcinogens_lagunin
    pub carcinogen: f32,
}

/// Estimates of how the molecule, in drug form, acts in the human body.
/// https://en.wikipedia.org/wiki/Pharmacokinetics
#[derive(Clone, Debug, Default)]
pub struct TherapeuticProperties {
    pub adme: Adme,
    pub toxicity: Toxicity,
}

impl TherapeuticProperties {
    pub fn new(mol: &MoleculeSmall, models: &mut HashMap<String, Infer>) -> io::Result<Self> {
        // The target names here must match the CSV names, as downloaded from TDC.
        let adme = Adme {
            // intestinal_permeability: infer_general(mol, "caco2_wang", models)?,
            // intestinal_absorption: infer_general(mol, "hia_hou", models)?,
            // pgp: infer_general(mol, "pgp_broccatelli", models)?,
            // oral_bioavailablity: infer_general(mol, "bioavailability_ma", models)?,
            // lipophilicity: infer_general(mol, "lipophilicity_astrazeneca", models)?,
            // solubility_water: infer_general(mol, "solubility_aqsoldb", models)?,
            blood_brain_barrier: infer_general(mol, "bbb_martins", models)?,
            // plasma_protein_binding_rate: infer_general(mol, "ppbr_az", models)?,
            ..Default::default()
        };

        let toxicity = Toxicity {
            ld50: infer_general(mol, "ld50_zhu", models)?,
            ..Default::default()
        };

        Ok(Self { adme, toxicity })
    }
}

/// PDB ID
fn get_common_drug_rel_proteins() -> Vec<String> {
    vec![]
}
