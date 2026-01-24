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

use std::{collections::HashMap, fmt::Display, io, io::ErrorKind, str::FromStr};

use serde_json::error::Category::Data;

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::infer::{Infer, infer_general},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DatasetTdc {
    Ames,
    BbbMartins,
    BioavailabilityMa,
    Caco2Wang,
    CarcinogensLagunin,
    ClearanceHepatocyteAz,
    Cyp2c19Veith,
    Cyp2d6Veith,
    Dili,
    HalfLifeObach,
    Herg,
    HiaHou,
    HydrationfreeenergyFreesolv,
    Ld50Zhu,
    LipophilicityAstrazeneca,
    PampaNcats,
    PgpBroccatelli,
    PpbrAz,
    SkinReaction,
    SolubilityAqsoldb,
    VdssLombardo,
}

impl DatasetTdc {
    pub fn name(self) -> String {
        match self {
            Self::Ames => "ames",
            Self::BbbMartins => "bbb_martins",
            Self::BioavailabilityMa => "bioavailability_ma",
            Self::Caco2Wang => "caco2_wang",
            Self::CarcinogensLagunin => "carcinogens_lagunin",
            Self::ClearanceHepatocyteAz => "clearance_hepatocyte_az",
            Self::Cyp2c19Veith => "cyp2c19_veith",
            Self::Cyp2d6Veith => "cyp2d6_veith",
            Self::Dili => "dili",
            Self::HalfLifeObach => "half_life_obach",
            Self::Herg => "herg",
            Self::HiaHou => "hia_hou",
            Self::HydrationfreeenergyFreesolv => "hydrationfreeenergy_freesolv",
            Self::Ld50Zhu => "ld50_zhu",
            Self::LipophilicityAstrazeneca => "lipophilicity_astrazeneca",
            Self::PampaNcats => "pampa_ncats",
            Self::PgpBroccatelli => "pgp_broccatelli",
            Self::PpbrAz => "ppbr_az",
            Self::SkinReaction => "skin_reaction",
            Self::SolubilityAqsoldb => "solubility_aqsoldb",
            Self::VdssLombardo => "vdss_lombardo",
        }
        .to_string()
    }
}

impl FromStr for DatasetTdc {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let s = s.strip_suffix(".csv").unwrap_or(s);
        let s = s.to_ascii_lowercase();

        match s.as_str() {
            "ames" => Ok(Self::Ames),
            "bbb_martins" => Ok(Self::BbbMartins),
            "bioavailability_ma" => Ok(Self::BioavailabilityMa),
            "caco2_wang" => Ok(Self::Caco2Wang),
            "carcinogens_lagunin" => Ok(Self::CarcinogensLagunin),
            "clearance_hepatocyte_az" => Ok(Self::ClearanceHepatocyteAz),
            "cyp2c19_veith" => Ok(Self::Cyp2c19Veith),
            "cyp2d6_veith" => Ok(Self::Cyp2d6Veith),
            "dili" => Ok(Self::Dili),
            "half_life_obach" => Ok(Self::HalfLifeObach),
            "herg" => Ok(Self::Herg),
            "hia_hou" => Ok(Self::HiaHou),
            "hydrationfreeenergy_freesolv" => Ok(Self::HydrationfreeenergyFreesolv),
            "ld50_zhu" => Ok(Self::Ld50Zhu),
            "lipophilicity_astrazeneca" => Ok(Self::LipophilicityAstrazeneca),
            "pampa_ncats" => Ok(Self::PampaNcats),
            "pgp_broccatelli" => Ok(Self::PgpBroccatelli),
            "ppbr_az" => Ok(Self::PpbrAz),
            "skin_reaction" => Ok(Self::SkinReaction),
            "solubility_aqsoldb" => Ok(Self::SolubilityAqsoldb),
            "vdss_lombardo" => Ok(Self::VdssLombardo),
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid dataset name",
            )),
        }
    }
}

impl Display for DatasetTdc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

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
    pub fn new(mol: &MoleculeSmall, models: &mut HashMap<DatasetTdc, Infer>) -> io::Result<Self> {
        // The target names here must match the CSV names, as downloaded from TDC.
        let adme = Adme {
            intestinal_permeability: infer_general(mol, DatasetTdc::Caco2Wang, models)?,
            intestinal_absorption: infer_general(mol, DatasetTdc::HiaHou, models)?,
            pgp: infer_general(mol, DatasetTdc::PgpBroccatelli, models)?,
            oral_bioavailablity: infer_general(mol, DatasetTdc::BioavailabilityMa, models)?,
            lipophilicity: infer_general(mol, DatasetTdc::LipophilicityAstrazeneca, models)?,
            solubility_water: infer_general(mol, DatasetTdc::SolubilityAqsoldb, models)?,
            blood_brain_barrier: infer_general(mol, DatasetTdc::BbbMartins, models)?,
            plasma_protein_binding_rate: infer_general(mol, DatasetTdc::PpbrAz, models)?,
            membrane_permeability: infer_general(mol, DatasetTdc::PampaNcats, models)?,
            hydration_free_energy: infer_general(
                mol,
                DatasetTdc::HydrationfreeenergyFreesolv,
                models,
            )?,
            vdss: infer_general(mol, DatasetTdc::VdssLombardo, models)?,
            cyp_2c19_inhibition: infer_general(mol, DatasetTdc::Cyp2c19Veith, models)?,
            cyp_2d6_inhibition: infer_general(mol, DatasetTdc::Cyp2d6Veith, models)?,
            half_life: infer_general(mol, DatasetTdc::HalfLifeObach, models)?,
            clearance: infer_general(mol, DatasetTdc::ClearanceHepatocyteAz, models)?,
        };

        let toxicity = Toxicity {
            ld50: infer_general(mol, DatasetTdc::Ld50Zhu, models)?,
            ether_a_go_go: infer_general(mol, DatasetTdc::Herg, models)?,
            mutagencity: infer_general(mol, DatasetTdc::Ames, models)?,
            drug_induced_liver_injury: infer_general(mol, DatasetTdc::Dili, models)?,
            skin_reaction: infer_general(mol, DatasetTdc::SkinReaction, models)?,
            carcinogen: infer_general(mol, DatasetTdc::CarcinogensLagunin, models)?,
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
