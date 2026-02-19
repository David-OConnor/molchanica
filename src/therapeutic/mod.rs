//! Measures properties of the molecules in the human body; relevant to drug development.
//!
//! Potential data sources and other resources.
//! [Therapeutics Data Commons](https://tdcommons.ai/) [Arxiv from 2021](https://arxiv.org/pdf/2102.09548)
//! [TDC download without PyTDC](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2F21LKWG&utm_source=chatgpt.com)
//! [Aqueous Solubility Data Curation](https://github.com/mcsorkun/AqSolDB) [Paper, 2019](https://www.nature.com/articles/s41597-019-0151-1)
//! [PK-DB](https://pk-db.com/) [Paper, 2020](https://academic.oup.com/nar/article/49/D1/D1358/5957165?login=false)
//!
//! [TDC ADME](https://tdcommons.ai/single_pred_tasks/adme/)
//!
//! todo: [Look up QUPKAKE?](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00328)
//!
//! [Qauntum descriptors](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00411j)

pub mod infer;

pub mod pharmacophore;
mod solubility;
pub mod train;

mod gnn;
#[cfg(feature = "train")]
pub mod model_eval;
mod mol_gen;
mod train_md;
mod train_test_split_indices;
// Pub to allow access from the training entry point.

use std::{
    collections::HashMap,
    fmt::Display,
    io,
    io::ErrorKind,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use bio_files::md_params::ForceFieldParams;

#[cfg(not(feature = "train"))]
use crate::therapeutic::train::MODEL_INCLUDE;
use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        infer::{Infer, infer_general},
        train::MODEL_DIR,
    },
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
    Cyp3a4Veith,
    Cyp1a2Veith,
    Cyp2c9Veith,
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
            Self::Cyp3a4Veith => "cyp3a4_veith",
            Self::Cyp1a2Veith => "cyp1a2_veith",
            Self::Cyp2c9Veith => "cypc9_veith",
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

    /// Returns (csv file path, mols (SDF) folder)
    fn csv_mol_paths(self, path: &Path) -> io::Result<(PathBuf, PathBuf)> {
        let name = self.name();

        let csv = path.join(format!("{name}.csv"));

        let mols = path.join(name);

        if !csv.is_file() {
            return Err(io::Error::new(ErrorKind::NotFound, "CSV file not found"));
        }

        if csv.extension().and_then(|s| s.to_str()) != Some("csv") {
            return Err(io::Error::new(ErrorKind::NotFound, "CSV file not found"));
        }

        if !mols.is_dir() {
            return Err(io::Error::new(ErrorKind::NotFound, "Mols folder not found"));
        }

        Ok((csv, mols))
    }

    #[cfg(feature = "train")]
    fn all() -> Vec<Self> {
        // todo: Update A/R
        use DatasetTdc::*;
        vec![
            Ames,
            BbbMartins,
            BioavailabilityMa,
            Caco2Wang,
            CarcinogensLagunin,
            ClearanceHepatocyteAz,
            Cyp2c19Veith,
            Cyp2d6Veith,
            Cyp3a4Veith,
            Cyp1a2Veith,
            Cyp2c9Veith,
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
        ]
    }

    /// Fet standardized filenames for the (model, scalar, config). Used in training.
    pub(in crate::therapeutic) fn model_paths(self) -> (PathBuf, PathBuf, PathBuf) {
        let model_dir = Path::new(MODEL_DIR);

        // Extension is implicit in the model, for Burn.
        // todo: Include bytes.
        let model = model_dir.join(format!("{self}_model"));
        let scaler = model_dir.join(format!("{self}_scaler.json"));
        let cfg = model_dir.join(format!("{self}_model_config.json"));

        (model, scaler, cfg)
    }

    /// Get the models, embedded in the executable. Used in inference (main app only).
    #[cfg(not(feature = "train"))]
    pub(in crate::therapeutic) fn data(
        self,
    ) -> io::Result<(&'static [u8], &'static [u8], &'static [u8])> {
        let model_name = format!("{self}_model.mpk");
        let scaler_name = format!("{self}_scaler.json");
        let cfg_name = format!("{self}_model_config.json");

        let load = |name| match MODEL_INCLUDE.get_file(name) {
            Some(v) => Ok(v.contents()),
            None => Err(io::Error::new(
                ErrorKind::NotFound,
                format!("Missing embedded file: {name}"),
            )),
        };

        let model = load(&model_name)?;
        let scaler = load(&scaler_name)?;
        let cfg = load(&cfg_name)?;

        Ok((model, scaler, cfg))
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
            "cyp3a4_veith" => Ok(Self::Cyp3a4Veith),
            "cyp1a2_veith" => Ok(Self::Cyp1a2Veith),
            "cyp2c9_veith" => Ok(Self::Cyp2c9Veith),
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
    pub cyp_3a4_inhibition: f32,
    pub cyp_1a2_inhibition: f32,
    pub cyp_2c9_inhibition: f32,
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
    pub fn new(
        mol: &MoleculeSmall,
        models: &mut HashMap<DatasetTdc, Infer>,
        ff_params: &ForceFieldParams,
    ) -> io::Result<Self> {
        println!("Inferring ADME properties...");
        let start = Instant::now();

        // Code cleaner
        let mut infer = |dataset| infer_general(mol, dataset, models, ff_params, false);

        // Warning: Unwrap or default here sets to 0 for individual properties

        // The target names here must match the CSV names, as downloaded from TDC.
        let adme = Adme {
            intestinal_permeability: infer(DatasetTdc::Caco2Wang).unwrap_or_default(),
            intestinal_absorption: infer(DatasetTdc::HiaHou).unwrap_or_default(),
            pgp: infer(DatasetTdc::PgpBroccatelli).unwrap_or_default(),
            oral_bioavailablity: infer(DatasetTdc::BioavailabilityMa).unwrap_or_default(),
            lipophilicity: infer(DatasetTdc::LipophilicityAstrazeneca).unwrap_or_default(),
            solubility_water: infer(DatasetTdc::SolubilityAqsoldb).unwrap_or_default(),
            blood_brain_barrier: infer(DatasetTdc::BbbMartins).unwrap_or_default(),
            plasma_protein_binding_rate: infer(DatasetTdc::PpbrAz).unwrap_or_default(),
            membrane_permeability: infer(DatasetTdc::PampaNcats).unwrap_or_default(),
            hydration_free_energy: infer(DatasetTdc::HydrationfreeenergyFreesolv)
                .unwrap_or_default(),
            vdss: infer(DatasetTdc::VdssLombardo).unwrap_or_default(),
            cyp_2c19_inhibition: infer(DatasetTdc::Cyp2c19Veith).unwrap_or_default(),
            cyp_2d6_inhibition: infer(DatasetTdc::Cyp2d6Veith).unwrap_or_default(),
            cyp_3a4_inhibition: infer(DatasetTdc::Cyp3a4Veith).unwrap_or_default(),
            cyp_1a2_inhibition: infer(DatasetTdc::Cyp1a2Veith).unwrap_or_default(),
            cyp_2c9_inhibition: infer(DatasetTdc::Cyp2c9Veith).unwrap_or_default(),
            half_life: infer(DatasetTdc::HalfLifeObach).unwrap_or_default(),
            clearance: infer(DatasetTdc::ClearanceHepatocyteAz).unwrap_or_default(),
        };

        let toxicity = Toxicity {
            ld50: infer(DatasetTdc::Ld50Zhu).unwrap_or_default(),
            ether_a_go_go: infer(DatasetTdc::Herg).unwrap_or_default(),
            mutagencity: infer(DatasetTdc::Ames).unwrap_or_default(),
            drug_induced_liver_injury: infer(DatasetTdc::Dili).unwrap_or_default(),
            skin_reaction: infer(DatasetTdc::SkinReaction).unwrap_or_default(),
            carcinogen: infer(DatasetTdc::CarcinogensLagunin).unwrap_or_default(),
        };

        let elapsed = start.elapsed().as_millis();
        println!("ADME inference complete in {elapsed} ms");

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
