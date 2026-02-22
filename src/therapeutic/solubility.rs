#![allow(unused)]

//! We attempt 3 approaches:
//! - ML, using, for example, the [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1)
//!   -- (AqSolDB data)[https://github.com/mcsorkun/AqSolDB]
//!   -- (AqSolDb web UI)[https://www.amdlab.nl/database/AqSolDB/]
//! - Simple metrics based on properties of properties of the molecule.
//! - MD simulation of the molecule in water
//!
//! We use the same floating point scale used by AqSolDb. Higher is more solubile (todo: QC)
//! Unit: LogS, where S is the aqueous solubility in mol/L (or M).
//!
//! Compounds with 0 and higher solubility value are highly soluble, those in the range of 0 to −2
//! are soluble, those in the range of −2 to −4 are slightly soluble and insoluble if less than −4

use dynamics::ParamError;

use crate::molecules::small::MoleculeSmall;

/// Using MD on the MqSolDB dataset.
pub fn solubility_from_ml(mol: &MoleculeSmall) -> f32 {
    0.
}

/// todo: Anything besides the mol and water? What properties are we measuring
/// todo to determine solubility?
pub fn solubility_from_md(mol: &MoleculeSmall) -> Result<f32, ParamError> {
    Ok(0.)
}
