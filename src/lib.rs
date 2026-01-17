// This feature-gate prevents having to specify `--bin molchanica` when running normally.
#![cfg(feature = "train-sol")]

//! Experimenting with giving the ML training access to Molecule related  data
//!
//! We expose here components which would make sense to be called byt training or other
//! *expernal* modules.

// pub mod mol_characterization;
// pub mod molecules;
// pub mod pharmacokinetics;
// pub mod sa_surface;
// pub mod util;

pub mod bond_inference;
pub mod drawing;
pub mod drug_design;
// pub mod md;
// pub mod mol_alignment;
pub mod mol_characterization;
pub mod molecules;
pub mod pharmacokinetics;
pub mod pharmacophore;
pub mod render;
pub mod sa_surface;
pub mod selection;
pub mod smiles;
pub mod state;
pub mod tautomers;
pub mod util;
pub mod viridis_lut;
