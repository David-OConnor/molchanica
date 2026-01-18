// This feature-gate prevents having to specify `--bin molchanica` when running normally.
#![cfg(feature = "train-sol")]

//! Experimenting with giving the ML training access to Molecule related  data
//!
//! We expose here components which would make sense to be called byt training or other
//! *external* modules.

// todo: Consider reorging, then trimming down this list.

pub mod bond_inference;
pub mod cam;
pub mod cli;
pub mod docking;
pub mod download_mols;
pub mod drawing;
pub mod drawing_wrappers;
pub mod drug_design;
pub mod file_io;
pub mod inputs;
pub mod md;
pub mod mol_alignment;
pub mod mol_characterization;
pub mod mol_editor;
pub mod mol_manip;
pub mod mol_screening;
pub mod molecules;
pub mod orca;
pub mod pharmacokinetics;
pub mod pharmacophore;
pub mod prefs;
pub mod reflection;
pub mod render;
pub mod ribbon_mesh;
pub mod sa_surface;
pub mod selection;
pub mod smiles;
pub mod state;
pub mod tautomers;
pub mod ui;
pub mod util;
pub mod viridis_lut;
