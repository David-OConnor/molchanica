#![cfg(feature = "train")]
#![recursion_limit = "256"] // todo: Troubleshooting a strange error with the  WGPU backend.

//! We expose here components which would make sense to be called by training or other
//! *external* modules. The only current use of this is for our ML training and evaluation executable.
//!
//! This includes much of the application, as it's directly or indirectly used
//! in the training executable. Wee should not, declare any modules here not used by our training pipeline.

// todo: Consider reorging, then trimming down this list.

pub mod bond_inference;
pub mod cam;
pub mod cli;
pub mod docking;
pub mod drawing;
pub mod file_io;
pub mod gromacs;
pub mod inputs;
pub mod md;
pub mod mol_alignment;
pub mod mol_components;
pub mod mol_editor;
pub mod mol_manip;
pub mod molecules;
pub mod orca;
pub mod prefs;
pub mod properties;
pub mod reflection;
pub mod render;
pub mod screening;
pub mod selection;
pub mod sfc_mesh;
pub mod smiles;
pub mod sonification;
pub mod state;
pub mod tautomers;
pub mod therapeutic;
pub mod threads;
pub mod ui;
pub mod util;
