//! This module contains loosely related objects which infer properties of molecules
//! through MD, or other methods.

use std::io;

use dynamics::{ParamError, alchemical::AlchemicalError};

pub mod crystal;
pub mod ionization;
pub mod logp;
pub mod mol_characterization;
pub mod water_sol;
mod water_sol_analytic;
mod water_sol_mix;

// todo: A/R
pub fn param_error(context: &str, err: AlchemicalError) -> ParamError {
    ParamError::new(&format!("{context}: {err}"))
}

pub fn io_error(context: &str, err: AlchemicalError) -> io::Error {
    io::Error::other(format!("{context}: {err}"))
}
