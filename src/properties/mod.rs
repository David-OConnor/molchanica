//! This module contains loosely related objects which infer properties of molecules
//! through MD, or other methods.

use std::{collections::HashMap, io};

use bio_files::md_params::ForceFieldParams;
use dynamics::{ParamError, alchemical::AlchemicalError, params::FfParamSet};

use crate::molecules::small::MoleculeSmall;

pub mod crystal;
pub mod ionization;
pub mod logp;
pub mod mol_characterization;
pub mod water_sol;
mod water_sol_analytic;
pub mod water_sol_mix;

// todo: A/R
pub fn param_error(context: &str, err: AlchemicalError) -> ParamError {
    ParamError::new(&format!("{context}: {err}"))
}

pub fn io_error(context: &str, err: AlchemicalError) -> io::Error {
    io::Error::other(format!("{context}: {err}"))
}

/// Prepare a small organic molecule for standalone MD property presets.
///
/// These presets do not have access to the application's state-level frcmod
/// cache, so they force a fresh local molecule-specific GAFF2 parameter pass.
pub(crate) fn prepare_mol_for_md(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
) -> io::Result<(MoleculeSmall, HashMap<String, ForceFieldParams>)> {
    let Some(gaff2) = param_set.small_mol.as_ref() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Missing GAFF2 small-molecule parameters.",
        ));
    };

    let mut mol = mol.clone();
    mol.common.selected_for_md = Some(1);
    mol.frcmod_loaded = false;

    let mut mol_specific_params = HashMap::new();
    mol.update_ff_related(&mut mol_specific_params, gaff2, false);

    if !mol.ff_params_loaded || !mol.frcmod_loaded {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Unable to infer force-field parameters for {}.",
                mol.common.ident
            ),
        ));
    }

    mol.update_characterization();

    Ok((mol, mol_specific_params))
}
