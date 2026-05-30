//! This module contains loosely related objects which infer properties of molecules
//! through MD, or other methods.

use std::{collections::HashMap, io};

use crate::molecules::small::MoleculeSmall;
use bio_files::md_params::ForceFieldParams;
use dynamics::{ParamError, alchemical::AlchemicalError, params::FfParamSet};
use lin_alg::f32::Vec3;

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
pub(in crate::properties) fn prepare_mol_for_md(
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

pub(in crate::properties) fn mean(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f32>() / values.len() as f32)
    }
}

pub(in crate::properties) fn min_image(mut delta: Vec3, extent: Vec3) -> Vec3 {
    if extent.x > 0.0 {
        delta.x -= extent.x * (delta.x / extent.x).round();
    }
    if extent.y > 0.0 {
        delta.y -= extent.y * (delta.y / extent.y).round();
    }
    if extent.z > 0.0 {
        delta.z -= extent.z * (delta.z / extent.z).round();
    }

    delta
}
