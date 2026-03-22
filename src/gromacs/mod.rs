//! Experimental integration with [GROMACS](https://www.gromacs.org/)

use std::{
    collections::{HashMap, HashSet},
    process::Command,
};

use bio_files::md_params::ForceFieldParams;
use dynamics::{ComputationDevice, FfMolType, MdConfig, params::FfParamSet, snapshot::Snapshot};

use crate::molecules::common::MoleculeCommon;

/// Conceptually similar to `md::build_dynamics` and `run_blocking`, but passed to GRAOMCS.
pub fn run_dynamics(
    dev: &ComputationDevice,
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    mut static_peptide: bool,
    mut peptide_only_near_lig: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    fast_init: bool,
    dt: f32,
    n_steps: usize,
) -> Vec<Snapshot> {
    Vec::new() // placeholder
}
