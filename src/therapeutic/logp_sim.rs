//! Creates a MD sim to measure LogP or LogD, by creating a solvent mixture of 50/50 water
//! and octonal, along with many copies of the solute being measured. The ratio of solute which
//! ends in the water vs octanol is the measurement result.
//!
//! This is a lipophilicity measurement.

use std::collections::HashSet;

use dynamics::{MdConfig, MdState, ParamError};

use crate::{md::build_dynamics, molecules::small::MoleculeSmall, state::State};

pub fn build_dynamics_logp(mol: &MoleculeSmall, state: &State) -> Result<MdState, ParamError> {
    let mols = vec![];

    build_dynamics(
        &state.dev,
        &mols,
        None,
        &state.ff_param_set,
        &state.mol_specific_params,
        &state.to_save.md_config,
        state.ui.md.peptide_static,
        None,
        &mut HashSet::new(),
        false,
        state.to_save.num_md_copies,
    )
}
