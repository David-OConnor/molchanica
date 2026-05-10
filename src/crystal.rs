//! For representing small molecules in homogenous crystal lattices. This has implications
//! for measuring self-affinity, for example, for the purposes of assessing solubility in water, or other
//! solvents. It models
//! and infers properties about how, for example, a drug-like molecule might exist as a crystalline
//! powder.

use std::io;

use dynamics::snapshot::Snapshot;

use crate::{md::MdBackend, molecules::small::MoleculeSmall};

const NUM_COPIES_FOR_MD: usize = 40; // todo: Set A/R
// todo: Consider making this dynamic once basic functionality in this module works. I.e., run until
// todo: a crystal structure is stable, and use this as an upper bound
const NUM_STEPS: usize = 400;
const TEMPERATURE: f32 = 300.; // K.  todo:  Set A/R
const PRESSURE: f32 = 1.; // Bar. todo: A/R.

/// Contains self-affinity, and other related results for a small organic molecule in a crystalline
/// molecule with only itself.
#[derive(Clone, Debug)]
pub struct CrystalData {
    pub self_affinity_score: f32,
    // todo: More data as required.
}

/// Runs an MD simulation of a number of the molecule being analyzed, with no solvent. Assesses
/// data about the crystal structure using <?>. Returns the resulting snapshots, which can be
/// used to visualize the results.
pub fn estimate_from_md(
    mol: &MoleculeSmall,
    backend: MdBackend,
) -> io::Result<(CrystalData, Vec<Snapshot>)> {
    Ok(())
}
