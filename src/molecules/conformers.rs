//! Representation[s] of molecules (Based on MoleculeCommon? MoleculeSmall?) based on their different
//! conformations: The range of shapes they can take.
//!
//! todo: Is the true name "conformation"?
//!
//! Note: We are likely limited more by collecting the conformation data than by
//! the representation. The representation may be non-trivial, but the collecting
//! is likely to be very slow, depending on how it's done.
//!
//! todo: Can we run a brief ML sim in water to sample conformations? Or is that likely
//! todo to missing binding conformations?
//!
//! todo: Gaussians?

use crate::molecules::common::MoleculeCommon;
use dynamics::ParamError;
use dynamics::snapshot::Snapshot;
use graphics::Scene;
use std::time::Instant;

use crate::md::build_dynamics;
use crate::state::State;
use lin_alg::f64::Vec3;

/// For a single atom.
pub struct PositSample {
    /// Discrete, to start. We may wish to generalize this to be a function over
    /// space. X, Y, Z in. Time the atom spends at that posit out.
    pub samples: Vec<Vec3>,
    // todo: Should we represent as a deviation from the initial position?
}

/// Attempt 0: Anchor to a molecule. These conormations map, for each atom,
/// the likely positions this atom can take. These positions are relative to the initial
/// positions of the molecule: `atoms[i].posit`: Not `atom_posits[i]`, as the former is a
/// more stable baseline.
pub struct Conformer {
    // todo: Should this be a ref?
    pub mol: MoleculeCommon,
    /// indexed by atom in `mol`.
    pub samples: Vec<PositSample>,
}

/// See the description on `sample_mol_properties_from_md`.
pub struct MdSampleData {
    // todo: Do we wish to include sample metadata like number of steps, dt, etc?
    pub conformer_samples: Vec<PositSample>, // Conformer struct instead?
    /// For which this molecule is the acceptor, and water is a donor.
    pub num_donor_h_bonds_avg: f32,
    /// For which this molecule is the donor, and water is an acceptor.
    pub num_acc_h_bonds_avg: f32,
    /// This includes both acceptor and donor data, and takes into account the H
    /// bond strength: Not just if there is an H bond.
    pub h_bond_total_str_avg: f32,
    // todo: You can imagine other remixes of donor, acceptor. Strength and count.
}

impl MdSampleData {
    pub fn new(snaps: &[Snapshot]) -> Self {
        // todo fill in
        unimplemented!();

        // todo: Prereq. You may need to add H bonds to your MD snap state.

        Self {
            conformer_samples: Vec::new(),
            num_donor_h_bonds_avg: 0.0,
            num_acc_h_bonds_avg: 0.0,
            h_bond_total_str_avg: 0.0,
        };
    }
}
/// WIP / experimental. Run a MD simulation of the molecule in water to sample various properties.
/// This includes conformation data (As is typical of this molecule this functionality currently lives in),
/// and other properties of interest. For example, perhaps we sample the number of hydrogen bonds the molecule
/// forms on average.
///
/// Limitation: This should be short, so it can be usd in screening. It's worth exploring, even if it
/// takes longer than desired for screening.
///
/// todo: For your MD applications, cache the computation of this while developing, and perhaps training.
///
/// todo: Evolve this over time, and move it where appropriate.
pub fn sample_mol_properties_from_md(
    mol: &MoleculeCommon, // todo: Should this be a MoleculeSmall?
    state: &State,        // todo: Do we need mut?
    scene: &Scene,        // todo: Why?
) -> Result<MdSampleData, ParamError> {
    println!("Starting per mol sim for {}...", mol.ident);
    let start = Instant::now();

    unimplemented!();
    // let md = build_dynamics()?;

    // let result = MdSampleData::new(&snaps);

    let elapsed = start.elapsed().as_millis();

    println!("Finished per mol sim in {} ms", elapsed);

    // result
}
