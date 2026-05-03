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
pub struct conformer {
    // todo: Should this be a ref?
    pub mol: MoleculeCommon,
    /// indexed by atom in `mol`.
    pub samples: Vec<PositSample>,
}
