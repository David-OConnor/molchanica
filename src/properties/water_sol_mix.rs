//! Estimate the solubility of a small organic molecule in water by adding a mixture of copies of the
//! [solute] molecule in OPC water.
//!
//! For use eith either the `dynamics`, or `GROMACS` backends.
//!
//! todo: If this works, we can add an octanol sim as well for lipophilicity.
//!
//! Experimental. Traditionally, this approach requires many steps. We are experimenmting to see
//! if we can find an approach which takes a reasonable number of steps.

use crate::md::MdBackend;
use crate::molecules::common::MoleculeCommon;
use dynamics::ComputationDevice;
//
// // todo: Experimetning with different approaches to computationally-cheaply estimate solubility, or
// // todo: other properties using M.
// /// A simulation of a solute probe molecule, at the boundary between two layers of molecules. One layer is water.
// /// the other is (todo: What?)
// pub fn boundary_layer_solute_in_middle(dev: &ComputationDevice, backend: MdBackend) {
//     // todo: Move this consts out of the fn if you wish. Here for now to isolate them.
//     const num_water: usize = 60;
//     // const num_octanol: usize = 30; // todo?
// }

/// A simulation of two touching layers, with no probe molecule: One layer is the molecule being
/// measured; the other is water.
pub fn boundary_layer_solute_water(
    mol: &MoleculeCommon,
    dev: &ComputationDevice,
    backend: MdBackend,
) {
    // todo: Move this consts out of the fn if you wish. Here for now to isolate them.
    const num_water: usize = 60;
    // const num_octanol: usize = 30; // todo?
}
