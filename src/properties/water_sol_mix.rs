//! Estimate the solubility of a small organic molecule in water by adding a mixture of copies of the
//! [solute] molecule in OPC water.
//!
//! For use eith either the `dynamics`, or `GROMACS` backends.
//!
//! todo: If this works, we can add an octanol sim as well for lipophilicity.
//!
//! Experimental. Traditionally, this approach requires many steps. We are experimenmting to see
//! if we can find an approach which takes a reasonable number of steps.
