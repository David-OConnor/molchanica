//! todo: Should not be part of therapeutic; here for now as it uses related MD code
//!
//! Use MD based on ORCA or similar quantum chemistry computations.
//!
//! We have some options, including training on Orca's ab-initio MD directly, and
//! deriving electron density from it, then using that. (For example, using ORCA to generate multipoles,
//! and calculating coulomb force from those)
