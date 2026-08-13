//! We recently moved most of the `therapeutic` module to the `adme_` module, and may
//! soon move it to a new `adme` standalone crate. This includes the parts of the old therapeutic
//! module which aren't associated with ADME ML.

pub mod ddg;
mod solubility;
