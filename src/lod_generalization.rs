//! An experimental module related to viewing and conducting MD sims on multiple,
//! or a gradual level of size and detail. For example, atom-level MD simulations
//! for a single protein or collection of small molecules, and coarser, more zoomed-out
//! models for viewing protein complexes, cells, etc.
//!
//! A single model can be used at various Level of Detail (LoD). We are defining high LoD
//! to mean more "zoomed-in", or to show more details. For example, we might model each atom
//! as a point in higher LoDs, while in lower LoDs, we don't consider atoms directly.

#[derive(Clone, Copy, PartialEq)]
enum LodCoarse {
    Atom,
    Molecule,
    MoleculeComplex,
    Cell,
}

/// A biological cell
pub struct Cell {}

/// Composed of atoms at higher LoDs; as a single body at lower ones.
/// Perhaps as residues, chains, or collections of residues at intermediate ones.
/// The concept of Protein may not have significance at LoDs below a certain point.
pub struct Protein {}

pub struct ModelScale {
    /// Is this a measure of some property in Angstrom?
    pub level_of_detail: f32,
    pub lod_coarse: LodCoarse,
}

impl ModelScale {
    /// E.g like in an MD sim
    pub fn step(&mut self) {}

    /// E.g. draw on the screen using meshes, at an appropriate level of detail. This might
    /// reflect how zoomed in a viewport is, i.e. how many items and of what scale are being
    /// rendered.
    pub fn render(&self) {}
}
