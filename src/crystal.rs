//! Experiments with applying our visualization, molecule creation, and MD onto [initially non-organic] crystals. This
//! is a departure from existing code which focuses on classes of organic molecules: Small molecules, proteins, lipids,
//! nucleic acid.
//!
//! To start, we begin with Carbon: Graphite, Diamond etc.

use lin_alg::f64::Vec3;
use na_seq::Element::{self, Carbon, Chlorine, Sodium};

pub struct AtomInCrystal {
    pub element: Element,
    pub posit: Vec3,
    /// Indices of covalently-bonded atoms. Todo: Are covalent bonds meaningful
    /// for crystals?
    pub adjacent: Vec<usize>,
    /// E.g. on the edge of the cell
    pub shared_with_neighbor: bool,
}

impl AtomInCrystal {
    pub fn new(
        element: Element,
        posit: Vec3,
        adjacent: Vec<usize>,
        shared_with_neighbor: bool,
    ) -> Self {
        Self {
            element,
            posit,
            adjacent,
            shared_with_neighbor,
        }
    }
}

pub struct AtomInCrystalGraph {
    pub element: Element,
    /// Indices of covalently-bonded atoms. Todo: Are covalent bonds meaningful
    /// for crystals?
    pub adjacent: Vec<usize>,
    /// E.g. on the edge of the cell
    pub shared_with_neighbor: bool,
}

/// A generic crystal cell which can be periodically tiled to arbitrary
/// size.
pub struct CrystalCell {
    // todo: Do we want to encode exact positions and distances here,
    // todo: Or can we leave it as a graph
    pub atoms: Vec<AtomInCrystal>,
    pub atoms_graph: Vec<AtomInCrystalGraph>,
}

impl CrystalCell {
    pub fn new_graphite() -> Self {
        unimplemented!()
    }

    pub fn new_diamond() -> Self {
        unimplemented!()
    }

    pub fn new_sodium_chloride() -> Self {
        // The convention we use here:
        // 14 Chlorine atoms; all shared.
        // 10 Sodium atoms; all but 1 shared.
        let atoms = vec![
            AtomInCrystal::new(
                Sodium,
                Vec3::new(0., 0., 0.),
                Vec::new(), // unused for now
                true,
            ),
            AtomInCrystal::new(Sodium, Vec3::new(0., 0., 0.), Vec::new(), true),
            AtomInCrystal::new(Sodium, Vec3::new(0., 0., 0.), Vec::new(), true),
            AtomInCrystal::new(Sodium, Vec3::new(0., 0., 0.), Vec::new(), true),
        ];

        Self {
            atoms,
            atoms_graph: Vec::new(), // empty for now.
        }
    }
}
