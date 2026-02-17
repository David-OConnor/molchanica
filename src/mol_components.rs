//! Represents small organic molecules by breaking them down into components.

use crate::molecules::common::MoleculeCommon;
use crate::molecules::{Atom, Bond};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct RingComponent {
    pub num_atoms: u8,
    pub aromatic: bool,
}

#[derive(Clone, Debug)]
pub enum ComponentType {
    Ring(RingComponent),
    Chain(usize), // len
    Hydroxyl,
    Carbonyl,
    Carboxyl,
    Sulfonamide,
    Sulfonimide,
}

impl ComponentType {
    pub fn to_atoms_bonds(&self) -> (Vec<Atom>, Vec<Bond>) {
        use ComponentType::*;
        match self {
            Ring(ring) => {}
            Chain(count) => {}
            Hydroxyl => {}
            Carbonyl => {}
            Carboxyl => {}
            Sulfonamide => {}
            Sulfonimide => {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct Component {
    pub comp_type: ComponentType,
}

impl Component {
    pub fn create(atoms: &[Atom], bonds: &[Bond]) -> Vec<Self> {
        Vec::new()
    }
}

/// Similar to the concept of a covalent bond between atoms, but between Components.
#[derive(Clone, Debug)]
pub struct Connection {
    pub comp_0: usize,
    /// Relative to the component
    pub atom_0: usize,
    pub comp_1: usize,
    pub atom_1: usize,
}

#[derive(Clone, Debug)]
pub struct MolFromComponents {
    pub components: Vec<Component>,
    pub connections: Vec<Connection>,
}

impl MolFromComponents {
    pub fn to_atoms_bonds(&self) -> (Vec<Atom>, Vec<Bond>) {
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();

        for con in &self.connections {
            let (atoms_0, bonds_0) = self.components[con.comp_0].comp_type.to_atoms_bonds();
            let (atoms_1, bonds_1) = self.components[con.comp_1].comp_type.to_atoms_bonds();

            atoms.extend(atoms_0);
            atoms.extend(atoms_1);

            // todo: Bond logic here.
        }

        let mut mol = MoleculeCommon::new(String::new(), atoms, bonds, HashMap::new(), None);
        mol.reassign_sns();

        (mol.atoms, mol.bonds)
    }
}
