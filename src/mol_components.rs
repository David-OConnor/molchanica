//! Represents small organic molecules by breaking them down into components.
//! Components correspond to chemically meaningful subgraphs (functional groups, rings, chains),
//! and are connected by `Connection`s that mirror the actual bonds that cross component boundaries.
//!
//! Primary uses:
//!   - As a graph neural network (GNN) feature representation for ML.
//!   - As an editor building block: swap or parametrically modify components.

use std::collections::{HashMap, HashSet, VecDeque};

use na_seq::Element::{self, *};

use crate::{
    mol_characterization::RingType,
    molecules::{Atom, Bond, common::MoleculeCommon, small::MoleculeSmall},
};

#[derive(Clone, Debug)]
pub struct RingComponent {
    /// For a fused ring system this is the total atom count across all rings in the system.
    pub num_atoms: u8,
    pub ring_type: RingType,
}

#[derive(Clone, Debug)]
pub enum ComponentType {
    /// Fallback for atoms that don't fit any other component category.
    Atom(Element),
    /// A single ring or a fused ring system (naphthalene, indole, etc.).
    Ring(RingComponent),
    /// A carbon chain (alkyl, alkenyl, …). Stores the number of carbon atoms.
    Chain(usize),
    Hydroxyl,
    Carbonyl,
    Carboxylate,
    Amine,
    Amide,
    Sulfonamide,
    Sulfonimide,
}

impl ComponentType {
    pub fn to_atoms_bonds(&self) -> (Vec<Atom>, Vec<Bond>) {
        // todo: Implement atom/bond synthesis for each component type.
        (Vec::new(), Vec::new())
    }
}

/// A component of a molecule — a functional group, ring system, chain, or fallback atom.
///
/// `atoms` holds indices into the parent molecule's atom array.  The key (junction-capable)
/// atom is always stored first: O for Hydroxyl, N for Amine/Amide/Sulfonamide/Sulfonimide,
/// C for Carbonyl/Carboxylate/Chain.
#[derive(Clone, Debug)]
pub struct Component {
    pub comp_type: ComponentType,
    /// Atom indices into the parent molecule's atom list.
    pub atoms: Vec<usize>,
}

impl Component {
    pub fn create(atoms: &[Atom], bonds: &[Bond]) -> Vec<Self> {
        // todo: Bottom-up detection from raw atoms/bonds (inverse of `MolComponents::new`).
        Vec::new()
    }
}

/// A bond between two components; analogous to a covalent bond between individual atoms.
///
/// `atom_0` / `atom_1` are indices *within* `components[comp_N].atoms`, identifying which
/// atom in each component participates in this inter-component bond.
#[derive(Clone, Debug)]
pub struct Connection {
    pub comp_0: usize,
    pub atom_0: usize,
    pub comp_1: usize,
    pub atom_1: usize,
}

/// The top-level data structure for representing a molecule as a set of connected components.
/// This can represent any small organic molecule without requiring atom positions.  Primary
/// uses are as a GNN input for ML and as an editor building block for quick structural edits.
#[derive(Clone, Debug)]
pub struct MolComponents {
    pub components: Vec<Component>,
    pub connections: Vec<Connection>,
}

impl MolComponents {
    /// Build a component graph from a molecule that already has a characterization.
    ///
    /// Atom-claiming priority (high → low):
    ///   1. Rings / fused ring systems
    ///   2. Carboxylates (before plain carbonyl to avoid double-counting the C)
    ///   3. Sulfonimides
    ///   4. Sulfonamides
    ///   5. Amides
    ///   6. Carbonyls (C=O not part of a carboxylate)
    ///   7. Amines
    ///   8. Hydroxyls
    ///   9. Carbon chains (≥2 connected unclaimed C atoms)
    ///  10. Singleton fallback for anything remaining
    ///
    /// Connections are then derived by walking the molecule's bond list and recording every
    /// bond whose two endpoint atoms belong to different components.
    pub fn new(mol: &MoleculeSmall) -> Option<Self> {
        let Some(char) = &mol.characterization else {
            return None;
        };

        let atoms = &mol.common.atoms;
        let adj = &mol.common.adjacency_list;
        let bonds = &mol.common.bonds;
        let n_atoms = atoms.len();

        let mut comps: Vec<Component> = Vec::new();
        // atom index → component index
        let mut atom_to_comp: HashMap<usize, usize> = HashMap::new();
        let mut claimed: HashSet<usize> = HashSet::new();

        // Inline helper: register a component, claim all of its atoms.
        macro_rules! add_comp {
            ($comp_type:expr, $comp_atoms:expr) => {{
                let ci = comps.len();
                let comp_atoms: Vec<usize> = $comp_atoms;
                for &a in &comp_atoms {
                    atom_to_comp.insert(a, ci);
                    claimed.insert(a);
                }
                comps.push(Component {
                    comp_type: $comp_type,
                    atoms: comp_atoms,
                });
            }};
        }

        // --- 1. Rings ---
        // Fused ring systems become a single component; isolated rings each get their own.
        let mut ring_in_system: HashSet<usize> = HashSet::new();

        for system in &char.ring_systems {
            let mut sys_atoms: Vec<usize> = Vec::new();
            for &ri in system {
                ring_in_system.insert(ri);
                for &a in &char.rings[ri].atoms {
                    if !sys_atoms.contains(&a) {
                        sys_atoms.push(a);
                    }
                }
            }
            // Aromatic takes precedence; then Aliphatic; finally Saturated.
            let ring_type = system.iter().map(|&i| char.rings[i].ring_type).fold(
                RingType::Saturated,
                |best, rt| match rt {
                    RingType::Aromatic => RingType::Aromatic,
                    RingType::Aliphatic if best != RingType::Aromatic => RingType::Aliphatic,
                    _ => best,
                },
            );
            let num_atoms = sys_atoms.len() as u8;
            add_comp!(
                ComponentType::Ring(RingComponent {
                    num_atoms,
                    ring_type
                }),
                sys_atoms
            );
        }

        for (ri, ring) in char.rings.iter().enumerate() {
            if ring_in_system.contains(&ri) {
                continue;
            }
            let ring_atoms: Vec<usize> = ring
                .atoms
                .iter()
                .filter(|&&a| !claimed.contains(&a))
                .copied()
                .collect();
            if ring_atoms.is_empty() {
                continue;
            }
            add_comp!(
                ComponentType::Ring(RingComponent {
                    num_atoms: ring.atoms.len() as u8,
                    ring_type: ring.ring_type,
                }),
                ring_atoms
            );
        }

        // --- 2. Carboxylates (C + both O atoms) ---
        let carboxylate_cs: HashSet<usize> = char.carboxylate.iter().copied().collect();
        for &c_idx in &char.carboxylate {
            if claimed.contains(&c_idx) {
                continue;
            }
            let mut comp_atoms = vec![c_idx]; // key atom first
            for &nb in &adj[c_idx] {
                if atoms[nb].element == Oxygen && !claimed.contains(&nb) {
                    comp_atoms.push(nb);
                    // Include H on the -OH oxygen.
                    for &h in &adj[nb] {
                        if atoms[h].element == Hydrogen && !claimed.contains(&h) {
                            comp_atoms.push(h);
                        }
                    }
                }
            }
            add_comp!(ComponentType::Carboxylate, comp_atoms);
        }

        // --- 3. Sulfonimides (N bonded to two sulfonyl S groups) ---
        for &n_idx in &char.sulfonimide {
            if claimed.contains(&n_idx) {
                continue;
            }
            let mut comp_atoms = vec![n_idx]; // key atom first
            for &nb in &adj[n_idx] {
                match atoms[nb].element {
                    Hydrogen if !claimed.contains(&nb) => comp_atoms.push(nb),
                    Sulfur if !claimed.contains(&nb) => {
                        comp_atoms.push(nb);
                        for &snb in &adj[nb] {
                            if atoms[snb].element == Oxygen && !claimed.contains(&snb) {
                                comp_atoms.push(snb);
                            }
                        }
                    }
                    _ => {}
                }
            }
            add_comp!(ComponentType::Sulfonimide, comp_atoms);
        }

        // --- 4. Sulfonamides (N bonded to one sulfonyl S group) ---
        for &n_idx in &char.sulfonamide {
            if claimed.contains(&n_idx) {
                continue;
            }
            let mut comp_atoms = vec![n_idx]; // key atom first
            for &nb in &adj[n_idx] {
                match atoms[nb].element {
                    Hydrogen if !claimed.contains(&nb) => comp_atoms.push(nb),
                    Sulfur if !claimed.contains(&nb) => {
                        comp_atoms.push(nb);
                        for &snb in &adj[nb] {
                            if atoms[snb].element == Oxygen && !claimed.contains(&snb) {
                                comp_atoms.push(snb);
                            }
                        }
                    }
                    _ => {}
                }
            }
            add_comp!(ComponentType::Sulfonamide, comp_atoms);
        }

        // --- 5. Amides (N only; the C=O carbon is handled by Carbonyl below) ---
        for &n_idx in &char.amides {
            if claimed.contains(&n_idx) {
                continue;
            }
            let mut comp_atoms = vec![n_idx]; // key atom first
            for &nb in &adj[n_idx] {
                if atoms[nb].element == Hydrogen && !claimed.contains(&nb) {
                    comp_atoms.push(nb);
                }
            }
            add_comp!(ComponentType::Amide, comp_atoms);
        }

        // --- 6. Carbonyls (C=O, excluding carboxylate carbons already claimed) ---
        for &c_idx in &char.carbonyl {
            if claimed.contains(&c_idx) || carboxylate_cs.contains(&c_idx) {
                continue;
            }
            let mut comp_atoms = vec![c_idx]; // key atom first
            for &nb in &adj[c_idx] {
                if atoms[nb].element == Oxygen && !claimed.contains(&nb) {
                    comp_atoms.push(nb);
                }
            }
            add_comp!(ComponentType::Carbonyl, comp_atoms);
        }

        // --- 7. Amines ---
        for &n_idx in &char.amines {
            if claimed.contains(&n_idx) {
                continue;
            }
            let mut comp_atoms = vec![n_idx]; // key atom first
            for &nb in &adj[n_idx] {
                if atoms[nb].element == Hydrogen && !claimed.contains(&nb) {
                    comp_atoms.push(nb);
                }
            }
            add_comp!(ComponentType::Amine, comp_atoms);
        }

        // --- 8. Hydroxyls (O-H; skip O atoms already claimed by e.g. carboxylate) ---
        for &o_idx in &char.hydroxyl {
            if claimed.contains(&o_idx) {
                continue;
            }
            let mut comp_atoms = vec![o_idx]; // key atom first
            for &nb in &adj[o_idx] {
                if atoms[nb].element == Hydrogen && !claimed.contains(&nb) {
                    comp_atoms.push(nb);
                }
            }
            add_comp!(ComponentType::Hydroxyl, comp_atoms);
        }

        // --- 9. Carbon chains ---
        // BFS over unclaimed carbons; runs of ≥2 become a Chain component.
        // Single isolated carbons fall through to the singleton fallback.
        let mut chain_seen = vec![false; n_atoms];
        for start in 0..n_atoms {
            if claimed.contains(&start) || chain_seen[start] {
                continue;
            }
            if atoms[start].element != Carbon {
                continue;
            }
            let mut chain_atoms: Vec<usize> = Vec::new();
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(start);
            chain_seen[start] = true;
            while let Some(cur) = queue.pop_front() {
                chain_atoms.push(cur);
                for &nb in &adj[cur] {
                    if !claimed.contains(&nb) && !chain_seen[nb] && atoms[nb].element == Carbon {
                        chain_seen[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
            if chain_atoms.len() >= 2 {
                let len = chain_atoms.len();
                add_comp!(ComponentType::Chain(len), chain_atoms);
            }
        }

        // --- 10. Fallback: singleton component for every remaining atom ---
        for i in 0..n_atoms {
            if !claimed.contains(&i) {
                add_comp!(ComponentType::Atom(atoms[i].element), vec![i]);
            }
        }

        // --- Build connections ---
        // Every bond whose two endpoints belong to different components becomes a Connection.
        // `atom_0` / `atom_1` are positions within the respective component's `atoms` list.
        let mut conns: Vec<Connection> = Vec::new();

        for bond in bonds {
            let a = bond.atom_0;
            let b = bond.atom_1;

            let (Some(&ca), Some(&cb)) = (atom_to_comp.get(&a), atom_to_comp.get(&b)) else {
                continue;
            };
            if ca == cb {
                continue; // internal bond, not a cross-component connection
            }

            let atom_0 = comps[ca].atoms.iter().position(|&x| x == a).unwrap_or(0);
            let atom_1 = comps[cb].atoms.iter().position(|&x| x == b).unwrap_or(0);

            conns.push(Connection {
                comp_0: ca,
                atom_0,
                comp_1: cb,
                atom_1,
            });
        }

        Some(Self {
            components: comps,
            connections: conns,
        })
    }

    pub fn to_atoms_bonds(&self) -> (Vec<Atom>, Vec<Bond>) {
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();

        for con in &self.connections {
            let (atoms_0, bonds_0) = self.components[con.comp_0].comp_type.to_atoms_bonds();
            let (atoms_1, bonds_1) = self.components[con.comp_1].comp_type.to_atoms_bonds();

            atoms.extend(atoms_0);
            atoms.extend(atoms_1);

            // todo: reconstruct the inter-component bond using con.atom_0 / atom_1 offsets.
        }

        let mut mol = MoleculeCommon::new(String::new(), atoms, bonds, HashMap::new(), None);
        mol.reassign_sns();

        (mol.atoms, mol.bonds)
    }
}
