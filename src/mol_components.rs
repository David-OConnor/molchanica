//! Represents small organic molecules by breaking them down into components.
//! Components correspond to chemically meaningful subgraphs (functional groups, rings, chains),
//! and are connected by `Connection`s that mirror the actual bonds that cross component boundaries.
//!
//! Primary uses:
//!   - As a graph neural network (GNN) feature representation for ML.
//!   - As an editor building block: swap or parametrically modify components.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
};

use bio_files::BondType;
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

impl Display for ComponentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ComponentType::*;
        let v = match self {
            Atom(element) => format!("Atom: {}", element),
            Ring(ring) => format!("Ring: {:?}", ring.ring_type),
            Chain(chain) => format!("Chain: {}", chain),
            Hydroxyl => "Hydroxyl".to_string(),
            Carbonyl => "Carbonyl".to_string(),
            Carboxylate => "Carboxylate".to_string(),
            Amine => "Amine".to_string(),
            Amide => "Amide".to_string(),
            Sulfonamide => "Sulfonamide".to_string(),
            Sulfonimide => "Sulfonimide".to_string(),
        };

        write!(f, "{}", v)
    }
}

impl ComponentType {
    /// Return the canonical atoms and bonds for this component type.
    ///
    /// Atom order mirrors the order used when *building* a component in `MolComponents::new`
    /// (key/junction atom first), so that `Connection::atom_0` / `atom_1` indices remain valid.
    /// Bond indices are 0-based and local to the returned atom slice.
    pub fn to_atoms_bonds(&self) -> (Vec<Atom>, Vec<Bond>) {
        // Convenience: atom with only the element set.
        let a = |element: Element| Atom {
            element,
            ..Default::default()
        };
        // Convenience: bond with correct indices and placeholder SNs (reassign_sns fixes these).
        let b = |i: usize, j: usize, bond_type: BondType| Bond {
            bond_type,
            atom_0_sn: (i + 1) as u32,
            atom_1_sn: (j + 1) as u32,
            atom_0: i,
            atom_1: j,
            is_backbone: false,
        };

        match self {
            // Single atom; no bonds.
            ComponentType::Atom(el) => (vec![a(*el)], vec![]),

            // Ring of `num_atoms` carbons; bond type reflects aromaticity.
            // Atom order: follows ring closure (0-1-2-…-(n-1)-0), key atom at index 0.
            ComponentType::Ring(ring) => {
                let n = ring.num_atoms as usize;
                let bt = match ring.ring_type {
                    RingType::Aromatic => BondType::Aromatic,
                    _ => BondType::Single,
                };
                let atoms: Vec<Atom> = (0..n).map(|_| a(Carbon)).collect();
                let mut bonds: Vec<Bond> = (0..n - 1).map(|i| b(i, i + 1, bt)).collect();
                bonds.push(b(n - 1, 0, bt)); // close the ring
                (atoms, bonds)
            }

            // Linear carbon chain; key atom (junction) is index 0.
            ComponentType::Chain(n) => {
                let atoms: Vec<Atom> = (0..*n).map(|_| a(Carbon)).collect();
                let bonds: Vec<Bond> = (0..*n - 1).map(|i| b(i, i + 1, BondType::Single)).collect();
                (atoms, bonds)
            }

            // O(0) — H(1)
            ComponentType::Hydroxyl => (
                vec![a(Oxygen), a(Hydrogen)],
                vec![b(0, 1, BondType::Single)],
            ),

            // O(0) = C(1)  (O is the key atom per component-building convention)
            ComponentType::Carbonyl => {
                (vec![a(Oxygen), a(Carbon)], vec![b(0, 1, BondType::Double)])
            }

            // C(0) =O(1), C(0)–O(2)–H(3)
            ComponentType::Carboxylate => (
                vec![a(Carbon), a(Oxygen), a(Oxygen), a(Hydrogen)],
                vec![
                    b(0, 1, BondType::Double), // C=O
                    b(0, 2, BondType::Single), // C-OH
                    b(2, 3, BondType::Single), // O-H
                ],
            ),

            // N(0)–H(1), N(0)–H(2)  (primary amine; lone Hs that were captured)
            ComponentType::Amine => (
                vec![a(Nitrogen), a(Hydrogen), a(Hydrogen)],
                vec![b(0, 1, BondType::Single), b(0, 2, BondType::Single)],
            ),

            // N(0)–H(1)  (amide N; the carbonyl C lives in a different component)
            ComponentType::Amide => (
                vec![a(Nitrogen), a(Hydrogen)],
                vec![b(0, 1, BondType::Single)],
            ),

            // N(0)–H(1), N(0)–S(2), S(2)=O(3), S(2)=O(4)
            ComponentType::Sulfonamide => (
                vec![a(Nitrogen), a(Hydrogen), a(Sulfur), a(Oxygen), a(Oxygen)],
                vec![
                    b(0, 1, BondType::Single), // N-H
                    b(0, 2, BondType::Single), // N-S
                    b(2, 3, BondType::Double), // S=O
                    b(2, 4, BondType::Double), // S=O
                ],
            ),

            // N(0)–H(1), N(0)=S(2), S(2)=O(3), S(2)=O(4)  (sulfonimide has N=S)
            ComponentType::Sulfonimide => (
                vec![a(Nitrogen), a(Hydrogen), a(Sulfur), a(Oxygen), a(Oxygen)],
                vec![
                    b(0, 1, BondType::Single), // N-H
                    b(0, 2, BondType::Double), // N=S
                    b(2, 3, BondType::Double), // S=O
                    b(2, 4, BondType::Double), // S=O
                ],
            ),
        }
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
    /// True if this the atom is shared between both components; false if they are
    /// separate components. For example, this will be true on the connection points
    /// between fused rings, and there will be two connections fusing the rings.
    pub shared_atoms: bool,
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
        // let mut ring_in_system: HashSet<usize> = HashSet::new();

        // for system in &char.ring_systems {
        //     let mut sys_atoms: Vec<usize> = Vec::new();
        //     for &ri in system {
        //         ring_in_system.insert(ri);
        //         for &a in &char.rings[ri].atoms {
        //             if !sys_atoms.contains(&a) {
        //                 sys_atoms.push(a);
        //             }
        //         }
        //     }
        //     // Aromatic takes precedence; then Aliphatic; finally Saturated.
        //     let ring_type = system.iter().map(|&i| char.rings[i].ring_type).fold(
        //         RingType::Saturated,
        //         |best, rt| match rt {
        //             RingType::Aromatic => RingType::Aromatic,
        //             RingType::Aliphatic if best != RingType::Aromatic => RingType::Aliphatic,
        //             _ => best,
        //         },
        //     );
        //     let num_atoms = sys_atoms.len() as u8;
        //     add_comp!(
        //         ComponentType::Ring(RingComponent {
        //             num_atoms,
        //             ring_type
        //         }),
        //         sys_atoms
        //     );
        // }

        for ring in &char.rings {
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

        // char.carbonyl now stores O atom indices (the =O oxygen, not the C).
        // Carboxylate O atoms are already claimed above, so they're naturally skipped.
        for &o_idx in &char.carbonyl {
            if claimed.contains(&o_idx) {
                continue;
            }
            let mut comp_atoms = vec![o_idx]; // key atom first (O)
            // Include the carbonyl C if it hasn't been claimed by a ring or chain yet.
            for &nb in &adj[o_idx] {
                if atoms[nb].element == Carbon && !claimed.contains(&nb) {
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
                let el = atoms[i].element;
                if el != Hydrogen {
                    add_comp!(ComponentType::Atom(el), vec![i]);
                }
            }
        }

        // --- Build connections ---
        // Every bond whose two endpoints belong to different components becomes a Connection.
        // `atom_0` / `atom_1` are positions within the respective component's `atoms` list.

        // Precompute how many detected rings each atom belongs to.
        // Atoms that appear in 2+ rings are bridgehead (fusion) atoms.
        let mut atom_ring_count: HashMap<usize, usize> = HashMap::new();
        for ring in &char.rings {
            for &a in &ring.atoms {
                *atom_ring_count.entry(a).or_insert(0) += 1;
            }
        }

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

            // A connection is at a ring fusion when both components are rings and at least
            // one endpoint is a bridgehead atom (present in 2+ detected rings). Because of
            // exclusive atom claiming, only one side of the fusion bond will be a bridgehead;
            // the other side is the non-bridgehead neighbour in the second ring component.
            let shared_atoms = matches!(comps[ca].comp_type, ComponentType::Ring(_))
                && matches!(comps[cb].comp_type, ComponentType::Ring(_))
                && (atom_ring_count.get(&a).copied().unwrap_or(0) > 1
                    || atom_ring_count.get(&b).copied().unwrap_or(0) > 1);

            conns.push(Connection {
                comp_0: ca,
                atom_0,
                comp_1: cb,
                atom_1,
                shared_atoms,
            });
        }

        Some(Self {
            components: comps,
            connections: conns,
        })
    }

    pub fn to_atoms_bonds(&self) -> (Vec<Atom>, Vec<Bond>) {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut bonds: Vec<Bond> = Vec::new();

        // Build each component's atoms/bonds and record where in the flat array it starts.
        let mut comp_offsets: Vec<usize> = Vec::with_capacity(self.components.len());
        for comp in &self.components {
            let offset = atoms.len();
            comp_offsets.push(offset);

            let (comp_atoms, comp_bonds) = comp.comp_type.to_atoms_bonds();

            // Shift intra-component bond indices to the global position.
            for cb in comp_bonds {
                bonds.push(Bond {
                    atom_0: cb.atom_0 + offset,
                    atom_1: cb.atom_1 + offset,
                    ..cb
                });
            }
            atoms.extend(comp_atoms);
        }

        // Add one bond per inter-component connection.
        // `con.atom_0/1` are positions within the respective component's local atom list,
        // so adding the component offset gives the global index.
        // Bond type is not stored in Connection; Single covers the common case.
        for con in &self.connections {
            let a0 = comp_offsets[con.comp_0] + con.atom_0;
            let a1 = comp_offsets[con.comp_1] + con.atom_1;
            bonds.push(Bond {
                bond_type: BondType::Single,
                atom_0_sn: 0, // fixed by reassign_sns below
                atom_1_sn: 0,
                atom_0: a0,
                atom_1: a1,
                is_backbone: false,
            });
        }

        let mut mol = MoleculeCommon::new(String::new(), atoms, bonds, HashMap::new(), None);
        mol.reassign_sns();

        (mol.atoms, mol.bonds)
    }
}

/// Mirrors the atoms/bonds based one.
pub fn build_adjacency_list_conn(conns: &[Connection], comps_len: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); comps_len];

    // For each conn, record its comps as neighbors of each other
    for conn in conns {
        // todo: Should we take into account the atoms joined at?
        result[conn.comp_0].push(conn.comp_1);
        result[conn.comp_1].push(conn.comp_0);
    }

    result
}
