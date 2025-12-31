//! For determining which bonds in a molecule allow free rotation, i.e. changing the dihedral
//! angle. For example, not rings.

use std::collections::VecDeque;

use bio_files::BondType;

use crate::molecules::{Bond, common::MoleculeCommon};

#[derive(Clone, Debug)]
pub struct RotatableBond {
    pub bond_i: usize,
    pub downstream_from_a1: Vec<usize>,
}

impl MoleculeCommon {
    pub fn find_rotatable_bonds(&self) -> Vec<RotatableBond> {
        let mut out = Vec::new();

        for (i, bond) in self.bonds.iter().enumerate() {
            if !matches!(bond.bond_type, BondType::Single) {
                continue;
            }

            let atom_0 = bond.atom_0;
            let atom_1 = bond.atom_1;

            if self.adjacency_list[atom_0].len() <= 1 || self.adjacency_list[atom_1].len() <= 1 {
                continue;
            }

            if bond.in_a_cycle(&self.adjacency_list) {
                continue;
            }

            let downstream = find_downstream_atoms(&self.adjacency_list, atom_0, atom_1);
            if downstream.is_empty() || downstream.len() == self.atoms.len() {
                continue;
            }

            out.push(RotatableBond {
                bond_i: i,
                downstream_from_a1: downstream,
            });
        }

        out
    }
}

/// We use this to rotate molecules around a bond pivot. For example, by the user directly, or
/// in ligand conformation algorithms that try different poses.
///
/// `pivot` and `side` are atom indices.
pub fn find_downstream_atoms(adj_list: &[Vec<usize>], pivot: usize, side: usize) -> Vec<usize> {
    if pivot == side {
        return Vec::new();
    }

    let mut visited = vec![false; adj_list.len()];
    visited[pivot] = true;

    // A bit of pre-allocation.
    let mut stack = Vec::with_capacity(16);
    let mut result = Vec::with_capacity(16);

    visited[side] = true;
    stack.push(side);

    while let Some(u) = stack.pop() {
        result.push(u);

        for &v in &adj_list[u] {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }

    result
}

/// Determines if a bond is part of a cycle (ring). This has implications, for example, in determining
/// if it can be used as a rotation pivot.
impl Bond {
    pub fn in_a_cycle(&self, adj_list: &[Vec<usize>]) -> bool {
        let a0 = self.atom_0;
        let a1 = self.atom_1;

        if a0 == a1 {
            return false;
        }

        let mut visited = vec![false; adj_list.len()];
        let mut stack = Vec::with_capacity(16);

        visited[a0] = true;
        stack.push(a0);

        while let Some(cur) = stack.pop() {
            if cur == a1 {
                return true;
            }

            for &nbr in &adj_list[cur] {
                if (cur == a0 && nbr == a1) || (cur == a1 && nbr == a0) {
                    continue;
                }
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }

        false
    }
}
