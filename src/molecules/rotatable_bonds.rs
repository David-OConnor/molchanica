//! For determining which bonds in a molecule allow free rotation, i.e. changing the dihedral
//! angle. For example, not rings.

use std::collections::VecDeque;

use bio_files::BondType;

use crate::molecules::common::MoleculeCommon;

#[derive(Clone, Debug)]
pub struct RotatableBond {
    pub bond_i: usize,
    pub downstream_from_a1: Vec<usize>,
}

impl MoleculeCommon {
    pub fn find_rotatable_bonds(&self) -> Vec<RotatableBond> {
        let mut out = Vec::new();

        for (i, b) in self.bonds.iter().enumerate() {
            if !matches!(b.bond_type, BondType::Single) {
                continue;
            }

            let a0 = b.atom_0;
            let a1 = b.atom_1;

            if self.adjacency_list[a0].len() <= 1 || self.adjacency_list[a1].len() <= 1 {
                continue;
            }

            if edge_in_ring(&self.adjacency_list, a0, a1) {
                continue;
            }

            let downstream = find_downstream_atoms(&self.adjacency_list, a0, a1);
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

fn edge_in_ring(adj: &[Vec<usize>], a: usize, b: usize) -> bool {
    fn edge_key(a: usize, b: usize) -> (usize, usize) {
        if a < b { (a, b) } else { (b, a) }
    }

    let ignore = edge_key(a, b);
    let mut q = VecDeque::new();
    let mut seen = vec![false; adj.len()];
    q.push_back(a);
    seen[a] = true;

    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if edge_key(u, v) == ignore {
                continue;
            }
            if !seen[v] {
                if v == b {
                    return true;
                }
                seen[v] = true;
                q.push_back(v);
            }
        }
    }

    false
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
