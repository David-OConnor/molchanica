//! For identifying Tautomers

use std::collections::HashSet;

use bio_files::BondType::{Double, Single};
use na_seq::Element::*;

use crate::molecules::{Atom, Bond, common::MoleculeCommon};

const MAX_PATH_LEN_BONDS: usize = 8;
const MAX_TAUTOMERS: usize = 512;

impl MoleculeCommon {
    /// Find tautomers of this molecule. These are structural isomers that readily interconvert.
    ///It shifts around single vs double bonds, and which corresponding atoms have hydrogens bound.
    ///
    /// Notes:
    /// - This version enumerates *explicit-H* prototropic tautomers (it requires H atoms in `atoms`).
    /// - It moves H along alternating single/double bond paths and flips bond orders along that path.
    pub fn find_tautomers(&self) -> Vec<MoleculeCommon> {
        // Map (min,max) -> bond index for fast lookups.
        let mut bond_ix = std::collections::HashMap::<(usize, usize), usize>::new();
        for (i, b) in self.bonds.iter().enumerate() {
            bond_ix.insert(edge_key(b.atom_0, b.atom_1), i);
        }

        // Identify explicit H atoms that are bound to a hetero atom donor.
        let mut h_donors: Vec<(usize, usize, usize)> = Vec::new(); // (h, donor, bond_idx)
        for h in 0..self.atoms.len() {
            if self.atoms[h].element != Hydrogen {
                continue;
            }
            if self.adjacency_list[h].len() != 1 {
                continue;
            }
            let donor = self.adjacency_list[h][0];

            if !is_hetero(&self.atoms[donor]) {
                continue;
            }
            if let Some(&bidx) = bond_ix.get(&edge_key(h, donor)) {
                // Only allow H-single bonds.
                if bond_order(&self.bonds[bidx]) == Some(1) {
                    h_donors.push((h, donor, bidx));
                }
            }
        }

        let mut out = Vec::<MoleculeCommon>::new();
        let mut seen = HashSet::<Vec<(usize, usize, u8)>>::new();

        for (h, donor, hbond_idx) in h_donors {
            let paths = enumerate_paths(self, &bond_ix, donor, MAX_PATH_LEN_BONDS);

            for p in paths {
                if out.len() >= MAX_TAUTOMERS {
                    return out;
                }

                let acceptor = *p.last().unwrap();
                if acceptor == donor {
                    continue;
                }

                // Create tautomer by:
                // - moving the H from donor to acceptor
                // - flipping single<->double along each heavy bond on the donor..acceptor path
                let mut m2 = self.clone();

                // Move H bond endpoint to acceptor (keep it single).
                {
                    let b = &mut m2.bonds[hbond_idx];
                    if b.atom_0 == h {
                        b.atom_1 = acceptor;
                    } else if b.atom_1 == h {
                        b.atom_0 = acceptor;
                    } else {
                        // If the bond structure differs, adjust this block.
                        continue;
                    }
                    // Ensure single H-bond.
                    // (If your BondType supports it, this keeps it single.)
                    // If you want to preserve existing type, remove this.
                    // (No-op if already single.)
                    // set_bond_order(b, 1);
                }

                // Flip bond orders along the path (donor=a0, ..., acceptor=ak)
                let mut ok = true;
                for w in p.windows(2) {
                    let a = w[0];
                    let c = w[1];
                    let Some(&bidx) = bond_ix.get(&edge_key(a, c)) else {
                        ok = false;
                        break;
                    };
                    let Some(ord) = bond_order(&m2.bonds[bidx]) else {
                        ok = false;
                        break;
                    };
                    let new_ord = if ord == 1 { 2 } else { 1 };
                    set_bond_order(&mut m2.bonds[bidx], new_ord);
                }
                if !ok {
                    continue;
                }

                // Rebuild adjacency_list to match bonds after edits.
                m2.build_adjacency_list();

                let sig = molecule_signature(&m2);
                if seen.insert(sig) {
                    out.push(m2);
                }
            }
        }

        out
    }
}

fn is_hetero(atom: &Atom) -> bool {
    matches!(atom.element, Nitrogen | Oxygen | Sulfur)
}

fn bond_order(b: &Bond) -> Option<u8> {
    use bio_files::BondType::*;
    match b.bond_type {
        Single => Some(1),
        Double => Some(2),
        _ => None,
    }
}

fn set_bond_order(b: &mut Bond, order: u8) {
    use bio_files::BondType::*;
    b.bond_type = match order {
        1 => Single,
        2 => Double,
        _ => b.bond_type,
    };
}

fn molecule_signature(m: &MoleculeCommon) -> Vec<(usize, usize, u8)> {
    let mut v: Vec<(usize, usize, u8)> = m
        .bonds
        .iter()
        .map(|b| {
            let (a, c) = edge_key(b.atom_0, b.atom_1);
            let ord = bond_order(b).unwrap_or(0);
            (a, c, ord)
        })
        .collect();
    v.sort_unstable();
    v
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn rec(
    mol: &MoleculeCommon,
    bond_ix: &std::collections::HashMap<(usize, usize), usize>,
    cur: usize,
    start: usize,
    last_order: Option<u8>,
    max_len_bonds: usize,
    visited: &mut [bool],
    path: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    // path contains atoms, bonds count is path.len()-1
    let bonds_len = path.len().saturating_sub(1);
    if bonds_len > max_len_bonds {
        return;
    }

    if cur != start && is_hetero(&mol.atoms[cur]) && bonds_len >= 2 {
        out.push(path.clone());
    }

    for &nbr in &mol.adjacency_list[cur] {
        // Skip H atoms; we only walk the heavy-atom graph for tautomeric paths.
        if matches!(mol.atoms[nbr].element, Hydrogen) {
            continue;
        }
        if visited[nbr] {
            continue;
        }

        let Some(&bidx) = bond_ix.get(&edge_key(cur, nbr)) else {
            continue;
        };
        let Some(ord) = bond_order(&mol.bonds[bidx]) else {
            continue;
        };

        if let Some(last) = last_order {
            if ord == last {
                continue; // enforce alternation
            }
        }

        visited[nbr] = true;
        path.push(nbr);
        rec(
            mol,
            bond_ix,
            nbr,
            start,
            Some(ord),
            max_len_bonds,
            visited,
            path,
            out,
        );
        path.pop();
        visited[nbr] = false;
    }
}

// DFS that enumerates alternating single/double paths from donor to hetero acceptors.
fn enumerate_paths(
    mol: &MoleculeCommon,
    bond_ix: &std::collections::HashMap<(usize, usize), usize>,
    start: usize,
    max_len_bonds: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut visited = vec![false; mol.atoms.len()];
    let mut path = Vec::<usize>::new();

    visited[start] = true;
    path.push(start);
    rec(
        mol,
        bond_ix,
        start,
        start,
        None,
        max_len_bonds,
        &mut visited,
        &mut path,
        &mut out,
    );
    out
}
