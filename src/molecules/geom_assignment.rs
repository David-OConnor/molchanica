//! Code for setting molecule geometry based on atoms and bonds. Can be used
//! to set initial geometry (e.g. from after loading from SMILES), or for correctly
//! freeform geometry from the molecule editor.

use std::collections::VecDeque;

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::molecules::{
    Bond,
    common::{BondGeom, MoleculeCommon, find_appended_posit},
};

/// Determine the local geometry at atom `i` from its bond types.
/// Triple bond → Linear; any double/aromatic bond → Planar; all single → Tetrahedral.
fn geom_for_atom(i: usize, bonds: &[Bond]) -> BondGeom {
    if bonds
        .iter()
        .any(|b| (b.atom_0 == i || b.atom_1 == i) && b.bond_type == BondType::Triple)
    {
        BondGeom::Linear
    } else if bonds.iter().any(|b| {
        (b.atom_0 == i || b.atom_1 == i)
            && matches!(b.bond_type, BondType::Double | BondType::Aromatic)
    }) {
        BondGeom::Planar
    } else {
        BondGeom::Tetrahedral
    }
}

/// Return the bond type between atoms `a` and `b`, defaulting to Single.
fn bond_type_between(a: usize, b: usize, bonds: &[Bond]) -> BondType {
    bonds
        .iter()
        .find(|bond| {
            (bond.atom_0 == a && bond.atom_1 == b) || (bond.atom_0 == b && bond.atom_1 == a)
        })
        .map(|bond| bond.bond_type)
        .unwrap_or(BondType::Single)
}

/// Estimate a covalent bond length (Å) from the elements' covalent radii and
/// the bond order.  A 0.5 Å floor handles unknown elements with radius 0.
fn estimate_bond_length(el0: Element, el1: Element, bt: BondType) -> f64 {
    let r0 = el0.covalent_radius().max(0.5);
    let r1 = el1.covalent_radius().max(0.5);
    let base = r0 + r1;
    match bt {
        BondType::Double => base * 0.87,
        BondType::Triple => base * 0.78,
        BondType::Aromatic => base * 0.91,
        _ => base,
    }
}

impl MoleculeCommon {
    /// Assign 3D coordinates to all atoms. This can be used when parsing atoms from SMILES,
    /// for example, as this format doesn't specify coordinates. This resets all positions to new
    /// absolute positions, and should not be used to adjust existing geometry.
    ///
    /// Algorithm: BFS from the first atom (placed at the origin).  For each
    /// unpositioned atom we call `find_appended_posit`, which computes a
    /// geometrically plausible position using the already-positioned neighbours
    /// of the parent for angular context (tetrahedral ≈109.5°, planar ≈120°,
    /// linear 180°).
    ///
    /// Ring-closure atoms are already positioned when the BFS reaches them via
    /// the "other path" around the ring and are simply skipped; the ring will
    /// be slightly strained but the bond connectivity is correct, and the energy
    /// minimiser will fix the geometry.
    ///
    /// Disconnected components (e.g. salt forms) are offset along the X axis so
    /// they do not overlap.
    pub fn assign_posits(&mut self) {
        let n = self.atoms.len();
        if n == 0 {
            return;
        }

        let mut positioned = vec![false; n];

        // Place first atom at the origin.
        self.atoms[0].posit = Vec3::new(0., 0., 0.);
        positioned[0] = true;

        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(0);

        // Lateral offset applied to each new disconnected component.
        let mut component_x = 0_f64;

        loop {
            // If the queue is empty, search for the next unpositioned atom
            // (start of a new disconnected component).
            if queue.is_empty() {
                match (0..n).find(|&i| !positioned[i]) {
                    None => break,
                    Some(start) => {
                        component_x += 5.;
                        self.atoms[start].posit = Vec3::new(component_x, 0., 0.);
                        positioned[start] = true;
                        queue.push_back(start);
                    }
                }
            }

            let u = match queue.pop_front() {
                Some(u) => u,
                None => continue,
            };

            let geom = geom_for_atom(u, &self.bonds);
            let posit_u = self.atoms[u].posit; // Copy — Vec3 is Copy

            // Clone the neighbour list to avoid borrow conflicts while mutating atoms.
            let neighbours: Vec<usize> = self.adjacency_list[u].to_vec();

            for v in neighbours {
                if positioned[v] {
                    // Already placed (either placed earlier in BFS, or a ring-closure
                    // partner placed via the other path around the ring).  Skip.
                    continue;
                }

                // Collect already-positioned neighbours of u (excluding v itself) to
                // give find_appended_posit its angular context.
                let adj_placed: Vec<usize> = self.adjacency_list[u]
                    .iter()
                    .copied()
                    .filter(|&w| w != v && positioned[w])
                    .collect();

                let bt = bond_type_between(u, v, &self.bonds);
                let bond_len =
                    estimate_bond_length(self.atoms[u].element, self.atoms[v].element, bt);

                let p = find_appended_posit(
                    posit_u,
                    &self.atoms,
                    &adj_placed,
                    Some(bond_len),
                    self.atoms[v].element,
                    geom,
                )
                .unwrap_or_else(|| posit_u + Vec3::new(bond_len, 0., 0.));

                self.atoms[v].posit = p;
                positioned[v] = true;
                queue.push_back(v);
            }
        }

        self.reset_posits();
    }

    /// This can be used, for example, by the editor to clean up free-form drawn geometry.
    /// this can produce similar results to the MD engine's energy minimization function, but
    /// is cheaper, and can work in cases where there is pathological geometry.
    ///
    /// Unlike `assign_posits`, this takes into account the existing positions (global and local),
    /// and uses them as a basis for the updated positions.
    pub fn cleanup_geometry(&mut self) {
        self.reset_posits();
    }
}
