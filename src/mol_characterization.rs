//! An experiment to categorize small molecules, and find similar ones

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{Display, Formatter},
};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::Element::*;

use crate::molecules::{Atom, common::MoleculeCommon, rotatable_bonds::RotatableBond};

/// Describes a small molecule by features practical for description and characterization.
#[derive(Clone, Default, Debug)]
pub struct MolCharacterization {
    pub num_atoms: usize,
    pub num_bonds: usize,
    pub num_heavy_atoms: usize,
    pub num_hetero_atoms: usize,
    pub num_aromatic_atoms: usize,
    pub mol_weight: f32,
    /// Single rings; either standalone, or part of a system.
    pub rings: Vec<Ring>,
    /// Fused rings, i.e. 2 or more rings with shared edges.
    /// These are indices of the `rings` field here. Note that we only have
    /// one level of ring systems. For example. 3 rings that have shared edges are one system of len 3;
    /// we don't also include the len-2 subsystems.
    pub ring_systems: Vec<Vec<usize>>,
    /// Bond index.
    pub rotatable_bonds: Vec<RotatableBond>,
    pub num_carbon: usize,
    pub num_hydrogen: usize,
    pub nitrogen: Vec<usize>,
    pub oxygen: Vec<usize>,
    pub sulfur: Vec<usize>,
    pub phosphorus: Vec<usize>,
    pub fluorine: Vec<usize>,
    pub chlorine: Vec<usize>,
    pub bromine: Vec<usize>,
    pub iodine: Vec<usize>,
    pub halogen: Vec<usize>,
    /// N atom
    pub amines: Vec<usize>,
    /// N atom
    pub amides: Vec<usize>,
    /// Lone pair not part of the aromatic sextet
    pub pyridine_like_aromatic_n: Vec<usize>,
    /// Lone pair is part of the aromatic sextet
    pub pyrrole_like_nh: Vec<usize>,
    /// Has a C=N double bond
    pub imine_like_n: Vec<usize>,
    /// C atom bound to O.
    pub carbonyl: Vec<usize>,
    /// O atom index.
    pub hydroxyl: Vec<usize>,
    pub h_bond_donor: Vec<usize>,
    pub h_bond_acceptor: Vec<usize>,
    pub net_partial_charge: Option<f32>,
    pub abs_partial_charge_sum: Option<f32>,
    pub num_sp3_carbon: usize,
    pub frac_csp3: f32,
    // todo
    pub topological_polar_surface_area: Option<f32>,
    pub calc_log_p: Option<f32>,
}

/// Represents a single ring; can be on its own, or part of a fused ring system.
#[derive(Clone, Debug)]
pub struct Ring {
    pub atoms: Vec<usize>,
    pub aromatic: bool,
    /// Note: This is rel to `Atom.posit; not `atom_posits`, as that can easily change.
    /// If `Atom.posit` changes, this must be updated.
    pub plane_norm: Vec3,
}

impl Ring {
    pub fn center(&self, atoms: &[Atom]) -> Vec3 {
        let mut sum = Vec3::new_zero();
        for i in &self.atoms {
            sum += atoms[*i].posit;
        }

        sum / self.atoms.len() as f64
    }
}

/// A helper to reduce repetition in string formatting.
fn count_disp(v: &mut String, count: usize, name: &str) {
    if count > 0 {
        *v += &format!(", {} {name}", count);
        // For plurals.
        // if count >= 2 {
        //     *v += "s";
        // }
    }
}

impl Display for MolCharacterization {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut v = format!(
            "#: {} Wt: {}",
            self.num_atoms,
            self.mol_weight.round() as u32
        );

        let mut rings_5 = 0;
        let mut rings_6 = 0;
        for ring in &self.rings {
            if ring.atoms.len() == 5 {
                rings_5 += 1;
            } else if ring.atoms.len() == 6 {
                rings_6 += 1;
            }
        }

        count_disp(&mut v, rings_5, "⬟");
        count_disp(&mut v, rings_6, "⬣");
        count_disp(&mut v, self.ring_systems.len(), "fused ring systems");

        count_disp(&mut v, self.amides.len(), "amide");
        count_disp(&mut v, self.amines.len(), "amine");

        count_disp(&mut v, self.pyridine_like_aromatic_n.len(), "pyridine-like");
        count_disp(&mut v, self.pyrrole_like_nh.len(), "pyrrole-like");
        count_disp(&mut v, self.imine_like_n.len(), "imine-like");

        count_disp(&mut v, self.carbonyl.len(), "carbonyl");
        count_disp(&mut v, self.hydroxyl.len(), "hydroxyl");

        count_disp(&mut v, self.sulfur.len(), "S");
        count_disp(&mut v, self.phosphorus.len(), "P");
        count_disp(&mut v, self.chlorine.len(), "Cl");
        count_disp(&mut v, self.halogen.len(), "halogen");

        writeln!(f, "{v}")
    }
}

impl MolCharacterization {
    pub fn new(mol: &MoleculeCommon) -> Self {
        fn bfs_reachable_ignoring_edge(
            adj: &[Vec<usize>],
            start: usize,
            goal: usize,
            ignore: (usize, usize),
        ) -> bool {
            let mut q = VecDeque::new();
            let mut seen = vec![false; adj.len()];
            seen[start] = true;
            q.push_back(start);

            while let Some(u) = q.pop_front() {
                if u == goal {
                    return true;
                }
                for &v in &adj[u] {
                    if edge_key(u, v) == ignore {
                        continue;
                    }
                    if !seen[v] {
                        seen[v] = true;
                        q.push_back(v);
                    }
                }
            }
            false
        }

        let num_atoms = mol.atoms.len();
        let num_bonds = mol.bonds.len();

        let mut bond_type_by_edge: HashMap<(usize, usize), BondType> =
            HashMap::with_capacity(num_bonds);
        for b in &mol.bonds {
            bond_type_by_edge.insert(edge_key(b.atom_0, b.atom_1), b.bond_type);
        }

        let mut mol_weight_f64 = 0.0f64;

        let mut num_carbon = 0;
        let mut num_hydrogen = 0;
        let mut nitrogen = Vec::new();
        let mut oxygen = Vec::new();
        let mut sulfur = Vec::new();
        let mut phosphorus = Vec::new();

        let mut fluorine = Vec::new();
        let mut chlorine = Vec::new();
        let mut bromine = Vec::new();
        let mut iodine = Vec::new();

        let mut num_heavy_atoms = 0;
        let mut num_hetero_atoms = 0;

        let mut all_charges_present = true;
        let mut net_q = 0.0f32;
        let mut abs_q = 0.0f32;

        for (i, atom) in mol.atoms.iter().enumerate() {
            mol_weight_f64 += atom.element.atomic_weight() as f64;

            if atom.element != Hydrogen {
                num_heavy_atoms += 1;
            }
            if atom.element != Hydrogen && atom.element != Carbon {
                num_hetero_atoms += 1;
            }

            match atom.element {
                Carbon => num_carbon += 1,
                Hydrogen => num_hydrogen += 1,
                Nitrogen => nitrogen.push(i),
                Oxygen => oxygen.push(i),
                Sulfur => sulfur.push(i),
                Phosphorus => phosphorus.push(i),

                Fluorine => fluorine.push(i),
                Chlorine => chlorine.push(i),
                Bromine => bromine.push(i),
                Iodine => iodine.push(i),
                _ => {}
            }

            match atom.partial_charge {
                Some(q) => {
                    net_q += q;
                    abs_q += q.abs();
                }
                None => {
                    all_charges_present = false;
                }
            }
        }

        let mut halogen =
            Vec::with_capacity(fluorine.len() + chlorine.len() + bromine.len() + iodine.len());
        halogen.extend(&fluorine);
        halogen.extend(&chlorine);
        halogen.extend(&bromine);
        halogen.extend(&iodine);

        let net_partial_charge = if all_charges_present && num_atoms > 0 {
            Some(net_q)
        } else {
            None
        };
        let abs_partial_charge_sum = if all_charges_present && num_atoms > 0 {
            Some(abs_q)
        } else {
            None
        };

        let adj = &mol.adjacency_list;

        let rings = rings(&adj, &mol.atoms, &bond_type_by_edge);
        let ring_systems = fused_rings(&rings, &mol.atoms);

        let aromatic_atoms: HashSet<usize> = rings
            .iter()
            .filter(|r| r.aromatic)
            .flat_map(|r| r.atoms.iter().copied())
            .collect();

        let num_aromatic_atoms = aromatic_atoms.len();

        let ring_systems = fused_rings(&rings, &mol.atoms);

        let mut carbonyl = Vec::new();
        let mut hydroxyl = Vec::new();
        let mut amines = Vec::new();
        let mut amides = Vec::new();
        let mut pyridine_like_aromatic_n = Vec::new();
        let mut pyrrole_like_nh = Vec::new();
        let mut imine_like_n = Vec::new();

        let mut h_bond_donor = Vec::new();
        let mut h_bond_acceptor = Vec::new();

        let is_double_bond = |a: usize, b: usize| -> bool {
            bond_type_by_edge
                .get(&edge_key(a, b))
                .map(|bt| *bt == BondType::Double)
                .unwrap_or(false)
        };

        let is_single_non_arom = |a: usize, b: usize| -> bool {
            bond_type_by_edge
                .get(&edge_key(a, b))
                .map(|bt| *bt == BondType::Single)
                .unwrap_or(false)
        };

        let carbon_has_double_bonded_oxygen = |c: usize| -> bool {
            if mol.atoms[c].element != Carbon {
                return false;
            }
            for &n in &adj[c] {
                if mol.atoms[n].element == Oxygen && is_double_bond(c, n) {
                    return true;
                }
            }
            false
        };

        let oxygen_is_carboxylic_oh = |o: usize| -> bool {
            if mol.atoms[o].element != Oxygen {
                return false;
            }
            let mut has_h = false;
            let mut carbon_neighbor: Option<usize> = None;
            for &n in &adj[o] {
                match mol.atoms[n].element {
                    Hydrogen => has_h = true,
                    Carbon => carbon_neighbor = Some(n),
                    _ => {}
                }
            }
            if !has_h {
                return false;
            }
            let Some(c) = carbon_neighbor else {
                return false;
            };
            if !is_single_non_arom(o, c) {
                return false;
            }
            carbon_has_double_bonded_oxygen(c)
        };

        let nitrogen_is_amide = |n_i: usize| -> bool {
            if mol.atoms[n_i].element != Nitrogen {
                return false;
            }
            for &nbr in &adj[n_i] {
                if mol.atoms[nbr].element == Carbon
                    && is_single_non_arom(n_i, nbr)
                    && carbon_has_double_bonded_oxygen(nbr)
                {
                    return true;
                }
            }
            false
        };

        for i in 0..num_atoms {
            let el = mol.atoms[i].element;

            if el == Carbon && carbon_has_double_bonded_oxygen(i) {
                carbonyl.push(i);
            }

            if el == Oxygen {
                let has_h = adj[i].iter().any(|&n| mol.atoms[n].element == Hydrogen);
                if has_h {
                    hydroxyl.push(i);
                }
            }

            if el == Nitrogen {
                let amide = nitrogen_is_amide(i);
                if amide {
                    amides.push(i);
                } else {
                    let in_aromatic_ring = aromatic_atoms.contains(&i);
                    let has_h = adj[i].iter().any(|&j| mol.atoms[j].element == Hydrogen);

                    let has_c_single = adj[i]
                        .iter()
                        .any(|&j| mol.atoms[j].element == Carbon && is_single_non_arom(i, j));

                    let has_c_double = adj[i]
                        .iter()
                        .any(|&j| mol.atoms[j].element == Carbon && is_double_bond(i, j));

                    if in_aromatic_ring {
                        if has_h {
                            pyrrole_like_nh.push(i);
                        } else {
                            pyridine_like_aromatic_n.push(i);
                        }
                    } else if has_c_double {
                        imine_like_n.push(i);
                    } else if has_c_single {
                        amines.push(i);
                    }
                }
            }

            let has_h = adj[i].iter().any(|&n| mol.atoms[n].element == Hydrogen);

            let donor = match el {
                Oxygen | Nitrogen | Sulfur => has_h,
                _ => false,
            };
            if donor {
                h_bond_donor.push(i);
            }

            let acceptor = match el {
                Oxygen => !oxygen_is_carboxylic_oh(i),
                Nitrogen => !nitrogen_is_amide(i),
                Sulfur => true,
                _ => false,
            };

            if acceptor {
                h_bond_acceptor.push(i);
            }
        }

        let mut num_sp3_carbon = 0usize;
        for i in 0..num_atoms {
            if mol.atoms[i].element != Carbon {
                continue;
            }
            let mut ok = true;
            for &j in &adj[i] {
                let Some(bt) = bond_type_by_edge.get(&edge_key(i, j)) else {
                    continue;
                };
                if *bt != BondType::Single {
                    ok = false;
                    break;
                }
            }
            if ok {
                num_sp3_carbon += 1;
            }
        }
        let frac_csp3 = if num_carbon > 0 {
            num_sp3_carbon as f32 / num_carbon as f32
        } else {
            0.0
        };

        Self {
            num_atoms,
            num_bonds,
            num_heavy_atoms,
            num_hetero_atoms,
            mol_weight: mol_weight_f64 as f32,
            rings,
            ring_systems,
            num_aromatic_atoms,
            rotatable_bonds: mol.find_rotatable_bonds(),
            num_carbon,
            num_hydrogen,
            nitrogen,
            oxygen,
            sulfur,
            phosphorus,
            fluorine,
            chlorine,
            bromine,
            iodine,
            halogen,

            amines,
            amides,
            pyridine_like_aromatic_n,
            pyrrole_like_nh,
            imine_like_n,
            carbonyl,
            hydroxyl,

            h_bond_donor,
            h_bond_acceptor,

            net_partial_charge,
            abs_partial_charge_sum,

            num_sp3_carbon,
            frac_csp3,

            topological_polar_surface_area: None,
            calc_log_p: None,
        }
    }
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn canonical_cycle(nodes: &[usize]) -> Vec<usize> {
    let n = nodes.len();

    let (min_i, _) = nodes.iter().enumerate().min_by_key(|(_, v)| **v).unwrap();

    let mut rot_fwd = Vec::with_capacity(n);
    for k in 0..n {
        rot_fwd.push(nodes[(min_i + k) % n]);
    }

    let mut rev = Vec::with_capacity(n);
    for k in 0..n {
        rev.push(nodes[(min_i + n - (k % n)) % n]);
    }

    if rev < rot_fwd { rev } else { rot_fwd }
}

fn count_cycles_len(adj: &[Vec<usize>], len: usize) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut cycles_set: HashSet<Vec<usize>> = HashSet::new();
    let mut stack: Vec<usize> = Vec::with_capacity(len);
    let mut visited = vec![false; n];

    fn dfs(
        adj: &[Vec<usize>],
        s: usize,
        u: usize,
        len: usize,
        stack: &mut Vec<usize>,
        visited: &mut [bool],
        cycles_set: &mut HashSet<Vec<usize>>,
    ) {
        if stack.len() == len {
            if adj[u].iter().any(|&v| v == s) {
                let cyc = canonical_cycle(stack);
                cycles_set.insert(cyc);
            }
            return;
        }

        for &v in &adj[u] {
            if v == s {
                continue;
            }
            if visited[v] {
                continue;
            }
            if v < s {
                continue;
            }

            visited[v] = true;
            stack.push(v);
            dfs(adj, s, v, len, stack, visited, cycles_set);
            stack.pop();
            visited[v] = false;
        }
    }

    for s in 0..n {
        visited[s] = true;
        stack.clear();
        stack.push(s);

        for &v in &adj[s] {
            if v < s {
                continue;
            }
            visited[v] = true;
            stack.push(v);
            dfs(adj, s, v, len, &mut stack, &mut visited, &mut cycles_set);
            stack.pop();
            visited[v] = false;
        }

        visited[s] = false;
    }

    cycles_set.into_iter().collect()
}

/// Ring plane normal using Newell's method. This is more robust than taking the cross product
/// of 2 arbitrary atom pairs, due to the possibility of the ring plane not being perfectly flat.
/// Assumes `ring` atom indices are in cyclic order.
fn find_plane_norm(ring: &[usize], atoms: &[Atom]) -> Vec3 {
    let n = ring.len();
    debug_assert!(n >= 3);

    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;

    for i in 0..n {
        let p0 = atoms[ring[i]].posit;
        let p1 = atoms[ring[(i + 1) % n]].posit;

        nx += (p0.y - p1.y) * (p0.z + p1.z);
        ny += (p0.z - p1.z) * (p0.x + p1.x);
        nz += (p0.x - p1.x) * (p0.y + p1.y);
    }

    let norm = Vec3::new(nx, ny, nz);
    let len2 = norm.dot(norm);
    if len2 == 0.0 {
        // Fallback: try any non-degenerate triple (keeps behavior similar to your original)
        for a in 0..n {
            let p0 = atoms[ring[a]].posit;
            let p1 = atoms[ring[(a + 1) % n]].posit;
            let p2 = atoms[ring[(a + 2) % n]].posit;
            let v0 = p1 - p0;
            let v1 = p2 - p0;
            let cr = v0.cross(v1);
            if cr.dot(cr) != 0.0 {
                return cr.to_normalized();
            }
        }
        // Truly degenerate; return something stable.
        return Vec3::new(0.0, 0.0, 1.0);
    }

    norm.to_normalized()
}
/// Identify all rings in the molecule, and count the total number of aromatic atoms.
fn rings(
    adj: &[Vec<usize>],
    atoms: &[Atom],
    bond_type_by_edge: &HashMap<(usize, usize), BondType>,
) -> Vec<Ring> {
    let rings_5_atom: Vec<[usize; 5]> = count_cycles_len(adj, 5)
        .into_iter()
        .map(|v| v.try_into().unwrap())
        .collect();

    let rings_6_atom: Vec<[usize; 6]> = count_cycles_len(adj, 6)
        .into_iter()
        .map(|v| v.try_into().unwrap())
        .collect();

    let is_kekule_aromatic_6c = |cyc: &[usize]| -> bool {
        if cyc.len() != 6 {
            return false;
        }
        for &a in cyc {
            match atoms[a].element {
                Carbon | Nitrogen | Oxygen | Sulfur => {}
                _ => return false,
            }
        }

        let mut kinds = [0; 6]; // 1=single, 2=double
        let mut singles = 0;
        let mut doubles = 0;

        for k in 0..6 {
            let a = cyc[k];
            let b = cyc[(k + 1) % 6];
            let Some(bt) = bond_type_by_edge.get(&edge_key(a, b)) else {
                return false;
            };

            match *bt {
                BondType::Single => {
                    kinds[k] = 1;
                    singles += 1;
                }
                BondType::Double => {
                    kinds[k] = 2;
                    doubles += 1;
                }
                BondType::Aromatic => return true,
                _ => return false,
            }
        }

        if singles != 3 || doubles != 3 {
            return false;
        }

        for k in 0..6 {
            if kinds[k] == kinds[(k + 1) % 6] {
                return false;
            }
        }

        true
    };

    let is_cycle_aromatic = |cyc: &[usize]| -> bool {
        let n = cyc.len();
        let mut all_bt_arom = true;

        for k in 0..n {
            let a = cyc[k];
            let b = cyc[(k + 1) % n];
            let Some(bt) = bond_type_by_edge.get(&edge_key(a, b)) else {
                return false;
            };
            if *bt != BondType::Aromatic {
                all_bt_arom = false;
                break;
            }
        }

        if all_bt_arom {
            return true;
        }

        is_kekule_aromatic_6c(cyc)
    };

    let mut result = Vec::new();
    for ring in rings_5_atom {
        result.push(Ring {
            atoms: ring.to_vec(),
            aromatic: is_cycle_aromatic(&ring),
            plane_norm: find_plane_norm(&ring, atoms),
        })
    }

    for ring in rings_6_atom {
        result.push(Ring {
            atoms: ring.to_vec(),
            aromatic: is_cycle_aromatic(&ring),
            plane_norm: find_plane_norm(&ring, &atoms),
        })
    }

    result
}

fn fused_rings(rings: &[Ring], _atoms: &[Atom]) -> Vec<Vec<usize>> {
    fn ring_edges(r: &Ring) -> HashSet<(usize, usize)> {
        let n = r.atoms.len();
        let mut e = HashSet::with_capacity(n);
        for k in 0..n {
            let a = r.atoms[k];
            let b = r.atoms[(k + 1) % n];
            e.insert(edge_key(a, b));
        }
        e
    }

    let n = rings.len();
    if n == 0 {
        return Vec::new();
    }

    // Precompute edges for each ring.
    let edges: Vec<HashSet<(usize, usize)>> = rings.iter().map(ring_edges).collect();

    // Build ring adjacency graph: edge-sharing => fused adjacency.
    let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if edges[i].intersection(&edges[j]).next().is_some() {
                radj[i].push(j);
                radj[j].push(i);
            }
        }
    }

    // Connected components; keep only components with 2+ rings.
    let mut seen = vec![false; n];
    let mut systems: Vec<Vec<usize>> = Vec::new();

    for s in 0..n {
        if seen[s] {
            continue;
        }

        let mut q = VecDeque::new();
        let mut comp = Vec::new();
        seen[s] = true;
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &v in &radj[u] {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }

        if comp.len() >= 2 {
            comp.sort_unstable();
            systems.push(comp);
        }
    }

    // Deterministic output ordering: sort systems by their smallest ring index, then lexicographically.
    systems.sort();
    systems
}
