//! An experiment to categorize small molecules, and find similar ones

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{Display, Formatter},
};

use bio_files::BondType;
use na_seq::Element::*;

use crate::molecules::common::MoleculeCommon;
use crate::molecules::rotatable_bonds::RotatableBond;

/// Describes a small molecule by features practical for description and characterization.
#[derive(Clone, Default, Debug)]
pub struct MolCharacterization {
    pub num_atoms: usize,
    pub num_bonds: usize,
    pub num_heavy_atoms: usize,
    pub num_hetero_atoms: usize,

    pub mol_weight: f32,

    pub num_rings_total: usize,
    // pub num_rings_5_atom: usize,
    // pub num_rings_6_atom: usize,
    /// Indices
    pub rings_5_atom: Vec<[usize; 5]>,
    pub rings_6_atom: Vec<[usize; 6]>,
    // pub num_aromatic_rings: usize,
    pub rings_aromatic_5_atom: Vec<[usize; 5]>,
    pub rings_aromatic_6_atom: Vec<[usize; 6]>,
    // pub num_aromatic_rings_5_atom: usize,
    // pub num_aromatic_rings_6_atom: usize,
    pub num_aromatic_atoms: usize,

    // pub num_rotatable_bonds: usize,
    /// Bond index.
    pub rotatable_bonds: Vec<RotatableBond>,

    pub num_carbon: usize,
    pub num_hydrogen: usize,
    pub nitrogen: Vec<usize>,
    pub oxygen: Vec<usize>,
    // pub num_sulfur: usize,
    pub sulfur: Vec<usize>,
    // pub num_phosphorus: usize,
    pub phosphorus: Vec<usize>,
    // pub num_fluorine: usize,
    pub fluorine: Vec<usize>,
    pub chlorine: Vec<usize>,
    pub bromine: Vec<usize>,
    // pub num_bromine: usize,
    // pub num_iodine: usize,
    pub iodine: Vec<usize>,
    pub halogen: Vec<usize>,
    // pub num_halogen: usize,

    // pub num_amines: usize,
    // pub num_amides: usize,
    /// N atom
    pub amines: Vec<usize>,
    /// N atom
    pub amides: Vec<usize>,
    // pub num_carbonyl: usize,
    /// C atom bound to O.
    pub carbonyl: Vec<usize>,
    /// O.
    pub hydroxyl: Vec<usize>,
    // pub num_hydroxyl: usize,

    pub h_bond_donor: Vec<usize>,
    pub h_bond_acceptor: Vec<usize>,

    pub net_partial_charge: Option<f32>,
    pub abs_partial_charge_sum: Option<f32>,

    pub num_sp3_carbon: usize,
    pub frac_csp3: f32,

    pub tpsa: Option<f32>,
    pub clogp: Option<f32>,
}

/// A helper to reduce repetition in string formatting.
fn count_disp(v: &mut String, count: usize, name: &str) {
    if count > 0 {
        *v += &format!(", {} {name}", count);
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
            self.mol_weight.round() as u16
        );

        count_disp(&mut v, self.rings_5_atom.len(), "pent");
        count_disp(&mut v, self.rings_6_atom.len(), "hex");


        count_disp(&mut v, self.amides.len(), "amide");
        count_disp(&mut v, self.amines.len(), "amine");
        count_disp(&mut v, self.carbonyl.len(), "carbonyl");
        count_disp(&mut v, self.hydroxyl.len(), "hydroxyl");
        count_disp(&mut v, self.sulfur.len(), "sulfur");
        count_disp(&mut v, self.phosphorus.len(), "phosphorus");
        count_disp(&mut v, self.chlorine.len(), "chlorine");

        writeln!(f, "{v}")
    }
}

impl MolCharacterization {
    pub fn new(mol: &MoleculeCommon) -> Self {
        fn edge_key(a: usize, b: usize) -> (usize, usize) {
            if a < b { (a, b) } else { (b, a) }
        }

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

        let mut halogen = Vec::with_capacity(fluorine.len() + chlorine.len() + bromine.len() + iodine.len());
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

        let num_rings_total = {
            let mut seen = vec![false; num_atoms];
            let mut components = 0usize;

            for i in 0..num_atoms {
                if seen[i] {
                    continue;
                }
                components += 1;
                let mut q = VecDeque::new();
                seen[i] = true;
                q.push_back(i);
                while let Some(u) = q.pop_front() {
                    for &v in &adj[u] {
                        if !seen[v] {
                            seen[v] = true;
                            q.push_back(v);
                        }
                    }
                }
            }

            let v = num_atoms as isize;
            let e = num_bonds as isize;
            let c = components as isize;

            let cyclomatic = e - v + c;
            if cyclomatic > 0 {
                cyclomatic as usize
            } else {
                0
            }
        };

        let rings_5_atom: Vec<[usize; 5]> = count_cycles_len(adj, 5)
            .into_iter().map(|v| v.try_into().unwrap()).collect();

        let rings_6_atom:Vec<[usize; 6]> = count_cycles_len(adj, 6)
            .into_iter().map(|v| v.try_into().unwrap()).collect();

        let is_kekule_aromatic_6c = |cyc: &[usize]| -> bool {
            if cyc.len() != 6 {
                return false;
            }
            for &a in cyc {
                if mol.atoms[a].element != Carbon {
                    return false;
                }
            }

            let mut kinds = [0u8; 6]; // 1=single, 2=double
            let mut singles = 0usize;
            let mut doubles = 0usize;

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

        let rings_aromatic_5_atom: Vec<_> = rings_5_atom.to_vec().into_iter().filter(|c| is_cycle_aromatic(c)).collect();
        let rings_aromatic_6_atom: Vec<_> = rings_6_atom.to_vec().into_iter().filter(|c| is_cycle_aromatic(c)).collect();

        // This logic takes fused rings into account.
        let num_aromatic_atoms = {
            let mut arom_atoms = HashSet::new();
            for r in & rings_aromatic_5_atom {
                arom_atoms.extend(r.iter().copied());
            }
            for r in &rings_aromatic_6_atom {
                arom_atoms.extend(r.iter().copied());
            }
            arom_atoms.len()
        };

        let mut bond_in_ring: HashMap<(usize, usize), bool> = HashMap::with_capacity(num_bonds);
        for b in &mol.bonds {
            let k = edge_key(b.atom_0, b.atom_1);
            let in_ring = bfs_reachable_ignoring_edge(adj, b.atom_0, b.atom_1, k);
            bond_in_ring.insert(k, in_ring);
        }

        let mut carbonyl = Vec::new();
        let mut hydroxyl = Vec::new();
        let mut amines = Vec::new();
        let mut amides = Vec::new();

        let mut hbd = 0;
        let mut hba = 0;

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
                .unwrap_or(true)
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
                    let has_c = adj[i].iter().any(|&n| mol.atoms[n].element == Carbon);
                    if has_c {
                        amines.push(i);
                    }
                }
            }

            let has_h = adj[i].iter().any(|&n| mol.atoms[n].element == Hydrogen);
            let positive = mol.atoms[i]
                .partial_charge
                .map(|q| q > 0.2)
                .unwrap_or(false);

            let donor = match el {
                Oxygen | Nitrogen | Sulfur => has_h,
                _ => false,
            };
            if donor {
                hbd += 1;
            }

            let acceptor = match el {
                Oxygen => !positive && !oxygen_is_carboxylic_oh(i),
                Nitrogen => !positive && !nitrogen_is_amide(i),
                Sulfur => !positive,
                _ => false,
            };
            if acceptor {
                hba += 1;
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

            num_rings_total,
            rings_5_atom,
            rings_6_atom,
            rings_aromatic_5_atom,
            rings_aromatic_6_atom,
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
            carbonyl,
            hydroxyl,

            h_bond_donor: hbd,
            h_bond_acceptor: hba,

            net_partial_charge,
            abs_partial_charge_sum,

            num_sp3_carbon,
            frac_csp3,

            tpsa: None,
            clogp: None,
        }
    }
}
