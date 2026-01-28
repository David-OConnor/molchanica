//! See the description of the struct.
//! Note: (AqSolDB data)[https://github.com/mcsorkun/AqSolDB] may be a good source to validate these.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{Display, Formatter},
    time::Instant,
};

use bio_files::BondType;
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::{Element, Element::*};

use crate::{
    molecules::{Atom, Bond, common::MoleculeCommon, rotatable_bonds::RotatableBond},
    sa_surface::{SOLVENT_RAD, make_sas_mesh},
};

/// Describes a small molecule by features practical for description and characterization.
/// These properties are derived quickly from a molecule, and are simple and objective.
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
    /// These ring counts are conveniences, derived from `rings`.
    pub num_rings_aromatic: usize,
    pub num_rings_saturated: usize,
    pub num_rings_aliphatic: usize,
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
    /// These charges are None if we are missing any partial charges.
    pub net_partial_charge: Option<f32>,
    pub abs_partial_charge_sum: Option<f32>,
    pub num_sp3_carbon: usize,
    pub frac_csp3: f32,
    /// Total polar surface area. A 2D/topology-based estimate (Å²) of how much of the molecule’s surface is polar (mainly
    /// contributions from N, O, S, P). Correlates with permeability/absorption and hydrogen bonding.
    /// This is the Ertl approach similar to RDKit, and is present in data sets.
    pub tpsa_ertl: f32,
    /// Uses geometry to compute TPSA. I think this will be more accurate, as it takes into account
    /// 3D geometry, but may be less effective in ML contexts, or when comparing to other data
    /// in general, as they don't usually use this.
    pub psa_topo: f32,
    /// The (calculated) log10 of the partition coefficient P between octanol and water for the
    /// neutral compound. Higher logP generally means more hydrophobic/lipophilic.
    /// This version is internally calculated; this is relevant for the purposes of ML training,
    /// even if we use the pubchem val elsewhere.
    pub log_p: f32,
    pub log_p_pubchem: Option<f32>,
    /// A measure related to the molecule’s polarizability and volume (often derived alongside logP
    /// in fragment methods like Wildman–Crippen). Higher MR usually means “bigger/more polarizable.”
    pub molar_refractivity: f32,
    pub num_valence_elecs: usize, // todo: Impl
    // todo: Topological ASA too;
    /// Solvent accessible surface area (ASA).
    pub asa_labute: f32,
    pub asa_topo: f32,
    /// The Balaban J index: a topological connectivity descriptor derived from the graph distance
    /// matrix. Captures aspects of branching/compactness and overall graph structure.
    pub balaban_j: f32,
    /// Bertz complexity index (Cₜ), a graph-based measure intended to quantify molecular structural
    /// complexity (branching, ring structure, heteroatom variety). Higher means “more complex.”
    pub bertz_ct: f32,
    /// We load volume and complexity from PubChem's API currently.
    /// Analytic volume of the first diverse conformer (default conformer) for a compound.
    pub volume: f32,
    /// Note: We currently display PubChem vol, but our internal calc is very similar to it.
    pub volume_pubchem: Option<f32>,
    /// The molecular complexity rating of a compound, computed using the Bertz/Hendrickson/Ihlenfeldt formula.
    pub complexity: Option<f32>,
    /// a topological index of a molecule, defined as the sum of the lengths of the shortest paths
    /// between all pairs of vertices in the chemical graph representing the non-hydrogen atoms
    /// in the molecule
    pub wiener_index: Option<u32>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RingType {
    Aromatic,
    Saturated,
    Aliphatic,
}

/// Represents a single ring; can be on its own, or part of a fused ring system.
#[derive(Clone, Debug)]
pub struct Ring {
    pub atoms: Vec<usize>,
    pub ring_type: RingType,
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

        count_disp(&mut v, self.h_bond_donor.len(), "H don");
        count_disp(&mut v, self.h_bond_acceptor.len(), "H acc");

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

        let mut num_rings_aromatic = 0;
        let mut num_rings_saturated = 0;
        let mut num_rings_aliphatic = 0;

        for ring in &rings {
            match ring.ring_type {
                RingType::Aromatic => num_rings_aromatic += 1,
                RingType::Saturated => num_rings_saturated += 1,
                RingType::Aliphatic => num_rings_aliphatic += 1,
            }
        }

        let aromatic_atoms: HashSet<usize> = rings
            .iter()
            .filter(|r| r.ring_type == RingType::Aromatic)
            .flat_map(|r| r.atoms.iter().copied())
            .collect();

        let num_aromatic_atoms = aromatic_atoms.len();

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

        // let amides_set: HashSet<usize> = amides.iter().copied().collect();
        // let hydroxyl_set: HashSet<usize> = hydroxyl.iter().copied().collect();

        // Exact valence electron count
        let mut num_valence_elecs = 0usize;
        for a in &mol.atoms {
            num_valence_elecs += a.element.valence_electrons();
        }

        let tpsa_ertl = tpsa_ertl(
            mol,
            &mol.adjacency_list,
            &bond_type_by_edge,
            &aromatic_atoms,
            None,
        );
        let (psa_topo, asa_topo, volume) = tpsa_topo(mol);

        let asa_labute = labute_asa_proxy(mol);

        let balaban_j = calc_balaban_j(mol);

        let bertz_ct = bertz_ct(&mol, 100) as f32;

        // Simple monotonic heuristics for logP and MR (good ML features, not RDKit-identical)
        let rings_ct = rings.len() as f32;
        let hal_ct = halogen.len() as f32;

        // todo: WHich?
        // let hetero_ct = num_hetero_atoms as f32;
        let hetero_ct = num_hetero_atoms as f32 - hal_ct;

        // todo: These values may be provincial.
        // todo: Move to the system RDKit uses, so you can mimic its results. Use the AqSolDb
        // todo data set as a reference. These may be good enough for now.
        // Note: For this dataset, the negative penalty for Hetero atoms was also
        // reducing accuracy for the large complex A-8, so it is adjusted.
        let calc_log_p = (0.13 * (num_carbon as f32)) + (0.10 * hal_ct) - (0.17 * rings_ct)
            + (0.07 * hetero_ct)
            + (0.01 * psa_topo)
            + 0.92; // Intercept

        // MR Fix: Increased MW weight (0.10 -> 0.27)
        let mol_refractivity = (0.27 * (mol_weight_f64 as f32)) + (2.30 * rings_ct)
            - (2.74 * hal_ct)
            - (3.27 * hetero_ct)
            + 5.92; // Intercept

        let wiener_index = wiener_index(mol);

        Self {
            num_atoms,
            num_bonds,
            num_heavy_atoms,
            num_hetero_atoms,
            mol_weight: mol_weight_f64 as f32,
            rings,
            num_rings_aromatic,
            num_rings_saturated,
            num_rings_aliphatic,
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
            //
            tpsa_ertl,
            psa_topo,
            log_p: calc_log_p,
            log_p_pubchem: None,
            molar_refractivity: mol_refractivity,
            num_valence_elecs,
            asa_labute,
            asa_topo,
            balaban_j,
            bertz_ct,
            volume,
            volume_pubchem: None,
            complexity: None,
            wiener_index,
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

    let atom_is_sp3_like = |i: usize| -> bool {
        for &j in &adj[i] {
            let Some(bt) = bond_type_by_edge.get(&edge_key(i, j)) else {
                continue;
            };
            if *bt != BondType::Single {
                return false;
            }
        }
        true
    };

    let ring_is_saturated = |cyc: &[usize]| -> bool {
        let n = cyc.len();
        // ring bonds must be single
        for k in 0..n {
            let a = cyc[k];
            let b = cyc[(k + 1) % n];
            let Some(bt) = bond_type_by_edge.get(&edge_key(a, b)) else {
                return false;
            };
            if *bt != BondType::Single {
                return false;
            }
        }
        // ring atoms must be sp3-like (no double/aromatic anywhere)
        cyc.iter().all(|&i| atom_is_sp3_like(i))
    };

    let mut result = Vec::new();
    for ring in rings_5_atom {
        let mut ring_type = RingType::Aliphatic;
        if is_cycle_aromatic(&ring) {
            ring_type = RingType::Aromatic;
        } else if ring_is_saturated(&ring) {
            ring_type = RingType::Saturated;
        }

        result.push(Ring {
            atoms: ring.to_vec(),
            ring_type,
            plane_norm: find_plane_norm(&ring, atoms),
        })
    }

    // todo: DRY
    for ring in rings_6_atom {
        let mut ring_type = RingType::Aliphatic;
        if is_cycle_aromatic(&ring) {
            ring_type = RingType::Aromatic;
        } else if ring_is_saturated(&ring) {
            ring_type = RingType::Saturated;
        }

        result.push(Ring {
            atoms: ring.to_vec(),
            ring_type,
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

/// Pairwise-cap-occlusion ASA proxy:
/// Start with each atom’s 4πr², then subtract a spherical-cap area for each *bonded* neighbor sphere overlap.
/// This is fast and stable and uses 3D coords; it ignores higher-order overlaps.
fn labute_asa_proxy(mol: &MoleculeCommon) -> f32 {
    let n = mol.atoms.len();
    if n == 0 {
        return 0.0;
    }

    let four_pi = 4.0 * std::f64::consts::PI;
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut total = 0.0f64;

    for i in 0..n {
        let ri = mol.atoms[i].element.vdw_radius() as f64;
        let mut area_i = four_pi * ri * ri;

        for &j in &mol.adjacency_list[i] {
            let rj = mol.atoms[j].element.vdw_radius() as f64;
            let d = (mol.atoms[i].posit - mol.atoms[j].posit).magnitude();
            if d <= 0.0 || d >= ri + rj {
                continue;
            }

            // i fully buried by j
            if d <= (rj - ri).abs() && rj >= ri {
                area_i = 0.0;
                break;
            }

            // Plane offset from center i along i->j axis:
            // x = (d^2 - rj^2 + ri^2) / (2d)
            let x = (d * d - rj * rj + ri * ri) / (2.0 * d);
            let hi = (ri - x).clamp(0.0, 2.0 * ri);

            // Covered spherical cap area on i: 2π r h
            let cap = two_pi * ri * hi;
            area_i -= cap;
        }

        if area_i > 0.0 {
            total += area_i;
        }
    }

    total as f32
}

#[allow(non_camel_case_types)]
/// todo: Make a self-contained module for this, and othe rthings that use this nomenclature? (e.g. logP and MR?)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TpsaAtomType {
    // N (aliphatic, neutral)
    N_3single,       // [N](-*)(-*)-*
    N_single_double, // [N](-*)d*
    N_triple,        // [N]#*
    N_nitro_like,    // [N](-*)(d*)d*
    N_azide_middle,  // [N](d*)#*
    N_3ring_subst,   // [N]1(-*)-*-*-1
    NH_2single,      // [NH](-*)-*
    NH_3ring,        // [NH]1-*-*-1
    NH_double,       // [NH]d*
    NH2_single,      // [NH2]-*

    // N (aliphatic, cation)
    Nplus_quat,           // [N+](-*)(-*)(-*)-*
    Nplus_2single_double, // [N+](-*)(-*)d*
    Nplus_isocyano,       // [N+](-*)#*
    NHplus_3single,       // [NH+](-*)(-*)-*
    NHplus_single_double, // [NH+](-*)d*
    NH2plus_2single,      // [NH2+](-*)-*
    NH2plus_double,       // [NH2+]d*
    NH3plus_single,       // [NH3+]-*

    // N aromatic (neutral/cation)
    n_arom2,            // [n](:*):*
    n_arom3,            // [n](:*)(:*):*
    n_single_arom2,     // [n](-*)(:*):*
    n_oxide_like,       // [n](d*)(:*):*
    nH_arom2,           // [nH](:*):*
    nplus_arom3,        // [n+](:*)(:*):*
    nplus_single_arom2, // [n+](-*)(:*):*
    nHplus_arom2,       // [nH+](:*):*

    // O
    O_2single,     // [O](-*)-*
    O_3ring,       // [O]1-*-*-1
    O_double,      // [O]d*
    OH_single,     // [OH]-*
    Ominus_single, // [O-]-*
    o_aromatic,    // [o](:*):*

    // S
    S_2single,         // [S](-*)-*
    S_double,          // [S]d*
    S_2single_double,  // [S](-*)(-*)d*
    S_2single_2double, // [S](-*)(-*)(d*)d*
    SH_single,         // [SH]-*
    s_aromatic,        // [s](:*):*
    s_aromatic_double, // [s](d*)(:*):*

    // P
    P_3single,         // [P](-*)(-*)-*
    P_single_double,   // [P](-*)d*
    P_3single_double,  // [P](-*)(-*)(-*)d*
    PH_2single_double, // [PH](-*)(-*)d*
}

fn tpsa_contrib(t: TpsaAtomType) -> f32 {
    use TpsaAtomType::*;
    match t {
        N_3single => 3.24,
        N_single_double => 12.36,
        N_triple => 23.79,
        N_nitro_like => 11.68,
        N_azide_middle => 13.60,
        N_3ring_subst => 3.01,
        NH_2single => 12.03,
        NH_3ring => 21.94,
        NH_double => 23.85,
        NH2_single => 26.02,

        Nplus_quat => 0.00,
        Nplus_2single_double => 3.01,
        Nplus_isocyano => 4.36,
        NHplus_3single => 4.44,
        NHplus_single_double => 13.97,
        NH2plus_2single => 16.61,
        NH2plus_double => 25.59,
        NH3plus_single => 27.64,

        n_arom2 => 12.89,
        n_arom3 => 4.41,
        n_single_arom2 => 4.93,
        n_oxide_like => 8.39,
        nH_arom2 => 15.79,
        nplus_arom3 => 4.10,
        nplus_single_arom2 => 3.88,
        nHplus_arom2 => 14.14,

        O_2single => 9.23,
        O_3ring => 12.53,
        O_double => 17.07,
        OH_single => 20.23,
        Ominus_single => 23.06,
        o_aromatic => 13.14,

        S_2single => 25.30,
        S_double => 32.09,
        S_2single_double => 19.21,
        S_2single_2double => 8.38,
        SH_single => 38.80,
        s_aromatic => 28.24,
        s_aromatic_double => 21.70,

        P_3single => 13.59,
        P_single_double => 34.14,
        P_3single_double => 9.81,
        PH_2single_double => 23.47,
    }
}

/// todo: For etrls TPSM
fn bond_order_counts(
    i: usize,
    mol: &MoleculeCommon,
    adj: &[Vec<usize>],
    bond_type_by_edge: &HashMap<(usize, usize), BondType>,
) -> (usize, usize, usize, usize) {
    let mut single = 0usize;
    let mut double = 0usize;
    let mut triple = 0usize;
    let mut aromatic = 0usize;

    for &j in &adj[i] {
        if mol.atoms[j].element == Hydrogen {
            continue;
        }
        match bond_type_by_edge
            .get(&edge_key(i, j))
            .copied()
            .unwrap_or(BondType::Single)
        {
            BondType::Single => single += 1,
            BondType::Double => double += 1,
            BondType::Triple => triple += 1,
            BondType::Aromatic => aromatic += 1,
            _ => single += 1,
        }
    }
    (single, double, triple, aromatic)
}

/// todo: For etrls TPSM
fn explicit_h_count(i: usize, mol: &MoleculeCommon, adj: &[Vec<usize>]) -> usize {
    adj[i]
        .iter()
        .filter(|&&j| mol.atoms[j].element == Hydrogen)
        .count()
}

/// Detect if atom i participates in any 3-member ring (triangle).
/// todo: For etrls TPSM
fn in_3_member_ring(
    i: usize,
    mol: &MoleculeCommon,
    adj: &[Vec<usize>],
    bond_type_by_edge: &HashMap<(usize, usize), BondType>,
) -> bool {
    let neigh: Vec<usize> = adj[i]
        .iter()
        .copied()
        .filter(|&j| mol.atoms[j].element != Hydrogen)
        .collect();

    for a_idx in 0..neigh.len() {
        for b_idx in (a_idx + 1)..neigh.len() {
            let a = neigh[a_idx];
            let b = neigh[b_idx];
            if bond_type_by_edge.contains_key(&edge_key(a, b)) {
                return true;
            }
        }
    }
    false
}

/// todo: For etrls TPSM
fn has_double_to_element(
    i: usize,
    el: na_seq::Element,
    mol: &MoleculeCommon,
    adj: &[Vec<usize>],
    bond_type_by_edge: &HashMap<(usize, usize), BondType>,
) -> bool {
    adj[i].iter().any(|&j| {
        mol.atoms[j].element == el
            && bond_type_by_edge
                .get(&edge_key(i, j))
                .copied()
                .unwrap_or(BondType::Single)
                == BondType::Double
    })
}

/// Ertl fragment TPSA (Å²).
///
/// `formal_charge`: optional per-atom integer formal charge.
/// `implicit_h`: optional per-atom implicit H count (in addition to any explicit H atoms).
pub fn tpsa_ertl(
    mol: &MoleculeCommon,
    adj: &[Vec<usize>],
    bond_type_by_edge: &HashMap<(usize, usize), BondType>,
    aromatic_atoms: &HashSet<usize>,
    formal_charge: Option<&[i8]>,
) -> f32 {
    use na_seq::Element::*;

    let mut sum = 0.0f32;

    for i in 0..mol.atoms.len() {
        let el = mol.atoms[i].element;
        let is_arom = aromatic_atoms.contains(&i);

        let q = formal_charge.map(|a| a[i] as i32).unwrap_or(0);
        let h = explicit_h_count(i, mol, adj);

        let (single, double, triple, aromatic) = bond_order_counts(i, mol, adj, bond_type_by_edge);
        let heavy_degree = single + double + triple + aromatic;

        let in3 = in_3_member_ring(i, mol, adj, bond_type_by_edge);

        let ty: Option<TpsaAtomType> = match el {
            Oxygen => {
                if is_arom {
                    Some(TpsaAtomType::o_aromatic)
                } else if q < 0 {
                    Some(TpsaAtomType::Ominus_single)
                } else if double >= 1 {
                    Some(TpsaAtomType::O_double)
                } else if h >= 1 {
                    Some(TpsaAtomType::OH_single)
                } else if in3 {
                    Some(TpsaAtomType::O_3ring)
                } else if heavy_degree >= 1 {
                    Some(TpsaAtomType::O_2single)
                } else {
                    None
                }
            }

            Nitrogen => {
                if is_arom {
                    let has_exocyclic_single = adj[i].iter().any(|&j| {
                        if mol.atoms[j].element == Hydrogen {
                            return false;
                        }
                        let bt = bond_type_by_edge
                            .get(&edge_key(i, j))
                            .copied()
                            .unwrap_or(BondType::Single);
                        bt == BondType::Single && !aromatic_atoms.contains(&j)
                    });

                    let has_n_oxide_like =
                        has_double_to_element(i, Oxygen, mol, adj, bond_type_by_edge);

                    if q > 0 {
                        if h >= 1 {
                            Some(TpsaAtomType::nHplus_arom2)
                        } else if has_exocyclic_single {
                            Some(TpsaAtomType::nplus_single_arom2)
                        } else {
                            Some(TpsaAtomType::nplus_arom3)
                        }
                    } else {
                        if h >= 1 {
                            Some(TpsaAtomType::nH_arom2)
                        } else if has_n_oxide_like {
                            Some(TpsaAtomType::n_oxide_like)
                        } else if has_exocyclic_single {
                            Some(TpsaAtomType::n_single_arom2)
                        } else if aromatic >= 3 || heavy_degree >= 3 {
                            Some(TpsaAtomType::n_arom3)
                        } else {
                            Some(TpsaAtomType::n_arom2)
                        }
                    }
                } else {
                    if q > 0 {
                        if h == 0 && heavy_degree >= 4 {
                            Some(TpsaAtomType::Nplus_quat)
                        } else if h == 0 && triple >= 1 {
                            Some(TpsaAtomType::Nplus_isocyano)
                        } else if h == 0 && double >= 1 && single >= 2 {
                            Some(TpsaAtomType::Nplus_2single_double)
                        } else if h == 1 && double >= 1 {
                            Some(TpsaAtomType::NHplus_single_double)
                        } else if h == 1 {
                            Some(TpsaAtomType::NHplus_3single)
                        } else if h == 2 && double >= 1 {
                            Some(TpsaAtomType::NH2plus_double)
                        } else if h == 2 {
                            Some(TpsaAtomType::NH2plus_2single)
                        } else if h >= 3 {
                            Some(TpsaAtomType::NH3plus_single)
                        } else {
                            None
                        }
                    } else {
                        if in3 && h == 0 && double == 0 && triple == 0 && heavy_degree == 3 {
                            Some(TpsaAtomType::N_3ring_subst)
                        } else if in3 && h == 1 && double == 0 && triple == 0 && heavy_degree == 2 {
                            Some(TpsaAtomType::NH_3ring)
                        } else if double >= 2 {
                            Some(TpsaAtomType::N_nitro_like)
                        } else if double >= 1 && triple >= 1 {
                            Some(TpsaAtomType::N_azide_middle)
                        } else if triple >= 1 && heavy_degree == 1 {
                            Some(TpsaAtomType::N_triple)
                        } else if h == 2 && double == 0 && triple == 0 && heavy_degree == 1 {
                            Some(TpsaAtomType::NH2_single)
                        } else if h == 1 && double >= 1 {
                            Some(TpsaAtomType::NH_double)
                        } else if h == 1 && double == 0 && triple == 0 && heavy_degree == 2 {
                            Some(TpsaAtomType::NH_2single)
                        } else if double >= 1 && heavy_degree >= 2 {
                            Some(TpsaAtomType::N_single_double)
                        } else if h == 0 && double == 0 && triple == 0 && heavy_degree == 3 {
                            Some(TpsaAtomType::N_3single)
                        } else {
                            None
                        }
                    }
                }
            }

            Sulfur => {
                if is_arom {
                    if double >= 1 {
                        Some(TpsaAtomType::s_aromatic_double)
                    } else {
                        Some(TpsaAtomType::s_aromatic)
                    }
                } else if h >= 1 && heavy_degree == 1 {
                    Some(TpsaAtomType::SH_single)
                } else if double >= 2 {
                    Some(TpsaAtomType::S_2single_2double)
                } else if double >= 1 && heavy_degree >= 3 {
                    Some(TpsaAtomType::S_2single_double)
                } else if double >= 1 {
                    Some(TpsaAtomType::S_double)
                } else if heavy_degree >= 1 {
                    Some(TpsaAtomType::S_2single)
                } else {
                    None
                }
            }

            Phosphorus => {
                if h >= 1 && double >= 1 {
                    Some(TpsaAtomType::PH_2single_double)
                } else if double >= 1 && single >= 3 {
                    Some(TpsaAtomType::P_3single_double)
                } else if double >= 1 {
                    Some(TpsaAtomType::P_single_double)
                } else if heavy_degree >= 3 {
                    Some(TpsaAtomType::P_3single)
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(t) = ty {
            sum += tpsa_contrib(t);
        }
    }

    sum
}

/// Compute TPSA, and volume by creating and analyzing a mesh. We count surface area
/// from triangles within a certain distance of O and N atoms, and Hs bonded to them.
fn tpsa_topo(mol: &MoleculeCommon) -> (f32, f32, f32) {
    // This radius is a mod to the VDW radius used to compute the mesh.
    let radius = -0.08;
    // Lower values take significantly longer, but are more accurate.
    let precision = 0.5;

    // If a triangle's center (?) is within this dist sq of a polar atom,
    // count it towards the TPSA.
    let dist_sq_thresh_tpsa = 1.5f32.powi(2); // todo

    let start = Instant::now();

    let mut atoms = Vec::with_capacity(mol.atoms.len());
    for atom in &mol.atoms {
        atoms.push((atom.posit.into(), atom.element.vdw_radius()));
    }

    // API issue here: Convert *back* to mcubes::Mesh, so we can access the `volume` method.
    let mesh_graphics = make_sas_mesh(&atoms, radius, precision);

    let vertices: Vec<mcubes::Vertex> = mesh_graphics
        .vertices
        .iter()
        // I'm not sure why we need to invert the normal here; same reason we use InsideOnly above.
        .map(|v| mcubes::Vertex {
            posit: Vec3F32::from_slice(&v.position).unwrap(),
            normal: -v.normal,
        })
        .collect();

    let mesh = mcubes::Mesh {
        indices: mesh_graphics.indices,
        vertices,
    };

    let vol = mesh.volume();

    let polar_atoms: Vec<_> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(i, a)| match a.element {
            Oxygen | Nitrogen => true,
            Hydrogen => {
                for adj in &mol.adjacency_list[*i] {
                    if matches!(mol.atoms[*adj].element, Oxygen | Nitrogen) {
                        return true;
                    }
                }
                false
            }
            _ => false,
        })
        .map(|(i, a)| a)
        .collect();

    // todo: Scale the area based on dist?
    let mut tpsa = 0.0;
    let mut asa = 0.;

    for tri in mesh.indices.chunks(3) {
        let i0 = tri[0];
        let i1 = tri[1];
        let i2 = tri[2];

        let p0 = mesh.vertices[i0].posit;
        let p1 = mesh.vertices[i1].posit;
        let p2 = mesh.vertices[i2].posit;

        let centroid = (p0 + p1 + p2) / 3.0;
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let area = 0.5 * e1.cross(e2).magnitude();

        asa += area;

        for polar in &polar_atoms {
            let p: Vec3F32 = polar.posit.into();

            if (centroid - p).magnitude_squared() < dist_sq_thresh_tpsa {
                tpsa += area;
                break;
            }
        }
    }

    let elapsed = start.elapsed().as_millis();
    // println!("TPSA (topo) computation took {} ms", elapsed);

    (tpsa, asa, vol)
}

/// Exact Balaban J (distance-sum connectivity index) per component.
/// Uses BFS distances (unweighted).
fn calc_balaban_j(mol: &MoleculeCommon) -> f32 {
    use na_seq::Element::Hydrogen;

    let n_all = mol.atoms.len();
    if n_all < 2 || mol.bonds.is_empty() {
        return 0.0;
    }

    // old -> heavy
    let mut old_to_h: Vec<Option<usize>> = vec![None; n_all];
    let mut h_to_old: Vec<usize> = Vec::new();
    for i in 0..n_all {
        if mol.atoms[i].element != Hydrogen {
            old_to_h[i] = Some(h_to_old.len());
            h_to_old.push(i);
        }
    }

    let n = h_to_old.len();
    if n < 2 {
        return 0.0;
    }

    // heavy adjacency
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (hi, &oi) in h_to_old.iter().enumerate() {
        for &oj in &mol.adjacency_list[oi] {
            if let Some(hj) = old_to_h[oj] {
                adj[hi].push(hj);
            }
        }
    }

    // heavy edges (dedup via edge_key + HashSet-like behavior using a map)
    let mut edges: Vec<(usize, usize)> = Vec::new();
    {
        let mut seen: HashMap<(usize, usize), ()> = HashMap::new();
        for b in &mol.bonds {
            let a = b.atom_0;
            let c = b.atom_1;
            let Some(ha) = old_to_h[a] else { continue };
            let Some(hc) = old_to_h[c] else { continue };
            let e = edge_key(ha, hc);
            if seen.insert(e, ()).is_none() {
                edges.push(e);
            }
        }
    }

    let m = edges.len();
    if m == 0 {
        return 0.0;
    }

    // components on heavy graph
    let mut comp_id = vec![usize::MAX; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    let mut q = VecDeque::new();

    for s in 0..n {
        if comp_id[s] != usize::MAX {
            continue;
        }
        let cid = comps.len();
        let mut comp = Vec::new();
        comp_id[s] = cid;
        q.clear();
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &v in &adj[u] {
                if comp_id[v] == usize::MAX {
                    comp_id[v] = cid;
                    q.push_back(v);
                }
            }
        }

        comps.push(comp);
    }

    // per component, edge-weighted average (your original aggregation)
    let mut total_edges = 0usize;
    let mut accum = 0.0f64;

    let mut dist = vec![u32::MAX; n];
    let mut bfs = VecDeque::new();

    for (cid, comp) in comps.iter().enumerate() {
        let nc = comp.len();
        if nc < 2 {
            continue;
        }

        let mut comp_edges: Vec<(usize, usize)> = Vec::new();
        for &(a, b) in &edges {
            if comp_id[a] == cid {
                comp_edges.push((a, b));
            }
        }
        let mc = comp_edges.len();
        if mc == 0 {
            continue;
        }

        let denom = (mc as i64 - nc as i64 + 2) as f64; // gamma+1 = (m-n+1)+1
        if denom <= 0.0 {
            continue;
        }

        // D_i for nodes in this component
        let mut dsum = vec![0u32; n];

        for &src in comp {
            dist.fill(u32::MAX);
            dist[src] = 0;
            bfs.clear();
            bfs.push_back(src);

            while let Some(u) = bfs.pop_front() {
                let du = dist[u];
                for &v in &adj[u] {
                    if dist[v] == u32::MAX {
                        dist[v] = du + 1;
                        bfs.push_back(v);
                    }
                }
            }

            let mut sum = 0u32;
            for &v in comp {
                sum = sum.saturating_add(dist[v]);
            }
            dsum[src] = sum;
        }

        let mut s = 0.0f64;
        for (a, b) in &comp_edges {
            let da = dsum[*a] as f64;
            let db = dsum[*b] as f64;
            if da > 0.0 && db > 0.0 {
                s += 1.0 / (da * db).sqrt();
            }
        }

        let j = (mc as f64 / denom) * s;

        total_edges += mc;
        accum += j * (mc as f64);
    }

    if total_edges == 0 {
        0.0
    } else {
        (accum / total_edges as f64) as f32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum ConnKey {
    Pair(u32, u32),
    Triple(u32, u32, u32),
}

fn bond_order_f64(b: &Bond) -> f64 {
    use BondType::*;
    match b.bond_type {
        Single => 1.0,
        Double => 2.0,
        Triple => 3.0,
        Aromatic => 1.5,
        _ => 1.0,
    }
}

fn info_entropy(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let inv = 1.0 / total;
    let ln2 = std::f64::consts::LN_2;

    let mut h = 0.0;
    for &c in counts {
        if c <= 0.0 {
            continue;
        }
        let p = c * inv;
        h -= p * (p.ln() / ln2); // log2
    }
    h
}

/// Unweighted BFS all-pairs shortest paths (u16), INF=u16::MAX.
/// This corresponds to RDKit GetDistanceMatrix(... useBO=0, useAtomWts=0).
fn all_pairs_shortest_paths(adj: &[Vec<usize>]) -> Vec<Vec<u16>> {
    let n = adj.len();
    let inf: u16 = u16::MAX;
    let mut dmat = vec![vec![inf; n]; n];

    for s in 0..n {
        let mut q = VecDeque::new();
        dmat[s][s] = 0;
        q.push_back(s);

        while let Some(v) = q.pop_front() {
            let dv = dmat[s][v];
            let nd = dv.saturating_add(1);
            for &u in &adj[v] {
                if dmat[s][u] == inf {
                    dmat[s][u] = nd;
                    q.push_back(u);
                }
            }
        }
    }
    dmat
}

/// Equivalent to RDKit _AssignSymmetryClasses(...).
/// Two atoms are in the same class if their sorted distance vectors match
/// out to the `cutoff`th entry (RDKit truncation behavior).
fn assign_symmetry_classes(dmat: &[Vec<u16>], num_atoms: usize, cutoff: usize) -> Vec<u32> {
    let use_k = cutoff.min(num_atoms);

    let mut keys_seen: Vec<Vec<u16>> = Vec::new();
    let mut sym_class: Vec<u32> = vec![0; num_atoms];

    for i in 0..num_atoms {
        let mut row = dmat[i].clone();
        row.sort_unstable();
        row.truncate(use_k);

        let mut found: Option<usize> = None;
        for (idx, k) in keys_seen.iter().enumerate() {
            if *k == row {
                found = Some(idx);
                break;
            }
        }

        let cls = match found {
            Some(idx) => (idx as u32) + 1,
            None => {
                keys_seen.push(row);
                keys_seen.len() as u32
            }
        };

        sym_class[i] = cls;
    }

    sym_class
}

/// Equivalent to RDKit _CreateBondDictEtc(mol, numAtoms)
/// Returns:
/// - bondDict: (i,j) -> bond order (f64)
/// - neighborList: adjacency list
/// - vdList: degree per atom
fn create_bond_dict_etc(
    mol: &MoleculeCommon,
) -> (HashMap<(usize, usize), f64>, Vec<Vec<usize>>, Vec<usize>) {
    let num_atoms = mol.atoms.len();

    let mut bond_dict: HashMap<(usize, usize), f64> = HashMap::with_capacity(mol.bonds.len());
    for b in &mol.bonds {
        bond_dict.insert(edge_key(b.atom_0, b.atom_1), bond_order_f64(b));
    }

    let neighbor_list: Vec<Vec<usize>> = mol.adjacency_list.clone();
    let mut vd_list: Vec<usize> = vec![0; num_atoms];
    for i in 0..num_atoms {
        vd_list[i] = neighbor_list[i].len();
    }

    (bond_dict, neighbor_list, vd_list)
}

/// Equivalent to RDKit _LookUpBondOrder(i, j, bondDict)
fn look_up_bond_order(i: usize, j: usize, bond_dict: &HashMap<(usize, usize), f64>) -> f64 {
    bond_dict.get(&edge_key(i, j)).copied().unwrap_or(1.0)
}

/// Equivalent to RDKit _CalculateEntropies(connectionDict, atomTypeDict, numAtoms)
fn calculate_entropies(
    connection_dict: &HashMap<ConnKey, f64>,
    atom_type_dict: &HashMap<u32, f64>,
    num_atoms: usize,
) -> f64 {
    let conn_counts: Vec<f64> = connection_dict.values().copied().collect();
    let tot_conn: f64 = conn_counts.iter().sum();

    let atom_counts: Vec<f64> = atom_type_dict.values().copied().collect();

    let ln2 = std::f64::consts::LN_2;

    // RDKit: connectionIE = totConn*(InfoEntropy(conn)+log2(totConn))
    let connection_ie = if tot_conn > 0.0 {
        tot_conn * (info_entropy(&conn_counts) + (tot_conn.ln() / ln2))
    } else {
        0.0
    };

    // RDKit: atomTypeIE = numAtoms * InfoEntropy(atomType)
    let atom_type_ie = (num_atoms as f64) * info_entropy(&atom_counts);

    atom_type_ie + connection_ie
}

/// RDKit-style BertzCT (structure-only), translated from the Python you posted.
/// - cutoff: same semantics as RDKit (distance-vector truncation length)
pub fn bertz_ct(mol: &MoleculeCommon, cutoff: usize) -> f64 {
    let num_atoms = mol.atoms.len();
    if num_atoms < 2 {
        return 0.0;
    }

    // RDKit: dMat = GetDistanceMatrix(... useBO=0, useAtomWts=0, force=1)
    let dmat = all_pairs_shortest_paths(&mol.adjacency_list);

    // RDKit: bondDict, neighborList, vdList = _CreateBondDictEtc(mol, numAtoms)
    let (bond_dict, neighbor_list, vd_list) = create_bond_dict_etc(mol);

    // RDKit: symmetryClasses = _AssignSymmetryClasses(...)
    let symmetry_classes = assign_symmetry_classes(&dmat, num_atoms, cutoff);

    let mut atom_type_dict: HashMap<u32, f64> = HashMap::new();
    let mut connection_dict: HashMap<ConnKey, f64> = HashMap::new();

    for atom_idx in 0..num_atoms {
        // RDKit: hingeAtomNumber = mol.GetAtomWithIdx(atomIdx).GetAtomicNum()
        let hinge_atom_number = mol.atoms[atom_idx].element.atomic_number() as u32;
        *atom_type_dict.entry(hinge_atom_number).or_insert(0.0) += 1.0;

        let hinge_atom_class = symmetry_classes[atom_idx];
        let num_neighbors = vd_list[atom_idx];

        for i in 0..num_neighbors {
            let neighbor_i_idx = neighbor_list[atom_idx][i];
            let ni_class = symmetry_classes[neighbor_i_idx];
            let bond_i_order = look_up_bond_order(atom_idx, neighbor_i_idx, &bond_dict);

            // RDKit: if (bond_i_order > 1) and (neighbor_iIdx > atomIdx):
            if bond_i_order > 1.0 && neighbor_i_idx > atom_idx {
                let num_connections = bond_i_order * (bond_i_order - 1.0) / 2.0;
                let a = hinge_atom_class.min(ni_class);
                let b = hinge_atom_class.max(ni_class);
                *connection_dict.entry(ConnKey::Pair(a, b)).or_insert(0.0) += num_connections;
            }

            for j in (i + 1)..num_neighbors {
                let neighbor_j_idx = neighbor_list[atom_idx][j];
                let nj_class = symmetry_classes[neighbor_j_idx];
                let bond_j_order = look_up_bond_order(atom_idx, neighbor_j_idx, &bond_dict);

                let num_connections = bond_i_order * bond_j_order;
                let a = ni_class.min(nj_class);
                let c = ni_class.max(nj_class);
                *connection_dict
                    .entry(ConnKey::Triple(a, hinge_atom_class, c))
                    .or_insert(0.0) += num_connections;
            }
        }
    }

    // RDKit: if not connectionDict: connectionDict = {'a': 1}
    if connection_dict.is_empty() {
        connection_dict.insert(ConnKey::Pair(1, 1), 1.0);
    }

    calculate_entropies(&connection_dict, &atom_type_dict, num_atoms)
}

/// Excludes hydrogens.

fn wiener_index(mol: &MoleculeCommon) -> Option<u32> {
    let n = mol.atoms.len();
    if mol.adjacency_list.len() != n {
        return None;
    }

    let heavy: Vec<usize> = (0..n)
        .filter(|&i| mol.atoms[i].element != Hydrogen)
        .collect();

    if heavy.len() < 2 {
        return Some(0);
    }

    let mut total: u32 = 0;

    let mut dist: Vec<i32> = vec![-1; n];
    let mut q: VecDeque<usize> = VecDeque::new();

    for (si, &start) in heavy.iter().enumerate() {
        dist.fill(-1);
        dist[start] = 0;
        q.clear();
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            let du = dist[u];

            for &v in &mol.adjacency_list[u] {
                if v >= n {
                    return None;
                }
                if mol.atoms[v].element == Hydrogen {
                    continue;
                }
                if dist[v] < 0 {
                    dist[v] = du + 1;
                    q.push_back(v);
                }
            }
        }

        for &end in &heavy[(si + 1)..] {
            let d = dist[end];
            if d < 0 {
                return None;
            }
            total += d as u32;
        }
    }

    Some(total)
}
