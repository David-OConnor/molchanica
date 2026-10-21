//! General OPLS ligand generation, with validated, persistent topology caching.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Write,
    fs,
    path::PathBuf,
    sync::{Mutex, atomic::{AtomicU64, Ordering}},
};

use bio_files::{AtomGeneric, BondGeneric, BondType};
use sha2::{Digest, Sha256};

use crate::{FfMolType, MolDynamics, ParamError, gromacs, params::FfParamSet};
use crate::import::ExplicitParams;

use super::known::Ligand;

// The public server runs one BOSS job at a time. Recheck the cache after taking
// this lock so copies and simultaneous consumers do not submit duplicate jobs.
static REQUEST: Mutex<()> = Mutex::new(());
static TEMP_ID: AtomicU64 = AtomicU64::new(0);
const SEPARATOR: &str = "\n$$$$\n";

#[derive(Debug, Default)]
pub(super) struct Generator {
    memory: Mutex<HashMap<String, Ligand>>,
}

fn error(message: impl AsRef<str>) -> ParamError {
    ParamError::new(message.as_ref())
}

/// Set DYNAMICS_OPLS_CACHE to override this directory, for example when
/// preparing a portable cache for an offline workstation.
pub fn cache_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("DYNAMICS_OPLS_CACHE") {
        return PathBuf::from(path).join("v1");
    }
    let root = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    root.join("dynamics").join("opls-ligpargen").join("v1")
}

struct Input {
    mol: String,
    upload: String,
    charge: i32,
    key: String,
}

fn input(atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> Result<Input, ParamError> {
    if atoms.is_empty() || atoms.len() > 999 || bonds.len() > 999 {
        return Err(error("OPLS ligand input needs 1–999 atoms and at most 999 bonds"));
    }
    let mut indices = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        if indices.insert(atom.serial_number, i).is_some() {
            return Err(error("Duplicate OPLS ligand atom serial number"));
        }
        if ![atom.posit.x, atom.posit.y, atom.posit.z].iter().all(|v| v.is_finite()) {
            return Err(error("OPLS ligand coordinates must be finite"));
        }
    }
    let mut edges = Vec::new();
    let mut adjacency = vec![Vec::new(); atoms.len()];
    let mut orders = vec![0.0_f32; atoms.len()];
    let mut aromatic = vec![0; atoms.len()];
    let mut seen = HashSet::new();
    for bond in bonds {
        let (&i, &j) = indices.get(&bond.atom_0_sn).zip(indices.get(&bond.atom_1_sn))
            .ok_or_else(|| error("OPLS ligand bond references an absent atom"))?;
        if i == j || !seen.insert((i.min(j), i.max(j))) {
            return Err(error("Duplicate or self bond in OPLS ligand"));
        }
        let kind = match bond.bond_type {
            BondType::Single | BondType::Amide => 1,
            BondType::Double => 2,
            BondType::Triple => 3,
            BondType::Aromatic => 4,
            _ => return Err(error("LigParGen needs explicit single, double, triple or aromatic bond orders")),
        };
        let order = if kind == 4 { 1.5 } else { kind as f32 };
        orders[i] += order;
        orders[j] += order;
        aromatic[i] += usize::from(kind == 4);
        aromatic[j] += usize::from(kind == 4);
        adjacency[i].push(j);
        adjacency[j].push(i);
        edges.push((i, j, kind));
    }
    let mut visited = HashSet::from([0]);
    let mut queue = VecDeque::from([0]);
    while let Some(i) = queue.pop_front() {
        for &j in &adjacency[i] {
            if visited.insert(j) { queue.push_back(j); }
        }
    }
    if visited.len() != atoms.len() {
        return Err(error("Parameterize each disconnected ligand or salt component separately"));
    }

    // AtomGeneric does not retain formal charges. Recover ordinary closed-shell
    // organic valences from the fully hydrogenated graph, including zwitterions.
    // Ambiguous/radical valences are rejected instead of guessing protonation.
    let mut charges = Vec::new();
    for (i, atom) in atoms.iter().enumerate() {
        let z = atom.element.atomic_number();
        let valence = orders[i];
        let q = match (z, valence as i32) {
            (8 | 16, _) if aromatic[i] == 2 => 0,
            (1, 1) | (6, 4) | (7, 3) | (8, 2) | (9 | 17 | 35 | 53, 1)
                | (5, 3) | (14, 4) | (15, 3 | 5) | (16, 2 | 4 | 6) => 0,
            (7, 4) if aromatic[i] == 2 => {
                // Pyrrole's [nH]/substituted N has no formal charge; pyridinium
                // has +1. A five-membered aromatic ring distinguishes them.
                let neighbors: Vec<_> = edges.iter().filter_map(|&(a,b,k)| {
                    if k != 4 { None } else if a == i { Some(b) }
                    else if b == i { Some(a) } else { None }
                }).collect();
                let mut frontier = vec![(neighbors[0], 0, vec![i])];
                let mut five = false;
                while let Some((node, depth, mut path)) = frontier.pop() {
                    if node == neighbors[1] && depth == 3 { five = true; break; }
                    if depth >= 3 || path.contains(&node) { continue; }
                    path.push(node);
                    for &(a,b,k) in &edges {
                        if k == 4 && (a == node || b == node) {
                            frontier.push((if a == node { b } else { a }, depth + 1, path.clone()));
                        }
                    }
                }
                if five { 0 } else { 1 }
            }
            (7, 4) | (8, 3) | (15, 4) | (16, 3) => 1,
            (7, 2) | (8, 1) | (16, 1) | (5, 4) => -1,
            _ => return Err(error(format!(
                "Cannot determine the closed-shell formal charge of OPLS ligand atom {} ({}), bond-order sum {valence}; include all hydrogens or supply an explicit topology",
                atom.serial_number, atom.element.to_letter()
            ))),
        };
        charges.push(q);
    }
    let charge = charges.iter().sum();
    let original: Vec<_> = (0..atoms.len()).collect();
    let mol = mol_block(atoms, &edges, &charges, &original);
    // BOSS's Z-matrix path needs a connected heavy-atom prefix. PubChem
    // frequently puts all heteroatoms first, which violates that assumption.
    let start = (0..atoms.len()).filter(|&i| atoms[i].element.atomic_number() != 1)
        .min_by_key(|&i| (adjacency[i].iter().filter(|&&j| atoms[j].element.atomic_number() != 1).count(), i))
        .unwrap_or(0);
    let mut order = Vec::new();
    let mut placed = HashSet::from([start]);
    let mut queue = VecDeque::from([start]);
    while let Some(i) = queue.pop_front() {
        order.push(i);
        for &j in &adjacency[i] {
            if atoms[j].element.atomic_number() != 1 && placed.insert(j) { queue.push_back(j); }
        }
    }
    order.extend((0..atoms.len()).filter(|i| !placed.contains(i)));
    let upload = mol_block(atoms, &edges, &charges, &order);

    // Permutation-invariant graph fingerprint. A full graph match still checks
    // every cache hit; a fingerprint collision cannot assign another topology.
    let mut labels: Vec<Vec<u8>> = atoms.iter().zip(&charges).map(|(a,q)| {
        Sha256::digest(format!("{}:{q}", a.element.atomic_number())).to_vec()
    }).collect();
    for _ in 0..atoms.len() {
        labels = (0..atoms.len()).map(|i| {
            let mut neighbors: Vec<_> = edges.iter().filter_map(|&(a,b,k)| {
                let j = if a == i { b } else if b == i { a } else { return None; };
                let mut label = vec![k];
                label.extend(&labels[j]);
                Some(label)
            }).collect();
            neighbors.sort();
            let mut hash = Sha256::new();
            hash.update(&labels[i]);
            for label in neighbors { hash.update(label); }
            hash.finalize().to_vec()
        }).collect();
    }
    labels.sort();
    let mut hash = Sha256::new();
    hash.update(b"dynamics-ligpargen-cm1a-opt0-v1");
    for label in labels { hash.update(label); }
    let key = format!("{:x}", hash.finalize());
    Ok(Input { mol, upload, charge, key })
}

fn mol_block(atoms: &[AtomGeneric], edges: &[(usize, usize, u8)], charges: &[i32], order: &[usize]) -> String {
    let mut mapped = vec![0; atoms.len()];
    let mut mol = format!("LIG\nDynamics          3D\n\n{:3}{:3}  0  0  0  0            999 V2000\n", atoms.len(), edges.len());
    for (i, &j) in order.iter().enumerate() {
        mapped[j] = i + 1;
        let atom = &atoms[j];
        writeln!(mol, "{:10.4}{:10.4}{:10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0",
            atom.posit.x, atom.posit.y, atom.posit.z, atom.element.to_letter()).unwrap();
    }
    for &(i, j, kind) in edges {
        writeln!(mol, "{:3}{:3}{:3}  0  0  0  0", mapped[i], mapped[j], kind).unwrap();
    }
    let nonzero: Vec<_> = charges.iter().enumerate().filter(|(_, q)| **q != 0).collect();
    for chunk in nonzero.chunks(8) {
        write!(mol, "M  CHG{:3}", chunk.len()).unwrap();
        for &(i, q) in chunk { write!(mol, " {:3} {:3}", mapped[i], q).unwrap(); }
        mol.push('\n');
    }
    mol.push_str("M  END\n");
    mol
}

impl Generator {
    pub fn remember(&self, mol: &MolDynamics) {
        if let (Ok(request), Some(template)) = (input(&mol.atoms, &mol.bonds), Ligand::from_prepared(mol)) {
            self.memory.lock().unwrap().insert(request.key, template);
        }
    }

    pub fn cached(&self, atoms: &mut [AtomGeneric], bonds: &[BondGeneric])
        -> Result<Option<ExplicitParams>, ParamError>
    {
        let request = input(atoms, bonds)?;
        if let Some(template) = self.memory.lock().unwrap().get(&request.key) {
            return template.build(atoms, bonds);
        }
        let path = cache_dir().join(format!("{}.opls", request.key));
        let Ok(text) = fs::read_to_string(path) else { return Ok(None); };
        let Some((mol, top)) = text.split_once(SEPARATOR) else { return Ok(None); };
        let Ok(template) = Ligand::from_text(mol, top) else { return Ok(None); };
        let result = template.build(atoms, bonds)?;
        if result.is_some() { self.memory.lock().unwrap().insert(request.key, template); }
        Ok(result)
    }

    pub fn generate(&self, atoms: &mut [AtomGeneric], bonds: &[BondGeneric])
        -> Result<ExplicitParams, ParamError>
    {
        let _guard = REQUEST.lock().unwrap();
        if let Some(params) = self.cached(atoms, bonds)? { return Ok(params); }
        let request = input(atoms, bonds)?;
        if atoms.len() > 200 {
            return Err(error("The public LigParGen service supports at most 200 atoms including hydrogens; import an explicit OPLS topology for this ligand"));
        }
        let generated = bio_apis::ligpargen::parameterize(&request.upload, request.charge)
            .map_err(|e| error(format!("Unable to generate OPLS parameters with LigParGen: {e}")))?;
        let template = Ligand::from_text(&generated.sdf, &generated.itp)?;
        let mut prepared = atoms.to_vec();
        let params = template.build_generated(&mut prepared, bonds)?
            .ok_or_else(|| error("LigParGen changed the ligand's elements, hydrogens or bond graph; generated parameters were rejected"))?;
        let q: f32 = prepared.iter().map(|a| a.partial_charge.unwrap()).sum();
        if !q.is_finite() || (q - request.charge as f32).abs() > 0.005 {
            return Err(error("LigParGen output charge does not match the input protonation state"));
        }
        let molecule = MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: prepared.clone(), bonds: bonds.to_vec(),
            explicit_params: Some(params.clone()), ..Default::default()
        };
        let set = FfParamSet::new_opls().map_err(|e| error(e.to_string()))?;
        let exported = gromacs::make_topology(&[("LIG".into(), molecule, 1)], &set, None, false)?;
        // Export uses unique per-atom names; restore the original OPLS labels
        // in this single-molecule cache so reopening preserves displayed types.
        let top = exported.lines().map(|line| {
            line.split_whitespace().map(|token| {
                token.strip_prefix("DYN_0_").and_then(|i| i.parse::<usize>().ok())
                    .and_then(|i| prepared.get(i))
                    .and_then(|a| a.force_field_type.as_deref()).unwrap_or(token)
            }).collect::<Vec<_>>().join(" ")
        }).collect::<Vec<_>>().join("\n");
        let cache = format!("{}{SEPARATOR}; LigParGen job {}\n{top}", request.mol.trim_end(), generated.job_id);
        let template = Ligand::from_text(&request.mol, &top)?;
        self.memory.lock().unwrap().insert(request.key.clone(), template);
        let directory = cache_dir();
        let path = directory.join(format!("{}.opls", request.key));
        let temporary = directory.join(format!("{}.{}.{}.tmp", request.key, std::process::id(), TEMP_ID.fetch_add(1, Ordering::Relaxed)));
        let persist = || -> std::io::Result<()> {
            fs::create_dir_all(&directory)?;
            fs::write(&temporary, cache)?;
            // Windows rename cannot replace an old/corrupt cache file.
            if path.exists() { fs::remove_file(&path)?; }
            fs::rename(&temporary, &path)
        };
        if let Err(e) = persist() {
            eprintln!("Unable to save OPLS cache {}: {e}", path.display());
            let _ = fs::remove_file(temporary);
        }
        atoms.clone_from_slice(&prepared);
        Ok(params)
    }
}
