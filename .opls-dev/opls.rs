//! Bundled OPLS-AA/L (GROMACS 2026.4). Bonded terms are converted to the same
//! harmonic/periodic forms and units used by Amber in this library.
//!
//! Protein typing uses residue templates, including the Kaminski side-chain
//! torsions. Ligands use Foyer's OPLS-AA SMARTS definitions, charges and bonded
//! tables, with automatic LigParGen generation and a persistent cache for
//! chemistry outside those rules. GAFF2 parameters are not transferable to OPLS.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt::Write,
    io,
};

use bio_files::{
    AtomGeneric, BondGeneric, ResidueGeneric, ResidueType, gromacs::top_parse::GromacsTopology,
};
use na_seq::{AminoAcid, Element};

use crate::{
    ParamError,
    import::{ExplicitParams, gromacs_molecule_params},
};

const NONBONDED: &str = include_str!("../param_data/opls/ffnonbonded.itp");
const BONDED: &str = include_str!("../param_data/opls/ffbonded.itp");
const RESIDUES: &str = include_str!("../param_data/opls/aminoacids.rtp");

#[derive(Debug, Clone)]
struct TemplateAtom {
    name: String,
    ff_type: String,
    charge: f32,
}

#[derive(Debug, Clone, Default)]
struct Template {
    atoms: Vec<TemplateAtom>,
    bonds: Vec<(String, String)>,
    dihedrals: Vec<Vec<String>>,
    impropers: Vec<Vec<String>>,
}

#[derive(Debug)]
pub struct OplsSet {
    templates: HashMap<String, Template>,
    small_mol: small_mol::SmallMolSet,
}

mod known;
pub mod ligpargen;
mod small_mol;
mod smarts;

fn error(message: impl AsRef<str>) -> ParamError {
    ParamError::new(message.as_ref())
}

fn name(atom: &AtomGeneric) -> String {
    atom.type_in_res
        .as_ref()
        .map(ToString::to_string)
        .or_else(|| atom.type_in_res_general.clone())
        .unwrap_or_default()
}

impl OplsSet {
    pub fn new() -> io::Result<Self> {
        let mut templates = HashMap::<String, Template>::new();
        let mut residue = String::new();
        let mut section = String::new();

        for line in RESIDUES.lines() {
            let line = line.split(';').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                let tag = line.trim_matches(['[', ']']).trim();
                if ["atoms", "bonds", "dihedrals", "impropers", "exclusions"].contains(&tag) {
                    section = tag.to_owned();
                } else {
                    residue = tag.to_owned();
                    section.clear();
                }
                continue;
            }
            let cols: Vec<_> = line.split_whitespace().collect();
            let template = templates.entry(residue.clone()).or_default();
            match section.as_str() {
                "atoms" => template.atoms.push(TemplateAtom {
                    name: cols[0].into(),
                    ff_type: cols[1].into(),
                    charge: cols[2]
                        .parse()
                        .map_err(|_| io::Error::other("Invalid OPLS charge"))?,
                }),
                "bonds" => template.bonds.push((cols[0].into(), cols[1].into())),
                "dihedrals" => template
                    .dihedrals
                    .push(cols.iter().map(|c| c.to_string()).collect()),
                "impropers" => template
                    .impropers
                    .push(cols.iter().map(|c| c.to_string()).collect()),
                _ => (),
            }
        }
        let small_mol = small_mol::SmallMolSet::new()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.descrip))?;
        Ok(Self {
            templates,
            small_mol,
        })
    }

    /// Assign OPLS-AA parameters to a hydrogenated ligand. Uses validated
    /// cached/generated parameters or complete bundled rules; unsupported local
    /// chemistry is submitted to the public LigParGen service and cached.
    /// This can block on network I/O. Use `build_small_molecule_cached` on UI
    /// threads; it never contacts a service. Existing labels change on success.
    pub fn build_small_molecule(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        self.small_mol.build(atoms, bonds)
    }

    /// Resolve a ligand using only bundled data or the local LigParGen cache.
    /// A cache miss outside the bundled chemistry returns a parameter error.
    pub fn build_small_molecule_cached(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        self.small_mol.build_cached(atoms, bonds)
    }

    /// Retain a validated background result for native and export consumers,
    /// even if the filesystem cache could not be written.
    pub fn remember_small_molecule(&self, mol: &crate::MolDynamics) {
        self.small_mol.remember(mol);
    }

    /// Build OPLS protein parameters without altering coordinates, atom ordering,
    /// serial numbers, or input hydrogen names. Residues must contain all atoms,
    /// including hydrogens. Peptide connectivity determines chain ends.
    pub fn build_peptide(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
        residues: &[ResidueGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        let n = atoms.len();
        let indices: HashMap<_, _> = atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.serial_number, i))
            .collect();
        let mut adjacency = vec![BTreeSet::new(); n];
        for bond in bonds {
            let (&i, &j) = indices
                .get(&bond.atom_0_sn)
                .zip(indices.get(&bond.atom_1_sn))
                .ok_or_else(|| error("OPLS bond refers to an absent atom"))?;
            adjacency[i].insert(j);
            adjacency[j].insert(i);
        }
        let names: Vec<_> = atoms.iter().map(name).collect();
        let mut members = Vec::new();
        let mut aas = Vec::new();
        let mut owner = vec![usize::MAX; n];
        for res in residues {
            let ResidueType::AminoAcid(aa) = res.res_type else {
                continue;
            };
            let ids: Vec<_> = res
                .atom_sns
                .iter()
                .filter_map(|s| indices.get(s).copied())
                .collect();
            if ids.is_empty() {
                continue;
            }
            for &i in &ids {
                if owner[i] != usize::MAX {
                    return Err(error("Overlapping OPLS residues"));
                }
                owner[i] = members.len();
            }
            members.push(ids);
            aas.push(aa);
        }
        if owner.contains(&usize::MAX) {
            return Err(error(
                "OPLS proteins require amino-acid residues for every atom; set MolDynamics::residues",
            ));
        }
        let mut previous = vec![None; members.len()];
        let mut next = previous.clone();
        for i in 0..n {
            if names[i] != "C" {
                continue;
            }
            for &j in &adjacency[i] {
                if names[j] == "N" && owner[i] != owner[j] {
                    next[owner[i]] = Some(owner[j]);
                    previous[owner[j]] = Some(owner[i]);
                }
            }
        }
        let mut mapped = HashMap::<(usize, String), usize>::new();
        let mut instances = Vec::new();

        for (ri, ids) in members.iter().enumerate() {
            let find = |label: &str| ids.iter().copied().find(|&i| names[i] == label);
            let h_count = |label: &str| {
                find(label)
                    .map(|i| {
                        adjacency[i]
                            .iter()
                            .filter(|&&j| atoms[j].element == Element::Hydrogen)
                            .count()
                    })
                    .unwrap_or(0)
            };
            let aa = aas[ri];
            let label = match aa {
                AminoAcid::His => match (h_count("ND1"), h_count("NE2")) {
                    (1, 0) => "HISD",
                    (0, 1) => "HISE",
                    (1, 1) => "HISH",
                    _ => return Err(error("OPLS histidine requires HD1 and/or HE2")),
                },
                AminoAcid::Cys if h_count("SG") == 1 => "CYSH",
                AminoAcid::Cys => {
                    let disulfide = find("SG").is_some_and(|i| {
                        adjacency[i]
                            .iter()
                            .any(|&j| atoms[j].element == Element::Sulfur && owner[j] != ri)
                    });
                    if !disulfide {
                        return Err(error(
                            "OPLS cysteine without HG requires a disulfide bond; thiolate is not bundled",
                        ));
                    }
                    "CYS2"
                }
                AminoAcid::Asp if h_count("OD2") == 1 => "ASPH",
                AminoAcid::Glu if h_count("OE2") == 1 => "GLUH",
                AminoAcid::Lys if h_count("NZ") == 3 => "LYSH",
                _ => residue_name(aa).ok_or_else(|| error("Unsupported OPLS amino acid"))?,
            };
            let mut template = self
                .templates
                .get(label)
                .cloned()
                .ok_or_else(|| error(format!("Missing OPLS template {label}")))?;
            let n_terminal = previous[ri].is_none();
            let c_terminal = next[ri].is_none();

            // Charged termini from aminoacids.n.tdb and aminoacids.c.tdb.
            // An isolated residue uses the zwitterion patches (different CA charge).
            if n_terminal {
                let (n_type, n_q, ca_type, ca_q, h_type, h_q, nh) = match aa {
                    AminoAcid::Pro => (
                        "opls_309",
                        -0.2,
                        "opls_246",
                        if c_terminal { 0.13 } else { 0.23 },
                        "opls_310",
                        0.31,
                        2,
                    ),
                    AminoAcid::Gly => (
                        "opls_287",
                        -0.3,
                        if c_terminal { "opls_298" } else { "opls_292B" },
                        if c_terminal { 0.09 } else { 0.19 },
                        "opls_290",
                        0.33,
                        3,
                    ),
                    _ => (
                        "opls_287",
                        -0.3,
                        if c_terminal { "opls_299" } else { "opls_293B" },
                        if c_terminal { 0.15 } else { 0.25 },
                        "opls_290",
                        0.33,
                        3,
                    ),
                };
                if h_count("N") != nh {
                    return Err(error(
                        "OPLS currently requires NH3+ (NH2+ for proline) at uncapped N termini",
                    ));
                }
                replace(&mut template, "N", n_type, n_q)?;
                replace(&mut template, "CA", ca_type, ca_q)?;
                if aa == AminoAcid::Pro {
                    replace(&mut template, "CD", "opls_296", 0.17)?;
                }
                template.atoms.retain(|a| a.name != "H");
                template.bonds.retain(|(a, b)| a != "H" && b != "H");
                template
                    .impropers
                    .retain(|d| !d.iter().take(4).any(|a| a == "-C"));
                for hi in 1..=nh {
                    let h = format!("H{hi}");
                    template.atoms.push(TemplateAtom {
                        name: h.clone(),
                        ff_type: h_type.into(),
                        charge: h_q,
                    });
                    template.bonds.push(("N".into(), h));
                }
            }
            if c_terminal {
                if !n_terminal {
                    let (t, q) = match aa {
                        AminoAcid::Gly => ("opls_284", -0.02),
                        AminoAcid::Pro => ("opls_285", -0.09),
                        _ => ("opls_283", 0.04),
                    };
                    replace(&mut template, "CA", t, q)?;
                }
                replace(&mut template, "C", "opls_271", 0.7)?;
                replace(&mut template, "O", "opls_272", -0.8)?;
                template.atoms.push(TemplateAtom {
                    name: "OXT".into(),
                    ff_type: "opls_272".into(),
                    charge: -0.8,
                });
                template.bonds.push(("C".into(), "OXT".into()));
                template
                    .impropers
                    .retain(|d| !d.iter().take(4).any(|a| a == "+N"));
                template.impropers.push(
                    ["CA", "O", "C", "OXT", "improper_O_C_X_Y"]
                        .map(str::to_owned)
                        .to_vec(),
                );
            }

            // Match heavy atoms first, then hydrogens by their bonded parent.
            // Amber and OPLS differ in methylene hydrogen numbering and Ile CD.
            for &i in ids
                .iter()
                .filter(|&&i| atoms[i].element != Element::Hydrogen)
            {
                let atom_name = match names[i].as_str() {
                    "CD1" if aa == AminoAcid::Ile => "CD",
                    "O1" if c_terminal => "O",
                    "O2" if c_terminal => "OXT",
                    a => a,
                };
                if !template.atoms.iter().any(|a| a.name == atom_name)
                    || mapped.insert((ri, atom_name.into()), i).is_some()
                {
                    return Err(error(format!(
                        "Unexpected or duplicate atom {} in OPLS {label}",
                        names[i]
                    )));
                }
            }
            for heavy in template.atoms.iter().filter(|a| !a.name.starts_with('H')) {
                let &parent = mapped
                    .get(&(ri, heavy.name.clone()))
                    .ok_or_else(|| error(format!("OPLS {label} is missing {}", heavy.name)))?;
                let expected: Vec<_> = template
                    .bonds
                    .iter()
                    .filter_map(|(a, b)| {
                        let h = if a == &heavy.name {
                            b
                        } else if b == &heavy.name {
                            a
                        } else {
                            return None;
                        };
                        h.starts_with('H').then_some(h.clone())
                    })
                    .collect();
                let mut actual: Vec<_> = adjacency[parent]
                    .iter()
                    .copied()
                    .filter(|&i| atoms[i].element == Element::Hydrogen)
                    .collect();
                if expected.len() != actual.len() {
                    return Err(error(format!(
                        "OPLS {label} {} requires {} hydrogens; found {}",
                        heavy.name,
                        expected.len(),
                        actual.len()
                    )));
                }
                for h in &expected {
                    if let Some(pos) = actual.iter().position(|&i| names[i] == *h) {
                        mapped.insert((ri, h.clone()), actual.remove(pos));
                    }
                }
                for h in expected {
                    if !mapped.contains_key(&(ri, h.clone())) {
                        mapped.insert((ri, h), actual.remove(0));
                    }
                }
            }
            for a in &template.atoms {
                let &i = mapped
                    .get(&(ri, a.name.clone()))
                    .ok_or_else(|| error("Missing OPLS template atom"))?;
                atoms[i].force_field_type = Some(a.ff_type.clone());
                atoms[i].partial_charge = Some(a.charge);
            }
            instances.push(template);
        }

        let resolve = |ri: usize, atom: &str| -> Option<usize> {
            let (r, label) = if let Some(a) = atom.strip_prefix('-') {
                (previous[ri]?, a)
            } else if let Some(a) = atom.strip_prefix('+') {
                (next[ri]?, a)
            } else {
                (ri, atom)
            };
            mapped.get(&(r, label.to_owned())).copied()
        };
        let mut propers = BTreeMap::<[usize; 4], String>::new();
        let mut impropers = Vec::new();
        for (ri, template) in instances.iter().enumerate() {
            for (a, b) in &template.bonds {
                if let (Some(i), Some(j)) = (resolve(ri, a), resolve(ri, b)) {
                    if !adjacency[i].contains(&j) {
                        return Err(error(format!("Missing OPLS {a}-{b} bond")));
                    }
                }
            }
            for d in &template.dihedrals {
                let ids: Option<Vec<_>> = d[..4].iter().map(|a| resolve(ri, a)).collect();
                if let Some(ids) = ids {
                    propers.insert(canonical(ids.try_into().unwrap()), d[4..].join(" "));
                }
            }
            for d in &template.impropers {
                let ids: Option<Vec<_>> = d[..4].iter().map(|a| resolve(ri, a)).collect();
                if let Some(ids) = ids {
                    impropers.push((ids, d[4..].join(" ")));
                }
            }
        }
        for b in 0..n {
            for &c in adjacency[b].range(b + 1..) {
                for &a in &adjacency[b] {
                    for &d in &adjacency[c] {
                        if a != c && d != b && a != d {
                            propers.entry(canonical([a, b, c, d])).or_default();
                        }
                    }
                }
            }
        }
        let mut text = format!(
            "[ defaults ]\n1 3 yes 0.5 0.5\n{NONBONDED}\n{BONDED}\n[ moleculetype ]\nOPLS_PROTEIN 3\n[ atoms ]\n"
        );
        for (i, a) in atoms.iter().enumerate() {
            writeln!(
                text,
                "{} {} {} RES A{} {} {}",
                i + 1,
                a.force_field_type.as_ref().unwrap(),
                owner[i] + 1,
                i + 1,
                i + 1,
                a.partial_charge.unwrap()
            )
            .unwrap();
        }
        text.push_str("[ bonds ]\n");
        for i in 0..n {
            for &j in adjacency[i].range(i + 1..) {
                writeln!(text, "{} {} 1", i + 1, j + 1).unwrap();
            }
        }
        text.push_str("[ angles ]\n");
        for (j, neighbors) in adjacency.iter().enumerate() {
            let v: Vec<_> = neighbors.iter().copied().collect();
            for (k, &i) in v.iter().enumerate() {
                for &l in &v[k + 1..] {
                    writeln!(text, "{} {} {} 1", i + 1, j + 1, l + 1).unwrap();
                }
            }
        }
        text.push_str("[ dihedrals ]\n");
        for ([a, b, c, d], p) in propers {
            writeln!(text, "{} {} {} {} 3 {p}", a + 1, b + 1, c + 1, d + 1).unwrap();
        }
        for (ids, p) in impropers {
            writeln!(
                text,
                "{} {} {} {} 4 {p}",
                ids[0] + 1,
                ids[1] + 1,
                ids[2] + 1,
                ids[3] + 1
            )
            .unwrap();
        }
        // Generate 1-4 pairs by shortest bond distance. Ring 1-2/1-3 exclusions
        // take precedence, and each pair appears once regardless of torsion count.
        text.push_str("[ pairs ]\n");
        for start in 0..n {
            let mut depth = vec![usize::MAX; n];
            depth[start] = 0;
            let mut queue = VecDeque::from([start]);
            while let Some(i) = queue.pop_front() {
                if depth[i] == 3 {
                    continue;
                }
                for &j in &adjacency[i] {
                    if depth[j] == usize::MAX {
                        depth[j] = depth[i] + 1;
                        queue.push_back(j);
                    }
                }
            }
            for (j, &d) in depth.iter().enumerate().skip(start + 1) {
                if d == 3 {
                    writeln!(text, "{} {} 1", start + 1, j + 1).unwrap();
                }
            }
        }
        let top = GromacsTopology::new(&text).map_err(|e| error(format!("OPLS topology: {e}")))?;
        gromacs_molecule_params(&top.molecule_types[0], &top)
    }
}

fn canonical(ids: [usize; 4]) -> [usize; 4] {
    let rev = [ids[3], ids[2], ids[1], ids[0]];
    ids.min(rev)
}

fn replace(
    template: &mut Template,
    name: &str,
    ff_type: &str,
    charge: f32,
) -> Result<(), ParamError> {
    let a = template
        .atoms
        .iter_mut()
        .find(|a| a.name == name)
        .ok_or_else(|| error("Missing OPLS terminal atom"))?;
    a.ff_type = ff_type.into();
    a.charge = charge;
    Ok(())
}

fn residue_name(aa: AminoAcid) -> Option<&'static str> {
    use AminoAcid::*;
    Some(match aa {
        Ala => "ALA",
        Arg => "ARG",
        Asn => "ASN",
        Asp => "ASP",
        Cys => "CYSH",
        Gln => "GLN",
        Glu => "GLU",
        Gly => "GLY",
        His => "HISE",
        Ile => "ILE",
        Leu => "LEU",
        Lys => "LYS",
        Met => "MET",
        Phe => "PHE",
        Pro => "PRO",
        Ser => "SER",
        Thr => "THR",
        Trp => "TRP",
        Tyr => "TYR",
        Val => "VAL",
        _ => return None,
    })
}
