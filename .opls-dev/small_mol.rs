//! Native OPLS-AA small-molecule typing using the bundled Foyer rule set.
//! Atom types, charges and bonded parameters all come from that same set.
//! No GAFF2 parameters or charge inference are used here.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fmt::Write,
};

use bio_files::{AtomGeneric, BondGeneric, gromacs::GromacsTopology};

use crate::{
    ParamError,
    import::{ExplicitParams, gromacs_molecule_params},
};

use super::smarts::{Graph, Pattern};

const DATA: &str = include_str!("../../param_data/opls/foyer-ligands.tsv");

#[derive(Debug)]
struct AtomType {
    class: String,
    mass: f64,
    charge: f64,
    sigma: f64,
    epsilon: f64,
}

#[derive(Debug)]
struct Rule {
    name: String,
    pattern: Pattern,
    overrides: Vec<String>,
}

#[derive(Debug)]
struct Term {
    classes: Vec<String>,
    values: Vec<f64>,
}

#[derive(Debug)]
pub(super) struct SmallMolSet {
    atom_types: HashMap<String, AtomType>,
    rules: Vec<Rule>,
    bonds: Vec<Term>,
    angles: Vec<Term>,
    torsions: Vec<Term>,
    known: super::known::Ligand,
    generator: super::ligpargen::Generator,
}

fn err(message: impl AsRef<str>) -> ParamError {
    ParamError::new(message.as_ref())
}

impl SmallMolSet {
    pub fn new() -> Result<Self, ParamError> {
        let mut result = Self {
            atom_types: HashMap::new(),
            rules: Vec::new(),
            bonds: Vec::new(),
            angles: Vec::new(),
            torsions: Vec::new(),
            known: super::known::Ligand::creatine()?,
            generator: Default::default(),
        };
        for line in DATA
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let cols: Vec<_> = line.split('\t').collect();
            let number = |i: usize| {
                cols.get(i)
                    .ok_or_else(|| err("Invalid bundled OPLS data"))?
                    .parse::<f64>()
                    .map_err(|_| err("Invalid bundled OPLS number"))
            };
            if cols[0] == "T" && cols.len() == 10 {
                let name = cols[1].to_owned();
                result.atom_types.insert(
                    name.clone(),
                    AtomType {
                        class: cols[2].to_owned(),
                        mass: number(4)?,
                        charge: number(5)?,
                        sigma: number(6)?,
                        epsilon: number(7)?,
                    },
                );
                if !cols[9].is_empty() {
                    result.rules.push(Rule {
                        name,
                        pattern: Pattern::parse(cols[9])?,
                        overrides: cols[8]
                            .split(',')
                            .filter(|v| !v.is_empty())
                            .map(str::to_owned)
                            .collect(),
                    });
                }
                continue;
            }
            let arity = match cols[0] {
                "B" => 2,
                "A" => 3,
                "R" => 4,
                _ => return Err(err("Invalid bundled OPLS record")),
            };
            let expected = 1 + arity + if arity == 4 { 6 } else { 2 };
            if cols.len() != expected {
                return Err(err("Invalid bundled OPLS bonded record"));
            }
            let term = Term {
                classes: cols[1..=arity].iter().map(|s| s.to_string()).collect(),
                values: (1 + arity..cols.len())
                    .map(number)
                    .collect::<Result<_, _>>()?,
            };
            match arity {
                2 => result.bonds.push(term),
                3 => result.angles.push(term),
                _ => result.torsions.push(term),
            }
        }
        Ok(result)
    }

    fn type_atoms(&self, atoms: &[AtomGeneric], graph: &Graph) -> Result<Vec<String>, ParamError> {
        let mut labels = vec![HashSet::<String>::new(); atoms.len()];
        let mut excluded = labels.clone();
        let mut converged = false;
        // Foyer's ordered, accumulating whitelist/blacklist iteration. Type
        // references see all matches, including subsequently overridden types.
        for _ in 0..10 {
            let mut changed = false;
            for rule in &self.rules {
                for i in 0..atoms.len() {
                    if !labels[i].contains(&rule.name) && rule.pattern.matches(i, graph, &labels) {
                        labels[i].insert(rule.name.clone());
                        excluded[i].extend(rule.overrides.iter().cloned());
                        changed = true;
                    }
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(err("OPLS ligand typing did not converge"));
        }
        let mut names = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            let candidates: BTreeSet<_> = labels[i].difference(&excluded[i]).cloned().collect();
            if candidates.len() != 1 {
                let reason = if candidates.is_empty() {
                    "no matching type".to_owned()
                } else {
                    format!(
                        "ambiguous types: {}",
                        candidates.into_iter().collect::<Vec<_>>().join(", ")
                    )
                };
                return Err(err(format!(
                    "OPLS-AA cannot type ligand atom {} ({}): {reason}. The bundled Foyer rules cover a subset of OPLS chemistry. Include all hydrogens; unsupported chemistry needs an explicit OPLS topology.",
                    atom.serial_number,
                    atom.element.to_letter(),
                )));
            }
            names.push(candidates.into_iter().next().unwrap());
        }
        Ok(names)
    }

    pub fn remember(&self, mol: &crate::MolDynamics) {
        self.generator.remember(mol);
    }

    pub fn build_cached(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        if let Ok(Some(params)) = self.generator.cached(atoms, bonds) {
            return Ok(params);
        }
        self.build_bundled(atoms, bonds)
    }

    pub fn build(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        if let Ok(params) = self.build_cached(atoms, bonds) {
            return Ok(params);
        }
        self.generator.generate(atoms, bonds)
    }

    fn build_bundled(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<ExplicitParams, ParamError> {
        if atoms.is_empty() {
            return Err(err("Cannot parameterize an empty OPLS ligand"));
        }
        let graph = Graph::new(atoms, bonds)?;
        if let Some(params) = self.known.build(atoms, bonds)? {
            return Ok(params);
        }
        let names = self.type_atoms(atoms, &graph)?;
        let types: Vec<_> = names.iter().map(|n| &self.atom_types[n]).collect();
        let total_charge: f64 = types.iter().map(|a| a.charge).sum();
        if (total_charge - total_charge.round()).abs() > 0.005 {
            return Err(err(format!(
                "OPLS ligand typing gives a non-integral charge ({total_charge:.5} e); this chemical environment needs explicit OPLS parameters"
            )));
        }
        let mut text = String::from("[ defaults ]\n1 3 yes 0.5 0.5\n[ atomtypes ]\n");
        let mut seen = HashSet::new();
        for (i, t) in types.iter().enumerate() {
            if seen.insert(&names[i]) {
                writeln!(
                    text,
                    "{} {:.12} 0 A {:.12e} {:.12e}",
                    names[i], t.mass, t.sigma, t.epsilon
                )
                .unwrap();
            }
        }
        text.push_str("[ moleculetype ]\nOPLS_LIGAND 3\n[ atoms ]\n");
        for (i, t) in types.iter().enumerate() {
            writeln!(
                text,
                "{} {} 1 LIG A{} {} {:.12} {:.12}",
                i + 1,
                names[i],
                i + 1,
                i + 1,
                t.charge,
                t.mass
            )
            .unwrap();
        }
        let lookup = |ids: &[usize],
                      terms: &[Term],
                      specific: bool|
         -> Result<Vec<f64>, ParamError> {
            let classes: Vec<_> = ids.iter().map(|&i| types[i].class.as_str()).collect();
            let mut found = None;
            for term in terms {
                let matches = |reverse: bool| {
                    term.classes.iter().enumerate().all(|(i, c)| {
                        c.is_empty() || c == classes[if reverse { ids.len() - 1 - i } else { i }]
                    })
                };
                if matches(false) || matches(true) {
                    if found.is_none() {
                        found = Some(term);
                    }
                    if !specific || term.classes.iter().all(|c| !c.is_empty()) {
                        found = Some(term);
                        break;
                    }
                }
            }
            found.map(|t| t.values.clone()).ok_or_else(|| err(format!(
                "Missing OPLS-AA {}-atom bonded parameter for {}; an explicit OPLS topology is required for this ligand",
                ids.len(), ids.iter().map(|&i| names[i].as_str()).collect::<Vec<_>>().join("-"),
            )))
        };
        text.push_str("[ bonds ]\n");
        for (i, neighbors) in graph.adj.iter().enumerate() {
            for &j in neighbors.iter().filter(|&&j| j > i) {
                let v = lookup(&[i, j], &self.bonds, false)?;
                writeln!(text, "{} {} 1 {:.12e} {:.12e}", i + 1, j + 1, v[0], v[1]).unwrap();
            }
        }
        text.push_str("[ angles ]\n");
        for (j, neighbors) in graph.adj.iter().enumerate() {
            for (a, &i) in neighbors.iter().enumerate() {
                for &k in &neighbors[a + 1..] {
                    let v = lookup(&[i, j, k], &self.angles, false)?;
                    writeln!(
                        text,
                        "{} {} {} 1 {:.12e} {:.12e}",
                        i + 1,
                        j + 1,
                        k + 1,
                        v[0].to_degrees(),
                        v[1]
                    )
                    .unwrap();
                }
            }
        }
        text.push_str("[ dihedrals ]\n");
        for (j, neighbors) in graph.adj.iter().enumerate() {
            for &k in neighbors.iter().filter(|&&k| k > j) {
                for &i in neighbors.iter().filter(|&&i| i != k) {
                    for &l in graph.adj[k].iter().filter(|&&l| l != j && l != i) {
                        let v = lookup(&[i, j, k, l], &self.torsions, true)?;
                        writeln!(
                            text,
                            "{} {} {} {} 3 {}",
                            i + 1,
                            j + 1,
                            k + 1,
                            l + 1,
                            v.iter()
                                .map(|c| format!("{c:.12e}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        )
                        .unwrap();
                    }
                }
            }
        }
        // Shortest-path exclusions: retain one scaled pair per 1-4 pair and
        // keep ring 1-2/1-3 pairs excluded even if a 3-bond path also exists.
        text.push_str("[ pairs ]\n");
        for start in 0..atoms.len() {
            let mut depth = vec![usize::MAX; atoms.len()];
            depth[start] = 0;
            let mut queue = VecDeque::from([start]);
            while let Some(i) = queue.pop_front() {
                if depth[i] == 3 {
                    continue;
                }
                for &j in &graph.adj[i] {
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
        let top =
            GromacsTopology::new(&text).map_err(|e| err(format!("OPLS ligand topology: {e}")))?;
        let params = gromacs_molecule_params(&top.molecule_types[0], &top)?;
        // Commit typing only after every atom and bonded term has parameters.
        for ((a, name), t) in atoms.iter_mut().zip(names).zip(types) {
            a.force_field_type = Some(name);
            a.partial_charge = Some(t.charge as f32);
        }
        Ok(params)
    }
}
