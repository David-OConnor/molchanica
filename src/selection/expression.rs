//! Atom selection expressions shared by the CLI, saved selections, and representation layers.
//! Saved expressions refer to object paths/identifiers and atom serial numbers, never vector indices.

use std::collections::{HashMap, HashSet};

use bio_files::ResidueType;
use lin_alg::f64::Vec3;
use mol_defs::molecules::{AtomRole, MolGenericRef, MolType};
use na_seq::Element;

use crate::{
    selection::{MoleculeAtoms, Selection},
    state::State,
};

#[derive(Clone, Debug, PartialEq)]
pub struct NamedSelection {
    pub name: String,
    pub expression: String,
}

pub fn objects(state: &State) -> Vec<(MolType, usize, MolGenericRef<'_>)> {
    state
        .peptides
        .iter()
        .enumerate()
        .map(|(i, m)| (MolType::Peptide, i, MolGenericRef::Peptide(m)))
        .chain(
            state
                .ligands
                .iter()
                .enumerate()
                .map(|(i, m)| (MolType::Ligand, i, MolGenericRef::Small(m))),
        )
        .chain(
            state
                .nucleic_acids
                .iter()
                .enumerate()
                .map(|(i, m)| (MolType::NucleicAcid, i, MolGenericRef::NucleicAcid(m))),
        )
        .chain(
            state
                .lipids
                .iter()
                .enumerate()
                .map(|(i, m)| (MolType::Lipid, i, MolGenericRef::Lipid(m))),
        )
        .chain(
            state
                .pockets
                .iter()
                .enumerate()
                .map(|(i, m)| (MolType::Pocket, i, MolGenericRef::Pocket(m))),
        )
        .collect()
}

pub fn object_key(mol: &MolGenericRef<'_>) -> String {
    mol.common()
        .path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| mol.common().ident.clone())
}

pub fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn object_expression(mol: &MolGenericRef<'_>) -> String {
    format!("model {}", quote(&object_key(mol)))
}

#[derive(Debug)]
enum Expr {
    Selector(String, String),
    Named(String),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    ByRes(Box<Expr>),
    Within(Box<Expr>, f64, Box<Expr>),
}

fn tokens(input: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        let mut token = String::new();
        if c == '"' || c == '\'' {
            let mut closed = false;
            while let Some(next) = chars.next() {
                if next == c {
                    closed = true;
                    break;
                }
                if next == '\\' && chars.peek().is_some_and(|&p| p == c || p == '\\') {
                    token.push(chars.next().unwrap());
                } else {
                    token.push(next);
                }
            }
            if !closed {
                return Err("Unclosed quote in selection".into());
            }
        } else if "()&|!<>=∧∨".contains(c) {
            token.push(c);
            if "<>=!".contains(c) && chars.peek() == Some(&'=') {
                token.push(chars.next().unwrap());
            }
        } else {
            token.push(c);
            while let Some(&next) = chars.peek() {
                if next.is_whitespace() || "()&|!<>=".contains(next) {
                    break;
                }
                token.push(chars.next().unwrap());
            }
        }
        result.push(token);
        if result.len() > 512 {
            return Err("Selection is too long (maximum 512 tokens)".into());
        }
    }
    Ok(result)
}

struct Parser {
    tokens: Vec<String>,
    i: usize,
}

impl Parser {
    fn eat(&mut self, value: &str) -> bool {
        if self
            .tokens
            .get(self.i)
            .is_some_and(|t| t.eq_ignore_ascii_case(value))
        {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn next(&mut self) -> Result<String, String> {
        let result = self
            .tokens
            .get(self.i)
            .cloned()
            .ok_or("Incomplete selection expression")?;
        self.i += 1;
        Ok(result)
    }

    fn or(&mut self, depth: usize) -> Result<Expr, String> {
        let mut left = self.and(depth)?;
        while self.eat("or") || self.eat("|") {
            left = Expr::Or(Box::new(left), Box::new(self.and(depth)?));
        }
        Ok(left)
    }

    fn and(&mut self, depth: usize) -> Result<Expr, String> {
        let mut left = self.unary(depth)?;
        while self.eat("and") || self.eat("&") {
            left = Expr::And(Box::new(left), Box::new(self.unary(depth)?));
        }
        Ok(left)
    }

    fn unary(&mut self, depth: usize) -> Result<Expr, String> {
        if depth > 64 {
            return Err("Selection nesting exceeds 64 levels".into());
        }
        if self.eat("not") || self.eat("!") {
            return Ok(Expr::Not(Box::new(self.unary(depth + 1)?)));
        }
        if self.eat("byres") || self.eat("br.") {
            return Ok(Expr::ByRes(Box::new(self.unary(depth + 1)?)));
        }
        let mut left = if self.eat("(") {
            let inner = self.or(depth + 1)?;
            if !self.eat(")") {
                return Err("Expected ')' in selection".into());
            }
            inner
        } else {
            let name = self.next()?;
            let key = name.to_ascii_lowercase();
            match key.as_str() {
                "model" | "object" | "chain" | "resi" | "resn" | "elem" | "name" | "id"
                | "index" | "alt" => {
                    let value = self.next()?;
                    if matches!(key.as_str(), "resi" | "id" | "index") {
                        number_ranges(&value)?;
                    }
                    Expr::Selector(key, value)
                }
                "q" | "charge" | "partial_charge" | "occupancy" => {
                    let op = self.next()?;
                    if !matches!(op.as_str(), "<" | ">" | "<=" | ">=" | "=" | "==" | "!=") {
                        return Err(format!("Expected a comparison after {key}"));
                    }
                    let value = self.next()?;
                    finite_number(&value)?;
                    Expr::Selector(key, format!("{op} {value}"))
                }
                "all" | "none" | "protein" | "polymer.protein" | "polymer" | "nucleic"
                | "polymer.nucleic" | "ligand" | "organic" | "lipid" | "solvent" | "water"
                | "hydro" | "hydrogen" | "backbone" | "sidechain" | "hetatm" | "sele" => {
                    Expr::Selector(key, String::new())
                }
                ")" | "and" | "or" | "within" | "of" => return Err(format!("Unexpected '{name}'")),
                _ => Expr::Named(name.trim_start_matches('%').to_owned()),
            }
        };
        while self.eat("within") {
            let distance = finite_number(&self.next()?)?;
            if distance < 0. {
                return Err("Distance must be nonnegative".into());
            }
            if !self.eat("of") {
                return Err("Expected 'of' after distance".into());
            }
            left = Expr::Within(Box::new(left), distance, Box::new(self.unary(depth + 1)?));
        }
        Ok(left)
    }
}

fn parse(input: &str) -> Result<Expr, String> {
    let mut p = Parser {
        tokens: tokens(input)?,
        i: 0,
    };
    let result = p.or(0)?;
    if p.i != p.tokens.len() {
        return Err(format!("Unexpected '{}'", p.tokens[p.i]));
    }
    Ok(result)
}

fn finite_number(value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("Invalid number '{value}'"))
}

fn number_ranges(value: &str) -> Result<Vec<(u32, u32)>, String> {
    value
        .split('+')
        .map(|part| {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            let a = a
                .parse::<u32>()
                .map_err(|_| format!("Invalid number or range '{part}'"))?;
            let b = b
                .parse::<u32>()
                .map_err(|_| format!("Invalid number or range '{part}'"))?;
            if a > b {
                return Err(format!("Reversed range '{part}'"));
            }
            Ok((a, b))
        })
        .collect()
}

fn text_matches(pattern: &str, text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    pattern.split('+').any(|pattern| {
        let pattern = pattern.to_ascii_lowercase();
        // Linear wildcard matching, without compiling a regex for every atom.
        let (p, t) = (pattern.as_bytes(), text.as_bytes());
        let (mut i, mut j, mut star, mut restart) = (0, 0, None, 0);
        while j < t.len() {
            if i < p.len() && p[i] != b'*' && p[i] == t[j] {
                i += 1;
                j += 1;
            } else if i < p.len() && p[i] == b'*' {
                star = Some(i);
                i += 1;
                restart = j;
            } else if let Some(s) = star {
                restart += 1;
                j = restart;
                i = s + 1;
            } else {
                return false;
            }
        }
        while i < p.len() && p[i] == b'*' {
            i += 1;
        }
        i == p.len()
    })
}

struct Context<'a> {
    state: &'a State,
    named: &'a [NamedSelection],
    objects: Vec<(MolType, usize, MolGenericRef<'a>)>,
    atoms: Vec<(usize, usize)>,
}

impl Context<'_> {
    fn position(&self, i: usize) -> Vec3 {
        let (o, a) = self.atoms[i];
        let mol = self.objects[o].2.common();
        mol.atom_posits
            .get(a)
            .copied()
            .unwrap_or(mol.atoms[a].posit)
    }

    fn eval(
        &self,
        expr: &Expr,
        visiting: &mut Vec<String>,
        memo: &mut HashMap<String, Vec<bool>>,
        depth: usize,
    ) -> Result<Vec<bool>, String> {
        if depth > 128 {
            return Err("Selection evaluation exceeds 128 levels".into());
        }
        let depth = depth + 1;
        let n = self.atoms.len();
        match expr {
            Expr::Named(name) => {
                if let Some(mask) = memo.get(name) {
                    return Ok(mask.clone());
                }
                if visiting.len() >= 64 || visiting.contains(name) {
                    return Err(format!("Circular or excessively nested selection '{name}'"));
                }
                let saved = self.named.iter().find(|s| s.name == *name).ok_or_else(|| {
                    format!("Unknown selection '{name}' (use model for an object)")
                })?;
                visiting.push(name.clone());
                let result = self.eval(&parse(&saved.expression)?, visiting, memo, depth);
                visiting.pop();
                if let Ok(mask) = &result {
                    memo.insert(name.clone(), mask.clone());
                }
                result
            }
            Expr::Not(inner) => Ok(self
                .eval(inner, visiting, memo, depth)?
                .into_iter()
                .map(|v| !v)
                .collect()),
            Expr::And(a, b) | Expr::Or(a, b) => {
                let a = self.eval(a, visiting, memo, depth)?;
                let b = self.eval(b, visiting, memo, depth)?;
                Ok(a.into_iter()
                    .zip(b)
                    .map(|(a, b)| {
                        if matches!(expr, Expr::And(..)) {
                            a && b
                        } else {
                            a || b
                        }
                    })
                    .collect())
            }
            Expr::ByRes(inner) => {
                let mask = self.eval(inner, visiting, memo, depth)?;
                let residues: HashSet<_> = self
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask[*i])
                    .filter_map(|(_, &(o, a))| {
                        self.objects[o].2.common().atoms[a].residue.map(|r| (o, r))
                    })
                    .collect();
                Ok(self
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(i, &(o, a))| {
                        mask[i]
                            || self.objects[o].2.common().atoms[a]
                                .residue
                                .is_some_and(|r| residues.contains(&(o, r)))
                    })
                    .collect())
            }
            Expr::Within(left, distance, right) => {
                let left = self.eval(left, visiting, memo, depth)?;
                let right = self.eval(right, visiting, memo, depth)?;
                let cell_size = distance.max(1.);
                let key = |p: Vec3| {
                    (
                        (p.x / cell_size).floor() as i64,
                        (p.y / cell_size).floor() as i64,
                        (p.z / cell_size).floor() as i64,
                    )
                };
                let mut cells: HashMap<_, Vec<Vec3>> = HashMap::new();
                for (i, selected) in right.iter().enumerate() {
                    if *selected {
                        let p = self.position(i);
                        cells.entry(key(p)).or_default().push(p);
                    }
                }
                let mut result = vec![false; n];
                for (i, selected) in left.iter().enumerate() {
                    if !selected {
                        continue;
                    }
                    let p = self.position(i);
                    let (x, y, z) = key(p);
                    'neighbors: for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                if cells
                                    .get(&(
                                        x.saturating_add(dx),
                                        y.saturating_add(dy),
                                        z.saturating_add(dz),
                                    ))
                                    .is_some_and(|ps| {
                                        ps.iter().any(|q| {
                                            (*q - p).magnitude_squared() <= distance * distance
                                        })
                                    })
                                {
                                    result[i] = true;
                                    break 'neighbors;
                                }
                            }
                        }
                    }
                }
                Ok(result)
            }
            Expr::Selector(key, value) => {
                let ranges = if matches!(key.as_str(), "resi" | "id" | "index") {
                    number_ranges(value)?
                } else {
                    Vec::new()
                };
                let selected = if key == "sele" {
                    current_atoms(self.state)
                } else {
                    Vec::new()
                };
                Ok(self
                    .atoms
                    .iter()
                    .map(|&(o, a)| {
                        let (kind, mol_i, mol) = &self.objects[o];
                        let atom = &mol.common().atoms[a];
                        let residue = match mol {
                            MolGenericRef::Peptide(p) => {
                                atom.residue.and_then(|r| p.residues.get(r))
                            }
                            MolGenericRef::NucleicAcid(p) => {
                                atom.residue.and_then(|r| p.residues.get(r))
                            }
                            MolGenericRef::Lipid(p) => atom.residue.and_then(|r| p.residues.get(r)),
                            _ => None,
                        };
                        let protein = residue
                            .is_some_and(|r| matches!(r.res_type, ResidueType::AminoAcid(_)));
                        let water = atom.role == Some(AtomRole::Water)
                            || residue.is_some_and(|r| r.res_type == ResidueType::Water);
                        let numeric =
                            |number| ranges.iter().any(|&(lo, hi)| number >= lo && number <= hi);
                        match key.as_str() {
                            "all" => true,
                            "none" => false,
                            "model" | "object" => {
                                text_matches(value, &object_key(mol))
                                    || text_matches(value, &mol.common().ident)
                                    || (!mol.common().filename.is_empty()
                                        && text_matches(value, &mol.common().filename))
                                    || mol
                                        .common()
                                        .name
                                        .as_ref()
                                        .is_some_and(|name| text_matches(value, name))
                            }
                            "chain" => match mol {
                                MolGenericRef::Peptide(p) => atom
                                    .chain
                                    .and_then(|c| p.chains.get(c))
                                    .is_some_and(|c| text_matches(value, &c.id)),
                                _ => false,
                            },
                            "resi" => residue.is_some_and(|r| numeric(r.serial_number)),
                            "resn" => residue.is_some_and(|r| match &r.res_type {
                                ResidueType::AminoAcid(aa) => {
                                    text_matches(value, &aa.to_str(na_seq::AaIdent::ThreeLetters))
                                }
                                ResidueType::Water => {
                                    text_matches(value, "HOH") || text_matches(value, "WAT")
                                }
                                ResidueType::Other(name) => text_matches(value, name),
                            }),
                            "elem" => text_matches(value, &atom.element.to_letter()),
                            "name" => {
                                atom.type_in_res_general
                                    .as_ref()
                                    .is_some_and(|s| text_matches(value, s))
                                    || atom
                                        .type_in_res
                                        .as_ref()
                                        .is_some_and(|s| text_matches(value, &s.to_string()))
                            }
                            "alt" => text_matches(
                                value,
                                atom.alt_conformation_id.as_deref().unwrap_or(""),
                            ),
                            "id" => numeric(atom.serial_number),
                            "index" => numeric(a as u32 + 1),
                            "protein" | "polymer.protein" => protein,
                            "polymer" => protein || *kind == MolType::NucleicAcid,
                            "nucleic" | "polymer.nucleic" => *kind == MolType::NucleicAcid,
                            "ligand" | "organic" => {
                                *kind == MolType::Ligand
                                    || (*kind == MolType::Peptide
                                        && atom.hetero
                                        && !water
                                        && !protein)
                            }
                            "lipid" => *kind == MolType::Lipid,
                            "water" | "solvent" => water,
                            "hydro" | "hydrogen" => atom.element == Element::Hydrogen,
                            "backbone" => atom.is_backbone(),
                            "sidechain" => protein && !atom.is_backbone(),
                            "hetatm" => atom.hetero,
                            "sele" => selected.iter().any(|s| {
                                s.mol_type == *kind
                                    && s.mol_i == *mol_i
                                    && s.atoms.binary_search(&a).is_ok()
                            }),
                            "charge" | "partial_charge" | "q" | "occupancy" => {
                                let number = if key == "occupancy" || key == "q" {
                                    atom.occupancy
                                } else {
                                    atom.partial_charge
                                };
                                let (op, v) = value.split_once(' ').unwrap();
                                let v: f64 = v.parse().unwrap();
                                number.is_some_and(|n| match op {
                                    "<" => (n as f64) < v,
                                    ">" => (n as f64) > v,
                                    "<=" => (n as f64) <= v,
                                    ">=" => (n as f64) >= v,
                                    "!=" => n as f64 != v,
                                    _ => n as f64 == v,
                                })
                            }
                            _ => false,
                        }
                    })
                    .collect())
            }
        }
    }
}

pub fn evaluate_with_named(
    state: &State,
    expression: &str,
    named: &[NamedSelection],
) -> Result<Vec<MoleculeAtoms>, String> {
    let objects = objects(state);
    let atoms = objects
        .iter()
        .enumerate()
        .flat_map(|(o, (_, _, mol))| (0..mol.common().atoms.len()).map(move |a| (o, a)))
        .collect();
    let context = Context {
        state,
        named,
        objects,
        atoms,
    };
    let mask = context.eval(&parse(expression)?, &mut Vec::new(), &mut HashMap::new(), 0)?;
    let mut result = Vec::new();
    let mut offset = 0;
    for (kind, i, mol) in &context.objects {
        let atoms: Vec<_> = (0..mol.common().atoms.len())
            .filter(|a| mask[offset + a])
            .collect();
        offset += mol.common().atoms.len();
        if !atoms.is_empty() {
            result.push(MoleculeAtoms {
                mol_type: *kind,
                mol_i: *i,
                atoms,
            });
        }
    }
    Ok(result)
}

pub fn evaluate(state: &State, expression: &str) -> Result<Vec<MoleculeAtoms>, String> {
    evaluate_with_named(state, expression, &state.to_save.named_selections)
}

pub fn save_named(state: &mut State, name: &str, expression: &str) -> Result<usize, String> {
    if name.is_empty()
        || !name
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        || !matches!(parse(name)?, Expr::Named(_))
    {
        return Err(
            "Selection names must start with a letter or underscore and must not be reserved words"
                .into(),
        );
    }
    // Resolve `sele` now so recalling the selection never depends on whichever item is picked later.
    let expression = freeze_current(state, expression)?;
    let mut named = state.to_save.named_selections.clone();
    let entry = NamedSelection {
        name: name.into(),
        expression,
    };
    if let Some(old) = named.iter_mut().find(|s| s.name == name) {
        *old = entry;
    } else {
        named.push(entry);
    }
    let result = evaluate_with_named(state, name, &named)?;
    let count = result.iter().map(|s| s.atoms.len()).sum();
    state.to_save.named_selections = named;
    state.ui.selection = Selection::Atoms(result);
    state.to_save.save_flag = true;
    Ok(count)
}

fn references(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Named(n) => n == name,
        Expr::Not(a) | Expr::ByRes(a) => references(a, name),
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Within(a, _, b) => {
            references(a, name) || references(b, name)
        }
        Expr::Selector(..) => false,
    }
}

pub fn delete_named(state: &mut State, name: &str) -> Result<(), String> {
    for selection in &state.to_save.named_selections {
        if selection.name != name && references(&parse(&selection.expression)?, name) {
            return Err(format!(
                "Selection '{}' uses '{name}'; edit or delete it first",
                selection.name
            ));
        }
    }
    for layer in &state.to_save.representations {
        if references(&parse(&layer.expression)?, name) {
            return Err(format!(
                "Layer '{}' uses '{name}'; edit or delete it first",
                layer.name
            ));
        }
    }
    if !state
        .to_save
        .named_selections
        .iter()
        .any(|s| s.name == name)
    {
        return Err(format!("Unknown selection '{name}'"));
    }
    state.to_save.named_selections.retain(|s| s.name != name);
    state.to_save.save_flag = true;
    Ok(())
}

pub fn freeze_current(state: &State, input: &str) -> Result<String, String> {
    let mut ts = tokens(input)?;
    // Only substitute a selector token; quoted model names are handled by the parser instead.
    let snapshot = current_expression(state);
    let mut value_next = false;
    for token in &mut ts {
        if value_next {
            *token = quote(token);
            value_next = false;
            continue;
        }
        let key = token.to_ascii_lowercase();
        if key == "sele" {
            *token = format!("({snapshot})");
        }
        value_next = matches!(
            key.as_str(),
            "model"
                | "object"
                | "chain"
                | "resi"
                | "resn"
                | "elem"
                | "name"
                | "id"
                | "index"
                | "alt"
        );
    }
    Ok(ts.join(" "))
}

pub fn current_atoms(state: &State) -> Vec<MoleculeAtoms> {
    use Selection::*;
    if let Atoms(items) = &state.ui.selection {
        return items.clone();
    }
    let peptide = state.peptide_for_tools_i().unwrap_or(usize::MAX);
    let (kind, i, mut atoms, bonds, residues) = match &state.ui.selection {
        AtomPeptide(a) => (MolType::Peptide, peptide, vec![*a], vec![], vec![]),
        AtomsPeptide(a) => (MolType::Peptide, peptide, a.clone(), vec![], vec![]),
        BondPeptide(b) => (MolType::Peptide, peptide, vec![], vec![*b], vec![]),
        Residue(r) => (MolType::Peptide, peptide, vec![], vec![], vec![*r]),
        Residues(r) => (MolType::Peptide, peptide, vec![], vec![], r.clone()),
        AtomLig((i, a)) => (MolType::Ligand, *i, vec![*a], vec![], vec![]),
        AtomsLig((i, a)) => (MolType::Ligand, *i, a.clone(), vec![], vec![]),
        BondLig((i, b)) => (MolType::Ligand, *i, vec![], vec![*b], vec![]),
        BondsLig((i, b)) => (MolType::Ligand, *i, vec![], b.clone(), vec![]),
        AtomNucleicAcid((i, a)) => (MolType::NucleicAcid, *i, vec![*a], vec![], vec![]),
        BondNucleicAcid((i, b)) => (MolType::NucleicAcid, *i, vec![], vec![*b], vec![]),
        AtomLipid((i, a)) => (MolType::Lipid, *i, vec![*a], vec![], vec![]),
        BondLipid((i, b)) => (MolType::Lipid, *i, vec![], vec![*b], vec![]),
        AtomPocket((i, a)) => (MolType::Pocket, *i, vec![*a], vec![], vec![]),
        BondPocket((i, b)) => (MolType::Pocket, *i, vec![], vec![*b], vec![]),
        _ => return Vec::new(),
    };
    let Some(mol) = state.get_mol(kind, i) else {
        return Vec::new();
    };
    atoms.extend(
        bonds
            .iter()
            .filter_map(|b| mol.common().bonds.get(*b))
            .flat_map(|b| [b.atom_0, b.atom_1]),
    );
    if let MolGenericRef::Peptide(p) = mol {
        atoms.extend(
            residues
                .iter()
                .filter_map(|r| p.residues.get(*r))
                .flat_map(|r| r.atoms.iter().copied()),
        );
    }
    atoms.sort_unstable();
    atoms.dedup();
    vec![MoleculeAtoms {
        mol_type: kind,
        mol_i: i,
        atoms,
    }]
}

pub fn current_expression(state: &State) -> String {
    let parts: Vec<_> = current_atoms(state)
        .iter()
        .filter_map(|s| {
            let mol = state.get_mol(s.mol_type, s.mol_i)?;
            let ids: Vec<_> = s
                .atoms
                .iter()
                .filter_map(|a| mol.common().atoms.get(*a))
                .map(|a| a.serial_number.to_string())
                .collect();
            if ids.is_empty() {
                return None;
            }
            Some(format!(
                "({} and id {})",
                object_expression(&mol),
                ids.join("+")
            ))
        })
        .collect();
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(" or ")
    }
}

pub fn positions(state: &State) -> Vec<Vec3> {
    current_atoms(state)
        .iter()
        .flat_map(|s| {
            state
                .get_mol(s.mol_type, s.mol_i)
                .map(|m| {
                    s.atoms
                        .iter()
                        .filter_map(|a| m.common().atom_posits.get(*a).copied())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect()
}
