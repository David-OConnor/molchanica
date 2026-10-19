//! Atom selection expressions shared by the CLI, saved selections, and representation layers.
//! Saved expressions refer to object paths/identifiers and atom serial numbers, never vector indices.

use std::collections::{HashMap, HashSet};

use bio_files::{ResidueType, SecondaryStructure};
use lin_alg::f64::Vec3;
use mol_defs::molecules::{Atom, AtomRole, MolGenericRef, MolType, Residue};
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
    Expand(String, Box<Expr>),
    Distance(String, Box<Expr>, f64, Box<Expr>),
    Bonds(String, Box<Expr>, usize),
    Match(String, Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
struct Token {
    value: String,
    quoted: bool,
}

fn tokens(input: &str) -> Result<Vec<Token>, String> {
    let mut result = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        let mut token = String::new();
        let quoted = c == '"' || c == '\'';
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
        } else if "(),&|!<>=∧∨".contains(c) {
            token.push(c);
            if ("<>=!".contains(c) && chars.peek() == Some(&'='))
                || ("&|".contains(c) && chars.peek() == Some(&c))
            {
                token.push(chars.next().unwrap());
            }
        } else {
            token.push(c);
            while let Some(&next) = chars.peek() {
                if next.is_whitespace() || "(),&|!<>=∧∨".contains(next) {
                    break;
                }
                token.push(chars.next().unwrap());
            }
        }
        result.push(Token {
            value: token,
            quoted,
        });
        if result.len() > 512 {
            return Err("Selection is too long (maximum 512 tokens)".into());
        }
    }
    Ok(result)
}

struct Parser {
    tokens: Vec<Token>,
    i: usize,
}

impl Parser {
    fn eat(&mut self, value: &str) -> bool {
        if self
            .tokens
            .get(self.i)
            .is_some_and(|t| !t.quoted && keyword(&t.value.to_ascii_lowercase()) == value)
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
            .map(|t| t.value.clone())
            .ok_or("Incomplete selection expression")?;
        self.i += 1;
        Ok(result)
    }

    fn or(&mut self, depth: usize) -> Result<Expr, String> {
        let mut left = self.and(depth)?;
        // As in PyMOL, adjacent expressions imply OR, not AND.
        while self.eat("or") || self.starts_expression() {
            left = Expr::Or(Box::new(left), Box::new(self.and(depth)?));
        }
        Ok(left)
    }

    fn and(&mut self, depth: usize) -> Result<Expr, String> {
        let mut left = self.unary(depth)?;
        while self.eat("and") {
            left = Expr::And(Box::new(left), Box::new(self.unary(depth)?));
        }
        Ok(left)
    }

    fn unary(&mut self, depth: usize) -> Result<Expr, String> {
        if depth > 64 {
            return Err("Selection nesting exceeds 64 levels".into());
        }
        if self.eat("not") {
            return Ok(Expr::Not(Box::new(self.unary(depth + 1)?)));
        }
        let key = self
            .tokens
            .get(self.i)
            .filter(|t| !t.quoted)
            .map(|t| keyword(&t.value.to_ascii_lowercase()).to_owned());
        if matches!(
            key.as_deref(),
            Some("byres" | "bychain" | "byobject" | "bycalpha" | "bymolecule")
        ) {
            let key = key.unwrap();
            self.i += 1;
            // PyMOL's entity expansions bind more weakly than AND and OR.
            return Ok(Expr::Expand(key, Box::new(self.or(depth + 1)?)));
        }
        if matches!(
            key.as_deref(),
            Some("neighbor" | "bound_to" | "first" | "last")
        ) {
            let key = key.unwrap();
            self.i += 1;
            return Ok(Expr::Bonds(key, Box::new(self.unary(depth + 1)?), 1));
        }
        let mut left = if self.eat("(") {
            let inner = self.or(depth + 1)?;
            if !self.eat(")") {
                return Err("Expected ')' in selection".into());
            }
            inner
        } else {
            let name = self.next()?;
            let key = keyword(&name.to_ascii_lowercase()).to_owned();
            match key.as_str() {
                "model" | "object" | "chain" | "resi" | "resn" | "elem" | "name" | "id"
                | "index" | "alt" | "ss" => {
                    let value = self.next()?;
                    if matches!(key.as_str(), "resi" | "id" | "index") {
                        number_ranges(&value)?;
                    }
                    Expr::Selector(key, value)
                }
                "q" | "charge" | "partial_charge" | "occupancy" | "x" | "y" | "z" => {
                    let op = self.next()?;
                    if !matches!(op.as_str(), "<" | ">" | "<=" | ">=" | "=" | "==" | "!=") {
                        return Err(format!("Expected a comparison after {key}"));
                    }
                    let value = self.next()?;
                    finite_number(&value)?;
                    Expr::Selector(key, format!("{op} {value}"))
                }
                "all" | "none" | "enabled" | "present" | "protein" | "polymer.protein"
                | "polymer" | "nucleic" | "polymer.nucleic" | "ligand" | "organic" | "lipid"
                | "solvent" | "water" | "hydro" | "hydrogen" | "hydrogens" | "backbone"
                | "sidechain" | "hetatm" | "sele" | "guide" | "metals" | "inorganic" | "bonded" => {
                    Expr::Selector(key, String::new())
                }
                "b" | "formal_charge" | "segi" | "pepseq" | "donors" | "acceptors" | "rep"
                | "visible" | "color" | "state" | "gap" | "byring" | "bycell" | "bysegi"
                | "flag" | "masked" | "protected" | "fixed" | "restrained" | "label" | "stereo"
                | "rank" => {
                    return Err(format!(
                        "'{name}' is not supported by Molchanica's selection data"
                    ));
                }
                ")" | "," | "and" | "or" | "within" | "near_to" | "beyond" | "of" | "around"
                | "expand" | "extend" | "in" | "like" => {
                    return Err(format!("Unexpected '{name}'"));
                }
                _ if name.contains('/') && !name.starts_with(['%', '?']) => selection_macro(&name)?,
                _ => Expr::Named(name),
            }
        };
        loop {
            let Some(token) = self.tokens.get(self.i) else {
                break;
            };
            let op = keyword(&token.value.to_ascii_lowercase()).to_owned();
            if token.quoted {
                break;
            }
            if op == "extend" {
                self.i += 1;
                let count = self
                    .next()?
                    .parse::<usize>()
                    .map_err(|_| "Expected a nonnegative bond count after extend")?;
                left = Expr::Bonds(op, Box::new(left), count);
                continue;
            }
            if op == "in" || op == "like" {
                self.i += 1;
                left = Expr::Match(op, Box::new(left), Box::new(self.unary(depth + 1)?));
                continue;
            }
            if !matches!(
                op.as_str(),
                "within" | "near_to" | "beyond" | "around" | "expand"
            ) {
                break;
            }
            self.i += 1;
            let distance = finite_number(&self.next()?)?;
            if distance < 0. {
                return Err("Distance must be nonnegative".into());
            }
            if matches!(op.as_str(), "around" | "expand") {
                left = Expr::Distance(
                    op,
                    Box::new(Expr::Selector("all".into(), String::new())),
                    distance,
                    Box::new(left),
                );
            } else {
                if !self.eat("of") {
                    return Err("Expected 'of' after distance".into());
                }
                left = Expr::Distance(
                    op,
                    Box::new(left),
                    distance,
                    Box::new(self.unary(depth + 1)?),
                );
            }
        }
        Ok(left)
    }

    fn starts_expression(&self) -> bool {
        self.tokens.get(self.i).is_some_and(|t| {
            t.quoted
                || !matches!(
                    keyword(&t.value.to_ascii_lowercase()),
                    ")" | "," | "and" | "or" | "of"
                )
        })
    }
}

// Explicit aliases stay predictable as the selection language grows.
fn keyword(value: &str) -> &str {
    match value {
        "&" | "&&" | "∧" => "and",
        "|" | "||" | "∨" => "or",
        "!" => "not",
        "*" => "all",
        _ => match value.to_ascii_lowercase().as_str() {
            "m." => "model",
            "c." => "chain",
            "i." => "resi",
            "r." => "resn",
            "n." => "name",
            "e." => "elem",
            "ele" => "elem",
            "idx." => "index",
            "alt." => "alt",
            "h." => "hydrogens",
            "bb." => "backbone",
            "sc." => "sidechain",
            "pol." => "polymer",
            "org." => "organic",
            "ino." => "inorganic",
            "sol." => "solvent",
            "pr." => "present",
            "br." | "byresi" | "byresidue" => "byres",
            "bc." => "bychain",
            "bo." => "byobject",
            "bca." => "bycalpha",
            "bm." | "byfragment" | "byfrag" | "bf." => "bymolecule",
            "nbr." => "neighbor",
            "bto." => "bound_to",
            "w." => "within",
            "a." => "around",
            "x." => "expand",
            "nto." => "near_to",
            "be." => "beyond",
            "xt." => "extend",
            "pc." => "partial_charge",
            "fc." => "formal_charge",
            "s." => "segi",
            "ps." => "pepseq",
            "bs." => "bysegi",
            "don." => "donors",
            "acc." => "acceptors",
            "v." => "visible",
            "l." => "like",
            _ => value,
        },
    }
}

fn selection_macro(value: &str) -> Result<Expr, String> {
    let absolute = value.starts_with('/');
    let fields: Vec<_> = value
        .strip_prefix('/')
        .unwrap_or(value)
        .split('/')
        .collect();
    if fields.len() > 5 || (absolute && fields.is_empty()) {
        return Err("Selection macros use /object/segment/chain/resi/name".into());
    }
    let mut result = Expr::Selector("all".into(), String::new());
    let start = if absolute { 0 } else { 5 - fields.len() };
    for (field, key) in fields.iter().zip(
        ["model", "segi", "chain", "resi", "name"]
            .iter()
            .skip(start),
    ) {
        if field.is_empty() {
            continue;
        }
        if *key == "segi" {
            return Err("Segment identifiers (segi) are not stored by Molchanica; leave that macro field empty".into());
        }
        if *key == "resi" {
            number_ranges(field)?;
        }
        result = Expr::And(
            Box::new(result),
            Box::new(Expr::Selector((*key).into(), (*field).into())),
        );
    }
    Ok(result)
}

/// Accept expressions and optional `select [name,]` command syntax in the editor and CLI.
pub fn selection_input(input: &str) -> Result<(Option<&str>, &str), String> {
    let input = input.trim();
    let expression = match input.split_once(char::is_whitespace) {
        Some((command, rest)) if command.eq_ignore_ascii_case("select") => rest.trim(),
        _ => input,
    };
    let mut quote = None;
    let mut escaped = false;
    let mut boundary = true;
    for (i, c) in expression.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
        } else if boundary && matches!(c, '\'' | '"') {
            quote = Some(c);
        } else if c == ',' {
            let name = expression[..i].trim();
            if !valid_name(name) {
                return Err("Expected a selection name before ',' (for example: select active_site, resi 10-20)".into());
            }
            return Ok((Some(name), expression[i + 1..].trim()));
        }
        boundary = c.is_whitespace() || c == '(';
    }
    Ok((None, expression))
}

fn parse(input: &str) -> Result<Expr, String> {
    let (_, input) = selection_input(input)?;
    let mut p = Parser {
        tokens: tokens(input)?,
        i: 0,
    };
    let result = p.or(0)?;
    if p.i != p.tokens.len() {
        return Err(format!("Unexpected '{}'", p.tokens[p.i].value));
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

fn atom_name(atom: &Atom) -> String {
    atom.type_in_res_general
        .clone()
        .or_else(|| atom.type_in_res.as_ref().map(ToString::to_string))
        .unwrap_or_default()
}

fn atom_name_matches(atom: &Atom, value: &str) -> bool {
    text_matches(value, &atom_name(atom))
}

fn polymer_backbone(atom: &Atom, protein: bool, nucleic: bool) -> bool {
    let name = atom_name(atom).to_ascii_uppercase();
    if protein {
        return atom.is_backbone() || matches!(name.as_str(), "N" | "CA" | "C" | "O" | "OXT" | "H");
    }
    nucleic
        && matches!(
            name.replace('*', "'").as_str(),
            "P" | "OP1"
                | "OP2"
                | "OP3"
                | "O1P"
                | "O2P"
                | "O3P"
                | "C1'"
                | "C2'"
                | "O2'"
                | "C3'"
                | "O3'"
                | "C4'"
                | "O4'"
                | "C5'"
                | "O5'"
                | "H1'"
                | "H3'"
                | "H4'"
                | "H2'"
                | "H2''"
                | "H12'"
                | "H22'"
                | "H5'"
                | "H5''"
                | "H15'"
                | "H25'"
                | "HO2'"
                | "HO3'"
                | "HO5'"
        )
}

fn residue_name(residue: &Residue) -> String {
    match &residue.res_type {
        ResidueType::AminoAcid(aa) => aa.to_str(na_seq::AaIdent::ThreeLetters).to_string(),
        ResidueType::Water => "HOH".into(),
        ResidueType::Other(name) => name.clone(),
    }
}

fn is_metal(element: Element) -> bool {
    use Element::*;
    matches!(
        element,
        Iron | Copper
            | Calcium
            | Potassium
            | Aluminum
            | Lead
            | Gold
            | Silver
            | Mercury
            | Tin
            | Zinc
            | Magnesium
            | Manganese
            | Tungsten
            | Rubidium
            | Sodium
            | Indium
            | Titanium
            | Barium
            | Molybdenum
            | Cobalt
    )
}

struct Context<'a> {
    state: &'a State,
    named: &'a [NamedSelection],
    objects: Vec<(MolType, usize, MolGenericRef<'a>)>,
    atoms: Vec<(usize, usize)>,
}

impl Context<'_> {
    /// Breadth first traversal visits each bond once, even for a whole connected molecule.
    fn bond_expansion(&self, mask: &[bool], count: usize) -> Vec<bool> {
        let mut result = mask.to_vec();
        let mut offset = 0;
        for (_, _, mol) in &self.objects {
            let len = mol.common().atoms.len();
            let mut adjacency = vec![Vec::new(); len];
            for bond in &mol.common().bonds {
                if bond.atom_0 < len && bond.atom_1 < len {
                    adjacency[bond.atom_0].push(bond.atom_1);
                    adjacency[bond.atom_1].push(bond.atom_0);
                }
            }
            let mut queue: std::collections::VecDeque<_> = (0..len)
                .filter(|&a| mask[offset + a])
                .map(|a| (a, 0usize))
                .collect();
            while let Some((a, steps)) = queue.pop_front() {
                if steps >= count {
                    continue;
                }
                for &b in &adjacency[a] {
                    if !result[offset + b] {
                        result[offset + b] = true;
                        queue.push_back((b, steps + 1));
                    }
                }
            }
            offset += len;
        }
        result
    }

    fn residue(&self, o: usize, a: usize) -> Option<&Residue> {
        let mol = &self.objects[o].2;
        let r = mol.common().atoms[a].residue?;
        match mol {
            MolGenericRef::Peptide(p) => p.residues.get(r),
            MolGenericRef::NucleicAcid(p) => p.residues.get(r),
            MolGenericRef::Lipid(p) => p.residues.get(r),
            _ => None,
        }
    }

    fn chain(&self, o: usize, a: usize) -> &str {
        let mol = &self.objects[o].2;
        match mol {
            MolGenericRef::Peptide(p) => mol.common().atoms[a]
                .chain
                .and_then(|c| p.chains.get(c))
                .map_or("", |c| &c.id),
            _ => "",
        }
    }

    fn object_matches(&self, o: usize, value: &str) -> bool {
        let mol = &self.objects[o].2;
        text_matches(value, &object_key(mol))
            || text_matches(value, &mol.common().ident)
            || (!mol.common().filename.is_empty() && text_matches(value, &mol.common().filename))
            || mol
                .common()
                .path
                .as_ref()
                .and_then(|p| p.file_stem())
                .is_some_and(|stem| text_matches(value, &stem.to_string_lossy()))
            || mol
                .common()
                .name
                .as_ref()
                .is_some_and(|name| text_matches(value, name))
    }

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
                let lookup = name.trim_start_matches(['%', '?']);
                let Some(saved) = self
                    .named
                    .iter()
                    .find(|s| s.name.eq_ignore_ascii_case(lookup))
                else {
                    let objects: Vec<_> = (0..self.objects.len())
                        .map(|o| self.object_matches(o, lookup))
                        .collect();
                    if name.starts_with('?') || objects.iter().any(|v| *v) {
                        return Ok(self.atoms.iter().map(|&(o, _)| objects[o]).collect());
                    }
                    let example = match lookup.to_ascii_lowercase().as_str() {
                        "alpha_carbons" => "protein and name CA",
                        "acidic_res" => "resn ASP+GLU",
                        "active_site" => "byres (protein within 4 of ligand)",
                        _ => "resi 10-20",
                    };
                    return Err(format!(
                        "Unknown selection or object '{lookup}'. Define it first, e.g. select {lookup}, {example}"
                    ));
                };
                if visiting.iter().any(|n| n.eq_ignore_ascii_case(&saved.name)) {
                    return Err(format!("Circular selection '{}'", saved.name));
                }
                visiting.push(saved.name.clone());
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
            Expr::Expand(kind, inner) => {
                let mask = self.eval(inner, visiting, memo, depth)?;
                if kind == "bymolecule" {
                    return Ok(self.bond_expansion(&mask, usize::MAX));
                }
                let residues: HashSet<_> = self
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask[*i])
                    .filter_map(|(_, &(o, a))| {
                        let atom = &self.objects[o].2.common().atoms[a];
                        match kind.as_str() {
                            "byobject" => Some((o, None)),
                            "bychain" => Some((o, atom.chain)),
                            _ => atom.residue.map(|r| (o, Some(r))),
                        }
                    })
                    .collect();
                Ok(self
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(i, &(o, a))| {
                        let atom = &self.objects[o].2.common().atoms[a];
                        match kind.as_str() {
                            "byobject" => residues.contains(&(o, None)),
                            "bychain" => residues.contains(&(o, atom.chain)),
                            "bycalpha" => {
                                atom.residue
                                    .is_some_and(|r| residues.contains(&(o, Some(r))))
                                    && atom_name_matches(atom, "CA")
                                    && self.residue(o, a).is_some_and(|r| {
                                        matches!(r.res_type, ResidueType::AminoAcid(_))
                                    })
                            }
                            _ => {
                                mask[i]
                                    || atom
                                        .residue
                                        .is_some_and(|r| residues.contains(&(o, Some(r))))
                            }
                        }
                    })
                    .collect())
            }
            Expr::Bonds(kind, inner, count) => {
                let mask = self.eval(inner, visiting, memo, depth)?;
                if kind == "first" || kind == "last" {
                    let index = if kind == "first" {
                        mask.iter().position(|v| *v)
                    } else {
                        mask.iter().rposition(|v| *v)
                    };
                    return Ok((0..n).map(|i| Some(i) == index).collect());
                }
                if kind == "extend" {
                    return Ok(self.bond_expansion(&mask, *count));
                }
                let mut result = vec![false; n];
                let mut offset = 0;
                for (_, _, mol) in &self.objects {
                    for bond in &mol.common().bonds {
                        let (a, b) = (offset + bond.atom_0, offset + bond.atom_1);
                        if bond.atom_0 >= mol.common().atoms.len()
                            || bond.atom_1 >= mol.common().atoms.len()
                        {
                            continue;
                        }
                        result[a] |= mask[b];
                        result[b] |= mask[a];
                    }
                    offset += mol.common().atoms.len();
                }
                if kind == "neighbor" {
                    for (i, selected) in result.iter_mut().enumerate() {
                        *selected &= !mask[i];
                    }
                }
                Ok(result)
            }
            Expr::Match(kind, left, right) => {
                let left = self.eval(left, visiting, memo, depth)?;
                let right = self.eval(right, visiting, memo, depth)?;
                let identity = |o: usize, a: usize| {
                    let atom = &self.objects[o].2.common().atoms[a];
                    let residue = self.residue(o, a);
                    (
                        atom_name(atom).to_ascii_lowercase(),
                        residue.map(|r| r.serial_number),
                        if kind == "in" {
                            residue
                                .map(residue_name)
                                .unwrap_or_default()
                                .to_ascii_lowercase()
                        } else {
                            String::new()
                        },
                        if kind == "in" {
                            self.chain(o, a).to_ascii_lowercase()
                        } else {
                            String::new()
                        },
                    )
                };
                let identities: HashSet<_> = self
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| right[*i])
                    .map(|(_, &(o, a))| identity(o, a))
                    .collect();
                Ok(self
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(i, &(o, a))| left[i] && identities.contains(&identity(o, a)))
                    .collect())
            }
            Expr::Distance(kind, left, distance, right) => {
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
                    if !selected
                        || (matches!(kind.as_str(), "around" | "near_to" | "beyond") && right[i])
                    {
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
                if kind == "beyond" {
                    for (i, selected) in result.iter_mut().enumerate() {
                        *selected = left[i] && !right[i] && !*selected;
                    }
                }
                Ok(result)
            }
            Expr::Selector(key, value) => {
                let matching_objects: Vec<_> = if matches!(key.as_str(), "model" | "object") {
                    (0..self.objects.len())
                        .map(|o| self.object_matches(o, value))
                        .collect()
                } else {
                    Vec::new()
                };
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
                // Classify non-polymer groups by whether they contain carbon, not by atom element.
                let organic: HashSet<_> = if matches!(key.as_str(), "organic" | "inorganic") {
                    self.atoms
                        .iter()
                        .filter_map(|&(o, a)| {
                            let atom = &self.objects[o].2.common().atoms[a];
                            let protein = self
                                .residue(o, a)
                                .is_some_and(|r| matches!(r.res_type, ResidueType::AminoAcid(_)));
                            (atom.element == Element::Carbon
                                && !protein
                                && self.objects[o].0 != MolType::NucleicAcid)
                                .then_some((o, atom.residue))
                        })
                        .collect()
                } else {
                    HashSet::new()
                };
                let bonded: HashSet<_> = if key == "bonded" {
                    self.objects
                        .iter()
                        .enumerate()
                        .flat_map(|(o, (_, _, mol))| {
                            mol.common()
                                .bonds
                                .iter()
                                .flat_map(move |b| [(o, b.atom_0), (o, b.atom_1)])
                        })
                        .collect()
                } else {
                    HashSet::new()
                };
                Ok(self
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(i, &(o, a))| {
                        let (kind, mol_i, mol) = &self.objects[o];
                        let atom = &mol.common().atoms[a];
                        let residue = self.residue(o, a);
                        let protein = residue
                            .is_some_and(|r| matches!(r.res_type, ResidueType::AminoAcid(_)));
                        let water = atom.role == Some(AtomRole::Water)
                            || residue.is_some_and(|r| r.res_type == ResidueType::Water);
                        let numeric =
                            |number| ranges.iter().any(|&(lo, hi)| number >= lo && number <= hi);
                        match key.as_str() {
                            "all" => true,
                            "none" => false,
                            "present" => true,
                            "enabled" => mol.common().visible,
                            "model" | "object" => matching_objects[o],
                            "chain" => text_matches(value, self.chain(o, a)),
                            "resi" => residue.is_some_and(|r| numeric(r.serial_number)),
                            "resn" => residue.is_some_and(|r| {
                                text_matches(value, &residue_name(r))
                                    || (r.res_type == ResidueType::Water
                                        && text_matches(value, "WAT"))
                            }),
                            "elem" => text_matches(value, &atom.element.to_letter()),
                            "name" => atom_name_matches(atom, value),
                            "alt" => text_matches(
                                value,
                                atom.alt_conformation_id.as_deref().unwrap_or(""),
                            ),
                            "id" => numeric(atom.serial_number),
                            "index" => numeric(a as u32 + 1),
                            "protein" | "polymer.protein" => protein,
                            "polymer" => protein || *kind == MolType::NucleicAcid,
                            "nucleic" | "polymer.nucleic" => *kind == MolType::NucleicAcid,
                            "ligand" => {
                                *kind == MolType::Ligand
                                    || (*kind == MolType::Peptide
                                        && atom.hetero
                                        && !water
                                        && !protein)
                            }
                            "organic" => !water && organic.contains(&(o, atom.residue)),
                            "inorganic" => {
                                !protein
                                    && *kind != MolType::NucleicAcid
                                    && !water
                                    && !organic.contains(&(o, atom.residue))
                            }
                            "lipid" => *kind == MolType::Lipid,
                            "water" | "solvent" => water,
                            "hydro" | "hydrogen" | "hydrogens" => atom.element == Element::Hydrogen,
                            "backbone" | "sidechain" => {
                                let backbone =
                                    polymer_backbone(atom, protein, *kind == MolType::NucleicAcid);
                                if key == "backbone" {
                                    backbone
                                } else {
                                    (protein || *kind == MolType::NucleicAcid) && !backbone
                                }
                            }
                            "guide" => {
                                (protein && atom_name_matches(atom, "CA"))
                                    || (*kind == MolType::NucleicAcid
                                        && atom_name_matches(atom, "C4'+C4*"))
                            }
                            "metals" => is_metal(atom.element),
                            "bonded" => bonded.contains(&(o, a)),
                            "ss" => {
                                protein
                                    && match mol {
                                        MolGenericRef::Peptide(p) => {
                                            let ss = p.secondary_structure.iter().find(|ss| {
                                                atom.serial_number >= ss.start_sn
                                                    && atom.serial_number <= ss.end_sn
                                            });
                                            let code = match ss.map(|ss| &ss.sec_struct) {
                                                Some(SecondaryStructure::Helix) => "H",
                                                Some(SecondaryStructure::Sheet) => "S",
                                                _ => "L",
                                            };
                                            text_matches(value, code)
                                                || (ss.is_none()
                                                    && value.split('+').any(str::is_empty))
                                        }
                                        _ => false,
                                    }
                            }
                            "hetatm" => atom.hetero,
                            "sele" => selected.iter().any(|s| {
                                s.mol_type == *kind
                                    && s.mol_i == *mol_i
                                    && s.atoms.binary_search(&a).is_ok()
                            }),
                            "charge" | "partial_charge" | "q" | "occupancy" | "x" | "y" | "z" => {
                                let number = match key.as_str() {
                                    "x" => Some(self.position(i).x),
                                    "y" => Some(self.position(i).y),
                                    "z" => Some(self.position(i).z),
                                    "q" | "occupancy" => atom.occupancy.map(f64::from),
                                    _ => atom.partial_charge.map(f64::from),
                                };
                                let (op, v) = value.split_once(' ').unwrap();
                                let v: f64 = v.parse().unwrap();
                                number.is_some_and(|n| match op {
                                    "<" => n < v,
                                    ">" => n > v,
                                    "<=" => n <= v,
                                    ">=" => n >= v,
                                    "!=" => n != v,
                                    _ => n == v,
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

/// Execute a selection input, saving it when a comma-separated name is supplied.
pub fn select(state: &mut State, input: &str) -> Result<usize, String> {
    let (name, expression) = selection_input(input)?;
    if let Some(name) = name {
        return save_named(state, name, expression);
    }
    let atoms = evaluate(state, expression)?;
    let count = atoms.iter().map(|a| a.atoms.len()).sum();
    state.ui.selection = Selection::Atoms(atoms);
    Ok(count)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

pub fn save_named(state: &mut State, name: &str, expression: &str) -> Result<usize, String> {
    if !valid_name(name)
        || name.eq_ignore_ascii_case("select")
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
    if let Some(old) = named.iter_mut().find(|s| s.name.eq_ignore_ascii_case(name)) {
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
        Expr::Named(n) => n.trim_start_matches(['%', '?']).eq_ignore_ascii_case(name),
        Expr::Not(a) | Expr::Expand(_, a) | Expr::Bonds(_, a, _) => references(a, name),
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Distance(_, a, _, b) | Expr::Match(_, a, b) => {
            references(a, name) || references(b, name)
        }
        Expr::Selector(..) => false,
    }
}

pub fn delete_named(state: &mut State, name: &str) -> Result<(), String> {
    for selection in &state.to_save.named_selections {
        if !selection.name.eq_ignore_ascii_case(name)
            && references(&parse(&selection.expression)?, name)
        {
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
        .any(|s| s.name.eq_ignore_ascii_case(name))
    {
        return Err(format!("Unknown selection '{name}'"));
    }
    state
        .to_save
        .named_selections
        .retain(|s| !s.name.eq_ignore_ascii_case(name));
    state.to_save.save_flag = true;
    Ok(())
}

pub fn freeze_current(state: &State, input: &str) -> Result<String, String> {
    // Serialize the parsed expression so aliases, macros and quoted values survive saving.
    fn render(expr: &Expr, snapshot: &str) -> String {
        match expr {
            Expr::Selector(key, _) if key == "sele" => format!("({snapshot})"),
            Expr::Selector(key, value)
                if value.is_empty()
                    && !matches!(
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
                            | "ss"
                    ) =>
            {
                key.clone()
            }
            Expr::Selector(key, value) => {
                let value = if matches!(
                    key.as_str(),
                    "q" | "charge" | "partial_charge" | "occupancy" | "x" | "y" | "z"
                ) {
                    value.clone()
                } else {
                    quote(value)
                };
                format!("{key} {value}")
            }
            Expr::Named(name) => name.clone(),
            Expr::Not(a) => format!("not ({})", render(a, snapshot)),
            Expr::Expand(op, a) => format!("{op} ({})", render(a, snapshot)),
            Expr::Bonds(op, a, count) => {
                if op == "extend" {
                    format!("({}) extend {count}", render(a, snapshot))
                } else {
                    format!("{op} ({})", render(a, snapshot))
                }
            }
            Expr::And(a, b) => format!("({}) and ({})", render(a, snapshot), render(b, snapshot)),
            Expr::Or(a, b) => format!("({}) or ({})", render(a, snapshot), render(b, snapshot)),
            Expr::Match(op, a, b) => {
                format!("({}) {op} ({})", render(a, snapshot), render(b, snapshot))
            }
            Expr::Distance(op, a, distance, b) => {
                if matches!(op.as_str(), "around" | "expand") {
                    format!("({}) {op} {distance}", render(b, snapshot))
                } else {
                    format!(
                        "({}) {op} {distance} of ({})",
                        render(a, snapshot),
                        render(b, snapshot)
                    )
                }
            }
        }
    }
    Ok(render(&parse(input)?, &current_expression(state)))
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
