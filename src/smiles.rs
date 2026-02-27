#![allow(unused)]

//! Convert between molecules and SMILES text

use std::{collections::HashMap, io};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::molecules::{Atom, Bond, common::MoleculeCommon};

impl MoleculeCommon {
    pub fn to_smiles(&self) -> String {
        if self.atoms.is_empty() {
            return String::new();
        }

        let n = self.atoms.len();

        // Step 1: Find ring-closure bonds (back-edges in a DFS spanning forest).
        // We track which atoms are currently on the DFS stack to detect back-edges.
        let mut dfs_visited = vec![false; n];
        let mut dfs_in_stack = vec![false; n];
        // Maps canonical (lo, hi) bond pair -> ring-closure number.
        let mut ring_bond_map: HashMap<(usize, usize), u8> = HashMap::new();
        let mut next_ring: u8 = 1;

        for start in 0..n {
            if !dfs_visited[start] {
                collect_ring_bonds(
                    start,
                    usize::MAX,
                    &self.adjacency_list,
                    &mut dfs_visited,
                    &mut dfs_in_stack,
                    &mut ring_bond_map,
                    &mut next_ring,
                );
            }
        }

        // Step 2: Build per-atom ring-closure list: atom_idx -> [(other_atom, ring_num)].
        let mut ring_closures: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
        for (&(lo, hi), &rnum) in &ring_bond_map {
            ring_closures[lo].push((hi, rnum));
            ring_closures[hi].push((lo, rnum));
        }

        // Step 3: DFS serialization.
        let mut write_visited = vec![false; n];
        let mut out = String::new();
        let mut first_component = true;

        for start in 0..n {
            if write_visited[start] {
                continue;
            }
            if !first_component {
                out.push('.');
            }
            first_component = false;
            write_atom(self, start, &mut write_visited, &ring_closures, &mut out);
        }

        out
    }

    pub fn from_smiles(data: &str) -> io::Result<Self> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut bonds: Vec<Bond> = Vec::new();
        let mut adjacency_list: Vec<Vec<usize>> = Vec::new();
        let mut atom_posits: Vec<Vec3> = Vec::new();

        let mut current: Option<usize> = None;
        let mut last_bond: Option<BondType> = None;
        // Stack saves the current atom at each branch open.
        let mut branch_stack: Vec<Option<usize>> = Vec::new();
        // ring_idx -> (atom_index, bond_type at open)
        let mut ring_map: HashMap<u32, (usize, BondType)> = HashMap::new();

        let mut chars = data.chars().peekable();
        let mut next_serial: u32 = 1;

        while let Some(&ch) = chars.peek() {
            match ch {
                // Explicit bond types
                '-' => {
                    last_bond = Some(BondType::Single);
                    chars.next();
                }
                '=' => {
                    last_bond = Some(BondType::Double);
                    chars.next();
                }
                '#' => {
                    last_bond = Some(BondType::Triple);
                    chars.next();
                }
                ':' => {
                    last_bond = Some(BondType::Aromatic);
                    chars.next();
                }
                // Stereo bonds — treat as single for connectivity purposes
                '/' | '\\' => {
                    last_bond = Some(BondType::Single);
                    chars.next();
                }

                // Branch open: push current atom so we can restore it on ')'
                '(' => {
                    branch_stack.push(current);
                    chars.next();
                }
                // Branch close: restore current atom; bond state resets
                ')' => {
                    current = branch_stack.pop().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "unmatched ')' in SMILES")
                    })?;
                    last_bond = None;
                    chars.next();
                }

                // Disconnected component separator
                '.' => {
                    current = None;
                    last_bond = None;
                    chars.next();
                }

                // Two-digit ring closure: %NN
                '%' => {
                    chars.next(); // consume '%'
                    let d1 = consume_digit(&mut chars)?;
                    let d2 = consume_digit(&mut chars)?;
                    handle_ring(
                        d1 * 10 + d2,
                        current,
                        last_bond.take().unwrap_or(BondType::Single),
                        &mut ring_map,
                        &mut bonds,
                        &mut adjacency_list,
                        &atoms,
                    )?;
                }

                // Single-digit ring closure
                '0'..='9' => {
                    let d = ch as u32 - '0' as u32;
                    chars.next();
                    handle_ring(
                        d,
                        current,
                        last_bond.take().unwrap_or(BondType::Single),
                        &mut ring_map,
                        &mut bonds,
                        &mut adjacency_list,
                        &atoms,
                    )?;
                }

                // Bracket atom: [isotope?symbol@?H?charge?:map?]
                '[' => {
                    let (element, _aromatic) = parse_bracket_atom(&mut chars)?;
                    let idx = push_atom(
                        next_serial,
                        element,
                        current,
                        last_bond.take(),
                        &mut atoms,
                        &mut bonds,
                        &mut adjacency_list,
                        &mut atom_posits,
                    );
                    next_serial += 1;
                    current = Some(idx);
                }

                // Organic-subset atom (bare symbol, no brackets)
                _ => match parse_organic_atom(&mut chars)? {
                    Some((element, _aromatic)) => {
                        let idx = push_atom(
                            next_serial,
                            element,
                            current,
                            last_bond.take(),
                            &mut atoms,
                            &mut bonds,
                            &mut adjacency_list,
                            &mut atom_posits,
                        );
                        next_serial += 1;
                        current = Some(idx);
                    }
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unrecognized SMILES character: '{ch}'"),
                        ));
                    }
                },
            }
        }

        if !ring_map.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unclosed ring closure index in SMILES",
            ));
        }

        Ok(MoleculeCommon {
            ident: data.trim().to_string(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            filename: String::from("From SMILES"),
            next_atom_sn: next_serial,
            ..Default::default()
        })
    }

    fn find_bond(&self, a: usize, b: usize) -> Option<&Bond> {
        self.bonds
            .iter()
            .find(|bb| (bb.atom_0 == a && bb.atom_1 == b) || (bb.atom_0 == b && bb.atom_1 == a))
    }
}

// ── to_smiles helpers ─────────────────────────────────────────────────────────

/// DFS that records all back-edges (ring-closure bonds) into `ring_bond_map`.
/// `parent` is the atom we arrived from (usize::MAX for the root).
fn collect_ring_bonds(
    u: usize,
    parent: usize,
    adj: &[Vec<usize>],
    visited: &mut Vec<bool>,
    in_stack: &mut Vec<bool>,
    ring_bond_map: &mut HashMap<(usize, usize), u8>,
    next_ring: &mut u8,
) {
    visited[u] = true;
    in_stack[u] = true;

    for &v in &adj[u] {
        // Skip the tree-edge back to our parent.
        if v == parent {
            continue;
        }
        if in_stack[v] {
            // Back-edge: ring closure bond.
            let key = (u.min(v), u.max(v));
            if !ring_bond_map.contains_key(&key) {
                ring_bond_map.insert(key, *next_ring);
                *next_ring += 1;
            }
        } else if !visited[v] {
            collect_ring_bonds(v, u, adj, visited, in_stack, ring_bond_map, next_ring);
        }
    }

    in_stack[u] = false;
}

/// Recursively write atom `u` and its DFS subtree into `out`.
/// Parent is already marked visited, so unvisited neighbors = tree-edge children.
fn write_atom(
    mol: &MoleculeCommon,
    u: usize,
    visited: &mut Vec<bool>,
    ring_closures: &[Vec<(usize, u8)>],
    out: &mut String,
) {
    visited[u] = true;

    // Write the atom symbol.
    out.push_str(&smiles_symbol(mol.atoms[u].element));

    // Write ring-closure digits attached to this atom.
    for &(other, rnum) in &ring_closures[u] {
        if let Some(b) = mol.find_bond(u, other) {
            push_bond_char(b.bond_type, out);
        }
        push_ring_num(rnum, out);
    }

    // Collect tree-edge children (unvisited neighbors).
    let children: Vec<usize> = mol.adjacency_list[u]
        .iter()
        .copied()
        .filter(|&v| !visited[v])
        .collect();

    let last_i = children.len().wrapping_sub(1);
    for (i, v) in children.into_iter().enumerate() {
        let bt = mol.find_bond(u, v).map(|b| b.bond_type);

        if i == last_i {
            // Last child continues inline (no parentheses).
            if let Some(t) = bt {
                push_bond_char(t, out);
            }
            write_atom(mol, v, visited, ring_closures, out);
        } else {
            // Earlier children are branches.
            out.push('(');
            if let Some(t) = bt {
                push_bond_char(t, out);
            }
            write_atom(mol, v, visited, ring_closures, out);
            out.push(')');
        }
    }
}

/// Append the explicit bond character for non-single bonds (single is implicit in SMILES).
fn push_bond_char(bt: BondType, out: &mut String) {
    match bt {
        BondType::Double => out.push('='),
        BondType::Triple => out.push('#'),
        BondType::Aromatic => out.push(':'),
        _ => {}
    }
}

/// Append the ring-closure number (single digit, or %NN for ≥ 10).
fn push_ring_num(rnum: u8, out: &mut String) {
    if rnum < 10 {
        out.push((b'0' + rnum) as char);
    } else {
        out.push('%');
        out.push((b'0' + rnum / 10) as char);
        out.push((b'0' + rnum % 10) as char);
    }
}

/// Returns the SMILES atom token for `el`.
/// Organic-subset elements are written bare; everything else gets `[symbol]`.
fn smiles_symbol(el: Element) -> String {
    match el {
        Element::Boron => "B".into(),
        Element::Carbon => "C".into(),
        Element::Nitrogen => "N".into(),
        Element::Oxygen => "O".into(),
        Element::Phosphorus => "P".into(),
        Element::Sulfur => "S".into(),
        Element::Fluorine => "F".into(),
        Element::Chlorine => "Cl".into(),
        Element::Bromine => "Br".into(),
        Element::Iodine => "I".into(),
        Element::Hydrogen => "[H]".into(),
        other => format!("[{}]", other.to_letter()),
    }
}

// ── from_smiles helpers ───────────────────────────────────────────────────────

/// Add a new atom to the vectors, bond it to `prev` if present, return its index.
fn push_atom(
    serial: u32,
    element: Element,
    prev: Option<usize>,
    bond_type: Option<BondType>,
    atoms: &mut Vec<Atom>,
    bonds: &mut Vec<Bond>,
    adj: &mut Vec<Vec<usize>>,
    atom_posits: &mut Vec<Vec3>,
) -> usize {
    let idx = atoms.len();
    atoms.push(Atom {
        serial_number: serial,
        posit: Vec3::new(0.0, 0.0, 0.0),
        element,
        ..Default::default()
    });
    adj.push(Vec::new());
    atom_posits.push(Vec3::new(0.0, 0.0, 0.0));

    if let Some(p) = prev {
        let bt = bond_type.unwrap_or(BondType::Single);
        add_bond(p, idx, bt, bonds, adj, atoms);
    }

    idx
}

/// Open or close a ring-closure bond.
fn handle_ring(
    ring_idx: u32,
    current: Option<usize>,
    bt: BondType,
    ring_map: &mut HashMap<u32, (usize, BondType)>,
    bonds: &mut Vec<Bond>,
    adj: &mut Vec<Vec<usize>>,
    atoms: &[Atom],
) -> io::Result<()> {
    let cur = current.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "ring closure digit without a current atom",
        )
    })?;

    match ring_map.remove(&ring_idx) {
        Some((other, bt_open)) => {
            // Closing: create the ring bond using the bond type recorded at open.
            add_bond(cur, other, bt_open, bonds, adj, atoms);
        }
        None => {
            // Opening: record this atom and bond type for later closure.
            ring_map.insert(ring_idx, (cur, bt));
        }
    }

    Ok(())
}

/// Parse a bracket atom `[isotope? symbol chirality? Hcount? charge? :map?]`.
/// The leading `[` must still be in the iterator; returns `(element, is_aromatic)`.
/// Isotope, chirality, H-count, charge, and atom-map are consumed and discarded.
fn parse_bracket_atom(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> io::Result<(Element, bool)> {
    chars.next(); // consume '['

    // Optional isotope (one or more digits before the element symbol)
    while chars.peek().map_or(false, |c| c.is_ascii_digit()) {
        chars.next();
    }

    // Element symbol: first letter (case determines aromaticity)
    let first = chars.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected end of input inside bracket atom",
        )
    })?;
    let aromatic = first.is_ascii_lowercase();
    let mut sym = String::from(first.to_ascii_uppercase());

    // Optional second letter (always lowercase, e.g. 'l' in Cl, 'r' in Br, 'g' in Hg)
    if chars.peek().map_or(false, |c| c.is_ascii_lowercase()) {
        sym.push(chars.next().unwrap());
    }

    // Optional chirality: @ or @@
    while chars.peek().copied() == Some('@') {
        chars.next();
    }

    // Optional H-count: H or Hn
    if chars.peek().copied() == Some('H') {
        chars.next();
        while chars.peek().map_or(false, |c| c.is_ascii_digit()) {
            chars.next();
        }
    }

    // Optional charge: +, -, ++, --, +n, -n
    if chars.peek().map_or(false, |&c| c == '+' || c == '-') {
        chars.next();
        while chars
            .peek()
            .map_or(false, |&c| c.is_ascii_digit() || c == '+' || c == '-')
        {
            chars.next();
        }
    }

    // Optional atom-map: :n
    if chars.peek().copied() == Some(':') {
        chars.next();
        while chars.peek().map_or(false, |c| c.is_ascii_digit()) {
            chars.next();
        }
    }

    // Closing ']'
    match chars.next() {
        Some(']') => {}
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected ']' to close bracket atom, found {:?}", other),
            ));
        }
    }

    let element = Element::from_letter(&sym)?;
    Ok((element, aromatic))
}

/// Parse an organic-subset atom (no brackets). Advances the iterator past the token.
/// Returns `None` for unrecognized characters (caller decides whether to error).
fn parse_organic_atom(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> io::Result<Option<(Element, bool)>> {
    let ch = match chars.peek().copied() {
        Some(c) => c,
        None => return Ok(None),
    };

    match ch {
        'C' => {
            chars.next();
            if chars.peek().copied() == Some('l') {
                chars.next();
                Ok(Some((Element::Chlorine, false)))
            } else {
                Ok(Some((Element::Carbon, false)))
            }
        }
        'B' => {
            chars.next();
            if chars.peek().copied() == Some('r') {
                chars.next();
                Ok(Some((Element::Bromine, false)))
            } else {
                Ok(Some((Element::Boron, false)))
            }
        }
        'N' => {
            chars.next();
            Ok(Some((Element::Nitrogen, false)))
        }
        'O' => {
            chars.next();
            Ok(Some((Element::Oxygen, false)))
        }
        'S' => {
            chars.next();
            Ok(Some((Element::Sulfur, false)))
        }
        'P' => {
            chars.next();
            Ok(Some((Element::Phosphorus, false)))
        }
        'F' => {
            chars.next();
            Ok(Some((Element::Fluorine, false)))
        }
        'I' => {
            chars.next();
            Ok(Some((Element::Iodine, false)))
        }
        'H' => {
            chars.next();
            Ok(Some((Element::Hydrogen, false)))
        }
        // Aromatic atoms (lowercase organic subset)
        'c' => {
            chars.next();
            Ok(Some((Element::Carbon, true)))
        }
        'n' => {
            chars.next();
            Ok(Some((Element::Nitrogen, true)))
        }
        'o' => {
            chars.next();
            Ok(Some((Element::Oxygen, true)))
        }
        's' => {
            chars.next();
            Ok(Some((Element::Sulfur, true)))
        }
        'p' => {
            chars.next();
            Ok(Some((Element::Phosphorus, true)))
        }
        _ => Ok(None),
    }
}

/// Consume a single ASCII digit from the iterator, returning its numeric value.
fn consume_digit(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> io::Result<u32> {
    match chars.next() {
        Some(c) if c.is_ascii_digit() => Ok(c as u32 - '0' as u32),
        Some(c) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected digit after '%', found '{c}'"),
        )),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected digit after '%', found end of input",
        )),
    }
}

/// Add a bond between atoms `a` and `b`, updating both `bonds` and `adj`.
/// Bond is stored with the lower index as atom_0.
fn add_bond(
    a: usize,
    b: usize,
    bond_type: BondType,
    bonds: &mut Vec<Bond>,
    adj: &mut Vec<Vec<usize>>,
    atoms: &[Atom],
) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    bonds.push(Bond {
        bond_type,
        atom_0_sn: atoms[lo].serial_number,
        atom_1_sn: atoms[hi].serial_number,
        atom_0: lo,
        atom_1: hi,
        is_backbone: false,
    });
    adj[a].push(b);
    adj[b].push(a);
}

/// Attempt to determine if a given string is likely to be SMILES.
///
/// Distinguishes from the other identifier types used in the search field:
///   • PDB ID       – exactly 4 alphanumeric chars, typically leading digit ("1ABC")
///   • DrugBank ID  – "DB" + 5 digits ("DB01234")
///   • PubChem CID  – all digits (already caught upstream by `parse::<u32>()`)
///   • Common name  – prose word(s) containing letters outside the SMILES alphabet
///
/// We assume the input is trimmed and lowercase.
pub fn is_smiles(s: &str) -> bool {
    // Minimum meaningful length.  Also excludes all 4-char PDB IDs.
    if s.len() < 5 {
        return false;
    }

    // Common names and multi-word queries contain spaces; SMILES never do.
    if s.contains(' ') {
        return false;
    }

    // SMILES never begins with a digit — ring-closure numbers always follow an atom.
    // This also catches CAS numbers ("64-17-5") and digit-leading PDB IDs ("1abc").
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }

    // Strong structural SMILES indicators: any of these characters appear only in SMILES
    // among the identifier types we care about.
    if s.bytes()
        .any(|b| matches!(b, b'=' | b'#' | b'(' | b')' | b'[' | b']' | b'.'))
    {
        return true;
    }

    // Fallback: every character must belong to the SMILES organic-subset alphabet.
    // Most common-name letters (a, d, e, g, h, j, k, m, q, r, t, u, v, w, x, y, z)
    // are absent from this set, so plain drug names like "aspirin" or "morphine" fail here.
    //
    // Alphabet: element letters C N O S P F B I (+ second chars l/r for Cl/Br),
    //           ring-closure digits 0–9, and the remaining bond/stereo tokens - / \ : @
    s.bytes().all(|b| {
        matches!(
            b,
            b'C' | b'N' | b'O' | b'S' | b'P' | b'F' | b'B' | b'I' |
            b'c' | b'n' | b'o' | b's' | b'p' | b'f' | b'b' | b'i' |
            b'l' | b'r' |          // second chars of Cl / Br
            b'0'..=b'9' | b'-' | b'/' | b'\\' | b':' | b'@'
        )
    })
}
