//! Convert between molecules and SMILES text
//! todo: Consider  moving this to bio_files. Although geometry considerations
//! may be a better suit for here.

use std::{
    collections::{HashMap, VecDeque},
    io,
};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::molecules::{
    Atom, Bond,
    common::{BondGeom, MoleculeCommon, find_appended_posit},
};

impl MoleculeCommon {
    pub fn to_smiles(&self) -> String {
        if self.atoms.is_empty() {
            return String::new();
        }

        let n = self.atoms.len();

        // Hydrogen atoms are implicit in SMILES and must be excluded from both DFS
        // passes.  Pre-marking them as "visited" keeps them out of start-atom selection
        // and out of the children lists, which would otherwise produce empty "()" branches.
        let h_mask: Vec<bool> = self
            .atoms
            .iter()
            .map(|a| a.element == Element::Hydrogen)
            .collect();

        // Step 1: Ring-closure detection.
        // IMPORTANT: must use the same per-component starting atom as the write pass
        // below, otherwise the two DFS traversals produce different spanning trees and
        // ring-closure digits end up misplaced.
        let mut detect_visited = h_mask.clone();
        let mut detect_in_stack = vec![false; n];
        let mut ring_bond_map: HashMap<(usize, usize), u8> = HashMap::new();
        let mut next_ring: u8 = 1;

        loop {
            match pick_start(n, &self.adjacency_list, &self.atoms, &detect_visited) {
                None => break,
                Some(start) => collect_ring_bonds(
                    start,
                    usize::MAX,
                    &self.adjacency_list,
                    &mut detect_visited,
                    &mut detect_in_stack,
                    &mut ring_bond_map,
                    &mut next_ring,
                ),
            }
        }

        // Step 2: Build per-atom ring-closure list: atom_idx -> [(other_atom, ring_num)].
        let mut ring_closures: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
        for (&(lo, hi), &rnum) in &ring_bond_map {
            ring_closures[lo].push((hi, rnum));
            ring_closures[hi].push((lo, rnum));
        }

        // Step 3: DFS serialization — same start-atom logic as step 1.
        let mut write_visited = h_mask;
        let mut out = String::new();
        let mut first_component = true;

        loop {
            match pick_start(n, &self.adjacency_list, &self.atoms, &write_visited) {
                None => break,
                Some(start) => {
                    if !first_component {
                        out.push('.');
                    }
                    first_component = false;
                    write_atom(self, start, &mut write_visited, &ring_closures, &mut out);
                }
            }
        }

        out
    }

    pub fn from_smiles(data: &str) -> io::Result<Self> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut bonds: Vec<Bond> = Vec::new();
        let mut adjacency_list: Vec<Vec<usize>> = Vec::new();
        let mut atom_posits: Vec<Vec3> = Vec::new();

        let mut current: Option<usize> = None;
        // Whether the current atom was written as aromatic (lowercase in SMILES).
        // Two consecutive aromatic atoms share an implicit aromatic bond; a mixed or
        // non-aromatic pair gets an implicit single bond.
        let mut current_aromatic: bool = false;
        let mut last_bond: Option<BondType> = None;
        // Stack saves (current atom index, aromaticity) at each branch open.
        let mut branch_stack: Vec<(Option<usize>, bool)> = Vec::new();
        // ring_idx -> (atom_index, explicit bond type at open (None = implicit), aromatic at open)
        let mut ring_map: HashMap<u32, (usize, Option<BondType>, bool)> = HashMap::new();

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

                // Branch open: push (current atom, aromaticity) so we can restore on ')'
                '(' => {
                    branch_stack.push((current, current_aromatic));
                    chars.next();
                }
                // Branch close: restore current atom and its aromaticity; bond state resets
                ')' => {
                    let (prev, prev_ar) = branch_stack.pop().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "unmatched ')' in SMILES")
                    })?;
                    current = prev;
                    current_aromatic = prev_ar;
                    last_bond = None;
                    chars.next();
                }

                // Disconnected component separator
                '.' => {
                    current = None;
                    current_aromatic = false;
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
                        current_aromatic,
                        last_bond.take(), // None = implicit; resolved inside handle_ring
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
                        current_aromatic,
                        last_bond.take(), // None = implicit; resolved inside handle_ring
                        &mut ring_map,
                        &mut bonds,
                        &mut adjacency_list,
                        &atoms,
                    )?;
                }

                // Bracket atom: [isotope?symbol@?H?charge?:map?]
                // Bracket atoms are always treated as non-aromatic for bond-type inference
                // (the bracket form explicitly states valence/Hs, so the bond type is
                // determined by an explicit bond char or defaults to single).
                '[' => {
                    let (element, is_aromatic) = parse_bracket_atom(&mut chars)?;
                    let bt = last_bond
                        .take()
                        .unwrap_or_else(|| implicit_bt(current_aromatic, is_aromatic, current));
                    let idx = push_atom(
                        next_serial,
                        element,
                        current,
                        Some(bt),
                        &mut atoms,
                        &mut bonds,
                        &mut adjacency_list,
                        &mut atom_posits,
                    );
                    next_serial += 1;
                    current = Some(idx);
                    current_aromatic = is_aromatic;
                }

                // Organic-subset atom (bare symbol, no brackets)
                _ => match parse_organic_atom(&mut chars)? {
                    Some((element, is_aromatic)) => {
                        let bt = last_bond
                            .take()
                            .unwrap_or_else(|| implicit_bt(current_aromatic, is_aromatic, current));
                        let idx = push_atom(
                            next_serial,
                            element,
                            current,
                            Some(bt),
                            &mut atoms,
                            &mut bonds,
                            &mut adjacency_list,
                            &mut atom_posits,
                        );
                        next_serial += 1;
                        current = Some(idx);
                        current_aromatic = is_aromatic;
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

        // Assign rough 3D coordinates via BFS geometry placement.
        assign_rough_coords(&mut atoms, &adjacency_list, &bonds);
        // Rebuild atom_posits from the freshly computed atom positions.
        let atom_posits: Vec<Vec3> = atoms.iter().map(|a| a.posit).collect();

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

/// Choose the DFS start atom for one connected component.
///
/// Convention (approximates canonical SMILES starting points):
///   • Prefer a **terminal carbon** (exactly one non-H neighbour, element = C).
///   • Fall back to any terminal non-H atom (degree 1).
///   • Fall back to any non-H atom (e.g., isolated heavy atom).
///   • Among equal-priority candidates pick the one with the **highest atom index**,
///     which for the typical SDF/Mol2 numbering order starts the traversal from the
///     far end of the main chain — matching the PubChem-style direction.
///
/// Returns `None` when every atom is already visited (all components done).
fn pick_start(n: usize, adj: &[Vec<usize>], atoms: &[Atom], visited: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_score: i32 = -1;

    for i in 0..n {
        if visited[i] {
            continue;
        }

        // Non-H degree (H atoms are pre-marked visited, so this is the heavy-atom degree).
        let heavy_deg = adj[i].iter().filter(|&&v| !visited[v]).count();

        let score: i32 = match (heavy_deg, atoms[i].element == Element::Carbon) {
            (1, true) => 3,  // terminal carbon  — best
            (1, false) => 2, // terminal heteroatom
            (0, _) => 1,     // isolated heavy atom
            _ => 0,          // internal atom    — last resort
        };

        if score > best_score || (score == best_score && i > best.unwrap_or(0)) {
            best_score = score;
            best = Some(i);
        }
    }

    best
}

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

    let el = mol.atoms[u].element;
    if el == Element::Hydrogen {
        return;
    }

    // Determine whether this atom is part of an aromatic system.
    // We use this to decide case (lowercase = aromatic) and to suppress the `:` bond
    // character between two aromatic atoms (it is implicit for lowercase pairs).
    let u_aromatic = atom_is_aromatic(u, &mol.bonds);

    // Write the atom symbol (lowercase if aromatic; bracket form for [nH] etc.).
    out.push_str(&smiles_symbol_for(el, u_aromatic, u, mol));

    // Write ring-closure digits attached to this atom.
    for &(other, rnum) in &ring_closures[u] {
        if let Some(b) = mol.find_bond(u, other) {
            let other_aromatic = atom_is_aromatic(other, &mol.bonds);
            push_bond_char_ctx(b.bond_type, u_aromatic, other_aromatic, out);
        }
        push_ring_num(rnum, out);
    }

    // Collect tree-edge children: unvisited neighbors that are NOT ring-closure partners.
    //
    // Ring-closure partners must be excluded here.  The opener atom (u) writes the ring
    // digit above, and the closer atom is reached naturally through the DFS tree (the
    // other path around the ring), where it writes the matching digit.  If we also
    // traversed the partner as a direct child we would visit it twice and write the
    // ring number twice, corrupting the SMILES.
    let children: Vec<usize> = mol.adjacency_list[u]
        .iter()
        .copied()
        .filter(|&v| !visited[v] && ring_closures[u].iter().all(|&(rc, _)| rc != v))
        .collect();

    let last_i = children.len().wrapping_sub(1);
    for (i, v) in children.into_iter().enumerate() {
        let bt = mol.find_bond(u, v).map(|b| b.bond_type);
        let v_aromatic = atom_is_aromatic(v, &mol.bonds);

        if i == last_i {
            // Last child continues inline (no parentheses).
            if let Some(t) = bt {
                push_bond_char_ctx(t, u_aromatic, v_aromatic, out);
            }
            write_atom(mol, v, visited, ring_closures, out);
        } else {
            // Earlier children are branches.
            out.push('(');
            if let Some(t) = bt {
                push_bond_char_ctx(t, u_aromatic, v_aromatic, out);
            }
            write_atom(mol, v, visited, ring_closures, out);
            out.push(')');
        }
    }
}

/// Append a bond character, taking both atoms' aromaticity into account.
///
/// Rule: an aromatic bond between two aromatic (lowercase) atoms is **implicit** — no
/// character needed.  All other non-single bonds are written explicitly.  Single bonds
/// are always implicit regardless of context.
fn push_bond_char_ctx(bt: BondType, u_aromatic: bool, v_aromatic: bool, out: &mut String) {
    match bt {
        BondType::Aromatic if u_aromatic && v_aromatic => {
            // Implicit for lowercase–lowercase pairs; writing nothing is correct.
        }
        BondType::Double => out.push('='),
        BondType::Triple => out.push('#'),
        BondType::Aromatic => out.push(':'), // unusual: aromatic bond to/from aliphatic atom
        _ => {}                              // Single — always implicit
    }
}

/// Returns `true` if atom `u` participates in at least one aromatic bond.
fn atom_is_aromatic(u: usize, bonds: &[Bond]) -> bool {
    bonds
        .iter()
        .any(|b| (b.atom_0 == u || b.atom_1 == u) && b.bond_type == BondType::Aromatic)
}

/// Atom symbol for use in SMILES output.
///
/// Aromatic atoms are written lowercase.  Heteroatoms that carry explicit H (like the
/// pyrrole-type [nH]) are written in bracket form so the H count is unambiguous.
fn smiles_symbol_for(el: Element, aromatic: bool, u: usize, mol: &MoleculeCommon) -> String {
    if !aromatic {
        return smiles_symbol(el);
    }
    // Count H atoms directly bonded to u.  These are in the atom list but are
    // pre-marked as visited in the DFS and never appear as children, so we must
    // encode them in the bracket atom token instead.
    let h = mol.adjacency_list[u]
        .iter()
        .filter(|&&v| mol.atoms[v].element == Element::Hydrogen)
        .count();
    match el {
        Element::Carbon => "c".into(),
        Element::Nitrogen => match h {
            0 => "n".into(),
            1 => "[nH]".into(),
            n => format!("[nH{n}]"),
        },
        Element::Oxygen => match h {
            0 => "o".into(),
            _ => "[oH]".into(),
        },
        Element::Sulfur => match h {
            0 => "s".into(),
            _ => "[sH]".into(),
        },
        Element::Phosphorus => "p".into(),
        Element::Boron => "b".into(),
        other => format!("[{}]", other.to_letter()), // non-standard aromatic element
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

/// Determine the implicit bond type between two adjacent atoms in SMILES.
/// Per the SMILES spec: if both atoms are aromatic (written lowercase) the
/// implicit bond is aromatic; otherwise it is single.
/// `has_prev` is false for the very first atom in a component (no bond to create).
#[inline]
fn implicit_bt(prev_aromatic: bool, new_aromatic: bool, prev: Option<usize>) -> BondType {
    if prev.is_some() && prev_aromatic && new_aromatic {
        BondType::Aromatic
    } else {
        BondType::Single
    }
}

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
///
/// `explicit_bt` is `Some` only if an explicit bond character (`=`, `#`, `:`, `-`) appeared
/// immediately before the ring-closure digit; otherwise `None` (implicit bond).
/// The bond type for an implicit ring closure is:
///   - aromatic  if both the opening and closing atoms are aromatic
///   - single    otherwise
fn handle_ring(
    ring_idx: u32,
    current: Option<usize>,
    current_aromatic: bool,
    explicit_bt: Option<BondType>,
    ring_map: &mut HashMap<u32, (usize, Option<BondType>, bool)>,
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
        Some((other, bt_open, open_aromatic)) => {
            // Closing: determine bond type.
            // An explicit bond at either end takes priority; otherwise use aromaticity.
            let bond_type = explicit_bt.or(bt_open).unwrap_or_else(|| {
                if open_aromatic && current_aromatic {
                    BondType::Aromatic
                } else {
                    BondType::Single
                }
            });
            add_bond(cur, other, bond_type, bonds, adj, atoms);
        }
        None => {
            // Opening: record atom index, any explicit bond type, and aromaticity.
            ring_map.insert(ring_idx, (cur, explicit_bt, current_aromatic));
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

// ── coordinate generation ─────────────────────────────────────────────────────

/// Assign rough 3D coordinates to all atoms after SMILES parsing.
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
fn assign_rough_coords(atoms: &mut Vec<Atom>, adj: &[Vec<usize>], bonds: &[Bond]) {
    let n = atoms.len();
    if n == 0 {
        return;
    }

    let mut positioned = vec![false; n];

    // Place first atom at the origin.
    atoms[0].posit = Vec3::new(0., 0., 0.);
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
                    atoms[start].posit = Vec3::new(component_x, 0., 0.);
                    positioned[start] = true;
                    queue.push_back(start);
                }
            }
        }

        let u = match queue.pop_front() {
            Some(u) => u,
            None => continue,
        };

        let geom = geom_for_atom(u, bonds);
        let posit_u = atoms[u].posit; // Copy — Vec3 is Copy

        // Clone the neighbour list to avoid borrow conflicts while mutating atoms.
        let neighbours: Vec<usize> = adj[u].to_vec();

        for v in neighbours {
            if positioned[v] {
                // Already placed (either placed earlier in BFS, or a ring-closure
                // partner placed via the other path around the ring).  Skip.
                continue;
            }

            // Collect already-positioned neighbours of u (excluding v itself) to
            // give find_appended_posit its angular context.
            let adj_placed: Vec<usize> = adj[u]
                .iter()
                .copied()
                .filter(|&w| w != v && positioned[w])
                .collect();

            let bt = bond_type_between(u, v, bonds);
            let bond_len = estimate_bond_length(atoms[u].element, atoms[v].element, bt);

            let p = find_appended_posit(
                posit_u,
                atoms,
                &adj_placed,
                Some(bond_len),
                atoms[v].element,
                geom,
            )
            .unwrap_or_else(|| posit_u + Vec3::new(bond_len, 0., 0.));

            atoms[v].posit = p;
            positioned[v] = true;
            queue.push_back(v);
        }
    }
}

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
