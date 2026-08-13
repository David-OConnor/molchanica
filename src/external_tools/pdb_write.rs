//! A minimal PDB writer, for handing structures to third-party tools.
//!
//! Molchanica keeps the raw mmCIF a peptide was opened from and saves that back out, so it has no
//! general-purpose structure writer. The tools here need one anyway: the MPNN family reads PDB and
//! nothing else. What they read of it is narrow — backbone coordinates, chain identifiers, residue
//! numbers, residue names, and (for LigandMPNN) hetero atoms as context — so this writes exactly
//! the `ATOM`/`HETATM` records those need rather than attempting a faithful round trip.
//!
//! It is deliberately not wired into `file_io::save`: this is an interchange format for
//! subprocesses, not a format Molchanica offers the user.

use std::{fmt::Write as _, io};

use bio_files::ResidueType;
use mol_defs::molecules::{Atom, peptide::MoleculePeptide};
use na_seq::AaIdent;

/// What to include in the written file.
#[derive(Clone, Debug)]
pub struct PdbWriteOptions {
    /// Chain identifiers to write. Empty writes every chain.
    pub chains: Vec<String>,
    /// Write hetero atoms as `HETATM`. LigandMPNN conditions on these; ProteinMPNN ignores them.
    pub include_hetero: bool,
    /// Write hydrogens. The MPNN models are backbone-only, and hydrogens are pure noise to them.
    pub include_hydrogen: bool,
    /// Write waters. Almost never wanted as design context.
    pub include_water: bool,
}

impl Default for PdbWriteOptions {
    fn default() -> Self {
        Self {
            chains: Vec::new(),
            include_hetero: false,
            include_hydrogen: false,
            include_water: false,
        }
    }
}

impl PdbWriteOptions {
    /// Backbone protein plus ligands and ions: what LigandMPNN wants as context.
    pub fn with_ligand_context() -> Self {
        Self {
            include_hetero: true,
            ..Self::default()
        }
    }
}

/// Render a peptide as PDB text.
///
/// Chain identifiers are truncated to the single column PDB gives them, since that is all the
/// downstream tools address chains by. Residues keep their original serial numbers so anything the
/// tool reports back can be mapped onto the structure the user is looking at.
pub fn peptide_to_pdb(mol: &MoleculePeptide, options: &PdbWriteOptions) -> io::Result<String> {
    let mut out = String::with_capacity(mol.common.atoms.len() * 81);
    let mut serial: u32 = 0;
    // PDB's serial field is 5 columns; beyond that, writers conventionally wrap rather than widen
    // the record, and every reader we feed tolerates it.
    let mut written_any = false;

    for (chain_index, chain) in mol.chains.iter().enumerate() {
        let chain_id = chain_letter(&chain.id, chain_index);
        if !options.chains.is_empty()
            && !options
                .chains
                .iter()
                .any(|wanted| chain_matches(wanted, &chain.id, chain_id))
        {
            continue;
        }

        for &residue_index in &chain.residues {
            let Some(residue) = mol.residues.get(residue_index) else {
                continue;
            };
            let is_water = residue.res_type == ResidueType::Water;
            if is_water && !options.include_water {
                continue;
            }
            let residue_name = residue_name(&residue.res_type);
            let hetero = !matches!(residue.res_type, ResidueType::AminoAcid(_));
            if hetero && !options.include_hetero {
                continue;
            }

            for &atom_index in &residue.atoms {
                let Some(atom) = mol.common.atoms.get(atom_index) else {
                    continue;
                };
                if atom.element == na_seq::Element::Hydrogen && !options.include_hydrogen {
                    continue;
                }
                serial = serial.wrapping_add(1);
                write_atom_record(
                    &mut out,
                    atom,
                    serial,
                    &residue_name,
                    chain_id,
                    residue.serial_number,
                    hetero || atom.hetero,
                )?;
                written_any = true;
            }
        }

        if written_any {
            let _ = writeln!(out, "TER");
        }
    }

    if !written_any {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no atoms matched the requested chains, so there is nothing to write",
        ));
    }

    let _ = writeln!(out, "END");
    Ok(out)
}

/// Whether a user-supplied chain selector refers to this chain.
///
/// Accepts either the molecule's own identifier (which may be longer than one character in mmCIF)
/// or the single letter this writer assigns it.
fn chain_matches(wanted: &str, chain_id: &str, letter: char) -> bool {
    wanted.eq_ignore_ascii_case(chain_id)
        || (wanted.len() == 1
            && wanted
                .chars()
                .next()
                .is_some_and(|c| c.eq_ignore_ascii_case(&letter)))
}

/// PDB has one column for the chain identifier, but mmCIF chain names can be longer. Take the
/// first character where there is one, and fall back to positional letters so two chains never
/// collapse onto the same identifier just because their names share a prefix.
pub fn chain_letter(chain_id: &str, chain_index: usize) -> char {
    match chain_id.chars().next() {
        Some(c) if c.is_ascii_alphanumeric() => c.to_ascii_uppercase(),
        _ => (b'A' + (chain_index % 26) as u8) as char,
    }
}

fn residue_name(res_type: &ResidueType) -> String {
    let name = match res_type {
        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::ThreeLetters),
        ResidueType::Water => "HOH".to_owned(),
        ResidueType::Other(name) => name.clone(),
    };
    let name = name.to_ascii_uppercase();
    // The field is three columns; a longer CCD-style code is truncated rather than pushing every
    // following field out of alignment, which would make the record unreadable.
    name.chars().take(3).collect()
}

fn write_atom_record(
    out: &mut String,
    atom: &Atom,
    serial: u32,
    residue_name: &str,
    chain_id: char,
    residue_number: u32,
    hetero: bool,
) -> io::Result<()> {
    let record = if hetero { "HETATM" } else { "ATOM  " };
    let element = atom.element.to_letter().to_ascii_uppercase();
    let name = atom
        .type_in_res
        .as_ref()
        .map(|t| t.to_string())
        .or_else(|| atom.type_in_res_general.clone())
        .unwrap_or_else(|| element.clone());

    let _ = writeln!(
        out,
        "{record}{serial:>5} {name:<4}{alt:1}{residue_name:>3} {chain_id}{residue_number:>4}{icode:1}   \
         {x:>8.3}{y:>8.3}{z:>8.3}{occupancy:>6.2}{b_factor:>6.2}          {element:>2}",
        serial = serial % 100_000,
        name = atom_name_field(&name, &element),
        alt = " ",
        icode = " ",
        x = atom.posit.x,
        y = atom.posit.y,
        z = atom.posit.z,
        occupancy = atom.occupancy.unwrap_or(1.0),
        b_factor = 0.0,
        residue_number = residue_number % 10_000,
    );
    Ok(())
}

/// The four-column atom-name field, laid out the way readers expect.
///
/// PDB's convention is that the element symbol occupies columns 13-14, so a one-character element
/// with a short name is written starting in column 14 — `_CA_` is C-alpha, whereas `CA__` is a
/// calcium ion. Getting this wrong makes every reader mis-assign elements.
fn atom_name_field(name: &str, element: &str) -> String {
    let name = name.trim();
    if name.len() >= 4 {
        return name.chars().take(4).collect();
    }
    if element.len() == 1 && name.len() <= 3 {
        format!(" {name}")
    } else {
        name.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_short_names_in_the_element_aware_column() {
        // Carbon alpha: one-character element, so the name is offset by one.
        assert_eq!(atom_name_field("CA", "C"), " CA");
        assert_eq!(atom_name_field("N", "N"), " N");
        // Calcium: two-character element, so the name starts at column 13.
        assert_eq!(atom_name_field("CA", "CA"), "CA");
        // Four characters fill the field exactly, with no offset available.
        assert_eq!(atom_name_field("HD11", "H"), "HD11");
    }

    #[test]
    fn falls_back_to_positional_chain_letters() {
        assert_eq!(chain_letter("A", 0), 'A');
        assert_eq!(chain_letter("heavy", 3), 'H');
        // Nothing usable in the name, so the position decides.
        assert_eq!(chain_letter("", 1), 'B');
        assert_eq!(chain_letter("_", 2), 'C');
    }

    #[test]
    fn matches_chains_by_name_or_letter() {
        assert!(chain_matches("A", "A", 'A'));
        assert!(chain_matches("heavy", "heavy", 'H'));
        // A single-character selector also matches the assigned letter.
        assert!(chain_matches("h", "heavy", 'H'));
        assert!(!chain_matches("L", "heavy", 'H'));
    }

    #[test]
    fn truncates_long_residue_names_to_three_columns() {
        assert_eq!(residue_name(&ResidueType::Other("NAG".into())), "NAG");
        assert_eq!(residue_name(&ResidueType::Other("ABCDE".into())), "ABC");
        assert_eq!(residue_name(&ResidueType::Water), "HOH");
    }
}
