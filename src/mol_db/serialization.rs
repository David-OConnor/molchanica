//! Binary serialization of `MoleculeSmall`.
//!
//! Layout for Atom:
//!   [0..4]   serial_number: u32 LE
//!   [4]      element: u8  (atomic number)
//!   [5..17]  posit: 3 × f32 LE  (12 bytes; precision loss accepted for screening)
//!   [17]     type_in_res_general length: u8
//!   [18..]   type_in_res_general: UTF-8 bytes
//
//! Layout for Bond (fixed 9 bytes):
//!   [0]      bond_type: u8
//!   [1..5]   atom_0_sn: u32 LE
//!   [5..9]   atom_1_sn: u32 LE
//
//! Layout for MoleculeSmall:
//!   [0]      ident length: u8
//!   [1..]    ident: UTF-8 bytes
//!   [..]     atom_count: u16 LE
//!   [..]     for each atom: atom_len: u16 LE, then atom_len bytes
//!   [..]     bond_count: u16 LE
//!   [..]     for each bond: 9 bytes (fixed)
//
//! The `idents` and `metadata` columns each get their own blob, serialized with bincode. They're
//! stored separately from `mol_data`, so they can be loaded a-la-carte; screening doesn't need them.

use std::{collections::HashMap, io};

use bincode::config;
use bio_files::BondType;
use lin_alg::f32::Vec3 as Vec3F32;
use na_seq::Element;

use crate::molecules::{Atom, Bond, MolIdent, small::MoleculeSmall};

/// Serialize a molecule's identifiers, for the `idents` Parquet column.
pub fn idents_to_bytes(idents: &[MolIdent]) -> io::Result<Vec<u8>> {
    bincode::encode_to_vec(idents, config::standard()).map_err(io::Error::other)
}

pub fn idents_from_bytes(bytes: &[u8]) -> io::Result<Vec<MolIdent>> {
    bincode::decode_from_slice::<Vec<MolIdent>, _>(bytes, config::standard())
        .map(|(v, _)| v)
        .map_err(io::Error::other)
}

/// Serialize a molecule's metadata (i.e. `common.metadata`), for the `metadata` Parquet column.
pub fn metadata_to_bytes(metadata: &HashMap<String, String>) -> io::Result<Vec<u8>> {
    bincode::encode_to_vec(metadata, config::standard()).map_err(io::Error::other)
}

pub fn metadata_from_bytes(bytes: &[u8]) -> io::Result<HashMap<String, String>> {
    bincode::decode_from_slice::<HashMap<String, String>, _>(bytes, config::standard())
        .map(|(v, _)| v)
        .map_err(io::Error::other)
}

impl Atom {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::new();

        // serial_number: u32 LE
        res.extend_from_slice(&self.serial_number.to_le_bytes());

        // element: u8
        res.push(self.element.atomic_number());

        // posit: 3 × f32 LE  (convert from f64)
        let posit: Vec3F32 = self.posit.into();
        res.extend_from_slice(&posit.to_le_bytes());

        // type_in_res_general: length-prefixed UTF-8
        let atom_type = self.type_in_res_general.as_deref().unwrap_or("");
        let atom_type_bytes = atom_type.as_bytes();
        res.push(atom_type_bytes.len() as u8);
        res.extend_from_slice(atom_type_bytes);

        res
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut i = 0;

        let serial_number = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        i += 4;

        let element = Element::from_atomic_number(bytes[i])?;
        i += 1;

        // Written as f32, so it must be read back as f32: the f64 `Vec3::from_le_bytes` would
        // read 24 bytes, and panic.
        let posit = Vec3F32::from_le_bytes(&bytes[i..i + 12]);
        i += 12;

        let type_len = bytes[i] as usize;
        i += 1;

        let type_in_res_general = if type_len == 0 {
            None
        } else {
            Some(
                String::from_utf8(bytes[i..i + type_len].to_vec())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            )
        };

        Ok(Self {
            serial_number,
            posit: posit.into(),
            element,
            type_in_res_general,
            ..Default::default()
        })
    }
}

impl Bond {
    pub fn to_bytes(&self) -> [u8; 9] {
        let mut res = [0u8; 9];
        res[0] = match self.bond_type {
            BondType::Single => 0,
            BondType::Double => 1,
            BondType::Triple => 2,
            BondType::Aromatic => 3,
            BondType::Amide => 4,
            BondType::Dummy => 5,
            BondType::Unknown => 6,
            BondType::NotConnected => 7,
            BondType::Quadruple => 8,
            BondType::Delocalized => 9,
            BondType::PolymericLink => 10,
        };
        res[1..5].copy_from_slice(&self.atom_0_sn.to_le_bytes());
        res[5..9].copy_from_slice(&self.atom_1_sn.to_le_bytes());
        res
    }

    pub fn from_bytes(bytes: &[u8; 9]) -> io::Result<Self> {
        let bond_type = match bytes[0] {
            0 => BondType::Single,
            1 => BondType::Double,
            2 => BondType::Triple,
            3 => BondType::Aromatic,
            4 => BondType::Amide,
            5 => BondType::Dummy,
            6 => BondType::Unknown,
            7 => BondType::NotConnected,
            8 => BondType::Quadruple,
            9 => BondType::Delocalized,
            10 => BondType::PolymericLink,
            b => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown bond type byte {b}"),
                ));
            }
        };
        let atom_0_sn = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
        let atom_1_sn = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
        Ok(Bond {
            bond_type,
            atom_0_sn,
            atom_1_sn,
            atom_0: 0,
            atom_1: 0,
            is_backbone: false,
        })
    }
}

impl MoleculeSmall {
    /// Serialize to a compact binary form for embedding in the Parquet mol_data column.
    /// Only includes atoms, bonds, and ident.
    ///
    /// We only include the information critical to ser/deser for screening.
    /// Note: We potentially lose precision information due to using f32 coordinates.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::new();

        // ident
        let ident = self.common.ident.as_bytes();
        res.push(ident.len() as u8);
        res.extend_from_slice(ident);

        // atoms
        res.extend_from_slice(&(self.common.atoms.len() as u16).to_le_bytes());
        for atom in &self.common.atoms {
            let atom_bytes = atom.to_bytes();
            res.extend_from_slice(&(atom_bytes.len() as u16).to_le_bytes());
            res.extend_from_slice(&atom_bytes);
        }

        // bonds
        res.extend_from_slice(&(self.common.bonds.len() as u16).to_le_bytes());
        for bond in &self.common.bonds {
            res.extend_from_slice(&bond.to_bytes());
        }

        res
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut i = 0;

        // ident
        let ident_len = bytes[i] as usize;
        i += 1;
        let ident = String::from_utf8(bytes[i..i + ident_len].to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        i += ident_len;

        // atoms
        let atom_count = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
        i += 2;
        let mut atoms = Vec::with_capacity(atom_count);
        for _ in 0..atom_count {
            let atom_len = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
            i += 2;
            atoms.push(Atom::from_bytes(&bytes[i..i + atom_len])?);
            i += atom_len;
        }

        // bonds — resolve atom indices from serial numbers after parsing
        let bond_count = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
        i += 2;
        let mut bonds = Vec::with_capacity(bond_count);
        for _ in 0..bond_count {
            let mut bond = Bond::from_bytes(bytes[i..i + 9].try_into().unwrap())?;
            bond.atom_0 = atoms
                .iter()
                .position(|a| a.serial_number == bond.atom_0_sn)
                .unwrap_or(0);
            bond.atom_1 = atoms
                .iter()
                .position(|a| a.serial_number == bond.atom_1_sn)
                .unwrap_or(0);
            bonds.push(bond);
            i += 9;
        }

        Ok(Self::new(ident, atoms, bonds, HashMap::new(), None))
    }
}
