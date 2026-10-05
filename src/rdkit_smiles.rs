//! SMILES conversion through an already-installed RDKit Python environment.

use std::{collections::HashMap, io, path::Path, time::Duration};

use bio_files::BondType;
use bio_tools::run::{self, CaptureLimits, CommandSpec};
use lin_alg::f64::Vec3;
use mol_defs::molecules::{Atom, Bond, common::MoleculeCommon};
use na_seq::Element;
use serde::{Deserialize, Serialize};

const PYTHON: &str = include_str!("rdkit_smiles.py");

#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Request<'a> {
    Parse {
        smiles: &'a str,
    },
    Serialize {
        atoms: Vec<u8>,
        bonds: Vec<SerializedBond>,
    },
}

#[derive(Serialize)]
struct SerializedBond {
    atom_0: usize,
    atom_1: usize,
    bond_type: &'static str,
}

#[derive(Deserialize)]
struct Response {
    error: Option<String>,
    smiles: Option<String>,
    atoms: Option<Vec<ParsedAtom>>,
    bonds: Option<Vec<ParsedBond>>,
}

#[derive(Deserialize)]
struct ParsedAtom {
    atomic_number: u8,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Deserialize)]
struct ParsedBond {
    atom_0: usize,
    atom_1: usize,
    order: f64,
}

/// Parse a SMILES string with RDKit and return its molecular graph and 2D coordinates.
pub fn parse(python: &Path, smiles: &str) -> io::Result<MoleculeCommon> {
    if smiles.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SMILES cannot be empty",
        ));
    }

    let response = invoke(python, &Request::Parse { smiles })?;
    if let Some(error) = response.error {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, error));
    }

    let parsed_atoms = response
        .atoms
        .ok_or_else(|| io::Error::other("RDKit returned no atoms"))?;
    let parsed_bonds = response
        .bonds
        .ok_or_else(|| io::Error::other("RDKit returned no bonds"))?;

    let atom_count = parsed_atoms.len();
    let atoms = parsed_atoms
        .into_iter()
        .enumerate()
        .map(|(index, atom)| {
            if !atom.x.is_finite() || !atom.y.is_finite() || !atom.z.is_finite() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "RDKit returned a non-finite atom coordinate",
                ));
            }

            let serial_number = u32::try_from(index + 1).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "RDKit returned too many atoms")
            })?;

            Ok(Atom {
                serial_number,
                posit: Vec3::new(atom.x, atom.y, atom.z),
                element: Element::from_atomic_number(atom.atomic_number)?,
                ..Default::default()
            })
        })
        .collect::<io::Result<Vec<_>>>()?;

    let bonds = parsed_bonds
        .into_iter()
        .map(|bond| {
            if bond.atom_0 >= atom_count || bond.atom_1 >= atom_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "RDKit returned a bond with an invalid atom index",
                ));
            }

            Ok(Bond {
                bond_type: bond_type_from_order(bond.order),
                atom_0_sn: atoms[bond.atom_0].serial_number,
                atom_1_sn: atoms[bond.atom_1].serial_number,
                atom_0: bond.atom_0,
                atom_1: bond.atom_1,
                is_backbone: false,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;

    let mut molecule =
        MoleculeCommon::new(smiles.trim().to_owned(), atoms, bonds, HashMap::new(), None);
    molecule.filename = "From SMILES".to_owned();
    molecule.is_2d = true;
    molecule.update_next_sn();
    Ok(molecule)
}

/// Serialize a molecular graph to canonical SMILES with RDKit.
pub fn serialize(python: &Path, molecule: &MoleculeCommon) -> io::Result<String> {
    let atoms = molecule
        .atoms
        .iter()
        .map(|atom| atom.element.atomic_number())
        .collect();
    let bonds = molecule
        .bonds
        .iter()
        .filter_map(|bond| {
            if bond.bond_type == BondType::NotConnected {
                return None;
            }

            Some(SerializedBond {
                atom_0: bond.atom_0,
                atom_1: bond.atom_1,
                bond_type: bond_type_for_rdkit(bond.bond_type),
            })
        })
        .collect();

    let response = invoke(python, &Request::Serialize { atoms, bonds })?;
    if let Some(error) = response.error {
        return Err(io::Error::new(io::ErrorKind::InvalidData, error));
    }

    response
        .smiles
        .ok_or_else(|| io::Error::other("RDKit returned no SMILES string"))
}

fn invoke(python: &Path, request: &Request<'_>) -> io::Result<Response> {
    let input = serde_json::to_vec(request)?;
    let command = CommandSpec::new(python.as_os_str())
        .args(["-E", "-c", PYTHON])
        .stdin(input)
        .timeout(Duration::from_secs(30))
        .capture_limits(CaptureLimits::new(4 * 1024 * 1024, 16 * 1024));
    let output = run::run(&command).map_err(|error| io::Error::other(error.to_string()))?;

    serde_json::from_slice(&output.stdout).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("RDKit returned invalid JSON: {error}"),
        )
    })
}

fn bond_type_from_order(order: f64) -> BondType {
    if (order - 1.5).abs() < 0.01 {
        BondType::Aromatic
    } else if (order - 2.0).abs() < 0.01 {
        BondType::Double
    } else if (order - 3.0).abs() < 0.01 {
        BondType::Triple
    } else if (order - 4.0).abs() < 0.01 {
        BondType::Quadruple
    } else if (order - 1.0).abs() < 0.01 {
        BondType::Single
    } else {
        BondType::Unknown
    }
}

fn bond_type_for_rdkit(bond_type: BondType) -> &'static str {
    match bond_type {
        BondType::Double => "double",
        BondType::Triple => "triple",
        BondType::Aromatic => "aromatic",
        BondType::Quadruple => "quadruple",
        BondType::Single
        | BondType::Amide
        | BondType::Dummy
        | BondType::Unknown
        | BondType::Delocalized
        | BondType::PolymericLink
        | BondType::NotConnected => "single",
    }
}
