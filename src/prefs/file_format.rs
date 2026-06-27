//! For saving and loading `ToSave` to disk. We use a binary format to save disk space and time. It's
//! a simple packet-based format; this makes existing save files robust to changes in the struct. Some parts
//! of it will break, but the parts broken are contained in the packet affected.
//!
//! Packets can be in any order, and there can be any number of packets. Each packet carries its own
//! length, so unknown packets (e.g. from a newer version) can be skipped without losing the rest of the file.
//!
//! Uses the `.mca` file extension.
//!
//! Serialization strategy: all *molchanica* structs are hand-rolled inline here (little-endian via the
//! `copy_le!` / `parse_le!` macros, length-prefixed for variable data). No helper sub-functions. Types
//! from external crates (`dynamics`, `bio_apis`, `lin_alg`, `graphics`) that already implement bincode's
//! `Encode`/`Decode` are stored as length-prefixed bincode blobs.
//!
//! Note: the `parse_le!` macro (and raw slicing) panics on truncated/corrupt payloads rather than
//! returning an error. The file-level framing (`McaHeader`, packet bounds) is still bounds-checked so a
//! missing or short file is reported as an error (and treated as "first run") instead of crashing.

#![allow(unused_assignments)]

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

use chrono::{TimeZone, Utc};

use crate::{
    copy_le,
    docking::DockingSite,
    drawing::MoleculeView,
    md::MdBackend,
    molecules::{
        MolIdent,
        lipid::LipidShape,
        nucleic_acid::{NucleicAcidType, Strands},
    },
    parse_le,
    prefs::{
        ControlSchemeType, ControlSettings, Graphics, MdPrefs, OpenHistory, OpenType, PerMolToSave,
        ToSave, UiPrefs,
    },
    selection::{Selection, ViewSelLevel},
    sfc_mesh::MeshColoring,
    state::{
        CamSnapshot, LabelVis, LipidUi, MsaaSetting, NucleicAcidUi, ResColoring, UiVisibility,
        Visibility,
    },
};

/// A sanity check: The start byte will always be this.
const MCA_START_BYTE: u8 = 0x69;
/// Bumped if we ever need to branch on layout changes at the file level.
const MCA_VERSION: u8 = 1;

const MCA_HEADER_SIZE: usize = 3; // start byte, version, packet count.
const PACKET_HEADER_SIZE: usize = 5; // packet type (1) + payload length (4).

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
/// The repr is the byte which uniquely identifies the packet type. These
/// loosely correspond to `ToSave` fields, but may diverge as required.
enum PacketType {
    PerMol = 0,
    OpenHistory = 1,
    ControlScheme = 2,
    Graphics = 3,
    ControlSettings = 4,
    Md = 5,
    UiPrefs = 6,
    PubchemProps = 7,
    Lipid = 8,
    NucleicAcid = 9,
    /// Loose scalar fields that don't warrant their own packet.
    Misc = 10,
}

impl PacketType {
    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::PerMol,
            1 => Self::OpenHistory,
            2 => Self::ControlScheme,
            3 => Self::Graphics,
            4 => Self::ControlSettings,
            5 => Self::Md,
            6 => Self::UiPrefs,
            7 => Self::PubchemProps,
            8 => Self::Lipid,
            9 => Self::NucleicAcid,
            10 => Self::Misc,
            _ => return None,
        })
    }
}

/// One at the top of the file.
pub struct McaHeader {
    /// We don't expect to exceed 255 packets, so u8 is fine.
    pub num_packets: u8,
}

impl McaHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = vec![0; MCA_HEADER_SIZE];
        result[0] = MCA_START_BYTE;
        result[1] = MCA_VERSION;
        result[2] = self.num_packets;
        result
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < MCA_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "mca: file too short for header",
            ));
        }
        if data[0] != MCA_START_BYTE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mca: bad start byte",
            ));
        }
        // data[1] is the format version, reserved for future migrations.
        Ok(Self {
            num_packets: data[2],
        })
    }
}

/// One per packet. The type is stored as a raw byte (not a `PacketType`) so unknown
/// packet types from newer versions can still be skipped via `payload_len`.
pub struct PacketHeader {
    pub packet_type: u8,
    pub payload_len: u32,
}

impl PacketHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = vec![0; PACKET_HEADER_SIZE];
        result[0] = self.packet_type;
        copy_le!(result, self.payload_len, 1..5);
        result
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < PACKET_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "mca: truncated packet header",
            ));
        }
        Ok(Self {
            packet_type: data[0],
            payload_len: parse_le!(data, u32, 1..5),
        })
    }
}

// --- Simple enum <-> u8 mappings for molchanica enums. ---

impl MsaaSetting {
    pub(crate) fn to_u8(self) -> u8 {
        self as u8 // None = 1, Four = 4
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::None,
            4 => Self::Four,
            _ => Self::default(),
        }
    }
}

impl ResColoring {
    pub(crate) fn to_u8(self) -> u8 {
        match self {
            Self::AminoAcid => 0,
            Self::Position => 1,
            Self::Hydrophobicity => 2,
            Self::SiftsUniprot => 3,
            Self::Chain => 4,
        }
    }
    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::AminoAcid,
            1 => Self::Position,
            2 => Self::Hydrophobicity,
            3 => Self::SiftsUniprot,
            4 => Self::Chain,
            _ => Self::default(),
        }
    }
}

impl MeshColoring {
    fn to_u8(self) -> u8 {
        match self {
            Self::Solid => 0,
            Self::Element => 1,
            Self::PartialCharge => 2,
            Self::Lipophilicity => 3,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Solid,
            1 => Self::Element,
            2 => Self::PartialCharge,
            3 => Self::Lipophilicity,
            _ => Self::default(),
        }
    }
}

impl MoleculeView {
    fn to_u8(self) -> u8 {
        match self {
            Self::Backbone => 0,
            Self::Sticks => 1,
            Self::BallAndStick => 2,
            Self::SpaceFill => 3,
            Self::Ribbon => 4,
            Self::Surface => 5,
            Self::Dots => 6,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Backbone,
            1 => Self::Sticks,
            2 => Self::BallAndStick,
            3 => Self::SpaceFill,
            4 => Self::Ribbon,
            5 => Self::Surface,
            6 => Self::Dots,
            _ => Self::default(),
        }
    }
}

impl ViewSelLevel {
    fn to_u8(self) -> u8 {
        match self {
            Self::Atom => 0,
            Self::Bond => 1,
            Self::Residue => 2,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Atom,
            1 => Self::Bond,
            2 => Self::Residue,
            _ => Self::default(),
        }
    }
}

impl MdBackend {
    fn to_u8(self) -> u8 {
        match self {
            Self::Dynamics => 0,
            Self::Gromacs => 1,
            Self::Orca => 2,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Dynamics,
            1 => Self::Gromacs,
            2 => Self::Orca,
            _ => Self::default(),
        }
    }
}

impl LipidShape {
    fn to_u8(self) -> u8 {
        match self {
            Self::Free => 0,
            Self::Membrane => 1,
            Self::Liposome => 2,
            Self::Lnp => 3,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Free,
            1 => Self::Membrane,
            2 => Self::Liposome,
            3 => Self::Lnp,
            _ => Self::default(),
        }
    }
}

impl NucleicAcidType {
    fn to_u8(self) -> u8 {
        match self {
            Self::Dna => 0,
            Self::Rna => 1,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Rna,
            _ => Self::Dna,
        }
    }
}

impl Strands {
    fn to_u8(self) -> u8 {
        match self {
            Self::Single => 0,
            Self::Double => 1,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Single,
            _ => Self::Double,
        }
    }
}

impl OpenType {
    pub(crate) fn to_u8(self) -> u8 {
        match self {
            Self::Peptide => 0,
            Self::Ligand => 1,
            Self::NucleicAcid => 2,
            Self::Lipid => 3,
            Self::Pocket => 4,
            Self::Map => 5,
            Self::Frcmod => 6,
            Self::Trajectory => 7,
            Self::MdParams => 8,
            Self::ParquetDb => 9,
            Self::MdMols => 10,
        }
    }
    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Ligand,
            2 => Self::NucleicAcid,
            3 => Self::Lipid,
            4 => Self::Pocket,
            5 => Self::Map,
            6 => Self::Frcmod,
            7 => Self::Trajectory,
            8 => Self::MdParams,
            9 => Self::ParquetDb,
            10 => Self::MdMols,
            _ => Self::Peptide,
        }
    }
}

impl ControlSchemeType {
    pub(crate) fn to_bytes(self) -> Vec<u8> {
        vec![match self {
            Self::Free => 0,
            Self::Arc => 1,
        }]
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(match data[0] {
            1 => Self::Arc,
            _ => Self::Free,
        })
    }
}

// --- Leaf molchanica structs. ---

impl LabelVis {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        vec![
            self.mol as u8,
            self.atom_sn as u8,
            self.atom_q as u8,
            self.atom_detailed as u8,
            self.bond as u8,
            self.chain as u8,
        ]
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(Self {
            mol: data[0] != 0,
            atom_sn: data[1] != 0,
            atom_q: data[2] != 0,
            atom_detailed: data[3] != 0,
            bond: data[4] != 0,
            chain: data[5] != 0,
        })
    }
}

impl Visibility {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![
            self.hide_sidechains as u8,
            self.hide_water as u8,
            self.hide_hetero as u8,
            self.hide_protein as u8,
            self.hide_ligand as u8,
            self.hide_nucleic_acids as u8,
            self.hide_lipids as u8,
            self.hide_hydrogen as u8,
            self.hide_pharmacophore as u8,
            self.hide_h_bonds as u8,
            self.dim_peptide as u8,
            self.hide_density_point_cloud as u8,
            self.hide_density_surface as u8,
            self.hide_pockets as u8,
        ];
        let labels = self.labels.to_bytes();
        out.extend_from_slice(&(labels.len() as u32).to_le_bytes());
        out.extend_from_slice(&labels);
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let labels_len = parse_le!(data, u32, 14..18) as usize;
        let labels = LabelVis::from_bytes(&data[18..18 + labels_len])?;
        Ok(Self {
            hide_sidechains: data[0] != 0,
            hide_water: data[1] != 0,
            hide_hetero: data[2] != 0,
            hide_protein: data[3] != 0,
            hide_ligand: data[4] != 0,
            hide_nucleic_acids: data[5] != 0,
            hide_lipids: data[6] != 0,
            hide_hydrogen: data[7] != 0,
            hide_pharmacophore: data[8] != 0,
            hide_h_bonds: data[9] != 0,
            dim_peptide: data[10] != 0,
            hide_density_point_cloud: data[11] != 0,
            hide_density_surface: data[12] != 0,
            hide_pockets: data[13] != 0,
            labels,
        })
    }
}

impl UiVisibility {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        vec![
            self.aa_seq as u8,
            self.smiles as u8,
            self.selfies as u8,
            self.lipids as u8,
            self.nucleic_acids as u8,
            self.amino_acids as u8,
            self.dynamics as u8,
            self.orca as u8,
            self.mol_char as u8,
            self.pharmacophore_list as u8,
        ]
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(Self {
            aa_seq: data[0] != 0,
            smiles: data[1] != 0,
            selfies: data[2] != 0,
            lipids: data[3] != 0,
            nucleic_acids: data[4] != 0,
            amino_acids: data[5] != 0,
            dynamics: data[6] != 0,
            orca: data[7] != 0,
            mol_char: data[8] != 0,
            pharmacophore_list: data[9] != 0,
        })
    }
}

impl CamSnapshot {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let pos =
            bincode::encode_to_vec(&self.position, bincode::config::standard()).unwrap_or_default();
        out.extend_from_slice(&(pos.len() as u32).to_le_bytes());
        out.extend_from_slice(&pos);
        let ori = bincode::encode_to_vec(&self.orientation, bincode::config::standard())
            .unwrap_or_default();
        out.extend_from_slice(&(ori.len() as u32).to_le_bytes());
        out.extend_from_slice(&ori);
        out.extend_from_slice(&self.far.to_le_bytes());
        out.extend_from_slice(&(self.name.len() as u32).to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let position = bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
            .unwrap()
            .0;
        i += len;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let orientation =
            bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
        i += len;
        let far = parse_le!(data, f32, i..i + 4);
        i += 4;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let name = String::from_utf8(data[i..i + len].to_vec()).unwrap();
        Ok(Self {
            position,
            orientation,
            far,
            name,
        })
    }
}

impl DockingSite {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let center = bincode::encode_to_vec(&self.site_center, bincode::config::standard())
            .unwrap_or_default();
        out.extend_from_slice(&(center.len() as u32).to_le_bytes());
        out.extend_from_slice(&center);
        out.extend_from_slice(&self.site_radius.to_le_bytes());
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let site_center =
            bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
        i += len;
        let site_radius = parse_le!(data, f64, i..i + 8);
        Ok(Self {
            site_center,
            site_radius,
        })
    }
}

impl Selection {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Selection::None => out.push(0),
            Selection::AtomPeptide(i) => {
                out.push(1);
                out.extend_from_slice(&(*i as u64).to_le_bytes());
            }
            Selection::Residue(i) => {
                out.push(2);
                out.extend_from_slice(&(*i as u64).to_le_bytes());
            }
            Selection::Residues(v) => {
                out.push(3);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&(*x as u64).to_le_bytes());
                }
            }
            Selection::AtomsPeptide(v) => {
                out.push(4);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&(*x as u64).to_le_bytes());
                }
            }
            Selection::AtomLig((a, b)) => {
                out.push(5);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::AtomsLig((a, v)) => {
                out.push(6);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&(*x as u64).to_le_bytes());
                }
            }
            Selection::AtomNucleicAcid((a, b)) => {
                out.push(7);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::AtomLipid((a, b)) => {
                out.push(8);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::AtomPocket((a, b)) => {
                out.push(9);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::BondPeptide(i) => {
                out.push(10);
                out.extend_from_slice(&(*i as u64).to_le_bytes());
            }
            Selection::BondLig((a, b)) => {
                out.push(11);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::BondsLig((a, v)) => {
                out.push(12);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&(*x as u64).to_le_bytes());
                }
            }
            Selection::BondNucleicAcid((a, b)) => {
                out.push(13);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::BondLipid((a, b)) => {
                out.push(14);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::BondPocket((a, b)) => {
                out.push(15);
                out.extend_from_slice(&(*a as u64).to_le_bytes());
                out.extend_from_slice(&(*b as u64).to_le_bytes());
            }
            Selection::ComponentEditor(i) => {
                out.push(16);
                out.extend_from_slice(&(*i as u64).to_le_bytes());
            }
        }
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 1;
        Ok(match data[0] {
            1 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                Selection::AtomPeptide(a)
            }
            2 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                Selection::Residue(a)
            }
            3 => {
                let n = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(parse_le!(data, u64, i..i + 8) as usize);
                    i += 8;
                }
                Selection::Residues(v)
            }
            4 => {
                let n = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(parse_le!(data, u64, i..i + 8) as usize);
                    i += 8;
                }
                Selection::AtomsPeptide(v)
            }
            5 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::AtomLig((a, b))
            }
            6 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let n = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(parse_le!(data, u64, i..i + 8) as usize);
                    i += 8;
                }
                Selection::AtomsLig((a, v))
            }
            7 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::AtomNucleicAcid((a, b))
            }
            8 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::AtomLipid((a, b))
            }
            9 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::AtomPocket((a, b))
            }
            10 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                Selection::BondPeptide(a)
            }
            11 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::BondLig((a, b))
            }
            12 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let n = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(parse_le!(data, u64, i..i + 8) as usize);
                    i += 8;
                }
                Selection::BondsLig((a, v))
            }
            13 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::BondNucleicAcid((a, b))
            }
            14 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::BondLipid((a, b))
            }
            15 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                i += 8;
                let b = parse_le!(data, u64, i..i + 8) as usize;
                Selection::BondPocket((a, b))
            }
            16 => {
                let a = parse_le!(data, u64, i..i + 8) as usize;
                Selection::ComponentEditor(a)
            }
            _ => Selection::None,
        })
    }
}

impl MolIdent {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            MolIdent::PubChem(n) => {
                out.push(0);
                out.extend_from_slice(&n.to_le_bytes());
            }
            MolIdent::DrugBank(s) => {
                out.push(1);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MolIdent::PdbeAmber(s) => {
                out.push(2);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MolIdent::Smiles(s) => {
                out.push(3);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MolIdent::InchI(s) => {
                out.push(4);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MolIdent::InchIKey(s) => {
                out.push(5);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MolIdent::IupacName(s) => {
                out.push(6);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MolIdent::PubchemTitle(s) => {
                out.push(7);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
        }
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 1;
        Ok(match data[0] {
            0 => MolIdent::PubChem(parse_le!(data, u32, i..i + 4)),
            1 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::DrugBank(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            2 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::PdbeAmber(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            3 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::Smiles(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            4 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::InchI(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            5 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::InchIKey(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            6 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::IupacName(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            7 => {
                let len = parse_le!(data, u32, i..i + 4) as usize;
                i += 4;
                MolIdent::PubchemTitle(String::from_utf8(data[i..i + len].to_vec()).unwrap())
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("mca: unknown MolIdent tag {other}"),
                ));
            }
        })
    }
}

impl LipidUi {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![0; 11];
        copy_le!(out, self.lipid_to_add as u64, 0..8);
        out[8] = self.shape.to_u8();
        copy_le!(out, self.mol_count, 9..11);
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(Self {
            lipid_to_add: parse_le!(data, u64, 0..8) as usize,
            shape: LipidShape::from_u8(data[8]),
            mol_count: parse_le!(data, u16, 9..11),
        })
    }
}

impl NucleicAcidUi {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.seq_to_create.len() as u32).to_le_bytes());
        out.extend_from_slice(self.seq_to_create.as_bytes());
        out.push(self.na_type.to_u8());
        out.push(self.strands.to_u8());
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let seq_to_create = String::from_utf8(data[i..i + len].to_vec()).unwrap();
        i += len;
        let na_type = NucleicAcidType::from_u8(data[i]);
        i += 1;
        let strands = Strands::from_u8(data[i]);
        Ok(Self {
            seq_to_create,
            na_type,
            strands,
        })
    }
}

impl OpenHistory {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.timestamp.timestamp().to_le_bytes());
        let path = self.path.to_string_lossy();
        out.extend_from_slice(&(path.len() as u32).to_le_bytes());
        out.extend_from_slice(path.as_bytes());
        out.push(self.type_.to_u8());
        match &self.ident {
            Some(s) => {
                out.push(1);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            None => out.push(0),
        }
        match &self.position {
            Some(p) => {
                out.push(1);
                let b = bincode::encode_to_vec(p, bincode::config::standard()).unwrap_or_default();
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(&b);
            }
            None => out.push(0),
        }
        out.push(self.last_session as u8);
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let secs = parse_le!(data, i64, i..i + 8);
        i += 8;
        let timestamp = Utc.timestamp_opt(secs, 0).single().unwrap_or_else(Utc::now);
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let path = PathBuf::from(String::from_utf8(data[i..i + len].to_vec()).unwrap());
        i += len;
        let type_ = OpenType::from_u8(data[i]);
        i += 1;
        let has_ident = data[i] != 0;
        i += 1;
        let ident = if has_ident {
            let len = parse_le!(data, u32, i..i + 4) as usize;
            i += 4;
            let s = String::from_utf8(data[i..i + len].to_vec()).unwrap();
            i += len;
            Some(s)
        } else {
            None
        };
        let has_pos = data[i] != 0;
        i += 1;
        let position = if has_pos {
            let len = parse_le!(data, u32, i..i + 4) as usize;
            i += 4;
            let p = bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
            i += len;
            Some(p)
        } else {
            None
        };
        let last_session = data[i] != 0;
        Ok(Self {
            timestamp,
            path,
            type_,
            ident,
            position,
            last_session,
        })
    }
}

// --- Direct `ToSave` substructs (one packet each). ---

impl Graphics {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.msaa.to_u8());
        let ao = bincode::encode_to_vec(&self.ambient_occlusion, bincode::config::standard())
            .unwrap_or_default();
        out.extend_from_slice(&(ao.len() as u32).to_le_bytes());
        out.extend_from_slice(&ao);
        match self.edge_cueing {
            Some(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_le_bytes());
            }
            None => out.push(0),
        }
        match self.depth_aware_halos {
            Some(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_le_bytes());
            }
            None => out.push(0),
        }
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let msaa = MsaaSetting::from_u8(data[i]);
        i += 1;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let ambient_occlusion =
            bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
        i += len;
        let has = data[i] != 0;
        i += 1;
        let edge_cueing = if has {
            let v = parse_le!(data, f32, i..i + 4);
            i += 4;
            Some(v)
        } else {
            None
        };
        let has = data[i] != 0;
        i += 1;
        let depth_aware_halos = if has {
            let v = parse_le!(data, f32, i..i + 4);
            i += 4;
            Some(v)
        } else {
            None
        };
        Ok(Self {
            msaa,
            ambient_occlusion,
            edge_cueing,
            depth_aware_halos,
        })
    }
}

impl ControlSettings {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        vec![self.movement_speed, self.rotation_sens, self.mol_move_sens]
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Ok(Self {
            movement_speed: data[0],
            rotation_sens: data[1],
            mol_move_sens: data[2],
        })
    }
}

impl MdPrefs {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let cfg =
            bincode::encode_to_vec(&self.config, bincode::config::standard()).unwrap_or_default();
        out.extend_from_slice(&(cfg.len() as u32).to_le_bytes());
        out.extend_from_slice(&cfg);
        out.extend_from_slice(&self.num_steps.to_le_bytes());
        out.extend_from_slice(&self.dt.to_le_bytes());
        out.push(self.backend.to_u8());
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let config = bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
            .unwrap()
            .0;
        i += len;
        let num_steps = parse_le!(data, u32, i..i + 4);
        i += 4;
        let dt = parse_le!(data, f32, i..i + 4);
        i += 4;
        let backend = MdBackend::from_u8(data[i]);
        Ok(Self {
            config,
            num_steps,
            dt,
            backend,
        })
    }
}

impl UiPrefs {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let sel = self.selection.to_bytes();
        out.extend_from_slice(&(sel.len() as u32).to_le_bytes());
        out.extend_from_slice(&sel);
        out.extend_from_slice(&(self.cam_snapshots.len() as u32).to_le_bytes());
        for cs in &self.cam_snapshots {
            let b = cs.to_bytes();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(&b);
        }
        out.push(self.mol_view.to_u8());
        out.push(self.view_sel_level.to_u8());
        let vis = self.visibility.to_bytes();
        out.extend_from_slice(&(vis.len() as u32).to_le_bytes());
        out.extend_from_slice(&vis);
        let uvis = self.ui_visibility.to_bytes();
        out.extend_from_slice(&(uvis.len() as u32).to_le_bytes());
        out.extend_from_slice(&uvis);
        out.push(self.near_sel_only as u8);
        out.push(self.near_lig_only as u8);
        out.extend_from_slice(&self.nearby_dist_thresh.to_le_bytes());
        out.push(self.mol_view_peptide.to_u8());
        out
    }
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let selection = Selection::from_bytes(&data[i..i + len])?;
        i += len;
        let n = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let mut cam_snapshots = Vec::with_capacity(n);
        for _ in 0..n {
            let len = parse_le!(data, u32, i..i + 4) as usize;
            i += 4;
            cam_snapshots.push(CamSnapshot::from_bytes(&data[i..i + len])?);
            i += len;
        }
        let mol_view = MoleculeView::from_u8(data[i]);
        i += 1;
        let view_sel_level = ViewSelLevel::from_u8(data[i]);
        i += 1;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let visibility = Visibility::from_bytes(&data[i..i + len])?;
        i += len;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let ui_visibility = UiVisibility::from_bytes(&data[i..i + len])?;
        i += len;
        let near_sel_only = data[i] != 0;
        i += 1;
        let near_lig_only = data[i] != 0;
        i += 1;
        let nearby_dist_thresh = parse_le!(data, u16, i..i + 2);
        i += 2;
        let mol_view_peptide = if i < data.len() {
            MoleculeView::from_u8(data[i])
        } else {
            mol_view
        };
        Ok(Self {
            selection,
            cam_snapshots,
            mol_view: mol_view.non_peptide_or_default(),
            mol_view_peptide,
            view_sel_level,
            visibility,
            ui_visibility,
            near_sel_only,
            near_lig_only,
            nearby_dist_thresh,
        })
    }
}

// --- Top-level save / load. ---

impl ToSave {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut packets: Vec<(PacketType, Vec<u8>)> = Vec::new();

        // PerMol: u32 count, then (u32 key len, key, u32 val len, val) per entry.
        {
            let mut out = Vec::new();
            out.extend_from_slice(&(self.per_mol.len() as u32).to_le_bytes());
            for (k, v) in &self.per_mol {
                out.extend_from_slice(&(k.len() as u32).to_le_bytes());
                out.extend_from_slice(k.as_bytes());
                let b = v.to_bytes();
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(&b);
            }
            packets.push((PacketType::PerMol, out));
        }

        // OpenHistory: u32 count, then (u32 len, item) per entry.
        {
            let mut out = Vec::new();
            out.extend_from_slice(&(self.open_history.len() as u32).to_le_bytes());
            for oh in &self.open_history {
                let b = oh.to_bytes();
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(&b);
            }
            packets.push((PacketType::OpenHistory, out));
        }

        packets.push((PacketType::ControlScheme, self.control_scheme.to_bytes()));
        packets.push((PacketType::Graphics, self.graphics.to_bytes()));
        packets.push((
            PacketType::ControlSettings,
            self.control_settings.to_bytes(),
        ));
        packets.push((PacketType::Md, self.md.to_bytes()));
        packets.push((PacketType::UiPrefs, self.ui_prefs.to_bytes()));

        // PubchemProps: u32 count, then (u32 key len, key, u32 val len, val-bincode) per entry.
        {
            let mut out = Vec::new();
            out.extend_from_slice(&(self.pubchem_properties_map.len() as u32).to_le_bytes());
            for (k, v) in &self.pubchem_properties_map {
                let kb = k.to_bytes();
                out.extend_from_slice(&(kb.len() as u32).to_le_bytes());
                out.extend_from_slice(&kb);
                let vb = bincode::encode_to_vec(v, bincode::config::standard()).unwrap_or_default();
                out.extend_from_slice(&(vb.len() as u32).to_le_bytes());
                out.extend_from_slice(&vb);
            }
            packets.push((PacketType::PubchemProps, out));
        }

        packets.push((PacketType::Lipid, self.lipid.to_bytes()));
        packets.push((PacketType::NucleicAcid, self.nucleic_acid.to_bytes()));

        // Misc: loose scalar fields.
        {
            let mut out = Vec::new();
            out.extend_from_slice(&self.sa_surface_precision.to_le_bytes());
            out.extend_from_slice(&self.ph.to_le_bytes());
            out.push(self.mesh_coloring.to_u8());
            out.push(self.auto_fog as u8);
            packets.push((PacketType::Misc, out));
        }

        let mut out = McaHeader {
            num_packets: packets.len() as u8,
        }
        .to_bytes();

        for (packet_type, payload) in &packets {
            out.extend_from_slice(
                &PacketHeader {
                    packet_type: *packet_type as u8,
                    payload_len: payload.len() as u32,
                }
                .to_bytes(),
            );
            out.extend_from_slice(payload);
        }

        fs::write(path, out)
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read(path)?;
        let header = McaHeader::from_bytes(&data)?;

        // Start from defaults; each packet present overrides its field(s). A truncated frame or an
        // unknown packet type is skipped; the rest of the file still loads.
        let mut to_save = ToSave::default();
        let mut i = MCA_HEADER_SIZE;

        for _ in 0..header.num_packets {
            if i + PACKET_HEADER_SIZE > data.len() {
                break; // Truncated packet header.
            }
            let ph = PacketHeader::from_bytes(&data[i..i + PACKET_HEADER_SIZE])?;
            i += PACKET_HEADER_SIZE;

            let len = ph.payload_len as usize;
            if i + len > data.len() {
                break; // Truncated payload.
            }
            let payload = &data[i..i + len];
            i += len;

            let Some(packet_type) = PacketType::from_u8(ph.packet_type) else {
                continue; // Unknown packet type from a newer version; skip.
            };

            match packet_type {
                PacketType::PerMol => {
                    let mut j = 0;
                    let n = parse_le!(payload, u32, j..j + 4) as usize;
                    j += 4;
                    let mut map = HashMap::with_capacity(n);
                    for _ in 0..n {
                        let klen = parse_le!(payload, u32, j..j + 4) as usize;
                        j += 4;
                        let k = String::from_utf8(payload[j..j + klen].to_vec()).unwrap();
                        j += klen;
                        let vlen = parse_le!(payload, u32, j..j + 4) as usize;
                        j += 4;
                        let v = PerMolToSave::from_bytes(&payload[j..j + vlen])?;
                        j += vlen;
                        map.insert(k, v);
                    }
                    to_save.per_mol = map;
                }
                PacketType::OpenHistory => {
                    let mut j = 0;
                    let n = parse_le!(payload, u32, j..j + 4) as usize;
                    j += 4;
                    let mut v = Vec::with_capacity(n);
                    for _ in 0..n {
                        let len = parse_le!(payload, u32, j..j + 4) as usize;
                        j += 4;
                        v.push(OpenHistory::from_bytes(&payload[j..j + len])?);
                        j += len;
                    }
                    to_save.open_history = v;
                }
                PacketType::ControlScheme => {
                    to_save.control_scheme = ControlSchemeType::from_bytes(payload)?;
                }
                PacketType::Graphics => {
                    to_save.graphics = Graphics::from_bytes(payload)?;
                }
                PacketType::ControlSettings => {
                    to_save.control_settings = ControlSettings::from_bytes(payload)?;
                }
                PacketType::Md => {
                    to_save.md = MdPrefs::from_bytes(payload)?;
                }
                PacketType::UiPrefs => {
                    to_save.ui_prefs = UiPrefs::from_bytes(payload)?;
                }
                PacketType::PubchemProps => {
                    let mut j = 0;
                    let n = parse_le!(payload, u32, j..j + 4) as usize;
                    j += 4;
                    let mut map = HashMap::with_capacity(n);
                    for _ in 0..n {
                        let klen = parse_le!(payload, u32, j..j + 4) as usize;
                        j += 4;
                        let k = MolIdent::from_bytes(&payload[j..j + klen])?;
                        j += klen;
                        let vlen = parse_le!(payload, u32, j..j + 4) as usize;
                        j += 4;
                        let v = bincode::decode_from_slice(
                            &payload[j..j + vlen],
                            bincode::config::standard(),
                        )
                        .unwrap()
                        .0;
                        j += vlen;
                        map.insert(k, v);
                    }
                    to_save.pubchem_properties_map = map;
                }
                PacketType::Lipid => {
                    to_save.lipid = LipidUi::from_bytes(payload)?;
                }
                PacketType::NucleicAcid => {
                    to_save.nucleic_acid = NucleicAcidUi::from_bytes(payload)?;
                }
                PacketType::Misc => {
                    let mut j = 0;
                    to_save.sa_surface_precision = parse_le!(payload, f32, j..j + 4);
                    j += 4;
                    to_save.ph = parse_le!(payload, f32, j..j + 4);
                    j += 4;
                    to_save.mesh_coloring = MeshColoring::from_u8(payload[j]);
                    j += 1;
                    to_save.auto_fog = payload[j] != 0;
                }
            }
        }

        Ok(to_save)
    }
}
