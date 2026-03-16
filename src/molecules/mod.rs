#![allow(non_camel_case_types)]

/// This module contains data structures and associated methods for various molecule types. For example:
/// Atoms, Bonds, Residues, Chains, etc. These contain application-specific data and organization that is distinct
/// from `bio_files::AtomGeneric` etc.
/// These are core to the operation of this application.
pub mod common;
mod geom_assignment;
pub mod lipid;
pub mod nucleic_acid;
pub mod peptide;
pub mod pocket;
pub mod rotatable_bonds;
pub mod small;

use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    io,
    io::ErrorKind,
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, BondType, ChainGeneric, Mol2, Pdbqt, ResidueEnd, ResidueGeneric,
    ResidueType, Sdf, Xyz,
};
use dynamics::Dihedral;
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, AtomTypeInRes, Element};
use peptide::MoleculePeptide;
use small::MoleculeSmall;

use crate::{
    drawing::EntityClass,
    molecules::{
        common::MoleculeCommon, lipid::MoleculeLipid, nucleic_acid::MoleculeNucleicAcid,
        pocket::Pocket,
    },
    prefs::OpenType,
};

// A flag indicating that a small molecule file describes a pocket, and
// should be loaded as such.
pub const POCKET_METADATA_KEY: &str = "is_pocket";
pub const POCKET_METADATA_VAL: &str = "true";

// A metadata field which contains the position of atoms
// in a pharmacophore-type small molecule's pocket
pub const PHARMACOPHORE_POCKET_ATOMS_KEY: &str = "pharmacophore_pocket_atoms";

/// A trait-based molecule.
// todo: Not, or barely used currently. We currently use MolGenricRef for the most part.
pub trait MolGeneric {
    fn common(&self) -> &MoleculeCommon;
    fn common_mut(&mut self) -> &mut MoleculeCommon;
    fn to_ref(&self) -> MolGenericRef<'_>;
    fn mol_type(&self) -> MolType;
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MolType {
    Peptide,
    Ligand,
    NucleicAcid,
    Lipid,
    Pocket,
    Water,
}

impl MolType {
    pub fn to_open_type(self) -> OpenType {
        use MolType::*;
        match self {
            Peptide => OpenType::Peptide,
            Ligand => OpenType::Ligand,
            NucleicAcid => OpenType::NucleicAcid,
            Lipid => OpenType::Lipid,
            Pocket => OpenType::Pocket,
            Water => panic!("Can't convert water to open type"),
        }
    }

    pub fn entity_type(self) -> EntityClass {
        use MolType::*;
        match self {
            Peptide => EntityClass::Protein,
            Ligand => EntityClass::Ligand,
            NucleicAcid => EntityClass::NucleicAcid,
            Lipid => EntityClass::Lipid,
            Pocket => EntityClass::Pocket,
            Water => EntityClass::Protein, // todo for now
        }
    }

    pub fn default_file_ext(self) -> String {
        use MolType::*;
        match self {
            Peptide => "cif",
            Ligand | Pocket => "mol2",
            NucleicAcid => "mol2", // todo?
            Lipid => "mol2",       // todo?
            Water => "",
        }
        .to_string()
    }

    pub fn color(self) -> (u8, u8, u8) {
        use MolType::*;
        match self {
            // todo: Update A/R
            Peptide => (0, 255, 255),
            Ligand => (0, 255, 0),
            NucleicAcid => (255, 255, 0),
            Lipid => (255, 0, 255),
            Pocket => (255, 255, 255),
            Water => (0, 0, 0),
        }
    }
}

#[derive(Debug)]
pub enum MoleculeGeneric {
    Peptide(MoleculePeptide),
    Small(MoleculeSmall),
    NucleicAcid(MoleculeNucleicAcid),
    Lipid(MoleculeLipid),
    Pocket(Pocket),
}

impl MoleculeGeneric {
    pub fn common(&self) -> &MoleculeCommon {
        use MoleculeGeneric::*;
        match self {
            Peptide(m) => &m.common,
            Small(m) => &m.common,
            NucleicAcid(m) => &m.common,
            Lipid(m) => &m.common,
            Pocket(m) => &m.common,
        }
    }

    pub fn common_mut(&mut self) -> &mut MoleculeCommon {
        use MoleculeGeneric::*;
        match self {
            Peptide(m) => &mut m.common,
            Small(m) => &mut m.common,
            NucleicAcid(m) => &mut m.common,
            Lipid(m) => &mut m.common,
            Pocket(m) => &mut m.common,
        }
    }

    pub fn mol_type(&self) -> MolType {
        use MoleculeGeneric::*;
        match self {
            Peptide(_) => MolType::Peptide,
            Small(_) => MolType::Ligand,
            NucleicAcid(_) => MolType::NucleicAcid,
            Lipid(_) => MolType::Lipid,
            Pocket(_) => MolType::Pocket,
        }
    }
}

/// We currently use this for mol description.
#[derive(Clone, Debug)]
pub enum MolGenericRef<'a> {
    Peptide(&'a MoleculePeptide),
    Small(&'a MoleculeSmall),
    NucleicAcid(&'a MoleculeNucleicAcid),
    Lipid(&'a MoleculeLipid),
    Pocket(&'a Pocket),
}

impl<'a> MolGenericRef<'a> {
    pub fn common(&self) -> &MoleculeCommon {
        use MolGenericRef::*;
        match self {
            Peptide(m) => &m.common,
            Small(m) => &m.common,
            NucleicAcid(m) => &m.common,
            Lipid(m) => &m.common,
            Pocket(m) => &m.common,
        }
    }

    pub fn mol_type(&self) -> MolType {
        use MolGenericRef::*;
        match self {
            Peptide(_) => MolType::Peptide,
            Small(_) => MolType::Ligand,
            NucleicAcid(_) => MolType::NucleicAcid,
            Lipid(_) => MolType::Lipid,
            Pocket(_) => MolType::Pocket,
        }
    }

    /// Note: Serial numbers mus be sequential prior to running this, as SDF format
    /// doesn't include serial numbers; this will break bonds.
    pub fn to_sdf(&self) -> io::Result<Sdf> {
        match self {
            Self::Small(l) => Ok(l.to_sdf()),
            Self::Pocket(p) => {
                // Mark the metadata so we know when loading to handle this as a pocket.
                let mut metadata = p.common.metadata.clone();
                metadata.insert(
                    POCKET_METADATA_KEY.to_string(),
                    POCKET_METADATA_VAL.to_string(),
                );

                Ok(MoleculeSmall {
                    common: MoleculeCommon {
                        metadata,
                        ..p.common.clone()
                    },
                    ..Default::default()
                }
                .to_sdf())
            }
            _ => Err(io::Error::other("Not implemented")),
        }
    }

    pub fn to_mol2(&self) -> io::Result<Mol2> {
        match self {
            Self::Small(l) => Ok(l.to_mol2()),
            Self::Pocket(p) => {
                // Mark the metadata so we know when loading to handle this as a pocket.
                let mut metadata = p.common.metadata.clone();
                metadata.insert(
                    POCKET_METADATA_KEY.to_string(),
                    POCKET_METADATA_VAL.to_string(),
                );

                Ok(MoleculeSmall {
                    common: MoleculeCommon {
                        metadata,
                        ..p.common.clone()
                    },
                    ..Default::default()
                }
                .to_mol2())
            }
            _ => Err(io::Error::other("Not implemented")),
        }
    }

    pub fn to_pdbqt(&self) -> io::Result<Pdbqt> {
        match self {
            Self::Small(l) => Ok(l.to_pdbqt()),
            Self::Pocket(p) => {
                // Mark the metadata so we know when loading to handle this as a pocket.
                let mut metadata = p.common.metadata.clone();
                metadata.insert(
                    POCKET_METADATA_KEY.to_string(),
                    POCKET_METADATA_VAL.to_string(),
                );

                Ok(MoleculeSmall {
                    common: MoleculeCommon {
                        metadata,
                        ..p.common.clone()
                    },
                    ..Default::default()
                }
                .to_pdbqt())
            }
            _ => Err(io::Error::other("Not implemented")),
        }
    }

    pub fn to_xyz(&self) -> io::Result<Xyz> {
        match self {
            Self::Small(l) => Ok(l.to_xyz()),
            Self::Pocket(p) => Ok(MoleculeSmall {
                common: p.common.clone(),
                ..Default::default()
            }
            .to_xyz()),
            _ => Err(io::Error::other("Not implemented")),
        }
    }
}

/// We currently use this for mol description.
#[derive(Debug)]
pub enum MolGenericRefMut<'a> {
    Peptide(&'a mut MoleculePeptide),
    Small(&'a mut MoleculeSmall),
    NucleicAcid(&'a mut MoleculeNucleicAcid),
    Lipid(&'a mut MoleculeLipid),
    Pocket(&'a mut Pocket),
}

impl<'a> MolGenericRefMut<'a> {
    pub fn common_mut(&mut self) -> &mut MoleculeCommon {
        use MolGenericRefMut::*;
        match self {
            Peptide(m) => &mut m.common,
            Small(m) => &mut m.common,
            NucleicAcid(m) => &mut m.common,
            Lipid(m) => &mut m.common,
            Pocket(m) => &mut m.common,
        }
    }

    pub fn common(&self) -> &MoleculeCommon {
        use MolGenericRefMut::*;
        match self {
            Peptide(m) => &m.common,
            Small(m) => &m.common,
            NucleicAcid(m) => &m.common,
            Lipid(m) => &m.common,
            Pocket(m) => &m.common,
        }
    }

    pub fn mol_type(&self) -> MolType {
        use MolGenericRefMut::*;
        match self {
            Peptide(_) => MolType::Peptide,
            Small(_) => MolType::Ligand,
            NucleicAcid(_) => MolType::NucleicAcid,
            Lipid(_) => MolType::Lipid,
            Pocket(_) => MolType::Lipid,
        }
    }

    // pub fn to_immut(&self) -> MoleculeGenericRef<'a> {
    //     match self {
    //         Self::Peptide(m) => MoleculeGenericRef::Peptide(m),
    //         Self::Ligand(m) => MoleculeGenericRef::Ligand(m),
    //         Self::NucleicAcid(m) => MoleculeGenericRef::NucleicAcid(m),
    //         Self::Lipid(m) => MoleculeGenericRef::Lipid(m),
    //     }
    // }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AtomRole {
    C_Alpha,
    C_Prime,
    N_Backbone,
    O_Backbone,
    H_Backbone,
    Sidechain,
    H_Sidechain,
    Water,
}

impl AtomRole {
    pub fn from_type_in_res(tir: &AtomTypeInRes) -> Self {
        match tir {
            AtomTypeInRes::CA => AtomRole::C_Alpha,
            AtomTypeInRes::C => AtomRole::C_Prime,
            AtomTypeInRes::N => AtomRole::N_Backbone,
            AtomTypeInRes::O => AtomRole::O_Backbone,
            AtomTypeInRes::H(h_type) => match h_type.as_ref() {
                "H" | "H1" | "H2" | "H3" | "HA" | "HA2" | "HA3" | "HN" | "HT1" | "HT2" | "HT3" => {
                    Self::H_Backbone
                }
                _ => Self::Sidechain,
            },
            _ => Self::Sidechain,
        }
    }
}

impl Display for AtomRole {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AtomRole::C_Alpha => write!(f, "Cα"),
            AtomRole::C_Prime => write!(f, "C'"),
            AtomRole::N_Backbone => write!(f, "N (bb)"),
            AtomRole::O_Backbone => write!(f, "O (bb)"),
            AtomRole::H_Backbone => write!(f, "H (bb)"),
            AtomRole::Sidechain => write!(f, "Side"),
            AtomRole::H_Sidechain => write!(f, "H SC"),
            AtomRole::Water => write!(f, "Water"),
        }
    }
}

/// Represents a covalent bond between two atoms. Includes both atom indices for fast lookups,
/// and serial numbers for stability.
#[derive(Debug, Clone)]
pub struct Bond {
    pub bond_type: BondType,
    pub atom_0_sn: u32,
    pub atom_1_sn: u32,
    /// Index
    pub atom_0: usize,
    /// Index
    pub atom_1: usize,
    pub is_backbone: bool,
}

impl Bond {
    /// For example, if we wish to populate indices later.
    pub fn new_basic(atom_0_sn: u32, atom_1_sn: u32, bond_type: BondType) -> Self {
        Self {
            bond_type,
            atom_0_sn,
            atom_1_sn,
            atom_0: 0,
            atom_1: 0,
            is_backbone: false,
        }
    }

    pub fn to_generic(&self) -> BondGeneric {
        BondGeneric {
            bond_type: self.bond_type,
            atom_0_sn: self.atom_0_sn,
            atom_1_sn: self.atom_1_sn,
        }
    }
}

impl Bond {
    pub fn from_generic(bond: &BondGeneric, atom_set: &[Atom]) -> io::Result<Self> {
        let atom_0 = match atom_sns_to_indices(bond.atom_0_sn, atom_set) {
            Some(i) => i,
            None => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Unable to find atom0 SN when loading from generic bond: {}",
                        bond.atom_0_sn
                    ),
                ));
            }
        };

        // todo DRY
        let atom_1 = match atom_sns_to_indices(bond.atom_1_sn, atom_set) {
            Some(i) => i,
            None => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Unable to find atom1 SN when loading from generic bond: {}",
                        bond.atom_1_sn
                    ),
                ));
            }
        };

        Ok(Self {
            bond_type: bond.bond_type,
            atom_0_sn: bond.atom_0_sn,
            atom_1_sn: bond.atom_1_sn,
            atom_0,
            atom_1,
            is_backbone: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HydrogenBond {
    /// All three atoms are atom indexes.
    pub donor: usize,
    pub acceptor: usize,
    pub hydrogen: usize,
    pub strength: f32,
}

impl HydrogenBond {
    pub fn new(donor: usize, acceptor: usize, hydrogen: usize, strength: f32) -> Self {
        Self {
            donor,
            acceptor,
            hydrogen,
            strength,
        }
    }
}

/// A bond between two molecules.
#[derive(Debug, Clone)]
pub struct HydrogenBondTwoMols {
    /// Donor and acceptor atoms are (mol, atom) indexes. hydrogen is an atom index, of the
    /// donor molecule.
    pub donor: (usize, usize),
    pub acceptor: (usize, usize),
    pub hydrogen: usize,
    pub strength: f32,
}

impl HydrogenBondTwoMols {
    pub fn new(
        donor: (usize, usize),
        acceptor: (usize, usize),
        hydrogen: usize,
        strength: f32,
    ) -> Self {
        Self {
            donor,
            acceptor,
            hydrogen,
            strength,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Residue {
    /// We use serial number of display, search etc, and array index to select. Residue serial number is not
    /// unique in the molecule; only in the chain.
    pub serial_number: u32,
    pub res_type: ResidueType,
    /// Serial number
    pub atom_sns: Vec<u32>,
    pub atoms: Vec<usize>, // Atom index
    pub dihedral: Option<Dihedral>,
    pub end: ResidueEnd,
}

impl Residue {
    pub fn to_generic(&self) -> ResidueGeneric {
        ResidueGeneric {
            serial_number: self.serial_number,
            res_type: self.res_type.clone(),
            atom_sns: self.atom_sns.clone(),
            end: self.end,
        }
    }
}

impl Residue {
    pub fn from_generic(res: &ResidueGeneric, atom_set: &[Atom]) -> io::Result<Self> {
        let mut atoms = Vec::with_capacity(res.atom_sns.len());

        for sn in &res.atom_sns {
            match atom_sns_to_indices(*sn, atom_set) {
                Some(i) => atoms.push(i),
                None => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find atom SN when loading from generic res",
                    ));
                }
            }
        }

        Ok(Self {
            serial_number: res.serial_number,
            res_type: res.res_type.clone(),
            atom_sns: res.atom_sns.clone(),
            atoms,
            dihedral: None,
            end: res.end,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Chain {
    pub id: String,
    // todo: Do we want both residues and atoms stored here? It's an overconstraint.
    /// Serial number
    pub residue_sns: Vec<u32>,
    pub residues: Vec<usize>,
    /// Serial number
    pub atom_sns: Vec<u32>,
    pub atoms: Vec<usize>,
    pub visible: bool,
}

impl Chain {
    pub fn from_generic(
        chain: &ChainGeneric,
        atom_set: &[Atom],
        res_set: &[Residue],
    ) -> io::Result<Self> {
        // todo: DRY with res code above.
        let mut atoms = Vec::with_capacity(chain.atom_sns.len());
        let mut residues = Vec::with_capacity(chain.residue_sns.len());

        for sn in &chain.atom_sns {
            match atom_sns_to_indices(*sn, atom_set) {
                Some(i) => atoms.push(i),
                None => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find atom SN when loading from generic chain",
                    ));
                }
            }
        }

        // We get this empty set with some small (or old?) residues from PDB, e.g. 1BOM.
        if !res_set.is_empty() {
            for sn in &chain.residue_sns {
                match res_sns_to_indices(*sn, res_set) {
                    Some(i) => residues.push(i),
                    None => {
                        return Err(io::Error::new(
                            ErrorKind::InvalidData,
                            "Unable to find res SN when loading from generic res",
                        ));
                    }
                }
            }
        }

        Ok(Self {
            id: chain.id.clone(),
            residue_sns: chain.residue_sns.clone(),
            residues,
            atom_sns: chain.atom_sns.clone(),
            atoms,
            visible: true,
        })
    }

    pub fn to_generic(&self) -> ChainGeneric {
        ChainGeneric {
            id: self.id.clone(),
            residue_sns: self.residue_sns.clone(),
            atom_sns: self.atom_sns.clone(),
        }
    }
}

/// Helper
fn atom_sns_to_indices(sn_tgt: u32, atom_set: &[Atom]) -> Option<usize> {
    for (i, atom) in atom_set.iter().enumerate() {
        if atom.serial_number == sn_tgt {
            return Some(i);
        }
    }

    None
}

/// Helper, and dry
fn res_sns_to_indices(sn_tgt: u32, res_set: &[Residue]) -> Option<usize> {
    for (i, atom) in res_set.iter().enumerate() {
        if atom.serial_number == sn_tgt {
            return Some(i);
        }
    }

    None
}

impl Display for Residue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "#{}: {}", self.serial_number, self.res_type)?;

        if let Some(dihedral) = &self.dihedral {
            write!(f, "   {}", dihedral)?;
        }

        match self.end {
            ResidueEnd::CTerminus => write!(f, " C-term")?,
            ResidueEnd::NTerminus => write!(f, " N-term")?,
            _ => (),
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Atom {
    pub serial_number: u32,
    pub posit: Vec3,
    pub element: Element,
    /// e.g. "HA", "C", "N", "HB3" etc.
    pub type_in_res: Option<AtomTypeInRes>,
    /// There are too many variants of this (with different numbers) to use an enum effectively
    /// Ideally, we would share a field between lipid and normal type in res, but don't due
    /// to the different in inner type. (Maybe an outer enum?)
    pub type_in_res_general: Option<String>,
    /// "Type 2" for proteins/AA. For ligands and small molecules, this
    /// is a "Type 3".
    /// E.g. "c6", "ca", "n3", "ha", "h0" etc, as seen in Mol2 files from AMBER.
    pub force_field_type: Option<String>,
    // todo: Note that `role` is a subset of `type_in_res`.
    pub role: Option<AtomRole>,
    /// We include these references to the residue and chain indices for speed; iterating through
    /// residues (or chains) to check for atom membership is slow.
    pub residue: Option<usize>,
    pub chain: Option<usize>,
    pub hetero: bool,
    /// For docking.
    pub occupancy: Option<f32>,
    /// Elementary charge. (Charge of a proton)
    pub partial_charge: Option<f32>,
    pub alt_conformation_id: Option<String>,
}

impl Atom {
    /// Note: This doesn't include backbone O etc; just the 3 main ones.
    pub fn is_backbone(&self) -> bool {
        match self.role {
            Some(r) => [
                AtomRole::C_Alpha,
                AtomRole::N_Backbone,
                AtomRole::C_Prime,
                AtomRole::O_Backbone,
            ]
            .contains(&r),
            None => false,
        }
    }

    pub fn to_generic(&self) -> AtomGeneric {
        AtomGeneric {
            serial_number: self.serial_number,
            type_in_res: self.type_in_res.clone(),
            type_in_res_general: self.type_in_res_general.clone(),
            posit: self.posit,
            element: self.element,
            partial_charge: self.partial_charge,
            force_field_type: self.force_field_type.clone(),
            ..Default::default()
        }
    }
}

impl From<&AtomGeneric> for Atom {
    fn from(atom: &AtomGeneric) -> Self {
        let role = atom.type_in_res.as_ref().map(|tir| Some(AtomRole::from_type_in_res(tir)));
        // We will fill out chain and residue later, after chains and residue are loaded.

        Self {
            serial_number: atom.serial_number,
            posit: atom.posit,
            element: atom.element,
            type_in_res: atom.type_in_res.clone(),
            type_in_res_general: atom.type_in_res_general.clone(),
            role,
            partial_charge: atom.partial_charge,
            force_field_type: atom.force_field_type.clone(),
            hetero: atom.hetero,
            alt_conformation_id: atom.alt_conformation_id.clone(),
            ..Default::default()
        }
    }
}

impl Display for Atom {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let ff_type = match &self.force_field_type {
            Some(f) => f,
            None => "None",
        };

        let q = match &self.partial_charge {
            Some(q_) => format!("{q_:.3}"),
            None => "None".to_string(),
        };

        write!(
            f,
            "Atom {}: {}, {}. {:?}, ff: {ff_type}, q: {q}",
            self.serial_number,
            self.element.to_letter(),
            self.posit,
            self.type_in_res,
        )?;

        if self.hetero {
            write!(f, ", Het")?;
        }

        Ok(())
    }
}

/// Can't find a PyMol equiv. Experimenting
pub const fn aa_color(aa: AminoAcid) -> (f32, f32, f32) {
    match aa {
        AminoAcid::Arg => (0.7, 0.2, 0.9),
        AminoAcid::His => (0.2, 1., 0.2),
        AminoAcid::Lys => (1., 0.3, 0.3),
        AminoAcid::Asp => (0.4, 0.4, 1.0),
        AminoAcid::Glu => (0.701, 0.7, 0.2),
        AminoAcid::Ser => (177. / 255., 187. / 255., 161. / 255.),
        AminoAcid::Thr => (1.0, 0.502, 0.),
        AminoAcid::Asn => (0.878, 0.4, 0.2),
        AminoAcid::Gln => (0.784, 0.502, 0.2),
        AminoAcid::Cys => (0.239, 1.0, 0.),
        AminoAcid::Sec => (0.561, 0.251, 0.831),
        AminoAcid::Gly => (0.749, 0.651, 0.651),
        AminoAcid::Pro => (0.74, 0.6, 0.4),
        AminoAcid::Ala => (1., 0.820, 0.137),
        AminoAcid::Val => (0.753, 0.753, 0.753),
        AminoAcid::Ile => (0.322, 0.722, 0.716),
        AminoAcid::Leu => (0.5, 0.702, 0.602),
        AminoAcid::Met => (0.490, 0.502, 0.690),
        AminoAcid::Phe => (0.780, 0., 0.780),
        AminoAcid::Tyr => (0.541, 1., 0.),
        AminoAcid::Trp => (0.121, 0.941, 0.121),
    }
}

pub fn build_adjacency_list(bonds: &[Bond], atoms_len: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); atoms_len];

    // For each bond, record its atoms as neighbors of each other
    for bond in bonds {
        result[bond.atom_0].push(bond.atom_1);
        result[bond.atom_1].push(bond.atom_0);
    }

    result
}

/// A helper, shared between mmCIF parsing, and H regenerating from changed pH.
fn init_bonds_chains_res(
    atoms_: &[AtomGeneric],
    bonds_: &[BondGeneric],
    residues_: &[ResidueGeneric],
    chains_: &[ChainGeneric],
    dihedrals: &[Dihedral],
) -> io::Result<(Vec<Atom>, Vec<Bond>, Vec<Residue>, Vec<Chain>)> {
    println!("Initializing protein atoms, bonds residues, chains...");
    let start = Instant::now();

    let mut atoms: Vec<_> = atoms_.iter().map(|a| a.into()).collect();

    // Build an O(n) SN → atom-index map used throughout this function instead of repeated
    // linear scans via atom_sns_to_indices (which made all the loops below O(n²) or worse).
    let sn_to_atom: HashMap<u32, usize> = atoms
        .iter()
        .enumerate()
        .map(|(i, a): (usize, &Atom)| (a.serial_number, i))
        .collect();

    let mut bonds = Vec::with_capacity(bonds_.len());
    for bond in bonds_ {
        let atom_0 = sn_to_atom.get(&bond.atom_0_sn).copied().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Unable to find atom0 SN when loading from generic bond: {}",
                    bond.atom_0_sn
                ),
            )
        })?;
        let atom_1 = sn_to_atom.get(&bond.atom_1_sn).copied().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Unable to find atom1 SN when loading from generic bond: {}",
                    bond.atom_1_sn
                ),
            )
        })?;
        bonds.push(Bond {
            bond_type: bond.bond_type,
            atom_0_sn: bond.atom_0_sn,
            atom_1_sn: bond.atom_1_sn,
            atom_0,
            atom_1,
            is_backbone: false,
        });
    }

    let len_matches = residues_.len() == dihedrals.len();
    if !len_matches {
        eprintln!(
            "Error: Diehedral, residue len mismatch. Dihedrals: {}, residues: {}",
            dihedrals.len(),
            residues_.len()
        );
    }

    let mut residues = Vec::with_capacity(residues_.len());
    for (i, res) in residues_.iter().enumerate() {
        let atom_indices: io::Result<Vec<usize>> = res
            .atom_sns
            .iter()
            .map(|sn| {
                sn_to_atom.get(sn).copied().ok_or_else(|| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find atom SN when loading from generic res",
                    )
                })
            })
            .collect();

        let mut r = Residue {
            serial_number: res.serial_number,
            res_type: res.res_type.clone(),
            atom_sns: res.atom_sns.clone(),
            atoms: atom_indices?,
            dihedral: None,
            end: res.end,
        };

        if len_matches {
            r.dihedral = Some(dihedrals[i].clone());
        }
        residues.push(r);
    }

    // Build SN → residue-index map for O(1) chain construction.
    let sn_to_res: HashMap<u32, usize> = residues
        .iter()
        .enumerate()
        .map(|(i, r)| (r.serial_number, i))
        .collect();

    let mut chains = Vec::with_capacity(chains_.len());
    for chain in chains_ {
        let atom_indices: io::Result<Vec<usize>> = chain
            .atom_sns
            .iter()
            .map(|sn| {
                sn_to_atom.get(sn).copied().ok_or_else(|| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find atom SN when loading from generic chain",
                    )
                })
            })
            .collect();
        let residue_indices: io::Result<Vec<usize>> = if residues.is_empty() {
            Ok(Vec::new())
        } else {
            chain
                .residue_sns
                .iter()
                .map(|sn| {
                    sn_to_res.get(sn).copied().ok_or_else(|| {
                        io::Error::new(
                            ErrorKind::InvalidData,
                            "Unable to find res SN when loading from generic chain",
                        )
                    })
                })
                .collect()
        };
        chains.push(Chain {
            id: chain.id.clone(),
            residue_sns: chain.residue_sns.clone(),
            residues: residue_indices?,
            atom_sns: chain.atom_sns.clone(),
            atoms: atom_indices?,
            visible: true,
        });
    }

    // Build reverse maps: atom SN → residue index, and atom SN → chain index.
    // Replaces the previous O(atoms × residues) and O(atoms × chains) nested loops.
    let mut atom_sn_to_res: HashMap<u32, usize> = HashMap::new();
    for (res_i, res) in residues.iter().enumerate() {
        for &sn in &res.atom_sns {
            atom_sn_to_res.insert(sn, res_i);
        }
    }
    let mut atom_sn_to_chain: HashMap<u32, usize> = HashMap::new();
    for (chain_i, chain) in chains.iter().enumerate() {
        for &sn in &chain.atom_sns {
            atom_sn_to_chain.insert(sn, chain_i);
        }
    }

    for atom in &mut atoms {
        if let Some(&res_i) = atom_sn_to_res.get(&atom.serial_number) {
            atom.residue = Some(res_i);
            if residues[res_i].res_type == ResidueType::Water {
                atom.role = Some(AtomRole::Water);
            }
        }
        if let Some(&chain_i) = atom_sn_to_chain.get(&atom.serial_number) {
            atom.chain = Some(chain_i);
        }
    }

    for bond in &mut bonds {
        if atoms[bond.atom_0].is_backbone() && atoms[bond.atom_1].is_backbone() {
            bond.is_backbone = true;
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Populated protein residues etc in {elapsed} ms");

    Ok((atoms, bonds, residues, chains))
}

/// For small organic molecules.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Decode, Encode)]
pub enum MolIdent {
    /// Known as CID.
    PubChem(u32),
    DrugBank(String),
    /// PDBe, or Amber Geostd
    PdbeAmber(String),
    /// A SMILES (Simplified Molecular Input Line Entry System) string, which includes both
    /// stereochemical and isotopic information. See the glossary entry on SMILES for more detail.
    Smiles(String),
    /// Standard IUPAC International Chemical Identifier (InChI). It does not allow for user
    /// selectable options in dealing with the stereochemistry and tautomer layers of the InChI string
    InchI(String),
    /// Hashed version of the full standard InChI, consisting of 27 characters.
    InchIKey(String),
    /// Chemical name systematically determined according to the IUPAC nomenclatures
    IupacName(String),
    /// The title used for the PubChem compound summary page.
    PubchemTitle(String),
}

impl MolIdent {
    /// Useful for some APIs, for example.
    pub fn ident_inner(&self) -> String {
        match self {
            Self::PubChem(cid) => cid.to_string(),
            Self::DrugBank(v) => v.clone(),
            Self::PdbeAmber(v) => v.clone(),
            Self::Smiles(v) => v.clone(),
            Self::InchI(v) => v.clone(),
            Self::InchIKey(v) => v.clone(),
            Self::IupacName(v) => v.clone(),
            Self::PubchemTitle(v) => v.clone(),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::PubChem(_) => "PubChem CID",
            Self::DrugBank(_) => "DrugBank",
            Self::PdbeAmber(_) => "PDBe",
            Self::Smiles(_) => "SMILES",
            Self::InchI(_) => "InChI",
            Self::InchIKey(_) => "InChIKey",
            Self::IupacName(_) => "IUPAC",
            Self::PubchemTitle(_) => "PubChem Title",
        }
        .to_owned()
    }
}

impl Display for MolIdent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let v = match self {
            Self::PubChem(cid) => format!("PubChem CID: {}", cid),
            Self::DrugBank(ident) => format!("DrugBank: {ident}"),
            Self::PdbeAmber(ident) => format!("PDBe: {ident}"),
            Self::Smiles(ident) => format!("SMILES: {ident}"),
            Self::InchI(ident) => format!("InchI: {ident}"),
            Self::InchIKey(ident) => format!("InChIKey: {ident}"),
            Self::IupacName(ident) => format!("IUPAC: {ident}"),
            Self::PubchemTitle(ident) => format!("Title: {ident}"),
        };

        write!(f, "{v}")
    }
}
