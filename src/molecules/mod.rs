#![allow(non_camel_case_types)]

/// This module contains data structures and associated methods for various molecule types. For example:
/// Atoms, Bonds, Residues, Chains, etc. These contain application-specific data and organization that is distinct
/// from `bio_files::AtomGeneric` etc.
/// These are core to the operation of this application.
pub mod common;
pub mod lipid;
pub mod nucleic_acid;
pub mod pocket;
pub mod rotatable_bonds;
pub mod small;

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Display, Formatter},
    io,
    io::ErrorKind,
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
    time::Instant,
};

use bincode::{Decode, Encode};
// todo: TO simplify various APIs, we may need a wrapped Molecule Enum that
// todo has a variant for each molecule type.
use bio_apis::{
    ReqError, rcsb,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::{
    AtomGeneric, BackboneSS, BondGeneric, BondType, ChainGeneric, DensityMap, ExperimentalMethod,
    MmCif, Mol2, Pdbqt, ResidueEnd, ResidueGeneric, ResidueType, Sdf, Xyz, create_bonds,
};
use dynamics::{
    Dihedral,
    params::{ProtFfChargeMapSet, prepare_peptide_mmcif},
    populate_hydrogens_dihedrals,
};
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, AtomTypeInRes, Element};
use small::MoleculeSmall;

use crate::{
    bond_inference::create_hydrogen_bonds_single_mol,
    drawing::EntityClass,
    molecules::{
        common::MoleculeCommon, lipid::MoleculeLipid, nucleic_acid::MoleculeNucleicAcid,
        pocket::Pocket,
    },
    prefs::OpenType,
    reflection::{DensityPt, DensityRect, ReflectionsData},
    selection::Selection,
    util::mol_center_size,
};

// A flag indicating that a small molecule file describes a pocket, and
// should be loaded as such.
pub const POCKET_METADATA_KEY: &str = "is_pocket";
pub const POCKET_METADATA_VAL: &str = "true";

// A metadata field which contains the position of atoms
// in a pharmacophore-type small molecule's pocket
pub const PHARMACOPHORE_POCKET_ATOMS_KEY: &str = "pharmacophore_pocket_atoms";

// todo: Experimenting
pub trait MolGenericTrait {
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

/// A polypeptide molecule, e.g. a protein.
#[derive(Debug, Default, Clone)]
pub struct MoleculePeptide {
    pub common: MoleculeCommon,
    pub bonds_hydrogen: Vec<HydrogenBond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>,
    /// We currently use this for aligning ligands to CIF etc data, where they may already be included
    /// in a protein/ligand complex as hetero atoms.
    pub het_residues: Vec<Residue>,
    // /// Solvent-accessible surface. Used as one of our visualization methods.
    // /// Current structure is a Vec of rings.
    // /// Initializes to empty; updated A/R when the appropriate view is selected.
    // pub sa_surface_pts: Option<Vec<Vec<Vec3F32>>>,
    pub secondary_structure: Vec<BackboneSS>,
    /// Center and size are used for lighting, and for rotating ligands.
    pub center: Vec3,
    pub size: f32,
    /// The full (Or partial while WIP) results from the RCSB data api.
    pub rcsb_data: Option<PdbDataResults>,
    pub rcsb_files_avail: Option<FilesAvailable>,
    pub reflections_data: Option<ReflectionsData>,
    /// This is the processed collection of electron density points, ready to be mapped
    /// to entities, with some amplitude processing. It not not explicitly grid or unit-cell based,
    /// although it was likely created from unit cell data.
    /// E.g. from a MAP or MTX file directly, or processed from raw reflections data
    /// in a 2fo-fc file.
    pub elec_density: Option<Vec<DensityPt>>,
    pub density_map: Option<DensityMap>,
    pub density_rect: Option<DensityRect>, // todo: Remove?
    pub aa_seq: Vec<AminoAcid>,
    pub experimental_method: Option<ExperimentalMethod>,
    /// E.g: ["A", "B"]. Inferred from atoms.
    pub alternate_conformations: Option<Vec<String>>,
    /// Index. Ones present are displayed. Used for various UI filers like "near lig only", or "nearby sel only"
    pub atoms_filtered_to_disp: Option<Vec<usize>>,
    // pub ff_params: Option<ForceFieldParamsIndexed>,
}

impl MoleculePeptide {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        chains: Vec<Chain>,
        residues: Vec<Residue>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let (center, size) = mol_center_size(&atoms);

        let mut result = Self {
            // We create bonds only after
            common: MoleculeCommon::new(ident, atoms, bonds, metadata, path),
            chains,
            residues,
            center,
            size,
            ..Default::default()
        };

        result.aa_seq = result.get_seq();
        result.bonds_hydrogen = create_hydrogen_bonds_single_mol(
            &result.common.atoms,
            &result.common.atom_posits,
            &result.common.bonds,
        );

        // Override the one set in Common::new(), now that we've added hydrogens.
        result.common.build_adjacency_list();

        for res in &result.residues {
            if let ResidueType::Other(_) = &res.res_type
                && res.atoms.len() >= 10
            {
                result.het_residues.push(res.clone());
            }
        }

        // Ideally, alternate conformations should go here, but we place them in from_mmcif
        // so they can be added prior to Hydrogens.
        result
    }

    /// If a residue, get the alpha C. If multiple, get an arbitrary one.
    /// todo: Make this work for non-peptides.
    pub fn get_sel_atom(&self, sel: &Selection) -> Option<&Atom> {
        match sel {
            Selection::AtomPeptide(i) => self.common.atoms.get(*i),
            // Selection::AtomLig((mol_i, atom_i)) => None,
            // Selection::AtomNucleicAcid((mol_i, atom_i)) => None,
            // Selection::AtomLipid((mol_i, atom_i)) => None,
            Selection::Residue(i) => {
                let res = &self.residues[*i];
                if !res.atoms.is_empty() {
                    for atom_i in &res.atoms {
                        let atom = &self.common.atoms[*atom_i];
                        if let Some(role) = atom.role
                            && role == AtomRole::C_Alpha
                        {
                            return Some(atom);
                        }
                    }

                    // If we can't find  C alpha, default to the first atom.
                    Some(&self.common.atoms[res.atoms[0]])
                } else {
                    None
                }
            }
            Selection::AtomsPeptide(is) => {
                // todo temp?
                self.common.atoms.get(is[0])
            }
            Selection::None => None,
            _ => None, // Bonds
        }
    }

    /// Load RCSB data, and the list of (non-coordinate) files available from the PDB. We do this
    /// in a new thread, to prevent blocking the UI, or delaying a molecule's loading.
    pub fn updates_rcsb_data(
        &mut self,
        pending_data: &mut Option<
            Receiver<(
                Result<PdbDataResults, ReqError>,
                Result<FilesAvailable, ReqError>,
            )>,
        >,
    ) {
        if (self.rcsb_files_avail.is_some() && self.rcsb_data.is_some()) || pending_data.is_some() {
            return;
        }

        let ident = self.common.ident.clone(); // data the worker needs
        let (tx, rx) = mpsc::channel(); // one-shot channel

        println!("Getting RCSB auxiliary data...");

        let start = Instant::now();

        thread::spawn(move || {
            let data = rcsb::get_all_data(&ident);
            let files_data = rcsb::get_files_avail(&ident);

            let elapsed = start.elapsed().as_millis();
            println!("RCSB data loaded in  {elapsed:.1}ms");

            let _ = tx.send((data, files_data));
        });

        *pending_data = Some(rx);
    }

    /// Call this periodically from the UI/event loop; it’s non-blocking.
    /// Returns if it updated, e.g. so we can update prefs.
    pub fn poll_mol_pending_data(
        &mut self,
        pending_data_avail: &mut Option<
            Receiver<(
                Result<PdbDataResults, ReqError>,
                Result<FilesAvailable, ReqError>,
            )>,
        >,
    ) -> bool {
        if let Some(rx) = pending_data_avail {
            match rx.try_recv() {
                Ok((Ok(pdb_data), Ok(files_avail))) => {
                    self.rcsb_data = Some(pdb_data);
                    self.rcsb_files_avail = Some(files_avail);

                    // todo: Save state here, but need to get a proper signal with &mut State available.

                    *pending_data_avail = None;
                    return true;
                }

                // PdbDataResults failed, but FilesAvailable might not have been sent:
                Ok((Err(e), _)) => {
                    eprintln!("Failed to fetch PDB data for {}: {e:?}", self.common.ident);
                    *pending_data_avail = None;
                }

                // FilesAvailable failed (even if PdbDataResults succeeded):
                Ok((_, Err(e))) => {
                    eprintln!("Failed to fetch file‐list for {}: {e:?}", self.common.ident);
                    *pending_data_avail = None;
                }

                // the worker hasn’t sent anything yet:
                Err(mpsc::TryRecvError::Empty) => {
                    // still pending; do nothing this frame
                }

                // the sender hung up before sending:
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Worker thread died before sending result");
                    *pending_data_avail = None;
                }
            }
        }
        false
    }

    /// Get the amino acid sequence from the currently opened molecule, if applicable.
    fn get_seq(&self) -> Vec<AminoAcid> {
        // todo: If not a polypeptide, should we return an error, or empty vec?
        let mut result = Vec::new();

        // todo This is fragile, I believe.
        for res in &self.residues {
            if let ResidueType::AminoAcid(aa) = res.res_type {
                result.push(aa);
            }
        }

        result
    }
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        let role = match &atom.type_in_res {
            Some(tir) => Some(AtomRole::from_type_in_res(tir)),
            None => None,
        };

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

impl MoleculePeptide {
    pub fn from_mmcif(
        mut m: MmCif,
        ff_map: &ProtFfChargeMapSet,
        path: Option<PathBuf>,
        ph: f32,
    ) -> Result<Self, io::Error> {
        // Add hydrogens, FF types, partial charge, and bonds.
        // Sort out alternate conformations prior to adding hydrogens.
        let mut alternate_conformations: Vec<String> = Vec::new();
        for atom in &mut m.atoms {
            if let Some(alt) = &atom.alt_conformation_id
                && !alternate_conformations.contains(alt)
            {
                alternate_conformations.push(alt.to_owned());
            }
        }

        // todo: Handle alternate conformations!
        // todo: For now, we force the first one. This is crude, and ignores alt conformations.
        if !alternate_conformations.is_empty() {
            let mut atoms_ = Vec::new();

            for atom in &m.atoms {
                if let Some(alt) = &atom.alt_conformation_id {
                    if alt == &alternate_conformations[0] {
                        atoms_.push(atom.clone());
                    } else {
                        for res in &mut m.residues {
                            res.atom_sns.retain(|sn| *sn != atom.serial_number);
                        }
                        for chain in &mut m.chains {
                            chain.atom_sns.retain(|sn| *sn != atom.serial_number);
                        }
                    }
                } else {
                    atoms_.push(atom.clone());
                }
            }

            m.atoms = atoms_;
        }
        for a in &m.atoms {
            if !a.hetero && a.serial_number < 200 {
                // println!("A: {a:?}");
            }
        }

        // if !alternate_conformations.is_empty() {
        //     result.alternate_conformations = Some(alternate_conformations);
        // }

        println!("Populating protein hydrogens, dihedral angles, FF types and partial charges...");
        let start = Instant::now();

        let (bonds_, dihedrals) = prepare_peptide_mmcif(&mut m, ff_map, ph).unwrap_or_else(|e| {
            eprintln!("Error: Unable to prepare a mmCIF file. Maybe it's not a protein? {e:?}");
            // Populate bonds directly in case of an error:
            let bonds = create_bonds(&m.atoms);
            (bonds, Vec::new())
        });

        // todo: Speed this up?
        let end = start.elapsed().as_millis();
        println!("Populated in {end:.1}ms");

        let (atoms, bonds, residues, chains) =
            init_bonds_chains_res(&m.atoms, &bonds_, &m.residues, &m.chains, &dihedrals)?;

        let mut result = Self::new(
            m.ident.clone(),
            atoms,
            bonds,
            chains,
            residues,
            m.metadata,
            path,
        );

        result.experimental_method = m.experimental_method;
        result.secondary_structure = m.secondary_structure.clone();

        if !alternate_conformations.is_empty() {
            result.alternate_conformations = Some(alternate_conformations);
        }

        Ok(result)
    }

    /// E.g. run this when pH changes. Removes all hydrogens, and re-adds per the pH. Rebuilds
    /// bonds.
    pub fn reassign_hydrogens(&mut self, ph: f32, ff_map: &ProtFfChargeMapSet) -> io::Result<()> {
        let non_h_sns: HashSet<u32> = self
            .common
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
            .map(|a| a.serial_number)
            .collect();

        let mut atoms_gen = self
            .common
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
            .map(|a| a.to_generic())
            .collect();

        println!("Reassigning H on protein at pH {ph:.1}");

        // Strip old H serial numbers from residues and chains so that
        // populate_hydrogens_dihedrals only appends fresh H SNs. Without
        // this, the stale H SNs remain in atom_sns and Residue::from_generic
        // fails to find them in the (H-filtered) atoms list.
        let mut res_gen: Vec<_> = self
            .residues
            .iter()
            .map(|r| {
                let mut rg = r.to_generic();
                rg.atom_sns.retain(|sn| non_h_sns.contains(sn));
                rg
            })
            .collect();

        let mut chains_gen: Vec<_> = self
            .chains
            .iter()
            .map(|c| {
                let mut cg = c.to_generic();
                cg.atom_sns.retain(|sn| non_h_sns.contains(sn));
                cg
            })
            .collect();

        println!("Populating Hydrogens and dihedral angles...");
        let mut start = Instant::now();
        // Note: These don't change here, but htis function populates them anyway, so why not.
        let dihedrals =
            populate_hydrogens_dihedrals(&mut atoms_gen, &mut res_gen, &mut chains_gen, ff_map, ph)
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.descrip))?;

        let bonds_gen = create_bonds(&atoms_gen);

        let (atoms, bonds, residues, chains) =
            init_bonds_chains_res(&atoms_gen, &bonds_gen, &res_gen, &chains_gen, &dihedrals)?;

        self.common.atoms = atoms;
        self.common.bonds = bonds;
        self.residues = residues;
        self.chains = chains;

        self.common.build_adjacency_list();
        self.common.reset_posits();

        let elapsed = start.elapsed().as_millis();
        let h_count = self
            .common
            .atoms
            .iter()
            .filter(|a| a.element == Element::Hydrogen)
            .count();

        println!("{h_count} Hydrogens populated in {elapsed:.1} ms");

        Ok(())
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
    let mut atoms: Vec<_> = atoms_.iter().map(|a| a.into()).collect();

    let mut bonds = Vec::with_capacity(bonds_.len());
    for bond in bonds_ {
        bonds.push(Bond::from_generic(bond, &atoms)?);
    }

    let mut residues = Vec::with_capacity(residues_.len());

    let len_matches = residues_.len() == dihedrals.len();
    if !len_matches {
        eprintln!(
            "Error: Diehedral, residue len mismatch. Dihedrals: {}, residues: {}",
            dihedrals.len(),
            residues_.len()
        );
    }

    for (i, res) in residues_.iter().enumerate() {
        let mut res = Residue::from_generic(res, &atoms)?;
        if len_matches {
            res.dihedral = Some(dihedrals[i].clone());
        }
        residues.push(res);
    }

    let mut chains = Vec::with_capacity(chains_.len());
    for c in chains_ {
        chains.push(Chain::from_generic(c, &atoms, &residues)?);
    }

    // Now that chains and residues are loaded, update atoms with their back-ref index.
    for atom in &mut atoms {
        for (i, res) in residues.iter().enumerate() {
            if res.atom_sns.contains(&atom.serial_number) {
                atom.residue = Some(i);

                // Update which atoms are waters.
                if residues[i].res_type == ResidueType::Water {
                    atom.role = Some(AtomRole::Water);
                }

                break;
            }
        }

        for (i, chain) in chains.iter().enumerate() {
            if chain.atom_sns.contains(&atom.serial_number) {
                atom.chain = Some(i);
                break;
            }
        }
    }

    for bond in &mut bonds {
        if atoms[bond.atom_0].is_backbone() && atoms[bond.atom_1].is_backbone() {
            bond.is_backbone = true;
        }
    }

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
