/// Proteins / polypeptides
use std::collections::{HashMap, HashSet};
use std::{
    io,
    io::ErrorKind,
    path::PathBuf,
    sync::{mpsc, mpsc::Receiver},
    thread,
    time::Instant,
};

use bio_apis::{
    ReqError,
    pdbe::SiftsUniprotMapping,
    rcsb,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::{BackboneSS, DensityMap, ExperimentalMethod, MmCif, ResidueType, create_bonds};
use dynamics::{
    params::{ProtFfChargeMapSet, prepare_peptide_mmcif},
    populate_hydrogens_dihedrals,
};
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, Element};

use crate::{
    bond_inference::create_hydrogen_bonds_single_mol,
    molecules,
    molecules::{Atom, AtomRole, Bond, Chain, HydrogenBond, Residue, common::MoleculeCommon},
    reflection::{DensityPt, DensityRect, ReflectionsData},
    selection::Selection,
    util::mol_center_size,
};

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
    /// For color-coding based on SIFTS (From Uniprot/PDBe)
    pub sifts_mapping: Option<Vec<SiftsUniprotMapping>>,
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

    #[allow(clippy::type_complexity)]
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
            println!("RCSB data loaded in {elapsed:.1}ms");

            let _ = tx.send((data, files_data));
        });

        *pending_data = Some(rx);
    }

    #[allow(clippy::type_complexity)]
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
        println!("Populated  protein hydrogens etc in {end:.1}ms");

        let (atoms, bonds, residues, chains) = molecules::init_bonds_chains_res(
            &m.atoms,
            &bonds_,
            &m.residues,
            &m.chains,
            &dihedrals,
        )?;

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
        let start = Instant::now();
        // Note: These don't change here, but htis function populates them anyway, so why not.
        let dihedrals =
            populate_hydrogens_dihedrals(&mut atoms_gen, &mut res_gen, &mut chains_gen, ff_map, ph)
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.descrip))?;

        let bonds_gen = create_bonds(&atoms_gen);

        let (atoms, bonds, residues, chains) = molecules::init_bonds_chains_res(
            &atoms_gen,
            &bonds_gen,
            &res_gen,
            &chains_gen,
            &dihedrals,
        )?;

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
