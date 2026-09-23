//! Allows downloading PDB files from various APIs.

use std::time::Instant;

use bio_apis::{
    ReqError, amber_geostd, chebi, drugbank, pdbe,
    pubchem::{self, StructureSearchNamespace},
    rcsb, uniprot,
};
use bio_files::{MmCif, Mol2, Sdf, md_params::ForceFieldParams};
use graphics::{ControlScheme, EngineUpdates, Scene};
use mol_defs::molecules::{
    MolGenericRefMut, MolIdent, MolType, MoleculeGeneric, peptide::MoleculePeptide,
    small::MoleculeSmall,
};

use crate::{
    drawing::EntityClass,
    file_io::{
        load_peptide,
        managed_mols::{self, ManagedMolProvider},
    },
    prefs::OpenType,
    render::set_flashlight,
    state::State,
    util::handle_err,
};

/// Where a protein's mmCIF file is downloaded from.
#[derive(Clone, Copy, PartialEq)]
pub enum CifSource {
    Rcsb,
    /// Serves the same entries as RCSB, by the same PDB ID.
    Pdbe,
    /// Predicted structures, by UniProt accession.
    AlphaFold,
}

impl CifSource {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rcsb => "RCSB",
            Self::Pdbe => "PDBe",
            Self::AlphaFold => "AlphaFold DB",
        }
    }

    fn provider(self) -> ManagedMolProvider {
        match self {
            Self::Rcsb => ManagedMolProvider::Rcsb,
            Self::Pdbe => ManagedMolProvider::Pdbe,
            Self::AlphaFold => ManagedMolProvider::AlphaFold,
        }
    }

    fn download(self, ident: &str) -> Result<String, ReqError> {
        match self {
            Self::Rcsb => rcsb::load_cif(ident),
            Self::Pdbe => pdbe::load_cif(ident),
            Self::AlphaFold => uniprot::load_alphafold_cif(ident),
        }
    }
}

/// Download an mmCIF file, and parse it into a struct.
pub fn load_cif(source: CifSource, ident: &str) -> Result<(MmCif, String), ReqError> {
    let cif_text = source.download(ident)?;

    let mmcif = MmCif::new(&cif_text).map_err(|e| {
        eprintln!("Error parsing mmCIF file: {e}");
        e
    });

    Ok((mmcif?, cif_text))
}

#[derive(Debug)]
pub struct DownloadedSmallMol {
    pub mol: MoleculeSmall,
    pub source_text: String,
}

/// Download an SDF file from DrugBank, retaining its source text for session persistence.
pub fn load_sdf_drugbank(ident: &str) -> Result<DownloadedSmallMol, ReqError> {
    let source_text = drugbank::load_sdf(ident)?;
    let sdf = Sdf::new(&source_text).map_err(ReqError::from)?;
    let mol = sdf.try_into().map_err(ReqError::from)?;
    Ok(DownloadedSmallMol { mol, source_text })
}

/// Download an SDF file from PubChem, retaining its source text for session persistence.
pub fn load_sdf_pubchem(cid: u32) -> Result<DownloadedSmallMol, ReqError> {
    let source_text = pubchem::load_sdf(StructureSearchNamespace::Cid, &cid.to_string())?;
    let sdf = Sdf::new(&source_text).map_err(ReqError::from)?;
    let mol = sdf.try_into().map_err(ReqError::from)?;
    Ok(DownloadedSmallMol { mol, source_text })
}

/// Download a structure from ChEBI as SDF, retaining its source text for session persistence. Note
/// that ChEBI's structures are 2D; they carry no z coordinates.
///
/// ChEBI serves bare Molfiles, with no data fields, so the accession is attached as an ident here
/// rather than being picked up from the file the way PubChem's CID is.
pub fn load_sdf_chebi(id: u32) -> Result<DownloadedSmallMol, ReqError> {
    let source_text = chebi::load_sdf(id)?;
    let sdf = Sdf::new(&source_text).map_err(ReqError::from)?;
    let mut mol: MoleculeSmall = sdf.try_into().map_err(ReqError::from)?;
    mol.idents.push(MolIdent::Chebi(id));

    Ok(DownloadedSmallMol { mol, source_text })
}

/// Download a chemical component, e.g. a ligand like `ATP`, from PDBe as SDF. This is the "ideal"
/// 3D conformer, with hydrogens.
///
/// PDBe's file has no data fields, so the component ID is attached as an ident here.
pub fn load_sdf_pdbe(ident: &str) -> Result<DownloadedSmallMol, ReqError> {
    let ident = ident.trim().to_uppercase();

    let source_text = pdbe::load_sdf(&ident)?;
    let sdf = Sdf::new(&source_text).map_err(ReqError::from)?;
    let mut mol: MoleculeSmall = sdf.try_into().map_err(ReqError::from)?;
    mol.idents.push(MolIdent::PdbeAmber(ident));

    Ok(DownloadedSmallMol { mol, source_text })
}

pub fn load_atom_coords_rcsb(
    ident: &str,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) {
    load_atom_coords(
        CifSource::Rcsb,
        ident,
        state,
        scene,
        updates,
        redraw,
        reset_cam,
    );
}

/// Download a protein's mmCIF file, and open it. `ident` is a PDB ID for RCSB and PDBe, and a
/// UniProt accession for AlphaFold DB. Returns `true` if the protein loaded; errors are reported to
/// the UI here.
pub fn load_atom_coords(
    source: CifSource,
    ident: &str,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) -> bool {
    let source_name = source.name();
    println!("Loading atom data from {source_name}...");
    let start = Instant::now();

    match load_cif(source, ident) {
        Ok((cif, cif_text)) => {
            // Key the cache on the entry ID the file itself reports, not on the query text: RCSB
            // serves the same structure for the bare 4-character ident and the 12-character
            // `pdb_`-prefixed one, so keying on what was typed would cache it twice. The query is
            // still recorded in the manifest.
            let cache_key = match cif.ident.is_empty() {
                true => ident,
                false => &cif.ident,
            };

            let cache_path = match managed_mols::store_text(
                &state.volatile.prefs_dir,
                source.provider(),
                cache_key,
                ident,
                "cif",
                &cif_text,
            ) {
                Ok(path) => path,
                Err(error) => {
                    handle_err(
                        &mut state.ui,
                        format!("Downloaded {ident} but could not cache it: {error}"),
                    );
                    return false;
                }
            };

            let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map else {
                handle_err(
                    &mut state.ui,
                    "Unable to find the peptide FF Q map in parameters; can't load the molecule"
                        .to_owned(),
                );
                return false;
            };

            let mut mol: MoleculePeptide = match MoleculePeptide::from_mmcif(
                cif,
                ff_map,
                Some(cache_path.clone()),
                state.to_save.ph,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Problem parsing mmCif data into molecule: {e:?}");
                    return false;
                }
            };
            if let Err(e) = mol.set_source_cif(cif_text) {
                eprintln!("Problem loading mmCIF component bonds: {e}");
                return false;
            }

            let (loaded_ident, centroid) = load_peptide(state, scene, mol, updates, true);
            state.update_history(&cache_path, OpenType::Peptide, Some(loaded_ident.clone()));
            if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
                *center = centroid.into();
            }
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Problem loading {ident} from {source_name}: {e:?}"),
            );
            return false;
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Protein loading from {source_name} complete in {elapsed:.1}ms");

    state.update_from_prefs();
    state.update_save_prefs();

    updates.entities.push_class(EntityClass::Protein as u32);

    let peptide_i = state.peptides.len() - 1;
    state.volatile.active_mol = Some((MolType::Peptide, peptide_i));
    state.volatile.active_peptide = Some(peptide_i);
    state.volatile.orbit_center = Some((MolType::Peptide, peptide_i));

    *redraw = true;
    *reset_cam = true;
    set_flashlight(scene);
    updates.lighting = true;

    // Predicted structures have no RCSB entry to fetch data for.
    if source == CifSource::AlphaFold {
        return true;
    }

    // todo: async
    // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
    let mut pending_data = None;
    state
        .peptides
        .last_mut()
        .unwrap()
        .updates_rcsb_data(&mut pending_data);
    if let Some(rx) = pending_data {
        state
            .volatile
            .thread_receivers
            .mol_pending_data_avail
            .push((peptide_i, rx));
    }

    true
}

// todo: DIff between this and the non-2 variant?
pub fn load_geostd2(
    state: &mut State,
    scene: &mut Scene,
    ident: &str,
    load_mol2: bool,
    load_frcmod: bool,
    engine_updates: &mut EngineUpdates,
) {
    match amber_geostd::load_mol_files(ident) {
        Ok(data) => {
            let cache_path = if load_mol2 {
                match managed_mols::store_geostd(
                    &state.volatile.prefs_dir,
                    ident,
                    &data.mol2,
                    data.pubchem_cid,
                    data.frcmod.as_deref().filter(|_| load_frcmod),
                    data.lib.as_deref(),
                ) {
                    Ok(path) => Some(path),
                    Err(error) => {
                        handle_err(
                            &mut state.ui,
                            format!("Downloaded GeoStd {ident} but could not cache it: {error}"),
                        );
                        return;
                    }
                }
            } else {
                None
            };

            // Load FRCmod first, then the Ligand constructor will populate that it loaded.
            if load_frcmod && let Some(frcmod) = data.frcmod.as_deref() {
                match ForceFieldParams::from_frcmod(frcmod) {
                    Ok(v) => {
                        state.mol_specific_params.insert(ident.to_uppercase(), v);
                    }
                    Err(e) => {
                        handle_err(&mut state.ui, format!("FRCmod empty from geostd: {e:?}"));
                    }
                }
                if let Some(lig) = state.active_mol_mut()
                    && let MolGenericRefMut::Small(l) = lig
                {
                    l.frcmod_loaded = true;
                }
            }

            if data.lib.is_some() {
                println!("todo: Lib data available from geostd; download?");
            }

            if load_mol2 {
                match Mol2::new(&data.mol2) {
                    Ok(mol2) => {
                        let mut mol: MoleculeSmall = mol2.try_into().unwrap();
                        mol.idents.push(MolIdent::PdbeAmber(ident.to_owned()));
                        if let Some(cid) = data.pubchem_cid {
                            mol.idents.push(MolIdent::PubChem(cid));
                        }

                        state.load_mol_to_state(
                            MoleculeGeneric::Small(mol),
                            scene,
                            engine_updates,
                            cache_path.as_deref(),
                        );
                    }
                    Err(e) => handle_err(
                        &mut state.ui,
                        format!("Unable to make a Mol2 from Geostd data: {:?}", e),
                    ),
                }
            }
        }
        Err(_) => handle_err(
            &mut state.ui,
            "Unable to load Amber Geostd data (Server or internet problem?".to_owned(),
        ),
    }
}
