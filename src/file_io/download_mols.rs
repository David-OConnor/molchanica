//! Allows downloading PDB files from various APIs.

use std::time::Instant;

use bio_apis::{
    ReqError, amber_geostd, drugbank,
    pubchem::{self, StructureSearchNamespace},
    rcsb,
};
use bio_files::{MmCif, Mol2, Sdf, md_params::ForceFieldParams};
use graphics::{ControlScheme, EngineUpdates, Scene};

use crate::{
    drawing::EntityClass,
    file_io::{
        load_peptide,
        managed_mols::{self, ManagedMolProvider},
    },
    molecules::{
        MolGenericRefMut, MolIdent, MolType, MoleculeGeneric, peptide::MoleculePeptide,
        small::MoleculeSmall,
    },
    prefs::OpenType,
    render::set_flashlight,
    state::State,
    util::handle_err,
};

/// Download mmCIF file from the RSCB, parse into a struct.
pub fn load_cif_rcsb(ident: &str) -> Result<(MmCif, String), ReqError> {
    let cif_text = rcsb::load_cif(ident)?;

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

pub fn load_atom_coords_rcsb(
    ident: &str,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) {
    println!("Loading atom data from RCSB...");
    let start = Instant::now();

    match load_cif_rcsb(ident) {
        Ok((cif, cif_text)) => {
            let cache_path = match managed_mols::store_text(
                &state.volatile.prefs_dir,
                ManagedMolProvider::Rcsb,
                ident,
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
                    return;
                }
            };

            let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map else {
                handle_err(
                    &mut state.ui,
                    "Unable to find the peptide FF Q map in parameters; can't load the molecule"
                        .to_owned(),
                );
                return;
            };

            let mol: MoleculePeptide = match MoleculePeptide::from_mmcif(
                cif,
                ff_map,
                Some(cache_path.clone()),
                state.to_save.ph,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Problem parsing mmCif data into molecule: {e:?}");
                    return;
                }
            };

            let (loaded_ident, centroid) = load_peptide(state, scene, mol, updates);
            state.update_history(&cache_path, OpenType::Peptide, Some(loaded_ident.clone()));
            if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
                *center = centroid.into();
            }

            state.cif_pdb_raw.insert(loaded_ident, cif_text);
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Problem loading molecule from CIF: {e:?}"),
            );
            return;
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Protein loading from RCSB complete in {elapsed:.1}ms");

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
