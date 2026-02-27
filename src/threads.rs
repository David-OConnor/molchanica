//! Handle threads for potentially long-running calls, e.g. HTTP.

use std::sync::mpsc::Receiver;

use bio_apis::{
    ReqError,
    amber_geostd::GeostdData,
    pubchem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;

use crate::{
    molecules::MolIdent,
    render::MESH_PEP_SOLVENT_SURFACE,
    sfc_mesh::{MeshColors, apply_mesh_colors},
    state::State,
    therapeutic::{TherapeuticProperties, pharmacophore::PhScreeningScore},
};

/// Contains receivers for threads. We use these for longer-running processes, as to
/// not block the UI. For example, computations, and HTTP calls.
#[derive(Default)]
pub struct ThreadReceivers {
    /// Receives thread data upon an HTTP result completion.
    pub mol_pending_data_avail: Option<
        Receiver<(
            Result<PdbDataResults, ReqError>,
            Result<FilesAvailable, ReqError>,
        )>,
    >,
    /// Receives thread data upon an HTTP result completion.
    pub pubchem_properties_avail:
        Option<Receiver<(MolIdent, Result<pubchem::Properties, ReqError>)>>,
    /// The first param is the index.
    pub therapeutic_properties_avail: Option<Receiver<(usize, TherapeuticProperties)>>,
    /// The first param is the index.
    pub amber_geostd_data_avail: Option<Receiver<(usize, Result<GeostdData, ReqError>)>>,
    pub peptide_mesh_coloring: Option<Receiver<Option<MeshColors>>>,
    /// Pharmacophore. Returned in batches, e.g. of a large directory.
    // /// This threads runs the whole outer loops, screening all molecules
    // pub ph_screening_outer: Option<Receiver<Vec<PhScreeningScore>>>,
    pub ph_screening: Option<Receiver<Vec<PhScreeningScore>>>,
}

/// Poll receivers for data on potentially long-running calls. E.g. HTTP.
pub fn handle_thread_rx(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    if let Some(rx) = &mut state.volatile.thread_receivers.pubchem_properties_avail {
        match rx.try_recv() {
            Ok((ident, http_result)) => {
                let mut mol = None;
                for mol_ in &mut state.ligands {
                    for ident_ in &mol_.idents {
                        if ident_ == &ident {
                            mol = Some(mol_);
                            break;
                        }
                    }
                }

                let Some(mol) = mol else {
                    state.volatile.thread_receivers.pubchem_properties_avail = None;
                    eprintln!(
                        "Unable to find the mol we requested PubChem properties for: {ident:?}"
                    );
                    return;
                };

                match http_result {
                    Ok(props) => {
                        println!("Received PubChem properties over HTTP.");
                        mol.update_idents_and_char_from_pubchem(&props);

                        state
                            .to_save
                            .pubchem_properties_map
                            .insert(ident.clone(), props.clone());
                    }
                    Err(e) => {
                        // Note: This is currently broken.
                        // println!("Unable to find Smiles for ident {ident_type:?}, generating one.");
                        eprintln!("Unable to find PubChem properties for ident {ident:?}: {e:?}");
                        // todo: Not saving to cache; not confident enough.
                        // mol.smiles = Some(mol.common.to_smiles());
                    }
                }
                state.volatile.thread_receivers.pubchem_properties_avail = None;
            }
            // E.g. no results. Could handle explicit errors too.
            Err(_) => {}
        }
    }

    if state
        .volatile
        .thread_receivers
        .mol_pending_data_avail
        .is_some()
        && let Some(mol) = &mut state.peptide
        && mol.poll_mol_pending_data(&mut state.volatile.thread_receivers.mol_pending_data_avail)
    {
        state.update_save_prefs();
    }

    if let Some(rx) = &mut state.volatile.thread_receivers.therapeutic_properties_avail
        && let Ok((i_mol, tp)) = rx.try_recv()
        && i_mol < state.ligands.len()
    {
        state.ligands[i_mol].therapeutic_props = Some(tp);
        state.volatile.thread_receivers.therapeutic_properties_avail = None;
    }

    if let Some(rx) = &mut state.volatile.thread_receivers.amber_geostd_data_avail
        && let Ok((i_mol, data)) = rx.try_recv()
    {
        if i_mol >= state.ligands.len() {
            eprintln!("Uhoh: Can't find a ligand we loaded Geostd data for");
            state.volatile.thread_receivers.amber_geostd_data_avail = None;
            return;
        }
        let mol = &mut state.ligands[i_mol];

        match data {
            Ok(d) => {
                mol.apply_geostd_data(d, &mut state.mol_specific_params);
            }
            Err(_) => {
                eprintln!(
                    " Unable to load GeoStd data for this molecule (Likely not in the data set.)"
                );
            }
        }
        state.volatile.thread_receivers.amber_geostd_data_avail = None;
    }

    // Poll for completed mesh coloring from thread.
    if let Some(rx) = &mut state.volatile.thread_receivers.peptide_mesh_coloring
        && let Ok(colors) = rx.try_recv()
    {
        apply_mesh_colors(&mut scene.meshes[MESH_PEP_SOLVENT_SURFACE], &colors);
        updates.meshes = true;
        state.volatile.thread_receivers.peptide_mesh_coloring = None;
    }

    if let Some(rx) = &mut state.volatile.thread_receivers.ph_screening {
        loop {
            match rx.try_recv() {
                Ok(batch) => state.pharmacophore.screening_results.extend(batch),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Thread finished; drop the receiver.
                    state.volatile.thread_receivers.ph_screening = None;
                    state.pharmacophore.screening_in_progress = false;
                    break;
                }
            }
        }
    }
}
