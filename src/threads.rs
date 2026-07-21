//! Handle threads for potentially long-running calls, e.g. HTTP.

use std::sync::mpsc::{Receiver, TryRecvError};

use bio_apis::{
    ReqError,
    amber_geostd::GeostdData,
    pdbe::SiftsUniprotMapping,
    pubchem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::gromacs::GromacsOutput;
use graphics::{EngineUpdates, Scene};
use na_seq::AaIdent;

use crate::{
    gromacs::on_gromacs_md_complete,
    mol_db::DbEnrichMsg,
    molecules::{MolIdent, MolType},
    render::MESH_PEP_SOLVENT_SURFACE,
    screening::pharmacophore::PhScreeningScore,
    sfc_mesh::{MeshColors, apply_mesh_colors},
    state::{DbSel, State},
    structure_prediction::StructurePredictionOutcome,
    therapeutic::TherapeuticProperties,
    util::{RedrawFlags, handle_err, handle_success},
};

/// A running background job filling in missing PubChem titles/CIDs for a molecule DB: which DB it
/// targets, the channel from the worker, and progress counters for the UI.
pub struct DbEnrichJob {
    pub db_sel: DbSel,
    pub rx: Receiver<DbEnrichMsg>,
    /// Target rows looked up so far, and the total, for a progress display.
    pub done: usize,
    pub total: usize,
}

/// Contains receivers for threads. We use these for longer-running processes, as to
/// not block the UI. For example, computations, and HTTP calls.
#[allow(clippy::type_complexity)]
#[derive(Default)]
pub struct ThreadReceivers {
    /// Receives thread data upon an HTTP result completion.
    pub mol_pending_data_avail: Vec<(
        usize,
        Receiver<(
            Result<PdbDataResults, ReqError>,
            Result<FilesAvailable, ReqError>,
        )>,
    )>,
    /// Receives thread data upon an HTTP result completion.
    pub pubchem_properties_avail:
        Option<Receiver<(MolIdent, Result<pubchem::Properties, ReqError>)>>,
    /// The first param is the index.
    pub therapeutic_properties_avail: Option<Receiver<(usize, TherapeuticProperties)>>,
    /// The first param is the index.
    pub amber_geostd_data_avail: Option<Receiver<(usize, Result<GeostdData, ReqError>)>>,
    pub sifts_mapping_avail: Vec<(usize, Receiver<Result<Vec<SiftsUniprotMapping>, ReqError>>)>,
    pub peptide_mesh_coloring: Option<Receiver<Option<MeshColors>>>,
    /// Pharmacophore. Returned in batches, e.g. of a large directory.
    // /// This threads runs the whole outer loops, screening all molecules
    // pub ph_screening_outer: Option<Receiver<Vec<PhScreeningScore>>>,
    pub ph_screening: Option<Receiver<Vec<PhScreeningScore>>>,
    /// GROMACS MD run. Carries `(out, mol_start_indices, elapsed_ms)`.
    // pub gromacs_md_avail: Option<Receiver<(GromacsOutput, Vec<usize>, u128)>>,
    pub gromacs_md_avail: Option<Receiver<(GromacsOutput, u128)>>,
    /// Structure prediction result. The worker streams model output directly while it runs.
    pub structure_prediction: Option<Receiver<StructurePredictionOutcome>>,
    /// A background job filling in missing PubChem titles/CIDs for a molecule DB.
    pub db_pubchem_enrich: Option<DbEnrichJob>,
}

/// Poll receivers for data on potentially long-running calls. E.g. HTTP.
pub fn handle_thread_rx(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    if let Some(rx) = &mut state.volatile.thread_receivers.pubchem_properties_avail
        && let Ok((ident, http_result)) = rx.try_recv()
    {
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
            eprintln!("Unable to find the mol we requested PubChem properties for: {ident:?}");
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

    let pending_rcsb = std::mem::take(&mut state.volatile.thread_receivers.mol_pending_data_avail);
    let mut pending_rcsb_remaining = Vec::with_capacity(pending_rcsb.len());
    let mut prefs_dirty = false;
    for (peptide_i, rx) in pending_rcsb {
        let mut rx = Some(rx);
        if let Some(mol) = state.peptide.get_mut(peptide_i) {
            prefs_dirty |= mol.poll_mol_pending_data(&mut rx);
        }
        if let Some(rx) = rx {
            pending_rcsb_remaining.push((peptide_i, rx));
        }
    }
    state.volatile.thread_receivers.mol_pending_data_avail = pending_rcsb_remaining;
    if prefs_dirty {
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

    let pending_sifts = std::mem::take(&mut state.volatile.thread_receivers.sifts_mapping_avail);
    let mut pending_sifts_remaining = Vec::with_capacity(pending_sifts.len());
    for (peptide_i, rx) in pending_sifts {
        match rx.try_recv() {
            Ok(result) => {
                if let Some(pep) = state.peptide.get_mut(peptide_i) {
                    match result {
                        Ok(mappings) => {
                            println!("{} SIFTS UniProt mappings loaded", mappings.len());
                            pep.sifts_mapping = Some(mappings);
                            state.volatile.flags.update_sas_coloring = true;
                            redraw.set(MolType::Peptide);
                        }
                        Err(e) => eprintln!("Failed to load SIFTS mappings: {e:?}"),
                    }
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                pending_sifts_remaining.push((peptide_i, rx));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                eprintln!("SIFTS worker thread died before sending a result");
            }
        }
    }
    state.volatile.thread_receivers.sifts_mapping_avail = pending_sifts_remaining;

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

    let gromacs_result = state
        .volatile
        .thread_receivers
        .gromacs_md_avail
        .as_ref()
        .and_then(|rx| rx.try_recv().ok());

    // if let Some((out, mol_start_indices, elapsed_ms)) = gromacs_result {
    if let Some((out, elapsed_ms)) = gromacs_result {
        state.volatile.thread_receivers.gromacs_md_avail = None;

        // crate::gromacs::on_gromacs_md_complete(state, &out, mol_start_indices, elapsed_ms);
        on_gromacs_md_complete(state, &out, elapsed_ms);
        state.volatile.md_local.gromacs_output = Some(out);
    }

    let structure_prediction_result = state
        .volatile
        .thread_receivers
        .structure_prediction
        .as_ref()
        .map(Receiver::try_recv);
    match structure_prediction_result {
        Some(Ok(outcome)) => {
            state.volatile.thread_receivers.structure_prediction = None;
            state.ui.structure_pred.finish_prediction();

            match outcome {
                StructurePredictionOutcome::Complete(molecule) => {
                    state.volatile.aa_seq_text = molecule
                        .aa_seq
                        .iter()
                        .map(|aa| aa.to_str(AaIdent::OneLetter))
                        .collect();
                    let peptide_i = state.peptide.len();
                    state.peptide.push(molecule);
                    state.volatile.active_mol = Some((MolType::Peptide, peptide_i));
                    state.volatile.active_peptide = Some(peptide_i);
                    state.volatile.orbit_center = Some((MolType::Peptide, peptide_i));
                    state.reset_selections();
                    state.volatile.flags.ss_mesh_created = false;
                    state.volatile.flags.sas_mesh_created = false;
                    state.volatile.flags.clear_density_drawing = true;
                    state.volatile.flags.new_mol_loaded = true;
                    redraw.peptide = true;
                    handle_success(
                        &mut state.ui,
                        "Structure prediction complete; loaded predicted molecule".to_owned(),
                    );
                }
                StructurePredictionOutcome::Cancelled => {
                    handle_success(&mut state.ui, "Structure prediction cancelled".to_owned());
                }
                StructurePredictionOutcome::Failed(error) => handle_err(
                    &mut state.ui,
                    format!("Structure prediction failed: {error}"),
                ),
            }
        }
        Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
            state.volatile.thread_receivers.structure_prediction = None;
            state.ui.structure_pred.finish_prediction();
            handle_err(
                &mut state.ui,
                "Structure prediction worker stopped before returning a result".to_owned(),
            );
        }
        Some(Err(std::sync::mpsc::TryRecvError::Empty)) | None => {}
    }

    poll_db_pubchem_enrich(state);
}

/// Drain the PubChem-enrichment worker's channel: advance the progress counters, and on completion
/// swap the rewritten DB into place (invalidating the table's cached view) or report the failure.
/// Taken out of the receiver up front so the completion branch can mutate `state` freely.
fn poll_db_pubchem_enrich(state: &mut State) {
    let Some(mut job) = state.volatile.thread_receivers.db_pubchem_enrich.take() else {
        return;
    };

    // `None` while still running; otherwise the final outcome.
    let mut result: Option<Result<(DbSel, Box<crate::mol_db::ParquetMolDb>, usize), String>> = None;

    loop {
        match job.rx.try_recv() {
            Ok(DbEnrichMsg::Progress(done)) => job.done = done,
            Ok(DbEnrichMsg::Done { db, updated }) => {
                result = Some(Ok((job.db_sel, db, updated)));
                break;
            }
            Ok(DbEnrichMsg::Failed(e)) => {
                result = Some(Err(e));
                break;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                result = Some(Err("the worker stopped before finishing".to_owned()));
                break;
            }
        }
    }

    match result {
        // Still running; keep the job for the next frame.
        None => state.volatile.thread_receivers.db_pubchem_enrich = Some(job),
        Some(Ok((db_sel, db, updated))) => {
            // Match by source, not just index: the user may have closed or reordered DBs while the
            // job ran, so index `i` could now point at a different (or no) DB.
            let target = match db_sel {
                DbSel::Loaded(i) => state
                    .volatile
                    .parquet_dbs
                    .get(i)
                    .filter(|existing| existing.source == db.source)
                    .map(|_| i),
                DbSel::Common => None,
            };

            match target {
                Some(i) => {
                    state.volatile.parquet_dbs[i] = *db;
                    // The table's cached key list referred to the pre-enrichment contents.
                    state.ui.popup.parquet_db_view = Default::default();
                    handle_success(
                        &mut state.ui,
                        format!("Filled in PubChem data: {updated} molecule(s) updated."),
                    );
                }
                None => handle_err(
                    &mut state.ui,
                    "The database was closed before the PubChem lookup finished; discarding \
                     results."
                        .to_owned(),
                ),
            }
        }
        Some(Err(e)) => handle_err(&mut state.ui, format!("PubChem population failed: {e}")),
    }
}
