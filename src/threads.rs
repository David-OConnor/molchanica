//! Handle threads for potentially long-running calls, e.g. HTTP.

use std::{
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
};

use adme::TherapeuticProperties;
use bio_apis::{
    ReqError,
    amber_geostd::GeostdData,
    chebi,
    pdbe::SiftsUniprotMapping,
    pubchem::{self, StructureSearchNamespace},
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::gromacs::GromacsOutput;
use graphics::{EngineUpdates, Scene};
use mol_defs::{
    molecules::{MolIdent, MolType},
    screening::pharmacophore::PhScreeningScore,
    sfc_mesh::MeshColors,
};
use na_seq::AaIdent;

use crate::{
    file_io::managed_mols,
    gromacs::on_gromacs_md_complete,
    render::MESH_PEP_SOLVENT_SURFACE,
    sfc_mesh::apply_mesh_colors,
    state::{IntegrationsAvail, State},
    structure_prediction::StructurePredictionOutcome,
    util::{RedrawFlags, handle_err, handle_success},
};

/// Contains receivers for threads. We use these for longer-running processes, as to
/// not block the UI. For example, computations, and HTTP calls.
#[allow(clippy::type_complexity)]
#[derive(Default)]
pub struct ThreadReceivers {
    /// Availability of optional third-party tools, detected during startup.
    pub integrations_avail: Option<Receiver<IntegrationsAvail>>,
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
    /// Identifiers found for a ligand by querying PubChem and ChEBI. The tuple carries the ligand
    /// index and internal name from when the request started, so a removed/reordered ligand does
    /// not receive another molecule's result.
    pub all_idents_avail: Option<(usize, String, Receiver<IdentLookupOutcome>)>,
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
}

impl ThreadReceivers {
    /// True while any background worker still needs periodic non-blocking polling.
    pub fn has_pending(&self) -> bool {
        self.integrations_avail.is_some()
            || !self.mol_pending_data_avail.is_empty()
            || self.pubchem_properties_avail.is_some()
            || self.all_idents_avail.is_some()
            || self.therapeutic_properties_avail.is_some()
            || self.amber_geostd_data_avail.is_some()
            || !self.sifts_mapping_avail.is_empty()
            || self.peptide_mesh_coloring.is_some()
            || self.ph_screening.is_some()
            || self.gromacs_md_avail.is_some()
            || self.structure_prediction.is_some()
    }
}

/// Start detecting optional third-party integrations without delaying application startup.
pub fn start_integrations_check(receivers: &mut ThreadReceivers) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(IntegrationsAvail::detect());
    });
    receivers.integrations_avail = Some(rx);
}

/// Result of asking the online small-molecule databases to fill identifier gaps.
#[derive(Debug, Default)]
pub struct IdentLookupOutcome {
    pub idents: Vec<MolIdent>,
    pub warnings: Vec<String>,
}

/// Start the shared online identifier lookup used by the metadata popup and characterization
/// sidebar. Only one lookup is kept at a time; callers disable their buttons while it is pending.
pub fn start_all_idents_lookup(
    receivers: &mut ThreadReceivers,
    ligand_i: usize,
    ligand_ident: String,
    idents: Vec<MolIdent>,
) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let outcome = load_all_idents(&idents);
        let _ = tx.send(outcome);
    });
    receivers.all_idents_avail = Some((ligand_i, ligand_ident, rx));
}

fn has_ident_kind(idents: &[MolIdent], candidate: &MolIdent) -> bool {
    let kind = std::mem::discriminant(candidate);
    idents
        .iter()
        .any(|ident| std::mem::discriminant(ident) == kind)
}

fn push_if_kind_missing(idents: &mut Vec<MolIdent>, ident: MolIdent) {
    if !has_ident_kind(idents, &ident) {
        idents.push(ident);
    }
}

fn apply_pubchem_properties(idents: &mut Vec<MolIdent>, props: &pubchem::Properties) {
    push_if_kind_missing(idents, MolIdent::PubChem(props.cid));
    if !props.smiles.is_empty() {
        push_if_kind_missing(idents, MolIdent::Smiles(props.smiles.clone()));
    }
    if !props.inchi.is_empty() {
        push_if_kind_missing(idents, MolIdent::InchI(props.inchi.clone()));
    }
    if !props.inchi_key.is_empty() {
        push_if_kind_missing(idents, MolIdent::InchIKey(props.inchi_key.clone()));
    }
    if !props.iupac_name.is_empty() {
        push_if_kind_missing(idents, MolIdent::IupacName(props.iupac_name.clone()));
    }
    if !props.title.is_empty() {
        push_if_kind_missing(idents, MolIdent::PubchemTitle(props.title.clone()));
    }
}

fn apply_chebi_compound(idents: &mut Vec<MolIdent>, compound: &chebi::Compound) {
    push_if_kind_missing(idents, MolIdent::Chebi(compound.id));

    let props = chebi::Properties::from(compound);
    if let Some(value) = props.smiles {
        push_if_kind_missing(idents, MolIdent::Smiles(value));
    }
    if let Some(value) = props.inchi {
        push_if_kind_missing(idents, MolIdent::InchI(value));
    }
    if let Some(value) = props.inchi_key {
        push_if_kind_missing(idents, MolIdent::InchIKey(value));
    }
    if let Some(value) = props.iupac_name {
        push_if_kind_missing(idents, MolIdent::IupacName(value));
    }

    if let Some(value) = compound.xrefs_from_source("DrugBank").into_iter().next() {
        push_if_kind_missing(idents, MolIdent::DrugBank(value));
    }
    if let Some(value) = compound.xrefs_from_source("PDBeChem").into_iter().next() {
        push_if_kind_missing(idents, MolIdent::PdbeAmber(value));
    }
}

fn pubchem_properties_from_idents(idents: &[MolIdent]) -> Result<pubchem::Properties, ReqError> {
    let mut last_error = None;

    for ident in idents {
        let query = match ident {
            MolIdent::PubChem(cid) => Some((StructureSearchNamespace::Cid, cid.to_string())),
            MolIdent::InchIKey(value) => Some((StructureSearchNamespace::InchiKey, value.clone())),
            MolIdent::InchI(value) => Some((StructureSearchNamespace::Inchi, value.clone())),
            MolIdent::Smiles(value) => Some((StructureSearchNamespace::Smiles, value.clone())),
            _ => None,
        };

        if let Some((namespace, value)) = query {
            match pubchem::properties(namespace, &value) {
                Ok(props) => return Ok(props),
                Err(error) => last_error = Some(error),
            }
        }
    }

    if let Some(MolIdent::PdbeAmber(value)) = idents
        .iter()
        .find(|ident| matches!(ident, MolIdent::PdbeAmber(_)))
    {
        match pubchem::properties_from_pdbe_id(value) {
            Ok(props) => return Ok(props),
            Err(error) => last_error = Some(error),
        }
    }

    // PubChem's name namespace also accepts many registry identifiers, including DrugBank IDs.
    for text in idents.iter().filter_map(|ident| match ident {
        MolIdent::DrugBank(value) | MolIdent::IupacName(value) | MolIdent::PubchemTitle(value) => {
            Some(value.as_str())
        }
        _ => None,
    }) {
        let lookup = pubchem::find_cids_from_search(text, false).and_then(|cids| {
            let cid = cids.into_iter().next().ok_or(ReqError::Deserialize)?;
            pubchem::properties(StructureSearchNamespace::Cid, &cid.to_string())
        });
        match lookup {
            Ok(props) => return Ok(props),
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error.unwrap_or(ReqError::Deserialize))
}

fn chebi_id_from_idents(idents: &[MolIdent]) -> Result<Option<u32>, ReqError> {
    if let Some(cid) = idents.iter().find_map(|ident| match ident {
        MolIdent::PubChem(value) => Some(*value),
        _ => None,
    }) {
        if let Some(id) = pubchem::chebi_id_from_cid(cid)? {
            return Ok(Some(id));
        }
    }

    if let Some(key) = idents.iter().find_map(|ident| match ident {
        MolIdent::InchIKey(value) => Some(value.as_str()),
        _ => None,
    }) {
        let normalized = key.trim().trim_start_matches("InChIKey=");
        let result = chebi::search(normalized, 1, 15)?;
        if let Some(hit) = result.results.into_iter().find(|hit| {
            hit.data
                .inchi_key
                .as_deref()
                .map(|value| value.trim().trim_start_matches("InChIKey=") == normalized)
                .unwrap_or(false)
        }) {
            return Ok(Some(hit.id));
        }
    }

    if let Some(drugbank_id) = idents.iter().find_map(|ident| match ident {
        MolIdent::DrugBank(value) => Some(value.as_str()),
        _ => None,
    }) {
        for hit in chebi::search(drugbank_id, 1, 15)?
            .results
            .into_iter()
            .take(3)
        {
            let compound = chebi::load_compound(hit.id)?;
            if compound
                .xrefs_from_source("DrugBank")
                .iter()
                .any(|value| value.eq_ignore_ascii_case(drugbank_id))
            {
                return Ok(Some(compound.id));
            }
        }
    }

    Ok(None)
}

/// Query PubChem and ChEBI using whichever identifiers are already available, then return one
/// value for every identifier kind those databases can resolve. This performs blocking HTTP and
/// is intended to run on a worker thread.
pub fn load_all_idents(existing: &[MolIdent]) -> IdentLookupOutcome {
    let mut outcome = IdentLookupOutcome {
        idents: existing.to_vec(),
        warnings: Vec::new(),
    };

    let mut chebi_loaded = false;
    if let Some(id) = existing.iter().find_map(|ident| match ident {
        MolIdent::Chebi(value) => Some(*value),
        _ => None,
    }) {
        match chebi::load_compound(id) {
            Ok(compound) => {
                apply_chebi_compound(&mut outcome.idents, &compound);
                chebi_loaded = true;
            }
            Err(error) => outcome
                .warnings
                .push(format!("ChEBI lookup for {id} failed: {error:?}")),
        }
    }

    let mut pubchem_loaded = false;
    match pubchem_properties_from_idents(&outcome.idents) {
        Ok(props) => {
            apply_pubchem_properties(&mut outcome.idents, &props);
            pubchem_loaded = true;
        }
        Err(error) => outcome
            .warnings
            .push(format!("PubChem lookup failed: {error:?}")),
    }

    if !chebi_loaded {
        match chebi_id_from_idents(&outcome.idents) {
            Ok(Some(id)) => match chebi::load_compound(id) {
                Ok(compound) => {
                    apply_chebi_compound(&mut outcome.idents, &compound);
                    chebi_loaded = true;
                }
                Err(error) => outcome
                    .warnings
                    .push(format!("ChEBI lookup for {id} failed: {error:?}")),
            },
            Ok(None) => {}
            Err(error) => outcome
                .warnings
                .push(format!("ChEBI identifier lookup failed: {error:?}")),
        }
    }

    // Starting from ChEBI often gives us the structure identifier PubChem needs.
    if !pubchem_loaded && chebi_loaded {
        match pubchem_properties_from_idents(&outcome.idents) {
            Ok(props) => apply_pubchem_properties(&mut outcome.idents, &props),
            Err(error) => outcome
                .warnings
                .push(format!("PubChem retry failed: {error:?}")),
        }
    }

    // If PubChem was reached only on the retry, make one final attempt at its curated ChEBI link.
    if !chebi_loaded
        && let Ok(Some(id)) = chebi_id_from_idents(&outcome.idents)
        && let Ok(compound) = chebi::load_compound(id)
    {
        apply_chebi_compound(&mut outcome.idents, &compound);
    }

    outcome
}

/// Poll receivers for data on potentially long-running calls. E.g. HTTP.
pub fn handle_thread_rx(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let integrations_result = state
        .volatile
        .thread_receivers
        .integrations_avail
        .as_ref()
        .map(Receiver::try_recv);
    match integrations_result {
        Some(Ok(integrations_avail)) => {
            state.volatile.thread_receivers.integrations_avail = None;
            println!("{}", integrations_avail.descrip());
            state.volatile.integrations_avail = integrations_avail;
        }
        Some(Err(TryRecvError::Disconnected)) => {
            state.volatile.thread_receivers.integrations_avail = None;
            eprintln!("Integration detection stopped before returning a result");
        }
        Some(Err(TryRecvError::Empty)) | None => {}
    }

    let all_idents_result = state
        .volatile
        .thread_receivers
        .all_idents_avail
        .as_ref()
        .map(|(ligand_i, ligand_ident, rx)| (*ligand_i, ligand_ident.clone(), rx.try_recv()));
    match all_idents_result {
        Some((requested_i, ligand_ident, Ok(outcome))) => {
            state.volatile.thread_receivers.all_idents_avail = None;

            let ligand_i = state
                .ligands
                .get(requested_i)
                .filter(|mol| mol.common.ident == ligand_ident)
                .map(|_| requested_i)
                .or_else(|| {
                    state
                        .ligands
                        .iter()
                        .position(|mol| mol.common.ident == ligand_ident)
                });
            let Some(ligand_i) = ligand_i else {
                handle_err(
                    &mut state.ui,
                    "The molecule was removed before its identifiers finished loading".to_owned(),
                );
                return;
            };
            // Taken before the molecule is borrowed mutably below.
            let prefs_dir = state.volatile.prefs_dir.clone();

            let Some(mol) = state.ligands.get_mut(ligand_i) else {
                return;
            };

            let count_before = mol.idents.len();
            for ident in outcome.idents {
                if !mol.idents.contains(&ident) {
                    mol.idents.push(ident);
                }
            }
            let added = mol.idents.len() - count_before;

            // A downloaded molecule's source file is ours to maintain; refresh it so the
            // identifiers we just resolved are there the next time the program opens it.
            let cache_error = if added > 0 {
                managed_mols::update_managed_mol(&prefs_dir, mol)
                    .err()
                    .map(|error| error.to_string())
            } else {
                None
            };

            if let Some(error) = cache_error {
                handle_err(
                    &mut state.ui,
                    format!(
                        "Loaded identifiers, but could not update the downloaded copy: {error}"
                    ),
                );
            }

            if added > 0 {
                state.to_save.save_flag = true;
                let suffix = if outcome.warnings.is_empty() {
                    String::new()
                } else {
                    format!(" ({} online lookup(s) failed)", outcome.warnings.len())
                };
                handle_success(
                    &mut state.ui,
                    format!("Loaded {added} additional molecule identifier(s){suffix}"),
                );
            } else if outcome.warnings.is_empty() {
                handle_success(
                    &mut state.ui,
                    "No additional molecule identifiers were found".to_owned(),
                );
            } else {
                handle_err(
                    &mut state.ui,
                    format!(
                        "No additional molecule identifiers were loaded: {}",
                        outcome.warnings.join("; ")
                    ),
                );
            }
        }
        Some((_, _, Err(TryRecvError::Disconnected))) => {
            state.volatile.thread_receivers.all_idents_avail = None;
            handle_err(
                &mut state.ui,
                "The molecule identifier lookup stopped before returning a result".to_owned(),
            );
        }
        Some((_, _, Err(TryRecvError::Empty))) | None => {}
    }

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
                state.to_save.save_flag = true;
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

    let mut prefs_dirty = false;
    let mut pending_i = 0;
    while pending_i < state.volatile.thread_receivers.mol_pending_data_avail.len() {
        let outcome = {
            let (peptide_i, rx) =
                &state.volatile.thread_receivers.mol_pending_data_avail[pending_i];
            state
                .peptides
                .get_mut(*peptide_i)
                .map_or(Some(false), |mol| mol.poll_mol_pending_data(rx))
        };

        if let Some(updated) = outcome {
            prefs_dirty |= updated;
            state
                .volatile
                .thread_receivers
                .mol_pending_data_avail
                .swap_remove(pending_i);
        } else {
            pending_i += 1;
        }
    }
    if prefs_dirty {
        state.to_save.save_flag = true;
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

    let mut pending_i = 0;
    while pending_i < state.volatile.thread_receivers.sifts_mapping_avail.len() {
        let result = state.volatile.thread_receivers.sifts_mapping_avail[pending_i]
            .1
            .try_recv();
        match result {
            Ok(result) => {
                let (peptide_i, _) = state
                    .volatile
                    .thread_receivers
                    .sifts_mapping_avail
                    .swap_remove(pending_i);
                if let Some(pep) = state.peptides.get_mut(peptide_i) {
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
            Err(TryRecvError::Empty) => pending_i += 1,
            Err(TryRecvError::Disconnected) => {
                eprintln!("SIFTS worker thread died before sending a result");
                state
                    .volatile
                    .thread_receivers
                    .sifts_mapping_avail
                    .swap_remove(pending_i);
            }
        }
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
                StructurePredictionOutcome::Complete(mut molecule) => {
                    state.volatile.aa_seq_text = molecule
                        .aa_seq
                        .iter()
                        .map(|aa| aa.to_str(AaIdent::OneLetter))
                        .collect();
                    state.volatile.aa_seq_display_cache.dirty = true;
                    // Register the model's raw mmCIF under the molecule's ident so it can be saved
                    // back out as a file, mirroring how on-disk molecules populate `cif_pdb_raw`.
                    if let Some(cif) = molecule.source_cif.take() {
                        state.cif_pdb_raw.insert(molecule.common.ident.clone(), cif);
                    }
                    let peptide_i = state.peptides.len();
                    state.peptides.push(molecule);
                    state.volatile.active_mol = Some((MolType::Peptide, peptide_i));
                    state.volatile.active_peptide = Some(peptide_i);
                    state.volatile.orbit_center = Some((MolType::Peptide, peptide_i));
                    state.reset_selections();
                    state.volatile.flags.ss_mesh_created = false;
                    state.volatile.flags.sas_mesh_created = false;
                    state.volatile.flags.clear_density_drawing = true;
                    state.volatile.flags.new_mol_loaded = true;
                    redraw.peptide = true;
                    let msg = "Structure prediction complete; loaded predicted molecule".to_owned();
                    state.ui.structure_pred.mark_complete(msg.clone());
                    handle_success(&mut state.ui, msg);
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
        Some(Err(TryRecvError::Disconnected)) => {
            state.volatile.thread_receivers.structure_prediction = None;
            state.ui.structure_pred.finish_prediction();
            handle_err(
                &mut state.ui,
                "Structure prediction worker stopped before returning a result".to_owned(),
            );
        }
        Some(Err(TryRecvError::Empty)) | None => {}
    }
}
