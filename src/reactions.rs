//! Session cache and background loading for the Rhea viewer.

use std::{
    collections::HashMap,
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
};

use bio_apis::{
    pdbe,
    rhea::{self, Reaction},
};
use mol_defs::molecules::{MolGenericRef, MolIdent, MolIdentType};

use crate::{
    file_io::download_mols::{DownloadedSmallMol, load_sdf_chebi},
    state::State,
    threads::start_all_idents_lookup,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Query {
    Chebi(u32),
    UniProt(Vec<String>),
    Pdb(String),
}

impl Query {
    pub fn search_links(&self) -> Vec<(String, String)> {
        match self {
            Self::Chebi(id) => vec![
                (
                    format!("CHEBI:{id}"),
                    format!("https://www.rhea-db.org/rhea?query=CHEBI%3A{id}"),
                ),
                (
                    "Exact ChEBI match".to_owned(),
                    format!("https://www.rhea-db.org/rhea?query=chebi_exact%3A{id}"),
                ),
            ],
            Self::UniProt(accessions) => accessions
                .iter()
                .map(|accession| {
                    (
                        format!("UniProt {accession}"),
                        format!("https://www.rhea-db.org/rhea?query=uniprot%3A{accession}"),
                    )
                })
                .collect(),
            Self::Pdb(_) => Vec::new(),
        }
    }
}

pub struct Results {
    pub query: Query,
    pub reactions: Vec<Reaction>,
    pub warnings: Vec<String>,
}

pub enum Entry {
    Loading(Receiver<Result<Results, String>>),
    Ready(Results),
    Failed(String),
}

struct PendingLigand {
    ident: String,
    started: bool,
}

/// Kept outside MoleculeSmall so cached API responses do not alter saved molecule formats.
/// Queries, including empty results, survive closing the popup and switching molecules.
#[derive(Default)]
pub struct ReactionsState {
    /// False opens the participant's ChEBI page; true imports its structure.
    pub download_on_click: bool,
    pub download_message: Option<String>,
    pub download_error: Option<String>,
    pub downloads: HashMap<u32, Receiver<Result<DownloadedSmallMol, String>>>,
    pub title: String,
    pub selected: Option<Query>,
    pub cache: HashMap<Query, Entry>,
    pub page: usize,
    pub message: Option<String>,
    pending_ligand: Option<PendingLigand>,
}

impl ReactionsState {
    pub fn download(&mut self, id: u32) {
        if self.downloads.contains_key(&id) {
            return;
        }

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = load_sdf_chebi(id)
                .map_err(|error| format!("Unable to download CHEBI:{id}: {error:?}"));
            let _ = tx.send(result);
        });
        self.downloads.insert(id, rx);
        self.download_error = None;
        self.download_message = None;
    }

    pub fn take_downloads(&mut self) -> Vec<(u32, Result<DownloadedSmallMol, String>)> {
        let mut completed = Vec::new();
        self.downloads.retain(|&id, rx| {
            let result = match rx.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Disconnected) => Err(format!(
                    "The download of CHEBI:{id} stopped before returning a result."
                )),
                Err(TryRecvError::Empty) => return true,
            };
            completed.push((id, result));
            false
        });
        completed
    }

    pub fn load(&mut self, query: Query) {
        if self.cache.contains_key(&query) {
            return;
        }

        let (tx, rx) = mpsc::channel();
        let requested = query.clone();
        thread::spawn(move || {
            let _ = tx.send(load_reactions(requested));
        });
        self.cache.insert(query, Entry::Loading(rx));
    }

    fn select(&mut self, query: Query) {
        self.load(query.clone());
        self.selected = Some(query);
        self.message = None;
    }

    pub fn poll(&mut self) -> bool {
        let mut pending = !self.downloads.is_empty();
        for entry in self.cache.values_mut() {
            let Entry::Loading(rx) = entry else {
                continue;
            };

            match rx.try_recv() {
                Ok(Ok(results)) => *entry = Entry::Ready(results),
                Ok(Err(error)) => *entry = Entry::Failed(error),
                Err(TryRecvError::Disconnected) => {
                    *entry = Entry::Failed(
                        "The Rhea lookup stopped before returning a result.".to_owned(),
                    );
                }
                Err(TryRecvError::Empty) => pending = true,
            }
        }
        pending
    }
}

fn load_reactions(query: Query) -> Result<Results, String> {
    let query = match query {
        Query::Pdb(ref pdb) => {
            let mappings = pdbe::load_uniprot_mappings(pdb)
                .map_err(|e| format!("Unable to load UniProt mappings for {pdb}: {e:?}"))?;
            let mut accessions: Vec<_> = mappings.into_iter().map(|m| m.accession).collect();
            accessions.sort();
            accessions.dedup();
            if accessions.is_empty() {
                return Err(format!("No UniProt mappings found for {pdb}."));
            }
            Query::UniProt(accessions)
        }
        other => other,
    };

    let mut results = Results {
        query: query.clone(),
        reactions: Vec::new(),
        warnings: Vec::new(),
    };
    match query {
        Query::Chebi(id) => {
            // Fetch all results; pagination belongs to the viewer, not a six-record API cap.
            results.reactions = rhea::reactions_from_chebi_exact(id, None)
                .map_err(|e| format!("Unable to load reactions for CHEBI:{id}: {e:?}"))?;
        }
        Query::UniProt(accessions) => {
            for accession in accessions {
                match rhea::reactions_from_uniprot(&accession, None) {
                    Ok(reactions) => results.reactions.extend(reactions),
                    Err(e) => results.warnings.push(format!("UniProt {accession}: {e:?}")),
                }
            }
        }
        Query::Pdb(_) => unreachable!(),
    }

    // A multichain protein can have multiple accessions annotating the same reaction.
    results.reactions.sort_by_key(|reaction| reaction.id);
    results.reactions.dedup_by_key(|reaction| reaction.id);
    Ok(results)
}

pub fn open_for_active(state: &mut State) {
    let Some(mol) = state.active_mol() else {
        return;
    };
    let title = mol.common().name(None).into_owned();
    let (query, pending_ligand) = match mol {
        MolGenericRef::Small(mol) => match mol.get_ident(MolIdentType::Chebi) {
            Some(MolIdent::Chebi(id)) => (Some(Query::Chebi(*id)), None),
            _ => (
                None,
                Some(PendingLigand {
                    ident: mol.common.ident.clone(),
                    started: false,
                }),
            ),
        },
        MolGenericRef::Peptide(mol) => {
            let mut accessions: Vec<_> = mol
                .sifts_mapping
                .as_ref()
                .into_iter()
                .flatten()
                .map(|mapping| mapping.accession.clone())
                .collect();
            accessions.sort();
            accessions.dedup();
            let query = if accessions.is_empty() {
                Query::Pdb(mol.common.ident.clone())
            } else {
                Query::UniProt(accessions)
            };
            (Some(query), None)
        }
        _ => return,
    };

    let reactions = &mut state.ui.reactions;
    reactions.title = title;
    reactions.page = 0;
    reactions.message = None;
    reactions.selected = None;
    reactions.pending_ligand = pending_ligand;
    if let Some(query) = query {
        reactions.select(query);
    }
    state.ui.popup.reactions = true;
}

/// Advance even with the popup closed. Keep the requested molecule's identity instead of using
/// whichever molecule happens to be active when its identifiers arrive.
pub fn poll(state: &mut State) -> bool {
    let loading = state.ui.reactions.poll();
    let Some(mut pending) = state.ui.reactions.pending_ligand.take() else {
        return loading;
    };
    let Some((index, mol)) = state
        .ligands
        .iter()
        .enumerate()
        .find(|(_, mol)| mol.common.ident == pending.ident)
    else {
        state.ui.reactions.message =
            Some("The molecule was removed before its identifiers loaded.".to_owned());
        return loading;
    };

    if let Some(MolIdent::Chebi(id)) = mol.get_ident(MolIdentType::Chebi) {
        state.ui.reactions.select(Query::Chebi(*id));
        return true;
    }

    if let Some((_, ident, _)) = &state.volatile.thread_receivers.all_idents_avail {
        // Reuse an existing lookup for this molecule, or wait for another molecule's lookup.
        pending.started |= *ident == pending.ident;
    } else if pending.started {
        state.ui.reactions.message = Some(
            "No ChEBI ID could be loaded. Check the molecule identifiers or connection, then click Reactions to retry.".to_owned(),
        );
        return loading;
    } else {
        start_all_idents_lookup(
            &mut state.volatile.thread_receivers,
            index,
            pending.ident.clone(),
            mol.idents.clone(),
        );
        pending.started = true;
    }
    state.ui.reactions.pending_ligand = Some(pending);
    true
}
