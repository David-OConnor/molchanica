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
use synthesis::{
    EcTopLevel,
    broad_target::{LibraryInventory, LibraryRoute, ReactionLibrary},
};

use crate::{
    file_io::download_mols::{DownloadedSmallMol, load_sdf_chebi, load_sdf_pubchem},
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

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum ParticipantAction {
    #[default]
    OpenChebiPage,
    Download,
    OpenRheaPage,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum SynthesisParticipantAction {
    #[default]
    OpenDatabasePage,
    Download,
    SearchRhea,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum SynthesisSort {
    #[default]
    TargetAscending,
    TargetDescending,
    Class,
    FewestSteps,
    MostSteps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SynthesisDownloadId {
    Chebi(u32),
    PubChem(u32),
}

pub type SynthesisDownloads =
    HashMap<SynthesisDownloadId, Receiver<Result<DownloadedSmallMol, String>>>;

impl SynthesisDownloadId {
    pub fn label(self) -> String {
        match self {
            Self::Chebi(id) => format!("CHEBI:{id}"),
            Self::PubChem(id) => format!("CID {id}"),
        }
    }
}

pub struct SynthesisLibraryData {
    pub summary: String,
    pub routes: Vec<LibraryRoute>,
    pub classes: Vec<String>,
    /// Top-level EC classes catalyzing at least one step, sorted.
    pub ec_classes: Vec<EcTopLevel>,
    pub inventory: LibraryInventory,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BuildingBlockKind {
    #[default]
    Feedstocks,
    Enzymes,
    Cofactors,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum BuildingBlockColumn {
    #[default]
    Name,
    PubChem,
    Chebi,
    Ec,
    UniProt,
    /// The role of a feedstock or cofactor, or the reaction families of an enzyme.
    Detail,
    Routes,
}

/// Browsing state for the table of the library's feedstocks, enzymes, and cofactors.
#[derive(Default)]
pub struct BuildingBlocksView {
    pub kind: BuildingBlockKind,
    pub search: String,
    pub sort: BuildingBlockColumn,
    pub descending: bool,
}

/// A building block routes can be filtered by: an index into a `LibraryInventory` list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteComponent {
    Feedstock(usize),
    Enzyme(usize),
    Cofactor(usize),
}

impl RouteComponent {
    pub fn name(self, inventory: &LibraryInventory) -> &str {
        let name = match self {
            Self::Feedstock(i) => inventory.feedstocks.get(i).map(|f| f.name.as_str()),
            Self::Enzyme(i) => inventory
                .enzymes
                .get(i)
                .map(|e| e.enzyme.common_name.as_str()),
            Self::Cofactor(i) => inventory.cofactors.get(i).map(|c| c.name.as_str()),
        };
        name.unwrap_or_default()
    }

    /// True if the route consumes this feedstock or cofactor, or lists this enzyme for a step.
    /// Catalytic cofactors count, although they are not consumed.
    pub fn used_by(self, route: &LibraryRoute, inventory: &LibraryInventory) -> bool {
        match self {
            Self::Feedstock(i) => inventory.feedstocks.get(i).is_some_and(|feedstock| {
                route
                    .starting_materials
                    .iter()
                    .any(|molecule| molecule.name == feedstock.name)
            }),
            Self::Enzyme(i) => inventory.enzymes.get(i).is_some_and(|catalyst| {
                route
                    .steps
                    .iter()
                    .any(|step| step.enzymes.contains(&catalyst.enzyme))
            }),
            Self::Cofactor(i) => inventory.cofactors.get(i).is_some_and(|cofactor| {
                route.steps.iter().any(|step| {
                    step.reactants
                        .iter()
                        .any(|molecule| molecule.name == cofactor.name)
                        || step.catalytic_cofactors.contains(&cofactor.name)
                })
            }),
        }
    }
}

/// True if an enzyme for any step of the route belongs to this top-level EC class.
pub fn route_has_ec_class(route: &LibraryRoute, top_level: EcTopLevel) -> bool {
    route
        .steps
        .iter()
        .flat_map(|step| &step.enzymes)
        .filter_map(|enzyme| enzyme.ec)
        .any(|ec| ec.top_level == top_level)
}

pub enum SynthesisEntry {
    NotLoaded,
    Loading(Receiver<Result<SynthesisLibraryData, String>>),
    Ready(SynthesisLibraryData),
    Failed(String),
}

impl Default for SynthesisEntry {
    fn default() -> Self {
        Self::NotLoaded
    }
}

/// Synthesis routes deliberately have their own state instead of sharing Rhea queries or results.
#[derive(Default)]
pub struct SynthesisReactionsState {
    pub diagrams: crate::mol_diagrams::DiagramCache,
    pub participant_action: SynthesisParticipantAction,
    pub download_message: Option<String>,
    pub download_error: Option<String>,
    pub downloads: SynthesisDownloads,
    pub entry: SynthesisEntry,
    pub search: String,
    pub class_filter: String,
    /// None shows every route; otherwise keep routes with a step catalyzed by this EC class.
    pub enzyme_class_filter: Option<EcTopLevel>,
    /// None shows every route; otherwise keep routes using this feedstock, enzyme, or cofactor.
    pub component_filter: Option<RouteComponent>,
    pub building_blocks: BuildingBlocksView,
    /// Zero shows routes of every length; other values are exact step counts.
    pub step_filter: usize,
    pub sort: SynthesisSort,
    pub page: usize,
}

impl SynthesisReactionsState {
    /// Build the deterministic library only when the user first opens the popup.
    pub fn load(&mut self) {
        if matches!(
            self.entry,
            SynthesisEntry::Loading(_) | SynthesisEntry::Ready(_)
        ) {
            return;
        }

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let library = ReactionLibrary::new();
            let summary = library.format_coverage_summary();
            let result = library
                .routes()
                .map(|routes| {
                    let mut classes: Vec<_> =
                        routes.iter().map(|route| route.class.clone()).collect();
                    classes.sort();
                    classes.dedup();

                    let mut ec_classes: Vec<_> = routes
                        .iter()
                        .flat_map(|route| &route.steps)
                        .flat_map(|step| &step.enzymes)
                        .filter_map(|enzyme| enzyme.ec.map(|ec| ec.top_level))
                        .collect();
                    ec_classes.sort();
                    ec_classes.dedup();

                    let inventory = library.inventory();

                    SynthesisLibraryData {
                        summary,
                        routes,
                        classes,
                        ec_classes,
                        inventory,
                    }
                })
                .map_err(|error| format!("Unable to build the synthesis library: {error}"));
            let _ = tx.send(result);
        });
        self.entry = SynthesisEntry::Loading(rx);
    }

    pub fn retry(&mut self) {
        // Its indices refer to the previous library.
        self.component_filter = None;
        self.entry = SynthesisEntry::NotLoaded;
        self.load();
    }

    pub fn poll(&mut self) -> bool {
        let SynthesisEntry::Loading(rx) = &self.entry else {
            return false;
        };

        match rx.try_recv() {
            Ok(Ok(data)) => self.entry = SynthesisEntry::Ready(data),
            Ok(Err(error)) => self.entry = SynthesisEntry::Failed(error),
            Err(TryRecvError::Disconnected) => {
                self.entry = SynthesisEntry::Failed(
                    "The synthesis library stopped loading before returning a result.".to_owned(),
                );
            }
            Err(TryRecvError::Empty) => return true,
        }
        false
    }

    pub fn download(&mut self, chebi_id: Option<u32>, pubchem_id: Option<u32>) {
        let Some(id) = chebi_id
            .map(SynthesisDownloadId::Chebi)
            .or_else(|| pubchem_id.map(SynthesisDownloadId::PubChem))
        else {
            self.download_error =
                Some("This participant has no ChEBI or PubChem identifier to download.".to_owned());
            return;
        };

        if self.downloads.contains_key(&id) {
            return;
        }

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = match id {
                SynthesisDownloadId::Chebi(id) => load_sdf_chebi(id)
                    .map_err(|error| format!("Unable to download CHEBI:{id}: {error:?}")),
                SynthesisDownloadId::PubChem(id) => load_sdf_pubchem(id)
                    .map_err(|error| format!("Unable to download CID {id}: {error:?}")),
            };
            let _ = tx.send(result);
        });
        self.downloads.insert(id, rx);
        self.download_error = None;
        self.download_message = None;
    }

    pub fn take_downloads(
        &mut self,
    ) -> Vec<(SynthesisDownloadId, Result<DownloadedSmallMol, String>)> {
        let mut completed = Vec::new();
        self.downloads.retain(|&id, rx| {
            let result = match rx.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Disconnected) => Err(format!(
                    "The download of {} stopped before returning a result.",
                    id.label()
                )),
                Err(TryRecvError::Empty) => return true,
            };
            completed.push((id, result));
            false
        });
        completed
    }
}

/// Kept outside MoleculeSmall so cached API responses do not alter saved molecule formats.
/// Queries, including empty results, survive closing the popup and switching molecules.
#[derive(Default)]
pub struct ReactionsState {
    pub diagrams: crate::mol_diagrams::DiagramCache,
    pub participant_action: ParticipantAction,
    pub download_message: Option<String>,
    pub download_error: Option<String>,
    pub downloads: HashMap<u32, Receiver<Result<DownloadedSmallMol, String>>>,
    pub title: String,
    pub pubchem_cid: Option<u32>,
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

    pub fn show_chebi_participant(&mut self, id: u32, name: String) {
        self.title = name;
        self.pubchem_cid = None;
        self.page = 0;
        self.pending_ligand = None;
        self.download_message = None;
        self.download_error = None;
        self.select(Query::Chebi(id));
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
    let title = mol.name().into_owned();
    let pubchem_cid = match mol {
        MolGenericRef::Small(mol) => match mol.get_ident(MolIdentType::PubChem) {
            Some(MolIdent::PubChem(cid)) => Some(*cid),
            _ => None,
        },
        _ => None,
    };
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
    reactions.pubchem_cid = pubchem_cid;
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
    let rhea_loading = state.ui.reactions.poll();
    let synthesis_loading = state.ui.synthesis_reactions.poll();
    let loading = rhea_loading || synthesis_loading;
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

    state.ui.reactions.pubchem_cid = match mol.get_ident(MolIdentType::PubChem) {
        Some(MolIdent::PubChem(cid)) => Some(*cid),
        _ => None,
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
