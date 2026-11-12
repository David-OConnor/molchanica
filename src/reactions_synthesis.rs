//! State and background loading for the synthesis reaction library popup.

use std::{
    collections::HashMap,
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
};

use bio_apis::rhea;
use egui_file_dialog::{DialogState, FileDialog, FileDialogConfig};
use synthesis::broad_target::{
    LibraryInventory, LibraryRoute, ReactionFilters, ReactionLibrary, create_protocol_for_route,
};

use crate::file_io::download_mols::{DownloadedSmallMol, load_sdf_chebi, load_sdf_pubchem};

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

pub enum RheaIds {
    Loading(Receiver<Result<Vec<u32>, String>>),
    Ready(Vec<u32>),
    Failed(String),
}

/// Resolve synthesis search links on demand, retaining results for the session.
#[derive(Default)]
pub struct RheaLinks {
    cache: HashMap<String, RheaIds>,
}

impl RheaLinks {
    pub fn get(&mut self, query: &str) -> &RheaIds {
        self.cache.entry(query.to_owned()).or_insert_with(|| {
            let (tx, rx) = mpsc::channel();
            let query = query.to_owned();
            thread::spawn(move || {
                let result = rhea::find_ids_from_search(&query, None)
                    .map(|mut ids| {
                        ids.sort_unstable();
                        ids.dedup();
                        ids
                    })
                    .map_err(|error| format!("Unable to load Rhea reaction IDs: {error:?}"));
                let _ = tx.send(result);
            });
            RheaIds::Loading(rx)
        })
    }

    fn poll(&mut self) -> bool {
        let mut pending = false;
        for entry in self.cache.values_mut() {
            let RheaIds::Loading(rx) = entry else {
                continue;
            };

            match rx.try_recv() {
                Ok(Ok(ids)) => *entry = RheaIds::Ready(ids),
                Ok(Err(error)) => *entry = RheaIds::Failed(error),
                Err(TryRecvError::Disconnected) => {
                    *entry = RheaIds::Failed(
                        "The Rhea lookup stopped before returning reaction IDs.".to_owned(),
                    );
                }
                Err(TryRecvError::Empty) => pending = true,
            }
        }
        pending
    }
}

impl SynthesisDownloadId {
    pub fn label(self) -> String {
        match self {
            Self::Chebi(id) => format!("CHEBI:{id}"),
            Self::PubChem(id) => format!("CID {id}"),
        }
    }
}

pub struct SynthesisLibraryData {
    pub library: ReactionLibrary,
    pub summary: String,
    pub routes: Vec<LibraryRoute>,
    pub classes: Vec<String>,
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
    SigmaAldrich,
    Ec,
    UniProt,
    /// The role of a feedstock or cofactor, or the reaction families of an enzyme.
    Detail,
    Safety,
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

/// A save dialog with a snapshot of the route chosen for protocol export.
pub struct ProtocolExport {
    dialog: FileDialog,
    pending: Option<(LibraryRoute, LibraryInventory)>,
    pub message: Option<String>,
    pub error: Option<String>,
}

impl Default for ProtocolExport {
    fn default() -> Self {
        let config = FileDialogConfig::default()
            .add_file_filter_extensions("Markdown", vec!["md"])
            .add_save_extension("Markdown", "md");
        Self {
            dialog: FileDialog::with_config(config)
                .id("synthesis_protocol_save")
                .title("Save synthesis protocol")
                .default_save_extension("Markdown"),
            pending: None,
            message: None,
            error: None,
        }
    }
}

impl ProtocolExport {
    pub fn begin(&mut self, route: &LibraryRoute, inventory: &LibraryInventory) {
        let stem: String = route
            .target
            .name
            .chars()
            .map(|character| {
                if character.is_control() || r#"<>:"/\|?*"#.contains(character) {
                    '_'
                } else {
                    character
                }
            })
            .collect();
        self.dialog.config_mut().default_file_name =
            format!("protocol-{}.md", stem.trim_matches(['.', ' ']));
        self.pending = Some((route.clone(), inventory.clone()));
        self.message = None;
        self.error = None;
        self.dialog.save_file();
    }

    /// Updated with the other file dialogs even when the reactions popup is closed.
    pub fn update(&mut self, context: &egui::Context) {
        self.dialog.update(context);
        if let Some(path) = self.dialog.take_picked() {
            if let Some((route, inventory)) = self.pending.take() {
                match create_protocol_for_route(&route, &inventory, &path) {
                    Ok(()) => {
                        self.message = Some(format!("Saved protocol to {}", path.display()));
                        self.error = None;
                    }
                    Err(error) => {
                        self.error = Some(format!("Unable to save protocol: {error}"));
                        self.message = None;
                    }
                }
            }
        } else if matches!(
            self.dialog.state(),
            DialogState::Cancelled | DialogState::Closed
        ) {
            self.pending = None;
        }
    }
}

/// Synthesis routes deliberately have their own state instead of sharing Rhea queries or results.
#[derive(Default)]
pub struct SynthesisReactionsState {
    pub protocol_export: ProtocolExport,
    pub diagrams: crate::mol_diagrams::DiagramCache,
    pub rhea_links: RheaLinks,
    pub participant_action: SynthesisParticipantAction,
    pub download_message: Option<String>,
    pub download_error: Option<String>,
    pub downloads: SynthesisDownloads,
    pub entry: SynthesisEntry,
    pub search: String,
    pub class_filter: String,
    pub filters: ReactionFilters,
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

                    let inventory = library.inventory();

                    SynthesisLibraryData {
                        library,
                        summary,
                        routes,
                        classes,
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
        let rhea_pending = self.rhea_links.poll();
        let SynthesisEntry::Loading(rx) = &self.entry else {
            return rhea_pending;
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
        rhea_pending
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
