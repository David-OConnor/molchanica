//! The windows for every tool run through `bio_tools`' shared adapter: a form built from the tool's
//! contract, a runner, and a browser over each run's retained results.
//!
//! Structure prediction, sequence design, and backbone design are the same window over different
//! groups of tools; see [`ToolWindowKind`]. Nothing here is specific to one tool — what differs
//! between tools comes from the contract `bio_tools` publishes for each.

use std::{
    collections::{HashMap, hash_map::Entry},
    fs, io,
    io::Read,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bio_tools::tool_definitions::catalog::DataCategory;
use egui::{
    Button, CollapsingHeader, Color32, ComboBox, DragValue, RichText, ScrollArea, TextEdit, Ui,
};
use egui_file_dialog::FileDialog;
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::peptide::MoleculePeptide;
use serde_json::{Map, Value, json};

use crate::{
    external_tools::{
        self, RunControl, Tool,
        pdb_write::{PdbWriteOptions, peptide_to_pdb},
        shared_adapter::{self, AdapterResult},
        tool_form::{FieldKind, FormContract, FormField, Preset},
    },
    state::State,
    ui::{COLOR_ACTION, util::open_dir},
    util::handle_err,
};

/// Outputs larger than this are not previewed inline.
const MAX_PREVIEW_BYTES: u64 = 2_000_000;

// ---------------------------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------------------------

/// One of the shared-adapter tool windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ToolWindowKind {
    StructurePrediction,
    SequenceDesign,
    BackboneDesign,
}

impl ToolWindowKind {
    fn heading(self) -> &'static str {
        match self {
            Self::StructurePrediction => "Structure prediction and co-folding",
            Self::SequenceDesign => "Sequence design and scoring",
            Self::BackboneDesign => "RFdiffusion3 backbone generation",
        }
    }

    /// The tools this window offers. The first is selected when the window is opened.
    fn tools(self) -> &'static [Tool] {
        match self {
            Self::StructurePrediction => &[
                Tool::OpenDde,
                Tool::Boltz2,
                Tool::Chai1,
                Tool::Protenix,
                Tool::EsmFold2,
            ],
            Self::SequenceDesign => &[Tool::ProteinMpnn, Tool::LigandMpnn],
            Self::BackboneDesign => &[Tool::RfDiffusion3],
        }
    }
}

/// State for every shared-adapter tool window.
pub(crate) struct ToolWindows {
    structure_prediction: ToolWindow,
    sequence_design: ToolWindow,
    backbone_design: ToolWindow,
}

impl Default for ToolWindows {
    fn default() -> Self {
        Self {
            structure_prediction: ToolWindow::new(ToolWindowKind::StructurePrediction),
            sequence_design: ToolWindow::new(ToolWindowKind::SequenceDesign),
            backbone_design: ToolWindow::new(ToolWindowKind::BackboneDesign),
        }
    }
}

/// Draw one tool window's contents, loading any structure the user picks from its results.
pub(in crate::ui) fn tool_window(
    state: &mut State,
    kind: ToolWindowKind,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    let windows = &mut state.ui.tool_windows;
    let window = match kind {
        ToolWindowKind::StructurePrediction => &mut windows.structure_prediction,
        ToolWindowKind::SequenceDesign => &mut windows.sequence_design,
        ToolWindowKind::BackboneDesign => &mut windows.backbone_design,
    };

    ui.heading(kind.heading());

    let load = ui
        .push_id(kind.heading(), |ui| {
            window.draw(kind.tools(), &state.peptides, ui)
        })
        .inner;

    if let Some(path) = load
        && let Err(error) = state.open_file(&path, scene, updates)
    {
        handle_err(&mut state.ui, error.to_string());
    }
}

// ---------------------------------------------------------------------------------------------
// Form
// ---------------------------------------------------------------------------------------------

/// One tool's contract, presets, and the values currently entered.
struct Form {
    contract: FormContract,
    presets: Vec<Preset>,
    values: HashMap<String, String>,
    mode: String,
    authoritative_mode: String,
    projection_error: Option<String>,
    preset: Option<usize>,
}

impl Form {
    fn new(tool: Tool) -> Result<Self, String> {
        let slug = shared_adapter::slug(tool);
        let contract = FormContract::load(slug)?;

        Ok(Self {
            mode: contract.default_mode(),
            authoritative_mode: contract.default_mode(),
            projection_error: None,
            values: contract.defaults(),
            presets: Preset::load_all(slug)?,
            contract,
            preset: None,
        })
    }

    fn apply_preset(&mut self, tool: Tool, index: usize) {
        let working = self.mode.clone();
        self.projection_error = None;
        self.values = self.contract.defaults();
        self.values.extend(self.presets[index].form_values());
        self.authoritative_mode = self
            .values
            .get("input_mode")
            .cloned()
            .unwrap_or_else(|| self.contract.default_mode());
        self.preset = Some(index);
        if working == "upload" || shared_adapter::slug(tool) != "rfd3" {
            self.mode = self.authoritative_mode.clone();
        } else {
            self.mode = working;
            self.project_mode();
        }
    }

    fn project_mode(&mut self) {
        self.projection_error = None;
        if self.mode == self.authoritative_mode || self.mode == "upload" {
            return;
        }
        let result = match (self.authoritative_mode.as_str(), self.mode.as_str()) {
            ("parameters", "text") => {
                let preset = self.preset.and_then(|index| self.presets.get(index));
                rfd3_document(&self.contract, &mut self.values, preset).map(|document| {
                    self.values.insert("inputs".into(), document);
                })
            }
            ("text", "parameters") => rfd3_parameters(&self.contract, &mut self.values),
            _ => Ok(()),
        };
        if let Err(error) = result {
            self.projection_error = Some(error);
        }
    }
}

fn rfd3_document(
    contract: &FormContract,
    values: &mut HashMap<String, String>,
    preset: Option<&Preset>,
) -> Result<String, String> {
    let name = values
        .get("job_name")
        .filter(|name| !name.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| "design".into());
    let previous =
        values
            .get("inputs")
            .and_then(|document| match serde_json::from_str::<Value>(document) {
                Ok(parsed) => Some(parsed),
                Err(_) => serde_yaml::from_str(document).ok(),
            });
    let explicit: Vec<String> = previous
        .as_ref()
        .and_then(|document| document.get(&name))
        .and_then(Value::as_object)
        .map(|design| design.keys().cloned().collect())
        .unwrap_or_else(|| {
            preset
                .map(|preset| preset.form_values().into_keys().collect())
                .unwrap_or_default()
        });
    let mut design = Map::new();
    for field in &contract.fields {
        if !field
            .input_modes
            .split(',')
            .any(|mode| mode == "parameters")
        {
            continue;
        }
        let owned = values.get(&field.name).cloned().unwrap_or_default();
        let value = owned.trim();
        if value.is_empty()
            || (value == field.default_text()
                && field.name != "length"
                && !explicit.contains(&field.name))
        {
            continue;
        }
        let parsed = if field.name == "input" {
            if let Some(asset) = value.strip_prefix("bio-tools://rfd3/") {
                Value::String(format!("../{asset}"))
            } else {
                values.insert("spec_input_file".into(), value.to_owned());
                Value::String("uploaded".into())
            }
        } else if field.name == "ori_token" && !value.starts_with('[') {
            let coordinates: Result<Vec<f64>, _> = value
                .split(',')
                .map(|part| part.trim().parse::<f64>())
                .collect();
            let coordinates =
                coordinates.map_err(|_| "ori_token needs three numbers".to_owned())?;
            if coordinates.len() != 3 || coordinates.iter().any(|number| !number.is_finite()) {
                return Err("ori_token needs three finite numbers".into());
            }
            json!(coordinates)
        } else if matches!(field.kind(), FieldKind::Checkbox)
            || matches!(field.name.as_str(), "is_non_loopy")
        {
            Value::Bool(matches!(value, "true" | "True" | "1" | "on"))
        } else if field.name == "length" && value == "null" {
            Value::Null
        } else if matches!(field.kind(), FieldKind::Number)
            || (field.name == "length" && value.parse::<u64>().is_ok())
            || field.name == "dialect"
        {
            serde_json::from_str(value)
                .map_err(|_| format!("{} is not a valid number", field.label))?
        } else if value.starts_with('{') || value.starts_with('[') {
            serde_json::from_str(value).map_err(|_| format!("{} is not valid JSON", field.label))?
        } else {
            Value::String(value.to_owned())
        };
        design.insert(field.name.clone(), parsed);
    }
    let document = json!({name: design});
    serde_json::to_string_pretty(&document).map_err(|error| error.to_string())
}

fn rfd3_parameters(
    contract: &FormContract,
    values: &mut HashMap<String, String>,
) -> Result<(), String> {
    let document = values.get("inputs").map(String::as_str).unwrap_or("");
    let parsed: Value = match serde_json::from_str(document) {
        Ok(parsed) => parsed,
        Err(_) => serde_yaml::from_str(document).map_err(|error| {
            format!("The RFDiffusion3 input is not valid JSON or YAML: {error}")
        })?,
    };
    let designs = parsed
        .as_object()
        .ok_or("The RFDiffusion3 input must be an object of named designs")?;
    if designs.len() != 1 {
        return Err(
            "Set parameters here describes one design. Keep multiple designs in the text mode."
                .into(),
        );
    }
    let (name, entry) = designs.iter().next().unwrap();
    let entry = entry
        .as_object()
        .ok_or("The named design must be an object")?;
    let fields: Vec<&FormField> = contract
        .fields
        .iter()
        .filter(|field| {
            field
                .input_modes
                .split(',')
                .any(|mode| mode == "parameters")
        })
        .collect();
    for key in entry.keys() {
        if !fields.iter().any(|field| &field.name == key) {
            return Err(format!("Set parameters here has no field for {key}"));
        }
    }
    let mut updated: HashMap<String, String> = fields
        .iter()
        .map(|field| (field.name.clone(), field.default_text()))
        .collect();
    for (key, value) in entry {
        let text = if key == "input" {
            let source = value.as_str().ok_or("input must be a structure path")?;
            if let Some(asset) = source.strip_prefix("../input_pdbs/") {
                let reference = format!("bio-tools://rfd3/input_pdbs/{asset}");
                bio_tools::tool_definitions::presets::input_text("rfd3", &reference)
                    .map_err(|error| error.to_string())?;
                reference
            } else if source == "uploaded" {
                values
                    .get("spec_input_file")
                    .filter(|path| !path.is_empty())
                    .cloned()
                    .ok_or("Choose the structure file before switching to parameters")?
            } else if source.starts_with("bio-tools://rfd3/") {
                bio_tools::tool_definitions::presets::input_text("rfd3", source)
                    .map_err(|error| error.to_string())?;
                source.to_owned()
            } else {
                return Err("Only a bundled example structure can be carried to parameters".into());
            }
        } else {
            match value {
                Value::Null if key == "length" => "null".into(),
                Value::Null => String::new(),
                Value::String(text) => text.clone(),
                Value::Array(_) | Value::Object(_) => {
                    serde_json::to_string_pretty(value).map_err(|error| error.to_string())?
                }
                _ => value.to_string(),
            }
        };
        updated.insert(key.clone(), text);
    }
    values.extend(updated);
    values.insert("job_name".into(), name.clone());
    Ok(())
}

enum FileAction {
    Input(Tool, String),
    Export(PathBuf),
    LoadRun(Tool),
}

enum JobResult {
    Run(Result<AdapterResult, String>),
    Install(Result<(), String>),
}

/// A tool window: which tool is selected, each tool's form, the job running, and past results.
pub(crate) struct ToolWindow {
    tool: Tool,
    forms: HashMap<Tool, Form>,
    receiver: Option<mpsc::Receiver<JobResult>>,
    control: Option<RunControl>,
    started: Option<Instant>,
    installing: bool,
    error: Option<String>,
    message: Option<String>,
    results: Vec<(Tool, AdapterResult)>,
    selected_result: usize,
    selected_file: Option<PathBuf>,
    preview: Option<String>,
    dialog: FileDialog,
    file_action: Option<FileAction>,
    opened_protein: usize,
}

impl ToolWindow {
    fn new(kind: ToolWindowKind) -> Self {
        Self {
            tool: kind.tools()[0],
            forms: HashMap::new(),
            receiver: None,
            control: None,
            started: None,
            installing: false,
            error: None,
            message: None,
            results: Vec::new(),
            selected_result: 0,
            selected_file: None,
            preview: None,
            dialog: FileDialog::new(),
            file_action: None,
            opened_protein: 0,
        }
    }

    fn poll(&mut self) {
        let Some(receiver) = &self.receiver else {
            return;
        };

        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                JobResult::Run(Err("The worker stopped without returning a result.".into()))
            }
        };

        match result {
            JobResult::Run(Ok(result)) => {
                let first = primary_file(self.tool, &result).cloned();
                self.results.push((self.tool, result));
                self.selected_result = self.results.len() - 1;
                self.preview = first.as_ref().map(|path| preview_text(path));
                self.selected_file = first;
                self.message = Some("Run completed. All raw results are retained on disk.".into());
            }
            JobResult::Run(Err(error)) | JobResult::Install(Err(error)) => self.error = Some(error),
            JobResult::Install(Ok(())) => self.message = Some("Installation completed.".into()),
        }

        self.receiver = None;
        self.control = None;
        self.started = None;
        self.installing = false;
    }

    /// Start a job on a worker thread, reporting through [`Self::poll`].
    fn start_job(
        &mut self,
        ui: &Ui,
        installing: bool,
        job: impl FnOnce() -> JobResult + Send + 'static,
    ) {
        let (sender, receiver) = mpsc::channel();
        let context = ui.ctx().clone();

        self.receiver = Some(receiver);
        self.started = Some(Instant::now());
        self.installing = installing;
        self.error = None;
        self.message = None;

        thread::spawn(move || {
            let _ = sender.send(job());
            context.request_repaint();
        });
    }

    /// Draw the window. Returns a result file the user asked to load as a structure.
    fn draw(
        &mut self,
        tools: &[Tool],
        proteins: &[MoleculePeptide],
        ui: &mut Ui,
    ) -> Option<PathBuf> {
        self.poll();
        self.handle_picked_file(ui);

        let busy = self.receiver.is_some();
        ui.add_enabled_ui(!busy, |ui| {
            ComboBox::from_id_salt("scientific_tool")
                .selected_text(self.tool.spec().name())
                .show_ui(ui, |ui| {
                    for tool in tools {
                        ui.selectable_value(&mut self.tool, *tool, tool.spec().name());
                    }
                });
        });

        let spec = self.tool.spec();
        ui.hyperlink_to("Tool documentation", spec.url());
        if !spec.platform.is_supported() {
            ui.colored_label(
                Color32::ORANGE,
                "This tool requires Linux. Inputs and results can still be viewed here.",
            );
        }

        if let Entry::Vacant(entry) = self.forms.entry(self.tool) {
            match Form::new(self.tool) {
                Ok(form) => {
                    entry.insert(form);
                }
                Err(error) => {
                    ui.colored_label(Color32::LIGHT_RED, error);
                    return None;
                }
            }
        }

        self.progress_ui(ui);

        let mut pick = None;
        let mut use_protein = false;
        let form = self.forms.get_mut(&self.tool).unwrap();

        ui.add_enabled_ui(!busy, |ui| {
            preset_ui(self.tool, form, ui);
            mode_and_task_ui(self.tool, form, ui);

            if form.contract.structure_field(&form.mode).is_some() && !proteins.is_empty() {
                ui.horizontal(|ui| {
                    ComboBox::from_id_salt("tool_open_protein")
                        .selected_text(
                            proteins
                                .get(self.opened_protein)
                                .map(|protein| {
                                    protein
                                        .common
                                        .name
                                        .as_deref()
                                        .unwrap_or(&protein.common.ident)
                                })
                                .unwrap_or("Select an opened protein"),
                        )
                        .show_ui(ui, |ui| {
                            for (index, protein) in proteins.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.opened_protein,
                                    index,
                                    protein
                                        .common
                                        .name
                                        .as_deref()
                                        .unwrap_or(&protein.common.ident),
                                );
                            }
                        });
                    use_protein = ui.button("Use opened structure").clicked();
                });
            }

            fields_ui(self.tool, form, &mut pick, ui);
        });

        if let Some(field) = pick {
            self.file_action = Some(FileAction::Input(self.tool, field));
            self.dialog.pick_file();
        }

        if use_protein
            && let Some(protein) = proteins.get(self.opened_protein)
            && let Some(field) = form.contract.structure_field(&form.mode)
        {
            match write_opened_structure(protein) {
                Ok(path) => {
                    form.values
                        .insert(field.name.clone(), path.display().to_string());
                    form.authoritative_mode = form.mode.clone();
                }
                Err(error) => self.error = Some(error.to_string()),
            }
        }

        let mut install = false;
        let mut run = false;

        ui.horizontal(|ui| {
            run = ui
                .add_enabled(
                    !busy && spec.platform.is_supported() && form.projection_error.is_none(),
                    Button::new(RichText::new("Run").color(COLOR_ACTION)),
                )
                .clicked();

            install = ui
                .add_enabled(
                    !busy && spec.can_install_here(),
                    Button::new("Install / repair tool"),
                )
                .clicked();
            ui.label(RichText::new("Raw outputs are saved locally.").small());
            if ui
                .button("Open saved run…")
                .on_hover_text("Choose result.json from a saved run folder.")
                .clicked()
            {
                self.file_action = Some(FileAction::LoadRun(self.tool));
                self.dialog.pick_file();
            }
        });

        let tool = self.tool;
        let payload = run.then(|| shared_adapter::payload(tool, &form.values, &form.mode));

        if install {
            self.start_job(ui, true, move || {
                JobResult::Install(external_tools::install(tool).map_err(|error| error.to_string()))
            });
        }

        if let Some(payload) = payload {
            match payload {
                Err(error) => self.error = Some(error.to_string()),
                Ok(payload) => {
                    let control = RunControl::default();
                    self.control = Some(control.clone());
                    self.start_job(ui, false, move || {
                        JobResult::Run(
                            shared_adapter::run(tool, payload, &control)
                                .map_err(|error| error.to_string()),
                        )
                    });
                }
            }
        }

        if let Some(error) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
        if let Some(message) = &self.message {
            ui.label(message);
        }

        self.results_ui(ui)
    }

    /// Route a file the dialog returned: into a form field, or as the destination of an export.
    fn handle_picked_file(&mut self, ui: &Ui) {
        self.dialog.update(ui.ctx());
        let Some(path) = self.dialog.take_picked() else {
            return;
        };

        match self.file_action.take() {
            Some(FileAction::Input(tool, field)) => {
                if let Some(form) = self.forms.get_mut(&tool) {
                    form.values.insert(field, path.display().to_string());
                    form.authoritative_mode = form.mode.clone();
                }
            }
            Some(FileAction::Export(source)) => match fs::copy(&source, &path) {
                Ok(_) => self.message = Some(format!("Saved {}", path.display())),
                Err(error) => self.error = Some(error.to_string()),
            },
            Some(FileAction::LoadRun(tool)) => {
                match AdapterResult::load(path.parent().unwrap_or(Path::new(".")).to_owned()) {
                    Ok(result) => {
                        let saved_slug = result.details.get("tool_slug").and_then(Value::as_str);
                        let tool = Tool::ALL
                            .into_iter()
                            .find(|candidate| {
                                saved_slug.is_some()
                                    && candidate.spec().adapter_slug() == saved_slug
                            })
                            .unwrap_or(tool);
                        self.selected_file = primary_file(tool, &result).cloned();
                        self.preview = self.selected_file.as_ref().map(|path| preview_text(path));
                        self.selected_result = self.results.len();
                        self.results.push((tool, result));
                    }
                    Err(error) => self.error = Some(error.to_string()),
                }
            }
            None => {}
        }
    }

    /// The spinner, elapsed time, and cancel button for a running job.
    fn progress_ui(&self, ui: &mut Ui) {
        let Some(started) = self.started else {
            return;
        };

        ui.horizontal(|ui| {
            ui.spinner();
            let elapsed = started.elapsed().as_secs();
            let activity = if self.installing {
                "Installing"
            } else {
                "Running"
            };
            ui.label(format!(
                "{activity} · {}m {:02}s",
                elapsed / 60,
                elapsed % 60
            ));

            if let Some(control) = &self.control {
                if ui
                    .add_enabled(!control.is_cancel_requested(), Button::new("Cancel run"))
                    .clicked()
                {
                    control.cancel();
                }
                if control.is_cancel_requested() {
                    ui.label("Stopping…");
                }
            }
        });
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    }

    /// Past runs: their files, a preview, and the adapter's diagnostics.
    fn results_ui(&mut self, ui: &mut Ui) -> Option<PathBuf> {
        if self.results.is_empty() {
            return None;
        }
        ui.separator();

        let previous = self.selected_result;
        ComboBox::from_id_salt("tool_result_history")
            .selected_text(format!(
                "Result {} · {}",
                self.selected_result + 1,
                self.results[self.selected_result].0.spec().name()
            ))
            .show_ui(ui, |ui| {
                for (index, (tool, _)) in self.results.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.selected_result,
                        index,
                        format!("Run {} · {}", index + 1, tool.spec().name()),
                    );
                }
            });
        if previous != self.selected_result {
            let (tool, result) = &self.results[self.selected_result];
            self.selected_file = primary_file(*tool, result).cloned();
            self.preview = self.selected_file.as_ref().map(|path| preview_text(path));
        }

        let result = &self.results[self.selected_result].1;
        let mut export = None;
        ui.horizontal(|ui| {
            if let Some(path) = &self.selected_file
                && ui.button("Save selected output…").clicked()
            {
                export = Some(path.clone());
            }
            if ui.button("Save raw results (.zip)…").clicked() {
                export = Some(result.archive.clone());
            }
            if ui.button("Open run folder").clicked()
                && let Err(error) = open_dir(&result.directory)
            {
                self.error = Some(error.to_string());
            }
        });

        let mut load = None;
        ScrollArea::vertical()
            .id_salt("tool_result_files")
            .max_height(180.0)
            .show(ui, |ui| {
                for path in &result.files {
                    ui.horizontal(|ui| {
                        if output_category(path) == Some(DataCategory::Structure)
                            && ui
                                .button(RichText::new("Load structure").color(COLOR_ACTION))
                                .clicked()
                        {
                            match structure_for_loading(path, &result.directory) {
                                Ok(path) => load = Some(path),
                                Err(error) => self.error = Some(error.to_string()),
                            }
                        }
                        if ui.button("Save…").clicked() {
                            export = Some(path.clone());
                        }

                        let log_root = result
                            .details
                            .get("run_log_dir")
                            .and_then(Value::as_str)
                            .map(Path::new);
                        let label = log_root
                            .and_then(|root| path.strip_prefix(root.join("outputs")).ok())
                            .unwrap_or(path)
                            .display()
                            .to_string();
                        if ui
                            .selectable_label(self.selected_file.as_ref() == Some(path), label)
                            .on_hover_text(path.display().to_string())
                            .clicked()
                        {
                            self.selected_file = Some(path.clone());
                            self.preview = Some(preview_text(path));
                        }
                    });
                }
            });

        if let Some(text) = &self.preview {
            if self
                .selected_file
                .as_ref()
                .is_some_and(|path| output_category(path) == Some(DataCategory::Sequence))
            {
                sequence_results_ui(text, ui);
            }
            ui.horizontal(|ui| {
                ui.label("Selected output");
                if ui.button("Copy text").clicked() {
                    ui.ctx().copy_text(text.clone());
                }
            });
            ScrollArea::vertical()
                .id_salt("tool_output_preview")
                .max_height(220.0)
                .show(ui, |ui| {
                    ui.add(
                        TextEdit::multiline(&mut text.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
        }

        CollapsingHeader::new("Run details and diagnostics").show(ui, |ui| {
            let text = serde_json::to_string_pretty(&result.details).unwrap_or_default();
            if ui.button("Copy details").clicked() {
                ui.ctx().copy_text(text.clone());
            }
            ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                ui.label(RichText::new(text).monospace());
            });
        });

        if let Some(path) = export {
            self.dialog.config_mut().default_file_name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            self.file_action = Some(FileAction::Export(path));
            self.dialog.save_file();
        }
        load
    }
}

/// The text shown when an output file is selected.
fn preview_text(path: &Path) -> String {
    read_output_text(path, MAX_PREVIEW_BYTES).unwrap_or_else(|error| error.to_string())
}

fn read_output_text(path: &Path, maximum: u64) -> io::Result<String> {
    let file = fs::File::open(path)?;
    let reader: Box<dyn Read> = if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gz"))
    {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut bytes = Vec::new();
    reader.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(io::Error::other(
            "This output is too large for inline preview. Save the original file to inspect it.",
        ));
    }
    String::from_utf8(bytes)
        .map_err(|_| io::Error::other("Binary output. Save the original file to inspect it."))
}

fn output_category(path: &Path) -> Option<DataCategory> {
    let name = path.to_string_lossy().to_lowercase();
    let name = name.strip_suffix(".gz").unwrap_or(&name);
    [
        DataCategory::Structure,
        DataCategory::Sequence,
        DataCategory::Table,
    ]
    .into_iter()
    .find(|kind| kind.suffixes().iter().any(|suffix| name.ends_with(suffix)))
}

fn primary_file(tool: Tool, result: &AdapterResult) -> Option<&PathBuf> {
    let category = tool
        .spec()
        .catalog()
        .and_then(|entry| entry.primary_output)
        .map(|kind| kind.category());
    result
        .files
        .iter()
        .find(|path| category.is_some() && output_category(path) == category)
        .or_else(|| result.files.first())
}

fn structure_for_loading(path: &Path, directory: &Path) -> io::Result<PathBuf> {
    use std::hash::{Hash, Hasher};
    let gzip = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gz"));
    let ent = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("ent"));
    if !gzip && !ent {
        return Ok(path.to_owned());
    }
    let name = if gzip {
        path.file_stem()
    } else {
        path.file_name()
    }
    .unwrap_or_default();
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hash);
    let folder = directory
        .join("viewer")
        .join(format!("{:x}", hash.finish()));
    fs::create_dir_all(&folder)?;
    let mut destination = folder.join(name);
    if destination
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("ent"))
    {
        destination.set_extension("pdb");
    }
    fs::write(&destination, read_output_text(path, 200_000_000)?)?;
    Ok(destination)
}

fn sequence_results_ui(fasta: &str, ui: &mut Ui) {
    let mut records: Vec<(String, String)> = Vec::new();
    for line in fasta.lines() {
        if let Some(header) = line.strip_prefix('>') {
            records.push((header.to_owned(), String::new()));
        } else if let Some((_, sequence)) = records.last_mut() {
            sequence.extend(line.chars().filter(|character| !character.is_whitespace()));
        }
    }
    if records.is_empty() {
        return;
    }
    ui.label(RichText::new(format!("{} sequence records", records.len())).strong());
    ScrollArea::vertical()
        .id_salt("designed_sequences")
        .max_height(240.0)
        .show(ui, |ui| {
            for (index, (header, sequence)) in records.iter().take(500).enumerate() {
                ui.push_id(index, |ui| {
                    ui.group(|ui| {
                        ui.label(header);
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "{} residues",
                                sequence.chars().filter(char::is_ascii_alphabetic).count()
                            ));
                            if ui.button("Copy sequence").clicked() {
                                ui.ctx().copy_text(sequence.clone());
                            }
                            if ui.button("Copy FASTA").clicked() {
                                ui.ctx().copy_text(format!(">{header}\n{sequence}\n"));
                            }
                        });
                        ui.label(RichText::new(sequence).monospace());
                    })
                });
            }
        });
    if records.len() > 500 {
        ui.label("Showing the first 500 records. Save the FASTA file for every sequence.");
    }
}

/// Write an opened protein to `process_executables/desktop_inputs`, for a form's structure field.
fn write_opened_structure(protein: &MoleculePeptide) -> io::Result<PathBuf> {
    let pdb = peptide_to_pdb(protein, &PdbWriteOptions::with_ligand_context())?;

    let directory = external_tools::process_executables_dir()?.join("desktop_inputs");
    fs::create_dir_all(&directory)?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = directory.join(format!("opened-{}-{stamp}.pdb", std::process::id()));

    fs::write(&path, pdb)?;
    Ok(path)
}

// ---------------------------------------------------------------------------------------------
// Form widgets
// ---------------------------------------------------------------------------------------------

/// The example picker, and the selected example's description.
fn preset_ui(tool: Tool, form: &mut Form, ui: &mut Ui) {
    ui.horizontal(|ui| {
        let mut chosen = form.preset;
        ComboBox::from_id_salt("tool_preset")
            .selected_text(
                chosen
                    .map(|index| form.presets[index].label.as_str())
                    .unwrap_or("Choose an example…"),
            )
            .show_ui(ui, |ui| {
                for (index, preset) in form.presets.iter().enumerate() {
                    ui.selectable_value(&mut chosen, Some(index), &preset.label)
                        .on_hover_text(&preset.description);
                }
            });

        let reload = ui
            .add_enabled(chosen.is_some(), Button::new("Load example"))
            .clicked();
        if (chosen != form.preset || reload)
            && let Some(index) = chosen
        {
            form.apply_preset(tool, index);
        }

        if ui.button("Reset fields").clicked()
            && let Ok(fresh) = Form::new(tool)
        {
            *form = fresh;
        }
    });

    if let Some(index) = form.preset {
        let preset = &form.presets[index];
        ui.label(&preset.description);
        if let Some(url) = &preset.source_url {
            ui.hyperlink_to("Example source", url);
        }
    }
    if let Some(error) = &form.projection_error {
        ui.colored_label(Color32::ORANGE, error);
    }
}

/// The input-mode radio buttons and the task picker, for the tools that have them.
fn mode_and_task_ui(tool: Tool, form: &mut Form, ui: &mut Ui) {
    if let Some(modes) = &form.contract.input_modes {
        let previous = form.mode.clone();
        ui.horizontal_wrapped(|ui| {
            ui.label(&modes.label);
            for option in &modes.options {
                ui.radio_value(
                    &mut form.mode,
                    option.value.clone(),
                    option.label.replace("Upload", "Choose file:"),
                );
            }
        });
        if form.mode != previous && shared_adapter::slug(tool) == "rfd3" {
            form.project_mode();
        }
    }

    if !form.contract.tasks.is_empty() {
        let task = form.values.entry("task".into()).or_default();
        ComboBox::from_id_salt("tool_task")
            .selected_text(
                form.contract
                    .tasks
                    .iter()
                    .find(|option| &option.value == task)
                    .map(|option| option.label.as_str())
                    .unwrap_or(task),
            )
            .show_ui(ui, |ui| {
                for option in &form.contract.tasks {
                    ui.selectable_value(task, option.value.clone(), &option.label);
                }
            });
    }
}

/// Every field visible in the current mode and task, under its group's heading.
fn fields_ui(tool: Tool, form: &mut Form, pick: &mut Option<String>, ui: &mut Ui) {
    let before = form.values.clone();
    ScrollArea::vertical()
        .id_salt("tool_form_scroll")
        .max_height(480.0)
        .show(ui, |ui| {
            let groups = form.contract.groups_in_mode(&form.mode);
            for (index, (label, group, fields)) in groups.into_iter().enumerate() {
                // Every group the contract names gets its own heading. The salt is the group name
                // rather than the heading text so that the sections keep distinct ids: the one
                // group a tool may leave unnamed is the only one drawn as "Parameters".
                let title = if label.is_empty() {
                    "Parameters"
                } else {
                    label
                };

                CollapsingHeader::new(title)
                    .id_salt((shared_adapter::slug(tool), label))
                    .default_open(index == 0)
                    .show(ui, |ui| {
                        if let Some(url) = group.and_then(|group| group.docs_url.as_deref()) {
                            ui.hyperlink_to("Parameter documentation", url);
                        }

                        let task = form.values.get("task").cloned().unwrap_or_default();
                        for field in fields {
                            if !field.applies_to_task(&task) {
                                continue;
                            }
                            ui.push_id(&field.name, |ui| {
                                draw_field(field, &mut form.values, pick, ui);
                            });
                        }
                    });
            }
        });
    let edited_input = form.contract.fields.iter().any(|field| {
        field.input_modes == form.mode && form.values.get(&field.name) != before.get(&field.name)
    });
    if edited_input && shared_adapter::slug(tool) == "rfd3" {
        form.authoritative_mode = form.mode.clone();
        form.projection_error = None;
    }
}

fn draw_field(
    field: &FormField,
    values: &mut HashMap<String, String>,
    pick: &mut Option<String>,
    ui: &mut Ui,
) {
    if field.managed_by_runner {
        ui.label(RichText::new(format!("{}: {}", field.label, field.help_note)).weak());
        return;
    }
    let value = values.entry(field.name.clone()).or_default();
    let label = format!("{}{}", field.label, if field.required { " *" } else { "" });

    let response = match field.kind() {
        FieldKind::Checkbox => {
            let mut checked = matches!(value.as_str(), "true" | "1" | "yes" | "on");
            let response = ui.checkbox(&mut checked, &label);
            if response.changed() {
                *value = checked.to_string();
            }
            response
        }
        FieldKind::Select => {
            ui.horizontal(|ui| {
                ui.label(&label);
                ComboBox::from_id_salt("choice")
                    .selected_text(
                        field
                            .options
                            .iter()
                            .find(|option| &option.value == value)
                            .map(|option| option.label.as_str())
                            .unwrap_or(value),
                    )
                    .show_ui(ui, |ui| {
                        for option in &field.options {
                            ui.selectable_value(value, option.value.clone(), &option.label);
                        }
                    });
            })
            .response
        }
        FieldKind::TextArea => {
            ui.label(&label);
            ui.add(
                TextEdit::multiline(value)
                    .desired_rows(field.rows.unwrap_or(3).clamp(2, 12))
                    .desired_width(f32::INFINITY),
            )
        }
        FieldKind::Molecules => {
            ui.label(&label);
            molecules(field, value, ui);
            ui.label(&field.help)
        }
        FieldKind::File => {
            ui.horizontal(|ui| {
                ui.label(&label);
                ui.add(TextEdit::singleline(value).desired_width(300.0));
                if ui.button("Choose…").on_hover_text(&field.accept).clicked() {
                    *pick = Some(field.name.clone());
                }
            })
            .response
        }
        FieldKind::Number | FieldKind::Text | FieldKind::Other => {
            let width = if field.kind() == FieldKind::Number {
                100.0
            } else {
                300.0
            };
            ui.horizontal(|ui| {
                ui.label(&label);
                ui.add(TextEdit::singleline(value).desired_width(width));
            })
            .response
        }
    };

    let mut help = format!("{}\n{}", field.help, field.help_note);
    if field.kind() == FieldKind::Number {
        if let Some(minimum) = field.minimum {
            help.push_str(&format!("\nMinimum: {minimum}"));
        }
        if let Some(maximum) = field.maximum {
            help.push_str(&format!("\nMaximum: {maximum}"));
        }
        if !field.step.is_null() {
            help.push_str(&format!("\nStep: {}", value_to_hint(&field.step)));
        }
    }
    response.on_hover_text(help);
}

fn value_to_hint(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

/// The molecule builder: one box per entity, over the JSON list the field holds.
fn molecules(field: &FormField, text: &mut String, ui: &mut Ui) {
    let editor_id = ui.id().with("molecule_json_editor");
    let mut raw = ui.data_mut(|data| data.get_temp::<bool>(editor_id).unwrap_or(false));
    ui.checkbox(&mut raw, "Edit molecules as JSON");
    ui.data_mut(|data| data.insert_temp(editor_id, raw));
    if raw {
        ui.add(
            TextEdit::multiline(text)
                .desired_rows(10)
                .desired_width(f32::INFINITY),
        );
        if let Err(error) = serde_json::from_str::<Vec<serde_json::Map<String, Value>>>(text) {
            ui.colored_label(Color32::LIGHT_RED, error.to_string());
        }
        return;
    }
    let Ok(mut boxes) = serde_json::from_str::<Vec<Value>>(text) else {
        ui.colored_label(
            Color32::LIGHT_RED,
            "Molecules must be a JSON list. Edit the document below to repair it.",
        );
        ui.add(TextEdit::multiline(text).desired_rows(5));
        return;
    };
    if boxes.iter().any(|value| !value.is_object()) {
        ui.colored_label(
            Color32::LIGHT_RED,
            "Each molecule must be a JSON object. Enable the JSON editor to repair it.",
        );
        return;
    }

    let features: Vec<_> = field.molecule_features.split(',').collect();
    let mut remove = None;

    for (index, molecule) in boxes.iter_mut().enumerate() {
        ui.push_id(index, |ui| {
            ui.group(|ui| {
                molecule_ui(field, &features, index, molecule, &mut remove, ui);
            });
        });
    }

    if let Some(index) = remove {
        boxes.remove(index);
    }
    if ui.button("Add molecule").clicked() {
        boxes.push(json!({"type": "protein", "sequence": "", "modifications": []}));
    }

    *text = serde_json::to_string(&boxes).unwrap_or_default();
}

/// One molecule's box in the builder.
fn molecule_ui(
    field: &FormField,
    features: &[&str],
    index: usize,
    molecule: &mut Value,
    remove: &mut Option<usize>,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.label(format!("Molecule {}", index + 1));

        let mut kind = molecule["type"].as_str().unwrap_or("protein").to_owned();
        ComboBox::from_id_salt("molecule_kind")
            .selected_text(&kind)
            .show_ui(ui, |ui| {
                for option in &field.options {
                    ui.selectable_value(&mut kind, option.value.clone(), &option.label);
                }
            });
        molecule["type"] = json!(kind);

        if ui.small_button("Remove").clicked() {
            *remove = Some(index);
        }
    });

    let kind = molecule["type"].as_str().unwrap_or("protein").to_owned();
    let key = match kind.as_str() {
        "ligand" => "ligand",
        "ion" => "ion",
        _ => "sequence",
    };
    let hint = if key == "sequence" {
        "Sequence"
    } else {
        "SMILES or CCD code"
    };

    let mut sequence = molecule[key].as_str().unwrap_or_default().to_owned();
    let edited = ui
        .add(
            TextEdit::multiline(&mut sequence)
                .desired_rows(2)
                .desired_width(f32::INFINITY)
                .hint_text(hint),
        )
        .changed();
    if edited {
        molecule[key] = json!(sequence);
    }

    if features.contains(&"id") {
        let mut id = match &molecule["id"] {
            Value::String(id) => id.clone(),
            Value::Array(ids) => ids
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(","),
            _ => ((b'A' + (index % 26) as u8) as char).to_string(),
        };
        ui.horizontal(|ui| {
            ui.label("Entity ID(s)");
            if ui.text_edit_singleline(&mut id).changed() {
                molecule["id"] = json!(id);
            }
        });
        if molecule.get("id").is_none() {
            molecule["id"] = json!(id);
        }
    }

    if features.contains(&"count") {
        let mut count = molecule["count"].as_u64().unwrap_or(1);
        ui.horizontal(|ui| {
            ui.label("Copies");
            if ui.add(DragValue::new(&mut count).range(1..=64)).changed() {
                molecule["count"] = json!(count);
            }
        });
    }

    if features.contains(&"cyclic") {
        let mut cyclic = molecule["cyclic"].as_bool().unwrap_or(false);
        if ui.checkbox(&mut cyclic, "Cyclic").changed() {
            molecule["cyclic"] = json!(cyclic);
        }
    }

    if !matches!(kind.as_str(), "ligand" | "ion") {
        if features.contains(&"modifications") || features.contains(&"zero_indexed_modifications") {
            modifications_ui(
                molecule,
                features.contains(&"zero_indexed_modifications"),
                ui,
            );
        }
        for (feature, key, label) in [
            (
                "protein_paired_msa",
                "paired_msa_path",
                "Paired protein MSA path",
            ),
            (
                "protein_unpaired_msa",
                "unpaired_msa_path",
                "Unpaired protein MSA path",
            ),
            (
                "protein_templates",
                "templates_path",
                "Protein templates path",
            ),
            ("rna_unpaired_msa", "unpaired_msa_path", "RNA MSA path"),
            ("protein_msa", "msa", "Protein MSA"),
            ("rna_msa", "msa", "RNA MSA"),
        ] {
            if features.contains(&feature) && feature.starts_with(&kind) {
                ui.horizontal(|ui| {
                    ui.label(label);
                    let mut value = match &molecule[key] {
                        Value::Null => String::new(),
                        Value::String(value) => value.clone(),
                        value => value.to_string(),
                    };
                    if ui.text_edit_singleline(&mut value).changed() {
                        // Preserve ordinary paths; the whole-document editor also accepts structured MSAs.
                        molecule[key] = json!(value);
                    }
                });
            }
        }
    }
}

fn modifications_ui(molecule: &mut Value, zero_based: bool, ui: &mut Ui) {
    CollapsingHeader::new("Modified residues").show(ui, |ui| {
        let base = if zero_based { 0 } else { 1 };
        ui.label(format!(
            "Positions start at {base}. Residue codes use the CCD, e.g. MSE."
        ));
        let modifications = molecule
            .as_object_mut()
            .unwrap()
            .entry("modifications")
            .or_insert_with(|| json!([]));
        let Some(modifications) = modifications.as_array_mut() else {
            ui.colored_label(
                Color32::LIGHT_RED,
                "Modifications must be a list; repair them in the JSON editor.",
            );
            return;
        };
        let mut remove = None;
        for (index, modification) in modifications.iter_mut().enumerate() {
            if !modification.is_object() {
                continue;
            }
            ui.push_id(index, |ui| {
                ui.horizontal(|ui| {
                    let mut position = modification["position"].as_i64().unwrap_or(base);
                    ui.label("Position");
                    if ui
                        .add(DragValue::new(&mut position).range(base..=i64::MAX))
                        .changed()
                    {
                        modification["position"] = json!(position);
                    }
                    let mut residue = modification["residue"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned();
                    ui.label("CCD");
                    if ui
                        .add(TextEdit::singleline(&mut residue).desired_width(80.0))
                        .changed()
                    {
                        modification["residue"] = json!(residue);
                    }
                    if ui.small_button("Remove").clicked() {
                        remove = Some(index);
                    }
                })
            });
        }
        if let Some(index) = remove {
            modifications.remove(index);
        }
        if ui.button("Add modification").clicked() {
            modifications.push(json!({"position": base, "residue": ""}));
        }
    });
}
