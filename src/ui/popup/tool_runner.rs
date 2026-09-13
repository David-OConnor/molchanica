//! The windows for every tool run through `bio_tools`' shared adapter: a form built from the tool's
//! contract, a runner, and a browser over each run's retained results.
//!
//! Structure prediction, sequence design, and backbone design are the same window over different
//! groups of tools; see [`ToolWindowKind`]. Nothing here is specific to one tool — what differs
//! between tools comes from the contract `bio_tools` publishes for each.

use std::{
    collections::{HashMap, hash_map::Entry},
    fs, io,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use egui::{
    Button, CollapsingHeader, Color32, ComboBox, DragValue, RichText, ScrollArea, TextEdit, Ui,
};
use egui_file_dialog::FileDialog;
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::peptide::MoleculePeptide;
use serde_json::{Value, json};

use super::close_btn;
use crate::{
    external_tools::{
        self, RunControl, Tool,
        pdb_write::{PdbWriteOptions, peptide_to_pdb},
        shared_adapter::{self, AdapterResult},
        tool_form::{FieldKind, FormContract, FormField, Preset},
    },
    state::State,
    ui::util::open_dir,
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
            Self::StructurePrediction => {
                &[Tool::OpenDde, Tool::Boltz2, Tool::Chai1, Tool::EsmFold2]
            }
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
    let popup = &mut state.ui.popup;
    let (window, open) = match kind {
        ToolWindowKind::StructurePrediction => {
            (&mut windows.structure_prediction, &mut popup.structure_pred)
        }
        ToolWindowKind::SequenceDesign => (&mut windows.sequence_design, &mut popup.sequence_pred),
        ToolWindowKind::BackboneDesign => (&mut windows.backbone_design, &mut popup.rfd3),
    };

    ui.horizontal(|ui| {
        ui.heading(kind.heading());
        close_btn(ui, open);
    });

    let load = window.draw(kind.tools(), &state.peptides, ui);

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
    preset: Option<usize>,
}

impl Form {
    fn new(tool: Tool) -> Result<Self, String> {
        let slug = shared_adapter::slug(tool);
        let contract = FormContract::load(slug)?;

        Ok(Self {
            mode: contract.default_mode(),
            values: contract.defaults(),
            presets: Preset::load_all(slug)?,
            contract,
            preset: None,
        })
    }

    fn apply_preset(&mut self, index: usize) {
        self.values = self.contract.defaults();
        self.values.extend(self.presets[index].form_values());
        self.mode = self
            .values
            .get("input_mode")
            .cloned()
            .unwrap_or_else(|| self.contract.default_mode());
        self.preset = Some(index);
    }
}

enum FileAction {
    Input(String),
    Export(PathBuf),
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
                self.results.push((self.tool, result));
                self.selected_result = self.results.len() - 1;
                self.selected_file = None;
                self.preview = None;
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
            mode_and_task_ui(form, ui);

            if form.contract.structure_field(&form.mode).is_some() && !proteins.is_empty() {
                ui.horizontal(|ui| {
                    ComboBox::from_id_salt("tool_open_protein")
                        .selected_text(
                            proteins
                                .get(self.opened_protein)
                                .map(|protein| protein.common.ident.as_str())
                                .unwrap_or("Select an opened protein"),
                        )
                        .show_ui(ui, |ui| {
                            for (index, protein) in proteins.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.opened_protein,
                                    index,
                                    &protein.common.ident,
                                );
                            }
                        });
                    use_protein = ui.button("Use opened structure").clicked();
                });
            }

            fields_ui(self.tool, form, &mut pick, ui);
        });

        if let Some(field) = pick {
            self.file_action = Some(FileAction::Input(field));
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
                }
                Err(error) => self.error = Some(error.to_string()),
            }
        }

        let mut install = false;
        let mut run = false;
        ui.horizontal(|ui| {
            run = ui
                .add_enabled(!busy && spec.platform.is_supported(), Button::new("Run"))
                .clicked();
            install = ui
                .add_enabled(
                    !busy && spec.can_install_here(),
                    Button::new("Install / repair tool"),
                )
                .clicked();
            ui.label(RichText::new("Raw outputs are saved locally.").small());
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
            Some(FileAction::Input(field)) => {
                if let Some(form) = self.forms.get_mut(&self.tool) {
                    form.values.insert(field, path.display().to_string());
                }
            }
            Some(FileAction::Export(source)) => match fs::copy(&source, &path) {
                Ok(_) => self.message = Some(format!("Saved {}", path.display())),
                Err(error) => self.error = Some(error.to_string()),
            },
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
            self.selected_file = None;
            self.preview = None;
        }

        let result = &self.results[self.selected_result].1;
        let mut export = None;
        ui.horizontal(|ui| {
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
                        let extension = path
                            .extension()
                            .and_then(|value| value.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if matches!(extension.as_str(), "pdb" | "cif" | "mmcif")
                            && ui.button("Load structure").clicked()
                        {
                            load = Some(path.clone());
                        }
                        if ui.button("Save…").clicked() {
                            export = Some(path.clone());
                        }

                        let label = path.file_name().unwrap_or_default().to_string_lossy();
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
            self.file_action = Some(FileAction::Export(path));
            self.dialog.save_file();
        }
        load
    }
}

/// The text shown when an output file is selected.
fn preview_text(path: &Path) -> String {
    if !fs::metadata(path).is_ok_and(|metadata| metadata.len() <= MAX_PREVIEW_BYTES) {
        return "This output is too large for inline preview. Use Save to export the original file."
            .into();
    }
    fs::read_to_string(path)
        .unwrap_or_else(|_| "Binary output. Use Save to export the original file.".into())
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
            form.apply_preset(index);
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
}

/// The input-mode radio buttons and the task picker, for the tools that have them.
fn mode_and_task_ui(form: &mut Form, ui: &mut Ui) {
    if let Some(modes) = &form.contract.input_modes {
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
    ScrollArea::vertical()
        .id_salt("tool_form_scroll")
        .max_height(480.0)
        .show(ui, |ui| {
            let groups = form.contract.groups_in_mode(&form.mode);
            for (index, (group, fields)) in groups.into_iter().enumerate() {
                let title = group
                    .map(|group| group.label.as_str())
                    .unwrap_or("Parameters");

                CollapsingHeader::new(title)
                    .id_salt((shared_adapter::slug(tool), title))
                    .default_open(index == 0)
                    .show(ui, |ui| {
                        if let Some(url) = group.and_then(|group| group.docs_url.as_deref()) {
                            ui.hyperlink_to("Parameter documentation", url);
                        }

                        let task = form.values.get("task").cloned().unwrap_or_default();
                        for field in fields {
                            if !field.task.is_empty()
                                && !field.task.split(',').any(|value| value.trim() == task)
                            {
                                continue;
                            }
                            ui.push_id(&field.name, |ui| {
                                draw_field(field, &mut form.values, pick, ui);
                            });
                        }
                    });
            }
        });
}

fn draw_field(
    field: &FormField,
    values: &mut HashMap<String, String>,
    pick: &mut Option<String>,
    ui: &mut Ui,
) {
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

    response.on_hover_text(format!("{}\n{}", field.help, field.help_note));
}

/// The molecule builder: one box per entity, over the JSON list the field holds.
fn molecules(field: &FormField, text: &mut String, ui: &mut Ui) {
    let Ok(mut boxes) = serde_json::from_str::<Vec<Value>>(text) else {
        ui.colored_label(
            Color32::LIGHT_RED,
            "Molecules must be a JSON list. Edit the document below to repair it.",
        );
        ui.add(TextEdit::multiline(text).desired_rows(5));
        return;
    };

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
    CollapsingHeader::new("Edit molecules as JSON").show(ui, |ui| {
        ui.add(
            TextEdit::multiline(text)
                .desired_rows(5)
                .desired_width(f32::INFINITY),
        );
    });
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
        let mut id = molecule["id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| ((b'A' + (index % 26) as u8) as char).to_string());
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

    CollapsingHeader::new("Molecule details (JSON)").show(ui, |ui| {
        ui.label("Modifications, alignments and template paths use the tool's molecule fields.");

        let mut document = serde_json::to_string_pretty(molecule).unwrap_or_default();
        let edited = ui
            .add(
                TextEdit::multiline(&mut document)
                    .desired_rows(5)
                    .desired_width(f32::INFINITY),
            )
            .changed();
        if edited && let Ok(value @ Value::Object(_)) = serde_json::from_str(&document) {
            *molecule = value;
        }
    });
}
