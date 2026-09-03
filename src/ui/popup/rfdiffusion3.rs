//! The RFdiffusion3 backbone-generation window.
//!
//! Every input this shows comes from `bio_tools`' shared form contract for `rfd3` — the same file
//! `bio_web` renders its page from — so the two offer the same fields, labels, help text, bounds,
//! and defaults, and a field added upstream appears here without being restated. See
//! [`crate::external_tools::tool_form`]. What this module adds is the parts a desktop application
//! has and a web form does not: designing against a protein already open in the scene, and loading
//! the generated backbones straight back into it.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use egui::{
    Align, Button, CollapsingHeader, Color32, ComboBox, DragValue, Layout, RichText, ScrollArea,
    TextEdit, Ui,
};
use graphics::{EngineUpdates, Scene};

use crate::{
    external_tools::{
        self, Tool,
        pdb_write::{PdbWriteOptions, peptide_to_pdb},
        rfdiffusion3::{self, InputMode, Rfd3Request, Rfd3Result, SuppliedStructure},
        tool_form::{FieldKind, FormContract, FormField, Preset},
    },
    state::State,
    ui::{
        COLOR_ACTION, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, popup::close_btn,
        util::open_dir,
    },
    util::handle_err,
};

/// Which structure an `input` of "uploaded" refers to.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum StructureSource {
    /// A protein already open in the scene, written to PDB for the run.
    #[default]
    Opened,
    /// Whatever path or bundled reference the `input` field itself holds.
    Field,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolActionKind {
    Install,
    Uninstall,
}

impl ToolActionKind {
    fn progress(self) -> &'static str {
        match self {
            Self::Install => "Installing",
            Self::Uninstall => "Uninstalling",
        }
    }
}

struct ToolAction {
    kind: ToolActionKind,
    receiver: mpsc::Receiver<Result<(), String>>,
}

#[derive(Default)]
struct DesignJob {
    receiver: Option<mpsc::Receiver<Result<Rfd3Result, String>>>,
    started_at: Option<Instant>,
    result: Option<Rfd3Result>,
    error: Option<String>,
}

impl DesignJob {
    fn is_running(&self) -> bool {
        self.receiver.is_some()
    }

    fn start(
        &mut self,
        context: &egui::Context,
        work: impl FnOnce() -> Result<Rfd3Result, String> + Send + 'static,
    ) {
        if self.is_running() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.started_at = Some(Instant::now());
        self.result = None;
        self.error = None;
        let context = context.clone();
        thread::spawn(move || {
            let _ = tx.send(work());
            context.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(receiver) = &self.receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(Ok(result)) => {
                self.result = Some(result);
                self.receiver = None;
                self.started_at = None;
            }
            Ok(Err(error)) => {
                self.error = Some(error);
                self.receiver = None;
                self.started_at = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.error =
                    Some("The RFdiffusion3 worker stopped without returning a result.".to_owned());
                self.receiver = None;
                self.started_at = None;
            }
        }
    }
}

pub(crate) struct Rfd3Ui {
    /// The form contract from `bio_tools`, or why it could not be read.
    contract: Result<FormContract, String>,
    presets: Vec<Preset>,
    /// Index into `presets`, offset by one so that zero means "no preset applied".
    selected_preset: usize,
    input_mode: InputMode,
    /// Field values, keyed by the names in the contract.
    values: HashMap<String, String>,
    structure_source: StructureSource,
    opened_protein: usize,
    job: DesignJob,
    tool_action: Option<ToolAction>,
    tool_action_result: Option<(ToolActionKind, Result<(), String>)>,
    confirm_uninstall: bool,
}

impl Default for Rfd3Ui {
    fn default() -> Self {
        let contract = rfdiffusion3::contract();
        let values = contract
            .as_ref()
            .map(FormContract::defaults)
            .unwrap_or_default();
        let input_mode = contract
            .as_ref()
            .map(|contract| InputMode::from_contract_value(&contract.default_mode()))
            .unwrap_or_default();

        Self {
            contract,
            presets: Preset::load_all("rfd3").unwrap_or_default(),
            selected_preset: 0,
            input_mode,
            values,
            structure_source: StructureSource::default(),
            opened_protein: 0,
            job: DesignJob::default(),
            tool_action: None,
            tool_action_result: None,
            confirm_uninstall: false,
        }
    }
}

impl Rfd3Ui {
    /// Reset every field to the contract's own defaults, then overlay one preset's values.
    ///
    /// Overlaying rather than merging matters: a preset that leaves `contig` out means a design
    /// with no contig, not whatever the last preset put there.
    fn apply_preset(&mut self, index: usize) {
        let Ok(contract) = &self.contract else {
            return;
        };
        let defaults = contract.defaults();
        let Some(preset) = self.presets.get(index) else {
            return;
        };
        let overlay = preset.form_values();

        self.input_mode = rfdiffusion3::preset_mode(&overlay);
        self.values = defaults;
        for (name, value) in overlay {
            self.values.insert(name, value);
        }
        // Presets carry their own bundled structures, so the `input` field is what feeds the run.
        self.structure_source = if self
            .values
            .get("input")
            .is_some_and(|input| !input.trim().is_empty())
        {
            StructureSource::Field
        } else {
            StructureSource::Opened
        };
    }

    fn value(&self, name: &str) -> &str {
        self.values
            .get(name)
            .map(String::as_str)
            .unwrap_or_default()
    }

    fn poll_tool_action(&mut self) {
        let Some(action) = &self.tool_action else {
            return;
        };
        let result = match action.receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("The tool-management worker stopped without returning a result.".to_owned())
            }
        };
        self.tool_action_result = Some((action.kind, result));
        self.tool_action = None;
        self.confirm_uninstall = false;
    }

    fn start_tool_action(&mut self, kind: ToolActionKind, context: &egui::Context) {
        if self.tool_action.is_some() || self.job.is_running() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.tool_action = Some(ToolAction { kind, receiver: rx });
        self.tool_action_result = None;
        let context = context.clone();
        thread::spawn(move || {
            let result = match kind {
                ToolActionKind::Install => external_tools::install(Tool::RfDiffusion3),
                ToolActionKind::Uninstall => external_tools::uninstall(Tool::RfDiffusion3),
            }
            .map_err(|error| error.to_string());
            let _ = tx.send(result);
            context.request_repaint();
        });
    }
}

pub(in crate::ui) fn rfdiffusion3_window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // A pick belongs to whichever file-valued field asked for it, so that one dialog can serve the
    // input structure, the input document, and anything a future contract adds.
    if let Some(path) = state.volatile.dialogs.rfd3_input.take_picked()
        && let Some(field) = state.volatile.dialogs.rfd3_input_field.take()
    {
        state
            .ui
            .rfd3
            .values
            .insert(field, path.display().to_string());
        state.ui.rfd3.structure_source = StructureSource::Field;
    }

    let protein_names: Vec<String> = state
        .peptides
        .iter()
        .map(|protein| protein.common.ident.clone())
        .collect();
    if state.ui.rfd3.opened_protein >= protein_names.len() {
        state.ui.rfd3.opened_protein = state
            .peptide_for_tools_i()
            .filter(|index| *index < protein_names.len())
            .unwrap_or(0);
    }

    state.ui.rfd3.job.poll();
    state.ui.rfd3.poll_tool_action();

    ui.horizontal(|ui| {
        ui.heading("RFdiffusion3");
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            close_btn(ui, &mut state.ui.popup.rfd3);
        });
    });

    let spec = Tool::RfDiffusion3.spec();
    ui.label(spec.summary());
    if let Err(error) = &state.ui.rfd3.contract {
        ui.label(RichText::new(error).color(Color32::LIGHT_RED));
        return;
    }

    ui.add_space(ROW_SPACING);

    let running = state.ui.rfd3.job.is_running();
    let managing_tool = state.ui.rfd3.tool_action.is_some();
    let mut preset_to_apply = None;
    let mut pick_for_field: Option<String> = None;
    let mut run_design = false;

    ui.add_enabled_ui(!running && !managing_tool, |ui| {
        presets_row(&mut state.ui.rfd3, &mut preset_to_apply, ui);
        input_mode_row(&mut state.ui.rfd3, ui);
        structure_row(&mut state.ui.rfd3, &protein_names, &mut pick_for_field, ui);

        ui.add_space(ROW_SPACING);
        fields(&mut state.ui.rfd3, &mut pick_for_field, ui);
    });

    if let Some(index) = preset_to_apply {
        state.ui.rfd3.apply_preset(index);
    }
    if let Some(field) = pick_for_field {
        state.volatile.dialogs.rfd3_input_field = Some(field);
        state.volatile.dialogs.rfd3_input.pick_file();
    }

    ui.add_space(ROW_SPACING);
    tool_controls(&mut state.ui.rfd3, running, ui);

    // A design only needs a structure when it names one; an unconditional monomer does not. The
    // `input` field is only what feeds the run in parameters mode — a document names its own
    // inputs — and every other input problem is the adapter's to report, in the tool's own terms.
    let structure_ready = state.ui.rfd3.input_mode != InputMode::Parameters
        || state.ui.rfd3.value("input") != rfdiffusion3::supplied_input_value()
        || !protein_names.is_empty();

    ui.horizontal(|ui| {
        if running {
            ui.spinner();
            ui.label(RichText::new("Generating backbones…").color(COLOR_ACTION));
            if let Some(started_at) = state.ui.rfd3.job.started_at {
                ui.label(format_elapsed(started_at.elapsed()));
                ui.ctx().request_repaint_after(Duration::from_secs(1));
            }
        } else {
            let response = ui.add_enabled(
                structure_ready
                    && !managing_tool
                    && spec.platform.is_supported()
                    && external_tools::is_installed(Tool::RfDiffusion3),
                Button::new(RichText::new("Generate backbones").color(COLOR_ACTION)),
            );
            if response.clicked() {
                run_design = true;
            }
            if !spec.platform.is_supported() {
                response.on_hover_text("RFdiffusion3 is Linux-only; it cannot run here.");
            }
        }
    });

    if run_design {
        start_design(state, ui.ctx());
    }

    if let Some(error) = &state.ui.rfd3.job.error {
        ui.label(RichText::new(error).color(Color32::LIGHT_RED));
    }
    if state.ui.rfd3.job.result.is_some() {
        show_result(state, scene, updates, ui);
    }
}

fn presets_row(rfd3: &mut Rfd3Ui, apply: &mut Option<usize>, ui: &mut Ui) {
    if rfd3.presets.is_empty() {
        return;
    }
    // Listed out first: the combo box writes the selection while reading the labels.
    let entries: Vec<(String, String)> = rfd3
        .presets
        .iter()
        .map(|preset| {
            (
                preset.label.clone(),
                // The id is the upstream example's own name, which is what to search the RFD3
                // documentation for, so it is worth showing beside the description.
                format!("{}\n\n{}", preset.id, preset.description),
            )
        })
        .collect();

    let mut selection = rfd3.selected_preset;
    ui.horizontal(|ui| {
        ui.label("Preset:");
        let selected = selection
            .checked_sub(1)
            .and_then(|index| entries.get(index))
            .map(|(label, _)| label.clone())
            .unwrap_or_else(|| "None".to_owned());
        ComboBox::from_id_salt("rfd3_preset")
            .width(280.0)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut selection, 0, "None");
                for (index, (label, description)) in entries.iter().enumerate() {
                    ui.selectable_value(&mut selection, index + 1, label.as_str())
                        .on_hover_text(description.as_str());
                }
            });
        if ui
            .add_enabled(selection > 0, Button::new("Load preset"))
            .on_hover_text(
                "Replace every field with this worked example from bio_tools, including its \
                 bundled input structure.",
            )
            .clicked()
        {
            *apply = Some(selection - 1);
        }
    });
    rfd3.selected_preset = selection;

    if let Some(preset) = rfd3
        .selected_preset
        .checked_sub(1)
        .and_then(|index| rfd3.presets.get(index))
    {
        ui.label(
            RichText::new(preset.description.as_str())
                .color(COLOR_INACTIVE)
                .small(),
        );
        if let Some(url) = &preset.source_url {
            ui.hyperlink(url.as_str());
        }
    }
}

fn input_mode_row(rfd3: &mut Rfd3Ui, ui: &mut Ui) {
    let Ok(contract) = &rfd3.contract else {
        return;
    };
    let Some(modes) = &contract.input_modes else {
        return;
    };
    // The modes themselves come from the contract. Two of the three web modes — pasting a
    // document and uploading one — are the same thing on the desktop, so the first of them to
    // appear provides the label and the rest are folded into it.
    let mut offered: Vec<(InputMode, String)> = Vec::new();
    for option in &modes.options {
        let mode = InputMode::from_contract_value(&option.value);
        if !offered.iter().any(|(existing, _)| *existing == mode) {
            offered.push((mode, option.label.clone()));
        }
    }

    ui.horizontal(|ui| {
        ui.label(format!("{}:", modes.label));
        for (mode, label) in &offered {
            let response = ui.radio_value(&mut rfd3.input_mode, *mode, label.as_str());
            if *mode == InputMode::Document {
                response.on_hover_text(
                    "A complete InputSpecification: one object per named design. YAML is passed \
                     to RFdiffusion3 as written, so its input paths must already be paths on \
                     this machine.",
                );
            }
        }
    });
}

fn structure_row(
    rfd3: &mut Rfd3Ui,
    protein_names: &[String],
    pick: &mut Option<String>,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.label("Input structure:");
        ui.radio_value(
            &mut rfd3.structure_source,
            StructureSource::Opened,
            "Opened protein",
        )
        .on_hover_text(
            "Write the selected protein to PDB for the run. Designs referring to it use \
             \"uploaded\", the same name the hosted form uses.",
        );
        ui.radio_value(
            &mut rfd3.structure_source,
            StructureSource::Field,
            "From the input field",
        );
    });

    match rfd3.structure_source {
        StructureSource::Opened => {
            ui.horizontal(|ui| {
                let selected = protein_names
                    .get(rfd3.opened_protein)
                    .map(String::as_str)
                    .unwrap_or("No proteins are open");
                ComboBox::from_id_salt("rfd3_opened_protein")
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        for (index, name) in protein_names.iter().enumerate() {
                            ui.selectable_value(&mut rfd3.opened_protein, index, name.as_str());
                        }
                    });
                if ui
                    .button("Use for input")
                    .on_hover_text(
                        "Set the design's `input` to \"uploaded\", so it designs against the \
                         protein selected here.",
                    )
                    .clicked()
                {
                    rfd3.values.insert(
                        "input".to_owned(),
                        rfdiffusion3::supplied_input_value().to_owned(),
                    );
                }
            });
        }
        StructureSource::Field => {
            let bundled = rfdiffusion3::is_bundled_structure(rfd3.value("input"));
            if bundled {
                ui.label(
                    RichText::new(
                        "The `input` field names a structure bundled with bio_tools; it is \
                         written into the run directory as-is.",
                    )
                    .color(COLOR_INACTIVE)
                    .small(),
                );
            }
            let choose = ui.button("Choose a structure file…");
            if choose.clicked() {
                *pick = Some("input".to_owned());
            }
            if let Ok(contract) = &rfd3.contract
                && let Some(field) = contract.field("input")
            {
                choose.on_hover_text(field.help.as_str());
            }
        }
    }
}

/// Draw every field the contract lists for the current input mode, grouped as `bio_tools` groups
/// them. The first group is expanded; the rest start collapsed, because RFD3's own documentation
/// treats the CLI arguments below as things most runs leave alone.
fn fields(rfd3: &mut Rfd3Ui, pick: &mut Option<String>, ui: &mut Ui) {
    let Ok(contract) = &rfd3.contract else {
        return;
    };
    let mode = rfd3.input_mode.contract_value();
    let groups = contract.groups_in_mode(mode);
    let values = &mut rfd3.values;

    for (index, (group, members)) in groups.into_iter().enumerate() {
        let label = group.map(|group| group.label.as_str()).unwrap_or("Other");
        CollapsingHeader::new(label)
            .id_salt(format!("rfd3_group_{label}"))
            .default_open(index == 0)
            .show(ui, |ui| {
                if let Some(url) = group.and_then(|group| group.docs_url.as_deref()) {
                    ui.hyperlink_to("Upstream documentation for these fields", url);
                }
                for field in &members {
                    render_field(field, values, pick, ui);
                }
            });
    }

    if !contract.references.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Reference:").color(COLOR_INACTIVE).small());
            for (name, url) in &contract.references {
                ui.hyperlink_to(RichText::new(name.replace('_', " ")).small(), url);
            }
        });
    }
}

fn render_field(
    field: &FormField,
    values: &mut HashMap<String, String>,
    pick: &mut Option<String>,
    ui: &mut Ui,
) {
    let value = values.entry(field.name.clone()).or_default();
    let mut label = if field.label.is_empty() {
        field.name.clone()
    } else {
        field.label.clone()
    };
    if field.required {
        label.push_str(" *");
    }

    let response = match field.kind() {
        FieldKind::Checkbox => {
            let mut flag = matches!(value.as_str(), "true" | "1" | "yes" | "on");
            let response = ui.checkbox(&mut flag, label.as_str());
            *value = flag.to_string();
            response
        }
        FieldKind::TextArea => {
            ui.label(label.as_str());
            ui.add(
                TextEdit::multiline(value)
                    .desired_rows(field.rows.unwrap_or(3).clamp(2, 20))
                    .desired_width(f32::INFINITY),
            )
        }
        FieldKind::Select => {
            ui.horizontal(|ui| {
                ui.label(label.as_str());
                let selected = field
                    .options
                    .iter()
                    .find(|option| &option.value == value)
                    .map(|option| option.label.clone())
                    .unwrap_or_else(|| value.clone());
                ComboBox::from_id_salt(format!("rfd3_field_{}", field.name))
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        for option in &field.options {
                            ui.selectable_value(value, option.value.clone(), option.label.as_str());
                        }
                    });
            })
            .response
        }
        FieldKind::Number => {
            ui.horizontal(|ui| {
                ui.label(label.as_str());
                // Numbers are held as text so that a field the contract allows to be blank — where
                // blank means "use the model's default" — stays blank rather than becoming zero.
                if value.trim().is_empty() {
                    ui.add(TextEdit::singleline(value).desired_width(90.0))
                        .on_hover_text("Blank: use RFdiffusion3's own default.");
                    return;
                }
                let mut number: f64 = value.trim().parse().unwrap_or_default();
                let minimum = field.minimum.unwrap_or(f64::MIN);
                let maximum = field.maximum.unwrap_or(f64::MAX);
                let mut drag = DragValue::new(&mut number).range(minimum..=maximum);
                if let Some(step) = field.step {
                    drag = drag.speed(step);
                }
                if ui.add(drag).changed() {
                    *value = format_number(number, field.step);
                }
                if ui.small_button("Clear").clicked() {
                    value.clear();
                }
            })
            .response
        }
        FieldKind::File => {
            ui.horizontal(|ui| {
                ui.label(label.as_str());
                ui.add(TextEdit::singleline(value).desired_width(280.0));
                let accepted = field.accepted_extensions().join(", ");
                let choose = ui.button("Choose…");
                if choose.clicked() {
                    *pick = Some(field.name.clone());
                }
                if !accepted.is_empty() {
                    choose.on_hover_text(format!("Accepts: {accepted}"));
                }
            })
            .response
        }
        FieldKind::Text | FieldKind::Other => {
            ui.horizontal(|ui| {
                ui.label(label.as_str());
                ui.add(TextEdit::singleline(value).desired_width(280.0));
            })
            .response
        }
    };

    let mut hover = field.help.clone();
    if !field.help_note.is_empty() {
        hover.push_str("\n\n");
        hover.push_str(&field.help_note);
    }
    if !hover.is_empty() {
        response.on_hover_text(hover);
    }
}

fn tool_controls(rfd3: &mut Rfd3Ui, design_running: bool, ui: &mut Ui) {
    let spec = Tool::RfDiffusion3.spec();
    if !spec.platform.is_supported() {
        ui.label(
            RichText::new("RFdiffusion3 is Linux-only; it cannot be installed or run here.")
                .color(Color32::ORANGE),
        );
        return;
    }

    let installed = external_tools::is_installed(Tool::RfDiffusion3);
    ui.horizontal(|ui| {
        if let Some(action) = &rfd3.tool_action {
            ui.spinner();
            ui.label(format!("{} RFdiffusion3…", action.kind.progress()));
        } else if installed {
            let label = if rfd3.confirm_uninstall {
                "Confirm uninstall"
            } else {
                "Uninstall"
            };
            if ui
                .add_enabled(!design_running, Button::new(label))
                .clicked()
            {
                if rfd3.confirm_uninstall {
                    rfd3.start_tool_action(ToolActionKind::Uninstall, ui.ctx());
                } else {
                    rfd3.confirm_uninstall = true;
                }
            }
            ui.label(RichText::new("RFdiffusion3 is installed").color(Color32::LIGHT_GREEN));
        } else {
            if ui
                .add_enabled(!design_running, Button::new("Install"))
                .on_hover_text(spec.install_command())
                .clicked()
            {
                rfd3.start_tool_action(ToolActionKind::Install, ui.ctx());
            }
            ui.label(
                RichText::new("RFdiffusion3 is not installed. It needs an NVIDIA GPU.")
                    .color(COLOR_INACTIVE),
            );
        }
    });

    if let Some((kind, result)) = &rfd3.tool_action_result {
        match result {
            Ok(()) => ui.label(
                RichText::new(match kind {
                    ToolActionKind::Install => "RFdiffusion3 installation completed.",
                    ToolActionKind::Uninstall => "RFdiffusion3 was uninstalled.",
                })
                .color(Color32::LIGHT_GREEN),
            ),
            Err(error) => ui.label(RichText::new(error).color(Color32::LIGHT_RED)),
        };
    }
}

fn start_design(state: &mut State, context: &egui::Context) {
    let rfd3 = &mut state.ui.rfd3;
    let supplied_structure = match rfd3.structure_source {
        StructureSource::Opened => {
            match state.peptides.get(rfd3.opened_protein) {
                Some(protein) => {
                    let options = PdbWriteOptions {
                        chains: Vec::new(),
                        // RFD3 conditions on ligands, nucleic acids, and metals, so the context
                        // has to survive the trip through PDB.
                        include_hetero: true,
                        include_hydrogen: false,
                        include_water: false,
                    };
                    match peptide_to_pdb(protein, &options) {
                        Ok(pdb) => Some(SuppliedStructure {
                            name: protein.common.ident.clone(),
                            pdb,
                        }),
                        Err(error) => {
                            rfd3.job.error = Some(error.to_string());
                            return;
                        }
                    }
                }
                None => None,
            }
        }
        StructureSource::Field => None,
    };

    let request = Rfd3Request {
        input_mode: rfd3.input_mode,
        values: rfd3.values.clone(),
        supplied_structure,
    };
    rfd3.job.start(context, move || {
        rfdiffusion3::run(&request).map_err(|error| error.to_string())
    });
}

fn show_result(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates, ui: &mut Ui) {
    let Some(result) = &state.ui.rfd3.job.result else {
        return;
    };
    let run_dir = result.run_dir.clone();
    let structures = result.structures.clone();
    let specification = result.specification.clone();
    let command = result.command.clone();

    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Generated backbones").strong());
        ui.label(
            RichText::new(run_dir.display().to_string())
                .color(COLOR_INACTIVE)
                .small(),
        );
        if ui.button("Open folder").clicked()
            && let Err(error) = open_dir(&run_dir)
        {
            handle_err(&mut state.ui, error.to_string());
        }
    });

    if structures.is_empty() {
        ui.label(
            RichText::new(
                "RFdiffusion3 finished without writing any structures. Its log is in the run \
                 folder.",
            )
            .color(COLOR_INACTIVE),
        );
    }

    let mut to_open: Option<PathBuf> = None;
    ScrollArea::vertical()
        .id_salt("rfd3_results")
        .max_height(240.0)
        .show(ui, |ui| {
            for path in &structures {
                ui.horizontal(|ui| {
                    if ui
                        .button(RichText::new("Load").color(COLOR_HIGHLIGHT))
                        .clicked()
                    {
                        to_open = Some(path.clone());
                    }
                    ui.label(
                        RichText::new(
                            path.strip_prefix(&run_dir)
                                .unwrap_or(path)
                                .display()
                                .to_string(),
                        )
                        .monospace(),
                    );
                });
            }
        });

    if let Some(path) = to_open
        && let Err(error) = state.open_file(&path, scene, updates)
    {
        handle_err(&mut state.ui, error.to_string());
    }

    CollapsingHeader::new("Submitted InputSpecification")
        .id_salt("rfd3_specification")
        .show(ui, |ui| {
            ui.label(RichText::new(&specification).monospace().small());
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(specification.clone());
            }
        });
    CollapsingHeader::new("Command")
        .id_salt("rfd3_command")
        .show(ui, |ui| {
            ui.label(RichText::new(&command).monospace().small());
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(command.clone());
            }
        });
}

/// Write a number back into its text field, keeping a whole number whole so that a field the
/// upstream examples spell `3` does not become `3.0000000001` after a drag.
fn format_number(value: f64, step: Option<f64>) -> String {
    if step.is_none_or(|step| step.fract() == 0.0) && value.fract() == 0.0 {
        return format!("{value:.0}");
    }
    let text = format!("{value:.6}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_owned()
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_starts_from_the_bio_tools_contract() {
        let ui = Rfd3Ui::default();
        assert!(ui.contract.is_ok(), "{:?}", ui.contract.as_ref().err());
        assert_eq!(ui.input_mode, InputMode::Parameters);
        assert_eq!(ui.value("length"), "150");
        assert!(!ui.presets.is_empty());
    }

    #[test]
    fn loading_a_preset_replaces_every_field() {
        let mut window = Rfd3Ui::default();
        window.values.insert("contig".to_owned(), "A1-9".to_owned());

        let index = window
            .presets
            .iter()
            .position(|preset| preset.id == "unconditional_monomer")
            .expect("unconditional_monomer");
        window.apply_preset(index);

        // The preset has no contig, so the stale one must be gone rather than merged through.
        assert_eq!(window.value("contig"), "");
        assert_eq!(window.value("length"), "150");
        assert_eq!(window.structure_source, StructureSource::Opened);
    }

    #[test]
    fn a_preset_with_a_bundled_structure_uses_the_input_field() {
        let mut window = Rfd3Ui::default();
        let index = window
            .presets
            .iter()
            .position(|preset| preset.id == "demo/dsDNA_basic")
            .expect("demo/dsDNA_basic");
        window.apply_preset(index);

        assert_eq!(window.structure_source, StructureSource::Field);
        assert!(rfdiffusion3::is_bundled_structure(window.value("input")));
        assert_eq!(window.value("contig"), "A1-10,/0,B15-24,/0,120-130");
    }

    #[test]
    fn numbers_keep_the_spelling_the_examples_use() {
        assert_eq!(format_number(3.0, None), "3");
        assert_eq!(format_number(1.5, Some(0.1)), "1.5");
        assert_eq!(format_number(200.0, Some(1.0)), "200");
    }
}
