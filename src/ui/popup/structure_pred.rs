//! UI for OpenDDE all-atom structure prediction and co-folding.

use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use egui::{
    Align, Button, CollapsingHeader, Color32, ComboBox, Context, DragValue, Layout, Resize,
    RichText, ScrollArea, TextEdit, Ui,
};

use crate::{
    external_tools::{self, Tool},
    state::State,
    structure_prediction::{
        PredictionControl, StructurePredictionModel, StructurePredictionOutcome,
        opendde::{self, OpenDdeCovalentBond, OpenDdeEntity, OpenDdeRequest, OpenDdeRuntimeInfo},
        predict_structure_from_request,
    },
    ui::{COLOR_ACTION, COLOR_INACTIVE, ROW_SPACING, popup::close_btn},
    util::handle_err,
};

const LINUX_ONLY_COLOR: Color32 = Color32::from_rgb(218, 165, 32);

type OpenDdeProbeResult = Result<OpenDdeRuntimeInfo, String>;
type ToolActionResult = Result<(), String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolActionKind {
    Install,
    Uninstall,
}

impl ToolActionKind {
    fn present_participle(self) -> &'static str {
        match self {
            Self::Install => "Installing",
            Self::Uninstall => "Uninstalling",
        }
    }
}

#[derive(Debug)]
struct ActiveToolAction {
    tool: Tool,
    kind: ToolActionKind,
    receiver: mpsc::Receiver<ToolActionResult>,
}

#[derive(Debug, Default)]
enum OpenDdeRuntimeStatus {
    #[default]
    Unchecked,
    Checking,
    Ready(OpenDdeRuntimeInfo),
    Failed(String),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum PredictionEntityType {
    #[default]
    Protein,
    Dna,
    Rna,
    Ligand,
    Ion,
}

impl PredictionEntityType {
    fn name(self) -> &'static str {
        match self {
            Self::Protein => "Protein",
            Self::Dna => "DNA",
            Self::Rna => "RNA",
            Self::Ligand => "Small molecule / ligand",
            Self::Ion => "Ion",
        }
    }

    fn input_hint(self) -> &'static str {
        match self {
            Self::Protein => "One-letter protein sequence",
            Self::Dna => "DNA sequence",
            Self::Rna => "RNA sequence",
            Self::Ligand => "SMILES, CCD_ATP, or FILE_/absolute/path.sdf",
            Self::Ion => "Unprefixed CCD code, e.g. MG or ZN",
        }
    }

    fn input_help(self) -> &'static str {
        match self {
            Self::Protein => "20 standard amino-acid letters plus X; whitespace is ignored.",
            Self::Dna => "Supported letters: A, T, G, C, N, and X.",
            Self::Rna => "Supported letters: A, U, G, C, N, and X.",
            Self::Ligand => {
                "Accepts SMILES, CCD_ codes, multiple CCD codes joined by underscores, or FILE_ references."
            }
            Self::Ion => "Use the CCD component name without the CCD_ prefix.",
        }
    }

    fn is_sequence(self) -> bool {
        matches!(self, Self::Protein | Self::Dna | Self::Rna)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PredictionComponentInput {
    entity_type: PredictionEntityType,
    id: String,
    value: String,
}

impl PredictionComponentInput {
    fn new(entity_type: PredictionEntityType, id: impl Into<String>) -> Self {
        Self {
            entity_type,
            id: id.into(),
            value: String::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CovalentBondInput {
    entity1: usize,
    position1: usize,
    atom1: String,
    entity2: usize,
    position2: usize,
    atom2: String,
}

impl CovalentBondInput {
    fn new(entity_count: usize) -> Self {
        Self {
            entity1: 1,
            position1: 1,
            atom1: String::new(),
            entity2: entity_count.min(2).max(1),
            position2: 1,
            atom2: String::new(),
        }
    }

    fn to_open_dde(&self) -> OpenDdeCovalentBond {
        OpenDdeCovalentBond {
            entity1: self.entity1,
            copy1: 1,
            position1: self.position1,
            atom1: self.atom1.trim().to_owned(),
            entity2: self.entity2,
            copy2: 1,
            position2: self.position2,
            atom2: self.atom2.trim().to_owned(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct StructurePredUi {
    components: Vec<PredictionComponentInput>,
    covalent_bonds: Vec<CovalentBondInput>,
    pub model: StructurePredictionModel,
    started_at: Option<Instant>,
    control: Option<PredictionControl>,
    open_dde_runtime_status: OpenDdeRuntimeStatus,
    open_dde_runtime_rx: Option<mpsc::Receiver<OpenDdeProbeResult>>,
    cancel_requested: bool,
    tool_action: Option<ActiveToolAction>,
    tool_action_result: Option<(Tool, ToolActionKind, ToolActionResult)>,
    confirm_uninstall: Option<Tool>,
    completed_message: Option<String>,
}

impl Default for StructurePredUi {
    fn default() -> Self {
        Self {
            components: vec![PredictionComponentInput::new(
                PredictionEntityType::Protein,
                "A",
            )],
            covalent_bonds: Vec::new(),
            // OpenDDE is the currently supported backend for this window.
            model: StructurePredictionModel::OpenDDE,
            started_at: None,
            control: None,
            open_dde_runtime_status: OpenDdeRuntimeStatus::Unchecked,
            open_dde_runtime_rx: None,
            cancel_requested: false,
            tool_action: None,
            tool_action_result: None,
            confirm_uninstall: None,
            completed_message: None,
        }
    }
}

impl StructurePredUi {
    pub(crate) fn is_running(&self) -> bool {
        self.started_at.is_some()
    }

    fn ensure_open_dde_runtime_probe(&mut self, context: &Context) {
        if matches!(
            self.open_dde_runtime_status,
            OpenDdeRuntimeStatus::Unchecked
        ) {
            self.start_open_dde_runtime_probe(context);
        }
        self.poll_open_dde_runtime_probe();
    }

    fn refresh_open_dde_runtime_probe(&mut self, context: &Context) {
        if self.open_dde_runtime_rx.is_none() {
            self.start_open_dde_runtime_probe(context);
        }
    }

    fn start_open_dde_runtime_probe(&mut self, context: &Context) {
        let (tx, rx) = mpsc::channel();
        let context = context.clone();
        self.open_dde_runtime_status = OpenDdeRuntimeStatus::Checking;
        self.open_dde_runtime_rx = Some(rx);

        thread::spawn(move || {
            let result = opendde::probe_runtime().map_err(|error| error.to_string());
            let _ = tx.send(result);
            context.request_repaint();
        });
    }

    fn poll_open_dde_runtime_probe(&mut self) {
        let result = self
            .open_dde_runtime_rx
            .as_ref()
            .map(mpsc::Receiver::try_recv);
        match result {
            Some(Ok(Ok(info))) => {
                self.open_dde_runtime_rx = None;
                self.open_dde_runtime_status = OpenDdeRuntimeStatus::Ready(info);
            }
            Some(Ok(Err(error))) => {
                self.open_dde_runtime_rx = None;
                self.open_dde_runtime_status = OpenDdeRuntimeStatus::Failed(error);
            }
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                self.open_dde_runtime_rx = None;
                self.open_dde_runtime_status = OpenDdeRuntimeStatus::Failed(
                    "OpenDDE diagnostics stopped before returning a result".to_owned(),
                );
            }
            Some(Err(mpsc::TryRecvError::Empty)) | None => {}
        }
    }

    pub(crate) fn finish_prediction(&mut self) {
        self.started_at = None;
        self.control = None;
        self.cancel_requested = false;
    }

    fn start_prediction(&mut self, control: PredictionControl) {
        self.started_at = Some(Instant::now());
        self.control = Some(control);
        self.cancel_requested = false;
        self.completed_message = None;
    }

    pub(crate) fn mark_complete(&mut self, message: String) {
        self.completed_message = Some(message);
    }

    fn cancel_prediction(&mut self) {
        if let Some(control) = &self.control {
            control.cancel();
            self.cancel_requested = true;
        }
    }

    fn tool_action_is_running(&self) -> bool {
        self.tool_action.is_some()
    }

    fn start_tool_action(&mut self, kind: ToolActionKind, context: &Context) {
        if self.is_running() || self.tool_action_is_running() {
            return;
        }

        let tool = self.model.tool();
        let spec = tool.spec();
        let allowed = spec.platform.is_supported()
            && match kind {
                ToolActionKind::Install => spec.can_install_here(),
                ToolActionKind::Uninstall => spec.molchanica_managed,
            };
        if !allowed {
            return;
        }

        let (tx, receiver) = mpsc::channel();
        self.tool_action = Some(ActiveToolAction {
            tool,
            kind,
            receiver,
        });
        self.tool_action_result = None;
        self.confirm_uninstall = None;

        let context = context.clone();
        thread::spawn(move || {
            let result = match kind {
                ToolActionKind::Install => external_tools::install(tool),
                ToolActionKind::Uninstall => external_tools::uninstall(tool),
            }
            .map_err(|error| error.to_string());
            let _ = tx.send(result);
            context.request_repaint();
        });
    }

    fn poll_tool_action(&mut self) {
        let Some(action) = &self.tool_action else {
            return;
        };
        let tool = action.tool;
        let kind = action.kind;
        let result = match action.receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err(format!(
                "{} stopped without reporting a result",
                kind.present_participle()
            )),
        };

        if tool == Tool::OpenDde && result.is_ok() {
            self.open_dde_runtime_status = OpenDdeRuntimeStatus::Unchecked;
            self.open_dde_runtime_rx = None;
        }
        self.tool_action_result = Some((tool, kind, result));
        self.tool_action = None;
    }
}

pub(in crate::ui) fn structure_prediction_window(state: &mut State, ui: &mut Ui) {
    state.ui.structure_pred.poll_tool_action();
    let model = state.ui.structure_pred.model;

    ui.horizontal(|ui| {
        ui.heading("Structure prediction and co-folding");
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            close_btn(ui, &mut state.ui.popup.structure_pred);
        });
    });
    ui.horizontal(|ui| {
        ui.label("All-atom prediction powered by");
        ui.hyperlink_to(model.label(), model.tool().spec().url);
    });

    if model == StructurePredictionModel::OpenDDE {
        state
            .ui
            .structure_pred
            .ensure_open_dde_runtime_probe(ui.ctx());
    }

    let elapsed = state
        .ui
        .structure_pred
        .started_at
        .map(|started_at| started_at.elapsed());
    if model == StructurePredictionModel::OpenDDE {
        open_dde_runtime_status_ui(&mut state.ui.structure_pred, elapsed, ui);
    } else if let Some(elapsed) = elapsed {
        ui.label(format!("Time running: {}", format_elapsed(elapsed)));
    }

    if let Some(elapsed) = elapsed {
        if state.ui.structure_pred.cancel_requested {
            ui.label("Cancellation requested...");
        }

        // The worker requests an immediate repaint when it sends its result. This timer is only
        // responsible for advancing the elapsed-time display while the process is otherwise quiet.
        let seconds_to_next_update = 30 - elapsed.as_secs() % 30;
        ui.ctx()
            .request_repaint_after(Duration::from_secs(seconds_to_next_update));
    } else if let Some(message) = &state.ui.structure_pred.completed_message {
        ui.label(RichText::new(message).color(Color32::WHITE));
    }
    ui.add_space(ROW_SPACING);

    let prediction_running = state.ui.structure_pred.is_running();
    let tool_action_running = state.ui.structure_pred.tool_action_is_running();
    ui.add_enabled_ui(!prediction_running && !tool_action_running, |ui| {
        ui.horizontal(|ui| {
            ui.label("Model:");
            ComboBox::from_id_salt("structure_prediction_model")
                .selected_text(model_name(state.ui.structure_pred.model))
                .show_ui(ui, |ui| {
                    for model in StructurePredictionModel::ALL {
                        ui.selectable_value(
                            &mut state.ui.structure_pred.model,
                            model,
                            model_name(model),
                        );
                    }
                });
        });
    });

    selected_model_tool_ui(&mut state.ui.structure_pred, prediction_running, ui);

    ui.add_enabled_ui(!prediction_running && !tool_action_running, |ui| {
        ui.add_space(ROW_SPACING);
        ui.label(
            "Each component below is one chain or molecule. Add multiple components to predict a complex.",
        );
        ui.add_space(ROW_SPACING);
        ScrollArea::vertical()
            .max_height(560.0)
            .auto_shrink([false, true])
            .show(ui, |ui| prediction_inputs(&mut state.ui.structure_pred, ui));
    });

    ui.add_space(ROW_SPACING);
    ui.horizontal(|ui| {
        let model_supported = state
            .ui
            .structure_pred
            .model
            .tool()
            .spec()
            .platform
            .is_supported();
        if prediction_running {
            if ui
                .add_enabled(
                    !state.ui.structure_pred.cancel_requested,
                    Button::new(RichText::new("Cancel").color(COLOR_ACTION)),
                )
                .clicked()
            {
                state.ui.structure_pred.cancel_prediction();
            }
        } else {
            let response = ui.add_enabled(
                model_supported && !tool_action_running,
                Button::new(RichText::new("Pred struct").color(COLOR_ACTION)),
            );
            let clicked = response.clicked();
            if !model_supported {
                response.on_hover_text("Linux only; cannot run on this operating system.");
            }
            if clicked {
                predict(state, ui.ctx());
            }
        }
    });
}

fn selected_model_tool_ui(
    prediction_ui: &mut StructurePredUi,
    prediction_running: bool,
    ui: &mut Ui,
) {
    let tool = prediction_ui.model.tool();
    let spec = tool.spec();
    if !spec.platform.is_supported() {
        ui.label(
            RichText::new("Linux only; can't install or run.")
                .color(LINUX_ONLY_COLOR)
                .strong(),
        );
        return;
    }

    let installed = external_tools::is_installed(tool);
    let managed_install_present = spec.molchanica_managed
        && match spec.kind {
            external_tools::ToolKind::VenvScript | external_tools::ToolKind::VenvPython => {
                spec.venv_root().is_some_and(|root| root.is_dir())
            }
            external_tools::ToolKind::Executable => installed,
        };
    let active_kind = prediction_ui
        .tool_action
        .as_ref()
        .filter(|action| action.tool == tool)
        .map(|action| action.kind);
    ui.horizontal(|ui| {
        if let Some(kind) = active_kind {
            ui.spinner();
            ui.label(
                RichText::new(format!("{} {}...", kind.present_participle(), spec.name))
                    .color(COLOR_ACTION),
            );
        } else if managed_install_present {
            let confirming = prediction_ui.confirm_uninstall == Some(tool);
            let label = if confirming {
                "Confirm uninstall"
            } else {
                "Uninstall"
            };
            let response = ui.add_enabled(
                !prediction_running && !prediction_ui.tool_action_is_running(),
                Button::new(label),
            );
            if response.clicked() {
                if confirming {
                    prediction_ui.start_tool_action(ToolActionKind::Uninstall, ui.ctx());
                } else {
                    prediction_ui.confirm_uninstall = Some(tool);
                }
            }
        } else if !installed && spec.can_install_here() {
            let response = ui.add_enabled(
                !prediction_running && !prediction_ui.tool_action_is_running(),
                Button::new("Install"),
            );
            if response.clicked() {
                prediction_ui.start_tool_action(ToolActionKind::Install, ui.ctx());
            }
        } else if !installed {
            ui.label(RichText::new(spec.install_hint).color(COLOR_INACTIVE));
        }

        ui.label(
            RichText::new(if installed {
                format!("{} is installed", spec.name)
            } else if managed_install_present {
                format!("{} setup is incomplete", spec.name)
            } else {
                format!("{} is not installed", spec.name)
            })
            .color(if installed {
                Color32::LIGHT_GREEN
            } else if managed_install_present {
                Color32::ORANGE
            } else {
                COLOR_INACTIVE
            }),
        );
    });

    if let Some((result_tool, kind, result)) = &prediction_ui.tool_action_result
        && *result_tool == tool
    {
        let action_name = match kind {
            ToolActionKind::Install => "Installation",
            ToolActionKind::Uninstall => "Uninstallation",
        };
        match result {
            Ok(()) => {
                ui.label(
                    RichText::new(format!("{action_name} completed.")).color(Color32::LIGHT_GREEN),
                );
            }
            Err(error) => {
                ui.label(
                    RichText::new(format!("{action_name} failed: {error}")).color(Color32::ORANGE),
                );
            }
        }
    }
}

fn open_dde_runtime_status_ui(
    prediction_ui: &mut StructurePredUi,
    elapsed: Option<Duration>,
    ui: &mut Ui,
) {
    let running = elapsed.is_some();
    let mut refresh_requested = false;
    ui.horizontal(|ui| {
        if let Some(elapsed) = elapsed {
            ui.label(format!("Time running: {}", format_elapsed(elapsed)));
            ui.separator();
        }

        match &prediction_ui.open_dde_runtime_status {
            OpenDdeRuntimeStatus::Unchecked | OpenDdeRuntimeStatus::Checking => {
                ui.label(RichText::new("Checking OpenDDE compute...").color(COLOR_INACTIVE));
            }
            OpenDdeRuntimeStatus::Ready(info) => {
                let label = if running {
                    format!("Running on {}", info.short_label())
                } else {
                    format!("Compute: {}", info.short_label())
                };
                ui.label(RichText::new(label).color(Color32::LIGHT_GREEN))
                    .on_hover_text(info.diagnostics_text());
            }
            OpenDdeRuntimeStatus::Failed(error) => {
                ui.label(
                    RichText::new("OpenDDE compute details unavailable").color(Color32::LIGHT_RED),
                )
                .on_hover_text(error);
            }
        }

        if !running
            && !matches!(
                prediction_ui.open_dde_runtime_status,
                OpenDdeRuntimeStatus::Checking
            )
            && ui.small_button("Refresh").clicked()
        {
            refresh_requested = true;
        }
    });

    match &prediction_ui.open_dde_runtime_status {
        OpenDdeRuntimeStatus::Ready(info) => {
            let details = info.detail_label();
            if !details.is_empty() {
                ui.label(RichText::new(details).color(COLOR_INACTIVE))
                    .on_hover_text(info.diagnostics_text());
            }
        }
        OpenDdeRuntimeStatus::Failed(error) => {
            ui.label(RichText::new(error).color(COLOR_INACTIVE));
        }
        OpenDdeRuntimeStatus::Unchecked | OpenDdeRuntimeStatus::Checking => {}
    }

    if refresh_requested {
        prediction_ui.refresh_open_dde_runtime_probe(ui.ctx());
    }
}

fn prediction_inputs(prediction_ui: &mut StructurePredUi, ui: &mut Ui) {
    let component_count = prediction_ui.components.len();
    let mut component_to_remove = None;

    for (index, component) in prediction_ui.components.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Component {}", index + 1)).strong());
                ComboBox::from_id_salt(("structure_prediction_entity_type", index))
                    .selected_text(component.entity_type.name())
                    .show_ui(ui, |ui| {
                        for entity_type in [
                            PredictionEntityType::Protein,
                            PredictionEntityType::Dna,
                            PredictionEntityType::Rna,
                            PredictionEntityType::Ligand,
                            PredictionEntityType::Ion,
                        ] {
                            ui.selectable_value(
                                &mut component.entity_type,
                                entity_type,
                                entity_type.name(),
                            );
                        }
                    });

                ui.label("Chain ID:");
                ui.add(
                    TextEdit::singleline(&mut component.id)
                        .desired_width(50.0)
                        .hint_text("A"),
                );

                if component_count > 1
                    && ui
                        .button(RichText::new("Remove").color(Color32::LIGHT_RED))
                        .clicked()
                {
                    component_to_remove = Some(index);
                }
            });

            if component.entity_type.is_sequence() {
                Resize::default()
                    .id_salt(("structure_prediction_sequence_input", index))
                    .default_size([520.0, 56.0])
                    .min_size([160.0, 36.0])
                    .with_stroke(false)
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            TextEdit::multiline(&mut component.value)
                                .desired_rows(1)
                                .desired_width(f32::INFINITY)
                                .hint_text(component.entity_type.input_hint()),
                        );
                    });
            } else {
                ui.add(
                    TextEdit::singleline(&mut component.value)
                        .desired_width(520.0)
                        .hint_text(component.entity_type.input_hint()),
                );
            }
            ui.label(RichText::new(component.entity_type.input_help()).color(COLOR_INACTIVE));
        });
        ui.add_space(ROW_SPACING);
    }

    if let Some(index) = component_to_remove {
        remove_component_and_reindex(prediction_ui, index);
    }

    let mut component_to_add = None;
    ui.horizontal(|ui| {
        ui.label("Add:");
        for (label, entity_type) in [
            ("Protein", PredictionEntityType::Protein),
            ("DNA", PredictionEntityType::Dna),
            ("RNA", PredictionEntityType::Rna),
            ("Ligand", PredictionEntityType::Ligand),
            ("Ion", PredictionEntityType::Ion),
        ] {
            if ui.button(format!("+ {label}")).clicked() {
                component_to_add = Some(entity_type);
            }
        }
    });

    if let Some(entity_type) = component_to_add {
        let id = next_chain_id(&prediction_ui.components);
        prediction_ui
            .components
            .push(PredictionComponentInput::new(entity_type, id));
    }

    ui.add_space(ROW_SPACING);
    covalent_bond_inputs(prediction_ui, ui);
}

fn remove_component_and_reindex(prediction_ui: &mut StructurePredUi, index: usize) {
    prediction_ui.components.remove(index);
    let removed_entity = index + 1;
    prediction_ui
        .covalent_bonds
        .retain(|bond| bond.entity1 != removed_entity && bond.entity2 != removed_entity);
    for bond in &mut prediction_ui.covalent_bonds {
        if bond.entity1 > removed_entity {
            bond.entity1 -= 1;
        }
        if bond.entity2 > removed_entity {
            bond.entity2 -= 1;
        }
    }
}

fn covalent_bond_inputs(prediction_ui: &mut StructurePredUi, ui: &mut Ui) {
    CollapsingHeader::new("Covalent bonds (optional)")
        .default_open(!prediction_ui.covalent_bonds.is_empty())
        .show(ui, |ui| {
            ui.label(
                "Entity and position references are 1-based. Copy number is 1 for each component.",
            );

            let entity_count = prediction_ui.components.len().max(1);
            let mut bond_to_remove = None;
            for (index, bond) in prediction_ui.covalent_bonds.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Bond {}", index + 1)).strong());
                        if ui
                            .button(RichText::new("Remove").color(Color32::LIGHT_RED))
                            .clicked()
                        {
                            bond_to_remove = Some(index);
                        }
                    });
                    bond_endpoint_input(
                        ui,
                        "From",
                        &mut bond.entity1,
                        &mut bond.position1,
                        &mut bond.atom1,
                        entity_count,
                    );
                    bond_endpoint_input(
                        ui,
                        "To",
                        &mut bond.entity2,
                        &mut bond.position2,
                        &mut bond.atom2,
                        entity_count,
                    );
                });
            }

            if let Some(index) = bond_to_remove {
                prediction_ui.covalent_bonds.remove(index);
            }
            if ui.button("+ Covalent bond").clicked() {
                prediction_ui
                    .covalent_bonds
                    .push(CovalentBondInput::new(entity_count));
            }
        });
}

fn bond_endpoint_input(
    ui: &mut Ui,
    label: &str,
    entity: &mut usize,
    position: &mut usize,
    atom: &mut String,
    entity_count: usize,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.label("entity");
        ui.add(DragValue::new(entity).range(1..=entity_count));
        ui.label("position");
        ui.add(DragValue::new(position).range(1..=usize::MAX));
        ui.label("atom");
        ui.add(
            TextEdit::singleline(atom)
                .desired_width(65.0)
                .hint_text("SG"),
        );
    });
}

fn next_chain_id(components: &[PredictionComponentInput]) -> String {
    for index in 0.. {
        let candidate = if index < 26 {
            char::from(b'A' + index as u8).to_string()
        } else {
            format!("C{}", index - 25)
        };
        if components
            .iter()
            .all(|component| component.id.trim() != candidate)
        {
            return candidate;
        }
    }
    unreachable!("the chain-ID generator has an unbounded alphanumeric namespace")
}

fn build_request(prediction_ui: &StructurePredUi) -> Result<OpenDdeRequest, String> {
    let entities = prediction_ui
        .components
        .iter()
        .map(|component| {
            let id = component.id.trim().to_owned();
            let sequence = || compact_sequence(&component.value);
            match component.entity_type {
                PredictionEntityType::Protein => OpenDdeEntity::protein_sequence(id, sequence()),
                PredictionEntityType::Dna => OpenDdeEntity::dna_sequence(id, sequence()),
                PredictionEntityType::Rna => OpenDdeEntity::rna(id, sequence()),
                PredictionEntityType::Ligand => OpenDdeEntity::ligand(id, component.value.trim()),
                PredictionEntityType::Ion => {
                    OpenDdeEntity::ion(id, component.value.trim().to_ascii_uppercase())
                }
            }
        })
        .collect();

    let mut request = OpenDdeRequest::new("molchanica_opendde_prediction", entities);
    request.covalent_bonds = prediction_ui
        .covalent_bonds
        .iter()
        .map(CovalentBondInput::to_open_dde)
        .collect();
    request.validate().map_err(|error| error.to_string())?;
    Ok(request)
}

fn compact_sequence(sequence: &str) -> String {
    sequence
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_uppercase)
        .collect()
}

fn predict(state: &mut State, context: &Context) {
    let request = match build_request(&state.ui.structure_pred) {
        Ok(request) => request,
        Err(error) => {
            handle_err(
                &mut state.ui,
                format!("Unable to start structure prediction: {error}"),
            );
            return;
        }
    };

    let Some(ff_map) = state.ff_param_set.peptide_ff_q_map.clone() else {
        handle_err(
            &mut state.ui,
            "Unable to start structure prediction: no peptide force-field parameter map is loaded"
                .to_owned(),
        );
        return;
    };

    let model = state.ui.structure_pred.model;
    let control = PredictionControl::default();
    let worker_control = control.clone();
    let (tx, rx) = mpsc::channel();
    let context = context.clone();

    state.volatile.thread_receivers.structure_prediction = Some(rx);
    state.ui.structure_pred.start_prediction(control);

    thread::spawn(move || {
        let prediction = predict_structure_from_request(model, &request, &ff_map, &worker_control);

        let outcome = if worker_control.is_cancel_requested() {
            StructurePredictionOutcome::Cancelled
        } else {
            match prediction {
                Ok(molecule) => StructurePredictionOutcome::Complete(molecule),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                    StructurePredictionOutcome::Cancelled
                }
                Err(error) => StructurePredictionOutcome::Failed(error.to_string()),
            }
        };
        let _ = tx.send(outcome);
        context.request_repaint();
    });
}

fn format_elapsed(elapsed: Duration) -> String {
    let total_seconds = elapsed.as_secs();
    let hours = total_seconds / 3_600;
    let minutes = total_seconds % 3_600 / 60;
    let seconds = total_seconds % 60;
    if hours == 0 {
        format!("{minutes:02}:{seconds:02}")
    } else {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    }
}

fn model_name(model: StructurePredictionModel) -> &'static str {
    model.label()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_mixed_complex_request() {
        let mut prediction_ui = StructurePredUi {
            components: vec![
                PredictionComponentInput {
                    entity_type: PredictionEntityType::Protein,
                    id: "A".to_owned(),
                    value: "acD\nEF".to_owned(),
                },
                PredictionComponentInput {
                    entity_type: PredictionEntityType::Rna,
                    id: "R".to_owned(),
                    value: "gu ac".to_owned(),
                },
                PredictionComponentInput {
                    entity_type: PredictionEntityType::Ligand,
                    id: "L".to_owned(),
                    value: "CCD_ATP".to_owned(),
                },
                PredictionComponentInput {
                    entity_type: PredictionEntityType::Ion,
                    id: "M".to_owned(),
                    value: "mg".to_owned(),
                },
            ],
            ..Default::default()
        };
        prediction_ui.covalent_bonds.push(CovalentBondInput {
            entity1: 1,
            position1: 2,
            atom1: "SG".to_owned(),
            entity2: 3,
            position2: 1,
            atom2: "C1".to_owned(),
        });

        let request = build_request(&prediction_ui).expect("mixed complex should be valid");

        assert_eq!(
            request.entities,
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDEF"),
                OpenDdeEntity::rna("R", "GUAC"),
                OpenDdeEntity::ligand("L", "CCD_ATP"),
                OpenDdeEntity::ion("M", "MG"),
            ]
        );
        assert_eq!(request.covalent_bonds.len(), 1);
    }

    #[test]
    fn removing_a_component_reindexes_or_removes_bonds() {
        let mut prediction_ui = StructurePredUi {
            components: vec![
                PredictionComponentInput::new(PredictionEntityType::Protein, "A"),
                PredictionComponentInput::new(PredictionEntityType::Ligand, "L"),
                PredictionComponentInput::new(PredictionEntityType::Ion, "M"),
            ],
            covalent_bonds: vec![
                CovalentBondInput {
                    entity1: 1,
                    position1: 1,
                    atom1: "CA".to_owned(),
                    entity2: 3,
                    position2: 1,
                    atom2: "MG".to_owned(),
                },
                CovalentBondInput {
                    entity1: 1,
                    position1: 1,
                    atom1: "SG".to_owned(),
                    entity2: 2,
                    position2: 1,
                    atom2: "C1".to_owned(),
                },
            ],
            ..Default::default()
        };

        remove_component_and_reindex(&mut prediction_ui, 1);

        assert_eq!(prediction_ui.covalent_bonds.len(), 1);
        assert_eq!(prediction_ui.covalent_bonds[0].entity2, 2);
    }
}
