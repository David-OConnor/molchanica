//! Dedicated ProteinMPNN sequence-prediction window.

use std::{
    path::PathBuf,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use egui::{
    Align, Button, CollapsingHeader, Color32, ComboBox, DragValue, Layout, RichText, ScrollArea,
    TextEdit, Ui,
};
use mol_defs::molecules::peptide::MoleculePeptide;

use crate::{
    external_tools::{
        self, Tool,
        mpnn::{self, DesignRequest, DesignResult, MpnnModel, ProteinMpnnCheckpoint},
    },
    state::State,
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, popup::close_btn},
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SequencePredictionModel {
    #[default]
    ProteinMpnn,
}

impl SequencePredictionModel {
    const ALL: [Self; 1] = [Self::ProteinMpnn];

    fn label(self) -> &'static str {
        match self {
            Self::ProteinMpnn => "ProteinMPNN",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ProteinSource {
    #[default]
    Opened,
    Disk,
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

struct PredictionJob {
    receiver: Option<mpsc::Receiver<Result<DesignResult, String>>>,
    started_at: Option<Instant>,
    result: Option<DesignResult>,
    error: Option<String>,
}

impl Default for PredictionJob {
    fn default() -> Self {
        Self {
            receiver: None,
            started_at: None,
            result: None,
            error: None,
        }
    }
}

impl PredictionJob {
    fn is_running(&self) -> bool {
        self.receiver.is_some()
    }

    fn start(
        &mut self,
        context: &egui::Context,
        work: impl FnOnce() -> Result<DesignResult, String> + Send + 'static,
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
                    Some("The ProteinMPNN worker stopped without returning a result.".to_owned());
                self.receiver = None;
                self.started_at = None;
            }
        }
    }
}

struct ToolAction {
    kind: ToolActionKind,
    receiver: mpsc::Receiver<Result<(), String>>,
}

pub(crate) struct SequencePredUi {
    model: SequencePredictionModel,
    source: ProteinSource,
    opened_protein: usize,
    disk_path: Option<PathBuf>,
    chains: String,
    designed_residues: String,
    homo_oligomer: bool,
    checkpoint: ProteinMpnnCheckpoint,
    num_sequences: usize,
    temperature: f32,
    backbone_noise: f32,
    seed: u64,
    omit_amino_acids: String,
    bias_amino_acids: String,
    bias_amino_acids_per_residue: String,
    omit_amino_acids_per_residue: String,
    job: PredictionJob,
    tool_action: Option<ToolAction>,
    tool_action_result: Option<(ToolActionKind, Result<(), String>)>,
    confirm_uninstall: bool,
}

impl Default for SequencePredUi {
    fn default() -> Self {
        Self {
            model: SequencePredictionModel::default(),
            source: ProteinSource::default(),
            opened_protein: 0,
            disk_path: None,
            chains: "A".to_owned(),
            designed_residues: String::new(),
            homo_oligomer: false,
            checkpoint: ProteinMpnnCheckpoint::default(),
            num_sequences: 4,
            temperature: 0.1,
            backbone_noise: 0.0,
            seed: 0,
            omit_amino_acids: "X".to_owned(),
            bias_amino_acids: String::new(),
            bias_amino_acids_per_residue: String::new(),
            omit_amino_acids_per_residue: String::new(),
            job: PredictionJob::default(),
            tool_action: None,
            tool_action_result: None,
            confirm_uninstall: false,
        }
    }
}

impl SequencePredUi {
    fn request(&self) -> DesignRequest {
        DesignRequest {
            model: MpnnModel::ProteinMpnn,
            chains_to_design: split_words(&self.chains),
            fixed_residues: Vec::new(),
            num_sequences: self.num_sequences,
            temperature: self.temperature,
            seed: self.seed,
            checkpoint: self.checkpoint,
            backbone_noise: self.backbone_noise,
            omit_amino_acids: self.omit_amino_acids.clone(),
            designed_residues: self.designed_residues.clone(),
            homo_oligomer: self.homo_oligomer,
            bias_amino_acids: self.bias_amino_acids.clone(),
            bias_amino_acids_per_residue: self.bias_amino_acids_per_residue.clone(),
            omit_amino_acids_per_residue: self.omit_amino_acids_per_residue.clone(),
        }
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
                ToolActionKind::Install => external_tools::install(Tool::ProteinMpnn),
                ToolActionKind::Uninstall => external_tools::uninstall(Tool::ProteinMpnn),
            }
            .map_err(|error| error.to_string());
            let _ = tx.send(result);
            context.request_repaint();
        });
    }
}

pub(in crate::ui) fn sequence_prediction_window(state: &mut State, ui: &mut Ui) {
    if let Some(path) = state.volatile.dialogs.sequence_prediction.take_picked() {
        state.ui.sequence_pred.disk_path = Some(path);
        state.ui.sequence_pred.source = ProteinSource::Disk;
    }

    let protein_names = state
        .peptides
        .iter()
        .map(|protein| protein.common.ident.clone())
        .collect::<Vec<_>>();
    if state.ui.sequence_pred.opened_protein >= protein_names.len() {
        state.ui.sequence_pred.opened_protein = state
            .peptide_for_tools_i()
            .filter(|index| *index < protein_names.len())
            .unwrap_or(0);
    }

    state.ui.sequence_pred.job.poll();
    state.ui.sequence_pred.poll_tool_action();

    ui.horizontal(|ui| {
        ui.heading("Protein sequence prediction");
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            close_btn(ui, &mut state.ui.popup.sequence_pred);
        });
    });
    ui.label("Predict amino-acid sequences for a fixed protein backbone.");
    ui.add_space(ROW_SPACING);

    let running = state.ui.sequence_pred.job.is_running();
    let managing_tool = state.ui.sequence_pred.tool_action.is_some();
    let mut pick_file = false;
    let mut run_prediction = false;

    ui.add_enabled_ui(!running && !managing_tool, |ui| {
        ui.horizontal(|ui| {
            ui.label("Model:");
            ComboBox::from_id_salt("sequence_prediction_model")
                .selected_text(state.ui.sequence_pred.model.label())
                .show_ui(ui, |ui| {
                    for model in SequencePredictionModel::ALL {
                        ui.selectable_value(
                            &mut state.ui.sequence_pred.model,
                            model,
                            model.label(),
                        );
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Protein:");
            ui.radio_value(
                &mut state.ui.sequence_pred.source,
                ProteinSource::Opened,
                "Opened",
            );
            ui.radio_value(
                &mut state.ui.sequence_pred.source,
                ProteinSource::Disk,
                "From disk",
            );
        });

        match state.ui.sequence_pred.source {
            ProteinSource::Opened => {
                ui.horizontal(|ui| {
                    ui.label("Opened protein:");
                    let selected = protein_names
                        .get(state.ui.sequence_pred.opened_protein)
                        .map(String::as_str)
                        .unwrap_or("No proteins are open");
                    ComboBox::from_id_salt("sequence_prediction_opened_protein")
                        .selected_text(selected)
                        .show_ui(ui, |ui| {
                            for (index, name) in protein_names.iter().enumerate() {
                                ui.selectable_value(
                                    &mut state.ui.sequence_pred.opened_protein,
                                    index,
                                    name,
                                );
                            }
                        });
                });
            }
            ProteinSource::Disk => {
                ui.horizontal(|ui| {
                    if ui.button("Choose mmCIF…").clicked() {
                        pick_file = true;
                    }
                    ui.label(
                        state
                            .ui
                            .sequence_pred
                            .disk_path
                            .as_deref()
                            .and_then(|path| path.file_name())
                            .and_then(|name| name.to_str())
                            .unwrap_or("No file selected"),
                    );
                });
            }
        }

        ui.add_space(ROW_SPACING);
        ui.horizontal(|ui| {
            ui.label("Chains to design:");
            ui.text_edit_singleline(&mut state.ui.sequence_pred.chains)
                .on_hover_text("Comma- or space-separated chain IDs. Leave blank to design all protein chains.");
        });
        ui.label("Designed residues (optional; one chain per line)");
        ui.add(
            TextEdit::multiline(&mut state.ui.sequence_pred.designed_residues)
                .desired_rows(2)
                .hint_text("B 26 27 28 29 30"),
        )
        .on_hover_text("When a chain is listed, positions not listed for it are held fixed.");
        ui.checkbox(
            &mut state.ui.sequence_pred.homo_oligomer,
            "Tie designed chains as a homo-oligomer",
        );

        ui.horizontal(|ui| {
            ui.label("Checkpoint:");
            ComboBox::from_id_salt("sequence_prediction_checkpoint")
                .selected_text(state.ui.sequence_pred.checkpoint.name())
                .show_ui(ui, |ui| {
                    for checkpoint in ProteinMpnnCheckpoint::ALL {
                        ui.selectable_value(
                            &mut state.ui.sequence_pred.checkpoint,
                            checkpoint,
                            checkpoint.name(),
                        );
                    }
                });
            ui.label("Sequences:");
            ui.add(DragValue::new(&mut state.ui.sequence_pred.num_sequences).range(1..=1_000));
        });
        ui.horizontal(|ui| {
            ui.label("Temperature:");
            ui.add(
                DragValue::new(&mut state.ui.sequence_pred.temperature)
                    .speed(0.01)
                    .range(0.01..=1.0),
            );
            ui.label("Backbone noise:");
            ui.add(
                DragValue::new(&mut state.ui.sequence_pred.backbone_noise)
                    .speed(0.01)
                    .range(0.0..=1.0),
            );
            ui.label("Seed:");
            ui.add(DragValue::new(&mut state.ui.sequence_pred.seed));
        });
        ui.horizontal(|ui| {
            ui.label("Omit amino acids:");
            ui.text_edit_singleline(&mut state.ui.sequence_pred.omit_amino_acids);
        });

        CollapsingHeader::new("Advanced sampling biases").show(ui, |ui| {
            ui.label("Global amino-acid bias");
            ui.text_edit_singleline(&mut state.ui.sequence_pred.bias_amino_acids)
                .on_hover_text("Example: W:3.0,P:3.0,C:3.0,A:-3.0");
            ui.label("Per-residue amino-acid bias (JSON)");
            ui.add(
                TextEdit::multiline(&mut state.ui.sequence_pred.bias_amino_acids_per_residue)
                    .desired_rows(2)
                    .hint_text(r#"{"C1":{"G":-0.3,"P":10.8}}"#),
            );
            ui.label("Per-residue omitted amino acids (JSON)");
            ui.add(
                TextEdit::multiline(&mut state.ui.sequence_pred.omit_amino_acids_per_residue)
                    .desired_rows(2)
                    .hint_text(r#"{"A1":"CP","A2":"CW"}"#),
            );
        });
    });

    if pick_file {
        state.volatile.dialogs.sequence_prediction.pick_file();
    }

    ui.add_space(ROW_SPACING);
    tool_controls(&mut state.ui.sequence_pred, running, ui);

    let input_ready = match state.ui.sequence_pred.source {
        ProteinSource::Opened => state.ui.sequence_pred.opened_protein < state.peptides.len(),
        ProteinSource::Disk => state.ui.sequence_pred.disk_path.is_some(),
    };
    ui.horizontal(|ui| {
        if running {
            ui.spinner();
            ui.label(RichText::new("Predicting sequence…").color(COLOR_ACTION));
            if let Some(started_at) = state.ui.sequence_pred.job.started_at {
                ui.label(format_elapsed(started_at.elapsed()));
                ui.ctx().request_repaint_after(Duration::from_secs(1));
            }
        } else if ui
            .add_enabled(
                input_ready && !managing_tool && external_tools::is_installed(Tool::ProteinMpnn),
                Button::new(RichText::new("Predict sequence").color(COLOR_ACTION)),
            )
            .clicked()
        {
            run_prediction = true;
        }
    });

    if run_prediction {
        start_prediction(state, ui.ctx());
    }

    if let Some(error) = &state.ui.sequence_pred.job.error {
        ui.label(RichText::new(error).color(Color32::LIGHT_RED));
    }
    if let Some(result) = &state.ui.sequence_pred.job.result {
        show_result(result, ui);
    }
}

fn tool_controls(prediction_ui: &mut SequencePredUi, prediction_running: bool, ui: &mut Ui) {
    let installed = external_tools::is_installed(Tool::ProteinMpnn);
    ui.horizontal(|ui| {
        if let Some(action) = &prediction_ui.tool_action {
            ui.spinner();
            ui.label(format!("{} ProteinMPNN…", action.kind.progress()));
        } else if installed {
            let label = if prediction_ui.confirm_uninstall {
                "Confirm uninstall"
            } else {
                "Uninstall"
            };
            if ui
                .add_enabled(!prediction_running, Button::new(label))
                .clicked()
            {
                if prediction_ui.confirm_uninstall {
                    prediction_ui.start_tool_action(ToolActionKind::Uninstall, ui.ctx());
                } else {
                    prediction_ui.confirm_uninstall = true;
                }
            }
            ui.label(RichText::new("ProteinMPNN is installed").color(Color32::LIGHT_GREEN));
        } else {
            if ui
                .add_enabled(!prediction_running, Button::new("Install"))
                .clicked()
            {
                prediction_ui.start_tool_action(ToolActionKind::Install, ui.ctx());
            }
            ui.label(RichText::new("ProteinMPNN is not installed").color(COLOR_INACTIVE));
        }
    });
    if let Some((kind, result)) = &prediction_ui.tool_action_result {
        match result {
            Ok(()) => ui.label(
                RichText::new(match kind {
                    ToolActionKind::Install => "ProteinMPNN installation completed.",
                    ToolActionKind::Uninstall => "ProteinMPNN was uninstalled.",
                })
                .color(Color32::LIGHT_GREEN),
            ),
            Err(error) => ui.label(RichText::new(error).color(Color32::LIGHT_RED)),
        };
    }
}

fn start_prediction(state: &mut State, context: &egui::Context) {
    let request = state.ui.sequence_pred.request();
    match state.ui.sequence_pred.source {
        ProteinSource::Opened => {
            let Some(protein) = state
                .peptides
                .get(state.ui.sequence_pred.opened_protein)
                .cloned()
            else {
                state.ui.sequence_pred.job.error = Some("Select an opened protein.".to_owned());
                return;
            };
            state.ui.sequence_pred.job.start(context, move || {
                predict_opened_protein(&protein, &request).map_err(|error| error.to_string())
            });
        }
        ProteinSource::Disk => {
            let Some(path) = state.ui.sequence_pred.disk_path.clone() else {
                state.ui.sequence_pred.job.error = Some("Select a protein file.".to_owned());
                return;
            };
            state.ui.sequence_pred.job.start(context, move || {
                mpnn::design_file(&path, &request).map_err(|error| error.to_string())
            });
        }
    }
}

fn predict_opened_protein(
    protein: &MoleculePeptide,
    request: &DesignRequest,
) -> std::io::Result<DesignResult> {
    mpnn::design_mmcif(protein, request)
}

fn show_result(result: &DesignResult, ui: &mut Ui) {
    ui.separator();
    ui.label(RichText::new("Predicted amino-acid sequences").strong());
    if result.designs.is_empty() {
        ui.label(
            RichText::new("ProteinMPNN returned no designed sequences.").color(COLOR_INACTIVE),
        );
        return;
    }
    ui.label("Best first; lower ProteinMPNN score is better.");
    ScrollArea::vertical()
        .id_salt("sequence_prediction_results")
        .max_height(320.0)
        .show(ui, |ui| {
            for (index, design) in result.designs.iter().enumerate() {
                ui.horizontal(|ui| {
                    let score = design
                        .score
                        .map(|score| format!("{score:.4}"))
                        .unwrap_or_else(|| "—".to_owned());
                    ui.label(
                        RichText::new(format!("#{:<3} score {score}", index + 1))
                            .color(COLOR_HIGHLIGHT)
                            .monospace(),
                    );
                    if ui.small_button("Copy").clicked() {
                        ui.ctx().copy_text(design.sequence.clone());
                    }
                });
                ui.label(RichText::new(&design.sequence).monospace());
                ui.add_space(4.0);
            }
        });
}

fn split_words(value: &str) -> Vec<String> {
    value
        .split([',', ' '])
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_chain_lists() {
        assert_eq!(split_words("A, B C"), ["A", "B", "C"]);
    }

    #[test]
    fn popup_request_always_selects_original_protein_mpnn() {
        let request = SequencePredUi::default().request();
        assert_eq!(request.model, MpnnModel::ProteinMpnn);
        assert_eq!(request.num_sequences, 4);
        assert_eq!(request.omit_amino_acids, "X");
    }
}
