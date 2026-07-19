//! UI for predicting a peptide or nucleotide structure from its sequence.

use std::{
    str::FromStr,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use egui::{Button, ComboBox, Context, RichText, TextEdit, Ui};
use na_seq::{AminoAcid, Nucleotide};

use crate::{
    button,
    state::State,
    structure_prediction::{
        PredictionControl, StructurePredictionModel, StructurePredictionOutcome,
        predict_structure_from_aas_with_control, predict_structure_from_nts_with_control,
    },
    ui::{COLOR_ACTION, COLOR_ACTIVE, COLOR_INACTIVE, ROW_SPACING, popup::close_btn},
    util::handle_err,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SequenceType {
    #[default]
    AminoAcid,
    Nucleotide,
}

#[derive(Debug)]
pub(crate) struct StructurePredUi {
    pub sequence: String,
    pub sequence_type: SequenceType,
    pub model: StructurePredictionModel,
    started_at: Option<Instant>,
    control: Option<PredictionControl>,
    cancel_requested: bool,
}

impl Default for StructurePredUi {
    fn default() -> Self {
        Self {
            sequence: String::new(),
            sequence_type: SequenceType::AminoAcid,
            // OpenDDE is the currently supported backend for this window.
            model: StructurePredictionModel::OpenDDE,
            started_at: None,
            control: None,
            cancel_requested: false,
        }
    }
}

impl StructurePredUi {
    pub(crate) fn is_running(&self) -> bool {
        self.started_at.is_some()
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
    }

    fn cancel_prediction(&mut self) {
        if let Some(control) = &self.control {
            control.cancel();
            self.cancel_requested = true;
        }
    }
}

pub(in crate::ui) fn structure_prediction_window(state: &mut State, ui: &mut Ui) {
    ui.heading("Structure prediction");
    if let Some(started_at) = state.ui.structure_pred.started_at {
        let elapsed = started_at.elapsed();
        ui.label(format!("Time running: {}", format_elapsed(elapsed)));
        if state.ui.structure_pred.cancel_requested {
            ui.label("Cancellation requested...");
        }

        // The worker requests an immediate repaint when it sends its result. This timer is only
        // responsible for advancing the elapsed-time display while the process is otherwise quiet.
        let seconds_to_next_update = 30 - elapsed.as_secs() % 30;
        ui.ctx()
            .request_repaint_after(Duration::from_secs(seconds_to_next_update));
    }
    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.label("Sequence type:");

        let color = if state.ui.structure_pred.sequence_type == SequenceType::AminoAcid {
            COLOR_ACTIVE
        } else {
            COLOR_INACTIVE
        };
        if button!(
            ui,
            "Amino acids (1 letter)",
            color,
            "Amino-acid sequence using single-letter identifiers"
        )
        .clicked()
        {
            state.ui.structure_pred.sequence_type = SequenceType::AminoAcid;
        }

        let color = if state.ui.structure_pred.sequence_type == SequenceType::Nucleotide {
            COLOR_ACTIVE
        } else {
            COLOR_INACTIVE
        };
        if button!(
            ui,
            "Nucleotides",
            color,
            "DNA nucleotide sequence using A, T, G, and C"
        )
        .clicked()
        {
            state.ui.structure_pred.sequence_type = SequenceType::Nucleotide;
        }
    });

    ui.add(
        TextEdit::multiline(&mut state.ui.structure_pred.sequence)
            .desired_rows(8)
            .desired_width(420.)
            .hint_text(match state.ui.structure_pred.sequence_type {
                SequenceType::AminoAcid => "Enter a single-letter amino-acid sequence",
                SequenceType::Nucleotide => "Enter a DNA sequence (A, T, G, C)",
            }),
    );

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.label("Model:");
        ComboBox::from_id_salt("structure_prediction_model")
            .selected_text(model_name(state.ui.structure_pred.model))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut state.ui.structure_pred.model,
                    StructurePredictionModel::OpenDDE,
                    model_name(StructurePredictionModel::OpenDDE),
                );
                ui.selectable_value(
                    &mut state.ui.structure_pred.model,
                    StructurePredictionModel::Boltz2,
                    model_name(StructurePredictionModel::Boltz2),
                );
            });
    });

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        if state.ui.structure_pred.is_running() {
            if ui
                .add_enabled(
                    !state.ui.structure_pred.cancel_requested,
                    Button::new(RichText::new("Cancel").color(COLOR_ACTION)),
                )
                .clicked()
            {
                state.ui.structure_pred.cancel_prediction();
            }
        } else if ui
            .button(RichText::new("Predict structure").color(COLOR_ACTION))
            .clicked()
        {
            predict(state, ui.ctx());
        }

        close_btn(ui, &mut state.ui.popup.structure_pred);
    });
}

enum PredictionSequence {
    AminoAcids(Vec<AminoAcid>),
    Nucleotides(Vec<Nucleotide>),
}

fn predict(state: &mut State, context: &Context) {
    let sequence = match state.ui.structure_pred.sequence_type {
        SequenceType::AminoAcid => {
            parse_amino_acids(&state.ui.structure_pred.sequence).map(PredictionSequence::AminoAcids)
        }
        SequenceType::Nucleotide => parse_nucleotides(&state.ui.structure_pred.sequence)
            .map(PredictionSequence::Nucleotides),
    };
    let sequence = match sequence {
        Ok(sequence) => sequence,
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
        let prediction = match sequence {
            PredictionSequence::AminoAcids(aas) => {
                predict_structure_from_aas_with_control(model, &aas, &ff_map, &worker_control)
            }
            PredictionSequence::Nucleotides(nts) => {
                predict_structure_from_nts_with_control(model, &nts, &ff_map, &worker_control)
            }
        };

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

fn parse_amino_acids(sequence: &str) -> Result<Vec<AminoAcid>, String> {
    parse_sequence(sequence, "amino-acid", |letter| {
        AminoAcid::from_str(&letter.to_string())
    })
}

fn parse_nucleotides(sequence: &str) -> Result<Vec<Nucleotide>, String> {
    parse_sequence(sequence, "nucleotide", |letter| {
        Nucleotide::from_str(&letter.to_string())
    })
}

fn parse_sequence<T, E>(
    sequence: &str,
    residue_name: &str,
    mut parse: impl FnMut(char) -> Result<T, E>,
) -> Result<Vec<T>, String> {
    let mut result = Vec::new();
    for letter in sequence.chars().filter(|letter| !letter.is_whitespace()) {
        let position = result.len() + 1;
        result.push(parse(letter).map_err(|_| {
            format!("Invalid {residue_name} identifier '{letter}' at sequence position {position}")
        })?);
    }

    if result.is_empty() {
        return Err(format!("The {residue_name} sequence is empty"));
    }

    Ok(result)
}

fn model_name(model: StructurePredictionModel) -> &'static str {
    match model {
        StructurePredictionModel::Boltz2 => "Boltz-2",
        StructurePredictionModel::OpenDDE => "OpenDDE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiline_sequences_and_ignores_whitespace() {
        assert_eq!(
            parse_amino_acids("ACD\n EF").unwrap(),
            vec![
                AminoAcid::Ala,
                AminoAcid::Cys,
                AminoAcid::Asp,
                AminoAcid::Glu,
                AminoAcid::Phe,
            ]
        );
        assert_eq!(
            parse_nucleotides("AT\n GC").unwrap(),
            vec![Nucleotide::A, Nucleotide::T, Nucleotide::G, Nucleotide::C]
        );
    }

    #[test]
    fn rejects_empty_and_invalid_sequences() {
        assert!(parse_amino_acids(" \n ").is_err());
        assert_eq!(
            parse_nucleotides("ATX").unwrap_err(),
            "Invalid nucleotide identifier 'X' at sequence position 3"
        );
    }

    #[test]
    fn formats_elapsed_prediction_time() {
        assert_eq!(format_elapsed(Duration::ZERO), "00:00");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "00:59");
        assert_eq!(format_elapsed(Duration::from_secs(3_661)), "01:01:01");
    }
}
