//! UI for predicting a peptide or nucleotide structure from its sequence.

use std::str::FromStr;

use egui::{ComboBox, RichText, TextEdit, Ui};
use na_seq::{AaIdent, AminoAcid, Nucleotide};

use crate::{
    state::State,
    structure_prediction::{
        StructurePredictionModel, predict_structure_from_aas, predict_structure_from_nts,
    },
    ui::{COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, ROW_SPACING, popup::close_btn},
    util::{handle_err, handle_success},
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
}

impl Default for StructurePredUi {
    fn default() -> Self {
        Self {
            sequence: String::new(),
            sequence_type: SequenceType::AminoAcid,
            // OpenDDE is the currently supported backend for this window.
            model: StructurePredictionModel::OpenDDE,
        }
    }
}

pub(in crate::ui) fn structure_prediction_window(
    state: &mut State,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
) {
    ui.heading("Structure prediction");
    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        let amino_acid_selected = state.ui.structure_pred.sequence_type == SequenceType::AminoAcid;
        let nucleotide_selected = state.ui.structure_pred.sequence_type == SequenceType::Nucleotide;
        ui.label("Sequence type:");
        ui.selectable_value(
            &mut state.ui.structure_pred.sequence_type,
            SequenceType::AminoAcid,
            RichText::new("AA").color(sequence_type_color(amino_acid_selected)),
        )
        .on_hover_text("Amino-acid sequence using single-letter identifiers");
        ui.selectable_value(
            &mut state.ui.structure_pred.sequence_type,
            SequenceType::Nucleotide,
            RichText::new("NT").color(sequence_type_color(nucleotide_selected)),
        )
        .on_hover_text("DNA nucleotide sequence using A, T, G, and C");
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
        if ui
            .button(RichText::new("Predict structure").color(COLOR_ACTION))
            .clicked()
        {
            predict(state, redraw_peptide);
        }

        close_btn(ui, &mut state.ui.popup.structure_pred);
    });
}

fn predict(state: &mut State, redraw_peptide: &mut bool) {
    let prediction = match state.ui.structure_pred.sequence_type {
        SequenceType::AminoAcid => {
            parse_amino_acids(&state.ui.structure_pred.sequence).map(|aas| {
                let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map else {
                    return Err("No peptide force-field parameter map is loaded".to_owned());
                };
                predict_structure_from_aas(state.ui.structure_pred.model, &aas, ff_map)
                    .map_err(|error| error.to_string())
            })
        }
        SequenceType::Nucleotide => {
            parse_nucleotides(&state.ui.structure_pred.sequence).map(|nts| {
                let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map else {
                    return Err("No peptide force-field parameter map is loaded".to_owned());
                };
                predict_structure_from_nts(state.ui.structure_pred.model, &nts, ff_map)
                    .map_err(|error| error.to_string())
            })
        }
    }
    .and_then(|result| result);

    match prediction {
        Ok(molecule) => {
            state.volatile.aa_seq_text = molecule
                .aa_seq
                .iter()
                .map(|aa| aa.to_str(AaIdent::OneLetter))
                .collect();
            state.peptide = Some(molecule);
            state.reset_selections();
            state.volatile.flags.ss_mesh_created = false;
            state.volatile.flags.sas_mesh_created = false;
            state.volatile.flags.clear_density_drawing = true;
            state.volatile.flags.new_mol_loaded = true;
            *redraw_peptide = true;
            handle_success(
                &mut state.ui,
                "Structure prediction complete; loaded predicted molecule".to_owned(),
            );
        }
        Err(error) => handle_err(
            &mut state.ui,
            format!("Structure prediction failed: {error}"),
        ),
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

fn sequence_type_color(selected: bool) -> egui::Color32 {
    if selected {
        COLOR_ACTIVE
    } else {
        COLOR_HIGHLIGHT
    }
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
}
