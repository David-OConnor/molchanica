//! Create peptides and amino acid sequences from single-letter amino acid codes.

use chrono::Local;
use egui::{RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::{MoleculeGeneric, aa_color, peptide::MoleculePeptide};
use na_seq::{AaIdent, AminoAcid, SeqType, Sequence, SequenceData};

use crate::{
    cam::move_mol_to_cam,
    drawing::aa_highlight::AMINO_ACIDS,
    state::State,
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, misc::section_box},
    util::{handle_err, handle_success, make_egui_color},
};

/// The amino acid buttons wrap at this width.
const SECTION_WIDTH: f32 = 800.;
/// Shared with the nucleic acid panel.
pub(in crate::ui) const SEQ_INPUT_WIDTH: f32 = 400.;

/// Parse the single-letter amino acid codes entered. Whitespace, digits and gaps are ignored.
fn parse_seq(text: &str) -> Result<Vec<AminoAcid>, String> {
    let (data, skipped) = SequenceData::from_letters(text, SeqType::AminoAcid);

    if skipped > 0 {
        return Err(format!(
            "{skipped} letter(s) in the sequence aren't amino acid codes"
        ));
    }

    match data {
        SequenceData::AminoAcid(seq) if !seq.is_empty() => Ok(seq),
        _ => Err("Enter an amino acid sequence first".to_owned()),
    }
}

fn make_sequence(seq: Vec<AminoAcid>) -> Sequence {
    let name = format!("Peptide {}aa", seq.len());
    Sequence::new(SequenceData::AminoAcid(seq), name)
}

/// A descriptive structure prediction job name, e.g. "MKTAYIAK... 2026-10-10".
fn job_name(sequence: &Sequence) -> String {
    const PREFIX_LEN: usize = 8;

    let letters = sequence.data.to_letters();
    let mut prefix: String = letters.chars().take(PREFIX_LEN).collect();

    if letters.len() > PREFIX_LEN {
        prefix.push_str("...");
    }

    format!("{prefix} {}", Local::now().format("%Y-%m-%d"))
}

pub(in crate::ui) fn create_unfolded_button(ui: &mut Ui) -> bool {
    ui.button(RichText::new("Create unfolded").color(COLOR_ACTION))
        .on_hover_text(
            "Create a peptide from this sequence, as an extended chain (β-strand). \
            Uses Amber residue templates.",
        )
        .clicked()
}

pub(in crate::ui) fn create_folded_button(ui: &mut Ui) -> bool {
    ui.button(RichText::new("Create folded").color(COLOR_HIGHLIGHT))
        .on_hover_text("Open the structure prediction window, with this sequence as its input.")
        .clicked()
}

/// Build an unfolded peptide, and add it in front of the camera.
pub(in crate::ui) fn create_unfolded(
    seq: &[AminoAcid],
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    match MoleculePeptide::from_seq(seq, &state.templates.amino_acid) {
        Ok(mut mol) => {
            move_mol_to_cam(&mut mol.common, &scene.camera);
            state.load_mol_to_state(MoleculeGeneric::Peptide(mol), scene, updates, None);
        }
        Err(e) => handle_err(&mut state.ui, format!("Problem making a peptide: {e}")),
    }
}

/// Open the structure prediction window, with this sequence as its input.
pub(in crate::ui) fn create_folded(sequence: Sequence, state: &mut State) {
    let job_name = job_name(&sequence);

    state
        .ui
        .tool_windows
        .load_structure_pred_seq(sequence, job_name);
    state.ui.popup.structure_pred = true;
}

/// Add and manage amino acids
pub(in crate::ui) fn aa_section(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        // The toolbar this is in lays out horizontally.
        ui.vertical(|ui| {
            ui.set_max_width(SECTION_WIDTH);

            ui.horizontal_wrapped(|ui| {
                for aa in AMINO_ACIDS {
                    let letter = aa.to_str(AaIdent::OneLetter);
                    let color = make_egui_color(aa_color(aa));

                    if ui
                        .button(RichText::new(aa.to_string()).color(color))
                        .on_hover_text(format!("Add {letter} to the end of the sequence"))
                        .clicked()
                    {
                        state.ui.aa_seq_to_create.push_str(&letter);
                    }
                }
            });

            ui.horizontal(|ui| {
                let help_text = "Enter the amino acid sequence, as single-letter codes, \
                    from the N terminus to the C terminus";

                ui.label("Seq").on_hover_text(help_text);

                ui.add(
                    TextEdit::multiline(&mut state.ui.aa_seq_to_create)
                        .desired_width(SEQ_INPUT_WIDTH)
                        .desired_rows(2),
                )
                .on_hover_text(help_text);

                if create_unfolded_button(ui) {
                    match parse_seq(&state.ui.aa_seq_to_create) {
                        Ok(seq) => create_unfolded(&seq, state, scene, updates),
                        Err(e) => handle_err(&mut state.ui, e),
                    }
                }

                if create_folded_button(ui) {
                    match parse_seq(&state.ui.aa_seq_to_create) {
                        Ok(seq) => create_folded(make_sequence(seq), state),
                        Err(e) => handle_err(&mut state.ui, e),
                    }
                }

                if ui
                    .button(RichText::new("Create sequence").color(COLOR_ACTION))
                    .on_hover_text(
                        "Add this as an amino acid sequence, e.g. to view, edit, or save.",
                    )
                    .clicked()
                {
                    match parse_seq(&state.ui.aa_seq_to_create) {
                        Ok(seq) => {
                            let sequence = make_sequence(seq);
                            let msg = format!("Added sequence {}", sequence.display_name());

                            state.sequences.push(sequence);
                            state.select_sequence(Some(state.sequences.len() - 1));

                            handle_success(&mut state.ui, msg);
                        }
                        Err(e) => handle_err(&mut state.ui, e),
                    }
                }
            });
        });
    });
}
