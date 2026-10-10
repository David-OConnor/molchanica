//! Create peptides and amino acid sequences from single-letter amino acid codes.

use egui::{RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::{MoleculeGeneric, peptide::MoleculePeptide};
use na_seq::{AaIdent, AminoAcid, SeqType, Sequence, SequenceData};

use crate::{
    cam::move_mol_to_cam,
    drawing::aa_highlight::AMINO_ACIDS,
    state::State,
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, misc::section_box},
    util::{handle_err, handle_success},
};

/// The amino acid buttons wrap at this width.
const SECTION_WIDTH: f32 = 640.;

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

/// Add and manage amino acids
pub(in crate::ui) fn aa_section(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        ui.set_max_width(SECTION_WIDTH);

        ui.horizontal_wrapped(|ui| {
            for aa in AMINO_ACIDS {
                let letter = aa.to_str(AaIdent::OneLetter);

                if ui
                    .button(aa.to_string())
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
                    .desired_width(240.)
                    .desired_rows(2),
            )
            .on_hover_text(help_text);

            if ui
                .button(RichText::new("Create unfolded").color(COLOR_ACTION))
                .on_hover_text(
                    "Create a peptide from this sequence, as an extended chain (β-strand). \
                    Uses Amber residue templates.",
                )
                .clicked()
            {
                let mol = parse_seq(&state.ui.aa_seq_to_create).and_then(|seq| {
                    MoleculePeptide::from_seq(&seq, &state.templates.amino_acid)
                        .map_err(|e| format!("Problem making a peptide: {e}"))
                });

                match mol {
                    Ok(mut mol) => {
                        move_mol_to_cam(&mut mol.common, &scene.camera);
                        state.load_mol_to_state(
                            MoleculeGeneric::Peptide(mol),
                            scene,
                            updates,
                            None,
                        );
                    }
                    Err(e) => handle_err(&mut state.ui, e),
                }
            }

            if ui
                .button(RichText::new("Create folded").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Open the structure prediction window, with this sequence as its input.",
                )
                .clicked()
            {
                match parse_seq(&state.ui.aa_seq_to_create) {
                    Ok(seq) => {
                        state
                            .ui
                            .tool_windows
                            .load_structure_pred_seq(make_sequence(seq));
                        state.ui.popup.structure_pred = true;
                    }
                    Err(e) => handle_err(&mut state.ui, e),
                }
            }

            if ui
                .button("Add sequence")
                .on_hover_text("Add this as an amino acid sequence, e.g. to view, edit, or save.")
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
}
