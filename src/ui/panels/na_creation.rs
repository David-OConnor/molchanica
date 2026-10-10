use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType, Strands};
use na_seq::{
    Nucleotide::{self, *},
    seq_from_str,
};

use crate::{
    drawing::wrappers::draw_all_nucleic_acids,
    state::State,
    ui::{COLOR_ACTION, misc::section_box, panels::aa_creation::SEQ_INPUT_WIDTH},
    util::handle_err,
};

/// Sanger chromatogram colors, lightened for a dark background. These match PlasCAD's.
fn nt_color(nt: Nucleotide) -> Color32 {
    match nt {
        A => Color32::from_rgb(20, 220, 20),
        C => Color32::from_rgb(130, 130, 255),
        T => Color32::from_rgb(255, 100, 100),
        G => Color32::from_rgb(200, 200, 200),
    }
}

fn nt_name(nt: Nucleotide) -> &'static str {
    match nt {
        A => "adenine",
        C => "cytosine",
        T => "thymine",
        G => "guanine",
    }
}

/// Add and manage nucleic acids
pub(in crate::ui) fn na_section(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                let help_text =
                    "Enter the nucleotide sequence of the DNA or RNA molecule to create";

                ui.label("Seq").on_hover_text(help_text);

                ui.add(
                    TextEdit::multiline(&mut state.to_save.nucleic_acid.seq_to_create)
                        .desired_width(SEQ_INPUT_WIDTH)
                        .desired_rows(2),
                )
                .on_hover_text(help_text);

                for nt in [A, C, G, T] {
                    let letter = nt.to_str_upper();

                    if ui
                        .button(RichText::new(&letter).color(nt_color(nt)))
                        .on_hover_text(format!(
                            "Add {letter} ({}) to the end of the sequence",
                            nt_name(nt)
                        ))
                        .clicked()
                    {
                        state.to_save.nucleic_acid.seq_to_create.push_str(&letter);
                    }
                }

                ComboBox::from_id_salt(12443)
                    .width(80.)
                    .selected_text(state.to_save.nucleic_acid.na_type.to_string())
                    .show_ui(ui, |ui| {
                        for v in &[NucleicAcidType::Dna, NucleicAcidType::Rna] {
                            ui.selectable_value(
                                &mut state.to_save.nucleic_acid.na_type,
                                *v,
                                v.to_string(),
                            );
                        }
                    });

                ComboBox::from_id_salt(12444)
                    .width(80.)
                    .selected_text(state.to_save.nucleic_acid.strands.to_string())
                    .show_ui(ui, |ui| {
                        // todo: Temp SS only until we sort out strand alignment
                        // for v in &[Strands::Single] {
                        for v in &[Strands::Single, Strands::Double] {
                            ui.selectable_value(
                                &mut state.to_save.nucleic_acid.strands,
                                *v,
                                v.to_string(),
                            );
                        }
                    });

                if ui
                    .button(RichText::new("Create").color(COLOR_ACTION))
                    .clicked()
                {
                    // todo: Handle RNA U.
                    let seq = seq_from_str(&state.to_save.nucleic_acid.seq_to_create);

                    let mol = match MoleculeNucleicAcid::from_seq(
                        &seq,
                        state.to_save.nucleic_acid.na_type,
                        state.to_save.nucleic_acid.strands,
                        &state.templates.dna,
                        &state.templates.rna,
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            handle_err(
                                &mut state.ui,
                                format!("Problem making a Nucleic acid: {e:?}"),
                            );
                            return;
                        }
                    };

                    state.nucleic_acids.push(mol);

                    draw_all_nucleic_acids(state, scene, updates);
                }
            });
        });
    });
}
