use egui::{ComboBox, Resize, RichText, ScrollArea, TextBuffer, TextEdit, TextStyle, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::{
    MolGenericRef, MolType, MoleculeGeneric,
    nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType, Strands},
};
use na_seq::Nucleotide::{self, *};

use super::sequence_input::{
    INVALID_LETTER_COLOR, create_button, layout, letter_counts, status_label,
};
use crate::{
    cam::move_cam_to_mol_front,
    file_io::managed_mols,
    nucleic_acid_creation::{ResidueGrouping, base_color, base_letter, group_residues, nucleotide},
    state::State,
    ui::{
        COLOR_ACTION, ROW_SPACING,
        misc::{section_box, selector},
        util::report_cache_result,
    },
    util::handle_err,
};

fn nt_name(nt: Nucleotide, na_type: NucleicAcidType) -> &'static str {
    match nt {
        A => "adenine",
        C => "cytosine",
        T if na_type == NucleicAcidType::Rna => "uracil",
        T => "thymine",
        G => "guanine",
    }
}

fn sequence_input(ui: &mut Ui, text: &mut String, na_type: NucleicAcidType, max_width: f32) {
    let help = "Enter a nucleotide sequence. Drag the lower-right corner to resize.";

    let mut layouter = |ui: &Ui, text: &dyn TextBuffer, wrap_width: f32| {
        let font_id = TextStyle::Monospace.resolve(ui.style());

        layout(ui, text.as_str(), wrap_width, font_id, |ch| {
            nucleotide(ch, na_type).map_or(INVALID_LETTER_COLOR, base_color)
        })
    };

    ui.label("Seq").on_hover_text(help);
    ui.vertical(|ui| {
        Resize::default()
            .id_salt("na_sequence_size")
            .default_size([600., 52.])
            .min_size([120., 38.])
            .max_width(max_width)
            .max_height((ui.ctx().content_rect().height() * 0.5).max(52.))
            .with_stroke(false)
            .show(ui, |ui| {
                // Keep the corner handle below the editor.
                let height = (ui.available_height() - ui.visuals().resize_corner_size).max(1.);
                ScrollArea::vertical()
                    .max_height(height)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add(
                            TextEdit::multiline(text)
                                .font(TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(1)
                                .min_size([0., height].into())
                                .layouter(&mut layouter),
                        )
                        .on_hover_text(help);
                    });
            });
    });
}

fn grouping_options(ui: &mut Ui, grouping: &mut ResidueGrouping) {
    use ResidueGrouping::*;

    ui.horizontal_wrapped(|ui| {
        ui.label("Residues");
        let options = [
            (
                Nucleotide,
                "Per nt",
                "Each base and its backbone form a residue.",
            ),
            (
                BasePair,
                "Per bp",
                "Both nucleotides of a pair form one residue.",
            ),
            (
                BasesAndBackbones,
                "Paired bases / bbs",
                "The paired bases form one residue and their backbones form another.",
            ),
            (
                BaseAndBackbone,
                "Separate bases / bbs",
                "Each base and each backbone form separate residues.",
            ),
        ];

        if let Some(selected) = selector(ui, *grouping, &options) {
            *grouping = selected;
        }
    });
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
            let na_type = state.to_save.nucleic_acid.na_type;
            let max_width = (ui.ctx().content_rect().width() - 210.).max(120.);
            ui.horizontal(|ui| {
                sequence_input(
                    ui,
                    &mut state.to_save.nucleic_acid.seq_to_create,
                    na_type,
                    max_width,
                );

                for nt in [A, C, G, T] {
                    let letter = base_letter(nt, na_type);

                    if ui
                        .button(RichText::new(letter.to_string()).color(base_color(nt)))
                        .on_hover_text(format!(
                            "Add {letter} ({}) to the end of the sequence",
                            nt_name(nt, na_type)
                        ))
                        .clicked()
                    {
                        state.to_save.nucleic_acid.seq_to_create.push(letter);
                    }
                }
            });

            grouping_options(ui, &mut state.ui.na_residue_grouping);
            ui.add_space(ROW_SPACING);

            ui.horizontal(|ui| {
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
                        for v in &[Strands::Single, Strands::Double] {
                            ui.selectable_value(
                                &mut state.to_save.nucleic_acid.strands,
                                *v,
                                v.to_string(),
                            );
                        }
                    });

                let na_type = state.to_save.nucleic_acid.na_type;
                let text = &state.to_save.nucleic_acid.seq_to_create;
                let (valid, invalid) = letter_counts(text, |ch| nucleotide(ch, na_type).is_some());
                let can_create = valid > 0 && invalid == 0;

                let create = ui.add_enabled(
                    can_create,
                    create_button("Create", COLOR_ACTION, can_create),
                );
                status_label(ui, valid, invalid, "nucleotide");

                if create.clicked() {
                    let seq: Vec<_> = text
                        .chars()
                        .filter_map(|ch| nucleotide(ch, na_type))
                        .collect();

                    let mut mol = match MoleculeNucleicAcid::from_seq(
                        &seq,
                        na_type,
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
                    group_residues(&mut mol, state.ui.na_residue_grouping);
                    let query: String = seq.iter().map(|&nt| base_letter(nt, na_type)).collect();

                    // Kept in a managed file, so it reopens next session; as for a download.
                    let result = managed_mols::store_created(
                        &state.volatile.prefs_dir,
                        MolGenericRef::NucleicAcid(&mol),
                        &query,
                    );
                    let path = report_cache_result(state, result);

                    // Makes it the active molecule.
                    state.load_mol_to_state(
                        MoleculeGeneric::NucleicAcid(mol),
                        scene,
                        updates,
                        path.as_deref(),
                    );

                    let molecule = (MolType::NucleicAcid, state.nucleic_acids.len() - 1);
                    move_cam_to_mol_front(state, scene, molecule, updates);
                }
            });
        });
    });
}
