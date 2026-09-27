//! Amino-acid composition and residue highlighting across protein representations.

use bio_files::ResidueType;
use egui::{Button, RichText, Ui};
use na_seq::{AaIdent, AminoAcid, amino_acids::AaCategory};

use crate::{
    drawing::{
        MoleculeView,
        aa_highlight::{AMINO_ACIDS, secondary_classes},
    },
    state::{AaSelection, State},
    ui::misc::section_box,
    util::make_egui_color,
};

#[derive(Clone, Copy)]
enum Group {
    Hydrophobic,
    Hydrophilic,
    PolarUncharged,
    Basic,
    Acidic,
    Charged,
    Uncharged,
    Aromatic,
}

impl Group {
    fn contains(self, aa: AminoAcid) -> bool {
        use AaCategory::*;
        match self {
            Self::Hydrophobic => aa.category() == Hydrophobic,
            Self::Hydrophilic => aa.category() != Hydrophobic,
            Self::PolarUncharged => aa.category() == Polar,
            Self::Basic => aa.category() == Basic,
            Self::Acidic => aa.category() == Acidic,
            Self::Charged => matches!(aa.category(), Basic | Acidic),
            Self::Uncharged => matches!(aa.category(), Hydrophobic | Polar),
            Self::Aromatic => matches!(aa, AminoAcid::Phe | AminoAcid::Tyr | AminoAcid::Trp),
        }
    }
}

pub(in crate::ui) fn aa_selection(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    let Some(mol) = state
        .peptide_for_tools_i()
        .and_then(|i| state.peptides.get(i))
    else {
        ui.label("Open a protein to inspect its amino-acid composition.");
        return;
    };

    let name = mol.common.name(None);
    let secondary = secondary_classes(mol);
    let residues: Vec<_> = mol
        .residues
        .iter()
        .enumerate()
        .filter_map(|(i, res)| match res.res_type {
            ResidueType::AminoAcid(aa) => Some((aa, secondary[i])),
            _ => None,
        })
        .collect();
    let no_structure = mol.secondary_structure.is_empty();
    let settings = &mut state.ui.aa_selection;
    let previous = settings.clone();

    section_box().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.strong("Amino-acid highlights");
            ui.checkbox(&mut settings.enabled, "Enable").on_hover_text(
                "Apply to all open proteins in every view. \
                    Hiding this panel keeps highlighting enabled.",
            );
            ui.checkbox(&mut settings.color_by_aa, "AA colors");
            if !settings.color_by_aa {
                ui.color_edit_button_rgb(&mut settings.highlight_color);
            }
            ui.label("Neutral:");
            ui.color_edit_button_rgb(&mut settings.neutral_color);
            if ui.button("Reset").clicked() {
                *settings = AaSelection::default();
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Types:");
            for (label, group) in [
                ("Hydrophobic / nonpolar", Group::Hydrophobic),
                ("Hydrophilic / polar", Group::Hydrophilic),
                ("Polar uncharged", Group::PolarUncharged),
                ("Basic (+)", Group::Basic),
                ("Acidic (−)", Group::Acidic),
                ("Charged", Group::Charged),
                ("Uncharged", Group::Uncharged),
                ("Aromatic", Group::Aromatic),
            ] {
                let members: Vec<_> = AMINO_ACIDS
                    .into_iter()
                    .filter(|aa| group.contains(*aa))
                    .collect();
                let labels = members
                    .iter()
                    .map(|aa| aa.to_str(AaIdent::ThreeLetters))
                    .collect::<Vec<_>>()
                    .join(", ");
                if ui
                    .button(label)
                    .on_hover_text(format!(
                        "Replace AA choices with: {labels}. Structure filters still apply."
                    ))
                    .clicked()
                {
                    settings.amino_acids = members.into_iter().collect();
                    settings.enabled = true;
                }
            }
        });

        ui.horizontal_wrapped(|ui| {
            for label in ["All", "None", "Invert"] {
                if ui.button(label).clicked() {
                    settings.amino_acids = AMINO_ACIDS
                        .into_iter()
                        .filter(|aa| {
                            label == "All"
                                || (label == "Invert" && !settings.amino_acids.contains(aa))
                        })
                        .collect();
                    settings.enabled = true;
                }
            }
            ui.label("Colors apply to all proteins; counts show the active protein (all chains).");
        });

        ui.horizontal_wrapped(|ui| {
            for aa in AMINO_ACIDS {
                let count = residues.iter().filter(|(kind, _)| *kind == aa).count();
                // Selenocysteine is supported by the molecule model; show it only when present.
                if aa == AminoAcid::Sec && count == 0 {
                    continue;
                }
                let selected = settings.amino_acids.contains(&aa);
                let color = make_egui_color(settings.matched_color(aa));
                let label = format!(
                    "{} ({}) · {count}",
                    aa.to_str(AaIdent::ThreeLetters),
                    aa.to_str(AaIdent::OneLetter),
                );
                let percent = 100. * count as f32 / residues.len().max(1) as f32;
                if ui
                    .add(Button::new(RichText::new(label).color(color)).selected(selected))
                    .on_hover_text(format!(
                        "{count} residues ({percent:.1}%). Click to toggle this type."
                    ))
                    .clicked()
                {
                    if selected {
                        settings.amino_acids.remove(&aa);
                    } else {
                        settings.amino_acids.insert(aa);
                    }
                    settings.enabled = true;
                }
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("AND structure:");
            for (i, label) in ["Helix", "Sheet", "Coil / loop / unassigned"]
                .into_iter()
                .enumerate()
            {
                let count = residues.iter().filter(|(_, class)| *class == i).count();
                if ui
                    .toggle_value(&mut settings.secondary[i], format!("{label} · {count}"))
                    .changed()
                {
                    settings.enabled = true;
                }
            }
            if ui.button("All structures").clicked() {
                settings.secondary = [true; 3];
                settings.enabled = true;
            }
        });

        let matched = residues
            .iter()
            .filter(|(aa, class)| settings.matches(*aa, *class))
            .count();
        let percent = 100. * matched as f32 / residues.len().max(1) as f32;
        ui.label(format!(
            "{name}: {matched} / {} residues match ({percent:.1}%).",
            residues.len(),
        ));
        if no_structure {
            ui.small("No secondary-structure annotations: all residues are treated as unassigned.");
        }
        ui.small(
            "Groups describe side-chain classes, not calculated charge at the current pH. \
            Basic includes His; aromatic includes Phe, Tyr, Trp.",
        );
    });

    if *settings != previous {
        *redraw = true;
        if state.ui.mol_view_peptide == MoleculeView::Ribbon {
            state.volatile.flags.update_ss_mesh = true;
        } else {
            state.volatile.flags.ss_mesh_dirty = true;
        }
        state.volatile.flags.update_sas_coloring = state.volatile.flags.sas_mesh_created;
        // Discard a pending surface result computed with the previous settings.
        state.volatile.thread_receivers.peptide_mesh_coloring = None;
    }
}
