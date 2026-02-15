use std::{collections::HashMap, path::Path};

use egui::{Align, Color32, ComboBox, Layout, RichText, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    drawing,
    drawing::blend_color,
    label, mol_manip,
    mol_manip::ManipMode,
    molecules::MolType,
    selection::Selection,
    state::{PopupState, State},
    therapeutic::pharmacophore::{
        FeatureRelation, Pharmacophore, PharmacophoreFeatType, PharmacophoreFeature,
        add_pharmacophore,
    },
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_INACTIVE, ROW_SPACING,
        util::color_egui_from_f32,
    },
    util::{RedrawFlags, handle_err, make_egui_color},
};

/// Assumes run from the editor for now.
pub(in crate::ui) fn pharmacophore_boolean_window(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        label!(ui, "Relate pharmacophore features", Color32::WHITE);

        ui.with_layout(Layout::top_down(Align::RIGHT), |ui| {
            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                state.ui.popup.pharmacophore_boolean = false;
            }
        });
    });

    ui.label(RichText::new("Select 2 features to mark with *Or* logic.").color(Color32::GRAY));

    ui.separator();
    ui.add_space(ROW_SPACING);

    for (i, feat) in state
        .mol_editor
        .mol
        .pharmacophore
        .features
        .iter_mut()
        .enumerate()
    {
        ui.horizontal(|ui| {
            label!(
                ui,
                format!("{}: {feat}", i + 1),
                make_egui_color(feat.feature_type.color())
            );

            ui.add_space(COL_SPACING);

            let color = if feat.ui_selected {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if ui.button(RichText::new("Select").color(color)).clicked() {
                feat.ui_selected = !feat.ui_selected
            }
        });
    }

    let num_selected = state
        .mol_editor
        .mol
        .pharmacophore
        .features
        .iter()
        .filter(|f| f.ui_selected)
        .count();

    if num_selected == 2 {
        ui.add_space(ROW_SPACING);

        if ui
            .button(RichText::new("Create OR logic").color(COLOR_ACTION))
            .on_hover_text(
                "Allow either of these features to match for a given ligand; both aren't required.",
            )
            .clicked()
        {
            let sel: Vec<_> = state
                .mol_editor
                .mol
                .pharmacophore
                .features
                .iter()
                .enumerate()
                .filter(|(_, f)| f.ui_selected)
                .map(|(i, _)| i)
                .collect();

            state
                .mol_editor
                .mol
                .pharmacophore
                .feature_relations
                .push(FeatureRelation::Or((sel[0], sel[1])));

            for v in sel {
                state.mol_editor.mol.pharmacophore.features[v].ui_selected = false;
            }
        }
    }

    boolean_list(
        &state.mol_editor.mol.pharmacophore.features,
        &mut state.mol_editor.mol.pharmacophore.feature_relations,
        ui,
    );
}

/// Show all boolean relations (*Or* logic etc)
fn boolean_list(feats: &[PharmacophoreFeature], relations: &mut Vec<FeatureRelation>, ui: &mut Ui) {
    if relations.is_empty() {
        return;
    }

    ui.separator();
    ui.add_space(ROW_SPACING);
    ui.label(RichText::new("Relations:").color(Color32::GRAY));

    let mut rel_removed = None;
    for (i_rel, rel) in relations.iter().enumerate() {
        ui.horizontal(|ui| {
            match rel {
                FeatureRelation::Or(pair) => {
                    // Blend to white so it's more visible against the dark UI.
                    label!(ui, "Or", Color32::WHITE);
                    for i in [pair.0, pair.1] {
                        if i >= feats.len() {
                            eprintln!("Invalid feature index in boolean list.");
                            return;
                        }

                        let color = blend_color(feats[i].feature_type.color(), (1., 1., 1.), 0.4);
                        let color = color_egui_from_f32(color);

                        label!(ui, &format!("{}: {}", i + 1, feats[i].feature_type), color);
                    }

                    if ui
                        .button(RichText::new("❌").color(Color32::LIGHT_RED))
                        .clicked()
                    {
                        println!("{rel:?}");
                        rel_removed = Some(i_rel);
                    }
                }
                _ => {}
            }
        });
    }

    if let Some(i) = rel_removed {
        relations.remove(i);
    }
}

/// Allows viewing and selecting each pharmacophore feature from a tabular etc display. I.e.,
/// to supplement positional renderings.
pub(in crate::ui) fn pharmacophore_list(
    pharmacophore: &mut Pharmacophore,
    popup: &mut PopupState,
    vis_flag: &mut bool,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        label!(ui, "Pharmacophores", Color32::GRAY);
        ui.add_space(COL_SPACING / 2.);

        if ui
            .button(RichText::new("And / Or logic").color(Color32::GRAY))
            .on_hover_text(
                "Add or change boolean logic between pharmacophore features. E.g. marking
            a pair of features as requiring a match of one or the other, but not both.",
            )
            .clicked()
        {
            popup.pharmacophore_boolean = !popup.pharmacophore_boolean;
        }

        ui.add_space(COL_SPACING / 2.);

        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            *vis_flag = false;
        }
    });
    ui.separator();

    let mut remove = None;
    for (i, feat) in pharmacophore.features.iter().enumerate() {
        let descrip = format!("{feat}");

        ui.horizontal(|ui| {
            label!(ui, descrip, Color32::WHITE);

            ui.add_space(COL_SPACING);
            if ui
                .button(RichText::new("❌").color(Color32::LIGHT_RED))
                .clicked()
            {
                remove = Some(i);
            }
        });
    }

    if let Some(i) = remove {
        pharmacophore.features.remove(i);
    }

    boolean_list(
        &pharmacophore.features,
        &mut pharmacophore.feature_relations,
        ui,
    );

    ui.add_space(ROW_SPACING);
}

pub(in crate::ui) fn pharmacophore_edit_tools(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
) {
    ui.horizontal(|ui| {
        label!(ui, "Pharmacophore: ", Color32::WHITE);

        let prev = state.ui.pharmacaphore_type;
        ComboBox::from_id_salt(11123)
            .width(120.)
            .selected_text(state.ui.pharmacaphore_type.to_string())
            .show_ui(ui, |ui| {
                for v in PharmacophoreFeatType::all() {
                    ui.selectable_value(&mut state.ui.pharmacaphore_type, v, v.to_string());
                }
            });

        if state.ui.pharmacaphore_type != prev {
            let mol = &state.mol_editor.mol;

            let Some(char) = &mol.characterization else {
                eprintln!("Missing char on mol editor when changing pharmacophore type.");
                return;
            };

            let hint_sites = state
                .ui
                .pharmacaphore_type
                .hint_sites(char, &mol.common.atom_posits);

            drawing::draw_pharmacophore_hint_sites(&mut scene.entities, &hint_sites, updates);
        }

        if let Selection::AtomLig((_mol_i, atom_i)) = &state.ui.selection
            && ui
                .button(RichText::new("Add pharmacophore").color(COLOR_ACTION))
                .clicked()
        {
            if add_pharmacophore(
                &mut state.mol_editor.mol,
                state.ui.pharmacaphore_type,
                *atom_i,
            )
            .is_err()
            {
                eprintln!("Error adding pharmacophore feature.");
            };

            *redraw = true;
        }

        if !state.mol_editor.mol.pharmacophore.features.is_empty()
            && ui
                .button(RichText::new("Save"))
                .on_hover_text("Save the pharmacophore to a file.")
                .clicked()
        {
            let name = &state.mol_editor.mol.common.ident;

            if state
                .mol_editor
                .mol
                .pharmacophore
                .save(&mut state.volatile.dialogs.save, name)
                .is_err()
            {
                handle_err(&mut state.ui, "Problem saving the pharmacophore".to_owned());
            }
        }

        // Pocket-related functionality.
        if state.mol_editor.mol.pharmacophore.pocket.is_some() {
            // todo: Match on selection, or active mol here?
            // let pocket_sel_prev = matches!(state.ui.selection, Selection::AtomPocket(_));
            let pocket_sel_prev = matches!(state.volatile.active_mol, Some((MolType::Pocket, _)));

            // Allow selecting the pocket, e.g. as opposed to a lig atom or bond.
            {
                let color = if pocket_sel_prev {
                    COLOR_ACTIVE
                } else {
                    COLOR_INACTIVE
                };

                if ui
                    .button(RichText::new("Select pocket").color(color))
                    .on_hover_text(
                        "Select the pocket, e.g. to manipulate it relative to the ligand.",
                    )
                    .clicked()
                {
                    state.ui.selection = if pocket_sel_prev {
                        Selection::None
                    } else {
                        Selection::AtomPocket((0, 0))
                    };

                    state.volatile.active_mol = if pocket_sel_prev {
                        None
                    } else {
                        Some((MolType::Pocket, 0))
                    };
                }
            }

            // if matches!(state.ui.selection, Selection::AtomPocket(_)) {
            if pocket_sel_prev {
                let mut color_move = COLOR_INACTIVE;
                let mut color_rotate = COLOR_INACTIVE;

                match state.volatile.mol_manip.mode {
                    ManipMode::Move((mol_type, _mol_i)) => {
                        if mol_type == MolType::Pocket {
                            color_move = COLOR_ACTIVE;
                        }
                    }
                    ManipMode::Rotate((mol_type, _mol_i)) => {
                        if mol_type == MolType::Pocket {
                            color_rotate = COLOR_ACTIVE;
                        }
                    }
                    ManipMode::None => (),
                }

                // dummy API interaction.
                let mut redraw_flags = RedrawFlags::default();
                redraw_flags.pocket = *redraw;

                if ui
                    .button(RichText::new("↔ pocket").color(color_move))
                    .on_hover_text("Move the pocket relative to the molecule.")
                    .clicked()
                {
                    mol_manip::set_manip(
                        state,
                        scene,
                        &mut redraw_flags,
                        &mut false,
                        ManipMode::Move((MolType::Pocket, 0)),
                        updates,
                    );
                    *redraw = redraw_flags.pocket;
                }

                if ui
                    .button(RichText::new("⟳ pocket").color(color_rotate))
                    .on_hover_text("Rotate the pocket relative to the molecule.")
                    .clicked()
                {
                    mol_manip::set_manip(
                        state,
                        scene,
                        &mut redraw_flags,
                        &mut false,
                        ManipMode::Rotate((MolType::Pocket, 0)),
                        updates,
                    );
                    *redraw = redraw_flags.pocket;
                }
            }

            // ui.add_space(COL_SPACING);
            // if ui
            //     .button("Use this pocket for pharmacophore")
            //     .on_hover_text("Use the displayed pocket as part of the pharmacophore")
            //     .clicked()
            // {
            //     copy_pocket_to_pharmacophore = true;
            // }
        }
    });
}

/// Similar to the `summary` method, but splits up labels for color-coding.
pub(in crate::ui) fn pharmacophore_summary(ph: &Pharmacophore, ui: &mut Ui) {
    let mut feat_counts = HashMap::new();
    for feat in &ph.features {
        *feat_counts.entry(feat.feature_type).or_insert(0) += 1;
    }

    let mut items: Vec<_> = feat_counts.into_iter().collect();
    items.sort_by(|(a_ft, _), (b_ft, _)| a_ft.cmp(b_ft));

    ui.horizontal(|ui| {
        for (ft, count) in items {
            // Blend to white so it's more visible against the dark UI.
            let color = blend_color(ft.color(), (1., 1., 1.), 0.4);
            let color = color_egui_from_f32(color);

            label!(ui, &format!("{ft}: {count} "), color);
        }

        if ui.button(RichText::new("screen")).clicked() {
            // todo: Use a file picker
            let path = Path::new("C:/Users/the_a/Desktop/bio_misc/Binding/ZINC22/H04");
            let path = Path::new("C:/Users/the_a/Desktop/");
        }
    });
}
