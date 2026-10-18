//! Shared editor for persistent selection expressions and additive display layers.

use egui::{ComboBox, ScrollArea, Slider, TextEdit, Ui};

use crate::{
    drawing::{MoleculeView, representations},
    selection::{Selection, expression},
    state::State,
    util::RedrawFlags,
};

#[derive(Default)]
pub struct SelectionEditor {
    name: String,
    expression: String,
    view: MoleculeView,
    message: String,
    layer: Option<usize>,
    layer_expression: String,
}

fn view_picker(ui: &mut Ui, id: &str, view: &mut MoleculeView) {
    ComboBox::from_id_salt(id)
        .selected_text(view.to_string())
        .show_ui(ui, |ui| {
            for option in MoleculeView::PEPTIDE_OPTIONS {
                ui.selectable_value(view, option, option.to_string());
            }
        });
}

pub fn editor(state: &mut State, ui: &mut Ui, redraw: &mut RedrawFlags) {
    let mut edit = std::mem::take(&mut state.ui.selection_editor);
    ui.label("Selections are expressions evaluated against loaded objects. Names and layers are saved with preferences.");
    ui.collapsing("Selection syntax", |ui| {
        ui.label("and / or / not, parentheses, byres (...), ... within 4 of (...)");
        ui.label("model, chain, resi 10-25+40, resn ASP+GLU, elem C+N, name CA, id, index (1-based), alt");
        ui.label("all, none, protein, nucleic, ligand, lipid, water, hydrogen, backbone, sidechain, hetatm, sele");
        ui.label("charge < 0, occupancy >= 0.5; use %name to reference a saved selection.");
        ui.label("Text values accept * wildcards and + lists. Quote paths or values containing spaces. Distances are in Å.");
        ui.label("Example: byres (protein within 4 of ligand)");
    });
    ui.horizontal(|ui| {
        if ui.button("Use picked selection").clicked() {
            edit.expression = expression::current_expression(state);
        }
        ComboBox::from_id_salt("selection_object")
            .selected_text("Use object…")
            .show_ui(ui, |ui| {
                for (_, _, mol) in expression::objects(state) {
                    if ui.button(&*mol.name()).clicked() {
                        edit.expression = expression::object_expression(&mol);
                    }
                }
            });
    });
    ui.add(
        TextEdit::multiline(&mut edit.expression)
            .desired_rows(2)
            .desired_width(f32::INFINITY)
            .hint_text("Selection expression"),
    );
    ui.horizontal(|ui| {
        if ui.button("Select / preview").clicked() {
            match expression::evaluate(state, &edit.expression) {
                Ok(atoms) => {
                    let count: usize = atoms.iter().map(|a| a.atoms.len()).sum();
                    state.ui.selection = Selection::Atoms(atoms);
                    edit.message = format!("Selected {count} atoms");
                    redraw.set_all();
                }
                Err(error) => edit.message = error,
            }
        }
        if ui.button("Clear picked selection").clicked() {
            state.ui.selection = Selection::None;
            redraw.set_all();
        }
    });
    ui.horizontal(|ui| {
        ui.add(
            TextEdit::singleline(&mut edit.name)
                .hint_text("Selection name")
                .desired_width(160.),
        );
        if ui.button("Save / update name").clicked() {
            edit.message = match expression::save_named(state, edit.name.trim(), &edit.expression) {
                Ok(count) => {
                    redraw.set_all();
                    format!("Saved {} ({count} atoms)", edit.name.trim())
                }
                Err(error) => error,
            };
        }
        view_picker(ui, "new_layer_view", &mut edit.view);
        if ui.button("Add layer").clicked() {
            edit.message = match representations::add(state, &edit.expression, edit.view) {
                Ok(count) => format!("Added layer ({count} atoms)"),
                Err(error) => error,
            };
        }
    });
    if !edit.message.is_empty() {
        ui.label(&edit.message);
    }
    ui.separator();
    ui.collapsing("Named selections", |ui| {
        for saved in state.to_save.named_selections.clone() {
            ui.horizontal(|ui| {
                if ui
                    .button(&saved.name)
                    .on_hover_text(&saved.expression)
                    .clicked()
                {
                    edit.name = saved.name.clone();
                    edit.expression = saved.expression.clone();
                    match expression::evaluate(state, &format!("%{}", saved.name)) {
                        Ok(atoms) => {
                            state.ui.selection = Selection::Atoms(atoms);
                            redraw.set_all();
                        }
                        Err(error) => edit.message = error,
                    }
                }
                if ui.button("Use in expression").clicked() {
                    edit.expression = format!("%{}", saved.name);
                }
                if ui.button("Delete").clicked() {
                    edit.message = expression::delete_named(state, &saved.name)
                        .map(|()| format!("Deleted {}", saved.name))
                        .unwrap_or_else(|e| e);
                }
            });
        }
    });
    ui.collapsing("Default object views", |ui| {
        ui.label("Turn off an object's default view to display only its layers. Object visibility still controls all its layers.");
        let objects: Vec<_> = expression::objects(state).iter().map(|(_, _, m)| (expression::object_key(m), m.name().into_owned())).collect();
        for (key, name) in objects {
            let mut visible = !state.to_save.hidden_default_objects.contains(&key);
            if ui.checkbox(&mut visible, name).changed() {
                state.to_save.hidden_default_objects.retain(|k| k != &key);
                if !visible { state.to_save.hidden_default_objects.push(key); }
                state.to_save.save_flag = true;
                state.volatile.flags.update_ss_mesh = true;
                redraw.set_all();
            }
        }
    });
    ui.label("Representation layers");
    ScrollArea::vertical().max_height(400.).show(ui, |ui| {
        let mut remove = None;
        let mut apply = None;
        for (i, layer) in state.to_save.representations.iter_mut().enumerate() {
            ui.push_id(i, |ui| {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut layer.visible, "Visible");
                    ui.text_edit_singleline(&mut layer.name);
                    view_picker(ui, "view", &mut layer.view);
                    if ui.button("Delete").clicked() {
                        remove = Some(i);
                    }
                });
                ui.horizontal(|ui| {
                    let mut custom = layer.color.is_some();
                    if ui.checkbox(&mut custom, "Uniform color").changed() {
                        layer.color = if custom { Some([0.6, 0.75, 0.9]) } else { None };
                    }
                    if let Some(color) = &mut layer.color {
                        ui.color_edit_button_rgb(color);
                    }
                    ui.add(Slider::new(&mut layer.opacity, 0.0..=1.0).text("Opacity"));
                    ui.add_enabled(
                        !matches!(layer.view, MoleculeView::Ribbon | MoleculeView::Surface),
                        Slider::new(&mut layer.size, 0.1..=3.0).text("Radius"),
                    );
                });
                if edit.layer == Some(i) {
                    ui.text_edit_singleline(&mut edit.layer_expression);
                    ui.horizontal(|ui| {
                        if ui.button("Apply selection").clicked() {
                            apply = Some(i);
                        }
                        if ui.button("Cancel").clicked() {
                            edit.layer = None;
                        }
                    });
                } else if ui
                    .button(&layer.expression)
                    .on_hover_text("Edit this layer's selection")
                    .clicked()
                {
                    edit.layer = Some(i);
                    edit.layer_expression = layer.expression.clone();
                }
            });
        }
        if let Some(i) = apply {
            match expression::freeze_current(state, &edit.layer_expression).and_then(|text| {
                expression::evaluate(state, &text)?;
                Ok(text)
            }) {
                Ok(text) => {
                    state.to_save.representations[i].expression = text;
                    edit.layer = None;
                }
                Err(error) => edit.message = error,
            }
        }
        if let Some(i) = remove {
            state.to_save.representations.remove(i);
            edit.layer = None;
        }
    });
    for error in &state.volatile.representations.errors {
        ui.colored_label(egui::Color32::LIGHT_RED, error);
    }
    state.ui.selection_editor = edit;
}
