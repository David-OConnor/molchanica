//! Shared editor for persistent selection expressions and additive display layers.

use egui::{
    ComboBox, Key, KeyboardShortcut, Modifiers, RichText, ScrollArea, Slider, TextEdit, Ui,
};
use mol_defs::molecules::aa_color;
use na_seq::{AaIdent, AminoAcid};

use crate::{
    drawing::{MoleculeView, representations},
    selection::{Selection, expression},
    state::State,
    ui::COLOR_ACTION,
    util::{RedrawFlags, make_egui_color},
};

#[derive(Default)]
pub struct SelectionEditor {
    name: String,
    expression: String,
    view: MoleculeView,
    message: String,
    layer: Option<usize>,
    layer_expression: String,
    builder: SelectionBuilder,
}

#[derive(Default)]
struct SelectionBuilder {
    all_when_empty: bool,
    classes: Vec<String>,
    objects: Vec<String>,
    chains: Vec<String>,
    residues: String,
    residue_names: Vec<String>,
    elements: Vec<String>,
    atom_names: String,
    subset: usize,
    no_hydrogens: bool,
    no_water: bool,
    whole_residues: bool,
    invert: bool,
    nearby: bool,
    distance: f64,
    near_picked: bool,
    picked_expression: String,
}

impl SelectionBuilder {
    fn filter_clauses(&self) -> Vec<String> {
        let mut clauses = Vec::new();
        if !self.classes.is_empty() {
            clauses.push(format!("({})", self.classes.join(" or ")));
        }
        if !self.objects.is_empty() {
            let objects: Vec<_> = self
                .objects
                .iter()
                .map(|o| format!("model {}", expression::quote(o)))
                .collect();
            clauses.push(format!("({})", objects.join(" or ")));
        }
        for (key, values) in [
            ("chain", &self.chains),
            ("resn", &self.residue_names),
            ("elem", &self.elements),
        ] {
            if !values.is_empty() {
                // Keep each chosen identifier visible in the generated expression.
                let terms: Vec<_> = values
                    .iter()
                    .map(|v| format!("{key} {}", expression::quote(v)))
                    .collect();
                clauses.push(format!("({})", terms.join(" or ")));
            }
        }
        if !self.residues.trim().is_empty() {
            clauses.push(format!("resi {}", expression::quote(self.residues.trim())));
        }
        if !self.atom_names.trim().is_empty() {
            clauses.push(format!(
                "name {}",
                expression::quote(self.atom_names.trim())
            ));
        }
        match self.subset {
            1 => clauses.push("backbone".into()),
            2 => clauses.push("sidechain".into()),
            3 => clauses.push("(protein and name CA)".into()),
            4 => clauses.push("guide".into()),
            _ => (),
        }
        if self.no_hydrogens {
            clauses.push("not hydrogens".into());
        }
        if self.no_water {
            clauses.push("not solvent".into());
        }
        if self.nearby {
            let target = if self.near_picked {
                self.picked_expression.as_str()
            } else {
                "ligand"
            };
            clauses.push(format!("(all within {} of ({target}))", self.distance));
        }
        clauses
    }

    fn expression(&self) -> String {
        let clauses = self.filter_clauses();
        let mut result = if clauses.is_empty() {
            if self.all_when_empty {
                "all".into()
            } else {
                "none".into()
            }
        } else {
            clauses.join(" and ")
        };
        if self.whole_residues {
            result = format!("byres ({result})");
        }
        if self.invert {
            result = format!("not ({result})");
        }
        result
    }
}

fn toggle(
    ui: &mut Ui,
    values: &mut Vec<String>,
    value: &str,
    label: impl Into<egui::WidgetText>,
) -> bool {
    let mut selected = values.iter().any(|v| v == value);
    if ui.toggle_value(&mut selected, label).changed() {
        values.retain(|v| v != value);
        if selected {
            values.push(value.into());
        }
        return true;
    }
    false
}

fn selection_builder(state: &State, ui: &mut Ui, builder: &mut SelectionBuilder) -> bool {
    let mut changed = false;
    egui::CollapsingHeader::new("Build a selection with controls")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Choices in each row are combined with OR; rows are combined with AND. Changes preview immediately.");
            ui.label("Using these controls replaces the expression below.");
            ScrollArea::vertical()
                .id_salt("selection_builder_scroll")
                .max_height(300.)
                .show(ui, |ui| {
                    changed |= builder_controls(state, ui, builder);
                });
        });
    changed
}

fn builder_controls(state: &State, ui: &mut Ui, builder: &mut SelectionBuilder) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("Types:");
        for (value, label) in [
            ("protein", "Protein"),
            ("nucleic", "DNA / RNA"),
            ("ligand", "Ligands"),
            ("lipid", "Lipids"),
            ("solvent", "Water"),
            ("metals", "Metals"),
        ] {
            changed |= toggle(ui, &mut builder.classes, value, label);
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Objects:");
        for (_, _, mol) in expression::objects(state) {
            changed |= toggle(
                ui,
                &mut builder.objects,
                &expression::object_key(&mol),
                mol.name().as_ref(),
            );
        }
    });
    let mut chains: Vec<_> = state
        .peptides
        .iter()
        .flat_map(|p| p.chains.iter().map(|c| c.id.clone()))
        .collect();
    chains.sort();
    chains.dedup();
    if !chains.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.label("Chains:");
            for chain in chains {
                let label = if chain.is_empty() { "(blank)" } else { &chain };
                changed |= toggle(ui, &mut builder.chains, &chain, label);
            }
        });
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Residue types:");
        for aa in [
            AminoAcid::Ala,
            AminoAcid::Arg,
            AminoAcid::Asn,
            AminoAcid::Asp,
            AminoAcid::Cys,
            AminoAcid::Gln,
            AminoAcid::Glu,
            AminoAcid::Gly,
            AminoAcid::His,
            AminoAcid::Ile,
            AminoAcid::Leu,
            AminoAcid::Lys,
            AminoAcid::Met,
            AminoAcid::Phe,
            AminoAcid::Pro,
            AminoAcid::Ser,
            AminoAcid::Thr,
            AminoAcid::Trp,
            AminoAcid::Tyr,
            AminoAcid::Val,
        ] {
            let name = aa.to_str(AaIdent::ThreeLetters).to_ascii_uppercase();
            let color = make_egui_color(aa_color(aa));

            changed |= toggle(
                ui,
                &mut builder.residue_names,
                &name,
                RichText::new(&name).color(color),
            );
        }
        if ui.button("Acidic").clicked() {
            builder.residue_names = vec!["ASP".into(), "GLU".into()];
            changed = true;
        }
        if ui.button("Basic").clicked() {
            builder.residue_names = vec!["ARG".into(), "LYS".into(), "HIS".into()];
            changed = true;
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Elements:");
        for element in [
            "C", "N", "O", "S", "P", "H", "F", "Cl", "Br", "I", "Na", "Mg", "Ca", "Fe", "Zn",
        ] {
            changed |= toggle(ui, &mut builder.elements, element, element);
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Residue numbers:");
        changed |= ui
            .add(
                TextEdit::singleline(&mut builder.residues)
                    .desired_width(150.)
                    .hint_text("e.g. 100-120+150"),
            )
            .changed();
        ui.label("Atom names:");
        changed |= ui
            .add(
                TextEdit::singleline(&mut builder.atom_names)
                    .desired_width(130.)
                    .hint_text("e.g. CA or C*"),
            )
            .changed();
    });
    ui.horizontal_wrapped(|ui| {
        let options = [
            "All atoms",
            "Backbone",
            "Side chains",
            "Alpha carbons",
            "Guide atoms",
        ];
        ComboBox::from_id_salt("builder_subset")
            .selected_text(options[builder.subset])
            .show_ui(ui, |ui| {
                for (i, label) in options.iter().enumerate() {
                    changed |= ui
                        .selectable_value(&mut builder.subset, i, *label)
                        .changed();
                }
            });
        changed |= ui
            .checkbox(&mut builder.no_hydrogens, "Exclude hydrogens")
            .changed();
        changed |= ui
            .checkbox(&mut builder.no_water, "Exclude water")
            .changed();
        changed |= ui
            .checkbox(&mut builder.whole_residues, "Whole residues")
            .changed();
        changed |= ui.checkbox(&mut builder.invert, "Invert").changed();
    });
    ui.horizontal_wrapped(|ui| {
        if ui
            .checkbox(&mut builder.nearby, "Within distance")
            .changed()
        {
            if builder.distance == 0. {
                builder.distance = 4.;
            }
            changed = true;
        }
        if builder.nearby {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut builder.distance)
                        .range(0.0..=100.0)
                        .speed(0.1)
                        .suffix(" Å"),
                )
                .changed();
            changed |= ui
                .selectable_value(&mut builder.near_picked, false, "of ligands")
                .changed();
            if ui
                .selectable_value(&mut builder.near_picked, true, "of picked atoms")
                .changed()
            {
                builder.picked_expression = expression::current_expression(state);
                changed = true;
            }
            if builder.near_picked && ui.button("Capture picked atoms").clicked() {
                builder.picked_expression = expression::current_expression(state);
                changed = true;
            }
        }
    });
    if changed && !builder.filter_clauses().is_empty() {
        builder.all_when_empty = false;
    }
    ui.horizontal(|ui| {
        if ui.button("Reset controls / select all").clicked() {
            *builder = SelectionBuilder {
                all_when_empty: true,
                ..Default::default()
            };
            changed = true;
        }
        if ui.button("Select none").clicked() {
            *builder = SelectionBuilder::default();
            changed = true;
        }
    });
    changed
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

    ui.horizontal_wrapped(|ui| {
        ui.label("PyMOL-style selections. Enter previews; Shift+Enter adds a line.");
        ui.hyperlink_to(
            "Syntax / examples",
            "https://www.athanorlab.com/docs/view_select.html#selection-expressions",
        );
    });
    ui.label("Use select name, expression to save a name, e.g. select alpha_carbons, protein and name CA.");

    let builder_changed = selection_builder(state, ui, &mut edit.builder);
    if builder_changed {
        edit.expression = edit.builder.expression();
    }

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
    let input = ui.add(
        TextEdit::multiline(&mut edit.expression)
            .id(ui.make_persistent_id("selection_expression"))
            .return_key(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter))
            .desired_rows(2)
            .desired_width(f32::INFINITY)
            .hint_text("Selection expression"),
    );
    let enter = input.has_focus()
        && ui.input_mut(|i| {
            // consume_key ignores extra Shift/Alt modifiers; only plain Enter submits.
            let mut pressed = false;
            i.events.retain(|event| {
                let submit = matches!(
                    event,
                    egui::Event::Key {
                        key: Key::Enter,
                        pressed: true,
                        modifiers,
                        ..
                    } if *modifiers == Modifiers::NONE
                );
                pressed |= submit;
                !submit
            });
            pressed
        });
    let mut preview = builder_changed || enter;
    ui.horizontal(|ui| {
        preview |= ui
            .button(RichText::new("Select / preview").color(COLOR_ACTION))
            .clicked();
        if ui.button("Clear picked selection").clicked() {
            state.ui.selection = Selection::None;
            redraw.set_all();
        }
    });
    if preview {
        edit.message = match expression::select(state, &edit.expression) {
            Ok(count) => {
                redraw.set_all();
                if let Ok((Some(name), _)) = expression::selection_input(&edit.expression) {
                    edit.name = name.into();
                    format!("Saved {name} ({count} atoms)")
                } else {
                    format!("Selected {count} atoms")
                }
            }
            Err(error) => error,
        };
    }
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
