use egui::{Color32, Id, Key, RichText, Ui};
use graphics::{ControlScheme, EngineUpdates, FWD_VEC, Scene};
use lin_alg::f64::Vec3;

use crate::{
    cam,
    inspection::{MeasureKind, measure},
    mol_manip::{self, ManipMode},
    selection::Selection,
    state::State,
};

#[derive(Clone, Default)]
struct Palette {
    open: bool,
    query: String,
    selected: usize,
}

pub(super) fn take(ui: &Ui, action: &str) -> bool {
    ui.ctx().data_mut(|data| {
        let id = Id::new("editor_action");
        if data.get_temp::<String>(id).as_deref() == Some(action) {
            data.remove::<String>(id);

            return true;
        }

        false
    })
}

pub(super) fn palette(state: &State, ui: &mut Ui) {
    ui.ctx()
        .data_mut(|data| data.remove::<String>(Id::new("editor_action")));
    let id = Id::new("editor_palette");
    let mut palette = ui
        .ctx()
        .data_mut(|data| data.get_temp::<Palette>(id).unwrap_or_default());
    let requested =
        super::edit_hotkey_pressed(ui, Key::F3) || ui.button("Search actions · F3").clicked();
    if requested {
        palette.open = true;
        palette.query.clear();
        palette.selected = 0;
    }

    if palette.open {
        let atom = matches!(
            state.ui.selection,
            Selection::AtomLig(_) | Selection::AtomsLig(_)
        );
        let bond = matches!(
            state.ui.selection,
            Selection::BondLig(_) | Selection::BondsLig(_)
        );
        let anchor = matches!(
            state.ui.selection,
            Selection::AtomLig(_) | Selection::BondLig(_)
        );
        let functional_group = match state.ui.selection {
            Selection::AtomLig((_, i)) => state
                .mol_editor
                .mol
                .common
                .adjacency_list
                .get(i)
                .is_some_and(|neighbors| {
                    neighbors.iter().any(|&j| {
                        state.mol_editor.mol.common.atoms[j].element != na_seq::Element::Hydrogen
                    })
                }),
            _ => false,
        };
        let has_selection = !selected_atoms(state).is_empty();
        let mut commands = vec![
            ("frame", "Frame selection / molecule", "F", true),
            ("neighbors", "Select bonded neighbors", "", has_selection),
            ("select_all", "Select all atoms", "", true),
            ("deselect", "Deselect", "Esc", has_selection),
            ("add", "Add carbon atom", "Tab", atom),
            ("add_select", "Add carbon and select it", "", atom),
            (
                "move",
                "Move / pull atom",
                "M",
                matches!(state.ui.selection, Selection::AtomLig(_)),
            ),
            ("rotate", "Rotate around bond", "R", can_rotate(state)),
            ("relax", "Relax geometry", "", true),
            ("cleanup", "Clean up geometry", "", true),
            ("simulate", "Start / pause IMD", "", true),
        ];
        for (action, label, shortcut) in [
            ("element_C", "Element: carbon", "C"),
            ("element_H", "Element: hydrogen", "H"),
            ("element_N", "Element: nitrogen", "N"),
            ("element_O", "Element: oxygen", "O"),
            ("element_S", "Element: sulfur", "S"),
            ("element_P", "Element: phosphorus", "P"),
            ("element_Cl", "Element: chlorine", "L"),
        ] {
            commands.push((action, label, shortcut, atom));
        }
        for (action, label, shortcut) in [
            ("bond_single", "Bond: single", "1"),
            ("bond_double", "Bond: double", "2"),
            ("bond_triple", "Bond: triple", "3"),
            ("bond_aromatic", "Bond: aromatic", "4"),
        ] {
            commands.push((action, label, shortcut, bond));
        }
        for (action, label) in [
            ("template_Cooh", "Add carboxylic acid"),
            ("template_Acetyl", "Add acetyl group"),
            ("template_Amide", "Add NH2 group"),
            ("template_AromaticRing", "Add aromatic ring"),
            ("template_Cyclohexane", "Add cyclohexane ring"),
            ("template_PentaRing", "Add five-membered ring"),
        ] {
            let ring = matches!(
                action,
                "template_AromaticRing" | "template_Cyclohexane" | "template_PentaRing"
            );
            commands.push((
                action,
                label,
                "",
                if ring { anchor } else { functional_group },
            ));
        }

        let mut chosen = None;
        let modal = egui::Modal::new(id).show(ui.ctx(), |ui| {
            ui.set_min_width(380.);
            ui.horizontal(|ui| {
                ui.heading("Editor actions");
                if ui.button("Close").clicked() {
                    ui.close();
                }
            });
            let response = ui.add(
                egui::TextEdit::singleline(&mut palette.query)
                    .hint_text("Search commands…")
                    .desired_width(f32::INFINITY),
            );
            if requested {
                response.request_focus();
            }
            if response.changed() {
                palette.selected = 0;
            }

            let query = palette.query.to_lowercase();
            let matches: Vec<_> = commands
                .iter()
                .filter(|(_, label, _, enabled)| *enabled && label.to_lowercase().contains(&query))
                .collect();
            palette.selected = palette.selected.min(matches.len().saturating_sub(1));
            if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, Key::ArrowDown)) {
                palette.selected = (palette.selected + 1).min(matches.len().saturating_sub(1));
            }
            if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, Key::ArrowUp)) {
                palette.selected = palette.selected.saturating_sub(1);
            }
            if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, Key::Enter))
                && let Some(command) = matches.get(palette.selected)
            {
                chosen = Some(command.0.to_owned());
            }
            egui::ScrollArea::vertical()
                .max_height(300.)
                .show(ui, |ui| {
                    for (i, (action, label, shortcut, _)) in matches.iter().enumerate() {
                        let response = ui.selectable_label(
                            palette.selected == i,
                            format!("{label}    {shortcut}"),
                        );
                        if response.clicked() {
                            chosen = Some((*action).to_owned());
                        }
                        if i == palette.selected {
                            response.scroll_to_me(None);
                        }
                    }
                    if matches.is_empty() {
                        ui.label("No available commands match.");
                    }
                });
            ui.small("Up/Down choose · Enter run · Esc close");
            if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, Key::Escape)) {
                palette.open = false;
            }
        });

        palette.open &= !modal.should_close();
        if let Some(chosen) = chosen {
            palette.open = false;
            ui.ctx()
                .data_mut(|data| data.insert_temp(Id::new("editor_action"), chosen));
        }
    }

    ui.ctx().data_mut(|data| data.insert_temp(id, palette));
}

pub(super) fn selected_atoms(state: &State) -> Vec<usize> {
    let mol = &state.mol_editor.mol.common;
    let mut result = match &state.ui.selection {
        Selection::AtomLig((_, i)) => vec![*i],
        Selection::AtomsLig((_, indices)) => indices.clone(),
        Selection::BondLig((_, i)) => mol
            .bonds
            .get(*i)
            .map(|b| vec![b.atom_0, b.atom_1])
            .unwrap_or_default(),
        Selection::BondsLig((_, indices)) => indices
            .iter()
            .filter_map(|&i| mol.bonds.get(i))
            .flat_map(|b| [b.atom_0, b.atom_1])
            .collect(),
        _ => Vec::new(),
    };
    result.retain(|&i| i < mol.atoms.len());
    let mut seen = std::collections::HashSet::new();
    result.retain(|i| seen.insert(*i));

    result
}

pub(super) fn can_rotate(state: &State) -> bool {
    let Selection::BondLig((_, i)) = state.ui.selection else {
        return false;
    };

    !state.mol_editor.md.running
        && state
            .mol_editor
            .mol
            .common
            .bonds
            .get(i)
            .is_some_and(|b| !b.in_a_cycle(&state.mol_editor.mol.common.adjacency_list))
}

pub(super) fn utilities(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    redraw: &mut bool,
) {
    let selected = selected_atoms(state);
    if ui
        .button("Frame · F")
        .on_hover_text("Frame the selection, or the whole molecule when nothing is selected.")
        .clicked()
        || super::edit_hotkey_pressed(ui, Key::F)
        || take(ui, "frame")
    {
        let positions = &state.mol_editor.mol.common.atom_posits;
        let points: Vec<_> = if selected.is_empty() {
            positions.clone()
        } else {
            selected
                .iter()
                .filter_map(|&i| positions.get(i).copied())
                .collect()
        };
        if !points.is_empty() {
            let center = points
                .iter()
                .copied()
                .fold(Vec3::new_zero(), |sum, p| sum + p)
                / points.len() as f64;
            let radius = points
                .iter()
                .map(|p| (*p - center).magnitude())
                .fold(1.5_f64, f64::max) as f32;
            let distance = cam::frame_dist(&scene.camera, radius);
            scene.camera.position = lin_alg::f32::Vec3::from(center)
                - scene.camera.orientation.rotate_vec(FWD_VEC) * distance;
            let controls = ControlScheme::Arc {
                center: center.into(),
            };
            if state.volatile.mol_manip.mode == ManipMode::None {
                scene.input_settings.control_scheme = controls;
            } else {
                state.volatile.control_scheme_prev = controls;
            }
            cam::set_fog(state, &mut scene.camera);
            state.ui.cam_snapshot = None;
            updates.camera = true;
        }
    }

    let neighbors = ui
        .add_enabled(!selected.is_empty(), egui::Button::new("Bonded neighbors"))
        .on_hover_text("Grow selection by one bond.")
        .clicked()
        || take(ui, "neighbors");
    let all = take(ui, "select_all");
    let deselect = take(ui, "deselect");
    if neighbors || all || deselect {
        mol_manip::end_transform(state, scene, false, &mut Default::default(), updates);
        let mol = &state.mol_editor.mol.common;
        let mut indices = selected.clone();
        if neighbors {
            for &i in &selected {
                for &j in &mol.adjacency_list[i] {
                    if !indices.contains(&j) {
                        indices.push(j);
                    }
                }
            }
        } else if all {
            indices = (0..mol.atoms.len()).collect();
        } else {
            indices.clear();
        }
        state.ui.selection = match indices.len() {
            0 => Selection::None,
            1 => Selection::AtomLig((0, indices[0])),
            _ => Selection::AtomsLig((0, indices)),
        };
        *redraw = true;
    }
}

pub(super) fn status(state: &State, ui: &mut Ui) {
    let indices = selected_atoms(state);
    let mol = &state.mol_editor.mol.common;
    let mode = match state.volatile.mol_manip.mode {
        ManipMode::Move(_) if state.mol_editor.md.running => "Pull",
        ManipMode::Move(_) => "Move",
        ManipMode::Rotate(_) => "Rotate bond",
        ManipMode::None => "Select",
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(mode).strong());
        ui.label(format!(
            "{} atoms · {} bonds · {} selected atoms",
            mol.atoms.len(),
            mol.bonds.len(),
            indices.len()
        ));
        if let Some(kind) = match indices.len() {
            2 => Some(MeasureKind::Distance),
            3 => Some(MeasureKind::Angle),
            4 => Some(MeasureKind::Dihedral),
            _ => None,
        } {
            let points: Vec<_> = indices.iter().map(|&i| mol.atom_posits[i]).collect();
            let order: Vec<_> = indices
                .iter()
                .map(|&i| mol.atoms[i].serial_number.to_string())
                .collect();
            let value = measure(kind, &points)
                .map(|v| {
                    format!(
                        "{v:.2}{}",
                        if kind == MeasureKind::Distance {
                            " Å"
                        } else {
                            "°"
                        }
                    )
                })
                .unwrap_or_else(|| "undefined".to_owned());
            ui.colored_label(
                Color32::LIGHT_BLUE,
                format!("{} ({}): {value}", kind.name(), order.join("–")),
            )
            .on_hover_text("Measured in selection order. Shift + right-click to add atoms.");
        }
        let hints =
            if state.mol_editor.md.running && state.volatile.mol_manip.mode != ManipMode::None {
                "Drag to pull · Release to let go · X/Y/Z axis · Esc end pull"
            } else if state.volatile.mol_manip.mode != ManipMode::None {
                "X/Y/Z axis · Shift+axis plane · Ctrl snap · Shift fine · Enter accept · Esc cancel"
            } else if state.ui.selection.is_bond() {
                "1/2/3/4 bond order · R rotate · F frame · F3 search"
            } else if !indices.is_empty() {
                "Tab add · C/N/O/S/P/H/L element · M move · Delete remove · F frame · F3 search"
            } else {
                "Right-click select · Shift+right-click extend · F frame all · F3 search"
            };
        ui.label(RichText::new(hints).color(Color32::GRAY));
    });
}
