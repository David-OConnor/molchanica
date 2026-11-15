mod actions;

use bio_files::BondType;
use dynamics::Solvent;
use egui::{Color32, ComboBox, Key, Modifiers, RichText, Slider, Ui};
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene};
use mol_defs::molecules::{Bond, MolType, small::MoleculeSmall};
use na_seq::{
    Element,
    Element::{Carbon, Chlorine, Hydrogen, Nitrogen, Oxygen, Phosphorus, Sulfur},
};

use crate::{
    cam::{cam_reset_controls, set_fog},
    drawing::MoleculeView,
    file_io::save_mol,
    label, mol_editor,
    mol_editor::{
        add_atoms::{
            HCountOp, add_atom, add_from_template, adjust_h_count, h_count, index_of_sn,
            refresh_hydrogens,
        },
        exit_edit_mode, sync_md,
        templates::Template,
    },
    mol_manip,
    mol_manip::{ManipMode, set_manip},
    render::MESH_POCKET_START,
    selection::{Selection, ViewSelLevel},
    sfc_mesh::{apply_mesh_colors, get_mesh_colors},
    state::{State, StateUi},
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_INACTIVE, COLOR_REMOVE, misc,
        misc::{active_color, section_box},
        panels::{
            md_viewer::energy_disp, mol_data::selected_data_mol_editor,
            view::mesh_coloring_selector,
        },
        popup::pharmacophore::pharmacophore_edit_tools,
        util::color_egui_from_f32,
    },
    util::{RedrawFlags, handle_err},
};
// todo: Have "Check DBs" also check if a similar molecule exists in a DB.

// todo: Use what you like from [Maestro's](https://www.youtube.com/watch?v=JpOOI5qyTXU&list=PL3dxdlKx_PccSO0YWKJqUx6lfQRyvyyG0&index=6)

// These are in ps.
const DT_MIN: f32 = 0.00001;
const DT_MAX: f32 = 0.0001;

// Atom serial number labels, drawn over each atom when "Atom #" is enabled. Cyan: distinct from the
// (pink) pharmacophore feature numbering, and from the element, pocket, and H bond colors.
pub const ATOM_SN_LABEL_COLOR: (u8, u8, u8, u8) = (40, 220, 255, 255);

/// Keep editor shortcuts available over the scene and buttons, but leave text input and
/// modified shortcuts alone. Consuming a press prevents it from activating twice in a frame.
fn edit_hotkey_pressed(ui: &Ui, key: Key) -> bool {
    let transforming = ui.ctx().data(|data| {
        data.get_temp::<bool>(egui::Id::new("editor_transform_active"))
            .unwrap_or(false)
    });
    if ui.ctx().text_edit_focused()
        || ui.ctx().memory(|memory| memory.top_modal_layer().is_some())
        || (transforming && key != Key::F)
    {
        return false;
    }

    ui.input_mut(|input| {
        let pressed = input.events.iter().any(|event| {
            matches!(
                event,
                egui::Event::Key {
                    key: pressed_key,
                    pressed: true,
                    modifiers,
                    ..
                } if *pressed_key == key && !modifiers.alt && !modifiers.ctrl && !modifiers.command
            )
        });

        pressed && input.consume_key(Modifiers::NONE, key)
    })
}

fn change_el_button(
    sel: &Selection,
    el: Element,
    hotkey: Key,
    ui: &mut Ui,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    mol: &mut MoleculeSmall,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    manip_mode: ManipMode,
) {
    // N's default blue is too dark against the button's background.
    let color = if el == Nitrogen {
        Color32::from_rgb(130, 130, 255)
    } else {
        color_egui_from_f32(el.color())
    };

    if ui
        .button(RichText::new(el.to_letter()).color(color))
        .on_hover_text(format!(
            "(Hotkey: \"{hotkey:?}\") Change the selected atom's element to {el}"
        ))
        .clicked()
        || edit_hotkey_pressed(ui, hotkey)
        || actions::take(ui, &format!("element_{}", el.to_letter()))
    {
        let (mol_i, idxs) = match sel {
            Selection::AtomLig((mol_i, i)) => (*mol_i, vec![*i]),
            Selection::AtomsLig((mol_i, i)) => (*mol_i, i.to_vec()),
            _ => {
                eprintln!("Attempting to change an element with an invalid selection");
                return;
            }
        };

        // let mol = &mut editor.mol;

        // Serial numbers, not indices: re-populating hydrogens removes atoms, which renumbers the
        // ones after them.
        let sns: Vec<u32> = idxs
            .iter()
            .filter_map(|i| mol.common.atoms.get(*i))
            .map(|a| a.serial_number)
            .collect();

        for &sn in &sns {
            let Some(i) = index_of_sn(&mol.common, sn) else {
                continue;
            };
            mol.common.atoms[i].element = el;

            refresh_hydrogens(
                &mut mol.common,
                sn,
                entities,
                state_ui,
                engine_updates,
                manip_mode,
                &mol.components,
            );
        }

        let indices: Vec<_> = sns
            .iter()
            .filter_map(|&sn| index_of_sn(&mol.common, sn))
            .collect();
        state_ui.selection = match indices.len() {
            0 => Selection::None,
            1 => Selection::AtomLig((mol_i, indices[0])),
            _ => Selection::AtomsLig((mol_i, indices)),
        };

        engine_updates.entities = EntityUpdate::All;

        *redraw = true;
        *rebuild_md = true;
    }
}

pub(in crate::ui) fn editor(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    mut redraw: bool,
    ui: &mut Ui,
) {
    ui.ctx().data_mut(|data| {
        data.insert_temp(
            egui::Id::new("editor_transform_active"),
            state.volatile.mol_manip.mode != ManipMode::None,
        )
    });
    ui.horizontal_wrapped(|ui| {
        ui.strong("Molecule editor");
        actions::palette(state, ui);
        file_controls(state, scene, updates, ui, &mut redraw);
        actions::utilities(state, scene, updates, ui, &mut redraw);
    });

    if state.volatile.operating_mode != crate::state::OperatingMode::MolEditor {
        return;
    }

    ui.horizontal_wrapped(|ui| {
        section_box().show(ui, |ui| {
            // ui.horizontal_wrapped(|ui| {
            let mut cam_changed = false;

            // todo: The distances this function resets to may not be ideal for our use case
            // todo here. Adjust A/R.
            cam_reset_controls(state, scene, ui, updates, &mut cam_changed);
            // if ui.button("Reset cam").clicked() {
            //     scene.camera.position = lin_alg::f32::Vec3::new(0., 0., -INIT_CAM_DIST);
            //     scene.camera.orientation = Quaternion::new_identity();
            // }

            if cam_changed {
                set_fog(state, &mut scene.camera);
            }

            ui.add_space(COL_SPACING / 2.);

            // todo: This is a C+P from the main editor
            let color = if state.ui.atom_color_by_charge {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };
            if ui
                .button(RichText::new("Color by q").color(color))
                .on_hover_text(
                    "Color the atom by partial charge, instead of element-specific colors",
                )
                .clicked()
            {
                state.ui.atom_color_by_charge = !state.ui.atom_color_by_charge;
                state.ui.view_sel_level = ViewSelLevel::AtomBond;

                redraw = true;
            }
            // });
        });

        // C+P from main editor, with fewer options.
        // todo: If on one of the unavail-here view modes when entering editor,
        // todo: chagne it.
        section_box().show(ui, |ui| {
            ui.label("View:");
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Sticks,
                        MoleculeView::BallAndStick,
                        MoleculeView::SpaceFill,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                });

            if state.ui.mol_view != prev_view {
                redraw = true;
            }
        });

        let mut redraw_ = RedrawFlags::default();

        section_box().show(ui, |ui| {
            ui.label("Vis:");

            misc::toggle_btn_inv(
                &mut state.ui.visibility.hide_hydrogen,
                "H",
                "Show or hide Hydrogen atoms",
                ui,
                &mut redraw_,
            );

            misc::toggle_btn_inv(
                &mut state.ui.visibility.hide_pockets,
                "Pocket",
                "Show or hide the pocket",
                ui,
                &mut redraw_,
            );

            misc::toggle_btn_inv(
                &mut state.ui.visibility.hide_h_bonds,
                "H bonds",
                "Show or hide hydrogen bonds. For example, between a pocket and the molecule being edited.",
                ui,
                &mut redraw_,
            );

            // Not inverted: this flag is "show", unlike the "hide" flags above.
            misc::toggle_btn(
                &mut state.mol_editor.show_pharmacophore_renders,
                "Pharm",
                "Show or hide the molecule's 3D pharmacophore feature indicators.",
                ui,
                &mut redraw_,
            );

            misc::toggle_btn(
                &mut state.mol_editor.show_atom_sns,
                "Atom #",
                "Show or hide atom serial number text overlays.",
                ui,
                &mut redraw_,
            );
        });

        section_box().show(ui, |ui| {
            ui.strong("Simulate");
            let prev = state.mol_editor.md.running;
            misc::toggle_btn(
                &mut state.mol_editor.md.running,
                "MD running",
                "Start a molecular dynamics simulation of the selected molecules",
                ui,
                &mut redraw_,
            );

            if actions::take(ui, "simulate") {
                state.mol_editor.md.running = !state.mol_editor.md.running;
            }

            if state.mol_editor.md.running {
                let color = active_color(state.mol_editor.md.solvent != Solvent::None);
                if ui
                    .button(RichText::new("MD Water").color(color))
                    .on_hover_text("If enabled, use the explicit solvation model. If disabled, does not model water.")
                    .clicked()
                {
                    state.mol_editor.md.solvent = if state.mol_editor.md.solvent == Solvent::None {
                        Solvent::WaterOpc
                    } else {
                        Solvent::None
                    };

                    redraw = true;
                    state.mol_editor.md.rebuild_required = true;
                }
            }

            let started = !prev && state.mol_editor.md.running;

            if prev != state.mol_editor.md.running {
                let running = state.mol_editor.md.running;
                state.mol_editor.md.running = prev;
                if state.volatile.mol_manip.mode != ManipMode::None {
                    mol_manip::end_transform(state, scene, false, &mut Default::default(), updates);
                }
                state.mol_editor.md.running = running;
            }
            if started {
                mol_editor::ensure_dynamics(state);
            } else if !state.mol_editor.md.running {
                state.mol_editor.md.pull_target = None;
            }

            if state.mol_editor.md.running && state.mol_editor.md.rebuild_required {
                mol_editor::ensure_dynamics(state);
            }
        });

        // The editor has a single redraw path; any flag set by the toggles above triggers it.
        if redraw_.ligand || redraw_.pocket {
            redraw = true;
        }

        if ui.button("Cleanup geom")
            .on_hover_text("A fast, analytic approach to fixing unphysical geometry.")
            .clicked() || actions::take(ui, "cleanup") {
            set_manip(
                state,
                scene,
                &mut Default::default(),
                &mut false,
                ManipMode::None,
                updates,
            );

            state.mol_editor.mol.common.cleanup_geometry();
            sync_md(state);
            redraw = true;
        }

        if ui.button("Relax")
            .on_hover_text("Relax geometry; adjust atom positions to minimize energy, based on a standard small-molecule force field.")
            .clicked() || actions::take(ui, "relax") {
            set_manip(
                state,
                scene,
                &mut Default::default(),
                &mut false,
                ManipMode::None,
                updates,
            );

            mol_editor::relax(state, scene, updates);
        }


    });

    ui.horizontal_wrapped(|ui| {
        ui.strong("Build");
        edit_tools(state, scene, ui, updates, &mut redraw);

        // todo: Show/hide this button A/R
        if ui
            .button("Tautomer")
            .on_hover_text("Cycle through different tautomers for the selected molecule.")
            .clicked()
        {
            let tautometers = state.mol_editor.mol.common.find_tautomers();

            // todo: Only the first for now. Later, implement cycle
            if !tautometers.is_empty() {
                state.mol_editor.mol.common = tautometers[0].clone();

                // todo: For now, so we don't need to add positioning logic to the find_tautomers function.
                // todo: Only re-populate hydrogens on the changed atoms.
                // Snapshot serial numbers first: each refresh removes and re-adds atoms, so
                // indices -- and the atom count -- shift as we go.
                let sns: Vec<u32> = state
                    .mol_editor
                    .mol
                    .common
                    .atoms
                    .iter()
                    .map(|a| a.serial_number)
                    .collect();

                for sn in sns {
                    refresh_hydrogens(
                        &mut state.mol_editor.mol.common,
                        sn,
                        &mut scene.entities,
                        &mut state.ui,
                        updates,
                        state.volatile.mol_manip.mode,
                        &state.mol_editor.mol.components,
                    );
                }

                mol_editor::relax_after_edit(state, scene, updates);
                redraw = true;
            }
        }
    });

    ui.horizontal(|ui| {
        pharmacophore_edit_tools(state, scene, ui, updates, &mut redraw);

        ui.add_space(COL_SPACING);

        if let Some(pocket) = &mut state.mol_editor.mol.pharmacophore.pocket {
            let mut updated_coloring = false;
            mesh_coloring_selector(&mut state.ui.mesh_coloring, &mut updated_coloring, ui);

            if updated_coloring {
                let colors = get_mesh_colors(
                    &pocket.surface_mesh,
                    &pocket.common,
                    state.ui.mesh_coloring,
                    updates,
                );

                apply_mesh_colors(&mut pocket.surface_mesh, &colors);
                apply_mesh_colors(
                    &mut scene.meshes[MESH_POCKET_START + pocket.mesh_i_rel],
                    &colors,
                );
                updates.meshes = true;

                redraw = true;
            }
        }
    });

    ui.horizontal_wrapped(|ui| {
        ui.strong("Transform");
        transform_tools(state, scene, updates, ui, &mut redraw);
    });
    simulation_controls(state, ui);
    actions::status(state, ui);

    selected_data_mol_editor(state, &state.ui.selection, ui);

    // Prevents the UI from jumping when going between a selection and none.
    if state.ui.selection == Selection::None {
        ui.add_space(6.);
    }

    if redraw {
        mol_editor::redraw(
            &mut scene.entities,
            &state.mol_editor,
            &state.ui,
            state.volatile.mol_manip.mode,
            updates,
        );
    }
}

fn file_controls(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    redraw: &mut bool,
) {
    ui.menu_button("File", |ui| {
        if ui.button("Load molecule…").clicked() {
            state.volatile.dialogs.load.pick_file();
            ui.close();
        }
        if ui.button("Save molecule…").clicked() {
            if save_mol(
                &state.mol_editor.mol.common,
                MolType::Ligand,
                &mut state.volatile.dialogs.save,
            )
            .is_err()
            {
                handle_err(&mut state.ui, "Problem saving this file".to_owned());
            }
            ui.close();
        }
        if ui.button("Check PubChem").clicked() {
            mol_editor::check_dbs(state);
            ui.close();
        }
        if ui.button("Reassign atom numbers").clicked() {
            state.mol_editor.mol.common.reassign_sns();
            *redraw = true;
            ui.close();
        }
        ui.separator();
        if ui
            .button(RichText::new("New molecule").color(COLOR_REMOVE))
            .on_hover_text("Discard these edits and start a fresh molecule.")
            .clicked()
        {
            mol_editor::clear_editor(state, scene, updates);
            *redraw = true;
            ui.close();
        }
    });
    ui.menu_button("Finish editing", |ui| {
        let original = state
            .mol_editor
            .mol_i_in_state
            .filter(|&i| i < state.ligands.len());
        for (label, exit, update, discard) in [
            ("Add and keep editing", false, false, false),
            ("Update and keep editing", false, true, false),
            ("Discard and keep editing", false, false, true),
            ("Add and exit", true, false, false),
            ("Update and exit", true, true, false),
            ("Discard and exit", true, false, true),
        ] {
            if ui
                .add_enabled(!update || original.is_some(), egui::Button::new(label))
                .clicked()
            {
                match (exit, update, discard) {
                    (false, false, false) => mol_editor::clear_and_add(state, scene, updates),
                    (false, true, _) => {
                        mol_editor::clear_and_update(state, scene, updates, original.unwrap())
                    }
                    (false, false, true) => mol_editor::clear_editor(state, scene, updates),
                    (true, false, false) => mol_editor::exit_and_add(state, scene, updates),
                    (true, true, _) => {
                        mol_editor::exit_and_update(state, scene, updates, original.unwrap())
                    }
                    (true, false, true) => exit_edit_mode(state, scene, updates),
                }
                *redraw = true;
                ui.close();
            }
        }
    });
}

fn simulation_controls(state: &mut State, ui: &mut Ui) {
    if let Some(error) = &state.mol_editor.md.error {
        ui.colored_label(Color32::LIGHT_RED, error);
    }

    if state.mol_editor.md.md.is_none() {
        return;
    }

    ui.horizontal_wrapped(|ui| {
        section_box().show(ui, |ui| {
            ui.add(Slider::new(&mut state.mol_editor.md.steps_per_update, 1..=100).text("IMD speed"))
                .on_hover_text("Integration steps per update. Slower machines may run fewer steps to keep input responsive.");
            ui.add(Slider::new(&mut state.mol_editor.md.dt, DT_MIN..=DT_MAX).logarithmic(true).text("Step (ps)"))
                .on_hover_text("Integration timestep. Smaller steps resolve fast bond vibrations.");
            ui.add(Slider::new(&mut state.mol_editor.md.pull_strength, 1.0..=100.0).text("Pull strength"))
                .on_hover_text("Spring stiffness in kcal/mol/Å². Pull force is capped at 100 kcal/mol/Å.");
        });

        let indices = match &state.ui.selection {
            Selection::AtomLig((_, i)) => vec![*i],
            Selection::AtomsLig((_, indices)) => indices.clone(),
            _ => Vec::new(),
        };
        let pinned = &state.mol_editor.md.pinned_atoms;
        let all_pinned = !indices.is_empty() && indices.iter().all(|i| pinned.contains(i));
        if ui.add_enabled(!indices.is_empty(), egui::Button::new(if all_pinned { "Unpin selected" } else { "Pin selected" }))
            .on_hover_text("Hold selected atoms fixed during IMD. Pins clear after structural edits.")
            .clicked()
        {
            state.mol_editor.md.set_pinned(&indices, !all_pinned);
        }
        if ui.add_enabled(!state.mol_editor.md.pinned_atoms.is_empty(), egui::Button::new("Unpin all")).clicked() {
            let pinned = state.mol_editor.md.pinned_atoms.clone();
            state.mol_editor.md.set_pinned(&pinned, false);
        }

        if let Some(md) = &state.mol_editor.md.md {
            ui.label(format!("Time: {:.4} ps", md.time));
        }
        if let Some(snap) = &state.mol_editor.md.snap {
            energy_disp(snap, ui);
        }
    });

    ui.horizontal_wrapped(|ui| {
        if state.mol_editor.md.running {
            ui.label(
                "IMD: select an atom, press M, then drag to pull. Release to let go; Esc ends the pull.",
            );
        }

        let editor = &state.mol_editor;
        if !editor.md.pinned_atoms.is_empty() {
            let labels: Vec<_> = editor
                .md
                .pinned_atoms
                .iter()
                .filter_map(|&i| editor.mol.common.atoms.get(i))
                .map(|a| a.serial_number.to_string())
                .collect();
            ui.label(format!("Pinned atoms: {}", labels.join(", ")));
        }
        if let Some((i, target)) = editor.md.pull_target
            && let Some(atom) = editor.md.md.as_ref().and_then(|md| md.atoms.get(i))
        {
            let distance = (target - atom.posit).magnitude();
            let force = (distance * editor.md.pull_strength).min(100.);
            ui.label(format!("Pull: {distance:.2} Å · {force:.1} kcal/mol/Å"));
        }
    });
}

fn bond_edit_tools(
    bond_sel_is: &[usize],
    bonds: &mut [Bond],
    ui: &mut Ui,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    rebuild_ff_params: &mut bool,
) {
    let mut new_bond_type = None;

    section_box().show(ui, |ui| {
        if ui
            .button("-")
            .on_hover_text("(Hotkey: \"1\") Change this to a single bond.")
            .clicked()
            || edit_hotkey_pressed(ui, Key::Num1)
            || actions::take(ui, "bond_single")
        {
            new_bond_type = Some(BondType::Single);
        }
        if ui
            .button("=")
            .on_hover_text("(Hotkey: \"2\") Change this to a double bond.")
            .clicked()
            || edit_hotkey_pressed(ui, Key::Num2)
            || actions::take(ui, "bond_double")
        {
            new_bond_type = Some(BondType::Double);
        }
        if ui
            .button("Tr")
            .on_hover_text("(Hotkey: \"3\") Change this to a triple bond.")
            .clicked()
            || edit_hotkey_pressed(ui, Key::Num3)
            || actions::take(ui, "bond_triple")
        {
            new_bond_type = Some(BondType::Triple);
        }
        if ui
            .button("Ar")
            .on_hover_text("(Hotkey: \"4\") Change this to an aromatic.")
            .clicked()
            || edit_hotkey_pressed(ui, Key::Num4)
            || actions::take(ui, "bond_aromatic")
        {
            new_bond_type = Some(BondType::Aromatic);
        }
    });

    if let Some(bt) = new_bond_type {
        for b in bond_sel_is {
            bonds[*b].bond_type = bt;
        }

        *redraw = true;
        *rebuild_md = true;
        *rebuild_ff_params = true;
    }
}

/// Buttons for setting how many hydrogens an atom carries.
///
/// The automatic count assumes a neutral atom in its usual valence state, which covers most of
/// what you draw but not charged centres: an ammonium N wants four bonds, an azide's terminal N
/// wants none beyond the one it has, and a carboxylate O wants none. Those need setting by hand.
#[allow(clippy::too_many_arguments)]
fn h_count_tools(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    mol_i: usize,
    selected_idxs: &[usize],
) {
    let h_now: usize = selected_idxs
        .iter()
        .map(|i| h_count(&state.mol_editor.mol.common, *i))
        .sum();

    label!(ui, format!("H: {h_now}"), Color32::GRAY);

    let mut op = None;

    if ui
        .button("−H")
        .on_hover_text("Remove one hydrogen from the selected atom[s].")
        .clicked()
    {
        op = Some(HCountOp::Remove);
    }

    if ui
        .button("+H")
        .on_hover_text(
            "Add one hydrogen to the selected atom[s], past what its valence implies. E.g. \
            for an ammonium nitrogen. Does nothing if there's no room left around the atom.",
        )
        .clicked()
    {
        op = Some(HCountOp::Add);
    }

    if ui
        .button("H↺")
        .on_hover_text(
            "Reset the selected atom[s] to the number of hydrogens their element and bonds imply.",
        )
        .clicked()
    {
        op = Some(HCountOp::Auto);
    }

    let Some(op) = op else {
        return;
    };

    // Serial numbers, not indices: removing a hydrogen renumbers the atoms after it.
    let sns: Vec<u32> = selected_idxs
        .iter()
        .filter_map(|i| state.mol_editor.mol.common.atoms.get(*i))
        .map(|a| a.serial_number)
        .collect();

    for sn in &sns {
        adjust_h_count(
            &mut state.mol_editor.mol.common,
            *sn,
            op,
            &mut scene.entities,
            &mut state.ui,
            engine_updates,
            state.volatile.mol_manip.mode,
            &state.mol_editor.mol.components,
        );
    }

    // Re-point the selection at the same atoms; their indices may have moved.
    let new_idxs: Vec<usize> = sns
        .iter()
        .filter_map(|sn| index_of_sn(&state.mol_editor.mol.common, *sn))
        .collect();

    state.ui.selection = match new_idxs.len() {
        0 => Selection::None,
        1 => Selection::AtomLig((mol_i, new_idxs[0])),
        _ => Selection::AtomsLig((mol_i, new_idxs)),
    };

    state.mol_editor.mol.update_characterization();

    *redraw = true;
    *rebuild_md = true;
}

fn edit_tools(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
) {
    let mut bond_mode = false;
    let mut rebuild_md = false;

    // todo: Clone temp to avoid borrow problems.
    let (mol_i, selected_idxs) = match state.ui.selection.clone() {
        Selection::AtomLig((mol_i, i)) => (mol_i, vec![i]),
        Selection::AtomsLig((mol_i, i)) => (mol_i, i),
        Selection::BondLig((mol_i, bond_i)) => {
            bond_mode = true;
            (mol_i, vec![bond_i])
        }
        Selection::BondsLig((mol_i, bonds_i)) => {
            bond_mode = true;
            (mol_i, bonds_i)
        }
        _ => {
            // Vertical pad to prevent UI jumping
            ui.vertical(|ui| {
                ui.add_space(24.);
            });
            return;
        }
    };

    let count = if bond_mode {
        state.mol_editor.mol.common.bonds.len()
    } else {
        state.mol_editor.mol.common.atoms.len()
    };
    if selected_idxs.is_empty() || selected_idxs.iter().any(|&i| i >= count) {
        state.ui.selection = Selection::None;

        return;
    }

    if bond_mode {
        let mut rebuild_ff_params = false;
        bond_edit_tools(
            &selected_idxs,
            &mut state.mol_editor.mol.common.bonds,
            ui,
            redraw,
            &mut rebuild_md,
            &mut rebuild_ff_params,
        );

        if rebuild_ff_params {
            // Rebuild hydrogens on the two atoms the changed bond joins; its new order changes how
            // much room each has. By serial number, since the rebuild shifts indices.
            let sns: Vec<u32> = {
                let mol = &state.mol_editor.mol.common;
                let mut sns: Vec<_> = selected_idxs
                    .iter()
                    .filter_map(|&i| mol.bonds.get(i))
                    .flat_map(|bond| [bond.atom_0_sn, bond.atom_1_sn])
                    .collect();
                sns.sort_unstable();
                sns.dedup();

                sns
            };

            for sn in sns {
                refresh_hydrogens(
                    &mut state.mol_editor.mol.common,
                    sn,
                    &mut scene.entities,
                    &mut state.ui,
                    engine_updates,
                    state.volatile.mol_manip.mode,
                    &state.mol_editor.mol.components,
                );
            }

            state.mol_editor.rebuild_ff_related(&state.ff_param_set);
        }
    }

    if !bond_mode {
        section_box().show(ui, |ui| {
            if ui
                .button("Add Atom")
                .on_hover_text(
                    "(Hotkey: Tab) Add a Carbon atom. (Can change to other elements after)",
                )
                .clicked()
                || actions::take(ui, "add")
            {
                let sns: Vec<_> = selected_idxs
                    .iter()
                    .filter_map(|&i| state.mol_editor.mol.common.atoms.get(i))
                    .map(|a| a.serial_number)
                    .collect();
                for sn in sns {
                    let Some(atom_i) = index_of_sn(&state.mol_editor.mol.common, sn) else {
                        continue;
                    };
                    add_atom(
                        &mut state.mol_editor.mol.common,
                        &mut scene.entities,
                        atom_i,
                        Carbon,
                        BondType::Single,
                        None, // Inferred from the completed local bonding environment below.
                        Some(1.4), // todo
                        // Some(0.13),           // todo
                        None,
                        &mut state.ui,
                        engine_updates,
                        &mut scene.input_settings.control_scheme,
                        state.volatile.mol_manip.mode,
                        &state.mol_editor.mol.components,
                    );

                    state.mol_editor.mol.update_characterization();
                }
                rebuild_md = true;

                // `add_atom` draws only the new atoms, but rebuilding MD may reassign serial
                // numbers, and the atom labels need to reflect that.
                if state.mol_editor.show_atom_sns {
                    *redraw = true;
                }
            }

            if ui
                .button("Add, sel atom")
                .on_hover_text(
                    "Add a Carbon atom, and select it. Useful for quickly adding chains.",
                )
                .clicked()
                || actions::take(ui, "add_select")
            {
                let sns: Vec<_> = selected_idxs
                    .iter()
                    .filter_map(|&i| state.mol_editor.mol.common.atoms.get(i))
                    .map(|a| a.serial_number)
                    .collect();
                for sn in sns {
                    let Some(atom_i) = index_of_sn(&state.mol_editor.mol.common, sn) else {
                        continue;
                    };
                    let new_i = add_atom(
                        &mut state.mol_editor.mol.common,
                        &mut scene.entities,
                        atom_i,
                        Carbon,
                        BondType::Single,
                        None, // Inferred from the completed local bonding environment below.
                        Some(1.4), // todo
                        // Some(0.13),            // todo
                        None,
                        &mut state.ui,
                        engine_updates,
                        &mut scene.input_settings.control_scheme,
                        state.volatile.mol_manip.mode,
                        &state.mol_editor.mol.components,
                    );

                    if let Some(i) = new_i {
                        state.ui.selection = Selection::AtomLig((mol_i, i));

                        // `add_atom` handles individual redrawing, but here we need something, or the previous
                        // atom will still show as the selected color.
                        *redraw = true;
                    }
                }
                rebuild_md = true;
            }

            ui.add_space(COL_SPACING / 2.);

            for (el, hotkey) in [
                (Carbon, Key::C),
                (Hydrogen, Key::H),
                (Oxygen, Key::O),
                (Nitrogen, Key::N),
                (Sulfur, Key::S),
                (Phosphorus, Key::P),
                (Chlorine, Key::L),
            ] {
                let sel = state.ui.selection.clone(); // todo :/
                change_el_button(
                    &sel,
                    el,
                    hotkey,
                    ui,
                    &mut scene.entities,
                    &mut state.ui,
                    &mut state.mol_editor.mol,
                    engine_updates,
                    redraw,
                    &mut rebuild_md,
                    state.volatile.mol_manip.mode,
                );
            }

            ui.add_space(COL_SPACING / 2.);
            let selected_now = actions::selected_atoms(state);
            h_count_tools(
                state,
                scene,
                ui,
                engine_updates,
                redraw,
                &mut rebuild_md,
                mol_i,
                &selected_now,
            );
        });

        let selected_idxs = actions::selected_atoms(state);
        if selected_idxs.len() == 2
            && !state.mol_editor.mol.common.adjacency_list[selected_idxs[0]]
                .contains(&selected_idxs[1])
        {
            ui.add_space(COL_SPACING);
            if ui
                .button("Add bond")
                .on_hover_text("Add a covalent bond connecting the selected atoms")
                .clicked()
            {
                let atom_0_sn = state.mol_editor.mol.common.atoms[selected_idxs[0]].serial_number;
                let atom_1_sn = state.mol_editor.mol.common.atoms[selected_idxs[1]].serial_number;

                state.mol_editor.mol.common.bonds.push(Bond {
                    bond_type: BondType::Single, // The user can change after.
                    atom_0_sn,
                    atom_1_sn,
                    atom_0: selected_idxs[0],
                    atom_1: selected_idxs[1],
                    is_backbone: false,
                });

                *redraw = true;
                rebuild_md = true;
            }
        }
    }

    ui.add_space(COL_SPACING / 2.);

    template_section(
        state,
        ui,
        redraw,
        &mut rebuild_md,
        &mut scene.input_settings.control_scheme,
    );

    if let Selection::AtomLig((_, i)) = state.ui.selection
        && ui
            .button(RichText::new("Delete atom").color(COLOR_REMOVE))
            .on_hover_text("Delete")
            .clicked()
    {
        state.mol_editor.remove_atom(i);
        state.ui.selection = Selection::None;
        rebuild_md = true;
    }

    if rebuild_md {
        mol_editor::relax_after_edit(state, scene, engine_updates);
        *redraw = true;
    }
}

fn transform_tools(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    redraw: &mut bool,
) {
    let atom = matches!(state.ui.selection, Selection::AtomLig(_));
    let rotate = actions::can_rotate(state);
    let mut requested = None;
    let label = if state.mol_editor.md.running {
        "Pull atom · M"
    } else {
        "Move atom · M"
    };
    if ui
        .add_enabled(
            atom,
            egui::Button::new(RichText::new(label).color(active_color(matches!(
                state.volatile.mol_manip.mode,
                ManipMode::Move(_)
            )))),
        )
        .clicked()
        || actions::take(ui, "move")
    {
        requested = Some(ManipMode::Move((MolType::Ligand, 0)));
    }
    if ui
        .add_enabled(
            rotate,
            egui::Button::new(
                RichText::new("Rotate bond · R").color(active_color(matches!(
                    state.volatile.mol_manip.mode,
                    ManipMode::Rotate(_)
                ))),
            ),
        )
        .on_disabled_hover_text("Select a bond outside a ring. Pause IMD for direct rotation.")
        .clicked()
        || actions::take(ui, "rotate")
    {
        requested = Some(ManipMode::Rotate((MolType::Ligand, 0)));
    }
    if let Some(mode) = requested {
        let mut flags = RedrawFlags::default();
        set_manip(state, scene, &mut flags, &mut false, mode, updates);
        *redraw = true;
    }
    if state.volatile.mol_manip.mode != ManipMode::None {
        ui.label(mol_manip::status(state));
        if ui.button("Accept").clicked() {
            mol_manip::end_transform(state, scene, false, &mut Default::default(), updates);
            *redraw = true;
        }
        let cancel_label = if state.mol_editor.md.running {
            "Release"
        } else {
            "Cancel"
        };
        if ui.button(cancel_label).clicked() {
            mol_manip::end_transform(state, scene, true, &mut Default::default(), updates);
            *redraw = true;
        }
    }
}

fn template_section(
    state: &mut State,
    ui: &mut Ui,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    controls: &mut ControlScheme,
) {
    section_box().show(ui, |ui| {
        let (anchor_idxs, anchor_sns, r_aligners, next_sn, next_i, r_aligner_is) = {
            let mol_com = &state.mol_editor.mol.common;

            // todo: Perhaps add multiples if there are multiple atoms or bonds selected,
            // todo but for now, this will do.
            let anchor_idxs = match &state.ui.selection {
                Selection::AtomLig((_, i)) => vec![*i],
                // Selection::AtomsLig((_, items)) => items.clone(),
                Selection::BondLig((_, i)) => {
                    if *i >= state.mol_editor.mol.common.bonds.len() {
                        eprintln!("Error: Bond index out of bounds.");
                        state.ui.selection = Selection::None;
                        return;
                    }

                    let bond = &state.mol_editor.mol.common.bonds[*i];
                    vec![bond.atom_0, bond.atom_1]
                }
                // Selection::BondsLig((_, items)) => items.clone(),
                _ => return,
            };

            for &i in &anchor_idxs {
                if i >= mol_com.atoms.len() {
                    eprintln!("Error: Sel out of range for mol editor");
                    state.ui.selection = Selection::None;
                    return;
                }
            }

            let mut anchor_sns = Vec::with_capacity(anchor_idxs.len());
            for &i in &anchor_idxs {
                anchor_sns.push(mol_com.atoms[i].serial_number);
            }

            let next_i = mol_com.atoms.len();

            // todo: Don't continuously compute orientation; move the fects to add_from_temp... params,
            // todo, and
            // todo: This is crude.
            let (mut r_aligners, mut r_aligner_is) = (Vec::new(), Vec::new());

            // todo: Hardcoded 0 here. Currently awkward as this is for non-rings only.
            for bonded in &mol_com.adjacency_list[anchor_idxs[0]] {
                if mol_com.atoms[*bonded].element == Hydrogen {
                    continue;
                }

                // todo: Which one? If you even keep this setup.
                r_aligners.push(mol_com.atoms[*bonded].posit);
                r_aligner_is.push(*bonded);
            }

            (
                anchor_idxs,
                anchor_sns,
                r_aligners,
                state.mol_editor.mol.common.next_atom_sn,
                next_i,
                r_aligner_is,
            )
        };

        // Helper
        let mut add_t = |template: Template, abbrev, name| {
            let ring = matches!(
                template,
                Template::AromaticRing | Template::Cyclohexane | Template::PentaRing
            );
            let available = ring || (anchor_idxs.len() == 1 && !r_aligner_is.is_empty());
            let requested = actions::take(ui, &format!("template_{template:?}"));
            if available
                && (ui
                    .button(abbrev)
                    .on_hover_text(format!("Add a {name} at the current selection"))
                    .clicked()
                    || requested)
            {
                add_from_template(
                    &mut state.mol_editor.mol.common,
                    template,
                    &anchor_sns,
                    &anchor_idxs,
                    &r_aligner_is,
                    &r_aligners,
                    next_sn,
                    next_i,
                    redraw,
                    rebuild_md,
                    &mut state.ui,
                    controls,
                    state.volatile.mol_manip.mode,
                    &state.mol_editor.mol.components,
                );
            }
        };

        add_t(Template::Cooh, "−COOH", "carboxylic acid functional group");

        add_t(Template::Acetyl, "−COCH₃", "acetyl functional group");

        add_t(Template::Amide, "−NH₂", "amide functional group");

        add_t(Template::AromaticRing, "Ar", "benzene/aromatic ring");
        add_t(Template::Cyclohexane, "⬣", "cyclohexane ring");

        add_t(Template::PentaRing, "⬟", "5-atom ring");
    });
}
