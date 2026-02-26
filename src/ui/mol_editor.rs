use std::sync::atomic::Ordering;

use bio_files::BondType;
use egui::{Color32, ComboBox, RichText, Slider, Ui};
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene};
use na_seq::{
    Element,
    Element::{Carbon, Chlorine, Hydrogen, Nitrogen, Oxygen, Phosphorus, Sulfur},
};

use crate::{
    cam::cam_reset_controls,
    drawing::MoleculeView,
    mol_editor,
    mol_editor::{
        add_atoms::{add_atom, add_from_template, populate_hydrogens_on_atom, remove_hydrogens},
        exit_edit_mode, sync_md,
        templates::Template,
    },
    mol_manip,
    mol_manip::ManipMode,
    molecules::{Bond, MolIdent, MolType, small::MoleculeSmall},
    render::MESH_POCKET_START,
    selection::{Selection, ViewSelLevel},
    sfc_mesh::{apply_mesh_colors, get_mesh_colors},
    state::{State, StateUi},
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_INACTIVE,
        md::energy_disp,
        misc,
        misc::{active_color, section_box},
        mol_data::selected_data,
        pharmacophore::pharmacophore_edit_tools,
        util::color_egui_from_f32,
        view::mesh_coloring_selector,
    },
    util::{RedrawFlags, handle_err},
};
// todo: Check DBs (with a button maybe?) to see if the molecule exists in a DB already, or if
// todo a similar one does.

// todo: Use what you like from [Maestro's](https://www.youtube.com/watch?v=JpOOI5qyTXU&list=PL3dxdlKx_PccSO0YWKJqUx6lfQRyvyyG0&index=6)

// These are in ps.
const DT_MIN: f32 = 0.00001;
const DT_MAX: f32 = 0.0001; // No more than 0.002 for stability. currently 0.5fs.

// Higher may be relax more, but takes longer. We aim for a small time so it can be done without
// a noticeable lag.
const MAX_RELAX_ITERS: usize = 300;

fn change_el_button(
    // atoms: &mut [Atom],
    // editor: &mut MolEditorState,
    sel: &Selection,
    el: Element,
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
        .on_hover_text(format!("Change the selected atom's element to {el}"))
        .clicked()
    {
        let idxs = match sel {
            Selection::AtomLig((_, i)) => vec![*i],
            Selection::AtomsLig((_, i)) => i.to_vec(),
            _ => {
                eprintln!("Attempting to change an element with an invalid selection");
                return;
            }
        };

        // let mol = &mut editor.mol;

        for i in idxs {
            mol.common.atoms[i].element = el;

            remove_hydrogens(&mut mol.common, i);
            populate_hydrogens_on_atom(
                &mut mol.common,
                i,
                entities,
                state_ui,
                engine_updates,
                manip_mode,
                &mol.components,
            );
        }

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
                state.ui.view_sel_level = ViewSelLevel::Atom;

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

        section_box().show(ui, |ui| {
            ui.label("Vis:");

            misc::toggle_btn_inv(
                &mut state.ui.visibility.hide_hydrogen,
                "H",
                "Show or hide Hydrogen atoms",
                ui,
                &mut redraw,
            );

            misc::toggle_btn_inv(
                &mut state.ui.visibility.hide_pockets,
                "Pocket",
                "Show or hide the pocket",
                ui,
                &mut redraw,
            );

            misc::toggle_btn_inv(
                &mut state.ui.visibility.hide_h_bonds,
                "H bonds",
                "Show or hide hydrogen bonds. For example, between a pocket and the molecule being edited.",
                ui,
                &mut redraw,
            );
        });

        section_box().show(ui, |ui| {
            let prev = state.mol_editor.md.running;
            misc::toggle_btn(
                &mut state.mol_editor.md.running,
                "MD running",
                "Start a molecular dynamics simulation of the selected molecules",
                ui,
                &mut redraw,
            );

            if state.mol_editor.md.running {
                let color = active_color(!state.mol_editor.md.skip_water);
                if ui
                    .button(RichText::new("MD Water").color(color))
                    .on_hover_text("If enabled, use the explicit solvation model. If disabled, does not model water.")
                    .clicked()
                {
                    state.mol_editor.md.skip_water = !state.mol_editor.md.skip_water;

                    redraw = true;
                    state.mol_editor.md.rebuild_required = true;
                }
            }

            let started = !prev && state.mol_editor.md.running;

            if started && (state.mol_editor.md.rebuild_required || state.mol_editor.md.md.is_none()) {
                match mol_editor::build_dynamics(
                    &state.dev,
                    &mut state.mol_editor,
                    &state.ff_param_set,
                    &state.to_save.md_config,
                ) {
                    Ok(d) => state.mol_editor.md.md = Some(d),
                    Err(e) => eprintln!("Problem setting up dynamics for the editor: {e:?}"),
                }
                state.mol_editor.md.rebuild_required = false;
            }
        });

        if ui.button(RichText::new("Clear all").color(Color32::LIGHT_RED))
            .on_hover_text("Delete all atoms; start fresh")
            .clicked() {

            // state.mol_editor.mol.common = Default::default();

            state.mol_editor.clear_mol(&mut state.ui.selection);

            state.mol_editor.md.rebuild_required = true;
            state.mol_editor.rebuild_ff_related(&state.ff_param_set);

            redraw = true;
        }

        ui.add_space(COL_SPACING / 2.);
        // todo: implement
        if ui.button("Metadata")
            .on_hover_text("View and edit metadata for this molecule. This will be stored in the file when saved.")
            .clicked() {}

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Save"))
            .on_hover_text("Save to a Mol2, SDF, or PDBQT file")
            .clicked()
        {
            if state
                .mol_editor
                .mol
                .common
                .save(MolType::Ligand, &mut state.volatile.dialogs.save)
                .is_err()
            {
                handle_err(&mut state.ui, "Problem saving this file".to_owned());
            }
        }
        if ui
            .button(RichText::new("Load"))
            .on_hover_text("Load from a Mol2, SDF XYZ etc file")
            .clicked()
        {
            state.volatile.dialogs.load.pick_file();
        }

        if ui
            .button(RichText::new("Re-assign SNs"))
            .on_hover_text("Reset atom serial numbers to be sequential without gaps.")
            .clicked()
        {
            state.mol_editor.mol.common.reassign_sns();
        }

        if ui.button("Relax")
            .on_hover_text("Relax geometry; adjust atom positions to minimize energy.")
            .clicked() {
            if state.mol_editor.md.md.is_none() {
                match mol_editor::build_dynamics(
                    &state.dev,
                    &mut state.mol_editor,
                    &state.ff_param_set,
                    &state.to_save.md_config,
                ) {
                    Ok(md) => {
                        state.mol_editor.md.md = Some(md);
                    }
                    Err(e) => eprintln!("Problem setting up dynamics for the editor: {e:?}"),
                }
            }

            if let Some(md) = &mut state.mol_editor.md.md {
                md.minimize_energy(&state.dev, MAX_RELAX_ITERS, None); // todo: Iters A/R.
                state.mol_editor.load_atom_posits_from_md(&mut scene.entities, &state.ui,
                                                          updates, state.volatile.mol_manip.mode, );
            }
        }

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Exit / add").color(Color32::LIGHT_RED))
            .on_hover_text("Exit the molecule editor, and load the edited molecule.")
            .clicked()
        {
            state.mol_editor.mol.common.reassign_sns();
            state.mol_editor.mol.update_characterization();

            // Load the edited molecule back into the state.
            state.ligands.push(
                state.mol_editor.mol.clone()
            );

            exit_edit_mode(state, scene, updates);
        }

        if let Some(mol_i) = state.mol_editor.mol_i_in_state && mol_i < state.ligands.len() {
            if ui
                .button(RichText::new("Exit / update").color(Color32::LIGHT_RED))
                .on_hover_text("Exit the molecule editor, and update the loaded molecule with changes made.")
                .clicked()
            {
                state.mol_editor.mol.common.reassign_sns();
                    // Load the edited molecule back into the state.
                    state.ligands[mol_i].common.atoms = state.mol_editor.mol.common.atoms.clone();
                    state.ligands[mol_i].common.bonds = state.mol_editor.mol.common.bonds.clone();
                    state.ligands[mol_i].pharmacophore = state.mol_editor.mol.pharmacophore.clone();

                    state.ligands[mol_i].common.build_adjacency_list();
                    state.ligands[mol_i].common.reset_posits();

                    state.ligands[mol_i].update_characterization();

                exit_edit_mode(state, scene, updates);
            }
        }

        if ui
            .button(RichText::new("Exit / discard").color(Color32::LIGHT_RED))
            .on_hover_text("Exit the mol editor, discarding all unsaved changes.")
            .clicked()
        {
            exit_edit_mode(state, scene, updates);
        }
    });

    ui.horizontal(|ui| {
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
                for i in 0..state.mol_editor.mol.common.atoms.len() {
                    // Re-populate hydrogens on any atoms bonded to this.
                    remove_hydrogens(&mut state.mol_editor.mol.common, i);
                    populate_hydrogens_on_atom(
                        &mut state.mol_editor.mol.common,
                        i,
                        &mut scene.entities,
                        &mut state.ui,
                        updates,
                        state.volatile.mol_manip.mode,
                        &state.mol_editor.mol.components,
                    );
                }

                sync_md(state);
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

    ui.horizontal(|ui| {
        for ident in &state.mol_editor.mol.idents {
            if let MolIdent::Smiles(smiles) = ident {
                ui.label(RichText::new(smiles));
                break;
            }
        }

        if state.mol_editor.md.md.is_some() {
            section_box().show(ui, |ui| {
                ui.label("MD speed:");

                ui.spacing_mut().slider_width = 200.;
                ui.add(Slider::new(
                    &mut state.mol_editor.md.dt,
                    DT_MIN..=DT_MAX,
                ))
                    .on_hover_text("Set the simulation ratio compared to normal time.");

                ui.add_space(COL_SPACING);

                if let Some(snap) = &state.mol_editor.md.snap {
                    energy_disp(snap, ui);
                }
            });

            let ratio_help = "The ratio of simulation time to real time. A higher value means \
            the sim is running faster relative to reality. If 1×10-¹⁵, it means for every second viewing in real time,\
            the simulation runs 10¹⁵ of computed time.";

            // todo: Cache; don't compute each frame.
            // See the dt field doc comments for how we get this computation. We insert an additional
            // factor of 10e5 to make the value more readable.
            // 10e5: 10e3 for ms run interval. 10e3 to get between 10e-12 (input dt) and 10e-15 (displayed value)
            // todo: Still an unaccounted for factor of 10...
            let ratio = state.mol_editor.md.dt * 10e5 / state.mol_editor.md.time_between_runs;
            ui.label(format!("Ratio: {ratio:.1}×10-¹⁵")).on_hover_text(ratio_help);
        }
    });

    // This trick prevents a clone.
    let mol = std::mem::take(&mut state.mol_editor.mol); // move out, leave default in place
    selected_data(state, &state.ui.selection, ui);
    state.mol_editor.mol = mol;

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
        );
        updates.entities = EntityUpdate::All;
    }
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
            .on_hover_text("Change this to a single bond.")
            .clicked()
        {
            new_bond_type = Some(BondType::Single);
        }
        if ui
            .button("=")
            .on_hover_text("Change this to a double bond.")
            .clicked()
        {
            new_bond_type = Some(BondType::Double);
        }
        if ui
            .button("Tr")
            .on_hover_text("Change this to a triple bond.")
            .clicked()
        {
            new_bond_type = Some(BondType::Triple);
        }
        if ui
            .button("Ar")
            .on_hover_text("Change this to an aromatic.")
            .clicked()
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
            // Rebuild hydrogens on the changed bond atoms.
            // for (i, atom) in state.mol_editor.mol.common.atoms.iter().enumerate() {
            for i in 0..state.mol_editor.mol.common.atoms.len() {
                let bond = &state.mol_editor.mol.common.bonds[selected_idxs[0]];
                if bond.atom_0 != i && bond.atom_1 != i {
                    continue;
                }

                // Re-populate hydrogens on any atoms bonded to this.
                remove_hydrogens(&mut state.mol_editor.mol.common, i);
                populate_hydrogens_on_atom(
                    &mut state.mol_editor.mol.common,
                    i,
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
            {
                for atom_i in &selected_idxs {
                    add_atom(
                        &mut state.mol_editor.mol.common,
                        &mut scene.entities,
                        *atom_i,
                        Carbon,
                        BondType::Single,
                        Some("c".to_owned()), // todo
                        Some(1.4),            // todo
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
            }

            if ui
                .button("Add, sel atom")
                .on_hover_text(
                    "Add a Carbon atom, and select it. Useful for quickly adding chains.",
                )
                .clicked()
            {
                for atom_i in &selected_idxs {
                    let new_i = add_atom(
                        &mut state.mol_editor.mol.common,
                        &mut scene.entities,
                        *atom_i,
                        Carbon,
                        BondType::Single,
                        Some("ca".to_owned()), // todo
                        Some(1.4),             // todo
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

            for el in [
                Carbon, Hydrogen, Oxygen, Nitrogen, Sulfur, Phosphorus, Chlorine,
            ] {
                let sel = state.ui.selection.clone(); // todo :/
                change_el_button(
                    &sel,
                    el,
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
        });

        if selected_idxs.len() == 2 {
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

    // todo: Eventually add moving multiple atoms from multi-sel.

    let mut color_move = COLOR_INACTIVE;
    let mut color_rotate = COLOR_INACTIVE;

    // todo
    match state.volatile.mol_manip.mode {
        ManipMode::Move(_) => {
            color_move = COLOR_ACTIVE;
        }
        ManipMode::Rotate(_) => {
            color_rotate = COLOR_ACTIVE;
        }
        ManipMode::None => (),
    }

    section_box().show(ui, |ui| {
        if &selected_idxs.len() == &1 {
            if ui
                .button(RichText::new("↔ Move atom").color(color_move))
                .on_hover_text("(Hotkey: M. M or Esc to stop) Move the selected atom")
                .clicked()
            {
                // dummy API interaction.
                let mut redraw_flags = RedrawFlags::default();
                redraw_flags.ligand = *redraw;

                mol_manip::set_manip(
                    state,
                    scene,
                    &mut redraw_flags,
                    &mut rebuild_md,
                    // Atom i is used instead of the primary mode's mol i, since we're moving a single atom.
                    ManipMode::Move((MolType::Ligand, selected_idxs[0])),
                    engine_updates,
                );
                *redraw = redraw_flags.ligand;
            }

            if bond_mode {
                if ui.button(RichText::new("⟳ Rot around bond").color(color_rotate))
                    .on_hover_text("(Hotkey: R. R or Esc to stop) Rotate the molecule around this bond")
                    .clicked() {

                    // Note: We allow rotating around double-bonds for the purpose of building molecules, even though they're
                    // considered to be not-rotatable in other contexts.
                    // Don't rotate around bonds that are part of a cycle (rings).
                    // todo: Cache this in `MoleculeSmall`?
                    let bond = &state.mol_editor.mol.common.bonds[selected_idxs[0]];
                    if !bond.in_a_cycle(&state.mol_editor.mol.common.adjacency_list) {

                        // dummy API interaction.
                        let mut redraw_flags = RedrawFlags::default();
                        redraw_flags.ligand = *redraw;
                        mol_manip::set_manip(
                            state,
                            scene,
                            &mut redraw_flags,
                            &mut rebuild_md,
                            // Atom i is used instead of the primary mode's mol i, since we're moving a single atom.
                            ManipMode::Rotate((MolType::Ligand, selected_idxs[0])),
                            engine_updates,
                        );
                        *redraw = redraw_flags.ligand;
                    }
                }
            }


            // if &selected_idxs.len() == &1 {
            let mol = &mut state.mol_editor.mol.common;

            if let Selection::AtomsLig((_, atoms_i)) = &state.ui.selection
                && atoms_i.len() == 2
            {
                let bond_exists = mol.adjacency_list[atoms_i[0]].contains(&atoms_i[1]);

                if !bond_exists {
                    // todo: Hotkey for this and other functionality.
                    if ui
                        .button(RichText::new("Create Bond").color(COLOR_ACTION))
                        .on_hover_text(
                            "(Hotkey: todo) Create a covalently bond between the two selected atoms",
                        )
                        .clicked()
                    {
                        let atom_0 = &mol.atoms[atoms_i[0]];
                        let atom_1 = &mol.atoms[atoms_i[1]];

                        mol.bonds.push(Bond {
                            bond_type: BondType::Single, // todo: Allow other types
                            atom_0_sn: atom_0.serial_number,
                            atom_1_sn: atom_1.serial_number,
                            atom_0: atoms_i[0],
                            atom_1: atoms_i[1],
                            is_backbone: false,
                        });
                        mol.build_adjacency_list();

                        *redraw = true;
                        rebuild_md = true;
                    }
                }
            }

            if let Selection::AtomLig((_, i)) = state.ui.selection {
                if ui
                    .button(RichText::new("Del atom").color(Color32::LIGHT_RED))
                    .on_hover_text("(Hotkey: Delete) Delete the selected atom")
                    .clicked()
                {
                    state.mol_editor.remove_atom(i);
                    rebuild_md = true;
                    *redraw = true;
                }
            }
        }
    });

    if rebuild_md {
        sync_md(state);
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
            if ui
                .button(abbrev)
                .on_hover_text(format!("Add a {name} at the current selection"))
                .clicked()
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

        add_t(Template::Amide, "−NH₂", "amide functional group");

        add_t(Template::AromaticRing, "Ar", "benzene/aromatic ring");
        add_t(Template::Cyclohexane, "⬣", "cyclohexane ring");

        add_t(Template::PentaRing, "⬟", "5-atom ring");
    });
}
