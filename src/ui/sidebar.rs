use egui::{Color32, Context, RichText, Ui};
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::{
    cam::{move_cam_to_mol, move_mol_to_cam},
    drawing::{COLOR_SELECTED, EntityClass, draw_pocket},
    label,
    mol_characterization::MolCharacterization,
    mol_manip::{ManipMode, set_manip},
    molecules::{MolGenericRef, MolType, common::MoleculeCommon},
    state::{OperatingMode, State},
    therapeutic::pharmacophore::Pharmacophore,
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_HIGHLIGHT,
        COLOR_INACTIVE, ROW_SPACING, char_adme, pharmacophore,
    },
    util::{RedrawFlags, close_mol, handle_err, orbit_center},
};

/// Abstracts over all molecule types. (Currently not protein though)
/// A single row for the molecule.
fn mol_picker_one(
    active_mol: &mut Option<(MolType, usize)>,
    orbit_center: &mut Option<(MolType, usize)>,
    i_mol: usize,
    mol: &mut MoleculeCommon,
    mol_char: &Option<MolCharacterization>,
    pharmacophore: Option<&Pharmacophore>,
    mol_type: MolType,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    recenter_orbit: &mut bool,
    close: &mut Option<(MolType, usize)>,
    cam_snapshot: &mut Option<usize>,
    pep_center: Vec3,
) {
    let help_text = "Make this molecule the active / selected one. Middle click to close it.";

    let active = match active_mol {
        Some((mol_type_active, i)) => *mol_type_active == mol_type && *i == i_mol,
        _ => false,
    };

    let color = if active {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    };

    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(COL_SPACING / 2.);
            if ui
                .button(RichText::new("❌").color(Color32::LIGHT_RED))
                .on_hover_text("(Hotkey: Delete) Close this molecule.")
                .clicked()
            {
                *close = Some((mol_type, i_mol));
            }

            let color_md = if mol.selected_for_md {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if mol_type != MolType::Pocket {
                if ui
                    .button(RichText::new("MD").color(color_md))
                    .on_hover_text(
                        "Select or deselect this molecule for molecular dynamics simulation.",
                    )
                    .clicked()
                {
                    mol.selected_for_md = !mol.selected_for_md;
                }
            }

            let color_vis = if mol.visible {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if ui.button(RichText::new("👁").color(color_vis)).clicked() {
                mol.visible = !mol.visible;

                *redraw = true; // todo Overkill; only need to redraw (or even just clear) one.
                // todo: Generalize.
                engine_updates.entities = EntityUpdate::All;
                // engine_updates.entities.push_class(mol_type.entity_class() as u32);
            }

            if ui
                .button(RichText::new("Cam"))
                .on_hover_text("Move camera near active molecule, looking at it.")
                .clicked()
            {
                let beyond = if mol_type == MolType::Peptide {
                    Vec3::new_zero()
                } else {
                    pep_center
                };
                // Setting mol center to 0 if no mol.
                move_cam_to_mol(mol, cam_snapshot, scene, beyond, engine_updates)
            }

            let row_h = ui.spacing().interact_size.y;

            let sel_btn = ui
                .add_sized(
                    egui::vec2(ui.available_width(), row_h),
                    egui::Button::new(RichText::new(mol.name()).color(color)),
                )
                .on_hover_text(help_text);

            if sel_btn.clicked() {
                if active && active_mol.is_some() {
                    *active_mol = None;
                } else {
                    *active_mol = Some((mol_type, i_mol));
                    *orbit_center = *active_mol;

                    *recenter_orbit = true;
                }

                *redraw = true; // To reflect the change in thickness, color etc.
            }

            if sel_btn.middle_clicked() {
                *close = Some((mol_type, i_mol));
            }
        });
    });

    if let Some(char) = mol_char {
        let color_details = if active {
            Color32::WHITE
        } else {
            Color32::GRAY
        };

        label!(ui, char.to_string().trim(), color_details);
    }

    if let Some(pm) = pharmacophore
        && !pm.features.is_empty()
    {
        pharmacophore::pharmacophore_summary(pm, ui);
    }
    ui.separator();
}

/// Select, close, hide etc molecules from ones opened.
fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    engine_updates: &mut EngineUpdates,
) {
    let mut recenter_orbit = false;
    let mut close = None; // Avoids borrow error.

    // Avoids a double borrow.
    let pep_center = match &state.peptide {
        Some(mol) => mol.center,
        None => Vec3::new_zero(),
    };

    if let Some(mol) = &mut state.peptide {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            0,
            &mut mol.common,
            &None,
            None,
            MolType::Peptide,
            scene,
            ui,
            engine_updates,
            &mut redraw.peptide,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.ligands.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &mol.characterization,
            Some(&mol.pharmacophore),
            MolType::Ligand,
            scene,
            ui,
            engine_updates,
            &mut redraw.ligand,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.lipids.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &None,
            None,
            MolType::Lipid,
            scene,
            ui,
            engine_updates,
            &mut redraw.lipid,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.nucleic_acids.iter_mut().enumerate() {
        // todo: Characterization, e.g. by dna seq?
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &None,
            None,
            MolType::NucleicAcid,
            scene,
            ui,
            engine_updates,
            &mut redraw.na,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.pockets.iter_mut().enumerate() {
        // todo: Characterization, e.g. by dna seq?
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &None,
            None,
            MolType::Pocket,
            scene,
            ui,
            engine_updates,
            &mut redraw.pocket,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    // todo: AAs here too?

    if let Some((mol_type, i_mol)) = close {
        close_mol(mol_type, i_mol, state, scene, engine_updates);
    }

    if recenter_orbit {
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

fn open_tools(state: &mut State, ui: &mut Ui) {
    let color_open_tools = if state.peptide.is_none() && state.ligands.is_empty() {
        COLOR_ACTION
    } else {
        COLOR_INACTIVE
    };

    if ui
        .button(RichText::new("Open").color(color_open_tools))
        .on_hover_text("Open a molecule, electron density, or other file from disk.")
        .clicked()
    {
        state.volatile.dialogs.load.pick_file();
    }

    if ui
        .button(RichText::new("Recent").color(color_open_tools))
        .on_hover_text("Select a recently-opened file to open")
        .clicked()
    {
        state.ui.popup.recent_files = !state.ui.popup.recent_files;
    }
}

fn manip_toolbar(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    ui.horizontal(|ui| {
        let Some((active_mol_type, active_mol_i)) = state.volatile.active_mol else {
            return;
        };

        {
            let mut color_move = COLOR_INACTIVE;
            let mut color_rotate = COLOR_INACTIVE;

            match state.volatile.mol_manip.mode {
                ManipMode::Move((mol_type, mol_i)) => {
                    if mol_type == active_mol_type && mol_i == active_mol_i {
                        color_move = COLOR_ACTIVE;
                    }
                }
                ManipMode::Rotate((mol_type, mol_i)) => {
                    if mol_type == active_mol_type && mol_i == active_mol_i {
                        color_rotate = COLOR_ACTIVE;
                    }
                }
                ManipMode::None => (),
            }

            // ✥ doesn't work in EGUI.
            if ui.button(RichText::new("↔").color(color_move))
                .on_hover_text("(Hotkey: M. M or Esc to stop)) Move the active molecule by clicking and dragging with /
                the mouse. Scroll to move it forward and back.")
                .clicked() {

                set_manip(&mut state.volatile, &mut state.pockets, &mut state.to_save.save_flag, scene, redraw,&mut false,
                          ManipMode::Move((active_mol_type, active_mol_i)), &state.ui.selection, engine_updates);
            }

            if ui.button(RichText::new("⟳").color(color_rotate))
                .on_hover_text("(Hotkey: R. R or Esc to stop) Rotate the active molecule by clicking and dragging with the mouse. Scroll to roll.")
                .clicked() {

                set_manip(&mut state.volatile, &mut state.pockets, &mut state.to_save.save_flag,
                          scene,redraw,&mut false,
                          ManipMode::Rotate((active_mol_type, active_mol_i)), &state.ui.selection ,engine_updates,);
            }
        }

        if let Some(mol) = &mut state.active_mol_mut() {
            if ui
                .button(RichText::new("Move to cam").color(COLOR_HIGHLIGHT))
                .on_hover_text("Move the molecule to be a short distance in front of the camera.")
                .clicked()
            {
                move_mol_to_cam(mol.common_mut(), &scene.camera);

                redraw.set(active_mol_type);
            }

            if ui
                .button(RichText::new("Reset pos").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Move the molecule to its absolute coordinates, e.g. as defined in /
                        its source mmCIF, Mol2 or SDF file.",
                )
                .clicked()
            {
                mol.common_mut().reset_posits();

                // todo: Use the inplace move.
                redraw.set(active_mol_type);
            }

            if ui.button("Details")
                .on_hover_text("Toggle the details display of the active molecule.")
                .clicked() {
                state.ui.ui_vis.mol_char = !state.ui.ui_vis.mol_char;
            }
        }
    });
}

pub(in crate::ui) fn sidebar(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    engine_updates: &mut EngineUpdates,
    ctx: &Context,
) {
    let out = egui::SidePanel::left("sidebar")
        .resizable(true) // let user drag the width
        .default_width(140.0)
        .width_range(60.0..=800.0)
        .show(ctx, |ui| {
            ui.label("Molecules opened");

            ui.horizontal(|ui| {
                let color_open_tools = if state.peptide.is_none() && state.ligands.is_empty() {
                    COLOR_ACTION
                } else {
                    COLOR_INACTIVE
                };

                if ui
                    .button(RichText::new("Open").color(color_open_tools))
                    .on_hover_text("Open a molecule, electron density, or other file from disk.")
                    .clicked()
                {
                    state.volatile.dialogs.load.pick_file();
                }

                if ui
                    .button(RichText::new("Recent").color(color_open_tools))
                    .on_hover_text("Select a recently-opened file to open")
                    .clicked()
                {
                    state.ui.popup.recent_files = !state.ui.popup.recent_files;
                    open_tools(state, ui);
                }

                let mut mol_to_save = None; // avoids dbl-borrow.
                if let Some(mol) = state.active_mol() {
                    let color = Color32::GRAY;

                    if ui
                        .button(RichText::new("Save").color(color))
                        .on_hover_text("Save the active molecule to a file.")
                        .clicked()
                    {
                        mol_to_save = Some(mol.common().clone());
                    }
                }
                if let Some(mol) = mol_to_save {
                    if mol.save(&mut state.volatile.dialogs.save).is_err() {
                        handle_err(&mut state.ui, "Problem saving this file".to_owned());
                    }
                }
            });

            ui.add_space(ROW_SPACING / 2.);

            manip_toolbar(state, scene, redraw, ui, engine_updates);

            ui.add_space(ROW_SPACING / 2.);
            ui.separator();
            ui.add_space(ROW_SPACING / 2.);

            // ScrollArea::vertical()
            //     .min_scrolled_height(600.0)
            //     .show(ui, |ui| {
            // todo: Function or macro to reduce this DRY.

            if state.volatile.operating_mode == OperatingMode::MolEditor {
                ui.label("Pockets");

                for (mol_i, pocket) in state.pockets.iter_mut().enumerate() {
                    let selected = state.mol_editor.pocket_i_in_state == Some(mol_i);

                    let color = if selected {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if ui
                        .button(RichText::new(&pocket.common.ident).color(color))
                        .on_hover_text(
                            "Display this pocket, and optionally use it as part of \
                        a pharmacophore, e.g. its excluded volume..",
                        )
                        .clicked()
                    {
                        scene
                            .entities
                            .retain(|e| e.class != EntityClass::Pocket as u32);

                        if selected {
                            state.mol_editor.pocket = None;
                            state.mol_editor.pocket_i_in_state = None;
                        } else {
                            // todo: This pocket adjustment code should probably be moved
                            // todo: H bonds`
                            pocket.common.center_local_posits_around_origin();
                            pocket.common.reset_posits();
                            pocket.rebuild_spheres();
                            pocket.regen_mesh_vol();
                            engine_updates.meshes = true;

                            draw_pocket(pocket, &[], &state.ui.visibility, &state.ui.selection);

                            state.mol_editor.pocket = Some(pocket.clone());
                            state.mol_editor.pocket_i_in_state = Some(mol_i);
                        }
                    }
                }
            } else {
                mol_picker(state, scene, ui, redraw, engine_updates);
            }

            // todo: Still list global pharmacophores
            if state.ui.ui_vis.pharmacophore_list
                && state.volatile.operating_mode == OperatingMode::MolEditor
            {
                // todo: Make this work eventually when out of hte mol editor.

                // if let Some(mol) = &state.active_mol() {
                // if let Some(mol) = &state.mol_editor.mol {
                let mol = &mut state.mol_editor.mol;
                let mut closed = false;
                // if let MolGenericRef::Small(mol) = mol {

                ui.add_space(ROW_SPACING);
                pharmacophore::pharmacophore_list(
                    &mut mol.pharmacophore,
                    &mut state.ui.popup,
                    &mut closed,
                    ui,
                );

                // }
                if closed {
                    // Broken out to avoid double borrow.
                    state.ui.ui_vis.pharmacophore_list = false;
                }
                // }
            }

            // todo: UI flag to show or hide this.
            if state.ui.ui_vis.mol_char && state.volatile.operating_mode != OperatingMode::MolEditor
            {
                let mut toggled = false; // Avoid double borrow.
                if let Some(m) = &state.active_mol() {
                    if let MolGenericRef::Small(mol) = m {
                        if ui
                            .button(RichText::new("Close details").color(Color32::LIGHT_RED))
                            .clicked()
                        {
                            toggled = true;
                        }
                        char_adme::mol_char_disp(mol, ui);
                    }
                }
                if toggled {
                    state.ui.ui_vis.mol_char = !state.ui.ui_vis.mol_char;
                }
            }
        });

    engine_updates.ui_reserved_px.0 = out.response.rect.width();
}
