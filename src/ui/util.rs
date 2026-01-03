use std::io;

use bio_apis::{amber_geostd, rcsb};
use egui::{Color32, Popup, PopupAnchor, Pos2, RectAlign, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};

use crate::{
    OperatingMode, State,
    cam_misc::reset_camera,
    download_mols::load_atom_coords_rcsb,
    drawing::draw_peptide,
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_alignment::run_alignment,
    mol_editor,
    mol_lig::MoleculeSmall,
    molecules::{MolGenericRef, MoleculeGeneric},
    render::{set_flashlight, set_static_light},
    ui::{COL_SPACING, COLOR_HIGHLIGHT, ROW_SPACING, set_window_title},
    util::{handle_err, reset_orbit_center},
};
use crate::ui::{COLOR_ACTIVE, COLOR_INACTIVE};

/// Run this each frame, after all UI elements that affect it are rendered.
pub fn update_file_dialogs(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) -> io::Result<()> {
    let ctx = ui.ctx();

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.save.update(ctx);

    if let Some(path) = &state.volatile.dialogs.load.take_picked() {
        if let Err(e) = match state.volatile.operating_mode {
            OperatingMode::Primary => state.open_file(path, Some(scene), engine_updates),
            OperatingMode::MolEditor => state.mol_editor.open_molecule(
                &state.dev,
                &state.ff_param_set,
                &state.mol_specific_params,
                &state.to_save.md_config,
                path,
                scene,
                engine_updates,
                &mut state.ui,
                state.volatile.mol_manip.mode,
            ),
            OperatingMode::ProteinEditor => unimplemented!(),
        } {
            handle_err(&mut state.ui, e.to_string());
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if let Some(path) = &state.volatile.dialogs.save.take_picked() {
        match state.volatile.operating_mode {
            OperatingMode::Primary => state.save(path)?,
            OperatingMode::MolEditor => mol_editor::save(state, path)?,
            OperatingMode::ProteinEditor => (),
        }
    }

    Ok(())
}

pub fn load_popups(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
    redraw_lig: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    if state.ui.popup.show_get_geostd {
        let popup_id = ui.make_persistent_id("no_ff_params_popup");

        Popup::new(
            popup_id,
            ui.ctx().clone(), // todo clone???
            PopupAnchor::Position(Pos2::new(60., 60.)),
            ui.layer_id(), // draw on top of the current layer
        )
        // .align(RectAlign::TOP)
        .align(RectAlign::BOTTOM_START)
        .open(true)
        .gap(4.0)
        .show(|ui| {
            let mut load_ff = false;
            let mut load_frcmod = false;
            // These vars avoid dbl borrow.
            // todo: Is this always the lig you want?
            if let Some(lig) = &state.active_mol() {
                if let MolGenericRef::Ligand(l) = lig {
                    load_ff = !l.ff_params_loaded;
                    load_frcmod = !l.frcmod_loaded;

                    // let Some(lig) = state.active_mol_mut() else {
                    //     return;
                    // };
                    let mut msg = String::from("Not ready for dynamics: ");

                    if !l.ff_params_loaded {
                        msg += "No FF params or partial charges are present on this ligand."
                    }

                    // if !lig.frcmod_loaded {
                    //     msg += "No FRCMOD parameters loaded for this ligand."
                    // }

                    ui.label(RichText::new(msg).color(Color32::LIGHT_RED));
                }

                ui.add_space(ROW_SPACING);

                // todo: What about cases where a SDF from pubchem or drugbank doesn't include teh name used by Amber?
                if ui.button("Check online").clicked() {
                    // let Some(lig) = state.ligand.as_mut() else {
                    //     return;
                    // };

                    match amber_geostd::find_mols(&lig.common().ident) {
                        Ok(data) => {
                            state.ui.popup.get_geostd_items = data;
                        }
                        Err(e) => handle_err(
                            &mut state.ui,
                            format!("Problem loading mol data online: {e:?}"),
                        ),
                    }
                }

                // This clone is annoying; db borrow.
                let items = state.ui.popup.get_geostd_items.clone();
                for mol_data in items {
                    if ui
                        .button(
                            RichText::new(format!("Load params for {}", mol_data.ident_pdbe))
                                .color(COLOR_HIGHLIGHT),
                        )
                        .clicked()
                    {
                        state.load_geostd_mol_data(
                            &mol_data.ident_pdbe,
                            load_ff,
                            load_frcmod,
                            engine_updates,
                            scene,
                        );

                        state.ui.popup.show_get_geostd = false;
                    }
                }
            }

            ui.add_space(ROW_SPACING);

            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                state.ui.popup.show_get_geostd = false;
            }
        });
    }

    if state.ui.popup.show_associated_structures {
        let mut associated_structs = Vec::new();
        if let Some(lig) = state.active_mol() {
            if let MolGenericRef::Ligand(l) = lig {
                // todo: I don't like this clone, but not sure how else to do it.
                associated_structs = l.associated_structures.clone();
            }
        }

        if state.active_mol().is_some() {
            let popup_id = ui.make_persistent_id("associated_structs_popup");
            Popup::new(
                popup_id,
                ui.ctx().clone(),
                PopupAnchor::Position(Pos2::new(300., 60.)),
                ui.layer_id(), // draw on top of the current layer
            )
            .align(RectAlign::TOP)
            .open(true)
            .gap(4.0)
            .show(|ui| {
                for s in &associated_structs {
                    ui.horizontal(|ui| {
                        if ui
                            .button(RichText::new(format!("{}", s.pdb_id)).color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            rcsb::open_overview(&s.pdb_id);
                        }
                        ui.add_space(COL_SPACING);

                        if ui
                            .button(
                                RichText::new(format!("Open this protein")).color(COLOR_HIGHLIGHT),
                            )
                            .clicked()
                        {
                            load_atom_coords_rcsb(
                                &s.pdb_id,
                                state,
                                scene,
                                engine_updates,
                                redraw_peptide,
                                reset_cam,
                            );
                        }
                    });

                    ui.label(RichText::new(format!("{}", s.description)));

                    ui.add_space(ROW_SPACING);
                }

                ui.add_space(ROW_SPACING);
            });

            ui.add_space(ROW_SPACING);
            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                state.ui.popup.show_associated_structures = false;
            }
        }
    }

    if state.ui.popup.alignment {
        let popup_id = ui.make_persistent_id("alignment_popup");

        Popup::new(
            popup_id,
            ui.ctx().clone(), // todo clone???
            PopupAnchor::Position(Pos2::new(200., 120.)),
            ui.layer_id(), // draw on top of the current layer
        )
        // .align(RectAlign::TOP)
        .align(RectAlign::BOTTOM_START)
        .open(true)
        .gap(4.0)
        .show(|ui| {
            ui.horizontal(|ui| {
                ui.label("Choose molecules to align:");

                ui.add_space(ROW_SPACING);
                if ui
                    .button(RichText::new("Close").color(Color32::LIGHT_RED))
                    .clicked()
                {
                    state.ui.popup.alignment = false;
                }
            });

            for (i, mol) in state.ligands.iter().enumerate() {
                let mta = &mut state.volatile.mols_to_align;

                let selected_pos = mta.iter().position(|&x| x == i);
                let color = if selected_pos.is_some() { COLOR_ACTIVE } else { COLOR_INACTIVE };

                if ui
                    .button(RichText::new(mol.common.name()).color(color))
                    .clicked()
                {
                    match selected_pos {
                        Some(pos) => {
                            // Toggle off.
                            mta.swap_remove(pos);
                        }
                        None => {
                            // Toggle on (cap at 2, avoid duplicates).
                            match mta.len() {
                                0 | 1 => mta.push(i),
                                _ => {
                                    // Keep the most recent selection and add this as the newest:
                                    // [old, recent] -> [recent, i]
                                    mta[0] = mta[1];
                                    mta[1] = i;
                                }
                            }
                        }
                    }
                }
            }

            if state.volatile.mols_to_align.len() == 2 && ui.button("Run alignment").clicked() {
                let mut redraw_lig = false;

                run_alignment(state, &mut redraw_lig);
            }
        });
    };
}


pub fn handle_redraw(
    state: &mut State,
    scene: &mut Scene,
    peptide: bool,
    lig: bool,
    na: bool,
    lipid: bool,
    reset_cam: bool,
    engine_updates: &mut EngineUpdates,
) {
    if peptide {
        draw_peptide(state, scene);
        // draw_all_ligs(state, scene); // todo: Hmm.

        if let Some(mol) = &state.peptide {
            set_window_title(&mol.common.ident, scene);
        }

        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Peptide as u32);

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            engine_updates.lighting = true;
        }
    }

    if lig {
        draw_all_ligs(state, scene);

        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Ligand as u32);

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            engine_updates.lighting = true;
        }
    }

    if na {
        draw_all_nucleic_acids(state, scene);
        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::NucleicAcid as u32);
    }

    if lipid {
        draw_all_lipids(state, scene);
        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Lipid as u32);
    }

    // Perform cleanup.
    if reset_cam {
        reset_camera(state, scene, engine_updates, FWD_VEC);
    }
}

/// Handles the case of opening a ligand remotely using the text input.
pub fn open_lig_from_input(
    state: &mut State,
    mol: MoleculeSmall,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.load_mol_to_state(
        MoleculeGeneric::Ligand(mol),
        Some(scene),
        engine_updates,
        None,
    );

    state.ui.db_input = String::new();
}

pub fn init_with_scene(state: &mut State, scene: &mut Scene, ctx: &egui::Context) {
    if state.peptide.is_some() {
        set_static_light(
            scene,
            state.peptide.as_ref().unwrap().center.into(),
            state.peptide.as_ref().unwrap().size,
        );
    } else if !state.ligands.is_empty() {
        let lig = &state.ligands[0];
        set_static_light(
            scene,
            lig.common.centroid().into(),
            3., // todo good enough?
        );
    }

    reset_orbit_center(state, scene);
}

/// An assistant to make a colored label.
#[macro_export]
macro_rules! label {
    ($ui:expr, $text:expr, $color:expr) => {
        $ui.label(egui::RichText::new($text).color($color))
    };
}
