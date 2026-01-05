use bio_apis::{amber_geostd, rcsb};
use egui::{Color32, Popup, PopupAnchor, Pos2, RectAlign, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::{
    State,
    download_mols::load_atom_coords_rcsb,
    drawing_wrappers::draw_all_ligs,
    mol_alignment::run_alignment,
    molecules::MolGenericRef,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        cam::move_cam_to_mol,
    },
    util::handle_err,
};

pub(in crate::ui) fn load_popups(
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
            PopupAnchor::Position(Pos2::new(300., 300.)),
            ui.layer_id(), // draw on top of the current layer
        )
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
                let color = if selected_pos.is_some() {
                    COLOR_ACTIVE
                } else {
                    COLOR_INACTIVE
                };

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

            ui.add_space(ROW_SPACING);
            ui.separator();
            if state.volatile.mols_to_align.len() == 2 && ui.button("Run alignment").clicked() {
                run_alignment(state, &mut false);

                // Set visualization settings to view the result.
                state.ligands[state.volatile.mols_to_align[0]]
                    .common
                    .visible = true;
                state.ligands[state.volatile.mols_to_align[1]]
                    .common
                    .visible = true;
                state.ui.color_by_mol = true;

                for (i, mol) in state.ligands.iter_mut().enumerate() {
                    // Required to make our snapshots work, in the current implementation.
                    mol.common.selected_for_md = i == state.volatile.mols_to_align[1];
                }

                *redraw_lig = true;
                // draw_all_ligs(state, scene);
                // engine_updates.entities = EntityUpdate::All; // todo: Just ligs.

                move_cam_to_mol(
                    &state.ligands[state.volatile.mols_to_align[0]].common,
                    &mut state.ui.cam_snapshot,
                    scene,
                    Vec3::new_zero(),
                    engine_updates,
                )
            }
        });
    };
}
