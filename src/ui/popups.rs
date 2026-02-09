use std::path::Path;

use bio_apis::{amber_geostd, rcsb};
use bio_files::ResidueType;
use chrono::Utc;
use egui::{
    Align, Color32, ComboBox, Layout, Popup, PopupAnchor, Pos2, RectAlign, RichText, ScrollArea,
    TextEdit, Ui,
};
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;
use na_seq::AaIdent;

use crate::drawing::EntityClass;
use crate::{
    cam::{move_cam_to_active_mol, move_cam_to_mol},
    drawing::wrappers::draw_all_pockets,
    file_io::{download_mols, download_mols::load_atom_coords_rcsb},
    inputs::{MOVEMENT_SENS, ROTATE_SENS, SENS_MOL_MOVE_SCROLL},
    label,
    mol_alignment::run_alignment,
    mol_screening,
    mol_screening::screen_by_alignment,
    molecules::{
        MolGenericRef, MolType,
        common::MoleculeCommon,
        pocket::{POCKET_DIST_THRESH_DEFAULT, Pocket},
    },
    prefs::OpenType,
    render::MESH_POCKET,
    selection::{Selection, ViewSelLevel},
    state::{MsaaSetting, PopupState, State},
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        mol_data::metadata, pharmacophore, rama_plot, recent_files, recent_files::NUM_TO_SHOW,
    },
    util::{RedrawFlags, handle_err, handle_success, make_lig_from_res, orbit_center},
};

/// Based on popup state, shows popups. This is the entry point for all popups.
pub(in crate::ui) fn load_popups(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    if state.ui.popup.show_get_geostd {
        popup("geostd", ui).show(|ui| {
            get_geostd(state, scene, engine_updates, ui);
        });
    }

    if state.ui.popup.show_associated_structures {
        popup("assoc_structs", ui).show(|ui| {
            associated_structures(
                state,
                scene,
                engine_updates,
                &mut redraw.peptide,
                reset_cam,
                ui,
            );
        });
    }

    if state.ui.popup.alignment {
        popup("alignment", ui).show(|ui| {
            alignment(state, scene, &mut redraw.ligand, engine_updates, ui);
        });
    };

    if state.ui.popup.alignment_screening {
        popup("align_screening", ui).show(|ui| {
            alignment_screening(state, ui);
        });
    }

    if state.ui.popup.show_settings {
        popup("settings", ui).show(|ui| {
            settings(state, scene, ui);
        });
    }

    if state.ui.popup.residue_selector {
        // todo: Show hide based on AaCategory? i.e. residue.amino_acid.category(). Hydrophilic, acidic etc.
        popup("res_sel", ui).show(|ui| {
            residue_selector(state, scene, ui, &mut redraw.peptide);
        });
    }

    if state.ui.popup.recent_files {
        popup("recent_files", ui).show(|ui| {
            recent_files_popup(state, scene, ui, engine_updates);
        });
    }

    if state.ui.popup.rama_plot {
        if let Some(mol) = &state.peptide {
            popup("rama", ui).show(|ui| {
                rama_plot::plot_rama(
                    &mol.residues,
                    &mol.common.ident,
                    ui,
                    &mut state.ui.popup.rama_plot,
                );
            });
        }
    }

    if state.ui.popup.pharmacophore_boolean {
        popup("pharmacophore_boolean", ui).show(|ui| {
            pharmacophore::pharmacophore_boolean_window(state, ui);
        });
    }

    if let Some((mol_type, i)) = state.ui.popup.metadata {
        popup("metadata", ui).show(|ui| {
            metadata(mol_type, i, state, ui);
        });
    }

    if state.ui.popup.lig_pocket_creation {
        popup("lig_pocket_creation", ui).show(|ui| {
            lig_pocket_from_het_res(state, scene, ui, engine_updates);
        });
    }
}

fn get_geostd(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    let mut load_ff = false;
    let mut load_frcmod = false;
    // These vars avoid dbl borrow.
    // todo: Is this always the lig you want?
    if let Some(lig) = &state.active_mol() {
        if let MolGenericRef::Small(l) = lig {
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
}

fn associated_structures(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw_peptide: &mut bool,
    reset_cam: &mut bool,
    ui: &mut Ui,
) {
    let mut associated_structs = Vec::new();
    if let Some(lig) = state.active_mol() {
        if let MolGenericRef::Small(l) = lig {
            // todo: I don't like this clone, but not sure how else to do it.
            associated_structs = l.associated_structures.clone();
        }
    }

    if state.active_mol().is_some() {
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
                    .button(RichText::new(format!("Open this protein")).color(COLOR_HIGHLIGHT))
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

        ui.add_space(ROW_SPACING);
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.show_associated_structures = false;
        }
    }
}

fn alignment_screening(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("Choose molecules to align:");

        ui.add_space(COL_SPACING);

        if ui
            .button("Select folder")
            .on_hover_text(
                "Perform a fast small molecule alignment screening from all \
                files in a selected folder",
            )
            .clicked()
        {
            state.volatile.dialogs.screening.pick_directory();
        }

        if let Some(path) = &state.volatile.alignment.screening_path {
            ui.add_space(COL_SPACING);
            if ui
                .button(RichText::new("Run screening").color(COLOR_ACTION))
                .clicked()
            {
                // todo: Don't block; launch a new thread.

                // let path = Path::new("C:/Users/the_a/Desktop/bio_misc/amber_geostd/c");
                if let Ok(mols) = mol_screening::load_mols(path) {
                    state.volatile.alignment.mols_passed_screening = Vec::new();

                    let template = mols[0].clone();
                    // todo: UI controls for these, along with path.
                    let score_thresh = 60.;
                    let size_diff_thresh = 0.4;

                    let result =
                        screen_by_alignment(&template, &mols, score_thresh, size_diff_thresh);

                    // todo: COnfigure this etc
                    let max_stored = 10_000;
                    let mut stored = 0;
                    for (i, _score) in result {
                        state
                            .volatile
                            .alignment
                            .mols_passed_screening
                            .push(mols[i].clone());

                        stored += 1;
                        if stored > max_stored {
                            break;
                        }
                    }

                    // result.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                } else {
                    handle_err(
                        &mut state.ui,
                        format!("Failed to load molecules from {path:?}"),
                    );
                }
            }
        }

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.alignment = false;
        }
    });

    if !state.volatile.alignment.mols_passed_screening.is_empty() {
        ui.label("Results:");
        // todo: Scroll area
        let max_to_show = 20;
        let mut shown = 0;

        for mol in &state.volatile.alignment.mols_passed_screening {
            // todo: Load this to state when clicked etc.
            label!(ui, format!("- {}", mol.common.name()), Color32::WHITE);

            shown += 1;
            if shown > max_to_show {
                break;
            }
        }

        //
        // let res = &state.volatile.alignment.results_screening[0];
        // ui.label("Results:");
        //
        // label!(ui, format!("Score: {:.2} E template: {:.2} E query: {:.2} Vol: {:.2}",
        // res.score, res.avg_potential_e_template, res.avg_potential_e_query, res.volume),
        //     Color32::WHITE);
    }
}

fn recent_files_popup(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    let mut open = None;
    let now = Utc::now();

    // todo: Take only recent ones
    for file in state.to_save.open_history.iter().rev().take(NUM_TO_SHOW) {
        ui.horizontal(|ui| {
            let filename = Path::new(&file.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap();

            // todo: Make the whole row clickable?
            let (r, g, b) = match file.type_ {
                OpenType::Peptide => MolType::Peptide.color(),
                OpenType::Ligand => MolType::Ligand.color(),
                OpenType::NucleicAcid => MolType::NucleicAcid.color(),
                OpenType::Lipid => MolType::Lipid.color(),
                _ => (255, 255, 255),
            };
            let color = Color32::from_rgb(r, g, b);
            if ui.button(RichText::new(filename).color(color)).clicked() {
                open = Some(file.path.clone());
            }

            ui.add_space(COL_SPACING);
            ui.label(RichText::new(file.type_.to_string()));

            let elapsed_minutes = (now - file.timestamp).num_minutes();

            let (descrip, color) = recent_files::recentness_descrip(elapsed_minutes);
            ui.label(RichText::new(descrip).color(color));
        });
    }

    if let Some(path) = open {
        if state.open_file(&path, Some(scene), engine_updates).is_err() {
            handle_err(&mut state.ui, format!("Problem opening file {:?}", path));
        }
        state.ui.popup.recent_files = false;
    }

    ui.add_space(ROW_SPACING);
    if ui
        .button(RichText::new("Close").color(Color32::LIGHT_RED))
        .clicked()
    {
        state.ui.popup.recent_files = false;
    }
}

pub(in crate::ui) fn metadata_popup(
    popup_state: &mut PopupState,
    mol: &MoleculeCommon,
    ui: &mut Ui,
) {
    ui.with_layout(Layout::top_down(Align::RIGHT), |ui| {
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            popup_state.metadata = None;
        }
    });

    ui.heading(RichText::new(format!("Metadata for {}", mol.ident)).color(Color32::WHITE));
    ui.add_space(ROW_SPACING);

    ScrollArea::vertical()
        .min_scrolled_height(800.0)
        .show(ui, |ui| {
            for (k, v) in mol.metadata.iter() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{k}: ")));
                    label!(ui, v.to_string(), Color32::WHITE);
                });
            }
        });
}

fn settings(state: &mut State, scene: &mut Scene, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.heading("Settings");
        ui.add_space(COL_SPACING);
        // todo: Make this consistent with your other controls.
        ui.label("MSAA (Restart the program to take effect):");

        let msaa_prev = state.to_save.msaa;
        ComboBox::from_id_salt(10)
            .width(40.)
            .selected_text(state.to_save.msaa.to_str())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut state.to_save.msaa,
                    MsaaSetting::None,
                    MsaaSetting::None.to_str(),
                );
                ui.selectable_value(
                    &mut state.to_save.msaa,
                    MsaaSetting::Four,
                    MsaaSetting::Four.to_str(),
                );
            });

        if state.to_save.msaa != msaa_prev {
            state.update_save_prefs(false);
        }

        ui.add_space(COL_SPACING);
        ui.label("Cam move speed:");
        if ui
            .add(TextEdit::singleline(&mut state.ui.movement_speed_input).desired_width(32.))
            .changed()
        {
            if let Ok(v) = &mut state.ui.movement_speed_input.parse::<u8>() {
                state.to_save.movement_speed = *v;
                scene.input_settings.move_sens = *v as f32;

                state.update_save_prefs(false);
            } else {
                // reset
                state.ui.movement_speed_input = state.to_save.movement_speed.to_string();
            }
        }

        ui.add_space(COL_SPACING / 2.);
        ui.label("Cam rot sensitivity:");
        if ui
            .add(TextEdit::singleline(&mut state.ui.rotation_sens_input).desired_width(32.))
            .changed()
        {
            if let Ok(v) = &mut state.ui.rotation_sens_input.parse::<u8>() {
                state.to_save.rotation_sens = *v;
                scene.input_settings.rotate_sens = *v as f32 / 100.;

                state.update_save_prefs(false);
            } else {
                // reset
                state.ui.rotation_sens_input = state.to_save.rotation_sens.to_string();
            }
        }

        ui.add_space(COL_SPACING);
        ui.label("Mol scroll move speed:").on_hover_text(
            "When using the scroll wheel to move molecules, this controls how fast they move.",
        );
        if ui
            .add(TextEdit::singleline(&mut state.ui.mol_move_sens_input).desired_width(32.))
            .changed()
        {
            if let Ok(v) = &mut state.ui.mol_move_sens_input.parse::<u8>() {
                state.to_save.mol_move_sens = *v;
                state.update_save_prefs(false);
            } else {
                // reset
                state.ui.mol_move_sens_input = state.to_save.mol_move_sens.to_string();
            }
        }

        ui.add_space(COL_SPACING / 2.);
        if ui.button("Reset sensitivities").clicked() {
            state.to_save.movement_speed = MOVEMENT_SENS as u8;
            state.ui.movement_speed_input = state.to_save.movement_speed.to_string();
            scene.input_settings.move_sens = MOVEMENT_SENS;

            state.to_save.rotation_sens = (ROTATE_SENS * 100.) as u8;
            state.ui.rotation_sens_input = state.to_save.rotation_sens.to_string();
            scene.input_settings.rotate_sens = ROTATE_SENS;

            state.to_save.mol_move_sens = (SENS_MOL_MOVE_SCROLL * 1_000.) as u8;
            state.ui.mol_move_sens_input = state.to_save.mol_move_sens.to_string();

            state.update_save_prefs(false);
        }
    });

    ui.add_space(ROW_SPACING);

    if ui
        .button(RichText::new("Close").color(Color32::LIGHT_RED))
        .clicked()
    {
        state.ui.popup.show_settings = false;
    }
}

fn alignment(
    state: &mut State,
    scene: &mut Scene,
    redraw_lig: &mut bool,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.label("Choose molecules to align:");

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.alignment = false;
        }
    });

    if !state.volatile.alignment.results.is_empty() {
        let res = &state.volatile.alignment.results[0];
        ui.label("Results:");

        label!(
            ui,
            format!(
                "Score: {:.2} E template: {:.2} E query: {:.2} Vol: {:.2}",
                res.score, res.avg_potential_e_template, res.avg_potential_e_query, res.volume
            ),
            Color32::WHITE
        );
    }

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        let help_text = "If selected, the template atom will be allowed to take on different \
                conformations based on its rotatable bonds. If not, only the query molecule will \
                be flexible.";

        ui.label("Treat the template molecule as flexible.")
            .on_hover_text(help_text);
        ui.checkbox(&mut state.volatile.alignment.flexible_template, "")
            .on_hover_text(help_text);
    });

    ui.add_space(ROW_SPACING);

    for (i, mol) in state.ligands.iter().enumerate() {
        let mta = &mut state.volatile.alignment.mols_to_align;

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
    if state.volatile.alignment.mols_to_align.len() == 2 && ui.button("Run alignment").clicked() {
        run_alignment(state, &mut false);

        // Set visualization settings to view the result.
        state.ligands[state.volatile.alignment.mols_to_align[0]]
            .common
            .visible = true;
        state.ligands[state.volatile.alignment.mols_to_align[1]]
            .common
            .visible = true;
        state.ui.color_by_mol = true;

        for (i, mol) in state.ligands.iter_mut().enumerate() {
            // Required to make our snapshots work, in the current implementation.
            mol.common.selected_for_md = i == state.volatile.alignment.mols_to_align[1];
        }

        *redraw_lig = true;
        // draw_all_ligs(state, scene);
        // engine_updates.entities = EntityUpdate::All; // todo: Just ligs.

        move_cam_to_mol(
            &state.ligands[state.volatile.alignment.mols_to_align[0]].common,
            &mut state.ui.cam_snapshot,
            scene,
            Vec3::new_zero(),
            engine_updates,
        )
    }
}

fn residue_selector(state: &mut State, scene: &mut Scene, ui: &mut Ui, redraw: &mut bool) {
    ui.with_layout(Layout::top_down(Align::RIGHT), |ui| {
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.residue_selector = false;
            state.ui.chain_to_pick_res = None;
        }
    });
    ui.add_space(ROW_SPACING);
    // This is a bit fuzzy, as the size varies by residue name (Not always 1 for non-AAs), and index digits.

    let mut update_arc_center = false;

    if let Some(mol) = &state.peptide {
        if let Some(chain_i) = state.ui.chain_to_pick_res {
            if chain_i >= mol.chains.len() {
                return;
            }
            let chain = &mol.chains[chain_i];

            ui.add_space(ROW_SPACING);

            // todo: Wrap not working in popup?
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                for (i, res) in mol.residues.iter().enumerate() {
                    if i > 800 {
                        break; // todo: Temp workaround to display blocking
                    }
                    // For now, peptide residues only.
                    if let ResidueType::Water = res.res_type {
                        continue;
                    }

                    // Only let the user select residue from the selected chain. This should keep
                    // it more organized, and keep UI space used down.
                    if !chain.residues.contains(&i) {
                        continue;
                    }

                    let name = match &res.res_type {
                        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::OneLetter),
                        ResidueType::Water => "Water".to_owned(),
                        ResidueType::Other(name) => name.clone(),
                    };

                    let mut color = Color32::GRAY;
                    if let Selection::Residue(sel_i) = state.ui.selection {
                        if sel_i == i {
                            color = COLOR_ACTIVE;
                        }
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} {name}", res.serial_number))
                                .size(10.)
                                .color(color),
                        )
                        .clicked()
                    {
                        state.ui.view_sel_level = ViewSelLevel::Residue;
                        state.ui.selection = Selection::Residue(i);

                        update_arc_center = true; // Avoids borrow error.

                        *redraw = true;
                    }
                }
            });
        }
    }

    if update_arc_center {
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

fn popup<'a>(name: &'a str, ui: &'a mut Ui) -> Popup<'a> {
    let popup_id = ui.make_persistent_id(name);

    Popup::new(
        popup_id,
        ui.ctx().clone(),
        PopupAnchor::Position(Pos2::new(300., 300.)),
        ui.layer_id(), // draw on top of the current layer
    )
    .align(RectAlign::BOTTOM_START)
    .open(true)
    .gap(4.0)
}

/// This handles creating ligands and pockets from hetero residues in protein data. This, at least
/// from RCSB protein files, is often associated with ligands binding to the protein.
///
/// Note: We're currently relying on remotely downloading geostd molecules for ligand creation.
/// todo: Update this to use something more robust, like a PubChem or RCSB API.
// todo: Move A/R
fn lig_pocket_from_het_res(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    let Some(mol) = &state.peptide else {
        return;
    };

    // Provide convenience functionality for loading ligands based on hetero residues
    // in the protein.
    // let mut load_data = None; // Avoids dbl-borrow.
    // let mut count_geostd_candidate = 0;
    //
    // for res in &mol.het_residues {
    //     if let ResidueType::Other(name) = &res.res_type {
    //         if name.len() == 3 {
    //             count_geostd_candidate += 1;
    //         }
    //     }
    // }

    ui.horizontal(|ui| {
        label!(
            ui,
            "Make ligands and pockets from het residues",
            Color32::WHITE
        );
        ui.add_space(COL_SPACING / 2.);
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.lig_pocket_creation = false;
        }
    });

    // .on_hover_text(
    //     "Attempt to load a ligand molecule and force field \
    //                     params from a hetero residue included in the protein file.",
    // );
    ui.separator();
    ui.add_space(ROW_SPACING);

    // This mechanism prevents buttons from duplicate hetero residues, e.g.
    // if more than one copy of a ligand is present in the data.
    let mut residue_names = Vec::new();
    // let mut res_to_load = None;

    // Avoids a double borrow.
    let mut create_lig_from_res = None;
    let mut pocket_to_add = None;

    for res in &mol.het_residues {
        let name = match &res.res_type {
            ResidueType::Other(name) => name,
            _ => "hetero residue",
        };

        // todo: Fragile?
        // if name.len() == 3 {
        if residue_names.contains(&name) {
            continue;
        }
        residue_names.push(name);

        ui.horizontal(|ui| {
            label!(ui, format!("{name}: "), Color32::WHITE);
            ui.add_space(COL_SPACING / 2.);

            if ui
                .button(RichText::new("Make lig").color(COLOR_ACTION))
                .on_hover_text("Create a ligand using molecules from this residue.")
                .clicked()
            {
                create_lig_from_res = Some(res.clone());

                // download_mols::load_geostd(name, &mut load_data, &mut state.ui);
                // res_to_load = Some(res.clone()); // Clone avoids borrow error.
            }

            if ui
                .button(RichText::new("Make pocket").color(COLOR_ACTION))
                .on_hover_text(
                    "Create a pocket around a hetero residue included in a protein file.",
                )
                .clicked()
            {
                let lig_ctr = {
                    let mut ctr = Vec3::new_zero();
                    for atom_i in &res.atoms {
                        // Using local coordinates; this should be independent of the user positioning the protein.
                        if *atom_i >= mol.common.atoms.len() {
                            eprintln!(
                                "Error: Atom index out of bounds: {} > {}",
                                atom_i,
                                mol.common.atoms.len()
                            );
                            continue;
                        }

                        ctr += mol.common.atoms[*atom_i].posit;
                    }
                    ctr / res.atoms.len() as f64
                };

                let ident = format!("Pocket: {name}");
                pocket_to_add = Some(Pocket::new(
                    mol,
                    lig_ctr,
                    POCKET_DIST_THRESH_DEFAULT,
                    &ident,
                ));
            }
        });
        ui.add_space(ROW_SPACING);
        // }
    }

    if let Some(pocket) = pocket_to_add {
        scene.meshes[MESH_POCKET] = pocket.surface_mesh.clone();

        draw_all_pockets(state, scene);
        state.pockets.push(pocket);

        engine_updates.meshes = true;
        // todo: DOesn't seem to be working.
        engine_updates.entities = EntityUpdate::Classes(vec![EntityClass::Pocket as u32]);
    }

    if let Some(res) = &create_lig_from_res {
        make_lig_from_res(state, res, scene, engine_updates);
    }
    //
    // // Avoids dbl-borrow
    // if let Some(data) = load_data {
    //     handle_success(
    //         &mut state.ui,
    //         format!("Loaded {} from Amber Geostd", data.ident_pdbe),
    //     );
    //
    //     // Crude check for success.
    //     // let lig_count_prev = state.ligands.len();
    //     state.load_geostd_mol_data(
    //         &data.ident_pdbe,
    //         true,
    //         data.frcmod_avail,
    //         engine_updates,
    //         scene,
    //     );
    //
    //     // Move camera to ligand; not ligand to camera, since we are generating a ligand
    //     // that may already be docked to the protein.
    //     // move_mol_to_cam(&mut state.ligands[i].common, &scene.camera);
    //     if let Some(mol) = &state.peptide {
    //         move_cam_to_active_mol(state, scene, mol.center, engine_updates);
    //     }
    // } else {
    //     if let Some(res) = res_to_load {
    //         // Use our normal "Lig from" logic.
    //         make_lig_from_res(state, &res, scene, engine_updates);
    //
    //         move_cam_to_active_mol(
    //             state,
    //             scene,
    //             state.ligands[0].common.centroid(),
    //             engine_updates,
    //         );
    //
    //         handle_success(
    //             &mut state.ui,
    //             "Unable to find FF params for this ligand; added without them".to_string(),
    //         );
    //     }
    // }
}
