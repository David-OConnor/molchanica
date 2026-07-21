use bio_files::{FrameSlice, md_params::ForceFieldParams};
use dynamics::{FfMolType, merge_params};
use egui::{Color32, RichText, TextEdit, Ui};
use graphics::{ControlScheme, EngineUpdates, FWD_VEC, Scene};
use lin_alg::f64::Vec3;
use na_seq::AaIdent;

use crate::{
    button,
    cam::{move_cam_to_mol, move_mol_to_cam, reset_camera, set_fog},
    label,
    md::{
        trajectory::{MAX_FRAMES_TO_ATTEMPT_LOADING, Trajectory, TrajectorySource, close_traj},
        viewer,
        viewer::ViewerMolSet,
    },
    mol_manip::{ManipMode, set_manip},
    molecules::{
        MolGenericRef, MolIdent, MolType, common::MoleculeCommon, nucleic_acid::NucleicAcidType,
    },
    properties::{
        crystal, logp, mol_characterization::MolCharacterization, sol_shrinking_box, water_sol,
        water_sol_mix,
    },
    screening::pharmacophore::{Pharmacophore, PharmacophoreState},
    sonification,
    state::{OperatingMode, PlayingAudio, PopupState, State},
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_HIGHLIGHT,
        COLOR_INACTIVE, ROW_SPACING, highlighted_box, num_field, panels::md_viewer,
        popup::pharmacophore,
    },
    util::{RedrawFlags, close_mol, handle_err, handle_success, orbit_center},
};

mod char_adme;
mod mol_editor_sidebar;

const SONIFICATION_INCLUDE_H: bool = true;

/// Width of the strip left in place of the sidebar when it's hidden; fits the show button.
const SIDEBAR_HIDDEN_WIDTH: f32 = 40.;

#[derive(Clone, Copy)]
enum AudioAction {
    Toggle(MolType, usize),
}

fn md_copies_field(copies: &mut usize, ui: &mut Ui) {
    let mut copies_str = copies.to_string();
    if ui
        .add_sized(
            [36., ui.spacing().interact_size.y],
            TextEdit::singleline(&mut copies_str),
        )
        .on_hover_text("Number of molecule copies to include in MD.")
        .changed()
        && let Ok(parsed) = copies_str.parse::<usize>()
    {
        *copies = parsed.max(1);
    }
}

/// Abstracts over all molecule types. (Currently vnot protein though)
/// A single row for the molecule.
fn mol_picker_one(
    active_mol: &mut Option<(MolType, usize)>,
    orbit_center: &mut Option<(MolType, usize)>,
    ph_state: &mut PharmacophoreState,
    i_mol: usize,
    mol: &mut MoleculeCommon,
    idents: Option<&Vec<MolIdent>>, // For small mols,
    mol_char: &Option<MolCharacterization>,
    pharmacophore: Option<&Pharmacophore>,
    mol_type: MolType,
    popup: &mut PopupState,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    recenter_orbit: &mut bool,
    close: &mut Option<(MolType, usize)>,
    cam_snapshot: &mut Option<usize>,
    pep_center: Vec3,
    reset_fog: &mut bool,
    mol_audio_playing: &Option<PlayingAudio>,
    audio_action: &mut Option<AudioAction>,
) {
    let active = match active_mol {
        Some((mol_type_active, i)) => *mol_type_active == mol_type && *i == i_mol,
        _ => false,
    };

    let color = if active {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    };

    highlighted_box(active, Color32::from_rgb(55, 40, 40)).show(ui, |ui| {
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

                if mol_type != MolType::Pocket {
                    if let Some(copies) = &mut mol.selected_for_md {
                        md_copies_field(copies, ui);
                    }

                    let color_md = if mol.selected_for_md.is_some() {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if ui
                        .button(RichText::new("MD").color(color_md))
                        .on_hover_text(
                            "Select or deselect this molecule for molecular dynamics simulation.",
                        )
                        .clicked()
                    {
                        mol.selected_for_md = match mol.selected_for_md {
                            Some(_) => None,
                            None => Some(1),
                        };
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
                    move_cam_to_mol(
                        mol,
                        mol_type,
                        i_mol,
                        cam_snapshot,
                        scene,
                        orbit_center,
                        beyond,
                        engine_updates,
                    );
                    *reset_fog = true;
                }

                let row_h = ui.spacing().interact_size.y;

                let help_text =
                    "Make this molecule the active / selected one. Middle click to close it.";
                let sel_btn = ui
                    .add_sized(
                        egui::vec2(ui.available_width(), row_h),
                        egui::Button::new(RichText::new(mol.name(idents)).color(color)),
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
            pharmacophore::pharmacophore_summary(pm, i_mol, popup, ph_state, ui);
        }

        let playing_this_mol = mol_audio_playing
            .as_ref()
            .is_some_and(|audio| audio.is_for(mol_type, i_mol));

        let (text, color, hover_text) = if playing_this_mol {
            ("Pause", COLOR_ACTIVE, "Stop sonifying this molecule.")
        } else {
            (
                "Play",
                COLOR_ACTION,
                "Sonify this molecule using its force-field bond-stretching parameters.",
            )
        };

        if mol_type != MolType::Pocket
            && ui
                .button(RichText::new(text).color(color))
                .on_hover_text(hover_text)
                .clicked()
        {
            *audio_action = Some(AudioAction::Toggle(mol_type, i_mol));
        }

        ui.separator();
    });
}

fn toggle_audio(state: &mut State, mol_type: MolType, i_mol: usize) {
    if state.volatile.is_playing_audio_for(mol_type, i_mol) {
        state.volatile.playing_audio = None;
        return;
    }

    let (mol, ff_params) = match sonification_input(state, mol_type, i_mol) {
        Ok(input) => input,
        Err(e) => {
            handle_err(&mut state.ui, e);
            return;
        }
    };

    match sonification::play(&mol, &ff_params, SONIFICATION_INCLUDE_H) {
        Ok(handle) => {
            state.volatile.playing_audio = Some(PlayingAudio::new(mol_type, i_mol, handle));
        }
        Err(e) => handle_err(&mut state.ui, format!("Unable to play molecule audio: {e}")),
    }
}

fn sonification_input(
    state: &State,
    mol_type: MolType,
    i_mol: usize,
) -> Result<(MoleculeCommon, ForceFieldParams), String> {
    match mol_type {
        MolType::Peptide => {
            let mol = state
                .peptide
                .get(i_mol)
                .ok_or_else(|| "Peptide index is out of bounds.".to_string())?;
            let params = state
                .ff_param_set
                .peptide
                .as_ref()
                .ok_or_else(|| "No peptide force-field parameters are loaded.".to_string())?;

            Ok((mol.common.clone(), params.clone()))
        }
        MolType::Ligand => {
            let mol = state
                .ligands
                .get(i_mol)
                .ok_or_else(|| "Ligand index is out of bounds.".to_string())?;
            let params = state.ff_param_set.small_mol.as_ref().ok_or_else(|| {
                "No small-molecule force-field parameters are loaded.".to_string()
            })?;

            Ok((
                mol.common.clone(),
                merge_mol_specific_params(
                    params,
                    mol_specific_params(&state.mol_specific_params, &mol.common.ident),
                ),
            ))
        }
        MolType::NucleicAcid => {
            let mol = state
                .nucleic_acids
                .get(i_mol)
                .ok_or_else(|| "Nucleic acid index is out of bounds.".to_string())?;
            let params = match mol.na_type {
                NucleicAcidType::Dna => &state.ff_param_set.dna,
                NucleicAcidType::Rna => &state.ff_param_set.rna,
            }
            .as_ref()
            .ok_or_else(|| format!("No {} force-field parameters are loaded.", mol.na_type))?;

            Ok((mol.common.clone(), params.clone()))
        }
        MolType::Lipid => {
            let mol = state
                .lipids
                .get(i_mol)
                .ok_or_else(|| "Lipid index is out of bounds.".to_string())?;
            let params = state
                .ff_param_set
                .lipids
                .as_ref()
                .ok_or_else(|| "No lipid force-field parameters are loaded.".to_string())?;

            Ok((mol.common.clone(), params.clone()))
        }
        MolType::Pocket | MolType::Water => {
            Err("This molecule type cannot be sonified.".to_string())
        }
    }
}

fn mol_specific_params<'a>(
    params: &'a std::collections::HashMap<String, ForceFieldParams>,
    ident: &str,
) -> Option<&'a ForceFieldParams> {
    params.get(ident).or_else(|| {
        params
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(ident))
            .map(|(_, params)| params)
    })
}

fn merge_mol_specific_params(
    general: &ForceFieldParams,
    specific: Option<&ForceFieldParams>,
) -> ForceFieldParams {
    match specific {
        Some(specific) => merge_params(general, specific),
        None => general.clone(),
    }
}

/// Select, close, hide etc molecules from ones opened.
fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let mut recenter_orbit = false;
    let mut close = None; // Avoids borrow error.

    // Avoids a double borrow. Non-peptide camera rows use the current peptide as context.
    let pep_center = state
        .peptide_for_tools()
        .map(|mol| mol.center)
        .unwrap_or_else(Vec3::new_zero);

    let mut reset_fog = false;
    let mut audio_action = None;

    for (i_mol, mol) in state.peptide.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            &mut state.pharmacophore,
            i_mol,
            &mut mol.common,
            None,
            &None,
            None,
            MolType::Peptide,
            &mut state.ui.popup,
            scene,
            ui,
            updates,
            &mut redraw.peptide,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
            &mut reset_fog,
            &state.volatile.playing_audio,
            &mut audio_action,
        );
    }

    for (i_mol, mol) in state.ligands.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            &mut state.pharmacophore,
            i_mol,
            &mut mol.common,
            Some(&mol.idents),
            &mol.characterization,
            Some(&mol.pharmacophore),
            MolType::Ligand,
            &mut state.ui.popup,
            scene,
            ui,
            updates,
            &mut redraw.ligand,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
            &mut reset_fog,
            &state.volatile.playing_audio,
            &mut audio_action,
        );
    }

    for (i_mol, mol) in state.lipids.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            &mut state.pharmacophore,
            i_mol,
            &mut mol.common,
            None,
            &None,
            None,
            MolType::Lipid,
            &mut state.ui.popup,
            scene,
            ui,
            updates,
            &mut redraw.lipid,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
            &mut reset_fog,
            &state.volatile.playing_audio,
            &mut audio_action,
        );
    }

    for (i_mol, mol) in state.nucleic_acids.iter_mut().enumerate() {
        // todo: Characterization, e.g. by dna seq?
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            &mut state.pharmacophore,
            i_mol,
            &mut mol.common,
            None,
            &None,
            None,
            MolType::NucleicAcid,
            &mut state.ui.popup,
            scene,
            ui,
            updates,
            &mut redraw.na,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
            &mut reset_fog,
            &state.volatile.playing_audio,
            &mut audio_action,
        );
    }

    for (i_mol, mol) in state.pockets.iter_mut().enumerate() {
        // todo: Characterization, e.g. by dna seq?
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            &mut state.pharmacophore,
            i_mol,
            &mut mol.common,
            None,
            &None,
            None,
            MolType::Pocket,
            &mut state.ui.popup,
            scene,
            ui,
            updates,
            &mut redraw.pocket,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
            &mut reset_fog,
            &state.volatile.playing_audio,
            &mut audio_action,
        );
    }

    // Removed, for now.

    // for (i_mol, pm) in state.pharmacophores.iter_mut().enumerate() {
    //     label!(
    //         ui,
    //         format!("Pharmacophore name: {} ident: {}", pm.name, pm.mol_ident),
    //         Color32::WHITE
    //     );
    //     //
    //     // if let Some(pm) = pharmacophore
    //     //     && !pm.features.is_empty()
    //     // {
    //     pharmacophore::pharmacophore_summary(
    //         pm,
    //         i_mol,
    //         &mut state.ui.popup,
    //         &mut state.pharmacophore,
    //         ui,
    //     );
    //     // }
    // }

    // todo: AAs here too?

    // Peptide-relative UI state follows whichever peptide was selected in the picker.
    if recenter_orbit
        && let Some((MolType::Peptide, peptide_i)) = state.volatile.active_mol
        && let Some(peptide) = state.peptide.get(peptide_i)
    {
        state.volatile.active_peptide = Some(peptide_i);
        state.volatile.aa_seq_text = peptide
            .aa_seq
            .iter()
            .map(|aa| aa.to_str(AaIdent::OneLetter))
            .collect();
        state.volatile.flags.ss_mesh_created = false;
        state.volatile.flags.sas_mesh_created = false;
    }
    if let Some(AudioAction::Toggle(mol_type, i_mol)) = audio_action {
        toggle_audio(state, mol_type, i_mol);
    }

    if let Some((mol_type, i_mol)) = close {
        close_mol(mol_type, i_mol, state, scene, redraw, updates);
    }

    if recenter_orbit
        && let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme
    {
        *center = orbit_center(state);
    }

    if reset_fog {
        set_fog(state, &mut scene.camera);
    }
}

fn open_tools(state: &mut State, ui: &mut Ui) {
    let color_open_tools = if state.peptide.is_empty() && state.ligands.is_empty() {
        COLOR_ACTION
    } else {
        COLOR_INACTIVE
    };

    if button!(
        ui,
        "Open",
        color_open_tools,
        "Open a molecule, electron density, or other file from disk."
    )
    .clicked()
    {
        state.volatile.dialogs.load.pick_file();
    }

    if button!(
        ui,
        "Recent",
        color_open_tools,
        "Select a recently-opened file to open"
    )
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
            if button!(
                ui,
                "↔",
                color_move,
                "(Hotkey: M. M or Esc to stop)) Move the active molecule by clicking and dragging with /
                the mouse. Scroll to move it forward and back."
            ).clicked() {
                set_manip(state, scene, redraw, &mut false,
                          ManipMode::Move((active_mol_type, active_mol_i)), engine_updates);
            }

            if button!(
                ui,
                "⟳",
                color_rotate,
                "(Hotkey: R. R or Esc to stop) Rotate the active molecule by clicking and dragging with the mouse. Scroll to roll."
            ).clicked() {
                set_manip(state,
                          scene, redraw, &mut false,
                          ManipMode::Rotate((active_mol_type, active_mol_i)), engine_updates, );
            }
        }

        let mut pocket_mesh_stale = false;
        if let Some(mol) = &mut state.active_mol_mut() {
            if ui
                .button(RichText::new("Move to cam").color(COLOR_HIGHLIGHT))
                .on_hover_text("Move the molecule to be a short distance in front of the camera.")
                .clicked()
            {
                move_mol_to_cam(mol.common_mut(), &scene.camera);
                if active_mol_type == MolType::Pocket {
                    pocket_mesh_stale = true;
                }
                redraw.set(active_mol_type);
            }

            if button!(
                ui,
                "Reset pos",
                COLOR_HIGHLIGHT,
                "Move the molecule to its absolute coordinates, e.g. as defined in /
                        its source mmCIF, Mol2 or SDF file."
            ).clicked() {
                mol.common_mut().reset_posits();
                if active_mol_type == MolType::Pocket {
                    pocket_mesh_stale = true;
                }
                // todo: Use the inplace move.
                redraw.set(active_mol_type);
            }

            let color_details = if state.ui.ui_vis.mol_char {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };
            if button!(
                ui,
                "Details",
                color_details,
                "Toggle the details display of the active molecule."
            ).clicked() {
                state.ui.ui_vis.mol_char = !state.ui.ui_vis.mol_char;
            }
        }

        // Pocket meshes are stored in world space, so a position change requires
        // regenerating the mesh; a simple entity redraw is not sufficient.
        if pocket_mesh_stale {
            let pocket = &mut state.pockets[active_mol_i];
            pocket.regen_mesh_vol(&mut scene.meshes, engine_updates);
        }
    });
}

pub(in crate::ui) fn sidebar(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    let edit_mode = state.volatile.operating_mode == OperatingMode::MolEditor;

    // When hidden, all that remains of the sidebar is a narrow strip with a button to show it again.
    if !state.ui.ui_vis.sidebar {
        let out = egui::Panel::left("sidebar_hidden")
            .resizable(false)
            .exact_size(SIDEBAR_HIDDEN_WIDTH)
            .show_inside(ui, |ui| {
                if button!(ui, "▶", COLOR_ACTION, "Show the sidebar").clicked() {
                    state.ui.ui_vis.sidebar = true;
                }
            });

        updates.ui_reserved_px.0 = out.response.rect.width();
        return;
    }

    let out = egui::Panel::left("sidebar")
        .resizable(true) // let user drag the width
        .default_size(140.0)
        .size_range(60.0..=800.0)
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if button!(ui, "◀", COLOR_ACTION, "Hide the sidebar").clicked() {
                    state.ui.ui_vis.sidebar = false;
                }
                ui.add_space(COL_SPACING);

                let color_open_tools = if state.peptide.is_empty() && state.ligands.is_empty() {
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
                if let Some(mol) = state.active_mol()
                    && ui
                        .button(RichText::new("Save"))
                        .on_hover_text("Save the active molecule to a file.")
                        .clicked()
                {
                    mol_to_save = Some((mol.common().clone(), mol.mol_type()));
                }

                if let Some((mol, mol_type)) = mol_to_save
                    && mol
                        .save(mol_type, &mut state.volatile.dialogs.save)
                        .is_err()
                {
                    handle_err(&mut state.ui, "Problem saving this file".to_owned());
                }
            });

            ui.add_space(ROW_SPACING / 2.);

            if !edit_mode {
                manip_toolbar(state, scene, redraw, ui, updates);
            }

            ui.add_space(ROW_SPACING / 2.);
            ui.separator();
            ui.add_space(ROW_SPACING / 2.);

            // todo: Function or macro to reduce this DRY.

            if edit_mode {
                mol_editor_sidebar::pocket_list(state, scene, updates, ui);
            } else {
                mol_picker(state, scene, ui, redraw, updates);
            }

            traj_items(state, scene, updates, ui, redraw);
            md_viewer::viewer_mol_set(state, scene, updates, ui, redraw);

            ui.add_space(ROW_SPACING);

            if state.ui.ui_vis.pharmacophore_list && edit_mode {
                mol_editor_sidebar::pharmacophore_list(state, ui);
            }

            if edit_mode {
                mol_editor_sidebar::component_list(state, ui, &mut redraw.ligand);
            }

            // todo: UI flag to show or hide this.
            if state.ui.ui_vis.mol_char && !edit_mode {
                // Avoid double borrow.
                let mut run_logp_sim = false;
                let mut run_crystal_sim = false;
                let mut run_water_sol_sim_mix = false;
                let mut run_water_sol_sim_layers = false;
                let mut run_shrinking_box = false;
                let mut new_crystal_mol = None;
                // let mut run_water_sol_sim_layers_middle = false;

                if let Some(m) = &state.active_mol()
                    && let MolGenericRef::Small(mol) = m
                {
                    char_adme::mol_char_disp(
                        mol,
                        ui,
                        &mut run_logp_sim,
                        &mut run_crystal_sim,
                        &mut run_water_sol_sim_mix,
                        &mut run_water_sol_sim_layers,
                        &mut run_shrinking_box,
                        &mut new_crystal_mol,
                    );
                }

                if let Some(mol) = new_crystal_mol {
                    let new_i = state.ligands.len();
                    state.ligands.push(mol);

                    state.volatile.active_mol = Some((MolType::Ligand, new_i));
                    state.volatile.orbit_center = Some((MolType::Ligand, new_i));
                    redraw.ligand = true;
                }

                // Run triggers for experimental MD-based algorithms. Most of these will be changed
                // or removed.
                // ------
                // todo: RM A/RE
                md_property_runners(
                    state,
                    scene,
                    updates,
                    redraw,
                    run_logp_sim,
                    run_crystal_sim,
                    run_water_sol_sim_mix,
                    run_water_sol_sim_layers,
                    run_shrinking_box,
                    // run_water_sol_sim_layers_middle,
                );
            }

            if !edit_mode
                && let Some(MolGenericRef::Peptide(mol)) = state.active_mol()
                && let Some(sifts) = &mol.sifts_mapping
            {
                label!(ui, "SIFTS Mappings", Color32::WHITE);

                for sift in sifts {
                    label!(
                        ui,
                        &format!("Accession: {}, Ident: {}", sift.accession, sift.identifier),
                        Color32::GRAY
                    );
                    for mapping in &sift.mappings {
                        // todo: Format as you wish
                        ui.horizontal(|ui| {
                            //     pub entity_id: u32,
                            //     /// PDB chain identifier (author label), e.g. `"A"`.
                            //     pub chain_id: String,
                            //     /// Internal asymmetric-unit chain ID used in mmCIF files.
                            //     pub struct_asym_id: String,
                            //     /// First residue of this segment in the **UniProt** sequence (1-based).
                            //     pub unp_start: u32,
                            //     /// Last residue of this segment in the **UniProt** sequence (1-based).
                            //     pub unp_end: u32,
                            //     /// First residue of this segment in the **PDB** structure.
                            //     pub start: SiftsResiduePosition,
                            //     /// Last residue of this segment in the **PDB** structure.
                            //     pub end: SiftsResiduePosition,
                            //     /// Sequence identity between the PDB chain and the UniProt sequence (0–1).
                            //     pub identity: f32,
                            //     /// Fraction of the UniProt sequence covered by this structure (0–1).
                            //     pub coverage: f32,

                            let summary = format!(
                                "ID: {}, Chain: {} Asym: {} Cov: {:.2}%",
                                mapping.entity_id,
                                mapping.chain_id,
                                mapping.struct_asym_id,
                                mapping.coverage
                            );

                            label!(ui, summary, Color32::GRAY);

                            ui.add_space(COL_SPACING);

                            if button!(ui, "Select", Color32::GREEN, "Select all atoms in this")
                                .clicked()
                            {
                                // todo
                            }
                        });
                    }
                    ui.add_space(ROW_SPACING);
                }
            }
        });

    updates.ui_reserved_px.0 = out.response.rect.width();
}

/// Let the user view open trajectories, and possibly change frames etc from them.
fn traj_items(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    if state.trajectories.is_empty() {
        return;
    }
    ui.add_space(ROW_SPACING);

    ui.label("MD Trajectories");
    ui.separator();

    let mut close = None;
    let mut snaps_loaded = false;
    let mut traj_active = None;

    for (i, traj) in state.trajectories.iter_mut().enumerate() {
        highlighted_box(traj.ui_active, Color32::from_rgb(40, 55, 40)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&traj.display_name).color(Color32::WHITE));

                if traj.num_frames <= MAX_FRAMES_TO_ATTEMPT_LOADING
                    && matches!(traj.source, TrajectorySource::File(_))
                    && traj.num_frames != 0
                    && button!(
                        ui,
                        "Load all frames",
                        COLOR_ACTION,
                        "Load all frames/snapshots from the trajectory into memory"
                    )
                    .clicked()
                {
                    match traj.load_snaps(FrameSlice::Index {
                        start: None,
                        end: None,
                    }) {
                        Ok(snaps) => {
                            state.volatile.md_local.replace_snaps(snaps);
                            snaps_loaded = true;
                            traj_active = Some(i);
                        }
                        Err(e) => {
                            handle_err(
                                &mut state.ui,
                                format!("Error loading snapshots from trajectory: {:?}", e),
                            );
                        }
                    }
                }

                // Load memory-only trajectories by replacing viewer snaps with theirs, but without
                // loading anything from disk.
                if let TrajectorySource::Memory(snaps) = &traj.source
                    && button!(
                        ui,
                        "View frames",
                        COLOR_ACTION,
                        "View all frames from this in-memory trajectory."
                    )
                    .clicked()
                {
                    state.volatile.md_local.replace_snaps(snaps.clone());
                    snaps_loaded = true;
                    traj_active = Some(i);
                }

                // todo: Allow end and start to be unbounded in UI, setting their val to None.
                num_field(&mut traj.ui_start_i, "", 44, ui);
                ui.label("-");
                num_field(&mut traj.ui_end_i, "", 44, ui);

                // todo: ALso check on time if that's the bounds. For now, we have index only, as a start.
                if traj.num_frames <= MAX_FRAMES_TO_ATTEMPT_LOADING
                    && matches!(traj.source, TrajectorySource::File(_))
                    && traj.ui_end_i < traj.num_frames
                    && traj.ui_start_i < traj.ui_end_i
                    && button!(
                        ui,
                        "Load rng",
                        COLOR_ACTION,
                        "Load frames/snapshots from the selected indices into memory"
                    )
                    .clicked()
                {
                    let start = if traj.ui_start_i == 0 {
                        None
                    } else {
                        Some(traj.ui_start_i)
                    };
                    let end = if traj.ui_end_i == 0 {
                        None
                    } else {
                        Some(traj.ui_end_i)
                    };

                    match traj.load_snaps(FrameSlice::Index { start, end }) {
                        Ok(snaps) => {
                            state.volatile.md_local.replace_snaps(snaps);
                            snaps_loaded = true;
                            traj_active = Some(i);
                        }
                        Err(e) => {
                            handle_err(
                                &mut state.ui,
                                format!("Error loading snapshots from trajectory: {:?}", e),
                            );
                        }
                    }
                }

                if ui
                    .button(RichText::new("❌").color(Color32::LIGHT_RED))
                    .on_hover_text("Close this trajectory.")
                    .clicked()
                {
                    close = Some(i);
                }
            });

            let txt = format!(
                "At: {}, Fr: {}, step: {:.3}, inter: {}, dt: {:.3}ps, end: {:.1}ps",
                traj.num_atoms,
                traj.num_frames,
                traj.start_step,
                traj.save_interval_steps,
                traj.dt,
                traj.end_time,
            );
            ui.label(RichText::new(txt).color(Color32::WHITE));

            if let Some(slice) = &traj.frames_open {
                label!(ui, format!("Open: {slice}"), COLOR_ACTIVE);
            }

            match state.volatile.md_local.viewer.get_active_mol_set() {
                Some(set) => {
                    if set.atom_count == traj.num_atoms {
                        label!(
                            ui,
                            format!(
                                "Set loaded with correct atom count. {} mols",
                                set.mols.len()
                            ),
                            COLOR_ACTIVE
                        );
                    } else {
                        label!(
                            ui,
                            format!("Mol set mismatch. {} atoms in set", set.atom_count),
                            Color32::YELLOW
                        );
                    }
                }
                None => {
                    label!(ui, "No mol set loaded", Color32::LIGHT_RED);
                }
            }
        });
    }

    ui.separator();

    if let Some(i) = close {
        close_traj(state, i);
    }

    // We have this as the function calls in this branch which call state have a borrow
    // error otherwise; the flag setting is convenience.
    if snaps_loaded {
        reset_camera(state, scene, updates, FWD_VEC);
        viewer::draw_mols(state, scene, updates);

        redraw.set_all();

        handle_success(
            &mut state.ui,
            format!(
                "Loaded {} frames into the viewer",
                state.volatile.md_local.viewer.snapshots.len()
            ),
        );
    }

    if let Some(i) = traj_active {
        // Note: `load_snaps` sets active, but doesn't clear this flag from others.
        for (j, traj) in state.trajectories.iter_mut().enumerate() {
            traj.ui_active = i == j;
        }
    }
}

/// todo: These are WIP; most or all will change or be removed.
fn md_property_runners(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    run_logp_sim: bool,
    run_crystal_sim: bool,
    run_water_sol_sim_mix: bool,
    run_water_sol_sim_layers: bool,
    run_shrinking_box: bool,
    // run_water_sol_sim_layers_middle: bool,
) {
    let Some(active_mol) = state.volatile.active_mol.as_ref() else {
        return;
    };
    let Some(mol) = state.get_small(active_mol.1).cloned() else {
        return;
    };

    if run_logp_sim {
        match logp::run(&mol, state, scene, updates) {
            Ok(v) => println!("LogP: sim result: {v}"),
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error running the LogP simulation: {e:?}"),
            ),
        }
    }

    if run_crystal_sim {
        // todo: for testing, let the UI control this.
        match crystal::run_crystal_sim(
            &mol,
            state.to_save.md.backend,
            &state.dev,
            &state.ff_param_set,
        ) {
            Ok((data, snaps)) => {
                state.trajectories.push(Trajectory::new_in_memory(
                    snaps,
                    "Crystal sim".to_string(),
                    0.002, // todo?
                ));

                let mol_set = ViewerMolSet::from_mols(
                    "Crystal sim".to_string(),
                    &vec![(
                        MolType::Ligand,
                        mol.common.clone(),
                        data.md_properties.as_ref().unwrap().copy_count,
                    )],
                );
                state.volatile.md_local.viewer.mol_sets.push(mol_set);
                state.volatile.md_local.viewer.mol_set_active =
                    Some(state.volatile.md_local.viewer.mol_sets.len() - 1);

                println!("Crystal sim result: {data:?}");
            }
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error running the crystal simulation: {e:?}"),
            ),
        }
    }

    if run_water_sol_sim_mix {
        match water_sol::run_sol_sim(
            &mol,
            state.to_save.md.backend, // todo: for testing, let the UI control this.
            &state.dev,
            &state.ff_param_set,
        ) {
            Ok((data, snaps)) => {
                let water_count = if data.water_molecule_count > 0 {
                    data.water_molecule_count
                } else {
                    snaps
                        .last()
                        .map(|snap| snap.water_o_posits.len())
                        .unwrap_or_default()
                };

                state.trajectories.push(Trajectory::new_in_memory(
                    snaps.clone(),
                    "Water sol sim".to_string(),
                    0.002, // todo?
                ));

                let viewer_mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];
                state
                    .volatile
                    .md_local
                    .viewer
                    .add_mol_set(&viewer_mols, water_count);

                let set_i = state
                    .volatile
                    .md_local
                    .viewer
                    .mol_sets
                    .len()
                    .saturating_sub(1);
                if let Some(set) = state.volatile.md_local.viewer.mol_sets.get_mut(set_i) {
                    set.name = "Water sol sim".to_string();
                }
                state.volatile.md_local.viewer.mol_set_active = Some(set_i);
                state.volatile.md_local.replace_snaps(snaps);
                viewer::draw_mols(state, scene, updates);
                redraw.set_all();

                let hydration_text = format!("{:.3} kcal/mol", data.hyd_free_energy);

                handle_success(
                    &mut state.ui,
                    format!(
                        "Water solvation complete. Hydration dG: {hydration_text}; affinity score: {:.3}",
                        data.md_water_affinity_score
                    ),
                );

                println!("\n\nWater sol sim result: {data:?}\n---\n");
                println!(
                    "Free en: alch en: {:?}, al sem: {:?}, hyd free en: {:?}, hyd sem: {:?}",
                    data.alch_decoupling_free_energy,
                    data.alch_decoupling_free_energy_sem,
                    data.hyd_free_energy,
                    data.hyd_free_energy_sem
                );
            }
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error running the water solubility simulation: {e:?}"),
            ),
        }
    }

    if run_water_sol_sim_layers {
        match water_sol_mix::run_boundary_layer_sol_sim(
            &mol,
            state.to_save.md.backend,
            &state.dev,
            &state.ff_param_set,
        ) {
            Ok((data, snaps)) => {
                let water_count = snaps
                    .last()
                    .map(|snap| snap.water_o_posits.len())
                    .unwrap_or_default();

                state.trajectories.push(Trajectory::new_in_memory(
                    snaps.clone(),
                    "Water/solute layer sim".to_string(),
                    0.002,
                ));

                let viewer_mols =
                    vec![(FfMolType::SmallOrganic, &mol.common, data.solute_copy_count)];
                state
                    .volatile
                    .md_local
                    .viewer
                    .add_mol_set(&viewer_mols, water_count);

                let set_i = state
                    .volatile
                    .md_local
                    .viewer
                    .mol_sets
                    .len()
                    .saturating_sub(1);
                if let Some(set) = state.volatile.md_local.viewer.mol_sets.get_mut(set_i) {
                    set.name = "Water/solute layer sim".to_string();
                }
                state.volatile.md_local.viewer.mol_set_active = Some(set_i);
                state.volatile.md_local.replace_snaps(snaps);
                viewer::draw_mols(state, scene, updates);
                redraw.set_all();

                handle_success(
                    &mut state.ui,
                    format!(
                        "Boundary-layer simulation complete. {} solute copies, {} waters loaded, box {:.1} x {:.1} x {:.1} A, slabs {:.1}/{:.1} A over {:.0} A2.",
                        data.solute_copy_count,
                        water_count,
                        data.box_extent_a.x,
                        data.box_extent_a.y,
                        data.box_extent_a.z,
                        data.solute_layer_depth_a,
                        data.water_layer_depth_a,
                        data.interface_area_a2,
                    ),
                );

                println!("\n\nWater/solute layer sim result: {data:?}\n---\n");
            }
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error running the water/solute layer simulation: {e:?}"),
            ),
        }
    }

    if run_shrinking_box {
        // sol_shrinking_box::runner::run_on_select_mols(&state.dev, &state.ff_param_set);

        match sol_shrinking_box::run_shrinking_box_sim(
            &mol,
            // sol_shrinking_box::ShrinkingBoxMode::HomogeneousMix,
            sol_shrinking_box::ShrinkingBoxMode::WaterSoluteLayers,
            state.to_save.md.backend,
            &state.dev,
            &state.ff_param_set,
        ) {
            Ok((data, snaps, playback_mols)) => {
                let water_count = snaps
                    .last()
                    .map(|snap| snap.water_o_posits.len())
                    .unwrap_or(data.water_molecule_count);

                state.trajectories.push(Trajectory::new_in_memory(
                    snaps.clone(),
                    "Shrinking box sim".to_string(),
                    0.002,
                ));

                let viewer_mols = playback_mols
                    .iter()
                    .map(|mol| (mol.mol_type, &mol.mol, mol.count))
                    .collect::<Vec<_>>();

                state
                    .volatile
                    .md_local
                    .viewer
                    .add_mol_set(&viewer_mols, water_count);

                let set_i = state
                    .volatile
                    .md_local
                    .viewer
                    .mol_sets
                    .len()
                    .saturating_sub(1);

                if let Some(set) = state.volatile.md_local.viewer.mol_sets.get_mut(set_i) {
                    set.name = "Shrinking box sim".to_string();
                }

                state.volatile.md_local.viewer.mol_set_active = Some(set_i);
                state.volatile.md_local.replace_snaps(snaps);

                viewer::draw_mols(state, scene, updates);
                redraw.set_all();

                handle_success(
                    &mut state.ui,
                    format!(
                        "Shrinking-box simulation complete. | Sol est: {:.4} BH: {:.4} |, {} solute copies, {} waters, density {:.3} g/cm3, solubility {:.3}, box {:.1} -> {:.1} A.",
                        data.solubility_estimate,
                        data.solubility_estimate_barnes_hut,
                        data.solute_copy_count,
                        water_count,
                        data.density_g_cm3,
                        data.solubility_estimate,
                        data.initial_box_extent_a.x,
                        data.final_box_extent_a.x,
                    ),
                );

                println!("\n\nShrinking box sim result: {data:?}\n---\n");
            }
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error running the shrinking-box simulation: {e:?}"),
            ),
        }
    }
}
