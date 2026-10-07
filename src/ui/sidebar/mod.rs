use bio_apis::pubchem;
use bio_files::{FrameSlice, md_params::ForceFieldParams};
use dynamics::{FfMolType, merge_params};
use egui::{Color32, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;
use mol_defs::molecules::{MolGenericRef, MolGenericRefMut, MolIdent, MolType};

use crate::{
    button,
    cam::{VIEW_DIR_FRONT, move_mol_to_cam, reset_camera},
    file_io::{save_mol, sequence::save_seq_dialog},
    label,
    md::{
        trajectory::{MAX_FRAMES_TO_ATTEMPT_LOADING, Trajectory, TrajectorySource, close_traj},
        viewer,
        viewer::ViewerMolSet,
    },
    mol_manip::{ManipMode, set_manip},
    pocket_render::PocketRender,
    properties::{crystal, logp, sol_shrinking_box, water_sol, water_sol_mix},
    state::{MetadataTarget, OperatingMode, State},
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        highlighted_box, load_all_idents_button, num_field,
        panels::md_viewer,
        util::{Idents, list_idents},
    },
    util::{RedrawFlags, handle_err, handle_success},
};

mod char_adme;
mod mol_editor_sidebar;
mod mol_picker;

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

fn open_tools(state: &mut State, ui: &mut Ui) {
    let color_open_tools = if state.peptides.is_empty() && state.ligands.is_empty() {
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
        let mut add_copy = false;
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
                "Add copy",
                COLOR_HIGHLIGHT,
                "Load an additional copy of this molecule"
            ).clicked() {
                // Defer the push until the `mol` borrow of `state` is released below.
                add_copy = true;
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

            {
                let color = if state.ui.ui_vis.mol_picker_details {
                    COLOR_ACTIVE
                } else {
                    COLOR_INACTIVE
                };
                if button!(
                    ui,
                    "Details",
                    color,
                    "Toggle details and controls under each molecule. If you have many molecules open at \
                    once, you may wish to deselect this to declutter the display."
                ).clicked() {
                    state.ui.ui_vis.mol_picker_details = !state.ui.ui_vis.mol_picker_details;
                }
            }

            {
                let color = if state.ui.ui_vis.sidebar_mol_properties {
                    COLOR_ACTIVE
                } else {
                    COLOR_INACTIVE
                };
                if button!(
                    ui,
                    "Properties",
                    color,
                    "Toggle the properties display of the active molecule. This includes detailed numerical data \
                    about the molecule, and ADME properties."
                ).clicked() {
                    state.ui.ui_vis.sidebar_mol_properties = !state.ui.ui_vis.sidebar_mol_properties;
                }
            }
        }

        if add_copy {
            match state.active_mol_mut() {
                Some(MolGenericRefMut::Peptide(m)) => {
                    let shift_amt = 40.;
                    let mut copy = m.clone();
                    copy.common.shift(Vec3::new(shift_amt, 0., 0.));

                    redraw.set(MolType::Peptide);
                    state.peptides.push(copy);
                }
                Some(MolGenericRefMut::Small(m)) => {
                    let shift_amt = 10.;
                    let mut copy = m.clone();
                    copy.common.shift(Vec3::new(shift_amt, 0., 0.));

                    redraw.set(MolType::Ligand);
                    state.ligands.push(copy);
                }
                Some(MolGenericRefMut::NucleicAcid(m)) => {
                    let shift_amt = 20.;
                    let mut copy = m.clone();
                    copy.common.shift(Vec3::new(shift_amt, 0., 0.));

                    redraw.set(MolType::NucleicAcid);
                    state.nucleic_acids.push(copy);
                }
                Some(MolGenericRefMut::Lipid(m)) => {
                    let shift_amt = 20.;
                    let mut copy = m.clone();
                    copy.common.shift(Vec3::new(shift_amt, 0., 0.));

                    state.lipids.push(copy);
                    redraw.set(MolType::Lipid);
                }
                Some(MolGenericRefMut::Pocket(m)) => {
                    let shift_amt = 40.;
                    let mut copy = m.clone();
                    copy.common.shift(Vec3::new(shift_amt, 0., 0.));
                    state.pockets.push(copy);

                    redraw.set(MolType::Pocket);
                }
                None => {}
            }

            engine_updates.meshes = true;
        }

        // Pocket meshes are stored in world space, so a position change requires
        // regenerating the mesh; a simple entity redraw is not sufficient.
        if pocket_mesh_stale {
            let pocket = &mut state.pockets[active_mol_i];
            pocket.regen_mesh_vol(&mut scene.meshes, engine_updates);
        }
    });
}

/// Per-molecule functionality, lookups, popups etc.
fn mol_specific_aux_btns(
    mol_type: MolType,
    load_all_idents: &mut bool,
    toggle_metadata_popup: &mut bool,
    show_reactions: &mut bool,
    find_assoc_structs: &mut Option<u32>,
    pubchem_cid: Option<u32>,
    loading_idents: bool,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        // The online identifier lookup is for small molecules only.
        if mol_type == MolType::Ligand && load_all_idents_button(ui, loading_idents) {
            *load_all_idents = true;
        }

        if button!(
            ui,
            "Metadata",
            Color32::GRAY,
            "Display metadata for this molecule"
        )
            .clicked()
        {
            *toggle_metadata_popup = true;
        }

        let reactions_help = if mol_type == MolType::Peptide {
            "Display Rhea reactions annotated to this protein through its UniProt mapping."
        } else {
            "Display Rhea (enzyme-catalogued) reactions involving this molecule; queries the Rhea API."
        };

        if button!(ui, "Reactions", Color32::GRAY, reactions_help).clicked() {
            *show_reactions = true;
        }

        if let Some(cid) = pubchem_cid
            && button!(
                ui,
                "Associated",
                Color32::GRAY,
                "Find proteins associated with this molecule, e.g. if it's a ligand which                 proteins it can bind to. This notably includes PDB urls"
            )
            .clicked()
        {
            *find_assoc_structs = Some(cid);
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
            .show(ui, |ui| {
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
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if button!(ui, "◀", COLOR_ACTION, "Hide the sidebar").clicked() {
                    state.ui.ui_vis.sidebar = false;
                }
                ui.add_space(COL_SPACING);

                let color_open_tools = if state.peptides.is_empty() && state.ligands.is_empty() {
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

                let active_seq = state
                    .volatile
                    .active_seq
                    .filter(|&i| i < state.sequences.len());
                if (state.active_mol().is_some() || active_seq.is_some())
                    && ui
                        .button(RichText::new("Save"))
                        .on_hover_text("Save the active molecule or sequence to a file.")
                        .clicked()
                {
                    if let Some(i) = active_seq {
                        save_seq_dialog(state, i);
                    } else if let Some(mol) = state.active_mol() {
                        // The dialog needs a mutable borrow of state below.
                        let common = mol.common().clone();
                        let mol_type = mol.mol_type();
                        if let Err(error) =
                            save_mol(&common, mol_type, &mut state.volatile.dialogs.save)
                        {
                            handle_err(&mut state.ui, format!("Problem saving this file: {error}"));
                        }
                    }
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
                mol_editor_sidebar::db_info(state, ui);
                mol_editor_sidebar::pocket_list(state, scene, updates, ui);
            } else {
                mol_picker::mol_picker(state, scene, ui, redraw, updates);
                mol_picker::seq_picker(state, ui);
                traj_items(state, scene, updates, ui, redraw);
                md_viewer::viewer_mol_set(state, scene, updates, ui, redraw);
            }

            ui.add_space(ROW_SPACING);

            if state.ui.ui_vis.pharmacophore_list && edit_mode {
                mol_editor_sidebar::pharmacophore_list(state, scene, updates, ui);
            }

            if edit_mode {
                mol_editor_sidebar::component_list(state, ui, &mut redraw.ligand);
            }

            // These are set by the aux buttons, which always display (regardless of the
            // Details toggle). Vars are to avoid a double borrow.
            let mut load_all_idents = false;
            let mut toggle_metadata_popup = false;
            let mut show_reactions = false;
            let mut find_assoc_structs = None; // PubChem CID to look up.

            let loading_idents = state.volatile.thread_receivers.all_idents_avail.is_some();

            let pubchem_cid = match state.active_mol() {
                Some(MolGenericRef::Small(m)) => m.idents.iter().find_map(|ident| match ident {
                    MolIdent::PubChem(cid) => Some(*cid),
                    _ => None,
                }),
                _ => None,
            };

            if let Some((mol_type, _)) = state.volatile.active_mol {
                mol_specific_aux_btns(
                    mol_type,
                    &mut load_all_idents,
                    &mut toggle_metadata_popup,
                    &mut show_reactions,
                    &mut find_assoc_structs,
                    pubchem_cid,
                    loading_idents,
                    ui,
                );
            }

            if !edit_mode {
                let name_change = match state.active_mol() {
                    Some(MolGenericRef::Small(mol)) => {
                        ui.add_space(ROW_SPACING);
                        list_idents(
                            Some(&mol.common.name),
                            Idents::Small(&mol.idents),
                            &mol.common.path,
                            &state.volatile.prefs_dir,
                            ui,
                        )
                    }
                    Some(MolGenericRef::Peptide(mol)) => {
                        ui.add_space(ROW_SPACING);
                        list_idents(
                            Some(&mol.common.name),
                            Idents::Peptide(&mol.idents),
                            &mol.common.path,
                            &state.volatile.prefs_dir,
                            ui,
                        )
                    }
                    _ => None,
                };

                if let Some(name) = name_change
                    && let Some(mut mol) = state.active_mol_mut()
                {
                    mol.common_mut().name = name;
                    redraw.set(mol.mol_type());
                }
            }

            if state.ui.ui_vis.sidebar_mol_properties && !edit_mode {
                // These vars are all to avoid a double borrow.
                let mut run_logp_sim = false;
                let mut run_crystal_sim = false;
                let mut run_water_sol_sim_mix = false;
                let mut run_water_sol_sim_layers = false;
                let mut run_shrinking_box = false;
                let mut new_crystal_mol = None;

                if let Some(MolGenericRef::Small(mol)) = state.active_mol() {
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

            if show_reactions {
                crate::reactions::open_for_active(state);
            }

            if let Some(cid) = find_assoc_structs {
                let already_loaded = match state.active_mol() {
                    Some(MolGenericRef::Small(m)) => !m.associated_structures.is_empty(),
                    _ => false,
                };

                if !already_loaded {
                    // todo: Don't block.
                    match pubchem::load_associated_structures(cid) {
                        Ok(data) => {
                            if let Some(MolGenericRefMut::Small(m)) = state.active_mol_mut() {
                                m.associated_structures = data;
                            }
                        }
                        Err(_) => handle_err(
                            &mut state.ui,
                            "Unable to find structures for this ligand".to_owned(),
                        ),
                    }
                }

                state.ui.popup.show_associated_structures = true;
            }

            if toggle_metadata_popup {
                state.ui.popup.metadata = match state.ui.popup.metadata {
                    Some(_) => None,
                    None => state
                        .volatile
                        .active_mol
                        .map(|(mol_type, i)| MetadataTarget::Mol(mol_type, i)),
                };
            }

            if load_all_idents
                && let Some((MolType::Ligand, ligand_i)) = state.volatile.active_mol
                && let Some(mol) = state.ligands.get(ligand_i)
            {
                crate::threads::start_all_idents_lookup(
                    &mut state.volatile.thread_receivers,
                    ligand_i,
                    mol.common.ident.clone(),
                    mol.idents.clone(),
                );
                handle_success(
                    &mut state.ui,
                    "Loading molecule identifiers from PubChem and ChEBI...".to_owned(),
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
        reset_camera(state, scene, updates, VIEW_DIR_FRONT);
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
