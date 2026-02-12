use std::{
    io::Cursor,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use bio_apis::{pubchem::find_cids_from_search, rcsb};
use bio_files::{DensityMap, density_from_2fo_fc_rcsb_gemmi};
use egui::{
    Color32, ComboBox, Context, Key, RichText, Slider, TextEdit, TextFormat, TextStyle,
    TopBottomPanel, Ui, text::LayoutJob,
};
use graphics::{ControlScheme, EngineUpdates, Scene};
use lin_alg::f32::Vec3;
use md::md_setup;
use mol_data::display_mol_data;
use na_seq::Element;
use popups::load_popups;

use crate::{
    cam,
    cam::{
        FOG_DIST_MAX, FOG_DIST_MIN, RENDER_DIST_NEAR, VIEW_DEPTH_NEAR_MAX, VIEW_DEPTH_NEAR_MIN,
        move_cam_to_sel,
    },
    cli,
    cli::autocomplete_cli,
    drawing::color_viridis,
    file_io::{
        download_mols::{load_atom_coords_rcsb, load_sdf_drugbank, load_sdf_pubchem},
        gemmi_path,
    },
    mol_editor::enter_edit_mode,
    molecules::{MolGenericRef, MolIdent},
    render::set_flashlight,
    selection::{Selection, ViewSelLevel},
    state::{CamSnapshot, OperatingMode, ResColoring, State},
    threads::handle_thread_rx,
    ui::{
        misc::section_box,
        mol_data::display_mol_data_peptide,
        mol_type_tools::mol_type_toolbars,
        orca::orca_input,
        sidebar::sidebar,
        util::{
            color_egui_from_f32, handle_redraw, init_with_scene, open_lig_from_input,
            update_file_dialogs,
        },
        view::{ui_section_vis, view_settings},
    },
    util::{
        RedrawFlags, check_prefs_save, close_mol, close_peptide, cycle_selected, handle_err,
        handle_scene_flags, handle_success, orbit_center, select_from_search,
    },
};

mod char_adme;
mod md;
pub mod misc;
mod mol_data;
mod mol_editor;
mod mol_type_tools;
mod orca;
mod pharmacophore;
mod popups;
mod rama_plot;
mod recent_files;
mod sidebar;
pub mod util;
mod view;

static INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

pub(in crate::ui) const ROW_SPACING: f32 = 10.;
pub(in crate::ui) const COL_SPACING: f32 = 30.;

const DENS_ISO_MIN: f32 = 0.6;
const DENS_ISO_MAX: f32 = 3.0;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

pub(in crate::ui) const COLOR_INACTIVE: Color32 = Color32::GRAY;
pub(in crate::ui) const COLOR_ACTIVE: Color32 = Color32::LIGHT_GREEN;
pub(in crate::ui) const COLOR_HIGHLIGHT: Color32 = Color32::LIGHT_BLUE;
pub(in crate::ui) const COLOR_ACTIVE_RADIO: Color32 = Color32::LIGHT_BLUE;
pub(in crate::ui) const _COLOR_ATTENTION: Color32 = Color32::ORANGE;

// Creation, simulation runs etc.
pub(in crate::ui) const COLOR_ACTION: Color32 = Color32::GOLD;

const COLOR_OUT_ERROR: Color32 = Color32::LIGHT_RED;
const COLOR_OUT_NORMAL: Color32 = Color32::WHITE;
const _COLOR_OUT_SUCCESS: Color32 = Color32::LIGHT_GREEN; // Unused for now

pub const _COLOR_POPUP: Color32 = Color32::from_rgb(20, 20, 30);

// Number of characters to display. E.g. the molecular description. Often long.
const MAX_TITLE_LEN: usize = 40;

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, scene: &mut Scene) {
    scene.window_title = title.to_owned();
    // ui.ctx().send_viewport_cmd(ViewportCommand::Title(title.to_string()));
}

pub fn num_field<T>(val: &mut T, label: &str, width: u16, ui: &mut Ui)
where
    T: std::fmt::Display + std::str::FromStr,
{
    ui.label(label);
    let mut val_str = val.to_string();

    if ui
        .add_sized(
            [width as f32, Ui::available_height(ui)],
            TextEdit::singleline(&mut val_str),
        )
        .changed()
    {
        if let Ok(v) = val_str.parse::<T>() {
            *val = v;
        }
    }
}

/// Handles keyboard and mouse input not associated with a widget.
pub fn handle_input(
    state: &mut State,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    scene: &mut Scene,
) {
    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                if let Err(e) = state.open_file(path, scene, engine_updates) {
                    handle_err(&mut state.ui, e.to_string());
                }
            }
        }
    });
}

fn get_snap_name(snap: Option<usize>, snaps: &[CamSnapshot]) -> String {
    match snap {
        Some(i) => snaps[i].name.clone(),
        None => "None".to_owned(),
    }
}

/// Toggles chain visibility
fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    let Some(mol) = &mut state.peptide else {
        return;
    };

    ui.horizontal(|ui| {
        ui.label("Chain vis:");
        for chain in &mut mol.chains {
            let color = misc::active_color(chain.visible);

            if ui
                .button(RichText::new(chain.id.clone()).color(color))
                .clicked()
            {
                chain.visible = !chain.visible;
                *redraw = true;
            }
        }

        ui.add_space(COL_SPACING);

        ui.label("Select residues from:");

        for (i, chain) in mol.chains.iter().enumerate() {
            let mut color = Color32::GRAY;
            if let Some(i_sel) = state.ui.chain_to_pick_res {
                if i == i_sel {
                    color = COLOR_ACTIVE
                }
            }
            if ui
                .button(RichText::new(chain.id.clone()).color(color))
                .clicked()
            {
                // Toggle behavior.
                if let Some(sel_i) = state.ui.chain_to_pick_res {
                    if i == sel_i {
                        state.ui.chain_to_pick_res = None;
                    } else {
                        state.ui.chain_to_pick_res = Some(i);
                    }
                } else {
                    state.ui.chain_to_pick_res = Some(i);
                }

                state.ui.popup.residue_selector = !state.ui.popup.residue_selector;
            }
        }

        if state.ui.chain_to_pick_res.is_some() {
            if ui.button("(None)").clicked() {
                state.ui.chain_to_pick_res = None;
            }
        }
    });
}

// todo: Update params A/R
fn draw_cli(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    reset_cam: &mut bool,
    ui: &mut Ui,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Out: ");
        let color = if state.ui.cmd_line_out_is_err {
            COLOR_OUT_ERROR
        } else {
            COLOR_OUT_NORMAL
        };
        ui.label(RichText::new(&state.ui.cmd_line_output).color(color));
    });

    ui.horizontal(|ui| {
        ui.label("In: ");
        let edit_resp = ui.add(
            TextEdit::singleline(&mut state.ui.cmd_line_input)
                .desired_width(200.)
                // Prevent losing focus on Tab;
                // .cursor_at_end(true)
                .lock_focus(true),
        );

        if edit_resp.changed() {
            // todo: Validate input and color-code?
        }

        ui.add_space(COL_SPACING / 2.);

        let button_clicked = ui.button(RichText::new("Submit")).clicked();
        // This  behavior of lost and has are due to the default EGUI beavhior of those keys.
        // We can use the `lock_focus(true)` method to prevent these, but we generally like them.
        let enter_pressed = edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        let tab_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Tab));
        let up_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::ArrowUp));
        let dn_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::ArrowDown));

        if tab_pressed && !state.ui.cmd_line_input.is_empty() {
            autocomplete_cli(&mut state.ui.cmd_line_input);

            // edit_resp.surrender_focus();
            // state.ui.cmd_line_input.push(' ');
            // state.ui.cmd_line_input.pop();
            // edit_resp.request_focus();
        }

        if up_pressed {
            if state.volatile.cli_input_selected != 0 {
                state.volatile.cli_input_selected -= 1;
            }
            if state.volatile.cli_input_history.len() > state.volatile.cli_input_selected {
                state.ui.cmd_line_input =
                    state.volatile.cli_input_history[state.volatile.cli_input_selected].clone();
            }
        }

        if dn_pressed {
            if state.volatile.cli_input_selected < state.volatile.cli_input_history.len() - 1 {
                state.volatile.cli_input_selected += 1;
            }
            if state.volatile.cli_input_history.len() > state.volatile.cli_input_selected {
                state.ui.cmd_line_input =
                    state.volatile.cli_input_history[state.volatile.cli_input_selected].clone();
            }
        }

        if (button_clicked || enter_pressed) && state.ui.cmd_line_input.len() >= 2 {
            // todo: Error color
            state.ui.cmd_line_output =
                match cli::handle_cmd(state, scene, engine_updates, &mut redraw.peptide, reset_cam)
                {
                    Ok(out) => {
                        state.ui.cmd_line_out_is_err = false;
                        out
                    }
                    Err(e) => {
                        eprintln!("Error processing command");
                        state.ui.cmd_line_out_is_err = true;
                        e.to_string()
                    }
                };

            state.ui.cmd_line_input = String::new();
            // Compensates for the default lose focus behavior; we still want the cursor to remain here.
            edit_resp.request_focus();
        }

        ui.add_space(COL_SPACING);
        residue_search(state, scene, redraw, ui);
    });
}

fn residue_search(state: &mut State, scene: &mut Scene, redraw: &mut RedrawFlags, ui: &mut Ui) {
    let (btn_text_p, btn_text_n, search_text) = match state.ui.view_sel_level {
        ViewSelLevel::Atom => ("Prev atom", "Next atom", "Find atom:"),
        ViewSelLevel::Residue => ("Prev AA", "Next AA", "Find res:"),
        ViewSelLevel::Bond => ("Prev bond", "Next bond", "Find bond:"),
    };

    ui.label(search_text);
    if ui
        .add(TextEdit::singleline(&mut state.ui.atom_res_search).desired_width(60.))
        .changed()
    {
        select_from_search(state);
        redraw.peptide = true;
    }

    if state.peptide.is_some() || !state.ligands.is_empty() {
        if ui
            .button(btn_text_p)
            .on_hover_text("(Hotkey: Left arrow)")
            .clicked()
        {
            cycle_selected(state, scene, true);

            match state.ui.selection {
                Selection::AtomPeptide(_) | Selection::Residue(_) | Selection::BondPeptide(_) => {
                    redraw.peptide = true
                }
                Selection::AtomLig(_) | Selection::BondLig(_) => redraw.ligand = true,
                Selection::AtomNucleicAcid(_) | Selection::BondNucleicAcid(_) => redraw.na = true,
                Selection::AtomLipid(_) | Selection::BondLipid(_) => redraw.lipid = true,
                Selection::AtomPocket(_) | Selection::BondPocket(_) => redraw.pocket = true,
                _ => (),
            }
        }
        // todo: DRY

        if ui
            .button(btn_text_n)
            .on_hover_text("(Hotkey: Right arrow)")
            .clicked()
        {
            cycle_selected(state, scene, false);

            match state.ui.selection {
                Selection::AtomPeptide(_) | Selection::Residue(_) => redraw.peptide = true,
                Selection::AtomLig(_) | Selection::BondLig(_) => redraw.ligand = true,
                Selection::AtomNucleicAcid(_) => redraw.na = true,
                Selection::AtomLipid(_) => redraw.lipid = true,
                Selection::AtomPocket(_) | Selection::BondPocket(_) => redraw.pocket = true,
                _ => (),
            }
        }

        // ui.add_space(COL_SPACING * 2.);
        //
        // let dock_tools_text = if state.ui.show_docking_tools {
        //     "Hide docking tools"
        // } else {
        //     "Show docking tools (Broken/WIP)"
        // };
        //
        // if ui.button(RichText::new(dock_tools_text)).clicked() {
        //     state.ui.show_docking_tools = !state.ui.show_docking_tools;
        // }
    }
}

/// The display for the amino acid sequence of an opened protein.
fn add_aa_seq(selection: &mut Selection, seq_text: &str, ui: &mut Ui, redraw: &mut bool) {
    let len = seq_text.len(); // One char per res.

    // This grey ensures that the whole viridis display range is clear, e.g. the purple
    // parse isn't blocked by our dark background.
    egui::Frame::new()
        // .fill(Color32::from_rgb(200, 200, 200))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (i, aa) in seq_text.chars().enumerate() {
                    let color = color_viridis(i, 0, len);
                    // todo: Find a cheaper way.
                    let mut color = Color32::from_rgb(
                        (color.0 * 255.) as u8,
                        (color.1 * 255.) as u8,
                        (color.2 * 255.) as u8,
                    );

                    if let Selection::Residue(sel) = selection {
                        if i == *sel {
                            color = Color32::from_rgb(255, 0, 0); // cheaper, but more maintenance than calling the const.
                        }
                    }

                    if ui.label(RichText::new(aa).color(color)).clicked() {
                        *selection = Selection::Residue(i);
                        *redraw = true;
                    }
                }
            });
        });
}

pub fn view_sel_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui, include_res: bool) {
    let help_text = "(Hotkeys:  ;  and  '  )";
    ui.label("View/Select:").on_hover_text(help_text);
    let prev_view = state.ui.view_sel_level;

    let mut views = vec![ViewSelLevel::Atom, ViewSelLevel::Bond];

    if include_res {
        views.push(ViewSelLevel::Residue);
    }

    // Ideally hover text here too, but I'm not sure how.
    ComboBox::from_id_salt(1)
        .width(80.)
        .selected_text(state.ui.view_sel_level.to_string())
        .show_ui(ui, |ui| {
            for view in &views {
                ui.selectable_value(&mut state.ui.view_sel_level, *view, view.to_string());
            }
        })
        .response
        .on_hover_text(help_text);

    if state.ui.view_sel_level != prev_view {
        *redraw = true;
        // If we change from atom to res, select the prev-selected atom's res. If vice-versa,
        // select that residue's Cα.
        // state.ui.selection = Selection::None;
        // todo: This section needs some updates, but isn't critical.
        if let Some(mol) = &state.peptide {
            match state.ui.view_sel_level {
                ViewSelLevel::Residue => {
                    state.ui.selection = match state.ui.selection {
                        Selection::AtomPeptide(i) => {
                            Selection::Residue(mol.common.atoms[i].residue.unwrap_or_default())
                        }
                        _ => Selection::None,
                    };
                }
                ViewSelLevel::Atom => {
                    state.ui.selection = match state.ui.selection {
                        // It seems [0] is often N, and [1] is Cα
                        Selection::Residue(i) => {
                            if i >= mol.residues.len() {
                                handle_err(&mut state.ui, "Residue bounds problem".to_string());
                                Selection::None
                            } else {
                                if mol.residues[i].atoms.len() <= 2 {
                                    Selection::AtomPeptide(mol.residues[i].atoms[1])
                                } else {
                                    Selection::None
                                }
                            }
                        }

                        _ => Selection::None,
                    };
                }
                ViewSelLevel::Bond => {}
            }
        }
    }

    // Buttons to alter the color profile, e.g. for res position, or partial charge.
    ui.add_space(COL_SPACING / 2.);
    match state.ui.view_sel_level {
        ViewSelLevel::Atom => {
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

                if state.peptide.is_some() {
                    state.volatile.flags.update_sas_coloring = true;
                }

                *redraw = true;
            }
        }
        ViewSelLevel::Residue => {
            let prev = state.ui.res_coloring;
            ComboBox::from_id_salt(11)
                .width(40.)
                .selected_text(state.ui.res_coloring.to_string())
                .show_ui(ui, |ui| {
                    for v in [
                        ResColoring::AminoAcid,
                        ResColoring::Position,
                        ResColoring::Hydrophobicity,
                    ] {
                        ui.selectable_value(&mut state.ui.res_coloring, v, v.to_string());
                    }
                });

            if state.ui.res_coloring != prev {
                if state.peptide.is_some() {
                    state.volatile.flags.update_sas_coloring = true;
                }
                *redraw = true;
            }
        }
        // todo: We could color these based on current vs nominal length and/or frequency.
        ViewSelLevel::Bond => (),
    }

    if state.ligands.len() >= 2 {
        let color = if state.ui.color_by_mol {
            COLOR_ACTIVE
        } else {
            COLOR_INACTIVE
        };

        if ui
            .button(RichText::new("Contrast ligs").color(color))
            .on_hover_text(
                "Color each small molecule a different color, so you can tell them apart.",
            )
            .clicked()
        {
            state.ui.color_by_mol = !state.ui.color_by_mol;
            *redraw = true;
        }
    }

    ui.add_space(COL_SPACING);
}
//
// fn mol_characterization(state: &mut State, ui: &mut Ui) {
//     if let Some(m) = state.active_mol() {
//         if let MolGenericRef::Small(mol) = m {
//             let Some(char) = &mol.characterization else {
//                 return;
//             };
//
//             // todo: Placeholder. Maybe we put this in a popup instead? Or at least make it hideable
//
//             section_box().show(ui, |ui| {
//                 ui.vertical(|ui| {
//
//                     ui.horizontal(|ui| {
//                         ui.label("Mol details: ");
//                         ui.label(char.to_string());
//
//                         let rings_sat = char.rings.iter().filter(|r| r.ring_type == RingType::Saturated).count();
//                         let rings_ar = char.rings.iter().filter(|r| r.ring_type == RingType::Aromatic).count();
//                         let rings_ali = char.rings.iter().filter(|r| r.ring_type == RingType::Aliphatic).count();
//
//                         ui.label(format!("Hvy: {} Het: {} Rot: {} Net q: {:.2} Abs Q: {:.2} \
//                      Sp3 C: {}, frac_csp3: {:.2} TPSA (Ertl): {:.2} TPSA (topo): {:.2} calc_log_p: {:.2}, m_r: {:.2} val elecs: {}, balaban: {:.2}, Bertz: {:.2} \
//                      Rings sat: {}  Rings ar: {} Rings ali: {} ASA: {}",
//                                          char.num_heavy_atoms,
//                                          char.num_hetero_atoms,
//                                          char.rotatable_bonds.len(),
//                                          char.net_partial_charge.unwrap_or(0.),
//                                          char.abs_partial_charge_sum.unwrap_or(0.),
//                                          char.num_sp3_carbon,
//                                          char.frac_csp3,
//                                          char.tpsa_ertl,
//                                          char.tpsa_topo,
//                                          char.calc_log_p,
//                                          char.molar_refractivity,
//                                          char.num_valence_elecs,
//                                          char.balaban_j,
//                                          char.bertz_ct,
//                                          rings_sat,
//                                          rings_ar,
//                                          rings_ali,
//                                          char.labute_asa
//                         ));
//                     });
//                     ui.horizontal(|ui| {
//                        let mut ident_text = String::new();
//                         for ident in &mol.idents {
//                             ident_text.push_str(&format!(" {ident}"));
//                         }
//
//                         ui.label(format!("Mol idents: {ident_text}"));
//                     });
//                 });
//             });
//         }
//     }
// }

fn selection_section(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: DRY with view.
    ui.horizontal_wrapped(|ui| {
        section_box().show(ui, |ui| {
            view_sel_selector(state, redraw, ui, true);

            let help = "Hide all protein atoms not near the selected atom or bond";
            ui.label("Near sel:").on_hover_text(help);
            if ui
                .checkbox(&mut state.ui.show_near_sel_only, "")
                .on_hover_text(help)
                .changed()
            {
                *redraw = true;

                // todo: For now, only allow one of near sel/lig
                if state.ui.show_near_sel_only {
                    state.ui.show_near_lig_only = false
                }
            }

            if state.active_mol().is_some() {
                let help = "Hide all protein atoms not near the ligand (Active small molecule)";
                ui.label("Near lig:").on_hover_text(help);
                if ui
                    .checkbox(&mut state.ui.show_near_lig_only, "")
                    .on_hover_text(help)
                    .changed()
                {
                    *redraw = true;

                    // todo: For now, only allow one of near sel/lig
                    if state.ui.show_near_lig_only {
                        state.ui.show_near_sel_only = false
                    }
                }
            }

            // todo: Slider for how near the sfc?
            let help = "Hide all protein atoms not near the surface of the protein. May assist \
            in visualizing interaction sites in some visualization modes, e.g. sticks or ball and stick.";
            ui.label("Near sfc:").on_hover_text(help);
            if ui
                .checkbox(&mut state.ui.show_near_sfc_only, "")
                .on_hover_text(help)
                .changed()
            {
                *redraw = true;
            }

            ui.label("pH:");
            if ui
                .add_sized(
                    [20., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.ph_input),
                )
                .changed()
            {
                if let Ok(v) = &mut state.ui.ph_input.parse::<f32>() {
                    state.to_save.ph = *v;

                    // Re-assign hydrogens, and redraw
                    if let Some(mol) = &mut state.peptide {
                        if let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map {
                            match mol.reassign_hydrogens(state.to_save.ph, ff_map) {
                                Ok(_) => *redraw = true,
                                Err(e) => {
                                    let msg = format!("Error reassigning hydrogens: {e:?}");
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                }
            }

            if state.ui.show_near_sel_only || state.ui.show_near_lig_only || state.ui.show_near_sfc_only {
                ui.label("Dist:");
                let dist_prev = state.ui.nearby_dist_thresh;
                ui.spacing_mut().slider_width = 160.;

                ui.add(Slider::new(
                    &mut state.ui.nearby_dist_thresh,
                    NEARBY_THRESH_MIN..=NEARBY_THRESH_MAX,
                ));

                if state.ui.nearby_dist_thresh != dist_prev {
                    *redraw = true;
                }
            }
        });

        if state.ui.selection != Selection::None {
            section_box().show(ui, |ui| {
                ui.horizontal(|ui| {
                    mol_data::selected_data(
                        &state,
                        &state.ligands,
                        &state.nucleic_acids,
                        &state.lipids,
                        &state.pockets,
                        &state.ui.selection,
                        ui,
                    );
                });
            });
        }
    });
}

fn mol_descrip(mol: &MolGenericRef, ui: &mut Ui) {
    ui.heading(RichText::new(mol.common().ident.clone()).color(Color32::GOLD));

    ui.label(format!("{} atoms", mol.common().atoms.len()));

    if let MolGenericRef::Peptide(m) = mol {
        if let Some(method) = m.experimental_method {
            ui.label(method.to_str_short());
        }
    }

    if let Some(title) = mol.common().metadata.get("_struct.title") {
        // Limit size to prevent UI problems.
        let mut title_abbrev: String = title.chars().take(MAX_TITLE_LEN).collect();

        if title_abbrev.len() != title.len() {
            title_abbrev += "...";

            // Allow hovering to see the full title.
            ui.label(RichText::new(title_abbrev).color(Color32::WHITE).size(12.))
                .on_hover_text(title);
        } else {
            ui.label(RichText::new(title_abbrev).color(Color32::WHITE).size(12.));
        }
    }
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    // We perform init items here that rely on  UI context.
    if !INIT_COMPLETE.swap(true, Ordering::AcqRel) {}

    let mut updates = EngineUpdates::default();

    // Checks each frame; takes action based on time since last save.
    check_prefs_save(state);

    // todo: Trying to set popup color; Not working
    // let mut style = (*ctx.style()).clone();
    // style.visuals.widgets.noninteractive.bg_fill = COLOR_POPUP;
    // ctx.set_style(style);

    let mut redraw = RedrawFlags::default();

    let mut reset_cam = false;

    // For getting DT for certain buttons when held. Does not seem to be the same as the 3D render DT.
    let start = Instant::now();

    sidebar(state, scene, &mut redraw, &mut updates, ctx);

    let out_main_panel = TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(
            state,
            ui,
            &mut updates,
            scene,
        );

        if state.volatile.operating_mode == OperatingMode::MolEditor {
            mol_editor::editor(state, scene, &mut updates, ui);

            if let Err(e) = update_file_dialogs(state, scene, ui, &mut updates) {
                handle_err(&mut state.ui, format!("Problem saving file: {e:?}"));
            }

            state.mol_editor.md_step(&state.dev, &mut scene.entities, &state.ui,
                                     &mut updates, state.volatile.mol_manip.mode, );

            load_popups(state, scene, ui, &mut redraw, &mut reset_cam, &mut updates);

            return;
        }


        ui.horizontal(|ui| {
            let color_settings = if state.ui.popup.show_settings {
                Color32::LIGHT_RED
            } else {
                Color32::GRAY
            };
            if ui
                .button(RichText::new("⚙").color(color_settings))
                .clicked()
            {
                state.ui.popup.show_settings = !state.ui.popup.show_settings;
            }

            if ui.button(RichText::new("Editor").color(COLOR_HIGHLIGHT)).clicked() {
                enter_edit_mode(state, scene, &mut updates);
            }

            let metadata_loaded = false; // avoids borrow error.

            {
                let mut close = false;
                display_mol_data_peptide(state, scene, ui, &mut redraw.peptide, &mut redraw.ligand, &mut close, &mut updates);

                if close {
                    close_peptide(state, scene, &mut updates);
                }
            }

            let mut dm_loaded = None; // avoids a double-borrow error.
            if let Some(mol) = &mut state.peptide {

                // todo: Move these A/R. LIkely in a sub menu.
                if let Some(files_avail) = &mol.rcsb_files_avail {
                    // if files_avail.structure_factors {
                    //     if ui
                    //         .button(RichText::new("SF").color(COLOR_HIGHLIGHT))
                    //         .clicked()
                    //     {
                    //         match rcsb::load_structure_factors_cif(&mol.common.ident) {
                    //             Ok(data) => {
                    //                 // println!("SF data: {:?}", data);
                    //             }
                    //             Err(_) => {
                    //                 let msg = format!(
                    //                     "Error loading RCSB structure factors for {:?}",
                    //                     &mol.common.ident
                    //                 );
                    //
                    //                 handle_err(&mut state.ui, msg);
                    //             }
                    //         }
                    //     }
                    // }
                    if files_avail.validation_2fo_fc {
                        if ui
                            .button(RichText::new("Fetch elec ρ").color(COLOR_HIGHLIGHT))
                            .on_hover_text("Load 2fo-fc electron density data from RCSB PDB. Convert to CCP4 map format and display.")
                            .clicked()
                        {
                            // todo: For now, we rely on Gemmi being available on the Path.
                            // todo: We will eventually get our own reflections loader working.

                            match density_from_2fo_fc_rcsb_gemmi(&mol.common.ident, gemmi_path()) {
                                Ok(dm) => {
                                    dm_loaded = Some(dm);
                                    handle_success(&mut state.ui, "Loaded density data from RSCB".to_owned());
                                }
                                Err(e) => {
                                    let msg = format!(
                                        "Error loading or processing RCSB 2fo-fc map for {:?}: {e:?}",
                                        &mol.common.ident
                                    );
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                    // todo: Add these if you end up with a way to use them. We currently use 2fo-fc only.
                    // if files_avail.validation_fo_fc {
                    //     if ui
                    //         .button(RichText::new("fo-fc").color(COLOR_HIGHLIGHT))
                    //         .clicked()
                    //     {
                    //         match rcsb::load_validation_fo_fc_cif(&mol.common.ident) {
                    //             Ok(data) => {
                    //                 // println!("SF data: {:?}", data);
                    //             }
                    //             Err(_) => {
                    //                 let msg = format!(
                    //                     "Error loading RCSB fo-fc map for {:?}",
                    //                     &mol.common.ident
                    //                 );
                    //                 handle_err(&mut state.ui, msg);
                    //             }
                    //         }
                    //     }
                    // }
                    //
                    // if files_avail.validation {
                    //     if ui
                    //         .button(RichText::new("Val").color(COLOR_HIGHLIGHT))
                    //         .clicked()
                    //     {
                    //         match rcsb::load_validation_cif(&mol.common.ident) {
                    //             Ok(data) => {
                    //                 // println!("VAL DATA: {:?}", data);
                    //             }
                    //             Err(_) => {
                    //                 let msg = format!(
                    //                     "Error loading RCSB validation for {:?}",
                    //                     &mol.common.ident
                    //                 );
                    //                 handle_err(&mut state.ui, msg);
                    //             }
                    //         }
                    //     }
                    // }

                    if files_avail.map {
                        if ui
                            .button(RichText::new("Map").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            match rcsb::load_map(&mol.common.ident) {
                                Ok(data) => {
                                    let mut cursor = Cursor::new(data);
                                    match DensityMap::open(&mut cursor) {
                                        Ok(dm) => {
                                            dm_loaded = Some(dm);
                                            println!("Succsesfully loaded Map rom RSCB.");
                                        }
                                        Err(_) => {
                                            let msg = format!(
                                                "Error loading RCSB Map for {:?}",
                                                &mol.common.ident
                                            );
                                            handle_err(&mut state.ui, msg);
                                        }
                                    }
                                }
                                Err(_) => {
                                    let msg =
                                        format!("Error loading RCSB Map for {:?}", &mol.common.ident);
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                }
            }

            if let Some(dm) = dm_loaded {
                state.load_density(dm);
            }


            if metadata_loaded {
                state.update_save_prefs(false);
            }

            ui.add_space(COL_SPACING);

            let color_open_tools = if state.peptide.is_none()
                && state.ligands.is_empty() {
                COLOR_ACTION
            } else {
                COLOR_INACTIVE
            };

            let query_help = "Download and view a molecule from RCSB PDB, PubChem, DrugBank, or Amber Geostd";
            ui.label(RichText::new("Query:").color(color_open_tools))
                .on_hover_text(query_help);

            let edit_resp = ui
                .add(TextEdit::singleline(&mut state.ui.db_input).desired_width(80.))
                .on_hover_text(query_help);

            let inp_lower = state.ui.db_input.to_lowercase();

            let mut enter_pressed = false;
            if state.ui.db_input.len() >= 3 {
                enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
            }

            if state.ui.db_input.len() == 4 || inp_lower.starts_with("pdb_") {
                let button_clicked = ui.button("Load RCSB").clicked();

                if (button_clicked || enter_pressed) && state.ui.db_input.trim().len() == 4 {
                    let ident = state.ui.db_input.clone().trim().to_owned();

                    load_atom_coords_rcsb(
                        &ident,
                        state,
                        scene,
                        &mut updates,
                        &mut redraw.peptide,
                        &mut reset_cam,
                    );

                    state.ui.db_input = String::new();
                }
            } else if state.ui.db_input.len() == 3 {
                let button_clicked = ui.button("Load Geostd").clicked();

                if button_clicked || enter_pressed {
                    let db_input = &state.ui.db_input.clone(); // Avoids a double borrow.
                    state.load_geostd_mol_data(&db_input, true, true, &mut updates, scene);

                    state.ui.db_input = String::new();
                }
            } else if state.ui.db_input.len() > 4 && inp_lower.starts_with("db") {
                let button_clicked = ui.button("Load DrugBank").clicked();

                if button_clicked || enter_pressed {
                    match load_sdf_drugbank(&state.ui.db_input) {
                        Ok(mol) => {
                            open_lig_from_input(state, mol, scene, &mut updates);
                            redraw.ligand = true;
                            // reset_cam = true;
                        }
                        Err(e) => {
                            let msg = format!("Error loading SDF file: {e:?}");
                            handle_err(&mut state.ui, msg);
                        }
                    }
                }
            }

            if let Ok(cid) = state.ui.db_input.parse::<u32>() {
                let button_clicked = ui.button("Load PubChem").clicked();
                if button_clicked || enter_pressed {
                    match load_sdf_pubchem(cid) {
                        Ok(mol) => {
                            open_lig_from_input(state, mol, scene, &mut updates);
                            redraw.ligand = true;
                            // reset_cam = true;
                        }
                        Err(e) => {
                            let msg = format!("Error loading SDF file: {e:?}");
                            handle_err(&mut state.ui, msg);
                        }
                    }
                }
            } else if state.ui.db_input.len() >= 5 && !inp_lower.starts_with("pdb_") && !inp_lower.starts_with("db") {
                let button_clicked = ui.button("Search PubChem").clicked();
                if button_clicked || enter_pressed {
                    let cids = find_cids_from_search(&state.ui.db_input.trim(), false);

                    match cids {
                        Ok(c) => {
                            if c.is_empty() {
                                handle_success(&mut state.ui, "No results found on Pubchem".to_owned());
                            } else {
                                // todo: DRY with the other pubchem branch above.
                                match load_sdf_pubchem(c[0]) {
                                    Ok(mol) => {
                                        open_lig_from_input(state, mol, scene, &mut updates);
                                        redraw.ligand = true;
                                        // reset_cam = true;


                                        let cids_str = c
                                            .iter()
                                            .map(u32::to_string)
                                            .collect::<Vec<_>>()
                                            .join(", ");

                                        handle_success(&mut state.ui, format!("Found the following Pubchem CIDs: {cids_str}. Loaded {}", c[0]));
                                    }
                                    Err(e) => {
                                        let msg = format!("Error loading SDF file: {e:?}");
                                        handle_err(&mut state.ui, msg);
                                    }
                                }
                            }
                        }
                        Err(e) => handle_err(&mut state.ui, format!("Error finding a mol from Pubchem {:?}", e)),
                    }
                }
            }

            if state.peptide.is_none() && state.active_mol().is_none() {
                ui.add_space(COL_SPACING / 2.);
                if ui
                    .button(RichText::new("I'm feeling lucky 🍀").color(color_open_tools))
                    .on_hover_text("Open a random recently-uploaded protein from RCSB PDB.")
                    .clicked()
                {
                    match rcsb::get_newly_released() {
                        Ok(ident) => {
                            load_atom_coords_rcsb(
                                &ident,
                                state,
                                scene,
                                &mut updates,
                                &mut redraw.peptide,
                                &mut reset_cam,
                            );
                        }
                        Err(e) => handle_err(&mut state.ui, format!("Error loading a protein from RCSB: {e:?}"))
                    }
                }
            }
        });

        let close_active_mol = false; // to avoid borrow error.

        ui.horizontal(|ui| {
            section_box().show(ui, |ui| {
                if !state.ligands.is_empty() || !state.lipids.is_empty() || !state.nucleic_acids.is_empty() || !state.pockets.is_empty() {
                    display_mol_data(state, ui);
                }

                if state.ligands.len() >= 2 && ui.button("Align")
                    .on_hover_text("Perform flexible alignment on two opened small molecules. This button opens a \
                    window which lets you configure and run this alignment, then view the resulting 3D conformations,\
                    and similarity metrics.")
                    .clicked() {
                    state.ui.popup.alignment = !state.ui.popup.alignment;

                    if state.volatile.alignment.mols_to_align.len() < 2 {
                        state.volatile.alignment.mols_to_align = vec![0, 1];
                    }
                }

                if ui.button("Alignment screen")
                    .on_hover_text("Perform a fast small molecule alignment screening from all \
                    files in a selected folder").clicked() {
                    state.ui.popup.alignment_screening = !state.ui.popup.alignment_screening;
                }
            });
        });
        // Prevents the UI from jumping.
        if state.volatile.active_mol.is_none() {
            ui.add_space(3.);
        }

        let redraw_prev = redraw.peptide;
        selection_section(state, &mut redraw.peptide, ui);

        // todo: Kludge
        if redraw.peptide && !redraw_prev {
            redraw.set_all();
        }

        ui.horizontal_wrapped(|ui| {
            cam_controls(scene, state, &mut updates, ui);
            cam_snapshots(state, scene, &mut updates, ui);
        });

        ui.horizontal(|ui| {
            view_settings(state, scene, &mut updates, &mut redraw, ui);

            ui.add_space(COL_SPACING);

            section_box().show(ui, |ui| {
                ui_section_vis(state, ui);
            });
        });

        chain_selector(state, &mut redraw.peptide, ui);

        mol_type_toolbars(state, scene, &mut updates, ui);

        if state.ui.ui_vis.dynamics {
            md_setup(state, scene, &mut updates, ui);
        }
        if state.ui.ui_vis.orca {
            orca_input(state, &mut redraw.ligand, ui);
        }

        // if state.ui.show_docking_tools {
        //     ui.add_space(ROW_SPACING);
        //
        //     docking(state, scene, &mut redraw.ligand, &mut engine_updates, ui);
        // }

        // todo: Allow switching between chains and secondary-structure features here.

        ui.add_space(ROW_SPACING / 2.);

        if state.ui.ui_vis.aa_seq {
            if state.peptide.is_some() {
                add_aa_seq(&mut state.ui.selection, &state.volatile.aa_seq_text, ui, &mut redraw.peptide);
            }
        }

        if state.ui.ui_vis.smiles {
            if let Some(mol) = &state.active_mol() &&
                let MolGenericRef::Small(m) = mol {
                for ident in &m.idents {
                    if let MolIdent::Smiles(smiles) = ident {
                        draw_smiles(smiles, ui);
                        break;
                    }
                }
            }
        }

        draw_cli(
            state,
            scene,
            &mut updates,
            &mut redraw,
            &mut reset_cam,
            ui,
        );

        // todo: Render area here (Dec 2025) for the 3D part?
        // {
        //     let avail = ui.available_size();
        //     let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        //
        //     // Important: attach your WGPU rendering to *this* rect.
        //     // ui.painter().add(egui::Shape::Callback(egui::epaint::PaintCallback {
        //     //     rect,
        //     //     callback: std::sync::Arc::new(MyWgpuCallback {
        //     //         // whatever handles/resources you need
        //     //     }),
        //     // }));
        // }

        load_popups(state, scene, ui, &mut redraw, &mut reset_cam, &mut updates);

        // -------UI above; clean-up items (based on flags) below

        if close_active_mol {
            if let Some((mol_type, i)) = state.volatile.active_mol {
                close_mol(mol_type, i, state, scene, &mut updates);
            }
        }

        if let Err(e) = update_file_dialogs(state, scene, ui, &mut updates) {
            handle_err(&mut state.ui, format!("Problem saving file: {e:?}"));
        }

        handle_redraw(
            state,
            scene, &mut redraw, reset_cam, &mut updates,
        )
    });

    // todo: Experimenting
    updates.ui_reserved_px.1 = out_main_panel.response.rect.height();

    // todo: Appropriate place for this?
    if state.volatile.inputs_commanded.inputs_present() {
        set_flashlight(scene);
        updates.lighting = true;
    }

    handle_scene_flags(state, scene, &mut updates);
    handle_thread_rx(state, scene, &mut updates);

    // Run one or more MD steps, if a MD computation is in progress.
    state.md_step(scene, &mut updates);

    state.ui.dt_render = start.elapsed().as_secs_f32();

    // Without this, no computation will happen while minimized.
    // todo: Not working.
    if state.volatile.md_local.running {
        ctx.request_repaint();
    }

    updates
}

pub fn flag_btn(val: &mut bool, label: &str, hover_text: &str, ui: &mut Ui) {
    let color = if *val { COLOR_ACTIVE } else { COLOR_INACTIVE };
    if ui
        .button(RichText::new(label).color(color))
        .on_hover_text(hover_text)
        .clicked()
    {
        *val = !(*val);
    }
}

fn draw_smiles(v: &str, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("SMILES: ");

        let font_id = TextStyle::Body.resolve(ui.style());

        let mut job = LayoutJob::default();
        for ch in v.chars() {
            let mut color = match Element::from_letter(&ch.to_string()) {
                Ok(e) => {
                    // Lighter color; N is showing too dark.
                    if e == Element::Nitrogen {
                        Color32::from_rgb(110, 110, 255)
                    } else {
                        color_egui_from_f32(e.color())
                    }
                }
                _ => Color32::GRAY,
            };

            if ch.is_ascii_digit() {
                color = Color32::from_rgb(255, 180, 50);
            }

            if ch == '@' {
                color = Color32::from_rgb(150, 170, 255);
            }

            job.append(
                &ch.to_string(),
                0.0,
                TextFormat {
                    font_id: font_id.clone(),
                    color,
                    ..Default::default()
                },
            );
        }

        ui.label(job);

        // for char in v.chars() {
        //     let color = match Element::from_letter(&char.to_string()) {
        //         Ok(e) => {
        //             let (r, g, b) = e.color();
        //             Color32::from_rgb(
        //                 (r * 255.) as u8,
        //                 (g * 255.) as u8,
        //                 (b * 255.) as u8,
        //             )
        //         },
        //         _ => Color32::GRAY
        //     };
        //
        //     ui.label(RichText::new(char).color(color));
        // }
    });
}

pub(crate) fn cam_controls(
    scene: &mut Scene,
    state: &mut State,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Here and at init, set the camera dist dynamically based on mol size.
    // todo: Set the position not relative to 0, but  relative to the center of the atoms.

    let mut changed = false;

    // let cam = &mut scene.camera;

    // This frame allows for a border to visually section this off.

    section_box()
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                cam::cam_reset_controls(state, scene, ui, engine_updates, &mut changed);

                ui.add_space(COL_SPACING);

                let free_active = scene.input_settings.control_scheme == ControlScheme::FreeCamera;
                let arc_active = scene.input_settings.control_scheme != ControlScheme::FreeCamera;

                if ui
                    .button(RichText::new("Free").color(misc::active_color_sel(free_active)))
                    .on_hover_text("Set the camera is a first-person mode, where your controls move its position. Similar to video games.")
                    .clicked()
                {
                    scene.input_settings.control_scheme = ControlScheme::FreeCamera;
                    state.to_save.control_scheme = ControlScheme::FreeCamera;
                }

                if ui
                    .button(RichText::new("Arc").color(misc::active_color_sel(arc_active)))
                    .on_hover_text("Set the camera to orbit around a point: Either the center of the molecule, or the selection.")
                    .clicked()
                {
                    let center = match &state.peptide {
                        Some(mol) => mol.center.into(),
                        None => Vec3::new_zero(),
                    };
                    scene.input_settings.control_scheme = ControlScheme::Arc { center };
                    state.to_save.control_scheme = ControlScheme::Arc { center };
                }

                if arc_active {
                    if ui
                        .button(
                            RichText::new("Orbit sel")
                                .color(misc::active_color(state.ui.orbit_selected_atom)),
                        )
                        .on_hover_text("Toggle whether the camera orbits around the selection, or the molecule center.")
                        .clicked()
                    {
                        state.ui.orbit_selected_atom = !state.ui.orbit_selected_atom;

                        let center = orbit_center(state);
                        scene.input_settings.control_scheme = ControlScheme::Arc { center };
                    }
                }

                ui.add_space(COL_SPACING);

                if state.ui.selection != Selection::None {
                    if ui
                        .button(RichText::new("Cam to sel").color(COLOR_HIGHLIGHT))
                        .on_hover_text("(Hotkey: Enter) Move camera near the selected atom or residue, looking at it.")
                        .clicked()
                    {
                        move_cam_to_sel(&mut state.ui, &state.peptide, &state.ligands, &state.nucleic_acids,
                                        &state.lipids, &state.pockets, &mut scene.camera, engine_updates);
                    }
                }

                // if state.volatile.active_mol.is_some() {
                //     if ui
                //         .button(RichText::new("Cam to mol").color(COLOR_HIGHLIGHT))
                //         .on_hover_text("Move camera near active molecule, looking at it.")
                //         .clicked()
                //     {
                //         let pep_center = match &state.peptide {
                //             Some(mol) => mol.center,
                //             None => lin_alg::f64::Vec3::new_zero(),
                //         };
                //         // Setting mol center to 0 if no mol.
                //         move_cam_to_active_mol(state, scene, pep_center, engine_updates)
                //     }
                // }

                ui.add_space(COL_SPACING);

                // todo: Grey-out, instead of setting render dist. (e.g. fog)
                let depth_prev = state.ui.view_depth;
                ui.spacing_mut().slider_width = 60.;

                let hover_text = "Don't render objects closer to the camera than this distance, in Å.";
                ui.label("Depth. Near(×10):")
                    .on_hover_text(hover_text);

                ui.add(Slider::new(
                    &mut state.ui.view_depth.0,
                    VIEW_DEPTH_NEAR_MIN..=VIEW_DEPTH_NEAR_MAX,
                )).on_hover_text(hover_text);

                let hover_text = "(Hotkey: Ctrl + scroll) Fade distant objects. This may make it easier to see objects near the camera.";
                ui.label("Far:")
                    .on_hover_text(hover_text);

                ui.add(Slider::new(
                    &mut state.ui.view_depth.1,
                    FOG_DIST_MIN..=FOG_DIST_MAX,
                )).on_hover_text(hover_text);

                if state.ui.view_depth != depth_prev {
                    // Interpret the slider being at min or max position to mean (effectively) unlimited.

                    scene.camera.near = if state.ui.view_depth.0 == VIEW_DEPTH_NEAR_MIN {
                        RENDER_DIST_NEAR
                    } else {
                        state.ui.view_depth.0 as f32 / 10.
                    };
                    // todo: Only if near changed.
                    scene.camera.update_proj_mat();

                    cam::set_fog_dist(&mut scene.camera, state.ui.view_depth.1);

                    changed = true;
                }
            });
        });

    if changed {
        engine_updates.camera = true;

        set_flashlight(scene);
        engine_updates.lighting = true; // flashlight.

        state.ui.cam_snapshot = None;
    }
}

pub(crate) fn cam_snapshots(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Wraping isn't working here.
    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Scenes");

            ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(60.))
                .on_hover_text("Choose a name to save this scene as.");

            if ui
                .button("Save")
                .on_hover_text("Save the current camera position and orientation to a scene.")
                .clicked()
            {
                let name = if !state.ui.cam_snapshot_name.is_empty() {
                    state.ui.cam_snapshot_name.clone()
                } else {
                    format!("Scene {}", state.cam_snapshots.len() + 1)
                };

                crate::util::save_snap(state, &scene.camera, &name);
            }

            let prev_snap = state.ui.cam_snapshot;
            let snap_name = get_snap_name(prev_snap, &state.cam_snapshots);

            ComboBox::from_id_salt(2)
                .width(80.)
                .selected_text(snap_name)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.ui.cam_snapshot, None, "(None)");
                    for (i, _snap) in state.cam_snapshots.iter().enumerate() {
                        ui.selectable_value(
                            &mut state.ui.cam_snapshot,
                            Some(i),
                            get_snap_name(Some(i), &state.cam_snapshots),
                        );
                    }
                })
                .response
                .on_hover_text("Set the camera to a previously-saved scene.");

            if let Some(i) = state.ui.cam_snapshot {
                if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
                    if i < state.cam_snapshots.len() {
                        state.cam_snapshots.remove(i);
                    }
                    state.ui.cam_snapshot = None;
                    state.update_save_prefs(false);
                }
            }

            if state.ui.cam_snapshot != prev_snap {
                crate::util::load_snap(state, scene, engine_updates);
            }
        });
    });
}
