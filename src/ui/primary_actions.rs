use egui::{Context, Id, Ui};
use graphics::{EngineUpdates, Scene};

use super::{action_search, open_docs, set_alignment_open, sidebar};
use crate::{cam, mol_editor::enter_edit_mode, selection::Selection, state::State};

pub(super) const DB_QUERY_ID: &str = "primary_db_query";
pub(super) const FIND_ID: &str = "primary_find_in_molecule";
const FOCUS_REQUEST_ID: &str = "primary_action_focus_request";

type Run = fn(&mut State, &mut Scene, &mut EngineUpdates, &Context);

pub(super) fn show(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates, ui: &mut Ui) {
    // egui keeps the modal's input layer active for one frame after closing.
    let request_id = Id::new(FOCUS_REQUEST_ID);
    if let Some(field) = ui.data(|data| data.get_temp::<Id>(request_id)) {
        if ui.memory(|mem| mem.top_modal_layer().is_none()) {
            ui.data_mut(|data| data.remove::<Id>(request_id));
            ui.memory_mut(|mem| mem.request_focus(field));
        } else {
            ui.request_repaint();
        }
    }

    let chosen = action_search::show(ui, Id::new("primary_palette"), "Actions", || {
        commands(state)
    });
    if let Some(run) = chosen {
        run(state, scene, updates, ui.ctx());
    }
}

fn focus_after_close(ctx: &Context, field: &str) {
    ctx.data_mut(|data| data.insert_temp(Id::new(FOCUS_REQUEST_ID), Id::new(field)));
}

fn commands(state: &State) -> Vec<action_search::Command<Run>> {
    let actions: &[(&str, Run)] = &[
        ("Open molecule or other file", |state, _, _, _| {
            state.volatile.dialogs.load.pick_file();
        }),
        ("Open recent files", |state, _, _, _| {
            state.ui.popup.recent_files = true;
        }),
        ("Open project", |state, _, _, _| {
            state.volatile.dialogs.project_load.pick_file();
        }),
        ("Save project", |state, _, _, _| {
            state.volatile.dialogs.project_save.save_file();
        }),
        ("Settings", |state, _, _, _| {
            state.ui.popup.show_settings = true;
        }),
        ("Molecule editor", |state, scene, updates, _| {
            enter_edit_mode(state, scene, updates);
        }),
        ("Force field parameters (FF params)", |state, _, _, _| {
            state.ui.popup.ff_params = true;
        }),
        ("Molecule databases (Mol DBs)", |state, _, _, _| {
            state.ui.popup.parquet_db = true;
        }),
        ("Protein design tools", |state, _, _, _| {
            state.ui.popup.protein_tools = true;
        }),
        ("Selections and representation layers", |state, _, _, _| {
            state.ui.popup.selections = true;
        }),
        ("Measurements and colors", |state, _, _, _| {
            state.ui.popup.measurement = true;
        }),
        ("Residue management", |state, _, _, _| {
            state.ui.popup.lig_pocket_creation = true;
        }),
        ("Publication figure", |state, _, _, _| {
            state.publication.open = true;
        }),
        ("Alignment screening", |state, _, _, _| {
            state.ui.popup.alignment_screening = true;
        }),
        ("Third-party tools", |state, _, _, _| {
            state.ui.popup.external_tools = true;
        }),
        ("Command files / macros", |state, _, _, _| {
            state.ui.command_files.open = true;
        }),
        ("Show molecular dynamics (MD)", |state, _, _, _| {
            state.ui.ui_vis.dynamics = true;
        }),
        ("Show ORCA", |state, _, _, _| {
            state.ui.ui_vis.orca = true;
        }),
        ("Show lipid creation", |state, _, _, _| {
            state.ui.ui_vis.lipids = true;
        }),
        (
            "Show nucleic acid creation (DNA / RNA)",
            |state, _, _, _| {
                state.ui.ui_vis.nucleic_acids = true;
            },
        ),
        ("Show amino acid creation", |state, _, _, _| {
            state.ui.ui_vis.amino_acids = true;
        }),
        ("Show amino acid selection", |state, _, _, _| {
            state.ui.ui_vis.aa_selection = true;
        }),
        ("Toggle sidebar", |state, _, _, _| {
            state.ui.ui_vis.sidebar = !state.ui.ui_vis.sidebar;
        }),
        ("Find atoms or residues", |_, _, _, ctx| {
            focus_after_close(ctx, FIND_ID);
        }),
        ("About Molchanica", |state, _, _, _| {
            state.ui.popup.about = true;
        }),
        ("Open documentation", |_, _, _, _| open_docs()),
    ];

    let mut commands: Vec<action_search::Command<Run>> = actions
        .iter()
        .map(|&(label, run)| (run, label, "", true))
        .collect();
    let contextual: [action_search::Command<Run>; 6] = [
        (
            |state, _, _, _| sidebar::save_active(state),
            "Save active molecule or sequence",
            "",
            sidebar::can_save_active(state),
        ),
        (
            |state, _, _, _| state.ui.popup.protein_comparison = true,
            "Compare and align proteins",
            "",
            state.peptides.len() >= 2,
        ),
        (
            |state, _, _, _| set_alignment_open(state, true),
            "Align small molecules",
            "",
            state.ligands.len() >= 2,
        ),
        (
            |state, _, _, _| state.ui.ui_vis.aa_seq = true,
            "Show protein sequence",
            "",
            !state.peptides.is_empty(),
        ),
        (
            |_, _, _, ctx| {
                focus_after_close(ctx, DB_QUERY_ID);
            },
            "Query online molecule databases",
            "",
            state.volatile.thread_receivers.query.is_none(),
        ),
        (
            |state, scene, updates, _| {
                cam::move_cam_to_sel(state, &mut scene.camera, updates);
            },
            "Camera to selection or active molecule",
            "Enter",
            state.ui.selection != Selection::None || state.active_mol().is_some(),
        ),
    ];
    commands.extend(contextual);

    #[cfg(feature = "synthesis")]
    commands.push((
        |state, _, _, _| super::set_synthesis_reactions_open(state, true),
        "Synthesis reaction library",
        "",
        true,
    ));

    commands
}
