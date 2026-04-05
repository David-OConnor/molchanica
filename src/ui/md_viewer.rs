use dynamics::snapshot::Snapshot;
use egui::{Color32, RichText, Slider, Ui};
use graphics::{EngineUpdates, FWD_VEC, Scene};

use crate::cam::reset_camera;
use crate::ui::popups::close_btn;
use crate::ui::{COLOR_ACTION, COLOR_ACTIVE, COLOR_INACTIVE};
use crate::util::handle_success;
use crate::{
    button, label,
    md::viewer,
    state::State,
    ui::{COL_SPACING, COLOR_HIGHLIGHT, ROW_SPACING},
    util::{RedrawFlags, handle_err},
};

pub(in crate::ui) fn dynamics_viewer(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        // let prev = state.ui.peptide_atom_posits;
        let help_text = "Toggle between viewing the original (pre-dynamics) atom positions, and \
        ones at the selected dynamics snapshot.";

        let ready = state.volatile.md_local.viewer.mols_and_traj_synced();

        {
            let text = if state.volatile.md_local.draw_md_mols {
                "Draw original"
            } else {
                "Draw MD"
            };

            if (state.volatile.md_local.draw_md_mols
                || (!state.volatile.md_local.viewer.snapshots.is_empty() && ready))
                && ui
                    .button(RichText::new(text).color(COLOR_HIGHLIGHT))
                    .on_hover_text(help_text)
                    .clicked()
            {
                state.volatile.md_local.draw_md_mols = !state.volatile.md_local.draw_md_mols;

                viewer::draw_mols(state, scene, updates);
            }
        }

        if !state.volatile.md_local.viewer.snapshots.is_empty()
            && ui
                .button(RichText::new("Clear Traj"))
                .on_hover_text("Clear all trajectory snapshots, e.g. erase the previous run.")
                .clicked()
        {
            state
                .volatile
                .md_local
                .clear_snaps(&mut scene.entities, updates, redraw);

            // todo: Make this call part of clear_snaps?
            for traj in &mut state.trajectories {
                traj.frames_open = None;
            }
        }

        if !ready {
            let txt = if state.volatile.md_local.viewer.snapshots.is_empty() {
                "Not ready for MD playback; no snapshots/frames loaded."
            } else {
                "Not ready for MD playback; trajectory and mol set atom count mismatch."
            };

            label!(ui, txt, Color32::GRAY);

            return;
        }

        let slider_posit_prev = state.volatile.md_local.viewer.slider_posit_ui;

        if !state.volatile.md_local.viewer.snapshots.is_empty() {
            ui.add_space(ROW_SPACING);

            ui.spacing_mut().slider_width = ui.available_width() - 100.;

            ui.add(Slider::new(
                &mut state.volatile.md_local.viewer.slider_posit_ui,
                0..=state.volatile.md_local.viewer.snapshots.len() - 1,
            ));

            if let Some(ss) = &state.volatile.md_local.viewer.get_active_snap() {
                ui.label(format!("{:.2} ps", ss.time));
            }

            let posit = state.volatile.md_local.viewer.slider_posit_ui;

            if posit != slider_posit_prev && posit <= state.volatile.md_local.viewer.snapshots.len()
            {
                state.volatile.md_local.viewer.current_snapshot = Some(posit);

                if let Err(e) = state.volatile.md_local.viewer.change_snapshot(posit) {
                    handle_err(&mut state.ui, format!("Error changing MD snapshot: {e:?}"));
                }

                viewer::draw_mols(state, scene, updates);
            }
        }
    });
}

pub(in crate::ui) fn energy_disp(snap: &Snapshot, ui: &mut Ui) {
    let Some(en) = &snap.energy_data else { return };

    ui.label(RichText::new("Energy (current snap):").color(Color32::WHITE));
    ui.add_space(COL_SPACING / 2.);

    ui.label("E (kcal/mol) KE: ");
    ui.label(RichText::new(format!("{:.1}", en.energy_kinetic)).color(Color32::GOLD));

    ui.label("KE/atom: ");
    // todo: Don't continuously run these computations!
    let atom_count = (snap.water_o_posits.len() * 3) as f32 + snap.atom_posits.len() as f32;
    let ke_per_atom = en.energy_kinetic / atom_count;
    label!(ui, format!("{:.2}", ke_per_atom), Color32::GOLD);

    ui.label("PE: ");
    label!(ui, format!("{:.2}", en.energy_potential), Color32::GOLD);

    if en.energy_potential_nonbonded.abs() > 0.0001 {
        ui.label("PE NB: ");
        label!(
            ui,
            format!("{:.2}", en.energy_potential_nonbonded),
            Color32::GOLD
        );
    }

    ui.label("PE/atom: ");
    // todo: Don't continuously run this!
    let pe_per_atom = en.energy_potential / atom_count;
    label!(ui, format!("{:.3}", pe_per_atom), Color32::GOLD);

    ui.label("E tot: ");
    // todo: Don't continuously run this!
    let e = en.energy_potential + en.energy_kinetic;
    label!(ui, format!("{:.2}", e), Color32::GOLD);

    // ui.label("PE between mols:");
    // // todo: One pair only for now
    // if en.energy_potential_between_mols.len() >= 2 {
    //     // todo: Which index?
    //     ui.label(
    //         RichText::new(format!("{:.2}", en.energy_potential_between_mols[1]))
    //             .color(Color32::GOLD),
    //     );
    // }

    ui.label("Temp: ");
    label!(ui, format!("{:.1} K", en.temperature), Color32::GOLD);

    ui.label("P: ");
    label!(ui, format!("{:.1} bar", en.pressure), Color32::GOLD);

    ui.label("Vol: ");
    label!(
        ui,
        format!("{:.1} k Å^3", en.volume / 1_000.),
        Color32::GOLD
    );

    ui.label("Dens: ");
    label!(ui, format!("{:.2} amu/Å^3", en.density), Color32::GOLD);
}

/// Create and edit MD molecule sets; these are associated with trajectories; they map a trajectory's
/// flat atom index to molecules, for display purposes.
///
/// Sets are stored in the .gro format; GROMACS's text-based format for this. This window lets you create
/// new sets, and edit existing ones.
pub(in crate::ui) fn md_mol_set_editor(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        label!(
            ui,
            "Edit and add molecule sets for MD playback",
            Color32::WHITE
        );
        close_btn(ui, &mut state.ui.popup.md_mol_set_editor);
    });
    label!(
        ui,
        "To open a set (e.g. .gro) format, open it like you would any other file format",
        Color32::GRAY
    );

    if !state.volatile.md_local.viewer.mol_sets.is_empty() {
        for (i, set) in state.volatile.md_local.viewer.mol_sets.iter().enumerate() {
            let active = Some(i) == state.ui.md.set_editor_active_set;
            let color = if active { COLOR_ACTIVE } else { COLOR_INACTIVE };

            if button!(ui, "Set", color, "Toggle if this is the active set.").clicked() {
                state.ui.md.set_editor_active_set = if active { None } else { Some(i) };
            }
        }
    } else {
        label!(ui, "(No sets open)", Color32::GRAY);
    }

    match state.ui.md.set_editor_active_set {
        Some(i) => {}
        None => {}
    }

    ui.add_space(ROW_SPACING);
}

/// Selected from loaded molecule maps, which map to atrajectory atoms. This might be loaded, for
/// example, from a .gro file.
pub(in crate::ui) fn md_viewer_mappings(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    let empty = state.volatile.md_local.viewer.mol_sets.is_empty();
    if empty && !state.trajectories.is_empty() {
        return;
    }

    ui.label("MD mol sets");
    ui.separator();

    // We show this if sets are empty, but trajectories are not.
    if button!(
        ui,
        "Edit mol sets",
        Color32::WHITE,
        "Add or edit molecule sets associated with MD trajectories. This lets you add open molecules,\
        and adjust which trajectory index range each is associated with."
    ).clicked() {
        state.ui.popup.md_mol_set_editor = !state.ui.popup.md_mol_set_editor;
    }

    if empty {
        return;
    }

    let mut close = None;
    let mut set_clicked = false;

    for (i, set) in state.volatile.md_local.viewer.mol_sets.iter().enumerate() {
        ui.horizontal(|ui| {

            let (active, color) = if Some(i) == state.volatile.md_local.viewer.mol_set_active {
                (true, COLOR_ACTIVE)
            } else {
                (false, Color32::WHITE)
            };

            ui.label(RichText::new(format!("{} | {} mols", set.name, set.mols.len())).color(color));

            ui.label(RichText::new(format!("At: {} | Rng: {}-{}", set.atom_count, set.range_covered.0, set.range_covered.1)).color(color));

            if set.range_overlaps {
                ui.label(RichText::new("Warning: Mol ranges overlap").color(Color32::LIGHT_RED));
            }

            if button!(
            ui,
            "Set",
            COLOR_ACTION,
            "Load this set of molecules into the MD trajectory atoms. This affects \
        how the atoms in the trajectory are visually mapped to molecules with covalent bonds, the correct element etc."
        ).clicked() {
                state.volatile.md_local.viewer.mol_set_active = if active {
                    None }
                else {
                    Some(i)
                };

                set_clicked = true;
                handle_success(&mut state.ui, format!("Set {} as the active mol set", set.name));
            }

            if button!(ui, "Save", COLOR_ACTION, "Save this mol set as a GRO file.").clicked() {
                let name = format!("{}.gro", set.name.replace(' ', "_"));
                state.volatile.dialogs.save_gro.config_mut().default_file_name = name;
                state.volatile.dialogs.save_gro_mol_set_i = Some(i);
                state.volatile.dialogs.save_gro.save_file();
            }

            if ui
                .button(RichText::new("❌").color(Color32::LIGHT_RED))
                .on_hover_text("Close this molecule set.")
                .clicked()
            {
                close = Some(i);
            }

        });

        let max_count = 5;
        for mol in set.mols.iter().take(max_count) {
            // todo: Consider a total solvent count, and group those together
            label!(
                ui,
                format!(
                    "{} | Atoms: {} Range: {}-{}",
                    mol.mol.ident,
                    mol.mol.atoms.len(),
                    mol.range.0,
                    mol.range.1,
                ),
                Color32::WHITE
            );
        }
        if set.mols.len() >= max_count {
            label!(ui, "...", Color32::WHITE);
        }
    }

    ui.separator();

    if let Some(i) = close {
        state
            .volatile
            .md_local
            .viewer
            .close_mol_set(&mut state.to_save.open_history, i);
    }

    if set_clicked {
        // We have this as the function calls in this branch which call state have a borrow
        // error otherwise; the flag setting is convenience.
        reset_camera(state, scene, updates, FWD_VEC);
        viewer::draw_mols(state, scene, updates);

        redraw.set_all();

        if state.volatile.md_local.viewer.change_snapshot(0).is_err() {
            handle_err(&mut state.ui, "Error changing snaps".to_string());
        }
    }
}
