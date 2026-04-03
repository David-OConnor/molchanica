use dynamics::snapshot::Snapshot;
use egui::{Color32, RichText, Slider, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};

use crate::{
    label,
    md::viewer,
    state::State,
    ui::{COL_SPACING, COLOR_HIGHLIGHT, ROW_SPACING},
    util::handle_err,
};

pub(in crate::ui) fn dynamics_viewer(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
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

                viewer::draw_mols(state, scene);
                updates.entities = EntityUpdate::All;
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

                viewer::draw_mols(state, scene);
                updates.entities = EntityUpdate::All;
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
