//! Misc utility-related UI functionality.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Slider, Stroke, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
const COLOR_SECTION_BOX: Color32 = Color32::from_rgb(100, 100, 140);

use crate::{
    md,
    state::State,
    ui::{COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE, ROW_SPACING, handle_input},
    util::{RedrawFlags, handle_err},
};

/// A box that shows its text highlighted if a flag is set.
pub fn toggle_btn_inv(
    val: &mut bool,
    text: &str,
    tooltip: &str,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    let color = active_color(!*val);
    if ui
        .button(RichText::new(text).color(color))
        .on_hover_text(tooltip)
        .clicked()
    {
        *val = !*val;
        // todo: Don't need to redraw everything.
        redraw.set_all();
    }
}

/// A box that shows its text highlighted if a flag is set.
pub fn toggle_btn(
    val: &mut bool,
    text: &str,
    tooltip: &str,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    let color = active_color(*val);
    if ui
        .button(RichText::new(text).color(color))
        .on_hover_text(tooltip)
        .clicked()
    {
        *val = !*val;
        // todo: Don't need to redraw everything.
        redraw.set_all();
    }
}

// #[derive(Clone, Copy, PartialEq)]
// pub enum MdMode {
//     Docking,
//     Peptide,
// }

pub fn active_color(val: bool) -> Color32 {
    if val { COLOR_ACTIVE } else { COLOR_INACTIVE }
}

/// Visually distinct; fore buttons that operate as radio buttons
pub fn active_color_sel(val: bool) -> Color32 {
    if val {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    }
}

pub fn dynamics_player(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.volatile.md_local.mol_dynamics.is_none() {
        return;
    }

    ui.horizontal(|ui| {
        // let prev = state.ui.peptide_atom_posits;
        let help_text = "Toggle between viewing the original (pre-dynamics) atom positions, and \
        ones at the selected dynamics snapshot.";
        ui.label("Show atoms:").on_hover_text(help_text);

        let snapshot_prev = state.ui.current_snapshot;

        if let Some(md) = &state.volatile.md_local.mol_dynamics {
            if !md.snapshots.is_empty() {
                ui.add_space(ROW_SPACING);

                ui.spacing_mut().slider_width = ui.available_width() - 100.;
                ui.add(Slider::new(
                    &mut state.ui.current_snapshot,
                    0..=md.snapshots.len() - 1,
                ));
                ui.label(format!(
                    "{:.2} ps",
                    state.ui.current_snapshot as f32 * state.to_save.md_dt
                ));
            }

            if state.ui.current_snapshot != snapshot_prev {
                if let Err(e) = state
                    .volatile
                    .md_local
                    .change_snapshot(state.ui.current_snapshot)
                {
                    handle_err(&mut state.ui, format!("Error changing snapshot: {e:?}"));
                }

                md::draw_mols(state, scene);
                engine_updates.entities = EntityUpdate::All;
            }
        };
    });
}

// A container that highlights a section of UI code, to make it visually distinct from neighboring areas.
pub fn section_box() -> Frame {
    Frame::new()
        .stroke(Stroke::new(1.0, COLOR_SECTION_BOX))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 2))
        .outer_margin(Margin::symmetric(0, 0))
}
