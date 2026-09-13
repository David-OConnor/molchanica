//! RFdiffusion3 backbone generation, using the shared bio_tools form and runner.
use egui::Ui;
use graphics::{EngineUpdates, Scene};

use super::{close_btn, tool_runner::ToolWindow};
use crate::{external_tools::Tool, state::State, util::handle_err};

pub(crate) struct Rfd3Ui {
    window: ToolWindow,
}

impl Default for Rfd3Ui {
    fn default() -> Self {
        Self {
            window: ToolWindow::new(Tool::RfDiffusion3),
        }
    }
}

pub(in crate::ui) fn rfdiffusion3_window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.heading("RFdiffusion3 backbone generation");
        close_btn(ui, &mut state.ui.popup.rfd3);
    });

    let load = state
        .ui
        .rfd3
        .window
        .draw(&[Tool::RfDiffusion3], &state.peptides, ui);
    if let Some(path) = load
        && let Err(error) = state.open_file(&path, scene, updates)
    {
        handle_err(&mut state.ui, error.to_string());
    }
}
