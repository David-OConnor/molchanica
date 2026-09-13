//! Structure prediction and co-folding, using the shared bio_tools form and runner.
use egui::Ui;
use graphics::{EngineUpdates, Scene};

use super::{close_btn, tool_runner::ToolWindow};
use crate::{external_tools::Tool, state::State, util::handle_err};

pub(crate) struct StructurePredUi {
    window: ToolWindow,
    message: Option<String>,
}

impl Default for StructurePredUi {
    fn default() -> Self {
        Self {
            window: ToolWindow::new(Tool::OpenDde),
            message: None,
        }
    }
}

impl StructurePredUi {
    // Sequence convenience actions still report completion through the main worker queue.
    pub(crate) fn finish_prediction(&mut self) {}
    pub(crate) fn mark_complete(&mut self, message: String) {
        self.message = Some(message);
    }
}

pub(in crate::ui) fn structure_prediction_window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.heading("Structure prediction and co-folding");
        close_btn(ui, &mut state.ui.popup.structure_pred);
    });
    if let Some(message) = &state.ui.structure_pred.message {
        ui.label(message);
    }
    let load = state.ui.structure_pred.window.draw(
        &[Tool::OpenDde, Tool::Boltz2, Tool::Chai1, Tool::EsmFold2],
        &state.peptides,
        ui,
    );
    if let Some(path) = load
        && let Err(error) = state.open_file(&path, scene, updates)
    {
        handle_err(&mut state.ui, error.to_string());
    }
}
