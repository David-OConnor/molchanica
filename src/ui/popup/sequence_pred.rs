//! Sequence design and scoring, using the shared bio_tools form and runner.
use egui::Ui;
use graphics::{EngineUpdates, Scene};

use super::{close_btn, tool_runner::ToolWindow};
use crate::{external_tools::Tool, state::State, util::handle_err};

pub(crate) struct SequencePredUi {
    window: ToolWindow,
}

impl Default for SequencePredUi {
    fn default() -> Self {
        Self {
            window: ToolWindow::new(Tool::ProteinMpnn),
        }
    }
}

pub(in crate::ui) fn sequence_prediction_window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.heading("Sequence design and scoring");
        close_btn(ui, &mut state.ui.popup.sequence_pred);
    });

    let load = state.ui.sequence_pred.window.draw(
        &[Tool::ProteinMpnn, Tool::LigandMpnn],
        &state.peptides,
        ui,
    );
    if let Some(path) = load
        && let Err(error) = state.open_file(&path, scene, updates)
    {
        handle_err(&mut state.ui, error.to_string());
    }
}
