//! Optional toolbars for nucleic acids, lipids etc.

use egui::Ui;
use graphics::{EngineUpdates, Scene};

use crate::{
    state::State,
    ui::{
        misc::wrapped_section,
        panels::{aa_creation::aa_section, lipid_creation, na_creation},
    },
};

pub(in crate::ui) fn mol_type_toolbars(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal_wrapped(|ui| {
        if state.ui.ui_vis.lipids {
            wrapped_section(ui, "lipids", |ui| {
                lipid_creation::lipid_section(state, scene, engine_updates, ui)
            });
        }
        if state.ui.ui_vis.nucleic_acids {
            wrapped_section(ui, "nucleic_acids", |ui| {
                na_creation::na_section(state, scene, engine_updates, ui)
            });
        }
        if state.ui.ui_vis.amino_acids {
            wrapped_section(ui, "amino_acids", |ui| {
                aa_section(state, scene, engine_updates, ui)
            });
        }
    });
}
