//! Optional toolbars for nucleic acids, lipids etc.

use egui::Ui;
use graphics::{EngineUpdates, Scene};

use crate::{
    state::State,
    ui::panels::{aa_creation::aa_section, lipid_creation, na_creation},
};

/// Draw a section in a wrapping layout, starting a new row if it won't fit on this one. egui can't
/// wrap containers on its own, as it doesn't know their size until they're drawn; we use the
/// section's width from the previous frame.
fn wrapped_section(
    ui: &mut Ui,
    name: &str,
    row_start: &mut bool,
    add_contents: impl FnOnce(&mut Ui),
) {
    let id = ui.id().with(name);
    let width_prev: Option<f32> = ui.data(|d| d.get_temp(id));

    if !*row_start && width_prev.is_some_and(|w| w > ui.available_size_before_wrap().x) {
        ui.end_row();
    }

    let width = ui.scope(add_contents).response.rect.width();
    ui.data_mut(|d| d.insert_temp(id, width));

    *row_start = false;
}

pub(in crate::ui) fn mol_type_toolbars(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal_wrapped(|ui| {
        let mut row_start = true;

        if state.ui.ui_vis.lipids {
            wrapped_section(ui, "lipids", &mut row_start, |ui| {
                lipid_creation::lipid_section(state, scene, engine_updates, ui)
            });
        }
        if state.ui.ui_vis.nucleic_acids {
            wrapped_section(ui, "nucleic_acids", &mut row_start, |ui| {
                na_creation::na_section(state, scene, engine_updates, ui)
            });
        }
        if state.ui.ui_vis.amino_acids {
            wrapped_section(ui, "amino_acids", &mut row_start, |ui| {
                aa_section(state, scene, engine_updates, ui)
            });
        }
        //
        // if let Some(mol) = &state.active_mol()
        //     && !state.peptides.is_empty()
        //     && let MolGenericRef::Small(_) = mol
        // {
        //     ui.add_space(COL_SPACING);
        //     ui.label("Docking:");
        //
        //     if ui
        //         .button(RichText::new("Dock").color(Color32::GOLD))
        //         .clicked()
        //     {
        //         // The other views make it tough to see the ligand rel the protein.
        //         // if !matches!(state.ui.mol_view, MoleculeView::SpaceFill | MoleculeView::Surface) {
        //         //     // todo: Dim peptide?
        //         //     state.ui.mol_view = MoleculeView::Surface;
        //         // }
        //
        //         if let Err(e) = dock(
        //             state,
        //             state.volatile.active_mol.unwrap().1,
        //             scene,
        //             engine_updates,
        //         ) {
        //             handle_err(&mut state.ui, format!("Problem setting up docking: {e:?}"));
        //         }
        //     }
        // }
    });
}
