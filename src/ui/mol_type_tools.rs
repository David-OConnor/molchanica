//! Optional toolbars for nucleic acids, lipids etc.

use egui::Ui;
use graphics::{EngineUpdates, Scene};

use crate::{
    state::State,
    ui::panels::{aa_creation::aa_section, lipid_creation, na_creation},
};

pub(in crate::ui) fn mol_type_toolbars(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        if state.ui.ui_vis.lipids {
            lipid_creation::lipid_section(state, scene, engine_updates, ui);
        }
        if state.ui.ui_vis.nucleic_acids {
            na_creation::na_section(state, scene, engine_updates, ui);
        }
        if state.ui.ui_vis.amino_acids {
            aa_section(state, scene, engine_updates, ui);
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
