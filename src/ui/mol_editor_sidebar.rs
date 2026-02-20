use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};

use crate::{
    button,
    drawing::{EntityClass, draw_pocket},
    label,
    selection::Selection,
    state::State,
    ui::{COL_SPACING, COLOR_ACTIVE, COLOR_INACTIVE, ROW_SPACING, pharmacophore},
};

pub(in crate::ui) fn pocket_list(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.label("Pockets");
    ui.separator();

    for (mol_i, pocket) in state.pockets.iter_mut().enumerate() {
        let selected = state.mol_editor.pocket_i_in_state == Some(mol_i);

        let color = if selected {
            COLOR_ACTIVE
        } else {
            COLOR_INACTIVE
        };

        if ui
            .button(RichText::new(&pocket.common.ident).color(color))
            .on_hover_text(
                "Display this pocket, and optionally use it as part of \
                        a pharmacophore, e.g. its excluded volume.",
            )
            .clicked()
        {
            scene
                .entities
                .retain(|e| e.class != EntityClass::Pocket as u32);

            state.mol_editor.update_h_bonds();
            // Not sure why updating the pocket alone isn't working; entityupdate::All
            // is working though.
            updates.meshes = true;
            updates.entities = EntityUpdate::All;

            if selected {
                state.mol_editor.mol.pharmacophore.pocket = None;
                state.mol_editor.pocket_i_in_state = None;
            } else {
                pocket.common.center_local_posits_around_origin();
                pocket.common.reset_posits();

                pocket.reset_post_manip(&mut scene.meshes, state.ui.mesh_coloring, updates);

                state.mol_editor.mol.pharmacophore.pocket = Some(pocket.clone());
                state.mol_editor.pocket_i_in_state = Some(mol_i);

                scene
                    .entities
                    .retain(|e| e.class != EntityClass::Pocket as u32);

                scene.entities.extend(draw_pocket(
                    pocket,
                    &state.mol_editor.h_bonds,
                    &state.mol_editor.mol.common.atom_posits,
                    &state.ui.visibility,
                    &state.ui.selection,
                    &state.volatile.mol_manip.mode,
                ));

                updates.meshes = true;
            }
        }
    }
}

pub(in crate::ui) fn pharmacophore_list(state: &mut State, ui: &mut Ui) {
    // todo: Make this work eventually when out of hte mol editor.

    let mol = &mut state.mol_editor.mol;
    let mut closed = false;

    ui.add_space(ROW_SPACING);

    // todo: Hmm. Need to redraw.
    let mut redraw_mol_editor = false;
    pharmacophore::pharmacophore_list(
        &mut mol.pharmacophore,
        &mut state.ui.popup,
        &mut closed,
        ui,
        &mut redraw_mol_editor,
    );

    if closed {
        // Broken out to avoid double borrow.
        state.ui.ui_vis.pharmacophore_list = false;
    }
}

/// e.g. functional groups, rings, etc.
pub(in crate::ui) fn component_list(state: &mut State, ui: &mut Ui) {
    // todo: Char  field based for now. Reconcile this with your component structures.
    let Some(char) = &state.mol_editor.mol.characterization else {
        return;
    };

    label!(ui, "Components (Char)", Color32::GRAY);
    ui.add_space(COL_SPACING / 2.);

    for g in &char.rings {
        ui.horizontal(|ui| {
            label!(ui, "Ring", Color32::WHITE);

            if ui.button("Sel").clicked() {}

            if button!(ui, "❌", Color32::LIGHT_RED, "").clicked() {}

            if ui.button("Chg to").clicked() {}
        });
    }

    // for g in &char.chains {
    //
    // }

    for g in &char.hydroxyl {
        ui.horizontal(|ui| {
            label!(ui, "Hydroxyl", Color32::WHITE);

            if ui.button("Sel").clicked() {}

            if ui
                .button(RichText::new("❌").color(Color32::LIGHT_RED))
                .clicked()
            {}

            if ui.button("Chg to").clicked() {}
        });
    }

    for g in &char.carbonyl {
        ui.horizontal(|ui| {
            label!(ui, "Carbonyl", Color32::WHITE);

            if ui.button("Sel").clicked() {}

            if ui
                .button(RichText::new("❌").color(Color32::LIGHT_RED))
                .clicked()
            {}

            if ui.button("Chg to").clicked() {}
        });
    }

    for g in &char.carboxylate {}

    for g in &char.amides {}

    for g in &char.amines {}

    // Component-based approach below; char-based approach above -------------

    let Some(comps) = &state.mol_editor.mol.components else {
        return;
    };

    ui.add_space(ROW_SPACING);
    ui.label("Components (mol comps)");
    ui.separator();

    for (i_comp, comp) in comps.components.iter().enumerate() {
        ui.horizontal(|ui| {
            // this loop is probably not great to bind conns. Todo: Some sort of hash
            // todo as a cheap way to speed up, if you use this apch.

            let mut conns_to_this = Vec::new();
            for conn in &comps.connections {
                if conn.comp_0 == i_comp {
                    conns_to_this.push(conn.comp_1);
                } else if conn.comp_1 == i_comp {
                    conns_to_this.push(conn.comp_0);
                }
            }

            label!(ui, format!("{i_comp}: {}", comp.comp_type), Color32::WHITE);

            ui.add_space(COL_SPACING);

            for con in conns_to_this {
                label!(ui, format!(" - {con}"), Color32::GRAY);
            }

            let selected = state.ui.selection == Selection::ComponentEditor(i_comp);
            let color_sel = if selected {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if button!(
                ui,
                "sel",
                color_sel,
                "Select this component for details and editing"
            )
            .clicked()
            {
                state.ui.selection = if selected {
                    Selection::None
                } else {
                    Selection::ComponentEditor(i_comp)
                };
            }

            if button!(
                ui,
                "❌",
                Color32::LIGHT_RED,
                "Delete this component and all atoms in it"
            )
            .clicked()
            {}

            if ui.button("Chg to").clicked() {}
        });
    }

    ui.add_space(ROW_SPACING);
    ui.label("Connections: ");
    for conn in &comps.connections {
        let mut descrip = format!(
            "Mol {} - {} | Atom {} - {}",
            conn.comp_0, conn.comp_1, conn.atom_0, conn.atom_1
        );

        if conn.shared_atoms {
            descrip.push_str(" - Shared");
        }

        label!(ui, descrip, Color32::WHITE);
    }
}
