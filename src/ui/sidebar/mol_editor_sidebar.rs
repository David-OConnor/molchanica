use bio_apis::pubchem;
use egui::{Color32, FontId, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use mol_defs::molecules::MolIdent;

use crate::{
    button,
    drawing::{EntityClass, draw_pocket},
    label,
    mol_editor::DbCheck,
    pocket_render::PocketRender,
    selection::Selection,
    state::State,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_INACTIVE, ROW_SPACING,
        misc::{selector_box, selector_option},
        popup::pharmacophore,
    },
};

/// The editor molecule's SMILES, and its PubChem CID if it was loaded with one, or "Check DBs"
/// found one. These are for its current structure.
pub(in crate::ui) fn db_info(state: &State, ui: &mut Ui) {
    let editor = &state.mol_editor;

    let mut cid = None;
    let mut title = None;
    for ident in &editor.mol.idents {
        match ident {
            MolIdent::PubChem(v) => cid = Some(*v),
            MolIdent::PubchemTitle(v) if !v.is_empty() => title = Some(v),
            _ => (),
        }
    }

    ui.label("Identifiers");
    ui.separator();

    ui.horizontal_wrapped(|ui| {
        label!(ui, "SMILES:", Color32::GRAY);

        let smiles = if editor.smiles.is_empty() {
            "—"
        } else {
            &editor.smiles
        };
        // Wrap long SMILES instead of widening the sidebar.
        ui.label(
            RichText::new(smiles)
                .color(Color32::WHITE)
                .font(FontId::proportional(10.)),
        );
    });

    ui.horizontal_wrapped(|ui| {
        label!(ui, "CID:", Color32::GRAY);

        if let Some(cid) = cid {
            label!(ui, cid.to_string(), Color32::WHITE);

            if ui
                .button("PubChem")
                .on_hover_text("Open this compound's PubChem page in your web browser.")
                .clicked()
            {
                pubchem::open_overview(cid);
            }
        } else {
            let (text, help) = match editor.db_check {
                None => (
                    "Not checked",
                    "Click \"Check DBs\" to look up this molecule on PubChem.",
                ),
                Some(DbCheck::Pending) => ("Checking...", "Looking up this molecule on PubChem."),
                Some(DbCheck::NotFound) | Some(DbCheck::Found) => (
                    "Not in PubChem",
                    "PubChem has no compound with this structure.",
                ),
                Some(DbCheck::Failed) => (
                    "Lookup failed",
                    "Unable to look up this molecule on PubChem, e.g. from a network problem.",
                ),
            };

            label!(ui, text, Color32::WHITE).on_hover_text(help);
        }
    });

    if let Some(title) = title {
        ui.horizontal_wrapped(|ui| {
            label!(ui, "Name:", Color32::GRAY);
            label!(ui, title, Color32::WHITE);
        });
    }

    ui.add_space(ROW_SPACING);
}

pub(in crate::ui) fn pocket_list(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.label("Pockets");
    ui.separator();

    selector_box().show(ui, |ui| {
        for (mol_i, pocket) in state.pockets.iter_mut().enumerate() {
            let selected = state.mol_editor.pocket_i_in_state == Some(mol_i);

            if selector_option(ui, selected, &pocket.common.ident)
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
    });
}

pub(in crate::ui) fn pharmacophore_list(state: &mut State, ui: &mut Ui) {
    // todo: Make this work eventually when out of hte mol editor.

    ui.add_space(ROW_SPACING);

    // todo: Hmm. Need to redraw.
    let mut redraw_mol_editor = false;

    // The visibility flag goes in directly: `pharmacophore_list` clears it when its Close button
    // is hit. These are disjoint fields of `state`, so borrowing them together is fine.
    pharmacophore::pharmacophore_list(
        &mut state.mol_editor.mol.pharmacophore,
        &mut state.ui.popup,
        &mut state.ui.ui_vis.pharmacophore_list,
        ui,
        &mut redraw_mol_editor,
    );
}

/// e.g. functional groups, rings, etc.
pub(in crate::ui) fn component_list(state: &mut State, ui: &mut Ui, redraw: &mut bool) {
    // let Some(char) = &state.mol_editor.mol.characterization else {
    //     return;
    // };
    //
    // label!(ui, "Components (Char)", Color32::GRAY);
    // ui.add_space(COL_SPACING / 2.);
    //
    // for g in &char.rings {
    //     ui.horizontal(|ui| {
    //         label!(ui, "Ring", Color32::WHITE);
    //
    //         if ui.button("Sel").clicked() {}
    //
    //         if button!(ui, "❌", Color32::LIGHT_RED, "").clicked() {}
    //
    //         if ui.button("Chg to").clicked() {}
    //     });
    // }
    //
    // // for g in &char.chains {
    // //
    // // }
    //
    // for g in &char.hydroxyl {
    //     ui.horizontal(|ui| {
    //         label!(ui, "Hydroxyl", Color32::WHITE);
    //
    //         if ui.button("Sel").clicked() {}
    //
    //         if ui
    //             .button(RichText::new("❌").color(Color32::LIGHT_RED))
    //             .clicked()
    //         {}
    //
    //         if ui.button("Chg to").clicked() {}
    //     });
    // }
    //
    // for g in &char.carbonyl {
    //     ui.horizontal(|ui| {
    //         label!(ui, "Carbonyl", Color32::WHITE);
    //
    //         if ui.button("Sel").clicked() {}
    //
    //         if ui
    //             .button(RichText::new("❌").color(Color32::LIGHT_RED))
    //             .clicked()
    //         {}
    //
    //         if ui.button("Chg to").clicked() {}
    //     });
    // }
    //
    // for g in &char.carboxylate {}
    //
    // for g in &char.amides {}
    //
    // for g in &char.amines {}

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

            // for con in conns_to_this {
            //     label!(ui, format!(" - {con}"), Color32::GRAY);
            // }

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

                *redraw = true;
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
