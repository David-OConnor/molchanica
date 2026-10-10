use egui::{Color32, ComboBox, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};
use mol_defs::molecules::lipid::{LipidShape, make_bacterial_lipids};

use crate::{
    drawing::{EntityClass, wrappers::draw_all_lipids},
    state::State,
    ui,
    ui::misc::section_box,
    util::clear_mol_entity_indices,
};

/// Add and manage lipids
pub(in crate::ui) fn lipid_section(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.to_save.lipid.lipid_to_add >= state.templates.lipid.len() {
        eprintln!("Error: Not enough lipid templates");
        return;
    }

    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Add lipids:");

            let add_standard_text = state.templates.lipid[state.to_save.lipid.lipid_to_add]
                .common
                .ident
                .clone();

            ComboBox::from_id_salt(102)
                .width(90.)
                .selected_text(state.to_save.lipid.shape.to_string())
                .show_ui(ui, |ui| {
                    for shape in [LipidShape::Free, LipidShape::Membrane, LipidShape::Lnp] {
                        ui.selectable_value(
                            &mut state.to_save.lipid.shape,
                            shape,
                            shape.to_string(),
                        );
                    }
                })
                .response
                .on_hover_text("Add lipids in this pattern");

            if state.to_save.lipid.shape == LipidShape::Free {
                ComboBox::from_id_salt(101)
                    .width(30.)
                    .selected_text(add_standard_text)
                    .show_ui(ui, |ui| {
                        for (i, mol) in state.templates.lipid.iter().enumerate() {
                            ui.selectable_value(
                                &mut state.to_save.lipid.lipid_to_add,
                                i,
                                &mol.common.ident,
                            );
                        }
                    })
                    .response
                    .on_hover_text("Add this lipid to the scene.");
            }

            ui::num_field(&mut state.to_save.lipid.mol_count, "# mols", 36, ui);

            // todo: Multiple and sets once this is validated
            if ui.button("+").clicked() {
                // Place in front of the camera.
                let center = scene.camera.position
                    + scene.camera.orientation.rotate_vec(FWD_VEC) * crate::cam::MOVE_TO_CAM_DIST;

                state.lipids.extend(make_bacterial_lipids(
                    state.to_save.lipid.mol_count as usize,
                    center.into(),
                    state.to_save.lipid.shape,
                    &state.templates.lipid,
                ));
                //
                // let mut mol = state.templates.lipid[state.ui.lipid_to_add].clone();
                // for p in &mut mol.common.atom_posits {
                //     *p = *p + Vec3::new_zero();
                // }
                //
                // state.lipids.push(mol);

                draw_all_lipids(state, scene, updates);
            }

            if !state.lipids.is_empty()
                && ui
                    .button(RichText::new("Close all lipids").color(Color32::LIGHT_RED))
                    .clicked()
            {
                state.lipids = Vec::new();
                scene
                    .entities
                    .retain(|e| e.class != EntityClass::Lipid as u32);
                clear_mol_entity_indices(state, None);

                updates.entities = EntityUpdate::All;
            }
        });
    });
}
