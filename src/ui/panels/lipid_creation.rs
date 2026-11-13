use egui::{ComboBox, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};
use mol_defs::molecules::{
    MolGenericRef, MolType,
    lipid::{LipidShape, make_bacterial_lipids},
};

use crate::{
    cam::move_cam_to_mol_front,
    drawing::{EntityClass, wrappers::draw_all_lipids},
    file_io::managed_mols,
    prefs::OpenType,
    state::State,
    ui,
    ui::{COLOR_REMOVE, misc::section_box, util::report_cache_result},
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

                let count_prev = state.lipids.len();
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

                // Each is kept in a managed file, so it reopens next session; as for a download.
                // Not loaded through `load_mol_to_state`, which would move each in front of the
                // camera, out of its pattern.
                for i in count_prev..state.lipids.len() {
                    let lipid = &state.lipids[i];
                    let result = managed_mols::store_created(
                        &state.volatile.prefs_dir,
                        MolGenericRef::Lipid(lipid),
                        &lipid.common.ident,
                    );

                    let Some(path) = report_cache_result(state, result) else {
                        break;
                    };

                    let lipid = &mut state.lipids[i];
                    lipid.common.update_path(&path);

                    let ident = Some(lipid.common.ident.clone());
                    state.update_history(&path, OpenType::Lipid, ident);
                }

                // Also records their positions.
                state.update_save_prefs();

                // The last one added.
                let molecule = (state.lipids.len() > count_prev)
                    .then(|| (MolType::Lipid, state.lipids.len() - 1));

                if molecule.is_some() {
                    state.volatile.active_mol = molecule;
                }

                draw_all_lipids(state, scene, updates);

                if let Some(molecule) = molecule {
                    move_cam_to_mol_front(state, scene, molecule, updates);
                }
            }

            if !state.lipids.is_empty()
                && ui
                    .button(RichText::new("Close all lipids").color(COLOR_REMOVE))
                    .clicked()
            {
                // Don't reopen them next session.
                let paths: Vec<_> = state
                    .lipids
                    .iter()
                    .filter_map(|l| l.common.path.clone())
                    .collect();
                for history in &mut state.to_save.open_history {
                    if history.type_ == OpenType::Lipid && paths.contains(&history.path) {
                        history.last_session = false;
                    }
                }
                state.update_save_prefs();

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
