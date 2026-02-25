use std::{fs::File, io, io::Write};

use egui::{Color32, Ui};
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};

use crate::{
    cam::reset_camera,
    drawing::{
        EntityClass, draw_peptide,
        wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids, draw_all_pockets},
    },
    mol_editor,
    molecules::{MolType, MoleculeGeneric, small::MoleculeSmall},
    render::{Color, set_flashlight, set_static_light},
    state::{OperatingMode, State},
    ui::set_window_title,
    util::{RedrawFlags, handle_err, reset_orbit_center},
};

/// Run this each frame, after all UI elements that affect it are rendered.
pub fn update_file_dialogs(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) -> io::Result<()> {
    let ctx = ui.ctx();

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    state.volatile.dialogs.screening.update(ctx);

    if let Some(path) = &state.volatile.dialogs.load.take_picked() {
        if let Err(e) = match state.volatile.operating_mode {
            OperatingMode::Primary => state.open_file(path, scene, engine_updates),
            OperatingMode::MolEditor => state.mol_editor.open_molecule(
                path,
                scene,
                engine_updates,
                &mut state.ui,
                state.volatile.mol_manip.mode,
            ),
            OperatingMode::ProteinEditor => unimplemented!(),
        } {
            handle_err(&mut state.ui, e.to_string());
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if let Some(path) = &state.volatile.dialogs.save.take_picked() {
        match state.volatile.operating_mode {
            OperatingMode::Primary => state.save(path)?,
            OperatingMode::MolEditor => {
                let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
                let extension = binding;

                // Deprecated, for now
                if extension == "pmp" {
                    let buf = state.mol_editor.mol.pharmacophore.to_bytes();
                    let mut file = File::create(path)?;
                    file.write_all(&buf)?;
                    println!("Saved Pharmacophore to {path:?}");
                } else {
                    mol_editor::save(state, path)?
                }
            }
            OperatingMode::ProteinEditor => (),
        }
    }

    if let Some(path) = &state.volatile.dialogs.screening.take_picked() {
        state.to_save.screening_path = Some(path.to_owned());
    }

    Ok(())
}

pub fn handle_redraw(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    reset_cam: bool,
    updates: &mut EngineUpdates,
) {
    if redraw.peptide {
        draw_peptide(state, scene);

        if let Some(mol) = &state.peptide {
            set_window_title(&mol.common.ident, scene);
        }

        updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Peptide as u32);

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            updates.lighting = true;
        }
    }

    if redraw.ligand {
        match state.volatile.operating_mode {
            OperatingMode::Primary => {
                draw_all_ligs(state, scene);
                // For docking light, but may be overkill here.
                if state.active_mol().is_some() {
                    updates.lighting = true;
                }
            }
            OperatingMode::MolEditor => mol_editor::redraw(
                &mut scene.entities,
                &state.mol_editor,
                &state.ui,
                state.volatile.mol_manip.mode,
            ),
            OperatingMode::ProteinEditor => unimplemented!(),
        }

        updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Ligand as u32);
    }

    if redraw.na {
        draw_all_nucleic_acids(state, scene);
        updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::NucleicAcid as u32);
    }

    if redraw.lipid {
        draw_all_lipids(state, scene);
        updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Lipid as u32);
    }

    if redraw.pocket {
        draw_all_pockets(state, scene);
        updates.entities = EntityUpdate::All;

        // engine_updates.entities.push_class(EntityClass::Pocket as u32);
    }

    // Perform cleanup.
    if reset_cam {
        reset_camera(state, scene, updates, FWD_VEC);
    }
}

/// Handles the case of opening a ligand remotely using the text input.
pub fn open_lig_from_input(
    state: &mut State,
    mol: MoleculeSmall,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.load_mol_to_state(MoleculeGeneric::Small(mol), scene, engine_updates, None);

    state.ui.db_input = String::new();
}

/// Contains functionality we wish to run at program load, but can't do until the scene is loaded.
/// Run this near the top of the UI initialization.
pub fn init_with_scene(state: &mut State, scene: &mut Scene) {
    // We must have loaded prefs prior to this, so we know which file to open.
    state.load_last_opened(scene);
    // todo trouble: It's somewhere around here, saving the inited-from-load atom posits, overwriting
    // todo the previously-saved ones.

    // todo: Workaround to allow us to apply params to the ligand once it's loaded. Unfortunate we have
    // todo to double-load prefs.
    {
        // state.load_prefs();

        // A default active small molecule.
        if !state.ligands.is_empty() {
            state.volatile.active_mol = Some((MolType::Ligand, 0));
        }
    }

    if state.peptide.is_some() {
        set_static_light(
            scene,
            state.peptide.as_ref().unwrap().center.into(),
            state.peptide.as_ref().unwrap().size,
        );
    } else if !state.ligands.is_empty() {
        let lig = &state.ligands[0];
        set_static_light(
            scene,
            lig.common.centroid().into(),
            3., // todo good enough?
        );

        //     let posit = state.to_save.per_mol[&mol.common.ident]
        //         .docking_site
        //         .site_center;
        //     // state.update_docking_site(posit);
    }

    // This updates the mesh and spheres after the initial prefs load, which may
    // have altered their posits. This prevents a visual jump upon the first re-render of pockets,
    // as the mesh moves to the correct location.
    let standalone_pocket_count = state.pockets.len();
    for (i, pocket) in state.pockets.iter_mut().enumerate() {
        pocket.mesh_i_rel = i;
        pocket.reset_post_manip(
            &mut scene.meshes,
            state.ui.mesh_coloring,
            &mut Default::default(),
        );
    }
    // Same treatment for pockets embedded in ligand pharmacophores.
    for (lig_i, lig) in state.ligands.iter_mut().enumerate() {
        if let Some(pocket) = &mut lig.pharmacophore.pocket {
            pocket.mesh_i_rel = standalone_pocket_count + lig_i;
            pocket.reset_post_manip(
                &mut scene.meshes,
                state.ui.mesh_coloring,
                &mut Default::default(),
            );
        }
    }

    reset_orbit_center(state, scene);

    draw_peptide(state, scene);
    draw_all_ligs(state, scene);
    draw_all_nucleic_acids(state, scene);
    draw_all_lipids(state, scene);
    draw_all_pockets(state, scene);

    set_flashlight(scene);
}

/// An assistant to make a colored label.
#[macro_export]
macro_rules! label {
    ($ui:expr, $text:expr, $color:expr) => {
        $ui.label(egui::RichText::new($text).color($color))
    };
}

/// An assistant to make a colored button.
#[macro_export]
macro_rules! button {
    ($ui:expr, $text:expr, $color:expr, $hover_text:expr) => {
        $ui.button(egui::RichText::new($text).color($color))
            .on_hover_text($hover_text)
    };
}

pub fn color_egui_from_f32(c: Color) -> Color32 {
    let (r, g, b) = c;
    Color32::from_rgb((r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8)
}
