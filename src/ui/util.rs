use std::{io, path::Path};

use egui::Ui;
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};

use crate::{
    OperatingMode, State,
    cam_misc::reset_camera,
    drawing::draw_peptide,
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_editor, mol_screening,
    mol_screening::screen_by_alignment,
    molecules::{MoleculeGeneric, small::MoleculeSmall},
    render::{set_flashlight, set_static_light},
    ui::set_window_title,
    util::{handle_err, reset_orbit_center},
};

/// Run this each frame, after all UI elements that affect it are rendered.
pub fn update_file_dialogs(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) -> io::Result<()> {
    let ctx = ui.ctx();

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    state.volatile.dialogs.screening.update(ctx);

    if let Some(path) = &state.volatile.dialogs.load.take_picked() {
        if let Err(e) = match state.volatile.operating_mode {
            OperatingMode::Primary => state.open_file(path, Some(scene), engine_updates),
            OperatingMode::MolEditor => state.mol_editor.open_molecule(
                &state.dev,
                &state.ff_param_set,
                &state.mol_specific_params,
                &state.to_save.md_config,
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
            OperatingMode::MolEditor => mol_editor::save(state, path)?,
            OperatingMode::ProteinEditor => (),
        }
    }

    if let Some(path) = &state.volatile.dialogs.screening.take_picked() {
        state.volatile.alignment.screening_path = Some(path.to_owned());
    }

    Ok(())
}

pub fn handle_redraw(
    state: &mut State,
    scene: &mut Scene,
    peptide: bool,
    lig: bool,
    na: bool,
    lipid: bool,
    reset_cam: bool,
    engine_updates: &mut EngineUpdates,
) {
    if peptide {
        draw_peptide(state, scene);
        // draw_all_ligs(state, scene); // todo: Hmm.

        if let Some(mol) = &state.peptide {
            set_window_title(&mol.common.ident, scene);
        }

        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Peptide as u32);

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            engine_updates.lighting = true;
        }
    }

    if lig {
        draw_all_ligs(state, scene);

        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Ligand as u32);

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            engine_updates.lighting = true;
        }
    }

    if na {
        draw_all_nucleic_acids(state, scene);
        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::NucleicAcid as u32);
    }

    if lipid {
        draw_all_lipids(state, scene);
        engine_updates.entities = EntityUpdate::All;
        // engine_updates.entities.push_class(EntityClass::Lipid as u32);
    }

    // Perform cleanup.
    if reset_cam {
        reset_camera(state, scene, engine_updates, FWD_VEC);
    }
}

/// Handles the case of opening a ligand remotely using the text input.
pub fn open_lig_from_input(
    state: &mut State,
    mol: MoleculeSmall,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.load_mol_to_state(
        MoleculeGeneric::Ligand(mol),
        Some(scene),
        engine_updates,
        None,
    );

    state.ui.db_input = String::new();
}

pub fn init_with_scene(state: &mut State, scene: &mut Scene, ctx: &egui::Context) {
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
    }

    reset_orbit_center(state, scene);
}

/// An assistant to make a colored label.
#[macro_export]
macro_rules! label {
    ($ui:expr, $text:expr, $color:expr) => {
        $ui.label(egui::RichText::new($text).color($color))
    };
}
