use std::{collections::HashMap, fs::File, io, io::Write};

use bio_apis::pubchem::find_cids_from_search;
use egui::{Color32, Ui};
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};

use crate::{
    cam::reset_camera,
    drawing::{
        EntityClass, draw_peptide,
        wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids, draw_all_pockets},
    },
    file_io::download_mols::{load_atom_coords_rcsb, load_sdf_drugbank, load_sdf_pubchem},
    mol_editor,
    molecules::{MolType, MoleculeGeneric, common::MoleculeCommon, small::MoleculeSmall},
    render::{Color, set_flashlight, set_static_light},
    smiles::is_smiles,
    state::{OperatingMode, State},
    ui::set_window_title,
    util::{RedrawFlags, handle_err, handle_success, reset_orbit_center},
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

/// Handles a general query, which could be a name, identifier etc. Attempts to query
/// the correct database based on the  text.
///
/// We assume input is lowercase and trimmed.
pub(in crate::ui) fn query(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    reset_cam: &mut bool,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    inp: &str,
    enter_pressed: bool,
) {
    // Lowercase copy used for identifier-prefix checks (PDB, DrugBank, etc.).
    // The original `inp` is passed to SMILES functions so that aromaticity case is
    // preserved (lowercase = aromatic atom in SMILES, uppercase = aliphatic).
    let inp_l = inp.to_ascii_lowercase();

    if inp.len() == 4 || inp_l.starts_with("pdb_") {
        let button_clicked = ui.button("Load RCSB").clicked();
        if (button_clicked || enter_pressed) && inp.len() == 4 {
            let ident = inp_l.clone();

            load_atom_coords_rcsb(
                &ident,
                state,
                scene,
                updates,
                &mut redraw.peptide,
                reset_cam,
            );

            state.ui.db_input = String::new();
            return;
        }

        return;
    }

    if inp.len() == 3 {
        let button_clicked = ui.button("Load Geostd").clicked();

        if button_clicked || enter_pressed {
            state.load_geostd_mol_data(&inp_l, true, true, updates, scene);

            state.ui.db_input = String::new();
        }

        return;
    }

    if inp.len() > 4 && inp_l.starts_with("db") {
        let button_clicked = ui.button("Load DrugBank").clicked();

        if button_clicked || enter_pressed {
            match load_sdf_drugbank(&inp_l) {
                Ok(mol) => {
                    open_lig_from_input(state, mol, scene, updates);
                    redraw.ligand = true;
                    // reset_cam = true;
                }
                Err(e) => {
                    let msg = format!("Error loading SDF file: {e:?}");
                    handle_err(&mut state.ui, msg);
                }
            }
        }

        return;
    }

    // PubChem CID.
    if let Ok(cid) = state.ui.db_input.parse::<u32>() {
        if ui.button("Load PubChem").clicked() || enter_pressed {
            match load_sdf_pubchem(cid) {
                Ok(mol) => {
                    open_lig_from_input(state, mol, scene, updates);
                    redraw.ligand = true;
                    // reset_cam = true;
                }
                Err(e) => {
                    let msg = format!("Error loading SDF file: {e:?}");
                    handle_err(&mut state.ui, msg);
                }
            }
        }

        return;
    }

    // I believe this is cheap enough to run here (continuously)
    if is_smiles(inp) {
        let button_clicked = ui.button("Load from SMILES").clicked();
        // Attempt ot infer if this is SMILES.
        if enter_pressed || button_clicked {
            match MoleculeCommon::from_smiles(inp) {
                Ok(m) => {
                    let smiles_start: String = inp.chars().take(5).collect();

                    let mol = MoleculeSmall::new(
                        format!("From SMILES {}", smiles_start),
                        m.atoms,
                        m.bonds,
                        HashMap::new(),
                        None,
                    );

                    // // todo temp
                    // println!("\n  Loaded SMILES atoms:  \n");
                    // for a in &mol.common.atoms {
                    //     println!("-{a}");
                    // }
                    //
                    // println!("\n  Loaded SMILES bonds:  \n");
                    // for b in &mol.common.bonds {
                    //     println!("-{b:?}");
                    // }

                    open_lig_from_input(state, mol, scene, updates);
                    redraw.ligand = true;
                }
                Err(e) => {
                    let msg = format!("Error loading a molecule from SMILES: {e:?}");
                    handle_err(&mut state.ui, msg);
                }
            }
        }
        return;
    }

    // PubChem name search.
    if inp.len() >= 5 && !inp_l.starts_with("pdb_") && !inp_l.starts_with("db") {
        let button_clicked = ui.button("Search PubChem").clicked();
        if button_clicked || enter_pressed {
            let cids = find_cids_from_search(&inp, false);

            match cids {
                Ok(c) => {
                    if c.is_empty() {
                        handle_success(&mut state.ui, "No results found on Pubchem".to_owned());
                    } else {
                        // todo: DRY with the other pubchem branch above.
                        match load_sdf_pubchem(c[0]) {
                            Ok(mol) => {
                                open_lig_from_input(state, mol, scene, updates);
                                redraw.ligand = true;
                                // reset_cam = true;

                                let cids_str =
                                    c.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");

                                handle_success(
                                    &mut state.ui,
                                    format!(
                                        "Found the following Pubchem CIDs: {cids_str}. Loaded {}",
                                        c[0]
                                    ),
                                );
                            }
                            Err(e) => {
                                let msg = format!("Error loading SDF file: {e:?}");
                                handle_err(&mut state.ui, msg);
                            }
                        }
                    }
                }
                Err(e) => handle_err(
                    &mut state.ui,
                    format!("Error finding a mol from Pubchem {:?}", e),
                ),
            }
        }
    }
}
