use std::{collections::HashMap, fs::File, io, io::Write, path::PathBuf, slice};

use bio_apis::pubchem::find_cids_from_search;
use egui::{Color32, Response, RichText, Ui};
use graphics::{EngineUpdates, FWD_VEC, Scene};

use crate::{
    cam::reset_camera,
    drawing::{
        draw_peptide,
        wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids, draw_all_pockets},
    },
    file_io::{
        download_mols::{load_atom_coords_rcsb, load_sdf_drugbank, load_sdf_pubchem},
        save_mol_set_as_gro,
    },
    gromacs,
    md::viewer,
    mol_db::{COMMON_MOL_DB_NAME, ParquetMolDb},
    mol_editor,
    molecules::{MolType, MoleculeGeneric, common::MoleculeCommon, small::MoleculeSmall},
    render::{Color, set_flashlight, set_static_light},
    screening::load_mol_batch,
    smiles::is_smiles,
    state::{DbSel, OperatingMode, State},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, set_window_title},
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
    state.volatile.dialogs.parquet_db_load.update(ctx);
    state.volatile.dialogs.parquet_db_save.update(ctx);
    state.volatile.dialogs.parquet_mols_dir.update(ctx);
    state.volatile.dialogs.parquet_mol_file.update(ctx);
    state.volatile.dialogs.save_md.update(ctx);
    state.volatile.dialogs.save_gro.update(ctx);

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

    // Perhaps deprecated in favor of using screening databases.
    // if let Some(path) = &state.volatile.dialogs.screening.take_picked() {
    // state.to_save.screening_path = Some(path.to_owned());
    // }

    if let Some(path) = &state.volatile.dialogs.parquet_db_save.take_picked() {
        match ParquetMolDb::new(path) {
            Ok(db) => {
                handle_success(
                    &mut state.ui,
                    format!("Created Parquet database at path {path:?}"),
                );

                state.volatile.parquet_dbs.push(db);
                state.volatile.parquet_db_active =
                    Some(DbSel::Loaded(state.volatile.parquet_dbs.len() - 1));
            }
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error creating Parquet database: {e}"),
            ),
        }
    }

    if let Some(path) = &state.volatile.dialogs.parquet_db_load.take_picked() {
        state.load_parquet_db(path);
    }

    if let Some(path) = &state.volatile.dialogs.parquet_mols_dir.take_picked() {
        // The built-in DB is read-only, so only a loaded one can be populated.
        if let Some(DbSel::Loaded(i)) = state.volatile.parquet_db_active {
            let db = &mut state.volatile.parquet_dbs[i];
            match db.populate(path) {
                Ok(()) => {
                    println!("Populated Parquet DB: {} molecules", db.index_meta.len());
                }
                Err(e) => {
                    eprintln!("Error populating parquet data: {e:?}")
                }
            }
        } else {
            handle_err(
                &mut state.ui,
                "Error: Missing the DB index to populate with mols".to_string(),
            );
        }
    }

    if let Some(path) = &state.volatile.dialogs.parquet_mol_file.take_picked() {
        add_mol_file_to_db(state, path);
    }

    if let Some(path) = &state.volatile.dialogs.save_md.take_picked() {
        match gromacs::save_input_files(state, path) {
            Ok(_) => {
                handle_success(
                    &mut state.ui,
                    "Saved MD files in GROMACS format".to_string(),
                );
            }
            Err(e) => handle_err(&mut state.ui, format!("Error saving MD files: {e}")),
        }
    }

    if let Some(path) = state.volatile.dialogs.save_gro.take_picked() {
        let i = state.volatile.dialogs.save_gro_mol_set_i.take();
        if let Some(i) = i {
            let mol_sets = &state.volatile.md_local.viewer.mol_sets;
            if i < mol_sets.len() {
                match save_mol_set_as_gro(&mol_sets[i], &path) {
                    Ok(()) => handle_success(
                        &mut state.ui,
                        format!(
                            "Saved mol set as GRO: {:?}",
                            path.file_name().unwrap_or_default()
                        ),
                    ),
                    Err(e) => handle_err(&mut state.ui, format!("Error saving GRO: {e}")),
                }
            }
        }
    }

    Ok(())
}

/// Load a single molecule file (SDF or Mol2) from disk, and add it to the active Parquet database.
/// The directory equivalent is `ParquetMolDb::populate`.
fn add_mol_file_to_db(state: &mut State, path: &PathBuf) {
    // The built-in DB is read-only, so only a loaded one can be added to.
    let Some(DbSel::Loaded(db_i)) = state.volatile.parquet_db_active else {
        handle_err(
            &mut state.ui,
            "Error: Missing the DB index to add a mol to".to_string(),
        );
        return;
    };

    if db_i >= state.volatile.parquet_dbs.len() {
        handle_err(
            &mut state.ui,
            "Error: Invalid Parquet DB active index".to_string(),
        );
        return;
    }

    // `load_mol_batch` only handles these two, and panics otherwise; the dialog filter doesn't
    // prevent a name being typed in directly.
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if !matches!(ext.as_str(), "sdf" | "mol2") {
        handle_err(
            &mut state.ui,
            format!("Unable to add {ext:?} files to a database; use SDF or Mol2"),
        );
        return;
    }

    // An SDF file may hold more than one molecule.
    let mols = match load_mol_batch(slice::from_ref(path)) {
        Ok((mols, _consumed)) => mols,
        Err(e) => {
            handle_err(&mut state.ui, format!("Error loading the molecule: {e}"));
            return;
        }
    };

    if mols.is_empty() {
        handle_err(
            &mut state.ui,
            format!(
                "No molecule loaded from {:?}",
                path.file_name().unwrap_or_default()
            ),
        );
        return;
    }

    let mol_added_count = mols.len();

    let db = &mut state.volatile.parquet_dbs[db_i];
    let result = db.add_mols(&mols);
    let mol_count = db.index_meta.len();

    match result {
        Ok(()) => handle_success(
            &mut state.ui,
            format!("Added {mol_added_count} molecule(s) to the database ({mol_count} molecules)"),
        ),
        Err(e) => handle_err(
            &mut state.ui,
            format!("Error adding molecules to the database: {e}"),
        ),
    }
}

pub fn handle_redraw(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    reset_cam: bool,
    updates: &mut EngineUpdates,
) {
    if state.volatile.md_local.draw_md_mols
        && (redraw.peptide || redraw.ligand || redraw.lipid || redraw.na)
    {
        viewer::draw_mols(state, scene, updates);

        *redraw = Default::default();
        return;
    }

    if redraw.peptide {
        draw_peptide(state, scene, updates);

        if let Some(mol) = state
            .peptide_for_tools_i()
            .and_then(|i| state.peptide.get(i))
        {
            set_window_title(&mol.common.ident, scene);
        }

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            updates.lighting = true;
        }
    }

    if redraw.ligand {
        match state.volatile.operating_mode {
            OperatingMode::Primary => {
                draw_all_ligs(state, scene, updates);
            }
            OperatingMode::MolEditor => mol_editor::redraw(
                &mut scene.entities,
                &state.mol_editor,
                &state.ui,
                state.volatile.mol_manip.mode,
                updates,
            ),
            OperatingMode::ProteinEditor => unimplemented!(),
        }
    }

    if redraw.na {
        draw_all_nucleic_acids(state, scene, updates);
    }

    if redraw.lipid {
        draw_all_lipids(state, scene, updates);
    }

    if redraw.pocket {
        draw_all_pockets(state, scene, updates);
    }

    // Perform cleanup.
    if reset_cam {
        reset_camera(state, scene, updates, FWD_VEC);
    }

    *redraw = Default::default();
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
pub fn init_with_scene(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    // We must have loaded prefs prior to this, so we know which file to open.
    state.load_last_opened(scene);
    // todo trouble: It's somewhere around here, saving the inited-from-load atom posits, overwriting
    // todo the previously-saved ones.

    // todo: Workaround to allow us to apply params to the ligand once it's loaded. Unfortunate we have
    // todo to double-load prefs.
    {
        // A default active small molecule.
        if !state.ligands.is_empty() {
            state.volatile.active_mol = Some((MolType::Ligand, 0));
        }
    }

    if !state.peptide.is_empty() {
        set_static_light(
            scene,
            state
                .peptide_for_tools_i()
                .and_then(|i| state.peptide.get(i))
                .unwrap()
                .center
                .into(),
            state
                .peptide_for_tools_i()
                .and_then(|i| state.peptide.get(i))
                .unwrap()
                .size,
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

    // FWD_VEC here means a "Front" look.
    reset_camera(state, scene, &mut EngineUpdates::default(), FWD_VEC);

    draw_peptide(state, scene, updates);
    draw_all_ligs(state, scene, updates);
    draw_all_nucleic_acids(state, scene, updates);
    draw_all_lipids(state, scene, updates);
    draw_all_pockets(state, scene, updates);

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

/// The most matches from the built-in database we'll offer as buttons at once. The query bar is a
/// single row, so a long list of them would push the remote-lookup buttons off screen.
const COMMON_DB_RESULTS_MAX: usize = 4;

/// Shortest query the Enter key acts on; below this it's ignored, so a one- or two-character input
/// isn't fired off at a database. The caller applies this when deciding whether Enter was pressed;
/// `query` applies it again to decide which button to highlight as the Enter target.
pub(in crate::ui) const QUERY_ENTER_LEN_MIN: usize = 3;

/// What the built-in database made of a query; see `query_common_db`.
enum CommonDbOutcome {
    /// A molecule was loaded from it. The caller must not also run a remote lookup.
    Loaded,
    /// Matches were shown but none chosen yet. Enter belongs to the top one, so the remote lookups
    /// below are reachable only by clicking their buttons.
    Matched,
    /// Nothing matched; the remote lookups own this query, including its Enter key.
    NoMatch,
}

/// Search the built-in molecule database (`State::mol_db`) for the query text, matching on CID,
/// SMILES, or PubChem title, and draw a load button for each match. The top match is highlighted:
/// it's what Enter will load.
///
/// Enter loads the best match, which is why this runs ahead of the remote lookups in `query`: a
/// molecule we already have is always preferable to a network round trip. `ParquetMolDb::search`
/// ranks exact CID and title matches first.
fn query_common_db(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    inp: &str,
    enter_pressed: bool,
    // Whether Enter acts on this query at all; only affects which button is highlighted.
    enter_live: bool,
) -> CommonDbOutcome {
    let Some(db) = &state.mol_db else {
        return CommonDbOutcome::NoMatch;
    };

    // Gathered up front: loading borrows `state` mutably, and the search borrows the DB inside it.
    // (SMILES key, display name.)
    let hits: Vec<(String, String)> = db
        .search(inp, COMMON_DB_RESULTS_MAX)
        .iter()
        .map(|meta| {
            let name = match &meta.pubchem_title {
                Some(title) => title.clone(),
                None => meta.smiles.clone(),
            };
            (meta.smiles.clone(), name)
        })
        .collect();

    if hits.is_empty() {
        return CommonDbOutcome::NoMatch;
    }

    let mut to_load = None;

    for (i, (smiles, name)) in hits.iter().enumerate() {
        // The head is the best match, and so the one Enter loads; highlight it to show that.
        let color = if i == 0 && enter_live {
            COLOR_HIGHLIGHT
        } else {
            COLOR_ACTION
        };

        if button!(
            ui,
            name,
            color,
            "Open this molecule from the database built into the application. No internet \
             connection is used."
        )
        .clicked()
        {
            to_load = Some((smiles.clone(), name.clone()));
        }
    }

    // Ranked best-first, so the head is the best match.
    if to_load.is_none() && enter_pressed {
        to_load = Some(hits[0].clone());
    }

    let Some((smiles, name)) = to_load else {
        return CommonDbOutcome::Matched;
    };

    // Re-borrowed here rather than reused from above: `load_mol` needs the DB, and opening the
    // molecule needs `state`.
    let mol = {
        let Some(db) = &state.mol_db else {
            return CommonDbOutcome::Matched;
        };

        match db.load_mol(&smiles) {
            Ok(mut mol) => {
                // `mol_data` and the idents/metadata are separate columns; see the `mol_db` module
                // docs. Missing idents are not fatal — the molecule is still usable.
                if let Err(e) = db.apply_idents_meta(slice::from_mut(&mut mol)) {
                    eprintln!("Error loading idents for {smiles}: {e}");
                }
                mol
            }
            Err(e) => {
                handle_err(
                    &mut state.ui,
                    format!("Error loading {smiles} from the built-in database: {e}"),
                );
                return CommonDbOutcome::Matched;
            }
        }
    };

    open_lig_from_input(state, mol, scene, updates);
    redraw.ligand = true;

    handle_success(
        &mut state.ui,
        format!("Loaded {name} ({smiles}) from {COMMON_MOL_DB_NAME}; no network query"),
    );

    CommonDbOutcome::Loaded
}

/// Draws one of the query bar's remote-lookup buttons, highlighting it if Enter would activate it.
/// Only one button in the bar is ever the Enter target.
fn query_btn(ui: &mut Ui, text: &str, is_enter_target: bool) -> Response {
    let text = RichText::new(text);

    ui.button(match is_enter_target {
        true => text.color(COLOR_HIGHLIGHT),
        false => text,
    })
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

    // Molecules we ship with the application. Checked ahead of the remote databases below, since
    // these load instantly and without an internet connection, and Enter picks the best local match
    // over any of them. Matches are still drawn as buttons, so a remote lookup stays one click away.
    // Whether Enter does anything at all for this query; a shorter one it ignores. Nothing is
    // highlighted as the Enter target when it isn't one.
    let enter_live = inp.len() >= QUERY_ENTER_LEN_MIN;

    let common = query_common_db(
        state,
        scene,
        redraw,
        updates,
        ui,
        inp,
        enter_pressed,
        enter_live,
    );

    // Whether one of the remote buttons below is what Enter activates: it is, unless the built-in
    // DB matched and claimed the key. Used to highlight that button.
    let enter_tgt = match common {
        CommonDbOutcome::Loaded => return,
        CommonDbOutcome::Matched => false,
        CommonDbOutcome::NoMatch => enter_live,
    };

    if inp.len() == 4 || inp_l.starts_with("pdb_") {
        // Enter only acts on a bare 4-character ident here, so a `pdb_`-prefixed one isn't a target.
        let button_clicked = query_btn(ui, "Load RCSB", enter_tgt && inp.len() == 4).clicked();
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
        let button_clicked = query_btn(ui, "Load Geostd", enter_tgt).clicked();

        if button_clicked || enter_pressed {
            state.load_geostd_mol_data(&inp_l, true, true, updates, scene);

            state.ui.db_input = String::new();
        }

        return;
    }

    if inp.len() > 4 && inp_l.starts_with("db") {
        let button_clicked = query_btn(ui, "Load DrugBank", enter_tgt).clicked();

        if button_clicked || enter_pressed {
            match load_sdf_drugbank(&inp_l) {
                Ok(mol) => {
                    open_lig_from_input(state, mol, scene, updates);
                    redraw.ligand = true;

                    handle_success(
                        &mut state.ui,
                        format!("Loaded {inp_l} from DrugBank (over the internet)"),
                    );
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
        if query_btn(ui, "Load PubChem", enter_tgt).clicked() || enter_pressed {
            match load_sdf_pubchem(cid) {
                Ok(mol) => {
                    open_lig_from_input(state, mol, scene, updates);
                    redraw.ligand = true;
                    // reset_cam = true;

                    handle_success(
                        &mut state.ui,
                        format!("Loaded CID {cid} from PubChem (over the internet)"),
                    );
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
        let button_clicked = query_btn(ui, "Load from SMILES", enter_tgt).clicked();
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

                    open_lig_from_input(state, mol, scene, updates);
                    redraw.ligand = true;

                    handle_success(
                        &mut state.ui,
                        String::from("Built this molecule from the SMILES entered; no database"),
                    );
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
        let button_clicked = query_btn(ui, "Search PubChem", enter_tgt).clicked();
        if button_clicked || enter_pressed {
            let cids = find_cids_from_search(inp, false);

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
                                        "Found the following Pubchem CIDs: {cids_str}. Loaded {} \
                                         from PubChem (over the internet)",
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
