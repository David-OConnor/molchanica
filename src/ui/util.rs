use std::{
    collections::HashMap,
    fs::File,
    io,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    slice,
};

use bio_apis::{chebi, pubchem::find_cids_from_search};
use egui::{Color32, Response, RichText, Ui};
use graphics::{EngineUpdates, FWD_VEC, Scene};
use mol_defs::{
    molecules::{MolIdent, MolType, MoleculeGeneric, common::MoleculeCommon, small::MoleculeSmall},
    smiles::is_smiles,
};

use crate::ui::COL_SPACING;
use crate::{
    cam::reset_camera,
    drawing::{
        draw_peptide,
        wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids, draw_all_pockets},
    },
    external_tools::home_directory,
    file_io::{
        download_mols::{
            load_atom_coords_rcsb, load_sdf_chebi, load_sdf_drugbank, load_sdf_pubchem,
        },
        managed_mols::{self, ManagedMolProvider},
        save_mol_set_as_gro,
    },
    gromacs,
    md::viewer,
    mol_db::{CHEBI_DB_NAME, ParquetMolDb},
    mol_editor,
    pocket_render::PocketRender,
    prefs::OpenType,
    render::{Color, set_flashlight, set_static_light},
    state::{DbSel, OperatingMode, State},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, set_window_title},
    util::{RedrawFlags, handle_err, handle_success, reset_orbit_center},
};

/// A path formatted for display: the home directory abbreviated to "~", and forward slashes on all
/// platforms, e.g. "~/Desktop/4091.mol2".
pub(in crate::ui) fn display_path(path: &Path) -> String {
    let abbreviated = home_directory()
        .and_then(|home| {
            path.strip_prefix(&home)
                .ok()
                .map(|rel| Path::new("~").join(rel))
        })
        .unwrap_or_else(|| path.to_path_buf());

    abbreviated.to_string_lossy().replace('\\', "/")
}

/// Show a folder in the platform's file browser. Shared by every "open this location" button in the
/// UI, so they all behave the same way, and so the per-platform command lives in one place.
pub(in crate::ui) fn open_dir(folder: &Path) -> io::Result<()> {
    #[cfg(target_os = "windows")]
    const OPENER: &str = "explorer.exe";
    #[cfg(target_os = "macos")]
    const OPENER: &str = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    const OPENER: &str = "xdg-open";

    Command::new(OPENER).arg(folder).spawn().map(|_| ())
}

/// Display small-molecule identifiers consistently wherever they are listed in the UI.
pub(in crate::ui) fn list_idents(
    idents: &[MolIdent],
    path: &Option<PathBuf>,
    prefs_dir: &Path,
    ui: &mut Ui,
) {
    if let Some(p) = path {
        ui.horizontal_wrapped(|ui| {
            // Managed molecules were never saved to disk by the user; their cache path is an
            // implementation detail, so describe where they came from instead.
            crate::label!(ui, "File:", Color32::GRAY);
            if managed_mols::is_managed_path(prefs_dir, p) {
                ui.label(RichText::new("Downloaded; not saved permanently").color(Color32::WHITE))
                    .on_hover_text(p.to_string_lossy());
            } else {
                // The unabbreviated, native-separator path is available on hover.
                ui.label(RichText::new(display_path(p)).color(Color32::WHITE))
                    .on_hover_text(p.to_string_lossy());
            }

            if let Some(dir) = p.parent() {
                if ui
                    .button("Open dir")
                    .on_hover_text(
                        "Open your OS's file browser to the directory containing this file.",
                    )
                    .clicked()
                {
                    // No `handle_err` here: this is drawn while the molecule is borrowed from
                    // state, so the CLI output line is out of reach.
                    if let Err(e) = open_dir(dir) {
                        eprintln!("Error opening the folder {}: {e}", dir.display());
                    }
                }
            }
        });
    }

    for ident in idents {
        // Wrap long identifiers instead of expanding the containing panel.
        ui.horizontal_wrapped(|ui| {
            crate::label!(ui, format!("{}:", ident.ident_type()), Color32::GRAY);

            let mut ident_text = RichText::new(ident.ident_inner()).color(Color32::WHITE);

            if matches!(
                ident,
                MolIdent::InchIKey(_)
                    | MolIdent::InchI(_)
                    | MolIdent::Smiles(_)
                    | MolIdent::IupacName(_)
            ) {
                ident_text = ident_text.font(egui::FontId::proportional(10.0));
            }

            ui.label(ident_text);
        });
    }
}

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
    state.volatile.dialogs.sequence_prediction.update(ctx);
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

                // Record it in the open history, so it's reopened on the next launch (mirrors
                // `State::load_parquet_db`).
                state.update_history(path, OpenType::ParquetDb, None);
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
            match db.add_mols_from_dir(path) {
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
        // The built-in DB is read-only, so only a loaded one can be populated.
        if let Some(DbSel::Loaded(i)) = state.volatile.parquet_db_active {
            let db = &mut state.volatile.parquet_dbs[i];
            match db.add_mols_from_file(path) {
                Ok(()) => {
                    println!(
                        "Added mols from file. DB now has {} molecules",
                        db.index_meta.len()
                    );
                }
                Err(e) => {
                    eprintln!("Error adding mols from file: {e:?}")
                }
            }
        } else {
            handle_err(
                &mut state.ui,
                "Error: Missing the DB index to add a mol to".to_string(),
            );
        }
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
            .and_then(|i| state.peptides.get(i))
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
    path: Option<&Path>,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.load_mol_to_state(MoleculeGeneric::Small(mol), scene, engine_updates, path);

    state.ui.db_input = String::new();
}

fn report_cache_result(state: &mut State, result: io::Result<PathBuf>) -> Option<PathBuf> {
    match result {
        Ok(path) => Some(path),
        Err(error) => {
            handle_err(
                &mut state.ui,
                format!("Could not save a restorable copy of this molecule: {error}"),
            );
            None
        }
    }
}

fn cache_sdf_source(
    state: &mut State,
    provider: ManagedMolProvider,
    key: &str,
    query: &str,
    source_text: &str,
) -> Option<PathBuf> {
    let result = managed_mols::store_text(
        &state.volatile.prefs_dir,
        provider,
        key,
        query,
        "sdf",
        source_text,
    );
    report_cache_result(state, result)
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

    if !state.peptides.is_empty() {
        set_static_light(
            scene,
            state
                .peptide_for_tools_i()
                .and_then(|i| state.peptides.get(i))
                .unwrap()
                .center
                .into(),
            state
                .peptide_for_tools_i()
                .and_then(|i| state.peptides.get(i))
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

/// Search the built-in ChEBI molecule database (`State::chebi_mol_db`) for the query text, matching
/// on CID, SMILES, or PubChem title, and draw a load button for each match. The top match is
/// highlighted: it's what Enter will load.
///
/// Enter loads the best match, which is why this runs ahead of the remote lookups in `query`: a
/// molecule we already have is always preferable to a network round trip. `ParquetMolDb::search`
/// ranks exact CID and title matches first.
// Temporarily unused: the caller's ChEBI integration is disabled because that DB is 2D-only. Kept
// (with its ranking wired to the shared `search`) so it can be switched back on with a 3D source.
#[allow(dead_code)]
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
    let Some(db) = &state.chebi_mol_db else {
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
        let Some(db) = &state.chebi_mol_db else {
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

    let cache_result = managed_mols::store_sdf(
        &state.volatile.prefs_dir,
        ManagedMolProvider::BuiltIn,
        &managed_mols::text_key(&smiles),
        &smiles,
        &mol.to_sdf(),
    );
    let Some(cache_path) = report_cache_result(state, cache_result) else {
        return CommonDbOutcome::Matched;
    };
    open_lig_from_input(state, mol, Some(&cache_path), scene, updates);
    redraw.ligand = true;

    handle_success(
        &mut state.ui,
        format!("Loaded {name} ({smiles}) from {CHEBI_DB_NAME}."),
    );

    CommonDbOutcome::Loaded
}

/// Which of the query bar's remote lookups the Enter key acts on. Several lookups can match one
/// query — a 4-digit number is both a PubChem CID and an RCSB ident — but only one of them owns
/// Enter, and it's the only one drawn highlighted.
#[derive(Clone, Copy, PartialEq)]
enum EnterTarget {
    /// Enter does nothing: the query is too short, matches no lookup, or the built-in DB claimed it.
    None,
    PubchemCid,
    /// A `chebi:`-prefixed accession, which no other lookup competes for.
    ChebiId,
    Rcsb,
    Geostd,
    DrugBank,
    Smiles,
    PubchemSearch,
}

/// Prefix that pins a query to ChEBI alone, e.g. `CHEBI:46195`. Matched case-insensitively, against
/// the lowercased query.
const CHEBI_PREFIX: &str = "chebi:";

/// Decide which lookup Enter activates, mirroring the order the buttons are drawn in below. This is
/// resolved once, ahead of drawing, so the highlighted button and the one Enter actually loads can't
/// disagree.
fn enter_target(inp: &str, inp_l: &str) -> EnterTarget {
    // An explicitly prefixed ChEBI accession names its database, so nothing else can claim it.
    if inp_l.starts_with(CHEBI_PREFIX) {
        return EnterTarget::ChebiId;
    }

    // A numeric query is a PubChem CID, and takes Enter even when it also looks like an RCSB ident
    // (4 digits) or a Geostd one (3 digits): those idents are alphanumeric in practice, so an
    // all-digit query is far more likely meant as a CID. Their buttons are still drawn, one click away.
    if inp.parse::<u32>().is_ok() {
        return EnterTarget::PubchemCid;
    }

    // A `pdb_`-prefixed ident draws the RCSB button but isn't an Enter target; only a bare
    // 4-character one is. Either way the query belongs to RCSB, so nothing below can claim it.
    if inp.len() == 4 {
        return EnterTarget::Rcsb;
    }
    if inp_l.starts_with("pdb_") {
        return EnterTarget::None;
    }

    if inp.len() == 3 {
        return EnterTarget::Geostd;
    }

    if inp.len() > 4 && inp_l.starts_with("db") {
        return EnterTarget::DrugBank;
    }

    if is_smiles(inp) {
        return EnterTarget::Smiles;
    }

    if inp.len() >= 5 {
        return EnterTarget::PubchemSearch;
    }

    EnterTarget::None
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

/// Download and open a molecule from ChEBI, by numeric accession. Shared by the bare-number query,
/// where ChEBI is the alternative to PubChem, and the `chebi:`-prefixed one, where it's the only
/// database consulted.
fn load_chebi_id(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
    id: u32,
) {
    match load_sdf_chebi(id) {
        Ok(downloaded) => {
            let key = id.to_string();
            // Cache the molecule rather than the file ChEBI served: that file has no data fields,
            // so it can't carry the accession we just attached, and the accession must survive a
            // restart. `to_sdf` writes our identifier metadata.
            let cache_result = managed_mols::store_sdf(
                &state.volatile.prefs_dir,
                ManagedMolProvider::Chebi,
                &key,
                &key,
                &downloaded.mol.to_sdf(),
            );
            let Some(cache_path) = report_cache_result(state, cache_result) else {
                return;
            };
            open_lig_from_input(state, downloaded.mol, Some(&cache_path), scene, updates);
            redraw.ligand = true;

            handle_success(
                &mut state.ui,
                format!(
                    "Loaded CHEBI:{id} from ChEBI (over the internet). Note that ChEBI structures \
                     are 2D."
                ),
            );
        }
        Err(e) => {
            let msg = format!("Error loading SDF file: {e:?}");
            handle_err(&mut state.ui, msg);
        }
    }
}

/// Handles a general query, which could be a name, identifier etc. Attempts to query
/// the correct database based on the  text.
///
/// inp is trimmed and case-preserving; inp_l is its ASCII-lowercase form.
pub(in crate::ui) fn load_mol_from_query(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    reset_cam: &mut bool,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    inp: &str,
    inp_l: &str,
    enter_pressed: bool,
) {
    // Molecules we ship with the application. Checked ahead of the remote databases below, since
    // these load instantly and without an internet connection, and Enter picks the best local match
    // over any of them. Matches are still drawn as buttons, so a remote lookup stays one click away.
    // Whether Enter does anything at all for this query; a shorter one it ignores. Nothing is
    // highlighted as the Enter target when it isn't one.
    let enter_live = inp.len() >= QUERY_ENTER_LEN_MIN;

    // DB integration disabled for now: the built-in ChEBI database is 2D-only, so molecules loaded
    // from it lack the 3D coordinates the rest of the app expects. Re-enable once we have a 3D
    // source. The query then falls straight through to the remote lookups below.
    // let common = query_common_db(
    //     state,
    //     scene,
    //     redraw,
    //     updates,
    //     ui,
    //     inp,
    //     enter_pressed,
    //     enter_live,
    // );
    let common = CommonDbOutcome::NoMatch;

    // Which remote lookup below Enter activates — `None` unless Enter is live for this query and the
    // built-in DB didn't already claim the key. Drives both the highlight and the Enter handling in
    // each branch, so exactly one lookup responds to the key.
    let enter_tgt = match common {
        CommonDbOutcome::Loaded => return,
        CommonDbOutcome::Matched => EnterTarget::None,
        CommonDbOutcome::NoMatch => match enter_live {
            true => enter_target(inp, inp_l),
            false => EnterTarget::None,
        },
    };

    // A prefixed ChEBI accession, e.g. `CHEBI:46195`. This names one database, so unlike the bare
    // number below it queries ChEBI alone, and nothing further down is offered for it.
    if inp_l.starts_with(CHEBI_PREFIX) {
        let is_tgt = enter_tgt == EnterTarget::ChebiId;
        let button_clicked = query_btn(ui, "Load ChEBI", is_tgt).clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            match chebi::parse_id(inp_l) {
                Ok(id) => load_chebi_id(state, scene, redraw, updates, id),
                Err(e) => handle_err(
                    &mut state.ui,
                    format!("{inp} is not a valid ChEBI accession: {e:?}"),
                ),
            }
        }

        return;
    }

    // PubChem CID. Don't return early here; continue to allow for other
    if let Ok(cid) = inp.parse::<u32>() {
        let is_tgt = enter_tgt == EnterTarget::PubchemCid;
        if query_btn(ui, "Load PubChem", is_tgt).clicked() || (enter_pressed && is_tgt) {
            match load_sdf_pubchem(cid) {
                Ok(downloaded) => {
                    let cid_key = cid.to_string();
                    let Some(cache_path) = cache_sdf_source(
                        state,
                        ManagedMolProvider::Pubchem,
                        &cid_key,
                        &cid_key,
                        &downloaded.source_text,
                    ) else {
                        return;
                    };
                    open_lig_from_input(state, downloaded.mol, Some(&cache_path), scene, updates);
                    redraw.ligand = true;

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

        // A bare number is also a ChEBI accession. PubChem owns Enter, being the larger database;
        // ChEBI is one click away, or unambiguous with a `chebi:` prefix.
        if query_btn(ui, "Load ChEBI", false).clicked() {
            load_chebi_id(state, scene, redraw, updates, cid);
        }
    }

    if inp.len() == 4 || inp_l.starts_with("pdb_") {
        // Enter only acts on a bare 4-character ident here, so a `pdb_`-prefixed one isn't a target.
        let is_tgt = enter_tgt == EnterTarget::Rcsb;
        let button_clicked = query_btn(ui, "Load RCSB", is_tgt).clicked();
        if (button_clicked || (enter_pressed && is_tgt)) && inp.len() == 4 {
            load_atom_coords_rcsb(inp_l, state, scene, updates, &mut redraw.peptide, reset_cam);

            state.ui.db_input = String::new();
            return;
        }

        return;
    }

    if inp.len() == 3 {
        let is_tgt = enter_tgt == EnterTarget::Geostd;
        let button_clicked = query_btn(ui, "Load Geostd", is_tgt).clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            state.load_geostd_mol_data(inp_l, true, true, updates, scene);

            state.ui.db_input = String::new();
        }

        return;
    }

    if inp.len() > 4 && inp_l.starts_with("db") {
        let is_tgt = enter_tgt == EnterTarget::DrugBank;
        let button_clicked = query_btn(ui, "Load DrugBank", is_tgt).clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            match load_sdf_drugbank(inp_l) {
                Ok(downloaded) => {
                    let Some(cache_path) = cache_sdf_source(
                        state,
                        ManagedMolProvider::Drugbank,
                        inp_l,
                        inp_l,
                        &downloaded.source_text,
                    ) else {
                        return;
                    };
                    open_lig_from_input(state, downloaded.mol, Some(&cache_path), scene, updates);
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

    // I believe this is cheap enough to run here (continuously)
    if is_smiles(inp) {
        let is_tgt = enter_tgt == EnterTarget::Smiles;
        let button_clicked = query_btn(ui, "Load from SMILES", is_tgt).clicked();
        // Attempt ot infer if this is SMILES.
        if (enter_pressed && is_tgt) || button_clicked {
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

                    let cache_result = managed_mols::store_sdf(
                        &state.volatile.prefs_dir,
                        ManagedMolProvider::Smiles,
                        &managed_mols::text_key(inp),
                        inp,
                        &mol.to_sdf(),
                    );
                    let Some(cache_path) = report_cache_result(state, cache_result) else {
                        return;
                    };
                    open_lig_from_input(state, mol, Some(&cache_path), scene, updates);
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
        let is_tgt = enter_tgt == EnterTarget::PubchemSearch;
        let button_clicked = query_btn(ui, "Search PubChem", is_tgt).clicked();
        if button_clicked || (enter_pressed && is_tgt) {
            let cids = find_cids_from_search(inp, false);

            match cids {
                Ok(c) => {
                    if c.is_empty() {
                        handle_success(&mut state.ui, "No results found on Pubchem".to_owned());
                    } else {
                        // todo: DRY with the other pubchem branch above.
                        match load_sdf_pubchem(c[0]) {
                            Ok(downloaded) => {
                                let cid_key = c[0].to_string();
                                let Some(cache_path) = cache_sdf_source(
                                    state,
                                    ManagedMolProvider::Pubchem,
                                    &cid_key,
                                    inp,
                                    &downloaded.source_text,
                                ) else {
                                    return;
                                };
                                open_lig_from_input(
                                    state,
                                    downloaded.mol,
                                    Some(&cache_path),
                                    scene,
                                    updates,
                                );
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
