//! A popup of a UI viewer and editor for molecule databases, e.g. as implemented
//! in parquet.

use std::{path::Path, slice, sync::mpsc, thread};

use egui::{Color32, Grid, RichText, ScrollArea, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    button, label,
    mol_db::{EnrichTarget, MolMeta, ParquetMolDb, run_pubchem_enrich},
    molecules::MoleculeGeneric,
    prefs::OpenType,
    state::{DbSel, PopupState, State},
    threads::DbEnrichJob,
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        popup,
    },
    util::{handle_err, handle_success},
};

/// Characters shown in a cell before it's truncated; the full text is available on hover.
const TITLE_CHARS_MAX: usize = 30;
const SMILES_CHARS_MAX: usize = 30;

// Fixed per-column widths for the molecule table. The narrow columns (Load, CID, Heavy atoms,
// Delete) are kept tight so the width they'd otherwise waste goes to Title and SMILES. The header
// and body are drawn as separate grids, so both must use these same widths to stay aligned.
const W_LOAD: f32 = 50.;
const W_CID: f32 = 90.;
const W_TITLE: f32 = 156.;
const W_SMILES: f32 = 210.;
const W_HEAVY: f32 = 84.;
const W_DELETE: f32 = 46.;

/// Rows shown per page in the molecule table.
const MOLS_PER_PAGE: usize = 40;

/// Characters of a database's file path shown in the "Databases loaded" list before its start is
/// elided; the full path is available on hover.
const PATH_TAIL_CHARS_MAX: usize = 44;

/// What a button in the molecule table asked for, by SMILES key. The table borrows the DB, so these
/// are acted on after it's drawn.
enum RowAction {
    Load(String),
    Delete(String),
}

pub(in crate::ui) fn parquet_db(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        label!(ui, "Molecule databases", Color32::WHITE);

        ui.add_space(COL_SPACING);

        if button!(
        ui,
        "Create DB",
        COLOR_ACTION,
        "Create a Parquet molecule database, for use with screening \
                algorithms. Saves it to disk. Much faster than screening a folder full of molecule files directly."
    )
            .clicked()
        {
            state.volatile.dialogs.parquet_db_save.save_file();
        }

        if button!(
        ui,
        "Load DB",
        COLOR_ACTION,
        "Load a Parquet molecule database. Can use this for screening, or add more molecules to it."
    )
            .clicked()
        {
            state.volatile.dialogs.parquet_db_load.pick_file();
        }

        ui.add_space(COL_SPACING);

        popup::close_btn(ui, &mut state.ui.popup.parquet_db);
    });

    db_selector(state, ui);

    // Display DB-specific data if there is an active one.
    if let Some(db_sel) = state.volatile.parquet_db_active {
        // The DB the selection points at may be gone: a loaded one was closed, or this build has no
        // embedded DB.
        let db_valid = match db_sel {
            DbSel::Common => state.mol_db.is_some(),
            DbSel::Loaded(i) => i < state.volatile.parquet_dbs.len(),
        };

        if !db_valid {
            handle_err(
                &mut state.ui,
                String::from("Error: Invalid Parquet DB active index"),
            );
            state.volatile.parquet_db_active = None;
            return;
        }

        ui.add_space(ROW_SPACING);

        // The built-in DB is fixed at compile time; only the loaded ones can be added to.
        let editable = matches!(db_sel, DbSel::Loaded(_));

        // A PubChem-population job running for *this* DB shows progress in place of the edit
        // controls and pauses editing: the worker rewrites the file when it finishes, so a
        // concurrent add or delete would be clobbered. Carries `(done, total)` for the display.
        let enrich_status = match &state.volatile.thread_receivers.db_pubchem_enrich {
            Some(job) if job.db_sel == db_sel => Some((job.done, job.total)),
            _ => None,
        };
        let can_edit = editable && enrich_status.is_none();

        // Keep the worker's channel (polled in `handle_thread_rx`) drained promptly even if nothing
        // else is driving repaints.
        if enrich_status.is_some() {
            ui.ctx().request_repaint();
        }

        // Gathered up front: the button row below borrows `state` mutably. Only ligands not already
        // in the DB get a button. (Index into `state.ligands`, and display name.)
        let ligs_to_add: Vec<(usize, String)> = match state.active_mol_db() {
            Some(db) if can_edit => state
                .ligands
                .iter()
                .enumerate()
                .filter(|(_, lig)| !db.contains_mol(lig))
                .map(|(i, lig)| (i, lig.common.name(Some(&lig.idents))))
                .collect(),
            _ => Vec::new(),
        };

        // Index into `state.ligands` of the ligand whose button was clicked, if any.
        let mut add_lig = None;
        // Set when the "Fill from PubChem" button is clicked; acted on after the row is drawn.
        let mut start_enrich = false;

        ui.horizontal_wrapped(|ui| {
            if let Some((done, total)) = enrich_status {
                label!(
                    ui,
                    format!(
                        "Populating from PubChem… {done} / {total}. Editing is paused until this \
                         finishes."
                    ),
                    COLOR_HIGHLIGHT
                );
            } else if editable {
                if button!(
                    ui,
                    "Add mols from dir",
                    COLOR_ACTION,
                    "Add all molecules in a folder (Mol2 or SDF, recursively) to this database."
                )
                .clicked()
                {
                    state.volatile.dialogs.parquet_mols_dir.pick_directory();
                }

                if button!(
                    ui,
                    "Add mol[s] from file",
                    COLOR_ACTION,
                    "Add a single molecule file (Mol2 or SDF) or multi-mol SDF file to this database."
                )
                .clicked()
                {
                    state.volatile.dialogs.parquet_mol_file.pick_file();
                }

                if button!(
                    ui,
                    "Fill titles/CIDs from PubChem",
                    COLOR_ACTION,
                    "Look up any missing PubChem titles and CIDs for the molecules in this \
                     database, and save them. Runs in the background, rate-limited; molecules that \
                     already have both are skipped."
                )
                .clicked()
                {
                    start_enrich = true;
                }
            } else {
                label!(
                    ui,
                    "This database ships with the application; it can't be modified.",
                    Color32::GRAY
                );
            }

            ui.add_space(COL_SPACING);

            // Placed ahead of the per-ligand buttons below, which vary in number, so the search box
            // doesn't move around as ligands are opened and closed.
            search_input(&mut state.ui.popup, ui);

            ui.add_space(COL_SPACING);

            for (i, name) in &ligs_to_add {
                if button!(
                    ui,
                    format!("Add {name}"),
                    COLOR_ACTION,
                    "Add this open molecule to the database."
                )
                .clicked()
                {
                    add_lig = Some((*i, name.clone()));
                }
            }
        });

        if let Some((i, name)) = add_lig
            && let DbSel::Loaded(db_i) = db_sel
        {
            let mol = state.ligands[i].clone();

            let db = &mut state.volatile.parquet_dbs[db_i];
            let result = db.add_mols(&[mol]);
            let mol_count = db.index_meta.len();

            match result {
                Ok(()) => handle_success(
                    &mut state.ui,
                    format!("Added {name} to the database ({mol_count} molecules)"),
                ),
                Err(e) => handle_err(
                    &mut state.ui,
                    format!("Error adding {name} to the database: {e}"),
                ),
            }
        }

        if start_enrich
            && let DbSel::Loaded(db_i) = db_sel
        {
            let db = &state.volatile.parquet_dbs[db_i];

            // Only rows actually missing something; molecules with both a title and CID are skipped.
            let targets: Vec<EnrichTarget> = db
                .index_meta
                .values()
                .filter(|m| m.pubchem_title.is_none() || m.pubchem_cid.is_none())
                .map(|m| EnrichTarget {
                    smiles: m.smiles.clone(),
                    cid: m.pubchem_cid,
                })
                .collect();

            if targets.is_empty() {
                handle_success(
                    &mut state.ui,
                    "Every molecule already has a PubChem title and CID.".to_owned(),
                );
            } else {
                let total = targets.len();
                let source = db.source.clone();

                let (tx, rx) = mpsc::channel();
                thread::spawn(move || run_pubchem_enrich(source, targets, tx));

                state.volatile.thread_receivers.db_pubchem_enrich = Some(DbEnrichJob {
                    db_sel,
                    rx,
                    done: 0,
                    total,
                });

                handle_success(
                    &mut state.ui,
                    format!("Looking up PubChem data for {total} molecule(s)…"),
                );
            }
        }

        //     if button!(
        //         ui,
        //         "Load all mols",
        //         COLOR_ACTION,
        //         "Load all molecule data in this database into memory"
        //     )
        //     .clicked()
        //     {
        //         match db.load_all() {
        //             Ok(mols) => {
        //                 // todo temp
        //                 for mol in &mols {
        //                     println!("MOL loaded: {:?}", mol);
        //                 }
        //             }
        //             Err(e) => {
        //                 handle_err(&mut state.ui, format!("Error loading molecules: {e:?}"));
        //             }
        //         }
        //     }

        del_confirmation(state, db_sel, ui);

        // Selected by field here rather than through `State::active_mol_db`, which borrows all of
        // `state`; the table also needs `state.ui.popup` mutably, and these fields are disjoint.
        let db = match db_sel {
            DbSel::Common => state.mol_db.as_ref(),
            DbSel::Loaded(i) => state.volatile.parquet_dbs.get(i),
        };

        let Some(db) = db else {
            return;
        };

        let table_action = db_summary_table(db, db_sel, can_edit, &mut state.ui.popup, ui);

        match table_action {
            Some(RowAction::Delete(smiles)) => {
                if let DbSel::Loaded(db_i) = db_sel {
                    state.ui.popup.parquet_db_mol_del = Some((db_i, smiles));
                }
            }
            Some(RowAction::Load(smiles)) => load_mol_from_db(state, &smiles, scene, updates),
            None => (),
        }
    }
}

/// Open a molecule from the DB as a ligand. `mol_data` and the idents + metadata are separate
/// columns, so both are read here; see the `mol_db` module docs.
fn load_mol_from_db(
    state: &mut State,
    smiles: &str,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    let mol = {
        let Some(db) = state.active_mol_db() else {
            return;
        };

        match db.load_mol(smiles) {
            Ok(mut mol) => {
                // A DB written before idents were stored has none to fold in; the molecule itself
                // is still fine.
                if let Err(e) = db.apply_idents_meta(slice::from_mut(&mut mol)) {
                    eprintln!("Error loading idents for {smiles}: {e}");
                }
                mol
            }
            Err(e) => {
                handle_err(
                    &mut state.ui,
                    format!("Error loading {smiles} from the database: {e}"),
                );
                return;
            }
        }
    };

    state.load_mol_to_state(MoleculeGeneric::Small(mol), scene, updates, None);
}

/// Shown while a delete requested from the table below is pending: deleting a molecule rewrites the
/// DB file, so we confirm first. `active` is the DB the table is showing.
fn del_confirmation(state: &mut State, active: DbSel, ui: &mut Ui) {
    let Some((db_i, smiles)) = state.ui.popup.parquet_db_mol_del.clone() else {
        return;
    };

    // The request is stale: the user selected or closed a DB while it was pending. Deletes are only
    // ever queued against a loaded DB; the built-in one is read-only.
    if active != DbSel::Loaded(db_i) || db_i >= state.volatile.parquet_dbs.len() {
        state.ui.popup.parquet_db_mol_del = None;
        return;
    }

    ui.add_space(ROW_SPACING);

    ui.horizontal_wrapped(|ui| {
        label!(
            ui,
            format!("Delete {smiles} from this database?"),
            Color32::LIGHT_RED
        );

        ui.add_space(COL_SPACING);

        if button!(
            ui,
            "Delete",
            Color32::LIGHT_RED,
            "Remove this molecule from the database. This rewrites the database file."
        )
        .clicked()
        {
            let db = &mut state.volatile.parquet_dbs[db_i];
            let result = db.remove_mol(&smiles);
            let mol_count = db.index_meta.len();

            match result {
                Ok(()) => handle_success(
                    &mut state.ui,
                    format!("Deleted {smiles} from the database ({mol_count} molecules)"),
                ),
                Err(e) => handle_err(
                    &mut state.ui,
                    format!("Error deleting {smiles} from the database: {e}"),
                ),
            }

            state.ui.popup.parquet_db_mol_del = None;
        }

        if button!(ui, "Cancel", COLOR_ACTION, "Keep this molecule.").clicked() {
            state.ui.popup.parquet_db_mol_del = None;
        }
    });

    ui.add_space(ROW_SPACING);
}

/// Lists the databases available, and lets one be selected as active, or closed. The built-in
/// database is listed alongside the ones the user has opened; it can be selected and viewed, but
/// not closed or modified.
pub(in crate::ui) fn db_selector(state: &mut State, ui: &mut Ui) {
    ui.add_space(ROW_SPACING);
    ui.separator();
    label!(ui, "Databases loaded", Color32::GRAY);

    // Only present if this build has one embedded; see `State::mol_db`. It has no file path (it's
    // baked into the binary), so unlike the loaded DBs below there's no path tail to show.
    if let Some(db) = &state.mol_db {
        let name = db.name();
        let mol_count = db.index_meta.len();
        let active = state.volatile.parquet_db_active == Some(DbSel::Common);

        ui.horizontal(|ui| {
            label!(
                ui,
                format!("{} : {mol_count} mols", db_display_name(&name)),
                Color32::WHITE
            );

            let color = if active { COLOR_ACTIVE } else { COLOR_INACTIVE };

            ui.add_space(COL_SPACING);

            if button!(
                ui,
                "Select",
                color,
                "Select this as the active database. This one ships with the application, and is \
                 read-only."
            )
            .clicked()
            {
                select_db(state, DbSel::Common);
            }
        });
    }

    // Applied after the loop: it borrows `parquet_dbs`, and selecting touches all of `state`.
    let mut select = None;

    let mut close_db = None;
    for (i, db) in state.volatile.parquet_dbs.iter().enumerate() {
        let active = state.volatile.parquet_db_active == Some(DbSel::Loaded(i));

        ui.horizontal(|ui| {
            label!(
                ui,
                format!(
                    "{} : {} mols",
                    db_display_name(&db.name()),
                    db.index_meta.len()
                ),
                Color32::WHITE
            );

            // The tail of the path it was loaded from, to tell apart DBs whose filenames match but
            // live in different folders. Full path on hover.
            if let Some(path) = db.path() {
                ui.add_space(COL_SPACING);
                label!(ui, truncate_path_tail(path, PATH_TAIL_CHARS_MAX), Color32::GRAY)
                    .on_hover_text(path.to_string_lossy());
            }

            let color = if active { COLOR_ACTIVE } else { COLOR_INACTIVE };

            ui.add_space(COL_SPACING);

            if button!(
                ui,
                "Select",
                color,
                "Select this as the active database. View or modify it."
            )
            .clicked()
            {
                select = Some(DbSel::Loaded(i));
            }

            if button!(ui, "Close", Color32::LIGHT_RED, "Close this database").clicked() {
                close_db = Some(i);

                // Make sure this doesn't cause a jump in the selected DB.
                match state.volatile.parquet_db_active {
                    Some(DbSel::Loaded(i_active)) if i_active == i => {
                        state.volatile.parquet_db_active = None;
                    }
                    Some(DbSel::Loaded(i_active)) if i_active > i => {
                        state.volatile.parquet_db_active = Some(DbSel::Loaded(i_active - 1));
                    }
                    _ => (),
                }

                // The embedded DB has no path, and was never in the open history.
                if let Some(path) = db.path() {
                    for history in &mut state.to_save.open_history {
                        if OpenType::ParquetDb == history.type_ && history.path == path {
                            history.last_session = false;
                        }
                    }
                }
            }
        });
    }

    if let Some(sel) = select {
        select_db(state, sel);
    }

    if let Some(i) = close_db {
        state.volatile.parquet_dbs.remove(i);
        state.update_save_prefs(); // to save teh history change.
    }
}

/// Make `sel` the active database, resetting the table state that referred to the previous one.
fn select_db(state: &mut State, sel: DbSel) {
    state.volatile.parquet_db_active = Some(sel);

    // The search and page refer to the DB we were showing, not this one.
    state.ui.popup.parquet_db_search.clear();
    state.ui.popup.parquet_db_page = 0;
}

/// A table of the molecules in a database. Shows the lightweight index columns only; `mol_data`,
/// `idents` and `metadata` are loaded on demand, and aren't in memory here.
///
/// Returns the row button the user clicked, if any; see `RowAction`. A deletion is confirmed before
/// it's applied; see `del_confirmation`.
fn db_summary_table(
    db: &ParquetMolDb,
    db_sel: DbSel,
    editable: bool,
    popup: &mut PopupState,
    ui: &mut Ui,
) -> Option<RowAction> {
    let index_meta = &db.index_meta;

    // Said explicitly rather than drawing nothing: an empty built-in DB means the parquet file
    // embedded at build time had no rows, which is otherwise indistinguishable from a UI bug.
    if index_meta.is_empty() {
        ui.add_space(ROW_SPACING);
        label!(ui, "This database contains no molecules.", Color32::GRAY);
        return None;
    }

    // Distinguishes this table's egui ids from the other DBs'.
    let id = match db_sel {
        DbSel::Common => "common".to_owned(),
        DbSel::Loaded(i) => i.to_string(),
    };

    let mut action = None;

    ui.add_space(ROW_SPACING);

    // Rebuild the sorted + search-filtered key list only when an input changes; collecting,
    // sorting, and filtering the whole index every frame makes a large DB hang, since egui redraws
    // the open popup continuously. The per-page rows below look their metadata up from `index_meta`
    // by key, which is cheap. See `MolDbTableView`.
    let search = popup.parquet_db_search.trim().to_lowercase();
    let view = &mut popup.parquet_db_view;
    if view.db_sel != Some(db_sel) || view.mol_count != index_meta.len() || view.search != search {
        let mut entries: Vec<&MolMeta> = if search.is_empty() {
            index_meta.values().collect()
        } else {
            index_meta
                .values()
                .filter(|meta| meta.matches_search(&search))
                .collect()
        };

        // Sort by SMILES (the row key) for a stable display order; `index_meta` is a HashMap.
        entries.sort_by(|a, b| a.smiles.cmp(&b.smiles));

        view.keys = entries
            .into_iter()
            .map(|meta| meta.smiles.clone())
            .collect();
        view.db_sel = Some(db_sel);
        view.mol_count = index_meta.len();
        view.search = search;
    }

    let keys = &popup.parquet_db_view.keys;
    let n_matching = keys.len();

    // The DB may have shrunk (a delete), or the search narrowed, since the page was set.
    let pages = n_matching.div_ceil(MOLS_PER_PAGE).max(1);
    if popup.parquet_db_page >= pages {
        popup.parquet_db_page = pages - 1;
    }
    let page = popup.parquet_db_page;

    if keys.is_empty() {
        label!(ui, "No molecules match this search.", Color32::GRAY);
        return None;
    }

    // A read-only DB has no delete column.
    let cols = if editable { 6 } else { 5 };

    // Column headers (outside the scroll area so they stay fixed).
    Grid::new(format!("parquet_mol_headers_{id}"))
        .num_columns(cols)
        .min_col_width(0.)
        .spacing([COL_SPACING, 4.])
        .show(ui, |ui| {
            cell(ui, W_LOAD, |ui| {
                label!(ui, "Load", Color32::GRAY);
            });
            cell(ui, W_CID, |ui| {
                label!(ui, "CID", Color32::GRAY);
            });
            cell(ui, W_TITLE, |ui| {
                label!(ui, "Title", Color32::GRAY);
            });
            cell(ui, W_SMILES, |ui| {
                label!(ui, "SMILES", Color32::GRAY);
            });
            cell(ui, W_HEAVY, |ui| {
                label!(ui, "Heavy atoms", Color32::GRAY);
            });
            if editable {
                cell(ui, W_DELETE, |ui| {
                    label!(ui, "Delete", Color32::GRAY);
                });
            }
            ui.end_row();
        });

    ui.separator();

    ScrollArea::vertical()
        .id_salt(format!("parquet_mol_list_{id}"))
        .min_scrolled_height(400.)
        .max_height(800.)
        .show(ui, |ui| {
            Grid::new(format!("parquet_mol_grid_{id}"))
                .num_columns(cols)
                .striped(true)
                .min_col_width(0.)
                .spacing([COL_SPACING, 4.])
                .show(ui, |ui| {
                    for smiles in keys.iter().skip(page * MOLS_PER_PAGE).take(MOLS_PER_PAGE) {
                        // The key came from `index_meta`, so this lookup succeeds; skip
                        // defensively rather than unwrap should the two ever drift.
                        let Some(meta) = index_meta.get(smiles) else {
                            continue;
                        };

                        if cell(ui, W_LOAD, |ui| {
                            button!(
                                ui,
                                "Load",
                                COLOR_ACTION,
                                "Open this molecule from the database as a ligand."
                            )
                            .clicked()
                        }) {
                            action = Some(RowAction::Load(smiles.clone()));
                        }

                        cell(ui, W_CID, |ui| match meta.pubchem_cid {
                            Some(cid) => {
                                label!(ui, cid.to_string(), Color32::LIGHT_BLUE);
                            }
                            None => {
                                label!(ui, "—", Color32::DARK_GRAY);
                            }
                        });

                        cell(ui, W_TITLE, |ui| match &meta.pubchem_title {
                            Some(title) => {
                                label!(ui, truncate(title, TITLE_CHARS_MAX), Color32::LIGHT_BLUE);
                            }
                            None => {
                                label!(ui, "—", Color32::DARK_GRAY);
                            }
                        });

                        cell(ui, W_SMILES, |ui| {
                            label!(ui, truncate(&meta.smiles, SMILES_CHARS_MAX), Color32::GRAY)
                                .on_hover_text(&meta.smiles);
                        });

                        cell(ui, W_HEAVY, |ui| {
                            label!(ui, meta.heavy_atom_count.to_string(), Color32::GRAY);
                        });

                        if editable
                            && cell(ui, W_DELETE, |ui| {
                                button!(
                                    ui,
                                    "X",
                                    Color32::LIGHT_RED,
                                    "Delete this molecule from the database. Asks for confirmation \
                                     first."
                                )
                                .clicked()
                            })
                        {
                            action = Some(RowAction::Delete(smiles.clone()));
                        }

                        ui.end_row();
                    }
                });
        });

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.add_enabled_ui(page > 0, |ui| {
            if ui
                .button(RichText::new("◀").color(COLOR_HIGHLIGHT))
                .clicked()
            {
                popup.parquet_db_page -= 1;
            }
        });

        ui.label(format!("Page {} / {pages}", page + 1));

        ui.add_enabled_ui(page + 1 < pages, |ui| {
            if ui
                .button(RichText::new("▶").color(COLOR_HIGHLIGHT))
                .clicked()
            {
                popup.parquet_db_page += 1;
            }
        });

        ui.add_space(COL_SPACING);

        label!(ui, format!("{} molecules", n_matching), Color32::GRAY);
    });

    action
}

/// The search box for the molecule table; filters on CID, title, or SMILES as the user types.
/// Drawn in the row of DB buttons, above the table it applies to.
fn search_input(popup: &mut PopupState, ui: &mut Ui) {
    label!(ui, "Search", Color32::GRAY);

    if ui
        .add(
            TextEdit::singleline(&mut popup.parquet_db_search)
                .desired_width(240.)
                .hint_text("CID, title, or SMILES"),
        )
        .changed()
    {
        // The result set changes as they type, so the page they were on is meaningless.
        popup.parquet_db_page = 0;
    }

    if !popup.parquet_db_search.is_empty()
        && button!(ui, "Clear", COLOR_ACTION, "Clear the search text.").clicked()
    {
        popup.parquet_db_search.clear();
        popup.parquet_db_page = 0;
    }
}

/// Lay out one table cell at a fixed width. This is what keeps the narrow columns narrow: egui's
/// `Grid` otherwise sizes every column to `min_col_width`, so the short Load/CID/Heavy/Delete
/// cells would claim as much room as Title and SMILES. Allocating an exact width per cell also
/// keeps the separate header and body grids aligned.
fn cell<R>(ui: &mut Ui, width: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ui.spacing().interact_size.y),
        egui::Layout::left_to_right(egui::Align::Center),
        add,
    )
    .inner
}

fn truncate(val: &str, len_max: usize) -> String {
    if val.chars().count() > len_max {
        let v: String = val.chars().take(len_max).collect();
        format!("{v}...")
    } else {
        val.to_owned()
    }
}

/// A database's name for the list, without the trailing `.parquet` extension. Names without it
/// (e.g. the built-in DB's) are returned unchanged.
fn db_display_name(name: &str) -> &str {
    const EXT: &str = ".parquet";
    if name.len() >= EXT.len() && name[name.len() - EXT.len()..].eq_ignore_ascii_case(EXT) {
        &name[..name.len() - EXT.len()]
    } else {
        name
    }
}

/// A file path shortened to its last `len_max` characters, with a leading "..." when elided (e.g.
/// "...molchanica/my_db.parquet"). Keeps the informative tail (folder + filename) rather than the
/// long absolute prefix.
fn truncate_path_tail(path: &Path, len_max: usize) -> String {
    let full = path.to_string_lossy();
    let chars: Vec<char> = full.chars().collect();

    if chars.len() <= len_max {
        return full.into_owned();
    }

    let tail: String = chars[chars.len() - (len_max - 3)..].iter().collect();
    format!("...{tail}")
}
