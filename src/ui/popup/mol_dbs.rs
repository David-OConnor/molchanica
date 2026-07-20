//! A popup of a UI viewer and editor for molecule databases, e.g. as implemented
//! in parquet.

use std::slice;

use egui::{Color32, Grid, RichText, ScrollArea, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    button, label,
    mol_db::{MolMeta, ParquetMolDb},
    molecules::MoleculeGeneric,
    prefs::OpenType,
    state::{DbSel, PopupState, State},
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        popup,
    },
    util::{handle_err, handle_success},
};

/// Characters shown in a cell before it's truncated; the full text is available on hover.
const SMILES_CHARS_MAX: usize = 10;

/// Rows shown per page in the molecule table.
const MOLS_PER_PAGE: usize = 40;

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

        // Gathered up front: the button row below borrows `state` mutably. Only ligands not already
        // in the DB get a button. (Index into `state.ligands`, and display name.)
        let ligs_to_add: Vec<(usize, String)> = match state.active_mol_db() {
            Some(db) if editable => state
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

        ui.horizontal_wrapped(|ui| {
            if editable {
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
                    "Add mol",
                    COLOR_ACTION,
                    "Add a single molecule file (Mol2 or SDF) to this database."
                )
                .clicked()
                {
                    state.volatile.dialogs.parquet_mol_file.pick_file();
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

        let table_action = db_summary_table(db, db_sel, editable, &mut state.ui.popup, ui);

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

    // Only present if this build has one embedded; see `State::mol_db`.
    if let Some(db) = &state.mol_db {
        let name = db.name();
        let mol_count = db.index_meta.len();
        let active = state.volatile.parquet_db_active == Some(DbSel::Common);

        ui.horizontal(|ui| {
            label!(ui, format!("{name} : {mol_count} mols"), Color32::WHITE);

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
                format!("{} : {} mols", db.name(), db.index_meta.len()),
                Color32::WHITE
            );

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

    // Sort by ident for stable display order.
    let mut entries: Vec<(&String, &MolMeta)> = index_meta.iter().collect();
    entries.sort_by_key(|(ident, _)| ident.as_str());

    let search = popup.parquet_db_search.trim().to_lowercase();
    if !search.is_empty() {
        entries.retain(|(_, meta)| meta.matches_search(&search));
    }

    // The DB may have shrunk (a delete), or the search narrowed, since the page was set.
    let pages = entries.len().div_ceil(MOLS_PER_PAGE).max(1);
    if popup.parquet_db_page >= pages {
        popup.parquet_db_page = pages - 1;
    }
    let page = popup.parquet_db_page;

    if entries.is_empty() {
        label!(ui, "No molecules match this search.", Color32::GRAY);
        return None;
    }

    // A read-only DB has no delete column.
    let cols = if editable { 6 } else { 5 };

    // Column headers (outside the scroll area so they stay fixed).
    Grid::new(format!("parquet_mol_headers_{id}"))
        .num_columns(cols)
        .min_col_width(120.)
        .spacing([COL_SPACING, 4.])
        .show(ui, |ui| {
            label!(ui, "Load", Color32::GRAY);
            label!(ui, "CID", Color32::GRAY);
            label!(ui, "Title", Color32::GRAY);
            label!(ui, "SMILES", Color32::GRAY);
            label!(ui, "Heavy atoms", Color32::GRAY);
            if editable {
                label!(ui, "Delete", Color32::GRAY);
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
                .min_col_width(120.)
                .spacing([COL_SPACING, 4.])
                .show(ui, |ui| {
                    for (smiles, meta) in
                        entries.iter().skip(page * MOLS_PER_PAGE).take(MOLS_PER_PAGE)
                    {
                        if button!(
                            ui,
                            "Load",
                            COLOR_ACTION,
                            "Open this molecule from the database as a ligand."
                        )
                        .clicked()
                        {
                            action = Some(RowAction::Load((*smiles).clone()));
                        }

                        match meta.pubchem_cid {
                            Some(cid) => {
                                label!(ui, cid.to_string(), Color32::LIGHT_BLUE)
                            }
                            None => label!(ui, "—", Color32::DARK_GRAY),
                        };

                        match &meta.pubchem_title {
                            Some(title) => {
                                label!(ui, title, Color32::LIGHT_BLUE)
                            }
                            None => label!(ui, "—", Color32::DARK_GRAY),
                        };

                        label!(ui, truncate(&meta.smiles, SMILES_CHARS_MAX), Color32::GRAY)
                            .on_hover_text(&meta.smiles);

                        label!(ui, meta.heavy_atom_count.to_string(), Color32::GRAY);

                        if editable
                            && button!(
                                ui,
                                "X",
                                Color32::LIGHT_RED,
                                "Delete this molecule from the database. Asks for confirmation \
                                 first."
                            )
                            .clicked()
                        {
                            action = Some(RowAction::Delete((*smiles).clone()));
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

        label!(ui, format!("{} molecules", entries.len()), Color32::GRAY);
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

fn truncate(val: &str, len_max: usize) -> String {
    if val.chars().count() > len_max {
        let v: String = val.chars().take(len_max).collect();
        format!("{v}...")
    } else {
        val.to_owned()
    }
}
