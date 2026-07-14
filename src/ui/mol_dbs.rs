//! A popup of a UI viewer and editor for molecule databases, e.g. as implemented
//! in parquet.

use egui::{Color32, Grid, ScrollArea, Ui};

use crate::{
    button, label,
    mol_db::{MolMeta, ParquetMolDb},
    molecules::MolIdent,
    prefs::OpenType,
    state::State,
    ui::{COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_INACTIVE, ROW_SPACING, popups},
    util::{handle_err, handle_success},
};

/// Characters shown in a cell before it's truncated; the full text is available on hover.
const SMILES_CHARS_MAX: usize = 10;
const IDENTS_CHARS_MAX: usize = 32;

pub(in crate::ui) fn parquet_db(state: &mut State, ui: &mut Ui) {
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

        popups::close_btn(ui, &mut state.ui.popup.parquet_db);
    });

    db_selector(state, ui);

    // Display DB-specific data if there is an active one.
    if let Some(db_i) = state.volatile.parquet_db_active {
        if db_i >= state.volatile.parquet_dbs.len() {
            handle_err(
                &mut state.ui,
                String::from("Error: Invalid Parquet DB active index"),
            );
            return;
        }

        ui.add_space(ROW_SPACING);

        // Gathered up front: the button row below borrows `state` mutably. Only ligands not already
        // in the DB get a button. (Index into `state.ligands`, and display name.)
        let db = &state.volatile.parquet_dbs[db_i];
        let ligs_to_add: Vec<(usize, String)> = state
            .ligands
            .iter()
            .enumerate()
            .filter(|(_, lig)| !db.contains_mol(lig))
            .map(|(i, lig)| (i, lig.common.name(Some(&lig.idents))))
            .collect();

        // Index into `state.ligands` of the ligand whose button was clicked, if any.
        let mut add_lig = None;

        ui.horizontal_wrapped(|ui| {
            if button!(
                ui,
                "Add mols from dir",
                COLOR_ACTION,
                "Add all molecules in a folder (Mol2 or SDF, recursively) to this database."
            )
            .clicked()
            {
                state.volatile.dialogs.parquet_mols_dir.pick_directory();
                state.volatile.parquet_db_active = Some(db_i);
            }

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

        if let Some((i, name)) = add_lig {
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

        del_confirmation(state, db_i, ui);

        if let Some(smiles) = db_summary_table(&mut state.volatile.parquet_dbs[db_i], db_i, ui) {
            state.ui.popup.parquet_db_mol_del = Some((db_i, smiles));
        }
    }
}

/// Shown while a delete requested from the table below is pending: deleting a molecule rewrites the
/// DB file, so we confirm first. `db_i_active` is the DB the table is showing.
fn del_confirmation(state: &mut State, db_i_active: usize, ui: &mut Ui) {
    let Some((db_i, smiles)) = state.ui.popup.parquet_db_mol_del.clone() else {
        return;
    };

    // The request is stale: the user selected or closed a DB while it was pending.
    if db_i != db_i_active || db_i >= state.volatile.parquet_dbs.len() {
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

/// Lists the open databases, and lets one be selected as active, or closed.
pub(in crate::ui) fn db_selector(state: &mut State, ui: &mut Ui) {
    ui.add_space(ROW_SPACING);
    ui.separator();
    label!(ui, "Databases loaded", Color32::GRAY);

    let mut close_db = None;
    for (i, db) in state.volatile.parquet_dbs.iter().enumerate() {
        let Some(file_name) = db.path.file_name() else {
            eprintln!("Error loading file path");
            continue;
        };

        let file_name = file_name.to_string_lossy();

        let active = state.volatile.parquet_db_active == Some(i);

        ui.horizontal(|ui| {
            label!(
                ui,
                format!("{file_name} : {} mols", db.index_meta.len()),
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
                state.volatile.parquet_db_active = Some(i);
            }

            if button!(ui, "Close", Color32::LIGHT_RED, "Close this database").clicked() {
                close_db = Some(i);

                // Make sure this doesn't cause a jump in the selected DB.
                if let Some(i_active) = state.volatile.parquet_db_active {
                    if i_active == i {
                        state.volatile.parquet_db_active = None;
                    } else if i_active > i {
                        state.volatile.parquet_db_active = Some(i_active - 1);
                    }
                }

                for history in &mut state.to_save.open_history {
                    if OpenType::ParquetDb == history.type_ && history.path == db.path {
                        history.last_session = false;
                    }
                }
            }
        });
    }

    if let Some(i) = close_db {
        state.volatile.parquet_dbs.remove(i);
        state.update_save_prefs(); // to save teh history change.
    }
}

/// A table of the molecules in a database. Shows the lightweight index columns, and each molecule's
/// idents; `mol_data` and `metadata` are loaded on demand, and aren't in memory here.
///
/// Returns the SMILES key of the molecule the user asked to delete, if any. The deletion itself is
/// confirmed first; see `del_confirmation`.
fn db_summary_table(db: &mut ParquetMolDb, db_i: usize, ui: &mut Ui) -> Option<String> {
    let (index_meta, idents) = db.index_and_idents();

    if index_meta.is_empty() {
        return None;
    }

    let mut del_req = None;

    ui.add_space(ROW_SPACING);

    // Sort by ident for stable display order.
    let mut entries: Vec<(&String, &MolMeta)> = index_meta.iter().collect();
    entries.sort_by_key(|(ident, _)| ident.as_str());

    // Column headers (outside the scroll area so they stay fixed).
    Grid::new(format!("parquet_mol_headers_{db_i}"))
        .num_columns(5)
        .min_col_width(120.)
        .spacing([COL_SPACING, 4.])
        .show(ui, |ui| {
            label!(ui, "PubChem CID", Color32::GRAY);
            label!(ui, "SMILES", Color32::GRAY);
            label!(ui, "Heavy atoms", Color32::GRAY);
            label!(ui, "Identifiers", Color32::GRAY);
            label!(ui, "Delete", Color32::GRAY);
            ui.end_row();
        });

    ui.separator();

    ScrollArea::vertical()
        .id_salt(format!("parquet_mol_list_{db_i}"))
        .min_scrolled_height(400.)
        .max_height(800.)
        .show(ui, |ui| {
            Grid::new(format!("parquet_mol_grid_{db_i}"))
                .num_columns(5)
                .striped(true)
                .min_col_width(120.)
                .spacing([COL_SPACING, 4.])
                .show(ui, |ui| {
                    for (smiles, meta) in &entries {
                        match meta.pubchem_cid {
                            Some(cid) => {
                                label!(ui, cid.to_string(), Color32::LIGHT_BLUE)
                            }
                            None => label!(ui, "—", Color32::DARK_GRAY),
                        };

                        label!(ui, truncate(&meta.smiles, SMILES_CHARS_MAX), Color32::GRAY)
                            .on_hover_text(&meta.smiles);

                        label!(ui, meta.heavy_atom_count.to_string(), Color32::GRAY);

                        match idents.get(*smiles) {
                            Some(ids) if !ids.is_empty() => {
                                let text = idents_text(ids);
                                label!(ui, truncate(&text, IDENTS_CHARS_MAX), Color32::GRAY)
                                    .on_hover_text(text);
                            }
                            _ => {
                                label!(ui, "—", Color32::DARK_GRAY);
                            }
                        }

                        if button!(
                            ui,
                            "X",
                            Color32::LIGHT_RED,
                            "Delete this molecule from the database. Asks for confirmation first."
                        )
                        .clicked()
                        {
                            del_req = Some((*smiles).clone());
                        }

                        ui.end_row();
                    }
                });
        });

    del_req
}

/// E.g. "PubChem CID: 702, SMILES: CCO".
fn idents_text(idents: &[MolIdent]) -> String {
    idents
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn truncate(val: &str, len_max: usize) -> String {
    if val.chars().count() > len_max {
        let v: String = val.chars().take(len_max).collect();
        format!("{v}...")
    } else {
        val.to_owned()
    }
}
