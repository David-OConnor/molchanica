use std::{
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use egui::{Align, Color32, Grid, Layout, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::MolType;

use crate::{
    file_io::managed_mols,
    prefs::OpenType,
    state::State,
    ui::{COL_SPACING, COLOR_ACTION, COLOR_HIGHLIGHT, ROW_SPACING, popup},
    util::{handle_err, handle_success},
};

/// Number of recent-file rows shown per page in the recent-files popup.
pub const PER_PAGE: usize = 15;

// todo: Return a color too
pub fn recentness_descrip(age_min: i64) -> (String, Color32) {
    match age_min {
        0..=30 => ("Active".to_string(), Color32::GREEN),
        31..=480 => ("Today".to_string(), Color32::LIGHT_GREEN),
        481..=10_080 => ("Recent".to_string(), Color32::YELLOW),
        _ => ("Older".to_string(), Color32::GRAY),
    }
}

/// Plural, human-readable category names, for summarizing pruned history entries.
fn type_descrip_plural(type_: OpenType) -> &'static str {
    match type_ {
        OpenType::Peptide => "proteins",
        OpenType::Ligand => "small molecules",
        OpenType::NucleicAcid => "nucleic acids",
        OpenType::Lipid => "lipids",
        OpenType::Pocket => "pockets",
        OpenType::Map => "density maps",
        OpenType::Frcmod => "force field files",
        OpenType::Trajectory => "MD trajectories",
        OpenType::MdParams => "MD param files",
        OpenType::ParquetDb => "molecule DBs",
        OpenType::MdMols => "MD mol files",
    }
}

/// Drop history entries whose file is no longer on disk, and report what was dropped, by
/// category. Returns whether anything was removed.
fn clear_removed(state: &mut State) {
    // Counts per category, in the order first encountered.
    let mut removed: Vec<(OpenType, usize)> = Vec::new();
    let prefs_dir = state.volatile.prefs_dir.clone();

    state.to_save.open_history.retain(|history| {
        // Only prune when we can confirm the file is gone; e.g. a permissions error, or a
        // temporarily-unavailable network drive, shouldn't drop an entry.
        if !matches!(fs::exists(&history.path), Ok(false)) {
            return true;
        }
        if let Err(error) = managed_mols::remove_entry(&prefs_dir, &history.path) {
            eprintln!("Unable to remove invalid managed molecule cache entry: {error}");
        }

        match removed
            .iter_mut()
            .find(|(type_, _)| *type_ == history.type_)
        {
            Some((_, count)) => *count += 1,
            None => removed.push((history.type_, 1)),
        }

        false
    });

    if removed.is_empty() {
        handle_success(
            &mut state.ui,
            "No missing files in recent history".to_owned(),
        );
        return;
    }

    let descrips: Vec<String> = removed
        .iter()
        .map(|(type_, count)| format!("{count} stale {}", type_descrip_plural(*type_)))
        .collect();

    handle_success(&mut state.ui, format!("Removed {}", descrips.join(", ")));

    // Persist the pruned history.
    state.update_save_prefs();
}

pub(in crate::ui::popup) fn recent_files_popup(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    let mut open = None;
    let now = Utc::now();

    // Title and filename filter.
    ui.horizontal(|ui| {
        ui.heading(RichText::new("Recent molecules").color(Color32::WHITE));
        ui.add_space(COL_SPACING);

        if ui
            .add(
                TextEdit::singleline(&mut state.ui.popup.recent_files_filter)
                    .desired_width(200.)
                    .hint_text("Filter by filename"),
            )
            .changed()
        {
            // Jump back to the first page whenever the filter changes.
            state.ui.popup.recent_files_page = 0;
        }
    });

    ui.add_space(ROW_SPACING);

    // Owned display data for the rows matching the filter, most recent first. Collecting
    // owned values (rather than borrowing `open_history`) lets us open a file below without
    // a borrow conflict.
    struct Row {
        path: PathBuf,
        filename: String,
        color: Color32,
        type_str: String,
        ident_str: String,
        descrip: String,
        descrip_color: Color32,
    }

    let filter = state.ui.popup.recent_files_filter.to_lowercase();

    let rows: Vec<Row> = state
        .to_save
        .open_history
        .iter()
        .rev()
        .filter_map(|file| {
            let filename = Path::new(&file.path).file_name()?.to_str()?.to_string();
            let ident = file.ident.clone().unwrap_or_default();

            // Filter matches on either the filename or the molecule identifier.
            if !filter.is_empty()
                && !filename.to_lowercase().contains(&filter)
                && !ident.to_lowercase().contains(&filter)
            {
                return None;
            }

            // Truncate long identifiers for display.
            const MAX_IDENT_LEN: usize = 20;
            let ident_str = if ident.chars().count() > MAX_IDENT_LEN {
                let truncated: String = ident.chars().take(MAX_IDENT_LEN).collect();
                format!("{truncated}...")
            } else {
                ident
            };

            let (r, g, b) = match file.type_ {
                OpenType::Peptide => MolType::Peptide.color(),
                OpenType::Ligand => MolType::Ligand.color(),
                OpenType::NucleicAcid => MolType::NucleicAcid.color(),
                OpenType::Lipid => MolType::Lipid.color(),
                _ => (255, 255, 255),
            };

            let elapsed_minutes = (now - file.timestamp).num_minutes();
            let (descrip, descrip_color) = recentness_descrip(elapsed_minutes);

            Some(Row {
                path: file.path.clone(),
                filename,
                color: Color32::from_rgb(r, g, b),
                type_str: file.type_.to_string(),
                ident_str,
                descrip,
                descrip_color,
            })
        })
        .collect();

    // Clamp the current page; the filtered list may have shrunk since the last frame.
    let num_pages = rows.len().div_ceil(PER_PAGE).max(1);
    if state.ui.popup.recent_files_page >= num_pages {
        state.ui.popup.recent_files_page = num_pages - 1;
    }
    let page = state.ui.popup.recent_files_page;

    Grid::new("recent_files_grid")
        .num_columns(4)
        .spacing([COL_SPACING, 4.])
        .show(ui, |ui| {
            for row in rows.iter().skip(page * PER_PAGE).take(PER_PAGE) {
                // todo: Make the whole row clickable?
                if ui
                    .button(RichText::new(&row.filename).color(row.color))
                    .clicked()
                {
                    open = Some(row.path.clone());
                }

                ui.label(RichText::new(&row.ident_str).color(Color32::WHITE));

                // Right-align the type and recentness columns.
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new(&row.type_str));
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new(&row.descrip).color(row.descrip_color));
                });

                ui.end_row();
            }
        });

    if let Some(path) = open {
        if state.open_file(&path, scene, engine_updates).is_err() {
            // The file is missing or otherwise unopenable; drop it from history so it doesn't
            // linger in this list, and persist the pruned history.
            state.to_save.open_history.retain(|h| h.path != path);
            if let Err(error) = managed_mols::remove_entry(&state.volatile.prefs_dir, &path) {
                eprintln!("Unable to remove invalid managed molecule cache entry: {error}");
            }

            state.update_save_prefs();

            handle_err(
                &mut state.ui,
                format!("Problem opening file {path:?}. Removed from recent history"),
            );
        }
        state.ui.popup.recent_files = false;
    }

    ui.add_space(ROW_SPACING);

    // Close button and pagination controls.
    ui.horizontal(|ui| {
        popup::close_btn(ui, &mut state.ui.popup.recent_files);

        ui.add_space(COL_SPACING);

        if ui
            .button(RichText::new("Clear removed").color(COLOR_ACTION))
            .on_hover_text("Remove entries whose file is no longer present on disk.")
            .clicked()
        {
            clear_removed(state);
        }

        ui.add_space(COL_SPACING);

        ui.add_enabled_ui(page > 0, |ui| {
            if ui
                .button(RichText::new("◀").color(COLOR_HIGHLIGHT))
                .clicked()
            {
                state.ui.popup.recent_files_page -= 1;
            }
        });

        ui.label(format!("Page {} / {num_pages}", page + 1));

        ui.add_enabled_ui(page + 1 < num_pages, |ui| {
            if ui
                .button(RichText::new("▶").color(COLOR_HIGHLIGHT))
                .clicked()
            {
                state.ui.popup.recent_files_page += 1;
            }
        });
    });
}
