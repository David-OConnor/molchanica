use std::{
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use egui::{Align, Color32, Grid, Layout, RichText, TextEdit, TextStyle, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::MolType;

use crate::{
    file_io::managed_mols,
    prefs::OpenType,
    state::State,
    ui::{COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING},
    util::{handle_err, handle_success},
};

/// Vertical gap between rows of the file list.
const ROW_GAP: f32 = 4.;

/// The order in which file-type filter buttons are displayed.
const TYPES_DISPLAY_ORDER: [OpenType; 12] = [
    OpenType::Peptide,
    OpenType::Ligand,
    OpenType::NucleicAcid,
    OpenType::Lipid,
    OpenType::Pocket,
    OpenType::Sequence,
    OpenType::Map,
    OpenType::Trajectory,
    OpenType::MdMols,
    OpenType::MdParams,
    OpenType::Frcmod,
    OpenType::ParquetDb,
];

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
        OpenType::Sequence => "sequence files",
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
        ui.heading(RichText::new("Recent files").color(Color32::WHITE));
        ui.add_space(COL_SPACING);

        if ui
            .add(
                TextEdit::singleline(&mut state.ui.popup.recent_files_filter)
                    .desired_width(200.)
                    .hint_text("Filter"),
            )
            .changed()
        {
            // Jump back to the first page whenever the filter changes.
            state.ui.popup.recent_files_page = 0;
        }
    });

    ui.add_space(ROW_SPACING);
    type_filter(state, ui);
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
    let hidden_types = &state.ui.popup.recent_files_hidden_types;

    let rows: Vec<Row> = state
        .to_save
        .open_history
        .iter()
        .rev()
        .filter(|file| !hidden_types.contains(&file.type_))
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

    // Fit as many rows as the window's height allows. We fix the row height (to a button's
    // natural height) so this is exact; content taller than the window would make it grow.
    let row_height = {
        let spacing = ui.spacing();

        (ui.text_style_height(&TextStyle::Button) + 2. * spacing.button_padding.y)
            .max(spacing.interact_size.y)
            .ceil()
    };
    // Space below the list: Item spacing after the grid, then the pagination controls.
    let footer_height = ROW_SPACING + row_height;
    let list_height = ui.available_height() - ui.spacing().item_spacing.y - footer_height;

    let per_page = (((list_height + ROW_GAP) / (row_height + ROW_GAP)).floor() as usize).max(1);

    // Clamp the current page; the filtered list, or the window, may have shrunk since the last
    // frame.
    let num_pages = rows.len().div_ceil(per_page).max(1);
    if state.ui.popup.recent_files_page >= num_pages {
        state.ui.popup.recent_files_page = num_pages - 1;
    }
    let page = state.ui.popup.recent_files_page;

    Grid::new("recent_files_grid")
        .num_columns(4)
        .spacing([COL_SPACING, ROW_GAP])
        .min_row_height(row_height)
        .show(ui, |ui| {
            for row in rows.iter().skip(page * per_page).take(per_page) {
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

    // Keep the pagination controls at the bottom of the window, e.g. on a partial last page.
    let pad = ui.available_height() - footer_height;
    if pad > 0. {
        ui.add_space(pad);
    }

    ui.add_space(ROW_SPACING);

    // Pagination controls.
    ui.horizontal(|ui| {
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

/// Toggle buttons for filtering by file type. Only types present in the history are shown.
fn type_filter(state: &mut State, ui: &mut Ui) {
    let mut changed = false;

    ui.horizontal_wrapped(|ui| {
        for type_ in TYPES_DISPLAY_ORDER {
            let count = state
                .to_save
                .open_history
                .iter()
                .filter(|h| h.type_ == type_)
                .count();

            if count == 0 {
                continue;
            }

            let hidden = &mut state.ui.popup.recent_files_hidden_types;
            let shown = !hidden.contains(&type_);
            let color = if shown { COLOR_ACTIVE } else { COLOR_INACTIVE };

            let resp = ui
                .button(RichText::new(format!("{type_} ({count})")).color(color))
                .on_hover_text(format!(
                    "Show or hide {} in this list. Right-click to show only this type.",
                    type_descrip_plural(type_)
                ));

            if resp.clicked() {
                if shown {
                    hidden.push(type_);
                } else {
                    hidden.retain(|t| *t != type_);
                }
                changed = true;
            } else if resp.secondary_clicked() {
                *hidden = TYPES_DISPLAY_ORDER
                    .into_iter()
                    .filter(|t| *t != type_)
                    .collect();
                changed = true;
            }
        }

        if !state.ui.popup.recent_files_hidden_types.is_empty()
            && ui
                .button(RichText::new("All").color(COLOR_HIGHLIGHT))
                .on_hover_text("Show all file types.")
                .clicked()
        {
            state.ui.popup.recent_files_hidden_types.clear();
            changed = true;
        }
    });

    if changed {
        // Jump back to the first page whenever the filter changes.
        state.ui.popup.recent_files_page = 0;
    }
}
