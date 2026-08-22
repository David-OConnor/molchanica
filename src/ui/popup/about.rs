//! The About popup: what this program is, which version is running, where to find it online, and
//! where it keeps its files.

use std::io;

use egui::{Color32, CursorIcon, Label, RichText, Sense, Ui};

use crate::{
    VERSION, external_tools,
    state::State,
    ui::{
        COLOR_HIGHLIGHT, ROW_SPACING,
        popup::close_btn,
        util::{display_path, open_dir},
    },
};

const REPO_URL: &str = "https://github.com/David-OConnor/molchanica";
const HOME_URL: &str = "https://www.athanorlab.com/molchanica";

/// A text hyperlink: blue and underlined, and opens the user's browser when clicked. This is the
/// same mechanism we use for the external database links, e.g. `pubchem::open_overview`.
fn link(ui: &mut Ui, text: &str, url: &str) {
    let resp = ui
        .add(
            Label::new(RichText::new(text).color(COLOR_HIGHLIGHT).underline())
                .sense(Sense::click()),
        )
        .on_hover_cursor(CursorIcon::PointingHand);

    if resp.clicked()
        && let Err(e) = webbrowser::open(url)
    {
        eprintln!("Failed to open the web browser: {:?}", e);
    }
}

/// Show the one folder Molchanica keeps everything in: the preferences file,
/// `managed_molecules/`, `gpu_cache/`, and the optional tools under `process_executables/`.
///
/// The tools panel has the equivalent button for `process_executables/` alone; this is the folder
/// that contains it, and the executable.
fn open_data_folder() -> io::Result<()> {
    let root = external_tools::data_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "unable to determine Molchanica's data directory",
        )
    })?;

    open_dir(&root)
}

pub(in crate::ui::popup) fn about_window(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.heading(RichText::new("Molchanica").color(Color32::WHITE));
        ui.label(RichText::new(format!("v{VERSION}")).color(Color32::GRAY));
    });

    ui.add_space(ROW_SPACING);

    ui.label("Molecule editing, visualization, and dynamics.");

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.label("Home page:");
        link(ui, HOME_URL, HOME_URL);
    });

    ui.horizontal(|ui| {
        ui.label("Source code:");
        link(ui, REPO_URL, REPO_URL);
    });

    ui.add_space(ROW_SPACING);

    ui.label("Preferences, downloaded molecules, and installed tools all live in one folder:");

    ui.horizontal(|ui| {
        if ui.button("Open data folder").clicked() {
            state.ui.popup.about_folder_error =
                open_data_folder().err().map(|error| error.to_string());
        }

        // The path itself is worth showing even when the button works: it is the first thing to
        // ask for when someone reports a problem. Unabbreviated on hover, as elsewhere in the UI.
        if let Some(root) = external_tools::data_root() {
            ui.label(RichText::new(display_path(&root)).color(Color32::WHITE))
                .on_hover_text(root.to_string_lossy());
        }
    });

    if let Some(error) = &state.ui.popup.about_folder_error {
        ui.label(
            RichText::new(format!("Could not open the data folder: {error}"))
                .color(Color32::ORANGE)
                .small(),
        );
    }

    ui.add_space(ROW_SPACING);

    close_btn(ui, &mut state.ui.popup.about);
}
