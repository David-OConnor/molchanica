//! The About popup: what this program is, which version is running, whether there's a newer one,
//! where to find it online, and where it keeps its files.

use std::{io, time::Duration};

use egui::{Color32, CursorIcon, Label, ProgressBar, RichText, Sense, Ui};
use software_manager::{Application, UpdateStatus};

use crate::{
    VERSION, external_tools,
    state::State,
    ui::{
        COLOR_HIGHLIGHT, ROW_SPACING,
        util::{display_path, open_dir},
    },
};

const REPO_URL: &str = "https://github.com/David-OConnor/molchanica";
const HOME_URL: &str = "https://www.athanorlab.com/molchanica";
pub(in crate::ui) const DOCS_URL: &str = "https://www.athanorlab.com/docs/index.html";

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

/// Check GitHub for a newer release, download it, and install it. Installing waits for Molchanica
/// to close, since its files can't be replaced while it runs; the release's own setup script then
/// installs it, and Molchanica reopens.
fn update_section(state: &mut State, ui: &mut Ui) {
    let updater = &mut state.ui.popup.updater;
    updater.poll();

    if updater.busy() {
        // So the download progress moves.
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }

    let mut check = false;
    let mut download = false;
    let mut install = false;

    ui.horizontal(|ui| match updater.status() {
        UpdateStatus::Idle => {
            check = ui.button("Check for updates").clicked();
        }
        UpdateStatus::Checking(_) => {
            ui.spinner();
            ui.label("Checking for updates…");
        }
        UpdateStatus::UpToDate => {
            ui.label(RichText::new("This is the latest version.").color(Color32::WHITE));
            check = ui.button("Check again").clicked();
        }
        UpdateStatus::Available(release) => {
            ui.label(
                RichText::new(format!("Version {} is available.", release.version))
                    .color(Color32::WHITE),
            );
            link(ui, "Release notes", &release.page_url);

            download = ui
                .button("Download and install")
                .on_hover_text(format!(
                    "Download the update ({:.0} MB). Installing it is the next step.",
                    release.size as f64 / 1_000_000.
                ))
                .clicked();
        }
        UpdateStatus::Downloading {
            release, progress, ..
        } => {
            ui.add(
                ProgressBar::new(progress.fraction())
                    .text(format!("Downloading {}…", release.version))
                    .desired_width(300.),
            );
        }
        UpdateStatus::Ready(staged) => {
            ui.label(format!("Version {} is downloaded.", staged.release.version));

            install = ui
                .button(RichText::new("Restart and install").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Close Molchanica, install the update, then reopen it. Preferences are saved \
                    first; save any other work before this.",
                )
                .clicked();
        }
        UpdateStatus::Installing => {
            ui.label("Installing…");
        }
        UpdateStatus::Failed(e) => {
            ui.label(
                RichText::new(format!("Unable to update: {e}"))
                    .color(Color32::ORANGE)
                    .small(),
            );
            check = ui.button("Try again").clicked();
        }
    });

    let ctx = ui.ctx().clone();
    let repaint = move || ctx.request_repaint();

    if check {
        updater.check(Application::Molchanica, VERSION, repaint);
    } else if download {
        updater.download(repaint);
    } else if install && updater.install_on_exit() {
        // The graphics engine doesn't let the UI close the window, so exit directly. The
        // installer is waiting for this process to end.
        state.update_save_prefs();
        std::process::exit(0);
    }
}

pub(in crate::ui::popup) fn about_window(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.heading(RichText::new("Molchanica").color(Color32::WHITE));
        ui.label(RichText::new(format!("v{VERSION}")).color(Color32::GRAY));
    });

    ui.add_space(ROW_SPACING);

    ui.label("Molecule editing, visualization, and dynamics.");

    ui.add_space(ROW_SPACING);

    update_section(state, ui);

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.label("Home page:");
        link(ui, HOME_URL, HOME_URL);
    });

    ui.horizontal(|ui| {
        ui.label("Documentation:");
        link(ui, DOCS_URL, DOCS_URL);
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
}
