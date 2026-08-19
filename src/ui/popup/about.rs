//! The About popup: what this program is, which version is running, and where to find it online.

use egui::{Color32, CursorIcon, Label, RichText, Sense, Ui};

use crate::{
    VERSION,
    state::State,
    ui::{COLOR_HIGHLIGHT, ROW_SPACING, popup::close_btn},
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

    close_btn(ui, &mut state.ui.popup.about);
}
