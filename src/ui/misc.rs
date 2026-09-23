//! Misc utility-related UI functionality.

use egui::{
    Color32, CornerRadius, CursorIcon, Frame, Label, Margin, Response, RichText, Sense, Stroke, Ui,
};
const COLOR_SECTION_BOX: Color32 = Color32::from_rgb(100, 100, 140);

use crate::{
    ui::{COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE},
    util::RedrawFlags,
};

/// A box that shows its text highlighted if a flag is set.
pub fn toggle_btn_inv(
    val: &mut bool,
    text: &str,
    tooltip: &str,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    let color = active_color(!*val);
    if ui
        .button(RichText::new(text).color(color))
        .on_hover_text(tooltip)
        .clicked()
    {
        *val = !*val;
        // todo: Don't need to redraw everything.
        redraw.set_all();
    }
}

/// A box that shows its text highlighted if a flag is set.
pub fn toggle_btn(
    val: &mut bool,
    text: &str,
    tooltip: &str,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    let color = active_color(*val);
    if ui
        .button(RichText::new(text).color(color))
        .on_hover_text(tooltip)
        .clicked()
    {
        *val = !*val;
        // todo: Don't need to redraw everything.
        redraw.set_all();
    }
}

// #[derive(Clone, Copy, PartialEq)]
// pub enum MdMode {
//     Docking,
//     Peptide,
// }

pub fn active_color(val: bool) -> Color32 {
    if val { COLOR_ACTIVE } else { COLOR_INACTIVE }
}

// A container that highlights a section of UI code, to make it visually distinct from neighboring areas.
pub fn section_box() -> Frame {
    Frame::new()
        .stroke(Stroke::new(1.0, COLOR_SECTION_BOX))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 2))
        .outer_margin(Margin::symmetric(0, 0))
}

/// One option of a selector between mutually-exclusive, named options. Displayed as plain text,
/// highlighted and underlined when selected, so it's visually distinct from action buttons.
/// Chain `.on_hover_text()` and `.clicked()` on the result, as with a button.
pub fn selector_option(ui: &mut Ui, selected: bool, label: impl Into<String>) -> Response {
    let color = if selected {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    };

    let resp = ui
        .add(Label::new(RichText::new(label).color(color)).sense(Sense::click()))
        .on_hover_cursor(CursorIcon::PointingHand);

    let rect = resp.rect;
    let y = rect.bottom() + 1.;

    if selected {
        ui.painter()
            .hline(rect.x_range(), y, Stroke::new(2.0, COLOR_ACTIVE_RADIO));
    } else if resp.hovered() {
        ui.painter()
            .hline(rect.x_range(), y, Stroke::new(1.0, COLOR_INACTIVE));
    }

    resp
}

/// A selector between mutually-exclusive, named options. `options` is `(value, label, tooltip)`.
/// Returns the newly-selected value, if the user changed it.
pub fn selector<T: PartialEq + Copy>(
    ui: &mut Ui,
    current: T,
    options: &[(T, &str, &str)],
) -> Option<T> {
    let mut result = None;

    for (val, label, tooltip) in options {
        if selector_option(ui, *val == current, *label)
            .on_hover_text(*tooltip)
            .clicked()
            && *val != current
        {
            result = Some(*val);
        }
    }

    result
}
