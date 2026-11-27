//! Misc utility-related UI functionality.

use egui::{
    AsIdSalt, Color32, CornerRadius, CursorIcon, Frame, Id, Label, Margin, Response, RichText,
    Sense, Slider, Stroke, Ui,
};
const COLOR_SECTION_BOX: Color32 = Color32::from_rgb(100, 100, 140);
/// Unselected options of a selector; a slight orange hue distinguishes them from plain labels.
const COLOR_SELECTOR_UNSEL: Color32 = Color32::from_rgb(210, 175, 140);
const COLOR_SELECTOR_BOX: Color32 = Color32::from_rgb(110, 95, 80);

use crate::{
    ui::{COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE},
    util::RedrawFlags,
};

pub(crate) fn opacity_slider(ui: &mut Ui, opacity: &mut f32) -> bool {
    ui.add(Slider::new(opacity, 0.08..=1.).text("Opacity"))
        .changed()
}

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

/// A box around a group of `selector_option`s, e.g.
/// `selector_box().show(ui, |ui| { /* selector_option calls */ });`
pub fn selector_box() -> Frame {
    Frame::new()
        .stroke(Stroke::new(1.0, COLOR_SELECTOR_BOX))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(Margin {
            left: 5,
            right: 5,
            top: 1,
            bottom: 3,
        })
}

/// One option of a selector between mutually-exclusive, named options. Displayed as text:
/// highlighted and underlined when selected, and slightly orange otherwise, so it's visually
/// distinct from action buttons and labels. Group these in a `selector_box`.
/// Chain `.on_hover_text()` and `.clicked()` on the result, as with a button.
pub fn selector_option(ui: &mut Ui, selected: bool, label: impl Into<String>) -> Response {
    let color = if selected {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_SELECTOR_UNSEL
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
            .hline(rect.x_range(), y, Stroke::new(1.0, COLOR_SELECTOR_UNSEL));
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

    let add_options = |ui: &mut Ui| {
        for (val, label, tooltip) in options {
            if selector_option(ui, *val == current, *label)
                .on_hover_text(*tooltip)
                .clicked()
                && *val != current
            {
                result = Some(*val);
            }
        }
    };

    // In a wrapping row, keep the options together on one line, and move the whole box to the
    // next row if it doesn't fit. Otherwise, the box is squeezed into what's left of the row.
    if ui.layout().main_wrap() && ui.layout().is_horizontal() {
        let id_salt = options
            .iter()
            .fold(Id::new("selector"), |id, (_, label, _)| id.with(label));

        wrapped_section(ui, id_salt, |ui| {
            selector_box().show(ui, |ui| ui.horizontal(add_options));
        });
    } else {
        selector_box().show(ui, add_options);
    }

    result
}

/// Draw a section, e.g. a framed group of widgets, as one item of a wrapping layout
/// (`horizontal_wrapped`), starting a new row if it won't fit on this one.
///
/// egui wraps widgets, but not containers: it doesn't know their size until they're drawn, so
/// it squeezes them into what's left of the row, and they overflow it if they can't shrink that
/// far. An overflowing item also widens the UI it's in, so every later row then wraps past the
/// edge of the window. We use the section's width from the previous frame.
pub fn wrapped_section<R>(
    ui: &mut Ui,
    id_salt: impl AsIdSalt,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> R {
    let id = ui.id().with(id_salt);
    let width_prev: Option<f32> = ui.data(|d| d.get_temp(id));

    // The same test egui uses for widgets. Ending a row with nothing on it would leave a gap.
    let row_start = ui.cursor().left() <= ui.max_rect().left();

    if !row_start && width_prev.is_some_and(|w| w > ui.available_size_before_wrap().x) {
        ui.end_row();
    }

    let resp = ui.scope(add_contents);
    ui.data_mut(|d| d.insert_temp(id, resp.response.rect.width()));

    resp.inner
}
