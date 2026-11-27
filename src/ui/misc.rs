//! Misc utility-related UI functionality.

use egui::{
    AsIdSalt, Color32, CornerRadius, CursorIcon, Frame, Id, Label, Margin, Rect, Response,
    RichText, ScrollArea, Sense, Slider, Stroke, TextEdit, Ui, UiBuilder, vec2,
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

    let id_salt = options
        .iter()
        .fold(Id::new("selector"), |id, (_, label, _)| id.with(label));

    wrap_together(ui, id_salt, |ui| {
        selector_box().show(ui, add_options);
    });

    result
}

/// Lay out one row of a panel no further right than `right`: the panel's edge, taken before
/// anything was added to it. When anything overflows a UI, egui widens that UI to match for the
/// rest of the frame. Without this, every row after one that overflowed would wrap past the edge
/// of the window too, hiding the items at its end.
pub fn bounded_row<R>(ui: &mut Ui, right: f32, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    let mut rect = ui.available_rect_before_wrap();
    rect.max.x = rect.max.x.min(right);

    ui.scope_builder(UiBuilder::new().max_rect(rect), add_contents)
        .inner
}

/// In a wrapping layout (`horizontal_wrapped`), keep a few items together on one line, moving
/// them to the next row as a unit if they don't fit on this one. E.g. a label and its dropdown or
/// slider. Elsewhere, this just adds them. `id_salt` must be unique within the row.
///
/// egui's `ComboBox` and `Slider`, like frames, are containers rather than widgets, so they don't
/// wrap on their own; see `wrapped_section`.
pub fn wrap_together<R>(
    ui: &mut Ui,
    id_salt: impl AsIdSalt,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> R {
    if ui.layout().main_wrap() && ui.layout().is_horizontal() {
        wrapped_section(ui, id_salt, |ui| ui.horizontal(add_contents).inner)
    } else {
        add_contents(ui)
    }
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

/// A resizable text box is never dragged shorter than this.
const RESIZE_MIN_HEIGHT: f32 = 40.;

/// A text box in its own scroll area, with a bar beneath it to drag its height by, like a
/// resizable text area. Until dragged, it fits its text, up to `fit_height`; double-clicking the
/// bar returns it to that. Dragging stops at `max_height`. The dragged height is kept under `id`,
/// for the session.
pub fn resizable_text_edit(
    ui: &mut Ui,
    id: Id,
    editor: TextEdit,
    fit_height: f32,
    max_height: f32,
) -> Response {
    let max_height = max_height.max(RESIZE_MIN_HEIGHT);
    let dragged: Option<f32> = ui.data(|data| data.get_temp(id));
    let height = match dragged {
        Some(height) => height.clamp(RESIZE_MIN_HEIGHT, max_height),
        None => fit_height.min(max_height),
    };

    let output = ScrollArea::vertical()
        .id_salt(id.with("scroll"))
        .max_height(height)
        .auto_shrink([false, dragged.is_none()])
        .show(ui, |ui| {
            if dragged.is_some() {
                // Fill the area, rather than end in blank space below short text.
                ui.add(editor.min_size(vec2(0., height)))
            } else {
                ui.add(editor)
            }
        });
    let area = output.inner_rect;

    let handle = resize_handle(ui, id.with("handle"));
    if handle.dragged()
        && let Some(pointer) = handle.interact_pointer_pos()
    {
        // Keep the bar's middle under the pointer.
        let offset = handle.rect.center().y - area.bottom();
        let height = (pointer.y - offset - area.top()).clamp(RESIZE_MIN_HEIGHT, max_height);
        ui.data_mut(|data| data.insert_temp(id, height));
    }
    if handle.double_clicked() {
        ui.data_mut(|data| data.remove::<f32>(id));
    }

    output.inner
}

/// A full-width bar, with a grip at its middle, to drag the height of what's above it by.
pub fn resize_handle(ui: &mut Ui, id: Id) -> Response {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 8.), Sense::hover());
    let response = ui
        .interact(rect, id, Sense::click_and_drag())
        .on_hover_and_drag_cursor(CursorIcon::ResizeVertical)
        .on_hover_text("Drag to resize. Double-click to fit the text again.");

    let active = response.hovered() || response.dragged();
    let visuals = ui.style().interact(&response);
    let line = if active {
        visuals.fg_stroke
    } else {
        ui.visuals().widgets.noninteractive.bg_stroke
    };
    let grip = if active {
        visuals.fg_stroke.color
    } else {
        visuals.bg_fill
    };
    let center = rect.center();

    let painter = ui.painter();
    painter.hline(rect.x_range(), center.y, line);
    painter.rect_filled(Rect::from_center_size(center, vec2(36., 4.)), 2., grip);

    response
}
