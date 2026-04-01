//! Misc utility-related UI functionality.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Stroke, Ui};
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

/// Visually distinct; fore buttons that operate as radio buttons
pub fn active_color_sel(val: bool) -> Color32 {
    if val {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    }
}

// A container that highlights a section of UI code, to make it visually distinct from neighboring areas.
pub fn section_box() -> Frame {
    Frame::new()
        .stroke(Stroke::new(1.0, COLOR_SECTION_BOX))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 2))
        .outer_margin(Margin::symmetric(0, 0))
}
