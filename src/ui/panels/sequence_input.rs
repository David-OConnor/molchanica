use std::sync::Arc;

use egui::{
    Button, Color32, FontId, Galley, RichText, Ui,
    text::{LayoutJob, TextFormat},
};

use crate::ui::{COLOR_ACTIVE, COLOR_INACTIVE};

pub(super) const INVALID_LETTER_COLOR: Color32 = Color32::from_rgb(255, 165, 0);

pub(super) fn layout(
    ui: &Ui,
    text: &str,
    wrap_width: f32,
    font_id: FontId,
    color: impl Fn(char) -> Color32,
) -> Arc<Galley> {
    let mut job = LayoutJob::default();
    job.wrap.max_width = wrap_width;
    job.keep_trailing_whitespace = true;

    for ch in text.chars() {
        job.append(
            ch.encode_utf8(&mut [0; 4]),
            0.,
            TextFormat::simple(font_id.clone(), color(ch)),
        );
    }

    ui.fonts_mut(|fonts| fonts.layout_job(job))
}

pub(super) fn letter_counts(text: &str, is_valid: impl Fn(char) -> bool) -> (usize, usize) {
    let valid = text.chars().filter(|&ch| is_valid(ch)).count();

    (valid, text.chars().count() - valid)
}

pub(super) fn status_label(ui: &mut Ui, valid: usize, invalid: usize, residue: &str) {
    if invalid > 0 {
        let suffix = if invalid == 1 { "" } else { "s" };
        ui.colored_label(
            INVALID_LETTER_COLOR,
            format!("{invalid} invalid {residue} letter{suffix}"),
        );
    } else if valid > 0 {
        let suffix = if valid == 1 { "" } else { "s" };
        ui.colored_label(COLOR_ACTIVE, format!("{valid} {residue}{suffix}"));
    }
}

pub(super) fn create_button(label: &'static str, color: Color32, enabled: bool) -> Button<'static> {
    let color = if enabled { color } else { COLOR_INACTIVE };

    Button::new(RichText::new(label).color(color))
}
