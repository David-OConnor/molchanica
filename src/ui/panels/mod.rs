use crate::drawing::color_viridis;
use crate::selection::Selection;
use crate::state::AaSeqDisplayCache;
use egui::text::LayoutJob;
use egui::text_selection::LabelSelectionState;
use egui::{Button, Color32, Frame, Pos2, Rect, Sense, Stroke, TextFormat, TextStyle, Ui, vec2};
use std::collections::HashSet;
use std::sync::Arc;

pub mod md;
pub mod md_viewer;
pub mod mol_data;
pub mod orca;
pub mod view;

/// Label on the button that copies the amino acid sequence to the clipboard.
const SEQ_COPY_TEXT: &str = "Copy seq";
/// Horizontal gap between the final residue and the copy button sharing its line.
const SEQ_COPY_PAD: f32 = 40.;
/// The sequence gets its own background strip, so residue colors are lifted against a known
/// value rather than whatever the panel happens to be.
const SEQ_BG: Color32 = Color32::from_gray(38);
/// Padding, in points, of `SEQ_BG` around the sequence text.
const SEQ_BG_PAD: f32 = 3.;
/// The selected residues. Red text alone is hard to pick out of a colored sequence.
const SEQ_SELECTED_BG: Color32 = Color32::from_rgb(170, 25, 25);
/// Minimum WCAG contrast ratio each residue's color is lifted to, against `SEQ_BG`.
const SEQ_MIN_CONTRAST: f32 = 3.5;
/// Extra space between residues, in points. Adjacent Viridis colors are nearly identical, so a
/// gap does as much for legibility as the letterforms do.
const SEQ_LETTER_SPACING: f32 = 1.2;
/// Faux-bold: egui's default fonts ship no bold face and aren't variable, so the glyphs are
/// painted a second time, offset by this many points, to thicken the strokes.
const SEQ_BOLD_OFFSET: f32 = 0.6;

/// WCAG relative luminance of an sRGB triple in `[0, 1]`.
fn luminance(color: (f32, f32, f32)) -> f32 {
    let lin = |v: f32| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };

    0.2126 * lin(color.0) + 0.7152 * lin(color.1) + 0.0722 * lin(color.2)
}

fn contrast(a: f32, b: f32) -> f32 {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    (hi + 0.05) / (lo + 0.05)
}

/// Blend `color` toward white until it clears `SEQ_MIN_CONTRAST` against `SEQ_BG`.
///
/// Viridis sweeps the whole luminance range, from a near-black purple to a bright yellow, so no
/// single background is readable across all of it: a grey light enough for the purple start
/// washes out the yellow tail, and its middle -- the teals -- lands on top of any grey in
/// between. A dark strip plus a floor under the dark end fixes both. Only the purple and blue
/// opening is touched; from the teals on, the color is raw Viridis, matching the 3D ribbon.
/// Lifting is one-directional so the sequence still reads as a single gradient, dark to bright.
fn lift_contrast(color: (f32, f32, f32)) -> (f32, f32, f32) {
    let bg = luminance((
        SEQ_BG.r() as f32 / 255.,
        SEQ_BG.g() as f32 / 255.,
        SEQ_BG.b() as f32 / 255.,
    ));
    if contrast(luminance(color), bg) >= SEQ_MIN_CONTRAST {
        return color;
    }

    let blend = |t: f32| {
        (
            color.0 + (1. - color.0) * t,
            color.1 + (1. - color.1) * t,
            color.2 + (1. - color.2) * t,
        )
    };

    // Bisect for the smallest blend that clears the threshold, so the hue is diluted as little
    // as it can be. Contrast against a darker background rises monotonically with `t`.
    let (mut lo, mut hi) = (0., 1.);
    for _ in 0..12 {
        let mid = (lo + hi) / 2.;
        if contrast(luminance(blend(mid)), bg) >= SEQ_MIN_CONTRAST {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    blend(hi)
}

/// The display for the amino acid sequence of an opened protein.
///
/// The colored sequence is one cached galley instead of one widget and color calculation per
/// residue on every frame. It is rebuilt only when its inputs actually change.
///
/// `res_indices` maps each position in `seq_text` to its residue index in the peptide.
pub(in crate::ui) fn pepide_aa_seq(
    selection: &mut Selection,
    seq_text: &str,
    res_indices: &[usize],
    cache: &mut AaSeqDisplayCache,
    ui: &mut Ui,
    redraw: &mut bool,
) {
    let selected: Vec<usize> = match selection {
        Selection::Residue(index) => vec![*index],
        Selection::Residues(indices) => indices.clone(),
        _ => Vec::new(),
    };
    let font_id = TextStyle::Body.resolve(ui.style());

    // The copy button shares the sequence's last line, so reserve its width before laying the
    // text out; every row then breaks early enough that the button can't overrun the panel.
    // `Ui::put` draws it at exactly this size, so size it the way the default style does: the
    // text plus `button_padding` on each side.
    let btn_text_size = ui
        .fonts_mut(|fonts| {
            fonts.layout_no_wrap(SEQ_COPY_TEXT.to_owned(), font_id.clone(), Color32::WHITE)
        })
        .size();

    let btn_padding = ui.spacing().button_padding;
    let btn_size = vec2(
        btn_text_size.x + 2. * btn_padding.x,
        (btn_text_size.y + 2. * btn_padding.y).max(ui.spacing().interact_size.y),
    );

    let wrap_width = (ui.available_width() - btn_size.x - SEQ_COPY_PAD).max(1.0);
    let pixels_per_point = ui.ctx().pixels_per_point();

    let rebuild = cache.dirty
        || cache.selected != selected
        || cache.font_id.as_ref() != Some(&font_id)
        || cache.wrap_width.to_bits() != wrap_width.to_bits()
        || cache.pixels_per_point.to_bits() != pixels_per_point.to_bits()
        || cache.galley.is_none();

    if rebuild {
        let len = seq_text.len(); // One ASCII character per residue.
        let selected_set: HashSet<usize> = selected.iter().copied().collect();
        let mut job = LayoutJob::default();
        job.wrap.max_width = wrap_width;
        job.wrap.break_anywhere = true;

        for (index, amino_acid) in seq_text.chars().enumerate() {
            let is_selected = res_indices
                .get(index)
                .is_some_and(|res_i| selected_set.contains(res_i));

            let (color, background) = if is_selected {
                (Color32::WHITE, SEQ_SELECTED_BG)
            } else {
                let color = lift_contrast(color_viridis(index, 0, len));
                (
                    Color32::from_rgb(
                        (color.0 * 255.0) as u8,
                        (color.1 * 255.0) as u8,
                        (color.2 * 255.0) as u8,
                    ),
                    Color32::TRANSPARENT,
                )
            };
            let mut encoded = [0; 4];
            job.append(
                amino_acid.encode_utf8(&mut encoded),
                0.0,
                TextFormat {
                    font_id: font_id.clone(),
                    extra_letter_spacing: SEQ_LETTER_SPACING,
                    color,
                    background,
                    ..Default::default()
                },
            );
        }

        cache.galley = Some(ui.fonts_mut(|fonts| fonts.layout_job(job)));
        cache.dirty = false;
        cache.selected = selected;
        cache.font_id = Some(font_id);
        cache.wrap_width = wrap_width;
        cache.pixels_per_point = pixels_per_point;
    }

    // Cloned out of the cache (an `Arc` bump) so the drag handling below can still borrow the
    // cache mutably.
    let Some(galley) = cache.galley.clone() else {
        return;
    };

    Frame::new().show(ui, |ui| {
        // Sit the button just past the final residue, on the line it ends on. The allocation
        // spans both, so the widgets below don't overlap the button.
        let last_row = galley
            .rows
            .last()
            .map(|row| row.rect())
            .unwrap_or(Rect::ZERO);
        let btn_offset = vec2(
            last_row.right() + SEQ_COPY_PAD,
            // Centered on the line it shares, but never above the galley: a single-row sequence
            // is shorter than the button.
            (last_row.center().y - btn_size.y / 2.).max(0.),
        );
        let size = vec2(
            galley.size().x.max(btn_offset.x + btn_size.x),
            galley.size().y.max(btn_offset.y + btn_size.y),
        );

        // `click_and_drag`, so a drag across the sequence selects text the way it does in a
        // normal label; a plain click still picks a single residue.
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        // Only behind the text: the copy button keeps the panel's own background.
        ui.painter().rect_filled(
            Rect::from_min_size(rect.min, galley.size()).expand(SEQ_BG_PAD),
            3.,
            SEQ_BG,
        );
        // Faux-bold, under the real glyphs; see `SEQ_BOLD_OFFSET`.
        ui.painter().galley(
            rect.min + vec2(SEQ_BOLD_OFFSET, 0.),
            Arc::clone(&galley),
            Color32::WHITE,
        );
        // Paints the galley, and handles drag-to-select and ctrl+C on the selected span. The
        // per-residue coloring doesn't get in the way: it's all one galley.
        LabelSelectionState::label_text_selection(
            ui,
            &response,
            rect.min,
            Arc::clone(&galley),
            Color32::WHITE,
            Stroke::NONE,
        );

        let btn_rect = Rect::from_min_size(rect.min + btn_offset, btn_size);
        if ui
            .put(btn_rect, Button::new(SEQ_COPY_TEXT))
            .on_hover_text("Copy this sequence to the clipboard")
            .clicked()
        {
            ui.ctx().copy_text(seq_text.to_owned());
        }

        // The sequence position under the pointer, clamped to the last residue: a pointer past
        // the end of a row resolves to one index beyond it.
        let seq_pos = |pointer: Pos2| {
            galley
                .cursor_from_pos(pointer - rect.min)
                .index
                .0
                .min(res_indices.len().saturating_sub(1))
        };

        // Dragging across the sequence selects the residues it spans, so the 3D view highlights
        // the same stretch of protein that egui highlights in the text.
        if response.drag_started()
            && let Some(pointer) = response.interact_pointer_pos()
            && !btn_rect.contains(pointer)
        {
            cache.drag_anchor = Some(seq_pos(pointer));
        }

        if let Some(anchor) = cache.drag_anchor
            && response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let current = seq_pos(pointer);
            let (start, end) = (anchor.min(current), anchor.max(current));

            let residues = res_indices
                .get(start..=end)
                .map(<[usize]>::to_vec)
                .unwrap_or_default();

            let new = match residues.len() {
                0 => Selection::None,
                1 => Selection::Residue(residues[0]),
                _ => Selection::Residues(residues),
            };

            if *selection != new {
                *selection = new;
                *redraw = true;
                ui.request_repaint();
            }
        }

        if response.drag_stopped() {
            cache.drag_anchor = None;
        }

        // The allocated area covers the button too, so skip clicks landing on it; otherwise
        // copying would also select the last residue.
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
            && !btn_rect.contains(pointer)
            && let Some(residue) = res_indices.get(seq_pos(pointer)).copied()
        {
            *selection = Selection::Residue(residue);
            *redraw = true;
            ui.request_repaint();
        }
    });
}
