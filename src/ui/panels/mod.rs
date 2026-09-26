use std::{collections::HashSet, sync::Arc};

use egui::{
    Button, Color32, Direction, FontId, Frame, Galley, Layout, Pos2, Rect, Sense, Stroke,
    TextFormat, TextStyle, Ui, UiBuilder, Vec2, pos2, text::LayoutJob,
    text_selection::LabelSelectionState, vec2,
};

use crate::{drawing::color_viridis, selection::Selection, state::AaSeqDisplayCache};

pub mod md;
pub mod md_viewer;
pub mod mol_data;
pub mod orca;
pub mod view;

/// Label on the button that copies the whole amino acid sequence to the clipboard.
const SEQ_COPY_ALL_TEXT: &str = "Copy all";
/// Label on the button that copies only the selected residues.
const SEQ_COPY_SEL_TEXT: &str = "Copy sel";
/// Horizontal gap between the final residue and the copy buttons sharing its line.
const SEQ_COPY_PAD: f32 = 40.;
/// Horizontal gap between the two copy buttons.
const SEQ_BTN_GAP: f32 = 6.;
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
/// Residues whose serial number is a multiple of this get a number label above them.
const SEQ_NUM_INTERVAL: u32 = 10;
/// Size of the residue number labels, relative to the sequence's font.
const SEQ_NUM_SCALE: f32 = 0.72;
/// Muted, so the numbers read as a ruler rather than competing with the residue colors.
const SEQ_NUM_COLOR: Color32 = Color32::from_gray(145);
/// Space above each band of residue numbers, separating it from the sequence row above.
const SEQ_NUM_BAND_PAD: f32 = 2.;
/// Minimum horizontal space between two number labels on a row. A label that would crowd the
/// one before it (e.g. after a gap in the numbering) is skipped.
const SEQ_NUM_GAP: f32 = 6.;

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

/// Open a gap of `band` points above each row of the sequence galley, to hold that row's residue
/// numbers. Only whole rows move, so the text, its per-residue backgrounds, and egui's
/// text-selection highlight are drawn exactly as before, and the numbers never become part of
/// the text.
fn space_seq_rows(galley: &mut Arc<Galley>, band: f32) {
    // The galley is shared with egui's layout cache; this modifies a copy.
    let galley = Arc::make_mut(galley);

    let mut y = 0.;
    let mut rect = Rect::ZERO;
    let mut mesh_bounds = Rect::NOTHING;

    for placed_row in &mut galley.rows {
        y += band;
        placed_row.pos.y = y;
        y += placed_row.row.size.y;

        rect |= placed_row.rect();
        mesh_bounds |= placed_row
            .row
            .visuals
            .mesh_bounds
            .translate(placed_row.pos.to_vec2());
    }

    // `rect` starts at the origin, so it includes the first row's band.
    galley.rect = rect;
    galley.mesh_bounds = mesh_bounds;
}

/// Residue number labels for the sequence, positioned relative to `galley` after
/// `space_seq_rows`. Labeled are the first residue, each one after a break in the numbering
/// (e.g. a new chain, or unresolved residues), and every `SEQ_NUM_INTERVAL`th. Each sits
/// directly above its residue, left-aligned with it.
fn seq_num_labels(
    galley: &Galley,
    res_sns: &[u32],
    font_id: &FontId,
    ui: &Ui,
) -> Vec<(Pos2, Arc<Galley>)> {
    let mut labels = Vec::new();
    let mut row_start = 0;

    // Insertion codes repeat a number; that isn't a break.
    let is_break = |i: usize, sn: u32| match i.checked_sub(1).and_then(|prev| res_sns.get(prev)) {
        Some(&prev_sn) => sn != prev_sn && prev_sn.checked_add(1) != Some(sn),
        None => true,
    };

    ui.fonts_mut(|fonts| {
        for placed_row in &galley.rows {
            // Horizontal extents of the labels placed on this row so far.
            let mut placed: Vec<(f32, f32)> = Vec::new();

            // Breaks first, so a regular label beside one can't crowd it out: unlike the
            // regular ones, a reader can't work them out by counting.
            for breaks_pass in [true, false] {
                for (col, glyph) in placed_row.row.glyphs.iter().enumerate() {
                    let i = row_start + col;
                    let Some(&sn) = res_sns.get(i) else {
                        continue;
                    };

                    let wanted = if breaks_pass {
                        is_break(i, sn)
                    } else {
                        !is_break(i, sn) && sn % SEQ_NUM_INTERVAL == 0
                    };
                    if !wanted {
                        continue;
                    }

                    let label =
                        fonts.layout_no_wrap(sn.to_string(), font_id.clone(), SEQ_NUM_COLOR);
                    let width = label.size().x;

                    // Kept within the sequence's width, so a label near the end of a row can't
                    // overrun the panel.
                    let x = (placed_row.pos.x + glyph.pos.x)
                        .min(galley.rect.right() - width)
                        .max(0.);

                    let crowded = placed.iter().any(|&(left, right)| {
                        x < right + SEQ_NUM_GAP && left < x + width + SEQ_NUM_GAP
                    });
                    if crowded {
                        continue;
                    }
                    placed.push((x, x + width));

                    let y = placed_row.min_y() - label.size().y;
                    labels.push((pos2(x, y), label));
                }
            }

            row_start += placed_row.row.glyphs.len();
        }
    });

    labels
}

/// The display for the amino acid sequence of an opened protein.
///
/// The colored sequence is one cached galley instead of one widget and color calculation per
/// residue on every frame. It is rebuilt only when its inputs actually change.
///
/// `res_indices` maps each position in `seq_text` to its residue index in the peptide, and
/// `res_sns` to its residue serial number. Rows are interleaved with residue numbers, which are
/// painted separately from the sequence; they aren't selectable, and aren't copied with it.
pub(in crate::ui) fn pepide_aa_seq(
    selection: &mut Selection,
    seq_text: &str,
    res_indices: &[usize],
    res_sns: &[u32],
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

    // The copy buttons share the sequence's last line, so reserve their width before laying
    // the text out; every row then breaks early enough that they can't overrun the panel.
    // `Ui::put` draws each at exactly the size given, so size them the way the default style
    // does: the text plus `button_padding` on each side.
    let btn_padding = ui.spacing().button_padding;
    let btn_size = |label: &str, ui: &Ui| -> Vec2 {
        let text = ui
            .fonts_mut(|fonts| {
                fonts.layout_no_wrap(label.to_owned(), font_id.clone(), Color32::WHITE)
            })
            .size();

        vec2(
            text.x + 2. * btn_padding.x,
            (text.y + 2. * btn_padding.y).max(ui.spacing().interact_size.y),
        )
    };

    let all_size = btn_size(SEQ_COPY_ALL_TEXT, ui);
    let sel_size = btn_size(SEQ_COPY_SEL_TEXT, ui);
    let btns_size = vec2(
        all_size.x + SEQ_BTN_GAP + sel_size.x,
        all_size.y.max(sel_size.y),
    );

    let wrap_width = (ui.available_width() - btns_size.x - SEQ_COPY_PAD).max(1.0);
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

        let mut galley = ui.fonts_mut(|fonts| fonts.layout_job(job));

        let num_font = FontId::new(font_id.size * SEQ_NUM_SCALE, font_id.family.clone());
        let num_height = ui.fonts_mut(|fonts| fonts.row_height(&num_font));
        // Whole pixels, so the rows below stay pixel-aligned and crisp.
        let band = ((num_height + SEQ_NUM_BAND_PAD) * pixels_per_point).round() / pixels_per_point;

        space_seq_rows(&mut galley, band);
        cache.num_labels = seq_num_labels(&galley, res_sns, &num_font, ui);

        cache.galley = Some(galley);
        cache.dirty = false;
        cache.selected.clone_from(&selected);
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
        // Sit the buttons just past the final residue, on the line it ends on. The allocation
        // spans both, so the widgets below don't overlap them.
        let last_row = galley
            .rows
            .last()
            .map(|row| row.rect())
            .unwrap_or(Rect::ZERO);
        let btn_offset = vec2(
            last_row.right() + SEQ_COPY_PAD,
            // Centered on the line it shares, but never above the galley: a single-row sequence
            // is shorter than the button.
            (last_row.center().y - btns_size.y / 2.).max(0.),
        );
        let size = vec2(
            galley.size().x.max(btn_offset.x + btns_size.x),
            galley.size().y.max(btn_offset.y + btns_size.y),
        );

        // `click_and_drag`, so a drag across the sequence selects text the way it does in a
        // normal label; a plain click still picks a single residue.
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        // Only behind the text: the copy buttons keep the panel's own background.
        ui.painter().rect_filled(
            Rect::from_min_size(rect.min, galley.size()).expand(SEQ_BG_PAD),
            3.,
            SEQ_BG,
        );
        for (pos, label) in &cache.num_labels {
            ui.painter()
                .galley(rect.min + pos.to_vec2(), Arc::clone(label), SEQ_NUM_COLOR);
        }
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

        let all_rect = Rect::from_min_size(rect.min + btn_offset, all_size);
        let sel_rect = Rect::from_min_size(
            rect.min + btn_offset + vec2(all_size.x + SEQ_BTN_GAP, 0.),
            sel_size,
        );

        if ui
            .put(all_rect, Button::new(SEQ_COPY_ALL_TEXT))
            .on_hover_text("Copy this sequence to the clipboard")
            .clicked()
        {
            ui.ctx().copy_text(seq_text.to_owned());
        }

        // `Ui::put`, but greyed out with nothing selected. (`put` has no enabled variant.)
        let copy_sel = ui
            .scope_builder(
                UiBuilder::new()
                    .max_rect(sel_rect)
                    .layout(Layout::centered_and_justified(Direction::TopDown)),
                |ui| ui.add_enabled(!selected.is_empty(), Button::new(SEQ_COPY_SEL_TEXT)),
            )
            .inner
            .on_hover_text("Copy the selected residues to the clipboard")
            .on_disabled_hover_text("Select residues first, by dragging across the sequence");

        if copy_sel.clicked() {
            // In sequence order regardless of how the residues were selected, and with any gap
            // between selected stretches simply closed up.
            let selected_set: HashSet<usize> = selected.iter().copied().collect();
            let text: String = seq_text
                .chars()
                .enumerate()
                .filter(|(index, _)| {
                    res_indices
                        .get(*index)
                        .is_some_and(|res_i| selected_set.contains(res_i))
                })
                .map(|(_, amino_acid)| amino_acid)
                .collect();

            ui.ctx().copy_text(text);
        }

        // The sequence position under the pointer, clamped to the last residue: a pointer past
        // the end of a row resolves to one index beyond it. A pointer over a row's residue
        // numbers counts as over that row, below them, instead of whichever row is closest.
        let seq_pos = |pointer: Pos2| {
            let mut pos = pointer - rect.min;
            if let Some(row) = galley.rows.iter().find(|row| pos.y < row.max_y()) {
                pos.y = pos.y.max(row.min_y());
            }

            galley
                .cursor_from_pos(pos)
                .index
                .0
                .min(res_indices.len().saturating_sub(1))
        };

        // Dragging across the sequence selects the residues it spans, so the 3D view highlights
        // the same stretch of protein that egui highlights in the text.
        if response.drag_started()
            && let Some(pointer) = response.interact_pointer_pos()
            && !all_rect.contains(pointer)
            && !sel_rect.contains(pointer)
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

        // The allocated area covers the buttons too, so skip clicks landing on them;
        // otherwise copying would also select the last residue.
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
            && !all_rect.contains(pointer)
            && !sel_rect.contains(pointer)
            && let Some(residue) = res_indices.get(seq_pos(pointer)).copied()
        {
            *selection = Selection::Residue(residue);
            *redraw = true;
            ui.request_repaint();
        }
    });
}
