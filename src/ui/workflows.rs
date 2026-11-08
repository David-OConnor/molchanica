//! Selection annotations and reusable commands.

use egui::{Color32, FontId, Pos2, Stroke, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    annotations::{LabelFormat, center_slab, label_selection},
    cli::scripts,
    selection::expression,
    state::State,
    util::RedrawFlags,
};

#[derive(Default)]
pub struct WorkflowUi {
    pub commands_open: bool,
    label_format: LabelFormat,
    message: String,
    body: String,
    macro_name: String,
    file: egui_file_dialog::FileDialog,
}

pub fn label_controls(state: &mut State, ui: &mut Ui) {
    ui.collapsing("Labels on selection", |ui| {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("selection_label_format")
                .selected_text(state.ui.workflows.label_format.title())
                .show_ui(ui, |ui| {
                    for format in [
                        LabelFormat::Residue,
                        LabelFormat::ChainResidue,
                        LabelFormat::AtomName,
                    ] {
                        ui.selectable_value(
                            &mut state.ui.workflows.label_format,
                            format,
                            format.title(),
                        );
                    }
                });
            if ui.button("Label this selection").clicked() {
                state.ui.workflows.message =
                    label_selection(state, "sele", Some(state.ui.workflows.label_format))
                        .map(|count| format!("Added {count} labels"))
                        .unwrap_or_else(|e| e);
            }
            if ui.button("Clear selection labels").clicked() {
                state.ui.workflows.message = label_selection(state, "sele", None)
                    .err()
                    .unwrap_or_default();
            }
        });
        ui.small("Drag labels in the viewport. Lines follow their atoms.");
        let mut remove = None;
        for (i, label) in state.to_save.inspection.labels.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let mut changed = ui.checkbox(&mut label.visible, &label.text).changed();
                if ui.small_button("Reset position").clicked() {
                    label.offset = [16., -16.];
                    changed = true;
                }
                if ui.small_button("Remove").clicked() {
                    remove = Some(i);
                }
                state.to_save.save_flag |= changed;
            });
        }
        if let Some(i) = remove {
            state.to_save.inspection.labels.remove(i);
            state.to_save.save_flag = true;
        }
        if !state.ui.workflows.message.is_empty() {
            ui.label(&state.ui.workflows.message);
        }
    });
}

pub fn slab_controls(state: &mut State, ui: &mut Ui) -> bool {
    let before = state.to_save.inspection.slab.clone();
    if ui.button("10 Å slice around selection").clicked() {
        state.ui.workflows.message = center_slab(state, "sele", 10.).err().unwrap_or_default();
    }

    if let Some(slab) = &mut state.to_save.inspection.slab {
        ui.label("Thickness:");
        ui.add(
            egui::DragValue::new(&mut slab.thickness)
                .range(0.1..=10000.)
                .speed(0.1)
                .suffix(" Å"),
        );

        ui.spacing_mut().slider_width = 60.;
        ui.add(
            egui::Slider::new(&mut slab.offset, -100.0..=100.0)
                .clamping(egui::SliderClamping::Never)
                .text("Slide (Å)"),
        )
        .on_hover_text("Positive offset moves away from you.");

        ui.label("Center offset:");
        ui.add(
            egui::DragValue::new(&mut slab.offset)
                .speed(0.1)
                .suffix(" Å"),
        );

        let thickness = slab.thickness;
        if ui.button("Recenter on selection").clicked() {
            state.ui.workflows.message = center_slab(state, "sele", thickness)
                .err()
                .unwrap_or_default();
        }
    }

    let changed = before != state.to_save.inspection.slab;
    state.to_save.save_flag |= changed;
    if !state.ui.workflows.message.is_empty() {
        ui.label(&state.ui.workflows.message);
    }

    changed
}

pub fn command_window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    ctx: &egui::Context,
) {
    if !state.ui.workflows.commands_open {
        return;
    }

    let mut edit = std::mem::take(&mut state.ui.workflows);
    let mut open = true;
    let mut run = None;
    edit.file.update(ctx);
    if let Some(path) = edit.file.take_picked() {
        run = Some(format!(
            "run {}",
            expression::quote(&path.to_string_lossy())
        ));
    }
    egui::Window::new("Command files / macros")
        .open(&mut open)
        .default_width(600.)
        .vscroll(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Run command file…").clicked() {
                    edit.file.pick_file();
                }
                if ui.button("Run editor").clicked() {
                    run = Some(edit.body.clone());
                }
            });
            ui.small("Commands use lines or semicolons. # starts comments. File paths are relative to the command file.");
            ui.add(
                egui::TextEdit::multiline(&mut edit.body)
                    .code_editor()
                    .desired_rows(10)
                    .desired_width(f32::INFINITY)
                    .hint_text("show sticks, sele\ncolor yellow, sele\nlabel sele, chain_residue\nzoom sele\nclip slab, 10, sele"),
            );
            ui.horizontal(|ui| {
                ui.label("Macro name:");
                ui.text_edit_singleline(&mut edit.macro_name);
                if ui.button("Save macro").clicked() {
                    edit.message = scripts::store_macro(state, &edit.macro_name, &edit.body)
                        .map(|_| "Macro saved".into())
                        .unwrap_or_else(|e| e.to_string());
                }
            });
            ui.small("Use $selection in a macro; invoke its name with an optional selection expression.");

            let mut remove = None;
            for (i, mac) in state.to_save.command_macros.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(&mac.name);
                    if ui.small_button("Run on selection").clicked() {
                        run = Some(mac.name.clone());
                    }
                    if ui.small_button("Edit").clicked() {
                        edit.macro_name = mac.name.clone();
                        edit.body = mac.body.clone();
                    }
                    if ui.small_button("Delete").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                state.to_save.command_macros.remove(i);
                state.to_save.save_flag = true;
            }
            if !edit.message.is_empty() {
                ui.label(&edit.message);
            }
        });
    if let Some(text) = run {
        let mut changed = false;
        edit.message = scripts::execute(state, scene, updates, &mut changed, &text)
            .unwrap_or_else(|e| e.to_string());
        if changed {
            redraw.set_all();
        }
    }
    edit.commands_open = open;
    state.ui.workflows = edit;
}

pub fn paint_label(
    painter: &egui::Painter,
    anchor: Pos2,
    text: &str,
    offset: [f32; 2],
    scale: f32,
    color: Color32,
) -> egui::Rect {
    let galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(16. * scale), color);
    let position = anchor + egui::vec2(offset[0], offset[1]) * scale;
    let rect = egui::Rect::from_min_size(position, galley.size()).expand(3. * scale);
    let end = rect.clamp(anchor);
    if anchor.distance(end) > 3. * scale {
        painter.line_segment([anchor, end], Stroke::new(1. * scale, color));
    }
    let background = if color.r() as u16 + color.g() as u16 + color.b() as u16 > 384 {
        Color32::from_black_alpha(160)
    } else {
        Color32::from_white_alpha(200)
    };
    painter.rect_filled(rect, 3., background);
    painter.galley(position, galley, color);

    rect
}

pub fn paint_labels(state: &mut State, scene: &Scene, ctx: &egui::Context, reserved: (f32, f32)) {
    let (camera, viewport) =
        super::measurements::measurement_viewport(scene, reserved, ctx.pixels_per_point());
    let positions: Vec<_> = state
        .to_save
        .inspection
        .labels
        .iter()
        .map(|l| {
            let p = l.position(state)?;
            let (visible, ndc) = camera.in_view(p.into());

            (l.visible && visible).then_some(Pos2::new(
                viewport.left() + (ndc.0 + 1.) * viewport.width() / 2.,
                viewport.top() + (1. - ndc.1) * viewport.height() / 2.,
            ))
        })
        .collect();
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("selection_label_lines"),
        ))
        .with_clip_rect(viewport);
    for (i, anchor) in positions.into_iter().enumerate() {
        let Some(anchor) = anchor else { continue };
        let label = &mut state.to_save.inspection.labels[i];
        let position = anchor + egui::vec2(label.offset[0], label.offset[1]);
        let position = viewport.shrink(8.).clamp(position);
        egui::Area::new(egui::Id::new(("selection_label", i)))
            .order(egui::Order::Middle)
            .movable(false)
            .fixed_pos(position)
            .show(ctx, |ui| {
                ui.set_clip_rect(viewport);
                let response = ui.add(
                    egui::Label::new(
                        egui::RichText::new(&label.text)
                            .color(Color32::WHITE)
                            .background_color(Color32::from_black_alpha(160))
                            .size(16.),
                    )
                    .sense(egui::Sense::click_and_drag()),
                );
                if response.dragged() {
                    let delta = response.drag_delta();
                    label.offset[0] += delta.x;
                    label.offset[1] += delta.y;
                    state.to_save.save_flag = true;
                }
                if response.double_clicked() {
                    label.offset = [16., -16.];
                    state.to_save.save_flag = true;
                }
                painter.line_segment(
                    [anchor, response.rect.clamp(anchor)],
                    Stroke::new(1., Color32::LIGHT_GRAY),
                );
                response.on_hover_text("Drag to move; double-click to reset");
            });
    }
}
