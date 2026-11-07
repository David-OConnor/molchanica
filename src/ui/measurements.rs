//! Measurement controls and viewport overlays.

use egui::{Align2, Color32, ComboBox, FontId, Pos2, Stroke, Ui};
use graphics::Scene;
use lin_alg::f64::Vec3;
use mol_defs::molecules::MolGenericRef;

use crate::{
    inspection::{
        AngleUnit, AtomRef, MeasureKind, Measurement, Palette, ScalarColor, add_colors, b_factors,
        import_values,
    },
    selection::ViewSelLevel,
    state::State,
    ui::ROW_SPACING,
    util::RedrawFlags,
};

pub(crate) const MEASUREMENT_LINE_WIDTH: f32 = 3.;
const MEASUREMENT_FONT_SIZE: f32 = 18.;

#[derive(Default)]
pub struct InspectionUi {
    pub solid_colors: std::collections::BTreeMap<String, std::collections::BTreeMap<u32, [f32; 3]>>,
    pub colors: Vec<ScalarColor>,
    pub open: bool,
    pub kind: MeasureKind,
    pub pending: Vec<AtomRef>,
    pub picking: bool,
    pub message: String,
    pub csv: String,
    pub csv_name: String,
    pub trajectory_pick: Option<AtomRef>,
    pub import: egui_file_dialog::FileDialog,
}

pub fn paint_measurements(
    measurements: &[Measurement],
    state: &State,
    camera: &graphics::Camera,
    painter: &egui::Painter,
    rect: egui::Rect,
    scale: f32,
) {
    let project = |p: Vec3| {
        let (visible, ndc) = camera.in_view(p.into());
        visible.then_some(Pos2::new(
            rect.left() + (ndc.0 + 1.) * 0.5 * rect.width(),
            rect.top() + (1. - ndc.1) * 0.5 * rect.height(),
        ))
    };

    for measurement in measurements.iter().filter(|m| m.visible) {
        let Some(points) = measurement.points(state) else {
            continue;
        };

        let projected: Option<Vec<_>> = points.iter().map(|&p| project(p)).collect();
        let Some(projected) = projected else {
            continue;
        };

        let color = Color32::from_rgb(
            measurement.color[0],
            measurement.color[1],
            measurement.color[2],
        );

        for pair in projected.windows(2) {
            let steps = ((pair[1] - pair[0]).length() / 6.).ceil().clamp(1., 2000.) as usize;
            for i in (0..steps).step_by(2) {
                painter.line_segment(
                    [
                        pair[0].lerp(pair[1], i as f32 / steps as f32),
                        pair[0].lerp(pair[1], (i + 1).min(steps) as f32 / steps as f32),
                    ],
                    Stroke::new(MEASUREMENT_LINE_WIDTH * scale, color),
                );
            }
        }

        let anchor = match measurement.kind {
            MeasureKind::Angle => projected[1],
            MeasureKind::Dihedral => projected[1].lerp(projected[2], 0.5),
            MeasureKind::Distance => projected[0].lerp(projected[1], 0.5),
        };

        paint_measurement_text(
            painter,
            anchor - egui::vec2(0., 8.),
            format!("{}: {}", measurement.name, measurement.label(state)),
            color,
            scale,
        );
    }
}

pub(crate) fn paint_measurement_text(
    painter: &egui::Painter,
    anchor: Pos2,
    text: String,
    color: Color32,
    scale: f32,
) {
    let galley = painter.layout_no_wrap(
        text,
        FontId::proportional(MEASUREMENT_FONT_SIZE * scale),
        color,
    );
    let position = Align2::CENTER_BOTTOM.anchor_size(anchor, galley.size()).min;

    // Thicken the glyphs without requiring a platform-specific bold font.
    for x in [-0.3, 0.3] {
        painter.galley(position + egui::vec2(x * scale, 0.), galley.clone(), color);
    }
}

pub fn show(state: &mut State, ctx: &egui::Context, redraw: &mut RedrawFlags) {
    if !state.ui.inspection.open {
        return;
    }

    let before = state.to_save.inspection.clone();
    let previous_charge = state.ui.atom_color_by_charge;
    let mut open = true;

    egui::Window::new("Measurements / property colors")
        .open(&mut open)
        .default_width(520.)
        .default_height(600.)
        .vscroll(true)
        .show(ctx, |ui| controls(state, ui, ctx));

    state.ui.inspection.open = open;
    if before != state.to_save.inspection || previous_charge != state.ui.atom_color_by_charge {
        if let Err(error) = state.to_save.inspection.validate() {
            state.ui.inspection.message = error;
            state.to_save.inspection = before;
        }
        state.ui.inspection.colors = state.to_save.inspection.colors.clone();
        state.to_save.save_flag = true;
        state.volatile.flags.update_ss_mesh = true;
        state.volatile.flags.update_sas_coloring = true;
        redraw.set_all();
    }
}

fn controls(state: &mut State, ui: &mut Ui, ctx: &egui::Context) {
    ui.heading("Measurements");
    ui.horizontal(|ui| {
        ui.label("Angle units:");
        for unit in [AngleUnit::Degrees, AngleUnit::Radians, AngleUnit::Turns] {
            ui.selectable_value(&mut state.to_save.inspection.angle_unit, unit, unit.label())
                .on_hover_text(if unit == AngleUnit::Turns {
                    "One turn is a full circle. Negative dihedrals wrap into 0–1."
                } else {
                    "Display units for angles and dihedrals"
                });
        }
    });

    ui.horizontal(|ui| {
        for kind in [
            MeasureKind::Distance,
            MeasureKind::Angle,
            MeasureKind::Dihedral,
        ] {
            if ui
                .selectable_value(&mut state.ui.inspection.kind, kind, kind.name())
                .changed()
            {
                state.ui.inspection.pending.clear();
            }
        }
        if ui.button("Pick atoms").clicked() {
            state.ui.inspection.pending.clear();
            state.ui.inspection.picking = true;
            state.ui.view_sel_level = ViewSelLevel::AtomBond;
        }
        if ui.button("Cancel").clicked() {
            state.ui.inspection.pending.clear();
            state.ui.inspection.picking = false;
        }
    });

    if state.ui.inspection.picking {
        ui.label(format!(
            "Right-click atoms in order: {} / {}",
            state.ui.inspection.pending.len(),
            state.ui.inspection.kind.count()
        ));
    }

    measurement_table(state, ui);
    ui.separator();
    ui.heading("Property colors");

    if ui
        .checkbox(
            &mut state.ui.atom_color_by_charge,
            "Color by partial charge (q)",
        )
        .on_hover_text("Color atoms in all molecules by partial charge instead of element")
        .changed()
        && state.ui.atom_color_by_charge
    {
        state.set_view_sel_level(ViewSelLevel::AtomBond);
        state.ui.aa_selection.enabled = false;
        for colors in &mut state.to_save.inspection.colors {
            colors.enabled = false;
        }
    }
    ui.add_space(ROW_SPACING);

    if let Some(mol) = state.active_mol() {
        ui.label(mol.name().into_owned());
    } else {
        ui.label("Select a molecule to load colors.");
    }

    ui.horizontal(|ui| {
        for (title, confidence) in [("B-factor", false), ("Confidence (pLDDT)", true)] {
            if ui.button(title).clicked() {
                let values = state
                    .active_mol()
                    .and_then(|m| match m {
                        MolGenericRef::Peptide(p) => p.source_cif.as_deref().map(b_factors),
                        _ => None,
                    })
                    .unwrap_or_default();
                state.ui.inspection.message = add_colors(state, title.into(), values, confidence)
                    .err()
                    .unwrap_or_default();
            }
        }
    });

    ui.small(
        "Confidence uses the source B column as pLDDT; use it only for prediction confidence.",
    );

    ui.collapsing("Import values (CSV / TSV)", |ui| import_controls(state, ui));
    state.ui.inspection.import.update(ctx);
    if let Some(path) = state.ui.inspection.import.take_picked() {
        match std::fs::read_to_string(path) {
            Ok(text) => state.ui.inspection.csv = text,
            Err(e) => state.ui.inspection.message = e.to_string(),
        }
    }

    for (i, colors) in state.to_save.inspection.colors.iter_mut().enumerate() {
        ui.push_id(("colors", i), |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut colors.enabled, &colors.title);
                ui.label(format!("{} values", colors.values.len()));
            });
            ui.horizontal(|ui| {
                let previous_palette = colors.palette;
                ComboBox::from_id_salt("palette")
                    .selected_text(format!("{:?}", colors.palette))
                    .show_ui(ui, |ui| {
                        for p in [Palette::Viridis, Palette::BlueWhiteRed, Palette::Confidence] {
                            ui.selectable_value(&mut colors.palette, p, format!("{p:?}"));
                        }
                    });
                if previous_palette != colors.palette && colors.palette == Palette::Confidence {
                    colors.minimum = 0.;
                    colors.maximum = 100.;
                }

                let continuous = colors.palette != Palette::Confidence;
                ui.add_enabled(
                    continuous,
                    egui::DragValue::new(&mut colors.minimum).prefix("Min "),
                );
                ui.add_enabled(
                    continuous,
                    egui::DragValue::new(&mut colors.maximum).prefix("Max "),
                );
                if ui
                    .add_enabled(continuous, egui::Button::new("Auto range"))
                    .clicked()
                {
                    colors.minimum = colors
                        .values
                        .values()
                        .copied()
                        .fold(f32::INFINITY, f32::min);
                    colors.maximum = colors
                        .values
                        .values()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                }
                ui.color_edit_button_rgb(&mut colors.missing);
            });

            let (rect, _) = ui.allocate_exact_size(egui::vec2(240., 12.), egui::Sense::hover());
            for x in 0..120 {
                let (r, g, b) = colors.color_value(
                    colors.minimum + (colors.maximum - colors.minimum) * x as f32 / 119.,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(
                        rect.min + egui::vec2(x as f32 * 2., 0.),
                        egui::vec2(2., 12.),
                    ),
                    0.,
                    Color32::from_rgb((r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8),
                );
            }
            ui.label(format!(
                "{}   {:.2} → {:.2}",
                colors.object, colors.minimum, colors.maximum
            ));
        });
    }

    if !state.ui.inspection.message.is_empty() {
        ui.colored_label(Color32::YELLOW, &state.ui.inspection.message);
    }
}

fn measurement_table(state: &mut State, ui: &mut Ui) {
    let labels: Vec<_> = state
        .to_save
        .inspection
        .measurements
        .iter()
        .map(|m| m.label(state))
        .collect();
    let mut delete = None;

    egui::Grid::new("measurements")
        .num_columns(6)
        .striped(true)
        .min_col_width(0.)
        .spacing(egui::vec2(10., 6.))
        .show(ui, |ui| {
            for header in ["Show", "Type", "Name", "Color", "Value", ""] {
                ui.strong(header);
            }
            ui.end_row();

            for (i, measurement) in state.to_save.inspection.measurements.iter_mut().enumerate() {
                ui.checkbox(&mut measurement.visible, "");
                ui.label(measurement.kind.name());
                ui.add(
                    egui::TextEdit::singleline(&mut measurement.name)
                        .id_salt(("measurement_name", i))
                        .desired_width(130.),
                );
                ui.color_edit_button_srgb(&mut measurement.color);
                ui.add_sized(
                    [96., ui.spacing().interact_size.y],
                    egui::Label::new(&labels[i]).wrap_mode(egui::TextWrapMode::Truncate),
                )
                .on_hover_text(&labels[i]);
                if ui.button("Delete").clicked() {
                    delete = Some(i);
                }
                ui.end_row();
            }
        });

    if let Some(i) = delete {
        state.to_save.inspection.measurements.remove(i);
    }
}

fn import_controls(state: &mut State, ui: &mut Ui) {
    ui.small("atom_serial,value or chain,residue,value. Use loaded structure numbering.");
    ui.add(egui::TextEdit::singleline(&mut state.ui.inspection.csv_name).hint_text("Dataset name"));
    ui.add(
        egui::TextEdit::multiline(&mut state.ui.inspection.csv)
            .hint_text("atom_serial,value\n1,25.0\n2,31.5")
            .desired_rows(5),
    );

    if ui.button("Load values file").clicked() {
        state.ui.inspection.import.pick_file();
    }

    if ui.button("Apply values").clicked() {
        let result = state
            .active_mol()
            .ok_or_else(|| "Select a molecule first".to_string())
            .and_then(|mol| import_values(&mol, &state.ui.inspection.csv));
        let title = if state.ui.inspection.csv_name.trim().is_empty() {
            "Imported values".into()
        } else {
            state.ui.inspection.csv_name.clone()
        };

        state.ui.inspection.message = result
            .and_then(|values| add_colors(state, title, values, false))
            .err()
            .unwrap_or_default();
    }
}

pub(crate) fn measurement_viewport(
    scene: &Scene,
    reserved: (f32, f32),
    pixels_per_point: f32,
) -> (graphics::Camera, egui::Rect) {
    let (x, y, width, height) = graphics::viewport_rect(
        reserved,
        scene.window_size.0 as u32,
        scene.window_size.1 as u32,
        &graphics::UiSettings::default(),
        0.,
    );

    let rect = egui::Rect::from_min_size(
        Pos2::new(x / pixels_per_point, y / pixels_per_point),
        egui::vec2(width / pixels_per_point, height / pixels_per_point),
    );

    let mut camera = scene.camera.clone();
    // The renderer updates its projection after the UI callback.
    camera.aspect = width / height;
    camera.update_proj_mat();

    (camera, rect)
}

pub fn paint(state: &State, scene: &Scene, ctx: &egui::Context, reserved: (f32, f32)) {
    let (camera, viewport) = measurement_viewport(scene, reserved, ctx.pixels_per_point());
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("measurements"),
        ))
        .with_clip_rect(viewport);

    paint_measurements(
        &state.to_save.inspection.measurements,
        state,
        &camera,
        &painter,
        viewport,
        1.,
    );

    if state.ui.inspection.picking {
        painter.text(
            viewport.center_bottom() - egui::vec2(0., 30.),
            Align2::CENTER_BOTTOM,
            format!(
                "{}: right-click atoms ({}/{}) · Esc cancels",
                state.ui.inspection.kind.name(),
                state.ui.inspection.pending.len(),
                state.ui.inspection.kind.count()
            ),
            FontId::proportional(17.),
            Color32::from_rgb(255, 220, 80),
        );
    }
}
