//! Box and lasso residue selection.

use std::collections::BTreeSet;

use egui::{Color32, Pos2, Stroke, Ui};
use graphics::{ControlScheme, Scene};
use lin_alg::f32::{Vec3, Vec4};
use mol_defs::molecules::{AtomRole, MolType};
use na_seq::Element;

use crate::{
    drawing::{MoleculeView, peptide::filter_pep_atoms_by_dist},
    selection::{Selection, ViewSelLevel},
    state::{OperatingMode, State},
    util::RedrawFlags,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum RegionTool {
    #[default]
    Off,
    Box,
    Lasso,
}

pub(crate) struct RegionSelection {
    pub(crate) tool: RegionTool,
    pub(crate) through: bool,
    pub(crate) points: Vec<(f32, f32)>,
    previous_controls: Option<ControlScheme>,
    add: bool,
    subtract: bool,
    radius: f64,
    summary: String,
}

impl Default for RegionSelection {
    fn default() -> Self {
        Self {
            tool: RegionTool::Off,
            through: false,
            points: Vec::new(),
            previous_controls: None,
            add: false,
            subtract: false,
            radius: 5.,
            summary: String::new(),
        }
    }
}

pub(crate) fn arm(state: &mut State, scene: &mut Scene, tool: RegionTool) {
    if state.volatile.operating_mode != OperatingMode::Primary
        || state.volatile.mol_manip.mode != crate::mol_manip::ManipMode::None
    {
        return;
    }

    let region = &mut state.volatile.region_selection;
    if region.tool == tool || tool == RegionTool::Off {
        cancel(state, scene);

        return;
    }
    if region.previous_controls.is_none() {
        region.previous_controls = Some(scene.input_settings.control_scheme);
    }
    region.tool = tool;
    region.points.clear();
    scene.input_settings.control_scheme = ControlScheme::None;
}

pub(crate) fn cancel(state: &mut State, scene: &mut Scene) {
    let region = &mut state.volatile.region_selection;
    region.tool = RegionTool::Off;
    region.points.clear();
    if let Some(controls) = region.previous_controls.take() {
        scene.input_settings.control_scheme = controls;
    }
    state.ui.left_click_down = false;
}

pub(crate) fn begin(state: &mut State) -> bool {
    if state.volatile.region_selection.tool == RegionTool::Off {
        return false;
    }
    let Some(cursor) = state.ui.cursor_pos else {
        return false;
    };
    let modifiers = state.volatile.key_modifiers.state();
    let region = &mut state.volatile.region_selection;
    region.points = vec![cursor, cursor];
    region.add = modifiers.shift_key();
    region.subtract = modifiers.control_key();

    true
}

pub(crate) fn motion(state: &mut State) -> bool {
    let Some(cursor) = state.ui.cursor_pos else {
        return false;
    };
    let region = &mut state.volatile.region_selection;
    if region.points.is_empty() {
        return false;
    }

    if region.tool == RegionTool::Box {
        region.points[1] = cursor;
    } else {
        let previous = *region.points.last().unwrap();
        if (cursor.0 - previous.0).hypot(cursor.1 - previous.1) >= 3. && region.points.len() < 8192
        {
            region.points.push(cursor);
        }
    }

    true
}

fn polygon(tool: RegionTool, points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if tool == RegionTool::Box && points.len() >= 2 {
        let (a, b) = (points[0], points[1]);

        return vec![a, (b.0, a.1), b, (a.0, b.1)];
    }

    points.to_vec()
}

fn inside(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut result = false;
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
        if (a.1 > point.1) != (b.1 > point.1)
            && point.0 < (b.0 - a.0) * (point.1 - a.1) / (b.1 - a.1) + a.0
        {
            result = !result;
        }
    }

    result
}

struct DepthBuffer {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

impl DepthBuffer {
    fn new(scene: &Scene, viewport: (f32, f32, f32, f32)) -> Self {
        let width = viewport.2.ceil().max(1.) as usize;
        let height = viewport.3.ceil().max(1.) as usize;
        let mut buffer = Self {
            width,
            height,
            values: vec![f32::INFINITY; width * height],
        };
        let projection = scene.camera.proj_mat.clone() * scene.camera.view_mat();
        for entity in &scene.entities {
            if entity.opacity < 0.98
                || entity.class == crate::drawing::EntityClass::SelectionHighlight as u32
            {
                continue;
            }
            let Some(mesh) = scene.meshes.get(entity.mesh) else {
                continue;
            };
            let scale =
                entity
                    .scale_partial
                    .unwrap_or(Vec3::new(entity.scale, entity.scale, entity.scale));
            let pivot = entity.pivot.unwrap_or(Vec3::new_zero());
            let vertices: Vec<_> = mesh
                .vertices
                .iter()
                .map(|v| {
                    let p = Vec3::new(
                        v.position[0] * scale.x,
                        v.position[1] * scale.y,
                        v.position[2] * scale.z,
                    );
                    let p = entity.position + pivot + entity.orientation.rotate_vec(p - pivot);

                    projection.clone() * Vec4::new(p.x, p.y, p.z, 1.)
                })
                .collect();
            let screen_point = |p: Vec4| {
                Vec3::new(
                    (p.x / p.w * 0.5 + 0.5) * width as f32,
                    (0.5 - p.y / p.w * 0.5) * height as f32,
                    p.z / p.w,
                )
            };

            for triangle in mesh.indices.chunks_exact(3) {
                let points = [
                    vertices[triangle[0]],
                    vertices[triangle[1]],
                    vertices[triangle[2]],
                ];
                if points.iter().all(|p| {
                    p.w > 0. && p.x.abs() <= p.w && p.y.abs() <= p.w && p.z >= 0. && p.z <= p.w
                }) {
                    buffer.triangle(points.map(screen_point));

                    continue;
                }

                let mut clipped = points.to_vec();
                for plane in 0..6 {
                    let distance = |p: Vec4| match plane {
                        0 => p.w + p.x,
                        1 => p.w - p.x,
                        2 => p.w + p.y,
                        3 => p.w - p.y,
                        4 => p.z,
                        _ => p.w - p.z,
                    };
                    let mut output = Vec::new();
                    for i in 0..clipped.len() {
                        let (a, b) = (clipped[i], clipped[(i + 1) % clipped.len()]);
                        let (da, db) = (distance(a), distance(b));
                        if da >= 0. {
                            output.push(a);
                        }
                        if (da >= 0.) != (db >= 0.) {
                            output.push(a + (b - a) * (da / (da - db)));
                        }
                    }
                    clipped = output;
                    if clipped.len() < 3 {
                        break;
                    }
                }
                for i in 1..clipped.len().saturating_sub(1) {
                    let points = [clipped[0], clipped[i], clipped[i + 1]].map(screen_point);
                    buffer.triangle(points);
                }
            }
        }

        buffer
    }

    fn triangle(&mut self, [a, b, c]: [Vec3; 3]) {
        let edge =
            |a: Vec3, b: Vec3, p: Vec3| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        let area = edge(a, b, c);
        if !area.is_finite() || area.abs() < 1.0e-6 {
            return;
        }

        let min_x = a.x.min(b.x).min(c.x).floor().max(0.) as usize;
        let max_x = (a.x.max(b.x).max(c.x).ceil() as usize).min(self.width - 1);
        let min_y = a.y.min(b.y).min(c.y).floor().max(0.) as usize;
        let max_y = (a.y.max(b.y).max(c.y).ceil() as usize).min(self.height - 1);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.);
                let weights = [
                    edge(b, c, p) / area,
                    edge(c, a, p) / area,
                    edge(a, b, p) / area,
                ];
                if weights.iter().all(|w| *w >= 0.) {
                    let depth = weights[0] * a.z + weights[1] * b.z + weights[2] * c.z;
                    let index = y * self.width + x;
                    self.values[index] = self.values[index].min(depth);
                }
            }
        }
    }

    fn visible(&self, scene: &Scene, point: Vec3, radius: f32) -> bool {
        let direction = (point - scene.camera.position).to_normalized();
        let surface = point - direction * radius;
        let projection = scene.camera.proj_mat.clone() * scene.camera.view_mat();
        let clip = projection * Vec4::new(surface.x, surface.y, surface.z, 1.);
        if clip.w <= 0. {
            return false;
        }

        let x = ((clip.x / clip.w * 0.5 + 0.5) * self.width as f32)
            .floor()
            .clamp(0., (self.width - 1) as f32) as usize;
        let y = ((0.5 - clip.y / clip.w * 0.5) * self.height as f32)
            .floor()
            .clamp(0., (self.height - 1) as f32) as usize;

        clip.z / clip.w <= self.values[y * self.width + x] + 3.0e-7
    }
}

pub(crate) fn finish(state: &mut State, scene: &mut Scene, redraw: &mut RedrawFlags) -> bool {
    let region = &state.volatile.region_selection;
    if region.points.is_empty() {
        return false;
    }
    let Some(peptide_i) = state.peptide_for_tools_i() else {
        cancel(state, scene);

        return true;
    };
    let shape = polygon(region.tool, &region.points);
    let min_x = shape.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let max_x = shape.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let min_y = shape.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y = shape.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
    if max_x - min_x < 3. || max_y - min_y < 3. {
        cancel(state, scene);

        return true;
    }

    let (add, subtract, through) = (region.add, region.subtract, region.through);
    let scale = state.volatile.mol_manip.pixels_per_point;
    let vp = crate::gizmo::viewport_points(scene, scale);
    let depth = (!through).then(|| DepthBuffer::new(scene, vp));
    let peptide = &state.peptides[peptide_i];
    let filtered = filter_pep_atoms_by_dist(state, peptide);
    let mut residues = BTreeSet::new();
    if add || subtract {
        for (i, _) in peptide.residues.iter().enumerate() {
            if state.ui.selection.includes_peptide_residue(
                i,
                &peptide.common.atoms,
                &peptide.common.bonds,
            ) {
                residues.insert(i);
            }
        }
        for object in crate::selection::expression::current_atoms(state) {
            if object.mol_type == MolType::Peptide && object.mol_i == peptide_i {
                for atom in object.atoms {
                    if let Some(residue) = peptide.common.atoms.get(atom).and_then(|a| a.residue) {
                        residues.insert(residue);
                    }
                }
            }
        }
    }
    let mut hits = BTreeSet::new();
    if shape.len() >= 3 && peptide.common.visible {
        for (i, atom) in peptide.common.atoms.iter().enumerate() {
            let Some(residue) = atom.residue else {
                continue;
            };
            let visibility = &state.ui.visibility;
            if filtered[i]
                || atom
                    .chain
                    .and_then(|c| peptide.chains.get(c))
                    .is_some_and(|c| !c.visible)
                || (visibility.hide_hydrogen && atom.element == Element::Hydrogen)
                || (visibility.hide_hetero && atom.hetero)
                || (visibility.hide_protein && !atom.hetero)
                || (visibility.hide_water && atom.role == Some(AtomRole::Water))
                || ((visibility.hide_sidechains
                    || state.ui.mol_view_peptide == MoleculeView::Backbone)
                    && !atom.hetero
                    && matches!(atom.role, Some(AtomRole::Sidechain | AtomRole::H_Sidechain)))
            {
                continue;
            }
            let position: Vec3 = peptide.common.atom_posits[i].into();
            let (visible, ndc) = scene.camera.in_view(position);
            let screen = (
                (vp.0 + (ndc.0 * 0.5 + 0.5) * vp.2) * scale,
                (vp.1 + (0.5 - ndc.1 * 0.5) * vp.3) * scale,
            );
            if !visible || !inside(screen, &shape) {
                continue;
            }
            let radius = match state.ui.mol_view_peptide {
                MoleculeView::SpaceFill => atom.element.vdw_radius(),
                MoleculeView::Ribbon => 0.8,
                MoleculeView::Surface | MoleculeView::Dots => {
                    atom.element.vdw_radius() + mol_defs::sfc_mesh::SOLVENT_RAD
                }
                _ if atom.element == Element::Hydrogen => {
                    crate::drawing::atoms_bonds::BALL_STICK_RADIUS_H
                }
                _ => crate::drawing::atoms_bonds::BALL_STICK_RADIUS,
            };
            if depth
                .as_ref()
                .is_none_or(|depth| depth.visible(scene, position, radius))
            {
                hits.insert(residue);
            }
        }
    }
    if subtract {
        residues.retain(|r| !hits.contains(r));
    } else {
        residues.extend(hits);
    }
    let count = residues.len();
    state.ui.selection = match count {
        0 => Selection::None,
        1 => Selection::Residue(*residues.first().unwrap()),
        _ => Selection::Residues(residues.into_iter().collect()),
    };
    state.volatile.active_mol = Some((MolType::Peptide, peptide_i));
    state.volatile.active_peptide = Some(peptide_i);
    state.ui.view_sel_level = ViewSelLevel::Residue;
    state.volatile.region_selection.summary = format!(
        "{count} residues · {}",
        if through {
            "Through structure"
        } else {
            "Visible only"
        }
    );
    cancel(state, scene);
    redraw.peptide = true;

    true
}

pub(crate) fn show(state: &mut State, scene: &mut Scene, ui: &mut Ui, redraw: &mut RedrawFlags) {
    if state.volatile.operating_mode != OperatingMode::Primary {
        return;
    }
    ui.collapsing("Region selection", |ui| {
        ui.small("Residues in the active protein");
        ui.horizontal_wrapped(|ui| {
            for (tool, label) in [
                (RegionTool::Box, "Box (B)"),
                (RegionTool::Lasso, "Lasso (L)"),
            ] {
                if ui
                    .selectable_label(state.volatile.region_selection.tool == tool, label)
                    .clicked()
                {
                    arm(state, scene, tool);
                }
            }
        });
        ui.radio_value(
            &mut state.volatile.region_selection.through,
            false,
            "Visible atoms only",
        );
        ui.radio_value(
            &mut state.volatile.region_selection.through,
            true,
            "Through structure (X-ray)",
        );
        ui.small("Drag LMB · Shift adds · Ctrl subtracts · Esc cancels");
        if !state.volatile.region_selection.summary.is_empty() {
            ui.label(&state.volatile.region_selection.summary);
        }
        ui.horizontal(|ui| {
            ui.label("Radius Å");
            ui.add(
                egui::DragValue::new(&mut state.volatile.region_selection.radius)
                    .speed(0.1)
                    .range(0.1..=1000.),
            );
        });
        if ui
            .add_enabled(
                state.ui.selection != Selection::None,
                egui::Button::new("Select neighborhood"),
            )
            .clicked()
        {
            crate::edit_history::neighborhood(
                state,
                state.volatile.region_selection.radius,
                redraw,
            );
        }
    });
}

pub(crate) fn paint(state: &State, scene: &Scene, ctx: &egui::Context) {
    let region = &state.volatile.region_selection;
    if region.tool == RegionTool::Off {
        return;
    }
    let scale = ctx.pixels_per_point();
    let vp = crate::gizmo::viewport_points(scene, scale);
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("region_selection"),
        ))
        .with_clip_rect(egui::Rect::from_min_size(
            Pos2::new(vp.0, vp.1),
            egui::vec2(vp.2, vp.3),
        ));
    let color = if region.through {
        Color32::from_rgb(255, 185, 70)
    } else {
        Color32::from_rgb(95, 200, 255)
    };
    let mut points: Vec<_> = polygon(region.tool, &region.points)
        .iter()
        .map(|p| Pos2::new(p.0 / scale, p.1 / scale))
        .collect();
    if !points.is_empty() {
        points.push(points[0]);
        painter.add(egui::Shape::line(points, Stroke::new(2., color)));
    }
    painter.text(
        Pos2::new(vp.0 + 14., vp.1 + 16.),
        egui::Align2::LEFT_TOP,
        format!(
            "{:?} · {} · Shift add / Ctrl subtract · Esc cancel",
            region.tool,
            if region.through {
                "X-RAY: THROUGH STRUCTURE"
            } else {
                "VISIBLE ATOMS ONLY"
            }
        ),
        egui::FontId::proportional(14.),
        color,
    );
}
