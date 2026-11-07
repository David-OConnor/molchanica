//! Camera and molecule transform gizmos.

use std::f32::consts::TAU;

use egui::{Align2, FontFamily};
use graphics::{
    Camera, FWD_VEC, OverlayAnchor, OverlayColor, OverlayPrimitive, OverlayStroke, RIGHT_VEC,
    Scene, UP_VEC, VectorOverlay,
};
use lin_alg::f32::{Quaternion, Vec3};
use mol_defs::molecules::MolType;

use crate::{
    mol_manip::{self, Axis, Constraint, Handle, ManipMode},
    state::{OperatingMode, State},
};

const X_COLOR: OverlayColor = (235, 78, 78, 255);
const Y_COLOR: OverlayColor = (92, 205, 112, 255);
const Z_COLOR: OverlayColor = (82, 142, 245, 255);
const TEXT_COLOR: OverlayColor = (225, 230, 238, 235);
const MUTED_COLOR: OverlayColor = (170, 180, 195, 100);
const HIGHLIGHT: OverlayColor = (255, 225, 90, 255);
const RING_RADIUS: f32 = 43.;
const RING_SEGMENTS: usize = 64;

#[derive(Clone, Copy, Debug)]
struct AxisProjection {
    axis: Axis,
    label: &'static str,
    color: OverlayColor,
    /// Unit axis in screen-aligned view coordinates; see `world_to_view`.
    view: Vec3,
}

/// Rotate a world-space direction into screen-aligned view coordinates: +X right, +Y up,
/// and +Z into the screen. This accounts for the camera's handedness, which may mirror X.
fn world_to_view(camera: &Camera, dir: Vec3) -> Vec3 {
    let mut result = camera.orientation.inverse().rotate_vec(dir);
    result.x *= camera.screen_x_sign();

    result
}

fn project_axes(camera: &Camera, frame_orientation: Quaternion) -> [AxisProjection; 3] {
    [
        (Axis::X, "X", X_COLOR, RIGHT_VEC),
        (Axis::Y, "Y", Y_COLOR, UP_VEC),
        (Axis::Z, "Z", Z_COLOR, FWD_VEC),
    ]
    .map(|(axis, label, color, vector)| AxisProjection {
        axis,
        label,
        color,
        view: world_to_view(camera, frame_orientation.rotate_vec(vector)),
    })
}

fn screen_point(view: Vec3, radius: f32) -> (f32, f32) {
    (view.x * radius, -view.y * radius)
}

fn magnitude_2d((x, y): (f32, f32)) -> f32 {
    (x * x + y * y).sqrt()
}

fn scaled((x, y): (f32, f32), scale: f32) -> (f32, f32) {
    (x * scale, y * scale)
}

fn add_axis(
    primitives: &mut Vec<OverlayPrimitive>,
    axis: AxisProjection,
    radius: f32,
    arrowhead: bool,
) {
    let end = screen_point(axis.view, radius);
    let projected_len = magnitude_2d(end);

    primitives.push(OverlayPrimitive::Line {
        start: scaled(end, -0.42),
        end: (0., 0.),
        stroke: OverlayStroke::new(1.25, MUTED_COLOR),
    });
    primitives.push(OverlayPrimitive::Line {
        start: (0., 0.),
        end,
        stroke: OverlayStroke::new(2.25, axis.color),
    });

    if arrowhead {
        if projected_len > 7. {
            let direction = (end.0 / projected_len, end.1 / projected_len);
            let perpendicular = (-direction.1, direction.0);
            let base = (end.0 - direction.0 * 9., end.1 - direction.1 * 9.);
            primitives.push(OverlayPrimitive::Polygon {
                points: vec![
                    end,
                    (
                        base.0 + perpendicular.0 * 4.5,
                        base.1 + perpendicular.1 * 4.5,
                    ),
                    (
                        base.0 - perpendicular.0 * 4.5,
                        base.1 - perpendicular.1 * 4.5,
                    ),
                ],
                fill: axis.color,
                stroke: None,
            });
        } else {
            // A view-aligned axis projects to a point, so retain a visible tip.
            primitives.push(OverlayPrimitive::Circle {
                center: end,
                radius: 5.,
                fill: Some(axis.color),
                stroke: Some(OverlayStroke::new(1., (255, 255, 255, 145))),
            });
        }
    } else {
        // The navigation gizmo uses Blender-style labeled pucks on each positive axis.
        primitives.push(OverlayPrimitive::Circle {
            center: end,
            radius: 9.,
            fill: Some(axis.color),
            stroke: Some(OverlayStroke::new(1., (255, 255, 255, 145))),
        });
        primitives.push(OverlayPrimitive::Text {
            position: end,
            text: axis.label.to_owned(),
            size: 12.,
            color: (255, 255, 255, 255),
            align: Align2::CENTER_CENTER,
            font_family: FontFamily::Proportional,
        });
    }
}

fn overlay_at(scene: &Scene, scale: f32, point: (f32, f32)) -> VectorOverlay {
    let (x, y, _, _) = graphics::viewport_rect(
        scene.gui_size,
        (scene.window_size.0 / scale).round() as u32,
        (scene.window_size.1 / scale).round() as u32,
        &graphics::UiSettings::default(),
        scale,
    );
    let mut overlay = VectorOverlay::new(OverlayAnchor::ViewportTopLeft);
    overlay.offset = (point.0 - x, point.1 - y);

    overlay
}

fn camera_overlay(state: &State, scene: &Scene) -> VectorOverlay {
    const RADIUS: f32 = 42.;

    let scale = state.volatile.mol_manip.pixels_per_point;
    let (x, y, w, _) = viewport_points(scene, scale);
    let mut overlay = overlay_at(scene, scale, (x + w - 72., y + 70.));
    overlay.primitives.push(OverlayPrimitive::Circle {
        center: (0., 0.),
        radius: 55.,
        fill: Some((12, 16, 24, 155)),
        stroke: Some(OverlayStroke::new(1., (190, 205, 225, 100))),
    });

    let axes = project_axes(&scene.camera, Quaternion::new_identity());
    for axis in axes {
        overlay.primitives.push(OverlayPrimitive::Line {
            start: screen_point(axis.view, -RADIUS * 0.8),
            end: screen_point(axis.view, RADIUS),
            stroke: OverlayStroke::new(2., axis.color),
        });
    }
    let mut ends: Vec<_> = axes
        .into_iter()
        .flat_map(|a| [(a, true), (a, false)])
        .collect();
    ends.sort_by(|(a, positive_a), (b, positive_b)| {
        (b.view.z * if *positive_b { 1. } else { -1. })
            .total_cmp(&(a.view.z * if *positive_a { 1. } else { -1. }))
    });
    for (axis, positive) in ends {
        let end = screen_point(axis.view, RADIUS * if positive { 1. } else { -0.8 });
        let color = if state.volatile.mol_manip.hovered == Some(Handle::Camera(axis.axis, positive))
        {
            HIGHLIGHT
        } else {
            axis.color
        };
        overlay.primitives.push(OverlayPrimitive::Circle {
            center: end,
            radius: if positive { 9. } else { 5. },
            fill: Some(color),
            stroke: Some(OverlayStroke::new(1., (255, 255, 255, 150))),
        });
        if positive {
            overlay.primitives.push(OverlayPrimitive::Text {
                position: end,
                text: axis.label.to_owned(),
                size: 12.,
                color: (255, 255, 255, 255),
                align: Align2::CENTER_CENTER,
                font_family: FontFamily::Proportional,
            });
        }
    }

    let p = scene.camera.position;
    overlay.primitives.push(OverlayPrimitive::Text {
        position: (55., 61.),
        text: format!("CAM  x {:+.1}  y {:+.1}  z {:+.1}", p.x, p.y, p.z),
        size: 11.,
        color: TEXT_COLOR,
        align: Align2::RIGHT_TOP,
        font_family: FontFamily::Monospace,
    });

    overlay
}

fn add_rotation_rings(
    primitives: &mut Vec<OverlayPrimitive>,
    camera: &Camera,
    frame_orientation: Quaternion,
    selected: Option<Axis>,
) {
    let mut back = Vec::new();
    let mut front = Vec::new();
    for (axis, color) in [(Axis::X, X_COLOR), (Axis::Y, Y_COLOR), (Axis::Z, Z_COLOR)] {
        let (a, b) = axis.plane_axes();
        let ring_color = if selected == Some(axis) {
            HIGHLIGHT
        } else {
            color
        };
        for i in 0..RING_SEGMENTS {
            let angle_a = TAU * i as f32 / RING_SEGMENTS as f32;
            let angle_b = TAU * (i + 1) as f32 / RING_SEGMENTS as f32;
            let local_a = a * angle_a.cos() + b * angle_a.sin();
            let local_b = a * angle_b.cos() + b * angle_b.sin();
            let view_a = world_to_view(camera, frame_orientation.rotate_vec(local_a));
            let view_b = world_to_view(camera, frame_orientation.rotate_vec(local_b));
            let is_front = (view_a.z + view_b.z) * 0.5 < 0.;
            let alpha = if is_front { 225 } else { 65 };
            let segment = OverlayPrimitive::Line {
                start: screen_point(view_a, RING_RADIUS),
                end: screen_point(view_b, RING_RADIUS),
                stroke: OverlayStroke::new(
                    if is_front { 2.25 } else { 1.25 },
                    (ring_color.0, ring_color.1, ring_color.2, alpha),
                ),
            };
            if is_front {
                front.push(segment);
            } else {
                back.push(segment);
            }
        }
    }
    primitives.extend(back);
    primitives.extend(front);
}

fn mol_type_name(mol_type: MolType) -> &'static str {
    match mol_type {
        MolType::Peptide => "peptide",
        MolType::Ligand => "ligand",
        MolType::NucleicAcid => "nucleic acid",
        MolType::Lipid => "lipid",
        MolType::Pocket => "pocket",
        MolType::Water => "water",
    }
}

pub(crate) fn manipulation_position(
    state: &State,
    mol_type: MolType,
    index: usize,
) -> Option<Vec3> {
    match state.volatile.operating_mode {
        OperatingMode::Primary => state
            .get_mol(mol_type, index)
            .map(|mol| mol.common().centroid().into()),
        OperatingMode::MolEditor => match mol_type {
            MolType::Ligand => match state.volatile.mol_manip.mode {
                ManipMode::Move(_) => state
                    .mol_editor
                    .mol
                    .common
                    .atom_posits
                    .get(index)
                    .copied()
                    .map(Into::into),
                ManipMode::Rotate(_) => {
                    let mol = &state.mol_editor.mol.common;
                    let bond = mol.bonds.get(index)?;

                    Some(
                        ((mol.atom_posits[bond.atom_0] + mol.atom_posits[bond.atom_1]) * 0.5)
                            .into(),
                    )
                }
                ManipMode::None => None,
            },
            MolType::Pocket => state
                .mol_editor
                .mol
                .pharmacophore
                .pocket
                .as_ref()
                .map(|pocket| pocket.common.centroid().into()),
            _ => None,
        },
        OperatingMode::ProteinEditor => None,
    }
}

fn molecule_overlay(state: &State, scene: &Scene) -> Option<VectorOverlay> {
    const AXIS_RADIUS: f32 = 48.;

    let (mode_label, mol_type, index, rotating) = match state.volatile.mol_manip.mode {
        ManipMode::None => return None,
        ManipMode::Move((mol_type, index)) => ("MOVE", mol_type, index, false),
        ManipMode::Rotate((mol_type, index)) => ("ROTATE", mol_type, index, true),
    };
    let position = manipulation_position(state, mol_type, index)?;
    let orientation = Quaternion::new_identity();
    let mut overlay = overlay_at(
        scene,
        state.volatile.mol_manip.pixels_per_point,
        molecule_screen_center(state, scene)?,
    );

    overlay.primitives.push(OverlayPrimitive::Circle {
        center: (0., 0.),
        radius: 6.,
        fill: Some((235, 240, 250, 190)),
        stroke: Some(OverlayStroke::new(1.5, (20, 24, 32, 230))),
    });

    if rotating {
        let selected = match state
            .volatile
            .mol_manip
            .dragged
            .or(state.volatile.mol_manip.hovered)
        {
            Some(Handle::Rotate(axis)) => Some(axis),
            _ => match state.volatile.mol_manip.constraint {
                Constraint::Axis(axis) => Some(axis),
                _ => None,
            },
        };
        add_rotation_rings(
            &mut overlay.primitives,
            &scene.camera,
            orientation,
            selected,
        );
    } else {
        for axis in Axis::ALL {
            let points = plane_points(&scene.camera, axis);
            let selected = state
                .volatile
                .mol_manip
                .dragged
                .or(state.volatile.mol_manip.hovered)
                == Some(Handle::MovePlane(axis))
                || state.volatile.mol_manip.constraint == Constraint::Plane(axis);
            let color = if selected {
                HIGHLIGHT
            } else {
                axis_color(axis)
            };
            overlay.primitives.push(OverlayPrimitive::Polygon {
                points: points.to_vec(),
                fill: (color.0, color.1, color.2, if selected { 150 } else { 60 }),
                stroke: Some(OverlayStroke::new(1.5, color)),
            });
        }
    }

    let mut axes = project_axes(&scene.camera, orientation);
    axes.sort_by(|a, b| b.view.z.total_cmp(&a.view.z));
    for mut axis in axes {
        if !rotating
            && (state
                .volatile
                .mol_manip
                .dragged
                .or(state.volatile.mol_manip.hovered)
                == Some(Handle::MoveAxis(axis.axis))
                || state.volatile.mol_manip.constraint == Constraint::Axis(axis.axis))
        {
            axis.color = HIGHLIGHT;
        }
        add_axis(
            &mut overlay.primitives,
            axis,
            if rotating {
                AXIS_RADIUS * 0.72
            } else {
                AXIS_RADIUS
            },
            true,
        );
    }

    overlay.primitives.push(OverlayPrimitive::Text {
        position: (0., 60.),
        text: format!(
            "{mode_label} {}  x {:+.2}  y {:+.2}  z {:+.2}",
            mol_type_name(mol_type),
            position.x,
            position.y,
            position.z
        ),
        size: 11.,
        color: TEXT_COLOR,
        align: Align2::CENTER_TOP,
        font_family: FontFamily::Monospace,
    });
    overlay.primitives.push(OverlayPrimitive::Text {
        position: (0., 76.),
        text: mol_manip::status(state),
        size: 12.,
        color: HIGHLIGHT,
        align: Align2::CENTER_TOP,
        font_family: FontFamily::Monospace,
    });

    Some(overlay)
}

/// Rebuild the small overlay command lists from the live camera and manipulation state.
pub(crate) fn update_overlays(state: &mut State, scene: &mut Scene, ctx: &egui::Context) {
    state.volatile.mol_manip.pixels_per_point = ctx.pixels_per_point();
    let cursor = ctx.input(|input| input.pointer.hover_pos().filter(|_| input.focused));
    state.volatile.mol_manip.hovered = cursor.and_then(|p| {
        let cursor = (p.x * ctx.pixels_per_point(), p.y * ctx.pixels_per_point());
        state.ui.cursor_pos = Some(cursor);

        hit_test(state, scene, cursor)
    });
    let molecule = molecule_overlay(state, scene);
    scene.vector_overlays.clear();
    scene.vector_overlays.push(camera_overlay(state, scene));
    if let Some(molecule) = molecule {
        scene.vector_overlays.push(molecule);
    }
}

fn axis_color(axis: Axis) -> OverlayColor {
    match axis {
        Axis::X => X_COLOR,
        Axis::Y => Y_COLOR,
        Axis::Z => Z_COLOR,
    }
}

fn plane_points(camera: &Camera, axis: Axis) -> [(f32, f32); 4] {
    let (a, b) = axis.plane_axes();

    [(0.22, 0.22), (0.48, 0.22), (0.48, 0.48), (0.22, 0.48)]
        .map(|(x, y)| screen_point(world_to_view(camera, a * x + b * y), 48.))
}

fn segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let v = (b.0 - a.0, b.1 - a.1);
    let t = (((p.0 - a.0) * v.0 + (p.1 - a.1) * v.1) / (v.0 * v.0 + v.1 * v.1).max(1.0e-6))
        .clamp(0., 1.);

    magnitude_2d((p.0 - a.0 - t * v.0, p.1 - a.1 - t * v.1))
}

pub(crate) fn viewport_points(scene: &Scene, scale: f32) -> (f32, f32, f32, f32) {
    let (x, y, w, h) = graphics::viewport_rect(
        scene.gui_size,
        scene.window_size.0 as u32,
        scene.window_size.1 as u32,
        &graphics::UiSettings::default(),
        scale,
    );

    (x / scale, y / scale, w / scale, h / scale)
}

fn molecule_screen_center(state: &State, scene: &Scene) -> Option<(f32, f32)> {
    let (kind, index) = state.volatile.mol_manip.mode.target()?;
    let position = manipulation_position(state, kind, index)?;
    let (visible, ndc) = scene.camera.in_view(position);
    if !visible {
        return None;
    }
    let (x, y, w, h) = viewport_points(scene, state.volatile.mol_manip.pixels_per_point);

    Some((x + (ndc.0 * 0.5 + 0.5) * w, y + (0.5 - ndc.1 * 0.5) * h))
}

pub(crate) fn hit_test(state: &State, scene: &Scene, cursor: (f32, f32)) -> Option<Handle> {
    let scale = state.volatile.mol_manip.pixels_per_point;
    let cursor = (cursor.0 / scale, cursor.1 / scale);
    let (x, y, w, h) = viewport_points(scene, scale);
    if cursor.0 < x || cursor.1 < y || cursor.0 > x + w || cursor.1 > y + h {
        return None;
    }

    let camera_p = (cursor.0 - (x + w - 72.), cursor.1 - (y + 70.));
    let mut axes = project_axes(&scene.camera, Quaternion::new_identity());
    axes.sort_by(|a, b| a.view.z.total_cmp(&b.view.z));
    let mut ends: Vec<_> = axes
        .into_iter()
        .flat_map(|a| [(a, true), (a, false)])
        .collect();
    ends.sort_by(|(a, positive_a), (b, positive_b)| {
        (a.view.z * if *positive_a { 1. } else { -1. })
            .total_cmp(&(b.view.z * if *positive_b { 1. } else { -1. }))
    });
    for (axis, positive) in ends {
        let end = screen_point(axis.view, if positive { 42. } else { -42. * 0.8 });
        if magnitude_2d((camera_p.0 - end.0, camera_p.1 - end.1)) <= if positive { 11. } else { 7. }
        {
            return Some(Handle::Camera(axis.axis, positive));
        }
    }
    let center = molecule_screen_center(state, scene)?;
    let p = (cursor.0 - center.0, cursor.1 - center.1);
    if matches!(state.volatile.mol_manip.mode, ManipMode::Rotate(_)) {
        let mut best = (f32::INFINITY, Axis::X);
        for axis in Axis::ALL {
            for i in 0..RING_SEGMENTS {
                let a = ring_point(&scene.camera, axis, TAU * i as f32 / RING_SEGMENTS as f32);
                let b = ring_point(
                    &scene.camera,
                    axis,
                    TAU * (i + 1) as f32 / RING_SEGMENTS as f32,
                );
                let distance = segment_distance(p, a, b);
                if distance < best.0 {
                    best = (distance, axis);
                }
            }
        }

        return (best.0 <= 6.).then_some(Handle::Rotate(best.1));
    }
    for axis in Axis::ALL {
        let points = plane_points(&scene.camera, axis);
        let cross = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| {
            (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
        };
        let area = cross(points[0], points[1], points[3]);
        if area.abs() > 8. && (0..4).all(|i| cross(points[i], points[(i + 1) % 4], p) * area >= 0.)
        {
            return Some(Handle::MovePlane(axis));
        }
    }
    let mut best = (f32::INFINITY, Axis::X);
    for axis in axes {
        let end = screen_point(axis.view, 48.);
        let distance = segment_distance(p, scaled(end, 0.24), end);
        if distance < best.0 {
            best = (distance, axis.axis);
        }
    }

    (best.0 <= 7.).then_some(Handle::MoveAxis(best.1))
}

fn ring_point(camera: &Camera, axis: Axis, angle: f32) -> (f32, f32) {
    let (a, b) = axis.plane_axes();

    screen_point(
        world_to_view(camera, a * angle.cos() + b * angle.sin()),
        RING_RADIUS,
    )
}

pub(crate) fn ring_tangent_at(
    state: &State,
    scene: &Scene,
    axis: Axis,
    cursor: (f32, f32),
) -> Option<(f32, f32)> {
    let center = molecule_screen_center(state, scene)?;
    let scale = state.volatile.mol_manip.pixels_per_point;
    let p = (cursor.0 / scale - center.0, cursor.1 / scale - center.1);
    let (a, b) = axis.plane_axes();
    let mut best = (f32::INFINITY, (0., 0.));
    for i in 0..RING_SEGMENTS {
        let angle = TAU * i as f32 / RING_SEGMENTS as f32;
        let step = TAU / RING_SEGMENTS as f32;
        let start = ring_point(&scene.camera, axis, angle);
        let end = ring_point(&scene.camera, axis, angle + step);
        let midpoint = angle + step * 0.5;
        let depth = world_to_view(&scene.camera, a * midpoint.cos() + b * midpoint.sin()).z;
        let score = segment_distance(p, start, end) + depth * 1.0e-3;
        if score < best.0 {
            best = (score, (end.0 - start.0, end.1 - start.1));
        }
    }

    Some(best.1)
}

pub(crate) fn ring_drag_delta(
    state: &State,
    scene: &Scene,
    axis: Axis,
    previous: (f32, f32),
    cursor: (f32, f32),
) -> f32 {
    let Some(center) = molecule_screen_center(state, scene) else {
        return 0.;
    };
    let scale = state.volatile.mol_manip.pixels_per_point;
    let p = (previous.0 / scale - center.0, previous.1 / scale - center.1);
    let q = (cursor.0 / scale - center.0, cursor.1 / scale - center.1);
    let a = ring_point(&scene.camera, axis, 0.);
    let b = ring_point(&scene.camera, axis, TAU * 0.25);
    let determinant = a.0 * b.1 - a.1 * b.0;
    if determinant.abs() > RING_RADIUS * RING_RADIUS * 0.08 {
        let angle = |p: (f32, f32)| {
            ((a.0 * p.1 - a.1 * p.0) / determinant).atan2((p.0 * b.1 - p.1 * b.0) / determinant)
        };
        let delta = angle(q) - angle(p);

        return (delta + TAU * 0.5).rem_euclid(TAU) - TAU * 0.5;
    }
    let mut best = (f32::INFINITY, (0., 0.));
    for i in 0..RING_SEGMENTS {
        let angle = TAU * i as f32 / RING_SEGMENTS as f32;
        let start = ring_point(&scene.camera, axis, angle);
        let end = ring_point(&scene.camera, axis, angle + TAU / RING_SEGMENTS as f32);
        let distance = segment_distance(p, start, end);
        if distance < best.0 {
            best = (distance, (end.0 - start.0, end.1 - start.1));
        }
    }
    let tangent = state.volatile.mol_manip.ring_tangent.unwrap_or(best.1);

    ((q.0 - p.0) * tangent.0 + (q.1 - p.1) * tangent.1)
        / magnitude_2d(tangent).max(1.0e-3)
        / RING_RADIUS
}
