//! View-only camera and molecule transform gizmos.

use std::f32::consts::TAU;

use egui::{Align2, FontFamily};
use graphics::{
    FWD_VEC, OverlayAnchor, OverlayColor, OverlayPrimitive, OverlayStroke, RIGHT_VEC, Scene,
    UP_VEC, VectorOverlay,
};
use lin_alg::f32::{Quaternion, Vec3};
use mol_defs::molecules::MolType;

use crate::{
    mol_manip::ManipMode,
    state::{OperatingMode, State},
};

const X_COLOR: OverlayColor = (235, 78, 78, 255);
const Y_COLOR: OverlayColor = (92, 205, 112, 255);
const Z_COLOR: OverlayColor = (82, 142, 245, 255);
const TEXT_COLOR: OverlayColor = (225, 230, 238, 235);
const MUTED_COLOR: OverlayColor = (170, 180, 195, 100);

#[derive(Clone, Copy, Debug)]
struct AxisProjection {
    label: &'static str,
    color: OverlayColor,
    /// Unit axis in camera/view coordinates.
    view: Vec3,
}

fn project_axes(
    camera_orientation: Quaternion,
    frame_orientation: Quaternion,
) -> [AxisProjection; 3] {
    let world_to_view = camera_orientation.inverse();
    [
        ("X", X_COLOR, RIGHT_VEC),
        ("Y", Y_COLOR, UP_VEC),
        ("Z", Z_COLOR, FWD_VEC),
    ]
    .map(|(label, color, axis)| AxisProjection {
        label,
        color,
        view: world_to_view.rotate_vec(frame_orientation.rotate_vec(axis)),
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

fn camera_overlay(scene: &Scene) -> VectorOverlay {
    const RADIUS: f32 = 42.;

    let mut overlay = VectorOverlay::new(OverlayAnchor::ViewportTopRight);
    overlay.offset = (-72., 70.);
    overlay.primitives.push(OverlayPrimitive::Circle {
        center: (0., 0.),
        radius: 55.,
        fill: Some((12, 16, 24, 155)),
        stroke: Some(OverlayStroke::new(1., (190, 205, 225, 100))),
    });

    // Far-pointing axes first and near-pointing axes last makes crossings read as a tiny 3D frame.
    let mut axes = project_axes(scene.camera.orientation, Quaternion::new_identity());
    axes.sort_by(|a, b| b.view.z.total_cmp(&a.view.z));
    for axis in axes {
        add_axis(&mut overlay.primitives, axis, RADIUS, false);
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
    camera_orientation: Quaternion,
    frame_orientation: Quaternion,
) {
    const RADIUS: f32 = 43.;
    const SEGMENTS: usize = 48;

    let world_to_view = camera_orientation.inverse();
    let rings = [
        (X_COLOR, UP_VEC, FWD_VEC),
        (Y_COLOR, FWD_VEC, RIGHT_VEC),
        (Z_COLOR, RIGHT_VEC, UP_VEC),
    ];

    let mut back = Vec::new();
    let mut front = Vec::new();
    for (ring_color, a, b) in rings {
        for i in 0..SEGMENTS {
            let angle_a = TAU * i as f32 / SEGMENTS as f32;
            let angle_b = TAU * (i + 1) as f32 / SEGMENTS as f32;
            let local_a = a * angle_a.cos() + b * angle_a.sin();
            let local_b = a * angle_b.cos() + b * angle_b.sin();
            let view_a = world_to_view.rotate_vec(frame_orientation.rotate_vec(local_a));
            let view_b = world_to_view.rotate_vec(frame_orientation.rotate_vec(local_b));
            let is_front = (view_a.z + view_b.z) * 0.5 < 0.;
            let alpha = if is_front { 225 } else { 65 };
            let segment = OverlayPrimitive::Line {
                start: screen_point(view_a, RADIUS),
                end: screen_point(view_b, RADIUS),
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

fn manipulation_position(state: &State, mol_type: MolType, index: usize) -> Option<Vec3> {
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
                ManipMode::Rotate(_) => Some(state.mol_editor.mol.common.centroid().into()),
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
    let orientation = state.volatile.mol_manip.gizmo_orientation;
    let mut overlay = VectorOverlay::new(OverlayAnchor::World(position));

    overlay.primitives.push(OverlayPrimitive::Circle {
        center: (0., 0.),
        radius: 6.,
        fill: Some((235, 240, 250, 190)),
        stroke: Some(OverlayStroke::new(1.5, (20, 24, 32, 230))),
    });

    if rotating {
        add_rotation_rings(
            &mut overlay.primitives,
            scene.camera.orientation,
            orientation,
        );
    }

    let mut axes = project_axes(scene.camera.orientation, orientation);
    axes.sort_by(|a, b| b.view.z.total_cmp(&a.view.z));
    for axis in axes {
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
        position: (0., 57.),
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

    Some(overlay)
}

/// Rebuild the small overlay command lists from the live camera and manipulation state.
pub(crate) fn update_overlays(state: &State, scene: &mut Scene) {
    let molecule = molecule_overlay(state, scene);
    scene.vector_overlays.clear();
    scene.vector_overlays.push(camera_overlay(scene));
    if let Some(molecule) = molecule {
        scene.vector_overlays.push(molecule);
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn camera_roll_rotates_the_screen_axes() {
        let camera = Quaternion::from_axis_angle(FWD_VEC, FRAC_PI_2);
        let axes = project_axes(camera, Quaternion::new_identity());

        assert_close(axes[0].view.x, 0.);
        assert_close(axes[0].view.y, -1.);
        assert_close(axes[1].view.x, 1.);
        assert_close(axes[1].view.y, 0.);
    }

    #[test]
    fn object_orientation_is_composed_before_camera_projection() {
        let object = Quaternion::from_axis_angle(UP_VEC, FRAC_PI_2);
        let axes = project_axes(Quaternion::new_identity(), object);

        assert_close(axes[0].view.x, 0.);
        assert_close(axes[0].view.z, -1.);
        assert_close(axes[2].view.x, 1.);
        assert_close(axes[2].view.z, 0.);
    }
}
