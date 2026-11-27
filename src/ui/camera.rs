use egui::{Color32, ComboBox, RichText, Slider, TextEdit, Ui};
use graphics::{ControlScheme, EngineUpdates, Scene};

use crate::{
    cam::{
        self, FOG_DIST_MAX, FOG_DIST_MIN, VIEW_DEPTH_NEAR_MAX, VIEW_DEPTH_NEAR_MIN,
        move_cam_to_sel, set_fog,
    },
    drawing::floor,
    prefs::{ControlSchemeType, DepthMode},
    render::set_flashlight,
    selection::Selection,
    state::State,
    ui,
    ui::{COL_SPACING, COLOR_HIGHLIGHT, misc},
    util::orbit_center,
};

pub(crate) fn cam_controls(
    scene: &mut Scene,
    state: &mut State,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Here and at init, set the camera dist dynamically based on mol size.
    // todo: Set the position not relative to 0, but  relative to the center of the atoms.

    let mut changed = false;

    cam::cam_reset_controls(state, scene, ui, engine_updates, &mut changed);

    ui.add_space(COL_SPACING);

    let free_active = scene.input_settings.control_scheme == ControlScheme::FreeCamera;
    let arc_active = !free_active;

    let scheme_current = if free_active {
        ControlSchemeType::Free
    } else {
        ControlSchemeType::Arc
    };

    let scheme_opts = [
        (
            ControlSchemeType::Free,
            "Free",
            "Set the camera is a first-person mode, where your controls move its position. Similar to video games.",
        ),
        (
            ControlSchemeType::Arc,
            "Arc",
            "Set the camera to orbit around a point: Either the center of the molecule, or the selection.",
        ),
    ];

    if let Some(scheme) = misc::selector(ui, scheme_current, &scheme_opts) {
        scene.input_settings.control_scheme = match scheme {
            ControlSchemeType::Free => ControlScheme::FreeCamera,
            ControlSchemeType::Arc => ControlScheme::Arc {
                center: orbit_center(state),
            },
        };
        state.to_save.control_scheme = scheme;
    }

    if ui
        .button(RichText::new("Floor").color(misc::active_color(state.to_save.ui_prefs.floor_visible)))
        .on_hover_text(
            "Toggle display of a flat surface \"below\" the molecules. This provides an absolute visual \
            reference, which may be helpful when positioning molecules relative to each other, and moving the \
            camera."
        )
        .clicked()
    {
        floor::toggle(state, scene, engine_updates);
    }

    if arc_active
        && ui
            .button(
                RichText::new("Orbit sel").color(misc::active_color(state.ui.orbit_selected_atom)),
            )
            .on_hover_text(
                "Toggle whether the camera orbits around the selection, or the molecule center.",
            )
            .clicked()
    {
        state.ui.orbit_selected_atom = !state.ui.orbit_selected_atom;

        let center = orbit_center(state);
        scene.input_settings.control_scheme = ControlScheme::Arc { center };
    }

    ui.add_space(COL_SPACING);

    if (state.ui.selection != Selection::None || state.active_mol().is_some())
        && ui
        .button(RichText::new("Cam to sel").color(COLOR_HIGHLIGHT))
        .on_hover_text(
            "(Hotkey: Enter) Look at the selected atom or residue, or the active molecule if nothing is selected.",
        )
        .clicked()
    {
        move_cam_to_sel(state, &mut scene.camera, engine_updates);
    }

    ui.add_space(COL_SPACING);
    ui.label("Depth:");
    let depth_mode = state.to_save.depth_mode;
    let depth_options = [
        (
            Some(DepthMode::Disabled),
            "Disabled",
            "Show objects at any depth without distance fading.",
        ),
        (
            Some(DepthMode::Auto),
            "Auto",
            "Automatically adjust the far distance based on visible molecules.",
        ),
        (
            Some(DepthMode::Manual((
                VIEW_DEPTH_NEAR_MIN,
                cam::VIEW_DEPTH_DEFAULT,
            ))),
            "Manual",
            "Set the near clip plane and far fade distance yourself.",
        ),
        (
            None,
            "Slab",
            "Clip to a slice that follows the viewing direction.",
        ),
    ];

    let slab_enabled = state
        .to_save
        .inspection
        .slab
        .as_ref()
        .is_some_and(|s| s.enabled);
    let selected = if slab_enabled {
        None
    } else {
        // Compare Manual by variant, since its distances vary.
        match depth_mode {
            DepthMode::Manual(_) => depth_options[2].0,
            mode => Some(mode),
        }
    };
    if let Some(mode) = misc::selector(ui, selected, &depth_options) {
        let enable_slab = mode.is_none();
        state
            .to_save
            .select_depth_mode(mode.unwrap_or(DepthMode::Disabled));

        if enable_slab
            && state.to_save.inspection.slab.is_none()
            && crate::annotations::center_slab(state, "sele", 10.).is_err()
        {
            state.to_save.inspection.slab = Some(crate::annotations::Slab {
                enabled: true,
                anchor: orbit_center(state).into(),
                offset: 0.,
                thickness: 10.,
            });
        }
        if let Some(slab) = &mut state.to_save.inspection.slab {
            slab.enabled = enable_slab;
        }

        state.to_save.save_flag = true;
        changed = true;
    }

    if state
        .to_save
        .inspection
        .slab
        .as_ref()
        .is_some_and(|s| s.enabled)
    {
        changed |= slab_controls(state, ui);
    } else if let DepthMode::Manual(depth) = &mut state.to_save.depth_mode {
        ui.spacing_mut().slider_width = 60.;
        let depth_prev = *depth;

        let near_help = "Don't render objects closer to the camera than this distance, in Å.";
        misc::wrap_together(ui, "depth_near", |ui| {
            ui.label("Near(×10):").on_hover_text(near_help);
            ui.add(Slider::new(
                &mut depth.0,
                VIEW_DEPTH_NEAR_MIN..=VIEW_DEPTH_NEAR_MAX,
            ))
            .on_hover_text(near_help);
        });

        let far_help = "(Hotkey: Ctrl + scroll) Fade distant objects. This may make it easier to see objects near the camera.";
        misc::wrap_together(ui, "depth_far", |ui| {
            ui.label("Far:").on_hover_text(far_help);
            ui.add(Slider::new(&mut depth.1, FOG_DIST_MIN..=FOG_DIST_MAX))
                .on_hover_text(far_help);
        });

        if *depth != depth_prev {
            changed = true;
        }
    }

    if changed {
        engine_updates.camera = true;
        set_fog(state, &mut scene.camera);

        set_flashlight(scene);
        engine_updates.lighting = true; // flashlight.

        state.ui.cam_snapshot = None;
    }
}

fn slab_controls(state: &mut State, ui: &mut Ui) -> bool {
    let before = state.to_save.inspection.slab.clone();

    if ui.button("10 Å around sel").clicked() {
        state.ui.slab_error = crate::annotations::center_slab(state, "sele", 10.)
            .err()
            .unwrap_or_default();
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
        misc::wrap_together(ui, "slab_slide", |ui| {
            ui.add(
                Slider::new(&mut slab.offset, -100.0..=100.0)
                    .clamping(egui::SliderClamping::Never)
                    .text("Slide:"),
            )
            .on_hover_text("Positive offset moves away from you.");
        });

        ui.label("Offset:");
        ui.add(
            egui::DragValue::new(&mut slab.offset)
                .speed(0.1)
                .suffix(" Å"),
        );

        let thickness = slab.thickness;
        if ui.button("Recenter on selection").clicked() {
            state.ui.slab_error = crate::annotations::center_slab(state, "sele", thickness)
                .err()
                .unwrap_or_default();
        }
    }

    let changed = before != state.to_save.inspection.slab;
    state.to_save.save_flag |= changed;

    if !state.ui.slab_error.is_empty() {
        ui.label(&state.ui.slab_error);
    }

    changed
}

pub(crate) fn cam_snapshots(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.label("Scenes");

    ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(60.))
        .on_hover_text("Choose a name to save this scene as.");

    if ui
        .button("Save")
        .on_hover_text(
            "Save camera, representations, colors, labels, visibility, measurements, and frame.",
        )
        .clicked()
    {
        let name = if !state.ui.cam_snapshot_name.is_empty() {
            state.ui.cam_snapshot_name.clone()
        } else {
            format!("Scene {}", state.cam_snapshots.len() + 1)
        };

        crate::util::save_snap(state, scene, &name);
    }

    let prev_snap = state.ui.cam_snapshot;
    let snap_name = ui::get_snap_name(prev_snap, &state.cam_snapshots);

    misc::wrap_together(ui, "scene_select", |ui| {
        ComboBox::from_id_salt(2)
            .width(80.)
            .selected_text(snap_name)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.ui.cam_snapshot, None, "(None)");
                for (i, _snap) in state.cam_snapshots.iter().enumerate() {
                    ui.selectable_value(
                        &mut state.ui.cam_snapshot,
                        Some(i),
                        ui::get_snap_name(Some(i), &state.cam_snapshots),
                    );
                }
            })
            .response
            .on_hover_text("Restore the saved scene appearance and camera.");
    });

    if let Some(index) = state.ui.cam_snapshot
        && ui.button("Update scene").clicked()
    {
        if let Some(name) = state.cam_snapshots.get(index).map(|s| s.name.clone()) {
            crate::util::save_snap(state, scene, &name);
        }
    }

    if let Some(i) = state.ui.cam_snapshot
        && ui.button(RichText::new("❌").color(Color32::RED)).clicked()
    {
        if i < state.cam_snapshots.len() {
            state.cam_snapshots.remove(i);
        }
        state.ui.cam_snapshot = None;
        state.update_save_prefs();
    }

    if state.ui.cam_snapshot != prev_snap {
        crate::util::load_snap(state, scene, engine_updates);
    }
}
