use std::ops::RangeInclusive;

use egui::{Color32, ComboBox, Grid, Response, RichText, Slider, Ui};
use graphics::{AmbientOcclusion, EngineUpdates, Scene};

use crate::{
    inputs::{MOVEMENT_SENS, ROTATE_SENS, SENS_MOL_MOVE_SCROLL},
    prefs::Graphics,
    state::{LabelSizes, MsaaSetting, OperatingMode, State},
    ui::{COL_SPACING, ROW_SPACING, misc::section_box},
    util::RedrawFlags,
};

const SLIDER_WIDTH: f32 = 160.;
/// The label column's width is shared by every section's grid, so their controls line up.
const LABEL_COL_WIDTH: f32 = 180.;

// Slider ranges for the graphics settings, in the engine's own units — no scaling between
// what the slider shows and what `GraphicsSettings` receives.
/// World-space SSAO sample radius, in Å.
const SSAO_RADIUS_RANGE: RangeInclusive<f32> = 0.1..=40.0;
/// Edge-cueing strength. Unitless; the engine treats 0 as disabled.
const EDGE_CUEING_RANGE: RangeInclusive<f32> = 0.0..=3.0;
/// Mesh inflation for depth-aware halos, in Å.
const HALO_RANGE: RangeInclusive<f32> = 0.0..=0.1;

// These are of the saved integer values; see `ControlSettings` for how each maps to the engine.
const MOVEMENT_SPEED_RANGE: RangeInclusive<u8> = 1..=100;
const ROTATION_SENS_RANGE: RangeInclusive<u8> = 1..=200;
const MOL_MOVE_SENS_RANGE: RangeInclusive<u8> = 1..=200;

pub(in crate::ui) fn settings(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    ui.heading("Settings");
    ui.add_space(ROW_SPACING / 2.);

    section(ui, "Graphics", |ui| {
        graphics_settings(state, ui, updates);
    });

    ui.add_space(ROW_SPACING);

    section(ui, "Camera and molecule movement", |ui| {
        sensitivities(state, scene, ui);
    });

    ui.add_space(ROW_SPACING);

    section(ui, "Label font sizes", |ui| {
        label_sizes(state, ui, redraw);
    });
}

/// A titled box around a group of related settings, to set it apart from its neighbors.
fn section(ui: &mut Ui, title: &str, contents: impl FnOnce(&mut Ui)) {
    section_box().show(ui, |ui| {
        // Fill the window's width, so the sections line up.
        ui.set_min_width(ui.available_width());

        ui.add_space(ROW_SPACING / 4.);
        ui.strong(title);
        ui.add_space(ROW_SPACING / 4.);

        contents(ui);
        ui.add_space(ROW_SPACING / 4.);
    });
}

/// One row of a two-column settings grid: a label, and its slider.
fn slider_row(ui: &mut Ui, label: &str, help: &str, slider: Slider) -> Response {
    ui.label(label).on_hover_text(help);
    let resp = ui.add(slider).on_hover_text(help);
    ui.end_row();

    resp
}

fn size_slider(val: &mut f32) -> Slider<'_> {
    Slider::new(val, LabelSizes::MIN..=LabelSizes::MAX).integer()
}

fn settings_grid(id: &str) -> Grid {
    Grid::new(id)
        .num_columns(2)
        .min_col_width(LABEL_COL_WIDTH)
        .spacing([COL_SPACING / 2., ROW_SPACING / 2.])
}

/// Returns true if clicked.
fn reset_button(ui: &mut Ui, text: &str) -> bool {
    ui.add_space(ROW_SPACING / 2.);
    ui.button(RichText::new(text).color(Color32::LIGHT_RED))
        .clicked()
}

/// True on the frame a slider interaction ends — the drag is released, or a keyboard edit of
/// the value box is committed. Dragging changes the value every frame, so this is what
/// distinguishes "the user is still choosing" from "the user has chosen", and keeps prefs
/// writes to one per adjustment.
///
/// Checked outside the value-changed branch on purpose: the frame a drag ends is usually not
/// a frame on which the value moved.
fn interaction_finished(resp: &Response) -> bool {
    resp.drag_stopped() || resp.lost_focus()
}

fn graphics_settings(state: &mut State, ui: &mut Ui, updates: &mut EngineUpdates) {
    // Set whenever a value changes, including mid-drag: the 3D view should track a slider
    // live. Writing the prefs file is gated separately, since that serializes the whole
    // state and hits the disk — doing it once per frame of a drag is what made saves spam.
    let mut changed = false;
    // Set only when an interaction *finishes* (drag released, or a discrete control used).
    let mut save = false;

    ui.spacing_mut().slider_width = SLIDER_WIDTH;
    let gr = &mut state.to_save.graphics;

    settings_grid("graphics").show(ui, |ui| {
        let msaa_prev = gr.msaa;

        ui.label("MSAA:").on_hover_text(
            "Multi-sample anti-aliasing. Reduces jagged edges. Recommend to enable.",
        );
        ComboBox::from_id_salt("msaa")
            .width(40.)
            .selected_text(gr.msaa.to_str())
            .show_ui(ui, |ui| {
                for v in [MsaaSetting::None, MsaaSetting::Four] {
                    ui.selectable_value(&mut gr.msaa, v, v.to_str());
                }
            });
        ui.end_row();

        if gr.msaa != msaa_prev {
            // An explicit choice; stop following the GPU-based default.
            gr.msaa_user_set = true;
            changed = true;
            save = true;
        }

        ui.label("Ambient occlusion:");
        let mut ao_en = gr.ambient_occlusion == AmbientOcclusion::Ssao;
        if ui.checkbox(&mut ao_en, "").changed() {
            gr.ambient_occlusion = if ao_en {
                AmbientOcclusion::Ssao
            } else {
                AmbientOcclusion::None
            };

            changed = true;
            save = true;
        }
        ui.end_row();

        {
            let help = "How far ambient occlusion looks for occluders, in Å. Small values shade \
                only tight crevices; large ones shade broad hollows, but lose fine detail.";

            ui.label("AO radius (Å):").on_hover_text(help);
            let resp = ui
                .add_enabled(
                    ao_en,
                    Slider::new(&mut gr.ssao_radius, SSAO_RADIUS_RANGE).fixed_decimals(1),
                )
                .on_hover_text(help);
            ui.end_row();

            changed |= resp.changed();
            save |= interaction_finished(&resp);
        }

        {
            let mut val = gr.edge_cueing.unwrap_or_default();
            let resp = slider_row(
                ui,
                "Edge cueing (strength):",
                "Darkens surfaces as they turn away from the camera. 0 disables it.",
                Slider::new(&mut val, EDGE_CUEING_RANGE).fixed_decimals(2),
            );

            if resp.changed() {
                // The engine takes `None` for "off"; the slider bottoming out means the same thing.
                gr.edge_cueing = if val <= 0. { None } else { Some(val) };
                changed = true;
            }
            save |= interaction_finished(&resp);
        }

        {
            let mut val = gr.depth_aware_halos.unwrap_or_default();
            let resp = slider_row(
                ui,
                "Halos (Å):",
                "Mesh inflation, in Å, used to draw dark rings where near geometry overlaps far \
                geometry. 0 disables it.",
                Slider::new(&mut val, HALO_RANGE).fixed_decimals(3),
            );

            if resp.changed() {
                gr.depth_aware_halos = if val <= 0. { None } else { Some(val) };
                changed = true;
            }
            save |= interaction_finished(&resp);
        }
    });

    if reset_button(ui, "Reset graphics") {
        *gr = Graphics::default();
        // Let MSAA follow the GPU again: `apply_msaa_gpu_default` re-runs on the next frame.
        state.volatile.msaa_gpu_default_applied = false;

        changed = true;
        save = true;
    }

    if changed {
        // Re-derive the engine settings from the prefs, rather than mirroring each control into
        // `graphics_settings` as it changes: `to_engine` is the one place that mapping lives.
        state.graphics_settings = state.to_save.graphics.to_engine();

        // Mark prefs dirty rather than writing them: `check_prefs_save` flushes within
        // PREFS_SAVE_INTERVAL. This is the backstop for changes that never finish an
        // interaction, such as nudging a slider with the arrow keys and then walking away.
        state.to_save.save_flag = true;
        // Triggers a graphics re-init
        updates.graphics_settings = Some(state.graphics_settings.clone());
    }

    if save {
        state.update_save_prefs();
    }
}

fn sensitivities(state: &mut State, scene: &mut Scene, ui: &mut Ui) {
    let mut changed = false;
    let mut save = false;

    ui.spacing_mut().slider_width = SLIDER_WIDTH;
    let cs = &mut state.to_save.control_settings;

    settings_grid("sensitivities").show(ui, |ui| {
        let resp = slider_row(
            ui,
            "Camera move speed:",
            "How fast the camera moves in response to the keyboard.",
            Slider::new(&mut cs.movement_speed, MOVEMENT_SPEED_RANGE),
        );
        if resp.changed() {
            scene.input_settings.move_sens = cs.movement_speed as f32;
            changed = true;
        }
        save |= interaction_finished(&resp);

        let resp = slider_row(
            ui,
            "Camera rotation sensitivity:",
            "How fast the camera rotates in response to the mouse and keyboard.",
            Slider::new(&mut cs.rotation_sens, ROTATION_SENS_RANGE),
        );
        if resp.changed() {
            scene.input_settings.rotate_sens = cs.rotation_sens as f32 / 100.;
            changed = true;
        }
        save |= interaction_finished(&resp);

        let resp = slider_row(
            ui,
            "Mol scroll move speed:",
            "When using the scroll wheel to move molecules, this controls how fast they move.",
            Slider::new(&mut cs.mol_move_sens, MOL_MOVE_SENS_RANGE),
        );
        changed |= resp.changed();
        save |= interaction_finished(&resp);
    });

    if reset_button(ui, "Reset sensitivities") {
        cs.movement_speed = MOVEMENT_SENS as u8;
        scene.input_settings.move_sens = MOVEMENT_SENS;

        cs.rotation_sens = (ROTATE_SENS * 100.) as u8;
        scene.input_settings.rotate_sens = ROTATE_SENS;

        cs.mol_move_sens = (SENS_MOL_MOVE_SCROLL * 1_000.) as u8;

        save = true;
    }

    if changed {
        // A backstop for changes that never finish an interaction; see `graphics_settings`.
        state.to_save.save_flag = true;
    }

    if save {
        state.update_save_prefs();
    }
}

/// Font sizes of the text labels drawn over the 3D view. These redraw live, as the sliders move.
fn label_sizes(state: &mut State, ui: &mut Ui, redraw: &mut RedrawFlags) {
    // Small molecules, lipids, and nucleic acids.
    let mut redraw_small = false;
    let mut redraw_peptide = false;
    let mut save = false;

    ui.spacing_mut().slider_width = SLIDER_WIDTH;
    let sizes = &mut state.ui.label_sizes;

    settings_grid("label_sizes").show(ui, |ui| {
        let resp = slider_row(
            ui,
            "Small molecule:",
            "Names of small molecules, lipids, and nucleic acids.",
            size_slider(&mut sizes.mol),
        );
        redraw_small |= resp.changed();
        save |= interaction_finished(&resp);

        let resp = slider_row(
            ui,
            "Protein:",
            "Names of proteins.",
            size_slider(&mut sizes.protein),
        );
        redraw_peptide |= resp.changed();
        save |= interaction_finished(&resp);

        let resp = slider_row(
            ui,
            "Chain:",
            "Protein chain IDs.",
            size_slider(&mut sizes.chain),
        );
        redraw_peptide |= resp.changed();
        save |= interaction_finished(&resp);

        let resp = slider_row(
            ui,
            "Residue:",
            "One-letter amino acid codes, at each residue's backbone center.",
            size_slider(&mut sizes.residue),
        );
        redraw_peptide |= resp.changed();
        save |= interaction_finished(&resp);

        let resp = slider_row(
            ui,
            "Atom number:",
            "Atom serial numbers, for all molecule types and in the molecule editor. \
            Also pharmacophore feature numbers.",
            size_slider(&mut sizes.atom_sn),
        );
        redraw_small |= resp.changed();
        redraw_peptide |= resp.changed();
        save |= interaction_finished(&resp);
    });

    if reset_button(ui, "Reset label sizes") {
        *sizes = Default::default();

        redraw_small = true;
        redraw_peptide = true;
        save = true;
    }

    if redraw_small || redraw_peptide {
        if state.volatile.operating_mode == OperatingMode::MolEditor {
            // The editor only draws its own molecule, which uses the ligand redraw.
            redraw.ligand = true;
        } else {
            redraw.peptide |= redraw_peptide;
            redraw.ligand |= redraw_small;
            redraw.na |= redraw_small;
            redraw.lipid |= redraw_small;
        }

        state.to_save.save_flag = true;
    }

    if save {
        state.update_save_prefs();
    }
}
