use std::path::{Path, PathBuf};

use bio_files::xtb::XtbOptimizer;
use dynamics::{Integrator, TAU_TEMP_DEFAULT};
use egui::{CollapsingHeader, ComboBox, TextEdit, Ui};

use crate::{state::State, ui::num_field};

pub(super) fn setup(state: &mut State, ui: &mut Ui) {
    CollapsingHeader::new("XTB settings").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.hyperlink_to(
                "xTB MD reference",
                "https://xtb-docs.readthedocs.io/en/latest/md.html",
            );
            ui.hyperlink_to(
                "GFN-FF",
                "https://xtb-docs.readthedocs.io/en/latest/gfnff.html",
            );
            ui.hyperlink_to(
                "Detailed input",
                "https://xtb-docs.readthedocs.io/en/latest/xcontrol.html",
            );
        });

        CollapsingHeader::new("Run control")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    num_field(&mut state.to_save.md.num_steps, "Steps:", 70, ui);
                    super::md::time_step(state, ui);
                    super::md::show_runtime(
                        ui,
                        f64::from(state.to_save.md.num_steps),
                        f64::from(state.to_save.md.dt),
                    );
                    temperature(state, ui);
                });
                ui.horizontal_wrapped(|ui| {
                    let options = &mut state.volatile.md_local.xtb_options;
                    ui.label("Hydrogen mass (u)")
                        .on_hover_text("hmass. Use 0 for natural masses.");
                    ui.add(
                        egui::DragValue::new(&mut options.hydrogen_mass).range(0..=i32::MAX as u32),
                    );
                    shake_mode(ui, &mut options.shake);
                    ui.add_enabled(
                        options.shake != 0,
                        egui::Checkbox::new(&mut options.shake_print, "Write SHAKE forces"),
                    )
                    .on_hover_text("shakeprint. Requires xTB 6.8 or newer.");
                });
            });

        CollapsingHeader::new("Output and restart").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                super::md::xtb_frame_interval(state, ui);
                let options = &mut state.volatile.md_local.xtb_options;
                num_field(
                    &mut options.coordinate_dump_fs,
                    "scoord interval (fs):",
                    70,
                    ui,
                );
                ui.checkbox(&mut options.velocities, "Write velocities to xtb.trj");
                ui.checkbox(&mut options.checkpoint_each_frame, "Checkpoint each frame")
                    .on_hover_text("Update mdrestart at each trajectory frame.");
            });
            path_field(
                ui,
                "Restart file:",
                &mut state.volatile.md_local.xtb_options.restart,
                "Path to an xTB mdrestart. Use the same atoms in the same order.",
            );
        });

        CollapsingHeader::new("Initial relaxation").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                super::md::relax(state, ui);
                if let Some(iters) = &mut state.to_save.md.config.max_init_relaxation_iters
                    && state.volatile.md_local.xtb_options.restart.is_none()
                {
                    num_field(iters, "Max cycles (0 = auto):", 60, ui);
                    let engine = &mut state.volatile.md_local.xtb_options.optimizer;
                    ComboBox::from_id_salt("optimizer")
                        .selected_text(optimizer_label(*engine))
                        .show_ui(ui, |ui| {
                            for value in [
                                XtbOptimizer::Ancopt,
                                XtbOptimizer::Lbfgs,
                                XtbOptimizer::Fire,
                            ] {
                                ui.selectable_value(engine, value, optimizer_label(value));
                            }
                        });
                }
            });
        });

        CollapsingHeader::new("GFN-FF model and executable").show(ui, |ui| {
            crate::force_fields::gfnff_solvent_selector(state, ui, "xtb_alpb");
            let Some(set) = state
                .force_fields
                .params
                .as_mut()
                .and_then(|p| p.gfnff_mut())
            else {
                return;
            };
            let mut charge = set.total_charge.unwrap_or(0.);
            let previous_charge = set.total_charge;
            let previous_params = set.parameter_file.clone();
            let previous_executable = set.executable.clone();
            ui.horizontal_wrapped(|ui| {
                num_field(&mut charge, "Total charge (e):", 60, ui);
                if charge != previous_charge.unwrap_or(0.) {
                    set.total_charge = Some(charge);
                }
                num_field(
                    &mut state.volatile.md_local.xtb_options.threads,
                    "Threads:",
                    48,
                    ui,
                );
            });
            path_field(
                ui,
                "Parameter file:",
                &mut set.parameter_file,
                "Blank uses the built-in GFN-FF parameters.",
            );
            path_field(
                ui,
                "Executable:",
                &mut set.executable,
                "Blank uses the installed xTB or XTB_EXECUTABLE/PATH.",
            );
            if set.total_charge != previous_charge
                || set.parameter_file != previous_params
                || set.executable != previous_executable
            {
                state.force_fields.invalidate();
                state.ui.ff_params.gfnff_report = None;
                state.ui.ff_params.gfnff_error = None;
                state.ui.ff_params.gfnff_job = None;
            }
        });

        let options = &mut state.volatile.md_local.xtb_options;
        CollapsingHeader::new("Restraints and topology").show(ui, |ui| {
            ui.label("Restraints ($constrain)");
            ui.add(
                TextEdit::multiline(&mut options.restraints)
                    .code_editor()
                    .desired_rows(3)
                    .hint_text("e.g. distance: 1, 2, auto"),
            );
            ui.label("Neighbor list overrides ($ffnb)");
            ui.add(
                TextEdit::multiline(&mut options.neighbors)
                    .code_editor()
                    .desired_rows(3)
                    .hint_text("e.g. nb = 1:2"),
            );
        });
    });
}

pub(super) fn temperature(state: &mut State, ui: &mut Ui) {
    let cfg = &mut state.to_save.md.config;
    let mut enabled = crate::xtb::thermostat_enabled(&cfg.integrator);
    if ui.checkbox(&mut enabled, "Berendsen (NVT)").changed() {
        cfg.integrator = Integrator::VerletVelocity {
            thermostat: enabled.then_some(TAU_TEMP_DEFAULT),
        };
    }
    num_field(&mut cfg.temp_target, "Temp (K):", 55, ui);
}

fn shake_mode(ui: &mut Ui, shake: &mut u8) {
    let labels = ["Off", "X–H bonds", "All bonds"];
    ComboBox::from_id_salt("shake")
        .selected_text(format!(
            "SHAKE · {}",
            labels.get(*shake as usize).unwrap_or(&"Unknown")
        ))
        .show_ui(ui, |ui| {
            for (value, label) in labels.into_iter().enumerate() {
                ui.selectable_value(shake, value as u8, label);
            }
        })
        .response
        .on_hover_text(
            "xTB recommends SHAKE off for GFN-FF. Some xTB builds can fail with SHAKE enabled.",
        );
}

fn optimizer_label(engine: XtbOptimizer) -> &'static str {
    match engine {
        XtbOptimizer::Ancopt => "ANCopt",
        XtbOptimizer::Lbfgs => "L-ANCopt",
        XtbOptimizer::Fire => "FIRE",
    }
}

fn path_field(ui: &mut Ui, label: &str, path: &mut Option<PathBuf>, help: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.label(label).on_hover_text(help);
        let id = ui.make_persistent_id(label);
        let mut value = ui
            .data_mut(|data| data.remove_temp::<String>(id))
            .unwrap_or_else(|| {
                path.as_deref()
                    .map(Path::to_string_lossy)
                    .unwrap_or_default()
                    .into_owned()
            });
        let response = ui.add(TextEdit::singleline(&mut value).id(id).desired_width(250.));
        if response.lost_focus() {
            let input = value.trim().trim_matches('"');
            *path = (!input.is_empty()).then(|| PathBuf::from(input));
        } else if response.has_focus() {
            ui.data_mut(|data| data.insert_temp(id, value));
        }
    });
}
