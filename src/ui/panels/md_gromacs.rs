use std::path::PathBuf;

use egui::{
    Align, Button, CollapsingHeader, ComboBox, FontSelection, Label, Layout, ScrollArea, TextEdit,
    TextStyle, Ui, vec2,
};

use crate::{
    gromacs::{InputFile, generated_mdp, mdp_field_value, mdp_value, set_mdp_value},
    state::State,
};

const MANUAL: &str = "https://manual.gromacs.org/2026.4/user-guide/mdp-options.html";

pub(super) fn pick(state: &mut State, file: InputFile) {
    let dialogs = &mut state.volatile.dialogs;
    let setup = &mut state.volatile.md_local.gromacs_setup;
    let current = file.path(setup).clone();
    let parent = current
        .as_deref()
        .or(setup.topology.as_deref())
        .or(setup.mdp_path.as_deref())
        .and_then(|path| path.parent());
    let config = dialogs.open_gromacs.config_mut();
    config.default_file_filter = Some(file.filter().into());
    if let Some(parent) = parent {
        config.initial_directory = parent.to_owned();
    }
    dialogs.gromacs_file = Some(file);
    if matches!(file, InputFile::Bundle) {
        dialogs.open_gromacs.pick_multiple();
    } else {
        dialogs.open_gromacs.pick_file();
    }
}

pub(super) fn setup(state: &mut State, ui: &mut Ui) {
    let running = crate::md::runs::running(state);
    ui.add_enabled_ui(!running, |ui| {
        let visible_width = ui.clip_rect().right() - ui.next_widget_position().x;
        ui.set_max_width(ui.available_width().min(visible_width.max(0.)));
        ui.spacing_mut().item_spacing = vec2(6., 2.);

        let mut pick_file = None;
        let mut save_mdp = false;
        let setup = &mut state.volatile.md_local.gromacs_setup;

        ui.horizontal_wrapped(|ui| {
            for (label, file) in [
                ("MDP", InputFile::Mdp),
                ("GRO", InputFile::Coordinates),
                ("TOP", InputFile::Topology),
                ("Index", InputFile::Index),
            ] {
                if file_button(ui, label, file.path(setup), file != InputFile::Mdp) {
                    pick_file = Some(file);
                }
            }

            if setup.uses_files() && ui.button("Use selected molecules").clicked() {
                setup.coordinates = None;
                setup.topology = None;
                setup.index = None;
                setup.reference = None;
                setup.continuation = None;
                setup.grompp_args.clear();
                setup.mdrun_args.clear();
            }
            if !setup.uses_files() && setup.mdp.is_some() && ui.button("Use panel settings").clicked() {
                setup.mdp = None;
                setup.mdp_path = None;
            }
        });

        CollapsingHeader::new("GROMACS settings")
            .default_open(setup.mdp.is_some() || setup.uses_files())
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.hyperlink_to("MDP reference", MANUAL);
                    save_mdp = ui.button("Save MDP").clicked();
                });

                if let Some(mdp) = &mut setup.mdp {
                    edit_mdp(ui, mdp);
                } else {
                    let original = if setup.uses_files() {
                        String::new()
                    } else {
                        generated_mdp(&state.to_save.md)
                    };
                    let mut draft = original.clone();
                    edit_mdp(ui, &mut draft);

                    if draft != original {
                        setup.mdp = Some(draft);
                    }
                }

                if setup.uses_files() {
                    CollapsingHeader::new("Command options").show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if file_button(ui, "Restraint GRO", &mut setup.reference, true) {
                                pick_file = Some(InputFile::Reference);
                            }
                            if file_button(ui, "Continuation", &mut setup.continuation, true) {
                                pick_file = Some(InputFile::Continuation);
                            }
                        });
                        command_args(ui, "grompp", &mut setup.grompp_args,
                            "Runs in the MDP folder, or the topology folder for an MDP created here. Enter each flag and value on its own line.");
                        command_args(ui, "mdrun", &mut setup.mdrun_args,
                            "Runs in the output folder. Use absolute paths for additional input files.");
                    });
                }
            });

        if let Some(file) = pick_file {
            pick(state, file);
        }
        if save_mdp {
            let dialog = &mut state.volatile.dialogs.save;
            let config = dialog.config_mut();
            let old_name = std::mem::replace(&mut config.default_file_name, "md.mdp".into());
            let old_extension = config.default_save_extension.replace("MDP (MD params)".into());
            dialog.save_file();
            let config = dialog.config_mut();
            config.default_file_name = old_name;
            config.default_save_extension = old_extension;
        }
    });
}

fn file_button(ui: &mut Ui, label: &str, path: &mut Option<PathBuf>, clear: bool) -> bool {
    let text = match path {
        Some(path) => format!(
            "{label} · {}",
            path.file_name().unwrap_or_default().to_string_lossy()
        ),
        None => format!("Choose {label}"),
    };
    let clicked = ui
        .add(Button::new(text).truncate())
        .on_hover_text(
            path.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
        .clicked();
    if path.is_some()
        && clear
        && ui
            .small_button("×")
            .on_hover_text(format!("Clear {label}"))
            .clicked()
    {
        *path = None;
    }

    clicked
}

fn command_args(ui: &mut Ui, name: &str, args: &mut String, help: &str) {
    ui.label(format!("{name} arguments · one argument per line"))
        .on_hover_text(help);
    ui.add(
        TextEdit::multiline(args)
            .id_salt(name)
            .code_editor()
            .desired_rows(2)
            .desired_width(340.),
    );
}

fn edit_mdp(ui: &mut Ui, mdp: &mut String) {
    ui.horizontal_wrapped(|ui| {
        option(
            ui,
            mdp,
            "integrator",
            &[
                "md",
                "md-vv",
                "md-vv-avek",
                "sd",
                "bd",
                "steep",
                "cg",
                "l-bfgs",
                "nm",
                "tpi",
                "tpic",
                "mimic",
            ],
        );
        fields(ui, mdp, &["nsteps", "dt"]);
        if let (Some(steps), Some(dt)) = (
            mdp_value(mdp, "nsteps").and_then(|v| v.parse::<i64>().ok()),
            mdp_value(mdp, "dt").and_then(|v| v.parse::<f64>().ok()),
        ) {
            if steps == -1 {
                ui.label("Unlimited steps");
            } else if steps >= 0 && dt.is_finite() && dt > 0. {
                ui.label(format!("{:.3} ps", steps as f64 * dt));
            }
        }
    });

    CollapsingHeader::new("Run control and constraints").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            option(ui, mdp, "continuation", &["no", "yes"]);
            option(ui, mdp, "gen-vel", &["no", "yes"]);
            option(
                ui,
                mdp,
                "constraints",
                &["none", "h-bonds", "all-bonds", "h-angles", "all-angles"],
            );
            option(ui, mdp, "constraint-algorithm", &["lincs", "shake"]);
            fields(
                ui,
                mdp,
                &[
                    "tinit",
                    "init-step",
                    "gen-temp",
                    "gen-seed",
                    "ld-seed",
                    "emtol",
                    "emstep",
                    "nstcgsteep",
                    "nbfgscorr",
                    "comm-mode",
                    "nstcomm",
                    "comm-grps",
                    "lincs-order",
                    "lincs-iter",
                    "lincs-warnangle",
                    "shake-tol",
                ],
            );
        });
    });

    CollapsingHeader::new("Temperature and pressure").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            option(
                ui,
                mdp,
                "tcoupl",
                &[
                    "no",
                    "berendsen",
                    "nose-hoover",
                    "v-rescale",
                    "andersen",
                    "andersen-massive",
                ],
            );
            fields(ui, mdp, &["tc-grps", "tau-t", "ref-t", "nsttcouple"]);
            option(
                ui,
                mdp,
                "pcoupl",
                &["no", "berendsen", "parrinello-rahman", "c-rescale", "mttk"],
            );
            option(
                ui,
                mdp,
                "pcoupltype",
                &[
                    "isotropic",
                    "semiisotropic",
                    "anisotropic",
                    "surface-tension",
                ],
            );
            fields(
                ui,
                mdp,
                &[
                    "tau-p",
                    "ref-p",
                    "compressibility",
                    "nstpcouple",
                    "refcoord-scaling",
                ],
            );
        });
    });

    CollapsingHeader::new("Nonbonded interactions and box").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            option(ui, mdp, "pbc", &["xyz", "xy", "no"]);
            option(
                ui,
                mdp,
                "coulombtype",
                &["cut-off", "reaction-field", "ewald", "PME", "P3M-AD"],
            );
            option(ui, mdp, "coulomb-modifier", &["none", "potential-shift"]);
            option(ui, mdp, "vdwtype", &["cut-off", "PME"]);
            option(
                ui,
                mdp,
                "vdw-modifier",
                &[
                    "none",
                    "potential-shift",
                    "force-switch",
                    "potential-switch",
                ],
            );
            option(ui, mdp, "DispCorr", &["no", "EnerPres", "Ener"]);
            fields(
                ui,
                mdp,
                &[
                    "nstlist",
                    "verlet-buffer-tolerance",
                    "rlist",
                    "rcoulomb",
                    "rvdw",
                    "rvdw-switch",
                    "epsilon-r",
                    "epsilon-rf",
                    "fourierspacing",
                    "pme-order",
                    "fourier-nx",
                    "fourier-ny",
                    "fourier-nz",
                    "ewald-rtol",
                    "ewald-rtol-lj",
                    "ewald-geometry",
                    "lj-pme-comb-rule",
                    "deform",
                ],
            );
        });
    });

    CollapsingHeader::new("Restraints, pulling, and free energy").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            option(ui, mdp, "pull", &["no", "yes"]);
            option(ui, mdp, "free-energy", &["no", "yes", "expanded"]);
            fields(
                ui,
                mdp,
                &[
                    "freezegrps",
                    "freezedim",
                    "pull-ngroups",
                    "pull-ncoords",
                    "init-lambda-state",
                    "fep-lambdas",
                    "coul-lambdas",
                    "vdw-lambdas",
                    "couple-moltype",
                    "couple-lambda0",
                    "couple-lambda1",
                    "nstdhdl",
                ],
            );
        });
    });

    CollapsingHeader::new("Output and topology preprocessing").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            fields(
                ui,
                mdp,
                &[
                    "nstlog",
                    "nstenergy",
                    "nstcalcenergy",
                    "nstxout",
                    "nstvout",
                    "nstfout",
                    "nstxout-compressed",
                    "compressed-x-precision",
                    "compressed-x-grps",
                    "energygrps",
                    "define",
                    "include",
                ],
            );
        });
    });

    CollapsingHeader::new("Full MDP").show(ui, |ui| {
        ScrollArea::vertical().max_height(220.).show(ui, |ui| {
            ui.add(
                TextEdit::multiline(mdp)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .desired_rows(8),
            );
        });
    });
}

fn option(ui: &mut Ui, mdp: &mut String, key: &str, choices: &[&str]) {
    let current = mdp_value(mdp, key).unwrap_or_default();
    let mut chosen = current;
    let display = if current.is_empty() {
        "Default"
    } else {
        current
    };
    let text = format!("{key} · {display}");
    let spacing = ui.spacing();
    let width = text_width(ui, &text, TextStyle::Button)
        + spacing.icon_width
        + spacing.icon_spacing
        + 2. * spacing.button_padding.x;

    inline_control(ui, width, |ui| {
        ComboBox::from_id_salt(key)
            .width(ui.available_width())
            .truncate()
            .selected_text(&text)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut chosen, "", "GROMACS default");
                for choice in choices {
                    ui.selectable_value(&mut chosen, choice, *choice);
                }
            })
            .response
            .on_hover_text(&text);
    });
    if chosen != current {
        let chosen = chosen.to_owned();
        set_mdp_value(mdp, key, &chosen);
    }
}

fn fields(ui: &mut Ui, mdp: &mut String, keys: &[&str]) {
    const INPUT_WIDTH: f32 = 66.;
    let gap = ui.spacing().item_spacing.x;

    for &key in keys {
        let mut value = mdp_field_value(mdp, key).unwrap_or_default().to_owned();
        let label_width = text_width(ui, key, TextStyle::Body);

        inline_control(ui, label_width + gap + INPUT_WIDTH, |ui| {
            let label_width = label_width.min((ui.available_width() - gap - INPUT_WIDTH).max(0.));
            ui.add_sized(
                vec2(label_width, ui.spacing().interact_size.y),
                Label::new(key).truncate(),
            )
            .on_hover_text(key);
            let response = ui.add(
                TextEdit::singleline(&mut value)
                    .id_salt(key)
                    .hint_text("Default")
                    .desired_width(INPUT_WIDTH),
            );
            if response.changed() {
                set_mdp_value(mdp, key, &value);
            }
        });
    }
}

fn text_width(ui: &Ui, text: &str, style: TextStyle) -> f32 {
    let font = FontSelection::Default.resolve_with_fallback(ui.style(), style.into());

    ui.painter()
        .layout_no_wrap(text.into(), font, ui.visuals().text_color())
        .size()
        .x
        .ceil()
}

fn inline_control(ui: &mut Ui, width: f32, contents: impl FnOnce(&mut Ui)) {
    // Reserve the whole control so a wrapped row moves it as one widget.
    let size = vec2(
        width.min(ui.available_width()),
        ui.spacing().interact_size.y,
    );
    ui.allocate_ui_with_layout(size, Layout::left_to_right(Align::Center), contents);
}
