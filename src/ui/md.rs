use dynamics::{
    ComputationDevice, HydrogenConstraint, Integrator, LANGEVIN_GAMMA_DEFAULT, LINCS_ITER_DEFAULT,
    LINCS_ORDER_DEFAULT, MdConfig, SHAKE_TOL_DEFAULT, SimBoxInit, TAU_TEMP_DEFAULT,
    snapshot::{Snapshot, SnapshotHandlers},
};
use egui::ImageData::Color;
use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f32::Vec3 as Vec3F32;

use crate::prefs::ToSave;
use crate::{
    button,
    drawing::EntityClass,
    file_io::save_trajectory,
    label, md,
    md::{MdBackend, clear_snaps, launch_md, post_run_cleanup, start_md_energy_computation},
    state::State,
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        flag_btn, misc, num_field,
    },
    util::handle_err,
};

pub fn md_setup(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates, ui: &mut Ui) {
    // This sequencing code is above the UI code below, so it's deferred a frame after any actions.
    if state.volatile.md_local.launching {
        state.volatile.md_local.launching = false;
        launch_md(state, true, false);
    } else if state.volatile.md_local.running {
        // This is spammed each frame, so don't print, which handle_success does.
        state.ui.cmd_line_output = "MD Running...".to_string();
        state.ui.cmd_line_out_is_err = false;
    }

    misc::section_box().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("MD setup:").color(Color32::WHITE));

            match &state.dev {
                ComputationDevice::Cpu => {
                    ui.label(RichText::new("CPU"));
                }
                #[cfg(feature = "cuda")]
                ComputationDevice::Gpu(_) => {
                    ui.label(RichText::new("GPU").color(Color32::LIGHT_GREEN));
                }
            }

            ui.add_space(COL_SPACING / 2.0);

            if let Some(mol) = &mut state.peptide {
                // flag_btn(&mut mol.common.selected_for_md, &mol.common.ident, "Toggle if we use this molecule for MD.", ui);

                let num_ligs = state.ligands.iter().filter(|l| l.common.selected_for_md).count();
                if mol.common.selected_for_md && num_ligs > 0 {
                    flag_btn(&mut state.ui.md.peptide_only_near_ligs, "Pep only near lig", "Only model the subset of peptide atoms near a small molecule", ui);
                    flag_btn(&mut state.ui.md.peptide_static, "Pep static", "Let peptide (protein) atoms affect other molecules, but they don't move themselves", ui);
                }
            }

            if !state.lipids.is_empty() {
                let prev_val = state.lipids[0].common.selected_for_md;
                let color = if prev_val { COLOR_ACTIVE } else { COLOR_INACTIVE };

                if button!(ui, "All lipids", color, "Select all lipids for MD")
                    .clicked()
                {
                    for l in &mut state.lipids {
                        l.common.selected_for_md = !prev_val;
                    }
                }
            }

            // for mol in &mut state.ligands {
            //     flag_btn(&mut mol.common.selected_for_md, &mol.common.ident, "Toggle if we use this molecule for MD.", ui);
            // }

            for mol in &mut state.nucleic_acids {
                flag_btn(&mut mol.common.selected_for_md, &mol.common.ident, "Toggle if we use this molecule for MD.", ui);
            }

            // ui.add_space(COL_SPACING / 2.);

            if ui.button("Save MD")
                .on_hover_text("Save MD state in GROMACS format: .gro (Molecules used for MD), \
                .mdp (MD configuration), and .top (Force field parameters / topology) files")
                .clicked() {
                state.volatile.dialogs.save_md.pick_directory();
            }

            if ui.button(RichText::new("Reset").color(Color32::LIGHT_RED))
                .on_hover_text("Reset all MD settings to defaults.")
                .clicked() {

                state.to_save.md_config = Default::default();

                let to_save_default = ToSave::default();

                state.to_save.md_dt = to_save_default.md_dt;
                state.to_save.num_md_steps = to_save_default.num_md_steps;

                state.ui.md.sync(&state.to_save.md_config, state.to_save.md_dt);
            }

            if let Some(md) = &state.volatile.md_local.mol_dynamics &&
                state.ui.current_snapshot < md.snapshots.len() &&
                !md.snapshots[state.ui.current_snapshot].water_o_posits.is_empty() {

                if ui
                    .button(RichText::new("Clear water"))
                    .on_hover_text("Clear all rendered water molecules from the display")
                    .clicked() {

                    scene
                        .entities
                        .retain(|ent| ent.class != EntityClass::WaterModel as u32);
                }

                if !md.snapshots.is_empty() &&
                    ui
                        .button(RichText::new("Clear Traj"))
                        .on_hover_text("Clear all trajectory snapshots, e.g. erase the previous run.")
                        .clicked() {

                    clear_snaps(state);
                    state.ui.current_snapshot = 0;

                    scene
                        .entities
                        .retain(|ent| ent.class != EntityClass::WaterModel as u32);
                }


                // todo: Setting water class isn't working for this update.
                // engine_updates.entities = EntityUpdate::Classes(vec![EntityClass::WaterModel as u32]);
                updates.entities = EntityUpdate::All;
            }

            if button!(
                ui,
                "Run",
                COLOR_ACTION,
                "Run a molecular dynamics simulation on all molecules selected, using the selected dynamics package."
            )
                .clicked() {
                md::start_md(state, scene, updates)
            }

            if state.volatile.gromacs_avail || state.volatile.orca_avail {
                ui.add_space(COL_SPACING / 2.);
                ui.label("Backend:");

                let mut backends = vec![MdBackend::Dynamics];
                if state.volatile.gromacs_avail {
                    backends.push(MdBackend::Gromacs);
                }

                // todo: For now, we launch Orca MD from its own UI section, but we may
                // todo wish to move it here.
                // if state.volatile.orca_avail {
                //     backends.push(MdBackend::Orca);
                // }

                ComboBox::from_id_salt(523)
                    .width(80.)
                    .selected_text(state.to_save.md_backend.to_string())
                    .show_ui(ui, |ui| {
                        for v in backends {
                            ui.selectable_value(&mut state.to_save.md_backend, v.clone(), v.to_string());
                        }
                    })
                    .response
                    .on_hover_text("Select the backend to perform MD: Dynamics (Native), GROMACS, or ORCA. If GROMACS or ORCA is \
                showing, it means we've found their program on the system path.");
            }


            if button!(
                ui,
                "Compute E",
                COLOR_ACTION,
                "Compute and display instantaneous energy of selected molecules."
            )
                .clicked() {
                start_md_energy_computation(state);
            }

            if state.volatile.md_local.running {
                if ui
                    .button(RichText::new("Abort").color(Color32::LIGHT_RED))
                    .on_hover_text("Stop the in-progress simulation")
                    .clicked()
                {
                    post_run_cleanup(state, scene, updates);
                }

                if let Some(md) = &state.volatile.md_local.mol_dynamics {
                    let count = (md.step_count / 100) * 100;

                    ui.label(
                        RichText::new(format!(
                            "MD running. Step {} of {}",
                            count, state.to_save.num_md_steps
                        ))
                            .color(COLOR_HIGHLIGHT),
                    );
                }
            }

            // if let Some(md) = &state.volatile.md_local.mol_dynamics && !md.snapshots.is_empty() && button!(
            //         ui,
            //         "Save traj",
            //         COLOR_ACTION,
            //         "Save the computed MD trajectory to a DCD or TRR file."
            //     )
            //     .clicked() && save_trajectory(&mut state.volatile.dialogs.save).is_err() {
            //     handle_err(&mut state.ui, "Problem saving this file".to_owned());
            //
            // }

            let num_steps_prev = state.to_save.num_md_steps;
            num_field(&mut state.to_save.num_md_steps, "Steps:", 50, ui);

            // todo: Copies must be moved to be a per-molecule basis.
            num_field(&mut state.to_save.num_md_copies, "Copies:", 32, ui);

            ui.add_space(COL_SPACING / 2.);

            output_control(state, ui);

            if state.to_save.num_md_steps != num_steps_prev {
                state.volatile.md_local.run_time = state.to_save.num_md_steps as f32 * state.to_save.md_dt;
            }

            ui.add_space(COL_SPACING / 2.);

            // todo: Add snapshot cfg
            // num_field(&mut state.to_save.md_config.snapshot_ratio_memory, "Snapshot ratio:", 22, ui);

            // int_field_usize(&mut state.to_save.md_config.snapshot_ratio_file, "Snapshot ratio:", ui);


            ui.add_space(COL_SPACING / 2.);
            ui.label(format!("Runtime: {:.1} ps", state.volatile.md_local.run_time));
        });

        ui.add_space(ROW_SPACING / 2.);

        let help = "Cutoff distances for non-bonded forces, in Å. For Coulomb, transitions between short-range, and \
        long-range reciprical forces at this distance. For LJ, may represent a hard-cutoff. See the appropriate docs for \
        the backend used, details vary.";
        ui.horizontal_wrapped(|ui| {
            integrator_cfg(state, ui);

            ui.add_space(COL_SPACING / 2.);
            ui.label("Cutoffs (Å).").on_hover_text(help);
            num_field(&mut state.to_save.md_config.coulomb_cutoff, "Coulomb:", 20, ui);
            num_field(&mut state.to_save.md_config.lj_cutoff, "LJ:", 20, ui);
        });

        ui.add_space(ROW_SPACING / 2.);
        ui.horizontal_wrapped(|ui| {
            temp_pressure(state, ui,);
        });

        if let Some(md) = &state.volatile.md_local.mol_dynamics && state.ui.current_snapshot < md.snapshots.len() {
            ui.add_space(ROW_SPACING / 2.);

            ui.horizontal_wrapped(|ui| {
                energy_disp(&md.snapshots[state.ui.current_snapshot], ui);
            });
        }
    });

    misc::dynamics_player(state, scene, updates, ui);
}

/// Section for integrator config, various modelling parameters etc
fn integrator_cfg(state: &mut State, ui: &mut Ui) {
    ui.label(RichText::new("Integrator:").color(Color32::WHITE));
    ui.add_space(COL_SPACING / 2.);

    {
        let help_text =
            "Set the integrator to use for molecular dynamics. Verlet Velocity is a good default.";
        ui.label("Integrator:").on_hover_text(help_text);

        let prev = state.to_save.md_config.integrator.clone();
        ComboBox::from_id_salt(4)
            .width(80.)
            .selected_text(state.to_save.md_config.integrator.to_string())
            .show_ui(ui, |ui| {
                for v in &[
                    Integrator::Leapfrog {
                        thermostat: Some(TAU_TEMP_DEFAULT),
                    },
                    Integrator::LangevinMiddle {
                        gamma: LANGEVIN_GAMMA_DEFAULT,
                    },
                    Integrator::VerletVelocity {
                        thermostat: Some(TAU_TEMP_DEFAULT),
                    },
                ] {
                    ui.selectable_value(
                        &mut state.to_save.md_config.integrator,
                        v.clone(),
                        v.to_string(),
                    );
                }
            })
            .response
            .on_hover_text(help_text);

        if state.to_save.md_config.integrator != prev {
            if let Integrator::LangevinMiddle { gamma } = state.to_save.md_config.integrator
                && !matches!(prev, Integrator::LangevinMiddle { gamma: _ })
            {
                state.ui.md.langevin_γ = gamma.to_string();
            }

            if let Integrator::VerletVelocity { thermostat } = state.to_save.md_config.integrator
                && !matches!(prev, Integrator::VerletVelocity { thermostat: _ })
            {
                if let Some(tau) = thermostat {
                    state.ui.md.temp_tau = tau.to_string();
                }
            }
        }
    }

    ui.add_space(COL_SPACING / 2.);

    let help = "The simulation time step, in picoseconds. 0.001 to 0.002 are good defaults. Higher than 0.002 is risky.";
    ui.label("dt (ps):").on_hover_text(help);
    if ui
        .add_sized(
            [46., Ui::available_height(ui)],
            TextEdit::singleline(&mut state.ui.md.dt_input),
        )
        .on_hover_text(help)
        .changed()
        && let Ok(v) = state.ui.md.dt_input.parse::<f32>()
    {
        state.to_save.md_dt = v;
        state.volatile.md_local.run_time = state.to_save.num_md_steps as f32 * v;
    }

    ui.add_space(COL_SPACING / 2.);

    {
        let help_text =
            "Set to Constrained to allow higher time steps; Flexible may more more accurate.";
        ui.label("H:").on_hover_text(help_text);
        ComboBox::from_id_salt(5)
            .width(80.)
            .selected_text(state.to_save.md_config.hydrogen_constraint.to_string())
            .show_ui(ui, |ui| {
                // todo: Don't hard-code shake tol
                for v in &[
                    HydrogenConstraint::Linear {
                        order: LINCS_ORDER_DEFAULT,
                        iter: LINCS_ITER_DEFAULT,
                    },
                    HydrogenConstraint::Shake {
                        shake_tolerance: SHAKE_TOL_DEFAULT,
                    },
                    HydrogenConstraint::Flexible,
                ] {
                    ui.selectable_value(
                        &mut state.to_save.md_config.hydrogen_constraint,
                        *v,
                        v.to_string(),
                    );
                }
            })
            .response
            .on_hover_text(help_text);
    }

    ui.add_space(COL_SPACING / 2.);

    {
        let mut relax = state.to_save.md_config.max_init_relaxation_iters.is_some();
        let relax_prev = relax;
        ui.label("Relax:");
        ui.checkbox(&mut relax, "").on_hover_text(
            "Perform an initial relaxation of atom positions prior to starting MD. \
            This minimizes energy, and can take some time.",
        );
        if relax != relax_prev {
            if relax {
                state.to_save.md_config.max_init_relaxation_iters =
                    MdConfig::default().max_init_relaxation_iters;
            } else {
                state.to_save.md_config.max_init_relaxation_iters = None;
            }
        }
    }
}

/// Section for temp/pressure/ambient data
fn temp_pressure(state: &mut State, ui: &mut Ui) {
    // We show this even if there is no thermostat, to set the initial temperature.
    ui.label(RichText::new("Ambient:").color(Color32::WHITE));
    ui.add_space(COL_SPACING / 2.);

    ui.label("Temp (K):");
    if ui
        .add_sized(
            [30., Ui::available_height(ui)],
            TextEdit::singleline(&mut state.ui.md.temp_input),
        )
        .changed()
        && let Ok(v) = &mut state.ui.md.temp_input.parse::<f32>()
    {
        state.to_save.md_config.temp_target = *v;
    }

    match &mut state.to_save.md_config.integrator {
        Integrator::Leapfrog { thermostat } | Integrator::VerletVelocity { thermostat } => {
            let help_text = "Enable or disable the thermostat";
            ui.label("Therm:").on_hover_text(help_text);
            let mut therm_en = thermostat.is_some();
            if ui
                .checkbox(&mut therm_en, "")
                .on_hover_text(help_text)
                .changed()
            {
                *thermostat = if therm_en {
                    Some(TAU_TEMP_DEFAULT)
                } else {
                    None
                };
            }

            if let Some(tau) = thermostat {
                let help_text = "Thermostat time constant for use with a non-Langevin integrator. \
                0.1ps is a good default.";

                ui.label("Therm tau (ps):").on_hover_text(help_text);
                if ui
                    .add_sized(
                        [22., Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.temp_tau),
                    )
                    .on_hover_text(help_text)
                    .changed()
                    && let Ok(v) = &mut state.ui.md.temp_tau.parse::<f64>()
                {
                    *tau = *v;
                }
            }
        }
        Integrator::LangevinMiddle { gamma } => {
            let help_text = "Thermostat time constant for use with the Langevin (stochastic) integrator. \
            0.5 1/ps is a good default.";

            ui.label("Therm γ (1/ps):").on_hover_text(help_text);
            if ui
                .add_sized(
                    [22., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.md.langevin_γ),
                )
                .on_hover_text(help_text)
                .changed()
                && let Ok(v) = &mut state.ui.md.langevin_γ.parse::<f32>()
            {
                *gamma = *v;
            }
        }
    }

    ui.add_space(COL_SPACING / 2.);

    // todo: A/R
    // ui.checkbox(&mut state.to_save.md_config.zero_com_drift, "Zero drift")
    //     .on_hover_text("Zero the center-of-mass of items in the simulation.");

    let help = "The target pressure, in bar, for the barostat to maintain. The sim box changes size in \
    order to meet this.";
    ui.label("Pres (bar):").on_hover_text(help);
    if ui
        .add_sized(
            [30., Ui::available_height(ui)],
            TextEdit::singleline(&mut state.ui.md.pressure_input),
        )
        .on_hover_text(help)
        .changed()
        && let Ok(v) = &mut state.ui.md.pressure_input.parse::<f32>()
    {
        state.to_save.md_config.pressure_target = *v;
    }

    // Sim box controls.
    sim_box(state, ui)
}

fn sim_box(state: &mut State, ui: &mut Ui) {
    {
        let help = "Set the sim box size. Affects wrapping, SPME logic, and \
        solvent initialization and behavior.";
        ui.label("Box (Å):").on_hover_text(help);

        // Select Fixed or Pad
        let txt = if matches!(state.to_save.md_config.sim_box, SimBoxInit::Fixed(_)) {
            "Fixed"
        } else {
            "Pad"
        };

        let prev = state.to_save.md_config.sim_box.clone();
        ComboBox::from_id_salt(10)
            .width(60.)
            .selected_text(txt)
            .show_ui(ui, |ui| {
                for v in &[
                    (SimBoxInit::Pad(12.), "Pad"),
                    (
                        SimBoxInit::Fixed((Vec3F32::new_zero(), Vec3F32::new_zero())),
                        "Fixed",
                    ),
                ] {
                    ui.selectable_value(&mut state.to_save.md_config.sim_box, v.0.clone(), v.1);
                }
            });

        if state.to_save.md_config.sim_box != prev {
            state
                .ui
                .md
                .sync(&state.to_save.md_config, state.to_save.md_dt);
        }

        ui.add_space(COL_SPACING / 2.);

        match &mut state.to_save.md_config.sim_box {
            SimBoxInit::Pad(pad_v) => {
                let hover_text = "Set the minimum distance to pad the molecule in solvent atoms. Large values \
            can be more realistic, but significantly increase computation time. This also sets the periodic boundary \
            condition wrapping, and extent of the simulated area.";

                ui.label("Pad (Å):").on_hover_text(hover_text);
                if ui
                    .add_sized(
                        [22., Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_pad_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = &mut state.ui.md.simbox_pad_input.parse::<f32>()
                {
                    *pad_v = *v;
                }
            }
            SimBoxInit::Fixed((start, end)) => {
                let hover_text = "Set the fixed simulation box bounds in Å. Min values define one corner of the box, \
             and max values define the opposite corner.";

                let coord_width = 32.;

                if ui
                    .add_sized(
                        [22., Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_x_min_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = state.ui.md.simbox_x_min_input.parse::<f32>()
                {
                    start.x = v;
                }

                if ui
                    .add_sized(
                        [coord_width, Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_y_min_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = state.ui.md.simbox_y_min_input.parse::<f32>()
                {
                    start.y = v;
                }

                if ui
                    .add_sized(
                        [coord_width, Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_z_min_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = state.ui.md.simbox_z_min_input.parse::<f32>()
                {
                    start.z = v;
                }

                ui.label("-");

                if ui
                    .add_sized(
                        [coord_width, Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_x_max_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = state.ui.md.simbox_x_max_input.parse::<f32>()
                {
                    end.x = v;
                }

                if ui
                    .add_sized(
                        [coord_width, Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_y_max_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = state.ui.md.simbox_y_max_input.parse::<f32>()
                {
                    end.y = v;
                }

                if ui
                    .add_sized(
                        [coord_width, Ui::available_height(ui)],
                        TextEdit::singleline(&mut state.ui.md.simbox_z_max_input),
                    )
                    .on_hover_text(hover_text)
                    .changed()
                    && let Ok(v) = state.ui.md.simbox_z_max_input.parse::<f32>()
                {
                    end.z = v;
                }
            }
        }
    }
}

pub(in crate::ui) fn energy_disp(snap: &Snapshot, ui: &mut Ui) {
    let Some(en) = &snap.energy_data else { return };

    ui.label(RichText::new("Energy (current snap):").color(Color32::WHITE));
    ui.add_space(COL_SPACING / 2.);

    ui.label("E (kcal/mol) KE: ");
    ui.label(RichText::new(format!("{:.1}", en.energy_kinetic)).color(Color32::GOLD));

    ui.label("KE/atom: ");
    // todo: Don't continuously run these computations!
    let atom_count = (snap.water_o_posits.len() * 3) as f32 + snap.atom_posits.len() as f32;
    let ke_per_atom = en.energy_kinetic / atom_count;
    label!(ui, format!("{:.2}", ke_per_atom), Color32::GOLD);

    ui.label("PE: ");
    label!(ui, format!("{:.2}", en.energy_potential), Color32::GOLD);

    ui.label("PE NB: ");
    label!(
        ui,
        format!("{:.2}", en.energy_potential_nonbonded),
        Color32::GOLD
    );

    ui.label("PE/atom: ");
    // todo: Don't continuously run this!
    let pe_per_atom = en.energy_potential / atom_count;
    label!(ui, format!("{:.3}", pe_per_atom), Color32::GOLD);

    ui.label("E tot: ");
    // todo: Don't continuously run this!
    let e = en.energy_potential + en.energy_kinetic;
    label!(ui, format!("{:.3}", e), Color32::GOLD);

    ui.label("PE between mols:");
    // todo: One pair only for now
    if en.energy_potential_between_mols.len() >= 2 {
        // todo: Which index?
        ui.label(
            RichText::new(format!("{:.2}", en.energy_potential_between_mols[1]))
                .color(Color32::GOLD),
        );
    }

    ui.label("Temp: ");
    label!(ui, format!("{:.1} K", en.temperature), Color32::GOLD);

    ui.label("P: ");
    label!(ui, format!("{:.1} bar", en.pressure), Color32::GOLD);
}

fn num_field_option<T>(val: &mut Option<T>, label: &str, width: u16, ui: &mut Ui)
where
    T: std::fmt::Display + std::str::FromStr + Default + PartialEq,
{
    if !label.is_empty() {
        ui.label(label);
    }
    let mut s = val
        .as_ref()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "0".to_string());
    if ui
        .add_sized(
            [width as f32, Ui::available_height(ui)],
            TextEdit::singleline(&mut s),
        )
        .changed()
        && let Ok(v) = s.parse::<T>()
    {
        *val = if v == T::default() { None } else { Some(v) };
    }
}

fn output_control(state: &mut State, ui: &mut Ui) {
    ui.label("Output.").on_hover_text("These settings control which formats, and how often \
    to save the output trajectory to. 'Mem' only affects the Dyanmics backend. TRR positions are required to display \
    GROMACS trajectories in the UI automatically");

    let mut sync_ui = false;
    const W: u16 = 44;
    let sh_def = SnapshotHandlers::default();
    // Use memory's default ratio as the shared fallback for all handler types.
    let default_file = sh_def.memory.map(|v| v as u32);

    // Memory
    {
        let help = "Save snapshots in memory.";
        ui.label("Mem:").on_hover_text(help);
        if ui
            .checkbox(&mut state.ui.md.mem_enabled, "")
            .on_hover_text(help)
            .changed()
        {
            state.to_save.md_config.snapshot_handlers.memory = if state.ui.md.mem_enabled {
                sh_def.memory
            } else {
                None
            };
            sync_ui = true;
        }
        if state.to_save.md_config.snapshot_handlers.memory.is_some() {
            num_field_option(
                &mut state.to_save.md_config.snapshot_handlers.memory,
                "",
                W,
                ui,
            );
        }
    }

    // TRR (coords / velocities / forces)
    {
        let help = "Save trajectory to a TRR file (coords, velocities, forces).";
        ui.label("TRR:").on_hover_text(help);
        if ui
            .checkbox(&mut state.ui.md.trr_enabled, "")
            .on_hover_text(help)
            .changed()
        {
            let g = &mut state.to_save.md_config.snapshot_handlers.gromacs;
            if state.ui.md.trr_enabled {
                g.nstxout = default_file;
                g.nstvout = default_file;
                g.nstfout = default_file;
            } else {
                g.nstxout = None;
                g.nstvout = None;
                g.nstfout = None;
            }
            sync_ui = true;
        }

        if state.ui.md.trr_enabled {
            let g = &mut state.to_save.md_config.snapshot_handlers.gromacs;
            num_field_option(&mut g.nstxout, "Pos:", W, ui);
            num_field_option(&mut g.nstvout, "V:", W, ui);
            num_field_option(&mut g.nstfout, "F:", W, ui);
        }

        let g = &mut state.to_save.md_config.snapshot_handlers.gromacs;

        let en_prev = g.nstenergy;
        num_field_option(&mut g.nstenergy, "En:", W, ui);
        if g.nstenergy != en_prev {
            g.nstcalcenergy = g.nstenergy;
        }
    }

    // XTC
    {
        let help = "Save compressed coordinates to an XTC file.";
        ui.label("XTC:").on_hover_text(help);
        if ui
            .checkbox(&mut state.ui.md.xtc_enabled, "")
            .on_hover_text(help)
            .changed()
        {
            state
                .to_save
                .md_config
                .snapshot_handlers
                .gromacs
                .nstxout_compressed = if state.ui.md.xtc_enabled {
                default_file
            } else {
                None
            };
            sync_ui = true;
        }
        if state
            .to_save
            .md_config
            .snapshot_handlers
            .gromacs
            .nstxout_compressed
            .is_some()
        {
            num_field_option(
                &mut state
                    .to_save
                    .md_config
                    .snapshot_handlers
                    .gromacs
                    .nstxout_compressed,
                "",
                W,
                ui,
            );
        }
    }

    // DCD
    {
        let help = "Save trajectory to a DCD file.";
        ui.label("DCD:").on_hover_text(help);
        if ui
            .checkbox(&mut state.ui.md.dcd_enabled, "")
            .on_hover_text(help)
            .changed()
        {
            state.to_save.md_config.snapshot_handlers.dcd = if state.ui.md.dcd_enabled {
                sh_def.memory
            } else {
                None
            };
            sync_ui = true;
        }

        if state.to_save.md_config.snapshot_handlers.dcd.is_some() {
            num_field_option(
                &mut state.to_save.md_config.snapshot_handlers.dcd,
                "",
                W,
                ui,
            );
        }
    }

    if sync_ui {
        state
            .ui
            .md
            .sync(&state.to_save.md_config, state.to_save.md_dt);
    }
}
