use dynamics::{
    ComputationDevice, HydrogenConstraint, Integrator, LANGEVIN_GAMMA_DEFAULT, LINCS_ITER_DEFAULT,
    LINCS_ORDER_DEFAULT, MdConfig, SHAKE_TOL_DEFAULT, SimBoxInit, TAU_TEMP_DEFAULT,
    snapshot::Snapshot,
};
use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f32::Vec3 as Vec3F32;
use lin_alg::f64::Vec3;

use crate::md::clear_snaps;
use crate::{
    button,
    cam::move_cam_to_active_mol,
    drawing::EntityClass,
    file_io::save_trajectory,
    gromacs, label,
    md::{MdBackend, launch_md, launch_md_energy_computation, post_run_cleanup},
    state::State,
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        flag_btn, misc, num_field,
    },
    util::{clear_cli_out, handle_err, handle_success},
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
            ui.label("MD:");

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

            // todo loc
            if ui.button("Save MD")
                .on_hover_text("Save MD state in GROMACS format: .gro (Molecules used for MD), \
                .mdp (MD configuration), and .top (Force field parameters / topology) files")
                .clicked() {
                state.volatile.dialogs.save_md.pick_directory();
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
                start_md(state, scene, updates)
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

            {
                let help_text = "Set the integrator to use for molecular dynamics. Verlet Velocity is a good default.";
                ui.label("Integrator:").on_hover_text(help_text);

                let prev = state.to_save.md_config.integrator.clone();
                ComboBox::from_id_salt(4)
                    .width(80.)
                    .selected_text(state.to_save.md_config.integrator.to_string())
                    .show_ui(ui, |ui| {
                        for v in &[
                            Integrator::Leapfrog { thermostat: Some(TAU_TEMP_DEFAULT) },
                            Integrator::LangevinMiddle { gamma: LANGEVIN_GAMMA_DEFAULT },
                            Integrator::VerletVelocity { thermostat: Some(TAU_TEMP_DEFAULT) },
                        ] {
                            ui.selectable_value(&mut state.to_save.md_config.integrator, v.clone(), v.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text);

                if state.to_save.md_config.integrator != prev {
                    if let Integrator::LangevinMiddle { gamma } = state.to_save.md_config.integrator &&
                        !matches!(prev, Integrator::LangevinMiddle {gamma: _}) {
                        state.ui.md.langevin_γ = gamma.to_string();
                    }

                    if let Integrator::VerletVelocity { thermostat } = state.to_save.md_config.integrator &&
                        !matches!(prev, Integrator::VerletVelocity {thermostat:  _}) {
                        if let Some(tau) = thermostat {
                            state.ui.md.temp_tau = tau.to_string();
                        }
                    }
                }
            }

            // todo: WIP
            if button!(
                ui,
                "Compute E",
                COLOR_ACTION,
                "Compute and display instantaneous energy of selected molecules."
            )
                .clicked() {

                // todo: DRY with teh run_md button above. C+P
                clear_cli_out(&mut state.ui); // todo: Not working; not loaded until next frame.
                let mut ready_to_run = true;

                // Check that we have FF params and mol-specific parameters.
                for lig in &state.ligands {
                    if !lig.common.selected_for_md {
                        continue;
                    }

                    if !lig.ff_params_loaded || !lig.frcmod_loaded {
                        state.ui.popup.show_get_geostd = true;
                        ready_to_run = false;
                    }
                }

                if ready_to_run {
                    let mut pep_md = state.volatile.md_local.peptide_selected.clone(); // Avoids borrow problem.
                    match launch_md_energy_computation(state, &mut pep_md) {
                        Ok(en) => {
                            let data = format!("E result. PE: {:.2}, PE NB: {:.3} PE Bonded: {:.2}",
                                               en.energy_potential, en.energy_potential_nonbonded, en.energy_potential_bonded);
                            println!("{data}");
                            handle_success(&mut state.ui, data);

                            state.volatile.md_local.peptide_selected = pep_md;

                        }
                        Err(e) => handle_err(&mut state.ui, format!("Error computing energy: {:?}", e))
                    }

                }
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

            if let Some(md) = &state.volatile.md_local.mol_dynamics && !md.snapshots.is_empty() && button!(
                    ui,
                    "Save traj",
                    COLOR_ACTION,
                    "Save the computed MD trajectory to a DCD or TRR file."
                )
                .clicked() && save_trajectory(&mut state.volatile.dialogs.save).is_err() {
                handle_err(&mut state.ui, "Problem saving this file".to_owned());

            }

            let num_steps_prev = state.to_save.num_md_steps;
            num_field(&mut state.to_save.num_md_steps, "Steps:", 50, ui);

            num_field(&mut state.to_save.num_md_copies, "Copies:", 32, ui);

            if state.to_save.num_md_steps != num_steps_prev {
                state.volatile.md_local.run_time = state.to_save.num_md_steps as f32 * state.to_save.md_dt;
            }

            ui.label("dt (ps):");
            if ui
                .add_sized(
                    [46., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.md.dt_input),
                )
                .changed()
                && let Ok(v) = state.ui.md.dt_input.parse::<f32>() {
                state.to_save.md_dt = v;
                state.volatile.md_local.run_time = state.to_save.num_md_steps as f32 * v;
            }

            ui.add_space(COL_SPACING / 2.);

            // todo: Dropdown to select shake vs linear vs no.
            {
                let help_text = "Set to Constrained to allow higher time steps; Flexible may more more accurate.";
                ui.label("H:").on_hover_text(help_text);
                ComboBox::from_id_salt(5)
                    .width(80.)
                    .selected_text(state.to_save.md_config.hydrogen_constraint.to_string())
                    .show_ui(ui, |ui| {
                        // todo: Don't hard-code shake tol
                        for v in &[
                            HydrogenConstraint::Linear { order: LINCS_ORDER_DEFAULT, iter: LINCS_ITER_DEFAULT },
                            HydrogenConstraint::Shake { shake_tolerance: SHAKE_TOL_DEFAULT},
                            HydrogenConstraint::Flexible
                        ] {
                            ui.selectable_value(&mut state.to_save.md_config.hydrogen_constraint, *v, v.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }
            ui.add_space(COL_SPACING / 2.);

            // todo: Add snapshot cfg
            // num_field(&mut state.to_save.md_config.snapshot_ratio_memory, "Snapshot ratio:", 22, ui);

            // int_field_usize(&mut state.to_save.md_config.snapshot_ratio_file, "Snapshot ratio:", ui);

            let mut relax = state.to_save.md_config.max_init_relaxation_iters.is_some();
            let relax_prev = relax;
            ui.label("Relax:");
            ui.checkbox(&mut relax, "").on_hover_text("Perform an initial relaxation of atom positions prior to starting MD. \
            This minimizes energy, and can take some time.");
            if relax != relax_prev {
                if relax {
                    state.to_save.md_config.max_init_relaxation_iters = MdConfig::default().max_init_relaxation_iters;
                } else {
                    state.to_save.md_config.max_init_relaxation_iters = None;
                }
            }

            ui.add_space(COL_SPACING / 2.);
            ui.label(format!("Runtime: {:.1} ps", state.volatile.md_local.run_time));

            if let Some(md) = &state.volatile.md_local.mol_dynamics && state.ui.current_snapshot < md.snapshots.len() {
                energy_disp(&md.snapshots[state.ui.current_snapshot], ui);
            }
        });

        ui.add_space(ROW_SPACING / 2.);
        ui.horizontal_wrapped(|ui| {
            temp_pressure(state, ui,);
        })
    });

    misc::dynamics_player(state, scene, updates, ui);
}

fn temp_pressure(state: &mut State, ui: &mut Ui) {
    // We show this even if there is no thermostat, to set the initial temperature.
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

    ui.label("Pres (bar):");
    if ui
        .add_sized(
            [30., Ui::available_height(ui)],
            TextEdit::singleline(&mut state.ui.md.pressure_input),
        )
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
        ui.label("Box (Å):");

        // Select Fixed or Pad
        let txt = if matches!(state.to_save.md_config.sim_box, SimBoxInit::Fixed(_)) {
            "Fixed"
        } else {
            "Pad"
        };

        let prev = state.to_save.md_config.sim_box.clone();
        ComboBox::from_id_salt(10)
            .width(80.)
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
    ui.add_space(COL_SPACING / 2.);
    ui.label("E (kcal/mol) KE: ");
    ui.label(RichText::new(format!("{:.1}", snap.energy_kinetic)).color(Color32::GOLD));

    ui.label("KE/atom: ");
    // todo: Don't continuously run these computations!
    let atom_count = (snap.water_o_posits.len() * 3) as f32 + snap.atom_posits.len() as f32;
    let ke_per_atom = snap.energy_kinetic / atom_count;
    label!(ui, format!("{:.2}", ke_per_atom), Color32::GOLD);

    ui.label("PE: ");
    label!(ui, format!("{:.2}", snap.energy_potential), Color32::GOLD);

    ui.label("PE NB: ");
    label!(
        ui,
        format!("{:.2}", snap.energy_potential_nonbonded),
        Color32::GOLD
    );

    ui.label("PE/atom: ");
    // todo: Don't continuously run this!
    let pe_per_atom = snap.energy_potential / atom_count;
    label!(ui, format!("{:.3}", pe_per_atom), Color32::GOLD);

    ui.label("E tot: ");
    // todo: Don't continuously run this!
    let e = snap.energy_potential + snap.energy_kinetic;
    label!(ui, format!("{:.3}", e), Color32::GOLD);

    ui.label("PE between mols:");
    // todo: One pair only for now
    if snap.energy_potential_between_mols.len() >= 2 {
        // todo: Which index?
        ui.label(
            RichText::new(format!("{:.2}", snap.energy_potential_between_mols[1]))
                .color(Color32::GOLD),
        );
    }

    ui.label("Temp: ");
    label!(ui, format!("{:.1} K", snap.temperature), Color32::GOLD);

    ui.label("P: ");
    label!(ui, format!("{:.1} bar", snap.pressure), Color32::GOLD);
}

fn start_md(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    clear_cli_out(&mut state.ui);
    let mut ready_to_run = true;

    // Check that we have FF params and mol-specific parameters.
    for lig in &state.ligands {
        if !lig.common.selected_for_md {
            continue;
        }
        if !lig.ff_params_loaded || !lig.frcmod_loaded {
            state.ui.popup.show_get_geostd = true;
            ready_to_run = false;
        }
    }

    if !ready_to_run {
        return;
    }

    let center = match &state.peptide {
        Some(m) => m.center,
        None => Vec3::new(0., 0., 0.),
    };
    // todo: Set a loading indicator, and trigger the build next GUI frame.
    move_cam_to_active_mol(state, scene, center, updates);

    match state.to_save.md_backend {
        MdBackend::Dynamics => {
            handle_success(
                &mut state.ui,
                "Running MD. Initializing water, and relaxing the molecules...".to_string(),
            );

            // We will wait a frame so we can display the message above.
            state.volatile.md_local.launching = true;
        }
        MdBackend::Gromacs => {
            handle_success(&mut state.ui, "\nRunning MD with GROMACS...".to_string());
            // We will wait a frame so we can display the message above.
            gromacs::launch_md(state)
        }
        // todo: For now, we launch ORCA MD from the ORCA UI.
        MdBackend::Orca => {}
    }
}
