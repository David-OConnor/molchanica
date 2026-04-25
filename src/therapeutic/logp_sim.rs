//! Creates a MD workflow to measure LogP or LogD by alchemically decoupling the
//! solute in water and in water-saturated octanol, then comparing the free energies.
//!
//! This uses thermodynamic integration over a set of fixed lambda windows. For now
//! we run the alchemical windows on the built-in `dynamics` backend and on CPU so
//! that the lambda-dependent interaction scaling and `dH/dlambda` bookkeeping are
//! consistent with the solvent templates in that crate.

use std::{collections::HashSet, path::Path};

use bio_files::gromacs::OutputControl;
use dynamics::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, ParamError, SimBox,
    SimBoxInit, Solvent, TAU_TEMP_DEFAULT,
    alchemical::{LambdaWindow, collect_window, free_energy_ti, log_p},
    snapshot::SnapshotHandlers,
};
use graphics::{EngineUpdates, Scene};
use lin_alg::f32::Vec3;

use crate::{
    file_io::save_mol_set_as_gro,
    md::{MdBackend, build_dynamics, custom_solvents_to_mol_commons, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
    state::State,
    util::handle_success,
};

// 46 is a good default, and is what we built the template for. Trying smaller to optimize.
// const OCTANOL_BOX_SIZE: f32 = 46.; // 356 octanol + 132 water ≈ 97,230 Å³ ≈ 46³ Å³
const OCTANOL_BOX_SIZE: f32 = 30.; // 356 octanol + 132 water ≈ 97,230 Å³ ≈ 46³ Å³
const WATER_BOX_SIZE: f32 = 35.; // Å — 35 Å → ~1,400 water mols; > 2× the 12 Å NB cutoff.

// Molecules per Å³
const OCTANOL_PER_UNIT_VOL: f32 = 356. / (46. * 46. * 46.);

// Note: We can probably depecate octanol count and water per octanol: These are now part
// of a template. But useful when *building the template*

// 27 mol% water in water-saturated 1-octanol (literature value).
// water/(water+octanol) = 0.27  →  water = octanol × 0.27/0.73 ≈ octanol × 0.37.
// const OCTANOL_COUNT: usize =
//     (OCTANOL_BOX_SIZE * OCTANOL_BOX_SIZE * OCTANOL_BOX_SIZE * OCTANOL_PER_UNIT_VOL) as usize;

// todo: or 0.26? I'm getting different values from different sources.
const WATER_MOL_PER_OCTANOL: f32 = 0.38;
// const OCTANOL_BOX_WATER_COUNT: usize = (OCTANOL_COUNT as f32 * WATER_MOL_PER_OCTANOL) as usize;

const DT: f32 = 0.002; // ps

// Starter settings so the workflow is usable from the UI. Production-quality LogP
// predictions should use much longer sampling and usually denser endpoint spacing.
const EQUIL_STEPS_PER_WINDOW: usize = 500;
const PROD_STEPS_PER_WINDOW: usize = 1_000;

const SNAPSHOT_RATIO: usize = 1; // todo: For now

const TEMP_TGT: f32 = 298.15; // 25 C
const LAMBDAS: &[f64] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

#[derive(Clone, Copy)]
enum Phase {
    Water,
    Octanol,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Water => "water",
            Self::Octanol => "octanol",
        }
    }

    fn solvent(self) -> Solvent {
        match self {
            Self::Water => Solvent::WaterOpc,
            Self::Octanol => Solvent::OctanolWithWater,
        }
    }

    fn box_size(self) -> f32 {
        match self {
            Self::Water => WATER_BOX_SIZE,
            Self::Octanol => OCTANOL_BOX_SIZE,
        }
    }
}

struct FreeEnergyEstimate {
    dg_kcal_mol: f64,
    dg_sem_kcal_mol: Option<f64>,
    windows: Vec<LambdaWindow>,
}

fn build_cfg(phase: Phase) -> MdConfig {
    MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        zero_com_drift: true,
        temp_target: TEMP_TGT,
        // Keep the cell fixed during TI so the free-energy estimator does not also
        // depend on barostat fluctuations.
        barostat_cfg: None,
        snapshot_handlers: SnapshotHandlers {
            memory: Some(SNAPSHOT_RATIO),
            // ..Default::default()
            // todo: For now at least
            gromacs: OutputControl {
                nstxout: Some(1),
                nstfout: Some(1), // todo A/R
                ..Default::default()
            },
            ..Default::default()
        },
        sim_box: SimBoxInit::new_cube(phase.box_size()),
        solvent: phase.solvent(),
        overrides: MdOverrides::default(),
        ..Default::default()
    }
}

fn integrate_ti_sem(windows: &[LambdaWindow]) -> Option<f64> {
    let mut variance = 0.0;

    for pair in windows.windows(2) {
        let delta_lambda = pair[1].lambda - pair[0].lambda;
        let prefactor = 0.5 * delta_lambda;
        let sem_0 = pair[0].sem_dh_dl?;
        let sem_1 = pair[1].sem_dh_dl?;
        variance += prefactor.powi(2) * (sem_0.powi(2) + sem_1.powi(2));
    }

    Some(variance.sqrt())
}

fn run_alchemical_window(
    mol: &MoleculeSmall,
    state: &State,
    phase: Phase,
    lambda: f64,
) -> Result<LambdaWindow, ParamError> {
    let mut mol = mol.clone();
    mol.common.selected_for_md = true;

    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];
    let cfg = build_cfg(phase);
    let mut pep_atom_set = HashSet::new();

    let dev = ComputationDevice::Cpu;
    let (mut md, _) = build_dynamics(
        &dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        &cfg,
        false,
        None,
        &mut pep_atom_set,
    )?;

    // The solute is the first user-supplied molecule. Solvent and ions are appended later.
    md.alch_mol_idx = Some(0);
    md.lambda_alch = lambda;

    run_dynamics_blocking(&mut md, &dev, DT, EQUIL_STEPS_PER_WINDOW);
    md.snapshots.clear();
    run_dynamics_blocking(&mut md, &dev, DT, PROD_STEPS_PER_WINDOW);

    if md.snapshots.is_empty() {
        return Err(ParamError::new(
            "Alchemical production run completed without recording any snapshots.",
        ));
    }

    let window = collect_window(lambda, &md.snapshots);

    println!(
        "Alchemical window complete: solvent={} lambda={lambda:.2} <dH/dlambda>={:.4} kcal/mol sem={}",
        phase.name(),
        window.mean_dh_dl,
        window
            .sem_dh_dl
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a".to_string())
    );

    Ok(window)
}

fn run_drag(
    // dev: &ComputationDevice,
    mol: &MoleculeSmall,
    // state: &State,
    state: &mut State,

    // todo :Mut state for now so we can easily visualize whne testing.
    phase: Phase,
) -> Result<f32, ParamError> {
    let mut mol = mol.clone();
    mol.common.selected_for_md = true;

    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];

    let atom_count_solute = mols[0].1.atoms.len();

    let cfg = {
        let mut v = build_cfg(phase);

        // For example: Long along the drag axis.
        // v.sim_box = SimBoxInit::Fixed((
        //     Vec3::new_zero(), Vec3::new_zero(),
        // ));

        v
    };
    let mut pep_atom_set = HashSet::new();

    let (mut md, custom_solvent) = build_dynamics(
        &state.dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        &cfg,
        false,
        None,
        &mut pep_atom_set,
    )?;

    // Includes the solute and if applicable, octanol. Does not include OPC water.
    let atom_count_all = md.atoms.len();

    // run_dynamics_blocking(&mut md, &dev, DT, EQUIL_STEPS_PER_WINDOW);

    // todo: Experimenting with our already-existing external force injection into step.
    let f_drag = Vec3::x() * 1.;

    // 1: 360 (water
    // 0.5: 34-41
    // 0.25: 220 - 230... Huh? How? or 600???
    // 2.0: 90 - 100

    let f_external = {
        // NOte Dynamics has special handling of rigid water, but treats explicit
        // solvents as normal MD atoms.
        // let num_md_atoms = match phase {
        //     Phase::Water => atom_count_solute,
        //     Phase::Octanol => md.atoms.len(),
        // }
        //
        let mut v = vec![Vec3::new_zero(); md.atoms.len()];

        for i in 0..atom_count_solute {
            v[i] = f_drag;
        }

        v
    };

    // let n_steps = 1_000;
    let n_steps = 2_000;
    // let n_steps = 500;

    println!("Running MD with drag test for {}...", phase.name());

    for step in 0..n_steps {
        // Step 0 will have no external force/no dragging.
        let force = if step == 0 {
            None
        } else {
            Some(f_external.clone())
        };
        md.step(&state.dev, DT, force);
    }
    md.flush_snapshot_queues();

    let mut baseline_f = Vec3::new_zero();
    let mut baseline_e = 0.;

    let mut f_avg = Vec3::new_zero();
    let mut f_mag_avg = 0.;
    let mut e_avg = 0.;

    for (i, snap) in md.snapshots.iter().enumerate() {
        // todo: FOrce on solvent too? Or just solute?
        match &snap.force {
            Some(f_by_atom) => {
                let mut f_total = Vec3::new_zero();
                for f in f_by_atom {
                    f_total += *f;
                }

                if i == 0 {
                    baseline_f = f_total;
                }

                let diff_from_baseline = f_total - baseline_f;
                let diff_mag = diff_from_baseline.magnitude();

                f_avg += f_total;
                f_mag_avg += f_total.magnitude();

                if i.is_multiple_of(500) {
                    println!(
                        "{} - F at snap {i}: {f_total:.5}. Diff: {diff_mag:.5}",
                        phase.name()
                    );
                }
            }
            None => {
                eprintln!("Error: No force");
            }
        }

        match &snap.energy_data {
            Some(e) => {
                // let mut f_total = Vec3::new_zero();
                // for f in e {
                //     f_total += *f;
                // }
                //
                // if i == 0 {
                //     baseline_f = f_total;
                // }
                //
                // let diff_from_baseline = f_total - baseline_f;
                // let diff_mag = diff_from_baseline.magnitude();

                let e_total = e.energy_potential + e.energy_kinetic;
                if i == 0 {
                    baseline_e = e_total;
                }

                e_avg += e_total;

                if i.is_multiple_of(500) {
                    println!(
                        "{} - E at snap {i}: {e_total:.0}. Base: {baseline_e} Diff: {:.0}",
                        phase.name(),
                        e_total - baseline_e
                    );
                }
            }
            None => {
                eprintln!("Error: No energy");
            }
        }
    }

    f_avg /= md.snapshots.len() as f32;
    f_mag_avg /= md.snapshots.len() as f32;
    e_avg /= md.snapshots.len() as f32;

    let f_avg_diff = f_avg - baseline_f;
    let f_mag_diff = f_mag_avg - baseline_f.magnitude();
    let e_avg_diff = e_avg - baseline_e;

    println!(
        "\n{} - F avg: {f_avg_diff:.3} F mag avg: {f_mag_avg:.3} E avg: {e_avg_diff:.0}\n",
        phase.name()
    );

    // Mirror the normal MD viewer setup so packed octanol copies appear before water
    // in the exported GRO, and snapshot-driven water positions replace the placeholders.
    let mut viewer_mol_data = vec![(FfMolType::SmallOrganic, mol.common.clone(), 1)];
    let custom_mol_commons =
        custom_solvents_to_mol_commons(&custom_solvent).map_err(|e| ParamError::new(&e))?;

    for mol_common in custom_mol_commons {
        viewer_mol_data.push((FfMolType::SmallOrganic, mol_common, 1));
    }

    // MdState stores counter-ions after the custom solvent molecules.
    let ion_start = md
        .mol_start_indices
        .get(mols.len() + custom_solvent.len())
        .copied()
        .unwrap_or(md.atoms.len());

    for a in &md.atoms[ion_start..] {
        let posit = lin_alg::f64::Vec3 {
            x: a.posit.x as f64,
            y: a.posit.y as f64,
            z: a.posit.z as f64,
        };
        let ion_atom = crate::molecules::Atom {
            serial_number: a.serial_number,
            element: a.element,
            posit,
            ..Default::default()
        };
        viewer_mol_data.push((
            FfMolType::SmallOrganic,
            crate::molecules::common::MoleculeCommon {
                ident: a.force_field_type.clone(),
                atoms: vec![ion_atom],
                atom_posits: vec![posit],
                selected_for_md: true,
                ..Default::default()
            },
            1,
        ));
    }

    let viewer_mol_refs = viewer_mol_data
        .iter()
        .map(|(ff, mol, count)| (*ff, mol, *count))
        .collect::<Vec<_>>();
    state
        .volatile
        .md_local
        .viewer
        .add_mol_set(&viewer_mol_refs, md.water.len());
    let new_set_i = state
        .volatile
        .md_local
        .viewer
        .mol_sets
        .len()
        .saturating_sub(1);
    state.volatile.md_local.viewer.mol_set_active = Some(new_set_i);
    state.volatile.md_local.replace_snaps(md.snapshots.clone());
    state.volatile.md_local.mol_dynamics = Some(md);

    // todo temp; for visualizing this.
    let gro_path = Path::new("./md_out").join(format!("logp_{}.gro", phase.name()));

    if let Some(mol_set) = state.volatile.md_local.viewer.mol_sets.last()
        && let Err(e) = save_mol_set_as_gro(mol_set, &gro_path)
    {
        eprintln!("Error auto-saving GRO: {e:?}");
    }

    println!("Drag test complete. Snapshots written.");
    Ok(0.)
}

fn run_phase_free_energy(
    mol: &MoleculeSmall,
    state: &State,
    phase: Phase,
) -> Result<FreeEnergyEstimate, ParamError> {
    let mut windows = Vec::with_capacity(LAMBDAS.len());

    for &lambda in LAMBDAS {
        windows.push(run_alchemical_window(mol, state, phase, lambda)?);
    }

    let dg_kcal_mol = free_energy_ti(&windows);
    let dg_sem_kcal_mol = integrate_ti_sem(&windows);

    println!(
        "TI complete for {}: dG={:.4} kcal/mol sem={}",
        phase.name(),
        dg_kcal_mol,
        dg_sem_kcal_mol
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "n/a".to_string())
    );

    Ok(FreeEnergyEstimate {
        dg_kcal_mol,
        dg_sem_kcal_mol,
        windows,
    })
}

/// Todo: Experimenting with different approachesl.
pub fn run_alchemical(
    mol: &MoleculeSmall,
    state: &mut State,
    _scene: &mut Scene,
    _updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    if state.to_save.md_backend != MdBackend::Dynamics {
        println!(
            "LogP alchemical free energy currently runs on the built-in dynamics backend (CPU),\
             ignoring the active non-dynamics backend for this workflow."
        );
    }

    // {
    //     let phase_ = Phase::Octanol;
    //
    //     let sim_box_init = SimBoxInit::new_cube(OCTANOL_BOX_SIZE);
    //     let cell = SimBox::from_solute_atoms(&[], &sim_box_init);
    //     if pack_solvent_with_shrinking_box(
    //         &state.dev,
    //         &make_octanol(),
    //         "octanol",
    //         OCTANOL_COUNT,
    //         OCTANOL_BOX_WATER_COUNT,
    //         cell,
    //         &state.ff_param_set,
    //         Path::new("./md_out"),
    //     )
    //     .is_err()
    //     {
    //         eprintln!("Error creating solvent box");
    //     }
    //     return Ok(0.);
    // }

    // todo: Temp: Building a new sim box

    handle_success(
        &mut state.ui,
        "Running alchemical free-energy windows for water and octanol...".to_string(),
    );

    println!("Running LogP MD for water...");
    let water = run_phase_free_energy(mol, state, Phase::Water)?;
    println!("Running LogP MD for octanol...");
    let octanol = run_phase_free_energy(mol, state, Phase::Octanol)?;

    let logp_value = log_p(water.dg_kcal_mol, octanol.dg_kcal_mol, TEMP_TGT as f64) as f32;

    let mut msg = format!(
        "LogP complete. dG_water = {:.3} kcal/mol, dG_octanol = {:.3} kcal/mol, LogP = {:.3}",
        water.dg_kcal_mol, octanol.dg_kcal_mol, logp_value
    );

    if let (Some(w_sem), Some(o_sem)) = (water.dg_sem_kcal_mol, octanol.dg_sem_kcal_mol) {
        msg.push_str(&format!(
            " (TI SEMs: water {:.3}, octanol {:.3} kcal/mol; {} windows each)",
            w_sem,
            o_sem,
            water.windows.len()
        ));
    }

    handle_success(&mut state.ui, msg);

    Ok(logp_value)
}

/// We shall try different techniques. For example: Dragging a molecule across both boxes,
/// and summing the forces in each.
pub fn run(
    mol: &MoleculeSmall,
    state: &mut State,
    _scene: &mut Scene,
    _updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    println!("Starting LogP MD simulation...");

    if state.to_save.md_backend != MdBackend::Dynamics {
        println!(
            "LogP alchemical free energy currently runs on the built-in dynamics backend (CPU), ignoring the active non-dynamics backend for this workflow."
        );
    }

    // todo: Temp: Building a new sim box

    handle_success(
        &mut state.ui,
        "Running alchemical free-energy windows for water and octanol...".to_string(),
    );

    println!("Running LogP MD for water...");
    let water = run_drag(mol, state, Phase::Water)?;

    println!("Running LogP MD for octanol...");
    let octanol = run_drag(mol, state, Phase::Octanol)?;

    Ok(0.)
}
