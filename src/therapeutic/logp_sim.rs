//! Creates a MD workflow to measure LogP or LogD by alchemically decoupling the
//! solute in water and in water-saturated octanol, then comparing the free energies.
//!
//! This uses thermodynamic integration over a set of fixed lambda windows. For now
//! we run the alchemical windows on the built-in `dynamics` backend and on CPU so
//! that the lambda-dependent interaction scaling and `dH/dlambda` bookkeeping are
//! consistent with the solvent templates in that crate.

use std::{collections::HashSet, path::Path};

use dynamics::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, ParamError, SimBox,
    SimBoxInit, Solvent, TAU_TEMP_DEFAULT,
    alchemical::{LambdaWindow, collect_window, free_energy_ti, log_p},
    make_octanol, pack_solvent_with_shrinking_box,
    snapshot::SnapshotHandlers,
};
use graphics::{EngineUpdates, Scene};

use crate::{
    md::{MdBackend, build_dynamics, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
    screening::MOL_CACHE_SIZE_ATOM_COUNT,
    state::State,
    util::handle_success,
};

const OCTANOL_BOX_SIZE: f32 = 46.; // 356 octanol + 132 water ≈ 97,230 Å³ ≈ 46³ Å³
const WATER_BOX_SIZE: f32 = 35.; // Å — 35 Å → ~1,400 water mols; > 2× the 12 Å NB cutoff.

const OCTANOL_COUNT: usize = 356; // or 362 ? Or 368.
// 27 mol% water in water-saturated 1-octanol (literature value).
// water/(water+octanol) = 0.27  →  water = octanol × 0.27/0.73 ≈ octanol × 0.37.

// todo: or 0.26? I'm getting different values from different sources.
const WATER_MOL_PER_OCTANOL: f32 = 0.38;
const OCTANOL_BOX_WATER_COUNT: usize = (OCTANOL_COUNT as f32 * WATER_MOL_PER_OCTANOL) as usize;

const DT: f32 = 0.002; // ps

// Starter settings so the workflow is usable from the UI. Production-quality LogP
// predictions should use much longer sampling and usually denser endpoint spacing.
const EQUIL_STEPS_PER_WINDOW: usize = 500;
const PROD_STEPS_PER_WINDOW: usize = 1_000;
const SNAPSHOT_RATIO: usize = 10;

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

fn build_alchemical_cfg(phase: Phase) -> MdConfig {
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
            ..Default::default()
        },
        sim_box: SimBoxInit::new_cube(phase.box_size()),
        solvent: phase.solvent(),
        overrides: MdOverrides {
            snapshots_during_equilibration: false,
            ..Default::default()
        },
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
    let cfg = build_alchemical_cfg(phase);
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

pub fn run(
    mol: &MoleculeSmall,
    state: &mut State,
    _scene: &mut Scene,
    _updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    if state.to_save.md_backend != MdBackend::Dynamics {
        println!(
            "LogP alchemical free energy currently runs on the built-in dynamics backend (CPU), ignoring the active non-dynamics backend for this workflow."
        );
    }

    {
        let phase_ = Phase::Octanol;

        let sim_box_init = SimBoxInit::new_cube(OCTANOL_BOX_SIZE);
        let cell = SimBox::from_solute_atoms(&[], &sim_box_init);
        if pack_solvent_with_shrinking_box(
            &state.dev,
            &make_octanol(),
            "octanol",
            OCTANOL_COUNT,
            OCTANOL_BOX_WATER_COUNT,
            cell,
            &state.ff_param_set,
            Path::new("./md_out"),
        )
        .is_err()
        {
            eprintln!("Error creating solvent box");
        }
        return Ok(0.);
    }

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
