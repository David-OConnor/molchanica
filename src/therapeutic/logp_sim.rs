//! Creates a MD sim to measure LogP or LogD. Measures a single molecule of solute in a box of
//! water, and one in a box of water-saturated octanol. Compares the alchemical free energy of
//! the two to estimate LogP or LogD.
//!
//! This is a lipophilicity measurement, and is an important metric for a molecule's use a drug.
//!
//! Rough targets to start: 500-2000 water mols in the water box
//! (What sim box size does this work out to?)
//!
//! 200-800 octanols in the octanol box. 27% mol water in it. So, 356 octanol + 132 water to start?
//! 1 copy of the solute in each box.

// todo: Create an octanol-initialization template.

use std::{collections::HashSet, time::Instant};

use dynamics::{
    FfMolType, Integrator, MdConfig, MdOverrides, ParamError, SimBoxInit, Solvent,
    TAU_TEMP_DEFAULT, snapshot::SnapshotHandlers,
};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;

use crate::{
    gromacs::{build_molecule_inputs, make_gromacs_input, on_gromacs_md_complete},
    md::{
        MdBackend, build_dynamics, custom_solvents_to_mol_commons, post_run_cleanup,
        run_dynamics_blocking,
    },
    molecules::{Atom, common::MoleculeCommon, small::MoleculeSmall},
    state::State,
    util::{handle_err, handle_success},
};
// add_copies uses bounding-sphere exclusion (~14 Å center-to-center for octanol).
// Max reliably placeable per box: ≈ (box - 16)³ × 0.64 / (4/3 π 7³).
// 55 Å → ~26 molecules; 20 is comfortably within that.
//
// TODO: add_copies is not designed for liquid-density packing. 20 octanol in a 55 Å box is
// only ~4% of liquid octanol density. A proper octanol phase requires a pre-equilibrated
// template (similar to the water template) or a lattice-based initializer.

// Rough box size by octanol count:
// 20 (8 water): 18
// 100 (37 water): 30
// 300: (111 water): 43
// 356: (132 water): 46
// 400: (148 water): 48

const OCTANOL_BOX_SIZE: f32 = 46.; // 356 octanol + 132 water ≈ 97,230 Å³ ≈ 46³ Å³
const WATER_BOX_SIZE: f32 = 35.; // Å — 35 Å → ~1,400 water mols; > 2× the 12 Å NB cutoff.

const OCTANOL_COUNT: usize = 356; // or 362 ? Or 368.
// 27 mol% water in water-saturated 1-octanol (literature value).
// water/(water+octanol) = 0.27  →  water = octanol × 0.27/0.73 ≈ octanol × 0.37.

// todo: or 0.26? I'm getting different values from different sources.
const WATER_MOL_PER_OCTANOL: f32 = 0.38;
const OCTANOL_BOX_WATER_COUNT: usize = (OCTANOL_COUNT as f32 * WATER_MOL_PER_OCTANOL) as usize;

const DT: f32 = 0.002; // ps
// Minimum ~100 ps (50_000 steps) for basic equilibration; production LogP needs ≥1 ns.
// const NUM_STEPS: usize = 50_000;
// const NUM_STEPS: usize = 2_000; // todo temp while testing the initialization
// todo temp
const NUM_STEPS: usize = 200;

const TEMP_TGT: f32 = 298.15; // Standard LogP is measured at 25 °C = 298.15 K.

// The conversion factor between ln and log10
const LOG_CONV: f32 = 1. / 2.303;

// todo: Consider using manual force field params and partial charges for Octanol, if required,
// from a vetted source.

/// A sim of the molecule in water-saturated Octanol. Returns free energy.
/// Runs the solute molecule in the given solvent configuration and returns free energy.
/// The `cfg` determines the box size and solvent; `component_name` is used for logging.
fn run_mol_in_solvent(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    cfg: MdConfig,
    component_name: &str,
) -> Result<f32, ParamError> {
    // Clone so we can set selected_for_md without mutating the caller's molecule.
    let mut mol = mol.clone();
    mol.common.selected_for_md = true;

    // Only the solute molecule goes in `mols`; the solvent is encoded in `cfg`.
    let mut mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];

    // todo: Don't let the GUI box determine which backend to use long-term; convenient now for testing.
    let energy = match state.to_save.md_backend {
        MdBackend::Dynamics => {
            let (mut md, custom_solvent) = build_dynamics(
                &state.dev,
                &mols,
                &state.ff_param_set,
                &state.mol_specific_params,
                &cfg,
                false,
                None,
                &mut HashSet::new(),
            )?;

            // todo: Consider a non-blocking approach.
            run_dynamics_blocking(&mut md, &state.dev, DT, NUM_STEPS);

            // todo: This is for the whole system including water molecules. Is this waht we want?
            let energy = {
                let snap = &md.snapshots[md.snapshots.len() - 1];
                snap.energy_data.as_ref().unwrap().energy_potential
                    + snap.energy_data.as_ref().unwrap().energy_kinetic
            };

            let custom_mol_commons =
                custom_solvents_to_mol_commons(&custom_solvent).map_err(|e| {
                    handle_err(&mut state.ui, e.clone());
                    ParamError { descrip: e }
                })?;
            for mol_common in &custom_mol_commons {
                mols.push((FfMolType::SmallOrganic, mol_common, 1));
            }

            // Add any counter-ions that add_ions may have appended to md.atoms beyond the
            // solute and custom-solvent atoms. Without this, mols_and_traj_synced fails.
            let n_custom_atoms: usize = custom_solvent.iter().map(|m| m.atoms.len()).sum();
            let ion_atom_start = mol.common.atoms.len() + n_custom_atoms;
            let ion_mol_commons: Vec<MoleculeCommon> = md.atoms[ion_atom_start..]
                .iter()
                .map(|a| {
                    let posit = Vec3 {
                        x: a.posit.x as f64,
                        y: a.posit.y as f64,
                        z: a.posit.z as f64,
                    };
                    let ion_atom = Atom {
                        serial_number: a.serial_number,
                        element: a.element,
                        posit,
                        ..Default::default()
                    };
                    MoleculeCommon {
                        ident: a.force_field_type.clone(),
                        atoms: vec![ion_atom],
                        atom_posits: vec![posit],
                        selected_for_md: true,
                        ..Default::default()
                    }
                })
                .collect();
            for mol_common in &ion_mol_commons {
                mols.push((FfMolType::SmallOrganic, mol_common, 1));
            }

            // `build_dynamics` normally adds mols for disp, but we haven't yet replaced mol_dynamics
            // with our computation here.
            state
                .volatile
                .md_local
                .viewer
                .add_mol_set(&mols, md.water.len());

            state.volatile.md_local.mol_dynamics = Some(md);

            // todo: For now; helps with vis.
            post_run_cleanup(state, scene, updates);

            // todo: Instead of running a sim, perhaps just use the one-off energy computation.
            energy
        }
        MdBackend::Gromacs => {
            let mdp = cfg.to_gromacs(NUM_STEPS, DT);

            // Build molecule entries for GROMACS input.
            let (mols_input, _pep_atom_set) = match build_molecule_inputs(
                &mols,
                &state.mol_specific_params,
                state.ui.md.peptide_static,
                None,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("GROMACS: failed to build molecule inputs: {e}");
                    return Err(ParamError {
                        descrip: "Failed to build GROMACS molecule inputs.".to_string(),
                    });
                }
            };

            let inp = match make_gromacs_input(
                mdp,
                &mols,
                mols_input,
                &state.ff_param_set,
                &cfg.sim_box,
                &cfg.solvent,
                cfg.max_init_relaxation_iters.is_some(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error creating GROMACS input: {e}");
                    return Err(ParamError {
                        descrip: "Error creating GROMACS input".to_string(),
                    });
                }
            };

            let start = Instant::now();
            handle_success(
                &mut state.ui,
                "Running MD for LogP computation with GROMACS...".to_string(),
            );
            match inp.run() {
                Ok(out) => {
                    // todo: Blocking for now.
                    let elapsed = start.elapsed().as_millis();
                    on_gromacs_md_complete(state, &out, elapsed);
                }
                Err(e) => {
                    eprintln!("Error running GROMACS: {e}");
                    return Err(ParamError {
                        descrip: "Error running GROMACS".to_string(),
                    });
                }
            }
            0. // todo for now during a refactor.
        }
        MdBackend::Orca => {
            return Err(ParamError {
                descrip: "ORCA is not supported for logp sim".to_string(),
            });
        }
    };

    println!(
        "Free energy computed for the {} component: {:.2}",
        component_name, energy
    );

    Ok(energy)
}

/// A sim of the molecule in water. Returns free energy.
fn run_water(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        temp_target: TEMP_TGT,
        barostat_cfg: Some(Default::default()),
        snapshot_handlers: SnapshotHandlers::default(),
        sim_box: SimBoxInit::new_cube(WATER_BOX_SIZE),
        solvent: Solvent::WaterOpc,
        overrides: MdOverrides {
            snapshots_during_equilibration: true,
            ..Default::default()
        },

        ..Default::default()
    };

    run_mol_in_solvent(mol, state, scene, updates, cfg, "water")
}

fn run_octanol(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    // // todo: This is of convenience: Pack an octanol box here to develop a template.
    // let atoms = pack

    // Build a MolDynamics from the octanol MoleculeSmall so it can be used in Solvent::Custom.
    // let octanol_dyn = {
    //     // let msp = state.mol_specific_params.get(&octanol.common.ident).cloned();
    //     MolDynamics {
    //         ff_mol_type: FfMolType::SmallOrganic,
    //         atoms: octanol
    //             .common
    //             .atoms
    //             .iter()
    //             .map(|a| a.to_generic())
    //             .collect(),
    //         atom_posits: Some(octanol.common.atom_posits.clone()),
    //         bonds: octanol
    //             .common
    //             .bonds
    //             .iter()
    //             .map(|b| b.to_generic())
    //             .collect(),
    //         adjacency_list: Some(octanol.common.adjacency_list.clone()),
    //         // todo: Do we need molecule-specific params for octanol?
    //         // mol_specific_params: msp,
    //         mol_specific_params: None,
    //         ..Default::default()
    //     }
    // };

    let sim_box_init = SimBoxInit::new_cube(OCTANOL_BOX_SIZE);

    // ----------------------

    // // todo: For now; TBD on how this template-creation API for custom molecules will end up.
    //
    // {
    //     // `dynamics` will build the cell based on the init for the main run; this is just for
    //     // the octanol template generation.
    //     let cell = SimBox::from_solute_atoms(&[], &sim_box_init);
    //     let (mols, snaps) = pack_solvent_with_shrinking_box(
    //         &state.dev,
    //         &octanol_dyn,
    //         "octanol",
    //         OCTANOL_COUNT,
    //         OCTANOL_BOX_WATER_COUNT,
    //         cell,
    //         &state.ff_param_set,
    //         Path::new("./md_out"), // todo troubleshooting
    //     )?;
    // }
    //
    // return Ok(0.); // todo temp!!

    // -------------------

    // todo: We have an octanol template now. How do we add our custom partial charges etc?

    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        temp_target: TEMP_TGT,
        barostat_cfg: Some(Default::default()),
        snapshot_handlers: SnapshotHandlers::default(),
        sim_box: sim_box_init.clone(),
        // solvent: Solvent::Custom((vec![(octanol_dyn, OCTANOL_COUNT)], OCTANOL_BOX_WATER_COUNT)),
        // todo: this must be set up correctly for out sim in terms of mol density and ratio
        // todo over water to octanol.
        solvent: Solvent::OctanolWithWater,
        overrides: MdOverrides {
            snapshots_during_equilibration: true,
            ..Default::default()
        },
        ..Default::default()
    };

    run_mol_in_solvent(mol, state, scene, updates, cfg, "octanol")
}

pub fn run(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    let e_water = 0.;
    let e_octanol = 0.;

    let e_water = run_water(mol, state, scene, updates)?;
    let e_octanol = run_octanol(mol, state, scene, updates)?;

    // todo: How do we calculate alchemical free energies?
    // todo: One LLM thinkjs I shoujld calculate something across different
    // todo lamda values (~20 of them, e.g. lamda = 0.0, 0.05, 0.1, to 1.0

    Ok((e_water - e_octanol) * LOG_CONV)
}
