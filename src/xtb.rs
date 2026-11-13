//! Native xTB GFN-FF MD and trajectory playback.

use std::{
    collections::HashSet,
    io,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use bio_files::{
    gromacs::OutputEnergy,
    xtb::{XtbFrame, XtbInput, XtbOutput},
};
use dynamics::{
    FfMolType, ForceFieldFamily, Integrator, SimBoxInit, params::potential::SystemPotentialBackend,
    snapshot::Snapshot,
};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::common::MoleculeCommon;

use crate::{
    cam::{VIEW_DIR_FRONT, reset_camera},
    file_io::save_mol_set_as_gro,
    md::{self, STATIC_ATOM_DIST_THRESH, trajectory::Trajectory},
    state::{ComputationType, State},
    util::{format_elapsed, handle_err, handle_success},
};

type ViewerMols = Vec<(FfMolType, MoleculeCommon, usize)>;

struct PreparedInput {
    input: XtbInput,
    mols: ViewerMols,
    peptide_atoms: HashSet<(usize, usize)>,
}

struct RunResult {
    output: XtbOutput,
    mols: ViewerMols,
    peptide_atoms: HashSet<(usize, usize)>,
    dt: f32,
    elapsed: Duration,
}

pub struct XtbRun {
    receiver: Receiver<Result<RunResult, String>>,
    cancel: Arc<AtomicBool>,
}

impl XtbRun {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for XtbRun {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn prepare_input(state: &State) -> io::Result<PreparedInput> {
    if state.force_fields.family != ForceFieldFamily::Gfnn {
        return Err(io::Error::other("The xTB backend requires GFN-FF."));
    }

    let cfg = &state.to_save.md.config;
    let selected = md::get_mols_sel_for_md(state);
    if selected.is_empty() {
        return Err(io::Error::other("Select at least one molecule for xTB MD."));
    }

    if state.ui.md.peptide_static
        && selected
            .iter()
            .any(|(kind, _, _)| *kind == FfMolType::Peptide)
    {
        return Err(io::Error::other("Disable Pep static for native xTB MD."));
    }

    let box_dims = match cfg.sim_box {
        SimBoxInit::Fixed((lo, hi)) => {
            let extent = hi - lo;
            if extent.to_arr().iter().any(|v| !v.is_finite() || *v <= 0.) {
                return Err(io::Error::other(
                    "The fixed box for xTB molecule placement needs finite, positive dimensions.",
                ));
            }

            Some((extent.x, extent.y, extent.z))
        }
        SimBoxInit::Pad(_) => None,
    };
    let ligand_selected = selected
        .iter()
        .any(|(kind, _, _)| *kind == FfMolType::SmallOrganic);
    let near_lig =
        (state.ui.md.peptide_only_near_ligs && ligand_selected).then_some(STATIC_ATOM_DIST_THRESH);
    let (molecules, peptide_atoms) = md::setup_mols_dyn_with_residues(
        &selected,
        &state.mol_specific_params,
        false,
        near_lig,
        box_dims,
        None,
        ForceFieldFamily::Gfnn,
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    let requested_count: usize = selected.iter().map(|(_, _, count)| (*count).max(1)).sum();
    if molecules.len() != requested_count {
        return Err(io::Error::other("Unable to place all xTB molecule copies."));
    }

    let mut atoms = Vec::new();
    let mut mols = Vec::new();
    let mut names = selected
        .iter()
        .flat_map(|(_, m, copies)| std::iter::repeat_n(m.ident.clone(), (*copies).max(1)));

    for molecule in molecules {
        let name = names.next().unwrap_or_else(|| "GFN-FF".into());
        let mut molecule_atoms = molecule.atoms.clone();
        for atom in &mut molecule_atoms {
            atom.force_field_type = None;
            atom.partial_charge = None;
        }

        if let Some(positions) = &molecule.atom_posits {
            if positions.len() != molecule_atoms.len() {
                return Err(io::Error::other(format!(
                    "{name} has mismatched atom and coordinate counts after xTB placement."
                )));
            }

            for (atom, position) in molecule_atoms.iter_mut().zip(positions) {
                atom.posit = *position;
            }
        }

        if let Some(atom) = molecule_atoms
            .iter()
            .find(|atom| atom.posit.to_arr().iter().any(|v| !v.is_finite()))
        {
            return Err(io::Error::other(format!(
                "{name} atom {} has non-finite coordinates after xTB placement.",
                atom.serial_number
            )));
        }

        let viewer_atoms: Vec<_> = molecule_atoms.iter().map(Into::into).collect();
        let bonds = molecule
            .bonds
            .iter()
            .map(|b| mol_defs::molecules::Bond::from_generic(b, &viewer_atoms))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| io::Error::other("Invalid xTB input bonds"))?;
        mols.push((
            molecule.ff_mol_type,
            MoleculeCommon::new(name, viewer_atoms, bonds, Default::default(), None),
            1,
        ));
        atoms.extend(molecule_atoms);
    }

    let input = native_input(state, atoms)?;

    Ok(PreparedInput {
        input,
        mols,
        peptide_atoms,
    })
}

fn native_input(state: &State, atoms: Vec<bio_files::AtomGeneric>) -> io::Result<XtbInput> {
    let set = state
        .selected_force_field()
        .gfnff()
        .ok_or_else(|| io::Error::other("GFN-FF parameters are not loaded"))?;
    let cfg = &state.to_save.md.config;
    set.validate_config(cfg)
        .map_err(|e| io::Error::other(e.to_string()))?;
    if state.ui.md.alchemical_enabled {
        return Err(io::Error::other(
            "Native xTB does not support alchemical windows.",
        ));
    }

    let charge = set.charge().map_err(|e| io::Error::other(e.to_string()))?;
    let thermostat = match cfg.integrator {
        Integrator::Leapfrog { thermostat } | Integrator::VerletVelocity { thermostat } => {
            thermostat.is_some()
        }
        Integrator::LangevinMiddle { .. } => true,
    };
    let input = XtbInput {
        atoms,
        dt: state.to_save.md.dt as f64,
        num_steps: state.to_save.md.num_steps as usize,
        temperature: cfg.temp_target as f64,
        thermostat,
        output_interval: cfg.snapshot_handlers.memory.unwrap_or(10),
        relaxation_iters: cfg.max_init_relaxation_iters,
        charge,
        alpb_solvent: set
            .alpb_solvent
            .as_deref()
            .map(str::trim)
            .map(str::to_owned),
        parameter_file: set.parameter_file.clone(),
        executable: set.executable.clone(),
        timeout: None,
    };
    input.validate()?;

    Ok(input)
}

fn prepare_run(state: &State) -> io::Result<PreparedInput> {
    if let Some(md::runs::RunInputs::Xtb { geometry, mols }) =
        &state.volatile.md_local.opened_inputs
    {
        let input = native_input(state, geometry.clone())?;

        return Ok(PreparedInput {
            input,
            mols: mols.clone(),
            peptide_atoms: HashSet::new(),
        });
    }

    prepare_input(state)
}

pub fn save_input_files(state: &State, path: &Path) -> io::Result<()> {
    prepare_run(state)?.input.save(path)
}

pub fn launch_md(state: &mut State) {
    if state.volatile.thread_receivers.xtb_md.is_some() {
        handle_err(&mut state.ui, "An xTB MD run is already active.".into());

        return;
    }

    let prepared = match prepare_run(state) {
        Ok(input) => input,
        Err(error) => {
            handle_err(&mut state.ui, error.to_string());

            return;
        }
    };
    let (tx, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    let computation = state.volatile.ongoing_computations.start(
        ComputationType::MolecularDynamics,
        "xTB",
        Some(format!("{} steps", prepared.input.num_steps)),
    );
    state.volatile.md_local.running = false;
    state.volatile.md_local.launching = false;
    state.volatile.md_local.mol_dynamics = None;
    state.volatile.md_local.computation = None;

    thread::spawn(move || {
        let start = Instant::now();
        let result = prepared
            .input
            .run_with_cancel(&worker_cancel)
            .map(|output| {
                let mols: Vec<_> = prepared
                    .mols
                    .iter()
                    .map(|(kind, mol, count)| ((*kind).into(), mol.clone(), *count))
                    .collect();
                let set = md::viewer::ViewerMolSet::from_mols("xTB MD".into(), &mols);
                let set_path = output.directory.join(crate::project::MD_MOLECULE_SET_FILE);
                if let Err(error) = crate::project::save_md_mol_set(&set_path, &set) {
                    eprintln!("Error saving xTB molecule set: {error}");
                }
                if let Err(error) =
                    save_mol_set_as_gro(&set, &output.directory.join("molecules.gro"))
                {
                    eprintln!("Error saving xTB viewer molecules: {error}");
                }

                RunResult {
                    output,
                    mols: prepared.mols,
                    peptide_atoms: prepared.peptide_atoms,
                    dt: prepared.input.dt as f32,
                    elapsed: start.elapsed(),
                }
            })
            .map_err(|error| error.to_string());
        drop(computation);
        let _ = tx.send(result);
    });
    state.volatile.thread_receivers.xtb_md = Some(XtbRun { receiver, cancel });
    handle_success(&mut state.ui, "Running native xTB MD...".into());
}

pub(super) fn trajectory_snapshots(frames: Vec<XtbFrame>) -> Vec<Arc<Snapshot>> {
    frames
        .into_iter()
        .map(|frame| {
            Arc::new(Snapshot {
                time: frame.time,
                atom_posits: frame
                    .geometry
                    .atoms
                    .iter()
                    .map(|a| a.posit.into())
                    .collect(),
                energy_data: frame.energy_hartree.map(|energy| {
                    OutputEnergy {
                        potential_energy: Some((energy * 627.509_474_063_1 * 4.184) as f32),
                        kinetic_energy: Some(f32::NAN),
                        temperature: Some(f32::NAN),
                        pressure: Some(f32::NAN),
                        volume: Some(f32::NAN),
                        density: Some(f32::NAN),
                        ..Default::default()
                    }
                    .into()
                }),
                ..Default::default()
            })
        })
        .collect()
}

pub fn poll_md(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let result = match state
        .volatile
        .thread_receivers
        .xtb_md
        .as_ref()
        .map(|run| run.receiver.try_recv())
    {
        Some(Ok(result)) => result,
        Some(Err(TryRecvError::Disconnected)) => Err("xTB MD worker disconnected".into()),
        _ => return,
    };
    state.volatile.thread_receivers.xtb_md = None;
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            handle_err(&mut state.ui, error);

            return;
        }
    };

    let snaps = trajectory_snapshots(result.output.frames);
    let mol_refs: Vec<_> = result
        .mols
        .iter()
        .map(|(kind, mol, count)| (*kind, mol, *count))
        .collect();
    state.volatile.md_local.viewer.add_mol_set(&mol_refs, 0);
    state.volatile.md_local.viewer.mol_set_active =
        Some(state.volatile.md_local.viewer.mol_sets.len() - 1);
    let run_path = result.output.directory.join("xtb.trj");
    if let Some(set) = state.volatile.md_local.viewer.mol_sets.last_mut() {
        set.name = "xTB MD".into();
        set.path = Some(run_path.clone());
    }
    state.volatile.md_local.pep_atom_set = result.peptide_atoms;

    for trajectory in &mut state.trajectories {
        trajectory.ui_active = false;
    }

    let mut trajectory = Trajectory::new_in_memory(
        snaps.clone(),
        format!("xTB run {}", state.trajectories.len() + 1),
        result.dt,
    );

    trajectory.ui_active = true;
    trajectory.run_mol_set_path = Some(run_path);
    state.trajectories.push(trajectory);
    state.volatile.md_local.replace_snaps(snaps);
    reset_camera(state, scene, updates, VIEW_DIR_FRONT);
    md::viewer::draw_mols(state, scene, updates);

    handle_success(
        &mut state.ui,
        format!(
            "xTB MD complete in {}. Files: {}",
            format_elapsed(result.elapsed),
            result.output.directory.display()
        ),
    );
}
