//! For creating ORCA QM inputs to run, and visualizing output.
//!
//! [ORCA recommendations for methods, basis fns etc](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)

use std::{
    fmt::Display,
    io,
    sync::{mpsc, mpsc::Receiver},
    thread,
};

use bio_files::orca::{
    GeomOptThresh, OrcaInput, OrcaOutput, basis_sets::BasisSetCategory, dynamics::DynamicsOutput,
};
use dynamics::snapshot::Snapshot;
use mol_defs::molecules::MolType;

use crate::{
    state::{ComputationType, State},
    util::{RedrawFlags, handle_err, handle_success},
};

/// An ORCA run in progress on a worker thread, and the molecule to apply its result to.
pub struct OrcaRun {
    rx: Receiver<io::Result<OrcaOutput>>,
    mol_type: MolType,
    mol_i: usize,
    /// Confirms the molecule at `mol_i` is still the one the run was for, e.g. if molecules were
    /// closed in the meantime.
    ident: String,
}

impl OrcaRun {
    pub fn try_recv(&self) -> Result<io::Result<OrcaOutput>, mpsc::TryRecvError> {
        self.rx.try_recv()
    }
}

/// Run ORCA with `state.orca.input` on the active molecule, on a worker thread. ORCA runs can take
/// minutes or more. The result is applied by [`on_run_complete`].
pub fn launch(state: &mut State) {
    let Some((mol_type, mol_i)) = state.volatile.active_mol else {
        return;
    };
    let Some(mol) = state.get_mol(mol_type, mol_i) else {
        return;
    };
    let ident = mol.common().ident.clone();

    let task_type = state.orca.task_type;
    let comp_type = match task_type {
        TaskType::MolDynamics => ComputationType::MolecularDynamics,
        _ => ComputationType::QuantumMechanics,
    };

    let computation = state.volatile.ongoing_computations.start(
        comp_type,
        "ORCA",
        Some(format!("{task_type} of {ident}")),
    );

    let input = state.orca.input.clone();
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let result = input.run();
        drop(computation);

        let _ = tx.send(result);
    });

    state.volatile.thread_receivers.orca_run = Some(OrcaRun {
        rx,
        mol_type,
        mol_i,
        ident,
    });
}

/// Apply the result of an ORCA run started by [`launch`].
pub fn on_run_complete(
    state: &mut State,
    run: OrcaRun,
    result: io::Result<OrcaOutput>,
    redraw: &mut RedrawFlags,
) {
    let out = match result {
        Ok(out) => out,
        Err(e) => {
            handle_err(&mut state.ui, format!("Problem running ORCA: {e:?}"));
            return;
        }
    };

    // Set when the output needs to be applied to the molecule the run was for.
    let mut atom_count_expected = None;

    // This should correspond to the task.
    match &out {
        OrcaOutput::Text(t) => {
            println!("ORCA run complete. Output: \n\n{t}");

            handle_success(&mut state.ui, "ORCA run complete".to_owned());
            return;
        }
        OrcaOutput::Dynamics(o) => {
            println!("\n\nMD Trajectory: \n\n{:?}", o.trajectory);
        }
        OrcaOutput::Charges(o) => {
            // println!("Charge output: {:?}", o);
            // println!("Orca raw output text: \n\n{:?}\n\n\n\n", o.text);

            println!("\n------\nORCA charge generation complete.\n\n Charge:");
            for charge in &o.charges {
                println!("-{charge:?}");
            }

            println!("\n\nDipole:");
            for charge in &o.dipole {
                println!("-{charge:?}");
            }

            println!("\n\nQuadrupole:");
            for charge in &o.quadrupole {
                println!("-{charge:?}");
            }

            println!("\n\nOctopole:");
            for charge in &o.octopole {
                println!("-{charge:?}");
            }

            println!("\n-------\n");

            atom_count_expected = Some(o.charges.len());
        }
        OrcaOutput::Geometry(p) => {
            println!("Updated Atom positions from ORCA:");
            for (i, p) in p.posits.iter().enumerate() {
                println!("{}: {p}", i + 1);
            }

            atom_count_expected = Some(p.posits.len());
        }
    }

    if let OrcaOutput::Dynamics(o) = out {
        update_snapshots(state, o);
        handle_success(&mut state.ui, "ORCA MD run complete".to_owned());
        return;
    }

    let Some(mut mol) = state
        .get_mol_mut(run.mol_type, run.mol_i)
        .filter(|m| m.common().ident == run.ident)
    else {
        handle_err(
            &mut state.ui,
            format!(
                "{} was closed before its ORCA run completed; the result wasn't applied.",
                run.ident
            ),
        );
        return;
    };

    if atom_count_expected != Some(mol.common().atoms.len()) {
        handle_err(
            &mut state.ui,
            format!(
                "The ORCA result's atom count doesn't match {}'s; it wasn't applied.",
                run.ident
            ),
        );
        return;
    }

    let msg = match out {
        OrcaOutput::Charges(o) => {
            for (i, q) in o.charges.into_iter().enumerate() {
                mol.common_mut().atoms[i].partial_charge = Some(q.charge as f32);
            }
            format!("MBIS charges assigned for {}", run.ident)
        }
        OrcaOutput::Geometry(p) => {
            for (i, posit) in p.posits.into_iter().enumerate() {
                mol.common_mut().atom_posits[i] = posit;
            }
            format!("Geometry optimized for {}", run.ident)
        }
        OrcaOutput::Text(_) | OrcaOutput::Dynamics(_) => unreachable!(),
    };

    // For charges, maybe only required if in color-by-charge mode.
    redraw.set(run.mol_type);
    handle_success(&mut state.ui, msg);
}

#[derive(Default)]
// todo: Some of this is UI state; move to a place that makes sense A/R.
pub struct StateOrca {
    pub input: OrcaInput,
    pub basis_set_cat: BasisSetCategory,
    pub task_type: TaskType,
    /// For the drop-down; only relevant for the geometry optimization task.
    pub geom_opt_thresh: GeomOptThresh,
    // pub dynamics: Dynamics,
}

/// A copy-type, e.g. without the inner values of bio_files::orca::Task.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TaskType {
    SinglePoint,
    /// Note: Not the Orca default, nor `bio_files::orca::Task` default.
    #[default]
    GeometryOptimization,
    MbisCharges,
    MolDynamics,
}

impl TaskType {
    pub fn help_text(self) -> String {
        use TaskType::*;
        match self {
            SinglePoint => "Compute the single-point energy of this molecule.",
            GeometryOptimization => "Compute optimal geometry, and apply this to this molecule's atoms.",
            MbisCharges =>                             "Compute and assign MBIS partial charges for this molecule. This is an accurate QM method, but is very \
                            slow; it may take 10 minutes or longer for a small organic molecule. This replaces any existing partial \
                            charges on this molecule.",
            MolDynamics =>                             "Run MD using ORCA. This is much slower than our normal MD system, but \
                             more accurate Uses settings from the MD section of the UI as well, including number of steps,\
                             dt, and temperature.",
        }.to_string()
    }
}

impl Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TaskType::*;

        let v = match self {
            SinglePoint => "Single pt energy",
            GeometryOptimization => "Optimize geom",
            MbisCharges => "MBIS charges",
            MolDynamics => "Mol dynamics",
        };

        write!(f, "{v}")
    }
}

pub fn update_snapshots(state: &mut State, out: DynamicsOutput) {
    match &mut state.volatile.md_local.mol_dynamics {
        Some(md) => {
            md.snapshots = Vec::new();

            for (i, step) in out.trajectory.iter().enumerate() {
                let time = i as f32 * state.to_save.md.dt * 1_000.;

                // todo: You still need to reassign the atoms to this
                // todo for the playback.
                let atom_posits: Vec<_> = step.atoms.iter().map(|a| a.posit.into()).collect();

                md.snapshots.push(Snapshot {
                    // todo: You can also get time from the comment.
                    time: time as f64,
                    atom_posits,
                    ..Default::default()
                })
            }
        }
        None => {
            // state.volatile.md_local.mol_dynamics = Some(MdState::new(
            //
            // ))
        }
    }
}
