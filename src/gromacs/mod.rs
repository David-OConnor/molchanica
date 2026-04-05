//! GROMACS integration - can create GROMACS inputs, and parse outputs (Trajectory, energy etc)
//! from data structures and visualization
//! in this program. Uses the `bio_files` lib's `gromacs` functionality; this module interfaces
//! Molchanica data structure's to its API.
//!
//! [`run_dynamics`] is the entry point. Its signature mirrors
//! [`crate::md::build_dynamics`] + [`crate::md::run_dynamics_blocking`], but
//! instead of running our own integrator it:
//!
//! 1. Converts Molchanica molecules and `MdConfig` into a [`GromacsInput`].
//! 2. Writes `.gro`, `.top`, `.mdp` files and executes `gmx grompp` + `gmx mdrun`.
//! 3. Parses the resulting trajectory and returns it as `Vec<Snapshot>` — the same
//!    type used by the `dynamics` crate — so the rest of the application can play it
//!    back without changes.
//!
//! For how data structures flow into this function see `md.rs`, which performs the
//! same pre-processing for the built-in `dynamics` pipeline.

use std::{
    collections::{HashMap, HashSet},
    io,
    path::Path,
    sync::mpsc,
    thread,
    time::Instant,
};

use bio_files::{
    AtomGeneric, BondGeneric, FrameSlice,
    gromacs::{GromacsInput, GromacsOutput, MoleculeInput, solvate::WaterModel},
    md_params::ForceFieldParams,
};
use dynamics::{FfMolType, SimBoxInit, Solvent, WATER_TEMPLATE_60A, WaterInitTemplate};

use crate::{
    md::{
        STATIC_ATOM_DIST_THRESH, filter_peptide_atoms, get_mols_sel_for_md, trajectory::Trajectory,
    },
    molecules::common::MoleculeCommon,
    state::State,
    util::{handle_err, handle_success},
};

pub fn make_gromacs_input(
    state: &State,
) -> io::Result<(GromacsInput, Vec<(FfMolType, MoleculeCommon, usize)>)> {
    // Filter molecules for docking by if they're selected.
    // mut so we can move their posits in the initial snapshot change.

    // Avoids borrow error.
    let mut md_pep_sel = state.volatile.md_local.peptide_selected.clone();

    // Clone selected molecules to release the immutable borrow of `state`
    // before the mutable borrow needed by update_mols_for_disp.
    let mols_owned: Vec<_> = {
        let mols = get_mols_sel_for_md(state);
        mols.iter()
            .map(|(ff, mol, count)| (*ff, (*mol).clone(), *count))
            .collect()
    };

    let mols: Vec<(FfMolType, &MoleculeCommon, usize)> = mols_owned
        .iter()
        .map(|(ff, mol, count)| (*ff, mol, *count))
        .collect();

    let near_lig_thresh = if state.ui.md.peptide_only_near_ligs {
        Some(STATIC_ATOM_DIST_THRESH)
    } else {
        None
    };

    // Build molecule entries for GROMACS input.
    let molecules = match build_molecule_inputs(
        &mols,
        &state.mol_specific_params,
        &mut md_pep_sel,
        state.ui.md.peptide_static,
        near_lig_thresh,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("GROMACS: failed to build molecule inputs: {e}");
            return Err(io::Error::new(io::ErrorKind::Other, e));
        }
    };

    let cfg = &state.to_save.md_config;

    // Determine simulation box dimensions (nm).
    let box_nm = sim_box_nm(&cfg.sim_box);

    // Map MdConfig + explicit dt/n_steps onto MdpParams.
    let mdp = state
        .to_save
        .md_config
        .to_gromacs(state.to_save.num_md_steps as usize, state.to_save.md_dt);

    // todo: I'm not sure about this. What if the same FF type is used for multiple molecules.
    // todo: Test adn assess.
    // Merge all loaded FF tables into a single global fallback so that mass/LJ
    // lookups succeed for any molecule type (peptide, small organic, DNA, etc.).
    // Entries from earlier sources win on conflicts; order: small_mol first so GAFF2
    // atom-type masses are not clobbered by the heavier protein FF tables.
    let ff_global = [
        state.ff_param_set.small_mol.as_ref(),
        state.ff_param_set.peptide.as_ref(),
        state.ff_param_set.dna.as_ref(),
        state.ff_param_set.rna.as_ref(),
        state.ff_param_set.lipids.as_ref(),
    ]
    .into_iter()
    .flatten()
    .fold(None::<ForceFieldParams>, |acc, ff| {
        Some(match acc {
            None => ff.clone(),
            Some(merged) => merged.merge_with(ff),
        })
    });

    let water_model = match &cfg.solvent {
        Solvent::None => None,
        Solvent::WaterOpc | Solvent::WaterOpcSpecifyMolCount(_) | Solvent::Custom(_) => Some(
            WaterModel::Opc(WaterInitTemplate::from_bytes(WATER_TEMPLATE_60A)?.to_gromacs()),
        ),
    };

    Ok((
        GromacsInput {
            mdp,
            molecules,
            box_nm,
            ff_global,
            water_model,
            minimize_energy: cfg.max_init_relaxation_iters.is_some(),
        },
        mols_owned,
    ))
}

// todo: Replaced by `launch_dynamics`?
// /// Run a GROMACS MD simulation and return the trajectory as `Vec<Snapshot>`.
// ///
// /// This is conceptually equivalent to calling `md::build_dynamics` followed by
// /// `md::run_dynamics_blocking`, but delegates execution to the external `gmx`
// /// binary instead of our own integrator.
// ///
// /// # Arguments
// ///
// /// Arguments mirror `md::build_dynamics` for drop-in use by the UI layer:
// ///
// /// - `_dev` — unused (GROMACS handles device selection internally)
// /// - `mols_in` — `(ff_mol_type, molecule, copy_count)` triples
// /// - `param_set` — global Amber force-field parameter set
// /// - `mol_specific_params` — per-ligand GAFF2 parameters keyed by molecule ident
// /// - `cfg` — MD configuration (timestep, temperature, box, etc.)
// /// - `static_peptide` — if `true`, the peptide atoms are held fixed
// /// - `peptide_only_near_lig` — if `Some(thresh)`, include only peptide atoms
// ///   within `thresh` Å of any ligand atom
// /// - `pep_atom_set` — populated with the `(mol_i, atom_i)` pairs of included
// ///   peptide atoms (same semantics as in `md.rs`)
// /// - `fast_init` — if `true`, skip water and initial energy relaxation
// /// - `dt` — timestep in **ps**
// /// - `n_steps` — total number of MD steps
// pub fn run_dynamics(state: &mut State) -> (Vec<Snapshot>, Vec<usize>) {
//     let (input, mols) = match make_gromacs_input(state) {
//         Ok(input) => input,
//         Err(e) => {
//             eprintln!("Error creating GROMACS input");
//             return (Vec::new(), Vec::new());
//         }
//     };
//
//     // Compute mol_start_indices from the actual molecules being passed to GROMACS.
//     // Each copy of each molecule gets its own entry; offsets are over solute atoms only
//     // (water is handled separately in convert_snapshots and not indexed here).
//     let mut offset = 0;
//     let mut mol_start_indices = Vec::new();
//     for m in &input.molecules {
//         for _ in 0..m.count {
//             mol_start_indices.push(offset);
//             offset += m.atoms.len();
//         }
//     }
//
//     let mols_ref: Vec<(FfMolType, &MoleculeCommon, usize)> =
//         mols.iter().map(|(ff, mol, c)| (*ff, mol, *c)).collect();
//
//     state.volatile.md_local.update_mols_for_disp(&mols_ref);
//
//     match input.run() {
//         Ok(out) => {
//             let snapshots = gromacs_frames_to_ss(&out.trajectory, out.solute_atom_count);
//             (snapshots, mol_start_indices)
//         }
//         Err(e) => {
//             eprintln!("\nGROMACS run failed: \n{e}");
//             (Vec::new(), Vec::new())
//         }
//     }
// }

/// Convert Molchanica molecule data into `Vec<MoleculeInput>` for GROMACS.
fn build_molecule_inputs(
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    static_peptide: bool,
    near_lig_thresh: Option<f64>,
) -> Result<Vec<MoleculeInput>, String> {
    use bio_files::create_bonds;

    let mut result = Vec::new();

    for (ff_mol_type, mol, count) in mols_in {
        if !mol.selected_for_md {
            continue;
        }

        let (atoms, bonds, name) = if *ff_mol_type == FfMolType::Peptide {
            // Apply the same peptide filtering as the built-in pipeline.
            let atoms = filter_peptide_atoms(pep_atom_set, mol, mols_in, near_lig_thresh);
            let bonds = create_bonds(&atoms);
            (atoms, bonds, "PEP".to_string())
        } else {
            let atoms: Vec<AtomGeneric> = mol.atoms.iter().map(|a| a.to_generic()).collect();
            let bonds: Vec<BondGeneric> = mol.bonds.iter().map(|b| b.to_generic()).collect();
            // Use the molecule ident as the topology name, truncated to 6 chars.
            let name = sanitise_mol_name(&mol.ident);
            (atoms, bonds, name)
        };

        let ff_params = mol_specific_params.get(&mol.ident).cloned();

        result.push(MoleculeInput {
            name,
            atoms,
            bonds,
            ff_params,
            count: *count,
        });

        // If the peptide is static, set all its positions fixed by noting them in
        // pep_atom_set (already done inside filter_peptide_atoms above). GROMACS
        // handles frozen groups via `freezegrps` / `freezedim` MDP options; that
        // extension is left for future work.
        let _ = static_peptide; // used via filter_peptide_atoms selection
    }

    Ok(result)
}

/// Sanitise a molecule ident for use as a GROMACS molecule name:
/// keep only alphanumerics, replace the rest with underscores, max 6 chars.
fn sanitise_mol_name(ident: &str) -> String {
    let s: String = ident
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .take(6)
        .collect();
    if s.is_empty() { "MOL".to_string() } else { s }
}

/// Convert `SimBoxInit` (Å) to GROMACS box dimensions (nm).
///
/// For `Pad` boxes the side-length is estimated as 2 × pad + a nominal
/// molecule diameter; GROMACS `grompp` will adjust it to fit the system.
fn sim_box_nm(sim_box: &SimBoxInit) -> Option<(f64, f64, f64)> {
    match sim_box {
        SimBoxInit::Fixed((lo, hi)) => {
            let x = ((hi.x - lo.x) as f64) / 10.0;
            let y = ((hi.y - lo.y) as f64) / 10.0;
            let z = ((hi.z - lo.z) as f64) / 10.0;
            Some((x, y, z))
        }
        SimBoxInit::Pad(margin) => {
            // Rough estimate — GROMACS adjusts this during grompp anyway.
            let side = ((*margin as f64) * 2.0 + 30.0) / 10.0; // convert Å → nm
            Some((side, side, side))
        }
    }
}

/// Similar to `md/launch_md`. Prepares GROMACS input on the calling thread, then
/// spawns a background thread for the blocking `gmx` execution. Results are
/// delivered via [`crate::threads::ThreadReceivers::gromacs_md_avail`] and
/// processed by [`on_gromacs_md_complete`].
pub fn launch_md(state: &mut State) {
    let (input, _mols) = match make_gromacs_input(state) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error creating GROMACS input: {e}");
            return;
        }
    };

    let mut offset = 0;
    let mut mol_start_indices = Vec::new();
    for m in &input.molecules {
        for _ in 0..m.count {
            mol_start_indices.push(offset);
            offset += m.atoms.len();
        }
    }

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let start = Instant::now();
        match input.run() {
            Ok(out) => {
                let elapsed = start.elapsed().as_millis();
                let _ = tx.send((out, mol_start_indices, elapsed));
            }
            Err(e) => {
                let elapsed = start.elapsed().as_millis();

                // Create an output with the error text, so we can display details in the UI.
                let out = GromacsOutput {
                    log_text: e.to_string(),
                    ..Default::default()
                };

                let _ = tx.send((out, mol_start_indices, elapsed));
            }
        }
    });

    state.volatile.thread_receivers.gromacs_md_avail = Some(rx);
}

fn extract_gromacs_fatal_error(log_text: &str) -> Option<String> {
    let mut lines = log_text.lines();

    while let Some(line) = lines.next() {
        if line.trim() == "Fatal error:" {
            let msg = lines
                .by_ref()
                .take_while(|line| !line.trim().is_empty())
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(" ");

            if msg.is_empty() {
                return None;
            }

            return Some(msg);
        }
    }

    None
}

/// Called by [`crate::threads::handle_thread_rx`] when the background GROMACS
/// thread completes. Updates state snapshots and notifies the UI.
pub fn on_gromacs_md_complete(
    state: &mut State,
    out: &GromacsOutput,
    _mol_start_indices: Vec<usize>,
    elapsed_ms: u128,
) {
    if out.log_text.contains("Fatal error") {
        let msg = extract_gromacs_fatal_error(&out.log_text)
            .map(|err| {
                format!("Problem running GROMACS: {err}. Check the logs or terminal for details.")
            })
            .unwrap_or_else(|| {
                "Problem running GROMACS; check the logs or terminal for details.".to_string()
            });

        state.ui.cmd_line_output = msg;
        state.ui.cmd_line_out_is_err = true;

        eprintln!("GROMACS error data: \n------\n{}\n------\n", out.log_text);
        return;
    }

    // Load the solvated input GRO as the mol set for trajectory playback.
    if let Some(gro_path) = &out.gro_path {
        if let Err(e) = state.volatile.md_local.viewer.load_gro(gro_path) {
            eprintln!("Error loading GRO after GROMACS run: {e}");
        }
    }

    // Load the TRR trajectory: register it and immediately show all frames.
    if let Some(trr_path) = &out.trr_path {
        match Trajectory::new(trr_path) {
            Ok(mut traj) => {
                match traj.load_snaps(FrameSlice::Index {
                    start: None,
                    end: None,
                }) {
                    Ok(snaps) => {
                        state.volatile.md_local.replace_snaps(snaps);
                        state.trajectories.push(traj);
                    }
                    Err(e) => eprintln!("Error loading TRR frames after GROMACS run: {e}"),
                }
            }
            Err(e) => eprintln!("Error opening TRR after GROMACS run: {e}"),
        }
    }

    state.volatile.md_local.draw_md_mols = true;

    handle_success(
        &mut state.ui,
        format!("GROMACS run complete in {elapsed_ms} ms"),
    );
}

/// Save .gro, .mdp, and .top files to disk, from our MD state, molecules etc.
pub fn save_input_files(state: &State, path: &Path) -> io::Result<()> {
    let (inp, _) = make_gromacs_input(state)?;
    inp.save(path)
}
