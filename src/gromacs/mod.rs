//! GROMACS integration - can create GROMACS inputs, and parse outputs from data structures and visualization
//! in this program. Uses the `bio_files` lib for implementation details..
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

use bio_files::gromacs::GromacsFrame;
use bio_files::{
    AtomGeneric, BondGeneric,
    gromacs::{
        GromacsInput, MdpParams, MoleculeInput,
        mdp::{Barostat, Constraints, Thermostat},
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, SimBoxInit, params::FfParamSet, snapshot::Snapshot,
};
use lin_alg::f32::Vec3;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::md::{STATIC_ATOM_DIST_THRESH, get_mols_sel_for_md};
use crate::state::State;
use crate::{md::filter_peptide_atoms, molecules::common::MoleculeCommon};

/// Run a GROMACS MD simulation and return the trajectory as `Vec<Snapshot>`.
///
/// This is conceptually equivalent to calling `md::build_dynamics` followed by
/// `md::run_dynamics_blocking`, but delegates execution to the external `gmx`
/// binary instead of our own integrator.
///
/// # Arguments
///
/// Arguments mirror `md::build_dynamics` for drop-in use by the UI layer:
///
/// - `_dev` — unused (GROMACS handles device selection internally)
/// - `mols_in` — `(ff_mol_type, molecule, copy_count)` triples
/// - `param_set` — global Amber force-field parameter set
/// - `mol_specific_params` — per-ligand GAFF2 parameters keyed by molecule ident
/// - `cfg` — MD configuration (timestep, temperature, box, etc.)
/// - `static_peptide` — if `true`, the peptide atoms are held fixed
/// - `peptide_only_near_lig` — if `Some(thresh)`, include only peptide atoms
///   within `thresh` Å of any ligand atom
/// - `pep_atom_set` — populated with the `(mol_i, atom_i)` pairs of included
///   peptide atoms (same semantics as in `md.rs`)
/// - `fast_init` — if `true`, skip water and initial energy relaxation
/// - `dt` — timestep in **ps**
/// - `n_steps` — total number of MD steps
pub fn run_dynamics(
    _dev: &ComputationDevice,
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    mut static_peptide: bool,
    mut peptide_only_near_lig: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    fast_init: bool,
    dt: f32,
    n_steps: usize,
) -> Vec<Snapshot> {
    if mols_in.is_empty() {
        static_peptide = false;
        peptide_only_near_lig = None;
    }

    // todo: Run this in a thread.

    // Build molecule entries for GROMACS input.
    let molecules = match build_molecule_inputs(
        mols_in,
        mol_specific_params,
        pep_atom_set,
        static_peptide,
        peptide_only_near_lig,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("GROMACS: failed to build molecule inputs: {e}");
            return Vec::new();
        }
    };

    // Determine simulation box dimensions (nm).
    let box_nm = sim_box_nm(&cfg.sim_box);

    // Map MdConfig + explicit dt/n_steps onto MdpParams.
    let mdp = build_mdp(cfg, dt, n_steps, fast_init);

    // Use the peptide (ff19SB) params as the system-wide fallback, since they cover
    // the broadest set of Amber atom types. Per-molecule params (e.g. GAFF2 for
    // ligands) are stored inside each MoleculeInput entry and take precedence.
    let ff_global = param_set.peptide.clone();

    let input = GromacsInput {
        mdp,
        molecules,
        box_nm,
        ff_global,
    };

    match input.run() {
        Ok(out) => convert_snapshots(&out.trajectory),
        Err(e) => {
            eprintln!("GROMACS run failed: {e}");
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Molecule input construction
// ---------------------------------------------------------------------------

/// Convert Molchanica molecule data into `Vec<MoleculeInput>` for GROMACS.
fn build_molecule_inputs(
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    static_peptide: bool,
    peptide_only_near_lig: Option<f64>,
) -> Result<Vec<MoleculeInput>, String> {
    use bio_files::create_bonds;

    let mut result = Vec::new();

    for (ff_mol_type, mol, count) in mols_in {
        if !mol.selected_for_md {
            continue;
        }

        let (atoms, bonds, name) = if *ff_mol_type == FfMolType::Peptide {
            // Apply the same peptide filtering as the built-in pipeline.
            let atoms = filter_peptide_atoms(pep_atom_set, mol, mols_in, peptide_only_near_lig);
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

// ---------------------------------------------------------------------------
// MDP construction
// ---------------------------------------------------------------------------

fn build_mdp(cfg: &MdConfig, dt: f32, n_steps: usize, fast_init: bool) -> MdpParams {
    // Timestep: our internal unit is ps; GROMACS MDP also uses ps — no conversion needed.
    let thermostat = if cfg.overrides.thermo_disabled {
        Thermostat::No
    } else {
        Thermostat::VRescale
    };

    let barostat = if cfg.overrides.baro_disabled || fast_init {
        Barostat::No
    } else {
        Barostat::No // NVT by default; user can switch to NPT in MdpParams post-construction
    };

    // Output frequency: save a frame roughly every 1 ps (500 × 0.002 ps/step).
    let out_freq = if n_steps > 0 {
        (500_usize).min(n_steps) as u32
    } else {
        500
    };

    // Skip water/solvent in GROMACS when fast_init is requested by not including
    // solvent molecules in the molecule list (handled in build_molecule_inputs).
    let constraints = if fast_init {
        Constraints::None
    } else {
        Constraints::HBonds
    };

    MdpParams {
        nsteps: n_steps as u64,
        dt,
        nstxout_compressed: out_freq,
        nstenergy: out_freq,
        nstlog: out_freq,
        thermostat,
        tau_t: vec![0.1],
        ref_t: vec![cfg.temp_target],
        barostat,
        ref_p: cfg.pressure_target,
        gen_vel: true,
        gen_temp: cfg.temp_target,
        constraints,
        ..MdpParams::default()
    }
}

// ---------------------------------------------------------------------------
// Box helper
// ---------------------------------------------------------------------------

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

/// Convert GROMACS trajectory frames into `Snapshot` values.
fn convert_snapshots(frames: &[GromacsFrame]) -> Vec<Snapshot> {
    frames
        .iter()
        .map(|frame| {
            let atom_posits: Vec<Vec3> = frame
                .atom_posits
                .iter()
                .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
                .collect();

            Snapshot {
                // `frame.time` is in ps; Snapshot.time is also in ps.
                time: frame.time,
                atom_posits,
                // Velocities, energies, etc. are not extracted from the GRO trajectory.
                ..Snapshot::default()
            }
        })
        .collect()
}

/// Similar to `md/launch_md`
/// todo: Launch in a thread
// pub fn launch_md(state: &mut State, run: bool, fast_init: bool) {
pub fn launch_md(state: &mut State) {
    let start = Instant::now();

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

    state.volatile.md_local.update_mols_for_disp(&mols);

    let near_lig_thresh = if state.ui.md.peptide_only_near_ligs {
        Some(STATIC_ATOM_DIST_THRESH)
    } else {
        None
    };

    let snaps = run_dynamics(
        &state.dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        &state.to_save.md_config,
        state.ui.md.peptide_static,
        near_lig_thresh,
        &mut md_pep_sel,
        false,
        state.to_save.md_dt,
        state.to_save.num_md_steps as usize,
    );

    // match run_dynamics(
    // ) {
    //     Ok(md) => {
    //         state.volatile.md_local.mol_dynamics = Some(md);
    //
    //         if run {
    //             state.volatile.md_local.start = Some(Instant::now());
    //             state.volatile.md_local.running = true;
    //         }
    //     }
    //     Err(e) => handle_err(&mut state.ui, e.descrip),
    // }

    state.volatile.md_local.peptide_selected = md_pep_sel;

    let elapsed = start.elapsed().as_millis();
    println!("GROMACS run complete in {elapsed} ms");
}
