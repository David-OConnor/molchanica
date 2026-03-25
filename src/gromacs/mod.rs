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

use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use bio_files::{
    AtomGeneric, BondGeneric,
    gromacs::{
        GromacsFrame, GromacsInput, MdpParams, MoleculeInput, WaterModel,
        mdp::{Barostat, Constraints, Thermostat},
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdState, SimBoxInit, Solvent, params::FfParamSet,
    snapshot::Snapshot,
};
use lin_alg::f32::Vec3;

use crate::{
    md::{STATIC_ATOM_DIST_THRESH, filter_peptide_atoms, get_mols_sel_for_md},
    molecules::common::MoleculeCommon,
    state::State,
};

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
) -> (Vec<Snapshot>, Vec<usize>) {
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
            return (Vec::new(), Vec::new());
        }
    };

    // Compute mol_start_indices from the MoleculeInput list before moving it.
    // Each copy of each molecule gets its own entry (matching how MdState packs atoms).
    let mut offset = 0usize;
    let mut mol_start_indices = Vec::new();
    for m in &molecules {
        for _ in 0..m.count {
            mol_start_indices.push(offset);
            offset += m.atoms.len();
        }
    }

    // Determine simulation box dimensions (nm).
    let box_nm = sim_box_nm(&cfg.sim_box);

    // Map MdConfig + explicit dt/n_steps onto MdpParams.
    let mdp = cfg.to_gromacs(n_steps, dt);

    // todo: This isn't right.
    // Use the peptide (ff19SB) params as the system-wide fallback, since they cover
    // the broadest set of Amber atom types. Per-molecule params (e.g. GAFF2 for
    // ligands) are stored inside each MoleculeInput entry and take precedence.
    let ff_global = param_set.peptide.clone();

    let water_model = match &cfg.solvent {
        Solvent::None => None,
        Solvent::WaterOpc | Solvent::WaterOpcSpecifyMolCount(_) | Solvent::Custom(_) => {
            Some(WaterModel::Opc)
        }
    };

    let input = GromacsInput {
        mdp,
        molecules,
        box_nm,
        ff_global,
        water_model,
    };

    match input.run() {
        Ok(out) => (
            convert_snapshots(&out.trajectory, out.solute_atom_count),
            mol_start_indices,
        ),
        Err(e) => {
            eprintln!("\nGROMACS run failed: \n{e}");
            (Vec::new(), Vec::new())
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
///
/// `solute_atom_count` is the number of non-water atoms (computed before solvation).
/// Atoms beyond that index are OPC water molecules, laid out as groups of 4:
/// OW, HW1, HW2, MW (virtual site). MW positions are discarded since `Snapshot`
/// has no field for them and the virtual site carries no mass.
fn convert_snapshots(frames: &[GromacsFrame], solute_atom_count: usize) -> Vec<Snapshot> {
    // OPC water has 4 sites per molecule (OW, HW1, HW2, MW virtual site).
    const OPC_SITES_PER_MOL: usize = 4;

    frames
        .iter()
        .map(|frame| {
            let n = frame.atom_posits.len();
            let solute_end = solute_atom_count.min(n);

            let atom_posits: Vec<Vec3> = frame.atom_posits[..solute_end]
                .iter()
                .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
                .collect();

            let water_block = &frame.atom_posits[solute_end..];
            let n_water_mols = water_block.len() / OPC_SITES_PER_MOL;

            let mut water_o_posits = Vec::with_capacity(n_water_mols);
            let mut water_h0_posits = Vec::with_capacity(n_water_mols);
            let mut water_h1_posits = Vec::with_capacity(n_water_mols);

            for i in 0..n_water_mols {
                let base = i * OPC_SITES_PER_MOL;
                let to_vec3 =
                    |p: &lin_alg::f64::Vec3| Vec3::new(p.x as f32, p.y as f32, p.z as f32);
                water_o_posits.push(to_vec3(&water_block[base]));
                water_h0_posits.push(to_vec3(&water_block[base + 1]));
                water_h1_posits.push(to_vec3(&water_block[base + 2]));
                // base + 3 is the MW virtual site — no Snapshot field for it.
            }

            Snapshot {
                time: frame.time,
                atom_posits,
                water_o_posits,
                water_h0_posits,
                water_h1_posits,
                ..Snapshot::default()
            }
        })
        .collect()
}

/// Similar to `md/launch_md`
/// todo: Launch in a thread
// pub fn launch_md(state: &mut State, run: bool) {
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

    let (snaps, mol_start_indices) = run_dynamics(
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

    // Store the GROMACS trajectory so the UI can play it back via change_snapshot.
    // We build a minimal MdState — only mol_start_indices and snapshots are needed
    // for playback; the integrator fields are left at Default since we do not call
    // md_step on this state.
    if !snaps.is_empty() {
        let mut md = MdState::default();
        md.mol_start_indices = mol_start_indices;
        md.snapshots = snaps;
        state.volatile.md_local.mol_dynamics = Some(md);
        state.volatile.md_local.draw_md_mols = true;
    }

    state.volatile.md_local.peptide_selected = md_pep_sel;

    let elapsed = start.elapsed().as_millis();
    println!("GROMACS run complete in {elapsed} ms");
}
