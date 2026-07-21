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
    AtomGeneric, BondGeneric, FrameSlice, create_bonds, gromacs,
    gromacs::{
        GromacsInput, GromacsOutput, MdpParams, MoleculeInput,
        gro::{AtomGro, Gro},
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    FfMolType, MolDynamics, OCTANOL_WATER_TEMPLATE, SimBox, SimBoxInit, Solvent, make_octanol,
    params::FfParamSet,
};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3 as Vec3F64};
use na_seq::Element;

use crate::{
    md::{
        STATIC_ATOM_DIST_THRESH, add_copies, filter_peptide_atoms, get_mols_sel_for_md,
        trajectory::Trajectory,
    },
    molecules::common::MoleculeCommon,
    state::State,
    util::{handle_err, handle_success},
};

const ANGSTROM_TO_NM: f64 = 0.10;

pub fn make_gromacs_input(
    mdp: MdpParams,
    ff_mol_types: &[FfMolType],
    mols_input: Vec<MoleculeInput>,
    ff_param_set: &FfParamSet,
    sim_box_init: &SimBoxInit,
    solvent: &Solvent,
    minimize_energy: bool,
) -> io::Result<GromacsInput> {
    // Warning: We could possibly have FF name conflicts between sets, downstream of this merge.
    let mut ff_global = ForceFieldParams::default();

    if ff_mol_types.contains(&FfMolType::Peptide) {
        if let Some(ff) = &ff_param_set.peptide {
            ff_global = ff_global.merge_with(ff);
        }
    }

    if ff_mol_types.contains(&FfMolType::SmallOrganic) {
        if let Some(ff) = &ff_param_set.small_mol {
            ff_global = ff_global.merge_with(ff);
        }
    }

    if ff_mol_types.contains(&FfMolType::Lipid) {
        if let Some(ff) = &ff_param_set.lipids {
            ff_global = ff_global.merge_with(ff);
        }
    }

    if ff_mol_types.contains(&FfMolType::Dna) {
        if let Some(ff) = &ff_param_set.dna {
            ff_global = ff_global.merge_with(ff);
        }
    }

    if ff_mol_types.contains(&FfMolType::Rna) {
        if let Some(ff) = &ff_param_set.rna {
            ff_global = ff_global.merge_with(ff);
        }
    }

    // Determine simulation box dimensions (nm).
    let box_nm = sim_box_nm(&mols_input, sim_box_init);

    let solvent_gmx = match solvent {
        Solvent::None => None,
        Solvent::WaterOpc => Some(gromacs::solvate::Solvent::WaterOpc),
        // todo: Implement.
        Solvent::WaterOpcSpecifyMolCount(_) => None,
        Solvent::WaterOpcCustomRegions(regions) => {
            Some(gromacs::solvate::Solvent::WaterOpcCustomRegions(
                gromacs_regions_nm(regions, sim_box_init)?,
            ))
        }
        Solvent::OctanolWithWater => Some(octanol_with_water_solvent()),
        Solvent::Custom(_) => None,
        // todo: Implment this.
        // Solvent::Custom(_) => Some(gromacs::solvate::Solvent::Custom(WaterInitTemplate)),
    };

    let coordinate_origin_a =
        match (solvent, sim_box_init) {
            (Solvent::WaterOpcCustomRegions(_), SimBoxInit::Fixed((box_low, _))) => Some(
                Vec3F64::new(box_low.x as f64, box_low.y as f64, box_low.z as f64),
            ),
            _ => None,
        };

    Ok(
        GromacsInput {
            mdp,
            molecules: mols_input,
            box_nm,
            coordinate_origin_a,
            ff_global: Some(ff_global),
            solvent: solvent_gmx,
            initial_gro: None,
            extra_molecule_counts: Vec::new(),
            topology_override: None,
            mdrun_extra_args: Vec::new(),
            skip_counterion_insertion: false,
            minimize_energy,
        },
        // mols_owned,
    )
}

fn gromacs_regions_nm(
    regions: &[SimBox],
    sim_box_init: &SimBoxInit,
) -> io::Result<Vec<(Vec3F32, Vec3F32)>> {
    let SimBoxInit::Fixed((box_low, _)) = sim_box_init else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "GROMACS custom solvent regions require a fixed simulation box.",
        ));
    };

    Ok(regions
        .iter()
        .map(|region| {
            (
                (region.bounds_low - *box_low) * ANGSTROM_TO_NM as f32,
                (region.bounds_high - *box_low) * ANGSTROM_TO_NM as f32,
            )
        })
        .collect())
}

fn octanol_with_water_solvent() -> gromacs::solvate::Solvent {
    let octanol = make_octanol();

    gromacs::solvate::Solvent::Custom(gromacs::solvate::CustomSolventTemplate {
        gro_text: gromacs_octanol_with_water_template()
            .unwrap_or_else(|_| OCTANOL_WATER_TEMPLATE.to_string()),
        topology_molecules: vec![molecule_input_from_template("octan", &octanol)],
        include_opc_water: true,
    })
}

fn gromacs_octanol_with_water_template() -> io::Result<String> {
    const OPC_VS_A: f64 = 0.147803;

    let gro = Gro::new(OCTANOL_WATER_TEMPLATE)?;
    let mut atoms_out = Vec::with_capacity(gro.atoms.len() + gro.atoms.len() / 3);

    let mut i = 0;
    while i < gro.atoms.len() {
        let atom = &gro.atoms[i];
        if atom.mol_name == "SOL" {
            let Some(hw1) = gro.atoms.get(i + 1) else {
                return Err(io::Error::other(
                    "Incomplete SOL molecule in octanol/water template",
                ));
            };
            let Some(hw2) = gro.atoms.get(i + 2) else {
                return Err(io::Error::other(
                    "Incomplete SOL molecule in octanol/water template",
                ));
            };

            if atom.atom_type != "OW" || hw1.atom_type != "HW1" || hw2.atom_type != "HW2" {
                return Err(io::Error::other(
                    "Unexpected SOL atom order in octanol/water template",
                ));
            }

            atoms_out.push(atom.clone());
            atoms_out.push(hw1.clone());
            atoms_out.push(hw2.clone());

            let mw_posit = atom.posit
                + (hw1.posit - atom.posit) * OPC_VS_A
                + (hw2.posit - atom.posit) * OPC_VS_A;
            atoms_out.push(AtomGro {
                mol_id: atom.mol_id,
                mol_name: atom.mol_name.clone(),
                element: Element::Oxygen,
                atom_type: "MW".to_string(),
                serial_number: 0,
                posit: mw_posit,
                velocity: Some(Vec3F64::new_zero()),
            });

            i += 3;
        } else {
            atoms_out.push(atom.clone());
            i += 1;
        }
    }

    for (idx, atom) in atoms_out.iter_mut().enumerate() {
        atom.serial_number = idx as u32 + 1;
    }

    let gro_out = Gro {
        atoms: atoms_out,
        head_text: gro.head_text,
        box_vec: gro.box_vec,
    };

    let mut bytes = Vec::new();
    gro_out.write_to(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|e| io::Error::other(format!("Invalid UTF-8 writing GRO template: {e}")))
}

fn molecule_input_from_template(name: &str, mol: &MolDynamics) -> MoleculeInput {
    MoleculeInput {
        name: name.to_string(),
        atoms: mol.atoms.clone(),
        bonds: mol.bonds.clone(),
        ff_params: mol.mol_specific_params.clone(),
        count: 0,
        copy_atom_posits: None,
    }
}

pub fn gromacs_input_from_state(
    state: &State,
    // ) -> io::Result<(GromacsInput, Vec<(FfMolType, MoleculeCommon, usize)>)> {
) -> io::Result<GromacsInput> {
    // Filter molecules for docking by if they're selected.
    // mut so we can move their posits in the initial snapshot change.

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

    let cfg = &state.to_save.md.config;

    // Build molecule entries for GROMACS input.
    let (mols_input, _md_pep_sel) = match build_molecule_inputs(
        &mols,
        &state.mol_specific_params,
        state.ui.md.peptide_static,
        near_lig_thresh,
        &cfg.sim_box,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("GROMACS: failed to build molecule inputs: {e}");
            return Err(io::Error::new(io::ErrorKind::Other, e));
        }
    };

    // todo: you may need to do this; update state with the pep sel state.
    // state.volatile.md_local.peptide_selected = md_pep_sel;

    // Map MdConfig + explicit dt/n_steps onto MdpParams.
    let mdp = state
        .to_save
        .md
        .config
        .to_gromacs(state.to_save.md.num_steps as usize, state.to_save.md.dt);

    let minimize_energy = cfg.max_init_relaxation_iters.is_some();
    let ff_mol_types: Vec<_> = mols
        .iter()
        .map(|(ff_mol_type, _, _)| *ff_mol_type)
        .collect();
    make_gromacs_input(
        mdp,
        &ff_mol_types,
        mols_input,
        &state.ff_param_set,
        &cfg.sim_box,
        &cfg.solvent,
        minimize_energy,
    )
}

/// Convert Molchanica molecule data into `Vec<MoleculeInput>` for GROMACS.
pub fn build_molecule_inputs(
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    static_peptide: bool,
    near_lig_thresh: Option<f64>,
    sim_box_init: &SimBoxInit,
) -> Result<(Vec<MoleculeInput>, HashSet<(usize, usize)>), String> {
    let mut result = Vec::new();
    let mut placed_mols = Vec::new();

    let mut pep_atom_set = HashSet::new();
    let mut peptide_i = 0;
    let box_dims = match sim_box_init {
        SimBoxInit::Fixed((lo, hi)) => Some((hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)),
        SimBoxInit::Pad(_) => None,
    };

    for (ff_mol_type, mol, count) in mols_in {
        if mol.selected_for_md.is_none() {
            continue;
        }
        let count = (*count).max(1);

        // todo: Do we need to do anything with the pep atom set here?
        let (dyn_mol, name, pep_set) = if *ff_mol_type == FfMolType::Peptide {
            // Apply the same peptide filtering as the built-in pipeline.
            let (atoms, pep_set) = filter_peptide_atoms(mol, mols_in, near_lig_thresh);
            let bonds = create_bonds(&atoms);
            (
                MolDynamics {
                    ff_mol_type: FfMolType::Peptide,
                    atoms,
                    atom_posits: None,
                    atom_init_velocities: None,
                    bonds,
                    adjacency_list: None,
                    static_: static_peptide,
                    bonded_only: false,
                    mol_specific_params: None,
                },
                if peptide_i == 0 {
                    "PEP".to_string()
                } else {
                    format!("PEP{}", peptide_i + 1)
                },
                pep_set,
            )
        } else {
            let atoms: Vec<AtomGeneric> = mol.atoms.iter().map(|a| a.to_generic()).collect();
            let bonds: Vec<BondGeneric> = mol.bonds.iter().map(|b| b.to_generic()).collect();
            // Use the molecule ident as the topology name, truncated to 6 chars.
            let name = sanitise_mol_name(&mol.ident);
            let ff_params = mol_specific_params.get(&mol.ident).cloned();
            (
                MolDynamics {
                    ff_mol_type: *ff_mol_type,
                    atoms,
                    atom_posits: Some(mol.atom_posits.clone()),
                    atom_init_velocities: None,
                    bonds,
                    adjacency_list: Some(mol.adjacency_list.clone()),
                    static_: false,
                    bonded_only: false,
                    mol_specific_params: ff_params,
                },
                name,
                HashSet::new(),
            )
        };

        let first_name = copy_mol_name(&name, 0, count);
        result.push(molecule_input_from_dyn(first_name, &dyn_mol));
        placed_mols.push(dyn_mol.clone());

        if count > 1 {
            let before_extra = placed_mols.len();
            add_copies(&mut placed_mols, &dyn_mol, count - 1, box_dims);
            let placed_extra = placed_mols.len() - before_extra;
            if placed_extra != count - 1 {
                return Err(format!(
                    "Could only place {placed_extra} of {} extra copies for {}",
                    count - 1,
                    mol.ident
                ));
            }

            for (copy_i, placed_mol) in placed_mols[before_extra..].iter().enumerate() {
                result.push(molecule_input_from_dyn(
                    copy_mol_name(&name, copy_i + 1, count),
                    placed_mol,
                ));
            }
        }

        if *ff_mol_type == FfMolType::Peptide {
            pep_atom_set.extend(pep_set.into_iter().map(|(_, atom_i)| (peptide_i, atom_i)));
            peptide_i += 1;
        }

        // If the peptide is static, set all its positions fixed by noting them in
        // pep_atom_set (already done inside filter_peptide_atoms above). GROMACS
        // handles frozen groups via `freezegrps` / `freezedim` MDP options; that
        // extension is left for future work.
        let _ = static_peptide; // used via peptide MolDynamics construction
    }

    Ok((result, pep_atom_set))
}

fn copy_mol_name(base: &str, copy_i: usize, copy_count: usize) -> String {
    if copy_count <= 1 {
        base.to_string()
    } else {
        format!("{base}_{}", copy_i + 1)
    }
}

fn molecule_input_from_dyn(name: String, mol: &MolDynamics) -> MoleculeInput {
    let mut atoms = mol.atoms.clone();
    if let Some(posits) = &mol.atom_posits {
        for (atom, posit) in atoms.iter_mut().zip(posits) {
            atom.posit = *posit;
        }
    }

    MoleculeInput {
        name,
        atoms,
        bonds: mol.bonds.clone(),
        ff_params: mol.mol_specific_params.clone(),
        count: 1,
        copy_atom_posits: None,
    }
}

/// Build one GROMACS molecule topology with explicit coordinates for each packed copy.
///
/// GROMACS topology counts and coordinate-file copies are separate concerns. Keeping
/// them separate avoids turning a packed collection into one disconnected molecule.
pub(crate) fn molecule_input_from_packed_copies(
    name: String,
    placed_mols: &[MolDynamics],
) -> io::Result<MoleculeInput> {
    let Some(template) = placed_mols.first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cannot build a packed GROMACS molecule input without any copies.",
        ));
    };

    let atom_count = template.atoms.len();
    let bond_count = template.bonds.len();
    let mut copy_atom_posits = Vec::with_capacity(placed_mols.len());

    for (copy_i, mol) in placed_mols.iter().enumerate() {
        if mol.atoms.len() != atom_count || mol.bonds.len() != bond_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Packed GROMACS molecule copy {} does not match the template topology.",
                    copy_i + 1,
                ),
            ));
        }

        let posits = mol
            .atom_posits
            .clone()
            .unwrap_or_else(|| mol.atoms.iter().map(|atom| atom.posit).collect());
        if posits.len() != atom_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Packed GROMACS molecule copy {} has {} positions for {} atoms.",
                    copy_i + 1,
                    posits.len(),
                    atom_count,
                ),
            ));
        }
        copy_atom_posits.push(posits);
    }

    let mut atoms = template.atoms.clone();
    for (atom, posit) in atoms.iter_mut().zip(&copy_atom_posits[0]) {
        atom.posit = *posit;
    }

    Ok(MoleculeInput {
        name,
        atoms,
        bonds: template.bonds.clone(),
        ff_params: template.mol_specific_params.clone(),
        count: placed_mols.len(),
        copy_atom_posits: Some(copy_atom_posits),
    })
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

/// This is a copy+paste of SimBox::from_atoms in dynamics, but suited to the input
/// and output types and units we use here.
pub fn sim_box_nm(mols: &[MoleculeInput], box_type: &SimBoxInit) -> Option<(f64, f64, f64)> {
    let (bounds_low, bounds_high) = match box_type {
        SimBoxInit::Pad(pad) => {
            let pad = *pad as f64;
            let (mut min, mut max) = (
                Vec3F64::splat(f64::INFINITY),
                Vec3F64::splat(f64::NEG_INFINITY),
            );

            let mut atom_count = 0;
            for mol in mols {
                if let Some(copy_atom_posits) = &mol.copy_atom_posits {
                    for posits in copy_atom_posits {
                        for posit in posits {
                            min = min.min(*posit);
                            max = max.max(*posit);
                            atom_count += 1;
                        }
                    }
                } else if mol.count > 0 {
                    for atom in &mol.atoms {
                        min = min.min(atom.posit.into());
                        max = max.max(atom.posit.into());
                        atom_count += 1;
                    }
                }
            }

            if atom_count == 0 {
                return None;
            }

            let bounds_low = min - Vec3F64::splat(pad);
            let bounds_high = max + Vec3F64::splat(pad);
            (bounds_low, bounds_high)
        }
        SimBoxInit::Fixed((bounds_low, bounds_high)) => {
            let l: Vec3F64 = (*bounds_low).into();
            let h: Vec3F64 = (*bounds_high).into();
            (l, h)
        }
    };

    let v = (bounds_high - bounds_low) * ANGSTROM_TO_NM;
    Some((v.x, v.y, v.z))
}

/// Similar to `md/launch_md`. Prepares GROMACS input on the calling thread, then
/// spawns a background thread for the blocking `gmx` execution. Results are
/// delivered via [`crate::threads::ThreadReceivers::gromacs_md_avail`] and
/// processed by [`on_gromacs_md_complete`].
pub fn launch_md(state: &mut State) {
    let input = match gromacs_input_from_state(state) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error creating GROMACS input: {e}");
            return;
        }
    };

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let start = Instant::now();
        match input.run() {
            Ok(out) => {
                let elapsed = start.elapsed().as_millis();
                // let _ = tx.send((out, mol_start_indices, elapsed));
                let _ = tx.send((out, elapsed));
            }
            Err(e) => {
                let elapsed = start.elapsed().as_millis();

                // handle_err(&mut state.ui, "Error setting up GROMACS.".to_string());

                eprintln!("\nError setting up GROMACS: \n{e:?}");

                // Create an output with the error text, so we can display details in the UI.
                let out = GromacsOutput {
                    log_text: e.to_string(),
                    setup_failure: true,
                    ..Default::default()
                };

                // let _ = tx.send((out, mol_start_indices, elapsed));
                let _ = tx.send((out, elapsed));
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
    // _mol_start_indices: Vec<usize>,
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

    if out.setup_failure {
        handle_err(&mut state.ui, "GROMACS setup failure error".to_string());
        return;
    }

    // Load the solvated input GRO as the mol set for trajectory playback.
    if let Some(gro_path) = &out.gro_path {
        if let Err(e) = state.volatile.md_local.viewer.load_gro(gro_path) {
            eprintln!("Error loading GRO after GROMACS run: {e}");
        }
    }

    for traj in &mut state.trajectories {
        traj.ui_active = false;
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
    let inp = gromacs_input_from_state(state)?;
    inp.save(path)
}
