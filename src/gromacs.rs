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
    collections::{BTreeMap, HashMap, HashSet},
    fs, io,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use bio_files::{
    AtomGeneric, BondGeneric, create_bonds, gromacs,
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
use mol_defs::molecules::{Atom, Bond, common::MoleculeCommon};
use na_seq::Element;

use crate::{
    md::{STATIC_ATOM_DIST_THRESH, add_copies, filter_peptide_atoms, get_mols_sel_for_md},
    state::{ComputationType, State},
    util::{format_elapsed, handle_err, handle_success},
};

const ANGSTROM_TO_NM: f64 = 0.10;

pub fn make_gromacs_input(
    mdp: MdpParams,
    _ff_mol_types: &[FfMolType],
    mols_input: Vec<MoleculeInput>,
    ff_param_set: &FfParamSet,
    sim_box_init: &SimBoxInit,
    solvent: &Solvent,
    minimize_energy: bool,
) -> io::Result<GromacsInput> {
    if ff_param_set.backend.system_potential().is_some() {
        return Err(io::Error::other(
            "GFN-FF requires Dynamics or xTB; GROMACS export is unavailable",
        ));
    }

    let ff_global = ForceFieldParams::default();

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
            topology_config: dynamics::gromacs::topology_config(ff_param_set, None),
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

/// Returns the input, and the solute molecules in GRO order, for playback.
pub fn gromacs_input_from_state(
    state: &State,
) -> io::Result<(GromacsInput, Vec<(FfMolType, MoleculeCommon)>)> {
    state.check_md_force_fields().map_err(io::Error::other)?;
    if state
        .selected_force_field()
        .backend
        .system_potential()
        .is_some()
    {
        return Err(io::Error::other(
            "GFN-FF requires Dynamics or xTB; GROMACS export is unavailable",
        ));
    }

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
    let selected_params = state.selected_force_field();
    let residues = crate::md::peptide_residue_map(state);

    // Build molecule entries for GROMACS input.
    let MoleculeInputs {
        inputs: mols_input,
        resolved_mols,
        viewer_mols,
        ..
    } = match build_molecule_inputs(
        &mols,
        &state.mol_specific_params,
        state.ui.md.peptide_static,
        near_lig_thresh,
        &cfg.sim_box,
        selected_params,
        &residues,
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
    let mut input = make_gromacs_input(
        mdp,
        &ff_mol_types,
        mols_input,
        selected_params,
        &cfg.sim_box,
        &cfg.solvent,
        minimize_energy,
    )?;
    input.topology_config =
        dynamics::gromacs::topology_config(selected_params, cfg.water_model.as_ref());
    {
        if cfg.ff_family != dynamics::ForceFieldFamily::Amber
            && matches!(cfg.solvent, Solvent::OctanolWithWater | Solvent::Custom(_))
        {
            return Err(io::Error::other(
                "Custom/octanol solvent requires parameters matching the selected force field",
            ));
        }
        input.topology_override = Some(
            dynamics::gromacs::make_topology(
                &resolved_mols,
                selected_params,
                cfg.water_model.as_ref(),
                input.solvent.is_some(),
            )
            .map_err(io::Error::other)?,
        );
    }

    Ok((input, viewer_mols))
}

pub struct MoleculeInputs {
    pub inputs: Vec<MoleculeInput>,
    pub pep_atom_set: HashSet<(usize, usize)>,
    pub resolved_mols: Vec<(String, MolDynamics, usize)>,
    /// One per GRO molecule, in GRO order.
    pub viewer_mols: Vec<(FfMolType, MoleculeCommon)>,
}

/// Convert Molchanica molecule data into `Vec<MoleculeInput>` for GROMACS.
pub fn build_molecule_inputs(
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    static_peptide: bool,
    near_lig_thresh: Option<f64>,
    sim_box_init: &SimBoxInit,
    param_set: &FfParamSet,
    peptide_residues: &HashMap<String, Vec<bio_files::ResidueGeneric>>,
) -> Result<MoleculeInputs, String> {
    let mut result = Vec::new();
    let mut placed_mols = Vec::new();
    let mut resolved_mols = Vec::new();
    let mut viewer_mols = Vec::new();

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
                    explicit_params: None,
                    residues: peptide_residues.get(&mol.ident).cloned(),
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
                    explicit_params: None,
                    residues: None,
                },
                name,
                HashSet::new(),
            )
        };

        let dyn_mol = param_set
            .parameterize(&dyn_mol)
            .map_err(|e| e.to_string())?;
        let first_name = copy_mol_name(&name, 0, count);
        resolved_mols.push((first_name.clone(), dyn_mol.clone(), 1));
        result.push(molecule_input_from_dyn(first_name, &dyn_mol));
        viewer_mols.push(viewer_mol(&mol.ident, &dyn_mol)?);
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
                resolved_mols.push((
                    copy_mol_name(&name, copy_i + 1, count),
                    placed_mol.clone(),
                    1,
                ));
                result.push(molecule_input_from_dyn(
                    copy_mol_name(&name, copy_i + 1, count),
                    placed_mol,
                ));
                viewer_mols.push(viewer_mol(&mol.ident, placed_mol)?);
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

    Ok(MoleculeInputs {
        inputs: result,
        pep_atom_set,
        resolved_mols,
        viewer_mols,
    })
}

/// Viewer molecule with the elements, bonds, and atom metadata GROMACS receives. Positions are
/// replaced from the GRO on playback.
fn viewer_mol(ident: &str, mol: &MolDynamics) -> Result<(FfMolType, MoleculeCommon), String> {
    let mut atoms: Vec<Atom> = mol.atoms.iter().map(Into::into).collect();
    if let Some(posits) = &mol.atom_posits {
        for (atom, posit) in atoms.iter_mut().zip(posits) {
            atom.posit = *posit;
        }
    }

    let bonds = mol
        .bonds
        .iter()
        .map(|b| Bond::from_generic(b, &atoms))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("Invalid bonds in {ident} for GROMACS playback"))?;

    Ok((
        mol.ff_mol_type,
        MoleculeCommon::new(ident.to_owned(), atoms, bonds, Default::default(), None),
    ))
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
struct PreparedRun {
    input: GromacsInput,
    mdp_text: String,
    index: Option<PathBuf>,
    viewer_mols: Vec<(FfMolType, MoleculeCommon)>,
}

fn prepare_run(state: &State) -> io::Result<PreparedRun> {
    let Some(crate::md::runs::RunInputs::Gromacs {
        coordinates,
        topology,
        mdp,
        mdp_text,
        index,
    }) = &state.volatile.md_local.opened_inputs
    else {
        let (input, viewer_mols) = gromacs_input_from_state(state)?;
        let mdp_text = input.make_mdp();

        return Ok(PreparedRun {
            input,
            mdp_text,
            index: None,
            viewer_mols,
        });
    };
    let prefs = &state.to_save.md;
    let cfg = &prefs.config;
    let updated = cfg.to_gromacs(prefs.num_steps as usize, prefs.dt);
    let original = crate::md::runs::gromacs_config(mdp);
    let original_mdp = mdp.clone();
    let mut mdp = mdp.clone();
    mdp.nsteps = updated.nsteps;
    mdp.dt = updated.dt;
    mdp.output_control = updated.output_control;
    if cfg.integrator != original.integrator {
        mdp.integrator = updated.integrator;
        mdp.thermostat = match cfg.integrator {
            dynamics::Integrator::Leapfrog { thermostat: None }
            | dynamics::Integrator::VerletVelocity { thermostat: None } => {
                gromacs::mdp::Thermostat::No
            }
            _ => updated.thermostat,
        };
        mdp.tau_t = updated.tau_t;
    }
    if cfg.temp_target != original.temp_target {
        mdp.ref_t.fill(cfg.temp_target);
        if mdp.ref_t.is_empty() {
            mdp.ref_t.push(cfg.temp_target);
        }
        mdp.gen_temp = cfg.temp_target;
    }
    if cfg.barostat_cfg != original.barostat_cfg {
        mdp.pcoupl = updated.pcoupl;
    }
    if cfg.hydrogen_constraint != original.hydrogen_constraint {
        mdp.constraints = updated.constraints;
    }
    if cfg.coulomb_cutoff != original.coulomb_cutoff {
        mdp.rcoulomb = updated.rcoulomb;
    }
    if cfg.lj_cutoff != original.lj_cutoff {
        mdp.rvdw = updated.rvdw;
    }
    let mdp_text = update_mdp_text(mdp_text, &original_mdp, &mdp);
    let input = GromacsInput {
        mdp,
        initial_gro: Some(fs::read_to_string(coordinates)?),
        topology_override: Some(read_topology(topology)?),
        minimize_energy: cfg.max_init_relaxation_iters.is_some(),
        skip_counterion_insertion: true,
        ..Default::default()
    };

    Ok(PreparedRun {
        input,
        mdp_text,
        index: index.clone(),
        viewer_mols: Vec::new(),
    })
}

fn update_mdp_text(text: &str, original: &MdpParams, updated: &MdpParams) -> String {
    fn assignment(line: &str) -> Option<(String, &str)> {
        let (key, value) = line.split(';').next()?.split_once('=')?;

        Some((
            key.trim().to_ascii_lowercase().replace('_', "-"),
            value.trim(),
        ))
    }

    let original = original.to_mdp_str();
    let updated = updated.to_mdp_str();
    let before: BTreeMap<_, _> = original.lines().filter_map(assignment).collect();
    let after: BTreeMap<_, _> = updated.lines().filter_map(assignment).collect();
    let mut changes: BTreeMap<_, _> = before
        .keys()
        .chain(after.keys())
        .filter_map(|key| {
            let new = after.get(key).copied();
            (before.get(key).copied() != new).then_some((key.clone(), new))
        })
        .collect();

    let mut result = String::new();
    let mut written = HashSet::new();
    for line in text.lines() {
        if let Some((key, _)) = assignment(line)
            && let Some(value) = changes.get(&key)
        {
            if written.insert(key.clone())
                && let Some(value) = value
            {
                result.push_str(&format!("{key} = {value}\n"));
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    changes.retain(|key, _| !written.contains(key));
    for (key, value) in changes {
        if let Some(value) = value {
            result.push_str(&format!("{key} = {value}\n"));
        }
    }

    result
}

fn read_topology(path: &Path) -> io::Result<String> {
    fn read(path: &Path, root: &Path, stack: &mut Vec<std::path::PathBuf>) -> io::Result<String> {
        if stack.len() >= 32 || stack.iter().any(|p| p == path) {
            return Err(io::Error::other("Recursive GROMACS topology include"));
        }
        stack.push(path.to_owned());
        let mut result = String::new();
        for line in fs::read_to_string(path)?.lines() {
            if let Some(include) = line
                .trim()
                .strip_prefix("#include")
                .and_then(|line| line.trim().strip_prefix('"'))
                .and_then(|line| line.split_once('"').map(|(file, _)| file))
                && let Ok(include) = path.with_file_name(include).canonicalize()
                && include.starts_with(root)
                && include.is_file()
            {
                result.push_str(&read(&include, root, stack)?);
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }
        stack.pop();

        Ok(result)
    }

    let path = path.canonicalize()?;
    read(&path, path.parent().unwrap(), &mut Vec::new())
}

pub fn launch_md(state: &mut State) {
    let PreparedRun {
        input,
        mdp_text,
        index,
        viewer_mols,
    } = match prepare_run(state) {
        Ok(v) => v,
        Err(e) => {
            handle_err(&mut state.ui, format!("Error creating GROMACS input: {e}"));
            return;
        }
    };

    let (tx, rx) = mpsc::channel();

    let computation = state.volatile.ongoing_computations.start(
        ComputationType::MolecularDynamics,
        "GROMACS",
        Some(format!("{} steps", state.to_save.md.num_steps)),
    );

    thread::spawn(move || {
        let start = Instant::now();
        let result = input.run_with_mdp(&mdp_text, index.as_deref());
        drop(computation);
        let elapsed = start.elapsed();

        match result {
            Ok(out) => {
                if let Some(gro) = &out.gro_path
                    && !viewer_mols.is_empty()
                    && let Some(dir) = crate::md::runs::gromacs_run_dir(gro)
                {
                    let mut viewer = crate::md::viewer::SnapshotViewer::default();
                    let saved = viewer
                        .load_gro_with_solute(gro, &viewer_mols)
                        .and_then(|()| {
                            crate::project::save_md_mol_set(
                                &dir.join(crate::project::MD_MOLECULE_SET_FILE),
                                &viewer.mol_sets[0],
                            )
                        });
                    if let Err(error) = saved {
                        eprintln!("Error saving GROMACS molecule set: {error}");
                    }
                }
                let _ = tx.send((out, elapsed));
            }
            Err(e) => {
                // handle_err(&mut state.ui, "Error setting up GROMACS.".to_string());

                eprintln!("\nError setting up GROMACS: \n{e:?}");

                // Create an output with the error text, so we can display details in the UI.
                let out = GromacsOutput {
                    log_text: e.to_string(),
                    setup_failure: true,
                    ..Default::default()
                };

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
    scene: &mut graphics::Scene,
    updates: &mut graphics::EngineUpdates,
    out: &GromacsOutput,
    elapsed: Duration,
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
        handle_err(
            &mut state.ui,
            format!(
                "GROMACS setup failed. {}",
                out.log_text.lines().next().unwrap_or_default()
            ),
        );
        return;
    }

    if let Err(error) = crate::md::runs::show_gromacs_output(state, scene, updates, out) {
        handle_err(
            &mut state.ui,
            format!("GROMACS finished. Could not open output: {error}"),
        );

        return;
    }

    handle_success(
        &mut state.ui,
        format!("GROMACS run complete in {}", format_elapsed(elapsed)),
    );
}

/// Save .gro, .mdp, and .top files to disk, from our MD state, molecules etc.
pub fn save_input_files(state: &State, path: &Path) -> io::Result<()> {
    let run = prepare_run(state)?;
    run.input
        .save_with_mdp(path, &run.mdp_text, run.index.as_deref())
}
