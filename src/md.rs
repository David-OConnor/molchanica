//! An interface to dynamics library.

use bio_files::{AtomGeneric, create_bonds, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdOverrides, MdState, MolDynamics, ParamError,
    compute_energy_snapshot, params::FfParamSet, snapshot::Snapshot,
};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::{Vec3, X_VEC, Y_VEC, Z_VEC};
use std::io::ErrorKind;
use std::{
    collections::{HashMap, HashSet},
    io,
    time::Instant,
};

use crate::{
    drawing::{draw_peptide, draw_water},
    molecules::{
        MoleculePeptide,
        common::MoleculeCommon,
        lipid::MoleculeLipid,
        nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType},
        small::MoleculeSmall,
    },
    state::State,
    util::{handle_err, handle_success},
};

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to be counted.
// Set this wide to take into account motion.
pub const STATIC_ATOM_DIST_THRESH: f64 = 14.;

// Run this many MD steps per frame. This is used to balance MD time with the rest of the application.
// A higher value will reduce the overall MD computation time,
// but make the UI laggier. If this is >= the number of steps, the whole MD operation will block
// the rest of the program. For initial test, a value above ~10 doesn't seem to
// noticeably increase total computation time. e.g. the frame time is small compared to this many
// MD steps for a small molecule + water sim.
const MD_STEPS_PER_APPLICATION_FRAME: usize = 10;

#[derive(Default)]
pub struct MdStateLocal {
    pub mol_dynamics: Option<MdState>,
    /// This flag lets us defer launch by a frame, so we can display a flag.
    pub launching: bool,
    pub running: bool,
    pub start: Option<Instant>,
    /// Cached so we don't compute each UI paint. Picoseconds.
    pub run_time: f32,
    /// We maintain a set of atom indices of peptides that are used in MD. This for example, might
    /// exclude hetero atoms and atoms not near a docking site. (mol i, atom i)
    pub peptide_selected: HashSet<(usize, usize)>,
    // todo: Consider if you want a dedicated MD playback operating mode. For now, this flag
    /// If true, render the molecules from this struct, instead of primary state molecules.
    pub draw_md_mols: bool,
    /// We maintain independent copies of these from the primary state. These are what we display
    /// after an MD run. They may be created as copies of moleclues in the primary state.
    /// (Mol, number of copies)
    /// For now, count is only set up on construction. We have a separate molecule instance for each count.
    pub peptides: Vec<MoleculePeptide>,
    pub mols_small: Vec<MoleculeSmall>,
    pub lipids: Vec<MoleculeLipid>,
    pub nucleic_acids: Vec<MoleculeNucleicAcid>,
}

impl MdStateLocal {
    /// Update the molecules stored here; we render these instead of the primary state molecules, if in an
    /// appropriate mode for viewing.
    pub fn update_mols_for_disp(&mut self, mols: &[(FfMolType, &MoleculeCommon, usize)]) {
        for (ff_mol_type, mol, count) in mols {
            // todo: Pass in the full mols so we are not reproducing a copy?
            match ff_mol_type {
                FfMolType::Peptide => {
                    for _ in 0..*count {
                        self.peptides.push(MoleculePeptide {
                            common: (*mol).clone(),
                            ..Default::default()
                        });
                    }
                }
                FfMolType::SmallOrganic => {
                    for _ in 0..*count {
                        self.mols_small.push(MoleculeSmall {
                            common: (*mol).clone(),
                            ..Default::default()
                        });
                    }
                }
                FfMolType::Dna | FfMolType::Rna => {
                    for _ in 0..*count {
                        self.nucleic_acids.push(MoleculeNucleicAcid {
                            common: (*mol).clone(),
                            ..Default::default()
                        });
                    }
                }
                FfMolType::Lipid => {
                    for _ in 0..*count {
                        self.lipids.push(MoleculeLipid {
                            common: (*mol).clone(),
                            ..Default::default()
                        });
                    }
                }
                _ => eprintln!("Invalid Mol type for MD"),
            }
        }
    }

    /// Set atom positions for molecules involve in dynamics to that of a snapshot. Ligs and lipids are only ones included
    /// in dynamics.
    pub fn change_snapshot(&mut self, snap_i: usize) -> io::Result<()> {
        let Some(md) = &self.mol_dynamics else {
            let txt = "Error: Attempting to change snapshot when there is no MD state";
            eprintln!("{txt}");
            return Err(io::Error::new(ErrorKind::InvalidData, txt));
        };

        let snap = &md.snapshots[snap_i];

        let posits_by_mol = snap.unflatten(&md.mol_start_indices)?;

        println!("Posits by mol len: {:?}", posits_by_mol.len()); // todo temp

        let mut i_posits = 0;

        // This assumes we pack each mol type in the same order.

        for mol in &mut self.peptides {
            // change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snap);
            for (i_p, p) in mol.common.atom_posits.iter_mut().enumerate() {
                *p = posits_by_mol[i_posits][i_p].0.into(); //Posit only; discard velocity.
            }
            i_posits += 1;
        }

        for mol in &mut self.mols_small {
            // change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snap);
            for (i_p, p) in mol.common.atom_posits.iter_mut().enumerate() {
                *p = posits_by_mol[i_posits][i_p].0.into();
            }
            i_posits += 1;
        }

        for mol in &mut self.lipids {
            // change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snap);
            for (i_p, p) in mol.common.atom_posits.iter_mut().enumerate() {
                *p = posits_by_mol[i_posits][i_p].0.into();
            }
            i_posits += 1;
        }

        for mol in &mut self.nucleic_acids {
            // change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snap);
            for (i_p, p) in mol.common.atom_posits.iter_mut().enumerate() {
                *p = posits_by_mol[i_posits][i_p].0.into();
            }
            i_posits += 1;
        }

        Ok(())
    }
}

/// For our non-blocking workflow. Run this once an MD run is complete.
pub fn post_run_cleanup(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    if state.volatile.md_local.mol_dynamics.is_none() {
        eprintln!("Can't run MD cleanup; MD state is None");
        return;
    }

    state.volatile.md_local.running = false;
    state.volatile.md_local.start = None;
    state.volatile.md_local.draw_md_mols = true;

    if let Some(p) = &state.peptide {
        // todo: Not using the helper as that's not iter_mut()
        let ligs: Vec<_> = state
            .ligands
            .iter()
            .filter(|l| l.common.selected_for_md)
            .collect();

        let lipids: Vec<_> = state
            .lipids
            .iter()
            .filter(|l| l.common.selected_for_md)
            .collect();

        let nucleic_acids: Vec<_> = state
            .nucleic_acids
            .iter()
            .filter(|l| l.common.selected_for_md)
            .collect();

        let md = state.volatile.md_local.mol_dynamics.as_mut().unwrap();
        reassign_snapshot_indices(
            p,
            &ligs,
            &lipids,
            &nucleic_acids,
            &mut md.snapshots,
            &state.volatile.md_local.peptide_selected,
        );
    }

    handle_success(&mut state.ui, "MD complete".to_string());

    // Tricky behavior here to prevent a dbl-borrow.
    let md = state.volatile.md_local.mol_dynamics.as_ref().unwrap();
    if !md.snapshots.is_empty() {
        let snap = &md.snapshots[0];

        draw_water(
            scene,
            &snap.water_o_posits,
            &snap.water_h0_posits,
            &snap.water_h1_posits,
            state.ui.visibility.hide_water,
        );

        // Put this back if we wish to re-generate the water template.
        // WaterInitTemplate::save(
        //     &md.water,
        //     (md.cell.bounds_low, md.cell.bounds_high),
        //     Path::new("water.water_init_template"),
        // )
        // .unwrap();
        // println!("\n Water init template saved.\n ")
    }

    draw_peptide(state, scene);
    // todo: Draw other mols too?
    // draw_mol();

    state.ui.current_snapshot = 0;

    updates.entities = EntityUpdate::All;
}

/// Filter out hetero atoms, and if necessary, atoms not close to a ligand.
pub fn filter_peptide_atoms(
    set: &mut HashSet<(usize, usize)>,
    pep: &MoleculeCommon,
    mols_non_pep: &[(FfMolType, &MoleculeCommon, usize)],
    only_near_lig: Option<f64>,
) -> Vec<AtomGeneric> {
    *set = HashSet::new();

    pep.atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let pass = if let Some(thresh) = only_near_lig {
                let mut closest_dist = f64::MAX;
                for lig in mols_non_pep {
                    for p in &lig.1.atom_posits {
                        let dist = (*p - pep.atom_posits[i]).magnitude();
                        if dist < closest_dist {
                            closest_dist = dist;
                        }
                    }
                }
                !a.hetero && closest_dist < thresh
            } else {
                !a.hetero
            };

            if pass {
                // The initial 0 is for the peptide mol number; we may support multiple
                // peptides in the future.
                set.insert((0, i));
                Some(a.to_generic())
            } else {
                None
            }
        })
        .collect()
}

fn add_copies(mols: &mut Vec<MolDynamics>, mol: &MolDynamics, copies: usize) {
    let offset_dirs = [X_VEC, -X_VEC, Y_VEC, -Y_VEC, Z_VEC, -Z_VEC];
    // todo: Dedicated fn for this?
    let offset_amt = 10.; // todo: Should depend on the mol.
    // todo: This produces atoms in a plus configuration; not a grid! Sloppy proxy..\
    let mut amt_i = 1; // Used to scale the distance.

    for i in 0..copies {
        let mut mol_extra = mol.clone();

        let offset = offset_dirs[i % 6] * offset_amt * (amt_i + 1) as f64;
        for atom in &mut mol_extra.atoms {
            atom.posit += offset;
        }
        // todo: Sort out how to do this position properly. Grid?

        // todo: Similar algorithm to how you set up water molecules; same idea.
        mols.push(mol_extra);

        if i != 0 && i.is_multiple_of(6) {
            amt_i += 1;
        }
    }
}

/// Set up MD for selected molecules. A general purpose wrapper around `dynamics` API, for
/// use in this application. Concerts molecules, atoms, and bonds into the bio_files format
/// `dynamics` expects as input.
///
/// Sets up peptide-specific settings A/R.
///
/// If `fast_init` is set, the water solvent, and relaxation are skipped.
pub fn build_dynamics(
    dev: &ComputationDevice,
    // Last item is number of copies.
    mols_in: &[(FfMolType, &MoleculeCommon, usize)],
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    mut static_peptide: bool,
    mut peptide_only_near_lig: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    fast_init: bool,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    if mols_in.is_empty() {
        static_peptide = false;
        peptide_only_near_lig = None;
    }

    let mols = setup_mols_dyn(
        mols_in,
        mol_specific_params,
        pep_atom_set,
        static_peptide,
        peptide_only_near_lig,
    )?;

    println!("Mols dyn:");
    for mol in &mols {
        println!("Mol: {}", mol.atoms.len());
    }

    // // Uncomment as required for validating individual processes.
    let mut cfg = MdConfig {
        overrides: MdOverrides {
            // skip_water: false,
            // skip_water_relaxation: false,
            // bonded_disabled: false,
            // coulomb_disabled: false,
            // lj_disabled: false,

            // todo temp
            long_range_recip_disabled: true,
            // thermo_disabled: false,
            // baro_disabled: false,
            // snapshots_during_equilibration: true,
            ..Default::default()
        },
        // max_init_relaxation_iters: None,
        ..cfg.clone()
    };

    if fast_init {
        cfg.overrides.skip_water = true;
        cfg.max_init_relaxation_iters = None;
    }

    println!("Initializing MD state...");
    let md_state = MdState::new(dev, &cfg, &mols, param_set)?;
    println!("MD init done.");

    Ok(md_state)
}

/// Run the dynamics in one go. Blocking.
pub fn run_dynamics_blocking(
    md_state: &mut MdState,
    dev: &ComputationDevice,
    dt: f32,
    n_steps: usize,
) {
    if n_steps == 0 {
        return;
    }

    let start = Instant::now();

    let i_20_pc = n_steps / 5;
    let mut disp_count = 0;
    for i in 0..n_steps {
        if i.is_multiple_of(i_20_pc) {
            println!("{}% Complete", disp_count * 20);
            disp_count += 1;
        }

        md_state.step(dev, dt, None);
    }

    let elapsed = start.elapsed();
    println!(
        "MD complete in {:.1} s",
        elapsed.as_millis() as f32 / 1_000.
    );
}

/// We filter peptide hetero atoms out of the MD workflow. Adjust snapshot indices and atom positions so they
/// are properly synchronized. This also handles the case of reassigning due to peptide atoms near the ligand.
pub fn reassign_snapshot_indices(
    pep: &MoleculePeptide,
    ligs: &[&MoleculeSmall],
    lipids: &[&MoleculeLipid],
    nucleic_acids: &[&MoleculeNucleicAcid],
    snapshots: &mut [Snapshot],
    pep_atom_set: &HashSet<(usize, usize)>,
) {
    if !pep.common.selected_for_md {
        return;
    }

    println!("Re-assigning snapshot indices to match atoms excluded for MD...");

    let pep_count = pep_atom_set.len();

    // Count how many ligand atoms precede the peptide in the snapshot ordering.
    let lig_atom_count: usize = ligs
        .iter()
        .filter(|l| l.common.selected_for_md)
        .map(|l| l.common.atoms.len())
        .sum();

    let lipid_atom_count: usize = lipids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .map(|l| l.common.atoms.len())
        .sum();

    let na_atom_count: usize = nucleic_acids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .map(|l| l.common.atoms.len())
        .sum();

    let pep_start_i = lig_atom_count + lipid_atom_count + na_atom_count;

    // Rebuild each snapshot's atom_posits: [ligands as-is] + [full peptide with holes filled]
    for snap in snapshots {
        if pep_start_i + pep_count > snap.atom_posits.len() {
            eprintln!(
                "Error: Invalid index when reassigning snapshot posits. \
            Snap atom count: {}, lig count {lig_atom_count} Pep start: {pep_start_i}, Pep count: {pep_count}",
                snap.atom_posits.len()
            );
            continue;
        }

        // Iterator over the peptide positions that actually participated in MD
        let mut pept_md_posits = snap.atom_posits[pep_start_i..pep_start_i + pep_count]
            .iter()
            .cloned();

        let mut pept_md_vels = snap.atom_velocities[pep_start_i..pep_start_i + pep_count]
            .iter()
            .cloned();

        let mut new_posits = Vec::with_capacity(pep_start_i + pep.common.atoms.len());
        let mut new_vels = Vec::with_capacity(pep_start_i + pep.common.atoms.len());

        // Keep ligand portion unchanged
        new_posits.extend_from_slice(&snap.atom_posits[..pep_start_i]);
        new_vels.extend_from_slice(&snap.atom_velocities[..pep_start_i]);

        // Reinsert peptide atoms in their original order
        for (i, atom) in pep.common.atoms.iter().enumerate() {
            let is_included = pep_atom_set.contains(&(0, i));

            if is_included {
                new_posits.push(
                    pept_md_posits
                        .next()
                        .expect("Ran out of peptide MD positions"),
                );
                new_vels.push(
                    pept_md_vels
                        .next()
                        .expect("Ran out of peptide MD velocities"),
                );
            } else {
                // Non-MD atom: use its original static position
                new_posits.push(atom.posit.into());
                new_vels.push(lin_alg::f32::Vec3::new_zero());
            }
        }

        // todo: This is screwing things up for purposes of saving DCD files without water, as it's combining
        // todo multiple mols into the posits.

        // Replace the snapshot's positions with the reindexed set
        snap.atom_posits = new_posits;
        snap.atom_velocities = new_vels;

        return;
    }

    println!("Done.");
}

// /// Unflattens.
// pub fn change_snapshot_helper(
//     posits: &mut [Vec3],
//     start_i_this_mol: &mut usize,
//     snapshot: &Snapshot,
// ) {
//     for (i_snap, posit) in snapshot.atom_posits.iter().enumerate() {
//         if i_snap < *start_i_this_mol || i_snap >= posits.len() + *start_i_this_mol {
//             continue;
//         }
//         posits[i_snap - *start_i_this_mol] = (*posit).into();
//     }
//
//     *start_i_this_mol += posits.len();
// }

impl State {
    /// Run MD for a single step if ready, and update atom positions immediately after. Blocks for
    /// a fixed number of steps only; intended to be run each frame until complete.
    pub fn md_step(&mut self, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
        if !self.volatile.md_local.running {
            return;
        }
        let Some(md) = &mut self.volatile.md_local.mol_dynamics else {
            return;
        };

        for _ in 0..MD_STEPS_PER_APPLICATION_FRAME {
            if md.step_count >= self.to_save.num_md_steps as usize {
                println!(
                    "\nMD computation time: {} \n Total run time: {} ms",
                    md.computation_time().unwrap(),
                    self.volatile.md_local.start.unwrap().elapsed().as_millis()
                );

                post_run_cleanup(self, scene, engine_updates);
                break;
            }
            md.step(&self.dev, self.to_save.md_dt, None);
        }
    }
}

/// Called directly from the UI; non-blocking if run = false. E.g., can launch here to load to state,
/// then call `md_step`.
pub fn launch_md(state: &mut State, run: bool, fast_init: bool) {
    // Filter molecules for docking by if they're selected.
    // mut so we can move their posits in the initial snapshot change.

    // Avoids borrow error.
    let mut md_pep_sel = state.volatile.md_local.peptide_selected.clone();

    // Clone selected molecules to release the immutable borrow of `state`
    // before the mutable borrow needed by update_mols_for_disp.
    let mols_owned: Vec<_> = {
        let mols = get_mols_sel_for_md(&state);
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

    match build_dynamics(
        &state.dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        &state.to_save.md_config,
        state.ui.md.peptide_static,
        near_lig_thresh,
        &mut md_pep_sel,
        fast_init,
    ) {
        Ok(md) => {
            state.volatile.md_local.mol_dynamics = Some(md);

            if run {
                state.volatile.md_local.start = Some(Instant::now());
                state.volatile.md_local.running = true;
            }
        }
        Err(e) => handle_err(&mut state.ui, e.descrip),
    }

    state.volatile.md_local.peptide_selected = md_pep_sel;
}

/// Called directly from the UI; computes energy.
pub fn launch_md_energy_computation(
    state: &State,
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<Snapshot, ParamError> {
    let mols_in = get_mols_sel_for_md(state);

    let peptide_only_near_lig = if state.ui.md.peptide_only_near_ligs {
        Some(STATIC_ATOM_DIST_THRESH)
    } else {
        None
    };

    let mols = setup_mols_dyn(
        &mols_in,
        &state.mol_specific_params,
        pep_atom_set,
        state.ui.md.peptide_static,
        peptide_only_near_lig,
    )?;

    compute_energy_snapshot(&state.dev, &mols, &state.ff_param_set)
}

/// A helper to reduce repetition. Loads references to mols in state that are selected for MD.
/// Doesn't convert to MolDynamics, but sets up in a way conducive to doing so.
fn get_mols_sel_for_md(state: &State) -> Vec<(FfMolType, &MoleculeCommon, usize)> {
    // todo: Quantities only currently apply to ligands.

    let mut res = Vec::new();

    if let Some(p) = &state.peptide
        && p.common.selected_for_md
    {
        res.push((FfMolType::Peptide, &p.common, 1));
    }

    let ligs: Vec<_> = state
        .ligands
        .iter()
        .filter(|l| l.common.selected_for_md)
        .collect();

    let lipids: Vec<_> = state
        .lipids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .collect();

    let nucleic_acids: Vec<_> = state
        .nucleic_acids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .collect();

    for m in &ligs {
        res.push((
            FfMolType::SmallOrganic,
            &m.common,
            state.to_save.num_md_copies,
        ));
    }

    for m in &lipids {
        res.push((FfMolType::Lipid, &m.common, 1));
    }

    for m in &nucleic_acids {
        let mol_type = match m.na_type {
            NucleicAcidType::Dna => FfMolType::Dna,
            NucleicAcidType::Rna => FfMolType::Rna,
        };
        res.push((mol_type, &m.common, 1));
    }

    res
}

fn setup_mols_dyn(
    mols: &[(FfMolType, &MoleculeCommon, usize)],
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    static_peptide: bool,
    peptide_only_near_lig: Option<f64>,
) -> Result<Vec<MolDynamics>, ParamError> {
    let mut res = Vec::new();

    for (ff_mol_type, mol, copies) in mols {
        if !mol.selected_for_md {
            continue;
        }

        // In the case of Peptides, we perform optional filtering, e.g. for only atoms near a ligand.
        // If so, we must rebuild bonds, as the indices they refer to will have changed, and some bonds
        // are no longer present.
        if *ff_mol_type == FfMolType::Peptide {
            // We assume hetero atoms are ligands, water etc, and are not part of the protein.
            // let atoms = filter_peptide_atoms(pep_atom_set, p, ligs, peptide_only_near_lig);
            let atoms = filter_peptide_atoms(pep_atom_set, mol, mols, peptide_only_near_lig);
            println!(
                "Peptide atom count: {}. Set count: {}",
                atoms.len(),
                pep_atom_set.len()
            );

            let bonds = create_bonds(&atoms);

            res.push(MolDynamics {
                ff_mol_type: FfMolType::Peptide,
                atoms,
                atom_posits: None,
                atom_init_velocities: None,
                bonds,
                adjacency_list: None,
                static_: static_peptide,
                bonded_only: false,
                mol_specific_params: None,
            });
            continue;
        }

        let atoms_gen: Vec<_> = mol.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = mol.bonds.iter().map(|b| b.to_generic()).collect();

        let msp = match mol_specific_params.get(&mol.ident) {
            Some(v) => Some(v.clone()),
            None => {
                if *ff_mol_type == FfMolType::SmallOrganic {
                    return Err(ParamError::new(&format!(
                        "Missing molecule-specific parameters for  {}",
                        mol.ident
                    )));
                }
                None
            }
        };

        let mol = MolDynamics {
            ff_mol_type: *ff_mol_type,
            atoms: atoms_gen,
            atom_posits: Some(mol.atom_posits.clone()),
            atom_init_velocities: None,
            bonds: bonds_gen,
            adjacency_list: Some(mol.adjacency_list.clone()),
            static_: false,
            bonded_only: false,
            mol_specific_params: msp,
        };

        if *copies > 1 && *ff_mol_type == FfMolType::SmallOrganic {
            add_copies(&mut res, &mol, *copies);
        } else {
            res.push(mol);
        }
    }

    Ok(res)
}
