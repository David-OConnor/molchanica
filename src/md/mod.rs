//! An interface to dynamics library.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    fmt::{Display, Formatter},
    path::Path,
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, bond_inference, create_bonds, gromacs::GromacsOutput, md_params::ForceFieldParams,
};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdOverrides, MdState, MolDynamics, ParamError,
    SimBoxInit, Solvent, compute_energy_snapshot, params::FfParamSet, snapshot::Snapshot,
};
use graphics::{EngineUpdates, Entity, FWD_VEC, Scene};
use lin_alg::f64::{Quaternion, Vec3};
use rand::Rng;
use viewer::SnapshotViewer;

use crate::{
    cam::{move_cam_to_active_mol, reset_camera},
    file_io::save_mol_set_as_gro,
    gromacs,
    md::trajectory::{TrajFormat, Trajectory},
    molecules::{Atom, Bond, common::MoleculeCommon, nucleic_acid::NucleicAcidType},
    state::State,
    util::{RedrawFlags, clear_cli_out, handle_err, handle_success},
};

pub mod trajectory;
pub mod viewer;

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

// pub type MolsForMd = (FfMolType, &MoleculeCommon, usize);

#[derive(Default)]
pub struct MdStateLocal {
    /// MD state, for when running using the Dynamics engine.
    pub mol_dynamics: Option<MdState>,
    /// This flag lets us defer launch by a frame, so we can display a flag.
    pub launching: bool,
    pub running: bool,
    pub start: Option<Instant>,
    /// Cached so we don't compute each UI paint. Picoseconds.
    pub run_time: f32,
    // todo: Consider if you want a dedicated MD playback operating mode. For now, this flag
    /// If true, render the molecules from this struct, instead of primary state molecules.
    pub draw_md_mols: bool,
    pub gromacs_output: Option<GromacsOutput>,
    pub viewer: SnapshotViewer,
    // todo: putting back for now
    /// We maintain a set of atom indices of peptides that are used in MD. This for example, might
    /// exclude hetero atoms and atoms not near a docking site. (mol i, atom i)
    /// todo: When you support multiple peptides, you will have to updates this. Consider moving it
    /// to the individual peptide.
    pub pep_atom_set: HashSet<(usize, usize)>,
}

impl MdStateLocal {
    /// todo: WIP, regarding how we pass in atoms and molecules.
    pub fn replace_snaps(&mut self, snaps: Vec<Snapshot>) {
        self.draw_md_mols = true;

        self.viewer.snapshots = snaps;

        if !self.viewer.snapshots.is_empty() {
            self.viewer.current_snapshot = Some(0);
            if let Err(e) = self.viewer.change_snapshot(0) {
                eprintln!("Error changing snapshots when replacing: {e:?}");
            }
        }
    }

    /// Removes all MD snapshots, and performs related cleanup.
    pub fn clear_snaps(&mut self, ents: &mut Vec<Entity>, redraw: &mut RedrawFlags) {
        self.viewer.snapshots = Vec::new();
        self.viewer.current_snapshot = None;
        self.viewer.mol_set_active = None;
        self.draw_md_mols = false;

        ents.clear();
        redraw.set_all();
    }
}

/// For our non-blocking workflow. Run this once an MD run using the `Dynamics` engine is complete.
pub fn post_run_cleanup(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let md = &mut state.volatile.md_local;
    if md.mol_dynamics.is_none() {
        // if md.mol_dynamics.is_none() {
        eprintln!("Can't run MD cleanup; MD state is None");
        return;
    };

    md.running = false;
    md.start = None;
    md.draw_md_mols = true;

    // Copy snapshots from MD state to the viewer.
    let snaps = md.mol_dynamics.as_ref().unwrap().snapshots.clone();
    md.viewer.snapshots = snaps.clone();

    for traj in &mut state.trajectories {
        traj.ui_active = false;
    }

    // Register an in-memory Trajectory so the run appears in the trajectory
    // sidebar and water molecules are visible in the mol-set list.
    let run_n = state
        .trajectories
        .iter()
        .filter(|t| t.format != TrajFormat::InMemory)
        .count();

    state.trajectories.push(Trajectory::new_in_memory(
        snaps,
        format!("In-memory run {}", run_n + 1),
        state.to_save.md_dt,
    ));

    // Auto-save the mol set as a GRO file alongside the trajectory files.
    let run_index = state
        .volatile
        .md_local
        .mol_dynamics
        .as_ref()
        .and_then(|md| md.run_index)
        .unwrap_or(0);

    let gro_path = Path::new("./md_out").join(format!("traj_{run_index}.gro"));
    // The mol set we just added is the last one in the viewer.
    if let Some(mol_set) = state.volatile.md_local.viewer.mol_sets.last() {
        if let Err(e) = save_mol_set_as_gro(mol_set, &gro_path) {
            eprintln!("Error auto-saving GRO: {e:?}");
        }
    }

    if state.volatile.md_local.viewer.change_snapshot(0).is_err() {
        handle_err(
            &mut state.ui,
            String::from("Error changing snapshot at MD completion."),
        );
        return;
    }

    if !state.volatile.md_local.viewer.mol_sets.is_empty() {
        state.volatile.md_local.viewer.mol_set_active =
            Some(state.volatile.md_local.viewer.mol_sets.len() - 1);
    }

    reset_camera(state, scene, updates, FWD_VEC);
    viewer::draw_mols(state, scene, updates);
    handle_success(&mut state.ui, "MD complete".to_string());
}

/// Filter out hetero atoms, and if necessary, atoms not close to a ligand.
pub fn filter_peptide_atoms(
    pep: &MoleculeCommon,
    mols_non_pep: &[(FfMolType, &MoleculeCommon, usize)],
    near_lig_thresh: Option<f64>,
) -> (Vec<AtomGeneric>, HashSet<(usize, usize)>) {
    let mut set = HashSet::new();

    let atoms = pep
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let pass = if let Some(thresh) = near_lig_thresh {
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
        .collect();

    (atoms, set)
}

fn mol_centroid(m: &MolDynamics) -> Vec3 {
    let posits: Vec<Vec3> = if let Some(ap) = &m.atom_posits {
        ap.clone()
    } else {
        m.atoms.iter().map(|a| a.posit).collect()
    };
    let n = posits.len().max(1) as f64;
    posits.iter().fold(Vec3::new_zero(), |acc, p| acc + *p) * (1.0 / n)
}

fn mol_bounding_radius(m: &MolDynamics, center: Vec3) -> f64 {
    let posits: Vec<Vec3> = if let Some(ap) = &m.atom_posits {
        ap.clone()
    } else {
        m.atoms.iter().map(|a| a.posit).collect()
    };
    posits
        .iter()
        .map(|p| (*p - center).magnitude())
        .fold(0.0_f64, f64::max)
}

/// Add multiple copies of each molecule with random orientations.
///
/// For fixed boxes (`box_dims = Some`): uses a regular cubic grid so every copy gets a
/// distinct cell — no origin fallback, no piling.  At each grid position we try
/// `MAX_ROT_ATTEMPTS` random rotations and keep whichever gives the largest minimum
/// atom-to-atom distance to already-placed atoms.  This correctly handles elongated
/// molecules (e.g. octanol) that can pack far more tightly than their bounding sphere
/// radius implies.  If no rotation achieves the preferred 1.5 Å clearance the best
/// available rotation is used and the energy minimiser resolves any remaining soft
/// overlaps.
///
/// For unbounded (`box_dims = None`): falls back to the original shell-expansion strategy
/// (bounding-sphere exclusion), which is fine for dilute / Pad-box placements.
fn add_copies(
    mols: &mut Vec<MolDynamics>,
    mol: &MolDynamics,
    copies: usize,
    box_dims: Option<(f32, f32, f32)>,
) {
    // Preferred minimum atom-to-atom distance between copies.
    // Soft overlaps at this scale are resolved cleanly by the energy minimiser.
    const MIN_ATOM_DIST_SQ: f64 = 1.5 * 1.5;
    // Keep atom positions this far from the box walls.
    const WALL_MARGIN: f64 = 0.5;
    // Rotation candidates tried per grid cell.
    const MAX_ROT_ATTEMPTS: usize = 200;
    // Position candidates for the unbounded path.
    const MAX_POS_ATTEMPTS: usize = 500;

    if copies > 1 {
        println!("Adding {copies} molecule copies...");
    }
    let start = Instant::now();
    let mut rng = rand::rng();

    let orig_centroid = mol_centroid(mol);
    let orig_radius = mol_bounding_radius(mol, orig_centroid);

    // Atom positions relative to centroid — rotated cheaply for every candidate.
    let orig_local: Vec<Vec3> = {
        let posits: Vec<Vec3> = if let Some(ap) = &mol.atom_posits {
            ap.clone()
        } else {
            mol.atoms.iter().map(|a| a.posit).collect()
        };
        posits.iter().map(|&p| p - orig_centroid).collect()
    };

    // All placed atom positions — grows as copies are committed.
    let mut placed_atoms: Vec<Vec3> = mols
        .iter()
        .flat_map(|m| {
            if let Some(ap) = &m.atom_posits {
                ap.clone()
            } else {
                m.atoms.iter().map(|a| a.posit).collect::<Vec<_>>()
            }
        })
        .collect();

    // Spatial filter: placed atoms farther than this from the candidate centroid
    // cannot possibly clash with any new atom.
    let search_sq = (orig_radius * 2.0 + 2.0).powi(2);

    // Inline helper: apply rotation + translation to a mol copy.
    let apply_rot_trans = |mol_copy: &mut MolDynamics, rot: Quaternion, translation: Vec3| {
        if let Some(posits) = &mut mol_copy.atom_posits {
            for p in posits.iter_mut() {
                let local = *p - orig_centroid;
                *p = rot.rotate_vec(local) + orig_centroid + translation;
            }
        }
        for atom in &mut mol_copy.atoms {
            let local = atom.posit - orig_centroid;
            atom.posit = rot.rotate_vec(local) + orig_centroid + translation;
        }
    };

    if let Some((bx, by, bz)) = box_dims {
        // ── Grid placement ────────────────────────────────────────────────────
        // Build a cubic grid with n³ ≥ copies cells. Use at least n=3 (27 cells)
        // so that even copies=1 has 27 candidates to choose from — preventing the
        // solute from being forced onto the same (0,0,0) cell as an existing molecule.
        let n = (copies as f64).cbrt().ceil() as usize;
        let n = n.max(3);
        let (sx, sy, sz) = (
            bx as f64 / n as f64,
            by as f64 / n as f64,
            bz as f64 / n as f64,
        );

        let mut grid: Vec<Vec3> = (0..n)
            .flat_map(|ix| {
                (0..n).flat_map(move |iy| {
                    (0..n).map(move |iz| {
                        Vec3::new(
                            -bx as f64 / 2.0 + (ix as f64 + 0.5) * sx,
                            -by as f64 / 2.0 + (iy as f64 + 0.5) * sy,
                            -bz as f64 / 2.0 + (iz as f64 + 0.5) * sz,
                        )
                    })
                })
            })
            .collect();

        let (hx, hy, hz) = (
            bx as f64 / 2.0 - WALL_MARGIN,
            by as f64 / 2.0 - WALL_MARGIN,
            bz as f64 / 2.0 - WALL_MARGIN,
        );

        // Greedy cell selection: for each copy, pick the available grid cell whose
        // centroid is furthest from all already-placed atom positions.  This ensures
        // copies=1 (or any small count) doesn't collide with molecules from a prior
        // add_copies call that happened to occupy the same grid center.
        for copy_i in 0..copies {
            // Score remaining grid cells by min-distance of centroid to placed atoms.
            let best_cell_idx = if placed_atoms.is_empty() {
                // Place first molecule at the cell closest to the box centre (origin).
                grid.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.magnitude_squared()
                            .partial_cmp(&b.magnitude_squared())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                grid.iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        let min_dsq = placed_atoms
                            .iter()
                            .map(|&p| (c - p).magnitude_squared())
                            .fold(f64::MAX, f64::min);
                        (i, min_dsq)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            let centroid = grid.remove(best_cell_idx);
            let mut best_rot = Quaternion::new(1., 0., 0., 0.);
            let mut best_posits: Vec<Vec3> = Vec::new();
            let mut best_min_sq = f64::NEG_INFINITY;

            for _ in 0..MAX_ROT_ATTEMPTS {
                let rot = {
                    let (w, x, y, z): (f64, f64, f64, f64) =
                        (rng.random(), rng.random(), rng.random(), rng.random());
                    Quaternion::new(w, x, y, z).to_normalized()
                };

                let new_posits: Vec<Vec3> = orig_local
                    .iter()
                    .map(|&local| rot.rotate_vec(local) + centroid)
                    .collect();

                // Wall check: all atoms must stay inside the box.
                if !new_posits
                    .iter()
                    .all(|p| p.x.abs() <= hx && p.y.abs() <= hy && p.z.abs() <= hz)
                {
                    continue;
                }

                // Atom-level clash check against all previously placed atoms.
                let mut min_sq = f64::MAX;
                'check: for &np in &new_posits {
                    for &pp in &placed_atoms {
                        // Skip placed atoms that are too far away to clash.
                        if (pp - centroid).magnitude_squared() > search_sq {
                            continue;
                        }
                        let dsq = (np - pp).magnitude_squared();
                        if dsq < min_sq {
                            min_sq = dsq;
                            if min_sq < MIN_ATOM_DIST_SQ {
                                break 'check; // Already worse than best; try next rotation.
                            }
                        }
                    }
                }

                if min_sq > best_min_sq {
                    best_min_sq = min_sq;
                    best_rot = rot;
                    best_posits = new_posits;
                }

                if best_min_sq >= MIN_ATOM_DIST_SQ {
                    break; // Clean placement found.
                }
            }

            if best_min_sq < MIN_ATOM_DIST_SQ {
                eprintln!(
                    "add_copies: copy {copy_i}: best min atom dist {:.2} Å at \
                     ({:.1},{:.1},{:.1}) — placing; energy minimiser will resolve.",
                    best_min_sq.max(0.0).sqrt(),
                    centroid.x,
                    centroid.y,
                    centroid.z,
                );
            }

            // If every rotation attempt failed the wall check, fall back to identity.
            if best_posits.is_empty() {
                best_posits = orig_local.iter().map(|&local| local + centroid).collect();
            }

            placed_atoms.extend_from_slice(&best_posits);

            let mut mol_copy = mol.clone();
            let translation = centroid - orig_centroid;
            apply_rot_trans(&mut mol_copy, best_rot, translation);
            mols.push(mol_copy);
        }
    } else {
        // ── Unbounded: shell expansion ────────────────────────────────────────
        let min_sep = 2.0 * orig_radius + 2.0;
        let mut sphere_placed: Vec<(Vec3, f64)> = mols
            .iter()
            .map(|m| {
                let c = mol_centroid(m);
                let r = mol_bounding_radius(m, c);
                (c, r)
            })
            .collect();

        for copy_i in 0..copies {
            let base_dist = min_sep * (1.0 + copy_i as f64).cbrt();
            let mut new_centroid = orig_centroid + Vec3::new(base_dist, 0., 0.);
            'free_search: for attempt in 0..MAX_POS_ATTEMPTS {
                let dx: f64 = rng.random::<f64>() * 2.0 - 1.0;
                let dy: f64 = rng.random::<f64>() * 2.0 - 1.0;
                let dz: f64 = rng.random::<f64>() * 2.0 - 1.0;
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                if len < 1e-10 {
                    continue;
                }
                let dir = Vec3::new(dx / len, dy / len, dz / len);
                let dist = base_dist + attempt as f64 * orig_radius * 0.5;
                let candidate = orig_centroid + dir * dist;
                for &(c, r) in &sphere_placed {
                    if (c - candidate).magnitude() < r + orig_radius + 2.0 {
                        continue 'free_search;
                    }
                }
                new_centroid = candidate;
                break;
            }
            sphere_placed.push((new_centroid, orig_radius));

            let rot = {
                let (w, x, y, z): (f64, f64, f64, f64) =
                    (rng.random(), rng.random(), rng.random(), rng.random());
                Quaternion::new(w, x, y, z).to_normalized()
            };
            let translation = new_centroid - orig_centroid;
            let mut mol_copy = mol.clone();
            apply_rot_trans(&mut mol_copy, rot, translation);
            mols.push(mol_copy);
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Copies added in {elapsed} ms");
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
    static_peptide: bool,
    near_lig_thresh: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<(MdState, Vec<MolDynamics>), ParamError> {
    println!("Setting up dynamics...");

    // Extract explicit box side-lengths so add_copies can keep molecules inside the boundary.
    // Only meaningful for Fixed boxes; Pad boxes are sized after molecule placement so we skip them.
    let box_dims = match &cfg.sim_box {
        SimBoxInit::Fixed((lo, hi)) => Some((hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)),
        SimBoxInit::Pad(_) => None,
    };

    let (mols, pep_set) = setup_mols_dyn(
        mols_in,
        mol_specific_params,
        static_peptide,
        near_lig_thresh,
        box_dims,
    )?;
    *pep_atom_set = pep_set;

    // Uncomment as required for validating individual processes.
    let cfg = MdConfig {
        overrides: MdOverrides {
            // skip_water: true,
            // skip_water_relaxation: false,
            // bonded_disabled: false,
            // coulomb_disabled: true,
            // lj_disabled: true,
            // long_range_recip_disabled: true,
            snapshots_during_equilibration: true,
            // Merge with caller-supplied overrides so flags like `skip_water` are preserved.
            ..cfg.overrides.clone()
        },
        // zero_com_drift: false,
        // max_init_relaxation_iters: None,
        ..cfg.clone()
    };

    println!("Initializing MD state...");
    let res = MdState::new(dev, &cfg, &mols, param_set)?;
    println!("MD init done.");

    Ok(res)
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
    let mut md_pep_sel = state.volatile.md_local.pep_atom_set.clone();

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

    let cfg = if fast_init {
        &MdConfig {
            solvent: Solvent::None,
            max_init_relaxation_iters: None,
            ..state.to_save.md_config.clone()
        }
    } else {
        &state.to_save.md_config
    };

    match build_dynamics(
        &state.dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        cfg,
        state.ui.md.peptide_static,
        near_lig_thresh,
        &mut md_pep_sel,
    ) {
        Ok((md, custom_solvent)) => {
            let num_water = md.water.len();

            // Build viewer mols from the actual MD atoms (post hetero-filter, post H-addition)
            // rather than from the original `mols`. The original mol for a protein includes
            // hetero atoms (crystal waters, ligands) that are stripped by filter_peptide_atoms,
            // and may gain or lose H during prepare_peptide. Using the original atom count
            // causes the viewer's atom_posits to mismatch the trajectory snapshot length,
            // producing out-of-bounds errors in change_snapshot.
            let n_input_mols = mols.len();
            let mut viewer_mol_data: Vec<(FfMolType, MoleculeCommon, usize)> = mols
                .iter()
                .enumerate()
                .map(|(i, (ff, m, count))| {
                    // Non-peptide mols (small organics, lipids, etc.) are not filtered or
                    // H-augmented by build_dynamics, so their atom count and bond set are
                    // unchanged. Cloning the original mol preserves bonds for rendering.
                    if *ff != FfMolType::Peptide {
                        return (*ff, (*m).clone(), *count);
                    }

                    // Peptide: build from the actual post-filter, post-H-addition MD atoms
                    // so that atom_posits.len() matches what the trajectory snapshots store.
                    let start = md.mol_start_indices.get(i).copied().unwrap_or(0);
                    // mol_start_indices[i+1] is the start of the next mol or first ion.
                    let end = md
                        .mol_start_indices
                        .get(i + 1)
                        .copied()
                        .unwrap_or(md.atoms.len());

                    let actual_atoms: Vec<Atom> = md.atoms[start..end]
                        .iter()
                        .map(|a| Atom {
                            serial_number: a.serial_number,
                            element: a.element,
                            posit: Vec3 {
                                x: a.posit.x as f64,
                                y: a.posit.y as f64,
                                z: a.posit.z as f64,
                            },
                            ..Default::default()
                        })
                        .collect();

                    // Infer covalent bonds from atom distances (covers added H).
                    let atom_generics: Vec<AtomGeneric> = actual_atoms
                        .iter()
                        .map(|a| AtomGeneric {
                            serial_number: a.serial_number,
                            posit: a.posit,
                            element: a.element,
                            ..Default::default()
                        })
                        .collect();
                    let bonds: Vec<Bond> = create_bonds(&atom_generics)
                        .iter()
                        .filter_map(|bg| Bond::from_generic(bg, &actual_atoms).ok())
                        .collect();

                    let atom_posits = actual_atoms.iter().map(|a| a.posit).collect();
                    let mut mol_common = MoleculeCommon {
                        ident: m.ident.clone(),
                        atoms: actual_atoms,
                        bonds,
                        atom_posits,
                        selected_for_md: m.selected_for_md,
                        ..Default::default()
                    };
                    mol_common.build_adjacency_list();
                    (*ff, mol_common, *count)
                })
                .collect();

            // Add a single-atom viewer mol for each counter-ion appended by MdState::new.
            // These live at md.atoms[ion_start..] and each has its own mol_start_indices entry,
            // so they need a matching ViewerMolecule or the non-water atom count in
            // mols_and_traj_synced will fall short of snapshot.atom_posits.len().
            //
            // mol_start_indices layout: [input_mol_0, ..., input_mol_{n-1},
            //                            custom_solvent_0, ..., custom_solvent_{K-1},
            //                            ion_0, ion_1, ...]
            // We must skip past BOTH input mols and custom solvents to reach the ions.
            let ion_start = md
                .mol_start_indices
                .get(n_input_mols + custom_solvent.len())
                .copied()
                .unwrap_or(md.atoms.len());
            for a in &md.atoms[ion_start..] {
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
                viewer_mol_data.push((
                    FfMolType::SmallOrganic,
                    MoleculeCommon {
                        ident: a.force_field_type.clone(),
                        atoms: vec![ion_atom],
                        atom_posits: vec![posit],
                        selected_for_md: true,
                        ..Default::default()
                    },
                    1,
                ));
            }
            let mut viewer_mol_refs: Vec<(FfMolType, &MoleculeCommon, usize)> = viewer_mol_data
                .iter()
                .map(|(ff, m, c)| (*ff, m, *c))
                .collect();

            let custom_mol_commons = match custom_solvents_to_mol_commons(&custom_solvent) {
                Ok(v) => v,
                Err(e) => {
                    handle_err(&mut state.ui, e);
                    return;
                }
            };
            for mol_common in &custom_mol_commons {
                viewer_mol_refs.push((FfMolType::SmallOrganic, mol_common, 1));
            }

            state
                .volatile
                .md_local
                .viewer
                .add_mol_set(&viewer_mol_refs, num_water);

            state.volatile.md_local.mol_dynamics = Some(md);

            if run {
                state.volatile.md_local.start = Some(Instant::now());
                state.volatile.md_local.running = true;
            }
        }
        Err(e) => handle_err(&mut state.ui, e.descrip),
    }
    state.volatile.md_local.pep_atom_set = md_pep_sel;
}

/// Converts the `custom_solvent` vec returned by `MdState::new` into owned `MoleculeCommon`
/// values suitable for the viewer mol-set. Returns an error string if bond construction fails.
pub fn custom_solvents_to_mol_commons(
    custom_solvent: &[MolDynamics],
) -> Result<Vec<MoleculeCommon>, String> {
    let mut result = Vec::new();
    for mol in custom_solvent {
        // Use the packed (placed) positions from atom_posits, not the template positions
        // stored in mol.atoms. Without this, all copies appear at the template origin.
        let atoms: Vec<Atom> = match &mol.atom_posits {
            Some(packed_posits) => mol
                .atoms
                .iter()
                .zip(packed_posits)
                .map(|(a, p)| {
                    let mut atom: Atom = a.into();
                    atom.posit = Vec3 {
                        x: p.x,
                        y: p.y,
                        z: p.z,
                    };
                    atom
                })
                .collect(),
            None => mol.atoms.iter().map(|a| a.into()).collect(),
        };
        let mut bonds = Vec::with_capacity(mol.bonds.len());
        for bond in &mol.bonds {
            let b = Bond::from_generic(bond, &atoms)
                .map_err(|_| "Error constructing bonds from custom solvent".to_string())?;
            bonds.push(b);
        }
        result.push(MoleculeCommon::new(
            "Solvent".to_owned(),
            atoms,
            bonds,
            HashMap::new(),
            None,
        ));
    }
    Ok(result)
}

/// Called directly from the UI; computes energy.
// pub fn launch_md_energy_computation(state: &mut State) -> Result<Snapshot, ParamError> {
pub fn launch_md_energy_computation(state: &State) -> Result<Snapshot, ParamError> {
    let mols_in = get_mols_sel_for_md(state);

    let peptide_only_near_lig = if state.ui.md.peptide_only_near_ligs {
        Some(STATIC_ATOM_DIST_THRESH)
    } else {
        None
    };

    let (mols, pep_atom_set) = setup_mols_dyn(
        &mols_in,
        &state.mol_specific_params,
        state.ui.md.peptide_static,
        peptide_only_near_lig,
        None,
    )?;

    // state.volatile.md_local.pep_atom_set = pep_atom_set;

    compute_energy_snapshot(&state.dev, &mols, &state.ff_param_set)
}

/// A helper to reduce repetition. Loads references to mols in state that are selected for MD.
/// Doesn't convert to MolDynamics, but sets up in a way conducive to doing so.
pub fn get_mols_sel_for_md(state: &State) -> Vec<(FfMolType, &MoleculeCommon, usize)> {
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
    static_peptide: bool,
    near_lig_thresh: Option<f64>,
    box_dims: Option<(f32, f32, f32)>,
) -> Result<(Vec<MolDynamics>, HashSet<(usize, usize)>), ParamError> {
    let mut res = Vec::new();

    let mut pep_atom_set = HashSet::new();
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
            let (atoms, pep_set) = filter_peptide_atoms(mol, mols, near_lig_thresh);
            println!(
                "Peptide atom count: {}. Set count: {}",
                atoms.len(),
                pep_atom_set.len()
            );
            pep_atom_set = pep_atom_set;

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

        if *ff_mol_type == FfMolType::SmallOrganic {
            add_copies(&mut res, &mol, *copies, box_dims);
        } else {
            res.push(mol);
        }
    }

    Ok((res, pep_atom_set))
}

#[derive(Clone, Copy, PartialEq, Default, Encode, Decode)]
pub enum MdBackend {
    #[default]
    Dynamics,
    Gromacs,
    Orca,
}

impl Display for MdBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let v = match self {
            Self::Dynamics => "Dynamics",
            Self::Gromacs => "GROMACS",
            Self::Orca => "ORCA",
        };

        write!(f, "{v}")
    }
}

/// For the UI functions to launch MD, and run energy computations.
fn ready_to_run_helper(state: &mut State) -> bool {
    clear_cli_out(&mut state.ui);
    let mut ready_to_run = true;

    // Check that we have FF params and mol-specific parameters.
    for lig in &state.ligands {
        if !lig.common.selected_for_md {
            continue;
        }

        if !lig.ff_params_loaded || !lig.frcmod_loaded {
            state.ui.popup.show_get_geostd = true;
            ready_to_run = false;
        }
    }

    ready_to_run
}
/// Launched from the UI; initiates MD.
pub fn start_md(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    if !ready_to_run_helper(state) {
        return;
    }

    let center = match &state.peptide {
        Some(m) => m.center,
        None => Vec3::new(0., 0., 0.),
    };
    // todo: Set a loading indicator, and trigger the build next GUI frame.
    move_cam_to_active_mol(state, scene, center, updates);

    match state.to_save.md_backend {
        MdBackend::Dynamics => {
            handle_success(
                &mut state.ui,
                "Running MD. Initializing water, and relaxing the molecules...".to_string(),
            );

            // We will wait a frame so we can display the message above.
            state.volatile.md_local.launching = true;
        }
        MdBackend::Gromacs => {
            handle_success(&mut state.ui, "Running MD with GROMACS...".to_string());
            // We will wait a frame so we can display the message above.
            gromacs::launch_md(state)
        }
        // todo: For now, we launch ORCA MD from the ORCA UI.
        MdBackend::Orca => {}
    }
}

pub fn start_md_energy_computation(state: &mut State) {
    // todo: WIP
    if !ready_to_run_helper(state) {
        return;
    }

    let mut pep_md = state.volatile.md_local.pep_atom_set.clone(); // Avoids borrow problem.
    match launch_md_energy_computation(state) {
        Ok(snap) => {
            if let Some(en) = &snap.energy_data {
                let data = format!(
                    "E result. PE: {:.2}, PE NB: {:.3} PE Bonded: {:.2}",
                    en.energy_potential, en.energy_potential_nonbonded, en.energy_potential_bonded
                );
                println!("{data}");
                handle_success(&mut state.ui, data);

                state.volatile.md_local.pep_atom_set = pep_md;
            }
        }
        Err(e) => handle_err(&mut state.ui, format!("Error computing energy: {:?}", e)),
    }
}

/// Populate energy on snapshots that are between ones that have eneryg, for display purposes.
pub fn carry_over_snap_energy(snaps: &mut [Snapshot]) {
    let mut last_en = None;
    for snap in snaps {
        match &mut snap.energy_data {
            Some(e) => last_en = Some(e.clone()),
            None => {
                if last_en.is_some() {
                    snap.energy_data = last_en.clone();
                }
            }
        }
    }
}
