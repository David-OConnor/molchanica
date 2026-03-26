//! An interface to dynamics library.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    fmt::{Display, Formatter},
    io,
    io::ErrorKind,
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, create_bonds, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdOverrides, MdState, MolDynamics, ParamError,
    SimBoxInit, Solvent, compute_energy_snapshot, params::FfParamSet, snapshot::Snapshot,
};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::{Quaternion, Vec3};
use rand::Rng;

use crate::{
    drawing::{
        draw_peptide, draw_water,
        wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    },
    molecules::{
        common::MoleculeCommon,
        lipid::MoleculeLipid,
        nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType},
        peptide::MoleculePeptide,
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
    /// after an MD run. They may be created as copies of molecules in the primary state.
    /// (Mol, number of copies)
    /// For now, count is only set up on construction. We have a separate molecule instance for each count.
    pub peptides: Vec<MoleculePeptide>,
    pub small: Vec<MoleculeSmall>,
    pub lipids: Vec<MoleculeLipid>,
    pub nucleic_acids: Vec<MoleculeNucleicAcid>,
    /// One entry per copy of each custom solvent species (from `Solvent::Custom`), in the same
    /// order they were packed into the simulation.  Positions are updated from snapshots just
    /// like other molecule types, but always *after* all regular mol types because custom solvents
    /// are appended last to `MdState::all_mols` and therefore last in `mol_start_indices`.
    pub custom_solvents: Vec<MoleculeSmall>,
}

impl MdStateLocal {
    /// Update the molecules stored here; we render these instead of the primary state molecules, if in an
    /// appropriate mode for viewing.
    pub fn update_mols_for_disp(&mut self, mols: &[(FfMolType, &MoleculeCommon, usize)]) {
        self.peptides.clear();
        self.small.clear();
        self.lipids.clear();
        self.nucleic_acids.clear();
        self.custom_solvents.clear();

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
                        self.small.push(MoleculeSmall {
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

    /// Populate `custom_solvents` with one copy of each solvent template per packed copy,
    /// ready for snapshot-driven position updates and drawing.
    ///
    /// Call this *after* `update_mols_for_disp` (which clears `custom_solvents`) and
    /// *after* `build_dynamics`, in the same order the solvent species appear in
    /// `Solvent::Custom`.
    pub fn update_custom_solvents_for_disp(&mut self, solvents: &[(&MoleculeSmall, usize)]) {
        self.custom_solvents.clear();
        for (template, count) in solvents {
            for _ in 0..*count {
                self.custom_solvents.push(MoleculeSmall {
                    common: template.common.clone(),
                    ..Default::default()
                });
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

        let mut i_posits = 0;

        // This assumes we pack each mol type in the same order.

        for mol in &mut self.peptides {
            // change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snap);
            for (i_p, p) in mol.common.atom_posits.iter_mut().enumerate() {
                *p = posits_by_mol[i_posits][i_p].0.into(); //Posit only; discard velocity.
            }
            i_posits += 1;
        }

        for mol in &mut self.small {
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

        // Custom solvents are always appended last in mol_start_indices (they come after all
        // regular mols in the packed all_mols slice), so we process them here, after all other
        // molecule types.
        for mol in &mut self.custom_solvents {
            if i_posits >= posits_by_mol.len() {
                eprintln!(
                    "change_snapshot: ran out of mol positions while updating custom solvents \
                     (expected {} more)",
                    self.custom_solvents.len()
                );
                break;
            }
            for (i_p, p) in mol.common.atom_posits.iter_mut().enumerate() {
                *p = posits_by_mol[i_posits][i_p].0.into();
            }
            i_posits += 1;
        }

        Ok(())
    }

    /// We filter peptide hetero atoms out of the MD workflow. Adjust snapshot indices and atom positions so they
    /// are properly synchronized. This also handles the case of reassigning due to peptide atoms near the ligand.
    pub fn reassign_snapshot_indices(&mut self) {
        // if !pep.common.selected_for_md {
        if self.peptides.is_empty() {
            return;
        }

        let Some(md) = &mut self.mol_dynamics else {
            return;
        };

        let pep_atom_set = &self.peptide_selected;

        let pep = &self.peptides[0];

        println!("Re-assigning snapshot indices to match atoms excluded for MD...");

        let pep_count = pep_atom_set.len();
        let lig_atom_count = self.small.len();
        let lipid_atom_count = self.lipids.len();
        let na_atom_count = self.nucleic_acids.len();

        let pep_start_i = lig_atom_count + lipid_atom_count + na_atom_count;

        // Rebuild each snapshot's atom_posits: [ligands as-is] + [full peptide with holes filled]
        for snap in &mut md.snapshots {
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

    state.volatile.md_local.reassign_snapshot_indices();

    state.ui.current_snapshot = 0;
    if state.volatile.md_local.change_snapshot(0).is_err() {
        handle_err(
            &mut state.ui,
            String::from("Error changing snapshot at MD completion."),
        );
        return;
    }

    draw_mols(state, scene);
    updates.entities = EntityUpdate::All;

    handle_success(&mut state.ui, "MD complete".to_string());
}

/// Filter out hetero atoms, and if necessary, atoms not close to a ligand.
pub fn filter_peptide_atoms(
    set: &mut HashSet<(usize, usize)>,
    pep: &MoleculeCommon,
    mols_non_pep: &[(FfMolType, &MoleculeCommon, usize)],
    near_lig_thresh: Option<f64>,
) -> Vec<AtomGeneric> {
    *set = HashSet::new();

    pep.atoms
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
        .collect()
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
    mut static_peptide: bool,
    mut near_lig_thresh: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    // if mols_in.is_empty() {
    //     static_peptide = false;
    //     peptide_only_near_lig = None;
    // }

    // Extract explicit box side-lengths so add_copies can keep molecules inside the boundary.
    // Only meaningful for Fixed boxes; Pad boxes are sized after molecule placement so we skip them.
    let box_dims = match &cfg.sim_box {
        SimBoxInit::Fixed((lo, hi)) => Some((hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)),
        SimBoxInit::Pad(_) => None,
    };

    let mols = setup_mols_dyn(
        mols_in,
        mol_specific_params,
        pep_atom_set,
        static_peptide,
        near_lig_thresh,
        box_dims,
    )?;

    // Uncomment as required for validating individual processes.
    let mut cfg = MdConfig {
        overrides: MdOverrides {
            // skip_water: true,
            // skip_water_relaxation: false,
            // bonded_disabled: false,
            // coulomb_disabled: true,
            // lj_disabled: true,
            // long_range_recip_disabled: true,
            // thermo_disabled: false,
            // baro_disabled: false,
            snapshots_during_equilibration: true,
            // Merge with caller-supplied overrides so flags like `skip_water` are preserved.
            ..cfg.overrides.clone()
        },
        // zero_com_drift: false,
        // max_init_relaxation_iters: None,
        ..cfg.clone()
    };

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
        None,
    )?;

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
    pep_atom_set: &mut HashSet<(usize, usize)>,
    static_peptide: bool,
    near_lig_thresh: Option<f64>,
    box_dims: Option<(f32, f32, f32)>,
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
            let atoms = filter_peptide_atoms(pep_atom_set, mol, mols, near_lig_thresh);
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

        if *ff_mol_type == FfMolType::SmallOrganic {
            add_copies(&mut res, &mol, *copies, box_dims);
        } else {
            res.push(mol);
        }
    }

    Ok(res)
}

/// Draw all molecules from the MD run.
pub fn draw_mols(state: &mut State, scene: &mut Scene) {
    // todo: Only if at least one lig is involved.
    if !state.volatile.md_local.peptides.is_empty() {
        draw_peptide(state, scene);
    }

    // When drawing MD molecules, custom solvents (e.g. octanol) are rendered as ligands
    // alongside the regular solute.  We temporarily merge them into `mols_small` so that
    // `draw_all_ligs` handles them in one unified pass, then restore the separation.
    let custom_solvent_count = if state.volatile.md_local.draw_md_mols {
        state.volatile.md_local.custom_solvents.len()
    } else {
        0
    };
    if custom_solvent_count > 0 {
        // Drain into mols_small; draw_all_ligs will render all of them.
        let custom: Vec<MoleculeSmall> =
            state.volatile.md_local.custom_solvents.drain(..).collect();
        state.volatile.md_local.small.extend(custom);
    }
    if !state.volatile.md_local.small.is_empty() {
        draw_all_ligs(state, scene);
    }
    if custom_solvent_count > 0 {
        // Restore: the custom solvents were appended to the end of mols_small.
        let split_at = state.volatile.md_local.small.len() - custom_solvent_count;
        state.volatile.md_local.custom_solvents =
            state.volatile.md_local.small.drain(split_at..).collect();
    }

    if !state.volatile.md_local.lipids.is_empty() {
        draw_all_lipids(state, scene);
    }

    if !state.volatile.md_local.nucleic_acids.is_empty() {
        draw_all_nucleic_acids(state, scene);
    }

    // Asymmetry here: We don't store a copy of water molecules in MdStateLocal,
    // so we draw them from the snap directly. For other mol steps, we update local
    // mol posits with posits from the snap before running this.
    if let Some(md) = &state.volatile.md_local.mol_dynamics {
        let snap = &md.snapshots[state.ui.current_snapshot];

        draw_water(
            scene,
            &snap.water_o_posits,
            &snap.water_h0_posits,
            &snap.water_h1_posits,
            state.ui.visibility.hide_water,
        );
    }
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

/// Removes all MD snapshots, and performs related cleanup.
pub fn clear_snaps(state: &mut State) {
    let Some(md) = &mut state.volatile.md_local.mol_dynamics else {
        let txt = "Error: Attempting to change snapshot when there is no MD state";
        eprintln!("{txt}");
        return;
        // return Err(io::Error::new(ErrorKind::InvalidData, txt));
    };

    md.snapshots = Vec::new();
    md.mol_start_indices = Vec::new();

    state.volatile.md_local.peptides = Vec::new();
    state.volatile.md_local.small = Vec::new();
    state.volatile.md_local.lipids = Vec::new();
    state.volatile.md_local.nucleic_acids = Vec::new();
    state.volatile.md_local.custom_solvents = Vec::new();

    state.ui.current_snapshot = 0;

    state.volatile.md_local.draw_md_mols = false;
}
