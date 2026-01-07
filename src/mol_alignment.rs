//! Performs flexible alignment of two small molecules.
//!
//! One application: Recovering native ligand binding poses.
//!
//! [Wang, 2023: Z-Align. The closest approach to ours](https://www.biorxiv.org/content/10.1101/2023.12.17.572051v2.full.pdf)
//! This uses an MD engine (OpenFF) with GAFF2 force fields, and energy minimization.
//!
//! [Brown, 2020: BCL::MolAlign](https://pmc.ncbi.nlm.nih.gov/articles/PMC6598199/)
//! [BCL on Github](https://github.com/BCLCommons/bcl)
//!
//! [Web based BCL::MolAlign. May be useful for validation.](http://servers.meilerlab.org/index.php/servers/molalign)
//!
//! Keywords etc for the reverse process? the specific term is 3D similarity searching (shape same) or field based similarity searching (shape+electrostatic field same)
//! https://pubs.acs.org/doi/abs/10.1021/ci700130j

use std::{
    collections::{HashMap, HashSet},
    f64::consts::TAU,
};
use std::time::Instant;
use dynamics::{
    FfMolType, HydrogenConstraint, Integrator, MdConfig, MdOverrides, MdState, ParamError,
};

use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3},
};
use na_seq::{Element, Element::*};
use rayon::prelude::*;

// For initial rotation. Higher values take longer, but provide more precise results.
const RING_ALIGN_ROT_COUNT: u16 = 3_000; // Radians

// Setting this higher prioritizes our synthetic alignment forces relative to normal MD forces.
const COEFF_F_SYNTHETIC: f32 = 10.;

// These coefficients are for computing the mesh which represents combined molecule
// volume.
const VOL_RADIUS: f32 = SOLVENT_RAD; // todo: A/R. mod from 1.4
const VOL_PRECISION: f32 = 0.6; // todo: A/R.

const RELAX_ITERS_FINAL: usize = 30;

const NUM_STEPS: usize = 400;
const DT: f32 = 0.004; // ps

const TEMP: f32 = 60.; // K. Very low to minimize jiggling.

use crate::{
    State,
    docking::Torsion,
    md::{build_dynamics, launch_md_energy_computation},
    mol_characterization::Ring,
    mol_lig::MoleculeSmall,
    molecules::{Atom, Bond, common::MoleculeCommon},
    util::rotate_about_point,
};
use crate::sa_surface::{make_sas_mesh, SOLVENT_RAD};

#[derive(Clone, Debug, Default)]
pub struct PoseAlignment {
    pub torsions: Vec<Torsion>,
    /// The offset of the ligand's anchor atom from the docking center.
    /// Only for rigid and torsion-set-based conformations.
    /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    pub anchor_atom_i: usize,
    /// Only for rigid and torsion-set-based conformations.
    pub orientation: Quaternion,
}

impl PoseAlignment {
    pub fn position_atoms(&self, mol: &MoleculeCommon) -> Vec<Vec3> {
        if self.anchor_atom_i >= mol.atoms.len() {
            eprintln!(
                "Error positioning ligand atoms: Anchor outside atom count. Atom cound: {:?}",
                mol.atoms.len()
            );
            return Vec::new();
        }
        let anchor = mol.atom_posits[self.anchor_atom_i];

        let mut result: Vec<_> = mol
            .atoms
            .par_iter()
            .map(|atom| {
                let posit_rel = atom.posit - anchor;
                anchor + self.orientation.rotate_vec(posit_rel)
            })
            .collect();

        // Second pass: Rotations. For each flexible bond, divide all atoms into two groups:
        // those upstream of this bond, and those downstream. For all downstream atoms, rotate
        // by `torsions[i]`: The dihedral angle along this bond. If there are ambiguities in this
        // process, it may mean the bond should not have been marked as flexible.

        let mut mol_ = mol.clone();
        for torsion in &self.torsions {
            mol_.rotate_around_bond(torsion.bond, torsion.dihedral_angle);
        }

        result
    }
}

/// For scores, a higher negative value indicates more similarity.
#[derive(Clone, Debug, Default)]
pub struct AlignmentResult {
    // pub pose: PoseAlignment,
    // todo: Evaluate if you want this.
    // pub matched_pairs: Vec<(usize, usize)>, // (atom_i in atom 0, atom_j in atom 1)
    /// Post-alignment atom positions.
    /// The template molecule has no overall orientation and position transform, but has rotations
    /// around rotatable bonds. The query molecule has an overall position and orientation as well,
    /// to align it relative to the template. Note that these are fuzzy distinctions, as both molecules are flexible.
    /// it mainly applies to screening many "query" molecules against a template, and in these absolute
    /// positions and orientations.
    pub posits_template: Vec<Vec3>,
    pub posits_query: Vec<Vec3>,
    /// Overall score?
    pub score: f32,
    // todo: How does this work given we have two molecules? For now just the Mol-to-align.
    /// This is the potential energy of the molecule, averaged over the number of atoms in the molecule.
    pub avg_potential_e: f32,
    // pub similarity_measure: f64,
    /// The combined volume, in Å^3 from all atoms in both molecules. A smaller volume
    /// may indiacate a better score.
    pub volume: f32,
    /// Grades chemical and/or shape similarity. Insufficient when the molecules are of sufficiently
    /// different sizes.
    pub tanimoto_coefficient: f32,
}

pub fn run_alignment(state: &mut State, redraw_lig: &mut bool) {
    let mta = &state.volatile.mols_to_align;

    // todo: You must explicitly set which is the template.
    if mta.len() != 2 {
        eprintln!("Error: Alignment requires exactly two molecules to align.");
        return;
    }

    // todo: Temp! This needs to be in the alignment flow.
    state.ligands[mta[0]].common.reset_posits();
    state.ligands[mta[1]].common.reset_posits();

    let Ok((alignments, md)) = align(state, &state.ligands[mta[0]], &state.ligands[mta[1]]) else {
        eprintln!("Error: Alignment failed.");
        return;
    };

    println!("Snaps: {:?}", md.snapshots.len());
    state.mol_dynamics = Some(md);

    // Assume sorted score high to low
    if !alignments.is_empty() {
        println!("Found {} ring-based alignments", alignments.len());
        println!(
            "Best alignment strain energy: {}",
            alignments[0].avg_potential_e
        );

        // If you want to *apply* the aligned coords back into ligand 0 (visualize):
        // (pick whichever molecule you want to move)

        // note: Try this as an alignment example: K3J and K2T
        // or neostigmine.sdf and physostigmine.sdf

        // [0] is the best score.
        state.ligands[mta[1]].common.atom_posits = alignments[0].posits_query.clone();

        *redraw_lig = true;
    }
}

fn run_md(alignment: &mut AlignmentResult,
          mol_template: &MoleculeCommon, mol_query: &MoleculeCommon, state: &State,
          cfg: &MdConfig) -> Result<(Vec<Vec3>, Vec<Vec3>, MdState), ParamError> {
    // This is what we use for the MD and synthetic force. It's the same as mol_query,
    // but has updated atom positions.
    let mut mol_q_md = MoleculeCommon {
        selected_for_md: true,
        atom_posits: alignment.posits_query.clone(),
        ..mol_query.clone()
    };

    let mols_md = vec![(FfMolType::SmallOrganic, &mol_q_md)];

    let mut md = build_dynamics(
        &state.dev,
        &mols_md,
        None,
        &state.ff_param_set,
        &state.mol_specific_params,
        &cfg,
        false,
        None,
        &mut HashSet::new(),
        false,
    )?;

    // // For out synthetic-force VV integrator.
    // let mut acc_by_atom = vec![Vec3F32::new_zero(); mol_query.atoms.len()];
    // let mut vel_by_atom = vec![Vec3F32::new_zero(); mol_query.atoms.len()];

    let mut bonds_q_by_atom = Vec::with_capacity(mol_query.atoms.len());
    for atom_q in &mol_query.atoms {
        let bonds: Vec<_> = mol_query
            .bonds
            .iter()
            .filter(|b| {
                b.atom_0_sn == atom_q.serial_number || b.atom_1_sn == atom_q.serial_number
            })
            .collect();
        bonds_q_by_atom.push(bonds);
    }

    let mut bonds_t_by_atom = Vec::with_capacity(mol_template.atoms.len());
    for atom_q in &mol_template.atoms {
        let bonds: Vec<_> = mol_template
            .bonds
            .iter()
            .filter(|b| {
                b.atom_0_sn == atom_q.serial_number || b.atom_1_sn == atom_q.serial_number
            })
            .collect();
        bonds_t_by_atom.push(bonds);
    }

    let mut forces_by_atom_q = vec![Vec3F32::new_zero(); mol_query.atoms.len()];
    println!("Running MD for alignment...");
    for _ in 0..NUM_STEPS {

        // Experimenting.
        // let (free_charge_t, free_charge_q) = free_charge(&mol_template.atoms, &mol_query.atoms);

        // Reset force each step.
        forces_by_atom_q = vec![Vec3F32::new_zero(); mol_query.atoms.len()];

        // We update the synthetic query atoms, as they're what's maintaining position and velocity
        // during the simulation.
        for (i_q, atom_q_dyn) in md.atoms.iter_mut().enumerate() {
            let atom_q = &mol_q_md.atoms[i_q];

            // Apply our synthetic potential, drawing it to the template.
            for (i_t, atom_t) in mol_template.atoms.iter().enumerate() {
                let force = force_synthetic(
                    atom_t,
                    atom_q,
                    &bonds_t_by_atom[i_t],
                    &bonds_q_by_atom[i_q],
                    md.step_count,
                );
                forces_by_atom_q[i_q] += force * COEFF_F_SYNTHETIC;
            }

            // Add the free charge; out of the inner-loop fn, as we also take into account other query
            // atoms.

            // let charge_q = atom_q.partial_charge.unwrap_or(0.0);

            // // Query: Similar charges are repulsive; to cancel out the attraction from the template.
            // for (i_charge_t, charge_t) in free_charge_t.iter().enumerate() {
            //     let diff = (charge_t - charge_q).abs();
            //     // Attraction:
            //
            //     forces_by_atom_q[i_q] +=
            // }
            //
            // // Template: Similar charges are attractive.
            // for (i_charge_t, charge_t) in free_charge_t.iter().enumerate() {
            //     let diff = (charge_t - charge_q).abs();
            //     // Attraction:
            //
            //     forces_by_atom_q[i_q] +=
            // }

        }

        // Step using bonded and intra-atom nonbonded forces.
        md.step(&state.dev, DT, Some(forces_by_atom_q));

        // Update our atom positions from the MD run.
        for (i, atom_q) in mol_q_md.atoms.iter_mut().enumerate() {
            atom_q.posit = md.atoms[i].posit.into();
            mol_q_md.atom_posits[i] = atom_q.posit;
        }
    }

    // Experiment with this, and rm obviously if youre whole process is to relax.
    md.minimize_energy(&state.dev, RELAX_ITERS_FINAL, None);
    for (i, atom_q) in mol_q_md.atoms.iter_mut().enumerate() {
        atom_q.posit = md.atoms[i].posit.into();
        mol_q_md.atom_posits[i] = atom_q.posit;
    }

    println!("Complete");

    Ok((mol_template.atom_posits.clone(),  mol_q_md.atom_posits, md))
}

/// Entry point for the alignment.
/// Returns MdState for the purpose of viewing snapshots, for debugging etc.
pub fn align(
    state: &State,
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
) -> Result<(Vec<AlignmentResult>, MdState), ParamError> {
    let start = Instant::now();

    let mut result = make_initial_alignment(state, mol_template, mol_query);

    let md_state = {
        let mut md = MdState::default();

        let cfg_md = MdConfig {
            // A lower thermostat value is more aggressive. We want aggressive for this.
            integrator: Integrator::VerletVelocity { thermostat: Some(0.001) },
            // hydrogen_constraint: HydrogenConstraint::Flexible,
            hydrogen_constraint: HydrogenConstraint::Constrained,
            // max_init_relaxation_iters: Some(300), // todo: A/R
            max_init_relaxation_iters: None,
            temp_target: TEMP,
            overrides: MdOverrides {
                skip_water: true,
                snapshots_during_energy_min: true,
                ..Default::default()
            },
            ..Default::default()
        };

        for alignment in &mut result {
          let (posits_t, posits_q, md_this_align) = run_md(alignment, &mol_template.common, &mol_query.common, state, &cfg_md)?;

            // Update the initial alignment with the computed positions.
            alignment.posits_template = posits_t;
            alignment.posits_query = posits_q;

            // Update the score.
            alignment.score = calc_score(mol_template, mol_query, &alignment.posits_template, &alignment.posits_query);
            alignment.volume = calc_volume(&mol_template.common, &mol_query.common, VOL_RADIUS, VOL_PRECISION);

            md = md_this_align;
            break;
        };

        md
    };

    let elapsed = start.elapsed().as_millis();
    println!("Aligned 1 mol in {elapsed} ms");

    Ok((result, md_state))
}

// todo: Crude/temporary
fn calc_score(
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
    posits_template: &[Vec3],
    posits_query: &[Vec3],
) -> f32 {
    const MISSING_ELEM_PENALTY: f64 = 25.0 * 25.0; // (Å^2) per unmatched atom

    fn chamfer_dir(a: &[Vec3], b: &[Vec3]) -> f64 {
        if a.is_empty() {
            return 0.0;
        }
        if b.is_empty() {
            return a.len() as f64 * MISSING_ELEM_PENALTY;
        }

        let mut sum = 0.0;
        for &pa in a {
            let mut best = f64::INFINITY;
            for &pb in b {
                let dist_sq = (pa - pb).magnitude_squared();
                if dist_sq < best {
                    best = dist_sq;
                }
            }
            sum += best;
        }
        sum / a.len() as f64
    }

    let atoms_t = &mol_template.common.atoms;
    let atoms_q = &mol_query.common.atoms;

    let mut by_el_t: HashMap<Element, Vec<Vec3>> = HashMap::new();
    let mut by_el_m: HashMap<Element, Vec<Vec3>> = HashMap::new();

    for a in atoms_t {
        if a.element == Hydrogen {
            continue;
        }
        by_el_t.entry(a.element).or_default().push(a.posit);
    }

    for (i, a) in atoms_q.iter().enumerate() {
        if a.element == Hydrogen {
            continue;
        }
        by_el_m.entry(a.element).or_default().push(posits_query[i]);
    }

    let mut els: HashSet<Element> = HashSet::new();
    els.extend(by_el_t.keys().copied());
    els.extend(by_el_m.keys().copied());

    let mut res = 0.0;
    for el in els {
        let a = by_el_t.get(&el).map(|v| v.as_slice()).unwrap_or(&[]);
        let b = by_el_m.get(&el).map(|v| v.as_slice()).unwrap_or(&[]);
        res += chamfer_dir(a, b) + chamfer_dir(b, a);
    }

    res as f32
}

/// For initial alignment when one or more mol is missing rings. Find the carbon with at least
/// two neighbors closest to the centroid.
fn find_closest_to_ctr(mol: &MoleculeCommon) -> Option<usize> {
    let centroid = mol.centroid();

    let mut closest = None;
    let mut closest_dist_sq = f64::INFINITY;

    for (i, atom) in mol.atoms.iter().enumerate() {
        // Only choose no-H atoms with at least 2 covalently-bonded non-H atoms.
        if atom.element == Hydrogen || mol.adjacency_list[i].len() < 2 {
            continue;
        }

        if mol.atoms[mol.adjacency_list[i][0]].element == Hydrogen
            || mol.atoms[mol.adjacency_list[i][1]].element == Hydrogen
        {
            continue;
        }

        let dist_sq = (atom.posit - centroid).magnitude_squared();

        if dist_sq < closest_dist_sq {
            closest = Some(i);
            closest_dist_sq = dist_sq;
        }
    }

    closest
}

fn find_best_rotation_about_axis(
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
    posits_q_aligned: &[Vec3],
    axis_center: Vec3,
    axis_dir_unit: Vec3,
    rot_count: u16,
) -> (u16, f32) {
    let rot_step = TAU / rot_count as f64;

    let mut best_score = f32::INFINITY;
    let mut best = 0;

    for i_rot in 0..rot_count {
        let mut posits_to_test = posits_q_aligned.to_vec();
        let rot_amt = rot_step * i_rot as f64;

        let rotator = Quaternion::from_axis_angle(axis_dir_unit, rot_amt);
        rotate_about_point(&mut posits_to_test, axis_center, rotator);

        let posits_template = mol_template.common.atoms.iter().map(|a| a.posit).collect::<Vec<_>>();

        let score = calc_score(mol_template, mol_query, &posits_template, &posits_to_test);
        if score < best_score {
            best_score = score;
            best = i_rot;
        }
    }

    (best, best_score)
}

fn apply_best_rotation_about_axis(
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
    posits_q_aligned: &mut Vec<Vec3>,
    axis_center: Vec3,
    axis_dir_unit: Vec3,
    rot_count: u16,
) -> (u16, f32) {
    let (best_i_rot, best_score) = find_best_rotation_about_axis(
        mol_template,
        mol_query,
        posits_q_aligned,
        axis_center,
        axis_dir_unit,
        rot_count,
    );

    let rot_step = TAU / rot_count as f64;
    let rot_amt = rot_step * best_i_rot as f64;
    let rotator = Quaternion::from_axis_angle(axis_dir_unit, rot_amt);
    rotate_about_point(posits_q_aligned, axis_center, rotator);

    (best_i_rot, best_score)
}

/// If there are no rings to perform an initial alignment with, choose a feature like two connected
/// bonds near the centroid to align.
fn align_from_similar_center(
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
) -> Option<Vec<(Vec<Vec3>, Vec<Vec3>, f32)>> {
    let mut result = Vec::new();

    let atoms_t = &mol_template.common.atoms;
    let atoms_q = &mol_query.common.atoms;

    let Some(closest_to_ctr_t) = find_closest_to_ctr(&mol_template.common) else {
        return None;
    };
    let Some(closest_to_ctr_q) = find_closest_to_ctr(&mol_query.common) else {
        return None;
    };

    let nbrs_t = &mol_template.common.adjacency_list[closest_to_ctr_t];
    let nbrs_q = &mol_query.common.adjacency_list[closest_to_ctr_q];
    if nbrs_t.is_empty() || nbrs_q.is_empty() {
        return None;
    }

    let posit_ctr_t = atoms_t[closest_to_ctr_t].posit;

    // Position the query molecule so that its atom closest to center is overlaid on that of
    // the template's.
    let mut posits_q_translated = {
        let offset = posit_ctr_t - atoms_q[closest_to_ctr_q].posit;

        let mut p = Vec::with_capacity(atoms_q.len());
        for atom in atoms_q {
            p.push(atom.posit + offset);
        }
        p
    };

    // Use the translated center (should match posit_ctr_t) as the rotation point.
    let posit_ctr_q = posits_q_translated[closest_to_ctr_q];

    // Try aligning each neighbor-bond direction pair, in both axis directions (bond is undirected).
    // Then rotate around the aligned bond axis to find the best score (same pattern as rings).
    const ROT_COUNT: u16 = RING_ALIGN_ROT_COUNT;

    for &nbr_t in nbrs_t {
        let bond_t_dir_raw = (atoms_t[nbr_t].posit - posit_ctr_t).to_normalized();

        for &nbr_q in nbrs_q {
            let bond_q_dir = (posits_q_translated[nbr_q] - posit_ctr_q).to_normalized();

            for sign in [-1.0_f64, 1.0_f64] {
                let bond_t_dir = bond_t_dir_raw * sign;

                // First rotate so the chosen query bond aligns to the chosen template bond.
                let rotator = Quaternion::from_unit_vecs(bond_q_dir, bond_t_dir);

                let mut posits_q_aligned = posits_q_translated.clone();
                rotate_about_point(&mut posits_q_aligned, posit_ctr_t, rotator);

                // Now rotate around the aligned bond axis, scoring to find the best.
                let (_best_i_rot, best_score) = apply_best_rotation_about_axis(
                    mol_template,
                    mol_query,
                    &mut posits_q_aligned,
                    posit_ctr_t,
                    bond_t_dir,
                    ROT_COUNT,
                );

                let posits_template = Vec::with_capacity(mol_template.common.atoms.len());
                result.push((posits_template, posits_q_aligned, best_score));
            }
        }
    }

    Some(result)
}

/// A crude approach to an initial alignment, by identifying rings near the molecules' center that are similar,
/// then aligning both the position and plane. The result should be that the positions generated have
/// these rings coplanar and with the same center. We align all ring combinations, in both directions,
/// then rotate around these ring alignments while maintaining the ring plane, scoring to find the best.

/// Returns (positions template, positions_query), and score.
fn align_from_rings(
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
    rings_t: &[Ring],
    rings_q: &[Ring],
) -> Option<Vec<(Vec<Vec3>, Vec<Vec3>, f32)>> {
    if rings_t.is_empty() || rings_q.is_empty() {
        return None;
    }

    let mut result = Vec::new();

    // Align the rings in plane, and position.
    for ring_t in rings_t {
        for ring_m in rings_q {
            let ring_t_center = ring_t.center(&mol_template.common.atoms);
            let ring_m_center = ring_m.center(&mol_query.common.atoms);

            // Try both orientations of the ring plane relative alignment.
            for ring_norm_sign in [-1., 1.] {
                let ring_t_plane_norm = ring_t.plane_norm * ring_norm_sign;

                let mut posits_q_aligned = {
                    let rotator = Quaternion::from_unit_vecs(ring_m.plane_norm, ring_t_plane_norm);

                    let mut p = Vec::with_capacity(mol_query.common.atoms.len());
                    for atom in &mol_query.common.atoms {
                        let posit_rel = atom.posit - ring_m_center;
                        p.push(ring_t_center + rotator.rotate_vec(posit_rel));
                    }

                    p
                };

                let (_best_i_rot, best_score) = apply_best_rotation_about_axis(
                    mol_template,
                    mol_query,
                    &mut posits_q_aligned,
                    ring_t_center,
                    ring_t_plane_norm,
                    RING_ALIGN_ROT_COUNT,
                );

                let posits_template = Vec::with_capacity(mol_template.common.atoms.len());
                result.push((posits_template, posits_q_aligned, best_score));
            }
        }
    }

    Some(result)
}

/// Using fast and crude methods, create a starting alignment, to base future ones off.
fn make_initial_alignment(
    state: &State,
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
) -> Vec<AlignmentResult> {
    // let torsions = Vec::new();
    // let anchor_atom_i = 0;
    // let orientation = Quaternion::new_identity();

    let rings_t = &mol_template.characterization.rings;
    let rings_q = &mol_query.characterization.rings;

    let posits_initial_alignment = if rings_t.is_empty() || rings_q.is_empty() {
        align_from_similar_center(mol_template, mol_query)
    } else {
        align_from_rings(mol_template, mol_query, rings_t, rings_q)
    };

    let Some(posits_initial_alignment) = posits_initial_alignment else {
        return Vec::new();
    };

    let mut result = Vec::with_capacity(posits_initial_alignment.len());

    for (posits_template, posits_aligned, score) in posits_initial_alignment {
        // let pose = PoseAlignment {
        //     torsions: torsions.clone(),
        //     anchor_atom_i,
        //     orientation,
        // };

        // let pose = PoseAlignment::default();
        // let matched_pairs = Vec::new();

        // todo: You must mark the mol in question as selected for MD, or this won't work.
        let mut avg_strain_energy = 0.0;
        let energy = launch_md_energy_computation(state, &mut HashSet::new());
        if let Ok(energy) = energy {
            avg_strain_energy = energy.energy_potential / posits_aligned.len() as f32;
        }

        result.push(AlignmentResult {
            // pose,
            // matched_pairs,
            posits_template,
            posits_query: posits_aligned,
            score,
            avg_potential_e: avg_strain_energy,
            volume: calc_volume(&mol_template.common, &mol_query.common, VOL_RADIUS, VOL_PRECISION),
            ..Default::default()
        })
    }

    // Lowest (best) score first.
    result.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    result
}

// /// Experimenting. For each atom in the template, attract like charges, until
// /// it is *filled*?. An example of filled may be a query atom within q=0.02  is colocated with
// /// it.
// ///
// /// To do this, template charges attract similar query charges. Other query charges repl query charges.
// ///
// /// Todo: Instead, should we set up a charge or potential grid?
// ///
// /// Output is broken down into (indexes_t, indexes_q); we could return a single list to compare all
// /// Q to, but this may help use save distance computations when applying. (?)
// fn free_charge(
//     atoms_t: &[Atom],
//     atoms_q: &[Atom],
// ) -> (Vec<f32>, Vec<f32>) {
//     let mut res_t = Vec::with_capacity(atoms_t.len());
//     let mut res_q = Vec::with_capacity(atoms_q.len());
//
//     for atom_t in atoms_t {
//         res_t.push(atom_t.partial_charge.unwrap_or_default());
//     }
//
//     for atom_q in atoms_q {
//         res_q.push(atom_q.partial_charge.unwrap_or_default());
//     }
//
//     (res_t, res_q)
// }

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Adjust force to create a smooth potential. When distance is close to 0, the force should be close
/// to 0. When distance is more than ~2Å or so, force should gradually ramp to 0. There should be a
/// *sweet spot* where force is at a maximum; for example, 0.5Å. At this maximum, we should return 1.
fn scale_force_w_dist(dist: f32, r_peak: f32, r_max_: f32) -> f32 {
    const R_MAX_DEFAULT: f32 = 2.0;
    let r_max = r_max_.max(r_peak * 1.5); // ensure r_max > r_peak

    if dist >= r_max {
        return 0.0;
    }

    if dist < r_peak {
        // Ramp up smoothly from 0 at dist=0 to 1 at dist=r_peak (zero slope at both ends).
        smoothstep01(dist / r_peak)
    } else {
        // Ramp down smoothly from 1 at dist=r_peak to 0 at dist=r_max (zero slope at both ends).
        1.0 - smoothstep01((dist - r_peak) / (r_max - r_peak))
    }
}

/// A synthetic potential and force we use to align molecules. Aligns the Query molecule to the Template one
/// by nudging it using forces, or an energy-minimization algorithm.  Components:
/// - Element
/// - FF type (GAFF2)
/// - Type in residue (?)
/// - Bond types
/// - Partial charge
///
/// Note: In the position diff convention we use, attractive potentials are negative.
/// `bonds_t` and `bonds_q` are only the bonds connected to the respective atoms.
// fn force_synthetic(atom_t: &Atom, atom_q: &Atom, i_q: usize, bonds_t: &[&Bond], bonds_q: &[&Bond], free_q: &(Vec<f32>, Vec<f32>)) -> Vec3F32 {
fn force_synthetic(atom_t: &Atom, atom_q: &Atom, bonds_t: &[&Bond], bonds_q: &[&Bond], step: usize) -> Vec3F32 {
    const COEFF_EL: f32 = 0.3;

    const COEFF_Q: f32 = 1.2;
    // If the charge diff is closer than this, attract. If farther, repel.
    const THRESH_Q: f32 = 0.06;

    const COEFF_ATOM_NAME: f32 = 0.5;
    const COEFF_FF_TYPE: f32 = 0.3;
    const COEFF_BONDS: f32 = 0.5;

    // todo: You likely want a different dist thresh per type.
    // todo: Go with dist, or dist_sq?
    const THRESH_DIST: f32 = 2.5; //  Å
    const THRESH_DIST_DOUBLED: f32 = THRESH_DIST * 2.;

    const THRESH_DIST_SQ: f32 = THRESH_DIST * THRESH_DIST;
    const THRESH_DIST_SQ_DOUBLED: f32 = THRESH_DIST_DOUBLED * THRESH_DIST_DOUBLED;

    // Experimenting with ramping force by dist
    const R_PEAK: f32 = 0.7;   // Å  (sweet spot; strongest correction)
    const R_MAX: f32 = 2.0;    // Å  (beyond this, no pull)
    // const SHARPNESS: f32 = 2.0;


    // Note: template is the "source"; query is the "target", to use our terminology
    // from elsewhere.
    let diff: Vec3F32 = (atom_q.posit - atom_t.posit).into();
    let dist_sq = diff.magnitude_squared();

    // This saves computation for distant atoms.
    if dist_sq > THRESH_DIST_SQ_DOUBLED {
        return Vec3F32::new_zero();
    }

    let dist = dist_sq.sqrt();

    // Charge diffs smaller than `THRESH_Q` are attractive; larger are repulsive. Map linearly.
    // For example, if atom atom's charge is -0.32, and the other is -0.31, they should attract. If
    // they're say, -0.32, and -0.01, they should repel, as that's a significant difference.
    let f_q = {
        let q_diff =
            (atom_t.partial_charge.unwrap_or(0.0) - atom_q.partial_charge.unwrap_or(0.0)).abs();

        let f = COEFF_Q * (q_diff / THRESH_Q - 1.0);

        f.min(0.)
    };

    // Attractive if they share element.
    let f_el = if atom_t.element == atom_q.element {
        -COEFF_EL
    } else {
        0.
    };

    // Attractive if they share forcefield type (e.g. from GAFF2).
    // todo equivalent FF types.
    // todo: Use `dynamics::param_inference::matches_def()` or similar, to make sure similar but not
    // todo identical FF types count as matches. E.g. cc and cd. Or perhaps have them scored partially.
    // if matches_def()

    // Should never be missing.
    let ff_t = &atom_t.force_field_type.clone().unwrap_or(String::new());
    let ff_q = &atom_q.force_field_type.clone().unwrap_or(String::new());

    let f_ff_type = if atom_t.force_field_type == atom_q.force_field_type {
        -COEFF_FF_TYPE
    } else {
        // todo:  Hacks. Leverage your existing system, and the param files.
        if (ff_t == "cc" && ff_q == "cd") || (ff_t == "cd" && ff_q == "cc") {
            -COEFF_FF_TYPE * 0.9
        } else {
            0.
        }
    };

    // If the bond count is the same, attractive. If different, slightly repulsive, depending on teh difference.
    let f_bonds = {
        let mut bond_count_t = 0.;
        for bond in bonds_t {
            bond_count_t += bond.bond_type.order();
        }

        let mut bond_count_q = 0.;
        for bond in bonds_q {
            bond_count_q += bond.bond_type.order();
        }

        let db = (bond_count_q - bond_count_t).abs();
        let thresh = 0.5; // “close enough” bond-order sum
        let width = 0.25; // smoothness
        COEFF_BONDS * ((db - thresh) / width).tanh()
    };

    let f_mag = f_el + f_q + f_ff_type;

    let dir = if dist > 1e-6 {
        diff / dist
    } else {
        Vec3F32::new_zero()
    };

    let dist_term = scale_force_w_dist(dist, R_PEAK, R_MAX);
    // Convert negative-attractive convention into a positive pull strength.
    // Spring-like force (goes to 0 as dist->0), gated by sweet-spot bump.
    let f_vec = dir * f_mag *  dist_term;

    if step.is_multiple_of(500) {
        // if atom_q.serial_number.is_multiple_of(4) && atom_t.serial_number.is_multiple_of(4) {
        //     // println!(
        //     //     "P T: {:.2} P Q: {:.2}, f_el: {:.2}, f_q: {:.2}, dist: {:.3}, d term: {:.3}, F: {:.4}",
        //     //     atom_t.posit, atom_q.posit, f_el, f_q, dist, dist_term, f_vec
        //     // );
        //     println!("Rel forces. El: {f_el:.3} Q: {f_q:.3} Bonds: {f_bonds:.3} FF: {f_ff_type:.3}")
        // }
    }

    f_vec
}

/// Use a Marching-Cubes-generated isosurface to compute the volume of both template and query
/// molecules overlaid. A smaller volume may indicate a better match.
fn calc_volume(mol_a: &MoleculeCommon, mol_b: &MoleculeCommon, radius: f32, precision: f32) -> f32 {
    let start = Instant::now();

    let mut atoms = Vec::with_capacity(mol_a.atoms.len() + mol_b.atoms.len());
    for atom in &mol_a.atoms {
        atoms.push((atom.posit.into(), atom.element.vdw_radius()));
    }
    for atom in &mol_b.atoms {
        atoms.push((atom.posit.into(), atom.element.vdw_radius()));
    }

    // API issue here: Convert *back* to mcubes::Mesh, so we can access the `volume` method.
    let mesh_graphics = make_sas_mesh(&atoms, radius, precision);

    let vertices: Vec<mcubes::Vertex> = mesh_graphics
        .vertices
        .iter()
        // I'm not sure why we need to invert the normal here; same reason we use InsideOnly above.
        .map(|v|
            mcubes::Vertex { posit: Vec3F32::from_slice(&v.position).unwrap(), normal: -v.normal }
        )
        .collect();

    let mesh = mcubes::Mesh {
        indices: mesh_graphics.indices,
        vertices
    };

    let res = mesh.volume();

    let elapsed = start.elapsed().as_millis();
    println!("Volume computation took {} ms", elapsed);

    res
}