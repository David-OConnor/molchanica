//! Performs flexible alignment of two small molecules.
//!
//! One application: Recovering native ligand binding poses.
//!
//! [Wang, 2023](https://www.biorxiv.org/content/10.1101/2023.12.17.572051v2.full.pdf)
//! [Brown, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6598199/)
//! [BCL on Github](https://github.com/BCLCommons/bcl)
//!
//! [Web based BCL::MolAlign](http://servers.meilerlab.org/index.php/servers/molalign)

use std::{
    collections::{HashMap, HashSet},
    f64::consts::TAU,
};

use bio_files::BondType;
use dynamics::compute_energy_snapshot;
use lin_alg::f64::{Quaternion, Vec3, X_VEC, Y_VEC};
use na_seq::{Element, Element::*};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

use crate::{
    State,
    docking::Torsion,
    md::launch_md_energy_computation,
    mol_alignment,
    mol_characterization::Ring,
    mol_lig::MoleculeSmall,
    molecules::{Atom, common::MoleculeCommon, rotatable_bonds::RotatableBond},
    util::rotate_about_axis,
};

#[derive(Clone, Debug, Default)]
pub struct PoseAlignment {
    pub torsions: Vec<Torsion>,
    /// The offset of the ligand's anchor atom from the docking center.
    /// Only for rigid and torsion-set-based conformations.
    /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    pub anchor_atom_i: usize,
    // pub anchor_posit: Vec3,
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
    pub pose: PoseAlignment,
    pub matched_pairs: Vec<(usize, usize)>, // (atom_i in atom 0, atom_j in atom 1)
    /// Atom_posits for atom 1; we leave atom 0 with its original positions.
    /// todo: Once we start flexing both molecules, these terms template and MTA don't make much sense.
    /// They make more sense for rigid molecules maybe?
    pub posits_template: Vec<Vec3>,
    pub posits_aligned: Vec<Vec3>,
    /// Overall score?
    pub score: f64,
    // todo: How does this work given we have two molecules? For now just the Mol-to-align.
    /// This is the potential energy of the molecule, averaged over the number of atoms in the molecule.
    pub avg_strain_energy: f32,
    pub similarity_measure: f64,
    /// Sum of avg_strain_energy and similarity_measure.
    pub alignment_score: f64,
    /// Grades chemical and/or shape similarity. Insufficient when the molecules are of sufficiently
    /// different sizes.
    pub tanimoto_coefficient: f64,
}

impl AlignmentResult {
    fn score(&mut self) {}
}

#[derive(Clone, Debug)]
pub struct MolAlignConfig {
    pub seed: u64,

    pub number_flexible_trajectories: usize,

    pub iterations: usize,
    pub filter_iterations: usize,
    pub refinement_iterations: usize,

    pub fraction_filtered_initially: f64,
    pub fraction_filtered_iteratively: f64,

    pub conformer_pairs: usize,

    pub mc_temperature: f64,
    pub mc_temperature_refine: f64,

    pub step_rot_radians: f64,
    pub step_trans: f64,

    pub torsion_step_radians: f64,
    pub torsion_moves_per_iter: usize,

    pub clash_scale: f64,
    pub clash_hard_fail: bool,

    pub max_pair_dist: f64,
    pub w_spatial: f64,
    pub w_prop: f64,
}

impl Default for MolAlignConfig {
    fn default() -> Self {
        Self {
            seed: 0xC0FFEE,

            number_flexible_trajectories: 5,

            iterations: 800,
            filter_iterations: 400,
            refinement_iterations: 100,

            fraction_filtered_initially: 0.25,
            fraction_filtered_iteratively: 0.50,

            conformer_pairs: 2500,

            mc_temperature: 1.0,
            mc_temperature_refine: 0.25,

            step_rot_radians: 10_f64.to_radians(),
            step_trans: 0.35,

            torsion_step_radians: 15_f64.to_radians(),
            torsion_moves_per_iter: 1,

            clash_scale: 0.80,
            clash_hard_fail: true,

            max_pair_dist: 3.5,
            w_spatial: 1.0,
            w_prop: 1.0,
        }
    }
}

/// Entry point for the alignment.
///
pub fn align(
    state: &State,
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
) -> Vec<AlignmentResult> {
    let char_template = &mol_template.characterization;
    let char_align = &mol_to_align.characterization;

    println!("\n\nChar template: {char_template:?}");
    println!("\nChar align: {char_align:?}");

    let mut result = make_initial_alignment(state, mol_template, mol_to_align);

    result
}

/// Early concept: Use the template's atom positions as centers of attraction for the mol to align.
/// Move the template along force gradients until it finds its position of minimum energy.
fn perform_md(mol: &MoleculeCommon, pose: &PoseAlignment) {}

// fn find_center_ring(rings: &[Ring], atoms: &[Atom], centroid: Vec3) -> Option<usize> {
//     if rings.is_empty() {
//         return None;
//     }
//
//     let mut closest = 0;
//     let mut closest_dist = f64::INFINITY;
//
//     for (i, ring) in rings.iter().enumerate() {
//         let dist = (ring.center(atoms) - centroid).magnitude();
//
//         if dist < closest_dist {
//             closest = i;
//             closest_dist = dist;
//         }
//     }
//
//     Some(closest)
// }

// todo: Crude/temporary
fn calc_score(
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
    posits_aligned: &[Vec3],
) -> f64 {
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
    let atoms_m = &mol_to_align.common.atoms;

    let mut by_el_t: HashMap<Element, Vec<Vec3>> = HashMap::new();
    let mut by_el_m: HashMap<Element, Vec<Vec3>> = HashMap::new();

    for a in atoms_t {
        if a.element == Hydrogen {
            continue;
        }
        by_el_t.entry(a.element).or_default().push(a.posit);
    }

    for (i, a) in atoms_m.iter().enumerate() {
        if a.element == Hydrogen {
            continue;
        }
        by_el_m
            .entry(a.element)
            .or_default()
            .push(posits_aligned[i]);
    }

    let mut els: HashSet<Element> = HashSet::new();
    els.extend(by_el_t.keys().copied());
    els.extend(by_el_m.keys().copied());

    let mut score = 0.0;
    for el in els {
        let a = by_el_t.get(&el).map(|v| v.as_slice()).unwrap_or(&[]);
        let b = by_el_m.get(&el).map(|v| v.as_slice()).unwrap_or(&[]);
        score += chamfer_dir(a, b) + chamfer_dir(b, a);
    }

    score
}
/// A crude approach to an initial alignment, by identifying rings near the molecules' center that are similar,
/// then aligning both the position and plane. The result should be that the positions generated have
/// these rings coplanar and with the same center.
///
/// Returns (positions template, positions_mta), and score.
///
/// todo: Other improvementes like checking ring atom count, taking advantage of ring systems,
/// todo: and rotating the result around the plane norm so they're fully aligned.
fn align_from_rings(
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
    rings_t: &[Ring],
    rings_mta: &[Ring],
) -> Vec<(Vec<Vec3>, Vec<Vec3>, f64)> {
    // todo: Break down this ring-based alignment into a dedicated fn.
    // Rough start: See if there are any rings near the center, which align between the two.
    // let ctr_ring_templ = find_center_ring(rings_t, &mol_template.common.atoms, centroid_template);
    // let ctr_ring_mta = find_center_ring(rings_mta, &mol_to_align.common.atoms, centroid_to_align);

    let mut result = Vec::new();

    const ROT_COUNT: u16 = 1_000; // Radians
    const ROT_STEP: f64 = TAU / ROT_COUNT as f64;

    // todo: This is a start: Handle multiple rings, fused rings with orientation, take into account ring size.
    // todo: Something like this for ring systems?

    // Align the rings in plane, and position.
    for ring_t in rings_t {
        for ring_m in rings_mta {
            let ring_t_center = ring_t.center(&mol_template.common.atoms);
            let ring_m_center = ring_m.center(&mol_to_align.common.atoms);

            // Try both orientations of the ring plane relative alignment.
            for ring_norm_sign in [-1., 1.] {
                let ring_t_plane_norm = ring_t.plane_norm * ring_norm_sign;

                // Align the two rings in orientation.
                let rotator = Quaternion::from_unit_vecs(ring_m.plane_norm, ring_t_plane_norm);

                let mut posits_these_rings = Vec::with_capacity(mol_to_align.common.atoms.len());
                for atom in &mol_to_align.common.atoms {
                    let posit_rel = atom.posit - ring_m_center;

                    posits_these_rings.push(ring_t_center + rotator.rotate_vec(posit_rel));
                }

                // todo: Could we, when scoring, create a "volume" of the combined molecules, and score
                // todo to minmize this volume? E.g. mols  closely alignmed would have a small volume.
                // todo: COuld you even use your Marching Cubes algorithm to do this? Try this!

                let mut best_score = f64::INFINITY;
                let mut best_i_rot = 0;

                for i_rot in 0..ROT_COUNT {
                    let mut posits_to_test = posits_these_rings.clone();

                    // Note: Here and in the final application, we seem to have to use ring_t
                    // for the center and plane norm, even though I believe they should be equivalent
                    // after our translation.
                    let rot_amt = ROT_STEP * i_rot as f64;
                    rotate_about_axis(
                        &mut posits_to_test,
                        ring_t_center,
                        ring_t_plane_norm,
                        rot_amt,
                    );

                    let mut score = calc_score(mol_template, mol_to_align, &posits_to_test);
                    if score < best_score {
                        best_score = score;
                        best_i_rot = i_rot;
                    }
                }

                // Apply the rotation with the best score.
                let rot_amt = ROT_STEP * best_i_rot as f64;
                rotate_about_axis(
                    &mut posits_these_rings,
                    ring_t_center,
                    ring_t_plane_norm,
                    rot_amt,
                );

                // Now, try rotating around the ring's norm to find the closest alignment.
                let posits_template = Vec::with_capacity(mol_template.common.atoms.len());
                result.push((posits_template, posits_these_rings, best_score));
            }
        }
    }

    result
}

/// Using fast and crude methods, create a starting alignment, to base future ones off.
fn make_initial_alignment(
    state: &State,
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
) -> Vec<AlignmentResult> {
    let torsions = Vec::new();
    let anchor_atom_i = 0;
    let orientation = Quaternion::new_identity();

    // todo: Kludge for now of resetting posits.

    let centroid_template = mol_template.common.centroid();
    let centroid_to_align = mol_to_align.common.centroid();

    let rings_t = &mol_template.characterization.rings;
    let rings_mta = &mol_to_align.characterization.rings;

    let mut posits_aligned: Vec<_> = mol_to_align
        .common
        .atoms
        .iter()
        .map(|atom| atom.posit)
        .collect();

    // todo: Consider trying multiple ring alignment configurations, instead of just closest-to-centroid
    // todo of each mol with each other.

    let posits_from_ring_alignment =
        align_from_rings(mol_template, mol_to_align, rings_t, rings_mta);

    // todo: Score these.

    let mut result = Vec::with_capacity(posits_from_ring_alignment.len());

    for (posits_template, posits_aligned, score) in posits_from_ring_alignment {
        let pose = PoseAlignment {
            torsions: torsions.clone(),
            anchor_atom_i,
            orientation,
        };

        let pose = PoseAlignment::default();
        let matched_pairs = Vec::new();

        let mut avg_strain_energy = 0.0;
        let energy = launch_md_energy_computation(state, &mut HashSet::new());
        if let Ok(energy) = energy {
            avg_strain_energy = energy.energy_potential / posits_aligned.len() as f32;
        }

        result.push(AlignmentResult {
            pose,
            matched_pairs,
            posits_template,
            posits_aligned,
            score,
            avg_strain_energy,
            ..Default::default()
        })
    }

    // Lowest (best) score first.
    result.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    result
}
