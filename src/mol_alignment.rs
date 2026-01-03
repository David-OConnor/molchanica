//! Performs flexible alignment of two small molecules.
//!
//! One application: Recovering native ligand binding poses.
//!
//! [Wang, 2023: Z-Align](https://www.biorxiv.org/content/10.1101/2023.12.17.572051v2.full.pdf)
//! [Brown, 2020: BCL::MolAlign](https://pmc.ncbi.nlm.nih.gov/articles/PMC6598199/)
//! [BCL on Github](https://github.com/BCLCommons/bcl)
//!
//! [Web based BCL::MolAlign](http://servers.meilerlab.org/index.php/servers/molalign)

use std::{
    collections::{HashMap, HashSet},
    f64::consts::TAU,
};

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{Element, Element::*};
use rayon::prelude::*;

const RING_ALIGN_ROT_COUNT: u16 = 1_000; // Radians

use crate::{
    State, docking::Torsion, md::launch_md_energy_computation, mol_alignment,
    mol_characterization::Ring, mol_lig::MoleculeSmall, molecules::common::MoleculeCommon,
    util::rotate_about_point,
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
    /// Post-alignment atom positions.
    /// The template molecule has no overall orientation and position transform, but has rotations
    /// around rotatable bonds. The query molecule has an overall position and orientation as well,
    /// to align it relative to the template.
    pub posits_template: Vec<Vec3>,
    pub posits_query: Vec<Vec3>,
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

    let alignments = align(
        state,
        &state.ligands[mta[0]],
        &state.ligands[mta[1]],
    );

    // Assume sorted score high to low
    if !alignments.is_empty() {
        println!("Found {} ring-based alignments", alignments.len());
        println!(
            "Best alignment strain energy: {}",
            alignments[0].avg_strain_energy
        );

        // If you want to *apply* the aligned coords back into ligand 0 (visualize):
        // (pick whichever molecule you want to move)

        // note: Try this as an alignment example: K3J and K2T
        // or neostigmine.sdf and physostigmine.sdf

        // [0] is the best score.
        state.ligands[1].common.atom_posits = alignments[0].posits_query.clone();

        *redraw_lig = true;
    }
}

/// Entry point for the alignment.
///
pub fn align(
    state: &State,
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
) -> Vec<AlignmentResult> {
    let mut result = make_initial_alignment(state, mol_template, mol_to_align);

    result
}

/// Early concept: Use the template's atom positions as centers of attraction for the query mol.
/// Move the template along force gradients until it finds its position of minimum energy.
fn perform_md(mol: &MoleculeCommon, pose: &PoseAlignment) {}

// todo: Crude/temporary
fn calc_score(
    mol_template: &MoleculeSmall,
    mol_query: &MoleculeSmall,
    posits_query: &[Vec3],
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

    let mut score = 0.0;
    for el in els {
        let a = by_el_t.get(&el).map(|v| v.as_slice()).unwrap_or(&[]);
        let b = by_el_m.get(&el).map(|v| v.as_slice()).unwrap_or(&[]);
        score += chamfer_dir(a, b) + chamfer_dir(b, a);
    }

    score
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
) -> (u16, f64) {
    let rot_step = TAU / rot_count as f64;

    let mut best_score = f64::INFINITY;
    let mut best = 0;

    for i_rot in 0..rot_count {
        let mut posits_to_test = posits_q_aligned.to_vec();
        let rot_amt = rot_step * i_rot as f64;

        let rotator = Quaternion::from_axis_angle(axis_dir_unit, rot_amt);
        rotate_about_point(&mut posits_to_test, axis_center, rotator);

        let score = calc_score(mol_template, mol_query, &posits_to_test);
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
) -> (u16, f64) {
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
) -> Option<Vec<(Vec<Vec3>, Vec<Vec3>, f64)>> {
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
) -> Option<Vec<(Vec<Vec3>, Vec<Vec3>, f64)>> {
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
    let torsions = Vec::new();
    let anchor_atom_i = 0;
    let orientation = Quaternion::new_identity();

    let rings_t = &mol_template.characterization.rings;
    let rings_q = &mol_query.characterization.rings;

    // todo: Consider trying multiple ring alignment configurations, instead of just closest-to-centroid
    // todo of each mol with each other.

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
        let pose = PoseAlignment {
            torsions: torsions.clone(),
            anchor_atom_i,
            orientation,
        };

        let pose = PoseAlignment::default();
        let matched_pairs = Vec::new();

        // todo: You must mark the mol in question as selected for MD, or this won't work.
        let mut avg_strain_energy = 0.0;
        let energy = launch_md_energy_computation(state, &mut HashSet::new());
        if let Ok(energy) = energy {
            avg_strain_energy = energy.energy_potential / posits_aligned.len() as f32;
        }
        // todo: Or should this be between the 2 mols?

        result.push(AlignmentResult {
            pose,
            matched_pairs,
            posits_template,
            posits_query: posits_aligned,
            score,
            avg_strain_energy,
            ..Default::default()
        })
    }

    // Lowest (best) score first.
    result.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    result
}
