//! Performs flexible alignment of two small molecules.
//!
//! One application: Recovering native ligand binding poses.
//!
//! [Wang, 2023](https://www.biorxiv.org/content/10.1101/2023.12.17.572051v2.full.pdf)
//! [Brown, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6598199/)
//! [BCL on Github](https://github.com/BCLCommons/bcl)
//!
//! [Web based BCL::MolAlign](http://servers.meilerlab.org/index.php/servers/molalign)

use std::{collections::HashSet, f64::consts::TAU};

use bio_files::BondType;
use lin_alg::f64::{Quaternion, Vec3, X_VEC, Y_VEC};
use na_seq::Element::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

use crate::{
    docking::Torsion,
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
    pub posits_aligned: Vec<Vec3>,
    /// Overall score?
    pub score: f64,
    pub avg_strain_energy: f64,
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
pub fn align(mol_template: &MoleculeSmall, mol_to_align: &MoleculeSmall) -> Vec<AlignmentResult> {
    let char_template = &mol_template.characterization;
    let char_align = &mol_to_align.characterization;

    println!("\n\nChar template: {char_template:?}");
    println!("\nChar align: {char_align:?}");

    let mut result = Vec::new();

    // todo temp to test the general interface.
    result.push(make_initial_alignment(mol_template, mol_to_align));

    result
}

/// Early concept: Use the template's atom positions as centers of attraction for the mol to align.
/// Move the template along force gradients until it finds its position of minimum energy.
fn perform_md(mol: &MoleculeCommon, pose: &PoseAlignment) {}

fn find_center_ring(rings: &[Ring], atoms: &[Atom], centroid: Vec3) -> Option<usize> {
    if rings.is_empty() {
        return None;
    }

    let mut closest = 0;
    let mut closest_dist = f64::INFINITY;

    for (i, ring) in rings.iter().enumerate() {
        let dist = (ring.center(atoms) - centroid).magnitude();

        if dist < closest_dist {
            closest = i;
            closest_dist = dist;
        }
    }

    Some(closest)
}

/// A crude approach to an initial alignment, by identifying rings near the molecules' center that are similar,
/// then aligning both the position and plane. The result should be that the positions generated have
/// these rings coplanar and with the same center.
///
/// todo: Other improvementes like checking ring atom count, taking advantage of ring systems,
/// todo: and rotating the result around the plane norm so they're fully aligned.
fn align_from_rings(
    posits_aligned: &mut [Vec3],
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
    rings_t: &[Ring],
    rings_mta: &[Ring],
    centroid_template: Vec3,
    centroid_to_align: Vec3,
) {
    // todo: Break down this ring-based alignment into a dedicated fn.
    // Rough start: See if there are any rings near the center, which align between the two.
    let ctr_ring_templ = find_center_ring(rings_t, &mol_template.common.atoms, centroid_template);
    let ctr_ring_mta = find_center_ring(rings_mta, &mol_to_align.common.atoms, centroid_to_align);

    // todo: This is a start: Handle multiple rings, fused rings with orientation, take into account ring size.
    // todo: Something like this for ring systems?

    // Align the rings in plane, and position.
    if let Some(c_t) = ctr_ring_templ {
        if let Some(c_m) = ctr_ring_mta {
            let ring_t = &rings_t[c_t];
            let ring_m = &rings_mta[c_m];

            let ring_t_center = ring_t.center(&mol_template.common.atoms);
            let ring_m_center = ring_m.center(&mol_to_align.common.atoms);

            // let posit_offset = ring_t_center - ring_m_center;

            // todo: Note that we have arbitrary sign on these norm vecs, so we may get a reversed alignment.

            // Align the two rings in orientation.
            // todo: QC this logic!
            let rotator = Quaternion::from_unit_vecs(ring_m.plane_norm, ring_t.plane_norm);

            for (i, atom) in mol_to_align.common.atoms.iter().enumerate() {
                let posit_rel = atom.posit - ring_m_center;

                posits_aligned[i] = ring_t_center + rotator.rotate_vec(posit_rel);
            }
        }

        // todo: Take advantage of this to match ring systems.
        for sys in &mol_template.characterization.ring_systems {
            if sys.contains(&c_t) {
                // todo: Compare the nature of this system to mol-to-align.
            }
        }
    }
}

/// Using fast and crude methods, create a starting alignment, to base future ones off.
fn make_initial_alignment(
    mol_template: &MoleculeSmall,
    mol_to_align: &MoleculeSmall,
) -> AlignmentResult {
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

    align_from_rings(
        &mut posits_aligned,
        mol_template,
        mol_to_align,
        rings_t,
        rings_mta,
        centroid_template,
        centroid_to_align,
    );

    let pose = PoseAlignment {
        torsions,
        anchor_atom_i,
        orientation,
    };

    let pose = PoseAlignment::default();
    let matched_pairs = Vec::new();

    AlignmentResult {
        pose,
        matched_pairs,
        posits_aligned,
        ..Default::default()
    }
}
