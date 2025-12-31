//! Performs flexible alignment of two small molecules.
//!
//! One application: Recovering native ligand binding poses.
//!
//! [Wang, 2023](https://www.biorxiv.org/content/10.1101/2023.12.17.572051v2.full.pdf)
//! [Brown, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6598199/)
//! [BCL on Github](https://github.com/BCLCommons/bcl)
//!
//!
//! [Web based BCL::MolAlign](http://servers.meilerlab.org/index.php/servers/molalign)
//! This one may be useful for validating your results.
//!

use std::collections::{HashMap, HashSet, VecDeque};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::Element::*;
use nalgebra as na;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::molecule::MoleculeCommon;

/// For scores, a higher negative value indicates more similarity.
pub struct MolAlignment {
    pub posits: Vec<Vec3>,
    pub avg_strain_energy: f32,

    pub similarity_measure: f32,
    /// Sum of avg_strain_energy and similarity_measure.
    pub alignment_score: f32,
    /// Grades chemical and/or shape similarity. Insufficient when the molecules are of sufficiently
    /// different sizes.
    pub tanimoto_coefficient: f32,
}

impl MolAlignment {
    /// Align two molecules. This is generally for small molecules. Scores by steric hindrance, and
    /// minimizing electrostatic interactions. The molecules being aligned should generally have some
    /// shared structure but will not be the same molecule.
    pub fn create(mol_0: &MoleculeCommon, mol_1: &MoleculeCommon) -> Vec<Self> {
        // Self {
        //     posits: Vec::new(),
        //     score_steric: 0.,
        //     score_electrostatic: 0.,
        // }

        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct AlignmentResult {
    pub score: f32,
    pub transform: RigidTransform,
    pub matched_pairs: Vec<(usize, usize)>, // (atom_i in A, atom_j in B)
    pub aligned_a_posits: Vec<Vec3>,        // atom_posits for A after transform
}

#[derive(Copy, Clone, Debug)]
pub struct RigidTransform {
    pub rot: na::Matrix3<f32>,
    pub trans: na::Vector3<f32>,
}

impl RigidTransform {
    pub fn identity() -> Self {
        Self {
            rot: na::Matrix3::identity(),
            trans: na::Vector3::zeros(),
        }
    }

    pub fn apply(&self, p: na::Vector3<f32>) -> na::Vector3<f32> {
        self.rot * p + self.trans
    }
}

#[derive(Clone, Debug)]
pub struct MolAlignConfig {
    pub seed: u64,

    pub rigid_mol_b: bool,

    pub number_flexible_trajectories: usize,

    pub iterations: usize,
    pub filter_iterations: usize,
    pub refinement_iterations: usize,

    pub fraction_filtered_initially: f32,
    pub fraction_filtered_iteratively: f32,

    pub conformer_pairs: usize,

    pub mc_temperature: f32,
    pub mc_temperature_refine: f32,

    pub step_rot_radians: f32,
    pub step_trans: f32,

    pub torsion_step_radians: f32,
    pub torsion_moves_per_iter: usize,

    pub clash_scale: f32,
    pub clash_hard_fail: bool,

    pub max_pair_dist: f32,
    pub w_spatial: f32,
    pub w_prop: f32,
}

impl Default for MolAlignConfig {
    fn default() -> Self {
        Self {
            seed: 0xC0FFEE,

            rigid_mol_b: false,

            number_flexible_trajectories: 5,

            iterations: 800,
            filter_iterations: 400,
            refinement_iterations: 100,

            fraction_filtered_initially: 0.25,
            fraction_filtered_iteratively: 0.50,

            conformer_pairs: 2500,

            mc_temperature: 1.0,
            mc_temperature_refine: 0.25,

            step_rot_radians: 10_f32.to_radians(),
            step_trans: 0.35,

            torsion_step_radians: 15_f32.to_radians(),
            torsion_moves_per_iter: 1,

            clash_scale: 0.80,
            clash_hard_fail: true,

            max_pair_dist: 3.5,
            w_spatial: 1.0,
            w_prop: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conformer {
    pub posits: Vec<Vec3>,
    pub plausibility: f32, // higher = better; can just be 0.0 if unknown
}

pub trait ConformerGenerator {
    fn generate(&self, mol: &MoleculeCommon, max: usize, rng: &mut SmallRng) -> Vec<Conformer>;
}

pub struct MolAlign<'a> {
    cfg: MolAlignConfig,
    gen_a: Option<&'a dyn ConformerGenerator>,
    gen_b: Option<&'a dyn ConformerGenerator>,

    props_a: Vec<AtomProps>,
    props_b: Vec<AtomProps>,

    rot_bonds_a: Vec<RotBond>,
    rot_bonds_b: Vec<RotBond>,
}

impl<'a> MolAlign<'a> {
    pub fn new(
        cfg: MolAlignConfig,
        gen_a: Option<&'a dyn ConformerGenerator>,
        gen_b: Option<&'a dyn ConformerGenerator>,
    ) -> Self {
        Self {
            cfg,
            gen_a,
            gen_b,
            props_a: Vec::new(),
            props_b: Vec::new(),
            rot_bonds_a: Vec::new(),
            rot_bonds_b: Vec::new(),
        }
    }

    pub fn create(&mut self, mol_a: &MoleculeCommon, mol_b: &MoleculeCommon) -> AlignmentResult {
        let mut rng = SmallRng::seed_from_u64(self.cfg.seed);

        self.props_a = compute_atom_props(mol_a);
        self.props_b = compute_atom_props(mol_b);

        self.rot_bonds_a = find_rotatable_bonds(mol_a);
        self.rot_bonds_b = find_rotatable_bonds(mol_b);

        let confs_a = self.make_confs(mol_a, self.gen_a, &mut rng);
        let confs_b = if self.cfg.rigid_mol_b {
            vec![Conformer {
                posits: mol_b.atom_posits.clone(),
                plausibility: 0.0,
            }]
        } else {
            self.make_confs(mol_b, self.gen_b, &mut rng)
        };

        let mut trajectories = Vec::with_capacity(self.cfg.number_flexible_trajectories);

        for _ in 0..self.cfg.number_flexible_trajectories {
            let (ia, ib) = (
                rng.random_range(0..confs_a.len()),
                rng.random_range(0..confs_b.len()),
            );

            let mut state = TrajState {
                conf_a: confs_a[ia].clone(),
                conf_b: confs_b[ib].clone(),
                t_ab: random_transform(&mut rng),
                best: None,
            };

            let best0 = self.run_tier(&mut rng, &mut state, mol_a, mol_b, Tier::Initial);
            state.best = Some(best0);
            trajectories.push(state);
        }

        filter_trajectories(&mut trajectories, self.cfg.fraction_filtered_initially);

        for s in trajectories.iter_mut() {
            let best = self.run_tier(&mut rng, s, mol_a, mol_b, Tier::Filter);
            s.best = Some(best);
        }

        filter_trajectories(&mut trajectories, self.cfg.fraction_filtered_iteratively);

        for s in trajectories.iter_mut() {
            let best = self.run_tier(&mut rng, s, mol_a, mol_b, Tier::Refine);
            s.best = Some(best);
        }

        trajectories
            .into_iter()
            .filter_map(|t| t.best)
            .min_by(|a, b| a.score.total_cmp(&b.score))
            .unwrap_or_else(|| fallback_identity(mol_a))
    }

    fn make_confs(
        &self,
        mol: &MoleculeCommon,
        gen_: Option<&dyn ConformerGenerator>,
        rng: &mut SmallRng,
    ) -> Vec<Conformer> {
        if let Some(g) = gen_ {
            let mut v = g.generate(mol, 50, rng);
            if v.is_empty() {
                v.push(Conformer {
                    posits: mol.atom_posits.clone(),
                    plausibility: 0.0,
                });
            }
            v
        } else {
            vec![Conformer {
                posits: mol.atom_posits.clone(),
                plausibility: 0.0,
            }]
        }
    }

    fn run_tier(
        &self,
        rng: &mut SmallRng,
        state: &mut TrajState,
        mol_a: &MoleculeCommon,
        mol_b: &MoleculeCommon,
        tier: Tier,
    ) -> AlignmentResult {
        let (iters, temp, step_rot, step_trans) = match tier {
            Tier::Initial => (
                self.cfg.iterations,
                self.cfg.mc_temperature,
                self.cfg.step_rot_radians,
                self.cfg.step_trans,
            ),
            Tier::Filter => (
                self.cfg.filter_iterations,
                self.cfg.mc_temperature * 0.5,
                self.cfg.step_rot_radians * 0.6,
                self.cfg.step_trans * 0.6,
            ),
            Tier::Refine => (
                self.cfg.refinement_iterations,
                self.cfg.mc_temperature_refine,
                self.cfg.step_rot_radians * 0.25,
                self.cfg.step_trans * 0.25,
            ),
        };

        let mut cur = self.evaluate_pose(state);
        let mut best = cur.clone();

        for _ in 0..iters {
            let proposal = self.propose_move(rng, state, mol_a, mol_b, &cur, step_rot, step_trans);

            let next = self.evaluate_pose_from(state, &proposal);

            let accept = metropolis_accept(rng, cur.score, next.score, temp);
            if accept {
                state.t_ab = proposal.t_ab;
                state.conf_a = proposal.conf_a;
                state.conf_b = proposal.conf_b;
                cur = next;

                if cur.score < best.score {
                    best = cur.clone();
                }
            }
        }

        best
    }

    fn propose_move(
        &self,
        rng: &mut SmallRng,
        state: &TrajState,
        mol_a: &MoleculeCommon,
        mol_b: &MoleculeCommon,
        cur: &PoseEval,
        step_rot: f32,
        step_trans: f32,
    ) -> Proposed {
        let mut p = Proposed {
            conf_a: state.conf_a.clone(),
            conf_b: state.conf_b.clone(),
            t_ab: state.t_ab,
        };

        let r: f32 = rng.random();
        if r < 0.50 && !cur.matched_pairs.is_empty() {
            let (ia, ib) = cur.matched_pairs[rng.random_range(0..cur.matched_pairs.len())];
            if let Some(t) = anchor_transform_from_atom_pair(
                mol_a,
                mol_b,
                &p.conf_a.posits,
                &p.conf_b.posits,
                ia,
                ib,
            ) {
                p.t_ab = t;
                return p;
            }
        }

        if r < 0.75 {
            p.t_ab = perturb_transform(rng, p.t_ab, step_rot, step_trans);
            return p;
        }

        if !self.rot_bonds_a.is_empty() {
            for _ in 0..self.cfg.torsion_moves_per_iter {
                let rb = &self.rot_bonds_a[rng.random_range(0..self.rot_bonds_a.len())];
                let angle = (rng.random::<f32>() * 2.0 - 1.0) * self.cfg.torsion_step_radians;
                let ok = apply_torsion_move(mol_a, &mut p.conf_a.posits, rb, angle);

                if ok {
                    let clash = clash_score(mol_a, &p.conf_a.posits, self.cfg.clash_scale);
                    if self.cfg.clash_hard_fail && clash.is_infinite() {
                        p.conf_a = state.conf_a.clone();
                    }
                }
            }
        }

        p
    }

    fn evaluate_pose(&self, state: &TrajState) -> AlignmentResult {
        self.evaluate_pose_from(state, state)
    }

    fn evaluate_pose_from(&self, state: &TrajState, proposal: &Proposed) -> AlignmentResult {
        let a_na: Vec<na::Vector3<f32>> = proposal
            .conf_a
            .posits
            .iter()
            .map(|&v| v3_to_na(v))
            .collect();
        let b_na: Vec<na::Vector3<f32>> = proposal
            .conf_b
            .posits
            .iter()
            .map(|&v| v3_to_na(v))
            .collect();

        let a_xf: Vec<na::Vector3<f32>> = a_na.iter().map(|&p| proposal.t_ab.apply(p)).collect();

        let (pairs, score) =
            dynamic_pair_and_score(&a_xf, &b_na, &self.props_a, &self.props_b, &self.cfg);

        let aligned_a_posits: Vec<Vec3> = a_xf.into_iter().map(na_to_v3).collect();

        AlignmentResult {
            score,
            transform: proposal.t_ab,
            matched_pairs: pairs,
            aligned_a_posits,
        }
    }
}

#[derive(Clone)]
struct PoseEval {
    score: f32,
    matched_pairs: Vec<(usize, usize)>,
}

#[derive(Clone)]
struct TrajState {
    conf_a: Conformer,
    conf_b: Conformer,
    t_ab: RigidTransform,
    best: Option<AlignmentResult>,
}

#[derive(Clone)]
struct Proposed {
    conf_a: Conformer,
    conf_b: Conformer,
    t_ab: RigidTransform,
}

impl Proposed {
    fn as_pose_eval(
        &self,
        props_a: &[AtomProps],
        props_b: &[AtomProps],
        cfg: &MolAlignConfig,
    ) -> PoseEval {
        let _ = (props_a, props_b, cfg);
        PoseEval {
            score: f32::INFINITY,
            matched_pairs: Vec::new(),
        }
    }
}

#[derive(Copy, Clone)]
enum Tier {
    Initial,
    Filter,
    Refine,
}

fn fallback_identity(mol_a: &MoleculeCommon) -> AlignmentResult {
    AlignmentResult {
        score: f32::INFINITY,
        transform: RigidTransform::identity(),
        matched_pairs: Vec::new(),
        aligned_a_posits: mol_a.atom_posits.clone(),
    }
}

fn filter_trajectories(ts: &mut Vec<TrajState>, frac_drop: f32) {
    if ts.is_empty() {
        return;
    }
    ts.sort_by(|a, b| {
        let sa = a.best.as_ref().map(|x| x.score).unwrap_or(f32::INFINITY);
        let sb = b.best.as_ref().map(|x| x.score).unwrap_or(f32::INFINITY);
        sa.total_cmp(&sb)
    });

    let keep = ((ts.len() as f32) * (1.0 - frac_drop)).ceil() as usize;
    let keep = keep.clamp(1, ts.len());
    ts.truncate(keep);
}

fn metropolis_accept(rng: &mut SmallRng, cur: f32, next: f32, temp: f32) -> bool {
    if next < cur {
        return true;
    }
    if !temp.is_finite() || temp <= 0.0 {
        return false;
    }
    let d = next - cur;
    let p = (-d / temp).exp();
    rng.random::<f32>() < p
}

/* ------------------------- Properties + scoring ------------------------- */

#[derive(Clone, Debug)]
struct AtomProps {
    elem_group: u8,
    hbond_donor: bool,
    hbond_acceptor: bool,
    aromaticish: bool,
    hydrophobicish: bool,
    charge_bucket: i8,
}

fn compute_atom_props(m: &MoleculeCommon) -> Vec<AtomProps> {
    let mut out = Vec::with_capacity(m.atoms.len());
    for (i, a) in m.atoms.iter().enumerate() {
        let deg = m.adjacency_list[i].len();

        let elem_group = match a.element {
            Hydrogen => 0,
            Carbon => 1,
            Nitrogen => 2,
            Oxygen => 3,
            Sulfur | Phosphorus => 4,
            _ => 5,
        };

        let aromaticish = a
            .force_field_type
            .as_deref()
            .map(|t| t.starts_with('c') || t.contains("ar"))
            .unwrap_or(false);

        let (don, acc) = approx_hbond(m, i);

        let hydrophobicish = matches!(a.element, Carbon) && !acc && !don;

        let q = a.partial_charge.unwrap_or(0.0);
        let charge_bucket = if q > 0.20 {
            1
        } else if q < -0.20 {
            -1
        } else {
            0
        };

        out.push(AtomProps {
            elem_group,
            hbond_donor: don,
            hbond_acceptor: acc,
            aromaticish,
            hydrophobicish,
            charge_bucket,
        });
    }
    out
}

fn approx_hbond(m: &MoleculeCommon, i: usize) -> (bool, bool) {
    let a = &m.atoms[i];
    let has_h_neighbor = m.adjacency_list[i]
        .iter()
        .any(|&j| matches!(m.atoms[j].element, Element::H));

    match a.element {
        Nitrogen => (has_h_neighbor, true),
        Oxygen => (has_h_neighbor, true),
        Sulfur => (false, true),
        _ => (false, false),
    }
}

fn prop_distance(pa: &AtomProps, pb: &AtomProps) -> f32 {
    let mut d = 0.0;

    if pa.elem_group != pb.elem_group {
        d += 1.0;
    }
    if pa.hbond_donor != pb.hbond_donor {
        d += 0.5;
    }
    if pa.hbond_acceptor != pb.hbond_acceptor {
        d += 0.5;
    }
    if pa.aromaticish != pb.aromaticish {
        d += 0.4;
    }
    if pa.hydrophobicish != pb.hydrophobicish {
        d += 0.3;
    }
    if pa.charge_bucket != pb.charge_bucket {
        d += 0.6;
    }

    d
}

fn dynamic_pair_and_score(
    a_pos: &[na::Vector3<f32>],
    b_pos: &[na::Vector3<f32>],
    props_a: &[AtomProps],
    props_b: &[AtomProps],
    cfg: &MolAlignConfig,
) -> (Vec<(usize, usize)>, f32) {
    let mut best_j_for_i = vec![None; a_pos.len()];
    let mut best_i_for_j = vec![None; b_pos.len()];

    for i in 0..a_pos.len() {
        let mut best = (f32::INFINITY, usize::MAX);
        for j in 0..b_pos.len() {
            let spatial = (a_pos[i] - b_pos[j]).norm();
            if spatial > cfg.max_pair_dist {
                continue;
            }
            let pd = prop_distance(&props_a[i], &props_b[j]);
            let cost = cfg.w_spatial * spatial + cfg.w_prop * pd;
            if cost < best.0 {
                best = (cost, j);
            }
        }
        if best.1 != usize::MAX {
            best_j_for_i[i] = Some((best.1, best.0));
        }
    }

    for j in 0..b_pos.len() {
        let mut best = (f32::INFINITY, usize::MAX);
        for i in 0..a_pos.len() {
            let spatial = (a_pos[i] - b_pos[j]).norm();
            if spatial > cfg.max_pair_dist {
                continue;
            }
            let pd = prop_distance(&props_a[i], &props_b[j]);
            let cost = cfg.w_spatial * spatial + cfg.w_prop * pd;
            if cost < best.0 {
                best = (cost, i);
            }
        }
        if best.1 != usize::MAX {
            best_i_for_j[j] = Some((best.1, best.0));
        }
    }

    let mut pairs = Vec::new();
    let mut used_a = vec![false; a_pos.len()];
    let mut used_b = vec![false; b_pos.len()];

    for i in 0..a_pos.len() {
        let Some((j, cost_ij)) = best_j_for_i[i] else {
            continue;
        };
        let Some((i2, _cost_ji)) = best_i_for_j[j] else {
            continue;
        };
        if i2 != i {
            continue;
        }
        if used_a[i] || used_b[j] {
            continue;
        }

        used_a[i] = true;
        used_b[j] = true;
        pairs.push((i, j));

        let _ = cost_ij;
    }

    let mut score = 0.0;
    for (i, j) in &pairs {
        let spatial = (a_pos[*i] - b_pos[*j]).norm();
        let pd = prop_distance(&props_a[*i], &props_b[*j]);
        score += cfg.w_spatial * spatial + cfg.w_prop * pd;
    }

    if pairs.is_empty() {
        score = f32::INFINITY;
    }

    (pairs, score)
}

/* ------------------------- Pose moves (rigid) ------------------------- */

fn random_transform(rng: &mut SmallRng) -> RigidTransform {
    let axis = random_unit_vec(rng);
    let angle = rng.random::<f32>() * std::f32::consts::TAU;
    let rot = na::Rotation3::from_axis_angle(&na::Unit::new_normalize(axis), angle)
        .matrix()
        .clone();
    let trans = na::Vector3::new(
        (rng.random::<f32>() * 2.0 - 1.0) * 2.0,
        (rng.random::<f32>() * 2.0 - 1.0) * 2.0,
        (rng.random::<f32>() * 2.0 - 1.0) * 2.0,
    );
    RigidTransform { rot, trans }
}

fn perturb_transform(
    rng: &mut SmallRng,
    t: RigidTransform,
    step_rot: f32,
    step_trans: f32,
) -> RigidTransform {
    let axis = random_unit_vec(rng);
    let angle = (rng.random::<f32>() * 2.0 - 1.0) * step_rot;
    let d_rot = na::Rotation3::from_axis_angle(&na::Unit::new_normalize(axis), angle)
        .matrix()
        .clone();

    let d_trans = na::Vector3::new(
        (rng.random::<f32>() * 2.0 - 1.0) * step_trans,
        (rng.random::<f32>() * 2.0 - 1.0) * step_trans,
        (rng.random::<f32>() * 2.0 - 1.0) * step_trans,
    );

    RigidTransform {
        rot: d_rot * t.rot,
        trans: t.trans + d_trans,
    }
}

fn random_unit_vec(rng: &mut SmallRng) -> na::Vector3<f32> {
    let mut v = na::Vector3::new(
        rng.random::<f32>() * 2.0 - 1.0,
        rng.random::<f32>() * 2.0 - 1.0,
        rng.random::<f32>() * 2.0 - 1.0,
    );
    let n = v.norm();
    if n < 1e-6 {
        v = na::Vector3::x();
    } else {
        v /= n;
    }
    v
}

fn anchor_transform_from_atom_pair(
    mol_a: &MoleculeCommon,
    mol_b: &MoleculeCommon,
    pos_a: &[Vec3],
    pos_b: &[Vec3],
    ia: usize,
    ib: usize,
) -> Option<RigidTransform> {
    let pa = v3_to_na(pos_a[ia]);
    let pb = v3_to_na(pos_b[ib]);

    let na_a = pick_frame_neighbor(mol_a, ia)?;
    let na_b = pick_frame_neighbor(mol_b, ib)?;

    let va = (v3_to_na(pos_a[na_a]) - pa).normalize();
    let vb = (v3_to_na(pos_b[na_b]) - pb).normalize();

    let rot = rotation_between(va, vb)?;
    let trans = pb - rot * pa;

    Some(RigidTransform { rot, trans })
}

fn pick_frame_neighbor(m: &MoleculeCommon, i: usize) -> Option<usize> {
    m.adjacency_list
        .get(i)?
        .iter()
        .copied()
        .find(|&j| !matches!(m.atoms[j].element, Hydrogen))
        .or_else(|| m.adjacency_list.get(i)?.first().copied())
}

fn rotation_between(a: na::Vector3<f32>, b: na::Vector3<f32>) -> Option<na::Matrix3<f32>> {
    let a = a.normalize();
    let b = b.normalize();

    let v = a.cross(&b);
    let c = a.dot(&b);

    if c > 0.999999 {
        return Some(na::Matrix3::identity());
    }

    if c < -0.999999 {
        let axis = orthogonal_unit(a);
        let r =
            na::Rotation3::from_axis_angle(&na::Unit::new_normalize(axis), std::f32::consts::PI);
        return Some(r.matrix().clone());
    }

    let vx = na::Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0);
    let rot = na::Matrix3::identity() + vx + (vx * vx) * (1.0 / (1.0 + c));
    Some(rot)
}

fn orthogonal_unit(a: na::Vector3<f32>) -> na::Vector3<f32> {
    let v = if a.x.abs() < 0.9 {
        na::Vector3::x()
    } else {
        na::Vector3::y()
    };
    a.cross(&v).normalize()
}

/* ------------------------- Torsions + clashes ------------------------- */

#[derive(Clone, Debug)]
struct RotBond {
    bond_i: usize,
    a0: usize,
    a1: usize,
    downstream_from_a1: Vec<usize>, // rotate this side when rotating around (a0-a1)
}

fn find_rotatable_bonds(m: &MoleculeCommon) -> Vec<RotBond> {
    let mut out = Vec::new();

    for (i, b) in m.bonds.iter().enumerate() {
        if !matches!(b.bond_type, BondType::Single) {
            continue;
        }

        let a0 = b.atom_0;
        let a1 = b.atom_1;

        if m.adjacency_list[a0].len() <= 1 || m.adjacency_list[a1].len() <= 1 {
            continue;
        }

        if edge_in_ring(&m.adjacency_list, a0, a1) {
            continue;
        }

        let downstream = downstream_atoms(&m.adjacency_list, a0, a1);
        if downstream.is_empty() || downstream.len() == m.atoms.len() {
            continue;
        }

        out.push(RotBond {
            bond_i: i,
            a0,
            a1,
            downstream_from_a1: downstream,
        });
    }

    out
}

fn edge_in_ring(adj: &[Vec<usize>], a: usize, b: usize) -> bool {
    fn edge_key(a: usize, b: usize) -> (usize, usize) {
        if a < b { (a, b) } else { (b, a) }
    }

    let ignore = edge_key(a, b);
    let mut q = VecDeque::new();
    let mut seen = vec![false; adj.len()];
    q.push_back(a);
    seen[a] = true;

    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if edge_key(u, v) == ignore {
                continue;
            }
            if !seen[v] {
                if v == b {
                    return true;
                }
                seen[v] = true;
                q.push_back(v);
            }
        }
    }

    false
}

fn downstream_atoms(adj: &[Vec<usize>], fixed: usize, start: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut q = VecDeque::new();
    let mut seen = vec![false; adj.len()];

    seen[fixed] = true;
    seen[start] = true;
    q.push_back(start);

    while let Some(u) = q.pop_front() {
        out.push(u);
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                q.push_back(v);
            }
        }
    }

    out
}

fn apply_torsion_move(m: &MoleculeCommon, pos: &mut [Vec3], rb: &RotBond, angle: f32) -> bool {
    let p0 = v3_to_na(pos[rb.a0]);
    let p1 = v3_to_na(pos[rb.a1]);
    let axis = p1 - p0;
    let n = axis.norm();
    if n < 1e-6 {
        return false;
    }
    let axis_u = na::Unit::new_normalize(axis);

    let rot = na::Rotation3::from_axis_angle(&axis_u, angle)
        .matrix()
        .clone();

    for &i in &rb.downstream_from_a1 {
        let pi = v3_to_na(pos[i]);
        let rel = pi - p0;
        let pi2 = rot * rel + p0;
        pos[i] = na_to_v3(pi2);
    }

    let _ = m;
    true
}

fn clash_score(m: &MoleculeCommon, pos: &[Vec3], scale: f32) -> f32 {
    let n = m.atoms.len();
    let mut bonded: HashSet<(usize, usize)> = HashSet::new();
    for b in &m.bonds {
        let (a, c) = if b.atom_0 < b.atom_1 {
            (b.atom_0, b.atom_1)
        } else {
            (b.atom_1, b.atom_0)
        };
        bonded.insert((a, c));
    }

    let mut worst = 0.0f32;

    for i in 0..n {
        let ri = v3_to_na(pos[i]);
        let rvi = m.atoms[i].element.vdw_radius() * scale;

        for j in (i + 1)..n {
            let (a, b) = (i, j);
            if bonded.contains(&(a, b)) {
                continue;
            }

            let rj = v3_to_na(pos[j]);
            let rvj = m.atoms[j].element.vdw_radius() * scale;

            let d = (ri - rj).norm();
            let cutoff = rvi + rvj;
            if d < 1e-6 {
                return f32::INFINITY;
            }
            if d < cutoff {
                worst = worst.max((cutoff - d) / cutoff);
            }
        }
    }

    worst
}

/* ------------------------- Vec3 <-> nalgebra ------------------------- */

fn v3_to_na(v: Vec3) -> na::Vector3<f64> {
    na::Vector3::new(v.x, v.y, v.z)
}

fn na_to_v3(v: na::Vector3<f64>) -> Vec3 {
    Vec3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}
