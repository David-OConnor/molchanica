//! Representation[s] of molecules (Based on MoleculeCommon? MoleculeSmall?) based on their different
//! conformations: The range of shapes they can take.
//!
//! todo: Is the true name "conformation"?
//!
//! Note: We are likely limited more by collecting the conformation data than by
//! the representation. The representation may be non-trivial, but the collecting
//! is likely to be very slow, depending on how it's done.
//!
//! todo: Can we run a brief ML sim in water to sample conformations? Or is that likely
//! todo to missing binding conformations?
//!
//! todo: Gaussians?

use std::{f64::consts::TAU, ops::Deref, time::Instant};

use bio_files::md_params::{DihedralParams, ForceFieldParams};
use dynamics::{ParamError, snapshot::Snapshot};
use graphics::Scene;
use lin_alg::f64::Vec3;
use na_seq::Element::Hydrogen;

use crate::{
    molecules::{common::MoleculeCommon, small::MoleculeSmall},
    state::State,
};

const DEFAULT_TORSION_BINS: usize = 72;
const DEFAULT_ATOM_DIST_HIST_BINS: usize = 24;
const DEFAULT_GLOBAL_HIST_BINS: usize = 24;
const BOLTZMANN_KCAL_PER_MOL_K: f64 = 0.001_987_204_1;
const DEFAULT_CONFORMER_TEMP_K: f64 = 298.15;
const TORSION_PROFILE_SMOOTHING: f64 = 0.05;
pub const CONFORMER_SUMMARY_FEATS: usize = 12;
pub const CONFORMER_SUMMARY_FEATURE_NAMES: [&str; CONFORMER_SUMMARY_FEATS] = [
    "conf_mean_atom_rmsf_ln",
    "conf_max_atom_rmsf_ln",
    "conf_mobile_atom_frac",
    "conf_mean_flex_depth_norm",
    "conf_mean_rotor_barrier_ln",
    "conf_weighted_rotor_barrier_ln",
    "conf_mean_rotor_circular_variance",
    "conf_weighted_rotor_circular_variance",
    "conf_mean_radius_of_gyration_ln",
    "conf_mean_max_radius_ln",
    "conf_mean_shape_anisotropy",
    "conf_mean_centroid_drift_ln",
];

/// A normalized 1D histogram. `bins` sum to 1 when samples are present.
#[derive(Clone, Debug, Default)]
pub struct Histogram1D {
    pub min: f64,
    pub max: f64,
    pub bins: Vec<f32>,
}

impl Histogram1D {
    pub fn from_values(values: &[f64], num_bins: usize) -> Self {
        let num_bins = num_bins.max(1);

        if values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                bins: vec![0.0; num_bins],
            };
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &value in values {
            min = min.min(value);
            max = max.max(value);
        }

        let mut bins = vec![0.0f32; num_bins];
        if (max - min).abs() <= 1.0e-12 {
            bins[0] = 1.0;
            return Self { min, max, bins };
        }

        let width = max - min;
        for &value in values {
            let pos = ((value - min) / width * num_bins as f64).floor() as usize;
            let idx = pos.min(num_bins - 1);
            bins[idx] += 1.0;
        }

        let inv_count = 1.0 / values.len() as f32;
        for bin in &mut bins {
            *bin *= inv_count;
        }

        Self { min, max, bins }
    }
}

/// For a single atom.
#[derive(Clone, Debug, Default)]
pub struct PositSample {
    /// Empirical positions sampled from the torsion-weighted conformer ensemble.
    pub samples: Vec<Vec3>,
    pub mean_posit: Vec3,
    /// Root mean square fluctuation around `mean_posit`.
    pub rmsf: f32,
    /// Largest displacement from the input coordinates.
    pub max_deviation: f32,
    /// Distances from a rigid-core anchor point.
    pub distance_from_reference: Histogram1D,
    pub mean_distance_from_reference: f32,
    /// Number of rotatable-bond moves that can displace this atom.
    pub flexibility_depth: usize,
}

#[derive(Clone, Debug)]
pub struct RotatableBondProfile {
    pub bond_i: usize,
    pub atom_0: usize,
    pub atom_1: usize,
    /// The smaller side rotated when this torsion changes.
    pub downstream_atoms: Vec<usize>,
    /// A representative dihedral angle from the input geometry, if available.
    pub reference_dihedral: Option<f64>,
    /// Torsion probability over `[0, TAU)`, normalized.
    pub angle_distribution: Histogram1D,
    /// `max(E) - min(E)` over the sampled torsion profile, in kcal/mol.
    pub effective_barrier: f32,
    /// Weighted average periodicity of the dihedral terms supporting this rotor.
    pub dominant_periodicity: f32,
    /// `1 - |E[e^{i phi}]|`; low means locked, high means broad / multimodal.
    pub circular_variance: f32,
}

#[derive(Clone, Debug, Default)]
pub struct GlobalConformationMetrics {
    pub radius_of_gyration: Histogram1D,
    pub max_radius_from_centroid: Histogram1D,
    pub shape_anisotropy: Histogram1D,
    pub centroid_drift: Histogram1D,
    pub mean_radius_of_gyration: f32,
    pub mean_max_radius_from_centroid: f32,
    pub mean_shape_anisotropy: f32,
    pub mean_centroid_drift: f32,
}

/// Attempt 0: Anchor to a molecule. These conormations map, for each atom,
/// the likely positions this atom can take. These positions are relative to the initial
/// positions of the molecule: `atoms[i].posit`: Not `atom_posits[i]`, as the former is a
/// more stable baseline.
#[derive(Clone, Debug)]
pub struct Conformer {
    pub mol: MoleculeCommon,
    /// A rigid-core anchor derived from the least-mobile heavy atoms.
    pub reference_point: Vec3,
    /// Atom indices used to define `reference_point`.
    pub reference_atoms: Vec<usize>,
    /// Indexed by atom in `mol`.
    pub atom_samples: Vec<PositSample>,
    pub rotatable_bonds: Vec<RotatableBondProfile>,
    pub global: GlobalConformationMetrics,
    pub sample_count: usize,
}

pub enum ConformerSource<'a> {
    Borrowed(&'a Conformer),
    Owned(Conformer),
}

impl Deref for ConformerSource<'_> {
    type Target = Conformer;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(conformer) => conformer,
            Self::Owned(conformer) => conformer,
        }
    }
}

impl Conformer {
    pub fn summary_features(&self) -> [f32; CONFORMER_SUMMARY_FEATS] {
        let ln = |x: f32| x.max(0.0).ln_1p();

        let atom_count = self.atom_samples.len().max(1) as f32;
        let max_depth = self.rotatable_bonds.len().max(1) as f32;

        let mut mean_rmsf = 0.0f32;
        let mut max_rmsf = 0.0f32;
        let mut mobile_atoms = 0usize;
        let mut mean_depth = 0.0;

        for atom_sample in &self.atom_samples {
            mean_rmsf += atom_sample.rmsf;
            max_rmsf = max_rmsf.max(atom_sample.rmsf);
            mean_depth += atom_sample.flexibility_depth as f32;
            if atom_sample.rmsf >= 0.35 {
                mobile_atoms += 1;
            }
        }

        mean_rmsf /= atom_count;
        mean_depth /= atom_count;

        let mut barrier_sum = 0.0f32;
        let mut weighted_barrier_sum = 0.0f32;
        let mut circular_variance_sum = 0.0f32;
        let mut weighted_circular_variance_sum = 0.0f32;
        let mut rotor_weight_sum = 0.0f32;

        for rotor in &self.rotatable_bonds {
            let weight = rotor.downstream_atoms.len().max(1) as f32;
            barrier_sum += rotor.effective_barrier;
            weighted_barrier_sum += rotor.effective_barrier * weight;
            circular_variance_sum += rotor.circular_variance;
            weighted_circular_variance_sum += rotor.circular_variance * weight;
            rotor_weight_sum += weight;
        }

        let rotor_count = self.rotatable_bonds.len().max(1) as f32;
        let mean_rotor_barrier = barrier_sum / rotor_count;
        let weighted_rotor_barrier = if rotor_weight_sum > 0.0 {
            weighted_barrier_sum / rotor_weight_sum
        } else {
            0.0
        };
        let mean_rotor_circular_variance = circular_variance_sum / rotor_count;
        let weighted_rotor_circular_variance = if rotor_weight_sum > 0.0 {
            weighted_circular_variance_sum / rotor_weight_sum
        } else {
            0.0
        };

        [
            ln(mean_rmsf),
            ln(max_rmsf),
            mobile_atoms as f32 / atom_count,
            mean_depth / max_depth,
            ln(mean_rotor_barrier),
            ln(weighted_rotor_barrier),
            mean_rotor_circular_variance,
            weighted_rotor_circular_variance,
            ln(self.global.mean_radius_of_gyration),
            ln(self.global.mean_max_radius_from_centroid),
            self.global.mean_shape_anisotropy,
            ln(self.global.mean_centroid_drift),
        ]
    }

    pub fn atom_feature_scale(&self) -> f32 {
        self.global
            .mean_max_radius_from_centroid
            .max(self.global.mean_radius_of_gyration)
            .max(1.0e-3)
    }
}

/// See the description on `sample_mol_properties_from_md`.
pub struct MdSampleData {
    // todo: Do we wish to include sample metadata like number of steps, dt, etc?
    pub conformer_samples: Vec<PositSample>, // Conformer struct instead?
    /// For which this molecule is the acceptor, and water is a donor.
    pub num_donor_h_bonds_avg: f32,
    /// For which this molecule is the donor, and water is an acceptor.
    pub num_acc_h_bonds_avg: f32,
    /// This includes both acceptor and donor data, and takes into account the H
    /// bond strength: Not just if there is an H bond.
    pub h_bond_total_str_avg: f32,
    // todo: You can imagine other remixes of donor, acceptor. Strength and count.
}

impl MdSampleData {
    pub fn new(snaps: &[Snapshot]) -> Self {
        // todo fill in
        unimplemented!();

        // todo: Prereq. You may need to add H bonds to your MD snap state.

        Self {
            conformer_samples: Vec::new(),
            num_donor_h_bonds_avg: 0.0,
            num_acc_h_bonds_avg: 0.0,
            h_bond_total_str_avg: 0.0,
        };
    }
}
/// WIP / experimental. Run a MD simulation of the molecule in water to sample various properties.
/// This includes conformation data (As is typical of this molecule this functionality currently lives in),
/// and other properties of interest. For example, perhaps we sample the number of hydrogen bonds the molecule
/// forms on average.
///
/// Limitation: This should be short, so it can be usd in screening. It's worth exploring, even if it
/// takes longer than desired for screening.
///
/// todo: For your MD applications, cache the computation of this while developing, and perhaps training.
///
/// todo: Evolve this over time, and move it where appropriate.
pub fn sample_mol_properties_from_md(
    mol: &MoleculeCommon, // todo: Should this be a MoleculeSmall?
    state: &State,        // todo: Do we need mut?
    scene: &Scene,        // todo: Why?
) -> Result<MdSampleData, ParamError> {
    println!("Starting per mol sim for {}...", mol.ident);
    let start = Instant::now();

    unimplemented!();
    // let md = build_dynamics()?;

    // let result = MdSampleData::new(&snaps);

    let elapsed = start.elapsed().as_millis();

    println!("Finished per mol sim in {} ms", elapsed);

    // result
}

/// For a given molecule, find its conformation using only data stored in the molecule itself. We
/// do not not run an MD sim, but use data more directly inferable from the molecule's properties, like
/// its rotatable bonds, and possibly its force field parameters. For example, we analyze each rotatable
/// bond.
pub fn characterize_conformations(
    mol: &MoleculeSmall,
    ff_params: &ForceFieldParams,
) -> Option<Conformer> {
    let char = mol.characterization.as_ref()?;

    // For this fast characterization pass we keep bond lengths / valence angles rigid and
    // spend the budget on the dominant low-energy degrees of freedom: torsions around
    // topologically rotatable single bonds.
    let mut mol_base = mol.common.clone();
    mol_base.reset_posits();

    let torsion_models = build_torsion_models(&mol_base, &char.rotatable_bonds, ff_params);
    let sample_count = ensemble_sample_count(torsion_models.len(), char.flexibility);
    let ensemble = build_conformation_ensemble(&mol_base, &torsion_models, sample_count);

    let flexibility_depth = atom_flexibility_depth(&mol_base, &torsion_models);
    let reference_atoms = choose_reference_atoms(&mol_base, &flexibility_depth);
    let reference_point = centroid_of_indices(&ensemble[0], &reference_atoms);

    let atom_samples =
        build_atom_samples(&mol_base, &ensemble, &flexibility_depth, reference_point);
    let global = build_global_metrics(&ensemble, &mol_base);

    Some(Conformer {
        mol: mol_base,
        reference_point,
        reference_atoms,
        atom_samples,
        rotatable_bonds: torsion_models
            .into_iter()
            .map(|model| model.profile)
            .collect(),
        global,
        sample_count: ensemble.len(),
    })
}

pub fn resolve_conformer<'a>(
    mol: &'a MoleculeSmall,
    ff_params: &ForceFieldParams,
) -> Option<ConformerSource<'a>> {
    if let Some(conformer) = mol.conformer.as_ref() {
        Some(ConformerSource::Borrowed(conformer))
    } else {
        characterize_conformations(mol, ff_params).map(ConformerSource::Owned)
    }
}

#[derive(Clone)]
struct ProperDihedralTerm {
    phi0: f64,
    params: Vec<DihedralParams>,
}

#[derive(Clone)]
struct TorsionModel {
    profile: RotatableBondProfile,
    cdf: Vec<f64>,
}

fn ensemble_sample_count(num_rotatable_bonds: usize, flexibility: f32) -> usize {
    if num_rotatable_bonds == 0 {
        1
    } else {
        let flex_load = flexibility.max(num_rotatable_bonds as f32);
        (32 + (flex_load * 10.0).round() as usize).clamp(48, 128)
    }
}

fn build_torsion_models(
    mol: &MoleculeCommon,
    rotatable_bonds: &[crate::molecules::rotatable_bonds::RotatableBond],
    ff_params: &ForceFieldParams,
) -> Vec<TorsionModel> {
    let mut models = Vec::with_capacity(rotatable_bonds.len());

    for rot_bond in rotatable_bonds {
        let Some(bond) = mol.bonds.get(rot_bond.bond_i) else {
            continue;
        };

        let terms = collect_proper_dihedral_terms(mol, bond.atom_0, bond.atom_1, ff_params);
        let reference_dihedral =
            representative_reference_dihedral(mol, bond.atom_0, bond.atom_1, &terms);
        let (angle_distribution, cdf, effective_barrier, dominant_periodicity, circular_variance) =
            torsion_profile(&terms);

        models.push(TorsionModel {
            cdf,
            profile: RotatableBondProfile {
                bond_i: rot_bond.bond_i,
                atom_0: bond.atom_0,
                atom_1: bond.atom_1,
                downstream_atoms: rot_bond.downstream_from_a1.clone(),
                reference_dihedral,
                angle_distribution,
                effective_barrier,
                dominant_periodicity,
                circular_variance,
            },
        });
    }

    models.sort_by(|a, b| {
        b.profile
            .downstream_atoms
            .len()
            .cmp(&a.profile.downstream_atoms.len())
            .then(a.profile.bond_i.cmp(&b.profile.bond_i))
    });

    models
}

fn collect_proper_dihedral_terms(
    mol: &MoleculeCommon,
    atom_1: usize,
    atom_2: usize,
    ff_params: &ForceFieldParams,
) -> Vec<ProperDihedralTerm> {
    let mut terms = Vec::new();

    for &atom_0 in mol.adjacency_list[atom_1].iter().filter(|&&i| i != atom_2) {
        for &atom_3 in mol.adjacency_list[atom_2].iter().filter(|&&i| i != atom_1) {
            if atom_0 == atom_3 {
                continue;
            }

            let Some(phi0) = dihedral_angle(
                mol.atoms[atom_0].posit,
                mol.atoms[atom_1].posit,
                mol.atoms[atom_2].posit,
                mol.atoms[atom_3].posit,
            ) else {
                continue;
            };

            let Some(params) = (match (
                mol.atoms[atom_0].force_field_type.clone(),
                mol.atoms[atom_1].force_field_type.clone(),
                mol.atoms[atom_2].force_field_type.clone(),
                mol.atoms[atom_3].force_field_type.clone(),
            ) {
                (Some(ff0), Some(ff1), Some(ff2), Some(ff3)) => {
                    ff_params.get_dihedral(&(ff0, ff1, ff2, ff3), true, true)
                }
                _ => None,
            }) else {
                continue;
            };

            terms.push(ProperDihedralTerm {
                phi0,
                params: params.clone(),
            });
        }
    }

    terms
}

fn representative_reference_dihedral(
    mol: &MoleculeCommon,
    atom_1: usize,
    atom_2: usize,
    terms: &[ProperDihedralTerm],
) -> Option<f64> {
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut weight_sum = 0.0;

    for term in terms {
        let weight = term
            .params
            .iter()
            .map(dihedral_weight)
            .sum::<f64>()
            .max(1.0e-6);
        sum_cos += weight * term.phi0.cos();
        sum_sin += weight * term.phi0.sin();
        weight_sum += weight;
    }

    if weight_sum > 1.0e-6 {
        return Some(sum_sin.atan2(sum_cos));
    }

    for &atom_0 in mol.adjacency_list[atom_1].iter().filter(|&&i| i != atom_2) {
        for &atom_3 in mol.adjacency_list[atom_2].iter().filter(|&&i| i != atom_1) {
            if atom_0 == atom_3 {
                continue;
            }

            if let Some(phi0) = dihedral_angle(
                mol.atoms[atom_0].posit,
                mol.atoms[atom_1].posit,
                mol.atoms[atom_2].posit,
                mol.atoms[atom_3].posit,
            ) {
                return Some(phi0);
            }
        }
    }

    None
}

fn torsion_profile(terms: &[ProperDihedralTerm]) -> (Histogram1D, Vec<f64>, f32, f32, f32) {
    let bin_count = DEFAULT_TORSION_BINS;
    let bin_centers: Vec<_> = (0..bin_count)
        .map(|i| TAU * (i as f64 + 0.5) / bin_count as f64)
        .collect();

    if terms.is_empty() {
        let uniform = vec![1.0 / bin_count as f32; bin_count];
        let cdf = cumulative_distribution(&uniform);
        return (
            Histogram1D {
                min: 0.0,
                max: TAU,
                bins: uniform,
            },
            cdf,
            0.0,
            0.0,
            1.0,
        );
    }

    let energies: Vec<f64> = bin_centers
        .iter()
        .map(|&delta| torsion_energy(delta, terms))
        .collect();

    let min_energy = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rt = BOLTZMANN_KCAL_PER_MOL_K * DEFAULT_CONFORMER_TEMP_K;

    let mut probs: Vec<f64> = energies
        .iter()
        .map(|&energy| (-(energy - min_energy) / rt).exp())
        .collect();

    let uniform = 1.0 / bin_count as f64;
    for prob in &mut probs {
        *prob = *prob * (1.0 - TORSION_PROFILE_SMOOTHING) + uniform * TORSION_PROFILE_SMOOTHING;
    }
    normalize_probs(&mut probs);

    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    for (&theta, &prob) in bin_centers.iter().zip(&probs) {
        sum_cos += prob * theta.cos();
        sum_sin += prob * theta.sin();
    }
    let circular_variance =
        (1.0 - (sum_cos.mul_add(sum_cos, sum_sin * sum_sin)).sqrt()).clamp(0.0, 1.0) as f32;

    let mut periodicity_sum = 0.0;
    let mut periodicity_weight = 0.0;
    for term in terms {
        for param in &term.params {
            let weight = dihedral_weight(param);
            periodicity_sum += weight * param.periodicity as f64;
            periodicity_weight += weight;
        }
    }
    let dominant_periodicity = if periodicity_weight > 1.0e-6 {
        (periodicity_sum / periodicity_weight) as f32
    } else {
        0.0
    };

    let bins: Vec<f32> = probs.iter().map(|&prob| prob as f32).collect();
    let cdf = cumulative_distribution_f64(&probs);

    (
        Histogram1D {
            min: 0.0,
            max: TAU,
            bins,
        },
        cdf,
        (max_energy - min_energy) as f32,
        dominant_periodicity,
        circular_variance,
    )
}

fn torsion_energy(delta: f64, terms: &[ProperDihedralTerm]) -> f64 {
    let mut energy = 0.0;

    for term in terms {
        let phi = term.phi0 + delta;
        for param in &term.params {
            let k = dihedral_weight(param);
            let dphi = param.periodicity as f64 * phi - param.phase as f64;
            energy += k * (1.0 + dphi.cos());
        }
    }

    energy
}

fn dihedral_weight(param: &DihedralParams) -> f64 {
    param.barrier_height as f64 / param.divider.max(1) as f64
}

fn normalize_probs(values: &mut [f64]) {
    let sum: f64 = values.iter().sum();
    if sum <= 1.0e-12 {
        let uniform = 1.0 / values.len().max(1) as f64;
        for value in values {
            *value = uniform;
        }
        return;
    }

    for value in values {
        *value /= sum;
    }
}

fn cumulative_distribution(values: &[f32]) -> Vec<f64> {
    cumulative_distribution_f64(&values.iter().map(|&v| v as f64).collect::<Vec<_>>())
}

fn cumulative_distribution_f64(values: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(values.len());
    let mut acc = 0.0;
    for &value in values {
        acc += value;
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

fn build_conformation_ensemble(
    mol: &MoleculeCommon,
    torsion_models: &[TorsionModel],
    sample_count: usize,
) -> Vec<Vec<Vec3>> {
    if torsion_models.is_empty() {
        return vec![mol.atoms.iter().map(|atom| atom.posit).collect()];
    }

    let mut ensemble = Vec::with_capacity(sample_count);
    for sample_i in 0..sample_count {
        let mut sample_mol = mol.clone();
        sample_mol.reset_posits();

        for (dim, model) in torsion_models.iter().enumerate() {
            let u = latin_hypercube_coordinate(sample_i, dim, sample_count);
            let angle = sample_angle_from_cdf(&model.cdf, u);
            sample_mol.rotate_around_bond(
                model.profile.bond_i,
                angle,
                Some(&model.profile.downstream_atoms),
            );
        }

        ensemble.push(sample_mol.atom_posits.clone());
    }

    ensemble
}

fn latin_hypercube_coordinate(sample_i: usize, dim_i: usize, sample_count: usize) -> f64 {
    if sample_count <= 1 {
        return 0.5;
    }

    let stride = coprime_stride(sample_count, dim_i + 1);
    let offset = (dim_i * (sample_count / 3 + 1) + 1) % sample_count;
    let idx = (sample_i * stride + offset) % sample_count;
    (idx as f64 + 0.5) / sample_count as f64
}

fn coprime_stride(modulus: usize, dim_seed: usize) -> usize {
    let mut step = dim_seed * 2 + 1;
    while gcd(step, modulus.max(1)) != 1 {
        step += 2;
    }
    step
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn sample_angle_from_cdf(cdf: &[f64], u: f64) -> f64 {
    let mut idx = cdf.partition_point(|&value| value < u);
    if idx >= cdf.len() {
        idx = cdf.len().saturating_sub(1);
    }
    TAU * (idx as f64 + 0.5) / cdf.len().max(1) as f64
}

fn atom_flexibility_depth(mol: &MoleculeCommon, torsion_models: &[TorsionModel]) -> Vec<usize> {
    let mut depth = vec![0usize; mol.atoms.len()];

    for model in torsion_models {
        for &atom_i in &model.profile.downstream_atoms {
            if atom_i == model.profile.atom_0 || atom_i == model.profile.atom_1 {
                continue;
            }
            depth[atom_i] += 1;
        }
    }

    depth
}

fn choose_reference_atoms(mol: &MoleculeCommon, flexibility_depth: &[usize]) -> Vec<usize> {
    let heavy_min = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.element != Hydrogen)
        .map(|(i, _)| flexibility_depth[i])
        .min();

    if let Some(min_depth) = heavy_min {
        let atoms: Vec<_> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, atom)| atom.element != Hydrogen)
            .filter(|(i, _)| flexibility_depth[*i] == min_depth)
            .map(|(i, _)| i)
            .collect();
        if !atoms.is_empty() {
            return atoms;
        }
    }

    let min_depth = flexibility_depth.iter().copied().min().unwrap_or(0);
    let atoms: Vec<_> = flexibility_depth
        .iter()
        .enumerate()
        .filter(|(_, depth)| **depth == min_depth)
        .map(|(i, _)| i)
        .collect();

    if atoms.is_empty() {
        (0..mol.atoms.len()).collect()
    } else {
        atoms
    }
}

fn build_atom_samples(
    mol: &MoleculeCommon,
    ensemble: &[Vec<Vec3>],
    flexibility_depth: &[usize],
    reference_point: Vec3,
) -> Vec<PositSample> {
    let sample_count = ensemble.len().max(1);
    let mut result = Vec::with_capacity(mol.atoms.len());

    for atom_i in 0..mol.atoms.len() {
        let samples: Vec<_> = ensemble.iter().map(|positions| positions[atom_i]).collect();
        let mean_posit = samples
            .iter()
            .copied()
            .fold(Vec3::new_zero(), |acc, posit| acc + posit)
            / sample_count as f64;

        let mean_sq =
            samples.iter().map(|posit| posit.dot(*posit)).sum::<f64>() / sample_count as f64;
        let rmsf_sq = (mean_sq - mean_posit.dot(mean_posit)).max(0.0);
        let max_deviation_sq = samples
            .iter()
            .map(|posit| (*posit - mol.atoms[atom_i].posit).magnitude_squared())
            .fold(0.0, f64::max);
        let dists: Vec<_> = samples
            .iter()
            .map(|posit| (*posit - reference_point).magnitude())
            .collect();
        let mean_distance_from_reference = mean(&dists) as f32;

        result.push(PositSample {
            samples,
            mean_posit,
            rmsf: rmsf_sq.sqrt() as f32,
            max_deviation: max_deviation_sq.sqrt() as f32,
            distance_from_reference: Histogram1D::from_values(&dists, DEFAULT_ATOM_DIST_HIST_BINS),
            mean_distance_from_reference,
            flexibility_depth: flexibility_depth[atom_i],
        });
    }

    result
}

fn build_global_metrics(ensemble: &[Vec<Vec3>], mol: &MoleculeCommon) -> GlobalConformationMetrics {
    let heavy_atoms: Vec<_> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.element != Hydrogen)
        .map(|(i, _)| i)
        .collect();
    let shape_atoms: Vec<_> = if heavy_atoms.is_empty() {
        (0..mol.atoms.len()).collect()
    } else {
        heavy_atoms
    };

    let base_centroid = centroid_of_indices(&ensemble[0], &shape_atoms);

    let mut radius_of_gyration = Vec::with_capacity(ensemble.len());
    let mut max_radius = Vec::with_capacity(ensemble.len());
    let mut shape_anisotropy = Vec::with_capacity(ensemble.len());
    let mut centroid_drift = Vec::with_capacity(ensemble.len());

    for positions in ensemble {
        let centroid = centroid_of_indices(positions, &shape_atoms);
        let mut rg_sq_sum = 0.0;
        let mut max_r_sq = 0.0f64;

        let mut g_xx = 0.0;
        let mut g_xy = 0.0;
        let mut g_xz = 0.0;
        let mut g_yy = 0.0;
        let mut g_yz = 0.0;
        let mut g_zz = 0.0;

        for &atom_i in &shape_atoms {
            let d = positions[atom_i] - centroid;
            let d_sq = d.magnitude_squared();
            rg_sq_sum += d_sq;
            max_r_sq = max_r_sq.max(d_sq);

            g_xx += d.x * d.x;
            g_xy += d.x * d.y;
            g_xz += d.x * d.z;
            g_yy += d.y * d.y;
            g_yz += d.y * d.z;
            g_zz += d.z * d.z;
        }

        let inv_n = 1.0 / shape_atoms.len().max(1) as f64;
        g_xx *= inv_n;
        g_xy *= inv_n;
        g_xz *= inv_n;
        g_yy *= inv_n;
        g_yz *= inv_n;
        g_zz *= inv_n;

        let trace = g_xx + g_yy + g_zz;
        let trace_sq = g_xx * g_xx
            + g_yy * g_yy
            + g_zz * g_zz
            + 2.0 * (g_xy * g_xy + g_xz * g_xz + g_yz * g_yz);
        let i2 = 0.5 * (trace * trace - trace_sq);
        let anisotropy = if trace <= 1.0e-12 {
            0.0
        } else {
            (1.0 - 3.0 * i2 / (trace * trace)).clamp(0.0, 1.0)
        };

        radius_of_gyration.push((rg_sq_sum * inv_n).sqrt());
        max_radius.push(max_r_sq.sqrt());
        shape_anisotropy.push(anisotropy);
        centroid_drift.push((centroid - base_centroid).magnitude());
    }

    GlobalConformationMetrics {
        mean_radius_of_gyration: mean(&radius_of_gyration) as f32,
        mean_max_radius_from_centroid: mean(&max_radius) as f32,
        mean_shape_anisotropy: mean(&shape_anisotropy) as f32,
        mean_centroid_drift: mean(&centroid_drift) as f32,
        radius_of_gyration: Histogram1D::from_values(&radius_of_gyration, DEFAULT_GLOBAL_HIST_BINS),
        max_radius_from_centroid: Histogram1D::from_values(&max_radius, DEFAULT_GLOBAL_HIST_BINS),
        shape_anisotropy: Histogram1D::from_values(&shape_anisotropy, DEFAULT_GLOBAL_HIST_BINS),
        centroid_drift: Histogram1D::from_values(&centroid_drift, DEFAULT_GLOBAL_HIST_BINS),
    }
}

fn centroid_of_indices(positions: &[Vec3], indices: &[usize]) -> Vec3 {
    if indices.is_empty() {
        return Vec3::new_zero();
    }

    let sum = indices
        .iter()
        .copied()
        .fold(Vec3::new_zero(), |acc, idx| acc + positions[idx]);
    sum / indices.len() as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

// todo: No. Replace this with the one in lin alg.
fn dihedral_angle(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Option<f64> {
    let b0 = p1 - p0;
    let b1 = p2 - p1;
    let b2 = p3 - p2;

    let n0 = b0.cross(b1);
    let n1 = b1.cross(b2);

    let n0_mag_sq = n0.magnitude_squared();
    let n1_mag_sq = n1.magnitude_squared();
    let b1_mag = b1.magnitude();

    if n0_mag_sq <= 1.0e-12 || n1_mag_sq <= 1.0e-12 || b1_mag <= 1.0e-12 {
        return None;
    }

    let b1_unit = b1 * (1.0 / b1_mag);
    let m1 = n0.cross(b1_unit);
    let x = n0.dot(n1);
    let y = m1.dot(n1);

    Some(y.atan2(x))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, f64::consts::PI};

    use bio_files::{
        BondType,
        md_params::{DihedralParams, ForceFieldParams},
    };
    use lin_alg::f64::Vec3;
    use na_seq::Element::Carbon;

    use super::{Conformer, characterize_conformations};
    use crate::molecules::{Atom, Bond, common::MoleculeCommon, small::MoleculeSmall};

    fn chain_molecule() -> MoleculeSmall {
        let atoms = vec![
            Atom {
                serial_number: 1,
                posit: Vec3::new(0.0, 0.0, 0.0),
                element: Carbon,
                force_field_type: Some("c3".to_string()),
                hetero: true,
                ..Default::default()
            },
            Atom {
                serial_number: 2,
                posit: Vec3::new(1.54, 0.0, 0.0),
                element: Carbon,
                force_field_type: Some("c3".to_string()),
                hetero: true,
                ..Default::default()
            },
            Atom {
                serial_number: 3,
                posit: Vec3::new(2.94, 1.0, 0.0),
                element: Carbon,
                force_field_type: Some("c3".to_string()),
                hetero: true,
                ..Default::default()
            },
            Atom {
                serial_number: 4,
                posit: Vec3::new(4.10, 1.2, 1.0),
                element: Carbon,
                force_field_type: Some("c3".to_string()),
                hetero: true,
                ..Default::default()
            },
        ];

        let bonds = vec![
            Bond {
                bond_type: BondType::Single,
                atom_0_sn: 1,
                atom_1_sn: 2,
                atom_0: 0,
                atom_1: 1,
                is_backbone: false,
            },
            Bond {
                bond_type: BondType::Single,
                atom_0_sn: 2,
                atom_1_sn: 3,
                atom_0: 1,
                atom_1: 2,
                is_backbone: false,
            },
            Bond {
                bond_type: BondType::Single,
                atom_0_sn: 3,
                atom_1_sn: 4,
                atom_0: 2,
                atom_1: 3,
                is_backbone: false,
            },
        ];

        let mut mol = MoleculeSmall {
            common: MoleculeCommon::new(
                "butane_like".to_string(),
                atoms,
                bonds,
                HashMap::new(),
                None,
            ),
            ..Default::default()
        };
        mol.update_characterization();
        mol
    }

    fn ff_params() -> ForceFieldParams {
        let mut ff = ForceFieldParams::default();
        ff.dihedral.insert(
            (
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
            ),
            vec![DihedralParams {
                atom_types: (
                    "c3".to_string(),
                    "c3".to_string(),
                    "c3".to_string(),
                    "c3".to_string(),
                ),
                divider: 1,
                barrier_height: 0.9,
                phase: PI as f32,
                periodicity: 3,
                comment: None,
            }],
        );
        ff
    }

    #[test]
    fn characterize_conformations_builds_rotor_profile() {
        let mol = chain_molecule();
        let conformer = characterize_conformations(&mol, &ff_params()).unwrap();

        assert_eq!(conformer.rotatable_bonds.len(), 1);
        assert!(conformer.sample_count >= 1);
        assert_eq!(conformer.atom_samples.len(), mol.common.atoms.len());

        let angle_hist = &conformer.rotatable_bonds[0].angle_distribution.bins;
        let prob_sum: f32 = angle_hist.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1.0e-4);

        assert!(conformer.atom_samples[3].rmsf > conformer.atom_samples[0].rmsf);
    }

    #[test]
    fn characterize_conformations_handles_rigid_molecules() {
        let mut mol = chain_molecule();
        mol.common.bonds[1].bond_type = BondType::Double;
        mol.update_characterization();

        let conformer: Conformer = characterize_conformations(&mol, &ff_params()).unwrap();
        assert_eq!(conformer.rotatable_bonds.len(), 0);
        assert_eq!(conformer.sample_count, 1);

        for atom_sample in &conformer.atom_samples {
            assert_eq!(atom_sample.samples.len(), 1);
        }
    }
}
