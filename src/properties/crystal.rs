//! For representing small molecules in homogenous crystal lattices. This has implications
//! for measuring self-affinity, for example, for the purposes of assessing solubility in water, or other
//! solvents. It models and infers properties about how, for example, a drug-like molecule
//! might exist as a crystalline powder.

use std::{
    collections::{HashMap, HashSet},
    io,
    io::ErrorKind,
};

use bio_files::{
    gromacs::{MoleculeInput, OutputControl},
    md_params::ForceFieldParams,
};
use dynamics::{
    BarostatCfg, ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MolDynamics,
    SimBoxInit, Solvent, TAU_TEMP_DEFAULT, alchemical,
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::f32::Vec3;

use crate::{
    gromacs::{make_gromacs_input, molecule_input_from_packed_copies},
    md::{MdBackend, build_dynamics, run_dynamics_blocking, setup_mols_dyn},
    molecules::small::MoleculeSmall,
    properties::{
        AMU_A3_TO_G_CM3, io_error, mean, min_image, mol_bounding_radius,
        mol_characterization::MolCharacterization, prepare_mol_for_md,
    },
};

// todo: Consider making this dynamic once basic functionality in this module works. I.e., run until
// todo: a crystal structure is stable, and use this as an upper bound.
const NUM_STEPS: usize = 20_000;
const SNAPSHOT_INTERVAL: usize = 10;
const TEMPERATURE: f32 = 300.; // K. todo: Set A/R
const PRESSURE: f32 = 1.; // Bar. todo: A/R.
const DT: f32 = 0.002; // ps

const DEFAULT_CRYSTAL_DENSITY_G_CM3: f32 = 1.20;
const MIN_CRYSTAL_DENSITY_G_CM3: f32 = 0.85;
const MAX_CRYSTAL_DENSITY_G_CM3: f32 = 1.65;
const MIN_CRYSTAL_BOX_SIDE_A: f32 = 36.;

// todo: Currently, we are having a difficult time packing. Given we probably don't want
// todo: PBCs for this (Unlike, for example, a solvent) (Or do we), just start with a large number
// todo of copies in a big box, and let it collapse into a crystal.

const CRYSTAL_MD_SYSTEM_SIZE_MULTIPLIER: usize = 1;

const BASE_MIN_COPIES_FOR_MD: usize = 32;
const BASE_MAX_COPIES_FOR_MD: usize = 96;
const BASE_MAX_ATOMS_FOR_CRYSTAL_MD: usize = 2_000;
const MIN_COPIES_FOR_MD: usize = BASE_MIN_COPIES_FOR_MD * CRYSTAL_MD_SYSTEM_SIZE_MULTIPLIER;
const MAX_COPIES_FOR_MD: usize = BASE_MAX_COPIES_FOR_MD * CRYSTAL_MD_SYSTEM_SIZE_MULTIPLIER;
const MAX_ATOMS_FOR_CRYSTAL_MD: usize =
    BASE_MAX_ATOMS_FOR_CRYSTAL_MD * CRYSTAL_MD_SYSTEM_SIZE_MULTIPLIER;

const CRYSTAL_INIT_RELAXATION_ITERS: usize = 80;
const MIN_WALL_PADDING_A: f32 = 2.;
const CRYSTAL_GRID_WALL_MARGIN_A: f32 = 0.6;

/// How the data was estimated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CrystalEstimateSource {
    Properties,
    MolecularDynamics,
}

/// todo: RM A/R
#[derive(Clone, Debug, Default)]
pub struct CrystalDataMdProperties {
    /// Number of molecule copies represented by the MD cell.
    pub copy_count: usize,
    /// Density used to size the fixed-volume dry crystal cell.
    pub target_density_g_cm3: f32,
    /// Cubic simulation-cell side length.
    pub box_side_a: f32,
    // pub pressure_target_bar: f32,
    pub density_g_cm3: f32,
    /// Molecular volume divided by volume available per molecule. This is a rough packing
    /// fraction because the molecular volume descriptor is itself approximate.
    pub packing_fraction_proxy: f32,
    pub volume_per_molecule_a3: f32,
    pub potential_energy_per_mol_kcal: f32,
    pub nonbonded_energy_per_mol_kcal: f32,
    /// Inter-molecular non-bonded energy per molecule. Dynamics snapshots use alchemical
    /// fully-coupled interaction bookkeeping; GROMACS snapshots fall back to pair energies.
    /// More negative means stronger cohesion.
    pub cohesive_energy_per_mol_kcal: f32,
    pub mean_pair_interaction_kcal: f32,
    pub nearest_neighbor_distance_a: f32,
    pub coordination_number: f32,
    pub h_bonds_per_molecule: f32,
    /// 0.5 is random uncorrelated axes; 1.0 is highly aligned.
    pub orientational_order: f32,
}

/// Contains self-affinity, packing, and related results for a small organic molecule in a
/// pure-molecule crystal-like environment.
///
/// `self_affinity_score` is dimensionless: higher means the molecule is expected to bind to
/// itself more strongly in the pure solid. It is intentionally separated from
/// `water_affinity_proxy`, so a later solubility model can combine solid-state cohesion with
/// molecule-water affinity instead of conflating the two.
#[derive(Clone, Debug)]
pub struct CrystalData {
    pub source: CrystalEstimateSource,
    pub self_affinity_score: f32,
    /// Higher means stronger crystal/self-binding pressure against dissolution.
    pub crystal_solubility_penalty: f32,
    /// Fast descriptor-only component of `self_affinity_score`.
    pub property_self_affinity_score: f32,
    /// Fast proxy for water affinity; useful as a competing term in solubility estimates.
    pub water_affinity_proxy: f32,
    pub h_bond_capacity: f32,
    pub hydrophobicity: f32,
    pub aromatic_stacking_propensity: f32,
    pub flexibility_penalty: f32,
    /// i.e., if the [slower] MD pipeline was run in addition to the analytic one.
    pub md_properties: Option<CrystalDataMdProperties>,
}

struct PropertyTerms {
    self_affinity: f32,
    water_affinity: f32,
    h_bond_capacity: f32,
    hydrophobicity: f32,
    aromatic_stacking: f32,
    flexibility_penalty: f32,
    crystal_solubility_penalty: f32,
}

fn property_terms(char: &MolCharacterization) -> PropertyTerms {
    let donors = char.h_bond_donor.len() as f32;
    let acceptors = char.h_bond_acceptor.len() as f32;
    let h_bond_capacity = donors + acceptors;

    let hydrophobicity = ((char.log_p + 1.0) / 4.0).clamp(0.0, 2.5);

    let aromatic_stacking = char.num_aromatic_atoms as f32 * 0.06
        + char.num_rings_aromatic as f32 * 0.25
        + char.ring_systems.len() as f32 * 0.30;

    let polar_contact = donors.min(6.0) * 0.22
        + acceptors.min(8.0) * 0.14
        + (char.tpsa_ertl / 120.0).min(1.5) * 0.40;

    let polarizability = (char.molar_refractivity / 80.0).clamp(0.0, 2.0)
        + char.halogen.len() as f32 * 0.08
        + char.sulfur.len() as f32 * 0.12
        + char.phosphorus.len() as f32 * 0.12;

    let size = (char.mol_weight / 350.0).clamp(0.1, 2.0);
    let flexibility_penalty =
        char.rotatable_bonds.len() as f32 * 0.08 + (char.flexibility / 8.0).clamp(0.0, 1.5);
    let abs_charge = char.abs_partial_charge_sum.unwrap_or_default();

    let self_affinity =
        (size + hydrophobicity + aromatic_stacking + polar_contact + polarizability
            - flexibility_penalty)
            .max(0.0);

    let water_affinity = (char.tpsa_ertl / 90.0).clamp(0.0, 2.5)
        + donors * 0.22
        + acceptors * 0.14
        + abs_charge * 0.50
        - char.log_p.max(0.0) * 0.15;

    let crystal_solubility_penalty = self_affinity - water_affinity * 0.60;

    PropertyTerms {
        self_affinity,
        water_affinity,
        h_bond_capacity,
        hydrophobicity,
        aromatic_stacking,
        flexibility_penalty,
        crystal_solubility_penalty,
    }
}

fn crystal_data_from_properties(
    char: &MolCharacterization,
    source: CrystalEstimateSource,
) -> CrystalData {
    let terms = property_terms(char);

    CrystalData {
        source,
        self_affinity_score: terms.self_affinity,
        crystal_solubility_penalty: terms.crystal_solubility_penalty,
        property_self_affinity_score: terms.self_affinity,
        water_affinity_proxy: terms.water_affinity,
        h_bond_capacity: terms.h_bond_capacity,
        hydrophobicity: terms.hydrophobicity,
        aromatic_stacking_propensity: terms.aromatic_stacking,
        flexibility_penalty: terms.flexibility_penalty,
        md_properties: None,
    }
}

fn mol_mass_from_atoms(mol: &MoleculeSmall) -> f32 {
    mol.common
        .atoms
        .iter()
        .map(|a| a.element.atomic_weight())
        .sum()
}

#[derive(Clone, Copy, Debug)]
struct CrystalMdSetup {
    copy_count: usize,
    requested_copy_count: usize,
    box_side: f32,
    target_density_g_cm3: f32,
}

fn estimated_crystal_packing_fraction(char: &MolCharacterization) -> f32 {
    let h_bond_sites = char.h_bond_donor.len() + char.h_bond_acceptor.len();
    let aromatic_bonus = (char.num_rings_aromatic as f32 * 0.010
        + char.ring_systems.len() as f32 * 0.015)
        .min(0.045);
    let h_bond_bonus = (h_bond_sites as f32 * 0.006).min(0.035);
    let flexibility_penalty =
        (char.rotatable_bonds.len() as f32 * 0.008 + char.flexibility * 0.004).min(0.080);

    (0.68 + aromatic_bonus + h_bond_bonus - flexibility_penalty).clamp(0.58, 0.76)
}

fn target_crystal_density_g_cm3(mol: &MoleculeSmall, char: &MolCharacterization) -> f32 {
    let mass = char.mol_weight.max(mol_mass_from_atoms(mol)).max(1.0);
    let mol_volume = char
        .volume_pubchem
        .filter(|v| *v > 0.0)
        .unwrap_or(char.volume);

    if mol_volume > 0.0 {
        let packing_fraction = estimated_crystal_packing_fraction(char);
        (mass * AMU_A3_TO_G_CM3 * packing_fraction / mol_volume)
            .clamp(MIN_CRYSTAL_DENSITY_G_CM3, MAX_CRYSTAL_DENSITY_G_CM3)
    } else {
        DEFAULT_CRYSTAL_DENSITY_G_CM3
    }
}

fn min_crystal_box_side(mol: &MoleculeSmall) -> f32 {
    let diameter_with_margin = mol_bounding_radius(mol) * 2.0 + MIN_WALL_PADDING_A * 2.0;
    MIN_CRYSTAL_BOX_SIDE_A.max(diameter_with_margin)
}

fn box_side_for_density(mass: f32, copy_count: usize, density_g_cm3: f32) -> f32 {
    let target_density_amu_a3 = density_g_cm3 / AMU_A3_TO_G_CM3;
    (mass * copy_count as f32 / target_density_amu_a3).cbrt()
}

fn grid_safe_box_side(copy_count: usize, mol_radius: f32) -> f32 {
    let required_half_cell = mol_radius + CRYSTAL_GRID_WALL_MARGIN_A;
    if required_half_cell <= 0.0 {
        return 0.0;
    }

    let grid_dim = (copy_count as f32).cbrt().ceil().max(3.0);
    grid_dim * 2.0 * required_half_cell
}

fn bounded_crystal_copy_count(requested_copy_count: usize, atom_count: usize) -> usize {
    let atom_count = atom_count.max(1);
    let atom_limited_cap = (MAX_ATOMS_FOR_CRYSTAL_MD / atom_count).max(1);
    let copy_cap = MAX_COPIES_FOR_MD.min(atom_limited_cap).max(1);

    requested_copy_count
        .min(copy_cap)
        .max(MIN_COPIES_FOR_MD.min(copy_cap))
}

fn crystal_md_setup(mol: &MoleculeSmall, char: &MolCharacterization) -> CrystalMdSetup {
    let mass = char.mol_weight.max(mol_mass_from_atoms(mol)).max(1.0);
    let target_density_g_cm3 = target_crystal_density_g_cm3(mol, char);
    let target_density_amu_a3 = target_density_g_cm3 / AMU_A3_TO_G_CM3;
    let min_box_side = min_crystal_box_side(mol);
    let mol_radius = mol_bounding_radius(mol);

    let base_requested_copy_count = ((target_density_amu_a3 * min_box_side.powi(3)) / mass)
        .ceil()
        .max(BASE_MIN_COPIES_FOR_MD as f32) as usize;
    let requested_copy_count =
        base_requested_copy_count.saturating_mul(CRYSTAL_MD_SYSTEM_SIZE_MULTIPLIER);
    let copy_count = bounded_crystal_copy_count(requested_copy_count, mol.common.atoms.len());
    let box_side = box_side_for_density(mass, copy_count, target_density_g_cm3)
        .max(grid_safe_box_side(copy_count, mol_radius))
        .max(min_box_side);

    CrystalMdSetup {
        copy_count,
        requested_copy_count,
        box_side,
        target_density_g_cm3,
    }
}

fn build_md_cfg(box_side: f32) -> MdConfig {
    MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        zero_com_drift: true,
        temp_target: TEMPERATURE,
        barostat_cfg: Some(BarostatCfg {
            pressure_target: PRESSURE,
            ..Default::default()
        }),
        snapshot_handlers: SnapshotHandlers {
            memory: Some(SNAPSHOT_INTERVAL),
            dcd: None,
            gromacs: OutputControl::default(),
        },
        sim_box: SimBoxInit::new_cube(box_side),
        solvent: Solvent::None,
        max_init_relaxation_iters: Some(CRYSTAL_INIT_RELAXATION_ITERS),
        recenter_sim_box: false,
        overrides: MdOverrides {
            skip_water_relaxation: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn build_gromacs_md_cfg(box_side: f32) -> MdConfig {
    let mut cfg = build_md_cfg(box_side);

    cfg.snapshot_handlers.memory = None;

    cfg.snapshot_handlers.gromacs = OutputControl {
        nstxout: Some(SNAPSHOT_INTERVAL as u32),
        nstcalcenergy: Some(SNAPSHOT_INTERVAL as u32),
        nstenergy: Some(SNAPSHOT_INTERVAL as u32),
        ..Default::default()
    };
    cfg
}

fn cohesive_energy_from_matrix(matrix: &[f32], n_mol: usize) -> Option<(f32, f32)> {
    if n_mol < 2 || matrix.len() < n_mol * n_mol {
        return None;
    }

    let mut total_pair_energy = 0.0;
    let mut pair_count = 0;

    for i in 0..n_mol {
        for j in (i + 1)..n_mol {
            total_pair_energy += matrix[i * n_mol + j];
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        return None;
    }

    Some((
        total_pair_energy / n_mol as f32,
        total_pair_energy / pair_count as f32,
    ))
}

fn molecule_centroids_and_axes(
    atom_posits: &[Vec3],
    mol_start_indices: &[usize],
    n_mol: usize,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut centroids = Vec::with_capacity(n_mol);
    let mut axes = Vec::with_capacity(n_mol);

    for mol_i in 0..n_mol {
        let start = mol_start_indices[mol_i];
        let end = mol_start_indices
            .get(mol_i + 1)
            .copied()
            .unwrap_or(atom_posits.len());

        if start >= end || end > atom_posits.len() {
            continue;
        }

        let mut centroid = Vec3::new_zero();
        for posit in &atom_posits[start..end] {
            centroid += *posit;
        }
        centroid /= (end - start) as f32;

        let mut axis = Vec3::new(1.0, 0.0, 0.0);
        let mut max_dist_sq = 0.0;
        for posit in &atom_posits[start..end] {
            let v = *posit - centroid;
            let dist_sq = v.magnitude_squared();
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
                axis = v;
            }
        }
        if max_dist_sq > 1.0e-6 {
            axis = axis.to_normalized();
        }

        centroids.push(centroid);
        axes.push(axis);
    }

    (centroids, axes)
}

fn structural_metrics(
    atom_posits: &[Vec3],
    mol_start_indices: &[usize],
    extent: Vec3,
    n_mol: usize,
) -> (Option<f32>, Option<f32>, Option<f32>) {
    if n_mol < 2 || mol_start_indices.len() < n_mol {
        return (None, None, None);
    }

    let (centroids, axes) = molecule_centroids_and_axes(atom_posits, mol_start_indices, n_mol);
    if centroids.len() < 2 {
        return (None, None, None);
    }

    let mut nearest = vec![f32::INFINITY; centroids.len()];

    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let dist = min_image(centroids[j] - centroids[i], extent).magnitude();
            nearest[i] = nearest[i].min(dist);
            nearest[j] = nearest[j].min(dist);
        }
    }

    let nearest_finite: Vec<_> = nearest.into_iter().filter(|v| v.is_finite()).collect();
    let Some(nearest_neighbor_distance) = mean(&nearest_finite) else {
        return (None, None, None);
    };

    let coordination_cutoff = nearest_neighbor_distance * 1.25 + 0.5;
    let mut coordination = vec![0usize; centroids.len()];
    let mut order_sum = 0.0;
    let mut order_count = 0;

    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let dist = min_image(centroids[j] - centroids[i], extent).magnitude();
            if dist <= coordination_cutoff {
                coordination[i] += 1;
                coordination[j] += 1;
                order_sum += axes[i].dot(axes[j]).abs();
                order_count += 1;
            }
        }
    }

    let coordination_number = coordination.iter().sum::<usize>() as f32 / coordination.len() as f32;
    let orientational_order = if order_count > 0 {
        Some(order_sum / order_count as f32)
    } else {
        None
    };

    (
        Some(nearest_neighbor_distance),
        Some(coordination_number),
        orientational_order,
    )
}

/// This is what we use to collect properties on self-affinity after the MD run.
fn add_md_metrics(
    data: &mut CrystalData,
    char: &MolCharacterization,
    snapshots: &[Snapshot],
    atom_posits: &[Vec3],
    mol_start_indices: &[usize],
    cell_extent: Vec3,
    setup: CrystalMdSetup,
) {
    let n_mol = mol_start_indices.len().min(setup.copy_count);
    if n_mol == 0 {
        return;
    }

    let mut metrics = CrystalDataMdProperties::default();

    metrics.copy_count = n_mol;
    metrics.target_density_g_cm3 = setup.target_density_g_cm3;
    metrics.box_side_a = setup.box_side;

    // metrics.temperature_k = TEMPERATURE;
    // metrics.pressure_target_bar = PRESSURE;

    let snaps = if snapshots.len() > 4 {
        &snapshots[snapshots.len() / 2..]
    } else {
        snapshots
    };

    let mut potentials = Vec::new();
    let mut nonbonded = Vec::new();
    let mut pair_cohesive = Vec::new();
    let mut pair_interactions = Vec::new();
    let mut pressures = Vec::new();
    let mut temperatures = Vec::new();
    let mut densities = Vec::new();
    let mut volumes_per_mol = Vec::new();
    let mut h_bonds = Vec::new();

    for snap in snaps {
        let Some(e) = &snap.energy_data else {
            continue;
        };

        potentials.push(e.energy_potential / n_mol as f32);
        nonbonded.push(e.energy_potential_nonbonded / n_mol as f32);
        pressures.push(e.pressure);
        temperatures.push(e.temperature);
        densities.push(e.density * AMU_A3_TO_G_CM3);
        volumes_per_mol.push(e.volume / n_mol as f32);
        h_bonds.push(e.hydrogen_bonds.len() as f32 / n_mol as f32);

        if let Some((cohesive_per_mol, mean_pair)) =
            cohesive_energy_from_matrix(&e.energy_potential_between_mols, n_mol)
        {
            pair_cohesive.push(cohesive_per_mol);
            pair_interactions.push(mean_pair);
        }
    }

    metrics.potential_energy_per_mol_kcal = mean(&potentials).unwrap_or(0.0);
    metrics.nonbonded_energy_per_mol_kcal = mean(&nonbonded).unwrap_or(0.0);
    metrics.cohesive_energy_per_mol_kcal = alchemical::mean_coupled_interaction_kcal(snaps)
        .or_else(|| mean(&pair_cohesive))
        .unwrap_or(0.0);

    metrics.mean_pair_interaction_kcal = mean(&pair_interactions).unwrap_or(0.0);

    metrics.density_g_cm3 = mean(&densities).unwrap_or(setup.target_density_g_cm3);
    metrics.volume_per_molecule_a3 =
        mean(&volumes_per_mol).unwrap_or_else(|| setup.box_side.powi(3) / n_mol as f32);
    metrics.h_bonds_per_molecule = mean(&h_bonds).unwrap_or(0.0);

    if metrics.volume_per_molecule_a3 > 0.0 && char.volume > 0.0 {
        metrics.packing_fraction_proxy =
            Some((char.volume / metrics.volume_per_molecule_a3).clamp(0.0, 1.5)).unwrap_or(0.0);
    }

    let (nearest, coordination, orientational_order) =
        structural_metrics(atom_posits, mol_start_indices, cell_extent, n_mol);

    metrics.nearest_neighbor_distance_a = nearest.unwrap_or(0.0);
    metrics.coordination_number = coordination.unwrap_or(0.0);
    metrics.orientational_order = orientational_order.unwrap_or(0.0);

    // todo: Magic?
    let md_self_affinity = metrics.cohesive_energy_per_mol_kcal
        + metrics.density_g_cm3 * 0.20
        + metrics.h_bonds_per_molecule * 0.10
        + metrics.coordination_number * 0.05;

    data.self_affinity_score = data.property_self_affinity_score + md_self_affinity;
    data.crystal_solubility_penalty = data.self_affinity_score - data.water_affinity_proxy * 0.60;

    data.md_properties = Some(metrics);
}

fn gromacs_crystal_molecule_input(
    placed_mols: &[MolDynamics],
    ident: &str,
) -> io::Result<(MoleculeInput, Vec<usize>)> {
    let input = molecule_input_from_packed_copies(ident.to_owned(), placed_mols)?;
    if input.ff_params.is_none() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing molecule-specific parameters for GROMACS crystal input.",
        ));
    }

    let atom_count = input.atoms.len();
    let mol_start_indices = (0..input.count).map(|copy_i| copy_i * atom_count).collect();

    Ok((input, mol_start_indices))
}

/// Launch using the dynamics backend.
fn run_dynamics(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    char: &MolCharacterization,
    setup: CrystalMdSetup,
    dev: &ComputationDevice,
) -> io::Result<(CrystalData, Vec<Snapshot>)> {
    let cfg = build_md_cfg(setup.box_side);

    let mols = vec![(FfMolType::SmallOrganic, &mol.common, setup.copy_count)];
    let (mut md, _) = build_dynamics(
        dev,
        &mols,
        param_set,
        mol_specific_params,
        &cfg,
        false,
        None,
        &mut HashSet::new(),
    )
    .map_err(|e| io::Error::other(e.descrip))?;

    md.configure_alchemical_window(dev, 0, 0.)
        .map_err(|e| io_error("Unable to configure crystal alchemical bookkeeping", e))?;

    run_dynamics_blocking(&mut md, &dev, DT, NUM_STEPS);

    let atom_posits: Vec<_> = md.atoms.iter().map(|atom| atom.posit).collect();
    let mut data = crystal_data_from_properties(char, CrystalEstimateSource::MolecularDynamics);

    add_md_metrics(
        &mut data,
        char,
        &md.snapshots,
        &atom_posits,
        &md.mol_start_indices,
        md.cell.extent,
        setup,
    );

    Ok((data, md.snapshots))
}

/// Launch using the GROMACS backend.
fn run_gromacs(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    char: &MolCharacterization,
    mut setup: CrystalMdSetup,
) -> io::Result<(CrystalData, Vec<Snapshot>)> {
    let cfg = build_gromacs_md_cfg(setup.box_side);
    let mdp = cfg.to_gromacs(NUM_STEPS, DT);

    let mols = vec![(FfMolType::SmallOrganic, &mol.common, setup.copy_count)];

    let (placed_mols, _) = setup_mols_dyn(
        &mols,
        mol_specific_params,
        false,
        None,
        Some((setup.box_side, setup.box_side, setup.box_side)),
    )
    .map_err(io::Error::other)?;

    if placed_mols.is_empty() {
        return Err(io::Error::other(
            "Crystal GROMACS setup did not place any molecule copies.",
        ));
    }

    setup.copy_count = placed_mols.len();
    let (mol_input, mol_start_indices) =
        gromacs_crystal_molecule_input(&placed_mols, &mol.common.ident)?;

    let input = make_gromacs_input(
        mdp,
        &[FfMolType::SmallOrganic],
        vec![mol_input],
        param_set,
        &cfg.sim_box,
        &cfg.solvent,
        cfg.max_init_relaxation_iters.is_some(),
    )?;

    let out = input.run()?;

    let snapshots = gromacs_frames_to_ss(&out);

    let last_atom_posits = snapshots
        .last()
        .map(|snap| snap.atom_posits.clone())
        .unwrap_or_default();

    let cell_extent = Vec3::new(setup.box_side, setup.box_side, setup.box_side);
    let mut data = crystal_data_from_properties(char, CrystalEstimateSource::MolecularDynamics);

    add_md_metrics(
        &mut data,
        char,
        &snapshots,
        &last_atom_posits,
        &mol_start_indices,
        cell_extent,
        setup,
    );

    Ok((data, snapshots))
}

/// Runs a molecular dynamics simulation of a number of copies of the molecule being analyzed,
/// with no solvent. Returns both crystal/self-affinity descriptors and snapshots that can be
/// used to visualize the dry pure-molecule run.
pub fn run_crystal_sim(
    mol: &MoleculeSmall,
    backend: MdBackend,
    dev: &ComputationDevice,
    param_set: &FfParamSet,
) -> io::Result<(CrystalData, Vec<Snapshot>)> {
    let (mol, mol_specific_params) = prepare_mol_for_md(mol, &param_set)?;

    let Some(char) = &mol.characterization else {
        return Err(io::Error::other(
            "Char missing when estimating crystal data",
        ));
    };

    let setup = crystal_md_setup(&mol, &char);

    println!(
        "Crystal MD setup ({backend}): density target {:.2} g/cm^3, box side {:.1} A, copies {}",
        setup.target_density_g_cm3, setup.box_side, setup.copy_count
    );

    if setup.copy_count < setup.requested_copy_count {
        println!(
            "Crystal MD setup: capped density-derived copies from {} to {} to keep packing \
             inside the box and bound initial minimization work.",
            setup.requested_copy_count, setup.copy_count
        );
    }

    match backend {
        MdBackend::Dynamics => {
            run_dynamics(&mol, &param_set, &mol_specific_params, &char, setup, dev)
        }
        MdBackend::Gromacs => run_gromacs(&mol, &param_set, &mol_specific_params, &char, setup),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Crystal MD estimation supports the Dynamics and GROMACS backends.",
        )),
    }
}
