//! For, using MD for example, assess a molecule's properties in water, including
//! its affinity for the water. This can be used, for example, in contrast with its self-affinity,
//! or affinity for other solvents.
//!

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
    BarostatCfg, ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, ParamError,
    SimBoxInit, Solvent, TAU_TEMP_DEFAULT,
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::f32::Vec3;
use na_seq::Element;

use crate::{
    md::{MdBackend, build_dynamics, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
    properties::mol_characterization::MolCharacterization,
};

const NUM_STEPS: usize = 5_000;
const SNAPSHOT_INTERVAL: usize = 10;
const TEMPERATURE: f32 = 300.; // K. todo: Set A/R
const PRESSURE: f32 = 1.; // Bar. todo: A/R.
const DT: f32 = 0.002; // ps

const AMU_A3_TO_G_CM3: f32 = 1.660_539;
const FIRST_HYDRATION_SHELL_CUTOFF_A: f32 = 3.6;

const H_BOND_O_O_DIST: f32 = 2.7;
const H_BOND_N_N_DIST: f32 = 3.05;
const H_BOND_O_N_DIST: f32 = 2.9;
const H_BOND_N_F_DIST: f32 = 2.75;
const H_BOND_N_S_DIST: f32 = 3.35;
const H_BOND_DIST_THRESH: f32 = 0.3;
const H_BOND_ANGLE_THRESH: f32 = std::f32::consts::TAU / 3.;
const H_BOND_STRENGTH_DIST_MIN: f32 = 2.4;
const H_BOND_STRENGTH_DIST_MAX: f32 = 3.6;
const H_BOND_STRENGTH_ANGLE_MIN: f32 = std::f32::consts::PI * 2. / 3.;

/// How the data was estimated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterSolEstimateSource {
    Properties,
    MolecularDynamics,
}

/// todo: RM A/R
#[derive(Clone, Debug, Default)]
pub struct WaterSolDataMdProperties {
    /// Number of OPC water molecules represented by the simulation cell.
    pub water_molecule_count: usize,
    pub box_volume_a3: f32,
    pub box_min_side_a: f32,
    pub density_g_cm3: f32,
    pub mean_temperature_k: f32,
    pub mean_pressure_bar: f32,
    pub potential_energy_kcal: f32,
    pub nonbonded_energy_kcal: f32,
    /// Solute-environment interaction energy from the Dynamics alchemical bookkeeping.
    /// More negative means stronger attraction. For charged solutes this can include
    /// counter-ion interactions if neutralizing ions were added.
    pub solute_water_interaction_kcal: f32,
    /// Geometrically inferred solute-water hydrogen bonds per snapshot.
    pub water_h_bonds: f32,
    /// Solute donates a hydrogen bond to water.
    pub water_h_bonds_donated: f32,
    /// Water donates a hydrogen bond to the solute.
    pub water_h_bonds_accepted: f32,
    /// Average 0..1 geometry strength of inferred solute-water hydrogen bonds.
    pub mean_water_h_bond_strength: f32,
    pub nearest_water_o_distance_a: f32,
    pub first_shell_water_count: f32,
    pub first_shell_water_per_heavy_atom: f32,
    pub mean_first_shell_water_o_distance_a: f32,
}

/// Contains water-affinity and hydration results for a small organic molecule.
///
/// `water_affinity_score` is dimensionless: higher means stronger expected
/// molecule-water affinity. It is intentionally separated from crystal/self-affinity
/// so solubility models can combine this with `properties::crystal` instead of
/// conflating dissolution forces.
#[derive(Clone, Debug)]
pub struct WaterSolData {
    pub source: WaterSolEstimateSource,
    pub water_affinity_score: f32,
    /// Higher means the molecule is expected to be easier to hydrate or dissolve in water.
    pub water_solubility_score: f32,
    /// Fast descriptor-only component of `water_affinity_score`.
    pub property_water_affinity_score: f32,
    /// Higher means the molecule has hydrophobic or size-related resistance to hydration.
    pub hydration_penalty: f32,
    pub h_bond_capacity: f32,
    pub polarity_score: f32,
    pub hydrophobic_penalty: f32,
    pub charge_affinity: f32,
    pub tpsa_score: f32,
    /// i.e., if the [slower] MD pipeline was run in addition to the analytic one.
    pub md_properties: Option<WaterSolDataMdProperties>,
}

struct PropertyTerms {
    water_affinity: f32,
    water_solubility: f32,
    hydration_penalty: f32,
    h_bond_capacity: f32,
    polarity_score: f32,
    hydrophobic_penalty: f32,
    charge_affinity: f32,
    tpsa_score: f32,
}

#[derive(Default)]
struct SnapshotWaterMetrics {
    h_bonds: f32,
    h_bonds_donated: f32,
    h_bonds_accepted: f32,
    h_bond_strength_sum: f32,
    nearest_water_o_distance_a: Option<f32>,
    first_shell_water_count: f32,
    first_shell_water_o_distance_sum: f32,
}

fn param_err(e: ParamError) -> io::Error {
    io::Error::other(e.descrip)
}

fn property_terms(char: &MolCharacterization) -> PropertyTerms {
    let donors = char.h_bond_donor.len() as f32;
    let acceptors = char.h_bond_acceptor.len() as f32;
    let h_bond_capacity = donors + acceptors;

    let tpsa_score = (char.tpsa_ertl / 90.0).clamp(0.0, 2.5);
    let geom_psa_score = (char.psa_topo / 120.0).clamp(0.0, 1.5);
    let hetero_score =
        (char.num_hetero_atoms as f32 / char.num_heavy_atoms.max(1) as f32).clamp(0.0, 1.0);

    let polarity_score = tpsa_score + geom_psa_score * 0.35 + hetero_score * 0.40;
    let charge_affinity = char.abs_partial_charge_sum.unwrap_or_default() * 0.50
        + char.net_partial_charge.unwrap_or_default().abs() * 0.40;

    let hydrophobic_penalty = (char.log_p.max(0.0) * 0.35
        + (char.greasiness / 6.0).clamp(0.0, 1.5) * 0.40
        + char.hydrophobic_carbon.len() as f32 * 0.015)
        .clamp(0.0, 3.5);

    let size_penalty = (char.mol_weight / 550.0).clamp(0.0, 1.5) * 0.35
        + (char.molar_refractivity / 120.0).clamp(0.0, 1.5) * 0.15;
    let flexibility_penalty =
        char.rotatable_bonds.len() as f32 * 0.04 + (char.flexibility / 12.0).clamp(0.0, 1.2);

    let donor_score = donors.min(8.0) * 0.30;
    let acceptor_score = acceptors.min(10.0) * 0.18;

    let water_affinity = (polarity_score + donor_score + acceptor_score + charge_affinity
        - hydrophobic_penalty * 0.45
        - flexibility_penalty * 0.08)
        .max(0.0);

    let hydration_penalty = (hydrophobic_penalty + size_penalty + flexibility_penalty * 0.05
        - polarity_score * 0.30)
        .max(0.0);

    let water_solubility = (water_affinity - hydration_penalty).max(0.0);

    PropertyTerms {
        water_affinity,
        water_solubility,
        hydration_penalty,
        h_bond_capacity,
        polarity_score,
        hydrophobic_penalty,
        charge_affinity,
        tpsa_score,
    }
}

fn water_sol_data_from_properties(
    char: &MolCharacterization,
    source: WaterSolEstimateSource,
) -> WaterSolData {
    let terms = property_terms(char);

    WaterSolData {
        source,
        water_affinity_score: terms.water_affinity,
        water_solubility_score: terms.water_solubility,
        property_water_affinity_score: terms.water_affinity,
        hydration_penalty: terms.hydration_penalty,
        h_bond_capacity: terms.h_bond_capacity,
        polarity_score: terms.polarity_score,
        hydrophobic_penalty: terms.hydrophobic_penalty,
        charge_affinity: terms.charge_affinity,
        tpsa_score: terms.tpsa_score,
        md_properties: None,
    }
}

fn build_md_cfg() -> MdConfig {
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
        sim_box: SimBoxInit::default(),
        solvent: Solvent::WaterOpc,
        recenter_sim_box: true,
        overrides: MdOverrides::default(),
        ..Default::default()
    }
}

fn build_gromacs_md_cfg() -> MdConfig {
    let mut cfg = build_md_cfg();
    cfg.snapshot_handlers.memory = None;
    cfg.snapshot_handlers.gromacs = OutputControl {
        nstxout: Some(SNAPSHOT_INTERVAL as u32),
        nstcalcenergy: Some(SNAPSHOT_INTERVAL as u32),
        nstenergy: Some(SNAPSHOT_INTERVAL as u32),
        ..Default::default()
    };
    cfg
}

fn prepare_mol_for_md(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
) -> io::Result<(MoleculeSmall, HashMap<String, ForceFieldParams>)> {
    let Some(gaff2) = param_set.small_mol.as_ref() else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing GAFF2 small-molecule parameters.",
        ));
    };

    let mut mol = mol.clone();
    mol.common.selected_for_md = true;
    // This function has no access to the application's state-level frcmod cache, so force
    // a fresh local molecule-specific parameter inference for the water simulation.
    mol.frcmod_loaded = false;

    let mut mol_specific_params = HashMap::new();
    mol.update_ff_related(&mut mol_specific_params, gaff2, false);

    if !mol.ff_params_loaded || !mol.frcmod_loaded {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!(
                "Unable to infer force-field parameters for {}.",
                mol.common.ident
            ),
        ));
    }

    mol.update_characterization();

    Ok((mol, mol_specific_params))
}

fn mean(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f32>() / values.len() as f32)
    }
}

fn min_image(mut delta: Vec3, extent: Vec3) -> Vec3 {
    if extent.x > 0.0 {
        delta.x -= extent.x * (delta.x / extent.x).round();
    }
    if extent.y > 0.0 {
        delta.y -= extent.y * (delta.y / extent.y).round();
    }
    if extent.z > 0.0 {
        delta.z -= extent.z * (delta.z / extent.z).round();
    }

    delta
}

fn h_bond_candidate_el(element: Element) -> bool {
    matches!(
        element,
        Element::Nitrogen | Element::Oxygen | Element::Sulfur | Element::Fluorine
    )
}

fn h_bond_dist_threshold(donor_element: Element, acceptor_element: Element) -> f32 {
    if donor_element == Element::Oxygen && acceptor_element == Element::Oxygen {
        H_BOND_O_O_DIST
    } else if donor_element == Element::Nitrogen && acceptor_element == Element::Nitrogen {
        H_BOND_N_N_DIST
    } else if (donor_element == Element::Oxygen && acceptor_element == Element::Nitrogen)
        || (donor_element == Element::Nitrogen && acceptor_element == Element::Oxygen)
    {
        H_BOND_O_N_DIST
    } else if (donor_element == Element::Fluorine && acceptor_element == Element::Nitrogen)
        || (donor_element == Element::Nitrogen && acceptor_element == Element::Fluorine)
    {
        H_BOND_N_F_DIST
    } else {
        H_BOND_N_S_DIST
    }
}

fn h_bond_strength(donor_posit: Vec3, h_posit: Vec3, acc_posit: Vec3) -> f32 {
    let dist = (donor_posit - acc_posit).magnitude();
    let dist_score = ((H_BOND_STRENGTH_DIST_MAX - dist)
        / (H_BOND_STRENGTH_DIST_MAX - H_BOND_STRENGTH_DIST_MIN))
        .clamp(0., 1.);

    let vec_hd = (donor_posit - h_posit).to_normalized();
    let vec_ha = (acc_posit - h_posit).to_normalized();
    let angle = vec_hd.dot(vec_ha).clamp(-1., 1.).acos();

    let angle_score = ((angle - H_BOND_STRENGTH_ANGLE_MIN)
        / (std::f32::consts::PI - H_BOND_STRENGTH_ANGLE_MIN))
        .clamp(0., 1.);

    dist_score * angle_score
}

fn h_bond_strength_if_present(
    donor_posit: Vec3,
    h_posit: Vec3,
    acc_posit: Vec3,
    donor_element: Element,
    acceptor_element: Element,
    cell_extent: Vec3,
) -> Option<f32> {
    let dist_thresh = h_bond_dist_threshold(donor_element, acceptor_element);
    let dist_thresh_min = dist_thresh - H_BOND_DIST_THRESH;
    let dist_thresh_max = dist_thresh + H_BOND_DIST_THRESH;

    let donor_acc = min_image(acc_posit - donor_posit, cell_extent);
    let dist = donor_acc.magnitude();
    if dist < dist_thresh_min || dist > dist_thresh_max {
        return None;
    }

    let donor_h = min_image(h_posit - donor_posit, cell_extent);
    let donor_acceptor = donor_acc * -1.0;

    let angle = donor_acceptor
        .to_normalized()
        .dot(donor_h.to_normalized())
        .clamp(-1., 1.)
        .acos();

    if angle <= H_BOND_ANGLE_THRESH {
        return None;
    }

    let acc_imaged = donor_posit + donor_acc;
    Some(h_bond_strength(donor_posit, h_posit, acc_imaged))
}

fn solute_donor_candidates(
    mol: &MoleculeSmall,
    atom_posits: &[Vec3],
) -> Vec<(Vec3, Vec3, Element)> {
    let mut result = Vec::new();

    for bond in &mol.common.bonds {
        if bond.atom_0 >= atom_posits.len() || bond.atom_1 >= atom_posits.len() {
            continue;
        }

        let atom_0 = &mol.common.atoms[bond.atom_0];
        let atom_1 = &mol.common.atoms[bond.atom_1];

        if atom_0.element == Element::Hydrogen && h_bond_candidate_el(atom_1.element) {
            result.push((
                atom_posits[bond.atom_1],
                atom_posits[bond.atom_0],
                atom_1.element,
            ));
        } else if atom_1.element == Element::Hydrogen && h_bond_candidate_el(atom_0.element) {
            result.push((
                atom_posits[bond.atom_0],
                atom_posits[bond.atom_1],
                atom_0.element,
            ));
        }
    }

    result
}

fn solute_acceptor_candidates(mol: &MoleculeSmall, atom_posits: &[Vec3]) -> Vec<(Vec3, Element)> {
    mol.common
        .atoms
        .iter()
        .zip(atom_posits)
        .filter_map(|(atom, &posit)| {
            h_bond_candidate_el(atom.element).then_some((posit, atom.element))
        })
        .collect()
}

fn analyze_snapshot_water_contacts(
    mol: &MoleculeSmall,
    snap: &Snapshot,
    cell_extent: Vec3,
) -> SnapshotWaterMetrics {
    let solute_atom_count = mol.common.atoms.len().min(snap.atom_posits.len());
    if solute_atom_count == 0 || snap.water_o_posits.is_empty() {
        return SnapshotWaterMetrics::default();
    }

    let atom_posits = &snap.atom_posits[..solute_atom_count];
    let solute_donors = solute_donor_candidates(mol, atom_posits);
    let solute_acceptors = solute_acceptor_candidates(mol, atom_posits);

    let water_count = snap
        .water_o_posits
        .len()
        .min(snap.water_h0_posits.len())
        .min(snap.water_h1_posits.len());

    let mut metrics = SnapshotWaterMetrics::default();

    for &(donor, hydrogen, donor_element) in &solute_donors {
        for &water_o in snap.water_o_posits.iter().take(water_count) {
            if let Some(strength) = h_bond_strength_if_present(
                donor,
                hydrogen,
                water_o,
                donor_element,
                Element::Oxygen,
                cell_extent,
            ) {
                metrics.h_bonds += 1.0;
                metrics.h_bonds_donated += 1.0;
                metrics.h_bond_strength_sum += strength;
            }
        }
    }

    for i in 0..water_count {
        let water_o = snap.water_o_posits[i];
        let water_h0 = snap.water_h0_posits[i];
        let water_h1 = snap.water_h1_posits[i];

        for &(acceptor, acceptor_element) in &solute_acceptors {
            if let Some(strength) = h_bond_strength_if_present(
                water_o,
                water_h0,
                acceptor,
                Element::Oxygen,
                acceptor_element,
                cell_extent,
            ) {
                metrics.h_bonds += 1.0;
                metrics.h_bonds_accepted += 1.0;
                metrics.h_bond_strength_sum += strength;
            }

            if let Some(strength) = h_bond_strength_if_present(
                water_o,
                water_h1,
                acceptor,
                Element::Oxygen,
                acceptor_element,
                cell_extent,
            ) {
                metrics.h_bonds += 1.0;
                metrics.h_bonds_accepted += 1.0;
                metrics.h_bond_strength_sum += strength;
            }
        }
    }

    let mut nearest = f32::INFINITY;

    for &water_o in snap.water_o_posits.iter().take(water_count) {
        let mut nearest_solute_atom = f32::INFINITY;

        for &atom_posit in atom_posits {
            let dist = min_image(water_o - atom_posit, cell_extent).magnitude();
            nearest_solute_atom = nearest_solute_atom.min(dist);
        }

        if nearest_solute_atom.is_finite() {
            nearest = nearest.min(nearest_solute_atom);
            if nearest_solute_atom <= FIRST_HYDRATION_SHELL_CUTOFF_A {
                metrics.first_shell_water_count += 1.0;
                metrics.first_shell_water_o_distance_sum += nearest_solute_atom;
            }
        }
    }

    if nearest.is_finite() {
        metrics.nearest_water_o_distance_a = Some(nearest);
    }

    metrics
}

/// This is what we use to collect properties on self-affinity after the MD run.
fn add_md_metrics(
    data: &mut WaterSolData,
    char: &MolCharacterization,
    mol: &MoleculeSmall,
    snapshots: &[Snapshot],
    cell_extent: Vec3,
) {
    if snapshots.is_empty() {
        return;
    }

    let snaps = if snapshots.len() > 4 {
        &snapshots[snapshots.len() / 2..]
    } else {
        snapshots
    };

    let mut metrics = WaterSolDataMdProperties {
        water_molecule_count: snapshots
            .last()
            .map(|snap| snap.water_o_posits.len())
            .unwrap_or_default(),
        box_min_side_a: cell_extent.x.min(cell_extent.y).min(cell_extent.z),
        box_volume_a3: cell_extent.x * cell_extent.y * cell_extent.z,
        ..Default::default()
    };

    let mut potentials = Vec::new();
    let mut nonbonded = Vec::new();
    let mut solute_water_interactions = Vec::new();
    let mut pressures = Vec::new();
    let mut temperatures = Vec::new();
    let mut densities = Vec::new();
    let mut volumes = Vec::new();

    let mut h_bonds = Vec::new();
    let mut h_bonds_donated = Vec::new();
    let mut h_bonds_accepted = Vec::new();
    let mut h_bond_strength = Vec::new();
    let mut nearest_water = Vec::new();
    let mut first_shell_water = Vec::new();
    let mut first_shell_dist = Vec::new();

    for snap in snaps {
        if let Some(e) = &snap.energy_data {
            potentials.push(e.energy_potential);
            nonbonded.push(e.energy_potential_nonbonded);
            pressures.push(e.pressure);
            temperatures.push(e.temperature);
            densities.push(e.density * AMU_A3_TO_G_CM3);
            volumes.push(e.volume);

            if let Some(dh_dl) = e.dh_dl {
                if dh_dl != 0.0 {
                    solute_water_interactions.push(-dh_dl);
                }
            }
        }

        let water_metrics = analyze_snapshot_water_contacts(mol, snap, cell_extent);
        h_bonds.push(water_metrics.h_bonds);
        h_bonds_donated.push(water_metrics.h_bonds_donated);
        h_bonds_accepted.push(water_metrics.h_bonds_accepted);
        first_shell_water.push(water_metrics.first_shell_water_count);

        if water_metrics.h_bonds > 0.0 {
            h_bond_strength.push(water_metrics.h_bond_strength_sum / water_metrics.h_bonds);
        }
        if let Some(v) = water_metrics.nearest_water_o_distance_a {
            nearest_water.push(v);
        }
        if water_metrics.first_shell_water_count > 0.0 {
            first_shell_dist.push(
                water_metrics.first_shell_water_o_distance_sum
                    / water_metrics.first_shell_water_count,
            );
        }
    }

    metrics.potential_energy_kcal = mean(&potentials).unwrap_or(0.0);
    metrics.nonbonded_energy_kcal = mean(&nonbonded).unwrap_or(0.0);
    metrics.solute_water_interaction_kcal = mean(&solute_water_interactions).unwrap_or(0.0);
    metrics.mean_pressure_bar = mean(&pressures).unwrap_or(0.0);
    metrics.mean_temperature_k = mean(&temperatures).unwrap_or(0.0);
    metrics.density_g_cm3 = mean(&densities).unwrap_or(0.0);
    metrics.box_volume_a3 = mean(&volumes).unwrap_or(metrics.box_volume_a3);
    metrics.water_h_bonds = mean(&h_bonds).unwrap_or(0.0);
    metrics.water_h_bonds_donated = mean(&h_bonds_donated).unwrap_or(0.0);
    metrics.water_h_bonds_accepted = mean(&h_bonds_accepted).unwrap_or(0.0);
    metrics.mean_water_h_bond_strength = mean(&h_bond_strength).unwrap_or(0.0);
    metrics.nearest_water_o_distance_a = mean(&nearest_water).unwrap_or(0.0);
    metrics.first_shell_water_count = mean(&first_shell_water).unwrap_or(0.0);
    metrics.first_shell_water_per_heavy_atom =
        metrics.first_shell_water_count / char.num_heavy_atoms.max(1) as f32;
    metrics.mean_first_shell_water_o_distance_a = mean(&first_shell_dist).unwrap_or(0.0);

    let interaction_score = if metrics.solute_water_interaction_kcal != 0.0 {
        (-metrics.solute_water_interaction_kcal / 20.0).clamp(-3.0, 3.0)
    } else {
        0.0
    };
    let h_bond_score = metrics.water_h_bonds * 0.22 + metrics.mean_water_h_bond_strength * 0.65;
    let shell_score = metrics.first_shell_water_per_heavy_atom.clamp(0.0, 3.0) * 0.30;

    let md_water_affinity = interaction_score + h_bond_score + shell_score;
    data.water_affinity_score = (data.property_water_affinity_score + md_water_affinity).max(0.0);
    data.water_solubility_score =
        (data.water_affinity_score - data.hydration_penalty * 0.55).max(0.0);

    data.md_properties = Some(metrics);
}

fn gromacs_water_mol_name(ident: &str) -> String {
    let name: String = ident
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .take(6)
        .collect();

    if name.is_empty() {
        "SOLUT".to_string()
    } else {
        name
    }
}

fn gromacs_water_molecule_input(
    mol: &MoleculeSmall,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
) -> io::Result<MoleculeInput> {
    let Some(ff_params) = mol_specific_params.get(&mol.common.ident).cloned() else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing molecule-specific parameters for GROMACS water-solvation input.",
        ));
    };

    Ok(MoleculeInput {
        name: gromacs_water_mol_name(&mol.common.ident),
        atoms: mol.common.atoms.iter().map(|a| a.to_generic()).collect(),
        bonds: mol.common.bonds.iter().map(|b| b.to_generic()).collect(),
        ff_params: Some(ff_params),
        count: 1,
    })
}

fn run_water_dynamics(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    char: &MolCharacterization,
    dev: &ComputationDevice,
) -> io::Result<(WaterSolData, Vec<Snapshot>)> {
    let cfg = build_md_cfg();
    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];

    let (mut md, _) = build_dynamics(
        &dev,
        &mols,
        param_set,
        mol_specific_params,
        &cfg,
        false,
        None,
        &mut HashSet::new(),
    )
    .map_err(param_err)?;

    println!("MD WATER COUNT: {}", md.water.len()); // todo temp

    // Keep the solute fully coupled, but enable interaction bookkeeping so snapshots
    // can report molecule-water attraction through dh/d_lambda.
    md.alchemical.mol_idx = Some(0);
    md.alchemical.lambda = 0.0;

    run_dynamics_blocking(&mut md, &dev, DT, NUM_STEPS);

    if md.snapshots.is_empty() {
        return Err(io::Error::other(
            "Water-solvation MD completed without recording snapshots.",
        ));
    }

    let mut data = water_sol_data_from_properties(char, WaterSolEstimateSource::MolecularDynamics);
    add_md_metrics(&mut data, char, mol, &md.snapshots, md.cell.extent);

    Ok((data, md.snapshots))
}

fn run_water_gromacs(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    char: &MolCharacterization,
) -> io::Result<(WaterSolData, Vec<Snapshot>)> {
    let cfg = build_gromacs_md_cfg();
    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];
    let mol_input = gromacs_water_molecule_input(mol, mol_specific_params)?;
    let mdp = cfg.to_gromacs(NUM_STEPS, DT);
    let input = crate::gromacs::make_gromacs_input(
        mdp,
        &mols,
        vec![mol_input],
        param_set,
        &cfg.sim_box,
        &cfg.solvent,
        cfg.max_init_relaxation_iters.is_some(),
    )?;
    let cell_extent = input
        .box_nm
        .map(|(x, y, z)| Vec3::new((x * 10.0) as f32, (y * 10.0) as f32, (z * 10.0) as f32))
        .unwrap_or_else(Vec3::new_zero);

    let out = input.run()?;
    if out.setup_failure {
        return Err(io::Error::other(
            "GROMACS setup failed while running water-solvation MD.",
        ));
    }
    if out.log_text.contains("Fatal error") {
        return Err(io::Error::other(
            "GROMACS reported a fatal error while running water-solvation MD.",
        ));
    }

    let snapshots = gromacs_frames_to_ss(&out);
    if snapshots.is_empty() {
        return Err(io::Error::other(
            "GROMACS water-solvation MD completed without recording snapshots.",
        ));
    }

    let mut data = water_sol_data_from_properties(char, WaterSolEstimateSource::MolecularDynamics);
    add_md_metrics(&mut data, char, mol, &snapshots, cell_extent);

    Ok((data, snapshots))
}

/// Runs a molecular dynamics simulation of the molecule in OPC water. Returns both
/// water-affinity descriptors and snapshots that can be used to visualize the solvated run.
pub fn estimate_from_md(
    mol: &MoleculeSmall,
    backend: MdBackend,
    dev: &ComputationDevice,
) -> io::Result<(WaterSolData, Vec<Snapshot>)> {
    let param_set = FfParamSet::new_amber()?;
    let (mol, mol_specific_params) = prepare_mol_for_md(mol, &param_set)?;
    let Some(char) = &mol.characterization else {
        return Err(io::Error::other(
            "Char missing when estimating crystal data",
        ));
    };

    println!(
        "Water-solvation MD setup ({backend}): OPC water, sim box pad from dynamics, one solute copy"
    );

    match backend {
        MdBackend::Dynamics => {
            run_water_dynamics(&mol, &param_set, &mol_specific_params, &char, dev)
        }
        MdBackend::Gromacs => run_water_gromacs(&mol, &param_set, &mol_specific_params, &char),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Water-solvation MD estimation supports the Dynamics and GROMACS backends.",
        )),
    }
}
