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
    alchemical::{
        LambdaWindow, collect_window, free_energy_ti_with_sem, mean_coupled_interaction_kcal,
    },
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::f32::Vec3;
use na_seq::Element;

use crate::gromacs::make_gromacs_input;
use crate::properties::{mean, min_image};
use crate::{
    bond_inference::h_bond_geometry_strength,
    md::{MdBackend, build_dynamics, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
    properties::{io_error, mol_characterization::MolCharacterization, prepare_mol_for_md},
};

// todo: Set higher once confident this works.
const NUM_STEPS: usize = 2_000;
const SNAPSHOT_INTERVAL: usize = 10;
const TEMPERATURE: f32 = 300.; // K.
const PRESSURE: f32 = 1.; // Bar.
const DT: f32 = 0.002; // ps

// TI: thermodynamic integration.
// Free-energy settings. These are still short for production scientific work,
// but they are long enough to collect a real TI estimate instead of a single
// lambda=0 interaction proxy. Downstream consumers should inspect the SEM and window data.
const HYDRATION_TI_BOX_SIZE_A: f32 = 35.0;
const HYDRATION_TI_EQUIL_STEPS_PER_WINDOW: usize = 5_000;
const HYDRATION_TI_PROD_STEPS_PER_WINDOW: usize = 20_000;

// These range from 0 (full interaction between solute and solvent) to 1 (no interaction).
const HYDRATION_TI_LAMBDAS: &[f64] = &[
    0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0,
];

const AMU_A3_TO_G_CM3: f32 = 1.660_539;
const FIRST_HYDRATION_SHELL_CUTOFF_A: f32 = 3.6;

/// How the data was estimated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterSolEstimateSource {
    Properties,
    MolecularDynamics,
}

#[derive(Clone, Debug)]
/// Units for dh_dl: kcal/mol
pub struct WaterSolAlchemicalWindow {
    pub lambda: f64,
    pub mean_dh_dl: f64,
    pub sem_dh_dl: f64,
}

#[derive(Clone, Debug, Default)]
/// All energies are in Kcal/Mol
pub struct WaterSolMdProperties {
    /// Number of OPC water molecules represented by the simulation cell.
    pub water_molecule_count: usize,
    pub box_volume_a3: f32,
    pub box_min_side_a: f32,
    pub density_g_cm3: f32,
    pub mean_temperature_k: f32,
    pub mean_pressure_bar: f32,
    pub potential_energy_kcal: f32,
    pub nonbonded_energy_kcal: f32,
    /// Solute-environment interaction proxy from lambda=0 Dynamics alchemical bookkeeping.
    /// More negative means stronger attraction. This is not a free energy.
    /// `None` means the selected backend did not record alchemical dH/dlambda data.
    /// todo: What does proxy mean here? Anecdotally, it usually means a garbage analytic
    /// todo: computation an LLM created
    pub solute_water_interaction_proxy_kcal_mol: f32,
    /// Thermodynamic-integration free energy for decoupling the solute from water.
    /// Positive values usually mean water stabilizes the coupled solute.
    pub alch_decoupling_free_energy: f64,
    pub alch_decoupling_free_energy_sem: f64,
    /// Hydration free energy estimate derived as the negative of the decoupling
    /// free energy. This currently does not include finite-size, restraint, or
    /// standard-state corrections.
    pub hyd_free_energy: f64,
    pub hyd_free_energy_sem: f64,
    pub alchemical_windows: Vec<WaterSolAlchemicalWindow>,
    pub md_water_affinity_score: f32,
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

struct HydrationTiRun {
    window: LambdaWindow,
    snapshots: Vec<Snapshot>,
    cell_extent: Vec3,
}

fn param_err(e: ParamError) -> io::Error {
    io::Error::other(e.descrip)
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

fn build_hydration_ti_cfg() -> MdConfig {
    MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        zero_com_drift: true,
        temp_target: TEMPERATURE,
        // Keep the cell fixed for thermodynamic integration. The lambda-dependent
        // estimator should not mix in barostat volume fluctuations.
        barostat_cfg: None,
        snapshot_handlers: SnapshotHandlers {
            memory: Some(SNAPSHOT_INTERVAL),
            dcd: None,
            gromacs: OutputControl::default(),
        },
        sim_box: SimBoxInit::new_cube(HYDRATION_TI_BOX_SIZE_A),
        solvent: Solvent::WaterOpc,
        recenter_sim_box: true,
        overrides: MdOverrides::default(),
        ..Default::default()
    }
}

fn h_bond_candidate_el(element: Element) -> bool {
    matches!(
        element,
        Element::Nitrogen | Element::Oxygen | Element::Sulfur | Element::Fluorine
    )
}

/// Applies periodic minimum-image correction to the hydrogen and acceptor positions
/// (relative to the donor heavy atom) and then defers to `bond_inference` for the
/// distance/angle geometry test and strength scoring.
fn h_bond_strength_if_present(
    donor_posit: Vec3,
    h_posit: Vec3,
    acc_posit: Vec3,
    donor_element: Element,
    acceptor_element: Element,
    cell_extent: Vec3,
) -> Option<f32> {
    let acc_imaged = donor_posit + min_image(acc_posit - donor_posit, cell_extent);
    let h_imaged = donor_posit + min_image(h_posit - donor_posit, cell_extent);

    h_bond_geometry_strength(
        donor_posit.into(),
        h_imaged.into(),
        acc_imaged.into(),
        donor_element,
        acceptor_element,
        false,
    )
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
fn create_water_sol_metrics(
    char: &MolCharacterization,
    mol: &MoleculeSmall,
    snapshots: &[Snapshot],
    cell_extent: Vec3,
) -> WaterSolMdProperties {
    let snaps = if snapshots.len() > 4 {
        &snapshots[snapshots.len() / 2..]
    } else {
        snapshots
    };

    let mut res = WaterSolMdProperties {
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

    res.potential_energy_kcal = mean(&potentials).unwrap_or(0.0);
    res.nonbonded_energy_kcal = mean(&nonbonded).unwrap_or(0.0);

    res.solute_water_interaction_proxy_kcal_mol = mean_coupled_interaction_kcal(snaps).unwrap();

    res.mean_pressure_bar = mean(&pressures).unwrap_or(0.0);
    res.mean_temperature_k = mean(&temperatures).unwrap_or(0.0);
    res.density_g_cm3 = mean(&densities).unwrap_or(0.0);
    res.box_volume_a3 = mean(&volumes).unwrap_or(res.box_volume_a3);
    res.water_h_bonds = mean(&h_bonds).unwrap_or(0.0);
    res.water_h_bonds_donated = mean(&h_bonds_donated).unwrap_or(0.0);
    res.water_h_bonds_accepted = mean(&h_bonds_accepted).unwrap_or(0.0);
    res.mean_water_h_bond_strength = mean(&h_bond_strength).unwrap_or(0.0);
    res.nearest_water_o_distance_a = mean(&nearest_water).unwrap_or(0.0);
    res.first_shell_water_count = mean(&first_shell_water).unwrap_or(0.0);
    res.first_shell_water_per_heavy_atom =
        res.first_shell_water_count / char.num_heavy_atoms.max(1) as f32;
    res.mean_first_shell_water_o_distance_a = mean(&first_shell_dist).unwrap_or(0.0);

    let interaction_score = (-res.solute_water_interaction_proxy_kcal_mol / 20.0).clamp(-3.0, 3.0);

    let h_bond_score = res.water_h_bonds * 0.22 + res.mean_water_h_bond_strength * 0.65;
    let shell_score = res.first_shell_water_per_heavy_atom.clamp(0.0, 3.0) * 0.30;

    res.md_water_affinity_score = interaction_score + h_bond_score + shell_score;

    res
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

fn water_sol_window_from_lambda_window(window: &LambdaWindow) -> WaterSolAlchemicalWindow {
    WaterSolAlchemicalWindow {
        lambda: window.lambda,
        mean_dh_dl: window.mean_dh_dl,
        sem_dh_dl: window.sem_dh_dl,
    }
}

fn apply_hydration_free_energy(
    data: &mut WaterSolMdProperties,
    windows: Vec<LambdaWindow>,
) -> io::Result<()> {
    let ti = free_energy_ti_with_sem(&windows)
        .map_err(|e| io_error("Unable to integrate water-solvation alchemical windows", e))?;

    data.alch_decoupling_free_energy = ti.free_energy;
    data.alch_decoupling_free_energy_sem = ti.standard_error;
    data.hyd_free_energy = -ti.free_energy;
    data.hyd_free_energy_sem = ti.standard_error;

    data.alchemical_windows = windows
        .iter()
        .map(water_sol_window_from_lambda_window)
        .collect();

    Ok(())
}

/// Run MD for thermodynamic integration at a specific lambda and keep the production
/// snapshots. The lambda=0 call is also used for water-contact descriptors and viewing.
fn run_hydration_ti_window(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    dev: &ComputationDevice,
    lambda: f64,
) -> io::Result<HydrationTiRun> {
    let cfg = build_hydration_ti_cfg();
    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];

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
    .map_err(param_err)?;

    md.configure_alchemical_window(dev, 0, lambda)
        .map_err(|e| {
            io_error(
                "Unable to configure water-solvation alchemical free-energy window",
                e,
            )
        })?;

    run_dynamics_blocking(&mut md, dev, DT, HYDRATION_TI_EQUIL_STEPS_PER_WINDOW);
    md.snapshots.clear();
    run_dynamics_blocking(&mut md, dev, DT, HYDRATION_TI_PROD_STEPS_PER_WINDOW);

    let window = collect_window(lambda, &md.snapshots).map_err(|e| {
        io_error(
            "Unable to collect water-solvation alchemical free-energy samples",
            e,
        )
    })?;

    Ok(HydrationTiRun {
        window,
        snapshots: md.snapshots,
        cell_extent: md.cell.extent,
    })
}

fn run_hydration_free_energy_ti(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    dev: &ComputationDevice,
    initial_lambda_zero: LambdaWindow,
) -> io::Result<Vec<LambdaWindow>> {
    let mut windows = Vec::with_capacity(HYDRATION_TI_LAMBDAS.len());
    windows.push(initial_lambda_zero);

    for &lambda in HYDRATION_TI_LAMBDAS {
        if lambda == 0.0 {
            // Handled outside this.
            continue;
        }

        println!("\n------Running water-solvation at λ={lambda:.3}...");
        let run = run_hydration_ti_window(mol, param_set, mol_specific_params, dev, lambda)?;
        let window = run.window;

        println!(
            "Water-solvation window complete: λ={lambda:.3} <dH/dλ>={:.4} kcal/mol sem={:.4}",
            window.mean_dh_dl, window.sem_dh_dl
        );
        windows.push(window);
    }

    Ok(windows)
}

/// Launch using the dynamics backend.
fn run_dynamics(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    char: &MolCharacterization,
    dev: &ComputationDevice,
) -> io::Result<(WaterSolMdProperties, Vec<Snapshot>)> {
    println!("\n------Running water-solvation at λ=0.000...");
    let lambda_zero = run_hydration_ti_window(mol, param_set, mol_specific_params, dev, 0.0)?;

    println!(
        "Water-solvation window complete: lambda=0.000 <dH/dλ>={:.4} kcal/mol sem={:.4}",
        lambda_zero.window.mean_dh_dl, lambda_zero.window.sem_dh_dl,
    );

    let mut data =
        create_water_sol_metrics(char, mol, &lambda_zero.snapshots, lambda_zero.cell_extent);

    let snapshots = lambda_zero.snapshots.clone();
    let windows =
        run_hydration_free_energy_ti(mol, param_set, mol_specific_params, dev, lambda_zero.window)?;

    apply_hydration_free_energy(&mut data, windows)?;

    Ok((data, snapshots))
}

fn run_gromacs(
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    char: &MolCharacterization,
) -> io::Result<(WaterSolMdProperties, Vec<Snapshot>)> {
    let cfg = build_gromacs_md_cfg();
    let mdp = cfg.to_gromacs(NUM_STEPS, DT);

    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];

    let mol_input = gromacs_water_molecule_input(mol, mol_specific_params)?;

    let input = make_gromacs_input(
        mdp,
        &mols,
        vec![mol_input],
        param_set,
        &cfg.sim_box,
        &cfg.solvent,
        cfg.max_init_relaxation_iters.is_some(),
    )?;

    let out = input.run()?;

    // if out.setup_failure {
    //     return Err(io::Error::other(
    //         "GROMACS setup failed while running water-solvation MD.",
    //     ));
    // }
    // if out.log_text.contains("Fatal error") {
    //     return Err(io::Error::other(
    //         "GROMACS reported a fatal error while running water-solvation MD.",
    //     ));
    // }

    let snapshots = gromacs_frames_to_ss(&out);

    let cell_extent = input
        .box_nm
        .map(|(x, y, z)| Vec3::new((x * 10.0) as f32, (y * 10.0) as f32, (z * 10.0) as f32))
        .unwrap_or_else(Vec3::new_zero);

    let data = create_water_sol_metrics(char, mol, &snapshots, cell_extent);

    Ok((data, snapshots))
}

/// Runs a molecular dynamics simulation of the molecule in OPC water. Returns both
/// water-affinity descriptors and snapshots that can be used to visualize the solvated run.
///
/// This is the entry point for this module.
pub fn estimate_from_md(
    mol: &MoleculeSmall,
    backend: MdBackend,
    dev: &ComputationDevice,
    param_set: &FfParamSet,
) -> io::Result<(WaterSolMdProperties, Vec<Snapshot>)> {
    let (mol, mol_specific_params) = prepare_mol_for_md(mol, &param_set)?;

    let Some(char) = &mol.characterization else {
        return Err(io::Error::other(
            "Characterization missing when estimating water-solvation data",
        ));
    };

    println!("\nRunning water-solvation MD. Backend: {backend}");

    match backend {
        MdBackend::Dynamics => run_dynamics(&mol, &param_set, &mol_specific_params, &char, dev),
        MdBackend::Gromacs => run_gromacs(&mol, &param_set, &mol_specific_params, &char),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Water-solvation MD estimation supports the Dynamics and GROMACS backends.",
        )),
    }
}
