//! Driven shrinking-box simulations for probing water/solute packing behavior.
//!
//! Unlike the template-preparation helper in `dynamics`, this is the property
//! simulation itself: snapshots cover the gradual compression from a dilute
//! starting cell to a target-density cell.

use std::{
    collections::HashMap,
    fmt,
    io::{self, ErrorKind},
};

use bio_files::{
    gromacs::{
        GromacsInput, GromacsOutput, MoleculeInput, OutputControl,
        gro::{AtomGro, Gro},
        solvate::Solvent as GromacsSolvent,
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MdState, MolDynamics,
    ShrinkingBoxCfg, SimBox, SimBoxInit, Solvent, TAU_TEMP_DEFAULT,
    params::FfParamSet,
    random_quaternion,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3 as Vec3F64},
};

use crate::{
    gromacs::{make_gromacs_input, molecule_input_from_packed_copies},
    md::MdBackend,
    molecules::small::MoleculeSmall,
    properties::{AMU_A3_TO_G_CM3, mean, mol_bounding_radius, prepare_mol_for_md},
};

const SOLUTE_COPY_COUNT: usize = 24;
const MIN_TARGET_BOX_SIDE_A: f32 = 34.0;
const MIN_COMPRESSION_BOX_SIDE_A: f32 = 24.0;
const COMPRESSION_LIMIT_SCALE: f32 = 0.65;
const WATER_MOLAR_VOLUME_A3: f32 = 29.97;
const HOMOGENEOUS_SOLUTE_VOLUME_FRACTION: f32 = 0.22;
const HOMOGENEOUS_WATER_VOLUME_FRACTION: f32 = 0.84;
const LAYER_SOLUTE_VOLUME_FRACTION: f32 = 0.42;
const LAYER_WATER_VOLUME_FRACTION: f32 = 0.50;
const LAYER_INTERFACE_GAP_A: f32 = 2.4;
const SOLUTE_WALL_MARGIN_A: f32 = 1.0;

const INITIAL_BOX_SCALE: f32 = 2.0;
const BOX_SHRINK_PER_STEP_A: f32 = 0.002;
const DYNAMICS_EQUILIBRATION_STEPS: usize = 5_000;
const SNAPSHOT_INTERVAL: usize = 10;
const CONTROL_WINDOW_SNAPSHOTS: usize = 5;
const GROMACS_CHUNK_STEPS: usize = 500;
const MAX_SHRINK_PRESSURE_BAR: f32 = 10_000.0;
const MAX_TEMPERATURE_K: f32 = 450.0;
const MAX_DENSITY_G_CM3: f32 = 2.5;
const MAX_ATOM_DENSITY_A3: f32 = 0.18;
const MAX_FORCE_KCAL_MOL_A: f32 = 500.0;
const TEMPERATURE_K: f32 = 300.0;
const DT_PS: f32 = 0.002;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShrinkingBoxMode {
    /// Interleave OPC water and solute copies throughout the whole cell.
    HomogeneousMix,
    /// Begin with a water slab touching a solute slab.
    WaterSoluteLayers,
}

impl fmt::Display for ShrinkingBoxMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HomogeneousMix => write!(f, "homogeneous mix"),
            Self::WaterSoluteLayers => write!(f, "water/solute layers"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionStopReason {
    Pressure,
    Temperature,
    MassDensity,
    AtomDensity,
    Force,
    CompressionLimit,
    GromacsChunkFailure,
}

#[derive(Clone, Debug)]
pub struct ShrinkingBoxMdData {
    pub mode: ShrinkingBoxMode,
    pub solute_copy_count: usize,
    pub water_molecule_count: usize,
    pub initial_box_extent_a: Vec3F32,
    pub final_box_extent_a: Vec3F32,
    pub target_box_extent_a: Vec3F32,
    pub compression_limit_box_extent_a: Vec3F32,
    pub reached_target_box: bool,
    pub reached_compression_limit: bool,
    pub stopped_for_density: bool,
    pub stopped_for_pressure: bool,
    pub stopped_for_temperature: bool,
    pub stopped_for_force: bool,
    pub stop_reason: Option<CompressionStopReason>,
    pub mean_temperature_k: f32,
    pub mean_pressure_bar: f32,
    pub density_g_cm3: f32,
    pub potential_energy_kcal: f32,
    pub nonbonded_energy_kcal: f32,
}

#[derive(Clone, Copy, Debug)]
struct ShrinkingBoxSetup {
    mode: ShrinkingBoxMode,
    solute_copy_count: usize,
    water_molecule_count: usize,
    target_cell: SimBox,
    compression_limit_cell: SimBox,
    initial_cell: SimBox,
    shrink_cfg: ShrinkingBoxCfg,
}

struct PlacedSolutes {
    mols: Vec<MolDynamics>,
}

fn centered_cube(side: f32) -> SimBox {
    let half = side / 2.0;
    SimBox::new(
        Vec3F32::new(-half, -half, -half),
        Vec3F32::new(half, half, half),
    )
}

fn setup_for(mol: &MoleculeSmall, mode: ShrinkingBoxMode) -> io::Result<ShrinkingBoxSetup> {
    let Some(char) = &mol.characterization else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Characterization is required for a shrinking-box simulation.",
        ));
    };

    let solute_volume_fraction = match mode {
        ShrinkingBoxMode::HomogeneousMix => HOMOGENEOUS_SOLUTE_VOLUME_FRACTION,
        ShrinkingBoxMode::WaterSoluteLayers => LAYER_SOLUTE_VOLUME_FRACTION,
    };
    let mol_volume_a3 = char.volume.max(1.0);
    let min_volume_a3 = MIN_TARGET_BOX_SIDE_A.powi(3);
    let solute_limited_volume_a3 =
        SOLUTE_COPY_COUNT as f32 * mol_volume_a3 / solute_volume_fraction;
    let mol_radius_a = mol_bounding_radius(mol);
    let molecule_limited_side_a = match mode {
        ShrinkingBoxMode::HomogeneousMix => 2.0 * (mol_radius_a + SOLUTE_WALL_MARGIN_A),
        ShrinkingBoxMode::WaterSoluteLayers => {
            4.0 * (mol_radius_a + SOLUTE_WALL_MARGIN_A) + LAYER_INTERFACE_GAP_A + 2.0
        }
    };
    let target_side_a = min_volume_a3
        .max(solute_limited_volume_a3)
        .cbrt()
        .max(molecule_limited_side_a);
    let target_volume_a3 = target_side_a.powi(3);
    let target_cell = centered_cube(target_side_a);
    let compression_limit_side_a = (target_side_a * COMPRESSION_LIMIT_SCALE)
        .max(MIN_COMPRESSION_BOX_SIDE_A)
        .max(molecule_limited_side_a);
    let compression_limit_cell = centered_cube(compression_limit_side_a);
    let target_water_volume_a3 = match mode {
        // Leave headroom for solute excluded volume and driven compression. Unlike the
        // Dynamics loop, a single GROMACS deform run cannot stop early on high pressure.
        ShrinkingBoxMode::HomogeneousMix => target_volume_a3 * HOMOGENEOUS_WATER_VOLUME_FRACTION,
        ShrinkingBoxMode::WaterSoluteLayers => target_volume_a3 * LAYER_WATER_VOLUME_FRACTION,
    };
    let water_molecule_count =
        ((target_water_volume_a3 / WATER_MOLAR_VOLUME_A3).round() as usize).max(1);
    let shrink_cfg = ShrinkingBoxCfg {
        initial_box_scale: INITIAL_BOX_SCALE,
        box_shrink_per_step: BOX_SHRINK_PER_STEP_A,
    };

    Ok(ShrinkingBoxSetup {
        mode,
        solute_copy_count: SOLUTE_COPY_COUNT,
        water_molecule_count,
        target_cell,
        compression_limit_cell,
        initial_cell: shrink_cfg.initial_cell(target_cell),
        shrink_cfg,
    })
}

fn solute_template(
    mol: &MoleculeSmall,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
) -> io::Result<MolDynamics> {
    let Some(ff_params) = mol_specific_params.get(&mol.common.ident).cloned() else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing molecule-specific parameters for shrinking-box input.",
        ));
    };

    Ok(MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: mol.common.atoms.iter().map(|a| a.to_generic()).collect(),
        atom_posits: Some(mol.common.atom_posits.clone()),
        atom_init_velocities: None,
        bonds: mol.common.bonds.iter().map(|b| b.to_generic()).collect(),
        adjacency_list: Some(mol.common.adjacency_list.clone()),
        static_: false,
        bonded_only: false,
        mol_specific_params: Some(ff_params),
    })
}

fn solute_region(setup: ShrinkingBoxSetup) -> SimBox {
    match setup.mode {
        ShrinkingBoxMode::HomogeneousMix => setup.initial_cell,
        ShrinkingBoxMode::WaterSoluteLayers => {
            let mid_z = setup.initial_cell.center().z - LAYER_INTERFACE_GAP_A / 2.0;
            SimBox::new(
                setup.initial_cell.bounds_low,
                Vec3F32::new(
                    setup.initial_cell.bounds_high.x,
                    setup.initial_cell.bounds_high.y,
                    mid_z,
                ),
            )
        }
    }
}

fn grid_dims(copies: usize, extent: Vec3F32) -> (usize, usize, usize) {
    let copies = copies.max(1);
    let ideal_side = (extent.x as f64 * extent.y as f64 * extent.z as f64 / copies as f64).cbrt();
    let mut nx = ((extent.x as f64 / ideal_side).floor() as usize).max(1);
    let mut ny = ((extent.y as f64 / ideal_side).floor() as usize).max(1);
    let mut nz = ((extent.z as f64 / ideal_side).floor() as usize).max(1);

    while nx * ny * nz < copies {
        let next_x = extent.x as f64 / (nx + 1) as f64;
        let next_y = extent.y as f64 / (ny + 1) as f64;
        let next_z = extent.z as f64 / (nz + 1) as f64;
        if next_x >= next_y && next_x >= next_z {
            nx += 1;
        } else if next_y >= next_z {
            ny += 1;
        } else {
            nz += 1;
        }
    }

    (nx, ny, nz)
}

fn place_solute_copies(
    mol: &MoleculeSmall,
    template: &MolDynamics,
    setup: ShrinkingBoxSetup,
) -> io::Result<PlacedSolutes> {
    let region = solute_region(setup);
    let template_posits = template
        .atom_posits
        .as_ref()
        .cloned()
        .unwrap_or_else(|| template.atoms.iter().map(|atom| atom.posit).collect());
    let centroid = template_posits
        .iter()
        .copied()
        .fold(Vec3F64::new_zero(), |sum, posit| sum + posit)
        * (1.0 / template_posits.len().max(1) as f64);
    let local_posits: Vec<_> = template_posits
        .iter()
        .map(|posit| *posit - centroid)
        .collect();
    // Centroids contract toward the box center while each molecule keeps its rigid
    // size. Reserve the final wall inset in the expanded starting cell as well.
    let inset_a =
        (SOLUTE_WALL_MARGIN_A + mol_bounding_radius(mol)) * setup.shrink_cfg.initial_box_scale;
    let usable_extent = region.extent - Vec3F32::splat(2.0 * inset_a);

    if usable_extent.x <= 0.0 || usable_extent.y <= 0.0 || usable_extent.z <= 0.0 {
        return Err(io::Error::other(format!(
            "Shrinking-box initial region {:?} is too small for a {:.2} A molecule radius.",
            region.extent,
            mol_bounding_radius(mol),
        )));
    }

    let (nx, ny, nz) = grid_dims(setup.solute_copy_count, usable_extent);
    let spacing = Vec3F64::new(
        usable_extent.x as f64 / nx as f64,
        usable_extent.y as f64 / ny as f64,
        usable_extent.z as f64 / nz as f64,
    );
    let low = Vec3F64::new(
        (region.bounds_low.x + inset_a) as f64,
        (region.bounds_low.y + inset_a) as f64,
        (region.bounds_low.z + inset_a) as f64,
    );
    let mut rng = rand::rng();
    let mut result = Vec::with_capacity(setup.solute_copy_count);

    for copy_i in 0..setup.solute_copy_count {
        let ix = copy_i % nx;
        let iy = (copy_i / nx) % ny;
        let iz = copy_i / (nx * ny);
        let center = Vec3F64::new(
            low.x + (ix as f64 + 0.5) * spacing.x,
            low.y + (iy as f64 + 0.5) * spacing.y,
            low.z + (iz as f64 + 0.5) * spacing.z,
        );
        let rotation: Quaternion = random_quaternion(&mut rng, None).into();
        let posits: Vec<_> = local_posits
            .iter()
            .map(|posit| rotation.rotate_vec(*posit) + center)
            .collect();
        let mut copy = template.clone();
        for (atom, posit) in copy.atoms.iter_mut().zip(&posits) {
            atom.posit = *posit;
        }
        copy.atom_posits = Some(posits);
        result.push(copy);
    }

    Ok(PlacedSolutes { mols: result })
}

fn make_md_cfg(
    setup: ShrinkingBoxSetup,
    memory_snapshots: bool,
    skip_counterion_insertion: bool,
) -> MdConfig {
    MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        zero_com_drift: true,
        temp_target: TEMPERATURE_K,
        barostat_cfg: None,
        snapshot_handlers: SnapshotHandlers {
            memory: memory_snapshots.then_some(SNAPSHOT_INTERVAL),
            dcd: None,
            gromacs: if memory_snapshots {
                OutputControl::default()
            } else {
                OutputControl {
                    nstxout: Some(SNAPSHOT_INTERVAL as u32),
                    nstvout: Some(SNAPSHOT_INTERVAL as u32),
                    nstcalcenergy: Some(SNAPSHOT_INTERVAL as u32),
                    nstenergy: Some(SNAPSHOT_INTERVAL as u32),
                    ..Default::default()
                }
            },
        },
        sim_box: SimBoxInit::Fixed((
            setup.initial_cell.bounds_low,
            setup.initial_cell.bounds_high,
        )),
        solvent: Solvent::WaterOpcSpecifyMolCount(setup.water_molecule_count),
        max_init_relaxation_iters: None,
        recenter_sim_box: false,
        overrides: MdOverrides {
            skip_water_relaxation: true,
            skip_counterion_insertion,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn move_water_to_upper_layer(md: &mut MdState) {
    let cell = md.cell;
    let center_z = cell.center().z;
    let low_z = center_z + LAYER_INTERFACE_GAP_A / 2.0;
    let high_z = cell.bounds_high.z;
    let scale_z = (high_z - low_z) / cell.extent.z;

    for water in &mut md.water {
        let old_z = water.o.posit.z;
        let new_z = low_z + (old_z - cell.bounds_low.z) * scale_z;
        let dz = new_z - old_z;
        water.o.posit.z += dz;
        water.h0.posit.z += dz;
        water.h1.posit.z += dz;
        water.m.posit.z += dz;
    }
}

fn build_initial_state(
    setup: ShrinkingBoxSetup,
    placed_solutes: &PlacedSolutes,
    param_set: &FfParamSet,
    dev: &ComputationDevice,
    memory_snapshots: bool,
    skip_counterion_insertion: bool,
) -> io::Result<MdState> {
    let cfg = make_md_cfg(setup, memory_snapshots, skip_counterion_insertion);
    let (mut md, _) = MdState::new(dev, &cfg, &placed_solutes.mols, param_set)
        .map_err(|e| io::Error::other(e.descrip))?;

    match setup.mode {
        ShrinkingBoxMode::HomogeneousMix => {}
        ShrinkingBoxMode::WaterSoluteLayers => {
            move_water_to_upper_layer(&mut md);
            md.rebuild_spatial_caches(dev);
        }
    }

    Ok(md)
}

fn recent_energy_mean(
    snapshots: &[Snapshot],
    value: impl Fn(&dynamics::snapshot::SnapshotEnergyData) -> f32,
) -> Option<f32> {
    let values: Vec<_> = snapshots
        .iter()
        .rev()
        .filter_map(|snap| snap.energy_data.as_ref().map(&value))
        .take(CONTROL_WINDOW_SNAPSHOTS)
        .collect();

    (values.len() == CONTROL_WINDOW_SNAPSHOTS)
        .then(|| mean(&values))
        .flatten()
}

fn physical_atom_count(solute_atom_count: usize, water_molecule_count: usize) -> usize {
    solute_atom_count + 3 * water_molecule_count
}

fn latest_stop_reason(
    snapshots: &[Snapshot],
    atom_count: usize,
    max_force_kcal_mol_a: Option<f32>,
) -> Option<CompressionStopReason> {
    if max_force_kcal_mol_a.is_some_and(|force| !force.is_finite() || force >= MAX_FORCE_KCAL_MOL_A)
    {
        return Some(CompressionStopReason::Force);
    }

    let latest = snapshots.last()?.energy_data.as_ref()?;
    if !latest.temperature.is_finite()
        || recent_energy_mean(snapshots, |data| data.temperature)
            .is_some_and(|temperature| temperature >= MAX_TEMPERATURE_K)
    {
        return Some(CompressionStopReason::Temperature);
    }

    let density_g_cm3 = latest.density * AMU_A3_TO_G_CM3;
    if !density_g_cm3.is_finite() || density_g_cm3 >= MAX_DENSITY_G_CM3 {
        return Some(CompressionStopReason::MassDensity);
    }

    if latest.volume > 0.0 {
        let atom_density = atom_count as f32 / latest.volume;
        if !atom_density.is_finite() || atom_density >= MAX_ATOM_DENSITY_A3 {
            return Some(CompressionStopReason::AtomDensity);
        }
    }

    recent_energy_mean(snapshots, |data| data.pressure)
        .is_some_and(|pressure| pressure >= MAX_SHRINK_PRESSURE_BAR)
        .then_some(CompressionStopReason::Pressure)
}

fn mark_stop(data: &mut ShrinkingBoxMdData, reason: CompressionStopReason) {
    data.stop_reason = Some(reason);
    data.stopped_for_density = matches!(
        reason,
        CompressionStopReason::MassDensity | CompressionStopReason::AtomDensity
    );
    data.stopped_for_pressure = reason == CompressionStopReason::Pressure;
    data.stopped_for_temperature = reason == CompressionStopReason::Temperature;
    data.stopped_for_force = matches!(
        reason,
        CompressionStopReason::Force | CompressionStopReason::GromacsChunkFailure
    );
    data.reached_compression_limit = reason == CompressionStopReason::CompressionLimit;
}

fn cell_at_or_below(cell: SimBox, limit: SimBox) -> bool {
    cell.extent.x <= limit.extent.x
        && cell.extent.y <= limit.extent.y
        && cell.extent.z <= limit.extent.z
}

fn shrink_step_count_between(initial_cell: SimBox, target_cell: SimBox) -> usize {
    let shrink_needed = initial_cell.extent - target_cell.extent;
    let max_shrink = shrink_needed.x.max(shrink_needed.y).max(shrink_needed.z);
    (max_shrink / BOX_SHRINK_PER_STEP_A.max(f32::EPSILON)).ceil() as usize
}

fn max_dynamics_force(md: &MdState) -> Option<f32> {
    md.atoms
        .iter()
        .map(|atom| atom.force.magnitude())
        .chain(md.water.iter().flat_map(|water| {
            [
                water.o.force.magnitude(),
                water.h0.force.magnitude(),
                water.h1.force.magnitude(),
                water.m.force.magnitude(),
            ]
        }))
        .reduce(f32::max)
}

fn max_gromacs_force(out: &GromacsOutput) -> Option<f32> {
    const KJ_NM_TO_KCAL_ANG: f32 = 1.0 / 41.84;

    out.trajectory
        .iter()
        .rev()
        .take(CONTROL_WINDOW_SNAPSHOTS)
        .flat_map(|frame| &frame.atom_forces)
        .map(|force| force.magnitude() as f32 * KJ_NM_TO_KCAL_ANG)
        .reduce(f32::max)
}

fn shrink_cell_by_steps(current_cell: SimBox, limit_cell: SimBox, steps: usize) -> SimBox {
    let shrink_a = BOX_SHRINK_PER_STEP_A * steps as f32;
    let extent = Vec3F32::new(
        (current_cell.extent.x - shrink_a).max(limit_cell.extent.x),
        (current_cell.extent.y - shrink_a).max(limit_cell.extent.y),
        (current_cell.extent.z - shrink_a).max(limit_cell.extent.z),
    );
    let center = current_cell.center();
    let half = extent / 2.0;
    SimBox::new(center - half, center + half)
}

fn gromacs_deform_nm_ps(current_cell: SimBox, next_cell: SimBox, steps: usize) -> [f32; 6] {
    let duration_ps = steps.max(1) as f32 * DT_PS;
    let rate = |current: f32, next: f32| (next - current) * 0.1 / duration_ps;

    [
        rate(current_cell.extent.x, next_cell.extent.x),
        rate(current_cell.extent.y, next_cell.extent.y),
        rate(current_cell.extent.z, next_cell.extent.z),
        0.0,
        0.0,
        0.0,
    ]
}

fn initial_data(setup: ShrinkingBoxSetup, water_molecule_count: usize) -> ShrinkingBoxMdData {
    ShrinkingBoxMdData {
        mode: setup.mode,
        solute_copy_count: setup.solute_copy_count,
        water_molecule_count,
        initial_box_extent_a: setup.initial_cell.extent,
        final_box_extent_a: setup.initial_cell.extent,
        target_box_extent_a: setup.target_cell.extent,
        compression_limit_box_extent_a: setup.compression_limit_cell.extent,
        reached_target_box: false,
        reached_compression_limit: false,
        stopped_for_density: false,
        stopped_for_pressure: false,
        stopped_for_temperature: false,
        stopped_for_force: false,
        stop_reason: None,
        mean_temperature_k: 0.0,
        mean_pressure_bar: 0.0,
        density_g_cm3: 0.0,
        potential_energy_kcal: 0.0,
        nonbonded_energy_kcal: 0.0,
    }
}

fn add_snapshot_metrics(data: &mut ShrinkingBoxMdData, snapshots: &[Snapshot]) {
    let snaps = if snapshots.len() > 4 {
        &snapshots[snapshots.len() / 2..]
    } else {
        snapshots
    };
    let mut temperatures = Vec::new();
    let mut pressures = Vec::new();
    let mut densities = Vec::new();
    let mut potentials = Vec::new();
    let mut nonbonded = Vec::new();

    for snap in snaps {
        let Some(energy) = &snap.energy_data else {
            continue;
        };
        temperatures.push(energy.temperature);
        pressures.push(energy.pressure);
        densities.push(energy.density * AMU_A3_TO_G_CM3);
        potentials.push(energy.energy_potential);
        nonbonded.push(energy.energy_potential_nonbonded);
    }

    data.mean_temperature_k = mean(&temperatures).unwrap_or(0.0);
    data.mean_pressure_bar = mean(&pressures).unwrap_or(0.0);
    data.density_g_cm3 = mean(&densities).unwrap_or(0.0);
    data.potential_energy_kcal = mean(&potentials).unwrap_or(0.0);
    data.nonbonded_energy_kcal = mean(&nonbonded).unwrap_or(0.0);
}

fn run_dynamics(
    setup: ShrinkingBoxSetup,
    placed_solutes: &PlacedSolutes,
    param_set: &FfParamSet,
    dev: &ComputationDevice,
) -> io::Result<(ShrinkingBoxMdData, Vec<Snapshot>)> {
    let mut md = build_initial_state(setup, placed_solutes, param_set, dev, true, false)?;
    let mut data = initial_data(setup, md.water.len());
    let shrink_steps = shrink_step_count_between(setup.initial_cell, setup.compression_limit_cell);
    let mut equilibration_steps = 0;

    for _ in 0..shrink_steps + DYNAMICS_EQUILIBRATION_STEPS {
        if data.stop_reason.is_none() {
            if let Some(reason) = latest_stop_reason(
                &md.snapshots,
                physical_atom_count(md.atoms.len(), md.water.len()),
                max_dynamics_force(&md),
            ) {
                mark_stop(&mut data, reason);
                break;
            }

            let shrank =
                md.shrink_cell_towards(dev, setup.compression_limit_cell, setup.shrink_cfg);
            data.reached_target_box |= cell_at_or_below(md.cell, setup.target_cell);

            if !shrank || cell_at_or_below(md.cell, setup.compression_limit_cell) {
                mark_stop(&mut data, CompressionStopReason::CompressionLimit);
            }
        } else if data.reached_compression_limit {
            equilibration_steps += 1;
        }

        md.step(dev, DT_PS, None);

        if data.reached_compression_limit && equilibration_steps >= DYNAMICS_EQUILIBRATION_STEPS {
            break;
        }
    }

    data.final_box_extent_a = md.cell.extent;
    add_snapshot_metrics(&mut data, &md.snapshots);

    Ok((data, md.snapshots))
}

fn gromacs_mol_name(ident: &str) -> String {
    let name: String = ident
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .take(5)
        .collect();

    if name.is_empty() {
        "MOL".to_string()
    } else {
        name
    }
}

fn gromacs_solute_input(ident: &str, solute_mols: &[MolDynamics]) -> io::Result<MoleculeInput> {
    let input = molecule_input_from_packed_copies(gromacs_mol_name(ident), solute_mols)?;
    if input.ff_params.is_none() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing molecule-specific parameters for shrinking-box GROMACS input.",
        ));
    }
    Ok(input)
}

fn gro_text_from_state(
    md: &MdState,
    solute_atom_count: usize,
    solute_mol_count: usize,
    mol_name: &str,
) -> io::Result<String> {
    let offset = md.cell.bounds_low;
    let to_nm = |posit: Vec3F32| -> Vec3F64 {
        let posit = posit - offset;
        Vec3F64::new(
            posit.x as f64 / 10.0,
            posit.y as f64 / 10.0,
            posit.z as f64 / 10.0,
        )
    };
    let mut atoms = Vec::new();
    let mut serial_number = 1_u32;
    let solute_atoms_per_mol = solute_atom_count / solute_mol_count.max(1);
    if solute_atoms_per_mol == 0 || !solute_atom_count.is_multiple_of(solute_mol_count.max(1)) {
        return Err(io::Error::other(
            "Cannot write shrinking-box GRO input with inconsistent solute counts.",
        ));
    }

    for (atom_i, atom) in md.atoms.iter().take(solute_atom_count).enumerate() {
        atoms.push(AtomGro {
            mol_id: (atom_i / solute_atoms_per_mol + 1) as u32,
            mol_name: mol_name.to_string(),
            element: atom.element.clone(),
            atom_type: atom.force_field_type.clone(),
            serial_number,
            posit: to_nm(atom.posit),
            velocity: None,
        });
        serial_number += 1;
    }

    let mut mol_id = solute_mol_count as u32 + 1;

    for water in &md.water {
        for (atom, atom_type) in [
            (&water.o, "OW"),
            (&water.h0, "HW1"),
            (&water.h1, "HW2"),
            (&water.m, "MW"),
        ] {
            atoms.push(AtomGro {
                mol_id,
                mol_name: "SOL".to_string(),
                element: atom.element.clone(),
                atom_type: atom_type.to_string(),
                serial_number,
                posit: to_nm(atom.posit),
                velocity: None,
            });
            serial_number += 1;
        }
        mol_id += 1;
    }

    let gro = Gro {
        atoms,
        head_text: "Shrinking-box property simulation".to_string(),
        box_vec: Vec3F64::new(
            md.cell.extent.x as f64 / 10.0,
            md.cell.extent.y as f64 / 10.0,
            md.cell.extent.z as f64 / 10.0,
        ),
    };
    let mut bytes = Vec::new();
    gro.write_to(&mut bytes)?;
    String::from_utf8(bytes).map_err(io::Error::other)
}

fn run_gromacs(
    mol: &MoleculeSmall,
    setup: ShrinkingBoxSetup,
    placed_solutes: &PlacedSolutes,
    param_set: &FfParamSet,
    dev: &ComputationDevice,
) -> io::Result<(ShrinkingBoxMdData, Vec<Snapshot>)> {
    let md = build_initial_state(setup, placed_solutes, param_set, dev, false, true)?;
    let cfg = make_md_cfg(setup, false, true);
    let solute_input = gromacs_solute_input(&mol.common.ident, &placed_solutes.mols)?;
    let solute_atom_count = solute_input.atoms.len() * solute_input.count;
    let initial_gro = gro_text_from_state(
        &md,
        solute_atom_count,
        setup.solute_copy_count,
        &solute_input.name,
    )?;
    let water_molecule_count = md.water.len();
    let atom_count = physical_atom_count(solute_atom_count, water_molecule_count);
    let mut mdp = cfg.to_gromacs(GROMACS_CHUNK_STEPS, DT_PS);
    mdp.output_control.nstxout = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstvout = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstfout = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstcalcenergy = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstenergy = Some(SNAPSHOT_INTERVAL as u32);
    mdp.deform_init_flow = true;
    let mut input: GromacsInput = make_gromacs_input(
        mdp,
        &[FfMolType::SmallOrganic],
        vec![solute_input],
        param_set,
        &cfg.sim_box,
        &cfg.solvent,
        false,
    )?;

    input.initial_gro = Some(initial_gro);
    input.solvent = Some(GromacsSolvent::WaterOpc);
    input
        .extra_molecule_counts
        .push(("SOL".to_string(), water_molecule_count));
    input.mdrun_extra_args = vec!["-ntmpi".to_string(), "1".to_string()];
    let mut data = initial_data(setup, water_molecule_count);
    let mut snapshots = Vec::new();
    let mut current_cell = setup.initial_cell;
    let mut elapsed_ps = 0.0;
    let mut first_chunk = true;

    while !cell_at_or_below(current_cell, setup.compression_limit_cell) {
        let steps_remaining = shrink_step_count_between(current_cell, setup.compression_limit_cell);
        let chunk_steps = steps_remaining.min(GROMACS_CHUNK_STEPS);
        let next_cell =
            shrink_cell_by_steps(current_cell, setup.compression_limit_cell, chunk_steps);

        input.mdp.nsteps = chunk_steps as u64;
        input.mdp.deform = Some(gromacs_deform_nm_ps(current_cell, next_cell, chunk_steps));
        input.mdp.gen_vel = first_chunk;
        input.mdp.deform_init_flow = first_chunk;
        input.minimize_energy = first_chunk;

        let out = match input.run() {
            Ok(out) => out,
            Err(err) if !snapshots.is_empty() => {
                eprintln!(
                    "Stopping adaptive GROMACS shrinking-box compression after a failed chunk: {err}"
                );
                mark_stop(&mut data, CompressionStopReason::GromacsChunkFailure);
                break;
            }
            Err(err) => return Err(err),
        };

        let max_force = max_gromacs_force(&out);
        let mut chunk_snapshots = gromacs_frames_to_ss(&out);
        for snapshot in &mut chunk_snapshots {
            snapshot.time += elapsed_ps;
        }
        elapsed_ps += chunk_steps as f64 * DT_PS as f64;
        snapshots.extend(chunk_snapshots);
        current_cell = next_cell;
        data.reached_target_box |= cell_at_or_below(current_cell, setup.target_cell);

        if let Some(reason) = latest_stop_reason(&snapshots, atom_count, max_force) {
            mark_stop(&mut data, reason);
            break;
        }

        if cell_at_or_below(current_cell, setup.compression_limit_cell) {
            mark_stop(&mut data, CompressionStopReason::CompressionLimit);
            break;
        }

        input.initial_gro = Some(out.final_gro_text.ok_or_else(|| {
            io::Error::other("GROMACS shrinking-box chunk did not write final coordinates.")
        })?);
        input.topology_override = Some(out.final_topology_text.ok_or_else(|| {
            io::Error::other("GROMACS shrinking-box chunk did not preserve its topology.")
        })?);
        input.skip_counterion_insertion = true;
        first_chunk = false;
    }

    data.final_box_extent_a = current_cell.extent;
    add_snapshot_metrics(&mut data, &snapshots);

    Ok((data, snapshots))
}

/// Run a driven shrinking-box property simulation.
pub fn run_shrinking_box_sim(
    mol: &MoleculeSmall,
    mode: ShrinkingBoxMode,
    backend: MdBackend,
    dev: &ComputationDevice,
    param_set: &FfParamSet,
) -> io::Result<(ShrinkingBoxMdData, Vec<Snapshot>)> {
    let (mol, mol_specific_params) = prepare_mol_for_md(mol, param_set)?;
    let setup = setup_for(&mol, mode)?;
    let template = solute_template(&mol, &mol_specific_params)?;
    let placed_solutes = place_solute_copies(&mol, &template, setup)?;

    println!(
        "Shrinking-box MD setup ({backend}, {mode}): {} solute copies, {} waters, {:.1} A -> adaptive stop (estimated {:.1} A, safety floor {:.1} A)",
        setup.solute_copy_count,
        setup.water_molecule_count,
        setup.initial_cell.extent.x,
        setup.target_cell.extent.x,
        setup.compression_limit_cell.extent.x,
    );

    let result = match backend {
        MdBackend::Dynamics => run_dynamics(setup, &placed_solutes, param_set, dev),
        MdBackend::Gromacs => run_gromacs(&mol, setup, &placed_solutes, param_set, dev),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Shrinking-box MD supports the Dynamics and GROMACS backends.",
        )),
    }?;

    println!(
        "Shrinking-box MD stopped at {:.1} A: {:?} (mean {:.0} K, {:.0} bar, {:.3} g/cm^3)",
        result.0.final_box_extent_a.x,
        result.0.stop_reason,
        result.0.mean_temperature_k,
        result.0.mean_pressure_bar,
        result.0.density_g_cm3,
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use dynamics::snapshot::SnapshotEnergyData;

    use super::*;

    fn snapshot(temp_k: f32, pressure_bar: f32, density_g_cm3: f32, volume_a3: f32) -> Snapshot {
        Snapshot {
            energy_data: Some(SnapshotEnergyData {
                energy_kinetic: 0.0,
                energy_potential: 0.0,
                energy_potential_between_mols: Vec::new(),
                energy_potential_nonbonded: 0.0,
                energy_potential_bonded: 0.0,
                hydrogen_bonds: Vec::new(),
                temperature: temp_k,
                pressure: pressure_bar,
                dh_dl: None,
                volume: volume_a3,
                density: density_g_cm3 / AMU_A3_TO_G_CM3,
            }),
            ..Snapshot::default()
        }
    }

    #[test]
    fn gromacs_chunk_shrinks_one_angstrom_at_the_expected_rate() {
        let current = centered_cube(68.0);
        let limit = centered_cube(24.0);
        let next = shrink_cell_by_steps(current, limit, GROMACS_CHUNK_STEPS);
        let deform = gromacs_deform_nm_ps(current, next, GROMACS_CHUNK_STEPS);

        assert_eq!(next.extent, Vec3F32::splat(67.0));
        assert_eq!(&deform[..3], &[-0.1, -0.1, -0.1]);
    }

    #[test]
    fn adaptive_stop_detects_force_and_atom_density_limits() {
        let stable = snapshot(300.0, 1.0, 1.0, 100.0);

        assert_eq!(
            latest_stop_reason(&[stable.clone()], 1, Some(MAX_FORCE_KCAL_MOL_A)),
            Some(CompressionStopReason::Force)
        );
        assert_eq!(
            latest_stop_reason(&[stable], 19, None),
            Some(CompressionStopReason::AtomDensity)
        );
    }

    #[test]
    fn adaptive_temperature_stop_ignores_a_transient_but_detects_sustained_heat() {
        let hot = snapshot(MAX_TEMPERATURE_K, 1.0, 1.0, 100.0);
        let stable = snapshot(300.0, 1.0, 1.0, 100.0);

        assert_eq!(latest_stop_reason(&[hot.clone()], 1, None), None);

        let sustained = vec![hot; CONTROL_WINDOW_SNAPSHOTS];
        assert_eq!(
            latest_stop_reason(&sustained, 1, None),
            Some(CompressionStopReason::Temperature)
        );

        let mut settling = sustained;
        settling.push(stable);
        assert_eq!(latest_stop_reason(&settling, 1, None), None);
    }
}
