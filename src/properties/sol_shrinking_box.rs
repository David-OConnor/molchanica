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
        GromacsInput, MoleculeInput, OutputControl,
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
const WATER_MOLAR_VOLUME_A3: f32 = 29.97;
const HOMOGENEOUS_SOLUTE_VOLUME_FRACTION: f32 = 0.22;
const HOMOGENEOUS_WATER_VOLUME_FRACTION: f32 = 0.72;
const LAYER_SOLUTE_VOLUME_FRACTION: f32 = 0.42;
const LAYER_WATER_VOLUME_FRACTION: f32 = 0.50;
const LAYER_INTERFACE_GAP_A: f32 = 2.4;
const SOLUTE_WALL_MARGIN_A: f32 = 1.0;

const INITIAL_BOX_SCALE: f32 = 1.8;
const BOX_SHRINK_PER_STEP_A: f32 = 0.04;
const DYNAMICS_EQUILIBRATION_STEPS: usize = 3_000;
const SNAPSHOT_INTERVAL: usize = 10;
const PRESSURE_WINDOW_SNAPSHOTS: usize = 4;
const MAX_SHRINK_PRESSURE_BAR: f32 = 2_500.0;
const SUFFICIENT_DENSITY_G_CM3: f32 = 0.95;
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

#[derive(Clone, Debug)]
pub struct ShrinkingBoxMdData {
    pub mode: ShrinkingBoxMode,
    pub solute_copy_count: usize,
    pub water_molecule_count: usize,
    pub initial_box_extent_a: Vec3F32,
    pub final_box_extent_a: Vec3F32,
    pub target_box_extent_a: Vec3F32,
    pub reached_target_box: bool,
    pub stopped_for_density: bool,
    pub stopped_for_pressure: bool,
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

fn recent_pressure_too_high(snapshots: &[Snapshot]) -> bool {
    let pressures: Vec<_> = snapshots
        .iter()
        .rev()
        .filter_map(|snap| snap.energy_data.as_ref().map(|data| data.pressure))
        .take(PRESSURE_WINDOW_SNAPSHOTS)
        .collect();

    pressures.len() == PRESSURE_WINDOW_SNAPSHOTS
        && mean(&pressures).is_some_and(|pressure| pressure >= MAX_SHRINK_PRESSURE_BAR)
}

fn latest_density_is_sufficient(snapshots: &[Snapshot]) -> bool {
    snapshots
        .last()
        .and_then(|snap| snap.energy_data.as_ref())
        .is_some_and(|data| data.density * AMU_A3_TO_G_CM3 >= SUFFICIENT_DENSITY_G_CM3)
}

fn initial_data(setup: ShrinkingBoxSetup, water_molecule_count: usize) -> ShrinkingBoxMdData {
    ShrinkingBoxMdData {
        mode: setup.mode,
        solute_copy_count: setup.solute_copy_count,
        water_molecule_count,
        initial_box_extent_a: setup.initial_cell.extent,
        final_box_extent_a: setup.initial_cell.extent,
        target_box_extent_a: setup.target_cell.extent,
        reached_target_box: false,
        stopped_for_density: false,
        stopped_for_pressure: false,
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
    let shrink_steps = setup.shrink_cfg.shrink_step_count(setup.target_cell);
    let mut equilibration_steps = 0;

    for _ in 0..shrink_steps + DYNAMICS_EQUILIBRATION_STEPS {
        if !data.reached_target_box && !data.stopped_for_density && !data.stopped_for_pressure {
            if latest_density_is_sufficient(&md.snapshots) {
                data.stopped_for_density = true;
            } else if recent_pressure_too_high(&md.snapshots) {
                data.stopped_for_pressure = true;
            } else {
                let shrank = md.shrink_cell_towards(dev, setup.target_cell, setup.shrink_cfg);
                data.reached_target_box = !shrank || md.cell == setup.target_cell;
            }
        } else {
            equilibration_steps += 1;
        }

        md.step(dev, DT_PS, None);

        if equilibration_steps >= DYNAMICS_EQUILIBRATION_STEPS {
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
    let shrink_steps = setup.shrink_cfg.shrink_step_count(setup.target_cell);
    let cfg = make_md_cfg(setup, false, true);
    let mut mdp = cfg.to_gromacs(shrink_steps, DT_PS);
    mdp.deform = Some(
        setup
            .shrink_cfg
            .gromacs_deform_nm_ps(setup.target_cell, DT_PS),
    );
    mdp.deform_init_flow = true;
    let solute_input = gromacs_solute_input(&mol.common.ident, &placed_solutes.mols)?;
    let solute_atom_count = solute_input.atoms.len() * solute_input.count;
    let initial_gro = gro_text_from_state(
        &md,
        solute_atom_count,
        setup.solute_copy_count,
        &solute_input.name,
    )?;
    let water_molecule_count = md.water.len();
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

    let snapshots = gromacs_frames_to_ss(&input.run()?);
    let mut data = initial_data(setup, water_molecule_count);
    data.final_box_extent_a = setup.target_cell.extent;
    data.reached_target_box = true;
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
        "Shrinking-box MD setup ({backend}, {mode}): {} solute copies, {} waters, {:.1} A -> {:.1} A box",
        setup.solute_copy_count,
        setup.water_molecule_count,
        setup.initial_cell.extent.x,
        setup.target_cell.extent.x,
    );

    match backend {
        MdBackend::Dynamics => run_dynamics(setup, &placed_solutes, param_set, dev),
        MdBackend::Gromacs => run_gromacs(&mol, setup, &placed_solutes, param_set, dev),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Shrinking-box MD supports the Dynamics and GROMACS backends.",
        )),
    }
}
