//! Driven shrinking-box simulations for probing water/solute packing behavior.
//!
//! Unlike the template-preparation helper in `dynamics`, this is the property
//! simulation itself: snapshots cover the gradual compression from a dilute
//! starting cell to a target-density cell.

use crate::{
    gromacs::{make_gromacs_input, molecule_input_from_packed_copies},
    md::MdBackend,
    molecules::{Atom, common::MoleculeCommon, small::MoleculeSmall},
    properties::{AMU_A3_TO_G_CM3, mean, mixing_analysis, mol_bounding_radius, prepare_mol_for_md},
};
use bio_files::{
    Sdf,
    gromacs::{
        GromacsInput, GromacsOutput, MoleculeInput, OutputControl,
        gro::{AtomGro, Gro},
        mdp::{ConstraintAlgorithm, Constraints},
        solvate::Solvent as GromacsSolvent,
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    AtomDynamics, ComputationDevice, FfMolType, HydrogenConstraint, Integrator, MdConfig,
    MdOverrides, MdState, MolDynamics, ShrinkingBoxCfg, SimBox, SimBoxInit, Solvent,
    TAU_TEMP_DEFAULT,
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3 as Vec3F64},
};
use na_seq::Element;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{
    collections::HashMap,
    fmt, fs,
    io::{self, ErrorKind},
};

// Baseline number of solute molecule copies. Molecules large enough that their own
// excluded volume sets the target box use exactly this count. Smaller molecules, whose
// target box is pinned to `MIN_TARGET_BOX_SIDE_A`, instead get *more* copies (see
// `solute_copies_for_fraction`) so the solute volume fraction stays fixed rather than
// falling with molecular size. A higher baseline may produce more reliable and accurate
// results; a smaller one runs faster.
// const SOLUTE_COPY_COUNT: usize = 32;
const SOLUTE_COPY_COUNT: usize = 40;

// Upper bound on the size-compensated copy count, so a very small solute cannot demand a
// pathological number of copies. Solutes tiny enough to hit this cap fall below the target
// volume fraction; everything else is held at it.
const MAX_SOLUTE_COPY_COUNT: usize = 512;

// Smallest target box side length allowed for the final packed cell, in angstroms.
const MIN_TARGET_BOX_SIDE_A: f32 = 34.0;
// Smallest box side length allowed while compressing past the target, in angstroms.
const MIN_COMPRESSION_BOX_SIDE_A: f32 = 24.0;

// Fraction of the target side length used as the compression-limit side length.
// Lower values make the resulting box smaller, with tighter compression.
const COMPRESSION_LIMIT_SCALE: f32 = 0.55;
// Layered slabs get an extra final X-only compression after Y/Z packing. This multiplier
// applies only to that final X push: `2.0` asks the X shrink to continue for twice the
// current distance, subject to `MIN_LAYER_X_PUSH_BOX_SIDE_A` and molecule-size clamping.
const LAYER_X_PUSH_DURATION_SCALE: f32 = 2.0;
// Layered-only lower bound for the final X side length. Lower values permit a smaller,
// higher-pressure final layered box, but may trigger stop guards earlier.
const MIN_LAYER_X_PUSH_BOX_SIDE_A: f32 = MIN_COMPRESSION_BOX_SIDE_A * 0.5;
// If true, slab mode first compresses Y/Z to pack the solute layer before the final X push.
// If false, slab mode only compresses along X, pushing the water and solute slabs together.
const SLAB_COMPRESS_LATERALLY: bool = true;

// Approximate volume occupied by one water molecule, in cubic angstroms.
const WATER_MOLAR_VOLUME_A3: f32 = 29.97;
// Extra OPC waters added after the mode-specific target water volume is computed.
const WATER_MOLECULE_COUNT_SCALE: f32 = 1.20;

// Target solute volume fraction for homogeneous solute/water packing. Lower values
// enlarge the final target box for the same solute volume and therefore add more water.
const HOMOGENEOUS_SOLUTE_VOLUME_FRACTION: f32 = 0.14;

// Target water volume fraction for homogeneous solute/water packing. Keep this below
// `1.0 - HOMOGENEOUS_SOLUTE_VOLUME_FRACTION` to leave modest packing headroom near
// the target cell.
const HOMOGENEOUS_WATER_VOLUME_FRACTION: f32 = 0.84;

// Target solute volume fraction when starting from layered slabs.
const LAYER_SOLUTE_VOLUME_FRACTION: f32 = 0.42;
// Target water volume fraction when starting from layered slabs.
const LAYER_WATER_VOLUME_FRACTION: f32 = 0.50;

// Empty spacing kept between the initial water and solute slabs, in angstroms.
const LAYER_INTERFACE_GAP_A: f32 = 2.4;
// Fractional X position of the center of the water/solute gap. Values above 0.5
// give the low-X solute slab a thicker starting layer than the high-X water slab.
const LAYER_SOLUTE_SLAB_FRACTION: f32 = 0.58;
// Phase-one handoff targets for layered boxes. The pressure threshold is deliberately
// below the hard safety stop because small driven boxes have noisy virials.
const LAYER_PACK_TARGET_DENSITY_G_CM3: f32 = 0.95;
const LAYER_PACK_PRESSURE_HANDOFF_BAR: f32 = 5_000.0;

// Minimum distance from each solute bounding sphere to the box wall, in angstroms.
const SOLUTE_WALL_MARGIN_A: f32 = 1.0;
// Minimum extra center-to-center clearance between initial solute copies, in angstroms.
const INITIAL_SOLUTE_CLEARANCE_A: f32 = 1.5;

// Starting box side-length multiplier relative to the target cell. Reduce this if
// the starting concentration is too dilute. Higher values
// require more steps, and produces a more dramatic compression.
const INITIAL_BOX_SCALE: f32 = 3.0;

// Box side-length reduction applied per MD shrink step, in angstroms.
// Higher values result in a briefer simulation; too high may produce anomalies in
// the system.
// const BOX_SHRINK_PER_STEP: f32 = 0.006;
const BOX_SHRINK_PER_STEP: f32 = 0.012;
// Use a slower final X push for layered slabs, giving the solvent/solute interface
// more steps to relax as it is compressed together.
const LAYER_X_PUSH_SHRINK_PER_STEP: f32 = BOX_SHRINK_PER_STEP * 0.5;

// Fixed-cell relaxation length after driven shrinking, in MD steps.
const EQUILIBRATION_STEPS: usize = 2_000;

// Interval between saved snapshots and backend energy outputs, in MD steps.
const SNAPSHOT_INTERVAL: usize = 10;
// Number of recent snapshots averaged for stop-condition control signals.
const CONTROL_WINDOW_SNAPSHOTS: usize = 5;

// Number of MD shrink steps batched into one deform run. Lower this to
// make the simulation faster?
const SHRINK_CHUNK_STEPS: usize = 250;

// Maximum recent mean pressure before stopping compression, in bar.
// const MAX_SHRINK_PRESSURE_BAR: f32 = 10_000.0;
const MAX_SHRINK_PRESSURE_BAR: f32 = 20_000.0;
// Maximum recent mean temperature before stopping compression, in kelvin.
const MAX_TEMPERATURE_K: f32 = 450.0;
// Maximum mass density before stopping compression, in grams per cubic centimeter.
const MAX_DENSITY_G_CM3: f32 = 2.5;
// Maximum atom number density before stopping compression, in atoms per cubic angstrom.
const MAX_ATOM_DENSITY_A3: f32 = 0.18;
// Maximum force magnitude before stopping compression, in kcal/mol/angstrom.
const MAX_FORCE_KCAL_MOL_A: f32 = 500.0;

// Thermostat target temperature used by both backends, in kelvin.
const TEMP: f32 = 300.0;

// MD integration timestep used by both backends, in picoseconds.
const DT: f32 = 0.001;

const SHRINK_TAU_TEMP_PS: f64 = 0.05;
const SHRINK_LINCS_ORDER: u8 = 8;
const SHRINK_LINCS_ITER: u8 = 2;
const PRE_SHRINK_MINIMIZATION_STEPS: usize = 5_000;

const AVOGADRO_PER_MOL: f64 = 6.022_140_76e23;
const A3_TO_L: f64 = 1.0e-27;

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
    pub final_solute_molarity_m: f32,
    pub potential_energy_kcal: f32,
    pub nonbonded_energy_kcal: f32,
    pub solubility_estimate: f32,
    pub solubility_estimate_barnes_hut: f32,
    pub solubility_raw_score: f32,
    pub solubility_local_mixing: f32,
    pub solubility_solute_dispersion: f32,
    pub solubility_mixture_score: f32,
    pub solubility_aggregation_factor: f32,
    pub solubility_aggregation_penalty: f32,
    pub solubility_largest_cluster_fraction: f32,
    pub solubility_contacted_fraction: f32,
    pub solubility_contact_pair_fraction: f32,
}

pub struct ShrinkingBoxPlaybackMol {
    pub mol_type: FfMolType,
    pub mol: MoleculeCommon,
    pub count: usize,
}

#[derive(Clone, Copy, Debug)]
struct ShrinkingBoxSetup {
    mode: ShrinkingBoxMode,
    solute_copy_count: usize,
    water_molecule_count: usize,
    target_cell: SimBox,
    compression_limit_cell: SimBox,
    layer_x_push_min_side_a: f32,
    initial_cell: SimBox,
    shrink_cfg: ShrinkingBoxCfg,
}

struct PlacedSolutes {
    mols: Vec<MolDynamics>,
}

struct ShrinkingBoxBackendResult {
    data: ShrinkingBoxMdData,
    snapshots: Vec<Snapshot>,
    extra_non_water_mols: Vec<MoleculeCommon>,
}

fn centered_cube(side: f32) -> SimBox {
    let half = side / 2.0;
    SimBox::new(
        Vec3F32::new(-half, -half, -half),
        Vec3F32::new(half, half, half),
    )
}

fn centered_cell_with_extent(extent: Vec3F32) -> SimBox {
    let half = extent / 2.0;
    SimBox::new(-half, half)
}

fn solute_grid_inset_a(mol_radius_a: f32) -> f32 {
    SOLUTE_WALL_MARGIN_A + mol_radius_a
}

fn solute_region_extent(initial_side_a: f32, mode: ShrinkingBoxMode) -> Vec3F32 {
    match mode {
        ShrinkingBoxMode::HomogeneousMix => Vec3F32::splat(initial_side_a),
        ShrinkingBoxMode::WaterSoluteLayers => Vec3F32::new(
            initial_side_a * LAYER_SOLUTE_SLAB_FRACTION - LAYER_INTERFACE_GAP_A / 2.0,
            initial_side_a,
            initial_side_a,
        ),
    }
}

fn layer_gap_center_x(cell: SimBox) -> f32 {
    cell.bounds_low.x + cell.extent.x * LAYER_SOLUTE_SLAB_FRACTION
}

fn layer_solute_high_x(cell: SimBox) -> f32 {
    layer_gap_center_x(cell) - LAYER_INTERFACE_GAP_A / 2.0
}

fn layer_solvent_low_x(cell: SimBox) -> f32 {
    layer_gap_center_x(cell) + LAYER_INTERFACE_GAP_A / 2.0
}

fn initial_cell_for_solutes(
    target_side_a: f32,
    mode: ShrinkingBoxMode,
    copies: usize,
    mol_radius_a: f32,
) -> SimBox {
    let inset_a = solute_grid_inset_a(mol_radius_a);
    let min_spacing_a = 2.0 * mol_radius_a + INITIAL_SOLUTE_CLEARANCE_A;
    let mut side_a = target_side_a * INITIAL_BOX_SCALE;

    loop {
        let usable_extent = solute_region_extent(side_a, mode) - Vec3F32::splat(2.0 * inset_a);
        if usable_extent.x > 0.0 && usable_extent.y > 0.0 && usable_extent.z > 0.0 {
            let (nx, ny, nz) = grid_dims(copies, usable_extent);
            let spacing = Vec3F32::new(
                usable_extent.x / nx as f32,
                usable_extent.y / ny as f32,
                usable_extent.z / nz as f32,
            );

            if spacing.x >= min_spacing_a
                && spacing.y >= min_spacing_a
                && spacing.z >= min_spacing_a
            {
                return centered_cube(side_a);
            }

            let scale = (min_spacing_a / spacing.x)
                .max(min_spacing_a / spacing.y)
                .max(min_spacing_a / spacing.z)
                .max(1.05);
            side_a *= scale;
        } else {
            side_a *= 1.5;
        }
    }
}

/// Number of solute copies needed to fill `box_volume_a3` to `solute_volume_fraction`,
/// mirroring how the water count fills the same cell to its water fraction. Holding this
/// fraction fixed across molecules - rather than the copy count - keeps small and large
/// solutes at the same concentration, so the mixing score reflects solubility rather than
/// molecular size. Clamped to `[SOLUTE_COPY_COUNT, MAX_SOLUTE_COPY_COUNT]`.
fn solute_copies_for_fraction(
    box_volume_a3: f32,
    mol_volume_a3: f32,
    solute_volume_fraction: f32,
) -> usize {
    let ideal = (solute_volume_fraction * box_volume_a3 / mol_volume_a3.max(1.0)).round();

    (ideal as usize).clamp(SOLUTE_COPY_COUNT, MAX_SOLUTE_COPY_COUNT)
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
    // Box volume that realizes `solute_volume_fraction` at the baseline copy count. For
    // large molecules this sets the target box; for small ones the `min_volume_a3` floor
    // wins, and we compensate below by scaling the copy count instead of the box.
    let baseline_solute_volume_a3 =
        SOLUTE_COPY_COUNT as f32 * mol_volume_a3 / solute_volume_fraction;
    let mol_radius_a = mol_bounding_radius(mol);
    let molecule_limited_side_a = match mode {
        ShrinkingBoxMode::HomogeneousMix => 2.0 * (mol_radius_a + SOLUTE_WALL_MARGIN_A),
        ShrinkingBoxMode::WaterSoluteLayers => {
            4.0 * (mol_radius_a + SOLUTE_WALL_MARGIN_A) + LAYER_INTERFACE_GAP_A + 2.0
        }
    };
    let target_side_a = min_volume_a3
        .max(baseline_solute_volume_a3)
        .cbrt()
        .max(molecule_limited_side_a);
    let target_volume_a3 = target_side_a.powi(3);
    let target_cell = centered_cube(target_side_a);
    // Hold the solute volume fraction fixed across molecules: fill the *actual* target
    // cell (which the floors above may have forced larger than `baseline_solute_volume_a3`)
    // to `solute_volume_fraction`, exactly as the water count fills it to the water
    // fraction. Without this, the box floor leaves small molecules far more dilute than
    // large ones, and the mixing score tracks molecular size instead of solubility.
    let solute_copy_count =
        solute_copies_for_fraction(target_volume_a3, mol_volume_a3, solute_volume_fraction);
    let compression_limit_side_a = (target_side_a * COMPRESSION_LIMIT_SCALE)
        .max(MIN_COMPRESSION_BOX_SIDE_A)
        .max(molecule_limited_side_a);
    let compression_limit_cell = centered_cube(compression_limit_side_a);
    let layer_x_push_min_side_a = MIN_LAYER_X_PUSH_BOX_SIDE_A.max(molecule_limited_side_a);
    let target_water_volume_a3 = match mode {
        // Leave headroom for solute excluded volume and driven compression.
        ShrinkingBoxMode::HomogeneousMix => target_volume_a3 * HOMOGENEOUS_WATER_VOLUME_FRACTION,
        ShrinkingBoxMode::WaterSoluteLayers => target_volume_a3 * LAYER_WATER_VOLUME_FRACTION,
    };
    let water_molecule_count = ((target_water_volume_a3 * WATER_MOLECULE_COUNT_SCALE
        / WATER_MOLAR_VOLUME_A3)
        .round() as usize)
        .max(1);
    let shrink_cfg = ShrinkingBoxCfg {
        initial_box_scale: INITIAL_BOX_SCALE,
        box_shrink_per_step: BOX_SHRINK_PER_STEP,
    };
    let initial_cell =
        initial_cell_for_solutes(target_side_a, mode, solute_copy_count, mol_radius_a);

    Ok(ShrinkingBoxSetup {
        mode,
        solute_copy_count,
        water_molecule_count,
        target_cell,
        compression_limit_cell,
        layer_x_push_min_side_a,
        initial_cell,
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
            let high_x = layer_solute_high_x(setup.initial_cell);
            SimBox::new(
                setup.initial_cell.bounds_low,
                Vec3F32::new(
                    high_x,
                    setup.initial_cell.bounds_high.y,
                    setup.initial_cell.bounds_high.z,
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
    // The setup has already enlarged the dilute initial cell enough that bounding
    // spheres in neighboring grid cells cannot overlap, regardless of rotation.
    let inset_a = solute_grid_inset_a(mol_bounding_radius(mol));
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
        let rotation = Quaternion::random(&mut rng, None);
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
        temp_target: TEMP,
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

fn move_water_to_solvent_layer(md: &mut MdState) {
    let cell = md.cell;
    let low_x = layer_solvent_low_x(cell);
    let high_x = cell.bounds_high.x;
    let scale_x = (high_x - low_x) / cell.extent.x;

    for water in &mut md.water {
        let old_x = water.o.posit.x;
        let new_x = low_x + (old_x - cell.bounds_low.x) * scale_x;
        let dx = new_x - old_x;
        water.o.posit.x += dx;
        water.h0.posit.x += dx;
        water.h1.posit.x += dx;
        water.m.posit.x += dx;
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
            move_water_to_solvent_layer(&mut md);
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

fn should_run_post_shrink_equilibration(reason: Option<CompressionStopReason>) -> bool {
    !matches!(reason, Some(CompressionStopReason::GromacsChunkFailure))
}

fn cell_at_or_below(cell: SimBox, limit: SimBox) -> bool {
    cell.extent.x <= limit.extent.x
        && cell.extent.y <= limit.extent.y
        && cell.extent.z <= limit.extent.z
}

fn layer_yz_pack_cell(setup: ShrinkingBoxSetup) -> SimBox {
    centered_cell_with_extent(Vec3F32::new(
        setup.initial_cell.extent.x,
        setup.compression_limit_cell.extent.y,
        setup.compression_limit_cell.extent.z,
    ))
}

fn layer_x_push_cell(current_cell: SimBox, setup: ShrinkingBoxSetup) -> SimBox {
    let default_x_limit_a = setup.compression_limit_cell.extent.x;
    let current_x_a = current_cell.extent.x;
    let x_shrink_a = (current_x_a - default_x_limit_a).max(0.0);
    let extended_x_limit_a = current_x_a - x_shrink_a * LAYER_X_PUSH_DURATION_SCALE.max(1.0);
    let x_limit_a = extended_x_limit_a
        .max(setup.layer_x_push_min_side_a)
        .min(default_x_limit_a);

    centered_cell_with_extent(Vec3F32::new(
        x_limit_a,
        current_cell.extent.y,
        current_cell.extent.z,
    ))
}

fn latest_layer_pack_handoff_reason(snapshots: &[Snapshot]) -> Option<CompressionStopReason> {
    let latest = snapshots.last()?.energy_data.as_ref()?;
    let density_g_cm3 = latest.density * AMU_A3_TO_G_CM3;
    if density_g_cm3.is_finite() && density_g_cm3 >= LAYER_PACK_TARGET_DENSITY_G_CM3 {
        return Some(CompressionStopReason::MassDensity);
    }

    recent_energy_mean(snapshots, |data| data.pressure)
        .is_some_and(|pressure| pressure.is_finite() && pressure >= LAYER_PACK_PRESSURE_HANDOFF_BAR)
        .then_some(CompressionStopReason::Pressure)
}

fn shrink_step_count_between(
    initial_cell: SimBox,
    target_cell: SimBox,
    shrink_per_step: f32,
) -> usize {
    let shrink_needed = initial_cell.extent - target_cell.extent;
    let max_shrink = shrink_needed.x.max(shrink_needed.y).max(shrink_needed.z);
    (max_shrink / shrink_per_step.max(f32::EPSILON)).ceil() as usize
}

fn shrink_cell_by_amount(current_cell: SimBox, limit_cell: SimBox, shrink_a: f32) -> SimBox {
    let extent = Vec3F32::new(
        (current_cell.extent.x - shrink_a).max(limit_cell.extent.x),
        (current_cell.extent.y - shrink_a).max(limit_cell.extent.y),
        (current_cell.extent.z - shrink_a).max(limit_cell.extent.z),
    );
    let center = current_cell.center();
    let half = extent / 2.0;
    SimBox::new(center - half, center + half)
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

fn gromacs_deform_nm_ps(
    current_cell: SimBox,
    next_cell: SimBox,
    steps: usize,
    dt_ps: f32,
) -> [f32; 6] {
    let duration_ps = steps.max(1) as f32 * dt_ps;
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
        final_solute_molarity_m: 0.0,
        potential_energy_kcal: 0.0,
        nonbonded_energy_kcal: 0.0,
        solubility_estimate: 0.0,
        solubility_estimate_barnes_hut: 0.0,
        solubility_raw_score: 0.0,
        solubility_local_mixing: 0.0,
        solubility_solute_dispersion: 0.0,
        solubility_mixture_score: 0.0,
        solubility_aggregation_factor: 0.0,
        solubility_aggregation_penalty: 0.0,
        solubility_largest_cluster_fraction: 0.0,
        solubility_contacted_fraction: 0.0,
        solubility_contact_pair_fraction: 0.0,
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

fn solute_molarity_m(copy_count: usize, cell: &SimBox) -> f32 {
    let volume_l = cell.volume() as f64 * A3_TO_L;
    if volume_l <= f64::EPSILON {
        return 0.0;
    }

    (copy_count as f64 / (AVOGADRO_PER_MOL * volume_l)) as f32
}

fn add_solubility_diagnostics(
    data: &mut ShrinkingBoxMdData,
    diagnostics: mixing_analysis::SolubilityMixingDiagnostics,
    barnes_hut: f32,
    final_cell: &SimBox,
) {
    data.final_solute_molarity_m = solute_molarity_m(data.solute_copy_count, final_cell);
    data.solubility_estimate = diagnostics.score;
    data.solubility_estimate_barnes_hut = barnes_hut;
    data.solubility_raw_score = diagnostics.raw_score;
    data.solubility_local_mixing = diagnostics.local_mixing;
    data.solubility_solute_dispersion = diagnostics.solute_dispersion;
    data.solubility_mixture_score = diagnostics.mixture_score;
    data.solubility_aggregation_factor = diagnostics.aggregation_factor;
    data.solubility_aggregation_penalty = diagnostics.aggregation_penalty;
    data.solubility_largest_cluster_fraction = diagnostics.largest_cluster_fraction;
    data.solubility_contacted_fraction = diagnostics.contacted_fraction;
    data.solubility_contact_pair_fraction = diagnostics.contact_pair_fraction;
}

fn single_atom_viewer_mol(atom: &AtomDynamics) -> MoleculeCommon {
    let posit: Vec3F64 = atom.posit.into();
    let ident = if atom.force_field_type.is_empty() {
        atom.element.to_letter()
    } else {
        atom.force_field_type.clone()
    };
    let atom = Atom {
        serial_number: atom.serial_number,
        posit,
        element: atom.element,
        type_in_res_general: Some(ident.clone()),
        force_field_type: Some(ident.clone()),
        hetero: true,
        ..Default::default()
    };

    MoleculeCommon {
        ident,
        atoms: vec![atom],
        atom_posits: vec![posit],
        selected_for_md: Some(1),
        ..Default::default()
    }
}

fn md_extra_non_water_mols(md: &MdState, placed_solutes: &PlacedSolutes) -> Vec<MoleculeCommon> {
    let solute_atom_count: usize = placed_solutes.mols.iter().map(|mol| mol.atoms.len()).sum();

    md.atoms
        .iter()
        .skip(solute_atom_count)
        .map(single_atom_viewer_mol)
        .collect()
}

fn prepare_dynamics_shrinking_state(md: &mut MdState, dev: &ComputationDevice) {
    md.cfg.integrator = Integrator::VerletVelocity {
        thermostat: Some(SHRINK_TAU_TEMP_PS),
    };
    md.cfg.hydrogen_constraint = HydrogenConstraint::Linear {
        order: SHRINK_LINCS_ORDER,
        iter: SHRINK_LINCS_ITER,
    };

    md.minimize_energy(dev, PRE_SHRINK_MINIMIZATION_STEPS, None);

    let zero_com_drift = md.cfg.zero_com_drift;
    md.initialize_velocities(TEMP, zero_com_drift);
}

fn run_dynamics_shrink_phase(
    md: &mut MdState,
    dev: &ComputationDevice,
    setup: ShrinkingBoxSetup,
    data: &mut ShrinkingBoxMdData,
    atom_count: usize,
    phase_target_cell: SimBox,
    box_shrink_per_step: f32,
    layer_pack_handoff: bool,
    mark_compression_limit_on_target: bool,
) {
    while data.stop_reason.is_none() && !cell_at_or_below(md.cell, phase_target_cell) {
        let shrink_steps_remaining =
            shrink_step_count_between(md.cell, phase_target_cell, box_shrink_per_step);
        let chunk_steps = shrink_steps_remaining.min(SHRINK_CHUNK_STEPS).max(1);
        let mut reached_phase_target = false;
        let shrink_cfg = ShrinkingBoxCfg {
            box_shrink_per_step,
            ..setup.shrink_cfg
        };

        for _ in 0..chunk_steps {
            let shrank = md.shrink_cell_towards(dev, phase_target_cell, shrink_cfg);
            data.reached_target_box |= cell_at_or_below(md.cell, setup.target_cell);

            reached_phase_target |= !shrank || cell_at_or_below(md.cell, phase_target_cell);

            md.step(dev, DT, None);

            if reached_phase_target {
                break;
            }
        }

        if let Some(reason) = latest_stop_reason(&md.snapshots, atom_count, max_dynamics_force(md))
        {
            mark_stop(data, reason);
            break;
        }

        if layer_pack_handoff && latest_layer_pack_handoff_reason(&md.snapshots).is_some() {
            break;
        }

        if reached_phase_target {
            if mark_compression_limit_on_target {
                mark_stop(data, CompressionStopReason::CompressionLimit);
            }
            break;
        }
    }

    if mark_compression_limit_on_target
        && data.stop_reason.is_none()
        && cell_at_or_below(md.cell, phase_target_cell)
    {
        mark_stop(data, CompressionStopReason::CompressionLimit);
    }
}

fn run_dynamics(
    setup: ShrinkingBoxSetup,
    placed_solutes: &PlacedSolutes,
    param_set: &FfParamSet,
    dev: &ComputationDevice,
) -> io::Result<ShrinkingBoxBackendResult> {
    let mut md = build_initial_state(setup, placed_solutes, param_set, dev, true, false)?;
    prepare_dynamics_shrinking_state(&mut md, dev);

    let mut data = initial_data(setup, md.water.len());
    let atom_count = physical_atom_count(md.atoms.len(), md.water.len());

    println!("Starting Dynamics shrinking box loop...");
    match setup.mode {
        ShrinkingBoxMode::HomogeneousMix => {
            run_dynamics_shrink_phase(
                &mut md,
                dev,
                setup,
                &mut data,
                atom_count,
                setup.compression_limit_cell,
                BOX_SHRINK_PER_STEP,
                false,
                true,
            );
        }
        ShrinkingBoxMode::WaterSoluteLayers => {
            if SLAB_COMPRESS_LATERALLY {
                println!("Starting Dynamics Y/Z layer packing phase...");
                run_dynamics_shrink_phase(
                    &mut md,
                    dev,
                    setup,
                    &mut data,
                    atom_count,
                    layer_yz_pack_cell(setup),
                    BOX_SHRINK_PER_STEP,
                    true,
                    false,
                );
            }

            if data.stop_reason.is_none() {
                println!("Starting Dynamics X layer compression phase...");
                let x_push_cell = layer_x_push_cell(md.cell, setup);
                data.compression_limit_box_extent_a = x_push_cell.extent;
                run_dynamics_shrink_phase(
                    &mut md,
                    dev,
                    setup,
                    &mut data,
                    atom_count,
                    x_push_cell,
                    LAYER_X_PUSH_SHRINK_PER_STEP,
                    false,
                    true,
                );
            }
        }
    }

    if should_run_post_shrink_equilibration(data.stop_reason) && !md.snapshots.is_empty() {
        for _ in 0..EQUILIBRATION_STEPS {
            md.step(dev, DT, None);
        }
    }

    data.final_box_extent_a = md.cell.extent;
    add_snapshot_metrics(&mut data, &md.snapshots);

    let extra_non_water_mols = md_extra_non_water_mols(&md, placed_solutes);

    Ok(ShrinkingBoxBackendResult {
        data,
        snapshots: md.snapshots,
        extra_non_water_mols,
    })
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
    // Keep coordinates centered around the Molchanica cell origin. GROMACS `deform`
    // applies its flow field relative to the coordinate origin, so translating into
    // a 0..L frame would make compression pull from one corner.
    let to_nm = centered_posit_nm;
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

fn centered_posit_nm(posit: Vec3F32) -> Vec3F64 {
    Vec3F64::new(
        posit.x as f64 / 10.0,
        posit.y as f64 / 10.0,
        posit.z as f64 / 10.0,
    )
}

fn append_gromacs_snapshots(
    snapshots: &mut Vec<Snapshot>,
    out: &GromacsOutput,
    elapsed_ps: &mut f64,
    steps: usize,
) {
    let mut new_snapshots = gromacs_frames_to_ss(out);
    for snapshot in &mut new_snapshots {
        snapshot.time += *elapsed_ps;
    }
    *elapsed_ps += steps as f64 * DT as f64;

    snapshots.extend(new_snapshots);
}

fn update_gromacs_restart_input(
    input: &mut GromacsInput,
    out: &mut GromacsOutput,
    context: &str,
) -> io::Result<()> {
    input.initial_gro = Some(out.final_gro_text.take().ok_or_else(|| {
        io::Error::other(format!(
            "GROMACS shrinking-box {context} did not write final coordinates."
        ))
    })?);
    input.topology_override = Some(out.final_topology_text.take().ok_or_else(|| {
        io::Error::other(format!(
            "GROMACS shrinking-box {context} did not preserve its topology."
        ))
    })?);
    input.skip_counterion_insertion = true;
    input.minimize_energy = false;

    Ok(())
}

fn run_gromacs_shrink_phase(
    input: &mut GromacsInput,
    snapshots: &mut Vec<Snapshot>,
    elapsed_ps: &mut f64,
    current_cell: &mut SimBox,
    data: &mut ShrinkingBoxMdData,
    atom_count: usize,
    setup_target_cell: SimBox,
    phase_target_cell: SimBox,
    box_shrink_per_step: f32,
    layer_pack_handoff: bool,
    mark_compression_limit_on_target: bool,
    initialize_deform_flow: &mut bool,
    generate_initial_velocities: &mut bool,
) -> io::Result<()> {
    while data.stop_reason.is_none() && !cell_at_or_below(*current_cell, phase_target_cell) {
        let next_cell = shrink_cell_by_amount(
            *current_cell,
            phase_target_cell,
            box_shrink_per_step * SHRINK_CHUNK_STEPS as f32,
        );

        input.mdp.nsteps = SHRINK_CHUNK_STEPS as u64;
        input.mdp.dt = DT;
        input.mdp.deform = Some(gromacs_deform_nm_ps(
            *current_cell,
            next_cell,
            SHRINK_CHUNK_STEPS,
            DT,
        ));
        input.mdp.gen_vel = *generate_initial_velocities;
        input.mdp.deform_init_flow = *initialize_deform_flow;
        input.minimize_energy = *generate_initial_velocities;

        let mut out = match input.run() {
            Ok(out) => out,
            Err(err) if !snapshots.is_empty() => {
                eprintln!(
                    "Stopping adaptive GROMACS shrinking-box compression after a failed chunk: {err}"
                );
                mark_stop(data, CompressionStopReason::GromacsChunkFailure);
                break;
            }
            Err(err) => return Err(err),
        };

        let max_force = max_gromacs_force(&out);
        append_gromacs_snapshots(snapshots, &out, elapsed_ps, SHRINK_CHUNK_STEPS);
        *current_cell = next_cell;
        update_gromacs_restart_input(input, &mut out, "chunk")?;
        *generate_initial_velocities = false;
        *initialize_deform_flow = false;

        data.reached_target_box |= cell_at_or_below(*current_cell, setup_target_cell);

        if let Some(reason) = latest_stop_reason(snapshots, atom_count, max_force) {
            mark_stop(data, reason);
            break;
        }

        if layer_pack_handoff && latest_layer_pack_handoff_reason(snapshots).is_some() {
            break;
        }

        if cell_at_or_below(*current_cell, phase_target_cell) {
            if mark_compression_limit_on_target {
                mark_stop(data, CompressionStopReason::CompressionLimit);
            }
            break;
        }
    }

    if mark_compression_limit_on_target
        && data.stop_reason.is_none()
        && cell_at_or_below(*current_cell, phase_target_cell)
    {
        mark_stop(data, CompressionStopReason::CompressionLimit);
    }

    Ok(())
}

fn run_gromacs(
    mol: &MoleculeSmall,
    setup: ShrinkingBoxSetup,
    placed_solutes: &PlacedSolutes,
    param_set: &FfParamSet,
    dev: &ComputationDevice,
) -> io::Result<ShrinkingBoxBackendResult> {
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

    let mut mdp = cfg.to_gromacs(SHRINK_CHUNK_STEPS, DT);

    mdp.output_control.nstxout = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstvout = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstfout = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstcalcenergy = Some(SNAPSHOT_INTERVAL as u32);
    mdp.output_control.nstenergy = Some(SNAPSHOT_INTERVAL as u32);
    mdp.tau_t = vec![0.05];
    mdp.constraints = Constraints::HBonds(ConstraintAlgorithm::Lincs { order: 8, iter: 2 });

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

    println!("Starting GROMACS shrinking box loop...");
    let mut initialize_deform_flow = true;
    let mut generate_initial_velocities = true;
    match setup.mode {
        ShrinkingBoxMode::HomogeneousMix => {
            run_gromacs_shrink_phase(
                &mut input,
                &mut snapshots,
                &mut elapsed_ps,
                &mut current_cell,
                &mut data,
                atom_count,
                setup.target_cell,
                setup.compression_limit_cell,
                BOX_SHRINK_PER_STEP,
                false,
                true,
                &mut initialize_deform_flow,
                &mut generate_initial_velocities,
            )?;
        }
        ShrinkingBoxMode::WaterSoluteLayers => {
            if SLAB_COMPRESS_LATERALLY {
                println!("Starting GROMACS Y/Z layer packing phase...");
                run_gromacs_shrink_phase(
                    &mut input,
                    &mut snapshots,
                    &mut elapsed_ps,
                    &mut current_cell,
                    &mut data,
                    atom_count,
                    setup.target_cell,
                    layer_yz_pack_cell(setup),
                    BOX_SHRINK_PER_STEP,
                    true,
                    false,
                    &mut initialize_deform_flow,
                    &mut generate_initial_velocities,
                )?;
            }

            if data.stop_reason.is_none() {
                println!("Starting GROMACS X layer compression phase...");
                let x_push_cell = layer_x_push_cell(current_cell, setup);
                data.compression_limit_box_extent_a = x_push_cell.extent;
                run_gromacs_shrink_phase(
                    &mut input,
                    &mut snapshots,
                    &mut elapsed_ps,
                    &mut current_cell,
                    &mut data,
                    atom_count,
                    setup.target_cell,
                    x_push_cell,
                    LAYER_X_PUSH_SHRINK_PER_STEP,
                    false,
                    true,
                    &mut initialize_deform_flow,
                    &mut generate_initial_velocities,
                )?;
            }
        }
    }

    if should_run_post_shrink_equilibration(data.stop_reason) && !snapshots.is_empty() {
        println!("Starting GROMACS post-shrink equilibration...");

        input.mdp.nsteps = EQUILIBRATION_STEPS as u64;
        input.mdp.dt = DT;
        input.mdp.deform = None;
        input.mdp.gen_vel = false;
        input.mdp.deform_init_flow = false;
        input.minimize_energy = false;

        let mut out = input.run()?;
        append_gromacs_snapshots(&mut snapshots, &out, &mut elapsed_ps, EQUILIBRATION_STEPS);
        update_gromacs_restart_input(&mut input, &mut out, "post-shrink equilibration")?;
    }

    data.final_box_extent_a = current_cell.extent;
    add_snapshot_metrics(&mut data, &snapshots);

    Ok(ShrinkingBoxBackendResult {
        data,
        snapshots,
        extra_non_water_mols: Vec::new(),
    })
}

/// Run a driven shrinking-box property simulation.
pub fn run_shrinking_box_sim(
    mol: &MoleculeSmall,
    mode: ShrinkingBoxMode,
    backend: MdBackend,
    dev: &ComputationDevice,
    param_set: &FfParamSet,
) -> io::Result<(
    ShrinkingBoxMdData,
    Vec<Snapshot>,
    Vec<ShrinkingBoxPlaybackMol>,
)> {
    let start = Instant::now();

    let (mol, mol_specific_params) = prepare_mol_for_md(mol, param_set)?;
    let setup = setup_for(&mol, mode)?;
    let template = solute_template(&mol, &mol_specific_params)?;
    let placed_solutes = place_solute_copies(&mol, &template, setup)?;
    let atoms_per_solute = template.atoms.len();
    let solute_atom_count: usize = placed_solutes.mols.iter().map(|mol| mol.atoms.len()).sum();

    let mut solubility_atom_indices: Vec<_> = mol
        .common
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| (atom.element != Element::Hydrogen).then_some(i))
        .collect();
    if solubility_atom_indices.is_empty() {
        solubility_atom_indices = (0..atoms_per_solute).collect();
    }

    println!(
        "Shrinking-box MD setup ({backend}, {mode}): {} solute copies, {} waters, {:.1} A -> adaptive stop (estimated {:.1} A, safety floor {:.1} A)",
        setup.solute_copy_count,
        setup.water_molecule_count,
        setup.initial_cell.extent.x,
        setup.target_cell.extent.x,
        setup.compression_limit_cell.extent.x,
    );

    let mut result = match backend {
        MdBackend::Dynamics => run_dynamics(setup, &placed_solutes, param_set, dev),
        MdBackend::Gromacs => run_gromacs(&mol, setup, &placed_solutes, param_set, dev),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Shrinking-box MD supports the Dynamics and GROMACS backends.",
        )),
    }?;

    println!(
        "Shrinking-box MD stopped at {:.1} A: {:?} (mean {:.0} K, {:.0} bar, {:.3} g/cm^3)",
        result.data.final_box_extent_a.x,
        result.data.stop_reason,
        result.data.mean_temperature_k,
        result.data.mean_pressure_bar,
        result.data.density_g_cm3,
    );

    let (solubility_diagnostics, solubility_estimate_barnes_hut, final_cell) = {
        let ss = result.snapshots.last().unwrap(); // todo: ERrror handling

        let solute_end = solute_atom_count.min(ss.atom_posits.len());
        let center = setup.initial_cell.center();
        let half_extent = result.data.final_box_extent_a / 2.0;
        let final_cell = SimBox::new(center - half_extent, center + half_extent);

        (
            mixing_analysis::compute_solubility_diagnostics(
                &ss.atom_posits[..solute_end],
                atoms_per_solute,
                &solubility_atom_indices,
                &ss.water_o_posits,
                &final_cell,
            ),
            mixing_analysis::compute_solubility_cell_list(
                &ss.atom_posits[..solute_end],
                atoms_per_solute,
                &solubility_atom_indices,
                &ss.water_o_posits,
                &final_cell,
            ),
            final_cell,
        )
    };

    add_solubility_diagnostics(
        &mut result.data,
        solubility_diagnostics,
        solubility_estimate_barnes_hut,
        &final_cell,
    );
    let solubility_estimate = result.data.solubility_estimate;
    println!(
        "\n\n---\nSolubility: {solubility_estimate:.3} BH: {solubility_estimate_barnes_hut:.3}\n\
         raw: {:.3} mix: {:.3} disp: {:.3} agg: {:.3} contact pairs: {:.3} molarity: {:.3} M\n\n",
        result.data.solubility_raw_score,
        result.data.solubility_local_mixing,
        result.data.solubility_solute_dispersion,
        result.data.solubility_aggregation_factor,
        result.data.solubility_contact_pair_fraction,
        result.data.final_solute_molarity_m,
    );

    let mut playback_mols = vec![ShrinkingBoxPlaybackMol {
        mol_type: FfMolType::SmallOrganic,
        mol: mol.common,
        count: setup.solute_copy_count,
    }];

    playback_mols.extend(result.extra_non_water_mols.into_iter().map(|mol| {
        ShrinkingBoxPlaybackMol {
            mol_type: FfMolType::SmallOrganic,
            mol,
            count: 1,
        }
    }));

    let elapsed = start.elapsed().as_secs();
    println!("\nShrinking box sim complete in {elapsed} s\n");

    Ok((result.data, result.snapshots, playback_mols))
}

/// For running multiple molecules in sequence.
pub mod runner {
    use super::*;

    fn file_stem_has_id(path: &Path, id: usize) -> bool {
        let id = id.to_string();
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| {
                stem.split(|c: char| !c.is_ascii_digit())
                    .any(|part| part == id)
            })
    }

    fn find_sdf_by_id(path: &Path, id: usize) -> io::Result<PathBuf> {
        let mut candidates = fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("sdf"))
                    && file_stem_has_id(path, id)
            })
            .collect::<Vec<_>>();

        candidates.sort();
        candidates.into_iter().next().ok_or_else(|| {
            io::Error::new(
                ErrorKind::NotFound,
                format!("No SDF file containing molecule id {id} found under {path:?}."),
            )
        })
    }

    fn load_sdf_molecule(path: &Path) -> io::Result<MoleculeSmall> {
        let mut mol: MoleculeSmall = Sdf::load(path)?.try_into()?;
        mol.common.update_path(path);
        Ok(mol)
    }

    fn parse_csv_line(line: &str) -> Vec<String> {
        let mut fields = Vec::new();
        let mut field = String::new();
        let mut chars = line.chars().peekable();
        let mut in_quotes = false;

        while let Some(c) = chars.next() {
            match c {
                '"' if in_quotes && chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => fields.push(std::mem::take(&mut field)),
                _ => field.push(c),
            }
        }

        fields.push(field);
        fields
    }

    fn load_nominal_solubilities(csv_path: &Path) -> io::Result<HashMap<usize, f32>> {
        let csv = fs::read_to_string(csv_path)?;
        let mut lines = csv.lines();
        let header = lines.next().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("Solubility CSV {csv_path:?} is empty."),
            )
        })?;
        let header_fields = parse_csv_line(header);
        let target_col = header_fields
            .iter()
            .position(|field| field.trim() == "Y")
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    format!("Solubility CSV {csv_path:?} does not contain a Y column."),
                )
            })?;

        let mut result = HashMap::new();
        for (row_i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let fields = parse_csv_line(line);
            let Some(target_str) = fields.get(target_col) else {
                continue;
            };
            let Ok(target) = target_str.trim().parse() else {
                continue;
            };

            result.insert(row_i, target);
        }

        Ok(result)
    }

    /// A hard-coded test; no output, but prints the results.
    pub fn run_on_select_mols(dev: &ComputationDevice, param_set: &FfParamSet) {
        let mols = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 206, 226, 603, 30, 169, 217, 590, 54, 508,
            16, 569, 18, 579,
        ];

        let csv_path =
            PathBuf::from("C:/Users/the_a/Desktop/bio_misc/tdc_data/solubility_aqsoldb.csv");
        let path = PathBuf::from("C:/Users/the_a/Desktop/bio_misc/tdc_data/solubility_aqsoldb");
        let nominal_solubilities = match load_nominal_solubilities(&csv_path) {
            Ok(values) => values,
            Err(e) => {
                eprintln!("Unable to read nominal solubilities from {csv_path:?}: {e}");
                return;
            }
        };

        let mut results = Vec::new();

        for mol_id in mols {
            println!("Running on mol {mol_id}");

            let Some(nominal_solubility) = nominal_solubilities.get(&mol_id).copied() else {
                eprintln!(
                    "Skipping molecule id {mol_id}; no nominal solubility found in {csv_path:?}."
                );
                continue;
            };

            let mol_path = match find_sdf_by_id(&path, mol_id) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Skipping molecule id {mol_id}: {e}");
                    continue;
                }
            };
            let mol = match load_sdf_molecule(&mol_path) {
                Ok(mol) => mol,
                Err(e) => {
                    eprintln!("Skipping molecule id {mol_id}; unable to load {mol_path:?}: {e}");
                    continue;
                }
            };

            let data = match run_shrinking_box_sim(
                &mol,
                ShrinkingBoxMode::HomogeneousMix,
                MdBackend::Gromacs,
                dev,
                param_set,
            ) {
                Ok((data, _, _)) => data,
                Err(e) => {
                    eprintln!(
                        "Skipping molecule id {mol_id}; shrinking-box simulation failed: {e}"
                    );
                    continue;
                }
            };

            println!("-- Sol for {mol_id}: {:.4}", data.solubility_estimate);

            results.push((mol_id, mol.common.ident.clone(), nominal_solubility, data));
        }

        println!("\n------\nShrinking box sim results:\n");
        for (mol_id, ident, nominal_solubility, data) in &results {
            println!(
                "---Mol #{mol_id} {ident} | Nominal sol: {nominal_solubility:.4} | Comp sol: {:.4} BH: {:.4} | raw {:.4}, mix {:.4}, disp {:.4}, agg {:.4}, contacts {:.4}, {:.3} M\n------\n",
                data.solubility_estimate,
                data.solubility_estimate_barnes_hut,
                data.solubility_raw_score,
                data.solubility_local_mixing,
                data.solubility_solute_dispersion,
                data.solubility_aggregation_factor,
                data.solubility_contact_pair_fraction,
                data.final_solute_molarity_m,
            );
        }
        println!("\n------\n");
    }
}
