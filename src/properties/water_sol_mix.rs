//! A 2-layer solute/water simulation, used to assess solubility in water. Attempts to model
//! both self-affinity of the target molecule, and its affinity to water.
//!
//! The initial system is a rectangular slab of solute copies touching a rectangular
//! slab of OPC water. The goal is a cheap boundary-layer experiment rather than a
//! production free-energy protocol.

use std::{
    collections::HashMap,
    io::{self, ErrorKind},
    path::Path,
};

use bio_files::{
    BondGeneric,
    gromacs::{MoleculeInput, OutputControl},
    md_params::ForceFieldParams,
};

use dynamics::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MdState, MolDynamics,
    ShrinkingBoxPackingCfg, SimBox, SimBoxInit, Solvent, TAU_TEMP_DEFAULT,
    pack_solvent_with_shrinking_box_cfg,
    params::FfParamSet,
    random_quaternion,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3 as Vec3F64},
};

use crate::gromacs::make_gromacs_input;
use crate::properties::{mean, mol_bounding_radius};
use crate::{
    md::{MdBackend, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
    properties::prepare_mol_for_md,
};

const TARGET_SOLUTE_COPIES: usize = 40;
const MIN_SOLUTE_COPIES: usize = 12;
const MAX_SOLUTE_COPIES: usize = 64;
const MAX_SOLUTE_ATOMS: usize = 1_800;

// Footprint side of the solute slab; sets the solute/water contact area.
const MIN_LAYER_SIDE_A: f32 = 44.0;
const MIN_SOLUTE_LAYER_DEPTH_A: f32 = 12.0;
// Random-orientation grid placement can't reach crystal-like fractions; 0.45
// is a realistic upper bound and gives the relaxation step room to breathe.
const SOLUTE_PACKING_FRACTION: f32 = 0.45;
const SOLUTE_LAYER_WALL_MARGIN_A: f32 = 1.2;
const SOLUTE_PACKING_INITIAL_BOX_SCALE: f32 = 1.8;
const SOLUTE_PACKING_SHRINK_PER_STEP_A: f32 = 0.05;
const SOLUTE_PACKING_EQUILIBRATION_STEPS: usize = 750;
// Water thickness for the explicit slab against the solute layer.
const WATER_SLAB_DEPTH_A: f32 = 24.0;
const LAYER_MARGIN_A: f32 = 2.0;
const INTERFACE_GAP_A: f32 = 2.2;

const NUM_STEPS: usize = 20_000;

const SNAPSHOT_INTERVAL: usize = 10;
const LAYER_INIT_RELAXATION_ITERS: usize = 120;
const TEMPERATURE: f32 = 300.0;
const DT: f32 = 0.002;
const AMU_A3_TO_G_CM3: f32 = 1.660_539;

#[derive(Clone, Debug)]
pub struct BoundaryLayerMdData {
    pub solute_copy_count: usize,
    pub box_extent_a: Vec3F32,
    pub interface_area_a2: f32,
    pub solute_layer_depth_a: f32,
    pub water_layer_depth_a: f32,
    pub mean_temperature_k: f32,
    pub mean_pressure_bar: f32,
    pub density_g_cm3: f32,
    pub potential_energy_kcal: f32,
    pub nonbonded_energy_kcal: f32,
}

#[derive(Clone, Copy, Debug)]
struct BoundaryLayerSetup {
    solute_copy_count: usize,
    box_extent_a: Vec3F32,
    solute_layer_depth_a: f32,
    water_layer_depth_a: f32,
    water_slab_low_z_a: f32,
    water_slab_high_z_a: f32,
}

/// Volume is in Angstrom^3; depth is in Angstrom
fn bounded_solute_copy_count(mol: &MoleculeSmall, mol_volume: f32, min_depth: f32) -> usize {
    let atom_count = mol.common.atoms.len();

    let atom_limited_cap = MAX_SOLUTE_ATOMS / atom_count;
    let copy_cap = MAX_SOLUTE_COPIES.min(atom_limited_cap);
    let min_copies = MIN_SOLUTE_COPIES.min(copy_cap);

    let min_layer_capacity = MIN_LAYER_SIDE_A.powi(2) * min_depth * SOLUTE_PACKING_FRACTION;

    let fill_min_layer_count = (min_layer_capacity / mol_volume)
        .ceil()
        .max(MIN_SOLUTE_COPIES as f32) as usize;

    let requested = TARGET_SOLUTE_COPIES.max(fill_min_layer_count.min(MAX_SOLUTE_COPIES));

    requested.min(copy_cap).max(min_copies)
}

fn boundary_layer_setup(mol: &MoleculeSmall) -> BoundaryLayerSetup {
    let solute_radius_a = mol_bounding_radius(mol);
    let mol_volume = mol.characterization.as_ref().unwrap().volume; // todo: Handle

    // Buffer around the placement region: wall margin plus the molecule's bounding
    // radius. Centroids placed inside the inset volume have all atoms strictly
    // inside the wall-margin envelope, regardless of rotation.
    let inset_a = SOLUTE_LAYER_WALL_MARGIN_A + solute_radius_a;
    let footprint_side =
        MIN_LAYER_SIDE_A.max(2.0 * inset_a + 2.0 * solute_radius_a + 2.0 * LAYER_MARGIN_A);
    let fillable_side = (footprint_side - 2.0 * inset_a).max(2.0 * solute_radius_a);

    let solute_copy_count = bounded_solute_copy_count(mol, mol_volume, MIN_SOLUTE_LAYER_DEPTH_A);

    // Fillable depth: enough to hold N molecules at the packing fraction inside
    // a `fillable_side × fillable_side` cross-section, but never thinner than one
    // molecule diameter.
    let fillable_min_depth = (2.0 * solute_radius_a).max(1.0);
    let target_fillable_vol = solute_copy_count as f32 * mol_volume / SOLUTE_PACKING_FRACTION;
    let fillable_depth =
        (target_fillable_vol / (fillable_side * fillable_side)).max(fillable_min_depth);
    let solute_layer_depth = (fillable_depth + 2.0 * inset_a).max(MIN_SOLUTE_LAYER_DEPTH_A);

    // Two explicit slabs in one fixed cell: solute below, OPC water above.
    let box_z =
        LAYER_MARGIN_A + solute_layer_depth + INTERFACE_GAP_A + WATER_SLAB_DEPTH_A + LAYER_MARGIN_A;
    let water_slab_low_z = -box_z / 2.0 + LAYER_MARGIN_A + solute_layer_depth + INTERFACE_GAP_A;
    let water_slab_high_z = water_slab_low_z + WATER_SLAB_DEPTH_A;

    BoundaryLayerSetup {
        solute_copy_count,
        box_extent_a: Vec3F32::new(footprint_side, footprint_side, box_z),
        solute_layer_depth_a: solute_layer_depth,
        water_layer_depth_a: WATER_SLAB_DEPTH_A,
        water_slab_low_z_a: water_slab_low_z,
        water_slab_high_z_a: water_slab_high_z,
    }
}

fn boundary_layer_cell(setup: BoundaryLayerSetup) -> SimBox {
    SimBox::new(
        Vec3F32::new(
            -setup.box_extent_a.x / 2.0,
            -setup.box_extent_a.y / 2.0,
            -setup.box_extent_a.z / 2.0,
        ),
        Vec3F32::new(
            setup.box_extent_a.x / 2.0,
            setup.box_extent_a.y / 2.0,
            setup.box_extent_a.z / 2.0,
        ),
    )
}

fn water_layer_cell(setup: BoundaryLayerSetup) -> SimBox {
    SimBox::new(
        Vec3F32::new(
            -setup.box_extent_a.x / 2.0,
            -setup.box_extent_a.y / 2.0,
            setup.water_slab_low_z_a,
        ),
        Vec3F32::new(
            setup.box_extent_a.x / 2.0,
            setup.box_extent_a.y / 2.0,
            setup.water_slab_high_z_a,
        ),
    )
}

fn make_md_cfg(setup: BoundaryLayerSetup, solvent: Solvent, memory_snapshots: bool) -> MdConfig {
    let cell = boundary_layer_cell(setup);

    MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        zero_com_drift: true,
        temp_target: TEMPERATURE,
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
                    nstfout: None,
                    nstcalcenergy: Some(SNAPSHOT_INTERVAL as u32),
                    nstenergy: Some(SNAPSHOT_INTERVAL as u32),
                    ..Default::default()
                }
            },
        },
        sim_box: SimBoxInit::Fixed((cell.bounds_low, cell.bounds_high)),
        solvent,
        max_init_relaxation_iters: Some(LAYER_INIT_RELAXATION_ITERS),
        recenter_sim_box: false,
        overrides: MdOverrides::default(),
        ..Default::default()
    }
}

fn solute_template(
    mol: &MoleculeSmall,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
) -> io::Result<MolDynamics> {
    let Some(ff_params) = mol_specific_params.get(&mol.common.ident).cloned() else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing molecule-specific parameters for boundary-layer input.",
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

fn solute_layer_center_z(setup: BoundaryLayerSetup) -> f32 {
    -setup.box_extent_a.z / 2.0 + LAYER_MARGIN_A + setup.solute_layer_depth_a / 2.0
}

fn translate_mols(mols: &mut [MolDynamics], offset: Vec3F64) {
    for mol in mols {
        if let Some(posits) = &mut mol.atom_posits {
            for (atom, posit) in mol.atoms.iter_mut().zip(posits.iter_mut()) {
                *posit += offset;
                atom.posit = *posit;
            }
        } else {
            let posits: Vec<_> = mol
                .atoms
                .iter_mut()
                .map(|atom| {
                    atom.posit += offset;
                    atom.posit
                })
                .collect();

            mol.atom_posits = Some(posits);
        }
    }
}

fn centered_solute_packing_cell(setup: BoundaryLayerSetup) -> SimBox {
    let half_x = (setup.box_extent_a.x / 2.0 - SOLUTE_LAYER_WALL_MARGIN_A).max(1.0);
    let half_y = (setup.box_extent_a.y / 2.0 - SOLUTE_LAYER_WALL_MARGIN_A).max(1.0);
    let half_z = (setup.solute_layer_depth_a / 2.0 - SOLUTE_LAYER_WALL_MARGIN_A).max(1.0);

    SimBox::new(
        Vec3F32::new(-half_x, -half_y, -half_z),
        Vec3F32::new(half_x, half_y, half_z),
    )
}

/// Pick (nx, ny, nz) cell counts that fit `copies` molecules in a box of the
/// given dimensions while keeping cells as close to cubic as possible. The
/// shared `add_copies` helper uses a cubic n³ grid, which breaks down for
/// slab-shaped layers — this lets us spread copies primarily in xy when z is
/// shallow.
fn slab_grid_dims(copies: usize, bx: f32, by: f32, bz: f32) -> (usize, usize, usize) {
    let copies = copies.max(1);
    let (bx, by, bz) = (bx as f64, by as f64, bz as f64);
    let ideal_side = (bx * by * bz / copies as f64).cbrt().max(f64::EPSILON);

    let mut nx = ((bx / ideal_side).floor() as usize).max(1);
    let mut ny = ((by / ideal_side).floor() as usize).max(1);
    let mut nz = ((bz / ideal_side).floor() as usize).max(1);

    // Expand whichever axis yields the largest cell size post-expansion, so
    // cells stay roughly isotropic instead of collapsing one dimension.
    while nx * ny * nz < copies {
        let cand_x = bx / (nx + 1) as f64;
        let cand_y = by / (ny + 1) as f64;
        let cand_z = bz / (nz + 1) as f64;
        if cand_x >= cand_y && cand_x >= cand_z {
            nx += 1;
        } else if cand_y >= cand_z {
            ny += 1;
        } else {
            nz += 1;
        }
    }
    (nx, ny, nz)
}

fn fallback_solute_layer(
    template: &MolDynamics,
    setup: BoundaryLayerSetup,
) -> io::Result<Vec<MolDynamics>> {
    let template_posits: Vec<Vec3F64> = template
        .atom_posits
        .clone()
        .unwrap_or_else(|| template.atoms.iter().map(|a| a.posit).collect());
    let atom_count = template_posits.len().max(1);
    let centroid = template_posits
        .iter()
        .fold(Vec3F64::new(0.0, 0.0, 0.0), |s, &p| s + p)
        * (1.0 / atom_count as f64);
    let local: Vec<Vec3F64> = template_posits.iter().map(|&p| p - centroid).collect();

    // Largest centroid-to-atom distance — any rotation keeps every atom inside a
    // sphere of this radius around the chosen cell center, so insetting the
    // placement region by it guarantees no atom pokes past the wall margin.
    let local_radius = local.iter().map(|p| p.magnitude()).fold(0.0_f64, f64::max) as f32;
    let centroid_inset = SOLUTE_LAYER_WALL_MARGIN_A + local_radius;

    let usable_x = setup.box_extent_a.x - 2.0 * centroid_inset;
    let usable_y = setup.box_extent_a.y - 2.0 * centroid_inset;
    let usable_z = setup.solute_layer_depth_a - 2.0 * centroid_inset;
    if usable_x <= 0.0 || usable_y <= 0.0 || usable_z <= 0.0 {
        return Err(io::Error::other(format!(
            "Boundary-layer solute slab {:.1}x{:.1}x{:.1} A is smaller than the molecule's \
             bounding radius {:.2} A plus wall margin {:.2} A; cannot place any copy safely.",
            setup.box_extent_a.x,
            setup.box_extent_a.y,
            setup.solute_layer_depth_a,
            local_radius,
            SOLUTE_LAYER_WALL_MARGIN_A,
        )));
    }

    let (nx, ny, nz) = slab_grid_dims(setup.solute_copy_count, usable_x, usable_y, usable_z);
    let cell_count = nx * ny * nz;
    if cell_count < setup.solute_copy_count {
        return Err(io::Error::other(format!(
            "Boundary-layer solute slab {:.1}x{:.1}x{:.1} A admits only {}x{}x{}={} grid cells \
             for {} requested copies.",
            usable_x, usable_y, usable_z, nx, ny, nz, cell_count, setup.solute_copy_count,
        )));
    }

    let (sx, sy, sz) = (
        usable_x as f64 / nx as f64,
        usable_y as f64 / ny as f64,
        usable_z as f64 / nz as f64,
    );
    let (x0, y0, z0) = (
        -usable_x as f64 / 2.0,
        -usable_y as f64 / 2.0,
        -usable_z as f64 / 2.0,
    );

    let mut rng = rand::rng();
    let mut placed = Vec::with_capacity(setup.solute_copy_count);

    // When cells > copies, stride through the grid so placements are evenly
    // distributed instead of clumped in one corner.
    let stride = (cell_count / setup.solute_copy_count.max(1)).max(1);

    for i in 0..setup.solute_copy_count {
        let cell_idx = (i * stride).min(cell_count - 1);
        let ix = cell_idx % nx;
        let iy = (cell_idx / nx) % ny;
        let iz = cell_idx / (nx * ny);

        let cell_center = Vec3F64::new(
            x0 + (ix as f64 + 0.5) * sx,
            y0 + (iy as f64 + 0.5) * sy,
            z0 + (iz as f64 + 0.5) * sz,
        );

        let rot: Quaternion = random_quaternion(&mut rng, None).into();
        let posits: Vec<Vec3F64> = local
            .iter()
            .map(|&l| rot.rotate_vec(l) + cell_center)
            .collect();

        let mut mol_copy = template.clone();
        for (atom, p) in mol_copy.atoms.iter_mut().zip(posits.iter()) {
            atom.posit = *p;
        }
        mol_copy.atom_posits = Some(posits);
        placed.push(mol_copy);
    }

    Ok(placed)
}

fn pack_solute_layer(
    template: &MolDynamics,
    setup: BoundaryLayerSetup,
    param_set: &FfParamSet,
    dev: &ComputationDevice,
) -> io::Result<Vec<MolDynamics>> {
    let cfg = ShrinkingBoxPackingCfg {
        initial_box_scale: SOLUTE_PACKING_INITIAL_BOX_SCALE,
        dt: 0.001,
        box_shrink_per_step: SOLUTE_PACKING_SHRINK_PER_STEP_A,
        equilibration_steps: SOLUTE_PACKING_EQUILIBRATION_STEPS,
        snapshot_interval: None,
        gromacs_output_interval: None,
        save_gro: false,
    };

    let packed = pack_solvent_with_shrinking_box_cfg(
        dev,
        template,
        "MOL",
        setup.solute_copy_count,
        0,
        centered_solute_packing_cell(setup),
        param_set,
        Path::new("./md_out"),
        cfg,
    );

    let mut placed = match packed {
        Ok((mols, _)) if mols.len() == setup.solute_copy_count => mols,
        Ok((mols, _)) => {
            eprintln!(
                "Boundary-layer shrink packing placed {} / {} solute copies; falling back to grid packing.",
                mols.len(),
                setup.solute_copy_count
            );
            fallback_solute_layer(template, setup)?
        }
        Err(e) => {
            eprintln!(
                "Boundary-layer shrink packing failed: {}; falling back to grid packing.",
                e.descrip
            );
            fallback_solute_layer(template, setup)?
        }
    };

    translate_mols(
        &mut placed,
        Vec3F64::new(0.0, 0.0, solute_layer_center_z(setup) as f64),
    );

    Ok(placed)
}

fn gromacs_layer_mol_name(ident: &str) -> String {
    let name: String = ident
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .take(6)
        .collect();

    if name.is_empty() {
        "MOL".to_string()
    } else {
        name
    }
}

fn gromacs_layer_molecule_input(
    placed_mols: &[MolDynamics],
    ident: &str,
) -> io::Result<MoleculeInput> {
    let Some(ff_params) = placed_mols
        .iter()
        .find_map(|mol| mol.mol_specific_params.clone())
    else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "Missing molecule-specific parameters for boundary-layer GROMACS input.",
        ));
    };

    let atom_count: usize = placed_mols.iter().map(|mol| mol.atoms.len()).sum();
    let bond_count: usize = placed_mols.iter().map(|mol| mol.bonds.len()).sum();

    let mut atoms = Vec::with_capacity(atom_count);
    let mut bonds = Vec::with_capacity(bond_count);
    let mut next_serial = 1_u32;

    for mol in placed_mols {
        let mut serial_map = HashMap::with_capacity(mol.atoms.len());

        for (i, atom) in mol.atoms.iter().enumerate() {
            let mut atom = atom.clone();
            if let Some(posits) = &mol.atom_posits
                && let Some(posit) = posits.get(i)
            {
                atom.posit = *posit;
            }

            let new_serial = next_serial;
            next_serial = next_serial.checked_add(1).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    "Too many atoms for boundary-layer GROMACS serial numbers.",
                )
            })?;

            serial_map.insert(atom.serial_number, new_serial);
            atom.serial_number = new_serial;
            atoms.push(atom);
        }

        for bond in &mol.bonds {
            let (Some(&atom_0_sn), Some(&atom_1_sn)) = (
                serial_map.get(&bond.atom_0_sn),
                serial_map.get(&bond.atom_1_sn),
            ) else {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "Boundary-layer GROMACS input bond references an atom outside its copy.",
                ));
            };

            bonds.push(BondGeneric {
                bond_type: bond.bond_type,
                atom_0_sn,
                atom_1_sn,
            });
        }
    }

    Ok(MoleculeInput {
        name: gromacs_layer_mol_name(ident),
        atoms,
        bonds,
        ff_params: Some(ff_params),
        count: 1,
    })
}

fn boundary_layer_data_from_setup(setup: BoundaryLayerSetup) -> BoundaryLayerMdData {
    BoundaryLayerMdData {
        solute_copy_count: setup.solute_copy_count,
        box_extent_a: setup.box_extent_a,
        interface_area_a2: setup.box_extent_a.x * setup.box_extent_a.y,
        solute_layer_depth_a: setup.solute_layer_depth_a,
        water_layer_depth_a: setup.water_layer_depth_a,
        mean_temperature_k: 0.0,
        mean_pressure_bar: 0.0,
        density_g_cm3: 0.0,
        potential_energy_kcal: 0.0,
        nonbonded_energy_kcal: 0.0,
    }
}

fn add_snapshot_metrics(data: &mut BoundaryLayerMdData, snapshots: &[Snapshot]) {
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
        let Some(e) = &snap.energy_data else {
            continue;
        };

        temperatures.push(e.temperature);
        pressures.push(e.pressure);
        densities.push(e.density * AMU_A3_TO_G_CM3);
        potentials.push(e.energy_potential);
        nonbonded.push(e.energy_potential_nonbonded);
    }

    data.mean_temperature_k = mean(&temperatures).unwrap_or(0.0);
    data.mean_pressure_bar = mean(&pressures).unwrap_or(0.0);
    data.density_g_cm3 = mean(&densities).unwrap_or(0.0);
    data.potential_energy_kcal = mean(&potentials).unwrap_or(0.0);
    data.nonbonded_energy_kcal = mean(&nonbonded).unwrap_or(0.0);
}

/// Launch using the dynamics backend.
fn run_dynamics(
    solute_mols: &[MolDynamics],
    water_cell: SimBox,
    param_set: &FfParamSet,
    setup: BoundaryLayerSetup,
    dev: &ComputationDevice,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let cfg = make_md_cfg(
        setup,
        Solvent::WaterOpcCustomRegions(vec![water_cell]),
        true,
    );

    let (mut md, _) =
        MdState::new(dev, &cfg, solute_mols, param_set).map_err(|e| io::Error::other(e.descrip))?;

    run_dynamics_blocking(&mut md, dev, DT, NUM_STEPS);

    let mut data = boundary_layer_data_from_setup(setup);

    add_snapshot_metrics(&mut data, &md.snapshots);

    Ok((data, md.snapshots))
}

fn run_gromacs(
    solute_mols: &[MolDynamics], // todo? Not in others
    water_cell: SimBox,
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    setup: BoundaryLayerSetup,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let cfg = make_md_cfg(
        setup,
        Solvent::WaterOpcCustomRegions(vec![water_cell]),
        false,
    );
    let mdp = cfg.to_gromacs(NUM_STEPS, DT);

    let mols = vec![(FfMolType::SmallOrganic, &mol.common, 1)];

    let solute_input = gromacs_layer_molecule_input(solute_mols, &mol.common.ident)?;
    let input = make_gromacs_input(
        mdp,
        &mols,
        vec![solute_input],
        param_set,
        &cfg.sim_box,
        &cfg.solvent,
        cfg.max_init_relaxation_iters.is_some(),
    )?;

    let out = input.run()?;

    // if out.setup_failure {
    //     return Err(io::Error::other(
    //         "GROMACS setup failed while running boundary-layer MD.",
    //     ));
    // }
    //
    // if out.log_text.contains("Fatal error") {
    //     return Err(io::Error::other(
    //         "GROMACS reported a fatal error while running boundary-layer MD.",
    //     ));
    // }

    let snapshots = gromacs_frames_to_ss(&out);
    // if snapshots.is_empty() {
    //     return Err(io::Error::other(
    //         "GROMACS boundary-layer MD completed without recording snapshots.",
    //     ));
    // }

    let mut data = boundary_layer_data_from_setup(setup);
    add_snapshot_metrics(&mut data, &snapshots);

    Ok((data, snapshots))
}

/// A simulation of two touching layers, with no probe molecule: one layer is the
/// molecule being measured, and the other is OPC water.
pub fn run_boundary_layer_sol_sim(
    mol: &MoleculeSmall,
    backend: MdBackend,
    dev: &ComputationDevice,
    param_set: &FfParamSet,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let (mol, mol_specific_params) = prepare_mol_for_md(mol, &param_set)?;

    let template = solute_template(&mol, &mol_specific_params)?;
    let setup = boundary_layer_setup(&mol);

    let solute_mols = pack_solute_layer(&template, setup, &param_set, dev)?;

    // We rely on the backends' respective code to generate OPC or similar rigid water in this slab.
    let water_cell = water_layer_cell(setup);

    println!(
        "Boundary-layer MD setup ({backend}): {} solute copies, footprint {:.1} x {:.1} A, solute depth {:.1} A, water depth {:.1} A, box z {:.1} A",
        setup.solute_copy_count,
        setup.box_extent_a.x,
        setup.box_extent_a.y,
        setup.solute_layer_depth_a,
        setup.water_layer_depth_a,
        setup.box_extent_a.z,
    );

    match backend {
        MdBackend::Dynamics => run_dynamics(&solute_mols, water_cell, &param_set, setup, dev),
        MdBackend::Gromacs => run_gromacs(&solute_mols, water_cell, &mol, &param_set, setup),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Boundary-layer water/solute MD supports the Dynamics and GROMACS backends.",
        )),
    }
}
