//! Preset two-layer solute/water simulations.
//!
//! The initial system is a rectangular slab of solute copies touching a rectangular
//! slab of OPC water. The goal is a cheap boundary-layer experiment rather than a
//! production free-energy protocol.

use std::{
    collections::HashMap,
    fmt::Write as _,
    fs,
    io::{self, ErrorKind, Write as _},
    path::Path,
};

use bio_files::{
    BondGeneric,
    gromacs::{
        self as bf_gromacs, GromacsInput, GromacsOutput, MoleculeInput, OutputControl,
        OutputEnergy, output::parse_gro_traj,
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MdState, MolDynamics,
    ParamError, SimBoxInit, Solvent, TAU_TEMP_DEFAULT,
    params::FfParamSet,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3 as Vec3F64},
};
use rand::Rng;

use crate::{
    md::{MdBackend, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
};

const TARGET_TOTAL_MOLECULES: usize = 200;
const TARGET_SOLUTE_COPIES: usize = 40;
const MIN_SOLUTE_COPIES: usize = 12;
const MAX_SOLUTE_COPIES: usize = 64;
const MAX_SOLUTE_ATOMS: usize = 1_800;
const MIN_WATER_MOLECULES: usize = 96;

const MIN_LAYER_SIDE_A: f32 = 42.0;
const MIN_WATER_DEPTH_A: f32 = 14.0;
const LAYER_MARGIN_A: f32 = 2.0;
const INTERFACE_GAP_A: f32 = 2.2;
const WATER_O_SPACING_A: f32 = 3.05;
const WATER_WALL_MARGIN_A: f32 = 1.7;
const WATER_MOLS_PER_A3: f32 = 0.030;

const NUM_STEPS: usize = 5_000;
const SNAPSHOT_INTERVAL: usize = 10;
const LAYER_INIT_RELAXATION_ITERS: usize = 120;
const TEMPERATURE: f32 = 300.0;
const DT: f32 = 0.002;
const AMU_A3_TO_G_CM3: f32 = 1.660_539;

const GROMACS_LAYER_DIR: &str = "gromacs_layer_out";
const GROMACS_CONF: &str = "conf.gro";
const GROMACS_TOP: &str = "topo.top";
const GROMACS_MDP: &str = "md.mdp";
const GROMACS_EM_MDP: &str = "em.mdp";
const GROMACS_TRAJ_GRO: &str = "traj.gro";
const GROMACS_TRR: &str = "traj.trr";
const GROMACS_XTC: &str = "traj.xtc";
const GROMACS_EDR: &str = "energy.edr";
const GROMACS_LOG: &str = "md.log";

const OPC_OH_A: f64 = 0.872_433_13;
const OPC_HOH_RAD: f64 = 1.808_161_105_066;
const OPC_VS_A: f64 = 0.147_803;

#[derive(Clone, Debug)]
pub struct BoundaryLayerMdData {
    pub solute_copy_count: usize,
    pub water_molecule_count: usize,
    pub requested_water_molecule_count: usize,
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
    water_molecule_count: usize,
    solute_grid: (usize, usize, usize),
    solute_spacing_a: f64,
    solute_z_spacing_a: f64,
    solute_radius_a: f64,
    box_extent_a: Vec3F32,
    solute_layer_depth_a: f32,
    water_layer_depth_a: f32,
    water_slab_low_z_a: f32,
    water_slab_high_z_a: f32,
}

#[derive(Clone, Copy, Debug)]
struct WaterGeometry {
    o: Vec3F64,
    h0: Vec3F64,
    h1: Vec3F64,
    m: Vec3F64,
}

fn param_err(e: ParamError) -> io::Error {
    io::Error::other(e.descrip)
}

fn mean(values: &[f32]) -> Option<f32> {
    let finite: Vec<_> = values.iter().copied().filter(|v| v.is_finite()).collect();
    (!finite.is_empty()).then(|| finite.iter().sum::<f32>() / finite.len() as f32)
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
    mol.common.selected_for_md = Some(1);
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

fn mol_bounding_radius(mol: &MoleculeSmall) -> f64 {
    if mol.common.atom_posits.is_empty() {
        return 2.0;
    }

    let center = mol.common.centroid();
    mol.common
        .atom_posits
        .iter()
        .map(|p| (*p - center).magnitude())
        .fold(2.0, f64::max)
}

fn boundary_layer_setup(mol: &MoleculeSmall) -> BoundaryLayerSetup {
    let atom_count = mol.common.atoms.len().max(1);
    let atom_limited_cap = (MAX_SOLUTE_ATOMS / atom_count).max(1);
    let copy_cap = MAX_SOLUTE_COPIES.min(atom_limited_cap).max(1);
    let solute_copy_count = TARGET_SOLUTE_COPIES
        .min(copy_cap)
        .max(MIN_SOLUTE_COPIES.min(copy_cap));

    let water_molecule_count = TARGET_TOTAL_MOLECULES
        .saturating_sub(solute_copy_count)
        .max(MIN_WATER_MOLECULES);

    let solute_radius_a = mol_bounding_radius(mol);
    let solute_layers_z = if solute_copy_count >= 24 { 2 } else { 1 };
    let solutes_per_z = solute_copy_count.div_ceil(solute_layers_z);
    let solute_nx = (solutes_per_z as f64).sqrt().ceil() as usize;
    let solute_ny = solutes_per_z.div_ceil(solute_nx).max(1);

    let solute_spacing_a = (2.0 * solute_radius_a + 2.8).max(5.0);
    let solute_z_spacing_a = (2.0 * solute_radius_a + 2.0).max(4.5);

    let solute_grid_width_x = (solute_nx.saturating_sub(1) as f64 * solute_spacing_a) as f32;
    let solute_grid_width_y = (solute_ny.saturating_sub(1) as f64 * solute_spacing_a) as f32;
    let fit_margin = (solute_radius_a as f32 + LAYER_MARGIN_A) * 2.0;

    let box_x = (solute_grid_width_x + fit_margin).max(MIN_LAYER_SIDE_A);
    let box_y = (solute_grid_width_y + fit_margin).max(MIN_LAYER_SIDE_A);

    let solute_layer_depth = ((solute_layers_z.saturating_sub(1) as f64 * solute_z_spacing_a)
        + 2.0 * solute_radius_a
        + 1.0) as f32;
    let interface_area = box_x * box_y;
    let density_depth = water_molecule_count as f32 / (WATER_MOLS_PER_A3 * interface_area);
    let water_layer_depth = density_depth.max(MIN_WATER_DEPTH_A);
    let box_z =
        LAYER_MARGIN_A + solute_layer_depth + INTERFACE_GAP_A + water_layer_depth + LAYER_MARGIN_A;

    let water_slab_low_z = -box_z / 2.0 + LAYER_MARGIN_A + solute_layer_depth + INTERFACE_GAP_A;
    let water_slab_high_z = box_z / 2.0 - LAYER_MARGIN_A;

    BoundaryLayerSetup {
        solute_copy_count,
        water_molecule_count,
        solute_grid: (solute_nx, solute_ny, solute_layers_z),
        solute_spacing_a,
        solute_z_spacing_a,
        solute_radius_a,
        box_extent_a: Vec3F32::new(box_x, box_y, box_z),
        solute_layer_depth_a: solute_layer_depth,
        water_layer_depth_a: water_layer_depth,
        water_slab_low_z_a: water_slab_low_z,
        water_slab_high_z_a: water_slab_high_z,
    }
}

fn make_md_cfg(setup: BoundaryLayerSetup, solvent: Solvent, memory_snapshots: bool) -> MdConfig {
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
        sim_box: SimBoxInit::Fixed((
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
        )),
        solvent,
        max_init_relaxation_iters: Some(LAYER_INIT_RELAXATION_ITERS),
        recenter_sim_box: false,
        overrides: MdOverrides::default(),
        ..Default::default()
    }
}

fn random_quaternion(rng: &mut impl Rng) -> Quaternion {
    let (w, x, y, z): (f64, f64, f64, f64) =
        (rng.random(), rng.random(), rng.random(), rng.random());
    Quaternion::new(w, x, y, z).to_normalized()
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

fn place_solute_layer(template: &MolDynamics, setup: BoundaryLayerSetup) -> Vec<MolDynamics> {
    let mut rng = rand::rng();
    let template_posits = template
        .atom_posits
        .clone()
        .unwrap_or_else(|| template.atoms.iter().map(|a| a.posit).collect());

    let centroid = template_posits
        .iter()
        .fold(Vec3F64::new(0.0, 0.0, 0.0), |acc, p| acc + *p)
        / template_posits.len().max(1) as f64;
    let locals: Vec<_> = template_posits.iter().map(|p| *p - centroid).collect();

    let (nx, ny, nz) = setup.solute_grid;
    let grid_width_x = nx.saturating_sub(1) as f64 * setup.solute_spacing_a;
    let grid_width_y = ny.saturating_sub(1) as f64 * setup.solute_spacing_a;
    let solute_low_z = -(setup.box_extent_a.z as f64) / 2.0 + LAYER_MARGIN_A as f64;
    let z0 = solute_low_z + setup.solute_radius_a + 0.5;

    let mut placed = Vec::with_capacity(setup.solute_copy_count);

    'copies: for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if placed.len() == setup.solute_copy_count {
                    break 'copies;
                }

                let center = Vec3F64::new(
                    -grid_width_x / 2.0 + ix as f64 * setup.solute_spacing_a,
                    -grid_width_y / 2.0 + iy as f64 * setup.solute_spacing_a,
                    z0 + iz as f64 * setup.solute_z_spacing_a,
                );
                let rot = random_quaternion(&mut rng);
                let posits: Vec<_> = locals
                    .iter()
                    .map(|local| rot.rotate_vec(*local) + center)
                    .collect();

                let mut mol = template.clone();
                for (atom, posit) in mol.atoms.iter_mut().zip(posits.iter()) {
                    atom.posit = *posit;
                }
                mol.atom_posits = Some(posits);
                placed.push(mol);
            }
        }
    }

    placed
}

fn make_water_geometry(o: Vec3F64, rng: &mut impl Rng) -> WaterGeometry {
    let rot = random_quaternion(rng);
    let z_local = rot.rotate_vec(Vec3F64::new(0.0, 0.0, 1.0));
    let x_local = rot.rotate_vec(Vec3F64::new(1.0, 0.0, 0.0));
    let half_angle = OPC_HOH_RAD / 2.0;

    let h0_dir = (z_local * half_angle.cos() + x_local * half_angle.sin()).to_normalized();
    let h1_dir = (z_local * half_angle.cos() - x_local * half_angle.sin()).to_normalized();

    let h0 = o + h0_dir * OPC_OH_A;
    let h1 = o + h1_dir * OPC_OH_A;
    let m = o + (h0 - o) * OPC_VS_A + (h1 - o) * OPC_VS_A;

    WaterGeometry { o, h0, h1, m }
}

fn make_water_layer(setup: BoundaryLayerSetup) -> Vec<WaterGeometry> {
    let mut rng = rand::rng();
    let usable_x = (setup.box_extent_a.x - 2.0 * WATER_WALL_MARGIN_A).max(WATER_O_SPACING_A);
    let usable_y = (setup.box_extent_a.y - 2.0 * WATER_WALL_MARGIN_A).max(WATER_O_SPACING_A);
    let usable_z =
        (setup.water_slab_high_z_a - setup.water_slab_low_z_a - 2.0 * WATER_WALL_MARGIN_A)
            .max(WATER_O_SPACING_A);

    let nx = ((usable_x / WATER_O_SPACING_A).floor() as usize).max(1);
    let ny = ((usable_y / WATER_O_SPACING_A).floor() as usize).max(1);
    let nz = setup.water_molecule_count.div_ceil(nx * ny).max(1);

    let dx = if nx > 1 {
        usable_x / (nx - 1) as f32
    } else {
        0.0
    };
    let dy = if ny > 1 {
        usable_y / (ny - 1) as f32
    } else {
        0.0
    };
    let dz = if nz > 1 {
        usable_z / (nz - 1) as f32
    } else {
        0.0
    };

    let x0 = -setup.box_extent_a.x / 2.0 + WATER_WALL_MARGIN_A;
    let y0 = -setup.box_extent_a.y / 2.0 + WATER_WALL_MARGIN_A;
    let z0 = setup.water_slab_low_z_a + WATER_WALL_MARGIN_A;

    let mut waters = Vec::with_capacity(setup.water_molecule_count);

    'grid: for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if waters.len() == setup.water_molecule_count {
                    break 'grid;
                }

                let jitter = Vec3F64::new(
                    rng.random_range(-0.12..0.12),
                    rng.random_range(-0.12..0.12),
                    rng.random_range(-0.08..0.08),
                );
                let o = Vec3F64::new(
                    (x0 + ix as f32 * dx) as f64,
                    (y0 + iy as f32 * dy) as f64,
                    (z0 + iz as f32 * dz) as f64,
                ) + jitter;

                waters.push(make_water_geometry(o, &mut rng));
            }
        }
    }

    waters
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

fn gromacs_atom_name(atom: &bio_files::AtomGeneric, serial: usize) -> String {
    atom.force_field_type
        .clone()
        .or_else(|| atom.type_in_res.as_ref().map(|t| t.to_string()))
        .or_else(|| atom.type_in_res_general.clone())
        .unwrap_or_else(|| format!("{}{}", atom.element.to_letter(), serial))
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
        name: gromacs_mol_name(ident),
        atoms,
        bonds,
        ff_params: Some(ff_params),
        count: 1,
    })
}

fn write_gro_atom_line(
    out: &mut String,
    mol_id: usize,
    mol_name: &str,
    atom_name: &str,
    atom_serial: usize,
    posit_a: Vec3F64,
    shift_a: Vec3F64,
) {
    let p_nm = (posit_a + shift_a) / 10.0;
    let _ = writeln!(
        out,
        "{:>5}{:<5}{:>5}{:>5}{:>8.3}{:>8.3}{:>8.3}",
        mol_id % 100_000,
        &mol_name[..mol_name.len().min(5)],
        &atom_name[..atom_name.len().min(5)],
        atom_serial % 100_000,
        p_nm.x,
        p_nm.y,
        p_nm.z,
    );
}

fn make_layer_gro(
    solute: &MoleculeInput,
    waters: &[WaterGeometry],
    setup: BoundaryLayerSetup,
) -> String {
    let total_atoms = solute.atoms.len() + waters.len() * 4;
    let shift = Vec3F64::new(
        setup.box_extent_a.x as f64 / 2.0,
        setup.box_extent_a.y as f64 / 2.0,
        setup.box_extent_a.z as f64 / 2.0,
    );

    let mut out = String::from("Molchanica boundary-layer solute/water preset\n");
    let _ = writeln!(out, "{total_atoms}");

    let mut atom_serial = 1usize;
    for atom in &solute.atoms {
        let atom_name = gromacs_atom_name(atom, atom_serial);
        write_gro_atom_line(
            &mut out,
            1,
            &solute.name,
            &atom_name,
            atom_serial,
            atom.posit,
            shift,
        );
        atom_serial += 1;
    }

    for (water_i, water) in waters.iter().enumerate() {
        let mol_id = water_i + 2;
        for (name, posit) in [
            ("OW", water.o),
            ("HW1", water.h0),
            ("HW2", water.h1),
            ("MW", water.m),
        ] {
            write_gro_atom_line(&mut out, mol_id, "SOL", name, atom_serial, posit, shift);
            atom_serial += 1;
        }
    }

    let _ = writeln!(
        out,
        "{:>10.5}{:>10.5}{:>10.5}",
        setup.box_extent_a.x / 10.0,
        setup.box_extent_a.y / 10.0,
        setup.box_extent_a.z / 10.0,
    );

    out
}

fn boundary_layer_data_from_setup(setup: BoundaryLayerSetup) -> BoundaryLayerMdData {
    BoundaryLayerMdData {
        solute_copy_count: setup.solute_copy_count,
        water_molecule_count: setup.water_molecule_count,
        requested_water_molecule_count: setup.water_molecule_count,
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
    if let Some(last) = snapshots.last() {
        data.water_molecule_count = last.water_o_posits.len();
    }

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

fn run_dynamics_backend(
    placed_mols: &[MolDynamics],
    param_set: &FfParamSet,
    setup: BoundaryLayerSetup,
    dev: &ComputationDevice,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let cfg = make_md_cfg(
        setup,
        Solvent::WaterOpcSpecifyMolCount(setup.water_molecule_count),
        true,
    );

    let (mut md, _) = MdState::new(dev, &cfg, placed_mols, param_set).map_err(param_err)?;
    run_dynamics_blocking(&mut md, dev, DT, NUM_STEPS);

    if md.snapshots.is_empty() {
        return Err(io::Error::other(
            "Boundary-layer Dynamics run completed without recording snapshots.",
        ));
    }

    let mut data = boundary_layer_data_from_setup(setup);
    add_snapshot_metrics(&mut data, &md.snapshots);

    Ok((data, md.snapshots))
}

fn em_mdp_str() -> &'static str {
    "; Energy minimization - generated by Molchanica\n\
     integrator               = steep\n\
     nsteps                   = 5000\n\
     emtol                    = 1000.0\n\
     emstep                   = 0.01\n\
     \n\
     cutoff-scheme            = Verlet\n\
     coulombtype              = PME\n\
     fourierspacing           = 0.16\n\
     rcoulomb                 = 1.0\n\
     vdw-type                 = Cut-off\n\
     rvdw                     = 1.0\n\
     \n\
     pbc                      = xyz\n\
     constraints              = none\n"
}

fn save_text(path: impl AsRef<Path>, text: &str) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    write!(f, "{text}")?;
    Ok(())
}

fn make_layer_top(
    solute_input: MoleculeInput,
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    setup: BoundaryLayerSetup,
) -> io::Result<String> {
    let cfg = make_md_cfg(setup, Solvent::None, false);
    let mdp = cfg.to_gromacs(NUM_STEPS, DT);
    let mols = vec![(
        FfMolType::SmallOrganic,
        &mol.common,
        setup.solute_copy_count,
    )];
    let mut input = crate::gromacs::make_gromacs_input(
        mdp,
        &mols,
        vec![solute_input],
        param_set,
        &cfg.sim_box,
        &Solvent::None,
        true,
    )?;
    input.solvent = Some(bf_gromacs::solvate::Solvent::Custom(
        bf_gromacs::solvate::CustomSolventTemplate {
            gro_text: String::new(),
            topology_molecules: Vec::new(),
            include_opc_water: true,
        },
    ));

    let mut top = input.make_top()?;
    top.push_str(&format!("{:<14}  {}\n", "SOL", setup.water_molecule_count));
    Ok(top)
}

fn run_gromacs_backend(
    placed_mols: &[MolDynamics],
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    setup: BoundaryLayerSetup,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let solute_input = gromacs_layer_molecule_input(placed_mols, &mol.common.ident)?;
    let waters = make_water_layer(setup);
    let gro_text = make_layer_gro(&solute_input, &waters, setup);
    let top = make_layer_top(solute_input.clone(), mol, param_set, setup)?;

    let cfg = make_md_cfg(setup, Solvent::None, false);
    let input_for_run = GromacsInput {
        mdp: cfg.to_gromacs(NUM_STEPS, DT),
        molecules: vec![solute_input],
        box_nm: Some((
            setup.box_extent_a.x as f64 / 10.0,
            setup.box_extent_a.y as f64 / 10.0,
            setup.box_extent_a.z as f64 / 10.0,
        )),
        ff_global: None,
        solvent: None,
        minimize_energy: true,
    };

    let dir = Path::new(GROMACS_LAYER_DIR);
    fs::create_dir_all(dir)?;
    save_text(dir.join(GROMACS_TOP), &top)?;

    let out = {
        save_text(dir.join(GROMACS_CONF), &gro_text)?;
        save_text(dir.join(GROMACS_MDP), &input_for_run.make_mdp())?;
        save_text(dir.join(GROMACS_EM_MDP), em_mdp_str())?;

        bf_gromacs::run_gmx(
            dir,
            &[
                "grompp",
                "-f",
                GROMACS_EM_MDP,
                "-c",
                GROMACS_CONF,
                "-p",
                GROMACS_TOP,
                "-o",
                "em.tpr",
                "-maxwarn",
                "5",
            ],
        )?;
        bf_gromacs::run_gmx(
            dir,
            &[
                "mdrun", "-s", "em.tpr", "-c", "em.gro", "-e", "em.edr", "-g", "em.log",
            ],
        )?;
        bf_gromacs::run_gmx(
            dir,
            &[
                "grompp",
                "-f",
                GROMACS_MDP,
                "-c",
                "em.gro",
                "-p",
                GROMACS_TOP,
                "-o",
                "topol.tpr",
                "-maxwarn",
                "5",
            ],
        )?;
        bf_gromacs::run_gmx(
            dir,
            &[
                "mdrun",
                "-s",
                "topol.tpr",
                "-o",
                GROMACS_TRR,
                "-x",
                GROMACS_XTC,
                "-c",
                "confout.gro",
                "-e",
                GROMACS_EDR,
                "-g",
                GROMACS_LOG,
            ],
        )?;
        bf_gromacs::run_gmx_stdin(
            dir,
            &[
                "trjconv",
                "-f",
                GROMACS_TRR,
                "-s",
                "topol.tpr",
                "-o",
                GROMACS_TRAJ_GRO,
            ],
            b"0\n",
        )?;

        let log_text = fs::read_to_string(dir.join(GROMACS_LOG)).unwrap_or_default();
        let traj_text = fs::read_to_string(dir.join(GROMACS_TRAJ_GRO))?;
        let frames = parse_gro_traj(&traj_text)?;
        let energies = OutputEnergy::from_edr(&dir.join(GROMACS_EDR)).unwrap_or_default();
        GromacsOutput::new(
            log_text,
            frames,
            energies,
            input_for_run.solute_atom_count(),
        )?
    };

    if out.setup_failure {
        return Err(io::Error::other(
            "GROMACS setup failed while running boundary-layer MD.",
        ));
    }
    if out.log_text.contains("Fatal error") {
        return Err(io::Error::other(
            "GROMACS reported a fatal error while running boundary-layer MD.",
        ));
    }

    let snapshots = gromacs_frames_to_ss(&out);
    if snapshots.is_empty() {
        return Err(io::Error::other(
            "GROMACS boundary-layer MD completed without recording snapshots.",
        ));
    }

    let mut data = boundary_layer_data_from_setup(setup);
    add_snapshot_metrics(&mut data, &snapshots);

    Ok((data, snapshots))
}

/// A simulation of two touching layers, with no probe molecule: one layer is the
/// molecule being measured, and the other is OPC water.
pub fn boundary_layer_solute_water(
    mol: &MoleculeSmall,
    backend: MdBackend,
    dev: &ComputationDevice,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let param_set = FfParamSet::new_amber()?;
    let (mol, mol_specific_params) = prepare_mol_for_md(mol, &param_set)?;
    let template = solute_template(&mol, &mol_specific_params)?;
    let setup = boundary_layer_setup(&mol);
    let placed_mols = place_solute_layer(&template, setup);

    println!(
        "Boundary-layer MD setup ({backend}): {} solute copies, {} waters, box {:.1} x {:.1} x {:.1} A",
        setup.solute_copy_count,
        setup.water_molecule_count,
        setup.box_extent_a.x,
        setup.box_extent_a.y,
        setup.box_extent_a.z,
    );

    match backend {
        MdBackend::Dynamics => run_dynamics_backend(&placed_mols, &param_set, setup, dev),
        MdBackend::Gromacs => run_gromacs_backend(&placed_mols, &mol, &param_set, setup),
        MdBackend::Orca => Err(io::Error::new(
            ErrorKind::Unsupported,
            "Boundary-layer water/solute MD supports the Dynamics and GROMACS backends.",
        )),
    }
}
