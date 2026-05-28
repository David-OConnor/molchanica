//! Preset two-layer solute/water simulations.
//!
//! The initial system is a rectangular slab of solute copies touching a rectangular
//! slab of OPC water. The goal is a cheap boundary-layer experiment rather than a
//! production free-energy protocol.

use std::{
    collections::HashMap,
    f32::consts::PI,
    io::{self, ErrorKind},
    path::Path,
};

use bio_files::{
    BondGeneric,
    gromacs::{
        self as bf_gromacs, MoleculeInput, OutputControl,
        gro::{AtomGro, Gro},
    },
    md_params::ForceFieldParams,
};
use dynamics::{
    ComputationDevice, FfMolType, Integrator, MdConfig, MdOverrides, MdState, MolDynamics,
    ParamError, ShrinkingBoxPackingCfg, SimBox, SimBoxInit, Solvent, SolventTemplateType,
    TAU_TEMP_DEFAULT, WaterInitTemplate, pack_solvent_with_shrinking_box_cfg,
    params::FfParamSet,
    random_quaternion,
    snapshot::{Snapshot, SnapshotHandlers, gromacs_frames_to_ss},
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3 as Vec3F64},
};
use na_seq::Element;
use rand::Rng;

use crate::{
    md::{MdBackend, add_copies, run_dynamics_blocking},
    molecules::small::MoleculeSmall,
    properties::prepare_mol_for_md,
};

const TARGET_TOTAL_MOLECULES: usize = 200;
const TARGET_SOLUTE_COPIES: usize = 40;
const MIN_SOLUTE_COPIES: usize = 12;
const MAX_SOLUTE_COPIES: usize = 64;
const MAX_SOLUTE_ATOMS: usize = 1_800;
const MIN_WATER_MOLECULES: usize = 96;

const MIN_LAYER_SIDE_A: f32 = 32.0;
const MIN_SOLUTE_LAYER_DEPTH_A: f32 = 10.0;
const SOLUTE_PACKING_FRACTION: f32 = 0.62;
const SOLUTE_LAYER_WALL_MARGIN_A: f32 = 1.2;
const SOLUTE_PACKING_INITIAL_BOX_SCALE: f32 = 1.8;
const SOLUTE_PACKING_SHRINK_PER_STEP_A: f32 = 0.05;
const SOLUTE_PACKING_EQUILIBRATION_STEPS: usize = 750;
const MIN_WATER_DEPTH_A: f32 = 14.0;
const LAYER_MARGIN_A: f32 = 2.0;
const INTERFACE_GAP_A: f32 = 2.2;
const WATER_O_SPACING_A: f32 = 3.05;
const WATER_WALL_MARGIN_A: f32 = 1.7;
const WATER_MOLS_PER_A3: f32 = 0.030;

const NUM_STEPS: usize = 2_000;

const SNAPSHOT_INTERVAL: usize = 10;
const LAYER_INIT_RELAXATION_ITERS: usize = 120;
const TEMPERATURE: f32 = 300.0;
const DT: f32 = 0.002;
const AMU_A3_TO_G_CM3: f32 = 1.660_539;

// todo: Hmmmm. Not sure we want this here vs in bio_files/gromacs and dynamics.
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

fn mol_volume_estimate_a3(mol: &MoleculeSmall, radius_a: f32) -> f32 {
    mol.characterization
        .as_ref()
        .and_then(|char| {
            char.volume_pubchem
                .filter(|volume| *volume > 0.0)
                .or_else(|| (char.volume > 0.0).then_some(char.volume))
        })
        .unwrap_or_else(|| (4.0 / 3.0) * PI * radius_a.powi(3) * 0.45)
        .max(40.0)
}

fn bounded_solute_copy_count(mol: &MoleculeSmall, mol_volume_a3: f32, min_depth_a: f32) -> usize {
    let atom_count = mol.common.atoms.len().max(1);
    let atom_limited_cap = (MAX_SOLUTE_ATOMS / atom_count).max(1);
    let copy_cap = MAX_SOLUTE_COPIES.min(atom_limited_cap).max(1);
    let min_copies = MIN_SOLUTE_COPIES.min(copy_cap);

    let min_layer_capacity = MIN_LAYER_SIDE_A.powi(2) * min_depth_a * SOLUTE_PACKING_FRACTION;
    let fill_min_layer_count = (min_layer_capacity / mol_volume_a3)
        .ceil()
        .max(MIN_SOLUTE_COPIES as f32) as usize;
    let requested = TARGET_SOLUTE_COPIES.max(fill_min_layer_count.min(MAX_SOLUTE_COPIES));

    requested.min(copy_cap).max(min_copies)
}

fn boundary_layer_setup(mol: &MoleculeSmall) -> BoundaryLayerSetup {
    let solute_radius_a = mol_bounding_radius(mol) as f32;
    let mol_volume_a3 = mol_volume_estimate_a3(mol, solute_radius_a);
    let solute_min_depth =
        MIN_SOLUTE_LAYER_DEPTH_A.max(2.0 * solute_radius_a + 2.0 * SOLUTE_LAYER_WALL_MARGIN_A);

    let solute_copy_count = bounded_solute_copy_count(mol, mol_volume_a3, solute_min_depth);

    let water_molecule_count = TARGET_TOTAL_MOLECULES
        .saturating_sub(solute_copy_count)
        .max(MIN_WATER_MOLECULES);

    let target_solute_layer_volume =
        solute_copy_count as f32 * mol_volume_a3 / SOLUTE_PACKING_FRACTION;
    let footprint_side = (target_solute_layer_volume / solute_min_depth)
        .sqrt()
        .max(MIN_LAYER_SIDE_A)
        .max(2.0 * solute_radius_a + 2.0 * LAYER_MARGIN_A);
    let interface_area = footprint_side * footprint_side;
    let solute_layer_depth = (target_solute_layer_volume / interface_area).max(solute_min_depth);
    let density_depth = water_molecule_count as f32 / (WATER_MOLS_PER_A3 * interface_area);
    let water_layer_depth = density_depth.max(MIN_WATER_DEPTH_A);
    let box_z =
        LAYER_MARGIN_A + solute_layer_depth + INTERFACE_GAP_A + water_layer_depth + LAYER_MARGIN_A;

    let water_slab_low_z = -box_z / 2.0 + LAYER_MARGIN_A + solute_layer_depth + INTERFACE_GAP_A;
    let water_slab_high_z = box_z / 2.0 - LAYER_MARGIN_A;

    BoundaryLayerSetup {
        solute_copy_count,
        water_molecule_count,
        box_extent_a: Vec3F32::new(footprint_side, footprint_side, box_z),
        solute_layer_depth_a: solute_layer_depth,
        water_layer_depth_a: water_layer_depth,
        water_slab_low_z_a: water_slab_low_z,
        water_slab_high_z_a: water_slab_high_z,
    }
}

fn make_md_cfg(
    setup: BoundaryLayerSetup,
    solvent: Solvent,
    memory_snapshots: bool,
    solvent_template_type: Option<SolventTemplateType>,
) -> MdConfig {
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
        solvent_template_type: solvent_template_type.unwrap_or_default(),
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

fn centered_solute_packing_cell(setup: BoundaryLayerSetup) -> SimBox {
    let half_x = (setup.box_extent_a.x / 2.0 - SOLUTE_LAYER_WALL_MARGIN_A).max(1.0);
    let half_y = (setup.box_extent_a.y / 2.0 - SOLUTE_LAYER_WALL_MARGIN_A).max(1.0);
    let half_z = (setup.solute_layer_depth_a / 2.0 - SOLUTE_LAYER_WALL_MARGIN_A).max(1.0);

    SimBox::new(
        Vec3F32::new(-half_x, -half_y, -half_z),
        Vec3F32::new(half_x, half_y, half_z),
    )
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

fn fallback_solute_layer(
    template: &MolDynamics,
    setup: BoundaryLayerSetup,
) -> io::Result<Vec<MolDynamics>> {
    let mut placed = Vec::with_capacity(setup.solute_copy_count);
    add_copies(
        &mut placed,
        template,
        setup.solute_copy_count,
        Some((
            (setup.box_extent_a.x - 2.0 * SOLUTE_LAYER_WALL_MARGIN_A).max(1.0),
            (setup.box_extent_a.y - 2.0 * SOLUTE_LAYER_WALL_MARGIN_A).max(1.0),
            (setup.solute_layer_depth_a - 2.0 * SOLUTE_LAYER_WALL_MARGIN_A).max(1.0),
        )),
    );

    if placed.len() != setup.solute_copy_count {
        return Err(io::Error::other(format!(
            "Boundary-layer solute packing placed {} / {} requested copies.",
            placed.len(),
            setup.solute_copy_count
        )));
    }

    translate_mols(
        &mut placed,
        Vec3F64::new(0.0, 0.0, solute_layer_center_z(setup) as f64),
    );
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
            return fallback_solute_layer(template, setup);
        }
        Err(e) => {
            eprintln!(
                "Boundary-layer shrink packing failed: {}; falling back to grid packing.",
                e.descrip
            );
            return fallback_solute_layer(template, setup);
        }
    };

    translate_mols(
        &mut placed,
        Vec3F64::new(0.0, 0.0, solute_layer_center_z(setup) as f64),
    );

    Ok(placed)
}

// todo: This is hella sus. This should be in dynamics.
fn make_water_geometry(o: Vec3F64, rng: &mut impl Rng) -> WaterGeometry {
    let rot: Quaternion = random_quaternion(rng, None).into();

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

fn water_layer_init_template(
    setup: BoundaryLayerSetup,
    waters: &[WaterGeometry],
) -> io::Result<WaterInitTemplate> {
    let to_f32 = |v: Vec3F64| Vec3F32::new(v.x as f32, v.y as f32, v.z as f32);
    let zero_vel = Vec3F32::new(0.0, 0.0, 0.0);

    WaterInitTemplate::from_parts(
        waters.iter().map(|water| to_f32(water.o)).collect(),
        waters.iter().map(|water| to_f32(water.h0)).collect(),
        waters.iter().map(|water| to_f32(water.h1)).collect(),
        vec![zero_vel; waters.len()],
        vec![zero_vel; waters.len()],
        vec![zero_vel; waters.len()],
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
        ),
    )
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

fn gro_posit_nm(posit_a: Vec3F64, shift_a: Vec3F64) -> Vec3F64 {
    (posit_a + shift_a) / 10.0
}

fn layer_gro_atom(
    mol_id: usize,
    mol_name: &str,
    atom_name: &str,
    atom_serial: usize,
    element: Element,
    posit_a: Vec3F64,
    shift_a: Vec3F64,
) -> AtomGro {
    AtomGro {
        mol_id: mol_id as u32,
        mol_name: mol_name.to_string(),
        element,
        atom_type: atom_name.to_string(),
        serial_number: atom_serial as u32,
        posit: gro_posit_nm(posit_a, shift_a),
        velocity: None,
    }
}

fn make_layer_gro(
    solute: &MoleculeInput,
    waters: &[WaterGeometry],
    setup: BoundaryLayerSetup,
) -> io::Result<String> {
    let total_atoms = solute.atoms.len() + waters.len() * 4;
    let shift = Vec3F64::new(
        setup.box_extent_a.x as f64 / 2.0,
        setup.box_extent_a.y as f64 / 2.0,
        setup.box_extent_a.z as f64 / 2.0,
    );

    let mut atoms = Vec::with_capacity(total_atoms);
    let mut atom_serial = 1usize;
    for atom in &solute.atoms {
        let atom_name = gromacs_atom_name(atom, atom_serial);
        atoms.push(layer_gro_atom(
            1,
            &solute.name,
            &atom_name,
            atom_serial,
            atom.element,
            atom.posit,
            shift,
        ));
        atom_serial += 1;
    }

    for (water_i, water) in waters.iter().enumerate() {
        let mol_id = water_i + 2;
        for (name, element, posit) in [
            ("OW", Element::Oxygen, water.o),
            ("HW1", Element::Hydrogen, water.h0),
            ("HW2", Element::Hydrogen, water.h1),
            ("MW", Element::Oxygen, water.m),
        ] {
            atoms.push(layer_gro_atom(
                mol_id,
                "SOL",
                name,
                atom_serial,
                element,
                posit,
                shift,
            ));
            atom_serial += 1;
        }
    }

    let gro = Gro {
        atoms,
        head_text: "Molchanica boundary-layer solute/water preset".to_string(),
        box_vec: Vec3F64::new(
            setup.box_extent_a.x as f64 / 10.0,
            setup.box_extent_a.y as f64 / 10.0,
            setup.box_extent_a.z as f64 / 10.0,
        ),
    };

    let mut bytes = Vec::new();
    gro.write_to(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|e| io::Error::other(format!("Invalid UTF-8 writing layer GRO: {e}")))
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
    let waters = make_water_layer(setup);
    let water_template = water_layer_init_template(setup, &waters)?;
    let cfg = make_md_cfg(
        setup,
        Solvent::WaterOpcSpecifyMolCount(setup.water_molecule_count),
        true,
        Some(SolventTemplateType::Custom(water_template)),
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

fn run_gromacs_backend(
    placed_mols: &[MolDynamics],
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    setup: BoundaryLayerSetup,
) -> io::Result<(BoundaryLayerMdData, Vec<Snapshot>)> {
    let solute_input = gromacs_layer_molecule_input(placed_mols, &mol.common.ident)?;
    let waters = make_water_layer(setup);
    let gro_text = make_layer_gro(&solute_input, &waters, setup)?;

    let cfg = make_md_cfg(setup, Solvent::None, false, None);
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

    input.initial_gro = Some(gro_text);
    input.solvent = Some(bf_gromacs::solvate::Solvent::Custom(
        bf_gromacs::solvate::CustomSolventTemplate {
            gro_text: String::new(),
            topology_molecules: Vec::new(),
            include_opc_water: true,
        },
    ));
    input
        .extra_molecule_counts
        .push(("SOL".to_string(), waters.len()));

    let out = input.run()?;

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
    let placed_mols = pack_solute_layer(&template, setup, &param_set, dev)?;

    println!(
        "Boundary-layer MD setup ({backend}): {} solute copies, {} waters, footprint {:.1} x {:.1} A, depths {:.1}/{:.1} A, box z {:.1} A",
        setup.solute_copy_count,
        setup.water_molecule_count,
        setup.box_extent_a.x,
        setup.box_extent_a.y,
        setup.solute_layer_depth_a,
        setup.water_layer_depth_a,
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
