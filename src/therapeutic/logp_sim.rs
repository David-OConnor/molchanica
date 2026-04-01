//! Creates a MD sim to measure LogP or LogD. Measures a single molecule of solute in a box of
//! water, and one in a box of water-saturated octanol. Compares the alchemical free energy of
//! the two to estimate LogP or LogD.
//!
//! This is a lipophilicity measurement, and is an important metric for a molecule's use a drug.
//!
//! Rough targets to start: 500-2000 water mols in the water box
//! (What sim box size does this work out to?)
//!
//! 200-800 octanols in the octanol box. 27% mol water in it. So, 356 octanol + 132 water to start?
//! 1 copy of the solute in each box.

// todo: Create an octanol-initialization template.

use std::collections::{HashMap, HashSet};

use bio_files::BondType;
use dynamics::{
    FfMolType, HydrogenConstraint, Integrator, MdConfig, MdOverrides, MolDynamics, ParamError,
    SimBoxInit, Solvent, TAU_TEMP_DEFAULT, snapshot::SnapshotHandlers,
};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;
use na_seq::Element::{Carbon, Hydrogen, Oxygen};

use crate::{
    md::{build_dynamics, post_run_cleanup, run_dynamics_blocking},
    molecules::{Atom, Bond, small::MoleculeSmall},
    state::State,
};

// add_copies uses bounding-sphere exclusion (~14 Å center-to-center for octanol).
// Max reliably placeable per box: ≈ (box - 16)³ × 0.64 / (4/3 π 7³).
// 55 Å → ~26 molecules; 20 is comfortably within that.
//
// TODO: add_copies is not designed for liquid-density packing. 20 octanol in a 55 Å box is
// only ~4% of liquid octanol density. A proper octanol phase requires a pre-equilibrated
// template (similar to the water template) or a lattice-based initializer.

// Rough box size by octanol count:
// 20 (8 water): 18
// 100 (37 water): 30
// 300: (111 water): 43
// 356: (132 water): 46
// 400: (148 water): 48

const OCTANOL_BOX_SIZE: f32 = 46.; // 356 octanol + 132 water ≈ 97,230 Å³ ≈ 46³ Å³
const WATER_BOX_SIZE: f32 = 35.; // Å — 35 Å → ~1,400 TIP3P water mols; > 2× the 12 Å NB cutoff.

const OCTANOL_COUNT: usize = 356;
// 27 mol% water in water-saturated 1-octanol (literature value).
// water/(water+octanol) = 0.27  →  water = octanol × 0.27/0.73 ≈ octanol × 0.37.
const WATER_MOL_PER_OCTANOL: f32 = 0.27 / 0.73;
const OCTANOL_BOX_WATER_COUNT: usize = (OCTANOL_COUNT as f32 * WATER_MOL_PER_OCTANOL) as usize;

const DT: f32 = 0.002; // ps
// Minimum ~100 ps (50_000 steps) for basic equilibration; production LogP needs ≥1 ns.
// const NUM_STEPS: usize = 50_000;
const NUM_STEPS: usize = 2_000; // todo temp while testing the initialization

const TEMP_TGT: f32 = 298.15; // Standard LogP is measured at 25 °C = 298.15 K.

// The conversion factor between ln and log10
const LOG_CONV: f32 = 1. / 2.303;

#[allow(clippy::doc_lazy_continuation)]
/// Using PubChem data as a reference. Partial charges are computed using ORCA. We us this input:
/// ! HF 6-31G* Opt TightSCF TightOpt RESP
///
/// * xyz 0 1
/// O       4.94420     -0.39760     -0.04630
/// ...
///
/// This may be less accurate than more modern basis sets and MBIS charges, but may perform
/// better with AMBER MD force fields.
pub fn make_octanol() -> MoleculeSmall {
    // todo: Get a higher quality source of partial charges.
    // (El, posit, FF name, partial charge.)
    #[rustfmt::skip]
    let atoms = [
        (Oxygen,   Vec3::new( 4.9442, -0.3976, -0.0463), "oh", -0.730420), //  1
        (Carbon,   Vec3::new( 0.0273, -0.3598,  0.0738), "c3",  0.154597), //  2
        (Carbon,   Vec3::new(-1.2646,  0.4624,  0.0583), "c3", -0.078612), //  3
        (Carbon,   Vec3::new( 1.2942,  0.4958,  0.0044), "c3", -0.034888), //  4
        (Carbon,   Vec3::new(-2.5319, -0.3950,  0.0532), "c3", -0.011862), //  5
        (Carbon,   Vec3::new( 2.5500, -0.3747, -0.0005), "c3", -0.092695), //  6
        (Carbon,   Vec3::new(-3.7874,  0.4741, -0.0356), "c3",  0.208880), //  7
        (Carbon,   Vec3::new( 3.8175,  0.4679, -0.0278), "c3",  0.388562), //  8
        (Carbon,   Vec3::new(-5.0492, -0.3730, -0.0796), "c3", -0.327900), //  9
        (Hydrogen, Vec3::new( 0.0503, -0.9725,  0.9835), "hc", -0.031046), // 10
        (Hydrogen, Vec3::new( 0.0176, -1.0547, -0.7753), "hc", -0.034188), // 11
        (Hydrogen, Vec3::new(-1.2784,  1.1276,  0.9306), "hc",  0.010158), // 12
        (Hydrogen, Vec3::new(-1.2632,  1.1063, -0.8301), "hc",  0.004831), // 13
        (Hydrogen, Vec3::new( 1.3198,  1.1783,  0.8623), "hc", -0.008392), // 14
        (Hydrogen, Vec3::new( 1.2714,  1.1141, -0.9010), "hc",  0.001771), // 15
        (Hydrogen, Vec3::new(-2.5656, -1.0062,  0.9629), "hc",  0.001025), // 16
        (Hydrogen, Vec3::new(-2.5025, -1.0845, -0.7992), "hc",  0.001729), // 17
        (Hydrogen, Vec3::new( 2.5379, -1.0481, -0.8666), "hc",  0.009347), // 18
        (Hydrogen, Vec3::new( 2.5665, -1.0240,  0.8838), "hc",  0.029857), // 19
        (Hydrogen, Vec3::new(-3.8347,  1.1483,  0.8274), "hc", -0.034454), // 20
        (Hydrogen, Vec3::new(-3.7454,  1.0998, -0.9347), "hc", -0.033270), // 21
        (Hydrogen, Vec3::new( 3.8553,  1.1097, -0.9137), "h1", -0.055477), // 22
        (Hydrogen, Vec3::new( 3.8880,  1.1005,  0.8622), "h1",  0.018941), // 23
        (Hydrogen, Vec3::new(-5.1386, -0.9902,  0.8201), "hc",  0.072069), // 24
        (Hydrogen, Vec3::new(-5.0475, -1.0346, -0.9517), "hc",  0.071193), // 25
        (Hydrogen, Vec3::new(-5.9342,  0.2680, -0.1411), "hc",  0.075105), // 26
        (Hydrogen, Vec3::new( 4.8901, -0.9332, -0.8561), "ho",  0.425140), // 27
    ];

    // Serial numbers (1 ripple)
    // (atom_0_sn, atom_1_sn)
    let bonds = [
        (1, 8),
        (1, 27),
        (2, 3),
        (2, 4),
        (2, 10),
        (2, 11),
        (3, 5),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 14),
        (4, 15),
        (5, 7),
        (5, 16),
        (5, 17),
        (6, 8),
        (6, 18),
        (6, 19),
        (7, 9),
        (7, 20),
        (7, 21),
        (8, 22),
        (8, 23),
        (9, 24),
        (9, 25),
        (9, 26),
    ];

    let atoms: Vec<_> = atoms
        .into_iter()
        .enumerate()
        .map(|(i, (element, posit, ff_name, q))| Atom {
            serial_number: i as u32 + 1,
            posit,
            element,
            type_in_res_general: Some(ff_name.to_string()),
            partial_charge: Some(q),
            ..Default::default()
        })
        .collect();

    let bonds: Vec<_> = bonds
        .into_iter()
        .map(|(atom_0_sn, atom_1_sn)| Bond {
            bond_type: BondType::Single,
            atom_0: atom_0_sn as usize - 1,
            atom_1: atom_1_sn as usize - 1,
            atom_0_sn,
            atom_1_sn,
            is_backbone: false,
        })
        .collect();

    MoleculeSmall::new("Octanol".to_string(), atoms, bonds, HashMap::new(), None)
}

/// A sim of the molecule in water-saturated Octanol. Returns free energy.
fn run_octanol(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    // Clone so we can set selected_for_md without mutating the caller's molecule.
    let mut mol = mol.clone();
    mol.common.selected_for_md = true;

    let mut octanol = make_octanol();
    octanol.update_ff_related(
        &mut state.mol_specific_params,
        state.ff_param_set.small_mol.as_ref().unwrap(),
        false,
    );

    // Build a MolDynamics from the octanol MoleculeSmall so it can be used in Solvent::Custom.
    let octanol_dyn = {
        // let msp = state.mol_specific_params.get(&octanol.common.ident).cloned();
        MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: octanol
                .common
                .atoms
                .iter()
                .map(|a| a.to_generic())
                .collect(),
            atom_posits: Some(octanol.common.atom_posits.clone()),
            bonds: octanol
                .common
                .bonds
                .iter()
                .map(|b| b.to_generic())
                .collect(),
            adjacency_list: Some(octanol.common.adjacency_list.clone()),
            // mol_specific_params: msp,
            mol_specific_params: None,
            ..Default::default()
        }
    };

    // Only the solute molecule goes in `mols`; octanol is the solvent, packed via Solvent::Custom.
    let mols = [(FfMolType::SmallOrganic, &mol.common, 1)];

    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        temp_target: TEMP_TGT,
        barostat_cfg: Some(Default::default()),
        hydrogen_constraint: Default::default(),
        snapshot_handlers: SnapshotHandlers::default(),
        sim_box: SimBoxInit::new_cube(OCTANOL_BOX_SIZE),
        solvent: Solvent::Custom((vec![(octanol_dyn, OCTANOL_COUNT)], OCTANOL_BOX_WATER_COUNT)),
        overrides: MdOverrides {
            // long_range_recip_disabled: true,
            snapshots_during_equilibration: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut md = build_dynamics(
        &state.dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        &cfg,
        false,
        None,
        &mut HashSet::new(),
    )?;

    // state.volatile.md_local.viewer.update_mols_for_disp(&mols);
    // // Register octanol copies so they appear in snapshot playback alongside the solute.
    // state
    //     .volatile
    //     .md_local
    //     .viewer
    //     .update_custom_solvents_for_disp(&[(&octanol, OCTANOL_COUNT)]);

    // Blocking.
    run_dynamics_blocking(&mut md, &state.dev, DT, NUM_STEPS);

    // todo: Instead of running a sim, perhaps just use the one-off energy computation.

    println!("Snaps: {:?}", md.snapshots.len());

    // todo: This is for the whole system including water molecules. Is this waht we want?
    let energy = {
        let snap = &md.snapshots[md.snapshots.len() - 1];
        if let Some(en) = &snap.energy_data {
            en.energy_potential + en.energy_kinetic
        } else {
            0.
        }
    };
    println!(
        "Free energy computed for the octanol component: {:.2}",
        energy
    );

    state.volatile.md_local.mol_dynamics = Some(md); // todo: Required to visualize?

    post_run_cleanup(state, scene, updates);

    Ok(energy)
}

/// A sim of the molecule in water. Returns free energy.
fn run_water(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    // Clone so we can set selected_for_md without mutating the caller's molecule.
    let mut mol = mol.clone();
    mol.common.selected_for_md = true;

    let mols = [(FfMolType::SmallOrganic, &mol.common, 1)];

    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(TAU_TEMP_DEFAULT),
        },
        temp_target: TEMP_TGT,
        barostat_cfg: Some(Default::default()),
        hydrogen_constraint: HydrogenConstraint::default(),
        snapshot_handlers: SnapshotHandlers::default(),
        sim_box: SimBoxInit::new_cube(WATER_BOX_SIZE),
        overrides: MdOverrides {
            // long_range_recip_disabled: true,
            snapshots_during_equilibration: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut md = build_dynamics(
        &state.dev,
        &mols,
        &state.ff_param_set,
        &state.mol_specific_params,
        &cfg,
        false,
        None,
        &mut HashSet::new(),
    )?;

    // state.volatile.md_local.viewer.update_mols_for_disp(&mols);

    // Blocking.
    run_dynamics_blocking(&mut md, &state.dev, DT, NUM_STEPS);

    println!("Snaps: {:?}", md.snapshots.len());

    // todo: This is for the whole system including water molecules. Is this waht we want?
    let energy = {
        let snap = &md.snapshots[md.snapshots.len() - 1];
        snap.energy_data.as_ref().unwrap().energy_potential
            + snap.energy_data.as_ref().unwrap().energy_kinetic
    };
    println!(
        "Free energy computed for the water component: {:.2}",
        energy
    );

    state.volatile.md_local.mol_dynamics = Some(md); // todo: Required to visualize?

    // todo?
    post_run_cleanup(state, scene, updates);

    // todo: DRY between the two fns.
    Ok(energy)
}

pub fn run(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    let e_water = 0.;
    let e_octanol = 0.;

    // todo: Octanol init is slow.
    // let e_water = run_water(mol, state, scene, updates)?;
    let e_octanol = run_octanol(mol, state, scene, updates)?;

    // todo: How do we calculate alchemical free energies?
    // todo: One LLM thinkjs I shoujld calculate something across different
    // todo lamda values (~20 of them, e.g. lamda = 0.0, 0.05, 0.1, to 1.0

    Ok((e_water - e_octanol) * LOG_CONV)
}
