//! Creates a MD sim to measure LogP or LogD, by creating a solvent mixture of 50/50 water
//! and octonal, along with many copies of the solute being measured. The ratio of solute which
//! ends in the water vs octanol is the measurement result.
//!
//! This is a lipophilicity measurement.

use std::collections::{HashMap, HashSet};

use bio_files::BondType;
use dynamics::{
    FfMolType, HydrogenConstraint, Integrator, MdConfig, MdOverrides, MdState, ParamError,
    SimBoxInit, snapshot::SnapshotHandler,
};
use graphics::{EngineUpdates, Scene};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::Element::{Carbon, Hydrogen, Oxygen};

use crate::{
    md::{build_dynamics, post_run_cleanup, run_dynamics_blocking},
    molecules::{Atom, Bond, small::MoleculeSmall},
    state::State,
};

/// Using PubChem data as a reference. Could also load from SMILES.
pub fn make_octanol() -> MoleculeSmall {
    // todo: Get a higher quality source of partial charges.
    // (El, posit, FF name, partial charge.)
    #[rustfmt::skip]
    let atoms = [
        (Oxygen,   Vec3::new( 4.9442, -0.3976, -0.0463), "oh", -0.674109), //  1
        (Carbon,   Vec3::new( 0.0273, -0.3598,  0.0738), "c3", -0.081901), //  2
        (Carbon,   Vec3::new(-1.2646,  0.4624,  0.0583), "c3", -0.082301), //  3
        (Carbon,   Vec3::new( 1.2942,  0.4958,  0.0044), "c3", -0.081815), //  4
        (Carbon,   Vec3::new(-2.5319, -0.3950,  0.0532), "c3", -0.082398), //  5
        (Carbon,   Vec3::new( 2.5500, -0.3747, -0.0005), "c3", -0.090847), //  6
        (Carbon,   Vec3::new(-3.7874,  0.4741, -0.0356), "c3", -0.081866), //  7
        (Carbon,   Vec3::new( 3.8175,  0.4679, -0.0278), "c3",  0.160930), //  8
        (Carbon,   Vec3::new(-5.0492, -0.3730, -0.0796), "c3", -0.094776), //  9
        (Hydrogen, Vec3::new( 0.0503, -0.9725,  0.9835), "hc",  0.048810), // 10
        (Hydrogen, Vec3::new( 0.0176, -1.0547, -0.7753), "hc",  0.048424), // 11
        (Hydrogen, Vec3::new(-1.2784,  1.1276,  0.9306), "hc",  0.047161), // 12
        (Hydrogen, Vec3::new(-1.2632,  1.1063, -0.8301), "hc",  0.046766), // 13
        (Hydrogen, Vec3::new( 1.3198,  1.1783,  0.8623), "hc",  0.047020), // 14
        (Hydrogen, Vec3::new( 1.2714,  1.1141, -0.9010), "hc",  0.046657), // 15
        (Hydrogen, Vec3::new(-2.5656, -1.0062,  0.9629), "hc",  0.048932), // 16
        (Hydrogen, Vec3::new(-2.5025, -1.0845, -0.7992), "hc",  0.048553), // 17
        (Hydrogen, Vec3::new( 2.5379, -1.0481, -0.8666), "hc",  0.048300), // 18
        (Hydrogen, Vec3::new( 2.5665, -1.0240,  0.8838), "hc",  0.048741), // 19
        (Hydrogen, Vec3::new(-3.8347,  1.1483,  0.8274), "hc",  0.047275), // 20
        (Hydrogen, Vec3::new(-3.7454,  1.0998, -0.9347), "hc",  0.046915), // 21
        (Hydrogen, Vec3::new( 3.8553,  1.1097, -0.9137), "h1",  0.042698), // 22
        (Hydrogen, Vec3::new( 3.8880,  1.1005,  0.8622), "h1",  0.042359), // 23
        (Hydrogen, Vec3::new(-5.1386, -0.9902,  0.8201), "hc",  0.046161), // 24
        (Hydrogen, Vec3::new(-5.0475, -1.0346, -0.9517), "hc",  0.046535), // 25
        (Hydrogen, Vec3::new(-5.9342,  0.2680, -0.1411), "hc",  0.045612), // 26
        (Hydrogen, Vec3::new( 4.8901, -0.9332, -0.8561), "ho",  0.482395), // 27
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

// pub fn build_dynamics_logp(mol: &MoleculeSmall, state: &State) -> Result<MdState, ParamError> {
pub fn run_dynamics_logp(
    mol: &MoleculeSmall,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> Result<f32, ParamError> {
    let octanol = make_octanol();

    let num_octanol = 500; // todo A?R

    let num_solute = 100; // todo?

    let mols = [
        (FfMolType::SmallOrganic, &octanol.common, num_octanol),
        (FfMolType::SmallOrganic, &mol.common, num_solute),
    ];

    let simbox_side_len: f32 = 40.; // todo: A/R

    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity {
            thermostat: Some(0.9),
        },
        zero_com_drift: true,
        temp_target: 310.,
        pressure_target: 1.,
        hydrogen_constraint: HydrogenConstraint::Flexible, // for now
        snapshot_handlers: vec![SnapshotHandler::default()],
        sim_box: SimBoxInit::new_cube(simbox_side_len),
        max_init_relaxation_iters: None, // todo A/R
        neighbor_skin: 1.,
        overrides: MdOverrides::default(),
    };

    let mut md = build_dynamics(
        &state.dev,
        &mols,
        None,
        &state.ff_param_set,
        &state.mol_specific_params,
        &cfg,
        state.ui.md.peptide_static,
        None,
        &mut HashSet::new(),
        false,
    )?;

    // Blocking.
    let dt = 0.002; // todo?
    let n_steps = 500;
    run_dynamics_blocking(&mut md, &state.dev, dt, n_steps);

    post_run_cleanup(state, scene, updates);

    Ok(0.)
}
