use std::{collections::HashMap, path::Path};

use bio_files::Sdf;
use dynamics::{
    ComputationDevice, ForceFieldFamily, HydrogenConstraint, MdConfig, MdState,
    MolDynamics, Solvent, gromacs, params::FfParamSet,
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let offline = args[1] == "offline";
    let set = FfParamSet::new_opls().unwrap();
    for file in &args[2..] {
        let source = Sdf::load(Path::new(file)).unwrap();
        let input = MolDynamics::from_sdf(&source, None);
        let mut mol = input.clone();
        let params = if offline {
            set.opls.as_ref().unwrap().build_small_molecule_cached(&mut mol.atoms, &mol.bonds).unwrap()
        } else {
            set.opls.as_ref().unwrap().build_small_molecule(&mut mol.atoms, &mol.bonds).unwrap()
        };
        assert_eq!(mol.atoms.len(), input.atoms.len());
        for (a, b) in mol.atoms.iter().zip(&input.atoms) {
            assert_eq!(a.serial_number, b.serial_number);
            assert_eq!(a.posit, b.posit);
            assert!(a.force_field_type.is_some());
        }
        mol.explicit_params = Some(params.clone());
        let top = gromacs::make_topology(&[("LIG".into(), mol.clone(), 1)], &set, None, false).unwrap();
        let name = Path::new(file).file_stem().unwrap().to_str().unwrap();
        std::fs::write(format!(".opls-dev/{name}.top"), &top).unwrap();

        let mut reversed = input.clone();
        reversed.atoms.reverse();
        let mut serials = HashMap::new();
        for (i, a) in reversed.atoms.iter_mut().enumerate() {
            serials.insert(a.serial_number, 1000 + i as u32 * 3);
            a.serial_number = 1000 + i as u32 * 3;
        }
        for b in &mut reversed.bonds {
            b.atom_0_sn = serials[&b.atom_0_sn];
            b.atom_1_sn = serials[&b.atom_1_sn];
        }
        let other_set = FfParamSet::new_opls().unwrap();
        let reversed_params = other_set.opls.as_ref().unwrap()
            .build_small_molecule_cached(&mut reversed.atoms, &reversed.bonds).unwrap();
        assert_eq!(reversed_params.bonds.len(), params.bonds.len());
        for (a, b) in mol.atoms.iter().zip(reversed.atoms.iter().rev()) {
            assert_eq!(a.force_field_type, b.force_field_type);
            assert!((a.partial_charge.unwrap() - b.partial_charge.unwrap()).abs() < 1e-5);
        }
        println!("{name}: {} atoms, {} bonds, {} angles, {} torsions, {} impropers, {} scaled pairs; reordered disk-cache hit passed",
            mol.atoms.len(), params.bonds.len(), params.angles.len(), params.dihedrals.len(), params.impropers.len(), params.pairs_14.len());

        let mut uncharged = mol.clone();
        for atom in &mut uncharged.atoms { atom.partial_charge = Some(0.); }
        let mut cfg = MdConfig::for_family(ForceFieldFamily::Opls);
        cfg.solvent = Solvent::None;
        cfg.hydrogen_constraint = HydrogenConstraint::Flexible;
        cfg.max_init_relaxation_iters = None;
        cfg.barostat_cfg = None;
        cfg.recenter_sim_box = false;
        cfg.lj_cutoff = 25.;
        cfg.overrides.long_range_recip_disabled = true;
        let (mut md, _) = MdState::new(&ComputationDevice::Cpu, &cfg, &[uncharged], &set).unwrap();
        md.step(&ComputationDevice::Cpu, 0., None);
        let energy = md.snapshots[0].energy_data.as_ref().unwrap();
        let rows: Vec<_> = md.atoms.iter().map(|a| json!({
            "position": [a.posit.x, a.posit.y, a.posit.z],
            "force": [a.force.x, a.force.y, a.force.z]
        })).collect();
        std::fs::write(format!(".opls-dev/{name}.json"), json!({
            "energy": energy.energy_potential,
            "atoms": rows,
        }).to_string()).unwrap();
        assert!(energy.energy_potential.is_finite());
        assert!(md.atoms.iter().all(|a| a.force.x.is_finite() && a.force.y.is_finite() && a.force.z.is_finite()));
    }
}
