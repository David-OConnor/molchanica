//! Split ligands into fragments, or join them with a new covalent bond.

use std::{f64::consts::TAU, io};

use bio_files::{BondType, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, HydrogenConstraint, MdConfig, MdOverrides, MdState, MolDynamics, SimBoxInit,
    Solvent, params::FfParamSet,
};
use graphics::{ControlScheme, EngineUpdates, Scene};
use lin_alg::f64::{Quaternion, Vec3};
use mol_defs::molecules::{Bond, MolIdent, MolType, MoleculeGeneric, small::MoleculeSmall};
use na_seq::Element;

use crate::{
    drawing::wrappers::{draw_all_ligs, draw_all_pockets},
    mol_manip::ManipMode,
    selection::Selection,
    state::State,
    util::{RedrawFlags, close_mol, handle_err, handle_success, orbit_center},
};

/// A temporary first endpoint. The snapshot prevents an edit between clicks from silently
/// joining a different atom. This workflow deliberately leaves Selection's format unchanged.
pub struct JoinLigand {
    pub endpoint: (usize, usize),
    atoms: Vec<(u32, Element)>,
    bonds: Vec<(usize, usize, BondType)>,
}

impl JoinLigand {
    pub fn new(lig: &MoleculeSmall, endpoint: (usize, usize)) -> Self {
        Self {
            endpoint,
            atoms: lig
                .common
                .atoms
                .iter()
                .map(|a| (a.serial_number, a.element))
                .collect(),
            bonds: lig
                .common
                .bonds
                .iter()
                .map(|b| (b.atom_0, b.atom_1, b.bond_type))
                .collect(),
        }
    }

    pub fn is_current(&self, state: &State) -> bool {
        state.ligands.get(self.endpoint.0).is_some_and(|lig| {
            self.atoms.iter().copied().eq(lig
                .common
                .atoms
                .iter()
                .map(|a| (a.serial_number, a.element)))
                && self.bonds.iter().copied().eq(lig
                    .common
                    .bonds
                    .iter()
                    .map(|b| (b.atom_0, b.atom_1, b.bond_type)))
        })
    }
}

fn build_joined(
    ligands: &[MoleculeSmall],
    first: (usize, usize),
    second: (usize, usize),
) -> io::Result<MoleculeSmall> {
    if first.0 == second.0 {
        return Err(io::Error::other("Select an atom in a different ligand."));
    }

    let mut parts = Vec::new();
    for (mol_i, atom_i) in [first, second] {
        let lig = ligands
            .get(mol_i)
            .ok_or_else(|| io::Error::other("The selected ligand is no longer open."))?;
        if atom_i >= lig.common.atoms.len() {
            return Err(io::Error::other("The selected atom no longer exists."));
        }
        let indices: Vec<_> = (0..lig.common.atoms.len()).collect();
        parts.push(lig.common.subset(&indices)?);
    }

    let b = parts.pop().unwrap();
    let mut a = parts.pop().unwrap();
    let offset = a.atoms.len();
    a.atoms.extend(b.atoms);
    a.bonds.extend(b.bonds.into_iter().map(|mut bond| {
        bond.atom_0 += offset;
        bond.atom_1 += offset;
        bond
    }));
    a.bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0: first.1,
        atom_1: offset + second.1,
        atom_0_sn: 0,
        atom_1_sn: 0,
        is_backbone: false,
    });
    a.reassign_sns();
    for atom in &mut a.atoms {
        atom.force_field_type = None;
        atom.partial_charge = None;
    }

    let mut joined = MoleculeSmall::new(
        format!("{} + {}", a.ident, b.ident),
        a.atoms,
        a.bonds,
        Default::default(),
        None,
    );
    joined.common.reassign_sns();
    Ok(joined)
}

/// Only two degrees of freedom: separation along the original joining axis and twist
/// around it. Construct every trial from the originals to avoid accumulated distortion.
struct JoinGeometry {
    origin: Vec3,
    axis: Vec3,
    fixed: Vec<Vec3>,
    moving: Vec<Vec3>,
}

impl JoinGeometry {
    fn new(mol: &MoleculeSmall, first: usize, second: usize, offset: usize) -> io::Result<Self> {
        let positions = &mol.common.atom_posits;
        let origin = positions[first];
        let delta = positions[second] - origin;
        if positions
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
            || delta.magnitude() < 1e-6
        {
            return Err(io::Error::other(
                "The joining atoms need finite, distinct positions to define a bond axis.",
            ));
        }

        Ok(Self {
            origin,
            axis: delta.to_normalized(),
            fixed: positions[..offset].iter().map(|p| *p - origin).collect(),
            moving: positions[offset..]
                .iter()
                .map(|p| *p - positions[second])
                .collect(),
        })
    }

    fn positions(&self, length: f64, twist: f64) -> Vec<Vec3> {
        let rotation = Quaternion::from_axis_angle(self.axis, twist);
        let mut result = self.fixed.clone();
        result.extend(
            self.moving
                .iter()
                .map(|p| self.axis * length + rotation.rotate_vec(*p)),
        );
        result
    }

    fn apply(&self, mol: &mut MoleculeSmall, length: f64, twist: f64) {
        // Write only the moving ligand; the first remains bit-for-bit unchanged.
        for (i, local) in self
            .positions(length, twist)
            .into_iter()
            .enumerate()
            .skip(self.fixed.len())
        {
            let posit = local + self.origin;
            mol.common.atom_posits[i] = posit;
            mol.common.atoms[i].posit = posit;
        }
    }
}

/// Reuse the editor's Amber energy model, but never integrate or minimize individual atoms.
/// The original joining axis and all internal lengths/angles stay fixed. The local search
/// is bounded around the force field's equilibrium length so clashes cannot stretch the
/// new bond arbitrarily. The caller can fall back to element-based geometry if this fails.
fn relax_join(
    mol: &mut MoleculeSmall,
    first: usize,
    second: usize,
    offset: usize,
    params: &FfParamSet,
    specific: Option<&ForceFieldParams>,
) -> io::Result<()> {
    let common = &mol.common;
    let types = (
        common.atoms[first].force_field_type.clone(),
        common.atoms[second].force_field_type.clone(),
    );
    let (Some(a), Some(b)) = types else {
        return Err(io::Error::other(
            "Unable to assign force-field types to the joining atoms.",
        ));
    };
    let key = (a.clone(), b.clone());
    let reverse = (b, a);
    let length = [specific, params.small_mol.as_ref()]
        .into_iter()
        .flatten()
        .find_map(|p| p.bond.get(&key).or_else(|| p.bond.get(&reverse)))
        .map(|p| p.r_0 as f64)
        .filter(|r| r.is_finite() && *r > 0.)
        .ok_or_else(|| {
            io::Error::other("No equilibrium bond length is available for this atom pair.")
        })?;

    let geometry = JoinGeometry::new(mol, first, second, offset)?;
    let initial = geometry.positions(length, 0.);
    // Accommodate every trial rotation without periodic images entering the cutoff.
    let radius = geometry
        .fixed
        .iter()
        .chain(&geometry.moving)
        .map(|p| p.magnitude())
        .fold(0., f64::max)
        + length * 1.15;
    let cfg = MdConfig {
        solvent: Solvent::None,
        max_init_relaxation_iters: None,
        hydrogen_constraint: HydrogenConstraint::Flexible,
        sim_box: SimBoxInit::new_cube((2. * (radius + 16.)) as f32),
        overrides: MdOverrides {
            skip_water_relaxation: true,
            skip_counterion_insertion: true,
            long_range_recip_disabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let input = MolDynamics {
        atoms: common.atoms.iter().map(|a| a.to_generic()).collect(),
        bonds: common.bonds.iter().map(|b| b.to_generic()).collect(),
        atom_posits: Some(initial),
        adjacency_list: Some(common.adjacency_list.clone()),
        mol_specific_params: specific.cloned(),
        ..Default::default()
    };
    // Small, repeated energy evaluations avoid GPU setup/transfer overhead.
    let dev = ComputationDevice::Cpu;
    let (mut md, _) = MdState::new(&dev, &cfg, &[input], params)
        .map_err(|e| io::Error::other(format!("Unable to set up constrained relaxation: {e:?}")))?;
    if md.atoms.len() != common.atoms.len() {
        return Err(io::Error::other("Relaxation changed the atom count."));
    }

    let mut energy = |r, theta| {
        let positions = geometry.positions(r, theta);
        for (atom, posit) in md.atoms.iter_mut().zip(positions) {
            atom.posit = posit.into();
        }
        let e = md.minimization_energy(&dev);
        if e.is_finite() { e } else { f64::INFINITY }
    };

    let lower = length * 0.85;
    let upper = length * 1.15;
    let mut best = (length, 0., f64::INFINITY);
    // Search the entire torsion circle, including several lengths, before refining.
    for i in 0..24 {
        let theta = TAU * i as f64 / 24.;
        for r in [length * 0.95, length, length * 1.05] {
            let e = energy(r, theta);
            if e < best.2 {
                best = (r, theta, e);
            }
        }
    }
    if !best.2.is_finite() {
        return Err(io::Error::other(
            "No finite-energy joining geometry was found.",
        ));
    }

    let mut dr = length * 0.025;
    let mut dt = TAU / 48.;
    for _ in 0..60 {
        let previous = best;
        for r_step in [-1., 0., 1.] {
            for t_step in [-1., 0., 1.] {
                if r_step == 0. && t_step == 0. {
                    continue;
                }
                let r = (previous.0 + dr * r_step).clamp(lower, upper);
                let theta = (previous.1 + dt * t_step).rem_euclid(TAU);
                let e = energy(r, theta);
                if e < best.2 {
                    best = (r, theta, e);
                }
            }
        }
        if best.2 >= previous.2 {
            dr *= 0.5;
            dt *= 0.5;
            if dr < 0.0001 && dt < 0.001 {
                break;
            }
        }
    }

    geometry.apply(mol, best.0, best.1);
    Ok(())
}

/// Parameter-free fallback: single-bond covalent radii and a steric torsion search.
/// This is a placement heuristic, not an Amber minimization or a repair of invalid valence.
fn place_join_by_elements(
    mol: &mut MoleculeSmall,
    first: usize,
    second: usize,
    offset: usize,
) -> io::Result<()> {
    let geometry = JoinGeometry::new(mol, first, second, offset)?;
    let radius = |element: Element| {
        let r = element.covalent_radius();
        if r.is_finite() && r > 0. { r } else { 0.75 }
    };
    let length = radius(mol.common.atoms[first].element) + radius(mol.common.atoms[second].element);

    // Across-fragment 1-2 and 1-3 pairs are constrained by the new bond and its
    // fixed valence angles. Exclude them from nonbonded crowding, as Amber does.
    let mut pairs = Vec::new();
    for i in 0..offset {
        for j in offset..mol.common.atoms.len() {
            if (i == first && (j == second || mol.common.adjacency_list[second].contains(&j)))
                || (j == second && mol.common.adjacency_list[first].contains(&i))
            {
                continue;
            }
            let contact = mol.common.atoms[i].element.vdw_radius() as f64
                + mol.common.atoms[j].element.vdw_radius() as f64;
            pairs.push((i, j, contact));
        }
    }
    let score = |twist| {
        let positions = geometry.positions(length, twist);
        pairs
            .iter()
            .map(|&(i, j, contact)| {
                let distance = (positions[i] - positions[j]).magnitude().max(0.1);
                // Smooth repulsion favors less crowded orientations without needing FF types.
                (contact / distance).powi(6)
            })
            .sum::<f64>()
    };
    let mut best = (0., score(0.));
    for i in 1..72 {
        let twist = TAU * i as f64 / 72.;
        let cost = score(twist);
        if cost < best.1 {
            best = (twist, cost);
        }
    }
    let mut step = TAU / 72.;
    for _ in 0..12 {
        let center = best.0;
        for twist in [center - step, center + step] {
            let cost = score(twist);
            if cost < best.1 {
                best = (twist, cost);
            }
        }
        step *= 0.5;
    }
    if !best.1.is_finite() {
        return Err(io::Error::other(
            "Unable to find finite element-based joining geometry.",
        ));
    }
    geometry.apply(mol, length, best.0);
    Ok(())
}

/// Return the Amber failure as a warning when the fallback succeeds. Invalid geometry
/// still fails without modifying the source ligands; unavailable FF data does not.
fn position_join(
    mol: &mut MoleculeSmall,
    first: usize,
    second: usize,
    offset: usize,
    params: &FfParamSet,
    specific: Option<&ForceFieldParams>,
) -> io::Result<Option<String>> {
    match relax_join(mol, first, second, offset, params, specific) {
        Ok(()) => Ok(None),
        Err(e) => {
            place_join_by_elements(mol, first, second, offset)?;
            Ok(Some(format!(
                "Warning: joined ligands using element-based bond length and steric twist; \
                force-field relaxation was unavailable: {e}"
            )))
        }
    }
}

/// Join two distinct ligands with a single bond and rigid-fragment relaxation. Build the complete replacement
/// before modifying state; molecule-specific identifiers and cached chemistry start fresh.
pub fn join_ligands(
    state: &mut State,
    first: (usize, usize),
    second: (usize, usize),
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    let mut joined = match build_joined(&state.ligands, first, second) {
        Ok(mol) => mol,
        Err(e) => {
            handle_err(&mut state.ui, format!("Unable to join ligands: {e}"));
            return;
        }
    };

    // Parameter caches are keyed by name; the new compound needs its own entry.
    let base_ident = joined.common.ident.clone();
    let mut suffix = 2;
    while state
        .mol_specific_params
        .keys()
        .any(|k| k.eq_ignore_ascii_case(&joined.common.ident))
        || state
            .ligands
            .iter()
            .any(|lig| lig.common.ident.eq_ignore_ascii_case(&joined.common.ident))
    {
        joined.common.ident = format!("{base_ident} ({suffix})");
        suffix += 1;
    }

    if let Some(params) = &state.ff_param_set.small_mol {
        joined.update_ff_related(&mut state.mol_specific_params, params, false);
    }
    let offset = state.ligands[first.0].common.atoms.len();
    let specific = state.mol_specific_params.get(&joined.common.ident);
    let warning = match position_join(
        &mut joined,
        first.1,
        offset + second.1,
        offset,
        &state.ff_param_set,
        specific,
    ) {
        Ok(warning) => warning,
        Err(e) => {
            handle_err(&mut state.ui, format!("Unable to join ligands: {e}"));
            return;
        }
    };
    joined.update_characterization();

    // Closing through the existing path also updates open-file history and audio state.
    // Remove from the end first so the original endpoint indices remain valid.
    let orbit_before = state.volatile.orbit_center;
    let mut redraw = RedrawFlags::default();
    for i in [first.0.max(second.0), first.0.min(second.0)] {
        close_mol(MolType::Ligand, i, state, scene, &mut redraw, updates);
        for lig in &mut state.ligands {
            lig.common.copy_for_md = lig.common.copy_for_md.and_then(|parent| {
                if parent == i {
                    None
                } else {
                    Some(parent - usize::from(parent > i))
                }
            });
        }
    }

    state.volatile.thread_receivers.therapeutic_properties_avail = None;
    state.volatile.thread_receivers.amber_geostd_data_avail = None;
    state.volatile.thread_receivers.all_idents_avail = None;
    state.volatile.md_local.mol_dynamics = None;
    state.volatile.mol_manip.mode = ManipMode::None;
    state.ui.selection = Selection::None;
    state.ui.join_ligand = None;
    state.ui.visibility.hide_ligand = false;
    state.ligands.push(joined);
    state.volatile.active_mol = Some((MolType::Ligand, state.ligands.len() - 1));
    state.volatile.orbit_center = match orbit_before {
        Some((MolType::Ligand, i)) if i == first.0 || i == second.0 => state.volatile.active_mol,
        Some((MolType::Ligand, i)) => Some((
            MolType::Ligand,
            i - usize::from(i > first.0) - usize::from(i > second.0),
        )),
        other => other,
    };
    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }
    state.update_save_prefs();
    draw_all_pockets(state, scene, updates);
    draw_all_ligs(state, scene, updates);
    let message = if let Some(warning) = warning {
        eprintln!("{warning}");
        warning
    } else {
        "Joined ligands; relaxed the new bond length and twist with both fragments rigid."
            .to_owned()
    };
    handle_success(&mut state.ui, message);
}

/// Break a ligand apart at one or more of its bonds, as separate molecules. The largest bonded
/// piece stays in this molecule; each of the others becomes a new ligand, where it sits.
///
/// Each bond given must be a place the molecule actually comes apart: cutting a single bond of a
/// ring leaves it joined the other way around, so a ring takes two cuts. Reports what to do if not.
pub fn split_lig_at_bonds(
    state: &mut State,
    lig_i: usize,
    bond_indexes: &[usize],
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    if state.ui.join_ligand.take().is_some() {
        draw_all_ligs(state, scene, engine_updates);
    }

    // Build the new molecules up front, while the ligand still holds the atoms these indices
    // refer to. Scoped so the error paths below can borrow `state.ui`.
    let split = {
        let Some(lig) = state.ligands.get(lig_i) else {
            handle_err(
                &mut state.ui,
                "Error: No ligand to split at its bonds.".to_owned(),
            );
            return;
        };

        let ident = lig.common.name(Some(&lig.idents));

        lig.common
            .components_without_bonds(bond_indexes)
            .and_then(|components| {
                // The first component is the largest; it's what the original molecule keeps.
                let fragments: Vec<_> = components[1..]
                    .iter()
                    .enumerate()
                    .map(|(i, atoms)| {
                        let ident = format!("{} frag {}", lig.common.ident, i + 1);
                        MoleculeSmall::from_fragment(ident, &lig.common, atoms)
                    })
                    .collect::<io::Result<_>>()?;

                let removed: Vec<usize> = components[1..].iter().flatten().copied().collect();
                Ok((fragments, removed))
            })
            .map_err(|e| format!("Unable to split {ident}: {e}"))
    };

    let (fragments, removed) = match split {
        Ok(v) => v,
        Err(e) => {
            handle_err(&mut state.ui, e);
            return;
        }
    };

    let (ident, frag_count) = {
        let lig = &mut state.ligands[lig_i];
        lig.common.remove_atoms(&removed);

        // Partial charges are assigned by serial number, and the removals left gaps.
        lig.common.reassign_sns();

        // Setting these to `None` on any atom triggers an FF param and partial charge rebuild;
        // both depend on the atoms' surroundings, which just changed.
        if let Some(atom) = lig.common.atoms.first_mut() {
            atom.force_field_type = None;
            atom.partial_charge = None;
        }
        lig.ff_params_loaded = false;
        lig.frcmod_loaded = false;

        // What's left is a different compound, so identifiers and data keyed to the whole
        // molecule (a CID, an InChI, assay or structure hits) would now name the wrong thing.
        // The SMILES comes back from the structure below.
        lig.idents.clear();
        lig.associated_structures.clear();
        lig.therapeutic_props = None;

        // Pharmacophore features point at atom indices, which have shifted. Its pocket doesn't,
        // and owns a mesh slot, so leave that in place.
        lig.pharmacophore.features.clear();
        lig.pharmacophore.feature_relations.clear();

        (lig.common.ident.clone(), fragments.len())
    };

    if let Some(params) = &state.ff_param_set.small_mol {
        let lig = &mut state.ligands[lig_i];
        lig.update_ff_related(&mut state.mol_specific_params, params, false);
    } else {
        handle_err(
            &mut state.ui,
            "Error: Unable to update the split molecule's params due to missing GAFF2.".to_owned(),
        );
    }

    let lig = &mut state.ligands[lig_i];
    lig.idents.push(MolIdent::Smiles(lig.common.to_smiles()));
    lig.update_characterization();

    // A simulation set up with this ligand indexes its old atoms.
    if lig.common.selected_for_md.is_some() {
        state.volatile.md_local.mol_dynamics = None;
    }

    // Ligand selections index into atoms and bonds that have shifted.
    if matches!(
        state.ui.selection,
        Selection::AtomLig(_)
            | Selection::AtomsLig(_)
            | Selection::BondLig(_)
            | Selection::BondsLig(_)
    ) {
        state.ui.selection = Selection::None;
    }

    for frag in fragments {
        // In place: the point of the split is that each piece stays where it was.
        state.load_mol_to_state_in_place(MoleculeGeneric::Small(frag), scene, engine_updates);
    }

    // Loading the fragments made the last of them active; the user was working on the original.
    state.volatile.active_mol = Some((MolType::Ligand, lig_i));
    state.ui.visibility.hide_ligand = false;

    handle_success(
        &mut state.ui,
        format!("Split {frag_count} molecule[s] off of {ident}."),
    );
}
