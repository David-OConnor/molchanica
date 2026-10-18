//! Additive representations. Each layer owns its selection, style, and mesh instances.
//! Default views remain available independently for every object.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use graphics::{EngineUpdates, Entity, EntityUpdate, Mesh, Scene};
use lin_alg::f32::{Quaternion, Vec3};
use mol_defs::{
    molecules::{MolGenericRef, MolType},
    sfc_mesh::{SOLVENT_RAD, make_sas_mesh},
};
use na_seq::Element;

use crate::{
    drawing::{
        ATOM_SHININESS, BALL_STICK_RADIUS, COLOR_SELECTED, EntityClass, MESH_BALL_STICK_SPHERE,
        MESH_SPACEFILL_SPHERE, MoleculeView, atoms_bonds::bond_entities,
        ribbon_mesh::build_ribbon_mesh,
    },
    render::MESH_POCKET_START,
    selection::{
        Selection, ViewSelLevel,
        expression::{self, object_key},
    },
    state::{OperatingMode, ResColoring, State},
    util::clear_mol_entity_indices,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Representation {
    pub name: String,
    pub expression: String,
    pub view: MoleculeView,
    pub visible: bool,
    /// None retains element/residue colors.
    pub color: Option<[f32; 3]>,
    pub opacity: f32,
    /// Atom and bond radius multiplier; coordinates are never scaled.
    pub size: f32,
}

impl Representation {
    pub fn new(expression: String, view: MoleculeView) -> Self {
        Self {
            name: format!("{view}: {expression}"),
            expression,
            view,
            visible: true,
            color: None,
            opacity: if view == MoleculeView::Surface {
                0.45
            } else {
                1.
            },
            size: 1.,
        }
    }
}

#[derive(Default)]
pub struct RepresentationCache {
    fingerprint: Option<u64>,
    mesh_slots: Vec<usize>,
    selection: Selection,
    pub errors: Vec<String>,
}

pub fn default_visible(state: &State, mol: &MolGenericRef<'_>) -> bool {
    !state
        .to_save
        .hidden_default_objects
        .contains(&object_key(mol))
}

/// Layer picking uses its own atom set, independently of the default-view filters.
pub fn pick(state: &State, ray: (Vec3, Vec3)) -> Option<(Selection, f32, (MolType, usize))> {
    let direction = (ray.1 - ray.0).to_normalized();
    let mut best = None;
    let (mut best_distance, mut best_depth) = (f32::INFINITY, f32::INFINITY);
    for layer in &state.to_save.representations {
        if !layer.visible || layer.opacity <= 0. {
            continue;
        }
        let Ok(selected) = expression::evaluate(state, &layer.expression) else {
            continue;
        };
        for selected in selected {
            let Some(mol) = state.get_mol(selected.mol_type, selected.mol_i) else {
                continue;
            };
            let common = mol.common();
            if !common.visible {
                continue;
            }
            if layer.view == MoleculeView::Ribbon && selected.mol_type != MolType::Peptide {
                continue;
            }
            let mut mask = vec![false; common.atoms.len()];
            for &a in &selected.atoms {
                mask[a] = true;
            }
            if layer.view == MoleculeView::Ribbon {
                let residues: std::collections::HashSet<_> = selected
                    .atoms
                    .iter()
                    .filter_map(|&a| common.atoms[a].residue)
                    .collect();
                for (i, atom) in common.atoms.iter().enumerate() {
                    mask[i] = atom.role == Some(mol_defs::molecules::AtomRole::C_Alpha)
                        && atom.residue.is_some_and(|r| residues.contains(&r));
                }
            } else if layer.view == MoleculeView::Backbone {
                for (i, atom) in common.atoms.iter().enumerate() {
                    mask[i] &= atom.is_backbone();
                }
            }
            let position = |a: usize| -> Vec3 {
                common
                    .atom_posits
                    .get(a)
                    .copied()
                    .unwrap_or(common.atoms[a].posit)
                    .into()
            };
            let mut consider = |position: Vec3, radius: f32, selection: Selection| {
                let offset = position - ray.0;
                let depth = offset.dot(direction);
                let distance = (offset - direction * depth).magnitude();
                if depth >= 0.
                    && distance < radius
                    && (distance < best_distance - 1e-4
                        || ((distance - best_distance).abs() <= 1e-4 && depth < best_depth))
                {
                    best_distance = distance;
                    best_depth = depth;
                    best = Some((selection, distance, (selected.mol_type, selected.mol_i)));
                }
            };
            for (a, &included) in mask.iter().enumerate() {
                if !included {
                    continue;
                }
                let radius = match layer.view {
                    MoleculeView::SpaceFill => common.atoms[a].element.vdw_radius() * layer.size,
                    MoleculeView::Ribbon | MoleculeView::Surface | MoleculeView::Dots => 1.2,
                    _ => 0.8 * layer.size,
                };
                consider(
                    position(a),
                    radius,
                    Selection::from_atom(selected.mol_type, selected.mol_i, a),
                );
            }
            if matches!(
                layer.view,
                MoleculeView::Sticks | MoleculeView::BallAndStick | MoleculeView::Backbone
            ) {
                for (i, bond) in common.bonds.iter().enumerate() {
                    if mask.get(bond.atom_0) == Some(&true) && mask.get(bond.atom_1) == Some(&true)
                    {
                        consider(
                            (position(bond.atom_0) + position(bond.atom_1)) / 2.,
                            0.6 * layer.size,
                            Selection::from_bond(selected.mol_type, selected.mol_i, i),
                        );
                    }
                }
            }
        }
    }
    best
}

pub fn add(state: &mut State, expression: &str, view: MoleculeView) -> Result<usize, String> {
    let expression = expression::freeze_current(state, expression)?;
    let atoms = expression::evaluate(state, &expression)?;
    if view == MoleculeView::Ribbon && atoms.iter().any(|a| a.mol_type != MolType::Peptide) {
        return Err("Cartoon layers require a protein selection".into());
    }
    let count = atoms.iter().map(|a| a.atoms.len()).sum();
    state
        .to_save
        .representations
        .push(Representation::new(expression, view));
    state.to_save.save_flag = true;
    Ok(count)
}

fn fingerprint(state: &State) -> u64 {
    let mut hash = DefaultHasher::new();
    format!("{:?}", state.to_save.named_selections).hash(&mut hash);
    format!("{:?}", state.to_save.representations).hash(&mut hash);
    state.to_save.sa_surface_precision.to_bits().hash(&mut hash);
    for (kind, i, mol) in expression::objects(state) {
        (kind as u8).hash(&mut hash);
        i.hash(&mut hash);
        object_key(&mol).hash(&mut hash);
        let common = mol.common();
        common.visible.hash(&mut hash);
        common.name.hash(&mut hash);
        common.filename.hash(&mut hash);
        common.ident.hash(&mut hash);
        common.atoms.len().hash(&mut hash);
        common.bonds.len().hash(&mut hash);
        for bond in &common.bonds {
            bond.atom_0.hash(&mut hash);
            bond.atom_1.hash(&mut hash);
            (bond.bond_type as u8).hash(&mut hash);
        }
        match &mol {
            MolGenericRef::Peptide(p) => {
                for residue in &p.residues {
                    residue.serial_number.hash(&mut hash);
                    residue.res_type.to_string().hash(&mut hash);
                }
                for chain in &p.chains {
                    chain.id.hash(&mut hash);
                }
                format!("{:?}", p.secondary_structure).hash(&mut hash);
            }
            MolGenericRef::NucleicAcid(p) => {
                for residue in &p.residues {
                    residue.serial_number.hash(&mut hash);
                    residue.res_type.to_string().hash(&mut hash);
                }
            }
            MolGenericRef::Lipid(p) => {
                for residue in &p.residues {
                    residue.serial_number.hash(&mut hash);
                    residue.res_type.to_string().hash(&mut hash);
                }
            }
            _ => (),
        }
        for (i, atom) in common.atoms.iter().enumerate() {
            atom.serial_number.hash(&mut hash);
            atom.element.hash(&mut hash);
            atom.residue.hash(&mut hash);
            atom.chain.hash(&mut hash);
            atom.partial_charge.map(f32::to_bits).hash(&mut hash);
            atom.occupancy.map(f32::to_bits).hash(&mut hash);
            atom.type_in_res_general.hash(&mut hash);
            atom.alt_conformation_id.hash(&mut hash);
            atom.role.map(|role| role as u8).hash(&mut hash);
            atom.type_in_res
                .as_ref()
                .map(std::mem::discriminant)
                .hash(&mut hash);
            if let Some(na_seq::AtomTypeInRes::H(name) | na_seq::AtomTypeInRes::Hetero(name)) =
                &atom.type_in_res
            {
                name.hash(&mut hash);
            }
            atom.hetero.hash(&mut hash);
            let p = common.atom_posits.get(i).copied().unwrap_or(atom.posit);
            p.x.to_bits().hash(&mut hash);
            p.y.to_bits().hash(&mut hash);
            p.z.to_bits().hash(&mut hash);
        }
    }
    hash.finish()
}

fn style_entity(entity: &mut Entity, layer: &Representation) {
    entity.class = EntityClass::Representation as u32;
    entity.opacity = layer.opacity;
    if let Some([r, g, b]) = layer.color {
        entity.color = (r, g, b);
    }
}

fn mesh_entity(
    mesh: Mesh,
    layer: &Representation,
    meshes: &mut Vec<Mesh>,
    entities: &mut Vec<Entity>,
    mesh_start: usize,
) {
    if mesh.vertices.is_empty() {
        return;
    }
    let mut mesh = mesh;
    if let Some([r, g, b]) = layer.color {
        for vertex in &mut mesh.vertices {
            vertex.color = Some(((r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8, 255));
        }
    }
    let mut entity = Entity::new(
        mesh_start + meshes.len(),
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0.6, 0.75, 0.9),
        ATOM_SHININESS,
    );
    style_entity(&mut entity, layer);
    meshes.push(mesh);
    entities.push(entity);
}

fn draw_layer(
    state: &State,
    layer: &Representation,
    mesh_start: usize,
    meshes: &mut Vec<Mesh>,
    entities: &mut Vec<Entity>,
) -> Result<(), String> {
    for selected in expression::evaluate(state, &layer.expression)? {
        let Some(mol) = state.get_mol(selected.mol_type, selected.mol_i) else {
            continue;
        };
        let common = mol.common();
        if !common.visible {
            continue;
        }
        let mut mask = vec![false; common.atoms.len()];
        for &a in &selected.atoms {
            mask[a] = true;
        }
        let position = |a: usize| -> Vec3 {
            common
                .atom_posits
                .get(a)
                .copied()
                .unwrap_or(common.atoms[a].posit)
                .into()
        };

        if layer.view == MoleculeView::Ribbon {
            let MolGenericRef::Peptide(pep) = mol else {
                return Err("Cartoon layers require a protein selection".into());
            };
            // Keep source indices intact. Clearing roles outside the selected residues makes
            // the ribbon builder omit their backbone frames and break at selection boundaries.
            let residues: std::collections::HashSet<_> = selected
                .atoms
                .iter()
                .filter_map(|&a| common.atoms[a].residue)
                .collect();
            let mut atoms = common.atoms.clone();
            for atom in &mut atoms {
                if !atom.residue.is_some_and(|r| residues.contains(&r)) {
                    atom.role = None;
                    atom.type_in_res = None;
                }
            }
            let mut chains = pep.chains.clone();
            for chain in &mut chains {
                chain.visible = true;
            }
            let mesh = build_ribbon_mesh(
                &pep.secondary_structure,
                &atoms,
                &common.bonds,
                &common.atom_posits,
                &pep.residues,
                &chains,
                ResColoring::Chain,
                ViewSelLevel::Residue,
                &Selection::None,
                &[],
                pep.sifts_mapping.as_deref(),
            );
            mesh_entity(mesh, layer, meshes, entities, mesh_start);
            continue;
        }

        if matches!(layer.view, MoleculeView::Surface | MoleculeView::Dots) {
            let atoms: Vec<_> = selected
                .atoms
                .iter()
                .map(|&a| (position(a), common.atoms[a].element.vdw_radius()))
                .collect();
            let precision = state.to_save.sa_surface_precision.max(0.3);
            // Avoid allocating an enormous voxel grid for a distant, disconnected selection.
            let mut low = atoms[0].0;
            let mut high = low;
            for (p, _) in &atoms {
                low.x = low.x.min(p.x);
                low.y = low.y.min(p.y);
                low.z = low.z.min(p.z);
                high.x = high.x.max(p.x);
                high.y = high.y.max(p.y);
                high.z = high.z.max(p.z);
            }
            let extent = high - low + Vec3::splat(12.);
            if (extent.x / precision) as f64
                * (extent.y / precision) as f64
                * (extent.z / precision) as f64
                > 30_000_000.
            {
                return Err("Surface selection spans too much space; use a smaller selection or coarser surface precision".into());
            }
            let mesh = make_sas_mesh(&atoms, SOLVENT_RAD, precision);
            if layer.view == MoleculeView::Dots {
                for vertex in mesh.vertices {
                    let mut entity = Entity::new(
                        MESH_BALL_STICK_SPHERE,
                        Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]),
                        Quaternion::new_identity(),
                        0.08 * layer.size,
                        (0.6, 0.75, 0.9),
                        ATOM_SHININESS,
                    );
                    style_entity(&mut entity, layer);
                    entities.push(entity);
                }
            } else {
                mesh_entity(mesh, layer, meshes, entities, mesh_start);
            }
            continue;
        }

        if layer.view == MoleculeView::Backbone {
            for (a, atom) in common.atoms.iter().enumerate() {
                mask[a] &= atom.is_backbone();
            }
        }
        if matches!(
            layer.view,
            MoleculeView::BallAndStick | MoleculeView::SpaceFill
        ) {
            for &a in &selected.atoms {
                let atom = &common.atoms[a];
                let (mesh, radius) = if layer.view == MoleculeView::SpaceFill {
                    (MESH_SPACEFILL_SPHERE, atom.element.vdw_radius())
                } else {
                    (MESH_BALL_STICK_SPHERE, BALL_STICK_RADIUS)
                };
                let mut entity = Entity::new(
                    mesh,
                    position(a),
                    Quaternion::new_identity(),
                    radius * layer.size,
                    atom.element.color(),
                    ATOM_SHININESS,
                );
                style_entity(&mut entity, layer);
                entities.push(entity);
            }
        }
        if layer.view == MoleculeView::SpaceFill {
            continue;
        }
        let mut neighbors = vec![Vec::new(); common.atoms.len()];
        for bond in &common.bonds {
            if bond.atom_0 < neighbors.len() && bond.atom_1 < neighbors.len() {
                neighbors[bond.atom_0].push(bond.atom_1);
                neighbors[bond.atom_1].push(bond.atom_0);
            }
        }
        for bond in &common.bonds {
            let (a, b) = (bond.atom_0, bond.atom_1);
            if !mask.get(a).copied().unwrap_or(false)
                || !mask.get(b).copied().unwrap_or(false)
                || (position(a) - position(b)).magnitude_squared() < 1e-10
            {
                continue;
            }
            let neighbor = neighbors[a]
                .iter()
                .copied()
                .find(|&i| i != b)
                .map(position)
                .unwrap_or(position(a));
            let color = |i: usize| {
                layer
                    .color
                    .map(|[r, g, b]| (r, g, b))
                    .unwrap_or(common.atoms[i].element.color())
            };
            let mut bonds = bond_entities(
                position(a),
                position(b),
                color(a),
                color(b),
                bond.bond_type,
                selected.mol_type,
                &common.ident,
                false,
                layer.view != MoleculeView::BallAndStick,
                (neighbor, false),
                false,
                common.atoms[a].element == Element::Hydrogen
                    || common.atoms[b].element == Element::Hydrogen,
            );
            for entity in &mut bonds {
                style_entity(entity, layer);
                if let Some(scale) = &mut entity.scale_partial {
                    scale.x *= layer.size;
                    scale.z *= layer.size;
                } else {
                    entity.scale *= layer.size;
                }
            }
            entities.extend(bonds);
        }
    }
    Ok(())
}

/// Called after normal drawing, including input-driven molecule transformations. Geometry is
/// rebuilt only when definitions or molecule data change, not when the camera moves.
pub fn update(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let enabled = state.volatile.operating_mode == OperatingMode::Primary
        && !state.volatile.md_local.draw_md_mols;
    let has_content = enabled
        && (!state.to_save.representations.is_empty()
            || matches!(state.ui.selection, Selection::Atoms(_)));
    let fingerprint = if has_content { fingerprint(state) } else { 0 };
    let pocket_end = state
        .pockets
        .iter()
        .map(|p| p.mesh_i_rel + 1)
        .chain(
            state
                .ligands
                .iter()
                .filter_map(|m| m.pharmacophore.pocket.as_ref().map(|p| p.mesh_i_rel + 1)),
        )
        .max()
        .unwrap_or(0)
        .max(state.pockets.len() + state.ligands.len());
    let mesh_start = MESH_POCKET_START + pocket_end;
    let cache = &state.volatile.representations;
    let geometry_changed = cache.fingerprint != Some(fingerprint)
        || cache.mesh_slots.first().is_some_and(|&i| i != mesh_start);
    if !geometry_changed {
        if cache.selection != state.ui.selection {
            update_highlight(state, scene, updates, enabled);
        }
        return;
    }

    let mut meshes = Vec::new();
    let mut entities = Vec::new();
    let mut errors = Vec::new();
    if has_content {
        for layer in &state.to_save.representations {
            if layer.visible && layer.opacity > 0. {
                let (mesh_count, entity_count) = (meshes.len(), entities.len());
                if let Err(error) = draw_layer(state, layer, mesh_start, &mut meshes, &mut entities)
                {
                    meshes.truncate(mesh_count);
                    entities.truncate(entity_count);
                    errors.push(format!("{}: {error}", layer.name));
                }
            }
        }
    }

    scene
        .entities
        .retain(|e| e.class != EntityClass::Representation as u32);
    scene.entities.extend(entities);
    let cache = &mut state.volatile.representations;
    let had_meshes = !cache.mesh_slots.is_empty();
    for &slot in &cache.mesh_slots {
        if slot >= mesh_start && slot < scene.meshes.len() {
            scene.meshes[slot] = Mesh::default();
        }
    }
    cache.mesh_slots.clear();
    if !meshes.is_empty() {
        scene.meshes.resize_with(
            scene.meshes.len().max(mesh_start + meshes.len()),
            Mesh::default,
        );
        for (i, mesh) in meshes.into_iter().enumerate() {
            scene.meshes[mesh_start + i] = mesh;
            cache.mesh_slots.push(mesh_start + i);
        }
    }
    updates.meshes |= had_meshes || !cache.mesh_slots.is_empty();
    cache.fingerprint = Some(fingerprint);
    cache.errors = errors;
    clear_mol_entity_indices(state, None);
    updates.entities = EntityUpdate::All;
    update_highlight(state, scene, updates, enabled);
}

fn update_highlight(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    enabled: bool,
) {
    scene
        .entities
        .retain(|e| e.class != EntityClass::SelectionHighlight as u32);
    if enabled && let Selection::Atoms(selected) = &state.ui.selection {
        for selected in selected {
            let Some(mol) = state.get_mol(selected.mol_type, selected.mol_i) else {
                continue;
            };
            if !mol.common().visible {
                continue;
            }
            for &i in &selected.atoms {
                let Some(point) = mol.common().atom_posits.get(i) else {
                    continue;
                };
                let mut entity = Entity::new(
                    MESH_BALL_STICK_SPHERE,
                    (*point).into(),
                    Quaternion::new_identity(),
                    0.35,
                    COLOR_SELECTED,
                    ATOM_SHININESS,
                );
                entity.class = EntityClass::SelectionHighlight as u32;
                scene.entities.push(entity);
            }
        }
    }
    state.volatile.representations.selection = state.ui.selection.clone();
    clear_mol_entity_indices(state, None);
    updates.entities = EntityUpdate::All;
}
