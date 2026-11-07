//! Handles logic for moving and rotating molecules from user inputs.
//! This is for both primary mode, and the mol editor. In the latter, it
//! can move individual atoms, and rotate parts of molecules around bonds.

use std::hash::{DefaultHasher, Hash, Hasher};

use graphics::{
    ControlScheme, EngineUpdates, FWD_VEC, Mesh, RIGHT_VEC, Scene, UP_VEC, event::MouseScrollDelta,
    winit::keyboard::KeyCode,
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::{Quaternion as QuaternionF64, Vec3 as Vec3F64},
};
use mol_defs::molecules::{MolType, common::MoleculeCommon};
use na_seq::Element;

use crate::{
    drawing::{EntityClass, MoleculeView, effective_mol_view_peptide},
    inputs::{SENS_MOL_ROT_MOUSE, SENS_MOL_ROT_SCROLL},
    pocket_render::PocketRender,
    selection::Selection,
    state::{OperatingMode, State},
    util::RedrawFlags,
};

#[derive(Clone, Copy)]
enum PeptideMeshDelta {
    Translate(Vec3),
    Rotate { rotation: Quaternion, pivot: Vec3 },
}

/// Accumulated world-space transform from a cached peptide mesh to the current atom positions.
///
/// Ribbon and surface meshes are expensive, but peptide manipulation is rigid. Keeping this
/// transform on their render entities avoids rebuilding and re-uploading every vertex for each
/// mouse event.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PeptideMeshTransform {
    pub(crate) rotation: Quaternion,
    pub(crate) translation: Vec3,
}

impl Default for PeptideMeshTransform {
    fn default() -> Self {
        Self {
            rotation: Quaternion::new_identity(),
            translation: Vec3::new_zero(),
        }
    }
}

impl PeptideMeshTransform {
    fn apply_delta(&mut self, delta: PeptideMeshDelta) {
        match delta {
            PeptideMeshDelta::Translate(translation) => {
                self.translation += translation;
            }
            PeptideMeshDelta::Rotate { rotation, pivot } => {
                self.rotation = (rotation * self.rotation).to_normalized();
                self.translation = rotation.rotate_vec(self.translation - pivot) + pivot;
            }
        }
    }

    pub(crate) fn transform_point(self, point: Vec3) -> Vec3 {
        self.rotation.rotate_vec(point) + self.translation
    }
}

/// Transform a mesh copy for CPU-side operations such as surface coloring. Manipulation itself
/// uses the much cheaper entity transform and does not mutate or upload the cached mesh.
pub(crate) fn transform_peptide_mesh(mesh: &mut Mesh, transform: PeptideMeshTransform) -> bool {
    if mesh.vertices.is_empty() {
        return false;
    }

    for vertex in &mut mesh.vertices {
        let position = Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
        vertex.position = transform.transform_point(position).to_arr();
        vertex.normal = transform.rotation.rotate_vec(vertex.normal);
        vertex.tangent = transform.rotation.rotate_vec(vertex.tangent);
        vertex.bitangent = transform.rotation.rotate_vec(vertex.bitangent);
    }

    true
}

fn record_cached_peptide_mesh_transform(
    state: &mut State,
    ss_mesh_created: bool,
    sas_mesh_created: bool,
    delta: PeptideMeshDelta,
) {
    let manip = &mut state.volatile.mol_manip;
    manip.peptide_mesh_manip_pending = true;

    if ss_mesh_created {
        manip.ribbon_mesh_transform.apply_delta(delta);
    }
    if sas_mesh_created {
        manip.surface_mesh_transform.apply_delta(delta);
    }
}

/// Update only the small instance record for the currently visible cached mesh. This avoids
/// `EngineUpdates::meshes`, which recreates the combined GPU vertex/index buffers.
fn update_visible_cached_peptide_mesh_entity(
    state: &State,
    scene: &mut Scene,
    ss_mesh_created: bool,
    sas_mesh_created: bool,
    updates: &mut EngineUpdates,
) {
    let Some(view) = state
        .peptide_for_tools_i()
        .map(|i| effective_mol_view_peptide(state, i))
    else {
        return;
    };

    let (created, class, transform) = match view {
        MoleculeView::Ribbon => (
            ss_mesh_created,
            EntityClass::SecondaryStructure,
            state.volatile.mol_manip.ribbon_mesh_transform,
        ),
        MoleculeView::Surface => (
            sas_mesh_created,
            EntityClass::SaSurface,
            state.volatile.mol_manip.surface_mesh_transform,
        ),
        _ => return,
    };

    if !created {
        return;
    }

    let mut changed = false;
    for entity in &mut scene.entities {
        if entity.class == class as u32
            && (class != EntityClass::SecondaryStructure
                || entity.mesh == crate::render::MESH_SECONDARY_STRUCTURE)
        {
            entity.position = transform.translation;
            entity.orientation = transform.rotation;
            entity.pivot = None;
            changed = true;
        }
    }

    if changed {
        updates.entities.push_class(class as u32);
    }
}

fn handle_cached_peptide_mesh_delta(
    state: &mut State,
    scene: &mut Scene,
    ss_mesh_created: bool,
    sas_mesh_created: bool,
    delta: PeptideMeshDelta,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let hide_dots = !state.volatile.mol_manip.peptide_mesh_manip_pending
        && state
            .peptide_for_tools_i()
            .is_some_and(|i| effective_mol_view_peptide(state, i) == MoleculeView::Dots);

    record_cached_peptide_mesh_transform(state, ss_mesh_created, sas_mesh_created, delta);
    update_visible_cached_peptide_mesh_entity(
        state,
        scene,
        ss_mesh_created,
        sas_mesh_created,
        updates,
    );

    if hide_dots {
        redraw.peptide = true;
    }
}

/// Complete a drag/tool interaction. Mesh transforms remain on the cheap instance path; a peptide
/// redraw is needed primarily to regenerate deferred Surface Dots at their final positions.
pub fn finish_peptide_mesh_manipulation(state: &mut State, redraw: &mut RedrawFlags) {
    if state.volatile.mol_manip.peptide_mesh_manip_pending {
        state.volatile.mol_manip.peptide_mesh_manip_pending = false;
        redraw.peptide = true;
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub enum ManipMode {
    #[default]
    None,
    Move((MolType, usize)),
    Rotate((MolType, usize)),
}

impl ManipMode {
    pub(crate) fn target(self) -> Option<(MolType, usize)> {
        match self {
            Self::Move(target) | Self::Rotate(target) => Some(target),
            Self::None => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub(crate) const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    pub(crate) fn vector(self) -> Vec3 {
        match self {
            Self::X => RIGHT_VEC,
            Self::Y => UP_VEC,
            Self::Z => FWD_VEC,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    pub(crate) fn plane_axes(self) -> (Vec3, Vec3) {
        match self {
            Self::X => (UP_VEC, FWD_VEC),
            Self::Y => (FWD_VEC, RIGHT_VEC),
            Self::Z => (RIGHT_VEC, UP_VEC),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Constraint {
    #[default]
    Free,
    Axis(Axis),
    Plane(Axis),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Handle {
    MoveAxis(Axis),
    MovePlane(Axis),
    Rotate(Axis),
    Camera(Axis, bool),
}

#[derive(Clone, Debug)]
pub(crate) struct Pose {
    pub(crate) mode: ManipMode,
    pub(crate) operating_mode: OperatingMode,
    pub(crate) positions: Vec<Vec3F64>,
    atom_positions: Vec<Vec3F64>,
    topology: u64,
    object_key: String,
    gizmo_orientation: Quaternion,
}

#[derive(Clone, Debug)]
pub(crate) struct TransformValues {
    pub(crate) translation: Vec3F64,
    pub(crate) axis: Vec3F64,
    pub(crate) angle: f64,
    pub(crate) rotation: QuaternionF64,
}

impl Default for TransformValues {
    fn default() -> Self {
        Self {
            translation: Vec3F64::new_zero(),
            axis: Vec3F64::new(0., 0., 1.),
            angle: 0.,
            rotation: QuaternionF64::new_identity(),
        }
    }
}

#[derive(Debug)]
struct TransformSession {
    before: Pose,
    values: TransformValues,
    raw_translation: Vec3F64,
    raw_angle: f64,
    raw_rotation: QuaternionF64,
}

#[derive(Debug)]
pub struct MolManip {
    pub mode: ManipMode,
    pub pivot: Option<Vec3>,
    pub(crate) gizmo_orientation: Quaternion,
    pub(crate) constraint: Constraint,
    pub(crate) hovered: Option<Handle>,
    pub(crate) dragged: Option<Handle>,
    pub(crate) ring_tangent: Option<(f32, f32)>,
    pub(crate) numeric: String,
    pub(crate) snap: bool,
    pub(crate) pixels_per_point: f32,
    session: Option<TransformSession>,
    pub(crate) ribbon_mesh_transform: PeptideMeshTransform,
    pub(crate) surface_mesh_transform: PeptideMeshTransform,
    pub(crate) peptide_mesh_manip_pending: bool,
}

impl Default for MolManip {
    fn default() -> Self {
        Self {
            mode: ManipMode::None,
            pivot: None,
            gizmo_orientation: Quaternion::new_identity(),
            constraint: Constraint::Free,
            hovered: None,
            dragged: None,
            ring_tangent: None,
            numeric: String::new(),
            snap: false,
            pixels_per_point: 1.,
            session: None,
            ribbon_mesh_transform: PeptideMeshTransform::default(),
            surface_mesh_transform: PeptideMeshTransform::default(),
            peptide_mesh_manip_pending: false,
        }
    }
}

fn target_mol(state: &State, mode: ManipMode, op: OperatingMode) -> Option<&MoleculeCommon> {
    let (kind, i) = mode.target()?;

    match op {
        OperatingMode::Primary => match kind {
            MolType::Peptide => state.peptides.get(i).map(|m| &m.common),
            MolType::Ligand => state.ligands.get(i).map(|m| &m.common),
            MolType::NucleicAcid => state.nucleic_acids.get(i).map(|m| &m.common),
            MolType::Lipid => state.lipids.get(i).map(|m| &m.common),
            MolType::Pocket => state.pockets.get(i).map(|m| &m.common),
            MolType::Water => None,
        },
        OperatingMode::MolEditor => match kind {
            MolType::Ligand => Some(&state.mol_editor.mol.common),
            MolType::Pocket => state
                .mol_editor
                .mol
                .pharmacophore
                .pocket
                .as_ref()
                .map(|p| &p.common),
            _ => None,
        },
        OperatingMode::ProteinEditor => None,
    }
}

fn target_mol_mut(
    state: &mut State,
    mode: ManipMode,
    op: OperatingMode,
) -> Option<&mut MoleculeCommon> {
    let (kind, i) = mode.target()?;

    match op {
        OperatingMode::Primary => match kind {
            MolType::Peptide => state.peptides.get_mut(i).map(|m| &mut m.common),
            MolType::Ligand => state.ligands.get_mut(i).map(|m| &mut m.common),
            MolType::NucleicAcid => state.nucleic_acids.get_mut(i).map(|m| &mut m.common),
            MolType::Lipid => state.lipids.get_mut(i).map(|m| &mut m.common),
            MolType::Pocket => state.pockets.get_mut(i).map(|m| &mut m.common),
            MolType::Water => None,
        },
        OperatingMode::MolEditor => match kind {
            MolType::Ligand => Some(&mut state.mol_editor.mol.common),
            MolType::Pocket => state
                .mol_editor
                .mol
                .pharmacophore
                .pocket
                .as_mut()
                .map(|p| &mut p.common),
            _ => None,
        },
        OperatingMode::ProteinEditor => None,
    }
}

impl Pose {
    fn topology(mol: &MoleculeCommon) -> u64 {
        let mut hash = DefaultHasher::new();
        for atom in &mol.atoms {
            atom.serial_number.hash(&mut hash);
            atom.element.hash(&mut hash);
            atom.residue.hash(&mut hash);
        }
        for bond in &mol.bonds {
            bond.atom_0.hash(&mut hash);
            bond.atom_1.hash(&mut hash);
        }

        hash.finish()
    }

    pub(crate) fn capture(state: &State, mode: ManipMode) -> Option<Self> {
        let op = state.volatile.operating_mode;
        let mol = target_mol(state, mode, op)?;
        if mol.atom_posits.is_empty() {
            return None;
        }

        Some(Self {
            mode,
            operating_mode: op,
            positions: mol.atom_posits.clone(),
            atom_positions: if op == OperatingMode::MolEditor {
                mol.atoms.iter().map(|a| a.posit).collect()
            } else {
                Vec::new()
            },
            topology: Self::topology(mol),
            object_key: mol.ident.clone(),
            gizmo_orientation: state.volatile.mol_manip.gizmo_orientation,
        })
    }

    pub(crate) fn matches(&self, state: &State) -> bool {
        state.volatile.operating_mode == self.operating_mode
            && target_mol(state, self.mode, self.operating_mode).is_some_and(|m| {
                m.ident == self.object_key
                    && m.atom_posits == self.positions
                    && Self::topology(m) == self.topology
            })
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        (self.positions.len() + self.atom_positions.len()) * std::mem::size_of::<Vec3F64>()
    }

    fn pivot(&self) -> Vec3F64 {
        self.positions
            .iter()
            .copied()
            .fold(Vec3F64::new_zero(), |sum, p| sum + p)
            / self.positions.len() as f64
    }
}

fn ensure_session(state: &mut State) -> bool {
    let mode = state.volatile.mol_manip.mode;
    if mode == ManipMode::None {
        state.volatile.mol_manip.session = None;

        return false;
    }
    if state.volatile.mol_manip.session.as_ref().is_some_and(|s| {
        s.before.mode == mode && s.before.operating_mode == state.volatile.operating_mode
    }) {
        return true;
    }

    let Some(before) = Pose::capture(state, mode) else {
        return false;
    };
    state.volatile.mol_manip.session = Some(TransformSession {
        before,
        values: TransformValues::default(),
        raw_translation: Vec3F64::new_zero(),
        raw_angle: 0.,
        raw_rotation: QuaternionF64::new_identity(),
    });

    true
}

fn update_pocket_and_bonds(state: &mut State, mode: ManipMode) {
    if let Some((MolType::Pocket, i)) = mode.target() {
        match state.volatile.operating_mode {
            OperatingMode::Primary => {
                if let Some(p) = state.pockets.get_mut(i) {
                    p.rebuild_spheres();
                }
            }
            OperatingMode::MolEditor => {
                if let Some(p) = &mut state.mol_editor.mol.pharmacophore.pocket {
                    p.rebuild_spheres();
                }
            }
            _ => (),
        }
    }
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        state.mol_editor.update_h_bonds();
    }
}

pub(crate) fn restore_pose(state: &mut State, pose: &Pose, redraw: &mut RedrawFlags) {
    if let Some(mol) = target_mol_mut(state, pose.mode, pose.operating_mode) {
        if mol.ident != pose.object_key || Pose::topology(mol) != pose.topology {
            return;
        }

        mol.atom_posits.clone_from(&pose.positions);
        for (atom, position) in mol.atoms.iter_mut().zip(&pose.atom_positions) {
            atom.posit = *position;
        }
    }
    state.volatile.mol_manip.gizmo_orientation = pose.gizmo_orientation;
    if matches!(pose.mode.target(), Some((MolType::Peptide, _))) {
        state.volatile.mol_manip.ribbon_mesh_transform = PeptideMeshTransform::default();
        state.volatile.mol_manip.surface_mesh_transform = PeptideMeshTransform::default();
        state.volatile.flags.ss_mesh_created = false;
        state.volatile.flags.sas_mesh_created = false;
        state.volatile.flags.update_ss_mesh = true;
        state.volatile.flags.update_sas_mesh = matches!(
            state.ui.mol_view_peptide,
            MoleculeView::Surface | MoleculeView::Dots
        );
        state.volatile.thread_receivers.peptide_mesh_coloring = None;
    }
    update_pocket_and_bonds(state, pose.mode);
    if let Some((kind, _)) = pose.mode.target() {
        redraw.set(kind);
    }
}

pub(crate) fn apply_values(state: &mut State, before: &Pose, values: &TransformValues) -> bool {
    let Some(mol) = target_mol_mut(state, before.mode, before.operating_mode) else {
        return false;
    };
    if mol.ident != before.object_key
        || mol.atom_posits.len() != before.positions.len()
        || Pose::topology(mol) != before.topology
    {
        return false;
    }

    mol.atom_posits.clone_from(&before.positions);
    let (kind, index) = before.mode.target().unwrap();
    match before.mode {
        ManipMode::Move(_) => {
            if before.operating_mode == OperatingMode::MolEditor && kind == MolType::Ligand {
                if index >= mol.atom_posits.len() {
                    return false;
                }

                mol.atom_posits[index] += values.translation;
                for &i in &mol.adjacency_list[index] {
                    if mol.atoms[i].element == Element::Hydrogen {
                        mol.atom_posits[i] += values.translation;
                    }
                }
            } else {
                for p in &mut mol.atom_posits {
                    *p += values.translation;
                }
            }
        }
        ManipMode::Rotate(_) => {
            if before.operating_mode == OperatingMode::MolEditor && kind == MolType::Ligand {
                mol.rotate_around_bond(index, values.angle, None);
            } else {
                let pivot = before.pivot();
                for p in &mut mol.atom_posits {
                    *p = pivot + values.rotation.rotate_vec(*p - pivot);
                }
            }
        }
        ManipMode::None => return false,
    }
    if before.operating_mode == OperatingMode::MolEditor {
        for (atom, position) in mol.atoms.iter_mut().zip(&mol.atom_posits) {
            atom.posit = *position;
        }
    }
    state.volatile.mol_manip.gizmo_orientation =
        (Quaternion::from(values.rotation) * before.gizmo_orientation).to_normalized();

    true
}

fn snapping(state: &State) -> bool {
    state.volatile.mol_manip.snap || state.volatile.key_modifiers.state().control_key()
}

fn fine_factor(state: &State) -> f64 {
    if state.volatile.key_modifiers.state().shift_key() {
        0.1
    } else {
        1.
    }
}

fn quantize(value: f64, step: f64, snap: bool) -> f64 {
    if snap {
        (value / step).round() * step
    } else {
        value
    }
}

fn preview(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let snap = snapping(state);
    let constraint = state.volatile.mol_manip.constraint;
    let numeric = state
        .volatile
        .mol_manip
        .numeric
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && v.abs() <= 1.0e9);
    let bond = state.volatile.operating_mode == OperatingMode::MolEditor
        && matches!(
            state.volatile.mol_manip.mode,
            ManipMode::Rotate((MolType::Ligand, _))
        );
    let Some(mut session) = state.volatile.mol_manip.session.take() else {
        return;
    };
    let old = session.values.clone();
    let values = &mut session.values;

    match session.before.mode {
        ManipMode::Move(_) => {
            let mut movement = session.raw_translation;
            if let Some(number) = numeric {
                let direction: Vec3F64 = match constraint {
                    Constraint::Axis(axis) => axis.vector().into(),
                    _ => RIGHT_VEC.into(),
                };
                movement = direction * number;
            } else {
                movement = Vec3F64::new(
                    quantize(movement.x, 1., snap),
                    quantize(movement.y, 1., snap),
                    quantize(movement.z, 1., snap),
                );
            }
            values.translation = match constraint {
                Constraint::Free => movement,
                Constraint::Axis(axis) => {
                    let axis: Vec3F64 = axis.vector().into();
                    axis * movement.dot(axis)
                }
                Constraint::Plane(axis) => {
                    let axis: Vec3F64 = axis.vector().into();
                    movement - axis * movement.dot(axis)
                }
            };
        }
        ManipMode::Rotate(_) => {
            let axis: Vec3F64 = match constraint {
                Constraint::Axis(axis) | Constraint::Plane(axis) => axis.vector().into(),
                Constraint::Free => scene.camera.orientation.rotate_vec(FWD_VEC).into(),
            };
            if constraint != Constraint::Free || numeric.is_some() || bond {
                values.angle = numeric
                    .map(f64::to_radians)
                    .unwrap_or_else(|| quantize(session.raw_angle, 5_f64.to_radians(), snap));
                values.axis = axis;
                values.rotation = QuaternionF64::from_axis_angle(axis, values.angle);
            } else {
                let q = session.raw_rotation.to_normalized();
                let sin_half = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
                values.axis = if sin_half > 1.0e-12 {
                    Vec3F64::new(q.x, q.y, q.z) / sin_half
                } else {
                    axis
                };
                values.angle = quantize(2. * sin_half.atan2(q.w), 5_f64.to_radians(), snap);
                values.rotation = QuaternionF64::from_axis_angle(values.axis, values.angle);
            }
        }
        ManipMode::None => (),
    }
    if apply_values(state, &session.before, values) {
        let (kind, i) = session.before.mode.target().unwrap();
        if kind == MolType::Peptide && state.peptide_for_tools_i() == Some(i) {
            let delta = match session.before.mode {
                ManipMode::Move(_) => {
                    PeptideMeshDelta::Translate((values.translation - old.translation).into())
                }
                _ => PeptideMeshDelta::Rotate {
                    rotation: (values.rotation * old.rotation.inverse()).into(),
                    pivot: session.before.pivot().into(),
                },
            };
            handle_cached_peptide_mesh_delta(
                state,
                scene,
                state.volatile.flags.ss_mesh_created,
                state.volatile.flags.sas_mesh_created,
                delta,
                redraw,
                updates,
            );
        }
        state.volatile.mol_manip.gizmo_orientation =
            (Quaternion::from(values.rotation) * session.before.gizmo_orientation).to_normalized();
        redraw.set(kind);
        update_pocket_and_bonds(state, session.before.mode);
    }
    state.volatile.mol_manip.session = Some(session);
}

fn plane_hit(scene: &Scene, cursor: (f32, f32), pivot: Vec3, normal: Vec3) -> Option<Vec3> {
    let (origin, point) = scene.screen_to_render(cursor);
    let direction = point - origin;
    let denominator = normal.dot(direction);
    if denominator.abs() < 1.0e-6 {
        return None;
    }

    let t = normal.dot(pivot - origin) / denominator;

    (t >= 0.).then_some(origin + direction * t)
}

pub(crate) fn begin_drag(state: &mut State, scene: &Scene, handle: Option<Handle>) {
    if state.volatile.mol_manip.mode == ManipMode::None || !ensure_session(state) {
        return;
    }

    let tangent = match (handle, state.ui.cursor_pos) {
        (Some(Handle::Rotate(axis)), Some(cursor)) => {
            crate::gizmo::ring_tangent_at(state, scene, axis, cursor)
        }
        _ => None,
    };
    let manip = &mut state.volatile.mol_manip;
    manip.ring_tangent = tangent;
    manip.dragged = handle;
    manip.numeric.clear();
    match handle {
        Some(Handle::MoveAxis(axis) | Handle::Rotate(axis)) => {
            manip.constraint = Constraint::Axis(axis)
        }
        Some(Handle::MovePlane(axis)) => manip.constraint = Constraint::Plane(axis),
        _ => (),
    }
    let before = &manip.session.as_ref().unwrap().before;
    manip.pivot = Some(before.pivot().into());
}

pub fn handle_mol_manip_in_plane(
    state: &mut State,
    scene: &mut Scene,
    delta: (f64, f64),
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    if !ensure_session(state) || !state.volatile.mol_manip.numeric.is_empty() {
        return;
    }
    let Some(cursor) = state.ui.cursor_pos else {
        return;
    };
    let previous = (cursor.0 - delta.0 as f32, cursor.1 - delta.1 as f32);
    let factor = fine_factor(state);
    let constraint = state.volatile.mol_manip.constraint;
    let mode = state.volatile.mol_manip.mode;

    match mode {
        ManipMode::Move(_) => {
            let pivot = state.volatile.mol_manip.pivot.unwrap_or_else(|| {
                state
                    .volatile
                    .mol_manip
                    .session
                    .as_ref()
                    .unwrap()
                    .before
                    .pivot()
                    .into()
            });
            let view = scene.camera.orientation.rotate_vec(FWD_VEC);
            let normal = match constraint {
                Constraint::Plane(axis) => axis.vector(),
                Constraint::Axis(axis) => {
                    let a = axis.vector();
                    let n = view - a * view.dot(a);
                    if n.magnitude_squared() < 0.01 {
                        view
                    } else {
                        n.to_normalized()
                    }
                }
                Constraint::Free => view,
            };
            let (Some(start), Some(end)) = (
                plane_hit(scene, previous, pivot, normal),
                plane_hit(scene, cursor, pivot, normal),
            ) else {
                return;
            };
            let mut movement = end - start;
            if let Constraint::Axis(axis) = constraint {
                let a = axis.vector();
                if a.dot(view).abs() > 0.99 {
                    let distance = (pivot - scene.camera.position).magnitude();
                    movement = a * (-delta.1 as f32 * distance / scene.window_size.1.max(1.));
                } else {
                    movement = a * movement.dot(a);
                }
            }
            state
                .volatile
                .mol_manip
                .session
                .as_mut()
                .unwrap()
                .raw_translation += Vec3F64::from(movement) * factor;
        }
        ManipMode::Rotate(_) => {
            let angle = if let Some(Handle::Rotate(axis)) = state.volatile.mol_manip.dragged {
                crate::gizmo::ring_drag_delta(state, scene, axis, previous, cursor) as f64
            } else {
                -delta.0 * SENS_MOL_ROT_MOUSE as f64 * scene.camera.screen_x_sign() as f64
            } * factor;
            let session = state.volatile.mol_manip.session.as_mut().unwrap();
            session.raw_angle += angle;
            let right = Vec3F64::from(scene.camera.orientation.rotate_vec(RIGHT_VEC));
            let up = Vec3F64::from(scene.camera.orientation.rotate_vec(UP_VEC));
            let x = QuaternionF64::from_axis_angle(
                right,
                -delta.1 * SENS_MOL_ROT_MOUSE as f64 * factor,
            );
            let y = QuaternionF64::from_axis_angle(up, angle);
            session.raw_rotation = (y * x * session.raw_rotation).to_normalized();
        }
        ManipMode::None => return,
    }
    preview(state, scene, redraw, updates);
}

pub fn handle_mol_manip_in_out(
    state: &mut State,
    scene: &mut Scene,
    delta: MouseScrollDelta,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    if !ensure_session(state) || !state.volatile.mol_manip.numeric.is_empty() {
        return;
    }

    let scroll = match delta {
        MouseScrollDelta::LineDelta(_, y) => y as f64,
        MouseScrollDelta::PixelDelta(p) => p.y / 120.,
    } * fine_factor(state);
    let session = state.volatile.mol_manip.session.as_mut().unwrap();
    let fwd: Vec3F64 = scene.camera.orientation.rotate_vec(FWD_VEC).into();
    match session.before.mode {
        ManipMode::Move(_) => {
            let direction = match state.volatile.mol_manip.constraint {
                Constraint::Axis(axis) => Vec3F64::from(axis.vector()),
                _ => fwd,
            };
            let distance =
                (session.before.pivot() - Vec3F64::from(scene.camera.position)).magnitude();
            let step = state.to_save.control_settings.mol_move_sens as f64 / 1_000. * distance;
            session.raw_translation += direction * (scroll * step);
        }
        ManipMode::Rotate(_) => {
            let angle = scroll * SENS_MOL_ROT_SCROLL as f64 * scene.camera.screen_x_sign() as f64;
            session.raw_angle += angle;
            session.raw_rotation =
                (QuaternionF64::from_axis_angle(fwd, angle) * session.raw_rotation).to_normalized();
        }
        ManipMode::None => return,
    }
    preview(state, scene, redraw, updates);
}

pub(crate) fn finish_drag(state: &mut State, redraw: &mut RedrawFlags) {
    if let Some(session) = state.volatile.mol_manip.session.take()
        && session.before.mode == state.volatile.mol_manip.mode
        && session.before.operating_mode == state.volatile.operating_mode
        && let Some(after) = Pose::capture(state, session.before.mode)
        && after.positions != session.before.positions
    {
        state
            .volatile
            .history
            .push(crate::edit_history::Operation::Transform {
                before: session.before,
                after,
                values: session.values,
            });
    }
    state.volatile.mol_manip.pivot = None;
    state.volatile.mol_manip.dragged = None;
    state.volatile.mol_manip.ring_tangent = None;
    finish_peptide_mesh_manipulation(state, redraw);
}

fn finish_pocket(
    state: &mut State,
    scene: &mut Scene,
    mode: ManipMode,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let Some((MolType::Pocket, i)) = mode.target() else {
        return;
    };
    let pocket = if state.volatile.operating_mode == OperatingMode::MolEditor {
        state.mol_editor.mol.pharmacophore.pocket.as_mut()
    } else {
        state.pockets.get_mut(i)
    };
    if let Some(pocket) = pocket {
        pocket.reset_post_manip(&mut scene.meshes, state.ui.mesh_coloring, updates);
        redraw.pocket = true;
    }
}

pub(crate) fn refresh_after_history(
    state: &mut State,
    scene: &mut Scene,
    mode: ManipMode,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    finish_pocket(state, scene, mode, redraw, updates);
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        crate::mol_editor::sync_md(state);
    }
}

pub(crate) fn end_transform(
    state: &mut State,
    scene: &mut Scene,
    cancel: bool,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let mode = state.volatile.mol_manip.mode;
    if cancel {
        if let Some(session) = state.volatile.mol_manip.session.take()
            && !session.before.matches(state)
        {
            restore_pose(state, &session.before, redraw);
        }
    } else {
        finish_drag(state, redraw);
    }
    finish_peptide_mesh_manipulation(state, redraw);
    state.volatile.mol_manip.mode = ManipMode::None;
    state.volatile.mol_manip.pivot = None;
    state.volatile.mol_manip.dragged = None;
    state.volatile.mol_manip.constraint = Constraint::Free;
    state.volatile.mol_manip.numeric.clear();
    state.ui.left_click_down = false;
    scene.input_settings.control_scheme = state.volatile.control_scheme_prev;
    finish_pocket(state, scene, mode, redraw, updates);
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        crate::mol_editor::sync_md(state);
    }
}

pub(crate) fn has_preview(state: &State) -> bool {
    state
        .volatile
        .mol_manip
        .session
        .as_ref()
        .is_some_and(|s| !s.before.matches(state))
}

pub(crate) fn handle_key(
    state: &mut State,
    scene: &mut Scene,
    code: KeyCode,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) -> bool {
    if state.volatile.mol_manip.mode == ManipMode::None {
        return false;
    }
    if code == KeyCode::Escape || code == KeyCode::Enter || code == KeyCode::NumpadEnter {
        let numeric = &state.volatile.mol_manip.numeric;
        if code != KeyCode::Escape
            && !numeric.is_empty()
            && numeric
                .parse::<f64>()
                .ok()
                .is_none_or(|n| !n.is_finite() || n.abs() > 1.0e9)
        {
            return true;
        }

        end_transform(state, scene, code == KeyCode::Escape, redraw, updates);

        return true;
    }
    if !ensure_session(state) {
        return false;
    }

    let axis = match code {
        KeyCode::KeyX => Some(Axis::X),
        KeyCode::KeyY => Some(Axis::Y),
        KeyCode::KeyZ => Some(Axis::Z),
        _ => None,
    };
    if let Some(axis) = axis {
        let plane = state.volatile.key_modifiers.state().shift_key()
            && matches!(state.volatile.mol_manip.mode, ManipMode::Move(_));
        let constraint = if plane {
            Constraint::Plane(axis)
        } else {
            Constraint::Axis(axis)
        };
        let manip = &mut state.volatile.mol_manip;
        manip.constraint = if manip.constraint == constraint {
            Constraint::Free
        } else {
            constraint
        };
        manip.dragged = None;
    } else {
        let ch = match code {
            KeyCode::Digit0 | KeyCode::Numpad0 => Some('0'),
            KeyCode::Digit1 | KeyCode::Numpad1 => Some('1'),
            KeyCode::Digit2 | KeyCode::Numpad2 => Some('2'),
            KeyCode::Digit3 | KeyCode::Numpad3 => Some('3'),
            KeyCode::Digit4 | KeyCode::Numpad4 => Some('4'),
            KeyCode::Digit5 | KeyCode::Numpad5 => Some('5'),
            KeyCode::Digit6 | KeyCode::Numpad6 => Some('6'),
            KeyCode::Digit7 | KeyCode::Numpad7 => Some('7'),
            KeyCode::Digit8 | KeyCode::Numpad8 => Some('8'),
            KeyCode::Digit9 | KeyCode::Numpad9 => Some('9'),
            KeyCode::Period | KeyCode::NumpadDecimal => Some('.'),
            _ => None,
        };
        let numeric = &mut state.volatile.mol_manip.numeric;
        if let Some(ch) = ch {
            if numeric.len() < 24 && (ch != '.' || !numeric.contains('.')) {
                numeric.push(ch);
            }
        } else if matches!(code, KeyCode::Minus | KeyCode::NumpadSubtract) {
            if numeric.starts_with('-') {
                numeric.remove(0);
            } else {
                numeric.insert(0, '-');
            }
        } else if code == KeyCode::Backspace {
            numeric.pop();
        } else {
            return false;
        }
    }
    preview(state, scene, redraw, updates);

    true
}

pub(crate) fn status(state: &State) -> String {
    let manip = &state.volatile.mol_manip;
    let constraint = match manip.constraint {
        Constraint::Free => "Free".to_owned(),
        Constraint::Axis(axis) => axis.label().to_owned(),
        Constraint::Plane(axis) => format!(
            "{} plane",
            match axis {
                Axis::X => "YZ",
                Axis::Y => "XZ",
                Axis::Z => "XY",
            }
        ),
    };
    let amount = if !manip.numeric.is_empty() {
        manip.numeric.clone()
    } else if let Some(session) = &manip.session {
        match manip.mode {
            ManipMode::Move(_) => {
                let v = session.values.translation;
                format!("Δ ({:+.3}, {:+.3}, {:+.3})", v.x, v.y, v.z)
            }
            _ => format!("{:+.2}", session.values.angle.to_degrees()),
        }
    } else {
        "0".to_owned()
    };
    let unit = if matches!(manip.mode, ManipMode::Move(_)) {
        "Å"
    } else {
        "°"
    };

    format!(
        "{constraint}  {amount} {unit}{}",
        if snapping(state) { "  SNAP" } else { "" }
    )
}

pub fn set_manip(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    rebuild_md_editor: &mut bool,
    requested: ManipMode,
    updates: &mut EngineUpdates,
) {
    if state.volatile.region_selection.tool != crate::region_select::RegionTool::Off {
        crate::region_select::cancel(state, scene);
    }
    let previous = state.volatile.mol_manip.mode;
    let op = state.volatile.operating_mode;
    let target = match op {
        OperatingMode::Primary => state.volatile.active_mol,
        OperatingMode::MolEditor => {
            if matches!(state.volatile.active_mol, Some((MolType::Pocket, _))) {
                Some((MolType::Pocket, 0))
            } else {
                match &state.ui.selection {
                    Selection::AtomLig((_, i)) if matches!(requested, ManipMode::Move(_)) => {
                        Some((MolType::Ligand, *i))
                    }
                    Selection::AtomsLig((_, indices))
                        if matches!(requested, ManipMode::Move(_)) =>
                    {
                        indices.first().map(|i| (MolType::Ligand, *i))
                    }
                    Selection::BondLig((_, i)) if matches!(requested, ManipMode::Rotate(_)) => {
                        state
                            .mol_editor
                            .mol
                            .common
                            .bonds
                            .get(*i)
                            .filter(|b| !b.in_a_cycle(&state.mol_editor.mol.common.adjacency_list))
                            .map(|_| (MolType::Ligand, *i))
                    }
                    _ => None,
                }
            }
        }
        OperatingMode::ProteinEditor => None,
    };
    let mode = match (requested, target) {
        (ManipMode::Move(_), Some(target)) => ManipMode::Move(target),
        (ManipMode::Rotate(_), Some(target)) => ManipMode::Rotate(target),
        _ => ManipMode::None,
    };
    if previous != ManipMode::None {
        end_transform(state, scene, false, redraw, updates);
        *rebuild_md_editor = op == OperatingMode::MolEditor;
    }
    if mode == ManipMode::None || mode == previous {
        return;
    }
    if state.volatile.md_local.running || state.mol_editor.md.running {
        return;
    }

    if scene.input_settings.control_scheme != ControlScheme::None {
        state.volatile.control_scheme_prev = scene.input_settings.control_scheme;
    }
    scene.input_settings.control_scheme = ControlScheme::None;
    state.volatile.mol_manip.mode = mode;
    state.volatile.mol_manip.session = None;
    state.volatile.mol_manip.constraint = Constraint::Free;
    state.volatile.mol_manip.numeric.clear();
    ensure_session(state);
    if let Some((kind, _)) = mode.target() {
        redraw.set(kind);
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use graphics::Vertex;

    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn assert_scalar_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        assert_scalar_close(actual.x, expected.x);
        assert_scalar_close(actual.y, expected.y);
        assert_scalar_close(actual.z, expected.z);
    }

    fn assert_position_close(actual: [f32; 3], expected: [f32; 3]) {
        for i in 0..3 {
            assert_scalar_close(actual[i], expected[i]);
        }
    }

    #[test]
    fn peptide_mesh_translation_moves_vertices_without_changing_normals() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let mut mesh = Mesh {
            vertices: vec![Vertex::new([1.0, 2.0, 3.0], normal)],
            indices: Vec::new(),
            material: 0,
        };

        let mut transform = PeptideMeshTransform::default();
        transform.apply_delta(PeptideMeshDelta::Translate(Vec3::new(4.0, -2.0, 1.0)));

        assert!(transform_peptide_mesh(&mut mesh, transform));

        assert_position_close(mesh.vertices[0].position, [5.0, 0.0, 4.0]);
        assert_vec_close(mesh.vertices[0].normal, normal);
    }

    #[test]
    fn peptide_mesh_rotation_uses_molecule_pivot_and_rotates_vertex_basis() {
        let mut vertex = Vertex::new([2.0, 1.0, 0.0], Vec3::new(1.0, 0.0, 0.0));
        vertex.tangent = Vec3::new(0.0, 1.0, 0.0);
        vertex.bitangent = Vec3::new(0.0, 0.0, 1.0);
        let mut mesh = Mesh {
            vertices: vec![vertex],
            indices: Vec::new(),
            material: 0,
        };

        let mut transform = PeptideMeshTransform::default();
        transform.apply_delta(PeptideMeshDelta::Rotate {
            rotation: Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2),
            pivot: Vec3::new(1.0, 1.0, 0.0),
        });

        assert!(transform_peptide_mesh(&mut mesh, transform));

        let vertex = &mesh.vertices[0];
        assert_position_close(vertex.position, [1.0, 2.0, 0.0]);
        assert_vec_close(vertex.normal, Vec3::new(0.0, 1.0, 0.0));
        assert_vec_close(vertex.tangent, Vec3::new(-1.0, 0.0, 0.0));
        assert_vec_close(vertex.bitangent, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn peptide_mesh_transform_composes_deltas_in_world_order() {
        let mut transform = PeptideMeshTransform::default();
        transform.apply_delta(PeptideMeshDelta::Translate(Vec3::new(1.0, 0.0, 0.0)));
        transform.apply_delta(PeptideMeshDelta::Rotate {
            rotation: Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2),
            pivot: Vec3::new_zero(),
        });

        assert_vec_close(
            transform.transform_point(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(0.0, 2.0, 0.0),
        );
    }
}
