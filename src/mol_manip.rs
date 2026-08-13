//! Handles logic for moving and rotating molecules from user inputs.
//! This is for both primary mode, and the mol editor. In the latter, it
//! can move individual atoms, and rotate parts of molecules around bonds.

use graphics::{
    Camera, ControlScheme, EngineUpdates, FWD_VEC, Mesh, RIGHT_VEC, Scene, UP_VEC,
    event::MouseScrollDelta,
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};
use mol_defs::molecules::{MolType, common::MoleculeCommon};
use na_seq::Element;

use crate::{
    drawing::{EntityClass, MoleculeView},
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
    let (created, class, transform) = match state.ui.mol_view_peptide {
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
        if entity.class == class as u32 {
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
        && state.ui.mol_view_peptide == MoleculeView::Dots;

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

/// Blender-style mouse dragging of the molecule. For movement, creates a plane of the camera view,
/// at the molecule's depth. The mouse cursor projects to this plane, moving the molecule
/// along it. (Movement). Handles rotation in a straightforward manner. This is for motion relative
/// to the 2D screen, e.g. from mouse movement.
pub fn handle_mol_manip_in_plane(
    state: &mut State,
    scene: &mut Scene,
    delta: (f64, f64),
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    if state.volatile.mol_manip.mode == ManipMode::None {
        return;
    }

    // We skip renders, as they are relatively slow. This produces choppy dragging,
    // but I don't have a better plan yet.
    static mut I: u16 = 0;

    let Some(cursor) = state.ui.cursor_pos else {
        return;
    };

    let op_mode = state.volatile.operating_mode;
    let peptide_meshes_are_target = op_mode == OperatingMode::Primary
        && matches!(
            state.volatile.mol_manip.mode,
            ManipMode::Move((MolType::Peptide, i)) | ManipMode::Rotate((MolType::Peptide, i))
                if state.peptide_for_tools_i() == Some(i)
        );
    let ss_mesh_created = peptide_meshes_are_target && state.volatile.flags.ss_mesh_created;
    let sas_mesh_created = peptide_meshes_are_target && state.volatile.flags.sas_mesh_created;

    let mut rebuild_pocket = None; // Avoids double borrow.

    match state.volatile.mol_manip.mode {
        ManipMode::Move((mol_type, mol_i)) => {
            let mol = match op_mode {
                OperatingMode::Primary => match mol_type {
                    MolType::Peptide => {
                        if let Some(p) = state.peptides.get_mut(mol_i) {
                            &mut p.common
                        } else {
                            eprintln!("Error: No peptide in state for mol manip");
                            return;
                        }
                    }
                    MolType::Ligand => &mut state.ligands[mol_i].common,
                    MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                    MolType::Lipid => &mut state.lipids[mol_i].common,
                    MolType::Pocket => &mut state.pockets[mol_i].common,
                    _ => unimplemented!(),
                },
                OperatingMode::MolEditor => match mol_type {
                    MolType::Ligand => &mut state.mol_editor.mol.common,
                    MolType::Pocket => {
                        &mut state
                            .mol_editor
                            .mol
                            .pharmacophore
                            .pocket
                            .as_mut()
                            .unwrap()
                            .common
                    }
                    _ => unimplemented!(),
                },
                OperatingMode::ProteinEditor => unimplemented!(),
            };

            // Ray from screen
            let (ray_origin, ray_point) = scene.screen_to_render(cursor);
            let ray_dir = (ray_point - ray_origin).to_normalized();

            state.volatile.mol_manip.setup_params(
                &scene.camera,
                mol,
                mol_type,
                ray_origin,
                ray_dir,
                op_mode,
                mol_i,
            );

            // Cached at drag start
            let pivot = state.volatile.mol_manip.pivot.unwrap();
            let pivot_norm = state.volatile.mol_manip.view_dir.unwrap();
            let prev_offset = state.volatile.mol_manip.offset;

            let denom = ray_dir.dot(pivot_norm);
            if denom.abs() > 1e-6 {
                // Fixed plane: n · (X - pivot) = 0
                let t = pivot_norm.dot(pivot - ray_origin) / denom;

                let hit = ray_origin + ray_dir * t;

                let offset = hit - pivot;
                let delta_ = offset - prev_offset;

                let movement_vec: Vec3F64 = delta_.into();

                match op_mode {
                    OperatingMode::Primary => {
                        for p in &mut mol.atom_posits {
                            *p += movement_vec;
                        }
                    }
                    OperatingMode::MolEditor => {
                        if mol_type == MolType::Pocket {
                            // Move the whole pocket.
                            // println!(
                            //     "Moving pocket!. Centroid: {:?} len: {:?}",
                            //     mol.centroid(),
                            //     mol.atom_posits.len()
                            // ); // todo temp

                            for p in &mut mol.atom_posits {
                                *p += movement_vec;
                            }
                        } else {
                            // `mol_i` = atom_i here.
                            mol.atom_posits[mol_i] += movement_vec;
                            mol.atoms[mol_i].posit = mol.atom_posits[mol_i];

                            // Move all hydrogens bonded to the atom too.
                            for i in &mol.adjacency_list[mol_i] {
                                if mol.atoms[*i].element == Element::Hydrogen {
                                    mol.atom_posits[*i] += movement_vec;
                                }
                            }
                        }
                    }
                    OperatingMode::ProteinEditor => (),
                }

                if peptide_meshes_are_target {
                    handle_cached_peptide_mesh_delta(
                        state,
                        scene,
                        ss_mesh_created,
                        sas_mesh_created,
                        PeptideMeshDelta::Translate(delta_),
                        redraw,
                        updates,
                    );
                }
                state.volatile.mol_manip.offset = offset;

                let ratio = 8;
                unsafe {
                    I += 1;
                    if I.is_multiple_of(ratio)
                        || (op_mode == OperatingMode::MolEditor && mol_type == MolType::Ligand)
                    {
                        redraw.set(mol_type);

                        // todo: Only regen upon exiting manip
                        if mol_type == MolType::Pocket {
                            rebuild_pocket = Some(mol_i);
                        }
                    }
                }
            }
        }
        ManipMode::Rotate((mol_type, mol_i)) => {
            // todo: DRY with above
            // And note that we will likely get a borrow error if we use `get_mol()` here.
            let mol = match op_mode {
                OperatingMode::Primary => match mol_type {
                    MolType::Peptide => {
                        if let Some(p) = state.peptides.get_mut(mol_i) {
                            &mut p.common
                        } else {
                            eprintln!("Error: No peptide in state for mol manip");
                            return;
                        }
                    }
                    MolType::Ligand => &mut state.ligands[mol_i].common,
                    MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                    MolType::Lipid => &mut state.lipids[mol_i].common,
                    MolType::Pocket => &mut state.pockets[mol_i].common,
                    MolType::Water => unimplemented!(),
                },
                OperatingMode::MolEditor => match mol_type {
                    MolType::Ligand => &mut state.mol_editor.mol.common,
                    MolType::Pocket => {
                        if let Some(p) = &mut state.mol_editor.mol.pharmacophore.pocket {
                            &mut p.common
                        } else {
                            return;
                        }
                    }
                    _ => unimplemented!(),
                },
                OperatingMode::ProteinEditor => unimplemented!(),
            };

            match op_mode {
                OperatingMode::Primary => {
                    // We handle rotation around the fwd/z axis using the scroll wheel.
                    // let fwd = scene.camera.orientation.rotate_vec(FWD_VEC);
                    let right = scene
                        .camera
                        .orientation
                        .rotate_vec(RIGHT_VEC)
                        .to_normalized();
                    let up = scene.camera.orientation.rotate_vec(UP_VEC).to_normalized();

                    let rot_x =
                        Quaternion::from_axis_angle(right, -delta.1 as f32 * SENS_MOL_ROT_MOUSE);
                    let rot_y =
                        Quaternion::from_axis_angle(up, -delta.0 as f32 * SENS_MOL_ROT_MOUSE);

                    let rot = rot_y * rot_x; // Note: Can swap the order for a slightly different effect.
                    let pivot = mol.centroid().into();

                    mol.rotate(rot.into(), None);
                    if peptide_meshes_are_target {
                        handle_cached_peptide_mesh_delta(
                            state,
                            scene,
                            ss_mesh_created,
                            sas_mesh_created,
                            PeptideMeshDelta::Rotate {
                                rotation: rot,
                                pivot,
                            },
                            redraw,
                            updates,
                        );
                    }

                    // // Pockets: We rotate a single mesh, instead of a collection of point-based items.
                    // // Update it.
                    // if mol_type == MolType::Pocket {
                    //     state.pockets[mol_i].mesh_orientation =
                    //         (rot * state.pockets[mol_i].mesh_orientation).to_normalized();
                    // }
                }
                OperatingMode::MolEditor => {
                    if mol_type == MolType::Pocket {
                        // Rotate the whole pocket, like Primary mode.
                        let right = scene
                            .camera
                            .orientation
                            .rotate_vec(RIGHT_VEC)
                            .to_normalized();
                        let up = scene.camera.orientation.rotate_vec(UP_VEC).to_normalized();

                        let rot_x = Quaternion::from_axis_angle(
                            right,
                            -delta.1 as f32 * SENS_MOL_ROT_MOUSE,
                        );
                        let rot_y =
                            Quaternion::from_axis_angle(up, -delta.0 as f32 * SENS_MOL_ROT_MOUSE);

                        let rot = rot_y * rot_x;

                        mol.rotate(rot.into(), None);
                    } else {
                        const ROT_FACTOR: f64 = 0.008;
                        mol.rotate_around_bond(mol_i, ROT_FACTOR * delta.0, None);
                    }
                }
                OperatingMode::ProteinEditor => (),
            }

            let ratio = 8;
            unsafe {
                I += 1;
                if I.is_multiple_of(ratio)
                    || (op_mode == OperatingMode::MolEditor && mol_type == MolType::Ligand)
                {
                    redraw.set(mol_type);

                    // todo: Only regen upon exiting manip
                    if mol_type == MolType::Pocket {
                        rebuild_pocket = Some(mol_i);
                    }
                }
            }
        }
        ManipMode::None => (),
    }

    // Rebuild H bonds if either the ligand or pocket is changed.
    if op_mode == OperatingMode::MolEditor && (rebuild_pocket.is_some() || redraw.ligand) {
        state.mol_editor.update_h_bonds();
    }

    if let Some(mol_i) = rebuild_pocket {
        if op_mode == OperatingMode::MolEditor {
            if let Some(p) = &mut state.mol_editor.mol.pharmacophore.pocket {
                p.rebuild_spheres();
            }
        } else {
            state.pockets[mol_i].rebuild_spheres();
        }
    }
}

/// Forward and backwards, with 0 line of sight to the camera. E.g. with the scroll wheel.
pub fn handle_mol_manip_in_out(
    state: &mut State,
    scene: &mut Scene,
    delta: MouseScrollDelta,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let mut rebuild_pocket = None; // Avoids double borrow.
    let op_mode = state.volatile.operating_mode;
    let peptide_meshes_are_target = op_mode == OperatingMode::Primary
        && matches!(
            state.volatile.mol_manip.mode,
            ManipMode::Move((MolType::Peptide, i)) | ManipMode::Rotate((MolType::Peptide, i))
                if state.peptide_for_tools_i() == Some(i)
        );
    let ss_mesh_created = peptide_meshes_are_target && state.volatile.flags.ss_mesh_created;
    let sas_mesh_created = peptide_meshes_are_target && state.volatile.flags.sas_mesh_created;

    // Move the molecule forward and backwards relative to the camera on scroll.
    match state.volatile.mol_manip.mode {
        ManipMode::Move((mol_type, mol_i)) => {
            // Note: We can't use `get_mol` here due to borrow errors I don't understand.
            let mol = match op_mode {
                OperatingMode::Primary => match mol_type {
                    MolType::Peptide => {
                        if let Some(p) = state.peptides.get_mut(mol_i) {
                            &mut p.common
                        } else {
                            eprintln!("Error: No peptide in state for mol manip");
                            return;
                        }
                    }
                    MolType::Ligand => {
                        if mol_i >= state.ligands.len() {
                            eprintln!("Error: Index out of bounds on ligand for mol manip");
                            return;
                        }
                        &mut state.ligands[mol_i].common
                    }
                    MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                    MolType::Lipid => &mut state.lipids[mol_i].common,
                    MolType::Pocket => &mut state.pockets[mol_i].common,
                    MolType::Water => return,
                },
                OperatingMode::MolEditor => match mol_type {
                    MolType::Ligand => &mut state.mol_editor.mol.common,
                    MolType::Pocket => {
                        if let Some(p) = &mut state.mol_editor.mol.pharmacophore.pocket {
                            &mut p.common
                        } else {
                            return;
                        }
                    }
                    _ => return,
                },
                OperatingMode::ProteinEditor => unimplemented!(),
            };

            let scroll: f32 = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
            };
            if scroll == 0.0 {
                return;
            }

            // todo: DRY.
            // Note: This mol manip state is relevant when scrolling with the mouse button down.
            // If not also supporting zooming in and out, we'd cache these values at the drag start.
            // If we do that, moving the mol and and out would be wonky mid-drag.
            {
                let pivot: Vec3 = match op_mode {
                    OperatingMode::Primary => mol.centroid().into(),
                    OperatingMode::MolEditor => {
                        if mol_i < mol.atom_posits.len() {
                            mol.atom_posits[mol_i].into()
                        } else {
                            mol.centroid().into()
                        }
                    }
                    OperatingMode::ProteinEditor => unimplemented!(),
                };

                let cam_pos32: Vec3 = scene.camera.position.into();
                let view_dir = (pivot - cam_pos32).to_normalized();

                state.volatile.mol_manip.pivot = Some(pivot);
                state.volatile.mol_manip.view_dir = Some(view_dir);
                state.volatile.mol_manip.offset = Vec3::new_zero();
            }

            if let (Some(pivot), _) = (
                state.volatile.mol_manip.pivot,
                state.volatile.mol_manip.view_dir,
            ) {
                let view_dir = (pivot - scene.camera.position).to_normalized();
                state.volatile.mol_manip.view_dir = Some(view_dir);

                let dist = (pivot - scene.camera.position).magnitude();
                let step = state.to_save.control_settings.mol_move_sens as f32 / 1_000. * dist;

                let dv = view_dir * (scroll * step);
                let movement_vec: Vec3F64 = dv.into();

                match op_mode {
                    OperatingMode::Primary => {
                        for p in &mut mol.atom_posits {
                            *p += movement_vec;
                        }
                    }
                    OperatingMode::MolEditor => {
                        if mol_type == MolType::Pocket {
                            for p in &mut mol.atom_posits {
                                *p += movement_vec;
                            }
                        } else {
                            mol.atom_posits[mol_i] += movement_vec;
                            mol.atoms[mol_i].posit = mol.atom_posits[mol_i];

                            // Move all hydrogens bonded to the atom too.
                            for i in &mol.adjacency_list[mol_i] {
                                if mol.atoms[*i].element == Element::Hydrogen {
                                    mol.atom_posits[*i] += movement_vec;
                                }
                            }
                        }
                    }
                    OperatingMode::ProteinEditor => (),
                }

                if peptide_meshes_are_target {
                    handle_cached_peptide_mesh_delta(
                        state,
                        scene,
                        ss_mesh_created,
                        sas_mesh_created,
                        PeptideMeshDelta::Translate(dv),
                        redraw,
                        updates,
                    );
                }

                let new_pivot = pivot + dv;
                state.volatile.mol_manip.pivot = Some(new_pivot);
                state.volatile.mol_manip.depth_bias += scroll * step;

                // todo: QC if you need/want this.
                // recompute intersection on the shifted plane to maintain stable drag
                if let Some(cursor) = state.ui.cursor_pos {
                    let (ray_origin, ray_point) = scene.screen_to_render(cursor);
                    let rd = (ray_point - ray_origin).to_normalized();

                    let denom = rd.dot(view_dir);
                    if denom.abs() > 1e-6 {
                        let t = view_dir.dot(new_pivot - ray_origin) / denom;
                        if t > 0.0 {
                            let hit = ray_origin + rd * t;
                            state.volatile.mol_manip.offset = hit - new_pivot;
                        }
                    }
                }
            }
            redraw.set(mol_type);
            // todo: Only regen upon exiting manip
            if mol_type == MolType::Pocket {
                rebuild_pocket = Some(mol_i);
            }
        }
        ManipMode::Rotate((mol_type, mol_i)) => {
            let scroll: f32 = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
            };
            if scroll == 0.0 {
                return;
            }

            // todo: C+P with slight changes from the mouse-move variant.
            let mol = match mol_type {
                MolType::Peptide => {
                    if let Some(p) = state.peptides.get_mut(mol_i) {
                        &mut p.common
                    } else {
                        eprintln!("Error: No peptide in state for mol manip");
                        return;
                    }
                }
                MolType::Ligand => {
                    if mol_i >= state.ligands.len() {
                        eprintln!("Error: Index out of bounds on ligand for mol manip");
                        return;
                    }

                    &mut state.ligands[mol_i].common
                }
                MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                MolType::Lipid => &mut state.lipids[mol_i].common,
                MolType::Pocket => match op_mode {
                    OperatingMode::Primary => &mut state.pockets[mol_i].common,
                    OperatingMode::MolEditor => {
                        if let Some(p) = &mut state.mol_editor.mol.pharmacophore.pocket {
                            &mut p.common
                        } else {
                            return;
                        }
                    }
                    _ => return,
                },
                MolType::Water => unimplemented!(),
            };

            let fwd = scene.camera.orientation.rotate_vec(FWD_VEC).to_normalized();

            let rot = Quaternion::from_axis_angle(fwd, scroll * SENS_MOL_ROT_SCROLL);
            let pivot = mol.centroid().into();
            mol.rotate(rot.into(), None);

            if peptide_meshes_are_target {
                handle_cached_peptide_mesh_delta(
                    state,
                    scene,
                    ss_mesh_created,
                    sas_mesh_created,
                    PeptideMeshDelta::Rotate {
                        rotation: rot,
                        pivot,
                    },
                    redraw,
                    updates,
                );
            }

            // Pockets: We rotate a single mesh, instead of a collection of point-based items.
            // Update it.
            if mol_type == MolType::Pocket {
                // state.pockets[mol_i].mesh_orientation =
                //     (rot * state.pockets[mol_i].mesh_orientation).to_normalized();
                rebuild_pocket = Some(mol_i);
            }

            redraw.set(mol_type);
        }
        ManipMode::None => (),
    }

    // Rebuild H bonds if either the ligand or pocket is changed.
    if op_mode == OperatingMode::MolEditor && (rebuild_pocket.is_some() || redraw.ligand) {
        state.mol_editor.update_h_bonds();
    }

    if let Some(mol_i) = rebuild_pocket {
        if op_mode == OperatingMode::MolEditor {
            if let Some(p) = &mut state.mol_editor.mol.pharmacophore.pocket {
                p.rebuild_spheres();
            }
        } else {
            state.pockets[mol_i].rebuild_spheres();
        }
    }
}

/// Sets the manipulation mode, and adjusts camera controls A/R. Called from inputs, or the UI.
/// Toggles as required, or switches mode. Performs cleanup as required.
pub fn set_manip(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    rebuild_md_editor: &mut bool,
    // Note: The mol itself is overwritten but this sets move/rotate,
    mode: ManipMode,
    updates: &mut EngineUpdates,
) {
    // Finish any previous peptide drag before toggling or switching manipulation tools. This is
    // especially important for regenerating a deferred Dots representation.
    finish_peptide_mesh_manipulation(state, redraw);

    let vol = &mut state.volatile;
    if mode == ManipMode::None {
        vol.mol_manip.mode = ManipMode::None;
        vol.mol_manip.pivot = None;
        return;
    }

    let op_mode = vol.operating_mode;

    // If in primary mode, the item to move is an entire molecule, indexec by the appropriate molecule set.
    // In the mol editor, we are interested in the selected atom or bond. (Or the pocket)
    let (mol_type_active, item_to_move_i) = match op_mode {
        OperatingMode::Primary => match vol.active_mol {
            Some(v) => v,
            None => return,
        },
        // In the editor mode, select the selected atom as the one to move.
        OperatingMode::MolEditor => {
            if matches!(vol.active_mol, Some((MolType::Pocket, _))) {
                (MolType::Pocket, 0)
            } else {
                match &state.ui.selection {
                    Selection::AtomLig((_, i)) => (MolType::Ligand, *i),
                    Selection::AtomsLig((_, i)) => {
                        // todo: How should we handle this?
                        (MolType::Ligand, i[0])
                    }
                    // Rotating around a bond.
                    Selection::BondLig((_, i)) => (MolType::Ligand, *i),
                    _ => return,
                    // Allow pocket manip without requiring a pocket atom selection.
                    // _ => match mode {
                    //     ManipMode::Move((MolType::Pocket, _)) | ManipMode::Rotate((MolType::Pocket, _)) => {
                    //         (MolType::Pocket, 0)
                    //     }
                    //     _ => return,
                    // },
                }
            }
        }
        OperatingMode::ProteinEditor => unimplemented!(),
    };

    // if matches!(
    //     mode,
    //     ManipMode::Move((MolType::Pocket, _)) | ManipMode::Rotate((MolType::Pocket, _))
    // ) && op_mode == OperatingMode::MolEditor
    // {
    //     println!("Override: manip pocket in editor");
    //     mol_type_active = MolType::Pocket;
    //     i_active = 0;
    // }

    let (move_active, rotate_active) = {
        let mut move_ = false;
        let mut rotate = false;

        match vol.mol_manip.mode {
            ManipMode::None => (),
            ManipMode::Move((mol_type, mol_i)) => {
                if mol_type == mol_type_active && mol_i == item_to_move_i {
                    move_ = true;
                }
            }
            ManipMode::Rotate((mol_type, mol_i)) => {
                if mol_type == mol_type_active && mol_i == item_to_move_i {
                    rotate = true;
                }
            }
        }

        (move_, rotate)
    };

    match mode {
        ManipMode::Move(_) => {
            if move_active {
                // Exiting a move
                scene.input_settings.control_scheme = vol.control_scheme_prev;
                vol.mol_manip.mode = ManipMode::None;
                vol.mol_manip.pivot = None;

                if op_mode == OperatingMode::MolEditor {
                    *rebuild_md_editor = true;
                }
            } else if rotate_active {
                // Entering a move from rotation
                vol.mol_manip.mode = ManipMode::Move((mol_type_active, item_to_move_i));
            } else {
                // Entering a move from no manip prior.

                if scene.input_settings.control_scheme != ControlScheme::None {
                    vol.control_scheme_prev = scene.input_settings.control_scheme;
                }
                scene.input_settings.control_scheme = ControlScheme::None;
                vol.mol_manip.mode = ManipMode::Move((mol_type_active, item_to_move_i));
            };
        }
        ManipMode::Rotate(_) => {
            if rotate_active {
                scene.input_settings.control_scheme = vol.control_scheme_prev;
                vol.mol_manip.mode = ManipMode::None;
                vol.mol_manip.pivot = None;

                if op_mode == OperatingMode::MolEditor {
                    *rebuild_md_editor = true;
                }
            } else if move_active {
                vol.mol_manip.mode = ManipMode::Rotate((mol_type_active, item_to_move_i));
            } else {
                if scene.input_settings.control_scheme != ControlScheme::None {
                    vol.control_scheme_prev = scene.input_settings.control_scheme;
                }
                scene.input_settings.control_scheme = ControlScheme::None;
                vol.mol_manip.mode = ManipMode::Rotate((mol_type_active, item_to_move_i));
            };
        }
        ManipMode::None => unreachable!(),
    }

    // Once complete with manip on a pocket, rebuild the volume, representation.
    if let Some((mol_type, i)) = vol.active_mol
        && mol_type == MolType::Pocket
    {
        let p = if op_mode == OperatingMode::MolEditor {
            if let Some(p_) = &mut state.mol_editor.mol.pharmacophore.pocket {
                p_
            } else {
                eprintln!("Missing pocket tos set manip on");
                return;
            }
        } else {
            &mut state.pockets[i]
        };

        p.reset_post_manip(&mut scene.meshes, state.ui.mesh_coloring, updates);
        redraw.pocket = true;
    }

    if op_mode == OperatingMode::MolEditor {
        state.mol_editor.update_h_bonds();
    }

    redraw.set(mol_type_active);
}

/// Chooses the atom closest to the cursor ray as the pivot.
/// This helps ensure, while drag-moving, that the molecule stays anchored to the cursor in
/// a way that feels intuitive; the part moving to the cursor's position is the part originally
/// under the cursor at drag start.
fn pick_movement_pivot(mol: &MoleculeCommon, ray_origin: Vec3, ray_dir: Vec3) -> Vec3 {
    const PICK_RADIUS: f64 = 4.0; // Å-ish // A/R
    let pick_r2 = PICK_RADIUS * PICK_RADIUS;

    let ro: Vec3F64 = ray_origin.into();
    let rd: Vec3F64 = ray_dir.into();

    let mut best_d2 = f64::INFINITY;
    let mut best_p: Option<Vec3F64> = None;

    for p in &mol.atom_posits {
        let w = *p - ro;
        let t = w.dot(rd); // rd assumed normalized
        if t < 0.0 {
            continue;
        }
        let closest = ro + rd * t;
        let d2 = (*p - closest).magnitude_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_p = Some(*p);
        }
    }

    if best_d2 <= pick_r2 {
        best_p.unwrap().into()
    } else {
        mol.centroid().into()
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub enum ManipMode {
    #[default]
    None,
    Move((MolType, usize)), // Index of mol
    Rotate((MolType, usize)),
}

/// State for dragging and rotating molecules.
#[derive(Default, Debug)]
pub struct MolManip {
    /// Allows the user to move a molecule around with mouse or keyboard.
    pub mode: ManipMode,
    /// For maintaining the screen plane when dragging the mol.
    pub pivot: Option<Vec3>,
    pub view_dir: Option<Vec3>,
    pub offset: Vec3,
    pub depth_bias: f32,
    /// Persistent transforms from the cached world-space mesh vertices to current peptide
    /// coordinates. They reset only when their corresponding mesh is regenerated.
    pub(crate) ribbon_mesh_transform: PeptideMeshTransform,
    pub(crate) surface_mesh_transform: PeptideMeshTransform,
    /// Dots expand a surface mesh into many entities, so defer rebuilding them until the current
    /// drag/tool interaction completes.
    pub(crate) peptide_mesh_manip_pending: bool,
}

/// Set pivot, view_dir, and offset based on mol positions and other data.
/// Run this whenever the position in plane is changed.
impl MolManip {
    fn setup_params(
        &mut self,
        cam: &Camera,
        mol: &MoleculeCommon,
        mol_type: MolType,
        ray_origin: Vec3,
        ray_dir: Vec3,
        mode: OperatingMode,
        atom_i: usize, // For edit mode, of the atom being moved.
    ) {
        if self.pivot.is_none() {
            // We set the pivot to be the coordinates of the molecule (e.g. nearest atom)
            // to the cursor. We're moving this pivot with the mouse cursor, so we take
            // this approach to prevent the mol jumping; this part *snaps* to the cursor.
            let pivot: Vec3 = match mode {
                OperatingMode::Primary => pick_movement_pivot(mol, ray_origin, ray_dir),
                OperatingMode::MolEditor => {
                    match mol_type {
                        MolType::Ligand => {
                            if atom_i < mol.atom_posits.len() {
                                mol.atom_posits[atom_i].into()
                            } else {
                                // Pocket manip uses index 0 as a placeholder; fall back to centroid.
                                pick_movement_pivot(mol, ray_origin, ray_dir)
                            }
                        }
                        MolType::Pocket => pick_movement_pivot(mol, ray_origin, ray_dir),
                        _ => unimplemented!(),
                    }
                }
                OperatingMode::ProteinEditor => unimplemented!(),
            };

            let n = cam.orientation.rotate_vec(FWD_VEC).to_normalized();

            self.pivot = Some(pivot);
            self.view_dir = Some(n);
            self.offset = Vec3::new_zero();
        }
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
