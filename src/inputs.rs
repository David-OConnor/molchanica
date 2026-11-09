//! Handles user inputs, e.g. from keyboard and mouse.

use bio_files::BondType;
use graphics::{
    ControlScheme, DeviceEvent, ElementState, EngineUpdates, FWD_VEC, Scene, WindowEvent,
    event::{MouseButton, MouseScrollDelta},
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::f32::Vec3;
use mol_defs::molecules::MolType;
use na_seq::Element::Carbon;

use crate::{
    cam,
    cam::{FOG_DIST_MAX, FOG_DIST_MIN, move_cam_to_sel},
    drawing::{EntityClass, MolTypeExt, draw_pocket, peptide, wrappers},
    mol_editor,
    mol_editor::{add_atoms::add_atom, sync_md},
    mol_manip,
    mol_manip::{ManipMode, set_manip},
    prefs::DepthMode,
    render::set_flashlight,
    selection,
    selection::{Selection, cycle_selected},
    state::{OperatingMode, State},
    util::{RedrawFlags, close_mol},
};

// These are defaults; overridden by the user A/R, and saved to prefs.
pub const MOVEMENT_SENS: f32 = 12.;
pub const ROTATE_SENS: f32 = 0.45;
pub const RUN_FACTOR: f32 = 6.; // i.e. shift key multiplier

pub const SCROLL_MOVE_AMT: f32 = 4.;
pub const SCROLL_ROTATE_AMT: f32 = 12.;

// Sensitives for mol manip.
pub const SENS_MOL_MOVE_SCROLL: f32 = 2.5e-2;
pub const SENS_MOL_ROT_SCROLL: f32 = 5e-2;
pub const SENS_MOL_ROT_MOUSE: f32 = 5e-3;

/// We use this lower-level input compared to window events to handle mouse motion.
pub fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _engine_inputs: bool,
    dt: f32,
) -> EngineUpdates {
    let mut updates = EngineUpdates::default();

    // Device events can happen even if the window isn't active; use the cursor position
    // to identify this application as active.
    if !state_.ui.mouse_in_window {
        return updates;
    }

    // todo: DRY with graphics.
    // This affects our app-specific commands, vs engine built-in ones. For example, hot keys
    // to change various modes.

    let redraw = RedrawFlags::default();
    let redraw_in_place = RedrawFlags::default();
    let redraw_mol_editor = false;

    match event {
        DeviceEvent::MouseMotion { delta } => {
            // Free look handled by the engine; handle middle-click-move here.
            if state_.ui.middle_click_down {
                // The same movement sensitivity scaler we use for the (1x effective multiplier)
                // on keyboard movement seems to work well enough here.
                // Camera-local X points left on screen if the view is mirrored.
                let mut movement_vec = Vec3::new(
                    delta.0 as f32 * MOVEMENT_SENS * dt * scene.camera.screen_x_sign(),
                    -delta.1 as f32 * MOVEMENT_SENS * dt,
                    0.,
                );

                if scene.input_settings.control_scheme != ControlScheme::FreeCamera {
                    movement_vec *= -1.;
                }

                scene.camera.position += scene.camera.orientation.rotate_vec(movement_vec);
                updates.camera = true;

                // Loose proxy that lets us trigger the inputs_present() flag.
                state_.volatile.inputs_commanded.fwd = true;

                set_flashlight(scene);
                updates.lighting = true;
            }
        }
        _ => (),
    }

    post_event_cleanup(
        state_,
        scene,
        &redraw,
        &redraw_in_place,
        redraw_mol_editor,
        &mut updates,
    );

    updates
}

pub fn event_win_handler(
    state_: &mut State,
    event: WindowEvent,
    scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    let mut updates = EngineUpdates::default();

    let selection_before = matches!(
        event,
        WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. }
    )
    .then(|| crate::edit_history::SelectionState::capture(state_));
    let history_revision = state_.volatile.history.revision;
    let editor_before = matches!(
        event,
        WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. }
    )
    .then(|| crate::edit_history::EditorSnapshot::capture(state_))
    .flatten();

    let mut redraw = RedrawFlags::default();
    let mut redraw_in_place = RedrawFlags::default();
    let mut redraw_mol_editor = false;

    match event {
        WindowEvent::CursorMoved {
            device_id: _,
            position,
        } => {
            let previous = state_.ui.cursor_pos;
            state_.ui.cursor_pos = Some((position.x as f32, position.y as f32));
            if state_.ui.left_click_down
                && !crate::region_select::motion(state_)
                && let Some(previous) = previous
            {
                mol_manip::handle_mol_manip_in_plane(
                    state_,
                    scene,
                    (
                        position.x - previous.0 as f64,
                        position.y - previous.1 as f64,
                    ),
                    &mut redraw_in_place,
                    &mut updates,
                );
            }
        }
        WindowEvent::CursorEntered { device_id: _ } => {
            state_.ui.mouse_in_window = true;
        }
        WindowEvent::CursorLeft { device_id: _ } => {
            state_.ui.mouse_in_window = false;
        }
        WindowEvent::Resized(_) => {
            state_.ui.mouse_in_window = true;
        }
        WindowEvent::Focused(val) => {
            state_.ui.mouse_in_window = val;
            if !val {
                if state_.volatile.mol_manip.mode != ManipMode::None {
                    mol_manip::end_transform(state_, scene, true, &mut redraw, &mut updates);
                }
                crate::region_select::cancel(state_, scene);
            }
        }
        WindowEvent::ModifiersChanged(val) => {
            state_.volatile.key_modifiers = val;
        }
        // If you wish to appease Wayland, use these instead of device events
        WindowEvent::KeyboardInput {
            device_id: _,
            event,
            is_synthetic: _,
        } => {
            if let Code(code) = event.physical_key {
                if event.state == ElementState::Pressed {
                    state_.volatile.handled_window_keys.push(code);
                }
                if event.repeat
                    && matches!(
                        code,
                        KeyCode::KeyM | KeyCode::KeyR | KeyCode::KeyB | KeyCode::KeyL
                    )
                {
                    return updates;
                }
                handle_physical_key(
                    state_,
                    scene,
                    &mut redraw,
                    &mut redraw_in_place,
                    &mut redraw_mol_editor,
                    &mut updates,
                    code,
                    event.state,
                );
            }
        }
        WindowEvent::MouseInput {
            device_id: _,
            state,
            button,
        } => handle_mouse_button(
            state_,
            scene,
            &mut redraw,
            &mut redraw_mol_editor,
            &mut updates,
            button,
            state,
        ),
        WindowEvent::MouseWheel {
            device_id: _,
            delta,
            phase: _,
        } => {
            if handle_scroll(state_, scene, &mut updates, &mut redraw_in_place, delta) {
                return updates;
            }
        }
        _ => (),
    }

    crate::edit_history::record_editor(state_, editor_before, history_revision);
    if let Some(selection_before) = selection_before {
        crate::edit_history::record_selection(state_, selection_before, history_revision);
    }

    post_event_cleanup(
        state_,
        scene,
        &redraw,
        &redraw_in_place,
        redraw_mol_editor,
        &mut updates,
    );

    updates
}

pub(crate) fn handle_ui_interactions(
    state: &mut State,
    scene: &mut Scene,
    ctx: &egui::Context,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    use egui::Key;
    use graphics::winit::keyboard::ModifiersState;

    let frame = ctx.cumulative_frame_nr();
    if state.volatile.last_interaction_frame == Some(frame) {
        return;
    }
    state.volatile.last_interaction_frame = Some(frame);

    let (modifiers, focused, cursor, primary_down, events) = ctx.input(|input| {
        (
            input.modifiers,
            input.focused,
            input.pointer.latest_pos(),
            input.pointer.primary_down(),
            input.events.clone(),
        )
    });
    let mut flags = ModifiersState::empty();
    flags.set(ModifiersState::SHIFT, modifiers.shift);
    flags.set(ModifiersState::CONTROL, modifiers.ctrl);
    flags.set(ModifiersState::ALT, modifiers.alt);
    flags.set(ModifiersState::SUPER, modifiers.mac_cmd);
    state.volatile.key_modifiers = flags.into();

    if !focused {
        if state.volatile.mol_manip.mode != ManipMode::None {
            mol_manip::end_transform(state, scene, true, redraw, updates);
        }
        crate::region_select::cancel(state, scene);
    } else if state.ui.left_click_down {
        if let Some(cursor) = cursor {
            let scale = ctx.pixels_per_point();
            let cursor = (cursor.x * scale, cursor.y * scale);
            let previous = state.ui.cursor_pos.replace(cursor);
            if !crate::region_select::motion(state)
                && let Some(previous) = previous
                && cursor != previous
            {
                mol_manip::handle_mol_manip_in_plane(
                    state,
                    scene,
                    (
                        (cursor.0 - previous.0) as f64,
                        (cursor.1 - previous.1) as f64,
                    ),
                    redraw,
                    updates,
                );
            }
        }
        if !primary_down {
            if !crate::region_select::finish(state, scene, redraw) {
                mol_manip::finish_drag(state, redraw);
            }
            state.ui.left_click_down = false;
        }
    }

    let handled = std::mem::take(&mut state.volatile.handled_window_keys);
    if ctx.text_edit_focused() {
        return;
    }

    for event in events {
        let egui::Event::Key {
            key,
            physical_key,
            pressed: true,
            repeat,
            ..
        } = event
        else {
            continue;
        };
        let code = match physical_key.unwrap_or(key) {
            Key::Escape => KeyCode::Escape,
            Key::Enter => KeyCode::Enter,
            Key::Backspace => KeyCode::Backspace,
            Key::Minus => KeyCode::Minus,
            Key::Period => KeyCode::Period,
            Key::Num0 => KeyCode::Digit0,
            Key::Num1 => KeyCode::Digit1,
            Key::Num2 => KeyCode::Digit2,
            Key::Num3 => KeyCode::Digit3,
            Key::Num4 => KeyCode::Digit4,
            Key::Num5 => KeyCode::Digit5,
            Key::Num6 => KeyCode::Digit6,
            Key::Num7 => KeyCode::Digit7,
            Key::Num8 => KeyCode::Digit8,
            Key::Num9 => KeyCode::Digit9,
            Key::X => KeyCode::KeyX,
            Key::Y => KeyCode::KeyY,
            Key::Z => KeyCode::KeyZ,
            Key::M => KeyCode::KeyM,
            Key::R => KeyCode::KeyR,
            Key::B => KeyCode::KeyB,
            Key::L => KeyCode::KeyL,
            _ => continue,
        };
        if handled.contains(&code)
            || (repeat
                && matches!(
                    code,
                    KeyCode::KeyM | KeyCode::KeyR | KeyCode::KeyB | KeyCode::KeyL
                ))
        {
            continue;
        }

        let mut in_place = RedrawFlags::default();
        let mut editor = false;
        handle_physical_key(
            state,
            scene,
            redraw,
            &mut in_place,
            &mut editor,
            updates,
            code,
            ElementState::Pressed,
        );
        redraw.peptide |= in_place.peptide;
        redraw.ligand |= in_place.ligand || editor;
        redraw.na |= in_place.na;
        redraw.lipid |= in_place.lipid;
        redraw.pocket |= in_place.pocket;
        ctx.input_mut(|input| {
            input.consume_key(modifiers, key);
        });
    }
}

#[allow(unused)]
/// Debug code to draw teh ray on screen, so we can see why the selection is off.
fn plot_ray() {
    // let center = (selected_ray.0 + selected_ray.1) / 2.;
    //
    // let diff = selected_ray.0 - selected_ray.1;
    // let diff_unit = diff.to_normalized();
    // let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
    //
    // let scale = Some(Vec3::new(0.3, diff.magnitude(), 0.3));
    //
    // let mut ent = Entity::new(
    //     MESH_BOND,
    //     center,
    //     orientation,
    //     1.,
    //     (1., 0., 1.),
    //     BODY_SHINYNESS,
    // );
    // ent.scale_partial = scale;
    //
    // scene.entities.push(ent);
    // updates.entities = true;
}

//         // let mol_i = state_.volatile.active_mol.unwrap().1;
//         //
//         // if mol_i >= state_.lipids.len() {
//         //     eprintln!("Uhoh: Index error on in-place redraw");
//         //     wrappers::draw_all_lipids(state_, scene);
//         //     return updates;
//         // }
//         //
//         // wrappers::update_single_lipid_inplace(mol_i, state_, scene);
//         //
//         // let mol = &mut state_.lipids[mol_i];
//         // updates.entities = match mol.common.entity_i_range {
//         //     Some(range) => EntityUpdate::Indexes(range),
//         //     None => {
//         //         eprintln!("Error: Missing entity index range.");
//         //         EntityUpdate::All
//         //     }
//         // };

/// Abstracts over mol types.
fn redraw_inplace_helper(
    mol_type: MolType,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    let Some((active_type, mol_i)) = state.volatile.active_mol else {
        return;
    };
    if active_type != mol_type {
        return;
    }

    let Some(mol) = state.get_mol(mol_type, mol_i) else {
        return;
    };
    let class = mol_type.entity_type() as u32;
    let valid_range = mol.common().entity_i_range.is_some_and(|(start, end)| {
        scene
            .entities
            .get(start..end)
            .is_some_and(|entities| entities.iter().all(|entity| entity.class == class))
    });

    // Representations and selection highlights can invalidate cached entity ranges. Recreate
    // the entities at the current atom positions instead of leaving them still until tool exit.
    if !valid_range {
        match mol_type {
            MolType::Peptide => peptide::draw_peptide(state, scene, updates),
            MolType::Ligand => wrappers::draw_all_ligs(state, scene, updates),
            MolType::NucleicAcid => wrappers::draw_all_nucleic_acids(state, scene, updates),
            MolType::Lipid => wrappers::draw_all_lipids(state, scene, updates),
            MolType::Pocket => wrappers::draw_all_pockets(state, scene, updates),
            MolType::Water => unreachable!(),
        }
        return;
    }

    match mol_type {
        MolType::Peptide => wrappers::update_single_peptide_inplace(mol_i, state, scene),
        MolType::Ligand => wrappers::update_single_ligand_inplace(mol_i, state, scene),
        MolType::NucleicAcid => wrappers::update_single_nucleic_acid_inplace(mol_i, state, scene),
        MolType::Lipid => wrappers::update_single_lipid_inplace(mol_i, state, scene),
        MolType::Pocket => wrappers::update_single_pocket_inplace(mol_i, state, scene),
        MolType::Water => unreachable!(),
    };

    updates.entities.push_class(class);
}

fn handle_mouse_button(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    redraw_mol_editor: &mut bool,
    updates: &mut EngineUpdates,
    button: MouseButton,
    btn_state: ElementState,
) {
    match button {
        MouseButton::Left => {
            state.ui.left_click_down = match btn_state {
                ElementState::Pressed => {
                    if crate::region_select::begin(state) {
                        true
                    } else {
                        let handle = state
                            .ui
                            .cursor_pos
                            .and_then(|p| crate::gizmo::hit_test(state, scene, p));
                        if let Some(mol_manip::Handle::Camera(axis, positive)) = handle {
                            let direction = axis.vector() * if positive { -1. } else { 1. };
                            cam::reset_camera(state, scene, updates, direction);
                            state.ui.cam_snapshot = None;
                            false
                        } else {
                            mol_manip::begin_drag(state, scene, handle);
                            true
                        }
                    }
                }
                ElementState::Released => {
                    if !crate::region_select::finish(state, scene, redraw) {
                        mol_manip::finish_drag(state, redraw);
                    }
                    false
                }
            }
        }
        MouseButton::Right => match btn_state {
            ElementState::Pressed => match state.volatile.operating_mode {
                OperatingMode::Primary => {
                    if state.ui.inspection.picking {
                        if let Some(cursor) = state.ui.cursor_pos {
                            crate::inspection::pick_atom(state, scene, cursor);
                        }
                    } else {
                        selection::handle_selection_attempt(state, scene, redraw);
                    }
                }
                OperatingMode::MolEditor => {
                    selection::handle_selection_attempt_mol_editor(
                        state,
                        scene,
                        redraw_mol_editor,
                        updates,
                    );
                }
                OperatingMode::ProteinEditor => (),
            },
            ElementState::Released => (),
        },
        MouseButton::Middle => {
            // Allow mouse movement to move the camera on middle click.
            state.ui.middle_click_down = match btn_state {
                ElementState::Pressed => true,
                ElementState::Released => false,
            }
        }
        _ => (),
    }
}

fn cycle_primary_view(state: &mut State, redraw: &mut RedrawFlags, forward: bool) {
    let peptide_only_scene = state.volatile.active_mol.is_none()
        && !state.peptides.is_empty()
        && state.ligands.is_empty()
        && state.nucleic_acids.is_empty()
        && state.lipids.is_empty();

    if matches!(state.volatile.active_mol, Some((MolType::Peptide, _))) || peptide_only_scene {
        state.ui.mol_view_peptide = if forward {
            state.ui.mol_view_peptide.next()
        } else {
            state.ui.mol_view_peptide.prev()
        };
        crate::protein_alignment::clear_comparison_view(state);
        redraw.peptide = true;
    } else {
        state.ui.mol_view = if forward {
            state.ui.mol_view.next_non_peptide()
        } else {
            state.ui.mol_view.prev_non_peptide()
        };
        redraw.ligand = true;
        redraw.na = true;
        redraw.lipid = true;
        // Ligands etc. that are part of a peptide use this view too.
        redraw.peptide = !state.peptides.is_empty();
    }
}

/// Tab adds a carbon atom in the molecule editor, including while the pointer is over the GUI.
pub(crate) fn add_atom_with_tab(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let atom_sel_i = match &state.ui.selection {
        Selection::AtomLig((_, i)) => *i,
        Selection::AtomsLig((_, indices)) => {
            // todo: How should we handle multiple selected atoms?
            indices[0]
        }
        _ => return,
    };

    // todo: DRY with the Add Atom button.
    add_atom(
        &mut state.mol_editor.mol.common,
        &mut scene.entities,
        atom_sel_i,
        Carbon,
        BondType::Single,
        Some("c".to_owned()), // todo
        Some(1.4),            // todo
        None,
        &mut state.ui,
        updates,
        &mut scene.input_settings.control_scheme,
        state.volatile.mol_manip.mode,
        &state.mol_editor.mol.components,
    );
    state.mol_editor.mol.update_characterization();

    sync_md(state);

    // `add_atom` draws only the new atoms, but `sync_md` may reassign serial numbers, and the atom
    // labels need to reflect that.
    if state.mol_editor.show_atom_sns {
        mol_editor::redraw(
            &mut scene.entities,
            &state.mol_editor,
            &state.ui,
            state.volatile.mol_manip.mode,
            updates,
        );
    }
}

/// Handles keyboard input from either device, or window events.
fn handle_physical_key(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    redraw_in_place: &mut RedrawFlags,
    redraw_mol_editor: &mut bool,
    updates: &mut EngineUpdates,
    code: KeyCode,
    key_state: ElementState,
) -> bool {
    let op_mode = state.volatile.operating_mode;
    if key_state == ElementState::Pressed {
        let modifiers = state.volatile.key_modifiers.state();
        if modifiers.control_key() && matches!(code, KeyCode::KeyZ | KeyCode::KeyY) {
            crate::edit_history::undo_redo(
                state,
                scene,
                code == KeyCode::KeyY || modifiers.shift_key(),
                redraw,
                updates,
            );

            return true;
        }
        if state.volatile.region_selection.tool != crate::region_select::RegionTool::Off
            && code == KeyCode::Escape
        {
            crate::region_select::cancel(state, scene);

            return true;
        }
        if mol_manip::handle_key(state, scene, code, redraw, updates) {
            return true;
        }
        if state.volatile.mol_manip.mode != ManipMode::None
            && !matches!(
                code,
                KeyCode::KeyM
                    | KeyCode::KeyR
                    | KeyCode::ShiftLeft
                    | KeyCode::ShiftRight
                    | KeyCode::ControlLeft
                    | KeyCode::ControlRight
                    | KeyCode::AltLeft
                    | KeyCode::AltRight
            )
        {
            return true;
        }
        if op_mode == OperatingMode::Primary
            && matches!(code, KeyCode::KeyB | KeyCode::KeyL)
            && !modifiers.control_key()
        {
            crate::region_select::arm(
                state,
                scene,
                if code == KeyCode::KeyB {
                    crate::region_select::RegionTool::Box
                } else {
                    crate::region_select::RegionTool::Lasso
                },
            );

            return true;
        }
    }

    match key_state {
        ElementState::Pressed => {
            match code {
                KeyCode::ArrowLeft => {
                    cycle_selected(state, scene, true);

                    match op_mode {
                        OperatingMode::Primary => match state.ui.selection {
                            Selection::AtomPeptide(_)
                            | Selection::Residue(_)
                            | Selection::BondPeptide(_) => redraw.peptide = true,
                            Selection::AtomLig(_) | Selection::BondLig(_) => redraw.ligand = true,
                            Selection::AtomNucleicAcid(_) | Selection::BondNucleicAcid(_) => {
                                redraw.na = true
                            }
                            Selection::AtomLipid(_) | Selection::BondLipid(_) => {
                                redraw.lipid = true
                            }
                            Selection::AtomPocket(_) | Selection::BondPocket(_) => {
                                redraw.pocket = true
                            }
                            _ => (),
                        },
                        OperatingMode::MolEditor => *redraw_mol_editor = true,
                        OperatingMode::ProteinEditor => (),
                    }
                }
                KeyCode::ArrowRight => {
                    cycle_selected(state, scene, false);

                    match op_mode {
                        OperatingMode::Primary => match state.ui.selection {
                            Selection::AtomPeptide(_)
                            | Selection::Residue(_)
                            | Selection::BondPeptide(_) => redraw.peptide = true,
                            Selection::AtomLig(_) | Selection::BondLig(_) => redraw.ligand = true,
                            Selection::AtomNucleicAcid(_) | Selection::BondNucleicAcid(_) => {
                                redraw.na = true
                            }
                            Selection::AtomLipid(_) | Selection::BondLipid(_) => {
                                redraw.lipid = true
                            }
                            Selection::AtomPocket(_) | Selection::BondPocket(_) => {
                                redraw.pocket = true
                            }
                            _ => (),
                        },
                        OperatingMode::MolEditor => *redraw_mol_editor = true,
                        OperatingMode::ProteinEditor => (),
                    }
                }
                KeyCode::Escape => {
                    if state.ui.inspection.picking {
                        state.ui.inspection.picking = false;
                        state.ui.inspection.pending.clear();

                        return true;
                    }
                    // If in manip mode, exit that, but don't remove selections.
                    if matches!(
                        state.volatile.mol_manip.mode,
                        ManipMode::Move(_) | ManipMode::Rotate(_)
                    ) {
                        // Toggles it off, and performs cleanup.
                        set_manip(
                            state,
                            scene,
                            redraw,
                            redraw_mol_editor,
                            state.volatile.mol_manip.mode,
                            updates,
                        );
                    } else if state.ui.selection != Selection::None {
                        // First press: clear the selection, but keep the molecule active.
                        state.ui.selection = Selection::None;
                    } else {
                        // No selection: deactivate the molecule.
                        state.volatile.active_mol = None;
                    }

                    redraw.set_all();
                }
                KeyCode::Enter => {
                    move_cam_to_sel(state, &mut scene.camera, updates);
                }
                KeyCode::BracketLeft => match op_mode {
                    OperatingMode::Primary => {
                        cycle_primary_view(state, redraw, false);
                    }
                    OperatingMode::MolEditor => {
                        state.ui.mol_view = state.ui.mol_view.prev_editor();
                        *redraw_mol_editor = true;
                    }
                    OperatingMode::ProteinEditor => (),
                },
                KeyCode::BracketRight => match op_mode {
                    OperatingMode::Primary => {
                        cycle_primary_view(state, redraw, true);
                    }
                    OperatingMode::MolEditor => {
                        state.ui.mol_view = state.ui.mol_view.next_editor();
                        *redraw_mol_editor = true;
                    }
                    OperatingMode::ProteinEditor => (),
                },
                KeyCode::Semicolon | KeyCode::Quote => match op_mode {
                    OperatingMode::Primary => {
                        state.set_view_sel_level(state.ui.view_sel_level.cycle());

                        redraw.set_all();
                    }
                    // The molecule editor only uses atom/bond selection.
                    OperatingMode::MolEditor | OperatingMode::ProteinEditor => (),
                },
                KeyCode::KeyM => {
                    let mol_type = if op_mode == OperatingMode::MolEditor {
                        MolType::Ligand
                    } else {
                        match state.active_mol() {
                            Some(m) => m.mol_type(),
                            None => return true,
                        }
                    };

                    let mut rebuild_md_editor = false;

                    set_manip(
                        state,
                        scene,
                        redraw_in_place,
                        &mut rebuild_md_editor,
                        ManipMode::Move((mol_type, 0)),
                        updates,
                    );

                    if rebuild_md_editor {
                        sync_md(state);
                    }
                }
                KeyCode::KeyR => {
                    let mol_type = if op_mode == OperatingMode::MolEditor {
                        MolType::Ligand
                    } else {
                        match state.active_mol() {
                            Some(m) => m.mol_type(),
                            None => return true,
                        }
                    };

                    let mut rebuild_md_editor = false;

                    let mut skip = false;
                    if op_mode == OperatingMode::MolEditor
                        && let Selection::BondLig((_, i)) = state.ui.selection
                    {
                        let bond = &state.mol_editor.mol.common.bonds[i];
                        if bond.in_a_cycle(&state.mol_editor.mol.common.adjacency_list) {
                            skip = true;
                        }
                    }
                    if !skip {
                        set_manip(
                            state,
                            scene,
                            redraw_in_place,
                            &mut rebuild_md_editor,
                            ManipMode::Rotate((mol_type, 0)),
                            updates,
                        );
                    }

                    if rebuild_md_editor {
                        sync_md(state);
                    }
                }
                KeyCode::Delete => {
                    match op_mode {
                        OperatingMode::Primary => {
                            // Close the active mol?
                            if let Some((mol_type, i)) = state.volatile.active_mol {
                                close_mol(mol_type, i, state, scene, redraw, updates);
                            }
                        }
                        OperatingMode::MolEditor => {
                            // Delete the selected atom.

                            if let Selection::AtomLig((_, i)) = state.ui.selection {
                                state.mol_editor.remove_atom(i);
                                *redraw_mol_editor = true;

                                sync_md(state);
                            }
                        }
                        OperatingMode::ProteinEditor => (),
                    }
                }
                KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                    state.volatile.inputs_commanded.run = true;
                }
                _ => (),
            }
        }
        ElementState::Released => match code {
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                state.volatile.inputs_commanded.run = false;
            }
            _ => (),
        },
    }

    // todo: If you enable a direction-dependent flashlight, you will need to modify the mouse movement
    // todo state too.
    // Similar code to the engine. Update state so we can update the flashlight while a key is held,
    // and reset snapshots.
    match key_state {
        ElementState::Pressed => match code {
            // Check the cases for the engine's built-in movement commands, to set the current-snapshot to None.
            // C+P partially, from `graphics`. These are for press and release.
            KeyCode::KeyW => {
                state.volatile.inputs_commanded.fwd = true;
            }
            KeyCode::KeyS => {
                state.volatile.inputs_commanded.back = true;
            }
            KeyCode::KeyA => {
                state.volatile.inputs_commanded.left = true;
            }
            KeyCode::KeyD => {
                state.volatile.inputs_commanded.right = true;
            }
            KeyCode::Space => {
                state.volatile.inputs_commanded.up = true;
            }
            KeyCode::KeyC => {
                state.volatile.inputs_commanded.down = true;
            }
            KeyCode::KeyQ => {
                state.volatile.inputs_commanded.roll_ccw = true;
            }
            KeyCode::KeyE => {
                state.volatile.inputs_commanded.roll_cw = true;
            }
            _ => (),
        },

        ElementState::Released => match code {
            KeyCode::KeyW => {
                state.volatile.inputs_commanded.fwd = false;
            }
            KeyCode::KeyS => {
                state.volatile.inputs_commanded.back = false;
            }
            KeyCode::KeyA => {
                state.volatile.inputs_commanded.left = false;
            }
            KeyCode::KeyD => {
                state.volatile.inputs_commanded.right = false;
            }
            KeyCode::Space => {
                state.volatile.inputs_commanded.up = false;
            }
            KeyCode::KeyC => {
                state.volatile.inputs_commanded.down = false;
            }
            KeyCode::KeyQ => {
                state.volatile.inputs_commanded.roll_ccw = false;
            }
            KeyCode::KeyE => {
                state.volatile.inputs_commanded.roll_cw = false;
            }
            _ => (),
        },
    }

    false
}

fn handle_scroll(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw_in_place: &mut RedrawFlags,
    delta: MouseScrollDelta,
) -> bool {
    if state.volatile.mol_manip.mode != ManipMode::None {
        mol_manip::handle_mol_manip_in_out(state, scene, delta, redraw_in_place, updates);

        return false;
    }
    let scroll = match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.,
    };

    // I'm not sure about the polarity here, but the intent is just to trigger `inputs_present()`
    // to be true.
    if scroll > 0. {
        state.volatile.inputs_commanded.scroll_up = true;
    } else {
        state.volatile.inputs_commanded.scroll_down = true;
    }

    if state.volatile.key_modifiers.state().control_key() {
        if let DepthMode::Manual((_, far)) = &mut state.to_save.depth_mode {
            *far = (*far as i32 + (scroll * 5.) as i32)
                .clamp(FOG_DIST_MIN as i32, FOG_DIST_MAX as i32) as u16;
            cam::set_fog(state, &mut scene.camera);
            updates.camera = true;
        }

        // Counteract the engine's default free look behavior. This is indirect, but good
        // enough for now.
        if matches!(
            scene.input_settings.control_scheme,
            ControlScheme::FreeCamera | ControlScheme::Arc { center: _ }
        ) {
            let fwd = scene.camera.orientation.rotate_vec(FWD_VEC);
            scene.camera.position += fwd * -scroll * SCROLL_MOVE_AMT;
            updates.camera = true;
        }

        return true;
    }

    set_flashlight(scene);
    updates.lighting = true;

    mol_manip::handle_mol_manip_in_out(state, scene, delta, redraw_in_place, updates);

    false
}

/// Run this after handling an event.
fn post_event_cleanup(
    state: &mut State,
    scene: &mut Scene,
    redraw: &RedrawFlags,
    redraw_in_place: &RedrawFlags,
    redraw_mol_editor: bool,
    updates: &mut EngineUpdates,
) {
    let op_mode = state.volatile.operating_mode;

    if redraw.peptide && op_mode == OperatingMode::Primary {
        peptide::draw_peptide(state, scene, updates);
    }

    if redraw.ligand {
        match op_mode {
            OperatingMode::Primary => wrappers::draw_all_ligs(state, scene, updates),

            OperatingMode::MolEditor => {
                mol_editor::redraw(
                    &mut scene.entities,
                    &state.mol_editor,
                    &state.ui,
                    state.volatile.mol_manip.mode,
                    updates,
                );
            }
            OperatingMode::ProteinEditor => (),
        }
    }

    if redraw.na && op_mode == OperatingMode::Primary {
        wrappers::draw_all_nucleic_acids(state, scene, updates);
    }

    if redraw.lipid && op_mode == OperatingMode::Primary {
        wrappers::draw_all_lipids(state, scene, updates);
    }

    if redraw.pocket && op_mode == OperatingMode::Primary {
        wrappers::draw_all_pockets(state, scene, updates);
    }

    if redraw_in_place.peptide && op_mode == OperatingMode::Primary {
        peptide::draw_peptide(state, scene, updates);
    }

    if redraw_in_place.ligand {
        match op_mode {
            OperatingMode::Primary => {
                redraw_inplace_helper(MolType::Ligand, state, scene, updates);
            }
            OperatingMode::MolEditor => {
                mol_editor::redraw(
                    &mut scene.entities,
                    &state.mol_editor,
                    &state.ui,
                    state.volatile.mol_manip.mode,
                    updates,
                );
            }
            OperatingMode::ProteinEditor => (),
        }
    }

    if redraw_in_place.na && op_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::NucleicAcid, state, scene, updates);
    }

    if redraw_in_place.lipid && op_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::Lipid, state, scene, updates);
    }

    if redraw_in_place.pocket && op_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::Pocket, state, scene, updates);
        updates.entities.push_class(EntityClass::Pocket as u32);
    }

    if redraw_in_place.pocket && op_mode == OperatingMode::MolEditor {
        if let Some(pocket) = &state.mol_editor.mol.pharmacophore.pocket {
            scene
                .entities
                .retain(|e| e.class != EntityClass::Pocket as u32);

            scene.entities.extend(draw_pocket(
                pocket,
                &state.mol_editor.h_bonds,
                &state.mol_editor.mol.common.atom_posits,
                &state.ui.visibility,
                &state.ui.selection,
                // Don't draw the mesh; we redraw it once manipulation is complete.
                // This prevents the original mesh position from being confusing or a
                // visual blocker.
                &state.volatile.mol_manip.mode,
            ));
        }

        updates.entities.push_class(EntityClass::Pocket as u32);
    }

    if redraw_mol_editor {
        mol_editor::redraw(
            &mut scene.entities,
            &state.mol_editor,
            &state.ui,
            state.volatile.mol_manip.mode,
            updates,
        );
    }

    // We handle the flashlight elsewhere, as this event handler only fires upon events; not while
    // a key is held.
    if state.volatile.inputs_commanded.inputs_present() {
        state.ui.cam_snapshot = None;

        // Update fog dynamically as the camera moves, on a ratio for performance.
        // find_nearest_mol_dist_to_cam already subsamples the protein, so this is cheap.
        if state.to_save.depth_mode == DepthMode::Auto {
            static mut I_FOG: u32 = 0;
            const FOG_RATIO: u32 = 6;

            unsafe {
                I_FOG += 1;
                if I_FOG.is_multiple_of(FOG_RATIO)
                    || state.volatile.inputs_commanded.scroll_down
                    || state.volatile.inputs_commanded.scroll_up
                {
                    cam::set_fog(state, &mut scene.camera);
                    updates.camera = true;
                }
            }
        }
    }
}
