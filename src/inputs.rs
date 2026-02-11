//! Handles user inputs, e.g. from keyboard and mouse.

use bio_files::BondType;
use graphics::{
    ControlScheme, DeviceEvent, ElementState, EngineUpdates, EntityUpdate, FWD_VEC, Scene,
    WindowEvent,
    event::{MouseButton, MouseScrollDelta},
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::f32::Vec3;
use na_seq::Element::Carbon;

use crate::{
    cam::{FOG_DIST_MIN, move_cam_to_sel, set_fog_dist},
    drawing,
    drawing::{EntityClass, wrappers},
    mol_editor,
    mol_editor::{add_atoms::add_atom, sync_md},
    mol_manip,
    mol_manip::ManipMode,
    molecules::MolType,
    render::{MESH_POCKET, set_flashlight},
    selection,
    selection::Selection,
    state::{OperatingMode, State},
    util::{RedrawFlags, close_mol, cycle_selected},
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
    let mut redraw_in_place = RedrawFlags::default();
    let redraw_mol_editor = false;

    match event {
        // Move the camera forward and back on scroll; handled by Graphics cam controls.
        // DeviceEvent::MouseWheel { delta } => {
        //     if handle_scroll(state_, scene, &mut updates, &mut redraw_in_place, delta) {
        //         return updates;
        //     }
        // }
        // DeviceEvent::Button { button, state } => {
        //     #[cfg(target_os = "linux")]
        //     let button_ = match button {
        //         1 => MouseButton::Left,
        //         3 => MouseButton::Right,
        //         2 => MouseButton::Middle,
        //         _ => MouseButton::Other(0), // Placeholder.
        //     };
        //     #[cfg(not(target_os = "linux"))]
        //     let button_ = match button {
        //         0 => MouseButton::Left,
        //         1 => MouseButton::Right,
        //         2 => MouseButton::Middle,
        //         _ => MouseButton::Other(0), // Placeholder.
        //     };
        //     handle_mouse_button(
        //         state_,
        //         scene,
        //         &mut redraw,
        //         &mut redraw_mol_editor,
        //         &mut updates,
        //         button_,
        //         state,
        //     )
        // }
        // DeviceEvent::Key(key) => {
        //     if let Code(key_code) = key.physical_key {
        //         if handle_physical_key(
        //             state_,
        //             scene,
        //             &mut redraw,
        //             &mut redraw_in_place,
        //             &mut redraw_mol_editor,
        //             &mut updates,
        //             key_code,
        //             key.state,
        //         ) {
        //             return updates;
        //         }
        //     };
        // }
        DeviceEvent::MouseMotion { delta } => {
            // Free look handled by the engine; handle middle-click-move here.
            if state_.ui.middle_click_down {
                // The same movement sensitivity scaler we use for the (1x effective multiplier)
                // on keyboard movement seems to work well enough here.
                let mut movement_vec = Vec3::new(
                    delta.0 as f32 * MOVEMENT_SENS * dt,
                    -delta.1 as f32 * MOVEMENT_SENS * dt,
                    0.,
                );

                if scene.input_settings.control_scheme != ControlScheme::FreeCamera {
                    movement_vec *= -1.;
                }

                scene.camera.position += scene.camera.orientation.rotate_vec(movement_vec);
                updates.camera = true;

                set_flashlight(scene);
                updates.lighting = true;
            }

            if state_.ui.left_click_down {
                mol_manip::handle_mol_manip_in_plane(state_, scene, delta, &mut redraw_in_place);

                set_flashlight(scene);
                updates.lighting = true;

                // todo: Experimenting
                // unsafe {
                //     I_FIND_NEAREST += 1;
                //     if I_FIND_NEAREST.is_multiple_of(RATIO_FIND_NEAREST) {
                //         state_.volatile.nearest_mol_dist_to_cam =
                //             find_nearest_mol_dist_to_cam(state_, &scene.camera);
                //
                //         if let Some(dist) = state_.volatile.nearest_mol_dist_to_cam {
                //
                //             let dist = max(dist as u16 + FOG_HALF_DEPTH/2 - 5, 5);
                //             set_fog_dist(&mut scene.camera, dist);
                //         }
                //     }
                // }
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

    let mut redraw = RedrawFlags::default();
    let mut redraw_in_place = RedrawFlags::default();
    let mut redraw_mol_editor = false;

    match event {
        WindowEvent::CursorMoved {
            device_id: _,
            position,
        } => state_.ui.cursor_pos = Some((position.x as f32, position.y as f32)),
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
                if handle_physical_key(
                    state_,
                    scene,
                    &mut redraw,
                    &mut redraw_in_place,
                    &mut redraw_mol_editor,
                    &mut updates,
                    code,
                    event.state,
                ) {
                    return updates;
                }
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
    // todo: Fragile? Error handle.
    let mol_i = state.volatile.active_mol.unwrap().1;

    let err = "Uhoh: Index error on in-place redraw";

    let mol = match mol_type {
        MolType::Peptide => {
            unimplemented!()
            // todo: A/R
            // wrappers::update_single_peptide_inplace(mol_i, state, scene);
            //
            // if mol_i >= state.ligands.len() {
            //     eprintln!("{err}");
            //     wrappers::draw_all_ligs(state, scene);
            //     return;
            // }
            //
            // &mut state.ligands[mol_i].common
        }
        MolType::Ligand => {
            wrappers::update_single_ligand_inplace(mol_i, state, scene);

            if mol_i >= state.ligands.len() {
                eprintln!("{err}");
                wrappers::draw_all_ligs(state, scene);
                return;
            }

            &mut state.ligands[mol_i].common
        }
        MolType::NucleicAcid => {
            wrappers::update_single_nucleic_acid_inplace(mol_i, state, scene);

            if mol_i >= state.nucleic_acids.len() {
                eprintln!("{err}");
                wrappers::draw_all_nucleic_acids(state, scene);
                return;
            }

            &mut state.nucleic_acids[mol_i].common
        }
        MolType::Lipid => {
            wrappers::update_single_lipid_inplace(mol_i, state, scene);

            if mol_i >= state.lipids.len() {
                eprintln!("{err}");
                wrappers::draw_all_lipids(state, scene);
                return;
            }

            &mut state.lipids[mol_i].common
        }
        MolType::Pocket => {
            wrappers::update_single_pocket_inplace(mol_i, state, scene);

            if mol_i >= state.pockets.len() {
                eprintln!("{err}");
                wrappers::draw_all_pockets(state, scene);
                return;
            }

            &mut state.pockets[mol_i].common
        }
        MolType::Water => unreachable!(),
    };

    updates.entities = match mol.entity_i_range {
        Some(range) => EntityUpdate::Indexes(range),
        None => {
            eprintln!("Error: Missing entity index range.");
            EntityUpdate::All
        }
    };
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
                ElementState::Pressed => true,
                ElementState::Released => {
                    // Part of our move logic.
                    state.volatile.mol_manip.pivot = None;
                    false
                }
            }
        }
        MouseButton::Right => match btn_state {
            ElementState::Pressed => match state.volatile.operating_mode {
                OperatingMode::Primary => {
                    selection::handle_selection_attempt(state, scene, redraw);
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
    match key_state {
        ElementState::Pressed => {
            match code {
                KeyCode::ArrowLeft => {
                    cycle_selected(state, scene, true);

                    match state.volatile.operating_mode {
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

                    match state.volatile.operating_mode {
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
                    // If in manip mode, exit that, but don't remove selections.
                    if matches!(
                        state.volatile.mol_manip.mode,
                        ManipMode::Move(_) | ManipMode::Rotate(_)
                    ) {
                        // Exit manip mode.
                        state.volatile.mol_manip.mode = ManipMode::None;
                        state.volatile.mol_manip.pivot = None;
                        scene.input_settings.control_scheme = state.volatile.control_scheme_prev;

                        // Clean up things, e.g. for moving the pocket to command its mesh
                        // to rebuild. Note required for most other things.
                        // c+p from set_manip to regen the pocket mesh and volume if exiting
                        // manip through the esc key.
                        if let Some((mol_type, i)) = state.volatile.active_mol
                            && mol_type == MolType::Pocket
                        {
                            state.pockets[i].regen_mesh_vol();
                            scene.meshes[MESH_POCKET] = state.pockets[i].surface_mesh.clone();

                            updates.meshes = true;
                            redraw.pocket = true;
                        }

                        if state.volatile.operating_mode == OperatingMode::MolEditor {
                            sync_md(state);
                        }
                    } else {
                        // Unselect everything.
                        state.ui.selection = Selection::None;
                        state.volatile.active_mol = None;
                    }

                    redraw.set_all();
                }
                KeyCode::Enter => {
                    move_cam_to_sel(
                        &mut state.ui,
                        &state.peptide,
                        &state.ligands,
                        &state.nucleic_acids,
                        &state.lipids,
                        &state.pockets,
                        &mut scene.camera,
                        updates,
                    );
                }
                KeyCode::BracketLeft => match state.volatile.operating_mode {
                    OperatingMode::Primary => {
                        state.ui.mol_view = state.ui.mol_view.prev();

                        redraw.set_all();
                    }
                    OperatingMode::MolEditor => {
                        state.ui.mol_view = state.ui.mol_view.prev_editor();
                        *redraw_mol_editor = true;
                    }
                    OperatingMode::ProteinEditor => (),
                },
                KeyCode::BracketRight => match state.volatile.operating_mode {
                    OperatingMode::Primary => {
                        state.ui.mol_view = state.ui.mol_view.next();

                        redraw.set_all();
                    }
                    OperatingMode::MolEditor => {
                        state.ui.mol_view = state.ui.mol_view.next_editor();
                        *redraw_mol_editor = true;
                    }
                    OperatingMode::ProteinEditor => (),
                },
                KeyCode::Semicolon => match state.volatile.operating_mode {
                    OperatingMode::Primary => {
                        state.ui.view_sel_level = state.ui.view_sel_level.prev();

                        redraw.set_all();
                    }
                    OperatingMode::MolEditor => {
                        state.ui.view_sel_level = state.ui.view_sel_level.prev();
                        *redraw_mol_editor = true;
                    }
                    OperatingMode::ProteinEditor => (),
                },
                KeyCode::Quote => match state.volatile.operating_mode {
                    OperatingMode::Primary => {
                        state.ui.view_sel_level = state.ui.view_sel_level.next();

                        redraw.set_all();
                    }
                    OperatingMode::MolEditor => {
                        state.ui.view_sel_level = state.ui.view_sel_level.next();
                        *redraw_mol_editor = true;
                    }
                    OperatingMode::ProteinEditor => (),
                },
                KeyCode::KeyM => {
                    let mol_type = match state.active_mol() {
                        Some(m) => m.mol_type(),
                        None => return true,
                    };

                    let mut rebuild_md_editor = false;
                    mol_manip::set_manip(
                        &mut state.volatile,
                        &mut state.pockets,
                        &mut state.to_save.save_flag,
                        scene,
                        redraw,
                        &mut rebuild_md_editor,
                        ManipMode::Move((mol_type, 0)),
                        &state.ui.selection,
                        updates,
                    );

                    if rebuild_md_editor {
                        sync_md(state);
                    }
                }
                KeyCode::KeyR => {
                    let mol_type = match state.active_mol() {
                        Some(m) => m.mol_type(),
                        None => return true,
                    };

                    let mut rebuild_md_editor = false;

                    let mut skip = false;
                    if state.volatile.operating_mode == OperatingMode::MolEditor {
                        if let Selection::BondLig((_, i)) = state.ui.selection {
                            let bond = &state.mol_editor.mol.common.bonds[i];
                            if bond.in_a_cycle(&state.mol_editor.mol.common.adjacency_list) {
                                skip = true;
                            }
                        }
                    }
                    if !skip {
                        mol_manip::set_manip(
                            &mut state.volatile,
                            &mut state.pockets,
                            &mut state.to_save.save_flag,
                            scene,
                            redraw_in_place,
                            &mut rebuild_md_editor,
                            ManipMode::Rotate((mol_type, 0)),
                            &state.ui.selection,
                            updates,
                        );
                    }

                    if rebuild_md_editor {
                        sync_md(state);
                    }
                }
                KeyCode::Delete => {
                    match state.volatile.operating_mode {
                        OperatingMode::Primary => {
                            // Close the active mol?
                            if let Some((mol_type, i)) = state.volatile.active_mol {
                                close_mol(mol_type, i, state, scene, updates);
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
                KeyCode::Tab => {
                    // todo: This is DRY/mostly C+P from the add atom button.
                    if state.volatile.operating_mode == OperatingMode::MolEditor {
                        let (_mol_i, atom_sel_i) = match &state.ui.selection {
                            Selection::AtomLig((mol_i, i)) => (*mol_i, *i),
                            Selection::AtomsLig((mol_i, i)) => {
                                // todo: How should we handle this?
                                (*mol_i, i[0])
                            }
                            _ => return true,
                        };

                        // todo: DRY here in some of the params with the button
                        add_atom(
                            &mut state.mol_editor.mol.common,
                            &mut scene.entities,
                            atom_sel_i,
                            Carbon,
                            BondType::Single,
                            Some("c".to_owned()), // todo
                            Some(1.4),            // todo
                            0.13,                 // todo
                            &mut state.ui,
                            updates,
                            &mut scene.input_settings.control_scheme,
                            state.volatile.mol_manip.mode,
                        );
                        state.mol_editor.mol.update_characterization();

                        sync_md(state);
                    }
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
    if state.volatile.key_modifiers.state().control_key() {
        let scroll: f32 = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
        };

        state.ui.view_depth.1 = (state.ui.view_depth.1 as i16 + (scroll * 5.) as i16) as u16;

        // Overflowed from subtraction.
        if state.ui.view_depth.1 > 2_000 {
            state.ui.view_depth.1 = FOG_DIST_MIN;
        }

        set_fog_dist(&mut scene.camera, state.ui.view_depth.1);

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

    mol_manip::handle_mol_manip_in_out(state, scene, delta, redraw_in_place);

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
    if redraw.peptide && state.volatile.operating_mode == OperatingMode::Primary {
        // todo:This is overkill for certain keys. Just change the color of the one[s] in question, and set update.entities = true.
        drawing::draw_peptide(state, scene);
        updates.entities = EntityUpdate::All;
        // updates.entities.push_class(EntityClass::Peptide as u32);
    }

    if redraw.ligand {
        match state.volatile.operating_mode {
            OperatingMode::Primary => wrappers::draw_all_ligs(state, scene),

            OperatingMode::MolEditor => {
                mol_editor::redraw(
                    &mut scene.entities,
                    &state.mol_editor.mol,
                    &state.mol_editor.pocket,
                    &state.mol_editor.h_bonds,
                    &state.ui,
                    state.volatile.mol_manip.mode,
                    state.ligands.len(),
                );
            }
            OperatingMode::ProteinEditor => (),
        }
        updates.entities = EntityUpdate::All;
    }

    if redraw.na && state.volatile.operating_mode == OperatingMode::Primary {
        wrappers::draw_all_nucleic_acids(state, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw.lipid && state.volatile.operating_mode == OperatingMode::Primary {
        wrappers::draw_all_lipids(state, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw.pocket && state.volatile.operating_mode == OperatingMode::Primary {
        wrappers::draw_all_pockets(state, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw_in_place.ligand {
        match state.volatile.operating_mode {
            OperatingMode::Primary => {
                redraw_inplace_helper(MolType::Ligand, state, scene, updates);
            }
            OperatingMode::MolEditor => {
                mol_editor::redraw(
                    &mut scene.entities,
                    &state.mol_editor.mol,
                    &state.mol_editor.pocket,
                    &state.mol_editor.h_bonds,
                    &state.ui,
                    state.volatile.mol_manip.mode,
                    state.ligands.len(),
                );
                updates.entities.push_class(EntityClass::Ligand as u32);
            }
            OperatingMode::ProteinEditor => (),
        }
    }

    if redraw_in_place.na && state.volatile.operating_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::NucleicAcid, state, scene, updates);
    }

    if redraw_in_place.lipid && state.volatile.operating_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::Lipid, state, scene, updates);
    }

    if redraw_in_place.pocket && state.volatile.operating_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::Pocket, state, scene, updates);
        // wrappers::draw_all_pockets(state, scene);
        updates.entities.push_class(EntityClass::Pocket as u32);
    }

    if redraw_mol_editor {
        mol_editor::redraw(
            &mut scene.entities,
            &state.mol_editor.mol,
            &state.mol_editor.pocket,
            &state.mol_editor.h_bonds,
            &state.ui,
            state.volatile.mol_manip.mode,
            state.ligands.len(),
        );
        updates.entities = EntityUpdate::All;
    }

    // We handle the flashlight elsewhere, as this event handler only fires upon events; not while
    // a key is held.
    if state.volatile.inputs_commanded.inputs_present() {
        state.ui.cam_snapshot = None;

        // todo: Experimenting
        // unsafe {
        //     I_FIND_NEAREST += 1;
        //     if I_FIND_NEAREST.is_multiple_of(RATIO_FIND_NEAREST) {
        //         state_.volatile.nearest_mol_dist_to_cam =
        //             find_nearest_mol_dist_to_cam(state_, &scene.camera);
        //
        //         if let Some(dist) = state_.volatile.nearest_mol_dist_to_cam {
        //             let dist = max(dist as u16 + FOG_HALF_DEPTH/2 - 5, 5);
        //             set_fog_dist(&mut scene.camera, dist);
        //         }
        //     }
        // }
    }
}
