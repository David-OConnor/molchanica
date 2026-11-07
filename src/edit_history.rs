//! Shared editing history.

use egui::{DragValue, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::small::MoleculeSmall;

use crate::{
    mol_manip::{self, ManipMode, Pose, TransformValues},
    selection::{Selection, ViewSelLevel},
    state::{OperatingMode, State},
    util::RedrawFlags,
};

const HISTORY_LIMIT: usize = 64;
const HISTORY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SelectionState {
    selection: Selection,
    active: Option<(mol_defs::molecules::MolType, usize)>,
    peptide: Option<usize>,
    level: ViewSelLevel,
    operating_mode: OperatingMode,
    objects: Vec<(String, usize, u32)>,
}

impl SelectionState {
    fn memory_bytes(&self) -> usize {
        let indices = match &self.selection {
            Selection::Atoms(objects) => objects.iter().map(|o| o.atoms.len()).sum(),
            Selection::Residues(indices)
            | Selection::AtomsPeptide(indices)
            | Selection::AtomsLig((_, indices))
            | Selection::BondsLig((_, indices)) => indices.len(),
            _ => 0,
        };

        indices * std::mem::size_of::<usize>()
            + self
                .objects
                .iter()
                .map(|(name, _, _)| name.len() + 32)
                .sum::<usize>()
    }

    pub(crate) fn capture(state: &State) -> Self {
        Self {
            selection: state.ui.selection.clone(),
            active: state.volatile.active_mol,
            peptide: state.volatile.active_peptide,
            level: state.ui.view_sel_level,
            operating_mode: state.volatile.operating_mode,
            objects: crate::selection::expression::objects(state)
                .iter()
                .map(|(_, _, m)| {
                    let m = m.common();

                    (m.ident.clone(), m.atoms.len(), m.next_atom_sn)
                })
                .collect(),
        }
    }

    fn restore(&self, state: &mut State) {
        state.ui.selection = self.selection.clone();
        state.volatile.active_mol = self.active;
        state.volatile.active_peptide = self.peptide;
        state.ui.view_sel_level = self.level;
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Operation {
    Pose {
        before: Pose,
        after: Pose,
        label: &'static str,
    },
    Editor {
        before: Box<EditorSnapshot>,
        after: Box<EditorSnapshot>,
    },
    Transform {
        before: Pose,
        after: Pose,
        values: TransformValues,
    },
    Selection {
        before: SelectionState,
        after: SelectionState,
        neighborhood: Option<(String, f64)>,
    },
}

impl Operation {
    fn memory_bytes(&self) -> usize {
        match self {
            Self::Transform { before, after, .. } | Self::Pose { before, after, .. } => {
                before.memory_bytes() + after.memory_bytes()
            }
            Self::Editor { before, after } => {
                (before.mol.common.atoms.len() + after.mol.common.atoms.len()) * 512
            }
            Self::Selection { before, after, .. } => before.memory_bytes() + after.memory_bytes(),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Pose { label, .. } => label,
            Self::Editor { .. } => "Edit molecule",
            Self::Transform { before, .. } => match before.mode {
                ManipMode::Move(_) => "Move",
                _ => "Rotate",
            },
            Self::Selection {
                neighborhood: Some(_),
                ..
            } => "Select neighborhood",
            Self::Selection { .. } => "Selection",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EditorSnapshot {
    mol: MoleculeSmall,
    selection: SelectionState,
    signature: String,
    rotatable_bonds: Vec<usize>,
    mol_i_in_state: Option<usize>,
    pocket_i_in_state: Option<usize>,
}

impl EditorSnapshot {
    fn signature(state: &State) -> String {
        let common = &state.mol_editor.mol.common;

        format!(
            "{:?}{:?}{:?}",
            common.atoms, common.bonds, common.atom_posits
        )
    }

    pub(crate) fn capture(state: &State) -> Option<Self> {
        if state.volatile.operating_mode != OperatingMode::MolEditor
            || state.mol_editor.md.running
            || state.volatile.mol_manip.mode != ManipMode::None
        {
            return None;
        }

        Some(Self {
            mol: state.mol_editor.mol.clone(),
            selection: SelectionState::capture(state),
            signature: Self::signature(state),
            rotatable_bonds: state.mol_editor.rotatable_bonds.clone(),
            mol_i_in_state: state.mol_editor.mol_i_in_state,
            pocket_i_in_state: state.mol_editor.pocket_i_in_state,
        })
    }

    fn restore(&self, state: &mut State, redraw: &mut RedrawFlags) {
        state.mol_editor.mol = self.mol.clone();
        self.selection.restore(state);
        state.mol_editor.selected_comp = None;
        state
            .mol_editor
            .rotatable_bonds
            .clone_from(&self.rotatable_bonds);
        state.mol_editor.mol_i_in_state = self.mol_i_in_state;
        state.mol_editor.pocket_i_in_state = self.pocket_i_in_state;
        crate::mol_editor::sync_md(state);
        state.mol_editor.update_h_bonds();
        redraw.ligand = true;
        redraw.pocket = true;
    }
}

pub(crate) fn record_editor(state: &mut State, before: Option<EditorSnapshot>, revision: u64) {
    if revision != state.volatile.history.revision {
        return;
    }
    if let Some(before) = before
        && state.volatile.operating_mode == OperatingMode::MolEditor
        && before.signature != EditorSnapshot::signature(state)
        && let Some(after) = EditorSnapshot::capture(state)
    {
        state.volatile.history.push(Operation::Editor {
            before: Box::new(before),
            after: Box::new(after),
        });
    }
}

#[derive(Default)]
pub(crate) struct EditHistory {
    undo: Vec<Operation>,
    redo: Vec<Operation>,
    pub(crate) revision: u64,
    message: String,
}

impl EditHistory {
    pub(crate) fn push(&mut self, operation: Operation) {
        self.undo.push(operation);
        while self.undo.len() > 1
            && (self.undo.len() > HISTORY_LIMIT
                || self.undo.iter().map(Operation::memory_bytes).sum::<usize>() > HISTORY_BYTES)
        {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.revision = self.revision.wrapping_add(1);
        self.message.clear();
    }
}

pub(crate) fn record_selection(state: &mut State, before: SelectionState, revision: u64) {
    let after = SelectionState::capture(state);
    if before.operating_mode != after.operating_mode {
        return;
    }
    if before.objects != after.objects {
        state.volatile.history.undo.clear();
        state.volatile.history.redo.clear();
        state.volatile.history.revision = state.volatile.history.revision.wrapping_add(1);

        return;
    }

    if before != after && revision == state.volatile.history.revision {
        state.volatile.history.push(Operation::Selection {
            before,
            after,
            neighborhood: None,
        });
    }
}

pub(crate) fn undo_redo(
    state: &mut State,
    scene: &mut Scene,
    redo: bool,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    if state.volatile.md_local.running || state.mol_editor.md.running {
        return;
    }
    if state.volatile.mol_manip.mode != ManipMode::None {
        let cancel = mol_manip::has_preview(state);
        mol_manip::end_transform(state, scene, cancel, redraw, updates);
        if cancel {
            return;
        }
    }
    let mut history = std::mem::take(&mut state.volatile.history);
    let operation = if redo {
        history.redo.pop()
    } else {
        history.undo.pop()
    };
    let Some(operation) = operation else {
        state.volatile.history = history;

        return;
    };
    let operating_mode = match &operation {
        Operation::Transform { before, .. } | Operation::Pose { before, .. } => {
            before.operating_mode
        }
        Operation::Editor { .. } => OperatingMode::MolEditor,
        Operation::Selection { before, .. } => before.operating_mode,
    };
    if state.volatile.operating_mode != operating_mode {
        if redo {
            history.redo.push(operation);
        } else {
            history.undo.push(operation);
        }
        let view = match operating_mode {
            OperatingMode::Primary => "the main molecule view",
            OperatingMode::MolEditor => "the molecule editor",
            OperatingMode::ProteinEditor => "the protein editor",
        };
        history.message = format!("Return to {view} to use this history entry.");
        state.volatile.history = history;

        return;
    }
    let valid = match &operation {
        Operation::Transform { before, after, .. } | Operation::Pose { before, after, .. } => {
            let (expected, target) = if redo {
                (before, after)
            } else {
                (after, before)
            };
            if expected.matches(state) {
                mol_manip::restore_pose(state, target, redraw);
                mol_manip::refresh_after_history(state, scene, target.mode, redraw, updates);
                true
            } else {
                false
            }
        }
        Operation::Editor { before, after } => {
            let (expected, target) = if redo {
                (before, after)
            } else {
                (after, before)
            };
            if state.volatile.operating_mode == OperatingMode::MolEditor
                && EditorSnapshot::signature(state) == expected.signature
            {
                target.restore(state, redraw);
                true
            } else {
                false
            }
        }
        Operation::Selection { before, after, .. } => {
            let (expected, target) = if redo {
                (before, after)
            } else {
                (after, before)
            };
            if SelectionState::capture(state) == *expected {
                target.restore(state);
                redraw.set_all();
                true
            } else {
                false
            }
        }
    };
    if valid {
        if redo {
            history.undo.push(operation);
        } else {
            history.redo.push(operation);
        }
        history.message.clear();
        let mobile = state
            .ui
            .protein_comparison
            .result
            .as_ref()
            .and_then(|result| crate::protein_alignment::resolve(state, &result.mobile.key).ok());
        if let Some(result) = &mut state.ui.protein_comparison.result
            && let Some(peptide) = mobile.and_then(|i| state.peptides.get(i))
        {
            if peptide.common.atom_posits == result.transformed {
                result.applied = true;
            } else if peptide.common.atom_posits == result.mobile.positions {
                result.applied = false;
            }
        }
    } else {
        history.undo.clear();
        history.redo.clear();
        history.message = "History cleared: the target changed outside this operation.".into();
    }
    history.revision = history.revision.wrapping_add(1);
    state.volatile.history = history;
}

pub(crate) fn neighborhood(state: &mut State, radius: f64, redraw: &mut RedrawFlags) {
    let seed = crate::selection::expression::snapshot(
        state,
        &crate::selection::expression::current_atoms(state),
    );
    let before = SelectionState::capture(state);
    if let Ok(atoms) = crate::selection::expression::evaluate(
        state,
        &format!("byres (all within {radius} of ({seed}))"),
    ) {
        state.ui.selection = Selection::Atoms(atoms);
        let after = SelectionState::capture(state);
        state.volatile.history.push(Operation::Selection {
            before,
            after,
            neighborhood: Some((seed, radius)),
        });
        redraw.set_all();
    }
}

fn adjust(
    state: &mut State,
    scene: &mut Scene,
    mut operation: Operation,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    if state.volatile.mol_manip.mode != ManipMode::None {
        mol_manip::end_transform(state, scene, false, redraw, updates);
    }
    let valid = match &mut operation {
        Operation::Transform {
            before,
            after,
            values,
        } => {
            if !after.matches(state) {
                false
            } else {
                mol_manip::restore_pose(state, before, redraw);
                values.rotation =
                    lin_alg::f64::Quaternion::from_axis_angle(values.axis, values.angle);
                mol_manip::apply_values(state, before, values);
                *after = Pose::capture(state, before.mode).unwrap();
                mol_manip::refresh_after_history(state, scene, before.mode, redraw, updates);
                true
            }
        }
        Operation::Selection {
            after,
            neighborhood: Some((seed, radius)),
            ..
        } => {
            if SelectionState::capture(state) != *after {
                false
            } else if let Ok(atoms) = crate::selection::expression::evaluate(
                state,
                &format!("byres (all within {radius} of ({seed}))"),
            ) {
                state.ui.selection = Selection::Atoms(atoms);
                *after = SelectionState::capture(state);
                redraw.set_all();
                true
            } else {
                false
            }
        }
        _ => false,
    };
    if valid {
        *state.volatile.history.undo.last_mut().unwrap() = operation;
        state.volatile.history.redo.clear();
        state.volatile.history.revision = state.volatile.history.revision.wrapping_add(1);
    } else {
        state.volatile.history.message =
            "The last operation's target changed; it cannot be adjusted.".into();
    }
}

pub(crate) fn show(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    ui.collapsing("Edit / history", |ui| {
        let idle = !state.ui.left_click_down
            && !mol_manip::has_preview(state)
            && !state.volatile.md_local.running
            && !state.mol_editor.md.running;
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    idle && !state.volatile.history.undo.is_empty(),
                    egui::Button::new("Undo"),
                )
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                undo_redo(state, scene, false, redraw, updates);
            }
            if ui
                .add_enabled(
                    idle && !state.volatile.history.redo.is_empty(),
                    egui::Button::new("Redo"),
                )
                .on_hover_text("Ctrl+Shift+Z / Ctrl+Y")
                .clicked()
            {
                undo_redo(state, scene, true, redraw, updates);
            }
        });
        ui.checkbox(&mut state.volatile.mol_manip.snap, "Snap transforms (Ctrl)");
        ui.small("Shift: fine · X/Y/Z: axis · Shift+axis: plane");
        ui.small("Enter: accept · Esc: cancel");
        if !state.volatile.history.message.is_empty() {
            ui.label(&state.volatile.history.message);
        }
        let Some(last) = state.volatile.history.undo.last() else {
            return;
        };
        ui.label(format!("Adjust last: {}", last.label()));
        let (moving, mut transform, mut radius) = match last {
            Operation::Transform { before, values, .. } => (
                matches!(before.mode, ManipMode::Move(_)),
                Some(values.clone()),
                None,
            ),
            Operation::Selection {
                neighborhood: Some((_, radius)),
                ..
            } => (false, None, Some(*radius)),
            _ => (false, None, None),
        };
        let mut changed = false;
        ui.add_enabled_ui(idle, |ui| {
            if let Some(values) = &mut transform {
                if moving {
                    for (label, value) in [
                        ("X Å", &mut values.translation.x),
                        ("Y Å", &mut values.translation.y),
                        ("Z Å", &mut values.translation.z),
                    ] {
                        ui.horizontal(|ui| {
                            ui.label(label);
                            changed |= ui
                                .add(DragValue::new(value).speed(0.05).range(-1.0e9..=1.0e9))
                                .changed();
                        });
                    }
                } else {
                    let mut degrees = values.angle.to_degrees();
                    ui.horizontal(|ui| {
                        ui.label("Angle °");
                        changed |= ui
                            .add(
                                DragValue::new(&mut degrees)
                                    .speed(0.5)
                                    .range(-1.0e9..=1.0e9),
                            )
                            .changed();
                    });
                    values.angle = degrees.to_radians();
                }
            } else if let Some(radius) = &mut radius {
                ui.horizontal(|ui| {
                    ui.label("Radius Å");
                    changed |= ui
                        .add(DragValue::new(radius).speed(0.1).range(0.1..=1000.))
                        .changed();
                });
            }
        });
        if changed {
            let mut last = state.volatile.history.undo.last().cloned().unwrap();
            match &mut last {
                Operation::Transform { values, .. } => *values = transform.unwrap(),
                Operation::Selection {
                    neighborhood: Some((_, value)),
                    ..
                } => *value = radius.unwrap(),
                _ => (),
            }
            adjust(state, scene, last, redraw, updates);
        }
        ui.collapsing("History", |ui| {
            egui::ScrollArea::vertical()
                .max_height(160.)
                .id_salt("edit_history_entries")
                .show(ui, |ui| {
                    for operation in state.volatile.history.undo.iter().rev() {
                        ui.label(operation.label());
                    }
                });
        });
    });
}
