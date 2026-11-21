//! The quantum chemistry window. Psi4 runs through the shared tool runner; ORCA through its own
//! form ([`orca`]) and the same runner's progress, history, and results browser.

pub(super) mod orca;

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use egui::{
    CollapsingHeader, Color32, ComboBox, Grid, RichText, ScrollArea, TextBuffer, TextEdit,
    TextFormat, TextStyle, Ui, text::LayoutJob,
};
use egui_file_dialog::{FileDialogConfig, Filter};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;
use mol_defs::molecules::common::MoleculeCommon;
use na_seq::Element;
use serde_json::Value;

use super::{
    compute_selector,
    external_tools::{install_btn, tool_links},
    flex_lists::PageLists,
    remote_token_ui,
    tool_runner::{ToolWindowKind, tool_window},
};
use crate::{
    external_tools::{self, Tool, remote::ComputeTarget, tool_form::FormField},
    orca::{Choice, MolTarget, OrcaJob, overridden, read_xyz_positions, xyz_rows},
    state::State,
    ui::{COL_SPACING, COLOR_AMBER},
    util::{RedrawFlags, handle_err, handle_success},
};

/// The backends this window offers.
const BACKENDS: [Tool; 2] = [Tool::Psi4, Tool::Orca];

/// Work the tool window asks of this window, which needs the rest of the program's state.
pub(super) enum QcRequest {
    /// Put the active molecule's atoms in the Psi4 geometry.
    Psi4Geometry,
    /// Put the active molecule's atoms in the ORCA geometry.
    OrcaGeometry,
    /// Write the ORCA input text from its parameters.
    OrcaText,
    OrcaRun,
    Apply(QcApply),
}

/// A result to apply to the molecule a run was made from.
pub(super) enum QcApply {
    Geometry {
        target: MolTarget,
        path: PathBuf,
    },
    Charges {
        target: MolTarget,
        scheme: String,
        values: Vec<f64>,
    },
}

pub(super) fn window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    ui: &mut Ui,
) {
    let qc = state.ui.tool_windows.quantum_chem();
    let busy = qc.busy();
    let mut tool = qc.tool();

    ui.horizontal(|ui| {
        ui.label("Backend");
        ui.add_enabled_ui(!busy, |ui| {
            ComboBox::from_id_salt("qc_backend")
                .selected_text(tool.spec().name())
                .show_ui(ui, |ui| {
                    for backend in BACKENDS {
                        ui.selectable_value(&mut tool, backend, backend.spec().name())
                            .on_hover_text(backend.spec().summary());
                    }
                });
        });

        ui.add_space(COL_SPACING);
        compute_selector(
            ui,
            &mut state.to_save.quantum_compute_target,
            "Run on this computer's CPU. Install the selected backend first.",
        );

        ui.add_space(COL_SPACING);
        tool_links(tool, ui);
    });
    state.ui.tool_windows.quantum_chem().set_tool(tool);

    if state.to_save.quantum_compute_target == ComputeTarget::Remote {
        if tool == Tool::Orca {
            ui.label(
                RichText::new("ORCA isn't available remotely. Set Compute to Local to run it.")
                    .color(COLOR_AMBER),
            );
        } else {
            remote_token_ui(ui, &mut state.to_save.athanor_api_token);
        }
    } else if state.ui.external_tools.has_pending_job(tool) || !external_tools::is_installed(tool) {
        install_btn(state, tool, ui);
    }

    ui.separator();

    let active = active_mol(state);
    state.ui.tool_windows.quantum_chem().active_mol = active;
    tool_window(state, ToolWindowKind::QuantumChemistry, scene, updates, ui);

    if let Some(request) = state.ui.tool_windows.quantum_chem().take_request() {
        handle_request(state, request, redraw, ui);
    }
}

/// Each atom's element and current position.
fn atom_positions(common: &MoleculeCommon) -> impl Iterator<Item = (Element, Vec3)> + '_ {
    let absolute = common.atom_posits.len() == common.atoms.len();

    common.atoms.iter().enumerate().map(move |(i, atom)| {
        let posit = if absolute {
            common.atom_posits[i]
        } else {
            atom.posit
        };

        (atom.element, posit)
    })
}

/// The active molecule's atoms, and how to find it again.
fn active_geometry(state: &State) -> Option<(MolTarget, Vec<(Element, Vec3)>)> {
    let (mol_type, index) = state.volatile.active_mol?;
    let mol = state.get_mol(mol_type, index)?;
    let common = mol.common();
    let geometry = atom_positions(common).collect();

    let target = MolTarget {
        mol_type,
        index,
        ident: common.ident.clone(),
        atom_count: common.atoms.len(),
    };

    Some((target, geometry))
}

fn active_mol(state: &State) -> Option<orca::ActiveMol> {
    let mol = state.active_mol()?;
    let common = mol.common();
    let atoms = &common.atoms;

    let charge_sum = atoms
        .iter()
        .map(|atom| atom.partial_charge)
        .sum::<Option<f32>>()
        .filter(|_| !atoms.is_empty());

    Some(orca::ActiveMol {
        name: common.name.clone().unwrap_or_else(|| common.ident.clone()),
        atom_count: atoms.len(),
        charge_sum,
    })
}

fn handle_request(state: &mut State, request: QcRequest, redraw: &mut RedrawFlags, ui: &Ui) {
    let geometry = match request {
        QcRequest::Psi4Geometry | QcRequest::OrcaGeometry => active_geometry(state),
        _ => None,
    };
    let computations = &state.volatile.ongoing_computations;
    let window = state.ui.tool_windows.quantum_chem();

    match request {
        QcRequest::Psi4Geometry | QcRequest::OrcaGeometry => {
            let Some((target, atoms)) = geometry else {
                window.set_error("No active molecule. Select one first.".to_owned());
                return;
            };

            let rows = xyz_rows(&atoms);
            if matches!(request, QcRequest::Psi4Geometry) {
                window.set_psi4_geometry(rows, &target.ident);
            } else {
                let charge_sum = window.active_mol.as_ref().and_then(|mol| mol.charge_sum);
                window.orca.set_geometry(rows, target, charge_sum);
            }
        }
        QcRequest::OrcaText => {
            let text = window.orca.params.make_input(window.orca.title());
            window.orca.set_text_from_params(text);
        }
        QcRequest::OrcaRun => {
            let form = &mut window.orca;
            form.refresh();

            let job = match form.mode {
                orca::InputMode::Parameters => Ok(OrcaJob::from_params(
                    &form.params,
                    form.title(),
                    form.target().cloned(),
                )),
                orca::InputMode::Text => Ok(OrcaJob::from_text(form.text.clone(), "text")),
                orca::InputMode::Upload => fs::read_to_string(form.input_file.trim())
                    .map(|text| OrcaJob::from_text(text, "upload"))
                    .map_err(|error| format!("Couldn't read the input file: {error}")),
            };

            match job {
                Ok(job) => window.start_orca(ui, computations, job),
                Err(error) => window.set_error(error),
            }
        }
        QcRequest::Apply(apply) => apply_result(state, apply, redraw),
    }
}

/// Apply a run's geometry or charges to the molecule it was made from.
fn apply_result(state: &mut State, apply: QcApply, redraw: &mut RedrawFlags) {
    let target = match &apply {
        QcApply::Geometry { target, .. } | QcApply::Charges { target, .. } => target.clone(),
    };

    let posits = match &apply {
        QcApply::Geometry { path, .. } => match read_xyz_positions(path) {
            Ok(posits) => Some(posits),
            Err(error) => {
                handle_err(
                    &mut state.ui,
                    format!("Couldn't read the final geometry: {error}"),
                );
                return;
            }
        },
        QcApply::Charges { .. } => None,
    };

    let Some(mut mol) = state
        .get_mol_mut(target.mol_type, target.index)
        .filter(|mol| mol.common().ident == target.ident)
    else {
        handle_err(
            &mut state.ui,
            format!(
                "{} is no longer open; the result wasn't applied.",
                target.ident
            ),
        );
        return;
    };

    let count = mol.common().atoms.len();
    let msg = match apply {
        QcApply::Geometry { .. } => {
            let posits = posits.unwrap_or_default();
            if posits.len() != count {
                let msg = format!(
                    "The geometry has {} atoms, but {} has {count}; it wasn't applied.",
                    posits.len(),
                    target.ident
                );
                handle_err(&mut state.ui, msg);
                return;
            }

            mol.common_mut().atom_posits = posits;
            format!("Applied the ORCA geometry to {}", target.ident)
        }
        QcApply::Charges { scheme, values, .. } => {
            if values.len() != count {
                let msg = format!(
                    "{} charges for {} atoms in {}; they weren't assigned.",
                    values.len(),
                    count,
                    target.ident
                );
                handle_err(&mut state.ui, msg);
                return;
            }

            for (atom, q) in mol.common_mut().atoms.iter_mut().zip(values) {
                atom.partial_charge = Some(q as f32);
            }
            format!("Assigned {scheme} charges to {}", target.ident)
        }
    };

    redraw.set(target.mol_type);
    handle_success(&mut state.ui, msg);
}

// ---------------------------------------------------------------------------------------------
// Input preview
// ---------------------------------------------------------------------------------------------

/// The input a run would use, shown above "Set parameters here" and kept current as they change.
#[derive(Default)]
pub(super) struct InputPreview {
    text: String,
    /// Per line of `text`: whether the last update changed it.
    changed: Vec<bool>,
    error: Option<String>,
    /// A newer version is being made.
    pub pending: bool,
}

impl InputPreview {
    pub(super) fn set(&mut self, text: String) {
        self.error = None;
        if text == self.text {
            return;
        }

        let mut old: HashMap<&str, usize> = HashMap::new();
        for line in self.text.lines() {
            *old.entry(line).or_default() += 1;
        }
        let first = self.text.is_empty();

        self.changed = text
            .lines()
            .map(|line| match old.get_mut(line) {
                Some(count) if *count > 0 => {
                    *count -= 1;
                    false
                }
                _ => !first,
            })
            .collect();
        self.text = text;
    }

    pub(super) fn set_error(&mut self, error: String) {
        self.error = Some(error);
    }
}

/// How long Psi4's parameters must be unchanged before its preview is remade. Each conversion
/// starts Python.
const PSI4_PREVIEW_DELAY: Duration = Duration::from_millis(400);

/// Keeps the Psi4 preview current: converts the parameters in the background, through the writer
/// Run uses, once they stop changing.
#[derive(Default)]
pub(super) struct Psi4Preview {
    pub preview: InputPreview,
    /// The values last converted, or being converted.
    requested: Option<HashMap<String, String>>,
    /// Values changed since, and when.
    waiting: Option<(HashMap<String, String>, Instant)>,
    receiver: Option<mpsc::Receiver<io::Result<HashMap<String, String>>>>,
}

impl Psi4Preview {
    /// Call each frame "Set parameters here" is shown, with its current values.
    pub(super) fn update(&mut self, values: HashMap<String, String>, ctx: &egui::Context) {
        if let Some(receiver) = &self.receiver {
            match receiver.try_recv() {
                Ok(Ok(mut projected)) => {
                    self.receiver = None;
                    self.preview
                        .set(projected.remove("psi4_input").unwrap_or_default());
                }
                Ok(Err(error)) => {
                    self.receiver = None;
                    self.preview.set_error(error.to_string());
                }
                Err(mpsc::TryRecvError::Disconnected) => self.receiver = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        if self.requested.as_ref() == Some(&values) {
            self.waiting = None;
        } else if self
            .waiting
            .as_ref()
            .is_none_or(|(waiting, _)| *waiting != values)
        {
            self.waiting = Some((values, Instant::now()));
        }

        if self.receiver.is_none()
            && let Some((_, since)) = &self.waiting
        {
            // The first preview needn't wait.
            let delay = if self.requested.is_some() {
                PSI4_PREVIEW_DELAY
            } else {
                Duration::ZERO
            };
            let elapsed = since.elapsed();

            if elapsed < delay {
                ctx.request_repaint_after(delay - elapsed);
            } else if let Some((values, _)) = self.waiting.take() {
                let (sender, receiver) = mpsc::channel();
                let ctx = ctx.clone();

                self.requested = Some(values.clone());
                self.receiver = Some(receiver);

                thread::spawn(move || {
                    let _ = sender.send(external_tools::psi4::project(&values, "text"));
                    ctx.request_repaint();
                });
            }
        }

        self.preview.pending = self.waiting.is_some() || self.receiver.is_some();
    }
}

/// The preview, read-only, with lines the last change made highlighted.
pub(super) fn input_preview_ui(
    preview: &InputPreview,
    id: &'static str,
    max_height: f32,
    ui: &mut Ui,
) {
    CollapsingHeader::new("Input")
        .id_salt(id)
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("What Run uses. Highlights show your last change.").small());
                if preview.pending {
                    ui.spinner();
                }
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(preview.text.clone());
                }
            });
            if let Some(error) = &preview.error {
                ui.colored_label(COLOR_AMBER, error);
            }

            let mut layouter = |ui: &Ui, text: &dyn TextBuffer, wrap_width: f32| {
                let font_id = TextStyle::Monospace.resolve(ui.style());
                let color = ui.visuals().text_color();
                let highlight = ui.visuals().selection.bg_fill.gamma_multiply(0.6);

                let mut job = LayoutJob::default();
                job.wrap.max_width = wrap_width;
                for (i, line) in text.as_str().split_inclusive('\n').enumerate() {
                    let mut format = TextFormat::simple(font_id.clone(), color);
                    if preview.changed.get(i) == Some(&true) {
                        format.background = highlight;
                    }
                    job.append(line, 0., format);
                }

                ui.fonts_mut(|fonts| fonts.layout_job(job))
            };

            ScrollArea::vertical()
                .id_salt(id)
                .max_height(max_height)
                .show(ui, |ui| {
                    ui.add(
                        TextEdit::multiline(&mut preview.text.as_str())
                            .code_editor()
                            .desired_rows(4)
                            .desired_width(f32::INFINITY)
                            .layouter(&mut layouter),
                    );
                });
        });
}

// ---------------------------------------------------------------------------------------------
// Dropdowns with text overrides
// ---------------------------------------------------------------------------------------------

/// A text box whose contents, when not blank, replace the dropdown to its left.
pub(super) fn override_box(ui: &mut Ui, custom: &mut String) {
    ui.add(
        TextEdit::singleline(custom)
            .hint_text("Other…")
            .desired_width(150.),
    )
    .on_hover_text("Anything typed here is used instead of the dropdown's choice.");
}

/// A dropdown of `choices`, grouped, with an [`override_box`] to its right.
pub(super) fn choice_override(
    ui: &mut Ui,
    id_salt: &str,
    choices: &[Choice],
    chosen: &mut String,
    custom: &mut String,
) {
    let is_overridden = !custom.trim().is_empty();
    let current = choices
        .iter()
        .find(|choice| choice.value.eq_ignore_ascii_case(chosen));
    let label = match current {
        Some(choice) => choice.label,
        None if chosen.is_empty() => "—",
        None => chosen.as_str(),
    };

    let mut text = RichText::new(label);
    if is_overridden {
        text = text.strikethrough().weak();
    }

    let response = ComboBox::from_id_salt(id_salt)
        .height(450.)
        .selected_text(text)
        .show_ui(ui, |ui| {
            let mut group = None;
            for choice in choices {
                if group != Some(choice.group) {
                    if group.is_some() {
                        ui.separator();
                    }
                    if !choice.group.is_empty() {
                        ui.strong(choice.group);
                    }
                    group = Some(choice.group);
                }

                let selected = !is_overridden && choice.value.eq_ignore_ascii_case(chosen);
                let mut response = ui.selectable_label(selected, choice.label);
                if !choice.help.is_empty() {
                    response = response.on_hover_text(choice.help);
                }
                if response.clicked() {
                    *chosen = choice.value.to_owned();
                    custom.clear();
                }
            }
        })
        .response;
    if let Some(choice) = current.filter(|choice| !choice.help.is_empty()) {
        response.on_hover_text(choice.help);
    }

    override_box(ui, custom);
}

/// Bring a dropdown and its override in line with `value`, if it was changed elsewhere, e.g. by an
/// example. A value in the list selects it; any other is put in the override.
fn sync_override(value: &str, choices: &[Choice], chosen: &mut String, custom: &mut String) {
    if overridden(chosen, custom) == value {
        return;
    }

    if choices
        .iter()
        .any(|choice| choice.value.eq_ignore_ascii_case(value))
    {
        *chosen = value.to_owned();
        custom.clear();
    } else {
        *custom = value.to_owned();
    }
}

/// A Psi4 text field drawn as a dropdown with an override: the method, and the basis set. `None`
/// for other fields.
pub(super) fn psi4_choice_field(
    field: &FormField,
    label: &str,
    value: &mut String,
    ui: &mut Ui,
) -> Option<egui::Response> {
    let choices = match field.name.as_str() {
        "method" => PSI4_METHODS,
        "basis" => PSI4_BASES,
        _ => return None,
    };

    let id = ui.id().with("psi4_override");
    let (mut chosen, mut custom) = ui
        .data_mut(|data| data.get_temp::<(String, String)>(id))
        .unwrap_or_default();
    sync_override(value, choices, &mut chosen, &mut custom);

    let response = ui
        .horizontal(|ui| {
            ui.label(label);
            choice_override(ui, &field.name, choices, &mut chosen, &mut custom);
        })
        .response;

    *value = overridden(&chosen, &custom).to_owned();
    ui.data_mut(|data| data.insert_temp(id, (chosen, custom)));

    Some(response)
}

/// Psi4's properties field: the comma-separated names, and a dropdown to add or remove them.
pub(super) fn psi4_properties_field(
    label: &str,
    value: &mut String,
    ui: &mut Ui,
) -> egui::Response {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(TextEdit::singleline(value).desired_width(300.));

        let mut names: Vec<String> = value
            .split(',')
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty())
            .collect();
        let mut changed = false;

        ComboBox::from_id_salt("psi4_properties")
            .selected_text("Add / remove")
            .height(400.)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show_ui(ui, |ui| {
                for property in PSI4_PROPERTIES {
                    let present = names
                        .iter()
                        .position(|name| name.eq_ignore_ascii_case(property.value));
                    let mut checked = present.is_some();
                    if ui
                        .checkbox(&mut checked, property.label)
                        .on_hover_text(property.help)
                        .changed()
                    {
                        match present {
                            Some(i) => {
                                names.remove(i);
                            }
                            None => names.push(property.value.to_owned()),
                        }
                        changed = true;
                    }
                }
            });

        if changed {
            *value = names.join(", ");
        }
    })
    .response
}

const fn c(
    value: &'static str,
    label: &'static str,
    group: &'static str,
    help: &'static str,
) -> Choice {
    Choice {
        value,
        label,
        group,
        help,
    }
}

#[rustfmt::skip]
const PSI4_METHODS: &[Choice] = &[
    c("scf", "SCF", "Hartree–Fock", "HF, or DFT if dft_functional is set."),
    c("hf", "HF", "Hartree–Fock", "No electron correlation."),
    c("hf3c", "HF-3c", "Composite (built-in basis)", "Set the basis to none."),
    c("pbeh3c", "PBEh-3c", "Composite (built-in basis)", "Set the basis to none."),
    c("r2scan3c", "r2SCAN-3c", "Composite (built-in basis)", "Set the basis to none."),
    c("b973c", "B97-3c", "Composite (built-in basis)", "Set the basis to none."),
    c("blyp", "BLYP", "GGA", ""),
    c("bp86", "BP86", "GGA", ""),
    c("pbe", "PBE", "GGA", ""),
    c("revpbe", "revPBE", "GGA", ""),
    c("b97-d3bj", "B97-D3(BJ)", "GGA", "Includes D3 dispersion."),
    c("tpss", "TPSS", "Meta-GGA", ""),
    c("scan", "SCAN", "Meta-GGA", ""),
    c("r2scan", "r2SCAN", "Meta-GGA", ""),
    c("m06-l", "M06-L", "Meta-GGA", ""),
    c("b97m-v", "B97M-V", "Meta-GGA", "Includes VV10 dispersion."),
    c("b3lyp", "B3LYP", "Hybrid", ""),
    c("b3lyp-d3bj", "B3LYP-D3(BJ)", "Hybrid", "Includes D3 dispersion."),
    c("pbe0", "PBE0", "Hybrid", ""),
    c("pbe0-d3bj", "PBE0-D3(BJ)", "Hybrid", "Includes D3 dispersion."),
    c("tpssh", "TPSSh", "Hybrid", ""),
    c("m06-2x", "M06-2X", "Hybrid", ""),
    c("pw6b95", "PW6B95", "Hybrid", ""),
    c("x3lyp", "X3LYP", "Hybrid", ""),
    c("wb97x", "ωB97X", "Range-separated hybrid", ""),
    c("wb97x-d", "ωB97X-D", "Range-separated hybrid", "Includes dispersion."),
    c("wb97x-d3bj", "ωB97X-D3(BJ)", "Range-separated hybrid", "Includes D3 dispersion."),
    c("wb97x-v", "ωB97X-V", "Range-separated hybrid", "Includes VV10 dispersion."),
    c("wb97m-v", "ωB97M-V", "Range-separated hybrid", "Includes VV10 dispersion."),
    c("cam-b3lyp", "CAM-B3LYP", "Range-separated hybrid", ""),
    c("b2plyp", "B2PLYP", "Double hybrid", ""),
    c("b2plyp-d3bj", "B2PLYP-D3(BJ)", "Double hybrid", "Includes D3 dispersion."),
    c("dsd-pbep86", "DSD-PBEP86", "Double hybrid", ""),
    c("pwpb95", "PWPB95", "Double hybrid", ""),
    c("mp2", "MP2", "Wavefunction", ""),
    c("scs-mp2", "SCS-MP2", "Wavefunction", ""),
    c("mp2.5", "MP2.5", "Wavefunction", ""),
    c("mp3", "MP3", "Wavefunction", ""),
    c("omp2", "OMP2", "Wavefunction", "Orbital-optimized MP2."),
    c("cisd", "CISD", "Wavefunction", ""),
    c("ccsd", "CCSD", "Wavefunction", "Small molecules only."),
    c("ccsd(t)", "CCSD(T)", "Wavefunction", "Small molecules only."),
    c("fno-ccsd(t)", "FNO-CCSD(T)", "Wavefunction", "CCSD(T) with frozen natural orbitals; cheaper."),
    c("eom-ccsd", "EOM-CCSD", "Wavefunction", "Excited states. Set roots_per_irrep in Additional options."),
    c("sapt0", "SAPT0", "SAPT (two fragments)", "Interaction energy. Separate the fragments with -- in the geometry."),
    c("sapt2+", "SAPT2+", "SAPT (two fragments)", "Interaction energy. Separate the fragments with -- in the geometry."),
];

#[rustfmt::skip]
const PSI4_BASES: &[Choice] = &[
    c("", "None", "", "The method includes its basis, e.g. a 3c method."),
    c("sto-3g", "STO-3G", "Pople", ""),
    c("3-21g", "3-21G", "Pople", ""),
    c("6-31g", "6-31G", "Pople", ""),
    c("6-31g*", "6-31G*", "Pople", ""),
    c("6-31g**", "6-31G**", "Pople", ""),
    c("6-31+g*", "6-31+G*", "Pople", ""),
    c("6-311g**", "6-311G**", "Pople", ""),
    c("6-311+g(2d,p)", "6-311+G(2d,p)", "Pople", ""),
    c("6-311++g(2d,2p)", "6-311++G(2d,2p)", "Pople", ""),
    c("cc-pvdz", "cc-pVDZ", "Dunning", ""),
    c("cc-pvtz", "cc-pVTZ", "Dunning", ""),
    c("cc-pvqz", "cc-pVQZ", "Dunning", ""),
    c("aug-cc-pvdz", "aug-cc-pVDZ", "Dunning", "Diffuse; anions and weak interactions."),
    c("aug-cc-pvtz", "aug-cc-pVTZ", "Dunning", "Diffuse; anions and weak interactions."),
    c("aug-cc-pvqz", "aug-cc-pVQZ", "Dunning", "Diffuse; anions and weak interactions."),
    c("jun-cc-pvdz", "jun-cc-pVDZ", "Dunning", "Partially augmented."),
    c("jun-cc-pvtz", "jun-cc-pVTZ", "Dunning", "Partially augmented."),
    c("cc-pcvdz", "cc-pCVDZ", "Dunning", "Core-valence."),
    c("def2-sv(p)", "def2-SV(P)", "Karlsruhe", ""),
    c("def2-svp", "def2-SVP", "Karlsruhe", ""),
    c("def2-svpd", "def2-SVPD", "Karlsruhe", ""),
    c("def2-tzvp", "def2-TZVP", "Karlsruhe", ""),
    c("def2-tzvpd", "def2-TZVPD", "Karlsruhe", ""),
    c("def2-tzvpp", "def2-TZVPP", "Karlsruhe", ""),
    c("def2-tzvppd", "def2-TZVPPD", "Karlsruhe", ""),
    c("def2-qzvp", "def2-QZVP", "Karlsruhe", ""),
    c("def2-qzvpp", "def2-QZVPP", "Karlsruhe", ""),
    c("pcseg-0", "pcseg-0", "Jensen", ""),
    c("pcseg-1", "pcseg-1", "Jensen", ""),
    c("pcseg-2", "pcseg-2", "Jensen", ""),
    c("aug-pcseg-1", "aug-pcseg-1", "Jensen", ""),
    c("cc-pv[dt]z", "cc-pV[DT]Z", "Extrapolated", "Two-point extrapolation to the basis set limit."),
    c("cc-pv[tq]z", "cc-pV[TQ]Z", "Extrapolated", "Two-point extrapolation to the basis set limit."),
    c("aug-cc-pv[dt]z", "aug-cc-pV[DT]Z", "Extrapolated", "Two-point extrapolation to the basis set limit."),
    c("aug-cc-pv[tq]z", "aug-cc-pV[TQ]Z", "Extrapolated", "Two-point extrapolation to the basis set limit."),
];

#[rustfmt::skip]
const PSI4_PROPERTIES: &[Choice] = &[
    c("dipole", "Dipole", "", ""),
    c("quadrupole", "Quadrupole", "", ""),
    c("mulliken_charges", "Mulliken charges", "", ""),
    c("lowdin_charges", "Löwdin charges", "", ""),
    c("mbis_charges", "MBIS charges", "", "Slow; accurate. Suitable for force fields."),
    c("mbis_volume_ratios", "MBIS volume ratios", "", ""),
    c("wiberg_lowdin_indices", "Wiberg–Löwdin bond indices", "", ""),
    c("mayer_indices", "Mayer bond indices", "", ""),
    c("no_occupations", "Natural orbital occupations", "", ""),
    c("esp_at_nuclei", "ESP at nuclei", "", ""),
    c("mo_extents", "MO extents", "", ""),
    c("grid_esp", "ESP on a grid", "", "Reads grid.dat from the run folder."),
    c("polarizability", "Polarizability", "", "Coupled-cluster and SCF response."),
    c("rotation", "Optical rotation", "", "Coupled-cluster response."),
    c("oscillator_strength", "Oscillator strengths", "", "Coupled-cluster excited states."),
];

pub(super) fn filter_input_files(config: &mut FileDialogConfig, field: &FormField) {
    let extensions = field.accepted_extensions();
    *config = std::mem::take(config).add_file_filter(
        "Psi4 inputs",
        Filter::new(move |path: &Path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    extensions
                        .iter()
                        .any(|ext| ext.eq_ignore_ascii_case(extension))
                })
        }),
    );
    config.default_file_filter = Some("Psi4 inputs".into());
}

pub(super) fn set_file_format(values: &mut HashMap<String, String>, path: &Path) {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return;
    };

    let format = match extension.to_ascii_lowercase().as_str() {
        "py" => "psiapi",
        "dat" | "in" | "inp" | "psi4" => "psithon",
        _ => return,
    };
    values.insert("input_format".into(), format.into());
}

fn output_path(path: &Path) -> Option<(&str, &str)> {
    let parent = path.parent()?.file_name()?.to_str()?;
    let folder = parent
        .split_once('-')
        .filter(|(prefix, _)| prefix.bytes().all(|byte| byte.is_ascii_digit()))
        .map_or(parent, |(_, folder)| folder);

    Some((folder, path.file_name()?.to_str()?))
}

pub(super) fn is_final_geometry(path: &Path) -> bool {
    matches!(
        output_path(path),
        Some(("summary", "geometry.xyz") | (_, "optimized.xyz"))
    )
}

pub(super) fn output_name(path: &Path) -> Option<&'static str> {
    if is_final_geometry(path) {
        return Some("Final geometry (XYZ)");
    }

    match output_path(path)? {
        ("summary", "summary.json") => Some("Calculation summary (JSON)"),
        ("summary", "frequencies.csv") => Some("Vibrational modes (CSV)"),
        ("psi4", "output.dat") | (_, "psi4.out") => Some("Psi4 output"),
        _ => None,
    }
}

pub(super) fn summary_ui(details: &Value, lists: &mut PageLists, ui: &mut Ui) {
    ui.strong("Psi4 calculation");
    ui.horizontal_wrapped(|ui| {
        match details["input_format"].as_str() {
            Some("psithon") => ui.label("Psithon input"),
            Some("psiapi") => ui.label("PsiAPI input"),
            _ => return,
        };
        if let Some(threads) = details["threads"].as_u64() {
            ui.label(format!(
                "{threads} CPU thread{}",
                if threads == 1 { "" } else { "s" }
            ));
        }
        if let Some(memory) = details["memory_mib"].as_f64() {
            ui.label(format!("{memory} MiB"));
        }
    });

    if let Some(energy) = details["energy_hartree"].as_f64() {
        ui.label(format!("Current energy"));
        ui.label(RichText::new(format!("{energy:.12} Eₕ")).color(Color32::WHITE));
    }

    if let Some(modes) = details["vibrational_modes"].as_array() {
        if modes.is_empty() {
            ui.label("Psi4 reported no vibrational modes.");
        } else {
            CollapsingHeader::new("Harmonic vibrational modes")
                .default_open(true)
                .show(ui, |ui| {
                    lists.show(ui, "psi4_modes", 150.0, |ui| {
                        Grid::new("psi4_modes_grid").striped(true).show(ui, |ui| {
                            for heading in [
                                "Mode",
                                "Symmetry",
                                "Frequency (cm⁻¹)",
                                "IR intensity (km/mol)",
                            ] {
                                ui.strong(heading);
                            }
                            ui.end_row();
                            for mode in modes {
                                let frequency = number(&mode["frequency_cm1"], 4);
                                let suffix = if mode["imaginary"].as_bool() == Some(true) {
                                    " (imaginary)"
                                } else {
                                    ""
                                };

                                ui.label(mode["mode"].to_string());
                                ui.label(mode["symmetry"].as_str().unwrap_or("—"));
                                ui.label(format!("{frequency}{suffix}"));
                                ui.label(number(&mode["ir_intensity_km_mol"], 4));
                                ui.end_row();
                            }
                        });
                    });
                });
        }
    }

    for (key, label, format_value) in [
        (
            "variables",
            "Psi4 variables",
            scalar_value as fn(&Value) -> String,
        ),
        ("array_variables", "Psi4 array variables", array_value),
    ] {
        if let Some(variables) = details[key]
            .as_object()
            .filter(|variables| !variables.is_empty())
        {
            CollapsingHeader::new(format!("{label} ({})", variables.len()))
                .id_salt(key)
                .show(ui, |ui| {
                    lists.show(ui, key, 150.0, |ui| {
                        Grid::new(key).striped(true).show(ui, |ui| {
                            for (name, value) in variables {
                                ui.label(name);
                                ui.monospace(
                                    RichText::new(format_value(value)).color(Color32::WHITE),
                                );
                                ui.end_row();
                            }
                        });
                    });
                });
        }
    }

    ui.separator();
}

fn number(value: &Value, precision: usize) -> String {
    value
        .as_f64()
        .map_or_else(|| "—".into(), |value| format!("{value:.precision$}"))
}

fn scalar_value(value: &Value) -> String {
    number(value, 10)
}

fn array_value(value: &Value) -> String {
    fn flatten<'a>(value: &'a Value, values: &mut Vec<&'a Value>) {
        if let Some(array) = value.as_array() {
            for item in array {
                flatten(item, values);
            }
        } else {
            values.push(value);
        }
    }

    let mut values = Vec::new();
    flatten(value, &mut values);
    if values.len() > 12 {
        return format!("{} values in the calculation summary", values.len());
    }

    values
        .into_iter()
        .map(|value| number(value, 8))
        .collect::<Vec<_>>()
        .join(", ")
}
