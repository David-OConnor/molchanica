//! The ORCA backend of the quantum chemistry window: its input form, and its results summary.
//! Inputs, runs, and output parsing are in [`crate::orca`].

use std::path::{Path, PathBuf};

use bio_files::orca::basis_sets::BasisSet;
use egui::{
    CollapsingHeader, Color32, ComboBox, DragValue, Grid, RichText, ScrollArea, TextEdit, Ui,
};
use egui_file_dialog::{FileDialogConfig, Filter};
use serde_json::Value;

use super::{InputPreview, QcApply, choice_override, input_preview_ui, override_box};
use crate::{
    orca::{
        AuxBasis, BASIS_CATEGORIES, ChargeScheme, DftGrid, Dispersion, FreqKind, Geometry, METHODS,
        MdThermostat, MolTarget, OPT_THRESHOLDS, OrcaParams, OrcaTask, Reference, RiApprox,
        SOLVENTS, ScfAid, ScfConv, Solvation,
    },
    ui::{
        COLOR_ACTION, COLOR_AMBER, ROW_SPACING,
        misc::{selector_box, selector_option},
        popup::flex_lists::PageLists,
    },
};

/// Where an ORCA run's input comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::ui::popup) enum InputMode {
    #[default]
    Parameters,
    Text,
    Upload,
}

impl InputMode {
    const ALL: [Self; 3] = [Self::Parameters, Self::Text, Self::Upload];

    fn label(self) -> &'static str {
        match self {
            Self::Parameters => "Set parameters here",
            Self::Text => "Enter input",
            Self::Upload => "Choose input file",
        }
    }
}

/// The active molecule, as the form describes it.
pub(in crate::ui::popup) struct ActiveMol {
    pub name: String,
    pub atom_count: usize,
    /// Of the neutral molecule, where every element is known.
    pub electrons: Option<i64>,
    /// The sum of the atoms' partial charges, where every atom has one.
    pub charge_sum: Option<f32>,
    pub has_hydrogens: bool,
}

/// The ORCA form's state.
#[derive(Default)]
pub(in crate::ui::popup) struct OrcaForm {
    pub mode: InputMode,
    pub params: OrcaParams,
    /// A complete input, for [`InputMode::Text`].
    pub text: String,
    /// For [`InputMode::Upload`].
    pub input_file: String,
    /// The parameters [`Self::text`] was last written from; regenerated on switching to the text
    /// mode if they have changed since.
    text_source: Option<OrcaParams>,
    /// The input the parameters make, shown above them.
    pub preview: InputPreview,
    /// The parameters and geometry fingerprint the preview was made from.
    preview_source: Option<(OrcaParams, Option<u64>)>,
}

impl OrcaForm {
    /// Why Run is unavailable, if it is.
    pub fn not_ready(&self, active: Option<&ActiveMol>) -> Option<&'static str> {
        match self.mode {
            InputMode::Parameters if active.is_none() => {
                Some("Select a molecule to run on; it's the active molecule that ORCA receives.")
            }
            InputMode::Text if self.text.trim().is_empty() => Some("Enter an ORCA input."),
            InputMode::Upload if self.input_file.trim().is_empty() => {
                Some("Choose an ORCA input file.")
            }
            _ => None,
        }
    }

    /// Record that the text was written from the current parameters.
    pub fn set_text_from_params(&mut self, text: String) {
        self.text = text;
        self.text_source = Some(self.params.clone());
    }

    /// Whether the preview is out of date with the parameters, or the active molecule's geometry.
    pub fn preview_stale(&self, fingerprint: Option<u64>) -> bool {
        self.preview_source
            .as_ref()
            .is_none_or(|(params, source)| *params != self.params || *source != fingerprint)
    }

    /// Remake the preview. Very large molecules are cut short, as previewing every atom would be
    /// slow to draw.
    pub fn update_preview(
        &mut self,
        geometry: Option<(&str, &Geometry)>,
        fingerprint: Option<u64>,
    ) {
        let text = match geometry {
            Some((title, atoms)) if atoms.len() > PREVIEW_MAX_ATOMS => {
                let text = self
                    .params
                    .make_input(Some(&atoms[..PREVIEW_MAX_ATOMS]), title);
                let end = text.trim_end().len() - 1;
                let rest = atoms.len() - PREVIEW_MAX_ATOMS;

                format!("{}  # …and {rest} more atoms\n*\n", &text[..end])
            }
            Some((title, atoms)) => self.params.make_input(Some(atoms), title),
            None => self.params.make_input(None, "Molecule"),
        };

        self.preview.set(text);
        self.preview_source = Some((self.params.clone(), fingerprint));
    }
}

/// Atoms beyond this are left out of the preview.
const PREVIEW_MAX_ATOMS: usize = 300;

/// What the form asks of the window, which it can't do itself.
#[derive(Default)]
pub(in crate::ui::popup) struct FormEvent {
    pub pick_file: bool,
    /// Write the input text from the parameters and the active molecule.
    pub make_text: bool,
}

/// The input mode selector, and the selected mode's inputs.
pub(in crate::ui::popup) fn form_ui(
    form: &mut OrcaForm,
    active: Option<&ActiveMol>,
    footer_height: f32,
    ui: &mut Ui,
) -> FormEvent {
    let mut event = FormEvent::default();

    ui.horizontal_wrapped(|ui| {
        ui.label("Input mode");
        selector_box().show(ui, |ui| {
            for mode in InputMode::ALL {
                if selector_option(ui, form.mode == mode, mode.label()).clicked()
                    && form.mode != mode
                {
                    form.mode = mode;
                    let stale = form.text_source.as_ref() != Some(&form.params);
                    if mode == InputMode::Text && (form.text.trim().is_empty() || stale) {
                        event.make_text = true;
                    }
                }
            }
        });
    });
    ui.add_space(ROW_SPACING);

    // Fill the window, less the footer.
    let mut height = ui.available_height() - footer_height - ui.spacing().item_spacing.y;

    match form.mode {
        InputMode::Parameters => {
            let top = ui.min_rect().bottom();
            input_preview_ui(
                &form.preview,
                "orca_input_preview",
                (height * 0.35).clamp(60., 280.),
                ui,
            );
            height -= ui.min_rect().bottom() - top;

            ScrollArea::vertical()
                .id_salt("orca_form_scroll")
                .max_height(height)
                .auto_shrink([true, false])
                .show(ui, |ui| parameters_ui(&mut form.params, active, ui));
        }
        InputMode::Text => {
            ui.horizontal(|ui| {
                ui.label("ORCA input");
                if ui
                    .button("Write from parameters")
                    .on_hover_text(
                        "Replace this text with an input made from the Set parameters here \
                        settings and the active molecule.",
                    )
                    .clicked()
                {
                    event.make_text = true;
                }
                ui.hyperlink_to(
                    "Input reference",
                    "https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/input.html",
                );
            });
            ui.label(
                RichText::new("Runs as written. Edits here don't carry back to the parameters.")
                    .small(),
            );

            ScrollArea::vertical()
                .id_salt("orca_text_scroll")
                .max_height((height - 40.).max(100.))
                .auto_shrink([true, false])
                .show(ui, |ui| {
                    ui.add(
                        TextEdit::multiline(&mut form.text)
                            .code_editor()
                            .desired_rows(20)
                            .desired_width(f32::INFINITY),
                    );
                });
        }
        InputMode::Upload => {
            ui.horizontal(|ui| {
                ui.label("ORCA input file");
                ui.add(TextEdit::singleline(&mut form.input_file).desired_width(300.));
                if ui.button("Choose…").clicked() {
                    event.pick_file = true;
                }
            });
            if let Some(name) = Path::new(form.input_file.trim()).file_name() {
                ui.colored_label(Color32::WHITE, name.to_string_lossy());
            }
            ui.label(
                RichText::new(
                    "Runs in a new folder, as input.inp. Give files the input refers to, e.g. \
                    with xyzfile, as absolute paths.",
                )
                .small(),
            );
        }
    }

    event
}

fn heading(title: &str, docs: &str, ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    CollapsingHeader::new(title)
        .id_salt(("orca", title))
        .default_open(true)
        .show(ui, |ui| {
            if !docs.is_empty() {
                ui.hyperlink_to("Documentation", docs);
            }
            add_contents(ui);
        });
}

/// A dropdown over a keyword enum.
macro_rules! keyword_combo {
    ($ui:expr, $label:expr, $id:expr, $value:expr, $ty:ty, $help:expr) => {{
        $ui.horizontal(|ui| {
            ui.label($label).on_hover_text($help);
            ComboBox::from_id_salt($id)
                .selected_text($value.label())
                .show_ui(ui, |ui| {
                    for option in <$ty>::ALL {
                        let response = ui.selectable_value($value, *option, option.label());
                        if !option.help().is_empty() {
                            response.on_hover_text(option.help());
                        }
                    }
                })
                .response
                .on_hover_text($help);
        });
    }};
}

const DOCS: &str = "https://www.faccts.de/docs/orca/6.1/manual/contents";

fn parameters_ui(params: &mut OrcaParams, active: Option<&ActiveMol>, ui: &mut Ui) {
    heading("Molecule", "", ui, |ui| {
        match active {
            Some(mol) => {
                ui.label(
                    RichText::new(format!("{} · {} atoms", mol.name, mol.atom_count))
                        .color(Color32::WHITE),
                )
                .on_hover_text("ORCA runs on the active molecule's current atom positions.");
                if !mol.has_hydrogens && mol.atom_count > 1 {
                    ui.colored_label(COLOR_AMBER, "No hydrogens. Add them before running.");
                }
                if mol.atom_count > 150 {
                    ui.colored_label(
                        COLOR_AMBER,
                        "Large for QM; consider xTB, a composite method, or a smaller model.",
                    );
                }
            }
            None => {
                ui.colored_label(
                    Color32::LIGHT_RED,
                    "No active molecule. Select one; ORCA runs on it.",
                );
            }
        }

        ui.horizontal(|ui| {
            ui.label("Charge");
            ui.add(DragValue::new(&mut params.charge).range(-20..=20));
            ui.add_space(8.);
            ui.label("Multiplicity")
                .on_hover_text("2S + 1. 1 for closed shells, 2 for one unpaired electron.");
            ui.add(DragValue::new(&mut params.multiplicity).range(1..=11));

            if let Some(sum) = active.and_then(|mol| mol.charge_sum) {
                let rounded = sum.round() as i32;
                if rounded != params.charge
                    && ui
                        .button(format!("Use {rounded:+}"))
                        .on_hover_text(format!(
                            "The partial charges on this molecule sum to {sum:.2}."
                        ))
                        .clicked()
                {
                    params.charge = rounded;
                }
            }
        });
    });

    heading(
        "Calculation",
        &format!("{DOCS}/quickstartguide/recommendations.html"),
        ui,
        |ui| {
            keyword_combo!(
                ui,
                "Task",
                "orca_task",
                &mut params.task,
                OrcaTask,
                params.task.help()
            );
            task_options_ui(params, ui);

            ui.horizontal(|ui| {
                ui.label("Method").on_hover_text(
                    "The Hamiltonian: HF, a DFT functional, a wavefunction method, or a \
                    composite or semi-empirical method that includes its own basis.",
                );
                choice_override(
                    ui,
                    "orca_method",
                    METHODS,
                    &mut params.method,
                    &mut params.method_custom,
                );
            });

            if params.uses_basis() {
                ui.horizontal(|ui| {
                    ui.label("Basis");
                    ComboBox::from_id_salt("orca_basis_cat")
                        .selected_text(params.basis_cat.to_string())
                        .show_ui(ui, |ui| {
                            for cat in BASIS_CATEGORIES {
                                ui.selectable_value(&mut params.basis_cat, cat, cat.to_string());
                            }
                        })
                        .response
                        .on_hover_text("The basis set family, to narrow the list to its right.");

                    let overridden = !params.basis_custom.trim().is_empty();
                    let mut text = RichText::new(params.basis.keyword());
                    if overridden {
                        text = text.strikethrough().weak();
                    }
                    ComboBox::from_id_salt("orca_basis")
                        .height(500.)
                        .selected_text(text)
                        .show_ui(ui, |ui| {
                            for basis in params.basis_cat.get_sets() {
                                if basis == BasisSet::None {
                                    continue;
                                }
                                if ui
                                    .selectable_label(
                                        !overridden && params.basis == basis,
                                        basis.keyword(),
                                    )
                                    .clicked()
                                {
                                    params.basis = basis;
                                    params.basis_custom.clear();
                                }
                            }
                        });
                    override_box(ui, &mut params.basis_custom);
                });

                keyword_combo!(
                    ui,
                    "Dispersion",
                    "orca_disp",
                    &mut params.dispersion,
                    Dispersion,
                    "Dispersion correction. Composite methods, and functionals named with -D \
                    or -V, include one."
                );
            } else {
                ui.label(RichText::new("This method includes its basis set.").small());
            }

            keyword_combo!(
                ui,
                "Reference",
                "orca_ref",
                &mut params.reference,
                Reference,
                "Restricted or unrestricted orbitals."
            );
        },
    );

    heading(
        "Solvation",
        &format!("{DOCS}/essentialelements/solvationmodels.html"),
        ui,
        |ui| {
            keyword_combo!(
                ui,
                "Model",
                "orca_solv",
                &mut params.solvation,
                Solvation,
                "Implicit solvent model."
            );
            if params.solvation != Solvation::None {
                ui.horizontal(|ui| {
                    ui.label("Solvent");
                    choice_override(
                        ui,
                        "orca_solvent",
                        SOLVENTS,
                        &mut params.solvent,
                        &mut params.solvent_custom,
                    );
                });
            }
        },
    );

    heading(
        "SCF and numerics",
        &format!("{DOCS}/essentialelements/scfconvergence.html"),
        ui,
        |ui| {
            keyword_combo!(
                ui,
                "SCF convergence",
                "orca_scfconv",
                &mut params.scf_conv,
                ScfConv,
                "SCF convergence thresholds."
            );
            keyword_combo!(
                ui,
                "Convergence aid",
                "orca_scfaid",
                &mut params.scf_aid,
                ScfAid,
                "Damping for SCFs that oscillate or diverge."
            );
            keyword_combo!(
                ui,
                "Integration grid",
                "orca_grid",
                &mut params.grid,
                DftGrid,
                "DFT integration grid."
            );
            keyword_combo!(
                ui,
                "RI approximation",
                "orca_ri",
                &mut params.ri,
                RiApprox,
                "Resolution of the identity, to speed up the Coulomb and exchange integrals."
            );
            if params.uses_basis() {
                keyword_combo!(
                    ui,
                    "Auxiliary basis",
                    "orca_aux",
                    &mut params.aux_basis,
                    AuxBasis,
                    "Fitting basis for the RI approximation."
                );
            }
        },
    );

    heading("Resources", "", ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("CPU cores").on_hover_text(
                "Parallel runs need OpenMPI. Each core uses its own memory allowance.",
            );
            ui.add(DragValue::new(&mut params.nprocs).range(1..=256));
            ui.add_space(8.);
            ui.label("Memory per core (MB)")
                .on_hover_text("%maxcore. 0 for ORCA's default.");
            ui.add(
                DragValue::new(&mut params.maxcore_mb)
                    .range(0..=1_000_000)
                    .speed(50),
            );
        });
    });

    heading(
        "Additional input",
        &format!("{DOCS}/essentialelements/input.html"),
        ui,
        |ui| {
            ui.horizontal(|ui| {
                ui.label("Keywords").on_hover_text(
                    "Any other simple-input keywords, added to the ! lines. E.g. NBO, or KDIIS.",
                );
                ui.add(
                    TextEdit::singleline(&mut params.extra_keywords)
                        .hint_text("e.g. NBO")
                        .desired_width(300.),
                );
            });
            ui.label("Blocks")
                .on_hover_text("Input blocks, written as-is.");
            ui.add(
                TextEdit::multiline(&mut params.extra_blocks)
                    .code_editor()
                    .hint_text("%scf\n  MaxIter 500\nend")
                    .desired_rows(3)
                    .desired_width(f32::INFINITY),
            );
        },
    );

    for warning in params.warnings(active.and_then(|mol| mol.electrons)) {
        ui.colored_label(COLOR_AMBER, warning);
    }
}

/// Options that only apply to some tasks.
fn task_options_ui(params: &mut OrcaParams, ui: &mut Ui) {
    let optimizes = matches!(
        params.task,
        OrcaTask::Optimize | OrcaTask::OptFreq | OrcaTask::TransitionState
    );
    let freq = matches!(
        params.task,
        OrcaTask::OptFreq | OrcaTask::TransitionState | OrcaTask::Frequencies
    );

    ui.indent("orca_task_options", |ui| {
        if optimizes {
            ui.horizontal(|ui| {
                if params.task != OrcaTask::TransitionState {
                    ui.label("Convergence");
                    ComboBox::from_id_salt("orca_opt_thresh")
                        .selected_text(params.opt_thresh.to_string())
                        .show_ui(ui, |ui| {
                            for thresh in OPT_THRESHOLDS {
                                ui.selectable_value(
                                    &mut params.opt_thresh,
                                    thresh,
                                    thresh.to_string(),
                                );
                            }
                        })
                        .response
                        .on_hover_text("Optimization convergence criteria. Tight for frequencies.");
                    ui.add_space(8.);
                }
                ui.label("Max steps")
                    .on_hover_text("Maximum optimization cycles. 0 for ORCA's default.");
                ui.add(DragValue::new(&mut params.opt_max_iter).range(0..=10_000));
            });
        }

        if freq {
            keyword_combo!(
                ui,
                "Frequencies",
                "orca_freq",
                &mut params.freq,
                FreqKind,
                "How the Hessian is computed."
            );
        }

        match params.task {
            OrcaTask::Charges => {
                keyword_combo!(
                    ui,
                    "Scheme",
                    "orca_charges",
                    &mut params.charge_scheme,
                    ChargeScheme,
                    "Population analysis scheme. Mulliken and Löwdin charges are always \
                    reported too."
                );
            }
            OrcaTask::ExcitedStates => {
                ui.horizontal(|ui| {
                    ui.label("States");
                    ui.add(DragValue::new(&mut params.tddft_roots).range(1..=200));
                    ui.checkbox(&mut params.tddft_triplets, "Triplets");
                });
            }
            OrcaTask::Dynamics => {
                ui.horizontal(|ui| {
                    ui.label("Steps");
                    ui.add(DragValue::new(&mut params.md_steps).range(1..=1_000_000));
                    ui.label("Timestep (fs)");
                    ui.add(
                        DragValue::new(&mut params.md_timestep_fs)
                            .range(0.05..=5.)
                            .speed(0.05),
                    );
                    ui.label("Temperature (K)");
                    ui.add(DragValue::new(&mut params.md_temp_k).range(0.0..=5_000.));
                });
                keyword_combo!(
                    ui,
                    "Thermostat",
                    "orca_thermostat",
                    &mut params.md_thermostat,
                    MdThermostat,
                    "Temperature control."
                );
            }
            _ => {}
        }
    });
}

pub(in crate::ui::popup) fn filter_input_files(config: &mut FileDialogConfig) {
    *config = std::mem::take(config).add_file_filter(
        "ORCA inputs",
        Filter::new(|path: &Path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    ["inp", "txt", "in"]
                        .iter()
                        .any(|ext| ext.eq_ignore_ascii_case(extension))
                })
        }),
    );
    config.default_file_filter = Some("ORCA inputs".into());
}

// ---------------------------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------------------------

/// How an output file is listed.
pub(in crate::ui::popup) fn output_name(path: &Path) -> Option<&'static str> {
    let name = path.file_name()?.to_str()?;

    Some(match name {
        "input.out" => "ORCA output",
        "input.inp" => "ORCA input",
        "input.xyz" => "Final geometry (XYZ)",
        "input_trj.xyz" => "Optimization trajectory (XYZ)",
        "input.globalminimum.xyz" => "Global minimum conformer (XYZ)",
        "input.finalensemble.xyz" => "Conformer ensemble (XYZ)",
        "input.md.xyz" => "MD trajectory (XYZ)",
        "input.hess" => "Hessian",
        "input.engrad" => "Energy and gradient",
        "input.property.txt" => "Properties",
        "input.gbw" => "Wavefunction (GBW)",
        _ => return None,
    })
}

/// The file selected first: the final geometry, else the output.
pub(in crate::ui::popup) fn primary_file(details: &Value, files: &[PathBuf]) -> Option<PathBuf> {
    details["final_geometry"]
        .as_str()
        .map(PathBuf::from)
        .or_else(|| {
            files
                .iter()
                .find(|path| path.file_name().is_some_and(|name| name == "input.out"))
                .cloned()
        })
}

/// The energies, properties, charges, and frequencies ORCA reported, and buttons to apply the
/// geometry or charges to the molecule the run was made from.
pub(in crate::ui::popup) fn summary_ui(
    details: &Value,
    lists: &mut PageLists,
    ui: &mut Ui,
) -> Option<QcApply> {
    let mut apply = None;
    let target = MolTarget::from_json(&details["molecule"]);

    ui.strong("ORCA calculation");
    let summary = &details["summary"];
    ui.horizontal_wrapped(|ui| {
        if let Some(task) = summary["task"].as_str() {
            ui.label(task);
        }
        if let Some(method) = summary["method"].as_str() {
            match summary["basis"].as_str().filter(|basis| !basis.is_empty()) {
                Some(basis) => ui.label(format!("{method}/{basis}")),
                None => ui.label(method),
            };
        }
        if let (Some(charge), Some(mult)) =
            (summary["charge"].as_i64(), summary["multiplicity"].as_u64())
        {
            ui.label(format!("charge {charge}, multiplicity {mult}"));
        }
        if let Some(target) = &target {
            ui.label(format!("on {}", target.ident));
        }
    });

    Grid::new("orca_summary").show(ui, |ui| {
        let mut row = |label: &str, value: String| {
            ui.label(label);
            ui.label(RichText::new(value).color(Color32::WHITE));
            ui.end_row();
        };

        if let Some(energy) = details["energy_hartree"].as_f64() {
            row("Final energy", format!("{energy:.10} Eₕ"));
        }
        if let Some(g) = details["gibbs_hartree"].as_f64() {
            row("Gibbs free energy", format!("{g:.10} Eₕ"));
        }
        if let (Some(homo), Some(lumo)) = (details["homo_ev"].as_f64(), details["lumo_ev"].as_f64())
        {
            row(
                "HOMO / LUMO",
                format!("{homo:.3} / {lumo:.3} eV · gap {:.3} eV", lumo - homo),
            );
        }
        if let Some(dipole) = details["dipole_debye"].as_f64() {
            row("Dipole moment", format!("{dipole:.4} D"));
        }
    });

    if let (Some(path), Some(target)) = (details["final_geometry"].as_str(), &target)
        && ui
            .button(
                RichText::new(format!("Apply geometry to {}", target.ident)).color(COLOR_ACTION),
            )
            .on_hover_text("Move the molecule's atoms to this run's final positions.")
            .clicked()
    {
        apply = Some(QcApply::Geometry {
            target: target.clone(),
            path: PathBuf::from(path),
        });
    }

    if let Some(charges) = details["charges"].as_object() {
        let schemes: Vec<(&String, Vec<f64>)> = charges
            .iter()
            .map(|(scheme, data)| {
                let values = data["values"]
                    .as_array()
                    .map(|values| values.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default();
                (scheme, values)
            })
            .collect();
        let elements: Vec<&str> = charges
            .values()
            .next()
            .and_then(|data| data["elements"].as_array())
            .map(|elements| elements.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        CollapsingHeader::new("Partial charges")
            .default_open(true)
            .show(ui, |ui| {
                if let Some(target) = &target {
                    ui.horizontal_wrapped(|ui| {
                        for (scheme, values) in &schemes {
                            if values.len() != target.atom_count {
                                continue;
                            }
                            if ui
                                .button(
                                    RichText::new(format!("Assign {scheme}")).color(COLOR_ACTION),
                                )
                                .on_hover_text(format!(
                                    "Replace {}'s partial charges with these.",
                                    target.ident
                                ))
                                .clicked()
                            {
                                apply = Some(QcApply::Charges {
                                    target: target.clone(),
                                    scheme: (*scheme).clone(),
                                    values: values.clone(),
                                });
                            }
                        }
                    });
                }

                lists.show(ui, "orca_charges", 150.0, |ui| {
                    Grid::new("orca_charges_grid").striped(true).show(ui, |ui| {
                        ui.strong("Atom");
                        for (scheme, _) in &schemes {
                            ui.strong(*scheme);
                        }
                        ui.end_row();

                        for (i, element) in elements.iter().enumerate() {
                            ui.label(format!("{} {element}", i + 1));
                            for (_, values) in &schemes {
                                ui.label(
                                    values
                                        .get(i)
                                        .map_or_else(|| "—".to_owned(), |q| format!("{q:+.4}")),
                                );
                            }
                            ui.end_row();
                        }
                    });
                });
            });
    }

    if let Some(modes) = details["vibrational_modes"].as_array() {
        let imaginary = modes
            .iter()
            .filter(|mode| mode["imaginary"].as_bool() == Some(true))
            .count();
        let title = match imaginary {
            0 => "Vibrational modes".to_owned(),
            n => format!("Vibrational modes · {n} imaginary"),
        };

        CollapsingHeader::new(title)
            .id_salt("orca_modes")
            .default_open(true)
            .show(ui, |ui| {
                lists.show(ui, "orca_modes", 150.0, |ui| {
                    Grid::new("orca_modes_grid").striped(true).show(ui, |ui| {
                        for heading in ["Mode", "Frequency (cm⁻¹)", "IR intensity (km/mol)"] {
                            ui.strong(heading);
                        }
                        ui.end_row();

                        for mode in modes {
                            let frequency = mode["frequency_cm1"].as_f64().unwrap_or_default();
                            let text = RichText::new(format!("{frequency:.2}"));
                            let text = if mode["imaginary"].as_bool() == Some(true) {
                                text.color(COLOR_AMBER)
                            } else {
                                text
                            };

                            ui.label(mode["mode"].to_string());
                            ui.label(text);
                            ui.label(
                                mode["ir_intensity_km_mol"]
                                    .as_f64()
                                    .map_or_else(|| "—".to_owned(), |v| format!("{v:.2}")),
                            );
                            ui.end_row();
                        }
                    });
                });
            });
    }

    ui.separator();

    apply
}
