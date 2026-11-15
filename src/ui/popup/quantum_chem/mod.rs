use std::{collections::HashMap, path::Path};

use egui::{CollapsingHeader, Grid, Ui};
use egui_file_dialog::{FileDialogConfig, Filter};
use graphics::{EngineUpdates, Scene};
use serde_json::Value;

use super::{
    flex_lists::PageLists,
    tool_runner::{ToolWindowKind, tool_window},
};
use crate::{external_tools::tool_form::FormField, state::State};

pub(super) fn window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    tool_window(state, ToolWindowKind::QuantumChemistry, scene, updates, ui);
}

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
        ui.label(format!("Current energy  {energy:.12} Eₕ"));
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
                                ui.monospace(format_value(value));
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
