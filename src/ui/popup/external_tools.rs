//! The third-party tools status panel.
//!
//! Molchanica drives a handful of optional external tools, and every one of them can be absent,
//! present-but-broken, or working. Before this panel each of those states surfaced only at the
//! moment a user tried to use the feature, as a one-line error in a different part of the UI — so
//! "why is the ORCA button doing nothing", "why did structure prediction fail", and "did the
//! installer actually work" were four separate support conversations with no single place to look.
//!
//! This is that place. It probes every entry in [`crate::external_tools`] concurrently on a worker
//! thread and reports, per tool, whether it runs, where it was found, what version answered, and —
//! when it is missing — an in-app installer for the tools Molchanica can manage itself.

use std::{collections::HashMap, sync::mpsc, thread};

use egui::{Color32, RichText, ScrollArea, Ui};

use crate::{
    external_tools::{self, CheckResult, Tool, ToolStatus},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, popup::close_btn},
};

type DiskUsage = Result<Option<u64>, String>;
type ProbeResult = (Vec<ToolStatus>, HashMap<Tool, DiskUsage>);

/// Panel state: the last probe's results, and the receiver for one in flight.
#[derive(Default)]
pub struct ExternalToolsUi {
    statuses: Vec<ToolStatus>,
    pending: Option<mpsc::Receiver<ProbeResult>>,
    install_pending: Option<(Tool, mpsc::Receiver<Result<(), String>>)>,
    install_results: HashMap<Tool, InstallResult>,
    disk_usage: HashMap<Tool, DiskUsage>,
    uninstall_pending: Option<(Tool, mpsc::Receiver<Result<(), String>>)>,
    uninstall_results: HashMap<Tool, UninstallResult>,
    confirm_uninstall: Option<Tool>,
    /// Set once, so opening the panel runs a probe without the user having to ask.
    probed_once: bool,
    /// Tool whose full detail text is expanded.
    expanded: Option<Tool>,
}

enum InstallResult {
    Success,
    Failed(String),
}

enum UninstallResult {
    Success,
    Failed(String),
}

impl ExternalToolsUi {
    fn is_probing(&self) -> bool {
        self.pending.is_some()
    }

    fn installing(&self) -> Option<Tool> {
        self.install_pending.as_ref().map(|(tool, _)| *tool)
    }

    fn uninstalling(&self) -> Option<Tool> {
        self.uninstall_pending.as_ref().map(|(tool, _)| *tool)
    }

    fn is_busy(&self) -> bool {
        self.is_probing() || self.installing().is_some() || self.uninstalling().is_some()
    }

    /// Start a probe on a worker thread.
    ///
    /// Off the UI thread because a probe launches a subprocess per tool, and the Python-based ones
    /// import Torch before answering — seconds each, on a cold cache. Blocking the UI for that
    /// would be worse than the problem this panel solves.
    fn start_probe(&mut self, context: &egui::Context) {
        if self.is_busy() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.probed_once = true;
        self.confirm_uninstall = None;

        let context = context.clone();
        thread::spawn(move || {
            let statuses = external_tools::check_all();
            let disk_usage = statuses
                .iter()
                .filter(|status| status.result != CheckResult::CantFind)
                .map(|status| {
                    (
                        status.tool,
                        external_tools::managed_disk_usage(status.tool)
                            .map_err(|error| error.to_string()),
                    )
                })
                .collect();
            let _ = tx.send((statuses, disk_usage));
            context.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(receiver) = &self.pending else { return };
        match receiver.try_recv() {
            Ok((statuses, disk_usage)) => {
                self.statuses = statuses;
                self.disk_usage = disk_usage;
                self.pending = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            // The worker died without sending; drop the receiver so the panel can be retried
            // rather than showing "checking" forever.
            Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
        }
    }

    /// Start one installer on a worker thread. Installers can download large model environments,
    /// so they must never block egui's render thread.
    fn start_install(&mut self, tool: Tool, context: &egui::Context) {
        if self.is_busy() || !tool.spec().molchanica_managed {
            return;
        }

        let (tx, rx) = mpsc::channel();
        self.install_pending = Some((tool, rx));
        self.install_results.remove(&tool);
        self.confirm_uninstall = None;
        self.uninstall_results.remove(&tool);

        let context = context.clone();
        thread::spawn(move || {
            let result = external_tools::install(tool).map_err(|error| error.to_string());
            let _ = tx.send(result);
            context.request_repaint();
        });
    }

    /// Poll the active installer, returning true once it finishes so the caller can re-probe all
    /// tools and immediately reflect the new installation state.
    fn poll_install(&mut self) -> bool {
        let Some((tool, receiver)) = &self.install_pending else {
            return false;
        };
        let tool = *tool;
        let result = match receiver.try_recv() {
            Ok(Ok(())) => InstallResult::Success,
            Ok(Err(error)) => InstallResult::Failed(error),
            Err(mpsc::TryRecvError::Empty) => return false,
            Err(mpsc::TryRecvError::Disconnected) => {
                InstallResult::Failed("installer stopped without reporting a result".to_owned())
            }
        };

        self.install_results.insert(tool, result);
        self.install_pending = None;
        true
    }

    fn start_uninstall(&mut self, tool: Tool, context: &egui::Context) {
        let has_managed_files =
            matches!(self.disk_usage.get(&tool), Some(Ok(Some(_))) | Some(Err(_)));
        if self.is_busy() || !tool.spec().molchanica_managed || !has_managed_files {
            return;
        }

        let (tx, rx) = mpsc::channel();
        self.uninstall_pending = Some((tool, rx));
        self.confirm_uninstall = None;
        self.install_results.remove(&tool);
        self.uninstall_results.remove(&tool);

        let context = context.clone();
        thread::spawn(move || {
            let result = external_tools::uninstall(tool).map_err(|error| error.to_string());
            let _ = tx.send(result);
            context.request_repaint();
        });
    }

    fn poll_uninstall(&mut self) -> bool {
        let Some((tool, receiver)) = &self.uninstall_pending else {
            return false;
        };
        let tool = *tool;
        let result = match receiver.try_recv() {
            Ok(Ok(())) => UninstallResult::Success,
            Ok(Err(error)) => UninstallResult::Failed(error),
            Err(mpsc::TryRecvError::Empty) => return false,
            Err(mpsc::TryRecvError::Disconnected) => {
                UninstallResult::Failed("uninstaller stopped without reporting a result".to_owned())
            }
        };

        self.uninstall_results.insert(tool, result);
        self.uninstall_pending = None;
        true
    }

    /// How many tools are ready, for a one-line summary elsewhere in the UI.
    pub fn ready_count(&self) -> usize {
        self.statuses
            .iter()
            .filter(|status| status.result == CheckResult::Pass)
            .count()
    }
}

fn result_color(result: CheckResult) -> Color32 {
    match result {
        CheckResult::Pass => Color32::LIGHT_GREEN,
        // Installed but not working is the state worth reading about, so it is the loud one.
        CheckResult::Error => Color32::ORANGE,
        CheckResult::CantFind => COLOR_INACTIVE,
    }
}

fn format_disk_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1_024.0 && unit < UNITS.len() - 1 {
        value /= 1_024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

pub fn external_tools_window(state: &mut crate::state::State, ui: &mut Ui) {
    let context = ui.ctx().clone();
    let tools = &mut state.ui.external_tools;
    tools.poll();
    let install_finished = tools.poll_install();
    let uninstall_finished = tools.poll_uninstall();
    if install_finished || uninstall_finished {
        tools.start_probe(&context);
    }

    if !tools.probed_once {
        tools.start_probe(&context);
    }

    ui.horizontal(|ui| {
        ui.heading("Third-party tools");
        ui.add_space(ROW_SPACING);
        if tools.is_probing() {
            ui.label(RichText::new("Checking…").color(COLOR_ACTION));
        } else if ui
            .add_enabled(
                !tools.is_busy(),
                egui::Button::new(RichText::new("Re-check").color(COLOR_HIGHLIGHT)),
            )
            .on_hover_text(
                "Run every tool's version probe again. Do this after installing something, or \
                 after changing an override environment variable.",
            )
            .clicked()
        {
            tools.start_probe(&context);
        }
    });

    ui.label(
        "Molchanica runs without any of these. Each one unlocks a feature; nothing here is \
         required to open, view, edit, or simulate molecules.",
    );
    ui.add_space(ROW_SPACING);

    if tools.statuses.is_empty() && tools.is_probing() {
        ui.spinner();
        ui.add_space(ROW_SPACING);
        close_btn(ui, &mut state.ui.popup.external_tools);
        return;
    }

    ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            let statuses = tools.statuses.clone();
            for status in &statuses {
                let spec = status.tool.spec();

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("[{}]", status.result.label()))
                            .color(result_color(status.result))
                            .monospace(),
                    );
                    ui.label(RichText::new(spec.name).strong());
                    if ui
                        .small_button("?")
                        .on_hover_text(format!(
                            "{}\n\nLicence: {}\n{}",
                            spec.summary, spec.license, spec.url
                        ))
                        .clicked()
                    {
                        tools.expanded = if tools.expanded == Some(status.tool) {
                            None
                        } else {
                            Some(status.tool)
                        };
                    }

                    if status.result != CheckResult::CantFind {
                        let uninstalling = tools.uninstalling();
                        let confirming = tools.confirm_uninstall == Some(status.tool);
                        let (has_managed_files, disk_text, disk_hover) =
                            if !spec.molchanica_managed {
                                (
                                    false,
                                    "Disk: managed externally".to_owned(),
                                    Some(
                                        "Molchanica cannot estimate a system- or vendor-managed \
                                         installation."
                                            .to_owned(),
                                    ),
                                )
                            } else {
                                match tools.disk_usage.get(&status.tool) {
                                    Some(Ok(Some(bytes))) => (
                                        true,
                                        format!("Disk: ~{}", format_disk_size(*bytes)),
                                        Some(
                                            "Molchanica-managed files only; upstream model caches \
                                             outside Molchanica's data directory are excluded."
                                                .to_owned(),
                                        ),
                                    ),
                                    Some(Ok(None)) => (
                                        false,
                                        "Disk: no managed files".to_owned(),
                                        Some(
                                            "The tool was found, but it is not installed in \
                                             Molchanica's managed data directory."
                                                .to_owned(),
                                        ),
                                    ),
                                    Some(Err(error)) => (
                                        true,
                                        "Disk: unavailable".to_owned(),
                                        Some(error.clone()),
                                    ),
                                    None => (false, "Disk: calculating…".to_owned(), None),
                                }
                            };

                        let button_text = if uninstalling == Some(status.tool) {
                            "Uninstalling…"
                        } else if confirming {
                            "Confirm uninstall"
                        } else {
                            "Uninstall"
                        };
                        let uninstall_hint = if !spec.molchanica_managed {
                            "This installation is managed outside Molchanica.".to_owned()
                        } else if !has_managed_files {
                            "No Molchanica-managed files were found for this tool.".to_owned()
                        } else if confirming {
                            "Click again to remove this tool's Molchanica-managed files.".to_owned()
                        } else {
                            "Remove this tool's Molchanica-managed files.".to_owned()
                        };
                        let response = ui
                            .add_enabled(
                                spec.molchanica_managed && has_managed_files && !tools.is_busy(),
                                egui::Button::new(button_text),
                            )
                            .on_hover_text(uninstall_hint);
                        if response.clicked() {
                            if confirming {
                                tools.start_uninstall(status.tool, &context);
                            } else {
                                tools.confirm_uninstall = Some(status.tool);
                            }
                        }
                        if uninstalling == Some(status.tool) {
                            ui.spinner();
                        }
                        let size_response =
                            ui.label(RichText::new(disk_text).color(COLOR_INACTIVE).small());
                        if let Some(hover) = disk_hover {
                            size_response.on_hover_text(hover);
                        }
                    }
                });

                ui.indent(spec.slug, |ui| {
                    ui.label(RichText::new(spec.summary).color(COLOR_INACTIVE));

                    match status.result {
                        CheckResult::Pass => {
                            ui.label(RichText::new(&status.detail).small());
                        }
                        CheckResult::Error => {
                            ui.label(
                                RichText::new(&status.detail)
                                    .color(result_color(status.result))
                                    .small(),
                            );
                        }
                        CheckResult::CantFind => {
                            let installing = tools.installing();
                            ui.horizontal(|ui| {
                                let response = ui
                                    .add_enabled(
                                        spec.molchanica_managed && !tools.is_busy(),
                                        egui::Button::new("Install"),
                                    )
                                    .on_hover_text(if spec.molchanica_managed {
                                        spec.install_command()
                                    } else {
                                        spec.install_hint.to_owned()
                                    });
                                if response.clicked() {
                                    tools.start_install(status.tool, &context);
                                }
                                if installing == Some(status.tool) {
                                    ui.spinner();
                                    ui.label(RichText::new("Installing…").color(COLOR_ACTION));
                                } else if !spec.molchanica_managed {
                                    ui.label(
                                        RichText::new(spec.install_hint)
                                            .color(COLOR_INACTIVE)
                                            .small(),
                                    );
                                }
                            });
                        }
                    }

                    if let Some(result) = tools.install_results.get(&status.tool) {
                        match result {
                            InstallResult::Success => {
                                ui.label(
                                    RichText::new("Successfully installed")
                                        .color(Color32::LIGHT_GREEN),
                                );
                            }
                            InstallResult::Failed(error) => {
                                ui.label(
                                    RichText::new(format!("Failed to install. Error: {error}"))
                                        .color(Color32::ORANGE),
                                );
                            }
                        }
                    }

                    if let Some(result) = tools.uninstall_results.get(&status.tool) {
                        match result {
                            UninstallResult::Success => {
                                ui.label(
                                    RichText::new("Successfully uninstalled")
                                        .color(Color32::LIGHT_GREEN),
                                );
                            }
                            UninstallResult::Failed(error) => {
                                ui.label(
                                    RichText::new(format!("Failed to uninstall. Error: {error}"))
                                        .color(Color32::ORANGE),
                                );
                            }
                        }
                    }

                    if tools.expanded == Some(status.tool) {
                        if let Some(path) = &status.path {
                            ui.label(
                                RichText::new(format!("Found at {}", path.display()))
                                    .color(COLOR_INACTIVE)
                                    .small(),
                            );
                        }
                        ui.label(
                            RichText::new(format!(
                                "Override with {}",
                                spec.exe_override_env
                            ))
                            .color(COLOR_INACTIVE)
                            .small(),
                        );
                        ui.label(
                            RichText::new(format!("Licence: {}", spec.license))
                                .color(COLOR_INACTIVE)
                                .small(),
                        );
                        ui.hyperlink(spec.url);
                    }
                });
                ui.add_space(6.0);
            }
        });

    ui.add_space(ROW_SPACING);
    ui.label(
        RichText::new(
            "Installers live in install_scripts/. Run install_tool with a tool's name, or `all`. \
             The Install buttons above run the same script for you.",
        )
        .color(COLOR_INACTIVE),
    );
    ui.add_space(ROW_SPACING);
    close_btn(ui, &mut state.ui.popup.external_tools);
}
