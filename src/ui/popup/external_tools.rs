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
//! when it is missing — the exact command that installs it.

use std::{sync::mpsc, thread};

use egui::{Color32, RichText, ScrollArea, Ui};

use crate::{
    external_tools::{self, CheckResult, Tool, ToolStatus},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, popup::close_btn},
};

/// Panel state: the last probe's results, and the receiver for one in flight.
#[derive(Default)]
pub struct ExternalToolsUi {
    statuses: Vec<ToolStatus>,
    pending: Option<mpsc::Receiver<Vec<ToolStatus>>>,
    /// Set once, so opening the panel runs a probe without the user having to ask.
    probed_once: bool,
    /// Tool whose full detail text is expanded.
    expanded: Option<Tool>,
}

impl ExternalToolsUi {
    fn is_probing(&self) -> bool {
        self.pending.is_some()
    }

    /// Start a probe on a worker thread.
    ///
    /// Off the UI thread because a probe launches a subprocess per tool, and the Python-based ones
    /// import Torch before answering — seconds each, on a cold cache. Blocking the UI for that
    /// would be worse than the problem this panel solves.
    fn start_probe(&mut self, context: &egui::Context) {
        if self.is_probing() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.probed_once = true;

        let context = context.clone();
        thread::spawn(move || {
            let _ = tx.send(external_tools::check_all());
            context.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(receiver) = &self.pending else { return };
        match receiver.try_recv() {
            Ok(statuses) => {
                self.statuses = statuses;
                self.pending = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            // The worker died without sending; drop the receiver so the panel can be retried
            // rather than showing "checking" forever.
            Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
        }
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

pub fn external_tools_window(state: &mut crate::state::State, ui: &mut Ui) {
    let context = ui.ctx().clone();
    let tools = &mut state.ui.external_tools;
    tools.poll();

    if !tools.probed_once {
        tools.start_probe(&context);
    }

    ui.horizontal(|ui| {
        ui.heading("Third-party tools");
        ui.add_space(ROW_SPACING);
        if tools.is_probing() {
            ui.label(RichText::new("Checking…").color(COLOR_ACTION));
        } else if ui
            .button(RichText::new("Re-check").color(COLOR_HIGHLIGHT))
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
        .max_height(520.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for status in &tools.statuses {
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
                });

                ui.indent(spec.slug, |ui| {
                    ui.label(RichText::new(spec.summary).color(COLOR_INACTIVE).small());

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
                            // The install command rather than the raw "not found" message: the
                            // useful part of a missing tool is what to do about it.
                            ui.label(
                                RichText::new(format!("Install with: {}", spec.install_command()))
                                    .color(COLOR_HIGHLIGHT)
                                    .small()
                                    .monospace(),
                            );
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
            "Installers live in install_scripts/. Run install_tool with a tool's name, or `all`.",
        )
        .color(COLOR_INACTIVE)
        .small(),
    );
    ui.add_space(ROW_SPACING);
    close_btn(ui, &mut state.ui.popup.external_tools);
}
