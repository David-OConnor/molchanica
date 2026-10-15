//! The third-party tools status panel. Used to install, uninstall, and view the
//! status of each of these tools. These include MD engines, ML-based protein
//! design tools, ORCA, Gemmi, and others.
//!
//! The core tool functionality is handled by the [`bio_tools`] dependency; Molchanica-native
//! functionality is primarily for the UI, and integrating tool inputs and outputs
//! with Molchanica's native types and molecule state.

use std::{
    collections::{HashMap, VecDeque},
    fs, io, mem,
    panic::{self, AssertUnwindSafe},
    sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc},
    thread,
    time::Duration,
};

use egui::{Color32, RichText, ScrollArea, Ui};

use crate::{
    external_tools::{self, CheckResult, Tool, ToolCheckUpdate, ToolStatus},
    state::{ComputationType, Computations},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, util::open_dir},
};

type DiskUsage = Result<Option<u64>, String>;

/// Panel state: the last probe's results, the probes in flight, and the install queue.
#[derive(Default)]
pub struct ExternalToolsUi {
    statuses: HashMap<Tool, ToolStatus>,
    /// Probes in flight: a full one from [`Self::start_probe`], and single-tool ones started as
    /// each queued job finishes.
    probes: Vec<mpsc::Receiver<ToolCheckUpdate>>,
    queue: JobQueue,
    install_results: HashMap<Tool, InstallResult>,
    disk_usage: HashMap<Tool, DiskUsage>,
    uninstall_results: HashMap<Tool, UninstallResult>,
    confirm_uninstall: Option<Tool>,
    /// Set once, so opening the panel runs a probe without the user having to ask.
    probed_once: bool,
    /// Tool whose full detail text is expanded.
    expanded: Option<Tool>,
    /// Case-insensitive filter for the tool list.
    search: String,
    /// Error from attempting to open the managed installation directory.
    install_folder_error: Option<String>,
}

enum InstallResult {
    Success,
    Failed(String),
}

enum UninstallResult {
    Success,
    Failed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobKind {
    Install,
    Uninstall,
}

impl JobKind {
    fn verb(self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::Uninstall => "uninstall",
        }
    }

    fn progress(self) -> &'static str {
        match self {
            Self::Install => "Installing",
            Self::Uninstall => "Uninstalling",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ToolJob {
    tool: Tool,
    kind: JobKind,
}

impl ToolJob {
    /// Blocks for as long as the installer or uninstaller runs.
    fn run(self) -> Result<(), String> {
        let result = match self.kind {
            JobKind::Install => external_tools::install(self.tool),
            JobKind::Uninstall => external_tools::uninstall(self.tool),
        };
        result.map_err(|error| error.to_string())
    }
}

/// The install queue's state, shared between the panel and its worker thread.
#[derive(Default)]
struct QueueState {
    /// Jobs not yet started, in the order they will run.
    waiting: VecDeque<ToolJob>,
    running: Option<ToolJob>,
    /// Finished jobs the panel has not collected yet.
    finished: Vec<(ToolJob, Result<(), String>)>,
    /// Whether a worker thread is draining `waiting`. There is at most one, so installers never
    /// compete with each other for the network, the disk, or uv's cache.
    worker_running: bool,
}

/// Installs and uninstalls, run one at a time on a worker thread, in the order they were queued.
///
/// A locked queue rather than a channel, so waiting jobs can be cancelled. The worker owns a
/// handle to it, so the queue keeps draining while the panel is closed.
#[derive(Clone, Default)]
struct JobQueue {
    state: Arc<Mutex<QueueState>>,
}

impl JobQueue {
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        // The lock is never held across a job, so a poisoned one still holds a consistent queue.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn snapshot(&self) -> QueueSnapshot {
        let state = self.lock();
        QueueSnapshot {
            running: state.running,
            waiting: state.waiting.iter().copied().collect(),
        }
    }

    /// Queue a job, starting the worker if it is idle. Does nothing if the tool already has a job
    /// running or waiting.
    fn push(&self, job: ToolJob, context: &egui::Context, computations: &Computations) {
        let mut state = self.lock();
        let already_queued = state.running.is_some_and(|queued| queued.tool == job.tool)
            || state.waiting.iter().any(|queued| queued.tool == job.tool);
        if already_queued {
            return;
        }

        state.waiting.push_back(job);
        if state.worker_running {
            return;
        }
        state.worker_running = true;
        drop(state);

        let queue = self.clone();
        let context = context.clone();
        let computations = computations.clone();
        thread::spawn(move || queue.drain(&context, &computations));
    }

    /// Remove a tool's job, if it has not started yet. A running installer can't be interrupted.
    fn cancel(&self, tool: Tool) {
        self.lock().waiting.retain(|job| job.tool != tool);
    }

    fn cancel_waiting(&self) {
        self.lock().waiting.clear();
    }

    fn take_finished(&self) -> Vec<(ToolJob, Result<(), String>)> {
        mem::take(&mut self.lock().finished)
    }

    /// The worker: run waiting jobs one after another, until none are left.
    fn drain(&self, context: &egui::Context, computations: &Computations) {
        loop {
            let job = {
                let mut state = self.lock();
                let Some(job) = state.waiting.pop_front() else {
                    // Cleared under the same lock `push` checks, so a job pushed after this
                    // starts a new worker rather than waiting on this one.
                    state.worker_running = false;
                    return;
                };
                state.running = Some(job);
                job
            };

            let computation = computations.start(
                ComputationType::Install,
                job.tool.spec().name(),
                Some(job.kind.progress().to_lowercase()),
            );

            // A panicking installer must not take the worker down with it, or every job queued
            // behind it would wait forever.
            let result = panic::catch_unwind(AssertUnwindSafe(|| job.run())).unwrap_or_else(|_| {
                Err(format!(
                    "the {}er stopped without reporting a result",
                    job.kind.verb()
                ))
            });
            drop(computation);

            {
                let mut state = self.lock();
                state.running = None;
                state.finished.push((job, result));
            }
            context.request_repaint();
        }
    }
}

/// The queue as of this frame, so drawing the rows doesn't take the lock once per tool.
struct QueueSnapshot {
    running: Option<ToolJob>,
    waiting: Vec<ToolJob>,
}

/// Where a tool's job is in the queue.
enum QueuePlace {
    Running(JobKind),
    /// 1-based position among the waiting jobs.
    Waiting(JobKind, usize),
}

impl QueueSnapshot {
    fn is_empty(&self) -> bool {
        self.running.is_none() && self.waiting.is_empty()
    }

    fn place(&self, tool: Tool) -> Option<QueuePlace> {
        if let Some(job) = self.running.filter(|job| job.tool == tool) {
            return Some(QueuePlace::Running(job.kind));
        }

        self.waiting
            .iter()
            .position(|job| job.tool == tool)
            .map(|i| QueuePlace::Waiting(self.waiting[i].kind, i + 1))
    }
}

impl ExternalToolsUi {
    fn is_probing(&self) -> bool {
        !self.probes.is_empty()
    }

    /// Start a probe of every tool on a worker thread, superseding any probe in flight.
    ///
    /// Off the UI thread because a probe launches a subprocess per tool, and the Python-based ones
    /// import Torch before answering — seconds each, on a cold cache. Blocking the UI for that
    /// would be worse than the problem this panel solves.
    fn start_probe(&mut self) {
        self.probes.clear();
        self.statuses.clear();
        self.disk_usage.clear();
        self.probes.push(external_tools::check_all());
        self.probed_once = true;
        self.confirm_uninstall = None;
    }

    /// Re-probe one tool without disturbing the other rows, e.g. once its queued job finishes.
    fn recheck(&mut self, tool: Tool) {
        self.statuses.remove(&tool);
        self.disk_usage.remove(&tool);
        self.probes.push(external_tools::check_tools(vec![tool]));
    }

    fn poll(&mut self) {
        let statuses = &mut self.statuses;
        let disk_usage = &mut self.disk_usage;

        self.probes.retain(|receiver| {
            loop {
                match receiver.try_recv() {
                    Ok(ToolCheckUpdate::Status(status)) => {
                        statuses.insert(status.tool, status);
                    }
                    Ok(ToolCheckUpdate::ManagedDiskUsage { tool, result }) => {
                        disk_usage.insert(tool, result.map_err(|error| error.to_string()));
                    }
                    Err(mpsc::TryRecvError::Empty) => break true,
                    // The worker either finished normally or died. In either case, drop the
                    // receiver so the panel can be retried rather than showing "checking" forever.
                    Err(mpsc::TryRecvError::Disconnected) => break false,
                }
            }
        });
    }

    /// Queue one installer. Installers can download large model environments, so they run on the
    /// queue's worker thread and never block egui's render thread.
    fn queue_install(&mut self, tool: Tool, context: &egui::Context, computations: &Computations) {
        if !tool.spec().can_install_here() {
            return;
        }

        self.confirm_uninstall = None;
        self.install_results.remove(&tool);
        self.uninstall_results.remove(&tool);

        let job = ToolJob {
            tool,
            kind: JobKind::Install,
        };
        self.queue.push(job, context, computations);
    }

    fn queue_uninstall(
        &mut self,
        tool: Tool,
        context: &egui::Context,
        computations: &Computations,
    ) {
        let has_managed_files =
            matches!(self.disk_usage.get(&tool), Some(Ok(Some(_))) | Some(Err(_)));
        if !tool.spec().molchanica_managed || !has_managed_files {
            return;
        }

        self.confirm_uninstall = None;
        self.install_results.remove(&tool);
        self.uninstall_results.remove(&tool);

        let job = ToolJob {
            tool,
            kind: JobKind::Uninstall,
        };
        self.queue.push(job, context, computations);
    }

    /// Record the results of jobs that finished since the last frame, and return those jobs.
    fn collect_finished(&mut self) -> Vec<ToolJob> {
        let mut jobs = Vec::new();

        for (job, result) in self.queue.take_finished() {
            match job.kind {
                JobKind::Install => {
                    let result = match result {
                        Ok(()) => InstallResult::Success,
                        Err(error) => InstallResult::Failed(error),
                    };
                    self.install_results.insert(job.tool, result);
                }
                JobKind::Uninstall => {
                    let result = match result {
                        Ok(()) => UninstallResult::Success,
                        Err(error) => UninstallResult::Failed(error),
                    };
                    self.uninstall_results.insert(job.tool, result);
                }
            }
            jobs.push(job);
        }

        jobs
    }

    /// How many tools are ready, for a one-line summary elsewhere in the UI.
    pub fn ready_count(&self) -> usize {
        self.statuses
            .values()
            .filter(|status| status.result == CheckResult::Pass)
            .count()
    }
}

/// Where a tool's job is in the queue, with a way to take it back out while it waits.
fn queue_place_ui(ui: &mut Ui, queue: &JobQueue, place: &QueuePlace, tool: Tool) {
    match place {
        QueuePlace::Running(kind) => {
            ui.spinner();
            ui.label(RichText::new(format!("{}…", kind.progress())).color(COLOR_ACTION));
        }
        QueuePlace::Waiting(kind, position) => {
            ui.label(
                RichText::new(format!("Queued to {} (#{position})", kind.verb()))
                    .color(COLOR_ACTION),
            );
            if ui
                .small_button("Cancel")
                .on_hover_text("Remove this from the queue.")
                .clicked()
            {
                queue.cancel(tool);
            }
        }
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

fn open_install_folder() -> io::Result<()> {
    let data_root = external_tools::data_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "unable to determine Molchanica's data directory",
        )
    })?;

    let folder = data_root.join("process_executables");
    fs::create_dir_all(&folder)?;

    open_dir(&folder)
}

pub fn external_tools_window(state: &mut crate::state::State, ui: &mut Ui) {
    let context = ui.ctx().clone();
    let computations = &state.volatile.ongoing_computations;
    let tools = &mut state.ui.external_tools;
    tools.poll();

    let finished = tools.collect_finished();
    if !finished.is_empty() {
        if finished.iter().any(|job| job.tool == Tool::RdKit) {
            // A Python executable can exist before pip finishes. Revisit any diagrams that
            // fell back to ChEBI during installation, including reinstalls at the same path.
            state.ui.reactions.diagrams = Default::default();
        }

        if tools.queue.snapshot().is_empty() {
            tools.start_probe();
        } else {
            // While the queue still runs, re-probe only the finished tools, so their rows stop
            // offering the action just taken; the full probe waits until the queue is empty.
            for job in &finished {
                tools.recheck(job.tool);
            }
        }
    }

    if !tools.probed_once {
        tools.start_probe();
    }
    if tools.is_probing() {
        context.request_repaint_after(Duration::from_millis(50));
    }

    let queue = tools.queue.snapshot();

    ui.horizontal(|ui| {
        ui.heading("Third-party tools");
        ui.add_space(ROW_SPACING);
        if tools.is_probing() {
            ui.label(RichText::new("Checking…").color(COLOR_ACTION));
        } else if ui
            .add_enabled(
                queue.is_empty(),
                egui::Button::new(RichText::new("Re-check").color(COLOR_HIGHLIGHT)),
            )
            .on_hover_text(
                "Run every tool's version probe again. Do this after installing something, or \
                 after changing an override environment variable.",
            )
            .clicked()
        {
            tools.start_probe();
        }
    });

    if !queue.is_empty() {
        ui.add_space(ROW_SPACING);
        ui.horizontal_wrapped(|ui| {
            if let Some(job) = queue.running {
                ui.spinner();
                ui.label(
                    RichText::new(format!(
                        "{} {}…",
                        job.kind.progress(),
                        job.tool.spec().name()
                    ))
                    .color(COLOR_ACTION),
                );
            }

            if !queue.waiting.is_empty() {
                let waiting: Vec<_> = queue
                    .waiting
                    .iter()
                    .map(|job| format!("{} {}", job.kind.verb(), job.tool.spec().name()))
                    .collect();

                ui.label(
                    RichText::new(format!("Queued: {}", waiting.join(", "))).color(COLOR_INACTIVE),
                );
                if ui
                    .button("Cancel queued")
                    .on_hover_text(
                        "Remove every job that has not started yet. The one in progress runs to \
                         completion.",
                    )
                    .clicked()
                {
                    tools.queue.cancel_waiting();
                }
            }
        });
    }

    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.label("Search:");
        ui.add(
            egui::TextEdit::singleline(&mut tools.search)
                .hint_text("Filter tools by name")
                .desired_width(220.0),
        );
        if ui.button("Open install folder").clicked() {
            tools.install_folder_error = open_install_folder().err().map(|error| error.to_string());
        }
    });

    if let Some(error) = &tools.install_folder_error {
        ui.label(
            RichText::new(format!("Could not open install folder: {error}"))
                .color(Color32::ORANGE)
                .small(),
        );
    }

    ui.add_space(ROW_SPACING);
    ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            let statuses = tools.statuses.clone();
            let mut ordered_tools = Tool::ALL;
            ordered_tools.sort_by_key(|tool| {
                statuses
                    .get(tool)
                    .map_or(u8::MAX, |status| status.result.rank())
            });

            let search = tools.search.trim().to_lowercase();
            let mut visible_tools = 0;

            for tool in ordered_tools {
                let spec = tool.spec();
                if !search.is_empty() && !spec.name().to_lowercase().contains(&search) {
                    continue;
                }
                visible_tools += 1;
                let Some(status) = statuses.get(&tool) else {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("[Checking…]")
                                .color(COLOR_ACTION)
                                .monospace(),
                        );
                        ui.label(RichText::new(spec.name()).strong());
                        if let Some(label) = spec.platform_label() {
                            ui.label(RichText::new(label).color(Color32::LIGHT_BLUE).small());
                        }
                        ui.spinner();
                    });
                    ui.indent(spec.slug(), |ui| {
                        ui.label(RichText::new(spec.summary()).color(COLOR_INACTIVE));
                    });
                    ui.add_space(6.0);
                    continue;
                };

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("[{}]", status.result.label()))
                            .color(result_color(status.result))
                            .monospace(),
                    );
                    ui.label(RichText::new(spec.name()).strong());
                    if let Some(label) = spec.platform_label() {
                        ui.label(
                            RichText::new(label).color(Color32::LIGHT_BLUE).small(),
                        );
                    }

                    if ui
                        .small_button("?")
                        .on_hover_text(format!(
                            "{}\n\nLicence: {} — {}\n{}",
                            spec.summary(),
                            spec.license(),
                            spec.license_details(),
                            spec.url()
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
                        let place = queue.place(status.tool);
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

                        let button_text = if confirming {
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
                                spec.molchanica_managed && has_managed_files && place.is_none(),
                                egui::Button::new(button_text),
                            )
                            .on_hover_text(uninstall_hint);
                        if response.clicked() {
                            if confirming {
                                tools.queue_uninstall(status.tool, &context, computations);
                            } else {
                                tools.confirm_uninstall = Some(status.tool);
                            }
                        }
                        if let Some(place) = &place {
                            queue_place_ui(ui, &tools.queue, place, status.tool);
                        }
                        let size_response =
                            ui.label(RichText::new(disk_text).color(COLOR_INACTIVE).small());
                        if let Some(hover) = disk_hover {
                            size_response.on_hover_text(hover);
                        }
                    }
                });

                ui.indent(spec.slug(), |ui| {
                    ui.label(RichText::new(spec.summary()).color(COLOR_INACTIVE));

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
                            if !spec.is_supported() {
                                ui.label(
                                    RichText::new(
                                        "Linux only — installation is unavailable on this platform.",
                                    )
                                    .color(COLOR_INACTIVE)
                                    .small(),
                                );
                            } else {
                                let place = queue.place(status.tool);
                                ui.horizontal(|ui| {
                                    let response = ui
                                        .add_enabled(
                                            spec.molchanica_managed && place.is_none(),
                                            egui::Button::new("Install"),
                                        )
                                        .on_hover_text(if spec.molchanica_managed {
                                            spec.install_command()
                                        } else {
                                            spec.install_hint.to_owned()
                                        });
                                    if response.clicked() {
                                        tools.queue_install(status.tool, &context, computations);
                                    }
                                    if let Some(place) = &place {
                                        queue_place_ui(ui, &tools.queue, place, status.tool);
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
                        if let Some(name) = spec.root_override_env {
                            ui.label(
                                RichText::new(format!("Environment root override: {name}"))
                                    .color(COLOR_INACTIVE)
                                    .small(),
                            );
                        }
                        if let Some(name) = spec.bundle_root_override_env {
                            ui.label(
                                RichText::new(format!("Tool/data root override: {name}"))
                                    .color(COLOR_INACTIVE)
                                    .small(),
                            );
                        }

                        ui.label(
                            RichText::new(format!(
                                "Licence: {} — {}",
                                spec.license(),
                                spec.license_details()
                            ))
                            .color(COLOR_INACTIVE)
                            .small(),
                        );
                        ui.hyperlink(spec.url());
                    }
                });
                ui.add_space(6.0);
            }
            if visible_tools == 0 {
                ui.label(RichText::new("No tools match the search.").color(COLOR_INACTIVE));
            }
        });

    ui.add_space(ROW_SPACING);
    ui.label(
        RichText::new(
            "Install buttons use the shared bio_tools Rust recipes and place each tool in \
             Molchanica's managed data directory. Installs and uninstalls are queued, and run one \
             at a time in the order clicked; they keep running if this window is closed.",
        )
        .color(COLOR_INACTIVE),
    );
}
