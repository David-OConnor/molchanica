//! The MDposit window: find and download trajectories for a PDB ID.

use std::{
    io,
    path::PathBuf,
    sync::mpsc::{self, TryRecvError},
    thread,
    time::Duration,
};

use bio_apis::mdposit::{Client, TrajectoryFormat};
use egui::{Button, Color32, ComboBox, Context, DragValue, Pos2, RichText, Ui, Window};
use graphics::{EngineUpdates, Scene};

use crate::{
    md::{
        mdposit::{self, Download, Import, JobResult, QuickTraj},
        trajectory::{Trajectory, TrajectorySource, view_traj},
        viewer::ViewerMolSet,
    },
    prefs::{MdPrefs, OpenType},
    state::State,
    ui::COLOR_ACTION,
    util::{RedrawFlags, handle_err, handle_success},
};

impl Import {
    pub fn open(&mut self, pdb_id: String, anchor: Pos2, ctx: &Context) {
        if self.worker.is_some() {
            self.anchor = Some(anchor);
            return;
        }

        *self = Self {
            anchor: Some(anchor),
            step: 1,
            quick: self.quick.take(),
            ..Self::default()
        };
        self.search(pdb_id, 1, ctx);
    }

    /// Start a Quick Traj download: no window, and it plays once ready. `residues` is the open
    /// structure's amino acid count, for matching. Ignored while one is in progress.
    pub fn quick(
        &mut self,
        pdb_id: String,
        residues: usize,
        md: &MdPrefs,
        prefs_dir: PathBuf,
        ctx: &Context,
    ) {
        if self.quick.is_some() {
            return;
        }

        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let id = pdb_id.clone();
        let (target_mb, span) = (md.quick_traj_target_mb, md.quick_traj_span);

        thread::spawn(move || {
            let _ = tx.send(mdposit::quick_download(
                &id, residues, target_mb, span, &prefs_dir,
            ));
            ctx.request_repaint();
        });
        self.quick = Some((pdb_id, rx));
    }

    pub fn quick_running(&self) -> bool {
        self.quick.is_some()
    }

    fn poll_quick(&mut self) -> Option<(String, io::Result<Option<QuickTraj>>)> {
        let (_, receiver) = self.quick.as_ref()?;
        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err(io::Error::other("MDposit worker stopped")),
        };
        let (pdb_id, _) = self.quick.take()?;

        Some((pdb_id, result))
    }

    fn launch(
        &mut self,
        status: &'static str,
        ctx: &Context,
        job: impl FnOnce() -> io::Result<JobResult> + Send + 'static,
    ) {
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        self.error = None;
        self.worker = Some((status, rx));

        thread::spawn(move || {
            let _ = tx.send(job());
            ctx.request_repaint();
        });
    }

    fn search(&mut self, pdb_id: String, page: usize, ctx: &Context) {
        self.results = None;
        self.launch("Finding simulations…", ctx, move || {
            mdposit::search(&Client::new(), pdb_id, page).map(JobResult::Search)
        });
    }

    fn poll(&mut self) -> Option<(Trajectory, ViewerMolSet)> {
        let (_, receiver) = self.worker.as_ref()?;
        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err(io::Error::other("MDposit worker stopped")),
        };
        self.worker = None;

        match result {
            Ok(JobResult::Search(results)) => {
                self.project = 0;
                self.replica = 0;
                self.results = Some(results);
                self.reset_range();
            }
            Ok(JobResult::Download(trajectory, set)) => {
                self.anchor = None;
                return Some((trajectory, set));
            }
            Err(error) => self.error = Some(error.to_string()),
        }

        None
    }

    fn reset_range(&mut self) {
        self.first = 1;
        self.last = self
            .results
            .as_ref()
            .and_then(|results| results.projects.get(self.project))
            .and_then(|project| project.replicas().get(self.replica))
            .and_then(|replica| replica.frames)
            .unwrap_or_default();
        self.step = mdposit::default_step(self.last);
    }

    fn options(&mut self, ui: &mut Ui) -> Option<Download> {
        let results = self.results.as_ref()?;
        if results.projects.is_empty() {
            ui.label(format!(
                "No MDposit simulations found for {}.",
                results.pdb_id
            ));
            return None;
        }

        let old_project = self.project;
        ComboBox::from_id_salt("mdposit-project")
            .selected_text(results.projects[self.project].label())
            .width(330.)
            .truncate()
            .show_ui(ui, |ui| {
                for (i, project) in results.projects.iter().enumerate() {
                    ui.selectable_value(&mut self.project, i, project.label());
                }
            });
        let project = &results.projects[self.project];
        if self.project != old_project {
            self.replica = 0;
        }

        let old_replica = self.replica;
        let replicas = project.replicas();
        if replicas.len() > 1 {
            ComboBox::from_id_salt("mdposit-replica")
                .selected_text(replicas[self.replica].label(self.replica))
                .show_ui(ui, |ui| {
                    for (i, replica) in replicas.iter().enumerate() {
                        ui.selectable_value(&mut self.replica, i, replica.label(i));
                    }
                });
        }
        let frames = replicas
            .get(self.replica)
            .and_then(|replica| replica.frames)
            .unwrap_or_default();
        if self.project != old_project || self.replica != old_replica {
            self.first = 1;
            self.last = frames;
            self.step = mdposit::default_step(frames);
        }
        if frames == 0 {
            ui.label("This replica has no trajectory frames.");
            return None;
        }

        let format = self.format.get_or_insert(TrajectoryFormat::Xtc);
        ui.horizontal(|ui| {
            ui.label("Format");
            ui.selectable_value(format, TrajectoryFormat::Xtc, "XTC");
            ui.selectable_value(format, TrajectoryFormat::Trr, "TRR");
            ui.selectable_value(format, TrajectoryFormat::Dcd, "DCD");
        });
        ui.horizontal(|ui| {
            ui.label("Frames");
            ui.add(DragValue::new(&mut self.first).range(1..=frames));
            ui.label("to");
            ui.add(DragValue::new(&mut self.last).range(1..=frames));
            ui.label("step");
            ui.add(DragValue::new(&mut self.step).range(1..=frames))
                .on_hover_text("Download every nth frame in the selected range.");
        });
        ui.checkbox(&mut self.protein_only, "Protein atoms only")
            .on_hover_text("Apply the same protein selection to the structure and trajectory.");

        let valid =
            self.first > 0 && self.last >= self.first && self.last <= frames && self.step > 0;
        if valid {
            let count = (self.last - self.first) / self.step + 1;
            ui.label(format!("{count} of {frames} frames selected"));
        } else {
            ui.label("The end frame must be at or after the start.");
        }

        ui.add_enabled(
            valid,
            Button::new(RichText::new("Download").color(COLOR_ACTION)),
        )
        .clicked()
        .then(|| Download {
            accession: project.accession.clone(),
            replica: self.replica + 1,
            first: self.first,
            last: self.last,
            step: self.step,
            format: *format,
            protein_only: self.protein_only,
        })
    }
}

/// Add a downloaded trajectory and its structure to the open trajectories. Returns its index.
fn add_download(state: &mut State, trajectory: Trajectory, set: ViewerMolSet) -> Option<usize> {
    let path = match &trajectory.source {
        TrajectorySource::File(path) => path.clone(),
        TrajectorySource::Memory(_) => unreachable!(),
    };

    mdposit::register(state, trajectory, Some(set));
    state.update_history(&path, OpenType::Trajectory, None);
    state.to_save.save_flag = true;
    state.ui.ui_vis.sidebar = true;

    state
        .trajectories
        .iter()
        .position(|t| matches!(&t.source, TrajectorySource::File(p) if *p == path))
}

/// Load a finished Quick Traj download's frames, and play them on a loop.
fn finish_quick(
    state: &mut State,
    pdb_id: &str,
    result: io::Result<Option<QuickTraj>>,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
) {
    let quick = match result {
        Ok(Some(quick)) => quick,
        // Not an error; MDposit has simulations of only some structures.
        Ok(None) => {
            handle_success(
                &mut state.ui,
                format!("Can't find a MdPosit trajectory for {pdb_id}"),
            );
            return;
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Error downloading an MDposit trajectory for {pdb_id}: {e}"),
            );
            return;
        }
    };

    let Some(i) = add_download(state, quick.trajectory, quick.set) else {
        return;
    };

    if let Err(e) = view_traj(state, i, scene, updates, redraw) {
        handle_err(
            &mut state.ui,
            format!("Error loading snapshots from trajectory: {e:?}"),
        );
        return;
    }

    let viewer = &mut state.volatile.md_local.viewer;
    viewer.playing = true;
    viewer.looping = true;
    viewer.playback_accum = 0.;

    handle_success(&mut state.ui, format!("Playing {}", quick.summary));
}

pub(in crate::ui) fn show(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    ctx: &Context,
) {
    if let Some((pdb_id, result)) = state.volatile.mdposit.poll_quick() {
        finish_quick(state, &pdb_id, result, scene, updates, redraw);
        ctx.request_repaint();
    } else if state.volatile.mdposit.quick_running() {
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    let importer = &mut state.volatile.mdposit;
    if let Some((trajectory, set)) = importer.poll() {
        add_download(state, trajectory, set);
        handle_success(
            &mut state.ui,
            "MDposit trajectory ready. Use Load under MD Trajectories in the sidebar.".to_owned(),
        );
        ctx.request_repaint();
        return;
    }

    let Some(anchor) = importer.anchor else {
        return;
    };

    let mut open = true;
    let mut download = None;
    Window::new("MDposit trajectory")
        .open(&mut open)
        .default_pos(anchor)
        .default_width(360.)
        .resizable(false)
        .show(ctx, |ui| {
            if let Some((status, _)) = &importer.worker {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(*status);
                });
                return;
            }

            if let Some(error) = &importer.error {
                ui.colored_label(Color32::LIGHT_RED, error);
            }
            if let Some(results) = &importer.results
                && results.pages > 1
            {
                let page = ui
                    .horizontal(|ui| {
                        let prev = ui.add_enabled(results.page > 1, Button::new("Previous"));
                        ui.label(format!("Page {} of {}", results.page, results.pages));
                        let next =
                            ui.add_enabled(results.page < results.pages, Button::new("Next"));

                        if prev.clicked() {
                            Some(results.page - 1)
                        } else if next.clicked() {
                            Some(results.page + 1)
                        } else {
                            None
                        }
                    })
                    .inner;
                if let Some(page) = page {
                    importer.search(results.pdb_id.clone(), page, ctx);
                }
            }
            download = importer.options(ui);
        });

    if let Some(download) = download {
        let prefs_dir = state.volatile.prefs_dir.clone();
        importer.launch("Downloading structure and trajectory…", ctx, move || {
            download
                .save(&prefs_dir)
                .map(|(trajectory, set)| JobResult::Download(trajectory, set))
        });
    }
    if !open {
        importer.anchor = None;
    }
    if importer.worker.is_some() {
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}
