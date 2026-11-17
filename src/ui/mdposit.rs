//! The MDposit window: find and download trajectories for a PDB ID.

use std::{
    io,
    sync::mpsc::{self, TryRecvError},
    thread,
    time::Duration,
};

use bio_apis::mdposit::{Client, TrajectoryFormat};
use egui::{Button, Color32, ComboBox, Context, DragValue, Pos2, RichText, Ui, Window};

use crate::{
    md::{
        mdposit::{self, Download, Import, JobResult},
        trajectory::{Trajectory, TrajectorySource},
        viewer::ViewerMolSet,
    },
    prefs::OpenType,
    state::State,
    ui::COLOR_ACTION,
    util::handle_success,
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
            ..Self::default()
        };
        self.search(pdb_id, 1, ctx);
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

pub(in crate::ui) fn show(state: &mut State, ctx: &Context) {
    let importer = &mut state.volatile.mdposit;
    if let Some((trajectory, set)) = importer.poll() {
        let path = match &trajectory.source {
            TrajectorySource::File(path) => path.clone(),
            TrajectorySource::Memory(_) => unreachable!(),
        };

        mdposit::register(state, trajectory, Some(set));
        state.update_history(&path, OpenType::Trajectory, None);
        state.to_save.save_flag = true;
        state.ui.ui_vis.sidebar = true;
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
            download.save(&prefs_dir)
        });
    }
    if !open {
        importer.anchor = None;
    }
    if importer.worker.is_some() {
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}
