//! For loading trajectories from file, and loading frames to memory as required.

// todo: Consider moving this to a new module with md.rs

// todo: Eval if this should move to bio_files or dynamics. If it doesn't use
// todo any code from Molchanica, the answer is yes.

use std::{
    io,
    path::{Path, PathBuf},
};

use bio_files::{
    FrameSlice,
    dcd::{DcdMetadata, read_dcd},
    gromacs::output::{TrrMetadata, read_trr},
    xtc::{XtcMetadata, read_xtc},
};
use dynamics::snapshot::Snapshot;

use crate::{prefs::OpenType, state::State};

// Don't let the user attempt to import more than this many frames at once
// todo: What should this be?
pub const MAX_FRAMES_TO_ATTEMPT_LOADING: usize = 100_000;

/// This affects how we operate on files.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrajFormat {
    Trr,
    Xtc,
    Dcd,
    /// All snapshots are already in memory — no file backing.
    InMemory,
}

impl TrajFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        Some(match ext.to_lowercase().as_str() {
            "trr" => TrajFormat::Trr,
            "xtc" => TrajFormat::Xtc,
            "dcd" => TrajFormat::Dcd,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub enum TrajectorySource {
    File(PathBuf),
    Memory(Vec<Snapshot>),
}

/// Represents a MD trajectory — either a file on disk or a set of snapshots held
/// entirely in memory (e.g. from a Dynamics-engine run).
///
/// Supports TRR, XTC, and DCD on disk.  In TRR and XTC, we can infer things like
/// number of frames by traversing the file between frame headers, without reading
/// the coordinate data.
///
/// `path == None` implies an in-memory trajectory; `snapshots` holds the frames.
///
/// We delegate file I/O to the `bio_files` and `dynamics` libraries.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub format: TrajFormat,
    pub source: TrajectorySource,
    /// e.g. derived from filename, or "In-memory run N".
    pub display_name: String,
    pub num_atoms: usize,           // In DCD header. Each TRR and XTC frame.
    pub num_frames: usize,          // in DCD header.
    pub start_step: f32,            // In DCD header. Each TRR and XTC frame.
    pub save_interval_steps: usize, // DCD header? todo
    pub dt: f32,                    // In DCD. ps?
    pub end_time: f32,              // Not in the file directly; we calculate this.
    pub frames_open: Option<FrameSlice>,
    /// This is an odd place for UI input box items, but it will do for now.
    /// Integers are OK directly, but we need an intermediate String for floats.
    pub ui_start_i: usize,
    pub ui_end_i: usize,
    pub ui_start_time: String,
    pub ui_end_time: String,
}

impl Trajectory {
    /// Load metadata from a trajectory file on disk, but don't load any frames.
    /// Supports TRR, XTC, and DCD formats.
    pub fn new(path: &Path) -> io::Result<Self> {
        let format = path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(TrajFormat::from_extension)
            .ok_or_else(|| {
                io::Error::other("Error determining trajectory format from extension")
            })?;

        let mut result = Self {
            format,
            source: TrajectorySource::File(path.into()),
            display_name: path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            num_atoms: 0,
            num_frames: 0,
            start_step: 0.,
            save_interval_steps: 0,
            dt: 0.,
            end_time: 0.,
            frames_open: None,
            ui_start_i: 0,
            ui_end_i: 0,
            ui_start_time: String::new(),
            ui_end_time: String::new(),
        };

        match format {
            TrajFormat::Trr => {
                let md = TrrMetadata::read(path)?;

                result.num_atoms = md.num_atoms;
                result.num_frames = md.num_frames;
                result.start_step = md.start_step;
                result.save_interval_steps = md.save_interval_steps;
                result.dt = md.dt;
                result.end_time = md.end_time;
            }
            TrajFormat::Dcd => {
                let md = DcdMetadata::read(path)?;

                result.num_atoms = md.num_atoms;
                result.num_frames = md.num_frames;
                result.start_step = md.start_step;
                result.save_interval_steps = md.save_interval_steps;
                result.dt = md.dt;
                result.end_time = md.end_time;
            }
            TrajFormat::Xtc => {
                let md = XtcMetadata::read(path)?;

                result.num_atoms = md.num_atoms;
                result.num_frames = md.num_frames;
                result.start_step = md.start_time;
                result.dt = md.dt;
                result.end_time = md.end_time;
            }
            TrajFormat::InMemory => unreachable!(),
        }

        Ok(result)
    }

    /// Create an in-memory trajectory from a set of snapshots, e.g. from a
    /// completed Dynamics-engine run.  All frames are immediately available;
    /// no file I/O is required.
    pub fn new_in_memory(snapshots: Vec<Snapshot>, display_name: String, sim_dt: f32) -> Self {
        let num_frames = snapshots.len();
        let first = snapshots.first();
        let last = snapshots.last();

        let start_step = first.map(|s| s.time as f32).unwrap_or(0.);
        let end_time = last.map(|s| s.time as f32).unwrap_or(0.);
        let dt = if snapshots.len() > 1 {
            (snapshots[1].time - snapshots[0].time) as f32
        } else {
            0.
        };

        // Total atom count including water for display purposes.
        let num_atoms = first
            .map(|s| {
                s.atom_posits.len()
                    + s.water_o_posits.len()
                    + s.water_h0_posits.len()
                    + s.water_h1_posits.len()
            })
            .unwrap_or(0);

        Self {
            format: TrajFormat::InMemory,
            source: TrajectorySource::Memory(snapshots),
            display_name,
            num_atoms,
            num_frames,
            start_step,
            save_interval_steps: if sim_dt > 0. && dt > 0. {
                (dt / sim_dt).round() as usize
            } else {
                0
            },
            dt,
            end_time,
            frames_open: None,
            ui_start_i: 0,
            ui_end_i: num_frames.saturating_sub(1),
            ui_start_time: String::new(),
            ui_end_time: String::new(),
        }
    }

    /// Load frames, applying a `FrameSlice` filter.  For file-backed trajectories
    /// this reads from disk; for in-memory trajectories it filters the stored snapshots.
    pub fn load_snaps(&mut self, slice: FrameSlice) -> io::Result<Vec<Snapshot>> {
        self.frames_open = Some(slice);

        match self.format {
            TrajFormat::InMemory => {
                let TrajectorySource::Memory(snaps) = &self.source else {
                    return Err(io::Error::other(
                        "Error: TrajFormat In memory, but source is not memory.",
                    ));
                };

                let filtered = match slice {
                    FrameSlice::Index { start, end } => {
                        let s = start.unwrap_or(0);
                        let e = end.unwrap_or(snaps.len().saturating_sub(1));
                        snaps
                            .get(s..=e.min(snaps.len().saturating_sub(1)))
                            .unwrap_or(&[])
                            .to_vec()
                    }
                    FrameSlice::Time { start, end } => snaps
                        .iter()
                        .filter(|s| {
                            start.map_or(true, |t| s.time >= t) && end.map_or(true, |t| s.time <= t)
                        })
                        .cloned()
                        .collect(),
                };
                Ok(filtered)
            }

            TrajFormat::Trr => {
                let TrajectorySource::File(path) = &self.source else {
                    return Err(io::Error::other(
                        "Error: TrajFormat not in memory, but not in memory source",
                    ));
                };

                Ok(read_trr(&path, slice)?
                    .into_iter()
                    .map(Snapshot::from)
                    .collect())
            }

            TrajFormat::Xtc => {
                let TrajectorySource::File(path) = &self.source else {
                    return Err(io::Error::other(
                        "Error: TrajFormat not in memory, but not in memory source",
                    ));
                };

                Ok(read_xtc(&path, slice)?
                    .into_iter()
                    .map(Snapshot::from)
                    .collect())
            }

            TrajFormat::Dcd => {
                let TrajectorySource::File(path) = &self.source else {
                    return Err(io::Error::other(
                        "Error: TrajFormat not in memory, but not in memory source",
                    ));
                };

                Ok(read_dcd(&path, slice)?
                    .into_iter()
                    .map(Snapshot::from)
                    .collect())
            }
        }
    }
}

pub fn close_traj(state: &mut State, i: usize) {
    if i >= state.trajectories.len() {
        eprintln!("Error: Trying to close a trajectory that's out of bounds");
        return;
    }

    let traj = &state.trajectories[i];

    if let TrajectorySource::File(path) = &traj.source {
        for history in &mut state.to_save.open_history {
            if history.type_ == OpenType::Trajectory && &history.path == path {
                history.last_session = false;
            }
        }
    }

    state.trajectories.remove(i);
}
