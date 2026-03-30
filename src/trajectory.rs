//! For loading trajectories from file, and loading frames to memory as required.

// todo: Consider moving this to a new module with md.rs

// todo: Eval if this should move to bio_files or dynamics. If it doesn't use
// todo any code from Molchanica, the answer is yes.

use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
};

use dynamics::snapshot::Snapshot;

use crate::{prefs::OpenType, state::State};

// Don't let the user attempt to import more than this many frames at once
// todo: What should this be?
pub const MAX_FRAMES_TO_ATTEMPT_LOADING: usize = 100_000;

/// A ay to allow users to slice by either physical time or frame index.
/// Times are in ps.
///
/// None values means unbounded on that side.
#[derive(Clone, Copy, Debug)]
pub enum FrameSlice {
    Time {
        start: Option<f64>,
        end: Option<f64>,
    },
    Index {
        start: Option<usize>,
        end: Option<usize>,
    },
}

/// Represents a MD trajectory file on disk. We use this to load snapshots into memory, and
/// write to them. These may be large, so we load and save at specific intervals.
///
/// Supports TRR, XTC, and DCD. In TRR and XTC, we can infer things like number of frames
/// by transversing the file between frame headers, without reading the coordinate data.
///
/// DCD contains a header; XTR and TRR are stream-based formats which do not have a global header;
/// they are just a series of frame blocks. Each frame has metadata.
///
/// We delegate operations as required to the `bio_files` and `dynamics` libraries, e.g. for
/// i/o with specific files.
///
/// Warning: DCD is not a well-defined format.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub path: PathBuf,
    /// e.g. derived from filename.
    pub display_name: String,
    pub num_atoms: usize,           // In DCD header. Each TRR and XTC frame.
    pub num_frames: usize,          // in DCD header.
    pub start_step: f32,            // In DCD header. Each TRR and XTC frame.
    pub save_interval_steps: usize, // DCD header? todo
    pub dt: f32,                    // In DCD. ps?
    pub end_time: f32,              // Not in the file directly; we calculate this.
    // todo: Evaluate how you want to handle this. One or more ranges?
    // todo: Time frames?
    pub frames_open: Vec<usize>,
    /// This is an odd place for UI input box items, but it will do for now.
    /// Integers are OK directly, but we need an intermediate String for floats.
    pub ui_start_i: usize,
    pub ui_end_i: usize,
    pub ui_start_time: String,
    pub ui_end_time: String,
}
impl Trajectory {
    /// Load this into memory including metadata, but don't load any frames. Supports
    /// TRR, XTC, and DCD formats.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_owned(),
            // i.e. `traj.trr`.
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
            frames_open: Vec::new(),
            ui_start_i: 0,
            ui_end_i: 0,
            ui_start_time: String::new(),
            ui_end_time: String::new(),
        }
    }

    /// Start and end are in ps.
    /// todo: Shouljd they be in indices instead? Allow coth?
    pub fn load_snaps(&self, slice: FrameSlice) -> io::Result<Vec<Snapshot>> {
        let mut snaps = Vec::new();

        Ok(snaps)
    }
}

pub fn close_traj(state: &mut State, i: usize) {
    if i >= state.trajectories.len() {
        eprintln!("Error: Trying to close a trajectory that's out of bounds");
        return;
    }

    let traj = &state.trajectories[i];

    for history in &mut state.to_save.open_history {
        if history.type_ == OpenType::Trajectory && history.path == traj.path {
            history.last_session = false;
        }
    }

    state.trajectories.remove(i);
}
