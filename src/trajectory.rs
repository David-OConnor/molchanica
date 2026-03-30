//! For loading trajectories from file, and loading frames to memory as required.

// todo: Consider moving this to a new module with md.rs

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Trajectory {
    pub path: PathBuf,
    // todo: Evaluate how you want to handle this. One or more ranges?
    // todo: Time frames?
    pub frames_open: Vec<usize>,
}
impl Trajectory {
    /// Load this into memory including metadata, but don't load any frames. Supports
    /// TRR, XTC, and DCD formats.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_owned(),
            frames_open: Vec::new(),
        }
    }
}
