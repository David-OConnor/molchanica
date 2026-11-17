//! For loading trajectories from file, and loading frames to memory as required.

// todo: Eval if this should move to bio_files. If it doesn't use
// todo any code from Molchanica, the answer is yes.

use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use bio_files::{
    AtomGeneric, FrameSlice,
    dcd::{DcdMetadata, read_dcd},
    gromacs::OutputEnergy,
    trr::{TrrMetadata, read_trr},
    xtb::XtbFrame,
    xtc::{XtcMetadata, read_xtc},
};
use dynamics::snapshot::Snapshot;
use lin_alg::f32::Vec3;

use crate::{prefs::OpenType, state::State};

// Don't let the user attempt to import more than this many frames at once
// todo: What should this be?
pub const MAX_FRAMES_TO_ATTEMPT_LOADING: usize = 100_000;

const GB: f64 = 1e9;

pub const MAX_AUTO_LOAD_GB_DEFAULT: f32 = 6.;
/// Share of available system memory a load may use.
const AVAIL_MEM_FRACTION: f64 = 0.8;

/// Which limits an auto load applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadKind {
    /// Reading frames from disk: the setting, available memory, and the frame count.
    Disk,
    /// Frames to be produced, e.g. by an MD run: the setting, and available memory.
    Pending,
    /// Frames already in memory: the setting.
    InMemory,
}

/// What forced an auto load's frame step above 1.
#[derive(Clone, Copy, Debug)]
pub enum LoadLimit {
    /// The max auto load size setting, in bytes.
    Setting(f64),
    /// Usable system memory, in bytes.
    AvailMem(f64),
    FrameCount,
}

impl LoadLimit {
    /// E.g. "to fit the 6.0 GB max auto load size".
    pub fn describe(&self) -> String {
        match self {
            Self::Setting(max) => format!("to fit the {} max auto load size", format_bytes(*max)),
            Self::AvailMem(avail) => {
                format!("to fit available memory ({} usable)", format_bytes(*avail))
            }
            Self::FrameCount => format!("to stay under {MAX_FRAMES_TO_ATTEMPT_LOADING} frames"),
        }
    }
}

/// A frame step that fits a trajectory load under memory limits.
#[derive(Clone, Copy, Debug)]
pub struct AutoLoad {
    pub step: usize,
    /// Set if `step` is above 1.
    pub limit: Option<LoadLimit>,
}

impl Default for AutoLoad {
    fn default() -> Self {
        Self {
            step: 1,
            limit: None,
        }
    }
}

impl AutoLoad {
    /// The smallest step that fits `n_frames` under the limits. `frame_bytes` is memory per loaded
    /// frame; `peak_bytes` is per frame while loading.
    pub fn new(
        n_frames: usize,
        frame_bytes: usize,
        peak_bytes: usize,
        max_gb: f32,
        kind: LoadKind,
    ) -> Self {
        let max = f64::from(max_gb.max(0.)) * GB;
        let avail = if kind == LoadKind::InMemory {
            None
        } else {
            available_memory().map(|m| m as f64 * AVAIL_MEM_FRACTION)
        };

        let limit = |step: usize| {
            let n = n_frames.div_ceil(step);

            if kind == LoadKind::Disk && n > MAX_FRAMES_TO_ATTEMPT_LOADING {
                Some(LoadLimit::FrameCount)
            } else if (n * frame_bytes) as f64 > max {
                Some(LoadLimit::Setting(max))
            } else if let Some(a) = avail
                && (n * peak_bytes) as f64 > a
            {
                Some(LoadLimit::AvailMem(a))
            } else {
                None
            }
        };

        let mut step = 1;
        let mut binding = None;
        while step < n_frames {
            let Some(l) = limit(step) else { break };

            binding = Some(l);
            step += 1;
        }

        Self {
            step,
            limit: binding,
        }
    }

    /// Applies `next` to frames this already stepped.
    pub fn then(self, next: Self) -> Self {
        Self {
            step: self.step * next.step,
            limit: next.limit.or(self.limit),
        }
    }

    /// Out section text, for `snaps` loaded from `total` frames.
    pub fn summary(&self, snaps: &[Arc<Snapshot>], total: usize) -> String {
        load_summary(snaps, Some(total), self.step, self.limit)
    }
}

/// Available system memory, in bytes.
fn available_memory() -> Option<u64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let avail = sys.available_memory();

    (avail > 0).then_some(avail)
}

/// Bytes one snapshot holds, including its heap data.
pub fn snapshot_bytes(snap: &Snapshot) -> usize {
    let opt = |v: &Option<Vec<Vec3>>| v.as_deref().map_or(0, size_of_val);
    let energy = snap.energy_data.as_ref().map_or(0, |e| {
        size_of_val(e.energy_potential_between_mols.as_slice())
            + size_of_val(e.hydrogen_bonds.as_slice())
    });

    size_of::<Snapshot>()
        + size_of_val(snap.atom_posits.as_slice())
        + size_of_val(snap.water_o_posits.as_slice())
        + size_of_val(snap.water_h0_posits.as_slice())
        + size_of_val(snap.water_h1_posits.as_slice())
        + opt(&snap.atom_velocities)
        + opt(&snap.water_velocities)
        + opt(&snap.force)
        + energy
}

/// Bytes per snapshot holding positions only.
pub fn posits_frame_bytes(num_atoms: usize) -> usize {
    size_of::<Snapshot>() + num_atoms * size_of::<Vec3>()
}

/// Bytes per parsed xTB trajectory frame. These are held until converted to snapshots.
pub fn xtb_frame_bytes(num_atoms: usize) -> usize {
    // Plus the comment line.
    size_of::<XtbFrame>() + num_atoms * size_of::<AtomGeneric>() + 128
}

/// E.g. "1.2 GB", "340 MB".
pub fn format_bytes(bytes: f64) -> String {
    if bytes >= GB {
        return format!("{:.1} GB", bytes / GB);
    }

    let mb = bytes / 1e6;
    if mb < 10. {
        format!("{mb:.1} MB")
    } else {
        format!("{mb:.0} MB")
    }
}

/// Out section text for frames loaded into memory.
pub fn load_summary(
    snaps: &[Arc<Snapshot>],
    total: Option<usize>,
    step: usize,
    limit: Option<LoadLimit>,
) -> String {
    let bytes: usize = snaps.iter().map(|s| snapshot_bytes(s)).sum();
    let n = match total {
        Some(total) => format!("{}/{total}", snaps.len()),
        None => snaps.len().to_string(),
    };

    let mut result = format!(
        "Loaded {n} frames into memory ({}), step {step}",
        format_bytes(bytes as f64)
    );
    if let Some(limit) = limit {
        result += &format!(", {}", limit.describe());
    }
    result.push('.');

    result
}

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
    Memory(Vec<Arc<Snapshot>>),
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
    pub run_mol_set_path: Option<PathBuf>,
    pub energies: Vec<OutputEnergy>,
    pub num_atoms: usize,           // In DCD header. Each TRR and XTC frame.
    pub num_frames: usize,          // in DCD header.
    pub start_step: f32,            // In DCD header. Each TRR and XTC frame.
    pub save_interval_steps: usize, // DCD header? todo
    pub dt: f32,                    // ps between frames.
    pub end_time: f32,              // Not in the file directly; we calculate this.
    /// Estimated memory per loaded frame, in bytes.
    pub frame_bytes: usize,
    pub frames_open: Option<FrameSlice>,
    /// This is an odd place for UI input box items, but it will do for now.
    /// Integers are OK directly, but we need an intermediate String for floats.
    pub ui_start_i: usize,
    pub ui_end_i: usize,
    /// Load every nth frame.
    pub ui_step: usize,
    pub ui_start_time: String,
    pub ui_end_time: String,
    // E.g. loaded frames from this trajectory most recently; used
    // for highlighting it.
    pub ui_active: bool,
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
            run_mol_set_path: None,
            energies: Vec::new(),
            num_atoms: 0,
            num_frames: 0,
            start_step: 0.,
            save_interval_steps: 0,
            dt: 0.,
            end_time: 0.,
            frame_bytes: 0,
            frames_open: None,
            ui_start_i: 0,
            ui_end_i: 0,
            ui_step: 1,
            ui_start_time: String::new(),
            ui_end_time: String::new(),
            ui_active: false,
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

                // Velocities and forces load too, if present.
                let vecs = 1 + usize::from(md.has_velocities) + usize::from(md.has_forces);
                result.frame_bytes =
                    size_of::<Snapshot>() + md.num_atoms * vecs * size_of::<Vec3>();
            }
            TrajFormat::Dcd => {
                let md = DcdMetadata::read(path)?;

                result.num_atoms = md.num_atoms;
                result.num_frames = md.num_frames;
                result.start_step = md.start_step;
                result.save_interval_steps = md.save_interval_steps;
                result.dt = md.dt;
                result.end_time = md.end_time;
                result.frame_bytes = posits_frame_bytes(md.num_atoms);
            }
            TrajFormat::Xtc => {
                let md = XtcMetadata::read(path)?;

                result.num_atoms = md.num_atoms;
                result.num_frames = md.num_frames;
                result.start_step = md.start_time;
                result.dt = md.dt;
                result.end_time = md.end_time;
                result.frame_bytes = posits_frame_bytes(md.num_atoms);
            }
            TrajFormat::InMemory => unreachable!(),
        }

        result.ui_end_i = result.num_frames.saturating_sub(1);

        Ok(result)
    }

    /// Create an in-memory trajectory from a set of snapshots, e.g. from a
    /// completed Dynamics-engine run.  All frames are immediately available;
    /// no file I/O is required.
    pub fn new_in_memory(snapshots: Vec<Arc<Snapshot>>, display_name: String, sim_dt: f32) -> Self {
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

        let frame_bytes = if num_frames > 0 {
            snapshots.iter().map(|s| snapshot_bytes(s)).sum::<usize>() / num_frames
        } else {
            0
        };

        Self {
            format: TrajFormat::InMemory,
            source: TrajectorySource::Memory(snapshots),
            display_name,
            run_mol_set_path: None,
            energies: Vec::new(),
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
            frame_bytes,
            frames_open: None,
            ui_start_i: 0,
            ui_end_i: num_frames.saturating_sub(1),
            ui_step: 1,
            ui_start_time: String::new(),
            ui_end_time: String::new(),
            ui_active: false,
        }
    }

    /// False if the file has no frame times, e.g. a DCD with DELTA = 0, or an XTC with all times 0.
    /// MDposit trajectories are like this.
    pub fn has_times(&self) -> bool {
        self.num_frames < 2 || self.dt != 0.
    }

    /// The file frame index of loaded frame `i`.
    pub fn frame_index(&self, i: usize) -> usize {
        match self.frames_open {
            Some(FrameSlice::Index { start, step, .. }) => start.unwrap_or(0) + i * step.max(1),
            _ => i,
        }
    }

    /// Number of frames the UI range and step select.
    pub fn ui_frames_selected(&self) -> usize {
        if self.ui_start_i > self.ui_end_i || self.ui_end_i >= self.num_frames {
            return 0;
        }

        (self.ui_end_i - self.ui_start_i) / self.ui_step.max(1) + 1
    }

    /// Frame step that fits loading the whole trajectory under the memory limits.
    pub fn auto_load(&self, max_gb: f32) -> AutoLoad {
        // TRR frames are read as f64, then converted to f32 snapshots.
        let peak = if self.format == TrajFormat::Trr {
            2 * self.frame_bytes
        } else {
            self.frame_bytes
        };
        let kind = match self.source {
            TrajectorySource::File(_) => LoadKind::Disk,
            TrajectorySource::Memory(_) => LoadKind::InMemory,
        };

        AutoLoad::new(self.num_frames, self.frame_bytes, peak, max_gb, kind)
    }

    /// Load frames, applying a `FrameSlice` filter.  For file-backed trajectories
    /// this reads from disk; for in-memory trajectories it filters the stored snapshots.
    pub fn load_snaps(&mut self, slice: FrameSlice) -> io::Result<Vec<Arc<Snapshot>>> {
        let result: io::Result<Vec<Arc<Snapshot>>> = match self.format {
            TrajFormat::InMemory => {
                let TrajectorySource::Memory(snaps) = &self.source else {
                    return Err(io::Error::other(
                        "Error: TrajFormat In memory, but source is not memory.",
                    ));
                };

                Ok(snaps
                    .iter()
                    .enumerate()
                    .filter(|(i, snap)| slice.contains(*i, snap.time))
                    .map(|(_, snap)| snap.clone())
                    .collect())
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
                    .map(Arc::new)
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
                    .map(Arc::new)
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
                    .map(Arc::new)
                    .collect())
            }
        };
        let mut snaps = result?;
        if self.run_mol_set_path.is_some() || !self.energies.is_empty() {
            snaps.retain(|snap| !snap.atom_posits.is_empty());
        }
        for snap in &mut snaps {
            let index = self
                .energies
                .partition_point(|en| en.time < snap.time - 1e-5);
            if let Some(en) = self
                .energies
                .get(index)
                .filter(|en| (en.time - snap.time).abs() < 1e-5)
            {
                Arc::make_mut(snap).energy_data = Some(en.clone().into());
            }
        }

        self.frames_open = Some(slice);
        self.ui_active = true;

        Ok(snaps)
    }
}

/// Marks trajectory `i` as the one whose frames the viewer holds. The viewer holds one trajectory's
/// frames at a time, so loading one frees the others'.
pub fn set_viewed(trajectories: &mut [Trajectory], i: Option<usize>) {
    for (j, traj) in trajectories.iter_mut().enumerate() {
        traj.ui_active = Some(j) == i;

        if !traj.ui_active {
            traj.frames_open = None;
        }
    }
}

/// Drop a trajectory's loaded frames from memory, but keep it open.
pub fn unload_traj(
    state: &mut State,
    i: usize,
    scene: &mut graphics::Scene,
    redraw: &mut crate::util::RedrawFlags,
    updates: &mut graphics::EngineUpdates,
) {
    let Some(traj) = state.trajectories.get_mut(i) else {
        eprintln!("Error: Trying to unload a trajectory that's out of bounds");
        return;
    };

    let showing_frames = traj.ui_active;
    traj.frames_open = None;
    traj.ui_active = false;

    if showing_frames {
        let md = &mut state.volatile.md_local;

        // Keep the mol set selection, so loading again pairs with it.
        let mol_set_active = md.viewer.mol_set_active;
        md.clear_snaps(&mut scene.entities, redraw);
        md.viewer.mol_set_active = mol_set_active;

        updates.entities = graphics::EntityUpdate::All;
    }
}

pub fn close_traj(
    state: &mut State,
    i: usize,
    scene: &mut graphics::Scene,
    redraw: &mut crate::util::RedrawFlags,
    updates: &mut graphics::EngineUpdates,
) {
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

    let trajectory = state.trajectories.remove(i);
    let showing_frames = match &trajectory.source {
        TrajectorySource::Memory(frames) => frames
            .first()
            .zip(state.volatile.md_local.viewer.snapshots.first())
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b)),
        TrajectorySource::File(_) => false,
    };
    if trajectory.ui_active || showing_frames {
        let md = &mut state.volatile.md_local;
        md.clear_snaps(&mut scene.entities, redraw);
        if !md.running() {
            md.mol_dynamics = None;
        }
        updates.entities = graphics::EntityUpdate::All;
    }
}
