//! Download MDposit trajectories for disk-backed playback.

use std::{
    collections::HashMap,
    fs::{self, File},
    io,
    path::Path,
    sync::mpsc::Receiver,
};

use bio_apis::mdposit::{Client, Params, TrajectoryFormat};
use bio_files::{MmCif, bond_inference};
use egui::Pos2;
use mol_defs::molecules::{Atom, Bond, MolType, PeptideIdent, common::MoleculeCommon};
use serde::Deserialize;

use super::{
    trajectory::{Trajectory, TrajectorySource},
    viewer::{ViewerMolSet, ViewerMolecule},
};
use crate::{state::State, ui::util::is_pdb_id};

const CACHE_DIR: &str = "mdposit";
const STRUCTURE: &str = "structure.cif";
const PAGE_SIZE: usize = 30;

/// The default step keeps downloads at or below this many frames.
const DEFAULT_MAX_FRAMES: usize = 100;

/// Quick Traj defaults; see `MdPrefs`. MDposit exports at roughly 0.2 to 0.6 MB/s, so this keeps
/// typical downloads to around 10 seconds.
pub const QUICK_TARGET_MB_DEFAULT: u32 = 4;
/// Mid-scale, which favors smooth motion; see [`FramePlan::new`].
pub const QUICK_SPAN_DEFAULT: u8 = 50;
/// Higher values bias the span scale toward smaller steps, and a shorter time covered.
const QUICK_SPAN_CURVE: f64 = 2.5;
/// Quick Traj keeps at least this many frames, even past its target size, so there's motion to see.
const QUICK_MIN_FRAMES: usize = 20;
/// Quick Traj ranks the first this many matching projects.
const QUICK_MAX_PROJECTS: usize = 100;
const QUICK_FIELDS: &str = "accession,metadata.NAME,metadata.PDBIDS,metadata.PROTATS,\
    metadata.PROTRES,mds.name,mds.atoms,mds.frames,mds.time";
/// In order of preference.
const QUICK_FORMATS: [TrajectoryFormat; 3] = [
    TrajectoryFormat::Xtc,
    TrajectoryFormat::Trr,
    TrajectoryFormat::Dcd,
];
/// Used for size estimates when MDposit doesn't list an atom count.
const UNKNOWN_ATOM_COUNT: usize = 20_000;

// Download size estimates, measured from MDposit exports. XTC compresses each coordinate to about
// 1.3 bytes; TRR and DCD store them as f32. The mmCIF structure is a fixed cost, and is large
// relative to the frames.
const XTC_BYTES_PER_ATOM: f64 = 3.9;
const RAW_BYTES_PER_ATOM: f64 = 12.;
const FRAME_HEADER_BYTES: f64 = 100.;
const MMCIF_BYTES_PER_ATOM: f64 = 82.;

#[derive(Deserialize)]
pub(crate) struct Project {
    pub accession: String,
    metadata: Option<ProjectMetadata>,
    mds: Option<Vec<Replica>>,
}

#[derive(Deserialize)]
struct ProjectMetadata {
    #[serde(rename = "NAME")]
    name: Option<String>,
    // The fields below are only requested by Quick Traj.
    #[serde(rename = "PDBIDS")]
    pdb_ids: Option<Vec<String>>,
    #[serde(rename = "PROTATS")]
    protein_atoms: Option<usize>,
    #[serde(rename = "PROTRES")]
    protein_residues: Option<usize>,
}

#[derive(Deserialize)]
pub(crate) struct Replica {
    name: Option<String>,
    pub frames: Option<usize>,
    // The fields below are only requested by Quick Traj.
    atoms: Option<usize>,
    /// Simulated time, in ns.
    time: Option<f64>,
}

impl Project {
    pub fn label(&self) -> String {
        match self
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.name.as_deref())
        {
            Some(name) => format!("{} · {name}", self.accession),
            None => self.accession.clone(),
        }
    }

    pub fn replicas(&self) -> &[Replica] {
        self.mds.as_deref().unwrap_or_default()
    }
}

impl Replica {
    pub fn label(&self, index: usize) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("Replica {}", index + 1))
    }
}

pub(crate) struct Results {
    pub pdb_id: String,
    pub page: usize,
    pub pages: usize,
    pub projects: Vec<Project>,
}

pub(crate) enum JobResult {
    Search(Results),
    Download(Trajectory, ViewerMolSet),
}

#[derive(Default)]
pub struct Import {
    pub(crate) anchor: Option<Pos2>,
    pub(crate) results: Option<Results>,
    pub(crate) project: usize,
    pub(crate) replica: usize,
    pub(crate) first: usize,
    pub(crate) last: usize,
    pub(crate) step: usize,
    pub(crate) format: Option<TrajectoryFormat>,
    pub(crate) protein_only: bool,
    pub(crate) error: Option<String>,
    pub(crate) worker: Option<(&'static str, Receiver<io::Result<JobResult>>)>,
    /// A Quick Traj download, for this PDB ID. Independent of the window.
    pub(crate) quick: Option<(String, Receiver<io::Result<Option<QuickTraj>>>)>,
}

/// The smallest step that loads no more than `DEFAULT_MAX_FRAMES` of `frames`.
pub(crate) fn default_step(frames: usize) -> usize {
    frames.div_ceil(DEFAULT_MAX_FRAMES).max(1)
}

pub fn pdb_id(idents: &[PeptideIdent]) -> Option<String> {
    let pdbe = idents.iter().find_map(|ident| match ident {
        PeptideIdent::Pdbe(id) => normalize_pdb_id(id),
        _ => None,
    });

    pdbe.or_else(|| {
        idents.iter().find_map(|ident| match ident {
            PeptideIdent::Rcsb(id) => normalize_pdb_id(id),
            _ => None,
        })
    })
}

fn normalize_pdb_id(id: &str) -> Option<String> {
    let id = id.trim().to_ascii_lowercase();
    if !is_pdb_id(&id) {
        return None;
    }

    // MDposit indexes existing four-character accessions in uppercase.
    let id = id
        .strip_prefix("pdb_0000")
        .filter(|short| is_pdb_id(short))
        .unwrap_or(&id);

    Some(id.to_ascii_uppercase())
}

fn pdb_filter(pdb_id: &str) -> String {
    if pdb_id.starts_with("PDB_") {
        // Extended IDs without a short form are published with a lowercase prefix.
        format!("pdbId=eq='{pdb_id}','{}'", pdb_id.to_ascii_lowercase())
    } else {
        format!("pdbId=eq='{pdb_id}'")
    }
}

pub(crate) fn search(client: &Client, pdb_id: String, page: usize) -> io::Result<Results> {
    let filter = pdb_filter(&pdb_id);
    let page_str = page.to_string();
    let limit = PAGE_SIZE.to_string();
    let result = client.projects(Params::Query(&[
        ("filter", &filter),
        ("return", "accession,metadata.NAME,mds"),
        ("limit", &limit),
        ("page", &page_str),
    ]))?;
    let projects = result
        .projects
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;

    Ok(Results {
        pdb_id,
        page,
        pages: (result.filtered_count as usize).div_ceil(PAGE_SIZE),
        projects,
    })
}

pub(crate) struct Download {
    pub accession: String,
    pub replica: usize,
    pub first: usize,
    pub last: usize,
    pub step: usize,
    pub format: TrajectoryFormat,
    pub protein_only: bool,
}

impl Download {
    pub fn save(self, prefs_dir: &Path) -> io::Result<(Trajectory, ViewerMolSet)> {
        let code = self.accession.strip_prefix("MD-").unwrap_or_default();
        if code.is_empty() || !code.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(io::Error::other("Invalid MDposit accession"));
        }

        let root = prefs_dir.join(CACHE_DIR);
        fs::create_dir_all(&root)?;
        let dir = root.join(format!("{:016x}", rand::random::<u64>()));
        fs::create_dir(&dir)?;

        let project = format!("{}.{}", self.accession, self.replica);
        let name = format!(
            "{project}_{}-{}_step{}.{}",
            self.first,
            self.last,
            self.step,
            self.format.as_str(),
        );
        let path = dir.join(name);
        let structure = dir.join(STRUCTURE);
        let frames = format!("{}:{}:{}", self.first, self.last, self.step);
        let client = Client::new();
        let mut structure_query = vec![("format", "mmcif")];
        let mut trajectory_query = vec![("frames", frames.as_str())];
        if self.protein_only {
            structure_query.push(("selection", "protein"));
            trajectory_query.push(("selection", "protein"));
        }

        let result = (|| {
            client.download_structure(
                &project,
                Params::Query(&structure_query),
                &mut File::create(&structure)?,
            )?;
            client.download_trajectory(
                &project,
                self.format,
                Params::Query(&trajectory_query),
                &mut File::create(&path)?,
            )?;
            read_download(&path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(&structure);
            let _ = fs::remove_dir(&dir);
        }

        result
    }
}

/// Frames to download: one-based, inclusive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FramePlan {
    pub first: usize,
    pub last: usize,
    pub step: usize,
}

impl FramePlan {
    /// Fits frames of a `total`-frame trajectory with `atoms` atoms into about `target_mb`, structure
    /// included, keeping at least `QUICK_MIN_FRAMES`. `span` (0 to 100) sets how those frames are
    /// spread: 0 takes consecutive frames from the start; 100 uses the largest step that still fits
    /// them, covering the whole run. Between, the step is the max step raised to
    /// (span / 100)^`QUICK_SPAN_CURVE`: geometric, since useful steps range from 1 to thousands,
    /// and curved, so most of the scale gives fine control over smooth playback. E.g. at 50, a max
    /// step of 36 gives 2, and 1,000 gives 3.
    pub fn new(
        total: usize,
        atoms: usize,
        format: TrajectoryFormat,
        target_mb: u32,
        span: u8,
    ) -> Self {
        let per_atom = match format {
            TrajectoryFormat::Xtc => XTC_BYTES_PER_ATOM,
            _ => RAW_BYTES_PER_ATOM,
        };
        let atoms = atoms as f64;
        let for_frames = f64::from(target_mb) * 1e6 - atoms * MMCIF_BYTES_PER_ATOM;
        let fit = (for_frames.max(0.) / (atoms * per_atom + FRAME_HEADER_BYTES)) as usize;
        let n = fit.max(QUICK_MIN_FRAMES).min(total).max(1);

        let step_max = total / n;
        let t = (f64::from(span.min(100)) / 100.).powf(QUICK_SPAN_CURVE);
        let step = ((step_max as f64).powf(t).round() as usize).clamp(1, step_max.max(1));

        Self {
            first: 1,
            last: (1 + (n - 1) * step).min(total.max(1)),
            step,
        }
    }

    pub fn count(&self) -> usize {
        (self.last - self.first) / self.step + 1
    }
}

/// A replica Quick Traj chose.
struct QuickPick<'a> {
    project: &'a Project,
    /// Zero-based.
    replica: usize,
    frames: usize,
    /// Atoms the download will have.
    atoms: usize,
    protein_only: bool,
}

/// How closely MDposit's protein residue count matches the open structure's: 0 is closest. Simulations
/// often add missing loops or termini, so this allows some difference.
fn residue_tier(sim: Option<usize>, open: usize) -> u8 {
    let Some(sim) = sim.filter(|&n| n > 0 && open > 0) else {
        return 2;
    };

    match sim.abs_diff(open) as f64 / open as f64 {
        d if d <= 0.1 => 0,
        d if d <= 0.25 => 1,
        d if d <= 0.5 => 2,
        _ => 3,
    }
}

/// The best replica for `pdb_id`, preferring, in order: simulations of that entry alone, rather than
/// together with other entries; protein residue counts near `residues`, the open structure's;
/// longer simulated time, for broader sampling; and fewer atoms, for a smaller download.
fn best_match<'a>(projects: &'a [Project], pdb_id: &str, residues: usize) -> Option<QuickPick<'a>> {
    projects
        .iter()
        .flat_map(|project| {
            let meta = project.metadata.as_ref();
            let with_others = !meta
                .and_then(|m| m.pdb_ids.as_deref())
                .is_some_and(|ids| ids.len() == 1 && ids[0].eq_ignore_ascii_case(pdb_id));
            let tier = residue_tier(meta.and_then(|m| m.protein_residues), residues);
            // Protein-only downloads keep size down, e.g. for membrane systems. Use all atoms if
            // the system has no protein, or doesn't say.
            let protein_atoms = meta.and_then(|m| m.protein_atoms).filter(|&n| n > 0);

            project
                .replicas()
                .iter()
                .enumerate()
                .filter_map(move |(i, replica)| {
                    let frames = replica.frames.filter(|&n| n > 0)?;
                    let atoms = protein_atoms
                        .or(replica.atoms)
                        .filter(|&n| n > 0)
                        .unwrap_or(UNKNOWN_ATOM_COUNT);
                    let time = replica.time.filter(|t| t.is_finite()).unwrap_or(0.);
                    let pick = QuickPick {
                        project,
                        replica: i,
                        frames,
                        atoms,
                        protein_only: protein_atoms.is_some(),
                    };

                    Some(((with_others, tier, time), pick))
                })
        })
        // On ties, the first in MDposit's order.
        .min_by(|(a, pick_a), (b, pick_b)| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(b.2.total_cmp(&a.2))
                .then(pick_a.atoms.cmp(&pick_b.atoms))
        })
        .map(|(_, pick)| pick)
}

/// A Quick Traj download, ready to view.
pub(crate) struct QuickTraj {
    pub trajectory: Trajectory,
    pub set: ViewerMolSet,
    pub summary: String,
}

/// Find the MDposit simulation that best matches `pdb_id`, and download frames from it sized by
/// `target_mb` and spread by `span`; see [`FramePlan`]. Uses XTC, then TRR, then DCD, if an export
/// fails. `residues` is the open structure's amino acid count. `None` if MDposit has no simulation.
pub(crate) fn quick_download(
    pdb_id: &str,
    residues: usize,
    target_mb: u32,
    span: u8,
    prefs_dir: &Path,
) -> io::Result<Option<QuickTraj>> {
    let filter = pdb_filter(pdb_id);
    let limit = QUICK_MAX_PROJECTS.to_string();
    let page = Client::new().projects(Params::Query(&[
        ("filter", &filter),
        ("return", QUICK_FIELDS),
        ("limit", &limit),
    ]))?;
    // Skip records that don't parse, rather than failing the whole search.
    let projects: Vec<Project> = page
        .projects
        .into_iter()
        .filter_map(|p| serde_json::from_value(p).ok())
        .collect();

    let Some(pick) = best_match(&projects, pdb_id, residues) else {
        return Ok(None);
    };

    let mut first_err = None;
    for format in QUICK_FORMATS {
        let plan = FramePlan::new(pick.frames, pick.atoms, format, target_mb, span);
        let download = Download {
            accession: pick.project.accession.clone(),
            replica: pick.replica + 1,
            first: plan.first,
            last: plan.last,
            step: plan.step,
            format,
            protein_only: pick.protein_only,
        };

        match download.save(prefs_dir) {
            Ok((trajectory, set)) => {
                let summary = quick_summary(&pick, &plan, format, &trajectory);
                return Ok(Some(QuickTraj {
                    trajectory,
                    set,
                    summary,
                }));
            }
            Err(e) => {
                eprintln!("MDposit {} export failed: {e}", format.as_str());
                first_err.get_or_insert(e);
            }
        }
    }

    Err(first_err.unwrap_or_else(|| io::Error::other("No MDposit trajectory format to try")))
}

/// E.g. "MD-A001NL · Gromacs 60 ns MD of SARS-CoV-2…, replica 1: 26 of 723 frames (1-301, step 12),
/// protein atoms, XTC, 9.6 MB."
fn quick_summary(
    pick: &QuickPick,
    plan: &FramePlan,
    format: TrajectoryFormat,
    trajectory: &Trajectory,
) -> String {
    const MAX_LABEL: usize = 80;

    let mut label = pick.project.label();
    if label.chars().count() > MAX_LABEL {
        label = label.chars().take(MAX_LABEL).collect::<String>() + "…";
    }
    let replica = pick.project.replicas()[pick.replica].label(pick.replica);
    let atoms = if pick.protein_only {
        "protein atoms"
    } else {
        "all atoms"
    };

    let mut result = format!(
        "{label}, {replica}: {} of {} frames ({}-{}, step {}), {atoms}, {}",
        plan.count(),
        pick.frames,
        plan.first,
        plan.last,
        plan.step,
        format.as_str().to_uppercase(),
    );

    // The structure and trajectory are the download.
    if let TrajectorySource::File(path) = &trajectory.source {
        let size = |p: &Path| fs::metadata(p).map_or(0, |m| m.len());
        let bytes = size(path) + size(&path.with_file_name(STRUCTURE));
        result += &format!(", {}", super::trajectory::format_bytes(bytes as f64));
    }
    result.push('.');

    result
}

fn read_download(path: &Path) -> io::Result<(Trajectory, ViewerMolSet)> {
    let mut trajectory = Trajectory::new(path)?;
    let structure_path = path.with_file_name(STRUCTURE);
    let cif = MmCif::new(&fs::read_to_string(&structure_path)?)?;
    if trajectory.num_frames == 0 || cif.atoms.is_empty() {
        return Err(io::Error::other("MDposit returned an empty simulation"));
    }
    if cif.atoms.len() != trajectory.num_atoms {
        return Err(io::Error::other(format!(
            "MDposit structure has {} atoms, but the trajectory has {}",
            cif.atoms.len(),
            trajectory.num_atoms,
        )));
    }

    // Keep the atom order of the downloaded structure for trajectory playback.
    let atoms: Vec<_> = cif.atoms.iter().map(Atom::from).collect();
    let index: HashMap<_, _> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial_number, i))
        .collect();
    let bonds = bond_inference::create_bonds(&cif.atoms)
        .into_iter()
        .map(|b| Bond {
            bond_type: b.bond_type,
            atom_0_sn: b.atom_0_sn,
            atom_1_sn: b.atom_1_sn,
            atom_0: index[&b.atom_0_sn],
            atom_1: index[&b.atom_1_sn],
            is_backbone: false,
        })
        .collect();
    let name = trajectory.display_name.clone();
    let mol = MoleculeCommon::new(
        cif.ident,
        atoms,
        bonds,
        cif.metadata,
        Some(structure_path.clone()),
    );

    trajectory.run_mol_set_path = Some(structure_path.clone());
    let set = ViewerMolSet::new(
        Some(structure_path),
        name,
        vec![ViewerMolecule {
            mol_type: MolType::Peptide,
            range: (0, mol.atoms.len()),
            mol,
        }],
    );

    Ok((trajectory, set))
}

/// Restore an MDposit trajectory together with its saved structure.
pub(crate) fn open_trajectory(
    path: &Path,
    prefs_dir: &Path,
) -> io::Result<(Trajectory, Option<ViewerMolSet>)> {
    if path.starts_with(prefs_dir.join(CACHE_DIR)) {
        return read_download(path).map(|(traj, set)| (traj, Some(set)));
    }

    Ok((Trajectory::new(path)?, None))
}

pub(crate) fn register(state: &mut State, mut trajectory: Trajectory, set: Option<ViewerMolSet>) {
    // Default to a step that fits the load limits, so the full range can load.
    trajectory.ui_step = trajectory
        .auto_load(state.to_save.md.max_traj_auto_load_gb)
        .step;

    let Some(set) = set else {
        state.trajectories.push(trajectory);
        return;
    };

    let viewer = &mut state.volatile.md_local.viewer;
    if !viewer
        .mol_sets
        .iter()
        .any(|current| current.path == set.path)
    {
        viewer.mol_sets.push(set);
    }
    if !state.trajectories.iter().any(|current| {
        matches!((&current.source, &trajectory.source), (TrajectorySource::File(a), TrajectorySource::File(b)) if a == b)
    }) {
        state.trajectories.push(trajectory);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Estimated download bytes for a plan.
    fn est_bytes(plan: &FramePlan, atoms: usize) -> f64 {
        let atoms = atoms as f64;
        atoms * MMCIF_BYTES_PER_ATOM
            + plan.count() as f64 * (atoms * XTC_BYTES_PER_ATOM + FRAME_HEADER_BYTES)
    }

    #[test]
    fn frame_plan_stays_in_bounds() {
        for total in [1, 2, 7, 19, 20, 21, 100, 723, 1_668, 10_000, 1_000_000] {
            for atoms in [0, 500, 1_231, 14_000, 52_329, 500_000] {
                for span in [0, 1, 50, 75, 99, 100] {
                    for target in [1, 10, 100] {
                        let p = FramePlan::new(total, atoms, TrajectoryFormat::Xtc, target, span);
                        assert_eq!(p.first, 1);
                        assert!(p.step >= 1);
                        assert!(p.last >= p.first && p.last <= total, "{p:?} of {total}");
                        assert!(p.count() >= QUICK_MIN_FRAMES.min(total), "{p:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn frame_plan_fits_target_and_span() {
        // A small protein with many frames: the target sets the frame count.
        let atoms = 1_231;
        let total = 1_000_000;
        let wide = FramePlan::new(total, atoms, TrajectoryFormat::Xtc, 10, 100);
        let narrow = FramePlan::new(total, atoms, TrajectoryFormat::Xtc, 10, 0);
        assert!(est_bytes(&wide, atoms) <= 10e6);
        assert!(est_bytes(&wide, atoms) > 9e6);
        assert_eq!(wide.count(), narrow.count());
        // Span 0 is consecutive; span 100 covers nearly the whole run.
        assert_eq!(narrow.step, 1);
        // The integer step can leave a sliver at the end.
        assert!(wide.last as f64 > 0.99 * total as f64, "{wide:?}");
        let mid = FramePlan::new(total, atoms, TrajectoryFormat::Xtc, 10, 50);
        assert!(mid.step > narrow.step && mid.step < wide.step);

        // A large protein gets fewer frames for the same target.
        let large = FramePlan::new(723, 52_329, TrajectoryFormat::Xtc, 10, 100);
        assert!(large.count() < 40 && large.count() >= QUICK_MIN_FRAMES);
        assert!(est_bytes(&large, 52_329) <= 10e6);

        // Raw formats are larger per frame.
        let trr = FramePlan::new(total, atoms, TrajectoryFormat::Trr, 10, 100);
        assert!(trr.count() < wide.count());

        // Short runs load whole.
        let short = FramePlan::new(12, atoms, TrajectoryFormat::Xtc, 10, 100);
        assert_eq!((short.first, short.last, short.step), (1, 12, 1));
    }

    fn projects(json: serde_json::Value) -> Vec<Project> {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn best_match_prefers_this_entry_then_residues_then_time() {
        let projects = projects(serde_json::json!([
            // Another entry too.
            {"accession": "MD-A", "metadata": {"PDBIDS": ["6VXX", "6VSB"], "PROTATS": 53000, "PROTRES": 3300},
             "mds": [{"name": "replica 1", "atoms": 64758, "frames": 1000, "time": 1000.0}]},
            // A partial construct, long run.
            {"accession": "MD-B", "metadata": {"PDBIDS": ["6VXX"], "PROTATS": 14346, "PROTRES": 900},
             "mds": [{"name": "replica 1", "atoms": 14512, "frames": 1668, "time": 2001.6}]},
            // The full structure.
            {"accession": "MD-C", "metadata": {"PDBIDS": ["6VXX"], "PROTATS": 52329, "PROTRES": 3250},
             "mds": [{"name": "replica 1", "atoms": 52329, "frames": 723, "time": 57.84},
                     {"name": "replica 2", "atoms": 52329, "frames": 0, "time": 99.0}]},
            {"accession": "MD-D", "metadata": {"PDBIDS": ["6vxx"], "PROTATS": 52329, "PROTRES": 3250},
             "mds": [{"name": "replica 1", "atoms": 52329, "frames": 900, "time": 90.0},
                     {"name": "replica 2", "atoms": 52329, "frames": 900, "time": 120.0}]},
        ]));

        let pick = best_match(&projects, "6VXX", 3200).unwrap();
        assert_eq!(pick.project.accession, "MD-D");
        assert_eq!(pick.replica, 1);
        assert_eq!(pick.atoms, 52329);
        assert!(pick.protein_only);

        // Without residue info, the longest run of this entry alone wins.
        let pick = best_match(&projects, "6VXX", 0).unwrap();
        assert_eq!(pick.project.accession, "MD-B");

        assert!(best_match(&projects[..1], "1ABC", 0).is_some());
        assert!(best_match(&[], "6VXX", 0).is_none());
    }

    #[test]
    fn best_match_without_protein_uses_all_atoms() {
        let projects = projects(serde_json::json!([
            {"accession": "MD-E", "metadata": {"PDBIDS": ["1BNA"], "PROTATS": 0},
             "mds": [{"name": "replica 1", "atoms": 800, "frames": 100}]},
        ]));
        let pick = best_match(&projects, "1BNA", 0).unwrap();
        assert_eq!(pick.atoms, 800);
        assert!(!pick.protein_only);
    }
}
