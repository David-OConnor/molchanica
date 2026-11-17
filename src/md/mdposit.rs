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
}

#[derive(Deserialize)]
pub(crate) struct Replica {
    name: Option<String>,
    pub frames: Option<usize>,
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

pub(crate) fn search(client: &Client, pdb_id: String, page: usize) -> io::Result<Results> {
    let filter = if pdb_id.starts_with("PDB_") {
        // Extended IDs without a short form are published with a lowercase prefix.
        format!("pdbId=eq='{pdb_id}','{}'", pdb_id.to_ascii_lowercase())
    } else {
        format!("pdbId=eq='{pdb_id}'")
    };
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
    pub fn save(self, prefs_dir: &Path) -> io::Result<JobResult> {
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
            let (trajectory, set) = read_download(&path)?;

            Ok(JobResult::Download(trajectory, set))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(&structure);
            let _ = fs::remove_dir(&dir);
        }

        result
    }
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
