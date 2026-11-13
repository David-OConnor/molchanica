use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use bio_files::{
    AtomGeneric, FrameSlice,
    gromacs::{GromacsTopology, MdpParams, OutputEnergy, gro::Gro},
    xyz::Xyz,
};
use dynamics::{ForceFieldFamily, Integrator, MdConfig, snapshot::Snapshot};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::{Atom, Bond, MolType, common::MoleculeCommon};

use super::{
    MdBackend,
    trajectory::{MAX_FRAMES_TO_ATTEMPT_LOADING, Trajectory},
    viewer::{self, SnapshotViewer, ViewerMolSet},
};
use crate::{
    cam::{VIEW_DIR_FRONT, reset_camera},
    project::{MD_INPUTS_FILE, MdInputs},
    state::State,
    util::{handle_err, handle_success},
};

pub const OUTPUT_DIR: &str = "md_out";
const TRAJECTORIES: &[&str] = &["trr", "xtc", "dcd"];
const FRAME_LIMIT_ATOMS: usize = 5_000_000;

pub struct RecentRun {
    pub directory: PathBuf,
    pub legacy: Option<String>,
    pub name: String,
    modified: SystemTime,
}

#[derive(Clone)]
pub enum RunInputs {
    Gromacs(Box<crate::gromacs::Setup>),
    Xtb {
        geometry: Vec<AtomGeneric>,
        mols: Vec<(dynamics::FfMolType, MoleculeCommon, usize)>,
    },
}

impl RunInputs {
    pub fn backend(&self) -> MdBackend {
        match self {
            Self::Gromacs(_) => MdBackend::Gromacs,
            Self::Xtb { .. } => MdBackend::xTb,
        }
    }
}

pub fn running(state: &State) -> bool {
    let md = &state.volatile.md_local;
    md.running
        || md.launching
        || state.volatile.thread_receivers.xtb_md.is_some()
        || state.volatile.thread_receivers.gromacs_md_avail.is_some()
}

fn files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .collect();
    files.sort();

    files
}

fn named(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

fn extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case(ext))
}

fn production_file(path: &Path) -> bool {
    let name = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    !matches!(
        name.as_str(),
        "em" | "ions" | "minim" | "minimize" | "energy_minimization"
    ) && !name.starts_with("em_")
        && !name.starts_with("ions_")
}

fn trajectory_files(dir: &Path) -> Vec<PathBuf> {
    let files = files(dir);
    let mut result = Vec::new();
    for ext in TRAJECTORIES {
        let path = dir.join(format!("traj.{ext}"));
        if path.is_file() {
            result.push(path);
        }
    }
    for path in files {
        if path.is_file()
            && TRAJECTORIES.iter().any(|ext| extension(&path, ext))
            && production_file(&path)
            && !result.contains(&path)
        {
            result.push(path);
        }
    }

    result
}

fn run_entry(dir: PathBuf) -> Option<RecentRun> {
    let path = named(&dir, &["xtb.trj"])
        .or_else(|| trajectory_files(&dir).into_iter().next())
        .or_else(|| named(&dir, &["confout.gro"]))?;
    let modified = path.metadata().ok()?.modified().ok()?;
    let name = dir.file_name()?.to_string_lossy().into_owned();

    Some(RecentRun {
        directory: dir,
        legacy: None,
        name,
        modified,
    })
}

pub fn recent(opened: &[PathBuf]) -> Vec<RecentRun> {
    let root = Path::new(OUTPUT_DIR);
    let entries = files(root);
    let mut result: Vec<_> = entries
        .iter()
        .filter(|path| path.is_dir())
        .cloned()
        .filter_map(run_entry)
        .collect();

    for path in &entries {
        let Some(index) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.strip_prefix("traj_"))
            .filter(|index| index.parse::<usize>().is_ok())
        else {
            continue;
        };
        if !TRAJECTORIES.iter().any(|ext| extension(path, ext))
            || result
                .iter()
                .any(|run| run.legacy.as_deref() == Some(index))
        {
            continue;
        }
        let name = if root.join(format!("traj_{index}.gro")).is_file() {
            format!("Dynamics {index}")
        } else if root.join(format!("mols_in_{index}.gro")).is_file()
            && !result
                .iter()
                .any(|run| run.directory == root.join(format!("gromacs_{index}")))
        {
            format!("GROMACS {index}")
        } else {
            continue;
        };
        let modified = path
            .metadata()
            .and_then(|md| md.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        result.push(RecentRun {
            directory: root.to_owned(),
            legacy: Some(index.into()),
            name,
            modified,
        });
    }

    result.sort_by(|a, b| b.modified.cmp(&a.modified).then(a.name.cmp(&b.name)));
    for dir in opened.iter().rev() {
        if !result
            .iter()
            .any(|run| run.directory.canonicalize().ok().as_ref() == Some(dir))
            && let Some(run) = run_entry(dir.clone())
        {
            result.insert(0, run);
        }
    }

    result
}

struct LoadedRun {
    backend: MdBackend,
    trajectory: Trajectory,
    snapshots: Vec<Arc<Snapshot>>,
    mol_set: ViewerMolSet,
    config: Option<(MdConfig, u32, f32)>,
    inputs: Option<RunInputs>,
    native_inputs: Option<MdInputs>,
    xtb_options: Option<(Option<f64>, Option<String>)>,
}

pub fn open(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    dir: &Path,
    legacy: Option<&str>,
) {
    if running(state) {
        handle_err(
            &mut state.ui,
            "Wait for the active MD run to finish.".into(),
        );

        return;
    }

    let result = dir
        .canonicalize()
        .and_then(|dir| load(&dir, legacy)?.install(state, scene, updates, &dir));
    match result {
        Ok(message) => {
            if legacy.is_none()
                && let Ok(dir) = dir.canonicalize()
                && !Path::new(OUTPUT_DIR)
                    .canonicalize()
                    .is_ok_and(|root| dir.parent() == Some(&root))
            {
                let recent = &mut state.to_save.recent_md_runs;
                recent.retain(|path| path != &dir);
                recent.insert(0, dir);
                recent.truncate(30);
                state.to_save.save_flag = true;
            }
            handle_success(&mut state.ui, message);
        }
        Err(error) => handle_err(&mut state.ui, format!("Could not open MD run: {error}")),
    }
}

fn load(dir: &Path, legacy: Option<&str>) -> io::Result<LoadedRun> {
    if legacy.is_none() && dir == Path::new(OUTPUT_DIR).canonicalize().unwrap_or_default() {
        return Err(io::Error::other(
            "Choose an individual run folder or pick a run from Recent MD runs.",
        ));
    }
    if legacy.is_none() && dir.join("xtb.trj").is_file() {
        return load_xtb(dir);
    }

    let (paths, structures, backend) = if let Some(index) = legacy {
        let dynamics_gro = dir.join(format!("traj_{index}.gro"));
        let dynamics = dynamics_gro.is_file();
        let gro = if dynamics {
            dynamics_gro
        } else {
            dir.join(format!("mols_in_{index}.gro"))
        };
        let paths = TRAJECTORIES
            .iter()
            .map(|ext| dir.join(format!("traj_{index}.{ext}")))
            .filter(|path| path.is_file())
            .collect();
        (
            paths,
            vec![gro],
            if dynamics {
                MdBackend::Dynamics
            } else {
                MdBackend::Gromacs
            },
        )
    } else {
        let is_dynamics = dir.join(MD_INPUTS_FILE).is_file()
            || dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .starts_with("dynamics_");
        let preferred = [
            "em.gro",
            "ionized.gro",
            "mols_in_solvated.gro",
            "conf.gro",
            "confout.gro",
        ];
        let mut structures: Vec<_> = preferred
            .iter()
            .map(|name| dir.join(name))
            .filter(|p| p.is_file())
            .collect();
        for path in files(dir) {
            if extension(&path, "gro") && production_file(&path) && !structures.contains(&path) {
                structures.push(path);
            }
        }
        (
            trajectory_files(dir),
            structures,
            if is_dynamics {
                MdBackend::Dynamics
            } else {
                MdBackend::Gromacs
            },
        )
    };

    let mut error = io::Error::other("No MD trajectory found. Select an individual run folder.");
    for path in paths {
        match load_binary_run(dir, &path, &structures, backend, legacy.is_some()) {
            Ok(run) => return Ok(run),
            Err(e) => error = e,
        }
    }
    if backend == MdBackend::Gromacs
        && legacy.is_none()
        && let Some(path) = named(dir, &["confout.gro"])
    {
        return load_gromacs_structure(dir, &path);
    }

    Err(error)
}

fn load_binary_run(
    dir: &Path,
    path: &Path,
    structures: &[PathBuf],
    backend: MdBackend,
    legacy: bool,
) -> io::Result<LoadedRun> {
    let mut trajectory = Trajectory::new(path)?;
    if trajectory.num_atoms == 0 || trajectory.num_frames == 0 {
        return Err(io::Error::other(
            "The trajectory contains no coordinate frames.",
        ));
    }
    let n_frames =
        (FRAME_LIMIT_ATOMS / trajectory.num_atoms).clamp(1, MAX_FRAMES_TO_ATTEMPT_LOADING);
    let slice = FrameSlice::Index {
        start: None,
        end: (trajectory.num_frames > n_frames).then_some(n_frames - 1),
    };
    if !legacy
        && let Some(path) = named(dir, &["energy_out.xvg", "energy.xvg"])
        && let Ok(mut energies) = OutputEnergy::from_xvg(&path)
    {
        energies.retain(|en| en.time.is_finite());
        energies.sort_by(|a, b| a.time.total_cmp(&b.time));
        trajectory.energies = energies;
    }
    trajectory.run_mol_set_path = Some(path.to_owned());
    let snapshots = trajectory.load_snaps(slice)?;
    if snapshots.is_empty()
        || snapshots
            .iter()
            .any(|snap| snap.atom_posits.len() != trajectory.num_atoms)
    {
        return Err(io::Error::other(
            "The trajectory has missing or inconsistent coordinates.",
        ));
    }

    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let named_top = dir.join(format!("{stem}.top"));
    let topology = named_top
        .is_file()
        .then_some(named_top)
        .or_else(|| named(dir, &["topo.top", "topol.top"]))
        .or_else(|| files(dir).into_iter().find(|path| extension(path, "top")));
    let coordinates = structures.iter().find_map(|path| {
        let gro = Gro::load(path).ok()?;
        (gro.atoms.len() == trajectory.num_atoms).then_some((path, gro))
    });

    let mut native_inputs = if !legacy && backend == MdBackend::Dynamics {
        named(dir, &[MD_INPUTS_FILE]).and_then(|path| match MdInputs::load(&path) {
            Ok(saved) => Some(saved),
            Err(error) => {
                eprintln!("Could not restore {}: {error}", path.display());

                None
            }
        })
    } else {
        None
    };
    if native_inputs
        .as_ref()
        .is_some_and(|saved| saved.viewer.atom_count != trajectory.num_atoms)
    {
        native_inputs = None;
    }
    let saved_set = if !legacy && backend == MdBackend::Gromacs {
        crate::project::load_md_mol_set(&dir.join(crate::project::MD_MOLECULE_SET_FILE))
            .ok()
            .filter(|set| set.atom_count == trajectory.num_atoms)
    } else {
        None
    };
    let mut mol_set = if let Some(set) = saved_set {
        set
    } else if let Some(saved) = native_inputs.as_mut() {
        std::mem::replace(
            &mut saved.viewer,
            ViewerMolSet::new(None, String::new(), Vec::new()),
        )
    } else {
        let (path, mut gro) = coordinates
            .as_ref()
            .map(|(path, gro)| (*path, gro.clone()))
            .ok_or_else(|| {
                io::Error::other("No GRO structure matches the trajectory atom count.")
            })?;
        let top = topology
            .as_deref()
            .and_then(|path| GromacsTopology::load(path).ok());
        if let Some(top) = &top {
            group_gromacs_atoms(&mut gro, top);
        }
        let mut viewer = SnapshotViewer::default();
        viewer.load_gro_data(path, gro, &[])?;
        let mut set = viewer.mol_sets.pop().unwrap();
        if let Some(top) = top {
            apply_topology_bonds(&mut set, &top);
        }

        set
    };

    let name = run_name(dir, path, legacy);
    mol_set.name = name.clone();
    mol_set.path = Some(path.to_owned());
    trajectory.display_name = name;

    let mdp_path = (!legacy)
        .then(|| {
            let named_mdp = dir.join(format!("{stem}.mdp"));
            named_mdp
                .is_file()
                .then_some(named_mdp)
                .or_else(|| named(dir, &["md.mdp", "mdout.mdp"]))
                .or_else(|| {
                    files(dir)
                        .into_iter()
                        .find(|path| extension(path, "mdp") && production_file(path))
                })
        })
        .flatten();
    let mdp_text = mdp_path
        .as_deref()
        .and_then(|path| fs::read_to_string(path).ok());
    let config = (backend != MdBackend::Gromacs)
        .then_some(mdp_text.as_deref())
        .flatten()
        .and_then(|text| MdpParams::from_mdp_str(text).ok())
        .and_then(|mdp| {
            let dt = mdp.dt;
            let steps = u32::try_from(mdp.nsteps).ok()?;
            if !dt.is_finite() || dt <= 0. {
                return None;
            }
            let cfg = native_inputs
                .as_ref()
                .map(|saved| saved.config.clone())
                .unwrap_or_else(|| gromacs_config(&mdp));

            Some((cfg, steps, dt))
        });
    let inputs = match (backend, coordinates, topology, mdp_text) {
        (MdBackend::Gromacs, Some((path, _)), Some(topology), Some(mdp)) => {
            Some(gromacs_inputs(dir, stem, path, topology, mdp_path, mdp))
        }
        _ => None,
    };

    Ok(LoadedRun {
        backend,
        trajectory,
        snapshots,
        mol_set,
        config,
        inputs,
        native_inputs,
        xtb_options: None,
    })
}

fn gromacs_inputs(
    dir: &Path,
    stem: &str,
    coordinates: &Path,
    topology: PathBuf,
    mdp_path: Option<PathBuf>,
    mdp: String,
) -> RunInputs {
    let index = named(dir, &[&format!("{stem}.ndx"), "index.ndx"])
        .or_else(|| files(dir).into_iter().find(|path| extension(path, "ndx")));
    let mut restart_files = files(&dir.join("continuation"))
        .into_iter()
        .filter(|path| path.is_file());
    let continuation = restart_files
        .next()
        .filter(|_| restart_files.next().is_none());
    let replay = dir.join("replay.mdp");
    let (mdp_path, mdp) = match fs::read_to_string(&replay) {
        Ok(text) => (Some(replay), text),
        Err(_) => (mdp_path, mdp),
    };

    RunInputs::Gromacs(Box::new(crate::gromacs::Setup {
        coordinates: Some(coordinates.to_owned()),
        topology: Some(topology),
        mdp: Some(mdp),
        mdp_path,
        index,
        reference: named(dir, &["restraint.gro"]),
        continuation,
        ..Default::default()
    }))
}

fn load_gromacs_structure(dir: &Path, path: &Path) -> io::Result<LoadedRun> {
    let mut gro = Gro::load(path)?;
    let top_path = named(dir, &["topo.top", "topol.top"]);
    let topology = top_path
        .as_deref()
        .and_then(|path| GromacsTopology::load(path).ok());
    if let Some(topology) = &topology {
        group_gromacs_atoms(&mut gro, topology);
    }
    let snapshot = Snapshot {
        atom_posits: gro
            .atoms
            .iter()
            .map(|atom| (atom.posit * 10.).into())
            .collect(),
        ..Default::default()
    };
    let mut mol_set =
        match crate::project::load_md_mol_set(&dir.join(crate::project::MD_MOLECULE_SET_FILE))
            .ok()
            .filter(|set| set.atom_count == gro.atoms.len())
        {
            Some(set) => set,
            None => {
                let mut viewer = SnapshotViewer::default();
                viewer.load_gro_data(path, gro, &[])?;
                let mut set = viewer.mol_sets.pop().unwrap();
                if let Some(topology) = &topology {
                    apply_topology_bonds(&mut set, topology);
                }

                set
            }
        };
    let name = run_name(dir, path, false);
    mol_set.name = name.clone();
    mol_set.path = Some(path.to_owned());
    let snapshots = vec![Arc::new(snapshot)];
    let mut trajectory = Trajectory::new_in_memory(snapshots.clone(), name, 0.);
    trajectory.ui_active = true;

    let inputs = top_path.and_then(|top| {
        let mdp = named(dir, &["md.mdp"])?;
        let text = fs::read_to_string(&mdp).ok()?;
        let coordinates = named(
            dir,
            &["em.gro", "ionized.gro", "mols_in_solvated.gro", "conf.gro"],
        )?;

        Some(gromacs_inputs(
            dir,
            "confout",
            &coordinates,
            top,
            Some(mdp),
            text,
        ))
    });

    Ok(LoadedRun {
        backend: MdBackend::Gromacs,
        trajectory,
        snapshots,
        mol_set,
        config: None,
        inputs,
        native_inputs: None,
        xtb_options: None,
    })
}

fn group_gromacs_atoms(gro: &mut Gro, top: &GromacsTopology) {
    let lengths: Option<Vec<_>> = top
        .molecules
        .iter()
        .map(|(name, count)| {
            let len = top.molecule_type(name)?.atoms.len();
            Some((name, *count, len))
        })
        .collect();
    let Some(lengths) = lengths else { return };
    let n_atoms = lengths.iter().try_fold(0usize, |total, (_, count, len)| {
        total.checked_add(count.checked_mul(*len)?)
    });
    if n_atoms != Some(gro.atoms.len()) || lengths.iter().any(|(_, _, len)| *len == 0) {
        return;
    }

    let mut atoms = gro.atoms.iter_mut();
    let mut mol_id = 0;
    for (name, count, len) in lengths {
        for _ in 0..count {
            mol_id += 1;
            for atom in atoms.by_ref().take(len) {
                atom.mol_id = mol_id;
                atom.mol_name.clone_from(name);
            }
        }
    }
}

fn apply_topology_bonds(set: &mut ViewerMolSet, top: &GromacsTopology) {
    for entry in &mut set.mols {
        if entry.mol_type == MolType::Water {
            continue;
        }
        let mol = &mut entry.mol;
        let Some(template) = top
            .molecule_type(&mol.ident)
            .filter(|t| t.atoms.len() == mol.atoms.len())
        else {
            continue;
        };
        for (atom, template) in mol.atoms.iter_mut().zip(&template.atoms) {
            if let Some(element) = top
                .atom_types
                .get(&template.atom_type)
                .and_then(|t| t.atomic_number)
                .and_then(|n| u8::try_from(n).ok())
                .and_then(|n| na_seq::Element::from_atomic_number(n).ok())
            {
                atom.element = element;
            }
            atom.force_field_type = Some(template.atom_type.clone());
            atom.partial_charge = Some(template.charge as f32);
        }
        let mut seen = std::collections::HashSet::new();
        mol.bonds = template
            .bonds
            .iter()
            .chain(&template.constraints)
            .filter_map(|term| {
                let &[a, b] = term.atoms.as_slice() else {
                    return None;
                };
                if a == b || !seen.insert((a.min(b), a.max(b))) {
                    return None;
                }

                Some(Bond {
                    atom_0: a,
                    atom_1: b,
                    atom_0_sn: mol.atoms.get(a)?.serial_number,
                    atom_1_sn: mol.atoms.get(b)?.serial_number,
                    bond_type: bio_files::BondType::Single,
                    is_backbone: false,
                })
            })
            .collect();
        mol.build_adjacency_list();
    }
}

fn run_name(dir: &Path, path: &Path, legacy: bool) -> String {
    if legacy {
        path.file_stem()
    } else {
        dir.file_name()
    }
    .unwrap_or_default()
    .to_string_lossy()
    .into_owned()
}

pub(crate) fn gromacs_config(mdp: &MdpParams) -> MdConfig {
    let mut cfg = MdConfig::from(mdp.clone());
    if mdp.integrator == bio_files::gromacs::mdp::Integrator::Md
        && let Integrator::VerletVelocity { thermostat } = cfg.integrator
    {
        cfg.integrator = Integrator::Leapfrog { thermostat };
    }
    cfg.max_init_relaxation_iters = None;

    cfg
}

fn load_xtb(dir: &Path) -> io::Result<LoadedRun> {
    let (cfg, steps, dt, has_settings) = xtb_config(dir)?;
    let interval = cfg.snapshot_handlers.memory.unwrap_or(1) as f64 * f64::from(dt);
    let path = dir.join("xtb.trj");
    let frames = bio_files::xtb::read_trajectory(&path, interval)?;
    let first = frames
        .first()
        .ok_or_else(|| io::Error::other("The xTB trajectory is empty."))?;
    let elements: Vec<_> = first
        .geometry
        .atoms
        .iter()
        .map(|atom| atom.element)
        .collect();
    let xyz = named(dir, &["geometry.xyz", "coord.xyz"]).or_else(|| {
        files(dir).into_iter().find(|path| {
            extension(path, "xyz")
                && !path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .starts_with("xtb")
        })
    });
    let geometry = xyz.as_deref().map(Xyz::load).transpose()?;
    if frames.iter().any(|frame| {
        !frame
            .geometry
            .atoms
            .iter()
            .map(|a| a.element)
            .eq(elements.iter().copied())
    }) || geometry.as_ref().is_some_and(|xyz| {
        !xyz.atoms
            .iter()
            .map(|a| a.element)
            .eq(elements.iter().copied())
    }) {
        return Err(io::Error::other(
            "xTB geometry and trajectory atom sequences differ.",
        ));
    }

    let name = run_name(dir, &path, false);
    let saved = crate::project::load_md_mol_set(&dir.join(crate::project::MD_MOLECULE_SET_FILE))
        .ok()
        .filter(|set| set.atom_count == elements.len());
    let mut mol_set = if let Some(set) = saved {
        set
    } else if let Some(mut gro) = named(dir, &["molecules.gro"])
        .and_then(|p| Gro::load(&p).ok())
        .filter(|gro| gro.atoms.len() == elements.len())
    {
        for (atom, element) in gro.atoms.iter_mut().zip(elements) {
            atom.element = element;
        }
        let mut viewer = SnapshotViewer::default();
        viewer.load_gro_data(&path, gro, &[])?;
        viewer.mol_sets.pop().unwrap()
    } else {
        let source = geometry.as_ref().unwrap_or(&first.geometry);
        let atoms: Vec<Atom> = source.atoms.iter().map(Into::into).collect();
        let bonds = bio_files::create_bonds(&source.atoms)
            .iter()
            .filter_map(|b| Bond::from_generic(b, &atoms).ok())
            .collect();
        let mol = MoleculeCommon::new(
            "xTB geometry".into(),
            atoms,
            bonds,
            Default::default(),
            None,
        );
        ViewerMolSet::from_mols(name.clone(), &[(MolType::Ligand, mol, 1)])
    };
    mol_set.name = name.clone();
    mol_set.path = Some(path);

    let snapshots = crate::xtb::trajectory_snapshots(frames);
    let mut trajectory = Trajectory::new_in_memory(snapshots.clone(), name, dt);
    trajectory.ui_active = true;
    let inputs = geometry.filter(|_| has_settings).map(|xyz| RunInputs::Xtb {
        geometry: xyz.atoms,
        mols: mol_set
            .mols
            .iter()
            .map(|mol| {
                let kind = match mol.mol_type {
                    MolType::Peptide => dynamics::FfMolType::Peptide,
                    MolType::NucleicAcid => dynamics::FfMolType::Dna,
                    MolType::Lipid => dynamics::FfMolType::Lipid,
                    _ => dynamics::FfMolType::SmallOrganic,
                };

                (kind, mol.mol.clone(), 1)
            })
            .collect(),
    });
    let cmd = fs::read_to_string(dir.join("command.txt")).unwrap_or_default();
    let arg = |name| {
        cmd.split_whitespace()
            .zip(cmd.split_whitespace().skip(1))
            .find_map(|(option, value)| (option == name).then_some(value))
    };
    let charge = arg("--chrg").and_then(|s| s.parse().ok()).or_else(|| {
        fs::read_to_string(dir.join(".CHRG"))
            .ok()?
            .trim()
            .parse()
            .ok()
    });
    let xtb_options = (charge, arg("--alpb").map(str::to_owned));

    Ok(LoadedRun {
        backend: MdBackend::xTb,
        trajectory,
        snapshots,
        mol_set,
        config: has_settings.then_some((cfg, steps, dt)),
        inputs,
        native_inputs: None,
        xtb_options: has_settings.then_some(xtb_options),
    })
}

fn xtb_config(dir: &Path) -> io::Result<(MdConfig, u32, f32, bool)> {
    let path = named(dir, &["md.inp", "xcontrol"])
        .or_else(|| files(dir).into_iter().find(|path| extension(path, "inp")));
    let has_settings = path.is_some();
    let text = path
        .map(fs::read_to_string)
        .transpose()?
        .unwrap_or_default();
    let mut section = "";
    let mut settings = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.starts_with('$') {
            section = line.split_whitespace().next().unwrap_or_default();
        } else if let Some((key, value)) = line.split_once('=') {
            settings.insert((section, key.trim()), value.trim());
        }
    }
    let number = |key, default: f64| -> io::Result<f64> {
        match settings.get(&("$md", key)) {
            Some(value) => value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v > 0.)
                .ok_or_else(|| io::Error::other(format!("Invalid xTB MD setting: {key}"))),
            None => Ok(default),
        }
    };
    let step = number("step", 0.5)?;
    let time = number("time", 50.)?;
    let dump = number("dump", 50.)?;
    let dt = step / 1000.;
    let steps = (time / dt + 1e-7).floor();
    if steps > f64::from(u32::MAX) || steps < 1. {
        return Err(io::Error::other("Invalid xTB MD run length."));
    }
    let mut cfg = MdConfig::for_family(ForceFieldFamily::Gfnn);
    cfg.temp_target = number("temp", 298.15)? as f32;
    let nvt = settings
        .get(&("$md", "nvt"))
        .is_none_or(|v| !v.eq_ignore_ascii_case("false") && *v != "0");
    cfg.integrator = Integrator::VerletVelocity {
        thermostat: nvt.then_some(dynamics::TAU_TEMP_DEFAULT),
    };
    cfg.snapshot_handlers.memory = Some((dump / step).round().max(1.) as usize);
    cfg.max_init_relaxation_iters = settings
        .get(&("$opt", "maxcycle"))
        .and_then(|v| v.parse().ok());

    Ok((cfg, steps as u32, dt as f32, has_settings))
}

impl LoadedRun {
    fn install(
        mut self,
        state: &mut State,
        scene: &mut Scene,
        updates: &mut EngineUpdates,
        dir: &Path,
    ) -> io::Result<String> {
        let name = self.trajectory.display_name.clone();
        let n_shown = self.snapshots.len();
        let total = self.trajectory.num_frames;
        let has_setup = self.inputs.is_some() || self.native_inputs.is_some();
        let has_settings =
            self.config.is_some() || matches!(&self.inputs, Some(RunInputs::Gromacs(_)));

        if let Some((cfg, steps, dt)) = self.config
            && !matches!(&self.inputs, Some(RunInputs::Gromacs(_)))
        {
            state
                .select_force_field(cfg.ff_family)
                .map_err(io::Error::other)?;
            state.to_save.md.config = cfg;
            state.to_save.md.backend = self.backend;
            state.to_save.md.num_steps = steps;
            state.to_save.md.dt = dt;
            state.ui.md.sync(&state.to_save.md.config, dt);
            state.volatile.md_local.run_time = steps as f32 * dt;
        }
        if self.native_inputs.is_none() {
            state.ui.md.peptide_only_near_ligs = false;
            state.ui.md.peptide_static = false;
            state.ui.md.alchemical_enabled = false;
        }
        if let Some(saved) = self.native_inputs.take() {
            saved.apply(state, dir);
        }
        if let Some((charge, solvent)) = self.xtb_options
            && let Some(set) = state
                .force_fields
                .params
                .as_mut()
                .and_then(|p| p.gfnff_mut())
        {
            set.total_charge = charge;
            set.alpb_solvent = solvent;
            set.parameter_file = named(dir, &[".param_gfnff.xtb"]);
            state.ui.md.alchemical_enabled = false;
            state.force_fields.invalidate();
        }
        let md = &mut state.volatile.md_local;
        md.opened_inputs = match self.inputs {
            Some(RunInputs::Gromacs(setup)) => {
                state.to_save.md.backend = MdBackend::Gromacs;
                md.gromacs_setup = *setup;

                None
            }
            inputs => inputs,
        };
        md.gromacs_output = None;
        md.pep_atom_set.clear();
        md.mol_dynamics = None;
        show_playback(
            state,
            scene,
            updates,
            self.trajectory,
            self.snapshots,
            self.mol_set,
        );

        let mut message = format!("Opened {name} ({n_shown}/{total} frames).");
        if !has_settings {
            message.push_str(" Saved MD settings are unavailable.");
        } else if !has_setup {
            message.push_str(" Select molecules to start a new run with these settings.");
        }

        Ok(message)
    }
}

fn show_playback(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    trajectory: Trajectory,
    snapshots: Vec<Arc<Snapshot>>,
    mol_set: ViewerMolSet,
) {
    let md = &mut state.volatile.md_local;
    md.viewer.playing = false;
    md.viewer.playback_accum = 0.;
    md.viewer.slider_posit_ui = 0;
    let path = mol_set.path.clone();
    let mut trajectory = trajectory;
    let set_i = path.as_ref().and_then(|path| {
        md.viewer
            .mol_sets
            .iter()
            .position(|set| set.path.as_ref() == Some(path))
    });
    let set_i = match set_i {
        Some(i) => {
            md.viewer.mol_sets[i] = mol_set;
            i
        }
        None => {
            md.viewer.mol_sets.push(mol_set);
            md.viewer.mol_sets.len() - 1
        }
    };
    md.viewer.mol_set_active = Some(set_i);
    md.replace_snaps(snapshots);

    for traj in &mut state.trajectories {
        traj.ui_active = false;
    }
    trajectory.run_mol_set_path = path.clone();
    if let Some(traj) = state
        .trajectories
        .iter_mut()
        .find(|traj| traj.run_mol_set_path.is_some() && traj.run_mol_set_path == path)
    {
        *traj = trajectory;
    } else {
        state.trajectories.push(trajectory);
    }
    reset_camera(state, scene, updates, VIEW_DIR_FRONT);
    viewer::draw_mols(state, scene, updates);
}

pub(crate) fn show_gromacs_output(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    output: &bio_files::gromacs::GromacsOutput,
) -> io::Result<()> {
    let Some(path) = output.gro_path.as_ref() else {
        return Ok(());
    };
    let dir =
        gromacs_run_dir(path).ok_or_else(|| io::Error::other("Missing GROMACS run folder"))?;
    let structures = [path.clone()];
    let run = output
        .trr_path
        .iter()
        .chain(&output.xtc_path)
        .find_map(|traj| load_binary_run(&dir, traj, &structures, MdBackend::Gromacs, false).ok())
        .map(Ok)
        .unwrap_or_else(|| load(&dir, None));
    let run = match run {
        Ok(run) => run,
        Err(_) if output.trr_path.is_none() && output.xtc_path.is_none() => return Ok(()),
        Err(error) => return Err(error),
    };
    show_playback(
        state,
        scene,
        updates,
        run.trajectory,
        run.snapshots,
        run.mol_set,
    );

    Ok(())
}

/// GROMACS output files live directly in their run folder.
pub(crate) fn gromacs_run_dir(path: &Path) -> Option<PathBuf> {
    path.parent()?.canonicalize().ok()
}

pub(super) fn save_memory_trajectory(
    dir: &Path,
    md: &dynamics::MdState,
    dt: f32,
) -> io::Result<()> {
    if TRAJECTORIES
        .iter()
        .any(|ext| dir.join(format!("traj.{ext}")).is_file())
    {
        return Ok(());
    }

    let path = dir.join("traj.trr");
    if !md.snapshots.is_empty() {
        dynamics::snapshot::write_trr_snapshots(&path, &md.snapshots, dt)
    } else if md.step_count > 0 {
        dynamics::snapshot::write_trr_snapshots(&path, &[Snapshot::new(md)], dt)
    } else {
        Ok(())
    }
}
