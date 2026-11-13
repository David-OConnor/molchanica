use std::{
    fmt::Write as _,
    fs,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::Command,
};

use bio_files::{
    gromacs::{GromacsOutput, OutputEnergy, mdp::Thermostat},
    md_out::create_run_dir,
};

use crate::{
    external_tools::{Tool, ToolWorkspace, find_executable},
    prefs::MdPrefs,
};

#[derive(Clone, Default)]
pub struct Setup {
    pub mdp: Option<String>,
    pub mdp_path: Option<PathBuf>,
    pub coordinates: Option<PathBuf>,
    pub topology: Option<PathBuf>,
    pub index: Option<PathBuf>,
    pub reference: Option<PathBuf>,
    pub continuation: Option<PathBuf>,
    pub grompp_args: String,
    pub mdrun_args: String,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum InputFile {
    Bundle,
    Mdp,
    Coordinates,
    Topology,
    Index,
    Reference,
    Continuation,
}

impl InputFile {
    pub fn filter(self) -> &'static str {
        match self {
            Self::Bundle => "GROMACS inputs",
            Self::Mdp => "MDP",
            Self::Coordinates | Self::Reference => "GRO",
            Self::Topology => "TOP",
            Self::Index => "NDX",
            Self::Continuation => "CPT / TRR / TNG",
        }
    }

    pub fn path(self, setup: &mut Setup) -> &mut Option<PathBuf> {
        match self {
            Self::Bundle | Self::Mdp => &mut setup.mdp_path,
            Self::Coordinates => &mut setup.coordinates,
            Self::Topology => &mut setup.topology,
            Self::Index => &mut setup.index,
            Self::Reference => &mut setup.reference,
            Self::Continuation => &mut setup.continuation,
        }
    }

    pub fn select(self, setup: &mut Setup, path: PathBuf) -> io::Result<()> {
        if self == Self::Mdp {
            setup.load_mdp(&path)?;
        } else {
            *self.path(setup) = Some(path);
        }

        Ok(())
    }
}

impl Setup {
    pub fn uses_files(&self) -> bool {
        self.coordinates.is_some() || self.topology.is_some()
    }

    pub fn load_mdp(&mut self, path: &Path) -> io::Result<()> {
        let mdp = fs::read_to_string(path)?;
        self.mdp = Some(mdp);
        self.mdp_path = Some(path.to_owned());

        Ok(())
    }

    pub fn load_files(&mut self, paths: Vec<PathBuf>) -> io::Result<()> {
        let mut new = if ["gro", "top"].iter().all(|ext| {
            paths.iter().any(|path| {
                path.extension()
                    .is_some_and(|value| value.eq_ignore_ascii_case(ext))
            })
        }) {
            Self {
                mdp: self.mdp.clone(),
                mdp_path: self.mdp_path.clone(),
                ..Default::default()
            }
        } else {
            self.clone()
        };
        let mut selected = std::collections::HashSet::new();
        for path in paths {
            let ext = path
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_ascii_lowercase();
            let file = match ext.as_str() {
                "mdp" => InputFile::Mdp,
                "gro" => InputFile::Coordinates,
                "top" => InputFile::Topology,
                "ndx" => InputFile::Index,
                "cpt" | "trr" | "tng" => InputFile::Continuation,
                "itp" => continue,
                _ => {
                    return Err(io::Error::other(format!(
                        "Choose the .top that includes {}",
                        path.display()
                    )));
                }
            };
            if !selected.insert(file) {
                return Err(io::Error::other(format!(
                    "Choose one {} file at a time.",
                    file.filter()
                )));
            }
            file.select(&mut new, path)?;
        }
        if selected.is_empty() {
            return Err(io::Error::other(
                "Choose the .top that includes the parameter files.",
            ));
        }
        *self = new;

        Ok(())
    }

    pub(super) fn validate(&self) -> io::Result<()> {
        if self.mdp.is_none() || self.coordinates.is_none() || self.topology.is_none() {
            return Err(io::Error::other("Choose an MDP, GRO, and TOP file."));
        }

        Ok(())
    }

    pub(super) fn run(&self) -> io::Result<GromacsOutput> {
        self.validate()?;
        let executable = absolute(&find_executable(Tool::Gromacs)?)?;
        let dir = absolute(&create_run_dir("gromacs")?)?;

        self.run_in(&executable, &dir)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", dir.display())))
    }

    pub(super) fn save(&self, dir: &Path) -> io::Result<()> {
        self.validate()?;
        let executable = absolute(&find_executable(Tool::Gromacs)?)?;
        let dir = absolute(dir)?;
        let scratch = ToolWorkspace::new(Tool::Gromacs)?;
        let staging = absolute(scratch.root())?;
        self.preprocess(&executable, &staging)?;

        fs::create_dir_all(&dir)?;
        for (source, dest) in [
            ("conf.gro", "conf.gro"),
            ("topo.top", "topo.top"),
            ("replay.mdp", "md.mdp"),
            ("topol.tpr", "topol.tpr"),
            ("index.ndx", "index.ndx"),
            ("restraint.gro", "restraint.gro"),
        ] {
            let source = staging.join(source);
            if source.is_file() {
                fs::copy(source, dir.join(dest))?;
            }
        }
        let restart = staging.join("continuation");
        if restart.is_dir() {
            let dest = dir.join("continuation");
            fs::create_dir_all(&dest)?;
            for entry in fs::read_dir(restart)? {
                let entry = entry?;
                fs::copy(entry.path(), dest.join(entry.file_name()))?;
            }
        }

        Ok(())
    }

    fn preprocess(&self, executable: &Path, dir: &Path) -> io::Result<()> {
        let topology = absolute(self.topology.as_deref().unwrap())?;
        let source_dir = self
            .mdp_path
            .as_deref()
            .map(absolute)
            .transpose()?
            .as_deref()
            .and_then(Path::parent)
            .unwrap_or_else(|| topology.parent().unwrap())
            .to_owned();
        let gro = dir.join("conf.gro");
        let mdp = dir.join("md.mdp");
        let tpr = dir.join("topol.tpr");
        let processed_top = dir.join("topo.top");
        fs::copy(self.coordinates.as_ref().unwrap(), &gro)?;
        fs::write(&mdp, self.mdp.as_ref().unwrap())?;

        let mut preprocess = command(executable, &source_dir);
        preprocess
            .args(["grompp", "-f"])
            .arg(&mdp)
            .arg("-c")
            .arg(&gro)
            .arg("-p")
            .arg(&topology)
            .arg("-po")
            .arg(dir.join("mdout.mdp"))
            .arg("-pp")
            .arg(&processed_top)
            .arg("-o")
            .arg(&tpr);
        let extra = arguments(&self.grompp_args);
        if let Some(index) = &self.index
            && !extra.contains(&"-n")
        {
            let dest = dir.join("index.ndx");
            fs::copy(index, &dest)?;
            preprocess.arg("-n").arg(dest);
        }
        if !extra.contains(&"-r") {
            let reference = match &self.reference {
                Some(path) => {
                    let dest = dir.join("restraint.gro");
                    fs::copy(path, &dest)?;
                    dest
                }
                None => gro,
            };
            preprocess.arg("-r").arg(reference);
        }
        if let Some(continuation) = &self.continuation
            && !extra.contains(&"-t")
        {
            // Store continuation inputs apart from outputs such as state.cpt and traj.trr.
            let restart_dir = dir.join("continuation");
            fs::create_dir_all(&restart_dir)?;
            let dest = restart_dir.join(continuation.file_name().unwrap_or_default());
            fs::copy(continuation, &dest)?;
            preprocess.arg("-t").arg(dest);
        }
        preprocess.args(extra);
        run_command(dir, "grompp", &mut preprocess)?;

        // The processed topology already contains the selected branches and includes.
        let mut replay = self.mdp.clone().unwrap();
        set_mdp_value(&mut replay, "define", "");
        set_mdp_value(&mut replay, "include", "");
        fs::write(dir.join("replay.mdp"), replay)
    }

    fn run_in(&self, executable: &Path, dir: &Path) -> io::Result<GromacsOutput> {
        self.preprocess(executable, dir)?;
        let mut run = command(executable, dir);
        let extra = arguments(&self.mdrun_args);
        run.arg("mdrun");
        for (flag, default) in [
            ("-s", "topol.tpr"),
            ("-o", "traj.trr"),
            ("-x", "traj.xtc"),
            ("-c", "confout.gro"),
            ("-e", "energy.edr"),
            ("-g", "md.log"),
        ] {
            if !extra.contains(&flag) {
                run.args([flag, default]);
            }
        }
        if extra.contains(&"-cpi") && !extra.contains(&"-append") && !extra.contains(&"-noappend") {
            run.arg("-noappend");
        }
        run.args(&extra);
        run_command(dir, "mdrun", &mut run)?;

        let output = |flag: &str, default: &str| {
            let path = argument_value(&extra, flag).unwrap_or(default);
            output_file(&dir.join(path))
        };
        let trr = output("-o", "traj.trr");
        let xtc = output("-x", "traj.xtc");
        let energy = output("-e", "energy.edr");
        if energy.is_file() {
            let _ = OutputEnergy::from_edr(&energy);
        }
        let final_gro = output("-c", "confout.gro");
        let standard_gro = dir.join("confout.gro");
        if final_gro.is_file()
            && final_gro
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gro"))
            && final_gro != standard_gro
        {
            fs::copy(&final_gro, standard_gro)?;
        }

        Ok(GromacsOutput {
            log_text: fs::read_to_string(output("-g", "md.log")).unwrap_or_default(),
            gro_path: Some(dir.join("conf.gro")),
            trr_path: trr.is_file().then_some(trr),
            xtc_path: xtc.is_file().then_some(xtc),
            final_gro_text: fs::read_to_string(final_gro).ok(),
            final_topology_text: fs::read_to_string(dir.join("topo.top")).ok(),
            ..Default::default()
        })
    }
}

fn output_file(path: &Path) -> PathBuf {
    // With -noappend, GROMACS inserts a part number before the extension.
    if !path.is_file()
        && let Some((stem, ext)) = path.file_stem().zip(path.extension())
        && let Some(dir) = path.parent()
    {
        let prefix = format!("{}.part", stem.to_string_lossy());
        let latest = fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|p| {
                p.extension() == Some(ext)
                    && p.file_stem()
                        .is_some_and(|s| s.to_string_lossy().starts_with(&prefix))
            })
            .max();
        if let Some(latest) = latest {
            return latest;
        }
    }

    path.to_owned()
}

fn absolute(path: &Path) -> io::Result<PathBuf> {
    // Unlike canonicalize on Windows, this keeps paths in the form expected by GROMACS.
    std::path::absolute(path)
}

fn command(executable: &Path, dir: &Path) -> Command {
    let mut command = Command::new(executable);
    command.current_dir(dir).env("GMX_MAXBACKUP", "-1");
    if std::env::var_os("GMXLIB").is_none()
        && let Some(prefix) = executable.parent().and_then(Path::parent)
    {
        let top = prefix.join("share/gromacs/top");
        if top.join("residuetypes.dat").is_file() {
            command.env("GMXLIB", top);
        }
    }

    command
}

fn run_command(dir: &Path, name: &str, command: &mut Command) -> io::Result<()> {
    let open_log = |file| {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(file))
    };
    let stderr_path = dir.join("stderr.log");
    let stderr = open_log("stderr.log")?;
    let start = stderr.metadata()?.len();
    let status = command
        .stdin(std::process::Stdio::null())
        .stdout(open_log("stdout.log")?)
        .stderr(stderr)
        .status()?;

    if !status.success() {
        let mut log = fs::File::open(stderr_path)?;
        let end = log.metadata()?.len();
        log.seek(SeekFrom::Start(start.max(end.saturating_sub(32768))))?;
        let mut tail = Vec::new();
        log.read_to_end(&mut tail)?;

        return Err(io::Error::other(format!(
            "gmx {name} exited with {status}.\n{}",
            String::from_utf8_lossy(&tail)
        )));
    }

    Ok(())
}

fn arguments(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

fn argument_value<'a>(args: &[&'a str], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|pair| (pair[0] == flag && !pair[1].starts_with('-')).then_some(pair[1]))
}

pub fn generated_mdp(prefs: &MdPrefs) -> String {
    let mut mdp = prefs.config.to_gromacs(prefs.num_steps as usize, prefs.dt);
    if matches!(
        prefs.config.integrator,
        dynamics::Integrator::Leapfrog { thermostat: None }
            | dynamics::Integrator::VerletVelocity { thermostat: None }
    ) {
        mdp.thermostat = Thermostat::No;
    }

    mdp.to_mdp_str()
}

fn key_matches(left: &str, right: &str) -> bool {
    left.bytes()
        .filter(|c| !matches!(c, b'-' | b'_'))
        .map(|c| c.to_ascii_lowercase())
        .eq(right
            .bytes()
            .filter(|c| !matches!(c, b'-' | b'_'))
            .map(|c| c.to_ascii_lowercase()))
}

fn assignment(line: &str) -> Option<(&str, &str)> {
    let (key, value) = line.split(';').next()?.split_once('=')?;

    Some((key.trim(), value.trim_start()))
}

pub fn mdp_field_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines()
        .filter_map(assignment)
        .rfind(|(name, _)| key_matches(name, key))
        .map(|(_, value)| value)
}

pub fn mdp_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    mdp_field_value(text, key).map(str::trim_end)
}

pub fn set_mdp_value(text: &mut String, key: &str, value: &str) {
    let mut updated = String::new();
    let mut written = false;
    for line in text.split_inclusive('\n') {
        if let Some((name, _)) = assignment(line)
            && key_matches(name, key)
        {
            if !written && !value.trim().is_empty() {
                let comment = line.trim_end_matches(['\r', '\n']).split_once(';');
                let ending = if line.ends_with("\r\n") {
                    "\r\n"
                } else if line.ends_with('\n') {
                    "\n"
                } else {
                    ""
                };
                let _ = write!(updated, "{key} = {value}");
                if let Some((_, comment)) = comment {
                    let _ = write!(updated, ";{comment}");
                }
                updated.push_str(ending);
            }
            written = true;
        } else {
            updated.push_str(line);
        }
    }
    if !written && !value.trim().is_empty() {
        if !updated.is_empty() && !updated.ends_with('\n') {
            updated.push('\n');
        }
        let _ = writeln!(updated, "{key} = {value}");
    }
    *text = updated;
}
