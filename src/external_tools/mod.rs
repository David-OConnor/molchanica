//! Registry of the optional third-party tools Molchanica can drive.
//!
//! Molchanica works without any of these installed; each one unlocks a feature. The problem this
//! module solves is that every tool was previously discovered, probed, and reported on by
//! hand-written code in a different place (`util::orca_avail`, `util::gemmi_avail`,
//! `opendde::find_executable`, ...), each with its own idea of where to look and what counts as
//! "installed". Adding a tool meant writing that logic again, plus one more install script per
//! operating system.
//!
//! Instead there is one table — [`Tool::spec`] — describing every tool, and three generic
//! operations over it:
//!
//! - [`find_executable`]: resolve a tool to an absolute path, or explain why it can't be found.
//! - [`check`] / [`check_all`]: probe whether it actually runs, for the tools status panel.
//! - `install_scripts/install_tool.{sh,ps1} <slug>`: install it, parameterized by the same slugs.
//!
//! # Where tools live
//!
//! Molchanica-managed installs go under one per-user data directory (see [`data_root`]):
//!
//! ```text
//! <data root>/molchanica/
//!     opendde-venv/            a Python virtual environment per Python-based tool, named
//!     boltz2-venv/             <slug>-venv. These deliberately do not share an interpreter:
//!     ligandmpnn-venv/         OpenDDE wants Python >= 3.11, Boltz-2 wants < 3.13, and
//!     anarcii-venv/            ProteinMPNN wants numpy < 2, whose newest wheel is cp312.
//!     tools/
//!         igblast/             a self-contained binary distribution
//!         LigandMPNN/          a checkout plus its downloaded model weights
//!         ProteinMPNN/
//! ```
//!
//! Resolution never relies on `PATH` alone: a desktop-launched application does not inherit
//! `PATH` additions made by a shell profile, which is the single most common reason an installed
//! tool appears missing. `PATH` is consulted last, so a system-wide install still works.

use std::{
    env, fmt, fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

pub mod anarcii;
pub mod igblast;
pub mod mpnn;
pub mod pdb_write;

/// How long a `--version`/`--help` style probe is given before it is killed.
///
/// Generous because some of these are Python entry points that import Torch before answering, and
/// a cold page cache makes that take seconds. A probe that times out is reported as an error
/// rather than as "not installed", since those call for different fixes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// A short probe for tools that are plain native binaries, where a slow answer means something is
/// wrong rather than merely cold. Also guards against the ORCA/screen-reader hang: an `orca` that
/// is actually the Orca screen reader never answers and must not block the UI.
const PROBE_TIMEOUT_NATIVE: Duration = Duration::from_secs(3);

/// One optional third-party tool.
///
/// Adding a variant plus its [`Tool::spec`] entry is all that is needed for it to appear in the
/// status panel and be resolvable through [`find_executable`]; the install scripts key off the
/// same [`ToolSpec::slug`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Tool {
    OpenDde,
    Boltz2,
    IgBlast,
    ProteinMpnn,
    LigandMpnn,
    Anarcii,
    Gromacs,
    Orca,
    Gemmi,
}

impl Tool {
    /// Every tool, in the order the status panel lists them: prediction and design first, then the
    /// simulation and file-format helpers.
    pub const ALL: [Self; 9] = [
        Self::OpenDde,
        Self::Boltz2,
        Self::LigandMpnn,
        Self::ProteinMpnn,
        Self::IgBlast,
        Self::Anarcii,
        Self::Gromacs,
        Self::Orca,
        Self::Gemmi,
    ];

    pub fn spec(self) -> &'static ToolSpec {
        REGISTRY
            .iter()
            .find(|spec| spec.tool == self)
            .expect("every Tool variant has a registry entry")
    }

    /// The tools Molchanica installs itself, in the order `install_tool <slug>` accepts.
    pub fn managed() -> impl Iterator<Item = Self> {
        Self::ALL.into_iter().filter(|t| t.spec().molchanica_managed)
    }
}

impl fmt::Display for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.spec().name)
    }
}

/// How a tool is launched, which determines where it is looked for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolKind {
    /// A standalone native binary. Looked for in the tool's bundle directory, then on `PATH`.
    Executable,
    /// A console script (`opendde`, `boltz`, ...) inside a Molchanica-managed virtual environment.
    VenvScript,
    /// The interpreter of a Molchanica-managed virtual environment. Used where the tool's code is
    /// a checkout rather than a package with an entry point, and where driving the Python API
    /// directly is more robust than depending on a CLI's argument shape.
    VenvPython,
}

/// A file that must exist before a tool can actually do anything — typically model weights, which
/// are downloaded separately from the code and are the usual reason an "installed" tool fails on
/// first use.
#[derive(Clone, Copy, Debug)]
pub struct RequiredAsset {
    /// Relative to the tool's bundle directory.
    pub relative_path: &'static str,
    pub description: &'static str,
}

/// Everything Molchanica knows about one tool.
pub struct ToolSpec {
    pub tool: Tool,
    /// How the tool is written wherever a person reads it.
    pub name: &'static str,
    /// The machine-readable identifier. Names the virtual environment (`<slug>-venv`), and is what
    /// `install_tool.sh`/`install_tool.ps1` take as their argument. Changing it moves both.
    pub slug: &'static str,
    /// One line, shown in the status panel.
    pub summary: &'static str,
    pub url: &'static str,
    /// Licence terms of the whole stack a run needs, not just the upstream repository's label.
    pub license: &'static str,
    pub kind: ToolKind,
    /// Base name of the console script or binary, without any platform suffix.
    pub executable: &'static str,
    /// An absolute path here overrides all discovery. Always `MOLCHANICA_<TOOL>_EXECUTABLE`.
    pub exe_override_env: &'static str,
    /// Points at the tool's bundle or virtual-environment root, overriding the managed location.
    pub root_override_env: Option<&'static str>,
    /// Subdirectory of `<data root>/molchanica/tools` holding a binary distribution or checkout.
    /// Self-contained distributions are unpacked whole: IgBLAST resolves `internal_data/` relative
    /// to itself, so the binary sits below the bundle root rather than at it.
    pub bundle_subdir: Option<&'static str>,
    /// Also look beside the Molchanica executable. For tools we may ship in the release zip.
    pub colocated: bool,
    /// Weights and data files, relative to the bundle directory.
    pub required_assets: &'static [RequiredAsset],
    /// Whether `install_tool` can install it. False for tools with a licence gate or an installer
    /// of their own (ORCA), which the user has to obtain themselves.
    pub molchanica_managed: bool,
    /// Shown when the tool cannot be found.
    pub install_hint: &'static str,
    /// Arguments to a probe that proves the right program answered.
    pub version_args: &'static [&'static str],
    /// A substring the probe's output must contain. Guards against name collisions — `orca` is
    /// also a screen reader, and `gmx` output has to actually be GROMACS.
    pub version_marker: &'static str,
    /// Whether the probe is expected to be slow because it imports a scientific Python stack.
    pub slow_probe: bool,
}

impl ToolSpec {
    /// `<data root>/molchanica/<slug>-venv`, or the `root_override_env` value.
    pub fn venv_root(&self) -> Option<PathBuf> {
        if let Some(name) = self.root_override_env
            && let Some(configured) = env::var_os(name)
        {
            return Some(PathBuf::from(configured));
        }
        data_root().map(|root| root.join(format!("{}-venv", self.slug)))
    }

    /// `<data root>/molchanica/tools/<bundle_subdir>`, or the `root_override_env` value.
    pub fn bundle_root(&self) -> Option<PathBuf> {
        if let Some(name) = self.root_override_env
            && let Some(configured) = env::var_os(name)
        {
            return Some(PathBuf::from(configured));
        }
        let subdir = self.bundle_subdir?;
        data_root().map(|root| root.join("tools").join(subdir))
    }

    /// The command that installs this tool, quoted for the current platform's shell.
    pub fn install_command(&self) -> String {
        if !self.molchanica_managed {
            return self.install_hint.to_owned();
        }
        if cfg!(target_os = "windows") {
            format!(".\\install_scripts\\install_tool.ps1 {}", self.slug)
        } else {
            format!("./install_scripts/install_tool.sh {}", self.slug)
        }
    }
}

/// The single description of every tool. See the module docs for how it is used.
static REGISTRY: &[ToolSpec] = &[
    ToolSpec {
        tool: Tool::OpenDde,
        name: "OpenDDE",
        slug: "opendde",
        summary: "All-atom co-folding: proteins, DNA/RNA, ligands, ions, and complexes.",
        url: "https://github.com/aurekaresearch/OpenDDE",
        license: "Apache 2.0. Commercial use permitted.",
        kind: ToolKind::VenvScript,
        executable: "opendde",
        exe_override_env: "MOLCHANICA_OPENDDE_EXECUTABLE",
        // Predates this registry and is documented, so it keeps its own name rather than becoming
        // MOLCHANICA_OPENDDE_ROOT; existing installs and shell profiles continue to work.
        root_override_env: Some("OPENDDE_VENV_DIR"),
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: "Run install_tool with `opendde`, or set MOLCHANICA_OPENDDE_EXECUTABLE.",
        version_args: &["--version"],
        version_marker: "opendde",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::Boltz2,
        name: "Boltz-2",
        slug: "boltz2",
        summary: "Co-folding with binding-affinity prediction.",
        url: "https://github.com/jwohlwend/boltz",
        license: "MIT. Commercial use permitted.",
        kind: ToolKind::VenvScript,
        executable: "boltz",
        exe_override_env: "MOLCHANICA_BOLTZ_EXECUTABLE",
        root_override_env: Some("MOLCHANICA_BOLTZ_VENV_DIR"),
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: "Run install_tool with `boltz2`, or set MOLCHANICA_BOLTZ_EXECUTABLE.",
        // Boltz has no --version; `boltz --help` exits 0 and lists its `predict` subcommand.
        version_args: &["--help"],
        version_marker: "predict",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::LigandMpnn,
        name: "LigandMPNN",
        slug: "ligandmpnn",
        summary: "Inverse folding: design sequences for a backbone, in ligand and nucleic-acid context.",
        url: "https://github.com/dauparas/LigandMPNN",
        license: "MIT. Commercial use permitted.",
        kind: ToolKind::VenvPython,
        executable: "python",
        exe_override_env: "MOLCHANICA_LIGANDMPNN_PYTHON",
        root_override_env: Some("MOLCHANICA_LIGANDMPNN_ROOT"),
        bundle_subdir: Some("LigandMPNN"),
        colocated: false,
        required_assets: &[
            RequiredAsset {
                relative_path: "run.py",
                description: "the LigandMPNN checkout",
            },
            RequiredAsset {
                relative_path: "model_params/ligandmpnn_v_32_010_25.pt",
                description: "the default LigandMPNN weights",
            },
        ],
        molchanica_managed: true,
        install_hint: "Run install_tool with `ligandmpnn`.",
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::ProteinMpnn,
        name: "ProteinMPNN",
        slug: "proteinmpnn",
        summary: "Inverse folding, plus the antibody-finetuned AbMPNN weights.",
        url: "https://github.com/dauparas/ProteinMPNN",
        license: "MIT code; AbMPNN weights CC BY 4.0. Commercial use permitted with attribution.",
        kind: ToolKind::VenvPython,
        executable: "python",
        exe_override_env: "MOLCHANICA_PROTEINMPNN_PYTHON",
        root_override_env: Some("MOLCHANICA_PROTEINMPNN_ROOT"),
        bundle_subdir: Some("ProteinMPNN"),
        colocated: false,
        required_assets: &[
            RequiredAsset {
                relative_path: "protein_mpnn_run.py",
                description: "the ProteinMPNN checkout",
            },
            RequiredAsset {
                relative_path: "vanilla_model_weights/v_48_020.pt",
                description: "the default ProteinMPNN weights",
            },
        ],
        molchanica_managed: true,
        install_hint: "Run install_tool with `proteinmpnn`.",
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::IgBlast,
        name: "IgBLAST",
        slug: "igblast",
        summary: "Antibody V(D)J germline assignment and framework/CDR delineation.",
        url: "https://ncbi.github.io/igblast/",
        license: "US Government public domain. No restrictions.",
        kind: ToolKind::Executable,
        executable: "igblastn",
        exe_override_env: "MOLCHANICA_IGBLAST_EXECUTABLE",
        root_override_env: Some("MOLCHANICA_IGBLAST_ROOT"),
        bundle_subdir: Some("igblast"),
        colocated: false,
        required_assets: &[RequiredAsset {
            relative_path: "internal_data",
            description: "IgBLAST's internal_data directory",
        }],
        molchanica_managed: true,
        install_hint: "Run install_tool with `igblast`.",
        // `igblastn -version` prints "igblastn: 1.22.0" plus the BLAST package line.
        version_args: &["-version"],
        version_marker: "igblast",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::Anarcii,
        name: "ANARCII",
        slug: "anarcii",
        summary: "Antibody/TCR numbering (IMGT, Kabat, Chothia, Martin) with insertion codes.",
        url: "https://github.com/oxpig/ANARCII",
        license: "BSD 3-Clause. Commercial use permitted.",
        kind: ToolKind::VenvPython,
        executable: "python",
        exe_override_env: "MOLCHANICA_ANARCII_PYTHON",
        root_override_env: Some("MOLCHANICA_ANARCII_VENV_DIR"),
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: "Run install_tool with `anarcii`.",
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::Gromacs,
        name: "GROMACS",
        slug: "gromacs",
        summary: "Molecular dynamics; an alternative backend to Molchanica's native MD.",
        url: "https://www.gromacs.org/",
        license: "LGPL 2.1. Commercial use permitted; redistribution is reciprocal.",
        kind: ToolKind::Executable,
        executable: "gmx",
        exe_override_env: "MOLCHANICA_GROMACS_EXECUTABLE",
        root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: false,
        install_hint: "Install GROMACS from https://www.gromacs.org/ and put `gmx` on PATH, \
                       or set MOLCHANICA_GROMACS_EXECUTABLE.",
        version_args: &["-version"],
        version_marker: "GROMACS version",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::Orca,
        name: "ORCA",
        slug: "orca",
        summary: "Quantum chemistry: geometry optimization, single-point energies, MBIS charges.",
        url: "https://www.faccts.de/orca/",
        license: "Free for academic use; a licence is required for commercial use.",
        kind: ToolKind::Executable,
        executable: "orca",
        exe_override_env: "MOLCHANICA_ORCA_EXECUTABLE",
        root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: false,
        install_hint: "Register at https://www.faccts.de/orca/ and put `orca` on PATH, \
                       or set MOLCHANICA_ORCA_EXECUTABLE.",
        // Not a valid ORCA flag, but it prints its banner anyway, which is what we match on. The
        // banner check matters: `orca` on Linux is often the GNOME screen reader.
        version_args: &["--help"],
        version_marker: "O   R   C   A",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::Gemmi,
        name: "Gemmi",
        slug: "gemmi",
        summary: "Converts MTZ and unprocessed electron-density files.",
        url: "https://gemmi.readthedocs.io/",
        license: "MPL 2.0. Commercial use permitted.",
        kind: ToolKind::Executable,
        executable: "gemmi",
        exe_override_env: "MOLCHANICA_GEMMI_EXECUTABLE",
        root_override_env: None,
        bundle_subdir: None,
        colocated: true,
        required_assets: &[],
        molchanica_managed: false,
        install_hint: "Install gemmi (`apt install gemmi`, `pip install gemmi`), \
                       or set MOLCHANICA_GEMMI_EXECUTABLE.",
        version_args: &["--help"],
        version_marker: "GEMMI library",
        slow_probe: false,
    },
];

/// Molchanica's per-user data directory, where managed tools are installed.
///
/// Matches the locations the install scripts write to. These are the platform conventions:
/// `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, and `$XDG_DATA_HOME`
/// (defaulting to `~/.local/share`) elsewhere.
pub fn data_root() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);

    #[cfg(target_os = "macos")]
    let base = home_directory().map(|home| home.join("Library/Application Support"));

    #[cfg(all(unix, not(target_os = "macos")))]
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| home_directory().map(|home| home.join(".local/share")));

    base.map(|base| base.join("molchanica"))
}

pub fn home_directory() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Resolve a tool to an absolute path, or explain what to do about it.
///
/// The override environment variable always wins, then the Molchanica-managed location, then
/// `PATH`. Preferring the managed location over `PATH` is deliberate: a user who ran our installer
/// should get the environment we built for them rather than an unrelated `pip install` that
/// happens to be earlier on `PATH`.
pub fn find_executable(tool: Tool) -> io::Result<PathBuf> {
    let spec = tool.spec();

    if let Some(configured) = env::var_os(spec.exe_override_env) {
        let configured = PathBuf::from(configured);
        if configured.is_file() {
            return Ok(configured);
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{} points to {}, but that file does not exist",
                spec.exe_override_env,
                configured.display()
            ),
        ));
    }

    match spec.kind {
        ToolKind::VenvScript => {
            if let Some(root) = spec.venv_root()
                && let Some(found) = executable_in(&venv_bin(&root), spec.executable)
            {
                return Ok(found);
            }
            // A `uv tool install` is the other way these console scripts commonly arrive.
            for directory in uv_tool_bin_directories() {
                if let Some(found) = executable_in(&directory, spec.executable) {
                    return Ok(found);
                }
            }
        }
        ToolKind::VenvPython => {
            if let Some(root) = spec.venv_root()
                && let Some(found) = executable_in(&venv_bin(&root), "python")
            {
                return Ok(found);
            }
            // Deliberately no PATH fallback: running third-party model code under whatever
            // `python` happens to be first on PATH is how the earlier Boltz-2 and ESMFold
            // integrations broke. An interpreter must be one we built or one explicitly named.
            return Err(not_found(spec));
        }
        ToolKind::Executable => {
            if spec.colocated
                && let Some(found) = colocated_executable(spec.executable)
            {
                return Ok(found);
            }
            if let Some(root) = spec.bundle_root() {
                for directory in [root.clone(), root.join("bin")] {
                    if let Some(found) = executable_in(&directory, spec.executable) {
                        return Ok(found);
                    }
                }
            }
        }
    }

    if let Some(found) = find_on_path(spec.executable) {
        return Ok(found);
    }

    Err(not_found(spec))
}

/// The directory a bundled tool's data files sit in, which several tools need passed to them
/// explicitly (IgBLAST's `IGDATA`, the MPNN checkouts' model parameters).
pub fn bundle_root(tool: Tool) -> io::Result<PathBuf> {
    tool.spec().bundle_root().filter(|root| root.is_dir()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{} is not installed. {}",
                tool.spec().name,
                tool.spec().install_command()
            ),
        )
    })
}

fn not_found(spec: &ToolSpec) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("{} was not found. {}", spec.name, spec.install_command()),
    )
}

fn venv_bin(root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        root.join("Scripts")
    } else {
        root.join("bin")
    }
}

fn colocated_executable(name: &str) -> Option<PathBuf> {
    let directory = env::current_exe().ok()?.parent()?.to_path_buf();
    executable_in(&directory, name)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .and_then(|path| env::split_paths(&path).find_map(|dir| executable_in(&dir, name)))
}

fn uv_tool_bin_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Some(path) = env::var_os("UV_TOOL_BIN_DIR") {
        directories.push(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("XDG_BIN_HOME") {
        directories.push(PathBuf::from(path));
    }
    if let Some(home) = home_directory() {
        directories.push(home.join(".local/bin"));
    }
    directories
}

/// A file named `name` in `directory` that we could plausibly execute.
pub fn executable_in(directory: &Path, name: &str) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let names = [
        format!("{name}.exe"),
        format!("{name}.cmd"),
        format!("{name}.bat"),
        name.to_owned(),
    ];
    #[cfg(not(target_os = "windows"))]
    let names = [name.to_owned()];

    names
        .into_iter()
        .map(|name| directory.join(name))
        .find(|candidate| candidate.is_file())
}

/// Outcome of probing whether a tool can be reached and run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckResult {
    /// Found, and it answered a version probe as itself.
    Pass,
    /// Not installed anywhere we look. The fix is to install it.
    CantFind,
    /// Installed but not working — wrong program, missing weights, a broken environment. The fix
    /// is to go read the detail, so this sorts above `CantFind`.
    Error,
}

impl CheckResult {
    /// Sort order for the status panel: working tools, then broken ones, then absent ones.
    pub fn rank(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Error => 1,
            Self::CantFind => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pass => "Ready",
            Self::CantFind => "Not installed",
            Self::Error => "Error",
        }
    }
}

/// One tool's status, as shown in the tools panel.
#[derive(Clone, Debug)]
pub struct ToolStatus {
    pub tool: Tool,
    pub result: CheckResult,
    /// A version string on success; the reason otherwise.
    pub detail: String,
    /// Where it was found, when it was.
    pub path: Option<PathBuf>,
}

impl ToolStatus {
    fn cant_find(tool: Tool, detail: String) -> Self {
        Self {
            tool,
            result: CheckResult::CantFind,
            detail,
            path: None,
        }
    }
}

/// Whether a tool resolves to a file that exists, without running anything.
///
/// A filesystem lookup only, so it is cheap enough for startup, where [`check`] is not: the
/// Python-based tools import Torch before they will answer a version probe, which is seconds each.
/// It cannot tell a working install from a broken one — that is what [`check`] and the tools panel
/// are for — but it does distinguish "installed" from "absent", which is what enabling or greying
/// out a button needs.
pub fn is_installed(tool: Tool) -> bool {
    let Ok(_) = find_executable(tool) else {
        return false;
    };
    let spec = tool.spec();
    let Some(root) = spec.bundle_root() else {
        return spec.required_assets.is_empty();
    };
    spec.required_assets
        .iter()
        .all(|asset| root.join(asset.relative_path).exists())
}

/// Probe one tool. Runs a subprocess, so keep it off the UI thread.
pub fn check(tool: Tool) -> ToolStatus {
    let spec = tool.spec();

    let executable = match find_executable(tool) {
        Ok(path) => path,
        Err(error) => return ToolStatus::cant_find(tool, error.to_string()),
    };

    // Weights and data directories are checked before the probe: a checkout whose model
    // parameters never downloaded passes a `python --version` probe and then fails on first use,
    // which is exactly the confusion this panel exists to prevent.
    if !spec.required_assets.is_empty() {
        let Some(root) = spec.bundle_root() else {
            return ToolStatus::cant_find(tool, format!("no data directory for {}", spec.name));
        };
        for asset in spec.required_assets {
            if !root.join(asset.relative_path).exists() {
                return ToolStatus {
                    tool,
                    result: CheckResult::Error,
                    detail: format!(
                        "Missing {} ({}). Re-run: {}",
                        asset.description,
                        root.join(asset.relative_path).display(),
                        spec.install_command()
                    ),
                    path: Some(executable),
                };
            }
        }
    }

    let timeout = if spec.slow_probe {
        PROBE_TIMEOUT
    } else {
        PROBE_TIMEOUT_NATIVE
    };

    let output = match probe(&executable, spec.version_args, timeout) {
        Ok(output) => output,
        Err(error) => {
            return ToolStatus {
                tool,
                result: CheckResult::Error,
                detail: error.to_string(),
                path: Some(executable),
            };
        }
    };

    if !output.to_lowercase().contains(&spec.version_marker.to_lowercase()) {
        return ToolStatus {
            tool,
            result: CheckResult::Error,
            detail: format!(
                "{} did not identify itself as {}. Its output was: {}",
                executable.display(),
                spec.name,
                first_line(&output)
            ),
            path: Some(executable),
        };
    }

    ToolStatus {
        tool,
        result: CheckResult::Pass,
        detail: first_line(&output).to_owned(),
        path: Some(executable),
    }
}

/// Probe every tool concurrently and sort so the actionable rows come last.
///
/// Concurrent because a serial pass costs the sum of every Python interpreter's Torch import.
pub fn check_all() -> Vec<ToolStatus> {
    use rayon::prelude::*;

    let mut statuses: Vec<_> = Tool::ALL.par_iter().map(|tool| check(*tool)).collect();
    statuses.sort_by_key(|status| {
        (
            status.result.rank(),
            Tool::ALL.iter().position(|t| *t == status.tool).unwrap_or(usize::MAX),
        )
    });
    statuses
}

/// Run a short-lived probe, killing it if it overruns.
///
/// Neither a nonzero exit nor stderr output is treated as failure: many CLIs answer a flag they
/// don't recognize with a usage message and a nonzero status, and we match on the text either way.
fn probe(executable: &Path, args: &[&str], timeout: Duration) -> io::Result<String> {
    let mut child = Command::new(executable)
        .args(args)
        // Without this, a Python entry point can buffer its whole answer and look like a hang.
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("unable to run {}: {error}", executable.display()),
            )
        })?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait()? {
            Some(_) => {
                let output = child.wait_with_output()?;
                return Ok(String::from_utf8_lossy(&output.stdout).to_string()
                    + &String::from_utf8_lossy(&output.stderr));
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "{} did not answer within {}s",
                        executable.display(),
                        timeout.as_secs()
                    ),
                ));
            }
            None => sleep(Duration::from_millis(20)),
        }
    }
}

fn first_line(text: &str) -> &str {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("(no output)")
}

/// A scratch directory for one tool run, removed when dropped.
///
/// Shared by every adapter here: they all write inputs to a temporary directory, run a process
/// against it, and read results back out.
pub struct ToolWorkspace {
    root: PathBuf,
}

impl ToolWorkspace {
    pub fn new(label: &str) -> io::Result<Self> {
        use std::{
            sync::atomic::{AtomicU64, Ordering},
            time::{SystemTime, UNIX_EPOCH},
        };
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        for _ in 0..100 {
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = env::temp_dir().join(format!(
                "molchanica-{label}-{}-{timestamp}-{counter}",
                std::process::id()
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Ok(Self { root }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "unable to allocate a unique tool workspace",
        ))
    }

    pub fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.root.join(relative)
    }

    pub fn create_dir(&self, relative: impl AsRef<Path>) -> io::Result<PathBuf> {
        let path = self.path(relative);
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}

impl Drop for ToolWorkspace {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root)
            && error.kind() != io::ErrorKind::NotFound
        {
            eprintln!(
                "Unable to remove tool workspace {}: {error}",
                self.root.display()
            );
        }
    }
}

/// Run a tool to completion, returning its stdout, with stderr folded into the error on failure.
///
/// Unlike `structure_prediction::run_model_command`, which streams output for long predictions,
/// this is for the seconds-scale runs the adapters here make.
pub fn run_to_completion(command: &mut Command, tool_name: &str) -> io::Result<String> {
    let output = command
        .stdin(Stdio::null())
        .env("PYTHONUNBUFFERED", "1")
        .output()
        .map_err(|error| {
            io::Error::new(error.kind(), format!("unable to start {tool_name}: {error}"))
        })?;

    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    Err(io::Error::other(format!(
        "{tool_name} exited with {}: {}",
        output.status,
        crate::util::truncate_str(detail, 4_096)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_variant_has_a_registry_entry() {
        for tool in Tool::ALL {
            let spec = tool.spec();
            assert_eq!(spec.tool, tool);
            assert!(!spec.slug.is_empty());
            assert!(!spec.version_marker.is_empty());
        }
    }

    #[test]
    fn slugs_and_override_variables_are_unique() {
        // A duplicate slug would make two tools share a virtual environment, and a duplicate
        // override variable would silently point one tool at another's executable.
        for (index, spec) in REGISTRY.iter().enumerate() {
            for other in &REGISTRY[index + 1..] {
                assert_ne!(spec.slug, other.slug, "duplicate slug {}", spec.slug);
                assert_ne!(
                    spec.exe_override_env, other.exe_override_env,
                    "duplicate override variable {}",
                    spec.exe_override_env
                );
            }
        }
    }

    #[test]
    fn managed_tools_name_an_install_command() {
        for tool in Tool::managed() {
            assert!(tool.spec().install_command().contains(tool.spec().slug));
        }
    }

    #[test]
    fn unmanaged_tools_explain_how_to_get_them() {
        for tool in Tool::ALL {
            if !tool.spec().molchanica_managed {
                assert_eq!(tool.spec().install_command(), tool.spec().install_hint);
            }
        }
    }
}
