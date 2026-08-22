//! Registry of the optional third-party tools Molchanica can drive.
//!
//! This should be kept somewhat in sync with `bio_web`, in tool selection (Although this will be
//! close to a subset), and approach to installing and managing their [e.g. python] environments.
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
//! - [`install`]: install a managed tool through the shared `bio_tools` Rust recipes.
//!
//! # Where tools live
//!
//! Molchanica-managed installs go under one per-user data directory (see [`data_root`]):
//!
//! ```text
//! <data root>/molchanica/
//!     opendde-venv/            a uv-managed Python environment per Python-based tool, named
//!     boltz2-venv/             <slug>-venv. These deliberately do not share an interpreter:
//!     ligandmpnn-venv/         OpenDDE wants Python >= 3.11, Boltz-2 wants < 3.13, and
//!     anarcii-venv/            ProteinMPNN wants numpy < 2, whose newest wheel is cp312.
//!     tools/
//!         igblast/             a self-contained binary distribution
//!         LigandMPNN/          a checkout plus its downloaded model weights
//!         ProteinMPNN/
//! ```
//!
//! Native tools may still be resolved through `PATH`. Python tools deliberately may not: they run
//! only from the uv environment built for that tool, unless an explicit override names another
//! executable. This prevents a desktop launch from silently selecting a system Python with an
//! incompatible package set.

use std::{
    env,
    ffi::OsStr,
    fmt, fs, io,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread::{self, sleep},
    time::{Duration, Instant},
};

use bio_tools::{install::Installer as ToolInstaller, tool_definitions::Tool as InstallableTool};

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
    AlphaFold3,
    OpenDde,
    Boltz2,
    Chai1,
    Protenix,
    EsmFold2,
    ImmuneBuilder,
    HighFold,
    BoltzGen,
    BindCraft,
    IgBlast,
    BioPhi,
    AntiFold,
    ProteinMpnn,
    LigandMpnn,
    ProteinMpnnDdg,
    RfDiffusion,
    RfAntibody,
    Germinal,
    Mber,
    IgDesign,
    ThermoMpnn,
    Genie3,
    DeepSp,
    DeepImmuno,
    TlImmuno2,
    NetSolP,
    DeepStabP,
    AggreScan3d,
    DlkCat,
    CatPred,
    Anarcii,
    Tap,
    Placer,
    Gromacs,
    Orca,
    Gemmi,
}

impl Tool {
    /// Every tool, in the order the status panel lists them: prediction and design first, then the
    /// simulation and file-format helpers.
    pub const ALL: [Self; 37] = [
        Self::AlphaFold3,
        Self::OpenDde,
        Self::Boltz2,
        Self::Chai1,
        Self::Protenix,
        Self::EsmFold2,
        Self::ImmuneBuilder,
        Self::HighFold,
        Self::BoltzGen,
        Self::BindCraft,
        Self::LigandMpnn,
        Self::ProteinMpnn,
        Self::ProteinMpnnDdg,
        Self::RfDiffusion,
        Self::RfAntibody,
        Self::Germinal,
        Self::Mber,
        Self::IgDesign,
        Self::ThermoMpnn,
        Self::Genie3,
        Self::DeepSp,
        Self::DeepImmuno,
        Self::TlImmuno2,
        Self::NetSolP,
        Self::DeepStabP,
        Self::AggreScan3d,
        Self::DlkCat,
        Self::CatPred,
        Self::IgBlast,
        Self::BioPhi,
        Self::AntiFold,
        Self::Anarcii,
        Self::Tap,
        Self::Placer,
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

    /// The tools Molchanica can install through the shared `bio_tools` recipes.
    pub fn managed() -> impl Iterator<Item = Self> {
        Self::ALL
            .into_iter()
            .filter(|t| t.spec().molchanica_managed)
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
    /// A console script (`opendde`, `boltz`, ...) inside a Molchanica-managed uv environment.
    VenvScript,
    /// The interpreter of a Molchanica-managed uv environment. Used where the tool's code is
    /// a checkout rather than a package with an entry point, and where driving the Python API
    /// directly is more robust than depending on a CLI's argument shape.
    VenvPython,
}

/// Operating systems on which an upstream tool can run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformSupport {
    All,
    LinuxOnly,
}

impl PlatformSupport {
    pub fn is_supported(self) -> bool {
        match self {
            Self::All => true,
            Self::LinuxOnly => cfg!(target_os = "linux"),
        }
    }

    pub fn label(self) -> Option<&'static str> {
        match self {
            Self::All => None,
            Self::LinuxOnly => Some("Linux only"),
        }
    }
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
    /// The machine-readable identifier used by the registry, managed environment layout, and the
    /// corresponding `bio_tools::install::Tool` recipe.
    pub slug: &'static str,
    /// One line, shown in the status panel.
    pub summary: &'static str,
    pub url: &'static str,
    /// Licence terms of the whole stack a run needs, not just the upstream repository's label.
    pub license: &'static str,
    /// Upstream platform support, independent of whether Molchanica can install the tool.
    pub platform: PlatformSupport,
    pub kind: ToolKind,
    /// Base name of the console script or binary, without any platform suffix.
    pub executable: &'static str,
    /// An absolute path here overrides all discovery. Always `MOLCHANICA_<TOOL>_EXECUTABLE`.
    pub exe_override_env: &'static str,
    /// Points at the tool's bundle or virtual-environment root, overriding the managed location.
    pub root_override_env: Option<&'static str>,
    /// Independently overrides a checkout or binary bundle for Python tools that need both a
    /// virtual environment and repository assets.
    pub bundle_root_override_env: Option<&'static str>,
    /// Subdirectory of `<data root>/process_executables` holding a binary distribution or checkout.
    /// Self-contained distributions are unpacked whole: IgBLAST resolves `internal_data/` relative
    /// to itself, so the binary sits below the bundle root rather than at it.
    pub bundle_subdir: Option<&'static str>,
    /// Also look beside the Molchanica executable. For tools we may ship in the release zip.
    pub colocated: bool,
    /// Weights and data files, relative to the bundle directory.
    pub required_assets: &'static [RequiredAsset],
    /// Whether Molchanica can install it through `bio_tools`. False for tools with a licence gate
    /// or vendor-managed installation (ORCA), which the user has to obtain themselves.
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
    /// `<data root>/process_executables/python_envs/<slug>`, or the override value.
    pub fn venv_root(&self) -> Option<PathBuf> {
        if let Some(name) = self.root_override_env
            && let Some(configured) = env::var_os(name)
        {
            return Some(PathBuf::from(configured));
        }
        data_root().map(|root| {
            root.join("process_executables")
                .join("python_envs")
                .join(self.slug)
        })
    }

    /// `<data root>/process_executables/<bundle_subdir>`, or the override value.
    pub fn bundle_root(&self) -> Option<PathBuf> {
        if let Some(name) = self.bundle_root_override_env.or(self.root_override_env)
            && let Some(configured) = env::var_os(name)
        {
            return Some(PathBuf::from(configured));
        }
        let subdir = self.bundle_subdir?;
        data_root().map(|root| root.join("process_executables").join(subdir))
    }

    /// User-facing instruction for installing this tool.
    pub fn install_command(&self) -> String {
        if !self.molchanica_managed {
            return self.install_hint.to_owned();
        }
        if !self.platform.is_supported() {
            return format!("{} is available on Linux only.", self.name);
        }
        format!(
            "Install {} from Molchanica's Tools panel (recipe `{}`).",
            self.name, self.slug
        )
    }

    pub fn can_install_here(&self) -> bool {
        self.molchanica_managed && self.platform.is_supported()
    }
}

/// Install one managed optional tool with the shared `bio_tools` Rust installer.
///
/// This blocks and is therefore intended to be called from a worker thread.
pub fn install(tool: Tool) -> io::Result<()> {
    let spec = tool.spec();
    if !spec.platform.is_supported() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("{} is available on Linux only", spec.name),
        ));
    }

    if !spec.molchanica_managed {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{} cannot be installed automatically. {}",
                spec.name, spec.install_hint
            ),
        ));
    }

    let recipe = spec
        .slug
        .parse::<InstallableTool>()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    let root = data_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "unable to determine Molchanica's data directory",
        )
    })?;
    let mut installer = ToolInstaller::for_process_executables(root.join("process_executables"))
        .map_err(|error| io::Error::other(error.to_string()))?;
    installer.config.support_root = installer_support_root();
    installer
        .install(recipe)
        .map_err(|error| io::Error::other(error.to_string()))
}

fn installer_support_root() -> Option<PathBuf> {
    let checkout = env::current_dir().ok();
    let release = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));
    checkout.into_iter().chain(release).find(|root| {
        root.join("scripts/convert_mpnn_weights.py").is_file()
            || root.join("convert_mpnn_weights.py").is_file()
    })
}

/// Estimate the space occupied by this tool inside Molchanica's managed data directory.
///
/// `None` means there are no managed files for the tool. This deliberately excludes upstream
/// caches outside Molchanica's data root, and never follows symlinks while measuring.
pub fn managed_disk_usage(tool: Tool) -> io::Result<Option<u64>> {
    let spec = tool.spec();
    if !spec.molchanica_managed {
        return Ok(None);
    }

    let data_root = data_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "unable to determine Molchanica's data directory",
        )
    })?;
    let mut found = false;
    let mut bytes = 0_u64;
    for path in managed_install_roots(tool, &data_root) {
        if path.exists() {
            found = true;
            bytes = bytes.saturating_add(path_disk_usage(&path)?);
        }
    }

    Ok(found.then_some(bytes))
}

/// Remove the files Molchanica installed for one managed optional tool.
///
/// Override paths and system installations are intentionally never touched. Every deletion target
/// is reconstructed beneath [`data_root`] rather than taken from an environment variable.
pub fn uninstall(tool: Tool) -> io::Result<()> {
    let spec = tool.spec();
    if !spec.molchanica_managed {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{} is managed outside Molchanica and cannot be uninstalled here",
                spec.name
            ),
        ));
    }

    let data_root = data_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "unable to determine Molchanica's data directory",
        )
    })?;
    let roots = managed_install_roots(tool, &data_root);
    if roots
        .iter()
        .any(|path| path == &data_root || !path.starts_with(&data_root))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("refusing to remove an unsafe path for {}", spec.name),
        ));
    }

    let existing: Vec<_> = roots.into_iter().filter(|path| path.exists()).collect();
    if existing.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no Molchanica-managed files were found for {}", spec.name),
        ));
    }

    for path in existing {
        let metadata = fs::symlink_metadata(&path)?;
        let result = if metadata.is_dir() && !metadata.file_type().is_symlink() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
        result.map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("unable to remove {}: {error}", path.display()),
            )
        })?;
    }
    Ok(())
}

fn managed_install_roots(tool: Tool, data_root: &Path) -> Vec<PathBuf> {
    let spec = tool.spec();
    let mut roots = Vec::new();
    if matches!(spec.kind, ToolKind::VenvScript | ToolKind::VenvPython) {
        roots.push(
            data_root
                .join("process_executables/python_envs")
                .join(spec.slug),
        );
    }
    // AlphaFold 3's checkout and model parameters are operator-provided licensed assets. The
    // shared installer manages only its interpreter, so never count or remove the checkout as
    // Molchanica-managed data.
    if tool != Tool::AlphaFold3
        && let Some(subdir) = spec.bundle_subdir
    {
        roots.push(data_root.join("process_executables").join(subdir));
    }
    roots
}

fn path_disk_usage(path: &Path) -> io::Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        return Ok(metadata.len());
    }

    let mut bytes = 0_u64;
    for entry in fs::read_dir(path)? {
        bytes = bytes.saturating_add(path_disk_usage(&entry?.path())?);
    }
    Ok(bytes)
}

/// The single description of every tool. See the module docs for how it is used.
macro_rules! venv_script_tool {
    (
        $tool:ident, $name:literal, $slug:literal, $summary:literal, $url:literal,
        $license:literal, $platform:expr, $executable:literal, $override:literal,
        $managed:literal, $hint:literal
    ) => {
        ToolSpec {
            tool: Tool::$tool,
            name: $name,
            slug: $slug,
            summary: $summary,
            url: $url,
            license: $license,
            platform: $platform,
            kind: ToolKind::VenvScript,
            executable: $executable,
            exe_override_env: $override,
            root_override_env: None,
            bundle_root_override_env: None,
            bundle_subdir: None,
            colocated: false,
            required_assets: &[],
            molchanica_managed: $managed,
            install_hint: $hint,
            version_args: &["--help"],
            version_marker: $executable,
            slow_probe: true,
        }
    };
}

macro_rules! venv_python_tool {
    (
        $tool:ident, $name:literal, $slug:literal, $summary:literal, $url:literal,
        $license:literal, $platform:expr, $override:literal, $venv_override:expr,
        $bundle_override:expr, $bundle:literal, $assets:expr, $managed:literal, $hint:literal
    ) => {
        ToolSpec {
            tool: Tool::$tool,
            name: $name,
            slug: $slug,
            summary: $summary,
            url: $url,
            license: $license,
            platform: $platform,
            kind: ToolKind::VenvPython,
            executable: "python",
            exe_override_env: $override,
            root_override_env: $venv_override,
            bundle_root_override_env: $bundle_override,
            bundle_subdir: Some($bundle),
            colocated: false,
            required_assets: $assets,
            molchanica_managed: $managed,
            install_hint: $hint,
            version_args: &["--version"],
            version_marker: "Python 3",
            slow_probe: false,
        }
    };
}

macro_rules! required_asset {
    ($path:literal, $description:literal) => {
        RequiredAsset {
            relative_path: $path,
            description: $description,
        }
    };
}

static REGISTRY: &[ToolSpec] = &[
    venv_python_tool!(
        AlphaFold3,
        "AlphaFold 3",
        "alphafold3",
        "Protein and protein-ligand structure prediction with AlphaFold 3.",
        "https://github.com/google-deepmind/alphafold3",
        "Apache 2.0 code; model parameters are non-commercial and must be obtained from Google.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_ALPHAFOLD3_PYTHON",
        Some("MOLCHANICA_ALPHAFOLD3_VENV_DIR"),
        Some("MOLCHANICA_ALPHAFOLD3_ROOT"),
        "AlphaFold3",
        &[required_asset!(
            "run_alphafold.py",
            "the operator-provided AlphaFold 3 checkout"
        )],
        true,
        "Prepare AlphaFold 3 from Molchanica's Tools panel, then add the licensed checkout and model parameters."
    ),
    venv_script_tool!(
        Chai1,
        "Chai-1",
        "chai1",
        "Protein and protein-ligand complex structure prediction.",
        "https://github.com/chaidiscovery/chai-lab",
        "Apache 2.0 code and weights. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "chai-lab",
        "MOLCHANICA_CHAI1_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_script_tool!(
        Protenix,
        "Protenix-v2",
        "protenix",
        "Protein and protein-ligand complex structure prediction.",
        "https://github.com/bytedance/Protenix",
        "Apache 2.0. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "protenix",
        "MOLCHANICA_PROTENIX_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_script_tool!(
        EsmFold2,
        "ESMFold 2",
        "esmfold2",
        "Fast single-sequence protein structure prediction.",
        "https://github.com/facebookresearch/esm",
        "MIT fair-esm and weights; Apache 2.0 OpenFold dependency.",
        PlatformSupport::LinuxOnly,
        "esm-fold",
        "MOLCHANICA_ESMFOLD2_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_script_tool!(
        ImmuneBuilder,
        "ImmuneBuilder",
        "immunebuilder",
        "Fast antibody, nanobody, and TCR structure prediction.",
        "https://github.com/oxpig/ImmuneBuilder",
        "BSD 3-Clause; OpenMM refinement is MIT/LGPL. Commercial use permitted.",
        PlatformSupport::All,
        "ABodyBuilder2",
        "MOLCHANICA_IMMUNEBUILDER_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        HighFold,
        "HighFold",
        "highfold",
        "Cyclic-peptide and cyclic-peptide complex prediction.",
        "https://github.com/hongliangduan/HighFold",
        "MIT code over CC BY 4.0 AlphaFold 2 parameters.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_HIGHFOLD_PYTHON",
        Some("MOLCHANICA_HIGHFOLD_VENV_DIR"),
        Some("MOLCHANICA_HIGHFOLD_ROOT"),
        "HighFold",
        &[required_asset!("README.md", "the HighFold checkout")],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_script_tool!(
        BoltzGen,
        "BoltzGen",
        "boltzgen",
        "Protein, peptide, nanobody, and antibody binder design.",
        "https://github.com/HannesStark/boltzgen",
        "MIT code, weights, and training data. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "boltzgen",
        "MOLCHANICA_BOLTZGEN_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        BindCraft,
        "BindCraft",
        "bindcraft",
        "De novo protein and peptide binder design.",
        "https://github.com/martinpacesa/BindCraft",
        "MIT code; required PyRosetta is non-commercial unless separately licensed.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_BINDCRAFT_PYTHON",
        Some("MOLCHANICA_BINDCRAFT_VENV_DIR"),
        Some("MOLCHANICA_BINDCRAFT_ROOT"),
        "BindCraft",
        &[
            required_asset!("bindcraft.py", "the BindCraft runner"),
            required_asset!("params/params_model_5_ptm.npz", "AlphaFold 2 parameters")
        ],
        false,
        "Install BindCraft with its upstream Conda installer, then set the Python and root overrides."
    ),
    venv_script_tool!(
        BioPhi,
        "BioPhi",
        "biophi",
        "Antibody humanization and humanness estimation.",
        "https://github.com/Merck/BioPhi",
        "MIT. Commercial use permitted.",
        PlatformSupport::All,
        "biophi",
        "MOLCHANICA_BIOPHI_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        AntiFold,
        "AntiFold",
        "antifold",
        "Score and sample antibody sequences from a structure.",
        "https://github.com/oxpig/AntiFold",
        "BSD 3-Clause. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_ANTIFOLD_PYTHON",
        Some("MOLCHANICA_ANTIFOLD_VENV_DIR"),
        Some("MOLCHANICA_ANTIFOLD_ROOT"),
        "AntiFold",
        &[required_asset!("antifold/main.py", "the AntiFold runner")],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_script_tool!(
        ProteinMpnnDdg,
        "ProteinMPNN-ddG",
        "proteinmpnn-ddg",
        "Predict stability effects for every point mutation in a protein chain.",
        "https://github.com/PeptoneLtd/proteinmpnn_ddg",
        "MIT. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "proteinmpnn-ddg",
        "MOLCHANICA_PROTEINMPNN_DDG_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        RfDiffusion,
        "RFdiffusion",
        "rfdiffusion",
        "Protein backbone and target-conditioned binder design.",
        "https://github.com/RosettaCommons/RFdiffusion",
        "University of Washington BSD-style licence; commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_RFDIFFUSION_PYTHON",
        Some("MOLCHANICA_RFDIFFUSION_VENV_DIR"),
        Some("MOLCHANICA_RFDIFFUSION_ROOT"),
        "RFdiffusion",
        &[
            required_asset!("scripts/run_inference.py", "the RFdiffusion runner"),
            required_asset!("models/Base_ckpt.pt", "the base RFdiffusion weights")
        ],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        RfAntibody,
        "RFantibody",
        "rfantibody",
        "Antibody and nanobody backbone design against a target.",
        "https://github.com/RosettaCommons/RFantibody",
        "MIT. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_RFANTIBODY_PYTHON",
        Some("MOLCHANICA_RFANTIBODY_VENV_DIR"),
        Some("MOLCHANICA_RFANTIBODY_ROOT"),
        "RFantibody",
        &[required_asset!("weights/RF2_ab.pt", "the RF2-Ab weights")],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        Germinal,
        "Germinal",
        "germinal",
        "De novo antibody design against a selected epitope.",
        "https://github.com/SantiagoMille/germinal",
        "MIT code; required PyRosetta and IgLM weights restrict commercial use.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_GERMINAL_PYTHON",
        Some("MOLCHANICA_GERMINAL_VENV_DIR"),
        Some("MOLCHANICA_GERMINAL_ROOT"),
        "germinal",
        &[required_asset!("run_germinal.py", "the Germinal runner")],
        false,
        "Install Germinal and PyRosetta with its upstream Conda installer, then set the overrides."
    ),
    venv_script_tool!(
        Mber,
        "mBER",
        "mber",
        "VHH binder design with AlphaFold-Multimer backpropagation.",
        "https://github.com/manifoldbio/mber-open",
        "MIT over Apache 2.0 ColabDesign and CC BY 4.0 AlphaFold 2 parameters.",
        PlatformSupport::LinuxOnly,
        "mber-vhh",
        "MOLCHANICA_MBER_EXECUTABLE",
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        IgDesign,
        "IgDesign",
        "igdesign",
        "Antibody CDR design by inverse folding.",
        "https://github.com/AbSciBio/igdesign",
        "Research release; confirm upstream terms before commercial use.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_IGDESIGN_PYTHON",
        Some("MOLCHANICA_IGDESIGN_VENV_DIR"),
        Some("MOLCHANICA_IGDESIGN_ROOT"),
        "igdesign",
        &[required_asset!("predict.py", "the IgDesign runner")],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        ThermoMpnn,
        "ThermoMPNN",
        "thermompnn",
        "Protein mutation stability prediction.",
        "https://github.com/Kuhlman-Lab/ThermoMPNN",
        "MIT. Commercial use permitted.",
        PlatformSupport::All,
        "MOLCHANICA_THERMOMPNN_PYTHON",
        Some("MOLCHANICA_THERMOMPNN_VENV_DIR"),
        Some("MOLCHANICA_THERMOMPNN_ROOT"),
        "ThermoMPNN",
        &[required_asset!(
            "analysis/custom_inference.py",
            "the ThermoMPNN runner"
        )],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        Genie3,
        "Genie 3",
        "genie3",
        "All-atom target-conditioned protein binder generation.",
        "https://github.com/aqlaboratory/genie3",
        "Apache 2.0. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_GENIE3_PYTHON",
        Some("MOLCHANICA_GENIE3_VENV_DIR"),
        Some("MOLCHANICA_GENIE3_ROOT"),
        "genie3",
        &[required_asset!(
            "pretrained",
            "the Genie 3 pretrained weights"
        )],
        false,
        "Install Genie 3 with its upstream Conda setup, then set the Python and root overrides."
    ),
    venv_python_tool!(
        DeepSp,
        "DeepSP",
        "deepsp",
        "Sequence-only antibody developability descriptors.",
        "https://github.com/Lailabcode/DeepSP",
        "MIT. Commercial use permitted.",
        PlatformSupport::All,
        "MOLCHANICA_DEEPSP_PYTHON",
        Some("MOLCHANICA_DEEPSP_VENV_DIR"),
        Some("MOLCHANICA_DEEPSP_ROOT"),
        "DeepSP",
        &[required_asset!(
            "DeepSP_predictor.ipynb",
            "the DeepSP model checkout"
        )],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        DeepImmuno,
        "DeepImmuno",
        "deepimmuno",
        "Peptide-MHC-I immunogenicity prediction.",
        "https://github.com/frankligy/DeepImmuno",
        "MIT. Commercial use permitted.",
        PlatformSupport::All,
        "MOLCHANICA_DEEPIMMUNO_PYTHON",
        Some("MOLCHANICA_DEEPIMMUNO_VENV_DIR"),
        Some("MOLCHANICA_DEEPIMMUNO_ROOT"),
        "DeepImmuno",
        &[required_asset!(
            "deepimmuno-cnn.py",
            "the DeepImmuno runner"
        )],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        TlImmuno2,
        "TLimmuno2",
        "tlimmuno2",
        "Peptide-MHC-II immunogenicity prediction.",
        "https://github.com/XSLiuLab/TLimmuno2",
        "Academic release; confirm upstream terms before commercial use.",
        PlatformSupport::All,
        "MOLCHANICA_TLIMMUNO2_PYTHON",
        Some("MOLCHANICA_TLIMMUNO2_VENV_DIR"),
        Some("MOLCHANICA_TLIMMUNO2_ROOT"),
        "TLimmuno2",
        &[required_asset!(
            "Python/TLimmuno2.py",
            "the TLimmuno2 runner"
        )],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        NetSolP,
        "NetSolP",
        "netsolp",
        "Protein solubility and expression-usability prediction.",
        "https://github.com/tvinet/NetSolP-1.0",
        "Academic use only; commercial use requires an agreement with DTU Health Tech.",
        PlatformSupport::All,
        "MOLCHANICA_NETSOLP_PYTHON",
        Some("MOLCHANICA_NETSOLP_VENV_DIR"),
        Some("MOLCHANICA_NETSOLP_ROOT"),
        "NetSolP-1.0",
        &[
            required_asset!("PredictionServer/predict.py", "the NetSolP runner"),
            required_asset!(
                "PredictionServer/models",
                "licensed NetSolP model checkpoints"
            )
        ],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        DeepStabP,
        "DeepSTABp",
        "deepstabp",
        "Protein melting-temperature prediction.",
        "https://github.com/CSBiology/deepStabP",
        "MIT. Commercial use permitted.",
        PlatformSupport::All,
        "MOLCHANICA_DEEPSTABP_PYTHON",
        Some("MOLCHANICA_DEEPSTABP_VENV_DIR"),
        Some("MOLCHANICA_DEEPSTABP_ROOT"),
        "deepStabP",
        &[required_asset!(
            "src/Api/app",
            "the DeepSTABp prediction service"
        )],
        true,
        "Install from Molchanica's Tools panel."
    ),
    ToolSpec {
        tool: Tool::AggreScan3d,
        name: "AggreScan3D",
        slug: "aggrescan3d",
        summary: "Map and score aggregation-prone regions on a protein structure.",
        url: "https://bitbucket.org/lcbio/aggrescan3d",
        license: "Free for academic use; commercial use requires a separate agreement.",
        platform: PlatformSupport::LinuxOnly,
        kind: ToolKind::VenvScript,
        executable: "aggrescan",
        exe_override_env: "MOLCHANICA_AGGRESCAN3D_EXECUTABLE",
        root_override_env: Some("MOLCHANICA_AGGRESCAN3D_VENV_DIR"),
        bundle_root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: "Install from Molchanica's Tools panel.",
        version_args: &["--help"],
        version_marker: "aggrescan",
        slow_probe: true,
    },
    venv_python_tool!(
        DlkCat,
        "DLKcat",
        "dlkcat",
        "Enzyme turnover prediction from sequence and substrate.",
        "https://github.com/SysBioChalmers/DLKcat",
        "MIT. Commercial use permitted.",
        PlatformSupport::All,
        "MOLCHANICA_DLKCAT_PYTHON",
        Some("MOLCHANICA_DLKCAT_VENV_DIR"),
        Some("MOLCHANICA_DLKCAT_ROOT"),
        "DLKcat",
        &[required_asset!(
            "DeeplearningApproach/Code/example/prediction_for_input.py",
            "the DLKcat prediction runner"
        )],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        CatPred,
        "CatPred",
        "catpred",
        "Enzyme kcat, Km, and Ki prediction with uncertainty.",
        "https://github.com/maranasgroup/CatPred",
        "MIT. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_CATPRED_PYTHON",
        Some("MOLCHANICA_CATPRED_VENV_DIR"),
        Some("MOLCHANICA_CATPRED_ROOT"),
        "CatPred",
        &[
            required_asset!("predict.py", "the CatPred runner"),
            required_asset!("checkpoint_links/kcat", "the CatPred kcat checkpoints"),
        ],
        true,
        "Install from Molchanica's Tools panel."
    ),
    venv_python_tool!(
        Tap,
        "TAP",
        "tap",
        "Therapeutic Antibody Profiler developability assessment.",
        "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap/",
        "Academic use only; obtain TAP directly from OPIG.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_TAP_PYTHON",
        Some("MOLCHANICA_TAP_VENV_DIR"),
        Some("MOLCHANICA_TAP_ROOT"),
        "TAP",
        &[required_asset!(
            "README.md",
            "the operator-provided TAP distribution"
        )],
        false,
        "Request TAP from OPIG and set MOLCHANICA_TAP_PYTHON and MOLCHANICA_TAP_ROOT."
    ),
    venv_python_tool!(
        Placer,
        "PLACER",
        "placer",
        "Protein-ligand pose and side-chain conformation generation.",
        "https://github.com/baker-laboratory/PLACER",
        "BSD 3-Clause. Commercial use permitted.",
        PlatformSupport::LinuxOnly,
        "MOLCHANICA_PLACER_PYTHON",
        Some("MOLCHANICA_PLACER_VENV_DIR"),
        Some("MOLCHANICA_PLACER_ROOT"),
        "PLACER",
        &[
            required_asset!("run_PLACER.py", "the PLACER runner"),
            required_asset!("weights", "the PLACER model weights")
        ],
        true,
        "Install from Molchanica's Tools panel."
    ),
    ToolSpec {
        tool: Tool::OpenDde,
        name: "OpenDDE",
        slug: "opendde",
        summary: "All-atom co-folding: proteins, DNA/RNA, ligands, ions, and complexes.",
        url: "https://github.com/aurekaresearch/OpenDDE",
        license: "Apache 2.0. Commercial use permitted.",
        platform: PlatformSupport::All,
        kind: ToolKind::VenvScript,
        executable: "opendde",
        exe_override_env: "MOLCHANICA_OPENDDE_EXECUTABLE",
        // Predates this registry and is documented, so it keeps its own name rather than becoming
        // MOLCHANICA_OPENDDE_ROOT; existing installs and shell profiles continue to work.
        bundle_root_override_env: None,
        root_override_env: Some("OPENDDE_VENV_DIR"),
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: "Install from Molchanica's Tools panel.",
        version_args: &["--version"],
        version_marker: "opendde",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::Boltz2,
        name: "Boltz-2",
        slug: "boltz2",
        summary: "Co-folding with binding-affinity prediction.",
        platform: PlatformSupport::All,
        url: "https://github.com/jwohlwend/boltz",
        license: "MIT. Commercial use permitted.",
        kind: ToolKind::VenvScript,
        bundle_root_override_env: None,
        executable: "boltz",
        exe_override_env: "MOLCHANICA_BOLTZ_EXECUTABLE",
        root_override_env: Some("MOLCHANICA_BOLTZ_VENV_DIR"),
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: "Install from Molchanica's Tools panel.",
        // Boltz has no --version; `boltz --help` exits 0 and lists its `predict` subcommand.
        version_args: &["--help"],
        version_marker: "predict",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::LigandMpnn,
        name: "LigandMPNN",
        platform: PlatformSupport::All,
        slug: "ligandmpnn",
        summary: "Inverse folding: design sequences for a backbone, in ligand and nucleic-acid context.",
        url: "https://github.com/dauparas/LigandMPNN",
        bundle_root_override_env: Some("MOLCHANICA_LIGANDMPNN_ROOT"),
        license: "MIT. Commercial use permitted.",
        kind: ToolKind::VenvPython,
        executable: "python",
        exe_override_env: "MOLCHANICA_LIGANDMPNN_PYTHON",
        root_override_env: Some("MOLCHANICA_LIGANDMPNN_VENV_DIR"),
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
        install_hint: "Install from Molchanica's Tools panel.",
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    ToolSpec {
        platform: PlatformSupport::All,
        tool: Tool::ProteinMpnn,
        name: "ProteinMPNN",
        slug: "proteinmpnn",
        bundle_root_override_env: Some("MOLCHANICA_PROTEINMPNN_ROOT"),
        summary: "Inverse folding, plus the antibody-finetuned AbMPNN weights.",
        url: "https://github.com/dauparas/ProteinMPNN",
        license: "MIT code; AbMPNN weights CC BY 4.0. Commercial use permitted with attribution.",
        kind: ToolKind::VenvPython,
        executable: "python",
        exe_override_env: "MOLCHANICA_PROTEINMPNN_PYTHON",
        root_override_env: Some("MOLCHANICA_PROTEINMPNN_VENV_DIR"),
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
        install_hint: "Install from Molchanica's Tools panel.",
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    ToolSpec {
        tool: Tool::IgBlast,
        bundle_root_override_env: None,
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
        install_hint: "Install from Molchanica's Tools panel.",
        // `igblastn -version` prints "igblastn: 1.22.0" plus the BLAST package line.
        version_args: &["-version"],
        platform: PlatformSupport::All,
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
        platform: PlatformSupport::All,
        install_hint: "Install from Molchanica's Tools panel.",
        version_args: &["--version"],
        version_marker: "Python 3",
        bundle_root_override_env: None,
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
        platform: PlatformSupport::All,
        molchanica_managed: false,
        install_hint: "Install GROMACS from https://www.gromacs.org/ and put `gmx` on PATH, \
                       or set MOLCHANICA_GROMACS_EXECUTABLE.",
        bundle_root_override_env: None,
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
        platform: PlatformSupport::All,
        molchanica_managed: false,
        install_hint: "Register at https://www.faccts.de/orca/ and put `orca` on PATH, \
                       or set MOLCHANICA_ORCA_EXECUTABLE.",
        bundle_root_override_env: None,
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
        platform: PlatformSupport::All,
        license: "MPL 2.0. Commercial use permitted.",
        kind: ToolKind::Executable,
        executable: "gemmi",
        exe_override_env: "MOLCHANICA_GEMMI_EXECUTABLE",
        bundle_root_override_env: None,
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
/// Matches the locations the install scripts write to. `MOLCHANICA_DATA_DIR` overrides the
/// platform conventions: `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS,
/// and `$XDG_DATA_HOME` (defaulting to `~/.local/share`) elsewhere.
pub fn data_root() -> Option<PathBuf> {
    if let Some(configured) = env::var_os("MOLCHANICA_DATA_DIR") {
        return Some(PathBuf::from(configured));
    }
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
/// The override environment variable always wins, then the Molchanica-managed location. Native
/// executables may additionally fall back to `PATH`; Python tools never do. A user who ran our
/// installer must get the uv-managed interpreter we built rather than an unrelated `pip install`.
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
            // Deliberately no PATH fallback. Even a console launcher ultimately selects a Python
            // interpreter, and ~/.local/bin may contain either a uv tool or a `pip --user` script.
            return Err(not_found(spec));
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
            if let Some(found) = find_on_path(spec.executable) {
                return Ok(found);
            }
        }
    }

    Err(not_found(spec))
}

/// The directory a bundled tool's data files sit in, which several tools need passed to them
/// explicitly (IgBLAST's `IGDATA`, the MPNN checkouts' model parameters).
pub fn bundle_root(tool: Tool) -> io::Result<PathBuf> {
    tool.spec()
        .bundle_root()
        .filter(|root| root.is_dir())
        .ok_or_else(|| {
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

/// Resolve the interpreter in a named Molchanica uv environment.
///
/// Used when an adapter needs the environment's interpreter in addition to its console script.
/// Keeping this lookup here prevents model integrations from reintroducing a bare `python` call.
pub(crate) fn uv_managed_python(slug: &str, override_env: &str) -> io::Result<PathBuf> {
    if let Some(configured) = env::var_os(override_env) {
        let configured = PathBuf::from(configured);
        if configured.is_file() {
            return Ok(configured);
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{override_env} points to {}, but that file does not exist",
                configured.display()
            ),
        ));
    }

    let root = data_root()
        .ok_or_else(|| io::Error::other("unable to determine Molchanica's data directory"))?
        .join("process_executables/python_envs")
        .join(slug);
    executable_in(&venv_bin(&root), "python").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "the uv-managed Python environment for {slug} was not found at {}",
                root.display()
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

/// One incremental result from probing the external-tool registry.
///
/// Status is sent before disk usage is calculated so the UI can update a row as soon as its
/// subprocess finishes, even when walking the tool's managed installation takes longer.
pub enum ToolCheckUpdate {
    Status(ToolStatus),
    ManagedDiskUsage {
        tool: Tool,
        result: io::Result<Option<u64>>,
    },
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

    if !output
        .to_lowercase()
        .contains(&spec.version_marker.to_lowercase())
    {
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

/// Probe every tool concurrently, streaming each result as soon as it is available.
///
/// Concurrent because a serial pass costs the sum of every Python interpreter's Torch import.
/// The channel disconnects after every status and applicable managed-disk measurement has been
/// sent, following the same non-blocking receiver pattern used by the rest of the UI workers.
pub fn check_all() -> Receiver<ToolCheckUpdate> {
    use rayon::prelude::*;

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        Tool::ALL.par_iter().for_each_with(tx, |tx, tool| {
            let started = Instant::now();
            let status = check(*tool);
            println!(
                "External tool check for {} took {:.3}ms",
                tool.spec().name,
                started.elapsed().as_millis()
            );

            let measure_disk_usage = status.result != CheckResult::CantFind;
            if tx.send(ToolCheckUpdate::Status(status)).is_err() {
                return;
            }
            if measure_disk_usage {
                let result = managed_disk_usage(*tool);
                let _ = tx.send(ToolCheckUpdate::ManagedDiskUsage {
                    tool: *tool,
                    result,
                });
            }
        });
    });
    rx
}

/// Run a short-lived probe, killing it if it overruns.
///
/// Neither a nonzero exit nor stderr output is treated as failure: many CLIs answer a flag they
/// don't recognize with a usage message and a nonzero status, and we match on the text either way.
fn probe(executable: &Path, args: &[&str], timeout: Duration) -> io::Result<String> {
    let mut command = Command::new(executable);
    scrub_python_environment(&mut command);
    let mut child = command
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
    scrub_python_environment(command);
    let output = command
        .stdin(Stdio::null())
        .env("PYTHONUNBUFFERED", "1")
        .output()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("unable to start {tool_name}: {error}"),
            )
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

/// Run a subprocess while mirroring its two output streams to Molchanica's corresponding Rust
/// streams. A descriptive banner and shell-free command rendering make GUI-launched model runs
/// diagnosable from the terminal without merging stderr into stdout.
pub fn run_to_completion_logged(
    command: &mut Command,
    tool_name: &str,
    workflow: &str,
) -> io::Result<String> {
    scrub_python_environment(command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONUNBUFFERED", "1");
    let _banner = ToolRunBanner::new(workflow, tool_name, command);

    let mut child = command.spawn().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("unable to start {tool_name}: {error}"),
        )
    })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other(format!("unable to capture {tool_name} stdout")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other(format!("unable to capture {tool_name} stderr")))?;

    let stdout_reader = thread::spawn(move || read_and_forward(stdout, false));
    let stderr_reader = thread::spawn(move || read_and_forward(stderr, true));
    let status = child.wait()?;
    let stdout = join_forward_reader(stdout_reader, tool_name, "stdout")?;
    let stderr = join_forward_reader(stderr_reader, tool_name, "stderr")?;

    if status.success() {
        return Ok(String::from_utf8_lossy(&stdout).into_owned());
    }

    let stderr = String::from_utf8_lossy(&stderr);
    let stdout = String::from_utf8_lossy(&stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    Err(io::Error::other(format!(
        "{tool_name} exited with {status}: {}",
        crate::util::truncate_str(detail, 4_096)
    )))
}

fn read_and_forward(mut pipe: impl Read, stderr: bool) -> io::Result<Vec<u8>> {
    let mut captured = Vec::new();
    let mut buffer = [0_u8; 4_096];
    loop {
        let count = pipe.read(&mut buffer)?;
        if count == 0 {
            return Ok(captured);
        }
        let bytes = &buffer[..count];
        if stderr {
            let mut output = io::stderr().lock();
            let _ = output.write_all(bytes);
            let _ = output.flush();
        } else {
            let mut output = io::stdout().lock();
            let _ = output.write_all(bytes);
            let _ = output.flush();
        }
        captured.extend_from_slice(bytes);
    }
}

fn join_forward_reader(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    tool_name: &str,
    stream: &str,
) -> io::Result<Vec<u8>> {
    reader.join().map_err(|_| {
        io::Error::other(format!(
            "{tool_name} {stream} reader thread terminated unexpectedly"
        ))
    })?
}

pub(crate) struct ToolRunBanner;

impl ToolRunBanner {
    pub(crate) fn new(workflow: &str, tool_name: &str, command: &Command) -> Self {
        let mut output = io::stdout().lock();
        let _ = writeln!(
            output,
            "\nRunning {workflow} with {tool_name}\n============\nCommand: {}",
            display_command(command)
        );
        if let Some(directory) = command.get_current_dir() {
            let _ = writeln!(output, "Working directory: {}", directory.display());
        }
        let _ = output.flush();
        Self
    }
}

impl Drop for ToolRunBanner {
    fn drop(&mut self) {
        let mut output = io::stdout().lock();
        let _ = writeln!(output, "\n============\n");
        let _ = output.flush();
    }
}

pub(crate) fn display_command(command: &Command) -> String {
    std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(display_command_argument)
        .collect::<Vec<_>>()
        .join(" ")
}

fn display_command_argument(argument: &OsStr) -> String {
    let value = argument.to_string_lossy();
    if value.is_empty() {
        return "\"\"".to_owned();
    }
    if value
        .chars()
        .any(|character| character.is_whitespace() || character == '"')
    {
        format!("\"{}\"", value.replace('"', "\\\""))
    } else {
        value.into_owned()
    }
}

/// Keep activated virtualenvs, Conda, and user Python settings from leaking into a managed tool.
/// Explicit venv interpreters and console-script launchers do not need activation.
pub(crate) fn scrub_python_environment(command: &mut Command) {
    for variable in [
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
    ] {
        command.env_remove(variable);
    }
    command.env("PYTHONNOUSERSITE", "1");
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
            let spec = tool.spec();
            if spec.platform.is_supported() {
                assert!(spec.install_command().contains(spec.slug));
            } else {
                assert!(spec.install_command().contains("Linux only"));
            }
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

    #[test]
    fn unsupported_platforms_never_offer_installation() {
        for tool in Tool::ALL {
            let spec = tool.spec();
            if !spec.platform.is_supported() {
                assert_eq!(spec.platform.label(), Some("Linux only"));
                assert!(!spec.can_install_here());
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn linux_only_install_returns_unsupported() {
        let error = install(Tool::HighFold).expect_err("HighFold is Linux-only");
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn every_managed_tool_has_a_bio_tools_recipe() {
        for tool in Tool::managed() {
            let spec = tool.spec();
            let recipe = spec
                .slug
                .parse::<InstallableTool>()
                .unwrap_or_else(|error| panic!("{}: {error}", spec.slug));
            assert_eq!(recipe.slug(), spec.slug);
        }
    }

    #[test]
    fn managed_install_roots_stay_below_the_data_root() {
        let root = Path::new("managed-data");
        for tool in Tool::managed() {
            let paths = managed_install_roots(tool, root);
            assert!(
                !paths.is_empty(),
                "{} has no managed install path",
                tool.spec().name
            );
            assert!(
                paths
                    .iter()
                    .all(|path| path.starts_with(root) && path != root)
            );
        }
    }

    #[test]
    fn unmanaged_tools_cannot_be_uninstalled_automatically() {
        let error = uninstall(Tool::Orca).expect_err("ORCA must stay vendor-managed");
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn diagnostic_command_rendering_quotes_paths_with_spaces() {
        let mut command = Command::new(r"C:\Program Files\Python\python.exe");
        command.arg("runner.py").arg("two words").arg("plain");
        assert_eq!(
            display_command(&command),
            r#""C:\Program Files\Python\python.exe" runner.py "two words" plain"#
        );
    }

    #[test]
    fn logged_runner_returns_stdout_without_merging_stderr() {
        let mut command = if cfg!(windows) {
            let mut command = Command::new("cmd");
            command.args(["/C", "echo normal & echo warning 1>&2"]);
            command
        } else {
            let mut command = Command::new("sh");
            command.args(["-c", "printf normal; printf warning >&2"]);
            command
        };
        let stdout = run_to_completion_logged(&mut command, "test tool", "test workflow").unwrap();
        assert!(stdout.contains("normal"));
        assert!(!stdout.contains("warning"));
    }
}
