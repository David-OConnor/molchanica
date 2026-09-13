//! The table of every optional tool: what it is, where it is installed, how it is launched, how it
//! is probed, and how Molchanica drives it. See the [module docs](super) for how it is used.

use std::{env, fmt, path::PathBuf};

use bio_tools::{
    License,
    tool_definitions::{
        Tool as InstallableTool,
        catalog::{self, CatalogEntry},
    },
};

use super::paths::{data_root, managed_venv_dir};

/// One optional third-party tool.
///
/// Adding a variant plus its [`REGISTRY`] entry is all that is needed for it to appear in the
/// status panel and be resolvable through [`find_executable`](super::find_executable).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Tool {
    OpenDde,
    Boltz2,
    Chai1,
    EsmFold2,
    ProteinMpnn,
    LigandMpnn,
    RfDiffusion3,
    Gromacs,
    Orca,
    Gemmi,
}

impl Tool {
    /// Every tool, in the order the status panel lists them: prediction and design first, then the
    /// simulation and file-format helpers.
    pub const ALL: [Self; 10] = [
        Self::OpenDde,
        Self::Boltz2,
        Self::Chai1,
        Self::EsmFold2,
        Self::LigandMpnn,
        Self::ProteinMpnn,
        Self::RfDiffusion3,
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
}

impl fmt::Display for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.spec().name())
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

/// How Molchanica runs a tool once it is installed.
///
/// This is the split that decides which code to read when changing how a tool is used: every
/// [`SharedAdapter`](Self::SharedAdapter) tool goes through the same form, runner, and result
/// browser, with no per-tool code in Molchanica.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolAdapter {
    /// `bio_tools`' Python adapter, fed by the form contract `bio_tools` publishes for the tool's
    /// catalog slug. See [`shared_adapter`](super::shared_adapter).
    SharedAdapter,
    /// Driven from elsewhere in Molchanica (the MD, ORCA, and electron-density code); this module
    /// only finds, probes, and reports on it.
    External,
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

/// Shown if a registry entry ever names a catalog slug `bio_tools` no longer has.
const MISSING_CATALOG: &str = "(the bio_tools catalog entry for this tool is missing)";

/// Where a tool's identity and human-readable description come from.
///
/// `bio_tools` already describes every tool worth describing — name, slug, summary, home page,
/// licence — and `bio_web` reads that same table. Restating any of it here is how the two drifted
/// apart, so the registry points at the shared catalog entry and holds only what is genuinely
/// Molchanica's own: where a tool is installed, how it is launched, and how it is probed.
#[derive(Clone, Copy, Debug)]
pub enum ToolIdentity {
    /// `bio_tools` catalogues the tool under this slug. Note that a catalog slug is not always the
    /// install recipe's slug, so the recipe, where one exists, is what [`ToolSpec::slug`] and
    /// [`ToolSpec::name`] prefer.
    Shared(&'static str),
    /// Molchanica-only: `bio_tools` neither installs nor catalogues it, so it is described here.
    Local {
        slug: &'static str,
        name: &'static str,
        summary: &'static str,
        url: &'static str,
        license: License,
        license_details: &'static str,
    },
}

impl ToolIdentity {
    /// The shared catalog entry, for the tools `bio_tools` describes.
    pub fn catalog(self) -> Option<&'static CatalogEntry> {
        match self {
            Self::Shared(slug) => catalog::by_slug(slug),
            Self::Local { .. } => None,
        }
    }

    /// The `bio_tools` install recipe backing this tool, where one exists. `None` for the tools
    /// only ever obtained from a vendor or a system package manager.
    pub fn recipe(self) -> Option<InstallableTool> {
        self.catalog().and_then(|entry| entry.identity.tool())
    }
}

/// Everything Molchanica knows about one tool.
///
/// Every [`REGISTRY`] entry lists these fields in this order, so that entries can be compared at a
/// glance.
pub struct ToolSpec {
    pub tool: Tool,
    /// Name, slug, summary, home page, and licence, sourced from `bio_tools` wherever it has them.
    pub identity: ToolIdentity,
    /// Upstream platform support, independent of whether Molchanica can install the tool.
    pub platform: PlatformSupport,
    pub kind: ToolKind,
    pub adapter: ToolAdapter,
    /// Base name of the console script or binary, without any platform suffix.
    pub executable: &'static str,
    /// An absolute path here overrides all discovery.
    pub exe_override_env: &'static str,
    /// Points at the tool's bundle or virtual-environment root, overriding the managed location.
    pub root_override_env: Option<&'static str>,
    /// Independently overrides a checkout or binary bundle for Python tools that need both a
    /// virtual environment and repository assets.
    pub bundle_root_override_env: Option<&'static str>,
    /// Subdirectory of `<data root>/process_executables` holding a binary distribution or checkout.
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
    /// The shared `bio_tools` catalog entry, where there is one. See [`ToolIdentity`].
    pub fn catalog(&self) -> Option<&'static CatalogEntry> {
        self.identity.catalog()
    }

    /// The `bio_tools` install recipe backing this tool, where one exists.
    pub fn recipe(&self) -> Option<InstallableTool> {
        self.identity.recipe()
    }

    /// The machine-readable identifier: the managed environment's directory name, and the key the
    /// `bio_tools` installer is asked for. Taken from the install recipe, so that the directory
    /// Molchanica builds and the one `bio_tools` installs into cannot disagree.
    pub fn slug(&self) -> &'static str {
        match self.identity {
            ToolIdentity::Shared(catalog_slug) => {
                self.recipe().map_or(catalog_slug, InstallableTool::slug)
            }
            ToolIdentity::Local { slug, .. } => slug,
        }
    }

    /// The key `bio_tools` publishes this tool's form contract, presets, and Python adapter under,
    /// for [`ToolAdapter::SharedAdapter`] tools. `None` for every other tool.
    pub fn adapter_slug(&self) -> Option<&'static str> {
        match (self.adapter, self.identity) {
            (ToolAdapter::SharedAdapter, ToolIdentity::Shared(catalog_slug)) => Some(catalog_slug),
            _ => None,
        }
    }

    /// How the tool is written wherever a person reads it.
    pub fn name(&self) -> &'static str {
        match self.identity {
            ToolIdentity::Shared(catalog_slug) => match self.recipe() {
                Some(recipe) => recipe.name(),
                None => self.catalog().map_or(catalog_slug, CatalogEntry::name),
            },
            ToolIdentity::Local { name, .. } => name,
        }
    }

    /// A short description, shown in the status panel.
    pub fn summary(&self) -> &'static str {
        match self.identity {
            ToolIdentity::Shared(_) => self
                .catalog()
                .map_or(MISSING_CATALOG, |entry| entry.spec.summary),
            ToolIdentity::Local { summary, .. } => summary,
        }
    }

    /// The page to send someone to for more: the project's home, else its repository, else docs.
    pub fn url(&self) -> &'static str {
        match self.identity {
            ToolIdentity::Shared(_) => self
                .catalog()
                .and_then(|entry| {
                    entry
                        .spec
                        .home_url
                        .or(entry.spec.repo_url)
                        .or(entry.spec.docs_url)
                })
                .unwrap_or(MISSING_CATALOG),
            ToolIdentity::Local { url, .. } => url,
        }
    }

    /// The licence's short label, e.g. `MIT`.
    pub fn license(&self) -> License {
        match self.identity {
            ToolIdentity::Shared(_) => self
                .catalog()
                .map_or(License::Other, |entry| entry.spec.license),
            ToolIdentity::Local { license, .. } => license,
        }
    }

    /// Licence terms of the whole stack a run needs, not just the upstream repository's label.
    pub fn license_details(&self) -> &'static str {
        match self.identity {
            ToolIdentity::Shared(_) => self
                .catalog()
                .map_or(MISSING_CATALOG, |entry| entry.spec.license_details),
            ToolIdentity::Local {
                license_details, ..
            } => license_details,
        }
    }

    /// `<data root>/process_executables/python_envs/<slug>`, or the override value.
    pub fn venv_root(&self) -> Option<PathBuf> {
        if let Some(name) = self.root_override_env
            && let Some(configured) = env::var_os(name)
        {
            return Some(PathBuf::from(configured));
        }
        data_root().map(|root| managed_venv_dir(&root, self.slug()))
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
            return format!("{} is available on Linux only.", self.name());
        }
        format!(
            "Install {} from Molchanica's Tools panel (recipe `{}`).",
            self.name(),
            self.slug()
        )
    }

    pub fn can_install_here(&self) -> bool {
        self.molchanica_managed && self.platform.is_supported()
    }
}

/// The install hint every Molchanica-managed tool shares.
const MANAGED_HINT: &str = "Install from Molchanica's Tools panel.";

/// The single description of every tool. See the module docs for how it is used.
pub static REGISTRY: &[ToolSpec] = &[
    // ---------------------------------------------------------------------------------------------
    // Structure prediction, run through bio_tools' shared adapter
    // ---------------------------------------------------------------------------------------------
    ToolSpec {
        tool: Tool::OpenDde,
        identity: ToolIdentity::Shared("opendde"),
        platform: PlatformSupport::All,
        kind: ToolKind::VenvScript,
        adapter: ToolAdapter::SharedAdapter,
        executable: "opendde",
        exe_override_env: "MOLCHANICA_OPENDDE_EXECUTABLE",
        // Predates this registry and is documented, so it keeps its own name rather than becoming
        // MOLCHANICA_OPENDDE_VENV_DIR; existing installs and shell profiles continue to work.
        root_override_env: Some("OPENDDE_VENV_DIR"),
        bundle_root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: MANAGED_HINT,
        version_args: &["--version"],
        version_marker: "opendde",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::Boltz2,
        identity: ToolIdentity::Shared("boltz2"),
        platform: PlatformSupport::All,
        kind: ToolKind::VenvScript,
        adapter: ToolAdapter::SharedAdapter,
        executable: "boltz",
        exe_override_env: "MOLCHANICA_BOLTZ_EXECUTABLE",
        root_override_env: Some("MOLCHANICA_BOLTZ_VENV_DIR"),
        bundle_root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: MANAGED_HINT,
        // Boltz has no --version; `boltz --help` exits 0 and lists its `predict` subcommand.
        version_args: &["--help"],
        version_marker: "predict",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::Chai1,
        identity: ToolIdentity::Shared("chai1"),
        platform: PlatformSupport::LinuxOnly,
        kind: ToolKind::VenvScript,
        adapter: ToolAdapter::SharedAdapter,
        executable: "chai-lab",
        exe_override_env: "MOLCHANICA_CHAI1_EXECUTABLE",
        root_override_env: None,
        bundle_root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: MANAGED_HINT,
        version_args: &["--help"],
        version_marker: "chai-lab",
        slow_probe: true,
    },
    ToolSpec {
        tool: Tool::EsmFold2,
        identity: ToolIdentity::Shared("esmfold2"),
        platform: PlatformSupport::LinuxOnly,
        kind: ToolKind::VenvScript,
        adapter: ToolAdapter::SharedAdapter,
        executable: "esm-fold",
        exe_override_env: "MOLCHANICA_ESMFOLD2_EXECUTABLE",
        root_override_env: None,
        bundle_root_override_env: None,
        bundle_subdir: None,
        colocated: false,
        required_assets: &[],
        molchanica_managed: true,
        install_hint: MANAGED_HINT,
        version_args: &["--help"],
        version_marker: "esm-fold",
        slow_probe: true,
    },
    // ---------------------------------------------------------------------------------------------
    // Sequence and backbone design, run through bio_tools' shared adapter
    // ---------------------------------------------------------------------------------------------
    ToolSpec {
        tool: Tool::LigandMpnn,
        identity: ToolIdentity::Shared("ligandmpnn"),
        platform: PlatformSupport::All,
        kind: ToolKind::VenvPython,
        adapter: ToolAdapter::SharedAdapter,
        executable: "python",
        exe_override_env: "MOLCHANICA_LIGANDMPNN_PYTHON",
        root_override_env: Some("MOLCHANICA_LIGANDMPNN_VENV_DIR"),
        bundle_root_override_env: Some("MOLCHANICA_LIGANDMPNN_ROOT"),
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
        install_hint: MANAGED_HINT,
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    // Also run natively, by the Protein design window's AbMPNN tab; see `mpnn`.
    ToolSpec {
        tool: Tool::ProteinMpnn,
        identity: ToolIdentity::Shared("proteinmpnn"),
        platform: PlatformSupport::All,
        kind: ToolKind::VenvPython,
        adapter: ToolAdapter::SharedAdapter,
        executable: "python",
        exe_override_env: "MOLCHANICA_PROTEINMPNN_PYTHON",
        root_override_env: Some("MOLCHANICA_PROTEINMPNN_VENV_DIR"),
        bundle_root_override_env: Some("MOLCHANICA_PROTEINMPNN_ROOT"),
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
        install_hint: MANAGED_HINT,
        version_args: &["--version"],
        version_marker: "Python 3",
        slow_probe: false,
    },
    // RFdiffusion3 is the only RFdiffusion generation Molchanica supports: 1 and 2 are gone from
    // `bio_tools`, and their Hydra-override command line has nothing in common with RFD3's JSON
    // `InputSpecification`.
    ToolSpec {
        tool: Tool::RfDiffusion3,
        identity: ToolIdentity::Shared("rfd3"),
        platform: PlatformSupport::LinuxOnly,
        kind: ToolKind::VenvScript,
        adapter: ToolAdapter::SharedAdapter,
        executable: "rfd3",
        exe_override_env: "MOLCHANICA_RFD3_EXECUTABLE",
        root_override_env: Some("MOLCHANICA_RFD3_VENV_DIR"),
        bundle_root_override_env: Some("MOLCHANICA_RFD3_ROOT"),
        bundle_subdir: Some("rfd3"),
        colocated: false,
        required_assets: &[RequiredAsset {
            relative_path: "checkpoints/rfd3_latest.ckpt",
            description: "the RFdiffusion3 checkpoint",
        }],
        molchanica_managed: true,
        install_hint: MANAGED_HINT,
        version_args: &["--help"],
        version_marker: "rfd3",
        slow_probe: true,
    },
    // ---------------------------------------------------------------------------------------------
    // Simulation and file formats, driven from elsewhere in Molchanica
    // ---------------------------------------------------------------------------------------------
    ToolSpec {
        tool: Tool::Gromacs,
        identity: ToolIdentity::Shared("gromacs"),
        platform: PlatformSupport::All,
        kind: ToolKind::Executable,
        adapter: ToolAdapter::External,
        executable: "gmx",
        exe_override_env: "MOLCHANICA_GROMACS_EXECUTABLE",
        root_override_env: None,
        bundle_root_override_env: None,
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
        identity: ToolIdentity::Shared("orca"),
        platform: PlatformSupport::All,
        kind: ToolKind::Executable,
        adapter: ToolAdapter::External,
        executable: "orca",
        exe_override_env: "MOLCHANICA_ORCA_EXECUTABLE",
        root_override_env: None,
        bundle_root_override_env: None,
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
        // The one tool `bio_tools` neither installs nor catalogues: a file-format converter
        // Molchanica reaches for on its own, not a step in a prediction or design pipeline.
        identity: ToolIdentity::Local {
            slug: "gemmi",
            name: "Gemmi",
            summary: "Converts MTZ and unprocessed electron-density files.",
            url: "https://gemmi.readthedocs.io/",
            license: License::Other,
            license_details: "MPL 2.0. Commercial use permitted.",
        },
        platform: PlatformSupport::All,
        kind: ToolKind::Executable,
        adapter: ToolAdapter::External,
        executable: "gemmi",
        exe_override_env: "MOLCHANICA_GEMMI_EXECUTABLE",
        root_override_env: None,
        bundle_root_override_env: None,
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
