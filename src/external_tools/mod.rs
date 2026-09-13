//! The optional third-party tools Molchanica can drive: finding, installing, probing, and running
//! them.
//!
//! Which tools exist, what they are called, what they do, and what they are licensed under is not
//! decided here: that lives once, in `bio_tools`' shared catalog, which `bio_web` reads too. Each
//! [`registry`] entry names its catalog slug and adds only what is Molchanica's own — where the
//! tool is installed, how it is launched, how it is probed, and how Molchanica drives it.
//! Molchanica drives a subset of the catalog, so a tool being absent from [`Tool`] does not mean
//! `bio_tools` lacks it.
//!
//! Molchanica works without any of these installed; each one unlocks a feature.
//!
//! # Layout
//!
//! Shared by every tool:
//!
//! - [`registry`]: the one table describing every tool. Start here.
//! - [`paths`]: the data root, and resolving a tool to a file ([`find_executable`]).
//! - [`manage`]: [`install`], [`uninstall`], and managed disk usage.
//! - [`status`]: [`check`](status::check) / [`check_all`] for the tools panel, and the cheap
//!   [`is_installed`].
//! - [`process`]: [`ToolWorkspace`], [`run_tool`], and [`RunControl`] — how any tool is run.
//!
//! How a tool is driven is set by its [`ToolAdapter`]:
//!
//! - **Shared adapter** (OpenDDE, Boltz-2, Chai-1, ESMFold 2, ProteinMPNN, LigandMPNN,
//!   RFdiffusion3): inputs, validation, and command lines all belong to `bio_tools`' Python
//!   adapters. [`tool_form`] loads the form contract and presets `bio_tools` publishes, and
//!   [`shared_adapter`] runs a submitted form. One form, runner, and result browser
//!   (`ui::popup::tool_runner`) serves all of them, so there is no per-tool code for these here.
//! - **External** (GROMACS, ORCA, Gemmi): driven from elsewhere in Molchanica; this module only
//!   finds, probes, and reports on them.
//!
//! [`mpnn`] is the one native adapter: it also runs the ProteinMPNN checkout directly, for the
//! Protein design window, which needs designs as typed values to rank rather than files to
//! browse. It is laid out the way any further native adapter should be — request types with
//! `validate`, result types, one blocking entry point that writes inputs into a [`ToolWorkspace`],
//! runs the tool with [`run_tool`], and reads its output files back, then the output parsers.
//!
//! [`pdb_write`] renders a peptide as PDB for the tools that take nothing else.
//!
//! # Where tools live
//!
//! Molchanica-managed installs go under [`data_root`], which is the folder the executable itself
//! lives in. Everything the program writes is in that one folder:
//!
//! ```text
//! <data root>/                     e.g. %LOCALAPPDATA%\Programs\Molchanica, or ~/molchanica
//!     molchanica[.exe]             the program, put here by the setup scripts
//!     molchanica_prefs.mca         the preferences file
//!     managed_molecules/           downloaded and generated molecules
//!     gpu_cache/                   the graphics pipeline cache
//!     process_executables/
//!         python_envs/
//!             opendde/             a uv-managed Python environment per Python-based tool. These
//!             boltz2/              deliberately do not share an interpreter: OpenDDE wants Python
//!             ligandmpnn/          >= 3.11, Boltz-2 wants < 3.13, and ProteinMPNN wants numpy < 2,
//!             proteinmpnn/         whose newest wheel is cp312.
//!             rfd3/
//!         LigandMPNN/              a checkout plus its downloaded model weights
//!         ProteinMPNN/
//!         results/<slug>/          every shared-adapter run, kept: inputs, logs, and outputs
//!         desktop_inputs/          structures written from opened molecules for those runs
//!     datasets/                    PDBbind and similar, when used
//! ```
//!
//! Native tools may still be resolved through `PATH`. Python tools deliberately may not: they run
//! only from the uv environment built for that tool, unless an explicit override names another
//! executable. This prevents a desktop launch from silently selecting a system Python with an
//! incompatible package set.

pub mod manage;
pub mod paths;
pub mod process;
pub mod registry;
pub mod status;

pub mod shared_adapter;
pub mod tool_form;

pub mod mpnn;

pub mod pdb_write;

pub use manage::{install, uninstall};
pub use paths::{
    bundle_root, data_root, find_executable, home_directory, migrate_legacy_data,
    process_executables_dir,
};
pub use process::{RunControl, ToolWorkspace, run_tool};
pub use registry::{Tool, ToolAdapter};
pub use status::{CheckResult, ToolCheckUpdate, ToolStatus, check_all, is_installed};
