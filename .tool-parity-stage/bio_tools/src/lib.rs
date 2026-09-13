//! Install, run, and inspect third-party computational biology and chemistry tools, e.g.
//! AlphaFold, Boltz, RFdiffusion3, and ProteinMPNN.
//!
//! The crate has three parts, usable independently:
//!
//! - [`tool_definitions::catalog`]: static, tool-agnostic metadata (summary, description,
//!   license, official links, cost, category) for every supported tool.
//! - [`install`]: [`install::Installer`] owns per-tool environments and assets under a
//!   caller-chosen root, replacing application-specific shell and PowerShell scripts.
//! - [`run`]: [`run::CommandSpec`] describes a shell-free invocation; [`run::run`] executes it
//!   with bounded capture, a timeout, and an optional on-disk run log.
//!
//! The `cli` feature (on by default) additionally builds the standalone `bio_tools` executable.
//! See the README for worked Rust and Python examples.

use std::{fmt, io, path::Path};

pub mod adapters;
mod input;
pub mod install;
pub mod run;
mod run_log;
pub mod status;
pub mod tool_definitions;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LaunchType {
    PythonLib,
    /// Installable with UV
    PythonBasedApp,
    /// Installable with Micromamba, or in some cases, requires the full Conda.
    CondaBasedApp,
    Executable,
}

impl fmt::Display for LaunchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::PythonLib => "Python library",
            Self::PythonBasedApp => "Python app (uv)",
            Self::CondaBasedApp => "Python app (conda)",
            Self::Executable => "Executable",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolCategory {
    Cheminformatics,
    StructurePrediction,
    ProteinDesign,
    PeptideBinderDesign,
    BackboneGeneration,
    MoleculeDynamics,
    QuantumChemistry,
    AntibodyDesign,
    SequencePrediction,
    SequenceAnalysis,
    PropertyPrediction,
    BindingData,
    Placeholder,
}

impl fmt::Display for ToolCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Cheminformatics => "Cheminformatics",
            Self::StructurePrediction => "Structure prediction",
            Self::ProteinDesign => "Protein Design",
            Self::PeptideBinderDesign => "Binder design",
            Self::BackboneGeneration => "Backbone design",
            Self::MoleculeDynamics => "Molecular simulation",
            Self::QuantumChemistry => "Quantum chemistry",
            Self::AntibodyDesign => "Antibody design",
            Self::SequencePrediction => "Sequence prediction",
            Self::SequenceAnalysis => "Sequence analysis",
            Self::PropertyPrediction => "Property prediction",
            Self::BindingData => "Binding data",
            Self::Placeholder => "Uncategorized",
        };
        write!(f, "{}", s)
    }
}

/// Used to categorize tools by which OS they support.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OperatingSystem {
    Linux,
    Windows,
    Mac,
}

/// This is coarse, and is loosely correlated to expected runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessExpense {
    /// i.e. ms
    Cheap,
    /// i.e. seconds
    Moderate,
    /// I.e. minutes or hours
    Expensive,
}

impl fmt::Display for ProcessExpense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Cheap => "Cheap (ms)",
            Self::Moderate => "Moderate (s)",
            Self::Expensive => "Expensive (min or hours)",
        };
        write!(f, "{}", s)
    }
}

/// The specific license under which a tool (or its weights) is released.
/// Coarser than this is [`LicenseCategory`]; this is what a link to the
/// license's own official text is built from -- see [`License::official_url`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum License {
    Mit,
    ApacheV2,
    Bsd3Clause,
    Lgpl21OrLater,
    PublicDomain,
    /// A non-standard, tool-specific, or otherwise unlisted license: no
    /// fixed official page exists for it, so [`CatalogEntry::license_url`]
    /// carries the link instead, if one is known.
    Other,
}

impl fmt::Display for License {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mit => "MIT",
            Self::ApacheV2 => "Apache-2.0",
            Self::Bsd3Clause => "BSD-3-Clause",
            Self::Lgpl21OrLater => "LGPL-2.1-or-later",
            Self::PublicDomain => "Public domain",
            Self::Other => "Other",
        })
    }
}

impl License {
    pub fn category(self) -> LicenseCategory {
        use License::*;
        match self {
            Mit | ApacheV2 | Bsd3Clause | PublicDomain => LicenseCategory::Permissive,
            Lgpl21OrLater => LicenseCategory::Copyleft,
            Other => LicenseCategory::Proprietary,
        }
    }

    /// The license's own official page, where a single fixed one exists for
    /// every license of this kind. `None` for [`License::Other`]: use
    /// [`CatalogEntry::license_url`] for a tool-specific override instead.
    pub const fn official_url(self) -> Option<&'static str> {
        match self {
            Self::Mit => Some("https://opensource.org/license/mit"),
            Self::ApacheV2 => Some("https://www.apache.org/licenses/LICENSE-2.0"),
            Self::Bsd3Clause => Some("https://opensource.org/license/bsd-3-clause"),
            Self::Lgpl21OrLater => Some("https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html"),
            Self::PublicDomain => None,
            Self::Other => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LicenseData {
    pub license: License,
    /// Only if non-standard, or details like attribution.
    pub details: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LicenseCategory {
    Permissive,
    Copyleft,
    NonCommercial,
    Proprietary,
}

impl fmt::Display for LicenseCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Permissive => "Permissive",
            Self::Copyleft => "Copyleft",
            Self::NonCommercial => "Non-commercial",
            Self::Proprietary => "Proprietary",
        };
        write!(f, "{}", s)
    }
}

/// Tool-specific descriptive data which does not depend on a UI framework or
/// execution environment or on the tool's identity (slug/name), so that the
/// same field list can also back the catalog's `&'static str`-based static
/// data -- see [`tool_definitions::catalog::CatalogEntry::spec`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecData<S> {
    /// Terser than Summary. May also include copy+paste+modify from an tool-official source.
    pub summary: S,
    /// Note: This may be taken directly (Copy + paste, or modified) directly from the tool's official
    /// website, docs, or paper. It may include a degree of marketing copy, but we try to make
    /// it descriptive if that's too heavy.
    pub description: S,
    pub availability: S,
    pub license_details: S,
    pub repo_url: Option<S>,
    pub home_url: Option<S>,
    pub docs_url: Option<S>,
    /// Documentation for the tool's input parameters.
    pub input_params_url: Option<S>,
    /// Official examples of use
    pub examples_url: Option<S>,
    pub paper_url: Option<S>,
    pub license: License,
    /// A tool-specific license page, used when `license` is
    /// [`License::Other`] (or any other variant without a fixed
    /// [`License::official_url`]).
    pub license_url: Option<S>,
    /// Tested, compared inputs and outputs against official docs if available,
    /// and had results evaluated in at least some cases, e.g. official examples.
    // todo: Once there are no more untested tools, remove this field
    pub tested: bool,
}

impl SpecData<&'static str> {
    /// Copy this static catalog data into an owned, runtime-facing form.
    pub fn to_owned_data(&self) -> SpecData<String> {
        SpecData {
            summary: self.summary.to_owned(),
            description: self.description.to_owned(),
            availability: self.availability.to_owned(),
            license_details: self.license_details.to_owned(),
            repo_url: self.repo_url.map(str::to_owned),
            home_url: self.home_url.map(str::to_owned),
            docs_url: self.docs_url.map(str::to_owned),
            input_params_url: self.input_params_url.map(str::to_owned),
            examples_url: self.examples_url.map(str::to_owned),
            paper_url: self.paper_url.map(str::to_owned),
            license: self.license,
            license_url: self.license_url.map(str::to_owned),
            tested: self.tested,
        }
    }
}

/// Tool-specific descriptive data, identified by slug.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Spec {
    pub slug: String,
    pub data: SpecData<String>,
}

impl Spec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slug: impl Into<String>,
        summary: impl Into<String>,
        description: impl Into<String>,
        availability: impl Into<String>,
        license_details: impl Into<String>,
        repo_url: Option<String>,
        home_url: Option<String>,
        docs_url: Option<String>,
        input_params_url: Option<String>,
        examples_url: Option<String>,
        paper_url: Option<String>,
        license: License,
        license_url: Option<String>,
        tested: bool,
    ) -> Self {
        Self {
            slug: slug.into(),
            data: SpecData {
                summary: summary.into(),
                description: description.into(),
                availability: availability.into(),
                license_details: license_details.into(),
                repo_url,
                home_url,
                docs_url,
                input_params_url,
                examples_url,
                paper_url,
                license,
                license_url,
                tested,
            },
        }
    }

    /// Official links in display order.
    pub fn links(&self) -> Vec<(&'static str, &str)> {
        [
            ("Documentation", self.data.docs_url.as_deref()),
            ("Input parameters", self.data.input_params_url.as_deref()),
            ("Examples", self.data.examples_url.as_deref()),
            ("Home page", self.data.home_url.as_deref()),
            ("Paper", self.data.paper_url.as_deref()),
            ("Source code", self.data.repo_url.as_deref()),
            (
                "License",
                self.data
                    .license
                    .official_url()
                    .or(self.data.license_url.as_deref()),
            ),
        ]
        .into_iter()
        .filter_map(|(label, url)| url.map(|url| (label, url)))
        .collect()
    }
}

/// Registry data shared by Rust and Python applications.
///
/// UI field descriptors and an adapter implementation are deliberately owned
/// by the consuming application; this type contains the stable, reusable
/// identity and classification of a tool.
#[derive(Clone, Debug, PartialEq)]
pub struct Process {
    pub name: String,
    pub id: u32,
    pub categories: Vec<ToolCategory>,
    pub launch_type: LaunchType,
    pub license_type: LicenseCategory,
    pub expense: ProcessExpense,
    pub top_choice: bool,
    pub spec: Spec,
}

impl Process {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        id: u32,
        categories: Vec<ToolCategory>,
        launch_type: LaunchType,
        license_type: LicenseCategory,
        expense: ProcessExpense,
        top_choice: bool,
        spec: Spec,
    ) -> Self {
        Self {
            name: name.into(),
            id,
            categories,
            launch_type,
            license_type,
            expense,
            top_choice,
            spec,
        }
    }

    /// Install this process to the caller's managed tools directory.
    pub fn install(&self, tools_path: &Path) -> io::Result<()> {
        install::Installer::from_environment(tools_path)
            .map_err(io::Error::other)?
            .install(self.tool()?)
            .map_err(io::Error::other)
    }

    /// Remove this process's environment and assets from the managed tools directory.
    pub fn uninstall(&self, tools_path: &Path) -> io::Result<install::UninstallReport> {
        install::Installer::from_environment(tools_path)
            .map_err(io::Error::other)?
            .uninstall(self.tool()?)
            .map_err(io::Error::other)
    }

    fn tool(&self) -> io::Result<tool_definitions::Tool> {
        self.spec
            .slug
            .parse::<tool_definitions::Tool>()
            .or_else(|_| self.name.parse::<tool_definitions::Tool>())
            .map_err(io::Error::other)
    }
}
