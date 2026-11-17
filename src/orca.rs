//! ORCA quantum chemistry: inputs built from parameters, runs in a kept results folder, and reading
//! results back from ORCA's output. The UI is in `ui::popup::quantum_chem::orca`.
//!
//! [ORCA 6 manual](https://www.faccts.de/docs/orca/6.1/manual/)
//! [Recommendations for methods, basis sets etc](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)

use std::{
    fmt::Write as _,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use bio_files::orca::{
    GeomOptThresh,
    basis_sets::{BasisSet, BasisSetCategory},
    dynamics::{Dynamics, Thermostat},
};
use lin_alg::f64::Vec3;
use mol_defs::molecules::MolType;
use na_seq::Element;
use serde_json::{Value, json};

use crate::external_tools::{
    RunControl, Tool, find_executable, process_executables_dir, run_tool_logged,
    shared_adapter::AdapterResult,
};

/// The base name of every run's files: ORCA names its outputs after the input.
const BASE: &str = "input";

/// An ORCA keyword choice, for the dropdowns.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub value: &'static str,
    pub label: &'static str,
    pub group: &'static str,
    pub help: &'static str,
}

const fn c(
    value: &'static str,
    label: &'static str,
    group: &'static str,
    help: &'static str,
) -> Choice {
    Choice {
        value,
        label,
        group,
        help,
    }
}

/// Methods offered in the dropdown. Any other ORCA method can be typed in.
#[rustfmt::skip]
pub const METHODS: &[Choice] = &[
    c("r2SCAN-3c", "r2SCAN-3c", "Composite (built-in basis)", "Meta-GGA composite. A robust, fast default for geometries and energies."),
    c("B97-3c", "B97-3c", "Composite (built-in basis)", "GGA composite. Faster than r2SCAN-3c."),
    c("wB97X-3c", "ωB97X-3c", "Composite (built-in basis)", "Range-separated hybrid composite. Good for energies and excited states."),
    c("PBEh-3c", "PBEh-3c", "Composite (built-in basis)", "Hybrid composite."),
    c("B3LYP-3c", "B3LYP-3c", "Composite (built-in basis)", "Hybrid composite."),
    c("HF-3c", "HF-3c", "Composite (built-in basis)", "Hartree–Fock composite. Very fast; large systems."),
    c("XTB2", "GFN2-xTB", "Semi-empirical (no basis)", "Tight-binding. Very fast; conformer searches and pre-optimization."),
    c("XTB1", "GFN1-xTB", "Semi-empirical (no basis)", "Tight-binding."),
    c("XTB0", "GFN0-xTB", "Semi-empirical (no basis)", "Non-self-consistent tight-binding."),
    c("AM1", "AM1", "Semi-empirical (no basis)", "NDDO semi-empirical."),
    c("PM3", "PM3", "Semi-empirical (no basis)", "NDDO semi-empirical."),
    c("HF", "Hartree–Fock", "Hartree–Fock", "No electron correlation."),
    c("BP86", "BP86", "GGA", ""),
    c("BLYP", "BLYP", "GGA", ""),
    c("PBE", "PBE", "GGA", ""),
    c("revPBE", "revPBE", "GGA", ""),
    c("B97-D3", "B97-D3", "GGA", "Includes D3 dispersion."),
    c("r2SCAN", "r2SCAN", "Meta-GGA", ""),
    c("TPSS", "TPSS", "Meta-GGA", ""),
    c("SCANFUNC", "SCAN", "Meta-GGA", ""),
    c("B97M-V", "B97M-V", "Meta-GGA", "Includes VV10 dispersion."),
    c("B97M-D4", "B97M-D4", "Meta-GGA", "Includes D4 dispersion."),
    c("B3LYP", "B3LYP", "Hybrid", ""),
    c("PBE0", "PBE0", "Hybrid", ""),
    c("TPSSh", "TPSSh", "Hybrid", ""),
    c("r2SCAN0", "r2SCAN0", "Hybrid", ""),
    c("r2SCANh", "r2SCANh", "Hybrid", ""),
    c("M06", "M06", "Hybrid", ""),
    c("M062X", "M06-2X", "Hybrid", ""),
    c("PW6B95", "PW6B95", "Hybrid", ""),
    c("BHANDHLYP", "BHandHLYP", "Hybrid", ""),
    c("wB97X-D4", "ωB97X-D4", "Range-separated hybrid", "Includes D4 dispersion."),
    c("wB97X-V", "ωB97X-V", "Range-separated hybrid", "Includes VV10 dispersion."),
    c("wB97M-V", "ωB97M-V", "Range-separated hybrid", "Includes VV10 dispersion."),
    c("wB97M-D4", "ωB97M-D4", "Range-separated hybrid", "Includes D4 dispersion."),
    c("wB97X-D3", "ωB97X-D3", "Range-separated hybrid", "Includes D3 dispersion."),
    c("CAM-B3LYP", "CAM-B3LYP", "Range-separated hybrid", ""),
    c("wr2SCAN", "ωr2SCAN", "Range-separated hybrid", ""),
    c("B2PLYP", "B2PLYP", "Double hybrid", ""),
    c("revDSD-PBEP86-D4/2021", "revDSD-PBEP86-D4", "Double hybrid", "Includes D4 dispersion."),
    c("PWPB95", "PWPB95", "Double hybrid", ""),
    c("wB97X-2", "ωB97X-2", "Double hybrid", ""),
    c("wB2PLYP", "ωB2PLYP", "Double hybrid", ""),
    c("RI-MP2", "RI-MP2", "Wavefunction", "Set the aux basis to AutoAux, or add a /C basis."),
    c("RI-SCS-MP2", "RI-SCS-MP2", "Wavefunction", "Set the aux basis to AutoAux, or add a /C basis."),
    c("OO-RI-MP2", "OO-RI-MP2", "Wavefunction", "Set the aux basis to AutoAux, or add a /C basis."),
    c("DLPNO-MP2", "DLPNO-MP2", "Wavefunction", "Set the aux basis to AutoAux, or add a /C basis."),
    c("CCSD", "CCSD", "Wavefunction", "Small molecules only."),
    c("CCSD(T)", "CCSD(T)", "Wavefunction", "Small molecules only."),
    c("DLPNO-CCSD(T)", "DLPNO-CCSD(T)", "Wavefunction", "Near-CCSD(T) for large molecules. Set the aux basis to AutoAux, or add a /C basis."),
    c("DLPNO-CCSD(T1)", "DLPNO-CCSD(T1)", "Wavefunction", "Iterative triples. Set the aux basis to AutoAux, or add a /C basis."),
];

/// Solvents for the implicit solvation models, spelled as ORCA keywords. Any other ORCA solvent
/// can be typed in. Not every model supports every solvent; ORCA reports it if not.
#[rustfmt::skip]
pub const SOLVENTS: &[Choice] = &[
    c("water", "Water", "", ""),
    c("acetonitrile", "Acetonitrile", "", ""),
    c("acetone", "Acetone", "", ""),
    c("aceticacid", "Acetic acid", "", ""),
    c("aniline", "Aniline", "", ""),
    c("benzene", "Benzene", "", ""),
    c("butanol", "1-Butanol", "", ""),
    c("ccl4", "Carbon tetrachloride", "", ""),
    c("ch2cl2", "Dichloromethane", "", ""),
    c("chloroform", "Chloroform", "", ""),
    c("chlorobenzene", "Chlorobenzene", "", ""),
    c("cs2", "Carbon disulfide", "", ""),
    c("cyclohexane", "Cyclohexane", "", ""),
    c("1,2-dichloroethane", "1,2-Dichloroethane", "", ""),
    c("diethylether", "Diethyl ether", "", ""),
    c("dioxane", "1,4-Dioxane", "", ""),
    c("dmf", "DMF", "", ""),
    c("dmso", "DMSO", "", ""),
    c("ethanol", "Ethanol", "", ""),
    c("ethylacetate", "Ethyl acetate", "", ""),
    c("heptane", "Heptane", "", ""),
    c("hexane", "Hexane", "", ""),
    c("isopropanol", "2-Propanol", "", ""),
    c("methanol", "Methanol", "", ""),
    c("nitromethane", "Nitromethane", "", ""),
    c("octanol", "1-Octanol", "", ""),
    c("propanol", "1-Propanol", "", ""),
    c("pyridine", "Pyridine", "", ""),
    c("thf", "THF", "", ""),
    c("toluene", "Toluene", "", ""),
    c("conductor", "Conductor (ε = ∞)", "", "CPCM only."),
];

/// Whether `method` brings its own basis, so that none is written.
pub fn method_has_basis(method: &str) -> bool {
    let method = method.trim().to_ascii_lowercase();

    !(method.ends_with("-3c")
        || method.starts_with("xtb")
        || method.starts_with("gfn")
        || matches!(method.as_str(), "am1" | "pm3" | "mndo" | "zindo/s"))
}

/// Whether `method` is xTB, which takes the xTB solvation models.
fn is_xtb(method: &str) -> bool {
    let method = method.trim().to_ascii_lowercase();

    method.starts_with("xtb") || method.starts_with("gfn")
}

/// Generates enums with a label, keyword, help text, and a list of variants for dropdowns.
macro_rules! keyword_enum {
    (
        $(#[$meta:meta])*
        $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident => ($label:expr, $keyword:expr, $help:expr) ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            pub const ALL: &[Self] = &[$(Self::$variant),+];

            pub fn label(self) -> &'static str {
                match self { $(Self::$variant => $label),+ }
            }

            /// The simple-input keyword. Empty for ORCA's default.
            #[allow(dead_code)]
            pub fn keyword(self) -> &'static str {
                match self { $(Self::$variant => $keyword),+ }
            }

            pub fn help(self) -> &'static str {
                match self { $(Self::$variant => $help),+ }
            }
        }
    };
}

keyword_enum! {
    /// The calculation ORCA runs.
    OrcaTask {
        SinglePoint => ("Single-point energy", "", "Energy and properties at this geometry."),
        Gradient => ("Energy and gradient", "EnGrad", "Energy and nuclear gradient at this geometry."),
        #[default]
        Optimize => ("Optimize geometry", "", "Minimize the energy with respect to atom positions."),
        OptFreq => ("Optimize, then frequencies", "", "Optimize, then compute vibrational frequencies to confirm a minimum, and thermochemistry."),
        TransitionState => ("Transition state search", "OptTS", "Optimize to a first-order saddle point, starting from an exact Hessian, then compute frequencies. Start from a guess near the TS."),
        Frequencies => ("Vibrational frequencies", "", "Harmonic frequencies, IR intensities, and thermochemistry at this geometry. Use an optimized geometry."),
        Charges => ("Partial charges", "", "Atomic partial charges. MBIS is slow but accurate; suitable for force-field parameterization."),
        ExcitedStates => ("Excited states (TD-DFT)", "", "Vertical excitation energies and the UV/Vis absorption spectrum."),
        Nmr => ("NMR shieldings", "NMR", "NMR chemical shieldings, and spin–spin couplings where requested."),
        ConformerSearch => ("Conformer search (GOAT)", "GOAT", "Global conformer search. Use a fast method, e.g. GFN2-xTB; it runs many optimizations."),
        Dynamics => ("Molecular dynamics", "MD", "Ab-initio MD. Much slower than force-field MD, but bonds may form and break."),
    }
}

keyword_enum! {
    /// How frequencies are computed.
    FreqKind {
        #[default]
        Analytic => ("Analytic", "Freq", "Analytic Hessian. Fast, where the method supports it."),
        Numeric => ("Numeric", "NumFreq", "Finite differences of gradients. Works for any method with gradients."),
    }
}

keyword_enum! {
    ChargeScheme {
        #[default]
        Mbis => ("MBIS", "MBIS", "Minimal basis iterative stockholder. Accurate; slow."),
        Chelpg => ("CHELPG", "CHELPG", "Fit to the electrostatic potential."),
        Hirshfeld => ("Hirshfeld", "Hirshfeld", "Hirshfeld population analysis."),
    }
}

keyword_enum! {
    Reference {
        #[default]
        Auto => ("Auto", "", "Restricted for closed shells; unrestricted otherwise."),
        Restricted => ("Restricted", "RHF", "RHF / RKS."),
        Unrestricted => ("Unrestricted", "UHF", "UHF / UKS."),
        RestrictedOpen => ("Restricted open-shell", "ROHF", "ROHF / ROKS."),
    }
}

keyword_enum! {
    Dispersion {
        #[default]
        None => ("None", "", "No dispersion correction, beyond any the method includes."),
        D4 => ("D4", "D4", "Recommended for DFT functionals without dispersion."),
        D3Bj => ("D3(BJ)", "D3BJ", "D3 with Becke–Johnson damping."),
        D3Zero => ("D3(0)", "D3ZERO", "D3 with zero damping."),
        Nl => ("VV10 (NL)", "NL", "Non-local VV10 dispersion."),
    }
}

keyword_enum! {
    RiApprox {
        #[default]
        Default => ("Default", "", "ORCA's default; RIJCOSX for hybrid DFT."),
        RijCosx => ("RIJCOSX", "RIJCOSX", "RI for Coulomb, seminumerical exchange."),
        RijOnx => ("RIJONX", "RIJONX", "RI for Coulomb, exact exchange."),
        Rijk => ("RIJK", "RIJK", "RI for Coulomb and exchange."),
        NoRi => ("No RI", "NORI", "Exact integrals. Slow."),
    }
}

keyword_enum! {
    AuxBasis {
        #[default]
        Default => ("Default", "", "ORCA picks one where it can."),
        AutoAux => ("AutoAux", "AutoAux", "Generate auxiliary bases automatically. Works with any orbital basis and method."),
        Def2J => ("def2/J", "def2/J", "Coulomb fitting."),
        Def2Jk => ("def2/JK", "def2/JK", "Coulomb and exchange fitting."),
    }
}

keyword_enum! {
    ScfConv {
        #[default]
        Default => ("Default", "", "ORCA's default for the task."),
        Loose => ("Loose", "LooseSCF", ""),
        Normal => ("Normal", "NormalSCF", ""),
        Strong => ("Strong", "StrongSCF", ""),
        Tight => ("Tight", "TightSCF", "Recommended for frequencies and properties."),
        VeryTight => ("Very tight", "VeryTightSCF", ""),
        Extreme => ("Extreme", "ExtremeSCF", ""),
    }
}

keyword_enum! {
    ScfAid {
        #[default]
        None => ("None", "", ""),
        SlowConv => ("SlowConv", "SlowConv", "Damping for difficult SCFs, e.g. transition metals."),
        VerySlowConv => ("VerySlowConv", "VerySlowConv", "Stronger damping."),
    }
}

keyword_enum! {
    DftGrid {
        #[default]
        Default => ("Default", "", "DefGrid2."),
        DefGrid1 => ("DefGrid1", "DefGrid1", "Coarse; fast."),
        DefGrid2 => ("DefGrid2", "DefGrid2", ""),
        DefGrid3 => ("DefGrid3", "DefGrid3", "Fine; for difficult functionals or tight results."),
    }
}

keyword_enum! {
    Solvation {
        #[default]
        None => ("None (gas phase)", "", ""),
        Cpcm => ("CPCM", "CPCM", "Conductor-like polarizable continuum."),
        Cpcmc => ("CPCMC", "CPCMC", "CPCM with the COSMO epsilon function."),
        Smd => ("SMD", "SMD", "CPCM with SMD non-electrostatic terms. Best for solvation free energies."),
        Alpb => ("ALPB (xTB)", "ALPB", "xTB methods only."),
        DdCosmo => ("ddCOSMO (xTB)", "DDCOSMO", "xTB methods only."),
        CpcmX => ("CPCM-X (xTB)", "CPCMX", "xTB methods only."),
    }
}

keyword_enum! {
    MdThermostat {
        #[default]
        Csvr => ("CSVR", "CSVR", "Canonical sampling through velocity rescaling."),
        Nhc => ("Nosé–Hoover chain", "NHC", ""),
        None => ("None (NVE)", "None", ""),
    }
}

/// The geometry-optimization thresholds, for the dropdown.
pub const OPT_THRESHOLDS: [GeomOptThresh; 4] = [
    GeomOptThresh::Loose,
    GeomOptThresh::Opt,
    GeomOptThresh::Tight,
    GeomOptThresh::VeryTight,
];

/// The basis set categories, for the dropdown.
pub const BASIS_CATEGORIES: [BasisSetCategory; 5] = [
    BasisSetCategory::KarlseruheDef2,
    BasisSetCategory::Ahlrich,
    BasisSetCategory::KarlseruheDhf,
    BasisSetCategory::Pople,
    BasisSetCategory::CorrelationConsistent,
];

/// Everything "Set parameters here" sets. Each dropdown that may lack an option has a text
/// override; when not empty, it replaces the dropdown's value.
#[derive(Clone, Debug, PartialEq)]
pub struct OrcaParams {
    pub task: OrcaTask,
    pub method: String,
    pub method_custom: String,
    pub basis_cat: BasisSetCategory,
    pub basis: BasisSet,
    pub basis_custom: String,
    pub charge: i32,
    pub multiplicity: u32,
    pub reference: Reference,
    pub dispersion: Dispersion,
    pub solvation: Solvation,
    pub solvent: String,
    pub solvent_custom: String,
    pub scf_conv: ScfConv,
    pub scf_aid: ScfAid,
    pub grid: DftGrid,
    pub ri: RiApprox,
    pub aux_basis: AuxBasis,
    pub opt_thresh: GeomOptThresh,
    /// 0 for ORCA's default.
    pub opt_max_iter: u32,
    pub freq: FreqKind,
    pub charge_scheme: ChargeScheme,
    pub tddft_roots: u32,
    pub tddft_triplets: bool,
    pub md_timestep_fs: f32,
    pub md_steps: u32,
    pub md_temp_k: f32,
    pub md_thermostat: MdThermostat,
    pub nprocs: u32,
    /// MB per core. 0 for ORCA's default.
    pub maxcore_mb: u32,
    /// Appended to the keyword lines.
    pub extra_keywords: String,
    /// Input blocks, e.g. `%scf MaxIter 500 end`, written as-is.
    pub extra_blocks: String,
}

impl Default for OrcaParams {
    fn default() -> Self {
        Self {
            task: OrcaTask::default(),
            method: "r2SCAN-3c".to_owned(),
            method_custom: String::new(),
            basis_cat: BasisSetCategory::KarlseruheDef2,
            basis: BasisSet::Def2Tzvp,
            basis_custom: String::new(),
            charge: 0,
            multiplicity: 1,
            reference: Reference::default(),
            dispersion: Dispersion::default(),
            solvation: Solvation::default(),
            solvent: "water".to_owned(),
            solvent_custom: String::new(),
            scf_conv: ScfConv::default(),
            scf_aid: ScfAid::default(),
            grid: DftGrid::default(),
            ri: RiApprox::default(),
            aux_basis: AuxBasis::default(),
            opt_thresh: GeomOptThresh::Opt,
            opt_max_iter: 0,
            freq: FreqKind::default(),
            charge_scheme: ChargeScheme::default(),
            tddft_roots: 10,
            tddft_triplets: false,
            md_timestep_fs: 0.5,
            md_steps: 200,
            md_temp_k: 300.,
            md_thermostat: MdThermostat::default(),
            nprocs: 1,
            maxcore_mb: 0,
            extra_keywords: String::new(),
            extra_blocks: String::new(),
        }
    }
}

/// `custom` when not blank; otherwise `chosen`.
pub fn overridden<'a>(chosen: &'a str, custom: &'a str) -> &'a str {
    match custom.trim() {
        "" => chosen,
        custom => custom,
    }
}

/// Atoms as ORCA receives them.
pub type Geometry = [(Element, Vec3)];

impl OrcaParams {
    pub fn method(&self) -> &str {
        overridden(&self.method, &self.method_custom)
    }

    pub fn basis(&self) -> String {
        match self.basis_custom.trim() {
            "" => self.basis.keyword(),
            custom => custom.to_owned(),
        }
    }

    pub fn solvent(&self) -> &str {
        overridden(&self.solvent, &self.solvent_custom)
    }

    pub fn uses_basis(&self) -> bool {
        method_has_basis(self.method())
    }

    /// The `!` keyword lines.
    pub fn keyword_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let basis = self.uses_basis();

        let mut model = vec![self.method().to_owned()];
        if basis {
            model.push(self.basis());
            model.push(self.aux_basis.keyword().to_owned());
            model.push(self.dispersion.keyword().to_owned());
        }
        model.push(self.ri.keyword().to_owned());
        model.push(self.reference.keyword().to_owned());
        lines.push(model);

        let task = match self.task {
            OrcaTask::Optimize => vec![self.opt_thresh.keyword()],
            OrcaTask::OptFreq => vec![self.opt_thresh.keyword(), self.freq.keyword().to_owned()],
            OrcaTask::TransitionState => {
                vec![
                    OrcaTask::TransitionState.keyword().to_owned(),
                    self.freq.keyword().to_owned(),
                ]
            }
            OrcaTask::Frequencies => vec![self.freq.keyword().to_owned()],
            OrcaTask::Charges => vec![self.charge_scheme.keyword().to_owned()],
            task => vec![task.keyword().to_owned()],
        };
        lines.push(task);

        let solvation = match self.solvation {
            Solvation::None => String::new(),
            model => format!("{}({})", model.keyword(), self.solvent()),
        };
        lines.push(vec![
            self.scf_conv.keyword().to_owned(),
            self.scf_aid.keyword().to_owned(),
            self.grid.keyword().to_owned(),
            solvation,
        ]);
        lines.push(vec![self.extra_keywords.trim().to_owned()]);

        lines
            .into_iter()
            .map(|words| {
                words
                    .into_iter()
                    .filter(|word| !word.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .filter(|line| !line.is_empty())
            .map(|line| format!("! {line}"))
            .collect()
    }

    /// Input blocks, e.g. `%pal`.
    fn blocks(&self) -> Vec<String> {
        let mut blocks = Vec::new();

        if self.nprocs > 1 {
            blocks.push(format!("%pal nprocs {} end", self.nprocs));
        }
        if self.maxcore_mb > 0 {
            blocks.push(format!("%maxcore {}", self.maxcore_mb));
        }

        let optimizes = matches!(
            self.task,
            OrcaTask::Optimize | OrcaTask::OptFreq | OrcaTask::TransitionState
        );
        let mut geom = Vec::new();
        if optimizes && self.opt_max_iter > 0 {
            geom.push(format!("  MaxIter {}", self.opt_max_iter));
        }
        if self.task == OrcaTask::TransitionState {
            geom.push("  Calc_Hess true".to_owned());
        }
        if !geom.is_empty() {
            blocks.push(format!("%geom\n{}\nend", geom.join("\n")));
        }

        if self.task == OrcaTask::ExcitedStates {
            blocks.push(format!(
                "%tddft\n  NRoots {}\n  Triplets {}\nend",
                self.tddft_roots.max(1),
                self.tddft_triplets
            ));
        }

        if self.task == OrcaTask::Dynamics {
            let thermostat = match self.md_thermostat {
                MdThermostat::Csvr => Thermostat::Csvr,
                MdThermostat::Nhc => Thermostat::Nhc,
                MdThermostat::None => Thermostat::None,
            };
            let md = Dynamics {
                timestep: self.md_timestep_fs,
                init_vel: self.md_temp_k,
                thermostat,
                thermostat_temp: self.md_temp_k,
                thermostat_timecon: 10.,
                traj_out_dir: PathBuf::from(format!("{BASE}.md.xyz")),
                steps: self.md_steps,
            };
            blocks.push(md.make_inp());
        }

        let extra = self.extra_blocks.trim();
        if !extra.is_empty() {
            blocks.push(extra.to_owned());
        }

        blocks
    }

    /// The complete input. `geometry` of `None` writes a placeholder, for previews.
    pub fn make_input(&self, geometry: Option<&Geometry>, title: &str) -> String {
        let mut text = format!("# {title}\n# Written by Molchanica\n");

        for line in self.keyword_lines() {
            text.push_str(&line);
            text.push('\n');
        }

        let blocks = self.blocks();
        if !blocks.is_empty() {
            text.push('\n');
            for block in blocks {
                text.push_str(&block);
                text.push('\n');
            }
        }

        let _ = writeln!(text, "\n* xyz {} {}", self.charge, self.multiplicity.max(1));
        match geometry {
            Some(atoms) => {
                for (element, posit) in atoms {
                    let _ = writeln!(
                        text,
                        "  {:<2} {:>14.6} {:>14.6} {:>14.6}",
                        element.to_letter(),
                        posit.x,
                        posit.y,
                        posit.z
                    );
                }
            }
            None => text.push_str("  …the active molecule's atoms…\n"),
        }
        text.push('*');
        text.push('\n');

        text
    }

    /// Problems with these parameters that ORCA would reject or misread.
    /// `electrons`: of the neutral molecule, where known.
    pub fn warnings(&self, electrons: Option<i64>) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.method().is_empty() {
            warnings.push("No method is set.".to_owned());
        }
        if self.uses_basis() && self.basis().is_empty() {
            warnings.push("No basis set is set.".to_owned());
        }
        if matches!(
            self.solvation,
            Solvation::Alpb | Solvation::DdCosmo | Solvation::CpcmX
        ) && !is_xtb(self.method())
        {
            warnings.push(format!(
                "{} solvation needs an xTB method.",
                self.solvation.label()
            ));
        }

        if let Some(electrons) = electrons {
            let electrons = electrons - self.charge as i64;
            let unpaired_odd = self.multiplicity % 2 == 0;
            if electrons < 0 {
                warnings.push("The charge removes more electrons than there are.".to_owned());
            } else if (electrons % 2 == 1) != unpaired_odd {
                warnings.push(format!(
                    "{electrons} electrons is incompatible with multiplicity {}.",
                    self.multiplicity
                ));
            }
        }

        warnings
    }

    /// The file the run's final geometry is written to, if this task makes one.
    fn geometry_file(&self) -> Option<String> {
        match self.task {
            OrcaTask::Optimize | OrcaTask::OptFreq | OrcaTask::TransitionState => {
                Some(format!("{BASE}.xyz"))
            }
            OrcaTask::ConformerSearch => Some(format!("{BASE}.globalminimum.xyz")),
            _ => None,
        }
    }
}

/// The electron count of a neutral molecule of these atoms. `None` if it includes an element we
/// don't know the atomic number of.
pub fn electron_count(elements: impl IntoIterator<Item = Element>) -> Option<i64> {
    use Element::*;

    elements
        .into_iter()
        .map(|element| {
            Some(match element {
                Hydrogen => 1,
                Boron => 5,
                Carbon => 6,
                Nitrogen => 7,
                Oxygen => 8,
                Fluorine => 9,
                Sodium => 11,
                Magnesium => 12,
                Aluminum => 13,
                Silicon => 14,
                Phosphorus => 15,
                Sulfur => 16,
                Chlorine => 17,
                Potassium => 19,
                Calcium => 20,
                Titanium => 22,
                Manganese => 25,
                Iron => 26,
                Cobalt => 27,
                Copper => 29,
                Zinc => 30,
                Selenium => 34,
                Bromine => 35,
                Rubidium => 37,
                Molybdenum => 42,
                Silver => 47,
                Indium => 49,
                Tin => 50,
                Tellurium => 52,
                Iodine => 53,
                Barium => 56,
                Tungsten => 74,
                Gold => 79,
                Mercury => 80,
                Lead => 82,
                Other => return None,
            })
        })
        .sum()
}

/// The molecule a run was made from, to apply its results to.
#[derive(Clone, Debug, PartialEq)]
pub struct MolTarget {
    pub mol_type: MolType,
    pub index: usize,
    /// Confirms the molecule at `index` is still the one the run was for.
    pub ident: String,
    pub atom_count: usize,
}

impl MolTarget {
    fn to_json(&self) -> Value {
        let mol_type = match self.mol_type {
            MolType::Peptide => "peptide",
            MolType::Ligand => "ligand",
            MolType::NucleicAcid => "nucleic_acid",
            MolType::Lipid => "lipid",
            MolType::Pocket => "pocket",
            MolType::Water => "water",
        };

        json!({
            "mol_type": mol_type,
            "index": self.index,
            "ident": self.ident,
            "atom_count": self.atom_count,
        })
    }

    pub fn from_json(value: &Value) -> Option<Self> {
        let mol_type = match value["mol_type"].as_str()? {
            "peptide" => MolType::Peptide,
            "ligand" => MolType::Ligand,
            "nucleic_acid" => MolType::NucleicAcid,
            "lipid" => MolType::Lipid,
            "pocket" => MolType::Pocket,
            _ => return None,
        };

        Some(Self {
            mol_type,
            index: value["index"].as_u64()? as usize,
            ident: value["ident"].as_str()?.to_owned(),
            atom_count: value["atom_count"].as_u64()? as usize,
        })
    }
}

/// One ORCA run, ready to start.
pub struct OrcaJob {
    pub input: String,
    /// "parameters", "text", or "upload".
    pub mode: &'static str,
    pub target: Option<MolTarget>,
    /// Shown in the results; e.g. the task and method.
    pub summary: Value,
    /// For the computation status display.
    pub activity: String,
    geometry_file: Option<String>,
}

impl OrcaJob {
    pub fn from_params(params: &OrcaParams, geometry: &Geometry, target: MolTarget) -> Self {
        Self {
            input: params.make_input(Some(geometry), &target.ident),
            mode: "parameters",
            summary: json!({
                "task": params.task.label(),
                "method": params.method(),
                "basis": if params.uses_basis() { params.basis() } else { String::new() },
                "charge": params.charge,
                "multiplicity": params.multiplicity,
            }),
            geometry_file: params.geometry_file(),
            activity: format!("{} of {}", params.task.label().to_lowercase(), target.ident),
            target: Some(target),
        }
    }

    /// From an input written or chosen by the user.
    pub fn from_text(input: String, mode: &'static str) -> Self {
        Self {
            input,
            mode,
            target: None,
            summary: json!({}),
            activity: "running an input".to_owned(),
            geometry_file: None,
        }
    }
}

/// `<process_executables>/results/orca/<timestamp>-<pid>`, created fresh.
fn create_results_dir() -> io::Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let directory = process_executables_dir()?
        .join("results")
        .join("orca")
        .join(format!("{stamp}-{}", std::process::id()));

    fs::create_dir_all(&directory)?;
    Ok(directory)
}

/// Run ORCA to completion, keeping its files. Blocking; call it from a worker thread.
pub fn run(job: OrcaJob, control: &RunControl) -> io::Result<AdapterResult> {
    // ORCA needs its full path to run in parallel.
    let executable = find_executable(Tool::Orca)?;
    let directory = create_results_dir()?;
    let input = directory.join(format!("{BASE}.inp"));
    let output = directory.join(format!("{BASE}.out"));
    fs::write(&input, &job.input)?;

    let mut command = Command::new(executable);
    command
        .arg(input.file_name().unwrap())
        .current_dir(&directory);
    let execution = run_tool_logged(
        &mut command,
        Tool::Orca,
        "a quantum chemistry calculation",
        Some(control),
        Some(&output),
    );
    if control.is_abort_requested() {
        return Err(io::Error::new(io::ErrorKind::Interrupted, "Run cancelled"));
    }

    let with_run_files = |message: String| {
        io::Error::other(format!("{message}\nRun files: {}", directory.display()))
    };

    let text = fs::read_to_string(&output).unwrap_or_default();
    let normal = text.contains("ORCA TERMINATED NORMALLY");
    if let Err(error) = execution {
        let reason = orca_error(&text).unwrap_or_else(|| error.to_string());
        return Err(with_run_files(reason));
    }
    if !normal {
        let reason = orca_error(&text)
            .unwrap_or_else(|| "ORCA did not terminate normally; see input.out.".to_owned());
        return Err(with_run_files(reason));
    }

    let mut files = Vec::new();
    collect_files(&directory, &mut files)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "result.json"));
    files.sort();

    let geometry = job
        .geometry_file
        .map(|name| directory.join(name))
        .or_else(|| {
            [format!("{BASE}.globalminimum.xyz"), format!("{BASE}.xyz")]
                .into_iter()
                .map(|name| directory.join(name))
                .find(|path| path.is_file())
        })
        .filter(|path| path.is_file());

    let mut details = parse_output(&text);
    details["tool_slug"] = json!("orca");
    details["input_mode"] = json!(job.mode);
    details["summary"] = job.summary;
    details["run_log_dir"] = json!(directory);
    details["output_files"] = json!(files);
    details["final_geometry"] = json!(geometry);
    if let Some(target) = &job.target {
        details["molecule"] = target.to_json();
    }

    fs::write(
        directory.join("result.json"),
        serde_json::to_vec_pretty(&json!({ "result": details }))?,
    )?;

    Ok(AdapterResult {
        archive: directory.join("raw-results.zip"),
        directory,
        files,
        details,
    })
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_files(&entry.path(), files)?;
        } else if kind.is_file() {
            files.push(entry.path());
        }
    }

    Ok(())
}

/// ORCA's own explanation of a failure, from its output.
fn orca_error(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let index = lines.iter().rposition(|line| {
        let line = line.to_ascii_uppercase();
        line.contains("ERROR") || line.contains("ABORTING") || line.contains("UNRECOGNIZED")
    })?;
    let start = index.saturating_sub(2);
    let end = (index + 4).min(lines.len());

    Some(
        lines[start..end]
            .iter()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

// ---------------------------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------------------------

/// Results read from ORCA's output, as the run details.
pub fn parse_output(text: &str) -> Value {
    let mut details = json!({});

    if let Some(energy) = last_number_after(text, "FINAL SINGLE POINT ENERGY") {
        details["energy_hartree"] = json!(energy);
    }
    if let Some(dipole) = last_number_after(text, "Magnitude (Debye)") {
        details["dipole_debye"] = json!(dipole);
    }
    if let Some(g) = last_number_after(text, "Final Gibbs free energy") {
        details["gibbs_hartree"] = json!(g);
    }
    if let Some((homo, lumo)) = frontier_orbitals(text) {
        details["homo_ev"] = json!(homo);
        details["lumo_ev"] = json!(lumo);
    }

    let mut charges = serde_json::Map::new();
    for (scheme, header) in [
        ("MBIS", "MBIS ANALYSIS"),
        ("CHELPG", "CHELPG Charges"),
        ("Hirshfeld", "HIRSHFELD ANALYSIS"),
        ("Mulliken", "MULLIKEN ATOMIC CHARGES"),
        ("Löwdin", "LOEWDIN ATOMIC CHARGES"),
    ] {
        if let Some((elements, values)) = atom_charges(text, header) {
            charges.insert(
                scheme.into(),
                json!({ "elements": elements, "values": values }),
            );
        }
    }
    if !charges.is_empty() {
        details["charges"] = Value::Object(charges);
    }

    if let Some(modes) = frequencies(text) {
        details["vibrational_modes"] = Value::Array(modes);
    }

    details
}

/// The first number on the last line containing `marker`, after the marker.
fn last_number_after(text: &str, marker: &str) -> Option<f64> {
    let line = text.lines().rev().find(|line| line.contains(marker))?;
    let rest = &line[line.find(marker)? + marker.len()..];

    rest.split_whitespace().find_map(|word| {
        word.trim_matches(|c: char| c == ':' || c == '=')
            .parse()
            .ok()
    })
}

/// The section after the last line containing `header`.
fn last_section<'a>(text: &'a str, header: &str) -> Option<std::str::Lines<'a>> {
    let start = text.rfind(header)?;
    let mut lines = text[start..].lines();
    lines.next();

    Some(lines)
}

/// Per-atom charges from a table like `0 O : -0.33` or `0 O -0.29 0.00`, in the last section
/// headed `header`.
fn atom_charges(text: &str, header: &str) -> Option<(Vec<String>, Vec<f64>)> {
    let mut elements = Vec::new();
    let mut values = Vec::new();

    for (i, line) in last_section(text, header)?.enumerate() {
        let words: Vec<&str> = line.split_whitespace().filter(|w| *w != ":").collect();
        let row = match words.as_slice() {
            [index, element, value, ..] => index
                .parse::<usize>()
                .ok()
                .filter(|index| *index == values.len())
                .and_then(|_| {
                    let element = element.trim_end_matches(':');
                    element.chars().all(char::is_alphabetic).then_some(element)
                })
                .zip(value.parse::<f64>().ok()),
            _ => None,
        };

        match row {
            Some((element, value)) => {
                elements.push(element.to_owned());
                values.push(value);
            }
            None if !values.is_empty() || i > 30 => break,
            None => {}
        }
    }

    (!values.is_empty()).then_some((elements, values))
}

/// Non-zero vibrational modes, with IR intensities where ORCA reports them.
fn frequencies(text: &str) -> Option<Vec<Value>> {
    let mut modes = Vec::new();
    let mut in_table = false;

    // A few header lines, then rows like `6:  1630.17 cm**-1`.
    for line in last_section(text, "VIBRATIONAL FREQUENCIES")?.take(1_000) {
        let row = line.split_once(':').and_then(|(index, rest)| {
            let index = index.trim().parse::<usize>().ok()?;
            let frequency = rest.split_whitespace().next()?.parse::<f64>().ok()?;

            Some((index, frequency, rest))
        });

        let Some((index, frequency, rest)) = row else {
            if in_table {
                break;
            }
            continue;
        };
        in_table = true;

        if frequency.abs() > 1e-6 {
            modes.push(json!({
                "mode": index,
                "frequency_cm1": frequency,
                "imaginary": frequency < 0. || rest.contains("imaginary"),
            }));
        }
    }

    // `Mode freq eps Int …`, where Int is in km/mol.
    if let Some(lines) = last_section(text, "IR SPECTRUM") {
        for line in lines.take(modes.len() + 12) {
            let words: Vec<&str> = line.split_whitespace().collect();
            let [index, _, _, intensity, ..] = words.as_slice() else {
                continue;
            };
            let (Ok(index), Ok(intensity)) = (
                index.trim_end_matches(':').parse::<usize>(),
                intensity.parse::<f64>(),
            ) else {
                continue;
            };

            if let Some(mode) = modes.iter_mut().find(|mode| mode["mode"] == index) {
                mode["ir_intensity_km_mol"] = json!(intensity);
            }
        }
    }

    (!modes.is_empty()).then_some(modes)
}

/// HOMO and LUMO energies in eV, from the last orbital energy table. For unrestricted runs, the
/// alpha orbitals.
fn frontier_orbitals(text: &str) -> Option<(f64, f64)> {
    let mut homo = None;

    for line in last_section(text, "ORBITAL ENERGIES")? {
        let words: Vec<&str> = line.split_whitespace().collect();
        let [index, occupation, _, ev, ..] = words.as_slice() else {
            if homo.is_some() {
                break;
            }
            continue;
        };
        let (Ok(_), Ok(occupation), Ok(ev)) = (
            index.parse::<usize>(),
            occupation.parse::<f64>(),
            ev.parse::<f64>(),
        ) else {
            continue;
        };

        if occupation > 0.5 {
            homo = Some(ev);
        } else if let Some(homo) = homo {
            return Some((homo, ev));
        }
    }

    None
}

/// Atom positions from an XYZ file's first frame.
pub fn read_xyz_positions(path: &Path) -> io::Result<Vec<Vec3>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let count: usize = lines
        .next()
        .and_then(|line| line.trim().parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Not an XYZ file"))?;
    lines.next();

    let posits: Vec<Vec3> = lines
        .take(count)
        .filter_map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let [_, x, y, z, ..] = words.as_slice() else {
                return None;
            };

            Some(Vec3::new(x.parse().ok()?, y.parse().ok()?, z.parse().ok()?))
        })
        .collect();

    if posits.len() != count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "The XYZ file is truncated",
        ));
    }

    Ok(posits)
}
