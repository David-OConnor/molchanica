// #![cfg_attr(
//     all(not(debug_assertions), target_os = "windows"),
//     windows_subsystem = "windows"
// )]

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
// Note: To test if it compiles on ARM:
// `rustup target add aarch64-pc-windows-msvc`
// `cargo check --target aarch64-pc-windows-msvc`
// note: Currently getting Clang errors when I attempt htis.

// todo: Consider APBS: https://github.com/Electrostatics/apbs (For a type of surface visulaization?)P

//! [S3 Gemmi link](https://daedalus-mols.s3.us-east-1.amazonaws.com/gemmi.exe)
//! [S3 Geostd link](https://daedalus-mols.s3.us-east-1.amazonaws.com/amber_geostd)

mod bond_inference;
mod docking;
mod download_mols;
mod drawing;
mod file_io;
mod forces;
mod inputs;
mod prefs;
mod render;
mod ribbon_mesh;
mod sa_surface;
mod ui;
mod util;

mod cli;
mod reflection;

mod cam_misc;
mod drawing_wrappers;
mod drug_design;
mod md;
mod mol_alignment;
mod mol_characterization;
mod mol_editor;
mod mol_manip;
mod mol_screening;
mod molecules;
mod orca;
mod pharmacokinetics;
mod pharmacophore;
mod selection;
mod smiles;
mod state;
mod tautomers;
#[cfg(test)]
mod tests;
mod viridis_lut;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    env, fmt,
    fmt::{Display, Formatter},
    path::PathBuf,
    process::Command,
    sync::mpsc::Receiver,
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_apis::{
    ReqError,
    amber_geostd::{GeostdData, GeostdItem},
    pubchem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::{md_params::ForceFieldParams, mol_templates::TemplateData};
#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream},
    nvrtc::Ptx,
};
use drawing::MoleculeView;
use dynamics::{ComputationDevice, Integrator, MdState, SimBoxInit, params::FfParamSet};
use egui_file_dialog::{FileDialog, FileDialogConfig};
use graphics::{Camera, ControlScheme, InputsCommanded, winit::event::Modifiers};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};
use mol_manip::MolManip;
use molecules::{
    MolGenericRef, MolGenericRefMut, MolIdent, MolType, MoleculePeptide,
    lipid::{LipidShape, MoleculeLipid, load_lipid_templates},
    nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType, Strands, load_na_templates},
    small::MoleculeSmall,
};
use selection::{Selection, ViewSelLevel};
use state::{State, StateUi, StateVolatile};

use crate::{
    mol_alignment::StateAlignment,
    mol_editor::MolEditorState,
    orca::StateOrca,
    prefs::ToSave,
    render::render,
    ui::cam::{FOG_DIST_DEFAULT, VIEW_DEPTH_NEAR_MIN},
    util::handle_err,
};

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../molchanica.ptx");

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
// For now, we check for differences between to_save and to_save prev, and write to disk
// if they're not equal.
const PREFS_SAVE_INTERVAL: u64 = 20; // seconds

/// The MdModule is owned by `dynamics::ComputationDevice`.
#[cfg(feature = "cuda")]
struct CudaFunctions {
    /// For processing as part of loading electron density data
    pub reflections: Arc<CudaFunction>,
}

// /// This wraps `dyanmics::ComputationDevice`. It's a bit awkard, but for now
// /// allows Dynamics to own ComputationDev with the MD model. We add our additional
// /// models here. Note: The CudaStream is owned by the inner `dynamics::ComputationDevice`.
// enum ComputationDevOuter {
//     Cpu,
//     #[cfg(feature = "cuda")]
//     Gpu((ComputationDevice, Arc<CudaModule>>)),
// }

struct FileDialogs {
    load: FileDialog,
    save: FileDialog,
    /// This is for selecting a folder; not file.
    screening: FileDialog,
}

impl Default for FileDialogs {
    fn default() -> Self {
        let cfg_all = FileDialogConfig::default()
            .add_file_filter_extensions(
                "All",
                vec![
                    "cif", "mol2", "sdf", "xyz", "pdbqt", "map", "mtz", "frcmod", "dat", "prmtop",
                ],
            )
            .add_file_filter_extensions(
                "Molecule (small)",
                vec!["mol2", "sdf", "xyz", "pdbqt", "prmtop"],
            )
            .add_file_filter_extensions("Protein (CIF)", vec!["cif"])
            .add_file_filter_extensions("Density", vec!["map", "mtz", "cif"])
            .add_file_filter_extensions("Mol dynamics", vec!["frcmod", "dat", "lib", "prmtop"])
            //
            .add_file_filter_extensions(
                "Molecule (small)",
                vec!["mol2", "sdf", "xyz", "pdbqt", "prmtop"],
            )
            .add_file_filter_extensions("DCD (trajectory)", vec!["dcd"])
            .add_file_filter_extensions("XTC (trajectory)", vec!["xtc"])
            .add_file_filter_extensions("MDT (trajectory)", vec!["mdt"])
            //
            .add_save_extension("Protein (CIF)", "cif")
            .add_save_extension("Mol2", "mol2")
            .add_save_extension("SDF", "sdf")
            .add_save_extension("XYZ", "xyz")
            .add_save_extension("Pdbqt", "pdbqt")
            .add_save_extension("Map", "map")
            .add_save_extension("MTZ", "mtz")
            .add_save_extension("Prmtop", "prmtop")
            .add_save_extension("DCD", "dcd")
            .add_save_extension("XTC", "xtc")
            .add_save_extension("MDT", "mdt"); // Our own trajectory format

        let load = FileDialog::with_config(cfg_all.clone()).default_file_filter("All");
        let save = FileDialog::with_config(cfg_all).default_save_extension("Protein");

        let cfg_screening = FileDialogConfig {
            title: Some("Select screening folder".to_string()),
            ..Default::default()
        };

        let screening = FileDialog::with_config(cfg_screening);

        Self {
            load,
            save,
            screening,
        }
    }
}

/// Flags to accomplish things that must be done somewhere with access to `Scene`.
#[derive(Default)]
struct SceneFlags {
    /// Secondary structure
    pub update_ss_mesh: bool,
    /// Solvent-accessible surface.
    pub update_sas_mesh: bool,
    pub update_sas_coloring: bool,
    pub ss_mesh_created: bool,
    pub sas_mesh_created: bool,
    pub make_density_iso_mesh: bool,
    pub clear_density_drawing: bool,
    pub new_density_loaded: bool,
    pub new_mol_loaded: bool,
}

// todo: Rename A/R
#[derive(Clone, Copy, PartialEq, Default)]
pub enum OperatingMode {
    #[default]
    Primary,
    /// For editing small molecules
    MolEditor,
    /// For editing proteins
    ProteinEditor,
}

// todo: Remove or augment A/R
#[derive(Default)]
pub struct MdStateLocal {
    /// This flag lets us defer launch by a frame, so we can display a flag.
    pub launching: bool,
    pub running: bool,
    pub start: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct Visibility {
    pub hide_sidechains: bool,
    pub hide_water: bool,
    /// Hide hetero atoms: i.e. ones not part of a polypeptide.
    pub hide_hetero: bool,
    pub hide_protein: bool,
    pub hide_ligand: bool,
    pub hide_nucleic_acids: bool,
    pub hide_lipids: bool,
    pub hide_hydrogen: bool,
    pub hide_h_bonds: bool,
    pub dim_peptide: bool,
    pub hide_density_point_cloud: bool,
    pub hide_density_surface: bool,
    pub labels_mol: bool,
    pub labels_atom_sn: bool,
    pub labels_atom_q: bool,
    pub labels_atom_detailed: bool,
    pub labels_bond: bool,
}

impl Default for Visibility {
    fn default() -> Self {
        Self {
            hide_sidechains: false,
            hide_water: false,
            hide_hetero: false,
            hide_protein: false,
            hide_ligand: false,
            hide_nucleic_acids: false,
            hide_lipids: false,
            hide_hydrogen: true,
            hide_h_bonds: false,
            dim_peptide: false,
            hide_density_point_cloud: false,
            hide_density_surface: false,
            labels_mol: true,
            labels_atom_sn: false,
            labels_atom_q: false,
            labels_atom_detailed: false,
            labels_bond: false,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Default, Encode, Decode)]
enum MsaaSetting {
    None = 1,
    // Two = 2, // todo: Not supported on this depth texture, but we could switch to a different one.
    #[default]
    Four = 4,
}

impl MsaaSetting {
    pub fn to_str(self) -> String {
        match self {
            Self::None => "None",
            // Self::Two => "2×",
            Self::Four => "4×",
        }
        .to_owned()
    }
}

#[derive(Default)]
struct PopupState {
    show_get_geostd: bool,
    show_associated_structures: bool,
    show_settings: bool,
    get_geostd_items: Vec<GeostdItem>,
    residue_selector: bool,
    rama_plot: bool,
    recent_files: bool,
    metadata: Option<(MolType, usize)>,
    alignment: bool,
    alignment_screening: bool,
}

struct StateUiMd {
    /// The state we store for this is a float, so we need to store state text too.
    dt_input: String,
    temp_input: String,
    pressure_input: String,
    simbox_pad_input: String,
    langevin_γ: String,
    /// Only perform MD on peptide atoms near a ligand.
    peptide_only_near_ligs: bool,
    /// Peptide atoms don't move, but exert forces.
    peptide_static: bool,
}

impl Default for StateUiMd {
    fn default() -> Self {
        Self {
            dt_input: Default::default(),
            temp_input: Default::default(),
            pressure_input: Default::default(),
            simbox_pad_input: Default::default(),
            langevin_γ: Default::default(),
            peptide_only_near_ligs: true,
            peptide_static: true,
        }
    }
}

#[derive(Clone, PartialEq, Encode, Decode)]
struct LipidUi {
    /// For the combo box. Stays at 0 if none loaded.
    pub lipid_to_add: usize,
    pub shape: LipidShape,
    pub mol_count: u16,
}

impl Default for LipidUi {
    fn default() -> Self {
        Self {
            lipid_to_add: 0,
            shape: Default::default(),
            mol_count: 10,
        }
    }
}

#[derive(Clone, PartialEq, Encode, Decode)]
struct NucleicAcidUi {
    pub seq_to_create: String,
    pub na_type: NucleicAcidType,
    pub strands: Strands,
}

impl Default for NucleicAcidUi {
    fn default() -> Self {
        Self {
            seq_to_create: String::from("ATCG"),
            na_type: Default::default(),
            strands: Default::default(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Default, Debug, Encode, Decode)]
pub enum ResColoring {
    #[default]
    /// A unique color per amino acid, to quickly differentiate them.
    AminoAcid,
    /// Position in sequence, e.g. mapped using viridis
    Position,
    /// Also with a Viridis-style approach.
    Hydrophobicity,
}

impl Display for ResColoring {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let v = match self {
            Self::AminoAcid => "AA",
            Self::Position => "Posit",
            Self::Hydrophobicity => "Hydro",
        };

        write!(f, "{v}")
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
/// For showing and hiding UI sections.
pub struct UiVisibility {
    aa_seq: bool,
    smiles: bool,
    selfies: bool,
    lipids: bool,
    nucleic_acids: bool,
    amino_acids: bool,
    dynamics: bool,
    orca: bool,
    mol_char: bool,
}

impl Default for UiVisibility {
    fn default() -> Self {
        Self {
            aa_seq: false,
            smiles: false,
            selfies: false,
            lipids: false,
            nucleic_acids: false,
            amino_acids: false,
            dynamics: true,
            orca: false,
            mol_char: true, // todo: For now.
        }
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct CamSnapshot {
    // We don't use camera directly, so we don't have to store the projection matrix, and so we can impl
    // Encode/Decode
    pub position: Vec3,
    pub orientation: Quaternion,
    pub far: f32,
    pub name: String,
}

impl CamSnapshot {
    pub fn from_cam(cam: &Camera, name: String) -> Self {
        Self {
            position: cam.position,
            orientation: cam.orientation,
            far: cam.far,
            name,
        }
    }
}

#[derive(Default)]
/// Molecule templates for building-block molecules.
struct Templates {
    /// Common lipid types, e.g. as derived from Amber's `lipids21.lib`, but perhaps not exclusively.
    /// These are loaded at init; there will be one of each type.
    pub lipid: Vec<MoleculeLipid>,
    // pub dna: Vec<MoleculeNucleicAcid>,
    // pub rna: Vec<MoleculeNucleicAcid>,
    pub dna: HashMap<String, TemplateData>,
    pub rna: HashMap<String, TemplateData>,
    // todo: A/R
    pub amino_acid: Vec<MoleculeSmall>,
}

fn main() {
    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;

    #[cfg(feature = "cuda")]
    let (dev, kernel_reflections) = util::get_computation_device();

    // let dev = ComputationDevice::Cpu;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("AVX-512 is available");
        } else if is_x86_feature_detected!("avx") {
            println!("AVX (256-bit) is available");
        } else {
            println!("AVX is not available.");
        }
    }

    println!("Using computing device: {:?}/n", dev);

    // todo: Consider a custom default impl. This is a substitute.
    let mut state = State {
        dev,
        ..Default::default()
    };

    #[cfg(feature = "cuda")]
    if let Some(k) = kernel_reflections {
        state.kernel_reflections = Some(k);
    }

    // todo: Consider if you want this here. Currently required when adding H to a molecule.
    // In release mode, takes 20ms on a fast CPU. (todo: Test on a slow CPU.)
    println!("Loading force field data from Amber lib/dat/frcmod...");
    let start = Instant::now();
    match FfParamSet::new_amber() {
        Ok(f) => {
            state.ff_param_set = f;
            let elapsed = start.elapsed().as_millis();
            println!("Loaded data in {elapsed}ms");
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load Amber force field data: {e:?}"),
            );
        }
    }

    state.load_prefs();

    {
        // Set these UI strings for numerical values up after loading prefs
        state.volatile.md_runtime = state.to_save.num_md_steps as f32 * state.to_save.md_dt;
        state.ui.ph_input = state.to_save.ph.to_string();
        state.ui.md.dt_input = state.to_save.md_dt.to_string();
        state.ui.md.pressure_input = (state.to_save.md_config.pressure_target as u16).to_string();
        state.ui.md.temp_input = (state.to_save.md_config.temp_target as u16).to_string();
        state.ui.md.simbox_pad_input = match state.to_save.md_config.sim_box {
            SimBoxInit::Pad(p) => (p as u16).to_string(),
            SimBoxInit::Fixed(_) => "0".to_string(), // We currently don't use this.
        };

        state.ui.md.langevin_γ = match state.to_save.md_config.integrator {
            Integrator::LangevinMiddle { gamma } => gamma.to_string(),
            _ => "0.".to_string(),
        };
    }

    // We must have loaded prefs prior to this, so we know which file to open.
    state.load_last_opened();

    // todo trouble: It's somewhere around here, saving the inited-from-load atom posits, overwriting
    // todo the previously-saved ones.

    // todo: Workaround to allow us to apply params to the ligand once it's loaded. Unfortunate we have
    // todo to double-load prefs.
    state.load_prefs();

    if let Some(mol) = &state.peptide {
        let posit = state.to_save.per_mol[&mol.common.ident]
            .docking_site
            .site_center;
        state.update_docking_site(posit);
    }

    // todo: Consider if you want this default, and if you also want to add default Lipids etc.
    if !state.ligands.is_empty() {
        state.volatile.active_mol = Some((MolType::Ligand, 0));
    }

    match load_lipid_templates() {
        Ok(t) => {
            state.templates.lipid = t;
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load lipid templates: {e}"),
            );
        }
    }

    match load_na_templates() {
        Ok((dna, rna)) => {
            state.templates.dna = dna;
            state.templates.rna = rna;
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load nucleic acid templates: {e}"),
            );
        }
    }

    // match load_aa_templates() {
    //     Ok(t) => {
    //         state.templates.amino_acid= t;
    //     }
    //     Err(e) => {
    //         handle_err(
    //             &mut state.ui,
    //             format!("Unable to load amino acid templates: {e}"),
    //         );
    //     }
    // }

    if let Ok(out) = Command::new("orca").output() {
        let out = String::from_utf8(out.stdout).unwrap();
        // No simpler way like version?
        if out.contains("This program requires") {
            state.volatile.orca_avail = true;
        }
    };

    render(state);
}
