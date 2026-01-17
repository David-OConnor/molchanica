//! Contains the most important application state structs.

use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    sync::mpsc::Receiver,
};

use bincode::{Decode, Encode};
use bio_apis::{
    ReqError,
    amber_geostd::{GeostdData, GeostdItem},
    pubchem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::md_params::ForceFieldParams;
use cudarc::driver::CudaFunction;
use dynamics::{ComputationDevice, MdState, params::FfParamSet};
use graphics::{Camera, ControlScheme, InputsCommanded, event::Modifiers};
use lin_alg::f64::Vec3 as Vec3F64;

use crate::{
    CamSnapshot, MdStateLocal, OperatingMode, ResColoring, SceneFlags, Templates, UiVisibility,
    drawing::MoleculeView,
    file_io::FileDialogs,
    md::StateUiMd,
    mol_alignment::StateAlignment,
    mol_editor::MolEditorState,
    mol_manip::MolManip,
    molecules::{
        MolGenericRef, MolGenericRefMut, MolIdent, MolType, MoleculePeptide, lipid::MoleculeLipid,
        nucleic_acid::MoleculeNucleicAcid, small::MoleculeSmall,
    },
    orca::StateOrca,
    prefs::ToSave,
    selection::{Selection, ViewSelLevel},
    ui::cam::{FOG_DIST_DEFAULT, VIEW_DEPTH_NEAR_MIN},
};

pub struct State {
    pub ui: StateUi,
    pub volatile: StateVolatile,
    pub cif_pdb_raw: Option<String>,
    // todo: Allow multiple?
    pub peptide: Option<MoleculePeptide>,
    pub ligands: Vec<MoleculeSmall>,
    pub nucleic_acids: Vec<MoleculeNucleicAcid>,
    pub lipids: Vec<MoleculeLipid>,
    pub cam_snapshots: Vec<CamSnapshot>,
    /// This allows us to keep in-memory data for other molecules.
    pub to_save: ToSave,
    /// We store the previous ToSave, to know when we need to write to disk.
    /// Note: This is simpler, but not as efficient as explicitly setting a flag
    /// whenever we change state.
    pub to_save_prev: ToSave,
    pub dev: ComputationDevice,
    /// This is None if Computation Device is CPU.
    #[cfg(feature = "cuda")]
    pub kernel_reflections: Option<CudaFunction>,
    pub mol_dynamics: Option<MdState>,
    // todo: Combine these params in a single struct.
    pub ff_param_set: FfParamSet,
    pub mol_specific_params: HashMap<String, ForceFieldParams>,
    pub templates: Templates,
    pub mol_editor: MolEditorState,
    pub orca: StateOrca,
}

impl Default for State {
    fn default() -> Self {
        // Many other UI defaults are loaded after the initial prefs load in `main`.
        let ui = StateUi {
            view_depth: (VIEW_DEPTH_NEAR_MIN, FOG_DIST_DEFAULT),
            nearby_dist_thresh: 15,
            density_iso_level: 1.8,
            ..Default::default()
        };

        Self {
            ui,
            volatile: Default::default(),
            cif_pdb_raw: Default::default(),
            peptide: Default::default(),
            ligands: Default::default(),
            nucleic_acids: Default::default(),
            lipids: Default::default(),
            cam_snapshots: Default::default(),
            to_save: Default::default(),
            to_save_prev: Default::default(),
            dev: Default::default(),
            #[cfg(feature = "cuda")]
            kernel_reflections: None,
            mol_dynamics: Default::default(),
            ff_param_set: Default::default(),
            mol_specific_params: Default::default(),
            templates: Default::default(),
            mol_editor: Default::default(),
            orca: Default::default(),
        }
    }
}

impl State {
    /// E.g. when loading a new molecule.
    pub fn reset_selections(&mut self) {
        self.ui.selection = Selection::None;
        self.cam_snapshots = Vec::new();
        self.ui.cam_snapshot = None;
        self.ui.chain_to_pick_res = None;
    }

    // todo: Re-implement with v2 A/R
    // /// Gets the docking setup, creating it if it doesn't exist. Returns `None` if molecule
    // /// or ligand are absent.
    // pub fn get_make_docking_setup(&mut self) -> Option<&DockingSetup> {
    //     None
    //     let (Some(mol), Some(lig)) = (&self.molecule, &mut self.ligand) else {
    //         return None;
    //     };
    //
    //     Some(self.volatile.docking_setup.get_or_insert_with(|| {
    //         DockingSetup::new(mol, lig, &self.volatile.lj_lookup_table, &self.bh_config)
    //     }))
    // }

    pub fn update_docking_site(&mut self, posit: Vec3F64) {
        // if let Some(lig) = &mut self.ligand {
        //     if let Some(data) = &mut lig.lig_data {
        //         data.docking_site.site_center = posit;
        //
        //         self.ui.docking_site_x = posit.x.to_string();
        //         self.ui.docking_site_y = posit.y.to_string();
        //         self.ui.docking_site_z = posit.z.to_string();
        //     }
        // }
    }

    /// Helper
    pub fn active_mol(&self) -> Option<MolGenericRef<'_>> {
        match self.volatile.active_mol {
            Some((mol_type, i)) => match mol_type {
                MolType::Peptide => {
                    if self.peptide.is_some() {
                        Some(MolGenericRef::Peptide(&self.peptide.as_ref().unwrap()))
                    } else {
                        None
                    }
                }
                MolType::Ligand => {
                    if i < self.ligands.len() {
                        Some(MolGenericRef::Small(&self.ligands[i]))
                    } else {
                        None
                    }
                }
                MolType::NucleicAcid => {
                    if i < self.nucleic_acids.len() {
                        Some(MolGenericRef::NucleicAcid(&self.nucleic_acids[i]))
                    } else {
                        None
                    }
                }
                MolType::Lipid => {
                    if i < self.lipids.len() {
                        Some(MolGenericRef::Lipid(&self.lipids[i]))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            None => None,
        }
    }

    /// Helper
    /// todo: DRy with the non-mutable variant.
    pub fn active_mol_mut(&mut self) -> Option<MolGenericRefMut<'_>> {
        match self.volatile.active_mol {
            Some((mol_type, i)) => match mol_type {
                MolType::Peptide => None,
                MolType::Ligand => {
                    if i < self.ligands.len() {
                        Some(MolGenericRefMut::Small(&mut self.ligands[i]))
                    } else {
                        None
                    }
                }
                MolType::NucleicAcid => {
                    if i < self.nucleic_acids.len() {
                        Some(MolGenericRefMut::NucleicAcid(&mut self.nucleic_acids[i]))
                    } else {
                        None
                    }
                }
                MolType::Lipid => {
                    if i < self.lipids.len() {
                        Some(MolGenericRefMut::Lipid(&mut self.lipids[i]))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            None => None,
        }
    }
}

/// Temporary, and generated state.
pub struct StateVolatile {
    pub dialogs: FileDialogs,
    // /// Center and size are used for setting the camera. Dependent on the molecule atom positions.
    // mol_center: Vec3,
    // mol_size: f32, // Dimension-agnostic
    /// We Use this to keep track of key press state for the camera movement, so we can continuously
    /// update the flashlight when moving.
    pub inputs_commanded: InputsCommanded,
    // todo: Replace with the V2 version A/R
    // docking_setup: Option<DockingSetup>,
    /// Receives thread data upon an HTTP result completion.
    pub mol_pending_data_avail: Option<
        Receiver<(
            Result<PdbDataResults, ReqError>,
            Result<FilesAvailable, ReqError>,
        )>,
    >,
    /// Receives thread data upon an HTTP result completion.
    pub pubchem_properties_avail:
        Option<Receiver<(MolIdent, Result<pubchem::Properties, ReqError>)>>,
    /// The first param is the index.
    pub amber_geostd_data_avail: Option<Receiver<(usize, Result<GeostdData, ReqError>)>>,
    /// We may change CWD during CLI navigation; keep prefs directory constant.
    pub prefs_dir: PathBuf,
    /// Entered by the user, for this session.
    pub cli_input_history: Vec<String>,
    pub cli_input_selected: usize,
    /// Pre-computed from the molecule
    pub aa_seq_text: String,
    pub flags: SceneFlags,
    /// Cached so we don't compute each UI paint. Picoseconds.
    pub md_runtime: f32,
    pub active_mol: Option<(MolType, usize)>,
    pub mol_manip: MolManip,
    /// For restoring after temprarily disabling mouse look.
    pub control_scheme_prev: ControlScheme,
    /// We maintain a set of atom indices of peptides that are used in MD. This for example, might
    /// exclude hetero atoms and atoms not near a docking site. (mol i, atom i)
    pub md_peptide_selected: HashSet<(usize, usize)>,
    /// Ctrl, alt, shift etc.
    pub key_modifiers: Modifiers,
    pub operating_mode: OperatingMode,
    /// Allows restoring after entering the mol edit mode.
    pub primary_mode_cam: Camera,
    pub mol_editing: Option<usize>,
    pub md_local: MdStateLocal,
    pub orbit_center: Option<(MolType, usize)>,
    /// ORCA is available on the system path.
    pub orca_avail: bool,
    // /// Per-protein. Computed as required; None before then.
    // hydropathy_data: Option<Vec<Vec<(usize, usize)>>>,
    // /// If present, there must be one per vertex. Rebuild this whenever we
    // /// rebuild this mesh.
    // sa_surface_mesh_colors: Option<Vec<(u8, u8, u8)>>,
    /// Outer the protein index. Inner: A collection of points on the surface, sufficient to
    /// determine if a given atom is near the surface.
    protein_sfc_mesh_coarse: Vec<Vec<f32>>,
    pub alignment: StateAlignment,
}

impl Default for StateVolatile {
    fn default() -> Self {
        Self {
            dialogs: Default::default(),
            inputs_commanded: Default::default(),
            pubchem_properties_avail: Default::default(),
            mol_pending_data_avail: Default::default(),
            amber_geostd_data_avail: Default::default(),
            prefs_dir: env::current_dir().unwrap(), // This is why we can't derive.
            cli_input_history: Default::default(),
            cli_input_selected: Default::default(),
            aa_seq_text: Default::default(),
            flags: Default::default(),
            md_runtime: Default::default(),
            active_mol: Default::default(),
            mol_manip: Default::default(),
            control_scheme_prev: Default::default(),
            md_peptide_selected: Default::default(),
            key_modifiers: Default::default(),
            operating_mode: Default::default(),
            primary_mode_cam: Default::default(),
            mol_editing: Default::default(),
            md_local: Default::default(),
            orbit_center: None,
            orca_avail: Default::default(),
            protein_sfc_mesh_coarse: Default::default(),
            alignment: Default::default(),
        }
    }
}

/// Ui text fields and similar.
#[derive(Default)]
pub struct StateUi {
    pub mol_view: MoleculeView,
    pub view_sel_level: ViewSelLevel,
    /// Mouse cursor
    pub cursor_pos: Option<(f32, f32)>,
    pub db_input: String,
    pub cam_snapshot_name: String,
    pub atom_res_search: String,
    /// To selection.
    pub show_near_sel_only: bool,
    pub show_near_lig_only: bool,
    /// Protein atoms near its surface; hide internal ones.
    pub show_near_sfc_only: bool,
    /// Angstrom. For selections, or ligand.
    pub nearby_dist_thresh: u16,
    pub view_depth: (u16, u16), // angstrom. min, max.
    pub cam_snapshot: Option<usize>,
    pub dt_render: f32, // Seconds
    // For selecting residues from the GUI.
    pub chain_to_pick_res: Option<usize>,
    /// Workaround for a bug or limitation in EGUI's `is_pointer_button_down_on`.
    // inputs_commanded: InputsCommanded,
    pub visibility: Visibility,
    pub selection: Selection,
    pub left_click_down: bool,
    pub middle_click_down: bool,
    pub autodock_path_valid: bool,
    pub mouse_in_window: bool,
    pub docking_site_x: String,
    pub docking_site_y: String,
    pub docking_site_z: String,
    pub docking_site_size: String,
    /// For the arc/orbit cam only.
    pub orbit_selected_atom: bool,
    // todo: Re-implement A/R
    // binding_energy_disp: Option<BindingEnergy>,
    pub current_snapshot: usize,
    /// A flag so we know to update the flashlight upon loading a new model; this should be done within
    /// a callback.
    pub show_docking_tools: bool,
    pub movement_speed_input: String,
    pub rotation_sens_input: String,
    pub mol_move_sens_input: String, // scroll
    pub cmd_line_input: String,
    pub cmd_line_output: String,
    /// Indicates CLI, or errors more broadly by changing its displayed color.
    pub cmd_line_out_is_err: bool,
    pub ui_vis: UiVisibility,
    /// Use a viridis or simialar colr scheme to color residues gradually based on their
    /// position in the sequence.
    pub res_coloring: ResColoring,
    pub atom_color_by_charge: bool,
    /// Affects the electron density mesh.
    pub density_iso_level: f32,
    // /// E.g. set to original for from the mmCIF file, or Dynamics to view it after MD.
    // peptide_atom_posits: PeptideAtomPosits,
    pub popup: PopupState,
    pub md: StateUiMd,
    pub ph_input: String,
    /// If true, the surface mesh is colored according to the atom or residue colors closest to
    /// it. (E.g. CPK, by partial charge, by hydrophobicity etc). If false, it's a solid color.
    pub color_surface_mesh: bool,
    /// Color ligands by molecule, to contrast.
    pub color_by_mol: bool,
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

#[derive(Default)]
pub struct PopupState {
    pub show_get_geostd: bool,
    pub show_associated_structures: bool,
    pub show_settings: bool,
    pub get_geostd_items: Vec<GeostdItem>,
    pub residue_selector: bool,
    pub rama_plot: bool,
    pub recent_files: bool,
    pub metadata: Option<(MolType, usize)>,
    pub alignment: bool,
    pub alignment_screening: bool,
}
