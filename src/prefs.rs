//! Code related to saving user *preferences*. e.g. opened molecules, view configuration etc.
//!
//! todo: Get away from bincode. Divide into packets, so perhaps there is some breaking for
//! todo individual packet changes, but teh whole thing can't get broken by a small change to
//! todo the format.

use std::{
    collections::HashMap,
    fmt::Display,
    path::{Path, PathBuf},
};

use bincode::{
    BorrowDecode, Decode, Encode,
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
};
use bio_apis::{
    pubchem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use chrono::{DateTime, TimeZone, Utc};
use dynamics::MdConfig;
use graphics::{
    ControlScheme,
    app_utils::{load, save},
};
use lin_alg::f64::Vec3;

use crate::{
    docking::DockingSite,
    drawing::MoleculeView,
    inputs::{MOVEMENT_SENS, ROTATE_SENS, SENS_MOL_MOVE_SCROLL},
    molecules::MolIdent,
    selection::{Selection, ViewSelLevel},
    sfc_mesh::MeshColoring,
    state::{
        CamSnapshot, LipidUi, MsaaSetting, NucleicAcidUi, ResColoring, State, UiVisibility,
        Visibility,
    },
};

pub const DEFAULT_PREFS_FILE: &str = "molchanica_prefs.mca";

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
// For now, we check for differences between to_save and to_save prev, and write to disk
// if they're not equal.
pub const PREFS_SAVE_INTERVAL: u64 = 20; // seconds

#[macro_export]
macro_rules! parse_le {
    ($bytes:expr, $t:ty, $range:expr) => {{ <$t>::from_le_bytes($bytes[$range].try_into().unwrap()) }};
}

#[macro_export]
macro_rules! copy_le {
    ($dest:expr, $src:expr, $range:expr) => {{ $dest[$range].copy_from_slice(&$src.to_le_bytes()) }};
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum PrefsPacketType {
    OpenHistory = 0,
    Misc = 1,
}

pub struct PrefsPacketHeader {
    pub packet_type: PrefsPacketType,
    // In bytes
    pub len: usize,
}

impl PrefsPacketHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = vec![0; 5];
        result[0] = self.packet_type as u8;

        copy_le!(result, self.len as u32, 1..5);

        result
    }
}

/// Used to sequence how we handle each file type.
#[derive(Clone, Copy, PartialEq, Debug, Encode, Decode)]
pub enum OpenType {
    Peptide,
    Ligand,
    NucleicAcid,
    Lipid,
    Pocket,
    Map,
    Frcmod,
    Trajectory,
}

impl Display for OpenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Peptide => "Peptide",
            Self::Ligand => "Small mol",
            Self::NucleicAcid => "Nucleic Acid",
            Self::Lipid => "Lipid",
            Self::Pocket => "Pocket",
            Self::Map => "Density map",
            Self::Frcmod => "Frcmod",
            Self::Trajectory => "MD Trajectory",
        };
        write!(f, "{v}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenHistory {
    pub timestamp: DateTime<Utc>,
    pub path: PathBuf,
    pub type_: OpenType,
    /// Only applicable for molecules and similar. A single central position.
    /// todo: We may wish to add per-atom positions later.
    pub position: Option<Vec3>,
    /// For determining which to open at program start.
    pub last_session: bool,
}

impl OpenHistory {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        result
    }

    // pub fn from_bytes(bytes: &[u8]) -> Self {
    //
    // }
}

// Manual bincode impls
impl Encode for OpenHistory {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // Store timestamp as i64 (seconds since Unix epoch)
        let ts = self.timestamp.timestamp();
        ts.encode(encoder)?;

        self.path.encode(encoder)?;
        self.type_.encode(encoder)?;
        self.position.encode(encoder)?;
        self.last_session.encode(encoder)?;

        Ok(())
    }
}

impl<T> Decode<T> for OpenHistory {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let ts = i64::decode(decoder)?;
        let path = PathBuf::decode(decoder)?;
        let type_ = OpenType::decode(decoder)?;
        let position = Option::<Vec3>::decode(decoder)?;
        let last_session = bool::decode(decoder)?;

        Ok(OpenHistory {
            timestamp: Utc
                .timestamp_opt(ts, 0)
                .single()
                .ok_or_else(|| DecodeError::OtherString("invalid timestamp".to_string()))?,
            path,
            type_,
            position,
            last_session,
        })
    }
}

impl<'de, C> BorrowDecode<'de, C> for OpenHistory {
    fn borrow_decode<D>(decoder: &mut D) -> Result<Self, DecodeError>
    where
        D: BorrowDecoder<'de, Context = C>,
    {
        Decode::decode(decoder)
    }
}

impl OpenHistory {
    // pub fn new(path: &Path, type_: OpenType, position: Option<Vec3>) -> Self {
    pub fn new(path: &Path, type_: OpenType) -> Self {
        Self {
            timestamp: Utc::now(),
            path: path.to_owned(),
            type_,
            position: None,
            last_session: true,
        }
    }
}

/// We maintain some of the state that is saved in the preferences file here, to keep
/// the save/load state streamlined, instead of in an intermediate struct between main state, and
/// saving/loading.
///
/// Note that this contains things which often fit cleanly in other state structs like `StateUi`,
/// but are here due to our save mechanic.
#[derive(Clone, PartialEq, Encode, Decode)]
pub struct ToSave {
    pub per_mol: HashMap<String, PerMolToSave>,
    pub open_history: Vec<OpenHistory>,
    pub control_scheme: ControlScheme,
    pub msaa: MsaaSetting,
    /// Direct conversion from engine standard
    pub movement_speed: u8,
    /// Divide this by 100 to get engine standard.
    pub rotation_sens: u8,
    pub mol_move_sens: u8,
    /// Solvent-accessible surface (and dots) precision. Lower is higher precision. A value of 0.5 - 0.6
    /// is a good default. Too low will cause crashes and very poor performance. Higher is too coarse.
    pub sa_surface_precision: f32,
    pub md_config: MdConfig,
    pub num_md_steps: u32,
    /// todo: We may move this to be per-molecule, etc.
    pub num_md_copies: usize,
    /// ps (10^-12). Typical values are 0.001 or 0.002.
    pub md_dt: f32,
    pub ph: f32,
    pub selection: Selection,
    pub cam_snapshots: Vec<CamSnapshot>,
    pub mol_view: MoleculeView,
    pub view_sel_level: ViewSelLevel,
    pub visibility: Visibility,
    pub ui_visibility: UiVisibility,
    pub near_sel_only: bool,
    pub near_lig_only: bool,
    pub nearby_dist_thresh: u16,
    pub pubchem_properties_map: HashMap<MolIdent, pubchem::Properties>,
    /// We use this to mark that we should perform a save, even if one
    /// of the fields here didn't change. E.g. after moving a molecule.
    /// This is caught during our periodic check for changes in this struct.
    pub save_flag: bool,
    pub lipid: LipidUi,
    pub nucleic_acid: NucleicAcidUi,
    pub mesh_coloring: MeshColoring,
    pub screening_path: Option<PathBuf>,
}

impl Default for ToSave {
    fn default() -> Self {
        Self {
            per_mol: Default::default(),
            open_history: Default::default(),
            control_scheme: Default::default(),
            msaa: Default::default(),
            movement_speed: MOVEMENT_SENS as u8,
            rotation_sens: (ROTATE_SENS * 100.) as u8,
            mol_move_sens: (SENS_MOL_MOVE_SCROLL * 1_000.) as u8,
            // We override this based on protein atom count.
            sa_surface_precision: 0.55,
            md_config: Default::default(),
            num_md_steps: 100,
            num_md_copies: 1,
            md_dt: 0.002,
            ph: 7.4,
            selection: Default::default(),
            cam_snapshots: Default::default(),
            mol_view: Default::default(),
            view_sel_level: Default::default(),
            near_sel_only: Default::default(),
            near_lig_only: Default::default(),
            nearby_dist_thresh: Default::default(),
            visibility: Default::default(),
            ui_visibility: Default::default(),
            pubchem_properties_map: Default::default(),
            save_flag: false,
            lipid: Default::default(),
            nucleic_acid: Default::default(),
            mesh_coloring: Default::default(),
            screening_path: None,
        }
    }
}

/// Generally, data here only applies if a protein is present.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct PerMolToSave {
    chain_vis: Vec<bool>,
    chain_to_pick_res: Option<usize>,
    pub docking_site: DockingSite,
    show_docking_tools: bool,
    res_coloring: ResColoring,
    aatom_color_by_charge: bool,
    show_aa_seq: bool,
    rcsb_data: Option<PdbDataResults>,
    rcsb_files_avail: Option<FilesAvailable>,
    docking_site_posit: Vec3,
    /// This is useful in the case of absolute positions. Ideally, this is per-ligand.
    /// todo: This needs a rework with your generalizations.
    lig_atom_positions: Vec<Vec3>,
}

impl PerMolToSave {
    pub fn from_state(state: &State) -> Self {
        let mut chain_vis = Vec::new();
        let mut rcsb_data = None;
        let mut rcsb_files_avail = None;

        if let Some(mol) = &state.peptide {
            chain_vis = mol.chains.iter().map(|c| c.visible).collect();

            rcsb_data = mol.rcsb_data.clone();
            rcsb_files_avail = mol.rcsb_files_avail.clone();
        }

        let docking_site = Default::default();

        let lig_posit = Vec3::new_zero();
        let lig_atom_positions = Vec::new();

        // todo: Hmm. No longer needed? Never worked?
        // if let Some(mol) = state.active_mol() {
        //     // Don't save this if on init; the data in lig is the default,
        //     // and we haven't loaded the posits to it yet.
        //     // todo: If you find a more robust way to handle saving order-dependent data,
        //     // todo: remove this.
        //     if !on_init {
        //         lig_atom_positions = mol.common().atom_posits.clone();
        //         if !mol.common().atom_posits.is_empty() {
        //             println!("\nSaving atom posits: {:?}", mol.common().atom_posits[0]); // todo temp
        //         }
        //     }
        // }

        Self {
            chain_vis,
            chain_to_pick_res: state.ui.chain_to_pick_res,
            // metadata,
            docking_site,
            show_docking_tools: state.ui.show_docking_tools,
            res_coloring: state.ui.res_coloring,
            aatom_color_by_charge: state.ui.atom_color_by_charge,
            show_aa_seq: state.ui.ui_vis.aa_seq,
            rcsb_data,
            rcsb_files_avail,
            docking_site_posit: lig_posit,
            lig_atom_positions,
        }
    }
}

impl State {
    /// We run this after loading a molecule.
    pub fn update_save_prefs_no_mol(&mut self) {
        if let Err(e) = save(
            &self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE),
            &self.to_save,
        ) {
            eprintln!("Error saving state: {e:?}");
        }
    }

    /// Update when prefs change, periodically etc.
    /// todo: See the note in PerMolsave::from_state. Workaround for order-related bugs.
    pub fn update_save_prefs(&mut self) {
        println!("Saving state to prefs file");

        // Sync molecule positions.
        // todo: Consider the same for proteins, and other mol types.
        for mol in &self.ligands {
            for oh in &mut self.to_save.open_history {
                if let Some(p) = &mol.common.path {
                    if &oh.path == p {
                        oh.position = Some(mol.common.centroid());
                    }
                }
            }
        }
        for mol in &self.pockets {
            for oh in &mut self.to_save.open_history {
                if let Some(p) = &mol.common.path {
                    if &oh.path == p {
                        oh.position = Some(mol.common.centroid());
                    }
                }
            }
        }

        if let Some(mol) = &self.peptide {
            let data = PerMolToSave::from_state(self);

            self.to_save.per_mol.insert(mol.common.ident.clone(), data);
        }

        self.to_save.selection = self.ui.selection.clone();
        self.to_save.cam_snapshots = self.cam_snapshots.clone();
        self.to_save.mol_view = self.ui.mol_view;
        self.to_save.view_sel_level = self.ui.view_sel_level;
        self.to_save.near_sel_only = self.ui.show_near_sel_only;
        self.to_save.near_lig_only = self.ui.show_near_lig_only;
        self.to_save.nearby_dist_thresh = self.ui.nearby_dist_thresh;
        self.to_save.visibility = self.ui.visibility.clone();
        self.to_save.ui_visibility = self.ui.ui_vis.clone();
        self.to_save.mesh_coloring = self.ui.mesh_coloring;

        self.to_save_prev = self.to_save.clone();

        if let Err(e) = save(
            &self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE),
            &self.to_save,
        ) {
            eprintln!("Error saving state: {e:?}");
        }
    }

    /// Run this when prefs, or a new molecule are loaded.
    pub fn update_from_prefs(&mut self) {
        println!("Updating state from prefs data");
        self.reset_selections();

        if let Some(mol) = &mut self.peptide {
            if self.to_save.per_mol.contains_key(&mol.common.ident) {
                let data = &self.to_save.per_mol[&mol.common.ident];

                self.ui.chain_to_pick_res = data.chain_to_pick_res;
                self.ui.show_docking_tools = data.show_docking_tools;
                self.ui.res_coloring = data.res_coloring;
                self.ui.atom_color_by_charge = data.aatom_color_by_charge;
                self.ui.ui_vis.aa_seq = data.show_aa_seq;

                for (i, chain) in mol.chains.iter_mut().enumerate() {
                    if i < data.chain_vis.len() {
                        chain.visible = data.chain_vis[i];
                    }
                }

                mol.rcsb_data = data.rcsb_data.clone();
                mol.rcsb_files_avail = data.rcsb_files_avail.clone();
            }
        }

        self.ui.selection = self.to_save.selection.clone();
        self.cam_snapshots = self.to_save.cam_snapshots.clone();
        self.ui.mol_view = self.to_save.mol_view;
        self.ui.view_sel_level = self.to_save.view_sel_level;
        self.ui.show_near_sel_only = self.to_save.near_sel_only;
        self.ui.show_near_lig_only = self.to_save.near_lig_only;
        self.ui.nearby_dist_thresh = self.to_save.nearby_dist_thresh;
        self.ui.visibility = self.to_save.visibility.clone();
        self.ui.ui_vis = self.to_save.ui_visibility.clone();

        self.ui.movement_speed_input = self.to_save.movement_speed.to_string();
        self.ui.rotation_sens_input = self.to_save.rotation_sens.to_string();
        self.ui.mol_move_sens_input = self.to_save.mol_move_sens.to_string();
        self.ui.mesh_coloring = self.to_save.mesh_coloring;

        println!("Done");
    }

    pub fn load_prefs(&mut self) {
        match load(&PathBuf::from(DEFAULT_PREFS_FILE)) {
            Ok(p) => self.to_save = p,
            Err(e) => {
                eprintln!("Unable to load save file; possibly the first time running: {e:?}.")
            }
        }

        self.update_from_prefs();
    }
}
