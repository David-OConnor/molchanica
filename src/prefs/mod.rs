//! Code related to saving user *preferences*. e.g. opened molecules, view configuration etc.
//!
//! todo: Get away from bincode. Divide into packets, so perhaps there is some breaking for
//! todo individual packet changes, but teh whole thing can't get broken by a small change to
//! todo the format.

use std::{
    collections::HashMap,
    fmt::Display,
    io,
    path::{Path, PathBuf},
};

use bio_apis::{
    pubchem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use chrono::{DateTime, Utc};
use dynamics::MdConfig;
use graphics::{AmbientOcclusion, ControlScheme};
use lin_alg::f64::Vec3;

mod file_format;

use crate::{
    docking::DockingSite,
    drawing::MoleculeView,
    inputs::{MOVEMENT_SENS, ROTATE_SENS, SENS_MOL_MOVE_SCROLL},
    md::MdBackend,
    molecules::{MolIdent, MolType, peptide::MoleculePeptide},
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

/// Used to sequence how we handle each file type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OpenType {
    Peptide,
    Ligand,
    NucleicAcid,
    Lipid,
    Pocket,
    Map,
    Frcmod,
    Trajectory,
    MdParams,
    ParquetDb,
    MdMols,
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
            Self::MdParams => "MD Params",
            Self::ParquetDb => "Parquet DB",
            Self::MdMols => "MD Mols",
        };
        write!(f, "{v}")
    }
}

impl From<MolType> for OpenType {
    fn from(v: MolType) -> Self {
        use MolType::*;

        match v {
            Peptide => Self::Peptide,
            Ligand => Self::Ligand,
            NucleicAcid => Self::NucleicAcid,
            Lipid => Self::Lipid,
            Pocket => Self::Pocket,
            Water => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenHistory {
    pub timestamp: DateTime<Utc>,
    pub path: PathBuf,
    pub type_: OpenType,
    pub ident: Option<String>,
    /// Only applicable for molecules and similar. A single central position.
    /// todo: We may wish to add per-atom positions later.
    pub position: Option<Vec3>,
    /// For determining which to open at program start.
    pub last_session: bool,
}

impl OpenHistory {
    // pub fn new(path: &Path, type_: OpenType, position: Option<Vec3>) -> Self {
    pub fn new(path: &Path, type_: OpenType, ident: Option<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            path: path.to_owned(),
            ident,
            type_,
            position: None,
            last_session: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ControlSchemeType {
    Free,
    Arc,
}

impl ControlSchemeType {
    /// I.e. without a default Arc center.
    pub fn to_scheme_default(self) -> ControlScheme {
        match self {
            Self::Free => ControlScheme::FreeCamera,
            Self::Arc => ControlScheme::Arc {
                center: lin_alg::f32::Vec3::new_zero(),
            },
        }
    }
}
#[derive(Clone, PartialEq)]
pub struct Graphics {
    pub msaa: MsaaSetting,
    pub ambient_occlusion: AmbientOcclusion,
    pub edge_cueing: Option<f32>,
    pub depth_aware_halos: Option<f32>,
}

impl Default for Graphics {
    fn default() -> Self {
        Self {
            msaa: Default::default(),
            ambient_occlusion: Default::default(),
            edge_cueing: Some(1.),
            depth_aware_halos: Some(0.03),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct ControlSettings {
    /// Direct conversion from engine standard
    pub movement_speed: u8,
    /// Divide this by 100 to get engine standard.
    pub rotation_sens: u8,
    pub mol_move_sens: u8,
}

impl Default for ControlSettings {
    fn default() -> Self {
        Self {
            movement_speed: MOVEMENT_SENS as u8,
            rotation_sens: (ROTATE_SENS * 100.) as u8,
            mol_move_sens: (SENS_MOL_MOVE_SCROLL * 1_000.) as u8,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct MdPrefs {
    pub config: MdConfig,
    pub num_steps: u32,
    pub dt: f32,
    pub backend: MdBackend,
}

impl Default for MdPrefs {
    fn default() -> Self {
        Self {
            config: Default::default(),
            num_steps: 100,
            dt: 0.002,
            backend: Default::default(),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct UiPrefs {
    pub selection: Selection,
    pub cam_snapshots: Vec<CamSnapshot>,
    pub mol_view: MoleculeView,
    pub mol_view_peptide: MoleculeView,
    pub view_sel_level: ViewSelLevel,
    pub visibility: Visibility,
    pub ui_visibility: UiVisibility,
    pub near_sel_only: bool,
    pub near_lig_only: bool,
    pub nearby_dist_thresh: u16,
}

impl Default for UiPrefs {
    fn default() -> Self {
        Self {
            selection: Default::default(),
            cam_snapshots: Default::default(),
            mol_view: MoleculeView::DEFAULT_NON_PEPTIDE,
            mol_view_peptide: MoleculeView::default(),
            view_sel_level: Default::default(),
            near_sel_only: Default::default(),
            near_lig_only: Default::default(),
            nearby_dist_thresh: Default::default(),
            visibility: Default::default(),
            ui_visibility: Default::default(),
        }
    }
}

/// We maintain some of the state that is saved in the preferences file here, to keep
/// the save/load state streamlined, instead of in an intermediate struct between main state, and
/// saving/loading.
///
/// Note that this contains things which often fit cleanly in other state structs like `StateUi`,
/// but are here due to our save mechanic.
#[derive(Clone, PartialEq)]
pub struct ToSave {
    pub per_mol: HashMap<String, PerMolToSave>,
    pub open_history: Vec<OpenHistory>,
    pub control_scheme: ControlSchemeType,
    pub graphics: Graphics,
    pub control_settings: ControlSettings,
    /// Solvent-accessible surface (and dots) precision. Lower is higher precision. A value of 0.5 - 0.6
    /// is a good default. Too low will cause crashes and very poor performance. Higher is too coarse.
    pub sa_surface_precision: f32,
    pub md: MdPrefs,
    pub ui_prefs: UiPrefs,
    pub ph: f32,
    pub pubchem_properties_map: HashMap<MolIdent, pubchem::Properties>,
    /// We use this to mark that we should perform a save, even if one
    /// of the fields here didn't change. E.g. after moving a molecule.
    /// This is caught during our periodic check for changes in this struct.
    pub save_flag: bool,
    pub lipid: LipidUi,
    pub nucleic_acid: NucleicAcidUi,
    pub mesh_coloring: MeshColoring,
    pub auto_fog: bool,
}

impl Default for ToSave {
    fn default() -> Self {
        Self {
            per_mol: Default::default(),
            open_history: Default::default(),
            control_scheme: ControlSchemeType::Free,
            graphics: Default::default(),
            control_settings: Default::default(),
            // We override this based on protein atom count.
            sa_surface_precision: 0.55,
            md: Default::default(),
            ui_prefs: Default::default(),
            ph: 7.4,
            pubchem_properties_map: Default::default(),
            save_flag: false,
            lipid: Default::default(),
            nucleic_acid: Default::default(),
            mesh_coloring: Default::default(),
            auto_fog: true,
        }
    }
}

/// Generally, data here only applies if a protein is present.
#[derive(Debug, Clone, PartialEq)]
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
    pub fn from_state(state: &State, mol: &MoleculePeptide) -> Self {
        let chain_vis = mol.chains.iter().map(|c| c.visible).collect();
        let rcsb_data = mol.rcsb_data.clone();
        let rcsb_files_avail = mol.rcsb_files_avail.clone();

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

    // Hand-rolled (de)serialization, inline with the `copy_le!` / `parse_le!` macros. Defined here,
    // rather than in `prefs_file_format`, so it can access this struct's private fields. The
    // deeply-nested external types (`PdbDataResults`, `FilesAvailable`, `Vec3`) are stored as
    // length-prefixed bincode blobs; everything else is hand-rolled.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();

        out.extend_from_slice(&(self.chain_vis.len() as u32).to_le_bytes());
        for v in &self.chain_vis {
            out.push(*v as u8);
        }

        match self.chain_to_pick_res {
            Some(x) => {
                out.push(1);
                out.extend_from_slice(&(x as u64).to_le_bytes());
            }
            None => out.push(0),
        }

        let ds = self.docking_site.to_bytes();
        out.extend_from_slice(&(ds.len() as u32).to_le_bytes());
        out.extend_from_slice(&ds);

        out.push(self.show_docking_tools as u8);
        out.push(self.res_coloring.to_u8());
        out.push(self.aatom_color_by_charge as u8);
        out.push(self.show_aa_seq as u8);

        match &self.rcsb_data {
            Some(x) => {
                out.push(1);
                let b = bincode::encode_to_vec(x, bincode::config::standard()).unwrap_or_default();
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(&b);
            }
            None => out.push(0),
        }
        match &self.rcsb_files_avail {
            Some(x) => {
                out.push(1);
                let b = bincode::encode_to_vec(x, bincode::config::standard()).unwrap_or_default();
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(&b);
            }
            None => out.push(0),
        }

        let c = bincode::encode_to_vec(&self.docking_site_posit, bincode::config::standard())
            .unwrap_or_default();
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
        out.extend_from_slice(&c);
        let l = bincode::encode_to_vec(&self.lig_atom_positions, bincode::config::standard())
            .unwrap_or_default();
        out.extend_from_slice(&(l.len() as u32).to_le_bytes());
        out.extend_from_slice(&l);

        out
    }

    #[allow(unused_assignments)]
    pub(crate) fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut i = 0;

        let n = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let mut chain_vis = Vec::with_capacity(n);
        for _ in 0..n {
            chain_vis.push(data[i] != 0);
            i += 1;
        }

        let has = data[i] != 0;
        i += 1;
        let chain_to_pick_res = if has {
            let x = parse_le!(data, u64, i..i + 8) as usize;
            i += 8;
            Some(x)
        } else {
            None
        };

        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let docking_site = DockingSite::from_bytes(&data[i..i + len])?;
        i += len;

        let show_docking_tools = data[i] != 0;
        i += 1;
        let res_coloring = ResColoring::from_u8(data[i]);
        i += 1;
        let aatom_color_by_charge = data[i] != 0;
        i += 1;
        let show_aa_seq = data[i] != 0;
        i += 1;

        let has = data[i] != 0;
        i += 1;
        let rcsb_data = if has {
            let len = parse_le!(data, u32, i..i + 4) as usize;
            i += 4;
            let v = bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
            i += len;
            Some(v)
        } else {
            None
        };
        let has = data[i] != 0;
        i += 1;
        let rcsb_files_avail = if has {
            let len = parse_le!(data, u32, i..i + 4) as usize;
            i += 4;
            let v = bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
            i += len;
            Some(v)
        } else {
            None
        };

        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let docking_site_posit =
            bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;
        i += len;
        let len = parse_le!(data, u32, i..i + 4) as usize;
        i += 4;
        let lig_atom_positions =
            bincode::decode_from_slice(&data[i..i + len], bincode::config::standard())
                .unwrap()
                .0;

        Ok(Self {
            chain_vis,
            chain_to_pick_res,
            docking_site,
            show_docking_tools,
            res_coloring,
            aatom_color_by_charge,
            show_aa_seq,
            rcsb_data,
            rcsb_files_avail,
            docking_site_posit,
            lig_atom_positions,
        })
    }
}

impl State {
    /// We run this after loading a molecule.
    pub fn update_save_prefs_no_mol(&mut self) {
        if let Err(e) = self
            .to_save
            .save(&self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE))
        {
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
                if let Some(p) = &mol.common.path
                    && &oh.path == p
                {
                    oh.position = Some(mol.common.centroid());
                }
            }
        }

        for mol in &self.peptide {
            for oh in &mut self.to_save.open_history {
                if let Some(p) = &mol.common.path
                    && &oh.path == p
                {
                    oh.position = Some(mol.common.centroid());
                }
            }
        }

        for mol in &self.pockets {
            for oh in &mut self.to_save.open_history {
                if let Some(p) = &mol.common.path
                    && &oh.path == p
                {
                    oh.position = Some(mol.common.centroid());
                }
            }
        }

        for mol in &self.peptide {
            let data = PerMolToSave::from_state(self, mol);
            self.to_save.per_mol.insert(mol.common.ident.clone(), data);
        }

        self.to_save.ui_prefs.selection = self.ui.selection.clone();
        self.to_save.ui_prefs.cam_snapshots = self.cam_snapshots.clone();
        self.to_save.ui_prefs.mol_view = self.ui.mol_view.non_peptide_or_default();
        self.to_save.ui_prefs.mol_view_peptide = self.ui.mol_view_peptide;
        self.to_save.ui_prefs.view_sel_level = self.ui.view_sel_level;
        self.to_save.ui_prefs.near_sel_only = self.ui.show_near_sel_only;
        self.to_save.ui_prefs.near_lig_only = self.ui.show_near_lig_only;
        self.to_save.ui_prefs.nearby_dist_thresh = self.ui.nearby_dist_thresh;
        self.to_save.ui_prefs.visibility = self.ui.visibility.clone();
        self.to_save.ui_prefs.ui_visibility = self.ui.ui_vis.clone();
        self.to_save.mesh_coloring = self.ui.mesh_coloring;

        self.to_save_prev = self.to_save.clone();

        if let Err(e) = self
            .to_save
            .save(&self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE))
        {
            eprintln!("Error saving state: {e:?}");
        }
    }

    /// Run this when prefs, or a new molecule are loaded.
    pub fn update_from_prefs(&mut self) {
        println!("Updating state from prefs data");
        self.reset_selections();

        for mol in &mut self.peptide {
            let Some(data) = self.to_save.per_mol.get(&mol.common.ident).cloned() else {
                continue;
            };

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

            mol.rcsb_data = data.rcsb_data;
            mol.rcsb_files_avail = data.rcsb_files_avail;
        }

        self.ui.selection = self.to_save.ui_prefs.selection.clone();
        self.cam_snapshots = self.to_save.ui_prefs.cam_snapshots.clone();
        self.ui.mol_view = self.to_save.ui_prefs.mol_view.non_peptide_or_default();
        self.ui.mol_view_peptide = self.to_save.ui_prefs.mol_view_peptide;
        self.ui.view_sel_level = self.to_save.ui_prefs.view_sel_level;
        self.ui.show_near_sel_only = self.to_save.ui_prefs.near_sel_only;
        self.ui.show_near_lig_only = self.to_save.ui_prefs.near_lig_only;
        self.ui.nearby_dist_thresh = self.to_save.ui_prefs.nearby_dist_thresh;
        self.ui.visibility = self.to_save.ui_prefs.visibility.clone();
        self.ui.ui_vis = self.to_save.ui_prefs.ui_visibility.clone();

        self.ui.movement_speed_input = self.to_save.control_settings.movement_speed.to_string();
        self.ui.rotation_sens_input = self.to_save.control_settings.rotation_sens.to_string();
        self.ui.mol_move_sens_input = self.to_save.control_settings.mol_move_sens.to_string();
        self.ui.mesh_coloring = self.to_save.mesh_coloring;

        println!("Done");
    }

    pub fn load_prefs(&mut self) {
        match ToSave::load(&self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE)) {
            Ok(p) => self.to_save = p,
            Err(e) => {
                eprintln!("Unable to load save file; possibly the first time running: {e:?}.")
            }
        }

        self.update_from_prefs();
    }
}
