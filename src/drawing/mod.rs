//! Handles drawing molecules, atoms, bonds, and other items of interest. This
//! adds entities to the scene based on structs.

use std::{fmt, fmt::Display, io, io::ErrorKind, str::FromStr, sync::OnceLock};

use bincode::{Decode, Encode};
use bio_files::BondType;
use egui::{Color32, FontFamily};
use graphics::{EngineUpdates, Entity, Scene, TextOverlay, UP_VEC};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
    map_linear,
};
use mol_defs::molecules::{
    Atom, AtomRole, Chain, HydrogenBondTwoMols, MolGenericRef, MolType, peptide::MoleculePeptide,
    pocket::Pocket, small::MoleculeSmall,
};
use na_seq::Element;

use crate::{
    drawing::{
        atoms_bonds::{
            ATOM_SHININESS, BALL_RADIUS_WATER_H, BALL_RADIUS_WATER_O, BALL_STICK_RADIUS,
            BALL_STICK_RADIUS_H, BODY_SHINYNESS, MD_SOLVENT_ATOM_RADIUS_SCALE,
            WATER_BOND_THICKNESS, draw_hydrogen_bond, hide_md_wrapped_covalent_bond,
            use_md_compact_solvent_style,
        },
        viridis_lut::VIRIDIS,
    },
    mol_manip::ManipMode,
    prefs::OpenType,
    render::{
        Color, MESH_BOND, MESH_CUBE, MESH_POCKET_START, MESH_SPHERE_HIGHRES, MESH_SPHERE_LOWRES,
        MESH_SPHERE_MEDRES,
    },
    selection::{Selection, ViewSelLevel},
    state::{OperatingMode, State, StateUi, Visibility},
    util::{aromatic_ring_centroid, find_neighbor_posit, truncate_str},
};

pub mod atoms_bonds;
pub mod peptide;
pub mod ribbon_mesh;
mod viridis_lut;
pub mod wrappers;

const COLOR_MOL_MOVING: Color = (1., 1., 1.);
const COLOR_MOL_ROTATE: Color = (0.65, 1., 0.65);
const COLOR_MD_NEAR_MOL: Color = (0.0, 0., 1.); // Blended into
const BLEND_AMT_MD_NEAR_MOL: f32 = 0.5; // A higher value means it's closer to the special color.

// i.e a flexible bond.
// const LIGAND_COLOR_FLEX: Color = (1., 1., 0.);
pub const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);
pub const COLOR_AA_NON_RESIDUE_EGUI: Color32 = Color32::from_rgb(0, 204, 255);

pub const COLOR_SELECTED: Color = (1., 0., 0.);

const COLOR_WATER_BOND: Color = (0.5, 0.5, 0.8);

const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);

const LABEL_SIZE_ATOM: f32 = 16.;
const LABEL_SIZE_CHAIN: f32 = 30.;
const LABEL_SIZE_MOL: f32 = 18.;
const LABEL_SIZE_MOL_LARGE: f32 = 40.;
const LABEL_COLOR_ATOM: (u8, u8, u8, u8) = (255, 60, 160, 255);
const LABEL_COLOR_CHAIN: (u8, u8, u8, u8) = (200, 100, 160, 255);
// const LABEL_COLOR_ATOM_SEL: (u8, u8, u8, u8) = (255, 20, 20, 255);
const LABEL_COLOR_MOL: (u8, u8, u8, u8) = (255, 120, 150, 255);
const LABEL_COLOR_MOL_SEL: (u8, u8, u8, u8) = (255, 10, 10, 255);

// Hetero residues in protein, so they stand out from the normal protein molecules.
// Lower blend values mean more of the original color.
const COLOR_HETERO_RES: Color = (0.0, 0.0, 1.0);
const BLEND_AMT_HETERO_RES: f32 = 0.4;
const LIGAND_COLOR: Color = (0., 0.4, 1.);
const LIGAND_BLEND_AMT: f32 = 0.4;
const LIPID_COLOR: Color = (1.0, 1.0, 0.);
const LIPID_BLEND_AMT: f32 = 0.3;
pub const BLEND_AMT_MD_GROUP: f32 = 0.72;

pub const WATER_OPACITY: f32 = 1.;
pub const PHARMACOPHORE_OPACITY: f32 = 0.3;
pub const RADIUS_PHARMACOPHORE_HINT: f32 = 0.25;

// const COLOR_DOCKING_BOX: Color = (0.3, 0.3, 0.9);
// pub const COLOR_DOCKING_SITE_MESH: Color = (0.5, 0.5, 0.9);
// const DOCKING_SITE_OPACITY: f32 = 0.1;

// These colors are the default solid ones, i.e. if not colored by an atom-etc-based scheme.
const COLOR_SA_SURFACE: Color = (0.3, 0.2, 1.);
// i.e. the ribbon mesh, assuming a solid color, which we may not use.
const COLOR_SECONDARY_STRUCTURE: Color = (0.7, 0.2, 1.);
// const COLOR_POCKET: Color = (0.3, 0.5, 0.8);
const COLOR_POCKET: Color = (0.3, 0., 0.8);
const COLOR_POCKET_SPHERES: Color = (1., 1., 0.);

// Absolute unit in Å.

// Of bond radius. Covalent to H.

// Two of these is the separation.

pub const SIZE_SFC_DOT: f32 = 0.03;

const DIMMED_PEPTIDE_AMT: f32 = 0.92; // Higher value means more dim.

pub const DENSITY_ISO_OPACITY: f32 = 0.5;
pub const SAS_ISO_OPACITY: f32 = 0.75;
pub const POCKET_SURFACE_OPACITY: f32 = 0.85;

// These min/maxes are based on possible values of `aa.hydropathicity()`.
pub const HYDROPHOBICITY_MIN: f32 = -4.5;
pub const HYDROPHOBICITY_MAX: f32 = -HYDROPHOBICITY_MIN;

// We use this for mapping partial charge (e.g. as loaded from Amber) to colors.
// This should tightly span the range of expected charges.
// Note that we observe some charges out of this range, but have it narrower
// to show better constrast.
pub const CHARGE_MAP_MIN: f32 = -0.9;
pub const CHARGE_MAP_MAX: f32 = 0.65;

// This allows us to more easily customize sphere mesh resolution.
pub const MESH_BALL_STICK_SPHERE: usize = MESH_SPHERE_MEDRES;
// todo: I believe this causes performance problems on many machines. But looks
// todo much nicer.
pub const MESH_SPACEFILL_SPHERE: usize = MESH_SPHERE_HIGHRES;
pub const MESH_WATER_SPHERE: usize = MESH_SPHERE_MEDRES;
pub const MESH_BOND_CAP: usize = MESH_SPHERE_LOWRES;

// This should ideally be high res, but we experience anomolies on viewing items inside it, while
// the cam is outside.
// const MESH_DOCKING_SITE: usize = MESH_DOCKING_BOX;

// Spheres look slightly better when close, but even our coarsest one leads to performance problems.
const MESH_SURFACE_DOT: usize = MESH_CUBE;

// Cache blend results.
static LIG_C: OnceLock<Color> = OnceLock::new();
static LIG_CL: OnceLock<Color> = OnceLock::new();
static LIG_O: OnceLock<Color> = OnceLock::new();
static LIG_H: OnceLock<Color> = OnceLock::new();
static LIG_N: OnceLock<Color> = OnceLock::new();

// todo: If more than a certain atom count, draw atom labels every Xth atom.

/// Display the molecule name, atom name, etc, depending on label configuration.
fn text_overlay(
    entity: &mut Entity,
    mol_ident: &str,
    i_atom: usize,
    atom: &Atom,
    sel: bool,
    chains: &[Chain],
    atom_count: usize,
    ui: &StateUi,
) {
    // todo: Global consts A/R
    // If more than this many atoms, don't draw all.
    const ATOM_LIMIT_FOR_ALL_SNS: usize = 200;
    const ATOM_DRAW_RATIO: usize = 30;

    if ui.visibility.labels.atom_sn {
        if atom_count <= ATOM_LIMIT_FOR_ALL_SNS || i_atom.is_multiple_of(ATOM_DRAW_RATIO) {
            entity.overlay_text = Some(TextOverlay {
                text: format!("{}", atom.serial_number),
                size: LABEL_SIZE_ATOM,
                color: LABEL_COLOR_ATOM,
                font_family: FontFamily::Proportional,
            });
        }
    }

    // todo: Only the first label on each chain!
    if ui.visibility.labels.chain
        && let Some(ch_i) = &atom.chain
    {
        if *ch_i >= chains.len() {
            eprintln!("Error drawing chain label; chain out of bounds.");
            return;
        }

        let chain = &chains[*ch_i];
        // Only draw on one atom in the chain. Pick one towards the middle of the sequence; this
        // may be an OK proxy for in the middle of that chain's structure.
        // todo: Not ideal to compute this each time.
        if i_atom == chain.atoms[chain.atoms.len() / 2] {
            entity.overlay_text = Some(TextOverlay {
                text: format!("{}", chain.id),
                size: LABEL_SIZE_CHAIN,
                color: LABEL_COLOR_ATOM,
                font_family: FontFamily::Proportional,
            });
        }
    }

    let color = if sel {
        LABEL_COLOR_MOL_SEL
    } else {
        LABEL_COLOR_MOL
    };

    let text_full = mol_ident.to_string();
    let (text, font_size) = if atom_count > 150 {
        (text_full, LABEL_SIZE_MOL_LARGE)
    } else {
        (truncate_str(&text_full, 20), LABEL_SIZE_MOL)
    };

    if ui.visibility.labels.mol && i_atom == 0 {
        entity.overlay_text = Some(TextOverlay {
            text,
            size: font_size,
            color,
            font_family: FontFamily::Proportional,
        });
    }
}

/// We use the Entity's class field to determine which graphics-engine entities to retain and remove.
/// This affects both local drawing logic, and engine-level entity setup.
///
/// The numerical values here are arbitrary, and can be changed at any time; they're just
/// so the engine has a unique identifier for each without knowing about application-specific types.
#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum EntityClass {
    Protein = 0,
    Ligand = 1,
    NucleicAcid = 2,
    Lipid = 3,
    DensityPoint = 4,
    DensitySurface = 5,
    SecondaryStructure = 6,
    SaSurface = 7,
    SaSurfaceDots = 8,
    DockingSite = 9,
    WaterModel = 10,
    Pharmacophore = 11,
    PharmacophoreHint = 12,
    Pocket = 13,
    Other = 99,
}

/// Maps a molecule type onto the application's vocabulary for it: which class of entity we draw it
/// as, and which slot it occupies in the file-open flow.
///
/// This is a trait on our side rather than methods on `MolType` because `EntityClass` and `OpenType`
/// are presentation concerns; `mol_defs` has no reason to know about either.
pub trait MolTypeExt {
    fn entity_type(self) -> EntityClass;
    fn to_open_type(self) -> OpenType;
}

impl MolTypeExt for MolType {
    fn entity_type(self) -> EntityClass {
        use MolType::*;
        match self {
            Peptide => EntityClass::Protein,
            Ligand => EntityClass::Ligand,
            NucleicAcid => EntityClass::NucleicAcid,
            Lipid => EntityClass::Lipid,
            Pocket => EntityClass::Pocket,
            Water => EntityClass::Protein, // todo for now
        }
    }

    fn to_open_type(self) -> OpenType {
        use MolType::*;
        match self {
            Peptide => OpenType::Peptide,
            Ligand => OpenType::Ligand,
            NucleicAcid => OpenType::NucleicAcid,
            Lipid => OpenType::Lipid,
            Pocket => OpenType::Pocket,
            Water => panic!("Can't convert water to open type"),
        }
    }
}

// todo: For ligands that are flexible, highlight the fleixble bonds in a bright color.

pub fn blend_color(color_0: Color, color_1: Color, portion: f32) -> Color {
    (
        map_linear(portion, (0., 1.), (color_0.0, color_1.0)),
        map_linear(portion, (0., 1.), (color_0.1, color_1.1)),
        map_linear(portion, (0., 1.), (color_0.2, color_1.2)),
    )
}

fn cache_lig_color(el: Element) -> Option<&'static OnceLock<Color>> {
    match el {
        Element::Carbon => Some(&LIG_C),
        Element::Oxygen => Some(&LIG_O),
        Element::Hydrogen => Some(&LIG_H),
        Element::Nitrogen => Some(&LIG_N),
        Element::Chlorine => Some(&LIG_CL),
        _ => None,
    }
}

/// Make ligands stand out visually, when colored by atom.
fn mod_color_for_ligand(
    color: &Color,
    el: Element,
    color_by_q: bool,
    color_by_mol: bool,
    mol_i: usize,
    num_mols: usize,
) -> Color {
    // For now, color by mol overrides others, but only for Carbon atoms.
    if color_by_mol {
        let mol_color = color_viridis(mol_i, 0, num_mols);
        if el == Element::Carbon {
            return mol_color;
        } else {
            return blend_color(*color, mol_color, LIGAND_BLEND_AMT);
        }
    }

    if color_by_q {
        return blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT);
    }

    if let Some(slot) = cache_lig_color(el) {
        *slot.get_or_init(|| blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT))
    } else {
        blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT)
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum MoleculeView {
    Backbone,
    Sticks,
    BallAndStick,
    /// i.e. Van der Waals radius, or CPK.
    SpaceFill,
    #[default]
    Ribbon,
    Surface,
    Dots,
}

impl MoleculeView {
    pub const PEPTIDE_OPTIONS: [Self; 7] = [
        Self::Backbone,
        Self::Sticks,
        Self::BallAndStick,
        Self::Ribbon,
        Self::SpaceFill,
        Self::Surface,
        Self::Dots,
    ];
    pub const NON_PEPTIDE_OPTIONS: [Self; 3] = [Self::Sticks, Self::BallAndStick, Self::SpaceFill];
    pub const DEFAULT_NON_PEPTIDE: Self = Self::BallAndStick;

    pub fn is_non_peptide(self) -> bool {
        Self::NON_PEPTIDE_OPTIONS.contains(&self)
    }

    pub fn non_peptide_or_default(self) -> Self {
        if self.is_non_peptide() {
            self
        } else {
            Self::DEFAULT_NON_PEPTIDE
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Backbone => Self::Sticks,
            Self::Sticks => Self::BallAndStick,
            Self::BallAndStick => Self::SpaceFill,
            Self::SpaceFill => Self::Ribbon,
            Self::Ribbon => Self::Surface,
            Self::Surface => Self::Dots,
            Self::Dots => Self::Backbone,
        }
    }

    // Note: repetitive
    pub fn prev(self) -> Self {
        match self {
            Self::Backbone => Self::Dots,
            Self::Sticks => Self::Backbone,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            Self::Ribbon => Self::SpaceFill,
            Self::Surface => Self::Ribbon,
            Self::Dots => Self::Surface,
        }
    }

    pub fn next_editor(self) -> Self {
        match self {
            Self::Sticks => Self::BallAndStick,
            Self::BallAndStick => Self::SpaceFill,
            Self::SpaceFill => Self::Sticks,
            _ => Self::Sticks,
        }
    }

    // Note: repetitive
    pub fn prev_editor(self) -> Self {
        match self {
            Self::Sticks => Self::SpaceFill,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            _ => Self::Sticks,
        }
    }

    pub fn next_non_peptide(self) -> Self {
        match self.non_peptide_or_default() {
            Self::Sticks => Self::BallAndStick,
            Self::BallAndStick => Self::SpaceFill,
            Self::SpaceFill => Self::Sticks,
            _ => unreachable!(),
        }
    }

    pub fn prev_non_peptide(self) -> Self {
        match self.non_peptide_or_default() {
            Self::Sticks => Self::SpaceFill,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            _ => unreachable!(),
        }
    }
}

/// The view a given peptide is actually drawn with. `ui.mol_view_peptide` is a single global
/// setting, but some views can only apply to one molecule at a time, or not at all:
///
/// - Ribbon, surface and dots each render from a mesh in a shared GPU slot, which only the
///   peptide selected for tools uses. Other open peptides fall back to atoms and bonds.
/// - A ribbon needs a Cα trace to follow; a structure without one (e.g. a backbone-less or
///   otherwise non-standard file) would render as nothing at all.
///
/// Selection and molecule manipulation share this with the drawing code, so what they act on
/// matches what's on screen.
pub fn effective_mol_view_peptide(state: &State, mol_i: usize) -> MoleculeView {
    let view = state.ui.mol_view_peptide;

    let mesh_view = matches!(
        view,
        MoleculeView::Ribbon | MoleculeView::Dots | MoleculeView::Surface
    );
    if mesh_view && state.peptide_for_tools_i() != Some(mol_i) {
        return MoleculeView::BallAndStick;
    }

    if view == MoleculeView::Ribbon
        && let Some(mol) = state.peptides.get(mol_i)
        && !has_ribbon_trace(mol)
    {
        return MoleculeView::BallAndStick;
    }

    view
}

/// Whether a ribbon can be built for this peptide: the mesh follows Cα atoms, and needs at least
/// two of them, in a visible chain. Note that missing secondary structure is fine on its own —
/// those residues render as coil.
fn has_ribbon_trace(mol: &MoleculePeptide) -> bool {
    let visible_chain = |atom: &Atom| {
        atom.chain
            .and_then(|c| mol.chains.get(c))
            .is_none_or(|c| c.visible)
    };

    mol.common
        .atoms
        .iter()
        .filter(|a| a.role == Some(AtomRole::C_Alpha) && visible_chain(a))
        .take(2)
        .count()
        == 2
}

impl FromStr for MoleculeView {
    type Err = io::Error;

    /// This includes some PyMol standard names, which map to the closest visualization we have.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sticks" | "lines" => Ok(MoleculeView::Sticks),
            "backbone" => Ok(MoleculeView::Backbone),
            "ballandstick" | "ball_and_stick" | "ball-and-stick" => Ok(MoleculeView::BallAndStick),
            "spacefill" | "space_fill" | "space-fill" | "spheres" => Ok(MoleculeView::SpaceFill),
            "cartoon" | "ribbon" => Ok(MoleculeView::Ribbon),
            "surface" => Ok(MoleculeView::Surface),
            "dots" => Ok(MoleculeView::Dots),
            other => Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("invalid MoleculeView: '{}'", other),
            )),
        }
    }
}

impl Display for MoleculeView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            Self::Backbone => "Backbone",
            Self::Sticks => "Sticks",
            Self::BallAndStick => "Ball and stick",
            Self::Ribbon => "Ribbon",
            Self::SpaceFill => "Spacefill",
            Self::Surface => "Surface (Van der Waals)",
            Self::Dots => "Dots (Van der Waals)",
        };

        write!(f, "{val}")
    }
}

/// A linear color map using the viridis scheme. Uses a LUT.
pub fn color_viridis(i: usize, min: usize, max: usize) -> Color {
    // Normalize i to [0.0, 1.0]
    let t = if max > min {
        // Compute as f32 and clamp to [0,1]
        let tt = (i.saturating_sub(min) as f32) / ((max - min) as f32);
        tt.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Scale t into the control‐point index range [0 .. VIRIDIS.len()-1]
    let n_pts = VIRIDIS.len();
    let scaled = t * ((n_pts - 1) as f32);
    let idx = scaled.floor() as usize;
    let idx_next = (idx + 1).min(n_pts - 1);
    let frac = scaled - (idx as f32);

    let (r1, g1, b1) = VIRIDIS[idx];
    let (r2, g2, b2) = VIRIDIS[idx_next];

    // Linear interpolation between the two nearest control points
    let r = r1 + (r2 - r1) * frac;
    let g = g1 + (g2 - g1) * frac;
    let b = b1 + (b2 - b1) * frac;

    (r, g, b)
}

/// Returns a vivid colour for item `i` in `[min, max]` such that adjacent indices are as
/// visually distinct as possible.  Uses the full HSV hue wheel at fixed high saturation and
/// brightness so every colour is saturated and easy to differentiate.  A bit-reversal
/// permutation maps sequential ranks to hues that are maximally far apart:
///   rank 0 → 0° (red), rank 1 → 180° (cyan), rank 2 → 90° (yellow-green),
///   rank 3 → 270° (violet), rank 4 → 45° (orange), …
pub fn color_alternating_contrast(i: usize, min: usize, max: usize) -> Color {
    const SATURATION: f32 = 0.88;
    const VALUE: f32 = 0.92;

    let n = max.saturating_sub(min) + 1;
    let rank = i.saturating_sub(min);

    // Map rank to hue in [0, 1) via bit-reversal so adjacent ranks are opposite in hue.
    let hue_t: f32 = if n <= 1 {
        0.0
    } else {
        let bits = (usize::BITS - (n - 1).leading_zeros()) as usize;
        let reversed = rank.reverse_bits() >> (usize::BITS as usize - bits);
        let denom = (1 << bits) - 1;
        reversed as f32 / denom as f32
    };

    // Convert HSV → RGB.
    let h = hue_t * 360.0;
    let c = VALUE * SATURATION;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = VALUE - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (r + m, g + m, b + m)
}

pub fn color_viridis_float(i: f32, min: f32, max: f32) -> Color {
    const RESOLUTION: usize = 2_048;

    // Normalize i into [0.0, 1.0]
    let t = if max > min {
        ((i - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let idx = (t * (RESOLUTION as f32)).round() as usize;

    color_viridis(idx, 0, RESOLUTION)
}

/// Water from a MD sim; not from atoms in experimental data.
pub fn draw_water(
    scene: &mut Scene,
    o_pos: &[Vec3],
    h0_pos: &[Vec3],
    h1_pos: &[Vec3],
    hide_water: bool,
) {
    scene
        .entities
        .retain(|ent| ent.class != EntityClass::WaterModel as u32);

    // todo: Borrow mut prob.
    // clear_mol_entity_indices(state);

    if hide_water {
        return;
    }

    for i in 0..o_pos.len() {
        let mut ent = Entity::new(
            MESH_WATER_SPHERE,
            o_pos[i],
            Quaternion::new_identity(),
            BALL_RADIUS_WATER_O,
            Element::Oxygen.color(),
            ATOM_SHININESS,
        );

        ent.opacity = WATER_OPACITY;
        ent.class = EntityClass::WaterModel as u32;
        scene.entities.push(ent);

        for pos in [h0_pos[i], h1_pos[i]].iter() {
            let mut ent = Entity::new(
                MESH_WATER_SPHERE,
                *pos,
                Quaternion::new_identity(),
                BALL_RADIUS_WATER_H,
                Element::Hydrogen.color(),
                ATOM_SHININESS,
            );

            ent.opacity = WATER_OPACITY;
            ent.class = EntityClass::WaterModel as u32;
            scene.entities.push(ent);
        }

        // Bonds
        for pair in [(o_pos[i], h0_pos[i]), (o_pos[i], h1_pos[i])] {
            let center = (pair.0 + pair.1) / 2.;
            let diff = pair.0 - pair.1;
            let dist = diff.magnitude();

            // This handles the case of atoms in the water molecule split across the periodic boundary
            // condition; don't draw the bond.
            if dist > 5. {
                continue;
            }

            let orientation = Quaternion::from_unit_vecs(UP_VEC, diff.to_normalized());
            let mut ent_bond = Entity::new(
                MESH_BOND,
                center,
                orientation,
                1.,
                COLOR_WATER_BOND,
                BODY_SHINYNESS,
            );
            let scale = Some(Vec3::new(WATER_BOND_THICKNESS, dist, WATER_BOND_THICKNESS));

            ent_bond.opacity = WATER_OPACITY;
            ent_bond.scale_partial = scale;
            ent_bond.class = EntityClass::WaterModel as u32;
            scene.entities.push(ent_bond);
        }
    }
}

/// For all molecule types (for now, not including peptide)
pub fn draw_mol(
    mol: MolGenericRef,
    mol_i: usize,
    ui: &StateUi,
    active_mol: &Option<(MolType, usize)>,
    manip_mode: ManipMode,
    mode: OperatingMode,
    num_mols: usize,
    draw_md_mols: bool,
) -> Vec<Entity> {
    let show_pharmacophore = !ui.visibility.hide_pharmacophore || mode == OperatingMode::MolEditor;

    draw_mol_with_pharmacophore_visibility(
        mol,
        mol_i,
        ui,
        active_mol,
        manip_mode,
        mode,
        num_mols,
        draw_md_mols,
        show_pharmacophore,
    )
}

/// Draw a molecule with an explicit override for its pharmacophore feature renders.
/// The molecule editor uses this instead of the global visibility preference.
#[allow(clippy::too_many_arguments)]
pub fn draw_mol_with_pharmacophore_visibility(
    mol: MolGenericRef,
    mol_i: usize,
    ui: &StateUi,
    active_mol: &Option<(MolType, usize)>,
    manip_mode: ManipMode,
    mode: OperatingMode,
    num_mols: usize,
    draw_md_mols: bool,
    show_pharmacophore: bool,
) -> Vec<Entity> {
    let mut result = Vec::new();

    if !mol.common().visible {
        return result;
    }

    let mol_active = if let Some((active_mol_type, active_i)) = active_mol {
        mol.mol_type() == *active_mol_type && mol_i == *active_i
    } else {
        false
    };

    let compact_md_solvent_style = use_md_compact_solvent_style(draw_md_mols, &mol.common().ident);

    // todo: You have problems with transparent objects like the view cube in conjunction
    // todo with the transparent surface; workaround to not draw the cube here.
    if ui.show_docking_tools && ui.mol_view != MoleculeView::Surface {
        // Add a visual indicator for the docking site.
    }

    let sel = if ui.selection.is_bond() {
        &Selection::None
    } else {
        &ui.selection
    };

    let components = match mol {
        MolGenericRef::Small(m) => &m.components,
        _ => &None,
    };

    if matches!(
        ui.mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        for (i_atom, atom) in mol.common().atoms.iter().enumerate() {
            if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
                continue;
            }

            // M/EP site on rigid water molecules; don't draw.
            if atom.type_in_res_general == Some("MW".to_string()) {
                continue;
            }

            let mut color = (0., 0., 0.);
            let mut manip_active = false;

            match manip_mode {
                ManipMode::Move((mol_type, i)) => match mode {
                    OperatingMode::Primary => {
                        if mol_type == mol.mol_type() && i == mol_i {
                            color = COLOR_MOL_MOVING;
                            manip_active = true;
                        }
                    }
                    OperatingMode::MolEditor => {
                        if i == i_atom {
                            color = COLOR_MOL_MOVING;
                            manip_active = true;
                        }
                    }
                    OperatingMode::ProteinEditor => (),
                },
                ManipMode::Rotate((mol_type, i)) => match mode {
                    OperatingMode::Primary => {
                        if mol_type == mol.mol_type() && i == mol_i {
                            color = COLOR_MOL_ROTATE;
                            manip_active = true;
                        }
                    }
                    OperatingMode::MolEditor => {
                        let bond = &mol.common().bonds[i];
                        if bond.atom_0 == i_atom || bond.atom_1 == i_atom {
                            color = COLOR_MOL_ROTATE;
                            manip_active = true;
                        }
                    }
                    OperatingMode::ProteinEditor => (),
                },
                ManipMode::None => (),
            }

            if !manip_active {
                color = atoms_bonds::atom_color(
                    atom,
                    mol_i,
                    i_atom,
                    &[],
                    None,
                    0,
                    0,
                    &[],
                    &[],
                    sel,
                    ViewSelLevel::Atom, // Always color lipids by atom.
                    false,
                    ui.res_coloring,
                    ui.atom_color_by_charge,
                    mol.mol_type(),
                    components,
                );

                if color != COLOR_SELECTED {
                    match mol.mol_type() {
                        MolType::Ligand => {
                            if mode == OperatingMode::Primary {
                                color = mod_color_for_ligand(
                                    &color,
                                    atom.element,
                                    ui.atom_color_by_charge,
                                    ui.color_by_mol,
                                    mol_i,
                                    num_mols,
                                )
                            }
                        }
                        // todo: Lipid and NA caches A/R
                        // todo: Color for NA
                        MolType::NucleicAcid => {
                            color = blend_color(color, LIPID_COLOR, LIPID_BLEND_AMT)
                        }
                        MolType::Lipid => color = blend_color(color, LIPID_COLOR, LIPID_BLEND_AMT),
                        _ => (),
                    }
                }
            }

            let (radius, mesh) = match ui.mol_view {
                MoleculeView::SpaceFill => (atom.element.vdw_radius(), MESH_SPACEFILL_SPHERE),
                _ => match atom.element {
                    Element::Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
                    _ => (BALL_STICK_RADIUS, MESH_BALL_STICK_SPHERE),
                },
            };

            let radius =
                if compact_md_solvent_style && !matches!(ui.mol_view, MoleculeView::SpaceFill) {
                    radius * MD_SOLVENT_ATOM_RADIUS_SCALE
                } else {
                    radius
                };

            let mut entity = Entity::new(
                mesh,
                mol.common().atom_posits[i_atom].into(),
                Quaternion::new_identity(),
                radius,
                color,
                ATOM_SHININESS,
            );

            if mode != OperatingMode::MolEditor {
                // Note: We draw these on the bond entities if not in a view that shows atoms.
                text_overlay(
                    &mut entity,
                    &mol.name(),
                    i_atom,
                    atom,
                    mol_active,
                    &[],
                    mol.common().atoms.len(),
                    ui,
                );
            }

            entity.class = mol.mol_type().entity_type() as u32;
            result.push(entity);
        }
    }

    // Aromatic-only adjacency list used for ring centroid BFS (so it finds the aromatic ring,
    // not a shorter fused non-aromatic ring).
    let aromatic_adj = {
        let n = mol.common().atoms.len();
        let mut adj = vec![Vec::new(); n];
        for b in &mol.common().bonds {
            if b.bond_type == BondType::Aromatic {
                adj[b.atom_0].push(b.atom_1);
                adj[b.atom_1].push(b.atom_0);
            }
        }
        adj
    };

    // todo: C+P from draw_molecule. With some removed, but much repeated.
    for (i_bond, bond) in mol.common().bonds.iter().enumerate() {
        let atom_0 = &mol.common().atoms[bond.atom_0];
        let atom_1 = &mol.common().atoms[bond.atom_1];

        if ui.visibility.hide_hydrogen
            && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
        {
            continue;
        }

        let posit_0: Vec3 = mol.common().atom_posits[bond.atom_0].into();
        let posit_1: Vec3 = mol.common().atom_posits[bond.atom_1].into();

        if hide_md_wrapped_covalent_bond(draw_md_mols, posit_0, posit_1) {
            continue;
        }

        // For determining how to orient multiple-bonds. Only run for relevant bonds to save
        // computation.
        let neighbor_posit = match bond.bond_type {
            BondType::Aromatic => {
                let mut hydrogen_is = Vec::with_capacity(mol.common().atoms.len());
                for atom in &mol.common().atoms {
                    hydrogen_is.push(atom.element == Element::Hydrogen);
                }
                let centroid: Vec3 = aromatic_ring_centroid(
                    &aromatic_adj,
                    &mol.common().atom_posits,
                    bond.atom_0,
                    bond.atom_1,
                    &hydrogen_is,
                )
                .map(|c| c.into())
                .unwrap_or_else(|| mol.common().atom_posits[0].into());
                (centroid, false)
            }
            BondType::Double | BondType::Triple => {
                let mut hydrogen_is = Vec::with_capacity(mol.common().atoms.len());
                for atom in &mol.common().atoms {
                    hydrogen_is.push(atom.element == Element::Hydrogen);
                }
                let neighbor_i = find_neighbor_posit(
                    &mol.common().adjacency_list,
                    bond.atom_0,
                    bond.atom_1,
                    &hydrogen_is,
                );
                match neighbor_i {
                    Some((i, p1)) => (mol.common().atom_posits[i].into(), p1),
                    None => (mol.common().atom_posits[0].into(), false),
                }
            }
            _ => (Vec3::new_zero(), false),
        };

        let mut color_0 = (0., 0., 0.);
        let mut color_1 = (0., 0., 0.);

        let mut manip_active = false;

        match manip_mode {
            ManipMode::Move((mol_type, i)) => match mode {
                OperatingMode::Primary => {
                    if mol_type == mol.mol_type() && i == mol_i {
                        color_0 = COLOR_MOL_MOVING;
                        color_1 = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                }
                OperatingMode::MolEditor => {
                    if i == bond.atom_0 {
                        // todo: You may need to clarify manip_1 active manip_0 active or similar,
                        // todo: otherwise the other bond half will not be colored by atom etc.
                        color_0 = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                    if i == bond.atom_1 {
                        color_1 = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                }
                OperatingMode::ProteinEditor => (),
            },
            ManipMode::Rotate((mol_type, i)) => match mode {
                OperatingMode::Primary => {
                    if mol_type == mol.mol_type() && i == mol_i {
                        color_0 = COLOR_MOL_ROTATE;
                        color_1 = COLOR_MOL_ROTATE;
                        manip_active = true;
                    }
                }
                OperatingMode::MolEditor => {
                    if i == i_bond {
                        // todo: You may need to clarify manip_1 active manip_0 active or similar,
                        // todo: otherwise the other bond half will not be colored by atom etc.
                        color_0 = COLOR_MOL_ROTATE;
                        color_1 = COLOR_MOL_ROTATE;
                        manip_active = true;
                    }
                }
                OperatingMode::ProteinEditor => (),
            },
            ManipMode::None => (),
        }

        if !manip_active {
            color_0 = atoms_bonds::atom_color(
                atom_0,
                mol_i,
                bond.atom_0,
                &[],
                None,
                0,
                0,
                &[],
                &[],
                sel,                // ignores bond coloring by adjacent atom if in bond sel mode.
                ViewSelLevel::Atom, // Always color ligands by atom.
                false,
                ui.res_coloring,
                ui.atom_color_by_charge,
                mol.mol_type(),
                components,
            );
            color_1 = atoms_bonds::atom_color(
                atom_1,
                mol_i,
                bond.atom_1,
                &[],
                None,
                0,
                0,
                &[],
                &[],
                sel,                // ignores bond coloring by adjacent atom if in bond sel mode.
                ViewSelLevel::Atom, // Always color ligands by atom.
                false,
                ui.res_coloring,
                ui.atom_color_by_charge,
                mol.mol_type(),
                components,
            );

            // If in atom sel mode, we color bonds normally above (The  half of each bond connected
            // to the selected atom). If in bond sel mode, we color the bond between two atoms below.

            match &ui.selection {
                Selection::BondLig((_mol_i, bond_i))
                | Selection::BondNucleicAcid((_mol_i, bond_i))
                | Selection::BondLipid((_mol_i, bond_i)) => {
                    if *bond_i == i_bond && *_mol_i == mol_i {
                        color_0 = COLOR_SELECTED;
                        color_1 = COLOR_SELECTED;
                    }
                }
                Selection::BondsLig((_mol_i, bonds_i)) => {
                    if bonds_i.contains(&i_bond) && *_mol_i == mol_i {
                        color_0 = COLOR_SELECTED;
                        color_1 = COLOR_SELECTED;
                    }
                }
                _ => (),
            };

            let helper = |atom: &Atom, color: &mut Color| {
                if *color != COLOR_SELECTED {
                    match mol.mol_type() {
                        MolType::Ligand => {
                            if mode == OperatingMode::Primary {
                                *color = mod_color_for_ligand(
                                    color,
                                    atom.element,
                                    ui.atom_color_by_charge,
                                    ui.color_by_mol,
                                    mol_i,
                                    num_mols,
                                )
                            }
                        }
                        // todo: Color for NA
                        MolType::NucleicAcid => {
                            *color = blend_color(*color, LIPID_COLOR, LIPID_BLEND_AMT)
                        }
                        MolType::Lipid => {
                            *color = blend_color(*color, LIPID_COLOR, LIPID_BLEND_AMT)
                        }
                        _ => (),
                    }
                }
            };

            helper(atom_0, &mut color_0);
            helper(atom_1, &mut color_1);
        }

        let to_hydrogen =
            atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen;

        let mut entities = atoms_bonds::bond_entities(
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            mol.mol_type(),
            &mol.common().ident,
            draw_md_mols,
            true,
            neighbor_posit,
            mol_active,
            to_hydrogen,
        );

        // Draw atom-based labels on bonds if not in a view mode that shows atoms.
        if !entities.is_empty()
            && mode != OperatingMode::MolEditor
            && !matches!(
                ui.mol_view,
                MoleculeView::BallAndStick | MoleculeView::SpaceFill
            )
        {
            text_overlay(
                &mut entities[0],
                &mol.name(),
                bond.atom_0,
                atom_0,
                mol_active, // todo
                &[],
                mol.common().bonds.len(),
                ui,
            );
        }

        if let MolGenericRef::Small(m) = &mol
            && show_pharmacophore
        {
            result.extend(draw_mol_pharmacophore(m, mode));
        }

        result.extend(entities);
    }

    // todo: Add back if you include lig H bonds.
    // if !state.ui.visibility.hide_h_bonds {
    //     for bond in &mol.bonds_hydrogen {
    //         let atom_donor = &atoms_positioned[bond.donor];
    //         let atom_acceptor = &atoms_positioned[bond.acceptor];
    //
    //         let posit_donor: Vec3 = atom_donor.posit.into();
    //
    //         let posit_acceptor: Vec3 = atom_acceptor.posit.into();
    //
    //         bond_entities(
    //             &mut scene.entities,
    //             posit_donor,
    //             posit_acceptor,
    //             COLOR_H_BOND,
    //             COLOR_H_BOND,
    //             BondType::Dummy,
    //             MolType::Ligand,
    //             true,
    //             (Vec3::new_zero(), false),
    //         );
    //     }
    // }

    // set_docking_light(scene, Some(&state.ligand.as_ref().unwrap().docking_site));

    result
}

/// Note: We currently have this combined with the same call time and entity class as other
/// small mols.
fn draw_mol_pharmacophore(mol: &MoleculeSmall, op_mode: OperatingMode) -> Vec<Entity> {
    let mut res = Vec::new();

    for (i, feat) in mol.pharmacophore.features.iter().enumerate() {
        let posit: Vec3 = feat
            .posit_from_atoms(&mol.common.atom_posits)
            .unwrap_or(feat.posit)
            .into();

        let mut ent = Entity::new(
            MESH_SPHERE_HIGHRES,
            posit,
            Quaternion::new_identity(),
            feat.feature_type.disp_radius(),
            feat.feature_type.color(),
            ATOM_SHININESS,
        );

        if op_mode == OperatingMode::MolEditor {
            ent.overlay_text = Some(TextOverlay {
                text: format!("{}", i + 1),
                size: LABEL_SIZE_ATOM,
                color: LABEL_COLOR_ATOM,
                font_family: FontFamily::Proportional,
            });
        }

        ent.opacity = PHARMACOPHORE_OPACITY;
        // ent.class = EntityClass::Pharmacophore as u32;
        ent.class = EntityClass::Ligand as u32;
        res.push(ent);
    }

    res
}

/// Display likely locations to place this category of pharmacophore based on
/// characteristics of the molecule.
pub fn draw_pharmacophore_hint_sites(
    entities: &mut Vec<Entity>,
    hint_sites: &[Vec3F64],
    engine_updates: &mut EngineUpdates,
) {
    entities.retain(|ent| ent.class != EntityClass::PharmacophoreHint as u32);

    for hint_site in hint_sites {
        let mut ent = Entity::new(
            MESH_SPHERE_HIGHRES,
            (*hint_site).into(),
            Quaternion::new_identity(),
            RADIUS_PHARMACOPHORE_HINT,
            (1., 0.1, 0.1), // Red
            ATOM_SHININESS,
        );

        ent.opacity = PHARMACOPHORE_OPACITY;
        ent.class = EntityClass::PharmacophoreHint as u32;
        entities.push(ent);
    }

    engine_updates
        .entities
        .push_class(EntityClass::PharmacophoreHint as u32);
}

/// Render spheres if manipulation is active, otherwise the mesh.
pub fn draw_pocket(
    pocket: &Pocket,
    hydrogen_bonds: &[HydrogenBondTwoMols],
    // Lig posits are for drawing Hydrogen bonds.
    lig_posits: &[Vec3F64],
    visibility: &Visibility,
    selection: &Selection,
    manip_mode: &ManipMode,
    // draw_mesh: bool, // E.g. false when moving.
) -> Vec<Entity> {
    let mut res = Vec::new();

    if visibility.hide_pockets {
        return res;
    }

    if !pocket.common.visible {
        return res;
    }

    let manipulating_pocket = matches!(
        manip_mode,
        ManipMode::Move((MolType::Pocket, _)) | ManipMode::Rotate((MolType::Pocket, _))
    );

    // todo: For now, drawing the spheres we use to compute exclusion.
    // todo: Likely not useful to the user, but useful for validating our approach and debugging.
    for sphere in &pocket.volume.spheres {
        let mut ent = Entity::new(
            MESH_SPHERE_HIGHRES,
            sphere.center.into(),
            Quaternion::new_identity(),
            sphere.radius,
            COLOR_POCKET_SPHERES,
            ATOM_SHININESS,
        );

        ent.class = EntityClass::Pocket as u32;
        // No transparency on spheres; makes it more confusing by adding clutter.

        // todo kludge to now show this, without updating the entity count.
        // todo: Opacity=0 is producing undesired effects.
        if !manipulating_pocket {
            ent.position += UP_VEC * 10_000.;
        }
        res.push(ent);
    }

    // let mesh_posit = pocket.common.atom_posits[0] - pocket.common.atoms[0].posit;

    let color_mesh = if matches!(
        selection,
        Selection::AtomPocket(_) | Selection::BondPocket(_)
    ) {
        COLOR_SELECTED
    } else {
        COLOR_POCKET
    };

    // Draw the surface mesh; pre-computed.
    let mut ent = Entity::new(
        MESH_POCKET_START + pocket.mesh_i_rel,
        // mesh_posit.into(),
        // mesh_posit.into(),
        Vec3::new_zero(),
        // pocket.mesh_orientation,
        Quaternion::new_identity(),
        1.,
        color_mesh,
        ATOM_SHININESS,
    );

    // ent.pivot = Some(pocket.mesh_pivot);
    ent.class = EntityClass::Pocket as u32;
    ent.opacity = POCKET_SURFACE_OPACITY;

    // todo kludge to now show this, without updating the entity count.
    // todo: Opacity=0 is producing undesired effects.
    if manipulating_pocket {
        ent.position += UP_VEC * 10_000.;
    }

    res.push(ent);

    if !visibility.hide_h_bonds {
        for bond in hydrogen_bonds {
            let posit_donor = if bond.donor.0 == 0 {
                if bond.donor.1 > lig_posits.len() {
                    eprintln!("Out of bounds error on drawing H bond (lig)");
                    continue;
                }
                lig_posits[bond.donor.1]
            } else {
                if bond.donor.1 > pocket.common.atom_posits.len() {
                    eprintln!("Out of bounds error on drawing H bond (pocket)");
                    continue;
                }
                pocket.common.atom_posits[bond.donor.1]
            };

            let posit_acc = if bond.acceptor.0 == 0 {
                if bond.acceptor.1 > lig_posits.len() {
                    eprintln!("Out of bounds error on drawing H bond (lig)");
                    continue;
                }
                lig_posits[bond.acceptor.1]
            } else {
                if bond.acceptor.1 > pocket.common.atom_posits.len() {
                    eprintln!("Out of bounds error on drawing H bond (pocket)");
                    continue;
                }
                pocket.common.atom_posits[bond.acceptor.1]
            };

            res.extend(draw_hydrogen_bond(
                posit_donor.into(),
                posit_acc.into(),
                MolType::Pocket,
                bond.strength,
                false,
            ));
        }
    }

    res
}

pub fn draw_md_hydrogen_bonds<F>(
    hydrogen_bonds: &[HydrogenBondTwoMols],
    visibility: &Visibility,
    mut atom_lookup: F,
) -> Vec<Entity>
where
    F: FnMut((usize, usize)) -> Option<(Vec3, MolType)>,
{
    if visibility.hide_h_bonds {
        return Vec::new();
    }

    let mut res = Vec::new();

    for bond in hydrogen_bonds {
        let Some((posit_donor, donor_type)) = atom_lookup(bond.donor) else {
            continue;
        };
        let Some((posit_acc, acceptor_type)) = atom_lookup(bond.acceptor) else {
            continue;
        };

        if visibility.hide_water
            && (donor_type == MolType::Water || acceptor_type == MolType::Water)
        {
            continue;
        }

        // Snapshot H bonds may span a periodic wrap; skip the long scene-space segment.
        if hide_md_wrapped_covalent_bond(true, posit_donor, posit_acc) {
            continue;
        }

        let mol_type = if donor_type != MolType::Water {
            donor_type
        } else {
            acceptor_type
        };

        res.extend(draw_hydrogen_bond(
            posit_donor,
            posit_acc,
            mol_type,
            bond.strength,
            true,
        ));
    }

    res
}
