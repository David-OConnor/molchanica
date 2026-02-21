//! Handles drawing molecules, atoms, bonds, and other items of interest. This
//! adds entities to the scene based on structs.

use std::{fmt, fmt::Display, io, io::ErrorKind, str::FromStr, sync::OnceLock};

use bincode::{Decode, Encode};
use bio_files::{BondType, ResidueType};
use egui::{Color32, FontFamily};
use graphics::{ControlScheme, EngineUpdates, Entity, Scene, TextOverlay, UP_VEC};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
    map_linear,
};
use na_seq::Element;

use crate::{
    drawing::{
        atoms_bonds::{
            ATOM_SHININESS, BALL_RADIUS_WATER_H, BALL_RADIUS_WATER_O, BALL_STICK_RADIUS,
            BALL_STICK_RADIUS_H, BODY_SHINYNESS, WATER_BOND_THICKNESS, draw_hydrogen_bond,
        },
        viridis_lut::VIRIDIS,
    },
    mol_manip::ManipMode,
    molecules::{
        Atom, AtomRole, Chain, HydrogenBondTwoMols, MolGenericRef, MolType, MoleculePeptide,
        pocket::Pocket, small::MoleculeSmall,
    },
    reflection::DensityPt,
    render::{
        Color, MESH_BOND, MESH_CUBE, MESH_DENSITY_SURFACE, MESH_PEP_SOLVENT_SURFACE, MESH_POCKET,
        MESH_SECONDARY_STRUCTURE, MESH_SPHERE_HIGHRES, MESH_SPHERE_LOWRES, MESH_SPHERE_MEDRES,
    },
    selection::{Selection, ViewSelLevel},
    sfc_mesh::{SOLVENT_RAD, make_sas_mesh},
    state::{OperatingMode, ResColoring, State, StateUi, Visibility},
    util::{aromatic_ring_centroid, clear_mol_entity_indices, find_neighbor_posit, orbit_center},
};
// const LIGAND_COLOR_ANCHOR: Color = (1., 0., 1.);

pub mod atoms_bonds;
pub mod ribbon_mesh;
mod viridis_lut;
pub mod wrappers;

const COLOR_MOL_MOVING: Color = (1., 1., 1.);
const COLOR_MOL_ROTATE: Color = (0.65, 1., 0.65);
const COLOR_MD_NEAR_MOL: Color = (0.0, 0., 1.); // Blended into
const BLEND_AMT_MD_NEAR_MOL: f32 = 0.5; // A higher value means it's closer to the special color.

// i.e a flexible bond.
const LIGAND_COLOR_FLEX: Color = (1., 1., 0.);
pub const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);
pub const COLOR_AA_NON_RESIDUE_EGUI: Color32 = Color32::from_rgb(0, 204, 255);

pub const COLOR_SELECTED: Color = (1., 0., 0.);

const COLOR_WATER_BOND: Color = (0.5, 0.5, 0.8);

const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);

const LABEL_SIZE_ATOM: f32 = 16.;
const LABEL_SIZE_MOL: f32 = 40.;
const LABEL_COLOR_ATOM: (u8, u8, u8, u8) = (255, 60, 160, 255);
const LABEL_COLOR_ATOM_SEL: (u8, u8, u8, u8) = (255, 20, 20, 255);
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

pub const WATER_OPACITY: f32 = 1.;
pub const PHARMACOPHORE_OPACITY: f32 = 0.3;
pub const RADIUS_PHARMACOPHORE_HINT: f32 = 0.25;

const COLOR_DOCKING_BOX: Color = (0.3, 0.3, 0.9);
pub const COLOR_DOCKING_SITE_MESH: Color = (0.5, 0.5, 0.9);
// const DOCKING_SITE_OPACITY: f32 = 0.1;

// These colors are the default solid ones, i.e. if not colored by an atom-etc-based scheme.
const COLOR_SA_SURFACE: Color = (0.3, 0.2, 1.);
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

fn text_overlay_bonds(
    entity: &mut Entity,
    mol_ident: &str,
    i_atom: usize,
    atom: &Atom,
    sel: bool,
    ui: &StateUi,
) {
    if !matches!(
        ui.mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        if ui.visibility.labels_atom_sn {
            entity.overlay_text = Some(TextOverlay {
                text: format!("{}", atom.serial_number),
                size: LABEL_SIZE_ATOM,
                color: LABEL_COLOR_ATOM,
                font_family: FontFamily::Proportional,
            });
        }

        let color = if sel {
            LABEL_COLOR_MOL_SEL
        } else {
            LABEL_COLOR_MOL
        };

        if ui.visibility.labels_mol && i_atom == 0 {
            entity.overlay_text = Some(TextOverlay {
                text: mol_ident.to_string(),
                size: LABEL_SIZE_MOL,
                color,
                font_family: FontFamily::Proportional,
            });
        }
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
        slot.get_or_init(|| blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT))
            .clone()
    } else {
        blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT)
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum MoleculeView {
    Backbone,
    #[default]
    Sticks,
    BallAndStick,
    /// i.e. Van der Waals radius, or CPK.
    SpaceFill,
    Ribbon,
    Surface,
    Dots,
}

impl MoleculeView {
    pub fn next(self) -> Self {
        match self {
            Self::Backbone => Self::Sticks,
            Self::Sticks => Self::BallAndStick,
            Self::BallAndStick => Self::SpaceFill,
            Self::SpaceFill => Self::Surface, // skip ribbon for now
            Self::Ribbon => Self::Surface,
            Self::Surface => Self::Dots,
            Self::Dots => Self::Backbone,
        }
    }

    // todo: repetitive
    pub fn prev(self) -> Self {
        match self {
            Self::Backbone => Self::Dots,
            Self::Sticks => Self::Backbone,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            Self::Ribbon => Self::SpaceFill,
            Self::Surface => Self::SpaceFill,
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

    // todo: repetitive
    pub fn prev_editor(self) -> Self {
        match self {
            Self::Sticks => Self::SpaceFill,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            _ => Self::Sticks,
        }
    }
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
            o_pos[i].into(),
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
                (*pos).into(),
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
                    0,
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
                atoms_bonds::text_overlay_atoms(
                    &mut entity,
                    &mol.common().ident,
                    i_atom,
                    &atom,
                    mol_active,
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
                0,
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
                0,
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
                    if *bond_i == i_bond {
                        color_0 = COLOR_SELECTED;
                        color_1 = COLOR_SELECTED;
                    }
                }
                Selection::BondsLig((_mol_i, bonds_i)) => {
                    if bonds_i.contains(&i_bond) {
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
            true,
            neighbor_posit,
            mol_active,
            to_hydrogen,
        );

        // todo: This seems to be related to a bug where atom labels are doubled for some in sticks mode.
        // Draw atom-based labels on bonds if not in a view mode that shows atoms.
        if !entities.is_empty() && mode != OperatingMode::MolEditor {
            text_overlay_bonds(
                &mut entities[0],
                &mol.common().ident,
                bond.atom_0,
                &atom_0,
                mol_active, // todo
                ui,
            );
        }

        if let MolGenericRef::Small(m) = &mol
            && (!ui.visibility.hide_pharmacophore || mode == OperatingMode::MolEditor)
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

/// A visual representation of volumetric electron density,
/// as loaded from .map files or similar. This is our point-based approach; not the isosurface.
/// We change size based on density, and not linearly, for visual effect.
pub fn draw_density_point_cloud(entities: &mut Vec<Entity>, density: &[DensityPt]) {
    entities.retain(|ent| ent.class != EntityClass::DensityPoint as u32);
    // clear_mol_entity_indices(state); // todo: Borrow mut problem.

    const EPS: f64 = 0.0000001;

    for point in density {
        // For example, points we filter out for not being near the atoms; we set them to 0 density,
        // vice omitting them. Skipping them here makes rendering more efficient.
        if point.density.abs() < EPS {
            continue;
        }

        // Todo: Sort out how you'll handle this. Currently, You discard these, or they'd go NaN
        // todo on the power computation.
        if point.density < 0.0 {
            continue;
        }

        let mut ent = Entity::new(
            MESH_SPHERE_LOWRES,
            point.coords.into(),
            Quaternion::new_identity(),
            0.03 * point.density.powf(1.3) as f32,
            (point.density as f32 * 2., 0.0, 0.2),
            ATOM_SHININESS,
        );
        ent.class = EntityClass::DensityPoint as u32;

        entities.push(ent);
    }
}

/// An isosurface of electron density,
/// as loaded from .map files or similar.
pub fn draw_density_surface(entities: &mut Vec<Entity>, state: &mut State) {
    entities.retain(|ent| ent.class != EntityClass::DensitySurface as u32);
    clear_mol_entity_indices(state, None);

    let mut ent = Entity::new(
        MESH_DENSITY_SURFACE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0., 1., 1.),
        ATOM_SHININESS,
    );
    ent.class = EntityClass::DensitySurface as u32;
    ent.opacity = DENSITY_ISO_OPACITY;
    entities.push(ent);
}

/// The dots view of solvent-accessible-surface
fn draw_dots(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    if scene.meshes[MESH_PEP_SOLVENT_SURFACE].vertices.len() > 1_000_000 {
        eprintln!("Not drawing dots due to a large-mol rendering problem.");
        return;
    }

    for vertex in &scene.meshes[MESH_PEP_SOLVENT_SURFACE].vertices {
        let mut entity = Entity::new(
            MESH_SURFACE_DOT,
            Vec3::from_slice(&vertex.position).unwrap(),
            Quaternion::new_identity(),
            SIZE_SFC_DOT,
            COLOR_SFC_DOT,
            ATOM_SHININESS,
        );
        entity.class = EntityClass::SaSurfaceDots as u32;
        scene.entities.push(entity);
    }
}

/// The mesh view of solvent-accessible-surface
// fn draw_sa_surface(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene, color_by_vertex: Option<Vec<(u8, u8, u8)>>) {
fn draw_sa_surface(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    let mut ent = Entity::new(
        MESH_PEP_SOLVENT_SURFACE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        COLOR_SA_SURFACE,
        ATOM_SHININESS,
    );

    ent.class = EntityClass::SaSurface as u32;
    ent.opacity = SAS_ISO_OPACITY;
    // if let Some(color) = color_by_vertex {
    //     ent.color_by_vertex = Some(color)
    // }

    scene.entities.push(ent);
}

/// Secondary structure, e.g. cartoon.
// pub fn draw_secondary_structure(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene, state: &mut State) {
pub fn draw_secondary_structure(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    scene
        .entities
        .retain(|ent| ent.class != EntityClass::SecondaryStructure as u32);
    // clear_mol_entity_indices(state);

    let mut ent = Entity::new(
        MESH_SECONDARY_STRUCTURE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0.7, 0.2, 1.), // todo: Make this customizable etc.
        ATOM_SHININESS,
    );
    ent.class = EntityClass::SecondaryStructure as u32;
    scene.entities.push(ent);
}

// todo: Move this A/R. Util? Molecule? Method on peptide?
/// Filter by distance to various items. Has some computational complexity.
/// Indexes in the result are filtered out. So, an empty Vec means no restrictions.
// pub fn filter_pep_atoms_by_dist(mol: &MoleculeCommon, ui: &StateUi, lig: Option<&MoleculeCommon>) -> Vec<usize> {
// pub fn filter_pep_atoms_by_dist<'a>(mol: &MoleculeCommon, ui: &StateUi, active_mol: Option<MolGenericRef<'a>>) -> Vec<usize> {
pub fn filter_pep_atoms_by_dist<'a>(
    pep: &MoleculePeptide,
    ui: &StateUi,
    active_mol: Option<MolGenericRef<'a>>,
) -> Vec<usize> {
    let mut result = Vec::new();

    let mol = &pep.common;

    // Speed up computations by using magnitude squared.
    let nearby_dist_thresh_sq = ui.nearby_dist_thresh.pow(2) as f32;

    // todo: Experimenting. I'm not sure why we need this, but the results don't filter enough otherwise.
    let nearby_dist_thresh_sfc_sq = (ui.nearby_dist_thresh / 2).pow(2) as f32;

    if !ui.show_near_lig_only && !ui.show_near_sel_only && !ui.show_near_sfc_only {
        return Vec::new();
    }

    let sfc_pts = if ui.show_near_sfc_only {
        // Higher means faster, but cruder.
        const NEAR_SFC_MESH_PRECISION: f32 = 5.;

        let atoms: Vec<(Vec3, _)> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.hetero)
            .map(|(i, a)| (mol.atom_posits[i].into(), a.element.vdw_radius()))
            .collect();

        // todo: DOn't create this each drawing! Cache the atoms near the sfc pre-computed.
        let mesh = make_sas_mesh(&atoms, SOLVENT_RAD, NEAR_SFC_MESH_PRECISION);
        mesh.vertices
            .iter()
            .map(|v| Vec3::from_slice(&v.position).unwrap())
            .collect()
    } else {
        Vec::new()
    };

    // An optimization: Measure dist^2 once per residue, instead of per atom. This, for better or worse,
    // shows complete residues only.
    if ui.show_near_sfc_only {
        for res in &pep.residues {
            if res.atoms.is_empty() {
                break;
            }

            // Arbitrary; pick two atoms on either SN end for variety. If either is close to the surface,
            // pass all atoms in the residue.
            let res_atom_0 = &res.atoms[0];
            let res_atom_1 = res.atoms.last().unwrap_or(&res.atoms[0]);

            let p_atom_0: Vec3 = mol.atom_posits[*res_atom_0].into();
            let p_atom_1: Vec3 = mol.atom_posits[*res_atom_1].into();

            // Check if near any surface point.
            let mut passed = false;
            for pt in &sfc_pts {
                for posit in [p_atom_0, p_atom_1] {
                    if (*pt - posit).magnitude_squared() < nearby_dist_thresh_sfc_sq {
                        passed = true;
                        break;
                    }
                }
            }

            if !passed {
                for i_atom in &res.atoms {
                    result.push(*i_atom);
                }
                continue;
            }
        }
    }

    for (i_atom, _atom) in mol.atoms.iter().enumerate() {
        let posit = mol.atom_posits[i_atom];

        if ui.show_near_sel_only
            && let Selection::AtomPeptide(i_sel) = &ui.selection
        {
            // todo: This will fail after moves and dynamics. You must pick the selected atom
            // todo posit correctly!

            if (posit - mol.atom_posits[*i_sel]).magnitude_squared() as f32 > nearby_dist_thresh_sq
            {
                result.push(i_atom);
                continue;
            }
        }

        if ui.show_near_lig_only
            && let Some(ref lig) = active_mol
        {
            let atom_sel = lig.common().atom_posits[0]; // todo: Centroid?

            if (posit - atom_sel).magnitude_squared() as f32 > nearby_dist_thresh_sq {
                result.push(i_atom);
                continue;
            }
        }

        // Per-atom path on this; we instead use the above res-based algo to improve speed.
        // if ui.show_near_sfc_only {
        //     let p_atom: Vec3 = posit.into();
        //
        //     // Check if near any surface point.
        //     let mut passed = false;
        //     for pt in &sfc_pts {
        //         if (*pt - p_atom).magnitude_squared() < nearby_dist_thresh_sfc_sq {
        //             passed = true;
        //             break;
        //         }
        //     }
        //     if !passed {
        //         result.push(i_atom);
        //         continue;
        //     }
        // }
    }

    result
}

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_peptide(state: &mut State, scene: &mut Scene) {
    // todo: You may wish to integrate Cartoon into this workflow.
    let initial_ent_count = scene.entities.len();

    let mol_i = 0; // todo for now.

    let mol_active = if let Some((active_mol_type, active_i)) = state.volatile.active_mol {
        MolType::Peptide == active_mol_type && mol_i == active_i
    } else {
        false
    };

    scene.entities.retain(|ent| {
        ent.class != EntityClass::Protein as u32
            && ent.class != EntityClass::SaSurface as u32
            && ent.class != EntityClass::SaSurfaceDots as u32
    });

    // Edit small molecules only; not proteins.
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        return;
    }

    let Some(mol) = state.peptide.as_ref() else {
        return;
    };

    if !mol.common.visible {
        return;
    }

    let filtered_out_by_dist = filter_pep_atoms_by_dist(&mol, &state.ui, state.active_mol());

    let start_i = scene.entities.len();
    let mut entities = Vec::new();

    // todo:  Unless colored by res #, set to 0 to save teh computation.
    let aa_count = mol
        .residues
        .iter()
        .filter(|r| {
            if let ResidueType::AminoAcid(_) = r.res_type {
                true
            } else {
                false
            }
        })
        .count();

    let ui = &state.ui;

    if ui.mol_view == MoleculeView::Ribbon {
        draw_secondary_structure(
            &mut state.volatile.flags.update_ss_mesh,
            state.volatile.flags.ss_mesh_created,
            scene,
        );
    }

    // Note that this renders over a sticks model.
    if !state.ui.visibility.hide_protein && ui.mol_view == MoleculeView::Dots {
        draw_dots(
            &mut state.volatile.flags.update_sas_mesh,
            state.volatile.flags.sas_mesh_created,
            scene,
        );
    }

    // todo: Consider if you handle this here, or in a sep fn.
    if !state.ui.visibility.hide_protein && ui.mol_view == MoleculeView::Surface {
        draw_sa_surface(
            &mut state.volatile.flags.update_sas_mesh,
            state.volatile.flags.sas_mesh_created,
            scene,
        );
    }

    let chains_invis: Vec<&Chain> = mol.chains.iter().filter(|c| !c.visible).collect();

    let sel = if ui.selection.is_bond() {
        &Selection::None
    } else {
        &ui.selection
    };

    // If sticks view, draw water molecules as balls.
    if matches!(
        ui.mol_view,
        MoleculeView::Sticks | MoleculeView::BallAndStick
    ) && !state.ui.visibility.hide_water
    {
        for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
            if atom.hetero {
                // todo: Excessive nesting.
                if let Some(role) = atom.role
                    && role == AtomRole::Water
                {
                    let color_atom = atoms_bonds::atom_color(
                        atom,
                        0,
                        i_atom,
                        &mol.residues,
                        aa_count,
                        sel,
                        state.ui.view_sel_level,
                        false,
                        ResColoring::default(),
                        false,
                        MolType::Peptide,
                        &None,
                    );

                    let mut entity = Entity::new(
                        MESH_WATER_SPHERE,
                        mol.common.atom_posits[i_atom].into(),
                        Quaternion::new_identity(),
                        BALL_RADIUS_WATER_O,
                        color_atom,
                        ATOM_SHININESS,
                    );

                    entity.class = EntityClass::Protein as u32;
                    entities.push(entity);
                }
            }
        }
    }

    // Draw atoms.
    if matches!(
        ui.mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
            if atom.hetero {
                let mut water = false;
                if let Some(role) = atom.role {
                    water = role == AtomRole::Water;
                }
                if !water && ui.mol_view == MoleculeView::SpaceFill {
                    // Don't draw VDW spheres for hetero atoms; draw as sticks.
                    continue;
                }
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&i_atom) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            if state.ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
                continue;
            }

            if filtered_out_by_dist.contains(&i_atom) {
                continue;
            }

            if let Some(role) = atom.role {
                if (state.ui.visibility.hide_sidechains
                    || state.ui.mol_view == MoleculeView::Backbone)
                    && matches!(role, AtomRole::Sidechain | AtomRole::H_Sidechain)
                {
                    continue;
                }
                if (state.ui.visibility.hide_water || ui.mol_view == MoleculeView::SpaceFill)
                    && role == AtomRole::Water
                {
                    continue;
                }
            }

            if (state.ui.visibility.hide_hetero && atom.hetero)
                || (state.ui.visibility.hide_protein && !atom.hetero)
            {
                continue;
            }

            let atom_posit = mol.common.atom_posits[i_atom];

            // todo: Use your new peptide field for filtered, instead of computing these each time.

            let (mut radius, mesh) = match ui.mol_view {
                MoleculeView::SpaceFill => (atom.element.vdw_radius(), MESH_SPACEFILL_SPHERE),
                _ => match atom.element {
                    Element::Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
                    _ => (BALL_STICK_RADIUS, MESH_BALL_STICK_SPHERE),
                },
            };

            if let Some(role) = atom.role {
                if role == AtomRole::Water {
                    radius = BALL_RADIUS_WATER_O
                }
            }

            let dim_peptide = state.ui.visibility.dim_peptide && !atom.hetero;

            let mut color_atom = (0., 0., 0.);
            let mut manip_active = false;

            match state.volatile.mol_manip.mode {
                ManipMode::Move((mol_type, i)) => {
                    if mol_type == MolType::Peptide && i == mol_i {
                        color_atom = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                }
                ManipMode::Rotate((mol_type, i)) => {
                    if mol_type == MolType::Peptide && i == mol_i {
                        color_atom = COLOR_MOL_ROTATE;
                        manip_active = true;
                    }
                }
                ManipMode::None => (),
            }

            if !manip_active {
                color_atom = atoms_bonds::atom_color(
                    atom,
                    0,
                    i_atom,
                    &mol.residues,
                    aa_count,
                    sel,
                    state.ui.view_sel_level,
                    dim_peptide,
                    state.ui.res_coloring,
                    state.ui.atom_color_by_charge,
                    MolType::Peptide,
                    &None,
                );
            }

            if atom.hetero && color_atom != COLOR_SELECTED {
                color_atom = blend_color(color_atom, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
            }

            if state.mol_dynamics.is_some()
                && state.ui.md.peptide_only_near_ligs
                && mol.common.selected_for_md
                && state
                    .ligands
                    .iter()
                    .filter(|l| l.common.selected_for_md)
                    .count()
                    != 0
                && state.volatile.md_peptide_selected.contains(&(0, i_atom))
            {
                color_atom = blend_color(color_atom, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            }

            let mut entity = Entity::new(
                mesh,
                atom_posit.into(),
                Quaternion::new_identity(),
                radius,
                color_atom,
                ATOM_SHININESS,
            );

            // Note: We draw these on the bond entities if not in a view that shows atoms.
            atoms_bonds::text_overlay_atoms(
                &mut entity,
                &mol.common.ident,
                i_atom,
                &atom,
                mol_active,
                ui,
            );

            entity.class = EntityClass::Protein as u32;
            entities.push(entity);
        }
    }

    // For determining inside of rings.
    let mut hydrogen_is = Vec::with_capacity(mol.common.atoms.len());
    for atom in &mol.common.atoms {
        hydrogen_is.push(atom.element == Element::Hydrogen);
    }

    // Aromatic-only adjacency list for ring centroid BFS.
    let aromatic_adj = {
        let n = mol.common.atoms.len();
        let mut adj = vec![Vec::new(); n];
        for b in &mol.common.bonds {
            if b.bond_type == BondType::Aromatic {
                adj[b.atom_0].push(b.atom_1);
                adj[b.atom_1].push(b.atom_0);
            }
        }
        adj
    };

    // Draw bonds.
    for (i_bond, bond) in mol.common.bonds.iter().enumerate() {
        if ui.mol_view == MoleculeView::Backbone && !bond.is_backbone {
            continue;
        }

        let atom_0 = &mol.common.atoms[bond.atom_0];
        let atom_1 = &mol.common.atoms[bond.atom_1];

        let atom_0_posit = mol.common.atom_posits[bond.atom_0];
        let atom_1_posit = mol.common.atom_posits[bond.atom_1];

        // Don't draw bonds if on the spacefill view, and the atoms aren't hetero.
        if ui.mol_view == MoleculeView::SpaceFill && !atom_0.hetero && !atom_1.hetero {
            continue;
        }

        if filtered_out_by_dist.contains(&bond.atom_0) {
            continue;
        }

        let mut chain_not_sel = false;
        for chain in &chains_invis {
            if chain.atoms.contains(&bond.atom_0) {
                chain_not_sel = true;
                break;
            }
        }
        if chain_not_sel {
            continue;
        }

        if state.ui.visibility.hide_hydrogen
            && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
        {
            continue;
        }

        // Assuming water won't be bonded to the main molecule.
        if (state.ui.visibility.hide_sidechains || state.ui.mol_view == MoleculeView::Backbone)
            && let Some(role_0) = atom_0.role
        {
            if let Some(role_1) = atom_1.role {
                if role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain {
                    continue;
                }
            }
        }

        if (state.ui.visibility.hide_hetero && atom_0.hetero && atom_1.hetero)
            || (state.ui.visibility.hide_protein && !atom_0.hetero && !atom_1.hetero)
        {
            continue;
        }

        let posit_0: Vec3 = atom_0_posit.into();
        let posit_1: Vec3 = atom_1_posit.into();

        // For determining how to orient multiple-bonds.
        let neighbor_posit = if bond.bond_type == BondType::Aromatic {
            let centroid: Vec3 = aromatic_ring_centroid(
                &aromatic_adj,
                &mol.common.atom_posits,
                bond.atom_0,
                bond.atom_1,
                &hydrogen_is,
            )
            .map(|c| c.into())
            .unwrap_or_else(|| mol.common.atom_posits[0].into());
            (centroid, false)
        } else {
            let neighbor_i = find_neighbor_posit(
                &mol.common.adjacency_list,
                bond.atom_0,
                bond.atom_1,
                &hydrogen_is,
            );
            match neighbor_i {
                Some((i, p1)) => (mol.common.atom_posits[i].into(), p1),
                None => (mol.common.atom_posits[0].into(), false),
            }
        };

        let dim_peptide_0 =
            state.ui.visibility.dim_peptide && !mol.common.atoms[bond.atom_0].hetero;
        let dim_peptide_1 =
            state.ui.visibility.dim_peptide && !mol.common.atoms[bond.atom_1].hetero;

        let mut color_0 = (0., 0., 0.);
        let mut color_1 = (0., 0., 0.);

        let mut manip_active = false;

        match state.volatile.mol_manip.mode {
            ManipMode::Move((mol_type, i)) => {
                if mol_type == MolType::Peptide && i == mol_i {
                    color_0 = COLOR_MOL_MOVING;
                    color_1 = COLOR_MOL_MOVING;
                    manip_active = true;
                }
            }
            ManipMode::Rotate((mol_type, i)) => {
                if mol_type == MolType::Peptide && i == mol_i {
                    color_0 = COLOR_MOL_ROTATE;
                    color_1 = COLOR_MOL_ROTATE;
                    manip_active = true;
                }
            }
            ManipMode::None => (),
        }

        if !manip_active {
            color_0 = atoms_bonds::atom_color(
                atom_0,
                0,
                bond.atom_0,
                &mol.residues,
                aa_count,
                sel,
                state.ui.view_sel_level,
                dim_peptide_0,
                state.ui.res_coloring,
                state.ui.atom_color_by_charge,
                MolType::Peptide,
                &None,
            );
            color_1 = atoms_bonds::atom_color(
                atom_1,
                0,
                bond.atom_1,
                &mol.residues,
                aa_count,
                sel,
                state.ui.view_sel_level,
                dim_peptide_1,
                state.ui.res_coloring,
                state.ui.atom_color_by_charge,
                MolType::Peptide,
                &None,
            );
        }

        if let Selection::BondPeptide(bond_i) = ui.selection {
            if bond_i == i_bond {
                color_0 = COLOR_SELECTED;
                color_1 = COLOR_SELECTED;
            }
        }

        if atom_0.hetero && color_0 != COLOR_SELECTED {
            color_0 = blend_color(color_0, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        if atom_1.hetero && color_1 != COLOR_SELECTED {
            color_1 = blend_color(color_1, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        if state.mol_dynamics.is_some()
            && state.ui.md.peptide_only_near_ligs
            && mol.common.selected_for_md
            && state
                .ligands
                .iter()
                .filter(|l| l.common.selected_for_md)
                .count()
                != 0
        {
            if state
                .volatile
                .md_peptide_selected
                .contains(&(0, bond.atom_0))
            {
                color_0 = blend_color(color_0, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            }
            if state
                .volatile
                .md_peptide_selected
                .contains(&(0, bond.atom_1))
            {
                color_1 = blend_color(color_1, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            }
        }

        let to_hydrogen =
            atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen;

        let mut ents_new = atoms_bonds::bond_entities(
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            MolType::Peptide,
            state.ui.mol_view != MoleculeView::BallAndStick,
            neighbor_posit,
            false,
            to_hydrogen,
        );

        if !ents_new.is_empty() {
            text_overlay_bonds(
                &mut ents_new[0],
                &mol.common.ident,
                bond.atom_0,
                &atom_0,
                mol_active,
                ui,
            );
        }

        entities.extend(ents_new);
    }

    // Draw H bonds.
    // todo: DRY with Ligand
    // todo: This incorrectly hides hetero-only H bonds.
    if !state.ui.visibility.hide_h_bonds
        && !state.ui.visibility.hide_protein
        && state.ui.mol_view != MoleculeView::SpaceFill
    {
        for bond in &mol.bonds_hydrogen {
            let atom_donor = &mol.common.atoms[bond.donor];
            let atom_acceptor = &mol.common.atoms[bond.acceptor];

            // todo: DRY with above.
            if state.ui.visibility.hide_sidechains || state.ui.mol_view == MoleculeView::Backbone {
                if let Some(role_0) = atom_donor.role
                    && let Some(role_1) = atom_acceptor.role
                    && (role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain)
                {
                    continue;
                }
            }

            // todo: Should we pre-filter these atoms-to-disp by index? Would be faster, but
            // todo I don't wish to expend the effort on that here.

            if filtered_out_by_dist.contains(&bond.donor) {
                continue;
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&bond.donor) || chain.atoms.contains(&bond.acceptor) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            if state.ui.visibility.hide_water {
                if let Some(role) = atom_donor.role
                    && role == AtomRole::Water
                {
                    continue;
                }
                if let Some(role) = atom_acceptor.role
                    && role == AtomRole::Water
                {
                    continue;
                }
            }
            entities.extend(draw_hydrogen_bond(
                mol.common.atom_posits[bond.donor].into(),
                mol.common.atom_posits[bond.acceptor].into(),
                MolType::Peptide,
                bond.strength,
            ));
        }
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }

    scene.entities.extend(entities);

    let end_i = scene.entities.len();
    if let Some(mol) = state.peptide.as_mut() {
        mol.common.entity_i_range = Some((start_i, end_i));
    } else {
        eprintln!("Uhoh!")
    }

    if scene.entities.len() != initial_ent_count {
        clear_mol_entity_indices(state, None);
    }
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
        MESH_POCKET,
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
            ));
        }
    }

    res
}
