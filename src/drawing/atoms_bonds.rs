use bio_files::BondType;
use egui::FontFamily;
use graphics::{Entity, TextOverlay, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use na_seq::Element;

use crate::{
    drawing,
    drawing::{
        CHARGE_MAP_MAX, CHARGE_MAP_MIN, COLOR_SELECTED, DIMMED_PEPTIDE_AMT, LABEL_COLOR_ATOM,
        LABEL_COLOR_MOL, LABEL_COLOR_MOL_SEL, LABEL_SIZE_ATOM, LABEL_SIZE_MOL, MESH_BOND_CAP,
    },
    molecules::{Atom, MolType, Residue},
    render::{BACKGROUND_COLOR, Color, MESH_BOND},
    selection::{Selection, ViewSelLevel},
    state::{ResColoring, StateUi},
    util::res_color,
};

pub const BOND_RADIUS_BASE: f32 = 0.10;
// These ratios scale the base bond radius above.
const BOND_RADIUS_LIG_RATIO: f32 = 1.3;
// Of bond radius.
const BOND_RADIUS_LIG_RATIO_SEL: f32 = 1.6;
// Of bond radius.
const BOND_RADIUS_RATIO_H: f32 = 0.5;
// Of bond radius. Covalent to H.
const BOND_RADIUS_LIPID_RATIO: f32 = 0.6;
// Aromatic inner radius, relative to bond radius.
const BOND_RADIUS_AR_INNER_RATIO: f32 = 0.4;
// Of bond radius.
const AR_INNER_OFFSET: f32 = 0.3;
// Å
// const AR_INNER_SHORTEN_FACTOR: f32 = 0.7;
const AR_SHORTEN_AMT: f32 = 0.4;
// Å. Applied to each half.
const DBL_BOND_OFFSET: f32 = 0.1;
// Two of these is the separation.
const TRIPLE_BOND_OFFSET: f32 = 0.1;

// todo: Shinyness broken?
pub const ATOM_SHININESS: f32 = 0.9;
pub(in crate::drawing) const BODY_SHINYNESS: f32 = 0.9;

pub const BALL_STICK_RADIUS: f32 = 0.3;
pub const BALL_STICK_RADIUS_H: f32 = 0.1;

pub(in crate::drawing) const BALL_RADIUS_WATER_O: f32 = 0.09;
pub(in crate::drawing) const BALL_RADIUS_WATER_H: f32 = 0.06;
pub(in crate::drawing) const WATER_BOND_THICKNESS: f32 = 0.1;

const COLOR_H_BOND: Color = (1., 0.5, 0.1);
const RADIUS_H_BOND: f32 = 0.2; // A scaler relative to covalent sticks.
// For the central cylinder which indicates strength
const RADIUS_H_BOND_CENTER: f32 = 0.8; // A scaler relative to covalent sticks.

const H_BOND_DASH_LEN: f32 = 0.15; // Å
const H_BOND_GAP_LEN: f32 = 0.15; // Å
// Maximum length of the central strength-indicating cylinder, at strength = 1.
const H_BOND_CENTER_LEN_MAX: f32 = 1.5; // Å

// todo: For this and overlay bonds: Make teh atom color based on if the atom is selected;
// todo not if the bond is selected.
pub fn text_overlay_atoms(
    entity: &mut Entity,
    mol_ident: &str,
    i_atom: usize,
    atom: &Atom,
    sel: bool,
    ui: &StateUi,
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
            text: format!("{}", mol_ident),
            size: LABEL_SIZE_MOL,
            color,
            font_family: FontFamily::Proportional,
        });
    }
}

/// A general function that sets color based criteria like element, partial charge,
/// if selected or not etc.
pub fn atom_color(
    atom: &Atom,
    mol_i: usize,
    i: usize,
    residues: &[Residue],
    aa_count: usize, // # AA residues; used for color-mapping.
    selection: &Selection,
    view_sel_level: ViewSelLevel,
    dimmed: bool,
    res_coloring: ResColoring,
    atom_color_by_q: bool,
    mol_type: MolType,
) -> Color {
    let mut result = match view_sel_level {
        ViewSelLevel::Atom => {
            if atom_color_by_q {
                if let Some(q) = atom.partial_charge {
                    drawing::color_viridis_float(q, CHARGE_MAP_MIN, CHARGE_MAP_MAX)
                } else {
                    // Don't revert to atom color, as that could be misinterpreted.
                    (0.5, 0.5, 0.5)
                }
            } else {
                atom.element.color()
            }
        }
        ViewSelLevel::Residue => {
            let mut color = Element::Hydrogen.color(); // todo temp workaround for a bug we haven't tracked down.

            if let Some(res_i) = &atom.residue {
                let res = &residues[*res_i];
                color = res_color(res, res_coloring, atom.residue, aa_count);

                // Todo: WOrkaround for a problem we're having with Hydrogen's showing like hetero atoms
                // todo in residue mode. Likely due to them not having their AA set.
                // our workaround of setting to 1, 1, 1 is taking effect instead...
                // This sets white vice the residue color, but this may be OK.
                if atom.element == Element::Hydrogen {
                    color = atom.element.color();
                }
            }
            color
        }
        ViewSelLevel::Bond => atom.element.color(), // Handled elsewhere.
    };

    // If selected, the selected color overrides the element or residue color.
    match selection {
        Selection::AtomPeptide(sel_i) => {
            if mol_type == MolType::Peptide && *sel_i == i {
                result = COLOR_SELECTED;
            }
        }
        Selection::Residue(sel_i) => {
            if let Some(res_i) = atom.residue
                && res_i == *sel_i
            {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomsPeptide(sel_is) => {
            if sel_is.contains(&i) {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomLig((i_mol, i_atom)) => {
            if mol_type == MolType::Ligand && *i_atom == i && *i_mol == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomsLig((i_mol, is_atom)) => {
            if mol_type == MolType::Ligand && is_atom.contains(&i) && *i_mol == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomNucleicAcid((lig_i, sel_i)) => {
            if mol_type == MolType::NucleicAcid && *sel_i == i && *lig_i == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomLipid((lig_i, sel_i)) => {
            if mol_type == MolType::Lipid && *sel_i == i && *lig_i == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::BondLig((lig_i, sel_i)) => {
            if mol_type == MolType::Ligand && *sel_i == i && *lig_i == mol_i {
                result = COLOR_SELECTED;
            }
        }
        _ => (),
    }

    if dimmed && result != COLOR_SELECTED {
        // Desaturate first; otherwise the more saturated initial colors will be relatively visible, while unsaturated
        // ones will appear blackish.
        result = drawing::blend_color(result, BACKGROUND_COLOR, DIMMED_PEPTIDE_AMT)
    }

    result
}

/// Adds a cylindrical bond. This is divided into two halves, so they can be color-coded by their side's
/// atom. Adds optional rounding. `thickness` is relative to BOND_RADIUS.
fn add_bond(
    posits: (Vec3, Vec3),
    colors: (Color, Color),
    center: Vec3,
    orientation: Quaternion,
    dist_half: f32,
    caps: bool,
    radius_scaler: f32,
    mol_type: MolType,
) -> Vec<Entity> {
    let mut result = Vec::new();

    // Split the bond into two entities, so you can color-code them separately based
    // on which atom the half is closer to.
    let center_0 = (posits.0 + center) / 2.;
    let center_1 = (posits.1 + center) / 2.;

    let entity_type = mol_type.entity_type() as u32;

    let mut entity_0 = Entity::new(
        MESH_BOND,
        center_0,
        orientation,
        1.,
        colors.0,
        BODY_SHINYNESS,
    );
    entity_0.class = entity_type;

    let mut entity_1 = Entity::new(
        MESH_BOND,
        center_1,
        orientation,
        1.,
        colors.1,
        BODY_SHINYNESS,
    );
    entity_1.class = entity_type;

    if caps {
        // These spheres are to put a rounded cap on each bond.
        // todo: You only need a dome; performance implications.
        let mut cap_0 = Entity::new(
            MESH_BOND_CAP,
            posits.0,
            Quaternion::new_identity(),
            BOND_RADIUS_BASE * radius_scaler,
            colors.0,
            BODY_SHINYNESS,
        );
        let mut cap_1 = Entity::new(
            MESH_BOND_CAP,
            posits.1,
            Quaternion::new_identity(),
            BOND_RADIUS_BASE * radius_scaler,
            colors.1,
            BODY_SHINYNESS,
        );

        cap_0.class = entity_type;
        cap_1.class = entity_type;
        result.push(cap_0);
        result.push(cap_1);
    }

    let scale = Some(Vec3::new(radius_scaler, dist_half, radius_scaler));
    entity_0.scale_partial = scale;
    entity_1.scale_partial = scale;

    result.push(entity_0);
    result.push(entity_1);

    result
}

pub fn bond_entities(
    posit_0: Vec3,
    posit_1: Vec3,
    color_0: Color,
    color_1: Color,
    bond_type: BondType,
    mol_type: MolType,
    // No caps for ball and stick
    caps: bool,
    // A Neighbor, in the case of aromataic, double bonds, and triple bonds. We use this to determine how
    // to orient the bond meshes, e.g. in plane with a ring. Second pararm is if the bond is from posit 1,
    // vice posit 0.
    neighbor: (Vec3, bool),
    mol_active: bool,  // i.e. selected
    to_hydrogen: bool, // A covalent bond to H
) -> Vec<Entity> {
    let mut result = Vec::new();

    let center: Vec3 = (posit_0 + posit_1) / 2.;

    let diff = posit_0 - posit_1;
    let diff_unit = diff.to_normalized();
    let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
    let dist_half = diff.magnitude() / 2.;

    let mut radius_scaler = 1.;

    match mol_type {
        MolType::Ligand | MolType::NucleicAcid => {
            if mol_active {
                radius_scaler *= BOND_RADIUS_LIG_RATIO_SEL
            } else {
                radius_scaler *= BOND_RADIUS_LIG_RATIO;
            }
        }
        MolType::Lipid => {
            if mol_active {
                radius_scaler *= BOND_RADIUS_LIG_RATIO_SEL
            } else {
                radius_scaler *= BOND_RADIUS_LIPID_RATIO
            }
        }
        _ => (),
    }

    if to_hydrogen {
        radius_scaler *= BOND_RADIUS_RATIO_H;
    }

    match bond_type {
        // Draw a normal mesh, the same as a single, and a second thinner and shorter inner one.
        BondType::Aromatic => {
            // Compute the dihedral angle so we always place the smaller, offset bond on the inside.

            let (posit_0_inner, posit_1_inner, center_inner, dist_half_inner) = {
                // A vector perpendicular to the plane of the bonds (e.g. the ring)
                // This direction only works in some cases; need a more reliable way.
                // Note: This has problems at the connections to rings, but is works for most aromatic
                // bonds. WHen it fails, it shows the shorter part on the outside.
                let perp_vec = if neighbor.1 {
                    diff.cross(neighbor.0 - posit_1)
                } else {
                    diff.cross(neighbor.0 - posit_0)
                }
                .to_normalized();

                let dir_in = perp_vec.cross(diff.to_normalized()).to_normalized();
                let offset = dir_in * AR_INNER_OFFSET;

                let mut p0 = posit_0 + offset;
                let mut p1 = posit_1 + offset;

                // Make the length shorter.
                let diff = p1 - p0;
                let dist = diff.magnitude();
                let dir = diff / dist;

                let shorten_vec = dir * AR_SHORTEN_AMT;
                p0 += shorten_vec;
                p1 -= shorten_vec;

                (p0, p1, center + offset, (p1 - p0).magnitude() / 2.)
            };

            let thickness_outer = radius_scaler;
            let thickness_inner = radius_scaler * BOND_RADIUS_AR_INNER_RATIO;

            // Primary bond exactly like a normal single bond.
            result.extend(add_bond(
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                thickness_outer,
                mol_type,
            ));

            // Smaller bond on the inside.
            result.extend(add_bond(
                (posit_0_inner, posit_1_inner),
                (color_0, color_1),
                center_inner,
                orientation,
                dist_half_inner,
                caps,
                thickness_inner,
                mol_type,
            ));
        }
        BondType::Double => {
            // Draw two offset bond cylinders.
            // See notes above in the Aromatic section.

            let (offset_a, offset_b) = {
                // The compare doesn't matter here, as it's symmetric.
                let perp_vec = diff.cross(posit_1 - neighbor.0).to_normalized();

                let dir_in = perp_vec.cross(diff.to_normalized()).to_normalized();
                let offset_a = dir_in * DBL_BOND_OFFSET;
                let offset_b = -dir_in * DBL_BOND_OFFSET;

                (offset_a, offset_b)
            };

            result.extend(add_bond(
                (posit_0 + offset_a, posit_1 + offset_a),
                (color_0, color_1),
                center + offset_a,
                orientation,
                dist_half,
                caps,
                0.5,
                mol_type,
            ));

            result.extend(add_bond(
                (posit_0 + offset_b, posit_1 + offset_b),
                (color_0, color_1),
                center + offset_b,
                orientation,
                dist_half,
                caps,
                0.5,
                mol_type,
            ));
        }
        BondType::Triple => {
            // Draw two offset bond cylinders.
            // todo: DRY
            let (offset_a, offset_b) = {
                // The compare doesn't matter here, as it's symmetric.
                let perp_vec = diff.cross(posit_1 - neighbor.0).to_normalized();

                let dir_in = perp_vec.cross(diff.to_normalized()).to_normalized();
                let offset_a = dir_in * TRIPLE_BOND_OFFSET;
                let offset_b = -dir_in * TRIPLE_BOND_OFFSET;

                (offset_a, offset_b)
            };

            result.extend(add_bond(
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                0.4,
                mol_type,
            ));
            result.extend(add_bond(
                (posit_0 + offset_a, posit_1 + offset_a),
                (color_0, color_1),
                center + offset_a,
                orientation,
                dist_half,
                caps,
                0.4,
                mol_type,
            ));
            result.extend(add_bond(
                (posit_0 + offset_b, posit_1 + offset_b),
                (color_0, color_1),
                center + offset_b,
                orientation,
                dist_half,
                caps,
                0.4,
                mol_type,
            ));
        }
        // Single bonds, and others.
        _ => {
            result.extend(add_bond(
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                radius_scaler,
                mol_type,
            ));
        }
    }

    result
}

// todo: You may need to make this not dependent on the h bond struct so you can handle
// todo also teh h boudn two mols variant
/// Draws a hydrogen bond as a dashed cylinder with a thicker central segment whose length
/// indicates bond strength, similar to MOE's visualization.
pub(in crate::drawing) fn draw_hydrogen_bond(
    posit_donor: Vec3,
    posit_acc: Vec3,
    mol_type: MolType,
    strength: f32,
) -> Vec<Entity> {
    let mut result = Vec::new();

    let diff = posit_donor - posit_acc;
    let total_len = diff.magnitude();
    if total_len < 0.001 {
        return result;
    }
    let dir = diff / total_len; // Unit vector from acceptor towards donor.
    let orientation = Quaternion::from_unit_vecs(UP_VEC, dir);
    let entity_type = mol_type.entity_type() as u32;
    let center = (posit_donor + posit_acc) / 2.;

    // Central thicker cylinder indicating strength. Its length scales with `strength` (0..1).
    let center_len = (strength.clamp(0., 1.) * H_BOND_CENTER_LEN_MAX).min(total_len * 0.4);

    let mut center_entity = Entity::new(
        MESH_BOND,
        center,
        orientation,
        1.,
        COLOR_H_BOND,
        BODY_SHINYNESS,
    );
    center_entity.class = entity_type;
    center_entity.scale_partial = Some(Vec3::new(
        RADIUS_H_BOND_CENTER,
        center_len,
        RADIUS_H_BOND_CENTER,
    ));
    result.push(center_entity);

    // Dashes on each side, from the outer atom towards the central cylinder.
    let center_half = center_len / 2.;
    let side_len = total_len / 2. - center_half;

    for side in 0..2 {
        let start = if side == 0 { posit_donor } else { posit_acc };
        // Direction from this atom towards the bond center.
        let inward = if side == 0 { -dir } else { dir };

        let mut offset = 0.0_f32;
        while offset < side_len {
            let remaining = side_len - offset;
            if remaining < 0.01 {
                break;
            }
            let dash_len = H_BOND_DASH_LEN.min(remaining);
            let dash_center = start + inward * (offset + dash_len / 2.);

            let mut dash = Entity::new(
                MESH_BOND,
                dash_center,
                orientation,
                1.,
                COLOR_H_BOND,
                BODY_SHINYNESS,
            );
            dash.class = entity_type;
            dash.scale_partial = Some(Vec3::new(RADIUS_H_BOND, dash_len, RADIUS_H_BOND));
            result.push(dash);

            offset += dash_len + H_BOND_GAP_LEN;
        }
    }

    result
}
