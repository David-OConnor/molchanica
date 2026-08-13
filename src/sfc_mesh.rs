//! Colouring for the meshes that wrap molecules — solvent-accessible surfaces, pockets, and the
//! like. The surfaces themselves are built in `mol_defs::sfc_mesh`; how we *shade* them is a
//! presentation concern, so it lives here alongside the rest of our drawing code.

use std::{collections::HashMap, time::Instant};

use graphics::{EngineUpdates, Mesh};
use lin_alg::f32::Vec3 as Vec3F32;
use mol_defs::{
    molecules::common::MoleculeCommon,
    sfc_mesh::{MeshColoring, MeshColors},
};
use na_seq::Element::{self, Hydrogen};

use crate::drawing::{CHARGE_MAP_MAX, CHARGE_MAP_MIN, SAS_ISO_OPACITY, color_viridis_float};

const LIPOPHILICITY_MIN: f32 = -1.5;
const LIPOPHILICITY_MAX: f32 = 1.5;

/// Atomic contribution to lipophilicity, based on element. Positive = hydrophobic, negative = hydrophilic.
fn atom_lipophilicity(element: Element) -> f32 {
    match element {
        Element::Carbon => 0.7,
        Element::Nitrogen => -1.0,
        Element::Oxygen => -1.2,
        Element::Sulfur => 0.2,
        Element::Fluorine => 0.4,
        Element::Chlorine => 0.6,
        Element::Bromine => 0.6,
        Element::Phosphorus => -0.5,
        Element::Hydrogen => 0.0,
        _ => 0.0,
    }
}

// todo: The optimizations are LLM mess

fn cell_key(p: Vec3F32, cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    (
        (p.x * inv).floor() as i32,
        (p.y * inv).floor() as i32,
        (p.z * inv).floor() as i32,
    )
}

/// We use this to apply coloring to meshes that surround molecules. For example, based on the atoms
/// and residues near them. Can color by residue position, greasiness, atom element, or partial charge.
/// Can be used for Protein SAS meshes, pockets, etc.
///
/// In the case of element-based coloring, we omit Hydrogens.
/// Returns a Vec of vertex colors, or none if there is no change.
pub fn get_mesh_colors(
    mesh: &Mesh,
    mol: &MoleculeCommon,
    coloring: MeshColoring,
    engine_updates: &mut EngineUpdates,
) -> Option<MeshColors> {
    if coloring == MeshColoring::Solid {
        return None;
    }

    println!("Loading SAS mesh coloring...");
    let start = Instant::now();

    let opacity = (SAS_ISO_OPACITY * 255.) as u8;

    const GRID_CELL_SIZE: f32 = 3.0;

    // Cache f32 positions once; avoids repeated f64→f32 conversion.
    let posits_f32: Vec<Vec3F32> = mol.atom_posits.iter().map(|p| (*p).into()).collect();

    // Build spatial grid of non-hydrogen atom indices.
    let mut atom_grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, &ap) in posits_f32.iter().enumerate() {
        if mol.atoms[i].element == Hydrogen {
            continue;
        }
        let k = cell_key(ap, GRID_CELL_SIZE);
        atom_grid.entry(k).or_default().push(i);
    }

    // Color each SAS vertex by its nearest atom.
    let result: MeshColors = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let vp = Vec3F32::from_slice(&vertex.position).unwrap();
            let (cx, cy, cz) = cell_key(vp, GRID_CELL_SIZE);

            let mut closest_atom_dist = f32::INFINITY;
            let mut closest_atom = None;

            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(cands) = atom_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &i in cands {
                                let dist = (posits_f32[i] - vp).magnitude_squared();
                                if dist < closest_atom_dist {
                                    closest_atom_dist = dist;
                                    closest_atom = Some(i);
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: full scan (rare; only if grid neighborhood is empty).
            if closest_atom.is_none() {
                for (i, &ap) in posits_f32.iter().enumerate() {
                    if mol.atoms[i].element == Hydrogen {
                        continue;
                    }
                    let dist = (ap - vp).magnitude_squared();
                    if dist < closest_atom_dist {
                        closest_atom_dist = dist;
                        closest_atom = Some(i);
                    }
                }
            }

            if let Some(i) = closest_atom {
                let atom = &mol.atoms[i];

                let (r, g, b, a) = match coloring {
                    MeshColoring::Element => {
                        let (r, g, b) = atom.element.color();
                        (r, g, b, opacity)
                    }
                    MeshColoring::PartialCharge => {
                        if let Some(q) = atom.partial_charge {
                            let (r, g, b) = color_viridis_float(q, CHARGE_MAP_MIN, CHARGE_MAP_MAX);
                            (r, g, b, opacity)
                        } else {
                            (0., 0., 0., 0)
                        }
                    }
                    MeshColoring::Lipophilicity => {
                        let lipo = atom_lipophilicity(atom.element);
                        let (r, g, b) =
                            color_viridis_float(lipo, LIPOPHILICITY_MIN, LIPOPHILICITY_MAX);
                        (r, g, b, opacity)
                    }
                    MeshColoring::Solid => unreachable!(),
                };

                Some(((r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8, a))
            } else {
                None
            }
        })
        .collect();

    engine_updates.meshes = true;

    println!("SAS mesh coloring done in {:?}", start.elapsed());

    Some(result)
}

/// Apply colors to a mesh, after they have been computed.
pub fn apply_mesh_colors(mesh: &mut Mesh, colors: &Option<MeshColors>) {
    if let Some(c) = colors {
        for (i, color) in c.iter().enumerate() {
            mesh.vertices[i].color = *color;
        }
    } else {
        // e.g. if solid coloring.
        for vertex in &mut mesh.vertices {
            vertex.color = None;
        }
    }
}
