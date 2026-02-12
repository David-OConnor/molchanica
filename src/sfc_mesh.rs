//! For calculating the accessible surface (AS), a proxy for accessibility of solvents
//! to a molecule. Used for drawing *surface*, *dots*, and related meshes. Related to the van der Waals
//! radius.
//!
//! Also used for other molecule-based meshes, like pockets for pharmacophores and docking.

use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    time::Instant,
};

use bincode::{Decode, Encode};
use graphics::{EngineUpdates, Mesh, Vertex};
use lin_alg::f32::{Vec3, Vec3 as Vec3F32};
use mcubes::{MarchingCubes, MeshSide};
use na_seq::Element::{self, Hydrogen};

use crate::{
    drawing::{CHARGE_MAP_MAX, CHARGE_MAP_MIN, SAS_ISO_OPACITY, color_viridis_float},
    molecules::common::MoleculeCommon,
};

const LIPOPHILICITY_MIN: f32 = -1.5;
const LIPOPHILICITY_MAX: f32 = 1.5;

pub type MeshColors = Vec<Option<(u8, u8, u8, u8)>>;

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

pub const SOLVENT_RAD: f32 = 1.4; // water probe

/// For  molecule-wrapping meshes. E.g. SAS around a protein, pockets etc.
#[derive(Clone, Copy, Debug, PartialEq, Default, Encode, Decode)]
pub enum MeshColoring {
    #[default]
    Solid, // todo: Wrap the color?
    Element,
    PartialCharge,
    /// aka greasiness
    Lipophilicity,
}

impl Display for MeshColoring {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Solid => "Solid",
            Self::Element => "Element",
            Self::PartialCharge => "Charge",
            Self::Lipophilicity => "Lipophilicity",
        };
        write!(f, "{v}")
    }
}

/// Create a mesh of the solvent-accessible surface. We do this using the ball-rolling method
/// based on Van-der-Waals radius, then use the Marching Cubes algorithm to generate an iso mesh with
/// iso value = 0.
///
/// Atoms is (posit, vdw radius).
pub fn make_sas_mesh(atoms: &[(Vec3, f32)], radius: f32, precision: f32) -> Mesh {
    if atoms.is_empty() {
        return Mesh::default();
    }

    // Bounding box and grid
    let mut bb_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut bb_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut r_max: f32 = 0.0;

    for (posit, vdw_radius) in atoms {
        let r = vdw_radius + radius;
        r_max = r_max.max(r);

        bb_min = Vec3::new(
            bb_min.x.min(posit.x),
            bb_min.y.min(posit.y),
            bb_min.z.min(posit.z),
        );

        bb_max = Vec3::new(
            bb_max.x.max(posit.x),
            bb_max.y.max(posit.y),
            bb_max.z.max(posit.z),
        );
    }
    bb_min -= Vec3::splat(r_max + precision);
    bb_max += Vec3::splat(r_max + precision);

    let dim_v = (bb_max - bb_min) / precision;
    let grid_dim = (
        dim_v.x.ceil() as usize + 1,
        dim_v.y.ceil() as usize + 1,
        dim_v.z.ceil() as usize + 1,
    );

    let n_voxels = grid_dim.0 * grid_dim.1 * grid_dim.2;

    // This can be any that is guaranteed to be well outside the SAS surface.
    // It prevents holes from appearing in the mesh due to not having a value outside to compare to.
    let far_val = (r_max + precision).powi(2) + 1.0;
    let mut field = vec![far_val; n_voxels];

    // Helper to flatten (x, y, z)
    let idx = |x: usize, y: usize, z: usize| -> usize { (z * grid_dim.1 + y) * grid_dim.0 + x };

    // Fill signed-squared-distance field
    for (center, vdw_radius) in atoms {
        let rad = *vdw_radius + radius;
        let rad2 = rad * rad;

        let lo = ((*center - Vec3::splat(rad)) - bb_min) / precision;
        let hi = ((*center + Vec3::splat(rad)) - bb_min) / precision;

        let (xi0, yi0, zi0) = (
            lo.x.floor().max(0.0) as usize,
            lo.y.floor().max(0.0) as usize,
            lo.z.floor().max(0.0) as usize,
        );
        let (xi1, yi1, zi1) = (
            hi.x.ceil().min((grid_dim.0 - 1) as f32) as usize,
            hi.y.ceil().min((grid_dim.1 - 1) as f32) as usize,
            hi.z.ceil().min((grid_dim.2 - 1) as f32) as usize,
        );

        for z in zi0..=zi1 {
            for y in yi0..=yi1 {
                for x in xi0..=xi1 {
                    let p = bb_min + Vec3::new(x as f32, y as f32, z as f32) * precision;
                    let d2 = (p - *center).magnitude_squared();
                    let v = d2 - rad2;
                    let f = &mut field[idx(x, y, z)];
                    if v < *f {
                        *f = v;
                    }
                }
            }
        }
    }

    // Convert to a mesh using Marchine Cubes.
    let sampling_interval = (
        grid_dim.0 as f32 - 1.0,
        grid_dim.1 as f32 - 1.0,
        grid_dim.2 as f32 - 1.0,
    );

    //  scale = precision because size / sampling_interval = precision
    let size = (
        sampling_interval.0 * precision,
        sampling_interval.1 * precision,
        sampling_interval.2 * precision,
    );

    // todo: The holes in our mesh seem related to the iso level chosen.
    let mc = MarchingCubes::new(grid_dim, size, sampling_interval, bb_min, field, 0.)
        .expect("marching cubes init");

    // Note: We're experiencing the opposite behavior than we expect here; we really want to draw outside.
    let mc_mesh = mc.generate(MeshSide::InsideOnly);

    let vertices: Vec<Vertex> = mc_mesh
        .vertices
        .iter()
        // I'm not sure why we need to invert the normal here; same reason we use InsideOnly above.
        .map(|v| Vertex::new([v.posit.x, v.posit.y, v.posit.z], -v.normal))
        .collect();

    Mesh {
        vertices,
        indices: mc_mesh.indices,
        material: 0,
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
pub fn update_mesh_coloring(
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
