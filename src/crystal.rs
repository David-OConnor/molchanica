//! Experiments with applying our visualization, molecule creation, and MD onto [initially non-organic] crystals. This
//! is a departure from existing code which focuses on classes of organic molecules: Small molecules, proteins, lipids,
//! nucleic acid.
//!
//! To start, we begin with Carbon: Graphite, Diamond etc.

use std::collections::{HashMap, HashSet};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::Element::{self, Carbon, Chlorine, Sodium};

use crate::molecules::{Atom, Bond, common::MoleculeCommon};

const SQRT_3_DIV_2: f64 = 0.866_025_403_784_438_6;
const LATTICE_DET_EPS: f64 = 1.0e-12;
const BOUNDS_EPS: f64 = 1.0e-9;
const CRYSTAL_BOND_RADIUS_SCALE: f64 = 1.18;
const CRYSTAL_BOND_MIN_DIST: f64 = 1.0e-6;

// Approximate room-temperature lattice constants, in Angstrom.
const GRAPHITE_LATTICE_A: f64 = 2.461;
const GRAPHITE_LATTICE_C: f64 = 6.708;

const DIAMOND_LATTICE_A: f64 = 3.567;

const SODIUM_CHLORIDE_LATTICE_A: f64 = 5.640;

/// An atom-like basis site in a periodically tiled crystal cell.
pub struct AtomInCrystal {
    /// Chemical identity of this crystal site.
    pub element: Element,
    /// Fractional coordinates in the unit cell, not Angstrom-space Cartesian
    /// coordinates. A position of `(0.5, 0.0, 0.0)` means halfway along the
    /// first lattice vector and on the lower faces of the second and third
    /// lattice vectors.
    pub posit: Vec3,
    /// Indices of atoms bonded inside this same stored cell. This cannot
    /// represent bonds to periodic images, so it should stay empty until we add
    /// cell-offset-aware bonds.
    pub adjacent: Vec<usize>,
    /// True when the canonical fractional position lies on a cell boundary.
    /// This is useful for drawing/debugging, but is not enough to derive a
    /// sharing multiplicity because corners, edges, and faces contribute
    /// different fractions to a conventional cell.
    pub shared_with_neighbor: bool,
}

impl AtomInCrystal {
    pub fn new(
        element: Element,
        posit: Vec3,
        adjacent: Vec<usize>,
        shared_with_neighbor: bool,
    ) -> Self {
        Self {
            element,
            posit,
            adjacent,
            shared_with_neighbor,
        }
    }
}

/// A generic crystal cell which can be periodically tiled to arbitrary
/// size.
pub struct CrystalCell {
    /// Cartesian lattice vectors in Å. Fractional atom coordinates are
    /// converted to Cartesian positions by `x * a + y * b + z * c`.
    pub lattice_vectors: [Vec3; 3],
    /// Canonical basis sites for one unit cell. These use fractional cell
    /// coordinates in a half-open convention: we keep representatives at 0.0 but
    /// avoid duplicating equivalent sites at 1.0.
    pub atoms: Vec<AtomInCrystal>,
}

impl CrystalCell {
    pub fn new_graphite() -> Self {
        // Bernal graphite in fractional coordinates of the conventional
        // hexagonal cell. The first two atoms are one graphene layer; the second
        // pair is the AB-stacked layer halfway along c.
        Self::from_fractional_basis(
            hexagonal_lattice(GRAPHITE_LATTICE_A, GRAPHITE_LATTICE_C),
            vec![
                (Carbon, 0.0, 0.0, 0.0),
                (Carbon, 1.0 / 3.0, 2.0 / 3.0, 0.0),
                (Carbon, 0.0, 0.0, 0.5),
                (Carbon, 2.0 / 3.0, 1.0 / 3.0, 0.5),
            ],
        )
    }

    pub fn new_diamond() -> Self {
        // Diamond cubic as an fcc carbon lattice plus a (1/4, 1/4, 1/4) basis.
        Self::from_fractional_basis(
            cubic_lattice(DIAMOND_LATTICE_A),
            vec![
                (Carbon, 0.0, 0.0, 0.0),
                (Carbon, 0.0, 0.5, 0.5),
                (Carbon, 0.5, 0.0, 0.5),
                (Carbon, 0.5, 0.5, 0.0),
                (Carbon, 0.25, 0.25, 0.25),
                (Carbon, 0.25, 0.75, 0.75),
                (Carbon, 0.75, 0.25, 0.75),
                (Carbon, 0.75, 0.75, 0.25),
            ],
        )
    }

    pub fn new_sodium_chloride() -> Self {
        // Rock-salt NaCl: chloride is an fcc lattice, sodium occupies the
        // octahedral holes. This stores the canonical periodic basis rather than
        // all conventional-cell boundary images.
        Self::from_fractional_basis(
            cubic_lattice(SODIUM_CHLORIDE_LATTICE_A),
            vec![
                (Chlorine, 0.0, 0.0, 0.0),
                (Chlorine, 0.0, 0.5, 0.5),
                (Chlorine, 0.5, 0.0, 0.5),
                (Chlorine, 0.5, 0.5, 0.0),
                (Sodium, 0.5, 0.0, 0.0),
                (Sodium, 0.0, 0.5, 0.0),
                (Sodium, 0.0, 0.0, 0.5),
                (Sodium, 0.5, 0.5, 0.5),
            ],
        )
    }

    pub fn fractional_to_cartesian(&self, posit: &Vec3) -> Vec3 {
        Vec3::new(
            posit.x * self.lattice_vectors[0].x
                + posit.y * self.lattice_vectors[1].x
                + posit.z * self.lattice_vectors[2].x,
            posit.x * self.lattice_vectors[0].y
                + posit.y * self.lattice_vectors[1].y
                + posit.z * self.lattice_vectors[2].y,
            posit.x * self.lattice_vectors[0].z
                + posit.y * self.lattice_vectors[1].z
                + posit.z * self.lattice_vectors[2].z,
        )
    }

    fn from_fractional_basis(
        lattice_vectors: [Vec3; 3],
        sites: Vec<(Element, f64, f64, f64)>,
    ) -> Self {
        let atoms = sites
            .into_iter()
            .map(|(element, x, y, z)| {
                AtomInCrystal::new(
                    element,
                    Vec3::new(x, y, z),
                    Vec::new(),
                    is_boundary_fractional(x, y, z),
                )
            })
            .collect();

        Self {
            lattice_vectors,
            atoms,
        }
    }

    /// Create a molecule which we can use elsewhere in the application.
    /// Attempts to create a volume roughly spanning between bounds_low and bounds_high.
    /// todo: how should we specify size? Num cells? volume? a bounding box?
    pub fn to_mol(&self, bounds_low: Vec3, bounds_high: Vec3) -> MoleculeCommon {
        let bounds = Bounds::new(bounds_low, bounds_high);
        let Some(inv_lattice) = inverse_lattice(self.lattice_vectors) else {
            return MoleculeCommon::new(
                "Crystal".to_string(),
                Vec::new(),
                Vec::new(),
                HashMap::new(),
                None,
            );
        };

        let (i_range, j_range, k_range) = cell_index_ranges(&bounds, inv_lattice);
        let mut atoms = Vec::new();
        let mut generated_sites: HashMap<(i32, i32, i32, usize), usize> = HashMap::new();

        for i in i_range.0..=i_range.1 {
            for j in j_range.0..=j_range.1 {
                for k in k_range.0..=k_range.1 {
                    let cell_offset = Vec3::new(i as f64, j as f64, k as f64);

                    for (basis_i, site) in self.atoms.iter().enumerate() {
                        let fractional_posit = site.posit + cell_offset;
                        let posit = self.fractional_to_cartesian(&fractional_posit);

                        if !bounds.contains(posit) {
                            continue;
                        }

                        let atom_i = atoms.len();
                        let serial_number = atom_i as u32 + 1;
                        atoms.push(Atom {
                            serial_number,
                            posit,
                            element: site.element,
                            type_in_res_general: Some(site.element.to_letter()),
                            hetero: true,
                            ..Default::default()
                        });
                        generated_sites.insert((i, j, k, basis_i), atom_i);
                    }
                }
            }
        }

        let mut bonds = Vec::new();
        let mut bond_pairs = HashSet::new();
        for i in i_range.0..=i_range.1 {
            for j in j_range.0..=j_range.1 {
                for k in k_range.0..=k_range.1 {
                    for (basis_i, site) in self.atoms.iter().enumerate() {
                        let Some(&atom_0) = generated_sites.get(&(i, j, k, basis_i)) else {
                            continue;
                        };

                        for &basis_j in &site.adjacent {
                            if basis_j <= basis_i {
                                continue;
                            }
                            let Some(&atom_1) = generated_sites.get(&(i, j, k, basis_j)) else {
                                continue;
                            };

                            add_crystal_bond(&mut bonds, &mut bond_pairs, &atoms, atom_0, atom_1);
                        }
                    }
                }
            }
        }

        infer_crystal_bonds(&atoms, &mut bonds, &mut bond_pairs);

        let mut metadata = HashMap::new();

        metadata.insert("source".to_string(), "CrystalCell::to_mol".to_string());
        metadata.insert("bounds_low".to_string(), vec3_to_metadata(bounds.low));
        metadata.insert("bounds_high".to_string(), vec3_to_metadata(bounds.high));
        metadata.insert("inferred_bonds".to_string(), bonds.len().to_string());

        let mut mol = MoleculeCommon::new("Crystal".to_string(), atoms, bonds, metadata, None);
        mol.update_next_sn();
        mol
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    low: Vec3,
    high: Vec3,
}

impl Bounds {
    fn new(a: Vec3, b: Vec3) -> Self {
        Self {
            low: Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
            high: Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
        }
    }

    fn contains(&self, posit: Vec3) -> bool {
        posit.x >= self.low.x - BOUNDS_EPS
            && posit.x <= self.high.x + BOUNDS_EPS
            && posit.y >= self.low.y - BOUNDS_EPS
            && posit.y <= self.high.y + BOUNDS_EPS
            && posit.z >= self.low.z - BOUNDS_EPS
            && posit.z <= self.high.z + BOUNDS_EPS
    }

    fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.low.x, self.low.y, self.low.z),
            Vec3::new(self.low.x, self.low.y, self.high.z),
            Vec3::new(self.low.x, self.high.y, self.low.z),
            Vec3::new(self.low.x, self.high.y, self.high.z),
            Vec3::new(self.high.x, self.low.y, self.low.z),
            Vec3::new(self.high.x, self.low.y, self.high.z),
            Vec3::new(self.high.x, self.high.y, self.low.z),
            Vec3::new(self.high.x, self.high.y, self.high.z),
        ]
    }
}

fn cell_index_ranges(
    bounds: &Bounds,
    inv_lattice: [[f64; 3]; 3],
) -> ((i32, i32), (i32, i32), (i32, i32)) {
    let mut min_frac = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max_frac = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

    for corner in bounds.corners() {
        let fractional = mat3_mul_vec(inv_lattice, corner);
        min_frac.x = min_frac.x.min(fractional.x);
        min_frac.y = min_frac.y.min(fractional.y);
        min_frac.z = min_frac.z.min(fractional.z);
        max_frac.x = max_frac.x.max(fractional.x);
        max_frac.y = max_frac.y.max(fractional.y);
        max_frac.z = max_frac.z.max(fractional.z);
    }

    (
        padded_index_range(min_frac.x, max_frac.x),
        padded_index_range(min_frac.y, max_frac.y),
        padded_index_range(min_frac.z, max_frac.z),
    )
}

fn padded_index_range(min: f64, max: f64) -> (i32, i32) {
    ((min.floor() as i32) - 1, (max.ceil() as i32) + 1)
}

fn inverse_lattice(lattice_vectors: [Vec3; 3]) -> Option<[[f64; 3]; 3]> {
    let a = lattice_vectors[0];
    let b = lattice_vectors[1];
    let c = lattice_vectors[2];

    let m00 = a.x;
    let m01 = b.x;
    let m02 = c.x;
    let m10 = a.y;
    let m11 = b.y;
    let m12 = c.y;
    let m20 = a.z;
    let m21 = b.z;
    let m22 = c.z;

    let det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);

    if det.abs() < LATTICE_DET_EPS {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        [
            (m11 * m22 - m12 * m21) * inv_det,
            (m02 * m21 - m01 * m22) * inv_det,
            (m01 * m12 - m02 * m11) * inv_det,
        ],
        [
            (m12 * m20 - m10 * m22) * inv_det,
            (m00 * m22 - m02 * m20) * inv_det,
            (m02 * m10 - m00 * m12) * inv_det,
        ],
        [
            (m10 * m21 - m11 * m20) * inv_det,
            (m01 * m20 - m00 * m21) * inv_det,
            (m00 * m11 - m01 * m10) * inv_det,
        ],
    ])
}

fn mat3_mul_vec(m: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

fn infer_crystal_bonds(
    atoms: &[Atom],
    bonds: &mut Vec<Bond>,
    bond_pairs: &mut HashSet<(usize, usize)>,
) {
    for atom_0 in 0..atoms.len() {
        for atom_1 in (atom_0 + 1)..atoms.len() {
            if crystal_atoms_bonded(&atoms[atom_0], &atoms[atom_1]) {
                add_crystal_bond(bonds, bond_pairs, atoms, atom_0, atom_1);
            }
        }
    }
}

fn add_crystal_bond(
    bonds: &mut Vec<Bond>,
    bond_pairs: &mut HashSet<(usize, usize)>,
    atoms: &[Atom],
    atom_0: usize,
    atom_1: usize,
) {
    if atom_0 == atom_1 {
        return;
    }

    let pair = if atom_0 < atom_1 {
        (atom_0, atom_1)
    } else {
        (atom_1, atom_0)
    };

    if !bond_pairs.insert(pair) {
        return;
    }

    bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0_sn: atoms[pair.0].serial_number,
        atom_1_sn: atoms[pair.1].serial_number,
        atom_0: pair.0,
        atom_1: pair.1,
        is_backbone: false,
    });
}

fn crystal_atoms_bonded(atom_0: &Atom, atom_1: &Atom) -> bool {
    let dist = (atom_0.posit - atom_1.posit).magnitude();

    dist > CRYSTAL_BOND_MIN_DIST && dist <= crystal_bond_cutoff(atom_0, atom_1)
}

fn crystal_bond_cutoff(atom_0: &Atom, atom_1: &Atom) -> f64 {
    (atom_0.element.covalent_radius() + atom_1.element.covalent_radius())
        * CRYSTAL_BOND_RADIUS_SCALE
}

fn vec3_to_metadata(v: Vec3) -> String {
    format!("{:.6},{:.6},{:.6}", v.x, v.y, v.z)
}

fn cubic_lattice(a: f64) -> [Vec3; 3] {
    [
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, 0.0, a),
    ]
}

fn hexagonal_lattice(a: f64, c: f64) -> [Vec3; 3] {
    [
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(-0.5 * a, SQRT_3_DIV_2 * a, 0.0),
        Vec3::new(0.0, 0.0, c),
    ]
}

fn is_boundary_fractional(x: f64, y: f64, z: f64) -> bool {
    x == 0.0 || y == 0.0 || z == 0.0 || x == 1.0 || y == 1.0 || z == 1.0
}
