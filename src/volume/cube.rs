use std::{fs, io, path::Path};

use bincode::{Decode, Encode};
use lin_alg::f32::Vec3;

const BOHR_TO_ANGSTROM: f32 = 0.529_177_2;

#[derive(Clone, Copy, Debug, Decode, Encode, PartialEq, Eq)]
pub(crate) enum Kind {
    Density,
    Orbital,
    Spin,
    Potential,
    Scalar,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Density => "Electron density",
            Self::Orbital => "Molecular orbital",
            Self::Spin => "Spin density",
            Self::Potential => "Electrostatic potential",
            Self::Scalar => "Scalar field",
        }
    }
}

#[derive(Clone, Decode, Encode)]
pub(crate) struct Atom {
    pub number: u8,
    pub position: Vec3,
}

#[derive(Clone, Decode, Encode)]
pub(crate) struct Dataset {
    pub label: String,
    /// X varies fastest, as required by marching cubes.
    pub values: Vec<f32>,
}

#[derive(Clone, Decode, Encode)]
pub(crate) struct Cube {
    pub kind: Kind,
    pub dims: [usize; 3],
    pub origin: Vec3,
    pub axes: [Vec3; 3],
    pub atoms: Vec<Atom>,
    pub datasets: Vec<Dataset>,
}

pub(crate) fn is_cube(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("cube") || ext.eq_ignore_ascii_case("cub"))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn number<T: std::str::FromStr>(token: Option<&str>, field: &str) -> io::Result<T> {
    token
        .and_then(|word| word.parse().ok())
        .ok_or_else(|| invalid(format!("Invalid Cube {field}")))
}

fn scalar(token: Option<&str>, field: &str) -> io::Result<f32> {
    let word = token.ok_or_else(|| invalid(format!("Missing Cube {field}")))?;
    let parsed = word
        .parse::<f32>()
        .or_else(|_| word.replace(['d', 'D'], "E").parse());
    parsed
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(format!("Invalid Cube {field}")))
}

fn point<'a>(words: &mut impl Iterator<Item = &'a str>, field: &str) -> io::Result<Vec3> {
    Ok(Vec3::new(
        scalar(words.next(), field)?,
        scalar(words.next(), field)?,
        scalar(words.next(), field)?,
    ))
}

fn product(dims: [usize; 3]) -> io::Result<usize> {
    dims.into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .filter(|_| dims.iter().all(|&size| size >= 2))
        .ok_or_else(|| invalid("Cube grid must have at least two points on every axis"))
}

fn classify(path: &Path, comments: &str, orbital_ids: bool) -> Kind {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let comments = comments.to_ascii_lowercase();

    if comments.contains("electrostatic potential")
        || comments.contains("property: esp")
        || name == "esp"
        || name.ends_with(".esp")
    {
        Kind::Potential
    } else if comments.contains("spin density") || name == "ds" || name.contains("spin-density") {
        Kind::Spin
    } else if orbital_ids
        || comments.contains("orbital")
        || name.contains("orbital")
        || name.starts_with("psi_a_")
        || name.starts_with("psi_b_")
        || name.starts_with("alpha_")
        || name.starts_with("beta_")
    {
        Kind::Orbital
    } else if comments.contains("density") || matches!(name.as_str(), "da" | "db" | "dt") {
        Kind::Density
    } else {
        Kind::Scalar
    }
}

impl Cube {
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::parse(&fs::read_to_string(path)?, path)
    }

    fn parse(text: &str, path: &Path) -> io::Result<Self> {
        let mut lines = text.lines();
        let title = lines.next().ok_or_else(|| invalid("Missing Cube title"))?;
        let comment = lines
            .next()
            .ok_or_else(|| invalid("Missing Cube comment"))?;
        let mut header = lines.next().unwrap_or_default().split_whitespace();
        let signed_atoms: i32 = number(header.next(), "atom count")?;
        let mut origin = point(&mut header, "origin")?;
        let values_per_point = header
            .next()
            .map_or(Ok(1), |word| number(Some(word), "dataset count"))?;

        let mut axes = [Vec3::new_zero(); 3];
        let mut counts = [0_i32; 3];
        for (count, axis) in counts.iter_mut().zip(&mut axes) {
            let mut row = lines.next().unwrap_or_default().split_whitespace();
            *count = number(row.next(), "grid dimension")?;
            *axis = point(&mut row, "grid axis")?;
        }
        let dims = counts.map(|size| size.unsigned_abs() as usize);
        let voxels = product(dims)?;
        let angstroms = counts.iter().all(|&size| size < 0);
        if !angstroms && counts.iter().any(|&size| size < 0) {
            return Err(invalid("Cube axes use inconsistent distance units"));
        }
        let scale = if angstroms { 1. } else { BOHR_TO_ANGSTROM };
        origin *= scale;
        axes = axes.map(|axis| axis * scale);

        let mut atoms = Vec::new();
        for _ in 0..signed_atoms.unsigned_abs() {
            let mut row = lines.next().unwrap_or_default().split_whitespace();
            let number = number(row.next(), "atomic number")?;
            let _ = scalar(row.next(), "nuclear charge")?;
            let position = point(&mut row, "atom position")? * scale;
            atoms.push(Atom { number, position });
        }

        let mut words = lines.flat_map(str::split_whitespace);
        let dataset_count = if signed_atoms < 0 {
            number(words.next(), "orbital dataset count")?
        } else {
            values_per_point
        };
        let count = voxels
            .checked_mul(dataset_count)
            .filter(|&n| n > 0 && n <= text.len() / 2 + 1)
            .ok_or_else(|| invalid("Cube dataset sizes do not match the file"))?;

        let kind = classify(path, &format!("{title} {comment}"), signed_atoms < 0);
        let mut datasets = Vec::new();
        for i in 0..dataset_count {
            let label = if signed_atoms < 0 {
                let id: i32 = number(words.next(), "orbital ID")?;
                format!("Orbital {id}")
            } else if dataset_count == 1 {
                kind.label().to_owned()
            } else {
                format!("Dataset {}", i + 1)
            };
            datasets.push(Dataset {
                label,
                values: vec![0.; voxels],
            });
        }

        let [nx, ny, nz] = dims;
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let i = x + nx * (y + ny * z);
                    for dataset in &mut datasets {
                        dataset.values[i] = scalar(words.next(), "grid value")?;
                    }
                }
            }
        }
        if words.next().is_some() {
            return Err(invalid(format!("Extra data after {count} Cube values")));
        }

        let cube = Self {
            kind,
            dims,
            origin,
            axes,
            atoms,
            datasets,
        };
        cube.validate()?;

        Ok(cube)
    }

    pub fn validate(&self) -> io::Result<()> {
        let voxels = product(self.dims)?;
        let [a, b, c] = self.axes;
        let det = a.dot(b.cross(c));
        if !det.is_finite()
            || det.abs() < 1.0e-12
            || !finite(self.origin)
            || self.atoms.iter().any(|atom| !finite(atom.position))
            || self.datasets.is_empty()
            || self.datasets.iter().any(|dataset| {
                dataset.values.len() != voxels || dataset.values.iter().any(|v| !v.is_finite())
            })
        {
            return Err(invalid("Invalid Cube axes, positions, or dataset values"));
        }

        Ok(())
    }

    pub fn matches_grid(&self, other: &Self) -> bool {
        const TOLERANCE: f32 = 1.0e-4;
        self.dims == other.dims
            && (self.origin - other.origin).magnitude() < TOLERANCE
            && self
                .axes
                .iter()
                .zip(other.axes)
                .all(|(a, b)| (*a - b).magnitude() < TOLERANCE)
            && self.atoms.len() == other.atoms.len()
            && self.atoms.iter().zip(&other.atoms).all(|(a, b)| {
                a.number == b.number && (a.position - b.position).magnitude() < TOLERANCE
            })
    }
}

fn finite(p: Vec3) -> bool {
    p.to_arr().into_iter().all(f32::is_finite)
}
