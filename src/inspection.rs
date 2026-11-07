//! Live annotations, scalar colors, and scene appearance.

use std::collections::{BTreeMap, HashMap, HashSet};

use bincode::{Decode, Encode};
use graphics::{ControlScheme, EngineUpdates, Scene};
use lin_alg::f64::Vec3;
use mol_defs::molecules::{MolGenericRef, MolType, peptide::MoleculePeptide};

use crate::{
    drawing::{MoleculeView, representations::Representation},
    prefs::DepthMode,
    selection::{
        Selection, ViewSelLevel,
        expression::{self, NamedSelection},
    },
    state::{LabelSizes, ResColoring, State, Visibility},
};

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct AtomRef {
    pub object: String,
    pub serial: u32,
    pub trajectory: bool,
}

impl AtomRef {
    pub fn position(&self, state: &State) -> Option<Vec3> {
        if self.trajectory {
            let viewer = &state.volatile.md_local.viewer;
            let set = viewer.get_active_mol_set()?;
            let (saved_set, index) = self.object.strip_prefix("trajectory:")?.split_once(':')?;
            if saved_set.parse::<usize>().ok()? != viewer.mol_set_active? {
                return None;
            }
            let index: usize = index.parse().ok()?;
            let mol = &set.mols.get(index)?.mol;
            let index = mol
                .atoms
                .iter()
                .position(|a| a.serial_number == self.serial)?;

            return mol.atom_posits.get(index).copied();
        }

        let objects = expression::objects(state);
        let mut matches = objects
            .iter()
            .filter(|(_, _, m)| expression::object_key(m) == self.object);
        let (_, _, mol) = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        let common = mol.common();
        let index = common
            .atoms
            .iter()
            .position(|a| a.serial_number == self.serial)?;

        common.atom_posits.get(index).copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Encode, Decode)]
pub enum MeasureKind {
    #[default]
    Distance,
    Angle,
    Dihedral,
}

impl MeasureKind {
    pub fn count(self) -> usize {
        match self {
            Self::Distance => 2,
            Self::Angle => 3,
            Self::Dihedral => 4,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Distance => "Distance",
            Self::Angle => "Angle",
            Self::Dihedral => "Dihedral",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Encode, Decode)]
pub enum AngleUnit {
    #[default]
    Degrees,
    Radians,
    Turns,
}

impl AngleUnit {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Degrees => "Degrees",
            Self::Radians => "Radians",
            Self::Turns => "Turns",
        }
    }

    fn format(self, kind: MeasureKind, value: f64) -> String {
        if kind == MeasureKind::Distance {
            return format!("{value:.2} Å");
        }

        match self {
            Self::Degrees => format!("{value:.2}°"),
            Self::Radians => format!("{:.3} rad", value.to_radians()),
            Self::Turns => {
                let turns = value.rem_euclid(360.) / 360.;

                format!("{:.4} turn", if turns == 0. { 0. } else { turns })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Measurement {
    pub name: String,
    pub kind: MeasureKind,
    pub atoms: Vec<AtomRef>,
    pub visible: bool,
    pub color: [u8; 3],
}

pub fn measure(kind: MeasureKind, points: &[Vec3]) -> Option<f64> {
    if points.len() != kind.count()
        || points
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
    {
        return None;
    }
    if kind == MeasureKind::Distance {
        return Some((points[1] - points[0]).magnitude());
    }

    let normalize = |v: Vec3| (v.magnitude_squared() > 1e-18).then(|| v.to_normalized());

    if kind == MeasureKind::Angle {
        let a = normalize(points[0] - points[1])?;
        let b = normalize(points[2] - points[1])?;

        return Some(a.dot(b).clamp(-1., 1.).acos().to_degrees());
    }

    let axis = normalize(points[2] - points[1])?;
    let a = points[0] - points[1];
    let b = points[3] - points[2];
    let v = normalize(a - axis * a.dot(axis))?;
    let w = normalize(b - axis * b.dot(axis))?;

    Some(axis.cross(v).dot(w).atan2(v.dot(w)).to_degrees())
}

impl Measurement {
    pub fn points(&self, state: &State) -> Option<Vec<Vec3>> {
        if self
            .atoms
            .iter()
            .any(|a| a.trajectory != state.volatile.md_local.draw_md_mols)
        {
            return None;
        }

        self.atoms.iter().map(|a| a.position(state)).collect()
    }

    pub fn label(&self, state: &State) -> String {
        self.points(state)
            .and_then(|p| measure(self.kind, &p))
            .map(|v| state.to_save.inspection.angle_unit.format(self.kind, v))
            .unwrap_or_else(|| "Unavailable".into())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Encode, Decode)]
pub enum Palette {
    #[default]
    Viridis,
    BlueWhiteRed,
    Confidence,
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct ScalarColor {
    pub object: String,
    pub title: String,
    pub values: BTreeMap<u32, f32>,
    pub minimum: f32,
    pub maximum: f32,
    pub palette: Palette,
    pub missing: [f32; 3],
    pub enabled: bool,
}

impl ScalarColor {
    pub fn color(&self, serial: u32) -> (f32, f32, f32) {
        let Some(&value) = self.values.get(&serial) else {
            return self.missing.into();
        };

        self.color_value(value)
    }

    pub fn color_value(&self, value: f32) -> (f32, f32, f32) {
        if !value.is_finite() {
            return self.missing.into();
        }

        if self.palette == Palette::Confidence {
            return if value >= 90. {
                (0.0, 0.32, 0.84)
            } else if value >= 70. {
                (0.4, 0.8, 0.95)
            } else if value >= 50. {
                (1., 0.86, 0.12)
            } else {
                (1., 0.49, 0.08)
            };
        }

        let t = if self.maximum > self.minimum {
            ((value - self.minimum) / (self.maximum - self.minimum)).clamp(0., 1.)
        } else {
            0.5
        };
        if self.palette == Palette::Viridis {
            return crate::drawing::color_viridis_float(t, 0., 1.);
        }
        let mix = |a: [f32; 3], b: [f32; 3], t: f32| {
            (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            )
        };

        if t < 0.5 {
            mix([0.15, 0.3, 0.85], [1.; 3], t * 2.)
        } else {
            mix([1.; 3], [0.85, 0.15, 0.15], (t - 0.5) * 2.)
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inspection {
    pub measurements: Vec<Measurement>,
    pub colors: Vec<ScalarColor>,
    pub angle_unit: AngleUnit,
    pub labels: Vec<crate::annotations::SelectionLabel>,
    pub slab: Option<crate::annotations::Slab>,
    pub solid_colors: BTreeMap<String, BTreeMap<u32, [f32; 3]>>,
}

impl Encode for Inspection {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        u64::MAX.encode(encoder)?;
        2u8.encode(encoder)?;
        self.angle_unit.encode(encoder)?;
        self.measurements.encode(encoder)?;

        self.colors.encode(encoder)?;
        self.labels.encode(encoder)?;
        self.slab.encode(encoder)?;

        self.solid_colors.encode(encoder)
    }
}

impl<Context> Decode<Context> for Inspection {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let marker = u64::decode(decoder)?;
        if marker == u64::MAX {
            let version = u8::decode(decoder)?;
            if !(1..=2).contains(&version) {
                return Err(bincode::error::DecodeError::Other(
                    "Unsupported inspection version",
                ));
            }

            let mut result = Self {
                angle_unit: AngleUnit::decode(decoder)?,
                measurements: Vec::decode(decoder)?,
                colors: Vec::decode(decoder)?,
                ..Default::default()
            };
            if version >= 2 {
                result.labels = Vec::decode(decoder)?;
                result.slab = Option::decode(decoder)?;
                result.solid_colors = BTreeMap::decode(decoder)?;
            }

            return Ok(result);
        }

        // Legacy inspection data starts with the measurement count.
        let count = usize::try_from(marker)
            .map_err(|_| bincode::error::DecodeError::OutsideUsizeRange(marker))?;
        decoder.claim_container_read::<Measurement>(count)?;
        let mut measurements = Vec::with_capacity(count);
        for _ in 0..count {
            decoder.unclaim_bytes_read(std::mem::size_of::<Measurement>());
            measurements.push(Measurement::decode(decoder)?);
        }

        Ok(Self {
            measurements,
            colors: Vec::decode(decoder)?,
            angle_unit: AngleUnit::Degrees,
            ..Default::default()
        })
    }
}

bincode::impl_borrow_decode!(Inspection);

impl Inspection {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .labels
            .iter()
            .any(|l| l.atoms.is_empty() || l.offset.iter().any(|v| !v.is_finite()))
            || self.slab.as_ref().is_some_and(|s| {
                !s.anchor.x.is_finite()
                    || !s.anchor.y.is_finite()
                    || !s.anchor.z.is_finite()
                    || !s.offset.is_finite()
                    || !s.thickness.is_finite()
                    || !(0.1..=10000.).contains(&s.thickness)
            })
            || self
                .solid_colors
                .values()
                .flat_map(|m| m.values())
                .flatten()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("Invalid labels, slab, or atom colors".into());
        }

        if self
            .measurements
            .iter()
            .any(|m| m.atoms.len() != m.kind.count())
        {
            return Err("Invalid measurement atom count".into());
        }
        if self.colors.iter().any(|c| {
            !c.minimum.is_finite()
                || !c.maximum.is_finite()
                || c.missing
                    .iter()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || c.values.values().any(|v| !v.is_finite())
        }) {
            return Err("Invalid property colors".into());
        }

        Ok(())
    }
}

pub fn color_for<'a>(state: &'a State, mol: &MolGenericRef<'_>) -> Option<&'a ScalarColor> {
    let key = expression::object_key(mol);

    state
        .to_save
        .inspection
        .colors
        .iter()
        .rev()
        .find(|c| c.enabled && c.object == key)
}

pub fn solid_color(state: &State, mol: &MolGenericRef<'_>, serial: u32) -> Option<(f32, f32, f32)> {
    state
        .to_save
        .inspection
        .solid_colors
        .get(&expression::object_key(mol))?
        .get(&serial)
        .copied()
        .map(Into::into)
}

pub fn color_selection(state: &mut State, input: &str, color: [f32; 3]) -> Result<usize, String> {
    let selected = expression::evaluate(state, input)?;
    let mut updates = Vec::new();
    for s in selected {
        let Some(mol) = state.get_mol(s.mol_type, s.mol_i) else {
            continue;
        };
        let ids: Vec<_> = s
            .atoms
            .iter()
            .map(|&i| mol.common().atoms[i].serial_number)
            .collect();
        updates.push((expression::object_key(&mol), ids));
    }
    let count = updates.iter().map(|(_, ids)| ids.len()).sum();
    for (key, ids) in updates {
        let colors = state
            .to_save
            .inspection
            .solid_colors
            .entry(key)
            .or_default();
        colors.extend(ids.into_iter().map(|id| (id, color)));
    }
    state.ui.inspection.solid_colors = state.to_save.inspection.solid_colors.clone();
    state.volatile.flags.update_ss_mesh = true;
    state.volatile.flags.update_sas_coloring = true;
    state.to_save.save_flag = true;

    Ok(count)
}

pub fn residue_colors(state: &State, mol: &MoleculePeptide) -> Vec<Option<(f32, f32, f32)>> {
    let mut result = state.ui.aa_selection.residue_colors(mol);
    if let Some(colors) = color_for(state, &MolGenericRef::Peptide(mol)) {
        result.resize(mol.residues.len(), None);
        for (i, residue) in mol.residues.iter().enumerate() {
            let values: Vec<_> = residue
                .atoms
                .iter()
                .filter_map(|&a| colors.values.get(&mol.common.atoms[a].serial_number))
                .collect();
            let color = if values.is_empty() {
                colors.missing.into()
            } else {
                colors.color_value(values.iter().map(|&&v| v).sum::<f32>() / values.len() as f32)
            };
            if !state.ui.aa_selection.enabled {
                result[i] = Some(color);
            }
        }
    }

    result
}

pub fn ribbon_colors(state: &State, mol: &MoleculePeptide) -> Vec<Option<(f32, f32, f32)>> {
    let mut result = residue_colors(state, mol);
    if let Some(colors) = state
        .to_save
        .inspection
        .solid_colors
        .get(&expression::object_key(&MolGenericRef::Peptide(mol)))
    {
        result.resize(mol.residues.len(), None);
        for (i, residue) in mol.residues.iter().enumerate() {
            if let Some(color) = residue
                .atoms
                .iter()
                .filter_map(|&a| colors.get(&mol.common.atoms[a].serial_number))
                .next()
            {
                result[i] = Some((*color).into());
            }
        }
    }

    result
}

pub fn atom_colors(state: &State, mol: &MolGenericRef<'_>) -> Vec<Option<(f32, f32, f32)>> {
    let scalar = color_for(state, mol);
    if scalar.is_none()
        && !state
            .to_save
            .inspection
            .solid_colors
            .contains_key(&expression::object_key(mol))
    {
        return Vec::new();
    }

    mol.common()
        .atoms
        .iter()
        .map(|a| {
            solid_color(state, mol, a.serial_number)
                .or_else(|| scalar.map(|c| c.color(a.serial_number)))
        })
        .collect()
}

// CIF tokens retain quoted identifiers and semicolon text fields.
pub fn cif_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut field = None::<String>;
    for line in text.lines() {
        if line.starts_with(';') {
            if let Some(value) = field.take() {
                tokens.push(value);
            } else {
                field = Some(line[1..].to_owned());
            }
            continue;
        }
        if let Some(value) = &mut field {
            value.push('\n');
            value.push_str(line);
            continue;
        }
        let chars: Vec<_> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            if chars[i] == '#' {
                break;
            }
            let quote = matches!(chars[i], '\'' | '"').then_some(chars[i]);
            if quote.is_some() {
                i += 1;
            }
            let mut value = String::new();
            while i < chars.len() {
                if quote == Some(chars[i]) && (i + 1 == chars.len() || chars[i + 1].is_whitespace())
                {
                    i += 1;
                    break;
                }
                if quote.is_none() && chars[i].is_whitespace() {
                    break;
                }
                value.push(chars[i]);
                i += 1;
            }
            tokens.push(value);
        }
    }

    tokens
}

pub fn b_factors(text: &str) -> BTreeMap<u32, f32> {
    let tokens = cif_tokens(text);
    let mut values = BTreeMap::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] != "loop_" {
            i += 1;
            continue;
        }
        i += 1;
        let start = i;
        while i < tokens.len() && tokens[i].starts_with('_') {
            i += 1;
        }
        let headers = &tokens[start..i];
        let id = headers
            .iter()
            .position(|s| s.eq_ignore_ascii_case("_atom_site.id"));
        let b = headers
            .iter()
            .position(|s| s.eq_ignore_ascii_case("_atom_site.B_iso_or_equiv"));
        let model = headers
            .iter()
            .position(|s| s.eq_ignore_ascii_case("_atom_site.pdbx_PDB_model_num"));
        let mut first_model = None;
        while !headers.is_empty() && i + headers.len() <= tokens.len() {
            if tokens[i].starts_with('_')
                || tokens[i] == "loop_"
                || tokens[i] == "stop_"
                || tokens[i].starts_with("data_")
            {
                break;
            }
            let row = &tokens[i..i + headers.len()];
            let use_row =
                model.is_none_or(|m| row[m] == *first_model.get_or_insert_with(|| row[m].clone()));
            if use_row
                && let (Some(id), Some(b)) = (id, b)
                && let (Ok(sn), Ok(value)) = (row[id].parse(), row[b].parse::<f32>())
                && value.is_finite()
            {
                values.entry(sn).or_insert(value);
            }
            i += headers.len();
        }
    }

    values
}

pub fn import_values(mol: &MolGenericRef<'_>, text: &str) -> Result<BTreeMap<u32, f32>, String> {
    let mut lines = text
        .lines()
        .filter(|s| !s.trim().is_empty() && !s.trim_start().starts_with('#'));
    let header = lines
        .next()
        .ok_or("Empty values file")?
        .trim_start_matches('\u{feff}');
    let separator = if header.contains('\t') { '\t' } else { ',' };
    let split = |s: &str| {
        s.split(separator)
            .map(|v| v.trim().trim_matches('"').to_ascii_lowercase())
            .collect::<Vec<_>>()
    };
    let headers = split(header);
    if headers.iter().collect::<HashSet<_>>().len() != headers.len() {
        return Err("Duplicate column names".into());
    }
    let value_col = headers
        .iter()
        .position(|v| v == "value")
        .ok_or("Expected a value column")?;
    let atom_col = headers
        .iter()
        .position(|v| matches!(v.as_str(), "atom" | "serial" | "atom_serial"));
    let residue_col = headers
        .iter()
        .position(|v| matches!(v.as_str(), "residue" | "resi"));
    let chain_col = headers.iter().position(|v| v == "chain");
    if atom_col.is_none() && residue_col.is_none() {
        return Err("Expected atom_serial,value or chain,residue,value columns".into());
    }
    let peptide = match mol {
        MolGenericRef::Peptide(p) => Some(*p),
        _ => None,
    };
    let serials: HashSet<_> = mol.common().atoms.iter().map(|a| a.serial_number).collect();
    let mut residue_index = HashMap::<u32, Vec<usize>>::new();
    if let Some(peptide) = peptide {
        for (i, residue) in peptide.residues.iter().enumerate() {
            residue_index
                .entry(residue.serial_number)
                .or_default()
                .push(i);
        }
    }

    let mut result = BTreeMap::new();
    for (line_i, line) in lines.enumerate() {
        let fields: Vec<_> = line
            .split(separator)
            .map(|v| v.trim().trim_matches('"'))
            .collect();
        if fields.len() != headers.len() {
            return Err(format!("Row {} has the wrong column count", line_i + 2));
        }
        let value: f32 = fields[value_col]
            .parse()
            .map_err(|_| format!("Invalid value on row {}", line_i + 2))?;
        if !value.is_finite() {
            return Err(format!("Nonfinite value on row {}", line_i + 2));
        }
        let atoms: Vec<u32> = if let Some(col) = atom_col {
            let sn = fields[col]
                .parse::<u32>()
                .map_err(|_| "Invalid atom serial")?;
            if !serials.contains(&sn) {
                return Err(format!("Unknown atom serial {sn}"));
            }
            vec![sn]
        } else {
            let peptide = peptide.ok_or("Residue imports require a protein")?;
            let number: u32 = fields[residue_col.unwrap()]
                .parse()
                .map_err(|_| "Invalid residue number")?;
            let matches: Vec<_> = residue_index
                .get(&number)
                .into_iter()
                .flatten()
                .map(|&i| &peptide.residues[i])
                .filter(|r| {
                    r.serial_number == number
                        && chain_col.is_none_or(|c| {
                            r.atoms.iter().any(|&a| {
                                peptide.common.atoms[a]
                                    .chain
                                    .and_then(|i| peptide.chains.get(i))
                                    .is_some_and(|ch| ch.id == fields[c])
                            })
                        })
                })
                .collect();
            if matches.len() != 1 {
                return Err(format!(
                    "Residue {number} is missing or ambiguous; include chain"
                ));
            }
            matches[0].atom_sns.clone()
        };
        for sn in atoms {
            if result.insert(sn, value).is_some() {
                return Err(format!("Duplicate value for atom {sn}"));
            }
        }
    }
    if result.is_empty() {
        return Err("No values to import".into());
    }

    Ok(result)
}

pub(crate) fn add_colors(
    state: &mut State,
    title: String,
    values: BTreeMap<u32, f32>,
    confidence: bool,
) -> Result<(), String> {
    let mol = state.active_mol().ok_or("Select a molecule first")?;
    let serials: HashSet<_> = mol.common().atoms.iter().map(|a| a.serial_number).collect();
    let values: BTreeMap<_, _> = values
        .into_iter()
        .filter(|(sn, v)| v.is_finite() && serials.contains(sn))
        .collect();
    if values.is_empty() {
        return Err("No matching values in this molecule".into());
    }
    let object = expression::object_key(&mol);
    let minimum = if confidence {
        0.
    } else {
        values.values().copied().fold(f32::INFINITY, f32::min)
    };
    let maximum = if confidence {
        100.
    } else {
        values.values().copied().fold(f32::NEG_INFINITY, f32::max)
    };
    state
        .to_save
        .inspection
        .colors
        .retain(|c| c.object != object || c.title != title);
    for colors in &mut state.to_save.inspection.colors {
        if colors.object == object {
            colors.enabled = false;
        }
    }
    state.to_save.inspection.colors.push(ScalarColor {
        object,
        title,
        values,
        minimum,
        maximum,
        palette: if confidence {
            Palette::Confidence
        } else {
            Palette::Viridis
        },
        missing: [0.5; 3],
        enabled: true,
    });
    state.to_save.save_flag = true;
    state.ui.aa_selection.enabled = false;
    state.ui.atom_color_by_charge = false;

    Ok(())
}

pub fn capture_pick(state: &mut State) {
    if !state.ui.inspection.picking {
        return;
    }
    let reference = if state.volatile.md_local.draw_md_mols {
        state.ui.inspection.trajectory_pick.take()
    } else {
        let selected = expression::current_atoms(state);
        if selected.iter().map(|s| s.atoms.len()).sum::<usize>() != 1 {
            return;
        }
        let selected = &selected[0];
        state
            .get_mol(selected.mol_type, selected.mol_i)
            .map(|m| AtomRef {
                object: expression::object_key(&m),
                serial: m.common().atoms[selected.atoms[0]].serial_number,
                trajectory: false,
            })
    };
    let Some(reference) = reference else {
        return;
    };
    if state.ui.inspection.pending.contains(&reference) {
        return;
    }
    state.ui.inspection.pending.push(reference);
    if state.ui.inspection.pending.len() == state.ui.inspection.kind.count() {
        let atoms = std::mem::take(&mut state.ui.inspection.pending);
        let kind = state.ui.inspection.kind;
        state.to_save.inspection.measurements.push(Measurement {
            name: format!(
                "{} {}",
                kind.name(),
                state.to_save.inspection.measurements.len() + 1
            ),
            kind,
            atoms,
            visible: true,
            color: [255, 220, 80],
        });
        state.ui.inspection.picking = false;
        state.to_save.save_flag = true;
    }
}

pub fn pick_trajectory(state: &mut State, scene: &Scene, cursor: (f32, f32)) {
    let (origin, end) = scene.screen_to_render(cursor);
    let origin: Vec3 = origin.into();
    let end: Vec3 = end.into();
    let direction = end - origin;
    let direction = direction.to_normalized();
    let Some(set) = state.volatile.md_local.viewer.get_active_mol_set() else {
        return;
    };
    let mut hit = None;
    for (i, mol) in set.mols.iter().enumerate() {
        if !mol.mol.visible
            || !set
                .groups
                .iter()
                .any(|g| g.visible && g.ranges.contains(&mol.range))
        {
            continue;
        }
        for (a, p) in mol.mol.atom_posits.iter().enumerate() {
            let along = (*p - origin).dot(direction);
            let distance = (*p - origin - direction * along).magnitude();
            if along > 0. && distance < 0.65 && hit.as_ref().is_none_or(|(depth, _)| along < *depth)
            {
                hit = Some((
                    along,
                    AtomRef {
                        object: format!(
                            "trajectory:{}:{i}",
                            state.volatile.md_local.viewer.mol_set_active.unwrap()
                        ),
                        serial: mol.mol.atoms[a].serial_number,
                        trajectory: true,
                    },
                ));
            }
        }
    }
    state.ui.inspection.trajectory_pick = hit.map(|(_, a)| a);
    capture_pick(state);
}

pub fn pick_atom(state: &mut State, scene: &Scene, cursor: (f32, f32)) {
    if state.volatile.md_local.draw_md_mols {
        pick_trajectory(state, scene, cursor);
        return;
    }
    let (origin, end) = scene.screen_to_render(cursor);
    let origin: Vec3 = origin.into();
    let end: Vec3 = end.into();
    let direction = (end - origin).to_normalized();
    let mut hit = None;
    for (kind, index, mol) in expression::objects(state) {
        if !mol.common().visible {
            continue;
        }
        if matches!(kind, MolType::Peptide) && state.ui.visibility.hide_protein {
            continue;
        }
        if matches!(kind, MolType::Ligand) && state.ui.visibility.hide_ligand {
            continue;
        }
        for (a, atom) in mol.common().atoms.iter().enumerate() {
            if state.ui.visibility.hide_hydrogen && atom.element == na_seq::Element::Hydrogen {
                continue;
            }
            if let MolGenericRef::Peptide(p) = &mol {
                if atom
                    .chain
                    .and_then(|i| p.chains.get(i))
                    .is_some_and(|c| !c.visible)
                {
                    continue;
                }
                if state.ui.visibility.hide_water
                    && atom.role == Some(mol_defs::molecules::AtomRole::Water)
                {
                    continue;
                }
            }
            let Some(&p) = mol.common().atom_posits.get(a) else {
                continue;
            };
            let along = (p - origin).dot(direction);
            if along < 0. || !scene.camera.in_view(p.into()).0 {
                continue;
            }
            let distance = (p - origin - direction * along).magnitude();
            if distance < 0.65 && hit.as_ref().is_none_or(|(depth, _, _, _)| along < *depth) {
                hit = Some((along, kind, index, a));
            }
        }
    }
    if let Some((_, kind, index, atom)) = hit {
        state.volatile.active_mol = Some((kind, index));
        state.ui.selection = Selection::Atoms(vec![crate::selection::MoleculeAtoms {
            mol_type: kind,
            mol_i: index,
            atoms: vec![atom],
        }]);
        capture_pick(state);
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct SceneState {
    pub graphics: GraphicsState,
    pub aa_selection: crate::state::AaSelection,
    pub atom_mode: bool,
    pub filter: crate::state::DistFilter,
    pub nearby: u16,
    pub color_by_mol: bool,
    pub mesh_color: u8,
    pub active: Option<String>,
    pub orbit_selected: bool,
    pub background: (f32, f32, f32),
    pub ambient: ([f32; 4], f32),
    pub lights: Vec<LightState>,
    pub near: f32,
    pub fov: f32,
    pub depth: Option<(u16, u16)>,
    pub depth_auto: bool,
    pub orbit: Option<lin_alg::f32::Vec3>,
    pub visibility: Visibility,
    pub label_sizes: LabelSizes,
    pub view: MoleculeView,
    pub peptide_view: MoleculeView,
    pub residue_color: ResColoring,
    pub atom_charge: bool,
    pub layers: Vec<Representation>,
    pub selections: Vec<NamedSelection>,
    pub hidden_defaults: Vec<String>,
    pub inspection: Inspection,
    pub objects: Vec<(String, bool, Vec<bool>)>,
    pub selection: String,
    pub density_iso: f32,
    pub frame: Option<usize>,
    pub trajectory_set: Option<usize>,
    pub draw_trajectory: bool,
}

impl SceneState {
    pub fn validate(&self) -> Result<(), String> {
        self.inspection.validate()?;
        if !self.near.is_finite()
            || self.near <= 0.
            || !self.fov.is_finite()
            || !(0.01..3.13).contains(&self.fov)
            || !self.density_iso.is_finite()
        {
            return Err("Invalid scene camera or density level".into());
        }
        if self.layers.iter().any(|l| {
            !l.opacity.is_finite()
                || !(0.0..=1.0).contains(&l.opacity)
                || !l.size.is_finite()
                || !(0.1..=3.0).contains(&l.size)
                || l.color
                    .is_some_and(|c| c.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
        }) {
            return Err("Invalid scene representation".into());
        }
        if [
            self.label_sizes.mol,
            self.label_sizes.protein,
            self.label_sizes.chain,
            self.label_sizes.residue,
            self.label_sizes.atom_sn,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(LabelSizes::MIN..=LabelSizes::MAX).contains(v))
        {
            return Err("Invalid scene label sizes".into());
        }

        Ok(())
    }
    pub fn capture(state: &State, scene: &Scene) -> Self {
        Self {
            graphics: GraphicsState::capture(&state.to_save.graphics),
            aa_selection: state.ui.aa_selection.clone(),
            atom_mode: state.ui.view_sel_level == ViewSelLevel::AtomBond,
            filter: state.ui.dist_filter,
            nearby: state.ui.nearby_dist_thresh,
            color_by_mol: state.ui.color_by_mol,
            mesh_color: match state.ui.mesh_coloring {
                mol_defs::sfc_mesh::MeshColoring::Solid => 0,
                mol_defs::sfc_mesh::MeshColoring::Element => 1,
                mol_defs::sfc_mesh::MeshColoring::PartialCharge => 2,
                mol_defs::sfc_mesh::MeshColoring::Lipophilicity => 3,
            },
            active: state.active_mol().map(|m| expression::object_key(&m)),
            orbit_selected: state.ui.orbit_selected_atom,
            background: scene.background_color,
            ambient: (
                scene.lighting.ambient_color,
                scene.lighting.ambient_intensity,
            ),
            lights: scene
                .lighting
                .point_lights
                .iter()
                .map(LightState::capture)
                .collect(),
            near: scene.camera.near,
            fov: scene.camera.fov_y,
            depth: match state.to_save.depth_mode {
                DepthMode::Manual(v) => Some(v),
                _ => None,
            },
            depth_auto: state.to_save.depth_mode == DepthMode::Auto,
            orbit: match scene.input_settings.control_scheme {
                ControlScheme::Arc { center } => Some(center),
                _ => None,
            },
            visibility: state.ui.visibility.clone(),
            label_sizes: state.ui.label_sizes.clone(),
            view: state.ui.mol_view,
            peptide_view: state.ui.mol_view_peptide,
            residue_color: state.ui.res_coloring,
            atom_charge: state.ui.atom_color_by_charge,
            layers: state.to_save.representations.clone(),
            selections: state.to_save.named_selections.clone(),
            hidden_defaults: state.to_save.hidden_default_objects.clone(),
            inspection: state.to_save.inspection.clone(),
            objects: expression::objects(state)
                .iter()
                .map(|(_, _, m)| {
                    (
                        expression::object_key(m),
                        m.common().visible,
                        match m {
                            MolGenericRef::Peptide(p) => {
                                p.chains.iter().map(|c| c.visible).collect()
                            }
                            _ => Vec::new(),
                        },
                    )
                })
                .collect(),
            selection: expression::current_expression(state),
            density_iso: state.ui.density_iso_level,
            frame: state.volatile.md_local.viewer.current_snapshot,
            trajectory_set: state.volatile.md_local.viewer.mol_set_active,
            draw_trajectory: state.volatile.md_local.draw_md_mols,
        }
    }

    pub fn apply(&self, state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
        state.to_save.graphics = self.graphics.restore();
        state.graphics_settings = state.to_save.graphics.to_engine();
        updates.graphics_settings = Some(state.graphics_settings.clone());
        state.ui.aa_selection = self.aa_selection.clone();
        state.ui.view_sel_level = if self.atom_mode {
            ViewSelLevel::AtomBond
        } else {
            ViewSelLevel::Residue
        };
        state.ui.dist_filter = self.filter;
        state.ui.nearby_dist_thresh = self.nearby;
        state.ui.color_by_mol = self.color_by_mol;
        state.ui.mesh_coloring = match self.mesh_color {
            1 => mol_defs::sfc_mesh::MeshColoring::Element,
            2 => mol_defs::sfc_mesh::MeshColoring::PartialCharge,
            3 => mol_defs::sfc_mesh::MeshColoring::Lipophilicity,
            _ => mol_defs::sfc_mesh::MeshColoring::Solid,
        };
        state.ui.orbit_selected_atom = self.orbit_selected;
        state.volatile.active_mol = self.active.as_ref().and_then(|key| {
            expression::objects(state)
                .iter()
                .find(|(_, _, m)| expression::object_key(m) == *key)
                .map(|(kind, index, _)| (*kind, *index))
        });
        if let Some((MolType::Peptide, i)) = state.volatile.active_mol {
            state.volatile.active_peptide = Some(i);
        }
        scene.background_color = self.background;
        scene.lighting.ambient_color = self.ambient.0;
        scene.lighting.ambient_intensity = self.ambient.1;
        scene.lighting.point_lights = self.lights.iter().map(LightState::restore).collect();
        updates.lighting = true;
        state.ui.visibility = self.visibility.clone();
        state.ui.label_sizes = self.label_sizes.clone();
        state.ui.mol_view = self.view;
        state.ui.mol_view_peptide = self.peptide_view;
        state.ui.res_coloring = self.residue_color;
        state.ui.atom_color_by_charge = self.atom_charge;
        state.to_save.representations = self.layers.clone();
        state.to_save.named_selections = self.selections.clone();
        state.to_save.hidden_default_objects = self.hidden_defaults.clone();
        state.to_save.inspection = self.inspection.clone();
        state.ui.inspection.colors = self.inspection.colors.clone();
        state.ui.inspection.solid_colors = self.inspection.solid_colors.clone();
        state.ui.density_iso_level = self.density_iso;
        state.to_save.depth_mode = if let Some(v) = self.depth {
            DepthMode::Manual(v)
        } else if self.depth_auto {
            DepthMode::Auto
        } else {
            DepthMode::Disabled
        };
        scene.input_settings.control_scheme = self
            .orbit
            .map(|center| ControlScheme::Arc { center })
            .unwrap_or(ControlScheme::FreeCamera);
        let objects: Vec<_> = expression::objects(state)
            .iter()
            .map(|(t, i, m)| (*t, *i, expression::object_key(m)))
            .collect();
        for (kind, index, key) in objects {
            if let Some((_, visible, chains)) = self.objects.iter().find(|(k, _, _)| *k == key)
                && let Some(mut mol) = state.get_mol_mut(kind, index)
            {
                mol.common_mut().visible = *visible;
                if let mol_defs::molecules::MolGenericRefMut::Peptide(p) = mol {
                    for (chain, visible) in p.chains.iter_mut().zip(chains) {
                        chain.visible = *visible;
                    }
                }
            }
        }
        state.ui.selection = expression::evaluate(state, &self.selection)
            .map(Selection::Atoms)
            .unwrap_or(Selection::None);
        if let Some(frame) = self.frame
            && self
                .trajectory_set
                .is_some_and(|i| i < state.volatile.md_local.viewer.mol_sets.len())
            && frame < state.volatile.md_local.viewer.snapshots.len()
        {
            let viewer = &mut state.volatile.md_local.viewer;
            viewer.mol_set_active = self.trajectory_set;
            viewer.current_snapshot = Some(frame);
            viewer.slider_posit_ui = frame;
            viewer.playing = false;
            let _ = viewer.change_snapshot(frame);
        }
        state.volatile.md_local.draw_md_mols = self.draw_trajectory
            && state
                .volatile
                .md_local
                .viewer
                .get_active_mol_set()
                .is_some();
        state.volatile.flags.update_ss_mesh = true;
        state.volatile.flags.update_sas_coloring = true;
        state.volatile.flags.make_density_iso_mesh = true;
        scene.entities.retain(|e| {
            e.class != crate::drawing::EntityClass::DensityPoint as u32
                && e.class != crate::drawing::EntityClass::DensitySurface as u32
        });
        state.volatile.flags.new_density_loaded = true;
        state.to_save.save_flag = true;
        scene.camera.fov_y = self.fov;
        scene.camera.near = self.near;
        crate::cam::set_fog(state, &mut scene.camera);
        scene.camera.update_proj_mat();
        updates.camera = true;
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct GraphicsState {
    msaa: crate::state::MsaaSetting,
    ao: u8,
    radius: f32,
    edges: Option<f32>,
    halos: Option<f32>,
}

impl GraphicsState {
    fn capture(g: &crate::prefs::Graphics) -> Self {
        Self {
            msaa: g.msaa,
            ao: match g.ambient_occlusion {
                graphics::AmbientOcclusion::None => 0,
                graphics::AmbientOcclusion::Ssao => 1,
                graphics::AmbientOcclusion::Gtao => 2,
            },
            radius: g.ssao_radius,
            edges: g.edge_cueing,
            halos: g.depth_aware_halos,
        }
    }

    fn restore(&self) -> crate::prefs::Graphics {
        crate::prefs::Graphics {
            msaa: self.msaa,
            msaa_user_set: true,
            ambient_occlusion: match self.ao {
                1 => graphics::AmbientOcclusion::Ssao,
                2 => graphics::AmbientOcclusion::Gtao,
                _ => graphics::AmbientOcclusion::None,
            },
            ssao_radius: self.radius,
            edge_cueing: self.edges,
            depth_aware_halos: self.halos,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct LightState {
    kind: u8,
    direction: lin_alg::f32::Vec3,
    fov: f32,
    position: lin_alg::f32::Vec3,
    diffuse: [f32; 4],
    specular: [f32; 4],
    diffuse_intensity: f32,
    specular_intensity: f32,
}

impl LightState {
    fn capture(light: &graphics::PointLight) -> Self {
        let (kind, direction, fov) = match light.type_ {
            graphics::LightType::Omnidirectional => (0, lin_alg::f32::Vec3::new_zero(), 0.),
            graphics::LightType::Directional { direction, fov } => (1, direction, fov),
            graphics::LightType::Diffuse => (2, lin_alg::f32::Vec3::new_zero(), 0.),
        };

        Self {
            kind,
            direction,
            fov,
            position: light.position,
            diffuse: light.diffuse_color,
            specular: light.specular_color,
            diffuse_intensity: light.diffuse_intensity,
            specular_intensity: light.specular_intensity,
        }
    }

    fn restore(&self) -> graphics::PointLight {
        graphics::PointLight {
            type_: match self.kind {
                1 => graphics::LightType::Directional {
                    direction: self.direction,
                    fov: self.fov,
                },
                2 => graphics::LightType::Diffuse,
                _ => graphics::LightType::Omnidirectional,
            },
            position: self.position,
            diffuse_color: self.diffuse,
            specular_color: self.specular,
            diffuse_intensity: self.diffuse_intensity,
            specular_intensity: self.specular_intensity,
        }
    }
}
