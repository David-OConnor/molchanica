//! Selection labels and camera-aligned clipping slabs.

use std::collections::HashSet;

use bincode::{Decode, Encode};
use graphics::{Camera, FWD_VEC};
use lin_alg::f64::Vec3;
use mol_defs::molecules::MolGenericRef;

use crate::{inspection::AtomRef, selection::expression, state::State};

#[derive(Clone, Copy, Debug, Default, PartialEq, Encode, Decode)]
pub enum LabelFormat {
    #[default]
    Residue,
    ChainResidue,
    AtomName,
}

impl LabelFormat {
    pub fn title(self) -> &'static str {
        match self {
            Self::Residue => "ASP 123",
            Self::ChainResidue => "A:ASP123",
            Self::AtomName => "Atom names",
        }
    }

    pub fn parse(input: &str) -> Result<Self, String> {
        match input.trim().to_ascii_lowercase().as_str() {
            "residue" | "resn resi" | "asp 123" => Ok(Self::Residue),
            "chain" | "chain_residue" | "a:asp123" => Ok(Self::ChainResidue),
            "name" | "atom" | "atom_name" => Ok(Self::AtomName),
            _ => Err("Label formats: residue, chain_residue, name; use off to remove".into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct SelectionLabel {
    pub atoms: Vec<AtomRef>,
    pub text: String,
    pub offset: [f32; 2],
    pub visible: bool,
}

impl SelectionLabel {
    pub fn position(&self, state: &State) -> Option<Vec3> {
        if state.volatile.md_local.draw_md_mols || self.atoms.is_empty() {
            return None;
        }

        let points: Option<Vec<_>> = self.atoms.iter().map(|a| a.position(state)).collect();
        let points = points?;

        Some(points.iter().copied().fold(Vec3::new_zero(), |a, b| a + b) / points.len() as f64)
    }
}

#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct Slab {
    pub enabled: bool,
    pub anchor: Vec3,
    pub offset: f32,
    pub thickness: f32,
}

impl Slab {
    pub fn planes(&self, camera: &Camera) -> (f32, f32) {
        let anchor: lin_alg::f32::Vec3 = self.anchor.into();
        let center =
            (anchor - camera.position).dot(camera.orientation.rotate_vec(FWD_VEC)) + self.offset;
        let near = (center - self.thickness / 2.).max(0.01);
        let far = (center + self.thickness / 2.).max(near + 0.01);

        (near, far)
    }
}

pub fn center_slab(state: &mut State, input: &str, thickness: f32) -> Result<(), String> {
    if !thickness.is_finite() || !(0.1..=10000.).contains(&thickness) {
        return Err("Slab thickness must be between 0.1 and 10000 Å".into());
    }

    let selected = expression::evaluate(state, input)?;
    let points: Vec<_> = selected
        .iter()
        .flat_map(|s| {
            state
                .get_mol(s.mol_type, s.mol_i)
                .map(|m| {
                    s.atoms
                        .iter()
                        .map(|&a| {
                            m.common()
                                .atom_posits
                                .get(a)
                                .copied()
                                .unwrap_or(m.common().atoms[a].posit)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();
    if points.is_empty() {
        return Err("Selection contains no atoms".into());
    }

    let anchor = points.iter().copied().fold(Vec3::new_zero(), |a, b| a + b) / points.len() as f64;
    state.to_save.inspection.slab = Some(Slab {
        enabled: true,
        anchor,
        offset: 0.,
        thickness,
    });
    state.to_save.save_flag = true;

    Ok(())
}

pub fn label_selection(
    state: &mut State,
    input: &str,
    format: Option<LabelFormat>,
) -> Result<usize, String> {
    let selected = expression::evaluate(state, input)?;
    let mut targets = HashSet::new();
    let mut labels = Vec::new();
    for s in selected {
        let Some(mol) = state.get_mol(s.mol_type, s.mol_i) else {
            continue;
        };
        let common = mol.common();
        let key = expression::object_key(&mol);
        let residues = match &mol {
            MolGenericRef::Peptide(m) => Some(&m.residues),
            MolGenericRef::NucleicAcid(m) => Some(&m.residues),
            MolGenericRef::Lipid(m) => Some(&m.residues),
            _ => None,
        };
        let mut seen = HashSet::new();
        for &a in &s.atoms {
            let atom = &common.atoms[a];
            targets.insert((key.clone(), atom.serial_number));
            let Some(format) = format else { continue };
            let residue = atom.residue.and_then(|r| residues?.get(r));
            let (text, members) = if format != LabelFormat::AtomName {
                let Some(residue) = residue else { continue };
                if !seen.insert((atom.chain, atom.residue)) {
                    continue;
                }

                let name = expression::residue_name(residue).to_ascii_uppercase();
                let number = residue.serial_number;
                let chain = match &mol {
                    MolGenericRef::Peptide(m) => atom
                        .chain
                        .and_then(|c| m.chains.get(c))
                        .map(|c| c.id.as_str())
                        .unwrap_or(""),
                    _ => "",
                };
                let text = if format == LabelFormat::ChainResidue {
                    format!("{chain}:{name}{number}")
                } else {
                    format!("{name} {number}")
                };
                let members = residue
                    .atoms
                    .iter()
                    .copied()
                    .filter(|&i| common.atoms.get(i).is_some_and(|a| a.chain == atom.chain))
                    .collect::<Vec<_>>();

                (text, members)
            } else {
                let name = expression::atom_name(atom);

                (
                    if name.is_empty() {
                        format!("{}{}", atom.element, atom.serial_number)
                    } else {
                        name
                    },
                    vec![a],
                )
            };
            let atoms = members
                .iter()
                .map(|&i| AtomRef {
                    object: key.clone(),
                    serial: common.atoms[i].serial_number,
                    trajectory: false,
                })
                .collect();
            labels.push(SelectionLabel {
                atoms,
                text,
                offset: [16., -16.],
                visible: true,
            });
        }
    }
    let count = if format.is_some() {
        labels.len()
    } else {
        targets.len()
    };
    state.to_save.inspection.labels.retain(|l| {
        !l.atoms
            .iter()
            .any(|a| targets.contains(&(a.object.clone(), a.serial)))
    });
    state.to_save.inspection.labels.extend(labels);
    state.to_save.save_flag = true;

    Ok(count)
}
