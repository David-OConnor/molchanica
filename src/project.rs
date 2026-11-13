//! Portable molecular workspaces.

use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};
use bio_files::{BackboneSS, BondType, DensityMap, ResidueEnd, ResidueType, SecondaryStructure};
use dynamics::{SimBox, snapshot::Snapshot};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;
use mol_defs::{
    molecules::{
        Atom, AtomRole, Bond, Chain, MolGenericRef, MolIdent, MolType, PeptideIdent, Residue,
        common::{MolVisibility, MoleculeCommon},
        lipid::MoleculeLipid,
        nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType},
        peptide::MoleculePeptide,
        pocket::Pocket,
        small::MoleculeSmall,
    },
    reflection::{DensityPt, DensityRect},
    screening::pharmacophore::{
        FeatureRelation, Motion, Pharmacophore, PharmacophoreFeatType, PharmacophoreFeature,
    },
};
use na_seq::{
    AtomTypeInRes, Element, Nucleotide, SeqFeature, SeqRange, SeqTopology, Sequence, SequenceData,
    Strand,
};

use crate::{
    inspection::SceneState,
    md::viewer::{ViewerMolSet, ViewerMolecule},
    selection::expression,
    state::{CamSnapshot, OperatingMode, State},
};

const MAGIC: &[u8] = b"MOLCHANICA-PROJECT\0\x02";
const LEGACY_MAGIC: &[u8] = b"MOLCHANICA-PROJECT\0\x01";
const MAX_BYTES: u64 = 1024 * 1024 * 1024;

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// 0 and 1 match the bool this replaced, so older projects load.
pub(crate) fn vis_to_byte(vis: MolVisibility) -> u8 {
    match vis {
        MolVisibility::Hidden => 0,
        MolVisibility::Visible => 1,
        MolVisibility::Dimmed => 2,
    }
}

pub(crate) fn vis_from_byte(v: u8) -> MolVisibility {
    match v {
        0 => MolVisibility::Hidden,
        2 => MolVisibility::Dimmed,
        _ => MolVisibility::Visible,
    }
}

struct Temporary(PathBuf);

impl Temporary {
    fn new(extension: &str) -> Self {
        Self(std::env::temp_dir().join(format!(
            "molchanica-{:016x}.{extension}",
            rand::random::<u64>()
        )))
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Encode, Decode)]
struct SavedAtom {
    serial: u32,
    role: Option<u8>,
    position: Vec3,
    element: String,
    name: Option<(u8, String)>,
    general_name: Option<String>,
    force_field: Option<String>,
    residue: Option<usize>,
    chain: Option<usize>,
    hetero: bool,
    occupancy: Option<f32>,
    charge: Option<f32>,
    alternate: Option<String>,
}

#[derive(Encode, Decode)]
struct Common {
    ident: String,
    name: Option<String>,
    path: Option<PathBuf>,
    filename: String,
    atoms: Vec<SavedAtom>,
    positions: Vec<Vec3>,
    bonds: Vec<(usize, usize, u8, bool)>,
    metadata: HashMap<String, String>,
    /// See `vis_to_byte`.
    visible: u8,
    md_copies: Option<usize>,
    is_2d: bool,
}

impl Common {
    fn capture(m: &MoleculeCommon) -> Self {
        Self {
            ident: m.ident.clone(),
            name: m.name.clone(),
            path: m.path.clone(),
            filename: m.filename.clone(),
            positions: m.atom_posits.clone(),
            metadata: m.metadata.clone(),
            visible: vis_to_byte(m.visibility),
            md_copies: m.selected_for_md,
            is_2d: m.is_2d,
            atoms: m
                .atoms
                .iter()
                .map(|a| SavedAtom {
                    serial: a.serial_number,
                    role: a.role.map(|r| r as u8),
                    position: a.posit,
                    element: a.element.to_letter().into(),
                    name: a.type_in_res.as_ref().map(|name| match name {
                        AtomTypeInRes::H(v) => (1, v.clone()),
                        AtomTypeInRes::Hetero(v) => (2, v.clone()),
                        _ => (0, name.to_string()),
                    }),
                    general_name: a.type_in_res_general.clone(),
                    force_field: a.force_field_type.clone(),
                    residue: a.residue,
                    chain: a.chain,
                    hetero: a.hetero,
                    occupancy: a.occupancy,
                    charge: a.partial_charge,
                    alternate: a.alt_conformation_id.clone(),
                })
                .collect(),
            bonds: m
                .bonds
                .iter()
                .map(|b| (b.atom_0, b.atom_1, b.bond_type as u8, b.is_backbone))
                .collect(),
        }
    }

    fn restore(self) -> io::Result<MoleculeCommon> {
        let mut serials = HashSet::new();
        if self.positions.len() != self.atoms.len() || self.positions.iter().any(|p| !finite(*p)) {
            return Err(invalid("Invalid atom coordinates"));
        }
        let atoms = self
            .atoms
            .into_iter()
            .map(|a| {
                if !serials.insert(a.serial) || !finite(a.position) {
                    return Err(invalid("Invalid or duplicate atom"));
                }
                let name = a
                    .name
                    .map(|(kind, name)| match kind {
                        0 => name.parse(),
                        1 => Ok(AtomTypeInRes::H(name)),
                        2 => Ok(AtomTypeInRes::Hetero(name)),
                        _ => Err(invalid("Invalid atom name")),
                    })
                    .transpose()?;
                let roles = [
                    AtomRole::C_Alpha,
                    AtomRole::C_Prime,
                    AtomRole::N_Backbone,
                    AtomRole::O_Backbone,
                    AtomRole::H_Backbone,
                    AtomRole::Sidechain,
                    AtomRole::H_Sidechain,
                    AtomRole::Water,
                ];
                let role = a
                    .role
                    .map(|r| {
                        roles
                            .get(r as usize)
                            .copied()
                            .ok_or_else(|| invalid("Invalid atom role"))
                    })
                    .transpose()?;

                Ok(Atom {
                    serial_number: a.serial,
                    posit: a.position,
                    element: Element::from_letter(&a.element)?,
                    role,
                    type_in_res: name,
                    type_in_res_general: a.general_name,
                    force_field_type: a.force_field,
                    residue: a.residue,
                    chain: a.chain,
                    hetero: a.hetero,
                    occupancy: a.occupancy,
                    partial_charge: a.charge,
                    alt_conformation_id: a.alternate,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let bonds = self
            .bonds
            .into_iter()
            .map(|(a, b, kind, backbone)| {
                let types = [
                    BondType::Single,
                    BondType::Double,
                    BondType::Triple,
                    BondType::Aromatic,
                    BondType::Amide,
                    BondType::Dummy,
                    BondType::Unknown,
                    BondType::NotConnected,
                    BondType::Quadruple,
                    BondType::Delocalized,
                    BondType::PolymericLink,
                ];
                if a >= atoms.len() || b >= atoms.len() || a == b {
                    return Err(invalid("Invalid bond endpoint"));
                }

                Ok(Bond {
                    atom_0: a,
                    atom_1: b,
                    atom_0_sn: atoms[a].serial_number,
                    atom_1_sn: atoms[b].serial_number,
                    bond_type: *types
                        .get(kind as usize)
                        .ok_or_else(|| invalid("Invalid bond type"))?,
                    is_backbone: backbone,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let mut common = MoleculeCommon::new(self.ident, atoms, bonds, self.metadata, self.path);
        common.name = self.name;
        common.filename = self.filename;
        common.atom_posits = self.positions;
        common.visibility = vis_from_byte(self.visible);
        common.selected_for_md = self.md_copies;
        common.is_2d = self.is_2d;
        common.update_next_sn();

        Ok(common)
    }
}

fn finite(p: Vec3) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

#[derive(Encode, Decode)]
struct SavedResidue {
    serial: u32,
    name: String,
    atoms: Vec<usize>,
    end: u8,
}

impl SavedResidue {
    fn capture(r: &Residue) -> Self {
        Self {
            serial: r.serial_number,
            name: match &r.res_type {
                ResidueType::Water => "HOH".into(),
                _ => r.res_type.to_string(),
            },
            atoms: r.atoms.clone(),
            end: r.end as u8,
        }
    }

    fn restore(self, atoms: &[Atom]) -> io::Result<Residue> {
        let ends = [
            ResidueEnd::Internal,
            ResidueEnd::NTerminus,
            ResidueEnd::CTerminus,
            ResidueEnd::Hetero,
        ];
        let atom_sns = self
            .atoms
            .iter()
            .map(|&a| {
                atoms
                    .get(a)
                    .map(|a| a.serial_number)
                    .ok_or_else(|| invalid("Invalid residue atom"))
            })
            .collect::<io::Result<_>>()?;

        Ok(Residue {
            serial_number: self.serial,
            res_type: ResidueType::from_str(&self.name),
            atoms: self.atoms,
            atom_sns,
            dihedral: None,
            end: *ends
                .get(self.end as usize)
                .ok_or_else(|| invalid("Invalid residue end"))?,
        })
    }
}

#[derive(Encode, Decode)]
struct SavedChain {
    id: String,
    residues: Vec<usize>,
    atoms: Vec<usize>,
    visible: bool,
}

#[derive(Encode, Decode)]
struct Density {
    map: Option<Vec<u8>>,
    points: Option<Vec<(Vec3, f64)>>,
    brick: Option<(Vec3, [f64; 3], [usize; 3], Vec<f32>)>,
}

impl Density {
    fn capture(p: &MoleculePeptide) -> io::Result<Self> {
        let map = if let Some(map) = &p.density_map {
            let file = Temporary::new("map");
            map.save(&file.0)?;
            Some(fs::read(&file.0)?)
        } else {
            None
        };

        Ok(Self {
            map,
            points: p
                .elec_density
                .as_ref()
                .map(|p| p.iter().map(|p| (p.coords, p.density)).collect()),
            brick: p
                .density_rect
                .as_ref()
                .map(|b| (b.origin_cart, b.step, b.dims, b.data.clone())),
        })
    }

    fn restore(self, p: &mut MoleculePeptide) -> io::Result<()> {
        if let Some(map) = self.map {
            let file = Temporary::new("map");
            fs::write(&file.0, map)?;
            p.density_map = Some(DensityMap::load(&file.0)?);
        }
        if let Some(points) = self.points {
            if points.iter().any(|(p, d)| !finite(*p) || !d.is_finite()) {
                return Err(invalid("Invalid density points"));
            }
            p.elec_density = Some(
                points
                    .into_iter()
                    .map(|(coords, density)| DensityPt { coords, density })
                    .collect(),
            );
        }
        if let Some((origin_cart, step, dims, data)) = self.brick {
            if dims.iter().try_fold(1usize, |n, d| n.checked_mul(*d)) != Some(data.len())
                || !finite(origin_cart)
                || step.iter().any(|v| !v.is_finite() || *v <= 0.)
                || data.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("Invalid density grid"));
            }
            p.density_rect = Some(DensityRect {
                origin_cart,
                step,
                dims,
                data,
            });
        }

        Ok(())
    }
}

/// Molecules held until the whole project validates. Not a `State`: building one on the UI thread
/// overflows the stack in debug builds.
#[derive(Default)]
struct RestoredMols {
    peptides: Vec<MoleculePeptide>,
    ligands: Vec<MoleculeSmall>,
    nucleic_acids: Vec<MoleculeNucleicAcid>,
    lipids: Vec<MoleculeLipid>,
    pockets: Vec<Pocket>,
}

#[derive(Encode, Decode)]
struct Molecule {
    pharmacophore: Option<SavedPharmacophore>,
    parameterized: (bool, bool),
    kind: u8,
    common: Common,
    residues: Vec<SavedResidue>,
    chains: Vec<SavedChain>,
    secondary: Vec<(u32, u32, u8)>,
    cif: Option<String>,
    density: Option<Density>,
    peptide_ids: Vec<PeptideIdent>,
    small_ids: Vec<MolIdent>,
    nucleic: Option<(
        NucleicAcidType,
        Vec<Nucleotide>,
        Vec<(String, (usize, usize))>,
    )>,
    lipid: Option<(String, String, String, String)>,
}

impl Molecule {
    fn capture(mol: &MolGenericRef<'_>, full: bool) -> io::Result<Self> {
        let (residues, chains) = match mol {
            MolGenericRef::Peptide(p) => (&p.residues[..], &p.chains[..]),
            MolGenericRef::NucleicAcid(p) => (&p.residues[..], &[][..]),
            MolGenericRef::Lipid(p) => (&p.residues[..], &[][..]),
            _ => (&[][..], &[][..]),
        };

        Ok(Self {
            pharmacophore: match mol {
                MolGenericRef::Small(p) if full => {
                    Some(SavedPharmacophore::capture(&p.pharmacophore))
                }
                _ => None,
            },
            parameterized: match mol {
                MolGenericRef::Small(p) => (p.ff_params_loaded, p.frcmod_loaded),
                _ => (false, false),
            },
            kind: kind_byte(mol.mol_type()),
            common: Common::capture(mol.common()),
            residues: residues.iter().map(SavedResidue::capture).collect(),
            chains: chains
                .iter()
                .map(|c| SavedChain {
                    id: c.id.clone(),
                    residues: c.residues.clone(),
                    atoms: c.atoms.clone(),
                    visible: c.visible,
                })
                .collect(),
            secondary: match mol {
                MolGenericRef::Peptide(p) => p
                    .secondary_structure
                    .iter()
                    .map(|s| (s.start_sn, s.end_sn, s.sec_struct as u8))
                    .collect(),
                _ => Vec::new(),
            },
            cif: match mol {
                MolGenericRef::Peptide(p) if full => p.source_cif.clone(),
                _ => None,
            },
            density: match mol {
                MolGenericRef::Peptide(p) if full => Some(Density::capture(p)?),
                _ => None,
            },
            peptide_ids: match mol {
                MolGenericRef::Peptide(p) => p.idents.clone(),
                _ => Vec::new(),
            },
            small_ids: match mol {
                MolGenericRef::Small(p) => p.idents.clone(),
                _ => Vec::new(),
            },
            nucleic: match mol {
                MolGenericRef::NucleicAcid(p) => {
                    Some((p.na_type, p.seq.clone(), p.features.clone()))
                }
                _ => None,
            },
            lipid: match mol {
                MolGenericRef::Lipid(p) => Some((
                    p.lmsd_id.clone(),
                    p.hmdb_id.clone(),
                    p.kegg_id.clone(),
                    p.common_name.clone(),
                )),
                _ => None,
            },
        })
    }

    fn restore(self, mols: &mut RestoredMols) -> io::Result<()> {
        let common = self.common.restore()?;
        let residues = self
            .residues
            .into_iter()
            .map(|r| r.restore(&common.atoms))
            .collect::<io::Result<Vec<_>>>()?;
        let kind = kind_from(self.kind)?;
        if matches!(
            kind,
            MolType::Peptide | MolType::NucleicAcid | MolType::Lipid
        ) && common
            .atoms
            .iter()
            .any(|a| a.residue.is_some_and(|i| i >= residues.len()))
        {
            return Err(invalid("Invalid atom residue"));
        }
        match kind {
            MolType::Peptide => {
                let chains = self
                    .chains
                    .into_iter()
                    .map(|c| {
                        if c.residues.iter().any(|&i| i >= residues.len())
                            || c.atoms.iter().any(|&i| i >= common.atoms.len())
                        {
                            return Err(invalid("Invalid chain membership"));
                        }
                        let residue_sns = c
                            .residues
                            .iter()
                            .map(|&i| residues[i].serial_number)
                            .collect();
                        let atom_sns = c
                            .atoms
                            .iter()
                            .map(|&i| common.atoms[i].serial_number)
                            .collect();

                        Ok(Chain {
                            id: c.id,
                            residues: c.residues,
                            atoms: c.atoms,
                            visible: c.visible,
                            residue_sns,
                            atom_sns,
                        })
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if common
                    .atoms
                    .iter()
                    .any(|a| a.chain.is_some_and(|i| i >= chains.len()))
                {
                    return Err(invalid("Invalid atom chain"));
                }
                let mut p = MoleculePeptide::new(
                    common.ident.clone(),
                    common.atoms.clone(),
                    common.bonds.clone(),
                    chains,
                    residues,
                    common.metadata.clone(),
                    common.path.clone(),
                );
                p.common = common;
                p.source_cif = self.cif;
                p.idents = self.peptide_ids;
                p.secondary_structure = self
                    .secondary
                    .into_iter()
                    .map(|(start_sn, end_sn, kind)| {
                        Ok(BackboneSS {
                            start_sn,
                            end_sn,
                            sec_struct: match kind {
                                0 => SecondaryStructure::Helix,
                                1 => SecondaryStructure::Sheet,
                                2 => SecondaryStructure::Coil,
                                _ => return Err(invalid("Invalid secondary structure")),
                            },
                        })
                    })
                    .collect::<io::Result<_>>()?;
                if let Some(density) = self.density {
                    density.restore(&mut p)?;
                }
                mols.peptides.push(p);
            }
            MolType::Ligand => {
                let pharmacophore = self
                    .pharmacophore
                    .map(|p| p.restore(common.atoms.len()))
                    .transpose()?
                    .unwrap_or_default();
                mols.ligands.push(MoleculeSmall {
                    common,
                    idents: self.small_ids,
                    pharmacophore,
                    ff_params_loaded: self.parameterized.0,
                    frcmod_loaded: self.parameterized.1,
                    ..Default::default()
                });
            }
            MolType::NucleicAcid => {
                let (na_type, seq, features) = self
                    .nucleic
                    .ok_or_else(|| invalid("Missing nucleic acid data"))?;
                mols.nucleic_acids.push(MoleculeNucleicAcid {
                    common,
                    residues,
                    na_type,
                    seq,
                    features,
                });
            }
            MolType::Lipid => {
                let (lmsd_id, hmdb_id, kegg_id, common_name) =
                    self.lipid.ok_or_else(|| invalid("Missing lipid data"))?;
                mols.lipids.push(MoleculeLipid {
                    common,
                    residues,
                    lmsd_id,
                    hmdb_id,
                    kegg_id,
                    common_name,
                });
            }
            MolType::Pocket => {
                let mut p = Pocket::from(common);
                p.mesh_i_rel = mols.pockets.len();
                mols.pockets.push(p);
            }
            MolType::Water => return Err(invalid("Standalone water object")),
        }

        Ok(())
    }
}

fn kind_byte(kind: MolType) -> u8 {
    match kind {
        MolType::Peptide => 0,
        MolType::Ligand => 1,
        MolType::NucleicAcid => 2,
        MolType::Lipid => 3,
        MolType::Pocket => 4,
        MolType::Water => 5,
    }
}
fn kind_from(kind: u8) -> io::Result<MolType> {
    match kind {
        0 => Ok(MolType::Peptide),
        1 => Ok(MolType::Ligand),
        2 => Ok(MolType::NucleicAcid),
        3 => Ok(MolType::Lipid),
        4 => Ok(MolType::Pocket),
        5 => Ok(MolType::Water),
        _ => Err(invalid("Invalid molecule type")),
    }
}

#[derive(Encode, Decode)]
struct SavedPharmacophore {
    name: String,
    ident: String,
    features: Vec<SavedPharmacophoreFeature>,
    relations: Vec<(bool, usize, usize)>,
    pocket: Option<Common>,
}

#[derive(Encode, Decode)]
struct SavedPharmacophoreFeature {
    kind: u8,
    position: Vec3,
    projected: Option<Vec3>,
    atoms: Vec<usize>,
    projected_atom: Option<usize>,
    strength: f32,
    tolerance: f32,
    motion: Option<Motion>,
    selected: bool,
}

impl SavedPharmacophore {
    fn capture(p: &Pharmacophore) -> Self {
        Self {
            name: p.name.clone(),
            ident: p.mol_ident.clone(),
            pocket: p.pocket.as_ref().map(|p| Common::capture(&p.common)),
            relations: p
                .feature_relations
                .iter()
                .map(|r| match r {
                    FeatureRelation::And((a, b)) => (false, *a, *b),
                    FeatureRelation::Or((a, b)) => (true, *a, *b),
                })
                .collect(),
            features: p
                .features
                .iter()
                .map(|f| SavedPharmacophoreFeature {
                    kind: f.feature_type as u8,
                    position: f.posit,
                    projected: f.posit_projected,
                    atoms: f.atom_i.clone(),
                    projected_atom: f.atom_i_projected,
                    strength: f.strength,
                    tolerance: f.tolerance,
                    motion: f.oscillation.clone(),
                    selected: f.ui_selected,
                })
                .collect(),
        }
    }

    fn restore(self, atom_count: usize) -> io::Result<Pharmacophore> {
        let features = self
            .features
            .into_iter()
            .map(|f| {
                if f.atoms.iter().any(|&a| a >= atom_count)
                    || f.projected_atom.is_some_and(|a| a >= atom_count)
                    || !finite(f.position)
                    || !f.strength.is_finite()
                    || !f.tolerance.is_finite()
                {
                    return Err(invalid("Invalid pharmacophore feature"));
                }

                Ok(PharmacophoreFeature {
                    feature_type: PharmacophoreFeatType::from_u8(f.kind)
                        .ok_or_else(|| invalid("Invalid pharmacophore type"))?,
                    posit: f.position,
                    posit_projected: f.projected,
                    atom_i: f.atoms,
                    atom_i_projected: f.projected_atom,
                    strength: f.strength,
                    tolerance: f.tolerance,
                    oscillation: f.motion,
                    ui_selected: f.selected,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let feature_relations = self
            .relations
            .into_iter()
            .map(|(or, a, b)| {
                if a >= features.len() || b >= features.len() {
                    return Err(invalid("Invalid pharmacophore relation"));
                }

                Ok(if or {
                    FeatureRelation::Or((a, b))
                } else {
                    FeatureRelation::And((a, b))
                })
            })
            .collect::<io::Result<_>>()?;
        let pocket = self
            .pocket
            .map(|p| p.restore().map(Pocket::from))
            .transpose()?;

        Ok(Pharmacophore {
            name: self.name,
            mol_ident: self.ident,
            features,
            feature_relations,
            pocket,
        })
    }
}

#[derive(Encode, Decode)]
struct Frame {
    time: f64,
    positions: Vec<lin_alg::f32::Vec3>,
    velocities: Option<Vec<lin_alg::f32::Vec3>>,
    cell: Option<SimBox>,
    water: [Vec<lin_alg::f32::Vec3>; 3],
}

impl Frame {
    fn capture(s: &Snapshot) -> Self {
        Self {
            time: s.time,
            positions: s.atom_posits.clone(),
            velocities: s.atom_velocities.clone(),
            cell: s.cell,
            water: [
                s.water_o_posits.clone(),
                s.water_h0_posits.clone(),
                s.water_h1_posits.clone(),
            ],
        }
    }

    fn restore(self) -> io::Result<Snapshot> {
        if !self.time.is_finite()
            || self
                .positions
                .iter()
                .chain(self.water.iter().flatten())
                .any(|p| !finite((*p).into()))
            || self.water[0].len() != self.water[1].len()
            || self.water[0].len() != self.water[2].len()
        {
            return Err(invalid("Invalid trajectory frame"));
        }
        let [water_o_posits, water_h0_posits, water_h1_posits] = self.water;

        Ok(Snapshot {
            time: self.time,
            atom_posits: self.positions,
            atom_velocities: self.velocities,
            cell: self.cell,
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
            ..Default::default()
        })
    }
}

#[derive(Encode, Decode)]
struct MolSet {
    name: String,
    mols: Vec<(u8, Common, (usize, usize))>,
    visibility: Vec<bool>,
}

impl MolSet {
    fn capture(set: &ViewerMolSet) -> Self {
        Self {
            name: set.name.clone(),
            mols: set
                .mols
                .iter()
                .map(|m| (kind_byte(m.mol_type), Common::capture(&m.mol), m.range))
                .collect(),
            visibility: set.groups.iter().map(|g| g.visible).collect(),
        }
    }

    fn restore(self) -> io::Result<ViewerMolSet> {
        let mols = self
            .mols
            .into_iter()
            .map(|(kind, common, range)| {
                let mol = common.restore()?;
                if range.1.checked_sub(range.0) != Some(mol.atoms.len()) {
                    return Err(invalid("Invalid trajectory atom range"));
                }

                Ok(ViewerMolecule {
                    mol_type: kind_from(kind)?,
                    mol,
                    range,
                })
            })
            .collect::<io::Result<_>>()?;
        let mut result = ViewerMolSet::new(None, self.name, mols);
        if result.range_overlaps {
            return Err(invalid("Overlapping trajectory atom ranges"));
        }
        for (g, visible) in result.groups.iter_mut().zip(self.visibility) {
            g.visible = visible;
        }

        Ok(result)
    }
}

pub(crate) const MD_MOLECULE_SET_FILE: &str = "molecules.mcmd";
const MD_MOL_SET_MAGIC: &[u8] = b"MCMS1";

pub(crate) fn save_md_mol_set(path: &Path, set: &ViewerMolSet) -> io::Result<()> {
    let bytes = bincode::encode_to_vec(MolSet::capture(set), bincode::config::standard())
        .map_err(invalid)?;
    let mut file = fs::File::create(path)?;
    file.write_all(MD_MOL_SET_MAGIC)?;
    file.write_all(&bytes)
}

pub(crate) fn load_md_mol_set(path: &Path) -> io::Result<ViewerMolSet> {
    let bytes = fs::read(path)?;
    let payload = bytes
        .strip_prefix(MD_MOL_SET_MAGIC)
        .ok_or_else(|| invalid("Invalid MD molecule set"))?;
    let (set, consumed): (MolSet, usize) = bincode::decode_from_slice(
        payload,
        bincode::config::standard().with_limit::<{ 256 << 20 }>(),
    )
    .map_err(invalid)?;
    if consumed != payload.len() {
        return Err(invalid("Unexpected data in MD molecule set"));
    }

    set.restore()
}

pub(crate) const MD_INPUTS_FILE: &str = "molchanica-md.bin";
const MD_INPUTS_MAGIC: &[u8] = b"MCMD1";

#[derive(Encode, Decode)]
struct SavedMdInputs {
    config: dynamics::MdConfig,
    molecules: Vec<Molecule>,
    viewer: MolSet,
    parameters: Vec<(String, String)>,
    peptide_only_near_ligs: bool,
    peptide_static: bool,
    alchemical: Option<f64>,
    gfnff: Option<(Option<f64>, Option<String>)>,
}

pub(crate) struct MdInputs {
    pub config: dynamics::MdConfig,
    pub viewer: ViewerMolSet,
    mols: RestoredMols,
    parameters: HashMap<String, bio_files::md_params::ForceFieldParams>,
    peptide_only_near_ligs: bool,
    peptide_static: bool,
    alchemical: Option<f64>,
    gfnff: Option<(Option<f64>, Option<String>)>,
}

pub(crate) fn save_md_inputs(
    state: &State,
    set: &ViewerMolSet,
    config: &dynamics::MdConfig,
    dir: &Path,
) -> io::Result<()> {
    let mut molecules = Vec::new();
    let mut parameters = Vec::new();
    for (_, _, mol) in expression::objects(state) {
        let common = mol.common();
        if common.selected_for_md.is_none() || mol.mol_type() == MolType::Pocket {
            continue;
        }
        if let Some(params) = state.mol_specific_params.get(&common.ident) {
            let file = Temporary::new("dat");
            params.save_dat(&file.0, "Molchanica MD parameters")?;
            parameters.push((common.ident.clone(), fs::read_to_string(&file.0)?));
        }
        molecules.push(Molecule::capture(&mol, false)?);
    }

    let gfnff = state.selected_force_field().gfnff();
    if let Some(path) = gfnff.and_then(|set| set.parameter_file.as_ref()) {
        fs::copy(path, dir.join(".param_gfnff.xtb"))?;
    }
    let saved = SavedMdInputs {
        config: config.clone(),
        molecules,
        viewer: MolSet::capture(set),
        parameters,
        peptide_only_near_ligs: state.ui.md.peptide_only_near_ligs,
        peptide_static: state.ui.md.peptide_static,
        alchemical: state
            .ui
            .md
            .alchemical_enabled
            .then_some(state.ui.md.alchemical_lambda),
        gfnff: gfnff.map(|set| (set.total_charge, set.alpb_solvent.clone())),
    };
    let bytes = bincode::encode_to_vec(&saved, bincode::config::standard()).map_err(invalid)?;
    let mut file = fs::File::create(dir.join(MD_INPUTS_FILE))?;
    file.write_all(MD_INPUTS_MAGIC)?;
    file.write_all(&bytes)
}

impl MdInputs {
    pub(crate) fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let bytes = bytes
            .strip_prefix(MD_INPUTS_MAGIC)
            .ok_or_else(|| invalid("Unsupported Molchanica MD input file"))?;
        let (saved, used): (SavedMdInputs, usize) = bincode::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<{ 256 << 20 }>(),
        )
        .map_err(invalid)?;
        if used != bytes.len() {
            return Err(invalid("Unexpected data in Molchanica MD inputs"));
        }

        let mut mols = RestoredMols::default();
        for molecule in saved.molecules {
            molecule.restore(&mut mols)?;
        }
        let parameters = saved
            .parameters
            .into_iter()
            .map(|(name, text)| {
                bio_files::md_params::ForceFieldParams::from_dat(&text).map(|params| (name, params))
            })
            .collect::<io::Result<_>>()?;

        Ok(Self {
            config: saved.config,
            viewer: saved.viewer.restore()?,
            mols,
            parameters,
            peptide_only_near_ligs: saved.peptide_only_near_ligs,
            peptide_static: saved.peptide_static,
            alchemical: saved.alchemical,
            gfnff: saved.gfnff,
        })
    }

    pub(crate) fn apply(self, state: &mut State, dir: &Path) {
        fn select<T>(
            current: &mut Vec<T>,
            saved: Vec<T>,
            common: impl Fn(&mut T) -> &mut MoleculeCommon,
        ) {
            for mol in current.iter_mut() {
                common(mol).selected_for_md = None;
            }
            for mut mol in saved {
                let reference = common(&mut mol);
                let existing = current.iter_mut().map(&common).find(|existing| {
                    existing.selected_for_md.is_none()
                        && existing.ident == reference.ident
                        && existing.atom_posits == reference.atom_posits
                        && existing
                            .atoms
                            .iter()
                            .map(|a| a.element)
                            .eq(reference.atoms.iter().map(|a| a.element))
                });
                if let Some(existing) = existing {
                    existing.selected_for_md = reference.selected_for_md;
                } else {
                    current.push(mol);
                }
            }
        }

        select(&mut state.peptides, self.mols.peptides, |mol| {
            &mut mol.common
        });
        select(&mut state.ligands, self.mols.ligands, |mol| &mut mol.common);
        select(&mut state.lipids, self.mols.lipids, |mol| &mut mol.common);
        select(&mut state.nucleic_acids, self.mols.nucleic_acids, |mol| {
            &mut mol.common
        });
        state.mol_specific_params.extend(self.parameters);
        state.ui.md.peptide_only_near_ligs = self.peptide_only_near_ligs;
        state.ui.md.peptide_static = self.peptide_static;
        state.ui.md.alchemical_enabled = self.alchemical.is_some();
        if let Some(lambda) = self.alchemical {
            state.ui.md.alchemical_lambda = lambda;
            state.ui.md.alchemical_lambda_input = lambda.to_string();
        }
        if let Some((charge, solvent)) = self.gfnff
            && let Some(set) = state
                .force_fields
                .params
                .as_mut()
                .and_then(|p| p.gfnff_mut())
        {
            set.total_charge = charge;
            set.alpb_solvent = solvent;
            let path = dir.join(".param_gfnff.xtb");
            set.parameter_file = path.is_file().then_some(path);
        }
        state.force_fields.invalidate();
        state.refresh_force_fields();
    }
}

#[derive(Encode, Decode)]
enum SequenceResidues {
    Protein(Vec<na_seq::AminoAcid>),
    Dna(Vec<Nucleotide>),
    Rna(Vec<Nucleotide>),
}

#[derive(Encode, Decode)]
struct Feature {
    kind: String,
    label: String,
    ranges: Vec<(usize, usize)>,
    strand: u8,
    color: Option<String>,
    qualifiers: Vec<(String, String)>,
}

#[derive(Encode, Decode)]
struct SavedSequence {
    data: SequenceResidues,
    name: String,
    metadata: HashMap<String, String>,
    topology: Option<SeqTopology>,
    features: Vec<Feature>,
    path: Option<PathBuf>,
}

impl SavedSequence {
    fn capture(s: &Sequence) -> Self {
        Self {
            data: match &s.data {
                SequenceData::AminoAcid(v) => SequenceResidues::Protein(v.clone()),
                SequenceData::Dna(v) => SequenceResidues::Dna(v.clone()),
                SequenceData::Rna(v) => SequenceResidues::Rna(v.clone()),
            },
            name: s.name.clone(),
            metadata: s.metadata.clone(),
            topology: s.topology,
            path: s.path.clone(),
            features: s
                .features
                .iter()
                .map(|f| Feature {
                    kind: f.kind.clone(),
                    label: f.label.clone(),
                    ranges: f.ranges.iter().map(|r| (r.start, r.end)).collect(),
                    strand: f.strand as u8,
                    color: f.color.clone(),
                    qualifiers: f.qualifiers.clone(),
                })
                .collect(),
        }
    }

    fn restore(self) -> io::Result<Sequence> {
        let data = match self.data {
            SequenceResidues::Protein(v) => SequenceData::AminoAcid(v),
            SequenceResidues::Dna(v) => SequenceData::Dna(v),
            SequenceResidues::Rna(v) => SequenceData::Rna(v),
        };
        let features = self
            .features
            .into_iter()
            .map(|f| {
                if f.ranges
                    .iter()
                    .any(|&(a, b)| a == 0 || b == 0 || a > data.len() || b > data.len())
                {
                    return Err(invalid("Invalid sequence feature range"));
                }

                Ok(SeqFeature {
                    kind: f.kind,
                    label: f.label,
                    ranges: f
                        .ranges
                        .into_iter()
                        .map(|(start, end)| SeqRange { start, end })
                        .collect(),
                    strand: match f.strand {
                        0 => Strand::None,
                        1 => Strand::Forward,
                        2 => Strand::Reverse,
                        _ => return Err(invalid("Invalid sequence strand")),
                    },
                    color: f.color,
                    qualifiers: f.qualifiers,
                })
            })
            .collect::<io::Result<_>>()?;

        Ok(Sequence {
            data,
            name: self.name,
            metadata: self.metadata,
            topology: self.topology,
            features,
            path: self.path,
        })
    }
}

#[derive(Encode, Decode)]
struct Project {
    sequences: Vec<SavedSequence>,
    parameters: Vec<(String, String)>,
    molecules: Vec<Molecule>,
    camera: CamSnapshot,
    appearance: SceneState,
    scenes: Vec<CamSnapshot>,
    frames: Vec<Frame>,
    mol_sets: Vec<MolSet>,
    active: Option<(u8, usize)>,
}

pub fn save(state: &State, scene: &Scene, path: &Path) -> io::Result<()> {
    if state.volatile.operating_mode != OperatingMode::Primary {
        return Err(invalid("Finish molecule editing before saving a project"));
    }
    let viewer = &state.volatile.md_local.viewer;
    let mut parameters = Vec::new();
    for (name, params) in &state.mol_specific_params {
        let file = Temporary::new("dat");
        params.save_dat(&file.0, "Molchanica project parameters")?;
        parameters.push((name.clone(), fs::read_to_string(&file.0)?));
    }
    let project = Project {
        sequences: state.sequences.iter().map(SavedSequence::capture).collect(),
        parameters,
        molecules: expression::objects(state)
            .iter()
            .map(|(_, _, m)| Molecule::capture(m, true))
            .collect::<io::Result<_>>()?,
        camera: CamSnapshot::from_cam(&scene.camera, "Current view".into()),
        appearance: SceneState::capture(state, scene),
        scenes: state.cam_snapshots.clone(),
        frames: viewer.snapshots.iter().map(|s| Frame::capture(s)).collect(),
        mol_sets: viewer.mol_sets.iter().map(MolSet::capture).collect(),
        active: state.volatile.active_mol.map(|(k, i)| (kind_byte(k), i)),
    };
    let bytes = bincode::encode_to_vec(project, bincode::config::standard()).map_err(invalid)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid("Project exceeds 1 GiB"));
    }
    let temporary = path.with_file_name(format!(".molchanica-{:016x}.tmp", rand::random::<u64>()));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(MAGIC)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(&bytes)?;
        encoder.finish()?.sync_all()?;
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }

    result
}

pub fn load(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    path: &Path,
) -> io::Result<()> {
    if !state.volatile.ongoing_computations.list().is_empty() {
        return Err(invalid(
            "Wait for active computations before opening a project",
        ));
    }
    let mut file = fs::File::open(path)?;
    let mut magic = vec![0; MAGIC.len()];
    file.read_exact(&mut magic)?;
    if magic != MAGIC && magic != LEGACY_MAGIC {
        return Err(invalid("Unsupported Molchanica project version"));
    }
    let mut bytes = Vec::new();
    GzDecoder::new(file)
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid("Project exceeds 1 GiB"));
    }
    let (project, read): (Project, usize) = bincode::decode_from_slice(
        &bytes,
        bincode::config::standard().with_limit::<1073741824>(),
    )
    .map_err(invalid)?;
    if read != bytes.len() {
        return Err(invalid("Trailing project data"));
    }
    project.appearance.validate().map_err(invalid)?;
    for snapshot in &project.scenes {
        if let Some(appearance) = &snapshot.scene {
            appearance.validate().map_err(invalid)?;
        }
    }
    let mut restored = RestoredMols::default();
    for molecule in project.molecules {
        molecule.restore(&mut restored)?;
    }
    let sequences = project
        .sequences
        .into_iter()
        .map(SavedSequence::restore)
        .collect::<io::Result<_>>()?;
    let parameters = project
        .parameters
        .into_iter()
        .map(|(name, text)| {
            bio_files::md_params::ForceFieldParams::from_dat(&text).map(|params| (name, params))
        })
        .collect::<io::Result<_>>()?;
    let frames = project
        .frames
        .into_iter()
        .map(|frame| frame.restore().map(std::sync::Arc::new))
        .collect::<io::Result<Vec<_>>>()?;
    let sets = project
        .mol_sets
        .into_iter()
        .map(MolSet::restore)
        .collect::<io::Result<_>>()?;
    if !finite(project.camera.position.into())
        || !project.camera.far.is_finite()
        || project.camera.far <= project.appearance.near
        || !project.appearance.near.is_finite()
        || project.appearance.near <= 0.
        || !project.appearance.fov.is_finite()
        || !(0.01..3.13).contains(&project.appearance.fov)
    {
        return Err(invalid("Invalid project camera"));
    }
    state.peptides = restored.peptides;
    state.ligands = restored.ligands;
    state.nucleic_acids = restored.nucleic_acids;
    state.lipids = restored.lipids;
    state.pockets = restored.pockets;
    let mesh_start = crate::render::MESH_POCKET_START;
    scene.meshes.resize_with(
        scene
            .meshes
            .len()
            .max(mesh_start + state.pockets.len() + state.ligands.len()),
        graphics::Mesh::default,
    );
    for pocket in &state.pockets {
        scene.meshes[mesh_start + pocket.mesh_i_rel] = pocket.surface_mesh.clone();
    }
    for (i, ligand) in state.ligands.iter_mut().enumerate() {
        if let Some(pocket) = &mut ligand.pharmacophore.pocket {
            pocket.mesh_i_rel = state.pockets.len() + i;
            scene.meshes[mesh_start + pocket.mesh_i_rel] = pocket.surface_mesh.clone();
        }
    }
    state.sequences = sequences;
    state.mol_specific_params = parameters;
    state.trajectories.clear();
    if !frames.is_empty() {
        state
            .trajectories
            .push(crate::md::trajectory::Trajectory::new_in_memory(
                frames.clone(),
                "Project trajectory".into(),
                0.,
            ));
        state.trajectories.last_mut().unwrap().ui_active = true;
    }
    state.volatile.thread_receivers = Default::default();
    state.volatile.session_restore_started = true;
    state.volatile.operating_mode = OperatingMode::Primary;
    state.volatile.mol_manip = Default::default();
    state.volatile.representations = Default::default();
    state.volatile.flags = Default::default();
    state.volatile.md_local = Default::default();
    state.volatile.md_local.viewer.snapshots = frames;
    state.volatile.md_local.viewer.mol_sets = sets;
    state.volatile.active_seq = None;
    state.ui.seq_selection.clear();
    state.volatile.aa_seq_display_cache.dirty = true;
    state.cam_snapshots = project.scenes;
    state.volatile.active_mol = project.active.and_then(|(k, i)| {
        kind_from(k)
            .ok()
            .and_then(|k| state.get_mol(k, i).map(|_| (k, i)))
    });
    state.volatile.active_peptide = (!state.peptides.is_empty()).then_some(0);
    state.ui.cam_snapshot = None;
    state.ui.inspection.pending.clear();
    state.ui.inspection.picking = false;
    scene.entities.clear();
    scene.camera.position = project.camera.position;
    scene.camera.orientation = project.camera.orientation;
    scene.camera.far = project.camera.far;
    project.appearance.apply(state, scene, updates);
    state.ui.inspection.colors = state.to_save.inspection.colors.clone();
    state.ui.inspection.charge_palette = state.to_save.inspection.charge_palette;
    state.ui.inspection.solid_colors = state.to_save.inspection.solid_colors.clone();
    state.volatile.flags.scene_restored = true;
    state.volatile.flags.new_density_loaded = true;
    updates.entities = EntityUpdate::All;
    updates.meshes = true;
    updates.lighting = true;

    Ok(())
}
