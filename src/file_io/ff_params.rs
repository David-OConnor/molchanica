//! Loading, saving, and restoring the general force field parameters shown in the force field
//! parameter editor. Formats depend on the force field, and molecule class:
//! - Amber bonded and Lennard-Jones parameters: `.dat`, and `.frcmod`. These can also be saved as
//!   GROMACS `.itp` tables.
//! - Amber partial charges and FF types, by residue: `.lib` (atom tables only).
//! - OPLS-AA/L bonded and atom type tables: GROMACS `.itp`. Residue templates: GROMACS `.rtp`.

use std::{
    fs, io,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use bio_files::{
    LipidStandard, ResidueEnd,
    gromacs::{GromacsFfParams, Rtp},
    md_params::{
        ChargeParams, ForceFieldParams, NucleotideTemplate, parse_lib_lipid,
        parse_lib_nucleic_acid, parse_lib_peptide, save_lib,
    },
};
use dynamics::{
    ForceFieldFamily, merge_params,
    params::{
        amber::{AmberSet, ProtFfChargeMap, ProtFfChargeMapSet},
        opls::OplsSet,
    },
};
use na_seq::{AaIdent, AminoAcidGeneral, Nucleotide};

use crate::{
    force_fields::family_name,
    state::State,
    ui::popup::ff_params::{FfCat, ProtMapSel},
};

/// A set of parameters in the editor, which we load into, save, or restore.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ParamsTarget {
    pub family: ForceFieldFamily,
    pub cat: FfCat,
    /// For Amber's peptide charges only.
    pub prot_map: ProtMapSel,
}

/// What to do with the file picked in `FileDialogs::ff_params`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ParamsFileOp {
    Load(ParamsTarget),
    Save(ParamsTarget),
}

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg.into())
}

fn not_loaded() -> io::Error {
    err("These parameters aren't loaded")
}

impl ParamsTarget {
    /// File formats this set can be saved in, and loaded from: (description, extension). The first
    /// is the default.
    pub fn formats(&self) -> &'static [(&'static str, &'static str)] {
        match (self.family, self.cat) {
            (ForceFieldFamily::Opls, FfCat::PeptideQ) => &[("GROMACS residue templates", "rtp")],
            (ForceFieldFamily::Opls, _) => &[("GROMACS parameters", "itp")],
            (_, FfCat::PeptideQ | FfCat::LipidQ | FfCat::DnaQ | FfCat::RnaQ) => {
                &[("Amber library", "lib")]
            }
            _ => &[
                ("Amber parameters", "dat"),
                ("Amber parameter modification", "frcmod"),
                ("GROMACS parameters", "itp"),
            ],
        }
    }

    /// Formats we can load; a subset of `formats`. (We don't convert GROMACS tables to Amber's.)
    pub fn load_formats(&self) -> Vec<(&'static str, &'static str)> {
        self.formats()
            .iter()
            .copied()
            .filter(|(_, ext)| self.family != ForceFieldFamily::Amber || *ext != "itp")
            .collect()
    }

    /// For file names, and messages. E.g. "OPLS-AA/L small mol".
    pub fn descrip(&self) -> String {
        let mut name = format!("{} {}", family_name(self.family), self.cat.name());
        if self.family == ForceFieldFamily::Amber && self.cat == FfCat::PeptideQ {
            name += &format!(" ({})", self.prot_map.name());
        }
        name
    }

    /// A default file name, without extension. E.g. "amber_small_mol".
    pub fn file_stem(&self) -> String {
        self.descrip()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .split('_')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// The lowercase extension, if it's one this set supports.
fn extension(target: ParamsTarget, path: &Path, loading: bool) -> io::Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let formats = if loading {
        target.load_formats()
    } else {
        target.formats().to_vec()
    };

    if formats.iter().any(|(_, e)| *e == ext) {
        return Ok(ext);
    }

    let exts: Vec<_> = formats.iter().map(|(_, e)| format!(".{e}")).collect();
    Err(err(format!(
        "{} parameters use {}; not {:?}",
        target.descrip(),
        exts.join(", "),
        path.file_name().unwrap_or_default(),
    )))
}

fn amber_set(state: &State) -> io::Result<&AmberSet> {
    state
        .ff_param_set
        .amber()
        .ok_or_else(|| err("Amber parameters aren't loaded"))
}

fn amber_set_mut(state: &mut State) -> io::Result<&mut AmberSet> {
    state
        .ff_param_set
        .amber_mut()
        .ok_or_else(|| err("Amber parameters aren't loaded"))
}

fn opls_set(state: &State) -> io::Result<&OplsSet> {
    state
        .force_fields
        .params
        .as_ref()
        .and_then(|p| p.opls())
        .ok_or_else(|| err("OPLS parameters aren't loaded; select OPLS as the force field"))
}

fn opls_set_mut(state: &mut State) -> io::Result<&mut OplsSet> {
    state
        .force_fields
        .params
        .as_mut()
        .and_then(|p| p.opls_mut())
        .ok_or_else(|| err("OPLS parameters aren't loaded; select OPLS as the force field"))
}

/// Bonded and Lennard-Jones parameters for one of Amber's molecule classes.
pub fn amber_ff(set: &AmberSet, cat: FfCat) -> Option<&Option<ForceFieldParams>> {
    Some(match cat {
        FfCat::Peptide => &set.peptide,
        FfCat::SmallMol => &set.small_mol,
        FfCat::Dna => &set.dna,
        FfCat::Rna => &set.rna,
        FfCat::Lipids => &set.lipids,
        FfCat::Carbohydrates => &set.carbohydrates,
        _ => return None,
    })
}

pub fn amber_ff_mut(set: &mut AmberSet, cat: FfCat) -> Option<&mut Option<ForceFieldParams>> {
    Some(match cat {
        FfCat::Peptide => &mut set.peptide,
        FfCat::SmallMol => &mut set.small_mol,
        FfCat::Dna => &mut set.dna,
        FfCat::Rna => &mut set.rna,
        FfCat::Lipids => &mut set.lipids,
        FfCat::Carbohydrates => &mut set.carbohydrates,
        _ => return None,
    })
}

fn prot_map(set: &ProtFfChargeMapSet, sel: ProtMapSel) -> &ProtFfChargeMap {
    match sel {
        ProtMapSel::Internal => &set.internal,
        ProtMapSel::NTerminus => &set.n_terminus,
        ProtMapSel::CTerminus => &set.c_terminus,
    }
}

fn prot_map_mut(set: &mut ProtFfChargeMapSet, sel: ProtMapSel) -> &mut ProtFfChargeMap {
    match sel {
        ProtMapSel::Internal => &mut set.internal,
        ProtMapSel::NTerminus => &mut set.n_terminus,
        ProtMapSel::CTerminus => &mut set.c_terminus,
    }
}

/// The residue name used in Amber libs. E.g. "ALA", or "NALA" in the N-terminus lib.
fn aa_lib_tag(aa: &AminoAcidGeneral, sel: ProtMapSel) -> String {
    let tag = match aa {
        AminoAcidGeneral::Standard(a) => a.to_str(AaIdent::ThreeLetters).to_uppercase(),
        AminoAcidGeneral::Variant(v) => v.to_string(),
    };

    // Caps aren't prefixed in the terminus libs.
    if ["ACE", "NHE", "NME"].contains(&tag.as_str()) {
        return tag;
    }

    match sel {
        ProtMapSel::Internal => tag,
        ProtMapSel::NTerminus => format!("N{tag}"),
        ProtMapSel::CTerminus => format!("C{tag}"),
    }
}

/// The residue name used in Amber libs. E.g. "DA5" in OL24's DNA lib, or "A5" in the RNA one.
fn nucleotide_lib_tag(nt: &NucleotideTemplate, rna: bool) -> String {
    let letter = match nt.nt {
        Nucleotide::A => "A",
        Nucleotide::C => "C",
        Nucleotide::G => "G",
        Nucleotide::T if rna => "U",
        Nucleotide::T => "T",
    };

    let end = match nt.end {
        ResidueEnd::NTerminus => "5",
        ResidueEnd::CTerminus => "3",
        ResidueEnd::Hetero => "N",
        ResidueEnd::Internal => "",
    };

    if rna {
        format!("{letter}{end}")
    } else {
        format!("D{letter}{end}")
    }
}

/// Lib entries for one of Amber's charge sets, sorted by residue.
fn amber_lib_entries(
    set: &AmberSet,
    target: ParamsTarget,
) -> io::Result<Vec<(String, Vec<ChargeParams>)>> {
    let mut result: Vec<(String, Vec<ChargeParams>)> = match target.cat {
        FfCat::PeptideQ => {
            let set = set.peptide_ff_q_map.as_ref().ok_or_else(not_loaded)?;
            prot_map(set, target.prot_map)
                .iter()
                .map(|(aa, charges)| {
                    let charges = charges
                        .iter()
                        .map(|c| ChargeParams {
                            type_in_res: c.type_in_res.to_string(),
                            ff_type: c.ff_type.clone(),
                            charge: c.charge,
                        })
                        .collect();
                    (aa_lib_tag(aa, target.prot_map), charges)
                })
                .collect()
        }
        FfCat::LipidQ => set
            .lipid_ff_q_map
            .as_ref()
            .ok_or_else(not_loaded)?
            .iter()
            .map(|(k, v)| {
                // lipid21's tag for this one is "PH-".
                let tag = match k {
                    LipidStandard::Ph => "PH-".to_owned(),
                    _ => k.to_string().to_uppercase(),
                };
                (tag, v.clone())
            })
            .collect(),
        FfCat::DnaQ | FfCat::RnaQ => {
            let rna = target.cat == FfCat::RnaQ;
            let map = if rna {
                &set.rna_ff_q_map
            } else {
                &set.dna_ff_q_map
            };

            map.as_ref()
                .ok_or_else(not_loaded)?
                .iter()
                .map(|(k, v)| (nucleotide_lib_tag(k, rna), v.clone()))
                .collect()
        }
        _ => return Err(err("Not a charge set")),
    };

    result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

pub fn save(state: &State, target: ParamsTarget, path: &Path) -> io::Result<()> {
    let ext = extension(target, path, false)?;
    let title = format!("{} parameters, written by Molchanica", target.descrip());

    match target.family {
        ForceFieldFamily::Amber => {
            let set = amber_set(state)?;

            if let Some(ff) = amber_ff(set, target.cat) {
                let ff = ff.as_ref().ok_or_else(not_loaded)?;

                return match ext.as_str() {
                    "dat" => ff.save_dat(path, &title),
                    "frcmod" => ff.save_frcmod(path, &title),
                    _ => fs::write(
                        path,
                        format!(
                            "; {title}\n; Converted from Amber. Use with [ defaults ]: \
                            1 2 yes 0.5 0.8333\n\n{}",
                            GromacsFfParams::from_amber(ff).to_itp()
                        ),
                    ),
                };
            }

            save_lib(&amber_lib_entries(set, target)?, path)
        }
        ForceFieldFamily::Opls => {
            let set = opls_set(state)?;

            let tables = match target.cat {
                FfCat::Peptide => &set.tables,
                FfCat::SmallMol => set.small_mol_tables(),
                FfCat::PeptideQ => return set.templates.save(path),
                _ => return Err(not_loaded()),
            };

            fs::write(path, format!("; {title}\n\n{}", tables.to_itp()))
        }
        f => Err(err(format!(
            "Saving {} parameters isn't supported",
            family_name(f)
        ))),
    }
}

/// Tables present in the file replace the current ones; others are kept. This lets a
/// `ffbonded.itp` be loaded on its own, for example.
fn merge_gromacs(dest: &mut GromacsFfParams, src: GromacsFfParams) -> io::Result<()> {
    if src == GromacsFfParams::default() {
        return Err(err("No force field parameters found in this file"));
    }

    let GromacsFfParams {
        atom_types,
        bond_types,
        constraint_types,
        angle_types,
        dihedral_types,
        pair_types,
        defines,
    } = src;

    for (d, s) in [
        (&mut dest.bond_types, bond_types),
        (&mut dest.constraint_types, constraint_types),
        (&mut dest.angle_types, angle_types),
        (&mut dest.dihedral_types, dihedral_types),
        (&mut dest.pair_types, pair_types),
    ] {
        if !s.is_empty() {
            *d = s;
        }
    }

    if !atom_types.is_empty() {
        dest.atom_types = atom_types;
    }
    if !defines.is_empty() {
        dest.defines = defines;
    }

    Ok(())
}

/// Load parameters into the target set. For Amber's bonded sets, a `.dat` replaces them, and a
/// `.frcmod` modifies them.
pub fn load(state: &mut State, target: ParamsTarget, path: &Path) -> io::Result<()> {
    if target.family != state.force_fields.family {
        return Err(err(format!(
            "The force field changed to {} before the file was picked",
            family_name(state.force_fields.family)
        )));
    }

    let ext = extension(target, path, true)?;

    match target.family {
        ForceFieldFamily::Amber => {
            let set = amber_set_mut(state)?;

            if let Some(ff) = amber_ff_mut(set, target.cat) {
                if ext == "dat" {
                    *ff = Some(ForceFieldParams::load_dat(path)?);
                } else {
                    let frcmod = ForceFieldParams::load_frcmod(path)?;
                    *ff = Some(match ff {
                        Some(base) => merge_params(base, &frcmod),
                        None => frcmod,
                    });
                }
                return Ok(());
            }

            let text = fs::read_to_string(path)?;
            match target.cat {
                FfCat::PeptideQ => {
                    let map = parse_lib_peptide(&text)?;
                    let q_set = set.peptide_ff_q_map.get_or_insert_with(Default::default);
                    *prot_map_mut(q_set, target.prot_map) = map;
                }
                FfCat::LipidQ => set.lipid_ff_q_map = Some(parse_lib_lipid(&text)?),
                FfCat::DnaQ => set.dna_ff_q_map = Some(parse_lib_nucleic_acid(&text)?),
                FfCat::RnaQ => set.rna_ff_q_map = Some(parse_lib_nucleic_acid(&text)?),
                _ => return Err(not_loaded()),
            }
        }
        ForceFieldFamily::Opls => {
            let set = opls_set_mut(state)?;

            match target.cat {
                FfCat::Peptide => merge_gromacs(&mut set.tables, GromacsFfParams::load(path)?)?,
                FfCat::SmallMol => {
                    merge_gromacs(set.small_mol_tables_mut(), GromacsFfParams::load(path)?)?
                }
                FfCat::PeptideQ => {
                    let rtp = Rtp::load(path)?;
                    if rtp.residues.is_empty() {
                        return Err(err("No residues found in this file"));
                    }
                    set.templates = rtp;
                }
                _ => return Err(not_loaded()),
            }

            state.force_fields.invalidate();
        }
        f => {
            return Err(err(format!(
                "Loading {} parameters isn't supported",
                family_name(f)
            )));
        }
    }

    Ok(())
}

/// Reset the target set to the parameters bundled with `dynamics`.
pub fn restore(state: &mut State, target: ParamsTarget) -> io::Result<()> {
    match target.family {
        ForceFieldFamily::Amber => {
            let mut fresh = AmberSet::bundled()?;
            let set = amber_set_mut(state)?;

            match target.cat {
                FfCat::Peptide => set.peptide = fresh.peptide.take(),
                FfCat::SmallMol => set.small_mol = fresh.small_mol.take(),
                FfCat::Dna => set.dna = fresh.dna.take(),
                FfCat::Rna => set.rna = fresh.rna.take(),
                FfCat::Lipids => set.lipids = fresh.lipids.take(),
                FfCat::Carbohydrates => set.carbohydrates = fresh.carbohydrates.take(),
                FfCat::PeptideQ => set.peptide_ff_q_map = fresh.peptide_ff_q_map.take(),
                FfCat::LipidQ => set.lipid_ff_q_map = fresh.lipid_ff_q_map.take(),
                FfCat::DnaQ => set.dna_ff_q_map = fresh.dna_ff_q_map.take(),
                FfCat::RnaQ => set.rna_ff_q_map = fresh.rna_ff_q_map.take(),
            }
        }
        ForceFieldFamily::Opls => {
            let mut fresh = OplsSet::bundled()?;
            let set = opls_set_mut(state)?;

            match target.cat {
                FfCat::Peptide => set.tables = std::mem::take(&mut fresh.tables),
                FfCat::SmallMol => {
                    *set.small_mol_tables_mut() = std::mem::take(fresh.small_mol_tables_mut())
                }
                FfCat::PeptideQ => set.templates = std::mem::take(&mut fresh.templates),
                _ => return Err(not_loaded()),
            }

            state.force_fields.invalidate();
        }
        f => {
            return Err(err(format!(
                "Restoring {} parameters isn't supported",
                family_name(f)
            )));
        }
    }

    Ok(())
}

/// The directory of a path, for starting the next dialog there.
pub fn parent_dir(path: &Path) -> Option<PathBuf> {
    path.parent().map(Path::to_path_buf)
}
