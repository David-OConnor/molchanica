//! OpenDDE all-atom co-folding through the official `opendde pred` CLI.
//!
//! OpenDDE is a preview release, so its process boundary is kept isolated here. The module supports
//! proteins, DNA, RNA, ligands (SMILES, `CCD_...`, or `FILE_...`), ions, and explicit covalent
//! links using the documented AlphaFold-Server-style JSON schema.

use std::{collections::HashSet, fs, io, process::Command};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};
use serde_json::{Value, json};

use crate::{
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionWorkspace, amino_acid_sequence, dna_sequence, executable, load_prediction,
        run_model_command,
    },
};

/// One entity in an OpenDDE co-folding request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenDdeEntity {
    Protein {
        id: String,
        sequence: String,
    },
    Dna {
        id: String,
        sequence: String,
    },
    Rna {
        id: String,
        sequence: String,
    },
    /// The value may be a SMILES string, a `CCD_...` code, or a `FILE_...` reference.
    Ligand {
        id: String,
        value: String,
    },
    /// OpenDDE expects a CCD component name without the `CCD_` prefix, such as `MG`.
    Ion {
        id: String,
        code: String,
    },
}

impl OpenDdeEntity {
    pub fn protein(id: impl Into<String>, aas: &[AminoAcid]) -> io::Result<Self> {
        Ok(Self::Protein {
            id: id.into(),
            sequence: amino_acid_sequence(aas)?,
        })
    }

    pub fn dna(id: impl Into<String>, nts: &[Nucleotide]) -> io::Result<Self> {
        Ok(Self::Dna {
            id: id.into(),
            sequence: dna_sequence(nts)?,
        })
    }

    pub fn rna(id: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self::Rna {
            id: id.into(),
            sequence: sequence.into(),
        }
    }

    pub fn ligand(id: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Ligand {
            id: id.into(),
            value: value.into(),
        }
    }

    pub fn ion(id: impl Into<String>, code: impl Into<String>) -> Self {
        Self::Ion {
            id: id.into(),
            code: code.into(),
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Protein { id, .. }
            | Self::Dna { id, .. }
            | Self::Rna { id, .. }
            | Self::Ligand { id, .. }
            | Self::Ion { id, .. } => id,
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::Protein { id, sequence } => json!({
                "proteinChain": {
                    "sequence": sequence.to_ascii_uppercase(), "count": 1, "id": [id]
                }
            }),
            Self::Dna { id, sequence } => json!({
                "dnaSequence": {
                    "sequence": sequence.to_ascii_uppercase(), "count": 1, "id": [id]
                }
            }),
            Self::Rna { id, sequence } => json!({
                "rnaSequence": {
                    "sequence": sequence.to_ascii_uppercase(), "count": 1, "id": [id]
                }
            }),
            Self::Ligand { id, value } => json!({
                "ligand": { "ligand": value, "count": 1, "id": [id] }
            }),
            Self::Ion { id, code } => json!({
                "ion": { "ion": code.to_ascii_uppercase(), "count": 1, "id": [id] }
            }),
        }
    }
}

/// An explicit covalent link between two 1-based entity positions in an OpenDDE request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenDdeCovalentBond {
    pub entity1: usize,
    pub copy1: usize,
    pub position1: usize,
    pub atom1: String,
    pub entity2: usize,
    pub copy2: usize,
    pub position2: usize,
    pub atom2: String,
}

impl OpenDdeCovalentBond {
    fn to_json(&self) -> Value {
        json!({
            "entity1": self.entity1.to_string(),
            "copy1": self.copy1,
            "position1": self.position1.to_string(),
            "atom1": self.atom1,
            "entity2": self.entity2.to_string(),
            "copy2": self.copy2,
            "position2": self.position2.to_string(),
            "atom2": self.atom2,
        })
    }
}

/// A complete OpenDDE co-folding job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenDdeRequest {
    pub name: String,
    pub seed: u64,
    pub entities: Vec<OpenDdeEntity>,
    pub covalent_bonds: Vec<OpenDdeCovalentBond>,
}

impl OpenDdeRequest {
    pub fn new(name: impl Into<String>, entities: Vec<OpenDdeEntity>) -> Self {
        Self {
            name: name.into(),
            seed: 101,
            entities,
            covalent_bonds: Vec::new(),
        }
    }

    fn validate(&self) -> io::Result<()> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "OpenDDE job name must contain only ASCII letters, digits, '_' or '-'",
            ));
        }
        if self.entities.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "OpenDDE request must contain at least one entity",
            ));
        }

        let mut ids = HashSet::new();
        for entity in &self.entities {
            let id = entity.id();
            if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid OpenDDE chain ID '{id}'"),
                ));
            }
            if !ids.insert(id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("duplicate OpenDDE chain ID '{id}'"),
                ));
            }

            let (kind, value, allowed): (&str, &str, Option<&[u8]>) = match entity {
                OpenDdeEntity::Protein { sequence, .. } => {
                    ("protein", sequence, Some(b"ACDEFGHIKLMNPQRSTVWYX"))
                }
                OpenDdeEntity::Dna { sequence, .. } => ("DNA", sequence, Some(b"ATGCNX")),
                OpenDdeEntity::Rna { sequence, .. } => ("RNA", sequence, Some(b"AUGCNX")),
                OpenDdeEntity::Ligand { value, .. } => ("ligand", value, None),
                OpenDdeEntity::Ion { code, .. } => ("ion", code, None),
            };
            if value.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("OpenDDE {kind} value cannot be empty"),
                ));
            }
            if let Some(allowed) = allowed
                && !value
                    .bytes()
                    .all(|byte| allowed.contains(&byte.to_ascii_uppercase()))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("OpenDDE {kind} sequence contains an unsupported residue"),
                ));
            }
            if matches!(entity, OpenDdeEntity::Ion { .. })
                && (!value.bytes().all(|byte| byte.is_ascii_alphanumeric())
                    || value.to_ascii_uppercase().starts_with("CCD"))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "OpenDDE ion must be an unprefixed CCD component name such as 'MG'",
                ));
            }
        }

        for bond in &self.covalent_bonds {
            if bond.entity1 == 0
                || bond.entity1 > self.entities.len()
                || bond.entity2 == 0
                || bond.entity2 > self.entities.len()
                || bond.copy1 == 0
                || bond.copy2 == 0
                || bond.position1 == 0
                || bond.position2 == 0
                || bond.atom1.is_empty()
                || bond.atom2.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "OpenDDE covalent-bond references are 1-based and atoms cannot be empty",
                ));
            }
        }

        Ok(())
    }

    fn to_json(&self) -> io::Result<Value> {
        self.validate()?;
        let mut job = json!({
            "name": self.name,
            "modelSeeds": [self.seed],
            "sequences": self.entities.iter().map(OpenDdeEntity::to_json).collect::<Vec<_>>(),
        });
        if !self.covalent_bonds.is_empty() {
            job["covalent_bonds"] = Value::Array(
                self.covalent_bonds
                    .iter()
                    .map(OpenDdeCovalentBond::to_json)
                    .collect(),
            );
        }
        Ok(Value::Array(vec![job]))
    }
}

/// Run a general OpenDDE protein/nucleic-acid/ligand/ion co-folding request.
pub fn predict_structure(
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let workspace = PredictionWorkspace::new("opendde")?;
    let input_path = workspace.path("input.json");
    let output_path = workspace.create_dir("output")?;
    let input = serde_json::to_vec_pretty(&request.to_json()?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    fs::write(&input_path, input)?;

    let mut command = Command::new(executable("MOLCHANICA_OPENDDE", "opendde"));
    command
        .arg("pred")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("-n")
        .arg("opendde_v1");
    run_model_command(&mut command, "OpenDDE")?;

    load_prediction(&output_path, ff_map)
}

pub(super) fn predict_structure_from_aas(
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let entity = OpenDdeEntity::protein("A", aas)?;
    predict_structure(&OpenDdeRequest::new("molchanica", vec![entity]), ff_map)
}

pub(super) fn predict_structure_from_dna(
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let entity = OpenDdeEntity::dna("D", nts)?;
    predict_structure(&OpenDdeRequest::new("molchanica", vec![entity]), ff_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_every_supported_entity() {
        let request = OpenDdeRequest::new(
            "all_entities",
            vec![
                OpenDdeEntity::Protein {
                    id: "A".into(),
                    sequence: "ACDE".into(),
                },
                OpenDdeEntity::Dna {
                    id: "D".into(),
                    sequence: "GATTACA".into(),
                },
                OpenDdeEntity::rna("R", "GUAC"),
                OpenDdeEntity::ligand("L", "CCD_ATP"),
                OpenDdeEntity::ion("M", "MG"),
            ],
        );
        let json = request.to_json().unwrap();
        let sequences = json[0]["sequences"].as_array().unwrap();
        assert!(sequences[0].get("proteinChain").is_some());
        assert!(sequences[1].get("dnaSequence").is_some());
        assert!(sequences[2].get("rnaSequence").is_some());
        assert!(sequences[3].get("ligand").is_some());
        assert!(sequences[4].get("ion").is_some());
    }

    #[test]
    fn rejects_path_like_job_names_and_bad_rna() {
        let request = OpenDdeRequest::new("../escape", vec![OpenDdeEntity::rna("R", "GUT")]);
        assert_eq!(
            request.validate().unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );

        let request = OpenDdeRequest::new("safe", vec![OpenDdeEntity::rna("R", "GUT")]);
        assert_eq!(
            request.validate().unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
