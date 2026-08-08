//! Linux-only structure prediction adapters backed by bio_tools-managed environments.

use std::{fs, io, process::Command};

use dynamics::params::ProtFfChargeMapSet;
use serde_json::{Value, json};

use crate::{
    external_tools::{self, Tool},
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionControl, PredictionWorkspace, StructurePredictionModel, load_prediction,
        opendde::{OpenDdeEntity, OpenDdeRequest},
        run_model_command,
    },
};

const PDB_TO_MMCIF: &str = r#"import pathlib
import sys

from Bio.PDB import MMCIFIO, PDBParser

source_dir = pathlib.Path(sys.argv[1])
candidates = sorted(source_dir.rglob("*.pdb"))
if not candidates:
    raise FileNotFoundError(f"No PDB prediction found under {source_dir}")

structure = PDBParser(QUIET=True).get_structure("prediction", candidates[0])
writer = MMCIFIO()
writer.set_structure(structure)
writer.save(sys.argv[2])
"#;

pub(super) fn predict_structure(
    model: StructurePredictionModel,
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let tool = model.tool();
    if !tool.spec().platform.is_supported() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{} is Linux-only and cannot run on this operating system",
                model.label()
            ),
        ));
    }

    match model {
        StructurePredictionModel::Chai1 => predict_chai1(request, ff_map, control),
        StructurePredictionModel::AlphaFold3 => predict_alphafold3(request, ff_map, control),
        StructurePredictionModel::EsmFold2 => predict_esmfold2(request, ff_map, control),
        StructurePredictionModel::OpenDDE | StructurePredictionModel::Boltz2 => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "model is not a Linux-only adapter",
            ))
        }
    }
}

fn predict_chai1(
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let workspace = PredictionWorkspace::new("chai1")?;
    let fasta_path = workspace.path("input.fasta");
    let output_dir = workspace.create_dir("output")?;

    fs::write(&fasta_path, chai_fasta(request)?)?;
    let executable = external_tools::find_executable(Tool::Chai1)?;
    let mut command = Command::new(executable);
    command.arg("fold").arg(&fasta_path).arg(&output_dir);
    run_model_command(&mut command, "Chai-1", control)?;
    load_prediction(&output_dir, ff_map)
}

fn chai_fasta(request: &OpenDdeRequest) -> io::Result<String> {
    request.validate()?;
    if !request.covalent_bonds.is_empty() {
        return Err(unsupported(
            "Chai-1 covalent restraints are not yet represented by this popup",
        ));
    }

    let mut fasta = String::new();
    for entity in &request.entities {
        let (kind, id, value) = match entity {
            OpenDdeEntity::Protein { id, sequence } => ("protein", id, sequence),
            OpenDdeEntity::Dna { id, sequence } => ("dna", id, sequence),
            OpenDdeEntity::Rna { id, sequence } => ("rna", id, sequence),
            OpenDdeEntity::Ligand { id, value }
                if !value.starts_with("CCD_") && !value.starts_with("FILE_") =>
            {
                ("ligand", id, value)
            }
            OpenDdeEntity::Ligand { .. } => {
                return Err(unsupported(
                    "Chai-1 accepts ligand SMILES here, but not CCD_ or FILE_ ligand references",
                ));
            }
            OpenDdeEntity::Ion { .. } => {
                return Err(unsupported(
                    "Chai-1 ion input is not available through this popup",
                ));
            }
        };
        fasta.push('>');
        fasta.push_str(kind);
        fasta.push_str("|name=");
        fasta.push_str(id);
        fasta.push('\n');
        fasta.push_str(value);
        fasta.push('\n');
    }
    Ok(fasta)
}

fn predict_alphafold3(
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let workspace = PredictionWorkspace::new("alphafold3")?;
    let input_path = workspace.path("input.json");
    let output_dir = workspace.create_dir("output")?;

    let input = serde_json::to_vec_pretty(&alphafold_input(request)?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    fs::write(&input_path, input)?;

    let python = external_tools::find_executable(Tool::AlphaFold3)?;
    let root = external_tools::bundle_root(Tool::AlphaFold3)?;
    let script = root.join("run_alphafold.py");
    let mut command = Command::new(python);
    command
        .arg(script)
        .arg(format!("--json_path={}", input_path.display()))
        .arg(format!("--output_dir={}", output_dir.display()))
        .current_dir(root);
    run_model_command(&mut command, "AlphaFold 3", control)?;
    load_prediction(&output_dir, ff_map)
}

fn alphafold_input(request: &OpenDdeRequest) -> io::Result<Value> {
    request.validate()?;
    let sequences = request
        .entities
        .iter()
        .map(|entity| match entity {
            OpenDdeEntity::Protein { id, sequence } => Ok(json!({
                "protein": {"id": id, "sequence": sequence.to_ascii_uppercase()}
            })),
            OpenDdeEntity::Dna { id, sequence } => Ok(json!({
                "dna": {"id": id, "sequence": sequence.to_ascii_uppercase()}
            })),
            OpenDdeEntity::Rna { id, sequence } => Ok(json!({
                "rna": {"id": id, "sequence": sequence.to_ascii_uppercase()}
            })),
            OpenDdeEntity::Ligand { id, value } => {
                if let Some(codes) = value.strip_prefix("CCD_") {
                    let codes = codes
                        .split('_')
                        .filter(|code| !code.is_empty())
                        .collect::<Vec<_>>();
                    if codes.is_empty() {
                        return Err(unsupported("AlphaFold 3 received an empty CCD ligand code"));
                    }
                    Ok(json!({"ligand": {"id": id, "ccdCodes": codes}}))
                } else if value.starts_with("FILE_") {
                    Err(unsupported(
                        "AlphaFold 3 FILE_ ligands require a user CCD and are not available here",
                    ))
                } else {
                    Ok(json!({"ligand": {"id": id, "smiles": value}}))
                }
            }
            OpenDdeEntity::Ion { id, code } => Ok(json!({
                "ligand": {"id": id, "ccdCodes": [code.to_ascii_uppercase()]}
            })),
        })
        .collect::<io::Result<Vec<_>>>()?;

    let bonds = request
        .covalent_bonds
        .iter()
        .map(|bond| {
            let first = request.entities.get(bond.entity1 - 1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "invalid first bonded entity")
            })?;
            let second = request.entities.get(bond.entity2 - 1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "invalid second bonded entity")
            })?;
            Ok(json!([
                [entity_id(first), bond.position1, bond.atom1],
                [entity_id(second), bond.position2, bond.atom2]
            ]))
        })
        .collect::<io::Result<Vec<_>>>()?;

    let mut input = json!({
        "name": request.name,
        "modelSeeds": [request.seed],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4
    });
    if !bonds.is_empty() {
        input["bondedAtomPairs"] = Value::Array(bonds);
    }
    Ok(input)
}

fn predict_esmfold2(
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let workspace = PredictionWorkspace::new("esmfold2")?;
    let fasta_path = workspace.path("input.fasta");
    let converter_path = workspace.path("pdb_to_mmcif.py");
    let output_dir = workspace.create_dir("output")?;
    let cif_path = output_dir.join("prediction.cif");

    let sequence = esmfold_sequence(request)?;
    fs::write(&fasta_path, format!(">{}\n{}\n", request.name, sequence))?;

    let executable = external_tools::find_executable(Tool::EsmFold2)?;
    let mut command = Command::new(executable);
    command
        .arg("-i")
        .arg(&fasta_path)
        .arg("-o")
        .arg(&output_dir);
    run_model_command(&mut command, "ESMFold 2", control)?;

    control.check_cancelled()?;
    fs::write(&converter_path, PDB_TO_MMCIF)?;
    let python = external_tools::uv_managed_python("esmfold2", "MOLCHANICA_ESMFOLD2_PYTHON")?;
    let mut convert = Command::new(python);
    convert.arg(&converter_path).arg(&output_dir).arg(&cif_path);
    run_model_command(&mut convert, "ESMFold 2 output conversion", control)?;
    load_prediction(&output_dir, ff_map)
}

fn esmfold_sequence(request: &OpenDdeRequest) -> io::Result<String> {
    request.validate()?;
    if !request.covalent_bonds.is_empty() {
        return Err(unsupported(
            "ESMFold 2 does not support covalent-bond inputs",
        ));
    }

    request
        .entities
        .iter()
        .map(|entity| match entity {
            OpenDdeEntity::Protein { sequence, .. } => Ok(sequence.as_str()),
            _ => Err(unsupported(
                "ESMFold 2 supports protein chains only; remove DNA, RNA, ligands, and ions",
            )),
        })
        .collect::<io::Result<Vec<_>>>()
        .map(|sequences| sequences.join(":"))
}

fn entity_id(entity: &OpenDdeEntity) -> &str {
    match entity {
        OpenDdeEntity::Protein { id, .. }
        | OpenDdeEntity::Dna { id, .. }
        | OpenDdeEntity::Rna { id, .. }
        | OpenDdeEntity::Ligand { id, .. }
        | OpenDdeEntity::Ion { id, .. } => id,
    }
}

fn unsupported(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure_prediction::opendde::OpenDdeCovalentBond;

    fn mixed_request() -> OpenDdeRequest {
        let mut request = OpenDdeRequest::new(
            "mixed",
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDE"),
                OpenDdeEntity::dna_sequence("B", "ATGC"),
                OpenDdeEntity::ligand("L", "CCD_ATP"),
                OpenDdeEntity::ion("M", "MG"),
            ],
        );
        request.covalent_bonds.push(OpenDdeCovalentBond {
            entity1: 1,
            copy1: 1,
            position1: 2,
            atom1: "SG".to_owned(),
            entity2: 3,
            copy2: 1,
            position2: 1,
            atom2: "C1".to_owned(),
        });
        request
    }

    #[test]
    fn alphafold_json_maps_entities_and_bonds() {
        let input = alphafold_input(&mixed_request()).expect("input should render");
        assert_eq!(input["sequences"][0]["protein"]["id"], "A");
        assert_eq!(input["sequences"][2]["ligand"]["ccdCodes"][0], "ATP");
        assert_eq!(input["sequences"][3]["ligand"]["ccdCodes"][0], "MG");
        assert_eq!(input["bondedAtomPairs"][0][0][0], "A");
        assert_eq!(input["bondedAtomPairs"][0][1][0], "L");
    }

    #[test]
    fn chai_fasta_uses_typed_headers() {
        let request = OpenDdeRequest::new(
            "chai",
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDE"),
                OpenDdeEntity::rna("R", "AUGC"),
                OpenDdeEntity::ligand("L", "CCO"),
            ],
        );
        let fasta = chai_fasta(&request).expect("input should render");
        assert!(fasta.contains(">protein|name=A\nACDE"));
        assert!(fasta.contains(">rna|name=R\nAUGC"));
        assert!(fasta.contains(">ligand|name=L\nCCO"));
    }

    #[test]
    fn esmfold_joins_protein_chains_and_rejects_other_entities() {
        let proteins = OpenDdeRequest::new(
            "esm",
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDE"),
                OpenDdeEntity::protein_sequence("B", "FGHI"),
            ],
        );
        assert_eq!(esmfold_sequence(&proteins).unwrap(), "ACDE:FGHI");
        assert!(esmfold_sequence(&mixed_request()).is_err());
    }
}
