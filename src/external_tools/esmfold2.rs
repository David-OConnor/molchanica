use std::{fs, io, process::Command};

use dynamics::params::ProtFfChargeMapSet;
use mol_defs::molecules::peptide::MoleculePeptide;

use crate::{
    external_tools::{
        self, Tool,
        opendde::{OpenDdeEntity, OpenDdeRequest},
    },
    structure_prediction::{
        PredictionControl, PredictionWorkspace, StructurePredictionModel, load_prediction,
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
        StructurePredictionModel::EsmFold2 => predict_esmfold2(request, ff_map, control),
        StructurePredictionModel::OpenDDE | StructurePredictionModel::Boltz2 => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "model is not a Linux-only adapter",
            ))
        }
    }
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

fn unsupported(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message)
}
