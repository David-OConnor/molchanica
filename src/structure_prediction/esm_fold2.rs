//! ESMFold2 structure prediction through its official Python API.
//!
//! ESMFold2 does not currently publish a dedicated CLI. Molchanica therefore writes a small Python
//! runner into a temporary workspace and invokes the separately installed `esm`/`transformers`
//! stack with the `python` (Windows) or `python3` (otherwise) found on `PATH`, so that stack must be
//! installed in whichever environment is active.
//!
//! We will focus on tools other than this for now, due to it requiring python. This is not
//! currently an available pipeline.

use std::{fs, io, process::Command};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};
use serde_json::json;

use crate::structure_prediction::PredictionControl;
use crate::{
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionWorkspace, amino_acid_sequence, dna_sequence, load_prediction, run_model_command,
    },
};

const PYTHON_RUNNER: &str = r#"import json
import sys

import torch
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

with open(sys.argv[1], encoding="utf-8") as input_file:
    payload = json.load(input_file)

if payload["kind"] == "protein":
    entity = ProteinInput(id="A", sequence=payload["sequence"])
elif payload["kind"] == "dna":
    entity = DNAInput(id="A", sequence=payload["sequence"])
else:
    raise ValueError(f"Unsupported ESMFold2 entity kind: {payload['kind']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(device).eval()
prediction_input = StructurePredictionInput(sequences=[entity])
result = ESMFold2InputBuilder().fold(
    model,
    prediction_input,
    num_loops=20,
    num_sampling_steps=100,
    num_diffusion_samples=1,
    seed=0,
)

with open(sys.argv[2], "w", encoding="utf-8") as output_file:
    output_file.write(result.complex.to_mmcif())
"#;

pub(super) fn predict_structure_from_aas(
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    predict("protein", &amino_acid_sequence(aas)?, ff_map)
}

pub(super) fn predict_structure_from_dna(
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    predict("dna", &dna_sequence(nts)?, ff_map)
}

fn predict(kind: &str, sequence: &str, ff_map: &ProtFfChargeMapSet) -> io::Result<MoleculePeptide> {
    let workspace = PredictionWorkspace::new("esmfold2")?;
    let runner_path = workspace.path("run_esmfold2.py");
    let input_path = workspace.path("input.json");
    let output_dir = workspace.create_dir("output")?;
    let output_path = output_dir.join("prediction.cif");

    fs::write(&runner_path, PYTHON_RUNNER)?;
    let input = serde_json::to_vec_pretty(&json!({
        "kind": kind,
        "sequence": sequence,
    }))
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    fs::write(&input_path, input)?;

    let python = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };
    let mut command = Command::new(python);
    command.arg(&runner_path).arg(&input_path).arg(&output_path);
    run_model_command(&mut command, "ESMFold2", &PredictionControl::default())?;

    load_prediction(&output_dir, ff_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_covers_both_supported_inputs() {
        assert!(PYTHON_RUNNER.contains("ProteinInput"));
        assert!(PYTHON_RUNNER.contains("DNAInput"));
        assert!(PYTHON_RUNNER.contains("to_mmcif"));
    }
}
