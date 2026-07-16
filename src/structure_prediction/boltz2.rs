//! Boltz-2 structure prediction.
//!
//! With the `python_for_structure_prediction` feature enabled, Boltz "just works": Molchanica
//! provisions a fully isolated Python environment on first use (see [`super::boltz_runtime`]) and
//! runs Boltz from it, so the user never installs Python, `uv`, Torch, or Boltz themselves. By
//! default the managed environment's `boltz` launcher is run as a child process; an opt-in
//! in-process path via the embedded PyO3 interpreter is available with `MOLCHANICA_BOLTZ_INPROCESS=1`
//! (see [`super::pyo3_interface`]).
//!
//! Without that feature, this falls back to a `boltz` executable the user has installed and put on
//! `PATH`, invoked as `boltz predict ...`.
//!
//! Protein inputs use Boltz's public MSA server; DNA inputs do not require an MSA.

#[cfg(not(feature = "python_for_structure_prediction"))]
use std::process::Command;
use std::{fs, io};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};

#[cfg(not(feature = "python_for_structure_prediction"))]
use crate::structure_prediction::run_model_command;
use crate::{
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionWorkspace, amino_acid_sequence, dna_sequence, load_prediction,
    },
};

pub(super) fn predict_structure_from_aas(
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let sequence = amino_acid_sequence(aas)?;
    predict(&boltz_yaml("protein", &sequence), true, ff_map)
}

pub(super) fn predict_structure_from_dna(
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let sequence = dna_sequence(nts)?;
    predict(&boltz_yaml("dna", &sequence), false, ff_map)
}

fn predict(
    input_yaml: &str,
    use_msa_server: bool,
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let workspace = PredictionWorkspace::new("boltz2")?;
    let input_path = workspace.path("molchanica.yaml");
    let output_path = workspace.create_dir("output")?;
    fs::write(&input_path, input_yaml)?;

    run_boltz(&input_path, &output_path, use_msa_server)?;

    load_prediction(&output_path, ff_map)
}

/// Managed path: provision (if needed) and run the isolated Boltz environment.
#[cfg(feature = "python_for_structure_prediction")]
fn run_boltz(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    use_msa_server: bool,
) -> io::Result<()> {
    use crate::structure_prediction::{boltz_runtime, pyo3_interface};

    let runtime = boltz_runtime::ensure()?;

    if boltz_runtime::in_process_requested() {
        match pyo3_interface::predict(&runtime, input_path, output_path, use_msa_server) {
            Ok(()) => return Ok(()),
            Err(error) => {
                eprintln!(
                    "Boltz in-process execution failed ({error}); falling back to the managed \
                     subprocess."
                );
            }
        }
    }

    runtime.predict(input_path, output_path, use_msa_server)
}

/// Legacy path: a `boltz` executable the user installed and put on `PATH`.
#[cfg(not(feature = "python_for_structure_prediction"))]
fn run_boltz(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    use_msa_server: bool,
) -> io::Result<()> {
    let mut command = Command::new("boltz");
    command
        .arg("predict")
        .arg(input_path)
        .arg("--out_dir")
        .arg(output_path);
    if use_msa_server {
        command.arg("--use_msa_server");
    }
    run_model_command(&mut command, "Boltz-2")
}

fn boltz_yaml(entity_type: &str, sequence: &str) -> String {
    format!("version: 1\nsequences:\n  - {entity_type}:\n      id: A\n      sequence: {sequence}\n")
}
