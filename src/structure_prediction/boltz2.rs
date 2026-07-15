//! Boltz-2 structure prediction through the official `boltz predict` CLI.
//!
//! Boltz is intentionally not a Molchanica dependency. Install it independently (for example,
//! `pip install -U "boltz[cuda]"`) and make `boltz` available on `PATH`, or point
//!
//! BOltz (CAO 2026-07-15) requires Numpy < 2.0, which requires Python 3.11 or 3.12; use UV.
//!
//! `MOLCHANICA_BOLTZ` at the executable. Protein inputs use Boltz's public MSA server; DNA inputs
//! do not require an MSA.
//!

use std::{fs, io, process::Command};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};

use crate::{
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionWorkspace, amino_acid_sequence, dna_sequence, executable, load_prediction,
        run_model_command,
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

    let mut command = Command::new(executable("MOLCHANICA_BOLTZ", "boltz"));
    command
        .arg("predict")
        .arg(&input_path)
        .arg("--out_dir")
        .arg(&output_path);
    if use_msa_server {
        command.arg("--use_msa_server");
    }
    run_model_command(&mut command, "Boltz-2")?;

    load_prediction(&output_path, ff_map)
}

fn boltz_yaml(entity_type: &str, sequence: &str) -> String {
    format!("version: 1\nsequences:\n  - {entity_type}:\n      id: A\n      sequence: {sequence}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_boltz_protein_yaml() {
        let yaml = boltz_yaml("protein", "MKT");
        assert!(yaml.contains("- protein:"));
        assert!(yaml.contains("sequence: MKT"));
    }

    #[test]
    fn creates_boltz_dna_yaml() {
        let yaml = boltz_yaml("dna", "GATTACA");
        assert!(yaml.contains("- dna:"));
        assert!(yaml.contains("sequence: GATTACA"));
    }
}
