//! Structure prediction through separately installed third-party models.
//!
//! The predictors deliberately use process boundaries: installing any of these large Python
//! stacks is optional, and a missing model does not prevent Molchanica itself from starting.
//! Predictions are blocking operations and should be moved to a worker thread when called by the
//! GUI.

use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use bio_files::MmCif;
use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AaIdent, AminoAcid, Nucleotide};

use crate::molecules::peptide::MoleculePeptide;

mod boltz2;
mod esm_fold2;
pub mod open_dde;

/// pH used when Molchanica adds hydrogens and force-field parameters to a prediction.
pub const DEFAULT_PREDICTION_PH: f32 = 7.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructurePredictionModel {
    Boltz2,
    EsmFold2,
    OpenDDE,
}

/// Predict a structure from an amino-acid sequence and convert its mmCIF output into Molchanica's
/// peptide representation.
///
/// The selected model must be installed separately. Boltz and OpenDDE must be available as
/// `boltz` and `opendde`, respectively. ESMFold2 is invoked through Python. These executable names
/// can be overridden with `MOLCHANICA_BOLTZ`, `MOLCHANICA_OPENDDE`, and `MOLCHANICA_PYTHON`.
pub fn predict_structure_from_aas(
    model: StructurePredictionModel,
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    match model {
        StructurePredictionModel::Boltz2 => boltz2::predict_structure_from_aas(aas, ff_map),
        StructurePredictionModel::EsmFold2 => esm_fold2::predict_structure_from_aas(aas, ff_map),
        StructurePredictionModel::OpenDDE => open_dde::predict_structure_from_aas(aas, ff_map),
    }
}

/// Predict a DNA structure and load the resulting all-atom mmCIF into `MoleculePeptide`.
///
/// `MoleculePeptide` is currently Molchanica's mmCIF-backed macromolecule container, so it is also
/// used here for nucleic-acid predictions despite its historical name.
pub fn predict_structure_from_dna(
    model: StructurePredictionModel,
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    match model {
        StructurePredictionModel::Boltz2 => boltz2::predict_structure_from_dna(nts, ff_map),
        StructurePredictionModel::EsmFold2 => esm_fold2::predict_structure_from_dna(nts, ff_map),
        StructurePredictionModel::OpenDDE => open_dde::predict_structure_from_dna(nts, ff_map),
    }
}

pub(super) fn amino_acid_sequence(aas: &[AminoAcid]) -> io::Result<String> {
    if aas.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "amino-acid sequence cannot be empty",
        ));
    }

    Ok(aas.iter().map(|aa| aa.to_str(AaIdent::OneLetter)).collect())
}

pub(super) fn dna_sequence(nts: &[Nucleotide]) -> io::Result<String> {
    if nts.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "DNA sequence cannot be empty",
        ));
    }

    Ok(nts
        .iter()
        .map(Nucleotide::to_u8_upper)
        .map(char::from)
        .collect())
}

pub(super) fn executable(environment_variable: &str, default: &str) -> PathBuf {
    env::var_os(environment_variable)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

pub(super) fn run_model_command(command: &mut Command, model: &str) -> io::Result<()> {
    let output = command.output().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "unable to start {model}; install it separately or configure its Molchanica executable environment variable: {error}"
            ),
        )
    })?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(io::Error::other(format!(
        "{model} exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        truncate_process_output(&stdout),
        truncate_process_output(&stderr),
    )))
}

fn truncate_process_output(output: &str) -> &str {
    const MAX_BYTES: usize = 16 * 1024;
    if output.len() <= MAX_BYTES {
        return output;
    }

    let mut boundary = MAX_BYTES;
    while !output.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &output[..boundary]
}

pub(super) fn load_prediction(
    output_dir: &Path,
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    let cif_path = find_prediction_cif(output_dir)?;
    let cif_text = fs::read_to_string(&cif_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("unable to read prediction {}: {error}", cif_path.display()),
        )
    })?;
    let cif = MmCif::new(&cif_text).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "model output {} is not valid mmCIF: {error}",
                cif_path.display()
            ),
        )
    })?;

    MoleculePeptide::from_mmcif(cif, ff_map, None, DEFAULT_PREDICTION_PH)
}

fn find_prediction_cif(output_dir: &Path) -> io::Result<PathBuf> {
    let mut candidates = Vec::new();
    collect_cif_files(output_dir, &mut candidates)?;
    candidates.sort_by(|left, right| {
        cif_priority(left)
            .cmp(&cif_priority(right))
            .then_with(|| left.cmp(right))
    });
    candidates.into_iter().next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "prediction command completed but produced no mmCIF file under {}",
                output_dir.display()
            ),
        )
    })
}

fn collect_cif_files(directory: &Path, candidates: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_cif_files(&path, candidates)?;
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("cif"))
        {
            candidates.push(path);
        }
    }
    Ok(())
}

fn cif_priority(path: &Path) -> u8 {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.contains("model_0") || name.contains("sample_0") {
        0
    } else {
        1
    }
}

static WORKSPACE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) struct PredictionWorkspace {
    root: PathBuf,
}

impl PredictionWorkspace {
    pub fn new(model: &str) -> io::Result<Self> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        for _ in 0..100 {
            let counter = WORKSPACE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = env::temp_dir().join(format!(
                "molchanica-{model}-{}-{timestamp}-{counter}",
                std::process::id()
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Ok(Self { root }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "unable to allocate a unique prediction workspace",
        ))
    }

    pub fn path(&self, relative_path: impl AsRef<Path>) -> PathBuf {
        self.root.join(relative_path)
    }

    pub fn create_dir(&self, relative_path: impl AsRef<Path>) -> io::Result<PathBuf> {
        let path = self.path(relative_path);
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}

impl Drop for PredictionWorkspace {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root)
            && error.kind() != io::ErrorKind::NotFound
        {
            eprintln!(
                "Unable to remove structure-prediction workspace {}: {error}",
                self.root.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_sequences() {
        assert_eq!(
            amino_acid_sequence(&[]).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            dna_sequence(&[]).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn serializes_native_sequence_types() {
        assert_eq!(
            amino_acid_sequence(&[AminoAcid::Met, AminoAcid::Gly]).unwrap(),
            "MG"
        );
        assert_eq!(
            dna_sequence(&[Nucleotide::A, Nucleotide::T, Nucleotide::G]).unwrap(),
            "ATG"
        );
    }
}
