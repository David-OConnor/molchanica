//! Structure prediction through third-party models.
//!
//! Predictions are blocking operations and should be moved to a worker thread when called by the
//! GUI.
//!
//! Boltz-2 is special-cased so it "just works" for end users of the standalone application: with the
//! `python_for_structure_prediction` feature enabled, Molchanica provisions a fully isolated Python
//! environment on first use (via `uv`; see the `boltz_runtime` module) and runs Boltz from it — the
//! user never installs Python, `uv`, Torch, or Boltz themselves. Execution defaults to the managed
//! environment's `boltz` launcher as a child process, with an opt-in in-process path through an
//! embedded PyO3 interpreter (see the `pyo3_interface` module).
//!
//! The other predictors (OpenDDE, ESMFold2) still use plain process boundaries against separately
//! installed tooling; a missing model does not prevent Molchanica from starting.

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
#[cfg(feature = "python_for_structure_prediction")]
mod boltz_runtime;
mod esm_fold2;
pub mod open_dde;
#[cfg(feature = "python_for_structure_prediction")]
mod pyo3_interface;

/// Whether the managed, self-provisioned Boltz environment is already installed and ready.
///
/// Cheap: it only checks the filesystem and never provisions or launches a heavy process, so it is
/// safe to call during startup availability probing. Always `false` unless the
/// `python_for_structure_prediction` feature is enabled.
pub fn boltz_runtime_ready() -> bool {
    #[cfg(feature = "python_for_structure_prediction")]
    {
        boltz_runtime::runtime_ready()
    }
    #[cfg(not(feature = "python_for_structure_prediction"))]
    {
        false
    }
}

/// pH used when Molchanica adds hydrogens and force-field parameters to a prediction.
pub const DEFAULT_PREDICTION_PH: f32 = 7.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructurePredictionModel {
    Boltz2,
    // EsmFold2 removed until it has a dedicated application or similar; it currently
    // requires interfacing with Python directly.
    // EsmFold2,
    OpenDDE,
}

/// Predict a structure from an amino-acid sequence and convert its mmCIF output into Molchanica's
/// peptide representation.
///
/// The selected model must be installed separately and available on `PATH`. Boltz and OpenDDE are
/// invoked as `boltz` and `opendde`, respectively. ESMFold2 is invoked through Python.
pub fn predict_structure_from_aas(
    model: StructurePredictionModel,
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    match model {
        StructurePredictionModel::Boltz2 => boltz2::predict_structure_from_aas(aas, ff_map),
        // StructurePredictionModel::EsmFold2 => esm_fold2::predict_structure_from_aas(aas, ff_map),
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
        // StructurePredictionModel::EsmFold2 => esm_fold2::predict_structure_from_dna(nts, ff_map),
        StructurePredictionModel::OpenDDE => open_dde::predict_structure_from_dna(nts, ff_map),
    }
}

/// Predict a DNA structure from a nucleotide sequence.
///
/// This is the sequence-oriented counterpart to [`predict_structure_from_aas`].
pub fn predict_structure_from_nts(
    model: StructurePredictionModel,
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
) -> io::Result<MoleculePeptide> {
    predict_structure_from_dna(model, nts, ff_map)
}

pub fn amino_acid_sequence(aas: &[AminoAcid]) -> io::Result<String> {
    if aas.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "amino-acid sequence cannot be empty",
        ));
    }

    Ok(aas.iter().map(|aa| aa.to_str(AaIdent::OneLetter)).collect())
}

pub fn dna_sequence(nts: &[Nucleotide]) -> io::Result<String> {
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

pub fn run_model_command(command: &mut Command, model: &str) -> io::Result<()> {
    let output = command.output().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("unable to start {model}; install it separately and put it on PATH: {error}"),
        )
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("{model} input command: {command:?}");
    println!("{model} stdout:\n{stdout}");
    eprintln!("{model} stderr:\n{stderr}");

    if output.status.success() {
        return Ok(());
    }

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

pub fn load_prediction(
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
