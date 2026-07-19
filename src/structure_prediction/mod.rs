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
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bio_files::MmCif;
use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AaIdent, AminoAcid, Nucleotide};

use crate::molecules::peptide::MoleculePeptide;

mod boltz2;
#[cfg(feature = "python_for_structure_prediction")]
mod boltz_runtime;
mod esm_fold2;
pub mod opendde;
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

/// A cloneable cancellation signal shared by the UI and a prediction worker.
#[derive(Clone, Debug, Default)]
pub struct PredictionControl {
    cancel_requested: Arc<AtomicBool>,
}

impl PredictionControl {
    pub fn cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    fn check_cancelled(&self) -> io::Result<()> {
        if self.is_cancel_requested() {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "structure prediction was cancelled",
            ))
        } else {
            Ok(())
        }
    }
}

/// Result sent from a structure-prediction worker to the UI thread.
pub enum StructurePredictionOutcome {
    Complete(MoleculePeptide),
    Cancelled,
    Failed(String),
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
    predict_structure_from_aas_with_control(model, aas, ff_map, &PredictionControl::default())
}

pub(crate) fn predict_structure_from_aas_with_control(
    model: StructurePredictionModel,
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    control.check_cancelled()?;
    match model {
        StructurePredictionModel::Boltz2 => boltz2::predict_structure_from_aas(aas, ff_map),
        // StructurePredictionModel::EsmFold2 => esm_fold2::predict_structure_from_aas(aas, ff_map),
        StructurePredictionModel::OpenDDE => {
            opendde::predict_structure_from_aas(aas, ff_map, control)
        }
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
    predict_structure_from_dna_with_control(model, nts, ff_map, &PredictionControl::default())
}

fn predict_structure_from_dna_with_control(
    model: StructurePredictionModel,
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    control.check_cancelled()?;
    match model {
        StructurePredictionModel::Boltz2 => boltz2::predict_structure_from_dna(nts, ff_map),
        // StructurePredictionModel::EsmFold2 => esm_fold2::predict_structure_from_dna(nts, ff_map),
        StructurePredictionModel::OpenDDE => {
            opendde::predict_structure_from_dna(nts, ff_map, control)
        }
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

pub(crate) fn predict_structure_from_nts_with_control(
    model: StructurePredictionModel,
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    predict_structure_from_dna_with_control(model, nts, ff_map, control)
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
    run_model_command_with_control(command, model, &PredictionControl::default())
}

/// Run a model process while forwarding its stdout and stderr as soon as either pipe produces
/// bytes. Reading both pipes concurrently avoids the deadlock risk of waiting on one full pipe.
pub fn run_model_command_with_control(
    command: &mut Command,
    model: &str,
    control: &PredictionControl,
) -> io::Result<()> {
    control.check_cancelled()?;
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    println!("{model} input command: {command:?}");

    let mut child = command.spawn().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("unable to start {model}; install it separately and put it on PATH: {error}"),
        )
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other(format!("unable to capture {model} stdout")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other(format!("unable to capture {model} stderr")))?;
    let (output_tx, output_rx) = mpsc::channel();
    let stdout_reader = spawn_output_reader(stdout, ProcessStream::Stdout, output_tx.clone());
    let stderr_reader = spawn_output_reader(stderr, ProcessStream::Stderr, output_tx);

    let mut captured_stdout = Vec::new();
    let mut captured_stderr = Vec::new();
    let status = loop {
        while let Ok(chunk) = output_rx.try_recv() {
            forward_process_chunk(&chunk, &mut captured_stdout, &mut captured_stderr);
        }

        if control.is_cancel_requested() {
            terminate_child(&mut child)?;
            let _ = child.wait();
            join_output_reader(stdout_reader, model, "stdout")?;
            join_output_reader(stderr_reader, model, "stderr")?;
            for chunk in output_rx.try_iter() {
                forward_process_chunk(&chunk, &mut captured_stdout, &mut captured_stderr);
            }
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                format!("{model} prediction was cancelled"),
            ));
        }

        if let Some(status) = child.try_wait()? {
            break status;
        }

        match output_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => forward_process_chunk(&chunk, &mut captured_stdout, &mut captured_stderr),
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {}
        }
    };

    join_output_reader(stdout_reader, model, "stdout")?;
    join_output_reader(stderr_reader, model, "stderr")?;
    for chunk in output_rx.try_iter() {
        forward_process_chunk(&chunk, &mut captured_stdout, &mut captured_stderr);
    }

    if status.success() {
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&captured_stdout);
    let stderr = String::from_utf8_lossy(&captured_stderr);
    Err(io::Error::other(format!(
        "{model} exited with {}\nstdout:\n{}\nstderr:\n{}",
        status,
        truncate_process_output(&stdout),
        truncate_process_output(&stderr),
    )))
}

#[derive(Clone, Copy)]
enum ProcessStream {
    Stdout,
    Stderr,
}

struct ProcessChunk {
    stream: ProcessStream,
    bytes: Vec<u8>,
}

fn spawn_output_reader(
    mut pipe: impl Read + Send + 'static,
    stream: ProcessStream,
    tx: mpsc::Sender<ProcessChunk>,
) -> thread::JoinHandle<io::Result<()>> {
    thread::spawn(move || {
        let mut buffer = [0; 4096];
        loop {
            let byte_count = pipe.read(&mut buffer)?;
            if byte_count == 0 {
                return Ok(());
            }
            if tx
                .send(ProcessChunk {
                    stream,
                    bytes: buffer[..byte_count].to_vec(),
                })
                .is_err()
            {
                return Ok(());
            }
        }
    })
}

fn join_output_reader(
    reader: thread::JoinHandle<io::Result<()>>,
    model: &str,
    stream_name: &str,
) -> io::Result<()> {
    reader.join().map_err(|_| {
        io::Error::other(format!(
            "{model} {stream_name} reader thread terminated unexpectedly"
        ))
    })?
}

fn forward_process_chunk(
    chunk: &ProcessChunk,
    captured_stdout: &mut Vec<u8>,
    captured_stderr: &mut Vec<u8>,
) {
    const MAX_CAPTURED_BYTES: usize = 64 * 1024;

    let (output, captured): (&mut dyn Write, &mut Vec<u8>) = match chunk.stream {
        ProcessStream::Stdout => (&mut io::stdout(), captured_stdout),
        ProcessStream::Stderr => (&mut io::stderr(), captured_stderr),
    };
    let _ = output.write_all(&chunk.bytes);
    let _ = output.flush();

    let remaining = MAX_CAPTURED_BYTES.saturating_sub(captured.len());
    captured.extend_from_slice(&chunk.bytes[..chunk.bytes.len().min(remaining)]);
}

fn terminate_child(child: &mut Child) -> io::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }

    // Python console-script launchers on Windows may leave their Python child alive if only the
    // launcher is killed. `taskkill /T` terminates that process tree; `Child::kill` is the fallback.
    #[cfg(target_os = "windows")]
    {
        let status = Command::new("taskkill")
            .arg("/PID")
            .arg(child.id().to_string())
            .arg("/T")
            .arg("/F")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if status.is_ok_and(|status| status.success()) {
            return Ok(());
        }
    }

    match child.kill() {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => Ok(()),
        Err(error) => Err(error),
    }
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
