//! Running a tool: the scratch directory its files are exchanged through, the process runner, and
//! the cancellation signal the UI holds.
//!
//! Every adapter, native or shared, runs a tool the same way:
//!
//! 1. Write inputs into a [`ToolWorkspace`] (or, for `bio_tools` adapter runs, a durable results
//!    directory).
//! 2. Build a [`Command`] and hand it to [`run_tool`], which streams the tool's output to the
//!    terminal, honours an optional [`RunControl`], and folds the output into the error on failure.
//! 3. Read results back out of files the tool wrote. Nothing parses the tool's stdout.

use std::{
    env,
    ffi::OsStr,
    fs,
    io::{self, Read, Write},
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

use super::registry::Tool;

/// A scratch directory for one tool run, removed when dropped.
pub struct ToolWorkspace {
    tool: Tool,
    root: PathBuf,
}

impl ToolWorkspace {
    pub fn new(tool: Tool) -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        for _ in 0..100 {
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = env::temp_dir().join(format!(
                "molchanica-{}-{}-{timestamp}-{counter}",
                tool.spec().slug(),
                std::process::id()
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Ok(Self { tool, root }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "unable to allocate a unique tool workspace",
        ))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.root.join(relative)
    }

    pub fn create_dir(&self, relative: impl AsRef<Path>) -> io::Result<PathBuf> {
        let path = self.path(relative);
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Write an input file, returning its path for the command line.
    pub fn write(
        &self,
        relative: impl AsRef<Path>,
        contents: impl AsRef<[u8]>,
    ) -> io::Result<PathBuf> {
        let path = self.path(relative);
        fs::write(&path, contents)?;
        Ok(path)
    }

    /// Read a file the tool was expected to write, saying which tool failed to write it.
    pub fn read_output(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "{} finished but did not write {}: {error}",
                    self.tool.spec().name(),
                    path.display()
                ),
            )
        })
    }
}

impl Drop for ToolWorkspace {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root)
            && error.kind() != io::ErrorKind::NotFound
        {
            eprintln!(
                "Unable to remove tool workspace {}: {error}",
                self.root.display()
            );
        }
    }
}

/// A cloneable cancellation signal shared by the UI and a worker running a tool.
#[derive(Clone, Debug, Default)]
pub struct RunControl {
    cancel_requested: Arc<AtomicBool>,
}

impl RunControl {
    pub fn cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    fn check_cancelled(&self, tool: Tool) -> io::Result<()> {
        if self.is_cancel_requested() {
            Err(cancelled(tool))
        } else {
            Ok(())
        }
    }
}

fn cancelled(tool: Tool) -> io::Error {
    io::Error::new(
        io::ErrorKind::Interrupted,
        format!("the {} run was cancelled", tool.spec().name()),
    )
}

/// Output kept from each stream for the error message, beyond which it is only forwarded.
const MAX_CAPTURED_BYTES: usize = 64 * 1024;

/// Output included in an error message, per stream.
const MAX_REPORTED_BYTES: usize = 16 * 1024;

/// Run a tool to completion.
///
/// Both output streams are forwarded to Molchanica's own as the tool produces them, under a banner
/// naming the workflow and the command, so a GUI-launched run is diagnosable from the terminal.
/// Both pipes are read concurrently, which avoids the deadlock of waiting on one full pipe. On
/// failure, the head of each stream is folded into the error.
///
/// `control`, where given, kills the process tree as soon as cancellation is requested.
pub fn run_tool(
    command: &mut Command,
    tool: Tool,
    workflow: &str,
    control: Option<&RunControl>,
) -> io::Result<()> {
    let name = tool.spec().name();
    if let Some(control) = control {
        control.check_cancelled(tool)?;
    }

    prepare_python_environment(command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let _banner = RunBanner::new(workflow, name, command);

    let mut child = command.spawn().map_err(|error| {
        io::Error::new(error.kind(), format!("unable to start {name}: {error}"))
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other(format!("unable to capture {name} stdout")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other(format!("unable to capture {name} stderr")))?;

    let (output_tx, output_rx) = mpsc::channel();
    let readers = [
        spawn_output_reader(stdout, Stream::Stdout, output_tx.clone()),
        spawn_output_reader(stderr, Stream::Stderr, output_tx),
    ];
    let mut captured = Captured::default();

    let status = loop {
        while let Ok(chunk) = output_rx.try_recv() {
            captured.forward(&chunk);
        }

        if control.is_some_and(RunControl::is_cancel_requested) {
            terminate_child(&mut child)?;
            let _ = child.wait();
            join_output_readers(readers, name)?;
            output_rx
                .try_iter()
                .for_each(|chunk| captured.forward(&chunk));
            return Err(cancelled(tool));
        }

        if let Some(status) = child.try_wait()? {
            break status;
        }

        if let Ok(chunk) = output_rx.recv_timeout(Duration::from_millis(100)) {
            captured.forward(&chunk);
        }
    };

    join_output_readers(readers, name)?;
    output_rx
        .try_iter()
        .for_each(|chunk| captured.forward(&chunk));

    if status.success() {
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&captured.stdout);
    let stderr = String::from_utf8_lossy(&captured.stderr);

    Err(io::Error::other(format!(
        "{name} exited with {status}\nstdout:\n{}\nstderr:\n{}",
        truncate_output(&stdout),
        truncate_output(&stderr),
    )))
}

#[derive(Clone, Copy)]
enum Stream {
    Stdout,
    Stderr,
}

struct Chunk {
    stream: Stream,
    bytes: Vec<u8>,
}

/// The head of each stream, kept for the error message.
#[derive(Default)]
struct Captured {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Captured {
    /// Forward a chunk to Molchanica's corresponding stream, and keep it if there is room.
    fn forward(&mut self, chunk: &Chunk) {
        let (output, captured): (&mut dyn Write, &mut Vec<u8>) = match chunk.stream {
            Stream::Stdout => (&mut io::stdout(), &mut self.stdout),
            Stream::Stderr => (&mut io::stderr(), &mut self.stderr),
        };
        let _ = output.write_all(&chunk.bytes);
        let _ = output.flush();

        let remaining = MAX_CAPTURED_BYTES.saturating_sub(captured.len());
        captured.extend_from_slice(&chunk.bytes[..chunk.bytes.len().min(remaining)]);
    }
}

fn spawn_output_reader(
    mut pipe: impl Read + Send + 'static,
    stream: Stream,
    tx: mpsc::Sender<Chunk>,
) -> thread::JoinHandle<io::Result<()>> {
    thread::spawn(move || {
        let mut buffer = [0; 4096];
        loop {
            let byte_count = pipe.read(&mut buffer)?;
            if byte_count == 0 {
                return Ok(());
            }

            let chunk = Chunk {
                stream,
                bytes: buffer[..byte_count].to_vec(),
            };
            if tx.send(chunk).is_err() {
                return Ok(());
            }
        }
    })
}

fn join_output_readers(
    readers: [thread::JoinHandle<io::Result<()>>; 2],
    name: &str,
) -> io::Result<()> {
    for reader in readers {
        reader.join().map_err(|_| {
            io::Error::other(format!(
                "{name} output reader thread terminated unexpectedly"
            ))
        })??;
    }
    Ok(())
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

fn truncate_output(output: &str) -> &str {
    if output.len() <= MAX_REPORTED_BYTES {
        return output;
    }

    let mut boundary = MAX_REPORTED_BYTES;
    while !output.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &output[..boundary]
}

/// Prints a header naming the run before it starts, and a rule under its output when dropped.
struct RunBanner;

impl RunBanner {
    fn new(workflow: &str, tool_name: &str, command: &Command) -> Self {
        let mut output = io::stdout().lock();
        let _ = writeln!(
            output,
            "\nRunning {workflow} with {tool_name}\n============\nCommand: {}",
            display_command(command)
        );
        if let Some(directory) = command.get_current_dir() {
            let _ = writeln!(output, "Working directory: {}", directory.display());
        }
        let _ = output.flush();
        Self
    }
}

impl Drop for RunBanner {
    fn drop(&mut self) {
        let mut output = io::stdout().lock();
        let _ = writeln!(output, "\n============\n");
        let _ = output.flush();
    }
}

/// A command rendered for a person to read, quoting arguments that need it. Not for a shell.
pub fn display_command(command: &Command) -> String {
    std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(display_command_argument)
        .collect::<Vec<_>>()
        .join(" ")
}

fn display_command_argument(argument: &OsStr) -> String {
    let value = argument.to_string_lossy();
    if value.is_empty() {
        return "\"\"".to_owned();
    }

    if value
        .chars()
        .any(|character| character.is_whitespace() || character == '"')
    {
        format!("\"{}\"", value.replace('"', "\\\""))
    } else {
        value.into_owned()
    }
}

/// Keep activated virtualenvs, Conda, and user Python settings from leaking into a managed tool,
/// and make Python flush output as it is written rather than when its buffer fills — otherwise a
/// run's progress arrives all at once at the end, and a probe can look like a hang.
///
/// Explicit venv interpreters and console-script launchers do not need activation. Harmless for
/// tools that are not Python at all.
pub(super) fn prepare_python_environment(command: &mut Command) {
    for variable in [
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
    ] {
        command.env_remove(variable);
    }

    command
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONUNBUFFERED", "1");
}
