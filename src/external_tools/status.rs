//! Probing whether tools are installed and actually run, for the tools status panel.

use std::{
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread::{self, sleep},
    time::{Duration, Instant},
};

use super::{
    manage::managed_disk_usage, paths::find_executable, process::prepare_python_environment,
    registry::Tool,
};

/// How long a `--version`/`--help` style probe is given before it is killed.
///
/// Generous because some of these are Python entry points that import Torch before answering, and
/// a cold page cache makes that take seconds. A probe that times out is reported as an error
/// rather than as "not installed", since those call for different fixes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// A short probe for tools that are plain native binaries, where a slow answer means something is
/// wrong rather than merely cold. Also guards against the ORCA/screen-reader hang: an `orca` that
/// is actually the Orca screen reader never answers and must not block the UI.
const PROBE_TIMEOUT_NATIVE: Duration = Duration::from_secs(3);

/// Outcome of probing whether a tool can be reached and run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckResult {
    /// Found, and it answered a version probe as itself.
    Pass,
    /// Not installed anywhere we look. The fix is to install it.
    CantFind,
    /// Installed but not working — wrong program, missing weights, a broken environment. The fix
    /// is to go read the detail, so this sorts above `CantFind`.
    Error,
}

impl CheckResult {
    /// Sort order for the status panel: working tools, then broken ones, then absent ones.
    pub fn rank(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Error => 1,
            Self::CantFind => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pass => "Ready",
            Self::CantFind => "Not installed",
            Self::Error => "Error",
        }
    }
}

/// One tool's status, as shown in the tools panel.
#[derive(Clone, Debug)]
pub struct ToolStatus {
    pub tool: Tool,
    pub result: CheckResult,
    /// A version string on success; the reason otherwise.
    pub detail: String,
    /// Where it was found, when it was.
    pub path: Option<PathBuf>,
}

impl ToolStatus {
    fn cant_find(tool: Tool, detail: String) -> Self {
        Self {
            tool,
            result: CheckResult::CantFind,
            detail,
            path: None,
        }
    }
}

/// One incremental result from probing the external-tool registry.
///
/// Status is sent before disk usage is calculated so the UI can update a row as soon as its
/// subprocess finishes, even when walking the tool's managed installation takes longer.
pub enum ToolCheckUpdate {
    Status(ToolStatus),
    ManagedDiskUsage {
        tool: Tool,
        result: io::Result<Option<u64>>,
    },
}

/// Whether a tool resolves to a file that exists, without running anything.
///
/// A filesystem lookup only, so it is cheap enough for startup, where [`check`] is not: the
/// Python-based tools import Torch before they will answer a version probe, which is seconds each.
/// It cannot tell a working install from a broken one — that is what [`check`] and the tools panel
/// are for — but it does distinguish "installed" from "absent", which is what enabling or greying
/// out a button needs.
pub fn is_installed(tool: Tool) -> bool {
    if find_executable(tool).is_err() {
        return false;
    }

    let spec = tool.spec();
    let Some(root) = spec.bundle_root() else {
        return spec.required_assets.is_empty();
    };

    spec.required_assets
        .iter()
        .all(|asset| root.join(asset.relative_path).exists())
}

/// Probe one tool. Runs a subprocess, so keep it off the UI thread.
pub fn check(tool: Tool) -> ToolStatus {
    let spec = tool.spec();

    let executable = match find_executable(tool) {
        Ok(path) => path,
        Err(error) => return ToolStatus::cant_find(tool, error.to_string()),
    };

    let failed = |detail: String| ToolStatus {
        tool,
        result: CheckResult::Error,
        detail,
        path: Some(executable.clone()),
    };

    // Weights and data directories are checked before the probe: a checkout whose model
    // parameters never downloaded passes a `python --version` probe and then fails on first use,
    // which is exactly the confusion this panel exists to prevent.
    if !spec.required_assets.is_empty() {
        let Some(root) = spec.bundle_root() else {
            return ToolStatus::cant_find(tool, format!("no data directory for {}", spec.name()));
        };
        for asset in spec.required_assets {
            let path = root.join(asset.relative_path);
            if !path.exists() {
                return failed(format!(
                    "Missing {} ({}). Re-run: {}",
                    asset.description,
                    path.display(),
                    spec.install_command()
                ));
            }
        }
    }

    let timeout = if spec.slow_probe {
        PROBE_TIMEOUT
    } else {
        PROBE_TIMEOUT_NATIVE
    };

    let output = match probe(&executable, spec.version_args, timeout) {
        Ok(output) => output,
        Err(error) => return failed(error.to_string()),
    };

    if !output
        .to_lowercase()
        .contains(&spec.version_marker.to_lowercase())
    {
        return failed(format!(
            "{} did not identify itself as {}. Its output was: {}",
            executable.display(),
            spec.name(),
            first_line(&output)
        ));
    }

    ToolStatus {
        tool,
        result: CheckResult::Pass,
        detail: first_line(&output).to_owned(),
        path: Some(executable),
    }
}

/// Probe every tool concurrently, streaming each result as soon as it is available.
///
/// Concurrent because a serial pass costs the sum of every Python interpreter's Torch import.
/// The channel disconnects after every status and applicable managed-disk measurement has been
/// sent, following the same non-blocking receiver pattern used by the rest of the UI workers.
pub fn check_all() -> Receiver<ToolCheckUpdate> {
    use rayon::prelude::*;

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        Tool::ALL.par_iter().for_each_with(tx, |tx, tool| {
            let started = Instant::now();
            let status = check(*tool);
            println!(
                "External tool check for {} took {:.3}ms",
                tool.spec().name(),
                started.elapsed().as_millis()
            );

            let measure_disk_usage = status.result != CheckResult::CantFind;
            if tx.send(ToolCheckUpdate::Status(status)).is_err() {
                return;
            }
            if measure_disk_usage {
                let result = managed_disk_usage(*tool);
                let _ = tx.send(ToolCheckUpdate::ManagedDiskUsage {
                    tool: *tool,
                    result,
                });
            }
        });
    });
    rx
}

/// Run a short-lived probe, killing it if it overruns.
///
/// Neither a nonzero exit nor stderr output is treated as failure: many CLIs answer a flag they
/// don't recognize with a usage message and a nonzero status, and we match on the text either way.
fn probe(executable: &Path, args: &[&str], timeout: Duration) -> io::Result<String> {
    let mut command = Command::new(executable);
    prepare_python_environment(&mut command);

    let mut child = command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("unable to run {}: {error}", executable.display()),
            )
        })?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait()? {
            Some(_) => {
                let output = child.wait_with_output()?;
                return Ok(String::from_utf8_lossy(&output.stdout).to_string()
                    + &String::from_utf8_lossy(&output.stderr));
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "{} did not answer within {}s",
                        executable.display(),
                        timeout.as_secs()
                    ),
                ));
            }
            None => sleep(Duration::from_millis(20)),
        }
    }
}

fn first_line(text: &str) -> &str {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("(no output)")
}
