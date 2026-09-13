//! Running a [`ToolAdapter::SharedAdapter`] tool through the scientific adapters `bio_tools` owns.
//!
//! Every such tool is run identically: the submitted form is written as JSON into a fresh results
//! directory, `bio_tools`' `desktop` coordinator is started in a small uv environment of its own,
//! and it validates the form, builds the tool's command line, runs it in the tool's managed
//! environment, and writes a result envelope back. Nothing here is specific to any one tool; what
//! differs between them is data, in the form contract [`tool_form`](super::tool_form) loads.
//!
//! Results directories are kept: runs take minutes to hours, and their outputs are the point.

use std::{
    collections::HashMap,
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value};

use super::{
    RunControl, Tool, ToolAdapter,
    paths::process_executables_dir,
    process::run_tool,
    tool_form::{FieldKind, FormContract},
};

/// The prefix a preset uses to name a file `bio_tools` bundles with it, rather than one on disk.
const BUNDLED_PREFIX: &str = "bio-tools://";

/// What a finished run left behind.
#[derive(Clone, Debug)]
pub struct AdapterResult {
    /// The run's results directory, holding the submitted form, the envelope, and the outputs.
    pub directory: PathBuf,
    pub archive: PathBuf,
    /// Every output file, sorted.
    pub files: Vec<PathBuf>,
    /// The adapter's `result` object, for the diagnostics panel.
    pub details: Value,
}

/// The key `bio_tools` publishes a shared-adapter tool's contract, presets, and adapter under.
///
/// Panics for tools not driven through the shared adapter; the tool runner only offers those.
pub fn slug(tool: Tool) -> &'static str {
    tool.spec().adapter_slug().unwrap_or_else(|| {
        panic!(
            "{} is not run through the bio_tools shared adapter",
            tool.spec().name()
        )
    })
}

/// Build the payload the adapter receives from the form's values.
///
/// Only fields that apply in `mode` are sent. File widgets hold local paths, but the adapters
/// accept file contents, as the web form uploads them, so those are read here.
pub fn payload(tool: Tool, values: &HashMap<String, String>, mode: &str) -> io::Result<Value> {
    let contract = FormContract::load(slug(tool)).map_err(io::Error::other)?;
    let mut result = Map::new();

    for field in &contract.fields {
        if !field.applies_to(mode) {
            continue;
        }
        let Some(value) = values.get(&field.name) else {
            continue;
        };

        let mut value = value.clone();
        if field.kind() == FieldKind::File
            && !value.trim().is_empty()
            && !value.starts_with(BUNDLED_PREFIX)
        {
            value = fs::read_to_string(value.trim())
                .map_err(|error| io::Error::other(format!("{}: {error}", field.label)))?;
        }
        result.insert(field.name.clone(), Value::String(value));
    }

    if !mode.is_empty() {
        result.insert("input_mode".into(), Value::String(mode.into()));
    }
    if let Some(task) = values.get("task") {
        result.insert("task".into(), Value::String(task.clone()));
    }
    Ok(Value::Object(result))
}

/// Run a shared-adapter tool to completion. Blocking; call it from a worker thread.
pub fn run(tool: Tool, payload: Value, control: &RunControl) -> io::Result<AdapterResult> {
    let spec = tool.spec();
    if spec.adapter != ToolAdapter::SharedAdapter {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} is not run through the bio_tools adapters", spec.name()),
        ));
    }
    if !spec.platform.is_supported() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("{} requires Linux", spec.name()),
        ));
    }

    let executables = process_executables_dir()?;
    let directory = create_results_dir(&executables, tool)?;
    let request = directory.join("submitted-form.json");
    let response = directory.join("result.json");
    fs::write(&request, serde_json::to_vec_pretty(&payload)?)?;

    let mut command = coordinator_command()?;
    command
        .arg(slug(tool))
        .arg(&request)
        .arg(&response)
        .env("BIO_TOOLS_EXECUTABLE_ROOT", &executables)
        .current_dir(&directory);

    let execution = run_tool(&mut command, tool, "the bio_tools adapter", Some(control));
    if control.is_cancel_requested() {
        return Err(io::Error::new(io::ErrorKind::Interrupted, "Run cancelled"));
    }

    // The envelope's own error explains a failure better than the exit status does, so it is
    // checked first; either way, the run files are kept and pointed at.
    let with_run_files = |message: String| {
        io::Error::other(format!("{message}\nRun files: {}", directory.display()))
    };

    let envelope: Value = fs::read(&response)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or(Value::Null);

    if let Some(error) = envelope.get("error").and_then(Value::as_str) {
        return Err(with_run_files(error.to_owned()));
    }
    execution.map_err(|error| with_run_files(error.to_string()))?;

    let details = envelope
        .get("result")
        .cloned()
        .ok_or_else(|| with_run_files("The adapter did not return results".to_owned()))?;
    let run_log_dir = details
        .get("run_log_dir")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            with_run_files("The adapter did not name its results directory".to_owned())
        })?;

    let mut files = Vec::new();
    collect_files(&Path::new(run_log_dir).join("outputs"), &mut files)?;
    files.sort();

    Ok(AdapterResult {
        archive: directory.join("raw-results.zip"),
        directory,
        files,
        details,
    })
}

/// `uv run` for `bio_tools`' `desktop` coordinator, missing only the per-run arguments.
///
/// uv caches this small, separate coordinator environment. Model packages stay in their own
/// `bio_tools`-managed environments, which often have incompatible pins.
fn coordinator_command() -> io::Result<Command> {
    let package = bio_tools::adapters::package_path()?;
    let uv = env::var_os("MOLCHANICA_UV")
        .or_else(|| env::var_os("BIO_TOOLS_UV"))
        .unwrap_or_else(|| "uv".into());

    let mut command = Command::new(uv);
    command
        .args(["run", "--no-project"])
        .args(["--with", "athanor_bio_tools>=0.1.2"])
        .args(["--with", "pyyaml"])
        .args(["--python", "3.12"])
        .args(["python", "-c"])
        .arg(
            "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); \
             runpy.run_module('bio_tool_adapters.desktop',run_name='__main__')",
        )
        .arg(package);
    Ok(command)
}

/// `<process_executables>/results/<slug>/<timestamp>-<pid>`, created fresh.
fn create_results_dir(executables: &Path, tool: Tool) -> io::Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let directory = executables
        .join("results")
        .join(slug(tool))
        .join(format!("{stamp}-{}", std::process::id()));

    fs::create_dir_all(&directory)?;
    Ok(directory)
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if !path.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_files(&entry.path(), files)?;
        } else if kind.is_file() {
            files.push(entry.path());
        }
    }
    Ok(())
}
