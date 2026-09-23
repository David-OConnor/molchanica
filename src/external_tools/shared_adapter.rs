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
    let task = values.get("task").map(String::as_str).unwrap_or("");

    for field in &contract.fields {
        if field.managed_by_runner || !field.applies_to(mode) || !field.applies_to_task(task) {
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
    if slug(tool) == "rfd3" {
        if mode == "parameters" && result.get("length").and_then(Value::as_str) == Some("null") {
            result.insert("length".into(), Value::String(String::new()));
        }
        for field in ["inputs", "inputs_file"] {
            if let Some(Value::String(document)) = result.get_mut(field) {
                normalize_rfd3_document(document)?;
            }
        }
    }
    Ok(Value::Object(result))
}

fn normalize_rfd3_document(document: &mut String) -> io::Result<()> {
    let parsed: Result<Value, _> = serde_json::from_str(document);
    let mut parsed = match parsed {
        Ok(value) => value,
        Err(_) => match serde_yaml::from_str(document) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        },
    };
    let Some(designs) = parsed.as_object_mut() else {
        return Ok(());
    };
    let mut changed = false;
    for design in designs.values_mut() {
        let Some(source) = design
            .get("input")
            .and_then(Value::as_str)
            .and_then(|source| source.strip_prefix("../input_pdbs/"))
        else {
            continue;
        };
        let reference = format!("bio-tools://rfd3/input_pdbs/{source}");
        bio_tools::tool_definitions::presets::input_text("rfd3", &reference)?;
        design["input"] = Value::String(reference);
        changed = true;
    }
    if changed {
        *document = serde_json::to_string(&parsed)?;
    }
    Ok(())
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
    if !spec.is_supported() {
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
    if let Some(environment) = spec.venv_root() {
        command.env("BIO_TOOLS_ADAPTER_ENVIRONMENT", environment);
    }
    if let Some(bundle) = spec.bundle_root() {
        command.env("BIO_TOOLS_ADAPTER_BUNDLE_ROOT", bundle);
    }
    if let Some(executable) = env::var_os(spec.exe_override_env) {
        let variable = match spec.kind {
            super::registry::ToolKind::VenvPython => "BIO_TOOLS_ADAPTER_PYTHON",
            _ => "BIO_TOOLS_ADAPTER_EXECUTABLE",
        };
        command.env(variable, executable);
    }

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
    AdapterResult::from_details(directory, details)
}

impl AdapterResult {
    pub fn load(directory: PathBuf) -> io::Result<Self> {
        let envelope: Value = serde_json::from_slice(&fs::read(directory.join("result.json"))?)?;
        let details = envelope
            .get("result")
            .cloned()
            .ok_or_else(|| io::Error::other("This run has no completed results"))?;
        Self::from_details(directory, details)
    }

    fn from_details(directory: PathBuf, details: Value) -> io::Result<Self> {
        let mut files = Vec::new();
        if let Some(outputs) = details.get("output_files").and_then(Value::as_array) {
            files.extend(outputs.iter().filter_map(Value::as_str).map(PathBuf::from));
        } else {
            let log = details
                .get("run_log_dir")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    io::Error::other("The adapter did not name its results directory")
                })?;
            collect_files(&Path::new(log).join("outputs"), &mut files)?;
        }
        files.sort();
        Ok(Self {
            archive: directory.join("raw-results.zip"),
            directory,
            files,
            details,
        })
    }
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
        .args(["--with", &python_bindings_requirement()])
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

/// The adapters `import bio_tools`: the Python bindings (`athanor_bio_tools`) for the same crate
/// this binary links, and embeds the adapters from.
///
/// `bio_tools` is a path dependency, so the adapters can be newer than any published bindings.
/// When its source tree is present, build the bindings from it; uv rebuilds them when the crate's
/// sources change (`cache-keys` in its `python/pyproject.toml`). Otherwise, fall back to PyPI, pinned
/// at or above the version whose APIs the embedded adapters expect.
fn python_bindings_requirement() -> String {
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bio_tools")
        .join("python");

    match local.join("pyproject.toml").is_file() {
        true => local.to_string_lossy().into_owned(),
        false => "athanor_bio_tools>=0.1.3".to_owned(),
    }
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
