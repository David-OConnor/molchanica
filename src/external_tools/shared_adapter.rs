//! Desktop transport for the scientific adapters owned by bio_tools.
use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value};

use super::{
    Tool, data_root,
    tool_form::{FieldKind, FormContract},
};
use crate::structure_prediction::{PredictionControl, run_model_command};

#[derive(Clone, Debug)]
pub struct AdapterResult {
    pub directory: PathBuf,
    pub archive: PathBuf,
    pub files: Vec<PathBuf>,
    pub details: Value,
}

pub fn slug(tool: Tool) -> &'static str {
    match tool {
        Tool::RfDiffusion3 => "rfd3",
        Tool::ProteinMpnn => "proteinmpnn",
        Tool::LigandMpnn => "ligandmpnn",
        Tool::OpenDde => "opendde",
        Tool::Boltz2 => "boltz2",
        Tool::Chai1 => "chai1",
        Tool::EsmFold2 => "esmfold2",
        _ => unreachable!("only scientific form tools use the shared adapter"),
    }
}

/// File widgets contain local paths; the shared adapters accept their UTF-8 contents.
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
            && !value.starts_with("bio-tools://")
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

pub fn run(tool: Tool, payload: Value, control: &PredictionControl) -> io::Result<AdapterResult> {
    if !tool.spec().platform.is_supported() {
        return Err(io::Error::other(format!(
            "{} requires Linux",
            tool.spec().name()
        )));
    }
    let root = data_root()
        .ok_or_else(|| io::Error::other("Cannot find the Molchanica data folder"))?
        .join("process_executables");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let directory = root
        .join("results")
        .join(slug(tool))
        .join(format!("{stamp}-{}", std::process::id()));
    fs::create_dir_all(&directory)?;
    let request = directory.join("submitted-form.json");
    let response = directory.join("result.json");
    fs::write(&request, serde_json::to_vec_pretty(&payload)?)?;
    let package = bio_tools::adapters::package_path()?;
    // uv caches this small, separate coordinator environment. Model packages stay
    // in their own bio_tools-managed environments, which often have incompatible pins.
    let uv = std::env::var_os("MOLCHANICA_UV")
        .or_else(|| std::env::var_os("BIO_TOOLS_UV"))
        .unwrap_or_else(|| "uv".into());
    let mut command = Command::new(uv);
    command.args(["run", "--no-project", "--with", "athanor_bio_tools>=0.1.2", "--with", "pyyaml", "--python", "3.12", "python", "-c", "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('bio_tool_adapters.desktop',run_name='__main__')"])
        .arg(package).arg(slug(tool)).arg(&request).arg(&response)
        .env("BIO_TOOLS_EXECUTABLE_ROOT", &root)
        .current_dir(&directory);
    let execution = run_model_command(&mut command, tool.spec().name(), control);
    if control.is_cancel_requested() {
        return Err(io::Error::new(io::ErrorKind::Interrupted, "Run cancelled"));
    }
    let envelope: Value = fs::read(&response)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or(Value::Null);
    if let Some(error) = envelope.get("error").and_then(Value::as_str) {
        return Err(io::Error::other(format!(
            "{error}\nRun files: {}",
            directory.display()
        )));
    }
    execution.map_err(|error| {
        io::Error::other(format!("{error}\nRun files: {}", directory.display()))
    })?;
    let details = envelope
        .get("result")
        .cloned()
        .ok_or_else(|| io::Error::other("The adapter did not return results"))?;
    let log = details
        .get("run_log_dir")
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other("Missing results directory"))?;
    let mut files = Vec::new();
    collect_files(&Path::new(log).join("outputs"), &mut files)?;
    files.sort();
    Ok(AdapterResult {
        archive: directory.join("raw-results.zip"),
        directory,
        files,
        details,
    })
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
