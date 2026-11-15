use std::{collections::HashMap, env, io, path::Path};

use bio_tools::{
    adapters,
    run::{self, CaptureLimits, CommandSpec},
    tool_definitions::Tool as InstallableTool,
};
use serde_json::json;

use super::{Tool, manage::managed_installer};

const DESKTOP: &str =
    "import runpy; runpy.run_module('bio_tool_adapters.desktop', run_name='__main__')";

const CONVERT: &str = r#"
import json
from bio_tool_adapters.psi4_input import from_boxes, to_boxes

request = json.load(sys.stdin)
values = request["values"]
try:
    if request["target"] == "text":
        result = {"psi4_input": from_boxes(values)}
    else:
        result = to_boxes(values.get("psi4_input", ""), values)
except (SyntaxError, ValueError) as error:
    print(json.dumps({"error": str(error)}))
else:
    print(json.dumps({"values": result}, ensure_ascii=False))
"#;

fn python() -> io::Result<CommandSpec> {
    Ok(managed_installer()?.tool_python_command(InstallableTool::Psi4))
}

fn with_adapter(python: CommandSpec, script: &str) -> io::Result<CommandSpec> {
    let bootstrap = format!(
        "import sys; sys.path.insert(0,sys.argv.pop(1)); \
         import bio_tools_desktop; bio_tools_desktop.install();\n{script}"
    );

    Ok(python
        .args(["-B", "-X", "utf8", "-c"])
        .arg(bootstrap)
        .arg(adapters::package_path()?)
        .env("BIO_TOOLS_ADAPTER_HOST", env::current_exe()?))
}

pub(super) fn coordinator() -> io::Result<CommandSpec> {
    super::find_executable(Tool::Psi4)?;

    with_adapter(python()?, DESKTOP)
}

/// Convert editable input modes through the same parser and writer as bio_web.
pub(crate) fn project(
    values: &HashMap<String, String>,
    target: &str,
) -> io::Result<HashMap<String, String>> {
    let mut python = python()?;
    if !Path::new(&python.program).is_file() {
        python = CommandSpec::new(if cfg!(windows) { "python" } else { "python3" }).arg("-I");
    }

    let command = with_adapter(python, CONVERT)?
        .stdin(serde_json::to_vec(
            &json!({ "values": values, "target": target }),
        )?)
        .capture_limits(CaptureLimits::new(2_000_000, 16_000));
    let output = run::run(&command).map_err(|error| {
        if error.io_error_kind() == Some(io::ErrorKind::NotFound) {
            io::Error::other("Input conversion needs Psi4 or Python 3 on PATH.")
        } else {
            io::Error::other(error)
        }
    })?;
    let result: Projection = serde_json::from_slice(&output.stdout)?;

    match result {
        Projection::Values { values } => Ok(values),
        Projection::Error { error } => Err(io::Error::other(error)),
    }
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum Projection {
    Values { values: HashMap<String, String> },
    Error { error: String },
}
