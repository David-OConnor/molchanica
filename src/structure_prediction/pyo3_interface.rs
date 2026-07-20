// todo: Inop, and may not be a good approach. Consider removing.

//! Optional in-process Boltz execution through an embedded Python interpreter (PyO3).
//!
//! This runs Boltz inside Molchanica's own process using the interpreter PyO3 links against
//! (`pyo3/auto-initialize`), rather than launching a child process. It is opt-in
//! (`MOLCHANICA_BOLTZ_INPROCESS=1`) because it comes with real constraints:
//!
//! * **ABI match required.** The embedded interpreter must be the *same* minor version as the
//!   managed environment built by [`super::boltz_runtime`] (currently Python 3.12), because Torch
//!   and friends are compiled C extensions. Build Molchanica with `PYO3_PYTHON` pointing at a 3.12
//!   interpreter to line these up. If they do not match, importing Torch aborts the process, so the
//!   caller only takes this path when explicitly requested.
//! * **libpython at runtime.** Linking PyO3 makes the Molchanica binary depend on a matching
//!   `libpython` being present (or bundled) at runtime.
//! * **Boltz internals.** Boltz uses PyTorch Lightning, which may use multiprocessing; running it
//!   in-process is inherently more fragile than the subprocess path.
//!
//! Because of the above, the subprocess path in [`super::boltz_runtime::BoltzRuntime::predict`] is
//! the default and recommended execution mode; this module exists for setups that have deliberately
//! aligned the embedded interpreter with the managed environment.

use std::{io, path::Path};

use pyo3::{prelude::*, types::PyDict};

use crate::structure_prediction::boltz_runtime::BoltzRuntime;

/// Python driver: put the managed venv's site-packages on `sys.path`, set `sys.argv` to the Boltz
/// CLI invocation, resolve the `boltz` console-script entry point via importlib metadata (robust to
/// Boltz's internal module layout), and run it. A clean `SystemExit` is treated as success.
const DRIVER: &std::ffi::CStr = cr#"
import sys

for entry in site_packages:
    if entry not in sys.path:
        sys.path.insert(0, entry)

sys.argv = ["boltz", "predict", input_path, "--out_dir", out_dir]
if use_msa_server:
    sys.argv.append("--use_msa_server")

from importlib.metadata import entry_points

eps = entry_points()
try:
    scripts = eps.select(group="console_scripts")
except AttributeError:  # Python < 3.10 dict interface
    scripts = eps.get("console_scripts", [])

target = None
for ep in scripts:
    if ep.name == "boltz":
        target = ep
        break

if target is None:
    raise RuntimeError("the 'boltz' console script is not installed in the embedded environment")

run = target.load()
try:
    run()
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
"#;

/// Run a Boltz prediction inside the embedded interpreter.
///
/// Returns an error (rather than aborting) for the recoverable failure modes; the caller falls back
/// to the subprocess runner. Note that a hard ABI mismatch when importing Torch can still abort the
/// process, which is why this path is opt-in.
pub(super) fn predict(
    runtime: &BoltzRuntime,
    input_path: &Path,
    output_dir: &Path,
    use_msa_server: bool,
) -> io::Result<()> {
    let site_packages = runtime.site_packages()?;

    Python::attach(|py| -> PyResult<()> {
        let globals = PyDict::new(py);
        globals.set_item("site_packages", site_packages)?;
        globals.set_item("input_path", input_path.to_string_lossy().as_ref())?;
        globals.set_item("out_dir", output_dir.to_string_lossy().as_ref())?;
        globals.set_item("use_msa_server", use_msa_server)?;
        py.run(DRIVER, Some(&globals), None)?;
        Ok(())
    })
    .map_err(|error| io::Error::other(format!("Boltz in-process execution failed: {error}")))
}
