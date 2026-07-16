---
name: boltz-structure-prediction
description: How Boltz-2 structure prediction is made to "just work" in molchanica, and why
metadata:
  type: project
---

Boltz-2 prediction (`src/structure_prediction/`) is designed to "just work" for end users of the
standalone app without them installing Python/uv/Torch/Boltz. Feature-gated behind
`python_for_structure_prediction`.

- `boltz_runtime.rs` (std-only, no PyO3): on first use, provisions a fully isolated env under the
  user data dir — obtains `uv` (PATH / `MOLCHANICA_UV` / downloads pinned release), `uv venv
  --python 3.12` (uv auto-fetches CPython; Boltz needs NumPy<2 ⇒ Py 3.11/3.12), `uv pip install
  boltz`. A `.provisioned` marker enables a cheap `runtime_ready()` startup check.
- `pyo3_interface.rs` (PyO3 0.29, `Python::attach`, `py.run(&CStr,…)`): opt-in in-process runner.
- `boltz2.rs::run_boltz`: feature-on → provision + run; default = subprocess of the managed venv's
  `boltz` launcher; in-process only when `MOLCHANICA_BOLTZ_INPROCESS=1` (then falls back to
  subprocess on error).

**Why subprocess is the default, not PyO3 in-process** (the user originally asked for PyO3
embedding): PyO3 does NOT bundle Python or deps — it only embeds an interpreter, and (a) importing
Torch needs the embedded interpreter's ABI to match the managed venv's Python (system here is 3.13,
managed is 3.12 ⇒ mismatch), and (b) `pyo3/auto-initialize` makes the binary require a libpython at
runtime. The auto-provisioned uv env is what actually removes the "delicate Python setup" pain; the
subprocess keeps Boltz's Lightning/multiprocessing machinery out of the GUI process. In-process is
kept as an opt-in for builds that deliberately align `PYO3_PYTHON` to 3.12.

**How to apply:** For true zero-touch shipping, bundle a standalone CPython + libpython, or keep the
in-process path opt-in and rely on the subprocess path. Consider SHA-256 verification of the
downloaded `uv` (currently a noted TODO in `download_uv`). See [[molchanica-build-validation]].
