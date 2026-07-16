---
name: molchanica-build-validation
description: How to build/validate molchanica in this environment (CUDA, features)
metadata:
  type: project
---

Building/validating molchanica locally:

- Default features = `["cuda"]`, which requires the CUDA 13 toolkit (cudarc dynamic-linking,
  `cuda-13000`). **This machine has no CUDA toolchain** (`nvcc` absent), so the default `cargo
  build`/`check` cannot complete here.
- `cargo check --no-default-features` does NOT work as a fallback: the tree has code that only
  coheres under `cuda`/other features, so it surfaces unrelated errors.
- The `python_for_structure_prediction` feature (PyO3) builds fine here; set
  `PYO3_PYTHON=C:\Users\david\AppData\Local\Programs\Python\Python313\python.exe` (plain `python3`
  on Windows is the Store stub). PyO3 0.29 uses `Python::attach` (not `with_gil`); `run`/`eval` take
  `&CStr` (use `cr#"..."#` literals).

As of 2026-07-15 the working tree also had pre-existing compile errors unrelated to structure
prediction (e.g. `state.rs` `IntegrationsAvail` has both `#[derive(Default)]` and a manual `impl
Default`; `main.rs:96` `println!` on a non-literal). These are the user's in-progress edits — verify
whether still present before assuming a build is green. Related: [[boltz-structure-prediction]].
