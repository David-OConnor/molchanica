//! A self-provisioning, isolated Python environment for Boltz-2.
//!
//! The goal of this module is that Boltz "just works" for an end user of the standalone
//! application, without them ever installing Python, `uv`, `torch`, or Boltz themselves. On first
//! use we build a fully isolated environment under the user's data directory:
//!
//! 1. Obtain `uv` (a single static binary). We use one already on `PATH`, one pointed to by
//!    `MOLCHANICA_UV`, a previously downloaded copy, or we download the pinned release.
//! 2. `uv venv --python 3.12` — `uv` fetches a managed CPython 3.12 automatically (Boltz needs
//!    NumPy < 2, which needs Python 3.11/3.12), so the host system's Python is irrelevant.
//! 3. `uv pip install boltz` into that venv, pulling Torch and the rest of the stack.
//!
//! After provisioning, predictions run by launching the venv's `boltz` executable as a child
//! process (see [`BoltzRuntime::predict`]). This keeps Boltz's heavy, multiprocessing/Lightning
//! based machinery out of the host GUI process, which is far more robust than running it in-process.
//! An opt-in in-process path (via the embedded PyO3 interpreter) is provided separately in
//! `pyo3_interface`; see [`in_process_requested`].
//!
//! Everything here is `std`-only and does not touch PyO3, so the provisioning/subprocess path does
//! not depend on an embedded interpreter.
//!
//! Relevant environment overrides:
//! * `MOLCHANICA_BOLTZ_HOME`         — root directory for the managed runtime.
//! * `MOLCHANICA_UV`                 — path to a `uv` executable to use instead of downloading one.
//! * `MOLCHANICA_UV_VERSION`         — `uv` release to download when one must be fetched.
//! * `MOLCHANICA_BOLTZ_PYTHON`       — Python version passed to `uv venv` (default `3.12`).
//! * `MOLCHANICA_BOLTZ_INSTALL_ARGS` — extra args appended to `uv pip install` (e.g. a CUDA index).
//! * `MOLCHANICA_BOLTZ_INPROCESS`    — when truthy, try the PyO3 in-process runner first.

use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

/// `uv` release downloaded when no `uv` is otherwise available. Overridable with
/// `MOLCHANICA_UV_VERSION`. Kept as a pinned default so we fetch a known-good binary rather than
/// whatever "latest" happens to be.
const DEFAULT_UV_VERSION: &str = "0.9.7";

/// Python version requested from `uv` for the managed virtual environment.
const DEFAULT_PYTHON_VERSION: &str = "3.12";

/// Written after a successful provision so we can cheaply tell that the runtime is ready.
const MARKER_FILE: &str = ".provisioned";

/// A ready-to-use, isolated Boltz environment.
pub(super) struct BoltzRuntime {
    /// The venv's Python interpreter.
    python: PathBuf,
    /// The venv's `boltz` console-script launcher.
    boltz: PathBuf,
}

impl BoltzRuntime {
    /// Site-packages directories of the managed venv (`purelib` and `platlib`).
    ///
    /// Only needed by the opt-in in-process runner; it shells out to the managed interpreter once,
    /// so it is not called on the common subprocess path.
    pub fn site_packages(&self) -> io::Result<Vec<String>> {
        let output = Command::new(&self.python)
            .arg("-c")
            .arg(
                "import json, sysconfig; p = sysconfig.get_paths(); \
                 print(json.dumps([p.get('purelib'), p.get('platlib')]))",
            )
            .output()
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("unable to query the managed Python's site-packages: {error}"),
                )
            })?;

        if !output.status.success() {
            return Err(io::Error::other(
                "managed Python failed to report its site-packages",
            ));
        }

        let parsed: Vec<Option<String>> = serde_json::from_slice(&output.stdout)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

        let mut dirs = Vec::new();
        for dir in parsed.into_iter().flatten() {
            if !dirs.contains(&dir) && Path::new(&dir).is_dir() {
                dirs.push(dir);
            }
        }
        Ok(dirs)
    }

    /// Run a Boltz prediction by launching the managed venv's `boltz` executable.
    ///
    /// Standard streams are inherited so the (lengthy) prediction progress is visible.
    pub fn predict(
        &self,
        input_path: &Path,
        output_dir: &Path,
        use_msa_server: bool,
    ) -> io::Result<()> {
        let mut command = Command::new(&self.boltz);
        command
            .arg("predict")
            .arg(input_path)
            .arg("--out_dir")
            .arg(output_dir);
        if use_msa_server {
            command.arg("--use_msa_server");
        }
        run_step(&mut command, "boltz predict")
    }
}

/// Whether the managed runtime is already provisioned, without doing any work.
///
/// Used for the cheap startup availability check; never provisions or spawns a heavy process.
pub(super) fn runtime_ready() -> bool {
    let Ok(root) = runtime_root() else {
        return false;
    };
    root.join(MARKER_FILE).is_file() && venv_python(&root).is_file() && venv_boltz(&root).is_file()
}

/// Whether the caller asked to try the in-process (PyO3) runner before the subprocess path.
pub(super) fn in_process_requested() -> bool {
    env::var_os("MOLCHANICA_BOLTZ_INPROCESS")
        .map(|value| {
            let value = value.to_string_lossy();
            let value = value.trim();
            !value.is_empty()
                && !value.eq_ignore_ascii_case("0")
                && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Ensure the isolated Boltz environment exists, provisioning it on first use, and return a handle.
///
/// The first call is expensive: it may download `uv`, have `uv` fetch a CPython build, and install
/// Boltz plus Torch (multiple GB). Subsequent calls are near-instant.
pub(super) fn ensure() -> io::Result<BoltzRuntime> {
    let root = runtime_root()?;
    let python = venv_python(&root);
    let boltz = venv_boltz(&root);

    if root.join(MARKER_FILE).is_file() && python.is_file() && boltz.is_file() {
        return Ok(BoltzRuntime { python, boltz });
    }

    fs::create_dir_all(&root)?;
    println!(
        "[boltz-runtime] Provisioning an isolated Boltz environment under {} (first run only; \
         this downloads Python, Torch, and Boltz and may take several minutes)...",
        root.display()
    );

    let uv = ensure_uv(&root)?;
    let venv_dir = root.join("venv");

    // Create the venv with a Python version Boltz supports; uv fetches it if the host lacks it.
    let python_version =
        env_string("MOLCHANICA_BOLTZ_PYTHON").unwrap_or_else(|| DEFAULT_PYTHON_VERSION.to_string());
    let mut venv_cmd = Command::new(&uv);
    venv_cmd
        .arg("venv")
        .arg("--python")
        .arg(&python_version)
        .arg(&venv_dir);
    run_step(&mut venv_cmd, "uv venv")?;

    // Install Boltz (and its Torch stack) into the venv.
    let mut install_cmd = Command::new(&uv);
    install_cmd
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .arg("boltz");
    if let Some(extra) = env_string("MOLCHANICA_BOLTZ_INSTALL_ARGS") {
        for arg in extra.split_whitespace() {
            install_cmd.arg(arg);
        }
    }
    run_step(&mut install_cmd, "uv pip install boltz")?;

    if !boltz.is_file() {
        return Err(io::Error::other(format!(
            "Boltz install completed but its launcher was not found at {}",
            boltz.display()
        )));
    }

    fs::write(
        root.join(MARKER_FILE),
        format!("schema=1\npython={python_version}\n"),
    )?;
    println!("[boltz-runtime] Boltz environment ready.");

    Ok(BoltzRuntime { python, boltz })
}

/// Resolve the managed runtime root, honoring `MOLCHANICA_BOLTZ_HOME`.
fn runtime_root() -> io::Result<PathBuf> {
    if let Some(dir) = env_string("MOLCHANICA_BOLTZ_HOME") {
        return Ok(PathBuf::from(dir));
    }
    Ok(user_data_dir()?.join("molchanica").join("boltz-runtime"))
}

/// Base per-user data directory for the current platform.
fn user_data_dir() -> io::Result<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);

    #[cfg(target_os = "macos")]
    let base =
        env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"));

    #[cfg(all(unix, not(target_os = "macos")))]
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")));

    base.ok_or_else(|| io::Error::other("unable to determine a per-user data directory"))
}

fn venv_python(root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        root.join("venv").join("Scripts").join("python.exe")
    } else {
        root.join("venv").join("bin").join("python")
    }
}

fn venv_boltz(root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        root.join("venv").join("Scripts").join("boltz.exe")
    } else {
        root.join("venv").join("bin").join("boltz")
    }
}

/// Locate a usable `uv`, downloading a pinned release into `root/bin` if necessary.
fn ensure_uv(root: &Path) -> io::Result<PathBuf> {
    // 1. Explicit override.
    if let Some(path) = env_string("MOLCHANICA_UV") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "MOLCHANICA_UV points at {}, which does not exist",
                path.display()
            ),
        ));
    }

    // 2. A previously downloaded copy.
    let bin_dir = root.join("bin");
    let downloaded = bin_dir.join(uv_exe_name());
    if downloaded.is_file() {
        return Ok(downloaded);
    }

    // 3. `uv` on PATH.
    if Command::new("uv").arg("--version").output().is_ok() {
        return Ok(PathBuf::from("uv"));
    }

    // 4. Download the pinned release.
    fs::create_dir_all(&bin_dir)?;
    download_uv(&bin_dir)
}

/// Download and extract a pinned `uv` release into `bin_dir`, returning the extracted binary path.
fn download_uv(bin_dir: &Path) -> io::Result<PathBuf> {
    let version =
        env_string("MOLCHANICA_UV_VERSION").unwrap_or_else(|| DEFAULT_UV_VERSION.to_string());
    let asset = uv_release_asset()?;
    let url = format!("https://github.com/astral-sh/uv/releases/download/{version}/{asset}");

    println!("[boltz-runtime] Downloading uv {version} from {url}");
    // NOTE(hardening): this downloads and executes a third-party binary. A future improvement is to
    // verify it against the published SHA-256 before use; set MOLCHANICA_UV to a vetted copy to
    // bypass the download entirely.
    let archive = bin_dir.join(&asset);
    download_file(&url, &archive)?;

    let extract_dir = bin_dir.join("uv-extract");
    let _ = fs::remove_dir_all(&extract_dir);
    fs::create_dir_all(&extract_dir)?;
    extract_archive(&archive, &extract_dir)?;

    let extracted = find_file(&extract_dir, uv_exe_name()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "uv binary not found inside the downloaded release archive",
        )
    })?;

    let dest = bin_dir.join(uv_exe_name());
    fs::copy(&extracted, &dest)?;
    let _ = fs::remove_dir_all(&extract_dir);
    let _ = fs::remove_file(&archive);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dest, perms)?;
    }

    Ok(dest)
}

fn uv_exe_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "uv.exe"
    } else {
        "uv"
    }
}

/// Release asset name for the current platform, matching Astral's `uv` release naming.
fn uv_release_asset() -> io::Result<String> {
    let target = if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        "aarch64-pc-windows-msvc"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "aarch64-apple-darwin"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "aarch64-unknown-linux-gnu"
    } else {
        return Err(io::Error::other(
            "no known uv release for this platform; install uv manually and set MOLCHANICA_UV",
        ));
    };

    Ok(if cfg!(target_os = "windows") {
        format!("uv-{target}.zip")
    } else {
        format!("uv-{target}.tar.gz")
    })
}

/// Download `url` to `dest` using `curl`, falling back to PowerShell on Windows.
fn download_file(url: &str, dest: &Path) -> io::Result<()> {
    let mut curl = Command::new("curl");
    curl.arg("-fL")
        .arg("--retry")
        .arg("3")
        .arg("-o")
        .arg(dest)
        .arg(url);
    if run_step(&mut curl, "curl download").is_ok() {
        return Ok(());
    }

    if cfg!(target_os = "windows") {
        let mut ps = Command::new("powershell");
        ps.arg("-NoProfile").arg("-Command").arg(format!(
            "Invoke-WebRequest -Uri '{url}' -OutFile '{}'",
            dest.display()
        ));
        return run_step(&mut ps, "PowerShell download");
    }

    Err(io::Error::other(format!("failed to download {url}")))
}

/// Extract a `.zip` (Windows) or `.tar.gz` (Unix) archive into `dest_dir`.
fn extract_archive(archive: &Path, dest_dir: &Path) -> io::Result<()> {
    let name = archive.to_string_lossy();
    if name.ends_with(".zip") {
        if cfg!(target_os = "windows") {
            let mut ps = Command::new("powershell");
            ps.arg("-NoProfile").arg("-Command").arg(format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                archive.display(),
                dest_dir.display()
            ));
            return run_step(&mut ps, "Expand-Archive");
        }
        let mut unzip = Command::new("unzip");
        unzip.arg("-o").arg(archive).arg("-d").arg(dest_dir);
        return run_step(&mut unzip, "unzip");
    }

    // tar handles .tar.gz on macOS, Linux, and modern Windows (bsdtar).
    let mut tar = Command::new("tar");
    tar.arg("-xzf").arg(archive).arg("-C").arg(dest_dir);
    run_step(&mut tar, "tar extract")
}

/// Recursively search `dir` for a file whose name equals `name`.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

/// Run a provisioning subprocess with inherited stdio, mapping failure to a clear error.
fn run_step(command: &mut Command, context: &str) -> io::Result<()> {
    let status = command.status().map_err(|error| {
        io::Error::new(error.kind(), format!("failed to start {context}: {error}"))
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("{context} failed with {status}")))
    }
}

/// Read an environment variable as a non-empty `String`.
fn env_string(key: &str) -> Option<String> {
    env::var_os(key)
        .map(|value| value.to_string_lossy().into_owned())
        .filter(|value| !value.trim().is_empty())
}
