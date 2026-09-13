//! Where Molchanica keeps its data, and how a tool is resolved to a file on disk.

use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use super::registry::{Tool, ToolKind, ToolSpec};

/// The single directory Molchanica keeps everything it writes in.
///
/// The preferences file, `managed_molecules/`, `gpu_cache/`, `process_executables/`, and
/// `datasets/` all live here, beside the executable itself, so an install is one folder that can
/// be found, backed up, or deleted in one place instead of data scattered across several per-user
/// OS locations. Both setup scripts install to somewhere the user can write for exactly this
/// reason: `%LOCALAPPDATA%\Programs\Molchanica` on Windows, `~/molchanica` on Linux.
///
/// Resolution order:
///
/// 1. `MOLCHANICA_DATA_DIR`, for putting multi-GB tool installs on another drive.
/// 2. The executable's own directory.
/// 3. The working directory, when the executable is in a Cargo build tree. `target/` is the wrong
///    place for this: `cargo clean` would delete multi-GB tool installs, and debug and release
///    builds would each keep a separate copy. The checkout keeps the one-folder property.
/// 4. The platform per-user data directory, as a last resort for an install into somewhere the
///    process cannot write, such as `Program Files` or a packaged `/usr/bin`.
///
/// Resolved once per run: the answer involves probing the filesystem, and it must not change
/// underneath a running session.
pub fn data_root() -> Option<PathBuf> {
    static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

    ROOT.get_or_init(|| {
        let root = resolve_data_root()?;
        // The prefs file and the tool installers all assume this exists.
        if let Err(error) = fs::create_dir_all(&root) {
            eprintln!(
                "Unable to create the data directory {}: {error}",
                root.display()
            );
            return None;
        }
        Some(root)
    })
    .clone()
}

/// [`data_root`], as an error rather than an `Option`, for the code paths that cannot continue
/// without it.
pub fn require_data_root() -> io::Result<PathBuf> {
    data_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "unable to determine Molchanica's data directory",
        )
    })
}

/// `<data root>/process_executables`: managed installs, bio_tools adapter runs, and their results.
pub fn process_executables_dir() -> io::Result<PathBuf> {
    Ok(require_data_root()?.join("process_executables"))
}

/// `<data root>/process_executables/python_envs/<slug>`: where a managed uv environment lives.
pub(super) fn managed_venv_dir(data_root: &Path, slug: &str) -> PathBuf {
    data_root
        .join("process_executables")
        .join("python_envs")
        .join(slug)
}

fn resolve_data_root() -> Option<PathBuf> {
    if let Some(configured) = env::var_os("MOLCHANICA_DATA_DIR") {
        return Some(PathBuf::from(configured));
    }

    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));

    let candidate = match exe_dir {
        Some(dir) if in_cargo_build_dir(&dir) => env::current_dir().ok(),
        other => other,
    };

    if let Some(dir) = candidate
        && is_writable(&dir)
    {
        return Some(dir);
    }

    platform_data_dir()
}

/// Whether `dir` is a Cargo build output directory: `.../target[/<triple>]/{debug,release}`.
fn in_cargo_build_dir(dir: &Path) -> bool {
    let is_profile_dir = dir
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name == "debug" || name == "release");

    // Two levels covers both the plain layout and the one `--target <triple>` produces. Searching
    // further up would match an unrelated directory that happens to be called `target`.
    is_profile_dir
        && dir
            .ancestors()
            .skip(1)
            .take(2)
            .any(|ancestor| ancestor.file_name().and_then(OsStr::to_str) == Some("target"))
}

/// Whether files can actually be created in `dir`, creating the directory itself if needed.
///
/// Probed rather than inferred from metadata: on Windows a read-only location such as
/// `Program Files` reports nothing useful until the write is attempted.
fn is_writable(dir: &Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }

    // Named per-process so two copies running at once cannot delete each other's probe.
    let probe = dir.join(format!(".molchanica-write-probe-{}", std::process::id()));
    match fs::File::create(&probe) {
        Ok(_) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// The OS convention for per-user application data: `%LOCALAPPDATA%` on Windows,
/// `~/Library/Application Support` on macOS, and `$XDG_DATA_HOME` (defaulting to
/// `~/.local/share`) elsewhere.
///
/// Only used as a fallback, and to find data left behind by versions that always wrote here. See
/// [`migrate_legacy_data`].
fn platform_data_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);

    #[cfg(target_os = "macos")]
    let base = home_directory().map(|home| home.join("Library/Application Support"));

    #[cfg(all(unix, not(target_os = "macos")))]
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| home_directory().map(|home| home.join(".local/share")));

    base.map(|base| base.join("molchanica"))
}

/// Top-level folders under the data root that earlier versions kept in the per-user data
/// directory instead.
const MIGRATED_DIRS: &[&str] = &["process_executables", "datasets"];

/// Move data written by a version that used the per-user data directory into the data root.
///
/// Without this, an upgrade silently loses every installed tool and offers to download several GB
/// again. A rename is instantaneous within a volume, which is the normal case — both paths are
/// under `%LOCALAPPDATA%` on Windows and under `$HOME` on Linux. Across volumes it fails, and
/// copying gigabytes at startup is not something to do unasked, so we say where the files are and
/// leave them alone.
///
/// Call once, before anything reads the data root.
pub fn migrate_legacy_data() {
    let (Some(root), Some(legacy)) = (data_root(), platform_data_dir()) else {
        return;
    };
    if legacy == root || !legacy.is_dir() {
        return;
    }

    for name in MIGRATED_DIRS {
        let from = legacy.join(name);
        let to = root.join(name);
        if !from.is_dir() || to.exists() {
            continue;
        }

        match fs::rename(&from, &to) {
            Ok(()) => println!("Moved {} to {}", from.display(), to.display()),
            Err(error) => {
                eprintln!(
                    "Could not move {} to {}: {error}",
                    from.display(),
                    to.display()
                );
                eprintln!(
                    "Move it there yourself, or set MOLCHANICA_DATA_DIR to {}.",
                    legacy.display()
                );
            }
        }
    }

    // Leaves the directory itself behind when anything else is in it.
    let _ = fs::remove_dir(&legacy);
}

pub fn home_directory() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Resolve a tool to an absolute path, or explain what to do about it.
///
/// The override environment variable always wins, then the Molchanica-managed location. Native
/// executables may additionally fall back to `PATH`; Python tools never do. A user who ran our
/// installer must get the uv-managed interpreter we built rather than an unrelated `pip install`.
pub fn find_executable(tool: Tool) -> io::Result<PathBuf> {
    let spec = tool.spec();

    if let Some(configured) = env::var_os(spec.exe_override_env) {
        let configured = PathBuf::from(configured);
        if configured.is_file() {
            return Ok(configured);
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{} points to {}, but that file does not exist",
                spec.exe_override_env,
                configured.display()
            ),
        ));
    }

    let found = match spec.kind {
        // Deliberately no PATH fallback for either Python kind. Even a console launcher ultimately
        // selects a Python interpreter, and running third-party model code under whatever `python`
        // happens to be first on PATH is how the earlier Boltz-2 and ESMFold integrations broke.
        ToolKind::VenvScript => spec
            .venv_root()
            .and_then(|root| executable_in(&venv_bin(&root), spec.executable)),
        ToolKind::VenvPython => spec
            .venv_root()
            .and_then(|root| executable_in(&venv_bin(&root), "python")),
        ToolKind::Executable => find_native_executable(spec),
    };

    found.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} was not found. {}", spec.name(), spec.install_command()),
        )
    })
}

/// Beside Molchanica (for tools we ship), then in the bundle directory, then on `PATH`.
fn find_native_executable(spec: &ToolSpec) -> Option<PathBuf> {
    if spec.colocated
        && let Some(found) = colocated_executable(spec.executable)
    {
        return Some(found);
    }
    if let Some(root) = spec.bundle_root() {
        for directory in [root.clone(), root.join("bin")] {
            if let Some(found) = executable_in(&directory, spec.executable) {
                return Some(found);
            }
        }
    }
    find_on_path(spec.executable)
}

/// The directory a bundled tool's data files sit in, which several tools need passed to them
/// explicitly (the MPNN checkouts' model parameters).
pub fn bundle_root(tool: Tool) -> io::Result<PathBuf> {
    let spec = tool.spec();
    spec.bundle_root()
        .filter(|root| root.is_dir())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{} is not installed. {}",
                    spec.name(),
                    spec.install_command()
                ),
            )
        })
}

fn venv_bin(root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        root.join("Scripts")
    } else {
        root.join("bin")
    }
}

fn colocated_executable(name: &str) -> Option<PathBuf> {
    let directory = env::current_exe().ok()?.parent()?.to_path_buf();
    executable_in(&directory, name)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .and_then(|path| env::split_paths(&path).find_map(|dir| executable_in(&dir, name)))
}

/// A file named `name` in `directory` that we could plausibly execute.
fn executable_in(directory: &Path, name: &str) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let names = [
        format!("{name}.exe"),
        format!("{name}.cmd"),
        format!("{name}.bat"),
        name.to_owned(),
    ];
    #[cfg(not(target_os = "windows"))]
    let names = [name.to_owned()];

    names
        .into_iter()
        .map(|name| directory.join(name))
        .find(|candidate| candidate.is_file())
}
