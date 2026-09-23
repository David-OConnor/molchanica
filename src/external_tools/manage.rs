//! Installing, measuring, and removing the tools Molchanica manages.

use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

use bio_tools::install::Installer as ToolInstaller;

use super::{
    paths::{managed_venv_dir, process_executables_dir, require_data_root},
    registry::{Tool, ToolKind},
};

/// Install one managed optional tool with the shared `bio_tools` Rust installer.
///
/// This blocks and is therefore intended to be called from a worker thread.
pub fn install(tool: Tool) -> io::Result<()> {
    let spec = tool.spec();
    if !spec.is_supported() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("{} is available on Linux only", spec.name()),
        ));
    }

    if !spec.molchanica_managed {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{} cannot be installed automatically. {}",
                spec.name(),
                spec.install_hint
            ),
        ));
    }

    let recipe = spec.recipe().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no bio_tools install recipe", spec.name()),
        )
    })?;

    let mut installer = ToolInstaller::for_process_executables(process_executables_dir()?)
        .map_err(|error| io::Error::other(error.to_string()))?;
    installer.config.support_root = installer_support_root();

    installer
        .install(recipe)
        .map_err(|error| io::Error::other(error.to_string()))
}

fn installer_support_root() -> Option<PathBuf> {
    let checkout = env::current_dir().ok();
    let release = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf));

    checkout.into_iter().chain(release).find(|root| {
        root.join("scripts/convert_mpnn_weights.py").is_file()
            || root.join("convert_mpnn_weights.py").is_file()
    })
}

/// Estimate the space occupied by this tool inside Molchanica's managed data directory.
///
/// `None` means there are no managed files for the tool. This deliberately excludes upstream
/// caches outside Molchanica's data root, and never follows symlinks while measuring.
pub fn managed_disk_usage(tool: Tool) -> io::Result<Option<u64>> {
    if !tool.spec().molchanica_managed {
        return Ok(None);
    }

    let data_root = require_data_root()?;
    let mut found = false;
    let mut bytes = 0_u64;

    for path in managed_install_roots(tool, &data_root) {
        if path.exists() {
            found = true;
            bytes = bytes.saturating_add(path_disk_usage(&path)?);
        }
    }

    Ok(found.then_some(bytes))
}

/// Remove the files Molchanica installed for one managed optional tool.
///
/// Override paths and system installations are intentionally never touched. Every deletion target
/// is reconstructed beneath the data root rather than taken from an environment variable.
pub fn uninstall(tool: Tool) -> io::Result<()> {
    let spec = tool.spec();
    if !spec.molchanica_managed {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{} is managed outside Molchanica and cannot be uninstalled here",
                spec.name()
            ),
        ));
    }

    let data_root = require_data_root()?;
    let roots = managed_install_roots(tool, &data_root);
    if roots
        .iter()
        .any(|path| path == &data_root || !path.starts_with(&data_root))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("refusing to remove an unsafe path for {}", spec.name()),
        ));
    }

    let existing: Vec<_> = roots.into_iter().filter(|path| path.exists()).collect();
    if existing.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no Molchanica-managed files were found for {}", spec.name()),
        ));
    }

    for path in existing {
        let metadata = fs::symlink_metadata(&path)?;
        let result = if metadata.is_dir() && !metadata.file_type().is_symlink() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };

        result.map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("unable to remove {}: {error}", path.display()),
            )
        })?;
    }
    Ok(())
}

/// Every directory a managed install of `tool` writes, reconstructed from the data root.
fn managed_install_roots(tool: Tool, data_root: &Path) -> Vec<PathBuf> {
    let spec = tool.spec();
    let mut roots = Vec::new();

    if matches!(spec.kind, ToolKind::VenvScript | ToolKind::VenvPython) {
        roots.push(managed_venv_dir(data_root, spec.slug()));
    }
    if let Some(subdir) = spec.bundle_subdir {
        roots.push(data_root.join("process_executables").join(subdir));
    }
    roots
}

fn path_disk_usage(path: &Path) -> io::Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        return Ok(metadata.len());
    }

    let mut bytes = 0_u64;
    for entry in fs::read_dir(path)? {
        bytes = bytes.saturating_add(path_disk_usage(&entry?.path())?);
    }
    Ok(bytes)
}
