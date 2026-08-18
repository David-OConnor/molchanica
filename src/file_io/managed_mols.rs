//! Persistent, application-managed source files for molecules that do not originate on disk.
//!
//! `OpenHistory` is deliberately path-based. Downloaded and generated molecules therefore get a
//! durable source file under the preferences directory and use that path just like a user-opened
//! molecule. Keeping the source payload outside the preferences file also avoids making that file
//! large and lets the normal molecule parsers handle session restoration.

use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

use bio_files::{Sdf, SdfFormat};
use mol_defs::molecules::small::MoleculeSmall;
use serde::{Deserialize, Serialize};

const MANAGED_MOLS_DIR: &str = "managed_molecules";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ManagedMolProvider {
    Rcsb,
    Pubchem,
    Drugbank,
    Chebi,
    Geostd,
    Smiles,
    BuiltIn,
}

impl ManagedMolProvider {
    fn as_str(self) -> &'static str {
        match self {
            Self::Rcsb => "rcsb",
            Self::Pubchem => "pubchem",
            Self::Drugbank => "drugbank",
            Self::Chebi => "chebi",
            Self::Geostd => "geostd",
            Self::Smiles => "smiles",
            Self::BuiltIn => "built-in",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ManagedMolManifest {
    pub provider: String,
    pub query: String,
    pub main_file: String,
    #[serde(default)]
    pub pubchem_cid: Option<u32>,
    pub frcmod_file: Option<String>,
    pub lib_file: Option<String>,
}

pub(crate) fn managed_mols_dir(prefs_dir: &Path) -> PathBuf {
    prefs_dir.join(MANAGED_MOLS_DIR)
}

/// A compact key for generated structures whose full input is unsuitable for a filename.
pub(crate) fn text_key(text: &str) -> String {
    // FNV-1a is small and deterministic across Rust versions. This is a cache key, not a security
    // boundary; path confinement is enforced independently below.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn safe_segment(value: &str) -> String {
    let mut result = String::with_capacity(value.len().min(64));

    for ch in value.chars().take(64) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push('_');
        }
    }

    if result.is_empty() {
        "molecule".to_owned()
    } else {
        result
    }
}

fn entry_paths(
    prefs_dir: &Path,
    provider: ManagedMolProvider,
    key: &str,
    extension: &str,
) -> (PathBuf, PathBuf, String) {
    let stem = format!("{}-{}", provider.as_str(), safe_segment(key));
    let entry_dir = managed_mols_dir(prefs_dir).join(&stem);
    let main_file = format!("{stem}.{extension}");
    let main_path = entry_dir.join(&main_file);
    (entry_dir, main_path, main_file)
}

fn managed_entry_dir(prefs_dir: &Path, path: &Path) -> Option<PathBuf> {
    let root = managed_mols_dir(prefs_dir);
    let relative = path.strip_prefix(&root).ok()?;
    let mut components = relative.components();
    let Some(Component::Normal(entry_name)) = components.next() else {
        return None;
    };
    let Some(Component::Normal(_file_name)) = components.next() else {
        return None;
    };

    if components.next().is_some() {
        return None;
    }

    Some(root.join(entry_name))
}

fn temp_path(path: &Path) -> io::Result<PathBuf> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid managed filename"))?;

    Ok(path.with_file_name(format!(".{filename}.{}.tmp", std::process::id())))
}

fn write_atomically(path: &Path, write: impl FnOnce(&Path) -> io::Result<()>) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Managed molecule path has no parent",
        )
    })?;
    fs::create_dir_all(parent)?;

    let temp = temp_path(path)?;
    if let Err(error) = write(&temp) {
        let _ = fs::remove_file(temp);
        return Err(error);
    }

    match fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        // Windows does not replace an existing destination with `rename`. Keep the old file in
        // place until the new payload has been completely written, then update it in-place.
        Err(_) if path.exists() => {
            let data = fs::read(&temp)?;
            fs::write(path, data)?;
            let _ = fs::remove_file(temp);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(temp);
            Err(error)
        }
    }
}

fn write_manifest(entry_dir: &Path, manifest: &ManagedMolManifest) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_atomically(&entry_dir.join(MANIFEST_FILE), |path| {
        fs::write(path, bytes)
    })
}

pub(crate) fn store_text(
    prefs_dir: &Path,
    provider: ManagedMolProvider,
    key: &str,
    query: &str,
    extension: &str,
    contents: &str,
) -> io::Result<PathBuf> {
    let (entry_dir, main_path, main_file) = entry_paths(prefs_dir, provider, key, extension);

    write_atomically(&main_path, |path| fs::write(path, contents))?;
    write_manifest(
        &entry_dir,
        &ManagedMolManifest {
            provider: provider.as_str().to_owned(),
            query: query.to_owned(),
            main_file,
            pubchem_cid: None,
            frcmod_file: None,
            lib_file: None,
        },
    )?;

    Ok(main_path)
}

pub(crate) fn store_sdf(
    prefs_dir: &Path,
    provider: ManagedMolProvider,
    key: &str,
    query: &str,
    sdf: &Sdf,
) -> io::Result<PathBuf> {
    let (entry_dir, main_path, main_file) = entry_paths(prefs_dir, provider, key, "sdf");

    write_atomically(&main_path, |path| sdf.save(path, SdfFormat::V2000))?;
    write_manifest(
        &entry_dir,
        &ManagedMolManifest {
            provider: provider.as_str().to_owned(),
            query: query.to_owned(),
            main_file,
            pubchem_cid: None,
            frcmod_file: None,
            lib_file: None,
        },
    )?;

    Ok(main_path)
}

pub(crate) fn store_geostd(
    prefs_dir: &Path,
    ident: &str,
    mol2: &str,
    pubchem_cid: Option<u32>,
    frcmod: Option<&str>,
    lib: Option<&str>,
) -> io::Result<PathBuf> {
    let provider = ManagedMolProvider::Geostd;
    let (entry_dir, main_path, main_file) = entry_paths(prefs_dir, provider, ident, "mol2");
    let stem = main_path
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid GeoStd filename"))?;

    write_atomically(&main_path, |path| fs::write(path, mol2))?;

    let frcmod_file = if let Some(contents) = frcmod {
        let filename = format!("{stem}.frcmod");
        write_atomically(&entry_dir.join(&filename), |path| fs::write(path, contents))?;
        Some(filename)
    } else {
        None
    };

    let lib_file = if let Some(contents) = lib {
        let filename = format!("{stem}.lib");
        write_atomically(&entry_dir.join(&filename), |path| fs::write(path, contents))?;
        Some(filename)
    } else {
        None
    };

    write_manifest(
        &entry_dir,
        &ManagedMolManifest {
            provider: provider.as_str().to_owned(),
            query: ident.to_owned(),
            main_file,
            pubchem_cid,
            frcmod_file,
            lib_file,
        },
    )?;

    Ok(main_path)
}

/// Rewrite a managed molecule's source file from the molecule we hold in memory, so data gained
/// after the download — e.g. the ChEBI and PDBe accessions "Load all idents" resolves — is still
/// present on the next run. Identifiers ride along in the file's metadata; see
/// `MoleculeSmall::metadata_with_ids_pocket`.
///
/// These files are our own cached copies rather than files the user chose, so updating one in
/// place is safe. Molecules the user has saved themselves, and managed files in formats we don't
/// write back (e.g. RCSB CIFs), are left alone; both return `false`.
pub(crate) fn update_managed_mol(prefs_dir: &Path, mol: &MoleculeSmall) -> io::Result<bool> {
    let Some(path) = &mol.common.path else {
        return Ok(false);
    };

    if !is_managed_path(prefs_dir, path) {
        return Ok(false);
    }

    let extension = path.extension().unwrap_or_default().to_ascii_lowercase();
    match extension.to_str().unwrap_or_default() {
        "sdf" | "mol" => {
            let sdf = mol.to_sdf();
            write_atomically(path, |temp| sdf.save(temp, SdfFormat::V2000))?;
        }
        "mol2" => {
            let mol2 = mol.to_mol2();
            write_atomically(path, |temp| mol2.save(temp))?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

pub(crate) fn is_managed_path(prefs_dir: &Path, path: &Path) -> bool {
    managed_entry_dir(prefs_dir, path).is_some()
}

fn is_single_filename(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

pub(crate) fn read_manifest(
    prefs_dir: &Path,
    main_path: &Path,
) -> io::Result<Option<ManagedMolManifest>> {
    let root = managed_mols_dir(prefs_dir);
    if !main_path.starts_with(&root) {
        return Ok(None);
    }

    let Some(entry_dir) = managed_entry_dir(prefs_dir, main_path) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid managed molecule path",
        ));
    };
    let manifest_path = entry_dir.join(MANIFEST_FILE);
    let data = fs::read(&manifest_path)?;
    let manifest: ManagedMolManifest = serde_json::from_slice(&data)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    if !is_single_filename(&manifest.main_file)
        || manifest
            .frcmod_file
            .as_deref()
            .is_some_and(|value| !is_single_filename(value))
        || manifest
            .lib_file
            .as_deref()
            .is_some_and(|value| !is_single_filename(value))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Managed molecule manifest contains an invalid filename",
        ));
    }

    if main_path.file_name().and_then(|name| name.to_str()) != Some(manifest.main_file.as_str()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Managed molecule manifest does not match its source file",
        ));
    }

    Ok(Some(manifest))
}

pub(crate) fn remove_entry(prefs_dir: &Path, main_path: &Path) -> io::Result<()> {
    let root = managed_mols_dir(prefs_dir);
    if !main_path.starts_with(&root) {
        return Ok(());
    }

    let Some(entry_dir) = managed_entry_dir(prefs_dir, main_path) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Refusing to remove an invalid managed molecule directory",
        ));
    };

    match fs::remove_dir_all(entry_dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(crate) fn cleanup_orphans(prefs_dir: &Path, referenced: &[PathBuf]) -> io::Result<()> {
    let root = managed_mols_dir(prefs_dir);
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };

    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let entry_path = entry.path();
        if referenced.iter().any(|path| path.starts_with(&entry_path)) {
            continue;
        }
        fs::remove_dir_all(entry_path)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "molchanica-managed-mols-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn managed_text_round_trip_and_manifest_are_confined() {
        let prefs_dir = test_dir("round-trip");
        let path = store_text(
            &prefs_dir,
            ManagedMolProvider::Rcsb,
            "../1ABC",
            "1ABC",
            "cif",
            "data_1ABC",
        )
        .unwrap();

        assert!(is_managed_path(&prefs_dir, &path));
        assert_eq!(fs::read_to_string(&path).unwrap(), "data_1ABC");
        assert!(!path.to_string_lossy().contains(".."));

        let manifest = read_manifest(&prefs_dir, &path).unwrap().unwrap();
        assert_eq!(manifest.provider, "rcsb");
        assert_eq!(manifest.query, "1ABC");

        remove_entry(&prefs_dir, &path).unwrap();
        assert!(!path.exists());
        let _ = fs::remove_dir_all(prefs_dir);
    }

    #[test]
    fn geostd_bundle_retains_companions_and_orphan_cleanup_respects_history() {
        let prefs_dir = test_dir("geostd");
        let path = store_geostd(
            &prefs_dir,
            "ATP",
            "@<TRIPOS>MOLECULE",
            Some(5957),
            Some("FRCMOD DATA"),
            None,
        )
        .unwrap();

        let manifest = read_manifest(&prefs_dir, &path).unwrap().unwrap();
        assert_eq!(manifest.provider, "geostd");
        assert_eq!(manifest.pubchem_cid, Some(5957));
        let frcmod_file = manifest.frcmod_file.unwrap();
        assert!(path.parent().unwrap().join(frcmod_file).exists());

        cleanup_orphans(&prefs_dir, std::slice::from_ref(&path)).unwrap();
        assert!(path.exists());
        cleanup_orphans(&prefs_dir, &[]).unwrap();
        assert!(!path.exists());
        let _ = fs::remove_dir_all(prefs_dir);
    }

    #[test]
    fn invalid_managed_paths_cannot_escape_the_cache_root() {
        let prefs_dir = test_dir("path-confinement");
        let outside_dir = prefs_dir.join("outside");
        fs::create_dir_all(&outside_dir).unwrap();
        let sentinel = outside_dir.join("sentinel.txt");
        fs::write(&sentinel, "keep").unwrap();

        let invalid_path = managed_mols_dir(&prefs_dir)
            .join("..")
            .join("outside")
            .join("molecule.sdf");
        assert!(!is_managed_path(&prefs_dir, &invalid_path));
        assert!(remove_entry(&prefs_dir, &invalid_path).is_err());
        assert!(sentinel.exists());

        let _ = fs::remove_dir_all(prefs_dir);
    }

    #[test]
    fn text_keys_are_stable_and_input_sensitive() {
        assert_eq!(text_key("CCO"), "0b783019aa3ace44");
        assert_ne!(text_key("CCO"), text_key("CCC"));
    }
}
