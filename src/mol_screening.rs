//! Small molecule screening. For ingesting large numbers of small molecules,
//! and filtering them based on criteria. For example, how well they align to other molecules,
//! or which ones might bind to a protein's pocket [receptor], and if so, with what affinity.
//!
//! We bypass parts of our molecule loading algorithm for speed: E.g. skipping inferring partial charge
//! ff type, characterization, and other properties that are not required.

// todo: Pre-categorizie small mols, and set this up in a DB. E.g. by PubChem ID

use std::{
    io,
    path::{Path, PathBuf},
    time::Instant,
};

use bio_files::{Mol2, Sdf};

use crate::{
    mol_alignment::{RING_ALIGN_ROT_COUNT_QUICK, make_initial_alignment},
    molecules::small::MoleculeSmall,
};

// We load molecules from disk in batches, to prevent using too much memory. We use
// atom count as a proxy; better than molecule count, but perhaps not as regular as bytes.
pub const MOL_CACHE_SIZE_ATOM_COUNT: u32 = 1_000_000;

// Every this many mols loaded and screened, pritn a status update.
const LOAD_STATUS_RATIO: usize = 10_000;

/// Screen small molecules by matching to a template. This is a cheap procedure that can be run
/// prior to a more careful screening or alignment.
pub fn screen_by_alignment(
    mol_template: &MoleculeSmall,
    mols_query: &[MoleculeSmall],
    score_thresh: f32,
    size_diff_thresh: f32,
) -> Vec<(usize, f32)> {
    // todo: You may need to load in chunks from disk if the set is large, but Mol2 and SDF files
    // todo are small, so maybe skip that for now.

    let t_atom_len = mol_template.common.atoms.len();

    let start = Instant::now();

    let mut rejected_size = 0;

    let mut res = Vec::new();
    for (i, mol_q) in mols_query.iter().enumerate() {
        let q_atom_len = mol_q.common.atoms.len();

        // Quick rejection based on size difference.
        if (t_atom_len as f32 - q_atom_len as f32).abs() / t_atom_len as f32 > size_diff_thresh {
            rejected_size += 1;
            continue;
        }

        let init_alignments =
            make_initial_alignment(mol_template, &mol_q, RING_ALIGN_ROT_COUNT_QUICK);

        // Assume sorted by score already.
        if !init_alignments.is_empty() && init_alignments[0].score <= score_thresh {
            res.push((i, init_alignments[0].score));
        }
    }

    res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let elapsed = start.elapsed().as_millis();
    println!(
        "Screened {} molecules in {elapsed} ms. Passed: {} Rejected from size: {rejected_size}",
        mols_query.len(),
        res.len()
    );

    res
}

/// Collect all SDF and Mol2 file paths from a directory tree into a sorted list.
///
/// Sorting ensures deterministic, consistent ordering across batches and runs. Call this
/// once at the start of a screening session, then pass the result (or slices of it) to
/// [`load_mol_batch`] repeatedly.
pub fn collect_mol_files(path: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut dirs_to_visit = vec![path.to_path_buf()];

    while let Some(dir) = dirs_to_visit.pop() {
        for entry in dir.read_dir()? {
            let entry = entry?;
            let p = entry.path();
            let ty = entry.file_type()?;

            if ty.is_dir() {
                dirs_to_visit.push(p);
                continue;
            }

            if !ty.is_file() {
                continue;
            }

            let ext = match p.extension().and_then(|e| e.to_str()) {
                Some(ext) => ext.to_ascii_lowercase(),
                None => continue,
            };

            match ext.as_str() {
                "sdf" | "mol2" => files.push(p),
                _ => {}
            }
        }
    }

    // Sort for deterministic ordering; read_dir order is not guaranteed.
    files.sort();

    println!("Found {} molecule files in {path:?}", files.len());
    Ok(files)
}

/// Load a batch of molecules from a pre-collected file list, up to the atom-count budget.
///
/// - Reads from the beginning of `files` until [`MOL_CACHE_SIZE_ATOM_COUNT`] atoms are loaded
///   or the slice is exhausted.
/// - Returns `(molecules, files_consumed)`. `files_consumed` includes files that failed to
///   parse (which are skipped with a warning), so callers can advance their file offset by
///   exactly this count to get a non-overlapping next batch.
///
/// Typical usage with [`collect_mol_files`]:
/// ```ignore
/// let files = collect_mol_files(&dir)?;
/// let mut offset = 0;
/// while offset < files.len() {
///     let (mols, consumed) = load_mol_batch(&files[offset..])?;
///     offset += consumed;
///     // … process mols …
/// }
/// ```
pub fn load_mol_batch(files: &[PathBuf]) -> io::Result<(Vec<MoleculeSmall>, usize)> {
    let start = Instant::now();

    let mut result = Vec::new();
    let mut atoms_loaded: u32 = 0;
    let mut files_consumed = 0;

    for path in files {
        // Check the atom budget before loading the next file.
        if atoms_loaded >= MOL_CACHE_SIZE_ATOM_COUNT {
            break;
        }

        files_consumed += 1;

        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => ext.to_ascii_lowercase(),
            None => continue,
        };

        // Use a closure so that `?` inside converts errors uniformly to io::Error.
        let mol_result: io::Result<MoleculeSmall> = (|| {
            Ok(match ext.as_str() {
                "sdf" => Sdf::load(path)?.try_into()?,
                "mol2" => Mol2::load(path)?.try_into()?,
                _ => unreachable!(),
            })
        })();

        let mut mol = match mol_result {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "Warning: Skipping {:?}: {e}",
                    path.file_name().unwrap_or_default()
                );
                continue;
            }
        };

        mol.common.update_path(path);
        mol.update_characterization();
        atoms_loaded += mol.common.atoms.len() as u32;
        result.push(mol);

        if result.len().is_multiple_of(LOAD_STATUS_RATIO) {
            println!(
                "Loading progress: {} mols, {atoms_loaded} atoms",
                result.len()
            );
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!(
        "Loaded {} molecules ({atoms_loaded} atoms) from {files_consumed} files in {elapsed} ms",
        result.len()
    );

    Ok((result, files_consumed))
}

/// Load `SDF` and `Mol2` files in a directory and its sub-dirs into memory.
///
/// This is a convenience wrapper around [`collect_mol_files`] + [`load_mol_batch`] for
/// callers that only need a single batch. For streaming through large directories use
/// those two functions directly to avoid re-traversing the directory on every call.
pub fn load_mols(path: &Path, skip: usize) -> io::Result<(Vec<MoleculeSmall>, bool)> {
    let files = collect_mol_files(path)?;
    let remaining = if skip < files.len() {
        &files[skip..]
    } else {
        &[]
    };
    let (mols, consumed) = load_mol_batch(remaining)?;
    let has_more = skip + consumed < files.len();
    Ok((mols, has_more))
}
