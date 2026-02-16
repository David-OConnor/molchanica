//! Small molecule screening. For ingesting large numbers of small molecules,
//! and filtering them based on criteria. For example, how well they align to other molecules,
//! or which ones might bind to a protein's pocket [receptor], and if so, with what affinity.
//!
//! We bypass parts of our molecule loading algorithm for speed: E.g. skipping inferring partial charge
//! ff type, characterization, and other properties that are not required.

use std::{io, path::Path, time::Instant};

use bio_files::{Mol2, Sdf};

use crate::{
    mol_alignment::{RING_ALIGN_ROT_COUNT_QUICK, make_initial_alignment},
    molecules::small::MoleculeSmall,
};

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

/// Load all `SDF` and `Mol2` files in a directory and its sub-dirs into memory. Bypass our normal loading pipeline,
/// which includes computation to load FF type, partial charge, SMILES,
/// camera considerations etc.
pub fn load_mols(path: &Path) -> io::Result<Vec<MoleculeSmall>> {
    println!("Loading molecules from path {path:?}...");
    let start = Instant::now();

    let mut result = Vec::new();

    let mut dirs_to_visit = vec![path.to_path_buf()];

    // todo: Limit the num loaded to memory
    let max_val = 6_000; // todo temp
    let mut loaded: u32 = 0;

    while let Some(dir) = dirs_to_visit.pop() {
        for entry in dir.read_dir()? {
            if loaded >= max_val {
                break;
            }

            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;

            if ty.is_dir() {
                dirs_to_visit.push(path);
                continue;
            }

            if !ty.is_file() {
                continue;
            }

            let ext = match path.extension().and_then(|e| e.to_str()) {
                Some(ext) => ext.to_ascii_lowercase(),
                None => continue,
            };

            let mut mol: MoleculeSmall = match ext.as_str() {
                "sdf" => Sdf::load(&path)?.try_into()?,
                "mol2" => Mol2::load(&path)?.try_into()?,
                _ => continue,
            };

            // todo RMed for now.
            // This is fast, and [partly] used in our screening workflows.
            mol.update_characterization();

            result.push(mol);
            loaded += 1;

            if loaded.is_multiple_of(2_000) {
                println!("Loading progress: {loaded} mols");
            }
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!(
        "Loaded {} molecules from disk in {elapsed} ms",
        result.len()
    );

    Ok(result)
}
