//! Reading a locally downloaded PDBbind release.
//!
//! [PDBbind+](https://www.pdbbind-plus.org.cn/) is the standard protein–ligand binding-affinity
//! dataset: a curated set of PDB complexes, each with a measured Kd, Ki, or IC50, and each unpacked
//! as a directory holding the protein, the pocket, and the ligand as separate files.
//!
//! This is a reader, not a tool integration. There is nothing to install and no process to run —
//! a PDBbind release is a directory tree, and everything Molchanica wants from it (an entry's
//! affinity, and paths to files it can already open) is a filesystem lookup and one index file to
//! parse. Wrapping that in a subprocess would add a dependency and take away the ability to iterate
//! the whole set, which is what makes it useful for the affinity models in `adme_`.
//!
//! # Licence
//!
//! No licence is granted here. PDBbind+ is distributed by its maintainers under registration, free
//! for academic use, with commercial use requiring a subscription. This reads a copy the user has
//! already obtained under their own agreement, which is why nothing downloads it for them.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!     index/INDEX_refined_data.2020        affinity per entry
//!     refined-set/1a30/1a30_protein.pdb
//!                     /1a30_pocket.pdb
//!                     /1a30_ligand.sdf
//!                     /1a30_ligand.mol2
//! ```

use std::{
    collections::HashMap,
    env, fs, io,
    path::{Path, PathBuf},
};

/// Which part of the release to search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Subset {
    /// The curated subset with the best-quality structures and measurements. The usual default.
    #[default]
    Refined,
    /// Everything, including entries the refined set filtered out.
    General,
    /// The CASF benchmark set, used for scoring-function evaluation.
    Core,
    /// Whichever of the above is present.
    Any,
}

impl Subset {
    pub const ALL: [Self; 4] = [Self::Refined, Self::General, Self::Core, Self::Any];

    pub fn label(self) -> &'static str {
        match self {
            Self::Refined => "Refined set",
            Self::General => "General set",
            Self::Core => "Core set (CASF)",
            Self::Any => "Any configured set",
        }
    }

    /// Directory names a release may use for this subset.
    ///
    /// Several spellings because they vary by release year and by how the set was obtained:
    /// `coreset` (no separator) is how the CASF-2016 benchmark package names it, and the core set
    /// is more commonly obtained that way than as a standalone PDBbind download.
    fn directory_names(self) -> &'static [&'static str] {
        const REFINED: &[&str] = &["refined-set", "refined_set", "refined"];
        const GENERAL: &[&str] = &[
            "general-set",
            "general_set",
            "general",
            "general-set-except-refined",
        ];
        const CORE: &[&str] = &["core-set", "core_set", "core", "coreset"];
        const ANY: &[&str] = &[
            "refined-set",
            "refined_set",
            "refined",
            "general-set",
            "general_set",
            "general",
            "general-set-except-refined",
            "core-set",
            "core_set",
            "core",
            "coreset",
        ];
        match self {
            Self::Refined => REFINED,
            Self::General => GENERAL,
            Self::Core => CORE,
            Self::Any => ANY,
        }
    }
}

/// What was measured. The three are not interchangeable — an IC50 depends on assay conditions in
/// a way a Kd does not — so anything training on this should know which it has.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffinityKind {
    Kd,
    Ki,
    Ic50,
}

impl AffinityKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Kd => "Kd",
            Self::Ki => "Ki",
            Self::Ic50 => "IC50",
        }
    }
}

/// How the measurement relates to the reported value. PDBbind records bounded measurements as
/// `Kd>100mM` and similar; treating those as exact would train a model on values that were never
/// measured.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Relation {
    Exact,
    GreaterThan,
    LessThan,
    Approximate,
}

impl Relation {
    pub fn is_exact(self) -> bool {
        self == Self::Exact
    }
}

/// One entry's measurement, as the index file records it.
#[derive(Clone, Debug, PartialEq)]
pub struct Affinity {
    pub kind: AffinityKind,
    pub relation: Relation,
    /// The dataset's own `-log10(Kd/Ki)`, which is what affinity models are conventionally trained
    /// against. Present even where the concentration could not be parsed.
    pub p_value: f32,
    /// The measurement in molar, derived from the value and unit.
    pub molar: Option<f64>,
    /// Crystallographic resolution in Å. Absent for NMR structures.
    pub resolution: Option<f32>,
    pub year: Option<u16>,
    /// The measurement exactly as written, e.g. `Kd=10mM`.
    pub raw: String,
}

impl Affinity {
    /// Whether this is a clean training target: an exact measurement of a binding constant.
    ///
    /// IC50s are excluded not because they are wrong but because they are not comparable to Kd/Ki
    /// without knowing the assay, and bounded values are excluded because the true affinity lies
    /// somewhere beyond the number recorded.
    pub fn is_regression_quality(&self) -> bool {
        self.relation.is_exact() && self.kind != AffinityKind::Ic50
    }
}

/// One complex in the release.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Lowercase four-character PDB identifier.
    pub pdb_id: String,
    /// The entry's directory.
    pub directory: PathBuf,
    /// The subset directory it was found under, relative to the release root.
    pub subset_directory: String,
    /// Full protein structure.
    pub protein: Option<PathBuf>,
    /// The binding site only, which is what a pocket-based workflow wants.
    pub pocket: Option<PathBuf>,
    /// Ligand as SDF, and as Mol2. Both are formats Molchanica opens directly.
    pub ligand_sdf: Option<PathBuf>,
    pub ligand_mol2: Option<PathBuf>,
    /// From the index files, when the entry appears in one.
    pub affinity: Option<Affinity>,
}

impl Entry {
    /// The ligand file to prefer. SDF first: it carries explicit bond orders, whereas the Mol2
    /// files in PDBbind carry Sybyl atom types that have to be mapped back.
    pub fn ligand(&self) -> Option<&Path> {
        self.ligand_sdf.as_deref().or(self.ligand_mol2.as_deref())
    }

    /// The structure to load: the pocket where one exists, since the full protein is usually far
    /// more than a docking or scoring workflow needs.
    pub fn structure_for_pocket_work(&self) -> Option<&Path> {
        self.pocket.as_deref().or(self.protein.as_deref())
    }
}

/// Where the release is unpacked.
///
/// `MOLCHANICA_PDBBIND_ROOT` overrides it; otherwise `<data root>/datasets/pdbbind`, beside the
/// managed tools rather than inside them, since this is data the user supplied.
pub fn root() -> Option<PathBuf> {
    if let Some(configured) = env::var_os("MOLCHANICA_PDBBIND_ROOT") {
        return Some(PathBuf::from(configured));
    }
    crate::external_tools::data_root().map(|root| root.join("datasets/pdbbind"))
}

/// Whether a release is present, and where.
pub fn installed_root() -> Option<PathBuf> {
    root().filter(|path| path.is_dir())
}

fn require_root() -> io::Result<PathBuf> {
    installed_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "No PDBbind release found at {}. Unpack one there, or set \
                 MOLCHANICA_PDBBIND_ROOT to where it is.",
                root()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "the Molchanica data directory".to_owned())
            ),
        )
    })
}

/// Reject anything that is not a PDB identifier.
///
/// This is also what makes path traversal impossible: an identifier that has passed here cannot
/// contain a separator or a `..`, so joining it onto the release root cannot escape it.
fn normalized_id(pdb_id: &str) -> io::Result<String> {
    let id = pdb_id.trim().to_ascii_lowercase();
    let valid = id.len() == 4
        && id.starts_with(|c: char| c.is_ascii_digit())
        && id.chars().all(|c| c.is_ascii_alphanumeric());
    if !valid {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("'{pdb_id}' is not a four-character PDB identifier"),
        ));
    }
    Ok(id)
}

/// Look one complex up.
///
/// Returns `Ok(None)` when the release is present but has no such entry, which is an ordinary
/// answer rather than an error: most PDB entries are not in PDBbind.
pub fn find(pdb_id: &str, subset: Subset) -> io::Result<Option<Entry>> {
    let root = require_root()?;
    let id = normalized_id(pdb_id)?;
    let index = load_index(&root).unwrap_or_default();

    // Some releases are unpacked with the entry directories directly under the root, without an
    // intervening subset directory.
    let mut candidates = vec![(String::new(), root.join(&id))];
    candidates.extend(
        subset
            .directory_names()
            .iter()
            .map(|name| ((*name).to_owned(), root.join(name).join(&id))),
    );

    for (subset_directory, directory) in candidates {
        if !directory.is_dir() {
            continue;
        }
        return Ok(Some(build_entry(&id, directory, subset_directory, &index)));
    }
    Ok(None)
}

/// Every entry in a subset, for feeding a training pipeline.
///
/// Ordered by identifier so a run is reproducible, and so a train/test split taken over it is
/// stable across machines.
pub fn entries(subset: Subset) -> io::Result<Vec<Entry>> {
    let root = require_root()?;
    let index = load_index(&root).unwrap_or_default();
    let mut found = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for name in subset.directory_names() {
        let directory = root.join(name);
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&directory)?.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(id) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Ok(id) = normalized_id(id) else { continue };
            // The general set in some releases contains the refined entries too; keep the first.
            if !seen.insert(id.clone()) {
                continue;
            }
            found.push(build_entry(&id, path, (*name).to_owned(), &index));
        }
    }

    found.sort_by(|left, right| left.pdb_id.cmp(&right.pdb_id));
    Ok(found)
}

fn build_entry(
    id: &str,
    directory: PathBuf,
    subset_directory: String,
    index: &HashMap<String, Affinity>,
) -> Entry {
    let file = |suffix: &str| -> Option<PathBuf> {
        let path = directory.join(format!("{id}_{suffix}"));
        path.is_file().then_some(path)
    };

    Entry {
        pdb_id: id.to_owned(),
        protein: file("protein.pdb"),
        pocket: file("pocket.pdb"),
        ligand_sdf: file("ligand.sdf"),
        ligand_mol2: file("ligand.mol2"),
        affinity: index.get(id).cloned(),
        directory,
        subset_directory,
    }
}

/// Parse every `index/INDEX_*_data.*` file in the release into one lookup.
///
/// All of them, because a release ships several (refined, general, and often a name-based index),
/// and an entry present in one but not another should still resolve. Later files do not overwrite
/// earlier ones, so the more curated refined index wins where both list an entry.
pub fn load_index(root: &Path) -> io::Result<HashMap<String, Affinity>> {
    let index_directory = root.join("index");
    if !index_directory.is_dir() {
        return Ok(HashMap::new());
    }

    let mut paths: Vec<PathBuf> = fs::read_dir(&index_directory)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("INDEX") && name.contains("_data"))
        })
        .collect();
    // Refined before general, so the curated measurement is the one that lands.
    paths.sort_by_key(|path| {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        (!name.contains("refined"), name)
    });

    let mut index = HashMap::new();
    for path in paths {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            let Some((id, affinity)) = parse_index_line(line) else {
                continue;
            };
            index.entry(id).or_insert(affinity);
        }
    }
    Ok(index)
}

/// Parse one data row of a PDBbind index file.
///
/// ```text
/// # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
/// 2r58  2.00  2007   2.00  Kd=10mM       // 2r58.pdf (MLY)
/// 1qkt  1.90  1999   9.30  Kd=0.5nM      // 1qkt.pdf (ASD)
/// ```
fn parse_index_line(line: &str) -> Option<(String, Affinity)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 5 {
        return None;
    }

    let id = normalized_id(fields[0]).ok()?;
    // "NMR" appears in the resolution column for solution structures.
    let resolution = fields[1].parse::<f32>().ok();
    let year = fields[2].parse::<u16>().ok();
    let p_value = fields[3].parse::<f32>().ok()?;
    let raw = fields[4].to_owned();
    let (kind, relation, molar) = parse_measurement(&raw)?;

    Some((
        id,
        Affinity {
            kind,
            relation,
            p_value,
            molar,
            resolution,
            year,
            raw,
        },
    ))
}

/// Split `Kd=10mM`, `Ki>1.2uM`, `IC50~45nM` into its parts.
fn parse_measurement(raw: &str) -> Option<(AffinityKind, Relation, Option<f64>)> {
    let separator = raw.find(['=', '>', '<', '~'])?;
    let (kind_text, rest) = raw.split_at(separator);
    let kind = match kind_text.to_ascii_lowercase().as_str() {
        "kd" => AffinityKind::Kd,
        "ki" => AffinityKind::Ki,
        "ic50" => AffinityKind::Ic50,
        _ => return None,
    };

    let mut characters = rest.chars();
    let relation = match characters.next()? {
        '=' => Relation::Exact,
        '>' => Relation::GreaterThan,
        '<' => Relation::LessThan,
        '~' => Relation::Approximate,
        _ => return None,
    };
    let value_text = characters.as_str();

    // The unit is the trailing alphabetic run; everything before it is the number.
    let split = value_text
        .find(|c: char| c.is_ascii_alphabetic())
        .unwrap_or(value_text.len());
    let (number, unit) = value_text.split_at(split);
    let molar = number.trim().parse::<f64>().ok().and_then(|value| {
        let scale = match unit.trim().to_ascii_lowercase().as_str() {
            "m" => 1.0,
            "mm" => 1e-3,
            "um" | "µm" => 1e-6,
            "nm" => 1e-9,
            "pm" => 1e-12,
            "fm" => 1e-15,
            _ => return None,
        };
        Some(value * scale)
    });

    Some((kind, relation, molar))
}

impl Default for Affinity {
    fn default() -> Self {
        Self {
            kind: AffinityKind::Kd,
            relation: Relation::Exact,
            p_value: 0.0,
            molar: None,
            resolution: None,
            year: None,
            raw: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_index_rows() {
        let (id, affinity) =
            parse_index_line("2r58  2.00  2007   2.00  Kd=10mM       // 2r58.pdf (MLY)")
                .expect("a data row should parse");

        assert_eq!(id, "2r58");
        assert_eq!(affinity.kind, AffinityKind::Kd);
        assert_eq!(affinity.relation, Relation::Exact);
        assert_eq!(affinity.p_value, 2.00);
        assert_eq!(affinity.molar, Some(1e-2));
        assert_eq!(affinity.resolution, Some(2.00));
        assert_eq!(affinity.year, Some(2007));
        assert!(affinity.is_regression_quality());
    }

    #[test]
    fn skips_comments_and_short_rows() {
        assert!(parse_index_line("# PDB code, resolution, release year").is_none());
        assert!(parse_index_line("").is_none());
        assert!(parse_index_line("1abc 2.0 2001").is_none());
    }

    #[test]
    fn handles_nmr_structures_without_a_resolution() {
        let (_, affinity) = parse_index_line("1a30  NMR   1998   5.00  Ki=10uM  // ref")
            .expect("an NMR row should parse");
        assert_eq!(affinity.resolution, None);
        // Compared with a tolerance: the conversion is a float multiply, so 10 × 1e-6 is not
        // bit-identical to 1e-5.
        assert!((affinity.molar.expect("concentration") - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn distinguishes_bounded_and_indirect_measurements() {
        let bounded = parse_measurement("Kd>100mM").expect("bounded measurement");
        assert_eq!(bounded.1, Relation::GreaterThan);

        let approximate = parse_measurement("Ki~2.5nM").expect("approximate measurement");
        assert_eq!(approximate.1, Relation::Approximate);

        let ic50 = parse_measurement("IC50=45nM").expect("IC50 measurement");
        assert_eq!(ic50.0, AffinityKind::Ic50);

        // Neither belongs in an affinity-regression target.
        for raw in ["Kd>100mM", "IC50=45nM", "Ki~2.5nM"] {
            let (kind, relation, _) = parse_measurement(raw).unwrap();
            let affinity = Affinity {
                kind,
                relation,
                ..Affinity::default()
            };
            assert!(
                !affinity.is_regression_quality(),
                "{raw} should not be treated as a clean target"
            );
        }
    }

    #[test]
    fn converts_every_concentration_unit() {
        let molar = |raw: &str| parse_measurement(raw).unwrap().2.unwrap();
        assert_eq!(molar("Kd=1M"), 1.0);
        assert_eq!(molar("Kd=1mM"), 1e-3);
        assert_eq!(molar("Kd=1uM"), 1e-6);
        assert_eq!(molar("Kd=1nM"), 1e-9);
        assert_eq!(molar("Kd=1pM"), 1e-12);
        assert_eq!(molar("Kd=1fM"), 1e-15);
        // An unrecognized unit leaves the concentration unknown rather than guessing.
        assert!(parse_measurement("Kd=1zM").unwrap().2.is_none());
    }

    #[test]
    fn rejects_identifiers_that_could_escape_the_release_root() {
        assert!(normalized_id("1a30").is_ok());
        assert_eq!(normalized_id("1A30").unwrap(), "1a30");

        for bad in ["..", "../etc", "1a3", "abcd", "1a30/x", "1a-0", ""] {
            assert!(
                normalized_id(bad).is_err(),
                "'{bad}' should be rejected as a PDB identifier"
            );
        }
    }

    #[test]
    fn every_subset_lists_directory_names() {
        for subset in Subset::ALL {
            assert!(!subset.directory_names().is_empty());
            assert!(!subset.label().is_empty());
        }
        // `Any` has to cover the others, or a lookup against it could miss a present entry.
        for subset in [Subset::Refined, Subset::General, Subset::Core] {
            for name in subset.directory_names() {
                assert!(Subset::Any.directory_names().contains(name));
            }
        }
    }
}
