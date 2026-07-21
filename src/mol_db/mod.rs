//! Small-molecule screening libraries using Apache Parquet. Uses its Arrow component
//! for an in-memory representation of the Parquet DB on disk.
//!
//! Note: feather/ipc may have better read speeds for molecule screening, as an alternative to Parquet.
//! Note: We're currently using "SMILES" everywhere here that we list "ident", including to index
//! rows.
//!
//! Heavy data is split across columns, so it can be read a-la-carte. There are two such
//! on-demand loads, and they're independent of each other:
//!   - `mol_data` (atoms + bonds): `load_mol`, `load_mols`, `load_all`.
//!   - `idents` + `metadata`: `load_idents_meta`, `load_idents_meta_multi`, `load_idents_meta_all`.
//!
//! Screening only needs the first. Use `apply_idents_meta` to fold the second into molecules
//! already loaded from the first. `index_and_idents` loads the idents of every row (caching them),
//! for UI display alongside the eagerly-loaded index.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::{Arc, mpsc::Sender},
    thread,
    time::Duration,
};

use arrow::{
    array::{Array, ArrayRef, LargeBinaryArray, StringArray, UInt16Array, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bio_apis::pubchem::{self, StructureSearchNamespace};
use bytes::Bytes;
use na_seq::Element;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
        arrow_writer::ArrowWriter,
    },
    basic::Compression,
    errors::ParquetError,
    file::{properties::WriterProperties, reader::ChunkReader},
};

use crate::{
    mol_db::serialization::{
        idents_from_bytes, idents_to_bytes, metadata_from_bytes, metadata_to_bytes,
    },
    molecules::{MolIdent, small::MoleculeSmall},
    prefs::OpenType,
    screening::{collect_mol_files, load_mol_batch},
    state::{DbSel, State},
    util::{handle_err, handle_success},
};

mod serialization;

/// Column names; keep in sync with `schema` and `StoredMol`.
const COL_SMILES: &str = "smiles";
const COL_PUBCHEM_CID: &str = "pubchem_cid";
const COL_PUBCHEM_TITLE: &str = "pubchem_title";
const COL_HEAVY_ATOM_COUNT: &str = "heavy_atom_count";
const COL_MOL_DATA: &str = "mol_data";
const COL_IDENTS: &str = "idents";
const COL_METADATA: &str = "metadata";

const BATCH_SIZE_READ: usize = 8_192;
const BATCH_SIZE_WRITE: usize = 2_048;

// We include a collection of common small molecules with the application, so they can
// be loaded without internet queries. This increases application size.
pub const COMMON_MOL_DB: &[u8] = include_bytes!("../../common_mol_db.parquet");

/// Name shown in the UI for the database embedded in the binary; it has no filename.
pub const COMMON_MOL_DB_NAME: &str = "Common molecules (built in)";

fn parquet_err_to_io(e: ParquetError) -> io::Error {
    io::Error::other(e)
}

fn arrow_err_to_io(e: arrow::error::ArrowError) -> io::Error {
    io::Error::other(e)
}

/// One row in the Parquet file.
///
/// Keep "search columns" separate from mol_data so you can scan/filter without
/// deserializing every molecule. I believe we can load data column by column, so
/// we can load idents of various types all at once, and other data (Like atoms and bonds, metadata etc), without
/// loading all data.
#[derive(Debug, Clone)]
struct StoredMol {
    smiles: String,
    pubchem_cid: Option<u32>,
    pubchem_title: Option<String>,
    heavy_atom_count: u16,
    /// Serialized as binary; see the `to_bytes` and `from_bytes` serializations
    /// in the `serialization` module. This may only be atoms, bonds, and common.ident.
    mol_data: Vec<u8>,
    idents: Vec<MolIdent>,
    /// i.e. common.metadata
    metadata: HashMap<String, String>,
}

impl StoredMol {
    /// Build a row from a molecule loaded from a file, e.g. Mol2 or SDF.
    ///
    /// When `look_up_pubchem` is set, and the molecule lacks a title, fill in the title (and CID)
    /// over HTTP. Callers that must stay offline (e.g. bulk file/directory imports) pass `false`,
    /// leaving the title and CID blank unless the source file's metadata already carried them.
    fn from_mol(mut m: MoleculeSmall, look_up_pubchem: bool) -> io::Result<Self> {
        let smiles = smiles_from_idents(&m.idents).unwrap_or_else(|| m.common.ident.clone());
        let (mut pubchem_cid, mut pubchem_title) = pubchem_cid_title_from_idents(&m.idents);

        // Molecule files generally don't carry a title, so look it up over HTTP. Without one, the
        // DB table has only a SMILES string to identify the molecule by. The idents we store are
        // updated to match, so a molecule loaded back out of the DB carries them too.
        if look_up_pubchem
            && pubchem_title.is_none()
            && let Some(props) = pubchem_props(pubchem_cid, &smiles)
        {
            m.update_idents_and_char_from_pubchem(&props);

            pubchem_cid = Some(props.cid);
            pubchem_title = Some(props.title);
        }

        let heavy_atom_count = m
            .common
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
            .count() as u16;

        let mol_data = m.to_bytes();

        Ok(Self {
            smiles,
            pubchem_cid,
            pubchem_title,
            heavy_atom_count,
            mol_data,
            idents: m.idents,
            metadata: m.common.metadata,
        })
    }
}

fn smiles_from_idents(idents: &[MolIdent]) -> Option<String> {
    idents.iter().find_map(|id| match id {
        MolIdent::Smiles(s) => Some(s.clone()),
        _ => None,
    })
}

fn pubchem_cid_from_idents(idents: &[MolIdent]) -> Option<u32> {
    idents.iter().find_map(|id| match id {
        MolIdent::PubChem(cid) => Some(*cid),
        _ => None,
    })
}

/// Repetitive with [`pubchem_cid_from_idents`], but may be more efficient to group this way.
fn pubchem_cid_title_from_idents(idents: &[MolIdent]) -> (Option<u32>, Option<String>) {
    let cid = idents.iter().find_map(|id| match id {
        MolIdent::PubChem(cid) => Some(*cid),
        _ => None,
    });

    let title = idents.iter().find_map(|id| match id {
        MolIdent::PubchemTitle(title) => Some(title.clone()),
        _ => None,
    });

    (cid, title)
}

/// PubChem rate-limits to a handful of requests per second, and returns a busy/timeout fault rather
/// than throttling, so populating a DB of any size hits transient failures routinely. Retry a few
/// times, backing off; this doubles as the throttle.
const PUBCHEM_ATTEMPTS: u32 = 3;
const PUBCHEM_BACKOFF_MS: u64 = 500;

/// How many CIDs to request titles for in a single PubChem call while enriching a DB. Batching
/// keeps the request count (and so the load we put on PubChem) low; the cap keeps the request URL
/// from growing long enough to be rejected.
const PUBCHEM_CIDS_PER_REQUEST: usize = 100;
/// Minimum spacing between PubChem requests while enriching a DB, to stay within their rate limit
/// (a few requests per second) rather than hammering the service.
const PUBCHEM_ENRICH_INTERVAL_MS: u64 = 250;
/// How many rows to look up between writes of the DB file while enriching. Flushing periodically,
/// rather than only once at the end, means a long run's progress survives the app being closed or
/// crashing partway through. Each flush rewrites the whole parquet file, so this trades some extra
/// I/O for that safety; PubChem's rate limit dwarfs the write cost regardless.
const PUBCHEM_ENRICH_FLUSH_EVERY: usize = 1_000;

/// Look up a molecule's PubChem properties over HTTP, to fill in the title (and CID) we store
/// alongside it. Queries by CID if we have one, else by SMILES.
///
/// Returns `None` if the molecule isn't in PubChem, or every attempt fails: a title is a nicety,
/// and shouldn't stop the molecule from being stored.
fn pubchem_props(cid: Option<u32>, smiles: &str) -> Option<pubchem::Properties> {
    let (namespace, id) = match cid {
        Some(cid) => (StructureSearchNamespace::Cid, cid.to_string()),
        None => (StructureSearchNamespace::Smiles, smiles.to_string()),
    };

    let mut last_err = None;
    for attempt in 0..PUBCHEM_ATTEMPTS {
        if attempt > 0 {
            thread::sleep(Duration::from_millis(PUBCHEM_BACKOFF_MS * (1 << attempt)));
        }

        match pubchem::properties(namespace.clone(), &id) {
            Ok(props) => return Some(props),
            Err(e) => last_err = Some(e),
        }
    }

    eprintln!("Unable to load PubChem properties for {id}: {last_err:?}");
    None
}

/// Lightweight metadata for a molecule stored in the DB. Excludes the heavy `mol_data` blob.
#[derive(Debug, Clone)]
pub struct MolMeta {
    pub smiles: String,
    pub pubchem_cid: Option<u32>,
    pub pubchem_title: Option<String>,
    pub heavy_atom_count: u16,
}

impl MolMeta {
    /// Whether this molecule matches a search: a substring of its SMILES, PubChem title, or CID.
    /// `search` must already be trimmed and lowercased; the caller usually does that once for a
    /// whole scan.
    pub fn matches_search(&self, search: &str) -> bool {
        if self.smiles.to_lowercase().contains(search) {
            return true;
        }

        if let Some(title) = &self.pubchem_title
            && title.to_lowercase().contains(search)
        {
            return true;
        }

        match self.pubchem_cid {
            Some(cid) => cid.to_string().contains(search),
            None => false,
        }
    }
}

/// A molecule's identifiers and metadata, as stored in the DB. Loaded on demand, and separately
/// from `mol_data`: screening workflows don't need these.
#[derive(Debug, Clone, Default)]
pub struct MolIdentsMeta {
    pub idents: Vec<MolIdent>,
    /// i.e. `common.metadata`
    pub metadata: HashMap<String, String>,
}

/// PubChem data looked up for a DB row while enriching, applied by SMILES key. Only fields that
/// were missing on the row are filled in; see [`run_pubchem_enrich`] and
/// [`ParquetMolDb::apply_pubchem_meta`].
#[derive(Debug, Clone, Default)]
pub struct PubchemMeta {
    pub cid: Option<u32>,
    pub title: Option<String>,
}

/// A DB row missing a PubChem title and/or CID, i.e. a target for enrichment. `cid` is `Some` when
/// only the title is missing, which lets those rows be looked up in batches keyed by CID rather
/// than one query per molecule.
#[derive(Debug, Clone)]
pub struct EnrichTarget {
    pub smiles: String,
    pub cid: Option<u32>,
}

/// Messages from the background PubChem-enrichment worker to the UI. See [`run_pubchem_enrich`].
pub enum DbEnrichMsg {
    /// Cumulative count of target rows looked up so far, for a progress display.
    Progress(usize),
    /// The rewritten DB (reopened, with a fresh index) and the number of rows updated.
    Done { db: Box<ParquetMolDb>, updated: usize },
    /// The rewrite failed; carries a message for the user.
    Failed(String),
}

/// Where a database's parquet data lives. Databases on disk can be added to and deleted from; the
/// one embedded in the binary is fixed at compile time, so it's read-only.
#[derive(Debug, Clone, PartialEq)]
pub enum DbSource {
    File(PathBuf),
    /// Shipped with the application; see [`COMMON_MOL_DB`].
    Embedded(&'static [u8]),
}

impl DbSource {
    /// The file this DB was loaded from, or `None` for the embedded one.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::File(p) => Some(p),
            Self::Embedded(_) => None,
        }
    }

    /// A name to show in the UI.
    pub fn name(&self) -> String {
        match self {
            Self::File(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string_lossy().into_owned()),
            Self::Embedded(_) => COMMON_MOL_DB_NAME.to_owned(),
        }
    }

    /// Whether this DB can be modified. The embedded one can't.
    pub fn writable(&self) -> bool {
        matches!(self, Self::File(_))
    }

    /// Whether there is anything to read: a file that hasn't been created yet has no rows, and
    /// neither does an embedded DB that wasn't built into this binary.
    fn readable(&self) -> bool {
        match self {
            Self::File(p) => p.exists(),
            Self::Embedded(b) => !b.is_empty(),
        }
    }
}

/// Struct representing the whole DB; used to open, update it, load data from disk in general.
pub struct ParquetMolDb {
    /// The parquet file on disk, or the bytes embedded in the binary.
    pub source: DbSource,
    /// Lightweight metadata index loaded eagerly on open: smiles: MolMeta.
    /// Does NOT include the heavy `mol_data`, `idents`, or `metadata` columns; those are read from
    /// disk on demand.
    pub index_meta: HashMap<String, MolMeta>,
    /// Every row's idents, keyed by SMILES. Filled by `index_and_idents` the first time something
    /// (e.g. the DB table in the UI) asks for them; `None` until then, and reset whenever the file
    /// is rewritten.
    idents_cache: Option<HashMap<String, Vec<MolIdent>>>,
}

impl ParquetMolDb {
    /// Create / open a DB at a parquet file path.
    ///
    /// Eagerly loads only the lightweight metadata columns (smiles, pubchem_cid, pubchem_title,
    /// heavy_atom_count) into `index_meta`. The heavy `mol_data`, `idents`, and `metadata` columns
    /// are NOT loaded here; they're read from disk on demand. See the module docs.
    pub fn new(path: &Path) -> io::Result<Self> {
        Self::open_source(DbSource::File(path.to_owned()))
    }

    /// Open the read-only database embedded in the binary. Pass [`COMMON_MOL_DB`].
    pub fn from_embedded(bytes: &'static [u8]) -> io::Result<Self> {
        Self::open_source(DbSource::Embedded(bytes))
    }

    /// Open a DB from either kind of source. `DbSource` is `Send`, so this is how a background
    /// thread (e.g. screening) reopens a DB the UI is holding.
    pub fn open_source(source: DbSource) -> io::Result<Self> {
        let mut res = Self {
            source,
            index_meta: HashMap::new(),
            idents_cache: None,
        };

        if res.source.readable() {
            res.rebuild_index()?;
        }

        Ok(res)
    }

    /// The file this DB was loaded from, or `None` for the embedded one.
    pub fn path(&self) -> Option<&Path> {
        self.source.path()
    }

    /// A name to show in the UI.
    pub fn name(&self) -> String {
        self.source.name()
    }

    /// Errors if this DB can't be modified, i.e. it's the one embedded in the binary. Called by
    /// every operation that rewrites the file.
    fn check_writable(&self) -> io::Result<()> {
        if self.source.writable() {
            return Ok(());
        }

        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "The built-in molecule database is read-only. Create or load a database to add \
             molecules to it.",
        ))
    }

    /// Keep this in sync with `StoredMol`
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(COL_SMILES, DataType::Utf8, false),
            Field::new(COL_PUBCHEM_CID, DataType::UInt32, true),
            Field::new(COL_PUBCHEM_TITLE, DataType::Utf8, true),
            Field::new(COL_HEAVY_ATOM_COUNT, DataType::UInt16, false),
            // This contains our atoms, bonds, and common.ident. Serialized as binary.
            Field::new(COL_MOL_DATA, DataType::LargeBinary, false),
            // These two are loaded together, on demand, and independently of mol_data.
            Field::new(COL_IDENTS, DataType::LargeBinary, false),
            Field::new(COL_METADATA, DataType::LargeBinary, false),
        ]))
    }

    /// Read molecules from a set of molecule files (Mol2 or SDF) on disk, and load them into the
    /// database, writing a fresh parquet file. Shared by `add_mols_from_dir` and
    /// `add_mols_from_file`.
    ///
    /// Parquet files are immutable, so to add molecules we read the existing rows, merge, and
    /// rewrite. Molecules already in the DB (matched on SMILES) are replaced by the incoming ones.
    fn add_mol_files(&mut self, files: &[PathBuf]) -> io::Result<()> {
        self.check_writable()?;

        let mut rows = self.read_all_rows()?;
        let mut row_i = row_index(&rows);

        let mut offset = 0;

        while offset < files.len() {
            let (mols, consumed) = load_mol_batch(&files[offset..])?;

            for m in mols {
                // Bulk imports stay offline: leave the title/CID blank unless the source file's
                // metadata already carried them.
                merge_row(&mut rows, &mut row_i, StoredMol::from_mol(m, false)?);
            }
            offset += consumed;
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;

        Ok(())
    }

    /// Read molecules from molecule files on disk, and load them into the database. Loads
    /// recursively from a given folder (Mol2 or SDF), then writes a fresh parquet file.
    pub fn add_mols_from_dir(&mut self, mol_path: &Path) -> io::Result<()> {
        let files = collect_mol_files(mol_path)?;
        self.add_mol_files(&files)
    }

    /// Load a single molecule file (SDF or Mol2) from disk, and add it to the database. An SDF file
    /// may hold more than one molecule. The directory equivalent is `add_mols_from_dir`.
    pub fn add_mols_from_file(&mut self, path: &Path) -> io::Result<()> {
        // `load_mol_batch` only handles these two, and panics otherwise; the dialog filter doesn't
        // prevent a name being typed in directly.
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        if !matches!(ext.as_str(), "sdf" | "mol2") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unable to add {ext:?} files to a database; use SDF or Mol2"),
            ));
        }

        self.add_mol_files(&[path.to_path_buf()])
    }

    /// Whether this molecule is already in the DB. Matches on the row key (SMILES), or on PubChem
    /// CID: the same molecule from a different source may have a SMILES that differs in form.
    /// Uses the lightweight index only; no disk read.
    pub fn contains_mol(&self, mol: &MoleculeSmall) -> bool {
        if let Some(smiles) = mol.get_smiles()
            && self.index_meta.contains_key(smiles)
        {
            return true;
        }

        let Some(cid) = pubchem_cid_from_idents(&mol.idents) else {
            return false;
        };

        self.index_meta.values().any(|m| m.pubchem_cid == Some(cid))
    }

    /// Add molecules already in memory (e.g. open ligands) to the DB. As with `populate`, the file
    /// is read, merged, and rewritten, and molecules already in the DB (matched on SMILES) are
    /// replaced by the incoming ones.
    pub fn add_mols(&mut self, mols: &[MoleculeSmall]) -> io::Result<()> {
        self.check_writable()?;

        let mut rows = self.read_all_rows()?;
        let mut row_i = row_index(&rows);

        for m in mols {
            merge_row(&mut rows, &mut row_i, StoredMol::from_mol(m.clone(), true)?);
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;

        Ok(())
    }

    /// Remove a molecule from the DB, by SMILES key. As with adding, parquet files are immutable,
    /// so the file is read, the row dropped, and the file rewritten.
    pub fn remove_mol(&mut self, smiles: &str) -> io::Result<()> {
        self.check_writable()?;

        let mut rows = self.read_all_rows()?;

        let len_orig = rows.len();
        rows.retain(|r| r.smiles != smiles);

        if rows.len() == len_orig {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Molecule not found: {smiles}"),
            ));
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;

        Ok(())
    }

    /// The lightweight index, and the `idents` of every molecule in the DB (keyed by SMILES);
    /// e.g. to display both in a table. The idents are read from disk on the first call, then
    /// cached until the file is next rewritten. `mol_data` and `metadata` are not read.
    ///
    /// A DB written before idents were stored yields an empty idents map.
    pub fn index_and_idents(
        &mut self,
    ) -> (&HashMap<String, MolMeta>, &HashMap<String, Vec<MolIdent>>) {
        if self.idents_cache.is_none() {
            let loaded = self.load_idents_meta_all().unwrap_or_else(|e| {
                eprintln!("Error loading idents from {}: {e}", self.name());
                HashMap::new()
            });

            self.idents_cache = Some(
                loaded
                    .into_iter()
                    .map(|(smiles, im)| (smiles, im.idents))
                    .collect(),
            );
        }

        (&self.index_meta, self.idents_cache.as_ref().unwrap())
    }

    /// Molecules whose SMILES, PubChem title, or CID contain `search`, best-first: exact matches
    /// on CID or title come before substring hits, and shorter SMILES before longer. Uses the
    /// in-memory index only, so this is cheap enough to run per-frame from the UI.
    ///
    /// Returns at most `limit` results.
    pub fn search(&self, search: &str, limit: usize) -> Vec<&MolMeta> {
        let search = search.trim().to_lowercase();
        if search.is_empty() {
            return Vec::new();
        }

        let mut matches: Vec<&MolMeta> = self
            .index_meta
            .values()
            .filter(|m| m.matches_search(&search))
            .collect();

        let rank = |m: &MolMeta| {
            let exact = m.pubchem_cid.is_some_and(|cid| cid.to_string() == search)
                || m.pubchem_title
                    .as_ref()
                    .is_some_and(|t| t.to_lowercase() == search);

            // Exact matches first, then by SMILES length.
            (!exact, m.smiles.len())
        };

        // Sorted alphabetically at the end so the list is stable across frames; `index_meta` is a
        // HashMap, and its iteration order is not.
        matches.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.smiles.cmp(&b.smiles)));

        matches.truncate(limit);
        matches
    }

    fn write_all_rows(&self, rows: &[StoredMol]) -> io::Result<()> {
        self.check_writable()?;

        let Some(path) = self.path() else {
            unreachable!("`check_writable` already rejected a source without a path")
        };

        let file = File::create(path)?;
        let schema = Self::schema();

        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(parquet_err_to_io)?;

        for chunk in rows.chunks(BATCH_SIZE_WRITE) {
            let batch = Self::make_batch(chunk, schema.clone())?;
            writer.write(&batch).map_err(parquet_err_to_io)?;
        }

        writer.close().map_err(parquet_err_to_io)?;
        Ok(())
    }

    fn make_batch(rows: &[StoredMol], schema: Arc<Schema>) -> io::Result<RecordBatch> {
        let smiles_arr: StringArray = rows.iter().map(|r| Some(r.smiles.as_str())).collect();
        let pubchem_arr: UInt32Array = rows.iter().map(|r| r.pubchem_cid).collect();
        let pubchem_title_arr: StringArray =
            rows.iter().map(|r| r.pubchem_title.as_deref()).collect();

        let heavy_count_arr: UInt16Array = rows.iter().map(|r| r.heavy_atom_count).collect();

        let mol_data_arr: LargeBinaryArray =
            rows.iter().map(|r| Some(r.mol_data.as_slice())).collect();

        // Serialize up front, so the arrays can borrow the resulting buffers.
        let idents_ser: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| idents_to_bytes(&r.idents))
            .collect::<io::Result<_>>()?;

        let metadata_ser: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| metadata_to_bytes(&r.metadata))
            .collect::<io::Result<_>>()?;

        let idents_arr: LargeBinaryArray = idents_ser.iter().map(|b| Some(b.as_slice())).collect();
        let metadata_arr: LargeBinaryArray =
            metadata_ser.iter().map(|b| Some(b.as_slice())).collect();

        let cols: Vec<ArrayRef> = vec![
            Arc::new(smiles_arr),
            Arc::new(pubchem_arr),
            Arc::new(pubchem_title_arr),
            Arc::new(heavy_count_arr),
            Arc::new(mol_data_arr),
            Arc::new(idents_arr),
            Arc::new(metadata_arr),
        ];

        RecordBatch::try_new(schema, cols).map_err(io::Error::other)
    }

    /// Reads only the three lightweight metadata columns from disk into `index_meta`.
    /// The heavy `mol_data`, `idents`, and `metadata` columns are intentionally excluded.
    fn rebuild_index(&mut self) -> io::Result<()> {
        self.index_meta.clear();
        // Every rewrite of the file goes through here, so this is where the cache goes stale.
        self.idents_cache = None;

        // A DB written before the title column existed is still readable; titles come back `None`,
        // and the column is gained when the file is next rewritten.
        let has_title = has_cols(&self.source, &[COL_PUBCHEM_TITLE])?;

        let mut cols = vec![COL_SMILES, COL_PUBCHEM_CID, COL_HEAVY_ATOM_COUNT];
        if has_title {
            cols.push(COL_PUBCHEM_TITLE);
        }

        let mut reader = open_reader(&self.source, &cols)?;

        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let smiles_col = str_col(&batch, COL_SMILES)?;
            let cid_col = u32_col(&batch, COL_PUBCHEM_CID)?;
            let title_col = match has_title {
                true => Some(str_col(&batch, COL_PUBCHEM_TITLE)?),
                false => None,
            };
            let heavy_atom_count_col = u16_col(&batch, COL_HEAVY_ATOM_COUNT)?;

            for i in 0..smiles_col.len() {
                let smiles = smiles_col.value(i).to_string();

                let pubchem_cid = if cid_col.is_null(i) {
                    None
                } else {
                    Some(cid_col.value(i))
                };

                let pubchem_title = title_col.and_then(|c| {
                    if c.is_null(i) {
                        None
                    } else {
                        Some(c.value(i).to_string())
                    }
                });

                self.index_meta.insert(
                    smiles.clone(),
                    MolMeta {
                        smiles,
                        pubchem_cid,
                        pubchem_title,
                        heavy_atom_count: heavy_atom_count_col.value(i),
                    },
                );
            }
        }

        Ok(())
    }

    /// Read every column of every row into memory. Parquet files are immutable, so this is the
    /// first step of any modification: read, change, rewrite. (See `populate`, `update_idents_meta`)
    fn read_all_rows(&self) -> io::Result<Vec<StoredMol>> {
        if !self.source.readable() {
            return Ok(Vec::new());
        }

        // A DB written before we stored idents + metadata (or the title column) is still readable;
        // those rows simply come back empty, and gain the columns when the file is rewritten.
        let has_idents_meta = has_idents_meta_cols(&self.source)?;
        let has_title = has_cols(&self.source, &[COL_PUBCHEM_TITLE])?;

        let mut cols = vec![
            COL_SMILES,
            COL_PUBCHEM_CID,
            COL_HEAVY_ATOM_COUNT,
            COL_MOL_DATA,
        ];
        if has_title {
            cols.push(COL_PUBCHEM_TITLE);
        }
        if has_idents_meta {
            cols.push(COL_IDENTS);
            cols.push(COL_METADATA);
        }

        let mut reader = open_reader(&self.source, &cols)?;

        let mut rows = Vec::with_capacity(self.index_meta.len());
        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let smiles_col = str_col(&batch, COL_SMILES)?;
            let cid_col = u32_col(&batch, COL_PUBCHEM_CID)?;
            let pubchem_title_col = match has_title {
                true => Some(str_col(&batch, COL_PUBCHEM_TITLE)?),
                false => None,
            };
            let heavy_atom_count_col = u16_col(&batch, COL_HEAVY_ATOM_COUNT)?;
            let mol_data_col = bin_col(&batch, COL_MOL_DATA)?;

            let idents_col = match has_idents_meta {
                true => Some(bin_col(&batch, COL_IDENTS)?),
                false => None,
            };
            let metadata_col = match has_idents_meta {
                true => Some(bin_col(&batch, COL_METADATA)?),
                false => None,
            };

            for i in 0..smiles_col.len() {
                let idents = match idents_col {
                    Some(c) => idents_from_bytes(c.value(i))?,
                    None => Vec::new(),
                };
                let metadata = match metadata_col {
                    Some(c) => metadata_from_bytes(c.value(i))?,
                    None => HashMap::new(),
                };

                let pubchem_cid = if cid_col.is_null(i) {
                    None
                } else {
                    Some(cid_col.value(i))
                };

                let pubchem_title = pubchem_title_col.and_then(|c| {
                    if c.is_null(i) {
                        None
                    } else {
                        Some(c.value(i).to_string())
                    }
                });

                rows.push(StoredMol {
                    smiles: smiles_col.value(i).to_string(),
                    pubchem_cid,
                    pubchem_title,
                    heavy_atom_count: heavy_atom_count_col.value(i),
                    mol_data: mol_data_col.value(i).to_vec(),
                    idents,
                    metadata,
                });
            }
        }

        Ok(rows)
    }

    /// Read `mol_data` for a single molecule from disk by scanning for its SMILES key.
    pub fn load_mol(&self, smiles: &str) -> io::Result<MoleculeSmall> {
        let mols = self.load_mols(&[smiles])?;

        mols.into_iter().next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Molecule not found: {smiles}"),
            )
        })
    }

    /// Read `mol_data` for a subset of molecules (by SMILES key) in a single disk pass.
    pub fn load_mols(&self, smiles_keys: &[&str]) -> io::Result<Vec<MoleculeSmall>> {
        let targets: HashSet<&str> = smiles_keys.iter().copied().collect();

        let mut reader = open_reader(&self.source, &[COL_SMILES, COL_MOL_DATA])?;

        let mut result = Vec::with_capacity(targets.len());
        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let smiles_col = str_col(&batch, COL_SMILES)?;
            let mol_data_col = bin_col(&batch, COL_MOL_DATA)?;

            for i in 0..smiles_col.len() {
                if targets.contains(smiles_col.value(i)) {
                    result.push(MoleculeSmall::from_bytes(mol_data_col.value(i))?);
                }
            }
        }

        Ok(result)
    }

    /// Read all `mol_data` from disk and deserialize into molecules.
    pub fn load_all(&self) -> io::Result<Vec<MoleculeSmall>> {
        let mut reader = open_reader(&self.source, &[COL_MOL_DATA])?;

        let mut result = Vec::with_capacity(self.index_meta.len());
        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let mol_data_col = bin_col(&batch, COL_MOL_DATA)?;

            for i in 0..mol_data_col.len() {
                result.push(MoleculeSmall::from_bytes(mol_data_col.value(i))?);
            }
        }

        Ok(result)
    }

    /// Read `idents` + `metadata` for a single molecule from disk, by SMILES key. This is
    /// independent of loading `mol_data`; see the module docs.
    pub fn load_idents_meta(&self, smiles: &str) -> io::Result<MolIdentsMeta> {
        let mut loaded = self.load_idents_meta_multi(&[smiles])?;

        loaded.remove(smiles).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Molecule not found: {smiles}"),
            )
        })
    }

    /// Read `idents` + `metadata` for a subset of molecules (by SMILES key) in a single disk pass.
    /// Keyed by SMILES, as the results otherwise have no reliable association with the molecules
    /// they were requested for.
    pub fn load_idents_meta_multi(
        &self,
        smiles_keys: &[&str],
    ) -> io::Result<HashMap<String, MolIdentsMeta>> {
        let targets: HashSet<&str> = smiles_keys.iter().copied().collect();
        self.read_idents_meta(Some(&targets))
    }

    /// Read `idents` + `metadata` for every molecule in the DB, keyed by SMILES.
    pub fn load_idents_meta_all(&self) -> io::Result<HashMap<String, MolIdentsMeta>> {
        self.read_idents_meta(None)
    }

    /// Reads the `idents` and `metadata` columns; `mol_data` is NOT read here. `targets` filters
    /// by SMILES key; `None` reads every row.
    fn read_idents_meta(
        &self,
        targets: Option<&HashSet<&str>>,
    ) -> io::Result<HashMap<String, MolIdentsMeta>> {
        if !has_idents_meta_cols(&self.source)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "This database was created before idents and metadata were stored. Re-add its \
                 molecules to populate them.",
            ));
        }

        let mut reader = open_reader(&self.source, &[COL_SMILES, COL_IDENTS, COL_METADATA])?;

        let mut result = HashMap::with_capacity(match targets {
            Some(t) => t.len(),
            None => self.index_meta.len(),
        });

        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let smiles_col = str_col(&batch, COL_SMILES)?;
            let idents_col = bin_col(&batch, COL_IDENTS)?;
            let metadata_col = bin_col(&batch, COL_METADATA)?;

            for i in 0..smiles_col.len() {
                let smiles = smiles_col.value(i);

                if let Some(t) = targets
                    && !t.contains(smiles)
                {
                    continue;
                }

                result.insert(
                    smiles.to_string(),
                    MolIdentsMeta {
                        idents: idents_from_bytes(idents_col.value(i))?,
                        metadata: metadata_from_bytes(metadata_col.value(i))?,
                    },
                );
            }
        }

        Ok(result)
    }

    /// Load `idents` + `metadata` for molecules already in memory (e.g. loaded from `mol_data` by
    /// `load_mols`), and apply them, in a single disk pass. This is how the two a-la-carte loads
    /// are combined.
    ///
    /// Molecules missing a SMILES ident, or absent from the DB, are left as-is.
    pub fn apply_idents_meta(&self, mols: &mut [MoleculeSmall]) -> io::Result<()> {
        let keys: Vec<String> = mols
            .iter()
            .filter_map(|m| m.get_smiles().map(|s| s.to_string()))
            .collect();

        let keys_ref: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let loaded = self.load_idents_meta_multi(&keys_ref)?;

        for mol in mols.iter_mut() {
            let Some(smiles) = mol.get_smiles().map(|s| s.to_string()) else {
                continue;
            };
            let Some(im) = loaded.get(&smiles) else {
                continue;
            };

            // Merge, rather than replace: a molecule deserialized from `mol_data` already has its
            // SMILES ident, and we don't want to drop it if the DB row is empty.
            for ident in &im.idents {
                if !mol.idents.contains(ident) {
                    mol.idents.push(ident.clone());
                }
            }
            for (key, val) in &im.metadata {
                mol.common.metadata.insert(key.clone(), val.clone());
            }
        }

        Ok(())
    }

    /// Update the `idents` and `metadata` stored for molecules already in the DB, keyed by SMILES.
    /// Parquet files are immutable, so this reads the file, replaces these fields on matching rows,
    /// and rewrites it. `mol_data` is preserved as-is; it is not deserialized.
    pub fn update_idents_meta(
        &mut self,
        updates: &HashMap<String, MolIdentsMeta>,
    ) -> io::Result<()> {
        self.check_writable()?;

        let mut rows = self.read_all_rows()?;

        for row in &mut rows {
            let Some(im) = updates.get(&row.smiles) else {
                continue;
            };

            row.idents = im.idents.clone();
            row.metadata = im.metadata.clone();

            // Keep the searchable CID and title columns in sync with the idents they're derived from.
            let (cid, title) = pubchem_cid_title_from_idents(&im.idents);
            if let Some(cid) = cid {
                row.pubchem_cid = Some(cid);
            }
            if let Some(title) = title {
                row.pubchem_title = Some(title);
            }
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()
    }

    /// Fill in missing `pubchem_cid` / `pubchem_title` (and the matching idents) for the rows named
    /// in `updates`, keyed by SMILES, then rewrite the file. Only fields that were empty are
    /// filled; existing values are left untouched. Returns the number of rows changed.
    ///
    /// As with adding and deleting, parquet files are immutable, so this reads the file, edits the
    /// matching rows, and rewrites it. See [`run_pubchem_enrich`], which produces `updates`.
    pub fn apply_pubchem_meta(
        &mut self,
        updates: &HashMap<String, PubchemMeta>,
    ) -> io::Result<usize> {
        self.check_writable()?;

        let mut rows = self.read_all_rows()?;
        let mut changed = 0;

        for row in &mut rows {
            let Some(meta) = updates.get(&row.smiles) else {
                continue;
            };

            let mut row_changed = false;

            if row.pubchem_cid.is_none()
                && let Some(cid) = meta.cid
            {
                row.pubchem_cid = Some(cid);
                if !row.idents.iter().any(|id| matches!(id, MolIdent::PubChem(_))) {
                    row.idents.push(MolIdent::PubChem(cid));
                }
                row_changed = true;
            }

            if row.pubchem_title.is_none()
                && let Some(title) = &meta.title
            {
                row.pubchem_title = Some(title.clone());
                // Replace any stale title ident, rather than accumulating duplicates.
                row.idents
                    .retain(|id| !matches!(id, MolIdent::PubchemTitle(_)));
                row.idents.push(MolIdent::PubchemTitle(title.clone()));
                row_changed = true;
            }

            if row_changed {
                changed += 1;
            }
        }

        // Skip the (expensive) rewrite if nothing matched: e.g. every lookup came back empty.
        if changed > 0 {
            self.write_all_rows(&rows)?;
            self.rebuild_index()?;
        }

        Ok(changed)
    }
}

/// Look up missing PubChem titles/CIDs for `targets` (rows of `source`), writing them back to the
/// DB in batches as they come in, then hand the finished DB back. Intended to run on a background
/// thread; progress and the result are sent over `tx`. See [`DbEnrichMsg`] and
/// [`ParquetMolDb::apply_pubchem_meta`].
///
/// Rows that already have a CID need only a title, and are fetched in batches keyed by CID. Rows
/// without a CID are looked up by SMILES, one request each (PubChem's SMILES namespace doesn't take
/// a list). Requests are spaced out to respect PubChem's rate limit.
///
/// Results are flushed to disk every [`PUBCHEM_ENRICH_FLUSH_EVERY`] rows rather than only at the
/// end, so a long run against PubChem's slow API doesn't lose everything if the app is closed
/// partway through.
pub fn run_pubchem_enrich(source: DbSource, targets: Vec<EnrichTarget>, tx: Sender<DbEnrichMsg>) {
    if let Err(e) = run_pubchem_enrich_inner(source, targets, &tx) {
        let _ = tx.send(DbEnrichMsg::Failed(e));
    }
}

/// The body of [`run_pubchem_enrich`], split out so the periodic flushes and the final write can
/// use `?`; any error is turned into a [`DbEnrichMsg::Failed`] by the caller. On success this sends
/// [`DbEnrichMsg::Done`] itself, since only it holds the finished DB.
fn run_pubchem_enrich_inner(
    source: DbSource,
    targets: Vec<EnrichTarget>,
    tx: &Sender<DbEnrichMsg>,
) -> Result<(), String> {
    // Opened up front (rather than only at the end) so results can be flushed to disk as they're
    // looked up. Each flush rewrites through this same handle, keeping its index current.
    let mut db =
        ParquetMolDb::open_source(source).map_err(|e| format!("reopening the database: {e}"))?;

    // Resolved rows not yet written. Flushed to disk (and cleared) every `PUBCHEM_ENRICH_FLUSH_EVERY`
    // rows looked up; `apply_pubchem_meta` only fills empty fields, so rows already written on an
    // earlier flush are untouched by later ones.
    let mut pending: HashMap<String, PubchemMeta> = HashMap::new();
    let mut updated = 0;
    let mut processed = 0;
    let mut processed_at_last_flush = 0;

    // `(cid, smiles)` for rows needing only a title; `smiles` for rows needing a CID (and title).
    let mut title_targets: Vec<(u32, String)> = Vec::new();
    let mut smiles_targets: Vec<String> = Vec::new();
    for t in targets {
        match t.cid {
            Some(cid) => title_targets.push((cid, t.smiles)),
            None => smiles_targets.push(t.smiles),
        }
    }

    // Batched CID -> title: many CIDs per request.
    for chunk in title_targets.chunks(PUBCHEM_CIDS_PER_REQUEST) {
        let cids: Vec<u32> = chunk.iter().map(|(cid, _)| *cid).collect();

        match pubchem::titles_for_cids(&cids) {
            Ok(titles) => {
                for (cid, smiles) in chunk {
                    if let Some(title) = titles.get(cid) {
                        pending.insert(
                            smiles.clone(),
                            PubchemMeta {
                                cid: None,
                                title: Some(title.clone()),
                            },
                        );
                    }
                }
            }
            // One bad batch shouldn't sink the whole run; log it and carry on.
            Err(e) => eprintln!("PubChem title lookup failed for {} CIDs: {e:?}", cids.len()),
        }

        processed += chunk.len();
        let _ = tx.send(DbEnrichMsg::Progress(processed));

        if processed - processed_at_last_flush >= PUBCHEM_ENRICH_FLUSH_EVERY {
            updated += flush_enrich(&mut db, &mut pending)?;
            processed_at_last_flush = processed;
        }

        thread::sleep(Duration::from_millis(PUBCHEM_ENRICH_INTERVAL_MS));
    }

    // Per-SMILES CID + title. `pubchem_props` already retries with backoff on transient failures.
    for smiles in smiles_targets {
        if let Some(props) = pubchem_props(None, &smiles) {
            pending.insert(
                smiles.clone(),
                PubchemMeta {
                    cid: Some(props.cid),
                    title: Some(props.title),
                },
            );
        }

        processed += 1;
        let _ = tx.send(DbEnrichMsg::Progress(processed));

        if processed - processed_at_last_flush >= PUBCHEM_ENRICH_FLUSH_EVERY {
            updated += flush_enrich(&mut db, &mut pending)?;
            processed_at_last_flush = processed;
        }

        thread::sleep(Duration::from_millis(PUBCHEM_ENRICH_INTERVAL_MS));
    }

    // Whatever's left under the batch threshold.
    updated += flush_enrich(&mut db, &mut pending)?;

    // Hand back the DB we've been writing to, so the UI can swap in its fresh index without
    // re-reading the file itself.
    let _ = tx.send(DbEnrichMsg::Done {
        db: Box::new(db),
        updated,
    });

    Ok(())
}

/// Write any resolved-but-unwritten rows to the DB file, returning the number of rows changed and
/// clearing them from `pending`. A no-op (no rewrite) when `pending` is empty. See
/// [`run_pubchem_enrich`], which calls this periodically so progress survives a restart.
fn flush_enrich(
    db: &mut ParquetMolDb,
    pending: &mut HashMap<String, PubchemMeta>,
) -> Result<usize, String> {
    if pending.is_empty() {
        return Ok(0);
    }

    let changed = db
        .apply_pubchem_meta(pending)
        .map_err(|e| format!("writing the database: {e}"))?;
    pending.clear();

    Ok(changed)
}

/// SMILES: index into `rows`. Used to prevent duplicate rows for the same molecule when merging.
fn row_index(rows: &[StoredMol]) -> HashMap<String, usize> {
    rows.iter()
        .enumerate()
        .map(|(i, r)| (r.smiles.clone(), i))
        .collect()
}

/// Append `row`, or replace the existing row with the same SMILES.
fn merge_row(rows: &mut Vec<StoredMol>, row_i: &mut HashMap<String, usize>, row: StoredMol) {
    match row_i.get(&row.smiles) {
        Some(i) => rows[*i] = row,
        None => {
            row_i.insert(row.smiles.clone(), rows.len());
            rows.push(row);
        }
    }
}

/// Open the parquet data, reading only the columns named in `cols`. Columns not listed are not
/// read. Note that the resulting batches keep the file's column order, not `cols`'; look
/// columns up by name. (See `str_col` etc)
fn open_reader(source: &DbSource, cols: &[&str]) -> io::Result<ParquetRecordBatchReader> {
    match source {
        DbSource::File(path) => reader_from_chunks(File::open(path)?, cols),
        DbSource::Embedded(bytes) => reader_from_chunks(Bytes::from_static(bytes), cols),
    }
}

/// The half of `open_reader` that doesn't care where the bytes came from. `ChunkReader` is what
/// parquet reads through; both `File` and `Bytes` implement it.
fn reader_from_chunks<R: ChunkReader + 'static>(
    chunks: R,
    cols: &[&str],
) -> io::Result<ParquetRecordBatchReader> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(chunks).map_err(parquet_err_to_io)?;

    let schema = builder.schema().clone();

    let mut indices = Vec::with_capacity(cols.len());
    for col in cols {
        indices.push(schema.index_of(col).map_err(io::Error::other)?);
    }

    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);

    builder
        .with_projection(mask)
        .with_batch_size(BATCH_SIZE_READ)
        .build()
        .map_err(parquet_err_to_io)
}

/// Whether this DB has all the named columns. Files written before a column was added to the
/// schema don't have it.
fn has_cols(source: &DbSource, cols: &[&str]) -> io::Result<bool> {
    let schema = match source {
        DbSource::File(path) => ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)
            .map_err(parquet_err_to_io)?
            .schema()
            .clone(),
        DbSource::Embedded(bytes) => {
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from_static(bytes))
                .map_err(parquet_err_to_io)?
                .schema()
                .clone()
        }
    };

    Ok(cols.iter().all(|c| schema.index_of(c).is_ok()))
}

/// Whether this DB has the `idents` and `metadata` columns. Files written before we stored them
/// don't.
fn has_idents_meta_cols(source: &DbSource) -> io::Result<bool> {
    has_cols(source, &[COL_IDENTS, COL_METADATA])
}

fn col<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a T> {
    batch
        .column_by_name(name)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Missing column: {name}"),
            )
        })?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Type mismatch on column: {name}"),
            )
        })
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a StringArray> {
    col(batch, name)
}

fn u32_col<'a>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a UInt32Array> {
    col(batch, name)
}

fn u16_col<'a>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a UInt16Array> {
    col(batch, name)
}

fn bin_col<'a>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a LargeBinaryArray> {
    col(batch, name)
}

impl State {
    pub fn load_parquet_db(&mut self, path: &Path) {
        match ParquetMolDb::new(path) {
            Ok(db) => {
                handle_success(
                    &mut self.ui,
                    format!(
                        "Loaded Parquet database from {path:?} ({} molecules)",
                        db.index_meta.len()
                    ),
                );

                self.volatile.parquet_dbs.push(db);
                if self.volatile.parquet_dbs.len() == 1 {
                    self.volatile.parquet_db_active = Some(DbSel::Loaded(0));
                }

                self.update_history(path, OpenType::ParquetDb, None);
            }
            Err(e) => handle_err(&mut self.ui, format!("Error loading Parquet database: {e}")),
        }
    }

    /// The DB the UI is currently showing, if any.
    pub fn active_mol_db(&self) -> Option<&ParquetMolDb> {
        match self.volatile.parquet_db_active? {
            DbSel::Common => self.mol_db.as_ref(),
            DbSel::Loaded(i) => self.volatile.parquet_dbs.get(i),
        }
    }
}

/// Load the read-only database embedded in the binary. Returns `None` if it wasn't built into this
/// binary, or is unreadable; it's a convenience, and its absence shouldn't stop the app from
/// starting.
pub fn load_common_mol_db() -> Option<ParquetMolDb> {
    if COMMON_MOL_DB.is_empty() {
        eprintln!("No common molecule database embedded in this build.");
        return None;
    }

    match ParquetMolDb::from_embedded(COMMON_MOL_DB) {
        Ok(db) => {
            println!(
                "Loaded the built-in molecule database: {} molecules",
                db.index_meta.len()
            );
            Some(db)
        }
        Err(e) => {
            eprintln!("Error loading the built-in molecule database: {e}");
            None
        }
    }
}
