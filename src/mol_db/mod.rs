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
    sync::Arc,
    thread,
    time::Duration,
};

use arrow::{
    array::{Array, ArrayRef, LargeBinaryArray, StringArray, UInt16Array, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bio_apis::pubchem::{self, StructureSearchNamespace};
use na_seq::Element;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
        arrow_writer::ArrowWriter,
    },
    basic::Compression,
    errors::ParquetError,
    file::properties::WriterProperties,
};

use crate::{
    mol_db::serialization::{
        idents_from_bytes, idents_to_bytes, metadata_from_bytes, metadata_to_bytes,
    },
    molecules::{MolIdent, small::MoleculeSmall},
    prefs::OpenType,
    screening::{collect_mol_files, load_mol_batch},
    state::State,
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
// pub const COMMON_MOL_DB: &[u8] = include_bytes!("../common_mol_db.parquet");

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
    fn from_mol(mut m: MoleculeSmall) -> io::Result<Self> {
        let smiles = smiles_from_idents(&m.idents).unwrap_or_else(|| m.common.ident.clone());
        let (mut pubchem_cid, mut pubchem_title) = pubchem_cid_title_from_idents(&m.idents);

        // Molecule files generally don't carry a title, so look it up over HTTP. Without one, the
        // DB table has only a SMILES string to identify the molecule by. The idents we store are
        // updated to match, so a molecule loaded back out of the DB carries them too.
        if pubchem_title.is_none()
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

/// A molecule's identifiers and metadata, as stored in the DB. Loaded on demand, and separately
/// from `mol_data`: screening workflows don't need these.
#[derive(Debug, Clone, Default)]
pub struct MolIdentsMeta {
    pub idents: Vec<MolIdent>,
    /// i.e. `common.metadata`
    pub metadata: HashMap<String, String>,
}

/// Struct representing the whole DB; used to open, update it, load data from disk in general.
pub struct ParquetMolDb {
    /// Path of the parquet file.
    pub path: PathBuf,
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
        let mut res = Self {
            path: path.to_owned(),
            index_meta: HashMap::new(),
            idents_cache: None,
        };

        if res.path.exists() {
            res.rebuild_index()?;
        }

        Ok(res)
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

    /// Read molecules from molecule files on disk, and loads them into a database. Loads
    /// recursively from a given folder (Mol2 or SDF), then writes a fresh parquet file.
    ///
    /// Parquet files are immutable, so to add molecules we read the existing rows, merge, and
    /// rewrite. Molecules already in the DB (matched on SMILES) are replaced by the incoming ones.
    pub fn populate(&mut self, mol_path: &Path) -> io::Result<()> {
        let files = collect_mol_files(mol_path)?;

        let mut rows = self.read_all_rows()?;
        let mut row_i = row_index(&rows);

        let mut offset = 0;

        while offset < files.len() {
            let (mols, consumed) = load_mol_batch(&files[offset..])?;

            for m in mols {
                merge_row(&mut rows, &mut row_i, StoredMol::from_mol(m)?);
            }
            offset += consumed;
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;

        Ok(())
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
        let mut rows = self.read_all_rows()?;
        let mut row_i = row_index(&rows);

        for m in mols {
            merge_row(&mut rows, &mut row_i, StoredMol::from_mol(m.clone())?);
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;

        Ok(())
    }

    /// Remove a molecule from the DB, by SMILES key. As with adding, parquet files are immutable,
    /// so the file is read, the row dropped, and the file rewritten.
    pub fn remove_mol(&mut self, smiles: &str) -> io::Result<()> {
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
                eprintln!("Error loading idents from {:?}: {e}", self.path);
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

    fn write_all_rows(&self, rows: &[StoredMol]) -> io::Result<()> {
        let file = File::create(&self.path)?;
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
        let has_title = has_cols(&self.path, &[COL_PUBCHEM_TITLE])?;

        let mut cols = vec![COL_SMILES, COL_PUBCHEM_CID, COL_HEAVY_ATOM_COUNT];
        if has_title {
            cols.push(COL_PUBCHEM_TITLE);
        }

        let mut reader = open_reader(&self.path, &cols)?;

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
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        // A DB written before we stored idents + metadata (or the title column) is still readable;
        // those rows simply come back empty, and gain the columns when the file is rewritten.
        let has_idents_meta = has_idents_meta_cols(&self.path)?;
        let has_title = has_cols(&self.path, &[COL_PUBCHEM_TITLE])?;

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

        let mut reader = open_reader(&self.path, &cols)?;

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

        let mut reader = open_reader(&self.path, &[COL_SMILES, COL_MOL_DATA])?;

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
        let mut reader = open_reader(&self.path, &[COL_MOL_DATA])?;

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
        if !has_idents_meta_cols(&self.path)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "This database was created before idents and metadata were stored. Re-add its \
                 molecules to populate them.",
            ));
        }

        let mut reader = open_reader(&self.path, &[COL_SMILES, COL_IDENTS, COL_METADATA])?;

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

/// Open the parquet file, reading only the columns named in `cols`. Columns not listed are not
/// read from disk. Note that the resulting batches keep the file's column order, not `cols`'; look
/// columns up by name. (See `str_col` etc)
fn open_reader(path: &Path, cols: &[&str]) -> io::Result<ParquetRecordBatchReader> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err_to_io)?;

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

/// Whether this file has all the named columns. Files written before a column was added to the
/// schema don't have it.
fn has_cols(path: &Path, cols: &[&str]) -> io::Result<bool> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err_to_io)?;
    let schema = builder.schema();

    Ok(cols.iter().all(|c| schema.index_of(c).is_ok()))
}

/// Whether this file has the `idents` and `metadata` columns. Files written before we stored them
/// don't.
fn has_idents_meta_cols(path: &Path) -> io::Result<bool> {
    has_cols(path, &[COL_IDENTS, COL_METADATA])
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
                    self.volatile.parquet_db_active = Some(0);
                }

                self.update_history(path, OpenType::ParquetDb, None);
            }
            Err(e) => handle_err(&mut self.ui, format!("Error loading Parquet database: {e}")),
        }
    }
}
