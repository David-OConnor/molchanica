//! Small-molecule screening libraries using Apache Parquet
//!
//! Note: feather/ipc may have better read speeds for mol screening, as an alternative to Parquet.
//! Note: We're currently using "SMILES" everywhere here that we list "ident", including to index
//! rows.

use std::{
    collections::HashMap,
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, LargeBinaryArray, StringArray, UInt16Array, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bio_files::BondType;
use lin_alg::f32::{Vec3 as Vec3f32, Vec3};
use na_seq::Element;
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, arrow_writer::ArrowWriter},
    basic::Compression,
    errors::ParquetError,
    file::properties::WriterProperties,
};

use crate::prefs::OpenType;
use crate::state::State;
use crate::util::{handle_err, handle_success};
use crate::{
    molecules::{Atom, Bond, small::MoleculeSmall},
    screening::{collect_mol_files, load_mol_batch},
};

fn parquet_err_to_io(e: ParquetError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

fn arrow_err_to_io(e: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

/// One row in the Parquet file.
///
/// Keep "search columns" separate from mol_data so you can scan/filter without
/// deserializing every molecule. I believe we can load data column by column, so
/// we can load idents of various types all at once, and other metadata of interest, without
/// loading all data.
///
/// todo: What other data should we include here?
#[derive(Debug, Clone)]
struct StoredMol {
    smiles: String,
    pubchem_cid: Option<u32>,
    // drugbank_id: Option<String>,
    heavy_atom_count: u16,
    /// Atoms, bonds, etc.
    mol_data: Vec<u8>,
}

/// Lightweight metadata for a molecule stored in the DB — excludes the heavy `mol_data` blob.
#[derive(Debug, Clone)]
pub struct MolMeta {
    pub smiles: String,
    pub pubchem_cid: Option<u32>,
    pub heavy_atom_count: u16,
}

pub struct ParquetMolDb {
    pub path: PathBuf,
    /// Lightweight metadata index loaded eagerly on open: smiles → MolMeta.
    /// Does NOT include the heavy mol_data blob; that is read from disk on demand.
    pub index_meta: HashMap<String, MolMeta>,
}

impl ParquetMolDb {
    /// Create / open a DB at a parquet file path.
    ///
    /// Eagerly loads only the lightweight metadata columns (smiles, pubchem_cid,
    /// heavy_atom_count) into `index_meta`. The heavy `mol_data` column is NOT loaded
    /// here; it is read from disk on demand via `load_all` / `load_mol` / `load_mols`.
    pub fn new(path: &Path) -> io::Result<Self> {
        let mut res = Self {
            path: path.to_owned(),
            index_meta: HashMap::new(),
        };

        if res.path.exists() {
            res.rebuild_index()?;
        }

        Ok(res)
    }

    /// Keep this in sync with `StoredMol`
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("smiles", DataType::Utf8, false),
            Field::new("pubchem_cid", DataType::UInt32, true),
            // Field::new("drugbank_id", DataType::Utf8, true),
            Field::new("heavy_atom_count", DataType::UInt16, false),
            // This contains our atoms, bonds, etc. Serialized as binary.
            Field::new("mol_data", DataType::LargeBinary, false),
        ]))
    }

    /// Read molecules from molecule files on disk, and loads them into a database. Loads
    /// recursively from a given folder (Mol2 or SDF), then writes a fresh parquet file.
    ///
    /// For "append", easiest is: read existing parquet, merge, rewrite.
    pub fn populate(&mut self, mol_path: &Path) -> io::Result<()> {
        let files = collect_mol_files(mol_path)?;

        let mut rows = Vec::new();
        let mut offset = 0;
        while offset < files.len() {
            let (mols, consumed) = load_mol_batch(&files[offset..])?;
            for m in mols {
                let smiles = m
                    .idents
                    .iter()
                    .find_map(|id| match id {
                        crate::molecules::MolIdent::Smiles(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| m.common.ident.clone());

                let pubchem_cid = m.idents.iter().find_map(|id| match id {
                    crate::molecules::MolIdent::PubChem(cid) => Some(*cid),
                    _ => None,
                });

                // let drugbank_id = m.idents.iter().find_map(|id| match id {
                //     crate::molecules::MolIdent::DrugBank(s) => Some(s.clone()),
                //     _ => None,
                // });

                let heavy_atom_count = m
                    .common
                    .atoms
                    .iter()
                    .filter(|m_| m_.element != Element::Hydrogen)
                    .count() as u16;

                let mol_data = m.to_bytes();

                rows.push(StoredMol {
                    smiles,
                    pubchem_cid,
                    // drugbank_id,
                    heavy_atom_count,
                    mol_data,
                });
            }
            offset += consumed;
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;
        Ok(())
    }

    fn write_all_rows(&self, rows: &[StoredMol]) -> io::Result<()> {
        let file = File::create(&self.path)?;
        let schema = Self::schema();

        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(parquet_err_to_io)?;

        const BATCH_SIZE: usize = 2048;
        for chunk in rows.chunks(BATCH_SIZE) {
            let batch = Self::make_batch(chunk, schema.clone())?;
            writer.write(&batch).map_err(parquet_err_to_io)?;
        }

        writer.close().map_err(parquet_err_to_io)?;
        Ok(())
    }

    fn make_batch(rows: &[StoredMol], schema: Arc<Schema>) -> io::Result<RecordBatch> {
        let smiles_arr: StringArray = rows.iter().map(|r| Some(r.smiles.as_str())).collect();
        let pubchem_arr: UInt32Array = rows.iter().map(|r| r.pubchem_cid).collect();
        // let drugbank_arr: StringArray = rows.iter().map(|r| r.drugbank_id.as_deref()).collect();

        let heavy_count_arr: UInt16Array = rows.iter().map(|r| r.heavy_atom_count).collect();

        let mol_data_arr: LargeBinaryArray =
            rows.iter().map(|r| Some(r.mol_data.as_slice())).collect();

        let cols: Vec<ArrayRef> = vec![
            Arc::new(smiles_arr),
            Arc::new(pubchem_arr),
            // Arc::new(drugbank_arr),
            Arc::new(heavy_count_arr),
            Arc::new(mol_data_arr),
        ];

        RecordBatch::try_new(schema, cols).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    /// Reads only the three lightweight metadata columns from disk into `index_meta`.
    /// The heavy `mol_data` column is intentionally excluded.
    fn rebuild_index(&mut self) -> io::Result<()> {
        self.index_meta.clear();

        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err_to_io)?;

        let arrow_schema = builder.schema().clone();

        let smiles_idx = arrow_schema
            .index_of("smiles")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let pubchem_idx = arrow_schema
            .index_of("pubchem_cid")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let heavy_atom_count_idx = arrow_schema
            .index_of("heavy_atom_count")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Project metadata columns only. mol_data is NOT read here.
        // Projected batch column order: smiles(0), pubchem_cid(1), heavy_atom_count(2).
        let mask = parquet::arrow::ProjectionMask::roots(
            builder.parquet_schema(),
            [smiles_idx, pubchem_idx, heavy_atom_count_idx],
        );

        let mut reader = builder
            .with_projection(mask)
            .with_batch_size(8192)
            .build()
            .map_err(parquet_err_to_io)?;

        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let smiles_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "smiles type mismatch")
                })?;
            let pubchem_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "pubchem_cid type mismatch")
                })?;
            let heavy_atom_count_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "heavy_atom_count type mismatch")
                })?;

            for i in 0..smiles_col.len() {
                let smiles = smiles_col.value(i).to_string();
                self.index_meta.insert(
                    smiles.clone(),
                    MolMeta {
                        smiles,
                        pubchem_cid: if pubchem_col.is_null(i) {
                            None
                        } else {
                            Some(pubchem_col.value(i))
                        },
                        heavy_atom_count: heavy_atom_count_col.value(i),
                    },
                );
            }
        }

        Ok(())
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
        use std::collections::HashSet;
        let targets: HashSet<&str> = smiles_keys.iter().copied().collect();

        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err_to_io)?;

        let arrow_schema = builder.schema().clone();

        let smiles_idx = arrow_schema
            .index_of("smiles")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let mol_data_idx = arrow_schema
            .index_of("mol_data")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let mask = parquet::arrow::ProjectionMask::roots(
            builder.parquet_schema(),
            [smiles_idx, mol_data_idx],
        );

        let mut reader = builder
            .with_projection(mask)
            .with_batch_size(8192)
            .build()
            .map_err(parquet_err_to_io)?;

        let mut result = Vec::new();
        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let smiles_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "smiles type mismatch")
                })?;
            let mol_data_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "mol_data type mismatch")
                })?;
            for i in 0..smiles_col.len() {
                if targets.contains(smiles_col.value(i)) {
                    result.push(MoleculeSmall::from_bytes(mol_data_col.value(i))?);
                }
            }
        }

        Ok(result)
    }

    /// Read all `mol_data` from disk and deserialize into molecules.
    /// This is the only function that actually reads mol_data from the Parquet file.
    pub fn load_all(&self) -> io::Result<Vec<MoleculeSmall>> {
        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err_to_io)?;

        let mol_data_idx = builder
            .schema()
            .index_of("mol_data")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let mask = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), [mol_data_idx]);

        let mut reader = builder
            .with_projection(mask)
            .with_batch_size(8192)
            .build()
            .map_err(parquet_err_to_io)?;

        let mut result = Vec::with_capacity(self.index_meta.len());
        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let mol_data_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "mol_data type mismatch")
                })?;
            for i in 0..mol_data_col.len() {
                result.push(MoleculeSmall::from_bytes(mol_data_col.value(i))?);
            }
        }

        Ok(result)
    }
}

// ── Binary serialization ──────────────────────────────────────────────────────
//
// Layout for Atom:
//   [0..4]   serial_number: u32 LE
//   [4]      element: u8  (atomic number)
//   [5..17]  posit: 3 × f32 LE  (12 bytes; precision loss accepted for screening)
//   [17]     type_in_res_general length: u8
//   [18..]   type_in_res_general: UTF-8 bytes
//
// Layout for Bond (fixed 9 bytes):
//   [0]      bond_type: u8
//   [1..5]   atom_0_sn: u32 LE
//   [5..9]   atom_1_sn: u32 LE
//
// Layout for MoleculeSmall:
//   [0]      ident length: u8
//   [1..]    ident: UTF-8 bytes
//   [..]     atom_count: u16 LE
//   [..]     for each atom: atom_len: u16 LE, then atom_len bytes
//   [..]     bond_count: u16 LE
//   [..]     for each bond: 9 bytes (fixed)

impl Atom {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::new();

        // serial_number: u32 LE
        res.extend_from_slice(&self.serial_number.to_le_bytes());

        // element: u8
        res.push(self.element.atomic_number());

        // posit: 3 × f32 LE  (convert from f64)
        let posit: Vec3f32 = self.posit.into();
        res.extend_from_slice(&posit.to_le_bytes());

        // type_in_res_general: length-prefixed UTF-8
        let atom_type = self.type_in_res_general.as_deref().unwrap_or("");
        let atom_type_bytes = atom_type.as_bytes();
        res.push(atom_type_bytes.len() as u8);
        res.extend_from_slice(atom_type_bytes);

        res
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut i = 0;

        let serial_number = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        i += 4;

        let element = Element::from_atomic_number(bytes[i])?;
        i += 1;

        let posit = Vec3::from_le_bytes(bytes[i..i + 12].try_into().unwrap());
        i += 12;

        let type_len = bytes[i] as usize;
        i += 1;
        let type_in_res_general = if type_len == 0 {
            None
        } else {
            Some(
                String::from_utf8(bytes[i..i + type_len].to_vec())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            )
        };

        Ok(Self {
            serial_number,
            posit: posit.into(),
            element,
            type_in_res_general,
            ..Default::default()
        })
    }
}

impl Bond {
    pub fn to_bytes(&self) -> [u8; 9] {
        let mut res = [0u8; 9];
        res[0] = match self.bond_type {
            BondType::Single => 0,
            BondType::Double => 1,
            BondType::Triple => 2,
            BondType::Aromatic => 3,
            BondType::Amide => 4,
            BondType::Dummy => 5,
            BondType::Unknown => 6,
            BondType::NotConnected => 7,
            BondType::Quadruple => 8,
            BondType::Delocalized => 9,
            BondType::PolymericLink => 10,
        };
        res[1..5].copy_from_slice(&self.atom_0_sn.to_le_bytes());
        res[5..9].copy_from_slice(&self.atom_1_sn.to_le_bytes());
        res
    }

    pub fn from_bytes(bytes: &[u8; 9]) -> io::Result<Self> {
        let bond_type = match bytes[0] {
            0 => BondType::Single,
            1 => BondType::Double,
            2 => BondType::Triple,
            3 => BondType::Aromatic,
            4 => BondType::Amide,
            5 => BondType::Dummy,
            6 => BondType::Unknown,
            7 => BondType::NotConnected,
            8 => BondType::Quadruple,
            9 => BondType::Delocalized,
            10 => BondType::PolymericLink,
            b => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown bond type byte {b}"),
                ));
            }
        };
        let atom_0_sn = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
        let atom_1_sn = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
        Ok(Bond {
            bond_type,
            atom_0_sn,
            atom_1_sn,
            atom_0: 0,
            atom_1: 0,
            is_backbone: false,
        })
    }
}

impl MoleculeSmall {
    /// Serialize to a compact binary form for embedding in the Parquet mol_data column.
    ///
    /// We only include the information critical to ser/deser for screening.
    /// Note: We potentially lose precision information due to using f32 coordinates.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::new();

        // ident
        let ident = self.common.ident.as_bytes();
        res.push(ident.len() as u8);
        res.extend_from_slice(ident);

        // atoms
        res.extend_from_slice(&(self.common.atoms.len() as u16).to_le_bytes());
        for atom in &self.common.atoms {
            let atom_bytes = atom.to_bytes();
            res.extend_from_slice(&(atom_bytes.len() as u16).to_le_bytes());
            res.extend_from_slice(&atom_bytes);
        }

        // bonds
        res.extend_from_slice(&(self.common.bonds.len() as u16).to_le_bytes());
        for bond in &self.common.bonds {
            res.extend_from_slice(&bond.to_bytes());
        }

        res
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut i = 0;

        // ident
        let ident_len = bytes[i] as usize;
        i += 1;
        let ident = String::from_utf8(bytes[i..i + ident_len].to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        i += ident_len;

        // atoms
        let atom_count = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
        i += 2;
        let mut atoms = Vec::with_capacity(atom_count);
        for _ in 0..atom_count {
            let atom_len = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
            i += 2;
            atoms.push(Atom::from_bytes(&bytes[i..i + atom_len])?);
            i += atom_len;
        }

        // bonds — resolve atom indices from serial numbers after parsing
        let bond_count = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
        i += 2;
        let mut bonds = Vec::with_capacity(bond_count);
        for _ in 0..bond_count {
            let mut bond = Bond::from_bytes(bytes[i..i + 9].try_into().unwrap())?;
            bond.atom_0 = atoms
                .iter()
                .position(|a| a.serial_number == bond.atom_0_sn)
                .unwrap_or(0);
            bond.atom_1 = atoms
                .iter()
                .position(|a| a.serial_number == bond.atom_1_sn)
                .unwrap_or(0);
            bonds.push(bond);
            i += 9;
        }

        Ok(Self::new(ident, atoms, bonds, HashMap::new(), None))
    }
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

                self.update_history(path, OpenType::ParquetDb);
            }
            Err(e) => handle_err(&mut self.ui, format!("Error loading Parquet database: {e}")),
        }
    }
}
