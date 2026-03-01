//! Small-molecule screening libraries using Apache Parquet

use std::{
    collections::HashMap,
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, StringArray, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, arrow_writer::ArrowWriter},
    basic::Compression,
    errors::ParquetError,
    file::properties::WriterProperties,
};

use crate::{
    mol_screening::{collect_mol_files, load_mol_batch},
    molecules::small::MoleculeSmall,
};

fn parquet_err_to_io(e: ParquetError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

fn arrow_err_to_io(e: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

/// One row in the Parquet file.
///
/// Keep "search columns" separate from source_path so you can scan/filter without
/// reloading every molecule from disk.
#[derive(Debug, Clone)]
struct StoredMol {
    ident: String,
    smiles: String,
    pubchem_cid: Option<u32>,
    drugbank_id: Option<String>,
    /// Absolute path to the original SDF / Mol2 file.
    source_path: String,
}

pub struct ParqetMolDb {
    parquet_path: PathBuf,

    // In-memory index: ident -> path of the source molecule file.
    index_by_ident: HashMap<String, PathBuf>,
}

impl ParqetMolDb {
    /// Create / open a DB at a parquet file path.
    ///
    /// This eagerly scans the ident and source_path columns to build an in-memory index.
    pub fn new(parquet_path: impl Into<PathBuf>) -> io::Result<Self> {
        let parquet_path = parquet_path.into();

        let mut db = Self {
            parquet_path,
            index_by_ident: HashMap::new(),
        };

        if db.parquet_path.exists() {
            db.rebuild_index()?;
        }

        Ok(db)
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("ident", DataType::Utf8, false),
            Field::new("smiles", DataType::Utf8, false),
            Field::new("pubchem_cid", DataType::UInt32, true),
            Field::new("drugbank_id", DataType::Utf8, true),
            Field::new("source_path", DataType::Utf8, false),
        ]))
    }

    /// Add molecules to a database. Loads recursively from a given folder (Mol2 or SDF),
    /// then writes a fresh parquet file.
    ///
    /// For "append", easiest is: read existing parquet, merge, rewrite.
    pub fn populate(&mut self, path: &Path) -> io::Result<()> {
        let files = collect_mol_files(path)?;

        let mut rows = Vec::new();
        let mut offset = 0;
        while offset < files.len() {
            let (mols, consumed) = load_mol_batch(&files[offset..])?;
            for (m, file_path) in mols.into_iter().zip(&files[offset..]) {
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

                let drugbank_id = m.idents.iter().find_map(|id| match id {
                    crate::molecules::MolIdent::DrugBank(s) => Some(s.clone()),
                    _ => None,
                });

                let source_path = file_path
                    .to_str()
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Non-UTF8 path in molecule file list",
                        )
                    })?
                    .to_string();

                rows.push(StoredMol {
                    ident: m.common.ident.clone(),
                    smiles,
                    pubchem_cid,
                    drugbank_id,
                    source_path,
                });
            }
            offset += consumed;
        }

        self.write_all_rows(&rows)?;
        self.rebuild_index()?;
        Ok(())
    }

    fn write_all_rows(&self, rows: &[StoredMol]) -> io::Result<()> {
        let file = File::create(&self.parquet_path)?;
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
        let ident_arr: StringArray = rows.iter().map(|r| Some(r.ident.as_str())).collect();
        let smiles_arr: StringArray = rows.iter().map(|r| Some(r.smiles.as_str())).collect();
        let pubchem_arr: UInt32Array = rows.iter().map(|r| r.pubchem_cid).collect();
        let drugbank_arr: StringArray = rows.iter().map(|r| r.drugbank_id.as_deref()).collect();
        let path_arr: StringArray = rows.iter().map(|r| Some(r.source_path.as_str())).collect();

        let cols: Vec<ArrayRef> = vec![
            Arc::new(ident_arr),
            Arc::new(smiles_arr),
            Arc::new(pubchem_arr),
            Arc::new(drugbank_arr),
            Arc::new(path_arr),
        ];

        RecordBatch::try_new(schema, cols).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    fn rebuild_index(&mut self) -> io::Result<()> {
        self.index_by_ident.clear();

        let file = File::open(&self.parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err_to_io)?;

        let arrow_schema = builder.schema().clone();

        let ident_idx = arrow_schema
            .index_of("ident")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let path_idx = arrow_schema
            .index_of("source_path")
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Project only the two columns we need for indexing.
        let mask =
            parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), [ident_idx, path_idx]);

        let mut reader = builder
            .with_projection(mask)
            .with_batch_size(8192)
            .build()
            .map_err(parquet_err_to_io)?;

        while let Some(batch) = reader.next().transpose().map_err(arrow_err_to_io)? {
            let ident_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "ident type mismatch"))?;
            let path_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "source_path type mismatch")
                })?;

            for i in 0..ident_col.len() {
                self.index_by_ident.insert(
                    ident_col.value(i).to_string(),
                    PathBuf::from(path_col.value(i)),
                );
            }
        }

        Ok(())
    }

    pub fn load_mol(&self, ident: &str) -> io::Result<MoleculeSmall> {
        let source_path = match self.index_by_ident.get(ident) {
            Some(p) => p.clone(),
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Molecule not found: {ident}"),
                ));
            }
        };

        let (mut mols, _) = load_mol_batch(&[source_path])?;
        mols.into_iter()
            .find(|m| m.common.ident == ident)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Molecule '{ident}' not found in its source file"),
                )
            })
    }

    pub fn load_all(&self) -> io::Result<Vec<MoleculeSmall>> {
        let paths: Vec<PathBuf> = self.index_by_ident.values().cloned().collect();
        let (mols, _) = load_mol_batch(&paths)?;
        Ok(mols)
    }

    pub fn load_mols(&self, idents: &[&str]) -> io::Result<Vec<MoleculeSmall>> {
        let mut out = Vec::with_capacity(idents.len());
        for id in idents {
            out.push(self.load_mol(id)?);
        }
        Ok(out)
    }
}
