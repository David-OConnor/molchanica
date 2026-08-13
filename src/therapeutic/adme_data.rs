//! Reader and integrity checks for canonical ADME observation Parquet snapshots.

use std::{
    collections::HashMap,
    fs::{self, File},
    io,
    path::{Component, Path, PathBuf},
};

use arrow::{
    array::{Array, BooleanArray, Float64Array, Int64Array, StringArray},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    },
    errors::ParquetError,
};
use sha2::{Digest, Sha256};

use crate::therapeutic::DatasetTdc;

pub const CANONICAL_SCHEMA_VERSION: u32 = 1;
const CANONICAL_SUBDIR: &str = "canonical/v1";
const STRUCTURE_STANDARDIZATION: &str = "molchanica_parent_v1";
const BATCH_SIZE: usize = 8_192;

const COL_DATASET: &str = "dataset_name";
const COL_ROW_ID: &str = "source_row_id";
const COL_PARENT_KEY: &str = "parent_inchi_key";
const COL_VALUE: &str = "value";
const COL_QUALIFIER: &str = "value_qualifier";
const COL_CENSORED: &str = "value_is_censored";
const COL_SPLIT: &str = "split";
const COL_PARENT_SDF: &str = "parent_sdf_relative_path";
const COL_ELIGIBLE: &str = "training_eligible";
const COL_EXCLUSION: &str = "exclusion_reason";

const READER_COLUMNS: [&str; 10] = [
    COL_DATASET,
    COL_ROW_ID,
    COL_PARENT_KEY,
    COL_VALUE,
    COL_QUALIFIER,
    COL_CENSORED,
    COL_SPLIT,
    COL_PARENT_SDF,
    COL_ELIGIBLE,
    COL_EXCLUSION,
];

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct CanonicalObservation {
    pub source_row_id: usize,
    pub parent_inchi_key: String,
    pub value: f32,
    pub split: String,
    pub parent_sdf_relative_path: Option<PathBuf>,
    pub training_eligible: bool,
    pub exclusion_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct CanonicalAdmeDataset {
    pub observations: Vec<CanonicalObservation>,
}

impl CanonicalAdmeDataset {
    pub fn path(data_root: &Path, dataset: DatasetTdc) -> PathBuf {
        data_root
            .join(CANONICAL_SUBDIR)
            .join(format!("{}.observations.parquet", dataset.name()))
    }

    pub fn load(data_root: &Path, csv_path: &Path, dataset: DatasetTdc) -> io::Result<Self> {
        let parquet_path = Self::path(data_root, dataset);
        let (mut reader, metadata) = open_reader(&parquet_path)?;

        require_metadata(
            &metadata,
            "adme_schema_version",
            &CANONICAL_SCHEMA_VERSION.to_string(),
            &parquet_path,
        )?;
        require_metadata(&metadata, "dataset_name", &dataset.name(), &parquet_path)?;
        require_metadata(
            &metadata,
            "structure_standardization",
            STRUCTURE_STANDARDIZATION,
            &parquet_path,
        )?;
        for key in ["dataset_version", "generator_version", "rdkit_version"] {
            require_nonempty_metadata(&metadata, key, &parquet_path)?;
        }
        for key in ["endpoint_registry_sha256", "split_manifest_sha256"] {
            let value = require_nonempty_metadata(&metadata, key, &parquet_path)?;
            if !is_sha256_hex(value) {
                return Err(invalid_data(format!(
                    "{parquet_path:?} metadata {key:?} is not a SHA-256"
                )));
            }
        }
        let source_file = csv_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid_data(format!("Invalid CSV path: {csv_path:?}")))?;
        require_metadata(&metadata, "source_file", source_file, &parquet_path)?;
        let row_count = metadata
            .get("row_count")
            .ok_or_else(|| invalid_data(format!("{parquet_path:?} has no row_count metadata")))?
            .parse::<usize>()
            .map_err(|e| invalid_data(format!("Invalid row_count metadata: {e}")))?;
        let source_hash = sha256_file(csv_path)?;
        require_metadata(&metadata, "source_file_sha256", &source_hash, &parquet_path)?;
        let split_manifest_path = csv_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("split_manifests")
            .join(format!("{}.split.json", dataset.name()));
        let split_hash = sha256_file(&split_manifest_path)?;
        require_metadata(
            &metadata,
            "split_manifest_sha256",
            &split_hash,
            &parquet_path,
        )?;

        let mut observations = Vec::with_capacity(row_count);
        for batch in &mut reader {
            append_batch(
                &mut observations,
                &batch.map_err(io::Error::other)?,
                dataset,
            )?;
        }
        if observations.len() != row_count {
            return Err(invalid_data(format!(
                "Canonical Parquet has {} rows, metadata declares {row_count}",
                observations.len()
            )));
        }
        for (expected, observation) in observations.iter().enumerate() {
            if observation.source_row_id != expected {
                return Err(invalid_data(format!(
                    "Canonical source row IDs are incomplete or out of order: expected {expected}, got {}",
                    observation.source_row_id
                )));
            }
            validate_observation_path(data_root, observation)?;
        }
        validate_csv_targets(csv_path, &observations)?;

        Ok(Self { observations })
    }
}

fn open_reader(path: &Path) -> io::Result<(ParquetRecordBatchReader, HashMap<String, String>)> {
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(File::open(path)?).map_err(parquet_error)?;
    let schema = builder.schema().clone();
    let metadata = schema.metadata().clone();
    let mut indices = Vec::with_capacity(READER_COLUMNS.len());
    for name in READER_COLUMNS {
        indices.push(schema.index_of(name).map_err(io::Error::other)?);
    }
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(BATCH_SIZE)
        .build()
        .map_err(parquet_error)?;
    Ok((reader, metadata))
}

fn append_batch(
    observations: &mut Vec<CanonicalObservation>,
    batch: &RecordBatch,
    requested_dataset: DatasetTdc,
) -> io::Result<()> {
    let datasets = string_column(batch, COL_DATASET)?;
    let row_ids = column::<Int64Array>(batch, COL_ROW_ID)?;
    let parent_keys = string_column(batch, COL_PARENT_KEY)?;
    let values = column::<Float64Array>(batch, COL_VALUE)?;
    let qualifiers = string_column(batch, COL_QUALIFIER)?;
    let censored = column::<BooleanArray>(batch, COL_CENSORED)?;
    let splits = string_column(batch, COL_SPLIT)?;
    let sdf_paths = string_column(batch, COL_PARENT_SDF)?;
    let eligible = column::<BooleanArray>(batch, COL_ELIGIBLE)?;
    let exclusions = string_column(batch, COL_EXCLUSION)?;

    for row in 0..batch.num_rows() {
        let dataset = required_string(datasets, row, COL_DATASET)?;
        if dataset != requested_dataset.name() {
            return Err(invalid_data(format!(
                "Canonical row belongs to {dataset:?}, requested {:?}",
                requested_dataset.name()
            )));
        }
        if row_ids.is_null(row)
            || values.is_null(row)
            || censored.is_null(row)
            || eligible.is_null(row)
        {
            return Err(invalid_data(
                "Canonical training columns contain null values",
            ));
        }
        let row_id = usize::try_from(row_ids.value(row))
            .map_err(|_| invalid_data("Canonical source_row_id is negative or too large"))?;
        let value = values.value(row);
        if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
            return Err(invalid_data(format!(
                "Canonical target at source row {row_id} is not a finite f32"
            )));
        }
        if required_string(qualifiers, row, COL_QUALIFIER)? != "=" || censored.value(row) {
            return Err(invalid_data(format!(
                "Training does not yet support censored/qualified value at source row {row_id}"
            )));
        }
        let parent_key = required_string(parent_keys, row, COL_PARENT_KEY)?;
        if !is_standard_inchi_key(parent_key) {
            return Err(invalid_data(format!(
                "Invalid parent InChIKey at source row {row_id}: {parent_key:?}"
            )));
        }
        let split = required_string(splits, row, COL_SPLIT)?;
        if !matches!(split, "train" | "validation" | "test") {
            return Err(invalid_data(format!(
                "Invalid canonical split {split:?} at source row {row_id}"
            )));
        }
        let is_eligible = eligible.value(row);
        let sdf_path = optional_string(sdf_paths, row).map(PathBuf::from);
        let exclusion_reason = optional_string(exclusions, row).map(str::to_owned);
        if is_eligible && (sdf_path.is_none() || exclusion_reason.is_some()) {
            return Err(invalid_data(format!(
                "Eligible source row {row_id} must have a parent SDF and no exclusion reason"
            )));
        }
        if !is_eligible && exclusion_reason.is_none() {
            return Err(invalid_data(format!(
                "Excluded source row {row_id} has no exclusion reason"
            )));
        }
        observations.push(CanonicalObservation {
            source_row_id: row_id,
            parent_inchi_key: parent_key.to_owned(),
            value: value as f32,
            split: split.to_owned(),
            parent_sdf_relative_path: sdf_path,
            training_eligible: is_eligible,
            exclusion_reason,
        });
    }
    Ok(())
}

fn validate_observation_path(
    data_root: &Path,
    observation: &CanonicalObservation,
) -> io::Result<()> {
    let Some(relative) = &observation.parent_sdf_relative_path else {
        return Ok(());
    };
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid_data(format!(
            "Unsafe parent SDF path at source row {}: {relative:?}",
            observation.source_row_id
        )));
    }
    if observation.training_eligible {
        let path = data_root.join(relative);
        let metadata = fs::metadata(&path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "Eligible source row {} has no readable parent SDF at {path:?}: {e}",
                    observation.source_row_id
                ),
            )
        })?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(invalid_data(format!(
                "Eligible source row {} has an empty/non-file parent SDF at {path:?}",
                observation.source_row_id
            )));
        }
    }
    Ok(())
}

fn require_metadata(
    metadata: &HashMap<String, String>,
    key: &str,
    expected: &str,
    path: &Path,
) -> io::Result<()> {
    let actual = metadata
        .get(key)
        .ok_or_else(|| invalid_data(format!("{path:?} has no {key:?} metadata")))?;
    if actual != expected {
        return Err(invalid_data(format!(
            "{path:?} metadata {key:?} is {actual:?}, expected {expected:?}; regenerate canonical data"
        )));
    }
    Ok(())
}

fn require_nonempty_metadata<'a>(
    metadata: &'a HashMap<String, String>,
    key: &str,
    path: &Path,
) -> io::Result<&'a str> {
    let value = metadata
        .get(key)
        .ok_or_else(|| invalid_data(format!("{path:?} has no {key:?} metadata")))?;
    if value.trim().is_empty() {
        return Err(invalid_data(format!("{path:?} metadata {key:?} is blank")));
    }
    Ok(value)
}

fn validate_csv_targets(csv_path: &Path, observations: &[CanonicalObservation]) -> io::Result<()> {
    let mut reader = csv::Reader::from_path(csv_path)?;
    let headers = reader.headers()?.clone();
    let target_column = headers
        .iter()
        .position(|header| {
            matches!(
                header.trim().to_ascii_lowercase().as_str(),
                "y" | "label" | "target"
            )
        })
        .ok_or_else(|| invalid_data(format!("No target column in {csv_path:?}")))?;
    let mut row_count = 0usize;
    for (row_id, record) in reader.records().enumerate() {
        let record = record?;
        let raw = record
            .get(target_column)
            .ok_or_else(|| invalid_data(format!("CSV source row {row_id} has no target")))?;
        let source_value = raw.trim().parse::<f64>().map_err(|error| {
            invalid_data(format!(
                "CSV source row {row_id} has invalid target {raw:?}: {error}"
            ))
        })?;
        if !source_value.is_finite()
            || source_value < f32::MIN as f64
            || source_value > f32::MAX as f64
        {
            return Err(invalid_data(format!(
                "CSV source row {row_id} has target outside finite f32 range"
            )));
        }
        let observation = observations
            .get(row_id)
            .ok_or_else(|| invalid_data(format!("Canonical data omits CSV source row {row_id}")))?;
        if observation.value.to_bits() != (source_value as f32).to_bits() {
            return Err(invalid_data(format!(
                "Canonical target differs from CSV at source row {row_id}: parquet={}, csv={source_value}",
                observation.value
            )));
        }
        row_count += 1;
    }
    if row_count != observations.len() {
        return Err(invalid_data(format!(
            "CSV has {row_count} rows, canonical Parquet has {}",
            observations.len()
        )));
    }
    Ok(())
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a T> {
    batch
        .column_by_name(name)
        .ok_or_else(|| invalid_data(format!("Missing canonical column {name:?}")))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| invalid_data(format!("Canonical column {name:?} has the wrong type")))
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> io::Result<&'a StringArray> {
    column(batch, name)
}

fn required_string<'a>(column: &'a StringArray, row: usize, name: &str) -> io::Result<&'a str> {
    if column.is_null(row) {
        Err(invalid_data(format!(
            "Canonical column {name:?} is null at batch row {row}"
        )))
    } else {
        Ok(column.value(row))
    }
}

fn optional_string(column: &StringArray, row: usize) -> Option<&str> {
    (!column.is_null(row)).then(|| column.value(row))
}

fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn is_standard_inchi_key(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 27
        && bytes[14] == b'-'
        && bytes[25] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, byte)| i == 14 || i == 25 || byte.is_ascii_uppercase())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parquet_error(error: ParquetError) -> io::Error {
    io::Error::other(error)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_observation(value: f32) -> CanonicalObservation {
        CanonicalObservation {
            source_row_id: 0,
            parent_inchi_key: "AAAAAAAAAAAAAA-BBBBBBBBBB-C".to_owned(),
            value,
            split: "train".to_owned(),
            parent_sdf_relative_path: None,
            training_eligible: false,
            exclusion_reason: Some("fixture".to_owned()),
        }
    }

    #[test]
    fn rejects_parent_directory_in_structure_path() {
        let mut observation = fixture_observation(1.0);
        observation.source_row_id = 7;
        observation.parent_sdf_relative_path = Some(PathBuf::from("../escape.sdf"));
        assert!(validate_observation_path(Path::new("fixture"), &observation).is_err());
    }

    #[test]
    fn rejects_target_that_differs_from_source_csv() {
        let path = std::env::temp_dir().join(format!(
            "molchanica-adme-target-{}-{}.csv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, "Drug,Y\nCC,1.25\n").unwrap();
        let result = validate_csv_targets(&path, &[fixture_observation(1.5)]);
        fs::remove_file(path).unwrap();
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires TDC_DATA_DIR to point at migrated canonical snapshots"]
    fn loads_all_canonical_snapshots() {
        let root = PathBuf::from(
            std::env::var_os("TDC_DATA_DIR")
                .expect("set TDC_DATA_DIR to the migrated TDC snapshot directory"),
        );
        for dataset in DatasetTdc::all() {
            let csv = root.join(format!("{}.csv", dataset.name()));
            let canonical = CanonicalAdmeDataset::load(&root, &csv, dataset).unwrap();
            assert!(!canonical.observations.is_empty());
        }
    }
}
