"""Build versioned canonical Parquet observations from local TDC CSV snapshots.

This is a lossless migration: every original source column is retained in
``source_raw_json`` and the original assayed SMILES is never replaced by the
standardized parent used for grouping and ML.

Example:

    python scripts/build_adme_parquet.py \
        --data-dir C:/Users/the_a/Desktop/tdc_data \
        --dataset-version tdc-local-2026-01-v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adme_data_model import (
    RDKit_VERSION,
    STRUCTURE_STANDARDIZATION,
    load_structure_overrides,
    sha256_bytes,
    sha256_file,
    standardize_structure,
    validate_assayed_sdf,
    validate_parent_sdf,
    validated_override,
)

SCHEMA_VERSION = 1
GENERATOR_VERSION = "molchanica_adme_parquet_v1"
CANONICAL_SUBDIR = Path("canonical") / "v1"
SPLITS = ("train", "validation", "test")
SMILES_COLUMNS = ("Drug", "X", "SMILES")
TARGET_COLUMNS = ("Y", "label", "target")
SOURCE_ID_COLUMNS = ("Drug_ID", "ID", "drug_id", "id")

# Completeness is descriptive, not a filter.  Some fields do not apply to every
# endpoint, but keeping one stable list makes cross-source audits comparable.
CONTEXT_FIELDS = (
    "species",
    "tissue",
    "cell_line",
    "biological_matrix",
    "assay_method",
    "assay_direction",
    "assay_ph",
    "temperature_c",
    "concentration_value",
    "dose_value",
    "dose_route",
    "formulation",
    "source_publication",
    "source_assay_id",
    "source_license",
    "replicate_id",
    "measurement_variance",
)


def canonical_schema(metadata: dict[bytes, bytes] | None = None) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("observation_id", pa.string(), nullable=False),
            pa.field("dataset_name", pa.string(), nullable=False),
            pa.field("dataset_version", pa.string(), nullable=False),
            pa.field("source_row_id", pa.int64(), nullable=False),
            pa.field("source_record_id", pa.string()),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("source_file_sha256", pa.string(), nullable=False),
            pa.field("source_raw_json", pa.large_string(), nullable=False),
            pa.field("source_collection", pa.string(), nullable=False),
            pa.field("source_url", pa.string()),
            pa.field("source_publication", pa.string()),
            pa.field("source_assay_id", pa.string()),
            pa.field("source_license", pa.string()),
            pa.field("tdc_distribution_license", pa.string()),
            pa.field("quality_tier", pa.string(), nullable=False),
            pa.field("metadata_completeness", pa.float64(), nullable=False),
            pa.field("metadata_missing_fields", pa.list_(pa.string()), nullable=False),
            pa.field("provenance_json", pa.large_string(), nullable=False),
            pa.field("upstream_link_status", pa.string(), nullable=False),
            pa.field("upstream_link_method", pa.string()),
            pa.field("upstream_link_confidence", pa.float64()),
            pa.field("upstream_record_id", pa.string()),
            pa.field("assayed_smiles", pa.string(), nullable=False),
            pa.field("assayed_canonical_smiles", pa.string()),
            pa.field("assayed_inchi", pa.string()),
            pa.field("assayed_inchi_key", pa.string()),
            pa.field("assayed_component_count", pa.int32()),
            pa.field("assayed_form_kind", pa.string(), nullable=False),
            pa.field("parent_smiles", pa.string(), nullable=False),
            pa.field("parent_inchi", pa.string(), nullable=False),
            pa.field("parent_inchi_key", pa.string(), nullable=False),
            pa.field("parent_scaffold", pa.string(), nullable=False),
            pa.field("removed_components", pa.list_(pa.string()), nullable=False),
            pa.field("structure_standardization", pa.string(), nullable=False),
            pa.field("structure_override_applied", pa.bool_(), nullable=False),
            pa.field("standardization_warnings", pa.list_(pa.string()), nullable=False),
            pa.field("endpoint_domain", pa.string(), nullable=False),
            pa.field("endpoint_name", pa.string(), nullable=False),
            pa.field("readout", pa.string(), nullable=False),
            pa.field("task_type", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("value_unit", pa.string()),
            pa.field("value_qualifier", pa.string(), nullable=False),
            pa.field("value_lower_bound", pa.float64()),
            pa.field("value_upper_bound", pa.float64()),
            pa.field("value_is_censored", pa.bool_(), nullable=False),
            pa.field("species", pa.string()),
            pa.field("strain", pa.string()),
            pa.field("tissue", pa.string()),
            pa.field("cell_line", pa.string()),
            pa.field("biological_matrix", pa.string()),
            pa.field("assay_method", pa.string()),
            pa.field("assay_direction", pa.string()),
            pa.field("assay_ph", pa.float64()),
            pa.field("temperature_c", pa.float64()),
            pa.field("concentration_value", pa.float64()),
            pa.field("concentration_unit", pa.string()),
            pa.field("dose_value", pa.float64()),
            pa.field("dose_unit", pa.string()),
            pa.field("dose_route", pa.string()),
            pa.field("formulation", pa.string()),
            pa.field("replicate_id", pa.string()),
            pa.field("replicate_count", pa.int32()),
            pa.field("measurement_stddev", pa.float64()),
            pa.field("measurement_variance", pa.float64()),
            pa.field("split", pa.string(), nullable=False),
            pa.field("split_manifest_sha256", pa.string(), nullable=False),
            pa.field("parent_sdf_relative_path", pa.string()),
            pa.field("assayed_sdf_relative_path", pa.string()),
            pa.field("training_eligible", pa.bool_(), nullable=False),
            pa.field("exclusion_reason", pa.string()),
            pa.field("eligibility_basis", pa.string(), nullable=False),
        ]
    )
    return schema.with_metadata(metadata) if metadata else schema


def _find_column(header: list[str], candidates: tuple[str, ...], kind: str) -> str:
    by_lower = {column.strip().lower(): column for column in header}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(f"No {kind} column in {header!r}; expected one of {candidates!r}")


def _optional_source_id_column(header: list[str]) -> str | None:
    by_lower = {column.strip().lower(): column for column in header}
    for candidate in SOURCE_ID_COLUMNS:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _load_registry(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported endpoint-registry schema at {path}")
    if not isinstance(payload.get("datasets"), dict) or not isinstance(
        payload.get("defaults"), dict
    ):
        raise ValueError(f"Invalid endpoint registry at {path}")
    return payload, sha256_bytes(raw)


def _endpoint_metadata(registry: dict, dataset: str) -> dict:
    try:
        endpoint = registry["datasets"][dataset]
    except KeyError as error:
        raise ValueError(f"No endpoint registry entry for {dataset}") from error
    result = dict(registry["defaults"])
    result.update(endpoint)
    required = ("domain", "endpoint_name", "readout", "task_type", "quality_tier")
    missing = [key for key in required if not result.get(key)]
    if missing:
        raise ValueError(f"Endpoint registry entry {dataset} is missing {missing}")
    return result


def _load_split_manifest(
    data_dir: Path, dataset: str, csv_hash: str
) -> tuple[dict, Path, str]:
    path = data_dir / "split_manifests" / f"{dataset}.split.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported split manifest: {path}")
    info = manifest.get("dataset", {})
    if info.get("name") != dataset or info.get("source_file_sha256") != csv_hash:
        raise ValueError(f"Stale or mismatched split manifest: {path}")
    return manifest, path, sha256_bytes(raw)


def _safe_relative_file(data_dir: Path, path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    resolved_root = data_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Structure path escapes data directory: {resolved_path}"
        ) from error
    return relative.as_posix()


def _metadata_completeness(observation: dict) -> tuple[float, list[str]]:
    missing = [field for field in CONTEXT_FIELDS if observation.get(field) is None]
    return (len(CONTEXT_FIELDS) - len(missing)) / len(CONTEXT_FIELDS), missing


def _parquet_metadata(
    *,
    dataset: str,
    dataset_version: str,
    source_file: str,
    source_hash: str,
    split_manifest_hash: str,
    registry_hash: str,
    override_hash: str | None,
    row_count: int,
) -> dict[bytes, bytes]:
    values = {
        "adme_schema_version": str(SCHEMA_VERSION),
        "generator_version": GENERATOR_VERSION,
        "dataset_name": dataset,
        "dataset_version": dataset_version,
        "source_file": source_file,
        "source_file_sha256": source_hash,
        "split_manifest_sha256": split_manifest_hash,
        "endpoint_registry_sha256": registry_hash,
        "structure_overrides_sha256": override_hash or "none",
        "structure_standardization": STRUCTURE_STANDARDIZATION,
        "rdkit_version": RDKit_VERSION,
        "row_count": str(row_count),
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".parquet", delete=False
    ) as temp:
        temp_path = Path(temp.name)
    try:
        pq.write_table(
            table,
            temp_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(encoded)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def build_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    dataset_version: str,
    registry: dict,
    registry_hash: str,
    overrides: dict,
    override_hash: str | None,
) -> tuple[Path, dict]:
    csv_path = data_dir / f"{dataset}.csv"
    source_hash = sha256_file(csv_path)
    manifest, manifest_path, manifest_hash = _load_split_manifest(
        data_dir, dataset, source_hash
    )
    endpoint = _endpoint_metadata(registry, dataset)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No header in {csv_path}")
        smiles_col = _find_column(reader.fieldnames, SMILES_COLUMNS, "SMILES")
        target_col = _find_column(reader.fieldnames, TARGET_COLUMNS, "target")
        id_col = _optional_source_id_column(reader.fieldnames)
        source_rows = list(reader)

    row_assignments = manifest.get("rows", {})
    if len(source_rows) != manifest["dataset"]["row_count"] or len(
        row_assignments
    ) != len(source_rows):
        raise ValueError(f"Split manifest does not fully cover {csv_path}")

    observations = []
    exclusion_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    assayed_form_counts: Counter[str] = Counter()
    used_overrides: set[str] = set()
    for row_id, source_row in enumerate(source_rows):
        raw_smiles = source_row[smiles_col].strip()
        override = validated_override(overrides, dataset, row_id, raw_smiles)
        if override is not None:
            used_overrides.add(str(row_id))
        structure = standardize_structure(
            raw_smiles,
            replacement_smiles=override["replacement_smiles"] if override else None,
        )
        assignment = row_assignments.get(str(row_id))
        if assignment is None:
            raise ValueError(f"{dataset} row {row_id} is missing from split manifest")
        if assignment["parent_inchi_key"] != structure.parent_inchi_key:
            raise ValueError(
                f"{dataset} row {row_id} parent differs from split manifest"
            )
        if assignment["scaffold_key"] != structure.parent_scaffold:
            raise ValueError(
                f"{dataset} row {row_id} scaffold differs from split manifest"
            )

        target = float(source_row[target_col].strip())
        if not math.isfinite(target):
            raise ValueError(f"{dataset} row {row_id} has non-finite target")
        if endpoint["task_type"] == "classification" and target not in (0.0, 1.0):
            raise ValueError(
                f"{dataset} row {row_id} has non-binary classification target"
            )

        filename = f"{dataset}_id_{row_id}.sdf"
        parent_path = data_dir / dataset / filename
        parent_exclusion = validate_parent_sdf(parent_path, structure.parent_inchi_key)
        parent_relative = (
            _safe_relative_file(data_dir, parent_path)
            if parent_exclusion is None
            else None
        )
        assayed_path = data_dir / "assayed_forms" / dataset / filename
        assayed_relative = (
            _safe_relative_file(data_dir, assayed_path)
            if validate_assayed_sdf(
                assayed_path,
                structure.assayed_inchi_key,
                structure.assayed_component_count,
            )
            is None
            else None
        )
        if parent_exclusion is not None:
            training_eligible = False
            exclusion_reason = parent_exclusion
            exclusion_counts[exclusion_reason] += 1
        else:
            training_eligible = True
            exclusion_reason = None

        species_column = endpoint.get("species_column")
        species = (
            source_row.get(species_column)
            if species_column
            else endpoint.get("species")
        )
        if species is not None:
            species = species.strip() or None
        split = assignment["split"]
        if split not in SPLITS:
            raise ValueError(f"{dataset} row {row_id} has invalid split {split!r}")

        observation = {
            "schema_version": SCHEMA_VERSION,
            "observation_id": f"tdc:{dataset}:{row_id}",
            "dataset_name": dataset,
            "dataset_version": dataset_version,
            "source_row_id": row_id,
            "source_record_id": source_row.get(id_col) if id_col else None,
            "source_file": csv_path.name,
            "source_file_sha256": source_hash,
            "source_raw_json": json.dumps(
                source_row, ensure_ascii=False, separators=(",", ":")
            ),
            "source_collection": endpoint["source_collection"],
            "source_url": endpoint.get("source_url"),
            "source_publication": endpoint.get("source_publication"),
            "source_assay_id": endpoint.get("source_assay_id"),
            "source_license": endpoint.get("source_license"),
            "tdc_distribution_license": endpoint.get("tdc_distribution_license"),
            "quality_tier": endpoint["quality_tier"],
            "metadata_completeness": 0.0,
            "metadata_missing_fields": [],
            "provenance_json": json.dumps(
                [
                    {
                        "role": "derived_dataset_snapshot",
                        "source_collection": endpoint["source_collection"],
                        "source_file": csv_path.name,
                        "source_file_sha256": source_hash,
                        "source_row_id": row_id,
                        "source_record_id": source_row.get(id_col) if id_col else None,
                        "source_url": endpoint.get("source_url"),
                        "source_publication": endpoint.get("source_publication"),
                        "source_assay_id": endpoint.get("source_assay_id"),
                        "source_license": endpoint.get("source_license"),
                    }
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "upstream_link_status": "not_attempted",
            "upstream_link_method": None,
            "upstream_link_confidence": None,
            "upstream_record_id": None,
            "assayed_smiles": structure.assayed_smiles,
            "assayed_canonical_smiles": structure.assayed_canonical_smiles,
            "assayed_inchi": structure.assayed_inchi,
            "assayed_inchi_key": structure.assayed_inchi_key,
            "assayed_component_count": structure.assayed_component_count,
            "assayed_form_kind": structure.assayed_form_kind,
            "parent_smiles": structure.parent_smiles,
            "parent_inchi": structure.parent_inchi,
            "parent_inchi_key": structure.parent_inchi_key,
            "parent_scaffold": structure.parent_scaffold,
            "removed_components": structure.removed_components,
            "structure_standardization": STRUCTURE_STANDARDIZATION,
            "structure_override_applied": structure.structure_override_applied,
            "standardization_warnings": structure.standardization_warnings,
            "endpoint_domain": endpoint["domain"],
            "endpoint_name": endpoint["endpoint_name"],
            "readout": endpoint["readout"],
            "task_type": endpoint["task_type"],
            "value": target,
            "value_unit": endpoint.get("unit"),
            "value_qualifier": "=",
            "value_lower_bound": target,
            "value_upper_bound": target,
            "value_is_censored": False,
            "species": species,
            "strain": endpoint.get("strain"),
            "tissue": endpoint.get("tissue"),
            "cell_line": endpoint.get("cell_line"),
            "biological_matrix": endpoint.get("matrix"),
            "assay_method": endpoint.get("assay_method"),
            "assay_direction": endpoint.get("assay_direction"),
            "assay_ph": endpoint.get("ph"),
            "temperature_c": endpoint.get("temperature_c"),
            "concentration_value": endpoint.get("concentration_value"),
            "concentration_unit": endpoint.get("concentration_unit"),
            "dose_value": endpoint.get("dose_value"),
            "dose_unit": endpoint.get("dose_unit"),
            "dose_route": endpoint.get("dose_route"),
            "formulation": endpoint.get("formulation"),
            "replicate_id": None,
            "replicate_count": None,
            "measurement_stddev": None,
            "measurement_variance": None,
            "split": split,
            "split_manifest_sha256": manifest_hash,
            "parent_sdf_relative_path": parent_relative,
            "assayed_sdf_relative_path": assayed_relative,
            "training_eligible": training_eligible,
            "exclusion_reason": exclusion_reason,
            "eligibility_basis": "parent_sdf_is_single_component_and_matches_parent_inchi_key",
        }
        completeness, missing = _metadata_completeness(observation)
        observation["metadata_completeness"] = completeness
        observation["metadata_missing_fields"] = missing
        observations.append(observation)
        split_counts[split] += 1
        assayed_form_counts[structure.assayed_form_kind] += 1

    unused_overrides = set(overrides.get(dataset, {})) - used_overrides
    if unused_overrides:
        raise ValueError(
            f"{dataset} has unused structure overrides: {sorted(unused_overrides)}"
        )

    metadata = _parquet_metadata(
        dataset=dataset,
        dataset_version=dataset_version,
        source_file=csv_path.name,
        source_hash=source_hash,
        split_manifest_hash=manifest_hash,
        registry_hash=registry_hash,
        override_hash=override_hash if used_overrides else None,
        row_count=len(observations),
    )
    table = pa.Table.from_pylist(observations, schema=canonical_schema(metadata))
    output_path = output_dir / f"{dataset}.observations.parquet"
    _write_parquet_atomic(output_path, table)
    validate_dataset(
        output_path, data_dir, dataset, len(observations), source_hash, manifest_hash
    )

    entry = {
        "dataset_version": dataset_version,
        "source_file": csv_path.name,
        "source_file_sha256": source_hash,
        "split_manifest": manifest_path.relative_to(data_dir).as_posix(),
        "split_manifest_sha256": manifest_hash,
        "parquet_file": output_path.relative_to(data_dir).as_posix(),
        "parquet_file_sha256": sha256_file(output_path),
        "row_count": len(observations),
        "training_eligible_count": sum(
            row["training_eligible"] for row in observations
        ),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "assayed_form_counts": dict(sorted(assayed_form_counts.items())),
        "mean_metadata_completeness": sum(
            row["metadata_completeness"] for row in observations
        )
        / len(observations),
    }
    print(
        f"{dataset}: rows={entry['row_count']} eligible={entry['training_eligible_count']} "
        f"excluded={entry['row_count'] - entry['training_eligible_count']}"
    )
    return output_path, entry


def _decoded_metadata(schema: pa.Schema) -> dict[str, str]:
    return {
        key.decode(): value.decode() for key, value in (schema.metadata or {}).items()
    }


def validate_dataset(
    path: Path,
    data_dir: Path,
    dataset: str,
    row_count: int,
    source_hash: str,
    manifest_hash: str,
) -> None:
    schema = pq.read_schema(path)
    expected_schema = canonical_schema()
    if not schema.remove_metadata().equals(expected_schema):
        raise ValueError(f"Canonical schema mismatch in {path}")
    metadata = _decoded_metadata(schema)
    required_metadata = {
        "adme_schema_version": str(SCHEMA_VERSION),
        "dataset_name": dataset,
        "source_file_sha256": source_hash,
        "split_manifest_sha256": manifest_hash,
        "row_count": str(row_count),
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{path}: metadata {key!r} does not equal {expected!r}")

    columns = [
        "observation_id",
        "source_row_id",
        "parent_inchi_key",
        "parent_scaffold",
        "value",
        "split",
        "training_eligible",
        "exclusion_reason",
        "parent_sdf_relative_path",
    ]
    table = pq.read_table(path, columns=columns)
    if table.num_rows != row_count:
        raise ValueError(f"{path}: row count mismatch")
    data = table.to_pydict()
    if data["source_row_id"] != list(range(row_count)):
        raise ValueError(f"{path}: source row IDs are not complete and ordered")
    if len(set(data["observation_id"])) != row_count:
        raise ValueError(f"{path}: duplicate observation IDs")

    parent_splits: dict[str, str] = {}
    scaffold_splits: dict[str, str] = {}
    for row_id in range(row_count):
        split = data["split"][row_id]
        if split not in SPLITS or not math.isfinite(data["value"][row_id]):
            raise ValueError(f"{path}: invalid target or split at row {row_id}")
        for key, seen in (
            (data["parent_inchi_key"][row_id], parent_splits),
            (data["parent_scaffold"][row_id], scaffold_splits),
        ):
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError(f"{path}: molecule/scaffold leakage at row {row_id}")
        eligible = data["training_eligible"][row_id]
        relative = data["parent_sdf_relative_path"][row_id]
        reason = data["exclusion_reason"][row_id]
        if eligible:
            if reason is not None or relative is None:
                raise ValueError(f"{path}: invalid eligible row {row_id}")
            resolved = (data_dir / relative).resolve()
            try:
                resolved.relative_to(data_dir.resolve())
            except ValueError as error:
                raise ValueError(f"{path}: structure path escapes data root") from error
            if not resolved.is_file() or resolved.stat().st_size == 0:
                raise ValueError(f"{path}: eligible row {row_id} has no parent SDF")
        elif not reason:
            raise ValueError(f"{path}: excluded row {row_id} has no reason")


def reuse_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    dataset_version: str,
    registry_hash: str,
    overrides: dict,
    override_hash: str | None,
) -> tuple[Path, dict] | None:
    """Reuse an already-current snapshot and reconstruct its catalog entry."""
    path = output_dir / f"{dataset}.observations.parquet"
    if not path.is_file():
        return None
    csv_path = data_dir / f"{dataset}.csv"
    source_hash = sha256_file(csv_path)
    _, manifest_path, manifest_hash = _load_split_manifest(
        data_dir, dataset, source_hash
    )
    expected_override_hash = override_hash if overrides.get(dataset) else None
    metadata = _decoded_metadata(pq.read_schema(path))
    required = {
        "adme_schema_version": str(SCHEMA_VERSION),
        "generator_version": GENERATOR_VERSION,
        "dataset_name": dataset,
        "dataset_version": dataset_version,
        "source_file": csv_path.name,
        "source_file_sha256": source_hash,
        "split_manifest_sha256": manifest_hash,
        "endpoint_registry_sha256": registry_hash,
        "structure_overrides_sha256": expected_override_hash or "none",
        "structure_standardization": STRUCTURE_STANDARDIZATION,
        "rdkit_version": RDKit_VERSION,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        return None
    row_count = int(metadata.get("row_count", "-1"))
    validate_dataset(path, data_dir, dataset, row_count, source_hash, manifest_hash)
    table = pq.read_table(
        path,
        columns=[
            "training_eligible",
            "exclusion_reason",
            "split",
            "assayed_form_kind",
            "metadata_completeness",
        ],
    ).to_pydict()
    exclusion_counts = Counter(
        reason
        for eligible, reason in zip(
            table["training_eligible"], table["exclusion_reason"], strict=True
        )
        if not eligible
    )
    split_counts = Counter(table["split"])
    form_counts = Counter(table["assayed_form_kind"])
    entry = {
        "dataset_version": dataset_version,
        "source_file": csv_path.name,
        "source_file_sha256": source_hash,
        "split_manifest": manifest_path.relative_to(data_dir).as_posix(),
        "split_manifest_sha256": manifest_hash,
        "parquet_file": path.relative_to(data_dir).as_posix(),
        "parquet_file_sha256": sha256_file(path),
        "row_count": row_count,
        "training_eligible_count": sum(table["training_eligible"]),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "assayed_form_counts": dict(sorted(form_counts.items())),
        "mean_metadata_completeness": sum(table["metadata_completeness"]) / row_count,
    }
    print(f"{dataset}: reusing {row_count} validated canonical rows")
    return path, entry


def _write_combined(
    output_dir: Path,
    dataset_paths: list[Path],
    dataset_version: str,
    registry_hash: str,
    override_hash: str | None,
) -> Path:
    path = output_dir / "all_observations.parquet"
    metadata = {
        b"adme_schema_version": str(SCHEMA_VERSION).encode(),
        b"generator_version": GENERATOR_VERSION.encode(),
        b"dataset_version": dataset_version.encode(),
        b"endpoint_registry_sha256": registry_hash.encode(),
        b"structure_overrides_sha256": (override_hash or "none").encode(),
        b"structure_standardization": STRUCTURE_STANDARDIZATION.encode(),
        b"rdkit_version": RDKit_VERSION.encode(),
        b"dataset_count": str(len(dataset_paths)).encode(),
    }
    schema = canonical_schema(metadata)
    with tempfile.NamedTemporaryFile(
        dir=output_dir, suffix=".parquet", delete=False
    ) as temp:
        temp_path = Path(temp.name)
    try:
        with pq.ParquetWriter(
            temp_path,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
        ) as writer:
            for dataset_path in dataset_paths:
                table = pq.read_table(dataset_path).replace_schema_metadata(metadata)
                writer.write_table(table)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reuse-current",
        action="store_true",
        help="Reuse snapshots whose complete source/version/hash metadata is current",
    )
    parser.add_argument(
        "--endpoint-registry",
        type=Path,
        default=Path(__file__).with_name("therapeutic_endpoint_registry.json"),
    )
    parser.add_argument(
        "--structure-overrides",
        type=Path,
        default=Path(__file__).with_name("therapeutic_structure_overrides.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / CANONICAL_SUBDIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dataset_version.strip():
        raise SystemExit("Dataset version must not be blank")

    registry, registry_hash = _load_registry(args.endpoint_registry)
    overrides, override_hash = load_structure_overrides(args.structure_overrides)
    known = list(registry["datasets"])
    datasets = args.dataset or known
    unknown = sorted(set(datasets) - set(known))
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}")

    entries = {}
    dataset_paths = []
    for dataset in datasets:
        reused = (
            reuse_dataset(
                data_dir=data_dir,
                output_dir=output_dir,
                dataset=dataset,
                dataset_version=args.dataset_version,
                registry_hash=registry_hash,
                overrides=overrides,
                override_hash=override_hash,
            )
            if args.reuse_current
            else None
        )
        if reused is None:
            path, entry = build_dataset(
                data_dir=data_dir,
                output_dir=output_dir,
                dataset=dataset,
                dataset_version=args.dataset_version,
                registry=registry,
                registry_hash=registry_hash,
                overrides=overrides,
                override_hash=override_hash,
            )
        else:
            path, entry = reused
        dataset_paths.append(path)
        entries[dataset] = entry

    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_version": args.dataset_version,
        "endpoint_registry": args.endpoint_registry.name,
        "endpoint_registry_sha256": registry_hash,
        "structure_overrides": args.structure_overrides.name,
        "structure_overrides_sha256": override_hash,
        "structure_standardization": STRUCTURE_STANDARDIZATION,
        "rdkit_version": RDKit_VERSION,
        "datasets": entries,
    }
    if set(datasets) == set(known):
        combined_path = _write_combined(
            output_dir,
            dataset_paths,
            args.dataset_version,
            registry_hash,
            override_hash,
        )
        catalog["combined_file"] = combined_path.relative_to(data_dir).as_posix()
        catalog["combined_file_sha256"] = sha256_file(combined_path)
        catalog["total_row_count"] = sum(
            entry["row_count"] for entry in entries.values()
        )
        catalog["training_eligible_count"] = sum(
            entry["training_eligible_count"] for entry in entries.values()
        )
    _write_json_atomic(output_dir / "catalog.json", catalog)
    print(f"Wrote {len(entries)} canonical Parquet snapshots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
