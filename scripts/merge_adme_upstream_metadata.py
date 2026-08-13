"""Merge observation-level upstream metadata into canonical ADME Parquet files.

The input must be CSV or Parquet. Fields intended for merging use canonical
column names; additional source-native columns are retained in provenance.
Matching is strict and auditable: ``observation_id`` is preferred, followed by
``dataset_name`` + ``source_row_id``, then a unique ``source_record_id``.
Existing non-null values are never overwritten; disagreements are written to a
conflict report for manual review.

This tool always writes a new output directory. It never edits a canonical
snapshot in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adme_data_model import sha256_file
from build_adme_parquet import CONTEXT_FIELDS, canonical_schema

MATCH_KEYS = {"observation_id", "dataset_name", "source_row_id", "source_record_id"}
NON_MERGE_FIELDS = MATCH_KEYS | {
    "schema_version",
    "dataset_version",
    "source_file",
    "source_file_sha256",
    "source_raw_json",
    "split",
    "split_manifest_sha256",
    "parent_sdf_relative_path",
    "assayed_sdf_relative_path",
    "training_eligible",
    "exclusion_reason",
    "eligibility_basis",
    "provenance_json",
    "upstream_link_status",
    "upstream_link_method",
    "upstream_link_confidence",
    "upstream_record_id",
}


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            return list(csv.DictReader(source, delimiter=delimiter))
    raise ValueError(f"Unsupported upstream file type: {path}")


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _coerce(value, data_type: pa.DataType):
    if _blank(value):
        return None
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return str(value).strip()
    if pa.types.is_boolean(data_type):
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
        raise ValueError(f"Cannot parse boolean {value!r}")
    if pa.types.is_integer(data_type):
        return int(value)
    if pa.types.is_floating(data_type):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite numeric value {value!r}")
        return result
    if pa.types.is_list(data_type):
        if isinstance(value, list):
            return value
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON list, got {value!r}")
        return parsed
    raise ValueError(f"Unsupported canonical type {data_type}")


def _equal(left, right) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def _indexes(rows: list[dict]) -> tuple[dict, dict, dict]:
    by_observation = {row["observation_id"]: index for index, row in enumerate(rows)}
    by_source_row = {
        (row["dataset_name"], row["source_row_id"]): index
        for index, row in enumerate(rows)
    }
    source_record_candidates: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["source_record_id"] is not None:
            source_record_candidates[
                (row["dataset_name"], str(row["source_record_id"]))
            ].append(index)
    unique_source_records = {
        key: indices[0]
        for key, indices in source_record_candidates.items()
        if len(indices) == 1
    }
    return by_observation, by_source_row, unique_source_records


def _match(
    source: dict,
    default_dataset: str | None,
    by_observation: dict,
    by_source_row: dict,
    by_source_record: dict,
) -> tuple[int | None, str | None, float | None]:
    observation_id = source.get("observation_id")
    if not _blank(observation_id):
        return by_observation.get(str(observation_id).strip()), "observation_id", 1.0
    dataset = source.get("dataset_name") or default_dataset
    if _blank(dataset):
        return None, None, None
    dataset = str(dataset).strip()
    source_row_id = source.get("source_row_id")
    if not _blank(source_row_id):
        return (
            by_source_row.get((dataset, int(source_row_id))),
            "dataset_source_row_id",
            1.0,
        )
    source_record_id = source.get("source_record_id")
    if not _blank(source_record_id):
        return (
            by_source_record.get((dataset, str(source_record_id).strip())),
            "unique_dataset_source_record_id",
            0.95,
        )
    return None, None, None


def _completeness(row: dict) -> tuple[float, list[str]]:
    missing = [field for field in CONTEXT_FIELDS if row.get(field) is None]
    return (len(CONTEXT_FIELDS) - len(missing)) / len(CONTEXT_FIELDS), missing


def _write_table(path: Path, table: pa.Table) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument(
        "--dataset", help="Default dataset_name if absent from source rows"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_dir = args.canonical_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == canonical_dir:
        raise SystemExit(
            "Refusing to enrich canonical data in place; choose a new --output-dir"
        )
    source_path = args.source.resolve()
    source_hash = sha256_file(source_path)
    upstream = _read_records(source_path)
    if not upstream:
        raise SystemExit("Upstream source contains no rows")

    schema = canonical_schema()
    fields = {field.name: field.type for field in schema}
    merge_fields = [
        name for name in upstream[0] if name in fields and name not in NON_MERGE_FIELDS
    ]
    if not merge_fields:
        raise SystemExit("Upstream source has no mergeable canonical metadata columns")

    dataset_files = sorted(canonical_dir.glob("*.observations.parquet"))
    dataset_rows: dict[str, list[dict]] = {}
    dataset_metadata: dict[str, dict[bytes, bytes] | None] = {}
    global_by_observation = {}
    global_by_source_row = {}
    global_by_source_record = {}
    for path in dataset_files:
        table = pq.read_table(path)
        rows = table.to_pylist()
        if not rows:
            continue
        dataset = rows[0]["dataset_name"]
        dataset_rows[dataset] = rows
        dataset_metadata[dataset] = table.schema.metadata
        by_observation, by_source_row, by_source_record = _indexes(rows)
        global_by_observation.update(
            {key: (dataset, index) for key, index in by_observation.items()}
        )
        global_by_source_row.update(
            {key: (dataset, index) for key, index in by_source_row.items()}
        )
        global_by_source_record.update(
            {key: (dataset, index) for key, index in by_source_record.items()}
        )

    conflicts = []
    unmatched = []
    linked = Counter()
    for upstream_index, source in enumerate(upstream):
        matched, method, confidence = _match(
            source,
            args.dataset,
            global_by_observation,
            global_by_source_row,
            global_by_source_record,
        )
        if matched is None:
            unmatched.append({"upstream_row": upstream_index, "record": source})
            continue
        dataset, row_index = matched
        row = dataset_rows[dataset][row_index]
        row_conflicts = []
        populated = []
        confirmed = []
        for field in merge_fields:
            incoming = _coerce(source.get(field), fields[field])
            if incoming is None:
                continue
            current = row.get(field)
            if current is None:
                row[field] = incoming
                populated.append(field)
            elif _equal(current, incoming):
                confirmed.append(field)
            else:
                conflict = {
                    "upstream_row": upstream_index,
                    "observation_id": row["observation_id"],
                    "field": field,
                    "canonical_value": current,
                    "upstream_value": incoming,
                }
                conflicts.append(conflict)
                row_conflicts.append(conflict)

        provenance = json.loads(row["provenance_json"])
        provenance.append(
            {
                "role": "upstream_observation_metadata",
                "source_name": args.source_name,
                "source_version": args.source_version,
                "source_file": source_path.name,
                "source_file_sha256": source_hash,
                "source_license": args.source_license,
                "upstream_row": upstream_index,
                "upstream_record": source,
                "link_method": method,
                "link_confidence": confidence,
                "populated_fields": populated,
                "confirmed_fields": confirmed,
                "conflict_count": len(row_conflicts),
            }
        )
        row["provenance_json"] = json.dumps(
            provenance,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        row["upstream_link_status"] = (
            "linked_with_conflicts" if row_conflicts else "linked"
        )
        row["upstream_link_method"] = method
        row["upstream_link_confidence"] = confidence
        row["upstream_record_id"] = (
            None
            if _blank(source.get("source_record_id"))
            else str(source["source_record_id"])
        )
        row["metadata_completeness"], row["metadata_missing_fields"] = _completeness(
            row
        )
        linked[dataset] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for dataset, rows in dataset_rows.items():
        metadata = dict(dataset_metadata[dataset] or {})
        metadata.update(
            {
                b"enrichment_source_sha256": source_hash.encode(),
                b"enrichment_source_name": args.source_name.encode(),
                b"enrichment_source_version": args.source_version.encode(),
            }
        )
        table = pa.Table.from_pylist(rows, schema=canonical_schema(metadata))
        output_path = output_dir / f"{dataset}.observations.parquet"
        _write_table(output_path, table)
        output_files[dataset] = {
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "row_count": len(rows),
            "linked_rows": linked[dataset],
        }

    report = {
        "schema_version": 1,
        "source_name": args.source_name,
        "source_version": args.source_version,
        "source_file": str(source_path),
        "source_file_sha256": source_hash,
        "source_license": args.source_license,
        "source_row_count": len(upstream),
        "linked_row_count": sum(linked.values()),
        "unmatched_row_count": len(unmatched),
        "conflict_count": len(conflicts),
        "datasets": output_files,
        "unmatched": unmatched,
        "conflicts": conflicts,
    }
    report_path = output_dir / "enrichment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Linked {report['linked_row_count']}/{len(upstream)} upstream rows; "
        f"unmatched={len(unmatched)} conflicts={len(conflicts)}"
    )
    print(f"Wrote enriched snapshots and report to {output_dir}")
    return 0 if not unmatched and not conflicts else 2


if __name__ == "__main__":
    raise SystemExit(main())
