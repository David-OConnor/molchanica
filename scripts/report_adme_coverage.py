"""Audit canonical ADME snapshots and emit metadata and structure reports."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adme_data_model import sha256_file
from build_adme_parquet import CONTEXT_FIELDS

AUDIT_COLUMNS = [
    "observation_id",
    "dataset_name",
    "source_row_id",
    "source_record_id",
    "source_file",
    "source_file_sha256",
    "assayed_smiles",
    "parent_smiles",
    "parent_inchi_key",
    "assayed_form_kind",
    "metadata_missing_fields",
    "upstream_link_status",
    "split",
    "training_eligible",
    "exclusion_reason",
]

REPAIR_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.string(), nullable=False),
        pa.field("dataset_name", pa.string(), nullable=False),
        pa.field("source_row_id", pa.int64(), nullable=False),
        pa.field("source_record_id", pa.string()),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_file_sha256", pa.string(), nullable=False),
        pa.field("assayed_smiles", pa.string(), nullable=False),
        pa.field("parent_smiles", pa.string(), nullable=False),
        pa.field("parent_inchi_key", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("exclusion_reason", pa.string(), nullable=False),
        pa.field("expected_parent_sdf_relative_path", pa.string(), nullable=False),
    ]
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(encoded)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
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


def _counter_dict(counter: Counter) -> dict:
    return dict(sorted(counter.items()))


def _summarize(rows: list[dict]) -> dict:
    exclusions = Counter(
        row["exclusion_reason"] for row in rows if not row["training_eligible"]
    )
    missing = Counter(field for row in rows for field in row["metadata_missing_fields"])
    unknown_context_fields = set(missing) - set(CONTEXT_FIELDS)
    if unknown_context_fields:
        raise ValueError(
            f"Unknown metadata fields in snapshots: {sorted(unknown_context_fields)}"
        )
    eligible = sum(row["training_eligible"] for row in rows)
    return {
        "row_count": len(rows),
        "training_eligible_count": eligible,
        "training_excluded_count": len(rows) - eligible,
        "exclusion_counts": _counter_dict(exclusions),
        "assayed_form_counts": _counter_dict(
            Counter(row["assayed_form_kind"] for row in rows)
        ),
        "upstream_link_status_counts": _counter_dict(
            Counter(row["upstream_link_status"] for row in rows)
        ),
        "missing_context_field_counts": {
            field: missing[field] for field in CONTEXT_FIELDS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    canonical_dir = parse_args().canonical_dir.resolve()
    paths = sorted(canonical_dir.glob("*.observations.parquet"))
    if not paths:
        raise SystemExit(f"No per-dataset canonical snapshots under {canonical_dir}")

    all_rows: list[dict] = []
    datasets = {}
    for path in paths:
        rows = pq.read_table(path, columns=AUDIT_COLUMNS).to_pylist()
        if not rows:
            raise ValueError(f"Empty canonical snapshot: {path}")
        dataset_names = {row["dataset_name"] for row in rows}
        if len(dataset_names) != 1:
            raise ValueError(f"Mixed dataset names in {path}: {sorted(dataset_names)}")
        dataset = dataset_names.pop()
        datasets[dataset] = _summarize(rows)
        all_rows.extend(rows)

    repair_rows = [
        {
            "observation_id": row["observation_id"],
            "dataset_name": row["dataset_name"],
            "source_row_id": row["source_row_id"],
            "source_record_id": row["source_record_id"],
            "source_file": row["source_file"],
            "source_file_sha256": row["source_file_sha256"],
            "assayed_smiles": row["assayed_smiles"],
            "parent_smiles": row["parent_smiles"],
            "parent_inchi_key": row["parent_inchi_key"],
            "split": row["split"],
            "exclusion_reason": row["exclusion_reason"],
            "expected_parent_sdf_relative_path": (
                f"{row['dataset_name']}/{row['dataset_name']}_id_{row['source_row_id']}.sdf"
            ),
        }
        for row in all_rows
        if not row["training_eligible"]
    ]
    repair_path = canonical_dir / "structure_repair_queue.parquet"
    _write_parquet_atomic(
        repair_path,
        pa.Table.from_pylist(repair_rows, schema=REPAIR_SCHEMA),
    )

    report = {
        "schema_version": 1,
        "canonical_directory": str(canonical_dir),
        "dataset_count": len(datasets),
        "summary": _summarize(all_rows),
        "datasets": dict(sorted(datasets.items())),
        "structure_repair_queue": repair_path.name,
        "structure_repair_queue_sha256": sha256_file(repair_path),
    }
    report_path = canonical_dir / "coverage_report.json"
    _write_json_atomic(report_path, report)
    print(
        f"Audited {report['summary']['row_count']} rows across {len(datasets)} datasets; "
        f"eligible={report['summary']['training_eligible_count']} "
        f"repair_queue={len(repair_rows)}"
    )
    print(f"Wrote {report_path} and {repair_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
