"""Generate deterministic, versioned therapeutic dataset split manifests.

Unlike the old PyTDC helper, this script never prints arrays for manual copy/paste.
It reads the exact local CSV snapshots used by training and writes one manifest per
dataset to::

    <data-dir>/split_manifests/<dataset>.split.json

Every source row is keyed in the manifest by its zero-based CSV row ID and its
RDKit-standardized parent InChIKey. Molecules are grouped by Bemis-Murcko scaffold
(and duplicate acyclic parents are grouped together) before a deterministic
70/10/20 assignment. The manifest records the source file SHA-256, an explicit
dataset version, RDKit version, class distributions/target summaries, and all
row assignments.

Example::

    python scripts/train_test_split.py \
        --data-dir C:/Users/the_a/Desktop/tdc_data \
        --dataset-version tdc-local-2026-01-v1

Training independently revalidates the file hash, full coverage, split overlap,
molecule leakage, scaffold leakage, and expected label distributions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from adme_data_model import (
    RDKit_VERSION,
    load_structure_overrides,
    sha256_bytes,
    sha256_file,
    standardize_parent,
    validated_override,
)

SCHEMA_VERSION = 1
ALGORITHM = "deterministic_scaffold_group"
ALGORITHM_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_FRACTIONS = {"train": 0.7, "validation": 0.1, "test": 0.2}
SPLITS = ("train", "validation", "test")

# Explicit task types keep a regression dataset with two coincidental values from
# silently becoming a classifier, and vice versa.
DATASET_TASKS = {
    "ames": "classification",
    "bbb_martins": "classification",
    "bioavailability_ma": "classification",
    "caco2_wang": "regression",
    "carcinogens_lagunin": "classification",
    "clearance_hepatocyte_az": "regression",
    "cyp1a2_veith": "classification",
    "cyp2c19_veith": "classification",
    "cyp2c9_veith": "classification",
    "cyp2d6_veith": "classification",
    "cyp3a4_veith": "classification",
    "dili": "classification",
    "half_life_obach": "regression",
    "herg": "classification",
    "hia_hou": "classification",
    "hydrationfreeenergy_freesolv": "regression",
    "ld50_zhu": "regression",
    "lipophilicity_astrazeneca": "regression",
    "pampa_ncats": "classification",
    "pgp_broccatelli": "classification",
    "ppbr_az": "regression",
    "skin_reaction": "classification",
    "solubility_aqsoldb": "regression",
    "vdss_lombardo": "regression",
}

SMILES_COLUMN_NAMES = ("drug", "x", "smiles")
TARGET_COLUMN_NAMES = ("y", "label", "target")


@dataclass(frozen=True)
class SourceRow:
    row_id: int
    smiles: str
    target: float
    parent_inchi_key: str
    scaffold_key: str


def find_column(header: list[str], candidates: Iterable[str], kind: str) -> int:
    by_lower = {name.strip().lower(): i for i, name in enumerate(header)}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(
        f"No {kind} column in header {header!r}; expected one of {tuple(candidates)!r}"
    )


def normalize_binary_target(value: float, dataset: str, row_id: int) -> str:
    if abs(value) <= 1e-12:
        return "0"
    if abs(value - 1.0) <= 1e-12:
        return "1"
    raise ValueError(
        f"{dataset} row {row_id} has non-binary target {value!r}, but the task is classification"
    )


def read_dataset(
    path: Path,
    dataset: str,
    task_type: str,
    overrides: dict,
) -> tuple[list[SourceRow], str, str, bool]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Dataset is empty: {path}")
        smiles_col = find_column(header, SMILES_COLUMN_NAMES, "SMILES")
        target_col = find_column(header, TARGET_COLUMN_NAMES, "target")
        rows: list[SourceRow] = []
        used_override = False
        dataset_overrides = overrides.get(dataset, {})
        used_override_ids: set[str] = set()

        for row_id, record in enumerate(reader):
            if len(record) <= max(smiles_col, target_col):
                raise ValueError(f"{dataset} row {row_id} is shorter than its header")
            raw_smiles = record[smiles_col].strip()
            override = validated_override(overrides, dataset, row_id, raw_smiles)
            if override:
                used_override_ids.add(str(row_id))
            smiles = override["replacement_smiles"] if override else raw_smiles
            used_override = used_override or override is not None
            try:
                target = float(record[target_col].strip())
            except ValueError as error:
                raise ValueError(
                    f"{dataset} row {row_id} has a non-numeric target"
                ) from error
            if not math.isfinite(target):
                raise ValueError(f"{dataset} row {row_id} has a non-finite target")
            if task_type == "classification":
                normalize_binary_target(target, dataset, row_id)
            try:
                parent_inchi_key, scaffold_key = standardize_parent(smiles)
            except Exception as error:
                raise ValueError(
                    f"{dataset} row {row_id} cannot be standardized: {raw_smiles!r}. "
                    "Add a documented correction to therapeutic_structure_overrides.json."
                ) from error
            rows.append(
                SourceRow(
                    row_id=row_id,
                    smiles=raw_smiles,
                    target=target,
                    parent_inchi_key=parent_inchi_key,
                    scaffold_key=scaffold_key,
                )
            )

        unused_override_ids = set(dataset_overrides) - used_override_ids
        if unused_override_ids:
            raise ValueError(
                f"{dataset} has unused structure override row IDs: "
                f"{sorted(unused_override_ids, key=int)}"
            )

    if not rows:
        raise ValueError(f"Dataset has no records: {path}")
    return rows, header[smiles_col], header[target_col], used_override


def stable_rank(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()


def assign_groups(
    rows: list[SourceRow],
    task_type: str,
    fractions: dict[str, float],
    seed: int,
) -> dict[int, str]:
    # A molecule can occasionally appear in multiple tautomer/aromatic forms that
    # standardize to one parent InChIKey but yield different written scaffolds (or
    # vice versa). Build connected components over both identities so neither kind
    # of relationship can cross a split.
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    parent_owner: dict[str, int] = {}
    scaffold_owner: dict[str, int] = {}
    for row in rows:
        row_index = row.row_id
        if row.parent_inchi_key in parent_owner:
            union(row_index, parent_owner[row.parent_inchi_key])
        else:
            parent_owner[row.parent_inchi_key] = row_index
        if row.scaffold_key in scaffold_owner:
            union(row_index, scaffold_owner[row.scaffold_key])
        else:
            scaffold_owner[row.scaffold_key] = row_index

    component_rows: dict[int, list[SourceRow]] = defaultdict(list)
    for row in rows:
        component_rows[find(row.row_id)].append(row)
    groups = []
    for group_rows in component_rows.values():
        identities = sorted(
            {f"P:{row.parent_inchi_key}" for row in group_rows}
            | {f"S:{row.scaffold_key}" for row in group_rows}
        )
        component_key = sha256_bytes("\n".join(identities).encode("utf-8"))
        groups.append((component_key, group_rows))

    target_rows = {split: len(rows) * fractions[split] for split in SPLITS}
    assigned_rows = Counter()
    class_totals = Counter()
    assigned_classes: dict[str, Counter] = {split: Counter() for split in SPLITS}
    if task_type == "classification":
        class_totals.update(
            normalize_binary_target(row.target, "dataset", row.row_id) for row in rows
        )

    # Large scaffold families are placed first so a late oversized family cannot
    # unexpectedly consume an entire validation or test split. Hash-based ties make
    # the result independent of CSV/group insertion order and Python hash randomization.
    ordered_groups = sorted(
        groups,
        key=lambda item: (-len(item[1]), stable_rank(seed, item[0])),
    )
    assignments: dict[int, str] = {}

    for _, group_rows in ordered_groups:
        group_size = len(group_rows)
        group_classes = Counter()
        if task_type == "classification":
            group_classes.update(
                normalize_binary_target(row.target, "dataset", row.row_id)
                for row in group_rows
            )

        def candidate_score(split: str) -> tuple[float, float, int]:
            row_fill = (assigned_rows[split] + group_size) / max(
                target_rows[split], 1.0
            )
            fills = [row_fill]
            if task_type == "classification":
                for class_name in ("0", "1"):
                    target_class = class_totals[class_name] * fractions[split]
                    fills.append(
                        (
                            assigned_classes[split][class_name]
                            + group_classes[class_name]
                        )
                        / max(target_class, 1.0)
                    )
            # Minimize the worst capacity fill, then average fill. This yields a
            # deterministic stratified group split without ever separating a scaffold.
            return max(fills), sum(fills) / len(fills), SPLITS.index(split)

        chosen = min(SPLITS, key=candidate_score)
        assigned_rows[chosen] += group_size
        assigned_classes[chosen].update(group_classes)
        for row in group_rows:
            assignments[row.row_id] = chosen

    return assignments


def summarize(
    rows: list[SourceRow], assignments: dict[int, str], task_type: str
) -> dict:
    split_counts = {split: 0 for split in SPLITS}
    values: dict[str, list[float]] = {split: [] for split in SPLITS}
    distributions: dict[str, Counter] = {
        split: Counter({"0": 0, "1": 0}) for split in SPLITS
    }
    for row in rows:
        split = assignments[row.row_id]
        split_counts[split] += 1
        values[split].append(row.target)
        if task_type == "classification":
            distributions[split][
                normalize_binary_target(row.target, "dataset", row.row_id)
            ] += 1

    expected = {"split_counts": split_counts}
    if task_type == "classification":
        expected["class_distribution"] = {
            split: {"0": distributions[split]["0"], "1": distributions[split]["1"]}
            for split in SPLITS
        }
    else:
        expected["target_summary"] = {
            split: {
                "count": len(values[split]),
                "min": min(values[split]),
                "max": max(values[split]),
                "mean": sum(values[split]) / len(values[split]),
            }
            for split in SPLITS
        }
    return expected


def assert_invariants(
    dataset: str,
    rows: list[SourceRow],
    assignments: dict[int, str],
    expected: dict,
    task_type: str,
) -> None:
    expected_ids = set(range(len(rows)))
    if set(assignments) != expected_ids:
        missing = sorted(expected_ids - set(assignments))
        extra = sorted(set(assignments) - expected_ids)
        raise AssertionError(
            f"{dataset}: incomplete row coverage; missing={missing}, extra={extra}"
        )

    parent_splits: dict[str, str] = {}
    scaffold_splits: dict[str, str] = {}
    for row in rows:
        split = assignments[row.row_id]
        previous = parent_splits.setdefault(row.parent_inchi_key, split)
        if previous != split:
            raise AssertionError(
                f"{dataset}: parent molecule leakage for {row.parent_inchi_key}"
            )
        previous = scaffold_splits.setdefault(row.scaffold_key, split)
        if previous != split:
            raise AssertionError(f"{dataset}: scaffold leakage for {row.scaffold_key}")

    if sum(expected["split_counts"].values()) != len(rows):
        raise AssertionError(f"{dataset}: split counts do not cover every row")
    if any(expected["split_counts"][split] == 0 for split in SPLITS):
        raise AssertionError(f"{dataset}: an empty split was generated")
    if task_type == "classification":
        for split in SPLITS:
            distribution = expected["class_distribution"][split]
            if distribution["0"] == 0 or distribution["1"] == 0:
                raise AssertionError(
                    f"{dataset}: {split} does not contain both expected classes: {distribution}"
                )


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(encoded)
        temp_path = Path(tmp.name)
    os.replace(temp_path, path)


def generate_manifest(
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    dataset_version: str,
    task_type: str,
    fractions: dict[str, float],
    seed: int,
    overrides: dict,
    override_hash: str | None,
) -> Path:
    csv_path = data_dir / f"{dataset}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset snapshot: {csv_path}")

    rows, smiles_column, target_column, used_override = read_dataset(
        csv_path, dataset, task_type, overrides
    )
    assignments = assign_groups(rows, task_type, fractions, seed)
    expected = summarize(rows, assignments, task_type)
    assert_invariants(dataset, rows, assignments, expected, task_type)

    manifest_rows = {
        str(row.row_id): {
            "parent_inchi_key": row.parent_inchi_key,
            "scaffold_key": row.scaffold_key,
            "split": assignments[row.row_id],
        }
        for row in rows
    }
    dataset_info = {
        "name": dataset,
        "version": dataset_version,
        "source_file": csv_path.name,
        "source_file_sha256": sha256_file(csv_path),
        "row_count": len(rows),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "rdkit_version": RDKit_VERSION,
    }
    if used_override:
        dataset_info["structure_overrides_sha256"] = override_hash

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_info,
        "split": {
            "algorithm": ALGORITHM,
            "algorithm_version": ALGORITHM_VERSION,
            "seed": seed,
            "fractions": fractions,
        },
        "expected": expected,
        "rows": manifest_rows,
    }
    output_path = output_dir / f"{dataset}.split.json"
    write_json_atomic(output_path, manifest)

    counts = expected["split_counts"]
    detail = ""
    if task_type == "classification":
        detail = " classes=" + json.dumps(
            expected["class_distribution"], separators=(",", ":")
        )
    print(
        f"{dataset}: rows={len(rows)} train={counts['train']} "
        f"validation={counts['validation']} test={counts['test']}{detail}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing <dataset>.csv snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Manifest folder (default: <data-dir>/split_manifests)",
    )
    parser.add_argument(
        "--dataset-version",
        required=True,
        help="Explicit immutable version label for this snapshot set",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_TASKS),
        help="Generate only this dataset; repeat as needed (default: every supported dataset)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-fraction", type=float, default=DEFAULT_FRACTIONS["train"]
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_FRACTIONS["validation"]
    )
    parser.add_argument(
        "--test-fraction", type=float, default=DEFAULT_FRACTIONS["test"]
    )
    parser.add_argument(
        "--structure-overrides",
        type=Path,
        default=Path(__file__).with_name("therapeutic_structure_overrides.json"),
        help="Documented corrections used only to derive parent structure keys",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "split_manifests").resolve()
    fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    if (
        any(value <= 0 for value in fractions.values())
        or abs(sum(fractions.values()) - 1.0) > 1e-12
    ):
        raise SystemExit(
            f"Split fractions must be positive and sum to 1.0: {fractions}"
        )
    if args.seed <= 0:
        raise SystemExit("Seed must be a positive integer")
    if not args.dataset_version.strip():
        raise SystemExit("Dataset version must not be blank")

    overrides, override_hash = load_structure_overrides(args.structure_overrides)
    datasets = args.dataset or list(DATASET_TASKS)
    for dataset in datasets:
        generate_manifest(
            data_dir=data_dir,
            output_dir=output_dir,
            dataset=dataset,
            dataset_version=args.dataset_version,
            task_type=DATASET_TASKS[dataset],
            fractions=fractions,
            seed=args.seed,
            overrides=overrides,
            override_hash=override_hash,
        )
    print(f"Wrote {len(datasets)} verified manifests to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
