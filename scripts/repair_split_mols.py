"""
TEMPORARY repair tool — delete once the already-downloaded SDFs have been fixed.

Some TDC / AqSolDB rows carry multi-component SMILES (a `.`-separated string of salts,
counterions, or genuine mixtures such as the six dimethylphenol isomers). The original
downloader handed those straight to PubChem, which returned a single multi-component SDF.
`download_mols_for_dataset.clean_smiles` now reduces such SMILES to a single organic parent
(or skips them), but that only affects *future* downloads.

This script repairs the *existing* download folders in place. Like `download_all.py`, point
it at the folder of dataset CSVs; for each `<folder>/<name>.csv` it works on the matching
`<folder>/<name>/` SDF directory. For every row whose SMILES is multi-component it:

  - re-downloads and OVERWRITES the SDF with the cleaned single-organic parent, or
  - deletes the existing (multi-component) SDF when no single organic parent remains, so the
    Rust loader simply skips that row instead of training on a stale mixture.

Datasets whose SMILES column contains no `.` are left completely untouched.

Run it from the `scripts/` folder, same as `download_all.py`:
    python repair_split_mols.py --path C:/Users/the_a/Desktop/bio_misc/tdc_data
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Make the sibling module importable no matter the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_mols_for_dataset import (  # noqa: E402
    SLEEP_BETWEEN_MOLS,
    TDC_SMILES_COL,
    clean_smiles,
    download_sdf,
    generate_sdf_local,
    sdf_component_count,
)


def repair_csv(csv_path: Path, smiles_col: int) -> None:
    stem = csv_path.stem
    sdf_dir = csv_path.parent / stem

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        print(f"{stem}: empty CSV; skipping")
        return

    # Drop the header. Data-row index i maps to `<stem>_id_<i>.sdf`, matching the 0-based
    # enumeration the downloader uses (it consumes the header, then enumerates from 0).
    data_rows = rows[1:]

    affected = [
        (i, row[smiles_col].strip())
        for i, row in enumerate(data_rows)
        if len(row) > smiles_col and "." in row[smiles_col]
    ]

    if not affected:
        print(f"{stem}: no multi-component SMILES; skipping")
        return

    if not sdf_dir.is_dir():
        print(f"{stem}: {len(affected)} affected rows but no SDF folder at {sdf_dir}; skipping")
        return

    print(f"\n{stem}: repairing {len(affected)} multi-component rows ...")

    repaired = 0
    repaired_local = 0
    removed = 0
    already_ok = 0
    failed = 0

    for i, smiles in affected:
        mol_id = f"{stem}_id_{i}"
        out_path = sdf_dir / f"{mol_id}.sdf"

        cleaned = clean_smiles(smiles)

        if cleaned is None:
            # No single organic parent (e.g. an isomer mixture). Remove the stale SDF so the
            # training/inference loader skips this row rather than using the mixture.
            if out_path.exists():
                out_path.unlink()
                removed += 1
                print(f"  removed (no single organic parent): {mol_id}  [{smiles}]")
            else:
                print(f"  skip (no parent, nothing downloaded): {mol_id}")
            continue

        # Resumability: if the SDF already on disk is a single connected component, an
        # earlier run already repaired this row — skip the network round-trip.
        if out_path.exists() and sdf_component_count(out_path) == 1:
            already_ok += 1
            print(f"  already repaired; skipping: {mol_id}")
            continue

        sdf_text = download_sdf(cleaned, timeout_s=10)
        source = "pubchem"
        if sdf_text is None:
            # PubChem had no record (or stayed unavailable); build the structure locally.
            sdf_text = generate_sdf_local(cleaned)
            source = "local"

        if sdf_text is None:
            failed += 1
            print(f"  FAILED (no PubChem record; local embedding failed): {mol_id}  [{cleaned}]")
        else:
            try:
                with open(out_path, "w", encoding="utf-8", newline="\n") as out_f:
                    out_f.write(sdf_text)
                repaired += 1
                if source == "local":
                    repaired_local += 1
                    print(f"  repaired (local): {mol_id}  {smiles} -> {cleaned}")
                else:
                    print(f"  repaired: {mol_id}  {smiles} -> {cleaned}")
            except OSError:
                failed += 1
                print(f"  FAILED (write error): {mol_id}  [{cleaned}]")

        if SLEEP_BETWEEN_MOLS > 0:
            time.sleep(SLEEP_BETWEEN_MOLS)

    print(
        f"{stem}: repaired={repaired} (local={repaired_local}) removed={removed} "
        f"already_ok={already_ok} failed={failed}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repair multi-component (salt/mixture) SDFs in place for downloaded datasets."
    )
    ap.add_argument(
        "--path", type=str, required=True, help="Folder containing the dataset CSVs"
    )
    ap.add_argument(
        "--smiles_col",
        type=int,
        default=TDC_SMILES_COL,
        help="0-based SMILES column index in each CSV (TDC default: 1)",
    )
    args = ap.parse_args()

    folder = Path(args.path)
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    for csv_file in sorted(folder.iterdir()):
        if csv_file.is_file() and csv_file.suffix.lower() == ".csv":
            repair_csv(csv_file, args.smiles_col)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
