# """
# This script downloads molecules from the AqSolDb `data_curated.csv` or Therapeutic Data Commons molecules
# from PubChem, storing them in a folder. For AqSolDb, the IDs are the filenames stored.
#
# For TDC, filenames are by index, starting at 0.
#
#
# Example running:
# `python download_mols_for_dataset.py --csv /set1.csv --out /sdf_out_set1
#
# Or for our current use:
# `python .\download_mols_for_dataset.py --csv C:\Users\the_a\Desktop\bio_misc\tdc_data\caco2_wang.csv --out C:\Users\the_a\Desktop\bio_misc\tdc_data\mols_caco2_wang`
# """

import argparse
import csv
import os
import time
import urllib.parse

import requests


AQ_SOL_ID_COL = 0
AQ_SOL_INCHIKEY_COL = 3
AQ_SOL_SMILES_COL = 4

TDC_SMILES_COL = 1

# PubCHem requests no more than 5 per second. We pad this.
SLEEP_BETWEEN_MOLS = 0.22  # Seconds.


# def sdf_url_from_smiles(ident: str) -> str:
#     """We use Smiles generally, as both TDC and AqSolDb use this. TDC also has common name.
#     AqSolDb has Inchi, InchiKey, and common nam.e"""
#     # PubChem PUG REST: /compound/smiles/<SMILES>/SDF?record_type=3d
#     # SMILES must be URL-encoded because it often contains characters like #, +, /, =, etc.
#     encoded = urllib.parse.quote(ident, safe="")
#     return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/SDF?record_type=3d"
#     # return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{encoded}/SDF?record_type=3d"


def download_sdf(ident: str, timeout_s: float) -> str:
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/SDF"

    headers = {"User-Agent": "aqsoldb-pubchem-sdf-downloader/1.0"}

    params = {"smiles": ident, "record_type": "3d"}
    resp = requests.get(base, params=params, headers=headers, timeout=timeout_s)

    # If PubChem has no 3D conformer, fall back to 2D.
    if resp.status_code == 404:
        params["record_type"] = "2d"
        resp = requests.get(base, params=params, headers=headers, timeout=timeout_s)

    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    try:
        return resp.text
    except UnicodeDecodeError:
        return resp.content.decode("latin-1")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download PubChem 3D SDFs associated with a data set."
    )
    ap.add_argument(
        "--csv", type=str, required=True, help="Path to the CSV listing mols"
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row index (0-based, excluding header)",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=None,
        help="End row index (exclusive, excluding header)",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Folder to place the downloaded molecules",
    )
    ap.add_argument("--smiles_col", type=int, default=TDC_SMILES_COL)
    ap.add_argument("--id_col", type=int, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            raise SystemExit("CSV appears empty.")

        for i, row in enumerate(rdr):
            if i < args.start:
                continue
            if args.end is not None and i >= args.end:
                break

            if len(row) <= args.smiles_col:
                failed += 1
                continue

            if args.id_col is None:
                dataset_stem = os.path.splitext(os.path.basename(args.csv))[0]
                mol_id = f"{dataset_stem}_id_{i + args.start}"
            else:
                mol_id = row[args.id_col].strip()

            # inchikey = row[AQ_SOL_INCHIKEY_COL].strip() # Unused for now, e.g. TDC CSVs don't have this.
            smiles = row[args.smiles_col].strip()

            if not mol_id or not smiles:
                skipped += 1
                continue

            out_path = os.path.join(args.out_path, f"{mol_id}.sdf")

            if os.path.exists(out_path):
                skipped += 1
                continue

            try:
                sdf_text = download_sdf(smiles, timeout_s=10)
                with open(out_path, "w", encoding="utf-8", newline="\n") as out_f:
                    out_f.write(sdf_text)

                print(f"Success: {mol_id}")
                downloaded += 1

            except requests.HTTPError:
                print(f"Failed (HTTP): {mol_id}")
                failed += 1

            except (requests.RequestException, OSError):
                print(f"Failed (Req exception): {mol_id}")
                failed += 1

            if SLEEP_BETWEEN_MOLS > 0:
                time.sleep(SLEEP_BETWEEN_MOLS)

    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
