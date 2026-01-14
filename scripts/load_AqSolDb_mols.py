#!/usr/bin/env python3
import argparse
import csv
import os
import time
import urllib.parse

import requests


ID_COL = 0
INCHIKEY_COL = 3
SMILES_COL = 4

OUT_PATH = "./AqSolDb_mols"


def sdf_url_from_smiles(smiles: str) -> str:
    # PubChem PUG REST: /compound/smiles/<SMILES>/SDF?record_type=3d
    # SMILES must be URL-encoded because it often contains characters like #, +, /, =, etc.
    encoded = urllib.parse.quote(smiles, safe="")
#     return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/SDF?record_type=3d"
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{encoded}/SDF?record_type=3d"


def download_sdf_smiles(smiles: str, timeout_s: float) -> str:
    url = sdf_url_from_smiles(smiles)

    resp = requests.get(
        url,
        headers={"User-Agent": "aqsoldb-pubchem-sdf-downloader/1.0"},
        timeout=timeout_s,
    )

    # If PubChem has no 3D conformer, fall back to 2D.
    if resp.status_code == 404:
        url_2d = url.replace("3d", "2d")
        resp = requests.get(
            url_2d,
            headers={"User-Agent": "aqsoldb-pubchem-sdf-downloader/1.0"},
            timeout=timeout_s,
        )

    resp.raise_for_status()

    # PubChem SDF should be UTF-8/ASCII; if requests couldn't confidently detect it, fall back.
    resp.encoding = resp.encoding or "utf-8"
    try:
        return resp.text
    except UnicodeDecodeError:
        return resp.content.decode("latin-1")



def main() -> int:
    ap = argparse.ArgumentParser(description="Download PubChem 3D SDFs for AqSolDB rows (ID + SMILES).")
    ap.add_argument("--csv", required=True, help="Path to AqSolDB data_curated.csv")
    ap.add_argument("--start", type=int, default=0, help="Start row index (0-based, excluding header)")
    ap.add_argument("--end", type=int, default=None, help="End row index (exclusive, excluding header)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between requests")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .sdf files")
    args = ap.parse_args()

    os.makedirs(OUT_PATH, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            raise SystemExit("CSV appears empty.")

        for idx, row in enumerate(rdr):
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break

            if len(row) <= max(ID_COL, SMILES_COL):
                failed += 1
                continue

            mol_id = row[ID_COL].strip()
            inchikey = row[INCHIKEY_COL].strip()
            smiles = row[SMILES_COL].strip()

            if not mol_id or not smiles:
                skipped += 1
                continue

            out_path = os.path.join(OUT_PATH, f"{mol_id}.sdf")

            if (not args.overwrite) and os.path.exists(out_path):
                skipped += 1
                continue

            try:
#                 sdf_text = download_sdf_smiles(smiles, timeout_s=args.timeout)
                sdf_text = download_sdf_smiles(inchikey, timeout_s=args.timeout)
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

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
