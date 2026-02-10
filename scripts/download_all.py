"""
A wrapper around the mol_downloader script, which handles all csvs in a folder.
It places the SDF folders in the same path as the parent.
"""

import argparse
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser(
    description="Download PubChem 3D SDFs associated with multiple data sets."
)
ap.add_argument("--path", type=str, required=True, help="The path containing the CSVs")
args = ap.parse_args()

for file in Path(args.path).iterdir():
    if file.is_file() and file.suffix.lower() == ".csv":
        name = file.stem
        out_dir = str(Path(args.path) / name)

        print(f"\n\nDownloading dataset for {name} ...\n")
        subprocess.run(
            [
                "python",
                "./download_mols_for_dataset.py",
                "--csv",
                str(file),
                "--out",
                out_dir,
            ],
            check=True,
        )
