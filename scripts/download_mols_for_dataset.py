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
from pathlib import Path

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from adme_data_model import (
    load_structure_overrides,
    standardize_structure,
    validate_assayed_sdf,
    validate_parent_sdf,
    validated_override,
)

# RDKit logs parse/sanitize chatter to stderr; silence it so our own output stays readable.
RDLogger.DisableLog("rdApp.*")

AQ_SOL_ID_COL = 0
AQ_SOL_INCHIKEY_COL = 3
AQ_SOL_SMILES_COL = 4

TDC_SMILES_COL = 1

# PubCHem requests no more than 5 per second. We pad this.
SLEEP_BETWEEN_MOLS = 0.22  # Seconds.

# Transient HTTP statuses worth retrying (PubChem throttling / brief server hiccups), as
# opposed to a definitive 404 (no such structure). When these persist past the retry budget
# the caller falls back to local 3D generation rather than treating it as "not found".
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 1.0  # Seconds; doubled each retry (1, 2, 4, ...).


def clean_smiles(smiles: str):
    """Return the standardized ML parent without modifying the assayed form.

    Kept as a compatibility helper for ``repair_split_mols.py``. New downloads
    call :func:`standardize_structure` and store assayed and parent SDFs separately.
    """

    try:
        return standardize_structure(smiles).parent_smiles
    except Exception:
        return None


# def sdf_url_from_smiles(ident: str) -> str:
#     """We use Smiles generally, as both TDC and AqSolDb use this. TDC also has common name.
#     AqSolDb has Inchi, InchiKey, and common nam.e"""
#     # PubChem PUG REST: /compound/smiles/<SMILES>/SDF?record_type=3d
#     # SMILES must be URL-encoded because it often contains characters like #, +, /, =, etc.
#     encoded = urllib.parse.quote(ident, safe="")
#     return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/SDF?record_type=3d"
#     # return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{encoded}/SDF?record_type=3d"


def download_sdf(ident: str, timeout_s: float, max_retries: int = MAX_HTTP_RETRIES):
    """Fetch a PubChem SDF for a SMILES, trying a 3D conformer first then 2D.

    Returns the SDF text, or None if PubChem has no matching record (404 on both detail
    levels) or stays unavailable after retries. Transient failures (429/5xx, connection
    drops, timeouts) are retried with exponential backoff so throttling isn't mistaken for
    "not found"; the caller treats None as a cue to generate the structure locally.
    """
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/SDF"
    headers = {"User-Agent": "aqsoldb-pubchem-sdf-downloader/1.0"}

    for record_type in ("3d", "2d"):
        params = {"smiles": ident, "record_type": record_type}

        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(
                    base, params=params, headers=headers, timeout=timeout_s
                )
            except requests.RequestException:
                # Connection error / timeout: back off and retry, else give up on PubChem.
                if attempt < max_retries:
                    time.sleep(HTTP_BACKOFF_BASE * (2**attempt))
                    continue
                return None

            if resp.status_code == 200:
                resp.encoding = resp.encoding or "utf-8"
                try:
                    return resp.text
                except UnicodeDecodeError:
                    return resp.content.decode("latin-1")

            if resp.status_code == 404:
                break  # No record at this detail level; try 2D next, then give up.

            if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
                time.sleep(HTTP_BACKOFF_BASE * (2**attempt))
                continue

            # Non-retryable (e.g. 400) or retries exhausted: stop trying PubChem.
            return None

    return None


def generate_sdf_local(
    ident: str,
    *,
    structure_role: str = "parent",
    assayed_smiles: str | None = None,
):
    """Generate a 3D conformer locally with RDKit, for when PubChem has no record.

    Returns SDF text (a molblock, a `GENERATED_BY` marker field, and the `$$$$` terminator),
    or None if the SMILES can't be parsed or embedded. This is the fallback that lets valid
    molecules which simply aren't in PubChem — unusual structures, off-tautomers, desalted
    parents — still get a usable 3D structure instead of being dropped.
    """
    mol = Chem.MolFromSmiles(ident)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D  # Deterministic conformers across runs.
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Distance-geometry embedding failed; retry from random coordinates.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None

    # Refine geometry; keep the embedded coordinates if force-field params are unavailable.
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    molblock = Chem.MolToMolBlock(mol)
    if not molblock.endswith("\n"):
        molblock += "\n"

    metadata = (
        f"> <GENERATED_BY>\nRDKit ETKDGv3\n\n"
        f"> <STRUCTURE_ROLE>\n{structure_role}\n\n"
        f"> <STRUCTURE_SMILES>\n{ident}\n\n"
    )
    if assayed_smiles is not None:
        metadata += f"> <ASSAYED_SMILES>\n{assayed_smiles}\n\n"
    return f"{molblock}{metadata}$$$$\n"


def sdf_component_count(path):
    """Disconnected-component count of the first molecule in an SDF, or None if it can't be
    read. Parsing skips sanitization so PubChem valence quirks don't matter — only the bond
    connectivity is needed. A clean/repaired SDF returns 1; an unrepaired multi-component
    (salt/mixture) SDF returns >1; a missing or corrupt file returns None. Used to tell
    whether a file already on disk is good (skip) or stale and in need of rebuilding.
    """
    mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        return None
    return len(Chem.GetMolFrags(mol))


def annotate_sdf(
    sdf_text: str, structure_role: str, smiles: str, assayed_smiles: str
) -> str:
    """Add provenance fields without changing the SDF molecular graph."""

    marker = sdf_text.rfind("$$$$")
    if marker < 0:
        return sdf_text
    metadata = (
        f"> <STRUCTURE_ROLE>\n{structure_role}\n\n"
        f"> <STRUCTURE_SMILES>\n{smiles}\n\n"
        f"> <ASSAYED_SMILES>\n{assayed_smiles}\n\n"
    )
    return f"{sdf_text[:marker]}{metadata}$$$$\n"


def write_sdf(path: str, sdf_text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write(sdf_text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare separate assayed-form and standardized-parent SDFs for a dataset. "
            "The original structure is never replaced by the parent."
        )
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
        "--out",
        dest="out_path",
        type=str,
        required=True,
        help="Folder for standardized-parent SDFs (legacy training location)",
    )
    ap.add_argument(
        "--assayed_out_path",
        type=str,
        default=None,
        help="Folder for exact assayed-form SDFs (default: <data-dir>/assayed_forms/<dataset>)",
    )
    ap.add_argument("--smiles_col", type=int, default=TDC_SMILES_COL)
    ap.add_argument("--id_col", type=int, default=None)
    ap.add_argument(
        "--structure_overrides",
        default=os.path.join(
            os.path.dirname(__file__), "therapeutic_structure_overrides.json"
        ),
        help="Documented source corrections used only to derive standardized parents",
    )

    args = ap.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    dataset_stem = os.path.splitext(os.path.basename(args.csv))[0]
    data_dir = os.path.dirname(os.path.abspath(args.csv))
    assayed_out_path = args.assayed_out_path or os.path.join(
        data_dir, "assayed_forms", dataset_stem
    )
    os.makedirs(assayed_out_path, exist_ok=True)
    overrides, _ = load_structure_overrides(Path(args.structure_overrides))

    parent_downloaded = 0
    parent_generated_local = 0
    parent_skipped = 0
    assayed_generated_local = 0
    assayed_skipped = 0
    failed = 0
    assayed_unavailable = 0

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
                mol_id = f"{dataset_stem}_id_{i}"
            else:
                mol_id = row[args.id_col].strip()

            # inchikey = row[AQ_SOL_INCHIKEY_COL].strip() # Unused for now, e.g. TDC CSVs don't have this.
            smiles = row[args.smiles_col].strip()

            if not mol_id or not smiles:
                failed += 1
                continue

            try:
                override = validated_override(overrides, dataset_stem, i, smiles)
                structure = standardize_structure(
                    smiles,
                    replacement_smiles=(
                        override["replacement_smiles"] if override is not None else None
                    ),
                )
            except Exception as error:
                print(f"Failed to standardize {mol_id}: {error}")
                failed += 1
                continue

            assayed_path = os.path.join(assayed_out_path, f"{mol_id}.sdf")
            if structure.assayed_component_count is None:
                print(f"Assayed form unavailable (unparseable original): {mol_id}")
                assayed_unavailable += 1
            elif (
                validate_assayed_sdf(
                    Path(assayed_path),
                    structure.assayed_inchi_key,
                    structure.assayed_component_count,
                )
                is None
            ):
                assayed_skipped += 1
            else:
                assayed_sdf = generate_sdf_local(
                    structure.assayed_canonical_smiles,
                    structure_role="assayed",
                    assayed_smiles=smiles,
                )
                if assayed_sdf is None:
                    print(f"Failed to generate exact assayed form: {mol_id}")
                    failed += 1
                else:
                    write_sdf(assayed_path, assayed_sdf)
                    invalid_assayed = validate_assayed_sdf(
                        Path(assayed_path),
                        structure.assayed_inchi_key,
                        structure.assayed_component_count,
                    )
                    if invalid_assayed is not None:
                        os.remove(assayed_path)
                        print(
                            f"Failed assayed-form identity validation "
                            f"({invalid_assayed}): {mol_id}"
                        )
                        failed += 1
                    else:
                        assayed_generated_local += 1

            parent_path = os.path.join(args.out_path, f"{mol_id}.sdf")
            if (
                validate_parent_sdf(Path(parent_path), structure.parent_inchi_key)
                is None
            ):
                parent_skipped += 1
                continue

            parent_sdf = download_sdf(structure.parent_smiles, timeout_s=10)
            if parent_sdf is not None:
                parent_source = "pubchem"
                parent_sdf = annotate_sdf(
                    parent_sdf,
                    "parent",
                    structure.parent_smiles,
                    smiles,
                )
            else:
                parent_source = "local"
                parent_sdf = generate_sdf_local(
                    structure.parent_smiles,
                    structure_role="parent",
                    assayed_smiles=smiles,
                )

            if parent_sdf is None:
                print(f"Failed to prepare standardized parent: {mol_id}")
                failed += 1
            else:
                write_sdf(parent_path, parent_sdf)
                invalid_reason = validate_parent_sdf(
                    Path(parent_path), structure.parent_inchi_key
                )
                if invalid_reason is not None:
                    os.remove(parent_path)
                    print(
                        f"Failed parent identity validation ({invalid_reason}): {mol_id}"
                    )
                    failed += 1
                else:
                    if parent_source == "pubchem":
                        parent_downloaded += 1
                    else:
                        parent_generated_local += 1
                    print(f"Prepared assayed/parent structures: {mol_id}")

            if SLEEP_BETWEEN_MOLS > 0:
                time.sleep(SLEEP_BETWEEN_MOLS)

    print(f"Parent downloaded:        {parent_downloaded}")
    print(f"Parent generated locally: {parent_generated_local}")
    print(f"Parent already present:   {parent_skipped}")
    print(f"Assayed generated locally:{assayed_generated_local}")
    print(f"Assayed already present:  {assayed_skipped}")
    print(f"Assayed unavailable:      {assayed_unavailable}")
    print(f"Failed:                   {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
