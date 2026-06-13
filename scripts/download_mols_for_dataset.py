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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit logs parse/sanitize chatter to stderr; silence it so our own output stays readable.
RDLogger.DisableLog("rdApp.*")

# Neutralizes charges where chemically possible (e.g. a carboxylate left after stripping a
# metal counterion becomes the neutral acid), while leaving permanent charges such as
# quaternary ammonium intact. PubChem's exact-structure lookup usually has the neutral
# parent on file but not the bare ion, so this markedly improves the hit rate after desalting.
_UNCHARGER = rdMolStandardize.Uncharger()


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


def _finalize_fragment(mol):
    """Neutralize a chosen parent fragment where chemically possible, then return its
    canonical SMILES. Falls back to the original (charged) form if neutralization fails."""
    try:
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def clean_smiles(smiles: str):
    """Reduce a (possibly multi-component) SMILES to a single, non-ionic organic molecule.

    Returns the cleaned SMILES string, or None if the entry should be skipped. The steps,
    applied only when the SMILES has more than one component (contains `.`):

      1. Split into fragments and dedupe identical ones (collapses e.g. 2:1 salts that
         repeat the same anion).
      2. Keep only organic fragments (those containing carbon). This drops monatomic metal
         cations, halides, and polyatomic inorganic counterions (sulfate, phosphate,
         silicate, ...), none of which contain carbon.
      3. Keep the largest remaining organic fragment — the drug-like parent — discarding
         smaller counterions / co-formers (acetate, piperazine, etc.) whatever their size.
         Ties on heavy-atom count are broken by canonical SMILES so the choice is reproducible.
      4. Return None only when nothing organic remains (e.g. a purely inorganic salt) or the
         SMILES can't be parsed.

    The chosen parent is then neutralized where chemically possible (so a carboxylate left
    after dropping a metal counterion becomes the neutral acid), which PubChem is far more
    likely to have on file than the bare ion.
    """
    smiles = smiles.strip()
    if not smiles:
        return None

    # Fast path: already a single component — pass through untouched, so well-formed
    # single-component SMILES reach PubChem exactly as the dataset wrote them.
    if "." not in smiles:
        return smiles

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    else:
        # RDKit couldn't parse the combined SMILES; fall back to parsing each fragment
        # string on its own so one bad counterion doesn't sink an otherwise-usable row.
        frags = [Chem.MolFromSmiles(s) for s in smiles.split(".") if s]
        frags = [f for f in frags if f is not None]
        if not frags:
            return None

    # Dedupe identical fragments by canonical SMILES.
    unique = {}
    for f in frags:
        unique.setdefault(Chem.MolToSmiles(f), f)
    frags = list(unique.values())

    # Organic = contains at least one carbon atom.
    organic = [f for f in frags if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]

    if not organic:
        return None

    # Always keep the largest organic fragment (the drug-like parent); ties on heavy-atom
    # count are broken by canonical SMILES so the pick is reproducible across runs and
    # independent of the order the fragments were written in.
    organic.sort(key=lambda f: (-f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))
    return _finalize_fragment(organic[0])


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
                resp = requests.get(base, params=params, headers=headers, timeout=timeout_s)
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


def generate_sdf_local(ident: str):
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

    return f"{molblock}> <GENERATED_BY>\nRDKit ETKDGv3 (no PubChem record)\n\n$$$$\n"


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
    desalted = 0
    generated_local = 0
    removed = 0

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

            # Reduce salts/mixtures to a single, non-ionic organic parent (or skip).
            cleaned = clean_smiles(smiles)
            out_path = os.path.join(args.out_path, f"{mol_id}.sdf")

            if cleaned is None:
                # Genuine mixture / all-ionic with no single organic parent. Skip the row,
                # and clear any stale file a pre-fix run downloaded from the raw
                # multi-component SMILES (so it isn't left for the loader to reject later).
                if os.path.exists(out_path):
                    os.remove(out_path)
                    print(f"Removed stale file (no single organic parent): {mol_id}")
                    removed += 1
                else:
                    print(f"Skipped (multi-component, no single organic parent): {mol_id}")
                    skipped += 1
                continue

            # Self-heal / resume: skip only when the file on disk is already a single-component
            # structure. A missing, multi-component (pre-fix), or unreadable file is rebuilt.
            if os.path.exists(out_path) and sdf_component_count(out_path) == 1:
                skipped += 1
                continue

            if cleaned != smiles:
                print(f"Desalted {mol_id}: {smiles} -> {cleaned}")
                desalted += 1
            smiles = cleaned

            sdf_text = download_sdf(smiles, timeout_s=10)
            source = "pubchem"
            if sdf_text is None:
                # PubChem had no record (or stayed unavailable); build the structure locally.
                sdf_text = generate_sdf_local(smiles)
                source = "local"

            if sdf_text is None:
                print(f"Failed (no PubChem record; local embedding failed): {mol_id}")
                failed += 1
            else:
                try:
                    with open(out_path, "w", encoding="utf-8", newline="\n") as out_f:
                        out_f.write(sdf_text)
                    if source == "local":
                        print(f"Generated locally (no PubChem record): {mol_id}")
                        generated_local += 1
                    else:
                        print(f"Success: {mol_id}")
                        downloaded += 1
                except OSError:
                    print(f"Failed (write error): {mol_id}")
                    failed += 1

            if SLEEP_BETWEEN_MOLS > 0:
                time.sleep(SLEEP_BETWEEN_MOLS)

    print(f"Downloaded:      {downloaded}")
    print(f"Generated local: {generated_local}")
    print(f"Skipped:         {skipped}")
    print(f"Removed stale:   {removed}")
    print(f"Failed:          {failed}")
    print(f"Desalted:        {desalted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
