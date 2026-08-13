# Canonical therapeutic observation data

The canonical training inputs live under `S:\bio_misc\tdc_data\canonical\v1`.
They are Parquet files with one lossless observation row for every row of each
source TDC CSV. `all_observations.parquet` concatenates all 24 datasets and
`catalog.json` records file hashes, row counts, structure coverage, and versions.
`coverage_report.json` makes missing context fields and structure exclusions
explicit, while `structure_repair_queue.parquet` lists every SDF that needs to
be regenerated or corrected.

## Rebuild

From the repository root, using a Python environment with RDKit and PyArrow:

```powershell
python scripts\build_adme_parquet.py `
  --data-dir S:\bio_misc\tdc_data `
  --dataset-version tdc-local-2026-01-v1
```

The build is deterministic for the same CSVs, split manifests, endpoint
registry, structure overrides, RDKit, and PyArrow writer version. It fails if a
split is stale, a parent/scaffold differs from the split manifest, a row is
missing, or an eligible SDF does not exist. Generate the audit and repair queue
after a build with:

```powershell
python scripts\report_adme_coverage.py `
  --canonical-dir S:\bio_misc\tdc_data\canonical\v1
```

## Assayed form versus ML parent

`assayed_smiles` is exactly what the source supplied. It can be a salt,
co-crystal, solvate, or mixture. `parent_smiles` is a separately derived RDKit
fragment parent used by the present model. Counterions/co-formers are retained
in `removed_components`; they are not silently discarded from provenance.

To prepare both representations for a dataset:

```powershell
python scripts\download_mols_for_dataset.py `
  --csv S:\bio_misc\tdc_data\pgp_broccatelli.csv `
  --out_path S:\bio_misc\tdc_data\pgp_broccatelli
```

Parent SDFs remain in the legacy per-dataset folder used by training. Exact
assayed-form SDFs are written to `assayed_forms\<dataset>`. The two documented
invalid AqSolDB strings preserve their original text while using the audited
replacement only for parent derivation.

## Recovering metadata absent from TDC

The three-column TDC exports cannot supply per-observation units, qualifiers,
assay conditions, dosing, formulations, replicate identities, or variances.
Do not infer these from compound or endpoint names. Acquire observation-level
tables in this order:

1. Original supplementary tables or author/depositor exports.
2. ChEMBL activity and assay exports identified by stable assay/activity IDs.
3. First-party database releases such as FreeSolv and AqSolDB.
4. Publication tables transcribed with a retained table/row identifier.

Keep the downloaded file unchanged and record its version, SHA-256, URL,
retrieval date, and license. Confirm that the license permits local storage and
model training. Unpublished conditions must remain null.

Map the fields you want to merge to canonical column names, then run the strict
enrichment tool into a new directory. Any remaining source-native columns are
kept unchanged inside the row's provenance record:

```powershell
python scripts\merge_adme_upstream_metadata.py `
  --canonical-dir S:\bio_misc\tdc_data\canonical\v1 `
  --source S:\bio_misc\tdc_data\upstream\freesolv.parquet `
  --output-dir S:\bio_misc\tdc_data\canonical\v1-freesolv `
  --source-name FreeSolv `
  --source-version 0.52 `
  --source-license MIT
```

Linking uses `observation_id`, `(dataset_name, source_row_id)`, or a unique
`(dataset_name, source_record_id)`. It never overwrites a populated conflicting
value. Every source row is embedded in `provenance_json`; unmatched rows and
conflicts are written to `enrichment_report.json` for review.

Structure-plus-value fuzzy matching is intentionally not automatic. If stable
IDs are absent, create a reviewed crosswalk containing `observation_id` and the
upstream record ID. Suggested matches may use standardized parent InChIKey,
endpoint, value, unit, species, and assay context, but a person should approve
ambiguous matches.

## Training behavior

Rust training reads target values and parent SDF paths from the canonical
Parquet file. It verifies the CSV hash, split-manifest hash, schema version,
ordered row coverage, target values against the CSV, qualifiers, safe relative
paths, and file presence. The migration validates each parent SDF's component
count and InChIKey identity before marking that row eligible.
Missing structures are reported through `training_eligible=false` and
`exclusion_reason`; they are no longer silently dropped.
