"""Add HMDB's cross-database identifiers to the HMDB structures SDF.

PubChem CIDs and ChEBI, DrugBank and KEGG accessions all live in the metabolite XML but not
in the structures SDF, where our ingest pipeline can see them. This copies each across, for
the metabolites that have it.

Download ``hmdb_metabolites.zip`` and ``structures.sdf`` from:
https://hmdb.ca/downloads

Put both files in the same directory, then run, for example:

    python scripts/populate_hmdb_cids.py ~/Desktop

The output is written beside them as ``structures_with_pubchem.sdf``.
Both large input files are streamed, so memory use does not scale with their size.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


ARCHIVE_NAME = "hmdb_metabolites.zip"
STRUCTURES_NAME = "structures.sdf"
OUTPUT_NAME = "structures_with_pubchem.sdf"

HMDB_NAMESPACE = "http://www.hmdb.ca"
METABOLITE_TAG = f"{{{HMDB_NAMESPACE}}}metabolite"
ACCESSION_TAG = f"{{{HMDB_NAMESPACE}}}accession"

DATABASE_ID_RE = re.compile(
    rb"(?m)^>\s*<DATABASE_ID>[^\r\n]*\r?\n([^\r\n]*)"
)


def _chebi_accession(value: str) -> str:
    """ChEBI accessions are conventionally written ``CHEBI:15377``; HMDB stores the bare number."""
    digits = value[6:].strip() if value[:6].upper() == "CHEBI:" else value
    return f"CHEBI:{digits}"


@dataclass(frozen=True)
class IdField:
    """One identifier, copied from a ``<metabolite>`` child element to an SDF data field."""

    xml_tag: str
    sdf_key: str
    format_value: Callable[[str], str] = str

    @property
    def element_tag(self) -> str:
        return f"{{{HMDB_NAMESPACE}}}{self.xml_tag}"

    @property
    def value_re(self) -> re.Pattern[bytes]:
        """Match an existing occurrence of this field in a record: the header, then its value."""
        key = re.escape(self.sdf_key.encode("utf-8"))
        return re.compile(rb"(?m)(^>\s*<" + key + rb">[^\r\n]*\r?\n)([^\r\n]*)")


# The SDF keys are the ones the ingest pipeline reads identifiers back out of; see `MD_KEYS_*`
# in `mol_defs`.
ID_FIELDS: tuple[IdField, ...] = (
    IdField("pubchem_compound_id", "PUBCHEM_COMPOUND_CID"),
    IdField("chebi_id", "ChEBI ID", _chebi_accession),
    IdField("drugbank_id", "DRUGBANK_ID"),
    IdField("kegg_id", "KEGG COMPOUND Database Links"),
)

VALUE_RE_BY_KEY = {field.sdf_key: field.value_re for field in ID_FIELDS}


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _counts_summary(counts: dict[str, int]) -> str:
    return ", ".join(f"{count:,} {key}" for key, count in counts.items())


def _xml_member_name(archive: zipfile.ZipFile) -> str:
    """Find the HMDB metabolites XML member without extracting the archive."""
    xml_members = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and name.lower().endswith(".xml")
    ]
    exact_matches = [
        name for name in xml_members if Path(name).name.lower() == "hmdb_metabolites.xml"
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(xml_members) == 1:
        return xml_members[0]
    if not xml_members:
        raise ValueError("the archive does not contain an XML file")
    raise ValueError(
        "the archive contains multiple XML files and none is named hmdb_metabolites.xml"
    )


def _metabolite_ids(element: ET.Element) -> dict[str, str]:
    """The identifiers one metabolite carries, keyed by SDF field name; absent ones are omitted."""
    ids: dict[str, str] = {}
    for field in ID_FIELDS:
        value = _text(element.find(field.element_tag))
        if value is not None:
            ids[field.sdf_key] = field.format_value(value)
    return ids


def load_ids(archive_path: Path) -> dict[str, dict[str, str]]:
    """Return a mapping of primary/secondary HMDB accessions to their cross-database identifiers."""
    ids_by_accession: dict[str, dict[str, str]] = {}
    metabolite_count = 0
    counts = {field.sdf_key: 0 for field in ID_FIELDS}

    with zipfile.ZipFile(archive_path) as archive:
        member_name = _xml_member_name(archive)
        with archive.open(member_name) as xml_file:
            context = ET.iterparse(xml_file, events=("start", "end"))
            _, root = next(context)

            for event, element in context:
                if event != "end" or element.tag != METABOLITE_TAG:
                    continue

                metabolite_count += 1
                ids = _metabolite_ids(element)
                primary_accession = _text(element.find(ACCESSION_TAG))

                if ids and primary_accession:
                    accessions = [primary_accession]
                    secondary = element.find(f"{{{HMDB_NAMESPACE}}}secondary_accessions")
                    if secondary is not None:
                        accessions.extend(
                            value
                            for accession in secondary.findall(ACCESSION_TAG)
                            if (value := _text(accession)) is not None
                        )

                    for accession in accessions:
                        known = ids_by_accession.setdefault(accession, {})
                        for key, value in ids.items():
                            previous = known.get(key)
                            if previous is not None and previous != value:
                                raise ValueError(
                                    f"conflicting {key} values for HMDB accession "
                                    f"{accession}: {previous} and {value}"
                                )
                            known[key] = value

                    for key in ids:
                        counts[key] += 1

                # Discard the just-processed subtree. This is essential for the multi-GB XML.
                root.clear()

                if metabolite_count % 10_000 == 0:
                    print(
                        f"[XML] processed {metabolite_count:,} metabolites; "
                        f"found {_counts_summary(counts)}",
                        flush=True,
                    )

    print(
        f"[XML] complete: {metabolite_count:,} metabolites; {_counts_summary(counts)}",
        flush=True,
    )
    return ids_by_accession


def _database_id(record: bytes) -> str | None:
    match = DATABASE_ID_RE.search(record)
    if match is None:
        return None
    return match.group(1).strip().decode("utf-8", errors="replace") or None


def _with_ids(record: bytes, ids: dict[str, str]) -> bytes:
    """Add the identifier properties to one complete SDF record, replacing any already present."""
    for key, value in ids.items():
        value_bytes = value.encode("utf-8")
        value_re = VALUE_RE_BY_KEY[key]

        if value_re.search(record):
            record = value_re.sub(
                lambda match, value_bytes=value_bytes: match.group(1) + value_bytes,
                record,
                count=1,
            )
            continue

        delimiter_match = re.search(rb"(?m)^\$\$\$\$(?:\r?\n)?\Z", record)
        if delimiter_match is None:
            raise ValueError("encountered an SDF record without a $$$$ delimiter")

        newline = b"\r\n" if b"\r\n" in record else b"\n"
        body = record[: delimiter_match.start()]
        delimiter = record[delimiter_match.start() :]
        if body and not body.endswith((b"\n", b"\r")):
            body += newline
        field = (
            b"> <"
            + key.encode("utf-8")
            + b">"
            + newline
            + value_bytes
            + newline
            + newline
        )
        record = body + field + delimiter

    return record


def populate_sdf(
    structures_path: Path,
    output_path: Path,
    ids_by_accession: dict[str, dict[str, str]],
) -> tuple[int, int, int]:
    """Stream SDF records to a temporary file and atomically install the result."""
    record_count = 0
    populated_count = 0
    missing_count = 0
    record = bytearray()
    temp_path: Path | None = None

    try:
        with structures_path.open("rb") as source, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temp_path = Path(destination.name)

            for line in source:
                record.extend(line)
                if line.rstrip(b"\r\n") != b"$$$$":
                    continue

                record_count += 1
                record_bytes = bytes(record)
                accession = _database_id(record_bytes)
                ids = ids_by_accession.get(accession) if accession else None
                if not ids:
                    missing_count += 1
                else:
                    record_bytes = _with_ids(record_bytes, ids)
                    populated_count += 1
                destination.write(record_bytes)
                record.clear()

                if record_count % 10_000 == 0:
                    print(
                        f"[SDF] processed {record_count:,} records; "
                        f"populated {populated_count:,}",
                        flush=True,
                    )

            if record:
                raise ValueError("the final SDF record is not terminated by $$$$")

            destination.flush()
            os.fsync(destination.fileno())

        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return record_count, populated_count, missing_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add PubChem, ChEBI, DrugBank and KEGG identifiers from hmdb_metabolites.zip "
            "to structures.sdf."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing hmdb_metabolites.zip and structures.sdf",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"replace an existing {OUTPUT_NAME}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    archive_path = directory / ARCHIVE_NAME
    structures_path = directory / STRUCTURES_NAME
    output_path = directory / OUTPUT_NAME

    if not directory.is_dir():
        raise SystemExit(f"error: not a directory: {directory}")
    for path in (archive_path, structures_path):
        if not path.is_file():
            raise SystemExit(f"error: required input file not found: {path}")
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"error: output already exists: {output_path} (pass --force to replace it)"
        )

    print(f"Reading identifiers from {archive_path}", flush=True)
    ids_by_accession = load_ids(archive_path)
    print(f"Writing {output_path}", flush=True)
    records, populated, missing = populate_sdf(
        structures_path, output_path, ids_by_accession
    )
    print(
        f"[done] {records:,} SDF records; {populated:,} populated; "
        f"{missing:,} without any matching identifier",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
