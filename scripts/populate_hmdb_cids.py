"""Add HMDB's PubChem compound IDs to the HMDB structures SDF.

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
from pathlib import Path


ARCHIVE_NAME = "hmdb_metabolites.zip"
STRUCTURES_NAME = "structures.sdf"
OUTPUT_NAME = "structures_with_pubchem.sdf"

HMDB_NAMESPACE = "http://www.hmdb.ca"
METABOLITE_TAG = f"{{{HMDB_NAMESPACE}}}metabolite"
ACCESSION_TAG = f"{{{HMDB_NAMESPACE}}}accession"
PUBCHEM_ID_TAG = f"{{{HMDB_NAMESPACE}}}pubchem_compound_id"

DATABASE_ID_RE = re.compile(
    rb"(?m)^>\s*<DATABASE_ID>[^\r\n]*\r?\n([^\r\n]*)"
)
PUBCHEM_VALUE_RE = re.compile(
    rb"(?m)(^>\s*<PUBCHEM_COMPOUND_CID>[^\r\n]*\r?\n)([^\r\n]*)"
)


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


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


def load_pubchem_cids(archive_path: Path) -> dict[str, str]:
    """Return a mapping of primary/secondary HMDB accessions to PubChem CIDs."""
    cid_by_accession: dict[str, str] = {}
    metabolite_count = 0
    cid_count = 0

    with zipfile.ZipFile(archive_path) as archive:
        member_name = _xml_member_name(archive)
        with archive.open(member_name) as xml_file:
            context = ET.iterparse(xml_file, events=("start", "end"))
            _, root = next(context)

            for event, element in context:
                if event != "end" or element.tag != METABOLITE_TAG:
                    continue

                metabolite_count += 1
                cid = _text(element.find(PUBCHEM_ID_TAG))
                primary_accession = _text(element.find(ACCESSION_TAG))

                if cid and primary_accession:
                    accessions = [primary_accession]
                    secondary = element.find(f"{{{HMDB_NAMESPACE}}}secondary_accessions")
                    if secondary is not None:
                        accessions.extend(
                            value
                            for accession in secondary.findall(ACCESSION_TAG)
                            if (value := _text(accession)) is not None
                        )

                    for accession in accessions:
                        previous = cid_by_accession.get(accession)
                        if previous is not None and previous != cid:
                            raise ValueError(
                                f"conflicting PubChem CIDs for HMDB accession {accession}: "
                                f"{previous} and {cid}"
                            )
                        cid_by_accession[accession] = cid
                    cid_count += 1

                # Discard the just-processed subtree. This is essential for the multi-GB XML.
                root.clear()

                if metabolite_count % 10_000 == 0:
                    print(
                        f"[XML] processed {metabolite_count:,} metabolites; "
                        f"found {cid_count:,} PubChem CIDs",
                        flush=True,
                    )

    print(
        f"[XML] complete: {metabolite_count:,} metabolites; "
        f"{cid_count:,} with PubChem CIDs",
        flush=True,
    )
    return cid_by_accession


def _database_id(record: bytes) -> str | None:
    match = DATABASE_ID_RE.search(record)
    if match is None:
        return None
    return match.group(1).strip().decode("utf-8", errors="replace") or None


def _with_pubchem_cid(record: bytes, cid: str) -> bytes:
    """Add the CID property to one complete SDF record, replacing it if present."""
    cid_bytes = cid.encode("ascii")
    if PUBCHEM_VALUE_RE.search(record):
        return PUBCHEM_VALUE_RE.sub(
            lambda match: match.group(1) + cid_bytes, record, count=1
        )

    delimiter_match = re.search(rb"(?m)^\$\$\$\$(?:\r?\n)?\Z", record)
    if delimiter_match is None:
        raise ValueError("encountered an SDF record without a $$$$ delimiter")

    newline = b"\r\n" if b"\r\n" in record else b"\n"
    body = record[: delimiter_match.start()]
    delimiter = record[delimiter_match.start() :]
    if body and not body.endswith((b"\n", b"\r")):
        body += newline
    field = b"> <PUBCHEM_COMPOUND_CID>" + newline + cid_bytes + newline + newline
    return body + field + delimiter


def populate_sdf(
    structures_path: Path,
    output_path: Path,
    cid_by_accession: dict[str, str],
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
                cid = cid_by_accession.get(accession) if accession else None
                if cid is None:
                    missing_count += 1
                else:
                    record_bytes = _with_pubchem_cid(record_bytes, cid)
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
            "Add PubChem compound IDs from hmdb_metabolites.zip to structures.sdf."
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

    print(f"Reading PubChem CIDs from {archive_path}", flush=True)
    cid_by_accession = load_pubchem_cids(archive_path)
    print(f"Writing {output_path}", flush=True)
    records, populated, missing = populate_sdf(
        structures_path, output_path, cid_by_accession
    )
    print(
        f"[done] {records:,} SDF records; {populated:,} populated; "
        f"{missing:,} without a matching PubChem CID",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
