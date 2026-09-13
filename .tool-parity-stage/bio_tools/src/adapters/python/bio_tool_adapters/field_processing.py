"""
Processes input fields on the web API of various tools. Accepts text input and constraints,
and outputs a native data type.

"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from . import AMINO_ACIDS, DNA_BASES, ToolInputError


def text(
    payload: dict[str, Any],
    name: str,
    *,
    required: bool = True,
    max_length: int = 200_000,
) -> str:
    value = str(payload.get(name, "")).strip()

    if required and not value:
        raise ToolInputError(f"{name} is required.")

    if len(value) > max_length:
        raise ToolInputError(f"{name} is too long (maximum {max_length} characters).")

    return value


def boolean(payload: dict[str, Any], name: str, default: bool = False) -> bool:
    value = payload.get(name, default)
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def integer(
    payload: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ToolInputError(f"{name} must be between {minimum} and {maximum}.")
    return value


def decimal(
    payload: dict[str, Any],
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be a number.") from exc
    if not minimum <= value <= maximum:
        raise ToolInputError(f"{name} must be between {minimum} and {maximum}.")
    return value


def choice(
    payload: dict[str, Any], name: str, allowed: Iterable[str], default: str
) -> str:
    value = str(payload.get(name, default)).strip()
    if value not in set(allowed):
        raise ToolInputError(f"Unsupported {name}: {value}.")
    return value


def sequence(
    payload: dict[str, Any],
    name: str,
    *,
    alphabet: frozenset[str] = AMINO_ACIDS,
    required: bool = True,
    max_length: int = 20_000,
) -> str:
    value = re.sub(
        r"\s+", "", text(payload, name, required=required, max_length=max_length)
    ).upper()
    if value and set(value) - alphabet:
        invalid = "".join(sorted(set(value) - alphabet))
        raise ToolInputError(
            f"{name} contains unsupported sequence characters: {invalid}."
        )
    return value


def lines(
    payload: dict[str, Any], name: str, *, required: bool = True, maximum: int = 100
) -> list[str]:
    raw = text(payload, name, required=required)
    values = [line.strip() for line in raw.splitlines() if line.strip()]
    if not values:
        if required:
            raise ToolInputError(f"{name} must contain at least one value.")
        return []
    if len(values) > maximum:
        raise ToolInputError(f"{name} accepts at most {maximum} lines.")
    return values


def sequence_list(
    payload: dict[str, Any],
    name: str,
    *,
    alphabet: frozenset[str] = AMINO_ACIDS,
    required: bool = True,
    max_length: int = 20_000,
    max_chains: int = 50,
) -> list[str]:
    """A single field holding one or more `:`-separated chains (a multimer)."""

    raw = text(payload, name, required=required, max_length=max_length)
    if not raw:
        return []
    chains = []
    for part in raw.split(":"):
        chain = re.sub(r"\s+", "", part).upper()
        if not chain:
            raise ToolInputError(f"{name} has an empty chain between ':' separators.")
        if set(chain) - alphabet:
            invalid = "".join(sorted(set(chain) - alphabet))
            raise ToolInputError(
                f"{name} contains unsupported sequence characters: {invalid}."
            )
        chains.append(chain)
    if len(chains) > max_chains:
        raise ToolInputError(f"{name} accepts at most {max_chains} chains.")
    return chains


def json_object(
    payload: dict[str, Any], name: str, *, max_length: int = 20_000
) -> dict[str, Any]:
    raw = text(payload, name, required=False, max_length=max_length)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"{name} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ToolInputError(f"{name} must be a JSON object.")
    return value


def json_list(
    payload: dict[str, Any], name: str, *, max_length: int = 20_000, maximum: int = 200
) -> list[Any]:
    raw = text(payload, name, required=False, max_length=max_length)
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"{name} must be valid JSON.") from exc
    if not isinstance(value, list):
        raise ToolInputError(f"{name} must be a JSON array.")
    if len(value) > maximum:
        raise ToolInputError(f"{name} accepts at most {maximum} entries.")
    return value


@dataclass
class Modification:
    """One CCD-code substitution at a 1-indexed position in a polymer chain."""

    position: int
    residue: str


@dataclass
class MoleculeBox:
    """One validated entity from a shared `molecule_builder` field."""

    kind: str
    # `chain` remains the convenient primary identifier used by adapters that
    # only support one copy. `ids` preserves native multi-copy entity IDs.
    chain: str
    ids: list[str] = dataclass_field(default_factory=list)
    count: int = 1
    sequence: str = ""
    ligand: str = ""
    ion: str = ""
    cyclic: bool = False
    modifications: list[Modification] = dataclass_field(default_factory=list)
    msa: Any = None
    paired_msa_path: str = ""
    unpaired_msa_path: str = ""
    templates_path: str = ""


def _molecule_ids(molecule: dict[str, Any], default: str) -> list[str]:
    if "id" not in molecule:
        return [default]
    raw = molecule.get("id")
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        values = [
            item.strip() for item in raw if isinstance(item, str) and item.strip()
        ]
        if len(values) != len(raw):
            raise ToolInputError(f'Molecule "{default}" has an invalid entity ID.')
    elif raw in (None, ""):
        values = []
    else:
        raise ToolInputError(
            f'Molecule "{default}" ID must be text or a list of text IDs.'
        )
    if len(values) > 64:
        raise ToolInputError(f'Molecule "{default}" accepts at most 64 entity IDs.')
    for value in values:
        if len(value) > 20 or any(character.isspace() for character in value):
            raise ToolInputError(
                f'Molecule "{default}" entity IDs must be 1-20 characters without spaces.'
            )
    if len(set(values)) != len(values):
        raise ToolInputError(f'Molecule "{default}" has duplicated entity IDs.')
    return values


def _molecule_path(molecule: dict[str, Any], key: str, label: str) -> str:
    value = molecule.get(key, "")
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ToolInputError(f"{label} must be text.")
    value = value.strip()
    if len(value) > 4_096:
        raise ToolInputError(f"{label} is too long (maximum 4096 characters).")
    return value


def molecule_boxes(
    payload: dict[str, Any],
    name: str = "sequence_molecules",
    *,
    allowed_kinds: frozenset[str] = frozenset({"protein", "dna", "rna", "ligand"}),
    maximum: int = 26,
    allow_ids: bool = False,
    allow_count: bool = False,
    require_ids: bool = False,
    id_count_matches: bool = False,
    modification_position_base: int = 1,
) -> list[MoleculeBox]:
    """Parse and validate a `molecule_builder` field: the JSON list of boxes
    the shared JS widget builds (see tool.js), one per chain/ligand/ion.

    Common to every adapter that offers this field; a tool converts the
    result into its own request format (JSON entities, YAML rows, FASTA
    records, ...) and handles cyclic/ligand/ion support (or the lack of it)
    itself, since those vary per tool.
    """

    molecules = json_list(payload, name, maximum=maximum)
    if not molecules:
        raise ToolInputError("At least one molecule is required.")

    boxes: list[MoleculeBox] = []
    used_ids: set[str] = set()
    for offset, molecule in enumerate(molecules):
        if not isinstance(molecule, dict):
            raise ToolInputError("Each molecule must be a JSON object.")
        kind = molecule.get("type")
        if kind not in allowed_kinds:
            raise ToolInputError(
                'Each molecule needs a "type" of '
                + ", ".join(sorted(allowed_kinds))
                + "."
            )
        automatic_chain = string.ascii_uppercase[offset]
        ids = (
            _molecule_ids(molecule, automatic_chain)
            if allow_ids
            else [automatic_chain]
        )
        if require_ids and not ids:
            raise ToolInputError(f'Molecule "{automatic_chain}" needs an entity ID.')
        chain = ids[0] if ids else automatic_chain
        duplicated_ids = used_ids.intersection(ids)
        if duplicated_ids:
            duplicated = ", ".join(sorted(duplicated_ids))
            raise ToolInputError(
                f"Entity IDs must be unique; duplicated: {duplicated}."
            )
        used_ids.update(ids)
        raw_count = (
            molecule.get("count", len(ids) or 1) if allow_count else len(ids) or 1
        )
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ToolInputError(f'Molecule "{chain}" count must be an integer.') from exc
        if not 1 <= count <= 64:
            raise ToolInputError(
                f'Molecule "{chain}" count must be between 1 and 64.'
            )
        if id_count_matches and ids and len(ids) != count:
            raise ToolInputError(
                f'Molecule "{chain}" has count {count}, but {len(ids)} entity ID(s).'
            )

        common = {
            "kind": kind,
            "chain": chain,
            "ids": ids,
            "count": count,
            "msa": molecule.get("msa"),
            "paired_msa_path": _molecule_path(
                molecule, "paired_msa_path", f'Molecule "{chain}" paired MSA path'
            ),
            "unpaired_msa_path": _molecule_path(
                molecule, "unpaired_msa_path", f'Molecule "{chain}" unpaired MSA path'
            ),
            "templates_path": _molecule_path(
                molecule, "templates_path", f'Molecule "{chain}" templates path'
            ),
        }

        if kind == "ligand":
            ligand = molecule.get("ligand")
            if not isinstance(ligand, str) or not ligand.strip():
                raise ToolInputError(
                    f'Molecule "{chain}" needs a SMILES string or a CCD_ code.'
                )
            boxes.append(MoleculeBox(**common, ligand=ligand.strip()))
            continue

        if kind == "ion":
            ion = molecule.get("ion")
            if not isinstance(ion, str) or not ion.strip():
                raise ToolInputError(f'Molecule "{chain}" needs an ion code.')
            boxes.append(MoleculeBox(**common, ion=ion.strip().upper()))
            continue

        raw_seq = molecule.get("sequence")
        if not isinstance(raw_seq, str) or not raw_seq.strip():
            raise ToolInputError(f'Molecule "{chain}" needs a sequence.')
        seq = "".join(raw_seq.split()).upper()
        alphabet = AMINO_ACIDS if kind == "protein" else DNA_BASES
        if set(seq) - alphabet:
            invalid = "".join(sorted(set(seq) - alphabet))
            raise ToolInputError(
                f'Molecule "{chain}" contains unsupported sequence characters: {invalid}.'
            )

        raw_modifications = molecule.get("modifications") or []
        if not isinstance(raw_modifications, list):
            raise ToolInputError(f'Molecule "{chain}" modifications must be a list.')
        if len(raw_modifications) > 50:
            raise ToolInputError(
                f'Molecule "{chain}" accepts at most 50 modifications.'
            )
        modifications: list[Modification] = []
        for mod in raw_modifications:
            if not isinstance(mod, dict):
                raise ToolInputError(f'Molecule "{chain}" has an invalid modification.')
            position = mod.get("position")
            residue = mod.get("residue")
            if position in (None, "") and not residue:
                continue
            try:
                position = int(position)
            except (TypeError, ValueError) as exc:
                raise ToolInputError(
                    f'Molecule "{chain}" has a modification with a non-numeric position.'
                ) from exc
            last_position = len(seq) - 1 + modification_position_base
            if not modification_position_base <= position <= last_position:
                raise ToolInputError(
                    f'Molecule "{chain}" modification position {position} is out of '
                    f"range for a sequence of length {len(seq)} using "
                    f"{modification_position_base}-based positions."
                )
            if not isinstance(residue, str) or len(residue.strip()) < 2:
                raise ToolInputError(
                    f'Molecule "{chain}" modification residue codes need at least '
                    "2 characters."
                )
            modifications.append(
                Modification(position=position, residue=residue.strip().upper())
            )

        boxes.append(
            MoleculeBox(
                **common,
                sequence=seq,
                cyclic=bool(molecule.get("cyclic")),
                modifications=modifications,
            )
        )

    return boxes


# What an `input_mode` selector offers: the tool's own document typed in or
# uploaded, or the parameter boxes this form collects and converts.
INPUT_MODES = frozenset({"parameters", "text", "upload"})


def document_input(
    payload: dict[str, Any],
    field: str,
    *,
    from_boxes: Callable[[dict[str, Any]], str],
    upload_field: str = "inputs_file",
    max_length: int = 1_000_000,
) -> dict[str, Any]:
    """One input document, however the form was asked to collect it.

    A tool whose real input is a whole JSON or YAML document offers three ways
    of arriving at one: typing it, uploading it, or filling in the molecule
    boxes and letting `from_boxes` write it. This resolves all three down to
    `payload[field]`, so the adapter's own parser has one thing to read and
    every mode goes through the same validation.

    A payload that names no mode is read as `text` when it carries a document
    already, which is what an API client posting one means, and as the boxes
    otherwise. The web form always says which mode it is in.
    """

    payload = dict(payload)
    mode = str(payload.get("input_mode") or "")
    if not mode:
        mode = "text" if str(payload.get(field) or "").strip() else "parameters"
    if mode not in INPUT_MODES:
        raise ToolInputError("input_mode must be parameters, text or upload.")

    if mode == "upload":
        payload[field] = text(payload, upload_field, max_length=max_length)
    elif mode == "parameters":
        payload[field] = from_boxes(payload)
    # An upload from an earlier submission must never override what the mode
    # in force actually asked for.
    payload[upload_field] = ""
    return payload


def safe_name(
    payload: dict[str, Any], name: str = "job_name", default: str = "athanor-job"
) -> str:
    value = str(payload.get(name, default)).strip() or default
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not value:
        raise ToolInputError(f"{name} must contain at least one letter or number.")
    return value[:80]


@dataclass
class Option:
    """One choice offered by a `select` field."""

    value: str
    label: str


# The molecule types shown by a `molecule_builder` field's type dropdown, for
# tools whose molecule API is limited to these four. Pass a tool-specific list
# instead (e.g. with an "ion" option added) via the field's own `options=`.
STANDARD_MOLECULE_TYPES = [
    Option("protein", "Protein"),
    Option("ligand", "Ligand"),
    Option("dna", "DNA"),
    Option("rna", "RNA"),
]


@dataclass
class Field:
    """A single form input: rendered by the template, validated by run()."""

    name: str
    label: str
    kind: str = "text"
    default: Any = ""
    required: bool = False
    help: str = ""
    options: list[Option] = dataclass_field(default_factory=list)
    # Rendered as HTML input attributes; omitted from the markup when None.
    rows: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | str | None = None  # Numeric step or HTML's "any".
    group: str = ""
    input_modes: str = ""
    # Comma-separated optional controls rendered inside molecule-builder boxes.
    # An empty value preserves the original cyclic/modification-only widget.
    molecule_features: str = ""
    help_note: str = ""
    maxlength: int | None = None
    # kind == "file" only: a comma-separated `accept` filter for the file
    # picker, e.g. ".pdb,.cif,.ent". The uploaded file's text is read
    # client-side and submitted under this field's name, so run() still
    # receives plain text exactly as it did when the field was a textarea.
    accept: str = ""
    # Which of Spec.tasks this field belongs to, e.g. "sequence" on a tool
    # offering sequence/list/molecules input modes. Comma-separate multiple
    # tasks, e.g. "list,molecules", to show one field on several but not all
    # of them. Empty means the field is shown (and submitted) no matter which
    # task is selected. Meaningless when Spec.tasks is empty.
    task: str = ""
