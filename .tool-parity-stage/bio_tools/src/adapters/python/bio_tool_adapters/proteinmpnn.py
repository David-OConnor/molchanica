"""ProteinMPNN sequence design, scoring and probabilities.

Native options follow dauparas/ProteinMPNN protein_mpnn_run.py.
Auxiliary JSON uses 1-based parsed positions, as in the official helpers.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

import bio_tools

from . import (
    ToolExecutionError,
    ToolInputError,
    ToolUnavailable,
    catalog_spec,
    preset_payload,
    readable_files,
    run_command,
    tool_fields,
    tool_python,
    tool_tasks,
    torch_device,
)
from .field_processing import (
    boolean,
    choice,
    safe_name,
    text,
)
from .status_check import (
    CheckResult,
    ToolStatus,
    probe_command,
    require_gpu_status,
)

# ProteinMPNN's alphabet order, used by --bias_by_res_jsonl's per-residue
# 21-column matrices.
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

SPEC = catalog_spec(
    "proteinmpnn",
    fields=tool_fields("proteinmpnn"),
    tasks=tool_tasks("proteinmpnn"),
)

WEB_MANAGED_FIELDS = {
    "out_folder": "The service creates a separate output directory for each job and archives the results.",
    "path_to_model_weights": "The model and soluble/CA-only controls select an installed official checkpoint.",
    "jsonl_path": "This page accepts one structure per job; folder-based official examples are split into presets per PDB.",
    "chain_id_jsonl": "pdb_path_chains supplies the designed chains; the other parsed chains are fixed context.",
}


def _looks_like_mmcif(content: str) -> bool:
    """An mmCIF file's first non-blank, non-comment line is `data_<name>`."""

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.lower().startswith("data_")
    return False


def _find_cif_loop(
    lines: list[str], prefix: str
) -> tuple[list[str], list[list[str]]] | None:
    """The first `loop_` block whose columns start with `prefix` (e.g.
    "_atom_site."), as (column names with the prefix stripped, data rows)."""

    index = 0
    total = len(lines)
    while index < total:
        if lines[index].strip() != "loop_":
            index += 1
            continue
        cursor = index + 1
        columns: list[str] = []
        while cursor < total and lines[cursor].strip().startswith(prefix):
            columns.append(lines[cursor].strip()[len(prefix) :])
            cursor += 1
        if not columns:
            index = cursor
            continue
        values: list[str] = []
        while cursor < total:
            row_line = lines[cursor].strip()
            if not row_line or row_line.startswith("#"):
                cursor += 1
                continue
            if row_line in ("loop_", "stop_") or row_line.startswith(
                ("_", "data_", "save_")
            ):
                break
            try:
                values.extend(shlex.split(row_line, comments=True))
            except ValueError as exc:
                raise ToolInputError(
                    "pdb_path: malformed quoted value in the mmCIF atom loop."
                ) from exc
            cursor += 1
        if len(values) % len(columns):
            raise ToolInputError("pdb_path: incomplete row in the mmCIF atom loop.")
        rows = [
            values[i : i + len(columns)] for i in range(0, len(values), len(columns))
        ]
        return columns, rows
    return None


# ProteinMPNN's own PDB reader (parse_PDB_biounits in protein_mpnn_utils.py)
# only understands legacy fixed-column PDB text, so an mmCIF upload is
# converted here first. It reads exactly six fields at fixed offsets --
# record type, atom name, residue name, chain, residue number, and x/y/z --
# and ignores everything else, which is all this needs to reproduce.
_MMCIF_ATOM_SITE_REQUIRED = {
    "group_PDB",
    "label_atom_id",
    "label_comp_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
}


def _mmcif_to_pdb(content: str) -> str:
    """Convert an mmCIF's atom_site loop into fixed-column legacy PDB ATOM
    and HETATM records."""

    found = _find_cif_loop(content.splitlines(), "_atom_site.")
    if found is None:
        raise ToolInputError(
            "pdb_path: could not find an _atom_site loop in the mmCIF file."
        )
    columns, rows = found
    positions = {name: index for index, name in enumerate(columns)}
    missing = _MMCIF_ATOM_SITE_REQUIRED - positions.keys()
    if missing:
        raise ToolInputError(
            "pdb_path: mmCIF atom_site loop is missing required columns: "
            + ", ".join(sorted(missing))
            + "."
        )
    # Preserve author numbering, which the ProteinMPNN parser expands into
    # chain-relative positions, including gaps and insertion-code residues.
    chain_col = "auth_asym_id" if "auth_asym_id" in positions else "label_asym_id"
    resnum_col = "auth_seq_id" if "auth_seq_id" in positions else "label_seq_id"
    atom_col = "auth_atom_id" if "auth_atom_id" in positions else "label_atom_id"
    resname_col = "auth_comp_id" if "auth_comp_id" in positions else "label_comp_id"
    icode_col = "pdbx_PDB_ins_code" if "pdbx_PDB_ins_code" in positions else None
    if chain_col not in positions or resnum_col not in positions:
        raise ToolInputError("pdb_path: mmCIF requires chain IDs and residue numbers.")

    def field(row: list[str], name: str) -> str:
        value = row[positions[name]]
        return "" if value in (".", "?") else value

    pdb_lines: list[str] = []
    first_model = None
    for row in rows:
        if "pdbx_PDB_model_num" in positions:
            model = field(row, "pdbx_PDB_model_num")
            if first_model is None:
                first_model = model
            if model != first_model:
                continue
        chain = field(row, chain_col)
        if len(chain) != 1:
            raise ToolInputError(
                f'pdb_path: mmCIF chain id "{chain}" is not a single character; legacy PDB '
                "chain ids (what the design tool reads) can only hold one."
            )
        try:
            resnum = int(field(row, resnum_col))
            x = float(field(row, "Cartn_x"))
            y = float(field(row, "Cartn_y"))
            z = float(field(row, "Cartn_z"))
        except ValueError as exc:
            raise ToolInputError(
                "pdb_path: mmCIF has a missing or invalid author residue number or coordinate."
            ) from exc
        if not -999 <= resnum <= 9999 or any(
            not math.isfinite(value) or len(f"{value:.3f}") > 8 for value in (x, y, z)
        ):
            raise ToolInputError(
                "pdb_path: mmCIF residue numbers or coordinates do not fit legacy PDB columns."
            )
        icode = field(row, icode_col) if icode_col else ""
        if len(icode) > 1 or icode and not icode.isascii():
            raise ToolInputError(
                "pdb_path: mmCIF insertion codes must fit one PDB character."
            )
        record = "HETATM" if field(row, "group_PDB") == "HETATM" else "ATOM  "
        atom_name = field(row, atom_col)[:4]
        resname = field(row, resname_col)[:3]
        serial = len(pdb_lines) + 1
        pdb_lines.append(
            f"{record}{serial % 100000:>5} {atom_name:<4} {resname:>3} {chain}{resnum:>4}{icode or ' '}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        )

    if not pdb_lines:
        raise ToolInputError("pdb_path: mmCIF file had no usable ATOM/HETATM records.")
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


def _input_text(payload: dict[str, Any], field: str, *, required: bool = False) -> str:
    value = text(payload, field, required=required, max_length=20_000_000)
    try:
        return bio_tools.catalog_input_text("proteinmpnn", value)
    except ValueError as exc:
        raise ToolInputError(f"{field}: {exc}") from exc


def _json_input(payload: dict[str, Any], field: str) -> Any:
    raw = payload.get(field)
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        value = raw
    else:
        try:
            value = json.loads(_input_text(payload, field))
        except (ValueError, TypeError) as exc:
            raise ToolInputError(
                f"{field} must contain one valid JSON object or array."
            ) from exc
    if not isinstance(value, (dict, list)):
        raise ToolInputError(f"{field} must contain a JSON object or array.")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ToolInputError(f"{field} must contain finite numbers.") from exc
    if isinstance(value, bool) or not math.isfinite(result):
        raise ToolInputError(f"{field} must contain finite numbers.")
    if (
        minimum is not None
        and result < minimum
        or maximum is not None
        and result > maximum
    ):
        raise ToolInputError(f"{field}: value {result} is out of range.")
    return result


def _count(
    payload: dict[str, Any], field: str, default: int, minimum: int, maximum: int
) -> int:
    value = _number(
        payload.get(field, default), field, minimum=minimum, maximum=maximum
    )
    if not value.is_integer():
        raise ToolInputError(f"{field} must be an integer.")
    return int(value)


def _residue_slots(pdb: str) -> dict[str, list[str | None]]:
    """Match parse_PDB_biounits: sorted numbers/insertions, with slots for gaps.

    The upstream parser treats only ATOM records and HETATM MSE as protein.
    PDB author numbers are retained for the older sparse constraint aliases.
    """
    residues: dict[str, dict[int, set[str]]] = {}
    for line in pdb.splitlines():
        if not (
            line.startswith("ATOM")
            or line.startswith("HETATM")
            and line[17:20] == "MSE"
        ):
            continue
        chain = line[21:22]
        if len(chain) != 1 or not chain.isascii() or not chain.isalnum():
            raise ToolInputError(
                "pdb_path: protein chains must have one ASCII letter or digit as their ID."
            )
        try:
            number = int(line[22:26])
            insertion = line[26:27].strip()
            if insertion and not insertion.isalpha():
                raise ValueError("invalid insertion code")
            if not all(
                math.isfinite(float(line[start : start + 8])) for start in (30, 38, 46)
            ):
                raise ValueError("non-finite coordinates")
        except ValueError as exc:
            raise ToolInputError(
                "pdb_path: invalid residue number, insertion code or atom coordinates."
            ) from exc
        residues.setdefault(chain, {}).setdefault(number, set()).add(insertion)
    if not residues:
        raise ToolInputError(
            "pdb_path: no protein ATOM records or MSE residues were found."
        )
    slots = {}
    for chain, numbers in residues.items():
        if max(numbers) - min(numbers) + len(numbers) > 400_000:
            raise ToolInputError(
                "pdb_path: residue numbering creates too many empty sequence positions."
            )
        slots[chain] = []
        for number in range(min(numbers), max(numbers) + 1):
            if number in numbers:
                slots[chain].extend(
                    f"{number}{insertion}" for insertion in sorted(numbers[number])
                )
            else:
                slots[chain].append(None)
    return slots


def _chains(payload: dict[str, Any], slots: dict[str, list[str | None]]) -> list[str]:
    raw = text(payload, "pdb_path_chains", required=False, max_length=200).replace(
        ",", " "
    )
    chains = raw.split() if raw else sorted(slots)
    if len(chains) != len(set(chains)) or set(chains) - slots.keys():
        raise ToolInputError(
            "pdb_path_chains must name distinct protein chains present in the structure."
        )
    return chains


def _chain_dict(value: Any, field: str, slots: dict[str, list[str | None]]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ToolInputError(
            f"{field} must be a chain dictionary, optionally keyed by structure name."
        )
    # Native helper files add an outer dictionary keyed by the PDB filename stem.
    if len(value) == 1 and isinstance(next(iter(value.values())), dict):
        key, inner = next(iter(value.items()))
        if key not in slots and set(inner) <= slots.keys():
            value = inner
    if set(value) - slots.keys():
        raise ToolInputError(
            f"{field} contains unknown chains or more than one target structure."
        )
    return value


def _positions(
    value: Any, field: str, chain: str, slots: dict[str, list[str | None]]
) -> list[int]:
    if not isinstance(value, list):
        raise ToolInputError(f"{field}: positions for chain {chain} must be an array.")
    positions = []
    for pos in value:
        if (
            isinstance(pos, bool)
            or not isinstance(pos, int)
            or not 1 <= pos <= len(slots[chain])
        ):
            raise ToolInputError(
                f"{field}: chain {chain} positions must be integers from 1 to {len(slots[chain])}."
            )
        positions.append(pos)
    if len(positions) != len(set(positions)):
        raise ToolInputError(f"{field}: repeated positions on chain {chain}.")
    return positions


def _amino_acids(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolInputError(f"{field} must contain amino-acid letters.")
    letters = "".join(value.upper().split())
    if set(letters) - set(_ALPHABET):
        raise ToolInputError(f"{field}: use one-letter codes from {_ALPHABET}.")
    return letters


def _matrix(
    value: Any, field: str, length: int, *, probabilities: bool = False
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != length:
        raise ToolInputError(
            f"{field} must have {length} rows, one per parsed position."
        )
    matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != len(_ALPHABET):
            raise ToolInputError(f"{field} must have 21 columns in {_ALPHABET} order.")
        row = [_number(v, field, minimum=0 if probabilities else None) for v in row]
        if probabilities and not math.isclose(sum(row), 1, rel_tol=1e-4, abs_tol=1e-5):
            raise ToolInputError(f"{field}: each probability row must sum to 1.")
        matrix.append(row)
    return matrix


def _legacy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    for old, new in [
        ("pdb", "pdb_path"),
        ("chains", "pdb_path_chains"),
        ("num_sequences", "num_seq_per_target"),
        ("temperature", "sampling_temp"),
        ("omit_amino_acids", "omit_AAs"),
    ]:
        if old in payload:
            payload.setdefault(new, payload[old])
    if "model_type" in payload:
        kind = choice(
            payload, "model_type", {"proteinmpnn", "solublempnn"}, "proteinmpnn"
        )
        payload.setdefault("use_soluble_model", kind == "solublempnn")
    if payload.get("bias_amino_acids") and "bias_AA_jsonl" not in payload:
        bias = {}
        for pair in str(payload["bias_amino_acids"]).split(","):
            aa, sep, value = pair.strip().partition(":")
            if not sep:
                raise ToolInputError(
                    "bias_amino_acids entries must look like A:-1.1,F:0.7."
                )
            bias[aa.strip().upper()] = _number(value, "bias_amino_acids")
        payload["bias_AA_jsonl"] = bias
    return payload


def _legacy_constraints(
    payload: dict[str, Any], slots: dict[str, list[str | None]], chains: list[str]
) -> None:
    def position(chain: str, author: str, field: str) -> int:
        if chain not in chains or author not in slots[chain]:
            raise ToolInputError(
                f"{field}: {chain}{author} is not a residue on a designed chain."
            )
        return slots[chain].index(author) + 1

    if payload.get("designed_residues") and not payload.get("fixed_positions_jsonl"):
        fixed = {c: [] for c in chains}
        seen = set()
        for line in str(payload["designed_residues"]).splitlines():
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            chain, *residues = parts
            if chain in seen or chain not in chains or not residues:
                raise ToolInputError(
                    "designed_residues requires one line per designed chain followed by PDB residue numbers."
                )
            seen.add(chain)
            allowed = {position(chain, r, "designed_residues") for r in residues}
            fixed[chain] = [
                i for i in range(1, len(slots[chain]) + 1) if i not in allowed
            ]
        payload["fixed_positions_jsonl"] = fixed
    for old, new in [
        ("bias_amino_acids_per_residue", "bias_by_res_jsonl"),
        ("omit_amino_acids_per_residue", "omit_AA_jsonl"),
    ]:
        if not payload.get(old) or payload.get(new):
            continue
        sparse = _json_input(payload, old)
        if not isinstance(sparse, dict):
            raise ToolInputError(
                f"{old} must be an object keyed by chain and PDB residue number."
            )
        values = {
            c: [[0.0] * 21 for _ in slots[c]] if new == "bias_by_res_jsonl" else []
            for c in chains
        }
        for key, value in sparse.items():
            match = re.fullmatch(r"([A-Za-z0-9])(-?\d+[A-Za-z]?)", key)
            if not match:
                raise ToolInputError(f"{old}: keys must look like A12 or A12B.")
            chain, author = match.groups()
            pos = position(chain, author, old)
            if new == "bias_by_res_jsonl":
                if not isinstance(value, dict):
                    raise ToolInputError(
                        f"{old}: each position needs an amino-acid bias object."
                    )
                for aa, bias in value.items():
                    aa = _amino_acids(aa, old)
                    if len(aa) != 1:
                        raise ToolInputError(
                            f"{old}: each bias must name one amino acid."
                        )
                    values[chain][pos - 1][_ALPHABET.index(aa)] = _number(bias, old)
            else:
                values[chain].append([[pos], _amino_acids(value, old)])
        payload[new] = values


def _constraints(
    payload: dict[str, Any], slots: dict[str, list[str | None]], chains: list[str]
) -> dict[str, Any]:
    documents = {}
    fixed_field = "fixed_positions_jsonl"
    fixed = _chain_dict(_json_input(payload, fixed_field), fixed_field, slots)
    if fixed:
        documents[fixed_field] = {
            c: _positions(fixed.get(c, []), fixed_field, c, slots) for c in slots
        }
    tied_field = "tied_positions_jsonl"
    tied = _json_input(payload, tied_field)
    if isinstance(tied, dict) and len(tied) == 1:
        tied = next(iter(tied.values()))
    if boolean(payload, "homo_oligomer"):
        if tied:
            raise ToolInputError("Use homo_oligomer or tied_positions_jsonl, not both.")
        if len(chains) < 2 or len({len(slots[c]) for c in chains}) != 1:
            raise ToolInputError(
                "homo_oligomer requires at least two designed chains of equal parsed length."
            )
        tied = [{c: [i] for c in chains} for i in range(1, len(slots[chains[0]]) + 1)]
    if tied:
        if not isinstance(tied, list):
            raise ToolInputError(
                "tied_positions_jsonl must contain an array of position groups."
            )
        seen = set()
        for group in tied:
            if not isinstance(group, dict) or not group or set(group) - slots.keys():
                raise ToolInputError(
                    "tied_positions_jsonl: each group must map known chains to positions."
                )
            members = []
            for c, value in group.items():
                weighted = (
                    isinstance(value, list) and value and isinstance(value[0], list)
                )
                positions = _positions(
                    value[0] if weighted else value, tied_field, c, slots
                )
                if weighted:
                    if (
                        len(value) != 2
                        or not isinstance(value[1], list)
                        or len(value[1]) != len(positions)
                    ):
                        raise ToolInputError(
                            "tied_positions_jsonl: each weighted group needs matching position and weight lists."
                        )
                    value[1] = [_number(w, tied_field) for w in value[1]]
                members.extend((c, p) for p in positions)
            if len(members) < 2 or set(members) & seen:
                raise ToolInputError(
                    "tied_positions_jsonl: groups need at least two distinct positions and cannot overlap."
                )
            seen.update(members)
        documents[tied_field] = tied
    bias_field = "bias_AA_jsonl"
    bias = _json_input(payload, bias_field)
    if bias:
        if not isinstance(bias, dict):
            raise ToolInputError("bias_AA_jsonl must be an amino-acid bias object.")
        normalized = {}
        for aa, value in bias.items():
            aa = _amino_acids(aa, bias_field)
            if len(aa) != 1:
                raise ToolInputError(
                    "bias_AA_jsonl keys must be single amino-acid codes."
                )
            normalized[aa] = _number(value, bias_field)
        documents[bias_field] = normalized
    omit_field = "omit_AA_jsonl"
    omit = _chain_dict(_json_input(payload, omit_field), omit_field, slots)
    global_omit = set(_amino_acids(payload.get("omit_AAs", "X"), "omit_AAs"))
    if set(_ALPHABET) <= global_omit:
        raise ToolInputError("omit_AAs must leave at least one amino acid available.")
    if omit:
        normalized = {c: [] for c in slots}
        for c, entries in omit.items():
            if not isinstance(entries, list):
                raise ToolInputError(
                    "omit_AA_jsonl: each chain needs an array of [positions, amino acids] pairs."
                )
            excluded = {}
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ToolInputError(
                        "omit_AA_jsonl entries must be [positions, amino acids] pairs."
                    )
                positions = _positions(entry[0], omit_field, c, slots)
                letters = _amino_acids(entry[1], omit_field)
                for p in positions:
                    excluded.setdefault(p, set(global_omit)).update(letters)
                    if set(_ALPHABET) <= excluded[p]:
                        raise ToolInputError(
                            f"omit_AA_jsonl: no amino acids remain at {c}{p}."
                        )
                normalized[c].append([positions, letters])
        documents[omit_field] = normalized
    matrix_field = "bias_by_res_jsonl"
    matrices = _chain_dict(_json_input(payload, matrix_field), matrix_field, slots)
    if matrices:
        documents[matrix_field] = {
            c: (
                _matrix(matrices[c], matrix_field, len(slots[c]))
                if c in matrices
                else [[0.0] * 21 for _ in slots[c]]
            )
            for c in slots
        }
    pssm_field = "pssm_jsonl"
    pssm = _chain_dict(_json_input(payload, pssm_field), pssm_field, slots)
    if pssm:
        normalized = {c: {} for c in slots}
        for c, entry in pssm.items():
            if not isinstance(entry, dict):
                raise ToolInputError(
                    "pssm_jsonl: each chain must contain a PSSM object."
                )
            if not entry:
                continue
            if set(entry) != {"pssm_coef", "pssm_bias", "pssm_log_odds"}:
                raise ToolInputError(
                    "pssm_jsonl requires pssm_coef, pssm_bias and pssm_log_odds for each guided chain."
                )
            coef = entry["pssm_coef"]
            if not isinstance(coef, list) or len(coef) != len(slots[c]):
                raise ToolInputError(
                    f"pssm_coef needs {len(slots[c])} entries for chain {c}."
                )
            normalized[c] = {
                "pssm_coef": [
                    _number(v, "pssm_coef", minimum=0, maximum=1) for v in coef
                ],
                "pssm_bias": _matrix(
                    entry["pssm_bias"], "pssm_bias", len(slots[c]), probabilities=True
                ),
                "pssm_log_odds": _matrix(
                    entry["pssm_log_odds"], "pssm_log_odds", len(slots[c])
                ),
            }
        documents[pssm_field] = normalized
    return documents


def _fasta(
    payload: dict[str, Any], chains: list[str], slots: dict[str, list[str | None]]
) -> str:
    content = _input_text(payload, "path_to_fasta")
    if not content:
        return ""
    records: list[tuple[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append((line, ""))
        elif records:
            header, seq = records[-1]
            records[-1] = (header, seq + line)
        else:
            raise ToolInputError(
                "path_to_fasta requires a >header before each sequence."
            )
    if not records:
        raise ToolInputError("path_to_fasta contains no FASTA records.")
    output = []
    expected = [len(slots[c]) for c in sorted(chains)]
    for header, seq in records:
        parts = seq.upper().split("/")
        if [len(p) for p in parts] != expected:
            raise ToolInputError(
                f"path_to_fasta: use /-separated designed chains in alphabetical order, with lengths {expected}."
            )
        for part in parts:
            if _amino_acids(part, "path_to_fasta") != part:
                raise ToolInputError(
                    "path_to_fasta: sequence lines must contain only amino-acid codes and / separators."
                )
        output.extend([header, "/".join(parts)])
    return "\n".join(output) + "\n"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    payload = preset_payload("proteinmpnn", _legacy_payload(payload))
    name = safe_name(payload, default="proteinmpnn-demo")
    task = choice(
        payload,
        "task",
        {"design", "score_only", "conditional_probs_only", "unconditional_probs_only"},
        "design",
    )
    # Accept native CLI mode flags from API callers as well as the form task.
    modes = [
        m
        for m in ("score_only", "conditional_probs_only", "unconditional_probs_only")
        if boolean(payload, m)
    ]
    if len(modes) > 1 or modes and task not in {"design", modes[0]}:
        raise ToolInputError("Choose one ProteinMPNN task.")
    if modes:
        task = modes[0]
    pdb = _input_text(payload, "pdb_path", required=True)
    if _looks_like_mmcif(pdb):
        pdb = _mmcif_to_pdb(pdb)
    # Use a single conformer; the upstream fixed-column reader otherwise
    # combines residues from all MODEL records into one design target.
    pdb = re.split(r"(?m)^ENDMDL", pdb, maxsplit=1)[0]
    slots = _residue_slots(pdb)
    chains = _chains(payload, slots)
    length = _count(payload, "max_length", 200000, 1, 200000)
    if sum(map(len, slots.values())) > length:
        raise ToolInputError(
            "pdb_path: total parsed sequence length exceeds max_length."
        )
    model = choice(
        payload,
        "model_name",
        {"v_48_002", "v_48_010", "v_48_020", "v_48_030"},
        "v_48_020",
    )
    ca_only = boolean(payload, "ca_only")
    soluble = boolean(payload, "use_soluble_model")
    if ca_only and soluble:
        raise ToolInputError("CA-only soluble-protein weights are not available.")
    if ca_only and model == "v_48_030":
        raise ToolInputError("CA-only models support v_48_002, v_48_010 and v_48_020.")
    samples = _count(payload, "num_seq_per_target", 1, 1, 1000)
    batch = _count(payload, "batch_size", 1, 1, 1000)
    if samples % batch:
        raise ToolInputError(
            "num_seq_per_target must be a multiple of batch_size; ProteinMPNN otherwise drops samples."
        )
    temps = str(payload.get("sampling_temp", "0.1")).split()
    if not temps or any(_number(t, "sampling_temp") <= 0 for t in temps):
        raise ToolInputError(
            "sampling_temp requires one or more positive temperatures separated by spaces."
        )
    seed = _count(payload, "seed", 0, 0, 2147483647)
    noise = _number(payload.get("backbone_noise", 0), "backbone_noise", minimum=0)
    _legacy_constraints(payload, slots, chains)
    documents = _constraints(payload, slots, chains)
    fasta = _fasta(payload, chains, slots)
    if fasta and task != "score_only":
        raise ToolInputError("path_to_fasta is only used in score-only mode.")
    pssm_multi = _number(
        payload.get("pssm_multi", 0), "pssm_multi", minimum=0, maximum=1
    )
    pssm_threshold = _number(payload.get("pssm_threshold", 0), "pssm_threshold")
    pssm_bias = boolean(payload, "pssm_bias_flag")
    pssm_log_odds = boolean(payload, "pssm_log_odds_flag")
    if (pssm_multi or pssm_bias or pssm_log_odds) and "pssm_jsonl" not in documents:
        raise ToolInputError("PSSM options require pssm_jsonl guidance.")
    if pssm_multi and not pssm_bias:
        raise ToolInputError("Enable pssm_bias_flag to use pssm_multi.")
    if (
        boolean(payload, "conditional_probs_only_backbone")
        and task != "conditional_probs_only"
    ):
        raise ToolInputError(
            "conditional_probs_only_backbone is only used for conditional probabilities."
        )

    runner_value = os.getenv("PROTEINMPNN_RUNNER")
    if not runner_value or not Path(runner_value).is_file():
        raise ToolUnavailable("Configure PROTEINMPNN_RUNNER with protein_mpnn_run.py.")
    runner = Path(runner_value).resolve()
    folder = (
        "ca_model_weights"
        if ca_only
        else "soluble_model_weights" if soluble else "vanilla_model_weights"
    )
    weights = runner.parent / folder
    if not (ca_only or soluble) and os.getenv("PROTEINMPNN_MODEL_PATH"):
        weights = Path(os.environ["PROTEINMPNN_MODEL_PATH"]).resolve()
    if not (weights / f"{model}.pt").is_file():
        raise ToolUnavailable(
            f"ProteinMPNN checkpoint {folder}/{model}.pt was not found."
        )
    python = tool_python("proteinmpnn", "PROTEINMPNN_PYTHON")
    with tempfile.TemporaryDirectory(prefix="bio-web-proteinmpnn-") as temporary:
        workdir = Path(temporary)
        (workdir / "input.pdb").write_text(pdb + "\n", encoding="utf-8")
        command = [
            python,
            str(runner),
            "--pdb_path",
            "input.pdb",
            "--pdb_path_chains",
            " ".join(chains),
            "--out_folder",
            "output",
            "--model_name",
            model,
            "--path_to_model_weights",
            str(weights) + os.sep,
            "--num_seq_per_target",
            str(samples),
            "--batch_size",
            str(batch),
            "--sampling_temp",
            " ".join(temps),
            "--seed",
            str(seed),
            "--backbone_noise",
            str(noise),
            "--max_length",
            str(length),
            "--omit_AAs",
            _amino_acids(payload.get("omit_AAs", "X"), "omit_AAs"),
        ]
        if ca_only:
            command.append("--ca_only")
        if soluble:
            command.append("--use_soluble_model")
        if task != "design":
            command += [f"--{task}", "1"]
        for flag in (
            "save_score",
            "save_probs",
            "conditional_probs_only_backbone",
            "suppress_print",
        ):
            command += [f"--{flag}", str(int(boolean(payload, flag)))]
        for field, document in documents.items():
            filename = field + ".json"
            # Upstream indexes auxiliary dictionaries by the PDB filename stem.
            value = document if field == "bias_AA_jsonl" else {"input": document}
            (workdir / filename).write_text(
                json.dumps(value, allow_nan=False) + "\n", encoding="utf-8"
            )
            command += [f"--{field}", filename]
        command += [
            "--pssm_multi",
            str(pssm_multi),
            "--pssm_threshold",
            str(pssm_threshold),
            "--pssm_bias_flag",
            str(int(pssm_bias)),
            "--pssm_log_odds_flag",
            str(int(pssm_log_odds)),
        ]
        if fasta:
            (workdir / "input.fa").write_text(fasta, encoding="utf-8")
            command += ["--path_to_fasta", "input.fa"]
        result = run_command(command, cwd=workdir)
        generated = readable_files(workdir / "output")
        if not generated:
            raise ToolExecutionError(
                "ProteinMPNN finished without producing result files. See the run log for details."
            )
    return {
        "status": "completed",
        "job_name": name,
        "task": task,
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    try:
        python = tool_python("proteinmpnn", "PROTEINMPNN_PYTHON")
    except ToolUnavailable as exc:
        return ToolStatus(CheckResult.NOT_INSTALLED, str(exc))
    code, output = probe_command([python, "--version"])
    if code is None:
        return ToolStatus(
            CheckResult.NOT_INSTALLED,
            f"Could not run the ProteinMPNN interpreter: {output}",
        )
    if code != 0:
        return ToolStatus(
            CheckResult.ERROR, output or f"python --version exited {code}."
        )
    runner = os.getenv("PROTEINMPNN_RUNNER")
    if not runner or not Path(runner).is_file():
        return ToolStatus(
            CheckResult.NOT_INSTALLED, "PROTEINMPNN_RUNNER is not configured."
        )
    model_path = Path(
        os.getenv("PROTEINMPNN_MODEL_PATH")
        or Path(runner).resolve().parent / "vanilla_model_weights"
    )
    if not (model_path / "v_48_020.pt").is_file():
        return ToolStatus(
            CheckResult.NOT_INSTALLED,
            "ProteinMPNN's default v_48_020.pt checkpoint was not found.",
        )
    return require_gpu_status(
        ToolStatus(CheckResult.PASS, output),
        torch_device(python),
        "ProteinMPNN",
    )
