"""LigandMPNN's official run.py and score.py interfaces, with job-local inputs."""

from __future__ import annotations

import json
import math
import os
import re
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
    text,
    tool_fields,
    tool_python,
    tool_tasks,
    torch_device,
)
from .field_processing import boolean, choice, safe_name
from .status_check import CheckResult, ToolStatus, probe_command, require_gpu_status

_MODELS = {
    "protein_mpnn": ("proteinmpnn_v_48_", ("002", "010", "020", "030"), "020", ".pt"),
    "ligand_mpnn": ("ligandmpnn_v_32_", ("005", "010", "020", "030"), "010", "_25.pt"),
    "soluble_mpnn": ("solublempnn_v_48_", ("002", "010", "020", "030"), "020", ".pt"),
    "per_residue_label_membrane_mpnn": (
        "per_residue_label_membrane_mpnn_v_48_",
        ("020",),
        "020",
        ".pt",
    ),
    "global_label_membrane_mpnn": (
        "global_label_membrane_mpnn_v_48_",
        ("020",),
        "020",
        ".pt",
    ),
}
_SC_CHECKPOINT = "ligandmpnn_sc_v_32_002_16.pt"
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_RESIDUE = re.compile(r"[A-Za-z0-9](-?\d+[A-Za-z]?)\Z")
_ALIASES = {
    "pdb": "pdb_path",
    "omit_amino_acids": "omit_AA",
    "bias_amino_acids": "bias_AA",
    "bias_amino_acids_per_residue": "bias_AA_per_residue",
    "omit_amino_acids_per_residue": "omit_AA_per_residue",
}
_CONSTRAINTS = (
    "fixed_residues",
    "redesigned_residues",
    "bias_AA_per_residue",
    "omit_AA_per_residue",
)

SPEC = catalog_spec(
    "ligandmpnn", fields=tool_fields("ligandmpnn"), tasks=tool_tasks("ligandmpnn")
)

WEB_MANAGED_FIELDS = {
    "out_folder": "The service creates a separate output directory for each job and archives the results.",
    "checkpoint_path_sc": "Side-chain packing uses the installed official ligandmpnn_sc_v_32_002_16.pt checkpoint.",
}


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolInputError(f"{field} must be a finite number.") from exc
    if (
        isinstance(value, bool)
        or not math.isfinite(result)
        or minimum is not None
        and result < minimum
    ):
        raise ToolInputError(
            f"{field} must be finite"
            + (f" and at least {minimum}." if minimum is not None else ".")
        )
    return result


def _count(payload: dict, field: str, default: int, minimum: int, maximum: int) -> int:
    value = _number(payload.get(field, default), field, minimum=minimum)
    if not value.is_integer() or value > maximum:
        raise ToolInputError(f"{field} must be an integer from {minimum} to {maximum}.")
    return int(value)


def _content(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolInputError(f"{field} must contain text or a bundled input reference.")
    if len(value) > 20_000_000:
        raise ToolInputError(f"{field} exceeds 20 million characters.")
    try:
        return bio_tools.catalog_input_text("ligandmpnn", value).strip()
    except ValueError as exc:
        raise ToolInputError(f"{field}: {exc}") from exc


def _object(value: Any, field: str) -> dict:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(_content(value, field))
        except (ValueError, TypeError) as exc:
            raise ToolInputError(f"{field} must contain a valid JSON object.") from exc
    if not isinstance(value, dict):
        raise ToolInputError(f"{field} must contain a JSON object.")
    return value


def _aa(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolInputError(f"{field} must contain one-letter amino-acid codes.")
    value = "".join(value.upper().split())
    if set(value) - set(_ALPHABET):
        raise ToolInputError(f"{field}: use amino acids from {_ALPHABET}.")
    return value


def _bias(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ToolInputError(f"{field} must map amino acids to numeric biases.")
    result = {}
    for aa, amount in value.items():
        aa = _aa(aa, field)
        if len(aa) != 1 or aa in result:
            raise ToolInputError(
                f"{field}: each bias must name one distinct amino acid."
            )
        result[aa] = _number(amount, field)
    return result


def _global_bias(payload: dict) -> str:
    value = text(payload, "bias_AA", required=False, max_length=5000)
    result = {}
    if value:
        for entry in value.split(","):
            aa, sep, amount = entry.partition(":")
            if not sep or aa.strip().upper() in result:
                raise ToolInputError(
                    "bias_AA requires distinct AA:value pairs, e.g. A:-1.0,P:2.3."
                )
            result.update(_bias({aa: amount}, "bias_AA"))
    return ",".join(f"{aa}:{amount}" for aa, amount in result.items())


def _chain_list(payload: dict, field: str) -> list[str]:
    raw = text(payload, field, required=False, max_length=200)
    chains = raw.replace(",", " ").split()
    if len(chains) != len(set(chains)) or any(
        not re.fullmatch(r"[A-Za-z0-9]", c) for c in chains
    ):
        raise ToolInputError(
            f"{field} requires distinct single-letter or digit chain IDs."
        )
    return chains


def _residues(
    pdb: str, include_zero: bool, parsed_chains: list[str]
) -> dict[str, set[str]]:
    """Inspect CA-bearing protein residues without filling numbering gaps.

    ProDy performs the final atom parsing. Match its first-model/default altloc
    and occupancy selections so residue constraints cannot silently miss targets.
    """
    residues: dict[str, set[str]] = {}
    atom_chains = set()
    for line in pdb.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21:22]
        if parsed_chains and chain not in parsed_chains:
            continue
        if line[16:17] not in (" ", "A"):
            continue
        try:
            occupancy = float(line[54:60].strip() or "0")
        except ValueError as exc:
            raise ToolInputError("pdb_path contains invalid atom occupancies.") from exc
        if not math.isfinite(occupancy):
            raise ToolInputError("pdb_path contains non-finite atom occupancies.")
        if not include_zero and occupancy <= 0:
            continue
        atom_chains.add(chain)
        if line[12:16].strip() != "CA":
            continue
        # Avoid mistaking ligand calcium for a protein alpha carbon.
        if line.startswith("HETATM") and line[17:20] not in {
            "MSE",
            "SEP",
            "TPO",
            "PTR",
            "CSO",
            "HYP",
        }:
            continue
        token = chain + line[22:26].strip() + line[26:27].strip()
        if not _RESIDUE.fullmatch(token):
            raise ToolInputError(
                "PDB protein residues need single-letter/digit chains and integer author residue numbers."
            )
        try:
            coords = [float(line[i : i + 8]) for i in (30, 38, 46)]
        except ValueError as exc:
            raise ToolInputError("pdb_path contains invalid atom coordinates.") from exc
        if not all(math.isfinite(v) for v in coords):
            raise ToolInputError("pdb_path contains non-finite coordinates.")
        residues.setdefault(chain, set()).add(token)
    if not residues:
        raise ToolInputError(
            "No protein alpha carbons remain after PDB chain/occupancy filtering. Upload a PDB file with usable protein atoms."
        )
    if set(parsed_chains) - atom_chains:
        raise ToolInputError(
            "parse_these_chains_only contains chains absent after occupancy filtering."
        )
    return residues


def _tokens(value: Any, field: str, known: set[str]) -> list[str]:
    if not isinstance(value, str):
        raise ToolInputError(f"{field} must be a space-separated residue list.")
    tokens = value.split()
    if len(tokens) != len(set(tokens)):
        raise ToolInputError(f"{field} contains repeated residues.")
    for token in tokens:
        if not _RESIDUE.fullmatch(token) or token not in known:
            raise ToolInputError(
                f"{field}: {token} is not a parsed protein residue. Use PDB IDs such as A12 or B82A."
            )
    return tokens


def _symmetry(
    payload: dict, residues: dict[str, set[str]], task: str
) -> tuple[str, str]:
    known = set().union(*residues.values())
    raw = text(payload, "symmetry_residues", required=False, max_length=100000)
    weights = (
        text(payload, "symmetry_weights", required=False, max_length=100000)
        if task == "design"
        else ""
    )
    if boolean(payload, "homo_oligomer"):
        if raw or weights:
            raise ToolInputError("Choose homo_oligomer or explicit symmetry groups.")
        ids = [{r[1:] for r in entries} for entries in residues.values()]
        if len(ids) < 2 or any(ids[0] != other for other in ids[1:]):
            raise ToolInputError(
                "homo_oligomer requires at least two parsed protein chains with matching PDB residue numbers and insertion codes."
            )
    groups = []
    seen = set()
    if raw:
        for group in raw.split("|"):
            members = _tokens(group.replace(",", " "), "symmetry_residues", known)
            if len(members) < 2 or seen.intersection(members):
                raise ToolInputError(
                    "Symmetry groups must contain at least two residues and cannot overlap."
                )
            groups.append(members)
            seen.update(members)
    parsed_weights = []
    if task == "design" and (raw or weights):
        parsed_weights = [
            [_number(w, "symmetry_weights") for w in group.split(",")]
            for group in weights.split("|")
        ]
        if [len(g) for g in groups] != [len(g) for g in parsed_weights]:
            raise ToolInputError(
                "symmetry_weights must give one weight for each residue in every symmetry group."
            )
    return "|".join(",".join(g) for g in groups), "|".join(
        ",".join(map(str, g)) for g in parsed_weights
    )


def _normalize(payload: dict) -> dict:
    payload = dict(payload)
    for old, new in _ALIASES.items():
        if old in payload:
            payload.setdefault(new, payload[old])
    if payload.get("task") in _MODELS:
        if payload.get("model_type", payload["task"]) != payload["task"]:
            raise ToolInputError("Legacy task and model_type select different models.")
        payload["model_type"] = payload["task"]
        payload["task"] = "design"
    modes = [
        m for m in ("autoregressive_score", "single_aa_score") if boolean(payload, m)
    ]
    if (
        len(modes) > 1
        or modes
        and payload.get("task", "design") not in ("design", modes[0])
    ):
        raise ToolInputError("Choose one LigandMPNN scoring task.")
    if modes:
        payload["task"] = modes[0]
    return payload


def run(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _normalize(preset_payload("ligandmpnn", _normalize(payload)))
    name = safe_name(payload, default="ligandmpnn-demo")
    task = choice(
        payload, "task", {"design", "autoregressive_score", "single_aa_score"}, "design"
    )
    model = choice(payload, "model_type", _MODELS, "protein_mpnn")
    multi = _object(payload.get("pdb_path_multi"), "pdb_path_multi")
    single = payload.get("pdb_path", "")
    if multi and single or not multi and not single:
        raise ToolInputError("Supply either pdb_path or pdb_path_multi.")
    if len(multi) > 100:
        raise ToolInputError("pdb_path_multi accepts at most 100 structures per run.")
    inputs = multi or {"input": single}
    if any(
        not isinstance(k, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", k)
        for k in inputs
    ):
        raise ToolInputError(
            "Structure names must start with a letter/digit and contain only letters, digits, dots, underscores or hyphens (80 characters maximum)."
        )
    parsed_chains = _chain_list(payload, "parse_these_chains_only")
    design_chains = _chain_list(payload, "chains_to_design")
    batch = _count(payload, "batch_size", 1, 1, 1000)
    batches = _count(payload, "number_of_batches", 1, 1, 100)
    seed = _count(payload, "seed", 0, 0, 2147483647)
    pack = boolean(payload, "pack_side_chains")
    if pack and task != "design":
        raise ToolInputError(
            "Side-chain packing is available with sequence design only."
        )
    common = {f: payload.get(f, "") for f in _CONSTRAINTS}
    per_input = {
        f: _object(payload.get(f + "_multi"), f + "_multi") for f in _CONSTRAINTS
    }
    for field, mapping in per_input.items():
        if mapping and not multi or set(mapping) - inputs.keys():
            raise ToolInputError(f"{field}_multi must use names from pdb_path_multi.")
    global_omit = _aa(payload.get("omit_AA", ""), "omit_AA") if task == "design" else ""
    if set(_ALPHABET[:20]) <= set(global_omit):
        raise ToolInputError(
            "omit_AA must leave at least one standard amino acid available."
        )
    prepared = {}
    for key, value in inputs.items():
        pdb = _content(value, "pdb_path")
        residues = _residues(
            pdb, boolean(payload, "parse_atoms_with_zero_occupancy"), parsed_chains
        )
        known = set().union(*residues.values())
        if set(design_chains) - residues.keys():
            raise ToolInputError(
                f"{key}: chains_to_design contains unparsed protein chains."
            )
        constraints = {}
        for field in _CONSTRAINTS:
            value = per_input[field].get(key, common[field])
            if field in ("fixed_residues", "redesigned_residues"):
                constraints[field] = _tokens(value, field, known)
            elif task == "design":
                obj = _object(value, field)
                for residue, entry in obj.items():
                    _tokens(residue, field, known)
                    if len(residue.split()) != 1:
                        raise ToolInputError(
                            f"{field} keys must be single residue IDs."
                        )
                constraints[field] = {
                    r: _bias(v, field) if field.startswith("bias") else _aa(v, field)
                    for r, v in obj.items()
                }
        if constraints["fixed_residues"] and constraints["redesigned_residues"]:
            raise ToolInputError(
                f"{key}: choose fixed_residues or redesigned_residues; upstream otherwise ignores fixed_residues."
            )
        if task == "design":
            for residue, omitted in constraints["omit_AA_per_residue"].items():
                if set(_ALPHABET[:20]) <= set(global_omit + omitted):
                    raise ToolInputError(
                        f"No standard amino acids remain at {residue}."
                    )
        sym, weights = _symmetry(payload, residues, task)
        if model == "per_residue_label_membrane_mpnn":
            buried = _tokens(
                payload.get("transmembrane_buried", ""), "transmembrane_buried", known
            )
            interface = _tokens(
                payload.get("transmembrane_interface", ""),
                "transmembrane_interface",
                known,
            )
            if set(buried) & set(interface):
                raise ToolInputError(
                    "Buried and interface membrane residue lists must not overlap."
                )
        prepared[key] = (pdb, constraints)

    command_flags = [
        "--model_type",
        model,
        "--batch_size",
        str(batch),
        "--number_of_batches",
        str(batches),
        "--seed",
        str(seed),
    ]
    for field in ("verbose", "parse_atoms_with_zero_occupancy", "homo_oligomer"):
        command_flags += [
            "--" + field,
            str(int(boolean(payload, field, field == "verbose"))),
        ]
    if design_chains:
        command_flags += ["--chains_to_design", ",".join(design_chains)]
    if parsed_chains:
        # score.py iterates this argument as characters; run.py splits commas.
        command_flags += [
            "--parse_these_chains_only",
            ("," if task == "design" else "").join(parsed_chains),
        ]
    if sym:
        command_flags += ["--symmetry_residues", sym]
        if task == "design":
            command_flags += ["--symmetry_weights", weights]
    ending = text(payload, "file_ending", required=False, max_length=80)
    if not re.fullmatch(r"[A-Za-z0-9_.-]*", ending):
        raise ToolInputError(
            "file_ending must contain only filename characters: letters, digits, _, . or -."
        )
    if ending:
        command_flags += ["--file_ending", ending]
    if model == "ligand_mpnn":
        for flag, default in (
            ("ligand_mpnn_use_atom_context", True),
            ("ligand_mpnn_use_side_chain_context", False),
        ):
            command_flags += ["--" + flag, str(int(boolean(payload, flag, default)))]
        cutoff = _number(
            payload.get("ligand_mpnn_cutoff_for_score", 8),
            "ligand_mpnn_cutoff_for_score",
            minimum=0.000001,
        )
        command_flags += ["--ligand_mpnn_cutoff_for_score", str(cutoff)]
    if model == "per_residue_label_membrane_mpnn":
        for flag in ("transmembrane_buried", "transmembrane_interface"):
            value = " ".join(str(payload.get(flag, "")).split())
            if value:
                command_flags += ["--" + flag, value]
    if model == "global_label_membrane_mpnn":
        command_flags += [
            "--global_transmembrane_label",
            choice(payload, "global_transmembrane_label", {"0", "1"}, "0"),
        ]
    if task == "design":
        temperature = _number(
            payload.get("temperature", 0.1), "temperature", minimum=0.000001
        )
        command_flags += ["--temperature", str(temperature)]
        bias = _global_bias(payload)
        if bias:
            command_flags += ["--bias_AA", bias]
        if global_omit:
            command_flags += ["--omit_AA", global_omit]
        for flag in ("save_stats", "zero_indexed"):
            # Upstream zero_indexed is declared str but tested for truthiness:
            # passing the string "0" would accidentally enable zero indexing.
            if boolean(payload, flag):
                command_flags += ["--" + flag, "1"]
        separator = (
            text(payload, "fasta_seq_separation", required=True, max_length=10)
            if "fasta_seq_separation" in payload
            else ":"
        )
        if any(c.isspace() or c == ">" for c in separator):
            raise ToolInputError(
                "fasta_seq_separation cannot contain whitespace or FASTA headers."
            )
        command_flags += ["--fasta_seq_separation", separator]
        if pack:
            command_flags += ["--pack_side_chains", "1"]
            for flag, default, maximum in (
                ("number_of_packs_per_design", 4, 32),
                ("sc_num_denoising_steps", 3, 100),
                ("sc_num_samples", 16, 1000),
            ):
                command_flags += [
                    "--" + flag,
                    str(_count(payload, flag, default, 1, maximum)),
                ]
            for flag in (
                "pack_with_ligand_context",
                "repack_everything",
                "force_hetatm",
            ):
                command_flags += [
                    "--" + flag,
                    str(
                        int(boolean(payload, flag, flag == "pack_with_ligand_context"))
                    ),
                ]
            suffix = str(payload.get("packed_suffix", "_packed"))
            if len(suffix) > 80 or not re.fullmatch(r"[A-Za-z0-9_.-]*", suffix):
                raise ToolInputError(
                    "packed_suffix must contain only letters, digits, _, . or -."
                )
            command_flags += ["--packed_suffix", suffix]
    else:
        command_flags += [
            "--autoregressive_score",
            str(int(task == "autoregressive_score")),
            "--single_aa_score",
            str(int(task == "single_aa_score")),
            "--use_sequence",
            str(int(boolean(payload, "use_sequence", True))),
        ]

    runner = Path(os.getenv("LIGANDMPNN_RUNNER", ""))
    if not runner.is_file():
        raise ToolUnavailable("Configure LIGANDMPNN_RUNNER with run.py.")
    runner = (
        runner.resolve() if task == "design" else runner.resolve().with_name("score.py")
    )
    if not runner.is_file():
        raise ToolUnavailable("The LigandMPNN checkout is missing score.py.")
    model_dir_value = os.getenv("LIGANDMPNN_MODEL_DIR")
    if not model_dir_value or not Path(model_dir_value).is_dir():
        raise ToolUnavailable(
            "Configure LIGANDMPNN_MODEL_DIR with the downloaded model weights."
        )
    model_dir = Path(model_dir_value).resolve()
    prefix, levels, default, suffix = _MODELS[model]
    checkpoint_field = "checkpoint_" + model
    checkpoint = choice(
        payload,
        checkpoint_field,
        {prefix + n + suffix for n in levels},
        prefix + default + suffix,
    )
    for filename in [checkpoint] + ([_SC_CHECKPOINT] if pack else []):
        if not (model_dir / filename).is_file():
            raise ToolUnavailable(
                f"{filename} was not found in LIGANDMPNN_MODEL_DIR. Install the official model weights."
            )
    command_flags += ["--" + checkpoint_field, str(model_dir / checkpoint)]
    if pack:
        command_flags += ["--checkpoint_path_sc", str(model_dir / _SC_CHECKPOINT)]
    python = tool_python("ligandmpnn", "LIGANDMPNN_PYTHON")
    with tempfile.TemporaryDirectory(prefix="bio-web-ligandmpnn-") as temporary:
        workdir = Path(temporary)
        paths = {}
        for index, (key, (pdb, constraints)) in enumerate(prepared.items()):
            filename = f"input_{index}.pdb" if multi else "input.pdb"
            (workdir / filename).write_text(pdb + "\n", encoding="utf-8")
            paths[key] = filename

        def write_json(field: str, document: Any) -> None:
            filename = field + ".json"
            (workdir / filename).write_text(
                json.dumps(document, allow_nan=False) + "\n", encoding="utf-8"
            )
            command_flags.extend(["--" + field, filename])

        if multi:
            write_json("pdb_path_multi", {p: "" for p in paths.values()})
        else:
            command_flags += ["--pdb_path", "input.pdb"]
        for field in _CONSTRAINTS:
            if task != "design" and field.startswith(("bias", "omit")):
                continue
            values = {paths[k]: entry[1][field] for k, entry in prepared.items()}
            if not any(values.values()):
                continue
            if field in ("fixed_residues", "redesigned_residues"):
                # run.py splits multi values; score.py currently expects arrays.
                if task == "design" or not multi:
                    values = {p: " ".join(v) for p, v in values.items()}
                if not multi:
                    command_flags += ["--" + field, next(iter(values.values()))]
                    continue
            write_json(
                field + "_multi" if multi else field,
                values if multi else next(iter(values.values())),
            )
        result = run_command(
            [python, str(runner), "--out_folder", "output", *command_flags], cwd=workdir
        )
        generated = readable_files(workdir / "output")
        if not generated:
            raise ToolExecutionError(
                "LigandMPNN finished without producing result files. See the run log."
            )
        if pack and not any((workdir / "output" / "packed").glob("*.pdb")):
            raise ToolExecutionError(
                "LigandMPNN did not produce requested packed structures. See the run log."
            )
    return {
        "status": "completed",
        "job_name": name,
        "task": task,
        "model_type": model,
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    try:
        python = tool_python("ligandmpnn", "LIGANDMPNN_PYTHON")
    except ToolUnavailable as exc:
        return ToolStatus(CheckResult.NOT_INSTALLED, str(exc))
    runner_value = os.getenv("LIGANDMPNN_RUNNER")
    if (
        not runner_value
        or not Path(runner_value).is_file()
        or not Path(runner_value).with_name("score.py").is_file()
    ):
        return ToolStatus(
            CheckResult.NOT_INSTALLED,
            "Configure LIGANDMPNN_RUNNER with a checkout containing run.py and score.py.",
        )
    model_dir = os.getenv("LIGANDMPNN_MODEL_DIR")
    required = [
        prefix + n + suffix
        for prefix, levels, _, suffix in _MODELS.values()
        for n in levels
    ] + [_SC_CHECKPOINT]
    missing = [
        f for f in required if not model_dir or not (Path(model_dir) / f).is_file()
    ]
    if missing:
        return ToolStatus(
            CheckResult.NOT_INSTALLED,
            "Missing LigandMPNN model weights: " + ", ".join(missing),
        )
    code, output = probe_command([python, str(Path(runner_value).resolve()), "--help"])
    if code != 0:
        return ToolStatus(
            CheckResult.ERROR,
            output or "LigandMPNN dependencies could not be imported.",
        )
    return require_gpu_status(
        ToolStatus(
            CheckResult.PASS, "LigandMPNN entry points and weights are available."
        ),
        torch_device(python),
        "LigandMPNN",
    )
