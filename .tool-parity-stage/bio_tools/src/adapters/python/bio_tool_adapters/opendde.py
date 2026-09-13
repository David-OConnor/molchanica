"""OpenDDE co-folding through its native batch JSON input."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from . import (
    ToolInputError,
    ToolUnavailable,
    catalog_spec,
    readable_files,
    run_command,
    text,
    tool_fields,
    tool_script,
    torch_device,
)
from .environments import environment_python
from .field_processing import (
    boolean,
    document_input,
    integer,
    json_list,
    molecule_boxes,
    safe_name,
)
from .status_check import CheckResult, ToolStatus, probe_cli

SPEC = catalog_spec("opendde", fields=tool_fields("opendde"))


# What one chain is called in OpenDDE's own input, by the molecule box's type.
_POLYMERS = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
}

_MODIFICATION_KEYS = {
    "protein": ("ptmType", "ptmPosition"),
    "dna": ("modificationType", "basePosition"),
    "rna": ("modificationType", "basePosition"),
}


def _from_boxes(payload: dict[str, Any]) -> str:
    """The molecule boxes as one OpenDDE job, for "Set parameters here".

    One job holding every box, which is what the boxes describe: a single
    system to fold. A run of several jobs is what the JSON mode is for.
    """

    boxes = molecule_boxes(
        payload,
        allowed_kinds=frozenset({"protein", "dna", "rna", "ligand", "ion"}),
        allow_ids=True,
        allow_count=True,
        id_count_matches=True,
    )
    sequences: list[dict[str, Any]] = []
    for box in boxes:
        entity: dict[str, Any] = {"count": box.count}
        if box.ids:
            entity["id"] = box.ids
        if box.kind == "ligand":
            entity["ligand"] = box.ligand
            sequences.append({"ligand": entity})
            continue
        if box.kind == "ion":
            entity["ion"] = box.ion
            sequences.append({"ion": entity})
            continue
        if box.cyclic:
            raise ToolInputError(
                f'Molecule "{box.chain}": OpenDDE does not support cyclic chains.'
            )
        entity["sequence"] = box.sequence
        if box.modifications:
            type_key, position_key = _MODIFICATION_KEYS[box.kind]
            entity["modifications"] = [
                {
                    type_key: f"CCD_{mod.residue.removeprefix('CCD_')}",
                    position_key: mod.position,
                }
                for mod in box.modifications
            ]
        if box.kind == "protein":
            if box.paired_msa_path:
                entity["pairedMsaPath"] = box.paired_msa_path
            if box.unpaired_msa_path:
                entity["unpairedMsaPath"] = box.unpaired_msa_path
            if box.templates_path:
                entity["templatesPath"] = box.templates_path
        elif box.kind == "rna" and box.unpaired_msa_path:
            entity["unpairedMsaPath"] = box.unpaired_msa_path
        sequences.append({_POLYMERS[box.kind]: entity})
    job = {
        "name": safe_name(payload, default="opendde-job"),
        "modelSeeds": [1],
        "sequences": sequences,
    }
    covalent_bonds = json_list(payload, "covalent_bonds", max_length=500_000)
    if covalent_bonds:
        job["covalent_bonds"] = covalent_bonds
    return json.dumps([job], indent=2)


def _input_document(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw = text(payload, "input_json", max_length=1_000_000)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(
            f"input_json is not valid JSON: {exc.msg} at line {exc.lineno}."
        ) from exc

    if not isinstance(document, list) or not document:
        raise ToolInputError("input_json must be a non-empty top-level list of jobs.")
    if len(document) > 32:
        raise ToolInputError("input_json accepts at most 32 jobs per run.")
    names: set[str] = set()
    for index, job in enumerate(document, start=1):
        if not isinstance(job, dict):
            raise ToolInputError(f"OpenDDE job {index} must be a JSON object.")
        if not isinstance(job.get("name"), str) or not job["name"].strip():
            raise ToolInputError(f"OpenDDE job {index} needs a non-empty name.")
        name = job["name"]
        if (
            name.casefold() == "err"
            or name in {".", ".."}
            or any(character in name for character in "/\\\0")
        ):
            raise ToolInputError(
                f'OpenDDE job name "{name}" must be a safe path component.'
            )
        if name in names:
            raise ToolInputError(f'OpenDDE job name "{name}" is duplicated.')
        names.add(name)

        model_seeds = job.get("modelSeeds")
        if model_seeds is not None:
            if not isinstance(model_seeds, list):
                raise ToolInputError(f'modelSeeds for job "{name}" must be a list.')
            invalid_seed = any(
                isinstance(seed, bool)
                or not (
                    isinstance(seed, int)
                    and 0 <= seed <= 4_294_967_295
                    or isinstance(seed, str)
                    and seed.strip() == seed
                    and seed.isdecimal()
                    and int(seed) <= 4_294_967_295
                )
                for seed in model_seeds
            )
            if invalid_seed:
                raise ToolInputError(
                    f'modelSeeds for job "{name}" must contain integer seeds from 0 to 4294967295.'
                )
        sequences = job.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ToolInputError(
                f'OpenDDE job "{job["name"]}" needs a non-empty sequences list.'
            )

    return document, json.dumps(document, indent=2)


def _seeds(payload: dict[str, Any]) -> str | None:
    raw = text(payload, "seeds", required=False, max_length=200).strip()
    if not raw:
        return None

    seeds: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        try:
            value = int(item)
        except ValueError as exc:
            raise ToolInputError(f'seeds entry "{item}" is not an integer.') from exc
        if not 0 <= value <= 4_294_967_295:
            raise ToolInputError(f'seeds entry "{item}" is out of range.')
        seeds.append(str(value))
    if len(seeds) > 32:
        raise ToolInputError("seeds accepts at most 32 values.")
    return ",".join(seeds)


def _choice(
    payload: dict[str, Any], field: str, choices: set[str], default: str
) -> str:
    value = str(payload.get(field) or default)
    if value not in choices:
        raise ToolInputError(f"Unsupported {field}: {value}.")
    return value


def run(payload: dict[str, Any]) -> dict[str, Any]:
    payload = document_input(payload, "input_json", from_boxes=_from_boxes)
    document, input_json = _input_document(payload)
    seeds = _seeds(payload)
    samples = integer(payload, "samples", default=5, minimum=1, maximum=64)
    steps = integer(payload, "steps", default=200, minimum=1, maximum=1_000)
    cycles = integer(payload, "cycles", default=10, minimum=1, maximum=100)
    dtype = _choice(payload, "dtype", {"fp32", "bf16"}, "fp32")
    device = _choice(payload, "device", {"auto", "cuda", "cpu"}, "auto")

    use_msa = boolean(payload, "use_msa")
    use_rna_msa = boolean(payload, "use_rna_msa")
    if use_rna_msa and not use_msa:
        raise ToolInputError("use_rna_msa requires use_msa to be enabled.")

    command = [
        tool_script("opendde", "opendde", "OPENDDE_EXECUTABLE"),
        "pred",
        "-i",
        "input.json",
        "-o",
        "output",
        "-n",
        "opendde_v1",
        "--use_msa",
        str(use_msa).lower(),
        "--use_template",
        str(boolean(payload, "use_template")).lower(),
        "--use_rna_msa",
        str(use_rna_msa).lower(),
        "--use_tfg_guidance",
        str(boolean(payload, "use_guidance")).lower(),
        "--sample",
        str(samples),
        "--step",
        str(steps),
        "--cycle",
        str(cycles),
        "--dtype",
        dtype,
        "--device",
        device,
        "--need_atom_confidence",
        str(boolean(payload, "need_atom_confidence", True)).lower(),
        "--deterministic",
        str(boolean(payload, "deterministic")).lower(),
    ]
    if seeds is not None:
        command.extend(["--seeds", seeds])

    with tempfile.TemporaryDirectory(prefix="bio-web-opendde-") as temporary:
        workdir = Path(temporary)
        input_path = workdir / "input.json"
        output_path = workdir / "output"
        input_path.write_text(input_json, encoding="utf-8")
        result = run_command(
            command,
            cwd=workdir,
            artifacts=[output_path, input_path],
        )
        generated = readable_files(output_path)

    return {
        "status": "completed",
        "input": {"json": document},
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    try:
        opendde = tool_script("opendde", "opendde", "OPENDDE_EXECUTABLE")
    except ToolUnavailable as exc:
        return ToolStatus(CheckResult.NOT_INSTALLED, str(exc))
    status = probe_cli([opendde])
    if status.result == CheckResult.PASS:
        status.device = torch_device(environment_python("opendde"))
    return status
