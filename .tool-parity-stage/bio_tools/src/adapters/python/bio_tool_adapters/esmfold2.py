"""ESMFold2 all-atom structure prediction adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from . import (
    ToolInputError,
    catalog_spec,
    readable_files,
    run_command,
    text,
    tool_fields,
    tool_python,
    torch_device,
)
from .environments import environment_python
from .field_processing import (
    decimal,
    document_input,
    integer,
    json_list,
    json_object,
    molecule_boxes,
    safe_name,
)
from .status_check import CheckResult, ToolStatus, probe_command, probe_python_package

RUNNER = Path(__file__).resolve().parent / "tool_scripts" / "esmfold2_inference.py"
SPEC = catalog_spec("esmfold2", fields=tool_fields("esmfold2"))


def _input_document(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    raw = text(payload, "input_json", max_length=500_000)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(
            f"input_json is not valid JSON: {exc.msg} at line {exc.lineno}."
        ) from exc

    if not isinstance(document, dict):
        raise ToolInputError("input_json must be a JSON object.")
    sequences = document.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ToolInputError("input_json needs a non-empty sequences list.")
    if len(sequences) > 32:
        raise ToolInputError("input_json accepts at most 32 sequence entries.")

    for index, entry in enumerate(sequences, start=1):
        if not isinstance(entry, dict):
            raise ToolInputError(f"Sequence entry {index} must be a JSON object.")
        kind = entry.get("type")
        if kind not in {"protein", "dna", "rna", "ligand"}:
            raise ToolInputError(
                f'Sequence entry {index} has unsupported type "{kind}".'
            )
        chain_id = entry.get("id")
        if not isinstance(chain_id, (str, list)) or not chain_id:
            raise ToolInputError(f"Sequence entry {index} needs a non-empty id.")
        if kind == "ligand":
            if not entry.get("smiles") and not entry.get("ccd"):
                raise ToolInputError(
                    f"Ligand entry {index} needs either smiles or a ccd list."
                )
        elif not isinstance(entry.get("sequence"), str) or not entry["sequence"]:
            raise ToolInputError(f"Sequence entry {index} needs a sequence.")

    return document, json.dumps(document, indent=2)


def _choice(
    payload: dict[str, Any], field: str, choices: set[str], default: str
) -> str:
    value = str(payload.get(field) or default)
    if value not in choices:
        raise ToolInputError(f"Unsupported {field}: {value}.")
    return value


def _from_boxes(payload: dict[str, Any]) -> str:
    """The molecule boxes as a StructurePredictionInput, for "Set parameters here"."""

    boxes = molecule_boxes(
        payload, allow_ids=True, require_ids=True, modification_position_base=0
    )
    sequences: list[dict[str, Any]] = []
    for box in boxes:
        native_id: str | list[str] = box.ids[0] if len(box.ids) == 1 else box.ids
        if box.kind == "ligand":
            # A CCD_ code names a component from the dictionary; anything else
            # is read as SMILES, which is how the other adapters take it too.
            entry: dict[str, Any] = {"type": "ligand", "id": native_id}
            ligand_parts = [item.strip() for item in box.ligand.split(",")]
            if all(item.upper().startswith("CCD_") for item in ligand_parts):
                entry["ccd"] = [item[4:].upper() for item in ligand_parts]
            else:
                entry["smiles"] = box.ligand
            sequences.append(entry)
            continue
        if box.cyclic:
            raise ToolInputError(
                f'Molecule "{box.chain}": ESMFold2 does not support cyclic chains.'
            )
        entry = {"type": box.kind, "id": native_id, "sequence": box.sequence}
        if box.modifications:
            entry["modifications"] = [
                {"position": mod.position, "ccd": mod.residue}
                for mod in box.modifications
            ]
        if box.kind in {"protein", "rna"} and box.msa not in (None, ""):
            msa = box.msa
            if isinstance(msa, str) and msa.lstrip().startswith(("{", '"')):
                try:
                    msa = json.loads(msa)
                except json.JSONDecodeError as exc:
                    raise ToolInputError(
                        f'Molecule "{box.chain}" MSA is not valid JSON: {exc.msg}.'
                    ) from exc
            if not isinstance(msa, (str, dict)):
                raise ToolInputError(
                    f'Molecule "{box.chain}" MSA must be serialized text or a '
                    "JSON object."
                )
            entry["msa"] = msa
        sequences.append(entry)

    document: dict[str, Any] = {"sequences": sequences}
    pocket = json_object(payload, "pocket", max_length=500_000)
    if pocket:
        document["pocket"] = pocket
    for field in ("distogram_conditioning", "covalent_bonds"):
        values = json_list(payload, field, max_length=500_000)
        if values:
            document[field] = values
    return json.dumps(document, indent=2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    name = safe_name(payload, default="esmfold2-demo")
    payload = document_input(payload, "input_json", from_boxes=_from_boxes)
    document, input_json = _input_document(payload)
    num_loops = integer(payload, "num_loops", default=20, minimum=1, maximum=64)
    sampling_steps = integer(
        payload, "num_sampling_steps", default=200, minimum=2, maximum=1_000
    )
    diffusion_samples = integer(
        payload, "num_diffusion_samples", default=1, minimum=1, maximum=16
    )
    seed = integer(payload, "seed", default=0, minimum=0, maximum=4_294_967_295)
    lm_dropout = decimal(payload, "lm_dropout", default=0.3, minimum=0, maximum=1)
    chunk_size = integer(payload, "chunk_size", default=64, minimum=0, maximum=512)
    device = _choice(payload, "device", {"auto", "cuda", "cpu"}, "auto")
    precision = _choice(payload, "precision", {"default", "bf16", "fp32"}, "default")

    python = tool_python("esmfold", "ESMFOLD_PYTHON")
    command = [
        python,
        str(RUNNER),
        "--input",
        "input.json",
        "--output-dir",
        name,
        "--name",
        name,
        "--num-loops",
        str(num_loops),
        "--num-sampling-steps",
        str(sampling_steps),
        "--num-diffusion-samples",
        str(diffusion_samples),
        "--seed",
        str(seed),
        "--lm-dropout",
        str(lm_dropout),
        "--chunk-size",
        str(chunk_size),
        "--device",
        device,
        "--precision",
        precision,
    ]

    with tempfile.TemporaryDirectory(prefix="bio-web-esmfold2-") as temporary:
        workdir = Path(temporary)
        input_path = workdir / "input.json"
        output_path = workdir / name
        input_path.write_text(input_json, encoding="utf-8")
        result = run_command(
            command,
            cwd=workdir,
            artifacts=[output_path, input_path],
        )
        generated = readable_files(output_path)

    return {
        "status": "completed",
        "input": document,
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    python = environment_python("esmfold2")
    if not python.is_file():
        return ToolStatus(
            CheckResult.NOT_INSTALLED,
            "No Python environment for esmfold2. Run `python install_tools.py esmfold2`.",
        )
    status = probe_python_package(python, "esm", "esm.models.esmfold2")
    if status.result == CheckResult.PASS:
        code, output = probe_command(
            [
                str(python),
                "-c",
                (
                    "from inspect import signature; "
                    "from esm.models.hub import read_safetensors_dir; "
                    "assert 'dtype' in signature(read_safetensors_dir).parameters, "
                    "'reinstall ESMFold2 to enable its low-memory checkpoint loader'"
                ),
            ]
        )
        if code != 0:
            detail = (
                output.splitlines()[-1] if output else "ESMFold2 needs reinstalling."
            )
            return ToolStatus(CheckResult.ERROR, detail[:200])
        status.device = torch_device(python)
    return status
