"""Boltz-2 structure and affinity prediction through its native YAML input."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

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
    decimal,
    document_input,
    integer,
    json_list,
    molecule_boxes,
    safe_name,
)
from .status_check import CheckResult, ToolStatus, probe_cli

SPEC = catalog_spec("boltz2", fields=tool_fields("boltz2"))


def _from_boxes(payload: dict[str, Any]) -> str:
    """The molecule boxes as Boltz YAML, for "Set parameters here"."""

    boxes = molecule_boxes(payload, allow_ids=True, require_ids=True)
    sequences: list[dict[str, Any]] = []
    for box in boxes:
        native_id: str | list[str] = box.ids[0] if len(box.ids) == 1 else box.ids
        if box.kind == "ligand":
            # A CCD_ code names a component from the dictionary; anything else
            # is read as SMILES, which is how the other adapters take it too.
            entity: dict[str, Any] = {"id": native_id}
            if box.ligand.upper().startswith("CCD_"):
                entity["ccd"] = box.ligand[4:].upper()
            else:
                entity["smiles"] = box.ligand
            sequences.append({"ligand": entity})
            continue
        entity = {"id": native_id, "sequence": box.sequence}
        if box.cyclic:
            entity["cyclic"] = True
        if box.modifications:
            entity["modifications"] = [
                {"position": mod.position, "ccd": mod.residue}
                for mod in box.modifications
            ]
        # Without an alignment Boltz wants to be told so explicitly, unless the
        # run is opting into the public MSA server below.
        if box.kind == "protein":
            if box.msa not in (None, ""):
                if not isinstance(box.msa, str):
                    raise ToolInputError(
                        f'Molecule "{box.chain}" MSA must be a path or "empty".'
                    )
                entity["msa"] = box.msa.strip()
            elif not boolean(payload, "use_msa_server", False):
                entity["msa"] = "empty"
        sequences.append({box.kind: entity})

    document: dict[str, Any] = {"version": 1, "sequences": sequences}
    for field in ("constraints", "templates", "properties"):
        values = json_list(payload, field, max_length=500_000)
        if values:
            document[field] = values
    return yaml.safe_dump(
        document,
        sort_keys=False,
        default_flow_style=False,
        width=1_000_000,
    )


def run(payload: dict[str, Any]) -> dict[str, Any]:
    name = safe_name(payload, default="boltz2-demo")
    payload = document_input(
        payload, "yaml_spec", from_boxes=_from_boxes, max_length=500_000
    )
    yaml_input = text(payload, "yaml_spec", max_length=500_000)

    recycling_steps = integer(
        payload, "recycling_steps", default=3, minimum=1, maximum=20
    )
    sampling_steps = integer(
        payload, "sampling_steps", default=200, minimum=10, maximum=1_000
    )
    diffusion_samples = integer(
        payload, "diffusion_samples", default=1, minimum=1, maximum=50
    )
    step_scale = decimal(payload, "step_scale", default=1.5, minimum=0, maximum=5)
    sampling_steps_affinity = integer(
        payload, "sampling_steps_affinity", default=200, minimum=1, maximum=1_000
    )
    diffusion_samples_affinity = integer(
        payload, "diffusion_samples_affinity", default=5, minimum=1, maximum=50
    )
    seed = integer(payload, "seed", default=0, minimum=0, maximum=4_294_967_295)
    output_format = str(payload.get("output_format") or "mmcif")
    if output_format not in {"mmcif", "pdb"}:
        raise ToolInputError(f"Unsupported output_format: {output_format}.")

    command = [
        tool_script("boltz", "boltz", "BOLTZ_EXECUTABLE"),
        "predict",
        "input.yaml",
        "--model",
        "boltz2",
        "--out_dir",
        name,
        "--recycling_steps",
        str(recycling_steps),
        "--sampling_steps",
        str(sampling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--step_scale",
        str(step_scale),
        "--sampling_steps_affinity",
        str(sampling_steps_affinity),
        "--diffusion_samples_affinity",
        str(diffusion_samples_affinity),
        "--seed",
        str(seed),
        "--output_format",
        output_format,
    ]
    if boolean(payload, "use_msa_server"):
        command.append("--use_msa_server")
    if boolean(payload, "use_potentials"):
        command.append("--use_potentials")
    if boolean(payload, "affinity_mw_correction"):
        command.append("--affinity_mw_correction")

    with tempfile.TemporaryDirectory(prefix="bio-web-boltz2-") as temporary:
        workdir = Path(temporary)
        input_path = workdir / "input.yaml"
        output_path = workdir / name
        input_path.write_text(yaml_input, encoding="utf-8")
        result = run_command(
            command,
            cwd=workdir,
            artifacts=[output_path, input_path],
        )
        generated = readable_files(output_path)

    return {
        "status": "completed",
        "input": {"yaml": yaml_input},
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    try:
        boltz = tool_script("boltz", "boltz", "BOLTZ_EXECUTABLE")
    except ToolUnavailable as exc:
        return ToolStatus(CheckResult.NOT_INSTALLED, str(exc))
    status = probe_cli([boltz])
    if status.result == CheckResult.PASS:
        status.device = torch_device(environment_python("boltz"))
    return status
