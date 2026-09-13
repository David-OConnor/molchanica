"""Chai-1 biomolecular structure prediction."""

from __future__ import annotations

import hashlib
import re
import string
import tempfile
from pathlib import Path
from typing import Any

from . import (
    ToolInputError,
    ToolUnavailable,
    catalog_spec,
    preset_payload,
    readable_files,
    run_command,
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
    lines,
    molecule_boxes,
)
from .status_check import CheckResult, ToolStatus, probe_cli, require_gpu_status

SPEC = catalog_spec(
    "chai1",
    fields=tool_fields("chai1"),
)
EXAMPLE_MSAS = Path(__file__).resolve().parent / "tool_data" / "chai1" / "msas"


_FASTA_HEADER = re.compile(
    r"^>(protein|dna|rna|ligand|glycan)\|(?:name=)?([^|\s]+)$", re.IGNORECASE
)
_REST_HEADER = "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id"


def _chain_name(index: int) -> str:
    name = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        name = string.ascii_uppercase[remainder] + name
        index -= 1
    return name


def _from_boxes(payload: dict[str, Any]) -> str:
    records: list[str] = []
    for box in molecule_boxes(payload):
        if box.kind == "ligand":
            if box.ligand.upper().startswith("CCD_"):
                raise ToolInputError(
                    f'Molecule "{box.chain}": Chai-1 only accepts SMILES ligands, not CCD codes.'
                )
            records.append(f">ligand|name={box.chain}\n{box.ligand}\n")
            continue

        if box.cyclic:
            raise ToolInputError(
                f'Molecule "{box.chain}" is cyclic, but Chai-1 does not support cyclic chains.'
            )

        sequence = box.sequence
        if box.modifications:
            mods_by_position = {mod.position: mod.residue for mod in box.modifications}
            sequence = "".join(
                f"({mods_by_position[index]})" if index in mods_by_position else char
                for index, char in enumerate(box.sequence, start=1)
            )
        records.append(f">{box.kind}|name={box.chain}\n{sequence}\n")
    return "".join(records)


def _legacy_fasta(payload: dict[str, Any], task: str) -> str:
    records: list[tuple[str, str, str]] = []
    if task == "list":
        for field, kind in (("proteins", "protein"), ("dnas", "dna"), ("rnas", "rna")):
            for sequence in lines(
                payload, field, required=field == "proteins", maximum=26
            ):
                records.append((kind, _chain_name(len(records)), sequence))
    else:
        for molecule in json_list(payload, "molecules", maximum=26):
            if not isinstance(molecule, dict) or molecule.get("type") not in {
                "protein",
                "dna",
                "rna",
            }:
                raise ToolInputError("Molecules need protein, DNA, or RNA entries.")
            name, sequence = molecule.get("chain"), molecule.get("sequence")
            if not isinstance(name, str) or not isinstance(sequence, str):
                raise ToolInputError("Molecules need chain and sequence strings.")
            records.append((molecule["type"], name, sequence))
        if not records:
            raise ToolInputError("Molecules need at least one polymer entry.")
    used = {name for _, name, _ in records}
    for smiles in lines(payload, "ligands", required=False, maximum=20):
        index = len(records)
        while _chain_name(index) in used:
            index += 1
        name = _chain_name(index)
        used.add(name)
        records.append(("ligand", name, smiles))
    if not records or len(records) > 26:
        raise ToolInputError(
            "The chain-list and molecule modes accept 1 to 26 entities."
        )
    return "".join(
        f">{kind}|name={name}\n{sequence}\n" for kind, name, sequence in records
    )


def _read_fasta(raw: str) -> tuple[str, dict[str, str], set[str]]:
    records: list[tuple[str, str, str]] = []
    name: str | None = None
    kind = ""
    parts: list[str] = []
    used: set[str] = set()

    def finish() -> None:
        if name is None:
            return
        sequence = "".join(parts).strip()
        if not sequence or (
            kind != "glycan" and any(char.isspace() for char in sequence)
        ):
            raise ToolInputError(
                f'Entity "{name}" needs a non-empty, continuous sequence.'
            )
        records.append((kind, name, sequence))

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            match = _FASTA_HEADER.fullmatch(line)
            if match is None:
                raise ToolInputError(f"Invalid Chai FASTA header: {line}.")
            kind, name = match.group(1).lower(), match.group(2)
            if name in used:
                raise ToolInputError(f'Entity name "{name}" is repeated.')
            used.add(name)
            parts = []
        else:
            if name is None:
                raise ToolInputError("FASTA sequence appears before its header.")
            parts.append(line)
    finish()
    if not records or len(records) > 64:
        raise ToolInputError("Chai FASTA needs 1 to 64 entities.")
    normalized = "".join(
        f">{kind}|name={name}\n{sequence}\n" for kind, name, sequence in records
    )
    sequences = {
        _chain_name(index): re.sub(r"\([A-Za-z0-9]+\)", "X", sequence).upper()
        for index, (kind, _, sequence) in enumerate(records)
        if kind in {"protein", "dna", "rna"}
    }
    nonpolymers = {
        _chain_name(index)
        for index, (kind, _, _) in enumerate(records)
        if kind in {"ligand", "glycan"}
    }
    return normalized, sequences, nonpolymers


def _residue_code(
    sequences: dict[str, str], chain: str, position: Any, field_name: str
) -> str:
    if chain not in sequences:
        raise ToolInputError(f'{field_name} references unknown chain "{chain}".')
    try:
        index = int(position)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{field_name} residue index must be an integer.") from exc
    sequence = sequences[chain]
    if not 1 <= index <= len(sequence):
        raise ToolInputError(
            f'{field_name} residue {index} is out of range for chain "{chain}".'
        )
    return f"{sequence[index - 1]}{index}"


def _restraints_csv(
    payload: dict[str, Any],
    sequences: dict[str, str],
    nonpolymer_chains: set[str],
    min_distance: float,
    max_distance: float,
) -> str:
    rows = [_REST_HEADER]
    restraint_id = 0

    def next_id(prefix: str) -> str:
        nonlocal restraint_id
        restraint_id += 1
        return f"{prefix}_{restraint_id}"

    for item in json_list(payload, "pocket_restraints", maximum=50):
        if (
            not isinstance(item, dict)
            or not {"chainA", "res_idxA", "chainB"} <= item.keys()
        ):
            raise ToolInputError(
                "pocket_restraints entries need chainA, res_idxA, and chainB."
            )
        chain_b = str(item["chainB"])
        if chain_b not in sequences and chain_b not in nonpolymer_chains:
            raise ToolInputError(
                f'pocket_restraints references unknown chain "{chain_b}".'
            )
        res_a = _residue_code(
            sequences, str(item["chainA"]), item["res_idxA"], "pocket_restraints"
        )
        # Chai's pocket restraints are asymmetric the other way round from how
        # this field is described: the chain-level ("any residue") side must
        # be column A and the specific pocket residue must be column B.
        rows.append(
            f"{chain_b},,{item['chainA']},{res_a},pocket,1.0,"
            f"{min_distance},{max_distance},-,{next_id('pocket')}"
        )

    for item in json_list(payload, "contact_restraints", maximum=50):
        required = ("chainA", "res_idxA", "chainB", "res_idxB")
        if not isinstance(item, dict) or any(key not in item for key in required):
            raise ToolInputError(
                "contact_restraints entries need chainA, res_idxA, chainB, and res_idxB."
            )
        res_a = _residue_code(
            sequences, str(item["chainA"]), item["res_idxA"], "contact_restraints"
        )
        res_b = _residue_code(
            sequences, str(item["chainB"]), item["res_idxB"], "contact_restraints"
        )
        rows.append(
            f"{item['chainA']},{res_a},{item['chainB']},{res_b},contact,1.0,"
            f"{min_distance},{max_distance},-,{next_id('contact')}"
        )

    for item in json_list(payload, "covalent_restraints", maximum=20):
        required = ("chainA", "covalentAtomA", "chainB", "covalentAtomB")
        if not isinstance(item, dict) or any(key not in item for key in required):
            raise ToolInputError(
                "covalent_restraints entries need chainA, covalentAtomA, chainB, and covalentAtomB. "
                "Polymer partners also need a residue index."
            )
        chain_a, chain_b = str(item["chainA"]), str(item["chainB"])
        for key in ("covalentAtomA", "covalentAtomB"):
            if not re.fullmatch(r"[A-Za-z0-9']{1,8}", str(item[key])):
                raise ToolInputError(f"{key} must be an atom name.")

        def _side(chain: str, position: Any, atom: str) -> str:
            if chain in nonpolymer_chains:
                return f"@{atom}"
            code = _residue_code(sequences, chain, position, "covalent_restraints")
            return f"{code}@{atom}"

        side_a = _side(chain_a, item.get("res_idxA"), str(item["covalentAtomA"]))
        side_b = _side(chain_b, item.get("res_idxB"), str(item["covalentAtomB"]))
        rows.append(
            f"{chain_a},{side_a},{chain_b},{side_b},covalent,1.0,0.0,0.0,-,{next_id('covalent')}"
        )

    return "\n".join(rows) + "\n"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    payload = preset_payload("chai1", payload)
    task = str(payload.get("task") or "")
    if (
        task in {"list", "molecules"}
        and not payload.get("input_mode")
        and not payload.get("input_fasta")
    ):
        payload = {
            **payload,
            "input_mode": "text",
            "input_fasta": _legacy_fasta(payload, task),
        }
    payload = document_input(
        payload,
        "input_fasta",
        from_boxes=_from_boxes,
        upload_field="input_file",
        max_length=500_000,
    )
    fasta, sequences, nonpolymers = _read_fasta(str(payload["input_fasta"]))

    use_msa = boolean(payload, "use_msa_server", False)
    use_templates = boolean(payload, "use_templates_server", False)
    msa_directory = str(payload.get("msa_directory") or "").strip()
    use_example_msas = boolean(payload, "use_example_msas", False)
    template_hits_path = str(payload.get("template_hits_path") or "").strip()
    if sum((bool(use_msa), bool(msa_directory), use_example_msas)) > 1:
        raise ToolInputError(
            "Choose one MSA source: server, local directory, or supplied examples."
        )
    if use_example_msas:
        if not EXAMPLE_MSAS.is_dir():
            raise ToolUnavailable(
                "Chai’s supplied MSA files are not installed with Bio Web."
            )
        filenames = {path.name for path in EXAMPLE_MSAS.glob("*.aligned.pqt")}
        hashes = {
            hashlib.sha256(sequence.encode()).hexdigest() + ".aligned.pqt"
            for sequence in sequences.values()
        }
        if len(filenames) != 2 or not filenames <= hashes:
            raise ToolInputError(
                "The supplied MSAs require Chai’s official example protein sequences."
            )
        msa_directory = str(EXAMPLE_MSAS)
    if use_templates and template_hits_path:
        raise ToolInputError(
            "Choose either the template server or a template hits file."
        )
    if use_templates and not use_msa:
        raise ToolInputError("The template server requires the MSA server.")
    msa_server_url = str(
        payload.get("msa_server_url") or "https://api.colabfold.com"
    ).strip()
    if not re.fullmatch(r"https://[^\s]+", msa_server_url):
        raise ToolInputError("msa_server_url must be an HTTPS URL.")
    device = str(payload.get("device") or "cuda:0").strip()
    if not re.fullmatch(r"cuda:\d+", device):
        raise ToolInputError("Chai-1 needs a CUDA device such as cuda:0.")
    if msa_directory:
        directory = Path(msa_directory).resolve()
        if not directory.is_dir():
            raise ToolInputError(
                "msa_directory must be a directory on the compute node."
            )
        msa_directory = str(directory)
    if template_hits_path:
        hits = Path(template_hits_path).resolve()
        if not hits.is_file():
            raise ToolInputError(
                "template_hits_path must be a file on the compute node."
            )
        template_hits_path = str(hits)
    min_distance = decimal(
        payload, "restraints_min_distance", default=0, minimum=0, maximum=100
    )
    max_distance = decimal(
        payload, "restraints_max_distance", default=5, minimum=0, maximum=100
    )
    if min_distance > max_distance:
        raise ToolInputError("Restraint minimum distance exceeds maximum distance.")
    num_samples = integer(payload, "num_samples", default=5, minimum=1, maximum=25)
    num_trunk_samples = integer(
        payload, "num_trunk_samples", default=1, minimum=1, maximum=10
    )
    num_recycles = integer(payload, "num_recycles", default=3, minimum=0, maximum=20)
    num_diffn_timesteps = integer(
        payload, "num_diffn_timesteps", default=200, minimum=1, maximum=1_000
    )
    seed = integer(payload, "seed", default=0, minimum=0, maximum=2_147_483_647)
    recycle_msa_subsample = integer(
        payload, "recycle_msa_subsample", default=0, minimum=0, maximum=1_000
    )

    runner = tool_script("chai1", "chai-lab", "CHAI1_EXECUTABLE")
    restraints_csv = _restraints_csv(
        payload, sequences, nonpolymers, min_distance, max_distance
    )
    raw_restraints = str(payload.get("restraints_csv") or "").strip()
    uploaded_restraints = str(payload.get("restraints_file") or "").strip()
    if raw_restraints and uploaded_restraints:
        raise ToolInputError("Paste or upload one restraint table, not both.")
    raw_restraints = raw_restraints or uploaded_restraints
    if raw_restraints and restraints_csv.strip() != _REST_HEADER:
        raise ToolInputError(
            "Use either the restraint table or the guided-restraint fields."
        )
    if raw_restraints:
        if not raw_restraints.startswith(_REST_HEADER):
            raise ToolInputError("Restraint table needs the official Chai CSV header.")
        restraints_csv = raw_restraints + "\n"
    has_restraints = restraints_csv.strip() != _REST_HEADER

    with tempfile.TemporaryDirectory(prefix="bio-web-chai1-") as temporary:
        workdir = Path(temporary)
        (workdir / "input.fasta").write_text(fasta, encoding="utf-8")
        command = [runner, "fold"]
        if use_msa:
            command.append("--use-msa-server")
            command += ["--msa-server-url", msa_server_url]
        if use_templates:
            command.append("--use-templates-server")
        if msa_directory:
            command += ["--msa-directory", msa_directory]
        if template_hits_path:
            command += ["--template-hits-path", template_hits_path]
        command += [
            "--use-esm-embeddings"
            if boolean(payload, "use_esm_embeddings", True)
            else "--no-use-esm-embeddings",
            "--low-memory"
            if boolean(payload, "low_memory", True)
            else "--no-low-memory",
            "--recycle-msa-subsample",
            str(recycle_msa_subsample),
            "--device",
            device,
        ]
        command += [
            "--num-trunk-recycles",
            str(num_recycles),
            "--num-diffn-timesteps",
            str(num_diffn_timesteps),
            "--num-diffn-samples",
            str(num_samples),
            "--num-trunk-samples",
            str(num_trunk_samples),
            "--seed",
            str(seed),
        ]
        if has_restraints:
            (workdir / "restraints.restraints").write_text(
                restraints_csv, encoding="utf-8"
            )
            command += ["--constraint-path", "restraints.restraints"]
        command.extend(["input.fasta", "output"])
        result = run_command(command, cwd=workdir)
        generated = readable_files(workdir / "output")
    return {
        "status": "completed",
        "input": {"fasta": fasta},
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    try:
        runner = tool_script("chai1", "chai-lab", "CHAI1_EXECUTABLE")
    except ToolUnavailable as exc:
        return ToolStatus(CheckResult.NOT_INSTALLED, str(exc))
    status = probe_cli([runner])
    if status.result != CheckResult.PASS:
        return status
    return require_gpu_status(
        status, torch_device(environment_python("chai1")), "Chai-1"
    )
