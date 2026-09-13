"""RFdiffusion3 all-atom backbone generation.

Unlike RFdiffusion, whose constraints are Hydra overrides on the command line,
RFD3 takes a JSON `InputSpecification`: one object per named design, holding
the contig, the ligands, and a per-atom account of what is held fixed. The
tasks below each build one such object out of ordinary form fields, and the
web form also accepts uploaded or entered JSON/YAML. Native InputSpecification
fields are shared across all design workflows; older mode-specific API payloads
are still accepted. Job-level settings control the inference command.
"""

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
    ToolInputError,
    ToolUnavailable,
    catalog_spec,
    preset_payload,
    tool_fields,
    tool_tasks,
    readable_files,
    run_command,
    text,
    tool_script,
    torch_device,
)
from .environments import environment_python
from .field_processing import boolean, decimal, integer, safe_name
from .status_check import (
    CheckResult,
    ToolStatus,
    probe_cli,
    require_gpu_status,
)

WEB_MANAGED_FIELDS = {
    "out_dir": "Bio Web creates a separate output directory for each run and collects its result files.",
    "ckpt_path": "Bio Web uses the installed RFD3 checkpoint configured on the server.",
    "skip_existing": "Each web run starts with a fresh output directory, so there are no existing designs to skip.",
}

SPEC = catalog_spec(
    "rfd3",
    fields=[
        field for field in tool_fields("rfd3") if field.name not in WEB_MANAGED_FIELDS
    ],
    tasks=tool_tasks("rfd3"),
)

# The name foundry's checkpoint registry gives the RFD3 weights, and so the
# name bio_tools' recipe downloads them under.
_CHECKPOINT_NAME = "rfd3_latest.ckpt"

# One contig segment: a chain break, a designed length or range (40-120), or a
# residue or range taken from the input structure (A10-25, M52).
_SEGMENT = re.compile(r"(?:/0|[A-Za-z]?\d+(?:-\d+)?)\Z")
# A PDB chemical component ID, as ligands are named in both `ligand` and the
# selection dictionaries.
_LIGAND = re.compile(r"[A-Za-z0-9]{1,5}\Z")
# Atom names carry primes on nucleic acids, e.g. O3'.
_ATOM = re.compile(r"[A-Za-z0-9']{1,6}\Z")
# Point groups RFD3's symmetry sampler accepts.
_SYMMETRY = re.compile(r"[CDcd][1-9]\d*\Z")
_LENGTH = re.compile(r"\d+(?:-\d+)?\Z")
# Shorthands the selection mini-language accepts in place of an atom list.
_ATOM_SHORTHANDS = {"ALL", "TIP", "BKBN"}


def _contig(payload: dict[str, Any], name: str, *, required: bool = True) -> str:
    """Validate a comma-separated contig string, and return it normalized."""

    value = text(payload, name, required=required, max_length=5_000)
    if not value:
        return ""
    segments = [segment.strip() for segment in value.split(",") if segment.strip()]
    if not segments:
        raise ToolInputError(f"{name} must contain at least one segment.")
    for segment in segments:
        if not _SEGMENT.fullmatch(segment):
            raise ToolInputError(
                f'{name} segment "{segment}" is not a contig segment such as '
                "A10-25, 40-120, or /0."
            )
    return ",".join(segments)


def _length(payload: dict[str, Any], name: str, *, required: bool = True):
    """A design length, as the int or "min-max" string the specification wants."""

    value = text(payload, name, required=required, max_length=40)
    if not value:
        return None
    if not _LENGTH.fullmatch(value):
        raise ToolInputError(
            f"{name} must be a number or a min-max range like 140-150."
        )
    if "-" in value:
        minimum, maximum = (int(part) for part in value.split("-", 1))
        if minimum < 1 or minimum > maximum:
            raise ToolInputError(f"{name} must be a positive, increasing length range.")
        return value
    if int(value) < 1:
        raise ToolInputError(f"{name} must be positive.")
    return int(value)


def _ligands(payload: dict[str, Any], name: str, *, required: bool = False) -> str:
    value = text(payload, name, required=required, max_length=200)
    if not value:
        return ""
    codes = [code.strip() for code in value.split(",") if code.strip()]
    for code in codes:
        if not _LIGAND.fullmatch(code):
            raise ToolInputError(
                f'{name} entry "{code}" is not a PDB chemical component ID.'
            )
    return ",".join(codes)


def _selection_key(key: str, field_name: str) -> str:
    """One left-hand side of a selection line: a contig, or a ligand code."""

    key = key.strip()
    if not key:
        raise ToolInputError(f"{field_name} has a line with no residue or ligand.")
    parts = [part.strip() for part in key.split(",") if part.strip()]
    if all(_SEGMENT.fullmatch(part) for part in parts):
        return ",".join(parts)
    if _LIGAND.fullmatch(key):
        return key
    raise ToolInputError(
        f'{field_name} entry "{key}" is not a residue selection such as A108, '
        "A2-10, or a ligand code such as NAI."
    )


def _atom_value(value: str, field_name: str) -> str:
    """One right-hand side: a shorthand, an atom list, or nothing at all."""

    value = value.strip()
    if not value or value.upper() in _ATOM_SHORTHANDS:
        return value.upper()
    atoms = [atom.strip() for atom in value.split(",") if atom.strip()]
    for atom in atoms:
        if not _ATOM.fullmatch(atom):
            raise ToolInputError(f'{field_name} atom "{atom}" is not an atom name.')
    return ",".join(atoms)


def _selection(payload: dict[str, Any], name: str, *, required: bool = False):
    """Accept RFD3 InputSelection JSON/contigs and the form's residue: atoms syntax."""

    value = payload.get(name, "")
    if value is None or value == "":
        if required:
            raise ToolInputError(f"{name} is required.")
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise ToolInputError(f"{name} must map residue selections to atom strings.")
        return {_selection_key(k, name): _atom_value(v, name) for k, v in value.items()}
    raw = text(payload, name, required=required, max_length=20_000)
    if not raw:
        return None
    if raw.startswith(("{", "[", '"')) or raw.lower() in {"true", "false", "null"}:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolInputError(f"{name} is not valid JSON: {exc}.") from exc
        if not isinstance(parsed, (dict, bool, str)):
            raise ToolInputError(
                f"{name} must be a contig, boolean or atom-selection object."
            )
        if isinstance(parsed, str):
            return _selection_key(parsed, name)
        return _selection({name: parsed}, name, required=required)
    if ":" not in raw:
        return _selection_key(",".join(raw.splitlines()), name)
    selection: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ToolInputError(
                f'{name} line "{line}" must be "residue: atoms", for example "A108: ND2,CG".'
            )
        key, value = line.split(":", 1)
        selection[_selection_key(key, name)] = _atom_value(value, name)
    if required and not selection:
        raise ToolInputError(f"{name} must name at least one residue.")
    return selection


def _unindex(payload: dict[str, Any], name: str, *, required: bool = False):
    value = _selection(payload, name, required=required)
    if isinstance(value, bool):
        raise ToolInputError(
            f"{name} must be a contig string or atom-selection object."
        )
    return value


def _hotspots(payload: dict[str, Any], name: str):
    """Hotspots, as a residue-level contig or an atom-level dictionary.

    Both forms are accepted by `select_hotspots`, and which one is produced
    depends on whether any line named atoms: a bare list of residues stays a
    contig string, which is what the binder-design examples use, and a single
    atom-level line promotes the whole selection to a dictionary.
    """

    value = payload.get(name, "")
    if isinstance(value, (dict, bool)) or str(value).lstrip().startswith("{"):
        return _selection(payload, name)
    raw = text(payload, name, required=False, max_length=5_000)
    residues: list[str] = []
    atoms: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            atoms[_selection_key(key, name)] = _atom_value(value, name)
        else:
            residues.append(_selection_key(line, name))
    if not atoms:
        return ",".join(residues)
    for residue in residues:
        atoms.setdefault(residue, "ALL")
    return atoms


def _ori_token(payload: dict[str, Any], name: str) -> list[float] | None:
    value = text(payload, name, required=False, max_length=100)
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ToolInputError(f"{name} must be three comma-separated numbers.")
    try:
        coordinates = [float(part) for part in parts]
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("non-finite coordinate")
        return coordinates
    except ValueError as exc:
        raise ToolInputError(f"{name} must be three numbers, e.g. 24,20,10.") from exc


def _write_structure(
    workdir: Path, payload: dict[str, Any], name: str, *, base: str = "input"
) -> str:
    """Write a pasted structure, picking .cif or .pdb by what was actually pasted.

    RFD3 dispatches its parser on the suffix, so an mmCIF saved as .pdb is read
    by the PDB parser and fails on its first line.
    """

    content = text(payload, name, required=True, max_length=5_000_000)
    if content.startswith("bio-tools://"):
        prefix = "bio-tools://rfd3/"
        if not content.startswith(prefix) or not content.endswith((".pdb", ".cif")):
            raise ToolInputError("Unknown bundled RFD3 structure.")
        try:
            content = bio_tools.catalog_input_text("rfd3", content)
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
    suffix = ".cif" if content.lstrip().startswith("data_") else ".pdb"
    path = workdir / f"{base}{suffix}"
    path.write_text(content + "\n", encoding="utf-8")
    return str(path)


def _monomer(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    return {
        "length": _length(payload, "monomer_length"),
        **_non_loopy(payload, "monomer_non_loopy"),
    }


def _non_loopy(payload: dict[str, Any], name: str) -> dict[str, bool]:
    if payload.get(name, True) in (None, ""):
        return {}
    return {"is_non_loopy": boolean(payload, name, True)}


def _binder_design(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    binder = _contig(payload, "binder_length")
    target = _contig(payload, "binder_target_contig")
    specification: dict[str, Any] = {
        "input": _write_structure(workdir, payload, "binder_target_file"),
        "contig": f"{binder},/0,{target}",
        **_non_loopy(payload, "binder_non_loopy"),
    }
    hotspots = _hotspots(payload, "binder_hotspots")
    if hotspots:
        specification["select_hotspots"] = hotspots
        # With hotspots given, placing the origin token off their centre of mass
        # is what aims the binder at the site rather than at the whole target.
        specification["infer_ori_strategy"] = "hotspots"
    return specification


def _small_molecule(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    ligands = _ligands(payload, "small_molecule_ligand", required=True)
    specification: dict[str, Any] = {
        "input": _write_structure(workdir, payload, "small_molecule_input_file"),
        "ligand": ligands,
        "length": _length(payload, "small_molecule_length"),
    }
    if not boolean(payload, "small_molecule_fix_ligand", False):
        # An empty selection unfixes a ligand's atoms, letting the pocket be
        # built around a ligand the sampler may still move.
        specification["select_fixed_atoms"] = {code: "" for code in ligands.split(",")}
    buried = _selection(payload, "small_molecule_buried")
    if buried is not None:
        specification["select_buried"] = buried
    exposed = _selection(payload, "small_molecule_exposed")
    if exposed is not None:
        specification["select_exposed"] = exposed
    return specification


def _nucleic_acid(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    specification: dict[str, Any] = {
        "input": _write_structure(workdir, payload, "nucleic_acid_input_file"),
        "contig": _contig(payload, "nucleic_acid_contig"),
        **_non_loopy(payload, "nucleic_acid_non_loopy"),
    }
    length = _length(payload, "nucleic_acid_length", required=False)
    if length is not None:
        specification["length"] = length
    ori_token = _ori_token(payload, "nucleic_acid_ori_token")
    if ori_token is not None:
        specification["ori_token"] = ori_token
    unindex = _unindex(payload, "nucleic_acid_unindex")
    if unindex is not None:
        specification["unindex"] = unindex
    fixed = _selection(payload, "nucleic_acid_fixed_atoms")
    if fixed is not None:
        specification["select_fixed_atoms"] = fixed
    return specification


def _motif_scaffolding(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    specification: dict[str, Any] = {
        "input": _write_structure(workdir, payload, "motif_input_file"),
        "contig": _contig(payload, "motif_contig"),
    }
    length = _length(payload, "motif_length", required=False)
    if length is not None:
        specification["length"] = length
    fixed = _selection(payload, "motif_fixed_atoms")
    if fixed is not None:
        specification["select_fixed_atoms"] = fixed
    if boolean(payload, "motif_redesign_sidechains", False):
        specification["redesign_motif_sidechains"] = True
    return specification


def _enzyme(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    specification: dict[str, Any] = {
        "input": _write_structure(workdir, payload, "enzyme_input_file"),
        "unindex": _unindex(payload, "enzyme_unindex", required=True),
        "length": _length(payload, "enzyme_length"),
    }
    fixed = _selection(payload, "enzyme_fixed_atoms")
    if fixed is not None:
        specification["select_fixed_atoms"] = fixed
    ligands = _ligands(payload, "enzyme_ligand")
    if ligands:
        specification["ligand"] = ligands
    return specification


def _symmetry(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    group = text(payload, "symmetry_id", max_length=8)
    if not _SYMMETRY.fullmatch(group):
        raise ToolInputError(
            f'symmetry_id "{group}" must be a supported cyclic or dihedral group, such as C2 or D4.'
        )
    symmetry: dict[str, Any] = {"id": group.upper()}
    specification: dict[str, Any] = {
        "symmetry": symmetry,
        **_non_loopy(payload, "symmetry_non_loopy"),
    }
    length = _length(payload, "symmetry_length", required=False)
    if length is not None:
        specification["length"] = length
    contig = _contig(payload, "symmetry_contig", required=False)
    if contig:
        specification["contig"] = contig
    if length is None and not contig:
        raise ToolInputError(
            "Symmetry requires length or contig for one asymmetric unit."
        )
    unsym = text(payload, "symmetry_is_unsym_motif", required=False, max_length=5_000)
    if unsym:
        symmetry["is_unsym_motif"] = _selection_key(unsym, "symmetry_is_unsym_motif")

    structure = text(
        payload, "symmetry_input_file", required=False, max_length=5_000_000
    )
    if not structure:
        return specification

    specification["input"] = _write_structure(workdir, payload, "symmetry_input_file")
    symmetry["is_symmetric_motif"] = boolean(payload, "symmetry_symmetric_motif", True)
    if not symmetry["is_symmetric_motif"]:
        raise ToolInputError(
            "RFD3 requires input motifs pre-symmetrized around the origin (is_symmetric_motif=true)."
        )
    ligands = _ligands(payload, "symmetry_ligand")
    if ligands:
        specification["ligand"] = ligands
    unindex = _unindex(payload, "symmetry_unindex")
    if unindex is not None:
        specification["unindex"] = unindex
    fixed = _selection(payload, "symmetry_fixed_atoms")
    if fixed is not None:
        specification["select_fixed_atoms"] = fixed
    return specification


def _partial_diffusion(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    specification: dict[str, Any] = {
        "input": _write_structure(workdir, payload, "partial_input_file"),
        "partial_t": decimal(payload, "partial_t", default=10, minimum=0, maximum=50),
    }
    contig = _contig(payload, "partial_contig", required=False)
    if contig:
        specification["contig"] = contig
    unindex = _unindex(payload, "partial_unindex")
    if unindex is not None:
        specification["unindex"] = unindex
    ligands = _ligands(payload, "partial_ligand")
    if ligands:
        specification["ligand"] = ligands
    fixed = _selection(payload, "partial_fixed_atoms")
    if fixed is not None:
        specification["select_fixed_atoms"] = fixed
    return specification


# Only path-free parser options are exposed to requests. In particular,
# cache_dir must never let a submitted specification read or write server files.
_PARSER_OPTIONS = {
    "load_from_cache",
    "save_to_cache",
    "fix_arginines",
    "add_missing_atoms",
    "remove_ccds",
    "hydrogen_policy",
    "extra_fields",
}


def _input_document(raw: Any, *, yaml_allowed: bool = False):
    if isinstance(raw, dict):
        parsed = raw
    elif not isinstance(raw, str) or len(raw) > 200_000:
        raise ToolInputError(
            "Inputs must be a JSON/YAML document of at most 200,000 characters."
        )
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if not yaml_allowed:
                raise ToolInputError(f"Invalid JSON: {exc}.") from exc
            import yaml

            try:
                parsed = yaml.safe_load(raw)
            except yaml.YAMLError as yaml_exc:
                raise ToolInputError(
                    f"Inputs are not valid JSON or YAML: {yaml_exc}."
                ) from yaml_exc
    try:
        encoded = json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(
            "Inputs must use JSON-compatible values and finite numbers. Quote YAML dates as strings."
        ) from exc
    if len(encoded) > 200_000:
        raise ToolInputError("Input document exceeds 200,000 characters.")
    return json.loads(encoded)


def _custom_specification(
    payload: dict[str, Any], workdir: Path, *, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw = payload.get("inputs_file") or payload.get("inputs")
    # Older clients used specification for the complete named input document.
    if raw is None and "inputs" not in payload:
        raw = payload.get("specification")
    if raw is None:
        parsed = None
    elif isinstance(raw, dict):
        parsed = json.loads(json.dumps(raw))
    else:
        parsed = _input_document(raw, yaml_allowed=True)
    if parsed is None and raw in ("null", None):
        parsed = {safe_name(payload, default="design"): {}}
    if not isinstance(parsed, dict) or not parsed:
        raise ToolInputError("inputs must be an object of named design configurations.")
    for index, (key, entry) in enumerate(parsed.items()):
        _filename(key, "Input name")
        if not isinstance(entry, dict):
            raise ToolInputError(f'inputs entry "{key}" must be an object.')
        entry.update(overrides or {})
        if entry.get("atom_array_input") is not None:
            raise ToolInputError(
                "atom_array_input is not supported; upload a structure instead."
            )
        parser_args = entry.get("cif_parser_args")
        if parser_args is not None:
            if not isinstance(parser_args, dict) or set(parser_args) - _PARSER_OPTIONS:
                raise ToolInputError(
                    "cif_parser_args contains an unsupported option or a local cache path."
                )
        # The parser only reads extra_fields out of mmCIF, and merely logs that it dropped
        # them otherwise. Refuse instead, so a request never looks like it took effect.
        extra_fields = bool(parser_args and parser_args.get("extra_fields"))
        source = entry.get("input")
        if source in (None, ""):
            selections = [
                name
                for name in (
                    "unindex",
                    "select_fixed_atoms",
                    "select_unfixed_sequence",
                    "select_buried",
                    "select_partially_buried",
                    "select_exposed",
                    "select_hbond_donor",
                    "select_hbond_acceptor",
                    "select_hotspots",
                )
                if isinstance(entry.get(name), (str, dict)) and entry[name]
            ]
            if (
                selections
                or entry.get("partial_t") is not None
                or re.search(r"[A-Za-z]", str(entry.get("contig", "")))
            ):
                raise ToolInputError(
                    f'Input structure is required for design "{key}"'
                    + (f" to use {', '.join(selections)}" if selections else "")
                    + ". Choose a preset with an included structure or upload a PDB/mmCIF."
                )
            if extra_fields:
                raise ToolInputError(
                    "cif_parser_args.extra_fields reads columns from an mmCIF structure, "
                    f'and design "{key}" has no input structure.'
                )
            entry.pop("input", None)
            continue
        if source == "uploaded":
            field = "spec_input_file"
            values = payload
        elif isinstance(source, str) and source.startswith("bio-tools://rfd3/"):
            field = "input"
            values = {field: source}
        else:
            raise ToolInputError(
                f'specification entry "{key}" input must be a bundled bio-tools://rfd3/ '
                'structure or "uploaded". Server paths and URLs are not accepted.'
            )
        entry["input"] = _write_structure(workdir, values, field, base=f"input_{index}")
        if extra_fields and not entry["input"].endswith(".cif"):
            raise ToolInputError(
                "cif_parser_args.extra_fields reads columns from an mmCIF structure, and "
                f'design "{key}" supplies PDB. Paste the mmCIF form or drop extra_fields.'
            )
    return parsed


def _filename(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value)
        or value in {".", ".."}
    ):
        raise ToolInputError(
            f"{label} must be a name using letters, digits, dots, underscores or hyphens."
        )
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        if value.lstrip().startswith("["):
            value = _input_document(value)
        else:
            value = [
                item.strip()
                for item in value.replace("\n", ",").split(",")
                if item.strip()
            ]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ToolInputError(
            f"{label} must be an array or comma-separated list of strings."
        )
    return list(dict.fromkeys(value))


def _sampler_options(payload: dict[str, Any], symmetric: bool) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for name, default, minimum in (
        ("cfg_t_max", None, 0),
        ("cfg_scale", 1.5, 0),
        ("s_trans", 1.0, 0),
        ("noise_scale", 1.003, 0),
        ("p", 7.0, 0.001),
        ("gamma_min", 1.0, 0),
        ("s_jitter_origin", 0.0, 0),
    ):
        key = f"inference_sampler.{name}"
        raw = payload.get(key, default)
        if raw in (None, "") and default is None:
            options[key] = None
            continue
        if raw in (None, ""):
            raw = default
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ToolInputError(f"{key} must be a number.") from exc
        if not math.isfinite(value) or value < minimum:
            raise ToolInputError(f"{key} must be finite and at least {minimum}.")
        options[key] = value
    recycle = payload.get("inference_sampler.n_recycle")
    options["inference_sampler.n_recycle"] = (
        None
        if recycle in (None, "")
        else integer(
            payload, "inference_sampler.n_recycle", default=2, minimum=0, maximum=1000
        )
    )
    if (
        recycle not in (None, "")
        and float(recycle) != options["inference_sampler.n_recycle"]
    ):
        raise ToolInputError("inference_sampler.n_recycle must be an integer.")
    center = str(payload.get("inference_sampler.center_option", "all"))
    if center not in {"all", "motif", "diffuse"}:
        raise ToolInputError(
            "inference_sampler.center_option must be all, motif or diffuse."
        )
    options["inference_sampler.center_option"] = center
    features = _string_list(
        payload.get(
            "inference_sampler.cfg_features",
            "active_donor,active_acceptor,ref_atomwise_rasa",
        ),
        "inference_sampler.cfg_features",
    )
    if set(features) - {"active_donor", "active_acceptor", "ref_atomwise_rasa"}:
        raise ToolInputError("Unknown classifier-free guidance feature.")
    options["inference_sampler.cfg_features"] = features
    for name in ("use_classifier_free_guidance", "allow_realignment"):
        key = f"inference_sampler.{name}"
        options[key] = boolean(payload, key, False)
    if symmetric and options["inference_sampler.use_classifier_free_guidance"]:
        raise ToolInputError(
            "The symmetry sampler does not support classifier-free guidance."
        )
    kind = str(payload.get("inference_sampler.kind", "auto"))
    expected = "symmetry" if symmetric else "default"
    if kind not in {"auto", expected}:
        raise ToolInputError(
            f"Selected inputs require inference_sampler.kind={expected}."
        )
    options["inference_sampler.kind"] = expected
    return options


def _conditioning(payload: dict[str, Any], specification: dict[str, Any]) -> None:
    for key in (
        "select_unfixed_sequence",
        "select_partially_buried",
        "select_hbond_acceptor",
        "select_hbond_donor",
    ):
        selection = _selection(payload, key)
        if selection is None:
            continue
        if key.startswith("select_hbond") and not isinstance(selection, dict):
            raise ToolInputError(f"{key} requires an atom-selection object.")
        if key == "select_partially_buried" and isinstance(selection, bool):
            raise ToolInputError(f"{key} requires a contig or atom-selection object.")
        specification[key] = selection
    if boolean(payload, "allow_ligand_on_existing_chain", False):
        specification["allow_ligand_on_existing_chain"] = True


def _native_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    """The documented InputSpecification contract, without workflow-specific aliases.

    Reference: https://rosettacommons.github.io/foundry/models/rfd3/input.html#inputspecification-fields
    """

    entry: dict[str, Any] = {}
    for key in ("contig", "ligand"):
        value = (
            _contig(payload, key, required=False)
            if key == "contig"
            else _ligands(payload, key)
        )
        if value:
            entry[key] = value
    length = _length(payload, "length", required=False)
    if length is not None:
        entry["length"] = length
    unindex = _unindex(payload, "unindex")
    if unindex is not None:
        entry["unindex"] = unindex
    for key in (
        "select_fixed_atoms",
        "select_unfixed_sequence",
        "select_buried",
        "select_partially_buried",
        "select_exposed",
        "select_hbond_donor",
        "select_hbond_acceptor",
        "select_hotspots",
    ):
        value = _selection(payload, key)
        if value is None:
            continue
        if key.startswith("select_hbond") and not isinstance(value, dict):
            raise ToolInputError(f"{key} requires an atom-selection object.")
        if key in {
            "select_buried",
            "select_partially_buried",
            "select_exposed",
        } and isinstance(value, bool):
            raise ToolInputError(f"{key} requires a contig or atom-selection object.")
        entry[key] = value
    for key in ("symmetry", "cif_parser_args", "extra"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        value = _input_document(raw)
        if not isinstance(value, dict):
            raise ToolInputError(f"{key} must be a JSON object.")
        if value:
            entry[key] = value
    entry["dialect"] = integer(payload, "dialect", default=2, minimum=1, maximum=2)
    for key, default in (
        ("plddt_enhanced", True),
        ("redesign_motif_sidechains", False),
    ):
        entry[key] = boolean(payload, key, default)
    if boolean(payload, "allow_ligand_on_existing_chain", False):
        entry["allow_ligand_on_existing_chain"] = True
    if payload.get("is_non_loopy") not in (None, ""):
        entry["is_non_loopy"] = boolean(payload, "is_non_loopy")
    strategy = text(payload, "infer_ori_strategy", required=False)
    if strategy:
        if strategy not in {"com", "hotspots"}:
            raise ToolInputError("infer_ori_strategy must be com or hotspots.")
        entry["infer_ori_strategy"] = strategy
    origin = payload.get("ori_token")
    if origin not in (None, ""):
        if isinstance(origin, str) and not origin.lstrip().startswith("["):
            origin = _ori_token(payload, "ori_token")
        elif isinstance(origin, str):
            origin = _input_document(origin)
        if not isinstance(origin, list) or len(origin) != 3:
            raise ToolInputError("ori_token must contain three coordinates.")
        try:
            origin = [float(v) for v in origin]
        except (ValueError, TypeError) as exc:
            raise ToolInputError("ori_token must contain three numbers.") from exc
        if not all(math.isfinite(v) for v in origin):
            raise ToolInputError("ori_token coordinates must be finite.")
        entry["ori_token"] = origin
    noise = payload.get("partial_t")
    if noise not in (None, ""):
        entry["partial_t"] = decimal(
            payload, "partial_t", default=10, minimum=0, maximum=float("inf")
        )
        if not math.isfinite(entry["partial_t"]):
            raise ToolInputError("partial_t must be finite.")
    structure = text(payload, "input", required=False, max_length=5_000_000)
    if structure:
        entry["input"] = (
            structure if structure.startswith("bio-tools://") else "uploaded"
        )
    if "partial_t" in entry and not structure:
        raise ToolInputError("Partial diffusion requires an input structure.")
    return entry


def _validate_symmetry(specification: dict[str, Any], gamma_0: float) -> bool:
    entries = list(specification.values())
    symmetric = [entry for entry in entries if entry.get("symmetry")]
    if not symmetric:
        return False
    if len(symmetric) != len(entries):
        raise ToolInputError(
            "Run symmetric and non-symmetric specifications separately; they use different samplers."
        )
    for entry in symmetric:
        symmetry = entry["symmetry"]
        if not isinstance(symmetry, dict) or not _SYMMETRY.fullmatch(
            str(symmetry.get("id", ""))
        ):
            raise ToolInputError(
                "Symmetry id must be a C or D group, such as C2 or D4."
            )
        symmetry["id"] = symmetry["id"].upper()
        if symmetry.get("is_symmetric_motif", True) is not True:
            raise ToolInputError("RFD3 only supports is_symmetric_motif=true.")
    if gamma_0 <= 0.5:
        raise ToolInputError(
            "Symmetry sampling requires gamma_0 greater than 0.5 (default 0.6)."
        )
    return True


_BUILDERS = {
    "monomer": _monomer,
    "binder_design": _binder_design,
    "small_molecule": _small_molecule,
    "nucleic_acid": _nucleic_acid,
    "motif_scaffolding": _motif_scaffolding,
    "enzyme": _enzyme,
    "symmetry": _symmetry,
    "partial_diffusion": _partial_diffusion,
}


def _checkpoint() -> Path:
    filename = _CHECKPOINT_NAME
    configured = os.getenv("RFD3_CHECKPOINT_DIR")
    if configured:
        root = Path(configured).expanduser().resolve()
        path = (root / filename).resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    raise ToolUnavailable(
        f"Configure RFD3_CHECKPOINT_DIR with the directory holding {filename}."
    )


def run(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    for old in ("num_timesteps", "step_scale", "gamma_0"):
        if old in payload:
            payload.setdefault(f"inference_sampler.{old}", payload[old])
    if (
        payload.get("task") == "spec"
        and "inputs" not in payload
        and not payload.get("inputs_file")
        and payload.get("specification")
    ):
        candidate = _input_document(payload["specification"])
        if (
            isinstance(candidate, dict)
            and candidate
            and all(isinstance(entry, dict) for entry in candidate.values())
        ):
            payload["inputs"] = payload.pop("specification")
    payload = preset_payload("rfd3", payload)
    name = safe_name(payload, default="rfd3-demo")
    task = str(payload.get("task") or "parameters")
    if task not in set(_BUILDERS) | {"spec", "parameters"}:
        raise ToolInputError(f"Unsupported task: {task}.")
    input_mode = str(
        payload.get("input_mode") or ("text" if task == "spec" else "parameters")
    )
    if input_mode not in {"upload", "text", "parameters"}:
        raise ToolInputError("input_mode must be upload, text or parameters.")
    if input_mode == "upload":
        payload["inputs"] = text(payload, "inputs_file", max_length=200_000)
        payload["inputs_file"] = ""
    elif input_mode == "text":
        # An earlier file choice must never override the text currently displayed.
        payload["inputs_file"] = ""
    elif task == "spec":
        raise ToolInputError(
            "Choose text or upload input mode for a custom input document."
        )

    batches = integer(payload, "n_batches", default=1, minimum=1, maximum=100)
    batch_size = integer(
        payload, "diffusion_batch_size", default=8, minimum=1, maximum=64
    )
    timesteps = integer(
        payload,
        "inference_sampler.num_timesteps",
        default=200,
        minimum=20,
        maximum=1_000,
    )
    step_scale = decimal(
        payload, "inference_sampler.step_scale", default=1.5, minimum=0.1, maximum=10
    )
    gamma_0 = decimal(
        payload, "inference_sampler.gamma_0", default=0.6, minimum=0, maximum=1
    )
    low_memory = boolean(payload, "low_memory_mode", False)
    trajectories = boolean(payload, "dump_trajectories", False)

    rfd3 = tool_script("rfd3", "rfd3", "RFD3_EXECUTABLE")
    checkpoint = _checkpoint()
    overrides = _input_document(payload.get("specification") or "{}")
    if not isinstance(overrides, dict):
        raise ToolInputError("specification overrides must be one JSON object.")

    with tempfile.TemporaryDirectory(prefix="bio-web-rfd3-") as temporary:
        workdir = Path(temporary)
        if input_mode != "parameters":
            specification = _custom_specification(payload, workdir, overrides=overrides)
        elif task == "parameters" or any(
            key in payload for key in ("input", "contig", "length")
        ):
            entry = _native_parameters(payload)
            specification = _custom_specification(
                {"inputs": {name: entry}, "spec_input_file": payload.get("input", "")},
                workdir,
                overrides=overrides,
            )
        else:
            entry = _BUILDERS[task](payload, workdir)
            _conditioning(payload, entry)
            if overrides:
                override_dir = workdir / "overrides"
                override_dir.mkdir()
                validated = _custom_specification(
                    {**payload, "inputs": {"overrides": overrides}, "inputs_file": ""},
                    override_dir,
                )
                entry.update(validated["overrides"])
            specification = {name: entry}
        subset = _string_list(payload.get("json_keys_subset"), "json_keys_subset")
        if payload.get("json_keys_subset") not in (None, "") and not subset:
            raise ToolInputError(
                "json_keys_subset must contain at least one name; leave it blank to run all inputs."
            )
        if set(subset) - specification.keys():
            raise ToolInputError("json_keys_subset contains unknown input names.")
        selected = (
            {key: specification[key] for key in subset} if subset else specification
        )
        symmetric = _validate_symmetry(selected, gamma_0)
        inputs = workdir / "inputs.json"
        inputs.write_text(json.dumps(specification, indent=2), encoding="utf-8")

        output = workdir / "outputs" / "output"
        command = [
            rfd3,
            "design",
            f"out_dir={output}",
            f"inputs={inputs}",
            f"ckpt_path={checkpoint}",
            f"n_batches={batches}",
            f"diffusion_batch_size={batch_size}",
            f"inference_sampler.num_timesteps={timesteps}",
            f"inference_sampler.step_scale={step_scale}",
            f"inference_sampler.gamma_0={gamma_0}",
            f"low_memory_mode={low_memory}",
            f"dump_trajectories={trajectories}",
        ]
        options = _sampler_options(payload, symmetric)
        options["json_keys_subset"] = subset or None
        for key, default in (
            ("prevalidate_inputs", False),
            ("cleanup_guideposts", True),
            ("cleanup_virtual_atoms", True),
            ("read_sequence_from_sequence_head", True),
            ("output_full_json", True),
            ("dump_prediction_metadata_json", True),
            ("align_trajectory_structures", False),
        ):
            options[key] = boolean(payload, key, default)
        prefix = text(payload, "global_prefix", required=False, max_length=160)
        if prefix:
            options["global_prefix"] = (
                "" if prefix == '""' else _filename(prefix, "global_prefix")
            )
        for key, value in options.items():
            # Lists and strings are quoted for Hydra's override parser, not a shell.
            command.append(f"{key}={json.dumps(value, separators=(',', ':'))}")

        # Hydra warns that its relative "configs" search path is unavailable, because the
        # working directory is this isolated run rather than the RFD3 install. Harmless:
        # every setting is passed as an override above, never read from a config group.
        result = run_command(command, cwd=workdir)
        generated = readable_files(output)
    return {
        "status": "completed",
        "job_name": name,
        "task": task,
        "input_mode": input_mode,
        "specification": specification,
        "generated_files": generated,
        **result,
    }


def check_status() -> ToolStatus:
    try:
        rfd3 = tool_script("rfd3", "rfd3", "RFD3_EXECUTABLE")
        _checkpoint()
    except ToolUnavailable as exc:
        return ToolStatus(CheckResult.NOT_INSTALLED, str(exc))
    return require_gpu_status(
        probe_cli([rfd3]),
        torch_device(environment_python("rfd3")),
        "RFdiffusion3",
    )
