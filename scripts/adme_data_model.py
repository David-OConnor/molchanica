"""Shared molecular standardization for canonical therapeutic observations.

The assayed form and the ML parent are deliberately separate.  Nothing in this
module overwrites the structure supplied by a source dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

STRUCTURE_STANDARDIZATION = "molchanica_parent_v1"
RDKit_VERSION = rdBase.rdkitVersion
_UNCHARGER = rdMolStandardize.Uncharger()


@dataclass(frozen=True)
class StandardizedStructure:
    assayed_smiles: str
    assayed_canonical_smiles: str | None
    assayed_inchi: str | None
    assayed_inchi_key: str | None
    assayed_component_count: int | None
    assayed_form_kind: str
    parent_smiles: str
    parent_inchi: str
    parent_inchi_key: str
    parent_scaffold: str
    removed_components: list[str]
    structure_override_applied: bool
    standardization_warnings: list[str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structure_overrides(path: Path | None) -> tuple[dict, str | None]:
    if path is None:
        return {}, None
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("datasets"), dict
    ):
        raise ValueError(f"Invalid structure override file: {path}")
    return payload["datasets"], sha256_bytes(raw)


def validated_override(
    overrides: dict,
    dataset: str,
    row_id: int,
    original_smiles: str,
) -> dict | None:
    override = overrides.get(dataset, {}).get(str(row_id))
    if override is None:
        return None
    required = ("original_smiles", "replacement_smiles", "reason", "source")
    missing = [key for key in required if not override.get(key)]
    if missing:
        raise ValueError(f"{dataset} structure override {row_id} is missing {missing}")
    if override["original_smiles"] != original_smiles:
        raise ValueError(
            f"{dataset} structure override {row_id} expected "
            f"{override['original_smiles']!r}, found {original_smiles!r}"
        )
    return override


def _parent_mol(mol: Chem.Mol) -> Chem.Mol:
    parent = rdMolStandardize.Cleanup(Chem.Mol(mol))
    parent = rdMolStandardize.FragmentParent(parent)
    parent = _UNCHARGER.uncharge(parent)
    Chem.SanitizeMol(parent)
    if parent.GetNumAtoms() == 0:
        raise ValueError("standardization produced an empty parent")
    return parent


def _scaffold_key(parent: Chem.Mol, parent_inchi_key: str) -> str:
    scaffold_mol = Chem.Mol(parent)
    Chem.RemoveStereochemistry(scaffold_mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=scaffold_mol,
        includeChirality=False,
    )
    return f"MURCKO:{scaffold}" if scaffold else f"ACYCLIC:{parent_inchi_key}"


def _fragment_parent_key(fragment: Chem.Mol) -> str | None:
    try:
        return inchi.MolToInchiKey(_parent_mol(fragment)) or None
    except Exception:
        return None


def standardize_structure(
    original_smiles: str,
    replacement_smiles: str | None = None,
) -> StandardizedStructure:
    """Preserve the assayed form and derive a separate, neutralized parent.

    ``replacement_smiles`` is used only when a documented source correction is
    needed to derive the parent.  The original assayed SMILES is still retained.
    """

    original_smiles = original_smiles.strip()
    if not original_smiles:
        raise ValueError("empty assayed SMILES")

    warnings: list[str] = []
    assayed = Chem.MolFromSmiles(original_smiles)
    parent_input = assayed
    override_applied = replacement_smiles is not None
    if parent_input is None:
        if replacement_smiles is None:
            raise ValueError("RDKit could not parse the assayed SMILES")
        parent_input = Chem.MolFromSmiles(replacement_smiles)
        if parent_input is None:
            raise ValueError("RDKit could not parse the documented replacement SMILES")
        warnings.append(
            "original_assayed_structure_unparseable; documented correction used for parent"
        )

    parent = _parent_mol(parent_input)
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    parent_inchi = inchi.MolToInchi(parent)
    parent_inchi_key = inchi.MolToInchiKey(parent)
    if not parent_inchi or not parent_inchi_key:
        raise ValueError("RDKit could not calculate the parent InChI/InChIKey")

    assayed_canonical_smiles = None
    assayed_inchi = None
    assayed_inchi_key = None
    assayed_component_count = None
    removed_components: list[str] = []
    if assayed is not None:
        assayed_canonical_smiles = Chem.MolToSmiles(
            assayed,
            canonical=True,
            isomericSmiles=True,
        )
        assayed_inchi = inchi.MolToInchi(assayed) or None
        assayed_inchi_key = inchi.MolToInchiKey(assayed) or None
        fragments = list(Chem.GetMolFrags(assayed, asMols=True, sanitizeFrags=True))
        assayed_component_count = len(fragments)
        if len(fragments) > 1:
            retained_parent = False
            for fragment in fragments:
                fragment_smiles = Chem.MolToSmiles(
                    fragment,
                    canonical=True,
                    isomericSmiles=True,
                )
                if (
                    not retained_parent
                    and _fragment_parent_key(fragment) == parent_inchi_key
                ):
                    retained_parent = True
                else:
                    removed_components.append(fragment_smiles)

    if assayed is None:
        assayed_form_kind = "unparseable_original_with_documented_parent"
    elif assayed_component_count == 1:
        assayed_form_kind = "single_component"
    else:
        assayed_form_kind = "multicomponent"

    return StandardizedStructure(
        assayed_smiles=original_smiles,
        assayed_canonical_smiles=assayed_canonical_smiles,
        assayed_inchi=assayed_inchi,
        assayed_inchi_key=assayed_inchi_key,
        assayed_component_count=assayed_component_count,
        assayed_form_kind=assayed_form_kind,
        parent_smiles=parent_smiles,
        parent_inchi=parent_inchi,
        parent_inchi_key=parent_inchi_key,
        parent_scaffold=_scaffold_key(parent, parent_inchi_key),
        removed_components=removed_components,
        structure_override_applied=override_applied,
        standardization_warnings=warnings,
    )


def standardize_parent(smiles: str) -> tuple[str, str]:
    """Compatibility helper for split generation."""

    structure = standardize_structure(smiles)
    return structure.parent_inchi_key, structure.parent_scaffold


def validate_parent_sdf(path: Path, expected_parent_inchi_key: str) -> str | None:
    """Return an exclusion reason, or ``None`` when an SDF is the expected parent."""

    if not path.is_file() or path.stat().st_size == 0:
        return "missing_parent_sdf"
    try:
        mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
        if mol is None:
            return "unparseable_parent_sdf"
        if len(Chem.GetMolFrags(mol)) != 1:
            return "multicomponent_parent_sdf"
        actual_parent_key = inchi.MolToInchiKey(_parent_mol(mol))
        # PubChem 3D records can assign an arbitrary configuration to tetrahedral
        # centers that were unspecified in the source SMILES. For an achiral/
        # unspecified expected parent (UHFFFAOYSA), validate the InChIKey
        # connectivity block. When the source explicitly specifies stereo, require
        # the complete standardized parent InChIKey.
        if expected_parent_inchi_key[15:25] == "UHFFFAOYSA":
            if actual_parent_key[:14] != expected_parent_inchi_key[:14]:
                return "parent_sdf_connectivity_mismatch"
        elif actual_parent_key != expected_parent_inchi_key:
            if actual_parent_key[:14] == expected_parent_inchi_key[:14]:
                return "parent_sdf_stereochemistry_mismatch"
            return "parent_sdf_connectivity_mismatch"
    except Exception:
        return "unparseable_parent_sdf"
    return None


def validate_assayed_sdf(
    path: Path,
    expected_assayed_inchi_key: str | None,
    expected_component_count: int | None,
) -> str | None:
    """Return a reason, or ``None`` when an SDF is the exact assayed form."""

    if expected_assayed_inchi_key is None or expected_component_count is None:
        return "assayed_structure_unparseable"
    if not path.is_file() or path.stat().st_size == 0:
        return "missing_assayed_sdf"
    try:
        mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
        if mol is None:
            return "unparseable_assayed_sdf"
        if len(Chem.GetMolFrags(mol)) != expected_component_count:
            return "assayed_sdf_component_count_mismatch"
        actual_assayed_key = inchi.MolToInchiKey(mol)
        if expected_assayed_inchi_key[15:25] == "UHFFFAOYSA":
            if actual_assayed_key[:14] != expected_assayed_inchi_key[:14]:
                return "assayed_sdf_identity_mismatch"
        elif actual_assayed_key != expected_assayed_inchi_key:
            return "assayed_sdf_identity_mismatch"
    except Exception:
        return "unparseable_assayed_sdf"
    return None
