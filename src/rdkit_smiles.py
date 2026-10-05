"""Small JSON bridge for SMILES operations in an installed RDKit environment."""

import json
import sys

from rdkit import Chem
from rdkit.Chem import rdDepictor


def parse_smiles(request):
    molecule = Chem.MolFromSmiles(request["smiles"])
    if molecule is None:
        raise ValueError("RDKit could not parse the SMILES string")

    rdDepictor.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    atoms = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "atomic_number": atom.GetAtomicNum(),
                "x": position.x,
                "y": position.y,
                "z": position.z,
            }
        )

    bonds = []
    for bond in molecule.GetBonds():
        bonds.append(
            {
                "atom_0": bond.GetBeginAtomIdx(),
                "atom_1": bond.GetEndAtomIdx(),
                "order": bond.GetBondTypeAsDouble(),
            }
        )

    return {"atoms": atoms, "bonds": bonds}


def serialize_smiles(request):
    molecule = Chem.RWMol()
    for atomic_number in request["atoms"]:
        molecule.AddAtom(Chem.Atom(atomic_number))

    bond_types = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
        "quadruple": Chem.BondType.QUADRUPLE,
    }
    aromatic_atoms = set()
    for bond in request["bonds"]:
        bond_type = bond_types[bond["bond_type"]]
        molecule.AddBond(bond["atom_0"], bond["atom_1"], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            aromatic_atoms.add(bond["atom_0"])
            aromatic_atoms.add(bond["atom_1"])

    result = molecule.GetMol()
    for atom_index in aromatic_atoms:
        result.GetAtomWithIdx(atom_index).SetIsAromatic(True)
    Chem.SanitizeMol(result)
    result = Chem.RemoveHs(result)

    return {
        "smiles": Chem.MolToSmiles(
            result,
            canonical=True,
            isomericSmiles=True,
        )
    }


try:
    request = json.load(sys.stdin)
    if request["operation"] == "parse":
        response = parse_smiles(request)
    elif request["operation"] == "serialize":
        response = serialize_smiles(request)
    else:
        raise ValueError("Unknown RDKit SMILES operation")
except Exception as error:
    response = {"error": str(error)}

json.dump(response, sys.stdout)
