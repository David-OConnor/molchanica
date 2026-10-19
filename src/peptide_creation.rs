//! Initial backbone conformations for peptides built from Amber residue templates.

use std::io;

use bio_files::{BackboneSS, SecondaryStructure};
use dynamics::Dihedral;
use mol_defs::{
    bond_inference::create_hydrogen_bonds_single_mol,
    molecules::{
        AtomRole, amino_acid::AminoAcidTemplates, common::MoleculeCommon, peptide::MoleculePeptide,
    },
};
use na_seq::AminoAcid;

#[derive(Clone, Copy, Default, PartialEq)]
pub enum PeptideGeometry {
    #[default]
    Line,
    BetaSheet,
    AlphaHelix,
}

/// Keep the existing extended chain for `Line`; the other presets change only backbone
/// torsions, preserving template bond lengths, bond angles, and side-chain geometry.
pub fn build_peptide(
    seq: &[AminoAcid],
    templates: &AminoAcidTemplates,
    geometry: PeptideGeometry,
) -> io::Result<MoleculePeptide> {
    let mut mol = MoleculePeptide::from_seq(seq, templates)?;

    let (phi, psi, sec_struct) = match geometry {
        PeptideGeometry::Line => return Ok(mol),
        PeptideGeometry::BetaSheet => (-135.0_f64, 135.0_f64, SecondaryStructure::Sheet),
        PeptideGeometry::AlphaHelix => (-60.0_f64, -45.0_f64, SecondaryStructure::Helix),
    };

    let backbone: Vec<[usize; 3]> = mol
        .residues
        .iter()
        .map(|res| {
            let find = |role| {
                res.atoms
                    .iter()
                    .copied()
                    .find(|&i| mol.common.atoms[i].role == Some(role))
                    .ok_or_else(|| io::Error::other("Peptide template is missing a backbone atom"))
            };

            Ok([
                find(AtomRole::N_Backbone)?,
                find(AtomRole::C_Alpha)?,
                find(AtomRole::C_Prime)?,
            ])
        })
        .collect::<io::Result<_>>()?;

    for (i, &[n, ca, c]) in backbone.iter().enumerate() {
        // Proline's N–CA bond belongs to its ring; retain its template's phi.
        if i > 0 && seq[i] != AminoAcid::Pro {
            set_torsion(
                &mut mol.common,
                [backbone[i - 1][2], n, ca, c],
                phi.to_radians(),
            )?;
        }

        if let Some(next) = backbone.get(i + 1) {
            set_torsion(&mut mol.common, [n, ca, c, next[0]], psi.to_radians())?;
        }
    }

    for (i, &[n, ca, c]) in backbone.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| backbone[j]);
        let next = backbone.get(i + 1);

        mol.residues[i].dihedral = Some(Dihedral {
            φ: prev.map(|p| torsion(&mol.common, [p[2], n, ca, c])),
            ψ: next.map(|p| torsion(&mol.common, [n, ca, c, p[0]])),
            ω: prev.map(|p| torsion(&mol.common, [p[1], p[2], n, ca])),
            ..Default::default()
        });
    }

    if backbone.len() > 1 {
        mol.secondary_structure.push(BackboneSS {
            start_sn: mol.common.atoms[backbone[0][0]].serial_number,
            end_sn: mol.common.atoms.last().unwrap().serial_number,
            sec_struct,
        });
    }

    mol.bonds_hydrogen = create_hydrogen_bonds_single_mol(
        &mol.common.atoms,
        &mol.common.atom_posits,
        &mol.common.bonds,
    );

    Ok(mol)
}

/// Signed a–b–c–d torsion, with trans peptide bonds at ±π.
fn torsion(mol: &MoleculeCommon, [a, b, c, d]: [usize; 4]) -> f64 {
    let [a, b, c, d] = [a, b, c, d].map(|i| mol.atom_posits[i]);
    let axis = (c - b).to_normalized();
    let v = (a - b) - axis * (a - b).dot(axis);
    let w = (d - c) - axis * (d - c).dot(axis);

    axis.cross(v).dot(w).atan2(v.dot(w))
}

fn set_torsion(mol: &mut MoleculeCommon, atoms: [usize; 4], target: f64) -> io::Result<()> {
    let [_, b, c, _] = atoms;
    let bond_i = mol
        .bonds
        .iter()
        .position(|bond| {
            (bond.atom_0 == b && bond.atom_1 == c) || (bond.atom_0 == c && bond.atom_1 == b)
        })
        .ok_or_else(|| io::Error::other("Peptide backbone atoms are not bonded"))?;

    let delta = target - torsion(mol, atoms);
    mol.rotate_around_bond(bond_i, delta, None);

    Ok(())
}
