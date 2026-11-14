use bio_files::{ResidueEnd, ResidueType};
use egui::Color32;
use mol_defs::molecules::{
    Residue,
    nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType},
};
use na_seq::Nucleotide::{self, *};

#[derive(Clone, Copy, Default, PartialEq)]
pub enum ResidueGrouping {
    #[default]
    Nucleotide,
    BasePair,
    BasesAndBackbones,
    BaseAndBackbone,
}

pub fn nucleotide(letter: char, na_type: NucleicAcidType) -> Option<Nucleotide> {
    match letter.to_ascii_uppercase() {
        'A' => Some(A),
        'C' => Some(C),
        'G' => Some(G),
        'T' => Some(T),
        'U' if na_type == NucleicAcidType::Rna => Some(T),
        _ => None,
    }
}

pub fn base_letter(nt: Nucleotide, na_type: NucleicAcidType) -> char {
    match nt {
        A => 'A',
        C => 'C',
        G => 'G',
        T if na_type == NucleicAcidType::Rna => 'U',
        T => 'T',
    }
}

/// Sanger chromatogram colors, lightened for a dark background.
pub fn base_color(nt: Nucleotide) -> Color32 {
    match nt {
        A => Color32::from_rgb(20, 220, 20),
        C => Color32::from_rgb(130, 130, 255),
        T => Color32::from_rgb(255, 100, 100),
        G => Color32::from_rgb(200, 200, 200),
    }
}

pub fn is_backbone_atom(name: &str) -> bool {
    name.contains(['\'', '*'])
        || matches!(name, "P" | "OP1" | "OP2" | "OP3" | "O1P" | "O2P" | "O3P")
}

/// Regroup the residues produced by the builder, which orders each strand 5′ to 3′.
pub fn group_residues(mol: &mut MoleculeNucleicAcid, grouping: ResidueGrouping) {
    let nucleotides = std::mem::take(&mut mol.residues);
    let (first, second) = nucleotides.split_at(mol.seq.len());

    let nucleotide_name = |nt| {
        let letter = base_letter(nt, mol.na_type);
        match mol.na_type {
            NucleicAcidType::Dna => format!("D{letter}"),
            NucleicAcidType::Rna => letter.to_string(),
        }
    };
    // Split via atom names without borrowing the atoms while assigning their new indices.
    let backbone = if matches!(
        grouping,
        ResidueGrouping::BasesAndBackbones | ResidueGrouping::BaseAndBackbone
    ) {
        mol.common
            .atoms
            .iter()
            .map(|a| {
                a.type_in_res_general
                    .as_deref()
                    .is_some_and(is_backbone_atom)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut add = |name: String, atoms: Vec<usize>, end: ResidueEnd| {
        let i = mol.residues.len();
        let atom_sns = atoms
            .iter()
            .map(|&a| {
                let atom = &mut mol.common.atoms[a];
                atom.residue = Some(i);
                atom.serial_number
            })
            .collect();

        mol.residues.push(Residue {
            serial_number: i as u32 + 1,
            res_type: ResidueType::Other(name),
            atoms,
            atom_sns,
            dihedral: None,
            end,
        });
    };

    match grouping {
        ResidueGrouping::Nucleotide | ResidueGrouping::BaseAndBackbone => {
            let nts = mol
                .seq
                .iter()
                .copied()
                .chain(mol.seq.iter().rev().map(|nt| nt.complement()));
            for (res, nt) in nucleotides.iter().zip(nts) {
                let name = nucleotide_name(nt);
                if grouping == ResidueGrouping::BaseAndBackbone {
                    let (bases, backbone_atoms) =
                        res.atoms.iter().copied().partition(|&a| !backbone[a]);
                    add(name, bases, res.end);
                    add("Backbone".to_owned(), backbone_atoms, res.end);
                } else {
                    add(name, res.atoms.clone(), res.end);
                }
            }
        }
        ResidueGrouping::BasePair | ResidueGrouping::BasesAndBackbones => {
            let complements = second.iter().rev().map(Some).chain(std::iter::repeat(None));
            for ((res, &nt), comp) in first.iter().zip(&mol.seq).zip(complements) {
                let mut name = nucleotide_name(nt);
                let mut atoms = res.atoms.clone();
                let mut end = res.end;
                if let Some(comp) = comp {
                    name += &format!("-{}", nucleotide_name(nt.complement()));
                    atoms.extend_from_slice(&comp.atoms);
                    end = ResidueEnd::Internal;
                }

                if grouping == ResidueGrouping::BasesAndBackbones {
                    let (bases, backbone_atoms) = atoms.into_iter().partition(|&a| !backbone[a]);
                    add(name, bases, end);
                    add("Backbone".to_owned(), backbone_atoms, end);
                } else {
                    add(name, atoms, end);
                }
            }
        }
    }
}
