//! Residue highlight colors shared by atom, bond, ribbon, and surface rendering.
//! Derived on redraw from the current molecule so editing or replacing a protein cannot
//! leave stale residue indices in a persistent highlight mask.

use bio_files::{ResidueType, SecondaryStructure};
use mol_defs::molecules::{AtomRole, aa_color, peptide::MoleculePeptide};
use na_seq::AminoAcid::{self, *};

use crate::state::AaSelection;

pub const AMINO_ACIDS: [AminoAcid; 21] = [
    Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr,
    Val, Sec,
];

/// Use the same C-alpha / atom-serial assignment as the ribbon builder. Missing secondary
/// structure records mean unassigned, displayed together with coil/loop regions.
pub fn secondary_classes(mol: &MoleculePeptide) -> Vec<usize> {
    let mut result = vec![2; mol.residues.len()];

    for atom in &mol.common.atoms {
        if atom.role != Some(AtomRole::C_Alpha) {
            continue;
        }
        let Some(class) = atom.residue.and_then(|i| result.get_mut(i)) else {
            continue;
        };

        if let Some(segment) = mol
            .secondary_structure
            .iter()
            .find(|segment| (segment.start_sn..=segment.end_sn).contains(&atom.serial_number))
        {
            *class = match segment.sec_struct {
                SecondaryStructure::Helix => 0,
                SecondaryStructure::Sheet => 1,
                SecondaryStructure::Coil => 2,
            };
        }
    }

    result
}

impl AaSelection {
    pub fn matches(&self, aa: AminoAcid, secondary: usize) -> bool {
        self.amino_acids.contains(&aa) && self.secondary[secondary]
    }

    pub fn matched_color(&self, aa: AminoAcid) -> (f32, f32, f32) {
        if self.color_by_aa {
            aa_color(aa)
        } else {
            self.highlight_color.into()
        }
    }

    /// Empty when disabled; non-protein residues retain their usual colors.
    pub fn residue_colors(&self, mol: &MoleculePeptide) -> Vec<Option<(f32, f32, f32)>> {
        if !self.enabled {
            return Vec::new();
        }

        let secondary = secondary_classes(mol);
        mol.residues
            .iter()
            .enumerate()
            .map(|(i, residue)| {
                let ResidueType::AminoAcid(aa) = residue.res_type else {
                    return None;
                };

                Some(if self.matches(aa, secondary[i]) {
                    self.matched_color(aa)
                } else {
                    self.neutral_color.into()
                })
            })
            .collect()
    }
}
