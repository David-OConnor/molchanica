//! Generate small molecules based on given protein/pocket targets and/or properties

use crate::molecules::{MoleculePeptide, small::MoleculeSmall};

/// By default:
/// - Solubility in water > 0.~~
/// - Must pass gut wall
/// - Not mutagenic
/// - Not
/// - Low LD50
/// - Membrane preability?
///
/// Oral availability implies by "oral bioavailibility", gut wall cross and related.
pub fn generate(
    library: &[MoleculeSmall],
    protein: Option<&MoleculePeptide>,
    oral_delivery: bool,
    blod_brain_barrier: bool,
) -> Vec<MoleculeSmall> {
    let result = Vec::new();

    for mol in library {
        let Some(char) = &mol.characterization else {
            eprintln!(
                "Skipping mol {} due to missing characterization",
                mol.common.ident
            );
            continue;
        };
        let Some(ther) = &mol.therapeutic_props else {
            eprintln!(
                "Skipping mol {} due to missing therapeutic properties",
                mol.common.ident
            );
            continue;
        };

        if oral_delivery {
            if ther.adme.oral_bioavailablity < 0.0 {
                continue;
            }
            if ther.adme.intestinal_permeability < 0.0 {
                continue;
            }
            if ther.adme.intestinal_absorption < 0.0 {
                continue;
            }
        }

        if blod_brain_barrier && ther.adme.blood_brain_barrier < 0.00 {
            continue;
        }
    }

    result
}
