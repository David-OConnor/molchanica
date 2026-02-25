use egui::{Color32, RichText, Ui};

use crate::{
    label,
    molecules::{MolIdent, small::MoleculeSmall},
    therapeutic::{Adme, Toxicity},
    ui::{COL_SPACING, ROW_SPACING},
};

fn adme_disp(adme: &Adme, ui: &mut Ui) {
    // label!(ui, "ADME:", Color32::GRAY);

    // Absorption
    char_item(
        ui,
        &[(
            "Intestinal permability",
            &format!("{:.2}", adme.intestinal_permeability),
            "cm/s",
        )],
    );
    char_item(
        ui,
        &[(
            "Intestinal absorption",
            &format!("{:.2}", adme.intestinal_absorption),
            "Binary",
        )],
    );
    char_item(ui, &[("pgp", &format!("{:.2}", adme.pgp), "binary")]);
    char_item(
        ui,
        &[(
            "Oral_bioavailability",
            &format!("{:.2}", adme.oral_bioavailablity),
            "Binary",
        )],
    );
    char_item(
        ui,
        &[(
            "Lipophilicity",
            &format!("{:.2}", adme.lipophilicity),
            "log-ratio",
        )],
    );
    char_item(
        ui,
        &[(
            "Solubility water",
            &format!("{:.2}", adme.solubility_water),
            "LogS, where S is the aqueous solubility. log mol/L",
        )],
    );
    char_item(
        ui,
        &[(
            "Membrane permeability",
            &format!("{:.2}", adme.membrane_permeability),
            "PAMPA (parallel artificial membrane permeability assay) is a commonly employed assay \
            to evaluate drug permeability across the cellular membrane. Binary.",
        )],
    );
    char_item(
        ui,
        &[(
            "Hydration free Energy",
            &format!("{:.2}", adme.hydration_free_energy),
            "The Free Solvation Database, FreeSolv(SAMPL), provides experimental and calculated hydration
free energy of small molecules in water. The calculated values are derived from alchemical
free energy calculations using molecular dynamics simulations."
        )],
    );

    // Distribution
    char_item(
        ui,
        &[(
            "blood_brain_barrier",
            &format!("{:.2}", adme.blood_brain_barrier),
            "Binary",
        )],
    );
    char_item(
        ui,
        &[(
            "plasma protein binding",
            &format!("{:.2}", adme.plasma_protein_binding_rate),
            "% binding value.",
        )],
    );
    char_item(
        ui,
        &[(
            "VDSS",
            &format!("{:.2}", adme.vdss),
            "Volume of Distribution at steady state.",
        )],
    );

    let descrip_cyp =
        " The CYP P450 genes are essential in the breakdown (metabolism) of various molecules and
chemicals within cells. A drug that can inhibit these enzymes would mean poor metabolism
to this drug and other drugs, which could lead to drug-drug interactions and adverse effects.

CYP2C19 gene provides instructions for making an enzyme called the endoplasmic reticulum,
which is involved in protein processing and transport.
Binary.";

    // Metabolism
    char_item(
        ui,
        &[(
            "CYP P450 2C19 inhibition",
            &format!("{:.2}", adme.cyp_2c19_inhibition),
            descrip_cyp,
        )],
    );
    char_item(
        ui,
        &[(
            "CYP P450 2D6 inhibition",
            &format!("{:.2}", adme.cyp_2d6_inhibition),
            descrip_cyp,
        )],
    );
    char_item(
        ui,
        &[(
            "CYP P450 3A4 inhibition",
            &format!("{:.2}", adme.cyp_3a4_inhibition),
            descrip_cyp,
        )],
    );

    char_item(
        ui,
        &[(
            "CYP P450 1A2 inhibition",
            &format!("{:.2}", adme.cyp_1a2_inhibition),
            descrip_cyp,
        )],
    );

    char_item(
        ui,
        &[(
            "CYP P450 2C9 inhibition",
            &format!("{:.2}", adme.cyp_2c9_inhibition),
            descrip_cyp,
        )],
    );

    // Excretion
    char_item(ui, &[("Half Life", &format!("{:.2}", adme.half_life), "")]);
    char_item(ui, &[("Clearance", &format!("{:.2}", adme.clearance), "")]);
}

/// (name, val, description)
fn char_item(ui: &mut Ui, items: &[(&str, &str, &str)]) {
    ui.horizontal(|ui| {
        for (i, (name, v, descrip)) in items.iter().enumerate() {
            label!(ui, format!("{name}:"), Color32::GRAY).on_hover_text(*descrip);
            label!(ui, *v, Color32::WHITE).on_hover_text(*descrip);

            if i != items.len() - 1 {
                ui.add_space(COL_SPACING / 2.);
            }
        }
    });
}

pub(in crate::ui) fn mol_char_disp(mol: &MoleculeSmall, ui: &mut Ui) {
    let Some(char) = &mol.characterization else {
        return;
    };

    // todo: Small font?
    for ident in &mol.idents {
        ui.horizontal(|ui| {
            // Wrap long names, like InChi etc.
            ui.horizontal_wrapped(|ui| {
                label!(ui, format!("{}:", ident.label()), Color32::GRAY);

                let mut ident_txt = RichText::new(ident.ident_inner()).color(Color32::WHITE);

                // These are longer idents.
                if matches!(
                    ident,
                    MolIdent::InchIKey(_)
                        | MolIdent::InchI(_)
                        | MolIdent::Smiles(_)
                        | MolIdent::IupacName(_)
                ) {
                    let font = egui::FontId::proportional(10.0);
                    ident_txt = ident_txt.font(font);
                }

                ui.label(ident_txt);
            });
        });
    }

    ui.add_space(ROW_SPACING);
    // Basics
    char_item(
        ui,
        &[
            ("Atoms", &char.num_atoms.to_string(), ""),
            ("Bonds", &char.num_bonds.to_string(), ""),
            (
                "Heavy",
                &char.num_heavy_atoms.to_string(),
                "Number of heavy (non-Hydrogen) atoms",
            ),
            (
                "Het",
                &char.num_hetero_atoms.to_string(),
                "Number of hetero (non-Carbon) atoms",
            ),
        ],
    );

    char_item(ui, &[("Weight", &format!("{:.2}", char.mol_weight), "")]);

    ui.add_space(ROW_SPACING);
    // Functional groups

    char_item(
        ui,
        &[
            ("Rings Ar", &char.num_rings_aromatic.to_string(), ""),
            ("Sat", &char.num_rings_saturated.to_string(), ""),
            ("Ali", &char.num_rings_aliphatic.to_string(), ""),
        ],
    );

    // todo: Rings
    char_item(
        ui,
        &[
            ("Amine", &char.amines.len().to_string(), ""),
            ("Amide", &char.amides.len().to_string(), ""),
            ("Carbonyl", &char.carbonyl.len().to_string(), ""),
            ("Hydroxyl", &char.hydroxyl.len().to_string(), ""),
            // other FGs like sulfur ones and carboxylate?
        ],
    );

    ui.add_space(ROW_SPACING);
    // Misc properties
    char_item(
        ui,
        &[
            (
                "H bond donor:",
                &char.h_bond_donor.len().to_string(),
                "Number of hydrogen bond donors",
            ),
            (
                "acceptor",
                &char.h_bond_acceptor.len().to_string(),
                "Number of hydrogen bond acceptors",
            ),
            ("Valence elecs", &char.num_valence_elecs.to_string(), ""),
        ],
    );

    // Geometry
    // ui.add_space(ROW_SPACING);

    char_item(
        ui,
        &[
            (
                "TPSA (Ertl)",
                &format!("{:.2}", char.tpsa_ertl),
                "Polar surface area; computed analytically.",
            ),
            (
                "PSA (Geom)",
                &format!("{:.2}", char.psa_topo),
                "Polar surface area; computed by creating a surface mesh, then measuring the polar part of it.",
            ),
            (
                "Greasiness",
                &format!("{:.2}", char.greasiness),
                "i.e. lipophilicity",
            ),
        ],
    );

    let vol = match char.volume_pubchem {
        Some(v) => v,
        None => char.volume,
    };
    char_item(
        ui,
        &[
            ("ASA (Labute)", &format!("{:.2}", char.asa_labute), ""),
            (
                "ASA (Geom)",
                &format!("{:.2}", char.asa_topo),
                "Computed by creating a surface mesh, then computing the fraction of its volume that is solvent-accessible",
            ),
            (
                "Volume",
                &format!("{:.2}", vol),
                "Computed by creating a surface mesh.",
            ),
        ],
    );

    // Computed properties
    ui.add_space(ROW_SPACING);

    let log_p = match char.log_p_pubchem {
        Some(v) => v,
        None => char.log_p,
    };
    char_item(
        ui,
        &[
            (
                "LogP",
                &format!("{:.2}", log_p),
                "Partition coefficient; a proxy for lipophilicity.",
            ),
            ("Mol Refrac", &format!("{:.2}", char.molar_refractivity), ""),
        ],
    );

    char_item(
        ui,
        &[
            ("Balaban J", &format!("{:.2}", char.balaban_j), ""),
            ("Bertz Complexity", &format!("{:.2}", char.bertz_ct), ""),
        ],
    );

    char_item(
        ui,
        &[(
            "Complexity",
            &format!("{:.2}", char.complexity.unwrap_or(0.0)),
            "",
        )],
    );

    char_item(
        ui,
        &[(
            "Wiener index",
            &format!("{}", char.wiener_index.unwrap_or(0)),
            "",
        )],
    );

    if let Some(ther) = &mol.therapeutic_props {
        ui.add_space(ROW_SPACING);
        ui.separator();
        label!(ui, "Therapeutic data", Color32::LIGHT_BLUE);

        adme_disp(&ther.adme, ui);
        tox_disp(&ther.toxicity, ui);
    }
}

fn tox_disp(tox: &Toxicity, ui: &mut Ui) {
    label!(ui, "Toxicity:", Color32::GRAY).on_hover_text("Of our cittyyyyyyy");

    char_item(
        ui,
        &[("LD50", &format!("{:.2}", tox.ld50), "log(1/(mol/kg))")],
    );
    char_item(
        ui,
        &[("hERG", &format!("{:.2}", tox.ether_a_go_go), "Binary")],
    );
    char_item(
        ui,
        &[("Mutagenicity", &format!("{:.2}", tox.mutagencity), "Binary")],
    );
    char_item(
        ui,
        &[(
            "Liver injury",
            &format!("{:.2}", tox.drug_induced_liver_injury),
            "DILI",
        )],
    );
    char_item(
        ui,
        &[(
            "Skin reaction",
            &format!("{:.2}", tox.skin_reaction),
            "Binary",
        )],
    );
    char_item(
        ui,
        &[("Carcinogen", &format!("{:.2}", tox.carcinogen), "Binary")],
    );
}
