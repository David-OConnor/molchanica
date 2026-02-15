use egui::{Color32, RichText, Ui};

use crate::{
    label,
    molecules::{MolIdent, small::MoleculeSmall},
    therapeutic::{Adme, Toxicity},
    ui::{COL_SPACING, ROW_SPACING},
};

fn adme_disp(adme: &Adme, ui: &mut Ui) {
    label!(ui, "ADME:", Color32::GRAY);

    // Absorption
    char_item(
        ui,
        &[(
            "Intestinal permability",
            &format!("{:.2}", adme.intestinal_permeability),
        )],
    );
    char_item(
        ui,
        &[(
            "Intestinal absorption",
            &format!("{:.2}", adme.intestinal_absorption),
        )],
    );
    char_item(ui, &[("pgp", &format!("{:.2}", adme.pgp))]);
    char_item(
        ui,
        &[(
            "Oral_bioavailability",
            &format!("{:.2}", adme.oral_bioavailablity),
        )],
    );
    char_item(
        ui,
        &[("Lipophilicity", &format!("{:.2}", adme.lipophilicity))],
    );
    char_item(
        ui,
        &[("Solubility water", &format!("{:.2}", adme.solubility_water))],
    );
    char_item(
        ui,
        &[(
            "Membrane permeability",
            &format!("{:.2}", adme.membrane_permeability),
        )],
    );
    char_item(
        ui,
        &[(
            "Hydration free Energy",
            &format!("{:.2}", adme.hydration_free_energy),
        )],
    );

    // Distribution
    char_item(
        ui,
        &[(
            "blood_brain_barrier",
            &format!("{:.2}", adme.blood_brain_barrier),
        )],
    );
    char_item(
        ui,
        &[(
            "plasma protein binding",
            &format!("{:.2}", adme.plasma_protein_binding_rate),
        )],
    );
    char_item(ui, &[("VDSS", &format!("{:.2}", adme.vdss))]);

    // Metabolism
    char_item(
        ui,
        &[(
            "CYP P450 2C19 inhibition",
            &format!("{:.2}", adme.cyp_2c19_inhibition),
        )],
    );
    char_item(
        ui,
        &[(
            "CYP P450 2D6 inhibition",
            &format!("{:.2}", adme.cyp_2d6_inhibition),
        )],
    );
    char_item(
        ui,
        &[(
            "CYP P450 3A4 inhibition",
            &format!("{:.2}", adme.cyp_3a4_inhibition),
        )],
    );

    char_item(
        ui,
        &[(
            "CYP P450 1A2 inhibition",
            &format!("{:.2}", adme.cyp_1a2_inhibition),
        )],
    );

    char_item(
        ui,
        &[(
            "CYP P450 2C9 inhibition",
            &format!("{:.2}", adme.cyp_2c9_inhibition),
        )],
    );

    // Excretion
    char_item(ui, &[("Half Life", &format!("{:.2}", adme.half_life))]);
    char_item(ui, &[("Clearance", &format!("{:.2}", adme.clearance))]);
}

fn char_item(ui: &mut Ui, items: &[(&str, &str)]) {
    ui.horizontal(|ui| {
        for (i, (name, v)) in items.iter().enumerate() {
            label!(ui, format!("{name}:"), Color32::GRAY);
            label!(ui, *v, Color32::WHITE);

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

                let mut ident_txt = RichText::new(ident.ident_innner()).color(Color32::WHITE);

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
            ("Atoms", &char.num_atoms.to_string()),
            ("Bonds", &char.num_bonds.to_string()),
            ("Heavy", &char.num_heavy_atoms.to_string()),
            ("Het", &char.num_hetero_atoms.to_string()),
        ],
    );

    char_item(ui, &[("Weight", &format!("{:.2}", char.mol_weight))]);

    ui.add_space(ROW_SPACING);
    // Functional groups

    char_item(
        ui,
        &[
            ("Rings Ar", &char.num_rings_aromatic.to_string()),
            ("Sat", &char.num_rings_saturated.to_string()),
            ("Ali", &char.num_rings_aliphatic.to_string()),
        ],
    );

    // todo: Rings
    char_item(
        ui,
        &[
            ("Amine", &char.amines.len().to_string()),
            ("Amide", &char.amides.len().to_string()),
            ("Carbonyl", &char.carbonyl.len().to_string()),
            ("Hydroxyl", &char.hydroxyl.len().to_string()),
            // other FGs like sulfur ones and carboxylate?
        ],
    );

    ui.add_space(ROW_SPACING);
    // Misc properties
    char_item(
        ui,
        &[
            ("H bond donor:", &char.h_bond_donor.len().to_string()),
            ("acceptor", &char.h_bond_acceptor.len().to_string()),
            ("Valence elecs", &char.num_valence_elecs.to_string()),
        ],
    );

    // Geometry
    // ui.add_space(ROW_SPACING);

    char_item(
        ui,
        &[
            ("TPSA (Ertl)", &format!("{:.2}", char.tpsa_ertl)),
            ("PSA (Geom)", &format!("{:.2}", char.psa_topo)),
            ("Greasiness", &format!("{:.2}", char.greasiness)),
        ],
    );

    let vol = match char.volume_pubchem {
        Some(v) => v,
        None => char.volume,
    };
    char_item(
        ui,
        &[
            ("ASA (Labute)", &format!("{:.2}", char.asa_labute)),
            ("ASA (Geom)", &format!("{:.2}", char.asa_topo)),
            ("Volume", &format!("{:.2}", vol)),
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
            ("LogP", &format!("{:.2}", log_p)),
            ("Mol Refrac", &format!("{:.2}", char.molar_refractivity)),
        ],
    );

    char_item(
        ui,
        &[
            ("Balaban J", &format!("{:.2}", char.balaban_j)),
            ("Bertz Complexity", &format!("{:.2}", char.bertz_ct)),
        ],
    );

    char_item(
        ui,
        &[(
            "Complexity",
            &format!("{:.2}", char.complexity.unwrap_or(0.0)),
        )],
    );

    char_item(
        ui,
        &[(
            "Wiener index",
            &format!("{}", char.wiener_index.unwrap_or(0)),
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

    char_item(ui, &[("LD50", &format!("{:.2}", tox.ld50))]);
    char_item(ui, &[("hERG", &format!("{:.2}", tox.ether_a_go_go))]);
    char_item(ui, &[("Mutagenicity", &format!("{:.2}", tox.mutagencity))]);
    char_item(
        ui,
        &[(
            "Liver injury",
            &format!("{:.2}", tox.drug_induced_liver_injury),
        )],
    );
    char_item(
        ui,
        &[("Skin reaction", &format!("{:.2}", tox.skin_reaction))],
    );
    char_item(ui, &[("Carcinogen", &format!("{:.2}", tox.carcinogen))]);
}
