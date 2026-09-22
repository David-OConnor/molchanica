use bio_files::ResidueType;
use egui::{Color32, ComboBox, RichText, ScrollArea, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;
use mol_defs::molecules::{
    MolType,
    pocket::{POCKET_DIST_THRESH_DEFAULT, Pocket},
};

use crate::{
    button,
    drawing::wrappers::draw_all_pockets,
    label,
    peptide_ligands::{attach_lig, detach_het_res, remove_het_res},
    render::MESH_POCKET_START,
    state::State,
    ui::{COL_SPACING, COLOR_ACTION, ROW_SPACING},
    util::make_lig_from_res,
};

/// Create ligands from hetero residues and pockets around protein residues.
// todo: Move A/R
pub(in crate::ui::popup) fn lig_pocket_from_het_res(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    updates: &mut EngineUpdates,
) {
    let Some(peptide_i) = state.peptide_for_tools_i() else {
        return;
    };
    let mol = &state.peptides[peptide_i];

    label!(
        ui,
        "Ligands and pockets from protein residues",
        Color32::WHITE
    );
    ui.separator();
    ui.add_space(ROW_SPACING);

    ui.checkbox(
        &mut state.ui.popup.lig_include_disconnected,
        "Include disconnected fragments",
    )
    .on_hover_text(
        "Applies to Make lig and Detach. When unchecked, keep only the bonded component with \
        the most heavy atoms (then total atoms), using the loaded bond graph. \
        Detach removes the whole residue from the protein; excluded fragments are discarded.",
    );
    let include_disconnected = state.ui.popup.lig_include_disconnected;

    // Avoids a double borrow.
    let mut create_lig_from_res = None;
    let mut pocket_to_add = None;
    let mut res_to_detach = None;
    let mut res_to_remove = None;
    let mut close = false;

    let mut residues: Vec<_> = mol
        .residues
        .iter()
        .enumerate()
        .filter(|(_, res)| !res.atoms.is_empty() && res.res_type != ResidueType::Water)
        .map(|(i, _)| i)
        .collect();
    residues.sort_by_key(|&i| !mol.is_ligand_res(i));
    if residues.is_empty() {
        ui.label("None");
    }

    ScrollArea::vertical().max_height(400.).show(ui, |ui| {
        for res_i in residues {
            let res = &mol.residues[res_i];
            // Match the detach API's definition: a hetero group, excluding amino acids/water.
            let is_ligand = mol.is_ligand_res(res_i);
            let name = res.res_type.to_string();
            let chain = res
                .atoms
                .first()
                .and_then(|&i| mol.common.atoms[i].chain)
                .and_then(|c| mol.chains.get(c))
                .map(|c| c.id.as_str())
                .unwrap_or("?");

            ui.horizontal(|ui| {
                label!(ui, name.clone(), Color32::WHITE);
                ui.label(format!(
                    "{}, chain {chain}, {} atoms",
                    res.serial_number,
                    res.atoms.len()
                ));
                ui.add_space(COL_SPACING / 2.);

                if is_ligand && ui
                    .button(RichText::new("Make lig").color(COLOR_ACTION))
                    .on_hover_text(
                        "Create a ligand using molecules from this residue. It stays part of the \
                        protein too.",
                    )
                    .clicked()
                {
                    let mut selected = res.clone();
                    if !include_disconnected {
                        selected.atoms = mol.common.largest_connected_component(&res.atoms);
                    }
                    create_lig_from_res = Some(selected);
                    close = true;
                }

                if !is_ligand
                    && ui
                        .button(RichText::new("Make pocket").color(COLOR_ACTION))
                        .on_hover_text("Create a pocket around this protein residue.")
                        .clicked()
                {
                    let lig_ctr = {
                        let mut ctr = Vec3::new_zero();
                        for atom_i in &res.atoms {
                            // Using local coordinates; this should be independent of the user positioning the protein.
                            if *atom_i >= mol.common.atoms.len() {
                                eprintln!(
                                    "Error: Atom index out of bounds: {} > {}",
                                    atom_i,
                                    mol.common.atoms.len()
                                );
                                continue;
                            }

                            ctr += mol.common.atoms[*atom_i].posit;
                        }
                        ctr / res.atoms.len() as f64
                    };

                    let ident = format!("Pocket_{name}");
                    pocket_to_add = Some(Pocket::new(
                        mol,
                        lig_ctr,
                        POCKET_DIST_THRESH_DEFAULT,
                        &ident,
                    ));

                    close = true;
                }

                if is_ligand && button!(
                    ui,
                    "Detach",
                    COLOR_ACTION,
                    "Remove this residue from the protein, and its mmCIF data, and open it as a \
                    standalone ligand in its current position. You can then move it, and add it \
                    back to the protein with \"Add to protein\"; this restores the retained atoms' mmCIF records."
                )
                .clicked()
                {
                    res_to_detach = Some(res_i);
                }

                if is_ligand && button!(
                    ui,
                    "Remove",
                    Color32::LIGHT_RED,
                    "Remove this residue from the protein, along with the records describing it \
                    in its mmCIF data."
                )
                .clicked()
                {
                    res_to_remove = Some(res_i);
                }
            });
            ui.add_space(ROW_SPACING / 2.);
        }
    });

    if close {
        state.ui.popup.lig_pocket_creation = false;
    }

    if let Some(mut pocket) = pocket_to_add {
        pocket.mesh_i_rel = state.pockets.len(); // relative: 0 for first pocket, 1 for second, …
        let target_mesh_i = MESH_POCKET_START + pocket.mesh_i_rel;
        while scene.meshes.len() <= target_mesh_i {
            scene.meshes.push(Default::default());
        }
        scene.meshes[target_mesh_i] = pocket.surface_mesh.clone();

        state.pockets.push(pocket);
        state.volatile.active_mol = Some((MolType::Pocket, state.pockets.len() - 1));
        draw_all_pockets(state, scene, updates);

        updates.meshes = true;
    }

    if let Some(res) = &create_lig_from_res {
        make_lig_from_res(state, res, scene, updates);
    }

    if let Some(res_i) = res_to_detach {
        detach_het_res(
            state,
            peptide_i,
            res_i,
            include_disconnected,
            scene,
            updates,
        );
    }

    if let Some(res_i) = res_to_remove {
        remove_het_res(state, peptide_i, res_i, scene, updates);
    }
}

/// Add a ligand to a protein as a hetero residue; e.g. to save them together as one mmCIF.
pub(in crate::ui::popup) fn lig_attach_popup(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    updates: &mut EngineUpdates,
) {
    let Some(mut attach) = state.ui.popup.lig_attach.take() else {
        return;
    };
    let Some(lig) = state.ligands.get(attach.lig_i) else {
        return;
    };
    if state.peptides.is_empty() {
        return;
    }
    attach.peptide_i = attach.peptide_i.min(state.peptides.len() - 1);

    label!(
        ui,
        format!(
            "Ligand: {}",
            lig.common.name.as_deref().unwrap_or(&lig.common.ident)
        ),
        Color32::WHITE
    );

    if let Some(origin) = &lig.cif_origin {
        let auth = origin
            .auth_ids
            .first()
            .map(|(chain, seq, _)| format!(" (chain {chain} {seq})"))
            .unwrap_or_default();

        ui.label(format!("Detached from {}{auth}", origin.source_ident))
            .on_hover_text(
                "Adding it back restores its mmCIF records: its entity and chemical component, \
                atom names, B-factors etc. If it's returned unmoved to the protein it came from, \
                this also restores its connections, binding sites, and validation records.",
            );
    }
    ui.add_space(ROW_SPACING);

    ui.horizontal(|ui| {
        ui.label("Protein:");

        let prev = attach.peptide_i;
        ComboBox::from_id_salt("lig_attach_peptide")
            .selected_text(
                state.peptides[attach.peptide_i]
                    .common
                    .name
                    .as_deref()
                    .unwrap_or(&state.peptides[attach.peptide_i].common.ident),
            )
            .show_ui(ui, |ui| {
                for (i, pep) in state.peptides.iter().enumerate() {
                    ui.selectable_value(
                        &mut attach.peptide_i,
                        i,
                        pep.common.name.as_deref().unwrap_or(&pep.common.ident),
                    );
                }
            });

        if attach.peptide_i != prev {
            attach.comp_id = state.peptides[attach.peptide_i].suggest_comp_id(lig);
        }
    });

    ui.horizontal(|ui| {
        ui.label("Residue name:").on_hover_text(
            "Its chemical component ID in the mmCIF, e.g. \"ATP\": 1-5 letters and digits. \
            Using one the protein already has makes this another copy of that component.",
        );
        ui.add(
            TextEdit::singleline(&mut attach.comp_id)
                .desired_width(50.)
                .char_limit(5),
        );
    });

    ui.checkbox(&mut attach.close_lig, "Close the standalone ligand")
        .on_hover_text("Once added, it's part of the protein; this closes the separate copy.");

    if state.peptides[attach.peptide_i].source_cif.is_none() {
        ui.label(
            RichText::new(
                "This protein has no mmCIF data; this adds the ligand to its model only.",
            )
            .color(Color32::GOLD),
        );
    }
    ui.add_space(ROW_SPACING);

    let mut done = false;
    if button!(
        ui,
        "Add",
        COLOR_ACTION,
        "Add the ligand to the protein, in its current position. Hydrogens are included only if \
        the protein's mmCIF has them."
    )
    .clicked()
    {
        done = attach_lig(state, &attach, scene, updates);
    }

    if !done {
        state.ui.popup.lig_attach = Some(attach);
    }
}
