//! Information and settings for the opened, or to-be opened molecules.

use bio_apis::{drugbank, lmsd, pdbe, pubchem, rcsb};
use bio_files::{ResidueType, md_params::ForceFieldParams};
use dynamics::params::FfParamSet;
use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::{
    cam::move_cam_to_active_mol,
    drawing,
    drawing::{
        CHARGE_MAP_MAX, CHARGE_MAP_MIN, COLOR_AA_NON_RESIDUE_EGUI, draw_pocket,
        wrappers::draw_all_pockets,
    },
    label,
    molecules::{
        Atom, Bond, MolGenericRef, MolGenericRefMut, MolIdent, MolType, Residue, aa_color,
        lipid::MoleculeLipid,
        nucleic_acid::MoleculeNucleicAcid,
        pocket::{POCKET_DIST_THRESH_DEFAULT, Pocket},
        small::MoleculeSmall,
    },
    render::MESH_POCKET,
    selection::Selection,
    state::State,
    ui::{COL_SPACING, COLOR_ACTION, COLOR_HIGHLIGHT, mol_descrip, popups},
    util::{handle_err, handle_success, make_egui_color, make_lig_from_res, move_mol_to_res},
};

/// `posit_override` is for example, relative atom positions, such as a positioned ligand.
fn disp_atom_data(
    atom: &Atom,
    residues: &[Residue],
    posit_override: Option<Vec3>,
    ui: &mut Ui,
    show_role: bool,
    show_posit: bool,
) {
    let text_0 = format!("#{}", atom.serial_number);
    label!(ui, text_0, Color32::WHITE);

    if show_posit {
        // Similar to `Vec3`'s format impl, but with fewer digits.
        let posit = match posit_override {
            Some(p) => p,
            None => atom.posit,
        };

        let posit_txt = format!("|{:.3}, {:.3}, {:.3}|", posit.x, posit.y, posit.z);
        label!(ui, posit_txt, Color32::GOLD);
    }

    let text_b = atom.element.to_letter();

    let atom_color = make_egui_color(atom.element.color());
    label!(ui, text_b, atom_color);

    // Hijacking `show_posit` here to hide res/dihedral info as well.
    if let Some(res_i) = atom.residue
        && show_posit
    {
        // Placeholder for water etc.
        let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;

        let res_txt = if res_i >= residues.len() {
            "Invalid res".to_owned()
        } else {
            let res = &residues[res_i];

            if let ResidueType::AminoAcid(aa) = res.res_type {
                res_color = make_egui_color(aa_color(aa));
            }

            format!("  {res}")
        };

        label!(ui, res_txt, res_color);
    }

    if show_role {
        let role = match atom.role {
            Some(r) => format!("Role: {r}"),
            None => String::new(),
        };
        label!(ui, role, Color32::LIGHT_GRAY);
    }

    if let Some(tir) = &atom.type_in_res {
        label!(ui, tir.to_string(), Color32::LIGHT_YELLOW);
    }

    if let Some(tir) = &atom.type_in_res_general {
        label!(ui, tir, Color32::LIGHT_YELLOW);
    }

    if let Some(ff) = &atom.force_field_type {
        label!(ui, format!("FF: {ff}"), Color32::LIGHT_YELLOW);
    }

    if let Some(q) = &atom.partial_charge {
        let plus = if *q > 0. { "+" } else { "" };
        let color = make_egui_color(drawing::color_viridis_float(
            *q,
            CHARGE_MAP_MIN,
            CHARGE_MAP_MAX,
        ));

        // todo: In some cases, this is getting rendered over the initial text? EGUI error?
        label!(ui, format!("{plus}q: {q:.2}"), color);
    }
}

// todo: This would ideally be a method on FfParamSet, but that lib doesn't have access to our MolType enum.
/// Get params for a single molecule type.
pub(in crate::ui) fn get_params(set: &FfParamSet, mol_type: MolType) -> &Option<ForceFieldParams> {
    match mol_type {
        MolType::Peptide => &set.peptide,
        MolType::Ligand => &set.small_mol,
        MolType::NucleicAcid => &set.dna, // todo: Could be RNA
        MolType::Lipid => &set.lipids,
        _ => &None,
    }
}

/// `posit_override` is for example, relative atom positions, such as a positioned ligand.
fn disp_bond_data(
    bond: &Bond,
    atoms: &[Atom],
    mol_type: MolType,
    params: &FfParamSet,
    ui: &mut Ui,
) {
    let atom_0 = &atoms[bond.atom_0];
    let atom_1 = &atoms[bond.atom_1];

    ui.label("Bond");
    label!(
        ui,
        format!("#{} - #{}", bond.atom_0_sn, bond.atom_1_sn),
        Color32::WHITE
    );

    label!(
        ui,
        format!(
            "{} {} {}",
            atom_0.element.to_letter(),
            bond.bond_type.to_visual_str(),
            atom_1.element.to_letter()
        ),
        Color32::LIGHT_BLUE
    );

    // todo: Cache??
    let posit_0 = atom_0.posit;
    let posit_1 = atom_1.posit;
    let dist = (posit_1 - posit_0).magnitude();

    label!(ui, format!("{dist:.3} Å"), Color32::LIGHT_YELLOW);

    if let Some(p) = get_params(params, mol_type) {
        if let (Some(ff_0), Some(ff_1)) = (
            atom_0.force_field_type.as_deref(),
            atom_1.force_field_type.as_deref(),
        ) {
            if let Some(b) = p.get_bond(&(ff_0.to_string(), ff_1.to_string()), true) {
                ui.label(RichText::new("Param len:"));
                label!(ui, format!("{:.3} Å", b.r_0), Color32::LIGHT_BLUE);

                // todo: Cache this; don't compute in the UI.
                let m0 = atom_0.element.atomic_weight();
                let m1 = atom_1.element.atomic_weight();
                let mu_amu = (m0 * m1) / (m0 + m1);
                // todo: QC this.
                let freq = 3.2555 * (b.k_b / mu_amu).sqrt(); // 1/ps
                ui.label(format!("Freq: {freq:.1}ps^-1"));
            }
        }
    }
}

/// Display text of the selected atom or residue.
pub(in crate::ui) fn selected_data(
    state: &State,
    ligands: &[MoleculeSmall],
    nucleic_acids: &[MoleculeNucleicAcid],
    lipids: &[MoleculeLipid],
    pockets: &[Pocket],
    selection: &Selection,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        // ui.horizontal_wrapped(|ui| {
        match selection {
            Selection::AtomPeptide(sel_i) => {
                let Some(mol) = &state.peptide else { return };
                if *sel_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*sel_i];
                disp_atom_data(atom, &mol.residues, None, ui, true, true);
            }
            Selection::AtomsPeptide(atom_is) => {
                let Some(mol) = &state.peptide else {
                    return;
                };

                ui.label(format!("{} atoms |", atom_is.len()));

                for atom_i in atom_is {
                    if *atom_i >= mol.common.atoms.len() {
                        return;
                    }
                    let atom = &mol.common.atoms[*atom_i];
                    let posit = mol.common.atom_posits[*atom_i];

                    disp_atom_data(atom, &mol.residues, Some(posit), ui, false, false);

                    ui.label("|");
                }
            }
            Selection::AtomLig((mol_i, atom_i)) => {
                if *mol_i >= ligands.len() {
                    return;
                }
                let mol = &ligands[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &[], Some(posit), ui, false, true);
            }
            // todo: Update A/R
            Selection::AtomsLig((mol_i, atom_is)) => {
                if *mol_i >= ligands.len() {
                    return;
                }
                let mol = &ligands[*mol_i];

                ui.label(format!("{} atoms |", atom_is.len()));

                for atom_i in atom_is {
                    if *atom_i >= mol.common.atoms.len() {
                        return;
                    }
                    let atom = &mol.common.atoms[*atom_i];
                    let posit = mol.common.atom_posits[*atom_i];

                    disp_atom_data(atom, &[], Some(posit), ui, false, false);
                    ui.label("|");
                }
            }
            // todo DRY
            Selection::AtomNucleicAcid((mol_i, atom_i)) => {
                if *mol_i >= nucleic_acids.len() {
                    return;
                }
                let mol = &nucleic_acids[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &mol.residues, Some(posit), ui, true, true);
            }
            // todo DRY
            Selection::AtomLipid((mol_i, atom_i)) => {
                if *mol_i >= lipids.len() {
                    return;
                }
                let mol = &lipids[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &mol.residues, Some(posit), ui, true, true);
            }
            // todo DRY
            Selection::AtomPocket((mol_i, atom_i)) => {
                if *mol_i >= pockets.len() {
                    return;
                }
                let mol = &pockets[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &[], Some(posit), ui, true, true);
            }
            Selection::Residue(sel_i) => {
                if let Some(mol) = &state.peptide {
                    if *sel_i >= mol.residues.len() {
                        return;
                    }

                    let res = &mol.residues[*sel_i];
                    // todo: Color-coding by part like atom, to make easier to view.

                    let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;

                    if let ResidueType::AminoAcid(aa) = res.res_type {
                        res_color = make_egui_color(aa_color(aa));
                    }
                    label!(ui, res.to_string(), res_color);
                }
            }
            Selection::BondPeptide(bond_i) => {
                let Some(mol) = &state.peptide else {
                    return;
                };
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Peptide,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondLig((mol_i, bond_i)) => {
                if *mol_i >= ligands.len() {
                    return;
                }
                let mol = &ligands[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Ligand,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondsLig((mol_i, bond_is)) => {
                if *mol_i >= ligands.len() {
                    return;
                }
                let mol = &ligands[*mol_i];

                for bond_i in bond_is {
                    if *bond_i >= mol.common.bonds.len() {
                        return;
                    }

                    let bond = &mol.common.bonds[*bond_i];
                    disp_bond_data(
                        bond,
                        &mol.common.atoms,
                        MolType::Ligand,
                        &state.ff_param_set,
                        ui,
                    );
                }
            }
            Selection::BondNucleicAcid((mol_i, bond_i)) => {
                if *mol_i >= nucleic_acids.len() {
                    return;
                }
                let mol = &nucleic_acids[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::NucleicAcid,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondLipid((mol_i, bond_i)) => {
                if *mol_i >= lipids.len() {
                    return;
                }
                let mol = &lipids[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Lipid,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondPocket((mol_i, bond_i)) => {
                if *mol_i >= pockets.len() {
                    return;
                }
                let mol = &pockets[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Pocket,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::None => {}
        }
    });
}

// todo: Unify this with non-peptide.
pub(in crate::ui) fn display_mol_data_peptide(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
    redraw_lig: &mut bool,
    close: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    // These variables prevent double borrows.
    let mut res_to_make = None;
    let mut move_lig_to_res = None;
    let mut move_lig_to_sel = None;
    let mut move_cam = false;

    ui.horizontal(|ui| {
        if let Some(pep) = &state.peptide {
            mol_descrip(&MolGenericRef::Peptide(pep), ui);

            // if ui.button(RichText::new("Close").color(Color32::LIGHT_RED)).clicked() {
            //     *close = true;
            // }

            if pep.common.ident.len() <= 5 {
                // todo: You likely need a better approach.
                if ui
                    .button("RCSB")
                    .on_hover_text("Open a web browser to the RCSB PDB page for this molecule.")
                    .clicked()
                {
                    rcsb::open_overview(&pep.common.ident);
                }
            }

            if ui.button("Dihe")
                .on_hover_text("Draw a Ramachandran plot of the dihedral angles of the peptide.")
                .clicked() {
                state.ui.popup.rama_plot = !state.ui.popup.rama_plot;
            }

            if ui.button("Meta")
                .on_hover_text("Display metadata for this molecule")
                .clicked() {
                if let Some((mol_type, _)) = state.ui.popup.metadata && mol_type == MolType::Peptide {
                    state.ui.popup.metadata = None;
                } else {
                    state.ui.popup.metadata = Some((MolType::Peptide, 0))
                }
            }

            let res_selected = match state.ui.selection {
                Selection::AtomPeptide(sel_i) => {
                    let atom = &pep.common.atoms[sel_i];
                    if let Some(res_i) = &atom.residue {
                        Some(&pep.residues[*res_i])
                    } else {
                        None
                    }
                }
                Selection::Residue(sel_i) => {
                    if sel_i >= pep.residues.len() {
                        handle_err(&mut state.ui, "Residue selection is out of bounds.".to_owned());
                        None
                    } else {
                        Some(&pep.residues[sel_i])
                    }
                }
                _ => None,
            };

            if let Some(res) = res_selected {
                if ui
                    .button(
                        RichText::new(format!("Lig from {}", res.res_type))
                            .color(COLOR_ACTION),
                    )
                    .on_hover_text(
                        "Create a ligand from this residue on the peptide. This can be \
                    saved to a Mol2 or SDF file, and used as a ligand. Molecular dynamics can be performed on it.",
                    )
                    .clicked()
                {
                    // todo: I don't like this clone, but it avoids a dbl-borrow.
                    res_to_make = Some(res.clone());
                }
            }

            if let Some(mol) = state.active_mol() {
                for res in &pep.het_residues {
                    // Note: This approach will fail if there are multiple hetero residues of similar len to
                    // this ligand.

                    if (res.atoms.len() as i32 - mol.common().atoms.len() as i32) < 3 {
                        let name = match &res.res_type {
                            ResidueType::Other(name) => name,
                            _ => "hetero residue",
                        };
                        ui.add_space(COL_SPACING / 2.);

                        if ui
                            .button(RichText::new(format!("Lig to {name}")).color(COLOR_HIGHLIGHT))
                            .on_hover_text("Move the ligand to be colocated with this residue. this is intended to \
                    be used to synchronize the ligand with a pre-positioned hetero residue in the protein file, e.g. \
                    prior to docking. In addition to moving \
                    its center, this attempts to align each atom with its equivalent on the residue.")
                            .clicked()
                        {
                            // todo: I don't like this clone, but it avoids a dbl-borrow.
                            move_lig_to_res = Some(res.clone());
                        }
                        break;
                    }
                }
            }

            if let Some((mol_type, _)) = state.volatile.active_mol {
                if mol_type == MolType::Ligand {
                    if !matches!(
                state.ui.selection,
                Selection::None | Selection::AtomLig(_)
            ) {
                        if ui
                            .button(RichText::new("Move lig to sel").color(COLOR_HIGHLIGHT))
                            .on_hover_text("Re-position the ligand to be colacated with the selected atom or residue.")
                            .clicked()
                        {
                            let peptide = state.peptide.as_ref().unwrap();
                            let atom_sel = peptide.get_sel_atom(&state.ui.selection);

                            if let Some(a) = atom_sel {
                                // See note on why we clone above.
                                move_lig_to_sel = Some(a.clone());
                            }
                        }
                    }
                }
            }
        }
    });

    if let Some(res) = res_to_make {
        make_lig_from_res(state, &res, scene, engine_updates);
        // if let Some(pep) = &state.peptide {
        //     move_cam_to_active_mol(state, scene, pep.center, engine_updates);
        // }
    }

    if let Some(res) = move_lig_to_res {
        if let Some((_, i)) = state.volatile.active_mol {
            let mol = &mut state.ligands[i];
            if let Some(pep) = &state.peptide {
                move_mol_to_res(&mut MolGenericRefMut::Small(mol), pep, &res);
                move_cam_to_active_mol(state, scene, pep.center, engine_updates);
            }
        }

        *redraw_lig = true;
    }

    if let Some(sel_atom) = move_lig_to_sel {
        let mut mol = state.active_mol_mut().unwrap();
        mol.common_mut().move_to(sel_atom.posit);

        let center = match &state.peptide {
            Some(p) => p.center,
            None => Vec3::new_zero(),
        };
        move_cam_to_active_mol(state, scene, center, engine_updates);

        move_cam = true;

        *redraw_lig = true;
    }

    if move_cam {
        let center = match &state.peptide {
            Some(m) => m.center,
            None => Vec3::new_zero(),
        };

        move_cam_to_active_mol(state, scene, center, engine_updates);
    }

    let mut pocket_to_add = None;

    if let Some(mol) = &state.peptide {
        // todo: Temp location
        if let Selection::AtomPeptide(sel_i) = state.ui.selection {
            if ui.button(RichText::new("Pocket from sel").color(COLOR_ACTION))
                .on_hover_text("Create a pocket around the selected atom. For screening, docking, pharmacophores etc.")
                .clicked() {

                // todo: Rel or abs?
                let posit = mol.common.atoms[sel_i].posit;
                let ident = format!("{} atom {sel_i}", mol.common.ident);
                pocket_to_add = Some(Pocket::new(mol, posit, POCKET_DIST_THRESH_DEFAULT, &ident));
            }
        }

        if ui
            .button(RichText::new("Make ligs/pockets").color(COLOR_ACTION))
            .on_hover_text("Create ligands or pockets based on hetero residues in the protein. \
                This may be used in pharmacophore creation, screening, etc. Hetero residues included in mmCIF \
                files may represent bound ligands.")
            .clicked()
        {
            state.ui.popup.lig_pocket_creation = !state.ui.popup.lig_pocket_creation;
        }
    }

    if let Some(pocket) = pocket_to_add {
        scene.meshes[MESH_POCKET] = pocket.surface_mesh.clone();
        draw_all_pockets(state, scene);
        state.pockets.push(pocket);

        engine_updates.meshes = true;
        engine_updates.entities = EntityUpdate::All;
    }
}

pub(in crate::ui) fn display_mol_data(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        let Some((active_mol_type, active_mol_i)) = state.volatile.active_mol else {
            return;
        };

        if let Some(mol) = state.active_mol() {
            mol_descrip(&mol, ui);
        }

        if active_mol_type == MolType::Ligand {
            if ui
                .button(RichText::new("Similar mols").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Using PubChem, find, download, and open similar molecules to this one.",
                )
                .clicked()
            {
                let mol = &state.ligands[active_mol_i];

                // todo: This needs to be in its own thread; long-running blockign call.

                // todo: Support more than CID. Requires a mode to rcsb api.
                for ident in &mol.idents {
                    println!("IDEN: {:?}", ident); // todo temp
                    if let MolIdent::PubChem(cid) = ident {
                        // todo: This doesn't show because it doesn't get a chance to render prior to the block.
                        handle_success(
                            &mut state.ui,
                            "Searching for similar molecules...".to_string(),
                        );
                        match pubchem::find_similar_mols(*cid) {
                            Ok(cids) => {
                                // todo: This is temp
                                println!("Similar mols to {cid}: {:?}", cids);
                                let max_results = 20;
                                let cids_str = cids
                                    .iter()
                                    .take(max_results)
                                    .map(|x| x.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");

                                handle_success(
                                    &mut state.ui,
                                    format!("Similar PubChem CIDs: {cids_str}..."),
                                );
                            }

                            Err(_) => {
                                handle_err(
                                    &mut state.ui,
                                    "Problem finding similar molecules on PubChem".to_owned(),
                                );
                            }
                        }
                        break;
                    }
                }
            }

            if ui
                .button("Metadata")
                .on_hover_text("Display metadata for this molecule")
                .clicked()
            {
                if let Some((mol_type, _)) = state.ui.popup.metadata
                    && mol_type == MolType::Ligand
                {
                    state.ui.popup.metadata = None;
                } else {
                    state.ui.popup.metadata = Some((MolType::Ligand, active_mol_i))
                }
            }
        }

        let mut update_cid = None; // to avoid a borrow error.

        if let Some(mol) = state.active_mol() {
            match mol {
                MolGenericRef::Peptide(_) => {}
                MolGenericRef::Small(m) => {
                    ui.add_space(COL_SPACING);
                    if !m.ff_params_loaded {
                        ui.label(RichText::new("FF/q").color(Color32::LIGHT_RED))
                            .on_hover_text(
                                "Green if force field names, and partial charges are assigned \
                for all ligand atoms. Required for ligand moleculer dynamics and docking.",
                            );
                    }

                    if !m.frcmod_loaded {
                        ui.label(RichText::new("Frcmod").color(Color32::LIGHT_RED))
                            .on_hover_text(
                                "Green if molecule-specific Amber force field parameters are \
                loaded for this ligand. Required for ligand molecular dynamics and docking.",
                            );
                    }

                    let mut pubchem_cid = None;
                    for ident in &m.idents {
                        match ident {
                            MolIdent::DrugBank(id) => {
                                if ui.button(format!("DrugBank: {id}")).clicked() {
                                    drugbank::open_overview(id);
                                }
                            }
                            MolIdent::PubChem(cid) => {
                                if ui.button(format!("PubChem: {cid}")).clicked() {
                                    pubchem::open_overview(*cid);
                                }
                                pubchem_cid = Some(*cid);
                            }
                            MolIdent::PdbeAmber(id) => {
                                if ui.button(format!("PDBe: {id}")).clicked() {
                                    pdbe::open_overview(id);
                                }
                            }
                            _ => (),
                        }
                    }

                    // If we have a PDBE ID, but no PubChem ID, try to find one using
                    // an intermediate SMILES representation, upon clicking the button.
                    if pubchem_cid.is_none() {
                        if ui.button("PubChem").clicked() {
                            // If we already have SMILES, this saves an API call.
                            for ident in &m.idents {
                                if let MolIdent::Smiles(smiles) = ident {
                                    // todo: Don't block ?
                                    let cids = pubchem::find_cids_from_search(&smiles, true)
                                        .unwrap_or_default();

                                    if !cids.is_empty() {
                                        let cid = cids[0];
                                        update_cid = Some(cid);
                                        pubchem::open_overview(cid);
                                    }

                                    break;
                                }
                            }

                            // This runs if we have neither CID, nor SMILES.
                            if let Ok((cid, _smiles)) =
                                pubchem::get_cid_from_pdbe_id(&mol.common().ident)
                            {
                                update_cid = Some(cid);
                                pubchem::open_overview(cid);
                            }
                        }
                    }

                    if let Some(cid) = pubchem_cid {
                        if ui.button("Find assoc structs").clicked() {
                            // todo: Don't block.
                            if m.associated_structures.is_empty() {
                                match pubchem::load_associated_structures(cid) {
                                    Ok(data) => {
                                        // todo: Put back! Borrow issue.
                                        // mol.common().associated_structures = data;
                                        state.ui.popup.show_associated_structures = true;
                                    }
                                    Err(_) => handle_err(
                                        &mut state.ui,
                                        "Unable to find structures for this ligand".to_owned(),
                                    ),
                                }
                            } else {
                                state.ui.popup.show_associated_structures = true;
                            }
                        }
                    }
                }
                MolGenericRef::Lipid(l) => {
                    if ui.button("View on LMSD").clicked() {
                        lmsd::open_overview(&l.lmsd_id);
                    }
                }
                _ => (),
            }
        }

        if let Some(mol) = state.active_mol_mut() {
            if let Some(cid) = update_cid
                && let MolGenericRefMut::Small(m) = mol
            {
                m.idents.push(MolIdent::PubChem(cid));
            }
        }
        ui.add_space(COL_SPACING);
    });
}

/// Display metadata stored for a given molecule.
pub(super) fn metadata(mol_type: MolType, i: usize, state: &mut State, ui: &mut Ui) {
    let mol = match mol_type {
        MolType::Peptide => {
            if state.peptide.is_none() {
                return;
            }
            &state.peptide.as_ref().unwrap().common
        }
        MolType::Ligand => {
            if i >= state.ligands.len() {
                return;
            }
            &state.ligands[i].common
        }
        MolType::NucleicAcid => {
            if i >= state.nucleic_acids.len() {
                return;
            }
            &state.nucleic_acids[i].common
        }
        MolType::Lipid => {
            if i >= state.lipids.len() {
                return;
            }
            &state.lipids[i].common
        }
        _ => return,
    };

    popups::metadata_popup(&mut state.ui.popup, mol, ui);
}
