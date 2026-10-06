use bio_files::md_params::ForceFieldParams;
use egui::{Color32, ComboBox, Id, Response, RichText, Sense, Stroke, TextEdit, Ui, vec2};
use graphics::{ControlScheme, EngineUpdates, Scene};
use lin_alg::f64::Vec3;
use mol_defs::molecules::{
    MolGenericRef, MolType, common::MoleculeCommon, nucleic_acid::NucleicAcidType,
};

use crate::{
    button,
    cam::{MolCameraTarget, VIEW_DIR_FRONT, move_cam_to_mol, set_fog},
    file_io::sequence::save_seq_dialog,
    label,
    state::{MetadataTarget, MolPickerState, MolSort, State},
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE, highlighted_box,
        panels::md_viewer::mol_type_label, popup::pharmacophore, sidebar, sidebar::AudioAction,
    },
    util::{RedrawFlags, close_mol, orbit_center},
};

/// Molecule types shown in the picker, in their default order.
const MOL_TYPES: [MolType; 5] = [
    MolType::Peptide,
    MolType::Ligand,
    MolType::Lipid,
    MolType::NucleicAcid,
    MolType::Pocket,
];

const SORTS: [(MolSort, &str); 11] = [
    (MolSort::Manual, "Manual"),
    (MolSort::Name, "Name"),
    (MolSort::Type, "Type"),
    (MolSort::AtomCount, "Atom count"),
    (MolSort::Weight, "Weight"),
    (MolSort::LogP, "LogP"),
    (MolSort::Tpsa, "TPSA"),
    (MolSort::Rings, "Rings"),
    (MolSort::RotatableBonds, "Rot. bonds"),
    (MolSort::HBondDonors, "H donors"),
    (MolSort::HBondAcceptors, "H acceptors"),
];

/// The drag-and-drop payload of a molecule row being rearranged.
struct DraggedMol(MolType, usize);

/// For a molecule or sequence row's select button: the name, truncated if long, and its hover
/// text. When truncated, the hover text starts with the full name.
fn picker_name(name: &str, help: &str) -> (String, String) {
    const MAX_NAME_LEN: usize = 30;

    if name.chars().count() > MAX_NAME_LEN {
        let truncated: String = name.chars().take(MAX_NAME_LEN - 1).collect();
        (format!("{truncated}…"), format!("{name}\n\n{help}"))
    } else {
        (name.to_owned(), help.to_owned())
    }
}

/// Actions requested by picker rows. Applied after all rows are drawn; avoids borrow errors.
#[derive(Default)]
struct PickerActions {
    recenter_orbit: bool,
    close: Option<(MolType, usize)>,
    reset_fog: bool,
    audio_action: Option<AudioAction>,
}

/// A grip to drag a molecule row by, to rearrange the rows.
fn drag_handle(ui: &mut Ui, mol_type: MolType, i_mol: usize) {
    let id = Id::new(("mol_picker_drag", mol_type as u8, i_mol));

    ui.dnd_drag_source(id, DraggedMol(mol_type, i_mol), |ui| {
        let size = vec2(8., ui.spacing().interact_size.y);
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());

        // A 2x3 grid of dots.
        for x in [-2., 2.] {
            for y in [-4., 0., 4.] {
                ui.painter()
                    .circle_filled(rect.center() + vec2(x, y), 1., Color32::GRAY);
            }
        }
    })
    .response
    .on_hover_text("Drag up or down to rearrange molecules.");
}

/// This displays a single molecule, for selection as the active molecule, and
/// a limited set of information and functionality. A list of these is displayed for
/// open molecules. Returns the row's response, for dropping dragged rows onto.
fn mol_picker_one(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    mol_type: MolType,
    i_mol: usize,
    actions: &mut PickerActions,
) -> Option<Response> {
    // Idents, characterization, and pharmacophore are for small mols only.
    let (mol, idents, mol_char, pharmacophore) = match mol_type {
        MolType::Peptide => (&mut state.peptides[i_mol].common, None, None, None),
        MolType::Ligand => {
            let m = &mut state.ligands[i_mol];
            let char = m.characterization.as_ref();
            (&mut m.common, Some(&m.idents), char, Some(&m.pharmacophore))
        }
        MolType::Lipid => (&mut state.lipids[i_mol].common, None, None, None),
        MolType::NucleicAcid => (&mut state.nucleic_acids[i_mol].common, None, None, None),
        MolType::Pocket => (&mut state.pockets[i_mol].common, None, None, None),
        MolType::Water => return None,
    };

    let active = state.volatile.active_mol == Some((mol_type, i_mol));

    let color = if active {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    };

    let row = highlighted_box(active, Color32::from_rgb(55, 40, 40)).show(ui, |ui| {
        ui.horizontal(|ui| {
            drag_handle(ui, mol_type, i_mol);

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(COL_SPACING / 2.);
                if ui
                    .button(RichText::new("❌").color(Color32::LIGHT_RED))
                    .on_hover_text("(Hotkey: Delete) Close this molecule.")
                    .clicked()
                {
                    actions.close = Some((mol_type, i_mol));
                }

                if mol_type != MolType::Pocket {
                    if let Some(copies) = &mut mol.selected_for_md {
                        sidebar::md_copies_field(copies, ui);
                    }

                    let color_md = if mol.selected_for_md.is_some() {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if ui
                        .button(RichText::new("MD").color(color_md))
                        .on_hover_text(
                            "Select or deselect this molecule for molecular dynamics simulation.",
                        )
                        .clicked()
                    {
                        mol.selected_for_md = match mol.selected_for_md {
                            Some(_) => None,
                            None => Some(1),
                        };
                    }
                }

                let color_vis = if mol.visible {
                    COLOR_ACTIVE
                } else {
                    COLOR_INACTIVE
                };

                if ui.button(RichText::new("👁").color(color_vis)).clicked() {
                    mol.visible = !mol.visible;

                    redraw.set(mol_type); // todo Overkill; only need to redraw (or even just clear) one.
                }

                if ui
                    .button(RichText::new("Cam"))
                    .on_hover_text("Move camera near near this molecule, looking at it.")
                    .clicked()
                {
                    let molecule_center: Vec3 = mol.centroid().into();
                    let forward: Vec3 = VIEW_DIR_FRONT.into();
                    let alignment = molecule_center + forward;

                    move_cam_to_mol(
                        MolCameraTarget::new(mol, (mol_type, i_mol)),
                        &mut state.ui.cam_snapshot,
                        scene,
                        &mut state.volatile.orbit_center,
                        alignment,
                        updates,
                    );
                    actions.reset_fog = true;
                }

                let row_h = ui.spacing().interact_size.y;

                let (name_disp, help_text) = picker_name(
                    &mol.name(idents),
                    "Make this molecule the active / selected one. Middle click to close it.",
                );

                let sel_btn = ui
                    .add_sized(
                        egui::vec2(ui.available_width(), row_h),
                        egui::Button::new(RichText::new(name_disp).color(color)),
                    )
                    .on_hover_text(help_text);

                if sel_btn.clicked() {
                    if active && state.volatile.active_mol.is_some() {
                        state.volatile.active_mol = None;
                    } else {
                        state.volatile.active_mol = Some((mol_type, i_mol));
                        state.volatile.orbit_center = state.volatile.active_mol;

                        actions.recenter_orbit = true;
                    }

                    redraw.set(mol_type); // To reflect the change in thickness, color etc.
                }

                if sel_btn.middle_clicked() {
                    actions.close = Some((mol_type, i_mol));
                }
            });
        });

        if state.ui.ui_vis.mol_picker_details {
            if let Some(char) = mol_char {
                let color_details = if active {
                    Color32::WHITE
                } else {
                    Color32::GRAY
                };

                label!(ui, char.to_string().trim(), color_details);
            }

            if let Some(pm) = pharmacophore
                && !pm.features.is_empty()
            {
                let (popup, ph_state) = (&mut state.ui.popup, &mut state.pharmacophore);
                pharmacophore::pharmacophore_summary(pm, i_mol, popup, ph_state, ui);
            }
            //
            // let playing_this_mol = state
            //     .volatile
            //     .playing_audio
            //     .as_ref()
            //     .is_some_and(|audio| audio.is_for(mol_type, i_mol));
            //
            // let (text, color, hover_text) = if playing_this_mol {
            //     ("Pause", COLOR_ACTIVE, "Stop sonifying this molecule.")
            // } else {
            //     (
            //         "Play",
            //         COLOR_ACTION,
            //         "Sonify this molecule using its force-field bond-stretching parameters.",
            //     )
            // };
            //
            // if mol_type != MolType::Pocket
            //     && ui
            //         .button(RichText::new(text).color(color))
            //         .on_hover_text(hover_text)
            //         .clicked()
            // {
            //     actions.audio_action = Some(AudioAction::Toggle(mol_type, i_mol));
            // }
        }

        ui.separator();
    });

    Some(row.response)
}

/// Search, molecule type filters, and sorting, above the molecule rows.
fn picker_toolbar(p: &mut MolPickerState, present: &[MolType], ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if !p.search.is_empty()
                && button!(ui, "❌", Color32::LIGHT_RED, "Clear the search.").clicked()
            {
                p.search.clear();
            }

            ui.add(
                TextEdit::singleline(&mut p.search)
                    .hint_text("Search names, idents, files")
                    .desired_width(ui.available_width()),
            );
        });
    });

    ui.horizontal_wrapped(|ui| {
        // Type filters are only useful with several types open, but stay up while any type is
        // hidden, so it can be shown again.
        if present.len() >= 2 || present.iter().any(|t| p.hidden_types.contains(t)) {
            for &t in present {
                let hidden = p.hidden_types.contains(&t);
                let color = if hidden { COLOR_INACTIVE } else { COLOR_ACTIVE };

                if button!(
                    ui,
                    mol_type_label(t),
                    color,
                    "Show or hide molecules of this type."
                )
                .clicked()
                {
                    if hidden {
                        p.hidden_types.retain(|&h| h != t);
                    } else {
                        p.hidden_types.push(t);
                    }
                }
            }

            ui.add_space(COL_SPACING / 2.);
        }

        let sort_label = SORTS
            .iter()
            .find(|(s, _)| *s == p.sort)
            .map_or("", |(_, l)| *l);

        ComboBox::from_id_salt("mol_picker_sort")
            .selected_text(format!("Sort: {sort_label}"))
            .show_ui(ui, |ui| {
                for (sort, label) in SORTS {
                    ui.selectable_value(&mut p.sort, sort, label);
                }
            })
            .response
            .on_hover_text(
                "Sort molecules. Characterization-based sorts (LogP etc) apply to small \
                molecules; others are placed last. Drag a molecule to rearrange manually.",
            );

        if p.sort != MolSort::Manual {
            let (text, help) = if p.sort_descending {
                ("⬇", "Sorted descending. Click to sort ascending.")
            } else {
                ("⬆", "Sorted ascending. Click to sort descending.")
            };

            if button!(ui, text, COLOR_INACTIVE, help).clicked() {
                p.sort_descending = !p.sort_descending;
            }
        }
    });
}

/// Updates the manual order for molecules opened or closed since the last frame. New molecules
/// go after the last one of their type (or an earlier type), as they would in the default order.
fn sync_order(order: &mut Vec<(MolType, usize)>, counts: &[(MolType, usize)]) {
    let count = |t: MolType| counts.iter().find(|c| c.0 == t).map_or(0, |c| c.1);
    order.retain(|&(t, i)| i < count(t));

    if order.len() == counts.iter().map(|c| c.1).sum::<usize>() {
        return;
    }

    let rank = |t: MolType| MOL_TYPES.iter().position(|&t2| t2 == t);

    for &(mol_type, count) in counts {
        for i in 0..count {
            if order.contains(&(mol_type, i)) {
                continue;
            }

            let pos = order
                .iter()
                .rposition(|&(t, _)| rank(t) <= rank(mol_type))
                .map_or(0, |p| p + 1);

            order.insert(pos, (mol_type, i));
        }
    }
}

/// Whether the molecule's name, filename, or any of its identifiers contain the query.
/// `query` must be lowercase.
fn matches_search(state: &State, (mol_type, i): (MolType, usize), query: &str) -> bool {
    let Some(mol) = state.get_mol(mol_type, i) else {
        return false;
    };

    let c = mol.common();
    let mut fields = vec![c.ident.clone(), c.filename.clone()];
    fields.extend(c.name.clone());

    match mol {
        MolGenericRef::Small(m) => fields.extend(m.idents.iter().map(|id| id.ident_inner())),
        MolGenericRef::Peptide(m) => fields.extend(m.idents.iter().map(|id| id.ident_inner())),
        MolGenericRef::Lipid(m) => {
            fields.extend([&m.common_name, &m.lmsd_id, &m.hmdb_id, &m.kegg_id].map(String::clone))
        }
        _ => (),
    }

    fields.iter().any(|f| f.to_lowercase().contains(query))
}

/// A molecule's value for numerical sorts. `None` if it has none, e.g. for characterization-based
/// sorts on molecules other than small ones.
fn sort_val(state: &State, (mol_type, i): (MolType, usize), sort: MolSort) -> Option<f32> {
    let mol = state.get_mol(mol_type, i)?;
    let c = mol.common();

    let char = match &mol {
        MolGenericRef::Small(m) => m.characterization.as_ref(),
        _ => None,
    };

    Some(match sort {
        MolSort::Type => MOL_TYPES.iter().position(|&t| t == mol_type)? as f32,
        MolSort::AtomCount => c.atoms.len() as f32,
        MolSort::Weight => c.atomic_weight(),
        MolSort::LogP => char?.log_p,
        MolSort::Tpsa => char?.tpsa_ertl,
        MolSort::Rings => char?.rings.len() as f32,
        MolSort::RotatableBonds => char?.rotatable_bonds.len() as f32,
        MolSort::HBondDonors => char?.h_bond_donor.len() as f32,
        MolSort::HBondAcceptors => char?.h_bond_acceptor.len() as f32,
        MolSort::Manual | MolSort::Name => return None,
    })
}

/// Moves a dragged molecule row before or after another. Rearranging sorted rows starts the
/// manual order from the sorted one.
fn move_mol(state: &mut State, dragged: (MolType, usize), target: (MolType, usize), after: bool) {
    if dragged == target {
        return;
    }

    let p = &state.ui.mol_picker;
    let mut order = p.order.clone();
    sort_mols(state, &mut order, p.sort, p.sort_descending);

    order.retain(|&id| id != dragged);
    if let Some(pos) = order.iter().position(|&id| id == target) {
        order.insert(pos + after as usize, dragged);
    }

    state.ui.mol_picker.order = order;
    state.ui.mol_picker.sort = MolSort::Manual;
}

/// Sorts molecules, starting from their manual order; ties keep it.
fn sort_mols(state: &State, ids: &mut [(MolType, usize)], sort: MolSort, descending: bool) {
    match sort {
        MolSort::Manual => (),
        MolSort::Name => {
            ids.sort_by_cached_key(|&(t, i)| match state.get_mol(t, i) {
                Some(MolGenericRef::Small(m)) => m.common.name(Some(&m.idents)).to_lowercase(),
                Some(m) => m.common().name(None).to_lowercase(),
                None => String::new(),
            });

            if descending {
                ids.reverse();
            }
        }
        _ => {
            let mut keyed: Vec<_> = ids
                .iter()
                .map(|&id| (sort_val(state, id, sort), id))
                .collect();

            keyed.sort_by(|(a, _), (b, _)| match (a, b) {
                (Some(a), Some(b)) if descending => b.total_cmp(a),
                (Some(a), Some(b)) => a.total_cmp(b),
                // Molecules without a value go last, in either direction.
                _ => b.is_some().cmp(&a.is_some()),
            });

            for (id, (_, sorted)) in ids.iter_mut().zip(keyed) {
                *id = sorted;
            }
        }
    }
}

pub fn sonification_input(
    state: &State,
    mol_type: MolType,
    i_mol: usize,
) -> Result<(MoleculeCommon, ForceFieldParams), String> {
    match mol_type {
        MolType::Peptide => {
            let mol = state
                .peptides
                .get(i_mol)
                .ok_or_else(|| "Peptide index is out of bounds.".to_string())?;
            let params = state
                .ff_param_set
                .peptide
                .as_ref()
                .ok_or_else(|| "No peptide force-field parameters are loaded.".to_string())?;

            Ok((mol.common.clone(), params.clone()))
        }
        MolType::Ligand => {
            let mol = state
                .ligands
                .get(i_mol)
                .ok_or_else(|| "Ligand index is out of bounds.".to_string())?;
            let params = state.ff_param_set.small_mol.as_ref().ok_or_else(|| {
                "No small-molecule force-field parameters are loaded.".to_string()
            })?;

            Ok((
                mol.common.clone(),
                sidebar::merge_mol_specific_params(
                    params,
                    sidebar::mol_specific_params(&state.mol_specific_params, &mol.common.ident),
                ),
            ))
        }
        MolType::NucleicAcid => {
            let mol = state
                .nucleic_acids
                .get(i_mol)
                .ok_or_else(|| "Nucleic acid index is out of bounds.".to_string())?;
            let params = match mol.na_type {
                NucleicAcidType::Dna => &state.ff_param_set.dna,
                NucleicAcidType::Rna => &state.ff_param_set.rna,
            }
            .as_ref()
            .ok_or_else(|| format!("No {} force-field parameters are loaded.", mol.na_type))?;

            Ok((mol.common.clone(), params.clone()))
        }
        MolType::Lipid => {
            let mol = state
                .lipids
                .get(i_mol)
                .ok_or_else(|| "Lipid index is out of bounds.".to_string())?;
            let params = state
                .ff_param_set
                .lipids
                .as_ref()
                .ok_or_else(|| "No lipid force-field parameters are loaded.".to_string())?;

            Ok((mol.common.clone(), params.clone()))
        }
        MolType::Pocket | MolType::Water => {
            Err("This molecule type cannot be sonified.".to_string())
        }
    }
}

/// Select, close, hide etc molecules from ones opened.
pub fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let mut actions = PickerActions::default();

    let counts = [
        (MolType::Peptide, state.peptides.len()),
        (MolType::Ligand, state.ligands.len()),
        (MolType::Lipid, state.lipids.len()),
        (MolType::NucleicAcid, state.nucleic_acids.len()),
        (MolType::Pocket, state.pockets.len()),
    ];
    let total: usize = counts.iter().map(|c| c.1).sum();
    let present: Vec<_> = counts.iter().filter(|c| c.1 > 0).map(|c| c.0).collect();

    let p = &mut state.ui.mol_picker;
    sync_order(&mut p.order, &counts);

    let filtering =
        !p.search.trim().is_empty() || present.iter().any(|t| p.hidden_types.contains(t));

    if total >= 2 || filtering {
        picker_toolbar(p, &present, ui);
    }

    // The molecules to show, in display order.
    let p = &state.ui.mol_picker;
    let query = p.search.trim().to_lowercase();

    let mut ids: Vec<_> = p
        .order
        .iter()
        .copied()
        .filter(|(t, _)| !p.hidden_types.contains(t))
        .filter(|&id| query.is_empty() || matches_search(state, id, &query))
        .collect();

    sort_mols(state, &mut ids, p.sort, p.sort_descending);

    if ids.len() < total {
        label!(
            ui,
            format!("Showing {} of {total}", ids.len()),
            Color32::GRAY
        );
    }

    // A row dropped onto another: (dragged, target, whether to place it after the target).
    let mut dropped = None;

    for &(mol_type, i) in &ids {
        let row = mol_picker_one(state, scene, ui, updates, redraw, mol_type, i, &mut actions);

        // Mark where a dragged row will go, and move it there on release.
        if let Some(row) = row
            && row.dnd_hover_payload::<DraggedMol>().is_some()
            && let Some(pointer) = ui.ctx().pointer_interact_pos()
        {
            let after = pointer.y > row.rect.center().y;
            let y = if after {
                row.rect.bottom()
            } else {
                row.rect.top()
            };
            ui.painter()
                .hline(row.rect.x_range(), y, Stroke::new(2., COLOR_ACTIVE));

            if let Some(dragged) = row.dnd_release_payload::<DraggedMol>() {
                dropped = Some(((dragged.0, dragged.1), (mol_type, i), after));
            }
        }
    }

    if let Some((dragged, target, after)) = dropped {
        move_mol(state, dragged, target, after);
    }

    // Removed, for now.

    // for (i_mol, pm) in state.pharmacophores.iter_mut().enumerate() {
    //     label!(
    //         ui,
    //         format!("Pharmacophore name: {} ident: {}", pm.name, pm.mol_ident),
    //         Color32::WHITE
    //     );
    //     //
    //     // if let Some(pm) = pharmacophore
    //     //     && !pm.features.is_empty()
    //     // {
    //     pharmacophore::pharmacophore_summary(
    //         pm,
    //         i_mol,
    //         &mut state.ui.popup,
    //         &mut state.pharmacophore,
    //         ui,
    //     );
    //     // }
    // }

    // todo: AAs here too?

    // Peptide-relative UI state follows whichever peptide was selected in the picker.
    if actions.recenter_orbit {
        state.volatile.active_seq = None;
        state.ui.seq_selection.clear();
    }

    if actions.recenter_orbit
        && let Some((MolType::Peptide, peptide_i)) = state.volatile.active_mol
        && let Some(peptide) = state.peptides.get(peptide_i)
    {
        state.volatile.active_peptide = Some(peptide_i);
        state.volatile.set_aa_seq(Some(peptide));
        state.volatile.flags.ss_mesh_created = false;
        state.volatile.flags.sas_mesh_created = false;
    }
    if let Some(AudioAction::Toggle(mol_type, i_mol)) = actions.audio_action {
        sidebar::toggle_audio(state, mol_type, i_mol);
    }

    if let Some((mol_type, i_mol)) = actions.close {
        close_mol(mol_type, i_mol, state, scene, redraw, updates);
    }

    if actions.recenter_orbit
        && let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme
    {
        *center = orbit_center(state);
    }

    if actions.reset_fog {
        set_fog(state, &mut scene.camera);
    }
}

/// Select, close, and save opened sequences. Like the molecule rows, but sequences aren't
/// rendered, so there are no visibility, camera, or MD controls.
pub fn seq_picker(state: &mut State, ui: &mut Ui) {
    // Applied after the loop; avoids borrow errors, and index shifts while iterating.
    let mut close = None;
    let mut save = None;
    let mut toggle_metadata = None;
    let mut toggle_active = None;

    for (i, seq) in state.sequences.iter().enumerate() {
        let active = state.volatile.active_seq == Some(i);

        let color = if active {
            COLOR_ACTIVE_RADIO
        } else {
            COLOR_INACTIVE
        };

        highlighted_box(active, Color32::from_rgb(40, 45, 60)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(COL_SPACING / 2.);
                    if ui
                        .button(RichText::new("❌").color(Color32::LIGHT_RED))
                        .on_hover_text("Close this sequence.")
                        .clicked()
                    {
                        close = Some(i);
                    }

                    let color_meta = if state.ui.popup.metadata == Some(MetadataTarget::Seq(i)) {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if button!(
                        ui,
                        "Meta",
                        color_meta,
                        "Display and edit this sequence's metadata."
                    )
                    .clicked()
                    {
                        toggle_metadata = Some(i);
                    }

                    if button!(
                        ui,
                        "Save",
                        COLOR_INACTIVE,
                        "Save this sequence to a FASTA, GenBank, or SnapGene (DNA only) file."
                    )
                    .clicked()
                    {
                        save = Some(i);
                    }

                    let row_h = ui.spacing().interact_size.y;

                    let (name_disp, help_text) = picker_name(
                        seq.display_name(),
                        "Make this sequence the active / selected one. Middle click to close it.",
                    );

                    let sel_btn = ui
                        .add_sized(
                            egui::vec2(ui.available_width(), row_h),
                            egui::Button::new(RichText::new(name_disp).color(color)),
                        )
                        .on_hover_text(help_text);

                    if sel_btn.clicked() {
                        toggle_active = Some(i);
                    }

                    if sel_btn.middle_clicked() {
                        close = Some(i);
                    }
                });
            });

            let color_details = if active {
                Color32::WHITE
            } else {
                Color32::GRAY
            };

            let details = format!(
                "{} · {} {}",
                seq.seq_type(),
                seq.data.len(),
                seq.seq_type().residue_unit()
            );

            let resp = label!(ui, details, color_details);
            if let Some(descrip) = seq.description() {
                resp.on_hover_text(descrip);
            }

            ui.separator();
        });
    }

    if let Some(i) = toggle_active {
        let next = if state.volatile.active_seq == Some(i) {
            None
        } else {
            Some(i)
        };
        state.select_sequence(next);
    }

    if let Some(i) = toggle_metadata {
        let target = MetadataTarget::Seq(i);

        state.ui.popup.metadata = if state.ui.popup.metadata == Some(target) {
            None
        } else {
            Some(target)
        };
    }

    if let Some(i) = save {
        save_seq_dialog(state, i);
    }

    if let Some(i) = close {
        state.close_sequence(i);
    }
}
