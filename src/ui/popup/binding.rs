//! Practical binding workflows over loaded structures, with a prepared Boltz-2 runner.

use std::{
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::Duration,
};

use egui::{Color32, ComboBox, DragValue, ScrollArea, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::MolType;

use super::tool_runner::ToolWindow;
use crate::{
    drug_design::binding::{self, PoseReport, PredictionOptions, PredictionRequest},
    protein_alignment,
    selection::{MoleculeAtoms, Selection, expression},
    state::State,
    util::{RedrawFlags, handle_err},
};

pub struct BindingUi {
    protein: String,
    ligand: String,
    chain: Option<usize>,
    smiles: String,
    residue_filter: String,
    pocket: Vec<usize>,
    pocket_source: Option<u64>,
    options: PredictionOptions,
    pending: Option<Receiver<Result<PoseReport, String>>>,
    report: Option<PoseReport>,
    message: String,
    runner: ToolWindow,
}

impl Default for BindingUi {
    fn default() -> Self {
        Self {
            protein: String::new(),
            ligand: String::new(),
            chain: None,
            smiles: String::new(),
            residue_filter: String::new(),
            pocket: Vec::new(),
            pocket_source: None,
            options: PredictionOptions::default(),
            pending: None,
            report: None,
            message: String::new(),
            runner: ToolWindow::for_binding(),
        }
    }
}

/// Poll while closed too, so completed jobs release their receiver and retain their results.
pub fn poll(state: &mut State, ui: &Ui) {
    let editor = &mut state.ui.binding;
    editor.runner.poll();
    if let Some(receiver) = &editor.pending {
        match receiver.try_recv() {
            Ok(result) => {
                editor.pending = None;
                match result {
                    Ok(report) => {
                        editor.report = Some(report);
                        editor.message.clear();
                    }
                    Err(error) => editor.message = error,
                }
            }
            Err(TryRecvError::Empty) => ui.ctx().request_repaint_after(Duration::from_millis(100)),
            Err(TryRecvError::Disconnected) => {
                editor.pending = None;
                editor.message = "Pose assessment worker stopped without a result.".into();
            }
        }
    }
}

fn object_picker(state: &State, key: &mut String, protein: bool, ui: &mut Ui) -> Option<usize> {
    let objects: Vec<_> = expression::objects(state)
        .into_iter()
        .filter(|(kind, _, _)| {
            *kind
                == if protein {
                    MolType::Peptide
                } else {
                    MolType::Ligand
                }
        })
        .collect();
    let label = if protein { "Protein" } else { "Ligand" };
    // Only initialize an empty picker; never retarget a selection whose object was closed.
    if key.is_empty()
        && let Some((_, _, mol)) = objects.first()
    {
        *key = expression::object_key(mol);
    }
    let matches: Vec<_> = objects
        .iter()
        .filter(|(_, _, mol)| expression::object_key(mol) == *key)
        .collect();
    let selected = if matches.len() == 1 {
        matches[0].2.name().into_owned()
    } else {
        format!("Choose {label}…")
    };
    ui.horizontal(|ui| {
        ui.label(label);
        ComboBox::from_id_salt(label)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                for (_, _, mol) in &objects {
                    ui.selectable_value(key, expression::object_key(mol), mol.name().as_ref());
                }
            });
    });
    let mut indices = objects
        .iter()
        .filter(|(_, _, mol)| expression::object_key(mol) == *key)
        .map(|(_, index, _)| *index);
    let index = indices.next();
    if indices.next().is_some() {
        ui.colored_label(Color32::ORANGE, "These objects have the same identifier. Give them distinct source paths or identifiers.");
        return None;
    }
    index
}

fn select_atoms(
    state: &mut State,
    protein: usize,
    ligand: usize,
    residues: &[usize],
    clashes: bool,
    report: &PoseReport,
) {
    let receptor = &state.peptides[protein];
    let mut protein_atoms = Vec::new();
    let mut ligand_atoms = Vec::new();
    if clashes {
        for pair in &report.clashes {
            protein_atoms.extend(
                receptor
                    .common
                    .atoms
                    .iter()
                    .enumerate()
                    .filter_map(|(i, a)| (a.serial_number == pair.protein_atom_sn).then_some(i)),
            );
            ligand_atoms.extend(
                state.ligands[ligand]
                    .common
                    .atoms
                    .iter()
                    .enumerate()
                    .filter_map(|(i, a)| (a.serial_number == pair.ligand_atom_sn).then_some(i)),
            );
        }
    } else {
        for &r in residues {
            if let Some(residue) = receptor.residues.get(r) {
                protein_atoms.extend_from_slice(&residue.atoms);
            }
        }
    }
    protein_atoms.sort_unstable();
    protein_atoms.dedup();
    ligand_atoms.sort_unstable();
    ligand_atoms.dedup();
    let mut selected = vec![MoleculeAtoms {
        mol_type: MolType::Peptide,
        mol_i: protein,
        atoms: protein_atoms,
    }];
    if !ligand_atoms.is_empty() {
        selected.push(MoleculeAtoms {
            mol_type: MolType::Ligand,
            mol_i: ligand,
            atoms: ligand_atoms,
        });
    }
    state.ui.selection = Selection::Atoms(selected);
}

pub fn editor(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let mut editor = std::mem::take(&mut state.ui.binding);
    ui.heading("Protein–ligand binding");
    ui.label("Assess a displayed pose, or predict a new complex and its affinity.");
    let old_protein = editor.protein.clone();
    let old_ligand = editor.ligand.clone();
    let old_chain = editor.chain;
    let mut protein_i = None;
    let mut ligand_i = None;
    ui.add_enabled_ui(!editor.runner.busy(), |ui| {
        protein_i = object_picker(state, &mut editor.protein, true, ui);
        ligand_i = object_picker(state, &mut editor.ligand, false, ui);
        if old_protein != editor.protein {
            editor.chain = protein_i
                .and_then(|i| {
                    protein_alignment::chains(&state.peptides[i])
                        .first()
                        .map(|c| c.0)
                })
                .flatten();
        }
        if old_ligand != editor.ligand {
            editor.smiles = ligand_i
                .and_then(|i| state.ligands[i].get_smiles())
                .unwrap_or("")
                .into();
        }
        if let Some(i) = protein_i {
            let chains = protein_alignment::chains(&state.peptides[i]);
            if editor.chain.is_none() && !chains.iter().any(|c| c.0.is_none()) {
                editor.chain = chains.first().and_then(|c| c.0);
            }
            let selected = chains
                .iter()
                .find(|c| c.0 == editor.chain)
                .map(|c| c.1.as_str())
                .unwrap_or("Choose chain…");
            ComboBox::from_id_salt("binding_chain")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (chain, label) in chains {
                        ui.selectable_value(&mut editor.chain, chain, label);
                    }
                });
        }
    });
    if old_protein != editor.protein || old_chain != editor.chain || old_ligand != editor.ligand {
        editor.pocket.clear();
        editor.pocket_source = None;
    }

    let load;
    if let (Some(protein_i), Some(ligand_i)) = (protein_i, ligand_i) {
        let protein = &state.peptides[protein_i];
        let ligand = &state.ligands[ligand_i];
        let current = binding::fingerprint(protein, &ligand.common, editor.chain);
        let report_current = editor
            .report
            .as_ref()
            .is_some_and(|r| r.fingerprint == current);
        ui.separator();
        ui.strong("1. Is the displayed pose geometrically plausible?");
        ui.small("Selected chain only. Heavy-atom contacts ≤4 Å; clashes have >0.8 Å van der Waals overlap. Waters, cofactors, and other chains are excluded.");
        if ui
            .add_enabled(
                editor.pending.is_none(),
                egui::Button::new("Assess current pose"),
            )
            .clicked()
        {
            let protein = protein.clone();
            let ligand = ligand.common.clone();
            let chain = editor.chain;
            let context = ui.ctx().clone();
            let (tx, rx) = mpsc::channel();
            editor.pending = Some(rx);
            editor.report = None;
            editor.message.clear();
            thread::spawn(move || {
                let _ = tx.send(binding::assess_pose(&protein, &ligand, chain));
                context.request_repaint();
            });
        }
        if editor.pending.is_some() {
            ui.spinner();
        }
        let mut select_residues = None;
        let mut select_clashes = false;
        if let Some(report) = &editor.report {
            if !report_current {
                ui.colored_label(Color32::ORANGE, "Stale pose report: the selected structure or coordinates changed. Assess again.");
            }
            ui.label(format!(
                "{} / {} ligand heavy atoms contact {} residues; {} clashing atom pairs.",
                report.contacting_ligand_atoms,
                report.ligand_heavy_atoms,
                report.contacts.len(),
                report.clash_count
            ));
            if !report.bond_length_outliers.is_empty() {
                ui.colored_label(
                    Color32::ORANGE,
                    format!(
                        "{} gross ligand bond-length outliers.",
                        report.bond_length_outliers.len()
                    ),
                );
            }
            if report.contacts.is_empty() {
                ui.colored_label(Color32::ORANGE, "No protein contacts in this pose. Predict a complex to explore a binding site.");
            }
            if report.clash_count > 0 {
                ui.colored_label(Color32::ORANGE, "Steric overlaps need inspection. Covalent attachments can also create overlaps in this noncovalent screen.");
            }
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(
                        report_current,
                        egui::Button::new("Select contacting residues"),
                    )
                    .clicked()
                {
                    select_residues = Some(
                        report
                            .contacts
                            .iter()
                            .map(|c| c.residue)
                            .collect::<Vec<_>>(),
                    );
                }
                select_clashes = ui
                    .add_enabled(
                        report_current && report.clash_count > 0,
                        egui::Button::new("Select clash atoms"),
                    )
                    .clicked();
                if ui.button("Copy pose report (JSON)").clicked() {
                    ui.ctx()
                        .copy_text(serde_json::to_string_pretty(report).unwrap_or_default());
                }
            });
            egui::CollapsingHeader::new("Contact and clash details").show(ui, |ui| {
                ScrollArea::vertical()
                    .id_salt("binding_pose_details")
                    .max_height(180.)
                    .show(ui, |ui| {
                        for contact in &report.contacts {
                            ui.label(format!(
                                "{}: {:.2} Å closest, {} atom pairs",
                                contact.label, contact.closest_distance_a, contact.atom_pairs
                            ));
                        }
                        for clash in &report.clashes {
                            ui.label(format!(
                                "Protein atom {} / ligand atom {}: {:.2} Å, overlap {:.2} Å",
                                clash.protein_atom_sn,
                                clash.ligand_atom_sn,
                                clash.distance_a,
                                clash.overlap_a
                            ));
                        }
                        for outlier in &report.bond_length_outliers {
                            ui.colored_label(Color32::ORANGE, outlier);
                        }
                    });
            });
            ui.small("A geometric screen only: heavy-atom bond lengths outside 0.7–2.5 Å are flagged. Protonation, stereochemistry, intraligand strain, and directional interactions are not validated. Passing does not establish binding.");
        }

        ui.separator();
        ui.strong("2. Where could it bind, and how strongly?");
        ui.label("Boltz-2 generates a new complex from the selected chain and ligand SMILES. Its affinity estimate belongs to that generated complex.");
        ui.add_enabled_ui(!editor.runner.busy(), |ui| {
            ui.label("Ligand SMILES (review stereochemistry and protonation)");
            ui.add(TextEdit::singleline(&mut editor.smiles).desired_width(f32::INFINITY));
            ui.small("The prediction uses this SMILES. The pose check uses the loaded ligand's coordinates.");
            ui.small("Boltz affinity supports small molecules up to 128 atoms after RDKit RemoveHs; its training domain mainly covers ligands up to 56 atoms.");
            ui.horizontal(|ui| {
                if ui.button("Use ligand's stored SMILES").clicked() {
                    editor.smiles = ligand.get_smiles().unwrap_or("").into();
                }
                ui.label("Job name");
                ui.text_edit_singleline(&mut editor.options.job_name);
            });
            ui.checkbox(&mut editor.options.use_template, "Use displayed protein chain as a structural template");
            ui.small("Only observed amino-acid residues are included. Missing residues are omitted; other chains and cofactors require the advanced structure-prediction form.");
            egui::CollapsingHeader::new("Review protein sequence and residue mapping").show(ui, |ui| {
                let residues = binding::chain_residues(protein, editor.chain);
                ui.label(format!("{} observed residues → Boltz chain A; ligand → Boltz chain B", residues.len()));
                let sequence: String = residues.iter().filter_map(|&r| {
                    match protein.residues[r].res_type {
                        bio_files::ResidueType::AminoAcid(aa) => Some(aa.to_str(na_seq::AaIdent::OneLetter)),
                        _ => None,
                    }
                }).collect();
                ui.add(TextEdit::multiline(&mut sequence.as_str()).desired_rows(3).desired_width(f32::INFINITY));
                ScrollArea::vertical().id_salt("binding_residue_mapping").max_height(120.).show(ui, |ui| {
                    for (position, &r) in residues.iter().enumerate() {
                        ui.label(format!("{} → A:{}", binding::residue_label(protein, r), position + 1));
                    }
                });
            });
            ui.checkbox(&mut editor.options.use_msa_server, "Generate an MSA using the public ColabFold server (sends this protein sequence)");
            if !editor.options.use_msa_server {
                ui.small("Single-sequence mode: no sequence is sent to an MSA server; prediction accuracy may be lower.");
            }
            ui.checkbox(&mut editor.options.use_potentials, "Use Boltz inference-time potentials");
            ui.horizontal_wrapped(|ui| {
                ui.label("Structure samples");
                ui.add(DragValue::new(&mut editor.options.structure_samples).range(1..=50));
                ui.label("Affinity samples");
                ui.add(DragValue::new(&mut editor.options.affinity_samples).range(1..=50));
                ui.label("Seed");
                ui.add(DragValue::new(&mut editor.options.seed));
            });
            egui::CollapsingHeader::new(format!("Optional binding-site guidance · {} residues", editor.pocket.len())).show(ui, |ui| {
                ui.label("Leave empty to predict without a specified site, or choose residues to guide the ligand toward a site.");
                ui.horizontal_wrapped(|ui| {
                    if ui.add_enabled(report_current, egui::Button::new("Use current-pose contacts")).clicked() {
                        editor.pocket = editor.report.as_ref().unwrap().contacts.iter().map(|c| c.residue).collect();
                        editor.pocket_source = Some(current);
                    }
                    if ui.button("Clear site").clicked() {
                        editor.pocket.clear();
                        editor.pocket_source = None;
                    }
                    ui.label("Pocket distance (Å)");
                    ui.add(DragValue::new(&mut editor.options.pocket_distance).range(4.0..=20.0));
                });
                ui.add(TextEdit::singleline(&mut editor.residue_filter).hint_text("Filter residue name or number"));
                let filter = editor.residue_filter.to_ascii_lowercase();
                ScrollArea::vertical().id_salt("binding_residues").max_height(150.).show(ui, |ui| {
                    for r in binding::chain_residues(protein, editor.chain) {
                        let label = binding::residue_label(protein, r);
                        if !label.to_ascii_lowercase().contains(&filter) {
                            continue;
                        }
                        let mut selected = editor.pocket.contains(&r);
                        if ui.checkbox(&mut selected, label).changed() {
                            if selected { editor.pocket.push(r); } else { editor.pocket.retain(|&i| i != r); }
                            editor.pocket_source = Some(current);
                        }
                    }
                });
            });
        });
        let stale_site = !editor.pocket.is_empty() && editor.pocket_source != Some(current);
        if stale_site {
            ui.colored_label(Color32::ORANGE, "Site guidance is stale. Clear it or choose residues again after changing the inputs.");
        }
        ui.hyperlink_to(
            "Boltz-2 input and output definitions",
            "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md",
        );
        load = editor
            .runner
            .binding_controls(&state.volatile.ongoing_computations, ui, || {
                if stale_site {
                    return Err(
                        "Refresh or clear the stale binding-site guidance before predicting."
                            .into(),
                    );
                }
                let mut request = PredictionRequest::new(
                    protein,
                    &ligand.common,
                    editor.chain,
                    &editor.smiles,
                    &editor.pocket,
                    &editor.options,
                )?;
                if report_current {
                    request.provenance["pose_report"] =
                        serde_json::to_value(&editor.report).map_err(|e| e.to_string())?;
                }
                Ok(request)
            });

        if let Some(report) = &editor.report {
            if let Some(residues) = select_residues {
                let previous_selection = state.ui.selection.clone();
                select_atoms(state, protein_i, ligand_i, &residues, false, report);
                redraw.set_from_sel(&previous_selection);
                redraw.peptide = true;
            }
            if select_clashes {
                let previous_selection = state.ui.selection.clone();
                select_atoms(state, protein_i, ligand_i, &[], true, report);
                redraw.set_from_sel(&previous_selection);
                redraw.peptide = true;
                redraw.ligand = true;
            }
        }
    } else {
        ui.label("Open a protein and a small molecule to start a binding workflow.");
        // Saved results remain accessible even after input objects have been closed.
        load = editor
            .runner
            .binding_controls(&state.volatile.ongoing_computations, ui, || {
                Err("Choose an open protein and ligand before predicting.".into())
            });
    }
    if !editor.message.is_empty() {
        ui.colored_label(Color32::ORANGE, &editor.message);
    }
    state.ui.binding = editor;
    if let Some(path) = load
        && let Err(error) = state.open_file(&path, scene, updates)
    {
        handle_err(&mut state.ui, error.to_string());
    }
}
