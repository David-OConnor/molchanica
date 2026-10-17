//! The protein comparison workflow: choose chains, fit, inspect deviations, and undo.
use crate::{
    cam,
    protein_alignment::{self, Comparison, Correspondence, Source},
    selection::expression,
    state::State,
    util::RedrawFlags,
};
use egui::{Color32, ComboBox, DragValue, ScrollArea, TextEdit, Ui};
use egui_plot::{Line, Plot, PlotPoints};
use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::{MolGenericRef, MolType};
use std::{
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::Duration,
};
#[derive(Default)]
pub struct SourceChoice {
    key: String,
    chain: Option<usize>,
    selection: String,
}

pub struct ProteinComparison {
    reference: SourceChoice,
    mobile: SourceChoice,
    method: Correspondence,
    reject_outliers: bool,
    cutoff: f64,
    comparison_layers: bool,
    highlight_threshold: f64,
    pending: Option<Receiver<Result<Comparison, String>>>,
    apply_when_ready: bool,
    pub result: Option<Comparison>,
    message: String,
    export_dialog: egui_file_dialog::FileDialog,
    export_text: Option<String>,
}

impl Default for ProteinComparison {
    fn default() -> Self {
        Self {
            reference: Default::default(),
            mobile: Default::default(),
            method: Default::default(),
            reject_outliers: false,
            cutoff: 2.,
            comparison_layers: true,
            highlight_threshold: 2.,
            pending: None,
            apply_when_ready: false,
            result: None,
            message: String::new(),
            export_dialog: egui_file_dialog::FileDialog::new(),
            export_text: None,
        }
    }
}

pub fn start_expressions(state: &mut State, mobile: &str, reference: &str) -> Result<(), String> {
    if state.ui.protein_comparison.pending.is_some() {
        return Err("A protein comparison is already running".into());
    }
    let r = protein_alignment::source_expression(state, reference)?;
    let m = protein_alignment::source_expression(state, mobile)?;
    if r.key == m.key {
        return Err("Choose different reference and mobile objects".into());
    }
    let ri = protein_alignment::resolve(state, &r.key)?;
    let mi = protein_alignment::resolve(state, &m.key)?;
    state.ui.protein_comparison.reference = SourceChoice {
        key: r.key.clone(),
        chain: state.peptides[ri].common.atoms[r.ca[0].atom].chain,
        selection: reference.into(),
    };
    state.ui.protein_comparison.mobile = SourceChoice {
        key: m.key.clone(),
        chain: state.peptides[mi].common.atoms[m.ca[0].atom].chain,
        selection: mobile.into(),
    };
    state.ui.protein_comparison.method = Correspondence::Sequence;
    start(&mut state.ui.protein_comparison, r, m, true);
    state.ui.popup.protein_comparison = true;
    Ok(())
}

fn start(editor: &mut ProteinComparison, reference: Source, mobile: Source, apply: bool) {
    let (tx, rx) = mpsc::channel();
    let method = editor.method;
    let cutoff = editor.reject_outliers.then_some(editor.cutoff);
    thread::spawn(move || {
        let _ = tx.send(protein_alignment::compare(
            reference, mobile, method, cutoff,
        ));
    });
    editor.pending = Some(rx);
    editor.apply_when_ready = apply;
    editor.message = "Computing Cα correspondence and rigid fit…".into();
}

/// Poll even when the popup is closed. Never apply a completed fit to coordinates that changed
/// while its worker was running.
pub fn poll(state: &mut State, ui: &Ui, redraw: &mut RedrawFlags) {
    let mut editor = std::mem::take(&mut state.ui.protein_comparison);
    editor.export_dialog.update(ui.ctx());
    if let Some(mut path) = editor.export_dialog.take_picked()
        && let Some(text) = editor.export_text.take()
    {
        if path.extension().is_none() {
            path.set_extension("cif");
        }
        editor.message = std::fs::write(&path, text)
            .map(|()| format!("Exported fitted coordinates to {}", path.display()))
            .unwrap_or_else(|e| e.to_string());
    }
    if let Some(rx) = &editor.pending {
        match rx.try_recv() {
            Ok(result) => {
                editor.pending = None;
                match result {
                    Ok(mut result) => {
                        let status = if editor.apply_when_ready {
                            result.apply(state, redraw, false)
                        } else {
                            result.validate(state).map(|_| ())
                        };
                        editor.message = match status {
                            Ok(()) => {
                                if editor.apply_when_ready && editor.comparison_layers {
                                    let _ = result.comparison_view(state, redraw);
                                }
                                if result.applied {
                                    "Mobile protein superposed. Reference unchanged.".into()
                                } else {
                                    "Comparison calculated. Coordinates unchanged; after-fit values describe the proposed fit.".into()
                                }
                            }
                            Err(error) => error,
                        };
                        editor.result = Some(result);
                    }
                    Err(error) => editor.message = error,
                }
                ui.request_repaint();
            }
            Err(TryRecvError::Empty) => ui.request_repaint_after(Duration::from_millis(100)),
            Err(TryRecvError::Disconnected) => {
                editor.pending = None;
                editor.message = "Protein comparison worker stopped without a result".into();
            }
        }
    }
    state.ui.protein_comparison = editor;
}

fn source_picker(state: &State, ui: &mut Ui, label: &str, choice: &mut SourceChoice) {
    ui.push_id(label, |ui| {
        ui.horizontal(|ui| {
            ui.label(label);
            let selected = protein_alignment::resolve(state, &choice.key)
                .ok()
                .map(|i| {
                    MolGenericRef::Peptide(&state.peptides[i])
                        .name()
                        .into_owned()
                })
                .unwrap_or_else(|| "Choose protein…".into());
            ComboBox::from_id_salt("protein")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (i, mol) in state.peptides.iter().enumerate() {
                        let mol_ref = MolGenericRef::Peptide(mol);
                        ui.push_id(i, |ui| {
                            if ui
                                .selectable_value(
                                    &mut choice.key,
                                    expression::object_key(&mol_ref),
                                    mol_ref.name().as_ref(),
                                )
                                .changed()
                            {
                                choice.chain =
                                    protein_alignment::chains(mol).first().and_then(|(c, _)| *c);
                            }
                        });
                    }
                });
            if let Ok(i) = protein_alignment::resolve(state, &choice.key) {
                let chains = protein_alignment::chains(&state.peptides[i]);
                ComboBox::from_id_salt("chain")
                    .selected_text(
                        chains
                            .iter()
                            .find(|(i, _)| *i == choice.chain)
                            .map(|(_, name)| format!("Chain {name}"))
                            .unwrap_or_else(|| "Choose chain…".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, name) in chains {
                            ui.selectable_value(&mut choice.chain, i, name);
                        }
                    });
            }
        });
        ui.add(
            TextEdit::singleline(&mut choice.selection)
                .hint_text("Optional fitting region, e.g. resi 20-150 or %core")
                .desired_width(600.),
        );
    });
}

pub fn editor(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    let mut editor = std::mem::take(&mut state.ui.protein_comparison);
    if editor.reference.key.is_empty()
        && let Some(mol) = state.peptides.first()
    {
        editor.reference.key = expression::object_key(&MolGenericRef::Peptide(mol));
        editor.reference.chain = protein_alignment::chains(mol).first().and_then(|(c, _)| *c);
    }
    if editor.mobile.key.is_empty()
        && let Some(mol) = state.peptides.get(1)
    {
        editor.mobile.key = expression::object_key(&MolGenericRef::Peptide(mol));
        editor.mobile.chain = protein_alignment::chains(mol).first().and_then(|(c, _)| *c);
    }
    ui.label(concat!(
        "Fit corresponding Cα atoms from one chain in each protein. The whole mobile ",
        "object moves rigidly; the reference stays fixed."
    ));
    ui.add_enabled_ui(editor.pending.is_none(), |ui| {
        source_picker(state, ui, "Reference", &mut editor.reference);
        source_picker(state, ui, "Mobile", &mut editor.mobile);
        if ui.button("Swap reference / mobile").clicked() {
            std::mem::swap(&mut editor.reference, &mut editor.mobile);
        }
        ui.horizontal(|ui| {
            ComboBox::from_id_salt("correspondence")
                .selected_text(editor.method.label())
                .show_ui(ui, |ui| {
                    for method in [
                        Correspondence::Sequence,
                        Correspondence::ResidueNumber,
                        Correspondence::Order,
                    ] {
                        ui.selectable_value(&mut editor.method, method, method.label());
                    }
                });
            ui.checkbox(&mut editor.reject_outliers, "Reject outliers");
            ui.add_enabled(
                editor.reject_outliers,
                DragValue::new(&mut editor.cutoff)
                    .range(0.1..=50.)
                    .suffix(" Å")
                    .speed(0.1),
            );
        });
        ui.checkbox(
            &mut editor.comparison_layers,
            "Create blue reference / orange mobile comparison layers when superposing",
        );
        ui.collapsing("Matching and fitting details", |ui| {
            ui.label(concat!(
                "Sequence: global alignment of residues with coordinates; identity +3, ",
                "substitution −1, gap open −5, extension −1. For related proteins; not a ",
                "sequence-independent fold search."
            ));
            ui.label(concat!(
                "Residue numbers: intersect serial numbers within the chosen chains. Order: ",
                "equal-length selections, paired in residue order. Missing Cα atoms are skipped; ",
                "one alternate conformation per residue is used."
            ));
            ui.label(concat!(
                "Optional rejection removes pairs beyond the cutoff and refits, up to five ",
                "cycles. All-pair RMSD still includes rejected pairs. At least three noncollinear ",
                "pairs are required."
            ));
            ui.label(concat!(
                "The fit moves embedded hetero residues too. Separately loaded molecules and ",
                "density maps stay where they are. Export fitted mmCIF below to save the aligned ",
                "coordinates; the ordinary protein Save preserves its source CIF."
            ));
        });
        ui.horizontal(|ui| {
            let apply = ui.button("Superpose").clicked();
            let measure = ui.button("Compare without moving").clicked();
            if apply || measure {
                let request = protein_alignment::source(
                    state,
                    &editor.reference.key,
                    editor.reference.chain,
                    &editor.reference.selection,
                )
                .and_then(|r| {
                    protein_alignment::source(
                        state,
                        &editor.mobile.key,
                        editor.mobile.chain,
                        &editor.mobile.selection,
                    )
                    .map(|m| (r, m))
                });
                match request {
                    Ok((r, m)) => start(&mut editor, r, m, apply),
                    Err(error) => editor.message = error,
                }
            }
        });
    });
    ui.label(&editor.message);
    if editor.pending.is_some() {
        ui.spinner();
        ui.request_repaint_after(Duration::from_millis(100));
    }
    if let Some(result) = &mut editor.result {
        ui.separator();
        ui.label(format!(
            "{} / {} → {} / {} ({})",
            result.mobile.label,
            result.mobile.chain,
            result.reference.label,
            result.reference.chain,
            result.method.label()
        ));
        let retained = result.pairs.iter().filter(|p| p.retained).count();
        let identical = result
            .pairs
            .iter()
            .filter(|p| result.reference.ca[p.reference].aa == result.mobile.ca[p.mobile].aa)
            .count();
        ui.label(format!(
            "{} matched Cα pairs; {} retained, {} rejected; {:.1}% sequence identity",
            result.pairs.len(),
            retained,
            result.pairs.len() - retained,
            100. * identical as f64 / result.pairs.len() as f64
        ));
        ui.label(format!(
            "Coverage: reference {}/{}; mobile {}/{} selected Cα residues",
            result.pairs.len(),
            result.reference.ca.len(),
            result.pairs.len(),
            result.mobile.ca.len()
        ));
        ui.label(format!(
            "Before: {:.3} Å   After fit (all pairs): {:.3} Å   After fit (retained): {:.3} Å",
            result.before_rmsd, result.all_rmsd, result.fit_rmsd
        ));
        let valid = result.validate(state);
        if let Err(error) = &valid {
            ui.colored_label(Color32::YELLOW, format!("Stored result: {error}"));
        } else if !result.applied {
            ui.label("The fitted transform is not currently applied.");
        }
        ui.horizontal(|ui| {
            if ui.button("Copy residue CSV").clicked() {
                ui.ctx().copy_text(result.csv());
            }
            if ui.button("Copy transform").clicked() {
                let r = result.transform.rotation;
                let t = result.transform.translation;
                ui.ctx().copy_text(format!(
                    "{},{},{},{}\n{},{},{},{}\n{},{},{},{}\n0,0,0,1\n",
                    r[0][0],
                    r[0][1],
                    r[0][2],
                    t.x,
                    r[1][0],
                    r[1][1],
                    r[1][2],
                    t.y,
                    r[2][0],
                    r[2][1],
                    r[2][2],
                    t.z
                ));
            }
        });
        let deviations = PlotPoints::from_iter(
            result
                .pairs
                .iter()
                .enumerate()
                .map(|(i, p)| [i as f64 + 1., p.distance]),
        );
        Plot::new("comparison_deviations")
            .height(130.)
            .include_y(0.)
            .x_axis_label("Matched residue pair")
            .y_axis_label("Å")
            .show(ui, |plot| {
                plot.line(Line::new("Cα deviation after fit (Å)", deviations));
            });
        ui.add_enabled_ui(valid.is_ok() && editor.pending.is_none(),|ui| {
            ui.horizontal(|ui| {
                if ui.button("Export fitted mmCIF").on_hover_text(concat!(
"Coordinates only: all mobile atoms in the fitted pose, without source ",
"experimental metadata or density maps."
)).clicked() {
                    match result.aligned_cif(state) {
                        Ok(text) => {
                            editor.export_text=Some(text);
                            editor.export_dialog.config_mut().default_file_name="aligned_protein.cif".into();
                            editor.export_dialog.save_file();
                        }
                        Err(error) => editor.message=error,
                    }
                }
                let undo=result.applied;
                if ui.button(if undo {
"Undo superposition"}
 else {
"Apply fitted transform"}
).clicked() {
                    editor.message=result.apply(state,redraw,undo).and_then(|()| {
                        if !undo && editor.comparison_layers {
 result.comparison_view(state,redraw)?;
 }
                        Ok(if undo {
"Restored the mobile coordinates".into()}
 else {
"Applied the fitted transform".into()}
)
                    }
).unwrap_or_else(|e|e);
                }
                if ui.button("Comparison layers").clicked() {
                    editor.message=result.comparison_view(state,redraw).map(|()|"Comparison layers are editable in Selections / layers".into()).unwrap_or_else(|e|e);
                }
                if ui.button("Frame reference").clicked() {
                    if let Ok((r,_))=result.validate(state) {
                        state.volatile.active_mol=Some((MolType::Peptide,r));
                        cam::reset_camera(state,scene,updates,cam::VIEW_DIR_FRONT);
                    }
                }
            }
);
            let mut select=None;
            ui.horizontal(|ui| {
                if ui.button("Select matched residues").clicked() {
 select=Some((0..result.pairs.len()).collect());
 }
                if ui.button("Select rejected").clicked() {
 select=Some(result.pairs.iter().enumerate().filter(|(_,p)|!p.retained).map(|(i,_)|i).collect());
 }
                ui.add(DragValue::new(&mut editor.highlight_threshold).range(0.1..=100.).speed(0.1).suffix(" Å"));
                if ui.button("Select above threshold").clicked() {
 select=Some(result.pairs.iter().enumerate().filter(|(_,p)|p.distance>editor.highlight_threshold).map(|(i,_)|i).collect());
 }
            }
);
            ui.label("Reference residue → Mobile residue | deviation after fit | click to select and focus both residues");
            ScrollArea::vertical().max_height(260.).show_rows(ui,22.,result.pairs.len(),|ui,range| {
                for i in range {
                    let pair=&result.pairs[i];
 let (r,m)=(&result.reference.ca[pair.reference],&result.mobile.ca[pair.mobile]);
                    if ui.button(format!("{} {} → {} {}    {:.3} Å    {}",r.number,r.aa,m.number,m.aa,pair.distance,if pair.retained {
"fit"}
 else {
"rejected"}
)).clicked() {
 select=Some(vec![i]);
 }
                }
            }
);
            if let Some(indices)=select {
                match result.selection(state,&indices) {
                    Ok(selection) => {
                        state.ui.selection=selection;
                        if indices.len()==1 {
 cam::move_cam_to_sel(state,&mut scene.camera,updates);
 }
                        redraw.set_all();
                    }
                    Err(error) => editor.message=error,
                }
            }
        }
);
    }
    state.ui.protein_comparison = editor;
}
