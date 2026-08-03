//! Protein design, stability scanning, and antibody annotation for the active peptide.
//!
//! Three things that all start from a loaded structure and all take long enough to need a worker
//! thread, grouped into one window rather than three:
//!
//! - **Design sequences** runs the MPNN family ([`crate::external_tools::mpnn`]) to propose
//!   sequences that would fold into the backbone on screen.
//! - **Stability scan** runs the native ΔΔG scanner ([`crate::therapeutic::ddg`]) over every
//!   position and every substitution in one pass.
//! - **Antibody** annotates chains, and — when ANARCII or IgBLAST is installed — replaces the
//!   sequence-position approximations with a real numbering assignment and germline calls.
//!
//! They belong together because they are used together: a scan says which positions to hold
//! fixed, design proposes sequences for the rest, and the antibody annotation supplies the CDR
//! residue lists both of those are usually scoped to.
//!
//! Each tab owns its own worker channel rather than routing through `threads::handle_thread_rx`.
//! There is no shared state to reconcile — a result is displayed in the panel that asked for it
//! and nowhere else — so a local receiver keeps the whole interaction in one file.

use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use egui::{Button, Color32, ComboBox, DragValue, RichText, ScrollArea, Ui};
use na_seq::AaIdent;

use crate::{
    antibody::{self, AnnotationSource, AntibodyAnnotation, CdrNumberingScheme},
    external_tools::{
        Tool, anarcii,
        mpnn::{self, DesignRequest, DesignResult, MpnnModel, designable_chains},
    },
    molecules::peptide::MoleculePeptide,
    state::State,
    therapeutic::ddg::{self, DdgScan},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, popup::close_btn},
};

/// How many rows of each result table to show before the user has to export.
const RESULT_ROWS: usize = 20;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DesignTab {
    #[default]
    Sequences,
    Stability,
    Antibody,
}

impl DesignTab {
    const ALL: [Self; 3] = [Self::Sequences, Self::Stability, Self::Antibody];

    fn label(self) -> &'static str {
        match self {
            Self::Sequences => "Design sequences",
            Self::Stability => "Stability scan (ΔΔG)",
            Self::Antibody => "Antibody",
        }
    }
}

/// A job running on a worker thread, and its last result.
///
/// One of these per tab. Generic because the three differ only in what they produce: the start,
/// poll, cancel-by-drop, and elapsed-time handling are identical, and writing them three times is
/// how they drift.
struct Job<T> {
    receiver: Option<mpsc::Receiver<Result<T, String>>>,
    started_at: Option<Instant>,
    result: Option<T>,
    error: Option<String>,
}

impl<T> Default for Job<T> {
    fn default() -> Self {
        Self {
            receiver: None,
            started_at: None,
            result: None,
            error: None,
        }
    }
}

impl<T: Send + 'static> Job<T> {
    fn is_running(&self) -> bool {
        self.receiver.is_some()
    }

    fn start(
        &mut self,
        context: &egui::Context,
        work: impl FnOnce() -> Result<T, String> + Send + 'static,
    ) {
        if self.is_running() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.started_at = Some(Instant::now());
        self.error = None;

        let context = context.clone();
        thread::spawn(move || {
            let _ = tx.send(work());
            context.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(receiver) = &self.receiver else { return };
        match receiver.try_recv() {
            Ok(Ok(value)) => {
                self.result = Some(value);
                self.receiver = None;
                self.started_at = None;
            }
            Ok(Err(message)) => {
                self.error = Some(message);
                self.receiver = None;
                self.started_at = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            // The worker panicked without sending. Report it rather than showing "running"
            // forever, which is indistinguishable from a very slow model.
            Err(mpsc::TryRecvError::Disconnected) => {
                self.error = Some("The worker stopped without returning a result.".to_owned());
                self.receiver = None;
                self.started_at = None;
            }
        }
    }

    fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|start| start.elapsed())
    }
}

/// Panel state.
pub struct ProteinDesignUi {
    tab: DesignTab,

    // Design
    model: MpnnModel,
    num_sequences: usize,
    temperature: f32,
    seed: u64,
    chains_to_design: String,
    fixed_residues: String,
    design: Job<DesignResult>,

    // Stability
    stability: Job<DdgScan>,

    // Antibody
    scheme: anarcii::NumberingScheme,
    run_igblast: bool,
    antibody: Job<AntibodyAnnotation>,
}

impl Default for ProteinDesignUi {
    fn default() -> Self {
        Self {
            tab: DesignTab::default(),
            model: MpnnModel::default(),
            num_sequences: 8,
            temperature: 0.1,
            seed: 37,
            chains_to_design: String::new(),
            fixed_residues: String::new(),
            design: Job::default(),
            stability: Job::default(),
            scheme: anarcii::NumberingScheme::default(),
            run_igblast: false,
            antibody: Job::default(),
        }
    }
}

impl ProteinDesignUi {
    /// Comma- or space-separated chain identifiers, cleaned up.
    fn parsed_chains(&self) -> Vec<String> {
        self.chains_to_design
            .split([',', ' '])
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_owned)
            .collect()
    }

    fn parsed_fixed_residues(&self) -> Vec<String> {
        self.fixed_residues
            .split([',', ' '])
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_uppercase)
            .collect()
    }

    fn design_request(&self) -> DesignRequest {
        DesignRequest {
            model: self.model,
            chains_to_design: self.parsed_chains(),
            fixed_residues: self.parsed_fixed_residues(),
            num_sequences: self.num_sequences,
            temperature: self.temperature,
            seed: self.seed,
        }
    }
}

pub fn protein_design_window(state: &mut State, ui: &mut Ui) {
    let context = ui.ctx().clone();

    let Some(peptide) = state
        .peptide_for_tools_i()
        .and_then(|index| state.peptides.get(index))
        .cloned()
    else {
        ui.label(
            RichText::new("Open a protein to use these tools.").color(COLOR_INACTIVE),
        );
        ui.add_space(ROW_SPACING);
        close_btn(ui, &mut state.ui.popup.protein_design);
        return;
    };

    let design_ui = &mut state.ui.protein_design;
    design_ui.design.poll();
    design_ui.stability.poll();
    design_ui.antibody.poll();

    ui.horizontal(|ui| {
        for tab in DesignTab::ALL {
            let selected = design_ui.tab == tab;
            let text = RichText::new(tab.label())
                .color(if selected { COLOR_HIGHLIGHT } else { COLOR_INACTIVE });
            if ui.selectable_label(selected, text).clicked() {
                design_ui.tab = tab;
            }
        }
    });
    ui.label(
        RichText::new(format!("Active protein: {}", peptide.common.ident))
            .color(COLOR_INACTIVE)
            .small(),
    );
    ui.separator();

    match design_ui.tab {
        DesignTab::Sequences => sequences_tab(design_ui, &peptide, &context, ui),
        DesignTab::Stability => stability_tab(design_ui, &peptide, &context, ui),
        DesignTab::Antibody => antibody_tab(design_ui, &peptide, &context, ui),
    }

    ui.add_space(ROW_SPACING);
    close_btn(ui, &mut state.ui.popup.protein_design);
}

// ---------------------------------------------------------------------------------------------
// Design sequences
// ---------------------------------------------------------------------------------------------

fn sequences_tab(
    design_ui: &mut ProteinDesignUi,
    peptide: &MoleculePeptide,
    context: &egui::Context,
    ui: &mut Ui,
) {
    let running = design_ui.design.is_running();

    ui.add_enabled_ui(!running, |ui| {
        ui.horizontal(|ui| {
            ui.label("Model:");
            ComboBox::from_id_salt("mpnn_model")
                .selected_text(design_ui.model.label())
                .show_ui(ui, |ui| {
                    for model in MpnnModel::ALL {
                        ui.selectable_value(&mut design_ui.model, model, model.label())
                            .on_hover_text(model.help());
                    }
                });
        });
        ui.label(RichText::new(design_ui.model.help()).color(COLOR_INACTIVE).small());
        ui.add_space(ROW_SPACING);

        ui.horizontal(|ui| {
            ui.label("Sequences:");
            ui.add(DragValue::new(&mut design_ui.num_sequences).range(1..=512));
            ui.add_space(20.);
            ui.label("Temperature:")
                .on_hover_text("Low values give conservative, near-consensus sequences; high values give diversity.");
            ui.add(DragValue::new(&mut design_ui.temperature).speed(0.01).range(0.0001..=2.0));
            ui.add_space(20.);
            ui.label("Seed:");
            ui.add(DragValue::new(&mut design_ui.seed));
        });

        ui.horizontal(|ui| {
            ui.label("Chains to design:");
            ui.text_edit_singleline(&mut design_ui.chains_to_design)
                .on_hover_text(format!(
                    "Blank designs every chain. This structure has: {}",
                    designable_chains(peptide).join(", ")
                ));
        });
        ui.horizontal(|ui| {
            ui.label("Keep fixed:");
            ui.text_edit_singleline(&mut design_ui.fixed_residues).on_hover_text(
                "Residues to hold at their current identity, as chain letter plus residue number \
                 (e.g. H97 L1). Use this to redesign CDRs while keeping a framework, or to \
                 preserve a catalytic site.",
            );
        });
    });

    ui.add_space(ROW_SPACING);
    let tool = design_ui.model.tool();
    ui.horizontal(|ui| {
        if running {
            ui.label(RichText::new("Designing…").color(COLOR_ACTION));
            if let Some(elapsed) = design_ui.design.elapsed() {
                ui.label(format_elapsed(elapsed));
            }
        } else if ui
            .add(Button::new(RichText::new("Design").color(COLOR_ACTION)))
            .clicked()
        {
            let peptide = peptide.clone();
            let request = design_ui.design_request();
            design_ui.design.start(context, move || {
                mpnn::design(&peptide, &request).map_err(|error| error.to_string())
            });
        }
        ui.label(
            RichText::new(format!("via {}", tool.spec().name))
                .color(COLOR_INACTIVE)
                .small(),
        );
    });

    show_error(&design_ui.design.error, tool, ui);

    if let Some(result) = &design_ui.design.result {
        ui.add_space(ROW_SPACING);
        if let Some(input) = &result.input_sequence {
            ui.label(
                RichText::new(format!("Input: {}", truncate(input, 90)))
                    .color(COLOR_INACTIVE)
                    .small()
                    .monospace(),
            );
        }
        ui.label(format!(
            "{} sequences, best first (lower score is better).",
            result.designs.len()
        ));
        ScrollArea::vertical()
            .max_height(300.)
            .id_salt("mpnn_designs")
            .show(ui, |ui| {
                for design in result.designs.iter().take(RESULT_ROWS) {
                    ui.horizontal(|ui| {
                        let score = design
                            .score
                            .map(|value| format!("{value:.3}"))
                            .unwrap_or_else(|| "-".to_owned());
                        let recovery = design
                            .sequence_recovery
                            .map(|value| format!("{:.0}%", value * 100.))
                            .unwrap_or_else(|| "-".to_owned());
                        ui.label(
                            RichText::new(format!("{score:>7}  {recovery:>4}"))
                                .color(COLOR_HIGHLIGHT)
                                .monospace(),
                        );
                        ui.label(
                            RichText::new(truncate(&design.sequence, 80))
                                .monospace()
                                .small(),
                        );
                    });
                }
            });
        if result.designs.len() > RESULT_ROWS {
            ui.label(
                RichText::new(format!(
                    "Showing {RESULT_ROWS} of {}.",
                    result.designs.len()
                ))
                .color(COLOR_INACTIVE)
                .small(),
            );
        }
        if ui.button("Copy best sequence").clicked()
            && let Some(best) = result.designs.first()
        {
            ui.ctx().copy_text(best.sequence.clone());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stability scan
// ---------------------------------------------------------------------------------------------

fn stability_tab(
    design_ui: &mut ProteinDesignUi,
    peptide: &MoleculePeptide,
    context: &egui::Context,
    ui: &mut Ui,
) {
    ui.label(
        "Scores every substitution at every position in a single pass. Positive ΔΔG is \
         destabilizing; negative is stabilizing.",
    );
    ui.label(
        RichText::new(
            "Log-likelihood units, not kcal/mol: it ranks substitutions, it does not claim a \
             calorimetric value. Runs natively — no Python at run time.",
        )
        .color(COLOR_INACTIVE)
        .small(),
    );
    ui.add_space(ROW_SPACING);

    let running = design_ui.stability.is_running();
    ui.horizontal(|ui| {
        if running {
            ui.label(RichText::new("Scanning…").color(COLOR_ACTION));
            if let Some(elapsed) = design_ui.stability.elapsed() {
                ui.label(format_elapsed(elapsed));
            }
        } else if ui
            .add(Button::new(RichText::new("Run scan").color(COLOR_ACTION)))
            .clicked()
        {
            let peptide = peptide.clone();
            design_ui.stability.start(context, move || {
                // Weights are loaded per run rather than cached: a scan takes seconds, the load
                // is milliseconds, and caching would mean a re-converted checkpoint is ignored
                // until restart.
                let weights = ddg::load_weights().map_err(|error| error.to_string())?;
                ddg::scan(&peptide, &weights).map_err(|error| error.to_string())
            });
        }
    });

    show_error(&design_ui.stability.error, Tool::ProteinMpnn, ui);

    if let Some(scan) = &design_ui.stability.result {
        ui.add_space(ROW_SPACING);
        ui.label(format!("{} positions scanned.", scan.positions.len()));

        ui.add_space(6.);
        ui.label(RichText::new("Most stabilizing substitutions").strong());
        ScrollArea::vertical()
            .max_height(200.)
            .id_salt("ddg_best")
            .show(ui, |ui| {
                for (position, mutant, value) in scan.best_mutations(RESULT_ROWS) {
                    ui.label(
                        RichText::new(format!(
                            "{:>4}  {:<8}  {value:+.3}",
                            position.chain_id,
                            position.mutation_label(mutant)
                        ))
                        .monospace()
                        .small(),
                    );
                }
            });

        ui.add_space(6.);
        ui.label(RichText::new("Most constrained positions").strong())
            .on_hover_text(
                "Positions the structure tolerates least. These are what to hold fixed when \
                 designing.",
            );
        ScrollArea::vertical()
            .max_height(160.)
            .id_salt("ddg_constrained")
            .show(ui, |ui| {
                for position in scan.most_constrained(RESULT_ROWS) {
                    ui.label(
                        RichText::new(format!(
                            "{:>4}  {}{:<6}  mean {:+.3}",
                            position.chain_id,
                            position.wild_type.to_str(AaIdent::OneLetter),
                            position.residue_number,
                            position.constraint()
                        ))
                        .monospace()
                        .small(),
                    );
                }
            });

        ui.add_space(6.);
        if ui
            .button("Copy full table (TSV)")
            .on_hover_text("Every position against all twenty substitutions.")
            .clicked()
        {
            ui.ctx().copy_text(scan.to_tsv());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Antibody
// ---------------------------------------------------------------------------------------------

fn antibody_tab(
    design_ui: &mut ProteinDesignUi,
    peptide: &MoleculePeptide,
    context: &egui::Context,
    ui: &mut Ui,
) {
    let running = design_ui.antibody.is_running();

    ui.add_enabled_ui(!running, |ui| {
        ui.horizontal(|ui| {
            ui.label("Numbering:");
            ComboBox::from_id_salt("anarcii_scheme")
                .selected_text(design_ui.scheme.label())
                .show_ui(ui, |ui| {
                    for scheme in anarcii::NumberingScheme::ALL {
                        ui.selectable_value(&mut design_ui.scheme, scheme, scheme.label());
                    }
                });
            ui.checkbox(&mut design_ui.run_igblast, "Germline assignment")
                .on_hover_text(
                    "Also run IgBLAST to identify which germline V and J genes each chain came \
                     from. Slower, and needs the germline databases installed.",
                );
        });
    });
    ui.label(
        RichText::new(
            "Without ANARCII installed you still get CDRs, but from sequence-position \
             approximations — the panel says which you are looking at.",
        )
        .color(COLOR_INACTIVE)
        .small(),
    );

    ui.add_space(ROW_SPACING);
    ui.horizontal(|ui| {
        if running {
            ui.label(RichText::new("Annotating…").color(COLOR_ACTION));
            if let Some(elapsed) = design_ui.antibody.elapsed() {
                ui.label(format_elapsed(elapsed));
            }
        } else if ui
            .add(Button::new(RichText::new("Annotate").color(COLOR_ACTION)))
            .clicked()
        {
            let peptide = peptide.clone();
            let scheme = design_ui.scheme;
            let run_igblast = design_ui.run_igblast;
            design_ui.antibody.start(context, move || {
                Ok(annotate(&peptide, scheme, run_igblast))
            });
        }
    });

    if let Some(annotation) = &design_ui.antibody.result {
        ui.add_space(ROW_SPACING);
        for note in &annotation.notes {
            ui.label(RichText::new(note).color(COLOR_INACTIVE).small());
        }
        ScrollArea::vertical()
            .max_height(360.)
            .id_salt("antibody_chains")
            .show(ui, |ui| {
                for chain in annotation.chains.iter().filter(|c| c.kind.is_antibody_like()) {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Chain {}", chain.chain_id)).strong());
                        ui.label(RichText::new(chain.kind.to_string()).color(COLOR_HIGHLIGHT));
                        // The source is shown, not buried: an approximation and a numbering
                        // assignment look identical once they are just boundaries.
                        ui.label(
                            RichText::new(format!("[{}]", chain.source))
                                .color(match chain.source {
                                    AnnotationSource::Approximate => Color32::ORANGE,
                                    _ => Color32::LIGHT_GREEN,
                                })
                                .small(),
                        );
                    });
                    ui.indent(&chain.chain_id, |ui| {
                        if !chain.germline_v.is_empty() {
                            ui.label(
                                RichText::new(format!(
                                    "Germline: {} / {}",
                                    chain.germline_v.first().map(String::as_str).unwrap_or("-"),
                                    chain.germline_j.first().map(String::as_str).unwrap_or("-"),
                                ))
                                .small(),
                            );
                        }
                        for cdr in &chain.cdrs {
                            ui.label(
                                RichText::new(format!(
                                    "{:<4} {:>4}-{:<4} {}",
                                    cdr.label.to_string(),
                                    cdr.start_position,
                                    cdr.end_position,
                                    cdr.sequence
                                ))
                                .monospace()
                                .small(),
                            );
                        }
                    });
                    ui.add_space(4.);
                }

                if !annotation.developability_issues.is_empty() {
                    ui.separator();
                    ui.label(RichText::new("Developability").strong());
                    for issue in &annotation.developability_issues {
                        ui.label(RichText::new(format!("{issue:?}")).small());
                    }
                }
            });

        ui.add_space(6.);
        if ui
            .button("Copy paratope selection (PyMOL)")
            .on_hover_text("The CDR residues, as a PyMOL selection expression.")
            .clicked()
        {
            ui.ctx().copy_text(annotation.paratope_pymol_selection());
        }
    }
}

/// Annotate, then refine with whichever tools are installed.
///
/// Refinement failures are folded into the annotation's notes rather than failing the whole
/// operation: an approximate annotation is still worth showing, and the note says why it was not
/// upgraded — which is usually "the tool is not installed", the one thing the user can act on.
fn annotate(
    peptide: &MoleculePeptide,
    scheme: anarcii::NumberingScheme,
    run_igblast: bool,
) -> AntibodyAnnotation {
    let mut annotation = antibody::annotate_antibody(peptide, CdrNumberingScheme::Imgt);

    if let Err(error) = antibody::refine_with_anarcii(&mut annotation, scheme) {
        annotation
            .notes
            .push(format!("ANARCII numbering unavailable: {error}"));
    }

    if run_igblast {
        match antibody::germline_assignments(&annotation) {
            Ok(assignments) => antibody::apply_germline_assignments(&mut annotation, &assignments),
            Err(error) => annotation
                .notes
                .push(format!("IgBLAST germline assignment unavailable: {error}")),
        }
    }

    annotation
}

// ---------------------------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------------------------

/// Show a failure, with the install command when the cause is a missing tool.
fn show_error(error: &Option<String>, tool: Tool, ui: &mut Ui) {
    let Some(message) = error else { return };
    ui.add_space(6.);
    ui.label(RichText::new(message).color(Color32::LIGHT_RED).small());
    // "was not found" is what the registry's own error says, so this catches exactly the case
    // where installing is the fix.
    if message.contains("not found") || message.contains("not installed") {
        ui.label(
            RichText::new(format!("Install with: {}", tool.spec().install_command()))
                .color(COLOR_HIGHLIGHT)
                .monospace()
                .small(),
        );
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    text.chars().take(max).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_and_residue_fields_accept_commas_or_spaces() {
        let mut ui = ProteinDesignUi::default();
        ui.chains_to_design = " A, B  C ".to_owned();
        assert_eq!(ui.parsed_chains(), ["A", "B", "C"]);

        ui.fixed_residues = "h97, l1".to_owned();
        // Upper-cased, since that is the form both repositories parse.
        assert_eq!(ui.parsed_fixed_residues(), ["H97", "L1"]);
    }

    #[test]
    fn empty_fields_mean_no_restriction_rather_than_an_empty_selection() {
        let ui = ProteinDesignUi::default();
        assert!(ui.parsed_chains().is_empty());
        assert!(ui.parsed_fixed_residues().is_empty());
        // An empty chain list is what the adapter reads as "design everything".
        assert!(ui.design_request().chains_to_design.is_empty());
    }

    #[test]
    fn the_request_carries_the_panel_settings() {
        let mut ui = ProteinDesignUi::default();
        ui.model = MpnnModel::AbMpnn;
        ui.num_sequences = 3;
        ui.temperature = 0.25;
        ui.seed = 99;

        let request = ui.design_request();
        assert_eq!(request.model, MpnnModel::AbMpnn);
        assert_eq!(request.num_sequences, 3);
        assert_eq!(request.seed, 99);
        assert!(request.validate().is_ok());
    }

    #[test]
    fn a_finished_job_stops_reporting_as_running() {
        let mut job: Job<u32> = Job::default();
        assert!(!job.is_running());

        let (tx, rx) = mpsc::channel();
        job.receiver = Some(rx);
        job.started_at = Some(Instant::now());
        assert!(job.is_running());

        tx.send(Ok(7)).unwrap();
        job.poll();
        assert!(!job.is_running());
        assert_eq!(job.result, Some(7));
        assert!(job.error.is_none());
        assert!(job.elapsed().is_none());
    }

    #[test]
    fn a_worker_that_dies_reports_an_error_rather_than_running_forever() {
        let mut job: Job<u32> = Job::default();
        let (tx, rx) = mpsc::channel::<Result<u32, String>>();
        job.receiver = Some(rx);
        job.started_at = Some(Instant::now());

        // The sender going away without a value is what a panicked worker looks like.
        drop(tx);
        job.poll();

        assert!(!job.is_running());
        assert!(job.error.is_some());
        assert!(job.result.is_none());
    }

    #[test]
    fn an_error_result_is_surfaced_and_clears_on_the_next_start() {
        let mut job: Job<u32> = Job::default();
        let (tx, rx) = mpsc::channel();
        job.receiver = Some(rx);
        tx.send(Err("boom".to_owned())).unwrap();
        job.poll();
        assert_eq!(job.error.as_deref(), Some("boom"));
        assert!(!job.is_running());
    }

    #[test]
    fn truncation_is_by_characters_not_bytes() {
        assert_eq!(truncate("abc", 5), "abc");
        assert_eq!(truncate("abcdef", 3), "abc…");
        // Multi-byte characters must not be split, which slicing by byte index would do.
        assert_eq!(truncate("ΔΔΔΔ", 2), "ΔΔ…");
    }

    #[test]
    fn elapsed_is_rendered_as_minutes_and_seconds() {
        assert_eq!(format_elapsed(Duration::from_secs(9)), "00:09");
        assert_eq!(format_elapsed(Duration::from_secs(75)), "01:15");
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "60:00");
    }
}
