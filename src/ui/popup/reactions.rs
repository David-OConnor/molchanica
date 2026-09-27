//! Paginated reaction cards shared by the ligand and protein sidebars.

use bio_apis::{
    chebi,
    rhea::{Reaction, ReactionSide},
};
use egui::{Align, Button, Color32, Frame, Layout, RichText, ScrollArea, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    reactions::{Entry, ParticipantAction, Query, ReactionsState},
    state::State,
    ui::{misc::selector, util::open_chebi_download},
    util::{RedrawFlags, handle_err, make_lig_3d},
};

const PER_PAGE: usize = 4;

/// Finish imports even when the user has closed the popup or changed the selected molecule.
pub(super) fn poll_downloads(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
) {
    for (id, result) in state.ui.reactions.take_downloads() {
        // Opening a ChEBI molecule appends it to the ligand list. Keep its index so the
        // conversion below targets the molecule from this download, regardless of selection.
        let downloaded_lig_i = state.ligands.len();
        let result = result.and_then(|downloaded| {
            open_chebi_download(state, scene, redraw, updates, id, downloaded)
        });

        if result.is_ok()
            && state
                .ligands
                .get(downloaded_lig_i)
                .is_some_and(|lig| lig.common.is_2d)
        {
            // We make 3D because most (or all?) of the ChEBI molecules we load from this
            // are 2d, with incorrect proportions and bond lengths.
            make_lig_3d(state, downloaded_lig_i, scene, updates);
        }
        match result {
            Ok(()) => {
                state.ui.reactions.download_message =
                    Some(format!("Opened CHEBI:{id} in Molchanica."));
            }
            Err(error) => {
                handle_err(&mut state.ui, error.clone());
                state.ui.reactions.download_error = Some(error);
            }
        }
    }
}

pub(super) fn reactions_window(state: &mut ReactionsState, ui: &mut Ui) {
    state.diagrams.poll(ui.ctx());
    ui.heading(RichText::new(&state.title).color(Color32::WHITE));
    let has_chebi_id = matches!(state.selected.as_ref(), Some(Query::Chebi(_)));
    if state.pubchem_cid.is_some() || has_chebi_id {
        ui.horizontal_wrapped(|ui| {
            if let Some(cid) = state.pubchem_cid {
                ui.colored_label(Color32::WHITE, format!("CID: {cid}"));
            }
            if let Some(Query::Chebi(id)) = &state.selected {
                ui.colored_label(Color32::WHITE, format!("CHEBI:{id}"));
            }
        });
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Click a molecule to:");

        if let Some(action) = selector(
            ui,
            state.participant_action,
            &[
                (
                    ParticipantAction::OpenChebiPage,
                    "Open ChEBI molecule page in browser",
                    "Clicking a molecule opens its ChEBI page in your web browser.",
                ),
                (
                    ParticipantAction::Download,
                    "Download and open in Molchanica",
                    "Clicking a molecule downloads it from ChEBI, and opens it in Molchanica.",
                ),
                (
                    ParticipantAction::OpenRheaPage,
                    "Open Rhea page for this molecule",
                    "Clicking a molecule shows its Rhea reactions in this window.",
                ),
            ],
        ) {
            state.participant_action = action;
        }
    });
    if !state.downloads.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.spinner();
            ui.label("Downloading from ChEBI:");
            for id in state.downloads.keys() {
                ui.label(format!("CHEBI:{id}"));
            }
        });
    }

    if let Some(message) = &state.download_message {
        ui.label(message);
    }

    if let Some(error) = &state.download_error {
        ui.colored_label(Color32::LIGHT_RED, error);
        ui.label("Click the molecule again to retry.");
    }

    let Some(query) = state.selected.clone() else {
        if let Some(message) = &state.message {
            ui.colored_label(Color32::LIGHT_RED, message);
        } else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Loading molecule identifiers to find its ChEBI ID…");
            });
        }
        return;
    };

    let Some(entry) = state.cache.get(&query) else {
        return;
    };
    let displayed_query = match entry {
        Entry::Ready(results) => &results.query,
        _ => &query,
    };

    ui.horizontal_wrapped(|ui| {
        if matches!(displayed_query, Query::Pdb(_)) {
            ui.label("Resolving the protein's UniProt mapping…");
        } else {
            ui.label("Search Rhea:");
        }
        for (label, url) in displayed_query.search_links() {
            ui.hyperlink_to(label, url);
        }
    });
    if matches!(query, Query::Chebi(_)) {
        ui.label(
            "Showing exact ChEBI matches. The general ChEBI search may include related compounds.",
        );
    }
    ui.separator();

    let mut retry = false;
    let mut clicked_participant = None;
    let mut diagram_requests = Vec::new();

    match entry {
        Entry::Loading(_) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Loading reactions from Rhea…");
            });
        }
        Entry::Failed(error) => {
            ui.colored_label(Color32::LIGHT_RED, error);
            retry = ui.button("Retry").clicked();
        }
        Entry::Ready(results) => {
            if !results.warnings.is_empty() {
                ui.colored_label(
                    Color32::YELLOW,
                    "Some UniProt lookups failed; results may be incomplete.",
                );
                for warning in &results.warnings {
                    ui.label(warning);
                }
                retry = ui.button("Retry lookups").clicked();
            }
            let count = results.reactions.len();
            if count == 0 {
                if results.warnings.is_empty() {
                    ui.label("No Rhea reactions found for this search.");
                }
            } else {
                let pages = count.div_ceil(PER_PAGE);
                state.page = state.page.min(pages - 1);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(state.page > 0, Button::new("Previous"))
                        .clicked()
                    {
                        state.page -= 1;
                    }
                    ui.label(format!("Page {} of {pages}", state.page + 1));
                    if ui
                        .add_enabled(state.page + 1 < pages, Button::new("Next"))
                        .clicked()
                    {
                        state.page += 1;
                    }
                    let first = state.page * PER_PAGE + 1;
                    let last = ((state.page + 1) * PER_PAGE).min(count);
                    ui.label(format!("{first}–{last} of {count} reactions"));
                });
                ui.add_space(6.0);

                ScrollArea::vertical()
                    .id_salt(("rhea_reactions", &query, state.page))
                    .max_height(ui.available_height())
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for reaction in results
                            .reactions
                            .iter()
                            .skip(state.page * PER_PAGE)
                            .take(PER_PAGE)
                        {
                            reaction_card(
                                reaction,
                                state,
                                &mut clicked_participant,
                                &mut diagram_requests,
                                ui,
                            );
                            ui.add_space(8.0);
                        }
                    });
            }
        }
    }

    state.diagrams.request(&diagram_requests, ui.ctx());

    if retry {
        state.cache.remove(&query);
        state.load(query);
        state.page = 0;
        ui.ctx().request_repaint();
    }

    if let Some((id, name)) = clicked_participant {
        match state.participant_action {
            ParticipantAction::OpenChebiPage => chebi::open_overview(id),
            ParticipantAction::Download => state.download(id),
            ParticipantAction::OpenRheaPage => state.show_chebi_participant(id, name),
        }
        ui.ctx().request_repaint();
    }
}

fn reaction_card(
    reaction: &Reaction,
    state: &ReactionsState,
    clicked: &mut Option<(u32, String)>,
    diagram_requests: &mut Vec<(u32, bool)>,
    ui: &mut Ui,
) {
    Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            ui.hyperlink_to(
                RichText::new(reaction.accession()).strong(),
                format!("https://www.rhea-db.org/rhea/{}", reaction.id),
            );
            if !reaction.ec_numbers.is_empty() {
                ui.weak(format!("EC {}", reaction.ec_numbers.join(", ")));
            }
            ui.weak(format!("{} annotated enzymes", reaction.enzyme_count));
        });
        ui.add(egui::Label::new(&reaction.equation).wrap());
        ui.add_space(6.0);

        // Split only at spaced delimiters: H(+) and names containing '+' remain intact, and
        // stoichiometric coefficients stay attached to their participants. Master reactions have
        // undefined direction, so '=' is more accurate than implying a one-way or reversible arrow.
        if let Some((left, right)) = reaction.equation.split_once(" = ") {
            let equals_width = 36.0;
            let spacing = ui.spacing().item_spacing.x;
            let available_for_sides = ui.available_width() - equals_width - 2.0 * spacing;
            let side_width = (available_for_sides / 2.0).max(0.0);

            ui.horizontal(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(side_width, 0.0),
                    Layout::top_down(Align::Min),
                    |ui| {
                        ui.set_max_width(side_width);
                        side(
                            left,
                            &reaction.reactants,
                            "Left side",
                            Color32::from_rgb(125, 195, 230),
                            state,
                            clicked,
                            diagram_requests,
                            ui,
                        );
                    },
                );

                ui.allocate_ui_with_layout(
                    egui::vec2(equals_width, 0.0),
                    Layout::top_down(Align::Center),
                    |ui| {
                        ui.add_space(18.0);
                        ui.label(RichText::new("=").size(28.0))
                            .on_hover_text("Direction unspecified");
                    },
                );

                ui.allocate_ui_with_layout(
                    egui::vec2(side_width, 0.0),
                    Layout::top_down(Align::Min),
                    |ui| {
                        ui.set_max_width(side_width);
                        side(
                            right,
                            &reaction.products,
                            "Right side",
                            Color32::from_rgb(150, 215, 170),
                            state,
                            clicked,
                            diagram_requests,
                            ui,
                        );
                    },
                );
            });
        }
        if let Some(go) = &reaction.go {
            ui.add_space(4.0);
            ui.weak(&go.label);
        }
    });
}

fn side(
    equation: &str,
    participants: &ReactionSide,
    label: &str,
    color: Color32,
    state: &ReactionsState,
    clicked: &mut Option<(u32, String)>,
    diagram_requests: &mut Vec<(u32, bool)>,
    ui: &mut Ui,
) {
    ui.label(RichText::new(label).small().color(color));
    let molecule_width = ui.available_width().min(crate::mol_diagrams::DIAGRAM_WIDTH);

    // Top-align the participant slots. `horizontal_wrapped` centers vertically, so each taller
    // slot (button + diagram) shifts the row's centerline, and the next slot lands lower.
    let row_layout = Layout::left_to_right(Align::Min).with_main_wrap(true);
    ui.with_layout(row_layout, |ui| {
        for (index, participant) in equation.split(" + ").enumerate() {
            if index > 0 {
                ui.label(RichText::new("+").color(color));
            }
            ui.allocate_ui_with_layout(
                egui::vec2(molecule_width, 0.0),
                Layout::top_down(Align::Min),
                |ui| {
                    ui.set_width(molecule_width);
                    // The filtered ChEBI-only list cannot be zipped with names: generic
                    // RHEA-COMP participants would shift every subsequent link.
                    let id = participant_chebi_id(participants, equation, index);
                    let queried = matches!(
                        (state.selected.as_ref(), id),
                        (Some(Query::Chebi(query_id)), Some(id)) if *query_id == id
                    );
                    if let Some(id) = id {
                        let loading = state.participant_action == ParticipantAction::Download
                            && state.downloads.contains_key(&id);
                        let action = match state.participant_action {
                            ParticipantAction::OpenChebiPage => {
                                "Open ChEBI molecule page in browser"
                            }
                            ParticipantAction::Download => "Download and open in Molchanica",
                            ParticipantAction::OpenRheaPage => "Open Rhea page for this molecule",
                        };
                        let fill = if queried {
                            Color32::from_rgb(90, 105, 35)
                        } else {
                            color.gamma_multiply(0.12)
                        };
                        let text_color = if queried { Color32::WHITE } else { color };
                        let button =
                            Button::new(RichText::new(participant).strong().color(text_color))
                                .fill(fill)
                                .frame(true)
                                .wrap_mode(egui::TextWrapMode::Wrap);

                        if ui
                            .add_enabled(!loading, button)
                            .on_hover_text(format!("CHEBI:{id} — {action}"))
                            .clicked()
                        {
                            let name = participants
                                .participant_names
                                .get(index)
                                .cloned()
                                .unwrap_or_else(|| participant.to_owned());
                            *clicked = Some((id, name));
                        }
                        let retry = state.diagrams.show(id, ui);
                        diagram_requests.push((id, retry));
                    } else {
                        Frame::new()
                            .fill(color.gamma_multiply(0.12))
                            .corner_radius(4)
                            .inner_margin(5)
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(RichText::new(participant).strong()).wrap(),
                                )
                                .on_hover_text(
                                    "Rhea does not provide a ChEBI ID for this participant.",
                                );
                            });
                        ui.weak("No 2D structure: no ChEBI ID supplied.");
                    }
                },
            );
        }
    });
}

fn participant_chebi_id(side: &ReactionSide, equation: &str, index: usize) -> Option<u32> {
    // If an unexpected equation cannot be aligned, leave it unlinked rather than guess an ID.
    if side.participant_identifiers.len() != equation.split(" + ").count() {
        return None;
    }
    side.participant_identifiers
        .get(index)?
        .strip_prefix("CHEBI:")?
        .parse()
        .ok()
}
