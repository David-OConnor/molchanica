//! Paginated reaction cards shared by the ligand and protein sidebars.

use bio_apis::{
    chebi,
    rhea::{Reaction, ReactionSide},
};
use egui::{Button, Color32, Frame, RichText, ScrollArea, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    reactions::{Entry, Query, ReactionsState},
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
    ui.heading(&state.title);
    ui.horizontal_wrapped(|ui| {
        ui.label("Click a molecule to:");

        if let Some(download) = selector(
            ui,
            state.download_on_click,
            &[
                (
                    false,
                    "Open ChEBI molecule page in browser",
                    "Clicking a molecule opens its ChEBI page in your web browser.",
                ),
                (
                    true,
                    "Download and open in Molchanica",
                    "Clicking a molecule downloads it from ChEBI, and opens it in Molchanica.",
                ),
            ],
        ) {
            state.download_on_click = download;
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
                            reaction_card(reaction, state, &mut clicked_participant, ui);
                            ui.add_space(8.0);
                        }
                    });
            }
        }
    }

    if retry {
        state.cache.remove(&query);
        state.load(query);
        state.page = 0;
        ui.ctx().request_repaint();
    }

    if let Some(id) = clicked_participant {
        if state.download_on_click {
            state.download(id);

            ui.ctx().request_repaint();
        } else {
            chebi::open_overview(id);
        }
    }
}

fn reaction_card(
    reaction: &Reaction,
    state: &ReactionsState,
    clicked: &mut Option<u32>,
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
            ui.columns(3, |columns| {
                side(
                    left,
                    &reaction.reactants,
                    "Left side",
                    Color32::from_rgb(125, 195, 230),
                    state,
                    clicked,
                    &mut columns[0],
                );

                columns[1].vertical_centered(|ui| {
                    ui.add_space(18.0);
                    ui.label(RichText::new("=").size(28.0));
                    ui.weak("Direction unspecified");
                });

                side(
                    right,
                    &reaction.products,
                    "Right side",
                    Color32::from_rgb(150, 215, 170),
                    state,
                    clicked,
                    &mut columns[2],
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
    clicked: &mut Option<u32>,
    ui: &mut Ui,
) {
    ui.label(RichText::new(label).small().color(color));
    ui.horizontal_wrapped(|ui| {
        for (index, participant) in equation.split(" + ").enumerate() {
            if index > 0 {
                ui.label(RichText::new("+").color(color));
            }
            Frame::new()
                .fill(color.gamma_multiply(0.12))
                .corner_radius(4)
                .inner_margin(5)
                .show(ui, |ui| {
                    // The filtered ChEBI-only list cannot be zipped with names: generic
                    // RHEA-COMP participants would shift every subsequent link.
                    let id = participant_chebi_id(participants, equation, index);
                    if let Some(id) = id {
                        let loading = state.download_on_click && state.downloads.contains_key(&id);
                        let action = if state.download_on_click {
                            "Download and open in Molchanica"
                        } else {
                            "Open ChEBI molecule page in browser"
                        };

                        if ui
                            .add_enabled(
                                !loading,
                                egui::Link::new(RichText::new(participant).strong()),
                            )
                            .on_hover_text(format!("CHEBI:{id} — {action}"))
                            .clicked()
                        {
                            *clicked = Some(id);
                        }
                    } else {
                        ui.add(egui::Label::new(RichText::new(participant).strong()).wrap())
                            .on_hover_text(
                                "Rhea does not provide a ChEBI ID for this participant.",
                            );
                    }
                });
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
