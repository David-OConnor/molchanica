//! Paginated reaction cards shared by the ligand and protein sidebars.

use std::cmp::Ordering;

use bio_apis::{
    chebi, pubchem,
    rhea::{Reaction, ReactionSide},
};
use egui::{
    Align, Align2, Button, CollapsingHeader, Color32, ComboBox, FontId, Frame, Grid, Layout,
    RichText, ScrollArea, Sense, Slider, TextEdit, TextStyle, Ui, pos2, vec2,
};
use graphics::{EngineUpdates, Scene};
use synthesis::{
    EnzymeCommission,
    broad_target::{LibraryEnzyme, LibraryMaterial, LibraryMolecule, LibraryRoute, LibraryStep},
};

use crate::{
    file_io::managed_mols::ManagedMolProvider,
    mol_diagrams::{DIAGRAM_SCALE_MAX, DIAGRAM_SCALE_MIN, DiagramRequest},
    prefs::ToSave,
    reactions::{
        BuildingBlockColumn, BuildingBlockKind, BuildingBlocksView, Entry, ParticipantAction,
        Query, ReactionsState, RouteComponent, SynthesisDownloadId, SynthesisDownloads,
        SynthesisEntry, SynthesisLibraryData, SynthesisParticipantAction, SynthesisReactionsState,
        SynthesisSort, route_has_ec_class,
    },
    state::State,
    ui::{
        misc::{selector, selector_option},
        util::{QueryResult, apply_query_result, open_chebi_download},
    },
    util::{RedrawFlags, handle_err, make_lig_3d},
};

const PER_PAGE: usize = 4;
const SYNTHESIS_ROUTES_PER_PAGE: usize = 2;
/// Width of the column holding a "+" between participants.
const PLUS_WIDTH: f32 = 14.0;
const BLOCK_COL_SPACING: f32 = 12.0;
const BLOCK_TABLE_HEIGHT: f32 = 260.0;

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

    for (id, result) in state.ui.synthesis_reactions.take_downloads() {
        let downloaded_lig_i = state.ligands.len();
        let label = id.label();
        let result = match (id, result) {
            (_, Err(error)) => Err(error),
            (SynthesisDownloadId::Chebi(id), Ok(downloaded)) => {
                open_chebi_download(state, scene, redraw, updates, id, downloaded)
            }
            (SynthesisDownloadId::PubChem(id), Ok(downloaded)) => {
                let mut unused_reset_cam = false;
                apply_query_result(
                    state,
                    scene,
                    redraw,
                    &mut unused_reset_cam,
                    updates,
                    QueryResult::Small {
                        downloaded,
                        provider: ManagedMolProvider::Pubchem,
                        key: id.to_string(),
                        query: id.to_string(),
                        message: format!("Loaded CID {id} from PubChem (over the internet)"),
                        canonical_sdf: false,
                    },
                );
                if state.ligands.len() > downloaded_lig_i {
                    Ok(())
                } else {
                    Err(format!("Unable to open {label} in Molchanica."))
                }
            }
        };

        if result.is_ok()
            && state
                .ligands
                .get(downloaded_lig_i)
                .is_some_and(|ligand| ligand.common.is_2d)
        {
            make_lig_3d(state, downloaded_lig_i, scene, updates);
        }

        match result {
            Ok(()) => {
                state.ui.synthesis_reactions.download_message =
                    Some(format!("Opened {label} in Molchanica."));
            }
            Err(error) => {
                handle_err(&mut state.ui, error.clone());
                state.ui.synthesis_reactions.download_error = Some(error);
            }
        }
    }
}

pub(super) fn synthesis_reactions_window(
    state: &mut SynthesisReactionsState,
    to_save: &mut ToSave,
    ui: &mut Ui,
) {
    state.diagrams.poll(ui.ctx());
    ui.heading(RichText::new("Synthesis reaction library").color(Color32::WHITE));
    if let Some(message) = &state.protocol_export.message {
        ui.label(message);
    }
    if let Some(error) = &state.protocol_export.error {
        ui.colored_label(Color32::LIGHT_RED, error);
    }

    let mut retry = false;
    match &state.entry {
        SynthesisEntry::NotLoaded => {
            state.load();
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Building the synthesis library…");
            });
            return;
        }
        SynthesisEntry::Loading(_) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Building and validating synthesis routes…");
            });
            return;
        }
        SynthesisEntry::Failed(error) => {
            ui.colored_label(Color32::LIGHT_RED, error);
            retry = ui.button("Retry").clicked();
        }
        SynthesisEntry::Ready(_) => {}
    }
    if retry {
        state.retry();
        ui.ctx().request_repaint();
        return;
    }

    let SynthesisEntry::Ready(data) = &state.entry else {
        return;
    };

    CollapsingHeader::new("Coverage summary")
        .id_salt("synthesis_coverage_summary")
        .show(ui, |ui| {
            ui.monospace(data.summary.trim());
        });

    if building_blocks(
        data,
        &mut state.building_blocks,
        &mut state.component_filter,
        ui,
    ) {
        state.page = 0;
    }

    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label("Search:");
        if ui
            .add(
                TextEdit::singleline(&mut state.search)
                    .desired_width(220.0)
                    .hint_text("target, molecule, enzyme, EC, key…"),
            )
            .changed()
        {
            state.page = 0;
        }

        ui.label("Target class:");
        let class_label = if state.class_filter.is_empty() {
            "All classes"
        } else {
            &state.class_filter
        };
        ComboBox::from_id_salt("synthesis_class_filter")
            .selected_text(class_label)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_value(&mut state.class_filter, String::new(), "All classes")
                    .changed()
                {
                    state.page = 0;
                }
                for class in &data.classes {
                    if ui
                        .selectable_value(&mut state.class_filter, class.clone(), class)
                        .changed()
                    {
                        state.page = 0;
                    }
                }
            });

        ui.label("Enzyme class:");
        let enzyme_class_label = match state.enzyme_class_filter {
            None => "All classes".to_owned(),
            Some(top) => top.to_string(),
        };
        ComboBox::from_id_salt("synthesis_enzyme_class_filter")
            .selected_text(enzyme_class_label)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_value(&mut state.enzyme_class_filter, None, "All classes")
                    .changed()
                {
                    state.page = 0;
                }
                for &top in &data.ec_classes {
                    if ui
                        .selectable_value(
                            &mut state.enzyme_class_filter,
                            Some(top),
                            top.to_string(),
                        )
                        .on_hover_text("Routes with any step catalyzed by an enzyme in this class.")
                        .changed()
                    {
                        state.page = 0;
                    }
                }
            });

        ui.label("Steps:");
        ComboBox::from_id_salt("synthesis_step_filter")
            .selected_text(match state.step_filter {
                0 => "Any".to_owned(),
                steps => steps.to_string(),
            })
            .show_ui(ui, |ui| {
                if ui
                    .selectable_value(&mut state.step_filter, 0, "Any")
                    .changed()
                {
                    state.page = 0;
                }
                let max_steps = data
                    .routes
                    .iter()
                    .map(|route| route.steps.len())
                    .max()
                    .unwrap_or(0);
                for steps in 1..=max_steps {
                    if ui
                        .selectable_value(&mut state.step_filter, steps, steps.to_string())
                        .changed()
                    {
                        state.page = 0;
                    }
                }
            });

        ui.label("Sort:");
        if let Some(sort) = selector(
            ui,
            state.sort,
            &[
                (
                    SynthesisSort::TargetAscending,
                    "Target A–Z",
                    "Sort by target name, ascending.",
                ),
                (
                    SynthesisSort::TargetDescending,
                    "Target Z–A",
                    "Sort by target name, descending.",
                ),
                (
                    SynthesisSort::Class,
                    "Target class",
                    "Sort by target class, then target name.",
                ),
                (
                    SynthesisSort::FewestSteps,
                    "Fewest steps",
                    "Show shorter routes first.",
                ),
                (
                    SynthesisSort::MostSteps,
                    "Most steps",
                    "Show longer routes first.",
                ),
            ],
        ) {
            if sort != state.sort {
                state.sort = sort;
                state.page = 0;
            }
        }
    });

    if let Some(component) = state.component_filter {
        ui.horizontal_wrapped(|ui| {
            let kind = match component {
                RouteComponent::Feedstock(_) => "feedstock",
                RouteComponent::Enzyme(_) => "enzyme",
                RouteComponent::Cofactor(_) => "cofactor",
            };
            ui.label(format!("Only routes using the {kind}"));
            ui.label(
                RichText::new(component.name(&data.inventory))
                    .strong()
                    .color(Color32::WHITE),
            );
            if ui
                .button("Clear")
                .on_hover_text("Show routes regardless of the building blocks they use.")
                .clicked()
            {
                state.component_filter = None;
                state.page = 0;
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
                    SynthesisParticipantAction::OpenDatabasePage,
                    "Open molecule page",
                    "Open ChEBI when available, otherwise PubChem.",
                ),
                (
                    SynthesisParticipantAction::Download,
                    "Open in Molchanica",
                    "Download the ChEBI or PubChem structure and open it in Molchanica.",
                ),
                (
                    SynthesisParticipantAction::SearchRhea,
                    "Search Rhea",
                    "Open Rhea reactions for the molecule's ChEBI identifier.",
                ),
            ],
        ) {
            state.participant_action = action;
        }

        ui.add_space(12.0);
        diagram_scale_slider(to_save, ui);
    });
    let diagram_scale = to_save.reaction_diagram_scale;

    if !state.downloads.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.spinner();
            ui.label("Downloading:");
            for id in state.downloads.keys() {
                ui.label(id.label());
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

    let search = state.search.trim().to_ascii_lowercase();
    let mut visible: Vec<_> = data
        .routes
        .iter()
        .enumerate()
        .filter(|(_, route)| {
            (state.class_filter.is_empty() || route.class == state.class_filter)
                && state
                    .enzyme_class_filter
                    .is_none_or(|top| route_has_ec_class(route, top))
                && state
                    .component_filter
                    .is_none_or(|component| component.used_by(route, &data.inventory))
                && (state.step_filter == 0 || route.steps.len() == state.step_filter)
                && route_matches(route, &search)
        })
        .map(|(index, _)| index)
        .collect();
    sort_routes(&mut visible, &data.routes, state.sort);

    let count = visible.len();
    if count == 0 {
        state.page = 0;
        ui.separator();
        ui.label("No synthesis routes match these filters.");
        return;
    }

    let pages = count.div_ceil(SYNTHESIS_ROUTES_PER_PAGE);
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
        let first = state.page * SYNTHESIS_ROUTES_PER_PAGE + 1;
        let last = ((state.page + 1) * SYNTHESIS_ROUTES_PER_PAGE).min(count);
        ui.label(format!("{first}–{last} of {count} routes"));
    });
    ui.add_space(6.0);

    let mut clicked_participant = None;
    let mut protocol_index = None;
    let mut diagram_requests = Vec::new();
    ScrollArea::vertical()
        .id_salt(("synthesis_reactions", state.page, &search))
        .max_height(ui.available_height())
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for &index in visible
                .iter()
                .skip(state.page * SYNTHESIS_ROUTES_PER_PAGE)
                .take(SYNTHESIS_ROUTES_PER_PAGE)
            {
                if synthesis_route_card(
                    &data.routes[index],
                    state.participant_action,
                    diagram_scale,
                    &state.downloads,
                    &mut state.diagrams,
                    &mut clicked_participant,
                    &mut diagram_requests,
                    ui,
                ) {
                    protocol_index = Some(index);
                }
                ui.add_space(8.0);
            }
        });
    state
        .diagrams
        .request(&diagram_requests, diagram_scale, ui.ctx());

    if let Some(index) = protocol_index {
        state
            .protocol_export
            .begin(&data.routes[index], &data.inventory);
    }

    if let Some(molecule) = clicked_participant {
        match state.participant_action {
            SynthesisParticipantAction::OpenDatabasePage => {
                if let Some(id) = molecule.chebi_id {
                    chebi::open_overview(id);
                } else if let Some(id) = molecule.pubchem_id {
                    pubchem::open_overview(id);
                }
            }
            SynthesisParticipantAction::Download => {
                state.download(molecule.chebi_id, molecule.pubchem_id);
            }
            SynthesisParticipantAction::SearchRhea => {
                if let Some(id) = molecule.chebi_id {
                    let _ = webbrowser::open(&format!(
                        "https://www.rhea-db.org/rhea?query=chebi_exact%3A{id}"
                    ));
                }
            }
        }
        ui.ctx().request_repaint();
    }
}

/// One building block, flattened so feedstocks, enzymes, and cofactors share a table.
struct BlockRow<'a> {
    component: RouteComponent,
    name: &'a str,
    pubchem_id: Option<u32>,
    chebi_id: Option<u32>,
    ec: Option<EnzymeCommission>,
    uniprot_id: Option<&'a str>,
    detail: String,
    hover: String,
    /// The number of library routes using this building block.
    routes: usize,
}

/// A sortable table of the library's feedstocks, enzymes, or cofactors. Clicking a name shows
/// only the routes using it. Returns true if that route filter changed.
fn building_blocks(
    data: &SynthesisLibraryData,
    view: &mut BuildingBlocksView,
    component_filter: &mut Option<RouteComponent>,
    ui: &mut Ui,
) -> bool {
    let inventory = &data.inventory;
    let mut changed = false;

    CollapsingHeader::new("Building blocks: feedstocks, enzymes, and cofactors")
        .id_salt("synthesis_building_blocks")
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let labels = [
                    format!("Feedstocks ({})", inventory.feedstocks.len()),
                    format!("Enzymes ({})", inventory.enzymes.len()),
                    format!("Cofactors ({})", inventory.cofactors.len()),
                ];
                if let Some(kind) = selector(
                    ui,
                    view.kind,
                    &[
                        (
                            BuildingBlockKind::Feedstocks,
                            &labels[0],
                            "Purchased organic inputs, and gases.",
                        ),
                        (
                            BuildingBlockKind::Enzymes,
                            &labels[1],
                            "Candidate catalysts for the selected reaction templates.",
                        ),
                        (
                            BuildingBlockKind::Cofactors,
                            &labels[2],
                            "Consumed carriers and donors, and catalytic cofactors.",
                        ),
                    ],
                ) {
                    view.kind = kind;
                }

                ui.add_space(12.0);
                ui.label("Filter:");
                ui.add(
                    TextEdit::singleline(&mut view.search)
                        .desired_width(200.0)
                        .hint_text("name, role, ID…"),
                );
            });

            let mut rows = block_rows(data, view.kind);
            let total = rows.len();
            let search = view.search.trim().to_ascii_lowercase();
            rows.retain(|row| block_matches(row, &search));

            let columns = block_columns(view.kind);
            // The sort column may belong to another kind, e.g. EC numbers for feedstocks.
            let sort = if columns.iter().any(|&(column, _, _)| column == view.sort) {
                view.sort
            } else {
                BuildingBlockColumn::Name
            };
            sort_blocks(&mut rows, sort, view.descending);

            ui.weak(format!(
                "{} of {total} shown. Click a name to show only the routes using it; click a \
                 column heading to sort.",
                rows.len()
            ));

            // The name column takes the width the others leave.
            let fixed: f32 = columns.iter().map(|&(_, _, width)| width).sum();
            let gaps = BLOCK_COL_SPACING * (columns.len() - 1) as f32;
            let name_width =
                (ui.available_width() - fixed - gaps - ui.spacing().scroll.bar_width - 4.0)
                    .max(160.0);
            let width = |column: BuildingBlockColumn, width: f32| {
                if column == BuildingBlockColumn::Name {
                    name_width
                } else {
                    width
                }
            };

            // Headings stay outside the scroll area, so they remain visible.
            Grid::new(("synthesis_block_headings", view.kind))
                .num_columns(columns.len())
                .min_col_width(0.0)
                .spacing([BLOCK_COL_SPACING, 4.0])
                .show(ui, |ui| {
                    for &(column, heading, w) in columns {
                        table_cell(ui, width(column, w), |ui| {
                            let selected = column == sort;
                            let heading = match (selected, view.descending) {
                                (false, _) => heading.to_owned(),
                                (true, false) => format!("{heading} ⬆"),
                                (true, true) => format!("{heading} ⬇"),
                            };

                            if selector_option(ui, selected, heading)
                                .on_hover_text("Sort by this column; click again to reverse.")
                                .clicked()
                            {
                                if selected {
                                    view.descending = !view.descending;
                                } else {
                                    view.sort = column;
                                    // Most-used first is the useful default for counts.
                                    view.descending = column == BuildingBlockColumn::Routes;
                                }
                            }
                        });
                    }
                    ui.end_row();
                });

            ScrollArea::vertical()
                .id_salt(("synthesis_block_rows", view.kind))
                .max_height(BLOCK_TABLE_HEIGHT)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    Grid::new(("synthesis_block_grid", view.kind))
                        .num_columns(columns.len())
                        .striped(true)
                        .min_col_width(0.0)
                        .spacing([BLOCK_COL_SPACING, 4.0])
                        .show(ui, |ui| {
                            for row in &rows {
                                for &(column, _, w) in columns {
                                    table_cell(ui, width(column, w), |ui| {
                                        block_cell(row, column, component_filter, &mut changed, ui);
                                    });
                                }
                                ui.end_row();
                            }
                        });
                });
        });

    changed
}

/// `(column, heading, width)`. The name column's width is set from the space left over.
fn block_columns(kind: BuildingBlockKind) -> &'static [(BuildingBlockColumn, &'static str, f32)] {
    use BuildingBlockColumn::*;

    match kind {
        BuildingBlockKind::Feedstocks | BuildingBlockKind::Cofactors => &[
            (Name, "Name", 0.0),
            (PubChem, "PubChem CID", 100.0),
            (Chebi, "ChEBI ID", 90.0),
            (Detail, "Role", 170.0),
            (Routes, "Routes", 70.0),
        ],
        BuildingBlockKind::Enzymes => &[
            (Name, "Name", 0.0),
            (Ec, "EC number", 100.0),
            (UniProt, "UniProt ID", 90.0),
            (Detail, "Reaction family", 230.0),
            (Routes, "Routes", 70.0),
        ],
    }
}

fn block_rows(data: &SynthesisLibraryData, kind: BuildingBlockKind) -> Vec<BlockRow<'_>> {
    let inventory = &data.inventory;
    let routes = |component: RouteComponent| {
        data.routes
            .iter()
            .filter(|route| component.used_by(route, inventory))
            .count()
    };

    match kind {
        BuildingBlockKind::Feedstocks => inventory
            .feedstocks
            .iter()
            .enumerate()
            .map(|(i, material)| {
                let component = RouteComponent::Feedstock(i);
                material_row(component, material, routes(component))
            })
            .collect(),
        BuildingBlockKind::Cofactors => inventory
            .cofactors
            .iter()
            .enumerate()
            .map(|(i, material)| {
                let component = RouteComponent::Cofactor(i);
                material_row(component, material, routes(component))
            })
            .collect(),
        BuildingBlockKind::Enzymes => inventory
            .enzymes
            .iter()
            .enumerate()
            .map(|(i, catalyst)| {
                let component = RouteComponent::Enzyme(i);
                let enzyme = &catalyst.enzyme;
                let families = catalyst.families.join("; ");

                BlockRow {
                    component,
                    name: &enzyme.common_name,
                    pubchem_id: None,
                    chebi_id: None,
                    ec: enzyme.ec,
                    uniprot_id: enzyme.uniprot_id.as_deref(),
                    hover: format!(
                        "Reaction families: {families}\n\nClick to show only the routes using it."
                    ),
                    detail: families,
                    routes: routes(component),
                }
            })
            .collect(),
    }
}

fn material_row(
    component: RouteComponent,
    material: &LibraryMaterial,
    routes: usize,
) -> BlockRow<'_> {
    BlockRow {
        component,
        name: &material.name,
        pubchem_id: material.pubchem_id,
        chebi_id: material.chebi_id,
        ec: None,
        uniprot_id: None,
        detail: material.role.clone(),
        hover: format!(
            "{} {}\n{}\n\nClick to show only the routes using it.",
            material.supplier, material.product_number, material.formulation_note
        ),
        routes,
    }
}

fn block_cell(
    row: &BlockRow,
    column: BuildingBlockColumn,
    component_filter: &mut Option<RouteComponent>,
    changed: &mut bool,
    ui: &mut Ui,
) {
    match column {
        BuildingBlockColumn::Name => {
            let selected = *component_filter == Some(row.component);
            if ui
                .selectable_label(selected, row.name)
                .on_hover_text(row.hover.as_str())
                .clicked()
            {
                *component_filter = if selected { None } else { Some(row.component) };
                *changed = true;
            }
        }
        BuildingBlockColumn::PubChem => match row.pubchem_id {
            Some(id) => {
                ui.hyperlink_to(
                    id.to_string(),
                    format!("https://pubchem.ncbi.nlm.nih.gov/compound/{id}"),
                );
            }
            None => {
                ui.weak("—");
            }
        },
        BuildingBlockColumn::Chebi => match row.chebi_id {
            Some(id) => {
                ui.hyperlink_to(
                    id.to_string(),
                    format!("https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{id}"),
                );
            }
            None => {
                ui.weak("—");
            }
        },
        BuildingBlockColumn::Ec => match row.ec {
            Some(ec) => {
                let number = ec.number();
                ui.hyperlink_to(&number, format!("https://enzyme.expasy.org/EC/{number}"));
            }
            None => {
                ui.weak("—");
            }
        },
        BuildingBlockColumn::UniProt => match row.uniprot_id {
            Some(accession) => {
                ui.hyperlink_to(
                    accession,
                    format!("https://www.uniprot.org/uniprotkb/{accession}/entry"),
                );
            }
            None => {
                ui.weak("—");
            }
        },
        BuildingBlockColumn::Detail => {
            ui.label(&row.detail).on_hover_text(row.detail.as_str());
        }
        BuildingBlockColumn::Routes => {
            ui.label(row.routes.to_string());
        }
    }
}

fn block_matches(row: &BlockRow, search: &str) -> bool {
    if search.is_empty() {
        return true;
    }

    let mut values = vec![
        row.name.to_ascii_lowercase(),
        row.detail.to_ascii_lowercase(),
    ];
    if let Some(id) = row.pubchem_id {
        values.push(format!("cid:{id}"));
        values.push(id.to_string());
    }
    if let Some(id) = row.chebi_id {
        values.push(format!("chebi:{id}"));
        values.push(id.to_string());
    }
    if let Some(ec) = row.ec {
        values.push(ec.to_string().to_ascii_lowercase());
    }
    if let Some(accession) = row.uniprot_id {
        values.push(accession.to_ascii_lowercase());
    }

    search
        .split_whitespace()
        .all(|term| values.iter().any(|value| value.contains(term)))
}

/// Ties sort by name, ascending.
fn sort_blocks(rows: &mut [BlockRow], column: BuildingBlockColumn, descending: bool) {
    let directed = |ordering: Ordering| {
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    };

    rows.sort_by(|a, b| {
        let name = || {
            a.name
                .to_ascii_lowercase()
                .cmp(&b.name.to_ascii_lowercase())
        };
        let ordering = match column {
            BuildingBlockColumn::Name => directed(name()),
            BuildingBlockColumn::Detail => directed(
                a.detail
                    .to_ascii_lowercase()
                    .cmp(&b.detail.to_ascii_lowercase()),
            ),
            BuildingBlockColumn::Routes => directed(a.routes.cmp(&b.routes)),
            BuildingBlockColumn::PubChem => cmp_present(a.pubchem_id, b.pubchem_id, descending),
            BuildingBlockColumn::Chebi => cmp_present(a.chebi_id, b.chebi_id, descending),
            BuildingBlockColumn::Ec => cmp_present(a.ec.map(ec_key), b.ec.map(ec_key), descending),
            BuildingBlockColumn::UniProt => cmp_present(a.uniprot_id, b.uniprot_id, descending),
        };
        ordering.then_with(name)
    });
}

/// Rows missing the value go last, in either direction.
fn cmp_present<T: Ord>(a: Option<T>, b: Option<T>, descending: bool) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) if descending => b.cmp(&a),
        (Some(a), Some(b)) => a.cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Numeric, level by level: EC 1.1.1.10 follows EC 1.1.1.9.
fn ec_key(ec: EnzymeCommission) -> (u8, u8, u8, u16, bool) {
    (
        ec.top_level as u8,
        ec.level_1,
        ec.level_2,
        ec.level_3,
        ec.preliminary,
    )
}

fn route_matches(route: &LibraryRoute, search: &str) -> bool {
    if search.is_empty() {
        return true;
    }

    let mut values = vec![route.class.to_ascii_lowercase()];
    push_molecule_search_values(&route.target, &mut values);
    for molecule in &route.starting_materials {
        push_molecule_search_values(molecule, &mut values);
    }
    for step in &route.steps {
        values.extend(
            [
                &step.key,
                &step.family,
                &step.generic_equation,
                &step.evidence,
                &step.reversibility,
                &step.limitations,
            ]
            .map(|value| value.to_ascii_lowercase()),
        );
        for molecule in step.reactants.iter().chain(&step.products) {
            push_molecule_search_values(molecule, &mut values);
        }
        for enzyme in &step.enzymes {
            values.push(enzyme.common_name.to_ascii_lowercase());
            if let Some(ec) = &enzyme.ec {
                values.push(ec.to_string().to_ascii_lowercase());
            }
            if let Some(accession) = &enzyme.uniprot_id {
                values.push(accession.to_ascii_lowercase());
            }
        }
        values.extend(
            step.catalytic_cofactors
                .iter()
                .chain(&step.requirements)
                .map(|value| value.to_ascii_lowercase()),
        );
    }
    values.extend(
        route
            .support_notes
            .iter()
            .map(|value| value.to_ascii_lowercase()),
    );

    search
        .split_whitespace()
        .all(|term| values.iter().any(|value| value.contains(term)))
}

fn push_molecule_search_values(molecule: &LibraryMolecule, values: &mut Vec<String>) {
    values.push(molecule.name.to_ascii_lowercase());
    if let Some(id) = molecule.chebi_id {
        values.push(format!("chebi:{id}"));
    }
    if let Some(id) = molecule.pubchem_id {
        values.push(format!("cid:{id}"));
        values.push(id.to_string());
    }
}

fn sort_routes(indices: &mut [usize], routes: &[LibraryRoute], sort: SynthesisSort) {
    let target = |index: usize| routes[index].target.name.to_ascii_lowercase();
    indices.sort_by(|&a, &b| match sort {
        SynthesisSort::TargetAscending => target(a).cmp(&target(b)),
        SynthesisSort::TargetDescending => target(b).cmp(&target(a)),
        SynthesisSort::Class => routes[a]
            .class
            .cmp(&routes[b].class)
            .then_with(|| target(a).cmp(&target(b))),
        SynthesisSort::FewestSteps => routes[a]
            .steps
            .len()
            .cmp(&routes[b].steps.len())
            .then_with(|| target(a).cmp(&target(b))),
        SynthesisSort::MostSteps => routes[b]
            .steps
            .len()
            .cmp(&routes[a].steps.len())
            .then_with(|| target(a).cmp(&target(b))),
    });
}

fn synthesis_route_card(
    route: &LibraryRoute,
    action: SynthesisParticipantAction,
    diagram_scale: f32,
    downloads: &SynthesisDownloads,
    diagrams: &mut crate::mol_diagrams::DiagramCache,
    clicked: &mut Option<LibraryMolecule>,
    diagram_requests: &mut Vec<DiagramRequest>,
    ui: &mut Ui,
) -> bool {
    let mut create_protocol = false;
    Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(&route.target.name).heading().strong());
            ui.weak(&route.class);
            ui.weak(format!(
                "{} step{}",
                route.steps.len(),
                if route.steps.len() == 1 { "" } else { "s" }
            ));
            molecule_identifier_links(&route.target, ui);
            create_protocol = ui
                .button("Create protocol")
                .on_hover_text(
                    "Save a Markdown protocol with reagent sources and explicit preparation gaps.",
                )
                .clicked();
        });
        ui.horizontal_wrapped(|ui| {
            ui.weak("Starting materials:");
            for (index, molecule) in route.starting_materials.iter().enumerate() {
                if index > 0 {
                    ui.weak("+");
                }
                ui.label(&molecule.name);
            }
        });
        ui.add_space(5.0);

        for (index, step) in route.steps.iter().enumerate() {
            synthesis_step_card(
                &route.target.name,
                index,
                step,
                action,
                diagram_scale,
                downloads,
                diagrams,
                clicked,
                diagram_requests,
                ui,
            );
            if index + 1 < route.steps.len() {
                ui.add_space(6.0);
            }
        }

        if !route.support_notes.is_empty() {
            CollapsingHeader::new("Route requirements and caveats")
                .id_salt(("route_notes", &route.target.name))
                .show(ui, |ui| {
                    for note in &route.support_notes {
                        ui.label(note);
                    }
                });
        }
    });
    create_protocol
}

fn synthesis_step_card(
    route_target: &str,
    index: usize,
    step: &LibraryStep,
    action: SynthesisParticipantAction,
    diagram_scale: f32,
    downloads: &SynthesisDownloads,
    diagrams: &mut crate::mol_diagrams::DiagramCache,
    clicked: &mut Option<LibraryMolecule>,
    diagram_requests: &mut Vec<DiagramRequest>,
    ui: &mut Ui,
) {
    Frame::canvas(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("Step {}", index + 1)).strong());
            ui.label(&step.family);
            ui.weak(format!("[{}]", step.key));
        });

        let equals_width = 36.0;
        let spacing = ui.spacing().item_spacing.x;
        let available_for_sides = ui.available_width() - equals_width - 2.0 * spacing;
        let side_width = (available_for_sides / 2.0).max(0.0);
        ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
            ui.allocate_ui_with_layout(vec2(side_width, 0.0), Layout::top_down(Align::Min), |ui| {
                ui.set_max_width(side_width);
                synthesis_side(
                    &step.reactants,
                    "Reactants",
                    Color32::from_rgb(125, 195, 230),
                    action,
                    diagram_scale,
                    downloads,
                    diagrams,
                    clicked,
                    diagram_requests,
                    ui,
                );
            });

            let label_height =
                ui.text_style_height(&TextStyle::Small) + ui.spacing().item_spacing.y;
            let name_height = name_row_height(ui);
            let (rect, response) = ui.allocate_exact_size(
                vec2(equals_width, label_height + name_height),
                Sense::hover(),
            );
            ui.painter().text(
                pos2(
                    rect.center().x,
                    rect.min.y + label_height + name_height / 2.0,
                ),
                Align2::CENTER_CENTER,
                "=",
                FontId::proportional(28.0),
                ui.visuals().text_color(),
            );
            response.on_hover_text("Planned synthesis direction: reactants to products");

            ui.allocate_ui_with_layout(vec2(side_width, 0.0), Layout::top_down(Align::Min), |ui| {
                ui.set_max_width(side_width);
                synthesis_side(
                    &step.products,
                    "Products",
                    Color32::from_rgb(150, 215, 170),
                    action,
                    diagram_scale,
                    downloads,
                    diagrams,
                    clicked,
                    diagram_requests,
                    ui,
                );
            });
        });

        enzyme_links(&step.enzymes, ui);
        if !step.catalytic_cofactors.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.weak("Catalytic cofactors:");
                ui.label(step.catalytic_cofactors.join(" + "));
            });
        }

        CollapsingHeader::new("Reaction scope and evidence")
            .id_salt(("step_scope", route_target, &step.key, index))
            .show(ui, |ui| {
                ui.label(&step.generic_equation);
                ui.horizontal_wrapped(|ui| {
                    ui.weak("Evidence:");
                    ui.label(&step.evidence);
                    ui.weak("Reversibility:");
                    ui.label(&step.reversibility);
                });
                if !step.requirements.is_empty() {
                    ui.weak("Requirements:");
                    for requirement in &step.requirements {
                        ui.label(format!("• {requirement}"));
                    }
                }
                if !step.limitations.is_empty() {
                    ui.weak("Limitations:");
                    ui.label(&step.limitations);
                }
                if !step.references.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        ui.weak("References:");
                        for (reference_index, reference) in step.references.iter().enumerate() {
                            ui.hyperlink_to(format!("{}", reference_index + 1), reference);
                        }
                    });
                }
            });
    });
}

fn synthesis_side(
    participants: &[LibraryMolecule],
    label: &str,
    color: Color32,
    action: SynthesisParticipantAction,
    diagram_scale: f32,
    downloads: &SynthesisDownloads,
    diagrams: &mut crate::mol_diagrams::DiagramCache,
    clicked: &mut Option<LibraryMolecule>,
    diagram_requests: &mut Vec<DiagramRequest>,
    ui: &mut Ui,
) {
    ui.label(RichText::new(label).small().color(color));

    let spacing = ui.spacing().item_spacing.x;
    let molecule_width = ui
        .available_width()
        .min(crate::mol_diagrams::DIAGRAM_WIDTH * diagram_scale);
    let column_width = molecule_width + PLUS_WIDTH + 2.0 * spacing;
    let per_row = (((ui.available_width() + spacing) / column_width) as usize).max(1);
    let row_layout = Layout::left_to_right(Align::Min);

    for (row, row_molecules) in participants.chunks(per_row).enumerate() {
        let first = row * per_row;
        ui.with_layout(row_layout, |ui| {
            for (index, molecule) in row_molecules.iter().enumerate() {
                let index = first + index;
                if index > 0 {
                    let (rect, _) = ui
                        .allocate_exact_size(vec2(PLUS_WIDTH, name_row_height(ui)), Sense::hover());
                    ui.painter().text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        "+",
                        TextStyle::Button.resolve(ui.style()),
                        color,
                    );
                }

                cell(ui, molecule_width, |ui| {
                    let download_id = molecule
                        .chebi_id
                        .map(SynthesisDownloadId::Chebi)
                        .or_else(|| molecule.pubchem_id.map(SynthesisDownloadId::PubChem));
                    let enabled = match action {
                        SynthesisParticipantAction::OpenDatabasePage
                        | SynthesisParticipantAction::Download => download_id.is_some(),
                        SynthesisParticipantAction::SearchRhea => molecule.chebi_id.is_some(),
                    };
                    let loading = action == SynthesisParticipantAction::Download
                        && download_id.is_some_and(|id| downloads.contains_key(&id));
                    let marker = if molecule.is_feedstock { " [FS]" } else { "" };
                    let button = Button::new(
                        RichText::new(format!("{}{marker}", molecule.name))
                            .strong()
                            .color(color),
                    )
                    .fill(color.gamma_multiply(0.12))
                    .frame(true)
                    .wrap_mode(egui::TextWrapMode::Wrap);
                    let hover = molecule_hover(molecule, action);
                    if ui
                        .add_enabled(enabled && !loading, button)
                        .on_hover_text(hover)
                        .clicked()
                    {
                        *clicked = Some(molecule.clone());
                    }
                });
            }
        });

        ui.with_layout(row_layout, |ui| {
            for (index, molecule) in row_molecules.iter().enumerate() {
                let index = first + index;
                if index > 0 {
                    ui.allocate_exact_size(vec2(PLUS_WIDTH, 0.0), Sense::hover());
                }
                cell(ui, molecule_width, |ui| {
                    if let Some(id) = molecule.chebi_id {
                        let retry = diagrams.show(id, diagram_scale, ui);
                        diagram_requests.push(DiagramRequest::new(id, molecule.smiles, retry));
                    } else {
                        ui.weak("No ChEBI diagram available.");
                    }
                });
            }
        });
    }
}

fn molecule_hover(molecule: &LibraryMolecule, action: SynthesisParticipantAction) -> String {
    let identifiers = match (molecule.chebi_id, molecule.pubchem_id) {
        (Some(chebi), Some(cid)) => format!("CHEBI:{chebi}; CID {cid}"),
        (Some(chebi), None) => format!("CHEBI:{chebi}"),
        (None, Some(cid)) => format!("CID {cid}"),
        (None, None) => "No ChEBI or PubChem identifier".to_owned(),
    };
    let action = match action {
        SynthesisParticipantAction::OpenDatabasePage => "Open molecule database page",
        SynthesisParticipantAction::Download => "Download and open in Molchanica",
        SynthesisParticipantAction::SearchRhea => "Search exact ChEBI matches in Rhea",
    };
    format!("{identifiers} — {action}")
}

fn molecule_identifier_links(molecule: &LibraryMolecule, ui: &mut Ui) {
    if let Some(id) = molecule.chebi_id {
        ui.hyperlink_to(
            format!("CHEBI:{id}"),
            format!("https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{id}"),
        );
        ui.hyperlink_to(
            "Rhea",
            format!("https://www.rhea-db.org/rhea?query=chebi_exact%3A{id}"),
        );
    }
    if let Some(id) = molecule.pubchem_id {
        ui.hyperlink_to(
            format!("CID {id}"),
            format!("https://pubchem.ncbi.nlm.nih.gov/compound/{id}"),
        );
    }
}

fn enzyme_links(enzymes: &[LibraryEnzyme], ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.weak("Enzyme candidates:");
        if enzymes.is_empty() {
            ui.label("none specified");
        }
        for (index, enzyme) in enzymes.iter().enumerate() {
            if index > 0 {
                ui.weak("or");
            }
            ui.label(&enzyme.common_name);
            if let Some(ec) = &enzyme.ec {
                let number = ec.number();
                ui.hyperlink_to(
                    ec.to_string(),
                    format!("https://enzyme.expasy.org/EC/{number}"),
                );
                ui.hyperlink_to(
                    "Rhea",
                    format!("https://www.rhea-db.org/rhea?query=ec%3A{number}"),
                );
            }
            if let Some(accession) = &enzyme.uniprot_id {
                ui.hyperlink_to(
                    format!("UniProt {accession}"),
                    format!("https://www.uniprot.org/uniprotkb/{accession}/entry"),
                );
            }
        }
    });
}

pub(super) fn reactions_window(state: &mut ReactionsState, to_save: &mut ToSave, ui: &mut Ui) {
    state.diagrams.poll(ui.ctx());
    ui.heading(RichText::new(&state.title).color(Color32::WHITE));
    let has_chebi_id = matches!(state.selected.as_ref(), Some(Query::Chebi(_)));
    if state.pubchem_cid.is_some() || has_chebi_id {
        ui.horizontal_wrapped(|ui| {
            if let Some(cid) = state.pubchem_cid {
                if ui
                    .button(RichText::new(format!("CID: {cid}")).color(Color32::WHITE))
                    .on_hover_text("Open this molecule's PubChem page in your web browser")
                    .clicked()
                {
                    pubchem::open_overview(cid);
                }
            }
            if let Some(Query::Chebi(id)) = &state.selected {
                if ui
                    .button(RichText::new(format!("CHEBI:{id}")).color(Color32::WHITE))
                    .on_hover_text("Open this molecule's ChEBI page in your web browser")
                    .clicked()
                {
                    chebi::open_overview(*id);
                }
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

        ui.add_space(12.0);
        diagram_scale_slider(to_save, ui);
    });
    let diagram_scale = to_save.reaction_diagram_scale;

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
                                diagram_scale,
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

    state
        .diagrams
        .request(&diagram_requests, diagram_scale, ui.ctx());

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
    diagram_scale: f32,
    clicked: &mut Option<(u32, String)>,
    diagram_requests: &mut Vec<DiagramRequest>,
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

            // Top-aligned, unlike `ui.horizontal`: that centers each column on the height of the
            // columns before it, which pushed the "=" and right side below the left side.
            ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
                ui.allocate_ui_with_layout(
                    vec2(side_width, 0.0),
                    Layout::top_down(Align::Min),
                    |ui| {
                        ui.set_max_width(side_width);
                        side(
                            left,
                            &reaction.reactants,
                            "Left side",
                            Color32::from_rgb(125, 195, 230),
                            state,
                            diagram_scale,
                            clicked,
                            diagram_requests,
                            ui,
                        );
                    },
                );

                // Line up with the molecule names, below each side's label.
                let label_height =
                    ui.text_style_height(&TextStyle::Small) + ui.spacing().item_spacing.y;
                let name_height = name_row_height(ui);
                let (rect, response) = ui.allocate_exact_size(
                    vec2(equals_width, label_height + name_height),
                    Sense::hover(),
                );
                ui.painter().text(
                    pos2(
                        rect.center().x,
                        rect.min.y + label_height + name_height / 2.0,
                    ),
                    Align2::CENTER_CENTER,
                    "=",
                    FontId::proportional(28.0),
                    ui.visuals().text_color(),
                );
                response.on_hover_text("Direction unspecified");

                ui.allocate_ui_with_layout(
                    vec2(side_width, 0.0),
                    Layout::top_down(Align::Min),
                    |ui| {
                        ui.set_max_width(side_width);
                        side(
                            right,
                            &reaction.products,
                            "Right side",
                            Color32::from_rgb(150, 215, 170),
                            state,
                            diagram_scale,
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
    diagram_scale: f32,
    clicked: &mut Option<(u32, String)>,
    diagram_requests: &mut Vec<DiagramRequest>,
    ui: &mut Ui,
) {
    ui.label(RichText::new(label).small().color(color));

    let names: Vec<&str> = equation.split(" + ").collect();
    let spacing = ui.spacing().item_spacing.x;
    let molecule_width = ui
        .available_width()
        .min(crate::mol_diagrams::DIAGRAM_WIDTH * diagram_scale);

    // Wrap manually, so each row of names sits on its own row of diagrams. Count a "+" column
    // with every molecule; this may fit one fewer on the first row, but never overflows.
    let column_width = molecule_width + PLUS_WIDTH + 2.0 * spacing;
    let per_row = (((ui.available_width() + spacing) / column_width) as usize).max(1);

    // `Align::Min`: a vertically-centered row staggers entries of different heights.
    let row_layout = Layout::left_to_right(Align::Min);

    for (row, row_names) in names.chunks(per_row).enumerate() {
        let first = row * per_row;

        // Names. A wrapped name makes the whole row taller, keeping the diagrams level.
        ui.with_layout(row_layout, |ui| {
            for (index, &participant) in row_names.iter().enumerate() {
                let index = first + index;
                if index > 0 {
                    let (rect, _) = ui
                        .allocate_exact_size(vec2(PLUS_WIDTH, name_row_height(ui)), Sense::hover());
                    ui.painter().text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        "+",
                        TextStyle::Button.resolve(ui.style()),
                        color,
                    );
                }

                cell(ui, molecule_width, |ui| {
                    // The filtered ChEBI-only list cannot be zipped with names: generic
                    // RHEA-COMP participants would shift every subsequent link.
                    let Some(id) = participant_chebi_id(participants, equation, index) else {
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
                        return;
                    };

                    let queried = matches!(
                        state.selected.as_ref(),
                        Some(Query::Chebi(query_id)) if *query_id == id
                    );
                    let loading = state.participant_action == ParticipantAction::Download
                        && state.downloads.contains_key(&id);
                    let action = match state.participant_action {
                        ParticipantAction::OpenChebiPage => "Open ChEBI molecule page in browser",
                        ParticipantAction::Download => "Download and open in Molchanica",
                        ParticipantAction::OpenRheaPage => "Open reaction",
                    };
                    let fill = if queried {
                        Color32::from_rgb(90, 105, 35)
                    } else {
                        color.gamma_multiply(0.12)
                    };
                    let text_color = if queried { Color32::WHITE } else { color };
                    let button = Button::new(RichText::new(participant).strong().color(text_color))
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
                });
            }
        });

        // Diagrams, in the same column positions as the names above.
        ui.with_layout(row_layout, |ui| {
            for index in first..first + row_names.len() {
                if index > 0 {
                    ui.allocate_exact_size(vec2(PLUS_WIDTH, 0.0), Sense::hover());
                }

                cell(ui, molecule_width, |ui| {
                    if let Some(id) = participant_chebi_id(participants, equation, index) {
                        let retry = state.diagrams.show(id, diagram_scale, ui);
                        diagram_requests.push(DiagramRequest::new(id, None, retry));
                    } else {
                        ui.weak("No 2D structure: no ChEBI ID supplied.");
                    }
                });
            }
        });
    }
}

/// Scales the 2D molecule diagrams in both reaction popups. Saved with the preferences.
fn diagram_scale_slider(to_save: &mut ToSave, ui: &mut Ui) {
    ui.label("Diagram size:");
    if ui
        .add(
            Slider::new(
                &mut to_save.reaction_diagram_scale,
                DIAGRAM_SCALE_MIN..=DIAGRAM_SCALE_MAX,
            )
            .fixed_decimals(1)
            .suffix("×"),
        )
        .on_hover_text("Scale the molecule structure diagrams.")
        .changed()
    {
        // Flushed by `check_prefs_save`, rather than writing the file on every drag step.
        to_save.save_flag = true;
    }
}

/// A fixed-width, top-aligned column entry, so names and diagrams share column positions.
fn cell(ui: &mut Ui, width: f32, add_contents: impl FnOnce(&mut Ui)) {
    ui.allocate_ui_with_layout(vec2(width, 0.0), Layout::top_down(Align::Min), |ui| {
        ui.set_width(width);
        add_contents(ui);
    });
}

/// A fixed-width, truncating table cell. Pinning both the minimum and maximum width keeps the
/// heading grid's columns aligned with the body grid's.
fn table_cell(ui: &mut Ui, width: f32, add_contents: impl FnOnce(&mut Ui)) {
    ui.allocate_ui_with_layout(
        vec2(width, ui.spacing().interact_size.y),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.set_min_width(width);
            ui.set_max_width(width);
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            add_contents(ui);
        },
    );
}

/// Height of a single-line molecule button; "+" and "=" are centered on it.
fn name_row_height(ui: &Ui) -> f32 {
    let text = ui.text_style_height(&TextStyle::Button) + 2.0 * ui.spacing().button_padding.y;
    text.max(ui.spacing().interact_size.y)
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
