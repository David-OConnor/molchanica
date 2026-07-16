//! A viewer and editor for the general force field parameters (`FfParamSet`) loaded at startup,
//! e.g. from Amber's parm19, ff19SB, gaff2, lipid21, and OL24. Implemented as a popup.
//!
//! Values typed into these tables are applied to `State::ff_param_set` immediately.

use std::{collections::HashMap, fmt::Display, hash::Hash, str::FromStr};

use bio_files::md_params::{ChargeParams, ForceFieldParams};
use dynamics::params::{FfParamSet, ProtFfChargeMap};
use egui::{Color32, Label, RichText, ScrollArea, TextEdit, Ui};
use na_seq::AminoAcidGeneral;

use crate::{
    button, label,
    state::State,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING, popups::close_btn,
    },
};

/// Height of a table cell; matches EGUI's default `interact_size.y`. Rows must be exactly this
/// tall, or the virtualized scroll position will drift from the rows drawn.
const CELL_H: f32 = 18.;
const TABLE_HEIGHT: f32 = 800.;

/// Gap left between the bottom of the popup and the bottom of the window.
const SCREEN_MARGIN: f32 = 16.;

// Column widths.
const W_TYPE: f32 = 54.;
const W_VAL: f32 = 84.;
const W_INT: f32 = 52.;
const W_RES: f32 = 96.;
const W_COMMENT: f32 = 300.;

/// Offset from the position shared by the other popups: this one is tall, so we start it higher
/// and further left to keep it on screen.
pub(in crate::ui) const POPUP_OFFSET: (f32, f32) = (-100., -200.);

/// Which field of `FfParamSet` is being viewed.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum FfCat {
    #[default]
    Peptide,
    SmallMol,
    Dna,
    Rna,
    Lipids,
    Carbohydrates,
    PeptideQ,
    LipidQ,
    DnaQ,
    RnaQ,
}

const CATS: [FfCat; 10] = [
    FfCat::Peptide,
    FfCat::SmallMol,
    FfCat::Dna,
    FfCat::Rna,
    FfCat::Lipids,
    FfCat::Carbohydrates,
    FfCat::PeptideQ,
    FfCat::LipidQ,
    FfCat::DnaQ,
    FfCat::RnaQ,
];

impl FfCat {
    fn name(self) -> &'static str {
        match self {
            Self::Peptide => "Peptide",
            Self::SmallMol => "Small mol",
            Self::Dna => "DNA",
            Self::Rna => "RNA",
            Self::Lipids => "Lipids",
            Self::Carbohydrates => "Carbohydrates",
            Self::PeptideQ => "Peptide charge",
            Self::LipidQ => "Lipid charge",
            Self::DnaQ => "DNA charge",
            Self::RnaQ => "RNA charge",
        }
    }

    fn hover(self) -> &'static str {
        match self {
            Self::Peptide => "Bonded and Lennard-Jones params for peptides. E.g. parm19 + ff19SB.",
            Self::SmallMol => {
                "Bonded and Lennard-Jones params for small organic molecules. E.g. gaff2."
            }
            Self::Dna => "Bonded and Lennard-Jones params for DNA. E.g. parm19 + OL24.",
            Self::Rna => "Bonded and Lennard-Jones params for RNA.",
            Self::Lipids => "Bonded and Lennard-Jones params for lipids. E.g. lipid21.",
            Self::Carbohydrates => "Bonded and Lennard-Jones params for carbohydrates.",
            Self::PeptideQ => {
                "Partial charge, and residue-type-to-FF-type mapping for amino acids. \
                E.g. amino19.lib, and its terminus variants."
            }
            Self::LipidQ => "Partial charge, and residue-type-to-FF-type mapping for lipids.",
            Self::DnaQ => "Partial charge, and residue-type-to-FF-type mapping for DNA.",
            Self::RnaQ => "Partial charge, and residue-type-to-FF-type mapping for RNA.",
        }
    }

    /// Is this set of params loaded?
    fn present(self, set: &FfParamSet) -> bool {
        match self {
            Self::Peptide => set.peptide.is_some(),
            Self::SmallMol => set.small_mol.is_some(),
            Self::Dna => set.dna.is_some(),
            Self::Rna => set.rna.is_some(),
            Self::Lipids => set.lipids.is_some(),
            Self::Carbohydrates => set.carbohydrates.is_some(),
            Self::PeptideQ => set.peptide_ff_q_map.is_some(),
            Self::LipidQ => set.lipid_ff_q_map.is_some(),
            Self::DnaQ => set.dna_ff_q_map.is_some(),
            Self::RnaQ => set.rna_ff_q_map.is_some(),
        }
    }
}

/// Which field of a `ForceFieldParams` is being viewed.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum FfKind {
    #[default]
    Bond,
    Angle,
    Dihedral,
    Improper,
    Mass,
    LennardJones,
}

const KINDS: [FfKind; 6] = [
    FfKind::Bond,
    FfKind::Angle,
    FfKind::Dihedral,
    FfKind::Improper,
    FfKind::Mass,
    FfKind::LennardJones,
];

impl FfKind {
    fn name(self) -> &'static str {
        match self {
            Self::Bond => "Bond",
            Self::Angle => "Angle",
            Self::Dihedral => "Dihedral",
            Self::Improper => "Improper",
            Self::Mass => "Mass",
            Self::LennardJones => "Lennard-Jones",
        }
    }

    fn hover(self) -> &'static str {
        match self {
            Self::Bond => "Bond stretching: length between two covalently-bonded atoms.",
            Self::Angle => "Angle bending: between three covalently-bonded atoms.",
            Self::Dihedral => "Proper dihedral, or torsion: rotation around the 2-3 bond.",
            Self::Improper => "Improper dihedral: hub-and-spoke; generally planar configurations.",
            Self::Mass => "Atomic mass, by force-field type.",
            Self::LennardJones => "Van der Waals and Pauli exclusion params, by force-field type.",
        }
    }
}

/// Which map of a `ProtFfChargeMapSet` is being viewed.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum ProtMapSel {
    #[default]
    Internal,
    NTerminus,
    CTerminus,
}

const PROT_MAPS: [ProtMapSel; 3] = [
    ProtMapSel::Internal,
    ProtMapSel::NTerminus,
    ProtMapSel::CTerminus,
];

impl ProtMapSel {
    fn name(self) -> &'static str {
        match self {
            Self::Internal => "Internal",
            Self::NTerminus => "N terminus",
            Self::CTerminus => "C terminus",
        }
    }
}

#[derive(Default)]
pub struct FfParamsUi {
    pub cat: FfCat,
    pub kind: FfKind,
    pub prot_map: ProtMapSel,
    pub filter: String,
    /// The cell currently being typed into: (cell id, raw text). We hold the raw text so
    /// partially-typed values, e.g. "-", or "1.", survive a round trip through the parsed value.
    pub cell_edit: Option<(String, String)>,
}

/// A text field for a table cell. Returns the new text, if it changed this frame.
fn edit_cell(
    ui: &mut Ui,
    id: &str,
    w: f32,
    display: String,
    cell_edit: &mut Option<(String, String)>,
) -> Option<String> {
    let mut text = match cell_edit {
        Some((id_active, t)) if id_active == id => t.clone(),
        _ => display,
    };

    let resp = ui.add_sized([w, CELL_H], TextEdit::singleline(&mut text).id_salt(id));

    if resp.changed() {
        *cell_edit = Some((id.to_owned(), text.clone()));
        return Some(text);
    }

    // Return to the canonical formatting once the user moves on.
    if resp.lost_focus()
        && cell_edit
            .as_ref()
            .is_some_and(|(id_active, _)| id_active == id)
    {
        *cell_edit = None;
    }

    None
}

/// A cell for any value displayable as, and parsable from text. E.g. floats and ints.
fn num_cell<T: Display + FromStr>(
    ui: &mut Ui,
    id: &str,
    val: &mut T,
    w: f32,
    cell_edit: &mut Option<(String, String)>,
) {
    if let Some(text) = edit_cell(ui, id, w, val.to_string(), cell_edit)
        && let Ok(v) = text.parse::<T>()
    {
        *val = v;
    }
}

/// Angles are stored in radians, but displayed and edited in degrees, as in the Amber
/// parameter files.
fn angle_cell(ui: &mut Ui, id: &str, val: &mut f32, cell_edit: &mut Option<(String, String)>) {
    let display = format!("{:.3}", val.to_degrees());

    if let Some(text) = edit_cell(ui, id, W_VAL, display, cell_edit)
        && let Ok(v) = text.parse::<f32>()
    {
        *val = v.to_radians();
    }
}

fn text_cell(
    ui: &mut Ui,
    id: &str,
    val: &mut String,
    w: f32,
    cell_edit: &mut Option<(String, String)>,
) {
    if let Some(text) = edit_cell(ui, id, w, val.clone(), cell_edit) {
        *val = text;
    }
}

/// A read-only cell.
fn label_cell(ui: &mut Ui, text: &str, w: f32, color: Color32) {
    ui.add_sized([w, CELL_H], Label::new(RichText::new(text).color(color)));
}

fn comment_cell(ui: &mut Ui, comment: &Option<String>) {
    if let Some(c) = comment {
        label_cell(ui, c, W_COMMENT, Color32::DARK_GRAY);
    }
}

fn header(ui: &mut Ui, cols: &[(&str, f32)], shown: usize, total: usize) {
    let count = if shown == total {
        format!("{total} entries")
    } else {
        format!("{shown} of {total} entries")
    };
    label!(ui, count, Color32::GRAY);

    ui.horizontal(|ui| {
        for (name, w) in cols {
            label_cell(ui, name, *w, Color32::GRAY);
        }
    });

    ui.separator();
}

/// These tables can have thousands of entries, so we only lay out the visible rows. Each row must
/// be exactly `CELL_H` tall for the scroll position to line up with the rows drawn.
fn table<R>(ui: &mut Ui, id: &str, rows: &[R], mut row: impl FnMut(&mut Ui, &R)) {
    ScrollArea::vertical()
        .id_salt(id)
        .max_height(TABLE_HEIGHT)
        .show_rows(ui, CELL_H, rows.len(), |ui, range| {
            for i in range {
                ui.horizontal(|ui| {
                    row(ui, &rows[i]);
                });
            }
        });
}

fn matches(filter: &str, text: &str) -> bool {
    filter.is_empty() || text.to_lowercase().contains(filter)
}

/// The tables for a `ForceFieldParams`: one per field, selected by `kind`.
fn ff_params_table(params: &mut ForceFieldParams, kind: FfKind, st: &mut FfParamsUi, ui: &mut Ui) {
    let filter = st.filter.to_lowercase();
    let cell_edit = &mut st.cell_edit;

    // We clone the key of each row out of the map, so we can take a mutable borrow of the value
    // while laying its row out.
    match kind {
        FfKind::Bond => {
            let mut keys = Vec::new();
            for key in params.bond.keys() {
                if matches(&filter, &format!("{}-{}", key.0, key.1)) {
                    keys.push(key.clone());
                }
            }
            keys.sort_unstable();

            header(
                ui,
                &[
                    ("Type 1", W_TYPE),
                    ("Type 2", W_TYPE),
                    ("k (kcal/mol/Å²)", W_VAL),
                    ("r₀ (Å)", W_VAL),
                    ("Comment", W_COMMENT),
                ],
                keys.len(),
                params.bond.len(),
            );

            table(ui, "ff_bond", &keys, |ui, key| {
                let Some(val) = params.bond.get_mut(key) else {
                    return;
                };

                label_cell(ui, &key.0, W_TYPE, Color32::WHITE);
                label_cell(ui, &key.1, W_TYPE, Color32::WHITE);

                num_cell(
                    ui,
                    &format!("bond_k_{key:?}"),
                    &mut val.k_b,
                    W_VAL,
                    cell_edit,
                );
                num_cell(
                    ui,
                    &format!("bond_r_{key:?}"),
                    &mut val.r_0,
                    W_VAL,
                    cell_edit,
                );

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::Angle => {
            let mut keys = Vec::new();
            for key in params.angle.keys() {
                if matches(&filter, &format!("{}-{}-{}", key.0, key.1, key.2)) {
                    keys.push(key.clone());
                }
            }
            keys.sort_unstable();

            header(
                ui,
                &[
                    ("Type 1", W_TYPE),
                    ("Type 2", W_TYPE),
                    ("Type 3", W_TYPE),
                    ("k (kcal/mol/rad²)", W_VAL),
                    ("θ₀ (°)", W_VAL),
                    ("Comment", W_COMMENT),
                ],
                keys.len(),
                params.angle.len(),
            );

            table(ui, "ff_angle", &keys, |ui, key| {
                let Some(val) = params.angle.get_mut(key) else {
                    return;
                };

                label_cell(ui, &key.0, W_TYPE, Color32::WHITE);
                label_cell(ui, &key.1, W_TYPE, Color32::WHITE);
                label_cell(ui, &key.2, W_TYPE, Color32::WHITE);

                num_cell(
                    ui,
                    &format!("angle_k_{key:?}"),
                    &mut val.k,
                    W_VAL,
                    cell_edit,
                );
                angle_cell(
                    ui,
                    &format!("angle_theta_{key:?}"),
                    &mut val.theta_0,
                    cell_edit,
                );

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::Dihedral | FfKind::Improper => {
            let improper = kind == FfKind::Improper;
            let id = if improper {
                "ff_improper"
            } else {
                "ff_dihedral"
            };

            let map = if improper {
                &mut params.improper
            } else {
                &mut params.dihedral
            };

            // A given set of atom types may have multiple terms; one row per term.
            let mut total = 0;
            let mut rows = Vec::new();
            for (key, terms) in map.iter() {
                total += terms.len();

                if !matches(&filter, &format!("{}-{}-{}-{}", key.0, key.1, key.2, key.3)) {
                    continue;
                }

                for i in 0..terms.len() {
                    rows.push((key.clone(), i));
                }
            }
            rows.sort_unstable();

            header(
                ui,
                &[
                    ("Type 1", W_TYPE),
                    ("Type 2", W_TYPE),
                    ("Type 3", W_TYPE),
                    ("Type 4", W_TYPE),
                    ("Term", W_INT),
                    ("Divider", W_INT),
                    ("V (kcal/mol)", W_VAL),
                    ("Phase (°)", W_VAL),
                    ("Periodicity", W_INT),
                    ("Comment", W_COMMENT),
                ],
                rows.len(),
                total,
            );

            table(ui, id, &rows, |ui, (key, i)| {
                let Some(val) = map.get_mut(key).and_then(|terms| terms.get_mut(*i)) else {
                    return;
                };

                label_cell(ui, &key.0, W_TYPE, Color32::WHITE);
                label_cell(ui, &key.1, W_TYPE, Color32::WHITE);
                label_cell(ui, &key.2, W_TYPE, Color32::WHITE);
                label_cell(ui, &key.3, W_TYPE, Color32::WHITE);
                label_cell(ui, &i.to_string(), W_INT, Color32::GRAY);

                num_cell(
                    ui,
                    &format!("{id}_div_{key:?}_{i}"),
                    &mut val.divider,
                    W_INT,
                    cell_edit,
                );
                num_cell(
                    ui,
                    &format!("{id}_barrier_{key:?}_{i}"),
                    &mut val.barrier_height,
                    W_VAL,
                    cell_edit,
                );
                angle_cell(
                    ui,
                    &format!("{id}_phase_{key:?}_{i}"),
                    &mut val.phase,
                    cell_edit,
                );
                num_cell(
                    ui,
                    &format!("{id}_period_{key:?}_{i}"),
                    &mut val.periodicity,
                    W_INT,
                    cell_edit,
                );

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::Mass => {
            let mut keys = Vec::new();
            for key in params.mass.keys() {
                if matches(&filter, key) {
                    keys.push(key.clone());
                }
            }
            keys.sort_unstable();

            header(
                ui,
                &[
                    ("Type", W_TYPE),
                    ("Mass (Da)", W_VAL),
                    ("Comment", W_COMMENT),
                ],
                keys.len(),
                params.mass.len(),
            );

            table(ui, "ff_mass", &keys, |ui, key| {
                let Some(val) = params.mass.get_mut(key) else {
                    return;
                };

                label_cell(ui, key, W_TYPE, Color32::WHITE);
                num_cell(ui, &format!("mass_{key}"), &mut val.mass, W_VAL, cell_edit);

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::LennardJones => {
            let mut keys = Vec::new();
            for key in params.lennard_jones.keys() {
                if matches(&filter, key) {
                    keys.push(key.clone());
                }
            }
            keys.sort_unstable();

            header(
                ui,
                &[("Type", W_TYPE), ("σ (Å)", W_VAL), ("ε (kcal/mol)", W_VAL)],
                keys.len(),
                params.lennard_jones.len(),
            );

            table(ui, "ff_lj", &keys, |ui, key| {
                let Some(val) = params.lennard_jones.get_mut(key) else {
                    return;
                };

                label_cell(ui, key, W_TYPE, Color32::WHITE);
                num_cell(
                    ui,
                    &format!("lj_sigma_{key}"),
                    &mut val.sigma,
                    W_VAL,
                    cell_edit,
                );
                num_cell(ui, &format!("lj_eps_{key}"), &mut val.eps, W_VAL, cell_edit);
            });
        }
    }
}

/// Partial charge, and FF type by type-in-residue, for amino acids.
fn charge_table_prot(map: &mut ProtFfChargeMap, id: &str, st: &mut FfParamsUi, ui: &mut Ui) {
    let filter = st.filter.to_lowercase();
    let cell_edit = &mut st.cell_edit;

    let mut total = 0;
    let mut rows = Vec::new();

    for (aa, charges) in map.iter() {
        total += charges.len();
        let name = aa_name(aa);

        for (i, c) in charges.iter().enumerate() {
            let text = format!("{name} {} {}", c.type_in_res, c.ff_type);
            if matches(&filter, &text) {
                rows.push((name.clone(), *aa, i));
            }
        }
    }
    rows.sort_unstable_by(|a, b| (&a.0, a.2).cmp(&(&b.0, b.2)));

    header(
        ui,
        &[
            ("Residue", W_TYPE),
            ("Type in res", W_RES),
            ("FF type", W_TYPE),
            ("Charge (e)", W_VAL),
        ],
        rows.len(),
        total,
    );

    table(ui, id, &rows, |ui, (name, aa, i)| {
        let Some(val) = map.get_mut(aa).and_then(|charges| charges.get_mut(*i)) else {
            return;
        };

        label_cell(ui, name, W_TYPE, Color32::WHITE);
        label_cell(ui, &val.type_in_res.to_string(), W_RES, Color32::GRAY);

        text_cell(
            ui,
            &format!("{id}_ff_{name}_{i}"),
            &mut val.ff_type,
            W_TYPE,
            cell_edit,
        );
        num_cell(
            ui,
            &format!("{id}_q_{name}_{i}"),
            &mut val.charge,
            W_VAL,
            cell_edit,
        );
    });
}

/// Partial charge, and FF type by type-in-residue. For lipids, and nucleic acids.
fn charge_table<K: Clone + Eq + Hash>(
    map: &mut HashMap<K, Vec<ChargeParams>>,
    key_name: impl Fn(&K) -> String,
    id: &str,
    st: &mut FfParamsUi,
    ui: &mut Ui,
) {
    let filter = st.filter.to_lowercase();
    let cell_edit = &mut st.cell_edit;

    let mut total = 0;
    let mut rows = Vec::new();

    for (key, charges) in map.iter() {
        total += charges.len();
        let name = key_name(key);

        for (i, c) in charges.iter().enumerate() {
            let text = format!("{name} {} {}", c.type_in_res, c.ff_type);
            if matches(&filter, &text) {
                rows.push((name.clone(), key.clone(), i));
            }
        }
    }
    rows.sort_unstable_by(|a, b| (&a.0, a.2).cmp(&(&b.0, b.2)));

    header(
        ui,
        &[
            ("Residue", W_RES),
            ("Type in res", W_RES),
            ("FF type", W_TYPE),
            ("Charge (e)", W_VAL),
        ],
        rows.len(),
        total,
    );

    table(ui, id, &rows, |ui, (name, key, i)| {
        let Some(val) = map.get_mut(key).and_then(|charges| charges.get_mut(*i)) else {
            return;
        };

        label_cell(ui, name, W_RES, Color32::WHITE);

        text_cell(
            ui,
            &format!("{id}_res_{name}_{i}"),
            &mut val.type_in_res,
            W_RES,
            cell_edit,
        );
        text_cell(
            ui,
            &format!("{id}_ff_{name}_{i}"),
            &mut val.ff_type,
            W_TYPE,
            cell_edit,
        );
        num_cell(
            ui,
            &format!("{id}_q_{name}_{i}"),
            &mut val.charge,
            W_VAL,
            cell_edit,
        );
    });
}

fn aa_name(aa: &AminoAcidGeneral) -> String {
    match aa {
        AminoAcidGeneral::Standard(a) => a.to_string(),
        AminoAcidGeneral::Variant(v) => v.to_string(),
    }
}

fn not_loaded(ui: &mut Ui) {
    label!(ui, "These parameters are not loaded.", COLOR_HIGHLIGHT);
}

/// View and edit the general force field parameters. E.g. Amber's parm19, gaff2, and amino19.
pub(in crate::ui) fn ff_param_editor(state: &mut State, ui: &mut Ui) {
    // A popup's area is only `spacing.default_area_size` (400px) tall, and the table's scroll area
    // can't exceed the height available to it: without this, `TABLE_HEIGHT` has no effect. Give it
    // everything between the top of the popup and the bottom of the window.
    let avail_h = ui.ctx().content_rect().bottom() - ui.min_rect().top() - SCREEN_MARGIN;
    ui.set_max_height(avail_h);

    // Disjoint borrows; the tables below edit the params in place.
    let params = &mut state.ff_param_set;
    let st = &mut state.ui.ff_params;

    ui.horizontal(|ui| {
        label!(ui, "Force field parameters", Color32::WHITE);
        ui.add_space(COL_SPACING);

        ui.label("Filter:");
        ui.add(TextEdit::singleline(&mut st.filter).desired_width(120.))
            .on_hover_text("Show only rows whose atom or residue types contain this text.");

        ui.add_space(COL_SPACING);

        close_btn(ui, &mut state.ui.popup.ff_params);
    });

    label!(
        ui,
        "Edits apply immediately to the loaded parameters. They are not saved to disk.",
        Color32::GRAY
    );

    ui.add_space(ROW_SPACING);

    // Top level: which field of the param set to view.
    ui.horizontal_wrapped(|ui| {
        for cat in CATS {
            if !cat.present(params) {
                label_cell(ui, cat.name(), W_RES, Color32::DARK_GRAY);
                continue;
            }

            let color = if st.cat == cat {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if button!(ui, cat.name(), color, cat.hover()).clicked() {
                st.cat = cat;
                st.cell_edit = None;
            }
        }
    });

    ui.add_space(ROW_SPACING);

    match st.cat {
        FfCat::Peptide
        | FfCat::SmallMol
        | FfCat::Dna
        | FfCat::Rna
        | FfCat::Lipids
        | FfCat::Carbohydrates => {
            let ff = match st.cat {
                FfCat::Peptide => &mut params.peptide,
                FfCat::SmallMol => &mut params.small_mol,
                FfCat::Dna => &mut params.dna,
                FfCat::Rna => &mut params.rna,
                FfCat::Lipids => &mut params.lipids,
                _ => &mut params.carbohydrates,
            };

            let Some(ff) = ff else {
                not_loaded(ui);
                return;
            };

            // Sub-selection: which field of the `ForceFieldParams`.
            ui.horizontal_wrapped(|ui| {
                for kind in KINDS {
                    let color = if st.kind == kind {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if button!(ui, kind.name(), color, kind.hover()).clicked() {
                        st.kind = kind;
                        st.cell_edit = None;
                    }
                }
            });

            ui.add_space(ROW_SPACING);

            ff_params_table(ff, st.kind, st, ui);
        }
        FfCat::PeptideQ => {
            let Some(q_map) = &mut params.peptide_ff_q_map else {
                not_loaded(ui);
                return;
            };

            // Sub-selection: internal residues, or one of the terminus variants.
            ui.horizontal_wrapped(|ui| {
                for sel in PROT_MAPS {
                    let color = if st.prot_map == sel {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if button!(
                        ui,
                        sel.name(),
                        color,
                        "Select this charge map to view and edit."
                    )
                    .clicked()
                    {
                        st.prot_map = sel;
                        st.cell_edit = None;
                    }
                }
            });

            ui.add_space(ROW_SPACING);

            let (map, id) = match st.prot_map {
                ProtMapSel::Internal => (&mut q_map.internal, "q_internal"),
                ProtMapSel::NTerminus => (&mut q_map.n_terminus, "q_n_term"),
                ProtMapSel::CTerminus => (&mut q_map.c_terminus, "q_c_term"),
            };

            charge_table_prot(map, id, st, ui);
        }
        FfCat::LipidQ => {
            let Some(map) = &mut params.lipid_ff_q_map else {
                not_loaded(ui);
                return;
            };

            charge_table(map, |k| k.to_string(), "q_lipid", st, ui);
        }
        FfCat::DnaQ | FfCat::RnaQ => {
            let (map, id) = if st.cat == FfCat::DnaQ {
                (&mut params.dna_ff_q_map, "q_dna")
            } else {
                (&mut params.rna_ff_q_map, "q_rna")
            };

            let Some(map) = map else {
                not_loaded(ui);
                return;
            };

            charge_table(map, |k| format!("{} {:?}", k.nt, k.end), id, st, ui);
        }
    }
}
