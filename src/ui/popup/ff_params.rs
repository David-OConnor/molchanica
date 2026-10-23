//! A viewer and editor for the general force field parameters loaded at startup. Implemented as a
//! popup. For Amber (`FfParamSet`), e.g. from parm19, ff19SB, gaff2, lipid21, and OL24. For
//! OPLS-AA/L, GROMACS' atom type and bonded tables, and residue templates for proteins, and Foyer's
//! tables for small molecules.
//!
//! The tables are view-only until editing is enabled. Values typed into them are applied to the
//! loaded parameters immediately. See `file_io::ff_params` for loading, saving, and restoring them.

use std::{collections::HashMap, fmt::Display, hash::Hash, path::Path, str::FromStr};

use bio_files::{
    gromacs::{
        GromacsFfParams, Rtp,
        ff_params::{ItpDefine, ItpTypeEntry},
    },
    md_params::{ChargeParams, ForceFieldParams},
};
use dynamics::{
    ForceFieldFamily,
    params::{
        amber::{AmberSet, ProtFfChargeMap},
        opls::OplsSet,
    },
};
use egui::{Color32, Label, RichText, ScrollArea, TextEdit, Ui};
use egui_file_dialog::{FileDialog, FileDialogConfig};
use na_seq::AminoAcidGeneral;

use crate::{
    button,
    file_io::ff_params::{self as ff_io, ParamsFileOp, ParamsTarget},
    force_fields::family_name,
    label,
    state::State,
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        misc::{selector_box, selector_option},
    },
    util::{handle_err, handle_success},
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

/// Read-only values.
const COLOR_VAL: Color32 = Color32::LIGHT_GRAY;

/// Offset from the position shared by the other popups: this one is tall, so we start it higher
/// and further left to keep it on screen.
pub(in crate::ui) const POPUP_OFFSET: (f32, f32) = (-100., -200.);

/// Which set of parameters is being viewed. For Amber, a field of `AmberSet`. OPLS has the
/// peptide, small molecule, and peptide charge (residue template) sets.
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
    pub(crate) fn name(self) -> &'static str {
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

    fn hover(self, family: ForceFieldFamily) -> &'static str {
        if family == ForceFieldFamily::Opls {
            return match self {
                Self::Peptide => {
                    "OPLS-AA/L atom types (Lennard-Jones, mass, charge), and bonded params by bond \
                    type, for proteins. From GROMACS' ffnonbonded.itp, and ffbonded.itp."
                }
                Self::SmallMol => {
                    "OPLS-AA atom types, and bonded params by class, for small organic molecules. \
                    From Foyer. Parameters cached from LigParGen don't use these."
                }
                Self::PeptideQ => {
                    "OPLS-AA/L residue templates: FF type, and partial charge by atom name, for \
                    amino acids. From aminoacids.rtp."
                }
                _ => "Not part of OPLS-AA/L.",
            };
        }

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
    fn present_amber(self, set: &AmberSet) -> bool {
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

    /// Is this set part of OPLS? Unlike Amber's, these are always loaded.
    fn present_opls(self) -> bool {
        matches!(self, Self::Peptide | Self::SmallMol | Self::PeptideQ)
    }

    /// Partial charges and FF types by residue, vice bonded and Lennard-Jones params.
    pub(crate) fn is_charges(self) -> bool {
        matches!(
            self,
            Self::PeptideQ | Self::LipidQ | Self::DnaQ | Self::RnaQ
        )
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

/// Which table of a `GromacsFfParams` (e.g. OPLS) is being viewed.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum GmxKind {
    #[default]
    AtomTypes,
    Bond,
    Angle,
    /// Ryckaert-Bellemans, and Fourier dihedral types.
    DihedralRb,
    /// Periodic propers and impropers, and harmonic impropers.
    DihedralPeriodic,
    Constraint,
    Macros,
}

const GMX_KINDS: [GmxKind; 7] = [
    GmxKind::AtomTypes,
    GmxKind::Bond,
    GmxKind::Angle,
    GmxKind::DihedralRb,
    GmxKind::DihedralPeriodic,
    GmxKind::Constraint,
    GmxKind::Macros,
];

/// GROMACS dihedral functional forms with 6, or 4 cosine-power or Fourier coefficients.
fn is_rb(funct: u8) -> bool {
    matches!(funct, 3 | 5)
}

impl GmxKind {
    fn name(self) -> &'static str {
        match self {
            Self::AtomTypes => "Atom types",
            Self::Bond => "Bond",
            Self::Angle => "Angle",
            Self::DihedralRb => "Dihedral (RB)",
            Self::DihedralPeriodic => "Dihedral (periodic)",
            Self::Constraint => "Constraint",
            Self::Macros => "Macros",
        }
    }

    fn hover(self) -> &'static str {
        match self {
            Self::AtomTypes => {
                "Mass, charge, and Lennard-Jones params by atom type, and the bond type each \
                uses to look up bonded params."
            }
            Self::Bond => "Bond stretching, by bond type. Harmonic for funct 1.",
            Self::Angle => "Angle bending, by bond type. Harmonic for funct 1.",
            Self::DihedralRb => {
                "Ryckaert-Bellemans (funct 3) dihedrals: V = Σ Cₙ cos(ψ)ⁿ, with ψ = φ - 180°. \
                OPLS' propers use these. (Fourier for funct 5.)"
            }
            Self::DihedralPeriodic => {
                "Periodic propers (funct 1, 9), and impropers (funct 4): V = k (1 + cos(n φ - φₛ)). \
                Harmonic impropers for funct 2."
            }
            Self::Constraint => "Constraint lengths, by bond type.",
            Self::Macros => {
                "Named parameters, e.g. OPLS' residue-specific dihedrals, and its impropers. \
                Residue templates reference these by name."
            }
        }
    }

    fn count(self, tables: &GromacsFfParams) -> usize {
        match self {
            Self::AtomTypes => tables.atom_types.len(),
            Self::Bond => tables.bond_types.len(),
            Self::Angle => tables.angle_types.len(),
            Self::DihedralRb => tables
                .dihedral_types
                .iter()
                .filter(|d| is_rb(d.funct))
                .count(),
            Self::DihedralPeriodic => tables
                .dihedral_types
                .iter()
                .filter(|d| !is_rb(d.funct))
                .count(),
            Self::Constraint => tables.constraint_types.len(),
            Self::Macros => tables.defines.len(),
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
    pub(crate) fn name(self) -> &'static str {
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
    pub gmx_kind: GmxKind,
    pub prot_map: ProtMapSel,
    pub filter: String,
    /// If false, the tables are view-only.
    pub editing: bool,
    /// The cell currently being typed into: (cell id, raw text). We hold the raw text so
    /// partially-typed values, e.g. "-", or "1.", survive a round trip through the parsed value.
    pub cell_edit: Option<(String, String)>,
    /// OPLS values changed. We re-assign molecules' types and charges once the cell being typed
    /// into loses focus, vice on every keystroke.
    pub refresh_pending: bool,
    /// Where the last parameter file was loaded from, or saved to. We start the next dialog there.
    pub last_dir: Option<std::path::PathBuf>,
    /// The result of the last load, save, or restore: (message, is error).
    pub status: Option<(String, bool)>,
}

/// Lays out the value cells of a table: text fields while editing is enabled, and labels
/// otherwise.
struct Cells<'a> {
    edit: &'a mut Option<(String, String)>,
    editable: bool,
    /// Set if any value changed.
    changed: bool,
}

impl Cells<'_> {
    /// A text field. Returns the new text, if it changed this frame.
    fn input(&mut self, ui: &mut Ui, id: &str, w: f32, display: String) -> Option<String> {
        if !self.editable {
            label_cell(ui, &display, w, COLOR_VAL);
            return None;
        }

        let mut text = match self.edit {
            Some((id_active, t)) if id_active == id => t.clone(),
            _ => display,
        };

        let resp = ui.add_sized([w, CELL_H], TextEdit::singleline(&mut text).id_salt(id));

        if resp.changed() {
            *self.edit = Some((id.to_owned(), text.clone()));
            return Some(text);
        }

        // Return to the canonical formatting once the user moves on.
        if resp.lost_focus()
            && self
                .edit
                .as_ref()
                .is_some_and(|(id_active, _)| id_active == id)
        {
            *self.edit = None;
        }

        None
    }

    /// A cell for any value displayable as, and parsable from text. E.g. floats and ints.
    fn num<T: Display + FromStr>(&mut self, ui: &mut Ui, id: &str, val: &mut T, w: f32) {
        if let Some(text) = self.input(ui, id, w, val.to_string())
            && let Ok(v) = text.parse::<T>()
        {
            *val = v;
            self.changed = true;
        }
    }

    /// Angles are stored in radians, but displayed and edited in degrees, as in the Amber
    /// parameter files.
    fn angle(&mut self, ui: &mut Ui, id: &str, val: &mut f32) {
        let display = format!("{:.3}", val.to_degrees());

        if let Some(text) = self.input(ui, id, W_VAL, display)
            && let Ok(v) = text.parse::<f32>()
        {
            *val = v.to_radians();
            self.changed = true;
        }
    }

    fn text(&mut self, ui: &mut Ui, id: &str, val: &mut String, w: f32) {
        if let Some(text) = self.input(ui, id, w, val.clone()) {
            *val = text;
            self.changed = true;
        }
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
fn ff_params_table(
    params: &mut ForceFieldParams,
    kind: FfKind,
    filter: &str,
    cells: &mut Cells,
    ui: &mut Ui,
) {
    // We clone the key of each row out of the map, so we can take a mutable borrow of the value
    // while laying its row out.
    match kind {
        FfKind::Bond => {
            let mut keys = Vec::new();
            for key in params.bond.keys() {
                if matches(filter, &format!("{}-{}", key.0, key.1)) {
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

                cells.num(ui, &format!("bond_k_{key:?}"), &mut val.k_b, W_VAL);
                cells.num(ui, &format!("bond_r_{key:?}"), &mut val.r_0, W_VAL);

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::Angle => {
            let mut keys = Vec::new();
            for key in params.angle.keys() {
                if matches(filter, &format!("{}-{}-{}", key.0, key.1, key.2)) {
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

                cells.num(ui, &format!("angle_k_{key:?}"), &mut val.k, W_VAL);
                cells.angle(ui, &format!("angle_theta_{key:?}"), &mut val.theta_0);

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

                if !matches(filter, &format!("{}-{}-{}-{}", key.0, key.1, key.2, key.3)) {
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

                cells.num(
                    ui,
                    &format!("{id}_div_{key:?}_{i}"),
                    &mut val.divider,
                    W_INT,
                );
                cells.num(
                    ui,
                    &format!("{id}_barrier_{key:?}_{i}"),
                    &mut val.barrier_height,
                    W_VAL,
                );
                cells.angle(ui, &format!("{id}_phase_{key:?}_{i}"), &mut val.phase);
                cells.num(
                    ui,
                    &format!("{id}_period_{key:?}_{i}"),
                    &mut val.periodicity,
                    W_INT,
                );

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::Mass => {
            let mut keys = Vec::new();
            for key in params.mass.keys() {
                if matches(filter, key) {
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
                cells.num(ui, &format!("mass_{key}"), &mut val.mass, W_VAL);

                comment_cell(ui, &val.comment);
            });
        }
        FfKind::LennardJones => {
            let mut keys = Vec::new();
            for key in params.lennard_jones.keys() {
                if matches(filter, key) {
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
                cells.num(ui, &format!("lj_sigma_{key}"), &mut val.sigma, W_VAL);
                cells.num(ui, &format!("lj_eps_{key}"), &mut val.eps, W_VAL);
            });
        }
    }
}

/// Partial charge, and FF type by type-in-residue, for amino acids.
fn charge_table_prot(
    map: &mut ProtFfChargeMap,
    id: &str,
    filter: &str,
    cells: &mut Cells,
    ui: &mut Ui,
) {
    let mut total = 0;
    let mut rows = Vec::new();

    for (aa, charges) in map.iter() {
        total += charges.len();
        let name = aa_name(aa);

        for (i, c) in charges.iter().enumerate() {
            let text = format!("{name} {} {}", c.type_in_res, c.ff_type);
            if matches(filter, &text) {
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

        cells.text(ui, &format!("{id}_ff_{name}_{i}"), &mut val.ff_type, W_TYPE);
        cells.num(ui, &format!("{id}_q_{name}_{i}"), &mut val.charge, W_VAL);
    });
}

/// Partial charge, and FF type by type-in-residue. For lipids, and nucleic acids.
fn charge_table<K: Clone + Eq + Hash>(
    map: &mut HashMap<K, Vec<ChargeParams>>,
    key_name: impl Fn(&K) -> String,
    id: &str,
    filter: &str,
    cells: &mut Cells,
    ui: &mut Ui,
) {
    let mut total = 0;
    let mut rows = Vec::new();

    for (key, charges) in map.iter() {
        total += charges.len();
        let name = key_name(key);

        for (i, c) in charges.iter().enumerate() {
            let text = format!("{name} {} {}", c.type_in_res, c.ff_type);
            if matches(filter, &text) {
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

        cells.text(
            ui,
            &format!("{id}_res_{name}_{i}"),
            &mut val.type_in_res,
            W_RES,
        );
        cells.text(ui, &format!("{id}_ff_{name}_{i}"), &mut val.ff_type, W_TYPE);
        cells.num(ui, &format!("{id}_q_{name}_{i}"), &mut val.charge, W_VAL);
    });
}

/// A `[ *types ]` table, e.g. bond types. `rows` are indices into `entries`, which we keep in file
/// order: GROMACS' dihedral matching depends on it. Entries with fewer parameters than `params`
/// leave the rest blank.
fn type_entry_table(
    entries: &mut [ItpTypeEntry],
    rows: &[usize],
    total: usize,
    n_types: usize,
    params: &[(&str, f32)],
    id: &str,
    cells: &mut Cells,
    ui: &mut Ui,
) {
    let type_names = ["Type 1", "Type 2", "Type 3", "Type 4"];

    let mut cols: Vec<_> = type_names[..n_types].iter().map(|t| (*t, W_TYPE)).collect();
    cols.push(("Funct", W_INT));
    cols.extend_from_slice(params);
    cols.push(("Comment", W_COMMENT));

    header(ui, &cols, rows.len(), total);

    table(ui, id, rows, |ui, &i| {
        let Some(entry) = entries.get_mut(i) else {
            return;
        };

        for j in 0..n_types {
            let t = entry.types.get(j).map(String::as_str).unwrap_or_default();
            label_cell(ui, t, W_TYPE, Color32::WHITE);
        }
        label_cell(ui, &entry.funct.to_string(), W_INT, Color32::GRAY);

        for (j, (_, w)) in params.iter().enumerate() {
            match entry.params.get_mut(j) {
                Some(v) => cells.num(ui, &format!("{id}_{i}_{j}"), v, *w),
                None => label_cell(ui, "", *w, COLOR_VAL),
            }
        }

        comment_cell(ui, &entry.comment);
    });
}

/// The GROMACS tables, e.g. OPLS': one per field, selected by `kind`.
fn gromacs_table(
    tables: &mut GromacsFfParams,
    kind: GmxKind,
    id: &str,
    filter: &str,
    cells: &mut Cells,
    ui: &mut Ui,
) {
    let matching = |entries: &[ItpTypeEntry], keep: &dyn Fn(&ItpTypeEntry) -> bool| {
        let mut rows = Vec::new();
        let mut total = 0;

        for (i, e) in entries.iter().enumerate() {
            if !keep(e) {
                continue;
            }
            total += 1;

            if matches(filter, &e.types.join("-")) {
                rows.push(i);
            }
        }
        (rows, total)
    };

    match kind {
        GmxKind::AtomTypes => {
            let types = &mut tables.atom_types;

            let rows: Vec<_> = (0..types.len())
                .filter(|&i| {
                    let t = &types[i];
                    matches(filter, &format!("{} {}", t.name, t.bond_type))
                })
                .collect();

            header(
                ui,
                &[
                    ("Name", W_RES),
                    ("Bond type", W_TYPE),
                    ("At. num", W_INT),
                    ("Mass (amu)", W_VAL),
                    ("Charge (e)", W_VAL),
                    ("Ptype", W_INT),
                    ("σ (nm)", W_VAL),
                    ("ε (kJ/mol)", W_VAL),
                    ("Comment", W_COMMENT),
                ],
                rows.len(),
                types.len(),
            );

            table(ui, &format!("{id}_atomtypes"), &rows, |ui, &i| {
                let Some(t) = types.get_mut(i) else {
                    return;
                };

                label_cell(ui, &t.name, W_RES, Color32::WHITE);
                cells.text(ui, &format!("{id}_at_bt_{i}"), &mut t.bond_type, W_TYPE);

                let at_num = t.atomic_number.map(|n| n.to_string()).unwrap_or_default();
                label_cell(ui, &at_num, W_INT, Color32::GRAY);

                cells.num(ui, &format!("{id}_at_mass_{i}"), &mut t.mass, W_VAL);
                cells.num(ui, &format!("{id}_at_q_{i}"), &mut t.charge, W_VAL);
                label_cell(ui, &t.ptype, W_INT, Color32::GRAY);
                cells.num(ui, &format!("{id}_at_v_{i}"), &mut t.v, W_VAL);
                cells.num(ui, &format!("{id}_at_w_{i}"), &mut t.w, W_VAL);

                comment_cell(ui, &t.comment);
            });
        }
        GmxKind::Bond => {
            let (rows, total) = matching(&tables.bond_types, &|_| true);
            type_entry_table(
                &mut tables.bond_types,
                &rows,
                total,
                2,
                &[("b₀ (nm)", W_VAL), ("k (kJ/mol/nm²)", W_VAL)],
                &format!("{id}_bond"),
                cells,
                ui,
            );
        }
        GmxKind::Angle => {
            let (rows, total) = matching(&tables.angle_types, &|_| true);
            type_entry_table(
                &mut tables.angle_types,
                &rows,
                total,
                3,
                &[("θ₀ (°)", W_VAL), ("k (kJ/mol/rad²)", W_VAL)],
                &format!("{id}_angle"),
                cells,
                ui,
            );
        }
        GmxKind::DihedralRb => {
            let (rows, total) = matching(&tables.dihedral_types, &|e| is_rb(e.funct));
            type_entry_table(
                &mut tables.dihedral_types,
                &rows,
                total,
                4,
                &[
                    ("C₀ (kJ/mol)", W_VAL),
                    ("C₁", W_VAL),
                    ("C₂", W_VAL),
                    ("C₃", W_VAL),
                    ("C₄", W_VAL),
                    ("C₅", W_VAL),
                ],
                &format!("{id}_dihe_rb"),
                cells,
                ui,
            );
        }
        GmxKind::DihedralPeriodic => {
            let (rows, total) = matching(&tables.dihedral_types, &|e| !is_rb(e.funct));
            type_entry_table(
                &mut tables.dihedral_types,
                &rows,
                total,
                4,
                &[
                    ("φₛ (°)", W_VAL),
                    ("k (kJ/mol)", W_VAL),
                    ("Multiplicity", W_VAL),
                ],
                &format!("{id}_dihe_periodic"),
                cells,
                ui,
            );
        }
        GmxKind::Constraint => {
            let (rows, total) = matching(&tables.constraint_types, &|_| true);
            type_entry_table(
                &mut tables.constraint_types,
                &rows,
                total,
                2,
                &[("b₀ (nm)", W_VAL)],
                &format!("{id}_constraint"),
                cells,
                ui,
            );
        }
        GmxKind::Macros => macro_table(&mut tables.defines, id, filter, cells, ui),
    }
}

fn macro_table(defines: &mut [ItpDefine], id: &str, filter: &str, cells: &mut Cells, ui: &mut Ui) {
    let rows: Vec<_> = (0..defines.len())
        .filter(|&i| matches(filter, &defines[i].name))
        .collect();

    // Macro names are long, e.g. "dih_SER_THR_chi1_N_C_C_O".
    let w_name = 2. * W_RES + 30.;

    header(
        ui,
        &[
            ("Name", w_name),
            ("Value", W_COMMENT),
            ("Comment", W_COMMENT),
        ],
        rows.len(),
        defines.len(),
    );

    table(ui, &format!("{id}_macros"), &rows, |ui, &i| {
        let Some(d) = defines.get_mut(i) else {
            return;
        };

        label_cell(ui, &d.name, w_name, Color32::WHITE);
        cells.text(ui, &format!("{id}_def_{i}"), &mut d.value, W_COMMENT);
        comment_cell(ui, &d.comment);
    });
}

/// FF type, and partial charge by atom name, for residue templates. E.g. OPLS' amino acids.
fn template_table(rtp: &mut Rtp, filter: &str, cells: &mut Cells, ui: &mut Ui) {
    let mut total = 0;
    let mut rows = Vec::new();

    for (name, res) in &rtp.residues {
        total += res.atoms.len();

        for (i, a) in res.atoms.iter().enumerate() {
            if matches(filter, &format!("{name} {} {}", a.name, a.ff_type)) {
                rows.push((name.clone(), i));
            }
        }
    }
    rows.sort_unstable();

    header(
        ui,
        &[
            ("Residue", W_TYPE),
            ("Atom", W_TYPE),
            ("FF type", W_RES),
            ("Charge (e)", W_VAL),
        ],
        rows.len(),
        total,
    );

    table(ui, "rtp_atoms", &rows, |ui, (name, i)| {
        let Some(atom) = rtp
            .residues
            .get_mut(name)
            .and_then(|res| res.atoms.get_mut(*i))
        else {
            return;
        };

        label_cell(ui, name, W_TYPE, Color32::WHITE);
        label_cell(ui, &atom.name, W_TYPE, Color32::GRAY);

        cells.text(ui, &format!("rtp_ff_{name}_{i}"), &mut atom.ff_type, W_RES);
        cells.num(ui, &format!("rtp_q_{name}_{i}"), &mut atom.charge, W_VAL);
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

/// Prompt for a file to load parameters from, or save them to. We handle the picked file in
/// `on_file_picked`.
fn open_file_dialog(state: &mut State, op: ParamsFileOp) {
    let mut cfg = FileDialogConfig::default();
    if let Some(dir) = &state.ui.ff_params.last_dir {
        cfg.initial_directory = dir.clone();
    }

    let dialog = match op {
        ParamsFileOp::Load(target) => {
            let exts: Vec<_> = target.load_formats().iter().map(|(_, e)| *e).collect();
            let filter = format!("{} parameters", target.descrip());

            let mut dialog = FileDialog::with_config(cfg.add_file_filter_extensions(&filter, exts))
                .title(&format!("Load {filter}"))
                .default_file_filter(&filter);

            dialog.pick_file();
            dialog
        }
        ParamsFileOp::Save(target) => {
            let formats = target.formats();
            for (descrip, ext) in formats {
                cfg = cfg.add_save_extension(descrip, ext);
            }

            let (descrip, ext) = formats[0];
            let mut dialog = FileDialog::with_config(cfg)
                .title(&format!("Save {} parameters", target.descrip()))
                .default_save_extension(descrip)
                .default_file_name(&format!("{}.{ext}", target.file_stem()));

            dialog.save_file();
            dialog
        }
    };

    state.volatile.dialogs.ff_params = dialog;
    state.volatile.dialogs.ff_params_op = Some(op);
}

/// Run from `update_file_dialogs`, once a file is picked.
pub fn on_file_picked(state: &mut State, op: ParamsFileOp, path: &Path) {
    let (result, verb, target) = match op {
        ParamsFileOp::Load(target) => (ff_io::load(state, target, path), "Loaded", target),
        ParamsFileOp::Save(target) => (ff_io::save(state, target, path), "Saved", target),
    };

    state.ui.ff_params.last_dir = ff_io::parent_dir(path);
    state.ui.ff_params.cell_edit = None;

    let file = path.file_name().unwrap_or_default().to_string_lossy();
    let msg = match result {
        Ok(()) => Ok(format!("{verb} {} parameters: {file}", target.descrip())),
        Err(e) => Err(format!("Error with {} parameters: {e}", target.descrip())),
    };

    report(state, msg);
}

fn report(state: &mut State, msg: Result<String, String>) {
    match msg {
        Ok(m) => {
            state.ui.ff_params.status = Some((m.clone(), false));
            handle_success(&mut state.ui, m);
        }
        Err(m) => {
            state.ui.ff_params.status = Some((m.clone(), true));
            handle_err(&mut state.ui, m);
        }
    }
}

/// Filter, editing lock, and loading, saving, and restoring the selected set.
fn toolbar(state: &mut State, target: ParamsTarget, ui: &mut Ui) {
    let mut restore = false;
    let mut file_op = None;

    ui.horizontal_wrapped(|ui| {
        let st = &mut state.ui.ff_params;

        ui.label("Filter:");
        ui.add(TextEdit::singleline(&mut st.filter).desired_width(120.))
            .on_hover_text("Show only rows whose atom or residue types contain this text.");

        ui.add_space(COL_SPACING / 2.);

        let (text, color, hover) = if st.editing {
            (
                "Lock editing",
                COLOR_ACTIVE,
                "Editing is enabled. Click to make the tables view-only.",
            )
        } else {
            (
                "Enable editing",
                COLOR_INACTIVE,
                "The tables are view-only. Click to edit their values.",
            )
        };

        if button!(ui, text, color, hover).clicked() {
            st.editing = !st.editing;
            st.cell_edit = None;
        }

        if button!(
            ui,
            "Restore original params",
            COLOR_ACTION,
            format!(
                "Reset the {} parameters to those bundled with the dynamics library, discarding \
                edits, and loaded files. Other parameter sets are unchanged.",
                target.descrip()
            )
        )
        .clicked()
        {
            restore = true;
        }

        ui.add_space(COL_SPACING / 2.);

        let exts = |formats: &[(&str, &str)]| {
            formats
                .iter()
                .map(|(d, e)| format!("{d} (.{e})"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let effect = if target.family == ForceFieldFamily::Opls && target.cat != FfCat::PeptideQ {
            "Tables in the file replace the current ones; others are kept."
        } else if target.family == ForceFieldFamily::Amber && !target.cat.is_charges() {
            "A .dat replaces them; a .frcmod modifies them."
        } else {
            "These replace the current ones."
        };

        let load_hover = format!(
            "Load {} parameters from {}. {effect}",
            target.descrip(),
            exts(&target.load_formats())
        );

        if button!(ui, "Load", COLOR_HIGHLIGHT, load_hover).clicked() {
            file_op = Some(ParamsFileOp::Load(target));
        }

        if button!(
            ui,
            "Save",
            COLOR_HIGHLIGHT,
            format!(
                "Save the {} parameters as {}.",
                target.descrip(),
                exts(target.formats())
            )
        )
        .clicked()
        {
            file_op = Some(ParamsFileOp::Save(target));
        }
    });

    if restore {
        state.ui.ff_params.cell_edit = None;

        let msg = match ff_io::restore(state, target) {
            Ok(()) => Ok(format!(
                "Restored the original {} parameters",
                target.descrip()
            )),
            Err(e) => Err(format!(
                "Unable to restore {} parameters: {e}",
                target.descrip()
            )),
        };
        report(state, msg);
    }

    if let Some(op) = file_op {
        open_file_dialog(state, op);
    }
}

/// Selects a value from `options`. Disabled options are shown grayed out.
fn selector_row<T: PartialEq + Copy>(
    ui: &mut Ui,
    current: &mut T,
    options: &[(T, &str, &str, bool)],
) -> bool {
    let mut changed = false;

    ui.horizontal_wrapped(|ui| {
        selector_box().show(ui, |ui| {
            for &(val, name, hover, enabled) in options {
                if !enabled {
                    label_cell(ui, name, W_RES, Color32::DARK_GRAY);
                    continue;
                }

                if selector_option(ui, *current == val, name)
                    .on_hover_text(hover)
                    .clicked()
                    && *current != val
                {
                    *current = val;
                    changed = true;
                }
            }
        });
    });

    changed
}

/// Returns true if a value changed.
fn amber_tables(set: &mut AmberSet, st: &mut FfParamsUi, ui: &mut Ui) -> bool {
    let filter = st.filter.to_lowercase();
    let mut cells = Cells {
        edit: &mut st.cell_edit,
        editable: st.editing,
        changed: false,
    };

    match st.cat {
        FfCat::Peptide
        | FfCat::SmallMol
        | FfCat::Dna
        | FfCat::Rna
        | FfCat::Lipids
        | FfCat::Carbohydrates => {
            let Some(Some(ff)) = ff_io::amber_ff_mut(set, st.cat) else {
                not_loaded(ui);
                return false;
            };

            // Sub-selection: which field of the `ForceFieldParams`.
            let options: Vec<_> = KINDS
                .iter()
                .map(|&k| (k, k.name(), k.hover(), true))
                .collect();
            if selector_row(ui, &mut st.kind, &options) {
                *cells.edit = None;
            }

            ui.add_space(ROW_SPACING);

            ff_params_table(ff, st.kind, &filter, &mut cells, ui);
        }
        FfCat::PeptideQ => {
            let Some(q_map) = &mut set.peptide_ff_q_map else {
                not_loaded(ui);
                return false;
            };

            // Sub-selection: internal residues, or one of the terminus variants.
            let options: Vec<_> = PROT_MAPS
                .iter()
                .map(|&m| {
                    (
                        m,
                        m.name(),
                        "Select this charge map to view and edit.",
                        true,
                    )
                })
                .collect();
            if selector_row(ui, &mut st.prot_map, &options) {
                *cells.edit = None;
            }

            ui.add_space(ROW_SPACING);

            let (map, id) = match st.prot_map {
                ProtMapSel::Internal => (&mut q_map.internal, "q_internal"),
                ProtMapSel::NTerminus => (&mut q_map.n_terminus, "q_n_term"),
                ProtMapSel::CTerminus => (&mut q_map.c_terminus, "q_c_term"),
            };

            charge_table_prot(map, id, &filter, &mut cells, ui);
        }
        FfCat::LipidQ => {
            let Some(map) = &mut set.lipid_ff_q_map else {
                not_loaded(ui);
                return false;
            };

            charge_table(map, |k| k.to_string(), "q_lipid", &filter, &mut cells, ui);
        }
        FfCat::DnaQ | FfCat::RnaQ => {
            let (map, id) = if st.cat == FfCat::DnaQ {
                (&mut set.dna_ff_q_map, "q_dna")
            } else {
                (&mut set.rna_ff_q_map, "q_rna")
            };

            let Some(map) = map else {
                not_loaded(ui);
                return false;
            };

            charge_table(
                map,
                |k| format!("{} {:?}", k.nt, k.end),
                id,
                &filter,
                &mut cells,
                ui,
            );
        }
    }

    cells.changed
}

/// Returns true if a value changed.
fn opls_tables(set: &mut OplsSet, st: &mut FfParamsUi, ui: &mut Ui) -> bool {
    let filter = st.filter.to_lowercase();
    let mut cells = Cells {
        edit: &mut st.cell_edit,
        editable: st.editing,
        changed: false,
    };

    match st.cat {
        FfCat::Peptide | FfCat::SmallMol => {
            let (tables, id) = if st.cat == FfCat::Peptide {
                (&mut set.tables, "opls_pep")
            } else {
                (set.small_mol_tables_mut(), "opls_sm")
            };

            // Sub-selection: which table. Some are empty, e.g. Foyer has no macros.
            let options: Vec<_> = GMX_KINDS
                .iter()
                .map(|&k| (k, k.name(), k.hover(), k.count(tables) > 0))
                .collect();
            if selector_row(ui, &mut st.gmx_kind, &options) {
                *cells.edit = None;
            }

            if st.gmx_kind.count(tables) == 0 {
                st.gmx_kind = GmxKind::AtomTypes;
            }

            ui.add_space(ROW_SPACING);

            gromacs_table(tables, st.gmx_kind, id, &filter, &mut cells, ui);
        }
        FfCat::PeptideQ => template_table(&mut set.templates, &filter, &mut cells, ui),
        _ => not_loaded(ui),
    }

    cells.changed
}

/// View and edit the general force field parameters. E.g. Amber's parm19, gaff2, and amino19,
/// or OPLS-AA/L's.
pub(in crate::ui) fn ff_param_editor(state: &mut State, ui: &mut Ui) {
    // A popup's area is only `spacing.default_area_size` (400px) tall, and the table's scroll area
    // can't exceed the height available to it: without this, `TABLE_HEIGHT` has no effect. Give it
    // everything between the top of the popup and the bottom of the window.
    let avail_h = ui.ctx().content_rect().bottom() - ui.min_rect().top() - SCREEN_MARGIN;
    ui.set_max_height(avail_h);

    // The same selector as in the sidebar, and MD panel.
    ui.horizontal_wrapped(|ui| {
        ui.label("Force field:");
        crate::force_fields::selector(state, ui, "ff_params_force_field");
    });

    let family = state.force_fields.family;

    // Sets that aren't loaded are shown dimmed, but can be selected, so they can be loaded from
    // a file, or restored.
    let cats: Vec<(FfCat, bool)> = match family {
        ForceFieldFamily::Amber => {
            let Some(set) = state.ff_param_set.amber() else {
                not_loaded(ui);
                return;
            };
            CATS.iter().map(|&c| (c, c.present_amber(set))).collect()
        }
        ForceFieldFamily::Opls => CATS
            .iter()
            .filter(|c| c.present_opls())
            .map(|&c| (c, true))
            .collect(),
        f => {
            ui.label(format!(
                "Viewing and editing {} parameters isn't supported yet.",
                family_name(f)
            ));
            return;
        }
    };

    // Keep the selection valid for this family. E.g. after switching from Amber's DNA params to
    // OPLS, which has none.
    {
        let st = &mut state.ui.ff_params;
        if family == ForceFieldFamily::Opls && !st.cat.present_opls() {
            st.cat = FfCat::Peptide;
            st.cell_edit = None;
        }
    }

    ui.add_space(ROW_SPACING / 2.);

    let target = ParamsTarget {
        family,
        cat: state.ui.ff_params.cat,
        prot_map: state.ui.ff_params.prot_map,
    };
    toolbar(state, target, ui);

    {
        let st = &state.ui.ff_params;

        let (text, color) = if st.editing {
            (
                "Edits apply immediately to the loaded parameters. Save them to keep them.",
                COLOR_ACTIVE,
            )
        } else {
            ("View only. Enable editing to change values.", Color32::GRAY)
        };
        label!(ui, text, color);

        if let Some((msg, is_err)) = &st.status {
            let color = if *is_err {
                Color32::LIGHT_RED
            } else {
                Color32::GRAY
            };
            label!(ui, msg, color);
        }
    }

    ui.add_space(ROW_SPACING);

    // Top level: which set of params to view.
    {
        let st = &mut state.ui.ff_params;
        let mut changed = false;

        ui.horizontal_wrapped(|ui| {
            selector_box().show(ui, |ui| {
                for &(cat, loaded) in &cats {
                    let mut resp = ui
                        .scope(|ui| {
                            if !loaded {
                                ui.multiply_opacity(0.4);
                            }
                            selector_option(ui, st.cat == cat, cat.name())
                        })
                        .inner;

                    if !loaded {
                        resp = resp.on_hover_text("Not loaded. Load these from a file.");
                    }

                    if resp.on_hover_text(cat.hover(family)).clicked() && st.cat != cat {
                        st.cat = cat;
                        changed = true;
                    }
                }
            });
        });

        if changed {
            st.cell_edit = None;
        }
    }

    ui.add_space(ROW_SPACING);

    // Disjoint borrows; the tables below edit the params in place.
    let st = &mut state.ui.ff_params;

    match family {
        ForceFieldFamily::Amber => {
            if let Some(set) = state.ff_param_set.amber_mut() {
                amber_tables(set, st, ui);
            }
        }
        _ => {
            let Some(set) = state
                .force_fields
                .params
                .as_mut()
                .and_then(|p| p.opls_mut())
            else {
                not_loaded(ui);
                return;
            };

            if opls_tables(set, st, ui) {
                st.refresh_pending = true;
            }

            // Molecules' OPLS types and charges are assigned from these tables. (MD reads them
            // again when it starts.) Re-assign them once the user moves on from the cell.
            if st.refresh_pending && st.cell_edit.is_none() {
                st.refresh_pending = false;
                state.force_fields.invalidate();
            }
        }
    }
}
