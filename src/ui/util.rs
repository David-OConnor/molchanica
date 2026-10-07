use std::{
    fmt::Display,
    fs::File,
    io,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    slice,
    sync::mpsc,
    thread,
};

use bio_apis::{amber_geostd, chebi, pubchem::find_cids_from_search, uniprot};
use bio_files::MmCif;
use egui::{Color32, Response, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use mol_defs::{
    molecules::{MolIdent, MolType, MoleculeGeneric, PeptideIdent, small::MoleculeSmall},
    smiles::is_smiles,
};

use crate::{
    cam::{VIEW_DIR_FRONT, reset_camera},
    drawing::{
        MoleculeView,
        peptide::draw_peptide,
        wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids, draw_all_pockets},
    },
    external_tools::home_directory,
    file_io::{
        download_mols::{
            CifSource, DownloadedSmallMol, load_cif, load_sdf_chebi, load_sdf_drugbank,
            load_sdf_pdbe, load_sdf_pubchem, open_atom_coords, open_geostd2,
        },
        managed_mols::{self, ManagedMolProvider},
        save_mol_set_as_gro,
    },
    gromacs,
    md::viewer,
    mol_db::{CHEBI_DB_NAME, ParquetMolDb},
    mol_editor,
    pocket_render::PocketRender,
    prefs::OpenType,
    render::{Color, set_flashlight, set_static_light},
    state::{DbSel, OperatingMode, State},
    ui::{COLOR_ACTION, COLOR_HIGHLIGHT, misc, set_window_title},
    util::{RedrawFlags, handle_err, handle_success, parse_smiles, reset_orbit_center},
};

/// A path formatted for display: the home directory abbreviated to "~", and forward slashes on all
/// platforms, e.g. "~/Desktop/4091.mol2".
pub(in crate::ui) fn display_path(path: &Path) -> String {
    let abbreviated = home_directory()
        .and_then(|home| {
            path.strip_prefix(&home)
                .ok()
                .map(|rel| Path::new("~").join(rel))
        })
        .unwrap_or_else(|| path.to_path_buf());

    abbreviated.to_string_lossy().replace('\\', "/")
}

/// Show a folder in the platform's file browser. Shared by every "open this location" button in the
/// UI, so they all behave the same way, and so the per-platform command lives in one place.
pub(in crate::ui) fn open_dir(folder: &Path) -> io::Result<()> {
    #[cfg(target_os = "windows")]
    const OPENER: &str = "explorer.exe";
    #[cfg(target_os = "macos")]
    const OPENER: &str = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    const OPENER: &str = "xdg-open";

    Command::new(OPENER).arg(folder).spawn().map(|_| ())
}

/// Edit the optional display name. An empty field restores the generated label.
pub(in crate::ui) fn edit_mol_name(name: &Option<String>, ui: &mut Ui) -> Option<Option<String>> {
    let mut text = name.clone().unwrap_or_default();
    let mut changed = false;

    ui.horizontal(|ui| {
        crate::label!(ui, "Name:", Color32::GRAY);
        let width = ui.available_width().min(200.0);
        changed = ui
            .add_sized(
                [width, ui.spacing().interact_size.y],
                TextEdit::singleline(&mut text),
            )
            .changed();
    });

    changed.then(|| {
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    })
}

/// A molecule's identifiers, for display: small molecules and proteins each have their own kind.
#[derive(Clone, Copy)]
pub(in crate::ui) enum Idents<'a> {
    Small(&'a Vec<MolIdent>),
    Peptide(&'a Vec<PeptideIdent>),
}

impl<'a> Idents<'a> {
    /// E.g. for `MoleculeCommon::name`, which draws on small-molecule identifiers.
    pub fn small(self) -> Option<&'a Vec<MolIdent>> {
        match self {
            Self::Small(idents) => Some(idents),
            Self::Peptide(_) => None,
        }
    }
}

/// Display a molecule's identifiers and, when supplied, an editable name above them.
/// Returns only changes to the name; the caller applies them after releasing molecule borrows.
pub(in crate::ui) fn list_idents(
    name: Option<&Option<String>>,
    idents: Idents,
    path: &Option<PathBuf>,
    prefs_dir: &Path,
    ui: &mut Ui,
) -> Option<Option<String>> {
    let name_change = name.and_then(|name| edit_mol_name(name, ui));

    if let Some(p) = path {
        ui.horizontal_wrapped(|ui| {
            // Managed molecules were never saved to disk by the user; their cache path is an
            // implementation detail, so describe where they came from instead.
            crate::label!(ui, "File:", Color32::GRAY);
            if managed_mols::is_managed_path(prefs_dir, p) {
                ui.label(RichText::new("Downloaded; not saved permanently").color(Color32::WHITE))
                    .on_hover_text(p.to_string_lossy());
            } else {
                // The unabbreviated, native-separator path is available on hover.
                ui.label(RichText::new(display_path(p)).color(Color32::WHITE))
                    .on_hover_text(p.to_string_lossy());
            }

            if let Some(dir) = p.parent() {
                if ui
                    .button("Open dir")
                    .on_hover_text(
                        "Open your OS's file browser to the directory containing this file.",
                    )
                    .clicked()
                {
                    // No `handle_err` here: this is drawn while the molecule is borrowed from
                    // state, so the CLI output line is out of reach.
                    if let Err(e) = open_dir(dir) {
                        eprintln!("Error opening the folder {}: {e}", dir.display());
                    }
                }
            }
        });
    }

    match idents {
        Idents::Small(idents) => {
            for ident in idents {
                let long = matches!(
                    ident,
                    MolIdent::InchIKey(_)
                        | MolIdent::InchI(_)
                        | MolIdent::Smiles(_)
                        | MolIdent::IupacName(_)
                );
                ident_row(ident.ident_type(), ident.ident_inner(), long, ui);
            }
        }
        Idents::Peptide(idents) => {
            for ident in idents {
                ident_row(ident.ident_type(), ident.ident_inner(), false, ui);
            }
        }
    }

    name_change
}

/// One identifier, labeled with its type. `long` ones, e.g. SMILES, are drawn in a smaller font.
fn ident_row(ident_type: impl Display, ident: String, long: bool, ui: &mut Ui) {
    // Wrap long identifiers instead of expanding the containing panel.
    ui.horizontal_wrapped(|ui| {
        crate::label!(ui, format!("{ident_type}:"), Color32::GRAY);

        let mut ident_text = RichText::new(ident).color(Color32::WHITE);
        if long {
            ident_text = ident_text.font(egui::FontId::proportional(10.0));
        }

        ui.label(ident_text);
    });
}

/// Run this each frame, after all UI elements that affect it are rendered.
pub fn update_file_dialogs(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) -> io::Result<()> {
    let ctx = ui.ctx();

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    state.volatile.dialogs.screening.update(ctx);
    state.volatile.dialogs.parquet_db_load.update(ctx);
    state.volatile.dialogs.parquet_db_save.update(ctx);
    state.volatile.dialogs.parquet_mols_dir.update(ctx);
    state.volatile.dialogs.parquet_mol_file.update(ctx);
    state.volatile.dialogs.save_md.update(ctx);
    state.volatile.dialogs.save_gro.update(ctx);
    state.volatile.dialogs.save_seq.update(ctx);
    state.ui.synthesis_reactions.protocol_export.update(ctx);

    if let Some(path) = &state.volatile.dialogs.load.take_picked() {
        if let Err(e) = match state.volatile.operating_mode {
            OperatingMode::Primary => state.open_file(path, scene, engine_updates),
            OperatingMode::MolEditor => state.mol_editor.open_molecule(
                path,
                scene,
                engine_updates,
                &mut state.ui,
                state.volatile.mol_manip.mode,
            ),
            OperatingMode::ProteinEditor => unimplemented!(),
        } {
            handle_err(&mut state.ui, e.to_string());
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if let Some(path) = &state.volatile.dialogs.save.take_picked() {
        match state.volatile.operating_mode {
            OperatingMode::Primary => state.save(path)?,
            OperatingMode::MolEditor => {
                let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
                let extension = binding;

                // Deprecated, for now
                if extension == "pmp" {
                    let buf = state.mol_editor.mol.pharmacophore.to_bytes();
                    let mut file = File::create(path)?;
                    file.write_all(&buf)?;
                    println!("Saved Pharmacophore to {path:?}");
                } else {
                    mol_editor::save(state, path)?
                }
            }
            OperatingMode::ProteinEditor => (),
        }
    }

    // Perhaps deprecated in favor of using screening databases.
    // if let Some(path) = &state.volatile.dialogs.screening.take_picked() {
    // state.to_save.screening_path = Some(path.to_owned());
    // }

    if let Some(path) = &state.volatile.dialogs.parquet_db_save.take_picked() {
        match ParquetMolDb::new(path) {
            Ok(db) => {
                handle_success(
                    &mut state.ui,
                    format!("Created Parquet database at path {path:?}"),
                );

                state.volatile.parquet_dbs.push(db);
                state.volatile.parquet_db_active =
                    Some(DbSel::Loaded(state.volatile.parquet_dbs.len() - 1));

                // Record it in the open history, so it's reopened on the next launch (mirrors
                // `State::load_parquet_db`).
                state.update_history(path, OpenType::ParquetDb, None);
            }
            Err(e) => handle_err(
                &mut state.ui,
                format!("Error creating Parquet database: {e}"),
            ),
        }
    }

    if let Some(path) = &state.volatile.dialogs.parquet_db_load.take_picked() {
        state.load_parquet_db(path);
    }

    if let Some(path) = &state.volatile.dialogs.parquet_mols_dir.take_picked() {
        // The built-in DB is read-only, so only a loaded one can be populated.
        if let Some(DbSel::Loaded(i)) = state.volatile.parquet_db_active {
            let db = &mut state.volatile.parquet_dbs[i];
            match db.add_mols_from_dir(path) {
                Ok(()) => {
                    println!("Populated Parquet DB: {} molecules", db.index_meta.len());
                }
                Err(e) => {
                    eprintln!("Error populating parquet data: {e:?}")
                }
            }
        } else {
            handle_err(
                &mut state.ui,
                "Error: Missing the DB index to populate with mols".to_string(),
            );
        }
    }

    if let Some(path) = &state.volatile.dialogs.parquet_mol_file.take_picked() {
        // The built-in DB is read-only, so only a loaded one can be populated.
        if let Some(DbSel::Loaded(i)) = state.volatile.parquet_db_active {
            let db = &mut state.volatile.parquet_dbs[i];
            match db.add_mols_from_file(path) {
                Ok(()) => {
                    println!(
                        "Added mols from file. DB now has {} molecules",
                        db.index_meta.len()
                    );
                }
                Err(e) => {
                    eprintln!("Error adding mols from file: {e:?}")
                }
            }
        } else {
            handle_err(
                &mut state.ui,
                "Error: Missing the DB index to add a mol to".to_string(),
            );
        }
    }

    if let Some(path) = &state.volatile.dialogs.save_md.take_picked() {
        match gromacs::save_input_files(state, path) {
            Ok(_) => {
                handle_success(
                    &mut state.ui,
                    "Saved MD files in GROMACS format".to_string(),
                );
            }
            Err(e) => handle_err(&mut state.ui, format!("Error saving MD files: {e}")),
        }
    }

    if let Some(path) = state.volatile.dialogs.save_gro.take_picked() {
        let i = state.volatile.dialogs.save_gro_mol_set_i.take();
        if let Some(i) = i {
            let mol_sets = &state.volatile.md_local.viewer.mol_sets;
            if i < mol_sets.len() {
                match save_mol_set_as_gro(&mol_sets[i], &path) {
                    Ok(()) => handle_success(
                        &mut state.ui,
                        format!(
                            "Saved mol set as GRO: {:?}",
                            path.file_name().unwrap_or_default()
                        ),
                    ),
                    Err(e) => handle_err(&mut state.ui, format!("Error saving GRO: {e}")),
                }
            }
        }
    }

    if let Some(path) = state.volatile.dialogs.save_seq.take_picked()
        && let Some(i) = state.volatile.dialogs.save_seq_i.take()
        && let Err(e) = state.save_sequence(i, &path)
    {
        handle_err(&mut state.ui, format!("Error saving the sequence: {e}"));
    }

    Ok(())
}

pub fn handle_redraw(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    reset_cam: bool,
    updates: &mut EngineUpdates,
) {
    if state.volatile.md_local.draw_md_mols
        && (redraw.peptide || redraw.ligand || redraw.lipid || redraw.na)
    {
        viewer::draw_mols(state, scene, updates);

        *redraw = Default::default();
        return;
    }

    if redraw.peptide {
        draw_peptide(state, scene, updates);

        if let Some(mol) = state
            .peptide_for_tools_i()
            .and_then(|i| state.peptides.get(i))
        {
            set_window_title(
                mol.common.name.as_deref().unwrap_or(&mol.common.ident),
                scene,
            );
        }

        // For docking light, but may be overkill here.
        if state.active_mol().is_some() {
            updates.lighting = true;
        }
    }

    if redraw.ligand {
        match state.volatile.operating_mode {
            OperatingMode::Primary => {
                draw_all_ligs(state, scene, updates);
            }
            OperatingMode::MolEditor => mol_editor::redraw(
                &mut scene.entities,
                &state.mol_editor,
                &state.ui,
                state.volatile.mol_manip.mode,
                updates,
            ),
            OperatingMode::ProteinEditor => unimplemented!(),
        }
    }

    if redraw.na {
        draw_all_nucleic_acids(state, scene, updates);
    }

    if redraw.lipid {
        draw_all_lipids(state, scene, updates);
    }

    if redraw.pocket {
        draw_all_pockets(state, scene, updates);
    }

    // Perform cleanup.
    if reset_cam {
        reset_camera(state, scene, updates, VIEW_DIR_FRONT);
    }

    *redraw = Default::default();
}

/// Handles the case of opening a ligand remotely using the text input.
pub fn open_lig_from_input(
    state: &mut State,
    mol: MoleculeSmall,
    path: Option<&Path>,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.load_mol_to_state(MoleculeGeneric::Small(mol), scene, engine_updates, path);

    state.ui.db_input = String::new();
}

fn report_cache_result(state: &mut State, result: io::Result<PathBuf>) -> Option<PathBuf> {
    match result {
        Ok(path) => Some(path),
        Err(error) => {
            handle_err(
                &mut state.ui,
                format!("Could not save a restorable copy of this molecule: {error}"),
            );
            None
        }
    }
}

fn cache_sdf_source(
    state: &mut State,
    provider: ManagedMolProvider,
    key: &str,
    query: &str,
    source_text: &str,
) -> Option<PathBuf> {
    let result = managed_mols::store_text(
        &state.volatile.prefs_dir,
        provider,
        key,
        query,
        "sdf",
        source_text,
    );
    report_cache_result(state, result)
}

/// Finish session restoration after all worker results have been applied. Expensive entity rebuilds
/// are deliberately batched here instead of running once for every restored molecule.
pub(crate) fn finish_session_restore(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    // todo: Workaround to allow us to apply params to the ligand once it's loaded. Unfortunate we have
    // todo to double-load prefs.
    {
        // A default active small molecule.
        if !state.ligands.is_empty() {
            state.volatile.active_mol = Some((MolType::Ligand, 0));
        }
    }

    if let Some(pep) = state.peptide_for_tools() {
        set_static_light(
            scene,
            pep.common.centroid().into(),
            pep.common.bounding_radius() as f32,
        );
    } else if !state.ligands.is_empty() {
        let lig = &state.ligands[0];
        set_static_light(
            scene,
            lig.common.centroid().into(),
            3., // todo good enough?
        );

        //     let posit = state.to_save.per_mol[&mol.common.ident]
        //         .docking_site
        //         .site_center;
        //     // state.update_docking_site(posit);
    }

    // This updates the mesh and spheres after the initial prefs load, which may
    // have altered their posits. This prevents a visual jump upon the first re-render of pockets,
    // as the mesh moves to the correct location.
    let standalone_pocket_count = state.pockets.len();
    for (i, pocket) in state.pockets.iter_mut().enumerate() {
        pocket.mesh_i_rel = i;
        pocket.reset_post_manip(&mut scene.meshes, state.ui.mesh_coloring, updates);
    }
    // Same treatment for pockets embedded in ligand pharmacophores.
    for (lig_i, lig) in state.ligands.iter_mut().enumerate() {
        if let Some(pocket) = &mut lig.pharmacophore.pocket {
            pocket.mesh_i_rel = standalone_pocket_count + lig_i;
            pocket.reset_post_manip(&mut scene.meshes, state.ui.mesh_coloring, updates);
        }
    }

    reset_orbit_center(state, scene);

    reset_camera(state, scene, updates, VIEW_DIR_FRONT);

    draw_peptide(state, scene, updates);
    draw_all_ligs(state, scene, updates);
    draw_all_nucleic_acids(state, scene, updates);
    draw_all_lipids(state, scene, updates);
    draw_all_pockets(state, scene, updates);

    set_flashlight(scene);
    updates.lighting = true;
}

/// An assistant to make a colored label.
#[macro_export]
macro_rules! label {
    ($ui:expr, $text:expr, $color:expr) => {
        $ui.label(egui::RichText::new($text).color($color))
    };
}

/// An assistant to make a colored button.
#[macro_export]
macro_rules! button {
    ($ui:expr, $text:expr, $color:expr, $hover_text:expr) => {
        $ui.button(egui::RichText::new($text).color($color))
            .on_hover_text($hover_text)
    };
}

pub fn color_egui_from_f32(c: Color) -> Color32 {
    let (r, g, b) = c;
    Color32::from_rgb((r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8)
}

/// The most matches from the built-in database we'll offer as buttons at once. The query bar is a
/// single row, so a long list of them would push the remote-lookup buttons off screen.
const COMMON_DB_RESULTS_MAX: usize = 4;

/// Shortest query the Enter key acts on; below this it's ignored, so a one- or two-character input
/// isn't fired off at a database. The caller applies this when deciding whether Enter was pressed;
/// `query` applies it again to decide which button to highlight as the Enter target.
pub(in crate::ui) const QUERY_ENTER_LEN_MIN: usize = 3;

/// What the built-in database made of a query; see `query_common_db`.
enum CommonDbOutcome {
    /// A molecule was loaded from it. The caller must not also run a remote lookup.
    Loaded,
    /// Matches were shown but none chosen yet. Enter belongs to the top one, so the remote lookups
    /// below are reachable only by clicking their buttons.
    Matched,
    /// Nothing matched; the remote lookups own this query, including its Enter key.
    NoMatch,
}

/// Search the built-in ChEBI molecule database (`State::chebi_mol_db`) for the query text, matching
/// on CID, SMILES, or PubChem title, and draw a load button for each match. The top match is
/// highlighted: it's what Enter will load.
///
/// Enter loads the best match, which is why this runs ahead of the remote lookups in `query`: a
/// molecule we already have is always preferable to a network round trip. `ParquetMolDb::search`
/// ranks exact CID and title matches first.
// Temporarily unused: the caller's ChEBI integration is disabled because that DB is 2D-only. Kept
// (with its ranking wired to the shared `search`) so it can be switched back on with a 3D source.
#[allow(dead_code)]
fn query_common_db(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    inp: &str,
    enter_pressed: bool,
    // Whether Enter acts on this query at all; only affects which button is highlighted.
    enter_live: bool,
) -> CommonDbOutcome {
    let Some(db) = &state.chebi_mol_db else {
        return CommonDbOutcome::NoMatch;
    };

    // Gathered up front: loading borrows `state` mutably, and the search borrows the DB inside it.
    // (SMILES key, display name.)
    let hits: Vec<(String, String)> = db
        .search(inp, COMMON_DB_RESULTS_MAX)
        .iter()
        .map(|meta| {
            let name = match &meta.pubchem_title {
                Some(title) => title.clone(),
                None => meta.smiles.clone(),
            };
            (meta.smiles.clone(), name)
        })
        .collect();

    if hits.is_empty() {
        return CommonDbOutcome::NoMatch;
    }

    let mut to_load = None;

    for (i, (smiles, name)) in hits.iter().enumerate() {
        // The head is the best match, and so the one Enter loads; highlight it to show that.
        let color = if i == 0 && enter_live {
            COLOR_HIGHLIGHT
        } else {
            COLOR_ACTION
        };

        if button!(
            ui,
            name,
            color,
            "Open this molecule from the database built into the application. No internet \
             connection is used."
        )
        .clicked()
        {
            to_load = Some((smiles.clone(), name.clone()));
        }
    }

    // Ranked best-first, so the head is the best match.
    if to_load.is_none() && enter_pressed {
        to_load = Some(hits[0].clone());
    }

    let Some((smiles, name)) = to_load else {
        return CommonDbOutcome::Matched;
    };

    // Re-borrowed here rather than reused from above: `load_mol` needs the DB, and opening the
    // molecule needs `state`.
    let mol = {
        let Some(db) = &state.chebi_mol_db else {
            return CommonDbOutcome::Matched;
        };

        match db.load_mol(&smiles) {
            Ok(mut mol) => {
                // `mol_data` and the idents/metadata are separate columns; see the `mol_db` module
                // docs. Missing idents are not fatal — the molecule is still usable.
                if let Err(e) = db.apply_idents_meta(slice::from_mut(&mut mol)) {
                    eprintln!("Error loading idents for {smiles}: {e}");
                }
                mol
            }
            Err(e) => {
                handle_err(
                    &mut state.ui,
                    format!("Error loading {smiles} from the built-in database: {e}"),
                );
                return CommonDbOutcome::Matched;
            }
        }
    };

    let cache_result = managed_mols::store_sdf(
        &state.volatile.prefs_dir,
        ManagedMolProvider::BuiltIn,
        &managed_mols::text_key(&smiles),
        &smiles,
        &mol.to_sdf(),
    );
    let Some(cache_path) = report_cache_result(state, cache_result) else {
        return CommonDbOutcome::Matched;
    };
    open_lig_from_input(state, mol, Some(&cache_path), scene, updates);
    redraw.ligand = true;

    handle_success(
        &mut state.ui,
        format!("Loaded {name} ({smiles}) from {CHEBI_DB_NAME}."),
    );

    CommonDbOutcome::Loaded
}

/// Which of the query bar's remote lookups the Enter key acts on. Several lookups can match one
/// query — a 4-digit number is both a PubChem CID and an RCSB ident — but only one of them owns
/// Enter, and it's the only one drawn highlighted.
#[derive(Clone, Copy, PartialEq)]
enum EnterTarget {
    /// Enter does nothing: the query is too short, matches no lookup, or the built-in DB claimed it.
    None,
    PubchemCid,
    /// A `chebi:`-prefixed accession, which no other lookup competes for.
    ChebiId,
    /// A `pdbe:`-prefixed PDB ID, e.g. `pdbe:1crn`.
    PdbeStructure,
    /// A `pdbe:`-prefixed chemical component ID, e.g. `pdbe:ATP`.
    PdbeLigand,
    /// A UniProtKB accession, e.g. `P09838`, optionally `uniprot:`-prefixed.
    Uniprot,
    Rcsb,
    Geostd,
    DrugBank,
    Smiles,
    PubchemSearch,
}

/// Prefix that pins a query to ChEBI alone, e.g. `CHEBI:46195`. Matched case-insensitively, against
/// the lowercased query.
const CHEBI_PREFIX: &str = "chebi:";

/// Prefix that pins a query to PDBe alone: a structure for a PDB ID, e.g. `pdbe:1crn`, and a chemical
/// component otherwise, e.g. `pdbe:ATP`. Matched case-insensitively, against the lowercased query.
const PDBE_PREFIX: &str = "pdbe:";

/// Whether a query's text, less any `pdbe:` prefix, is a PDB ID rather than a chemical component ID:
/// the digit-leading, 4-character legacy form, or the 12-character `pdb_`-prefixed one.
fn is_pdb_id(inp_l: &str) -> bool {
    let is_legacy = inp_l.len() == 4
        && inp_l
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_digit() && b != b'0')
        && inp_l.bytes().all(|b| b.is_ascii_alphanumeric());
    let is_extended = inp_l.len() == 12
        && inp_l.starts_with("pdb_")
        && inp_l[4..].bytes().all(|b| b.is_ascii_alphanumeric());

    is_legacy || is_extended
}

/// Decide which lookup Enter activates, mirroring the order the buttons are drawn in below. This is
/// resolved once, ahead of drawing, so the highlighted button and the one Enter actually loads can't
/// disagree.
fn enter_target(inp: &str, inp_l: &str) -> EnterTarget {
    // An explicitly prefixed ChEBI accession names its database, so nothing else can claim it.
    if inp_l.starts_with(CHEBI_PREFIX) {
        return EnterTarget::ChebiId;
    }

    if let Some(id) = inp_l.strip_prefix(PDBE_PREFIX) {
        return match is_pdb_id(id.trim()) {
            true => EnterTarget::PdbeStructure,
            false => EnterTarget::PdbeLigand,
        };
    }

    // UniProt accessions have a fixed format that no other lookup's idents share.
    if uniprot::is_accession(inp) {
        return EnterTarget::Uniprot;
    }

    // A numeric query is a PubChem CID, and takes Enter even when it also looks like an RCSB ident
    // (4 digits) or a Geostd one (3 digits): those idents are alphanumeric in practice, so an
    // all-digit query is far more likely meant as a CID. Their buttons are still drawn, one click away.
    if inp.parse::<u32>().is_ok() {
        return EnterTarget::PubchemCid;
    }

    // An RCSB ident, in either the bare 4-character form or the 12-character `pdb_`-prefixed one.
    // Both belong to RCSB, so nothing below can claim them.
    if is_pdb_id(inp_l) {
        return EnterTarget::Rcsb;
    }

    // Uppercase three-character inputs conventionally denote PDB chemical components. Lowercase
    // inputs such as `trp` are more likely PubChem name searches; Geostd and PDBe remain available
    // as explicit buttons for either spelling.
    if inp.len() == 3
        && inp
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    {
        return EnterTarget::Geostd;
    }

    if inp.len() > 4 && inp_l.starts_with("db") {
        return EnterTarget::DrugBank;
    }

    if is_smiles(inp) {
        return EnterTarget::Smiles;
    }

    if inp.len() >= QUERY_ENTER_LEN_MIN {
        return EnterTarget::PubchemSearch;
    }

    EnterTarget::None
}

/// Draws one of the query bar's remote-lookup buttons, highlighting it if Enter would activate it.
/// Only one button in the bar is ever the Enter target.
fn query_btn(ui: &mut Ui, text: &str, is_enter_target: bool) -> Response {
    let text = RichText::new(text);

    ui.button(match is_enter_target {
        true => text.color(COLOR_HIGHLIGHT),
        false => text,
    })
}

/// Apply an already-downloaded ChEBI structure on the UI thread. Shared with the Rhea popup's
/// background downloads so identifiers, managed files, history, and rendering stay consistent.
pub(in crate::ui) fn open_chebi_download(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    updates: &mut EngineUpdates,
    id: u32,
    downloaded: DownloadedSmallMol,
) -> Result<(), String> {
    let key = id.to_string();
    // ChEBI's original file has no data fields; our SDF preserves the accession across restarts.
    let cache_path = managed_mols::store_sdf(
        &state.volatile.prefs_dir,
        ManagedMolProvider::Chebi,
        &key,
        &key,
        &downloaded.mol.to_sdf(),
    )
    .map_err(|error| format!("Downloaded CHEBI:{id}, but could not cache it: {error}"))?;

    open_lig_from_input(state, downloaded.mol, Some(&cache_path), scene, updates);
    redraw.ligand = true;
    handle_success(
        &mut state.ui,
        format!(
            "Loaded CHEBI:{id} from ChEBI (over the internet). Note that ChEBI structures are 2D."
        ),
    );
    Ok(())
}

enum QueryRequest {
    Protein(CifSource, String),
    Uniprot(String),
    PubchemCid(u32),
    PubchemSearch(String),
    Chebi(u32),
    PdbeLigand(String),
    Drugbank(String),
    Geostd(String),
    Smiles(String),
}

pub(crate) enum QueryResult {
    Protein {
        source: CifSource,
        ident: String,
        cif: MmCif,
        cif_text: String,
        message: Option<String>,
    },
    Small {
        downloaded: DownloadedSmallMol,
        provider: ManagedMolProvider,
        key: String,
        query: String,
        message: String,
        canonical_sdf: bool,
    },
    Geostd {
        ident: String,
        data: amber_geostd::GeostdData,
    },
    Smiles {
        input: String,
        mol: MoleculeSmall,
    },
    NoResults,
    Error(String),
}

fn start_query(state: &mut State, request: QueryRequest) {
    if state.volatile.thread_receivers.query.is_some() {
        return;
    }

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(run_query(request));
    });
    state.volatile.thread_receivers.query = Some(rx);
}

fn run_query(request: QueryRequest) -> QueryResult {
    match request {
        QueryRequest::Protein(source, ident) => protein_query(source, ident, None),
        QueryRequest::Uniprot(input) => {
            let accession = uniprot::parse_accession(&input);
            let pdb_ids = match uniprot::best_pdb_ids(&accession) {
                Ok(ids) => ids,
                Err(error) => {
                    return QueryResult::Error(format!(
                        "Error finding structures of UniProt {accession}. Is the accession correct? \
                         {error:?}"
                    ));
                }
            };

            if let Some(pdb_id) = pdb_ids.first() {
                protein_query(
                    CifSource::Rcsb,
                    pdb_id.clone(),
                    Some(format!(
                        "Loaded {}, the best of {} experimental structures of UniProt \
                         {accession} by sequence coverage and resolution, from RCSB \
                         (over the internet)",
                        pdb_id.to_uppercase(),
                        pdb_ids.len(),
                    )),
                )
            } else {
                protein_query(
                    CifSource::AlphaFold,
                    accession.clone(),
                    Some(format!(
                        "UniProt {accession} has no experimental structures; loaded its predicted \
                         structure from AlphaFold DB (over the internet)"
                    )),
                )
            }
        }
        QueryRequest::PubchemCid(cid) => pubchem_query(cid, cid.to_string(), None),
        QueryRequest::PubchemSearch(input) => {
            let cids = match find_cids_from_search(&input, false) {
                Ok(cids) => cids,
                Err(error) => {
                    return QueryResult::Error(format!(
                        "Error finding a mol from Pubchem {error:?}"
                    ));
                }
            };
            let Some(&cid) = cids.first() else {
                return QueryResult::NoResults;
            };
            let cids_str = cids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            pubchem_query(
                cid,
                input,
                Some(format!(
                    "Found the following Pubchem CIDs: {cids_str}. Loaded {cid} from \
                     PubChem (over the internet)"
                )),
            )
        }
        QueryRequest::Chebi(id) => match load_sdf_chebi(id) {
            Ok(downloaded) => QueryResult::Small {
                downloaded,
                provider: ManagedMolProvider::Chebi,
                key: id.to_string(),
                query: id.to_string(),
                message: format!(
                    "Loaded CHEBI:{id} from ChEBI (over the internet). Note that ChEBI \
                     structures are 2D."
                ),
                canonical_sdf: true,
            },
            Err(error) => QueryResult::Error(format!("Error loading SDF file: {error:?}")),
        },
        QueryRequest::PdbeLigand(ident) => {
            let ident = ident.trim().to_uppercase();
            match load_sdf_pdbe(&ident) {
                Ok(downloaded) => QueryResult::Small {
                    downloaded,
                    provider: ManagedMolProvider::Pdbe,
                    key: ident.clone(),
                    query: ident.clone(),
                    message: format!(
                        "Loaded chemical component {ident} from PDBe (over the internet)"
                    ),
                    canonical_sdf: true,
                },
                Err(error) => QueryResult::Error(format!(
                    "Error loading chemical component {ident} from PDBe: {error:?}"
                )),
            }
        }
        QueryRequest::Drugbank(ident) => match load_sdf_drugbank(&ident) {
            Ok(downloaded) => QueryResult::Small {
                downloaded,
                provider: ManagedMolProvider::Drugbank,
                key: ident.clone(),
                query: ident.clone(),
                message: format!("Loaded {ident} from DrugBank (over the internet)"),
                canonical_sdf: false,
            },
            Err(error) => QueryResult::Error(format!("Error loading SDF file: {error:?}")),
        },
        QueryRequest::Geostd(ident) => match amber_geostd::load_mol_files(&ident) {
            Ok(data) => QueryResult::Geostd { ident, data },
            Err(error) => {
                QueryResult::Error(format!("Unable to load Amber Geostd data: {error:?}"))
            }
        },
        QueryRequest::Smiles(input) => {
            let mut common = match parse_smiles(&input) {
                Ok(common) => common,
                Err(error) => {
                    return QueryResult::Error(format!(
                        "Error loading a molecule from SMILES: {error:?}"
                    ));
                }
            };
            let smiles_start: String = input.chars().take(5).collect();
            common.ident = format!("From SMILES {smiles_start}");
            let mol = MoleculeSmall {
                common,
                idents: vec![MolIdent::Smiles(input.clone())],
                ..Default::default()
            };
            QueryResult::Smiles { input, mol }
        }
    }
}

fn protein_query(source: CifSource, ident: String, message: Option<String>) -> QueryResult {
    match load_cif(source, &ident) {
        Ok((cif, cif_text)) => QueryResult::Protein {
            source,
            ident,
            cif,
            cif_text,
            message,
        },
        Err(error) => QueryResult::Error(format!(
            "Problem loading {ident} from {}: {error:?}",
            source.name(),
        )),
    }
}

fn pubchem_query(cid: u32, query: String, message: Option<String>) -> QueryResult {
    match load_sdf_pubchem(cid) {
        Ok(downloaded) => QueryResult::Small {
            downloaded,
            provider: ManagedMolProvider::Pubchem,
            key: cid.to_string(),
            query,
            message: message
                .unwrap_or_else(|| format!("Loaded CID {cid} from PubChem (over the internet)")),
            canonical_sdf: false,
        },
        Err(error) => QueryResult::Error(format!("Error loading SDF file: {error:?}")),
    }
}

/// Finish a query on the UI thread, after the worker has sent its downloaded data.
pub(crate) fn apply_query_result(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut RedrawFlags,
    reset_cam: &mut bool,
    updates: &mut EngineUpdates,
    result: QueryResult,
) {
    match result {
        QueryResult::Protein {
            source,
            ident,
            cif,
            cif_text,
            message,
        } => {
            if open_atom_coords(
                source,
                &ident,
                Ok((cif, cif_text)),
                state,
                scene,
                updates,
                &mut redraw.peptide,
                reset_cam,
            ) {
                state.ui.db_input.clear();
                if let Some(message) = message {
                    handle_success(&mut state.ui, message);
                }
            }
        }
        QueryResult::Small {
            downloaded,
            provider,
            key,
            query,
            message,
            canonical_sdf,
        } => {
            let cache_path = if canonical_sdf {
                report_cache_result(
                    state,
                    managed_mols::store_sdf(
                        &state.volatile.prefs_dir,
                        provider,
                        &key,
                        &query,
                        &downloaded.mol.to_sdf(),
                    ),
                )
            } else {
                cache_sdf_source(state, provider, &key, &query, &downloaded.source_text)
            };
            let Some(cache_path) = cache_path else {
                return;
            };
            open_lig_from_input(state, downloaded.mol, Some(&cache_path), scene, updates);
            redraw.ligand = true;
            handle_success(&mut state.ui, message);
        }
        QueryResult::Geostd { ident, data } => {
            open_geostd2(state, scene, &ident, true, true, updates, Ok(data));
            state.ui.db_input.clear();
        }
        QueryResult::Smiles { input, mol } => {
            let cache_result = managed_mols::store_sdf(
                &state.volatile.prefs_dir,
                ManagedMolProvider::Smiles,
                &managed_mols::text_key(&input),
                &input,
                &mol.to_sdf(),
            );
            let Some(cache_path) = report_cache_result(state, cache_result) else {
                return;
            };
            open_lig_from_input(state, mol, Some(&cache_path), scene, updates);
            redraw.ligand = true;
            handle_success(
                &mut state.ui,
                "Built this molecule from the SMILES entered; no database".to_owned(),
            );
        }
        QueryResult::NoResults => {
            handle_success(&mut state.ui, "No results found on Pubchem".to_owned());
        }
        QueryResult::Error(error) => handle_err(&mut state.ui, error),
    }
}

/// Handles a general query, which could be a name, identifier etc. Attempts to query
/// the correct database based on the  text.
///
/// inp is trimmed and case-preserving; inp_l is its ASCII-lowercase form.
pub(in crate::ui) fn load_mol_from_query(
    state: &mut State,
    ui: &mut Ui,
    inp: &str,
    inp_l: &str,
    enter_pressed: bool,
) {
    // Molecules we ship with the application. Checked ahead of the remote databases below, since
    // these load instantly and without an internet connection, and Enter picks the best local match
    // over any of them. Matches are still drawn as buttons, so a remote lookup stays one click away.
    // Whether Enter does anything at all for this query; a shorter one it ignores. Nothing is
    // highlighted as the Enter target when it isn't one.
    let enter_live = inp.len() >= QUERY_ENTER_LEN_MIN;

    // DB integration disabled for now: the built-in ChEBI database is 2D-only, so molecules loaded
    // from it lack the 3D coordinates the rest of the app expects. Re-enable once we have a 3D
    // source. The query then falls straight through to the remote lookups below.
    // let common = query_common_db(
    //     state,
    //     scene,
    //     redraw,
    //     updates,
    //     ui,
    //     inp,
    //     enter_pressed,
    //     enter_live,
    // );
    let common = CommonDbOutcome::NoMatch;

    // Which remote lookup below Enter activates — `None` unless Enter is live for this query and the
    // built-in DB didn't already claim the key. Drives both the highlight and the Enter handling in
    // each branch, so exactly one lookup responds to the key.
    let enter_tgt = match common {
        CommonDbOutcome::Loaded => return,
        CommonDbOutcome::Matched => EnterTarget::None,
        CommonDbOutcome::NoMatch => match enter_live {
            true => enter_target(inp, inp_l),
            false => EnterTarget::None,
        },
    };

    // A prefixed ChEBI accession, e.g. `CHEBI:46195`. This names one database, so unlike the bare
    // number below it queries ChEBI alone, and nothing further down is offered for it.
    if inp_l.starts_with(CHEBI_PREFIX) {
        let is_tgt = enter_tgt == EnterTarget::ChebiId;
        let button_clicked = query_btn(ui, "Load ChEBI", is_tgt).clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            match chebi::parse_id(inp_l) {
                Ok(id) => start_query(state, QueryRequest::Chebi(id)),
                Err(e) => handle_err(
                    &mut state.ui,
                    format!("{inp} is not a valid ChEBI accession: {e:?}"),
                ),
            }
        }

        return;
    }

    // A `pdbe:`-prefixed ident, e.g. `pdbe:1crn` or `pdbe:ATP`. Like `chebi:`, this names one
    // database, so nothing further down is offered for it.
    if let Some(id) = inp_l.strip_prefix(PDBE_PREFIX) {
        let id = id.trim();
        if id.is_empty() {
            return;
        }

        if is_pdb_id(id) {
            let is_tgt = enter_tgt == EnterTarget::PdbeStructure;
            if query_btn(ui, "Load PDBe", is_tgt).clicked() || (enter_pressed && is_tgt) {
                start_query(state, QueryRequest::Protein(CifSource::Pdbe, id.to_owned()));
            }
        } else {
            let is_tgt = enter_tgt == EnterTarget::PdbeLigand;
            if query_btn(ui, "Load PDBe", is_tgt).clicked() || (enter_pressed && is_tgt) {
                start_query(state, QueryRequest::PdbeLigand(id.to_owned()));
            }
        }

        return;
    }

    // A UniProtKB accession, e.g. `P09838`. UniProt has no structures itself, so we load the best
    // experimental one from the PDB, or AlphaFold DB's prediction on request.
    if uniprot::is_accession(inp) {
        let is_tgt = enter_tgt == EnterTarget::Uniprot;
        let button_clicked = query_btn(ui, "Load UniProt", is_tgt)
            .on_hover_text(
                "Load the best experimental structure of this protein from the PDB, ranked by \
                 sequence coverage, then resolution. Loads the AlphaFold DB prediction if there \
                 are none.",
            )
            .clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            start_query(state, QueryRequest::Uniprot(inp.to_owned()));
        }

        if query_btn(ui, "Load AlphaFold", false)
            .on_hover_text("Load the predicted structure of this protein from AlphaFold DB.")
            .clicked()
        {
            start_query(
                state,
                QueryRequest::Protein(CifSource::AlphaFold, uniprot::parse_accession(inp)),
            );
        }

        return;
    }

    // PubChem CID. Don't return early here; continue to allow for other
    if let Ok(cid) = inp.parse::<u32>() {
        let is_tgt = enter_tgt == EnterTarget::PubchemCid;
        if query_btn(ui, "Load PubChem", is_tgt).clicked() || (enter_pressed && is_tgt) {
            start_query(state, QueryRequest::PubchemCid(cid));
        }

        // A bare number is also a ChEBI accession. PubChem owns Enter, being the larger database;
        // ChEBI is one click away, or unambiguous with a `chebi:` prefix.
        if query_btn(ui, "Load ChEBI", false).clicked() {
            start_query(state, QueryRequest::Chebi(cid));
        }
    }

    if is_pdb_id(inp_l) {
        // Both ident forms load: the bare 4-character one, and the 12-character `pdb_`-prefixed one
        // RCSB has moved to. `files.rcsb.org` accepts either.
        let is_tgt = enter_tgt == EnterTarget::Rcsb;
        let button_clicked = query_btn(ui, "Load RCSB", is_tgt).clicked();
        if button_clicked || (enter_pressed && is_tgt) {
            start_query(
                state,
                QueryRequest::Protein(CifSource::Rcsb, inp_l.to_owned()),
            );
            return;
        }

        // The same entry, from PDBe's mirror of the archive. RCSB owns Enter.
        if query_btn(ui, "Load PDBe", false).clicked() {
            start_query(
                state,
                QueryRequest::Protein(CifSource::Pdbe, inp_l.to_owned()),
            );
        }

        return;
    }

    if inp.len() == 3 {
        let is_tgt = enter_tgt == EnterTarget::Geostd;
        let button_clicked = query_btn(ui, "Load Geostd", is_tgt).clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            start_query(state, QueryRequest::Geostd(inp_l.to_owned()));
        }

        // A chemical component from PDBe, e.g. `ATP`. These share their IDs with Geostd, which owns
        // Enter as it also provides force field parameters; PDBe covers components Geostd doesn't.
        if query_btn(ui, "Load PDBe", false)
            .on_hover_text("Load this chemical component (e.g. a ligand) from PDBe.")
            .clicked()
        {
            start_query(state, QueryRequest::PdbeLigand(inp.to_owned()));
        }
    }

    if inp.len() > 4 && inp_l.starts_with("db") {
        let is_tgt = enter_tgt == EnterTarget::DrugBank;
        let button_clicked = query_btn(ui, "Load DrugBank", is_tgt).clicked();

        if button_clicked || (enter_pressed && is_tgt) {
            start_query(state, QueryRequest::Drugbank(inp_l.to_owned()));
        }

        return;
    }

    // Recognizing SMILES is cheap enough to do while drawing the input; building the molecule
    // happens only after submission, on the worker.
    if is_smiles(inp) {
        let is_tgt = enter_tgt == EnterTarget::Smiles;
        let button_clicked = query_btn(ui, "Load from SMILES", is_tgt).clicked();
        if (enter_pressed && is_tgt) || button_clicked {
            start_query(state, QueryRequest::Smiles(inp.to_owned()));
        }
        return;
    }

    // PubChem name search.
    let is_prefixed_database_query = inp_l.starts_with("pdb_") || inp_l.starts_with("db");
    if inp.len() >= QUERY_ENTER_LEN_MIN && !is_prefixed_database_query {
        let is_tgt = enter_tgt == EnterTarget::PubchemSearch;
        let button_clicked = query_btn(ui, "Search PubChem", is_tgt).clicked();
        if button_clicked || (enter_pressed && is_tgt) {
            start_query(state, QueryRequest::PubchemSearch(inp.to_owned()));
        }
    }
}

/// Toggles chain visibility
pub(in crate::ui) fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    let Some(mol) = state
        .peptide_for_tools_i()
        .and_then(|i| state.peptides.get_mut(i))
    else {
        return;
    };

    ui.horizontal_wrapped(|ui| {
        ui.label("Chain vis:");
        for chain in &mut mol.chains {
            let color = misc::active_color(chain.visible);

            if ui
                .button(RichText::new(chain.id.clone()).color(color))
                .clicked()
            {
                chain.visible = !chain.visible;
                if state.ui.mol_view_peptide == MoleculeView::Ribbon {
                    state.volatile.flags.update_ss_mesh = true;
                } else {
                    state.volatile.flags.ss_mesh_dirty = true;
                }
                *redraw = true;
            }
        }

        ui.add_space(crate::ui::COL_SPACING);

        ui.label("Select residues from:");

        for (i, chain) in mol.chains.iter().enumerate() {
            let mut color = Color32::GRAY;
            if let Some(i_sel) = state.ui.chain_to_pick_res
                && i == i_sel
            {
                color = crate::ui::COLOR_ACTIVE
            }
            if ui
                .button(RichText::new(chain.id.clone()).color(color))
                .clicked()
            {
                // Toggle behavior.
                if let Some(sel_i) = state.ui.chain_to_pick_res {
                    if i == sel_i {
                        state.ui.chain_to_pick_res = None;
                    } else {
                        state.ui.chain_to_pick_res = Some(i);
                    }
                } else {
                    state.ui.chain_to_pick_res = Some(i);
                }

                state.ui.popup.residue_selector = !state.ui.popup.residue_selector;
            }
        }

        if state.ui.chain_to_pick_res.is_some() {
            if ui.button("(None)").clicked() {
                state.ui.chain_to_pick_res = None;
            }
        }
    });
}
