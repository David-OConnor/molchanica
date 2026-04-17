use dynamics::snapshot::Snapshot;
use egui::{Color32, RichText, ScrollArea, Slider, TextEdit, Ui};
use graphics::{EngineUpdates, FWD_VEC, Scene};
use na_seq::Element;

use crate::{
    button,
    cam::reset_camera,
    label,
    md::{viewer, viewer::ViewerMolecule},
    molecules::{Atom, MolType, common::MoleculeCommon},
    state::State,
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, ROW_SPACING,
        highlighted_box, num_field, popups::close_btn,
    },
    util::{RedrawFlags, handle_err, handle_success},
};

pub(in crate::ui) fn dynamics_viewer(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        // let prev = state.ui.peptide_atom_posits;
        let help_text = "Toggle between viewing the original (pre-dynamics) atom positions, and \
        ones at the selected dynamics snapshot.";

        let ready = state.volatile.md_local.viewer.mols_and_traj_synced();

        {
            let text = if state.volatile.md_local.draw_md_mols {
                "Draw original"
            } else {
                "Draw MD"
            };

            if (state.volatile.md_local.draw_md_mols
                || (!state.volatile.md_local.viewer.snapshots.is_empty() && ready))
                && ui
                    .button(RichText::new(text).color(COLOR_HIGHLIGHT))
                    .on_hover_text(help_text)
                    .clicked()
            {
                state.volatile.md_local.draw_md_mols = !state.volatile.md_local.draw_md_mols;

                redraw.set_all();
            }
        }

        if !state.volatile.md_local.viewer.snapshots.is_empty()
            && ui
                .button(RichText::new("Clear Traj"))
                .on_hover_text("Clear all trajectory snapshots, e.g. erase the previous run.")
                .clicked()
        {
            state
                .volatile
                .md_local
                .clear_snaps(&mut scene.entities, redraw);

            // todo: Make this call part of clear_snaps?
            for traj in &mut state.trajectories {
                traj.frames_open = None;
            }
        }

        if !ready {
            let txt = if state.volatile.md_local.viewer.snapshots.is_empty() {
                "Not ready for MD playback; no snapshots/frames loaded.".to_owned()
            } else {
                let mut t =
                    "Not ready for MD playback; trajectory and mol set atom count mismatch: "
                        .to_owned();

                let viewer = &state.volatile.md_local.viewer;

                if !viewer.snapshots.is_empty()
                    && let Some(set) = &viewer.get_active_mol_set()
                {
                    t.push_str(&format!(
                        "Traj: {} Mol set: {}",
                        state.volatile.md_local.viewer.snapshots[0]
                            .atom_posits
                            .len(),
                        state
                            .volatile
                            .md_local
                            .viewer
                            .get_active_mol_set()
                            .unwrap()
                            .atom_count
                    ));
                }
                t
            };

            label!(ui, txt, Color32::GRAY);

            return;
        }

        let slider_posit_prev = state.volatile.md_local.viewer.slider_posit_ui;

        if !state.volatile.md_local.viewer.snapshots.is_empty() {
            ui.add_space(ROW_SPACING);

            ui.spacing_mut().slider_width = ui.available_width() - 100.;

            ui.add(Slider::new(
                &mut state.volatile.md_local.viewer.slider_posit_ui,
                0..=state.volatile.md_local.viewer.snapshots.len() - 1,
            ));

            if let Some(ss) = &state.volatile.md_local.viewer.get_active_snap() {
                ui.label(format!("{:.2} ps", ss.time));
            }

            let posit = state.volatile.md_local.viewer.slider_posit_ui;

            if posit != slider_posit_prev && posit <= state.volatile.md_local.viewer.snapshots.len()
            {
                state.volatile.md_local.viewer.current_snapshot = Some(posit);

                if let Err(e) = state.volatile.md_local.viewer.change_snapshot(posit) {
                    handle_err(&mut state.ui, format!("Error changing MD snapshot: {e:?}"));
                }

                viewer::draw_mols(state, scene, updates);
            }
        }
    });
}

pub(in crate::ui) fn energy_disp(snap: &Snapshot, ui: &mut Ui) {
    let Some(en) = &snap.energy_data else { return };

    ui.label(RichText::new("Energy (current snap):").color(Color32::WHITE));
    ui.add_space(COL_SPACING / 2.);

    ui.label("E (kcal/mol) KE: ");
    ui.label(RichText::new(format!("{:.1}", en.energy_kinetic)).color(Color32::GOLD));

    ui.label("KE/atom: ");
    // todo: Don't continuously run these computations!
    let atom_count = (snap.water_o_posits.len() * 3) as f32 + snap.atom_posits.len() as f32;
    let ke_per_atom = en.energy_kinetic / atom_count;
    label!(ui, format!("{:.2}", ke_per_atom), Color32::GOLD);

    ui.label("PE: ");
    label!(ui, format!("{:.2}", en.energy_potential), Color32::GOLD);

    if en.energy_potential_nonbonded.abs() > 0.0001 {
        ui.label("PE NB: ");
        label!(
            ui,
            format!("{:.2}", en.energy_potential_nonbonded),
            Color32::GOLD
        );
    }

    ui.label("PE/atom: ");
    // todo: Don't continuously run this!
    let pe_per_atom = en.energy_potential / atom_count;
    label!(ui, format!("{:.3}", pe_per_atom), Color32::GOLD);

    ui.label("E tot: ");
    // todo: Don't continuously run this!
    let e = en.energy_potential + en.energy_kinetic;
    label!(ui, format!("{:.2}", e), Color32::GOLD);

    // ui.label("PE between mols:");
    // // todo: One pair only for now
    // if en.energy_potential_between_mols.len() >= 2 {
    //     // todo: Which index?
    //     ui.label(
    //         RichText::new(format!("{:.2}", en.energy_potential_between_mols[1]))
    //             .color(Color32::GOLD),
    //     );
    // }

    ui.label("Temp: ");
    label!(ui, format!("{:.1} K", en.temperature), Color32::GOLD);

    ui.label("P: ");
    label!(ui, format!("{:.1} bar", en.pressure), Color32::GOLD);

    ui.label("Vol: ");
    label!(
        ui,
        format!("{:.1} k Å^3", en.volume / 1_000.),
        Color32::GOLD
    );

    ui.label("Dens: ");
    label!(ui, format!("{:.2} amu/Å^3", en.density), Color32::GOLD);
}

fn mol_type_label(t: MolType) -> &'static str {
    match t {
        MolType::Peptide => "Protein",
        MolType::Ligand => "Ligand",
        MolType::NucleicAcid => "Nucleic Acid",
        MolType::Lipid => "Lipid",
        MolType::Water => "Water",
        MolType::Pocket => "Pocket",
    }
}

/// Build a single water `ViewerMolecule` whose range starts at `start_i`.
fn make_water_mol(start_i: usize) -> ViewerMolecule {
    let zero = lin_alg::f64::Vec3::default();
    let atoms = vec![
        Atom {
            serial_number: start_i as u32,
            posit: zero,
            type_in_res_general: Some("OW".to_owned()),
            hetero: true,
            element: Element::Oxygen,
            ..Default::default()
        },
        Atom {
            serial_number: (start_i + 1) as u32,
            posit: zero,
            type_in_res_general: Some("HW1".to_owned()),
            hetero: true,
            element: Element::Hydrogen,
            ..Default::default()
        },
        Atom {
            serial_number: (start_i + 2) as u32,
            posit: zero,
            type_in_res_general: Some("HW2".to_owned()),
            hetero: true,
            element: Element::Hydrogen,
            ..Default::default()
        },
    ];
    let atom_posits = vec![zero; 3];
    ViewerMolecule {
        mol_type: MolType::Water,
        mol: MoleculeCommon {
            ident: "SOL".to_owned(),
            atoms,
            atom_posits,
            ..Default::default()
        },
        range: (start_i, start_i + 3),
    }
}

/// Create and edit MD molecule sets; these are associated with trajectories; they map a trajectory's
/// flat atom index to molecules, for display purposes.
///
/// Sets are stored in the .gro format; GROMACS's text-based format for this. This window lets you create
/// new sets, and edit existing ones.
pub(in crate::ui) fn md_mol_set_editor(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        label!(
            ui,
            "Edit and add molecule sets for MD playback",
            Color32::WHITE
        );
        ui.add_space(COL_SPACING);

        if button!(ui, "New", COLOR_ACTION, "Create a new empty molecule set.").clicked() {
            let name = format!("Set {}", state.volatile.md_local.viewer.mol_sets.len() + 1);
            state
                .volatile
                .md_local
                .viewer
                .mol_sets
                .push(viewer::ViewerMolSet::new(None, name, vec![]));
            state.ui.md.set_editor_active_set =
                Some(state.volatile.md_local.viewer.mol_sets.len() - 1);
        }
        ui.add_space(COL_SPACING);

        close_btn(ui, &mut state.ui.popup.md_mol_set_editor);
    });
    label!(
        ui,
        "Select a set to edit. To open a set from a file (e.g. .gro), open it like any other file. Use this editor to map molecules to trajectory atom ranges.",
        Color32::GRAY
    );

    ui.add_space(ROW_SPACING);

    // --- Set selection ---
    if state.volatile.md_local.viewer.mol_sets.is_empty() {
        label!(ui, "(No sets open)", Color32::GRAY);
        return;
    }

    ui.horizontal(|ui| {
        for (i, set) in state.volatile.md_local.viewer.mol_sets.iter().enumerate() {
            let active = Some(i) == state.ui.md.set_editor_active_set;
            let color = if active { COLOR_ACTIVE } else { COLOR_INACTIVE };
            if button!(ui, &set.name, color, "Select this set to edit.").clicked() {
                state.ui.md.set_editor_active_set = if active { None } else { Some(i) };
            }
        }
    });

    let Some(set_i) = state.ui.md.set_editor_active_set else {
        return;
    };

    if set_i >= state.volatile.md_local.viewer.mol_sets.len() {
        state.ui.md.set_editor_active_set = None;
        return;
    }

    ui.separator();
    ui.add_space(ROW_SPACING);
    ui.label(RichText::new("Molecules in set:").color(Color32::WHITE));

    // --- Edit current mols in the set (sorted by range, water grouped) ---
    let mols_len = state.volatile.md_local.viewer.mol_sets[set_i].mols.len();

    // Sorted indices, non-water only (water shown as a group below).
    let mut sorted_indices: Vec<usize> = (0..mols_len)
        .filter(|&j| {
            state.volatile.md_local.viewer.mol_sets[set_i].mols[j].mol_type != MolType::Water
        })
        .collect();
    sorted_indices.sort_by_key(|&j| {
        state.volatile.md_local.viewer.mol_sets[set_i].mols[j]
            .range
            .0
    });

    let water_count = state.volatile.md_local.viewer.mol_sets[set_i]
        .mols
        .iter()
        .filter(|m| m.mol_type == MolType::Water)
        .count();
    let (water_range_start, water_range_end) = {
        let set = &state.volatile.md_local.viewer.mol_sets[set_i];

        let starts = set
            .mols
            .iter()
            .filter(|m| m.mol_type == MolType::Water)
            .map(|m| m.range.0);

        let ends = set
            .mols
            .iter()
            .filter(|m| m.mol_type == MolType::Water)
            .map(|m| m.range.1);

        (starts.min().unwrap_or(0), ends.max().unwrap_or(0))
    };

    let mut remove_j: Option<usize> = None;
    let mut clear_water = false;

    if mols_len == 0 {
        label!(ui, "(No molecules in this set)", Color32::GRAY);
    }

    // Scrollable list of non-water mols + water summary row.
    ScrollArea::vertical()
        .id_salt("mol_set_editor_scroll")
        .max_height(260.)
        .show(ui, |ui| {
            for &j in &sorted_indices {
                let mol = &mut state.volatile.md_local.viewer.mol_sets[set_i].mols[j];
                ui.horizontal(|ui| {
                    label!(
                        ui,
                        format!(
                            "{} [{}] ({} atoms)",
                            mol.mol.ident,
                            mol_type_label(mol.mol_type),
                            mol.mol.atoms.len()
                        ),
                        Color32::WHITE
                    );
                    num_field(&mut mol.range.0, "Start:", 54, ui);
                    num_field(&mut mol.range.1, "End:", 54, ui);
                    if ui
                        .button(RichText::new("❌").color(Color32::LIGHT_RED))
                        .on_hover_text("Remove from set")
                        .clicked()
                    {
                        remove_j = Some(j);
                    }
                });
            }

            // Water summary row.
            if water_count > 0 {
                ui.horizontal(|ui| {
                    label!(
                        ui,
                        format!(
                            "Water: {} mols | Start: {} End: {}",
                            water_count, water_range_start, water_range_end
                        ),
                        Color32::from_rgb(100, 180, 255)
                    );
                    if button!(
                        ui,
                        "Clear water",
                        Color32::LIGHT_RED,
                        "Remove all water molecules from this set."
                    )
                    .clicked()
                    {
                        clear_water = true;
                    }
                });
            }
        });

    if state.volatile.md_local.viewer.mol_sets[set_i].range_overlaps {
        label!(ui, "Warning: Mol ranges overlap", Color32::LIGHT_RED);
    }

    if let Some(j) = remove_j {
        state.volatile.md_local.viewer.mol_sets[set_i]
            .mols
            .remove(j);
    }
    if clear_water {
        state.volatile.md_local.viewer.mol_sets[set_i]
            .mols
            .retain(|m| m.mol_type != MolType::Water);
    }
    if remove_j.is_some() || clear_water {
        state.volatile.md_local.viewer.mol_sets[set_i].update_derivative_vals();
    }

    // Always keep derivative vals up-to-date for range edits (num_field writes in-place).
    state.volatile.md_local.viewer.mol_sets[set_i].update_derivative_vals();

    ui.separator();
    ui.label(RichText::new("Add from open molecules:").color(Color32::WHITE));
    label!(
        ui,
        "Clicking Add places the molecule after the current last range. Edit the range fields above as needed.",
        Color32::GRAY
    );

    // Next suggested range start: after the current last range end.
    let next_range_start = state.volatile.md_local.viewer.mol_sets[set_i]
        .mols
        .iter()
        .map(|m| m.range.1)
        .max()
        .unwrap_or(0);

    // Collect any pending add to apply after reads (avoids borrow conflict).
    let mut to_add: Option<ViewerMolecule> = None;

    // Protein
    if let Some(pep) = &state.peptide {
        let atom_count = pep.common.atoms.len();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Protein:").color(Color32::CYAN));
            if button!(
                ui,
                format!("{} ({} atoms)", pep.common.ident, atom_count),
                COLOR_ACTION,
                "Add this protein to the mol set"
            )
            .clicked()
            {
                to_add = Some(ViewerMolecule {
                    mol_type: MolType::Peptide,
                    mol: pep.common.clone(),
                    range: (next_range_start, next_range_start + atom_count),
                });
            }
        });
    }

    // Ligands / small mols
    if !state.ligands.is_empty() {
        ui.label(RichText::new("Ligands:").color(Color32::GREEN));
        for lig in state.ligands.iter() {
            let atom_count = lig.common.atoms.len();
            let common = lig.common.clone();
            ui.horizontal(|ui| {
                if button!(
                    ui,
                    format!("{} ({} atoms)", common.ident, atom_count),
                    COLOR_ACTION,
                    "Add this ligand to the mol set"
                )
                .clicked()
                    && to_add.is_none()
                {
                    to_add = Some(ViewerMolecule {
                        mol_type: MolType::Ligand,
                        mol: common,
                        range: (next_range_start, next_range_start + atom_count),
                    });
                }
            });
        }
    }

    // Nucleic acids
    if !state.nucleic_acids.is_empty() {
        ui.label(RichText::new("Nucleic Acids:").color(Color32::YELLOW));
        for na in state.nucleic_acids.iter() {
            let atom_count = na.common.atoms.len();
            let common = na.common.clone();
            ui.horizontal(|ui| {
                if button!(
                    ui,
                    format!("{} ({} atoms)", common.ident, atom_count),
                    COLOR_ACTION,
                    "Add this nucleic acid to the mol set"
                )
                .clicked()
                    && to_add.is_none()
                {
                    to_add = Some(ViewerMolecule {
                        mol_type: MolType::NucleicAcid,
                        mol: common,
                        range: (next_range_start, next_range_start + atom_count),
                    });
                }
            });
        }
    }

    // Lipids
    if !state.lipids.is_empty() {
        ui.label(RichText::new("Lipids:").color(Color32::from_rgb(255, 128, 255)));
        for lip in state.lipids.iter() {
            let atom_count = lip.common.atoms.len();
            let common = lip.common.clone();
            ui.horizontal(|ui| {
                if button!(
                    ui,
                    format!("{} ({} atoms)", common.ident, atom_count),
                    COLOR_ACTION,
                    "Add this lipid to the mol set"
                )
                .clicked()
                    && to_add.is_none()
                {
                    to_add = Some(ViewerMolecule {
                        mol_type: MolType::Lipid,
                        mol: common,
                        range: (next_range_start, next_range_start + atom_count),
                    });
                }
            });
        }
    }

    if let Some(mol) = to_add {
        state.volatile.md_local.viewer.mol_sets[set_i]
            .mols
            .push(mol);
        state.volatile.md_local.viewer.mol_sets[set_i].update_derivative_vals();
    }

    // --- Fill rest with water ---
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Fill rest with water:").color(Color32::from_rgb(100, 180, 255)));
        ui.label("Final atom idx:");
        ui.add_sized(
            [64., Ui::available_height(ui)],
            TextEdit::singleline(&mut state.ui.md.water_fill_end_input),
        );

        let fill_clicked = button!(
            ui,
            "Fill",
            COLOR_ACTION,
            "Remove all current water mols and repopulate from the last non-water atom index up \
            to the given final index. The range must be divisible by 3 (OW + HW1 + HW2)."
        )
        .clicked();

        if fill_clicked {
            match state.ui.md.water_fill_end_input.trim().parse::<usize>() {
                Err(_) => {
                    handle_err(
                        &mut state.ui,
                        "Final atom index must be a positive integer.".to_owned(),
                    );
                }
                Ok(final_idx) => {
                    // Start = last non-water range end (or 0).
                    let water_start = state.volatile.md_local.viewer.mol_sets[set_i]
                        .mols
                        .iter()
                        .filter(|m| m.mol_type != MolType::Water)
                        .map(|m| m.range.1)
                        .max()
                        .unwrap_or(0);

                    let span = final_idx.saturating_sub(water_start);
                    // Round up to the nearest whole water molecule (3 atoms each).
                    let n_water = (span + 2) / 3;

                    // Remove existing water.
                    state.volatile.md_local.viewer.mol_sets[set_i]
                        .mols
                        .retain(|m| m.mol_type != MolType::Water);

                    // Add one ViewerMolecule per water molecule.
                    for k in 0..n_water {
                        let start = water_start + k * 3;
                        state.volatile.md_local.viewer.mol_sets[set_i]
                            .mols
                            .push(make_water_mol(start));
                    }

                    state.volatile.md_local.viewer.mol_sets[set_i].update_derivative_vals();
                }
            }
        }
    });

    ui.separator();
    if button!(ui, "Save", COLOR_ACTION, "Save this mol set as a GRO file.").clicked() {
        let name = format!(
            "{}.gro",
            state.volatile.md_local.viewer.mol_sets[set_i]
                .name
                .replace(' ', "_")
        );
        state
            .volatile
            .dialogs
            .save_gro
            .config_mut()
            .default_file_name = name;

        state.volatile.dialogs.save_gro_mol_set_i = Some(set_i);
        state.volatile.dialogs.save_gro.save_file();
    }

    ui.add_space(ROW_SPACING);
}

/// Selected from loaded molecule sets, which map molecules to a trajectory's flat atoms.
/// This might be loaded, for example, from a .gro file.
pub(in crate::ui) fn viewer_mol_set(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    ui: &mut Ui,
    redraw: &mut RedrawFlags,
) {
    let empty = state.volatile.md_local.viewer.mol_sets.is_empty();
    if empty && !state.trajectories.is_empty() {
        return;
    }

    ui.horizontal(|ui| {
        ui.label("MD mol sets");

        ui.add_space(COL_SPACING);

        // We show this if sets are empty, but trajectories are not.
        if button!(
        ui,
        "Edit sets",
        Color32::WHITE,
        "Add or edit molecule sets associated with MD trajectories. This lets you add open molecules,\
        and adjust which trajectory index range each is associated with."
    ).clicked() {
            state.ui.popup.md_mol_set_editor = !state.ui.popup.md_mol_set_editor;
        }

    });

    if empty {
        return;
    }

    ui.separator();

    let mut close = None;
    let mut set_clicked = false;

    for i in 0..state.volatile.md_local.viewer.mol_sets.len() {
        let mut redraw_active_set = false;
        let (active, color) = if Some(i) == state.volatile.md_local.viewer.mol_set_active {
            (true, COLOR_ACTIVE)
        } else {
            (false, Color32::WHITE)
        };

        // Extract display data before the closure to avoid borrow conflicts with &mut state.
        let (
            set_name,
            set_mols_len,
            set_atom_count,
            set_range_covered,
            set_range_overlaps,
            sorted_groups_display,
        ) = {
            if i >= state.volatile.md_local.viewer.mol_sets.len() {
                eprintln!("Error active mol set out of range");
                return;
            }
            let set = &state.volatile.md_local.viewer.mol_sets[i];
            (
                set.name.clone(),
                set.mols.len(),
                set.atom_count,
                set.range_covered,
                set.range_overlaps,
                set.groups_display(),
            )
        };

        highlighted_box(active, Color32::from_rgb(40, 40, 55)).show(ui, |ui| {
            ui.horizontal(|ui| {

                ui.label(RichText::new(format!("{} | {} mols", set_name, set_mols_len)).color(color));

                ui.label(RichText::new(format!("At: {} | Rng: {}-{}", set_atom_count, set_range_covered.0, set_range_covered.1)).color(color));

                if set_range_overlaps {
                    ui.label(RichText::new("Warning: Mol ranges overlap").color(Color32::LIGHT_RED));
                }

                if button!(
            ui,
            "Set",
            COLOR_ACTION,
            "Load this set of molecules into the MD trajectory atoms. This affects \
        how the atoms in the trajectory are visually mapped to molecules with covalent bonds, the correct element etc."
        ).clicked() {
                    state.volatile.md_local.viewer.mol_set_active = if active {
                        None }
                    else {
                        Some(i)
                    };

                    set_clicked = true;
                    handle_success(&mut state.ui, format!("Set {} as the active mol set", set_name));
                }

                if button!(ui, "Save", COLOR_ACTION, "Save this mol set as a GRO file.").clicked() {
                    let name = format!("{}.gro", set_name.replace(' ', "_"));
                    state.volatile.dialogs.save_gro.config_mut().default_file_name = name;
                    state.volatile.dialogs.save_gro_mol_set_i = Some(i);
                    state.volatile.dialogs.save_gro.save_file();
                }

                if ui
                    .button(RichText::new("❌").color(Color32::LIGHT_RED))
                    .on_hover_text("Close this molecule set.")
                    .clicked()
                {
                    close = Some(i);
                }

            });

            let max_count = 5;
            for group in sorted_groups_display.iter().take(max_count) {
                ui.horizontal(|ui| {
                    let mut text = if group.mol_count > 1 {
                        format!(
                            "{} ({} mols) | Atoms: {} Range: {}-{}",
                            group.ident,
                            group.mol_count,
                            group.atom_count,
                            group.range_covered.0,
                            group.range_covered.1,
                        )
                    } else {
                        format!(
                            "{} | Atoms: {} Range: {}-{}",
                            group.ident,
                            group.atom_count,
                            group.range_covered.0,
                            group.range_covered.1,
                        )
                    };

                    let mut visible = state.volatile.md_local.viewer.mol_sets[i].groups
                        [group.group_i]
                        .visible;

                    label!(
                        ui,
                        text,
                        if visible {
                            Color32::WHITE
                        } else {
                            Color32::GRAY
                        }
                    );

                    if ui.checkbox(&mut visible, "vis").changed() {
                        state.volatile.md_local.viewer.mol_sets[i].groups[group.group_i].visible =
                            visible;
                        redraw_active_set = active;
                    }
                });
            }

            if sorted_groups_display.len() > max_count {
                label!(ui, "...", Color32::WHITE);
            }

            ui.separator();

            if let Some(i) = close {
                state
                    .volatile
                    .md_local
                    .viewer
                    .close_mol_set(&mut state.to_save.open_history, i);
            }

            if set_clicked {
                // We have this as the function calls in this branch which call state have a borrow
                // error otherwise; the flag setting is convenience.
                redraw.set_all();

                if state.volatile.md_local.viewer.mol_set_active.is_some() {
                    let snap_i = state.volatile.md_local.viewer.current_snapshot.unwrap_or(0);

                    if state.volatile.md_local.viewer.change_snapshot(snap_i).is_err() {
                        handle_err(&mut state.ui, "Error changing snaps".to_string());
                    }

                    reset_camera(state, scene, updates, FWD_VEC);
                    viewer::draw_mols(state, scene, updates);
                }
            }

            if redraw_active_set && state.volatile.md_local.draw_md_mols {
                viewer::draw_mols(state, scene, updates);
                redraw.set_all();
            }
        });
    }
}
