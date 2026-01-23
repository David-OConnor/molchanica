use egui::{Color32, Context, RichText, Ui};
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::{
    cam::{move_cam_to_mol, move_mol_to_cam},
    label,
    mol_characterization::MolCharacterization,
    mol_manip::{ManipMode, set_manip},
    molecules::{MolGenericRef, MolIdent, MolType, common::MoleculeCommon, small::MoleculeSmall},
    state::State,
    therapeutic::{Adme, Toxicity},
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_HIGHLIGHT,
        COLOR_INACTIVE, ROW_SPACING,
    },
    util::{close_mol, handle_err, orbit_center},
};

/// Abstracts over all molecule types. (Currently not protein though)
/// A single row for the molecule.
fn mol_picker_one(
    active_mol: &mut Option<(MolType, usize)>,
    orbit_center: &mut Option<(MolType, usize)>,
    i_mol: usize,
    mol: &mut MoleculeCommon,
    mol_char: &Option<MolCharacterization>,
    mol_type: MolType,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    recenter_orbit: &mut bool,
    close: &mut Option<(MolType, usize)>,
    cam_snapshot: &mut Option<usize>,
    pep_center: Vec3,
) {
    let help_text = "Make this molecule the active / selected one. Middle click to close it.";

    let active = match active_mol {
        Some((mol_type_active, i)) => *mol_type_active == mol_type && *i == i_mol,
        _ => false,
    };

    let color = if active {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    };

    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(COL_SPACING / 2.);
            if ui
                .button(RichText::new("❌").color(Color32::LIGHT_RED))
                .on_hover_text("(Hotkey: Delete) Close this molecule.")
                .clicked()
            {
                *close = Some((mol_type, i_mol));
            }

            let color_md = if mol.selected_for_md {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if ui
                .button(RichText::new("MD").color(color_md))
                .on_hover_text(
                    "Select or deselect this molecule for molecular dynamics simulation.",
                )
                .clicked()
            {
                mol.selected_for_md = !mol.selected_for_md;
            }

            let color_vis = if mol.visible {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if ui.button(RichText::new("👁").color(color_vis)).clicked() {
                mol.visible = !mol.visible;

                *redraw = true; // todo Overkill; only need to redraw (or even just clear) one.
                // todo: Generalize.
                engine_updates.entities = EntityUpdate::All;
                // engine_updates.entities.push_class(mol_type.entity_class() as u32);
            }

            if ui
                .button(RichText::new("Cam"))
                .on_hover_text("Move camera near active molecule, looking at it.")
                .clicked()
            {
                // Setting mol center to 0 if no mol.
                move_cam_to_mol(mol, cam_snapshot, scene, pep_center, engine_updates)
            }

            let row_h = ui.spacing().interact_size.y;

            let sel_btn = ui
                .add_sized(
                    egui::vec2(ui.available_width(), row_h),
                    egui::Button::new(RichText::new(mol.name()).color(color)),
                )
                .on_hover_text(help_text);

            if sel_btn.clicked() {
                if active && active_mol.is_some() {
                    *active_mol = None;
                } else {
                    *active_mol = Some((mol_type, i_mol));
                    *orbit_center = *active_mol;

                    *recenter_orbit = true;
                }

                *redraw = true; // To reflect the change in thickness, color etc.
            }

            if sel_btn.middle_clicked() {
                *close = Some((mol_type, i_mol));
            }
        });
    });

    if let Some(char) = mol_char {
        let color = if active {
            Color32::WHITE
        } else {
            Color32::GRAY
        };

        label!(ui, char.to_string(), color);
    }
}

/// Select, close, hide etc molecules from ones opened.
fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_lipid: &mut bool,
    redraw_na: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    let mut recenter_orbit = false;
    let mut close = None; // Avoids borrow error.

    // Avoids a double borrow.
    let pep_center = match &state.peptide {
        Some(mol) => mol.center,
        None => Vec3::new_zero(),
    };

    if let Some(mol) = &mut state.peptide {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            0,
            &mut mol.common,
            &None,
            MolType::Peptide,
            scene,
            ui,
            engine_updates,
            redraw_pep,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.ligands.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &mol.characterization,
            MolType::Ligand,
            scene,
            ui,
            engine_updates,
            redraw_lig,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.lipids.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &None,
            MolType::Lipid,
            scene,
            ui,
            engine_updates,
            redraw_lipid,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    for (i_mol, mol) in state.nucleic_acids.iter_mut().enumerate() {
        // todo: Characterization, e.g. by dna seq?
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            &None,
            MolType::NucleicAcid,
            scene,
            ui,
            engine_updates,
            redraw_na,
            &mut recenter_orbit,
            &mut close,
            &mut state.ui.cam_snapshot,
            pep_center,
        );
    }

    // todo: AAs here too?

    if let Some((mol_type, i_mol)) = close {
        close_mol(mol_type, i_mol, state, scene, engine_updates);
    }

    if recenter_orbit {
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

fn open_tools(state: &mut State, ui: &mut Ui) {
    let color_open_tools = if state.peptide.is_none() && state.ligands.is_empty() {
        COLOR_ACTION
    } else {
        COLOR_INACTIVE
    };

    if ui
        .button(RichText::new("Open").color(color_open_tools))
        .on_hover_text("Open a molecule, electron density, or other file from disk.")
        .clicked()
    {
        state.volatile.dialogs.load.pick_file();
    }

    if ui
        .button(RichText::new("Recent").color(color_open_tools))
        .on_hover_text("Select a recently-opened file to open")
        .clicked()
    {
        state.ui.popup.recent_files = !state.ui.popup.recent_files;
    }
}

fn manip_toolbar(
    state: &mut State,
    scene: &mut Scene,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_lipid: &mut bool,
    redraw_na: &mut bool,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        let Some((active_mol_type, active_mol_i)) = state.volatile.active_mol else {
            return;
        };

        {
            let mut color_move = COLOR_INACTIVE;
            let mut color_rotate = COLOR_INACTIVE;

            match state.volatile.mol_manip.mode {
                ManipMode::Move((mol_type, mol_i)) => {
                    if mol_type == active_mol_type && mol_i == active_mol_i {
                        color_move = COLOR_ACTIVE;
                    }
                }
                ManipMode::Rotate((mol_type, mol_i)) => {
                    if mol_type == active_mol_type && mol_i == active_mol_i {
                        color_rotate = COLOR_ACTIVE;
                    }
                }
                ManipMode::None => (),
            }

            // ✥ doesn't work in EGUI.
            if ui.button(RichText::new("↔").color(color_move))
                .on_hover_text("(Hotkey: M. M or Esc to stop)) Move the active molecule by clicking and dragging with \
                the mouse. Scroll to move it forward and back.")
                .clicked() {

                set_manip(&mut state.volatile,&mut state.to_save.save_flag, scene, redraw_pep, redraw_lig, redraw_na, redraw_lipid,&mut false,
                          ManipMode::Move((active_mol_type, active_mol_i)), &state.ui.selection,);
            }

            if ui.button(RichText::new("⟳").color(color_rotate))
                .on_hover_text("(Hotkey: R. R or Esc to stop) Rotate the active molecule by clicking and dragging with the mouse. Scroll to roll.")
                .clicked() {

                set_manip(&mut state.volatile,&mut state.to_save.save_flag, scene, redraw_pep, redraw_lig,redraw_na, redraw_lipid,&mut false,
                          ManipMode::Rotate((active_mol_type, active_mol_i)), &state.ui.selection,);
            }
        }

        if let Some(mol) = &mut state.active_mol_mut() {
            if ui
                .button(RichText::new("Move to cam").color(COLOR_HIGHLIGHT))
                .on_hover_text("Move the molecule to be a short distance in front of the camera.")
                .clicked()
            {
                move_mol_to_cam(mol.common_mut(), &scene.camera);

                match active_mol_type {
                    MolType::Ligand => *redraw_lig = true,
                    MolType::NucleicAcid => *redraw_na = true,
                    MolType::Lipid => *redraw_lipid = true,
                    _ => unimplemented!(),
                }
            }

            if ui
                .button(RichText::new("Reset pos").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Move the molecule to its absolute coordinates, e.g. as defined in \
                        its source mmCIF, Mol2 or SDF file.",
                )
                .clicked()
            {
                mol.common_mut().reset_posits();

                // todo: Use the inplace move.
                match active_mol_type {
                    MolType::Ligand => *redraw_lig = true,
                    MolType::NucleicAcid => *redraw_na = true,
                    MolType::Lipid => *redraw_lipid = true,
                    _ => unimplemented!(),
                }
            }

            if ui.button("Details")
                .on_hover_text("Toggle the details display of the active molecule.")
                .clicked() {
                state.ui.ui_vis.mol_char = !state.ui.ui_vis.mol_char;
            }
        }
    });
}

pub(in crate::ui) fn sidebar(
    state: &mut State,
    scene: &mut Scene,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_lipid: &mut bool,
    redraw_na: &mut bool,
    engine_updates: &mut EngineUpdates,
    ctx: &Context,
) {
    let out = egui::SidePanel::left("sidebar")
        .resizable(true) // let user drag the width
        .default_width(140.0)
        .width_range(60.0..=800.0)
        .show(ctx, |ui| {
            ui.label("Molecules opened");

            ui.horizontal(|ui| {
                let color_open_tools = if state.peptide.is_none() && state.ligands.is_empty() {
                    COLOR_ACTION
                } else {
                    COLOR_INACTIVE
                };

                if ui
                    .button(RichText::new("Open").color(color_open_tools))
                    .on_hover_text("Open a molecule, electron density, or other file from disk.")
                    .clicked()
                {
                    state.volatile.dialogs.load.pick_file();
                }

                if ui
                    .button(RichText::new("Recent").color(color_open_tools))
                    .on_hover_text("Select a recently-opened file to open")
                    .clicked()
                {
                    state.ui.popup.recent_files = !state.ui.popup.recent_files;
                    open_tools(state, ui);
                }

                let mut mol_to_save = None; // avoids dbl-borrow.
                if let Some(mol) = state.active_mol() {
                    let color = Color32::GRAY;

                    if ui
                        .button(RichText::new("Save").color(color))
                        .on_hover_text("Save the active molecule to a file.")
                        .clicked()
                    {
                        mol_to_save = Some(mol.common().clone());
                    }
                }
                if let Some(mol) = mol_to_save {
                    if mol.save(&mut state.volatile.dialogs.save).is_err() {
                        handle_err(&mut state.ui, "Problem saving this file".to_owned());
                    }
                }
            });

            ui.add_space(ROW_SPACING / 2.);

            manip_toolbar(
                state,
                scene,
                redraw_pep,
                redraw_lig,
                redraw_lipid,
                redraw_na,
                ui,
            );

            ui.add_space(ROW_SPACING / 2.);
            ui.separator();
            ui.add_space(ROW_SPACING / 2.);

            // ScrollArea::vertical()
            //     .min_scrolled_height(600.0)
            //     .show(ui, |ui| {
            // todo: Function or macro to reduce this DRY.
            mol_picker(
                state,
                scene,
                ui,
                redraw_pep,
                redraw_lig,
                redraw_lipid,
                redraw_na,
                engine_updates,
            );

            // todo: UI flag to show or hide this.
            if state.ui.ui_vis.mol_char {
                let mut toggled = false; // Avoid double borrow.
                if let Some(m) = &state.active_mol() {
                    if let MolGenericRef::Small(mol) = m {
                        if ui
                            .button(RichText::new("Close details").color(Color32::LIGHT_RED))
                            .clicked()
                        {
                            toggled = true;
                        }
                        mol_char_disp(mol, ui);
                    }
                }
                if toggled {
                    state.ui.ui_vis.mol_char = !state.ui.ui_vis.mol_char;
                }
            }
        });

    engine_updates.ui_reserved_px.0 = out.response.rect.width();
}

fn char_item(ui: &mut Ui, items: &[(&str, &str)]) {
    ui.horizontal(|ui| {
        for (i, (name, v)) in items.iter().enumerate() {
            label!(ui, format!("{name}:"), Color32::GRAY);
            label!(ui, *v, Color32::WHITE);

            if i != items.len() - 1 {
                ui.add_space(COL_SPACING / 2.);
            }
        }
    });
}

fn mol_char_disp(mol: &MoleculeSmall, ui: &mut Ui) {
    let Some(char) = &mol.characterization else {
        return;
    };

    // todo: Small font?
    for ident in &mol.idents {
        ui.horizontal(|ui| {
            // Wrap long names, like InChi etc.
            ui.horizontal_wrapped(|ui| {
                label!(ui, format!("{}:", ident.label()), Color32::GRAY);

                let mut ident_txt = RichText::new(ident.ident_innner()).color(Color32::WHITE);

                // These are longer idents.
                if matches!(
                    ident,
                    MolIdent::InchIKey(_)
                        | MolIdent::InchI(_)
                        | MolIdent::Smiles(_)
                        | MolIdent::IupacName(_)
                ) {
                    let font = egui::FontId::proportional(10.0);
                    ident_txt = ident_txt.font(font);
                }

                ui.label(ident_txt);
            });
        });
    }

    ui.add_space(ROW_SPACING);
    // Basics
    char_item(
        ui,
        &[
            ("Atoms", &char.num_atoms.to_string()),
            ("Bonds", &char.num_bonds.to_string()),
            ("Heavy", &char.num_heavy_atoms.to_string()),
            ("Het", &char.num_hetero_atoms.to_string()),
        ],
    );

    char_item(ui, &[("Weight", &format!("{:.2}", char.mol_weight))]);

    ui.add_space(ROW_SPACING);
    // Functional groups

    char_item(
        ui,
        &[
            ("Rings Ar", &char.num_rings_aromatic.to_string()),
            ("Sat", &char.num_rings_saturated.to_string()),
            ("Ali", &char.num_rings_aliphatic.to_string()),
        ],
    );

    // todo: Rings
    char_item(
        ui,
        &[
            ("Amine", &char.amines.len().to_string()),
            ("Amide", &char.amides.len().to_string()),
            ("Carbonyl", &char.carbonyl.len().to_string()),
            ("Hydroxyl", &char.hydroxyl.len().to_string()),
        ],
    );

    ui.add_space(ROW_SPACING);
    // Misc properties
    char_item(
        ui,
        &[
            ("H bond donor:", &char.h_bond_donor.len().to_string()),
            ("acceptor", &char.h_bond_acceptor.len().to_string()),
            ("Valence elecs", &char.num_valence_elecs.to_string()),
        ],
    );

    // Geometry
    // ui.add_space(ROW_SPACING);

    char_item(
        ui,
        &[
            ("TPSA (Ertl)", &format!("{:.2}", char.tpsa_ertl)),
            ("PSA (Geom)", &format!("{:.2}", char.psa_topo)),
        ],
    );

    let vol = match char.volume_pubchem {
        Some(v) => v,
        None => char.volume,
    };
    char_item(
        ui,
        &[
            ("ASA (Labute)", &format!("{:.2}", char.asa_labute)),
            ("ASA (Geom)", &format!("{:.2}", char.asa_topo)),
            ("Volume", &format!("{:.2}", vol)),
        ],
    );

    // Computed properties
    ui.add_space(ROW_SPACING);

    let log_p = match char.log_p_pubchem {
        Some(v) => v,
        None => char.log_p,
    };
    char_item(
        ui,
        &[
            ("LogP", &format!("{:.2}", log_p)),
            ("Mol Refrac", &format!("{:.2}", char.molar_refractivity)),
        ],
    );

    char_item(
        ui,
        &[
            ("Balaban J", &format!("{:.2}", char.balaban_j)),
            ("Bertz Complexity", &format!("{:.2}", char.bertz_ct)),
        ],
    );

    char_item(
        ui,
        &[(
            "Complexity",
            &format!("{:.2}", char.complexity.unwrap_or(0.0)),
        )],
    );

    char_item(
        ui,
        &[(
            "Wiener index",
            &format!("{}", char.wiener_index.unwrap_or(0)),
        )],
    );

    if let Some(ther) = &mol.therapeutic_props {
        ui.add_space(ROW_SPACING);
        ui.separator();
        label!(ui, "Therapeutic data", Color32::LIGHT_BLUE);

        adme_disp(&ther.adme, ui);
        tox_disp(&ther.toxicity, ui);
    }
}

fn adme_disp(adme: &Adme, ui: &mut Ui) {
    label!(ui, "ADME:", Color32::GRAY);

    // Absorption
    char_item(
        ui,
        &[(
            "Intestinal permability",
            &format!("{:.2}", adme.intestinal_permeability),
        )],
    );
    char_item(
        ui,
        &[(
            "Intestinal absorption",
            &format!("{:.2}", adme.intestinal_absorption),
        )],
    );
    char_item(ui, &[("pgp", &format!("{:.2}", adme.pgp))]);
    char_item(
        ui,
        &[(
            "Oral_bioavailability",
            &format!("{:.2}", adme.oral_bioavailablity),
        )],
    );
    char_item(
        ui,
        &[("Lipophilicity", &format!("{:.2}", adme.lipophilicity))],
    );
    char_item(
        ui,
        &[("Solubility water", &format!("{:.2}", adme.solubility_water))],
    );
    char_item(
        ui,
        &[(
            "Membrane permeability",
            &format!("{:.2}", adme.membrane_permeability),
        )],
    );
    char_item(
        ui,
        &[(
            "Hydration free Energy",
            &format!("{:.2}", adme.hydration_free_energy),
        )],
    );

    // Distribution
    char_item(
        ui,
        &[(
            "blood_brain_barrier",
            &format!("{:.2}", adme.blood_brain_barrier),
        )],
    );
    char_item(
        ui,
        &[(
            "plasma protein binding",
            &format!("{:.2}", adme.plasma_protein_binding_rate),
        )],
    );
    char_item(ui, &[("VDSS", &format!("{:.2}", adme.vdss))]);

    // Metabolism
    char_item(
        ui,
        &[(
            "CYP P450 2C19 inhibition",
            &format!("{:.2}", adme.cyp_2c19_inhibition),
        )],
    );
    char_item(
        ui,
        &[(
            "CYP P450 2D6 inhibition",
            &format!("{:.2}", adme.cyp_2d6_inhibition),
        )],
    );

    // Excretion
    char_item(ui, &[("Half Life", &format!("{:.2}", adme.half_life))]);
    char_item(ui, &[("Clearance", &format!("{:.2}", adme.clearance))]);
}

fn tox_disp(tox: &Toxicity, ui: &mut Ui) {
    label!(ui, "Toxicity:", Color32::GRAY).on_hover_text("Of our cittyyyyyyy");

    char_item(ui, &[("LD50", &format!("{:.2}", tox.ld50))]);
    char_item(ui, &[("hERG", &format!("{:.2}", tox.ether_a_go_go))]);
    char_item(ui, &[("Mutagenicity", &format!("{:.2}", tox.mutagencity))]);
    char_item(
        ui,
        &[(
            "Liver injury",
            &format!("{:.2}", tox.drug_induced_liver_injury),
        )],
    );
    char_item(
        ui,
        &[("Skin reaction", &format!("{:.2}", tox.skin_reaction))],
    );
    char_item(ui, &[("Carcinogen", &format!("{:.2}", tox.carcinogen))]);
}
