//! Command files and macros.

use graphics::{EngineUpdates, Scene};

use crate::{cli::scripts, selection::expression, state::State, util::RedrawFlags};

#[derive(Default)]
pub struct CommandFiles {
    pub open: bool,
    message: String,
    body: String,
    macro_name: String,
    file: egui_file_dialog::FileDialog,
}

pub fn command_window(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut RedrawFlags,
    ctx: &egui::Context,
) {
    if !state.ui.command_files.open {
        return;
    }

    let mut edit = std::mem::take(&mut state.ui.command_files);
    let mut open = true;
    let mut run = None;

    edit.file.update(ctx);
    if let Some(path) = edit.file.take_picked() {
        run = Some(format!(
            "run {}",
            expression::quote(&path.to_string_lossy())
        ));
    }

    egui::Window::new("Command files / macros")
        .open(&mut open)
        .default_width(600.)
        .vscroll(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Run command file…").clicked() {
                    edit.file.pick_file();
                }

                if ui.button("Run editor").clicked() {
                    run = Some(edit.body.clone());
                }
            });

            ui.small("Commands use lines or semicolons. # starts comments. File paths are relative to the command file.");
            ui.add(
                egui::TextEdit::multiline(&mut edit.body)
                    .code_editor()
                    .desired_rows(10)
                    .desired_width(f32::INFINITY)
                    .hint_text("show sticks, sele\ncolor yellow, sele\nlabel sele, chain_residue\nzoom sele\nclip slab, 10, sele"),
            );

            ui.horizontal(|ui| {
                ui.label("Macro name:");
                ui.text_edit_singleline(&mut edit.macro_name);

                if ui.button("Save macro").clicked() {
                    edit.message = scripts::store_macro(state, &edit.macro_name, &edit.body)
                        .map(|_| "Macro saved".into())
                        .unwrap_or_else(|e| e.to_string());
                }
            });

            ui.small("Use $selection in a macro; invoke its name with an optional selection expression.");

            let mut remove = None;
            for (i, mac) in state.to_save.command_macros.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(&mac.name);

                    if ui.small_button("Run on selection").clicked() {
                        run = Some(mac.name.clone());
                    }

                    if ui.small_button("Edit").clicked() {
                        edit.macro_name = mac.name.clone();
                        edit.body = mac.body.clone();
                    }

                    if ui.small_button("Delete").clicked() {
                        remove = Some(i);
                    }
                });
            }

            if let Some(i) = remove {
                state.to_save.command_macros.remove(i);
                state.to_save.save_flag = true;
            }

            if !edit.message.is_empty() {
                ui.label(&edit.message);
            }
        });

    if let Some(text) = run {
        let mut changed = false;
        edit.message = scripts::execute(state, scene, updates, &mut changed, &text)
            .unwrap_or_else(|e| e.to_string());

        if changed {
            redraw.set_all();
        }
    }

    edit.open = open;
    state.ui.command_files = edit;
}
