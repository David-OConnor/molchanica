use egui::{Event, Id, Key, Modal, Modifiers, ScrollArea, TextEdit, Ui};

pub(super) type Command<T> = (T, &'static str, &'static str, bool);

#[derive(Clone, Default)]
struct Palette {
    open: bool,
    query: String,
    selected: usize,
}

pub(super) fn show<T: Copy>(
    ui: &mut Ui,
    id: Id,
    title: &str,
    commands: impl FnOnce() -> Vec<Command<T>>,
) -> Option<T> {
    let clicked = ui.button("Search actions · F3").clicked();
    let shortcut = !ui.ctx().text_edit_focused()
        && ui.memory(|mem| mem.top_modal_layer().is_none())
        && ui.input_mut(|input| {
            let pressed = input.events.iter().any(|event| {
                matches!(
                    event,
                    Event::Key { key: Key::F3, pressed: true, repeat: false, modifiers, .. }
                        if modifiers.is_none()
                )
            });

            pressed && input.consume_key(Modifiers::NONE, Key::F3)
        });

    let mut palette = ui.data(|data| data.get_temp::<Palette>(id).unwrap_or_default());
    let opening = clicked || shortcut;
    if opening {
        palette = Palette {
            open: true,
            ..Default::default()
        };
    }
    if !palette.open {
        return None;
    }

    let query_id = id.with("query");
    let commands = commands();
    let modal = Modal::new(id).show(ui.ctx(), |ui| {
        let (up, down, enter) = ui.input_mut(|input| {
            (
                input.consume_key(Modifiers::NONE, Key::ArrowUp),
                input.consume_key(Modifiers::NONE, Key::ArrowDown),
                input.consume_key(Modifiers::NONE, Key::Enter),
            )
        });

        ui.set_min_width(380.);
        ui.horizontal(|ui| {
            ui.heading(title);
            if ui.button("Close").clicked() {
                ui.close();
            }
        });
        let response = ui.add(
            TextEdit::singleline(&mut palette.query)
                .id(query_id)
                .hint_text("Search commands…")
                .desired_width(f32::INFINITY),
        );
        if opening {
            response.request_focus();
        }
        let query_changed = response.changed();
        if query_changed {
            palette.selected = 0;
        }

        let query = palette.query.to_lowercase();
        let matches: Vec<_> = commands
            .iter()
            .filter(|(_, label, _, enabled)| {
                *enabled && {
                    let label = label.to_lowercase();
                    query.split_whitespace().all(|word| label.contains(word))
                }
            })
            .collect();
        let last = matches.len().saturating_sub(1);
        palette.selected = palette.selected.min(last);
        if up {
            palette.selected = palette.selected.saturating_sub(1);
        }
        if down {
            palette.selected = (palette.selected + 1).min(last);
        }

        let mut chosen = None;
        if enter && let Some(command) = matches.get(palette.selected) {
            chosen = Some(command.0);
        }

        ScrollArea::vertical()
            .max_height(300.)
            .auto_shrink([true, false])
            .show(ui, |ui| {
                for (i, (action, label, shortcut, _)) in matches.iter().enumerate() {
                    let text = if shortcut.is_empty() {
                        (*label).to_owned()
                    } else {
                        format!("{label}    {shortcut}")
                    };
                    let selected = palette.selected == i;
                    let response = ui.selectable_label(selected, text);
                    if response.clicked() {
                        chosen = Some(*action);
                    }
                    if selected && (opening || up || down || query_changed) {
                        response.scroll_to_me(None);
                    }
                }
                if matches.is_empty() {
                    ui.label("No available commands match.");
                }
            });
        ui.small("Up/Down choose · Enter run · Esc close");

        chosen
    });

    let chosen = modal.inner;
    palette.open = !modal.should_close() && chosen.is_none();
    if !palette.open {
        ui.memory_mut(|mem| mem.surrender_focus(query_id));
        ui.request_repaint();
    }
    ui.data_mut(|data| data.insert_temp(id, palette));

    chosen
}
