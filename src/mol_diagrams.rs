//! Skeletal molecule diagrams for the reaction viewer.
//!
//! ChEBI's public compound/structure endpoint supplies RDKit-generated SVGs, including
//! outlined atom labels and stereochemical bonds. Reusing these avoids a local chemistry
//! runtime and preserves the structure associated with the button's exact ChEBI ID.
//! API: https://www.ebi.ac.uk/chebi/backend/api/docs/
//!
//! SVG is the interchange format; resvg rasterizes it off the UI thread at double the
//! display resolution. egui owns the resulting textures. This module deliberately takes
//! ChEBI IDs rather than SMILES: it displays existing depictions, not arbitrary molecules.

use std::{
    collections::HashMap,
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::Duration,
};

use egui::{ColorImage, Context, TextureHandle, TextureOptions, Ui, vec2};

pub const DIAGRAM_WIDTH: f32 = 300.0;
const DIAGRAM_HEIGHT: f32 = 180.0;
const MAX_DOWNLOADS: usize = 4;
const CACHE_CAPACITY: usize = 128;

enum Entry {
    Loading(Receiver<Result<Option<ColorImage>, String>>),
    Ready(TextureHandle),
    Unavailable,
    Failed(String),
}

struct CachedDiagram {
    entry: Entry,
    last_used: u64,
}

/// Shared across reactions and queries; only participants on the current page are requested.
/// Old textures are released when the bounded session cache fills up.
#[derive(Default)]
pub struct DiagramCache {
    entries: HashMap<u32, CachedDiagram>,
    generation: u64,
}

impl DiagramCache {
    /// Upload completed images on the UI thread. Downloads finish safely with the popup closed;
    /// their results are collected next time it opens.
    pub fn poll(&mut self, ctx: &Context) {
        for (&id, cached) in &mut self.entries {
            let Entry::Loading(rx) = &cached.entry else {
                continue;
            };

            cached.entry = match rx.try_recv() {
                Ok(Ok(Some(image))) => Entry::Ready(ctx.load_texture(
                    format!("chebi-diagram-{id}"),
                    image,
                    TextureOptions::LINEAR,
                )),
                Ok(Ok(None)) => Entry::Unavailable,
                Ok(Err(error)) => Entry::Failed(error),
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => {
                    Entry::Failed("The diagram download stopped unexpectedly.".to_owned())
                }
            };
        }
    }

    /// Draw a stable-size slot below the molecule button. Return whether Retry was clicked.
    pub fn show(&self, id: u32, ui: &mut Ui) -> bool {
        let width = ui.available_width().min(DIAGRAM_WIDTH).max(1.0);
        let size = vec2(width, width * DIAGRAM_HEIGHT / DIAGRAM_WIDTH);
        let mut retry = false;

        ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Center), |ui| {
            ui.set_min_size(size);
            match self.entries.get(&id).map(|cached| &cached.entry) {
                Some(Entry::Ready(texture)) => {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(size))
                        .on_hover_text(format!("2D structure from ChEBI — CHEBI:{id}"));
                }
                Some(Entry::Unavailable) => {
                    ui.weak("No 2D structure available from ChEBI.");
                }
                Some(Entry::Failed(error)) => {
                    ui.weak("Could not load the 2D structure.")
                        .on_hover_text(error);
                    retry = ui.small_button("Retry diagram").clicked();
                }
                entry => {
                    ui.spinner();
                    ui.weak(if entry.is_some() {
                        "Loading 2D structure…"
                    } else {
                        "Waiting for 2D structure…"
                    });
                }
            }
        });

        retry
    }

    /// Called after drawing so pagination and retries apply to the page actually on screen.
    /// Unstarted requests are not queued: switching pages gives the new page priority.
    pub fn request(&mut self, participants: &[(u32, bool)], ctx: &Context) {
        self.generation += 1;

        // Mark all displayed entries before eviction, including ones later in the list.
        for &(id, retry) in participants {
            if retry
                && matches!(
                    self.entries.get(&id).map(|c| &c.entry),
                    Some(Entry::Failed(_))
                )
            {
                self.entries.remove(&id);
            }
            if let Some(cached) = self.entries.get_mut(&id) {
                cached.last_used = self.generation;
            }
        }

        let mut downloading = self
            .entries
            .values()
            .filter(|cached| matches!(cached.entry, Entry::Loading(_)))
            .count();

        for &(id, _) in participants {
            if self.entries.contains_key(&id) || downloading >= MAX_DOWNLOADS {
                continue;
            }

            if self.entries.len() >= CACHE_CAPACITY {
                let oldest = self
                    .entries
                    .iter()
                    .filter(|(_, cached)| {
                        cached.last_used != self.generation
                            && !matches!(cached.entry, Entry::Loading(_))
                    })
                    .min_by_key(|(_, cached)| cached.last_used)
                    .map(|(&id, _)| id);

                if let Some(oldest) = oldest {
                    self.entries.remove(&oldest);
                } else {
                    continue;
                }
            }

            let (tx, rx) = mpsc::channel();
            let ctx = ctx.clone();
            thread::spawn(move || {
                let _ = tx.send(load_chebi_diagram(id));
                ctx.request_repaint();
            });
            self.entries.insert(
                id,
                CachedDiagram {
                    entry: Entry::Loading(rx),
                    last_used: self.generation,
                },
            );
            downloading += 1;
        }

        if downloading > 0 {
            // Also handles a worker disconnecting before it can request a repaint.
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

/// Fetch and render the SVG in a worker. ChEBI returns 404 for entities without structures.
fn load_chebi_diagram(id: u32) -> Result<Option<ColorImage>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(15)))
        .build()
        .into();
    let url = format!(
        "https://www.ebi.ac.uk/chebi/backend/api/public/compound/{id}/structure/?width=300&height=180"
    );

    let mut response = match agent.get(&url).call() {
        Ok(response) => response,
        Err(ureq::Error::StatusCode(404)) => return Ok(None),
        Err(error) => return Err(format!("CHEBI:{id}: {error}")),
    };
    let svg = response
        .body_mut()
        .with_config()
        .limit(2 * 1024 * 1024)
        .read_to_vec()
        .map_err(|error| format!("CHEBI:{id}: {error}"))?;

    if svg.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }

    // The server outlines text, so fonts and a system font scan are unnecessary.
    egui_extras::image::load_svg_bytes_with_size(
        &svg,
        egui::load::SizeHint::Size {
            width: 600,
            height: 360,
            maintain_aspect_ratio: true,
        },
        &Default::default(),
    )
    .map(Some)
    .map_err(|error| format!("Unable to render CHEBI:{id}: {error}"))
}
