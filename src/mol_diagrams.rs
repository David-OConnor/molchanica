//! Skeletal molecule diagrams for the reaction viewer.
//!
//! Prefer the local RDKit installation configured in Tools. Reaction participants only carry
//! ChEBI IDs, so this route still fetches SMILES via bio_apis before drawing locally.
//! Without a working RDKit installation, bio_apis::chebi supplies ChEBI's remote SVG.
//!
//! SVG is the interchange format; resvg rasterizes it off the UI thread at double the
//! display resolution. egui owns the resulting textures. HTTP belongs in bio_apis;
//! the local depiction implementation belongs in bio_tools::rdkit.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::Duration,
};

use egui::{Color32, ColorImage, Context, TextureHandle, TextureOptions, Ui, vec2};

use crate::external_tools::{Tool, find_executable, find_rdkit_python};

pub const DIAGRAM_WIDTH: f32 = 75.0;
const DIAGRAM_HEIGHT: f32 = 45.0;
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
    rdkit_python: Option<PathBuf>,
    foreground: Option<Color32>,
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
                        .on_hover_text(format!("2D structure for CHEBI:{id}"));
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
        let rdkit_python = find_executable(Tool::RdKit).ok();
        let foreground = ctx.global_style().visuals.strong_text_color();
        if rdkit_python != self.rdkit_python || self.foreground != Some(foreground) {
            // Installing/uninstalling RDKit in Tools changes the backend without a restart.
            self.entries.clear();
            self.rdkit_python = rdkit_python;
            self.foreground = Some(foreground);
        }
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
            let rdkit_python = self.rdkit_python.clone();
            thread::spawn(move || {
                let python = rdkit_python.or_else(|| find_rdkit_python().ok());
                let _ = tx.send(load_chebi_diagram(id, python.as_deref(), foreground));
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

/// Choose the depiction backend and rasterize in a worker. Failures of a configured local
/// installation are logged before falling back, so a broken environment doesn't hide diagrams.
fn load_chebi_diagram(
    id: u32,
    rdkit_python: Option<&Path>,
    foreground: Color32,
) -> Result<Option<ColorImage>, String> {
    if let Some(python) = rdkit_python {
        match load_local_diagram(id, python, foreground) {
            Ok(Some(image)) => {
                println!(
                    "Molecule diagram CHEBI:{id}: built locally with RDKit ({})",
                    python.display()
                );
                return Ok(Some(image));
            }
            Ok(None) => {
                println!(
                    "Molecule diagram CHEBI:{id}: local RDKit selected, but ChEBI has no SMILES"
                );
                // ChEBI may still have a drawable Molfile for a generic structure.
            }
            Err(error) => {
                println!(
                    "Molecule diagram CHEBI:{id}: local RDKit failed; using remote ChEBI diagram: {error}"
                );
            }
        }
    }

    println!("Molecule diagram CHEBI:{id}: calling remote ChEBI diagram endpoint");
    let Some(svg) = bio_apis::chebi::load_diagram(id, 300, 180)
        .map_err(|error| format!("CHEBI:{id}: {error:?}"))?
    else {
        return Ok(None);
    };
    render_svg(&svg, foreground).map(Some)
}

fn load_local_diagram(
    id: u32,
    python: &Path,
    foreground: Color32,
) -> Result<Option<ColorImage>, String> {
    println!(
        "Molecule diagram CHEBI:{id}: fetching ChEBI SMILES for local RDKit (not a diagram request)"
    );
    let compound = bio_apis::chebi::load_compound(id)
        .map_err(|error| format!("Unable to load CHEBI:{id} SMILES: {error:?}"))?;
    let Some(smiles) = compound
        .default_structure
        .and_then(|structure| structure.smiles)
    else {
        return Ok(None);
    };
    let svg = bio_tools::rdkit::depict_smiles(python, &smiles, 300, 180)
        .map_err(|error| error.to_string())?;
    render_svg(&svg, foreground).map(Some)
}

fn render_svg(svg: &[u8], foreground: Color32) -> Result<ColorImage, String> {
    // RDKit SVGs (local and ChEBI) use hex paints in inline styles and attributes.
    // Remove the white paint and recolor the remaining bonds/outlined labels. Applying this
    // before rasterization preserves antialiasing against the actual panel background.
    static PAINT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let paint = PAINT.get_or_init(|| {
        regex::Regex::new(r##"(?i)\b(fill|stroke)(\s*[:=]\s*['"]?)(#[0-9a-f]{6})\b"##).unwrap()
    });
    let svg = std::str::from_utf8(svg).map_err(|error| error.to_string())?;
    let color = format!(
        "#{:02X}{:02X}{:02X}",
        foreground.r(),
        foreground.g(),
        foreground.b()
    );
    let styled = paint.replace_all(svg, |captures: &regex::Captures<'_>| {
        let replacement = if captures[3].eq_ignore_ascii_case("#FFFFFF") {
            "none"
        } else {
            &color
        };
        format!("{}{}{replacement}", &captures[1], &captures[2])
    });

    // Both backends outline text, so fonts and a system font scan are unnecessary.
    egui_extras::image::load_svg_bytes_with_size(
        styled.as_bytes(),
        egui::load::SizeHint::Size {
            width: (2.0 * DIAGRAM_WIDTH) as u32,
            height: (2.0 * DIAGRAM_HEIGHT) as u32,
            maintain_aspect_ratio: true,
        },
        &Default::default(),
    )
    .map_err(|error| format!("Unable to render molecule SVG: {error}"))
}
