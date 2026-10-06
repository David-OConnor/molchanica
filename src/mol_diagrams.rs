//! Skeletal molecule diagrams for the reaction viewer.
//!
//! Prefer the local RDKit installation configured in Tools. Callers may supply a static SMILES;
//! otherwise this route fetches one through bio_apis before drawing locally. Without a working
//! RDKit installation, bio_apis::chebi supplies ChEBI's remote SVG. Raw remote SVGs use a bounded
//! persistent cache; recoloring and rasterization still happen locally for the current theme.
//!
//! SVG is the interchange format; resvg rasterizes it off the UI thread at double the
//! display resolution. egui owns the resulting textures. HTTP belongs in bio_apis;
//! the local depiction implementation belongs in bio_tools::rdkit.

use std::{
    borrow::Cow,
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::{Duration, SystemTime},
};

use egui::{Color32, ColorImage, Context, TextureHandle, TextureOptions, Ui, vec2};

use crate::external_tools::{Tool, find_executable, find_rdkit_python};

pub const DIAGRAM_WIDTH: f32 = 75.0;
const DIAGRAM_HEIGHT: f32 = 45.0;
const MAX_DOWNLOADS: usize = 4;
const CACHE_CAPACITY: usize = 128;
const REMOTE_CACHE_CAPACITY: usize = 256;
const REMOTE_CACHE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const REMOTE_CACHE_DIRECTORY: &str = "molecule_diagrams/chebi-svg-v1";

static REMOTE_CACHE_TEMP_ID: AtomicU64 = AtomicU64::new(0);
static REMOTE_CACHE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug)]
pub struct DiagramRequest {
    chebi_id: u32,
    smiles: Option<&'static str>,
    retry: bool,
}

impl DiagramRequest {
    pub fn new(chebi_id: u32, smiles: Option<&'static str>, retry: bool) -> Self {
        Self {
            chebi_id,
            smiles,
            retry,
        }
    }
}

enum Entry {
    Loading(Receiver<Result<Option<ColorImage>, String>>),
    Ready(TextureHandle),
    Unavailable,
    Failed(String),
}

enum RdKitDiscovery {
    Unresolved,
    Discovering(Receiver<Option<PathBuf>>),
    Ready(Option<PathBuf>),
}

impl Default for RdKitDiscovery {
    fn default() -> Self {
        Self::Unresolved
    }
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
    rdkit: RdKitDiscovery,
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
    pub fn request(&mut self, participants: &[DiagramRequest], ctx: &Context) {
        let foreground = ctx.global_style().visuals.strong_text_color();
        if self.foreground != Some(foreground) {
            // Rasterized images contain the current text color, so a theme change invalidates them.
            self.entries.clear();
            self.foreground = Some(foreground);
        }

        // System RDKit discovery may need to probe Python interpreters. Finish that once before
        // launching any diagram jobs, so discovering the interpreter cannot invalidate and
        // duplicate a page of work on the next frame.
        let Some(rdkit_python) = self.poll_rdkit_python(ctx) else {
            return;
        };
        self.generation += 1;

        // Mark all displayed entries before eviction, including ones later in the list.
        for request in participants {
            if request.retry
                && matches!(
                    self.entries.get(&request.chebi_id).map(|c| &c.entry),
                    Some(Entry::Failed(_))
                )
            {
                self.entries.remove(&request.chebi_id);
            }
            if let Some(cached) = self.entries.get_mut(&request.chebi_id) {
                cached.last_used = self.generation;
            }
        }

        let mut downloading = self
            .entries
            .values()
            .filter(|cached| matches!(cached.entry, Entry::Loading(_)))
            .count();

        for request in participants {
            let id = request.chebi_id;
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
            let rdkit_python = rdkit_python.clone();
            let smiles = request.smiles;
            thread::spawn(move || {
                let _ = tx.send(load_chebi_diagram(
                    id,
                    smiles,
                    rdkit_python.as_deref(),
                    foreground,
                ));
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

    /// The outer `Option` is `None` while one asynchronous discovery is in progress. The inner
    /// `Option` is `None` when discovery completed without finding RDKit.
    fn poll_rdkit_python(&mut self, ctx: &Context) -> Option<Option<PathBuf>> {
        let directly_resolved = find_executable(Tool::RdKit).ok();
        let state = std::mem::take(&mut self.rdkit);

        match state {
            RdKitDiscovery::Unresolved => {
                if let Some(python) = directly_resolved {
                    self.rdkit = RdKitDiscovery::Ready(Some(python.clone()));
                    Some(Some(python))
                } else {
                    self.start_rdkit_discovery(ctx);
                    None
                }
            }
            RdKitDiscovery::Discovering(rx) => {
                if let Some(python) = directly_resolved {
                    self.rdkit = RdKitDiscovery::Ready(Some(python.clone()));
                    return Some(Some(python));
                }

                match rx.try_recv() {
                    Ok(python) => {
                        self.rdkit = RdKitDiscovery::Ready(python.clone());
                        Some(python)
                    }
                    Err(TryRecvError::Empty) => {
                        self.rdkit = RdKitDiscovery::Discovering(rx);
                        ctx.request_repaint_after(Duration::from_millis(100));
                        None
                    }
                    Err(TryRecvError::Disconnected) => {
                        self.rdkit = RdKitDiscovery::Ready(None);
                        Some(None)
                    }
                }
            }
            RdKitDiscovery::Ready(current) if directly_resolved == current => {
                self.rdkit = RdKitDiscovery::Ready(current.clone());
                Some(current)
            }
            RdKitDiscovery::Ready(current) => {
                self.entries.clear();

                if let Some(python) = directly_resolved {
                    self.rdkit = RdKitDiscovery::Ready(Some(python.clone()));
                    Some(Some(python))
                } else if current.is_some() {
                    // A managed or explicitly configured interpreter disappeared. Search once for
                    // a system installation before using the remote fallback.
                    self.start_rdkit_discovery(ctx);
                    None
                } else {
                    self.rdkit = RdKitDiscovery::Ready(None);
                    Some(None)
                }
            }
        }
    }

    fn start_rdkit_discovery(&mut self, ctx: &Context) {
        let (tx, rx) = mpsc::channel();
        let worker_ctx = ctx.clone();
        thread::spawn(move || {
            let _ = tx.send(find_rdkit_python().ok());
            worker_ctx.request_repaint();
        });
        self.rdkit = RdKitDiscovery::Discovering(rx);
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

/// Choose the depiction backend and rasterize in a worker. Failures of a configured local
/// installation are logged before falling back, so a broken environment doesn't hide diagrams.
fn load_chebi_diagram(
    id: u32,
    provided_smiles: Option<&str>,
    rdkit_python: Option<&Path>,
    foreground: Color32,
) -> Result<Option<ColorImage>, String> {
    if let Some(python) = rdkit_python {
        match load_local_diagram(id, provided_smiles, python, foreground) {
            Ok(Some(image)) => return Ok(Some(image)),
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

    if let Some((svg, path)) = load_cached_remote_diagram(id, 300, 180) {
        match render_svg(&svg, foreground) {
            Ok(image) => {
                println!(
                    "Molecule diagram CHEBI:{id}: loaded from the persistent ChEBI diagram cache \
                     (originally downloaded, not computed locally with RDKit)"
                );
                return Ok(Some(image));
            }
            Err(error) => {
                println!(
                    "Molecule diagram CHEBI:{id}: ignoring invalid cached ChEBI diagram: {error}"
                );
                if let Err(remove_error) = remove_cached_remote_diagram(&path) {
                    eprintln!(
                        "Unable to remove invalid molecule-diagram cache file {}: {remove_error}",
                        path.display()
                    );
                }
            }
        }
    }

    println!(
        "Molecule diagram CHEBI:{id}: requesting the remote ChEBI diagram \
         (not computed locally with RDKit)"
    );
    let Some(svg) = bio_apis::chebi::load_diagram(id, 300, 180)
        .map_err(|error| format!("CHEBI:{id}: {error:?}"))?
    else {
        return Ok(None);
    };
    let image = render_svg(&svg, foreground)?;
    if let Err(error) = store_remote_diagram(id, 300, 180, &svg) {
        eprintln!("Unable to cache the remote CHEBI:{id} diagram: {error}");
    }
    println!(
        "Molecule diagram CHEBI:{id}: loaded from the remote ChEBI service \
         (not computed locally with RDKit)"
    );
    Ok(Some(image))
}

fn remote_diagram_cache_path(id: u32, width: u32, height: u32) -> Option<PathBuf> {
    Some(
        crate::external_tools::data_root()?
            .join(REMOTE_CACHE_DIRECTORY)
            .join(format!("chebi-{id}-{width}x{height}.svg")),
    )
}

fn load_cached_remote_diagram(id: u32, width: u32, height: u32) -> Option<(Vec<u8>, PathBuf)> {
    let _cache_guard = REMOTE_CACHE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = remote_diagram_cache_path(id, width, height)?;
    fs::read(&path).ok().map(|svg| (svg, path))
}

fn remove_cached_remote_diagram(path: &Path) -> io::Result<()> {
    let _cache_guard = REMOTE_CACHE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    fs::remove_file(path)
}

fn store_remote_diagram(id: u32, width: u32, height: u32, svg: &[u8]) -> io::Result<()> {
    let _cache_guard = REMOTE_CACHE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(path) = remote_diagram_cache_path(id, width, height) else {
        return Ok(());
    };
    let Some(directory) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(directory)?;

    let temp_id = REMOTE_CACHE_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let temp = directory.join(format!(".chebi-{id}-{}-{temp_id}.tmp", std::process::id()));
    fs::write(&temp, svg)?;

    if let Err(error) = fs::rename(&temp, &path) {
        if path.exists() {
            // Another diagram cache may have stored the same ChEBI entry concurrently.
            let _ = fs::remove_file(&temp);
        } else {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
    }

    prune_remote_diagram_cache(directory)
}

fn prune_remote_diagram_cache(directory: &Path) -> io::Result<()> {
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "svg")
        {
            files.push((
                entry.path(),
                metadata.len(),
                metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            ));
        }
    }

    files.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
    let mut total_bytes: u64 = files.iter().map(|(_, size, _)| size).sum();
    let mut first_retained = 0;

    while files.len() - first_retained > REMOTE_CACHE_CAPACITY
        || total_bytes > REMOTE_CACHE_MAX_BYTES
    {
        let (path, size, _) = &files[first_retained];
        fs::remove_file(path)?;
        total_bytes = total_bytes.saturating_sub(*size);
        first_retained += 1;
    }

    Ok(())
}

fn load_local_diagram(
    id: u32,
    provided_smiles: Option<&str>,
    python: &Path,
    foreground: Color32,
) -> Result<Option<ColorImage>, String> {
    let smiles = if let Some(smiles) = provided_smiles {
        println!("Molecule diagram CHEBI:{id}: using embedded synthesis-library SMILES");
        Cow::Borrowed(smiles)
    } else {
        println!(
            "Molecule diagram CHEBI:{id}: fetching ChEBI SMILES for local RDKit \
             (not a diagram request)"
        );
        let compound = bio_apis::chebi::load_compound(id)
            .map_err(|error| format!("Unable to load CHEBI:{id} SMILES: {error:?}"))?;
        let Some(smiles) = compound
            .default_structure
            .and_then(|structure| structure.smiles)
        else {
            return Ok(None);
        };
        Cow::Owned(smiles)
    };
    let svg = bio_tools::rdkit::depict_smiles(python, &smiles, 300, 180)
        .map_err(|error| error.to_string())?;
    let source = if provided_smiles.is_some() {
        "embedded synthesis-library SMILES"
    } else {
        "ChEBI SMILES"
    };
    println!(
        "Molecule diagram CHEBI:{id}: generated from {source} using RDKit ({})",
        python.display()
    );
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
