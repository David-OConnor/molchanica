//! Publication figures use a frozen scene and their own offscreen target. No window pixels,
//! editor controls, selection outlines, or manipulation gizmos enter the exported image.

mod renderer;
mod worker;

use std::{
    collections::BTreeMap,
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use egui::{Color32, Context, DragValue, Slider, TextureHandle, TextureOptions};
use graphics::Scene;
use lin_alg::f32::Vec3;
use serde::{Deserialize, Serialize};
pub(crate) use worker::worker_main;

use crate::{selection::expression, state::State};

/// Stage next to the destination, so a failed write leaves an existing figure/style file intact.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("A file name is required")?
        .to_string_lossy();
    let temporary = parent.join(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = (|| {
        let mut writer = BufWriter::new(file);
        writer.write_all(bytes)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(|e| e.to_string())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
#[serde(default)]
pub(super) struct Settings {
    width: u32,
    height: u32,
    dpi: u32,
    samples: u32,
    sphere_detail: u32,
    transparent: bool,
    background: [f32; 3],
    labels: bool,
    label_scale: f32,
    label_color: [u8; 3],
    ambient: f32,
    key: f32,
    fill: f32,
    specular: f32,
    light_angle: f32,
    occlusion: f32,
    outlines: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            width: 2400,
            height: 1800,
            dpi: 300,
            samples: 2,
            sphere_detail: 4,
            transparent: false,
            background: [1.; 3],
            labels: true,
            label_scale: 1.,
            label_color: [25; 3],
            ambient: 0.3,
            key: 0.65,
            fill: 0.25,
            specular: 0.2,
            light_angle: -35.,
            occlusion: 0.25,
            outlines: 0.,
        }
    }
}

impl Settings {
    fn validate(&self) -> Result<(), String> {
        if !(64..=8192).contains(&self.width) || !(64..=8192).contains(&self.height) {
            return Err("Each dimension must be between 64 and 8192 pixels.".into());
        }
        if ![1, 2, 3].contains(&self.samples)
            || ![3, 4].contains(&self.sphere_detail)
            || !(36..=2400).contains(&self.dpi)
        {
            return Err("Invalid antialiasing or DPI setting.".into());
        }
        let pixels = self.width as u64 * self.height as u64 * self.samples.pow(2) as u64;
        if pixels > 32_000_000 {
            return Err(
                "Limit: 32 million render samples. Reduce dimensions or antialiasing.".into(),
            );
        }
        for (value, min, max) in [
            (self.label_scale, 0.25, 4.),
            (self.ambient, 0., 1.),
            (self.key, 0., 2.),
            (self.fill, 0., 1.),
            (self.specular, 0., 1.),
            (self.light_angle, -180., 180.),
            (self.occlusion, 0., 1.),
            (self.outlines, 0., 3.),
        ]
        .into_iter()
        .chain(self.background.map(|v| (v, 0., 1.)))
        {
            if !value.is_finite() || !(min..=max).contains(&value) {
                return Err("Figure style contains an invalid value.".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, bincode::Encode, bincode::Decode)]
pub(super) struct Measurement {
    a: Vec3,
    b: Vec3,
}

pub(super) struct Figure {
    settings: Settings,
    rgba: Vec<u8>,
}

impl Figure {
    fn save(&self, path: &Path) -> Result<(), String> {
        // Encode before opening the destination: a rendering/encoding failure cannot truncate it.
        let mut bytes = Vec::new();
        {
            let mut encoder =
                png::Encoder::new(&mut bytes, self.settings.width, self.settings.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
            let ppm = (self.settings.dpi as f64 / 0.0254).round() as u32;
            encoder.set_pixel_dims(Some(png::PixelDimensions {
                xppu: ppm,
                yppu: ppm,
                unit: png::Unit::Meter,
            }));
            let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
            writer
                .write_image_data(&self.rgba)
                .map_err(|e| e.to_string())?;
            writer.finish().map_err(|e| e.to_string())?;
        }
        atomic_write(path, &bytes)
    }
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct StyleLibrary {
    settings: Settings,
    styles: BTreeMap<String, Settings>,
}

#[derive(Default)]
pub struct Publication {
    pub open: bool,
    library: StyleLibrary,
    loaded: bool,
    style_name: String,
    measurements: Vec<Measurement>,
    endpoint: Option<Vec3>,
    render_rx: Option<mpsc::Receiver<RenderEvent>>,
    render_cancel: Option<Arc<AtomicBool>>,
    render_started: Option<Instant>,
    save_rx: Option<mpsc::Receiver<Result<PathBuf, String>>>,
    figure: Option<Arc<Figure>>,
    pending_save: Option<Arc<Figure>>,
    preview: Option<TextureHandle>,
    dialog: egui_file_dialog::FileDialog,
    status: String,
}

enum RenderEvent {
    Progress(String),
    Finished(Result<Figure, String>),
}

impl Drop for Publication {
    fn drop(&mut self) {
        if let Some(cancel) = &self.render_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl Publication {
    fn persist(&mut self, dir: &Path) {
        let result = (|| -> Result<(), String> {
            fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let bytes = serde_json::to_vec_pretty(&self.library).map_err(|e| e.to_string())?;
            atomic_write(&dir.join("figure_styles.json"), &bytes)
        })();
        if let Err(e) = result {
            self.status = format!("Could not save figure styles: {e}");
        }
    }

    fn poll(&mut self, ctx: &Context) {
        while let Some(rx) = &self.render_rx {
            match rx.try_recv() {
                Ok(RenderEvent::Progress(status)) => self.status = status,
                Ok(RenderEvent::Finished(result)) => {
                    self.render_rx = None;
                    self.render_cancel = None;
                    self.render_started = None;
                    match result {
                        Ok(figure) => {
                            // Keep the GPU preview bounded; retain the exact full-resolution PNG pixels.
                            let stride = figure
                                .settings
                                .width
                                .max(figure.settings.height)
                                .div_ceil(1200);
                            let w = figure.settings.width.div_ceil(stride);
                            let h = figure.settings.height.div_ceil(stride);
                            let mut thumbnail = Vec::with_capacity((w * h * 4) as usize);
                            for y in 0..h {
                                for x in 0..w {
                                    let i = ((y * stride * figure.settings.width + x * stride) * 4)
                                        as usize;
                                    thumbnail.extend_from_slice(&figure.rgba[i..i + 4]);
                                }
                            }
                            self.preview = Some(ctx.load_texture(
                                "publication-preview",
                                egui::ColorImage::from_rgba_unmultiplied(
                                    [w as usize, h as usize],
                                    &thumbnail,
                                ),
                                TextureOptions::LINEAR,
                            ));
                            self.figure = Some(Arc::new(figure));
                            self.status =
                                "Preview ready. Save PNG writes these exact rendered pixels."
                                    .into();
                        }
                        Err(e) => self.status = format!("Render failed: {e}"),
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.render_rx = None;
                    self.render_cancel = None;
                    self.render_started = None;
                    self.status = "The render worker stopped unexpectedly.".into();
                }
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        if let Some(rx) = &self.save_rx {
            match rx.try_recv() {
                Ok(result) => {
                    self.save_rx = None;
                    self.status = match result {
                        Ok(path) => format!("Saved {}", path.display()),
                        Err(e) => format!("Save failed: {e}"),
                    };
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.save_rx = None;
                    self.status = "The PNG worker stopped unexpectedly.".into();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.render_rx.is_some() || self.save_rx.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

pub fn show(state: &mut State, scene: &Scene, ctx: &Context) {
    let selected = if state.publication.open {
        let points = expression::positions(state);
        (points.len() == 1)
            .then(|| Vec3::new(points[0].x as f32, points[0].y as f32, points[0].z as f32))
    } else {
        None
    };
    let prefs_dir = &state.volatile.prefs_dir;
    let p = &mut state.publication;
    p.poll(ctx);
    p.dialog.update(ctx);
    if let Some(path) = p.dialog.take_picked()
        && let Some(figure) = p.pending_save.take()
    {
        let (tx, rx) = mpsc::channel();
        p.save_rx = Some(rx);
        p.status = "Writing PNG…".into();
        std::thread::spawn(move || {
            let result = figure.save(&path).map(|()| path);
            let _ = tx.send(result);
        });
    }
    if !p.open {
        return;
    }
    if !p.loaded {
        p.loaded = true;
        match fs::read(prefs_dir.join("figure_styles.json")) {
            Ok(bytes) => match serde_json::from_slice::<StyleLibrary>(&bytes) {
                Ok(library)
                    if library.settings.validate().is_ok()
                        && library.styles.values().all(|s| s.validate().is_ok()) =>
                {
                    p.library = library
                }
                _ => p.status = "Could not read figure styles; using defaults.".into(),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => p.status = format!("Could not read figure styles: {e}"),
        }
    }

    let mut open = p.open;
    egui::Window::new("Publication figure")
        .open(&mut open)
        .default_width(620.)
        .vscroll(true)
        .show(ctx, |ui| {
            ui.label("Render a frozen view at an exact size, then save the preview as PNG.");
            let busy = p.render_rx.is_some() || p.save_rx.is_some();
            ui.add_enabled_ui(!busy, |ui| {
                let s = &mut p.library.settings;
                ui.horizontal(|ui| {
                    ui.label("Pixels:");
                    ui.add(DragValue::new(&mut s.width).range(64..=8192));
                    ui.label("×");
                    ui.add(DragValue::new(&mut s.height).range(64..=8192));
                    if ui.button("Use view aspect").clicked() {
                        s.width = (s.height as f32 * scene.camera.aspect)
                            .round()
                            .clamp(64., 8192.) as u32;
                    }
                    ui.label("DPI:");
                    ui.add(DragValue::new(&mut s.dpi).range(36..=2400));
                });
                ui.label(format!("Print size: {:.2} × {:.2} cm. DPI sets print metadata; pixel size controls detail.",
                    s.width as f32 / s.dpi as f32 * 2.54, s.height as f32 / s.dpi as f32 * 2.54));
                ui.horizontal(|ui| {
                    ui.label("Antialiasing:");
                    for (n, title) in [(1, "None"), (2, "2× (4 samples)"), (3, "3× (9 samples)")] {
                        ui.selectable_value(&mut s.samples, n, title);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Geometry detail:");
                    ui.selectable_value(&mut s.sphere_detail, 3, "Standard");
                    ui.selectable_value(&mut s.sphere_detail, 4, "High")
                        .on_hover_text("Smoother close-ups; takes longer for large structures.");
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut s.transparent, "Transparent background");
                    if !s.transparent {
                        ui.color_edit_button_rgb(&mut s.background);
                    }
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut s.labels, "Include visible labels");
                    ui.color_edit_button_srgb(&mut s.label_color);
                    ui.add(Slider::new(&mut s.label_scale, 0.25..=4.).text("Label scale"));
                });
                ui.collapsing("Lighting, depth and outlines", |ui| {
                    ui.add(Slider::new(&mut s.ambient, 0.0..=1.).text("Ambient"));
                    ui.add(Slider::new(&mut s.key, 0.0..=2.).text("Key light"));
                    ui.add(Slider::new(&mut s.fill, 0.0..=1.).text("Fill light"));
                    ui.add(Slider::new(&mut s.specular, 0.0..=1.).text("Highlights"));
                    ui.add(Slider::new(&mut s.light_angle, -180.0..=180.).text("Light angle"));
                    ui.add(Slider::new(&mut s.occlusion, 0.0..=1.).text("Contact shading"));
                    ui.add(Slider::new(&mut s.outlines, 0.0..=3.).text("Outline width"));
                    ui.small("Contact shading uses screen-space depth, not ray-traced cast shadows. The figure uses studio lighting and no viewport fog.");
                });
                ui.collapsing("Saved figure styles", |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Paper").clicked() {
                            p.library.settings = Settings::default();
                        }
                        if ui.button("Presentation").clicked() {
                            p.library.settings = Settings {
                                width: 1920,
                                height: 1080,
                                dpi: 150,
                                background: [0.025, 0.035, 0.055],
                                label_color: [245; 3],
                                ..Settings::default()
                            };
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut p.style_name);
                        let valid = !p.style_name.trim().is_empty()
                            && p.library.settings.validate().is_ok();
                        if ui.add_enabled(valid, egui::Button::new("Save / replace style")).clicked() {
                            p.library.styles.insert(
                                p.style_name.trim().to_owned(),
                                p.library.settings.clone(),
                            );
                            p.persist(prefs_dir);
                        }
                    });
                    let mut delete = None;
                    for (name, settings) in &p.library.styles {
                        ui.horizontal(|ui| {
                            if ui.button(name).clicked() {
                                p.library.settings = settings.clone();
                            }
                            if ui.small_button("Delete").clicked() {
                                delete = Some(name.clone());
                            }
                        });
                    }
                    if let Some(name) = delete {
                        p.library.styles.remove(&name);
                        p.persist(prefs_dir);
                    }
                    ui.small("Styles save size, DPI, background, labels, and lighting. Camera and molecule representation stay in the scene.");
                });
                ui.collapsing("Distance measurements", |ui| {
                    ui.label("Select one atom in the scene, capture the start, then select another atom and add the distance.");
                    ui.horizontal(|ui| {
                        if ui.add_enabled(
                            selected.is_some(),
                            egui::Button::new("Capture start atom"),
                        ).clicked() {
                            p.endpoint = selected;
                        }
                        if ui.add_enabled(
                            selected.is_some() && p.endpoint.is_some(),
                            egui::Button::new("Add distance"),
                        ).clicked() {
                            p.measurements.push(Measurement {
                                a: p.endpoint.take().unwrap(),
                                b: selected.unwrap(),
                            });
                            p.figure = None;
                            p.preview = None;
                        }
                        if ui.button("Clear distances").clicked() {
                            p.measurements.clear();
                            p.endpoint = None;
                            p.figure = None;
                            p.preview = None;
                        }
                    });
                    if p.endpoint.is_some() {
                        ui.label("Start captured. Select the end atom.");
                    }
                    for m in &p.measurements {
                        ui.label(format!("{:.3} Å", (m.b - m.a).magnitude()));
                    }
                    ui.small("Captured coordinates are fixed for this figure; recapture after moving molecules. Distance lines and text are always included.");
                });
                if let Err(e) = p.library.settings.validate() {
                    ui.colored_label(Color32::YELLOW, e);
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(
                        p.library.settings.validate().is_ok(),
                        egui::Button::new("Render preview"),
                    ).clicked() {
                        let snapshot = match worker::snapshot(scene) {
                            Ok(snapshot) => snapshot,
                            Err(e) => {
                                p.status = e;
                                return;
                            }
                        };
                        let settings = p.library.settings.clone();
                        let measurements = p.measurements.clone();
                        let prefs_path = prefs_dir.join("figure_styles.json");
                        let prefs_bytes = serde_json::to_vec_pretty(&p.library);
                        let (tx, rx) = mpsc::channel();
                        let cancel = Arc::new(AtomicBool::new(false));
                        p.render_cancel = Some(cancel.clone());
                        p.render_started = Some(Instant::now());
                        p.render_rx = Some(rx);
                        p.figure = None;
                        p.preview = None;
                        p.status = "Preparing figure in a separate process…".into();
                        let repaint = ctx.clone();
                        ctx.request_repaint();
                        if let Ok(bytes) = prefs_bytes {
                            std::thread::spawn(move || {
                                let _ = atomic_write(&prefs_path, &bytes);
                            });
                        }
                        std::thread::spawn(move || {
                            // Disk sync, serialization, GPU initialization and mesh generation
                            // must never run inside the UI callback.
                            let result = std::panic::catch_unwind(
                                std::panic::AssertUnwindSafe(|| {
                                    worker::run(snapshot, settings, measurements, &cancel, &tx)
                                }),
                            ).unwrap_or_else(|_| {
                                Err("GPU rendering failed. Try a smaller image.".into())
                            });
                            let _ = tx.send(RenderEvent::Finished(result));
                            repaint.request_repaint();
                        });
                    }
                    let ready = p.figure.as_ref().is_some_and(|f| f.settings == p.library.settings);
                    if ui.add_enabled(ready, egui::Button::new("Save PNG…")).clicked() {
                        p.pending_save = p.figure.clone();
                        p.dialog = egui_file_dialog::FileDialog::new()
                            .default_file_name("figure.png")
                            .add_save_extension("PNG", "png")
                            .default_save_extension("PNG");
                        p.dialog.save_file();
                    }
                });
            });
            if busy {
                ui.spinner();
            }
            if let Some(cancel) = &p.render_cancel {
                ui.horizontal(|ui| {
                    if let Some(start) = p.render_started {
                        ui.label(format!("{} s elapsed · 120 s limit", start.elapsed().as_secs()));
                    }
                    if ui.button("Cancel render").clicked() {
                        cancel.store(true, Ordering::Relaxed);
                        p.status = "Cancelling render…".into();
                    }
                });
            }
            ui.label(&p.status);
            if let Some(preview) = &p.preview {
                if p.figure.as_ref().is_some_and(|f| f.settings != p.library.settings) {
                    ui.colored_label(Color32::YELLOW, "Settings changed. Render a new preview before saving.");
                }
                ui.small("Frozen preview; camera or scene changes require Render preview again.");
                let size = preview.size_vec2();
                let scale = (ui.available_width() / size.x).min(420. / size.y).min(1.);
                let (rect, _) = ui.allocate_exact_size(size * scale, egui::Sense::hover());
                let painter = ui.painter().with_clip_rect(rect);
                for y in 0..(rect.height() / 16.).ceil() as i32 {
                    for x in 0..(rect.width() / 16.).ceil() as i32 {
                        let tile = egui::Rect::from_min_size(
                            rect.min + egui::vec2(x as f32 * 16., y as f32 * 16.),
                            egui::vec2(16., 16.),
                        );
                        let gray = if (x + y) % 2 == 0 { 210 } else { 240 };
                        painter.rect_filled(tile, 0., Color32::from_gray(gray));
                    }
                }
                painter.image(
                    preview.id(),
                    rect,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                    Color32::WHITE,
                );
            }
        });
    p.open = open;
}
