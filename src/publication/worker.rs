//! Export owns a separate process, not just a separate thread. A blocked GPU/driver call
//! cannot hold the viewport's wgpu locks, and the parent can cancel or time out the worker.

use std::{
    fs,
    io::{BufReader, BufWriter, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bincode::{Decode, Encode};
use graphics::{Camera, Entity, Handedness, Mesh, Scene, TextOverlay};
use lin_alg::f32::{Quaternion, Vec3};

use super::{Figure, Measurement, RenderEvent, Settings, renderer};

const TIMEOUT: Duration = Duration::from_secs(120);
const MAX_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;
const PRIMITIVE_MESHES: [usize; 4] = [
    crate::render::MESH_SPHERE_HIGHRES,
    crate::render::MESH_SPHERE_LOWRES,
    crate::render::MESH_SPHERE_MEDRES,
    crate::render::MESH_BOND,
];

/// Copy only data required for the frozen figure. Do not clone unused density/surface meshes,
/// UI overlays, or primitive meshes that the worker will replace. Bound the synchronous copy.
pub(super) fn snapshot(scene: &Scene) -> Result<Scene, String> {
    if !scene.gaussians.is_empty() {
        return Err(
            "Gaussian volumes cannot be exported yet. Switch to a mesh representation.".into(),
        );
    }
    let mut used = vec![false; scene.meshes.len()];
    let mut bytes = scene
        .entities
        .len()
        .saturating_mul(std::mem::size_of::<Entity>());
    for entity in &scene.entities {
        let entry = used
            .get_mut(entity.mesh)
            .ok_or("Invalid figure mesh reference")?;
        *entry = true;
        if let Some(label) = &entity.overlay_text {
            bytes = bytes.saturating_add(label.text.len());
        }
    }
    for (i, mesh) in scene.meshes.iter().enumerate() {
        if used[i] && !PRIMITIVE_MESHES.contains(&i) {
            bytes = bytes.saturating_add(
                mesh.vertices
                    .len()
                    .saturating_mul(std::mem::size_of::<graphics::Vertex>()),
            );
            bytes = bytes.saturating_add(
                mesh.indices
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            );
        }
    }
    if bytes > MAX_SNAPSHOT_BYTES {
        return Err(
            "Figure snapshot exceeds 256 MiB. Hide or simplify some geometry before exporting."
                .into(),
        );
    }
    Ok(Scene {
        meshes: scene
            .meshes
            .iter()
            .enumerate()
            .map(|(i, mesh)| {
                if used[i] && !PRIMITIVE_MESHES.contains(&i) {
                    mesh.clone()
                } else {
                    Mesh::default()
                }
            })
            .collect(),
        entities: scene.entities.clone(),
        camera: scene.camera.clone(),
        ..Default::default()
    })
}

#[derive(Encode, Decode)]
struct Label {
    text: String,
    size: f32,
    color: (u8, u8, u8, u8),
    monospace: bool,
}

#[derive(Encode, Decode)]
struct Instance {
    mesh: usize,
    position: Vec3,
    orientation: Quaternion,
    pivot: Option<Vec3>,
    scale: f32,
    scale_partial: Option<Vec3>,
    color: (f32, f32, f32),
    opacity: f32,
    label: Option<Label>,
}

#[derive(Encode, Decode)]
struct Job {
    settings: Settings,
    measurements: Vec<Measurement>,
    meshes: Vec<Mesh>,
    entities: Vec<Instance>,
    camera_position: Vec3,
    camera_orientation: Quaternion,
    fov_y: f32,
    near: f32,
    far: f32,
    right_handed: bool,
}

impl Job {
    fn new(scene: Scene, settings: Settings, measurements: Vec<Measurement>) -> Self {
        Self {
            settings,
            measurements,
            meshes: scene.meshes,
            entities: scene
                .entities
                .into_iter()
                .map(|e| Instance {
                    mesh: e.mesh,
                    position: e.position,
                    orientation: e.orientation,
                    pivot: e.pivot,
                    scale: e.scale,
                    scale_partial: e.scale_partial,
                    color: e.color,
                    opacity: e.opacity,
                    label: e.overlay_text.map(|t| Label {
                        text: t.text,
                        size: t.size,
                        color: t.color,
                        monospace: t.font_family == egui::FontFamily::Monospace,
                    }),
                })
                .collect(),
            camera_position: scene.camera.position,
            camera_orientation: scene.camera.orientation,
            fov_y: scene.camera.fov_y,
            near: scene.camera.near,
            far: scene.camera.far,
            right_handed: scene.camera.handedness == Handedness::Right,
        }
    }

    fn scene(&mut self) -> Scene {
        Scene {
            meshes: std::mem::take(&mut self.meshes),
            entities: std::mem::take(&mut self.entities)
                .into_iter()
                .map(|e| Entity {
                    mesh: e.mesh,
                    position: e.position,
                    orientation: e.orientation,
                    pivot: e.pivot,
                    scale: e.scale,
                    scale_partial: e.scale_partial,
                    color: e.color,
                    opacity: e.opacity,
                    overlay_text: e.label.map(|t| TextOverlay {
                        text: t.text,
                        size: t.size,
                        color: t.color,
                        font_family: if t.monospace {
                            egui::FontFamily::Monospace
                        } else {
                            egui::FontFamily::Proportional
                        },
                    }),
                    ..Default::default()
                })
                .collect(),
            camera: Camera {
                position: self.camera_position,
                orientation: self.camera_orientation,
                fov_y: self.fov_y,
                near: self.near,
                far: self.far,
                handedness: if self.right_handed {
                    Handedness::Right
                } else {
                    Handedness::Left
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Result<Self, String> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("molchanica-figure-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        // This is the unique directory created by this job, never a user-selected path.
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        // Reap before dropping Workspace, including on cancellation, timeout, and panic.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(super) fn run(
    scene: Scene,
    settings: Settings,
    measurements: Vec<Measurement>,
    cancel: &AtomicBool,
    events: &Sender<RenderEvent>,
) -> Result<Figure, String> {
    let started = Instant::now();
    let workspace = Workspace::new()?;
    settings.validate()?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Render cancelled.".into());
    }
    let job = Job::new(scene, settings.clone(), measurements);
    let mut input =
        BufWriter::new(fs::File::create(workspace.0.join("scene.bin")).map_err(|e| e.to_string())?);
    bincode::encode_into_std_write(&job, &mut input, bincode::config::standard())
        .map_err(|e| e.to_string())?;
    use std::io::Write;
    input.flush().map_err(|e| e.to_string())?;
    drop(input);
    drop(job);
    if cancel.load(Ordering::Relaxed) {
        return Err("Render cancelled.".into());
    }

    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command.arg("--render-figure").arg(&workspace.0);
    command.stdin(Stdio::null()).stdout(Stdio::null());
    command.stderr(fs::File::create(workspace.0.join("worker.log")).map_err(|e| e.to_string())?);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut worker = Worker(
        command
            .spawn()
            .map_err(|e| format!("Cannot start figure worker: {e}"))?,
    );
    wait_for_worker(
        &mut worker.0,
        &workspace.0,
        cancel,
        events,
        started,
        TIMEOUT,
    )?;
    let result_path = workspace.0.join("pixels.rgba");
    let expected = settings.width as u64 * settings.height as u64 * 4;
    if fs::metadata(&result_path).map_err(|e| e.to_string())?.len() != expected {
        return Err("Figure worker returned an incomplete image.".into());
    }
    let rgba = fs::read(result_path).map_err(|e| e.to_string())?;
    Ok(Figure { settings, rgba })
}

fn wait_for_worker(
    child: &mut Child,
    root: &Path,
    cancel: &AtomicBool,
    events: &Sender<RenderEvent>,
    started: Instant,
    timeout: Duration,
) -> Result<(), String> {
    let mut previous = String::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("Render cancelled.".into());
        }
        if started.elapsed() >= timeout {
            return Err("Render exceeded 120 seconds and was stopped. Try Standard geometry or lower antialiasing.".into());
        }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            if !status.success() {
                let error = fs::read_to_string(root.join("error.txt")).unwrap_or_default();
                if !error.is_empty() {
                    return Err(error);
                }
                let mut log = String::new();
                if let Ok(file) = fs::File::open(root.join("worker.log")) {
                    let _ = file.take(4096).read_to_string(&mut log);
                }
                return Err(format!("Figure worker exited with {status}. {log}"));
            }
            return Ok(());
        }
        if let Ok(progress) = fs::read_to_string(root.join("progress.txt"))
            && !progress.is_empty()
            && progress != previous
        {
            let _ = events.send(RenderEvent::Progress(progress.clone()));
            previous = progress;
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn render_job(root: &Path) -> Result<(), String> {
    let mut input =
        BufReader::new(fs::File::open(root.join("scene.bin")).map_err(|e| e.to_string())?);
    let mut job: Job = bincode::decode_from_std_read(
        &mut input,
        bincode::config::standard().with_limit::<536_870_912>(),
    )
    .map_err(|e| e.to_string())?;
    job.settings.validate()?;
    let mut scene = job.scene();
    fs::write(root.join("progress.txt"), "Preparing smooth geometry…")
        .map_err(|e| e.to_string())?;
    for index in PRIMITIVE_MESHES {
        if !scene.entities.iter().any(|e| e.mesh == index) {
            continue;
        }
        let mesh = scene
            .meshes
            .get_mut(index)
            .ok_or("Invalid primitive mesh reference")?;
        *mesh = if index == crate::render::MESH_BOND {
            Mesh::new_cylinder(
                1.,
                crate::drawing::atoms_bonds::BOND_RADIUS_BASE,
                if job.settings.sphere_detail == 4 {
                    48
                } else {
                    32
                },
            )
        } else {
            Mesh::new_sphere(1., job.settings.sphere_detail)
        };
    }
    let mut last_update = Instant::now() - Duration::from_secs(1);
    let figure = renderer::render(scene, job.settings, &job.measurements, |progress| {
        if last_update.elapsed() >= Duration::from_millis(100) {
            let _ = fs::write(root.join("progress.txt"), progress);
            last_update = Instant::now();
        }
    })?;
    fs::write(root.join("pixels.rgba"), figure.rgba).map_err(|e| e.to_string())
}

/// Invoked only by the private CLI branch, before the interactive renderer starts.
pub(crate) fn worker_main() -> i32 {
    let Some(root) = std::env::args_os().nth(2).map(PathBuf::from) else {
        return 2;
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| render_job(&root)))
        .unwrap_or_else(|_| {
            Err("The figure GPU worker failed. Try Standard geometry or a smaller image.".into())
        });
    match result {
        Ok(()) => 0,
        Err(e) => {
            let _ = fs::write(root.join("error.txt"), e);
            1
        }
    }
}
