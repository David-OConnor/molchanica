//! Hardware summary for the sidebar: GPU, driver, CUDA and SIMD support.
//!
//! Detected once, on a background thread. The GPU query goes through NVML rather than CUDA, so no
//! CUDA context is created and no GPU memory used. Whether CUDA actually works is still decided by
//! `util::get_computation_device`, on first computation.

use std::{sync::OnceLock, thread, time::Instant};

static CAPS: OnceLock<ComputeCaps> = OnceLock::new();

#[derive(Default)]
pub struct ComputeCaps {
    /// E.g. "NVIDIA GeForce RTX 4080".
    pub gpus: Vec<String>,
    /// E.g. "580.88".
    pub driver: Option<String>,
    /// Newest CUDA version the driver supports, as (major, minor).
    pub cuda: Option<(i32, i32)>,
    /// E.g. "AVX-512".
    pub simd: &'static str,
}

impl ComputeCaps {
    /// Compact, for the sidebar row. E.g. "RTX 4080 · 580.88 · CUDA 13.0 · AVX-512".
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        match self.gpus.first() {
            Some(gpu) => parts.push(short_gpu_name(gpu).to_owned()),
            None if cfg!(feature = "cuda") => parts.push("No GPU".to_owned()),
            None => (),
        }
        if let Some(driver) = &self.driver {
            parts.push(driver.clone());
        }
        if let Some((major, minor)) = self.cuda {
            parts.push(format!("CUDA {major}.{minor}"));
        }
        parts.push(self.simd.to_owned());

        parts.join(" · ")
    }

    pub fn details(&self) -> String {
        let mut lines = Vec::new();

        if self.gpus.is_empty() {
            if cfg!(feature = "cuda") {
                lines.push("GPU: None found".to_owned());
            } else {
                lines.push("GPU: Not checked; built without CUDA".to_owned());
            }
        }
        for (i, gpu) in self.gpus.iter().enumerate() {
            if self.gpus.len() == 1 {
                lines.push(format!("GPU: {gpu}"));
            } else {
                lines.push(format!("GPU {i}: {gpu}"));
            }
        }
        if let Some(driver) = &self.driver {
            lines.push(format!("Driver: {driver}"));
        }
        if let Some((major, minor)) = self.cuda {
            lines.push(format!(
                "CUDA: {major}.{minor} (newest the driver supports)"
            ));
        }
        lines.push(format!("SIMD: {}", self.simd));

        lines.join("\n")
    }
}

/// `None` until the background detection finishes.
pub fn get() -> Option<&'static ComputeCaps> {
    CAPS.get()
}

pub fn detect_in_background() {
    thread::spawn(|| {
        let start = Instant::now();

        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut caps = ComputeCaps {
            simd: simd(),
            ..Default::default()
        };
        #[cfg(feature = "cuda")]
        detect_gpu(&mut caps);

        let elapsed = start.elapsed().as_millis();
        println!("Compute capability ({elapsed}ms): {}", caps.summary());

        let _ = CAPS.set(caps);
    });
}

#[cfg(feature = "cuda")]
fn detect_gpu(caps: &mut ComputeCaps) {
    use nvml_wrapper::{Nvml, cuda_driver_version_major, cuda_driver_version_minor};

    // Drivers install only the versioned name; the bare one comes with dev packages.
    #[cfg(target_os = "linux")]
    let nvml = Nvml::builder()
        .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
        .init()
        .or_else(|_| Nvml::init());
    #[cfg(not(target_os = "linux"))]
    let nvml = Nvml::init();

    let Ok(nvml) = nvml else {
        return;
    };

    let count = nvml.device_count().unwrap_or(0);
    caps.gpus = (0..count)
        .filter_map(|i| nvml.device_by_index(i).ok()?.name().ok())
        .collect();
    caps.driver = nvml.sys_driver_version().ok();
    caps.cuda = nvml
        .sys_cuda_driver_version()
        .ok()
        .map(|v| (cuda_driver_version_major(v), cuda_driver_version_minor(v)));
}

/// "NVIDIA GeForce RTX 4080" -> "RTX 4080".
fn short_gpu_name(name: &str) -> &str {
    let name = name.strip_prefix("NVIDIA ").unwrap_or(name);

    name.strip_prefix("GeForce ").unwrap_or(name)
}

/// Backend selected by the packed arithmetic library.
fn simd() -> &'static str {
    lin_alg::simd_backend()
}
