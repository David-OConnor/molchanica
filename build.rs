//! Configure optional bundled databases and compile CUDA C++ code when enabled.

use std::{env, path::PathBuf};

#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_ptx};
use winresource::WindowsResource;

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    for (filename, cfg) in [
        ("hmdb.parquet", "embedded_hmdb"),
        ("ChEBI.parquet", "embedded_chebi"),
    ] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
        let path = manifest_dir.join(filename);

        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
            println!("cargo:rustc-cfg={cfg}");
        } else {
            // Watch the parent so adding a database is detected on the next build.
            println!("cargo:rerun-if-changed={}", manifest_dir.display());
        }
    }

    #[cfg(feature = "cuda")]
    build_ptx(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/cuda.cu", "src/cuda/util.cu"],
        "molchanica",
    );

    if env::var_os("CARGO_CFG_WINDOWS").is_some() {
        WindowsResource::new()
            // This path can be absolute, or relative to your crate root.
            .set_icon("resources/icon.ico")
            .compile()
            .unwrap();
    }
}
