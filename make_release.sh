#!/usr/bin/env bash
set -euo pipefail

app_name="molchanica"

# The version is defined in one place only: Cargo.toml's `[package] version`. Read it from there,
# so a release never ships under a version that disagrees with the one the app reports.
# Archive names use underscores, e.g. 0.3.7 -> 0_3_7.
version=$(sed -n -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' Cargo.toml | head -n 1 | tr '.' '_')
if [[ -z "$version" ]]; then
    echo "Could not read the version from Cargo.toml" >&2
    exit 1
fi
outdir=target/release
exe="$outdir/$app_name"
readme="README.md"
setup="install_scripts/setup_${app_name}.sh"
icon="resources/icon.png"
cufft="/usr/local/cuda/lib64/libcufft.so.12"
# Needed by the bio_tools ProteinMPNN recipe for native ΔΔG scanning. Its absence is handled (the
# conversion is skipped), but shipping it means one less reason to need the repository.
mpnn_convert="scripts/convert_mpnn_weights.py"

# Prevents NVCC from having to be on the path.
export PATH=/usr/local/cuda/bin:$PATH

chmod +x "$setup"

# The zip takes the setup script as it is on disk, not as git stores it. On Windows, that copy can
# have CRLF endings (core.autocrlf, or an editor), which makes bash fail on the first line for
# whoever installs the release. Strip them here so the archive is right regardless.
sed -i 's/\r$//' "$setup"

cargo build --release

# The CUDA build is also what people without an Nvidia GPU download, so it must not name any CUDA
# library as a load-time dependency: the dynamic loader would refuse to start it there, before the
# program ever gets a chance to fall back to the CPU. cudarc's `dynamic-loading` feature and the
# `dlopen` in ewald's cufft.cu keep these out; a stray `cargo:rustc-link-lib` would put them back,
# so fail the release here rather than in a bug report.
if ldd "$exe" | grep -Eq 'libcuda\.so|libcufft\.so|libnvrtc\.so|libcudart\.so'; then
    echo "Error: $exe links a CUDA library at load time, so it will not start without the" >&2
    echo "Nvidia driver installed. Offending entries:" >&2
    ldd "$exe" | grep -E 'libcuda\.so|libcufft\.so|libnvrtc\.so|libcudart\.so' >&2
    exit 1
fi

zip -j -r "${app_name}_${version}_linux.zip" "$exe" "$readme" "$setup" "$icon" \
    "$mpnn_convert" "$cufft"

# We don't use a second binary for non-CUDA; the Cuda-compiled binary should work
# on non-CUDA-available setups. The only difference in the packages is inclusion of hte cuFFT library.
#cargo build --release --no-default-features
zip -j -r "${app_name}_${version}_linux_nocuda.zip" "$exe" "$readme" "$setup" "$icon" \
    "$mpnn_convert"
