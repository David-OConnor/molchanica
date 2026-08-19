#!/usr/bin/env bash
set -euo pipefail

version=0_3_7
outdir=target/release
exe="$outdir/molchanica"
readme="README.md"
setup="install_scripts/setup_linux.sh"
icon="resources/icon.png"
cufft="/usr/local/cuda/lib64/libcufft.so.12"
# Needed by the bio_tools ProteinMPNN recipe for native ΔΔG scanning. Its absence is handled (the
# conversion is skipped), but shipping it means one less reason to need the repository.
mpnn_convert="scripts/convert_mpnn_weights.py"

# Prevents NVCC from having to be on the path.
export PATH=/usr/local/cuda/bin:$PATH

chmod +x "$setup"

cargo build --release
zip -j -r "molchanica_${version}_linux.zip" "$exe" "$readme" "$setup" "$icon" \
    "$mpnn_convert" "$cufft"

cargo build --release --no-default-features
zip -j -r "molchanica_${version}_linux_nocuda.zip" "$exe" "$readme" "$setup" "$icon" \
    "$mpnn_convert"
