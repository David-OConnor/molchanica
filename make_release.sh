#!/usr/bin/env bash
set -euo pipefail

version=0_3_5
outdir=target/release
exe="$outdir/molchanica"
readme="README.md"
setup="install_scripts/setup_linux_desktop.sh"
icon="resources/icon.png"
cufft="/usr/local/cuda/lib64/libcufft.so.12"
opendde="install_scripts/install_opendde.sh"

# Prevents NVCC from having to be on the path.
export PATH=/usr/local/cuda/bin:$PATH

chmod +x "$setup"

cargo build --release
zip -j -r "molchanica_${version}_linux.zip" "$exe" "$readme" "$setup" "$icon" "$opendde" "$cufft"

cargo build --release --no-default-features
zip -j -r "molchanica_${version}_linux_nocuda.zip" "$exe" "$readme" "$setup" "$icon" "$opendde"
