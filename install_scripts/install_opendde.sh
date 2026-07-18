#!/usr/bin/env sh

# Install OpenDDE as an isolated uv tool for Molchanica. CUDA 12.6 is selected automatically when
# a working NVIDIA GPU and compatible Linux driver are present; otherwise this uses CPU.
#
# Optional overrides:
#   OPENDDE_PYTHON_VERSION=3.11
#   OPENDDE_TORCH_BACKEND=auto|cpu|cu126

set -eu

PYTHON_VERSION="${OPENDDE_PYTHON_VERSION:-3.11}"
REQUESTED_BACKEND="${OPENDDE_TORCH_BACKEND:-auto}"

version_at_least() {
    awk -v current="$1" -v minimum="$2" 'BEGIN {
        split(current, a, "."); split(minimum, b, ".");
        for (i = 1; i <= 4; i++) {
            ai = a[i] + 0; bi = b[i] + 0;
            if (ai > bi) exit 0;
            if (ai < bi) exit 1;
        }
        exit 0;
    }'
}

cuda_126_available() {
    [ "$(uname -s)" = "Linux" ] || return 1
    [ "$(uname -m)" = "x86_64" ] || return 1
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1 || return 1

    NVIDIA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
        | head -n 1 | tr -d '\r')"
    [ -n "$NVIDIA_DRIVER" ] && version_at_least "$NVIDIA_DRIVER" "560.28.03"
}

case "$REQUESTED_BACKEND" in
    auto|cu126)
        if cuda_126_available; then
            TORCH_BACKEND="cu126"
            printf 'Detected an NVIDIA GPU with driver %s; selecting CUDA 12.6.\n' "$NVIDIA_DRIVER"
        else
            TORCH_BACKEND="cpu"
            printf 'No NVIDIA GPU with a CUDA 12.6-compatible driver was found; selecting CPU.\n'
        fi
        ;;
    cpu)
        TORCH_BACKEND="cpu"
        ;;
    *)
        printf 'Error: OPENDDE_TORCH_BACKEND must be auto, cpu, or cu126.\n' >&2
        exit 1
        ;;
esac

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi

    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

UV="$(find_uv || true)"
if [ -z "$UV" ]; then
    printf 'uv was not found; installing it with the official Astral installer...\n'
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        printf 'Error: installing uv requires curl or wget.\n' >&2
        exit 1
    fi

    UV="$(find_uv || true)"
    if [ -z "$UV" ]; then
        printf 'Error: uv was installed, but its executable could not be located.\n' >&2
        exit 1
    fi
fi

install_backend() {
    backend="$1"
    if [ "$backend" = "cu126" ]; then
        package="opendde[gpu]"
    else
        package="opendde"
    fi

    printf 'Installing %s with Python %s and the %s PyTorch backend...\n' \
        "$package" "$PYTHON_VERSION" "$backend"
    "$UV" tool install \
        --force \
        --python "$PYTHON_VERSION" \
        --torch-backend "$backend" \
        "$package"
}

if [ "$TORCH_BACKEND" = "cu126" ]; then
    GPU_INSTALL_OK=false
    if install_backend "cu126"; then
        TOOL_ROOT="$("$UV" tool dir)"
        TOOL_PYTHON="$TOOL_ROOT/opendde/bin/python"
        if [ -x "$TOOL_PYTHON" ] && "$TOOL_PYTHON" -c \
            'import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith("12.6"); torch.zeros(1, device="cuda")'; then
            GPU_INSTALL_OK=true
        fi
    fi

    if [ "$GPU_INSTALL_OK" != "true" ]; then
        printf 'CUDA installation or runtime verification failed; falling back to CPU.\n' >&2
        TORCH_BACKEND="cpu"
        install_backend "cpu"
    fi
else
    install_backend "cpu"
fi

TOOL_BIN="$("$UV" tool dir --bin)"
OPENDDE="$TOOL_BIN/opendde"
if [ ! -x "$OPENDDE" ]; then
    printf 'Error: uv completed, but %s was not created.\n' "$OPENDDE" >&2
    exit 1
fi

# This makes the CLI convenient in future shells. Molchanica also discovers uv's tool bin directly,
# so a shell-profile update failure does not prevent structure prediction from working.
if ! "$UV" tool update-shell; then
    printf 'Warning: uv could not update the shell PATH; Molchanica can still locate OpenDDE.\n' >&2
fi

printf '\nVerifying the OpenDDE installation...\n'
"$OPENDDE" --version
"$OPENDDE" doctor

printf '\nOpenDDE is installed in an isolated uv environment.\n'
printf 'Executable: %s\n' "$OPENDDE"
printf 'Restart Molchanica if it is currently running.\n'
