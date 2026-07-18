#!/usr/bin/env sh

# Install OpenDDE in a dedicated standard-library Python virtual environment for Molchanica.
# CUDA 12.6 is selected automatically when a working NVIDIA GPU and compatible Linux driver are
# present; otherwise this uses CPU. Molchanica discovers this environment without activation.
#
# Optional overrides:
#   OPENDDE_PYTHON=/path/to/python
#   OPENDDE_VENV_DIR=/path/to/opendde-venv
#   OPENDDE_TORCH_BACKEND=auto|cpu|cu126

set -eu

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

python_is_compatible() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
        >/dev/null 2>&1
}

find_python() {
    if [ -n "${OPENDDE_PYTHON:-}" ]; then
        if [ ! -x "$OPENDDE_PYTHON" ] || ! python_is_compatible "$OPENDDE_PYTHON"; then
            printf 'Error: OPENDDE_PYTHON must be an executable running Python 3.11 or newer.\n' >&2
            return 1
        fi
        printf '%s\n' "$OPENDDE_PYTHON"
        return 0
    fi

    for name in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$name" >/dev/null 2>&1; then
            candidate="$(command -v "$name")"
            if python_is_compatible "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done

    return 1
}

PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
    printf 'Error: Python 3.11 or newer is required to install OpenDDE.\n' >&2
    exit 1
fi

if [ -n "${OPENDDE_VENV_DIR:-}" ]; then
    VENV_DIR="$OPENDDE_VENV_DIR"
elif [ "$(uname -s)" = "Darwin" ]; then
    VENV_DIR="$HOME/Library/Application Support/molchanica/opendde-venv"
else
    DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
    VENV_DIR="$DATA_HOME/molchanica/opendde-venv"
fi

printf 'Using %s\n' "$("$PYTHON" --version 2>&1)"
printf 'Creating an isolated OpenDDE environment at %s...\n' "$VENV_DIR"
"$PYTHON" -m venv --clear "$VENV_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"
OPENDDE="$VENV_DIR/bin/opendde"
if [ ! -x "$VENV_PYTHON" ]; then
    printf 'Error: the virtual-environment Python was not created at %s.\n' "$VENV_PYTHON" >&2
    exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip

install_backend() {
    backend="$1"
    if [ "$backend" = "cu126" ]; then
        package="opendde[gpu]"
        torch_index="https://download.pytorch.org/whl/cu126"
    else
        package="opendde"
        torch_index="https://download.pytorch.org/whl/cpu"
    fi

    printf 'Installing %s with the %s PyTorch backend...\n' "$package" "$backend"
    # Match OpenDDE's currently pinned PyTorch packages while choosing their CPU/CUDA wheel index.
    if [ "$(uname -s)" = "Darwin" ]; then
        "$VENV_PYTHON" -m pip install \
            "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" || return 1
    else
        "$VENV_PYTHON" -m pip install \
            "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
            --index-url "$torch_index" || return 1
    fi
    "$VENV_PYTHON" -m pip install "$package" || return 1
}

if [ "$TORCH_BACKEND" = "cu126" ]; then
    GPU_INSTALL_OK=false
    if install_backend "cu126" && "$VENV_PYTHON" -c \
        'import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith("12.6"); torch.zeros(1, device="cuda")'; then
        GPU_INSTALL_OK=true
    fi

    if [ "$GPU_INSTALL_OK" != "true" ]; then
        printf 'CUDA installation or runtime verification failed; rebuilding for CPU.\n' >&2
        TORCH_BACKEND="cpu"
        "$PYTHON" -m venv --clear "$VENV_DIR"
        "$VENV_PYTHON" -m pip install --upgrade pip
        install_backend "cpu"
    fi
else
    install_backend "cpu"
fi

if [ ! -x "$OPENDDE" ]; then
    printf 'Error: pip completed, but %s was not created.\n' "$OPENDDE" >&2
    exit 1
fi

printf '\nVerifying the OpenDDE installation...\n'
"$OPENDDE" --version
"$OPENDDE" doctor

printf '\nOpenDDE is installed in a dedicated Python virtual environment.\n'
printf 'Executable: %s\n' "$OPENDDE"
printf 'No activation is required. Restart Molchanica if it is currently running.\n'
