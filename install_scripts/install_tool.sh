#!/usr/bin/env sh

# Install one of Molchanica's optional third-party tools.
#
#   ./install_tool.sh opendde
#   ./install_tool.sh ligandmpnn proteinmpnn
#   ./install_tool.sh all
#   ./install_tool.sh --list
#
# There is one script rather than one per tool because every one of them needs the same handful of
# things: a uv-managed Python of a particular version, an isolated virtual environment, a Torch build matching
# whatever GPU is present, a download or two, and a check that the result actually runs. Writing
# that six times, twice (once per platform), is how installers drift apart. The per-tool part is
# the `install_<slug>` function at the bottom; everything above it is shared.
#
# Locations match what src/external_tools/mod.rs looks in, and the slugs match its registry:
#
#   <data root>/molchanica/<slug>-venv     per-tool Python environment
#   <data root>/molchanica/tools/<name>    binary distributions and checkouts
#
# Optional overrides:
#   MOLCHANICA_DATA_DIR=/path         where everything is installed
#   MOLCHANICA_TORCH_BACKEND=auto|cpu|cu126
#   MOLCHANICA_UV=/path/to/uv         uses an existing uv executable

set -eu

# --------------------------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------------------------

if [ -n "${MOLCHANICA_DATA_DIR:-}" ]; then
    DATA_ROOT="$MOLCHANICA_DATA_DIR"
elif [ "$(uname -s)" = "Darwin" ]; then
    DATA_ROOT="$HOME/Library/Application Support/molchanica"
else
    DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/molchanica"
fi
TOOLS_ROOT="$DATA_ROOT/tools"

venv_dir() {
    # OPENDDE_VENV_DIR predates this script and is documented, so it still wins for that one tool.
    if [ "$1" = "opendde" ] && [ -n "${OPENDDE_VENV_DIR:-}" ]; then
        printf '%s\n' "$OPENDDE_VENV_DIR"
    else
        printf '%s\n' "$DATA_ROOT/$1-venv"
    fi
}

venv_python() { printf '%s/bin/python\n' "$(venv_dir "$1")"; }
venv_script() { printf '%s/bin/%s\n' "$(venv_dir "$1")" "$2"; }

# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

section() { printf '\n%s\n========================================================\n' "$1"; }
note() { printf '  %s\n' "$1"; }
fail() { printf 'Error: %s\n' "$1" >&2; exit 1; }

# --------------------------------------------------------------------------------------------
# uv and Python
# --------------------------------------------------------------------------------------------

UV_EXECUTABLE=""

uv_works() { "$1" --version >/dev/null 2>&1; }

# Locate uv, or install it into Molchanica's data directory with Astral's official standalone
# installer. Keeping this path explicit matters for desktop launches, which do not read shell
# profile PATH changes. uv itself needs neither Python nor Rust to be present.
ensure_uv() {
    [ -n "$UV_EXECUTABLE" ] && return 0

    if [ -n "${MOLCHANICA_UV:-}" ]; then
        uv_works "$MOLCHANICA_UV" || fail "MOLCHANICA_UV does not name a working uv executable."
        UV_EXECUTABLE="$MOLCHANICA_UV"
        return 0
    fi

    managed_uv="$DATA_ROOT/uv-bin/uv"
    if [ -x "$managed_uv" ] && uv_works "$managed_uv"; then
        UV_EXECUTABLE="$managed_uv"
        return 0
    fi

    if command -v uv >/dev/null 2>&1 && uv_works "$(command -v uv)"; then
        UV_EXECUTABLE="$(command -v uv)"
        return 0
    fi

    # Astral's default installer location is not always on PATH in a non-login shell.
    fallback_uv="$HOME/.local/bin/uv"
    if [ -x "$fallback_uv" ] && uv_works "$fallback_uv"; then
        UV_EXECUTABLE="$fallback_uv"
        return 0
    fi

    section "Installing uv"
    mkdir -p "$DATA_ROOT/uv-bin"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_UNMANAGED_INSTALL="$DATA_ROOT/uv-bin" sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh \
            | env UV_UNMANAGED_INSTALL="$DATA_ROOT/uv-bin" sh
    else
        fail "uv is required, and neither curl nor wget is available to install it."
    fi

    uv_works "$managed_uv" || fail "Astral's installer completed, but uv was not found at $managed_uv."
    UV_EXECUTABLE="$managed_uv"
}

uv_pip_install() {
    slug="$1"
    shift
    ensure_uv
    run_uv pip install --python "$(venv_python "$slug")" "$@"
}

# Neither an activated environment nor user Python settings should influence an explicitly
# targeted uv operation. `UV_NO_MANAGED_PYTHON` is removed so it cannot contradict the required
# `--managed-python` flag below.
run_uv() {
    ensure_uv
    env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u CONDA_PREFIX \
        -u PYTHONHOME -u PYTHONPATH -u UV_PYTHON -u UV_NO_MANAGED_PYTHON \
        "$UV_EXECUTABLE" "$@"
}

# Create a clean environment for a tool using an exact, uv-managed Python minor version. The
# `--managed-python` flag is important: without it, uv may reuse a matching system interpreter.
make_venv() {
    slug="$1"; python_version="$2"
    ensure_uv
    target="$(venv_dir "$slug")"
    note "Creating $target with uv-managed Python $python_version"
    mkdir -p "$(dirname "$target")"
    run_uv venv --managed-python --python "$python_version" --clear "$target"
    [ -x "$(venv_python "$slug")" ] || fail "the virtual environment at $target has no interpreter."
    note "Using $("$(venv_python "$slug")" --version 2>&1)"
}

# --------------------------------------------------------------------------------------------
# Torch backend selection
# --------------------------------------------------------------------------------------------

# CUDA 12.6 wheels need at least this driver. Below it, the wheels install and then fail at run
# time with an unhelpful error, so it is checked before installing rather than after.
LINUX_MIN_DRIVER="560.28.03"

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

# Whether CUDA 12.6 wheels are worth installing here.
#
# macOS is excluded on purpose and separately from "no GPU found": Apple ships no CUDA driver at
# all, so an explicit cu126 request there is a mistake to report rather than a preference to
# quietly ignore.
cuda_126_available() {
    [ "$(uname -m)" = "x86_64" ] || return 1
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1 || return 1

    NVIDIA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
        | head -n 1 | tr -d '\r')"
    [ -n "$NVIDIA_DRIVER" ] && version_at_least "$NVIDIA_DRIVER" "$LINUX_MIN_DRIVER"
}

# Sets TORCH_BACKEND to cpu or cu126.
select_torch_backend() {
    requested="${MOLCHANICA_TORCH_BACKEND:-auto}"
    case "$requested" in
        cpu)
            TORCH_BACKEND="cpu"
            return 0
            ;;
        auto|cu126) ;;
        *) fail "MOLCHANICA_TORCH_BACKEND must be auto, cpu, or cu126." ;;
    esac

    if [ "$(uname -s)" = "Darwin" ]; then
        # An explicit request is an error; `auto` just means CPU, which is the only option here.
        [ "$requested" = "cu126" ] && fail "CUDA is not available on macOS. Use MOLCHANICA_TORCH_BACKEND=cpu."
        TORCH_BACKEND="cpu"
        note "macOS: installing the CPU PyTorch build (Apple GPUs are not CUDA devices)."
        return 0
    fi

    if cuda_126_available; then
        TORCH_BACKEND="cu126"
        note "Detected an NVIDIA GPU with driver $NVIDIA_DRIVER; selecting CUDA 12.6."
    else
        if [ "$requested" = "cu126" ]; then
            fail "CUDA 12.6 was requested, but no NVIDIA GPU with a driver >= $LINUX_MIN_DRIVER was found."
        fi
        TORCH_BACKEND="cpu"
        note "No NVIDIA GPU with a CUDA 12.6-compatible driver was found; selecting CPU."
    fi
}

torch_index_url() {
    if [ "$1" = "cu126" ]; then
        printf 'https://download.pytorch.org/whl/cu126\n'
    else
        printf 'https://download.pytorch.org/whl/cpu\n'
    fi
}

# Install a pinned Torch from the backend's own wheel index.
#
# Pinned rather than left to the dependency resolver so that the CPU/CUDA choice above is actually
# honoured: an unpinned `pip install torch` pulls whatever default wheel PyPI serves, which is a
# CUDA build regardless of whether there is a GPU to run it on.
install_torch() {
    slug="$1"; version="$2"; backend="$3"
    if [ "$(uname -s)" = "Darwin" ]; then
        # macOS wheels are only on PyPI; the CPU index has no darwin builds.
        uv_pip_install "$slug" "torch==$version"
    else
        uv_pip_install "$slug" "torch==$version" \
            --index-url "$(torch_index_url "$backend")"
    fi
}

# Confirm the installed Torch actually reaches the GPU it was built for.
#
# A CUDA wheel that cannot see a device is the failure mode worth catching here: everything
# installs cleanly, and only the first real run reports it, by which point the user has waited
# through a model download.
torch_cuda_works() {
    "$(venv_python "$1")" -c \
        'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda")' \
        >/dev/null 2>&1
}

# --------------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------------

download() {
    url="$1"; destination="$2"
    mkdir -p "$(dirname "$destination")"
    if [ -s "$destination" ]; then
        note "Already have $(basename "$destination")"
        return 0
    fi
    note "Downloading $(basename "$destination")"
    if command -v curl >/dev/null 2>&1; then
        curl -fLsS "$url" -o "$destination.partial"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$destination.partial" "$url"
    else
        fail "neither curl nor wget is available."
    fi
    # Renamed only on success, so an interrupted download is retried rather than treated as done.
    mv "$destination.partial" "$destination"
}

download_python_script() {
    url="$1"; destination="$2"; slug="$3"
    source_file="$destination.source"
    download "$url" "$source_file"
    {
        printf '#!%s\n' "$(venv_python "$slug")"
        cat "$source_file"
    } >"$destination.partial"
    mv "$destination.partial" "$destination"
    rm -f "$source_file"
    chmod +x "$destination"
}


clone_or_update() {
    url="$1"; target="$2"
    command -v git >/dev/null 2>&1 || fail "git is required to install this tool."
    if [ -d "$target/.git" ]; then
        note "Updating $(basename "$target")"
        git -C "$target" fetch --depth 1 origin HEAD
        git -C "$target" reset --hard FETCH_HEAD
    else
        note "Cloning $(basename "$target")"
        mkdir -p "$(dirname "$target")"
        git clone --depth 1 "$url" "$target"
    fi
}

# --------------------------------------------------------------------------------------------
# Per-tool installers
# --------------------------------------------------------------------------------------------

# OpenDDE: all-atom co-folding. OpenDDE 1.0.x supports Python 3.11 through 3.13.
install_opendde() {
    section "OpenDDE"
    select_torch_backend
    make_venv opendde 3.13

    install_backend_opendde() {
        backend="$1"
        note "Installing opendde with the $backend PyTorch backend"
        if [ "$backend" = "cu126" ]; then
            package="opendde[gpu]"
        else
            package="opendde"
        fi
        # OpenDDE pins this trio; the index decides CPU or CUDA.
        if [ "$(uname -s)" = "Darwin" ]; then
            uv_pip_install opendde \
                "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" || return 1
        else
            uv_pip_install opendde \
                "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
                --index-url "$(torch_index_url "$backend")" || return 1
        fi
        uv_pip_install opendde "$package" || return 1
    }

    if [ "$TORCH_BACKEND" = "cu126" ]; then
        if install_backend_opendde cu126 && "$(venv_python opendde)" -c \
            'import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith("12.6"); torch.zeros(1, device="cuda")'
        then
            :
        else
            note "CUDA installation or runtime verification failed; rebuilding for CPU."
            TORCH_BACKEND="cpu"
            make_venv opendde 3.13
            install_backend_opendde cpu
        fi
    else
        install_backend_opendde cpu
    fi

    opendde="$(venv_script opendde opendde)"
    [ -x "$opendde" ] || fail "pip completed, but $opendde was not created."

    note "Verifying"
    "$opendde" --version
    "$opendde" doctor

    prewarm_opendde "$opendde"
    note "OpenDDE installed. No activation is required."
}

# Fetch the model checkpoint now rather than on the user's first prediction.
#
# `opendde pred` downloads what it needs when it is missing, so this is not required — but without
# it the first prediction anyone runs stalls for a multi-gigabyte download inside what looks like
# a hung job. Doing it here, where a progress log is expected, is the difference.
#
# The tiny job below is the download trigger; whether it converges is irrelevant, so a failure is
# reported and ignored. MSA, template, and RNA-MSA search are all off, matching how Molchanica
# invokes OpenDDE, so only the checkpoint and common files are fetched — not the far larger
# search databases.
prewarm_opendde() {
    opendde="$1"
    if [ -d "${OPENDDE_ROOT_DIR:-$HOME/.cache/opendde}/checkpoint" ]; then
        note "OpenDDE model data is already present."
        return 0
    fi

    note "Fetching the OpenDDE model checkpoint (several GB; this is a one-time download)."
    workdir="$(mktemp -d)"
    cat > "$workdir/input.json" <<'JSON'
[{"name": "prewarm", "modelSeeds": [101], "sequences": [{"proteinChain": {"sequence": "ACDEFG", "count": 1, "id": ["A"]}}]}]
JSON
    if (cd "$workdir" && "$opendde" pred -i input.json -o output -n opendde_v1 \
            --use_msa false --use_template false --use_rna_msa false \
            --sample 1 --step 1 --cycle 1) >/dev/null 2>&1; then
        note "Model data cached; the first real prediction will start immediately."
    else
        note "Could not pre-fetch the model data. It will download on the first prediction instead."
    fi
    rm -rf "$workdir"
}

# Boltz-2: co-folding plus binding-affinity prediction. Requires Python >= 3.10, < 3.13.
install_boltz2() {
    section "Boltz-2"
    select_torch_backend
    # The upper bound is Boltz's own `requires-python`, and is why this cannot share OpenDDE's
    # environment.
    make_venv boltz2 3.12

    install_torch boltz2 2.7.1 "$TORCH_BACKEND"

    # The [cuda] extra pulls cuequivariance wheels that exist for Linux x86_64 only, so it is
    # requested only where it can resolve. Plain `boltz` is a pure-Python wheel and works
    # everywhere, GPU included — the extra is a speed-up, not a requirement.
    if [ "$TORCH_BACKEND" = "cu126" ] && [ "$(uname -s)" = "Linux" ]; then
        if ! uv_pip_install boltz2 "boltz[cuda]~=2.2.1"; then
            note "The CUDA extra did not resolve; installing Boltz without it."
            uv_pip_install boltz2 "boltz~=2.2.1"
        fi
    else
        uv_pip_install boltz2 "boltz~=2.2.1"
    fi

    boltz="$(venv_script boltz2 boltz)"
    [ -x "$boltz" ] || fail "pip completed, but $boltz was not created."
    note "Verifying"
    "$boltz" --help >/dev/null || fail "boltz was installed but does not run."
    if [ "$TORCH_BACKEND" = "cu126" ]; then
        torch_cuda_works boltz2 || note "Warning: Torch cannot reach the GPU; Boltz will run on CPU."
    fi
    note "Boltz-2 installed. Model weights download on first use."
}

# The Torch runtime shared by both MPNN checkouts.
#
# numpy is held below 2 because the MPNN code predates the numpy 2 API removals, and that in turn
# caps Python at 3.12: numpy 1.26 publishes no cp313 wheel.
install_mpnn_runtime() {
    slug="$1"
    select_torch_backend
    make_venv "$slug" 3.12
    install_torch "$slug" 2.7.1 "$TORCH_BACKEND"
    uv_pip_install "$slug" "numpy<2"
    if [ "$TORCH_BACKEND" = "cu126" ]; then
        torch_cuda_works "$slug" || note "Warning: Torch cannot reach the GPU; designs will run on CPU."
    fi
}

LIGANDMPNN_WEIGHTS_URL="https://files.ipd.uw.edu/pub/ligandmpnn"

install_ligandmpnn() {
    section "LigandMPNN"
    install_mpnn_runtime ligandmpnn

    target="$TOOLS_ROOT/LigandMPNN"
    clone_or_update https://github.com/dauparas/LigandMPNN "$target"

    # Fetched directly rather than through the repository's get_model_params.sh, which is a bash
    # script and so would not run on Windows; the PowerShell installer downloads the same files.
    # These three are the checkpoints src/external_tools/mpnn.rs selects between.
    for weights in ligandmpnn_v_32_010_25.pt proteinmpnn_v_48_020.pt solublempnn_v_48_020.pt; do
        download "$LIGANDMPNN_WEIGHTS_URL/$weights" "$target/model_params/$weights"
    done

    [ -s "$target/model_params/ligandmpnn_v_32_010_25.pt" ] \
        || fail "the LigandMPNN weights did not download."
    note "LigandMPNN installed at $target"
}

# The AbMPNN checkpoint: ProteinMPNN's architecture finetuned on antibodies (Frey et al., ICML
# 2023 CompBio workshop), published under CC BY 4.0.
ABMPNN_WEIGHTS_URL="https://zenodo.org/records/8164693/files/abmpnn.pt?download=1"

install_proteinmpnn() {
    section "ProteinMPNN (and the AbMPNN weights)"
    install_mpnn_runtime proteinmpnn

    target="$TOOLS_ROOT/ProteinMPNN"
    clone_or_update https://github.com/dauparas/ProteinMPNN "$target"
    # The vanilla checkpoints are committed to the repository, so cloning is the whole install for
    # them; only AbMPNN has to be fetched.
    [ -s "$target/vanilla_model_weights/v_48_020.pt" ] \
        || fail "the ProteinMPNN checkout has no vanilla_model_weights/v_48_020.pt."

    # Named v_48_020.pt so that --model_name's default matches for both weight sets and the
    # adapter never has to pass that flag; see src/external_tools/mpnn.rs.
    download "$ABMPNN_WEIGHTS_URL" "$target/abmpnn_weights/v_48_020.pt"

    convert_mpnn_weights "$target"
    note "ProteinMPNN installed at $target"
}

# Convert the checkpoint for Molchanica's native ΔΔG scanner.
#
# Optional: skipping it costs only the ΔΔG feature, so a failure here must not fail the install of
# the design tool the user actually asked for.
convert_mpnn_weights() {
    target="$1"
    here="$(cd "$(dirname "$0")" && pwd)"
    # Two layouts: the repository, where this script is in install_scripts/ and the converter is
    # in scripts/; and a release archive, which is flat.
    converter=""
    for candidate in "$here/../scripts/convert_mpnn_weights.py" "$here/convert_mpnn_weights.py"; do
        [ -f "$candidate" ] && converter="$candidate" && break
    done
    if [ -z "$converter" ]; then
        note "convert_mpnn_weights.py not found; skipping the native ΔΔG weight conversion."
        return 0
    fi
    note "Converting the checkpoint for the native ΔΔG scanner"
    if "$(venv_python proteinmpnn)" "$converter" \
        --checkpoint "$target/vanilla_model_weights/v_48_020.pt" \
        --output "$target/converted/v_48_020.mcnn" \
        --repo "$target"
    then
        note "Native ΔΔG scanning is available."
    else
        note "The conversion failed; ΔΔG scanning will be unavailable. The design tools still work."
    fi
}

IGBLAST_VERSION="${IGBLAST_VERSION:-1.22.0}"
NCBI_IGBLAST_FTP="https://ftp.ncbi.nih.gov/blast/executables/igblast/release"

install_igblast() {
    section "IgBLAST $IGBLAST_VERSION"
    [ "$(uname -m)" = "x86_64" ] \
        || fail "NCBI publishes IgBLAST binaries for x86_64 only; this is $(uname -m)."

    target="$TOOLS_ROOT/igblast"
    if [ "$(cat "$target/.version" 2>/dev/null)" = "$IGBLAST_VERSION" ]; then
        note "IgBLAST $IGBLAST_VERSION is already installed."
    else
        case "$(uname -s)" in
            Linux) tarball="ncbi-igblast-$IGBLAST_VERSION-x64-linux.tar.gz" ;;
            Darwin) tarball="ncbi-igblast-$IGBLAST_VERSION-x64-macosx.tar.gz" ;;
            *) fail "unsupported platform $(uname -s) for IgBLAST." ;;
        esac

        staging="$(mktemp -d)"
        download "$NCBI_IGBLAST_FTP/$IGBLAST_VERSION/$tarball" "$staging/$tarball"
        tar -xzf "$staging/$tarball" -C "$staging"

        unpacked="$staging/ncbi-igblast-$IGBLAST_VERSION"
        [ -x "$unpacked/bin/igblastn" ] && [ -d "$unpacked/internal_data" ] \
            || fail "unexpected tarball layout under $unpacked."

        # Germline databases survive a version bump; they are large and independent of the binary.
        preserved=""
        if [ -d "$target/germline_db" ]; then
            preserved="$(mktemp -d)"
            mv "$target/germline_db" "$preserved/germline_db"
        fi

        mkdir -p "$TOOLS_ROOT"
        rm -rf "$target"
        # Installed whole: igblastn resolves internal_data/ and optional_file/ relative to IGDATA,
        # so cherry-picking the binaries out would leave it unable to annotate anything.
        cp -r "$unpacked" "$target"
        [ -n "$preserved" ] && mv "$preserved/germline_db" "$target/germline_db" && rm -rf "$preserved"
        printf '%s\n' "$IGBLAST_VERSION" > "$target/.version"
        rm -rf "$staging"
    fi

    install_igblast_databases "$target"
    note "Verifying"
    IGDATA="$target" "$target/bin/igblastn" -version
    note "IgBLAST installed at $target"
}

# NCBI publishes germline databases already built with makeblastdb, so nothing here needs
# edit_imgt_file.pl or a makeblastdb run. Each archive unpacks flat.
install_igblast_databases() {
    target="$1"
    germline="$target/germline_db"
    if find "$germline" -name '*.nhr' -o -name '*.phr' 2>/dev/null | grep -q .; then
        note "Germline databases are already installed."
        return 0
    fi

    note "Installing the germline databases"
    staging="$(mktemp -d)"
    mkdir -p "$germline"
    for archive in \
        database/airr/airr_c_human.tar \
        database/airr/airr_c_mouse.tar \
        database/mouse_gl_VDJ.tar \
        database/rhesus_monkey_VJ.tar \
        database/ncbi_human_c_genes.tar
    do
        filename="$(basename "$archive")"
        download "$NCBI_IGBLAST_FTP/$archive" "$staging/$filename"
        tar -xf "$staging/$filename" -C "$germline"
    done
    rm -rf "$staging"

    find "$germline" -name '*.nhr' -o -name '*.phr' | grep -q . \
        || fail "no BLAST databases landed in $germline."
}

# ANARCII: antibody/TCR numbering. A pure-Python wheel whose only heavy dependency is Torch, which
# is what makes proper numbering available on Windows at all.
install_anarcii() {
    section "ANARCII"
    select_torch_backend
    make_venv anarcii 3.12
    install_torch anarcii 2.7.1 "$TORCH_BACKEND"
    uv_pip_install anarcii anarcii

    note "Verifying"
    "$(venv_python anarcii)" -c 'import anarcii; print("anarcii", anarcii.__version__)' \
        || fail "anarcii was installed but cannot be imported."
    note "ANARCII installed."
}
install_immunebuilder() {
    section "ImmuneBuilder"
    make_venv immunebuilder 3.11
    uv_pip_install immunebuilder ImmuneBuilder openmm pdbfixer anarci
    "$(venv_script immunebuilder ABodyBuilder2)" --help >/dev/null
}

install_biophi() {
    section "BioPhi"
    make_venv biophi 3.11
    uv_pip_install biophi "biophi @ git+https://github.com/Merck/BioPhi@main" abnumber
    "$(venv_script biophi biophi)" --help >/dev/null
}

install_thermompnn() {
    section "ThermoMPNN"
    select_torch_backend
    make_venv thermompnn 3.12
    install_torch thermompnn 2.7.1 "$TORCH_BACKEND"
    uv_pip_install thermompnn "numpy<2" pandas biopython tqdm omegaconf pytorch-lightning
    clone_or_update https://github.com/Kuhlman-Lab/ThermoMPNN "$TOOLS_ROOT/ThermoMPNN"
}

install_deepsp() {
    section "DeepSP"
    select_torch_backend
    make_venv deepsp 3.11
    install_torch deepsp 2.7.1 "$TORCH_BACKEND"
    uv_pip_install deepsp tensorflow pandas numpy biopython anarcii
    clone_or_update https://github.com/Lailabcode/DeepSP "$TOOLS_ROOT/DeepSP"
}

install_deepimmuno() {
    section "DeepImmuno"
    make_venv deepimmuno 3.10
    uv_pip_install deepimmuno "tensorflow<2.16" pandas "numpy<2" scikit-learn
    clone_or_update https://github.com/frankligy/DeepImmuno "$TOOLS_ROOT/DeepImmuno"
}

install_tlimmuno2() {
    section "TLimmuno2"
    make_venv tlimmuno2 3.10
    uv_pip_install tlimmuno2 "tensorflow<2.16" pandas pyarrow "numpy<2" scikit-learn
    clone_or_update https://github.com/XSLiuLab/TLimmuno2 "$TOOLS_ROOT/TLimmuno2"
}

install_netsolp() {
    section "NetSolP"
    select_torch_backend
    make_venv netsolp 3.11
    install_torch netsolp 2.7.1 "$TORCH_BACKEND"
    uv_pip_install netsolp "fair-esm~=2.0.0" pandas "numpy<2"
    clone_or_update https://github.com/tvinet/NetSolP-1.0 "$TOOLS_ROOT/NetSolP-1.0"
    note "NetSolP model checkpoints require separate DTU licence acceptance."
}

install_deepstabp() {
    section "DeepSTABp"
    select_torch_backend
    make_venv deepstabp 3.11
    install_torch deepstabp 2.7.1 "$TORCH_BACKEND"
    uv_pip_install deepstabp "transformers<5" sentencepiece protobuf biopython pandas pytorch-lightning
    clone_or_update https://github.com/CSBiology/deepStabP "$TOOLS_ROOT/deepStabP"
}

install_dlkcat() {
    section "DLKcat"
    select_torch_backend
    make_venv dlkcat 3.10
    install_torch dlkcat 2.7.1 "$TORCH_BACKEND"
    uv_pip_install dlkcat "numpy<2" rdkit scikit-learn
    clone_or_update https://github.com/SysBioChalmers/DLKcat "$TOOLS_ROOT/DLKcat"
}


install_chai1() {
    section "Chai-1"
    make_venv chai1 3.11
    uv_pip_install chai1 "chai_lab==0.6.1"
    "$(venv_script chai1 chai-lab)" --help >/dev/null
}

install_protenix() {
    section "Protenix-v2"
    make_venv protenix 3.11
    uv_pip_install protenix protenix
    "$(venv_script protenix protenix)" --help >/dev/null
}

install_esmfold2() {
    section "ESMFold 2"
    select_torch_backend
    make_venv esmfold2 3.11
    install_torch esmfold2 2.7.1 "$TORCH_BACKEND"
    run_uv pip install --python "$(venv_python esmfold2)" --no-build-isolation         "fair-esm[esmfold]~=2.0.0"         "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
    download_python_script "https://raw.githubusercontent.com/facebookresearch/esm/v2.0.0/scripts/esmfold_inference.py" "$(venv_script esmfold2 esm-fold)" esmfold2
}

install_boltzgen() {
    section "BoltzGen"
    make_venv boltzgen 3.12
    uv_pip_install boltzgen "boltzgen~=0.3.2"
    "$(venv_script boltzgen boltzgen)" --help >/dev/null
}

install_proteinmpnn_ddg() {
    section "ProteinMPNN-ddG"
    make_venv proteinmpnn-ddg 3.10
    uv_pip_install proteinmpnn-ddg         "ProteinMPNN-ddG[cuda12] @ git+https://github.com/PeptoneLtd/proteinmpnn_ddg.git@main"         "dm-haiku==0.0.13"
    download_python_script "https://raw.githubusercontent.com/PeptoneLtd/proteinmpnn_ddg/main/predict.py" "$(venv_script proteinmpnn-ddg proteinmpnn-ddg)" proteinmpnn-ddg
}


install_rfdiffusion() {
    section "RFdiffusion"
    make_venv rfdiffusion 3.10
    uv_pip_install rfdiffusion         "dgl @ https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl"         "numpy<2" "e3nn==0.3.3" hydra-core icecream opt_einsum scipy pandas decorator         pyrsistent "dllogger @ git+https://github.com/NVIDIA/dllogger.git@master"         "se3-transformer @ git+https://github.com/RosettaCommons/RFdiffusion.git@main#subdirectory=env/SE3Transformer"         "rfdiffusion @ git+https://github.com/RosettaCommons/RFdiffusion.git@main"
    target="$TOOLS_ROOT/RFdiffusion"
    clone_or_update https://github.com/RosettaCommons/RFdiffusion "$target"
    mkdir -p "$target/models"
    download https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt "$target/models/Base_ckpt.pt"
    download https://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt "$target/models/Complex_base_ckpt.pt"
    download https://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt "$target/models/Complex_Fold_base_ckpt.pt"
    download https://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt "$target/models/InpaintSeq_ckpt.pt"
    download https://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt "$target/models/InpaintSeq_Fold_ckpt.pt"
    download https://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt "$target/models/ActiveSite_ckpt.pt"
    download https://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt "$target/models/Base_epoch8_ckpt.pt"
}

install_rfantibody() {
    section "RFantibody"
    make_venv rfantibody 3.10
    uv_pip_install rfantibody         "dgl @ https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl"         "rfantibody @ git+https://github.com/RosettaCommons/RFantibody.git@main"
    target="$TOOLS_ROOT/RFantibody"
    clone_or_update https://github.com/RosettaCommons/RFantibody "$target"
    mkdir -p "$target/weights"
    download https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt "$target/weights/RFdiffusion_Ab.pt"
    download https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt "$target/weights/ProteinMPNN_v48_noise_0.2.pt"
    download https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt "$target/weights/RF2_ab.pt"
    download "https://zenodo.org/records/17488258/files/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt?download=1"         "$target/weights/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt"
}

install_igdesign() {
    section "IgDesign"
    make_venv igdesign 3.10
    uv_pip_install igdesign "igdesign @ git+https://github.com/AbSciBio/igdesign.git@main"
    target="$TOOLS_ROOT/igdesign"
    clone_or_update https://github.com/AbSciBio/igdesign "$target"
    if [ -f "$target/download_ckpts.sh" ]; then
        (cd "$target" && sh download_ckpts.sh)
    else
        note "IgDesign checkpoints must be downloaded manually into $target/ckpts."
    fi
}

install_catpred() {
    section "CatPred"
    make_venv catpred 3.10
    uv_pip_install catpred "catpred @ git+https://github.com/maranasgroup/CatPred.git@main"
    clone_or_update https://github.com/maranasgroup/CatPred "$TOOLS_ROOT/CatPred"
    note "CatPred checkpoints are a separate approximately 10 GiB download."
}

install_placer() {
    section "PLACER"
    make_venv placer 3.10
    uv_pip_install placer         "dgl @ https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl"         "torch==2.3.1" "opt_einsum==3.4.0" openbabel "networkx>=3.2" "numpy<2"         "pandas==2.2.3" "e3nn==0.5.4"
    clone_or_update https://github.com/baker-laboratory/PLACER "$TOOLS_ROOT/PLACER"
}


require_conda() {
    command -v conda >/dev/null 2>&1 ||
        fail "conda is required for this legacy scientific stack; install Miniconda and retry."
}

install_highfold() {
    section "HighFold"
    require_conda
    target="$TOOLS_ROOT/HighFold"
    prefix="$(venv_dir highfold)"
    clone_or_update https://github.com/hongliangduan/HighFold "$target"
    conda env remove --prefix "$prefix" -y >/dev/null 2>&1 || true
    conda create --prefix "$prefix" python=3.10 -y
    conda install --prefix "$prefix" -y -c conda-forge -c bioconda openmm pdbfixer kalign2 hhsuite
    conda run --prefix "$prefix" python -m pip install --upgrade "jax[cuda12]"
    conda run --prefix "$prefix" python -m pip install "$target"
}

install_antifold() {
    section "AntiFold"
    require_conda
    target="$TOOLS_ROOT/AntiFold"
    prefix="$(venv_dir antifold)"
    clone_or_update https://github.com/oxpig/AntiFold "$target"
    conda env remove --prefix "$prefix" -y >/dev/null 2>&1 || true
    conda create --prefix "$prefix" python=3.10 -y
    conda install --prefix "$prefix" -c conda-forge "pytorch==2.2.0" -y
    conda run --prefix "$prefix" python -m pip install "$target"
}

install_mber() {
    section "mBER"
    require_conda
    target="$TOOLS_ROOT/mber-open"
    prefix="$(venv_dir mber)"
    clone_or_update https://github.com/manifoldbio/mber-open "$target"
    conda env remove --prefix "$prefix" -y >/dev/null 2>&1 || true
    conda env create --prefix "$prefix" -f "$target/environment.yml"
    conda run --prefix "$prefix" python -m pip install -e "$target/protocols"
    (cd "$target" && sh download_weights.sh)
}

install_aggrescan3d() {
    section "AggreScan3D"
    require_conda
    prefix="$(venv_dir aggrescan3d)"
    conda env remove --prefix "$prefix" -y >/dev/null 2>&1 || true
    conda create --prefix "$prefix" python=2.7 -y
    conda run --prefix "$prefix" python -m pip install         "git+https://bitbucket.org/lcbio/aggrescan3d.git@master"
}



# --------------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------------

PORTABLE_TOOLS="opendde boltz2 ligandmpnn proteinmpnn igblast anarcii immunebuilder biophi thermompnn deepsp deepimmuno tlimmuno2 netsolp deepstabp dlkcat"
LINUX_ONLY_TOOLS="chai1 protenix esmfold2 boltzgen proteinmpnn-ddg rfdiffusion rfantibody igdesign catpred placer highfold antifold mber aggrescan3d"
if [ "$(uname -s)" = "Linux" ]; then
    ALL_TOOLS="$PORTABLE_TOOLS $LINUX_ONLY_TOOLS"
else
    ALL_TOOLS="$PORTABLE_TOOLS"
fi

usage() {
    cat <<EOF
Usage: $0 <tool>... | all | --list

Tools:
  opendde       All-atom co-folding (proteins, nucleic acids, ligands, ions, complexes).
  boltz2        Co-folding plus binding-affinity prediction.
  ligandmpnn    Inverse folding in ligand and nucleic-acid context.
  proteinmpnn   Inverse folding, the AbMPNN antibody weights, and native ΔΔG scanning.
  igblast       Antibody V(D)J germline assignment and CDR delineation.
  anarcii       Antibody/TCR numbering with insertion codes.
  immunebuilder Fast antibody, nanobody, and TCR structure prediction.
  biophi        Antibody humanization and humanness estimation.
  thermompnn    Protein mutation stability prediction.
  deepsp        Antibody developability descriptors.
  deepimmuno    Peptide-MHC-I immunogenicity prediction.
  tlimmuno2     Peptide-MHC-II immunogenicity prediction.
  netsolp       Protein solubility prediction (licensed checkpoints are separate).
  deepstabp     Protein melting-temperature prediction.
  dlkcat        Enzyme turnover prediction.
  chai1         Chai-1 complex structure prediction (Linux only).
  protenix       Protenix-v2 structure prediction (Linux only).
  esmfold2       ESMFold 2 single-sequence folding (Linux only).
  boltzgen       Protein and peptide binder design (Linux only).
  proteinmpnn-ddg Mutation stability scans (Linux only).
  rfdiffusion    Protein backbone design (Linux only).
  rfantibody     Antibody and nanobody design (Linux only).
  igdesign       Antibody CDR design (Linux only).
  catpred        Enzyme property prediction (Linux only).
  placer         Protein-ligand pose generation (Linux only).
  highfold       Cyclic-peptide structure prediction via Conda (Linux only).
  antifold       Antibody sequence design via Conda (Linux only).
  mber           VHH binder design via Conda (Linux only).
  aggrescan3d    Structural aggregation profiling via Conda (Linux only).

Installed under $DATA_ROOT
EOF
}

[ $# -eq 0 ] && { usage; exit 1; }

case "$1" in
    --list|-l) printf '%s\n' $ALL_TOOLS; exit 0 ;;
    --help|-h) usage; exit 0 ;;
    all) set -- $ALL_TOOLS ;;
esac

# Validated up front so that `install_tool.sh opendde typo` fails before spending ten minutes on
# the first tool.
for requested in "$@"; do
    case " $ALL_TOOLS " in
        *" $requested "*) ;;
        *) printf 'Unknown tool: %s\n\n' "$requested" >&2; usage >&2; exit 1 ;;
    esac
done

failed=""
for requested in "$@"; do
    # Each tool is attempted independently: a broken upstream release for one should not cost the
    # user the others, which is the same reason bio_web's environment installer keeps going.
    function_name="install_$(printf '%s' "$requested" | tr '-' '_')"
    if ! "$function_name"; then
        printf 'Installing %s failed.\n' "$requested" >&2
        failed="$failed $requested"
    fi
done

if [ -n "$failed" ]; then
    printf '\nFailed:%s\n' "$failed" >&2
    exit 1
fi

printf '\nDone. Restart Molchanica, then open the "Tools" panel to confirm.\n'
