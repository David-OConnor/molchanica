#!/usr/bin/env sh

# Install one of Molchanica's optional third-party tools.
#
#   ./install_tool.sh opendde
#   ./install_tool.sh ligandmpnn proteinmpnn
#   ./install_tool.sh all
#   ./install_tool.sh --list
#
# There is one script rather than one per tool because every one of them needs the same handful of
# things: a Python of a particular version, an isolated virtual environment, a Torch build matching
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
#   MOLCHANICA_PYTHON=/path/to/python forces the interpreter used to build environments

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
# Python
# --------------------------------------------------------------------------------------------

# Whether $1 is a Python whose version is at least $2.$3 and, when $4 is given, below $4.$5.
python_version_ok() {
    "$1" -c "
import sys
lo = ($2, $3)
hi = ${4:-99}, ${5:-0}
raise SystemExit(0 if lo <= sys.version_info[:2] < hi else 1)
" >/dev/null 2>&1
}

# Print the first interpreter satisfying the version window, preferring newer.
#
# Named interpreters are tried before bare `python3` because a distribution's default is often
# older than what it also ships; several of these tools care which minor version they get.
find_python() {
    min_major="$1"; min_minor="$2"; max_major="${3:-99}"; max_minor="${4:-0}"

    if [ -n "${MOLCHANICA_PYTHON:-}" ]; then
        if python_version_ok "$MOLCHANICA_PYTHON" "$min_major" "$min_minor" "$max_major" "$max_minor"; then
            printf '%s\n' "$MOLCHANICA_PYTHON"
            return 0
        fi
        fail "MOLCHANICA_PYTHON is not a Python between $min_major.$min_minor and $max_major.$max_minor."
    fi

    for name in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$name" >/dev/null 2>&1; then
            candidate="$(command -v "$name")"
            if python_version_ok "$candidate" "$min_major" "$min_minor" "$max_major" "$max_minor"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

# Create a clean environment for a tool. Always `--clear`: reusing one risks carrying over a
# package set built for a different interpreter or a different Torch backend.
make_venv() {
    slug="$1"; min_major="$2"; min_minor="$3"; max_major="${4:-99}"; max_minor="${5:-0}"

    python="$(find_python "$min_major" "$min_minor" "$max_major" "$max_minor" || true)"
    if [ -z "$python" ]; then
        window="$min_major.$min_minor or newer"
        [ "$max_major" != "99" ] && window="$min_major.$min_minor up to (not including) $max_major.$max_minor"
        fail "$slug needs Python $window, which was not found. Install it, or set MOLCHANICA_PYTHON."
    fi

    target="$(venv_dir "$slug")"
    note "Using $("$python" --version 2>&1)"
    note "Creating $target"
    mkdir -p "$(dirname "$target")"
    "$python" -m venv --clear "$target"
    [ -x "$(venv_python "$slug")" ] || fail "the virtual environment at $target has no interpreter."
    "$(venv_python "$slug")" -m pip install --upgrade pip >/dev/null
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
        "$(venv_python "$slug")" -m pip install "torch==$version"
    else
        "$(venv_python "$slug")" -m pip install "torch==$version" \
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

# OpenDDE: all-atom co-folding. Python >= 3.11.
install_opendde() {
    section "OpenDDE"
    select_torch_backend
    make_venv opendde 3 11

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
            "$(venv_python opendde)" -m pip install \
                "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" || return 1
        else
            "$(venv_python opendde)" -m pip install \
                "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
                --index-url "$(torch_index_url "$backend")" || return 1
        fi
        "$(venv_python opendde)" -m pip install "$package" || return 1
    }

    if [ "$TORCH_BACKEND" = "cu126" ]; then
        if install_backend_opendde cu126 && "$(venv_python opendde)" -c \
            'import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith("12.6"); torch.zeros(1, device="cuda")'
        then
            :
        else
            note "CUDA installation or runtime verification failed; rebuilding for CPU."
            TORCH_BACKEND="cpu"
            make_venv opendde 3 11
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
    make_venv boltz2 3 10 3 13

    install_torch boltz2 2.7.1 "$TORCH_BACKEND"

    # The [cuda] extra pulls cuequivariance wheels that exist for Linux x86_64 only, so it is
    # requested only where it can resolve. Plain `boltz` is a pure-Python wheel and works
    # everywhere, GPU included — the extra is a speed-up, not a requirement.
    if [ "$TORCH_BACKEND" = "cu126" ] && [ "$(uname -s)" = "Linux" ]; then
        if ! "$(venv_python boltz2)" -m pip install "boltz[cuda]~=2.2.1"; then
            note "The CUDA extra did not resolve; installing Boltz without it."
            "$(venv_python boltz2)" -m pip install "boltz~=2.2.1"
        fi
    else
        "$(venv_python boltz2)" -m pip install "boltz~=2.2.1"
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
    make_venv "$slug" 3 10 3 13
    install_torch "$slug" 2.7.1 "$TORCH_BACKEND"
    "$(venv_python "$slug")" -m pip install "numpy<2"
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
    make_venv anarcii 3 11
    install_torch anarcii 2.7.1 "$TORCH_BACKEND"
    "$(venv_python anarcii)" -m pip install anarcii

    note "Verifying"
    "$(venv_python anarcii)" -c 'import anarcii; print("anarcii", anarcii.__version__)' \
        || fail "anarcii was installed but cannot be imported."
    note "ANARCII installed."
}

# --------------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------------

ALL_TOOLS="opendde boltz2 ligandmpnn proteinmpnn igblast anarcii"

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
    if ! "install_$requested"; then
        printf 'Installing %s failed.\n' "$requested" >&2
        failed="$failed $requested"
    fi
done

if [ -n "$failed" ]; then
    printf '\nFailed:%s\n' "$failed" >&2
    exit 1
fi

printf '\nDone. Restart Molchanica, then open the "Tools" panel to confirm.\n'
