[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Tools
)

# Install one of Molchanica's optional third-party tools. The Windows counterpart of
# install_tool.sh, which documents the design; both take the same tool names and install to the
# locations src/external_tools/mod.rs looks in.
#
#   .\install_tool.ps1 opendde
#   .\install_tool.ps1 ligandmpnn proteinmpnn
#   .\install_tool.ps1 all
#   .\install_tool.ps1 --list
#
# Optional overrides:
#   $env:MOLCHANICA_DATA_DIR         where everything is installed
#   $env:MOLCHANICA_TORCH_BACKEND    auto | cpu | cu126
#   $env:MOLCHANICA_UV               uses an existing uv executable

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------------------------

if ($env:MOLCHANICA_DATA_DIR) {
    $DataRoot = $env:MOLCHANICA_DATA_DIR
} else {
    if (-not $env:LOCALAPPDATA) {
        throw "LOCALAPPDATA is not set; set MOLCHANICA_DATA_DIR explicitly."
    }
    $DataRoot = Join-Path $env:LOCALAPPDATA "molchanica"
}
$ToolsRoot = Join-Path $DataRoot "tools"

function Get-VenvDir {
    param([string]$Slug)
    # OPENDDE_VENV_DIR predates this script and is documented, so it still wins for that one tool.
    if ($Slug -eq "opendde" -and $env:OPENDDE_VENV_DIR) { return $env:OPENDDE_VENV_DIR }
    Join-Path $DataRoot "$Slug-venv"
}

function Get-VenvPython {
    param([string]$Slug)
    Join-Path (Get-VenvDir $Slug) "Scripts\python.exe"
}

function Get-VenvScript {
    param([string]$Slug, [string]$Name)
    Join-Path (Get-VenvDir $Slug) "Scripts\$Name.exe"
}

# ---------------------------------------------------------------------------------------------
# Output and process helpers
# ---------------------------------------------------------------------------------------------

function Write-Section { param([string]$Text) Write-Host "`n$Text`n========================================================" }
function Write-Note { param([string]$Text) Write-Host "  $Text" }

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Executable @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

# Run a command, returning whether it succeeded instead of throwing. For the places where a
# failure is a decision point rather than an error — the CUDA probe, the optional weight
# conversion.
function Test-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    try {
        & $Executable @Arguments *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------------------------------------
# uv and Python
# ---------------------------------------------------------------------------------------------

$script:UvExecutable = $null

# Locate uv, or install it into Molchanica's data directory with Astral's official standalone
# installer. The explicit managed path works even when a desktop launch does not inherit PATH.
function Get-Uv {
    if ($script:UvExecutable) { return $script:UvExecutable }

    if ($env:MOLCHANICA_UV) {
        if (-not (Test-Command $env:MOLCHANICA_UV "--version")) {
            throw "MOLCHANICA_UV does not name a working uv executable."
        }
        $script:UvExecutable = $env:MOLCHANICA_UV
        return $script:UvExecutable
    }

    $managedUv = Join-Path $DataRoot "uv-bin\uv.exe"
    if ((Test-Path -LiteralPath $managedUv -PathType Leaf) -and (Test-Command $managedUv "--version")) {
        $script:UvExecutable = $managedUv
        return $script:UvExecutable
    }

    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $command -and (Test-Command $command.Source "--version")) {
        $script:UvExecutable = $command.Source
        return $script:UvExecutable
    }

    # Astral's default installer location is not always on PATH in a non-login shell.
    $fallbackUv = Join-Path $HOME ".local\bin\uv.exe"
    if ((Test-Path -LiteralPath $fallbackUv -PathType Leaf) -and (Test-Command $fallbackUv "--version")) {
        $script:UvExecutable = $fallbackUv
        return $script:UvExecutable
    }

    Write-Section "Installing uv"
    $uvDirectory = Join-Path $DataRoot "uv-bin"
    New-Item -ItemType Directory -Force $uvDirectory | Out-Null
    $previousInstall = [Environment]::GetEnvironmentVariable("UV_UNMANAGED_INSTALL", "Process")
    try {
        $env:UV_UNMANAGED_INSTALL = $uvDirectory
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } finally {
        if ($null -eq $previousInstall) {
            Remove-Item Env:UV_UNMANAGED_INSTALL -ErrorAction SilentlyContinue
        } else {
            $env:UV_UNMANAGED_INSTALL = $previousInstall
        }
    }

    if (-not (Test-Path -LiteralPath $managedUv -PathType Leaf) -or
        -not (Test-Command $managedUv "--version")) {
        throw "Astral's installer completed, but uv was not found at $managedUv."
    }
    $script:UvExecutable = $managedUv
    return $script:UvExecutable
}

# Run uv without allowing an activated venv, Conda, or user Python-selection variables to redirect
# it. In particular, UV_NO_MANAGED_PYTHON must not contradict New-ToolVenv's guarantee below.
function Invoke-UvChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $names = @(
        "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH",
        "UV_PYTHON", "UV_NO_MANAGED_PYTHON"
    )
    $previous = @{}
    foreach ($name in $names) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
    try {
        Invoke-Checked (Get-Uv) @Arguments
    } finally {
        foreach ($name in $names) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
        }
    }
}

function Install-PythonPackages {
    param(
        [Parameter(Mandatory = $true)][string]$Slug,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    $uvArguments = @("pip", "install", "--python", (Get-VenvPython $Slug)) + $Arguments
    Invoke-UvChecked @uvArguments
}

# Create a clean environment for a tool using an exact, uv-managed Python minor version. The
# `--managed-python` flag prevents uv from silently selecting a matching system interpreter.
function New-ToolVenv {
    param([string]$Slug, [string]$PythonVersion)

    $target = Get-VenvDir $Slug
    Write-Note "Creating $target with uv-managed Python $PythonVersion"

    New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
    Invoke-UvChecked "venv" "--managed-python" "--python" $PythonVersion "--clear" $target

    $venvPython = Get-VenvPython $Slug
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "The virtual environment at $target has no interpreter."
    }
    $version = (& $venvPython "--version" 2>&1 | Out-String).Trim()
    Write-Note "Using $version"
}

# ---------------------------------------------------------------------------------------------
# Torch backend selection
# ---------------------------------------------------------------------------------------------

# CUDA 12.6 wheels need at least this Windows driver. Below it the wheels install and then fail at
# run time, so it is checked before installing rather than after.
$WindowsMinDriver = [version]"560.76"

function Find-NvidiaSmi {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    $candidates = @(
        (Join-Path $env:SystemRoot "System32\nvidia-smi.exe"),
        (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Select-TorchBackend {
    $requested = if ($env:MOLCHANICA_TORCH_BACKEND) { $env:MOLCHANICA_TORCH_BACKEND } else { "auto" }
    if ($requested -notin @("auto", "cpu", "cu126")) {
        throw "MOLCHANICA_TORCH_BACKEND must be auto, cpu, or cu126."
    }
    if ($requested -eq "cpu") { return "cpu" }

    $nvidiaSmi = Find-NvidiaSmi
    if (-not $nvidiaSmi) {
        if ($requested -eq "cu126") { throw "CUDA 12.6 was requested, but no NVIDIA driver was found." }
        Write-Note "No NVIDIA driver was found; selecting CPU."
        return "cpu"
    }

    $gpuAvailable = Test-Command $nvidiaSmi "-L"
    $driverOutput = & $nvidiaSmi "--query-gpu=driver_version" "--format=csv,noheader" 2> $null
    if ($LASTEXITCODE -ne 0 -or -not $driverOutput) {
        if ($requested -eq "cu126") { throw "CUDA 12.6 was requested, but the driver version could not be read." }
        Write-Note "The NVIDIA GPU or driver query failed; selecting CPU."
        return "cpu"
    }

    try {
        $driverVersion = [version](($driverOutput | Select-Object -First 1).Trim())
    } catch {
        if ($requested -eq "cu126") { throw "CUDA 12.6 was requested, but the driver version could not be parsed." }
        Write-Note "The NVIDIA driver version could not be parsed; selecting CPU."
        return "cpu"
    }

    if ($gpuAvailable -and $driverVersion -ge $WindowsMinDriver) {
        Write-Note "Detected an NVIDIA GPU with driver $driverVersion; selecting CUDA 12.6."
        return "cu126"
    }
    if ($requested -eq "cu126") {
        throw "CUDA 12.6 needs an NVIDIA driver of at least $WindowsMinDriver; this system has $driverVersion."
    }
    Write-Note "The NVIDIA driver is older than $WindowsMinDriver; selecting CPU."
    return "cpu"
}

function Get-TorchIndexUrl {
    param([string]$Backend)
    if ($Backend -eq "cu126") { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }
}

# Pinned rather than left to the resolver, so the CPU/CUDA choice above is actually honoured: an
# unpinned `pip install torch` takes whatever default wheel PyPI serves.
function Install-Torch {
    param([string]$Slug, [string]$Version, [string]$Backend)
    Install-PythonPackages $Slug "torch==$Version" `
        "--index-url" (Get-TorchIndexUrl $Backend)
}

# A CUDA wheel that cannot reach a device is the failure worth catching here: everything installs
# cleanly and only the first real run reports it.
function Test-TorchCuda {
    param([string]$Slug)
    Test-Command (Get-VenvPython $Slug) "-c" `
        'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda")'
}

# ---------------------------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------------------------

function Get-RemoteFile {
    param([string]$Url, [string]$Destination)
    if ((Test-Path -LiteralPath $Destination -PathType Leaf) -and (Get-Item $Destination).Length -gt 0) {
        Write-Note "Already have $(Split-Path -Leaf $Destination)"
        return
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $Destination) | Out-Null
    Write-Note "Downloading $(Split-Path -Leaf $Destination)"
    $partial = "$Destination.partial"
    # The progress bar makes Invoke-WebRequest an order of magnitude slower on large files.
    $previous = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
    } finally {
        $ProgressPreference = $previous
    }
    # Renamed only on success, so an interrupted download is retried rather than treated as done.
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

function Sync-Checkout {
    param([string]$Url, [string]$Target)
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is required to install this tool."
    }
    if (Test-Path -LiteralPath (Join-Path $Target ".git")) {
        Write-Note "Updating $(Split-Path -Leaf $Target)"
        Invoke-Checked git "-C" $Target "fetch" "--depth" "1" "origin" "HEAD"
        Invoke-Checked git "-C" $Target "reset" "--hard" "FETCH_HEAD"
    } else {
        Write-Note "Cloning $(Split-Path -Leaf $Target)"
        New-Item -ItemType Directory -Force (Split-Path -Parent $Target) | Out-Null
        Invoke-Checked git "clone" "--depth" "1" $Url $Target
    }
}

# ---------------------------------------------------------------------------------------------
# Per-tool installers
# ---------------------------------------------------------------------------------------------

function Install-Opendde {
    Write-Section "OpenDDE"
    $backend = Select-TorchBackend
    New-ToolVenv "opendde" "3.13"

    $install = {
        param([string]$Backend)
        $package = if ($Backend -eq "cu126") { "opendde[gpu]" } else { "opendde" }
        Write-Note "Installing $package with the $Backend PyTorch backend"
        $python = Get-VenvPython "opendde"
        try {
            # OpenDDE pins this trio; the index decides CPU or CUDA.
            Invoke-UvChecked "pip" "install" "--python" $python "torch==2.7.1" `
                "torchvision==0.22.1" "torchaudio==2.7.1" `
                "--index-url" (Get-TorchIndexUrl $Backend)
            Invoke-UvChecked "pip" "install" "--python" $python $package
            return $true
        } catch {
            Write-Warning $_
            return $false
        }
    }

    $installed = & $install $backend
    if ($installed -and $backend -eq "cu126") {
        $installed = Test-Command (Get-VenvPython "opendde") "-c" `
            "import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith('12.6'); torch.zeros(1, device='cuda')"
    }
    if (-not $installed -and $backend -eq "cu126") {
        Write-Warning "CUDA installation or runtime verification failed; rebuilding for CPU."
        $backend = "cpu"
        New-ToolVenv "opendde" "3.13"
        $installed = & $install "cpu"
    }
    if (-not $installed) { throw "Unable to install the OpenDDE $backend backend." }

    $opendde = Get-VenvScript "opendde" "opendde"
    if (-not (Test-Path -LiteralPath $opendde -PathType Leaf)) {
        throw "pip completed, but $opendde was not created."
    }

    Write-Note "Verifying"
    Invoke-Checked $opendde "--version"
    Invoke-Checked $opendde "doctor"

    Initialize-OpenddeModelData $opendde
    Write-Note "OpenDDE installed. No activation is required."
}

# Fetch the model checkpoint now rather than on the user's first prediction.
#
# `opendde pred` downloads what it needs when missing, so this is not required — but without it
# the first prediction stalls for a multi-gigabyte download inside what looks like a hung job.
# Search is fully disabled, matching how Molchanica invokes OpenDDE, so this fetches the checkpoint
# and common files and not the far larger template/MSA databases.
function Initialize-OpenddeModelData {
    param([string]$Opendde)

    $root = if ($env:OPENDDE_ROOT_DIR) { $env:OPENDDE_ROOT_DIR } else { Join-Path $HOME ".cache\opendde" }
    if (Test-Path -LiteralPath (Join-Path $root "checkpoint")) {
        Write-Note "OpenDDE model data is already present."
        return
    }

    Write-Note "Fetching the OpenDDE model checkpoint (several GB; this is a one-time download)."
    $workdir = Join-Path ([System.IO.Path]::GetTempPath()) ("molchanica-prewarm-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force $workdir | Out-Null
    try {
        $json = '[{"name": "prewarm", "modelSeeds": [101], "sequences": [{"proteinChain": {"sequence": "ACDEFG", "count": 1, "id": ["A"]}}]}]'
        Set-Content -LiteralPath (Join-Path $workdir "input.json") -Value $json -Encoding utf8
        Push-Location $workdir
        try {
            # Whether the job converges is irrelevant; it exists to trigger the download.
            $ok = Test-Command $Opendde "pred" "-i" "input.json" "-o" "output" "-n" "opendde_v1" `
                "--use_msa" "false" "--use_template" "false" "--use_rna_msa" "false" `
                "--sample" "1" "--step" "1" "--cycle" "1"
        } finally {
            Pop-Location
        }
        if ($ok) {
            Write-Note "Model data cached; the first real prediction will start immediately."
        } else {
            Write-Note "Could not pre-fetch the model data. It will download on the first prediction instead."
        }
    } finally {
        Remove-Item -Recurse -Force -LiteralPath $workdir -ErrorAction SilentlyContinue
    }
}

function Install-Boltz2 {
    Write-Section "Boltz-2"
    $backend = Select-TorchBackend
    # The upper bound is Boltz's own requires-python, and is why this cannot share OpenDDE's
    # environment.
    New-ToolVenv "boltz2" "3.12"
    Install-Torch "boltz2" "2.7.1" $backend

    # The [cuda] extra pulls cuequivariance wheels published for Linux only, so Windows always
    # takes the plain package. That is a speed-up forgone, not a capability: the pure-Python wheel
    # still uses the GPU through Torch.
    Install-PythonPackages "boltz2" "boltz~=2.2.1"

    $boltz = Get-VenvScript "boltz2" "boltz"
    if (-not (Test-Path -LiteralPath $boltz -PathType Leaf)) {
        throw "pip completed, but $boltz was not created."
    }
    Write-Note "Verifying"
    if (-not (Test-Command $boltz "--help")) { throw "boltz was installed but does not run." }
    if ($backend -eq "cu126" -and -not (Test-TorchCuda "boltz2")) {
        Write-Warning "Torch cannot reach the GPU; Boltz will run on CPU."
    }
    Write-Note "Boltz-2 installed. Model weights download on first use."
}

# The Torch runtime shared by both MPNN checkouts.
#
# numpy is held below 2 because the MPNN code predates the numpy 2 API removals, and that caps
# Python at 3.12: numpy 1.26 publishes no cp313 wheel.
function Install-MpnnRuntime {
    param([string]$Slug)
    $backend = Select-TorchBackend
    New-ToolVenv $Slug "3.12"
    Install-Torch $Slug "2.7.1" $backend
    Install-PythonPackages $Slug "numpy<2"
    if ($backend -eq "cu126" -and -not (Test-TorchCuda $Slug)) {
        Write-Warning "Torch cannot reach the GPU; designs will run on CPU."
    }
}

function Install-Ligandmpnn {
    Write-Section "LigandMPNN"
    Install-MpnnRuntime "ligandmpnn"

    $target = Join-Path $ToolsRoot "LigandMPNN"
    Sync-Checkout "https://github.com/dauparas/LigandMPNN" $target

    # Fetched directly rather than through the repository's get_model_params.sh, which is bash and
    # would not run here. These are the checkpoints src/external_tools/mpnn.rs selects between.
    foreach ($weights in @("ligandmpnn_v_32_010_25.pt", "proteinmpnn_v_48_020.pt", "solublempnn_v_48_020.pt")) {
        Get-RemoteFile "https://files.ipd.uw.edu/pub/ligandmpnn/$weights" (Join-Path $target "model_params\$weights")
    }
    if (-not (Test-Path -LiteralPath (Join-Path $target "model_params\ligandmpnn_v_32_010_25.pt"))) {
        throw "The LigandMPNN weights did not download."
    }
    Write-Note "LigandMPNN installed at $target"
}

function Install-Proteinmpnn {
    Write-Section "ProteinMPNN (and the AbMPNN weights)"
    Install-MpnnRuntime "proteinmpnn"

    $target = Join-Path $ToolsRoot "ProteinMPNN"
    Sync-Checkout "https://github.com/dauparas/ProteinMPNN" $target
    # The vanilla checkpoints are committed to the repository, so cloning is the whole install for
    # them; only AbMPNN has to be fetched.
    if (-not (Test-Path -LiteralPath (Join-Path $target "vanilla_model_weights\v_48_020.pt"))) {
        throw "The ProteinMPNN checkout has no vanilla_model_weights\v_48_020.pt."
    }

    # AbMPNN: ProteinMPNN's architecture finetuned on antibodies (Frey et al., ICML 2023 CompBio
    # workshop), CC BY 4.0. Named v_48_020.pt so --model_name's default matches for both weight
    # sets and the adapter never has to pass that flag; see src/external_tools/mpnn.rs.
    Get-RemoteFile "https://zenodo.org/records/8164693/files/abmpnn.pt?download=1" `
        (Join-Path $target "abmpnn_weights\v_48_020.pt")

    Convert-MpnnWeights $target
    Write-Note "ProteinMPNN installed at $target"
}

# Optional: skipping it costs only the ΔΔG feature, so a failure must not fail the install of the
# design tool the user actually asked for.
function Convert-MpnnWeights {
    param([string]$Target)
    # Two layouts: the repository, where this script is in install_scripts\ and the converter is
    # in scripts\; and a release archive, which is flat.
    $candidates = @(
        (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\convert_mpnn_weights.py"),
        (Join-Path $PSScriptRoot "convert_mpnn_weights.py")
    )
    $converter = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $converter) {
        Write-Note "convert_mpnn_weights.py not found; skipping the native ddG weight conversion."
        return
    }
    Write-Note "Converting the checkpoint for the native ΔΔG scanner"
    $ok = Test-Command (Get-VenvPython "proteinmpnn") $converter `
        "--checkpoint" (Join-Path $Target "vanilla_model_weights\v_48_020.pt") `
        "--output" (Join-Path $Target "converted\v_48_020.mcnn") `
        "--repo" $Target
    if ($ok) {
        Write-Note "Native ΔΔG scanning is available."
    } else {
        Write-Note "The conversion failed; ΔΔG scanning will be unavailable. The design tools still work."
    }
}

$IgblastVersion = if ($env:IGBLAST_VERSION) { $env:IGBLAST_VERSION } else { "1.22.0" }
$NcbiIgblastFtp = "https://ftp.ncbi.nih.gov/blast/executables/igblast/release"

function Install-Igblast {
    Write-Section "IgBLAST $IgblastVersion"
    $target = Join-Path $ToolsRoot "igblast"
    $versionMarker = Join-Path $target ".version"

    $installed = (Test-Path -LiteralPath $versionMarker) -and
                 ((Get-Content -LiteralPath $versionMarker -Raw).Trim() -eq $IgblastVersion)
    if ($installed) {
        Write-Note "IgBLAST $IgblastVersion is already installed."
    } else {
        $tarball = "ncbi-igblast-$IgblastVersion-x64-win64.tar.gz"
        $staging = Join-Path ([System.IO.Path]::GetTempPath()) ("molchanica-igblast-" + [guid]::NewGuid())
        New-Item -ItemType Directory -Force $staging | Out-Null
        try {
            Get-RemoteFile "$NcbiIgblastFtp/$IgblastVersion/$tarball" (Join-Path $staging $tarball)
            # bsdtar ships with Windows 10 1803 and later, and handles .tar.gz natively.
            Invoke-Checked tar "-xzf" (Join-Path $staging $tarball) "-C" $staging

            $unpacked = Join-Path $staging "ncbi-igblast-$IgblastVersion"
            if (-not (Test-Path -LiteralPath (Join-Path $unpacked "bin\igblastn.exe")) -or
                -not (Test-Path -LiteralPath (Join-Path $unpacked "internal_data"))) {
                throw "Unexpected tarball layout under $unpacked."
            }

            # Germline databases survive a version bump; they are large and independent of the
            # binary.
            $germline = Join-Path $target "germline_db"
            $preserved = $null
            if (Test-Path -LiteralPath $germline) {
                $preserved = Join-Path $staging "germline_db"
                Move-Item -LiteralPath $germline -Destination $preserved
            }

            New-Item -ItemType Directory -Force $ToolsRoot | Out-Null
            if (Test-Path -LiteralPath $target) { Remove-Item -Recurse -Force -LiteralPath $target }
            # Installed whole: igblastn resolves internal_data/ and optional_file/ relative to
            # IGDATA, so cherry-picking the binaries out would leave it unable to annotate.
            Copy-Item -Recurse -LiteralPath $unpacked -Destination $target
            if ($preserved) { Move-Item -LiteralPath $preserved -Destination $germline }
            Set-Content -LiteralPath $versionMarker -Value $IgblastVersion
        } finally {
            Remove-Item -Recurse -Force -LiteralPath $staging -ErrorAction SilentlyContinue
        }
    }

    Install-IgblastDatabases $target
    Write-Note "Verifying"
    $env:IGDATA = $target
    Invoke-Checked (Join-Path $target "bin\igblastn.exe") "-version"
    Write-Note "IgBLAST installed at $target"
}

# NCBI publishes germline databases already built with makeblastdb, so nothing here needs
# edit_imgt_file.pl or a makeblastdb run. Each archive unpacks flat.
function Install-IgblastDatabases {
    param([string]$Target)
    $germline = Join-Path $Target "germline_db"
    $existing = @(Get-ChildItem -LiteralPath $germline -Recurse -Include *.nhr, *.phr -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) {
        Write-Note "Germline databases are already installed."
        return
    }

    Write-Note "Installing the germline databases"
    New-Item -ItemType Directory -Force $germline | Out-Null
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) ("molchanica-germline-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force $staging | Out-Null
    try {
        $archives = @(
            "database/airr/airr_c_human.tar",
            "database/airr/airr_c_mouse.tar",
            "database/mouse_gl_VDJ.tar",
            "database/rhesus_monkey_VJ.tar",
            "database/ncbi_human_c_genes.tar"
        )
        foreach ($archive in $archives) {
            $filename = Split-Path -Leaf $archive
            Get-RemoteFile "$NcbiIgblastFtp/$archive" (Join-Path $staging $filename)
            Invoke-Checked tar "-xf" (Join-Path $staging $filename) "-C" $germline
        }
    } finally {
        Remove-Item -Recurse -Force -LiteralPath $staging -ErrorAction SilentlyContinue
    }

    $installed = @(Get-ChildItem -LiteralPath $germline -Recurse -Include *.nhr, *.phr -ErrorAction SilentlyContinue)
    if ($installed.Count -eq 0) { throw "No BLAST databases landed in $germline." }
}

# ANARCII: antibody/TCR numbering. A pure-Python wheel whose only heavy dependency is Torch, which
# is what makes proper numbering available on Windows at all — the older ANARCI stack needs HMMER.
function Install-Anarcii {
    Write-Section "ANARCII"
    $backend = Select-TorchBackend
    New-ToolVenv "anarcii" "3.12"
    Install-Torch "anarcii" "2.7.1" $backend
    Install-PythonPackages "anarcii" "anarcii"

    Write-Note "Verifying"
    Invoke-Checked (Get-VenvPython "anarcii") "-c" 'import anarcii; print("anarcii", anarcii.__version__)'
    Write-Note "ANARCII installed."
}
function Install-Immunebuilder {
    Write-Section "ImmuneBuilder"
    New-ToolVenv "immunebuilder" "3.11"
    Install-PythonPackages "immunebuilder" "ImmuneBuilder" "openmm" "pdbfixer" "anarci"
    Invoke-Checked (Get-VenvScript "immunebuilder" "ABodyBuilder2") "--help"
}

function Install-Biophi {
    Write-Section "BioPhi"
    New-ToolVenv "biophi" "3.11"
    Install-PythonPackages "biophi" "biophi @ git+https://github.com/Merck/BioPhi@main" "abnumber"
    Invoke-Checked (Get-VenvScript "biophi" "biophi") "--help"
}

function Install-Thermompnn {
    Write-Section "ThermoMPNN"
    $backend = Select-TorchBackend
    New-ToolVenv "thermompnn" "3.12"
    Install-Torch "thermompnn" "2.7.1" $backend
    Install-PythonPackages "thermompnn" "numpy<2" "pandas" "biopython" "tqdm" "omegaconf" "pytorch-lightning"
    Sync-Checkout "https://github.com/Kuhlman-Lab/ThermoMPNN" (Join-Path $ToolsRoot "ThermoMPNN")
}

function Install-Deepsp {
    Write-Section "DeepSP"
    $backend = Select-TorchBackend
    New-ToolVenv "deepsp" "3.11"
    Install-Torch "deepsp" "2.7.1" $backend
    Install-PythonPackages "deepsp" "tensorflow" "pandas" "numpy" "biopython" "anarcii"
    Sync-Checkout "https://github.com/Lailabcode/DeepSP" (Join-Path $ToolsRoot "DeepSP")
}

function Install-Deepimmuno {
    Write-Section "DeepImmuno"
    New-ToolVenv "deepimmuno" "3.10"
    Install-PythonPackages "deepimmuno" "tensorflow<2.16" "pandas" "numpy<2" "scikit-learn"
    Sync-Checkout "https://github.com/frankligy/DeepImmuno" (Join-Path $ToolsRoot "DeepImmuno")
}

function Install-Tlimmuno2 {
    Write-Section "TLimmuno2"
    New-ToolVenv "tlimmuno2" "3.10"
    Install-PythonPackages "tlimmuno2" "tensorflow<2.16" "pandas" "pyarrow" "numpy<2" "scikit-learn"
    Sync-Checkout "https://github.com/XSLiuLab/TLimmuno2" (Join-Path $ToolsRoot "TLimmuno2")
}

function Install-Netsolp {
    Write-Section "NetSolP"
    $backend = Select-TorchBackend
    New-ToolVenv "netsolp" "3.11"
    Install-Torch "netsolp" "2.7.1" $backend
    Install-PythonPackages "netsolp" "fair-esm~=2.0.0" "pandas" "numpy<2"
    Sync-Checkout "https://github.com/tvinet/NetSolP-1.0" (Join-Path $ToolsRoot "NetSolP-1.0")
    Write-Note "NetSolP model checkpoints require separate DTU licence acceptance."
}

function Install-Deepstabp {
    Write-Section "DeepSTABp"
    $backend = Select-TorchBackend
    New-ToolVenv "deepstabp" "3.11"
    Install-Torch "deepstabp" "2.7.1" $backend
    Install-PythonPackages "deepstabp" "transformers<5" "sentencepiece" "protobuf" "biopython" "pandas" "pytorch-lightning"
    Sync-Checkout "https://github.com/CSBiology/deepStabP" (Join-Path $ToolsRoot "deepStabP")
}

function Install-Dlkcat {
    Write-Section "DLKcat"
    $backend = Select-TorchBackend
    New-ToolVenv "dlkcat" "3.10"
    Install-Torch "dlkcat" "2.7.1" $backend
    Install-PythonPackages "dlkcat" "numpy<2" "rdkit" "scikit-learn"
    Sync-Checkout "https://github.com/SysBioChalmers/DLKcat" (Join-Path $ToolsRoot "DLKcat")
}



# ---------------------------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------------------------

$AllTools = @(
    "opendde", "boltz2", "ligandmpnn", "proteinmpnn", "igblast", "anarcii",
    "immunebuilder", "biophi", "thermompnn", "deepsp", "deepimmuno", "tlimmuno2",
    "netsolp", "deepstabp", "dlkcat"
)

function Show-Usage {
    Write-Host @"
Usage: .\install_tool.ps1 <tool>... | all | --list

Tools:
  opendde       All-atom co-folding (proteins, nucleic acids, ligands, ions, complexes).
  boltz2        Co-folding plus binding-affinity prediction.
  ligandmpnn    Inverse folding in ligand and nucleic-acid context.
  proteinmpnn   Inverse folding, the AbMPNN antibody weights, and native ddG scanning.
  igblast       Antibody V(D)J germline assignment and CDR delineation.
  immunebuilder Fast antibody, nanobody, and TCR structure prediction.
  biophi        Antibody humanization and humanness estimation.
  thermompnn    Protein mutation stability prediction.
  deepsp        Antibody developability descriptors.
  deepimmuno    Peptide-MHC-I immunogenicity prediction.
  tlimmuno2     Peptide-MHC-II immunogenicity prediction.
  netsolp       Protein solubility prediction (licensed checkpoints are separate).
  deepstabp     Protein melting-temperature prediction.
  dlkcat        Enzyme turnover prediction.
  anarcii       Antibody/TCR numbering with insertion codes.

Installed under $DataRoot
"@
}

if (-not $Tools -or $Tools.Count -eq 0) { Show-Usage; exit 1 }

switch ($Tools[0]) {
    { $_ -in @("--list", "-l") } { $AllTools | ForEach-Object { Write-Host $_ }; exit 0 }
    { $_ -in @("--help", "-h") } { Show-Usage; exit 0 }
    "all" { $Tools = $AllTools }
}

# Validated up front so a typo fails before spending ten minutes on the first tool.
foreach ($requested in $Tools) {
    if ($requested -notin $AllTools) {
        Write-Error "Unknown tool: $requested"
        Show-Usage
        exit 1
    }
}

$failed = @()
foreach ($requested in $Tools) {
    # Each tool is attempted independently: a broken upstream release for one should not cost the
    # user the others.
    $function = "Install-" + $requested.Substring(0, 1).ToUpper() + $requested.Substring(1)
    try {
        & $function
    } catch {
        Write-Error "Installing $requested failed: $_"
        $failed += $requested
    }
}

if ($failed.Count -gt 0) {
    Write-Error "Failed: $($failed -join ', ')"
    exit 1
}

Write-Host "`nDone. Restart Molchanica, then open the `"Tools`" panel to confirm."
