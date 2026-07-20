[CmdletBinding()]
param(
    [string]$PythonExecutable = $env:OPENDDE_PYTHON,
    [string]$VenvDirectory = $env:OPENDDE_VENV_DIR,
    [string]$TorchBackend = $(if ($env:OPENDDE_TORCH_BACKEND) { $env:OPENDDE_TORCH_BACKEND } else { "auto" })
)

# Install OpenDDE in a dedicated standard-library Python virtual environment for Molchanica.
# CUDA 12.6 is selected automatically when a working NVIDIA GPU and sufficiently new Windows
# driver are present; otherwise this uses CPU. No activation or PATH changes are required.

$ErrorActionPreference = "Stop"

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$NativeArguments
    )

    & $Executable @NativeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($NativeArguments -join ' ')"
    }
}

function Find-NvidiaSmi {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:SystemRoot "System32\nvidia-smi.exe"),
        (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}

function Install-OpenDdeBackend {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$Backend
    )

    $package = if ($Backend -eq "cu126") { "opendde[gpu]" } else { "opendde" }
    $torchIndex = if ($Backend -eq "cu126") {
        "https://download.pytorch.org/whl/cu126"
    } else {
        "https://download.pytorch.org/whl/cpu"
    }

    Write-Host "Installing $package with the $Backend PyTorch backend..."
    # Match OpenDDE's currently pinned PyTorch packages while choosing their CPU/CUDA wheel index.
    & $Python "-m" "pip" "install" `
        "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" `
        "--index-url" $torchIndex | Out-Host
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    & $Python "-m" "pip" "install" $package | Out-Host
    return $LASTEXITCODE -eq 0
}

function Test-CompatiblePython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    try {
        & $Executable @PrefixArguments "-c" "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" `
            *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Find-CompatiblePython {
    if ($PythonExecutable) {
        if (Test-CompatiblePython $PythonExecutable) {
            return [PSCustomObject]@{
                Executable = $PythonExecutable
                PrefixArguments = @()
            }
        }
        throw "PythonExecutable must run Python 3.11 or newer."
    }

    $candidates = @(
        @{ Name = "py"; PrefixArguments = @("-3") },
        @{ Name = "python"; PrefixArguments = @() },
        @{ Name = "python3"; PrefixArguments = @() }
    )
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Name -ErrorAction SilentlyContinue
        if ($null -ne $command -and
            (Test-CompatiblePython $command.Source $candidate.PrefixArguments)) {
            return [PSCustomObject]@{
                Executable = $command.Source
                PrefixArguments = $candidate.PrefixArguments
            }
        }
    }

    throw "Python 3.11 or newer is required to install OpenDDE."
}

$python = Find-CompatiblePython
if ($TorchBackend -notin @("auto", "cpu", "cu126")) {
    throw "TorchBackend must be 'auto', 'cpu', or 'cu126'."
}

$selectedBackend = "cpu"
if ($TorchBackend -ne "cpu") {
    $nvidiaSmi = Find-NvidiaSmi
    if ($nvidiaSmi) {
        & $nvidiaSmi "-L" *> $null
        $gpuAvailable = $LASTEXITCODE -eq 0
        $driverOutput = & $nvidiaSmi "--query-gpu=driver_version" "--format=csv,noheader" 2> $null
        if ($LASTEXITCODE -eq 0 -and $driverOutput) {
            try {
                $driverVersion = [version](($driverOutput | Select-Object -First 1).Trim())
                if ($gpuAvailable -and $driverVersion -ge [version]"560.76") {
                    $selectedBackend = "cu126"
                    Write-Host "Detected an NVIDIA GPU with driver $driverVersion; selecting CUDA 12.6."
                } else {
                    Write-Host "The NVIDIA driver is older than 560.76; selecting CPU."
                }
            } catch {
                Write-Host "The NVIDIA driver version could not be parsed; selecting CPU."
            }
        }
    }
    if ($selectedBackend -eq "cpu" -and -not $nvidiaSmi) {
        Write-Host "No working NVIDIA driver was found; selecting CPU."
    } elseif ($selectedBackend -eq "cpu" -and (-not $gpuAvailable -or -not $driverOutput)) {
        Write-Host "The NVIDIA GPU or driver query failed; selecting CPU."
    }
}

if (-not $VenvDirectory) {
    if (-not $env:LOCALAPPDATA) {
        throw "LOCALAPPDATA is not set; provide VenvDirectory explicitly."
    }
    $VenvDirectory = Join-Path $env:LOCALAPPDATA "molchanica\opendde-venv"
}

$versionArguments = @($python.PrefixArguments) + @("--version")
$pythonVersion = (& $python.Executable @versionArguments 2>&1 | Out-String).Trim()
Write-Host "Using $pythonVersion"
Write-Host "Creating an isolated OpenDDE environment at $VenvDirectory..."
$venvArguments = @($python.PrefixArguments) + @("-m", "venv", "--clear", $VenvDirectory)
Invoke-CheckedNative $python.Executable @venvArguments

$venvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$openDde = Join-Path $VenvDirectory "Scripts\opendde.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "The virtual-environment Python was not created at $venvPython."
}

Invoke-CheckedNative $venvPython "-m" "pip" "install" "--upgrade" "pip"

$installed = Install-OpenDdeBackend $venvPython $selectedBackend
if ($installed -and $selectedBackend -eq "cu126") {
    & $venvPython "-c" `
        "import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith('12.6'); torch.zeros(1, device='cuda')"
    $installed = $LASTEXITCODE -eq 0
}

if (-not $installed -and $selectedBackend -eq "cu126") {
    Write-Warning "CUDA installation or runtime verification failed; rebuilding for CPU."
    $selectedBackend = "cpu"
    Invoke-CheckedNative $python.Executable @venvArguments
    Invoke-CheckedNative $venvPython "-m" "pip" "install" "--upgrade" "pip"
    $installed = Install-OpenDdeBackend $venvPython "cpu"
}
if (-not $installed) {
    throw "Unable to install the OpenDDE $selectedBackend backend."
}
$TorchBackend = $selectedBackend

if (-not (Test-Path -LiteralPath $openDde -PathType Leaf)) {
    throw "pip completed, but the OpenDDE executable was not created at $openDde."
}

Write-Host "`nVerifying the OpenDDE installation..."
Invoke-CheckedNative $openDde "--version"
Invoke-CheckedNative $openDde "doctor"

Write-Host "`nOpenDDE is installed in a dedicated Python virtual environment."
Write-Host "Executable: $openDde"
Write-Host "No activation is required. Restart Molchanica if it is currently running."
