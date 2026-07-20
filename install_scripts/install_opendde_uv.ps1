[CmdletBinding()]
param(
    [string]$PythonVersion = $(if ($env:OPENDDE_PYTHON_VERSION) { $env:OPENDDE_PYTHON_VERSION } else { "3.11" }),
    [string]$TorchBackend = $(if ($env:OPENDDE_TORCH_BACKEND) { $env:OPENDDE_TORCH_BACKEND } else { "auto" })
)

# Install OpenDDE as an isolated uv tool for Molchanica. CUDA 12.6 is selected automatically when
# a working NVIDIA GPU and sufficiently new Windows driver are present; otherwise this uses CPU.

$ErrorActionPreference = "Stop"

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $HOME ".cargo\bin\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}

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
        [string]$Uv,
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$Backend
    )

    $package = if ($Backend -eq "cu126") { "opendde[gpu]" } else { "opendde" }
    Write-Host "Installing $package with Python $Python and the $Backend PyTorch backend..."
    & $Uv `
        "tool" "install" `
        "--force" `
        "--python" $Python `
        "--torch-backend" $Backend `
        $package | Out-Host
    return $LASTEXITCODE -eq 0
}

$uv = Find-Uv
if (-not $uv) {
    Write-Host "uv was not found; installing it with the official Astral installer..."
    $installer = Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1"
    Invoke-Expression $installer

    $uv = Find-Uv
    if (-not $uv) {
        throw "uv was installed, but its executable could not be located."
    }
}

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

$installed = Install-OpenDdeBackend $uv $PythonVersion $selectedBackend
if ($installed -and $selectedBackend -eq "cu126") {
    $toolRootOutput = & $uv "tool" "dir"
    $toolRoot = ($toolRootOutput | Out-String).Trim()
    $toolPython = Join-Path $toolRoot "opendde\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $toolPython -PathType Leaf)) {
        $installed = $false
    } else {
        & $toolPython "-c" `
            "import torch; assert torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith('12.6'); torch.zeros(1, device='cuda')"
        $installed = $LASTEXITCODE -eq 0
    }
}

if (-not $installed -and $selectedBackend -eq "cu126") {
    Write-Warning "CUDA installation or runtime verification failed; falling back to CPU."
    $selectedBackend = "cpu"
    $installed = Install-OpenDdeBackend $uv $PythonVersion "cpu"
}
if (-not $installed) {
    throw "Unable to install the OpenDDE $selectedBackend backend."
}
$TorchBackend = $selectedBackend

$toolBinOutput = & $uv "tool" "dir" "--bin"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine the uv tool executable directory."
}
$toolBin = ($toolBinOutput | Out-String).Trim()
$openDde = Join-Path $toolBin "opendde.exe"
if (-not (Test-Path -LiteralPath $openDde -PathType Leaf)) {
    $openDde = Join-Path $toolBin "opendde"
}
if (-not (Test-Path -LiteralPath $openDde -PathType Leaf)) {
    throw "uv completed, but the OpenDDE executable was not created in $toolBin."
}

# Update future shells and make the launcher available to the remainder of this script. Molchanica
# also discovers uv's tool bin directly, so desktop launches do not rely solely on shell profiles.
& $uv "tool" "update-shell"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "uv could not update the shell PATH; Molchanica can still locate OpenDDE."
}
$env:Path = "$toolBin;$env:Path"

Write-Host "`nVerifying the OpenDDE installation..."
Invoke-CheckedNative $openDde "--version"
Invoke-CheckedNative $openDde "doctor"

Write-Host "`nOpenDDE is installed in an isolated uv environment."
Write-Host "Executable: $openDde"
Write-Host "Restart Molchanica if it is currently running."
