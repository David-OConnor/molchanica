[CmdletBinding()]
param(
    [string]$SourceDirectory,
    [string]$InstallDirectory
#     [switch]$InstallOpenDde,
#     [switch]$SkipOpenDde
)

# This file installs the application for the current user, and creates a Start menu entry.
#
# We install to %LOCALAPPDATA%\Programs, rather than Program Files, so that no administrator
# rights are required, and so the application can write everything it keeps into its own install
# folder: the preferences file, the cache of molecules downloaded from the web, the graphics
# pipeline cache, and the optional third-party tools installed from the Tools panel. Molchanica
# resolves all of those relative to its own executable, so this one directory is the whole
# install. Program Files is not writable by a normal process, which is why it isn't used.
#
# Run it from the folder you extracted the release into, e.g. by double-clicking
# setup_molchanica.bat, or with:
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_molchanica.ps1

$ErrorActionPreference = "Stop"

$NAME_UPPER = "Molchanica"
$NAME = "molchanica"
$EXE = "$NAME.exe"
$GEMMI_DIR = "gemmi"
$CUFFT_LIB = "cufft64_12.dll"
$MPNN_CONVERTER = "convert_mpnn_weights.py"
$DESCRIPTION = "Molecule and protein viewer"

function Copy-Payload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $exeSource = Join-Path $Source $EXE
    if (-not (Test-Path -LiteralPath $exeSource -PathType Leaf)) {
        throw "$EXE was not found in $Source. Run this script from the folder you extracted the release into."
    }

    if (Get-Process -Name $NAME -ErrorAction SilentlyContinue) {
        throw "$NAME_UPPER is currently running, so its files can't be replaced. Close it, then run this script again."
    }

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Copy-Item -LiteralPath $exeSource -Destination $Destination -Force
    Write-Host "Copied $EXE to $Destination."

    # The Rust bio_tools installer uses this optional adapter to convert ProteinMPNN's checkpoint
    # for Molchanica's native ΔΔG scanner.
    $converterSource = Join-Path $Source $MPNN_CONVERTER
    if (-not (Test-Path -LiteralPath $converterSource -PathType Leaf)) {
        $converterSource = Join-Path $Source "scripts\$MPNN_CONVERTER"
    }
    if (Test-Path -LiteralPath $converterSource -PathType Leaf) {
        Copy-Item -LiteralPath $converterSource -Destination $Destination -Force
        Write-Host "Copied $MPNN_CONVERTER to $Destination."
    } else {
        Write-Warning "$MPNN_CONVERTER was not found; native ProteinMPNN ΔΔG conversion will be skipped."
    }

    # The application looks for Gemmi in a folder colocated with the executable, before falling
    # back to the system path. (See `file_io::gemmi_path`.)
    $gemmiSource = Join-Path $Source $GEMMI_DIR
    if (Test-Path -LiteralPath $gemmiSource -PathType Container) {
        $gemmiDest = Join-Path $Destination $GEMMI_DIR
        if (Test-Path -LiteralPath $gemmiDest) {
            Remove-Item -LiteralPath $gemmiDest -Recurse -Force
        }
        Copy-Item -LiteralPath $gemmiSource -Destination $gemmiDest -Recurse -Force
        Write-Host "Copied the $GEMMI_DIR folder to $Destination."
    } else {
        Write-Warning "No $GEMMI_DIR folder was found in $Source; opening MTZ and 2fo-fc files will require Gemmi on the system path."
    }

    # If the cuda FFT lib is packaged with the download, keep it next to the executable, so Windows
    # resolves it without the CUDA toolkit being on the Path.
    $cufftSource = Join-Path $Source $CUFFT_LIB
    if (Test-Path -LiteralPath $cufftSource -PathType Leaf) {
        Copy-Item -LiteralPath $cufftSource -Destination $Destination -Force
        Write-Host "Copied the $CUFFT_LIB library (for the cuFFT dependency) to $Destination."
    }
}

function New-StartMenuEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $programs = [Environment]::GetFolderPath("Programs")
    if (-not $programs) {
        throw "The Start menu folder for this user could not be located."
    }

    $shortcutPath = Join-Path $programs "$NAME_UPPER.lnk"
    $target = Join-Path $Destination $EXE

    $shell = New-Object -ComObject WScript.Shell
    try {
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $target
        # Molchanica locates its data next to its executable, not via the working directory, so
        # this is only so that file dialogs and any relative paths start somewhere sensible.
        $shortcut.WorkingDirectory = $Destination
        # The executable embeds its own icon; see build.rs.
        $shortcut.IconLocation = "$target,0"
        $shortcut.Description = $DESCRIPTION
        $shortcut.Save()
    } finally {
        [Runtime.InteropServices.Marshal]::ReleaseComObject($shell) | Out-Null
    }

    Write-Host "Created the Start menu entry at $shortcutPath."
}

# An all-users install from an earlier version of this script is no longer used, and its Start menu
# entry launches a copy that cannot write to its own folder, so it would fall back to scattering
# data under %LOCALAPPDATA%. We only point it out; removing it requires administrator rights, so we
# leave that to the user.
function Show-PreviousInstallNotice {
    $stale = @()

    if ($env:ProgramFiles) {
        $legacyDir = Join-Path $env:ProgramFiles $NAME_UPPER
        if ((Test-Path -LiteralPath $legacyDir -PathType Container) -and
            $legacyDir -ne $InstallDirectory) {
            $stale += @{ Path = $legacyDir; Recurse = $true }
        }
    }

    $commonPrograms = [Environment]::GetFolderPath("CommonPrograms")
    if ($commonPrograms) {
        $legacyShortcut = Join-Path $commonPrograms "$NAME_UPPER.lnk"
        if (Test-Path -LiteralPath $legacyShortcut -PathType Leaf) {
            $stale += @{ Path = $legacyShortcut; Recurse = $false }
        }
    }

    if (-not $stale) { return }

    Write-Host ""
    Write-Warning "An earlier all-users install is still present, and its Start menu entry points at a copy that cannot save preferences."
    Write-Host "To remove it, run these from an administrator PowerShell:"
    foreach ($item in $stale) {
        $recurse = if ($item.Recurse) { " -Recurse" } else { "" }
        Write-Host "  Remove-Item -LiteralPath `"$($item.Path)`"$recurse -Force"
    }
}

# if ($InstallOpenDde -and $SkipOpenDde) {
#     throw "InstallOpenDde and SkipOpenDde are mutually exclusive."
# }

# We default this here rather than in the param block: Windows PowerShell 5.1 leaves $PSScriptRoot
# empty while binding parameters when a script is run with -File, as setup_molchanica.bat does. It is
# populated by the time the body runs, in every host.
if (-not $SourceDirectory) {
    $SourceDirectory = $PSScriptRoot
}
if (-not $SourceDirectory) {
    $SourceDirectory = (Get-Location).Path
}
if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
    throw "The source directory $SourceDirectory does not exist."
}

if (-not $InstallDirectory) {
    if (-not $env:LOCALAPPDATA) {
        throw "LOCALAPPDATA is not set; provide InstallDirectory explicitly."
    }
    $InstallDirectory = Join-Path $env:LOCALAPPDATA "Programs\$NAME_UPPER"
}

Copy-Payload -Source $SourceDirectory -Destination $InstallDirectory
New-StartMenuEntry -Destination $InstallDirectory
# Optional third-party tools are installed in-process from Molchanica's Tools panel.

Write-Host ""
Write-Host "$NAME_UPPER is installed in $InstallDirectory."
Write-Host "Everything it writes stays in that one folder, and survives a re-run of this script:"
Write-Host "  molchanica_prefs.mca    preferences and per-molecule settings"
Write-Host "  managed_molecules\      molecules downloaded or generated in the app"
Write-Host "  gpu_cache\              the graphics pipeline cache"
Write-Host "  process_executables\    optional third-party tools installed from the Tools panel"
Write-Host "Set MOLCHANICA_DATA_DIR to keep them somewhere else, e.g. on another drive."
Write-Host "You can launch it from the Start menu (e.g., search `"$NAME_UPPER`"), and/or pin it to the taskbar."

Show-PreviousInstallNotice
