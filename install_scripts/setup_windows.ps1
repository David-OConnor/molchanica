[CmdletBinding()]
param(
    [string]$SourceDirectory,
    [string]$InstallDirectory,
    [switch]$InstallOpenDde,
    [switch]$SkipOpenDde
)

# This file installs the application for the current user, and creates a Start menu entry.
#
# We install to %LOCALAPPDATA%\Programs, rather than Program Files, so that no administrator
# rights are required, and so the application can write its preferences file and its cache of
# molecules downloaded from the web; it stores both alongside its executable, and Program Files
# is not writable by a normal process.
#
# Run it from the folder you extracted the release into, e.g. by double-clicking
# setup_windows.bat, or with:
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1

$ErrorActionPreference = "Stop"

$NAME_UPPER = "Molchanica"
$NAME = "molchanica"
$EXE = "$NAME.exe"
$GEMMI_DIR = "gemmi"
$CUFFT_LIB = "cufft64_12.dll"
$OPENDDE_SCRIPT = "install_opendde.ps1"
$TOOL_SCRIPT = "install_tool.ps1"
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
        # The application resolves the colocated `gemmi` folder relative to the working directory.
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
# entry launches a copy that cannot save preferences. We only point it out; removing it requires
# administrator rights, so we leave that to the user.
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

if ($InstallOpenDde -and $SkipOpenDde) {
    throw "InstallOpenDde and SkipOpenDde are mutually exclusive."
}

# We default this here rather than in the param block: Windows PowerShell 5.1 leaves $PSScriptRoot
# empty while binding parameters when a script is run with -File, as setup_windows.bat does. It is
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

$installOpenDdeNow = $false
if ($InstallOpenDde) {
    $installOpenDdeNow = $true
} elseif (-not $SkipOpenDde) {
    $ans = Read-Host "Install OpenDDE to a Python virtual environment, to support structure prediction? Warning: Multi-Gb. [y/n]"
    $installOpenDdeNow = $ans -match "^[Yy]"
}

if ($installOpenDdeNow) {
    $openDdeScript = Join-Path $SourceDirectory $OPENDDE_SCRIPT
    if (Test-Path -LiteralPath $openDdeScript -PathType Leaf) {
        & $openDdeScript
        Write-Host "`nOpenDDE installed into a dedicated Python virtual environment."
    } else {
        Write-Warning "$OPENDDE_SCRIPT was not found in $SourceDirectory; skipping the OpenDDE install."
    }
}

# The antibody tools are offered separately because they are a different proposition: tens of
# megabytes rather than gigabytes, and useful the moment a structure is open.
if (-not $SkipOpenDde) {
    $toolScript = Join-Path $SourceDirectory $TOOL_SCRIPT
    if (Test-Path -LiteralPath $toolScript -PathType Leaf) {
        $ans = Read-Host "Install the antibody tools (IgBLAST and ANARCII)? These are comparatively small. [y/n]"
        if ($ans -match "^[Yy]") {
            & $toolScript igblast anarcii
        }
        Write-Host ""
        Write-Host "Other optional tools (boltz2, ligandmpnn, proteinmpnn) can be installed at any time with"
        Write-Host "$TOOL_SCRIPT. Run it with --list to see them all, or 'all' for everything."
        Write-Host "Molchanica's `"Tools`" panel shows which are installed and working."
    }
}

Write-Host ""
Write-Host "$NAME_UPPER is installed in $InstallDirectory."
Write-Host "Its preferences file and downloaded-molecule cache are kept there too, and survive a re-run of this script."
Write-Host "You can launch it from the Start menu (e.g., search `"$NAME_UPPER`"), and/or pin it to the taskbar."

Show-PreviousInstallNotice
