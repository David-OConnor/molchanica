param(
    # Skip Authenticode signing. Unsigned builds are blocked by Smart App Control, and warned on by
    # SmartScreen, so only use this for local test packages.
    [switch]$Unsigned
)

$ErrorActionPreference = "Stop"
$appName = "molchanica"

. (Join-Path $PSScriptRoot "win_exe_signing.ps1")

# The version is defined in one place only: Cargo.toml's `[package] version`. Read it from there,
# so a release never ships under a version that disagrees with the one the app reports.
$versionMatch = Select-String -Path "Cargo.toml" -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionMatch) { throw "Could not read the version from Cargo.toml" }
$appVersion = $versionMatch.Matches[0].Groups[1].Value
# Archive names use underscores, e.g. 0.3.7 -> 0_3_7.
$version = $appVersion -replace '\.', '_'
$outDir = "target\release"
$exe    = Join-Path $outDir "${appName}.exe"
$cufft = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cufft64_12.dll"
# The installer's files are listed in this script, not here.
$iss = Join-Path $PSScriptRoot "install_scripts\${appName}.iss"

function Find-Iscc {
    $found = Get-ChildItem "${env:ProgramFiles(x86)}\Inno Setup *\ISCC.exe", "$env:ProgramFiles\Inno Setup *\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup *\ISCC.exe" -ErrorAction SilentlyContinue |
        Sort-Object { $_.Directory.Name } -Descending |
        Select-Object -First 1
    if (-not $found) {
        throw "ISCC.exe was not found. Install Inno Setup: winget install -e --id JRSoftware.InnoSetup"
    }

    $found.FullName
}

cargo build --release --features adme,synthesis

# The CUDA build is also what people without an Nvidia GPU download, so it must not name any CUDA
# DLL in its import table: Windows would refuse to start it there, before the program ever gets a
# chance to fall back to the CPU. cudarc's `dynamic-loading` feature and the `LoadLibrary` in
# ewald's cufft.cu keep these out; a stray `cargo:rustc-link-lib` would put them back, so fail the
# release here rather than in a bug report.
#
# The same reasoning applies to the Visual C++ runtime, which fails the same way: on a machine
# that has never had the redistributable installed, a dynamically linked CRT means a "System
# error" dialog naming VCRUNTIME140.dll, raised before `main` runs and so impossible for the
# program to report on itself. `.cargo/config.toml` links the CRT statically to keep it out of the
# import table, and an overriding RUSTFLAGS in the environment is enough to undo that silently.
$dumpbin = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dumpbin) {
    $dependents = & $dumpbin.FullName /dependents $exe

    $cudaImports = $dependents | Select-String -Pattern "nvcuda|cufft|nvrtc|cudart"
    if ($cudaImports) {
        throw "$exe imports a CUDA DLL at load time, so it will not start without CUDA installed. Offending entries: $($cudaImports -join ', ')"
    }

    # api-ms-win-crt-* resolves to ucrtbase.dll, which is a part of Windows 10 and 11 and so would
    # load fine on its own. We fail on it anyway: it is the clearest signal that the static CRT has
    # stopped being applied, and it shows up whether or not VCRUNTIME140 does.
    $crtImports = $dependents | Select-String -Pattern "VCRUNTIME|MSVCP|api-ms-win-crt"
    if ($crtImports) {
        throw "$exe imports the Visual C++ runtime at load time, so it will not start without VC_redist installed. Check that the +crt-static rustflag in .cargo/config.toml is still being applied. Offending entries: $($crtImports -join ', ')"
    }
} else {
    Write-Warning "dumpbin was not found, so the release was not checked for load-time CUDA or VC++ runtime imports."
}

# Windows ships as a signed installer rather than a zip: batch files can't be signed, so the old
# setup_molchanica.bat always drew a security warning. cuFFT is already signed by Nvidia.
Invoke-Sign -Paths $exe -Unsigned:$Unsigned

$iscc = Find-Iscc
$isccArgs = "/Q", "/DAppVersion=$appVersion", "/O$PSScriptRoot"
if (-not $Unsigned) {
    # Inno signs the setup and its uninstaller through this. $q and $f are its quote and quoted-file
    # tokens.
    $powershell = (Get-Process -Id $PID).Path
    $signScript = Join-Path $PSScriptRoot "win_exe_signing.ps1"
    $isccArgs += "/DSign", "/Smolsign=`$q$powershell`$q -NoProfile -ExecutionPolicy Bypass -File `$q$signScript`$q `$f"
}

& $iscc @isccArgs "/F${appName}_${version}_win_setup" "/DCufft=$cufft" $iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE." }

# We don't use a second binary for non-CUDA; the Cuda-compiled binary should work
# on non-CUDA-available setups. The only difference in the packages is inclusion of hte cuFFT library.
# cargo build --release --no-default-features --features adme,synthesis
& $iscc @isccArgs "/F${appName}_${version}_win_nocuda_setup" $iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE." }

# Build and package the Linux release too, in the Ubuntu 22.04 WSL instance. 22.04 specifically,
# since it sets the glibc version the Linux binary requires; building on a newer Ubuntu would
# produce a binary that refuses to start on older distros.
$wslDistro = "Ubuntu-22.04"
# `wsl -l` writes UTF-16, which PowerShell reads with embedded nulls; strip them before comparing.
$wslDistros = (wsl.exe -l -q) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ }
if ($wslDistros -notcontains $wslDistro) {
    throw "The WSL instance '$wslDistro' was not found, so the Linux release was not built. It must be Ubuntu 22.04. Installed instances: $($wslDistros -join ', ')"
}

# `--exec` skips the distro's shell, so os-release is read as-is and parsed here.
$wslOsRelease = wsl.exe -d $wslDistro --exec cat /etc/os-release
$wslVersion = $wslOsRelease | Select-String -Pattern '^VERSION_ID="?([^"]+)"?' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or $wslVersion -ne "22.04") {
    throw "The WSL instance '$wslDistro' is not Ubuntu 22.04 (VERSION_ID: '$wslVersion'), so the Linux release was not built."
}

# A login shell, so ~/.profile puts cargo on the PATH.
wsl.exe -d $wslDistro --cd $PSScriptRoot -- bash -l ./make_release.sh
if ($LASTEXITCODE -ne 0) {
    throw "make_release.sh failed in WSL ($wslDistro) with exit code $LASTEXITCODE."
}
