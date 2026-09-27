$ErrorActionPreference = "Stop"

# The version is defined in one place only: Cargo.toml's `[package] version`. Read it from there,
# so a release never ships under a version that disagrees with the one the app reports.
$versionMatch = Select-String -Path "Cargo.toml" -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionMatch) { throw "Could not read the version from Cargo.toml" }
# Archive names use underscores, e.g. 0.3.7 -> 0_3_7.
$version = $versionMatch.Matches[0].Groups[1].Value -replace '\.', '_'
$outDir = "target\release"
$exe    = Join-Path $outDir "molchanica.exe"
$readme = "README.md"
$setup = "install_scripts/setup_molchanica.ps1"
$setupLauncher = "install_scripts/setup_molchanica.bat"
$cufft = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cufft64_12.dll"
# Needed by the bio_tools ProteinMPNN recipe for native ddG scanning. Its absence is handled (the
# conversion is skipped), but shipping it means one less reason to need the repository.
$mpnnConvert = "scripts/convert_mpnn_weights.py"
$gemmi  = "C:\Program Files\gemmi"

cargo build --release

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

$zip1 = "molchanica_${version}_win.zip"
if (Test-Path $zip1) { Remove-Item $zip1 -Force }
Compress-Archive -LiteralPath $exe, $gemmi, $readme, $setup, $setupLauncher, $mpnnConvert, $cufft -DestinationPath $zip1 -Force

# We don't use a second binary for non-CUDA; the Cuda-compiled binary should work
# on non-CUDA-available setups. The only difference in the packages is inclusion of hte cuFFT library.
# cargo build --release --no-default-features
$zip2 = "molchanica_${version}_win_nocuda.zip"
if (Test-Path $zip2) { Remove-Item $zip2 -Force }
Compress-Archive -LiteralPath $exe, $gemmi, $readme, $setup, $setupLauncher, $mpnnConvert -DestinationPath $zip2 -Force

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
