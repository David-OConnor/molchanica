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
$dumpbin = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dumpbin) {
    $cudaImports = & $dumpbin.FullName /dependents $exe | Select-String -Pattern "nvcuda|cufft|nvrtc|cudart"
    if ($cudaImports) {
        throw "$exe imports a CUDA DLL at load time, so it will not start without CUDA installed. Offending entries: $($cudaImports -join ', ')"
    }
} else {
    Write-Warning "dumpbin was not found, so the release was not checked for load-time CUDA imports."
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
