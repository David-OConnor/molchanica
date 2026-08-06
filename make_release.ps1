$ErrorActionPreference = "Stop"

$version = "0_3_6"
$outDir = "target\release"
$exe    = Join-Path $outDir "molchanica.exe"
$readme = "README.md"
$setup = "install_scripts/setup_windows.ps1"
$setupLauncher = "install_scripts/setup_windows.bat"
$cufft = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cufft64_12.dll"
# Needed by the bio_tools ProteinMPNN recipe for native ddG scanning. Its absence is handled (the
# conversion is skipped), but shipping it means one less reason to need the repository.
$mpnnConvert = "scripts/convert_mpnn_weights.py"
$gemmi  = "C:\Program Files\gemmi"

cargo build --release
$zip1 = "molchanica_${version}_win.zip"
if (Test-Path $zip1) { Remove-Item $zip1 -Force }
Compress-Archive -LiteralPath $exe, $gemmi, $readme, $setup, $setupLauncher, $mpnnConvert, $cufft -DestinationPath $zip1 -Force

cargo build --release --no-default-features
$zip2 = "molchanica_${version}_win_nocuda.zip"
if (Test-Path $zip2) { Remove-Item $zip2 -Force }
Compress-Archive -LiteralPath $exe, $gemmi, $readme, $setup, $setupLauncher, $mpnnConvert -DestinationPath $zip2 -Force
