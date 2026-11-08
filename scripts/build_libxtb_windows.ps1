param(
    [Parameter(Mandatory = $true)][string]$Gfortran,
    [Parameter(Mandatory = $true)][string]$RuntimeDirectory,
    [string]$XtbRoot = (Join-Path $PSScriptRoot '../process_executables/xtb'),
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$compiler = (Resolve-Path -LiteralPath $Gfortran).Path
$runtime = (Resolve-Path -LiteralPath $RuntimeDirectory).Path
$xtb = (Resolve-Path -LiteralPath $XtbRoot).Path

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $xtb 'libxtb/Library/bin'
}

$dependencies = @(
    'libgcc_s_seh-1.dll', 'libgfortran-5.dll', 'libgomp-1.dll',
    'libquadmath-0.dll', 'libwinpthread-1.dll',
    'libblas.dll', 'liblapack.dll', 'openblas.dll', 'vcruntime140.dll'
)
$archives = @((Join-Path $xtb 'lib/libxtb.a'), (Join-Path $xtb 'lib/libcpx.a'))

foreach ($path in ($archives + @($dependencies | ForEach-Object { Join-Path $runtime $_ }))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing build input: $path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path
$stagedLibrary = Join-Path $output "libxtb-$([guid]::NewGuid()).tmp.dll"
$arguments = @(
    '-shared', '-o', $stagedLibrary,
    '-Wl,--export-all-symbols,--whole-archive', $archives[0],
    '-Wl,--no-whole-archive', $archives[1],
    "-L$runtime", '-l:libblas.dll', '-l:liblapack.dll', '-fopenmp'
)

& $compiler @arguments
if ($LASTEXITCODE -ne 0) {
    throw "libxtb linking failed ($LASTEXITCODE)"
}

foreach ($name in $dependencies) {
    $source = Join-Path $runtime $name
    $destination = Join-Path $output $name
    if ([string]::Equals($source, $destination, [StringComparison]::OrdinalIgnoreCase)) {
        continue
    }

    $staged = Join-Path $output "$name-$([guid]::NewGuid()).tmp"
    Copy-Item -LiteralPath $source -Destination $staged
    Move-Item -LiteralPath $staged -Destination $destination -Force
}

Move-Item -LiteralPath $stagedLibrary -Destination (Join-Path $output 'libxtb.dll') -Force
Write-Output "Built libxtb at $output"
