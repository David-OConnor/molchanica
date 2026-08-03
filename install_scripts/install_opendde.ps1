[CmdletBinding()]
param(
    [string]$PythonExecutable = $env:OPENDDE_PYTHON,
    [string]$VenvDirectory = $env:OPENDDE_VENV_DIR,
    [string]$TorchBackend = $env:OPENDDE_TORCH_BACKEND
)

# Install OpenDDE. This is a shim: the real work is in install_tool.ps1, which installs every
# optional tool from one place.
#
# It is kept because the README, setup_windows.ps1, and existing user notes all name this script,
# and because its parameters are documented. They are mapped onto install_tool.ps1's environment
# variables below, so anything that worked before still works.

$ErrorActionPreference = "Stop"

if ($PythonExecutable) { $env:MOLCHANICA_PYTHON = $PythonExecutable }
if ($TorchBackend) { $env:MOLCHANICA_TORCH_BACKEND = $TorchBackend }
# VenvDirectory needs no mapping beyond restoring the variable install_tool.ps1 reads directly, so
# that the location Molchanica's registry also honours stays the single source of truth.
if ($VenvDirectory) { $env:OPENDDE_VENV_DIR = $VenvDirectory }

& (Join-Path $PSScriptRoot "install_tool.ps1") opendde
exit $LASTEXITCODE
