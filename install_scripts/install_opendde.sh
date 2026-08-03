#!/usr/bin/env sh

# Install OpenDDE. This is a shim: the real work is in install_tool.sh, which installs every
# optional tool from one place.
#
# It is kept because the README, setup_linux.sh, and existing user notes all name this script, and
# because its own environment variables are documented. Those are mapped onto install_tool.sh's
# below, so anything that worked before still works.
#
# Optional overrides (all still honoured):
#   OPENDDE_PYTHON=/path/to/python
#   OPENDDE_VENV_DIR=/path/to/opendde-venv
#   OPENDDE_TORCH_BACKEND=auto|cpu|cu126

set -eu

[ -n "${OPENDDE_PYTHON:-}" ] && export MOLCHANICA_PYTHON="$OPENDDE_PYTHON"
[ -n "${OPENDDE_TORCH_BACKEND:-}" ] && export MOLCHANICA_TORCH_BACKEND="$OPENDDE_TORCH_BACKEND"
# OPENDDE_VENV_DIR needs no mapping: install_tool.sh reads it directly, so that the location
# Molchanica's registry also honours stays the single source of truth.

exec "$(dirname "$0")/install_tool.sh" opendde
