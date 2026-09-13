"""Paths into the per-tool uv environments that `LaunchType.PythonBasedApp` tools run in.

Every such tool gets an environment of its own, because they disagree about both
their dependencies and Python itself: Boltz requires Python <3.13 and numpy<2,
OpenDDE pins numpy 2.4.1, BoltzGen pins numpy 2.0.2 (which has no cp313 wheel),
and ESMFold2 requires Python 3.12 with its own PyTorch stack. No two can share an environment,
and none of them can share the server's.

Building them belongs to bio_tools, which `install_tools.py` and the Install
button on /statuses drive. It owns each tool's interpreter version, package
pins, PyTorch wheel index, and post-install probes; this module only resolves
paths into what it produced, so an adapter can find its interpreter without
loading any installer machinery.

Nothing here reads or writes the project's pyproject.toml, uv.lock, or .venv,
so installing, upgrading, or rebuilding a tool cannot disturb the interpreter
Django runs under.
"""

from __future__ import annotations

import os
from pathlib import Path


def process_executables_root() -> Path:
    """The gitignored tree setup_system.sh unpacks third-party tools into."""

    configured = os.getenv("BIO_TOOLS_EXECUTABLE_ROOT") or os.getenv("BIO_WEB_EXECUTABLE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "process_executables"


def environment_root() -> Path:
    """Where the per-tool environments live.

    One level below process_executables/ so that `executable()`'s bundle globs,
    which look at `*/bin/<name>`, cannot reach into an environment and hand a
    tool an interpreter-scoped script by accident.
    """

    return process_executables_root() / "python_envs"


BIO_TOOLS_ENVIRONMENT_NAMES = {
    "boltz": "boltz2",
    "esmfold": "esmfold2",
    "abmpnn": "proteinmpnn",
    "proteinmpnn_ddg": "proteinmpnn-ddg",
    "boltz_adme": "boltz-adme",
    "tlimmuno": "tlimmuno2",
    "antibody_annotator": "anarcii",
}


def environment_path(name: str) -> Path:
    """Resolve legacy adapter names to bio_tools' stable environment slugs."""

    return environment_root() / BIO_TOOLS_ENVIRONMENT_NAMES.get(name, name)


def _binary_directory(name: str) -> Path:
    return environment_path(name) / ("Scripts" if os.name == "nt" else "bin")


def environment_python(name: str) -> Path:
    """The interpreter for one tool's environment, whether or not it exists."""

    return _binary_directory(name) / ("python.exe" if os.name == "nt" else "python")


def environment_script(name: str, script: str) -> Path:
    """A console script inside one tool's environment, whether or not it exists."""

    directory = _binary_directory(name)
    if os.name != "nt":
        return directory / script
    # uv writes a .exe launcher, but a package can install a .cmd shim instead.
    for suffix in (".exe", ".cmd", ".bat"):
        candidate = directory / f"{script}{suffix}"
        if candidate.is_file():
            return candidate
    return directory / f"{script}.exe"
