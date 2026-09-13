"""
For evaluating tool statuses.
"""

from __future__ import annotations

import logging

import bio_tools
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main.tools import Process

from .environments import process_executables_root


class CheckResult(Enum):
    """
    Outcome of probing whether one tool can actually be reached and run.

    I believe this must be kept in sync with `bio_tools`.
    """

    PASS = "Pass"
    NOT_INSTALLED = "Can't find"
    ERROR = "Error"
    # Nothing was probed. On a split deployment the web node cannot see the
    # compute node's tools while that machine is stopped, and starting a GPU
    # instance to fill in a table is not worth what it costs; this says so
    # instead of reporting them absent, which is what probing here would find.
    NOT_CHECKED = "Not checked"

    @property
    def rank(self) -> int:
        """Sort order for the status table, since the labels are alphabetical
        by accident. Working tools first, then installed-but-broken, then
        absent, then the ones nothing is known about: an ERROR is something to
        go read, a CANT_FIND is something to go install, and a NOT_CHECKED is
        only something to come back for.
        """

        return {
            self.PASS: 0,
            self.ERROR: 1,
            self.NOT_INSTALLED: 2,
            self.NOT_CHECKED: 3,
        }[self]


@dataclass
class ToolStatus:
    """One tool's status, as reported by a bio_tools Tool status probe."""

    result: CheckResult
    detail: str = ""
    # "CPU" or "GPU", for tools whose runtime can use either; None where a
    # device distinction doesn't apply, or couldn't be determined.
    device: str | None = None


def probe_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
) -> tuple[int | None, str]:
    """Run a non-raising probe through bio_tools' Rust command runner."""

    try:
        completed = bio_tools.CommandSpec(
            command,
            cwd=cwd,
            timeout=timeout,
            check=False,
            output_limit=4_000,
        ).run()
    except bio_tools.RunError as exc:
        return None, str(exc)
    output = (completed.stdout + completed.stderr).strip()[-4_000:]
    return completed.return_code, output


def _is_crash(output: str) -> bool:
    """Whether `output` looks like an interpreter crash rather than real CLI output.

    A CLI that dies before reaching argparse (e.g. an import error in a broken
    virtualenv) still exits nonzero with nonempty output, which the loose
    "any output counts as PASS" rule below would otherwise take as a passing
    usage banner. Checked before that rule so a torn install is reported as
    ERROR with the traceback, not PASS with it.
    """

    return output.lstrip().startswith("Traceback (most recent call last):")


def probe_cli(
    command: list[str], *, cwd: Path | None = None, timeout: int = 120
) -> ToolStatus:
    """Best-effort ToolStatus for a CLI whose exact version flag isn't known.

    Tries --version, then --help: between the two, most scientific CLIs answer
    at least one, and some answer neither with a zero exit status while still
    printing a usage banner. Only "the process could not be started at all"
    counts as CANT_FIND; any non-crash output at all counts as PASS, since the
    point is confirming the binary runs, not parsing its particular version
    scheme.

    The generous default timeout matters here: several of these CLIs (biophi,
    boltzgen) import torch/pandas/sqlalchemy-heavy stacks before they can even
    reach argparse, which alone can take 45-60s on a cold start -- longer than
    this used to allow, which made an installed, working tool misreport as
    CANT_FIND.
    """

    last_detail = "Produced no output for --version or --help."
    for flag in ("--version", "--help"):
        code, output = probe_command([*command, flag], cwd=cwd, timeout=timeout)
        if code is None:
            return ToolStatus(CheckResult.NOT_INSTALLED, output)
        if output and not _is_crash(output):
            return ToolStatus(CheckResult.PASS, output.splitlines()[0][:200])
        if output:
            last_detail = output.splitlines()[-1][:200]
        else:
            last_detail = (
                f"`{shlex.join([*command, flag])}` exited {code} with no output."
            )
    return ToolStatus(CheckResult.ERROR, last_detail)


def probe_python_package(
    python: str | Path,
    distribution: str,
    module: str,
    *,
    timeout: int = 120,
) -> ToolStatus:
    """Import one package in its own interpreter and report its distribution version.

    This covers Python applications that expose a library API instead of a
    meaningful local CLI, including hosted-service clients such as Boltz ADME.
    It still runs out of process, so incompatible scientific stacks never enter
    the Django interpreter.
    """

    code, output = probe_command(
        [
            str(python),
            "-c",
            (
                "import importlib, importlib.metadata as m, sys; "
                "importlib.import_module(sys.argv[2]); "
                "print(sys.argv[1] + ' ' + m.version(sys.argv[1]))"
            ),
            distribution,
            module,
        ],
        timeout=timeout,
    )
    if code is None:
        return ToolStatus(CheckResult.NOT_INSTALLED, output)
    if code != 0:
        detail = output.splitlines()[-1] if output else f"Could not import {module}."
        return ToolStatus(CheckResult.ERROR, detail[:200])
    return ToolStatus(CheckResult.PASS, output.splitlines()[-1][:200])


def require_gpu_status(
    status: ToolStatus, device: str | None, runtime: str
) -> ToolStatus:
    """Turn an otherwise passing check into an error when CUDA is unavailable."""

    if status.result != CheckResult.PASS:
        return status
    if device == "GPU":
        status.device = device
        return status
    # None and "CPU" are different findings and must not share a message: a
    # probe that could not be run or timed out has established nothing about
    # the hardware, whereas "CPU" is the runtime's own answer.
    detail = (
        f"{runtime} is installed, but its GPU probe did not complete, so its "
        "CUDA support could not be determined."
        if device is None
        else f"{runtime} is installed, but it does not report a usable CUDA GPU."
    )
    return ToolStatus(CheckResult.ERROR, detail, device=device)


def _check_one(tool: Process, *, full: bool) -> tuple[Process, ToolStatus]:
    """One tool's check, timed, with any unexpected failure turned into ERROR."""

    started = time.perf_counter()

    try:
        installable = bio_tools.Tool(tool.spec.slug)
        probe = installable.status_full if full else installable.status_quick
        native = probe(process_executables_root())
        status = ToolStatus(CheckResult(native.result), native.detail, native.device)
    except Exception as exc:
        status = ToolStatus(CheckResult.ERROR, f"bio_tools status() raised: {exc}")

    elapsed = time.perf_counter() - started
    level = "Full status" if full else "Quick status"
    logging.warning(f"{level} ({tool.name}, {elapsed:.1f}s)")
    return tool, status


def check_statuses(
    tools: list[Process] | None = None,
    *,
    full: bool = False,
) -> list[tuple[Process, ToolStatus]]:
    """
    Validate for each service if it's installed and able to be launched.

    `tools` defaults to the whole registry. A web node in a split
    deployment passes the subset it actually installs, since probing for the
    rest would only report the compute node's tools as missing.

    Quick checks are the default: they inspect installation files without
    importing application code. ``full=True`` launches each tool's help/version
    probe and checks its compute runtime. Unexpected binding or probe failures
    are converted to CheckResult.ERROR rather than breaking the whole page.

    The checks run one thread apiece. For full checks, nearly all of the wall
    time is a thread blocked in the Rust command runner while another
    interpreter imports torch or jax. The GIL is released throughout and the
    adapters share no mutable state, so the total becomes roughly the slowest
    single check instead of the sum of all of them.
    """

    # Deferred: main.tools imports this module, so it cannot be imported
    # at module level here.
    from main.tools import all_tools

    if tools is None:
        tools = all_tools()
    if not tools:
        return []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(len(tools), 1)) as pool:
        # map(), not as_completed(): the page lists tools in registry order.
        statuses = list(pool.map(partial(_check_one, full=full), tools))

    level = "full" if full else "quick"
    logging.warning(
        f"Checked {len(statuses)} {level} tool statuses in "
        f"{time.perf_counter() - started:.1f}s"
    )
    return statuses
