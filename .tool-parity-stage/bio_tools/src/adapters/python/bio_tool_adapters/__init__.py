"""Shared helpers for the third-party scientific tool adapters.

Each module in this package exposes a SPEC (a `Spec` instance) used by the web
UI and a run(payload) function used by the JSON API. Every adapter runs its
tool: there is no dry-run mode, so a request either produces real output or
fails with one of the errors below.

Executables are always launched by absolute path. Standalone binaries resolve
through `executable()`, from PROCESS_EXECUTABLES (the gitignored directory
setup_system.sh populates) before falling back to PATH. Tools that need their
own Python resolve through `tool_script()` or `tool_python()` instead, which
look only inside that tool's uv environment; see environments.py.
"""

from __future__ import annotations

import logging
import os
import shutil
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable

import bio_tools
from bio_tools import (
    LaunchType,
    Process,
    catalog_process,
    catalog_spec,
)

from .environments import (
    environment_python,
    environment_script,
    process_executables_root,
)
from .status_check import probe_command

logger = logging.getLogger(__name__)

AMINO_ACIDS = frozenset("ABCDEFGHIKLMNPQRSTVWXYZ")
DNA_BASES = frozenset("ACGTUNRYKMSWBDHV")

# Tools that are not installed system-wide live here: setup_system.sh unpacks
# each distribution into its own subdirectory, and this tree is gitignored
# because the contents are large third-party binaries and datasets.
PROCESS_EXECUTABLES = process_executables_root()
RUN_LOGS = PROCESS_EXECUTABLES / "run_logs"
_RUN_LOG_NAME: ContextVar[str | None] = ContextVar("tool_run_log_name", default=None)

# How long any one tool invocation may take. This is a guard against a wedged
# process, not a schedule: every tool here is a fold, a dock or a design run
# whose cost is set by the model and the input, and plenty of them legitimately
# run for hours. A cap exists at all because submissions wait in a serialized
# lane, where a process that never exits would hold every later job behind it;
# it is this long because COMPUTE_JOB_TIMEOUT is what a job forwarded to the
# compute node already gets, and a local run should not be cut off sooner. An
# adapter overrides it only when the caller named a budget of their own.
TOOL_RUN_TIMEOUT = 86_400


class ToolInputError(ValueError):
    """The client supplied invalid or incomplete input."""


class ToolExecutionError(RuntimeError):
    """A configured third-party tool exited unsuccessfully."""


class ToolUnavailable(RuntimeError):
    """The requested third-party dependency is not available on this host."""


# What the exceptions above become on the wire, checked in order since
# ToolUnavailable and ToolExecutionError both descend from RuntimeError.
# Beside the exceptions rather than in main.views because two callers need it:
# the API, which returns the code, and main.job_records, which stores it on the
# run's row. main.api_docs documents these straight from this table.
ERROR_CODES: tuple[tuple[type[Exception], str], ...] = (
    (ToolInputError, "invalid_input"),
    (ToolUnavailable, "tool_unavailable"),
    (ToolExecutionError, "execution_failed"),
)


def job_error(tool_name: str, error: Exception) -> dict[str, str]:
    """The code and message one adapter failure is reported as."""

    for exc_type, code in ERROR_CODES:
        if isinstance(error, exc_type):
            return {"code": code, "message": str(error)}
    logger.exception("Unexpected %s tool failure", tool_name, exc_info=error)
    return {"code": "internal_error", "message": "The tool failed unexpectedly."}


def run_tool(tool: Process, payload: dict[str, Any]) -> Any:
    """Run an adapter with its slug available to shared command logging."""

    token = _RUN_LOG_NAME.set(tool.spec.slug)
    try:
        return tool.module.run(preset_payload(tool.spec.slug, payload))
    finally:
        _RUN_LOG_NAME.reset(token)


def preset_payload(slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Expand a catalog preset for API clients; explicit field values win."""

    preset_id = payload.get("preset")
    if not preset_id:
        return payload
    try:
        return bio_tools.catalog_preset(
            slug,
            str(preset_id),
            overrides={key: value for key, value in payload.items() if key != "preset"},
        )
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc


# Imported below the exceptions and alphabets it validates against, and above
# the adapters that need all of it: this is the one order in which the package
# can import its own field helpers without a partially initialized module.
from .field_processing import Field, Option, text  # noqa: E402


def tool_fields(
    slug: str, *, dynamic_options: dict[str, list[tuple[str, str]]] | None = None
) -> list[Field]:
    """Build this app's form objects from bio_tools' shared tool contract."""

    return bio_tools.catalog_fields(
        slug,
        field_type=Field,
        option_type=Option,
        dynamic_options=dynamic_options,
    )


def tool_tasks(slug: str) -> list[Option]:
    """Build this app's task selectors from bio_tools' shared tool contract."""

    return bio_tools.catalog_tasks(slug, option_type=Option)


def _bundled_candidates(name: str) -> Iterable[Path]:
    """Paths under PROCESS_EXECUTABLES where setup_system.sh may have put `name`."""

    # Self-contained distributions (IgBLAST, ORCA) are unpacked whole because
    # they resolve sibling data directories relative to their own location, so
    # the binary sits one or two levels below the bundle root.
    for filename in (name, f"{name}.exe") if os.name == "nt" else (name,):
        yield PROCESS_EXECUTABLES / filename
        yield PROCESS_EXECUTABLES / "bin" / filename
        yield from sorted(PROCESS_EXECUTABLES.glob(f"*/{filename}"))
        yield from sorted(PROCESS_EXECUTABLES.glob(f"*/bin/{filename}"))


def executable(env_name: str, *names: str) -> str:
    """Resolve a tool to an absolute path.

    The `env_name` override wins, then the bundled process_executables/ tree,
    then PATH for tools installed globally. The result is always absolute: a
    request never launches a process by relying on the server's PATH lookup.
    """

    configured = os.getenv(env_name)
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return str(Path(resolved).resolve())
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return str(configured_path.resolve())
        raise ToolUnavailable(
            f"{env_name} points to an executable that cannot be found."
        )

    for name in names:
        for candidate in _bundled_candidates(name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return str(Path(resolved).resolve())

    friendly = ", ".join(names)
    raise ToolUnavailable(
        f"No executable found for {friendly}. Install it under {PROCESS_EXECUTABLES}, "
        f"set {env_name} to its full path, or add it to PATH."
    )


def _override(env_name: str) -> str | None:
    """An operator-supplied absolute path for a tool, if one is configured."""

    configured = os.getenv(env_name)
    if not configured:
        return None
    resolved = shutil.which(configured)
    if resolved:
        return str(Path(resolved).resolve())
    path = Path(configured).expanduser()
    if path.is_file():
        return str(path.resolve())
    raise ToolUnavailable(f"{env_name} points to a file that cannot be found.")


def _setup_hint(environment: str) -> str:
    return (
        f"Run `python install_tools.py {environment}` "
        "(setup_system.sh does this through bio_tools)"
    )


def tool_script(environment: str, script: str, env_name: str) -> str:
    """Resolve a console script inside one PythonBasedApp's uv environment.

    Deliberately narrower than `executable()`: a tool that owns an environment
    is launched from that environment and nowhere else, so a same-named command
    on PATH -- including one belonging to a different tool's environment --
    can never be picked up by accident. Only `env_name` overrides it.
    """

    override = _override(env_name)
    if override:
        return override

    path = environment_script(environment, script)
    if path.is_file():
        return str(path.resolve())
    raise ToolUnavailable(
        f"{script} is not installed in the {environment} environment. "
        f"{_setup_hint(environment)}, or set {env_name} to its full path."
    )


def tool_python(environment: str, env_name: str) -> str:
    """Resolve the interpreter one PythonBasedApp's scripts must run under.

    For tools whose code is a checkout rather than a package: the checkout is
    named separately, and this supplies the interpreter it imports from. Never
    falls back to `sys.executable`, which would run third-party model code
    inside the server's own environment.
    """

    override = _override(env_name)
    if override:
        return override

    path = environment_python(environment)
    if path.is_file():
        # Not .resolve(): uv venvs put a symlink to a shared base interpreter
        # at bin/python, and that base interpreter has no pyvenv.cfg of its
        # own. Invoking it by its resolved target path (instead of through
        # the symlink) makes CPython's site init treat it as a bare
        # interpreter with none of this venv's installed packages on
        # sys.path -- an `import torch` that works through the symlink then
        # fails with ModuleNotFoundError through the resolved path.
        return str(path)
    raise ToolUnavailable(
        f"No Python environment for {environment}. {_setup_hint(environment)}, "
        f"or set {env_name} to an interpreter that can run it."
    )


def configured_python(env_name: str, tool: str) -> str:
    """The interpreter for a tool this project does not install itself.

    Conda-managed tools have no uv environment to resolve, but they must still
    name their interpreter explicitly rather than inheriting the server's.
    """

    override = _override(env_name)
    if override:
        return override
    raise ToolUnavailable(
        f"Set {env_name} to the Python interpreter of the {tool} Conda "
        f"environment (for example `conda run -n {tool} which python`)."
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = TOOL_RUN_TIMEOUT,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    artifacts: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Execute and durably audit a shell-free command through bio_tools."""

    try:
        completed = bio_tools.CommandSpec(
            command,
            cwd=cwd,
            timeout=timeout,
            stdin=stdin,
            env=env,
            check=True,
            output_limit=100_000,
            run_log_dir=RUN_LOGS,
            run_name=_RUN_LOG_NAME.get() or Path(command[0]).stem,
            artifacts=list(artifacts) if artifacts is not None else [cwd],
        ).run()
    except bio_tools.RunError as exc:
        raise ToolExecutionError(str(exc)) from exc

    return {
        "command": completed.command,
        "return_code": completed.return_code,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "run_log_dir": str(completed.run_log_dir),
    }


def readable_files(root: Path, *, maximum_files: int = 30) -> list[str]:
    """List generated relative paths without returning potentially huge model artifacts."""

    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
            if len(files) >= maximum_files:
                break
    return files


def _device_probe(
    python: str | Path, snippet: str, *, timeout: int = 300
) -> str | None:
    """Run a tiny probe under `python` and return "GPU"/"CPU", or None if unclear.

    The snippet itself is trivial, but importing torch or jax is not: it is a
    few hundred megabytes of shared objects, and `check_statuses` runs every
    adapter's probe at once, so a dozen interpreters page their CUDA runtimes
    in simultaneously. Measured on this project's own environments that costs
    35-45s each even with a warm page cache -- the timeout has to clear that
    by a wide margin, because a probe that times out is indistinguishable here
    from one that answered "CPU", and the tool is then reported as having no
    GPU when it has a perfectly good one.
    """

    code, output = probe_command([str(python), "-c", snippet], timeout=timeout)
    if code != 0:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    device = lines[-1] if lines else ""
    return device if device in {"GPU", "CPU"} else None


def torch_device(python: str | Path) -> str | None:
    """Whether `python`'s PyTorch build reports a usable GPU, without importing torch here."""

    return _device_probe(
        python, "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')"
    )


def jax_device(python: str | Path) -> str | None:
    """Whether `python`'s JAX build reports a usable GPU, without importing jax here."""

    return _device_probe(
        python,
        "import jax; print('GPU' if any(d.platform == 'gpu' for d in jax.devices()) else 'CPU')",
    )


# Imported last: each adapter module pulls the helpers above out of this
# package, so they must already be defined by the time these run.
