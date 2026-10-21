"""
LigParGen Hugging Face Space.

Mirrors the core of https://zarbi.chem.yale.edu/ligpargen/: SMILES input
(typed or drawn via Ketcher), PDB/MOL file upload, optimization level,
charge model, and a results zip with every output format the CLI already
produces -- plus a 3D preview of the optimized geometry the original site
doesn't have.

BOSS is fetched into this container at startup from a private HF Dataset
repo (see docs/adr/0003 in the main repo) -- never committed here, never
baked into the image.

Also runs as an MCP server (Gradio's built-in mcp_server=True) alongside
the web UI, exposing run_ligpargen() as an MCP tool at the same host's
/gradio_api/mcp/sse -- so an agent can submit a molecule and get back
parameter files the same way a human using the form does. That function's
type hints and docstring are what an MCP client sees as the tool's
argument schema and description, so keep them accurate if its signature
changes.
"""
import collections
import glob
import json
import multiprocessing
import os
import random
import shutil
import signal
import string
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import gradio as gr
from gradio_molecule3d import Molecule3D
from huggingface_hub import CommitScheduler, snapshot_download
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D

from LigParGen.Converter import convert

MAX_ATOMS = 200
BOSS_DIR = "/home/user/boss"
BOSS_REPO = os.environ.get("BOSS_ASSET_REPO", "lsdodda/ligpargen-boss-assets")

# A single BOSS run blocking the (concurrency_limit=1) queue indefinitely on
# a pathological input would stall every other visitor's job behind it --
# see run_convert_with_timeout().
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "60"))

# Basic per-client submission cap -- see check_rate_limit(). In-memory only
# (resets on restart/redeploy): acceptable for a single-worker Space with no
# other persistent state, and avoids standing up external infra for v1.
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "5"))
RATE_LIMIT_WINDOW_S = int(os.environ.get("RATE_LIMIT_WINDOW_S", "600"))
_rate_limit_lock = threading.Lock()
_submission_log = collections.defaultdict(list)

# Anonymous usage metrics -- job counts/sizes/outcomes only, never the
# submitted structure itself or any request/session identifier -- pushed
# periodically to a private HF Dataset. See setup_usage_scheduler().
USAGE_REPO = os.environ.get("USAGE_METRICS_REPO", "lsdodda/ligpargen-usage-metrics")
USAGE_LOG_DIR = Path("/tmp/usage_logs")
USAGE_LOG_FILE = USAGE_LOG_DIR / "usage.jsonl"
_usage_scheduler = None


def fetch_boss():
    """Pull BOSS from the private asset store into BOSS_DIR and fix permissions.

    HF's storage does not preserve the executable bit (confirmed empirically
    while provisioning the asset store), so every file needs chmod +x after
    the fetch -- a plain read-only volume mount can't do this, which is why
    this is a snapshot_download to a writable directory instead.
    """
    if os.path.isfile(os.path.join(BOSS_DIR, "BOSS")) and os.access(
        os.path.join(BOSS_DIR, "BOSS"), os.X_OK
    ):
        return  # already fetched (e.g. hot-reload during dev)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN secret is not set on this Space -- required to fetch "
            "BOSS from the private asset store. Set it in Space Settings."
        )

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    try:
        snapshot_download(
            repo_id=BOSS_REPO,
            repo_type="dataset",
            token=token,
            local_dir=BOSS_DIR,
        )
    except Exception as exc:  # noqa: BLE001 -- surface any failure loudly at startup
        raise RuntimeError(
            f"Failed to fetch BOSS from {BOSS_REPO}: {exc}"
        ) from exc

    for root, _dirs, files in os.walk(BOSS_DIR):
        for name in files:
            path = os.path.join(root, name)
            os.chmod(path, os.stat(path).st_mode | 0o111)

    boss_bin = os.path.join(BOSS_DIR, "BOSS")
    if not os.access(boss_bin, os.X_OK):
        raise RuntimeError(f"{boss_bin} still isn't executable after fetch+chmod")


def count_heavy_and_h_atoms(smiles=None, file_path=None):
    """Best-effort atom count (heavy + explicit/implicit H) for the 200-atom guard."""
    mol = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    elif file_path:
        if file_path.lower().endswith(".pdb"):
            mol = Chem.MolFromPDBFile(file_path, sanitize=False)
        else:
            mol = Chem.MolFromMolFile(file_path, sanitize=False)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()


def build_preview_sdf(pdb_path, resname, template_smiles=None):
    """Build an SDF (bond orders intact) from the final optimized PDB, for
    the 3D viewer.

    PDB has no bond-order field, so feeding Molecule3D the raw output PDB
    rendered every bond as an undifferentiated single stick, no visible
    double/triple/aromatic bonds -- gradio_molecule3d only accepts
    pdb/sdf/mol2/pdb1 (checked directly against its bundled JS), not a bare
    .mol, so SDF (a MOL block plus an optional data section RDKit's
    SDWriter already produces) is the right target format here, not MOL.

    Reuses the same AssignBondOrdersFromTemplate technique as
    LigParGen.mol_boss.convert_pdb2mol_with_smiles when a trusted SMILES is
    available (maps the final geometry's connectivity through known-correct
    bond orders), falling back to RDKit's geometry-based rdDetermineBonds
    when it isn't (a plain PDB upload with no accompanying SMILES).

    Returns the SDF path, or None if nothing could be built -- callers
    should fall back to the raw PDB preview in that case rather than show
    nothing.
    """
    try:
        pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
        if pdb_mol is None:
            print("build_preview_sdf: MolFromPDBFile returned None for %r" % pdb_path)
            return None

        fixed = None
        if template_smiles:
            template = Chem.MolFromSmiles(template_smiles)
            if template is None:
                print("build_preview_sdf: MolFromSmiles(%r) returned None" % template_smiles)
            else:
                try:
                    fixed = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
                except ValueError as exc:
                    print("build_preview_sdf: AssignBondOrdersFromTemplate failed (%s) -- "
                          "falling back to geometry-based perception" % exc)
                    fixed = None

        if fixed is None:
            rw = Chem.RWMol()
            conf = Chem.Conformer(pdb_mol.GetNumAtoms())
            pdb_conf = pdb_mol.GetConformer()
            for i, atom in enumerate(pdb_mol.GetAtoms()):
                rw.AddAtom(Chem.Atom(atom.GetSymbol()))
                pos = pdb_conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))
            rw.AddConformer(conf, assignId=True)
            rdDetermineBonds.DetermineBonds(rw, charge=0, embedChiral=False)
            Chem.SanitizeMol(rw)
            fixed = rw

        sdf_path = os.path.join(os.path.dirname(pdb_path), "%s_preview.sdf" % resname)
        with Chem.SDWriter(sdf_path) as writer:
            writer.write(fixed)
        return sdf_path
    except Exception as exc:  # noqa: BLE001 -- never let a preview-only step break the job
        import traceback
        print("build_preview_sdf: failed for %r (resname=%s, template_smiles=%r): %s" % (
            pdb_path, resname, template_smiles, exc))
        traceback.print_exc()
        return None


def setup_usage_scheduler():
    """Set up anonymous usage-metrics logging (job counts/sizes/outcomes,
    never the submitted structure or any request/session identifier) via a
    CommitScheduler pushing to a private HF Dataset -- the same
    runtime-fetch/private-repo pattern already used for the BOSS asset
    store (docs/adr/0003). Returns None (and log_usage() becomes a no-op)
    if HF_TOKEN isn't set or the scheduler fails to start, so a
    misconfigured or rate-limited token disables logging rather than
    blocking the app from serving conversions at all -- usage tracking is
    a nice-to-have, not a hard dependency for the app's actual job."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set -- usage metrics logging disabled.")
        return None
    try:
        return CommitScheduler(
            repo_id=USAGE_REPO,
            repo_type="dataset",
            folder_path=USAGE_LOG_DIR,
            path_in_repo="logs",
            every=5,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 -- logging is best-effort, never block startup
        print("Failed to set up usage metrics scheduler: %s" % exc)
        return None


def log_usage(input_type, n_atoms, opt_iters, lbcc, success, duration_s, error_type=None):
    if _usage_scheduler is None:
        return
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_type": input_type,
        "n_atoms": n_atoms,
        "opt_iters": opt_iters,
        "lbcc": lbcc,
        "success": success,
        "error_type": error_type,
        "duration_s": round(duration_s, 2),
    }
    try:
        # CommitScheduler's own background thread reads this file on its
        # push interval -- .lock (a plain threading.Lock it exposes for
        # exactly this) keeps a partial line from ever being read mid-write.
        with _usage_scheduler.lock:
            with USAGE_LOG_FILE.open("a") as f:
                f.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 -- usage logging must never break a job
        print("log_usage: failed to write usage row: %s" % exc)


def check_rate_limit(client_ip):
    """True if this client is still under RATE_LIMIT_MAX submissions in the
    trailing RATE_LIMIT_WINDOW_S seconds (and records this attempt either
    way). In-memory sliding window, not persisted -- see RATE_LIMIT_MAX's
    module-level comment."""
    now = time.time()
    with _rate_limit_lock:
        recent = [t for t in _submission_log[client_ip] if now - t < RATE_LIMIT_WINDOW_S]
        recent.append(now)
        _submission_log[client_ip] = recent
        return len(recent) <= RATE_LIMIT_MAX


def _client_ip(request):
    """Best-effort real client IP behind HF Spaces' reverse proxy -- used
    only as a rate-limiting key, never logged or stored (see
    check_rate_limit()/log_usage())."""
    if request is None:
        return "unknown"
    try:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
    except Exception:  # noqa: BLE001 -- IP extraction is best-effort, never block a submission over it
        pass
    return "unknown"


def _classify_input(smiles_text, upload_path):
    if upload_path:
        if upload_path.lower().endswith(".pdb"):
            return "pdb+smiles" if smiles_text else "pdb"
        return "mol"
    return "smiles"


def _run_convert_in_child(kwargs, job_dir, result_queue):
    os.setsid()  # new process group -- lets the parent kill BOSS's whole
                 # subprocess tree on timeout, not just this one process.
    try:
        os.chdir(job_dir)
        convert(**kwargs)
        result_queue.put(("ok", None, None))
    except ValueError as exc:
        result_queue.put(("value_error", str(exc), None))
    except Exception as exc:  # noqa: BLE001 -- serialize any failure back to the parent
        result_queue.put(("error", str(exc), type(exc).__name__))


def run_convert_with_timeout(kwargs, job_dir, timeout_s):
    """Run LigParGen.Converter.convert() in a child process, killing its
    whole process group if it doesn't finish within timeout_s.

    convert() ultimately drives BOSS through blocking os.system() calls
    with no timeout of their own (LigParGen/BOSSReader.py's Get_OPT) -- a
    pathological input hanging BOSS would otherwise stall this Space's
    single-worker queue (concurrency_limit=1) for every other visitor
    indefinitely. Running convert() in its own process (started as its own
    session/process group via os.setsid() in the child) means a timeout can
    actually reclaim it: os.killpg() reaches BOSS's csh/xZCM1A/etc.
    descendants too, not just the immediate Python child, which a plain
    thread-based timeout could never do (Python threads can't be killed,
    and terminating just the child process would orphan BOSS's own
    subprocesses to keep running).
    """
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_run_convert_in_child, args=(kwargs, job_dir, result_queue), daemon=True)
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.join(5)
        raise gr.Error(
            "This molecule took longer than %d minutes to process and was stopped -- "
            "it may be too large or complex for this Space. Try a smaller molecule or "
            "fewer optimization iterations." % (timeout_s // 60)
        )

    if result_queue.empty():
        raise gr.Error(
            "LigParGen's process exited unexpectedly (exit code %s). Please try again "
            "or open an issue on GitHub." % proc.exitcode
        )

    status, msg, exc_type = result_queue.get()
    if status == "value_error":
        raise gr.Error(msg)
    if status == "error":
        raise gr.Error(
            "LigParGen failed while processing this molecule (%s). If this keeps "
            "happening, please open an issue on GitHub." % (exc_type or "unknown error")
        )


def run_ligpargen(
    smiles_text: str,
    upload_file: Optional[str],
    opt_iters: Literal["0", "1", "2", "3"],
    charge_model: Literal[
        "1.14*CM1A-LBCC (neutral molecules)",
        "1.14*CM1A (neutral or charged)",
    ],
    charge: Literal["0", "-1", "-2", "1", "2"],
    request: gr.Request,
    progress=gr.Progress(),
) -> tuple:
    """Generate OPLS-AA/CM1A force-field parameters for an organic ligand.

    Runs BOSS + LigParGen on a molecule and returns a zip containing
    parameter/topology files for OpenMM, GROMACS, CHARMM, LAMMPS, TINKER,
    CNS/X-PLOR, Q, DESMOND, and BOSS/MCPRO, plus a PQR file and a 3D
    preview of the optimized geometry. Molecules are limited to 200 atoms
    (heavy atoms plus hydrogens); a single job is stopped if it runs
    longer than about a minute, and submissions from the same client are
    rate-limited.

    Provide smiles_text (a typed SMILES string, e.g. "c1ccccc1" for
    benzene) or upload_file (a MOL file with all hydrogens already
    explicit, or a PDB file). A PDB upload_file should be paired with
    smiles_text, which is then used as a trusted template to fix the
    PDB's bond orders and add any missing hydrogens. charge_model
    "1.14*CM1A-LBCC (neutral molecules)" forces charge to "0"
    regardless of the charge argument; only "1.14*CM1A (neutral or
    charged)" honors a nonzero charge.

    Args:
        smiles_text: SMILES string for the molecule.
        upload_file: PDB or MOL file to convert, as a path or URL.
        opt_iters: BOSS optimization iterations, "0" to "3".
        charge_model: Partial-charge scheme -- see description above.
        charge: Net formal charge of the molecule.

    Returns:
        A 3-tuple: path to a .zip file with all output formats, path to
        an SDF or PDB file of the optimized geometry for 3D preview (or
        None if a preview could not be built), and a short human-readable
        status message.
    """
    start_time = time.perf_counter()
    smiles_text = (smiles_text or "").strip()
    upload_path = upload_file if upload_file is not None else None

    # Every submission attempt counts toward the rate limit, checked first --
    # cheapest possible gate, before spending any RDKit/BOSS work on it.
    if not check_rate_limit(_client_ip(request)):
        raise gr.Error(
            "Rate limit reached (%d submissions per %d minutes). Please wait "
            "before submitting again." % (RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_S // 60)
        )

    if not smiles_text and not upload_path:
        raise gr.Error("Provide a SMILES string (typed or drawn) or upload a PDB/MOL file.")
    if smiles_text and upload_path and not upload_path.lower().endswith(".pdb"):
        raise gr.Error(
            "SMILES + upload is only supported for PDB uploads (used to fix bond "
            "orders/missing Hs) -- submit either SMILES or a MOL file, not both."
        )

    input_type = _classify_input(smiles_text, upload_path)

    # Prefer the SMILES for the atom count when both are given: PDB uploads are
    # often missing hydrogens, which would undercount against MAX_ATOMS.
    n_atoms = count_heavy_and_h_atoms(smiles=smiles_text or None, file_path=None if smiles_text else upload_path)
    if n_atoms is not None and n_atoms > MAX_ATOMS:
        log_usage(input_type, n_atoms, int(opt_iters), None, success=False,
                  duration_s=time.perf_counter() - start_time, error_type="TooManyAtoms")
        raise gr.Error(f"Molecule has {n_atoms} atoms; maximum allowed is {MAX_ATOMS}.")

    lbcc = charge_model == "1.14*CM1A-LBCC (neutral molecules)"
    resolved_charge = 0 if lbcc else int(charge)

    # PDB's resName field is a strict 3-character fixed-width column (spec cols
    # 18-20) -- LigParGen's own PDB/GRO/etc. writers use fixed-width formatting
    # too, so anything longer silently overflows into the chainID/resSeq/coordinate
    # columns downstream. Confirmed live: an 8-char resname ("LPG"+5 hex chars)
    # shifted every coordinate column and produced a non-numeric resSeq, which
    # RDKit's own PDB parser (MolFromPDBFile) outright rejected and PyMOL misread.
    # 3 random letters -- matches the CLI's own convention ("-r, should be a 3
    # LETTER WORD") and gives 26**3 = 17,576 combinations, plenty for this app's
    # single-worker, serialized-queue usage.
    resname = "".join(random.choices(string.ascii_uppercase, k=3))
    job_dir = tempfile.mkdtemp(prefix="ligpargen_")

    # Trusted SMILES for the 3D preview's bond-order fix (build_preview_sdf) --
    # separate from kwargs["smiles"], which only LigParGen.Converter.convert()
    # itself uses (and only for the PDB-upload case, to fix the actual BOSS
    # input). Typed/drawn SMILES and PDB+SMILES both already have one; a plain
    # MOL upload already carries correct bond orders of its own, so derive an
    # equivalent SMILES from it too, rather than leaving the preview to the
    # geometry-only fallback when a perfectly good source of truth exists.
    preview_template_smiles = smiles_text or None

    kwargs = dict(opt=int(opt_iters), charge=resolved_charge, lbcc=lbcc, resname=resname)
    if upload_path:
        staged = os.path.join(job_dir, os.path.basename(upload_path))
        shutil.copyfile(upload_path, staged)
        if upload_path.lower().endswith(".pdb"):
            kwargs["pdb"] = os.path.basename(staged)
            # A SMILES supplied alongside the PDB is used as a trusted template
            # to fix connectivity/bond orders/missing Hs -- see
            # LigParGen.mol_boss.convert_pdb2mol_with_smiles.
            if smiles_text:
                kwargs["smiles"] = smiles_text
        else:
            kwargs["mol"] = os.path.basename(staged)
            if preview_template_smiles is None:
                mol_for_smiles = Chem.MolFromMolFile(staged, sanitize=False)
                if mol_for_smiles is not None:
                    try:
                        Chem.SanitizeMol(mol_for_smiles)
                        preview_template_smiles = Chem.MolToSmiles(mol_for_smiles)
                    except Exception:  # noqa: BLE001 -- preview-only; geometry fallback still applies
                        pass
    else:
        kwargs["smiles"] = smiles_text

    progress(0.1, desc="Running BOSS + LigParGen...")
    try:
        # Runs convert() in its own killable process group -- see
        # run_convert_with_timeout()'s docstring for why a plain in-process
        # call (the prior behavior) can't be safely bounded by a timeout.
        run_convert_with_timeout(kwargs, job_dir, JOB_TIMEOUT_S)

        zip_path = os.path.join(job_dir, f"{resname}.zip")
        if not os.path.isfile(zip_path):
            raise gr.Error("LigParGen did not produce output -- check the molecule is valid.")

        preview_pdb_candidates = glob.glob(f"/tmp/{resname}.pdb")
        preview_pdb = preview_pdb_candidates[0] if preview_pdb_candidates else None

        preview_file = preview_pdb
        if preview_pdb:
            preview_sdf = build_preview_sdf(preview_pdb, resname, template_smiles=preview_template_smiles)
            if preview_sdf:
                preview_file = preview_sdf
    except gr.Error as exc:
        log_usage(input_type, n_atoms, int(opt_iters), lbcc, success=False,
                  duration_s=time.perf_counter() - start_time, error_type=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 -- never let an unanticipated error surface raw internals
        log_usage(input_type, n_atoms, int(opt_iters), lbcc, success=False,
                  duration_s=time.perf_counter() - start_time, error_type=type(exc).__name__)
        raise gr.Error("An unexpected error occurred. Please try again, or open an issue on GitHub if it persists.")

    log_usage(input_type, n_atoms, int(opt_iters), lbcc, success=True,
              duration_s=time.perf_counter() - start_time)
    progress(1.0, desc="Done")
    return zip_path, preview_file, f"Done -- {resname}"


KETCHER_HTML = """
<div id="loading" style="display:flex;justify-content:center;align-items:center;height:420px">
<p style="color:#8a8272;font-size:0.9rem;font-family:'IBM Plex Mono',ui-monospace,monospace">loading structure editor&hellip;</p>
</div>
<div id="root" style="height:420px;border:1px solid #e7e1d3;border-radius:10px;overflow:hidden"></div>
<button id="ketcher-use-btn" type="button"
        style="width:100%;margin-top:10px;padding:10px 16px;border-radius:8px;border:1px solid #4f46e5;
               background:#4f46e5;color:#fff;font-family:inherit;font-weight:600;font-size:0.92rem;
               cursor:pointer;transition:background 0.15s ease,border-color 0.15s ease">
  Use drawn structure
</button>
"""

# Ketcher 2.7.2, pinned per docs/research/ketcher-and-molecule3d-integration.md --
# newer versions have a known getSmiles() regression on some structures.
#
# The "Use drawn structure" button lives inside this HTML block as a plain
# <button>, wired here via direct DOM manipulation rather than Gradio's
# js=/fn=None event-return mechanism -- that mechanism did not reliably fire
# on real user clicks in this Gradio version (reproduced live: getSmiles()
# worked when called from the console, but neither a real click nor a
# programmatic .click() on a Gradio-wired button ever invoked it, per
# instrumentation that counted calls to a wrapped getSmiles()). Writing
# straight to the target textbox's <textarea> and dispatching a native
# "input" event is the standard, robust way to feed a Gradio-bound
# component from custom JS.
KETCHER_LOAD_JS = """
async () => {
  let url = "https://huggingface.co/datasets/simonduerr/ketcher-2.7.2/raw/main/static/css/main.6a646761.css";
  fetch(url).then(r => r.text()).then(text => {
    const style = document.createElement('style');
    style.textContent = text;
    document.head.appendChild(style);
  });
  url = "https://huggingface.co/datasets/simonduerr/ketcher-2.7.2/resolve/main/static/js/main.5445f351.js";
  fetch(url).then(r => r.text()).then(text => {
    const script = document.createElement('script');
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
    document.getElementById('loading').style.display = 'none';
  });

  document.getElementById('ketcher-use-btn').addEventListener('click', async () => {
    const smi = await ketcher.getSmiles();
    const textarea = document.querySelector('#smiles_box textarea');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(textarea, smi);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
"""

# Gradio defaults to the visitor's OS/browser color-scheme preference, which
# put this page in dark mode for anyone with dark mode on. That broke Ketcher
# specifically: its toolbar SVG icons use `fill: currentColor` with no color
# of their own, so they inherited Gradio's near-white dark-mode body text
# color and rendered almost invisible against Ketcher's own light toolbar
# chrome (confirmed live: computed icon color rgb(244,244,245) on a white
# toolbar). Force light mode via Gradio's built-in `__theme` query param --
# simpler and more robust than chasing every inherited color it broke.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Public Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "SFMono-Regular", "monospace"],
).set(
    body_background_fill="*neutral_50",
    background_fill_primary="white",
    block_background_fill="white",
    block_border_color="*neutral_200",
    block_label_text_color="*neutral_500",
    block_title_text_color="*neutral_800",
    body_text_color="*neutral_800",
    body_text_color_subdued="*neutral_500",
    border_color_primary="*neutral_200",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="*neutral_50",
    button_secondary_border_color="*neutral_300",
    button_secondary_text_color="*neutral_800",
    input_background_fill="white",
    input_border_color="*neutral_300",
    input_border_color_focus="*primary_500",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,500;0,600;1,500&display=swap');

#app-header .app-kicker {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--body-text-color-subdued);
  margin: 0 0 0.35rem 0;
}
#app-header h1 {
  font-family: 'Newsreader', ui-serif, Georgia, serif;
  font-style: italic;
  font-weight: 500;
  font-size: 2.4rem;
  letter-spacing: -0.01em;
  margin: 0;
}
.app-subtitle { max-width: 46rem; }
.app-subtitle p {
  color: var(--body-text-color-subdued);
  font-size: 1rem;
  line-height: 1.6;
}
.about-panel {
  max-width: 46rem;
  margin-top: 0.7rem;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-primary);
  overflow: hidden;
}
.about-panel summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.7rem 1rem;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--body-text-color-subdued);
  user-select: none;
}
.about-panel summary::-webkit-details-marker { display: none; }
.about-panel summary:hover { color: var(--body-text-color); }
.about-panel .about-chevron {
  font-family: ui-sans-serif, sans-serif;
  font-size: 0.85rem;
  transition: transform 0.15s ease;
}
.about-panel[open] .about-chevron { transform: rotate(180deg); }
.about-panel .about-body {
  margin: 0;
  padding: 0 1.1rem 0.9rem 1.1rem;
  border-top: 1px solid var(--border-color-primary);
  padding-top: 0.85rem;
  color: var(--body-text-color-subdued);
  font-size: 0.88rem;
  line-height: 1.65;
}
.about-panel .about-body a {
  color: var(--body-text-color-subdued);
  text-decoration: underline;
  text-decoration-color: var(--border-color-primary);
}
.about-panel .about-body a:hover { color: var(--primary-600); text-decoration-color: var(--primary-600); }
.section-label p {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 1.05rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--body-text-color-subdued);
  border-bottom: 1px solid var(--border-color-primary);
  padding-bottom: 0.6rem;
  margin-bottom: 0 !important;
}
.panel {
  border-radius: var(--radius-lg) !important;
  padding: 1.1rem !important;
}
#smiles_box textarea, #status_box textarea {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}
.citations ol {
  padding-left: 1.2rem;
  margin: 0;
}
.citations li {
  color: var(--body-text-color-subdued);
  font-size: 0.85rem;
  line-height: 1.55;
  margin-bottom: 0.85rem;
}
.citations li:last-child { margin-bottom: 0; }
.citations li strong { color: var(--body-text-color); font-weight: 500; }
.citations a {
  color: var(--body-text-color-subdued);
  text-decoration: underline;
  text-decoration-color: var(--border-color-primary);
}
.citations a:hover { color: var(--primary-600); text-decoration-color: var(--primary-600); }

/* Belt-and-braces alongside FORCE_LIGHT_JS above: pin Ketcher's own DOM to
   light regardless of the page's color scheme, since its bundle assumes a
   light host and inherited color is otherwise how it broke in the first
   place. */
#ketcher-panel, #ketcher-panel * { color: #1a1a1a; }
#ketcher-panel { color-scheme: light; }
#ketcher-panel #ketcher-use-btn:hover { background: #4338ca !important; border-color: #4338ca !important; }
"""


# Same "about" text as the original Yale webserver, with the CLI/issues
# links pointed at this repo instead of the original site's own pages.
# Collapsed by default (a <details> disclosure, not gr.Markdown) so this
# secondary text doesn't compete with the hero header or push Step 1 below
# the fold -- expand on demand instead.
ABOUT_HTML = """
<details class="about-panel">
  <summary>
    <span>About LigParGen</span>
    <span class="about-chevron">&#9662;</span>
  </summary>
  <p class="about-body">
    LigParGen is a web-based service that provides force field (FF)
    parameters for organic molecules or ligands, offered by the Jorgensen
    group. LigParGen provides bond, angle, dihedral, and Lennard-Jones
    OPLS-AA parameters with 1.14*CM1A or 1.14*CM1A-LBCC partial atomic
    charges. Server provides parameter and topology files for commonly
    used molecular dynamics and Monte Carlo packages OpenMM, Gromacs,
    NAMD, CHARMM, LAMMPS, TINKER, CNS/X-PLOR, Q, DESMOND, BOSS and MCPRO.
    Also, the PQR file is generated. Supported input formats: SMILES, MOL
    and PDB. Maximum ligand size allowed is 200 atoms. Check
    <a href="https://github.com/leelasd/LigParGen_2.3" target="_blank" rel="noopener">this link</a>
    to use LigParGen software from command-line in your local computer.
    Please, report any issue on the
    <a href="https://github.com/leelasd/LigParGen_2.3/issues" target="_blank" rel="noopener">LigParGen issues</a>
    page.
  </p>
</details>
"""


# Same citations as the CLI's own --help text (LigParGen/Converter.py) plus
# the core OPLS-AA potential paper, reproduced here for the web UI in place
# of the original webserver's References section.
REFERENCES_MD = """
1. Dodda, L. S.; Cabeza de Vaca, I.; Tirado-Rives, J.; Jorgensen, W. L.
   **LigParGen web server: an automatic OPLS-AA parameter generator for
   organic ligands.** *Nucleic Acids Res.* **2017**, *45* (W1), W331-W336.
   [doi:10.1093/nar/gkx312](https://doi.org/10.1093/nar/gkx312)
2. Dodda, L. S.; Vilseck, J. Z.; Tirado-Rives, J.; Jorgensen, W. L.
   **1.14\\*CM1A-LBCC: Localized Bond-Charge Corrected CM1A Charges for
   Condensed-Phase Simulations.** *J. Phys. Chem. B* **2017**, *121* (15),
   3864-3870. [doi:10.1021/acs.jpcb.7b00272](https://doi.org/10.1021/acs.jpcb.7b00272)
3. Udier-Blagovic, M.; Morales De Tirado, P.; Pearlman, S. A.; Jorgensen, W. L.
   **Accuracy of free energies of hydration using CM1 and CM3 atomic
   charges.** *J. Comput. Chem.* **2004**, *25*, 1322-1332.
   [doi:10.1002/jcc.20059](https://doi.org/10.1002/jcc.20059)
4. Jorgensen, W. L.; Tirado-Rives, J.
   **Potential energy functions for atomic-level simulations of water and
   organic and biomolecular systems.** *Proc. Natl. Acad. Sci. USA* **2005**,
   *102*, 6665-6670.
   [doi:10.1073/pnas.0408037102](https://doi.org/10.1073/pnas.0408037102)
"""


def build_ui():
    with gr.Blocks(title="LigParGen", theme=THEME, css=CSS) as demo:
        gr.Markdown('<p class="app-kicker">OPLS-AA / CM1A &middot; BOSS</p>\n\n# LigParGen', elem_id="app-header")
        gr.Markdown(
            "OPLS-AA/CM1A force-field parameter generator for organic ligands.",
            elem_classes="app-subtitle",
        )
        gr.HTML(ABOUT_HTML)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Step 1 &mdash; Input structure", elem_classes="section-label")
                with gr.Group(elem_classes="panel"):
                    smiles_box = gr.Textbox(
                        label="SMILES", placeholder="Enter SMILES, e.g. c1ccccc1", elem_id="smiles_box"
                    )
                    gr.Button("Sample: Benzene", size="sm").click(
                        lambda: "c1ccccc1", inputs=None, outputs=smiles_box, api_name=False
                    )

                    gr.Markdown("**Or draw a structure:**")
                    ketcher_html = gr.HTML(KETCHER_HTML, elem_id="ketcher-panel")

                    gr.Markdown(
                        "**Or upload a MOL/PDB file:** MOL files must include all "
                        "hydrogens. For a PDB upload, also supply the SMILES above "
                        "-- it's used to fix bond orders and add any missing "
                        "hydrogens (PDB files don't encode bond order and are "
                        "often missing Hs)."
                    )
                    upload = gr.File(label="MOL or PDB file", file_types=[".mol", ".pdb"])

                gr.Markdown("Step 2 &mdash; Options", elem_classes="section-label")
                with gr.Group(elem_classes="panel"):
                    opt_iters = gr.Dropdown(["0", "1", "2", "3"], value="0", label="Molecule optimization iterations")
                    charge_model = gr.Radio(
                        ["1.14*CM1A-LBCC (neutral molecules)", "1.14*CM1A (neutral or charged)"],
                        value="1.14*CM1A-LBCC (neutral molecules)",
                        label="Charge model",
                    )
                    charge = gr.Dropdown(["0", "-1", "-2", "1", "2"], value="0", label="Molecule charge")

                    submit = gr.Button("Submit Molecule", variant="primary")

            with gr.Column():
                gr.Markdown("Results", elem_classes="section-label")
                with gr.Group(elem_classes="panel"):
                    status = gr.Textbox(label="Status", interactive=False, elem_id="status_box")
                    output_zip = gr.File(label="Download all output formats (.zip)")
                    preview = Molecule3D(label="3D preview (optimized geometry)", reps=[{"style": "stick"}])

                gr.Markdown("References", elem_classes="section-label")
                gr.Markdown(REFERENCES_MD, elem_classes="citations")

        submit.click(
            run_ligpargen,
            inputs=[smiles_box, upload, opt_iters, charge_model, charge],
            outputs=[output_zip, preview, status],
            api_name="run_ligpargen",
        )

        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_LIGHT_JS)
        demo.load(fn=None, inputs=None, outputs=None, js=KETCHER_LOAD_JS)

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    fetch_boss()
    os.environ["BOSSdir"] = BOSS_DIR
    _usage_scheduler = setup_usage_scheduler()
    build_ui().launch(server_name="0.0.0.0", server_port=7860, mcp_server=True)
