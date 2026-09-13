"""Web registry backed by the shared bio_tools adapters."""
import os
import sys
from pathlib import Path
import bio_tools

os.environ.setdefault("BIO_WEB_EXECUTABLE_ROOT", str(Path(__file__).resolve().parents[2] / "process_executables"))
_adapter_path = bio_tools.adapter_package_path()
if _adapter_path not in sys.path:
    sys.path.insert(0, _adapter_path)
import bio_tool_adapters as _shared
globals().update({name: value for name, value in vars(_shared).items() if not name.startswith("__")})

def all_tools() -> list[Process]:
    """The single registry of tools; urls.py and views.py both derive from it.

    Name, categories, launch type, license type, expense, and top-choice
    status all live centrally in bio_tools' catalog (keyed by each module's
    `SPEC.slug`); this list supplies only what is genuinely this app's own:
    a stable id for its own database records, and the adapter module itself.
    """

    return [
        catalog_process(1, rdkit),
        # todo:
        # Cannot become fully healthy unattended. The code is open, but model parameters must come
        # directly from Google and the databases require roughly 630 GB unpacked. Official
        # installation guide: https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md
        # catalog_process(2, alphafold3),
        catalog_process(3, opendde),
        catalog_process(4, boltz2),
        catalog_process(5, chai1),
        catalog_process(6, protenix),
        catalog_process(7, esmfold2),
        catalog_process(8, immunebuilder),
        catalog_process(9, highfold),
        # todo: Requires a registered/licensed dataset, unpacked under process_executables/pdbbind
        #  or configured with PDBBIND_ROOT.
        # catalog_process(10, pdbbind),
        catalog_process(11, boltzgen),
        catalog_process(12, bindcraft),
        catalog_process(13, gromacs),
        # todo: Avoiding ORCA now due to its license, and long run times.
        # catalog_process(14, orca),
        catalog_process(15, igblast),
        catalog_process(16, biophi),
        catalog_process(17, antifold),
        catalog_process(18, abpmnn),
        catalog_process(19, proteinmpnn),
        catalog_process(20, ligandmpnn),
        catalog_process(21, proteinmpnn_ddg),
        catalog_process(41, rfdiffusion3),
        catalog_process(23, rfantibody),
        catalog_process(24, germinal),
        catalog_process(25, mber),
        catalog_process(26, igdesign),
        catalog_process(27, thermompnn),
        catalog_process(28, boltz_adme),
        catalog_process(29, genie3),
        catalog_process(30, deepsp),
        catalog_process(31, deepimmuno),
        catalog_process(32, tlimmuno),
        catalog_process(33, netsolp),
        catalog_process(34, deepstabp),
        catalog_process(35, aggrescan3d),
        catalog_process(36, dlkcat),
        catalog_process(37, catpred),
        catalog_process(38, antibody_annotator),
        # TODO: Manual: requires the licensed SAbPred distribution, then TAP_PYTHON and TAP_RUNNER.
        # catalog_process(39, tap),
        catalog_process(40, placer),
    ]


def tool_by_id(tool_id: int) -> Process | None:
    """The registry entry a stored `Process.id` refers to, if it is still here.

    None for a tool that has been taken out of the registry since something
    saved its id. Callers decide what to show for one; the id is never reused,
    so a stale reference stays stale rather than becoming a different tool.
    """

    return next((t for t in all_tools() if t.id == tool_id), None)


def tool_choices() -> list[tuple[int, str]]:
    """`Process.id` to display name, for the `choices` of a model field.

    Given to the field as this function rather than as its result, so the
    registry is read when a form is rendered instead of when models.py is
    imported: adding a tool needs no migration, and the migration that created
    the column refers to this function instead of freezing a copy of the list.

    Alphabetical, because it is read as a list of names -- `all_tools()`
    order is grouping for the catalog pages, which is not much help in a picker
    of forty.
    """

    return sorted(
        ((tool.id, tool.name) for tool in all_tools()),
        key=lambda choice: choice[1].lower(),
    )


from . import (  # noqa: E402
    abpmnn,
    aggrescan3d,
    alphafold3,
    antibody_annotator,
    antifold,
    bindcraft,
    biophi,
    boltz2,
    boltz_adme,
    boltzgen,
    catpred,
    chai1,
    deepimmuno,
    deepsp,
    deepstabp,
    dlkcat,
    esmfold2,
    genie3,
    germinal,
    gromacs,
    highfold,
    igblast,
    igdesign,
    immunebuilder,
    ligandmpnn,
    mber,
    netsolp,
    opendde,
    orca,
    pdbbind,
    placer,
    protenix,
    proteinmpnn,
    proteinmpnn_ddg,
    rdkit,
    rfantibody,
    rfdiffusion3,
    tap,
    thermompnn,
    tlimmuno,
)
