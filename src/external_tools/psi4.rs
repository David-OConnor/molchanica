use std::collections::HashMap;

use super::tool_form::FormContract;

const CUBE_FIELDS: &str = r#"[
    {"name":"cube_density","label":"Electron and spin density","kind":"checkbox","default":false,"group":"Cube files","input_modes":"parameters"},
    {"name":"cube_esp","label":"Electrostatic potential","kind":"checkbox","default":false,"help":"Also writes the matching total electron density cube.","group":"Cube files","input_modes":"parameters"},
    {"name":"cube_frontier","label":"Frontier orbitals","kind":"checkbox","default":false,"help":"Highest occupied and lowest unoccupied orbitals for each distinct spin channel.","group":"Cube files","input_modes":"parameters"},
    {"name":"cube_orbitals","label":"Orbital indices","kind":"text","default":"","help":"1-based indices in energy order within each spin channel. Separate with spaces or commas. Positive for alpha, negative for beta.","group":"Cube files","input_modes":"parameters"}
]"#;

pub(super) const CONVERT: &str = r#"
import ast
import json
import re
from bio_tool_adapters.psi4_input import from_boxes, to_boxes

CUBE_FLAGS = {"cube_density": "DENSITY", "cube_esp": "ESP", "cube_frontier": "FRONTIER_ORBITALS"}
CUBE_MARKER = "\n# Molchanica cube export\n"
CUBE_DRIVER = "_cube_result, cube_wfn = "
CUBE_OPTIONS = "cube_options = "
CUBE_CALL = """\
cube_tasks = cube_options['cubeprop_tasks']
cube_indices = cube_options.get('cubeprop_orbitals', [])
if any(abs(i) > cube_wfn.nmo() for i in cube_indices):
    raise ValueError('An orbital index exceeds the number of orbitals in this wavefunction.')

density_tasks = [task for task in cube_tasks if task in ('DENSITY', 'ESP')]
if 'ESP' in density_tasks:
    try:
        auxiliary = cube_wfn.get_basisset('DF_BASIS_SCF')
    except RuntimeError:
        auxiliary = None
    if auxiliary is None or auxiliary.name() == psi4.core.BasisSet.zero_ao_basis_set().name():
        orbital_basis = cube_wfn.basisset()
        auxiliary = psi4.core.BasisSet.build(
            cube_wfn.molecule(), 'DF_BASIS_SCF', psi4.core.get_option('SCF', 'DF_BASIS_SCF'),
            'JKFIT', orbital_basis.name(), puream=orbital_basis.has_puream())
        cube_wfn.set_basisset('DF_BASIS_SCF', auxiliary)
if density_tasks:
    psi4.set_options({'cubeprop_tasks': density_tasks})
    psi4.cubeprop(cube_wfn)

frontier = 'FRONTIER_ORBITALS' in cube_tasks
if frontier or cube_indices:
    # Use portable filenames. Some Psi4 symmetry labels contain quotes.
    grid = psi4.core.CubeProperties(cube_wfn)
    spin_channels = [(1, 'alpha', cube_wfn.Ca_subset, cube_wfn.nalpha()),
                     (-1, 'beta', cube_wfn.Cb_subset, cube_wfn.nbeta())]
    paired = cube_wfn.same_a_b_orbs() and cube_wfn.nalpha() == cube_wfn.nbeta()
    for spin, name, coefficients, occupied in spin_channels:
        indices = [abs(i) - 1 for i in cube_indices if i * spin > 0]
        if indices:
            grid.compute_orbitals(coefficients('AO', 'ALL'), indices, ['MO'] * len(indices), name)
        if frontier and (spin > 0 or not paired):
            if occupied:
                grid.compute_orbitals(coefficients('AO', 'OCC'), [occupied - 1], ['HOMO'], name)
            if occupied < cube_wfn.nmo():
                grid.compute_orbitals(coefficients('AO', 'VIR'), [0], ['LUMO'], name)
"""


def write_cubes(document, values):
    tasks = [task for name, task in CUBE_FLAGS.items() if values.get(name) == "true"]
    orbitals = []
    for value in re.split(r"[\s,]+", values.get("cube_orbitals", "").strip()):
        if not value:
            continue
        if not re.fullmatch(r"[+-]?[0-9]+", value) or not 0 < abs(int(value)) <= 2147483647:
            raise ValueError("Enter nonzero orbital indices separated by spaces or commas.")
        orbitals.append(int(value))
    if not tasks and not orbitals:
        return document

    options = {"cubeprop_tasks": tasks}
    if orbitals:
        tasks.append("ORBITALS")
        options["cubeprop_orbitals"] = list(dict.fromkeys(orbitals))

    before, _, driver = document.rstrip().rpartition("\n")
    return (before + CUBE_MARKER + CUBE_DRIVER + driver[:-1] + ", return_wfn=True)\n"
            + CUBE_OPTIONS + repr(options) + "\n" + CUBE_CALL)


def read_cubes(document, values):
    before, marker, suffix = document.rpartition(CUBE_MARKER)
    if not marker:
        return to_boxes(document, values)

    driver, _, rest = suffix.partition("\n")
    options, _, calls = rest.partition("\n")
    if (not driver.startswith(CUBE_DRIVER) or not driver.endswith(", return_wfn=True)")
            or not options.startswith(CUBE_OPTIONS) or calls.rstrip() != CUBE_CALL.rstrip()):
        raise ValueError("This cube export cannot be represented in the form.")

    options = ast.literal_eval(options[len(CUBE_OPTIONS):])
    if not isinstance(options, dict) or set(options) - {"cubeprop_tasks", "cubeprop_orbitals"}:
        raise ValueError("This cube export cannot be represented in the form.")
    tasks = options.get("cubeprop_tasks")
    orbitals = options.get("cubeprop_orbitals", [])
    if (not isinstance(tasks, list)
            or any(task not in (*CUBE_FLAGS.values(), "ORBITALS") for task in tasks)
            or not isinstance(orbitals, list) or any(type(index) is not int for index in orbitals)
            or ("ORBITALS" in tasks) != bool(orbitals)):
        raise ValueError("This cube export cannot be represented in the form.")

    driver = driver[len(CUBE_DRIVER):-len(", return_wfn=True)")] + ")"
    cube_values = {name: str(task in tasks).lower() for name, task in CUBE_FLAGS.items()}
    cube_values["cube_orbitals"] = ", ".join(map(str, orbitals))
    return to_boxes(before + "\n" + driver + "\n", values) | cube_values


request = json.load(sys.stdin)
values = request["values"]
try:
    if request["target"] == "text":
        result = {"psi4_input": write_cubes(from_boxes(values), values)}
    else:
        result = read_cubes(values.get("psi4_input", ""), values)
except (SyntaxError, ValueError) as error:
    print(json.dumps({"error": str(error)}))
else:
    print(json.dumps({"values": result}, ensure_ascii=False))
"#;

pub(crate) fn add_cube_fields(contract: &mut FormContract) -> Result<(), String> {
    let fields: Vec<_> = serde_json::from_str(CUBE_FIELDS).map_err(|error| error.to_string())?;
    contract.fields.extend(fields);

    Ok(())
}

pub(crate) fn uses_basis(values: &HashMap<String, String>) -> bool {
    let Some(method) = values.get("method") else {
        return true;
    };
    let method = method.trim().to_ascii_lowercase();

    !method.contains('/')
        && !matches!(
            method.as_str(),
            "hf3c"
                | "hf-3c"
                | "pbeh3c"
                | "pbeh-3c"
                | "b973c"
                | "b97-3c"
                | "r2scan3c"
                | "r2scan-3c"
                | "wb97x3c"
                | "wb97x-3c"
        )
}

pub(crate) fn wants_cubes(values: &HashMap<String, String>) -> bool {
    ["cube_density", "cube_esp", "cube_frontier"]
        .iter()
        .any(|name| values.get(*name).is_some_and(|value| value == "true"))
        || values
            .get("cube_orbitals")
            .is_some_and(|value| !value.trim().is_empty())
}
