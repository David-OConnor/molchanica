# Molchanica

[![Docs](https://docs.rs/dynamics/badge.svg)](https://www.athanorlab.com/docs)

[Home page](https://www.athanorlab.com/molchanica)
[Documentation](https://www.athanorlab.com/docs)

[Molchanica](https://www.athanorlab.com/molchanica) is a molecule visualization, editing, and computation tool. It combines functionality found in [PyMol](https://www.pymol.org/), [Chimera](https://www.cgl.ucsf.edu/chimera/), [Coot](https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/), [VMD](https://www.ks.uiuc.edu/Research/vmd/), [GROMACS](https://www.gromacs.org/), and [Avogadro](https://avogadro.cc/), and operates with the focus on visualizing molecules over time and space. Any molecule you open or build can move using molecular dynamics. it integrates with machine-learning tools for protein engineering, including RFDiffusion3, Boltz-2, OpenDDE, ProteinMPNN, and Chai-1.

Molchanica includes Computer Aided Drug Design (CADD) tools. For example, molecule screening and generation based on pharmacophores, machine-learning from empirical binding data and more. It automatically estimates ADME and toxicity parameters using a neural network trained on [Therapeutic data commons (TDC)](https://tdcommons.ai/) data.

We hope that Molchanica is intuitive enough that formal documentation isn't required; installing, running, opening, and viewing molecules should be easy for someone to figure out by downloading, launching, and clicking around the GUI. That said, see [the documentation](https://www.athanorlab.com/docs) for details.


## Installation

### Windows and Linux

[Download, unzip, and run](https://github.com/David-OConnor/molchanica/releases).

Setup scripts are included; these are optional. The application can be launched directly, but these
useful for setting up shortcuts in your OS.

- **Linux**: Run `setup_molchanica.sh`, included in the zip. (Do not run this with `sudo`.)
- **Windows**: Run `setup_molchanica.ps1`. The first time you run the program, you may get the message
  *"  Microsoft Defender prevented an unrecognized app from starting"*. To bypass this, click *More
  info*, then *Run
  Anyway*. A similar obstacle may be encountered on Windows: If it refuses to launch, with an error about "Smart App Control", you may disable "Smart App Control" in
  *Windows Settings*.

![Ligand dynamics](screenshots/docked_2.png)

### Mac, and linux distros we don't provide a binary for

Compile from source by [downloading and installing Rust](https://www.rust-lang.org/tools/install), then running `cargo build --release` from a CLI in the project directory. See notes in the *compiling* section below about setting up Amber parameter files, and either installing the CUDA toolkit, or disabling CUDA.

### Installing and using integrated third-party tools

Supports, and manages the installation of third-party tool. E.g. GROMACS, ORCA, Gemmi, and ML-based protein design tools. See [the documentation](https://www.athanorlab.com/docs/third_party_tools.html) for details.


### Protein design

- **Backbone design** using RFDiffusion3
- **Structure prediction** using OpenDDE, Boltz-2, Chai-1, and EsmFold2
- **Sequence prediction** using ProteinMPNN and LigandMPNN

## Functionality


- View the 3D structure of proteins and small molecules
- Create figures for publication
- Pymol-like selection syntax
- Molecular dynamics, using Amber force fields, and the OPC water model
- Visualize ligand docking
- Edit small molecules in 3D with integrated dynamics
- Visualize electron density from crystallography and Cryo-Em data
- Run [GROMACS](https://www.gromacs.org/) MD, and view trajectories using the same GUI as our native MD.
- Run [ORCA](https://www.faccts.de/orca/) quantum chemistry procedures on molecules (if ORCA is installed).
- Run [OpenDDE](https://github.com/aurekaresearch/OpenDDE) structure prediction (Proteins, ligands, nucleic acids, ions, complexes)
- Run [Boltz-2](https://github.com/jwohlwend/boltz) co-folding, including binding-affinity prediction for a ligand in the folded complex
- Design sequences for a backbone with [LigandMPNN](https://github.com/dauparas/LigandMPNN) and
  [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), conditioned on ligands and nucleic acids where present
- Scan every point mutation at every position for stability (ΔΔG), in a single pass, natively — no Python needed at run time
- Assess protein-ligand binding (affinity, pose validation, binding spot identification etc)
- Antibody workflows: CDR selection, paratope contacts, and developability triage
- Read a local [PDBbind](https://www.pdbbind-plus.org.cn/) release: look up a complex's measured binding affinity and open its protein, pocket, and ligand files
- Assess pharmacokinetics, screen small molecules for binding to pockets, pharmacophore features, molecule alignment,
  and more.
- Screen molecules based on binding affinity and desired properties
- Pharmacophore based modeling and screening

## Molecule types supported for viewing and dynamics

- Proteins
- Small organic molecules (e.g. ligands)
- DNA and RNA; double and single stranded
- Lipids

## Getting started

Launch the program. Either open a molecule using the "Open" or "Open Lig" buttons, drag the file into the program
window,
enter a protein identifier in the *Query databases* field, or click *I'm feeling
lucky*, to load a recently-uploaded
protein
from the [RCSB PDB](https://www.rcsb.org/).

**Most UI items provide tooltip descriptions, when you hover the mouse over them.**

## Goals

- Fast
- Easy-to-use
- Practical workflow
- Updates responsive to user feedback

## File formats

- **Proteins**: mmCIF (aka PDBx)
- **Small molecules**: SDF, Mol2, GRO, XYZ, and PDBQT
- **Electron density**: 2fo-fc mmCIF, Map, and MTZ
- **Force field parameters**: dat, lib, frcmod, prmtop (Amber), and top (GROMACS)
- **MD**: GRO, MDP, DCD, TRR, XTC


## Third-party tool integrations

Molchanica integrates with a number of third-party tools, acting on the molecules Molchanica has open,
or creating them. See [the docs page for details, and a full list of tools](https://www.athanorlab.com/docs/third_party_tools.html). These can be accessed directly from the computer if you have them installed,
or installed within the application's UI.


### Example tools supported:


- GROMACS
- ORCA
- RDKit
- RfDiffusion3
- Chai-1
- OpenDDE
- Boltz-2
- MPNN (Protein, ligand etc)


## A note on internet connectivity

Most functionality works without internet connectivity. If you have internet, it includes API integrations. For example:

- Load molecules and sequences automatically from PubChem, UniProt, drugbank, and RCSB PDB.
- Improved structure prediction accuracy using MSA servers
- Reactions from Rhea
- OPLS MD parameterization for small molecules
- Buttons and links to open relevant database pages for molecules operating on
- Finds data on the molecule you're building or editing as you work, if it's on PubChem.


![UI image showing mol properties and a protien](/screenshots/molchanica_ui_apr_2026.png)

## Parallel computing

If an Nvidia GPU of at least RTX 3 series is available, molecular dynamics, docking, and electron density calculations will be performed using the GPU (via CUDA kernels). If not, the CPU will be used, leveraging thread pools and SIMD instructions. It uses all cores available, and either 512-bit, or 256-bit, SIMD instructions, depending on CPU capability.

GPU functionality requires Nvidia driver version 580 or higher.

The standard Windows and Linux downloads are built with CUDA support, but do not require it: CUDA's
libraries are looked up when the program starts rather than being linked into the executable. On a
machine with no Nvidia GPU, no driver, or a driver too old for the CUDA version we build against,
the program starts normally and reports `Cpu` as its computing device. The separate `nocuda`
downloads are smaller, and are otherwise equivalent on such machines.


## Molecular dynamics

This application integrates classical molecular dynamics setup and playback into the GUI.
For details, see [the documentation](https://www.athanorlab.com/docs/md.html). It can use its [native backend]([Dynamics rust library](https://github.com/david-oconnor/dynamics)), or GROMACS (if the latter is installed). It uses AMBER, CHARMM, or OPLS force fields, with automatic parameterization.

It supports running MD directly on the following molecule types, without loading any other data:

- Proteins/amino acids
- Small organic molecules
- DNA and RNA
- Lipids
- Carbohydrates (WIP)


Overview of MD components:

- [Amber, CHARMM, or OPLS parameters](https://ambermd.org/AmberModels.php)
- Velocity Verlet integrator
- [CSVR](https://arxiv.org/pdf/0803.4060) or Langevin Middle thermostats
- Explicit OPC water model
- SPME Coulomb forces
- AM1-BCC partial charges (MBIS if you have ORCA installed)


## Editing molecules

We provide an editor for small organic molecules. This is fully-3D, and has integrated dynamics, which can be run continuously while you edit. It ensures the molecules you create are realistic, and take the appropriate shape. See the documentation for details on how the editor works.


## The camera

The camera is set up with 6 degrees of freedom, using either keyboard + mouse, or mouse only. This makes
it easy to get any view of the system you want.

There are two camera control schemes, selectable using buttons in the *camera* section of the GUI.

### Free camera

The *free
camera* mode is intended to be used with a keyboard and mouse together. They operate on the perspective of
the viewer, vice the molecule. You can move and rotate and move the camera
in 6 degrees of freedom, allowing you to easily view the molecule from any perspective.

### Arc camera

Similar to traditional molecular viewing software. The camera arcs (or orbits) around the molecule, when holding the
left
mouse button and dragging. Other controls, like scroll wheel and middle mouse, operate similar to the free camera.
If *orbit
sel* is set in the GUI, the orbit center will be the selected atom or residue, vice the molecule center.

![Surface example](screenshots/surface_a.png)

### Hotkeys

Most of this functionality is also available in the GUI, and you can find hotkeys from the GUI's tooltips. We recommend
starting with the GUI, and integrating hotkeys into your workflow as you go, using tooltips.
See below for a reference.

#### Mouse controls:

- Hold the **left mouse button while dragging** to rotate the camera in pitch and yaw.
- Hold the **middle mouse button while dragging** to move the camera left, right, up, and down.
- **Scroll** to move the camera forward and backwards.
- **Scroll while holding left mouse button** to roll.
- **Right click** to select the atom or residue under the cursor. This also selects the molecule to manipulate.

#### Camera hotkeys

- **Tab**: Outside the molecule editor, reset to the Front view of the active molecule.
- **W**: Move forward
- **A**: Move right
- **A**: Move left
- **D**: Move back
- **Space**: Move up
- **C**: Move down
- **Q**: Roll counter-clockwise
- **R**: Roll clockwise

- **Shift** (left): Hold to increase camera movement and rotation speed.
- **Scroll whlie holding left mouse**: Roll (Alternative to Q/R)

### Non-camera hotkeys

- **Left arrow**: select previous residue
- **Right arrow**: select next residue

- **Left arrow**: Select previous residue
- **Right arrow**: Select next residue

- **Left backet**: Previous view mode (sticks, surface mesh etc)
- **Right bracket**: Next view mode

- **Semicolon**: View/select level (Atoms, bonds, residues)
- **Right bracket**: Next view/select level

- **M**: Move a molecule with the mouse and scroll wheel
- **R**: Rotate a molecule with the mouse and scroll wheel

- **Enter**: Move the camera to the selected atom or residue.
- **Esc**: Exit move/rotate mode if in one. Otherwise, clear the selection (keeping the molecule active); if nothing is selected, deactivate the active molecule.

- **Ctrl + scroll**: Change the fog distance. (hide distant atoms)
- **Delete**: In the primary mode, close the active molecule. In the molecule editor, this deletes the selected atom or bond.
- **Enter**: Move the camera to the selected atom, bond, or residue.
- **Tab**: In the molecule editor, add an atom.

Note: The molecule editor has a different set of hot keys; see its tooltips for details.

![Protein B](screenshots/protein_b.png)


## Reflections and electron density

Supports volumetric and isosurface views for electron density data, e.g. from Cryo-EM and X-Ray crystallography data.

It can download this data from RCSB PDB, or load files directly. To open *2fo-fc* and
*MTZ* files, we use the [Gemmi](https://gemmi.readthedocs.io/en/latest/install.html) program. For this to work, the `gemmi` folder we include must remain co-located with the program's executable, or be on the system path. For convenience, we package it with the Windows release, and the linux install script optionally installs it with `apt`.

It can import Map files directly, and save load density to Map format.


## Pharmacokinetics

Uses neural networks to infer pharmacokinetic properties of arbitrary small molecules,
using [Therapeutic Data Commons](https://tdcommons.ai/) data. See the docs for details

## Protein superposition and comparison

See the docs for details


## PyMol-like Command line interface

Molchanica supports a very limited subset of PyMol's CLI interface. Supported commands:

![Solvent accessible surface mesh](screenshots/surface_mesh_transparent.png)

### General

- `help`: Lists commands
- `pwd`
- `ls`
- `cd`
- `set seq_view`

### File IO

- `fetch`: Loads a protein from the RCSB PDB. e.g. `fetch 1C8K`
- `save`: Save the opened protein or small molecule to disk. e.g. `save molecules/1htm.cif`
- `load`: Load a protein or small molecule from disk. e.g. `load ../1htm.cif`

### View and edit

- `show representation, expression`: Add a representation layer, e.g. `show sticks, ligand`. Without an expression, applies to `all`.
- `show_as`: Set the global default view mode, e.g. `show_as cartoon`. Existing layers remain independent.
- `view`: Save and load scenes. e.g. `view v1`, `view v1 store`, `view v2 recall`
- `hide`: Limited options available, e.g. `resn HOH`, `hydro`, `chain`, `hetatm` etc.
- `remove`: Limited options available, e.g. `resn HOH`, `hydro`, `chain`, `hetatm` etc.

### Selections

- `select expression`: Select atoms across loaded molecules. `sele` also works as the command name.
- `select name, expression`: Save or update a named selection.
- `select %name`: Recall a named selection. Bare names also work.
- `delete name`: Delete an unused named selection. References from other selections or layers must be removed first.

Supports most selection commands from Pymol, e.g.:

```text
select site, byres (protein within 4 of ligand)
show cartoon, protein
show sticks, site and not hydrogen
show spheres, ligand
show surface, protein
```

It also supports building selection commands via the GUI.

### Camera controls

- `turn`
- `move`
- `orient`
- `reset`

![Example of seq prediction with MPNN](screenshots/molchanica_seq_pred_mpnn.png)

### Selection modes

Using a dropdown in the UI, you can change selection mode between *Atom*, (covalent) *Bond*, and
*Residue*. These are
mostly self-explanatory.
When selecting a component using *right
click*, or toggling using the UI buttons or arrow keys, this changes
what is selected, and the nature of information displayed. For example, in
*Bond* mode, you can see bond length,
elements of the atoms connected, and the expected oscillation frequency of the bond. In
*Atom* mode, you can see
dihedral angles, the atom name as part of its residue, the atom's force field name, and its partial charge.

### MD playback

Molchanica can play back molecular dynamics trajectories run from other programs, including GROMACS, OpenMM, and NAMD.

![MD playback/viewer, minimal example](/screenshots/md_viewer_march_2026.png)

### Adding nucleic acids and lipids

You can add DNA, RNA, and lipids in various configurations without loading files; this program can create them procedurally using the GUI. It can create DNA and RNA from a given nucleic acid or amino acid sequence. It can create lipids arranged freely, as membrances, or as lipid nanoparticles (LNPs).

### The preferences file, and where Molchanica keeps its data

*molchanica_prefs.mca* is a small binary file containing application state. It's what lets the program remember the last file opened, current view settings etc. It will grow with the number of molecules you've opened, as it stores per-molecule settings. Deleting it is harmless, other than resetting these conveniences.

Molchanica keeps its data in one folder: the one its executable is in.


#### Upgrading from an earlier version

Versions before this change always kept `process_executables/` and `datasets/` in the OS per-user
data directory, separate from the executable. On first launch, Molchanica moves them into the
install folder — a rename, so it's instant and doesn't re-download anything. If the two are on
different drives the rename can't work; Molchanica then prints both paths and leaves the files
alone, so move them yourself or point `MOLCHANICA_DATA_DIR` at the old location.

![ELectron density](screenshots/iso_a.png)

### Compiling

This application is written in rust, so it compiles normally using `cargo build --release`, which produces a
standalone executable.

If you're not compiling on a machine with an Nvidia GPU or without the CUDA toolkit installed, append the
`--no-default-features` to the build command. This will disable GPU support on the executable produced.

The bundled molecule databases are optional. `hmdb.parquet` and `ChEBI.parquet` in the project root are embedded when present; missing files are reported in the molecule database popup. Add either file and rebuild to bundle it, or use **Load DB** to open a database at runtime.

#### Compiling with GPU support

If compiling with GPU support, your compiling PC must have
the [CUDA Toolkit, v13+](https://developer.nvidia.com/cuda-downloads) installed. You must set the environment var
`LD_LIBARARY_PATH` (Linux) or `Path` (Windows) to your CUDA bin directory, e.g.
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`. You may also need the build tools containing
`cl.exe` or
similar in the path, e.g.:
`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`

### Errata

- Opening electron density files in general can be slow. This can lead to the program starting slowly if it was
- Minimizing may pause molecular dynamics computations.
- On some displays (Ones with pixel scaling), dragging the MD time slider may also move the camera. To workaround, click
  the slider instead of dragging.

![Electron densities and small molecules](screenshots/density_example_nov_2025.png)

[//]: # (Wub from LabRats didn't dissaprove of the project name!)
