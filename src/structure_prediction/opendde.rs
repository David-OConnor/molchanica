//! OpenDDE all-atom co-folding through the official `opendde pred` CLI.
//!
//! [Github](https://github.com/aurekaresearch/OpenDDE)
//! [Paper](https://arxiv.org/html/2607.03787v1)
//! [Inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md)
//! [Website](https://aurekaresearch.github.io/OpenDDE-Website/)
//!
//! The four installers in `install_scripts` automatically select the CUDA 12.6 backend when a
//! working NVIDIA GPU and compatible Windows or Linux driver are detected. They fall back to CPU
//! if detection, installation, or runtime verification fails.
//!
//! The module supports
//! proteins, DNA, RNA, ligands (SMILES, `CCD_...`, or `FILE_...`), ions, and explicit covalent
//! links using the documented AlphaFold-Server-style JSON schema.
//!
//! Example use from CLI; we call it in a similar way.
//! ```json
//! [
//!     {
//!         "name": "tiny",
//!         "modelSeeds": [101],
//!         "sequences": [
//!             {
//!                 "proteinChain": {
//!                     "sequence": "ACDEFGHIK",
//!                     "count": 1
//!                 }
//!             }
//!         ]
//!     }
//! ]
//! ```
//! ```
//! opendde pred \
//!   -i tiny.json \
//!   -o ./output \
//!   -n opendde_v1 \
//!   --use_msa false \
//!   --use_template false \
//!   --use_rna_msa false \
//!   --sample 1 \
//!   --step 200 \
//!   --cycle 10
//! ```

use std::{
    collections::HashSet,
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};
use serde_json::{Value, json};

const PROTEIN_SEQUENCE_ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWYX";
const DNA_SEQUENCE_ALPHABET: &[u8] = b"ATGCNX";
const RNA_SEQUENCE_ALPHABET: &[u8] = b"AUGCNX";
const OPENDDE_EXECUTABLE_ENV: &str = "MOLCHANICA_OPENDDE_EXECUTABLE";
const OPENDDE_VENV_DIR_ENV: &str = "OPENDDE_VENV_DIR";

use crate::{
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionControl, PredictionWorkspace, amino_acid_sequence, dna_sequence, load_prediction,
        run_model_command,
    },
};

/// One entity in an OpenDDE co-folding request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenDdeEntity {
    Protein {
        id: String,
        sequence: String,
    },
    Dna {
        id: String,
        sequence: String,
    },
    Rna {
        id: String,
        sequence: String,
    },
    /// The value may be a SMILES string, a `CCD_...` code, or a `FILE_...` reference.
    Ligand {
        id: String,
        value: String,
    },
    /// OpenDDE expects a CCD component name without the `CCD_` prefix, such as `MG`.
    Ion {
        id: String,
        code: String,
    },
}

impl OpenDdeEntity {
    pub fn protein(id: impl Into<String>, aas: &[AminoAcid]) -> io::Result<Self> {
        Ok(Self::Protein {
            id: id.into(),
            sequence: amino_acid_sequence(aas)?,
        })
    }

    pub fn dna(id: impl Into<String>, nts: &[Nucleotide]) -> io::Result<Self> {
        Ok(Self::Dna {
            id: id.into(),
            sequence: dna_sequence(nts)?,
        })
    }

    pub fn rna(id: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self::Rna {
            id: id.into(),
            sequence: sequence.into(),
        }
    }

    pub fn ligand(id: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Ligand {
            id: id.into(),
            value: value.into(),
        }
    }

    pub fn ion(id: impl Into<String>, code: impl Into<String>) -> Self {
        Self::Ion {
            id: id.into(),
            code: code.into(),
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Protein { id, .. }
            | Self::Dna { id, .. }
            | Self::Rna { id, .. }
            | Self::Ligand { id, .. }
            | Self::Ion { id, .. } => id,
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::Protein { id, sequence } => json!({
                "proteinChain": {
                    "sequence": sequence.to_ascii_uppercase(), "count": 1, "id": [id]
                }
            }),
            Self::Dna { id, sequence } => json!({
                "dnaSequence": {
                    "sequence": sequence.to_ascii_uppercase(), "count": 1, "id": [id]
                }
            }),
            Self::Rna { id, sequence } => json!({
                "rnaSequence": {
                    "sequence": sequence.to_ascii_uppercase(), "count": 1, "id": [id]
                }
            }),
            Self::Ligand { id, value } => json!({
                "ligand": { "ligand": value, "count": 1, "id": [id] }
            }),
            Self::Ion { id, code } => json!({
                "ion": { "ion": code.to_ascii_uppercase(), "count": 1, "id": [id] }
            }),
        }
    }
}

/// An explicit covalent link between two 1-based entity positions in an OpenDDE request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenDdeCovalentBond {
    pub entity1: usize,
    pub copy1: usize,
    pub position1: usize,
    pub atom1: String,
    pub entity2: usize,
    pub copy2: usize,
    pub position2: usize,
    pub atom2: String,
}

impl OpenDdeCovalentBond {
    fn to_json(&self) -> Value {
        json!({
            "entity1": self.entity1.to_string(),
            "copy1": self.copy1,
            "position1": self.position1.to_string(),
            "atom1": self.atom1,
            "entity2": self.entity2.to_string(),
            "copy2": self.copy2,
            "position2": self.position2.to_string(),
            "atom2": self.atom2,
        })
    }
}

/// A complete OpenDDE co-folding job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenDdeRequest {
    pub name: String,
    pub seed: u64,
    pub entities: Vec<OpenDdeEntity>,
    pub covalent_bonds: Vec<OpenDdeCovalentBond>,
}

impl OpenDdeRequest {
    pub fn new(name: impl Into<String>, entities: Vec<OpenDdeEntity>) -> Self {
        Self {
            name: name.into(),
            seed: 101,
            entities,
            covalent_bonds: Vec::new(),
        }
    }

    fn validate(&self) -> io::Result<()> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "OpenDDE job name must contain only ASCII letters, digits, '_' or '-'",
            ));
        }
        if self.entities.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "OpenDDE request must contain at least one entity",
            ));
        }

        let mut ids = HashSet::new();
        for entity in &self.entities {
            let id = entity.id();
            if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid OpenDDE chain ID '{id}'"),
                ));
            }
            if !ids.insert(id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("duplicate OpenDDE chain ID '{id}'"),
                ));
            }

            let (kind, value, sequence_alphabet): (&str, &str, Option<&[u8]>) = match entity {
                OpenDdeEntity::Protein { sequence, .. } => {
                    ("protein", sequence, Some(PROTEIN_SEQUENCE_ALPHABET))
                }
                OpenDdeEntity::Dna { sequence, .. } => {
                    ("DNA", sequence, Some(DNA_SEQUENCE_ALPHABET))
                }
                OpenDdeEntity::Rna { sequence, .. } => {
                    ("RNA", sequence, Some(RNA_SEQUENCE_ALPHABET))
                }
                OpenDdeEntity::Ligand { value, .. } => ("ligand", value, None),
                OpenDdeEntity::Ion { code, .. } => ("ion", code, None),
            };

            if value.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("OpenDDE {kind} value cannot be empty"),
                ));
            }

            if let Some(alphabet) = sequence_alphabet
                && !value
                    .bytes()
                    .all(|byte| alphabet.contains(&byte.to_ascii_uppercase()))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("OpenDDE {kind} sequence contains an unsupported residue"),
                ));
            }
            if matches!(entity, OpenDdeEntity::Ion { .. })
                && (!value.bytes().all(|byte| byte.is_ascii_alphanumeric())
                    || value.to_ascii_uppercase().starts_with("CCD"))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "OpenDDE ion must be an unprefixed CCD component name such as 'MG'",
                ));
            }
        }

        for bond in &self.covalent_bonds {
            if bond.entity1 == 0
                || bond.entity1 > self.entities.len()
                || bond.entity2 == 0
                || bond.entity2 > self.entities.len()
                || bond.copy1 == 0
                || bond.copy2 == 0
                || bond.position1 == 0
                || bond.position2 == 0
                || bond.atom1.is_empty()
                || bond.atom2.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "OpenDDE covalent-bond references are 1-based and atoms cannot be empty",
                ));
            }
        }

        Ok(())
    }

    fn to_json(&self) -> io::Result<Value> {
        self.validate()?;
        let mut job = json!({
            "name": self.name,
            "modelSeeds": [self.seed],
            "sequences": self.entities.iter().map(OpenDdeEntity::to_json).collect::<Vec<_>>(),
        });
        if !self.covalent_bonds.is_empty() {
            job["covalent_bonds"] = Value::Array(
                self.covalent_bonds
                    .iter()
                    .map(OpenDdeCovalentBond::to_json)
                    .collect(),
            );
        }
        Ok(Value::Array(vec![job]))
    }
}

fn predict_structure(
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    control.check_cancelled()?;
    let workspace = PredictionWorkspace::new("opendde")?;
    let input_path = workspace.path("input.json");
    let output_path = workspace.create_dir("output")?;
    let input = serde_json::to_vec_pretty(&request.to_json()?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    fs::write(&input_path, input)?;

    // todo: Evaluate how to set various params here.

    let mut command = Command::new(find_executable()?);
    command
        // OpenDDE is a Python CLI. This makes progress and logging visible through redirected
        // pipes without waiting for Python's block buffer to fill.
        .env("PYTHONUNBUFFERED", "1")
        .arg("pred")
        .arg("-i")
        .arg(&input_path)
        .arg("-o")
        .arg(&output_path)
        .arg("-n")
        .arg("opendde_v1")
        .arg("--use_msa")
        .arg("false")
        .arg("--use_template")
        .arg("false")
        .arg("--use_rna_msa")
        .arg("false")
        .arg("--sample")
        .arg("1")
        .arg("--step")
        .arg("200")
        .arg("--cycle")
        .arg("10");

    run_model_command(&mut command, "OpenDDE", control)?;
    control.check_cancelled()?;

    load_prediction(&output_path, ff_map)
}

/// Locate OpenDDE in a Molchanica-managed environment or on `PATH`.
///
/// Checking managed directories directly is important for desktop-launched applications, which may
/// not inherit PATH additions made by a shell profile. The executable override is useful for custom
/// or development installations.
pub(crate) fn find_executable() -> io::Result<PathBuf> {
    if let Some(configured) = env::var_os(OPENDDE_EXECUTABLE_ENV) {
        let configured = PathBuf::from(configured);
        if configured.is_file() {
            return Ok(configured);
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{OPENDDE_EXECUTABLE_ENV} points to {}, but that file does not exist",
                configured.display()
            ),
        ));
    }

    // The dedicated venv installer is explicitly scoped to Molchanica, so prefer it when present.
    if let Some(executable) = find_managed_venv_executable()? {
        return Ok(executable);
    }

    // Prefer the isolated uv installation created by Molchanica's installer over a potentially
    // conflicting `pip install` in a global Python environment.
    for directory in known_uv_tool_bin_directories() {
        if let Some(executable) = executable_in(&directory, "opendde") {
            return Ok(executable);
        }
    }

    if let Some(uv) = find_uv_executable()
        && let Ok(output) = Command::new(uv).args(["tool", "dir", "--bin"]).output()
        && output.status.success()
        && let Ok(directory) = String::from_utf8(output.stdout)
        && let Some(executable) = executable_in(Path::new(directory.trim()), "opendde")
    {
        return Ok(executable);
    }

    if let Some(executable) = find_on_path("opendde") {
        return Ok(executable);
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "OpenDDE was not found. Run an install_opendde script from install_scripts, or set \
         MOLCHANICA_OPENDDE_EXECUTABLE.",
    ))
}

fn find_managed_venv_executable() -> io::Result<Option<PathBuf>> {
    if let Some(configured) = env::var_os(OPENDDE_VENV_DIR_ENV) {
        let root = PathBuf::from(configured);
        return venv_executable(&root).map(Some).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{OPENDDE_VENV_DIR_ENV} points to {}, but it contains no OpenDDE executable",
                    root.display()
                ),
            )
        });
    }

    Ok(default_managed_venv_root().and_then(|root| venv_executable(&root)))
}

fn default_managed_venv_root() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let data_home = env::var_os("LOCALAPPDATA").map(PathBuf::from);

    #[cfg(target_os = "macos")]
    let data_home = home_directory().map(|home| home.join("Library/Application Support"));

    #[cfg(all(unix, not(target_os = "macos")))]
    let data_home = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| home_directory().map(|home| home.join(".local/share")));

    data_home.map(|root| root.join("molchanica/opendde-venv"))
}

fn venv_executable(root: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let bin = root.join("Scripts");
    #[cfg(not(target_os = "windows"))]
    let bin = root.join("bin");

    executable_in(&bin, "opendde")
}

fn find_uv_executable() -> Option<PathBuf> {
    find_on_path("uv").or_else(|| {
        home_directory().and_then(|home| {
            [home.join(".local/bin"), home.join(".cargo/bin")]
                .into_iter()
                .find_map(|directory| executable_in(&directory, "uv"))
        })
    })
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path).find_map(|directory| executable_in(&directory, name))
    })
}

fn known_uv_tool_bin_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Some(path) = env::var_os("UV_TOOL_BIN_DIR") {
        directories.push(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("XDG_BIN_HOME") {
        directories.push(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("XDG_DATA_HOME") {
        directories.push(PathBuf::from(path).join("../bin"));
    }
    if let Some(home) = home_directory() {
        directories.push(home.join(".local/bin"));
    }
    directories
}

fn home_directory() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn executable_in(directory: &Path, name: &str) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let names = [
        format!("{name}.exe"),
        format!("{name}.cmd"),
        format!("{name}.bat"),
        name.to_owned(),
    ];
    #[cfg(not(target_os = "windows"))]
    let names = [name.to_owned()];

    names
        .into_iter()
        .map(|name| directory.join(name))
        .find(|candidate| candidate.is_file())
}

pub(super) fn predict_structure_from_aas(
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let entity = OpenDdeEntity::protein("A", aas)?;

    let aa_str: String = amino_acid_sequence(aas)?.chars().take(5).collect();
    let name = format!("opendde_pred_{aa_str}");

    predict_structure(&OpenDdeRequest::new(name, vec![entity]), ff_map, control)
}

pub(super) fn predict_structure_from_dna(
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let entity = OpenDdeEntity::dna("D", nts)?;

    let nt_str: String = dna_sequence(nts)?.chars().take(5).collect();
    let name = format!("opendde_pred_{nt_str}");

    predict_structure(&OpenDdeRequest::new(name, vec![entity]), ff_map, control)
}
