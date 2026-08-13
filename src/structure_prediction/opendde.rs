//! OpenDDE all-atom co-folding through the official `opendde pred` CLI.
//!
//! [Github](https://github.com/aurekaresearch/OpenDDE)
//! [Paper](https://arxiv.org/html/2607.03787v1)
//! [Inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md)
//! [Website](https://aurekaresearch.github.io/OpenDDE-Website/)
//!
//! The shared `bio_tools` installer automatically selects the CUDA 12.6 backend when a working
//! NVIDIA GPU and compatible Windows or Linux driver are detected. It falls back to CPU if
//! detection, installation, or runtime verification fails.
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
    collections::{HashMap, HashSet},
    fs, io,
    path::PathBuf,
    process::Command,
};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};
use serde_json::{Value, json};

const PROTEIN_SEQUENCE_ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWYX";
const DNA_SEQUENCE_ALPHABET: &[u8] = b"ATGCNX";
const RNA_SEQUENCE_ALPHABET: &[u8] = b"AUGCNX";

use mol_defs::molecules::peptide::MoleculePeptide;

use crate::{
    external_tools::{self, Tool},
    structure_prediction::{
        PredictionControl, PredictionWorkspace, amino_acid_sequence, dna_sequence, load_prediction,
        run_model_command,
    },
    util::truncate_str,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenDdeComputeBackend {
    Cpu,
    Cuda,
    Other,
}

/// Compute environment selected by OpenDDE's `--device auto` mode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenDdeRuntimeInfo {
    pub backend: OpenDdeComputeBackend,
    pub selected_device: String,
    pub device_name: Option<String>,
    pub cuda_available: Option<bool>,
    pub cuda_version: Option<String>,
    pub nvidia_driver: Option<String>,
    pub pytorch_version: Option<String>,
    pub triangle_kernel: Option<String>,
    pub python_version: Option<String>,
    pub platform: Option<String>,
}

impl OpenDdeRuntimeInfo {
    pub fn short_label(&self) -> String {
        let backend = match self.backend {
            OpenDdeComputeBackend::Cpu => "CPU",
            OpenDdeComputeBackend::Cuda => "GPU/CUDA",
            OpenDdeComputeBackend::Other => self.selected_device.as_str(),
        };
        match self.device_name.as_deref() {
            Some(name) if !name.is_empty() => format!("{backend} ({name})"),
            _ => backend.to_owned(),
        }
    }

    pub fn detail_label(&self) -> String {
        let mut details = Vec::new();
        if let Some(version) = &self.cuda_version {
            details.push(format!("CUDA {version}"));
        }
        if let Some(driver) = &self.nvidia_driver {
            details.push(format!("NVIDIA driver {driver}"));
        }
        if let Some(version) = &self.pytorch_version {
            details.push(format!("PyTorch {version}"));
        }
        if let Some(kernel) = &self.triangle_kernel {
            details.push(triangle_kernel_label(kernel));
        }
        if self.backend == OpenDdeComputeBackend::Cpu
            && let Some(platform) = &self.platform
        {
            details.push(platform.clone());
        }
        details.join(" · ")
    }

    pub fn diagnostics_text(&self) -> String {
        let mut details = vec![format!("Inference device: {}", self.selected_device)];
        if let Some(name) = &self.device_name {
            details.push(format!("Device: {name}"));
        }
        if let Some(available) = self.cuda_available {
            details.push(format!("CUDA available: {available}"));
        }
        if let Some(version) = &self.cuda_version {
            details.push(format!("CUDA version: {version}"));
        }
        if let Some(driver) = &self.nvidia_driver {
            details.push(format!("NVIDIA driver: {driver}"));
        }
        if let Some(version) = &self.pytorch_version {
            details.push(format!("PyTorch: {version}"));
        }
        if let Some(kernel) = &self.triangle_kernel {
            details.push(format!("Triangle kernel: {kernel}"));
        }
        if let Some(version) = &self.python_version {
            details.push(format!("Python: {version}"));
        }
        if let Some(platform) = &self.platform {
            details.push(format!("Platform: {platform}"));
        }
        details.join("\n")
    }
}

fn triangle_kernel_label(kernel: &str) -> String {
    match kernel.to_ascii_lowercase().as_str() {
        "torch" => "PyTorch triangle kernels".to_owned(),
        "cueq" | "cuequivariance" => "cuEquivariance triangle kernels".to_owned(),
        _ => format!("{kernel} triangle kernels"),
    }
}

/// Ask OpenDDE which compute device and kernels its automatic inference mode will select.
///
/// `opendde doctor` imports PyTorch and may take a few seconds, so callers should run this away
/// from the UI thread.
pub fn probe_runtime() -> io::Result<OpenDdeRuntimeInfo> {
    let executable = find_executable()?;
    let output = Command::new(&executable)
        .env("PYTHONUNBUFFERED", "1")
        .arg("doctor")
        .output()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "unable to run OpenDDE diagnostics using {}: {error}",
                    executable.display()
                ),
            )
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };
        return Err(io::Error::other(format!(
            "OpenDDE doctor exited with {}: {}",
            output.status,
            truncate_str(detail, 2_048)
        )));
    }

    parse_doctor_report(&stdout)
}

fn parse_doctor_report(report: &str) -> io::Result<OpenDdeRuntimeInfo> {
    let values: HashMap<&str, &str> = report
        .lines()
        .filter_map(|line| {
            let item = line.trim().strip_prefix("- ")?;
            let (key, value) = item.split_once(':')?;
            Some((key.trim(), value.trim()))
        })
        .collect();
    let value = |key: &str| values.get(key).copied();

    let cuda_available =
        value("torch.cuda.is_available").map(|available| available.eq_ignore_ascii_case("true"));
    let selected_device = value("Selected inference device for auto mode")
        .map(str::to_owned)
        .or_else(|| match cuda_available {
            Some(true) => Some("cuda:0".to_owned()),
            Some(false) => Some("cpu".to_owned()),
            None => None,
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "OpenDDE doctor did not report its selected inference device",
            )
        })?;

    let selected_device_lower = selected_device.to_ascii_lowercase();
    let backend = if selected_device_lower.starts_with("cuda") {
        OpenDdeComputeBackend::Cuda
    } else if selected_device_lower.starts_with("cpu") {
        OpenDdeComputeBackend::Cpu
    } else {
        OpenDdeComputeBackend::Other
    };

    let device_index = selected_device_lower
        .strip_prefix("cuda:")
        .and_then(|index| index.parse::<usize>().ok())
        .unwrap_or(0);
    let device_key = format!("CUDA device {device_index}");
    let nvidia_smi = value("nvidia-smi").filter(|value| useful_diagnostic_value(value));
    let (smi_device, nvidia_driver) = nvidia_smi
        .and_then(|value| value.rsplit_once(','))
        .map(|(device, driver)| (Some(device.trim()), Some(driver.trim())))
        .unwrap_or((None, None));
    let device_name = value(&device_key)
        .filter(|value| useful_diagnostic_value(value))
        .or(smi_device)
        .map(str::to_owned);

    Ok(OpenDdeRuntimeInfo {
        backend,
        selected_device,
        device_name,
        cuda_available,
        cuda_version: value("torch CUDA version")
            .filter(|value| useful_diagnostic_value(value))
            .map(str::to_owned),
        nvidia_driver: nvidia_driver.map(str::to_owned),
        pytorch_version: value("PyTorch")
            .filter(|value| useful_diagnostic_value(value))
            .map(str::to_owned),
        triangle_kernel: value("Selected triangle kernel for auto mode")
            .filter(|value| useful_diagnostic_value(value))
            .map(str::to_owned),
        python_version: value("Python")
            .filter(|value| useful_diagnostic_value(value))
            .map(str::to_owned),
        platform: value("Platform")
            .filter(|value| useful_diagnostic_value(value))
            .map(str::to_owned),
    })
}

fn useful_diagnostic_value(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "none" | "missing" | "not found" | "unavailable" | "unknown"
    )
}

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
    pub fn protein_sequence(id: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self::Protein {
            id: id.into(),
            sequence: sequence.into(),
        }
    }

    pub fn protein(id: impl Into<String>, aas: &[AminoAcid]) -> io::Result<Self> {
        Ok(Self::protein_sequence(id, amino_acid_sequence(aas)?))
    }

    pub fn dna_sequence(id: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self::Dna {
            id: id.into(),
            sequence: sequence.into(),
        }
    }

    pub fn dna(id: impl Into<String>, nts: &[Nucleotide]) -> io::Result<Self> {
        Ok(Self::dna_sequence(id, dna_sequence(nts)?))
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

    pub fn validate(&self) -> io::Result<()> {
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

pub(super) fn predict_structure(
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
        .arg("--device")
        .arg("auto")
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

/// Locate OpenDDE in its Molchanica-managed uv environment.
///
/// This is now the registry's generic resolution — explicit override, then the managed uv
/// environment, with no bare `PATH` fallback. It is kept as a named function here because
/// `MOLCHANICA_OPENDDE_EXECUTABLE` and `OPENDDE_VENV_DIR` are documented, and the registry entry
/// preserves both.
pub(crate) fn find_executable() -> io::Result<PathBuf> {
    external_tools::find_executable(Tool::OpenDde)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gpu_runtime_details_from_open_dde_doctor() {
        let report = r#"OpenDDE environment
- Python: 3.13.5
- Platform: Windows-11-10.0.26200-SP0
- PyTorch: 2.7.1+cu126
- torch.cuda.is_available: True
- torch CUDA version: 12.6
- CUDA device count: 1
- CUDA device 0: NVIDIA GeForce RTX 4080
- CUDA probe error: none
- nvidia-smi: NVIDIA GeForce RTX 4080, 595.97
- Selected inference device for auto mode: cuda:0
- Selected triangle kernel for auto mode: torch
"#;

        let info = parse_doctor_report(report).expect("GPU report should parse");

        assert_eq!(info.backend, OpenDdeComputeBackend::Cuda);
        assert_eq!(info.selected_device, "cuda:0");
        assert_eq!(info.device_name.as_deref(), Some("NVIDIA GeForce RTX 4080"));
        assert_eq!(info.cuda_version.as_deref(), Some("12.6"));
        assert_eq!(info.nvidia_driver.as_deref(), Some("595.97"));
        assert_eq!(info.short_label(), "GPU/CUDA (NVIDIA GeForce RTX 4080)");
        assert_eq!(
            info.detail_label(),
            "CUDA 12.6 · NVIDIA driver 595.97 · PyTorch 2.7.1+cu126 · PyTorch triangle kernels"
        );
    }

    #[test]
    fn parses_cpu_runtime_details_from_open_dde_doctor() {
        let report = r#"OpenDDE environment
- Python: 3.12.9
- Platform: Linux-6.8.0-x86_64-with-glibc2.39
- PyTorch: 2.7.1+cpu
- torch.cuda.is_available: False
- torch CUDA version: none
- CUDA device count: 0
- nvidia-smi: unavailable
- Selected inference device for auto mode: cpu
- Selected triangle kernel for auto mode: torch
"#;

        let info = parse_doctor_report(report).expect("CPU report should parse");

        assert_eq!(info.backend, OpenDdeComputeBackend::Cpu);
        assert_eq!(info.selected_device, "cpu");
        assert_eq!(info.device_name, None);
        assert_eq!(info.cuda_version, None);
        assert_eq!(info.short_label(), "CPU");
        assert_eq!(
            info.detail_label(),
            "PyTorch 2.7.1+cpu · PyTorch triangle kernels · Linux-6.8.0-x86_64-with-glibc2.39"
        );
    }

    #[test]
    fn serializes_a_mixed_complex_using_the_open_dde_schema() {
        let mut request = OpenDdeRequest::new(
            "mixed_complex",
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDEX"),
                OpenDdeEntity::dna_sequence("D", "ATGCNX"),
                OpenDdeEntity::rna("R", "AUGCNX"),
                OpenDdeEntity::ligand("L", "CCD_ATP"),
                OpenDdeEntity::ion("M", "MG"),
            ],
        );
        request.covalent_bonds.push(OpenDdeCovalentBond {
            entity1: 1,
            copy1: 1,
            position1: 2,
            atom1: "SG".to_owned(),
            entity2: 4,
            copy2: 1,
            position2: 1,
            atom2: "C1".to_owned(),
        });

        assert_eq!(
            request.to_json().expect("request should serialize"),
            json!([{
                "name": "mixed_complex",
                "modelSeeds": [101],
                "sequences": [
                    {"proteinChain": {"sequence": "ACDEX", "count": 1, "id": ["A"]}},
                    {"dnaSequence": {"sequence": "ATGCNX", "count": 1, "id": ["D"]}},
                    {"rnaSequence": {"sequence": "AUGCNX", "count": 1, "id": ["R"]}},
                    {"ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["L"]}},
                    {"ion": {"ion": "MG", "count": 1, "id": ["M"]}},
                ],
                "covalent_bonds": [{
                    "entity1": "1",
                    "copy1": 1,
                    "position1": "2",
                    "atom1": "SG",
                    "entity2": "4",
                    "copy2": 1,
                    "position2": "1",
                    "atom2": "C1",
                }],
            }])
        );
    }

    #[test]
    fn rejects_invalid_mixed_complex_inputs() {
        let invalid_entities = [
            OpenDdeEntity::protein_sequence("A", "ACDB"),
            OpenDdeEntity::dna_sequence("D", "AUGC"),
            OpenDdeEntity::rna("R", "ATGC"),
            OpenDdeEntity::ion("M", "CCD_MG"),
        ];
        for entity in invalid_entities {
            let request = OpenDdeRequest::new("invalid", vec![entity]);
            assert!(request.validate().is_err());
        }

        let duplicate_ids = OpenDdeRequest::new(
            "duplicates",
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDE"),
                OpenDdeEntity::ligand("A", "CCO"),
            ],
        );
        assert!(duplicate_ids.validate().is_err());

        let mut bad_bond = OpenDdeRequest::new(
            "bad_bond",
            vec![OpenDdeEntity::protein_sequence("A", "ACDE")],
        );
        bad_bond.covalent_bonds.push(OpenDdeCovalentBond {
            entity1: 1,
            copy1: 1,
            position1: 1,
            atom1: "SG".to_owned(),
            entity2: 2,
            copy2: 1,
            position2: 1,
            atom2: "C1".to_owned(),
        });
        assert!(bad_bond.validate().is_err());
    }
}
