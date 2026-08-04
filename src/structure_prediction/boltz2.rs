//! Boltz-2 co-folding and binding-affinity prediction.
//!
//! [Boltz](https://github.com/jwohlwend/boltz) ·
//! [Paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
//!
//! Boltz-2 predicts the same kind of complex OpenDDE does, and adds something no other predictor
//! Molchanica drives will do: a binding-affinity estimate for a ligand in the complex it just
//! folded. That is a direct complement to the docking and screening work — a structure and a
//! predicted affinity from one run, without a separate scoring pass.
//!
//! # Why this was previously disabled, and what changed
//!
//! The earlier integration provisioned its own environment at runtime and, failing that, fell back
//! to whatever `boltz` happened to be on `PATH`. Both were fragile for the same reason the module
//! docs in [`super`] describe: they depended on the user's Python setup. Boltz now installs the
//! same way OpenDDE does — a dedicated virtual environment built by `install_tool`, discovered by
//! the [`crate::external_tools`] registry, never resolved through a bare `PATH` lookup.
//!
//! Boltz requires Python `>=3.10,<3.13`, so its uv environment uses Python 3.12 rather than
//! sharing OpenDDE's 3.13 environment;
//! the per-tool environment layout exists precisely for conflicts like this. Its wheel is
//! `py3-none-any`, so Windows and Linux are both fine — only the optional `[cuda]` extra, which
//! pulls Linux-only `cuequivariance` wheels, is not, and the installer omits it off Linux.
//!
//! # Multiple sequence alignments
//!
//! Boltz is materially more accurate on proteins with an MSA, and its convenient route to one is
//! `--use_msa_server`, which sends the query to a public server. Molchanica runs offline by
//! default, and silently uploading a user's sequence would be a poor default for a desktop tool,
//! so single-sequence mode is the default and the server is opt-in per request.

use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use dynamics::params::ProtFfChargeMapSet;
use na_seq::{AminoAcid, Nucleotide};
use serde_json::Value;

use crate::{
    external_tools::{Tool, find_executable},
    molecules::peptide::MoleculePeptide,
    structure_prediction::{
        PredictionControl, PredictionWorkspace, amino_acid_sequence, dna_sequence, load_prediction,
        opendde::{OpenDdeEntity, OpenDdeRequest},
        run_model_command,
    },
};

/// Which compute device to ask Boltz for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Accelerator {
    /// Let Boltz pick, which uses a GPU when its Torch build has one.
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl Accelerator {
    fn argument(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Gpu => Some("gpu"),
            Self::Cpu => Some("cpu"),
        }
    }
}

/// Run settings. The defaults match Boltz's own, other than the MSA policy.
#[derive(Clone, Debug)]
pub struct BoltzOptions {
    /// Send protein sequences to Boltz's public MSA server.
    ///
    /// Off by default: it is a network round trip that discloses the query. Accuracy on proteins is
    /// noticeably better with it, so it is worth offering — but as a choice, not a surprise.
    pub use_msa_server: bool,
    /// Structures sampled from the diffusion model. More samples give a better chance of a good
    /// pose, at proportional cost.
    pub diffusion_samples: usize,
    /// Recycling passes through the trunk.
    pub recycling_steps: usize,
    pub accelerator: Accelerator,
    /// Predict binding affinity for this chain, which must be a ligand in the request.
    pub affinity_binder: Option<String>,
}

impl Default for BoltzOptions {
    fn default() -> Self {
        Self {
            use_msa_server: false,
            diffusion_samples: 1,
            recycling_steps: 3,
            accelerator: Accelerator::default(),
            affinity_binder: None,
        }
    }
}

impl BoltzOptions {
    fn validate(&self) -> io::Result<()> {
        if !(1..=25).contains(&self.diffusion_samples) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "diffusion samples must be between 1 and 25",
            ));
        }
        if !(1..=10).contains(&self.recycling_steps) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "recycling steps must be between 1 and 10",
            ));
        }
        Ok(())
    }
}

/// Boltz-2's affinity prediction for one ligand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoltzAffinity {
    /// Boltz's affinity value, as `log10(IC50)` with IC50 in µM. Lower binds more tightly.
    pub predicted_log_ic50: f32,
    /// The model's own probability that this is a binder at all. Read this first: an affinity
    /// value from a complex the model does not believe in is not worth interpreting.
    pub binary_probability: Option<f32>,
}

impl BoltzAffinity {
    /// IC50 in µM.
    pub fn ic50_micromolar(&self) -> f32 {
        10f32.powf(self.predicted_log_ic50)
    }

    /// The conventional `pIC50` (`-log10(IC50 / M)`), for comparison against measured data such as
    /// the `p_value` column of a PDBbind index.
    pub fn p_ic50(&self) -> f32 {
        // IC50 is in µM, so converting to molar subtracts log10(1e-6) = -6.
        -(self.predicted_log_ic50 - 6.0)
    }

    pub fn summary(&self) -> String {
        let mut summary = format!(
            "pIC50 {:.2} (IC50 ≈ {:.3} µM)",
            self.p_ic50(),
            self.ic50_micromolar()
        );
        if let Some(probability) = self.binary_probability {
            summary.push_str(&format!(" · binder probability {probability:.2}"));
        }
        summary
    }
}

/// What a run produced.
pub struct BoltzOutcome {
    pub molecule: MoleculePeptide,
    /// Present when an affinity binder was requested and Boltz wrote a prediction for it.
    pub affinity: Option<BoltzAffinity>,
}

/// Predict a complex, and optionally its binding affinity.
///
/// The request type is [`OpenDdeRequest`] because it is the application's co-folding request —
/// proteins, nucleic acids, ligands, ions, and covalent links — which OpenDDE merely named first.
/// Everything it can express maps onto Boltz's schema.
pub fn predict(
    request: &OpenDdeRequest,
    options: &BoltzOptions,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<BoltzOutcome> {
    request.validate()?;
    options.validate()?;
    control.check_cancelled()?;

    if let Some(binder) = &options.affinity_binder {
        let is_ligand = request
            .entities
            .iter()
            .any(|entity| matches!(entity, OpenDdeEntity::Ligand { id, .. } if id == binder));
        if !is_ligand {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "affinity can only be predicted for a ligand chain; '{binder}' is not one"
                ),
            ));
        }
    }

    let executable = find_executable(Tool::Boltz2)?;
    let workspace = PredictionWorkspace::new("boltz2")?;
    // Boltz names its output directory after the input file's stem, so this fixes where results
    // land without having to search for whatever name it chose.
    let input_path = workspace.path("molchanica.yaml");
    let output_path = workspace.create_dir("output")?;
    fs::write(&input_path, build_yaml(request, options)?)?;

    let mut command = Command::new(&executable);
    command
        .arg("predict")
        .arg(&input_path)
        .arg("--out_dir")
        .arg(&output_path)
        .arg("--output_format")
        .arg("mmcif")
        .arg("--diffusion_samples")
        .arg(options.diffusion_samples.to_string())
        .arg("--recycling_steps")
        .arg(options.recycling_steps.to_string())
        // Otherwise Boltz opens a progress display that is noise in a redirected pipe.
        .arg("--no_kernels");

    if options.use_msa_server {
        command.arg("--use_msa_server");
    }
    if let Some(accelerator) = options.accelerator.argument() {
        command.arg("--accelerator").arg(accelerator);
    }

    run_model_command(&mut command, "Boltz-2", control)?;
    control.check_cancelled()?;

    let affinity = options
        .affinity_binder
        .as_ref()
        .and_then(|_| read_affinity(&output_path));

    Ok(BoltzOutcome {
        molecule: load_prediction(&output_path, ff_map)?,
        affinity,
    })
}

/// The structure-only entry point, for the shared prediction dispatch in [`super`].
pub(super) fn predict_structure(
    request: &OpenDdeRequest,
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    predict(request, &BoltzOptions::default(), ff_map, control).map(|outcome| outcome.molecule)
}

pub(super) fn predict_structure_from_aas(
    aas: &[AminoAcid],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let sequence = amino_acid_sequence(aas)?;
    let name: String = format!("boltz_pred_{}", sequence.chars().take(5).collect::<String>());
    let request = OpenDdeRequest::new(name, vec![OpenDdeEntity::protein_sequence("A", sequence)]);
    predict_structure(&request, ff_map, control)
}

pub(super) fn predict_structure_from_dna(
    nts: &[Nucleotide],
    ff_map: &ProtFfChargeMapSet,
    control: &PredictionControl,
) -> io::Result<MoleculePeptide> {
    let sequence = dna_sequence(nts)?;
    let name: String = format!("boltz_pred_{}", sequence.chars().take(5).collect::<String>());
    let request = OpenDdeRequest::new(name, vec![OpenDdeEntity::dna_sequence("D", sequence)]);
    predict_structure(&request, ff_map, control)
}

/// Render the request as Boltz's YAML input schema.
///
/// Written directly rather than through a serializer because the schema is small, fixed, and
/// nested in a way that reads more clearly as a template than as a builder — and because adding a
/// YAML dependency to emit nine lines is not a good trade.
fn build_yaml(request: &OpenDdeRequest, options: &BoltzOptions) -> io::Result<String> {
    let mut yaml = String::from("version: 1\nsequences:\n");

    for entity in &request.entities {
        match entity {
            OpenDdeEntity::Protein { id, sequence } => {
                yaml.push_str(&format!(
                    "  - protein:\n      id: {id}\n      sequence: {}\n",
                    sequence.to_ascii_uppercase()
                ));
                if !options.use_msa_server {
                    // Single-sequence mode. Without either this or an MSA, Boltz refuses to run
                    // rather than quietly proceeding, so it has to be stated explicitly.
                    yaml.push_str("      msa: empty\n");
                }
            }
            OpenDdeEntity::Dna { id, sequence } => {
                yaml.push_str(&format!(
                    "  - dna:\n      id: {id}\n      sequence: {}\n",
                    sequence.to_ascii_uppercase()
                ));
            }
            OpenDdeEntity::Rna { id, sequence } => {
                yaml.push_str(&format!(
                    "  - rna:\n      id: {id}\n      sequence: {}\n",
                    sequence.to_ascii_uppercase()
                ));
            }
            OpenDdeEntity::Ligand { id, value } => {
                yaml.push_str(&format!("  - ligand:\n      id: {id}\n"));
                yaml.push_str(&ligand_body(value)?);
            }
            OpenDdeEntity::Ion { id, code } => {
                // Boltz has no separate ion entity; an ion is a ligand named by its CCD code.
                yaml.push_str(&format!(
                    "  - ligand:\n      id: {id}\n      ccd: {}\n",
                    code.to_ascii_uppercase()
                ));
            }
        }
    }

    if !request.covalent_bonds.is_empty() {
        yaml.push_str("constraints:\n");
        for bond in &request.covalent_bonds {
            // Boltz addresses bond ends as [chain, residue index, atom name]; the request holds
            // 1-based entity indices, which map onto the chain identifiers in declaration order.
            let chain = |index: usize| -> io::Result<&str> {
                request
                    .entities
                    .get(index - 1)
                    .map(entity_id)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("covalent bond references entity {index}, which does not exist"),
                        )
                    })
            };
            yaml.push_str(&format!(
                "  - bond:\n      atom1: [{}, {}, {}]\n      atom2: [{}, {}, {}]\n",
                chain(bond.entity1)?,
                bond.position1,
                bond.atom1,
                chain(bond.entity2)?,
                bond.position2,
                bond.atom2,
            ));
        }
    }

    if let Some(binder) = &options.affinity_binder {
        yaml.push_str(&format!(
            "properties:\n  - affinity:\n      binder: {binder}\n"
        ));
    }

    Ok(yaml)
}

/// The `smiles:` or `ccd:` line for a ligand.
///
/// The request's ligand field carries the OpenDDE conventions — a bare SMILES, a `CCD_` code, or a
/// `FILE_` reference — and only the first two have Boltz equivalents.
fn ligand_body(value: &str) -> io::Result<String> {
    let value = value.trim();
    if let Some(code) = value.strip_prefix("CCD_").or_else(|| value.strip_prefix("ccd_")) {
        return Ok(format!("      ccd: {}\n", code.to_ascii_uppercase()));
    }
    if value.to_ascii_uppercase().starts_with("FILE_") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Boltz-2 takes ligands as SMILES or CCD codes; it cannot read a FILE_ reference. \
             Use OpenDDE for that, or supply the ligand as SMILES.",
        ));
    }
    Ok(format!("      smiles: '{}'\n", escape_single_quoted(value)))
}

/// YAML single-quoted scalars escape a quote by doubling it, and need no other escaping — which is
/// what makes them the right choice for SMILES, full of backslashes and brackets.
fn escape_single_quoted(value: &str) -> String {
    value.replace('\'', "''")
}

fn entity_id(entity: &OpenDdeEntity) -> &str {
    match entity {
        OpenDdeEntity::Protein { id, .. }
        | OpenDdeEntity::Dna { id, .. }
        | OpenDdeEntity::Rna { id, .. }
        | OpenDdeEntity::Ligand { id, .. }
        | OpenDdeEntity::Ion { id, .. } => id,
    }
}

/// Find and parse the affinity prediction Boltz writes beside the structures.
///
/// Absent rather than an error when missing: a run without an affinity binder writes none, and a
/// structure prediction that succeeded should not be discarded because an optional extra did not
/// appear.
fn read_affinity(output_dir: &Path) -> Option<BoltzAffinity> {
    let path = find_affinity_json(output_dir)?;
    let parsed: Value = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;

    Some(BoltzAffinity {
        predicted_log_ic50: parsed.get("affinity_pred_value").and_then(Value::as_f64)? as f32,
        binary_probability: parsed
            .get("affinity_probability_binary")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
    })
}

fn find_affinity_json(directory: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(directory).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_affinity_json(&path) {
                return Some(found);
            }
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("affinity") && name.ends_with(".json"))
        {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure_prediction::opendde::OpenDdeCovalentBond;

    fn request() -> OpenDdeRequest {
        OpenDdeRequest::new(
            "complex",
            vec![
                OpenDdeEntity::protein_sequence("A", "ACDEFG"),
                OpenDdeEntity::ligand("B", "CC(=O)O"),
                OpenDdeEntity::ion("C", "MG"),
            ],
        )
    }

    #[test]
    fn writes_single_sequence_mode_by_default() {
        let yaml = build_yaml(&request(), &BoltzOptions::default()).expect("should render");

        assert!(yaml.starts_with("version: 1\nsequences:\n"));
        assert!(yaml.contains("  - protein:\n      id: A\n      sequence: ACDEFG\n      msa: empty\n"));
        // Nothing should have been sent anywhere, and no affinity block was requested.
        assert!(!yaml.contains("properties"));
    }

    #[test]
    fn omits_the_empty_msa_marker_when_the_server_is_used() {
        let options = BoltzOptions {
            use_msa_server: true,
            ..BoltzOptions::default()
        };
        let yaml = build_yaml(&request(), &options).expect("should render");
        assert!(!yaml.contains("msa: empty"));
    }

    #[test]
    fn maps_ligands_ions_and_ccd_codes() {
        let yaml = build_yaml(&request(), &BoltzOptions::default()).expect("should render");

        assert!(yaml.contains("  - ligand:\n      id: B\n      smiles: 'CC(=O)O'\n"));
        // An ion becomes a CCD-named ligand, since Boltz has no ion entity.
        assert!(yaml.contains("  - ligand:\n      id: C\n      ccd: MG\n"));

        let ccd = OpenDdeRequest::new("x", vec![OpenDdeEntity::ligand("L", "CCD_ATP")]);
        let yaml = build_yaml(&ccd, &BoltzOptions::default()).expect("should render");
        assert!(yaml.contains("      ccd: ATP\n"));
    }

    #[test]
    fn quotes_smiles_containing_quotes() {
        // Not chemically meaningful, but the escaping must hold or the YAML would be truncated.
        assert_eq!(escape_single_quoted("C'C"), "C''C");
        let body = ligand_body("C'C").expect("should render");
        assert_eq!(body, "      smiles: 'C''C'\n");
    }

    #[test]
    fn rejects_file_ligands_with_an_actionable_message() {
        let error = ligand_body("FILE_/tmp/x.sdf").expect_err("FILE_ is not supported");
        assert!(error.to_string().contains("SMILES or CCD"));
    }

    #[test]
    fn renders_covalent_bonds_against_chain_identifiers() {
        let mut request = request();
        request.covalent_bonds.push(OpenDdeCovalentBond {
            entity1: 1,
            copy1: 1,
            position1: 3,
            atom1: "SG".to_owned(),
            entity2: 2,
            copy2: 1,
            position2: 1,
            atom2: "C1".to_owned(),
        });

        let yaml = build_yaml(&request, &BoltzOptions::default()).expect("should render");
        assert!(yaml.contains("constraints:\n"));
        // Entity 1 is chain A and entity 2 is chain B, resolved by declaration order.
        assert!(yaml.contains("      atom1: [A, 3, SG]\n"));
        assert!(yaml.contains("      atom2: [B, 1, C1]\n"));
    }

    #[test]
    fn requests_affinity_for_a_named_binder() {
        let options = BoltzOptions {
            affinity_binder: Some("B".to_owned()),
            ..BoltzOptions::default()
        };
        let yaml = build_yaml(&request(), &options).expect("should render");
        assert!(yaml.ends_with("properties:\n  - affinity:\n      binder: B\n"));
    }

    #[test]
    fn converts_affinity_units_the_conventional_way() {
        // Boltz reports log10(IC50) with IC50 in µM, so 0 is 1 µM, which is pIC50 6.
        let affinity = BoltzAffinity {
            predicted_log_ic50: 0.0,
            binary_probability: Some(0.9),
        };
        assert!((affinity.ic50_micromolar() - 1.0).abs() < 1e-6);
        assert!((affinity.p_ic50() - 6.0).abs() < 1e-6);

        // A tighter binder: 1 nM is 1e-3 µM, so log10 is -3 and pIC50 is 9.
        let tight = BoltzAffinity {
            predicted_log_ic50: -3.0,
            binary_probability: None,
        };
        assert!((tight.p_ic50() - 9.0).abs() < 1e-6);
        assert!(tight.summary().contains("pIC50 9.00"));
    }

    #[test]
    fn rejects_out_of_range_sampling_options() {
        let mut options = BoltzOptions::default();
        options.diffusion_samples = 0;
        assert!(options.validate().is_err());

        options = BoltzOptions::default();
        options.recycling_steps = 50;
        assert!(options.validate().is_err());
    }
}
