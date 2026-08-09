//! Inverse folding through the MPNN family: LigandMPNN, ProteinMPNN, and the AbMPNN weights.
//!
//! [LigandMPNN](https://github.com/dauparas/LigandMPNN) ·
//! [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
//!
//! These answer the question Molchanica otherwise cannot: given a backbone, what sequences would
//! fold into it? That pairs directly with structures already loaded — a designed binder, a
//! stabilized variant, a resurfaced antibody — and with the pocket and docking work, because
//! LigandMPNN conditions on ligands, nucleic acids, and ions rather than treating the protein as
//! though it sat in a vacuum.
//!
//! # Why three models behind one adapter
//!
//! They are the same architecture with different weights and different command-line front ends:
//!
//! - **LigandMPNN** (`run.py`) is the successor repository, and the one to reach for by default.
//!   One entry point covers protein, ligand, soluble, and membrane model types.
//! - **ProteinMPNN** (`protein_mpnn_run.py`) is the original. Kept because it is what the AbMPNN
//!   checkpoint was trained as a drop-in for.
//! - **AbMPNN** is ProteinMPNN's network with antibody-finetuned weights (Frey et al., ICML 2023
//!   CompBio workshop, CC BY 4.0). It is not a separate installation: the weights sit beside the
//!   vanilla ones in the same checkout, which is why it is a `MpnnModel` variant rather than its
//!   own registry entry.
//!
//! # Platform note
//!
//! Both are plain PyTorch against a checkout — no compiled CUDA kernels, no conda — so they run on
//! Windows and Linux alike, on CPU or GPU. The install script picks the Torch backend the same way
//! the OpenDDE one does.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use bio_files::{MmCif, ResidueType};
use serde_json::{Map, Value, json};

use crate::{
    external_tools::{
        Tool, ToolWorkspace, bundle_root, find_executable,
        pdb_write::{PdbWriteOptions, chain_letter, peptide_to_pdb},
    },
    molecules::peptide::MoleculePeptide,
};

/// Which network and weights to run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MpnnModel {
    /// LigandMPNN conditioned on the non-protein context present in the structure.
    #[default]
    LigandMpnn,
    /// LigandMPNN's protein-only model type: equivalent in scope to vanilla ProteinMPNN, but from
    /// the maintained repository.
    ProteinMpnnViaLigand,
    /// LigandMPNN's soluble model type, trained without membrane context.
    SolubleMpnn,
    /// The original ProteinMPNN checkout and weights.
    ProteinMpnn,
    /// The ProteinMPNN network with the antibody-finetuned AbMPNN checkpoint.
    AbMpnn,
}

impl MpnnModel {
    pub const ALL: [Self; 5] = [
        Self::LigandMpnn,
        Self::ProteinMpnnViaLigand,
        Self::SolubleMpnn,
        Self::ProteinMpnn,
        Self::AbMpnn,
    ];

    /// Which registry entry — and therefore which checkout and virtual environment — this uses.
    pub fn tool(self) -> Tool {
        match self {
            Self::LigandMpnn | Self::ProteinMpnnViaLigand | Self::SolubleMpnn => Tool::LigandMpnn,
            Self::ProteinMpnn | Self::AbMpnn => Tool::ProteinMpnn,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LigandMpnn => "LigandMPNN (ligand context)",
            Self::ProteinMpnnViaLigand => "ProteinMPNN (LigandMPNN repo)",
            Self::SolubleMpnn => "SolubleMPNN",
            Self::ProteinMpnn => "ProteinMPNN (original)",
            Self::AbMpnn => "AbMPNN (antibody-tuned)",
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            Self::LigandMpnn => {
                "Conditions on ligands, nucleic acids, and ions in the structure. The default \
                 choice when the backbone has any non-protein context worth keeping."
            }
            Self::ProteinMpnnViaLigand => {
                "Protein-only design from the maintained repository. Use when the non-protein \
                 content should be ignored."
            }
            Self::SolubleMpnn => {
                "Trained without membrane proteins; biases against hydrophobic surfaces."
            }
            Self::ProteinMpnn => "The original network and weights.",
            Self::AbMpnn => {
                "ProteinMPNN finetuned on antibody structures. Best for CDR and framework design; \
                 pairs with the antibody tools."
            }
        }
    }

    /// `--model_type` for LigandMPNN's `run.py`.
    fn ligand_model_type(self) -> Option<&'static str> {
        match self {
            Self::LigandMpnn => Some("ligand_mpnn"),
            Self::ProteinMpnnViaLigand => Some("protein_mpnn"),
            Self::SolubleMpnn => Some("soluble_mpnn"),
            Self::ProteinMpnn | Self::AbMpnn => None,
        }
    }

    /// Whether non-protein atoms should be written into the input file at all.
    fn wants_hetero_context(self) -> bool {
        matches!(self, Self::LigandMpnn)
    }
}

/// One design run.
#[derive(Clone, Debug)]
pub struct DesignRequest {
    pub model: MpnnModel,
    /// Chains to redesign. Empty designs every chain.
    pub chains_to_design: Vec<String>,
    /// Residues held at their current identity, as `<chain><residue number>`, e.g. `H97`. Used to
    /// keep a framework fixed while redesigning CDRs, or to preserve a catalytic site.
    pub fixed_residues: Vec<String>,
    /// How many sequences to generate.
    pub num_sequences: usize,
    /// Sampling temperature. Low values give conservative, near-consensus sequences; high values
    /// give diversity at the cost of predicted stability. 0.1 is both repositories' default.
    pub temperature: f32,
    pub seed: u64,
    /// Original ProteinMPNN checkpoint to use. Ignored by LigandMPNN-backed models.
    pub checkpoint: ProteinMpnnCheckpoint,
    /// Gaussian coordinate noise added at inference time.
    pub backbone_noise: f32,
    /// Amino acids which must never be sampled, as one-letter codes.
    pub omit_amino_acids: String,
    /// Optional design-only residue lists, one chain per line (`A 12 13 14`). Positions outside
    /// these lists are fixed. This mirrors the ProteinMPNN web interface.
    pub designed_residues: String,
    /// Tie equivalent positions across designed chains.
    pub homo_oligomer: bool,
    /// Global sampling bias, e.g. `W:3.0,P:3.0,A:-3.0`.
    pub bias_amino_acids: String,
    /// Sparse JSON keyed by chain and residue, e.g. `{"A12":{"G":-0.3}}`.
    pub bias_amino_acids_per_residue: String,
    /// Sparse JSON keyed by chain and residue, e.g. `{"A12":"CP"}`.
    pub omit_amino_acids_per_residue: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProteinMpnnCheckpoint {
    Noise002,
    Noise010,
    #[default]
    Noise020,
    Noise030,
}

impl ProteinMpnnCheckpoint {
    pub const ALL: [Self; 4] = [
        Self::Noise002,
        Self::Noise010,
        Self::Noise020,
        Self::Noise030,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Noise002 => "v_48_002",
            Self::Noise010 => "v_48_010",
            Self::Noise020 => "v_48_020",
            Self::Noise030 => "v_48_030",
        }
    }
}

impl Default for DesignRequest {
    fn default() -> Self {
        Self {
            model: MpnnModel::default(),
            chains_to_design: Vec::new(),
            fixed_residues: Vec::new(),
            num_sequences: 8,
            temperature: 0.1,
            seed: 37,
            checkpoint: ProteinMpnnCheckpoint::default(),
            backbone_noise: 0.0,
            omit_amino_acids: "X".to_owned(),
            designed_residues: String::new(),
            homo_oligomer: false,
            bias_amino_acids: String::new(),
            bias_amino_acids_per_residue: String::new(),
            omit_amino_acids_per_residue: String::new(),
        }
    }
}

impl DesignRequest {
    /// Reject a request before a process is started, so a typo in a fixed-residue list surfaces
    /// as a message beside the field rather than as a model that quietly fixed nothing.
    pub fn validate(&self) -> io::Result<()> {
        if self.num_sequences == 0 || self.num_sequences > 1_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "number of sequences must be between 1 and 1000",
            ));
        }
        if !(0.0001..=2.0).contains(&self.temperature) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sampling temperature must be between 0.0001 and 2.0",
            ));
        }
        if !(0.0..=1.0).contains(&self.backbone_noise) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "backbone noise must be between 0 and 1",
            ));
        }
        let omit = compact_letters(&self.omit_amino_acids);
        if !omit.chars().all(|letter| letter.is_ascii_alphabetic()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "omitted amino acids must contain only one-letter amino-acid codes",
            ));
        }
        parse_designed_residues(&self.designed_residues)?;
        parse_bias_amino_acids(&self.bias_amino_acids)?;
        parse_json_object(
            &self.bias_amino_acids_per_residue,
            "per-residue amino-acid bias",
        )?;
        parse_json_object(
            &self.omit_amino_acids_per_residue,
            "per-residue omitted amino acids",
        )?;
        for residue in &self.fixed_residues {
            // The repositories both parse these as a chain letter followed by a residue number.
            let mut chars = residue.chars();
            let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                && chars.clone().count() > 0
                && chars.all(|c| c.is_ascii_digit());
            if !valid {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "fixed residue '{residue}' should be a chain letter followed by a residue \
                         number, e.g. H97"
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// One generated sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignedSequence {
    /// The designed sequence, chains joined by `/` as both repositories write them.
    pub sequence: String,
    /// Mean per-residue negative log likelihood over the designed positions. Lower is better; this
    /// is the model's own confidence, and is what to rank designs by.
    pub score: Option<f32>,
    /// Fraction of positions matching the input sequence. Useful for spotting a run that changed
    /// almost nothing, or almost everything.
    pub sequence_recovery: Option<f32>,
    /// Sampling temperature this sequence was drawn at, as recorded in the FASTA header.
    pub temperature: Option<f32>,
}

impl DesignedSequence {
    /// The chains, split back apart.
    pub fn chains(&self) -> Vec<&str> {
        self.sequence.split('/').collect()
    }
}

/// What a run produced.
#[derive(Clone, Debug, Default)]
pub struct DesignResult {
    /// The input sequence as the model saw it, from the first FASTA record.
    pub input_sequence: Option<String>,
    /// Generated sequences, best (lowest score) first.
    pub designs: Vec<DesignedSequence>,
    pub raw_fasta: String,
}

/// Run an MPNN design against a loaded peptide.
///
/// Blocking, and slow enough to want a worker thread: the process start and Torch import dominate
/// for small designs.
pub fn design(mol: &MoleculePeptide, request: &DesignRequest) -> io::Result<DesignResult> {
    let options = PdbWriteOptions {
        // Chain filtering happens in the model, not the file: it needs the whole structure as
        // context even when only part of it is redesigned.
        chains: Vec::new(),
        include_hetero: request.model.wants_hetero_context(),
        include_hydrogen: false,
        include_water: false,
    };
    design_pdb_text(&peptide_to_pdb(mol, &options)?, request)
}

/// Run ProteinMPNN against an mmCIF selected from disk without adding it to the scene.
pub fn design_file(path: &Path, request: &DesignRequest) -> io::Result<DesignResult> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "cif" | "mmcif") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sequence prediction input must be an mmCIF (.cif or .mmcif) file",
        ));
    }
    let text = fs::read_to_string(path)?;
    design_protein_mpnn_structure(mmcif_to_mpnn_structure(&MmCif::new(&text)?)?, request)
}

/// Run the original ProteinMPNN workflow from an opened protein without using PDB as an
/// interchange format. The structure is encoded directly in ProteinMPNN's structure JSONL schema.
pub fn design_mmcif(mol: &MoleculePeptide, request: &DesignRequest) -> io::Result<DesignResult> {
    design_protein_mpnn_structure(peptide_to_mpnn_structure(mol)?, request)
}

#[derive(Debug)]
struct MpnnStructure {
    value: Value,
    residues: BTreeMap<String, Vec<i32>>,
    chains: Vec<String>,
}

#[derive(Debug)]
struct MpnnResidue {
    number: i32,
    amino_acid: String,
    atoms: BTreeMap<String, [f64; 3]>,
}

fn design_protein_mpnn_structure(
    structure: MpnnStructure,
    request: &DesignRequest,
) -> io::Result<DesignResult> {
    request.validate()?;
    if !matches!(request.model, MpnnModel::ProteinMpnn | MpnnModel::AbMpnn) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the mmCIF-only sequence workflow supports ProteinMPNN and AbMPNN",
        ));
    }

    let python = find_executable(request.model.tool())?;
    let checkout = bundle_root(request.model.tool())?;
    let workspace = ToolWorkspace::new("mpnn")?;
    let input_path = workspace.path("input.jsonl");
    fs::write(&input_path, serde_json::to_string(&structure.value)? + "\n")?;
    let output_dir = workspace.create_dir("output")?;

    let mut command = Command::new(&python);
    configure_protein_mpnn_jsonl(&mut command, &checkout, &workspace, &structure, request)?;
    command
        .arg("--jsonl_path")
        .arg(&input_path)
        .arg("--out_folder")
        .arg(&output_dir)
        .arg("--seed")
        .arg(request.seed.to_string())
        .current_dir(&checkout);

    crate::external_tools::run_to_completion_logged(
        &mut command,
        request.model.label(),
        "sequence prediction",
    )?;

    let fasta_path = find_output_fasta(&output_dir)?;
    let fasta = fs::read_to_string(&fasta_path)?;
    let mut result = parse_design_fasta(&fasta);
    result.raw_fasta = fasta;
    Ok(result)
}

fn design_pdb_text(pdb: &str, request: &DesignRequest) -> io::Result<DesignResult> {
    request.validate()?;
    if !pdb.lines().any(|line| line.starts_with("ATOM")) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the selected structure contains no protein ATOM records",
        ));
    }

    let python = find_executable(request.model.tool())?;
    let checkout = bundle_root(request.model.tool())?;
    let workspace = ToolWorkspace::new("mpnn")?;
    let input_path = workspace.path("input.pdb");
    fs::write(&input_path, pdb)?;
    let output_dir = workspace.create_dir("output")?;

    let mut command = Command::new(&python);
    let ligand_model_type = request.model.ligand_model_type();
    match ligand_model_type {
        Some(model_type) => {
            configure_ligand_mpnn(&mut command, &checkout, model_type, request)?;
        }
        None => configure_protein_mpnn(&mut command, &checkout, &workspace, pdb, request)?,
    }
    // The original ProteinMPNN runner uses the complete --pdb_path value as its output FASTA
    // stem. An absolute Windows path therefore becomes an invalid filename containing `C:\`.
    // Run that checkout from the workspace and give it a relative input name. LigandMPNN does
    // not have this bug and still needs the checkout as its working directory.
    let workspace_root = workspace.path("");
    let (pdb_argument, current_dir) = mpnn_process_paths(
        ligand_model_type.is_none(),
        &input_path,
        &workspace_root,
        &checkout,
    );
    command
        .arg("--pdb_path")
        .arg(pdb_argument)
        .arg("--out_folder")
        .arg(&output_dir)
        .arg("--seed")
        .arg(request.seed.to_string())
        .current_dir(current_dir);

    crate::external_tools::run_to_completion_logged(
        &mut command,
        request.model.label(),
        "sequence prediction",
    )?;

    let fasta_path = find_output_fasta(&output_dir)?;
    let fasta = fs::read_to_string(&fasta_path)?;
    let mut result = parse_design_fasta(&fasta);
    result.raw_fasta = fasta;
    Ok(result)
}

fn mpnn_process_paths(
    original_protein_mpnn: bool,
    input_path: &Path,
    workspace: &Path,
    checkout: &Path,
) -> (PathBuf, PathBuf) {
    if original_protein_mpnn {
        (PathBuf::from("input.pdb"), workspace.to_owned())
    } else {
        (input_path.to_owned(), checkout.to_owned())
    }
}

fn configure_ligand_mpnn(
    command: &mut Command,
    checkout: &Path,
    model_type: &str,
    request: &DesignRequest,
) -> io::Result<()> {
    let runner = checkout.join("run.py");
    if !runner.is_file() {
        return Err(missing_checkout(&runner));
    }
    // Each model type has its own checkpoint flag and file; passing the wrong pair silently
    // produces a model that was never trained for the requested task.
    let (checkpoint_flag, checkpoint_file) = match model_type {
        "ligand_mpnn" => (
            "--checkpoint_ligand_mpnn",
            "model_params/ligandmpnn_v_32_010_25.pt",
        ),
        "soluble_mpnn" => (
            "--checkpoint_soluble_mpnn",
            "model_params/solublempnn_v_48_020.pt",
        ),
        _ => (
            "--checkpoint_protein_mpnn",
            "model_params/proteinmpnn_v_48_020.pt",
        ),
    };
    let checkpoint = checkout.join(checkpoint_file);
    if !checkpoint.is_file() {
        return Err(missing_weights(&checkpoint, Tool::LigandMpnn));
    }

    command
        .arg(&runner)
        .arg("--model_type")
        .arg(model_type)
        .arg(checkpoint_flag)
        .arg(&checkpoint)
        .arg("--batch_size")
        .arg("1")
        .arg("--number_of_batches")
        .arg(request.num_sequences.to_string())
        .arg("--temperature")
        .arg(request.temperature.to_string());

    if !request.chains_to_design.is_empty() {
        command
            .arg("--chains_to_design")
            .arg(request.chains_to_design.join(","));
    }
    if !request.fixed_residues.is_empty() {
        // Space-separated for LigandMPNN, comma-separated for ProteinMPNN: the two repositories
        // disagree, and mixing them up is silently accepted as "nothing is fixed".
        command
            .arg("--fixed_residues")
            .arg(request.fixed_residues.join(" "));
    }
    Ok(())
}

fn configure_protein_mpnn(
    command: &mut Command,
    checkout: &Path,
    workspace: &ToolWorkspace,
    pdb: &str,
    request: &DesignRequest,
) -> io::Result<()> {
    configure_protein_mpnn_base(command, checkout, request)?;

    let chains = if request.chains_to_design.is_empty() {
        pdb_chain_residues(pdb).into_keys().collect::<Vec<_>>()
    } else {
        request.chains_to_design.clone()
    };
    if !chains.is_empty() {
        command.arg("--pdb_path_chains").arg(chains.join(" "));
    }
    configure_protein_mpnn_constraints(command, workspace, pdb, &chains, request)?;
    Ok(())
}

fn configure_protein_mpnn_base(
    command: &mut Command,
    checkout: &Path,
    request: &DesignRequest,
) -> io::Result<()> {
    let runner = checkout.join("protein_mpnn_run.py");
    if !runner.is_file() {
        return Err(missing_checkout(&runner));
    }

    let weights_dir = match request.model {
        MpnnModel::AbMpnn => checkout.join("abmpnn_weights"),
        _ => checkout.join("vanilla_model_weights"),
    };
    // `protein_mpnn_run.py` builds the checkpoint path as
    // `f"{args.path_to_model_weights}{args.model_name}.pt"` — plain string concatenation, not a
    // path join — so the directory it is given must end in a separator or the file is never found.
    let mut weights_argument = weights_dir.to_string_lossy().into_owned();
    if !weights_argument.ends_with(['/', '\\']) {
        weights_argument.push(std::path::MAIN_SEPARATOR);
    }
    // The AbMPNN checkpoint is installed under the vanilla model's name so that this default
    // matches for both; see the install script.
    let model_name = request.checkpoint.name();
    let checkpoint = weights_dir.join(format!("{model_name}.pt"));
    if !checkpoint.is_file() {
        return Err(missing_weights(&checkpoint, Tool::ProteinMpnn));
    }

    command
        .arg(&runner)
        .arg("--path_to_model_weights")
        .arg(weights_argument)
        .arg("--model_name")
        .arg(model_name)
        .arg("--num_seq_per_target")
        .arg(request.num_sequences.to_string())
        .arg("--sampling_temp")
        .arg(request.temperature.to_string())
        .arg("--backbone_noise")
        .arg(request.backbone_noise.to_string())
        .arg("--omit_AAs")
        .arg(compact_letters(&request.omit_amino_acids))
        .arg("--batch_size")
        .arg("1");

    Ok(())
}

fn configure_protein_mpnn_jsonl(
    command: &mut Command,
    checkout: &Path,
    workspace: &ToolWorkspace,
    structure: &MpnnStructure,
    request: &DesignRequest,
) -> io::Result<()> {
    configure_protein_mpnn_base(command, checkout, request)?;
    let chains = if request.chains_to_design.is_empty() {
        structure.chains.clone()
    } else {
        request.chains_to_design.clone()
    };
    let fixed_chains = structure
        .chains
        .iter()
        .filter(|chain| !chains.contains(chain))
        .cloned()
        .collect::<Vec<_>>();
    write_named_jsonl(workspace, "chain_ids.jsonl", json!([chains, fixed_chains]))?;
    command
        .arg("--chain_id_jsonl")
        .arg(workspace.path("chain_ids.jsonl"));
    configure_protein_mpnn_constraints_from_residues(
        command,
        workspace,
        &structure.residues,
        &chains,
        request,
    )
}

fn configure_protein_mpnn_constraints(
    command: &mut Command,
    workspace: &ToolWorkspace,
    pdb: &str,
    chains: &[String],
    request: &DesignRequest,
) -> io::Result<()> {
    let residues = pdb_chain_residues(pdb);
    configure_protein_mpnn_constraints_from_residues(command, workspace, &residues, chains, request)
}

fn configure_protein_mpnn_constraints_from_residues(
    command: &mut Command,
    workspace: &ToolWorkspace,
    residues: &BTreeMap<String, Vec<i32>>,
    chains: &[String],
    request: &DesignRequest,
) -> io::Result<()> {
    let designed = parse_designed_residues(&request.designed_residues)?;
    let mut fixed: BTreeMap<String, BTreeSet<i32>> = BTreeMap::new();

    for residue in &request.fixed_residues {
        let (chain, number) = split_residue_key(residue, "fixed residue")?;
        fixed.entry(chain).or_default().insert(number);
    }
    if !designed.is_empty() {
        for chain in chains {
            let keep = designed.get(chain).cloned().unwrap_or_default();
            fixed.entry(chain.clone()).or_default().extend(
                residues
                    .get(chain)
                    .into_iter()
                    .flatten()
                    .filter(|number| !keep.contains(number)),
            );
        }
    }
    if !fixed.is_empty() {
        let value = fixed
            .into_iter()
            .map(|(chain, numbers)| (chain, numbers.into_iter().collect::<Vec<_>>()))
            .collect::<BTreeMap<_, _>>();
        write_named_jsonl(workspace, "fixed_positions.jsonl", json!(value))?;
        command
            .arg("--fixed_positions_jsonl")
            .arg(workspace.path("fixed_positions.jsonl"));
    }

    let bias = parse_bias_amino_acids(&request.bias_amino_acids)?;
    if !bias.is_empty() {
        let path = workspace.path("bias_aa.jsonl");
        fs::write(&path, serde_json::to_string(&bias)? + "\n")?;
        command.arg("--bias_AA_jsonl").arg(path);
    }

    let per_residue_bias = protein_mpnn_bias_matrix(
        parse_json_object(
            &request.bias_amino_acids_per_residue,
            "per-residue amino-acid bias",
        )?,
        &residues,
        chains,
    )?;
    if !per_residue_bias.is_null() {
        write_named_jsonl(workspace, "bias_by_res.jsonl", per_residue_bias)?;
        command
            .arg("--bias_by_res_jsonl")
            .arg(workspace.path("bias_by_res.jsonl"));
    }

    let per_residue_omit = protein_mpnn_omit_map(
        parse_json_object(
            &request.omit_amino_acids_per_residue,
            "per-residue omitted amino acids",
        )?,
        chains,
    )?;
    if !per_residue_omit.is_null() {
        write_named_jsonl(workspace, "omit_by_res.jsonl", per_residue_omit)?;
        command
            .arg("--omit_AA_jsonl")
            .arg(workspace.path("omit_by_res.jsonl"));
    }

    if request.homo_oligomer && chains.len() > 1 {
        let shortest = chains
            .iter()
            .filter_map(|chain| residues.get(chain).map(Vec::len))
            .min()
            .unwrap_or(0);
        let tied = (0..shortest)
            .map(|index| {
                chains
                    .iter()
                    .filter_map(|chain| {
                        residues
                            .get(chain)
                            .and_then(|numbers| numbers.get(index))
                            .map(|number| (chain.clone(), vec![*number]))
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        if !tied.is_empty() {
            write_named_jsonl(workspace, "tied_positions.jsonl", json!(tied))?;
            command
                .arg("--tied_positions_jsonl")
                .arg(workspace.path("tied_positions.jsonl"));
        }
    }
    Ok(())
}

fn write_named_jsonl(workspace: &ToolWorkspace, filename: &str, value: Value) -> io::Result<()> {
    let path = workspace.path(filename);
    fs::write(
        &path,
        serde_json::to_string(&json!({"input": value}))? + "\n",
    )
}

fn compact_letters(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != ',')
        .flat_map(char::to_uppercase)
        .collect()
}

fn parse_designed_residues(value: &str) -> io::Result<BTreeMap<String, BTreeSet<i32>>> {
    let mut result = BTreeMap::new();
    for line in value.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let normalized = line.replace(',', " ");
        let mut parts = normalized.split_whitespace();
        let chain = parts.next().unwrap_or_default().to_owned();
        let numbers = parts
            .map(|part| {
                part.parse::<i32>().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("designed residue '{part}' is not a residue number"),
                    )
                })
            })
            .collect::<io::Result<BTreeSet<_>>>()?;
        if chain.is_empty() || numbers.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "each designed-residues line must contain a chain and at least one residue number",
            ));
        }
        result.insert(chain, numbers);
    }
    Ok(result)
}

fn parse_bias_amino_acids(value: &str) -> io::Result<BTreeMap<String, f64>> {
    let mut result = BTreeMap::new();
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let (letter, amount) = part.split_once(':').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("amino-acid bias '{part}' should look like W:3.0"),
            )
        })?;
        let letter = letter.trim().to_ascii_uppercase();
        if letter.len() != 1 || !letter.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("amino-acid bias '{part}' must name one amino-acid letter"),
            ));
        }
        let amount = amount.trim().parse::<f64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("amino-acid bias '{part}' has a non-numeric value"),
            )
        })?;
        result.insert(letter, amount);
    }
    Ok(result)
}

fn parse_json_object(value: &str, label: &str) -> io::Result<Map<String, Value>> {
    if value.trim().is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_str::<Value>(value)
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {label}: {error}"),
            )
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{label} must be a JSON object"),
            )
        })
}

fn split_residue_key(value: &str, label: &str) -> io::Result<(String, i32)> {
    let split = value
        .char_indices()
        .find(|(_, character)| character.is_ascii_digit() || *character == '-')
        .map(|(index, _)| index)
        .unwrap_or(value.len());
    let (chain, number) = value.split_at(split);
    if chain.is_empty() || !chain.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} '{value}' should look like A12"),
        ));
    }
    let number = number.parse::<i32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} '{value}' should look like A12"),
        )
    })?;
    Ok((chain.to_owned(), number))
}

fn pdb_chain_residues(pdb: &str) -> BTreeMap<String, Vec<i32>> {
    let mut result: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    for line in pdb.lines().filter(|line| line.starts_with("ATOM")) {
        if line.len() < 26 {
            continue;
        }
        let chain = line[21..22].trim();
        let Ok(number) = line[22..26].trim().parse::<i32>() else {
            continue;
        };
        let numbers = result.entry(chain.to_owned()).or_default();
        if numbers.last() != Some(&number) {
            numbers.push(number);
        }
    }
    result
}

fn protein_mpnn_bias_matrix(
    sparse: Map<String, Value>,
    residues: &BTreeMap<String, Vec<i32>>,
    chains: &[String],
) -> io::Result<Value> {
    if sparse.is_empty() {
        return Ok(Value::Null);
    }
    const ALPHABET: &str = "ACDEFGHIKLMNPQRSTVWYX";
    let mut matrices = chains
        .iter()
        .map(|chain| {
            let rows = residues.get(chain).map_or(0, Vec::len);
            (chain.clone(), vec![vec![0.0_f64; ALPHABET.len()]; rows])
        })
        .collect::<BTreeMap<_, _>>();
    for (key, biases) in sparse {
        let (chain, number) = split_residue_key(&key, "per-residue bias key")?;
        let row = residues
            .get(&chain)
            .and_then(|numbers| numbers.iter().position(|candidate| *candidate == number))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("residue {key} was not found in the structure"),
                )
            })?;
        let biases = biases.as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("per-residue bias for {key} must be a JSON object"),
            )
        })?;
        let matrix = matrices.get_mut(&chain).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("chain {chain} is not selected for design"),
            )
        })?;
        for (letter, amount) in biases {
            let letter = letter.to_ascii_uppercase();
            let column = ALPHABET.find(&letter).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported amino acid '{letter}' in bias for {key}"),
                )
            })?;
            matrix[row][column] = amount.as_f64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("bias for {key}/{letter} must be numeric"),
                )
            })?;
        }
    }
    Ok(json!(matrices))
}

fn protein_mpnn_omit_map(sparse: Map<String, Value>, chains: &[String]) -> io::Result<Value> {
    if sparse.is_empty() {
        return Ok(Value::Null);
    }
    let mut grouped = chains
        .iter()
        .map(|chain| (chain.clone(), Vec::<Value>::new()))
        .collect::<BTreeMap<_, _>>();
    for (key, letters) in sparse {
        let (chain, number) = split_residue_key(&key, "per-residue omission key")?;
        let letters = letters.as_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("omitted amino acids for {key} must be a string"),
            )
        })?;
        let letters = compact_letters(letters);
        if !letters.chars().all(|letter| letter.is_ascii_alphabetic()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("omitted amino acids for {key} must be letters"),
            ));
        }
        grouped
            .get_mut(&chain)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("chain {chain} is not selected for design"),
                )
            })?
            .push(json!([[number], letters]));
    }
    Ok(json!(grouped))
}

fn peptide_to_mpnn_structure(mol: &MoleculePeptide) -> io::Result<MpnnStructure> {
    let mut chains = Vec::new();
    let mut used = BTreeSet::new();
    for (chain_index, chain) in mol.chains.iter().enumerate() {
        let chain_id = unique_mpnn_chain_id(&chain.id, chain_index, &mut used)?;
        let mut residues = Vec::new();
        for &residue_index in &chain.residues {
            let Some(residue) = mol.residues.get(residue_index) else {
                continue;
            };
            let ResidueType::AminoAcid(amino_acid) = residue.res_type else {
                continue;
            };
            let mut atoms = BTreeMap::new();
            for &atom_index in &residue.atoms {
                let Some(atom) = mol.common.atoms.get(atom_index) else {
                    continue;
                };
                let Some(name) = atom
                    .type_in_res
                    .as_ref()
                    .map(ToString::to_string)
                    .or_else(|| atom.type_in_res_general.clone())
                else {
                    continue;
                };
                if matches!(name.to_ascii_uppercase().as_str(), "N" | "CA" | "C" | "O") {
                    atoms.insert(
                        name.to_ascii_uppercase(),
                        [atom.posit.x, atom.posit.y, atom.posit.z],
                    );
                }
            }
            residues.push(MpnnResidue {
                number: residue.serial_number as i32,
                amino_acid: amino_acid.to_str(na_seq::AaIdent::OneLetter),
                atoms,
            });
        }
        if !residues.is_empty() {
            chains.push((chain_id, residues));
        }
    }
    build_mpnn_structure(chains)
}

fn mmcif_to_mpnn_structure(cif: &MmCif) -> io::Result<MpnnStructure> {
    let atoms = cif
        .atoms
        .iter()
        .map(|atom| (atom.serial_number, atom))
        .collect::<BTreeMap<_, _>>();
    let mut chains = Vec::new();
    let mut used = BTreeSet::new();
    for (chain_index, chain) in cif.chains.iter().enumerate() {
        let chain_id = unique_mpnn_chain_id(&chain.id, chain_index, &mut used)?;
        let mut chain_residues = Vec::new();
        for residue in cif.residues.iter().filter(|residue| {
            chain.residue_sns.contains(&residue.serial_number)
                && residue
                    .atom_sns
                    .iter()
                    .any(|serial| chain.atom_sns.contains(serial))
        }) {
            let ResidueType::AminoAcid(amino_acid) = residue.res_type else {
                continue;
            };
            let mut backbone = BTreeMap::new();
            for atom_sn in &residue.atom_sns {
                let Some(atom) = atoms.get(atom_sn) else {
                    continue;
                };
                let Some(name) = atom
                    .type_in_res
                    .as_ref()
                    .map(ToString::to_string)
                    .or_else(|| atom.type_in_res_general.clone())
                else {
                    continue;
                };
                if matches!(name.to_ascii_uppercase().as_str(), "N" | "CA" | "C" | "O") {
                    backbone.insert(
                        name.to_ascii_uppercase(),
                        [atom.posit.x, atom.posit.y, atom.posit.z],
                    );
                }
            }
            chain_residues.push(MpnnResidue {
                number: residue.serial_number as i32,
                amino_acid: amino_acid.to_str(na_seq::AaIdent::OneLetter),
                atoms: backbone,
            });
        }
        if !chain_residues.is_empty() {
            chains.push((chain_id, chain_residues));
        }
    }
    build_mpnn_structure(chains)
}

fn unique_mpnn_chain_id(
    source_id: &str,
    chain_index: usize,
    used: &mut BTreeSet<String>,
) -> io::Result<String> {
    let preferred = chain_letter(source_id, chain_index).to_string();
    if used.insert(preferred.clone()) {
        return Ok(preferred);
    }
    for candidate in ('A'..='Z').map(|letter| letter.to_string()) {
        if used.insert(candidate.clone()) {
            return Ok(candidate);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "ProteinMPNN supports at most 26 uniquely addressable chains",
    ))
}

fn build_mpnn_structure(chains: Vec<(String, Vec<MpnnResidue>)>) -> io::Result<MpnnStructure> {
    if chains.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the mmCIF structure contains no protein chains",
        ));
    }
    let mut value = Map::new();
    value.insert("name".to_owned(), json!("input"));
    value.insert("num_of_chains".to_owned(), json!(chains.len()));
    let mut full_sequence = String::new();
    let mut residue_numbers = BTreeMap::new();
    let mut chain_ids = Vec::new();

    for (chain_id, residues) in chains {
        let mut sequence = String::new();
        let mut coords = Map::new();
        for atom_name in ["N", "CA", "C", "O"] {
            let mut atom_coords = Vec::with_capacity(residues.len());
            for residue in &residues {
                let coordinate = residue.atoms.get(atom_name).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "mmCIF chain {chain_id} residue {} is missing backbone atom {atom_name}",
                            residue.number
                        ),
                    )
                })?;
                atom_coords.push(*coordinate);
            }
            coords.insert(format!("{atom_name}_chain_{chain_id}"), json!(atom_coords));
        }
        for residue in &residues {
            sequence.push_str(&residue.amino_acid);
        }
        full_sequence.push_str(&sequence);
        residue_numbers.insert(
            chain_id.clone(),
            residues.iter().map(|residue| residue.number).collect(),
        );
        value.insert(format!("seq_chain_{chain_id}"), json!(sequence));
        value.insert(format!("coords_chain_{chain_id}"), Value::Object(coords));
        chain_ids.push(chain_id);
    }
    value.insert("seq".to_owned(), json!(full_sequence));
    Ok(MpnnStructure {
        value: Value::Object(value),
        residues: residue_numbers,
        chains: chain_ids,
    })
}

fn missing_checkout(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "{} was not found; the checkout is incomplete",
            path.display()
        ),
    )
}

fn missing_weights(path: &Path, tool: Tool) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "model weights {} were not downloaded. Re-run: {}",
            path.display(),
            tool.spec().install_command()
        ),
    )
}

/// Both entry points write `<out_folder>/seqs/<input stem>.fa`.
fn find_output_fasta(output_dir: &Path) -> io::Result<PathBuf> {
    let seqs = output_dir.join("seqs");
    let directory = if seqs.is_dir() {
        seqs
    } else {
        output_dir.to_path_buf()
    };

    fs::read_dir(&directory)?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("fa"))
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "the design run completed but wrote no FASTA under {}",
                    directory.display()
                ),
            )
        })
}

/// Parse the FASTA both repositories write.
///
/// The first record is the input, with a header naming the source and the chains; every record
/// after it is a design, with `T=`, `sample=`, `score=`, and `seq_recovery=` fields. Fields are
/// read by name rather than position, since the exact set varies between the two repositories and
/// between their versions.
fn parse_design_fasta(fasta: &str) -> DesignResult {
    let mut result = DesignResult::default();
    let mut header: Option<String> = None;
    let mut sequence = String::new();

    let flush = |header: &Option<String>, sequence: &mut String, result: &mut DesignResult| {
        let sequence_text = sequence.trim().to_owned();
        sequence.clear();
        if sequence_text.is_empty() {
            return;
        }
        let Some(header) = header else { return };
        let fields = header_fields(header);

        // Both repositories write the native/input sequence first. Original ProteinMPNN includes
        // a score on that record, while LigandMPNN does not, so record order is the reliable marker.
        if result.input_sequence.is_none() {
            result.input_sequence = Some(sequence_text);
            return;
        }

        let field = |name: &str| -> Option<f32> {
            fields
                .iter()
                .find(|(key, _)| key == name)
                .and_then(|(_, value)| value.parse().ok())
        };
        result.designs.push(DesignedSequence {
            sequence: sequence_text,
            score: field("score").or_else(|| field("global_score")),
            sequence_recovery: field("seq_recovery"),
            temperature: field("T").or_else(|| field("temperature")),
        });
    };

    for line in fasta.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            flush(&header, &mut sequence, &mut result);
            header = Some(rest.to_owned());
        } else {
            sequence.push_str(line.trim());
        }
    }
    flush(&header, &mut sequence, &mut result);

    // Best first: the score is a mean negative log likelihood, so lower is better. Sequences
    // without a score sort last rather than being dropped.
    result.designs.sort_by(|a, b| match (a.score, b.score) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    result
}

/// Split a FASTA header into its `key=value` pairs, ignoring the leading identifier.
fn header_fields(header: &str) -> Vec<(String, String)> {
    header
        .split(',')
        .filter_map(|part| {
            let (key, value) = part.split_once('=')?;
            Some((key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

/// The chain identifiers a design request can name, as this peptide presents them.
pub fn designable_chains(mol: &MoleculePeptide) -> Vec<String> {
    mol.chains
        .iter()
        .enumerate()
        .map(|(index, chain)| chain_letter(&chain.id, index).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_FASTA: &str = "\
>input, T=0.1, seed=37, num_res=8, num_ligand_res=0
AAAAAAAA
>input, id=1, T=0.1, seed=37, overall_confidence=0.55, score=1.2345, seq_recovery=0.4000
EVQLVESG
>input, id=2, T=0.1, seed=37, overall_confidence=0.61, score=0.9876, seq_recovery=0.5000
QVQLQESG
";

    #[test]
    fn parses_designs_and_ranks_them_by_score() {
        let result = parse_design_fasta(SAMPLE_FASTA);

        assert_eq!(result.input_sequence.as_deref(), Some("AAAAAAAA"));
        assert_eq!(result.designs.len(), 2);
        // Lower score is better, so the second record sorts first.
        assert_eq!(result.designs[0].sequence, "QVQLQESG");
        assert_eq!(result.designs[0].score, Some(0.9876));
        assert_eq!(result.designs[0].sequence_recovery, Some(0.5));
        assert_eq!(result.designs[0].temperature, Some(0.1));
        assert_eq!(result.designs[1].sequence, "EVQLVESG");
    }

    #[test]
    fn splits_multi_chain_designs() {
        let result = parse_design_fasta(
            ">input, T=0.1\nAAAA/BBBB\n>input, id=1, T=0.1, score=1.0\nEVQL/QSAL\n",
        );
        assert_eq!(result.designs[0].chains(), ["EVQL", "QSAL"]);
    }

    #[test]
    fn rejects_malformed_fixed_residues() {
        let mut request = DesignRequest::default();
        request.fixed_residues = vec!["H97".to_owned(), "L1".to_owned()];
        assert!(request.validate().is_ok());

        for bad in ["97", "H", "HH97", "H97a", ""] {
            request.fixed_residues = vec![bad.to_owned()];
            assert!(
                request.validate().is_err(),
                "'{bad}' should be rejected as a fixed residue"
            );
        }
    }

    #[test]
    fn rejects_out_of_range_sampling_parameters() {
        let mut request = DesignRequest::default();
        request.num_sequences = 0;
        assert!(request.validate().is_err());

        request = DesignRequest::default();
        request.temperature = 0.0;
        assert!(request.validate().is_err());

        request.temperature = 5.0;
        assert!(request.validate().is_err());
    }

    #[test]
    fn maps_each_model_to_its_installation() {
        assert_eq!(MpnnModel::LigandMpnn.tool(), Tool::LigandMpnn);
        assert_eq!(MpnnModel::SolubleMpnn.tool(), Tool::LigandMpnn);
        // AbMPNN is ProteinMPNN's network, so it comes from that checkout.
        assert_eq!(MpnnModel::AbMpnn.tool(), Tool::ProteinMpnn);
        assert_eq!(MpnnModel::ProteinMpnn.tool(), Tool::ProteinMpnn);
        // Only the ligand model wants non-protein atoms written into the input.
        assert!(MpnnModel::LigandMpnn.wants_hetero_context());
        assert!(!MpnnModel::ProteinMpnnViaLigand.wants_hetero_context());
    }

    #[test]
    fn parses_protein_mpnn_customization_fields() {
        let designed = parse_designed_residues("A 1 3 5\nB, 10, 12").unwrap();
        assert_eq!(designed["A"], BTreeSet::from([1, 3, 5]));
        assert_eq!(designed["B"], BTreeSet::from([10, 12]));

        let bias = parse_bias_amino_acids("W:3.0, A:-1.5").unwrap();
        assert_eq!(bias["W"], 3.0);
        assert_eq!(bias["A"], -1.5);
    }

    #[test]
    fn finds_chains_and_residue_numbers_in_pdb_input() {
        let pdb = concat!(
            "ATOM      1  N   ALA A   1      11.000  12.000  13.000\n",
            "ATOM      2  CA  ALA A   1      12.000  12.000  13.000\n",
            "ATOM      3  N   GLY A   4      13.000  12.000  13.000\n",
            "ATOM      4  N   SER B  10      14.000  12.000  13.000\n",
        );
        let residues = pdb_chain_residues(pdb);
        assert_eq!(residues["A"], [1, 4]);
        assert_eq!(residues["B"], [10]);
    }

    #[test]
    fn rejects_invalid_advanced_json() {
        let mut request = DesignRequest::default();
        request.bias_amino_acids_per_residue = "[]".to_owned();
        assert!(request.validate().is_err());

        request.bias_amino_acids_per_residue = r#"{"A1":{"G":1.0}}"#.to_owned();
        request.omit_amino_acids_per_residue = r#"{"A1":"CP"}"#.to_owned();
        assert!(request.validate().is_ok());
    }

    #[test]
    fn original_protein_mpnn_uses_a_relative_pdb_path() {
        let input = Path::new(r"C:\Temp\molchanica-mpnn\input.pdb");
        let workspace = Path::new(r"C:\Temp\molchanica-mpnn");
        let checkout = Path::new(r"C:\Tools\ProteinMPNN");

        let (argument, directory) = mpnn_process_paths(true, input, workspace, checkout);
        assert_eq!(argument, Path::new("input.pdb"));
        assert_eq!(directory, workspace);

        let (argument, directory) = mpnn_process_paths(false, input, workspace, checkout);
        assert_eq!(argument, input);
        assert_eq!(directory, checkout);
    }

    #[test]
    #[ignore = "requires the locally installed ProteinMPNN weights"]
    fn smoke_tests_mmcif_jsonl_with_installed_protein_mpnn() {
        let mut request = DesignRequest::default();
        request.model = MpnnModel::ProteinMpnn;
        request.num_sequences = 1;
        request.fixed_residues = vec!["A1".to_owned()];
        request.bias_amino_acids = "W:0.1".to_owned();
        request.bias_amino_acids_per_residue = r#"{"A2":{"G":0.1}}"#.to_owned();
        request.omit_amino_acids_per_residue = r#"{"A3":"CP"}"#.to_owned();
        let result = design_file(Path::new("molecules/1ubq.cif"), &request).unwrap();
        assert_eq!(result.designs.len(), 1);
    }
}
