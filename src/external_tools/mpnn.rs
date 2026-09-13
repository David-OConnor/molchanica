//! Inverse folding with the original ProteinMPNN checkout, and its antibody-tuned AbMPNN weights.
//!
//! [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
//!
//! This answers the question Molchanica otherwise cannot: given a backbone, what sequences would
//! fold into it? It backs the Protein design window's sequence tab, which pairs AbMPNN with the
//! CDR annotation beside it and needs designs as typed values to rank and display.
//!
//! The general-purpose MPNN runs — LigandMPNN, and ProteinMPNN with every option `bio_tools`
//! exposes — go through the shared tool window and [`shared_adapter`](super::shared_adapter)
//! instead, and are not duplicated here.
//!
//! # Why AbMPNN is a model rather than a tool
//!
//! AbMPNN is ProteinMPNN's network with antibody-finetuned weights (Frey et al., ICML 2023 CompBio
//! workshop, CC BY 4.0). It is not a separate installation: the weights sit beside the vanilla ones
//! in the same checkout, which is why it is an [`MpnnModel`] variant rather than a registry entry.
//!
//! # Platform note
//!
//! Plain PyTorch against a checkout — no compiled CUDA kernels, no conda — so it runs on Windows
//! and Linux alike, on CPU or GPU.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use mol_defs::molecules::peptide::MoleculePeptide;
use serde_json::{Map, Value, json};

use crate::external_tools::{
    Tool, ToolWorkspace, bundle_root, find_executable,
    pdb_write::{PdbWriteOptions, chain_letter, peptide_to_pdb},
    run_tool,
};

/// The input file name, relative to the workspace the runner is started in.
///
/// `protein_mpnn_run.py` uses the complete `--pdb_path` value as its output FASTA stem, so an
/// absolute Windows path becomes an invalid file name containing `C:\`. The run is therefore
/// started in the workspace and given this relative name.
const INPUT_PDB: &str = "input.pdb";

/// The residue alphabet ProteinMPNN's bias matrices are indexed by.
const ALPHABET: &str = "ACDEFGHIKLMNPQRSTVWYX";

// ---------------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------------

/// Which weights to run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MpnnModel {
    /// The original ProteinMPNN weights.
    #[default]
    ProteinMpnn,
    /// The ProteinMPNN network with the antibody-finetuned AbMPNN checkpoint.
    AbMpnn,
}

impl MpnnModel {
    /// Which registry entry — and therefore which checkout and virtual environment — this uses.
    pub fn tool(self) -> Tool {
        Tool::ProteinMpnn
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ProteinMpnn => "ProteinMPNN (original)",
            Self::AbMpnn => "AbMPNN (antibody-tuned)",
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            Self::ProteinMpnn => "The original network and weights.",
            Self::AbMpnn => {
                "ProteinMPNN finetuned on antibody structures. Best for CDR and framework design; \
                 pairs with the antibody tools."
            }
        }
    }

    /// The checkout directory holding this model's weights.
    fn weights_dir(self) -> &'static str {
        match self {
            Self::ProteinMpnn => "vanilla_model_weights",
            Self::AbMpnn => "abmpnn_weights",
        }
    }
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
    pub fn name(self) -> &'static str {
        match self {
            Self::Noise002 => "v_48_002",
            Self::Noise010 => "v_48_010",
            Self::Noise020 => "v_48_020",
            Self::Noise030 => "v_48_030",
        }
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
    /// give diversity at the cost of predicted stability. 0.1 is the repository's default.
    pub temperature: f32,
    pub seed: u64,
    /// Which noise level's checkpoint to use.
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
            return Err(invalid_input(
                "number of sequences must be between 1 and 1000".to_owned(),
            ));
        }
        if !(0.0001..=2.0).contains(&self.temperature) {
            return Err(invalid_input(
                "sampling temperature must be between 0.0001 and 2.0".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.backbone_noise) {
            return Err(invalid_input(
                "backbone noise must be between 0 and 1".to_owned(),
            ));
        }

        let omit = compact_letters(&self.omit_amino_acids);
        if !omit.chars().all(|letter| letter.is_ascii_alphabetic()) {
            return Err(invalid_input(
                "omitted amino acids must contain only one-letter amino-acid codes".to_owned(),
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
            // The runner parses these as a chain letter followed by a residue number.
            let mut chars = residue.chars();
            let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                && chars.clone().count() > 0
                && chars.all(|c| c.is_ascii_digit());
            if !valid {
                return Err(invalid_input(format!(
                    "fixed residue '{residue}' should be a chain letter followed by a residue \
                     number, e.g. H97"
                )));
            }
        }
        Ok(())
    }
}

/// The chain identifiers a design request can name, as this peptide presents them.
pub fn designable_chains(mol: &MoleculePeptide) -> Vec<String> {
    mol.chains
        .iter()
        .enumerate()
        .map(|(index, chain)| chain_letter(&chain.id, index).to_string())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------------------------

/// One generated sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignedSequence {
    /// The designed sequence, chains joined by `/` as the runner writes them.
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

/// What a run produced.
#[derive(Clone, Debug, Default)]
pub struct DesignResult {
    /// The input sequence as the model saw it, from the first FASTA record.
    pub input_sequence: Option<String>,
    /// Generated sequences, best (lowest score) first.
    pub designs: Vec<DesignedSequence>,
    pub raw_fasta: String,
}

// ---------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------

/// Run a design against a loaded peptide.
///
/// Blocking, and slow enough to want a worker thread: the process start and Torch import dominate
/// for small designs.
pub fn design(mol: &MoleculePeptide, request: &DesignRequest) -> io::Result<DesignResult> {
    request.validate()?;

    // Chain filtering happens in the model, not the file: it needs the whole structure as context
    // even when only part of it is redesigned.
    let pdb = peptide_to_pdb(mol, &PdbWriteOptions::default())?;
    if !pdb.lines().any(|line| line.starts_with("ATOM")) {
        return Err(invalid_input(
            "the selected structure contains no protein ATOM records".to_owned(),
        ));
    }

    let tool = request.model.tool();
    let python = find_executable(tool)?;
    let checkout = bundle_root(tool)?;
    let workspace = ToolWorkspace::new(tool)?;

    workspace.write(INPUT_PDB, &pdb)?;
    let output_dir = workspace.create_dir("output")?;

    let mut command = Command::new(&python);
    add_runner_args(&mut command, &checkout, request)?;
    add_constraint_args(&mut command, &workspace, &pdb, request)?;
    command
        .arg("--pdb_path")
        .arg(INPUT_PDB)
        .arg("--out_folder")
        .arg(&output_dir)
        .arg("--seed")
        .arg(request.seed.to_string())
        .current_dir(workspace.root());

    let workflow = format!("sequence design ({})", request.model.label());
    run_tool(&mut command, tool, &workflow, None)?;

    let fasta = workspace.read_output(&find_output_fasta(&output_dir)?)?;
    let mut result = parse_design_fasta(&fasta);
    result.raw_fasta = fasta;
    Ok(result)
}

/// The runner script, its weights, and the sampling settings.
fn add_runner_args(
    command: &mut Command,
    checkout: &Path,
    request: &DesignRequest,
) -> io::Result<()> {
    let runner = checkout.join("protein_mpnn_run.py");
    if !runner.is_file() {
        return Err(missing_checkout(&runner));
    }

    let weights_dir = checkout.join(request.model.weights_dir());
    // `protein_mpnn_run.py` builds the checkpoint path as
    // `f"{args.path_to_model_weights}{args.model_name}.pt"` — plain string concatenation, not a
    // path join — so the directory it is given must end in a separator or the file is never found.
    let mut weights_argument = weights_dir.to_string_lossy().into_owned();
    if !weights_argument.ends_with(['/', '\\']) {
        weights_argument.push(std::path::MAIN_SEPARATOR);
    }

    // The AbMPNN checkpoint is installed under the vanilla model's name so that this default
    // matches for both; see the install recipe.
    let model_name = request.checkpoint.name();
    let checkpoint = weights_dir.join(format!("{model_name}.pt"));
    if !checkpoint.is_file() {
        return Err(missing_weights(&checkpoint, request.model.tool()));
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

/// Which chains to design, and every per-position constraint, each written as the JSONL file the
/// runner reads it from.
fn add_constraint_args(
    command: &mut Command,
    workspace: &ToolWorkspace,
    pdb: &str,
    request: &DesignRequest,
) -> io::Result<()> {
    let residues = pdb_chain_residues(pdb);
    let chains = if request.chains_to_design.is_empty() {
        residues.keys().cloned().collect::<Vec<_>>()
    } else {
        request.chains_to_design.clone()
    };
    if !chains.is_empty() {
        command.arg("--pdb_path_chains").arg(chains.join(" "));
    }

    let fixed = fixed_positions(&residues, &chains, request)?;
    if !fixed.is_empty() {
        let path = write_named_jsonl(workspace, "fixed_positions.jsonl", json!(fixed))?;
        command.arg("--fixed_positions_jsonl").arg(path);
    }

    let bias = parse_bias_amino_acids(&request.bias_amino_acids)?;
    if !bias.is_empty() {
        let path = workspace.write("bias_aa.jsonl", serde_json::to_string(&bias)? + "\n")?;
        command.arg("--bias_AA_jsonl").arg(path);
    }

    let per_residue_bias = bias_matrix(
        parse_json_object(
            &request.bias_amino_acids_per_residue,
            "per-residue amino-acid bias",
        )?,
        &residues,
        &chains,
    )?;
    if !per_residue_bias.is_null() {
        let path = write_named_jsonl(workspace, "bias_by_res.jsonl", per_residue_bias)?;
        command.arg("--bias_by_res_jsonl").arg(path);
    }

    let per_residue_omit = omit_map(
        parse_json_object(
            &request.omit_amino_acids_per_residue,
            "per-residue omitted amino acids",
        )?,
        &chains,
    )?;
    if !per_residue_omit.is_null() {
        let path = write_named_jsonl(workspace, "omit_by_res.jsonl", per_residue_omit)?;
        command.arg("--omit_AA_jsonl").arg(path);
    }

    if request.homo_oligomer && chains.len() > 1 {
        let tied = tied_positions(&residues, &chains);
        if !tied.is_empty() {
            let path = write_named_jsonl(workspace, "tied_positions.jsonl", json!(tied))?;
            command.arg("--tied_positions_jsonl").arg(path);
        }
    }
    Ok(())
}

/// Explicitly fixed residues, plus everything outside the design-only lists where there are any.
fn fixed_positions(
    residues: &BTreeMap<String, Vec<i32>>,
    chains: &[String],
    request: &DesignRequest,
) -> io::Result<BTreeMap<String, Vec<i32>>> {
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

    Ok(fixed
        .into_iter()
        .map(|(chain, numbers)| (chain, numbers.into_iter().collect()))
        .collect())
}

/// Tie the n-th residue of every chain together, up to the shortest chain.
fn tied_positions(
    residues: &BTreeMap<String, Vec<i32>>,
    chains: &[String],
) -> Vec<BTreeMap<String, Vec<i32>>> {
    let shortest = chains
        .iter()
        .filter_map(|chain| residues.get(chain).map(Vec::len))
        .min()
        .unwrap_or(0);

    (0..shortest)
        .map(|index| {
            chains
                .iter()
                .filter_map(|chain| {
                    residues
                        .get(chain)
                        .and_then(|numbers| numbers.get(index))
                        .map(|number| (chain.clone(), vec![*number]))
                })
                .collect()
        })
        .collect()
}

/// Write `{"input": value}` as one JSONL line: the runner keys every constraint by input name.
fn write_named_jsonl(
    workspace: &ToolWorkspace,
    filename: &str,
    value: Value,
) -> io::Result<PathBuf> {
    workspace.write(
        filename,
        serde_json::to_string(&json!({"input": value}))? + "\n",
    )
}

fn bias_matrix(
    sparse: Map<String, Value>,
    residues: &BTreeMap<String, Vec<i32>>,
    chains: &[String],
) -> io::Result<Value> {
    if sparse.is_empty() {
        return Ok(Value::Null);
    }

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
                invalid_input(format!("residue {key} was not found in the structure"))
            })?;
        let biases = biases.as_object().ok_or_else(|| {
            invalid_input(format!("per-residue bias for {key} must be a JSON object"))
        })?;
        let matrix = matrices
            .get_mut(&chain)
            .ok_or_else(|| invalid_input(format!("chain {chain} is not selected for design")))?;

        for (letter, amount) in biases {
            let letter = letter.to_ascii_uppercase();
            let column = ALPHABET.find(&letter).ok_or_else(|| {
                invalid_input(format!(
                    "unsupported amino acid '{letter}' in bias for {key}"
                ))
            })?;
            matrix[row][column] = amount
                .as_f64()
                .ok_or_else(|| invalid_input(format!("bias for {key}/{letter} must be numeric")))?;
        }
    }
    Ok(json!(matrices))
}

fn omit_map(sparse: Map<String, Value>, chains: &[String]) -> io::Result<Value> {
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
            invalid_input(format!("omitted amino acids for {key} must be a string"))
        })?;
        let letters = compact_letters(letters);
        if !letters.chars().all(|letter| letter.is_ascii_alphabetic()) {
            return Err(invalid_input(format!(
                "omitted amino acids for {key} must be letters"
            )));
        }

        grouped
            .get_mut(&chain)
            .ok_or_else(|| invalid_input(format!("chain {chain} is not selected for design")))?
            .push(json!([[number], letters]));
    }
    Ok(json!(grouped))
}

// ---------------------------------------------------------------------------------------------
// Input parsing
// ---------------------------------------------------------------------------------------------

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
                    invalid_input(format!("designed residue '{part}' is not a residue number"))
                })
            })
            .collect::<io::Result<BTreeSet<_>>>()?;

        if chain.is_empty() || numbers.is_empty() {
            return Err(invalid_input(
                "each designed-residues line must contain a chain and at least one residue number"
                    .to_owned(),
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
            invalid_input(format!("amino-acid bias '{part}' should look like W:3.0"))
        })?;

        let letter = letter.trim().to_ascii_uppercase();
        if letter.len() != 1 || !letter.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(invalid_input(format!(
                "amino-acid bias '{part}' must name one amino-acid letter"
            )));
        }

        let amount = amount.trim().parse::<f64>().map_err(|_| {
            invalid_input(format!("amino-acid bias '{part}' has a non-numeric value"))
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
        .map_err(|error| invalid_input(format!("invalid {label}: {error}")))?
        .as_object()
        .cloned()
        .ok_or_else(|| invalid_input(format!("{label} must be a JSON object")))
}

fn split_residue_key(value: &str, label: &str) -> io::Result<(String, i32)> {
    let split = value
        .char_indices()
        .find(|(_, character)| character.is_ascii_digit() || *character == '-')
        .map(|(index, _)| index)
        .unwrap_or(value.len());
    let (chain, number) = value.split_at(split);

    let malformed = || invalid_input(format!("{label} '{value}' should look like A12"));
    if chain.is_empty() || !chain.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(malformed());
    }
    let number = number.parse::<i32>().map_err(|_| malformed())?;

    Ok((chain.to_owned(), number))
}

/// Residue numbers per chain, in file order, from the `ATOM` records of a PDB.
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

// ---------------------------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------------------------

/// The runner writes `<out_folder>/seqs/<input stem>.fa`.
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

/// Parse the FASTA the runner writes.
///
/// The first record is the input, with a header naming the source and the chains; every record
/// after it is a design, with `T=`, `sample=`, `score=`, and `seq_recovery=` fields. Fields are
/// read by name rather than position, since the exact set varies between versions.
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

        // The native/input sequence is written first. It carries a score too, so record order,
        // not the presence of a score, is what marks it.
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

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
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
