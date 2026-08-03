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
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    external_tools::{
        Tool, ToolWorkspace, bundle_root, find_executable,
        pdb_write::{PdbWriteOptions, chain_letter, peptide_to_pdb},
        run_to_completion,
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
        }
    }
}

impl DesignRequest {
    /// Reject a request before a process is started, so a typo in a fixed-residue list surfaces
    /// as a message beside the field rather than as a model that quietly fixed nothing.
    pub fn validate(&self) -> io::Result<()> {
        if self.num_sequences == 0 || self.num_sequences > 512 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "number of sequences must be between 1 and 512",
            ));
        }
        if !(0.0001..=2.0).contains(&self.temperature) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sampling temperature must be between 0.0001 and 2.0",
            ));
        }
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
    request.validate()?;

    let python = find_executable(request.model.tool())?;
    let checkout = bundle_root(request.model.tool())?;
    let workspace = ToolWorkspace::new("mpnn")?;

    let options = PdbWriteOptions {
        // Chain filtering happens in the model, not the file: it needs the whole structure as
        // context even when only part of it is redesigned.
        chains: Vec::new(),
        include_hetero: request.model.wants_hetero_context(),
        include_hydrogen: false,
        include_water: false,
    };
    let input_path = workspace.path("input.pdb");
    fs::write(&input_path, peptide_to_pdb(mol, &options)?)?;
    let output_dir = workspace.create_dir("output")?;

    let mut command = Command::new(&python);
    match request.model.ligand_model_type() {
        Some(model_type) => {
            configure_ligand_mpnn(&mut command, &checkout, model_type, request)?;
        }
        None => configure_protein_mpnn(&mut command, &checkout, request)?,
    }
    command
        .arg("--pdb_path")
        .arg(&input_path)
        .arg("--out_folder")
        .arg(&output_dir)
        .arg("--seed")
        .arg(request.seed.to_string())
        // Run from the checkout: both entry points resolve auxiliary data relative to the working
        // directory rather than to the script.
        .current_dir(&checkout);

    run_to_completion(&mut command, request.model.label())?;

    let fasta_path = find_output_fasta(&output_dir)?;
    let fasta = fs::read_to_string(&fasta_path)?;
    let mut result = parse_design_fasta(&fasta);
    result.raw_fasta = fasta;
    Ok(result)
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
    let model_name = "v_48_020";
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
        .arg("--batch_size")
        .arg("1");

    if !request.chains_to_design.is_empty() {
        command
            .arg("--pdb_path_chains")
            .arg(request.chains_to_design.join(" "));
    }
    Ok(())
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
    let directory = if seqs.is_dir() { seqs } else { output_dir.to_path_buf() };

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

        // The input record has no score; that is what distinguishes it from the designs.
        if result.input_sequence.is_none() && !fields.iter().any(|(key, _)| key == "score") {
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
}
