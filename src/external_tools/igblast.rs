//! NCBI IgBLAST: antibody V(D)J germline assignment and framework/CDR delineation.
//!
//! [IgBLAST](https://ncbi.github.io/igblast/)
//!
//! `antibody.rs` annotates CDRs from sequence-position approximations, which is fine for triage,
//! selection, and MD region setup, but is not a numbering assignment: it cannot tell you which
//! germline gene a chain came from, and it drifts wherever a chain has insertions or deletions
//! relative to the canonical length. IgBLAST does the real thing — it aligns the query against
//! germline databases and reports both the gene calls and the framework/CDR boundaries the
//! alignment implies.
//!
//! IgBLAST is a plain native binary with no Python anywhere, and NCBI publishes both `x64-linux`
//! and `x64-win64` builds, which is why this is integrated as a process rather than reimplemented.
//!
//! Two output formats are used, because IgBLAST supports different ones per query type:
//!
//! - Nucleotide queries go through `igblastn -outfmt 19`, the AIRR rearrangement TSV. This is the
//!   richest output: gene calls, productivity, junction, and every region in both nucleotide and
//!   amino-acid form, as named columns.
//! - Protein queries go through `igblastp -outfmt 7`, which has no AIRR mode. Its human-readable
//!   report carries an alignment-summary table with per-region coordinates, which is parsed here.
//!   This is the path that matters for structures, where we have residues rather than reads.

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use na_seq::{AaIdent, AminoAcid};

use crate::external_tools::{
    RequiredAsset, Tool, ToolWorkspace, bundle_root, executable_in, find_executable,
    run_to_completion,
};

/// Query molecule type, which also picks the binary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SequenceType {
    /// Rearranged reads. Uses `igblastn`, and is the only path that reports D genes.
    Nucleotide,
    /// Variable-domain amino-acid sequences, e.g. a chain from a loaded structure.
    #[default]
    Protein,
}

impl SequenceType {
    fn binary(self) -> &'static str {
        match self {
            Self::Nucleotide => "igblastn",
            Self::Protein => "igblastp",
        }
    }

    /// BLAST writes one header file per database, and its extension says whether the database
    /// holds nucleotide or protein sequences, which has to match the binary.
    fn header_suffix(self) -> &'static str {
        match self {
            Self::Nucleotide => ".nhr",
            Self::Protein => ".phr",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Nucleotide => "Nucleotide",
            Self::Protein => "Protein",
        }
    }
}

/// Which convention the framework/CDR boundaries follow.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DomainSystem {
    #[default]
    Imgt,
    Kabat,
}

impl DomainSystem {
    fn argument(self) -> &'static str {
        match self {
            Self::Imgt => "imgt",
            Self::Kabat => "kabat",
        }
    }

    /// The suffix IgBLAST's alignment-summary rows carry, as in `FR1-IMGT`.
    fn row_suffix(self) -> &'static str {
        match self {
            Self::Imgt => "-IMGT",
            Self::Kabat => "-Kabat",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Imgt => "IMGT",
            Self::Kabat => "Kabat",
        }
    }
}

/// Organisms IgBLAST ships internal annotation data for.
pub const ORGANISMS: [&str; 5] = ["human", "mouse", "rat", "rabbit", "rhesus_monkey"];

/// One region of a variable domain, with 1-based inclusive positions into the query.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// `FR1`, `CDR1`, ... as IgBLAST names it, with the domain-system suffix stripped.
    pub name: String,
    pub start: usize,
    pub end: usize,
    /// The query subsequence, when the output carried it.
    pub sequence: Option<String>,
    /// Percent identity to the germline over this region, when reported.
    pub percent_identity: Option<f32>,
}

impl Region {
    /// 0-based offsets into the query sequence, for slicing.
    pub fn range(&self) -> Option<std::ops::Range<usize>> {
        (self.start >= 1 && self.end >= self.start).then(|| self.start - 1..self.end)
    }

    pub fn is_cdr(&self) -> bool {
        self.name.to_ascii_uppercase().starts_with("CDR")
    }
}

/// What IgBLAST found for one query.
#[derive(Clone, Debug, Default)]
pub struct IgBlastResult {
    pub query_name: String,
    /// Best V/D/J hits, most significant first. D is nucleotide-only, and absent from light-chain
    /// rearrangements even then.
    pub v_calls: Vec<String>,
    pub d_calls: Vec<String>,
    pub j_calls: Vec<String>,
    /// `IGH`, `IGK`, `IGL`, ... when IgBLAST could determine it.
    pub locus: Option<String>,
    /// Whether the rearrangement encodes a functional receptor. Nucleotide queries only.
    pub productive: Option<bool>,
    /// CDR3 as amino acids, the region that dominates specificity.
    pub cdr3_aa: Option<String>,
    pub junction_aa: Option<String>,
    /// Percent identity to the top V germline over the whole alignment.
    pub v_identity: Option<f32>,
    /// Frameworks and CDRs in query coordinates, in order along the sequence.
    pub regions: Vec<Region>,
    /// The report as IgBLAST wrote it, for display and for diagnosing surprises.
    pub raw_report: String,
}

impl IgBlastResult {
    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions
            .iter()
            .find(|region| region.name.eq_ignore_ascii_case(name))
    }

    pub fn cdrs(&self) -> impl Iterator<Item = &Region> {
        self.regions.iter().filter(|region| region.is_cdr())
    }

    pub fn top_v(&self) -> Option<&str> {
        self.v_calls.first().map(String::as_str)
    }

    /// A one-line summary for the UI, e.g. `IGHV3-23*01 / IGHJ4*02 · 95.9% V identity`.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(v) = self.top_v() {
            parts.push(v.to_owned());
        }
        if let Some(j) = self.j_calls.first() {
            parts.push(j.clone());
        }
        let mut summary = parts.join(" / ");
        if let Some(identity) = self.v_identity {
            summary.push_str(&format!(" · {identity:.1}% V identity"));
        }
        if summary.is_empty() {
            summary.push_str("no germline hit");
        }
        summary
    }
}

/// A query to run.
#[derive(Clone, Debug)]
pub struct IgBlastQuery {
    pub name: String,
    pub sequence: String,
    pub sequence_type: SequenceType,
    pub organism: String,
    pub domain_system: DomainSystem,
    /// Database names relative to [`germline_root`]. Left empty, the first installed database
    /// holding that segment is used, which is what a user who ran our installer wants.
    pub germline_v: Option<String>,
    pub germline_d: Option<String>,
    pub germline_j: Option<String>,
    pub num_alignments: usize,
}

impl IgBlastQuery {
    /// A protein query against the human germline databases: the common case for a chain taken
    /// from a loaded structure.
    pub fn protein(name: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sequence: sequence.into(),
            sequence_type: SequenceType::Protein,
            organism: "human".to_owned(),
            domain_system: DomainSystem::Imgt,
            germline_v: None,
            germline_d: None,
            germline_j: None,
            num_alignments: 5,
        }
    }

    pub fn from_amino_acids(name: impl Into<String>, aas: &[AminoAcid]) -> io::Result<Self> {
        if aas.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IgBLAST query sequence cannot be empty",
            ));
        }
        let sequence: String = aas.iter().map(|aa| aa.to_str(AaIdent::OneLetter)).collect();
        Ok(Self::protein(name, sequence))
    }

    fn validate(&self) -> io::Result<()> {
        if self.sequence.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IgBLAST query sequence cannot be empty",
            ));
        }
        if !ORGANISMS.contains(&self.organism.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported IgBLAST organism '{}'; expected one of {}",
                    self.organism,
                    ORGANISMS.join(", ")
                ),
            ));
        }
        Ok(())
    }

    /// Uppercased, with whitespace removed, as FASTA wants it.
    fn cleaned_sequence(&self) -> String {
        self.sequence
            .chars()
            .filter(|c| !c.is_whitespace())
            .flat_map(char::to_uppercase)
            .collect()
    }
}

/// Where the germline BLAST databases live.
pub fn germline_root() -> Option<PathBuf> {
    if let Ok(configured) = std::env::var("MOLCHANICA_IGBLAST_GERMLINE_ROOT") {
        return Some(PathBuf::from(configured));
    }
    bundle_root(Tool::IgBlast).ok().map(|root| root.join("germline_db"))
}

/// Names of the BLAST databases installed under the germline root, relative to it.
///
/// Discovered rather than hardcoded, so installing another germline set makes it selectable
/// without a code change.
pub fn available_databases(sequence_type: SequenceType) -> Vec<String> {
    let Some(root) = germline_root() else {
        return Vec::new();
    };
    let mut names = Vec::new();
    collect_databases(&root, &root, sequence_type.header_suffix(), &mut names);
    names.sort();
    names.dedup();
    names
}

fn collect_databases(root: &Path, directory: &Path, suffix: &str, names: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_databases(root, &path, suffix, names);
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = file_name.strip_suffix(suffix) else {
            continue;
        };
        // Databases large enough to be split carry a volume number, as in `human_gl_V.00.nhr`;
        // every volume maps back to the one database name BLAST is given.
        let stem = strip_volume_suffix(stem);
        let Ok(relative) = path.with_file_name(stem).strip_prefix(root).map(Path::to_path_buf)
        else {
            continue;
        };
        names.push(relative.to_string_lossy().replace('\\', "/"));
    }
}

fn strip_volume_suffix(stem: &str) -> &str {
    match stem.rsplit_once('.') {
        Some((head, tail)) if tail.len() >= 2 && tail.chars().all(|c| c.is_ascii_digit()) => head,
        _ => stem,
    }
}

/// Installed databases holding `segment` (`V`, `D`, or `J`).
///
/// Databases are conventionally named after the segment they hold, as either `airr_c_human_ig.V`
/// (NCBI's AIRR-C sets) or `mouse_gl_V` (the older sets). Where nothing matches that convention
/// every database is offered, so a custom `makeblastdb` set is still usable.
pub fn databases_for(segment: char, sequence_type: SequenceType) -> Vec<String> {
    let names = available_databases(sequence_type);
    let matching: Vec<_> = names
        .iter()
        .filter(|name| segment_of(name) == Some(segment.to_ascii_uppercase()))
        .cloned()
        .collect();
    if matching.is_empty() { names } else { matching }
}

fn segment_of(name: &str) -> Option<char> {
    let mut chars = name.chars().rev();
    let segment = chars.next()?.to_ascii_uppercase();
    if !matches!(segment, 'V' | 'D' | 'J') {
        return None;
    }
    matches!(chars.next()?, '.' | '_' | '-').then_some(segment)
}

fn resolve_database(
    selected: &Option<String>,
    segment: char,
    sequence_type: SequenceType,
) -> io::Result<PathBuf> {
    let root = germline_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "IgBLAST is not installed, so it has no germline databases",
        )
    })?;
    let installed = databases_for(segment, sequence_type);

    let name = match selected {
        Some(name) if !name.trim().is_empty() => {
            let name = name.trim();
            if !installed.iter().any(|candidate| candidate == name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "germline {segment} database '{name}' is not installed. Installed: {}",
                        if installed.is_empty() {
                            "(none)".to_owned()
                        } else {
                            installed.join(", ")
                        }
                    ),
                ));
            }
            name.to_owned()
        }
        _ => installed.first().cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "no germline {segment} database is installed under {}. Re-run: {}",
                    root.display(),
                    Tool::IgBlast.spec().install_command()
                ),
            )
        })?,
    };

    Ok(root.join(name))
}

/// The unpacked IgBLAST directory holding `internal_data` and `optional_file`.
///
/// IgBLAST resolves these through `IGDATA` rather than relative to the binary, so it has to be set
/// on every invocation or the run fails with an unhelpful message about missing annotation data.
fn install_root(binary: &Path) -> io::Result<PathBuf> {
    if let Ok(configured) = std::env::var("IGDATA") {
        let configured = PathBuf::from(configured);
        if configured.join("internal_data").is_dir() {
            return Ok(configured);
        }
    }
    if let Ok(root) = bundle_root(Tool::IgBlast)
        && root.join("internal_data").is_dir()
    {
        return Ok(root);
    }
    // The tarball layout is <root>/bin/igblastn alongside <root>/internal_data.
    if let Some(root) = binary.parent().and_then(Path::parent)
        && root.join("internal_data").is_dir()
    {
        return Ok(root.to_path_buf());
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "IgBLAST's internal_data directory was not found. Install the full NCBI distribution, \
         or set IGDATA to its directory.",
    ))
}

/// Resolve the binary for this query type. [`find_executable`] resolves `igblastn`; `igblastp`
/// sits beside it in the same distribution.
fn binary_for(sequence_type: SequenceType) -> io::Result<PathBuf> {
    let igblastn = find_executable(Tool::IgBlast)?;
    if sequence_type == SequenceType::Nucleotide {
        return Ok(igblastn);
    }
    let directory = igblastn.parent().ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "IgBLAST executable has no directory")
    })?;
    executable_in(directory, "igblastp").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("igblastp was not found beside {}", igblastn.display()),
        )
    })
}

/// Run IgBLAST against one query.
///
/// Blocking; call it from a worker thread when the GUI is waiting.
pub fn run(query: &IgBlastQuery) -> io::Result<IgBlastResult> {
    query.validate()?;

    let binary = binary_for(query.sequence_type)?;
    let root = install_root(&binary)?;
    let workspace = ToolWorkspace::new("igblast")?;
    let query_path = workspace.path("query.fasta");
    let name = sanitize_name(&query.name);
    fs::write(
        &query_path,
        format!(">{name}\n{}\n", query.cleaned_sequence()),
    )?;

    let mut command = Command::new(&binary);
    command
        .env("IGDATA", &root)
        .arg("-query")
        .arg(&query_path)
        .arg("-organism")
        .arg(&query.organism)
        .arg("-domain_system")
        .arg(query.domain_system.argument())
        .arg("-germline_db_V")
        .arg(resolve_database(&query.germline_v, 'V', query.sequence_type)?)
        .arg("-num_alignments_V")
        .arg(query.num_alignments.clamp(1, 100).to_string());

    match query.sequence_type {
        SequenceType::Nucleotide => {
            command
                .arg("-germline_db_J")
                .arg(resolve_database(&query.germline_j, 'J', query.sequence_type)?)
                .arg("-num_alignments_J")
                .arg(query.num_alignments.clamp(1, 100).to_string());

            // D applies to heavy-chain rearrangements only, so a missing D database is not an
            // error: light chains simply have no D segment to call.
            if let Ok(database) = resolve_database(&query.germline_d, 'D', query.sequence_type) {
                command.arg("-germline_db_D").arg(database);
            }

            // The auxiliary file supplies each J gene's coding-frame offset, without which
            // IgBLAST cannot place FR4 or report productivity.
            let auxiliary = root
                .join("optional_file")
                .join(format!("{}_gl.aux", query.organism));
            if auxiliary.is_file() {
                command.arg("-auxiliary_data").arg(auxiliary);
            }

            // AIRR rearrangement TSV: named columns rather than a report to scrape.
            command.arg("-outfmt").arg("19");
        }
        SequenceType::Protein => {
            // igblastp has no AIRR mode; 7 is the tabular report whose alignment-summary table
            // carries the per-region coordinates.
            command.arg("-outfmt").arg("7");
        }
    }

    let stdout = run_to_completion(&mut command, "IgBLAST")?;

    let mut result = match query.sequence_type {
        SequenceType::Nucleotide => parse_airr(&stdout)?,
        SequenceType::Protein => parse_tabular_report(&stdout, query.domain_system),
    };
    result.query_name = name;
    result.raw_report = stdout;
    Ok(result)
}

/// IgBLAST rejects a FASTA identifier containing whitespace, and truncates at the first space.
fn sanitize_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if cleaned.trim_matches('_').is_empty() {
        "query".to_owned()
    } else {
        cleaned
    }
}

/// Parse `-outfmt 19`, the AIRR rearrangement TSV: a header line of column names and one row per
/// query. Columns are addressed by name because IgBLAST's set of them varies by version.
fn parse_airr(report: &str) -> io::Result<IgBlastResult> {
    let mut lines = report
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'));

    let header = lines.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "IgBLAST produced no AIRR output",
        )
    })?;
    let columns: HashMap<&str, usize> = header
        .split('\t')
        .enumerate()
        .map(|(index, name)| (name.trim(), index))
        .collect();

    let row: Vec<&str> = lines
        .next()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IgBLAST's AIRR output had a header but no result row",
            )
        })?
        .split('\t')
        .collect();

    let field = |name: &str| -> Option<&str> {
        columns
            .get(name)
            .and_then(|index| row.get(*index))
            .map(|value| value.trim())
            .filter(|value| !value.is_empty() && *value != "NA")
    };

    // Gene calls are comma-separated, best first, and may carry a trailing score in parentheses.
    let calls = |name: &str| -> Vec<String> {
        field(name)
            .map(|value| {
                value
                    .split(',')
                    .map(|call| call.split('(').next().unwrap_or(call).trim().to_owned())
                    .filter(|call| !call.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    };

    let mut regions = Vec::new();
    for name in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"] {
        // AIRR reports region coordinates 1-based inclusive in `<name>_start` / `<name>_end`.
        let (Some(start), Some(end)) = (
            field(&format!("{name}_start")).and_then(|v| v.parse().ok()),
            field(&format!("{name}_end")).and_then(|v| v.parse().ok()),
        ) else {
            continue;
        };
        regions.push(Region {
            name: name.to_ascii_uppercase().replace("FWR", "FR"),
            start,
            end,
            sequence: field(&format!("{name}_aa")).map(str::to_owned),
            percent_identity: None,
        });
    }

    Ok(IgBlastResult {
        query_name: field("sequence_id").unwrap_or_default().to_owned(),
        v_calls: calls("v_call"),
        d_calls: calls("d_call"),
        j_calls: calls("j_call"),
        locus: field("locus").map(str::to_owned),
        productive: field("productive").map(|value| value.eq_ignore_ascii_case("T")),
        cdr3_aa: field("cdr3_aa").map(str::to_owned),
        junction_aa: field("junction_aa").map(str::to_owned),
        v_identity: field("v_identity").and_then(|value| value.parse().ok()),
        regions,
        raw_report: String::new(),
    })
}

/// Parse `-outfmt 7`, the tabular report `igblastp` produces.
///
/// Two things are wanted from it. The alignment-summary table gives per-region coordinates:
///
/// ```text
/// # Alignment summary between query and top germline V gene hit (from, to, length, matches, ...)
/// FR1-IMGT	1	25	25	21	4	0	84
/// CDR1-IMGT	26	33	8	5	3	0	62.5
/// Total	N/A	N/A	98	78	20	0	79.6
/// ```
///
/// The hit table below it gives the germline calls, as BLAST tabular rows whose second column is
/// the subject identifier.
fn parse_tabular_report(report: &str, domain_system: DomainSystem) -> IgBlastResult {
    let mut result = IgBlastResult::default();
    let suffix = domain_system.row_suffix();
    // Set once we have passed the "# Fields:" line, so we only treat rows below it as hits.
    let mut in_hit_table = false;

    for line in report.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            if comment.trim_start().starts_with("Fields:") {
                in_hit_table = true;
            } else if comment.trim_start().starts_with("Alignment summary") {
                in_hit_table = false;
            }
            continue;
        }

        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() < 2 {
            continue;
        }

        // An alignment-summary row: `FR1-IMGT`, `CDR1-IMGT`, ... The `Total` row is a rollup, and
        // supplies the overall identity rather than a region.
        if let Some(name) = columns[0].strip_suffix(suffix) {
            let (Some(start), Some(end)) = (
                columns.get(1).and_then(|v| v.trim().parse().ok()),
                columns.get(2).and_then(|v| v.trim().parse().ok()),
            ) else {
                continue;
            };
            result.regions.push(Region {
                name: name.to_ascii_uppercase(),
                start,
                end,
                sequence: None,
                percent_identity: columns.get(7).and_then(|v| v.trim().parse().ok()),
            });
            continue;
        }
        if columns[0].eq_ignore_ascii_case("Total") {
            result.v_identity = columns.get(7).and_then(|v| v.trim().parse().ok());
            continue;
        }

        if in_hit_table {
            // BLAST tabular: query id, subject id, % identity, ... The subject is the germline
            // gene, and rows arrive best-first.
            let call = columns[1].trim().to_owned();
            if !call.is_empty() && !result.v_calls.contains(&call) {
                result.v_calls.push(call);
                if result.v_identity.is_none() {
                    result.v_identity = columns.get(2).and_then(|v| v.trim().parse().ok());
                }
            }
        }
    }

    result.regions.sort_by_key(|region| region.start);
    result
}

/// The germline databases IgBLAST needs, reported per segment, for the tools panel.
pub const REQUIRED_DATABASES: [RequiredAsset; 1] = [RequiredAsset {
    relative_path: "germline_db",
    description: "the germline BLAST databases",
}];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_protein_alignment_summary() {
        let report = "\
# IGBLASTP 1.22.0+
# Query: heavy
# Database: airr_c_human_ig.V
# Domain classification requested: imgt

# Alignment summary between query and top germline V gene hit (from, to, length, matches, mismatches, gaps, percent identity)
FR1-IMGT\t1\t25\t25\t24\t1\t0\t96
CDR1-IMGT\t26\t33\t8\t7\t1\t0\t87.5
FR2-IMGT\t34\t50\t17\t17\t0\t0\t100
CDR2-IMGT\t51\t58\t8\t6\t2\t0\t75
FR3-IMGT\t59\t96\t38\t36\t2\t0\t94.7
Total\tN/A\tN/A\t96\t90\t6\t0\t93.8

# Hit table (the first field indicates the chain type of the hit)
# Fields: query id, subject id, % identity, alignment length
# 2 hits found
heavy\tIGHV3-23*01\t93.750\t96
heavy\tIGHV3-48*02\t89.583\t96
";

        let result = parse_tabular_report(report, DomainSystem::Imgt);

        assert_eq!(result.top_v(), Some("IGHV3-23*01"));
        assert_eq!(result.v_calls.len(), 2);
        assert_eq!(result.v_identity, Some(93.8));
        assert_eq!(result.regions.len(), 5);

        let cdr1 = result.region("CDR1").expect("CDR1 should be present");
        assert_eq!((cdr1.start, cdr1.end), (26, 33));
        assert_eq!(cdr1.range(), Some(25..33));
        assert!(cdr1.is_cdr());
        assert_eq!(result.cdrs().count(), 2);
    }

    #[test]
    fn parses_an_airr_rearrangement_row() {
        let report = "\
sequence_id\tlocus\tproductive\tv_call\td_call\tj_call\tv_identity\tcdr3_aa\tjunction_aa\tcdr3_start\tcdr3_end\tcdr3_aa_x\tfwr1_start\tfwr1_end
read1\tIGH\tT\tIGHV3-23*01,IGHV3-23*04\tIGHD3-10*01\tIGHJ4*02\t98.5\tARDRGYSSGWYFDY\tCARDRGYSSGWYFDYW\t295\t336\tx\t1\t75
";

        let result = parse_airr(report).expect("AIRR output should parse");

        assert_eq!(result.locus.as_deref(), Some("IGH"));
        assert_eq!(result.productive, Some(true));
        assert_eq!(result.v_calls, ["IGHV3-23*01", "IGHV3-23*04"]);
        assert_eq!(result.d_calls, ["IGHD3-10*01"]);
        assert_eq!(result.j_calls, ["IGHJ4*02"]);
        assert_eq!(result.v_identity, Some(98.5));
        assert_eq!(result.cdr3_aa.as_deref(), Some("ARDRGYSSGWYFDY"));
        // Only the regions with both coordinates present are emitted.
        assert_eq!(result.regions.len(), 2);
        assert_eq!(result.regions[0].name, "FR1");
        assert_eq!(result.regions[1].name, "CDR3");
        assert!(result.summary().starts_with("IGHV3-23*01 / IGHJ4*02"));
    }

    #[test]
    fn recognizes_segment_suffixes_in_database_names() {
        assert_eq!(segment_of("airr_c_human_ig.V"), Some('V'));
        assert_eq!(segment_of("mouse_gl_J"), Some('J'));
        assert_eq!(segment_of("rhesus_monkey-D"), Some('D'));
        // No separator before the letter, so this is not a segment-named database.
        assert_eq!(segment_of("customdbV"), None);
        assert_eq!(segment_of("imgt_human_ig_c"), None);
    }

    #[test]
    fn strips_blast_volume_suffixes() {
        assert_eq!(strip_volume_suffix("human_gl_V.00"), "human_gl_V");
        assert_eq!(strip_volume_suffix("human_gl_V"), "human_gl_V");
        // A single digit is not a volume number, and `airr_c_human_ig.V` must survive intact.
        assert_eq!(strip_volume_suffix("airr_c_human_ig.V"), "airr_c_human_ig.V");
    }

    #[test]
    fn sanitizes_fasta_identifiers() {
        assert_eq!(sanitize_name("heavy chain 1"), "heavy_chain_1");
        assert_eq!(sanitize_name("  "), "query");
        assert_eq!(sanitize_name("7XYZ_H"), "7XYZ_H");
    }
}
