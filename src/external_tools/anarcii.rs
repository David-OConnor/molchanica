//! ANARCII: antibody and TCR variable-domain numbering.
//!
//! [ANARCII](https://github.com/oxpig/ANARCII) ·
//! [Paper](https://www.biorxiv.org/content/10.1101/2025.04.16.648720v1)
//!
//! `antibody.rs` says of its own CDR annotation that it "uses sequence-position approximations …
//! it should not be treated as a substitute for a full antibody numbering assignment with
//! insertion codes." This is that substitute. ANARCII assigns a canonical position and insertion
//! code to every residue, which is what makes CDR boundaries correct on chains that are longer or
//! shorter than the canonical length — precisely the chains anyone cares about, since an unusual
//! CDR loop is usually the point.
//!
//! It is used here in preference to the more familiar ANARCI/HMMER stack for one practical reason:
//! ANARCII is a language model, so its wheel is `py3-none-any` and its only heavy dependency is
//! Torch. There is no HMMER to build, which is what made antibody numbering effectively Linux-only
//! before. It installs and runs the same way on Windows and Linux.
//!
//! # How it is driven
//!
//! Through the Python API rather than a CLI, via a small bridge script written into the run's
//! workspace. A library API is a far more stable target than an argument parser, and the exchange
//! is JSON in and JSON out, so nothing here has to scrape human-readable output.

use std::{collections::BTreeMap, fs, io, ops::RangeInclusive, process::Command};

use na_seq::{AaIdent, AminoAcid};
use serde_json::Value;

use crate::external_tools::{Tool, ToolWorkspace, find_executable, run_to_completion};

/// Numbering conventions ANARCII can convert to.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NumberingScheme {
    /// The scheme ANARCII numbers in natively; everything else is a conversion of it.
    #[default]
    Imgt,
    Kabat,
    Chothia,
    /// Chothia with the Abhinandan/Martin corrections.
    Martin,
    Aho,
}

impl NumberingScheme {
    pub const ALL: [Self; 5] = [
        Self::Imgt,
        Self::Kabat,
        Self::Chothia,
        Self::Martin,
        Self::Aho,
    ];

    fn argument(self) -> &'static str {
        match self {
            Self::Imgt => "imgt",
            Self::Kabat => "kabat",
            Self::Chothia => "chothia",
            Self::Martin => "martin",
            Self::Aho => "aho",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Imgt => "IMGT",
            Self::Kabat => "Kabat",
            Self::Chothia => "Chothia",
            Self::Martin => "Martin",
            Self::Aho => "AHo",
        }
    }

    /// CDR position ranges under this scheme, given the chain class.
    ///
    /// These are ranges over *scheme* positions, not sequence offsets, which is the whole reason
    /// for numbering: an insertion adds residues without moving the boundary. Kabat and Chothia
    /// number heavy and light chains differently, so the class has to be known; IMGT and AHo are
    /// uniform across chain types, which is their main practical advantage.
    pub fn cdr_ranges(self, heavy: bool) -> [RangeInclusive<i32>; 3] {
        match (self, heavy) {
            (Self::Imgt, _) => [27..=38, 56..=65, 105..=117],
            (Self::Aho, _) => [25..=40, 58..=77, 109..=137],
            (Self::Kabat, true) => [31..=35, 50..=65, 95..=102],
            (Self::Kabat, false) => [24..=34, 50..=56, 89..=97],
            (Self::Chothia | Self::Martin, true) => [26..=32, 52..=56, 95..=102],
            (Self::Chothia | Self::Martin, false) => [24..=34, 50..=56, 89..=97],
        }
    }
}

/// Which family of receptor to number. Picking the right one improves both accuracy and speed;
/// `Unknown` lets ANARCII decide.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReceptorType {
    #[default]
    Antibody,
    Tcr,
    /// VNAR (shark) domains.
    Shark,
    Unknown,
}

impl ReceptorType {
    fn argument(self) -> &'static str {
        match self {
            Self::Antibody => "antibody",
            Self::Tcr => "tcr",
            Self::Shark => "shark",
            Self::Unknown => "unknown",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Antibody => "Antibody",
            Self::Tcr => "TCR",
            Self::Shark => "VNAR (shark)",
            Self::Unknown => "Auto-detect",
        }
    }
}

/// How a run should be configured.
#[derive(Clone, Debug)]
pub struct NumberingOptions {
    pub receptor: ReceptorType,
    pub scheme: NumberingScheme,
    /// ANARCII's `speed` mode trades a little accuracy for a much smaller model.
    pub fast: bool,
    /// Force CPU. Left false, ANARCII uses a GPU when its Torch build has one.
    pub force_cpu: bool,
}

impl Default for NumberingOptions {
    fn default() -> Self {
        Self {
            receptor: ReceptorType::default(),
            scheme: NumberingScheme::default(),
            fast: false,
            force_cpu: false,
        }
    }
}

/// One residue with its canonical position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumberedResidue {
    /// The scheme position. Stable across chains of different lengths.
    pub position: i32,
    /// The insertion code, where the scheme needed one to fit an extra residue at `position`.
    pub insertion: Option<char>,
    pub amino_acid: char,
    /// 0-based offset into the query sequence, so a caller can map back to the structure.
    pub query_index: usize,
}

impl NumberedResidue {
    /// The position as it is conventionally written, e.g. `100A`.
    pub fn label(&self) -> String {
        match self.insertion {
            Some(code) => format!("{}{code}", self.position),
            None => self.position.to_string(),
        }
    }

    /// A gap: ANARCII emits `-` where the scheme has a position this chain does not fill.
    pub fn is_gap(&self) -> bool {
        self.amino_acid == '-'
    }
}

/// One numbered chain.
#[derive(Clone, Debug)]
pub struct NumberedChain {
    /// `H`, `K`, `L` for antibodies; `A`, `B`, `G`, `D` for TCRs.
    pub chain_type: Option<String>,
    /// Model confidence. Low scores mean the sequence may not be a variable domain at all.
    pub score: Option<f32>,
    /// 0-based inclusive bounds of the numbered region within the query, which is how leader
    /// sequences and constant domains outside the variable domain are excluded.
    pub query_start: Option<usize>,
    pub query_end: Option<usize>,
    pub scheme: NumberingScheme,
    /// Set where ANARCII could not number the sequence.
    pub error: Option<String>,
    pub residues: Vec<NumberedResidue>,
}

impl NumberedChain {
    /// Whether this is a heavy or heavy-equivalent chain, which decides Kabat/Chothia boundaries.
    pub fn is_heavy(&self) -> bool {
        matches!(
            self.chain_type.as_deref().map(str::to_ascii_uppercase).as_deref(),
            Some("H" | "B" | "D")
        )
    }

    /// The three CDRs, as residue slices. Gap positions are dropped, so these are the residues
    /// actually present.
    pub fn cdrs(&self) -> [Vec<NumberedResidue>; 3] {
        let ranges = self.scheme.cdr_ranges(self.is_heavy());
        ranges.map(|range| {
            self.residues
                .iter()
                .filter(|residue| !residue.is_gap() && range.contains(&residue.position))
                .copied()
                .collect()
        })
    }

    /// The CDR sequences, in order.
    pub fn cdr_sequences(&self) -> [String; 3] {
        self.cdrs()
            .map(|residues| residues.iter().map(|residue| residue.amino_acid).collect())
    }

    /// Residues by scheme label, for annotating a structure.
    pub fn by_label(&self) -> BTreeMap<String, NumberedResidue> {
        self.residues
            .iter()
            .filter(|residue| !residue.is_gap())
            .map(|residue| (residue.label(), *residue))
            .collect()
    }

    /// The numbered region's sequence, gaps removed.
    pub fn sequence(&self) -> String {
        self.residues
            .iter()
            .filter(|residue| !residue.is_gap())
            .map(|residue| residue.amino_acid)
            .collect()
    }

    /// A one-line summary for the UI.
    pub fn summary(&self) -> String {
        if let Some(error) = &self.error {
            return format!("could not be numbered: {error}");
        }
        let chain = self.chain_type.as_deref().unwrap_or("?");
        let [cdr1, cdr2, cdr3] = self.cdr_sequences();
        format!(
            "chain {chain} · {} · CDR1 {} · CDR2 {} · CDR3 {}",
            self.scheme.label(),
            if cdr1.is_empty() { "-" } else { &cdr1 },
            if cdr2.is_empty() { "-" } else { &cdr2 },
            if cdr3.is_empty() { "-" } else { &cdr3 },
        )
    }
}

/// The bridge run inside ANARCII's environment.
///
/// Kept to the documented public API — construct `Anarcii`, call `number`, optionally
/// `to_scheme` — so that it depends on as little of ANARCII's internals as possible. Results are
/// emitted in the order `number` returned them, which for a list input is the order it was given,
/// and the count is asserted on the Rust side rather than trusting key names to round-trip.
const BRIDGE: &str = r#"
import json
import sys

from anarcii import Anarcii

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    request = json.load(handle)

model = Anarcii(
    seq_type=request["receptor"],
    mode="speed" if request["fast"] else "accuracy",
    batch_size=max(1, min(32, len(request["sequences"]))),
    cpu=request["force_cpu"],
    verbose=False,
)

numbered = model.number(request["sequences"])
if request["scheme"] != "imgt":
    converted = model.to_scheme(request["scheme"])
    # to_scheme returns the original output when a conversion does not apply; either way what
    # comes back is what should be reported.
    if converted is not None:
        numbered = converted

results = []
for value in numbered.values():
    entries = []
    for position, residue in value.get("numbering") or []:
        number, insertion = position
        insertion = (insertion or "").strip()
        entries.append([number, insertion, residue])
    results.append(
        {
            "chain_type": value.get("chain_type"),
            "score": value.get("score"),
            "query_start": value.get("query_start"),
            "query_end": value.get("query_end"),
            "scheme": value.get("scheme"),
            "error": value.get("error"),
            "numbering": entries,
        }
    )

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(results, handle)
"#;

/// Number one or more variable-domain sequences.
///
/// Blocking, and the first call in a process pays for a Torch import; run it on a worker thread.
/// Sequences are batched into a single invocation because that import dominates the cost.
pub fn number(sequences: &[String], options: &NumberingOptions) -> io::Result<Vec<NumberedChain>> {
    if sequences.is_empty() {
        return Ok(Vec::new());
    }
    if sequences.len() > 4_096 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ANARCII is called with at most 4096 sequences at a time",
        ));
    }
    let cleaned: Vec<String> = sequences
        .iter()
        .map(|sequence| {
            sequence
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .flat_map(char::to_uppercase)
                .collect()
        })
        .collect();
    if let Some(index) = cleaned.iter().position(String::is_empty) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("sequence {} is empty after removing non-letters", index + 1),
        ));
    }

    let python = find_executable(Tool::Anarcii)?;
    let workspace = ToolWorkspace::new("anarcii")?;
    let bridge_path = workspace.path("bridge.py");
    let request_path = workspace.path("request.json");
    let response_path = workspace.path("response.json");

    fs::write(&bridge_path, BRIDGE)?;
    fs::write(
        &request_path,
        serde_json::to_vec(&serde_json::json!({
            "sequences": cleaned,
            "receptor": options.receptor.argument(),
            "scheme": options.scheme.argument(),
            "fast": options.fast,
            "force_cpu": options.force_cpu,
        }))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
    )?;

    let mut command = Command::new(&python);
    command.arg(&bridge_path).arg(&request_path).arg(&response_path);
    run_to_completion(&mut command, "ANARCII")?;

    let response = fs::read_to_string(&response_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("ANARCII finished but wrote no result: {error}"),
        )
    })?;
    let parsed = parse_response(&response, options.scheme)?;

    if parsed.len() != cleaned.len() {
        return Err(io::Error::other(format!(
            "ANARCII returned {} results for {} sequences",
            parsed.len(),
            cleaned.len()
        )));
    }
    Ok(parsed)
}

/// Number a single chain taken from a structure.
pub fn number_amino_acids(
    aas: &[AminoAcid],
    options: &NumberingOptions,
) -> io::Result<NumberedChain> {
    if aas.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot number an empty chain",
        ));
    }
    let sequence: String = aas.iter().map(|aa| aa.to_str(AaIdent::OneLetter)).collect();
    number(&[sequence], options)?.into_iter().next().ok_or_else(|| {
        io::Error::other("ANARCII returned no result for the requested chain")
    })
}

fn parse_response(response: &str, scheme: NumberingScheme) -> io::Result<Vec<NumberedChain>> {
    let parsed: Value = serde_json::from_str(response)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let entries = parsed.as_array().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "ANARCII's bridge did not return a list of results",
        )
    })?;

    entries.iter().map(|entry| parse_chain(entry, scheme)).collect()
}

fn parse_chain(entry: &Value, requested: NumberingScheme) -> io::Result<NumberedChain> {
    let query_start = entry
        .get("query_start")
        .and_then(Value::as_u64)
        .map(|value| value as usize);

    let mut residues = Vec::new();
    if let Some(numbering) = entry.get("numbering").and_then(Value::as_array) {
        // ANARCII reports the numbered region only, so query offsets are counted from
        // `query_start`. Gaps consume a scheme position but no query residue, so they must not
        // advance the offset.
        let mut offset = query_start.unwrap_or(0);
        for item in numbering {
            let Some(fields) = item.as_array() else {
                continue;
            };
            let (Some(position), Some(insertion), Some(amino_acid)) = (
                fields.first().and_then(Value::as_i64),
                fields.get(1).and_then(Value::as_str),
                fields.get(2).and_then(Value::as_str),
            ) else {
                continue;
            };
            let amino_acid = amino_acid.chars().next().unwrap_or('-');
            residues.push(NumberedResidue {
                position: position as i32,
                insertion: insertion.chars().next().filter(|c| !c.is_whitespace()),
                amino_acid,
                query_index: offset,
            });
            if amino_acid != '-' {
                offset += 1;
            }
        }
    }

    // ANARCII echoes the scheme it produced; trust that over what we asked for, so a conversion
    // that silently did not apply is visible rather than mislabelled.
    let scheme = entry
        .get("scheme")
        .and_then(Value::as_str)
        .and_then(scheme_from_str)
        .unwrap_or(requested);

    Ok(NumberedChain {
        chain_type: entry
            .get("chain_type")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        score: entry
            .get("score")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
        query_start,
        query_end: entry
            .get("query_end")
            .and_then(Value::as_u64)
            .map(|value| value as usize),
        scheme,
        error: entry
            .get("error")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        residues,
    })
}

fn scheme_from_str(value: &str) -> Option<NumberingScheme> {
    NumberingScheme::ALL
        .into_iter()
        .find(|scheme| scheme.argument().eq_ignore_ascii_case(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed IMGT numbering: positions 27-38 are CDR1, and 32-33 are a gap, which is the
    /// normal way IMGT represents a short CDR1.
    const RESPONSE: &str = r#"[
      {
        "chain_type": "H",
        "score": 0.97,
        "query_start": 0,
        "query_end": 12,
        "scheme": "imgt",
        "error": null,
        "numbering": [
          [26, "", "C"],
          [27, "", "G"],
          [28, "", "F"],
          [29, "", "T"],
          [30, "", "F"],
          [31, "", "S"],
          [32, "", "-"],
          [33, "", "-"],
          [34, "", "S"],
          [35, "", "Y"],
          [36, "", "A"],
          [37, "", "M"],
          [38, "", "S"],
          [39, "", "W"]
        ]
      }
    ]"#;

    fn chain() -> NumberedChain {
        parse_response(RESPONSE, NumberingScheme::Imgt)
            .expect("response should parse")
            .remove(0)
    }

    #[test]
    fn parses_positions_and_metadata() {
        let chain = chain();

        assert_eq!(chain.chain_type.as_deref(), Some("H"));
        assert_eq!(chain.score, Some(0.97));
        assert_eq!(chain.scheme, NumberingScheme::Imgt);
        assert!(chain.error.is_none());
        assert!(chain.is_heavy());
        assert_eq!(chain.residues.len(), 14);
    }

    #[test]
    fn gaps_do_not_consume_query_positions() {
        let chain = chain();

        // Positions 32 and 33 are gaps, so 34 must take the query offset that follows 31's.
        let position = |number: i32| {
            chain
                .residues
                .iter()
                .find(|residue| residue.position == number)
                .copied()
                .expect("position should be present")
        };
        assert_eq!(position(31).query_index, 5);
        assert!(position(32).is_gap());
        assert_eq!(position(34).query_index, 6);
        // The sequence skips the gaps entirely.
        assert_eq!(chain.sequence(), "CGFTFSSYAMSW");
    }

    #[test]
    fn extracts_cdr1_from_scheme_positions_rather_than_offsets() {
        let chain = chain();
        let [cdr1, cdr2, cdr3] = chain.cdr_sequences();

        // IMGT CDR1 is 27-38; the two gaps drop out, and 26 and 39 are framework.
        assert_eq!(cdr1, "GFTFSSYAMS");
        // Nothing in this fragment reaches CDR2 or CDR3.
        assert!(cdr2.is_empty());
        assert!(cdr3.is_empty());
    }

    #[test]
    fn labels_insertions_the_conventional_way() {
        let residue = NumberedResidue {
            position: 100,
            insertion: Some('A'),
            amino_acid: 'G',
            query_index: 0,
        };
        assert_eq!(residue.label(), "100A");

        let plain = NumberedResidue {
            insertion: None,
            ..residue
        };
        assert_eq!(plain.label(), "100");
    }

    #[test]
    fn light_and_heavy_chains_get_different_kabat_boundaries() {
        // The point of tracking chain class: Kabat CDR1 starts at 31 on heavy, 24 on light.
        assert_eq!(NumberingScheme::Kabat.cdr_ranges(true)[0], 31..=35);
        assert_eq!(NumberingScheme::Kabat.cdr_ranges(false)[0], 24..=34);
        // IMGT is uniform, which is why it is the default.
        assert_eq!(
            NumberingScheme::Imgt.cdr_ranges(true),
            NumberingScheme::Imgt.cdr_ranges(false)
        );
    }

    #[test]
    fn reports_a_failed_numbering_rather_than_silently_succeeding() {
        let response = r#"[{"chain_type": null, "error": "no variable domain found", "numbering": []}]"#;
        let chain = parse_response(response, NumberingScheme::Imgt)
            .expect("an error result should still parse")
            .remove(0);

        assert_eq!(chain.error.as_deref(), Some("no variable domain found"));
        assert!(chain.residues.is_empty());
        assert!(chain.summary().contains("could not be numbered"));
    }
}
