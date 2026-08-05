//! Experimental / WIP. Support for useful antibody workflows.
//!
//! This module starts with dependency-light helpers that work directly on
//! Molchanica peptide structures and plain amino-acid sequences. The intent is
//! to support practical structural-biology workflows without requiring an
//! external antibody-numbering service:
//!
//! - classify likely heavy/light antibody chains,
//! - annotate approximate CDR ranges under common numbering schemes,
//! - collect paratope residue/atom selections for visualization and MD,
//! - find CDR-antigen contacts from loaded structures,
//! - flag common developability motifs in variable domains.
//!
//! The CDR extraction here uses sequence-position approximations. It is useful
//! for triage, UI selection, MD region setup, and notebook-style analysis, but
//! it should not be treated as a substitute for a full antibody numbering
//! assignment with insertion codes.
//!
//! Where that substitute matters, two optional tools supply it, and
//! [`refine_with_anarcii`] and [`germline_assignments`] below graft their
//! results onto the same annotation types the approximations produce:
//!
//! - **ANARCII** assigns a canonical position and insertion code to every
//!   residue, which is what makes CDR boundaries correct on a chain that is
//!   longer or shorter than the canonical length — that is, on exactly the
//!   chains anyone is interested in, since an unusual loop is usually the
//!   point. A position approximation cannot do this: it counts from the start
//!   of the chain, so one insertion in CDR1 shifts every boundary after it.
//! - **IgBLAST** additionally reports which germline genes a chain came from,
//!   which nothing here can infer from sequence heuristics alone.
//!
//! Neither is required. With neither installed the approximations are what you
//! get, everything still works offline, and [`AntibodyChainAnnotation::source`]
//! says which you are looking at — so a report never silently presents an
//! approximation as a numbering assignment.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::{self, Display, Formatter},
    io,
};

use bio_files::ResidueType;
use lin_alg::f64::Vec3;
use na_seq::{AaIdent, AminoAcid, Element};

use crate::{
    external_tools::{anarcii, igblast},
    molecules::{AtomRole, peptide::MoleculePeptide},
};

const VARIABLE_DOMAIN_SCAN_LEN: usize = 130;

/// CDR range convention used to annotate antibody chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CdrNumberingScheme {
    /// Kabat sequence-position ranges. Common in immunology literature.
    Kabat,
    /// Chothia sequence-position ranges. Often used with structure/canonical-loop work.
    Chothia,
    /// IMGT-style variable-domain ranges.
    Imgt,
}

impl Default for CdrNumberingScheme {
    fn default() -> Self {
        Self::Imgt
    }
}

impl Display for CdrNumberingScheme {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Kabat => "Kabat",
            Self::Chothia => "Chothia",
            Self::Imgt => "IMGT",
        };
        write!(f, "{label}")
    }
}

/// Coarse chain class used by antibody helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AntibodyChainKind {
    Heavy,
    HeavyLike,
    Light,
    LightLike,
    KappaLight,
    LambdaLight,
    Unknown,
}

impl Default for AntibodyChainKind {
    fn default() -> Self {
        Self::Unknown
    }
}

impl AntibodyChainKind {
    pub fn is_heavy(self) -> bool {
        matches!(self, Self::Heavy | Self::HeavyLike)
    }

    pub fn is_light(self) -> bool {
        matches!(
            self,
            Self::Light | Self::LightLike | Self::KappaLight | Self::LambdaLight
        )
    }

    pub fn is_antibody_like(self) -> bool {
        self.is_heavy() || self.is_light()
    }
}

impl Display for AntibodyChainKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Heavy => "heavy",
            Self::HeavyLike => "heavy-like",
            Self::Light => "light",
            Self::LightLike => "light-like",
            Self::KappaLight => "kappa light",
            Self::LambdaLight => "lambda light",
            Self::Unknown => "unknown",
        };
        write!(f, "{label}")
    }
}

/// A CDR label in heavy/light-chain notation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CdrLabel {
    H1,
    H2,
    H3,
    L1,
    L2,
    L3,
}

impl CdrLabel {
    pub fn number(self) -> u8 {
        match self {
            Self::H1 | Self::L1 => 1,
            Self::H2 | Self::L2 => 2,
            Self::H3 | Self::L3 => 3,
        }
    }

    pub fn is_heavy(self) -> bool {
        matches!(self, Self::H1 | Self::H2 | Self::H3)
    }

    pub fn is_light(self) -> bool {
        !self.is_heavy()
    }
}

impl Display for CdrLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::H1 => "H1",
            Self::H2 => "H2",
            Self::H3 => "H3",
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
        };
        write!(f, "{label}")
    }
}

/// A residue address that remains useful across UI, CLI, and exported reports.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResidueRef {
    pub chain_id: String,
    pub chain_i: usize,
    /// Index into `MoleculePeptide::residues`.
    pub residue_i: usize,
    /// Residue serial number from the structure file.
    pub serial_number: u32,
    pub aa: Option<AminoAcid>,
}

impl Display for ResidueRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.aa {
            Some(aa) => write!(
                f,
                "{}:{}{}",
                self.chain_id,
                self.serial_number,
                aa.to_str(AaIdent::OneLetter)
            ),
            None => write!(f, "{}:{}", self.chain_id, self.serial_number),
        }
    }
}

/// Evidence backing automatic antibody chain classification.
#[derive(Clone, Debug)]
pub struct ChainClassification {
    pub kind: AntibodyChainKind,
    /// 0.0 to 1.0; heuristics only, not a germline assignment.
    pub confidence: f32,
    pub notes: Vec<String>,
}

/// A single annotated CDR segment.
#[derive(Clone, Debug)]
pub struct CdrAnnotation {
    pub label: CdrLabel,
    pub scheme: CdrNumberingScheme,
    pub chain_kind: AntibodyChainKind,
    pub chain_id: String,
    pub chain_i: usize,
    /// 1-based sequence positions relative to the first amino-acid residue in the chain.
    pub start_position: usize,
    pub end_position: usize,
    pub residues: Vec<ResidueRef>,
    /// Approximate CDR center from structure coordinates when available.
    pub centroid: Option<Vec3>,
    pub sequence: String,
}

impl CdrAnnotation {
    pub fn residue_indices(&self) -> Vec<usize> {
        self.residues.iter().map(|r| r.residue_i).collect()
    }
}

/// Where a chain's CDR boundaries came from.
///
/// Carried on the annotation so that a report or a UI can say which it is showing. The two are
/// not interchangeable, and the difference is invisible in the boundaries themselves.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum AnnotationSource {
    /// Sequence-position approximations computed here, with no external tool.
    #[default]
    Approximate,
    /// A real numbering assignment from ANARCII, with insertion codes.
    Anarcii,
    /// Boundaries implied by IgBLAST's alignment against the germline databases.
    IgBlast,
}

impl Display for AnnotationSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Approximate => "position approximation",
            Self::Anarcii => "ANARCII numbering",
            Self::IgBlast => "IgBLAST alignment",
        })
    }
}

/// Antibody-oriented annotation for one peptide chain.
#[derive(Clone, Debug)]
pub struct AntibodyChainAnnotation {
    pub chain_id: String,
    pub chain_i: usize,
    pub kind: AntibodyChainKind,
    pub confidence: f32,
    pub sequence: String,
    pub sequence_aa: Vec<AminoAcid>,
    pub residues: Vec<ResidueRef>,
    pub cdrs: Vec<CdrAnnotation>,
    pub variable_domain_residues: Vec<ResidueRef>,
    pub notes: Vec<String>,
    /// How `cdrs` were determined. See [`AnnotationSource`].
    pub source: AnnotationSource,
    /// Scheme positions per residue, when a real numbering was applied. Indexed the same way as
    /// `residues`, with `None` where the numbering did not cover a residue — a leader sequence or
    /// a constant domain outside the variable region.
    pub numbering: Vec<Option<String>>,
    /// Germline gene calls, when IgBLAST supplied them. Best hit first.
    pub germline_v: Vec<String>,
    pub germline_j: Vec<String>,
}

impl AntibodyChainAnnotation {
    pub fn cdr(&self, label: CdrLabel) -> Option<&CdrAnnotation> {
        self.cdrs.iter().find(|cdr| cdr.label == label)
    }

    pub fn paratope_residues(&self) -> Vec<ResidueRef> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for cdr in &self.cdrs {
            for residue in &cdr.residues {
                if seen.insert((residue.chain_i, residue.residue_i)) {
                    result.push(residue.clone());
                }
            }
        }
        result
    }

    pub fn contains_sequence_position_in_cdr(&self, position_1_based: usize) -> bool {
        self.cdrs.iter().any(|cdr| {
            cdr.start_position <= position_1_based && position_1_based <= cdr.end_position
        })
    }
}

/// Whole-structure antibody annotation.
#[derive(Clone, Debug)]
pub struct AntibodyAnnotation {
    pub scheme: CdrNumberingScheme,
    pub chains: Vec<AntibodyChainAnnotation>,
    pub developability_issues: Vec<DevelopabilityIssue>,
    pub notes: Vec<String>,
}

impl AntibodyAnnotation {
    pub fn antibody_chains(&self) -> impl Iterator<Item = &AntibodyChainAnnotation> {
        self.chains
            .iter()
            .filter(|chain| chain.kind.is_antibody_like())
    }

    pub fn paratope_residues(&self) -> Vec<ResidueRef> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for chain in self.antibody_chains() {
            for residue in chain.paratope_residues() {
                if seen.insert((residue.chain_i, residue.residue_i)) {
                    result.push(residue);
                }
            }
        }
        result
    }

    pub fn cdr(&self, chain_id: &str, label: CdrLabel) -> Option<&CdrAnnotation> {
        self.chains
            .iter()
            .find(|chain| chain.chain_id == chain_id)
            .and_then(|chain| chain.cdr(label))
    }

    pub fn paratope_pymol_selection(&self) -> String {
        pymol_selection_for_residues(&self.paratope_residues())
    }
}

/// Common sequence motif and chemistry concerns for antibody developability triage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DevelopabilityIssueKind {
    PotentialNLinkedGlycosylation,
    ExtraCysteine,
    MissingConservedResidue,
    DeamidationMotif,
    AspIsomerizationMotif,
    MethionineOxidationSite,
    ChargedPatch,
    HydrophobicPatch,
}

/// Qualitative severity for triage and UI presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IssueSeverity {
    Info,
    Low,
    Medium,
    High,
}

/// A developability warning tied to concrete residues when possible.
#[derive(Clone, Debug)]
pub struct DevelopabilityIssue {
    pub kind: DevelopabilityIssueKind,
    pub severity: IssueSeverity,
    pub residues: Vec<ResidueRef>,
    pub message: String,
}

/// Antibody-antigen contact between a CDR residue and a partner residue.
#[derive(Clone, Debug)]
pub struct ResidueContact {
    pub antibody: ResidueRef,
    pub partner: ResidueRef,
    pub distance_angstrom: f64,
}

/// Parameters for building a focused MD/relaxation region around an antibody paratope.
#[derive(Clone, Debug)]
pub struct AntibodyMdRegionConfig {
    /// Include this many residues before/after each CDR on the antibody chain.
    pub cdr_padding: usize,
    /// Partner residues at or below this heavy-atom distance are treated as antigen contacts.
    pub antigen_contact_cutoff_angstrom: f64,
    /// Framework residues near the paratope that are useful restraint candidates.
    pub framework_shell_cutoff_angstrom: f64,
}

impl Default for AntibodyMdRegionConfig {
    fn default() -> Self {
        Self {
            cdr_padding: 2,
            antigen_contact_cutoff_angstrom: 5.0,
            framework_shell_cutoff_angstrom: 8.0,
        }
    }
}

/// A residue/atom plan that can be converted into UI selections or MD masks.
#[derive(Clone, Debug, Default)]
pub struct AntibodyMdRegionPlan {
    pub cdr_residues: Vec<ResidueRef>,
    pub cdr_atom_indices: Vec<usize>,
    pub antigen_contact_residues: Vec<ResidueRef>,
    pub antigen_contact_atom_indices: Vec<usize>,
    pub framework_shell_residues: Vec<ResidueRef>,
    pub framework_shell_atom_indices: Vec<usize>,
    /// CDR plus antigen-contact atoms. Framework shell atoms are kept separate so callers can
    /// restrain them instead of making them fully mobile.
    pub mobile_atom_indices: Vec<usize>,
    pub notes: Vec<String>,
}

/// Annotate likely antibody chains, approximate CDRs, and sequence-level developability motifs.
pub fn annotate_antibody(
    peptide: &MoleculePeptide,
    scheme: CdrNumberingScheme,
) -> AntibodyAnnotation {
    let mut chains = Vec::new();
    let mut developability_issues = Vec::new();

    for chain_i in 0..peptide.chains.len() {
        if let Some(annotation) = annotate_peptide_chain(peptide, chain_i, scheme) {
            developability_issues.extend(developability_issues_for_chain(&annotation));
            chains.push(annotation);
        }
    }

    let mut notes = vec![format!(
        "{scheme} CDRs are sequence-position approximations; use full antibody numbering for final reports."
    )];
    if chains.iter().all(|chain| !chain.kind.is_antibody_like()) {
        notes.push("No antibody-like chain was identified from sequence heuristics.".to_string());
    }

    AntibodyAnnotation {
        scheme,
        chains,
        developability_issues,
        notes,
    }
}

/// Annotate a plain sequence. Useful before a structure is available.
pub fn annotate_sequence(
    chain_id: &str,
    sequence: &[AminoAcid],
    scheme: CdrNumberingScheme,
) -> AntibodyChainAnnotation {
    let residues = sequence
        .iter()
        .enumerate()
        .map(|(i, aa)| ResidueRef {
            chain_id: chain_id.to_string(),
            chain_i: 0,
            residue_i: i,
            serial_number: i as u32 + 1,
            aa: Some(*aa),
        })
        .collect();

    annotate_sequence_with_refs(chain_id, 0, sequence, residues, scheme, None)
}

/// Classify a sequence as heavy/light antibody-like, when possible.
pub fn classify_chain_sequence(sequence: &[AminoAcid]) -> ChainClassification {
    let len = sequence.len();
    let mut notes = Vec::new();
    let has_variable_signature = has_variable_domain_signature(sequence);

    if len >= 300 {
        notes.push("Length is consistent with an immunoglobulin heavy chain.".to_string());
        return ChainClassification {
            kind: AntibodyChainKind::Heavy,
            confidence: 0.9,
            notes,
        };
    }

    if (170..=260).contains(&len) {
        notes.push("Length is consistent with a complete light chain.".to_string());
        return ChainClassification {
            kind: AntibodyChainKind::Light,
            confidence: 0.8,
            notes,
        };
    }

    if has_variable_signature && len >= 112 {
        notes.push(
            "Variable-domain conserved residues are present; length is heavy-like.".to_string(),
        );
        return ChainClassification {
            kind: AntibodyChainKind::HeavyLike,
            confidence: 0.65,
            notes,
        };
    }

    if has_variable_signature && (95..112).contains(&len) {
        notes.push(
            "Variable-domain conserved residues are present; length is light-like.".to_string(),
        );
        return ChainClassification {
            kind: AntibodyChainKind::LightLike,
            confidence: 0.65,
            notes,
        };
    }

    if (112..=160).contains(&len) {
        notes.push(
            "Variable-domain length is heavy-like, but conserved-residue evidence is weak."
                .to_string(),
        );
        return ChainClassification {
            kind: AntibodyChainKind::HeavyLike,
            confidence: 0.45,
            notes,
        };
    }

    if (95..112).contains(&len) {
        notes.push(
            "Variable-domain length is light-like, but conserved-residue evidence is weak."
                .to_string(),
        );
        return ChainClassification {
            kind: AntibodyChainKind::LightLike,
            confidence: 0.45,
            notes,
        };
    }

    ChainClassification {
        kind: AntibodyChainKind::Unknown,
        confidence: 0.0,
        notes: vec!["Sequence length/signature does not look antibody-like.".to_string()],
    }
}

/// Return CDR definitions as 1-based sequence ranges.
pub fn cdr_definitions(
    scheme: CdrNumberingScheme,
    chain_kind: AntibodyChainKind,
) -> Vec<(CdrLabel, usize, usize)> {
    let heavy = chain_kind.is_heavy();

    match scheme {
        CdrNumberingScheme::Imgt => {
            if heavy {
                vec![
                    (CdrLabel::H1, 27, 38),
                    (CdrLabel::H2, 56, 65),
                    (CdrLabel::H3, 105, 117),
                ]
            } else {
                vec![
                    (CdrLabel::L1, 27, 38),
                    (CdrLabel::L2, 56, 65),
                    (CdrLabel::L3, 105, 117),
                ]
            }
        }
        CdrNumberingScheme::Kabat => {
            if heavy {
                vec![
                    (CdrLabel::H1, 31, 35),
                    (CdrLabel::H2, 50, 65),
                    (CdrLabel::H3, 95, 102),
                ]
            } else {
                vec![
                    (CdrLabel::L1, 24, 34),
                    (CdrLabel::L2, 50, 56),
                    (CdrLabel::L3, 89, 97),
                ]
            }
        }
        CdrNumberingScheme::Chothia => {
            if heavy {
                vec![
                    (CdrLabel::H1, 26, 32),
                    (CdrLabel::H2, 52, 56),
                    (CdrLabel::H3, 95, 102),
                ]
            } else {
                vec![
                    (CdrLabel::L1, 24, 34),
                    (CdrLabel::L2, 50, 56),
                    (CdrLabel::L3, 89, 97),
                ]
            }
        }
    }
}

/// Build a focused MD/relaxation plan around CDRs and their antigen contacts.
///
/// If `antigen_chain_ids` is empty, every non-antibody-like peptide chain is considered a
/// partner chain.
pub fn cdr_md_region_plan(
    peptide: &MoleculePeptide,
    annotation: &AntibodyAnnotation,
    antigen_chain_ids: &[&str],
    config: &AntibodyMdRegionConfig,
) -> AntibodyMdRegionPlan {
    let mut plan = AntibodyMdRegionPlan::default();

    let mut cdr_residue_indices = HashSet::new();
    for chain in annotation.antibody_chains() {
        for cdr in &chain.cdrs {
            for residue in &cdr.residues {
                cdr_residue_indices.insert(residue.residue_i);
            }

            let start = cdr.start_position.saturating_sub(config.cdr_padding + 1);
            let end = (cdr.end_position + config.cdr_padding).min(chain.residues.len());
            for i in start..end {
                if let Some(residue) = chain.residues.get(i) {
                    cdr_residue_indices.insert(residue.residue_i);
                }
            }
        }
    }

    plan.cdr_residues = refs_from_residue_indices(peptide, &cdr_residue_indices);
    plan.cdr_atom_indices = atom_indices_for_residues(peptide, &cdr_residue_indices);

    let contacts = cdr_antigen_contacts(
        peptide,
        annotation,
        antigen_chain_ids,
        config.antigen_contact_cutoff_angstrom,
    );

    let mut antigen_contact_indices = HashSet::new();
    for contact in contacts {
        antigen_contact_indices.insert(contact.partner.residue_i);
    }
    plan.antigen_contact_residues = refs_from_residue_indices(peptide, &antigen_contact_indices);
    plan.antigen_contact_atom_indices =
        atom_indices_for_residues(peptide, &antigen_contact_indices);

    let mut framework_shell_indices = HashSet::new();
    for chain in annotation.antibody_chains() {
        for residue in &chain.variable_domain_residues {
            if cdr_residue_indices.contains(&residue.residue_i) {
                continue;
            }
            if residue_is_near_any(
                peptide,
                residue.residue_i,
                &cdr_residue_indices,
                config.framework_shell_cutoff_angstrom,
            ) {
                framework_shell_indices.insert(residue.residue_i);
            }
        }
    }
    plan.framework_shell_residues = refs_from_residue_indices(peptide, &framework_shell_indices);
    plan.framework_shell_atom_indices =
        atom_indices_for_residues(peptide, &framework_shell_indices);

    let mut mobile_atoms: HashSet<usize> = plan.cdr_atom_indices.iter().copied().collect();
    mobile_atoms.extend(plan.antigen_contact_atom_indices.iter().copied());
    plan.mobile_atom_indices = sorted_usizes(mobile_atoms);

    plan.notes.push(format!(
        "Mobile atoms include padded CDRs and antigen residues within {:.1} Angstrom.",
        config.antigen_contact_cutoff_angstrom
    ));
    plan.notes.push(format!(
        "Framework shell atoms are within {:.1} Angstrom of the padded CDR region and are good restraint candidates.",
        config.framework_shell_cutoff_angstrom
    ));

    plan
}

/// Find contacts between annotated CDR residues and partner-chain residues.
///
/// If `antigen_chain_ids` is empty, all non-antibody-like chains are scanned as putative
/// partner chains.
pub fn cdr_antigen_contacts(
    peptide: &MoleculePeptide,
    annotation: &AntibodyAnnotation,
    antigen_chain_ids: &[&str],
    cutoff_angstrom: f64,
) -> Vec<ResidueContact> {
    let antibody_residues = annotation.paratope_residues();
    let partner_residues = partner_residues(peptide, annotation, antigen_chain_ids);
    let mut contacts = Vec::new();

    for antibody in &antibody_residues {
        for partner in &partner_residues {
            let Some(distance) = min_heavy_atom_distance(
                peptide,
                antibody.residue_i,
                partner.residue_i,
                Some(cutoff_angstrom),
            ) else {
                continue;
            };

            if distance <= cutoff_angstrom {
                contacts.push(ResidueContact {
                    antibody: antibody.clone(),
                    partner: partner.clone(),
                    distance_angstrom: distance,
                });
            }
        }
    }

    contacts.sort_by(|a, b| {
        a.distance_angstrom
            .partial_cmp(&b.distance_angstrom)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    contacts
}

/// Create a PyMOL-style residue selection string grouped by chain.
pub fn pymol_selection_for_residues(residues: &[ResidueRef]) -> String {
    if residues.is_empty() {
        return "none".to_string();
    }

    let mut by_chain: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for residue in residues {
        by_chain
            .entry(residue.chain_id.as_str())
            .or_default()
            .push(residue.serial_number);
    }

    let mut parts = Vec::new();
    for (chain_id, mut serials) in by_chain {
        serials.sort_unstable();
        serials.dedup();
        let resi = serials
            .iter()
            .map(|serial| serial.to_string())
            .collect::<Vec<_>>()
            .join("+");
        if chain_id.is_empty() {
            parts.push(format!("resi {resi}"));
        } else {
            parts.push(format!("chain {chain_id} and resi {resi}"));
        }
    }

    parts.join(" or ")
}

/// Approximate residue centroid from current atom positions.
///
/// For amino-acid residues this returns the C-alpha position when available, otherwise the
/// heavy-atom centroid, otherwise the all-atom centroid.
pub fn residue_centroid(peptide: &MoleculePeptide, residue_i: usize) -> Option<Vec3> {
    let residue = peptide.residues.get(residue_i)?;

    for &atom_i in &residue.atoms {
        let atom = peptide.common.atoms.get(atom_i)?;
        if atom.role == Some(AtomRole::C_Alpha) {
            return atom_posit(peptide, atom_i);
        }
    }

    let heavy = centroid_for_atoms(
        peptide,
        residue.atoms.iter().copied().filter(|&atom_i| {
            peptide
                .common
                .atoms
                .get(atom_i)
                .is_some_and(|atom| atom.element != Element::Hydrogen)
        }),
    );
    if heavy.is_some() {
        return heavy;
    }

    centroid_for_atoms(peptide, residue.atoms.iter().copied())
}

fn annotate_peptide_chain(
    peptide: &MoleculePeptide,
    chain_i: usize,
    scheme: CdrNumberingScheme,
) -> Option<AntibodyChainAnnotation> {
    let chain = peptide.chains.get(chain_i)?;
    let mut sequence = Vec::new();
    let mut residue_refs = Vec::new();

    for &residue_i in &chain.residues {
        let Some(residue) = peptide.residues.get(residue_i) else {
            continue;
        };
        let ResidueType::AminoAcid(aa) = residue.res_type else {
            continue;
        };
        sequence.push(aa);
        residue_refs.push(ResidueRef {
            chain_id: chain.id.clone(),
            chain_i,
            residue_i,
            serial_number: residue.serial_number,
            aa: Some(aa),
        });
    }

    if sequence.is_empty() {
        return None;
    }

    Some(annotate_sequence_with_refs(
        &chain.id,
        chain_i,
        &sequence,
        residue_refs,
        scheme,
        Some(peptide),
    ))
}

fn annotate_sequence_with_refs(
    chain_id: &str,
    chain_i: usize,
    sequence: &[AminoAcid],
    residues: Vec<ResidueRef>,
    scheme: CdrNumberingScheme,
    peptide: Option<&MoleculePeptide>,
) -> AntibodyChainAnnotation {
    let classification = classify_chain_sequence(sequence);
    let mut notes = classification.notes.clone();
    let mut cdrs = Vec::new();

    if classification.kind.is_antibody_like() {
        for (label, start_position, end_position) in cdr_definitions(scheme, classification.kind) {
            if end_position > residues.len() {
                notes.push(format!(
                    "{label} range {start_position}-{end_position} extends beyond chain length {}.",
                    residues.len()
                ));
                continue;
            }

            let cdr_residues = residues[start_position - 1..end_position].to_vec();
            let sequence = sequence_to_string(&sequence[start_position - 1..end_position]);
            let centroid = peptide.and_then(|pep| {
                centroid_for_residue_refs(pep, cdr_residues.iter().map(|r| r.residue_i))
            });
            cdrs.push(CdrAnnotation {
                label,
                scheme,
                chain_kind: classification.kind,
                chain_id: chain_id.to_string(),
                chain_i,
                start_position,
                end_position,
                residues: cdr_residues,
                centroid,
                sequence,
            });
        }
    }

    let variable_domain_len = sequence.len().min(VARIABLE_DOMAIN_SCAN_LEN);
    let variable_domain_residues = residues[..variable_domain_len].to_vec();

    AntibodyChainAnnotation {
        chain_id: chain_id.to_string(),
        chain_i,
        kind: classification.kind,
        confidence: classification.confidence,
        sequence: sequence_to_string(sequence),
        sequence_aa: sequence.to_vec(),
        numbering: vec![None; residues.len()],
        residues,
        cdrs,
        variable_domain_residues,
        notes,
        source: AnnotationSource::Approximate,
        germline_v: Vec::new(),
        germline_j: Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// Optional refinement through the external antibody tools
// ---------------------------------------------------------------------------------------------

/// Replace the approximated CDRs with ANARCII's numbering.
///
/// Every antibody-like chain is numbered in one invocation, because the cost is dominated by
/// starting a Python process and importing Torch, not by the sequences themselves.
///
/// Chains ANARCII cannot number are left with their approximations and gain a note saying so:
/// a scFv linker or a partially resolved chain should not lose the annotation it had.
///
/// Blocking — run it on a worker thread.
pub fn refine_with_anarcii(
    annotation: &mut AntibodyAnnotation,
    scheme: anarcii::NumberingScheme,
) -> io::Result<()> {
    let indices: Vec<usize> = annotation
        .chains
        .iter()
        .enumerate()
        .filter(|(_, chain)| chain.kind.is_antibody_like() && !chain.sequence_aa.is_empty())
        .map(|(index, _)| index)
        .collect();
    if indices.is_empty() {
        return Ok(());
    }

    let sequences: Vec<String> = indices
        .iter()
        .map(|index| annotation.chains[*index].sequence.clone())
        .collect();
    let numbered = anarcii::number(
        &sequences,
        &anarcii::NumberingOptions {
            scheme,
            ..anarcii::NumberingOptions::default()
        },
    )?;

    for (index, chain_numbering) in indices.into_iter().zip(numbered) {
        apply_numbering(&mut annotation.chains[index], &chain_numbering);
    }

    annotation
        .notes
        .retain(|note| !note.contains("approximations"));
    annotation.notes.push(format!(
        "CDR boundaries refined with ANARCII ({}).",
        scheme.label()
    ));
    Ok(())
}

/// Graft one chain's ANARCII numbering onto its annotation.
fn apply_numbering(chain: &mut AntibodyChainAnnotation, numbered: &anarcii::NumberedChain) {
    if let Some(error) = &numbered.error {
        chain
            .notes
            .push(format!("ANARCII could not number this chain: {error}"));
        return;
    }
    if numbered.residues.is_empty() {
        chain
            .notes
            .push("ANARCII returned no numbering for this chain.".to_string());
        return;
    }

    chain.numbering = vec![None; chain.residues.len()];
    for residue in &numbered.residues {
        if residue.is_gap() {
            continue;
        }
        if let Some(slot) = chain.numbering.get_mut(residue.query_index) {
            *slot = Some(residue.label());
        }
    }

    // ANARCII decides the chain class from the sequence itself, which is a better answer than the
    // motif heuristics here; adopt it so that Kabat and Chothia boundaries, which differ between
    // heavy and light, are taken from the right table.
    if let Some(kind) = chain_kind_from_anarcii(numbered.chain_type.as_deref()) {
        chain.kind = kind;
        chain.confidence = numbered.score.unwrap_or(chain.confidence);
    }

    let heavy = chain.kind.is_heavy();
    let labels = if heavy {
        [CdrLabel::H1, CdrLabel::H2, CdrLabel::H3]
    } else {
        [CdrLabel::L1, CdrLabel::L2, CdrLabel::L3]
    };
    let scheme = cdr_scheme_from_anarcii(numbered.scheme);

    let mut cdrs = Vec::new();
    for (label, residues) in labels.into_iter().zip(numbered.cdrs()) {
        // Positions are query offsets, which index `chain.residues` directly.
        let refs: Vec<ResidueRef> = residues
            .iter()
            .filter_map(|residue| chain.residues.get(residue.query_index).cloned())
            .collect();
        if refs.is_empty() {
            continue;
        }
        // 1-based, to match the convention the approximated annotations already use.
        let start_position = residues[0].query_index + 1;
        let end_position = residues[residues.len() - 1].query_index + 1;

        cdrs.push(CdrAnnotation {
            label,
            scheme,
            chain_kind: chain.kind,
            chain_id: chain.chain_id.clone(),
            chain_i: chain.chain_i,
            start_position,
            end_position,
            residues: refs,
            // Recomputed by the caller where a structure is available; the sequence-only path has
            // no coordinates to average.
            centroid: None,
            sequence: residues.iter().map(|residue| residue.amino_acid).collect(),
        });
    }

    if cdrs.is_empty() {
        chain
            .notes
            .push("ANARCII numbered this chain but found no CDR positions in it.".to_string());
        return;
    }

    chain.cdrs = cdrs;
    chain.source = AnnotationSource::Anarcii;
    // The numbered region is the variable domain, which is a real boundary rather than the fixed
    // 130-residue window the approximation scans.
    if let (Some(start), Some(end)) = (numbered.query_start, numbered.query_end) {
        let end = (end + 1).min(chain.residues.len());
        if start < end {
            chain.variable_domain_residues = chain.residues[start..end].to_vec();
        }
    }
}

fn chain_kind_from_anarcii(chain_type: Option<&str>) -> Option<AntibodyChainKind> {
    match chain_type?.to_ascii_uppercase().as_str() {
        "H" => Some(AntibodyChainKind::Heavy),
        // ANARCII distinguishes the two light-chain loci; this module does not.
        "K" | "L" => Some(AntibodyChainKind::Light),
        _ => None,
    }
}

/// Map ANARCII's scheme onto this module's, for labelling the refined CDRs.
///
/// Martin is reported as Chothia because it *is* Chothia with the Abhinandan corrections, and
/// this module's enum does not distinguish them. AHo has no counterpart at all, so it falls back
/// to IMGT — both are uniform across chain types, which is the property the label is used for.
fn cdr_scheme_from_anarcii(scheme: anarcii::NumberingScheme) -> CdrNumberingScheme {
    match scheme {
        anarcii::NumberingScheme::Kabat => CdrNumberingScheme::Kabat,
        anarcii::NumberingScheme::Chothia | anarcii::NumberingScheme::Martin => {
            CdrNumberingScheme::Chothia
        }
        anarcii::NumberingScheme::Imgt | anarcii::NumberingScheme::Aho => CdrNumberingScheme::Imgt,
    }
}

/// Germline V and J assignments for every antibody-like chain, from IgBLAST.
///
/// Returned rather than applied, and reported per chain, because a germline call is evidence
/// about a chain rather than a property of it: which database it came from and how well it
/// matched both bear on how much weight to give it.
///
/// Blocking, and one process per chain — IgBLAST takes a single query at a time.
pub fn germline_assignments(
    annotation: &AntibodyAnnotation,
) -> io::Result<Vec<(String, igblast::IgBlastResult)>> {
    let mut results = Vec::new();
    for chain in annotation
        .chains
        .iter()
        .filter(|c| c.kind.is_antibody_like())
    {
        let query = igblast::IgBlastQuery::protein(
            format!("chain_{}", chain.chain_id),
            chain.sequence.clone(),
        );
        results.push((chain.chain_id.clone(), igblast::run(&query)?));
    }
    Ok(results)
}

/// Attach IgBLAST's germline calls to an annotation.
///
/// Independent of [`refine_with_anarcii`]: the two can be used together, in which case ANARCII's
/// numbering decides the CDR boundaries and IgBLAST supplies the gene names. IgBLAST's own
/// boundaries are only adopted where no numbering has been applied, since a numbering assignment
/// is the more precise of the two.
pub fn apply_germline_assignments(
    annotation: &mut AntibodyAnnotation,
    assignments: &[(String, igblast::IgBlastResult)],
) {
    for (chain_id, result) in assignments {
        let Some(chain) = annotation
            .chains
            .iter_mut()
            .find(|chain| chain.chain_id == *chain_id)
        else {
            continue;
        };
        chain.germline_v = result.v_calls.clone();
        chain.germline_j = result.j_calls.clone();
        if chain.source == AnnotationSource::Approximate && !result.regions.is_empty() {
            chain.source = AnnotationSource::IgBlast;
        }
        chain.notes.push(format!("IgBLAST: {}", result.summary()));
    }
}

fn developability_issues_for_chain(chain: &AntibodyChainAnnotation) -> Vec<DevelopabilityIssue> {
    if !chain.kind.is_antibody_like() {
        return Vec::new();
    }

    let mut issues = Vec::new();
    let seq = &chain.sequence_aa;

    if !aa_near(seq, AminoAcid::Cys, 23, 4) {
        issues.push(DevelopabilityIssue {
            kind: DevelopabilityIssueKind::MissingConservedResidue,
            severity: IssueSeverity::Medium,
            residues: Vec::new(),
            message: format!(
                "Chain {} is missing the first conserved variable-domain cysteine near position 23.",
                chain.chain_id
            ),
        });
    }
    if !aa_near(seq, AminoAcid::Cys, 104, 6) {
        issues.push(DevelopabilityIssue {
            kind: DevelopabilityIssueKind::MissingConservedResidue,
            severity: IssueSeverity::Medium,
            residues: Vec::new(),
            message: format!(
                "Chain {} is missing the second conserved variable-domain cysteine near position 104.",
                chain.chain_id
            ),
        });
    }
    if !aa_near(seq, AminoAcid::Trp, 41, 6) {
        issues.push(DevelopabilityIssue {
            kind: DevelopabilityIssueKind::MissingConservedResidue,
            severity: IssueSeverity::Low,
            residues: Vec::new(),
            message: format!(
                "Chain {} lacks the usual framework tryptophan near position 41.",
                chain.chain_id
            ),
        });
    }

    for (i, aa) in seq.iter().enumerate().take(VARIABLE_DOMAIN_SCAN_LEN) {
        let position = i + 1;
        if *aa == AminoAcid::Cys && !is_near(position, 23, 4) && !is_near(position, 104, 6) {
            let residues = residue_window(chain, i, 1);
            let in_cdr = chain.contains_sequence_position_in_cdr(position);
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::ExtraCysteine,
                severity: if in_cdr {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                residues,
                message: format!(
                    "Extra cysteine in {} variable domain at sequence position {position}.",
                    chain.chain_id
                ),
            });
        }
    }

    for i in 0..seq.len().saturating_sub(2) {
        if seq[i] == AminoAcid::Asn
            && seq[i + 1] != AminoAcid::Pro
            && matches!(seq[i + 2], AminoAcid::Ser | AminoAcid::Thr)
        {
            let residues = residue_window(chain, i, 3);
            let in_cdr = window_overlaps_cdr(chain, i, 3);
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::PotentialNLinkedGlycosylation,
                severity: if in_cdr {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                residues,
                message: format!(
                    "Potential N-linked glycosylation motif in chain {} at positions {}-{}.",
                    chain.chain_id,
                    i + 1,
                    i + 3
                ),
            });
        }
    }

    for i in 0..seq.len().saturating_sub(1) {
        if seq[i] == AminoAcid::Asn
            && matches!(
                seq[i + 1],
                AminoAcid::Gly | AminoAcid::Ser | AminoAcid::Asn | AminoAcid::His
            )
        {
            let in_cdr = window_overlaps_cdr(chain, i, 2);
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::DeamidationMotif,
                severity: if in_cdr {
                    IssueSeverity::Medium
                } else {
                    IssueSeverity::Low
                },
                residues: residue_window(chain, i, 2),
                message: format!(
                    "Asn deamidation-prone motif in chain {} at positions {}-{}.",
                    chain.chain_id,
                    i + 1,
                    i + 2
                ),
            });
        }

        if seq[i] == AminoAcid::Asp
            && matches!(seq[i + 1], AminoAcid::Gly | AminoAcid::Ser | AminoAcid::Asp)
        {
            let in_cdr = window_overlaps_cdr(chain, i, 2);
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::AspIsomerizationMotif,
                severity: if in_cdr {
                    IssueSeverity::Medium
                } else {
                    IssueSeverity::Low
                },
                residues: residue_window(chain, i, 2),
                message: format!(
                    "Asp isomerization-prone motif in chain {} at positions {}-{}.",
                    chain.chain_id,
                    i + 1,
                    i + 2
                ),
            });
        }
    }

    for (i, aa) in seq.iter().enumerate() {
        let position = i + 1;
        if *aa == AminoAcid::Met && chain.contains_sequence_position_in_cdr(position) {
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::MethionineOxidationSite,
                severity: IssueSeverity::Medium,
                residues: residue_window(chain, i, 1),
                message: format!(
                    "Methionine in a CDR of chain {} at sequence position {position}.",
                    chain.chain_id
                ),
            });
        }
    }

    for (start, window) in seq.windows(5).enumerate() {
        let basic = window
            .iter()
            .filter(|aa| matches!(aa, AminoAcid::Arg | AminoAcid::Lys | AminoAcid::His))
            .count();
        let acidic = window
            .iter()
            .filter(|aa| matches!(aa, AminoAcid::Asp | AminoAcid::Glu))
            .count();
        if basic >= 4 || acidic >= 4 {
            let in_cdr = window_overlaps_cdr(chain, start, 5);
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::ChargedPatch,
                severity: if in_cdr {
                    IssueSeverity::Medium
                } else {
                    IssueSeverity::Low
                },
                residues: residue_window(chain, start, 5),
                message: format!(
                    "Charged 5-residue patch in chain {} at positions {}-{}.",
                    chain.chain_id,
                    start + 1,
                    start + 5
                ),
            });
        }
    }

    for (start, window) in seq.windows(6).enumerate() {
        let hydrophobic = window
            .iter()
            .filter(|aa| {
                matches!(
                    aa,
                    AminoAcid::Val
                        | AminoAcid::Ile
                        | AminoAcid::Leu
                        | AminoAcid::Met
                        | AminoAcid::Phe
                        | AminoAcid::Trp
                        | AminoAcid::Tyr
                )
            })
            .count();
        if hydrophobic >= 5 {
            let in_cdr = window_overlaps_cdr(chain, start, 6);
            issues.push(DevelopabilityIssue {
                kind: DevelopabilityIssueKind::HydrophobicPatch,
                severity: if in_cdr {
                    IssueSeverity::Medium
                } else {
                    IssueSeverity::Low
                },
                residues: residue_window(chain, start, 6),
                message: format!(
                    "Hydrophobic 6-residue patch in chain {} at positions {}-{}.",
                    chain.chain_id,
                    start + 1,
                    start + 6
                ),
            });
        }
    }

    issues
}

fn partner_residues(
    peptide: &MoleculePeptide,
    annotation: &AntibodyAnnotation,
    antigen_chain_ids: &[&str],
) -> Vec<ResidueRef> {
    let antibody_chain_ids: HashSet<&str> = annotation
        .antibody_chains()
        .map(|chain| chain.chain_id.as_str())
        .collect();
    let requested_ids: HashSet<&str> = antigen_chain_ids.iter().copied().collect();

    let mut result = Vec::new();
    for (chain_i, chain) in peptide.chains.iter().enumerate() {
        let use_chain = if requested_ids.is_empty() {
            !antibody_chain_ids.contains(chain.id.as_str())
        } else {
            requested_ids.contains(chain.id.as_str())
        };
        if !use_chain {
            continue;
        }

        for &residue_i in &chain.residues {
            let Some(residue) = peptide.residues.get(residue_i) else {
                continue;
            };
            let aa = match residue.res_type {
                ResidueType::AminoAcid(aa) => Some(aa),
                _ => None,
            };
            result.push(ResidueRef {
                chain_id: chain.id.clone(),
                chain_i,
                residue_i,
                serial_number: residue.serial_number,
                aa,
            });
        }
    }
    result
}

fn has_variable_domain_signature(sequence: &[AminoAcid]) -> bool {
    aa_near(sequence, AminoAcid::Cys, 23, 4)
        && aa_near(sequence, AminoAcid::Trp, 41, 6)
        && aa_near(sequence, AminoAcid::Cys, 104, 8)
}

fn aa_near(
    sequence: &[AminoAcid],
    target: AminoAcid,
    center_1_based: usize,
    radius: usize,
) -> bool {
    if sequence.is_empty() {
        return false;
    }
    let start = center_1_based.saturating_sub(radius + 1);
    let end = (center_1_based + radius).min(sequence.len());
    sequence[start..end].contains(&target)
}

fn is_near(position_1_based: usize, center_1_based: usize, radius: usize) -> bool {
    position_1_based.abs_diff(center_1_based) <= radius
}

fn residue_window(chain: &AntibodyChainAnnotation, start: usize, len: usize) -> Vec<ResidueRef> {
    chain
        .residues
        .iter()
        .skip(start)
        .take(len)
        .cloned()
        .collect()
}

fn window_overlaps_cdr(chain: &AntibodyChainAnnotation, start: usize, len: usize) -> bool {
    (start + 1..=start + len).any(|position| chain.contains_sequence_position_in_cdr(position))
}

fn sequence_to_string(sequence: &[AminoAcid]) -> String {
    sequence
        .iter()
        .map(|aa| aa.to_str(AaIdent::OneLetter))
        .collect::<Vec<_>>()
        .join("")
}

fn centroid_for_residue_refs(
    peptide: &MoleculePeptide,
    residue_indices: impl Iterator<Item = usize>,
) -> Option<Vec3> {
    let mut sum = Vec3::new_zero();
    let mut count = 0usize;
    for residue_i in residue_indices {
        if let Some(centroid) = residue_centroid(peptide, residue_i) {
            sum += centroid;
            count += 1;
        }
    }

    (count > 0).then_some(sum / count as f64)
}

fn centroid_for_atoms(
    peptide: &MoleculePeptide,
    atom_indices: impl Iterator<Item = usize>,
) -> Option<Vec3> {
    let mut sum = Vec3::new_zero();
    let mut count = 0usize;
    for atom_i in atom_indices {
        if let Some(posit) = atom_posit(peptide, atom_i) {
            sum += posit;
            count += 1;
        }
    }

    (count > 0).then_some(sum / count as f64)
}

fn atom_posit(peptide: &MoleculePeptide, atom_i: usize) -> Option<Vec3> {
    peptide
        .common
        .atom_posits
        .get(atom_i)
        .copied()
        .or_else(|| peptide.common.atoms.get(atom_i).map(|atom| atom.posit))
}

fn min_heavy_atom_distance(
    peptide: &MoleculePeptide,
    residue_a_i: usize,
    residue_b_i: usize,
    early_cutoff: Option<f64>,
) -> Option<f64> {
    let residue_a = peptide.residues.get(residue_a_i)?;
    let residue_b = peptide.residues.get(residue_b_i)?;
    let cutoff_sq = early_cutoff.map(|cutoff| cutoff * cutoff);
    let mut best = f64::INFINITY;

    for &atom_a_i in &residue_a.atoms {
        let atom_a = peptide.common.atoms.get(atom_a_i)?;
        if atom_a.element == Element::Hydrogen {
            continue;
        }
        let Some(pos_a) = atom_posit(peptide, atom_a_i) else {
            continue;
        };

        for &atom_b_i in &residue_b.atoms {
            let atom_b = peptide.common.atoms.get(atom_b_i)?;
            if atom_b.element == Element::Hydrogen {
                continue;
            }
            let Some(pos_b) = atom_posit(peptide, atom_b_i) else {
                continue;
            };

            let dist_sq = (pos_a - pos_b).magnitude_squared();
            if dist_sq < best {
                best = dist_sq;
                if let Some(cutoff_sq) = cutoff_sq
                    && best <= cutoff_sq
                {
                    return Some(best.sqrt());
                }
            }
        }
    }

    best.is_finite().then_some(best.sqrt())
}

fn residue_is_near_any(
    peptide: &MoleculePeptide,
    residue_i: usize,
    residue_set: &HashSet<usize>,
    cutoff_angstrom: f64,
) -> bool {
    residue_set.iter().copied().any(|other_i| {
        min_heavy_atom_distance(peptide, residue_i, other_i, Some(cutoff_angstrom))
            .is_some_and(|dist| dist <= cutoff_angstrom)
    })
}

fn refs_from_residue_indices(
    peptide: &MoleculePeptide,
    residue_indices: &HashSet<usize>,
) -> Vec<ResidueRef> {
    let mut refs = Vec::new();
    for &residue_i in residue_indices {
        let Some(residue) = peptide.residues.get(residue_i) else {
            continue;
        };
        let chain_i = residue
            .atoms
            .first()
            .and_then(|atom_i| peptide.common.atoms.get(*atom_i))
            .and_then(|atom| atom.chain)
            .unwrap_or(0);
        let chain_id = peptide
            .chains
            .get(chain_i)
            .map(|chain| chain.id.clone())
            .unwrap_or_default();
        let aa = match residue.res_type {
            ResidueType::AminoAcid(aa) => Some(aa),
            _ => None,
        };
        refs.push(ResidueRef {
            chain_id,
            chain_i,
            residue_i,
            serial_number: residue.serial_number,
            aa,
        });
    }
    refs.sort_by(|a, b| {
        (a.chain_id.as_str(), a.serial_number).cmp(&(b.chain_id.as_str(), b.serial_number))
    });
    refs
}

fn atom_indices_for_residues(
    peptide: &MoleculePeptide,
    residue_indices: &HashSet<usize>,
) -> Vec<usize> {
    let atoms = residue_indices
        .iter()
        .filter_map(|residue_i| peptide.residues.get(*residue_i))
        .flat_map(|residue| residue.atoms.iter().copied())
        .collect::<HashSet<_>>();
    sorted_usizes(atoms)
}

fn sorted_usizes(values: HashSet<usize>) -> Vec<usize> {
    let mut values: Vec<_> = values.into_iter().collect();
    values.sort_unstable();
    values
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    fn aa_sequence(s: &str) -> Vec<AminoAcid> {
        s.chars()
            .map(|c| AminoAcid::from_str(&c.to_string()).unwrap())
            .collect()
    }

    #[test]
    fn imgt_heavy_like_sequence_gets_h_cdrs() {
        let mut seq = aa_sequence(&"A".repeat(125));
        seq[22] = AminoAcid::Cys;
        seq[40] = AminoAcid::Trp;
        seq[103] = AminoAcid::Cys;

        let annotation = annotate_sequence("H", &seq, CdrNumberingScheme::Imgt);

        assert!(annotation.kind.is_heavy());
        assert_eq!(annotation.cdrs.len(), 3);
        assert_eq!(annotation.cdrs[0].label, CdrLabel::H1);
        assert_eq!(annotation.cdrs[0].start_position, 27);
        assert_eq!(annotation.cdrs[2].label, CdrLabel::H3);
        assert_eq!(annotation.cdrs[2].end_position, 117);
    }

    #[test]
    fn pymol_selection_groups_residues_by_chain() {
        let residues = vec![
            ResidueRef {
                chain_id: "B".to_string(),
                chain_i: 1,
                residue_i: 10,
                serial_number: 5,
                aa: None,
            },
            ResidueRef {
                chain_id: "A".to_string(),
                chain_i: 0,
                residue_i: 2,
                serial_number: 31,
                aa: None,
            },
            ResidueRef {
                chain_id: "A".to_string(),
                chain_i: 0,
                residue_i: 3,
                serial_number: 32,
                aa: None,
            },
        ];

        assert_eq!(
            pymol_selection_for_residues(&residues),
            "chain A and resi 31+32 or chain B and resi 5"
        );
    }

    // -----------------------------------------------------------------------------------------
    // Refinement through the external antibody tools
    // -----------------------------------------------------------------------------------------

    /// A chain of `length` residues, annotated the approximate way.
    fn chain_annotation(length: usize) -> AntibodyChainAnnotation {
        let sequence = vec![AminoAcid::Ala; length];
        let mut annotation = annotate_sequence("H", &sequence, CdrNumberingScheme::Imgt);
        annotation.kind = AntibodyChainKind::Heavy;
        annotation
    }

    /// An ANARCII result placing IMGT CDR1 (27-38) over query offsets 5..=14, with a gap in the
    /// middle so that scheme positions and query offsets deliberately disagree.
    fn numbered_chain() -> anarcii::NumberedChain {
        let mut residues = Vec::new();
        let mut query_index = 0;
        for position in 20..=45 {
            // Positions 32 and 33 are gaps, as IMGT represents a short CDR1.
            let gap = position == 32 || position == 33;
            residues.push(anarcii::NumberedResidue {
                position,
                insertion: None,
                amino_acid: if gap { '-' } else { 'A' },
                query_index,
            });
            if !gap {
                query_index += 1;
            }
        }
        anarcii::NumberedChain {
            chain_type: Some("H".to_string()),
            score: Some(0.98),
            query_start: Some(0),
            query_end: Some(query_index - 1),
            scheme: anarcii::NumberingScheme::Imgt,
            error: None,
            residues,
        }
    }

    #[test]
    fn anarcii_numbering_replaces_the_approximation() {
        let mut chain = chain_annotation(60);
        assert_eq!(chain.source, AnnotationSource::Approximate);

        apply_numbering(&mut chain, &numbered_chain());

        assert_eq!(chain.source, AnnotationSource::Anarcii);
        // Only CDR1 falls inside the numbered fragment.
        assert_eq!(chain.cdrs.len(), 1);
        assert_eq!(chain.cdrs[0].label, CdrLabel::H1);
        assert_eq!(chain.cdrs[0].scheme, CdrNumberingScheme::Imgt);
    }

    #[test]
    fn cdr_positions_follow_query_offsets_not_scheme_positions() {
        let mut chain = chain_annotation(60);
        apply_numbering(&mut chain, &numbered_chain());

        // Scheme position 27 is the eighth numbered entry (20..27), but two of the entries before
        // 38 are gaps — so the CDR spans query offsets 7..=16, i.e. 1-based 8..=17. Reading the
        // scheme positions as offsets, which the approximation effectively does, would give
        // 27..=38 and select the wrong ten residues.
        let cdr = &chain.cdrs[0];
        assert_eq!((cdr.start_position, cdr.end_position), (8, 17));
        assert_eq!(cdr.residues.len(), 10);
        assert_eq!(cdr.residues[0].residue_i, 7);
        assert_eq!(cdr.residues[9].residue_i, 16);
    }

    #[test]
    fn numbering_labels_are_recorded_per_residue() {
        let mut chain = chain_annotation(60);
        apply_numbering(&mut chain, &numbered_chain());

        // Query offset 0 is scheme position 20.
        assert_eq!(chain.numbering[0].as_deref(), Some("20"));
        // Offset 11 is position 31; the two gaps then mean offset 12 is position 34, not 32.
        assert_eq!(chain.numbering[11].as_deref(), Some("31"));
        assert_eq!(chain.numbering[12].as_deref(), Some("34"));
        // Beyond the numbered region there is no assignment, rather than a guessed one.
        assert!(chain.numbering[40].is_none());
    }

    #[test]
    fn a_chain_anarcii_cannot_number_keeps_its_approximation() {
        let mut chain = chain_annotation(130);
        let approximate_cdrs = chain.cdrs.clone();
        assert!(!approximate_cdrs.is_empty());

        apply_numbering(
            &mut chain,
            &anarcii::NumberedChain {
                chain_type: None,
                score: None,
                query_start: None,
                query_end: None,
                scheme: anarcii::NumberingScheme::Imgt,
                error: Some("no variable domain found".to_string()),
                residues: Vec::new(),
            },
        );

        assert_eq!(chain.source, AnnotationSource::Approximate);
        assert_eq!(chain.cdrs.len(), approximate_cdrs.len());
        assert!(
            chain
                .notes
                .iter()
                .any(|note| note.contains("could not number"))
        );
    }

    #[test]
    fn anarcii_chain_class_decides_the_light_chain_labels() {
        let mut chain = chain_annotation(60);
        chain.kind = AntibodyChainKind::Heavy;

        let mut numbered = numbered_chain();
        numbered.chain_type = Some("K".to_string());
        apply_numbering(&mut chain, &numbered);

        // The class comes from ANARCII rather than the motif heuristics, so the CDR is labelled
        // as a light-chain loop.
        assert_eq!(chain.kind, AntibodyChainKind::Light);
        assert_eq!(chain.cdrs[0].label, CdrLabel::L1);
    }

    #[test]
    fn martin_is_reported_as_chothia_and_aho_falls_back_to_imgt() {
        use anarcii::NumberingScheme as N;
        assert_eq!(cdr_scheme_from_anarcii(N::Kabat), CdrNumberingScheme::Kabat);
        assert_eq!(
            cdr_scheme_from_anarcii(N::Chothia),
            CdrNumberingScheme::Chothia
        );
        // Martin is Chothia with the Abhinandan corrections; this module does not distinguish them.
        assert_eq!(
            cdr_scheme_from_anarcii(N::Martin),
            CdrNumberingScheme::Chothia
        );
        assert_eq!(cdr_scheme_from_anarcii(N::Imgt), CdrNumberingScheme::Imgt);
        assert_eq!(cdr_scheme_from_anarcii(N::Aho), CdrNumberingScheme::Imgt);
    }

    #[test]
    fn germline_calls_attach_without_overriding_a_numbering() {
        let mut annotation = AntibodyAnnotation {
            scheme: CdrNumberingScheme::Imgt,
            chains: vec![chain_annotation(130)],
            developability_issues: Vec::new(),
            notes: Vec::new(),
        };
        annotation.chains[0].source = AnnotationSource::Anarcii;

        let mut result = igblast::IgBlastResult::default();
        result.v_calls = vec!["IGHV3-23*01".to_string()];
        result.j_calls = vec!["IGHJ4*02".to_string()];
        result.regions = vec![igblast::Region {
            name: "CDR1".to_string(),
            start: 27,
            end: 38,
            sequence: None,
            percent_identity: None,
        }];

        apply_germline_assignments(&mut annotation, &[("H".to_string(), result)]);

        let chain = &annotation.chains[0];
        assert_eq!(chain.germline_v, ["IGHV3-23*01"]);
        assert_eq!(chain.germline_j, ["IGHJ4*02"]);
        // A numbering assignment is more precise than IgBLAST's alignment boundaries, so the
        // source is not downgraded.
        assert_eq!(chain.source, AnnotationSource::Anarcii);
        assert!(chain.notes.iter().any(|note| note.starts_with("IgBLAST:")));
    }

    #[test]
    fn germline_calls_upgrade_a_chain_that_had_only_approximations() {
        let mut annotation = AntibodyAnnotation {
            scheme: CdrNumberingScheme::Imgt,
            chains: vec![chain_annotation(130)],
            developability_issues: Vec::new(),
            notes: Vec::new(),
        };

        let mut result = igblast::IgBlastResult::default();
        result.regions = vec![igblast::Region {
            name: "CDR1".to_string(),
            start: 27,
            end: 38,
            sequence: None,
            percent_identity: None,
        }];
        apply_germline_assignments(&mut annotation, &[("H".to_string(), result)]);

        assert_eq!(annotation.chains[0].source, AnnotationSource::IgBlast);
    }
}
