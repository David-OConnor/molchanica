//! Saturation-mutagenesis ΔΔG scanning, natively.
//!
//! Answers "which mutations would this protein tolerate, and which would destabilize it?" for
//! every position and every substitution at once. That is the question behind stability
//! engineering, affinity maturation, and developability triage, and it is the natural companion to
//! the inverse-folding designs in [`lib_::external_tools::mpnn`]: MPNN proposes sequences, this
//! scores every single-point change against the structure you already have.
//!
//! # Why this one is native
//!
//! [ProteinMPNN-ddG](https://github.com/PeptoneLtd/proteinmpnn_ddg) is the method, and its
//! headline property is that the whole scan is one forward pass rather than one per position. But
//! it is built on JAX with CUDA 12 wheels, which are published for Linux only, so it could not be
//! one of the installable tools. The network underneath is small and the arithmetic is ordinary,
//! so it is implemented directly in [`mpnn`] instead — which also removes the Python dependency
//! from a capability that belongs in the middle of a workflow rather than at the end of one.
//!
//! # What the numbers mean
//!
//! ΔΔG here is the model's log-likelihood difference between the mutant and wild-type residue at a
//! position, given the backbone:
//!
//! ```text
//! ΔΔG(i, mutant) = −(log p(mutant | structure) − log p(wild type | structure))
//! ```
//!
//! Positive means destabilizing, matching the experimental convention. It is in log-likelihood
//! units, not kcal/mol: it ranks substitutions well, and does not claim a calorimetric value.
//! [`DdgScan::correlation_scale`] records the linear factor that maps it onto kcal/mol if the user
//! has calibrated one against their own data.

pub mod mpnn;

use std::{io, path::PathBuf};

use bio_files::ResidueType;
use mol_defs::molecules::{AtomRole, peptide::MoleculePeptide};
use na_seq::{AaIdent, AminoAcid};

use crate::{
    external_tools::Tool,
    therapeutic_misc::ddg::mpnn::{ALPHABET, Backbone, ProteinMpnnWeights},
};

/// The twenty proteinogenic amino acids, in the alphabet's own order. `X` is excluded: it is the
/// unknown class, not a residue anyone mutates to.
pub const MUTABLE: usize = 20;

/// One position's row of the scan.
#[derive(Clone, Debug)]
pub struct PositionScan {
    /// Index into the peptide's `residues`, so a caller can select or colour the structure.
    pub residue_index: usize,
    /// Residue number as the structure numbers it.
    pub residue_number: u32,
    pub chain_id: String,
    pub wild_type: AminoAcid,
    /// ΔΔG for each of the twenty substitutions, in [`ALPHABET`] order. The wild-type entry is
    /// zero by construction.
    pub ddg: [f32; MUTABLE],
    /// The model's log-probability of the wild-type residue. A very low value flags a position
    /// whose native identity the model does not explain — often a functional residue held in
    /// place by something the backbone alone does not show, so its ΔΔG row should be read with
    /// more caution than the rest.
    pub wild_type_log_probability: f32,
}

impl PositionScan {
    /// ΔΔG for one substitution.
    pub fn ddg_for(&self, mutant: AminoAcid) -> Option<f32> {
        alphabet_index(mutant).map(|index| self.ddg[index])
    }

    /// The substitutions the model prefers to the wild type, most favourable first.
    ///
    /// These are the candidates a stabilizing-mutation campaign starts from.
    pub fn stabilizing(&self) -> Vec<(AminoAcid, f32)> {
        let mut candidates: Vec<_> = (0..MUTABLE)
            .filter(|index| self.ddg[*index] < 0.0)
            .filter_map(|index| amino_acid_at(index).map(|aa| (aa, self.ddg[index])))
            .collect();
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        candidates
    }

    /// Mean ΔΔG over all substitutions: how constrained this position is overall.
    ///
    /// High values mark positions the structure will not tolerate changing — a buried core, or a
    /// tightly packed interface — which is what to hold fixed when designing.
    pub fn constraint(&self) -> f32 {
        self.ddg.iter().sum::<f32>() / MUTABLE as f32
    }

    /// `A97G` style label for one substitution.
    pub fn mutation_label(&self, mutant: AminoAcid) -> String {
        format!(
            "{}{}{}",
            self.wild_type.to_str(AaIdent::OneLetter),
            self.residue_number,
            mutant.to_str(AaIdent::OneLetter)
        )
    }
}

/// A whole scan.
#[derive(Clone, Debug)]
pub struct DdgScan {
    pub positions: Vec<PositionScan>,
    /// Multiply a ΔΔG by this to read it as kcal/mol. `None` until calibrated; see the module docs.
    pub correlation_scale: Option<f32>,
}

impl DdgScan {
    /// Every substitution, sorted most stabilizing first.
    ///
    /// Capped because a 300-residue protein has 5,700 of them and only the head of that list is
    /// ever acted on.
    pub fn best_mutations(&self, limit: usize) -> Vec<(&PositionScan, AminoAcid, f32)> {
        let mut all: Vec<_> = self
            .positions
            .iter()
            .flat_map(|position| {
                (0..MUTABLE).filter_map(move |index| {
                    let aa = amino_acid_at(index)?;
                    (aa != position.wild_type).then_some((position, aa, position.ddg[index]))
                })
            })
            .collect();
        all.sort_by(|a, b| a.2.total_cmp(&b.2));
        all.truncate(limit);
        all
    }

    /// Positions ordered by how constrained they are, most constrained first.
    pub fn most_constrained(&self, limit: usize) -> Vec<&PositionScan> {
        let mut positions: Vec<_> = self.positions.iter().collect();
        positions.sort_by(|a, b| b.constraint().total_cmp(&a.constraint()));
        positions.truncate(limit);
        positions
    }

    /// The scan as a tab-separated table, for export.
    pub fn to_tsv(&self) -> String {
        let mut out = String::from("chain\tposition\twild_type");
        for index in 0..MUTABLE {
            out.push('\t');
            out.push(ALPHABET[index]);
        }
        out.push('\n');

        for position in &self.positions {
            out.push_str(&format!(
                "{}\t{}\t{}",
                position.chain_id,
                position.residue_number,
                position.wild_type.to_str(AaIdent::OneLetter)
            ));
            for value in position.ddg {
                out.push_str(&format!("\t{value:.4}"));
            }
            out.push('\n');
        }
        out
    }
}

fn alphabet_index(aa: AminoAcid) -> Option<usize> {
    let letter = aa.to_str(AaIdent::OneLetter).chars().next()?;
    ALPHABET[..MUTABLE].iter().position(|c| *c == letter)
}

fn amino_acid_at(index: usize) -> Option<AminoAcid> {
    use AminoAcid::*;
    // In ALPHABET order: A C D E F G H I K L M N P Q R S T V W Y.
    const ORDER: [AminoAcid; MUTABLE] = [
        Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Lys, Leu, Met, Asn, Pro, Gln, Arg, Ser, Thr, Val,
        Trp, Tyr,
    ];
    ORDER.get(index).copied()
}

/// Where a converted ProteinMPNN checkpoint is expected.
///
/// Beside the ProteinMPNN checkout the installer already creates, since that is where the source
/// `.pt` lives and where `scripts/convert_mpnn_weights.py` writes its output.
pub fn weights_path() -> Option<PathBuf> {
    if let Some(configured) = std::env::var_os("MOLCHANICA_MPNN_WEIGHTS") {
        return Some(PathBuf::from(configured));
    }
    Tool::ProteinMpnn
        .spec()
        .bundle_root()
        .map(|root| root.join("converted/v_48_020.mcnn"))
}

/// Load the network once, so a caller scanning several structures pays for it once.
pub fn load_weights() -> io::Result<ProteinMpnnWeights> {
    let path = weights_path().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no location is known for the ProteinMPNN weights",
        )
    })?;
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "converted ProteinMPNN weights were not found at {}. Install ProteinMPNN, then \
                 run scripts/convert_mpnn_weights.py to convert its checkpoint.",
                path.display()
            ),
        ));
    }
    mpnn::load_weights(&path)
}

/// Extract the backbone the network runs on.
///
/// Residues missing any of N, CA, C, or O are skipped rather than filled in: the edge features are
/// built from all four plus a virtual Cβ derived from three of them, so a guessed atom would
/// propagate into every distance feature for that residue and its neighbours.
pub fn backbone_from_peptide(mol: &MoleculePeptide) -> (Backbone, Vec<PositionMetadata>) {
    let mut backbone = Backbone::default();
    let mut metadata = Vec::new();

    for (chain_number, chain) in mol.chains.iter().enumerate() {
        for &residue_index in &chain.residues {
            let Some(residue) = mol.residues.get(residue_index) else {
                continue;
            };
            let ResidueType::AminoAcid(wild_type) = residue.res_type else {
                continue;
            };

            let mut n = None;
            let mut ca = None;
            let mut c = None;
            let mut o = None;
            for &atom_index in &residue.atoms {
                let Some(atom) = mol.common.atoms.get(atom_index) else {
                    continue;
                };
                let position = [
                    atom.posit.x as f32,
                    atom.posit.y as f32,
                    atom.posit.z as f32,
                ];
                match atom.role {
                    Some(AtomRole::N_Backbone) => n = Some(position),
                    Some(AtomRole::C_Alpha) => ca = Some(position),
                    Some(AtomRole::C_Prime) => c = Some(position),
                    Some(AtomRole::O_Backbone) => o = Some(position),
                    _ => {}
                }
            }
            let (Some(n), Some(ca), Some(c), Some(o)) = (n, ca, c, o) else {
                continue;
            };

            backbone.n.push(n);
            backbone.ca.push(ca);
            backbone.c.push(c);
            backbone.o.push(o);
            backbone.residue_index.push(residue.serial_number as i32);
            backbone.chain_index.push(chain_number as i32);
            metadata.push(PositionMetadata {
                residue_index,
                residue_number: residue.serial_number,
                chain_id: chain.id.clone(),
                wild_type,
            });
        }
    }

    (backbone, metadata)
}

/// What the scan needs to report a position that the network itself does not carry.
#[derive(Clone, Debug)]
pub struct PositionMetadata {
    pub residue_index: usize,
    pub residue_number: u32,
    pub chain_id: String,
    pub wild_type: AminoAcid,
}

/// Scan every position of a peptide in one forward pass.
///
/// Blocking and compute-heavy — seconds for a few hundred residues — so run it on a worker thread.
pub fn scan(mol: &MoleculePeptide, weights: &ProteinMpnnWeights) -> io::Result<DdgScan> {
    let (backbone, metadata) = backbone_from_peptide(mol);
    if backbone.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "this molecule has no amino-acid residues with a complete backbone",
        ));
    }

    let log_probs = mpnn::forward(weights, &backbone)?;
    Ok(build_scan(&log_probs, &metadata))
}

/// Turn per-position log-probabilities into ΔΔG rows.
///
/// Split out from [`scan`] so the conversion — which is where the sign convention and the
/// wild-type reference live — is testable without a network or a structure.
fn build_scan(log_probs: &mpnn::LogProbabilities, metadata: &[PositionMetadata]) -> DdgScan {
    let positions = metadata
        .iter()
        .enumerate()
        .take(log_probs.length)
        .map(|(index, meta)| {
            let row = log_probs.position(index);
            let wild_type_log_probability = alphabet_index(meta.wild_type)
                .map(|wt| row[wt])
                // A residue outside the twenty (selenocysteine, say) has no wild-type reference,
                // so its row is measured against the unknown class instead of being dropped.
                .unwrap_or(row[ALPHABET.len() - 1]);

            let mut ddg = [0.0f32; MUTABLE];
            for (mutant, value) in ddg.iter_mut().enumerate() {
                // Negated so that positive means destabilizing, as experimentalists write it.
                *value = -(row[mutant] - wild_type_log_probability);
            }

            PositionScan {
                residue_index: meta.residue_index,
                residue_number: meta.residue_number,
                chain_id: meta.chain_id.clone(),
                wild_type: meta.wild_type,
                ddg,
                wild_type_log_probability,
            }
        })
        .collect();

    DdgScan {
        positions,
        correlation_scale: None,
    }
}

/// Replay the reference forward pass `scripts/convert_mpnn_weights.py` recorded, and report the
/// largest disagreement with it.
///
/// This is how the native implementation is checked against upstream. The converter runs the real
/// ProteinMPNN on a fixed synthetic backbone and stores both the inputs and the resulting
/// log-probabilities; this feeds the same inputs through [`mpnn::forward`] and compares. Anything
/// above about 1e-3 means a layer here does not match the one it mirrors.
pub fn verify(reference_path: &std::path::Path) -> io::Result<f32> {
    let file = mpnn::TensorFile::load(reference_path)?;
    let backbone = file.reference_backbone()?;
    let expected = file.reference_log_probs()?;

    let weights = load_weights()?;
    let actual = mpnn::forward(&weights, &backbone)?;

    if actual.data.len() != expected.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "the reference has {} log-probabilities but this implementation produced {}",
                expected.len(),
                actual.data.len()
            ),
        ));
    }
    Ok(actual
        .data
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max))
}
