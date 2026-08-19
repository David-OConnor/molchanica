//! Related to finding molecules which bind to a target (protein or pocket)
//!
//! Set up a "Bio misc" folder with a pile of datasets you can download
//!
//! # Datasets that are readable now
//!
//! PDBbind is implemented: [`super::pdbbind`] reads a locally downloaded release directly, with
//! no Python and no subprocess. It gives an entry's measured affinity (Kd/Ki/IC50, with bounded
//! and approximate measurements flagged as such) alongside paths to the protein, the pocket, and
//! the ligand in formats Molchanica already opens.
//!
//! ```no_run
//! use molchanica::drug_design::pdbbind::{self, Subset};
//!
//! // One complex.
//! if let Some(entry) = pdbbind::find("1a30", Subset::Refined)? {
//!     println!("{:?} {:?}", entry.affinity, entry.ligand());
//! }
//!
//! // Or the whole refined set, filtered to clean regression targets: exact measurements of a
//! // binding constant, excluding IC50s and bounded values, which are not comparable.
//! let training: Vec<_> = pdbbind::entries(Subset::Refined)?
//!     .into_iter()
//!     .filter(|entry| {
//!         entry.affinity.as_ref().is_some_and(|a| a.is_regression_quality())
//!     })
//!     .collect();
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! Point it at a release with `MOLCHANICA_PDBBIND_ROOT`, or unpack one into
//! `<data dir>/molchanica/datasets/pdbbind`. Nothing downloads it: PDBbind+ is distributed under
//! registration, free for academic use and paid for commercial use, so the copy has to be one the
//! user obtained under their own agreement.
//
// [CASF-2016 : PDBBind scoring](https://www.pdbbind-plus.org.cn/casf)
//
// Poseless:
// - [BindingDB](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)
// - [ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
//
// - [ Zinc/CartBlanche](https://cartblanche.docking.org/): Huge molecule library
//
// Pocket-based:
// - [BioLiP2](https://www.aideepmed.com/BioLiP/download.html)
// - [sc-PDB](https://drugdesign.unistra.fr/scPDB/)
// - Pocketome: Under construction?
//
// MOre:
// - D) Virtual screening benchmarks (actives + decoys)
// These are for evaluating screening/ranking, not affinity regression.
// DUD-E: classic docking benchmark with actives/decoys for many targets.
//
// DEKOIS 2.0: challenging docking benchmark sets; often used to test screening workflows.
//
// LIT-PCBA: designed as an “unbiased” benchmark, but there are recent audits reporting leakage/redundancy issues—so treat it carefully.
//
// [CrossDocked2020](https://bits.csb.pitt.edu/files/crossdock2020/)
// Huge set of docked poses across related pockets; widely used for structure-based ML on poses.
//
// PLINDER: a newer large-scale protein–ligand interaction dataset + splits/tooling aimed at more realistic generalization.
//
//
// Make or update a module, and include links or names for each.
//
// Build an algo to quickly score a ligand's ability to fit into a pocket... i.e. docking. Do it!
