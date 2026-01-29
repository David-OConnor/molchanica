//! Related to finding molecules which bind to a target (protein or pocket)
//!
//! Set up a "Bio misc" folder with a pile of datasets you can download
//
// DBs: PDBbind: RCSB PDB binding affinity annotations (CHekc the API)
// [CASF-2016 : PDBBind scoring](https://www.pdbbind-plus.org.cn/casf)
//
// Poseless:
// - [BindingDB](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)
// - [ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
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
