//! Measures properties of the molecules in the human body; relevant to drug development.
//!
//! Potential data sources and other resources
//! [Therapeutics Data Commons](https://tdcommons.ai/) [Arxiv from 2021](https://arxiv.org/pdf/2102.09548)
//! [Aqueous Solubility Data Curation](https://github.com/mcsorkun/AqSolDB) [Paper, 2019](https://www.nature.com/articles/s41597-019-0151-1)
//! [PK-DB](https://pk-db.com/) [Paper, 2020](https://academic.oup.com/nar/article/49/D1/D1358/5957165?login=false)

mod sol_infer;
mod sol_train;
mod solubility;

use crate::{mol_characterization::MolCharacterization, molecules::small::MoleculeSmall};

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn estimate_tpsa(ch: &MolCharacterization) -> f32 {
    // Very rough: hetero atoms + HBA/HBD drive PSA up.
    let o = ch.oxygen.len() as f32;
    let n = ch.nitrogen.len() as f32;
    let s = ch.sulfur.len() as f32;
    let p = ch.phosphorus.len() as f32;
    let hba = ch.h_bond_acceptor.len() as f32;
    let hbd = ch.h_bond_donor.len() as f32;

    // Ballpark contributions; clamp to a sane range.
    (17.0 * o + 12.0 * n + 25.0 * s + 13.0 * p + 1.5 * hba + 2.0 * hbd).clamp(0.0, 300.0)
}

fn estimate_log_p(ch: &MolCharacterization) -> f32 {
    // Very rough: carbons/halogens increase logP; hetero/HBD/HBA decrease it; rings nudge it up.
    let c = ch.num_carbon as f32;
    let hetero = ch.num_hetero_atoms as f32;
    let hal = ch.halogen.len() as f32;
    let hba = ch.h_bond_acceptor.len() as f32;
    let hbd = ch.h_bond_donor.len() as f32;
    let rings = ch.rings.len() as f32;
    let arom = ch.num_aromatic_atoms as f32;

    let mut lp: f32 = 0.04 * c + 0.35 * rings + 0.02 * arom + 0.55 * hal
        - 0.25 * hetero
        - 0.18 * hba
        - 0.25 * hbd;
    lp = lp.clamp(-2.0, 8.0);
    lp
}

/// Estimates of how the molecule, in drug form, acts in the human body.
/// https://en.wikipedia.org/wiki/Pharmacokinetics
#[derive(Clone, Debug, Default)]
pub struct Pharmacokinetics {
    /// LogS, where S is the aqueous solubility in mol/L (or M).
    pub solubility_water: f32,
    pub solubility_lipid: f32,
    pub blood_brain_barrier: f32,
    pub gut_wall: f32,
    pub liver_toxicity: f32,
    // todo: Other toxicities
    pub half_life_blood_stream: f32, // seconds (?)
    pub breakdown_products: Vec<MoleculeSmall>,
    /// the process of active pharmaceutical ingredients (API) separating from its pharmaceutical
    /// formulation.[3][4] See also IVIVC.
    pub liberation: f32,
    /// the process of a drug entering into systemic circulation from the site of administration.
    pub absorption: f32,
    ///  the dispersion or dissemination of substances throughout the fluids and tissues of the body.
    pub distribution: f32,
    /// (or biotransformation, or inactivation) – the chemical reactions of the drug and
    /// irreversible breakdown into metabolites (e.g. by metabolic enzymes such as cytochrome P450
    /// or glucuronosyltransferase enzymes).
    pub metabolism: f32,
    /// the removal of the substance or metabolites from the body. In rare cases, some drugs
    /// irreversibly accumulate in body tissue.[5]
    pub excretion: f32,
}

impl Pharmacokinetics {
    pub fn new(mol: &MoleculeSmall) -> Self {
        let ch = match mol.characterization.as_ref() {
            Some(ch) => ch,
            None => return Self::default(),
        };

        let mw = ch.mol_weight.max(1.0);
        let rot = ch.rotatable_bonds.len() as f32;
        let hbd = ch.h_bond_donor.len() as f32;
        let hba = ch.h_bond_acceptor.len() as f32;

        let tpsa = ch
            .topological_polar_surface_area
            .unwrap_or_else(|| estimate_tpsa(ch));

        let log_p = ch.calc_log_p.unwrap_or_else(|| estimate_log_p(ch));

        let abs_net_charge = ch.net_partial_charge.map(|q| q.abs()).unwrap_or(0.0);

        // If it looks appreciably ionized, water solubility tends to go up and BBB tends to go down.
        let ionized = sigmoid((abs_net_charge - 0.25) * 6.0); // 0..1
        let ion_penalty = 1.0 - ionized;

        // Water solubility (0..1): prefer lower logP, higher polarity, lower MW; boost if ionized.
        let solubility_water = clamp01(
            0.55 * sigmoid((1.0 - log_p) * 1.2)
                + 0.30 * sigmoid((140.0 - tpsa) / 35.0)
                + 0.15 * sigmoid((550.0 - mw) / 140.0)
                + 0.25 * ionized,
        );

        // Lipid solubility (0..1): prefer moderate/high logP, not too polar.
        let solubility_lipid = clamp01(
            0.70 * sigmoid((log_p - 1.2) * 1.1) * sigmoid((110.0 - tpsa) / 30.0)
                + 0.20 * sigmoid((log_p - 2.0) * 0.8)
                + 0.10 * sigmoid((mw - 120.0) / 250.0),
        );

        // BBB permeability proxy (0..1): favors logP ~ 1-3, low tPSA, low HBD, lower MW, neutral charge.
        let bbb_logp_window = sigmoid((log_p - 1.0) * 1.5) * sigmoid((3.5 - log_p) * 1.5);
        let blood_brain_barrier = clamp01(
            bbb_logp_window
                * sigmoid((90.0 - tpsa) / 18.0)
                * sigmoid((450.0 - mw) / 70.0)
                * sigmoid((1.5 - hbd) / 0.7)
                * sigmoid((10.0 - (hba + hbd)) / 2.5)
                * ion_penalty,
        );

        // Gut wall permeability / oral absorption proxy (0..1): Lipinski-ish + rotatable bonds.
        let gut_wall = {
            let s_mw = sigmoid((500.0 - mw) / 85.0);
            let s_logp = sigmoid((5.0 - log_p) / 1.0);
            let s_hbd = sigmoid((5.0 - hbd) / 1.0);
            let s_hba = sigmoid((10.0 - hba) / 1.6);
            let s_tpsa = sigmoid((140.0 - tpsa) / 25.0);
            let s_rot = sigmoid((10.0 - rot) / 2.0);

            clamp01((s_mw + s_logp + s_hbd + s_hba + s_tpsa + s_rot) / 6.0)
        };

        // Very heuristic liver-toxicity risk proxy (0..1): lipophilicity + aromaticity + size + halogens.
        let liver_toxicity = {
            let arom_atoms = ch.num_aromatic_atoms as f32;
            let hal = ch.halogen.len() as f32;

            clamp01(
                0.45 * sigmoid((log_p - 3.0) / 0.7)
                    + 0.30 * sigmoid((arom_atoms - 12.0) / 4.0)
                    + 0.15 * sigmoid((mw - 450.0) / 80.0)
                    + 0.10 * sigmoid((hal - 2.0) / 1.0),
            )
        };

        // Half-life guess (seconds): wide, heuristic mapping from “stickiness” (lipophilicity, size) vs polarity/clearance.
        let half_life_blood_stream = {
            let arom_atoms = ch.num_aromatic_atoms as f32;

            let mut t_half_hours = 2.0
                + 10.0 * sigmoid((log_p - 2.0) / 1.0)
                + 8.0 * sigmoid((mw - 350.0) / 90.0)
                + 3.0 * sigmoid((arom_atoms - 10.0) / 4.0)
                - 8.0 * sigmoid((tpsa - 120.0) / 20.0)
                - 4.0 * sigmoid(((hba + hbd) - 10.0) / 2.0)
                + 2.0 * ionized;

            t_half_hours = t_half_hours.clamp(0.5, 96.0);
            t_half_hours * 3600.0
        };

        // ADME-ish scalars (0..1), stitched from the above.
        let liberation = solubility_water;
        let absorption = clamp01(liberation * 0.65 + gut_wall * 0.55);
        let distribution = clamp01(
            0.55 * solubility_lipid + 0.25 * blood_brain_barrier + 0.20 * (1.0 - solubility_water),
        );
        let metabolism = {
            let arom_atoms = ch.num_aromatic_atoms as f32;
            clamp01(
                0.50 * sigmoid((log_p - 2.5) / 0.8)
                    + 0.25 * sigmoid((rot - 6.0) / 2.0)
                    + 0.25 * sigmoid((arom_atoms - 10.0) / 4.0),
            )
        };
        let excretion = clamp01(
            0.60 * solubility_water + 0.20 * ionized + 0.20 * sigmoid((tpsa - 90.0) / 20.0),
        );

        Self {
            solubility_water,
            solubility_lipid,
            blood_brain_barrier,
            gut_wall,
            liver_toxicity,
            half_life_blood_stream,
            breakdown_products: Vec::new(),
            liberation,
            absorption: absorption,
            distribution,
            metabolism,
            excretion,
        }
    }
}

/// PDB ID
fn get_common_drug_rel_proteins() -> Vec<String> {
    vec![]
}
