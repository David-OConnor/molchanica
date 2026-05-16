//! Analytic properties for water solubility. We will likely remove in favor of ML and MD approaches.

use crate::properties::mol_characterization::MolCharacterization;

#[derive(Clone, Debug)]
pub(in crate::properties) struct WaterSolAnalyticProps {
    pub water_affinity: f32,
    /// Higher means the molecule is expected to be easier to hydrate or dissolve in water.
    pub water_solubility: f32,
    /// Fast descriptor-only component of `water_affinity_score`.
    pub property_water_affinity_score: f32,
    /// Higher means the molecule has hydrophobic or size-related resistance to hydration.
    pub hydration_penalty: f32,
    pub h_bond_capacity: f32,
    pub polarity_score: f32,
    pub hydrophobic_penalty: f32,
    pub charge_affinity: f32,
    pub tpsa_score: f32,
}

pub(in crate::properties) fn property_terms(char: &MolCharacterization) -> WaterSolAnalyticProps {
    let donors = char.h_bond_donor.len() as f32;
    let acceptors = char.h_bond_acceptor.len() as f32;
    let h_bond_capacity = donors + acceptors;

    let tpsa_score = (char.tpsa_ertl / 90.0).clamp(0.0, 2.5);
    let geom_psa_score = (char.psa_topo / 120.0).clamp(0.0, 1.5);
    let hetero_score =
        (char.num_hetero_atoms as f32 / char.num_heavy_atoms.max(1) as f32).clamp(0.0, 1.0);

    let polarity_score = tpsa_score + geom_psa_score * 0.35 + hetero_score * 0.40;
    let charge_affinity = char.abs_partial_charge_sum.unwrap_or_default() * 0.50
        + char.net_partial_charge.unwrap_or_default().abs() * 0.40;

    let hydrophobic_penalty = (char.log_p.max(0.0) * 0.35
        + (char.greasiness / 6.0).clamp(0.0, 1.5) * 0.40
        + char.hydrophobic_carbon.len() as f32 * 0.015)
        .clamp(0.0, 3.5);

    let size_penalty = (char.mol_weight / 550.0).clamp(0.0, 1.5) * 0.35
        + (char.molar_refractivity / 120.0).clamp(0.0, 1.5) * 0.15;
    let flexibility_penalty =
        char.rotatable_bonds.len() as f32 * 0.04 + (char.flexibility / 12.0).clamp(0.0, 1.2);

    let donor_score = donors.min(8.0) * 0.30;
    let acceptor_score = acceptors.min(10.0) * 0.18;

    let water_affinity = (polarity_score + donor_score + acceptor_score + charge_affinity
        - hydrophobic_penalty * 0.45
        - flexibility_penalty * 0.08)
        .max(0.0);

    let hydration_penalty = (hydrophobic_penalty + size_penalty + flexibility_penalty * 0.05
        - polarity_score * 0.30)
        .max(0.0);

    let water_solubility = (water_affinity - hydration_penalty).max(0.0);

    WaterSolAnalyticProps {
        water_affinity,
        water_solubility,
        hydration_penalty,
        h_bond_capacity,
        polarity_score,
        hydrophobic_penalty,
        charge_affinity,
        tpsa_score,
        property_water_affinity_score: 0., // todo??
    }
}
