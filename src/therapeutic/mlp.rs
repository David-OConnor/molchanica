//! Code specific to our MLP neural net: This handles features of a molecule as a whole; it is not
//! part of a GNN. Molecular weight, counts of various types of atoms, total net charge etc.

use std::io;

use bio_files::md_params::ForceFieldParams;

use crate::{
    molecules::{
        conformers::{
            CONFORMER_MOTION_HIST_FEATS, CONFORMER_SUMMARY_FEATS, Conformer, resolve_conformer,
        },
        small::MoleculeSmall,
    },
    properties::{
        crystal,
        crystal::{CrystalDataMdProperties, CrystalEstimateSource},
    },
};

// Note: We can make variants of this A/R tuned to specific inference items. For now, we are using
// a single set of features for all targets.
/// Extract molecule-level features from a molecule that are relevant for inferring the target parameter. We use this
/// in both training and inference workflows.
///
/// We avoid features that may be more robustly represented by GNNs. For example, the count of rings,
/// functional groups, and H bond donors/acceptors.
pub(in crate::therapeutic) fn mlp_feats_from_mol(
    mol: &MoleculeSmall,
    ff_params: &ForceFieldParams,
    conformation_enabled: bool,
) -> io::Result<Vec<f32>> {
    let conformer = if conformation_enabled {
        resolve_conformer(mol, ff_params)
    } else {
        None
    };
    mlp_feats_from_mol_with_conformer(mol, conformer.as_deref(), conformation_enabled)
}

pub(in crate::therapeutic) fn mlp_feats_from_mol_with_conformer(
    mol: &MoleculeSmall,
    conformer: Option<&Conformer>,
    conformation_enabled: bool,
) -> io::Result<Vec<f32>> {
    let Some(c) = &mol.characterization else {
        return Err(io::Error::other("Missing mol characterization"));
    };

    // Helper to compress large ranges (Log1p)
    // We use abs() to handle potential negative LogP inputs safely if you apply it there,
    // though usually we only apply this to Counts and Weights.
    let ln = |x: f32| (x + 1.0).ln();

    // ----

    // We are generally apply ln to values that can be "large".
    // Note: We do seem to get better results using ln values.

    // todo: Many of these are suspect.

    // Ring count: Pos
    // Function groups: Pos
    // Valence: Neg
    // c.rings.len() as f32 * 6. / c.num_atoms as f32: Pos
    // Ring count: Pos
    // Wiener index: Neg impact
    // Mol weight: neg impact
    // Num bonds: Positive impact
    // Rot bond count: Positive impact
    // ln(c.psa_topo / c.asa_topo): Pos
    // psa topo: Pos
    // SAS topo: Big pos
    // Num heavy: pos
    // Het: Pos
    // Halogen: Pos
    // Volume: Pos (big)

    // -----

    // todo: experimenting with cryhstral data. Here may be a temp place to compute it.
    // todo: is thsi being re-computed for each property?
    let crystal_data = crystal::estimate_from_properties(mol).unwrap();

    let mut result = vec![
        ln(c.num_atoms as f32),
        ln(c.num_bonds as f32),
        // ln(c.mol_weight), // mol weight might make results slightly worse.
        // ln(c.num_heavy_atoms as f32),
        ln(c.num_heavy_atoms as f32),
        // c.h_bond_acceptor.len() as f32,
        // c.h_bond_donor.len() as f32,
        c.num_hetero_atoms as f32,
        c.halogen.len() as f32,
        c.rotatable_bonds.len() as f32,
        c.flexibility / 4.,
        // Note: 2026-05-07: Despite including a component-based GNN which includes functional groups,
        // it appears that adding these explicitly here improves results slightly.
        c.amines.len() as f32,
        c.amides.len() as f32,
        // -    with just the above: 0.333
        c.carbonyl.len() as f32,
        c.hydroxyl.len() as f32,
        // with just the above (cum): 0.316
        c.carboxylate.len() as f32,
        c.sulfonamide.len() as f32,
        c.sulfonimide.len() as f32,
        // with just the above (cum): 0.309
        // c.num_valence_elecs as f32,
        // c.num_rings_aromatic as f32,
        // Valence + aro: .313 (tiny regression). Aromatic only: 0.320. So, let's not use either.
        // with just the above (cum): ___
        c.num_rings_saturated as f32,
        c.num_rings_aliphatic as f32,
        // with just the above (cum): 0.306
        // c.rings.len() as f32,
        // with just the above (cum): 0.311 (Slightly worse when adding rings.)
        c.log_p,
        c.molar_refractivity,
        ln(c.psa_topo),
        ln(c.asa_topo),
        ln(c.volume),
        // The ratio of surface area to volume on its own may be useful.

        // Note: In one test, adding asa/volume improved results slightly. And ln(asa/vol) produced
        // slightly better results than without ln.
        ln(c.asa_topo / c.volume),
        ln(c.psa_topo / c.asa_topo),
        // ln(c.wiener_index.unwrap_or(0) as f32),
        c.rings.len() as f32 * 6. / c.num_atoms as f32, // todo temp
        // ln(c.greasiness),

        // pub struct CrystalData {
        //     pub source: CrystalEstimateSource,
        //     pub self_affinity_score: f32,
        //     /// Higher means stronger crystal/self-binding pressure against dissolution.
        //     pub crystal_solubility_penalty: f32,
        //     /// Fast descriptor-only component of `self_affinity_score`.
        //     pub property_self_affinity_score: f32,
        //     /// Fast proxy for water affinity; useful as a competing term in solubility estimates.
        //     pub water_affinity_proxy: f32,
        //     pub h_bond_capacity: f32,
        //     pub hydrophobicity: f32,
        //     pub aromatic_stacking_propensity: f32,
        //     pub flexibility_penalty: f32,
        //     pub md_properties: Option<CrystalDataMdProperties>,
        // }

        // todo: Ideally, we only use these cyrstal data things for specific
        // todo properties, e.g. solubility
        // todo: It seems `flexibility_penalty` and `hydrophobicity` have
        // todo a notable negative correlation with solubility. But a very noisy one.
        crystal_data.flexibility_penalty,
        crystal_data.hydrophobicity,
        // crystal_data.crystal_solubility_penalty,
        // crystal_data.property_self_affinity_score,
        // crystal_data.water_affinity_proxy,
        // todo: Experimenting with crystal data
    ];

    if conformation_enabled {
        let conformer_features = if let Some(conformer) = conformer {
            conformer.summary_features().to_vec()
        } else {
            vec![0.0; CONFORMER_SUMMARY_FEATS]
        };
        result.extend(conformer_features);

        let conformer_motion_hist = if let Some(conformer) = conformer {
            conformer.motion_histogram_features().to_vec()
        } else {
            vec![0.0; CONFORMER_MOTION_HIST_FEATS]
        };
        result.extend(conformer_motion_hist);
    }

    Ok(result)
}
