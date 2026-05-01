//! Code specific to our MLP neural net: This handles features of a molecule as a whole; it is not
//! part of a GNN. Molecular weight, counts of various types of atoms, total net charge etc.

use std::io;

use crate::molecules::small::MoleculeSmall;

// Note: We can make variants of this A/R tuned to specific inference items. For now, we are using
// a single set of features for all targets.
/// Extract molecule-level features from a molecule that are relevant for inferring the target parameter. We use this
/// in both training and inference workflows.
///
/// We avoid features that may be more robustly represented by GNNs. For example, the count of rings,
/// functional groups, and H bond donors/acceptors.
pub(in crate::therapeutic) fn mlp_feats_from_mol(mol: &MoleculeSmall) -> io::Result<Vec<f32>> {
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

    Ok(vec![
        // c.num_atoms as f32,
        // c.num_bonds as f32,
        // c.mol_weight,
        // c.num_heavy_atoms as f32,
        // c.h_bond_acceptor.len() as f32,
        // c.h_bond_donor.len() as f32,
        // c.num_hetero_atoms as f32,
        // c.halogen.len() as f32,
        // c.rotatable_bonds.len() as f32,
        // c.amines.len() as f32,
        // c.amides.len() as f32,
        // c.carbonyl.len() as f32,
        // c.hydroxyl.len() as f32,
        // // c.num_valence_elecs as f32,
        // c.num_rings_aromatic as f32,
        // c.num_rings_saturated as f32,
        // c.num_rings_aliphatic as f32,
        // c.rings.len() as f32,
        // c.log_p,
        // c.molar_refractivity,
        // c.psa_topo,
        // c.asa_topo,
        // c.volume,
        // c.wiener_index.unwrap_or(0) as f32,
        //
        // ----
        //
        ln(c.num_atoms as f32),
        ln(c.num_bonds as f32),
        // ln(c.mol_weight),
        ln(c.num_heavy_atoms as f32),
        // c.h_bond_acceptor.len() as f32,
        // c.h_bond_donor.len() as f32,
        c.num_hetero_atoms as f32,
        c.halogen.len() as f32,
        c.rotatable_bonds.len() as f32,
        c.flexibility / 4., // normalizationish?
        // c.amines.len() as f32,
        // c.amides.len() as f32,
        // c.carbonyl.len() as f32,
        // c.hydroxyl.len() as f32,
        // c.carboxylate.len() as f32,
        // c.sulfonamide.len() as f32,
        // c.sulfonimide.len() as f32,
        // c.num_valence_elecs as f32,
        // c.num_rings_aromatic as f32,
        // c.num_rings_saturated as f32,
        // c.num_rings_aliphatic as f32,
        // c.rings.len() as f32,
        c.log_p,
        c.molar_refractivity,
        ln(c.psa_topo),
        ln(c.asa_topo),
        ln(c.volume),
        // ln(c.wiener_index.unwrap_or(0) as f32),
        c.rings.len() as f32 * 6. / c.num_atoms as f32, // todo temp
        ln(c.psa_topo / c.asa_topo),
        // ln(c.greasiness),
    ])
}
