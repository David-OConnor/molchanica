//! Analyzes a set of molecules (e.g. from an MD snapshot), to assess how well
//! mixed they are. E.g. of a solute and solvent to assess solubility.

use dynamics::SimBox;
use lin_alg::f32::Vec3 as Vec3F32;

const SOLUBILITY_KERNEL_SIGMAS_A: [f32; 3] = [4.0, 7.0, 10.0];
const SOLUBILITY_CONTACT_CUTOFF_A: f32 = 4.2;
const SOLUBILITY_AGGREGATION_PENALTY_STRENGTH: f32 = 3.5;
const SOLUBILITY_LOG_EXPANSION_GAIN: f32 = 80.0;

fn valid_solubility_cell(cell: &SimBox) -> bool {
    cell.extent.x.is_finite()
        && cell.extent.y.is_finite()
        && cell.extent.z.is_finite()
        && cell.extent.x > f32::EPSILON
        && cell.extent.y > f32::EPSILON
        && cell.extent.z > f32::EPSILON
}

fn selected_solute_mols(
    solute_atom_posits: &[Vec3F32],
    atoms_per_solute: usize,
    solute_atom_indices: &[usize],
) -> Vec<Vec<Vec3F32>> {
    if atoms_per_solute == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();

    for mol in solute_atom_posits.chunks_exact(atoms_per_solute) {
        let mut selected = Vec::new();

        if solute_atom_indices.is_empty() {
            selected.extend(mol.iter().copied().filter(|posit| finite_posit(*posit)));
        } else {
            for &atom_i in solute_atom_indices {
                if let Some(&posit) = mol.get(atom_i) {
                    if finite_posit(posit) {
                        selected.push(posit);
                    }
                }
            }

            if selected.is_empty() {
                selected.extend(mol.iter().copied().filter(|posit| finite_posit(*posit)));
            }
        }

        if !selected.is_empty() {
            result.push(selected);
        }
    }

    result
}

fn finite_posit(posit: Vec3F32) -> bool {
    posit.x.is_finite() && posit.y.is_finite() && posit.z.is_finite()
}

fn solute_aggregation_factor(solute_mols: &[Vec<Vec3F32>], cell: &SimBox) -> f32 {
    let n = solute_mols.len();
    if n < 2 {
        return 1.0;
    }

    let mut parent: Vec<_> = (0..n).collect();
    let mut degree = vec![0usize; n];
    let mut contact_pairs = 0usize;

    for i in 0..n {
        for j in i + 1..n {
            if solute_mols_touch(&solute_mols[i], &solute_mols[j], cell) {
                union_roots(&mut parent, i, j);
                degree[i] += 1;
                degree[j] += 1;
                contact_pairs += 1;
            }
        }
    }

    let mut component_sizes = vec![0usize; n];
    for i in 0..n {
        let root = find_root(&mut parent, i);
        component_sizes[root] += 1;
    }

    let largest_component = component_sizes.into_iter().max().unwrap_or(1);
    let contacted_fraction = degree.iter().filter(|&&degree| degree > 0).count() as f32 / n as f32;
    let possible_contact_pairs = n.saturating_sub(1) * n / 2;
    let contact_pair_fraction = if possible_contact_pairs > 0 {
        contact_pairs as f32 / possible_contact_pairs as f32
    } else {
        0.0
    };
    let largest_cluster_penalty = (largest_component.saturating_sub(1) as f32
        / n.saturating_sub(1).max(1) as f32)
        .clamp(0.0, 1.0);
    let aggregation_penalty = (0.55 * largest_cluster_penalty.powf(1.25)
        + 0.30 * contacted_fraction.powi(2)
        + 0.15 * contact_pair_fraction.sqrt())
    .clamp(0.0, 1.0);

    (-SOLUBILITY_AGGREGATION_PENALTY_STRENGTH * aggregation_penalty)
        .exp()
        .clamp(0.0, 1.0)
}

fn local_solute_water_mixing_score(
    solute_mols: &[Vec<Vec3F32>],
    water_o_posits: &[Vec3F32],
    cell: &SimBox,
) -> f32 {
    let solute_atom_count: usize = solute_mols.iter().map(Vec::len).sum();
    if solute_atom_count == 0 {
        return 0.0;
    }

    let sigmas = solubility_kernel_sigmas(cell);
    let water_norm = water_o_posits.len().max(1) as f32;
    let mut total_score = 0.0;

    for sigma in sigmas {
        let mut scale_score = 0.0;

        for (mol_i, mol) in solute_mols.iter().enumerate() {
            let solute_norm = solute_atom_count.saturating_sub(mol.len()).max(1) as f32;

            for &solute_posit in mol {
                let mut local_solute = 0.0;
                let mut local_water = 0.0;

                for (other_mol_i, other_mol) in solute_mols.iter().enumerate() {
                    if mol_i == other_mol_i {
                        continue;
                    }

                    for &other_posit in other_mol {
                        let d2 = cell
                            .min_image(other_posit - solute_posit)
                            .magnitude_squared();
                        if d2.is_finite() {
                            local_solute += gaussian_weight(d2, sigma);
                        }
                    }
                }

                for &water_posit in water_o_posits {
                    let d2 = cell
                        .min_image(water_posit - solute_posit)
                        .magnitude_squared();
                    if d2.is_finite() {
                        local_water += gaussian_weight(d2, sigma);
                    }
                }

                let local_solute = local_solute / solute_norm;
                let local_water = local_water / water_norm;
                let local_density = local_solute + local_water;
                let atom_score = if local_density > f32::EPSILON {
                    (2.0 * local_water / local_density).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                scale_score += atom_score;
            }
        }

        total_score += scale_score / solute_atom_count as f32;
    }

    total_score / sigmas.len() as f32
}

fn solute_center_dispersion_score(solute_mols: &[Vec<Vec3F32>], cell: &SimBox) -> f32 {
    if solute_mols.len() < 2 {
        return 1.0;
    }

    let expected_uniform_rms =
        ((cell.extent.x.powi(2) + cell.extent.y.powi(2) + cell.extent.z.powi(2)) / 12.0).sqrt();
    if expected_uniform_rms <= f32::EPSILON {
        return 0.0;
    }

    let centers = solute_mol_centers(solute_mols, cell);
    let mut sampled_pairs = 0usize;
    let mut d2_sum = 0.0;

    for (i, &a) in centers.iter().enumerate() {
        for &b in &centers[i + 1..] {
            let d2 = cell.min_image(b - a).magnitude_squared();
            if d2.is_finite() {
                d2_sum += d2 as f64;
                sampled_pairs += 1;
            }
        }
    }

    if sampled_pairs == 0 {
        return 0.0;
    }

    let observed_rms = (d2_sum / sampled_pairs as f64).sqrt() as f32;
    (observed_rms / expected_uniform_rms).clamp(0.0, 1.0)
}

fn union_roots(parent: &mut [usize], a: usize, b: usize) {
    let a_root = find_root(parent, a);
    let b_root = find_root(parent, b);

    if a_root != b_root {
        parent[b_root] = a_root;
    }
}

fn solute_mols_touch(a: &[Vec3F32], b: &[Vec3F32], cell: &SimBox) -> bool {
    let cutoff_d2 = SOLUBILITY_CONTACT_CUTOFF_A.powi(2);

    for &pa in a {
        for &pb in b {
            let d2 = cell.min_image(pb - pa).magnitude_squared();
            if d2.is_finite() && d2 <= cutoff_d2 {
                return true;
            }
        }
    }

    false
}

fn find_root(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find_root(parent, parent[i]);
    }

    parent[i]
}

fn solubility_kernel_sigmas(cell: &SimBox) -> [f32; 3] {
    let half_min_extent = 0.5 * cell.extent.x.min(cell.extent.y).min(cell.extent.z).max(1.0);

    SOLUBILITY_KERNEL_SIGMAS_A.map(|sigma| sigma.min(0.9 * half_min_extent).max(1.0))
}

fn gaussian_weight(d2: f32, sigma_a: f32) -> f32 {
    (-0.5 * d2 / sigma_a.powi(2)).exp()
}

fn log_expanded_solubility_score(raw_score: f32) -> f32 {
    let raw_score = raw_score.clamp(0.0, 1.0);

    (1.0 + SOLUBILITY_LOG_EXPANSION_GAIN * raw_score).ln()
        / (1.0 + SOLUBILITY_LOG_EXPANSION_GAIN).ln()
}

fn solute_mol_center(mol: &[Vec3F32], cell: &SimBox) -> Vec3F32 {
    let anchor = cell.wrap(mol[0]);
    let mut sum = Vec3F32::new_zero();

    for &posit in mol {
        sum += anchor + cell.min_image(posit - anchor);
    }

    cell.wrap(sum / mol.len() as f32)
}

fn solute_mol_centers(solute_mols: &[Vec<Vec3F32>], cell: &SimBox) -> Vec<Vec3F32> {
    solute_mols
        .iter()
        .map(|mol| solute_mol_center(mol, cell))
        .collect()
}

/// Estimate solubility from the final frame of the simulation.
///
/// This will eventually share a similar scale to that used by AqSolDb, but for now, it is arbitrary.
/// It is intended to order solutes correctly, but the absolute scale is arbitrary. Higher values
/// mean higher solubility.
///
/// `solute_atom_posits` is for all solute atoms, grouped by solute copy. `solute_atom_indices`
/// may exclude hydrogens.
///
/// We treat relatively high solute aggregation as less soluble. Results are on a log-expanded
/// 0 to 1 scale: 0 means no useful water/solute mixing evidence, while 1 means an even
/// distribution of solute and solvent in space. The scale is intentionally not linear; low raw
/// scores get more room so poorly soluble bulky/lipid-like molecules do not all collapse to
/// `0.000`.
///
/// We use a sim cell with PBCs, so distance calculations take periodic images into account.
pub(in crate::properties) fn compute_solubility(
    solute_atom_posits: &[Vec3F32],
    atoms_per_solute: usize,
    solute_atom_indices: &[usize],
    water_o_posits: &[Vec3F32],
    cell: &SimBox,
) -> f32 {
    if !valid_solubility_cell(cell) || solute_atom_posits.is_empty() || water_o_posits.is_empty() {
        return 0.0;
    }

    let solute_mols =
        selected_solute_mols(solute_atom_posits, atoms_per_solute, solute_atom_indices);
    let water_o_posits: Vec<_> = water_o_posits
        .iter()
        .copied()
        .filter(|posit| finite_posit(*posit))
        .collect();

    if solute_mols.is_empty() || water_o_posits.is_empty() {
        return 0.0;
    }

    let aggregation_factor = solute_aggregation_factor(&solute_mols, cell);
    let local_mixing = local_solute_water_mixing_score(&solute_mols, &water_o_posits, cell);
    let solute_dispersion = solute_center_dispersion_score(&solute_mols, cell);
    let mixture_score = 0.60 * local_mixing + 0.40 * solute_dispersion;
    let raw_score = (aggregation_factor * mixture_score).clamp(0.0, 1.0);

    log_expanded_solubility_score(raw_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_expansion_preserves_bounds_and_order() {
        let low = log_expanded_solubility_score(0.001);
        let mid = log_expanded_solubility_score(0.10);
        let high = log_expanded_solubility_score(1.0);

        assert_eq!(log_expanded_solubility_score(0.0), 0.0);
        assert!(low > 0.001);
        assert!(low < mid);
        assert!(mid < high);
        assert_eq!(high, 1.0);
    }

    #[test]
    fn aggregation_factor_keeps_clustered_cases_distinguishable() {
        let cell = SimBox::new(Vec3F32::splat(-20.0), Vec3F32::splat(20.0));
        let isolated = vec![
            vec![Vec3F32::new(-8.0, 0.0, 0.0)],
            vec![Vec3F32::new(8.0, 0.0, 0.0)],
        ];
        let touching = vec![
            vec![Vec3F32::new(-1.0, 0.0, 0.0)],
            vec![Vec3F32::new(1.0, 0.0, 0.0)],
        ];

        let isolated_factor = solute_aggregation_factor(&isolated, &cell);
        let touching_factor = solute_aggregation_factor(&touching, &cell);

        assert_eq!(isolated_factor, 1.0);
        assert!(touching_factor > 0.0);
        assert!(touching_factor < isolated_factor);
    }
}
