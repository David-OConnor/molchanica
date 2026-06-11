//! Analyzes a set of molecules (e.g. from an MD snapshot), to assess how well
//! mixed they are. E.g. of a solute and solvent to assess solubility.

use dynamics::SimBox;
use lin_alg::f32::Vec3 as Vec3F32;

const SOLUBILITY_KERNEL_SIGMAS_A: [f32; 3] = [4.0, 7.0, 10.0];
const SOLUBILITY_CONTACT_CUTOFF_A: f32 = 4.2;
const SOLUBILITY_AGGREGATION_PENALTY_STRENGTH: f32 = 3.5;
const SOLUBILITY_LOG_EXPANSION_GAIN: f32 = 80.0;
const SOLUBILITY_BH_MAX_TREE_DEPTH: usize = 14;
const SOLUBILITY_BH_MIN_LEAF_WIDTH_A: f32 = 0.75;
const SOLUBILITY_BH_HYDRATION_SHELL_A: f32 = SOLUBILITY_CONTACT_CUTOFF_A;
const SOLUBILITY_BH_EXPECTED_WATER_FLOOR: f32 = 0.75;

#[derive(Clone, Copy, Debug)]
struct SoluteTreePoint {
    posit: Vec3F32,
}

#[derive(Clone, Copy, Debug)]
struct OctreeBounds {
    low: Vec3F32,
    high: Vec3F32,
    center: Vec3F32,
    extent: Vec3F32,
}

impl OctreeBounds {
    fn new(low: Vec3F32, high: Vec3F32) -> Self {
        let low_ordered = Vec3F32::new(low.x.min(high.x), low.y.min(high.y), low.z.min(high.z));
        let high_ordered = Vec3F32::new(low.x.max(high.x), low.y.max(high.y), low.z.max(high.z));
        let low = low_ordered;
        let high = high_ordered;
        let extent = high - low;

        Self {
            low,
            high,
            center: (low + high) * 0.5,
            extent,
        }
    }

    fn from_cell(cell: &SimBox) -> Self {
        Self::new(cell.bounds_low, cell.bounds_high)
    }

    fn child(&self, octant: usize) -> Self {
        let mut low = self.low;
        let mut high = self.high;

        if octant & 0b001 == 0 {
            high.x = self.center.x;
        } else {
            low.x = self.center.x;
        }

        if octant & 0b010 == 0 {
            high.y = self.center.y;
        } else {
            low.y = self.center.y;
        }

        if octant & 0b100 == 0 {
            high.z = self.center.z;
        } else {
            low.z = self.center.z;
        }

        Self::new(low, high)
    }

    fn octant_for(&self, posit: Vec3F32) -> usize {
        let mut result = 0;

        if posit.x > self.center.x {
            result |= 0b001;
        }
        if posit.y > self.center.y {
            result |= 0b010;
        }
        if posit.z > self.center.z {
            result |= 0b100;
        }

        result
    }

    fn max_width(&self) -> f32 {
        self.extent.x.max(self.extent.y).max(self.extent.z)
    }

    fn volume(&self) -> f32 {
        self.extent.x * self.extent.y * self.extent.z
    }

    fn expanded_volume(&self, expansion: f32, cell: &SimBox) -> f32 {
        let expansion = expansion.max(0.0);
        (self.extent.x + 2.0 * expansion).min(cell.extent.x)
            * (self.extent.y + 2.0 * expansion).min(cell.extent.y)
            * (self.extent.z + 2.0 * expansion).min(cell.extent.z)
    }

    fn contains_periodic_expanded(&self, posit: Vec3F32, expansion: f32, cell: &SimBox) -> bool {
        let expansion = expansion.max(0.0);
        let half = self.extent * 0.5 + Vec3F32::splat(expansion);
        let half = Vec3F32::new(
            half.x.min(cell.extent.x * 0.5),
            half.y.min(cell.extent.y * 0.5),
            half.z.min(cell.extent.z * 0.5),
        );
        let delta = cell.min_image(posit - self.center);

        delta.x.abs() <= half.x && delta.y.abs() <= half.y && delta.z.abs() <= half.z
    }
}

#[derive(Debug)]
struct MixingOctreeNode {
    bounds: OctreeBounds,
    children: Vec<usize>,
    solute_indices: Vec<usize>,
    water_indices: Vec<usize>,
}

#[derive(Debug)]
struct MixingOctree {
    cell: SimBox,
    solutes: Vec<SoluteTreePoint>,
    water_posits: Vec<Vec3F32>,
    nodes: Vec<MixingOctreeNode>,
}

impl MixingOctree {
    fn new(solute_mols: &[Vec<Vec3F32>], water_o_posits: &[Vec3F32], cell: &SimBox) -> Self {
        let solutes = solute_mol_centers(solute_mols, cell)
            .into_iter()
            .map(|posit| SoluteTreePoint {
                posit: cell.wrap(posit),
            })
            .collect();

        let water_posits = water_o_posits
            .iter()
            .copied()
            .filter(|posit| finite_posit(*posit))
            .map(|posit| cell.wrap(posit))
            .collect::<Vec<_>>();

        let mut result = Self {
            cell: *cell,
            solutes,
            water_posits,
            nodes: Vec::new(),
        };

        if !result.solutes.is_empty() || !result.water_posits.is_empty() {
            let solute_indices = (0..result.solutes.len()).collect();
            let water_indices = (0..result.water_posits.len()).collect();
            result.build_node(
                OctreeBounds::from_cell(cell),
                solute_indices,
                water_indices,
                0,
            );
        }

        result
    }

    fn build_node(
        &mut self,
        bounds: OctreeBounds,
        solute_indices: Vec<usize>,
        water_indices: Vec<usize>,
        depth: usize,
    ) -> usize {
        let should_split = depth < SOLUBILITY_BH_MAX_TREE_DEPTH
            && bounds.max_width() > SOLUBILITY_BH_MIN_LEAF_WIDTH_A
            && solute_indices.len() > 1;

        let mut solutes_by_octant: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
        let mut water_by_octant: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());

        if should_split {
            for &solute_i in &solute_indices {
                let octant = bounds.octant_for(self.solutes[solute_i].posit);
                solutes_by_octant[octant].push(solute_i);
            }

            for &water_i in &water_indices {
                let octant = bounds.octant_for(self.water_posits[water_i]);
                water_by_octant[octant].push(water_i);
            }
        }

        let node_i = self.nodes.len();
        self.nodes.push(MixingOctreeNode {
            bounds,
            children: Vec::new(),
            solute_indices,
            water_indices,
        });

        if should_split {
            for octant in 0..8 {
                if solutes_by_octant[octant].is_empty() && water_by_octant[octant].is_empty() {
                    continue;
                }

                let child_i = self.build_node(
                    bounds.child(octant),
                    std::mem::take(&mut solutes_by_octant[octant]),
                    std::mem::take(&mut water_by_octant[octant]),
                    depth + 1,
                );
                self.nodes[node_i].children.push(child_i);
            }
        }

        node_i
    }

    fn water_count_in_expanded_bounds(&self, bounds: &OctreeBounds, expansion: f32) -> usize {
        self.water_posits
            .iter()
            .filter(|&&posit| bounds.contains_periodic_expanded(posit, expansion, &self.cell))
            .count()
    }
}

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

fn local_solute_water_mixing_score_barnes_hut(
    solute_mols: &[Vec<Vec3F32>],
    water_o_posits: &[Vec3F32],
    cell: &SimBox,
) -> f32 {
    let tree = MixingOctree::new(solute_mols, water_o_posits, cell);
    if tree.solutes.is_empty() || tree.water_posits.is_empty() {
        return 0.0;
    }

    let cell_volume = cell.volume().max(f32::EPSILON);
    let water_number_density = tree.water_posits.len() as f32 / cell_volume;
    let reference_leaf_volume = reference_octree_leaf_volume(cell_volume, tree.solutes.len());
    let mut weighted_score = 0.0;
    let mut solute_weight = 0usize;

    for node in tree
        .nodes
        .iter()
        .filter(|node| node.children.is_empty() && !node.solute_indices.is_empty())
    {
        let solute_count = node.solute_indices.len();
        let same_leaf_expected = water_number_density * node.bounds.volume();
        let same_leaf_water = occupancy_count_score(node.water_indices.len(), same_leaf_expected);
        let shell_water_count =
            tree.water_count_in_expanded_bounds(&node.bounds, SOLUBILITY_BH_HYDRATION_SHELL_A);
        let shell_expected = water_number_density
            * node
                .bounds
                .expanded_volume(SOLUBILITY_BH_HYDRATION_SHELL_A, cell);
        let shell_water = occupancy_count_score(shell_water_count, shell_expected);
        let partition_size =
            partition_size_score(node.bounds.volume(), reference_leaf_volume, solute_count);
        let leaf_score = 0.25 * same_leaf_water + 0.55 * shell_water + 0.20 * partition_size;

        weighted_score += leaf_score * solute_count as f32;
        solute_weight += solute_count;
    }

    if solute_weight == 0 {
        return 0.0;
    }

    (weighted_score / solute_weight as f32).clamp(0.0, 1.0)
}

fn occupancy_count_score(observed: usize, expected: f32) -> f32 {
    if observed == 0 {
        return 0.0;
    }

    let expected = expected.max(SOLUBILITY_BH_EXPECTED_WATER_FLOOR);
    let ratio = observed as f32 / expected;
    let score_at_expected = 1.0 - (-1.0_f32).exp();

    ((1.0 - (-ratio).exp()) / score_at_expected).clamp(0.0, 1.0)
}

fn reference_octree_leaf_volume(cell_volume: f32, solute_count: usize) -> f32 {
    let mut reference_leaf_count = 1usize;
    let solute_count = solute_count.max(1);

    while reference_leaf_count < solute_count {
        reference_leaf_count *= 8;
    }

    cell_volume / reference_leaf_count as f32
}

fn partition_size_score(leaf_volume: f32, reference_leaf_volume: f32, solute_count: usize) -> f32 {
    if reference_leaf_volume <= f32::EPSILON {
        return 0.0;
    }

    let spacing_score = (leaf_volume / reference_leaf_volume).sqrt().clamp(0.0, 1.0);
    let crowding_score = (1.0 / solute_count.max(1) as f32).sqrt();

    spacing_score * crowding_score
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

/// Map betweren the units this library outputs, and AqSolDB's range.
fn map_to_aqsoldb(v: f32) -> f32 {
    // todo: A/R.
    lin_alg::map_linear(v, (0., 1.), (-4., 2.))
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
    // map_to_aqsoldb(res)
}

/// Estimate solubility with a Barnes-Hut-style octree over the final MD frame.
///
/// This builds a partition over
/// solute-copy centers, subdividing until each terminal box contains at most one solute copy center.
/// Each solute leaf is then scored by water occupancy in the leaf, water occupancy in a
/// cube-expanded hydration shell, and the leaf size needed to isolate that solute copy. This makes
/// the result a cube occupancy metric rather than a pairwise Gaussian-density metric.
pub(in crate::properties) fn compute_solubility_cell_list(
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

    let raw_score = local_solute_water_mixing_score_barnes_hut(&solute_mols, &water_o_posits, cell);

    log_expanded_solubility_score(raw_score)
}
