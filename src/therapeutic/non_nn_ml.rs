//! Graph analysis features not associated with the neural net.

use std::{
    collections::{HashMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::{
    molecules::Atom,
    therapeutic::{gnn, gnn::GnnAnalysisTools},
};

/// Current AtomGraph analysis bundle.
///
/// We keep this as a helper rather than a constant because `graphlets` owns a `Vec`.
/// The configuration is also persisted into `ModelConfig`, so inference reloads the
/// same feature set that training used.
pub(in crate::therapeutic) fn atom_graph_analysis_tools() -> GnnAnalysisTools {
    GnnAnalysisTools {
        weisfeiler_lehman: true,
        graphlets: Some(vec![3]),
        path_based_methods: true,
        local_overlap_statistics: true,
        katz_index: false,
        lhn_similarity: false,
        random_walk_methods: false,
    }
}

/// Current ComponentGraph analysis bundle.
pub(in crate::therapeutic) fn component_graph_analysis_tools() -> GnnAnalysisTools {
    GnnAnalysisTools {
        weisfeiler_lehman: true,
        graphlets: Some(vec![3]),
        path_based_methods: true,
        local_overlap_statistics: true,
        katz_index: false,
        lhn_similarity: false,
        random_walk_methods: false,
    }
}

/// Current Pharmacophore/SpatialGraph analysis bundle.
pub(in crate::therapeutic) fn spacial_graph_analysis_tools() -> GnnAnalysisTools {
    GnnAnalysisTools {
        weisfeiler_lehman: true,
        graphlets: Some(vec![3]),
        path_based_methods: true,
        local_overlap_statistics: true,
        katz_index: false,
        lhn_similarity: false,
        random_walk_methods: false,
    }
}

pub fn atom_graph_analysis_features(
    tools: &GnnAnalysisTools,
    atoms: &[Atom],
    adj: &[Vec<usize>],
    is_h_bond_acceptor: &[bool],
    is_h_bond_donor: &[bool],
    is_aromatic_ring: &[bool],
) -> Vec<f32> {
    let base_labels: Vec<u64> = atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let mut hasher = DefaultHasher::new();
            gnn::vocab_lookup_element(atom.element).hash(&mut hasher);
            adj[i].len().hash(&mut hasher);
            is_h_bond_acceptor[i].hash(&mut hasher);
            is_h_bond_donor[i].hash(&mut hasher);
            is_aromatic_ring[i].hash(&mut hasher);
            hasher.finish()
        })
        .collect();

    graph_analysis_features(tools, &base_labels, adj)
}

pub fn graph_analysis_features(
    tools: &GnnAnalysisTools,
    base_labels: &[u64],
    adj: &[Vec<usize>],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(tools.feature_dim());

    if tools.weisfeiler_lehman {
        out.extend(wl_features(base_labels, adj));
    }

    if let Some(graphlets) = &tools.graphlets {
        for &size in graphlets {
            if size == 3 {
                out.extend(graphlet_size_3_features(adj));
            }
        }
    }

    if tools.path_based_methods {
        out.extend(path_based_features(adj));
    }

    if tools.local_overlap_statistics {
        out.extend(local_overlap_features(adj));
    }

    if tools.lhn_similarity {
        out.extend(lhn_similarity_features(adj));
    }

    out
}

pub fn build_spacial_analysis_adj(dist_mat: &[f32], num_nodes: usize) -> Vec<Vec<usize>> {
    const SPACIAL_ANALYSIS_RADIUS_A: f32 = 5.5;
    const SPACIAL_ANALYSIS_MIN_NEIGHBORS: usize = 2;

    if num_nodes == 0 {
        return Vec::new();
    }

    let mut adj = vec![Vec::new(); num_nodes];

    for i in 0..num_nodes {
        for j in (i + 1)..num_nodes {
            if dist_mat[i * num_nodes + j] <= SPACIAL_ANALYSIS_RADIUS_A {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    for i in 0..num_nodes {
        let mut nearest = Vec::with_capacity(num_nodes.saturating_sub(1));
        for j in 0..num_nodes {
            if i != j {
                nearest.push((dist_mat[i * num_nodes + j], j));
            }
        }
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));

        let target_degree = SPACIAL_ANALYSIS_MIN_NEIGHBORS.min(num_nodes.saturating_sub(1));
        for &(_, j) in &nearest {
            if adj[i].len() >= target_degree {
                break;
            }

            if !adj[i].contains(&j) {
                adj[i].push(j);
            }
            if !adj[j].contains(&i) {
                adj[j].push(i);
            }
        }
    }

    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }

    adj
}

pub fn graphlet_size_3_features(adj: &[Vec<usize>]) -> [f32; 3] {
    let n = adj.len();
    let total_triplets = choose3(n);
    let adj_mat = adjacency_matrix(adj);

    let mut wedges = 0usize;
    let mut triangles = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let edge_count = usize::from(has_edge(&adj_mat, n, i, j))
                    + usize::from(has_edge(&adj_mat, n, i, k))
                    + usize::from(has_edge(&adj_mat, n, j, k));

                match edge_count {
                    2 => wedges += 1,
                    3 => triangles += 1,
                    _ => {}
                }
            }
        }
    }

    let connected_triplets: usize = adj.iter().map(|nbrs| choose2(nbrs.len())).sum();
    let denom_triplets = total_triplets.max(1) as f32;
    let transitivity = if connected_triplets > 0 {
        (3 * triangles) as f32 / connected_triplets as f32
    } else {
        0.0
    };

    [
        wedges as f32 / denom_triplets,
        triangles as f32 / denom_triplets,
        transitivity,
    ]
}

fn wl_features(base_labels: &[u64], adj: &[Vec<usize>]) -> [f32; 4] {
    let n = base_labels.len().max(1) as f32;
    let hop_1 = wl_refine_labels(base_labels, adj);
    let hop_2 = wl_refine_labels(&hop_1, adj);

    [
        unique_label_count(base_labels) as f32 / n,
        unique_label_count(&hop_1) as f32 / n,
        unique_label_count(&hop_2) as f32 / n,
        dominant_label_frac(&hop_2),
    ]
}

fn wl_refine_labels(labels: &[u64], adj: &[Vec<usize>]) -> Vec<u64> {
    let mut refined = Vec::with_capacity(labels.len());
    let mut neighbors = Vec::new();

    for (i, &label) in labels.iter().enumerate() {
        neighbors.clear();
        neighbors.extend(adj[i].iter().map(|&j| labels[j]));
        neighbors.sort_unstable();

        let mut hasher = DefaultHasher::new();
        label.hash(&mut hasher);
        neighbors.hash(&mut hasher);
        refined.push(hasher.finish());
    }

    refined
}

pub fn path_based_features(adj: &[Vec<usize>]) -> [f32; 4] {
    let n = adj.len();
    let total_pairs = choose2(n);
    if total_pairs == 0 {
        return [0.0; 4];
    }

    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    let mut reachable_pairs = 0usize;
    let mut dist_sum = 0usize;
    let mut diameter = 0usize;
    let mut long_paths = 0usize;

    for src in 0..n {
        dist.fill(usize::MAX);
        dist[src] = 0;
        queue.clear();
        queue.push_back(src);

        while let Some(node) = queue.pop_front() {
            let next_dist = dist[node] + 1;
            for &nbr in &adj[node] {
                if dist[nbr] == usize::MAX {
                    dist[nbr] = next_dist;
                    queue.push_back(nbr);
                }
            }
        }

        for &d in dist.iter().skip(src + 1) {
            if d == usize::MAX {
                continue;
            }

            reachable_pairs += 1;
            dist_sum += d;
            diameter = diameter.max(d);
            if d >= 4 {
                long_paths += 1;
            }
        }
    }

    let reachable_pairs_f = reachable_pairs.max(1) as f32;
    [
        reachable_pairs as f32 / total_pairs as f32,
        dist_sum as f32 / reachable_pairs_f / n.max(1) as f32,
        diameter as f32 / n.max(1) as f32,
        long_paths as f32 / reachable_pairs_f,
    ]
}

pub fn local_overlap_features(adj: &[Vec<usize>]) -> [f32; 2] {
    let n = adj.len();
    let adj_mat = adjacency_matrix(adj);

    let mut edge_count = 0usize;
    let mut jaccard_sum = 0.0f32;
    let mut sorensen_sum = 0.0f32;

    for u in 0..n {
        for &v in &adj[u] {
            if u >= v {
                continue;
            }

            let common = common_neighbor_count(adj, &adj_mat, u, v);
            let deg_sum = adj[u].len() + adj[v].len();
            let union = deg_sum.saturating_sub(common);

            if union > 0 {
                jaccard_sum += common as f32 / union as f32;
            }
            if deg_sum > 0 {
                sorensen_sum += 2.0 * common as f32 / deg_sum as f32;
            }
            edge_count += 1;
        }
    }

    if edge_count == 0 {
        [0.0; 2]
    } else {
        [
            jaccard_sum / edge_count as f32,
            sorensen_sum / edge_count as f32,
        ]
    }
}

pub fn lhn_similarity_features(adj: &[Vec<usize>]) -> [f32; 3] {
    let n = adj.len();
    let total_pairs = choose2(n);
    if total_pairs == 0 {
        return [0.0; 3];
    }

    let adj_mat = adjacency_matrix(adj);
    let mut pair_sum = 0.0f32;
    let mut pair_max = 0.0f32;
    let mut edge_sum = 0.0f32;
    let mut edge_count = 0usize;

    for u in 0..n {
        let deg_u = adj[u].len();
        if deg_u == 0 {
            continue;
        }

        for v in (u + 1)..n {
            let deg_v = adj[v].len();
            if deg_v == 0 {
                continue;
            }

            // We summarise the standard degree-normalised LHN link-prediction
            // score, which is a practical small-graph member of the LHN
            // similarity family for our molecular descriptors.
            let common = common_neighbor_count(adj, &adj_mat, u, v) as f32;
            let score = common / (deg_u * deg_v) as f32;

            pair_sum += score;
            pair_max = pair_max.max(score);

            if has_edge(&adj_mat, n, u, v) {
                edge_sum += score;
                edge_count += 1;
            }
        }
    }

    [
        pair_sum / total_pairs as f32,
        pair_max,
        if edge_count > 0 {
            edge_sum / edge_count as f32
        } else {
            0.0
        },
    ]
}

fn unique_label_count(labels: &[u64]) -> usize {
    let mut sorted = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

fn dominant_label_frac(labels: &[u64]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &label in labels {
        *counts.entry(label).or_default() += 1;
    }

    counts.values().copied().max().unwrap_or_default() as f32 / labels.len() as f32
}

fn adjacency_matrix(adj: &[Vec<usize>]) -> Vec<bool> {
    let n = adj.len();
    let mut out = vec![false; n * n];

    for (i, nbrs) in adj.iter().enumerate() {
        for &j in nbrs {
            out[i * n + j] = true;
        }
    }

    out
}

pub fn has_edge(adj_mat: &[bool], n: usize, i: usize, j: usize) -> bool {
    adj_mat[i * n + j]
}

fn choose2(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1)) / 2
}

fn choose3(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1))
        .saturating_mul(n.saturating_sub(2))
        / 6
}

pub fn bucket_scalar(value: f32, bins_per_unit: f32) -> i16 {
    (value.max(0.0) * bins_per_unit).round().clamp(0.0, 127.0) as i16
}

fn common_neighbor_count(adj: &[Vec<usize>], adj_mat: &[bool], u: usize, v: usize) -> usize {
    let (small, other) = if adj[u].len() <= adj[v].len() {
        (&adj[u], v)
    } else {
        (&adj[v], u)
    };

    small
        .iter()
        .filter(|&&nbr| nbr != u && nbr != v && has_edge(adj_mat, adj.len(), other, nbr))
        .count()
}
