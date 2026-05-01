//! Code specific to the spacial / pharmacophore neural net. This is experimental. We may end up
//! handling spacial relations in a different way: With or without an explicit association with
//! pharmacophore features.
//!
//! Pharmacophore features are nodes; each pair of them is an edge.

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io,
};

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        gnn::{PER_PHARM_SCALARS, PER_SPACIAL_EDGE_FEATS},
        non_nn_ml,
        non_nn_ml::GnnAnalysisTools,
    },
};

// Tunable parameters for the spacial/pharmacophore GNN.
const SPACIAL_ADJ_SIGMA_SQ: f32 = 16.0; // sigma=4 Å for adjacency Gaussian
const SPACIAL_DIST_SCALE: f32 = 10.0; // Normalise raw distances to ~O(1)
const SPACIAL_RBF_SIGMA_SQ: f32 = 2.25; // sigma=1.5 Å for RBF basis functions
const SPACIAL_RBF_CENTERS: [f32; 4] = [2.0, 4.0, 6.0, 8.0]; // Å

/// Pharmacophore-feature graph for 3-D spatial/geometric structure.
///
/// Nodes = pharmacophore sites derived from the molecule characterization:
///   H-bond donors (type 1), H-bond acceptors (type 2),
///   hydrophobic carbons (type 3), aromatic ring centroids (type 4).
///
/// Edges are fully connected (pharmacophore graphs are small, ~5-30 nodes).
/// Edge weights use a Gaussian of Euclidean distance; edge features encode the
/// distance via a raw scaled value plus 4 RBF Gaussian basis functions, giving the
/// network explicit geometric information. Angles are captured implicitly by
/// multi-hop message passing (law-of-cosines encodes them after ≥2 GNN layers).
///
/// Graph properties
/// ----------------
/// Edges: Undirected complete graph over pharmacophore sites, weighted by a Gaussian of
///   Euclidean distance
/// Underlying pharmacophore graph has self-connections: No
/// GNN message-passing adjacency has self-loops: Yes (`adj[i,i] = 1.0` is added explicitly)
/// Nodes have multiple types (heterogeneous): No in the formal hetero-GNN sense; this is a
///   single pharmacophore-node graph, with donor/acceptor/hydrophobic/aromatic role encoded by
///   `pharm_type_indices`
/// Edges have multiple types: No separate relation types; every node pair is connected by the
///   same spatial relation, with distance encoded by the adjacency weight and RBF edge features
/// Multipartite (edges can connect only to nodes of a diff type): No
/// Multiplex (Edges exist in layers; nodes are on all layers): No
/// Heterophily (Nodes are preferentially connected to others which have diff labels): Not fixed
///   by the construction; because the graph is complete, nodes connect to both same-type and
///   different-type sites
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataSpacial {
    /// Integer type index for each node: 1=Donor, 2=Acceptor, 3=Hydrophobic, 4=Aromatic.
    pub pharm_type_indices: Vec<i32>,
    /// Per-node scalar features: [r_from_pharm_centroid, mean_pairwise_dist] (both normalised).
    pub scalars: Vec<f32>,
    /// Graph-level analysis features computed from a sparse proximity graph over
    /// pharmacophore nodes, rather than the fully connected message-passing graph.
    pub analysis_features: Vec<f32>,
    /// Gaussian distance-weighted adjacency (fully connected + self-loops). Flat [N²].
    pub adj: Vec<f32>,
    /// Per-edge geometric features: [scaled_dist, rbf0..rbf3]. Flat [N² × PER_SPACIAL_EDGE_FEATS].
    pub edge_feats: Vec<f32>,
    pub num_nodes: usize,
}

impl GraphDataSpacial {
    fn empty() -> Self {
        Self {
            pharm_type_indices: Vec::new(),
            scalars: Vec::new(),
            analysis_features: Vec::new(),
            adj: Vec::new(),
            edge_feats: Vec::new(),
            num_nodes: 0,
        }
    }

    pub(in crate::therapeutic) fn new(
        mol: &MoleculeSmall,
        analysis_tools: &GnnAnalysisTools,
    ) -> io::Result<Self> {
        let Some(char) = mol.characterization.as_ref() else {
            return Ok(Self::empty());
        };

        let atom_posits = &mol.common.atom_posits;

        // Collect (position as [f32;3], type_index) for each pharmacophore site.
        let mut nodes: Vec<([f32; 3], i32)> = Vec::new();

        for &i in &char.h_bond_donor {
            if i < atom_posits.len() {
                let p = atom_posits[i];
                nodes.push(([p.x as f32, p.y as f32, p.z as f32], 1));
            }
        }
        for &i in &char.h_bond_acceptor {
            if i < atom_posits.len() {
                let p = atom_posits[i];
                nodes.push(([p.x as f32, p.y as f32, p.z as f32], 2));
            }
        }
        for &i in &char.hydrophobic_carbon {
            if i < atom_posits.len() {
                let p = atom_posits[i];
                nodes.push(([p.x as f32, p.y as f32, p.z as f32], 3));
            }
        }
        for ring in &char.rings {
            let c = ring.center(atom_posits);
            nodes.push(([c.x as f32, c.y as f32, c.z as f32], 4));
        }

        let num_nodes = nodes.len();
        if num_nodes == 0 {
            return Ok(Self::empty());
        }

        // Pharmacophore centroid.
        let (mut cx, mut cy, mut cz) = (0f32, 0f32, 0f32);
        for (p, _) in &nodes {
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        let inv_n = 1.0 / num_nodes as f32;
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        // Precompute all pairwise Euclidean distances (symmetric).
        let mut dist_mat = vec![0f32; num_nodes * num_nodes];
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                let pi = nodes[i].0;
                let pj = nodes[j].0;
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                dist_mat[i * num_nodes + j] = d;
                dist_mat[j * num_nodes + i] = d;
            }
        }

        // Node features.
        let mut pharm_type_indices = Vec::with_capacity(num_nodes);
        let mut scalars = Vec::with_capacity(num_nodes * PER_PHARM_SCALARS);

        for (i, (p, ty)) in nodes.iter().enumerate() {
            pharm_type_indices.push(*ty);

            // Rotation-invariant distance from pharmacophore centroid, normalised.
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            let r = (dx * dx + dy * dy + dz * dz).sqrt() / SPACIAL_DIST_SCALE;
            scalars.push(r);

            // Mean pairwise distance to all other pharmacophore nodes, normalised.
            let mean_d = if num_nodes > 1 {
                let mut sum = 0f32;
                for j in 0..num_nodes {
                    if i != j {
                        sum += dist_mat[i * num_nodes + j];
                    }
                }
                sum / ((num_nodes - 1) as f32 * SPACIAL_DIST_SCALE)
            } else {
                0.0
            };
            scalars.push(mean_d);
        }

        let analysis_adj = non_nn_ml::build_spacial_analysis_adj(&dist_mat, num_nodes);
        let mut base_labels = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut hasher = DefaultHasher::new();
            pharm_type_indices[i].hash(&mut hasher);
            analysis_adj[i].len().hash(&mut hasher);
            non_nn_ml::bucket_scalar(scalars[i * PER_PHARM_SCALARS], 4.0).hash(&mut hasher);
            non_nn_ml::bucket_scalar(scalars[i * PER_PHARM_SCALARS + 1], 4.0).hash(&mut hasher);
            base_labels.push(hasher.finish());
        }
        let analysis_features =
            non_nn_ml::graph_analysis_features(analysis_tools, &base_labels, &analysis_adj);

        // Edge features and adjacency (fully connected; all pairs connected).
        let n2 = num_nodes * num_nodes;
        let mut adj = vec![0f32; n2];
        let mut edge_feats = vec![0f32; n2 * PER_SPACIAL_EDGE_FEATS];

        // Self-loops with weight 1; edge features at self-loop stay 0.
        for i in 0..num_nodes {
            adj[i * num_nodes + i] = 1.0;
        }

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    continue;
                }
                let d = dist_mat[i * num_nodes + j];
                let d_sq = d * d;

                // Gaussian-weighted adjacency (same spirit as the atom GNN bond kernel).
                adj[i * num_nodes + j] = (-d_sq / SPACIAL_ADJ_SIGMA_SQ).exp();

                // Edge feature vector: [scaled_dist, rbf_0 .. rbf_3].
                // RBF encoding explicitly gives the network multi-resolution distance info.
                let base = (i * num_nodes + j) * PER_SPACIAL_EDGE_FEATS;
                edge_feats[base] = (d / SPACIAL_DIST_SCALE).clamp(0.0, 1.5);
                for (k, &mu) in SPACIAL_RBF_CENTERS.iter().enumerate() {
                    let dmu = d - mu;
                    edge_feats[base + 1 + k] = (-(dmu * dmu) / SPACIAL_RBF_SIGMA_SQ).exp();
                }
            }
        }

        // Symmetric normalization is applied in the model after gating; see GraphDataAtom.

        Ok(Self {
            pharm_type_indices,
            scalars,
            analysis_features,
            adj,
            edge_feats,
            num_nodes,
        })
    }
}
