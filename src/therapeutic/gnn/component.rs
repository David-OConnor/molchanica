//! Code specific to the component (Similar concept to functional group) GNN. Components are nodes;
//! connections between components are edges. Single layer.

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io,
};

use bio_files::md_params::ForceFieldParams;

use crate::{
    mol_components::{ComponentType, MolComponents, build_adjacency_list_conn},
    molecules::small::MoleculeSmall,
    therapeutic::{
        gnn,
        gnn::{DIHEDRAL_PARAM_SUMMARY_FEATS, PER_COMP_SCALARS, PER_EDGE_COMP_FEATS},
        non_nn_ml,
        non_nn_ml::GnnAnalysisTools,
    },
};

/// Instead of atoms and bonds, this operates on components. (Fgs etc)
///
/// Graph properties
/// ----------------
/// Edges: Undirected inter-component connections derived from cross-component bonds or shared atoms
/// Underlying component graph has self-connections: No
/// GNN message-passing adjacency has self-loops: Yes (`adj[i,i] = 1.0` is added explicitly)
/// Nodes have multiple types (heterogeneous): No in the formal hetero-GNN sense; this is a
///   homogeneous component graph, with component category encoded by `comp_type_indices`
/// Edges have multiple types: No separate relation types; edges are binary component-adjacency
///   links, with `edge_feats` marking whether the connection uses shared atoms, whether the
///   underlying cross-component bond is rotatable, and aggregated proper-dihedral params when
///   that bond is the middle bond of a torsion
/// Multipartite (edges can connect only to nodes of a diff type): No
/// Multiplex (Edges exist in layers; nodes are on all layers): No
/// Heterophily (Nodes are preferentially connected to others which have diff labels): Not fixed
///   by the construction; components can connect to either similar or different categories
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataComponent {
    pub comp_type_indices: Vec<i32>,
    pub scalars: Vec<f32>,
    pub analysis_features: Vec<f32>,
    pub adj: Vec<f32>,
    pub edge_feats: Vec<f32>,
    pub num_comps: usize,
}

/// Converts component nodes and inter-component connections into flat vectors for tensors.
/// Used by both Training and Inference.
impl GraphDataComponent {
    pub(in crate::therapeutic) fn new(
        mol: &MoleculeSmall,
        mol_comps: &MolComponents,
        ff_params: &ForceFieldParams,
        analysis_tools: &GnnAnalysisTools,
    ) -> io::Result<Self> {
        let comps = &mol_comps.components;
        let conns = &mol_comps.connections;

        let adj = build_adjacency_list_conn(conns, comps.len());

        let num_comps = comps.len();
        if num_comps == 0 {
            return Err(io::Error::other("Molecule has 0 components"));
        }

        // Node features (Indices and Scalars)
        let mut comp_type_indices = Vec::with_capacity(num_comps);

        let mut scalars = Vec::with_capacity(num_comps * PER_COMP_SCALARS);

        for (i, comp) in comps.iter().enumerate() {
            comp_type_indices.push(vocab_lookup_component(&comp.comp_type));

            // Degree is the number of edges incident to a node.
            let degree = adj.get(i).map(|n| n.len()).unwrap_or(0);
            scalars.push(degree as f32 / 6.0);
            // Number of atoms owned by this component, normalised.
            scalars.push(comp.atoms.len() as f32 / 10.0);
        }

        let mut base_labels = Vec::with_capacity(num_comps);
        for (i, comp) in comps.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            vocab_lookup_component(&comp.comp_type).hash(&mut hasher);
            adj[i].len().hash(&mut hasher);
            non_nn_ml::bucket_scalar(comp.atoms.len() as f32 / 10.0, 4.0).hash(&mut hasher);
            base_labels.push(hasher.finish());
        }
        let analysis_features =
            non_nn_ml::graph_analysis_features(analysis_tools, &base_labels, &adj);

        let proper_dihedral_summary_by_bond = gnn::proper_dihedral_summaries_by_central_bond(
            &mol.common.atoms,
            &mol.common.adjacency_list,
            ff_params,
        );

        // Conn features (Weighted Adjacency)
        let n_atoms_sq = num_comps.pow(2);
        let mut adj_list = vec![0.; n_atoms_sq];
        let mut edge_feats = vec![0.; n_atoms_sq * PER_EDGE_COMP_FEATS];

        let edge_feats_i = |i: usize, j: usize, k: usize, n: usize| -> usize {
            (i * n + j) * PER_EDGE_COMP_FEATS + k
        };

        // Self loops
        for i in 0..num_comps {
            adj_list[i * num_comps + i] = 1.0;
        }

        for conn in conns {
            let a0 = conn.comp_0;
            let a1 = conn.comp_1;
            if a0 >= num_comps || a1 >= num_comps {
                continue;
            }

            adj_list[a0 * num_comps + a1] = 1.0;
            adj_list[a1 * num_comps + a0] = 1.0;

            // Edge features: [shared_atoms, rotatable, dihedral-param summary...].
            let shared = if conn.shared_atoms { 1. } else { 0. };
            let rotatable = if conn.rotatable { 1. } else { 0. };
            let atom_i = comps[a0].atoms.get(conn.atom_0).copied().unwrap_or(0);
            let atom_j = comps[a1].atoms.get(conn.atom_1).copied().unwrap_or(0);
            let proper_summary = proper_dihedral_summary_by_bond
                .get(&gnn::bond_pair_key(atom_i, atom_j))
                .copied()
                .unwrap_or([0.0; DIHEDRAL_PARAM_SUMMARY_FEATS]);
            let features = component_edge_features(shared, rotatable, proper_summary);

            for (k, &value) in features.iter().enumerate() {
                edge_feats[edge_feats_i(a0, a1, k, num_comps)] = value;
                edge_feats[edge_feats_i(a1, a0, k, num_comps)] = value;
            }
        }

        // Symmetric normalization is applied in the model after gating; see GraphDataAtom.

        Ok(Self {
            comp_type_indices,
            scalars,
            analysis_features,
            adj: adj_list,
            edge_feats,
            num_comps,
        })
    }
}

fn component_edge_features(
    shared: f32,
    rotatable: f32,
    proper_summary: [f32; DIHEDRAL_PARAM_SUMMARY_FEATS],
) -> [f32; PER_EDGE_COMP_FEATS] {
    [
        shared,
        rotatable,
        proper_summary[0],
        proper_summary[1],
        proper_summary[2],
        proper_summary[3],
        proper_summary[4],
    ]
}

/// Maps component type to integer indices for the embedding layer.
/// 0 is reserved for padding in the Batcher, so we start at 1.
fn vocab_lookup_component(comp_type: &ComponentType) -> i32 {
    use crate::mol_components::ComponentType::*;
    match comp_type {
        Atom(_) => 1,
        Ring(_) => 2,
        Chain(_) => 3,
        Hydroxyl => 4,
        Carbonyl => 5,
        Carboxylate => 6,
        Amine => 7,
        Amide => 8,
        Sulfonamide => 9,
        Sulfonimide => 10,
    }
}
