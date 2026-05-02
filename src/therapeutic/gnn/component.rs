//! Code specific to the component (Similar concept to functional group) GNN. Components are nodes;
//! connections between components are edges. These connections can be either one or more covalent bonds,
//! or one or more shared atom. Single node and edge layers.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    io,
};

use bio_files::md_params::ForceFieldParams;
use lin_alg::f64::Vec3;

use crate::{
    mol_characterization::RingType,
    mol_components::{Component, MolComponents, build_adjacency_list_conn},
    molecules::small::MoleculeSmall,
    therapeutic::{
        gnn,
        gnn::{DIHEDRAL_PARAM_SUMMARY_FEATS, vocab_lookup_element},
        non_nn_ml,
        non_nn_ml::GnnAnalysisTools,
    },
};

// Degree, number of atoms in the component, distance from mol center.
// Component type is probably the most important node feature, but it's not included here,
// as it's not a scalar.\
// Keep this in sync with `GraphDataComponent::new`
pub(in crate::therapeutic) const PER_COMP_SCALARS: usize = 3;

// Keep this in sync with `GraphDataComponent::new`.
pub(in crate::therapeutic) const COMP_EDGE_GEOM_FEATS: usize = 2;

pub(in crate::therapeutic) const PER_EDGE_COMP_FEATS: usize =
    2 + COMP_EDGE_GEOM_FEATS + DIHEDRAL_PARAM_SUMMARY_FEATS;

// This is the number of componenent types. One for each element, common ring sizes,
// one for each functional group, etc.
pub(in crate::therapeutic) const COMPONENT_VOCAB_SIZE: usize = 23;

/// Very similar to `MoleculeCommon::centroid`.
fn comp_centroid(comp: &Component, atom_posits: &[Vec3]) -> Vec3 {
    let n = comp.atoms.len() as f64;

    let sum = comp
        .atoms
        .iter()
        .map(|i| atom_posits[*i])
        .fold(Vec3::new_zero(), |a, b| a + b);

    sum / n
}

/// Set up scalars for the component GNN. These are per-node, floating point features. They
/// do not include integer (e.g. component type) per-node features.
///
/// todo: 0ing these out appears to have no effect on the result. They are therefor currently not working
/// todo: properly.
fn setup_scalars(
    comps: &[Component],
    num_comps: usize,
    mol: &MoleculeSmall,
    mol_centroid: Vec3,
    adj: &[Vec<usize>],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(num_comps * PER_COMP_SCALARS);

    for (i, comp) in comps.iter().enumerate() {
        // Degree is the number of edges incident to a node.
        let degree = adj.get(i).map(|n| n.len()).unwrap_or(0);

        res.push(degree as f32 / 3.0);

        // Number of atoms owned by this component, normalized. This divider
        // assumes no Hydrogens.

        // scalars.push(comp.atoms.len() as f32 / 4.0);

        let dist = (mol_centroid - comp_centroid(comp, &mol.common.atom_posits)).magnitude();
        // let dist =
        //     (mol_centroid - comp_centroid(comp, &mol.common.atom_posits)).magnitude_squared();

        // todo note: setting dist and/or degree to 0 seems to have no notable effect on results.
        // todo: Yikes. Not a good sign for this.

        res.push(dist as f32 / 8.0);
    }

    // todo?
    vec![0.; num_comps * PER_COMP_SCALARS]
    // res
}

/// Instead of atoms and bonds, this operates on components. (Functioal groups, rings, etc)
///
/// Graph properties
/// ----------------
/// Edges: Undirected component-adjacency links derived from cross-component bonds. If multiple
/// bonds connect the same component pair, their chemistry is aggregated onto one edge.
///
/// Underlying component graph has self-connections: No
///
/// GNN message-passing adjacency has self-loops: Yes (`adj[i,i] = 1.0` is added explicitly)
/// Nodes have multiple types (heterogeneous): No in the formal hetero-GNN sense; this is a
///   homogeneous component graph, with component category encoded by `comp_type_indices`
///
/// Edges have multiple types: No separate relation types; edges are binary component-adjacency
///   links, with `edge_feats` aggregating overlap/rotatability and proper-dihedral geometry for
///   the cross-component bonds between a component pair
///
/// Multipartite (edges can connect only to nodes of a diff type): No
///
/// Multiplex (Edges exist in layers; nodes are on all layers): No
///
/// Heterophily (Nodes are preferentially connected to others which have diff labels): Not fixed
///   by the construction; components can connect to either similar or different categories.
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
/// Used by both training and inference.
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

        let mol_centroid = mol.common.centroid();

        // todo: Even scarier than scalars not having much effect: We're not getting much effect from
        // todo: comp type either???
        // Node features (Indices and Scalars)
        // todo temp!
        let comp_type_indices = comps
            .iter()
            .map(|c| vocab_lookup_component(c, mol))
            .collect();
        // let comp_type_indices = comps.iter().map(|c| 0).collect();

        let scalars = setup_scalars(comps, num_comps, mol, mol_centroid, &adj);

        let mut base_labels = Vec::with_capacity(num_comps);
        for (i, comp) in comps.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            vocab_lookup_component(comp, mol).hash(&mut hasher);
            adj[i].len().hash(&mut hasher);

            non_nn_ml::bucket_scalar(comp.atoms.len() as f32 / 10., 4.).hash(&mut hasher);
            base_labels.push(hasher.finish());
        }

        let analysis_features =
            non_nn_ml::graph_analysis_features(analysis_tools, &base_labels, &adj);

        let proper_dihedral_summary_by_bond = gnn::proper_dihedral_summaries_by_central_bond(
            &mol.common.atoms,
            &mol.common.adjacency_list,
            ff_params,
        );

        let proper_dihedral_stats_by_bond = gnn::proper_dihedral_stats_by_central_bond(
            &mol.common.atoms,
            &mol.common.adjacency_list,
            ff_params,
        );

        let mut pair_to_conns: HashMap<(usize, usize), Vec<_>> = HashMap::new();
        for conn in conns {
            if conn.comp_0 >= num_comps || conn.comp_1 >= num_comps || conn.comp_0 == conn.comp_1 {
                continue;
            }
            pair_to_conns
                .entry(component_pair_key(conn.comp_0, conn.comp_1))
                .or_default()
                .push(conn);
        }

        // Connection features (Weighted Adjacency)
        let (edge_feats, adj_list) = {
            let n_comps_sq = num_comps.pow(2);
            let mut adj_list = vec![0.; n_comps_sq];
            let mut edge_feats = vec![0.; n_comps_sq * PER_EDGE_COMP_FEATS];

            let edge_feats_i = |i: usize, j: usize, k: usize, n: usize| -> usize {
                (i * n + j) * PER_EDGE_COMP_FEATS + k
            };

            // Self loops
            for i in 0..num_comps {
                adj_list[i * num_comps + i] = 1.0;
            }

            for ((a0, a1), pair_conns) in pair_to_conns {
                let pair_conn_count = pair_conns.len();
                let shared = if pair_conns.iter().any(|conn| conn.shared_atoms) {
                    1.0
                } else {
                    0.0
                };
                let rotatable = pair_conns.iter().filter(|conn| conn.rotatable).count() as f32
                    / pair_conn_count as f32;
                let mut torsion_stats = [0.0; 2];
                let mut proper_summary = [0.0; DIHEDRAL_PARAM_SUMMARY_FEATS];

                for conn in &pair_conns {
                    let (local_atom_0, local_atom_1) = if conn.comp_0 == a0 {
                        (conn.atom_0, conn.atom_1)
                    } else {
                        (conn.atom_1, conn.atom_0)
                    };
                    let Some(&atom_i) = comps[a0].atoms.get(local_atom_0) else {
                        continue;
                    };
                    let Some(&atom_j) = comps[a1].atoms.get(local_atom_1) else {
                        continue;
                    };
                    let bond_key = gnn::bond_pair_key(atom_i, atom_j);
                    let bond_torsion_stats = proper_dihedral_stats_by_bond
                        .get(&bond_key)
                        .copied()
                        .unwrap_or([0.0; 2]);
                    let bond_proper_summary = proper_dihedral_summary_by_bond
                        .get(&bond_key)
                        .copied()
                        .unwrap_or([0.0; DIHEDRAL_PARAM_SUMMARY_FEATS]);

                    torsion_stats[0] += bond_torsion_stats[0];
                    torsion_stats[1] += bond_torsion_stats[1];
                    for (dst, src) in proper_summary.iter_mut().zip(bond_proper_summary) {
                        *dst += src;
                    }
                }

                let inv_conn_count = 1.0 / pair_conn_count as f32;
                torsion_stats[0] *= inv_conn_count;
                torsion_stats[1] *= inv_conn_count;
                for value in &mut proper_summary {
                    *value *= inv_conn_count;
                }

                adj_list[a0 * num_comps + a1] = 1.0;
                adj_list[a1 * num_comps + a0] = 1.0;

                let features =
                    component_edge_features(shared, rotatable, torsion_stats, proper_summary);

                for (k, &value) in features.iter().enumerate() {
                    edge_feats[edge_feats_i(a0, a1, k, num_comps)] = value;
                    edge_feats[edge_feats_i(a1, a0, k, num_comps)] = value;
                }
            }

            (edge_feats, adj_list)
        };

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
    torsion_stats: [f32; 2],
    proper_summary: [f32; DIHEDRAL_PARAM_SUMMARY_FEATS],
) -> [f32; PER_EDGE_COMP_FEATS] {
    [
        shared,
        rotatable,
        torsion_stats[0],
        torsion_stats[1],
        proper_summary[0],
        proper_summary[1],
        proper_summary[2],
        proper_summary[3],
    ]
}

/// Maps component type to integer indices for the embedding layer.
/// 0 is reserved for padding in the Batcher, so we start at 1.
fn vocab_lookup_component(comp: &crate::mol_components::Component, mol: &MoleculeSmall) -> i32 {
    use crate::mol_components::ComponentType::*;

    let token = match &comp.comp_type {
        Atom(el) => vocab_lookup_element(*el),
        // This must be kept in sync with the atom lookup so as not to overlap
        // with it.
        Ring(ring) => match ring.ring_type {
            RingType::Aromatic => 11,
            RingType::Saturated => 12,
            RingType::Aliphatic => 13,
        },
        Chain(_) => {
            if component_chain_is_branched(comp, mol) {
                15
            } else {
                14
            }
        }
        Hydroxyl => 16,
        Carbonyl => 17,
        Carboxylate => 18,
        Amine => 19,
        Amide => 20,
        Sulfonamide => 21,
        Sulfonimide => 22,
    };
    debug_assert!((token as usize) < COMPONENT_VOCAB_SIZE);
    token
}

fn component_chain_is_branched(
    comp: &crate::mol_components::Component,
    mol: &MoleculeSmall,
) -> bool {
    let mut in_component = vec![false; mol.common.atoms.len()];

    for &atom_i in &comp.atoms {
        if atom_i < in_component.len() {
            in_component[atom_i] = true;
        }
    }

    comp.atoms.iter().any(|&atom_i| {
        mol.common.adjacency_list[atom_i]
            .iter()
            .filter(|&&nbr| in_component[nbr])
            .count()
            > 2
    })
}

fn component_pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests {
    use bio_files::BondType;
    use lin_alg::f64::Vec3;
    use na_seq::Element::{Bromine, Carbon, Chlorine};

    use super::*;
    use crate::{
        mol_components::{Component, ComponentType, Connection},
        molecules::{Atom, Bond},
    };

    fn test_atom(element: na_seq::Element) -> Atom {
        Atom {
            element,
            posit: Vec3::new_zero(),
            ..Default::default()
        }
    }

    fn test_bond(atom_0: usize, atom_1: usize) -> Bond {
        Bond {
            bond_type: BondType::Single,
            atom_0_sn: atom_0 as u32 + 1,
            atom_1_sn: atom_1 as u32 + 1,
            atom_0,
            atom_1,
            is_backbone: false,
        }
    }

    #[test]
    fn component_vocab_distinguishes_ring_atom_and_chain_chemistry() {
        let atoms = vec![
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Chlorine),
            test_atom(Bromine),
        ];
        let bonds = vec![
            test_bond(0, 1),
            test_bond(1, 2),
            test_bond(3, 4),
            test_bond(4, 5),
            test_bond(4, 6),
        ];
        let mol = MoleculeSmall::new(
            String::from("component_vocab"),
            atoms,
            bonds,
            HashMap::new(),
            None,
        );

        let aromatic_ring = Component {
            comp_type: ComponentType::Ring(crate::mol_components::RingComponent {
                num_atoms: 6,
                ring_type: RingType::Aromatic,
            }),
            atoms: vec![0, 1, 2, 3, 4, 5],
        };
        let saturated_ring = Component {
            comp_type: ComponentType::Ring(crate::mol_components::RingComponent {
                num_atoms: 6,
                ring_type: RingType::Saturated,
            }),
            atoms: vec![0, 1, 2, 3, 4, 5],
        };
        let linear_chain = Component {
            comp_type: ComponentType::Chain(3),
            atoms: vec![0, 1, 2],
        };
        let branched_chain = Component {
            comp_type: ComponentType::Chain(4),
            atoms: vec![3, 4, 5, 6],
        };
        let chlorine_atom = Component {
            comp_type: ComponentType::Atom(Chlorine),
            atoms: vec![7],
        };
        let bromine_atom = Component {
            comp_type: ComponentType::Atom(Bromine),
            atoms: vec![8],
        };

        assert_ne!(
            vocab_lookup_component(&aromatic_ring, &mol),
            vocab_lookup_component(&saturated_ring, &mol)
        );
        assert_ne!(
            vocab_lookup_component(&linear_chain, &mol),
            vocab_lookup_component(&branched_chain, &mol)
        );
        assert_ne!(
            vocab_lookup_component(&chlorine_atom, &mol),
            vocab_lookup_component(&bromine_atom, &mol)
        );
    }

    #[test]
    fn graph_data_component_aggregates_multi_bond_pairs() {
        let atoms = vec![
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
            test_atom(Carbon),
        ];
        let bonds = vec![
            test_bond(0, 2),
            test_bond(1, 3),
            test_bond(0, 1),
            test_bond(2, 3),
        ];
        let mol = MoleculeSmall::new(String::from("pair_agg"), atoms, bonds, HashMap::new(), None);
        let comps = MolComponents {
            components: vec![
                Component {
                    comp_type: ComponentType::Chain(2),
                    atoms: vec![0, 1],
                },
                Component {
                    comp_type: ComponentType::Chain(2),
                    atoms: vec![2, 3],
                },
            ],
            connections: vec![
                Connection {
                    comp_0: 0,
                    atom_0: 0,
                    comp_1: 1,
                    atom_1: 0,
                    shared_atoms: false,
                    rotatable: true,
                },
                Connection {
                    comp_0: 0,
                    atom_0: 1,
                    comp_1: 1,
                    atom_1: 1,
                    shared_atoms: false,
                    rotatable: false,
                },
            ],
        };

        let graph = GraphDataComponent::new(
            &mol,
            &comps,
            &ForceFieldParams::default(),
            &GnnAnalysisTools::default(),
        )
        .expect("component graph");

        assert_eq!(graph.scalars[0], 1.0 / 6.0);
        assert_eq!(graph.scalars[2], 1.0 / 6.0);
        assert_eq!(graph.adj[1], 1.0);
        assert_eq!(graph.adj[2], 1.0);

        let rotatable_feat_i = ((0 * 2 + 1) * PER_EDGE_COMP_FEATS) + 1;
        assert_eq!(graph.edge_feats[rotatable_feat_i], 0.5);
    }
}
