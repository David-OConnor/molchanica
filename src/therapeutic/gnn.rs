//! Graph neural nets: Represent molecules by their covalent bond connections, and related.
//! This includes atom and bond networks, and per-atom, per-bond, etc features. We are also
//! attempting to construct graphs from functional groups, pharmacophore features, and other
//! concepts. With and without geometry. (Distance/angles in space)
//!
//! We use *Graph classification or regression* as our primary technique in the set of Graph ML tools:
//! We view molecules as graphs, and infer properties of the entire graph. We may also implement
//! graph clustering

use std::{
    collections::{HashMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    io,
    iter::repeat_n,
};

use bio_files::{BondType, md_params::ForceFieldParams};
use na_seq::{
    Element,
    Element::{
        Bromine, Carbon, Chlorine, Fluorine, Hydrogen, Iodine, Nitrogen, Oxygen, Phosphorus, Sulfur,
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    mol_components::{ComponentType, MolComponents, build_adjacency_list_conn},
    molecules::{Atom, build_adjacency_list, small::MoleculeSmall},
    therapeutic::train::{BOND_SIGMA_SQ, EXCLUDE_HYDROGEN, FF_BUCKETS},
};

// Degree (Number of edges incident to a node), partial charge, geometry (radius from molecular centroid, mean neighbor distance),
// is H-bond acceptor, is H-bond donor, in aromatic ring.
// Keep this in sync with `GraphDataAtom::new`.
pub(in crate::therapeutic) const PER_ATOM_SCALARS: usize = 7;
// Bond type one-hot [single, double, triple, aromatic], plus scaled/modified proxies for r_0, k_b
// Keep this in sync with `GraphDataAtom::new`.
pub(in crate::therapeutic) const PER_EDGE_FEATS: usize = 6;

// Degree, component size
// Keep this in sync with `GraphDataComponent::new`
pub(in crate::therapeutic) const PER_COMP_SCALARS: usize = 2;
// Keep this in sync with `GraphDataComponent::new`
pub(in crate::therapeutic) const PER_EDGE_COMP_FEATS: usize = 1;
// For bond-stretching force-field params, used in the atom-based GNN.
const KB_REF: f32 = 300.0;

// Spacial (pharmacophore) GNN constants. Keep this in sync with (Where?)
// Node scalar features: [r_from_pharm_centroid, mean_pairwise_dist]. Keep these in sync with
// `GraphDataSpacial::new`.
pub(in crate::therapeutic) const PER_PHARM_SCALARS: usize = 2;
// Edge features: [scaled_dist, rbf_0, rbf_1, rbf_2, rbf_3]
pub(in crate::therapeutic) const PER_SPACIAL_EDGE_FEATS: usize = 5;
// Node type vocab: 0=pad, 1=HBondDonor, 2=HBondAcceptor, 3=Hydrophobic, 4=Aromatic
pub(in crate::therapeutic) const SPACIAL_VOCAB_SIZE: usize = 5;

// Tunable parameters for the spacial/pharmacophore GNN.
const SPACIAL_ADJ_SIGMA_SQ: f32 = 16.0; // sigma=4 Å for adjacency Gaussian
const SPACIAL_DIST_SCALE: f32 = 10.0; // Normalise raw distances to ~O(1)
const SPACIAL_RBF_SIGMA_SQ: f32 = 2.25; // sigma=1.5 Å for RBF basis functions
const SPACIAL_RBF_CENTERS: [f32; 4] = [2.0, 4.0, 6.0, 8.0]; // Å

const BOND_DIST_SPACIAL_SCALE: f32 = 0.15;

/// See `Graph Representation Learning` by William L Hamilton, 2020.
///
/// Defines tools used for analyzing graphs. Focuses on ones listed by Hamilton as specific to
/// graph-level analysis (as opposed to node-level analysis), as well as tools that can be used by both.
/// We this this to configure out GNNs to use various combinations of these in the ML analysis.
///
/// Note that we do not include "Bag of nodes" (Aggregating node-level statistics), as it's probalby
/// too naive.
///
/// todo: QC which of these make sense for molecule graphs (Of the various sorts we have). I feel like
/// todo: Molecule-base graphs represent a small subset of the general tyhpes used here, and they may not
/// todo: Make sense. For example, there are a lot of rules for molecule based graphs we can take advantage
/// todo of, and/or that make these tools less relevant (?)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(in crate::therapeutic) struct GnnAnalysisTools {
    /// Weisfeiler-Lehman (WL) kernel.
    /// "The idea with these approaches is to extract
    /// node-level features that contain more information than just their local ego graph,
    /// and then to aggregate these richer features into a graph-level representation."
    pub weisfeiler_lehman: bool,
    /// "simply count the occurrence of different small subgraph structures, usually called graphlets
    /// in this context. Formally, the graphlet kernel involves enumerating all possible graph structures
    /// of a particular size and counting how many times they occur in the full graph."
    /// If `Some`, contains a vec of node counts to analyze graphlets for.
    pub graphlets: Option<Vec<u8>>,
    /// "In these approaches, rather than enumerating graphlets, one simply
    /// examines the different kinds of paths that occur in the graph. For example, the
    /// random walk kernel proposed by Kashima et al. [2003] involves running ran-
    /// dom walks over the graph and then counting the occurrence of different degree
    /// sequences,3 while the shortest-path kernel of Borgwardt and Kriegel [2005] in-
    /// volves a similar idea but uses only the shortest-paths between nodes (rather
    /// 3Other node labels can also be used."
    pub path_based_methods: bool,
    /// "Local overlap statistics are simply functions of the number of common neighbors
    /// two nodes share, i.e. |N(u) ∩ N(v)|. For instance, the Sorensen index defines
    /// a matrix SSorenson ∈ R|V|×|V| of node-node neighborhood overlaps with entries
    /// given by...In general, these measures seek to quantify the overlap between node neighbor-
    /// hoods while minimizing any biases due to node degrees. There are many further
    /// variations of this approach in the literature [L¨u and Zhou, 2011]."
    pub local_overlap_statistics: bool,
    /// "The Katz index is the most basic global overlap statistic. To compute the Katz
    /// index we simply count the number of paths of all lengths between a pair of
    /// nodes... The Katz index is one example of a geo-
    /// metric series of matrices, variants of which occur frequently in graph anal-
    /// ysis and graph representation learning. "
    pub katz_index: bool,
    /// "One issue with the Katz index is that it is strongly biased by node degree.
    /// Equation (2.14) is generally going to give higher overall similarity scores when
    /// considering high-degree nodes, compared to low-degree ones, since high-degree
    /// nodes will generally be involved in more paths. To alleviate this, Leicht et al.
    /// [2006] propose an improved metric by considering the ratio between the actual
    /// number of observed paths and the number of expected paths between two nodes"
    pub lhn_similarity: bool,
    /// "Another set of global similarity measures consider random walks rather than
    /// exact counts of paths over the graph. For example, we can directly apply a
    /// variant of the famous PageRank approach [Page et al., 1999]4—known as the
    /// Personalized PageRank algorithm [Leskovec et al., 2020]—where we define the
    /// stochastic matrix P = AD−1 and compute:"
    pub random_walk_methods: bool,
}

impl GnnAnalysisTools {
    /// The feature order emitted by `GraphDataAtom::analysis_features`.
    ///
    /// At the moment we only emit features for the implemented AtomGraph analyses.
    /// Unsupported flags remain inert until we add concrete features for them.
    pub fn feature_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();

        if self.weisfeiler_lehman {
            names.extend([
                "atom_wl_unique_labels_base",
                "atom_wl_unique_labels_hop_1",
                "atom_wl_unique_labels_hop_2",
                "atom_wl_dominant_label_frac_hop_2",
            ]);
        }

        if let Some(graphlets) = &self.graphlets {
            for &size in graphlets {
                if size == 3 {
                    names.extend([
                        "atom_graphlet_wedge_frac",
                        "atom_graphlet_triangle_frac",
                        "atom_graphlet_transitivity",
                    ]);
                }
            }
        }

        if self.path_based_methods {
            names.extend([
                "atom_path_reachable_pair_frac",
                "atom_path_mean_shortest_path",
                "atom_path_diameter",
                "atom_path_long_shortest_path_frac",
            ]);
        }

        if self.local_overlap_statistics {
            names.extend([
                "atom_overlap_mean_jaccard",
                "atom_overlap_mean_sorensen",
            ]);
        }

        names
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_names().len()
    }
}

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

/// State for our atom-and-bond-based neural network. Atoms are nodes; covalent bonds are
/// edges.
///
/// Graph properties
/// ----------------
/// Edges: Undirected covalent-bond graph, with symmetric adjacency weights from a Gaussian
///   distance kernel
/// Underlying chemical graph has self-connections: No
/// GNN message-passing adjacency has self-loops: Yes (`adj[i,i] = 1.0` is added explicitly)
/// Nodes have multiple types (heterogeneous): No in the formal hetero-GNN sense; this is a
///   homogeneous atom graph, with element and FF bucket encoded as node embeddings/features
/// Edges have multiple types: Yes (single/double/triple/aromatic), represented as categorical
///   edge features; the adjacency only marks bonded neighbors and geometric proximity
/// Multipartite (edges can connect only to nodes of a diff type): No
/// Multiplex (Edges exist in layers; nodes are on all layers): No
/// Heterophily (Nodes are preferentially connected to others which have diff labels): Not fixed
///   by the construction; molecules can exhibit both homophilic and heterophilic local chemistry
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataAtom {
    /// Assign each element (Which we reasonably expect to encounter) an integer
    /// assignment to comply with the neural net's input requirements.
    pub elem_indices: Vec<i32>,
    /// An integer assignment for each force field type.
    pub ff_indices: Vec<i32>,
    /// Per-atom scalars:
    /// [degree, partial_charge, r_from_mol_centroid, mean_neighbor_dist,
    ///  is_h_bond_acceptor, is_h_bond_donor, in_aromatic_ring]
    pub scalars: Vec<f32>,
    /// Graph-level analysis features assembled according to
    /// `GnnAnalysisTools::feature_names()`. These are currently wired only into
    /// the AtomGraph branch.
    pub analysis_features: Vec<f32>,
    pub adj: Vec<f32>,
    pub edge_feats: Vec<f32>,
    pub num_atoms: usize,
}

impl GraphDataAtom {
    /// Converts raw Atoms and Bonds into Flat vectors for Tensors.
    /// Used by both Training and Inference.
    pub fn new(
        mol: &MoleculeSmall,
        ff_params: &ForceFieldParams,
        analysis_tools: &GnnAnalysisTools,
    ) -> io::Result<Self> {
        let (atoms, bonds, adj, old_to_new) = if EXCLUDE_HYDROGEN {
            let a: Vec<_> = mol
                .common
                .atoms
                .iter()
                .filter(|a| a.element != Hydrogen)
                .cloned()
                .collect();

            let mut sn_to_new = HashMap::with_capacity(a.len());
            let mut old_to_new = vec![None; mol.common.atoms.len()];
            for (new_i, a) in a.iter().enumerate() {
                sn_to_new.insert(a.serial_number, new_i);
            }
            for (old_i, atom) in mol.common.atoms.iter().enumerate() {
                old_to_new[old_i] = sn_to_new.get(&atom.serial_number).copied();
            }

            let mut bonds_ = Vec::new();
            for b in mol.common.bonds.iter() {
                if let (Some(&u), Some(&v)) =
                    (sn_to_new.get(&b.atom_0_sn), sn_to_new.get(&b.atom_1_sn))
                {
                    let mut b2 = b.clone();
                    b2.atom_0 = u;
                    b2.atom_1 = v;
                    bonds_.push(b2);
                }
            }
            let adj = build_adjacency_list(&bonds_, a.len());
            (a, bonds_, adj, old_to_new)
        } else {
            (
                mol.common.atoms.clone(),
                mol.common.bonds.clone(),
                mol.common.adjacency_list.clone(),
                (0..mol.common.atoms.len()).map(Some).collect(),
            )
        };

        let num_atoms = atoms.len();
        if num_atoms == 0 {
            return Err(io::Error::other("Molecule has 0 atoms"));
        }

        let Some(char) = &mol.characterization else {
            eprintln!("Missing char");
            return Err(io::Error::other("Missing characterization"));
        };

        let mut is_h_bond_acceptor = vec![false; num_atoms];
        for &old_i in &char.h_bond_acceptor {
            if let Some(Some(new_i)) = old_to_new.get(old_i) {
                is_h_bond_acceptor[*new_i] = true;
            }
        }

        let mut is_h_bond_donor = vec![false; num_atoms];
        for &old_i in &char.h_bond_donor {
            if let Some(Some(new_i)) = old_to_new.get(old_i) {
                is_h_bond_donor[*new_i] = true;
            }
        }

        let mut is_aromatic_ring = vec![false; num_atoms];
        for ring in &char.rings {
            for &old_i in &ring.atoms {
                if let Some(Some(new_i)) = old_to_new.get(old_i) {
                    is_aromatic_ring[*new_i] = true;
                }
            }
        }

        // Node features (Indices and Scalars)
        let mut elem_indices = Vec::with_capacity(num_atoms);
        let mut ff_indices = Vec::with_capacity(num_atoms);

        let mut scalars = Vec::with_capacity(num_atoms * PER_ATOM_SCALARS);

        let geom = atom_geom_scalars(&atoms, &adj);

        for (i, atom) in atoms.iter().enumerate() {
            elem_indices.push(vocab_lookup_element(atom.element));
            ff_indices.push(vocab_lookup_ff(atom.force_field_type.as_ref()));

            // Degree is the number of edges incident to a node.
            let degree = adj.get(i).map(|n| n.len()).unwrap_or(0);
            scalars.push(degree as f32 / 6.0);

            // Note: Including partial charge and FF type appears to be beneficial.
            scalars.push(atom.partial_charge.unwrap_or(0.0));

            // if let Some(lj_data) = ff_params
            //     .lennard_jones
            //     .get(atom.force_field_type.as_ref().unwrap())
            // {
            //     scalars.push(lj_data.sigma);
            //     scalars.push(lj_data.eps);
            // } else {
            //     eprintln!("Missing LJ for FF type {:?}", atom.force_field_type);
            //
            //     scalars.push(0.);
            //     scalars.push(0.);
            // }

            // scalars.push(ff_params.lennard_jones[atom.force_field_type.as_ref().unwrap()].sigma);
            // scalars.push(ff_params.lennard_jones[atom.force_field_type.as_ref().unwrap()].eps);

            let (r, mean_nb_dist) = geom[i];
            scalars.push(r);
            scalars.push(mean_nb_dist);

            let h_bond_acc = if is_h_bond_acceptor[i] { 1. } else { 0. };
            let h_bond_donor = if is_h_bond_donor[i] { 1. } else { 0. };
            scalars.push(h_bond_acc);
            scalars.push(h_bond_donor);

            let in_aromatic_ring = if is_aromatic_ring[i] { 1. } else { 0. };
            scalars.push(in_aromatic_ring);
        }

        let analysis_features = atom_graph_analysis_features(
            analysis_tools,
            &atoms,
            &adj,
            &is_h_bond_acceptor,
            &is_h_bond_donor,
            &is_aromatic_ring,
        );

        // Edge features and weighted adjacency
        let n_atoms_sq = num_atoms.pow(2);
        let mut adj_list = vec![0.; n_atoms_sq];
        let mut edge_feats = vec![0.; n_atoms_sq * PER_EDGE_FEATS];

        let edge_feats_i =
            |i: usize, j: usize, k: usize, n: usize| -> usize { (i * n + j) * PER_EDGE_FEATS + k };

        // Self loops
        for i in 0..num_atoms {
            adj_list[i * num_atoms + i] = 1.0;
        }

        for bond in &bonds {
            let a0 = bond.atom_0;
            let a1 = bond.atom_1;
            if a0 >= num_atoms || a1 >= num_atoms {
                continue;
            }

            let bond_type_one_hot = match bond.bond_type {
                BondType::Single => [1.0, 0.0, 0.0, 0.0],
                BondType::Double => [0.0, 1.0, 0.0, 0.0],
                BondType::Triple => [0.0, 0.0, 1.0, 0.0],
                BondType::Aromatic => [0.0, 0.0, 0.0, 1.0],
                _ => [1.0, 0.0, 0.0, 0.0],
            };

            // todo: Lennard Jones?

            let p1 = atoms[a0].posit;
            let p2 = atoms[a1].posit;
            let dist_sq = (p1 - p2).magnitude_squared() as f32;
            let dist = dist_sq.sqrt();

            {
                let ff0 = atoms[a0].force_field_type.clone().unwrap();
                let ff1 = atoms[a1].force_field_type.clone().unwrap();
                let bond_stretching = ff_params.get_bond(&(ff0.clone(), ff1.clone()), true);

                let (dr_norm, log_kb) = if let Some(v) = bond_stretching {
                    let dr = ((dist - v.r_0) / BOND_DIST_SPACIAL_SCALE).clamp(-5.0, 5.0);
                    let log_kb = (v.k_b / KB_REF).ln_1p();
                    (dr, log_kb)
                } else {
                    // Try a small set of coarse fallbacks; if none hit, leave the edge
                    // features as zero rather than panicking.
                    let candidates: &[(&str, &str)] = if (&ff0 == "cc" && ff1.starts_with("n"))
                        || (&ff1 == "cc" && ff0.starts_with("n"))
                    {
                        &[("cc", "n4"), ("cc", "n")]
                    } else if (&ff0 == "cg" && ff1.starts_with("c"))
                        || (&ff1 == "cg" && ff0.starts_with("c"))
                    {
                        &[("cg", "cg"), ("cc", "cc")]
                    } else if atoms[a0].element == Carbon && atoms[a1].element == Carbon {
                        &[("cc", "cc")]
                    } else if atoms[a0].element == Nitrogen && atoms[a1].element == Nitrogen {
                        &[("n", "n")]
                    } else {
                        &[]
                    };

                    let mut found = None;
                    for (a, b) in candidates {
                        if let Some(v) =
                            ff_params.get_bond(&((*a).to_owned(), (*b).to_owned()), false)
                        {
                            found = Some(v);
                            break;
                        }
                    }

                    if let Some(v) = found {
                        let dr = ((dist - v.r_0) / BOND_DIST_SPACIAL_SCALE).clamp(-5.0, 5.0);
                        let log_kb = (v.k_b / KB_REF).ln_1p();
                        (dr, log_kb)
                    } else {
                        eprintln!(
                            "Missing bond stretching and no fallback for bond {bond:?}. \nAtoms {ff0} | {ff1}. Using zero bond-stretch edge features.",
                        );
                        (0.0, 0.0)
                    }
                };

                for (k, &v) in bond_type_one_hot.iter().enumerate() {
                    edge_feats[edge_feats_i(a0, a1, k, num_atoms)] = v;
                    edge_feats[edge_feats_i(a1, a0, k, num_atoms)] = v;
                }
                edge_feats[edge_feats_i(a0, a1, 4, num_atoms)] = dr_norm;
                edge_feats[edge_feats_i(a0, a1, 5, num_atoms)] = log_kb;

                edge_feats[edge_feats_i(a1, a0, 4, num_atoms)] = dr_norm;
                edge_feats[edge_feats_i(a1, a0, 5, num_atoms)] = log_kb;
            }

            let k = (-dist_sq / (2.0 * BOND_SIGMA_SQ)).exp();
            let weight = k;

            adj_list[a0 * num_atoms + a1] = weight;
            adj_list[a1 * num_atoms + a0] = weight;
        }

        // Note: symmetric normalization (D^-1/2 A D^-1/2) is applied in the model's
        // forward pass *after* the learned edge gate, so we ship the raw weighted
        // adjacency (with self-loops) here. Bond order itself lives in `edge_feats`.

        Ok(Self {
            elem_indices,
            ff_indices,
            scalars,
            analysis_features,
            adj: adj_list,
            edge_feats,
            num_atoms,
        })
    }
}

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
///   links, with `edge_feats` marking whether the connection uses shared atoms
/// Multipartite (edges can connect only to nodes of a diff type): No
/// Multiplex (Edges exist in layers; nodes are on all layers): No
/// Heterophily (Nodes are preferentially connected to others which have diff labels): Not fixed
///   by the construction; components can connect to either similar or different categories
#[derive(Clone, Debug)]
pub(in crate::therapeutic) struct GraphDataComponent {
    pub comp_type_indices: Vec<i32>,
    pub scalars: Vec<f32>,
    pub adj: Vec<f32>,
    pub edge_feats: Vec<f32>,
    pub num_comps: usize,
}

/// Converts component nodes and inter-component connections into flat vectors for tensors.
/// Used by both Training and Inference.
impl GraphDataComponent {
    pub fn new(mol_comps: &MolComponents) -> io::Result<Self> {
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

            // An edge feature for if the connection uses shared atoms or not.
            let shared = if conn.shared_atoms { 1. } else { 0. };
            edge_feats[edge_feats_i(a0, a1, 0, num_comps)] = shared;
            edge_feats[edge_feats_i(a1, a0, 0, num_comps)] = shared;
        }

        // Symmetric normalization is applied in the model after gating; see GraphDataAtom.

        Ok(Self {
            comp_type_indices,
            scalars,
            adj: adj_list,
            edge_feats,
            num_comps,
        })
    }
}

/// These are numerical properties of individual atoms. Partial charge, FF type etc.
fn atom_geom_scalars(atoms: &[Atom], adj: &[Vec<usize>]) -> Vec<(f32, f32)> {
    let n = atoms.len().max(1);

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for a in atoms {
        cx += a.posit.x;
        cy += a.posit.y;
        cz += a.posit.z;
    }

    let inv_n = 1.0 / (n as f64);
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    // Scale by RMS radius to keep numbers ~O(1)
    let mut r2_sum = 0.0f64;
    for a in atoms {
        let dx = a.posit.x - cx;
        let dy = a.posit.y - cy;
        let dz = a.posit.z - cz;
        r2_sum += dx * dx + dy * dy + dz * dz;
    }

    let rms = (r2_sum * inv_n).sqrt().max(1e-6);

    let mut out = Vec::with_capacity(atoms.len());

    for (i, a) in atoms.iter().enumerate() {
        let dx = (a.posit.x - cx) / rms;
        let dy = (a.posit.y - cy) / rms;
        let dz = (a.posit.z - cz) / rms;

        let r = (dx * dx + dy * dy + dz * dz).sqrt(); // invariant

        // Mean neighbor distance
        let mut dist_sum = 0.0_f64;
        let mut count = 0_u32;
        for &j in adj.get(i).unwrap_or(&Vec::new()).iter() {
            let b = &atoms[j];
            dist_sum += (a.posit - b.posit).magnitude();
            count += 1;
        }
        let mean_nb_dist = if count > 0 {
            dist_sum as f32 / count as f32
        } else {
            0.0
        };

        out.push((r as f32, mean_nb_dist));
    }

    out
}

fn atom_graph_analysis_features(
    tools: &GnnAnalysisTools,
    atoms: &[Atom],
    adj: &[Vec<usize>],
    is_h_bond_acceptor: &[bool],
    is_h_bond_donor: &[bool],
    is_aromatic_ring: &[bool],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(tools.feature_dim());

    if tools.weisfeiler_lehman {
        out.extend(wl_atom_graph_features(
            atoms,
            adj,
            is_h_bond_acceptor,
            is_h_bond_donor,
            is_aromatic_ring,
        ));
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

    out
}

fn wl_atom_graph_features(
    atoms: &[Atom],
    adj: &[Vec<usize>],
    is_h_bond_acceptor: &[bool],
    is_h_bond_donor: &[bool],
    is_aromatic_ring: &[bool],
) -> [f32; 4] {
    let n = atoms.len().max(1) as f32;

    let base_labels: Vec<u64> = atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let mut hasher = DefaultHasher::new();
            vocab_lookup_element(atom.element).hash(&mut hasher);
            adj[i].len().hash(&mut hasher);
            is_h_bond_acceptor[i].hash(&mut hasher);
            is_h_bond_donor[i].hash(&mut hasher);
            is_aromatic_ring[i].hash(&mut hasher);
            hasher.finish()
        })
        .collect();

    let hop_1 = wl_refine_labels(&base_labels, adj);
    let hop_2 = wl_refine_labels(&hop_1, adj);

    [
        unique_label_count(&base_labels) as f32 / n,
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

fn graphlet_size_3_features(adj: &[Vec<usize>]) -> [f32; 3] {
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

fn path_based_features(adj: &[Vec<usize>]) -> [f32; 4] {
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

fn local_overlap_features(adj: &[Vec<usize>]) -> [f32; 2] {
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

fn has_edge(adj_mat: &[bool], n: usize, i: usize, j: usize) -> bool {
    adj_mat[i * n + j]
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

fn choose2(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1)) / 2
}

fn choose3(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1))
        .saturating_mul(n.saturating_sub(2))
        / 6
}

/// Helper: Pads a single graph to MAX_ATOMS.
/// Returns (PaddedNodes, PaddedAdj, PaddedMask) as flat vectors.
///
/// For comps too.
pub(in crate::therapeutic) fn pad_adj_and_mask(
    raw_adj: &[f32],
    num_atoms: usize,
    max: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = num_atoms.min(max);

    // Mask: 1.0 for atoms, 0.0 for pad
    let mut p_mask = Vec::with_capacity(max);
    p_mask.extend(repeat_n(1.0, n));
    p_mask.extend(repeat_n(0.0, max - n));

    //Adj: Reconstruct row-by-row to handle 2D padding
    let mut p_adj = Vec::with_capacity(max * max);
    for r in 0..n {
        let row_start = r * num_atoms; // Input is flat [num_atoms * num_atoms]
        // Copy valid columns
        p_adj.extend_from_slice(&raw_adj[row_start..row_start + n]);
        // Pad columns (right side of matrix)
        p_adj.extend(repeat_n(0.0, max - n));
    }
    // Pad rows (bottom of matrix)
    let remaining_rows = max - n;
    p_adj.extend(repeat_n(0.0, remaining_rows * max));

    (p_adj, p_mask)
}

/// For comps too
pub(in crate::therapeutic) fn pad_edge_feats(
    edge_feats: &[f32], // [num_atoms*num_atoms*PER_EDGE_FEATS]
    num_atoms: usize,
    num_feats: usize,
    max: usize,
) -> Vec<f32> {
    let n = num_atoms.min(max);

    let mut out = vec![0.0f32; max.pow(2) * num_feats];

    for i in 0..n {
        for j in 0..n {
            let src_base = (i * num_atoms + j) * num_feats;
            let dst_base = (i * max + j) * num_feats;
            out[dst_base..dst_base + num_feats]
                .copy_from_slice(&edge_feats[src_base..src_base + num_feats]);
        }
    }

    out
}

/// Pad a 1-D index vector to `max`, filling the tail with `0` (the padding token).
pub(in crate::therapeutic) fn pad_indices(src: &[i32], num: usize, max: usize) -> Vec<i32> {
    let n = num.min(max);
    let mut v = Vec::with_capacity(max);
    v.extend_from_slice(&src[0..n]);
    v.extend(repeat_n(0_i32, max - n));
    v
}

/// Pad a per-node scalar block (`num × n_per` flat) to `max × n_per`.
pub(in crate::therapeutic) fn pad_scalars(
    src: &[f32],
    num: usize,
    n_per: usize,
    max: usize,
) -> Vec<f32> {
    let n = num.min(max);
    let mut v = Vec::with_capacity(max * n_per);
    v.extend_from_slice(&src[0..n * n_per]);
    v.extend(repeat_n(0.0_f32, (max - n) * n_per));
    v
}

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
    /// Gaussian distance-weighted adjacency (fully connected + self-loops). Flat [N²].
    pub adj: Vec<f32>,
    /// Per-edge geometric features: [scaled_dist, rbf0..rbf3]. Flat [N² × PER_SPACIAL_EDGE_FEATS].
    pub edge_feats: Vec<f32>,
    pub num_nodes: usize,
}

impl GraphDataSpacial {
    pub(in crate::therapeutic) fn empty() -> Self {
        Self {
            pharm_type_indices: Vec::new(),
            scalars: Vec::new(),
            adj: Vec::new(),
            edge_feats: Vec::new(),
            num_nodes: 0,
        }
    }

    pub fn new(mol: &MoleculeSmall) -> io::Result<Self> {
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
            adj,
            edge_feats,
            num_nodes,
        })
    }
}

fn vocab_lookup_ff(ff: Option<&String>) -> i32 {
    // 0 is Padding.
    match ff {
        Some(s) => {
            // Hash to range [1..20]
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            ((h.finish() % (FF_BUCKETS as u64)) + 1) as i32
        }
        None => (FF_BUCKETS as i32) + 1, // Unknown bucket
    }
}

/// Maps element to values the neural net can use.
fn vocab_lookup_element(el: Element) -> i32 {
    // 0 is reserved for Padding in the Batcher, so we start at 1.
    match el {
        Hydrogen => 1,
        Carbon => 2,
        Nitrogen => 3,
        Oxygen => 4,
        Fluorine => 5,
        Phosphorus => 6,
        Sulfur => 7,
        Chlorine => 8,
        Bromine => 9,
        Iodine => 10,
        _ => 11, // "Other" bucket
    }
}

/// Maps component type to integer indices for the embedding layer.
/// 0 is reserved for padding in the Batcher, so we start at 1.
fn vocab_lookup_component(comp_type: &ComponentType) -> i32 {
    use ComponentType::*;
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

#[cfg(test)]
mod tests {
    use super::{
        atom_graph_analysis_tools, graphlet_size_3_features, local_overlap_features,
        path_based_features,
    };

    #[test]
    fn atom_graph_analysis_feature_count_matches_names() {
        let tools = atom_graph_analysis_tools();
        assert_eq!(tools.feature_dim(), tools.feature_names().len());
    }

    #[test]
    fn graphlets_detect_triangle() {
        let triangle = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let feats = graphlet_size_3_features(&triangle);

        assert_eq!(feats[0], 0.0);
        assert_eq!(feats[1], 1.0);
        assert_eq!(feats[2], 1.0);
    }

    #[test]
    fn path_and_overlap_features_detect_chain_structure() {
        let chain = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];

        let path_feats = path_based_features(&chain);
        assert_eq!(path_feats[0], 1.0);
        assert!(path_feats[2] >= 0.75);

        let overlap_feats = local_overlap_features(&chain);
        assert_eq!(overlap_feats[0], 0.0);
        assert_eq!(overlap_feats[1], 0.0);
    }
}
