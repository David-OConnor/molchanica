//! Code for setting molecule geometry based on atoms and bonds. Can be used
//! to set initial geometry (e.g. from after loading from SMILES), or for correctly
//! freeform geometry from the molecule editor.

use std::{
    collections::{HashSet, VecDeque},
    f64::consts::PI,
};

use bio_files::BondType;
use dynamics::find_tetra_posits;
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::molecules::{
    Atom, Bond,
    common::{BondGeom, MoleculeCommon, find_appended_posit},
};

// ── Public impl ───────────────────────────────────────────────────────────────

impl MoleculeCommon {
    /// Assign 3D coordinates to all atoms.
    ///
    /// Strategy
    /// 1. Detect all rings (SSSR via DFS back-edge + BFS shortest-path per back-edge).
    /// 2. Place each ring as a regular polygon.  Fused rings are anchored to the
    ///    shared edge of any already-placed ring.
    /// 3. BFS from all placed ring atoms to position chains and substituents.
    /// 4. Disconnected components are offset along X so they don't overlap.
    pub fn assign_posits(&mut self) {
        let n = self.atoms.len();
        if n == 0 {
            return;
        }

        let mut positioned = vec![false; n];
        let mut component_x = 0.0_f64;

        loop {
            let start = match (0..n).find(|&i| !positioned[i]) {
                None => break,
                Some(s) => s,
            };

            let offset = Vec3::new(component_x, 0., 0.);
            let component = find_component(&self.adjacency_list, start, n);
            let rings = find_rings_sssr(&component, &self.adjacency_list);

            if rings.is_empty() {
                // Acyclic component: anchor first atom at offset, BFS does the rest.
                self.atoms[start].posit = offset;
                positioned[start] = true;
            } else {
                place_ring_systems(
                    &rings,
                    &mut self.atoms,
                    &mut positioned,
                    &self.bonds,
                    &self.adjacency_list,
                    offset,
                );
            }

            // BFS from all placed atoms to reach substituents and chains.
            let mut queue: VecDeque<usize> = component
                .iter()
                .copied()
                .filter(|&a| positioned[a])
                .collect();

            bfs_place_substituents(
                &mut queue,
                &mut self.atoms,
                &mut positioned,
                &self.bonds,
                &self.adjacency_list,
            );

            // Advance offset for the next disconnected component.
            let max_x = component
                .iter()
                .map(|&i| self.atoms[i].posit.x)
                .fold(component_x, f64::max);
            component_x = max_x + 5.;
        }

        self.reset_posits();
    }

    /// Geometry cleanup for the editor.
    ///
    /// Less aggressive than `assign_posits`: existing positions are kept whenever
    /// the local geometry is physically reasonable.  Only atoms whose bonds to *all*
    /// neighbours are significantly wrong (outside `[LO, HI] × expected`) are
    /// considered "floating" and re-placed via BFS from the well-placed atoms that
    /// surround them.
    ///
    /// Typical trigger: the user dragged one atom in the editor, leaving that atom
    /// far from its neighbours while everything else is fine.
    ///
    /// Falls back to a full `assign_posits` when no atom can serve as an anchor
    /// (e.g. all atoms freshly added without positions).
    pub fn cleanup_geometry(&mut self) {
        let n = self.atoms.len();
        if n == 0 {
            return;
        }

        // Fraction-of-expected-bond-length range we consider "reasonable".
        // Generous enough not to disturb minor editor distortions, strict enough
        // to catch atoms that have been dragged far from their neighbours.
        const LO: f64 = 0.4;
        const HI: f64 = 2.5;

        // Build per-atom anchor flags.
        // An atom is anchored (kept in place) if at least one of its bonds is
        // within [LO, HI] × expected length.  Isolated atoms (no bonds) are
        // always anchored.  Atoms with *all* bonds outside the range are floating
        // and will be re-placed by the BFS pass below.
        let anchored: Vec<bool> = (0..n)
            .map(|i| {
                let mut any_ok = false;
                let mut has_bond = false;

                for b in &self.bonds {
                    let j = if b.atom_0 == i {
                        b.atom_1
                    } else if b.atom_1 == i {
                        b.atom_0
                    } else {
                        continue;
                    };
                    has_bond = true;

                    let expected = estimate_bond_length(
                        self.atoms[i].element,
                        self.atoms[j].element,
                        b.bond_type,
                    );
                    let actual = (self.atoms[i].posit - self.atoms[j].posit).magnitude();

                    if actual >= expected * LO && actual <= expected * HI {
                        any_ok = true;
                        break; // one good bond is enough
                    }
                }

                !has_bond || any_ok // isolated atoms count as anchored
            })
            .collect();

        // If nothing is anchored (e.g. a fresh molecule where every atom sits at
        // the origin), there are no reference points for a partial BFS repair;
        // do a full re-assignment instead.
        let any_anchored = anchored.iter().any(|&a| a);
        if !any_anchored {
            self.assign_posits();
            return;
        }

        // BFS from all anchored atoms.  Floating atoms (positioned = false) are
        // re-placed using the same geometry logic as assign_posits; anchored atoms
        // (positioned = true from the start) are never moved.
        let mut positioned = anchored.clone();
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| anchored[i]).collect();

        bfs_place_substituents(
            &mut queue,
            &mut self.atoms,
            &mut positioned,
            &self.bonds,
            &self.adjacency_list,
        );

        self.reset_posits();
    }
}

// ── Graph utilities ───────────────────────────────────────────────────────────

/// Collect all atom indices reachable from `start` (one connected component).
fn find_component(adj: &[Vec<usize>], start: usize, n: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut out = Vec::new();
    let mut q = VecDeque::new();
    visited[start] = true;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        out.push(u);
        for &v in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                q.push_back(v);
            }
        }
    }
    out
}

/// SSSR-like ring detection.
///
/// 1. DFS on the component subgraph to collect back-edges.
/// 2. For each back-edge (u, v), BFS finds the *shortest* cycle containing it
///    (path from v to u not using the direct edge u–v, then closing u→v).
///
/// Duplicate rings (same atom set) are filtered out.
fn find_rings_sssr(component: &[usize], adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n_total = adj.len();
    let mut visited = vec![false; n_total];
    let mut in_stack = vec![false; n_total];
    let mut back_edges: Vec<(usize, usize)> = Vec::new();

    for &start in component {
        if !visited[start] {
            dfs_back_edges(
                start,
                usize::MAX,
                adj,
                &mut visited,
                &mut in_stack,
                &mut back_edges,
            );
        }
    }

    let mut rings: Vec<Vec<usize>> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();

    for (u, v) in back_edges {
        let skip = (u.min(v), u.max(v));
        if let Some(path) = bfs_shortest_path(v, u, skip, adj) {
            let mut key = path.clone();
            key.sort_unstable();
            if seen.insert(key) {
                rings.push(path);
            }
        }
    }

    rings
}

fn dfs_back_edges(
    u: usize,
    parent: usize,
    adj: &[Vec<usize>],
    visited: &mut Vec<bool>,
    in_stack: &mut Vec<bool>,
    back_edges: &mut Vec<(usize, usize)>,
) {
    visited[u] = true;
    in_stack[u] = true;
    for &v in &adj[u] {
        if v == parent {
            continue;
        }
        if in_stack[v] {
            back_edges.push((u, v));
        } else if !visited[v] {
            dfs_back_edges(v, u, adj, visited, in_stack, back_edges);
        }
    }
    in_stack[u] = false;
}

/// BFS shortest path from `start` to `end`, skipping `skip_edge` (stored as (lo, hi)).
/// Returns `[start, …, end]` or `None` if unreachable.
fn bfs_shortest_path(
    start: usize,
    end: usize,
    skip_edge: (usize, usize),
    adj: &[Vec<usize>],
) -> Option<Vec<usize>> {
    let n = adj.len();
    let mut prev = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut q = VecDeque::new();
    visited[start] = true;
    q.push_back(start);

    while let Some(u) = q.pop_front() {
        if u == end {
            let mut path = Vec::new();
            let mut cur = end;
            loop {
                path.push(cur);
                let p = prev[cur];
                if p == usize::MAX {
                    break;
                }
                cur = p;
            }
            path.reverse();
            return Some(path);
        }
        for &v in &adj[u] {
            let e = (u.min(v), u.max(v));
            if e == skip_edge || visited[v] {
                continue;
            }
            visited[v] = true;
            prev[v] = u;
            q.push_back(v);
        }
    }
    None
}

// ── Ring placement ────────────────────────────────────────────────────────────

/// Place all rings in a component.
///
/// Iterates until every ring is handled:
///  - Rings with no placed atoms AND external connection → `place_ring_attached`.
///  - Rings with no placed atoms AND no external connection → regular polygon at `offset`.
///  - Rings with ≥ 2 adjacent placed atoms → fused ring anchored to that edge.
///  - Rings with exactly 1 placed atom (spiro centre) → spiro placement.
///  - Rings already fully placed → skip.
///
/// Sort order: rings that share atoms with other rings (fused) come first so that
/// pendant rings can anchor themselves to the already-placed core.
fn place_ring_systems(
    rings: &[Vec<usize>],
    atoms: &mut [Atom],
    positioned: &mut Vec<bool>,
    bonds: &[Bond],
    adj: &[Vec<usize>],
    offset: Vec3,
) {
    // How many atoms in ring[ri] also appear in some other ring?
    // Rings with more shared atoms are more "central" (fused) and should be placed first.
    let shared_counts: Vec<usize> = (0..rings.len())
        .map(|ri| {
            let set: HashSet<usize> = rings[ri].iter().copied().collect();
            rings
                .iter()
                .enumerate()
                .filter(|(rj, _)| *rj != ri)
                .flat_map(|(_, other)| other.iter())
                .filter(|&&a| set.contains(&a))
                .count()
        })
        .collect();

    let mut order: Vec<usize> = (0..rings.len()).collect();
    // Primary sort: more shared atoms first (fused/central rings before pendant rings).
    // Tie-break: smaller rings first (better for 5+6 bicyclics etc.).
    order.sort_by(|&a, &b| {
        shared_counts[b]
            .cmp(&shared_counts[a])
            .then(rings[a].len().cmp(&rings[b].len()))
    });

    let mut done = vec![false; rings.len()];
    let mut progress = true;

    while progress {
        progress = false;
        for &ri in &order {
            if done[ri] {
                continue;
            }
            let ring = &rings[ri];
            let placed_count = ring.iter().filter(|&&a| positioned[a]).count();

            if placed_count == ring.len() {
                done[ri] = true;
                progress = true;
                continue;
            }

            if placed_count == 0 {
                // Check whether any ring atom has an already-placed *external* neighbour
                // (i.e., a neighbour that is not in this ring but is already positioned).
                // If so, anchor the ring to that connection; otherwise use offset.
                let has_external = ring
                    .iter()
                    .any(|&a| adj[a].iter().any(|&nb| positioned[nb]));

                if has_external {
                    if place_ring_attached(ring, atoms, positioned, bonds, adj) {
                        done[ri] = true;
                        progress = true;
                    }
                    // If place_ring_attached fails (external not yet placed), defer.
                } else {
                    place_ring_regular(ring, atoms, positioned, bonds, offset);
                    done[ri] = true;
                    progress = true;
                }
            } else if place_ring_fused(ring, atoms, positioned, bonds) {
                done[ri] = true;
                progress = true;
            } else if placed_count == 1 {
                // Spiro ring: one shared atom, no shared edge.  Place the two
                // ring-neighbours of the spiro centre via tetrahedral geometry,
                // then let place_ring_fused finish the polygon.
                if place_ring_spiro(ring, atoms, positioned, bonds, adj) {
                    place_ring_fused(ring, atoms, positioned, bonds);
                    done[ri] = true;
                    progress = true;
                }
            }
        }
    }
}

/// Place a ring as a regular N-gon centered at `center` in the XY plane.
/// The first atom is placed at 12 o'clock; subsequent atoms go clockwise.
fn place_ring_regular(
    ring: &[usize],
    atoms: &mut [Atom],
    positioned: &mut Vec<bool>,
    bonds: &[Bond],
    center: Vec3,
) {
    let n = ring.len();
    if n == 0 {
        return;
    }
    let bond_len = avg_ring_bond_len(ring, atoms, bonds);
    let radius = ring_circumradius(bond_len, n);

    for (i, &ai) in ring.iter().enumerate() {
        let angle = PI / 2.0 - 2.0 * PI * i as f64 / n as f64;
        atoms[ai].posit = center + Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.);
        positioned[ai] = true;
    }
}

/// Attempt to place a fused ring anchored to the first adjacent pair of
/// already-placed atoms found in ring order.  Returns `true` on success.
fn place_ring_fused(
    ring: &[usize],
    atoms: &mut [Atom],
    positioned: &mut Vec<bool>,
    bonds: &[Bond],
) -> bool {
    let n = ring.len();

    // Find the first adjacent pair of placed atoms (including the wrap-around edge).
    let anchor = (0..n).find_map(|i| {
        let j = (i + 1) % n;
        let a = ring[i];
        let b = ring[j];
        if positioned[a] && positioned[b] {
            Some((i, a, b))
        } else {
            None
        }
    });

    let (i0, a0, a1) = match anchor {
        Some(x) => x,
        None => return false,
    };

    let pos0 = atoms[a0].posit;
    let pos1 = atoms[a1].posit;

    let edge_len = (pos1 - pos0).magnitude();
    if edge_len < 1e-6 {
        return false;
    }

    let radius = ring_circumradius(edge_len, n);
    let edge_mid = (pos0 + pos1) * 0.5;
    let edge_dir = (pos1 - pos0) * (1.0 / edge_len);
    // 90° CCW rotation in XY plane.
    let perp = Vec3::new(-edge_dir.y, edge_dir.x, 0.0);

    // Distance from edge midpoint to ring centre.
    let half_edge = edge_len * 0.5;
    let center_dist = (radius * radius - half_edge * half_edge).max(0.0).sqrt();

    // The new ring goes on the OPPOSITE side of the edge from the existing ring's
    // centre of mass, so fused rings extend outward rather than overlapping.
    let (com_sum, nplaced) = ring
        .iter()
        .filter(|&&a| positioned[a])
        .fold((Vec3::new(0., 0., 0.), 0usize), |(acc, cnt), &a| {
            (acc + atoms[a].posit, cnt + 1)
        });
    let existing_com = if nplaced > 0 {
        com_sum * (1.0 / nplaced as f64)
    } else {
        edge_mid
    };

    let new_center = if (existing_com - edge_mid).dot(perp) > 0.0 {
        edge_mid - perp * center_dist
    } else {
        edge_mid + perp * center_dist
    };

    // Determine traversal direction (CW or CCW) by matching a1's actual angle.
    let theta0 = {
        let d = pos0 - new_center;
        d.y.atan2(d.x)
    };
    let theta1_actual = {
        let d = pos1 - new_center;
        d.y.atan2(d.x)
    };
    let step = 2.0 * PI / n as f64;
    let step_sign: f64 =
        if angle_diff(theta1_actual, theta0 + step) <= angle_diff(theta1_actual, theta0 - step) {
            1.0
        } else {
            -1.0
        };

    for i in 0..n {
        let ai = ring[i];
        if positioned[ai] {
            continue;
        }
        let steps = ((i as isize - i0 as isize).rem_euclid(n as isize)) as f64;
        let angle = theta0 + steps * step * step_sign;
        atoms[ai].posit = new_center + Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.);
        positioned[ai] = true;
    }

    true
}

/// Places a ring that has **no already-positioned ring atoms** but IS connected to
/// the rest of the molecule via at least one external bond.
///
/// Algorithm:
/// 1. Find the ring atom `ra` whose external (non-ring) neighbour `ext_nb` is placed.
/// 2. Position `ra` using the same geometry logic as BFS substituent placement
///    (`find_appended_posit` on `ext_nb`).
/// 3. The ring centre is placed at `ra_pos + bond_dir × circumradius`, extending
///    the ring away from `ext_nb`.
/// 4. Remaining ring atoms are placed as a regular polygon around this centre.
///
/// Returns `true` on success.
fn place_ring_attached(
    ring: &[usize],
    atoms: &mut [Atom],
    positioned: &mut Vec<bool>,
    bonds: &[Bond],
    adj: &[Vec<usize>],
) -> bool {
    let n = ring.len();

    // Find the first ring atom that has a placed external neighbour.
    let (ra_idx, ra, ext_nb) = match ring.iter().enumerate().find_map(|(idx, &a)| {
        adj[a]
            .iter()
            .find(|&&nb| positioned[nb])
            .map(|&nb| (idx, a, nb))
    }) {
        Some(x) => x,
        None => return false,
    };

    let ext_nb_pos = atoms[ext_nb].posit;

    // All placed neighbours of ext_nb (the already-positioned atoms around it).
    let ext_adj_placed: Vec<usize> = adj[ext_nb]
        .iter()
        .copied()
        .filter(|&nb| positioned[nb])
        .collect();

    let ext_geom = geom_for_atom(ext_nb, bonds);
    let bt = bond_type_between(ra, ext_nb, bonds);
    let bond_len = estimate_bond_length(atoms[ra].element, atoms[ext_nb].element, bt);
    let ra_element = atoms[ra].element;

    // Place ra using the geometry of ext_nb, exactly like BFS does for substituents.
    let ra_pos = find_appended_posit(
        ext_nb_pos,
        atoms,
        &ext_adj_placed,
        Some(bond_len),
        ra_element,
        ext_geom,
    )
    .unwrap_or_else(|| ext_nb_pos + Vec3::new(bond_len, 0., 0.));

    // Direction ext_nb → ra.  The ring centre is placed further in this direction
    // so the ring polygon extends away from the core.
    let bond_dir = {
        let d = ra_pos - ext_nb_pos;
        if d.magnitude_squared() < 1e-12 {
            Vec3::new(1., 0., 0.)
        } else {
            d.to_normalized()
        }
    };

    let avg_bl = avg_ring_bond_len(ring, atoms, bonds);
    let radius = ring_circumradius(avg_bl, n);
    let ring_center = ra_pos + bond_dir * radius;

    // Place ra (the anchor atom) on the circumcircle, then fill the rest CCW.
    atoms[ra].posit = ra_pos;
    positioned[ra] = true;

    let ra_angle = {
        let d = ra_pos - ring_center;
        d.y.atan2(d.x)
    };
    let step = 2.0 * PI / n as f64;

    for k in 1..n {
        let ai = ring[(ra_idx + k) % n];
        if positioned[ai] {
            continue;
        }
        let angle = ra_angle + k as f64 * step;
        atoms[ai].posit = ring_center + Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.);
        positioned[ai] = true;
    }

    true
}

/// Handles a spiro ring — a ring that shares exactly ONE atom (the spiro centre)
/// with previously-placed rings.  `place_ring_fused` cannot proceed without a
/// shared *edge*, so this function first seeds the two ring-atoms adjacent to
/// the spiro centre using tetrahedral geometry, after which `place_ring_fused`
/// can complete the polygon.
///
/// Returns `true` if placement was possible.
fn place_ring_spiro(
    ring: &[usize],
    atoms: &mut [Atom],
    positioned: &mut Vec<bool>,
    bonds: &[Bond],
    adj: &[Vec<usize>],
) -> bool {
    let n = ring.len();

    // Find the single placed atom (spiro centre).
    let (sc_idx, sc) = match ring.iter().enumerate().find(|&(_, &a)| positioned[a]) {
        Some((i, &a)) => (i, a),
        None => return false,
    };

    // Collect placed neighbours of the spiro centre (all from the already-placed ring).
    let placed_nbrs: Vec<usize> = adj[sc]
        .iter()
        .copied()
        .filter(|&nb| positioned[nb])
        .collect();

    if placed_nbrs.len() < 2 {
        return false; // not enough context to determine orientation
    }

    let sc_pos = atoms[sc].posit;
    let pn0 = atoms[placed_nbrs[0]].posit;
    let pn1 = atoms[placed_nbrs[1]].posit;

    // The two tetrahedral positions orthogonal to the existing ring bonds.
    let (tp0, tp1) = find_tetra_posits(sc_pos, pn0, pn1);

    let prev_ai = ring[(sc_idx + n - 1) % n];
    let next_ai = ring[(sc_idx + 1) % n];

    let len_prev = estimate_bond_length(
        atoms[sc].element,
        atoms[prev_ai].element,
        bond_type_between(sc, prev_ai, bonds),
    );
    let len_next = estimate_bond_length(
        atoms[sc].element,
        atoms[next_ai].element,
        bond_type_between(sc, next_ai, bonds),
    );

    atoms[prev_ai].posit = sc_pos + (tp0 - sc_pos).to_normalized() * len_prev;
    positioned[prev_ai] = true;
    atoms[next_ai].posit = sc_pos + (tp1 - sc_pos).to_normalized() * len_next;
    positioned[next_ai] = true;

    true
}

// ── BFS substituent placement ─────────────────────────────────────────────────

fn bfs_place_substituents(
    queue: &mut VecDeque<usize>,
    atoms: &mut Vec<Atom>,
    positioned: &mut Vec<bool>,
    bonds: &[Bond],
    adj: &[Vec<usize>],
) {
    while let Some(u) = queue.pop_front() {
        let geom = geom_for_atom(u, bonds);
        let posit_u = atoms[u].posit;
        let neighbours: Vec<usize> = adj[u].to_vec();

        for v in neighbours {
            if positioned[v] {
                continue;
            }

            let adj_placed: Vec<usize> = adj[u]
                .iter()
                .copied()
                .filter(|&w| w != v && positioned[w])
                .collect();

            let bt = bond_type_between(u, v, bonds);
            let bond_len = estimate_bond_length(atoms[u].element, atoms[v].element, bt);

            // Read v's element before passing `atoms` to find_appended_posit.
            let v_element = atoms[v].element;
            let n_placed_before_v = adj_placed.len();

            let p =
                find_appended_posit(posit_u, atoms, &adj_placed, Some(bond_len), v_element, geom)
                    .unwrap_or_else(|| {
                        no_context_position(posit_u, bond_len, geom, n_placed_before_v)
                    });

            atoms[v].posit = p;
            positioned[v] = true;
            queue.push_back(v);
        }
    }
}

/// Fallback when `find_appended_posit` returns `None` (degenerate/overcrowded cases).
/// Distributes directions evenly in the XY plane according to the local geometry,
/// varying by `n_placed` so multiple substituents don't stack on each other.
fn no_context_position(parent: Vec3, bond_len: f64, geom: BondGeom, n_placed: usize) -> Vec3 {
    let (n_spokes, base_angle): (usize, f64) = match geom {
        BondGeom::Linear => (2, 0.0),
        BondGeom::Planar => (3, 0.0),
        BondGeom::Tetrahedral => (4, PI / 6.0),
    };
    let angle = base_angle + 2.0 * PI * n_placed as f64 / n_spokes as f64;
    parent + Vec3::new(bond_len * angle.cos(), bond_len * angle.sin(), 0.)
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

fn geom_for_atom(i: usize, bonds: &[Bond]) -> BondGeom {
    let atom_bonds: Vec<&Bond> = bonds
        .iter()
        .filter(|b| b.atom_0 == i || b.atom_1 == i)
        .collect();

    if atom_bonds.iter().any(|b| b.bond_type == BondType::Triple) {
        return BondGeom::Linear;
    }

    // Cumulated diene (allene-type, e.g. C=C=C): the central atom carries two
    // double bonds and is sp-hybridised (linear), not sp2.
    let double_count = atom_bonds
        .iter()
        .filter(|b| b.bond_type == BondType::Double)
        .count();
    if double_count >= 2 {
        return BondGeom::Linear;
    }

    if atom_bonds
        .iter()
        .any(|b| matches!(b.bond_type, BondType::Double | BondType::Aromatic))
    {
        BondGeom::Planar
    } else {
        BondGeom::Tetrahedral
    }
}

fn bond_type_between(a: usize, b: usize, bonds: &[Bond]) -> BondType {
    bonds
        .iter()
        .find(|bond| {
            (bond.atom_0 == a && bond.atom_1 == b) || (bond.atom_0 == b && bond.atom_1 == a)
        })
        .map(|bond| bond.bond_type)
        .unwrap_or(BondType::Single)
}

fn estimate_bond_length(el0: Element, el1: Element, bt: BondType) -> f64 {
    let r0 = el0.covalent_radius().max(0.5);
    let r1 = el1.covalent_radius().max(0.5);
    let base = r0 + r1;
    match bt {
        BondType::Double => base * 0.87,
        BondType::Triple => base * 0.78,
        BondType::Aromatic => base * 0.91,
        _ => base,
    }
}

/// Average bond length of all edges in the ring (including the closing edge).
fn avg_ring_bond_len(ring: &[usize], atoms: &[Atom], bonds: &[Bond]) -> f64 {
    let n = ring.len();
    if n < 2 {
        return 1.5;
    }
    let total: f64 = (0..n)
        .map(|i| {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let bt = bond_type_between(a, b, bonds);
            estimate_bond_length(atoms[a].element, atoms[b].element, bt)
        })
        .sum();
    total / n as f64
}

/// Circumradius of a regular N-gon with side length `bond_len`.
fn ring_circumradius(bond_len: f64, n: usize) -> f64 {
    if n < 2 {
        return bond_len;
    }
    bond_len / (2.0 * (PI / n as f64).sin())
}

/// Smallest angular distance between two angles (radians).
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = ((a - b) % (2.0 * PI)).abs();
    if d > PI { 2.0 * PI - d } else { d }
}
