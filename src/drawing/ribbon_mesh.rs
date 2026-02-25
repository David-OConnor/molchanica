//! Cartoon ("ribbon") mesh generation for protein secondary structure.
//!
//! 1. Extract Cα and backbone-O positions per residue from each SecondaryStructure segment.
//! 2. Fit a Catmull-Rom spline through the Cα positions and sample it at
//!    `SPLINE_DIVS` points per residue pair.
//! 3. At each sample point, build a local Frenet-like frame whose orientation is
//!    anchored by the backbone carbonyl oxygen (the "guide" vector).  The guide
//!    is interpolated/smoothed between residue frames and then re-orthogonalised
//!    against the current spline tangent to keep it perpendicular.
//! 4. Extrude a cross-section profile appropriate to the secondary structure:
//!    - **Coil / loop** – circular tube
//!    - **α-helix**     – flat oval ribbon (wide face shows the helical spiral)
//!    - **β-strand**    – flat rectangular ribbon with an arrowhead at the C-terminus
//! 5. Stitch consecutive cross-sections into quads, compute per-vertex normals, and
//!    add end-caps.
//!
//! Coil/loop regions are identified as residues *not* covered by any `BackboneSS` entry
//! and rendered as a thin tube, extended by one residue into each adjacent SS segment
//! for a seamless join.

use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
};

use bio_files::{BackboneSS, SecondaryStructure};
use graphics::{Mesh, Vertex};
use lin_alg::f32::Vec3 as Vec3F32;

use crate::{
    drawing::color_viridis_float,
    molecules::{Atom, AtomRole},
};

// ── Dimensions (Å) ────────────────────────────────────────────────────────────

/// Half-width of helix ribbon along the guide direction (radially in/out from helix axis).
const HELIX_HALF_W: f32 = 1.0;
/// Half-thickness of helix ribbon along the binormal direction.
const HELIX_HALF_H: f32 = 0.2;

/// Half-width of β-strand ribbon.
const SHEET_HALF_W: f32 = 1.0;
/// Half-thickness of β-strand ribbon.
const SHEET_HALF_H: f32 = 0.15;
/// Half-width of the β-strand arrowhead at its widest point.
const SHEET_ARROW_HALF_W: f32 = 1.5;
/// Number of residues the arrowhead spans before the tip.
const SHEET_ARROW_RES: usize = 1;

/// Radius of the coil / loop tube.
const COIL_RADIUS: f32 = 0.25;

// ── Tessellation ──────────────────────────────────────────────────────────────

/// Spline sample subdivisions per residue pair (between consecutive Cα atoms).
const SPLINE_DIVS: usize = 8;

/// Number of vertices around one cross-section ring.
const N_PROFILE: usize = 8;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Per-residue data extracted from the atom list.
#[derive(Clone)]
struct ResidueFrame {
    /// Cα position.
    ca: Vec3F32,
    /// Backbone oxygen position (used for ribbon orientation).
    /// None when the atom is missing from the structure.
    o: Option<Vec3F32>,
    /// The residue's index in the full atom list, used for Viridis coloring.
    res_idx: usize,
}

// ── Catmull-Rom spline ────────────────────────────────────────────────────────

/// Evaluate a Catmull-Rom spline at `t ∈ [0, 1]` between `p1` and `p2`.
/// Returns 2× the true value; caller must multiply by 0.5.
fn catmull_rom(p0: Vec3F32, p1: Vec3F32, p2: Vec3F32, p3: Vec3F32, t: f32) -> Vec3F32 {
    let t2 = t * t;
    let t3 = t2 * t;
    p1 * 2.0
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (p0 * (-1.0) + p1 * 3.0 - p2 * 3.0 + p3) * t3
}

/// Derivative of the Catmull-Rom spline at `t` (unnormalised tangent direction).
fn catmull_rom_tangent(p0: Vec3F32, p1: Vec3F32, p2: Vec3F32, p3: Vec3F32, t: f32) -> Vec3F32 {
    let t2 = t * t;
    (p2 - p0)
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (2.0 * t)
        + (p0 * (-1.0) + p1 * 3.0 - p2 * 3.0 + p3) * (3.0 * t2)
}

// ── Guide-vector helpers ──────────────────────────────────────────────────────

/// Project the Cα→O vector perpendicular to `tangent` to get the ribbon guide.
fn compute_guide(ca: Vec3F32, o: Vec3F32, tangent: Vec3F32) -> Vec3F32 {
    let raw = (o - ca).to_normalized();
    let proj = tangent * raw.dot(tangent);
    let perp = raw - proj;
    if perp.magnitude() < 1e-6 {
        fallback_perp(tangent)
    } else {
        perp.to_normalized()
    }
}

/// Smooth a sequence of unit guide vectors with `passes` of weighted 3-point averaging.
fn smooth_guides(guides: &[Vec3F32], passes: usize) -> Vec<Vec3F32> {
    let n = guides.len();
    let mut g = guides.to_vec();
    for _ in 0..passes {
        let prev = g.clone();
        for i in 0..n {
            let a = if i > 0 { prev[i - 1] } else { prev[i] };
            let c = if i < n - 1 { prev[i + 1] } else { prev[i] };
            let sum = a + prev[i] * 2.0 + c;
            let len = sum.magnitude();
            g[i] = if len < 1e-8 {
                prev[i]
            } else {
                sum * (1.0 / len)
            };
        }
    }
    g
}

/// Smooth a sequence of positions with `passes` of weighted 3-point averaging.
/// Reduces the per-residue Cα zigzag in beta strands.
fn smooth_positions(positions: &[Vec3F32], passes: usize) -> Vec<Vec3F32> {
    let n = positions.len();
    let mut p = positions.to_vec();
    for _ in 0..passes {
        let prev = p.clone();
        // Skip i=0 and i=n-1 so the endpoints stay pinned at their original positions.
        for i in 1..n - 1 {
            p[i] = (prev[i - 1] + prev[i] * 2.0 + prev[i + 1]) * 0.25;
        }
    }
    p
}

/// Return an arbitrary unit vector perpendicular to `v`.
fn fallback_perp(v: Vec3F32) -> Vec3F32 {
    let helper = if v.z.abs() < 0.9 {
        Vec3F32::new(0.0, 0.0, 1.0)
    } else {
        Vec3F32::new(0.0, 1.0, 0.0)
    };
    v.cross(helper).to_normalized()
}

// ── Cross-section profile ─────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Profile {
    hw: f32, // half-width along binormal
    hh: f32, // half-height along guide
}

impl Profile {
    fn coil() -> Self {
        Self {
            hw: COIL_RADIUS,
            hh: COIL_RADIUS,
        }
    }
    fn helix() -> Self {
        // Wide in guide direction (radially outward), thin in binormal.
        Self {
            hw: HELIX_HALF_H,
            hh: HELIX_HALF_W,
        }
    }
    fn sheet(half_w: f32) -> Self {
        // Wide along guide (across-strand, in the sheet plane), thin along binormal (sheet normal).
        Self {
            hw: SHEET_HALF_H,
            hh: half_w,
        }
    }
}

/// Emit `N_PROFILE` (position, outward-normal) pairs for one cross-section ellipse.
fn cross_section(
    center: Vec3F32,
    guide: Vec3F32,
    binormal: Vec3F32,
    p: Profile,
) -> [(Vec3F32, Vec3F32); N_PROFILE] {
    let mut out = [(Vec3F32::new(0., 0., 0.), Vec3F32::new(0., 0., 0.)); N_PROFILE];
    for k in 0..N_PROFILE {
        let theta = TAU * (k as f32) / (N_PROFILE as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let pos = center + binormal * (p.hw * cos_t) + guide * (p.hh * sin_t);
        // Exact normal via gradient of the ellipse equation.
        let nx = cos_t / p.hw;
        let ny = sin_t / p.hh;
        let len = (nx * nx + ny * ny).sqrt().max(1e-8);
        let normal = (binormal * (nx / len) + guide * (ny / len)).to_normalized();
        out[k] = (pos, normal);
    }
    out
}

// ── Mesh assembly helpers ─────────────────────────────────────────────────────

fn append_ring_and_stitch(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<usize>,
    ring_a_base: usize,
    cross: &[(Vec3F32, Vec3F32); N_PROFILE],
    color: Option<(u8, u8, u8, u8)>,
) -> usize {
    let ring_b_base = verts.len();
    for &(pos, norm) in cross.iter() {
        let mut v = Vertex::new(pos.to_arr(), norm);
        v.color = color;
        verts.push(v);
    }
    for k in 0..N_PROFILE {
        let k1 = (k + 1) % N_PROFILE;
        let a0 = ring_a_base + k;
        let a1 = ring_a_base + k1;
        let b0 = ring_b_base + k;
        let b1 = ring_b_base + k1;
        indices.push(a0);
        indices.push(b1);
        indices.push(b0);
        indices.push(a0);
        indices.push(a1);
        indices.push(b1);
    }
    ring_b_base
}

fn emit_cap(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<usize>,
    ring_base: usize,
    center: Vec3F32,
    cap_normal: Vec3F32,
    color: Option<(u8, u8, u8, u8)>,
    inward: bool,
) {
    let center_idx = verts.len();
    let mut cv = Vertex::new(center.to_arr(), cap_normal);
    cv.color = color;
    verts.push(cv);
    for k in 0..N_PROFILE {
        let k1 = (k + 1) % N_PROFILE;
        let (i0, i1) = if inward {
            (ring_base + k1, ring_base + k)
        } else {
            (ring_base + k, ring_base + k1)
        };
        indices.push(center_idx);
        indices.push(i0);
        indices.push(i1);
    }
}

// ── Main segment builder ──────────────────────────────────────────────────────

fn build_segment_mesh(
    frames: &[ResidueFrame],
    ss: SecondaryStructure,
    max_residue: usize,
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<usize>,
) {
    if frames.len() < 2 {
        return;
    }
    let n = frames.len();

    // ── 1. Per-residue tangents (central difference over Cα) ─────────────────
    let mut tangents: Vec<Vec3F32> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = if i == 0 {
            frames[0].ca
        } else {
            frames[i - 1].ca
        };
        let next = if i == n - 1 {
            frames[n - 1].ca
        } else {
            frames[i + 1].ca
        };
        let t = (next - prev).to_normalized();
        tangents.push(if t.magnitude() < 1e-8 {
            Vec3F32::new(0., 1., 0.)
        } else {
            t
        });
    }

    // ── 2. Guide vectors ──────────────────────────────────────────────────────
    let mut guides_raw: Vec<Vec3F32> = Vec::with_capacity(n);
    for i in 0..n {
        let g = match frames[i].o {
            Some(o) => compute_guide(frames[i].ca, o, tangents[i]),
            None => fallback_perp(tangents[i]),
        };
        guides_raw.push(g);
    }

    // Sign-normalise: flip each guide that disagrees with the previous.
    let mut guides: Vec<Vec3F32> = Vec::with_capacity(n);
    guides.push(guides_raw[0]);
    for i in 1..n {
        let prev = guides[i - 1];
        let g = if guides_raw[i].dot(prev) < 0.0 {
            guides_raw[i] * (-1.0)
        } else {
            guides_raw[i]
        };
        guides.push(g);
    }

    // Stabilise per SS type.
    let guides: Vec<Vec3F32> = match ss {
        SecondaryStructure::Sheet => {
            // Use the segment-wide average direction: eliminates all per-residue twist.
            let sum = guides
                .iter()
                .fold(Vec3F32::new(0., 0., 0.), |acc, &g| acc + g);
            vec![sum.to_normalized(); n]
        }
        SecondaryStructure::Helix => smooth_guides(&guides, 3),
        SecondaryStructure::Coil => smooth_guides(&guides, 1),
    };

    // ── 3. Smoothed Cα positions (for sheets, reduces the backbone zigzag) ───
    let spline_ca: Vec<Vec3F32> = if ss == SecondaryStructure::Sheet {
        smooth_positions(&frames.iter().map(|f| f.ca).collect::<Vec<_>>(), 3)
    } else {
        frames.iter().map(|f| f.ca).collect()
    };

    let ca_ext = |i: isize| -> Vec3F32 {
        if i < 0 {
            spline_ca[0] + (spline_ca[0] - spline_ca[1])
        } else if i as usize >= n {
            spline_ca[n - 1] + (spline_ca[n - 1] - spline_ca[n - 2])
        } else {
            spline_ca[i as usize]
        }
    };

    // ── 4. Arrow parameters ───────────────────────────────────────────────────
    let arrow_start_res = if ss == SecondaryStructure::Sheet && n > SHEET_ARROW_RES {
        n - 1 - SHEET_ARROW_RES
    } else {
        n // disabled
    };

    // ── 5. Color helper ───────────────────────────────────────────────────────
    let res_color = |res_f: f32| -> Option<(u8, u8, u8, u8)> {
        let (cr, cg, cb) = color_viridis_float(res_f, 0.0, max_residue as f32);
        Some((
            (cr * 255.0) as u8,
            (cg * 255.0) as u8,
            (cb * 255.0) as u8,
            255u8,
        ))
    };

    // ── 6. Sample spline and emit geometry ───────────────────────────────────
    let total_samples = (n - 1) * SPLINE_DIVS + 1;
    let mut ring_base: Option<usize> = None;
    let mut last_pos = spline_ca[0];

    for s in 0..total_samples {
        let seg_idx = ((s / SPLINE_DIVS) as isize).min((n - 2) as isize);
        let local_t = (s % SPLINE_DIVS) as f32 / SPLINE_DIVS as f32;
        let (seg_idx, t) = if s == total_samples - 1 {
            ((n - 2) as isize, 1.0_f32)
        } else {
            (seg_idx, local_t)
        };
        let i = seg_idx as usize;

        let p0 = ca_ext(seg_idx - 1);
        let p1 = ca_ext(seg_idx);
        let p2 = ca_ext(seg_idx + 1);
        let p3 = ca_ext(seg_idx + 2);

        let pos = catmull_rom(p0, p1, p2, p3, t) * 0.5;
        let tan_raw = catmull_rom_tangent(p0, p1, p2, p3, t);
        let tan = if tan_raw.magnitude() < 1e-8 {
            Vec3F32::new(0., 1., 0.)
        } else {
            tan_raw.to_normalized()
        };

        let g0 = guides[i];
        let g1 = guides[(i + 1).min(n - 1)];
        let g_lerp = g0 + (g1 - g0) * t;
        let g_proj = tan * g_lerp.dot(tan);
        let g_perp = g_lerp - g_proj;
        let guide = if g_perp.magnitude() < 1e-8 {
            fallback_perp(tan)
        } else {
            g_perp.to_normalized()
        };
        let binormal = tan.cross(guide).to_normalized();

        let res_f = {
            let r0 = frames[i].res_idx as f32;
            let r1 = frames[(i + 1).min(n - 1)].res_idx as f32;
            r0 + (r1 - r0) * t
        };
        let color = res_color(res_f);

        let profile = match ss {
            SecondaryStructure::Coil => Profile::coil(),
            SecondaryStructure::Helix => Profile::helix(),
            SecondaryStructure::Sheet => {
                let global_res = i as f32 + t;
                if global_res >= arrow_start_res as f32 {
                    let arrow_t = ((global_res - arrow_start_res as f32) / SHEET_ARROW_RES as f32)
                        .clamp(0.0, 1.0);
                    Profile::sheet((SHEET_ARROW_HALF_W * (1.0 - arrow_t)).max(0.01))
                } else {
                    Profile::sheet(SHEET_HALF_W)
                }
            }
        };

        let ring = cross_section(pos, guide, binormal, profile);

        match ring_base {
            None => {
                let base = verts.len();
                for &(p, nm) in ring.iter() {
                    let mut v = Vertex::new(p.to_arr(), nm);
                    v.color = color;
                    verts.push(v);
                }
                emit_cap(verts, indices, base, pos, tan * (-1.0), color, true);
                ring_base = Some(base);
            }
            Some(prev_base) => {
                let new_base = append_ring_and_stitch(verts, indices, prev_base, &ring, color);
                ring_base = Some(new_base);
            }
        }

        last_pos = pos;
    }

    if let Some(last_base) = ring_base {
        let end_tan = (spline_ca[n - 1] - spline_ca[n - 2]).to_normalized();
        let end_color = res_color(frames[n - 1].res_idx as f32);
        emit_cap(
            verts, indices, last_base, last_pos, end_tan, end_color, false,
        );
    }
}

// ── Data extraction ───────────────────────────────────────────────────────────

fn extract_frames(seg: &BackboneSS, atoms: &[Atom]) -> Vec<ResidueFrame> {
    let mut ca_map: HashMap<usize, Vec3F32> = HashMap::new();
    let mut o_map: HashMap<usize, Vec3F32> = HashMap::new();

    for atom in atoms {
        if atom.serial_number < seg.start_sn || atom.serial_number > seg.end_sn {
            continue;
        }
        let res_idx = match atom.residue {
            Some(r) => r,
            None => continue,
        };
        match atom.role {
            Some(AtomRole::C_Alpha) => {
                ca_map.insert(res_idx, vec3(atom));
            }
            Some(AtomRole::O_Backbone) => {
                o_map.insert(res_idx, vec3(atom));
            }
            _ => {}
        }
    }

    if ca_map.is_empty() {
        return Vec::new();
    }

    let mut res_indices: Vec<usize> = ca_map.keys().copied().collect();
    res_indices.sort_unstable();
    res_indices
        .into_iter()
        .map(|ri| ResidueFrame {
            ca: ca_map[&ri],
            o: o_map.get(&ri).copied(),
            res_idx: ri,
        })
        .collect()
}

/// Convert an atom's f64 position to a f32 Vec3.
#[inline]
fn vec3(atom: &Atom) -> Vec3F32 {
    Vec3F32::new(
        atom.posit.x as f32,
        atom.posit.y as f32,
        atom.posit.z as f32,
    )
}

// ── Coil-region helpers ───────────────────────────────────────────────────────

/// Group a sorted list of residue indices into consecutive runs.
fn group_consecutive(sorted: &[usize]) -> Vec<Vec<usize>> {
    if sorted.is_empty() {
        return Vec::new();
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![sorted[0]];
    for &r in &sorted[1..] {
        if r == *current.last().unwrap() + 1 {
            current.push(r);
        } else {
            groups.push(std::mem::replace(&mut current, vec![r]));
        }
    }
    groups.push(current);
    groups
}

/// Extend a coil run by one residue into each adjacent SS segment so the tube
/// meets the ribbon/helix without a gap.
fn extend_run(
    run: &[usize],
    covered: &HashSet<usize>,
    ca_map: &HashMap<usize, Vec3F32>,
) -> Vec<usize> {
    let mut ext = Vec::with_capacity(run.len() + 2);

    if let Some(&first) = run.first() {
        if first >= 1 {
            let p1 = first - 1;
            if covered.contains(&p1) && ca_map.contains_key(&p1) {
                ext.push(p1);
            }
        }
    }

    ext.extend_from_slice(run);

    if let Some(&last) = run.last() {
        let n1 = last + 1;
        if covered.contains(&n1) && ca_map.contains_key(&n1) {
            ext.push(n1);
        }
    }

    ext
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn build_cartoon_mesh(backbone: &[BackboneSS], atoms: &[Atom]) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let max_residue = atoms.iter().filter_map(|a| a.residue).max().unwrap_or(1);

    // ── 1. Collect all Cα positions ───────────────────────────────────────────
    let mut all_ca: HashMap<usize, Vec3F32> = HashMap::new();
    for atom in atoms {
        if atom.role == Some(AtomRole::C_Alpha) {
            if let Some(r) = atom.residue {
                all_ca.entry(r).or_insert_with(|| vec3(atom));
            }
        }
    }

    // ── 2. Render helix / sheet segments; track covered residues ─────────────
    let mut covered: HashSet<usize> = HashSet::new();

    for seg in backbone {
        let frames = extract_frames(seg, atoms);
        if frames.len() < 2 {
            continue;
        }
        for f in &frames {
            covered.insert(f.res_idx);
        }
        build_segment_mesh(
            &frames,
            seg.sec_struct,
            max_residue,
            &mut vertices,
            &mut indices,
        );
    }

    // ── 3. Render coil / loop regions ─────────────────────────────────────────
    // Any Cα residue not covered by an SS segment is a coil.
    let mut coil_residues: Vec<usize> = all_ca
        .keys()
        .copied()
        .filter(|r| !covered.contains(r))
        .collect();
    coil_residues.sort_unstable();

    for run in group_consecutive(&coil_residues) {
        // Extend by 1 into adjacent SS segments for a seamless join.
        let ext = extend_run(&run, &covered, &all_ca);
        if ext.len() < 2 {
            continue;
        }

        let frames: Vec<ResidueFrame> = ext
            .iter()
            .filter_map(|&r| {
                all_ca.get(&r).map(|&ca| ResidueFrame {
                    ca,
                    o: None,
                    res_idx: r,
                })
            })
            .collect();

        if frames.len() < 2 {
            continue;
        }

        build_segment_mesh(
            &frames,
            SecondaryStructure::Coil,
            max_residue,
            &mut vertices,
            &mut indices,
        );
    }

    Mesh {
        vertices,
        indices,
        material: 0,
    }
}
