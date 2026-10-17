//! Sequence correspondence and rigid Cα superposition, independent of small-molecule alignment.

use std::{
    collections::{BTreeMap, HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use bio_files::ResidueType;
use lin_alg::f64::Vec3;
use mol_defs::molecules::{AtomRole, MolGenericRef, MolType, peptide::MoleculePeptide};
use na_seq::{AaIdent, AtomTypeInRes};

use crate::{
    drawing::{MoleculeView, representations::Representation},
    mol_manip::{ManipMode, PeptideMeshTransform},
    selection::{MoleculeAtoms, Selection, expression},
    state::{OperatingMode, State},
    util::RedrawFlags,
};

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub enum Correspondence {
    #[default]
    Sequence,
    ResidueNumber,
    Order,
}

impl Correspondence {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sequence => "Sequence alignment",
            Self::ResidueNumber => "Residue numbers",
            Self::Order => "Selected residue order",
        }
    }
}

#[derive(Clone)]
pub struct AlphaCarbon {
    pub atom: usize,
    pub residue: usize,
    pub number: u32,
    pub aa: String,
    pub position: Vec3,
}

#[derive(Clone)]
pub struct Source {
    pub key: String,
    pub label: String,
    pub chain: String,
    pub positions: Vec<Vec3>,
    topology: u64,
    pub ca: Vec<AlphaCarbon>,
}

fn topology(mol: &MoleculePeptide) -> u64 {
    let mut hash = DefaultHasher::new();
    for atom in &mol.common.atoms {
        atom.serial_number.hash(&mut hash);
        atom.residue.hash(&mut hash);
        atom.chain.hash(&mut hash);
        atom.role.map(|r| r as u8).hash(&mut hash);
        atom.type_in_res
            .as_ref()
            .map(std::mem::discriminant)
            .hash(&mut hash);
    }
    for residue in &mol.residues {
        residue.serial_number.hash(&mut hash);
        residue.res_type.to_string().hash(&mut hash);
    }
    for chain in &mol.chains {
        chain.id.hash(&mut hash);
    }
    hash.finish()
}

pub fn resolve(state: &State, key: &str) -> Result<usize, String> {
    let mut matches = state
        .peptides
        .iter()
        .enumerate()
        .filter(|(_, mol)| expression::object_key(&MolGenericRef::Peptide(mol)) == key);
    let Some((i, _)) = matches.next() else {
        return Err("A comparison protein is no longer open".into());
    };
    if matches.next().is_some() {
        return Err("Comparison proteins must have distinct source paths or identifiers".into());
    }
    Ok(i)
}

fn is_ca(atom: &mol_defs::molecules::Atom) -> bool {
    atom.role == Some(AtomRole::C_Alpha) || atom.type_in_res == Some(AtomTypeInRes::CA)
}

pub fn chains(mol: &MoleculePeptide) -> Vec<(Option<usize>, String)> {
    let mut result = BTreeMap::new();
    for atom in &mol.common.atoms {
        if is_ca(atom) {
            result.entry(atom.chain).or_insert_with(|| {
                atom.chain
                    .and_then(|c| mol.chains.get(c))
                    .map(|c| c.id.clone())
                    .unwrap_or_else(|| "Unassigned".into())
            });
        }
    }
    result.into_iter().collect()
}

/// One chain and one optional expression define the fitting region. All coordinates of the
/// mobile object are transformed later, including hetero residues attached to that object.
pub fn source(
    state: &State,
    key: &str,
    chain: Option<usize>,
    selection: &str,
) -> Result<Source, String> {
    let i = resolve(state, key)?;
    let mol = &state.peptides[i];
    if mol.common.atom_posits.len() != mol.common.atoms.len() {
        return Err("Protein atom coordinates are incomplete".into());
    }
    if mol
        .common
        .atom_posits
        .iter()
        .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
    {
        return Err("Protein contains nonfinite coordinates".into());
    }
    let selected = expression::evaluate(
        state,
        if selection.trim().is_empty() {
            "protein"
        } else {
            selection
        },
    )?;
    let atoms = selected
        .iter()
        .find(|s| s.mol_type == MolType::Peptide && s.mol_i == i)
        .map(|s| s.atoms.as_slice())
        .unwrap_or(&[]);
    let mut by_residue: BTreeMap<usize, usize> = BTreeMap::new();
    for &a in atoms {
        let atom = &mol.common.atoms[a];
        if !is_ca(atom) || atom.chain != chain {
            continue;
        }
        let Some(r) = atom.residue else {
            continue;
        };
        if !mol
            .residues
            .get(r)
            .is_some_and(|r| matches!(r.res_type, ResidueType::AminoAcid(_)))
        {
            continue;
        }
        // Use one alternate per residue: unlabelled/A first, then highest occupancy.
        let rank = |a: usize| {
            let a = &mol.common.atoms[a];
            let alt = a.alt_conformation_id.as_deref().unwrap_or("");
            (
                if alt.is_empty() {
                    2
                } else if alt == "A" {
                    1
                } else {
                    0
                },
                a.occupancy.unwrap_or(0.),
            )
        };
        if by_residue.get(&r).is_none_or(|&old| rank(a) > rank(old)) {
            by_residue.insert(r, a);
        }
    }
    let ca: Vec<_> = by_residue
        .into_iter()
        .map(|(r, a)| {
            let residue = &mol.residues[r];
            let ResidueType::AminoAcid(aa) = residue.res_type else {
                unreachable!()
            };
            AlphaCarbon {
                atom: a,
                residue: r,
                number: residue.serial_number,
                aa: aa.to_str(AaIdent::OneLetter).into(),
                position: mol.common.atom_posits[a],
            }
        })
        .collect();
    if ca.len() < 3 {
        return Err("Select at least three Cα residues in each chain".into());
    }
    Ok(Source {
        key: key.into(),
        label: MolGenericRef::Peptide(mol).name().into_owned(),
        chain: chain
            .and_then(|c| mol.chains.get(c))
            .map(|c| c.id.clone())
            .unwrap_or_else(|| "Unassigned".into()),
        positions: mol.common.atom_posits.clone(),
        topology: topology(mol),
        ca,
    })
}

pub fn source_expression(state: &State, selection: &str) -> Result<Source, String> {
    let mut target = None;
    for selected in expression::evaluate(state, selection)? {
        if selected.mol_type != MolType::Peptide {
            continue;
        }
        for a in selected.atoms {
            let atom = &state.peptides[selected.mol_i].common.atoms[a];
            if !is_ca(atom) {
                continue;
            }
            let candidate = (selected.mol_i, atom.chain);
            if target.is_some_and(|old| old != candidate) {
                return Err(
                    "Each alignment expression must select Cα atoms from one protein chain".into(),
                );
            }
            target = Some(candidate);
        }
    }
    let Some((i, chain)) = target else {
        return Err("No protein Cα atoms match the expression".into());
    };
    let key = expression::object_key(&MolGenericRef::Peptide(&state.peptides[i]));
    source(state, &key, chain, selection)
}

/// Global sequence alignment with affine gaps. Scores: identity +3, substitution -1,
/// gap opening -5, extension -1. Traceback is bounded to keep long-chain requests predictable.
fn sequence_pairs(
    reference: &[AlphaCarbon],
    mobile: &[AlphaCarbon],
) -> Result<Vec<(usize, usize)>, String> {
    let (n, m) = (reference.len(), mobile.len());
    let cells = (n + 1)
        .checked_mul(m + 1)
        .filter(|&n| n <= 4_000_000)
        .ok_or("Sequence alignment exceeds four million cells; restrict the fitting regions")?;
    let mut trace = vec![[0u8; 3]; cells];
    let neg = i32::MIN / 4;
    let mut previous = vec![[neg; 3]; m + 1];
    previous[0][0] = 0;
    for j in 1..=m {
        previous[j][2] = -5 - (j as i32 - 1);
        trace[j][2] = if j == 1 { 0 } else { 2 };
    }
    let best = |scores: [i32; 3]| -> (i32, u8) {
        let mut index = 0;
        for i in 1..3 {
            if scores[i] > scores[index] {
                index = i;
            }
        }
        (scores[index], index as u8)
    };
    for i in 1..=n {
        let mut current = vec![[neg; 3]; m + 1];
        current[0][1] = -5 - (i as i32 - 1);
        trace[i * (m + 1)][1] = if i == 1 { 0 } else { 1 };
        for j in 1..=m {
            let at = i * (m + 1) + j;
            let (score, from) = best(previous[j - 1]);
            current[j][0] = score
                + if reference[i - 1].aa == mobile[j - 1].aa {
                    3
                } else {
                    -1
                };
            trace[at][0] = from;
            let (score, from) = best([previous[j][0] - 5, previous[j][1] - 1, previous[j][2] - 5]);
            current[j][1] = score;
            trace[at][1] = from;
            let (score, from) = best([
                current[j - 1][0] - 5,
                current[j - 1][1] - 5,
                current[j - 1][2] - 1,
            ]);
            current[j][2] = score;
            trace[at][2] = from;
        }
        previous = current;
    }
    let (mut i, mut j, mut state) = (n, m, best(previous[m]).1);
    let mut pairs = Vec::new();
    while i > 0 || j > 0 {
        let next = trace[i * (m + 1) + j][state as usize];
        match state {
            0 => {
                pairs.push((i - 1, j - 1));
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
        state = next;
    }
    pairs.reverse();
    Ok(pairs)
}

fn correspondence(
    reference: &[AlphaCarbon],
    mobile: &[AlphaCarbon],
    method: Correspondence,
) -> Result<Vec<(usize, usize)>, String> {
    match method {
        Correspondence::Sequence => sequence_pairs(reference, mobile),
        Correspondence::Order => {
            if reference.len() != mobile.len() {
                return Err("Residue-order fitting requires equal Cα counts".into());
            }
            Ok((0..reference.len()).map(|i| (i, i)).collect())
        }
        Correspondence::ResidueNumber => {
            let mut lookup = HashMap::new();
            for (j, a) in mobile.iter().enumerate() {
                if lookup.insert(a.number, j).is_some() {
                    return Err(
                        "Duplicate residue numbers; use sequence alignment or residue order".into(),
                    );
                }
            }
            let mut seen = std::collections::HashSet::new();
            let mut pairs = Vec::new();
            for (i, a) in reference.iter().enumerate() {
                if !seen.insert(a.number) {
                    return Err(
                        "Duplicate residue numbers; use sequence alignment or residue order".into(),
                    );
                }
                if let Some(&j) = lookup.get(&a.number) {
                    pairs.push((i, j));
                }
            }
            Ok(pairs)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Transform {
    pub rotation: [[f64; 3]; 3],
    pub translation: Vec3,
}

impl Transform {
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let r = self.rotation;
        Vec3::new(
            r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z,
            r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z,
            r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z,
        ) + self.translation
    }
}

/// Horn's quaternion solution, using Jacobi diagonalization of the symmetric 4×4 matrix.
/// A unit quaternion always produces a proper rotation, never a reflection or a scale change.
pub fn fit(reference: &[Vec3], mobile: &[Vec3]) -> Result<Transform, String> {
    if reference.len() != mobile.len() || reference.len() < 3 {
        return Err("A fit needs at least three paired Cα atoms".into());
    }
    let center = |points: &[Vec3]| {
        points.iter().copied().fold(Vec3::new_zero(), |a, b| a + b) / points.len() as f64
    };
    let (cr, cm) = (center(reference), center(mobile));
    for points in [reference, mobile] {
        let origin = points[0];
        let axis = points
            .iter()
            .map(|&p| p - origin)
            .max_by(|a, b| a.magnitude_squared().total_cmp(&b.magnitude_squared()))
            .unwrap();
        let area = points
            .iter()
            .map(|&p| axis.cross(p - origin).magnitude_squared())
            .fold(0., f64::max);
        if !area.is_finite() || area <= 1e-12 * axis.magnitude_squared().powi(2).max(1.) {
            return Err(
                "The fitting atoms are coincident or collinear; choose a larger region".into(),
            );
        }
    }
    let mut s = [[0.; 3]; 3];
    for (&r, &m) in reference.iter().zip(mobile) {
        let r = r - cr;
        let m = m - cm;
        let (r, m) = ([r.x, r.y, r.z], [m.x, m.y, m.z]);
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += m[i] * r[j];
            }
        }
    }
    let trace = s[0][0] + s[1][1] + s[2][2];
    let mut a = [
        [
            trace,
            s[1][2] - s[2][1],
            s[2][0] - s[0][2],
            s[0][1] - s[1][0],
        ],
        [
            s[1][2] - s[2][1],
            s[0][0] - s[1][1] - s[2][2],
            s[0][1] + s[1][0],
            s[2][0] + s[0][2],
        ],
        [
            s[2][0] - s[0][2],
            s[0][1] + s[1][0],
            -s[0][0] + s[1][1] - s[2][2],
            s[1][2] + s[2][1],
        ],
        [
            s[0][1] - s[1][0],
            s[2][0] + s[0][2],
            s[1][2] + s[2][1],
            -s[0][0] - s[1][1] + s[2][2],
        ],
    ];
    let mut v = [[0.; 4]; 4];
    for i in 0..4 {
        v[i][i] = 1.;
    }
    for _ in 0..80 {
        let (mut p, mut q) = (0, 1);
        for i in 0..4 {
            for j in i + 1..4 {
                if a[i][j].abs() > a[p][q].abs() {
                    p = i;
                    q = j;
                }
            }
        }
        if a[p][q].abs() < 1e-14 * trace.abs().max(1.) {
            break;
        }
        let angle = 0.5 * (2. * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (sn, cs) = angle.sin_cos();
        let (pp, qq, pq) = (a[p][p], a[q][q], a[p][q]);
        for k in 0..4 {
            if k != p && k != q {
                let (kp, kq) = (a[k][p], a[k][q]);
                a[k][p] = cs * kp - sn * kq;
                a[p][k] = a[k][p];
                a[k][q] = sn * kp + cs * kq;
                a[q][k] = a[k][q];
            }
            let (kp, kq) = (v[k][p], v[k][q]);
            v[k][p] = cs * kp - sn * kq;
            v[k][q] = sn * kp + cs * kq;
        }
        a[p][p] = cs * cs * pp - 2. * sn * cs * pq + sn * sn * qq;
        a[q][q] = sn * sn * pp + 2. * sn * cs * pq + cs * cs * qq;
        a[p][q] = 0.;
        a[q][p] = 0.;
    }
    let best = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap();
    let [w, x, y, z] = [v[0][best], v[1][best], v[2][best], v[3][best]];
    let mut transform = Transform {
        rotation: [
            [
                1. - 2. * (y * y + z * z),
                2. * (x * y - z * w),
                2. * (x * z + y * w),
            ],
            [
                2. * (x * y + z * w),
                1. - 2. * (x * x + z * z),
                2. * (y * z - x * w),
            ],
            [
                2. * (x * z - y * w),
                2. * (y * z + x * w),
                1. - 2. * (x * x + y * y),
            ],
        ],
        translation: Vec3::new_zero(),
    };
    transform.translation = cr - transform.apply(cm);
    Ok(transform)
}

pub struct ResiduePair {
    pub reference: usize,
    pub mobile: usize,
    pub distance: f64,
    pub retained: bool,
}

pub struct Comparison {
    pub reference: Source,
    pub mobile: Source,
    pub transformed: Vec<Vec3>,
    pub transform: Transform,
    pub pairs: Vec<ResiduePair>,
    pub before_rmsd: f64,
    pub all_rmsd: f64,
    pub fit_rmsd: f64,
    pub method: Correspondence,
    pub applied: bool,
}

pub fn compare(
    reference: Source,
    mobile: Source,
    method: Correspondence,
    cutoff: Option<f64>,
) -> Result<Comparison, String> {
    if reference.key == mobile.key {
        return Err("Choose different reference and mobile objects".into());
    }
    if cutoff.is_some_and(|c| !c.is_finite() || c <= 0.) {
        return Err("Outlier cutoff must be positive and finite".into());
    }
    let indices = correspondence(&reference.ca, &mobile.ca, method)?;
    if indices.len() < 3 {
        return Err("Fewer than three corresponding Cα residues were found".into());
    }
    let mut retained = vec![true; indices.len()];
    let solve = |retained: &[bool]| {
        let r: Vec<_> = indices
            .iter()
            .zip(retained)
            .filter(|(_, keep)| **keep)
            .map(|(&(r, _), _)| reference.ca[r].position)
            .collect();
        let m: Vec<_> = indices
            .iter()
            .zip(retained)
            .filter(|(_, keep)| **keep)
            .map(|(&(_, m), _)| mobile.ca[m].position)
            .collect();
        fit(&r, &m)
    };
    let mut transform = solve(&retained)?;
    if let Some(cutoff) = cutoff {
        for _ in 0..5 {
            let next: Vec<_> = indices
                .iter()
                .zip(&retained)
                .map(|(&(r, m), &keep)| {
                    keep && (reference.ca[r].position - transform.apply(mobile.ca[m].position))
                        .magnitude()
                        <= cutoff
                })
                .collect();
            if next == retained {
                break;
            }
            if next.iter().filter(|&&v| v).count() < 3 {
                return Err("Outlier rejection leaves fewer than three pairs; increase the cutoff or disable rejection".into());
            }
            transform = solve(&next)?;
            retained = next;
        }
    }
    let before_rmsd = (indices
        .iter()
        .map(|&(r, m)| (reference.ca[r].position - mobile.ca[m].position).magnitude_squared())
        .sum::<f64>()
        / indices.len() as f64)
        .sqrt();
    let pairs: Vec<_> = indices
        .into_iter()
        .zip(retained)
        .map(|((r, m), retained)| ResiduePair {
            reference: r,
            mobile: m,
            retained,
            distance: (reference.ca[r].position - transform.apply(mobile.ca[m].position))
                .magnitude(),
        })
        .collect();
    let all_rmsd =
        (pairs.iter().map(|p| p.distance * p.distance).sum::<f64>() / pairs.len() as f64).sqrt();
    let kept: Vec<_> = pairs.iter().filter(|p| p.retained).collect();
    let fit_rmsd =
        (kept.iter().map(|p| p.distance * p.distance).sum::<f64>() / kept.len() as f64).sqrt();
    let transformed = mobile
        .positions
        .iter()
        .map(|&p| transform.apply(p))
        .collect();
    Ok(Comparison {
        reference,
        mobile,
        transformed,
        transform,
        pairs,
        before_rmsd,
        all_rmsd,
        fit_rmsd,
        method,
        applied: false,
    })
}

impl Comparison {
    pub fn validate(&self, state: &State) -> Result<(usize, usize), String> {
        let (r, m) = (
            resolve(state, &self.reference.key)?,
            resolve(state, &self.mobile.key)?,
        );
        let mobile_positions = if self.applied {
            &self.transformed
        } else {
            &self.mobile.positions
        };
        if topology(&state.peptides[r]) != self.reference.topology
            || topology(&state.peptides[m]) != self.mobile.topology
            || state.peptides[r].common.atom_posits != self.reference.positions
            || state.peptides[m].common.atom_posits != *mobile_positions
        {
            return Err("Structures changed since this comparison; run it again".into());
        }
        Ok((r, m))
    }

    pub fn apply(
        &mut self,
        state: &mut State,
        redraw: &mut RedrawFlags,
        undo: bool,
    ) -> Result<(), String> {
        if state.volatile.operating_mode != OperatingMode::Primary
            || state.volatile.md_local.running
            || state.volatile.md_local.draw_md_mols
        {
            return Err(
                "Return to the main molecule view and pause MD before superposing proteins".into(),
            );
        }
        let (_, m) = self.validate(state)?;
        if undo != self.applied {
            return Err("This fit is already in the requested state".into());
        }
        state.peptides[m].common.atom_posits = if undo {
            self.mobile.positions.clone()
        } else {
            self.transformed.clone()
        };
        self.applied = !undo;
        state.volatile.mol_manip.mode = ManipMode::None;
        state.volatile.mol_manip.ribbon_mesh_transform = PeptideMeshTransform::default();
        state.volatile.mol_manip.surface_mesh_transform = PeptideMeshTransform::default();
        state.volatile.flags.ss_mesh_created = false;
        state.volatile.flags.update_ss_mesh = true;
        state.volatile.flags.sas_mesh_created = false;
        state.volatile.thread_receivers.peptide_mesh_coloring = None;
        if matches!(
            state.ui.mol_view_peptide,
            MoleculeView::Surface | MoleculeView::Dots
        ) {
            state.volatile.flags.update_sas_mesh = true;
        }
        state.to_save.save_flag = true;
        redraw.set_all();
        Ok(())
    }

    pub fn selection(&self, state: &State, pair_indices: &[usize]) -> Result<Selection, String> {
        let (r, m) = self.validate(state)?;
        let mut selected = Vec::new();
        for (index, source, reference) in [(r, &self.reference, true), (m, &self.mobile, false)] {
            let mut atoms = Vec::new();
            for &i in pair_indices {
                let pair = &self.pairs[i];
                let ca = &source.ca[if reference {
                    pair.reference
                } else {
                    pair.mobile
                }];
                atoms.extend(
                    state.peptides[index]
                        .common
                        .atoms
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.residue == Some(ca.residue))
                        .map(|(i, _)| i),
                );
            }
            atoms.sort_unstable();
            atoms.dedup();
            selected.push(MoleculeAtoms {
                mol_type: MolType::Peptide,
                mol_i: index,
                atoms,
            });
        }
        Ok(Selection::Atoms(selected))
    }

    pub fn comparison_view(
        &self,
        state: &mut State,
        redraw: &mut RedrawFlags,
    ) -> Result<(), String> {
        let (r, m) = self.validate(state)?;
        for (i, label, color) in [
            (r, "Reference", [0.2, 0.65, 1.]),
            (m, "Mobile", [1., 0.55, 0.15]),
        ] {
            let mol = MolGenericRef::Peptide(&state.peptides[i]);
            let key = expression::object_key(&mol);
            let mut layer = Representation::new(
                format!("({}) and protein", expression::object_expression(&mol)),
                MoleculeView::Ribbon,
            );
            layer.name = format!("Comparison {label}: {key}");
            layer.color = Some(color);
            if let Some(existing) = state
                .to_save
                .representations
                .iter_mut()
                .find(|l| l.name == layer.name)
            {
                *existing = layer;
            } else {
                state.to_save.representations.push(layer);
            }
            if !state.to_save.hidden_default_objects.contains(&key) {
                state.to_save.hidden_default_objects.push(key);
            }
            state.peptides[i].common.visible = true;
        }
        state.volatile.flags.update_ss_mesh = true;
        state.to_save.save_flag = true;
        redraw.set_all();
        Ok(())
    }

    pub fn csv(&self) -> String {
        let quote = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
        let mut out = String::from(
            "reference_object,reference_chain,reference_residue,reference_aa,mobile_object,mobile_chain,mobile_residue,mobile_aa,distance_angstrom,retained_in_fit\n",
        );
        for pair in &self.pairs {
            let (r, m) = (
                &self.reference.ca[pair.reference],
                &self.mobile.ca[pair.mobile],
            );
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.6},{}\n",
                quote(&self.reference.key),
                quote(&self.reference.chain),
                r.number,
                r.aa,
                quote(&self.mobile.key),
                quote(&self.mobile.chain),
                m.number,
                m.aa,
                pair.distance,
                pair.retained
            ));
        }
        out
    }

    /// Export every mobile atom at its fitted coordinate. This deliberately writes a coordinate
    /// document, rather than reusing untransformed source CIF text or implying a transformed map.
    pub fn aligned_cif(&self, state: &State) -> Result<String, String> {
        use mol_defs::mmcif_edit::{CifCategory, CifDoc, CifRow};
        let (_, m) = self.validate(state)?;
        let mol = &state.peptides[m];
        let mut doc = CifDoc::new("data_molchanica_aligned\n#\n").map_err(|e| e.to_string())?;
        let mut atoms = CifCategory::new_loop(
            "_atom_site",
            [
                "group_PDB",
                "id",
                "type_symbol",
                "label_atom_id",
                "label_alt_id",
                "label_comp_id",
                "label_asym_id",
                "label_seq_id",
                "auth_asym_id",
                "auth_seq_id",
                "Cartn_x",
                "Cartn_y",
                "Cartn_z",
                "occupancy",
                "pdbx_PDB_model_num",
            ]
            .map(str::to_owned)
            .to_vec(),
        );
        for (i, (atom, position)) in mol.common.atoms.iter().zip(&self.transformed).enumerate() {
            let residue = atom.residue.and_then(|r| mol.residues.get(r));
            let res_name = residue
                .map(|r| match &r.res_type {
                    ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::ThreeLetters).to_owned(),
                    ResidueType::Water => "HOH".into(),
                    ResidueType::Other(name) => name.clone(),
                })
                .unwrap_or_else(|| "UNK".into());
            let chain = atom
                .chain
                .and_then(|c| mol.chains.get(c))
                .map(|c| c.id.clone())
                .unwrap_or_else(|| "A".into());
            let number = residue
                .map(|r| r.serial_number.to_string())
                .unwrap_or_else(|| "0".into());
            let name = atom
                .type_in_res_general
                .clone()
                .or_else(|| atom.type_in_res.as_ref().map(ToString::to_string))
                .unwrap_or_else(|| format!("{}{}", atom.element.to_letter(), i + 1));
            atoms.push_row(CifRow::new(vec![
                if atom.hetero { "HETATM" } else { "ATOM" }.into(),
                (i + 1).to_string(),
                atom.element.to_letter(),
                name,
                atom.alt_conformation_id
                    .clone()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| ".".into()),
                res_name,
                chain.clone(),
                number.clone(),
                chain,
                number,
                format!("{:.6}", position.x),
                format!("{:.6}", position.y),
                format!("{:.6}", position.z),
                atom.occupancy
                    .map(|o| o.to_string())
                    .unwrap_or_else(|| "1.0".into()),
                "1".into(),
            ]));
        }
        doc.insert_category(None, atoms);
        Ok(doc.to_text())
    }
}
