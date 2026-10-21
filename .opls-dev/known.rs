//! Complete ligand-specific OPLS parameterizations outside Foyer's SMARTS set.
//! Match exact elemental/bond-order graphs, including explicit hydrogens, then
//! map parameters into the caller's order. Coordinates and atom names are not
//! involved in recognition. Each data set documents its tautomer and source.

use std::collections::HashMap;

use bio_files::{AtomGeneric, BondGeneric, Sdf, gromacs::GromacsTopology};

use crate::{
    ParamError,
    import::{ExplicitParams, gromacs_molecule_params},
};

#[derive(Debug)]
pub(super) struct Ligand {
    atoms: Vec<AtomGeneric>,
    edges: Vec<Vec<u8>>,
    degree: Vec<usize>,
    types: Vec<String>,
    charges: Vec<f32>,
    params: ExplicitParams,
}

fn edges(atoms: &[AtomGeneric], bonds: &[BondGeneric]) -> Result<Vec<Vec<u8>>, ParamError> {
    let indices: HashMap<_, _> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial_number, i))
        .collect();
    let mut result = vec![vec![0; atoms.len()]; atoms.len()];
    for bond in bonds {
        let (&i, &j) = indices
            .get(&bond.atom_0_sn)
            .zip(indices.get(&bond.atom_1_sn))
            .ok_or_else(|| ParamError::new("OPLS ligand bond references an absent atom"))?;
        // Zero means no edge; preserve every BondType value distinctly.
        result[i][j] = if bond.bond_type == bio_files::BondType::Amide {
            bio_files::BondType::Single as u8 + 1
        } else {
            bond.bond_type as u8 + 1
        };
        result[j][i] = result[i][j];
    }
    Ok(result)
}

impl Ligand {
    pub fn creatine() -> Result<Self, ParamError> {
        Self::from_text(
            include_str!("../../param_data/opls/creatine.sdf"),
            include_str!("../../param_data/opls/creatine.itp"),
        )
    }

    pub fn from_text(sdf: &str, itp: &str) -> Result<Self, ParamError> {
        let source = Sdf::new(sdf)
            .map_err(|e| ParamError::new(&format!("OPLS ligand structure: {e}")))?;
        let text = if itp.lines().any(|l| l.split(';').next().unwrap_or("").trim() == "[ defaults ]") {
            itp.to_owned()
        } else {
            format!("[ defaults ]\n1 3 yes 0.5 0.5\n{itp}")
        };
        let top = GromacsTopology::new(&text)
            .map_err(|e| ParamError::new(&format!("OPLS ligand topology: {e}")))?;
        if top.molecule_types.len() != 1 || top.defaults.comb_rule != 3
            || (top.defaults.fudge_lj - 0.5).abs() > 1e-8
            || (top.defaults.fudge_qq - 0.5).abs() > 1e-8
        {
            return Err(ParamError::new("Unexpected OPLS ligand topology conventions"));
        }
        let mol = &top.molecule_types[0];
        let params = gromacs_molecule_params(mol, &top)?;
        let graph = edges(&source.atoms, &source.bonds)?;
        if source.atoms.len() != mol.atoms.len()
            || params.bonds.len() != source.bonds.len()
            || params.bonds.iter().any(|&((i, j), _)| graph[i][j] == 0)
            || source
                .atoms
                .iter()
                .zip(&params.masses)
                .any(|(a, mass)| (a.element.atomic_weight() - mass).abs() > 0.02)
            || !params.urey_bradley.is_empty()
            || !params.harmonic_impropers.is_empty()
            || !params.cmaps.is_empty()
            || !params.nbfix.is_empty()
        {
            return Err(ParamError::new(
                "OPLS ligand structure and parameter layout do not match",
            ));
        }
        Ok(Self {
            degree: graph
                .iter()
                .map(|row| row.iter().filter(|&&v| v != 0).count())
                .collect(),
            atoms: source.atoms,
            edges: graph,
            types: mol.atoms.iter().map(|a| a.atom_type.clone()).collect(),
            charges: mol.atoms.iter().map(|a| a.charge as f32).collect(),
            params,
        })
    }

    pub fn from_prepared(mol: &crate::MolDynamics) -> Option<Self> {
        let params = mol.explicit_params.clone()?;
        let graph = edges(&mol.atoms, &mol.bonds).ok()?;
        Some(Self {
            degree: graph.iter().map(|row| row.iter().filter(|&&v| v != 0).count()).collect(),
            atoms: mol.atoms.clone(),
            edges: graph,
            types: mol.atoms.iter().map(|a| a.force_field_type.clone()).collect::<Option<_>>()?,
            charges: mol.atoms.iter().map(|a| a.partial_charge).collect::<Option<_>>()?,
            params,
        })
    }

    fn match_graph(
        &self,
        atoms: &[AtomGeneric],
        graph: &[Vec<u8>],
        degree: &[usize],
        mapped: &mut [Option<usize>],
        aromatic: bool,
    ) -> bool {
        let next = (0..mapped.len())
            .filter(|&i| mapped[i].is_none())
            .max_by_key(|&i| {
                (
                    mapped
                        .iter()
                        .enumerate()
                        .filter(|(j, v)| v.is_some() && self.edges[i][*j] != 0)
                        .count(),
                    self.degree[i],
                )
            });
        let Some(i) = next else {
            return true;
        };
        for j in 0..atoms.len() {
            if mapped.contains(&Some(j))
                || atoms[j].element != self.atoms[i].element
                || degree[j] != self.degree[i]
                || mapped
                    .iter()
                    .enumerate()
                    .any(|(k, v)| v.is_some_and(|l| {
                        let a = self.edges[i][k];
                        let b = graph[j][l];
                        a != b && !(aromatic && b == bio_files::BondType::Aromatic as u8 + 1
                            && [bio_files::BondType::Single as u8 + 1,
                                bio_files::BondType::Double as u8 + 1].contains(&a))
                    }))
            {
                continue;
            }
            mapped[i] = Some(j);
            if self.match_graph(atoms, graph, degree, mapped, aromatic) {
                return true;
            }
            mapped[i] = None;
        }
        false
    }

    pub fn build(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<Option<ExplicitParams>, ParamError> {
        self.build_inner(atoms, bonds, false)
    }

    /// The service's preview kekulizes aromatic MOL bonds. Only aromatic input
    /// edges may match a single/double preview edge; other bond orders remain exact.
    pub fn build_generated(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
    ) -> Result<Option<ExplicitParams>, ParamError> {
        self.build_inner(atoms, bonds, true)
    }

    fn build_inner(
        &self,
        atoms: &mut [AtomGeneric],
        bonds: &[BondGeneric],
        aromatic: bool,
    ) -> Result<Option<ExplicitParams>, ParamError> {
        if atoms.len() != self.atoms.len() {
            return Ok(None);
        }
        let graph = edges(atoms, bonds)?;
        let degree: Vec<_> = graph
            .iter()
            .map(|row| row.iter().filter(|&&v| v != 0).count())
            .collect();
        let mut mapped = vec![None; atoms.len()];
        if !self.match_graph(atoms, &graph, &degree, &mut mapped, aromatic) {
            return Ok(None);
        }
        let m: Vec<_> = mapped.into_iter().map(Option::unwrap).collect();
        let mut p = self.params.clone();
        for (i, &j) in m.iter().enumerate() {
            p.masses[j] = self.params.masses[i];
            p.lj[j] = self.params.lj[i];
        }
        let pair = |i: usize, j: usize| (m[i].min(m[j]), m[i].max(m[j]));
        for ((i, j), _) in &mut p.bonds {
            (*i, *j) = (m[*i], m[*j]);
        }
        for ((i, j, k), _) in &mut p.angles {
            (*i, *j, *k) = (m[*i], m[*j], m[*k]);
        }
        for ((i, j, k, l), _) in p.dihedrals.iter_mut().chain(&mut p.impropers) {
            (*i, *j, *k, *l) = (m[*i], m[*j], m[*k], m[*l]);
        }
        for (i, j) in &mut p.exclusions {
            (*i, *j) = pair(*i, *j);
        }
        for ((i, j), _) in &mut p.pairs_14 {
            (*i, *j) = pair(*i, *j);
        }
        for (i, &j) in m.iter().enumerate() {
            atoms[j].force_field_type = Some(self.types[i].clone());
            atoms[j].partial_charge = Some(self.charges[i]);
        }
        Ok(Some(p))
    }
}
