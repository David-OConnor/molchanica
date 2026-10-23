//! Protein–ligand geometry checks and prepared Boltz-2 affinity requests.
//! Geometry, model confidence, and affinity remain separate results.

use std::{
    collections::{BTreeMap, HashMap, hash_map::DefaultHasher},
    fs,
    hash::{Hash, Hasher},
    io,
};

use bio_files::ResidueType;
use lin_alg::f64::Vec3;
use mol_defs::molecules::{Chain, common::MoleculeCommon, peptide::MoleculePeptide};
use na_seq::{AaIdent, Element};
use serde::Serialize;
use serde_json::{Value, json};

use crate::external_tools::{
    RunControl, Tool,
    pdb_write::{PdbWriteOptions, peptide_to_pdb},
    shared_adapter::{self, AdapterResult},
    tool_form::FormContract,
};

pub const CONTACT_DISTANCE: f64 = 4.;
/// Conservative heavy-atom overlap screen, in Å.
pub const CLASH_OVERLAP: f64 = 0.8;

/// The same observed residues define the sequence, template, and pocket numbering.
pub fn chain_residues(protein: &MoleculePeptide, chain: Option<usize>) -> Vec<usize> {
    protein
        .residues
        .iter()
        .enumerate()
        .filter_map(|(i, residue)| {
            (matches!(residue.res_type, ResidueType::AminoAcid(_))
                && residue.atoms.iter().any(|&a| {
                    protein
                        .common
                        .atoms
                        .get(a)
                        .is_some_and(|a| a.chain == chain)
                }))
            .then_some(i)
        })
        .collect()
}

pub fn residue_label(protein: &MoleculePeptide, residue: usize) -> String {
    let res = &protein.residues[residue];
    let name = match &res.res_type {
        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::ThreeLetters).to_ascii_uppercase(),
        other => other.to_string(),
    };
    format!("{name} {}", res.serial_number)
}

fn validate_coordinates(mol: &MoleculeCommon) -> Result<(), String> {
    if mol.is_2d {
        return Err("Generate or load a 3D conformer before assessing this pose.".into());
    }
    if mol.atoms.is_empty() || mol.atoms.len() != mol.atom_posits.len() {
        return Err("The molecule has missing atom coordinates.".into());
    }
    if mol
        .atom_posits
        .iter()
        .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
    {
        return Err("The molecule contains nonfinite atom coordinates.".into());
    }
    let mut alternatives = mol
        .atoms
        .iter()
        .filter_map(|a| a.alt_conformation_id.as_deref())
        .filter(|id| !id.is_empty() && *id != "." && *id != "?")
        .collect::<Vec<_>>();
    alternatives.sort_unstable();
    alternatives.dedup();
    if alternatives.len() > 1 {
        return Err(
            "Choose one alternate conformation before assessing or templating this structure."
                .into(),
        );
    }
    Ok(())
}

/// Track displayed coordinates and chemistry, including edits and rigid transforms.
pub fn fingerprint(
    protein: &MoleculePeptide,
    ligand: &MoleculeCommon,
    chain: Option<usize>,
) -> u64 {
    let mut hash = DefaultHasher::new();
    chain.hash(&mut hash);
    for mol in [&protein.common, ligand] {
        mol.ident.hash(&mut hash);
        mol.path.hash(&mut hash);
        mol.is_2d.hash(&mut hash);
        for atom in &mol.atoms {
            atom.serial_number.hash(&mut hash);
            atom.element.hash(&mut hash);
            atom.residue.hash(&mut hash);
            atom.chain.hash(&mut hash);
            atom.alt_conformation_id.hash(&mut hash);
        }
        for position in &mol.atom_posits {
            position.x.to_bits().hash(&mut hash);
            position.y.to_bits().hash(&mut hash);
            position.z.to_bits().hash(&mut hash);
        }
        for bond in &mol.bonds {
            bond.atom_0.hash(&mut hash);
            bond.atom_1.hash(&mut hash);
            std::mem::discriminant(&bond.bond_type).hash(&mut hash);
        }
    }
    for residue in &protein.residues {
        residue.serial_number.hash(&mut hash);
        residue.res_type.to_string().hash(&mut hash);
        residue.atoms.hash(&mut hash);
    }
    hash.finish()
}

#[derive(Clone, Serialize)]
pub struct ResidueContact {
    pub residue: usize,
    pub label: String,
    pub closest_distance_a: f64,
    pub atom_pairs: usize,
}

#[derive(Clone, Serialize)]
pub struct Clash {
    pub protein_atom_sn: u32,
    pub ligand_atom_sn: u32,
    pub distance_a: f64,
    pub overlap_a: f64,
}

#[derive(Clone, Serialize)]
pub struct PoseReport {
    pub protein_identifier: String,
    pub ligand_identifier: String,
    pub chain_index: Option<usize>,
    pub fingerprint: u64,
    pub contact_distance_a: f64,
    pub clash_overlap_a: f64,
    pub ligand_heavy_atoms: usize,
    pub contacting_ligand_atoms: usize,
    pub contacts: Vec<ResidueContact>,
    pub clash_count: usize,
    /// At most 200 examples; `clash_count` counts every pair.
    pub clashes: Vec<Clash>,
    pub bond_length_outliers: Vec<String>,
}

/// Screen the displayed pose without optimization or affinity calculation.
pub fn assess_pose(
    protein: &MoleculePeptide,
    ligand: &MoleculeCommon,
    chain: Option<usize>,
) -> Result<PoseReport, String> {
    validate_coordinates(&protein.common)?;
    validate_coordinates(ligand)?;
    let residues = chain_residues(protein, chain);
    if residues.is_empty() {
        return Err("Choose a chain with observed amino-acid residues.".into());
    }
    let mut report = PoseReport {
        protein_identifier: protein.common.ident.clone(),
        ligand_identifier: ligand.ident.clone(),
        chain_index: chain,
        fingerprint: fingerprint(protein, ligand, chain),
        contact_distance_a: CONTACT_DISTANCE,
        clash_overlap_a: CLASH_OVERLAP,
        ligand_heavy_atoms: 0,
        contacting_ligand_atoms: 0,
        contacts: Vec::new(),
        clash_count: 0,
        clashes: Vec::new(),
        bond_length_outliers: Vec::new(),
    };
    let cell = |p: Vec3| {
        (
            (p.x / CONTACT_DISTANCE).floor() as i64,
            (p.y / CONTACT_DISTANCE).floor() as i64,
            (p.z / CONTACT_DISTANCE).floor() as i64,
        )
    };
    let mut grid: HashMap<_, Vec<(usize, usize)>> = HashMap::new();
    let mut receptor_radius = 0_f64;
    for &r in &residues {
        for &a in &protein.residues[r].atoms {
            let Some(atom) = protein.common.atoms.get(a) else {
                return Err("Protein residue references a missing atom.".into());
            };
            if atom.chain == chain && atom.element != Element::Hydrogen {
                receptor_radius = receptor_radius.max(atom.element.vdw_radius() as f64);
                grid.entry(cell(protein.common.atom_posits[a]))
                    .or_default()
                    .push((r, a));
            }
        }
    }
    let mut contacts: BTreeMap<usize, ResidueContact> = BTreeMap::new();
    for (l, atom) in ligand.atoms.iter().enumerate() {
        if atom.element == Element::Hydrogen {
            continue;
        }
        report.ligand_heavy_atoms += 1;
        let position = ligand.atom_posits[l];
        let (x, y, z) = cell(position);
        let mut contacting = false;
        let search_distance = CONTACT_DISTANCE
            .max(receptor_radius + atom.element.vdw_radius() as f64 - CLASH_OVERLAP);
        let reach = (search_distance / CONTACT_DISTANCE).ceil() as i64;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    let (Some(nx), Some(ny), Some(nz)) =
                        (x.checked_add(dx), y.checked_add(dy), z.checked_add(dz))
                    else {
                        continue;
                    };
                    let Some(neighbors) = grid.get(&(nx, ny, nz)) else {
                        continue;
                    };
                    for &(r, a) in neighbors {
                        let distance = (position - protein.common.atom_posits[a]).magnitude();
                        if distance <= CONTACT_DISTANCE {
                            contacting = true;
                            let contact = contacts.entry(r).or_insert_with(|| ResidueContact {
                                residue: r,
                                label: residue_label(protein, r),
                                closest_distance_a: distance,
                                atom_pairs: 0,
                            });
                            contact.atom_pairs += 1;
                            contact.closest_distance_a = contact.closest_distance_a.min(distance);
                        }
                        let receptor_atom = &protein.common.atoms[a];
                        let overlap = atom.element.vdw_radius() as f64
                            + receptor_atom.element.vdw_radius() as f64
                            - distance;
                        if overlap > CLASH_OVERLAP {
                            report.clash_count += 1;
                            if report.clashes.len() < 200 {
                                report.clashes.push(Clash {
                                    protein_atom_sn: receptor_atom.serial_number,
                                    ligand_atom_sn: atom.serial_number,
                                    distance_a: distance,
                                    overlap_a: overlap,
                                });
                            }
                        }
                    }
                }
            }
        }
        report.contacting_ligand_atoms += usize::from(contacting);
    }
    if report.ligand_heavy_atoms == 0 {
        return Err("The ligand has no heavy atoms.".into());
    }
    report.contacts = contacts.into_values().collect();
    report
        .contacts
        .sort_by(|a, b| a.closest_distance_a.total_cmp(&b.closest_distance_a));
    for bond in &ligand.bonds {
        if bond.bond_type == bio_files::BondType::NotConnected {
            continue;
        }
        let Some(a) = ligand.atoms.get(bond.atom_0) else {
            return Err("A ligand bond references a missing atom.".into());
        };
        let Some(b) = ligand.atoms.get(bond.atom_1) else {
            return Err("A ligand bond references a missing atom.".into());
        };
        if a.element == Element::Hydrogen || b.element == Element::Hydrogen {
            continue;
        }
        let length =
            (ligand.atom_posits[bond.atom_0] - ligand.atom_posits[bond.atom_1]).magnitude();
        if !(0.7..=2.5).contains(&length) {
            report.bond_length_outliers.push(format!(
                "Atoms {}–{}: {length:.2} Å",
                a.serial_number, b.serial_number
            ));
        }
    }
    Ok(report)
}

pub struct PredictionOptions {
    pub job_name: String,
    pub use_template: bool,
    pub use_msa_server: bool,
    pub use_potentials: bool,
    pub structure_samples: u32,
    pub affinity_samples: u32,
    pub seed: u32,
    pub pocket_distance: f64,
}

impl Default for PredictionOptions {
    fn default() -> Self {
        Self {
            job_name: "binding".into(),
            use_template: true,
            use_msa_server: false,
            use_potentials: true,
            structure_samples: 1,
            affinity_samples: 5,
            seed: 0,
            pocket_distance: 6.,
        }
    }
}

/// The model receives SMILES and regenerates the ligand pose.
pub struct PredictionRequest {
    pub payload: Value,
    pub template: Option<String>,
    pub provenance: Value,
}

impl PredictionRequest {
    pub fn new(
        protein: &MoleculePeptide,
        ligand: &MoleculeCommon,
        chain: Option<usize>,
        smiles: &str,
        pocket_residues: &[usize],
        options: &PredictionOptions,
    ) -> Result<Self, String> {
        let smiles = smiles.trim();
        if smiles.is_empty() || smiles.contains(char::is_whitespace) || smiles.len() > 100_000 {
            return Err("Supply one ligand as SMILES without whitespace.".into());
        }
        if !(1..=50).contains(&options.structure_samples)
            || !(1..=50).contains(&options.affinity_samples)
            || !(4.0..=20.).contains(&options.pocket_distance)
        {
            return Err("Invalid prediction sample count or pocket distance.".into());
        }
        let residues = chain_residues(protein, chain);
        if residues.is_empty() {
            return Err("Choose a protein chain with observed amino-acid residues.".into());
        }
        let mut sequence = String::new();
        for &r in &residues {
            let ResidueType::AminoAcid(aa) = protein.residues[r].res_type else {
                unreachable!();
            };
            let letter = aa.to_str(AaIdent::OneLetter);
            if !"ACDEFGHIKLMNPQRSTVWY".contains(&letter) {
                return Err(format!(
                    "{} needs a modified-residue input in the advanced Boltz form.",
                    residue_label(protein, r)
                ));
            }
            sequence.push_str(&letter);
        }
        let mut contacts = Vec::new();
        for r in pocket_residues {
            let position = residues
                .iter()
                .position(|candidate| candidate == r)
                .ok_or_else(|| {
                    "A pocket residue is outside the selected protein chain.".to_owned()
                })?;
            contacts.push(json!(["A", position + 1]));
        }
        let mut values = FormContract::load("boltz2")?.defaults();
        values.insert("job_name".into(), options.job_name.clone());
        values.insert(
            "sequence_molecules".into(),
            json!([
                {"type": "protein", "id": "A", "sequence": sequence, "modifications": []},
                {"type": "ligand", "id": "B", "ligand": smiles}
            ])
            .to_string(),
        );
        values.insert(
            "properties".into(),
            json!([{"affinity": {"binder": "B"}}]).to_string(),
        );
        if !contacts.is_empty() {
            values.insert(
                "constraints".into(),
                json!([{"pocket": {
                    "binder": "B", "contacts": contacts,
                    "max_distance": options.pocket_distance, "force": false
                }}])
                .to_string(),
            );
        }
        for (name, value) in [
            ("use_msa_server", options.use_msa_server.to_string()),
            ("use_potentials", options.use_potentials.to_string()),
            ("diffusion_samples", options.structure_samples.to_string()),
            (
                "diffusion_samples_affinity",
                options.affinity_samples.to_string(),
            ),
            ("seed", options.seed.to_string()),
        ] {
            values.insert(name.into(), value);
        }
        let payload = shared_adapter::payload(Tool::Boltz2, &values, "parameters", &HashMap::new())
            .map_err(|e| e.to_string())?;
        let template = if options.use_template {
            validate_coordinates(&protein.common)?;
            if residues.len() > 9_999 {
                return Err("This chain is too large for the PDB template format.".into());
            }
            let mut atom_count = 0;
            for &r in &residues {
                for &a in &protein.residues[r].atoms {
                    let position = protein.common.atom_posits.get(a).ok_or_else(|| {
                        "A template residue references a missing atom.".to_owned()
                    })?;
                    if [position.x, position.y, position.z]
                        .iter()
                        .any(|&v| !(-999.999..=9999.999).contains(&v))
                    {
                        return Err("Template coordinates exceed PDB field limits. Recenter the protein first.".into());
                    }
                    atom_count += 1;
                }
            }
            if atom_count > 99_999 {
                return Err("This chain has too many atoms for a PDB template.".into());
            }
            let mut protein = protein.clone();
            for (atom, &position) in protein
                .common
                .atoms
                .iter_mut()
                .zip(&protein.common.atom_posits)
            {
                atom.posit = position;
            }
            for (position, &r) in residues.iter().enumerate() {
                protein.residues[r].serial_number = position as u32 + 1;
            }
            protein.chains = vec![Chain {
                id: "A".into(),
                residues: residues.clone(),
                residue_sns: Vec::new(),
                atoms: Vec::new(),
                atom_sns: Vec::new(),
                visible: true,
            }];
            Some(peptide_to_pdb(&protein, &PdbWriteOptions::default()).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let snapshot = |mol: &MoleculeCommon| {
            json!({
                "identifier": mol.ident, "path": mol.path,
                "atoms": mol.atoms.iter().enumerate().map(|(i, a)| json!({
                    "serial": a.serial_number, "element": a.element.to_letter(),
                    "residue_index": a.residue, "chain_index": a.chain,
                    "position_a": mol.atom_posits.get(i).map(|p| [p.x, p.y, p.z])
                })).collect::<Vec<_>>(),
                "bonds": mol.bonds.iter().map(|b| json!({
                    "atoms": [b.atom_0, b.atom_1], "type": b.bond_type.to_string()
                })).collect::<Vec<_>>()
            })
        };
        Ok(Self {
            payload,
            template,
            provenance: json!({
                "molchanica_version": crate::VERSION,
                "job_name": options.job_name,
                "backend": "boltz2", "coordinates_regenerated": true,
                "protein_chain_index": chain, "ligand_smiles": smiles,
                "protein": snapshot(&protein.common), "displayed_ligand": snapshot(ligand),
                "residue_mapping": residues.iter().enumerate().map(|(position, &r)| json!({
                    "boltz_chain": "A", "boltz_position": position + 1,
                    "source_residue_index": r, "source_label": residue_label(protein, r)
                })).collect::<Vec<_>>(),
                "pocket_residue_indices": pocket_residues,
                "scope": "One protein chain and one small molecule; waters, cofactors, and other chains excluded."
            }),
        })
    }

    pub fn run(mut self, control: &RunControl) -> io::Result<AdapterResult> {
        shared_adapter::run_prepared(Tool::Boltz2, control, |directory| {
            fs::write(
                directory.join("binding-input.json"),
                serde_json::to_vec_pretty(&self.provenance)?,
            )?;
            if let Some(template) = self.template {
                let path = directory.join("receptor.pdb");
                fs::write(&path, template)?;
                self.payload["templates"] = Value::String(
                    json!([
                        {"pdb": path, "chain_id": "A", "force": false}
                    ])
                    .to_string(),
                );
            }
            Ok(self.payload)
        })
    }
}

/// Missing or invalid predictions never become zeros.
#[derive(Debug)]
pub struct AffinityPrediction {
    pub log_ic50_um: Option<f64>,
    pub binder_probability: Option<f64>,
}

impl AffinityPrediction {
    pub fn parse(value: &Value) -> Result<Self, String> {
        let number = |key: &str| -> Result<Option<f64>, String> {
            match value.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => v
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .map(Some)
                    .ok_or_else(|| format!("Invalid Boltz output: {key}")),
            }
        };
        let log_ic50_um = number("affinity_pred_value")?;
        let binder_probability = number("affinity_probability_binary")?;
        if binder_probability.is_some_and(|p| !(0.0..=1.).contains(&p)) {
            return Err("Boltz binder probability is outside 0–1.".into());
        }
        if log_ic50_um.is_none() && binder_probability.is_none() {
            return Err("The affinity file contains neither affinity prediction head.".into());
        }
        Ok(Self {
            log_ic50_um,
            binder_probability,
        })
    }
}
