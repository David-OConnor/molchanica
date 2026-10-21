//! The selected force field for loaded molecules and MD. Structural preparation
//! still uses Amber templates; this layer assigns the selected family's atom
//! types and charges without changing coordinates, residue names or hydrogens.

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
};

use bio_files::ResidueGeneric;
use dynamics::{
    ExplicitParams, FfMolType, ForceFieldFamily, MdConfig, MolDynamics, params::FfParamSet,
};
use egui::{Color32, ComboBox, Ui};
use mol_defs::molecules::{MolType, common::MoleculeCommon, nucleic_acid::NucleicAcidType};

use crate::{state::State, util::handle_err};

type Annotation = (Option<String>, Option<f32>);

struct Assignment {
    topology: u64,
    annotations: u64,
    family: ForceFieldFamily,
    explicit: Option<ExplicitParams>,
    atom_serials: Vec<u32>,
    error: Option<String>,
    pending: bool,
}

#[derive(Default)]
pub struct ForceFields {
    pub family: ForceFieldFamily,
    /// Non-Amber data. Amber preparation data lives in State::ff_param_set.
    pub params: Option<FfParamSet>,
    assignments: HashMap<(u8, usize), Assignment>,
    amber: HashMap<(u8, u64), Vec<Annotation>>,
    dirty: bool,
    opls_jobs: HashMap<u64, Receiver<Result<MolDynamics, String>>>,
    opls_results: HashMap<u64, Result<MolDynamics, String>>,
}

pub fn family_name(family: ForceFieldFamily) -> &'static str {
    match family {
        ForceFieldFamily::Amber => "Amber",
        ForceFieldFamily::Charmm36 => "CHARMM36m",
        ForceFieldFamily::Opls => "OPLS-AA/L",
    }
}

fn topology_hash(mol: &MoleculeCommon, residues: &[ResidueGeneric]) -> u64 {
    let mut h = DefaultHasher::new();
    mol.ident.hash(&mut h);
    mol.path.hash(&mut h);
    mol.atoms.len().hash(&mut h);
    mol.bonds.len().hash(&mut h);
    for atom in &mol.atoms {
        atom.serial_number.hash(&mut h);
        atom.element.atomic_number().hash(&mut h);
        atom.type_in_res
            .as_ref()
            .map(ToString::to_string)
            .hash(&mut h);
        atom.type_in_res_general.hash(&mut h);
        atom.hetero.hash(&mut h);
    }
    for bond in &mol.bonds {
        bond.atom_0_sn.hash(&mut h);
        bond.atom_1_sn.hash(&mut h);
        (bond.bond_type as u8).hash(&mut h);
    }
    for res in residues {
        res.atom_sns.hash(&mut h);
        format!("{:?}{:?}", res.res_type, res.end).hash(&mut h);
    }
    h.finish()
}

fn annotation_hash(mol: &MoleculeCommon) -> u64 {
    let mut h = DefaultHasher::new();
    for atom in &mol.atoms {
        atom.force_field_type.hash(&mut h);
        atom.partial_charge.map(f32::to_bits).hash(&mut h);
    }
    h.finish()
}

fn annotations(mol: &MoleculeCommon) -> Vec<Annotation> {
    mol.atoms
        .iter()
        .map(|a| (a.force_field_type.clone(), a.partial_charge))
        .collect()
}

fn amber_annotations(
    mol: &MoleculeCommon,
    kind: FfMolType,
    residues: &[ResidueGeneric],
    set: &FfParamSet,
    specific: &mut HashMap<String, bio_files::md_params::ForceFieldParams>,
) -> Result<Vec<Annotation>, String> {
    let general = match kind {
        FfMolType::Peptide => &set.peptide,
        FfMolType::SmallOrganic => &set.small_mol,
        FfMolType::Lipid => &set.lipids,
        FfMolType::Dna => &set.dna,
        FfMolType::Rna => &set.rna,
        _ => &set.carbohydrates,
    };
    if mol.atoms.iter().all(|a| {
        a.partial_charge.is_some()
            && a.force_field_type
                .as_ref()
                .is_some_and(|t| general.as_ref().is_some_and(|p| p.mass.contains_key(t)))
    }) {
        return Ok(annotations(mol));
    }
    let mut atoms: Vec<_> = mol.atoms.iter().map(|a| a.to_generic()).collect();
    let bonds: Vec<_> = mol.bonds.iter().map(|b| b.to_generic()).collect();
    for a in &mut atoms {
        a.force_field_type = None;
        a.partial_charge = None;
    }
    match kind {
        FfMolType::SmallOrganic => {
            let params = set
                .small_mol_typer
                .assign(&mut atoms, &bonds, None, set.small_mol.as_ref())
                .map_err(|e| e.to_string())?;
            specific.insert(mol.ident.clone(), params);
        }
        FfMolType::Peptide => {
            let map = set
                .peptide_ff_q_map
                .as_ref()
                .ok_or("Missing Amber protein templates")?;
            dynamics::params::populate_peptide_ff_and_q(&mut atoms, residues, map)
                .map_err(|e| e.descrip)?;
        }
        _ => {
            return Err(format!(
                "{}: reload the Amber template after topology changes",
                mol.ident
            ));
        }
    }
    Ok(atoms
        .into_iter()
        .map(|a| (a.force_field_type, a.partial_charge))
        .collect())
}

impl ForceFields {
    pub fn has_pending(&self) -> bool {
        !self.opls_jobs.is_empty()
    }

    fn poll_opls(&mut self) {
        let mut completed = Vec::new();
        for (&topology, rx) in &self.opls_jobs {
            match rx.try_recv() {
                Ok(result) => completed.push((topology, result)),
                Err(TryRecvError::Disconnected) => completed.push((topology,
                    Err("OPLS parameter generation worker stopped unexpectedly".into()))),
                Err(TryRecvError::Empty) => (),
            }
        }
        for (topology, result) in completed {
            self.opls_jobs.remove(&topology);
            self.opls_results.insert(topology, result);
            self.assignments.retain(|_, a| {
                a.topology != topology || a.family != ForceFieldFamily::Opls
            });
            self.dirty = true;
        }
    }

    fn retry_opls(&mut self) {
        self.opls_results.retain(|_, result| result.is_ok());
        self.assignments.retain(|_, a| a.family != ForceFieldFamily::Opls || a.error.is_none());
        self.dirty = true;
    }

    /// Amber-specific analysis and editing tools work on this copy. Keep their
    /// trained models and preparation parameters consistent while the loaded
    /// molecule continues to display the user's selected family.
    pub fn amber_copy(&self, mol: &MoleculeCommon) -> MoleculeCommon {
        let mut copy = mol.clone();
        let key = (MolType::Ligand as u8, topology_hash(mol, &[]));
        if let Some(source) = self.amber.get(&key) {
            for (a, (name, q)) in copy.atoms.iter_mut().zip(source) {
                a.force_field_type = name.clone();
                a.partial_charge = *q;
            }
        } else if copy.atoms.iter().any(|a| {
            a.force_field_type
                .as_deref()
                .is_some_and(|t| t.starts_with("opls_"))
        }) {
            // Allow the Amber preparer to type a new, uncached copy.
            for a in &mut copy.atoms {
                a.force_field_type = None;
                a.partial_charge = None;
            }
        }
        copy
    }

    /// A single entry point for retyping existing molecules and new arrivals.
    /// Cache keys include topology and annotations, so coordinate-only updates
    /// and MD playback do not cause repeated parameterization.
    fn assign(
        &mut self,
        key: (u8, usize),
        mol: &mut MoleculeCommon,
        kind: FfMolType,
        residues: Vec<ResidueGeneric>,
        amber_data: &FfParamSet,
        specific: &mut HashMap<String, bio_files::md_params::ForceFieldParams>,
    ) -> bool {
        let topology = topology_hash(mol, &residues);
        let current = annotation_hash(mol);
        let previous = self
            .assignments
            .get(&key)
            .filter(|a| a.topology == topology);
        if previous.is_some_and(|a| a.family == self.family && a.annotations == current) {
            return false;
        }

        let restoring_amber = self.family == ForceFieldFamily::Amber
            && previous.is_some_and(|a| a.family != self.family && a.annotations == current);
        let specific_params = specific.get(&mol.ident).or_else(|| {
            specific
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(&mol.ident))
                .map(|(_, p)| p)
        });
        let imported_foreign_types = self.family == ForceFieldFamily::Amber
            && !restoring_amber
            && kind == FfMolType::SmallOrganic
            && mol.atoms.iter().any(|a| {
                a.force_field_type.as_ref().is_none_or(|t| {
                    !amber_data
                        .small_mol
                        .as_ref()
                        .is_some_and(|p| p.mass.contains_key(t))
                        && !specific_params.is_some_and(|p| p.mass.contains_key(t))
                })
            });

        // Keep the original Amber annotations, including supplied ligand
        // charges. Copies and reordered picker entries can share this source.
        let amber = if imported_foreign_types {
            amber_annotations(mol, kind, &residues, amber_data, specific)
                .unwrap_or_else(|_| vec![(None, None); mol.atoms.len()])
        } else if let Some(source) = self.amber.get(&(key.0, topology)) {
            source.clone()
        } else if self.family == ForceFieldFamily::Amber {
            annotations(mol)
        } else {
            // A graph edited while OPLS/CHARMM was active may still carry its
            // old family labels. Rebuild its Amber preparation source rather
            // than caching those labels as Amber for a future switch back.
            amber_annotations(mol, kind, &residues, amber_data, specific)
                .unwrap_or_else(|_| vec![(None, None); mol.atoms.len()])
        };

        let mut explicit = None;
        let mut atom_serials = Vec::new();
        let mut error = None;
        let mut pending = false;
        if self.family == ForceFieldFamily::Amber {
            // An Amber-side load or parameter edit already supplied new data.
            // Otherwise restore the annotations saved before switching family.
            if restoring_amber || imported_foreign_types {
                for (a, (name, q)) in mol.atoms.iter_mut().zip(&amber) {
                    a.force_field_type = name.clone();
                    a.partial_charge = *q;
                }
            }
            if mol
                .atoms
                .iter()
                .filter(|a| kind != FfMolType::Peptide || !a.hetero)
                .any(|a| a.force_field_type.is_none() || a.partial_charge.is_none())
            {
                error = Some(format!(
                    "{}: Amber atom types or charges are unavailable; \
                    reload or prepare this molecule",
                    mol.ident
                ));
            }
        } else {
            let input = MolDynamics {
                ff_mol_type: kind,
                atoms: mol
                    .atoms
                    .iter()
                    .filter(|a| kind != FfMolType::Peptide || !a.hetero)
                    .map(|a| a.to_generic())
                    .collect(),
                bonds: mol.bonds.iter().map(|b| b.to_generic()).collect(),
                residues: (!residues.is_empty()).then_some(residues),
                ..Default::default()
            };
            let allowed: std::collections::HashSet<_> =
                input.atoms.iter().map(|a| a.serial_number).collect();
            let input = MolDynamics {
                bonds: input
                    .bonds
                    .iter()
                    .filter(|b| allowed.contains(&b.atom_0_sn) && allowed.contains(&b.atom_1_sn))
                    .cloned()
                    .collect(),
                ..input
            };
            let result = if self.family == ForceFieldFamily::Opls && kind == FfMolType::SmallOrganic {
                if let Some(result) = self.opls_results.get(&topology) {
                    if let Ok(prepared) = result {
                        self.params.as_ref().unwrap().opls.as_ref().unwrap()
                            .remember_small_molecule(prepared);
                    }
                    Some(result.clone())
                } else if self.opls_jobs.contains_key(&topology) {
                    None
                } else {
                    let mut prepared = input.clone();
                    let local = self.params.as_ref().unwrap().opls.as_ref().unwrap()
                        .build_small_molecule_cached(&mut prepared.atoms, &prepared.bonds);
                    match local {
                        Ok(params) => {
                            prepared.explicit_params = Some(params);
                            Some(Ok(prepared))
                        }
                        Err(_) => {
                            let (tx, rx) = mpsc::channel();
                            self.opls_jobs.insert(topology, rx);
                            thread::spawn(move || {
                                let result = FfParamSet::new_opls()
                                    .map_err(|e| e.to_string())
                                    .and_then(|set| dynamics::gromacs::parameterize(&input, &set)
                                        .map_err(|e| e.descrip));
                                let _ = tx.send(result);
                            });
                            None
                        }
                    }
                }
            } else {
                Some(dynamics::gromacs::parameterize(&input, self.params.as_ref().unwrap())
                    .map_err(|e| e.descrip))
            };
            // A failed family assignment must not leave Amber names/charges
            // attached to a molecule presented as CHARMM or OPLS.
            for a in &mut mol.atoms {
                a.force_field_type = None;
                a.partial_charge = None;
            }
            match result {
                Some(Ok(prepared)) => {
                    atom_serials = prepared.atoms.iter().map(|a| a.serial_number).collect();
                    let by_serial: HashMap<_, _> = prepared
                        .atoms
                        .iter()
                        .map(|a| (a.serial_number, a))
                        .collect();
                    for a in &mut mol.atoms {
                        if let Some(source) = by_serial.get(&a.serial_number) {
                            a.force_field_type = source.force_field_type.clone();
                            a.partial_charge = source.partial_charge;
                        }
                    }
                    explicit = prepared.explicit_params;
                }
                Some(Err(e)) => error = Some(format!("{}: {e}", mol.ident)),
                None => pending = true,
            }
        }
        let amber = if self.family == ForceFieldFamily::Amber {
            annotations(mol)
        } else {
            amber
        };
        self.amber.insert((key.0, topology), amber.clone());
        self.assignments.insert(
            key,
            Assignment {
                topology,
                annotations: annotation_hash(mol),
                family: self.family,
                explicit,
                atom_serials,
                error,
                pending,
            },
        );
        true
    }

    pub fn errors(&self) -> impl Iterator<Item = &str> {
        self.assignments.values().filter_map(|a| a.error.as_deref())
    }

    pub fn bond_params(
        &self,
        kind: MolType,
        index: usize,
        a: u32,
        b: u32,
    ) -> Option<(&bio_files::md_params::BondStretchingParams, f32, f32)> {
        let entry = self.assignments.get(&(kind as u8, index))?;
        let i = entry.atom_serials.iter().position(|&sn| sn == a)?;
        let j = entry.atom_serials.iter().position(|&sn| sn == b)?;
        let explicit = entry.explicit.as_ref()?;
        explicit
            .bonds
            .iter()
            .find(|((x, y), _)| (*x == i && *y == j) || (*x == j && *y == i))
            .map(|(_, p)| (p, explicit.masses[i], explicit.masses[j]))
    }

    #[cfg(feature = "sonification")]
    pub fn sonification_input(
        &self,
        kind: MolType,
        index: usize,
        mol: &MoleculeCommon,
    ) -> Result<(MoleculeCommon, bio_files::md_params::ForceFieldParams), String> {
        use bio_files::md_params::{ForceFieldParams, MassParams};

        let entry = self
            .assignments
            .get(&(kind as u8, index))
            .ok_or("The molecule has not been parameterized yet")?;
        let explicit = entry.explicit.as_ref().ok_or_else(|| {
            entry
                .error
                .clone()
                .unwrap_or("Missing molecule parameters".into())
        })?;
        let mut copy = mol.clone();
        let mut params = ForceFieldParams::default();
        let mut names = Vec::new();
        for (i, &sn) in entry.atom_serials.iter().enumerate() {
            let atom = copy
                .atoms
                .iter_mut()
                .find(|a| a.serial_number == sn)
                .ok_or("Parameter atom is absent from molecule")?;
            // Sonification's type-keyed API needs unique local keys to retain
            // residue-specific bond parameters. Only this audio copy uses them.
            let name = format!(
                "{}@{sn}",
                atom.force_field_type.as_deref().unwrap_or("atom")
            );
            atom.force_field_type = Some(name.clone());
            params.mass.insert(
                name.clone(),
                MassParams {
                    atom_type: name.clone(),
                    mass: explicit.masses[i],
                    comment: None,
                },
            );
            names.push(name);
        }
        for (&(i, j), source) in explicit.bonds.iter().map(|(ids, p)| (ids, p)) {
            let mut p = source.clone();
            p.atom_types = (names[i].clone(), names[j].clone());
            params.bond.insert(p.atom_types.clone(), p);
        }
        Ok((copy, params))
    }
}

impl State {
    /// Avoid starting synchronous network work from MD/energy/export UI paths.
    /// Only molecules actually selected for this simulation can block it.
    pub fn check_md_force_fields(&self) -> Result<(), String> {
        if self.force_fields.family == ForceFieldFamily::Amber {
            return Ok(());
        }
        let molecules = self.peptides.iter().enumerate()
            .map(|(i, m)| (MolType::Peptide, i, &m.common))
            .chain(self.ligands.iter().enumerate().map(|(i, m)| (MolType::Ligand, i, &m.common)))
            .chain(self.lipids.iter().enumerate().map(|(i, m)| (MolType::Lipid, i, &m.common)))
            .chain(self.nucleic_acids.iter().enumerate().map(|(i, m)| (MolType::NucleicAcid, i, &m.common)));

        for (kind, i, mol) in molecules {
            if mol.selected_for_md.is_none() { continue; }
            let assignment = self.force_fields.assignments.get(&(kind as u8, i))
                .filter(|a| a.family == self.force_fields.family)
                .ok_or_else(|| format!("{}: force field assignment is not ready yet", mol.ident))?;
            if assignment.pending {
                return Err(format!("{}: generating OPLS parameters with LigParGen; wait for parameterization to finish", mol.ident));
            }
            if let Some(error) = &assignment.error { return Err(error.clone()); }
        }
        Ok(())
    }

    pub fn selected_force_field(&self) -> &FfParamSet {
        self.force_fields
            .params
            .as_ref()
            .unwrap_or(&self.ff_param_set)
    }

    pub fn select_force_field(&mut self, family: ForceFieldFamily) -> Result<(), String> {
        if family == self.force_fields.family {
            return Ok(());
        }
        // Capture newly loaded Amber annotations before changing the family.
        self.refresh_force_fields();
        let params = if family == ForceFieldFamily::Amber {
            None
        } else {
            Some(FfParamSet::from_family(family).map_err(|e| e.to_string())?)
        };
        self.force_fields.params = params;
        self.force_fields.family = family;
        if family == ForceFieldFamily::Opls {
            self.force_fields.retry_opls();
        }
        let preset = MdConfig::for_family(family);
        let cfg = &mut self.to_save.md.config;
        cfg.ff_family = family;
        cfg.coulomb_cutoff = preset.coulomb_cutoff;
        cfg.lj_cutoff = preset.lj_cutoff;
        cfg.lj_modifier = preset.lj_modifier;
        cfg.water_model = None;
        self.ui.md.sync(cfg, self.to_save.md.dt);

        self.volatile.md_local.running = false;
        self.volatile.md_local.launching = false;
        self.volatile.md_local.mol_dynamics = None;
        self.volatile.md_local.computation = None;
        self.volatile.md_local.draw_md_mols = false;
        #[cfg(feature = "sonification")]
        {
            self.volatile.playing_audio = None;
        }
        self.refresh_force_fields();
        self.force_fields.dirty = true;
        Ok(())
    }

    /// Called before drawing and after background loads. The same path handles
    /// every molecule source, copies, and topology edits.
    pub fn refresh_force_fields(&mut self) -> bool {
        self.force_fields.poll_opls();
        let mut changed = std::mem::take(&mut self.force_fields.dirty);
        for (i, mol) in self.peptides.iter_mut().enumerate() {
            changed |= self.force_fields.assign(
                (MolType::Peptide as u8, i),
                &mut mol.common,
                FfMolType::Peptide,
                mol.residues.iter().map(|r| r.to_generic()).collect(),
                &self.ff_param_set,
                &mut self.mol_specific_params,
            );
        }
        for (i, mol) in self.ligands.iter_mut().enumerate() {
            if self.force_fields.assign(
                (MolType::Ligand as u8, i),
                &mut mol.common,
                FfMolType::SmallOrganic,
                Vec::new(),
                &self.ff_param_set,
                &mut self.mol_specific_params,
            ) {
                mol.conformer = None;
                mol.ff_params_loaded = mol
                    .common
                    .atoms
                    .iter()
                    .all(|a| a.force_field_type.is_some() && a.partial_charge.is_some());
                mol.frcmod_loaded = self
                    .mol_specific_params
                    .keys()
                    .any(|key| key.eq_ignore_ascii_case(&mol.common.ident));
                changed = true;
            }
        }
        for (i, mol) in self.lipids.iter_mut().enumerate() {
            changed |= self.force_fields.assign(
                (MolType::Lipid as u8, i),
                &mut mol.common,
                FfMolType::Lipid,
                mol.residues.iter().map(|r| r.to_generic()).collect(),
                &self.ff_param_set,
                &mut self.mol_specific_params,
            );
        }
        for (i, mol) in self.nucleic_acids.iter_mut().enumerate() {
            let kind = match mol.na_type {
                NucleicAcidType::Dna => FfMolType::Dna,
                NucleicAcidType::Rna => FfMolType::Rna,
            };
            changed |= self.force_fields.assign(
                (MolType::NucleicAcid as u8, i),
                &mut mol.common,
                kind,
                mol.residues.iter().map(|r| r.to_generic()).collect(),
                &self.ff_param_set,
                &mut self.mol_specific_params,
            );
        }
        self.force_fields.assignments.retain(|&(kind, i), _| {
            let count = if kind == MolType::Peptide as u8 {
                self.peptides.len()
            } else if kind == MolType::Ligand as u8 {
                self.ligands.len()
            } else if kind == MolType::Lipid as u8 {
                self.lipids.len()
            } else {
                self.nucleic_acids.len()
            };
            i < count
        });
        let assignments = &self.force_fields.assignments;
        self.force_fields.amber.retain(|&(category, topology), _| {
            assignments
                .iter()
                .any(|(&(kind, _), a)| kind == category && a.topology == topology)
        });
        changed
    }
}

/// Shared by the sidebar and the MD panel, so the two cannot diverge.
pub fn selector(state: &mut State, ui: &mut Ui, id: &str) {
    let mut family = state.force_fields.family;
    ComboBox::from_id_salt(id)
        .selected_text(family_name(family))
        .show_ui(ui, |ui| {
            for value in [
                ForceFieldFamily::Amber,
                ForceFieldFamily::Charmm36,
                ForceFieldFamily::Opls,
            ] {
                ui.selectable_value(&mut family, value, family_name(value));
            }
        })
        .response
        .on_hover_text(
            "Assign this force field's atom types and charges to open and newly loaded molecules. \
             Also updates MD defaults. Unsupported molecules are left unparameterized.",
        );
    if family != state.force_fields.family {
        if let Err(e) = state.select_force_field(family) {
            handle_err(&mut state.ui, format!("Unable to select force field: {e}"));
        }
        ui.request_repaint();
    }
    let errors: Vec<_> = state.force_fields.errors().collect();
    if !errors.is_empty() {
        ui.colored_label(Color32::YELLOW, format!("{} unparameterized", errors.len()))
            .on_hover_text(format!(
                "Across all open molecules, including those not selected for MD. \
                 Both force field selectors show the same count.\n\n{}",
                errors.join("\n\n")
            ));
        if state.force_fields.family == ForceFieldFamily::Opls
            && ui.small_button("Retry").on_hover_text(
                "Retry failed LigParGen requests. An internet connection is needed for uncached ligands."
            ).clicked()
        {
            state.force_fields.retry_opls();
        }
    }
    if state.force_fields.has_pending() {
        ui.spinner();
        ui.label(format!("{} parameterizing", state.force_fields.opls_jobs.len()))
            .on_hover_text("Generating OPLS ligand parameters with LigParGen; results are cached locally.");
        ui.request_repaint_after(std::time::Duration::from_millis(50));
    }
}
