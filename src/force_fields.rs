//! The selected force field for loaded molecules and MD. Structural preparation
//! still uses Amber templates; this layer assigns the selected family's atom
//! types and charges without changing coordinates, residue names or hydrogens.

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
};

use bio_files::ResidueGeneric;
use dynamics::{
    ExplicitParams, FfMolType, ForceFieldFamily, HydrogenConstraint, MdConfig, MolDynamics,
    Solvent,
    params::{FfParamSet, gfnff::GfnffSet},
};
use egui::{Color32, ComboBox, RichText, Ui};
use mol_defs::molecules::{MolType, common::MoleculeCommon, nucleic_acid::NucleicAcidType};

use crate::{
    external_tools::{Tool, find_executable, is_installed},
    md::MdBackend,
    state::State,
    ui::popup::external_tools::install_btn,
    util::handle_err,
};

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
    recalculate_charges: bool,
    opls_jobs: HashMap<u64, Receiver<Result<MolDynamics, String>>>,
    opls_results: HashMap<u64, Result<MolDynamics, String>>,
    gfnff_jobs: HashMap<u64, Receiver<Result<MolDynamics, String>>>,
    gfnff_results: HashMap<u64, Result<MolDynamics, String>>,
    /// The xTB last found by [`Self::sync_xtb`], to tell it from a path the user typed.
    resolved_xtb: Option<PathBuf>,
}

pub fn family_name(family: ForceFieldFamily) -> &'static str {
    match family {
        ForceFieldFamily::Amber => "Amber",
        ForceFieldFamily::Charmm36 => "CHARMM36m",
        ForceFieldFamily::Opls => "OPLS-AA/L",
        ForceFieldFamily::Gfnn => "GFN-FF",
        ForceFieldFamily::Custom => "Custom",
    }
}

/// Translate legacy solvent choices to ALPB and apply GFN-FF constraints.
pub fn configure_gfnff_md(cfg: &mut MdConfig, set: &mut GfnffSet) -> Result<(), String> {
    match &cfg.solvent {
        Solvent::None => (),
        Solvent::WaterOpc
        | Solvent::WaterOpcSpecifyMolCount(_)
        | Solvent::WaterOpcCustomRegions(_) => set.alpb_solvent = Some("water".into()),
        Solvent::OctanolWithWater => set.alpb_solvent = Some("woctanol".into()),
        Solvent::Custom(_) => {
            return Err("GFN-FF requires a named ALPB solvent. Select it in MD setup.".into());
        }
    }

    cfg.ff_family = ForceFieldFamily::Gfnn;
    cfg.solvent = Solvent::None;
    cfg.barostat_cfg = None;
    cfg.hydrogen_constraint = HydrogenConstraint::Flexible;
    cfg.overrides.skip_counterion_insertion = true;

    Ok(())
}

pub fn gfnff_solvent_selector(state: &mut State, ui: &mut Ui, id: &str) {
    let Some(set) = state
        .force_fields
        .params
        .as_mut()
        .and_then(|p| p.gfnff_mut())
    else {
        return;
    };
    let previous = set.alpb_solvent.clone();
    ui.label("Implicit solvent (ALPB):");
    ComboBox::from_id_salt(id)
        .selected_text(set.alpb_solvent.as_deref().unwrap_or("None (vacuum)"))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut set.alpb_solvent, None, "None (vacuum)");

            for (name, label) in [
                ("water", "Water"),
                ("methanol", "Methanol"),
                ("ethanol", "Ethanol"),
                ("acetone", "Acetone"),
                ("acetonitrile", "Acetonitrile"),
                ("dmso", "DMSO"),
                ("dmf", "DMF"),
                ("benzene", "Benzene"),
                ("toluene", "Toluene"),
                ("ether", "Ether"),
                ("thf", "THF"),
                ("hexane", "Hexane"),
                ("octanol", "Octanol"),
                ("woctanol", "Octanol saturated with water"),
            ] {
                ui.selectable_value(&mut set.alpb_solvent, Some(name.into()), label);
            }
        })
        .response
        .on_hover_text(
            "Solvent effects on energies and forces without generating solvent molecules.",
        );

    if set.alpb_solvent != previous {
        state.to_save.md.config.solvent = Solvent::None;
        state.force_fields.invalidate();
        state.ui.ff_params.gfnff_report = None;
        state.ui.ff_params.gfnff_error = None;
        state.ui.ff_params.gfnff_job = None;
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
    recompute: bool,
) -> Result<Vec<Annotation>, String> {
    let input = MolDynamics {
        ff_mol_type: kind,
        atoms: mol.atoms.iter().map(|a| a.to_generic()).collect(),
        bonds: mol.bonds.iter().map(|b| b.to_generic()).collect(),
        residues: (!residues.is_empty()).then(|| residues.to_vec()),
        mol_specific_params: specific.get(&mol.ident).cloned(),
        ..Default::default()
    };
    let prepared = set
        .parameterize_with_options(
            &input,
            &dynamics::params::ParameterizationOptions {
                reassign_atom_types: recompute,
                charge_policy: if recompute {
                    dynamics::params::ChargePolicy::Recompute
                } else {
                    dynamics::params::ChargePolicy::BackendDefault
                },
                ..dynamics::params::ParameterizationOptions::local_only()
            },
        )
        .map_err(|e| e.to_string())?;
    if let Some(parameters) = prepared.mol_specific_params {
        specific.insert(mol.ident.clone(), parameters);
    }
    Ok(prepared
        .atoms
        .into_iter()
        .map(|a| (a.force_field_type, a.partial_charge))
        .collect())
}

impl ForceFields {
    pub fn has_pending(&self) -> bool {
        !self.opls_jobs.is_empty() || !self.gfnff_jobs.is_empty()
    }

    /// Re-assign atom types and charges to all molecules on the next refresh. E.g. after editing
    /// the selected family's parameters.
    pub fn invalidate(&mut self) {
        self.assignments.clear();
        self.gfnff_jobs.clear();
        self.gfnff_results.clear();
        self.dirty = true;
        self.sync_xtb();
    }

    /// Point GFN-FF at the xTB from the Tools panel, or on PATH. A typed path stays.
    fn sync_xtb(&mut self) {
        let Some(set) = self.params.as_mut().and_then(|p| p.gfnff_mut()) else {
            return;
        };
        if set.executable.is_some() && set.executable != self.resolved_xtb {
            return;
        }

        self.resolved_xtb = find_executable(Tool::Xtb).ok();
        set.executable = self.resolved_xtb.clone();
    }

    fn poll_gfnff(&mut self) {
        let mut completed = Vec::new();
        for (&topology, rx) in &self.gfnff_jobs {
            match rx.try_recv() {
                Ok(result) => completed.push((topology, result)),
                Err(TryRecvError::Disconnected) => {
                    completed.push((topology, Err("GFN-FF worker stopped".into())))
                }
                Err(TryRecvError::Empty) => (),
            }
        }
        for (topology, result) in completed {
            self.gfnff_jobs.remove(&topology);
            self.gfnff_results.insert(topology, result);
            self.assignments.retain(|_, a| {
                a.family != ForceFieldFamily::Gfnn || (a.topology != topology && !a.pending)
            });
            self.dirty = true;
        }
    }

    fn poll_opls(&mut self) {
        let mut completed = Vec::new();
        for (&topology, rx) in &self.opls_jobs {
            match rx.try_recv() {
                Ok(result) => completed.push((topology, result)),
                Err(TryRecvError::Disconnected) => completed.push((
                    topology,
                    Err("OPLS parameter generation worker stopped unexpectedly".into()),
                )),
                Err(TryRecvError::Empty) => (),
            }
        }
        for (topology, result) in completed {
            self.opls_jobs.remove(&topology);
            self.opls_results.insert(topology, result);
            self.assignments
                .retain(|_, a| a.topology != topology || a.family != ForceFieldFamily::Opls);
            self.dirty = true;
        }
    }

    fn retry_opls(&mut self) {
        self.opls_results.retain(|_, result| result.is_ok());
        self.assignments
            .retain(|_, a| a.family != ForceFieldFamily::Opls || a.error.is_none());
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
        } else if self.family != ForceFieldFamily::Amber {
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
        recalculate_charges: bool,
    ) -> bool {
        let topology = topology_hash(mol, &residues);
        let current = annotation_hash(mol);
        let previous = self
            .assignments
            .get(&key)
            .filter(|a| a.topology == topology);
        if !recalculate_charges
            && previous.is_some_and(|a| a.family == self.family && a.annotations == current)
        {
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
                        .amber()
                        .unwrap()
                        .small_mol
                        .as_ref()
                        .is_some_and(|p| p.mass.contains_key(t))
                        && !specific_params.is_some_and(|p| p.mass.contains_key(t))
                })
            });

        // Keep Amber annotations for Amber-only analysis copies.
        let amber = if self.family == ForceFieldFamily::Amber
            && (imported_foreign_types
                || (recalculate_charges
                    && matches!(kind, FfMolType::SmallOrganic | FfMolType::Peptide)))
        {
            amber_annotations(
                mol,
                kind,
                &residues,
                amber_data,
                specific,
                recalculate_charges,
            )
            .unwrap_or_else(|_| vec![(None, None); mol.atoms.len()])
        } else if let Some(source) = self.amber.get(&(key.0, topology)) {
            source.clone()
        } else if self.family == ForceFieldFamily::Amber {
            annotations(mol)
        } else {
            vec![(None, None); mol.atoms.len()]
        };

        let mut explicit = None;
        let mut atom_serials = Vec::new();
        let mut error = None;
        let mut pending = false;
        if self.family == ForceFieldFamily::Amber {
            // Apply only Amber assignments.
            if recalculate_charges || restoring_amber || imported_foreign_types {
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
                    .map(|a| {
                        let mut atom = a.to_generic();
                        atom.force_field_type = None;
                        atom.partial_charge = None;

                        atom
                    })
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
            let result = if self.family == ForceFieldFamily::Gfnn {
                if let Some(result) = self.gfnff_results.get(&topology) {
                    Some(result.clone())
                } else if self.gfnff_jobs.contains_key(&topology) || self.gfnff_jobs.len() >= 2 {
                    None
                } else {
                    let mut backend = self.params.as_ref().unwrap().gfnff().unwrap().clone();
                    // The override is for the combined simulation, not each displayed molecule.
                    backend.total_charge = None;
                    let (tx, rx) = mpsc::channel();
                    self.gfnff_jobs.insert(topology, rx);
                    thread::spawn(move || {
                        use dynamics::params::potential::SystemPotentialBackend;

                        let _ = tx.send(backend.prepare(&input).map_err(|e| e.descrip));
                    });
                    None
                }
            } else if self.family == ForceFieldFamily::Opls && kind == FfMolType::SmallOrganic {
                if let Some(result) = self.opls_results.get(&topology) {
                    if let Ok(prepared) = result {
                        self.params
                            .as_ref()
                            .unwrap()
                            .opls()
                            .unwrap()
                            .remember_small_molecule(prepared);
                    }
                    Some(result.clone())
                } else if self.opls_jobs.contains_key(&topology) {
                    None
                } else {
                    let local = self.params.as_ref().unwrap().parameterize_with_options(
                        &input,
                        &dynamics::params::ParameterizationOptions::local_only(),
                    );
                    match local {
                        Ok(prepared) => Some(Ok(prepared)),
                        Err(_) => {
                            let (tx, rx) = mpsc::channel();
                            self.opls_jobs.insert(topology, rx);
                            thread::spawn(move || {
                                let result =
                                    FfParamSet::new_opls().map_err(|e| e.to_string()).and_then(
                                        |set| set.parameterize(&input).map_err(|e| e.descrip),
                                    );
                                let _ = tx.send(result);
                            });
                            None
                        }
                    }
                }
            } else {
                Some(
                    self.params
                        .as_ref()
                        .unwrap()
                        .parameterize(&input)
                        .map_err(|e| e.descrip),
                )
            };
            // Pending and failed assignments must not display another model's charges.
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
        self.amber.insert((key.0, topology), amber);
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
    pub fn sync_gfnff_md_config(&mut self) -> Result<(), String> {
        if self.force_fields.family != ForceFieldFamily::Gfnn {
            return Ok(());
        }

        let set = self
            .force_fields
            .params
            .as_mut()
            .and_then(|p| p.gfnff_mut())
            .ok_or("GFN-FF parameters are not loaded")?;
        let previous = set.alpb_solvent.clone();
        configure_gfnff_md(&mut self.to_save.md.config, set)?;

        if set.alpb_solvent != previous {
            self.force_fields.invalidate();
            self.ui.ff_params.gfnff_report = None;
            self.ui.ff_params.gfnff_error = None;
            self.ui.ff_params.gfnff_job = None;
        }

        if self.to_save.md.dt > 0.0005 {
            self.to_save.md.dt = 0.0005;
            self.ui.md.dt_input = self.to_save.md.dt.to_string();
            self.volatile.md_local.run_time = self.to_save.md.num_steps as f32 * self.to_save.md.dt;
        }
        if !matches!(
            self.to_save.md.backend,
            crate::md::MdBackend::Dynamics | crate::md::MdBackend::xTb
        ) {
            self.to_save.md.backend = crate::md::MdBackend::Dynamics;
        }

        Ok(())
    }

    /// Avoid starting synchronous network work from MD/energy/export UI paths.
    /// Only molecules actually selected for this simulation can block it.
    pub fn check_md_force_fields(&self) -> Result<(), String> {
        if matches!(
            self.force_fields.family,
            ForceFieldFamily::Amber | ForceFieldFamily::Gfnn
        ) {
            return Ok(());
        }
        let molecules = self
            .peptides
            .iter()
            .enumerate()
            .map(|(i, m)| (MolType::Peptide, i, &m.common))
            .chain(
                self.ligands
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (MolType::Ligand, i, &m.common)),
            )
            .chain(
                self.lipids
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (MolType::Lipid, i, &m.common)),
            )
            .chain(
                self.nucleic_acids
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (MolType::NucleicAcid, i, &m.common)),
            );

        for (kind, i, mol) in molecules {
            if mol.selected_for_md.is_none() {
                continue;
            }
            let assignment = self
                .force_fields
                .assignments
                .get(&(kind as u8, i))
                .filter(|a| a.family == self.force_fields.family)
                .ok_or_else(|| format!("{}: force field assignment is not ready yet", mol.ident))?;
            if assignment.pending {
                return Err(format!(
                    "{}: force field assignment is still running",
                    mol.ident
                ));
            }
            if let Some(error) = &assignment.error {
                return Err(error.clone());
            }
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
        let mut params = if family == ForceFieldFamily::Amber {
            None
        } else {
            Some(FfParamSet::from_family(family).map_err(|e| e.to_string())?)
        };
        if let Some(set) = params.as_mut().and_then(|p| p.gfnff_mut()) {
            configure_gfnff_md(&mut self.to_save.md.config, set)?;
        }

        self.force_fields.params = params;
        self.force_fields.sync_xtb();
        self.ui.ff_params.gfnff_executable = None;
        self.ui.ff_params.gfnff_library = None;
        let previous_family = self.force_fields.family;
        self.force_fields.family = family;
        self.force_fields.recalculate_charges = true;
        self.force_fields.opls_jobs.clear();
        self.force_fields.opls_results.clear();
        self.force_fields.invalidate();

        let preset = MdConfig::for_family(family);
        let cfg = &mut self.to_save.md.config;
        cfg.ff_family = family;
        cfg.coulomb_cutoff = preset.coulomb_cutoff;
        cfg.lj_cutoff = preset.lj_cutoff;
        cfg.lj_modifier = preset.lj_modifier;
        cfg.water_model = None;
        if family == ForceFieldFamily::Gfnn {
            cfg.solvent = preset.solvent;
            cfg.barostat_cfg = None;
            cfg.hydrogen_constraint = preset.hydrogen_constraint;
            cfg.overrides.skip_counterion_insertion = true;
            self.to_save.md.dt = self.to_save.md.dt.min(0.0005);
            self.to_save.md.backend = MdBackend::xTb;
        } else if previous_family == ForceFieldFamily::Gfnn {
            cfg.overrides.skip_counterion_insertion = false;
            if self.to_save.md.backend == crate::md::MdBackend::xTb {
                self.to_save.md.backend = crate::md::MdBackend::Dynamics;
            }
        }
        self.ui.md.sync(cfg, self.to_save.md.dt);

        self.volatile.md_local.running = false;
        self.volatile.thread_receivers.xtb_md = None;
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
        self.force_fields.poll_gfnff();
        let recalculate_charges = std::mem::take(&mut self.force_fields.recalculate_charges);
        let mut changed = std::mem::take(&mut self.force_fields.dirty);
        for (i, mol) in self.peptides.iter_mut().enumerate() {
            changed |= self.force_fields.assign(
                (MolType::Peptide as u8, i),
                &mut mol.common,
                FfMolType::Peptide,
                mol.residues.iter().map(|r| r.to_generic()).collect(),
                &self.ff_param_set,
                &mut self.mol_specific_params,
                recalculate_charges,
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
                recalculate_charges,
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
                recalculate_charges,
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
                recalculate_charges,
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

/// Whether GFN-FF has an xTB executable: installed, or a path typed into the parameters window.
fn xtb_available(state: &State) -> bool {
    let set_path = state
        .force_fields
        .params
        .as_ref()
        .and_then(|p| p.gfnff())
        .and_then(|set| set.executable.as_ref());

    set_path.is_some_and(|path| path.is_file()) || is_installed(Tool::Xtb)
}

/// Shared by the sidebar and the MD panel, so the two cannot diverge. `compact` shows only the
/// combobox, as the rest is too wide for the sidebar.
pub fn selector(state: &mut State, ui: &mut Ui, id: &str, compact: bool) {
    let mut family = state.force_fields.family;
    ComboBox::from_id_salt(id)
        .selected_text(family_name(family))
        .show_ui(ui, |ui| {
            for value in [
                ForceFieldFamily::Amber,
                // Commented out: Charmm currently requires manually-loaded force fields, and I haven't tested it. The
                // others work in a simple manner.
                // ForceFieldFamily::Charmm36,
                ForceFieldFamily::Opls,
                ForceFieldFamily::Gfnn,
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

    if compact {
        // Keep polling background jobs, as the progress indicators below do.
        if !state.force_fields.gfnff_jobs.is_empty() || !state.force_fields.opls_jobs.is_empty() {
            ui.request_repaint_after(std::time::Duration::from_millis(50));
        }

        return;
    }

    let needs_xtb = state.force_fields.family == ForceFieldFamily::Gfnn
        || state.to_save.md.backend == MdBackend::xTb;
    if needs_xtb && !xtb_available(state) {
        install_btn(state, Tool::Xtb, ui);
    }

    let errors: Vec<_> = state.force_fields.errors().collect();
    let has_errors = !errors.is_empty();
    if !errors.is_empty() {
        ui.menu_button(
            RichText::new(format!("{} unparameterized", errors.len())).color(Color32::YELLOW),
            |ui| {
                ui.set_max_width(440.);
                ui.strong("Force field assignment failed");
                ui.label("This counts all open molecules, including those not selected for MD.");
                for error in &errors {
                    ui.separator();
                    ui.add(egui::Label::new(*error).wrap().selectable(true));
                }
            },
        )
        .response
        .on_hover_text(format!(
            "Click for parameterization errors. Both force field selectors show \
             the same count across all open molecules.\n\n{}",
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
    if !state.force_fields.gfnff_jobs.is_empty() {
        ui.spinner();
        ui.label("Preparing GFN-FF…");
        ui.request_repaint_after(std::time::Duration::from_millis(50));
    }
    if state.force_fields.family == ForceFieldFamily::Gfnn
        && has_errors
        && ui.small_button("Retry GFN-FF").clicked()
    {
        state.force_fields.invalidate();
    }
    if !state.force_fields.opls_jobs.is_empty() {
        ui.spinner();
        ui.label(format!(
            "{} queued / parameterizing",
            state.force_fields.opls_jobs.len()
        ))
        .on_hover_text(
            "Generating OPLS ligand parameters with LigParGen and caching results locally. \
             The public service allows five submissions per ten minutes; further requests \
             wait automatically in the background. Cached molecules are available immediately.",
        );
        ui.request_repaint_after(std::time::Duration::from_millis(50));
    }
}
