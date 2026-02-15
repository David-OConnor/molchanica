//! Fundamental data structures for small organic molecules / ligands

use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::{mpsc, mpsc::Receiver},
    thread,
};

use bio_apis::{
    ReqError, amber_geostd,
    amber_geostd::GeostdData,
    pubchem,
    pubchem::{ProteinStructure, StructureSearchNamespace},
};
use bio_files::{
    ChargeType, Mol2, MolType, Pdbqt, PharmacophoreFeatureGeneric, Sdf, Xyz, create_bonds,
    md_params::{ForceFieldParams, ForceFieldParamsVec},
};
use dynamics::{
    param_inference::{AmberDefSet, assign_missing_params, find_ff_types},
    partial_charge_inference::infer_charge,
};
use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::{
    mol_characterization::MolCharacterization,
    molecules::{
        Atom, Bond, Chain, MolGenericRef, MolGenericTrait, MolIdent, Residue,
        common::MoleculeCommon,
    },
    therapeutic::{
        DatasetTdc, TherapeuticProperties,
        infer::Infer,
        pharmacophore::{Pharmacophore, PharmacophoreFeature},
    },
};

const LIGAND_ABS_POSIT_OFFSET: f64 = 15.; // Å

/// A molecule representing a small organic molecule. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeSmall {
    pub common: MoleculeCommon,
    // pub lig_data: Option<Ligand>,
    pub idents: Vec<MolIdent>,
    /// FF type and partial charge on all atoms. Quick lookup flag.
    pub ff_params_loaded: bool,
    /// E.g., overrides for dihedral angles (part of the *bonded* dynamics calculation) for this
    /// specific molecule, as provided by Amber. Quick lookup flag.
    pub frcmod_loaded: bool,
    /// E.g. loaded proteins from Pubchem.
    pub associated_structures: Vec<ProteinStructure>,
    // /// Simplified Molecular Input Line Entry System
    // /// A cache for display as required. This is a text representation of a molecular formula.
    // pub smiles: Option<String>,
    pub characterization: Option<MolCharacterization>,
    pub pharmacophore: Pharmacophore,
    pub therapeutic_props: Option<TherapeuticProperties>,
}

impl MoleculeSmall {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let mut idents = Vec::new();

        if let Some(id) = metadata.get("PUBCHEM_COMPOUND_CID") {
            if let Ok(cid) = id.parse::<u32>() {
                idents.push(MolIdent::PubChem(cid));
            };
        }

        if let Some(db_name) = metadata.get("DATABASE_NAME")
            && db_name.to_lowercase() == "drugbank"
        {
            if let Some(id) = metadata.get("DATABASE_ID") {
                idents.push(MolIdent::DrugBank(id.clone()));
            }
            // This seems to be valid for Drugbank-sourced molecules.
            if let Ok(id) = ident.parse::<u32>() {
                idents.push(MolIdent::PubChem(id));
            }
        }

        if ident.len() <= 4 && ident.parse::<u32>().is_err() {
            // This is a guess
            idents.push(MolIdent::PdbeAmber(ident.clone()));
        }

        let common = MoleculeCommon::new(ident, atoms, bonds, metadata, path);

        Self {
            common,
            idents,
            ..Default::default()
        }
    }

    pub fn update_characterization(&mut self) {
        self.characterization = Some(MolCharacterization::new(&self.common))
    }
}

impl MolGenericTrait for MoleculeSmall {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::Small(self)
    }

    fn mol_type(&self) -> crate::molecules::MolType {
        crate::molecules::MolType::Ligand
    }
}

impl TryFrom<Mol2> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Mol2) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Note: We don't compute bonds here; we assume they're included in the molecule format.
        // Handle path after; not supported by TryFrom.

        let mut result = Self::new(m.ident, atoms, bonds, m.metadata.clone(), None);
        result.pharmacophore = pharmacophore_from_biofiles(
            &m.pharmacophore_features,
            &result.common.atoms,
            &result.common.ident,
        )?;

        Ok(result)
    }
}

impl TryFrom<Sdf> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Sdf) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();
        // let mut residues = Vec::with_capacity(m.residues.len());
        // for res in &m.residues {
        //     residues.push(Residue::from_generic(res, &atoms, ResidueEnd::Hetero)?);
        // }

        // let mut chains = Vec::with_capacity(m.chains.len());
        // for c in &m.chains {
        //     chains.push(Chain::from_generic(c, &atoms, &residues)?);
        // }

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Handle path and state-specific items after; not supported by TryFrom.
        let mut result = Self::new(m.ident, atoms, bonds, m.metadata.clone(), None);

        result.pharmacophore = pharmacophore_from_biofiles(
            &m.pharmacophore_features,
            &result.common.atoms,
            &result.common.ident,
        )?;

        Ok(result)
    }
}

impl MoleculeSmall {
    pub fn from_xyz(m: Xyz, path: &Path) -> io::Result<Self> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds_gen = create_bonds(&m.atoms);
        let bonds: Vec<Bond> = bonds_gen
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert(String::from("Comment"), m.comment.clone());

        // Handle path and state-specific items after; not supported by TryFrom.
        Ok(Self::new(
            filename,
            atoms,
            bonds,
            metadata,
            Some(path.to_owned()),
        ))
    }
}

impl TryFrom<Pdbqt> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Pdbqt) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();
        let mut residues = Vec::with_capacity(m.residues.len());
        for res in &m.residues {
            residues.push(Residue::from_generic(res, &atoms)?);
        }

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Handle path after; not supported by TryFrom.
        Ok(Self::new(
            m.ident,
            atoms,
            bonds,
            HashMap::new(), // todo: Metadata?
            None,
        ))
    }
}

impl MoleculeSmall {
    pub fn to_mol2(&self) -> Mol2 {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        Mol2 {
            ident: self.common.ident.clone(),
            atoms,
            bonds,
            metadata: self.common.metadata.clone(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            pharmacophore_features: pharmacophore_to_biofiles(&self.pharmacophore)
                .unwrap_or_default(),
            comment: None,
        }
    }

    pub fn to_sdf(&self) -> Sdf {
        // SDF doesn't support explicit atom SNs; they use order. This reassignment makes sure
        // the bond atom assignments aren't lost in this process.
        let (atoms, bonds) = {
            let mut common_reassigned = self.common.clone();
            common_reassigned.reassign_sns();

            let a = common_reassigned
                .atoms
                .iter()
                .map(|a| a.to_generic())
                .collect();
            let b = common_reassigned
                .bonds
                .iter()
                .map(|b| b.to_generic())
                .collect();

            (a, b)
        };

        let mut metadata = self.common.metadata.clone();

        // Note: These may be redundant with metadata already loaded.

        for ident in &self.idents {
            match ident {
                MolIdent::PubChem(cid) => {
                    metadata.insert("PUBCHEM_COMPOUND_CID".to_string(), cid.to_string());
                }
                MolIdent::DrugBank(id) => {
                    metadata.insert("DATABASE_ID".to_string(), id.clone());
                    metadata.insert("DATABASE_NAME".to_string(), "drugbank".to_string());
                }
                _ => (),
            }
        }

        Sdf {
            ident: self.common.ident.clone(),
            metadata,
            atoms,
            bonds,
            chains: Vec::new(),
            residues: Vec::new(),
            pharmacophore_features: pharmacophore_to_biofiles(&self.pharmacophore)
                .unwrap_or_default(),
        }
    }

    pub fn to_xyz(&self) -> Xyz {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();

        let comment = match self.common.metadata.get("Comment") {
            Some(v) => v.to_owned(),
            None => String::new(),
        };

        Xyz { atoms, comment }
    }

    pub fn to_pdbqt(&self) -> Pdbqt {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        Pdbqt {
            ident: self.common.ident.clone(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            comment: None,
            atoms,
            bonds,
            chains: Vec::new(),
            residues: Vec::new(),
        }
    }
}

impl MoleculeSmall {
    /// For example, this can be used to create a ligand from a residue that was loaded with a mmCIF
    /// file from RCSB. It can then be used for docking, or saving to a Mol2 or SDF file.
    ///
    /// `atoms` here should be the full set, as indexed by `res`, unless `use_sns` is true.
    /// `use_sns` = false is faster.
    ///
    /// We assume the residue is already populated with hydrogens.
    ///
    /// We reposition its atoms to be around the origin.
    pub fn from_res(res: &Residue, atoms: &[Atom], bonds: &[Bond]) -> Self {
        let mut atoms_this = Vec::with_capacity(res.atoms.len());

        // We use this map when rebuilding bonds.
        // Old index: (new index, new sn)
        let mut bond_map = HashMap::new();

        for (i, &atom_i_orig) in res.atoms.iter().enumerate() {
            let atom = &atoms[atom_i_orig];

            let serial_number = i as u32 + 1;
            bond_map.insert(atom_i_orig, (i, serial_number));

            atoms_this.push(Atom {
                serial_number,
                residue: None,
                chain: None,
                ..atom.clone()
            });
        }

        let atom_orig_i: Vec<_> = bond_map.keys().collect();
        let mut bonds_this: Vec<_> = bonds
            .iter()
            .filter(|b| atom_orig_i.contains(&&b.atom_0) && atom_orig_i.contains(&&b.atom_1))
            .cloned()
            .collect();

        let mut bonds_new = Vec::with_capacity(bonds_this.len());
        for bond in &bonds_this {
            let (atom_0, atom_0_sn) = bond_map.get(&bond.atom_0).unwrap();
            let (atom_1, atom_1_sn) = bond_map.get(&bond.atom_1).unwrap();

            bonds_new.push(Bond {
                bond_type: bond.bond_type,
                atom_0_sn: *atom_0_sn,
                atom_1_sn: *atom_1_sn,
                atom_0: *atom_0,
                atom_1: *atom_1,
                is_backbone: false,
            })
        }

        let name = res.res_type.to_string();
        let mut result = Self::new(name.clone(), atoms_this, bonds_new, HashMap::new(), None);

        result.common.center_local_posits_around_origin();

        result.idents.push(MolIdent::PdbeAmber(name));

        result
    }

    pub fn apply_geostd_data(
        &mut self,
        data: GeostdData,
        lig_specific: &mut HashMap<String, ForceFieldParams>,
    ) {
        if !self.ff_params_loaded {
            let Ok(mol2) = Mol2::new(&data.mol2) else {
                eprintln!("Error: No Mol2 available from Geostd");
                return;
            };

            let mut count_c_orig: u32 = 0;
            let mut count_n_orig: u32 = 0;
            let mut count_o_orig: u32 = 0;
            let mut count_h_orig: u32 = 0;
            //
            let mut count_c_amber: u32 = 0;
            let mut count_n_amber: u32 = 0;
            let mut count_o_amber: u32 = 0;
            let mut count_h_amber: u32 = 0;

            for atom in &self.common.atoms {
                match atom.element {
                    Element::Carbon => count_c_orig += 1,
                    Element::Nitrogen => count_n_orig += 1,
                    Element::Oxygen => count_o_orig += 1,
                    Element::Hydrogen => count_h_orig += 1,
                    _ => {}
                }
            }
            for atom in &mol2.atoms {
                match atom.element {
                    Element::Carbon => count_c_amber += 1,
                    Element::Nitrogen => count_n_amber += 1,
                    Element::Oxygen => count_o_amber += 1,
                    Element::Hydrogen => count_h_amber += 1,
                    _ => {}
                }
            }

            if count_c_orig != count_c_amber
                || count_n_orig != count_n_amber
                || count_o_orig != count_o_amber
                || count_h_orig != count_h_amber
            {
                eprintln!(
                    "Unable to load Amber Geostd data for this molecule; atom count mismatch."
                );
                return;
            }

            let mol: Self = match mol2.try_into() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Problem loading Mol2 from geostd: {e}");
                    return; // OK only if this fn returns ()
                }
            };

            self.common.atoms = mol.common.atoms;
            self.common.bonds = mol.common.bonds;
            self.common.atom_posits = mol.common.atom_posits;
            self.common.adjacency_list = mol.common.adjacency_list;

            self.ff_params_loaded = true;
            println!("Loaded Amber Geostd FF data for {}", self.common.ident);
        }

        if !self.frcmod_loaded
            && let Some(f) = data.frcmod
            && let Ok(frcmod) = ForceFieldParamsVec::from_frcmod(&f)
        {
            lig_specific.insert(self.common.ident.clone(), ForceFieldParams::new(&frcmod));
            self.frcmod_loaded = true;

            println!("Loaded Amber FRCMOD data for {}", self.common.ident);
        }
    }

    /// Attempt to find FF type, partial charge, and FRCMOD overrides for a given molecule.
    /// Launch this in a thread.
    ///
    /// Unfortunately, we can't directly map atoms from our original molecule to
    /// the Geostd one. We could do this with coordinates, but that might be complicated.
    /// For now, we perform a sanity check about atom count by element. If it passes,
    /// we replace molecule atom and bond data with that loaded from the mol2.
    fn _search_geostd(
        &mut self,
        ident: &str,
        geostd_thread: &mut Option<Receiver<(usize, Result<GeostdData, ReqError>)>>,
        mol_i: usize,
    ) {
        println!("Attempting to load Amber Geostd dynamics data for this molecule...");

        let (tx, rx) = mpsc::channel(); // one-shot channel
        let ident_for_thread = ident.to_string();

        thread::spawn(move || {
            let data = amber_geostd::load_mol_files(&ident_for_thread);
            let _ = tx.send((mol_i, data));
        });

        *geostd_thread = Some(rx);
    }

    pub fn update_aux(
        &mut self,
        pubchem_properties_map: &HashMap<MolIdent, pubchem::Properties>,
        pubchem_properties_avail: &mut Option<
            Receiver<(MolIdent, Result<pubchem::Properties, ReqError>)>,
        >,
        models: &mut HashMap<DatasetTdc, Infer>,
        ff_params: &ForceFieldParams,
        therapeutic_properties_avail: &mut Option<Receiver<(usize, TherapeuticProperties)>>,
        mol_i: usize,
    ) {
        self.update_characterization();

        // Load PubChem properties from either our prefs file, or online. If online,
        // launch this in a separate thread.
        let mut pubchem_ident_exists = false;

        for ident in &self.idents {
            match pubchem_properties_map.get(ident) {
                Some(props) => {
                    println!("Loaded Properties for {ident:?} from our local DB.");

                    self.update_idents_and_char_from_pubchem(props);
                    break;
                }
                None => {
                    let (tx, rx) = mpsc::channel(); // one-shot channel
                    let ident_for_thread = ident.clone();

                    match ident {
                        MolIdent::PubChem(_) => {
                            println!("\nLoading PubChem properties for {ident:?} over HTTP...");

                            thread::spawn(move || {
                                // Part of our borrow-checker workaround
                                let cid: u32 = ident_for_thread.ident_innner().parse().unwrap();
                                let data = pubchem::properties(
                                    StructureSearchNamespace::Cid,
                                    &cid.to_string(),
                                );

                                let _ = tx.send((ident_for_thread, data));
                            });

                            pubchem_ident_exists = true;
                            *pubchem_properties_avail = Some(rx);
                            break;
                        }
                        _ => (),
                    }
                }
            }
            break;
        }

        // If we don't have a PubChemID, load SMILES, then get a PubChem ID.
        if !pubchem_ident_exists {
            for ident in &self.idents {
                let (tx, rx) = mpsc::channel(); // one-shot channel
                let ident_for_thread = ident.clone();

                if let MolIdent::PdbeAmber(_) = ident {
                    println!("\nLoading PubChem properties for {ident:?} over HTTP...");
                    thread::spawn(move || {
                        let data =
                            pubchem::properties_from_pdbe_id(&ident_for_thread.ident_innner());

                        let _ = tx.send((ident_for_thread, data));
                    });

                    *pubchem_properties_avail = Some(rx);
                    break;
                }
            }
        }

        // todo: We may wish to run this after updating params from PubChem, but this is fine for now,
        // todo, or in general if you get everything you need Hi-fi from calculations.

        let (tx, rx) = mpsc::channel();
        let mol_for_thread = self.clone();
        let ff_params_for_thread = ff_params.clone();
        let mut models_for_thread = std::mem::take(models);

        thread::spawn(move || {
            match TherapeuticProperties::new(
                &mol_for_thread,
                &mut models_for_thread,
                &ff_params_for_thread,
            ) {
                Ok(tp) => {
                    let _ = tx.send((mol_i, tp));
                }
                Err(e) => eprintln!("Error loading therapeutic properties: {e}"),
            }
        });

        *therapeutic_properties_avail = Some(rx);
    }

    pub fn update_idents_and_char_from_pubchem(&mut self, props: &pubchem::Properties) {
        let mut smiles_exists = false;
        let mut inchi_exists = false;
        let mut inchi_key_exists = false;
        let mut iupac_name_exists = false;
        let mut title_exists = false;

        for ident in &self.idents {
            if matches!(ident, MolIdent::Smiles(_)) {
                smiles_exists = true;
            }
            if matches!(ident, MolIdent::InchI(_)) {
                inchi_exists = true;
            }
            if matches!(ident, MolIdent::InchIKey(_)) {
                inchi_key_exists = true;
            }
            if matches!(ident, MolIdent::IupacName(_)) {
                iupac_name_exists = true;
            }
            if matches!(ident, MolIdent::PubchemTitle(_)) {
                title_exists = true;
            }
        }

        if !smiles_exists {
            self.idents.push(MolIdent::Smiles(props.smiles.clone()));
        }
        if !inchi_exists {
            self.idents.push(MolIdent::InchI(props.inchi.clone()));
        }
        if !inchi_key_exists {
            self.idents
                .push(MolIdent::InchIKey(props.inchi_key.clone()));
        }
        if !iupac_name_exists {
            self.idents
                .push(MolIdent::IupacName(props.iupac_name.clone()));
        }
        if !title_exists {
            self.idents
                .push(MolIdent::PubchemTitle(props.title.clone()));
        }

        if let Some(char) = &mut self.characterization {
            println!(
                "LogP Calc:{:.1} | PubChem: {:.2} TPSA calc: {:.1} PubChem: {:.2}\n",
                char.log_p, props.log_p, char.tpsa_ertl, props.total_polar_surface_area
            );

            char.log_p_pubchem = Some(props.log_p);
            char.tpsa_ertl = props.total_polar_surface_area;
            char.volume_pubchem = Some(props.volume);
            char.complexity = Some(props.complexity);
        }
    }

    /// Update partial charges, FF types, and mol-specific params.
    /// Note: Perhaps we restructure? Not all of these need access to state.
    ///
    /// We currently skip mol-specific params for ML training, where we need FF type
    /// and partial charge, but not them.
    pub fn update_ff_related(
        &mut self,
        mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
        gaff2: &ForceFieldParams,
        skip_mol_specific: bool,
    ) {
        self.ff_params_loaded = true;
        for atom in &self.common.atoms {
            if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                self.ff_params_loaded = false;
                break;
            }
        }

        if mol_specific_param_set
            .keys()
            .any(|k| k.eq_ignore_ascii_case(&self.common.ident))
        {
            self.frcmod_loaded = true;
        }

        // println!("Inferring FF parameter data...");
        // Note: There is an all-in-one `update_small_mol_params` fn we can use as well; it's
        // easier to use nominally, but this approach works better for our this-project Atom and bond types,
        // and loaded flags.

        let mut atoms_gen: Vec<_> = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = self.common.bonds.iter().map(|a| a.to_generic()).collect();

        if !self.ff_params_loaded {
            let defs = AmberDefSet::new().unwrap();
            let ff_types = find_ff_types(&atoms_gen, &bonds_gen, &defs);

            for (i, atom) in self.common.atoms.iter_mut().enumerate() {
                atom.force_field_type = Some(ff_types[i].clone());

                // We re-use `atoms_gen` for mol specific params below; update atoms gen here.
                atoms_gen[i].force_field_type = Some(ff_types[i].clone());
            }

            let charge = match infer_charge(&atoms_gen, &bonds_gen) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error inferring params: {e:?}");
                    return;
                }
            };

            for (i, atom) in self.common.atoms.iter_mut().enumerate() {
                atom.partial_charge = Some(charge[i]);
            }

            // // todo: This print and loop are temp.
            // println!("\n FF types computed:");
            // for atom in &self.common.atoms {
            //     println!(
            //         "--{}: {} {:.4}",
            //         atom.serial_number,
            //         atom.force_field_type.as_ref().unwrap(),
            //         atom.partial_charge.unwrap()
            //     );
            // }

            self.ff_params_loaded = true;
        }

        if !self.frcmod_loaded && !skip_mol_specific {
            let mol_specific_params =
                match assign_missing_params(&atoms_gen, &self.common.adjacency_list, gaff2) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "Error inferring params for mol {}: {e:?}",
                            self.common.ident
                        );
                        return;
                    }
                };

            // println!("\n\nDihe FRCMOD created:");
            // for p in &mol_specific_params.dihedral {
            //     println!("\nDihe: {:?}", p);
            // }

            // println!("\n\nImproper FRCMOD created:");
            // for p in &mol_specific_params.improper {
            //     println!("Improp: {:?}", p);
            // }

            mol_specific_param_set.insert(self.common.ident.to_owned(), mol_specific_params);
            self.frcmod_loaded = true;
        }
        // println!("Inference complete.");
    }
}

/// Convert the bio_files SDF-based Pharmacophore layout to our own.
fn pharmacophore_from_biofiles(
    feats: &[PharmacophoreFeatureGeneric],
    atoms: &[Atom],
    ident: &str,
) -> io::Result<Pharmacophore> {
    let def = PharmacophoreFeature::default(); // For default vals.

    let mut features = Vec::with_capacity(feats.len());

    for feat in feats {
        // Average position, if multiple atoms.
        let mut posit = Vec3::new_zero();
        let mut atom_i = Vec::with_capacity(feat.atom_sns.len());

        for a in feat.atom_sns.iter() {
            let i = *a as usize - 1;
            if i >= atoms.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Pharmacophore index out of bounds",
                ));
            }

            posit += atoms[i].posit;
            atom_i.push(i);
        }
        posit /= feat.atom_sns.len() as f64;

        let atom_i = if !atom_i.is_empty() {
            Some(atom_i)
        } else {
            None
        };

        features.push(PharmacophoreFeature {
            feature_type: feat.type_.clone().into(),
            posit,
            atom_i,
            ..def.clone()
        });
    }

    Ok(Pharmacophore {
        name: ident.to_string(),
        features,
        ..Default::default()
    })
}

fn pharmacophore_to_biofiles(ph: &Pharmacophore) -> io::Result<Vec<PharmacophoreFeatureGeneric>> {
    let mut result = Vec::new();

    for feat in &ph.features {
        let Some(atom_i) = &feat.atom_i else {
            eprintln!("Pharmacophore feature missing atom index");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Pharmacophore feature missing atom index",
            ));
        };

        let atom_sns = atom_i.iter().map(|i| *i as u32 + 1).collect();
        result.push(PharmacophoreFeatureGeneric {
            atom_sns,
            type_: feat.feature_type.to_generic(),
        });
    }

    Ok(result)
}
