use bio_files::BondType;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::Vec3;
use mol_defs::{
    mol_components::MolComponents,
    molecules::{
        Bond,
        common::{MoleculeCommon, bonds_avail},
    },
};
// Shared with the rest of the app, and used unchanged here. Re-exported so callers in the editor
// don't need to know which crate they live in, and so there's a single copy of the tables.
pub use mol_defs::molecules::common::{hydrogens_avail, remove_hydrogens};
use na_seq::{Element, Element::Hydrogen};

use crate::{
    mol_editor,
    mol_editor::{MolEditorState, templates::Template},
    mol_manip::ManipMode,
    state::StateUi,
};

/// A button that adds atoms to the editor molecule from a template. Things can be single atoms, but we are
/// currently using it for rings, functional groups, etc.
pub fn add_from_template(
    mol: &mut MoleculeCommon,
    template: Template,
    anchor_sns: &[u32],
    anchor_is: &[usize],
    r_aligner_is: &[usize],
    r_aligners: &[Vec3],
    start_sn: u32,
    start_i: usize,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    state_ui: &mut StateUi,
    controls: &mut ControlScheme,
    manip_mode: ManipMode,
    components: &Option<MolComponents>,
) {
    let anchor_posits = anchor_is
        .iter()
        .map(|i| mol.atoms[*i].posit)
        .collect::<Vec<_>>();

    let (atoms, bonds) = template.atoms_bonds(
        anchor_is,
        anchor_sns,
        &anchor_posits,
        r_aligners,
        start_sn,
        start_i,
    );

    mol.next_atom_sn += atoms.len() as u32;

    // Serial numbers rather than indices: we remove atoms below, which renumbers everything after
    // them. Used for populating H.
    let added_sns: Vec<u32> = atoms.iter().map(|a| a.serial_number).collect();

    for atom in &atoms {
        mol.atoms.push(atom.clone());
    }
    for bond in bonds {
        mol.bonds.push(bond);
    }

    if !template.is_ring() {
        // Add back the bond between this atom and the aligner atom.
        mol.bonds.push(Bond {
            bond_type: BondType::Single,
            atom_0_sn: mol.atoms[r_aligner_is[0]].serial_number,
            atom_1_sn: mol.atoms[start_i].serial_number,
            atom_0: r_aligner_is[0],
            atom_1: start_i,
            is_backbone: false,
        });
    }

    mol.reset_posits();
    mol.build_adjacency_list();

    // Set the anchor bond to Aromatic type if appropriate. The anchors' hydrogens are refreshed
    // below, which picks up the changed bond order.
    if template == Template::AromaticRing && anchor_is.len() == 2 {
        for bond in &mut mol.bonds {
            if (bond.atom_0 == anchor_is[0] && bond.atom_1 == anchor_is[1])
                || (bond.atom_0 == anchor_is[1] && bond.atom_1 == anchor_is[0])
            {
                bond.bond_type = BondType::Aromatic;
            }
        }
    }

    // For non-rings, the template's own anchor atom replaces the selected one, so the selected one
    // goes -- along with its hydrogens, which would otherwise be left bonded to nothing.
    if !template.is_ring() {
        for sn in anchor_sns {
            if let Some(i) = index_of_sn(mol, *sn) {
                remove_hydrogens(mol, i);
            }
            if let Some(i) = index_of_sn(mol, *sn) {
                mol.remove_atom(i);
            }
        }
    }

    for sn in &added_sns {
        let Some(i) = index_of_sn(mol, *sn) else {
            continue;
        };

        populate_hydrogens_on_atom(
            mol,
            i,
            &mut Vec::new(),
            state_ui,
            &mut Default::default(),
            manip_mode,
            components,
        );
    }

    // Ring anchors stay in the molecule, but now carry the ring's bonds, so how much room they
    // have for hydrogens has changed.
    if template.is_ring() {
        for sn in anchor_sns {
            refresh_hydrogens(
                mol,
                *sn,
                &mut Vec::new(),
                state_ui,
                &mut Default::default(),
                manip_mode,
                components,
            );
        }
    }

    *controls = ControlScheme::Arc {
        center: mol.centroid().into(),
    };

    *redraw = true;
    *rebuild_md = true;
}

impl MolEditorState {
    /// Wrapper to ensure we remove hydrogens.
    pub fn remove_atom(&mut self, i: usize) {
        remove_hydrogens(&mut self.mol.common, i);
        self.mol.common.remove_atom(i);
    }
}

/// An atom's index, from its serial number. Indices shift whenever an atom is removed; serial
/// numbers don't, so we hold onto those across edits that both add and remove atoms.
pub fn index_of_sn(mol: &MoleculeCommon, sn: u32) -> Option<usize> {
    mol.atoms.iter().position(|a| a.serial_number == sn)
}

/// The number of hydrogens covalently bonded to an atom.
pub fn h_count(mol: &MoleculeCommon, i: usize) -> usize {
    let Some(adj) = mol.adjacency_list.get(i) else {
        return 0;
    };

    adj.iter()
        .filter(|j| mol.atoms[**j].element == Hydrogen)
        .count()
}

/// Which way a hydrogen-count edit goes. See `adjust_h_count`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HCountOp {
    /// One more hydrogen than the atom currently has.
    Add,
    /// One fewer.
    Remove,
    /// However many the atom's element and bonds imply.
    Auto,
}

/// The hydrogen force-field type and bond length to use when adding one to an atom of a given
/// force field type. Falls back to a generic 1.1 Å where we have no entry, e.g. because the atom
/// hasn't been typed yet.
fn h_bond_params(ff_type: &Option<String>) -> (Option<String>, f64) {
    // Grabbing the first, arbitrarily.
    match hydrogens_avail(ff_type).into_iter().next() {
        Some((ff, bond_len)) => (Some(ff), bond_len),
        None => (None, 1.1),
    }
}

/// Add or remove a single hydrogen on an atom, or reset it to the count its valence implies.
///
/// The automatic count assumes a neutral atom in its usual valence state, so it gets charged
/// centres wrong: an ammonium N takes four bonds, and the terminal N of an azide takes none
/// beyond the one it has. This is the escape hatch for those, and for any other case where the
/// implied count isn't what you're drawing.
///
/// Takes a serial number rather than an index, since removing a hydrogen renumbers the atoms
/// after it.
pub fn adjust_h_count(
    mol: &mut MoleculeCommon,
    sn: u32,
    op: HCountOp,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    engine_updates: &mut EngineUpdates,
    manip_mode: ManipMode,
    components: &Option<MolComponents>,
) {
    let Some(i) = index_of_sn(mol, sn) else {
        return;
    };

    if mol.atoms[i].element == Hydrogen {
        return; // Hydrogen carries none of its own.
    }

    match op {
        HCountOp::Auto => refresh_hydrogens(
            mol,
            sn,
            entities,
            state_ui,
            engine_updates,
            manip_mode,
            components,
        ),
        HCountOp::Remove => {
            let Some(adj) = mol.adjacency_list.get(i) else {
                return;
            };
            let Some(&j) = adj.iter().find(|j| mol.atoms[**j].element == Hydrogen) else {
                return;
            };

            mol.remove_atom(j);
            engine_updates.entities = EntityUpdate::All;
        }
        HCountOp::Add => {
            let (ff_type, bond_len) = h_bond_params(&mol.atoms[i].force_field_type);

            // `add_atom` returns `None` if the coordination sphere is full, in which case there's
            // nothing to do.
            add_atom(
                mol,
                entities,
                i,
                Hydrogen,
                BondType::Single,
                ff_type,
                Some(bond_len),
                None,
                state_ui,
                engine_updates,
                &mut ControlScheme::None,
                manip_mode,
                components,
            );
        }
    }
}

/// Strip an atom's hydrogens, and add back however many its element and bonds now imply. Use this
/// after changing an atom's element, or the type of a bond it's part of. Identifies the atom by
/// serial number, as removing its hydrogens shifts indices.
pub fn refresh_hydrogens(
    mol: &mut MoleculeCommon,
    sn: u32,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    engine_updates: &mut EngineUpdates,
    manip_mode: ManipMode,
    components: &Option<MolComponents>,
) {
    if let Some(i) = index_of_sn(mol, sn) {
        remove_hydrogens(mol, i);
    }

    if let Some(i) = index_of_sn(mol, sn) {
        populate_hydrogens_on_atom(
            mol,
            i,
            entities,
            state_ui,
            engine_updates,
            manip_mode,
            components,
        );
    }
}

/// Consolidates actions which we take upon adding an atom bonded to another in the molecule,
/// including drawing the individual atom and the bond to it,
/// Returns the index of the atom added.
pub fn add_atom(
    mol: &mut MoleculeCommon,
    entities: &mut Vec<Entity>,
    i_par: usize, // Of the parent atom; this atom is bonded to it.
    element: Element,
    bond_type: BondType,
    ff_type: Option<String>,
    bond_len: Option<f64>,
    q: Option<f32>,
    ui: &mut StateUi,
    updates: &mut EngineUpdates,
    control: &mut ControlScheme,
    manip_mode: ManipMode,
    components: &Option<MolComponents>,
) -> Option<usize> {
    // Adding a non-hydrogen strips the parent's hydrogens to make room, which shifts indices. The
    // parent's serial number survives that.
    let par_sn = mol.atoms.get(i_par)?.serial_number;

    let (i_new_atom, i_new_bond) = mol.add_atom(i_par, element, bond_type, ff_type, bond_len, q)?;

    mol_editor::draw_atom(entities, &mol.atoms[i_new_atom], components, ui);
    mol_editor::draw_bond(
        entities,
        &mol.bonds[i_new_bond],
        &mol.atoms,
        &mol.bonds,
        &mol.adjacency_list,
        ui,
        components,
    );

    if element != Hydrogen {
        // Hydrogens on the new atom, then back onto the parent: the bond we just made used up one
        // of the parent's free valences, but any others it had still want filling. Without this
        // second step, building a chain leaves every atom but the last one bare.
        populate_hydrogens_on_atom(
            mol, i_new_atom, entities, ui, updates, manip_mode, components,
        );

        if let Some(i_par) = index_of_sn(mol, par_sn) {
            populate_hydrogens_on_atom(mol, i_par, entities, ui, updates, manip_mode, components);
        }

        *control = ControlScheme::Arc {
            center: mol.centroid().into(),
        };
    }

    // todo: Ideally just add the single entity, and add it to the
    // todo index buffer.
    updates.entities = EntityUpdate::All;

    Some(i_new_atom)
}

/// Populate hydrogens on a single atom. Uses tetrahedral, or planar geometry as required
/// based on atoms in the vicinity.
///
/// Note:  We can also use this outside the editor, for example, when loading small molecules that don't
/// have hydrogens. This is intended for use on small molecules; not proteins. For proteins, use the
/// template-based algorithm in *Dynamics*. Note that when used outside the editor  workflow, this
/// approach is overkill.
pub fn populate_hydrogens_on_atom(
    mol: &mut MoleculeCommon,
    i: usize,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    engine_updates: &mut EngineUpdates,
    manip_mode: ManipMode,
    components: &Option<MolComponents>,
) {
    if i >= mol.atoms.len() {
        eprintln!("Error: Invalid atom index when populating Hydrogens.");
        return;
    }

    let el = mol.atoms[i].element;
    if el == Hydrogen {
        return;
    }

    let h_to_add = bonds_avail(i, mol, el);

    for _ in 0..h_to_add {
        let (ff_type, bond_len) = h_bond_params(&mol.atoms[i].force_field_type);

        add_atom(
            mol,
            entities,
            i,
            Hydrogen,
            BondType::Single,
            ff_type,
            Some(bond_len),
            None,
            state_ui,
            engine_updates,
            &mut ControlScheme::None,
            manip_mode,
            components,
        );
    }
}
