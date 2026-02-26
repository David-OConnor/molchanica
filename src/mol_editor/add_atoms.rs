use std::sync::atomic::Ordering;

use bio_files::BondType;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::Vec3;
use na_seq::{Element, Element::Hydrogen};

use crate::molecules::common::bonds_avail;
use crate::{
    mol_components::MolComponents,
    mol_editor,
    mol_editor::{MolEditorState, templates::Template},
    mol_manip::ManipMode,
    molecules::{Bond, common::MoleculeCommon},
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

    let mut i_added = Vec::new(); // Used for populating H.

    for atom in &atoms {
        mol.atoms.push(atom.clone());
        i_added.push(mol.atoms.len() - 1);
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

    // Set the anchor bond to Aromatic type if appropriate.
    if template == Template::AromaticRing && anchor_is.len() == 2 {
        let mut atoms_to_update_h = Vec::new(); // avoids db-borrow error.
        for bond in &mut mol.bonds {
            if (bond.atom_0 == anchor_is[0] && bond.atom_1 == anchor_is[1])
                | (bond.atom_0 == anchor_is[1] && bond.atom_1 == anchor_is[0])
            {
                bond.bond_type = BondType::Aromatic;
            }

            // todo: This section is dry with the GUI buttons to change bond types. Use a common fn for htis.
            for i in 0..mol.atoms.len() {
                if bond.atom_0 != i && bond.atom_1 != i {
                    continue;
                }
                atoms_to_update_h.push(i);
            }
        }

        for i in atoms_to_update_h {
            remove_hydrogens(mol, i);
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
    }

    // Get the FF type for this atom prior to adding H.
    // For non-rings, we are currently the selected atom with the added group's anchor.
    // So, remove it and its H atoms.
    for &anchor_i in anchor_is {
        if !template.is_ring() {
            mol.remove_atom(anchor_i);
        }
    }

    for i in i_added {
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

    for &anchor_i in anchor_is {
        remove_hydrogens(mol, anchor_i);
        populate_hydrogens_on_atom(
            mol,
            anchor_i,
            &mut Vec::new(),
            state_ui,
            &mut Default::default(),
            manip_mode,
            components,
        );
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

/// Remove all hydrogens bonded to an atom.
pub fn remove_hydrogens(mol: &mut MoleculeCommon, i: usize) {
    let mut h_to_del = Vec::new();

    // Remove Hydrogens; we'll add any back as applicable.
    for j in &mol.adjacency_list[i] {
        if mol.atoms[*j].element == Hydrogen {
            h_to_del.push(*j);
        }
    }

    h_to_del.sort_unstable_by(|a, b| b.cmp(a));
    for j in h_to_del {
        mol.remove_atom(j);
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
    let Some((i_new_atom, i_new_bond)) =
        mol.add_atom(i_par, element, bond_type, ff_type, bond_len, q)
    else {
        return None;
    };

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

    // Add hydrogens to the new atom.
    // Up to one recursion to add hydrogens to this parent and to the new atom.
    if element != Hydrogen {
        populate_hydrogens_on_atom(
            mol, i_new_atom, entities, ui, updates, manip_mode, components,
        );
        *control = ControlScheme::Arc {
            center: mol.centroid().into(),
        };
    }

    // todo: Ideally just add the single entity, and add it to the
    // todo index buffer.
    updates.entities = EntityUpdate::All;

    Some(i_new_bond)
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

    // let bonds_remaining = bonds_avail.saturating_sub(adj.len());

    for _ in 0..h_to_add {
        let atom = &mol.atoms[i];
        let (ff_type, bond_len) = {
            let mut v = (None, 1.1);

            // Grabbing the first, arbitrarily.
            for (ff, bl) in hydrogens_avail(&atom.force_field_type) {
                v.0 = Some(ff);
                v.1 = bl;
                break;
            }

            v
        };

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

// todo: I think this approach is wrong. You can add multiple of the same one...
/// This is built from Amber's gaff2.dat. Returns each H FF type that can be bound to a given atom
/// (by force field type), and the bond distance in Å.
pub fn hydrogens_avail(ff_type: &Option<String>) -> Vec<(String, f64)> {
    let Some(f) = ff_type else { return Vec::new() };
    match f.as_ref() {
        // Water
        "ow" => vec![("hw".to_owned(), 0.9572)],
        "hw" => vec![("hw".to_owned(), 1.5136)],

        // Generic sp carbon (c )
        "c" => vec![
            ("h4".to_owned(), 1.1123),
            ("h5".to_owned(), 1.1053),
            ("ha".to_owned(), 1.1010),
        ],

        // sp2 carbon families
        "c1" => vec![("ha".to_owned(), 1.0666), ("hc".to_owned(), 1.0600)],
        "c2" => vec![
            ("h4".to_owned(), 1.0865),
            ("h5".to_owned(), 1.0908),
            ("ha".to_owned(), 1.0882),
            ("hc".to_owned(), 1.0870),
            ("hx".to_owned(), 1.0836),
        ],
        "c3" => vec![
            ("h1".to_owned(), 1.0969),
            ("h2".to_owned(), 1.0950),
            ("h3".to_owned(), 1.0938),
            ("hc".to_owned(), 1.0962),
            ("hx".to_owned(), 1.0911),
        ],
        "c5" => vec![
            ("h1".to_owned(), 1.0972),
            ("h2".to_owned(), 1.0955),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0954),
            ("hx".to_owned(), 1.0917),
        ],
        "c6" => vec![
            ("h1".to_owned(), 1.0984),
            ("h2".to_owned(), 1.0985),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0979),
            ("hx".to_owned(), 1.0931),
        ],

        // Aromatic/condensed ring carbons
        "ca" => vec![
            ("ha".to_owned(), 1.0860),
            ("h4".to_owned(), 1.0885),
            ("h5".to_owned(), 1.0880),
        ],
        "cc" => vec![
            ("h4".to_owned(), 1.0809),
            ("h5".to_owned(), 1.0820),
            ("ha".to_owned(), 1.0838),
            ("hx".to_owned(), 1.0827),
        ],
        "cd" => vec![
            ("h4".to_owned(), 1.0818),
            ("h5".to_owned(), 1.0821),
            ("ha".to_owned(), 1.0835),
            ("hx".to_owned(), 1.0801),
        ],
        "ce" => vec![
            ("h4".to_owned(), 1.0914),
            ("h5".to_owned(), 1.0895),
            ("ha".to_owned(), 1.0880),
        ],
        "cf" => vec![
            ("h4".to_owned(), 1.0942),
            ("ha".to_owned(), 1.0885),
            // table also lists h5-cf (reverse order) at 1.0890
            ("h5".to_owned(), 1.0890),
        ],
        "cg" => Vec::new(), // no H entries shown for cg in the provided snippet

        // Other carbon families frequently seen
        "cu" => vec![("ha".to_owned(), 1.0786)],
        "cv" => vec![("ha".to_owned(), 1.0878)],
        "cx" => vec![
            ("h1".to_owned(), 1.0888),
            ("h2".to_owned(), 1.0869),
            ("hc".to_owned(), 1.0865),
            ("hx".to_owned(), 1.0849),
        ],
        "cy" => vec![
            ("h1".to_owned(), 1.0946),
            ("h2".to_owned(), 1.0930),
            ("hc".to_owned(), 1.0947),
            ("hx".to_owned(), 1.0913),
        ],

        // Nitrogen families: protonated H type is "hn"
        "n1" => vec![("hn".to_owned(), 0.9860)],
        "n2" => vec![("hn".to_owned(), 1.0221)],
        "n3" => vec![("hn".to_owned(), 1.0190)],
        "n4" => vec![("hn".to_owned(), 1.0300)],
        "n" => vec![("hn".to_owned(), 1.0130)],
        "n5" => vec![("hn".to_owned(), 1.0211)],
        "n6" => vec![("hn".to_owned(), 1.0183)],
        "n7" => vec![("hn".to_owned(), 1.0195)],
        "n8" => vec![("hn".to_owned(), 1.0192)],
        "n9" => vec![("hn".to_owned(), 1.0192)],
        "na" => vec![("hn".to_owned(), 1.0095)],
        "nh" => vec![("hn".to_owned(), 1.0120)],
        "nj" => vec![("hn".to_owned(), 1.0130)],
        "nl" => vec![("hn".to_owned(), 1.0476)],
        "no" => vec![("hn".to_owned(), 1.0440)],
        "np" => vec![("hn".to_owned(), 1.0210)],
        "nq" => vec![("hn".to_owned(), 1.0180)],
        "ns" => vec![("hn".to_owned(), 1.0132)],
        "nt" => vec![("hn".to_owned(), 1.0105)],
        "nu" => vec![("hn".to_owned(), 1.0137)],
        "nv" => vec![("hn".to_owned(), 1.0114)],
        "nx" => vec![("hn".to_owned(), 1.0338)],
        "ny" => vec![("hn".to_owned(), 1.0339)],
        "nz" => vec![("hn".to_owned(), 1.0271)],

        // Oxygen families: hydroxyl H type is "ho"
        "o" => vec![("ho".to_owned(), 0.9810)],
        "oh" => vec![("ho".to_owned(), 0.9725)],

        // Sulfur families: thiol H type is "hs"
        "s" => vec![("hs".to_owned(), 1.3530)],
        "s4" => vec![("hs".to_owned(), 1.3928)],
        "s6" => vec![("hs".to_owned(), 1.3709)],
        "sh" => vec![("hs".to_owned(), 1.3503)],
        "sy" => vec![("hs".to_owned(), 1.3716)],

        // Phosphorus families: acidic phosphate H type is "hp"
        "p2" => vec![("hp".to_owned(), 1.4272)],
        "p3" => vec![("hp".to_owned(), 1.4256)],
        "p4" => vec![("hp".to_owned(), 1.4271)],
        "p5" => vec![("hp".to_owned(), 1.4205)],
        "py" => vec![("hp".to_owned(), 1.4150)],

        _ => Vec::new(),
    }
}
