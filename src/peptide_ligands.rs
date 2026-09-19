//! Adding ligands to proteins, and removing or detaching them; e.g. a ligand bound to a protein in
//! an mmCIF file from the RCSB. The protein's atoms and its mmCIF text are kept in sync; see
//! `mol_defs::molecules::peptide_ligands`.

use graphics::{EngineUpdates, Scene};
use mol_defs::molecules::{MolType, MoleculeGeneric};

use crate::{
    drawing::{peptide::draw_peptide, wrappers::draw_all_ligs},
    selection::Selection,
    state::State,
    util::{RedrawFlags, close_mol, handle_err, handle_success},
};

/// State for the popup that adds a ligand to a protein.
#[derive(Debug)]
pub struct LigAttachUi {
    pub lig_i: usize,
    pub peptide_i: usize,
    /// Residue name (chemical component ID) to give it, e.g. "ATP".
    pub comp_id: String,
    /// Close the standalone ligand once it's part of the protein.
    pub close_lig: bool,
}

impl LigAttachUi {
    pub fn new(state: &State, lig_i: usize) -> Self {
        let peptide_i = state.peptide_for_tools_i().unwrap_or_default();

        let comp_id = match (state.ligands.get(lig_i), state.peptides.get(peptide_i)) {
            (Some(lig), Some(pep)) => pep.suggest_comp_id(lig),
            _ => "LIG".to_owned(),
        };

        Self {
            lig_i,
            peptide_i,
            comp_id,
            close_lig: true,
        }
    }
}

/// Follow-up after changing a peptide's atoms. Optionally selects a residue.
fn after_peptide_edit(
    state: &mut State,
    peptide_i: usize,
    select_res: Option<usize>,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    // Peptide selections index into its atoms and residues, which have shifted.
    if state.ui.selection.is_peptide() {
        state.ui.selection = Selection::None;
    }
    if let Some(res_i) = select_res {
        state.ui.selection = Selection::Residue(res_i);
    }

    if state.peptide_for_tools_i() == Some(peptide_i) {
        // The solvent-accessible surface may include hetero atoms.
        state.volatile.flags.sas_mesh_created = false;
        state.volatile.flags.ss_mesh_dirty = true;
    }

    // A simulation set up with this peptide indexes its old atoms; as when closing it.
    if state
        .peptides
        .get(peptide_i)
        .is_some_and(|p| p.common.selected_for_md.is_some())
    {
        state.volatile.md_local.mol_dynamics = None;
        state.volatile.md_local.pep_atom_set.clear();
    }

    draw_peptide(state, scene, updates);
}

/// Remove a ligand (or ion, cofactor etc.) from a protein, and add it as a standalone ligand where
/// it is. Its mmCIF records go with it, so adding it back to the protein restores them.
pub fn detach_het_res(
    state: &mut State,
    peptide_i: usize,
    res_i: usize,
    include_disconnected: bool,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    let Some(pep) = state.peptides.get_mut(peptide_i) else {
        return;
    };
    let name = pep
        .residues
        .get(res_i)
        .map(|r| r.res_type.to_string())
        .unwrap_or_default();
    let ident = pep.common.ident.clone();

    match pep.detach_het_residue_with_fragments(res_i, include_disconnected) {
        Ok(lig) => {
            after_peptide_edit(state, peptide_i, None, scene, updates);

            state.load_mol_to_state_in_place(MoleculeGeneric::Small(lig), scene, updates);
            state.ui.visibility.hide_ligand = false;

            handle_success(
                &mut state.ui,
                format!(
                    "Detached {name} from {ident}. Use \"Add to protein\" on it to put it back. \
                    Save the protein to keep this change."
                ),
            );
        }
        Err(e) => handle_err(&mut state.ui, format!("Unable to detach {name}: {e}")),
    }
}

/// Remove a ligand (or ion, cofactor etc.) from a protein, and its mmCIF.
pub fn remove_het_res(
    state: &mut State,
    peptide_i: usize,
    res_i: usize,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    let Some(pep) = state.peptides.get_mut(peptide_i) else {
        return;
    };
    let name = pep
        .residues
        .get(res_i)
        .map(|r| r.res_type.to_string())
        .unwrap_or_default();
    let ident = pep.common.ident.clone();

    match pep.remove_het_residue(res_i) {
        Ok(_) => {
            after_peptide_edit(state, peptide_i, None, scene, updates);
            handle_success(
                &mut state.ui,
                format!("Removed {name} from {ident}. Save the protein to keep this change."),
            );
        }
        Err(e) => handle_err(&mut state.ui, format!("Unable to remove {name}: {e}")),
    }
}

/// Add a ligand to a protein, where it is, as a hetero residue; and to the protein's mmCIF.
/// Returns true on success.
pub fn attach_lig(
    state: &mut State,
    attach: &LigAttachUi,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) -> bool {
    let (Some(lig), Some(pep)) = (
        state.ligands.get(attach.lig_i),
        state.peptides.get_mut(attach.peptide_i),
    ) else {
        handle_err(
            &mut state.ui,
            "Unable to add the ligand: it or the protein is no longer open".to_owned(),
        );
        return false;
    };
    let lig_ident = lig.common.ident.clone();
    let ident = pep.common.ident.clone();

    let res_i = match pep.attach_ligand(lig, &attach.comp_id) {
        Ok(i) => i,
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to add {lig_ident} to {ident}: {e}"),
            );
            return false;
        }
    };

    if attach.close_lig {
        let mut redraw = RedrawFlags::default();
        close_mol(
            MolType::Ligand,
            attach.lig_i,
            state,
            scene,
            &mut redraw,
            updates,
        );
        draw_all_ligs(state, scene, updates);
    }

    // Show the protein, with its new residue selected.
    state.volatile.active_mol = Some((MolType::Peptide, attach.peptide_i));
    state.volatile.active_peptide = Some(attach.peptide_i);
    after_peptide_edit(state, attach.peptide_i, Some(res_i), scene, updates);

    handle_success(
        &mut state.ui,
        format!(
            "Added {lig_ident} to {ident} as {}. Save the protein to keep this change.",
            attach.comp_id.trim().to_uppercase()
        ),
    );
    true
}
