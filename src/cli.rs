//! Our CLI system. Apes PyMol's syntax. We don't introduce our own commands, as this functionality
//! is primarily for PyMol users who are comfortable with this workflow.
//!
//! On PyMol selection syntax: https://pymolwiki.org/index.php/Selection_Algebra

use std::{
    env,
    f32::consts::TAU,
    fs, io,
    io::ErrorKind,
    path::{Path, PathBuf},
    str::FromStr,
};

use graphics::{EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC, arc_rotation};
use mol_defs::molecules::AtomRole;
use na_seq::Element;
use regex::Regex;

use crate::{
    cam::{VIEW_DIR_FRONT, move_cam_to_sel, reset_camera},
    file_io::download_mols,
    render::set_flashlight,
    selection::Selection,
    state::State,
    util,
};

fn new_invalid(msg: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg)
}

/// A comma inside a quoted model path is part of the expression, not a name separator.
fn selection_args(input: &str) -> Option<(&str, &str)> {
    let mut quote = None;
    let mut chars = input.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if let Some(q) = quote {
            if c == '\\'
                && chars
                    .peek()
                    .is_some_and(|&(_, next)| next == q || next == '\\')
            {
                chars.next();
            } else if c == q {
                quote = None;
            }
        } else if c == '"' || c == '\'' {
            quote = Some(c);
        } else if c == ',' {
            return Some((&input[..i], &input[i + 1..]));
        }
    }
    None
}

// We use this for autocomplete.
pub const CLI_CMDS: [&str; 21] = [
    "help",
    "fetch",
    "save",
    "load",
    "show",
    "show_as",
    "align mobile, reference",
    "view",
    "hide",
    "remove",
    "orient",
    "turn",
    "move",
    "reset",
    "pwd",
    "ls",
    "cd",
    "select name, expression",
    "select expression",
    "delete selection_name",
    "set",
];

/// Process a raw CLI command from the user. Return the CLI output from the entered command.
pub fn handle_cmd(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) -> io::Result<String> {
    let input = state.ui.cmd_line_input.trim().to_string();

    state.volatile.cli_input_history.push(input.clone());
    state.volatile.cli_input_selected += 1;

    if let Some((command, args)) = input.split_once(char::is_whitespace) {
        let args = args.trim();
        match command.to_ascii_lowercase().as_str() {
            "align" => {
                let (mobile, reference) = selection_args(args).ok_or_else(|| {
                    new_invalid("Usage: align mobile_expression, reference_expression")
                })?;
                crate::ui::popup::protein_comparison::start_expressions(
                    state,
                    mobile.trim(),
                    reference.trim(),
                )
                .map_err(|e| new_invalid(&e))?;
                return Ok(
                    "Protein comparison started; results will appear in Compare proteins".into(),
                );
            }
            "select" | "sele" => {
                let count = if let Some((name, expression)) = selection_args(args) {
                    crate::selection::expression::save_named(state, name.trim(), expression.trim())
                        .map_err(|e| new_invalid(&e))?
                } else {
                    let atoms = crate::selection::expression::evaluate(state, args)
                        .map_err(|e| new_invalid(&e))?;
                    let count: usize = atoms.iter().map(|a| a.atoms.len()).sum();
                    state.ui.selection = Selection::Atoms(atoms);
                    count
                };
                *redraw = true;
                return Ok(format!("Selected {count} atoms"));
            }
            "delete" => {
                crate::selection::expression::delete_named(state, args.trim_start_matches('%'))
                    .map_err(|e| new_invalid(&e))?;
                return Ok(format!("Deleted selection {args}"));
            }
            "show" => {
                let (view, expression) = args.split_once(',').unwrap_or((args, "all"));
                let count = crate::drawing::representations::add(
                    state,
                    expression.trim(),
                    view.trim().parse()?,
                )
                .map_err(|e| new_invalid(&e))?;
                return Ok(format!(
                    "Added representation layer ({count} atoms). Edit it in Selections / layers."
                ));
            }
            _ => (),
        }
    }

    // todo: Helpers to reduce regex DRY.
    let re_help = Regex::new(r"(?i)^help$").unwrap();
    //
    let re_fetch = Regex::new(r"(?i)^fetch\s+([a-z0-9]{4})$").unwrap();
    let re_save = Regex::new(r"(?i)^save\s+([a-z0-9./]+)$").unwrap();
    let re_load = Regex::new(r"(?i)^load\s+([a-z0-9./]+)$").unwrap();
    //
    let re_show = Regex::new(r"(?i)^(?:show|show_as)\s+([a-z0-9./\-_]+)$").unwrap();
    // todo: Shoudl this be get_view and set_view? Have seen both.
    let re_view = Regex::new(r"(?i)^view\s+([^,\s]+)(?:\s*,\s*(store|recall))?\s*$").unwrap();
    let re_hide = Regex::new(r"(?i)^hide\s+([a-z0-9\s]+)$").unwrap();
    let re_remove = Regex::new(r"(?i)^remove\s+([a-z0-9\s]+)$").unwrap();
    //
    let re_orient = Regex::new(r"(?i)^orient\s*(?:sel)?$").unwrap();
    let re_turn = Regex::new(r"(?i)^turn\s+([xyz]),\s*(-*\d{1,4})$").unwrap();
    let re_move = Regex::new(r"(?i)^move\s+([xyz]),\s*(-*\d{1,4})$").unwrap();
    let _re_zoom = Regex::new(r"(?i)^zoom\s+([a-z0-9\s]+)$").unwrap();
    let re_reset = Regex::new(r"(?i)^reset\s*$").unwrap();
    //
    let re_pwd = Regex::new(r"(?i)^pwd\s*$").unwrap();
    let re_ls = Regex::new(r"(?i)^ls\s*$").unwrap();
    let re_cd = Regex::new(r"(?i)^cd\s+(.+)$").unwrap();

    let re_set = Regex::new(r"(?i)^set\s+([a-z0-9\s\-_]+)(?:,\s*([a-z0-9]+))?$").unwrap();

    if let Some(_caps) = re_help.captures(&input) {
        // todo: Multiline, once you set that up.
        return Ok(format!(
            "The following commands are available: {}",
            CLI_CMDS.join(", ")
        ));
    }

    if let Some(caps) = re_fetch.captures(&input) {
        let ident = &caps[1];
        download_mols::load_atom_coords_rcsb(ident, state, scene, updates, redraw, reset_cam);

        return Ok(format!("Loaded {ident} molecule from RCSB PDB"));
    }

    // todo: Save and load: Limited functionalitiy, and DRY with ui.

    if let Some(caps) = re_save.captures(&input) {
        let filename = &caps[1];
        let path = PathBuf::from_str(filename).unwrap();

        state.save(&path)?;

        return Ok(format!("Saved {filename}"));
    }

    // todo: Load other types of file, e.g. map and mtz.
    if let Some(caps) = re_load.captures(&input) {
        let filename = &caps[1];
        let path = PathBuf::from_str(filename).unwrap();

        state.open_mol_from_file(&path, scene, updates)?;

        set_flashlight(scene);
        updates.lighting = true;

        return Ok(format!("Loaded {filename}"));
    }

    // show_as retains the global default-view shortcut; show creates additive layers.
    if let Some(caps) = re_show.captures(&input) {
        let mode = &caps[1];
        let view = mode.parse()?;

        state.ui.mol_view_peptide = view;
        if view.is_non_peptide() {
            state.ui.mol_view = view;
        }
        *redraw = true;
        return Ok("Complete".to_owned());
    }

    if let Some(caps) = re_view.captures(&input) {
        let name = &caps[1];

        let mut recall = false;
        match caps.get(2) {
            Some(action) => {
                if action.as_str().eq_ignore_ascii_case("store") {
                    util::save_snap(state, &scene.camera, name);
                } else {
                    recall = true;
                }
            }
            None => recall = true,
        }

        if recall {
            let mut found = false; // Avoids borrow error.
            for (i, snap) in state.cam_snapshots.iter().enumerate() {
                if snap.name.to_lowercase().trim() == name.to_lowercase().trim() {
                    state.ui.cam_snapshot = Some(i);
                    found = true;
                    break;
                }
            }
            if found {
                util::load_snap(state, scene, updates);
            }
        }

        return Ok("Complete".to_owned());
    }

    if let Some(caps) = re_hide.captures(&input) {
        let item = &caps[1].to_lowercase();

        // todo: To match PyMol, this should be much more robust. Hiding chains, residues etc.
        // todo: A lot of these just hide, not remove... Should remove probably.
        match item.as_ref() {
            "solvents" => {
                state.ui.visibility.hide_hetero = true;
            }
            "hetatm" => {
                state.ui.visibility.hide_hetero = true;
                state.ui.visibility.hide_ligand = true;
            }
            "resn hoh" => {
                // todo: The space won't work in the regex.
                state.ui.visibility.hide_water = true;
            }
            "hydro" => {
                // todo: The space won't work in the regex.
                state.ui.visibility.hide_hydrogen = true;
            }
            _ => (),
        }

        *redraw = true;
        return Ok("Complete".to_owned());
    }

    #[allow(unused)]
    if let Some(caps) = re_remove.captures(&input) {
        let item = &caps[1].to_lowercase();

        // todo: To match PyMol, this should be much more robust. Removing chains, residues etc.
        // todo: A lot of these just hide, not remove... Should remove probably.

        return Ok("Remove is temporarily disabled".to_owned());

        // todo uhoh: When you remove atoms, their indices in the vec get screwed up! You may need to use
        // todo a unique id!
        if let Some(mol) = state
            .peptide_for_tools_i()
            .and_then(|i| state.peptides.get_mut(i))
        {
            match item.as_ref() {
                "solvents" => {
                    // todo: Remove residues as well?
                    mol.common.atoms.retain(|a| {
                        if let Some(role) = a.role {
                            role != AtomRole::Water
                        } else {
                            true
                        }
                    });
                }
                "hetatm" => {
                    mol.common.atoms.retain(|a| !a.hetero);
                }
                "resn hoh" => {
                    // todo: Remove residues as well?
                    mol.common.atoms.retain(|a| {
                        if let Some(role) = a.role {
                            role != AtomRole::Water
                        } else {
                            true
                        }
                    });
                }
                "hydro" => {
                    mol.common.atoms.retain(|a| a.element != Element::Hydrogen);
                    state.ui.visibility.hide_hydrogen = true;
                }
                _ => (),
            }
        }

        //
        // match item.as_ref() {
        //     "" => {}
        //     _ => (),
        // }

        *redraw = true;

        return Ok("Complete".to_owned());
    }

    if let Some(_caps) = re_orient.captures(&input) {
        updates.camera = true;

        return Ok("Complete".to_owned());
    }

    if let Some(caps) = re_turn.captures(&input) {
        let Some(mol) = state
            .peptide_for_tools_i()
            .and_then(|i| state.peptides.get(i))
        else {
            return Ok(String::from("Can't turn without a molecule"));
        };

        let axis = match caps[1].to_lowercase().as_ref() {
            "x" => RIGHT_VEC,
            "y" => UP_VEC,
            "z" => FWD_VEC,
            _ => unreachable!(),
        };
        let amt: f32 = caps[2]
            .parse()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid angle."))?;

        let amt = amt * TAU / 360.;

        arc_rotation(&mut scene.camera, axis, amt, mol.common.centroid().into());

        updates.camera = true;

        return Ok("Complete".to_owned());
    }

    if let Some(caps) = re_move.captures(&input) {
        let axis = match caps[1].to_lowercase().as_ref() {
            "x" => RIGHT_VEC,
            "y" => UP_VEC,
            "z" => FWD_VEC,
            _ => unreachable!(),
        };
        let amt: f32 = caps[2]
            .parse()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid move amount."))?;

        let movement = axis * amt;
        scene.camera.position += movement;
        updates.camera = true;

        return Ok("Complete".to_owned());
    }

    if let Some(_caps) = re_orient.captures(&input) {
        move_cam_to_sel(state, &mut scene.camera, updates);

        return Ok("Complete".to_owned());
    }

    if re_reset.captures(&input).is_some() {
        reset_camera(state, scene, updates, VIEW_DIR_FRONT);
        return Ok("Complete".to_owned());
    }

    if re_pwd.captures(&input).is_some() {
        return Ok(format!("{}", env::current_dir()?.display()));
    }

    if re_ls.captures(&input).is_some() {
        let names = get_files_curdir()?;
        return Ok(names.join("   "));
    }

    if let Some(caps) = re_cd.captures(&input) {
        let dir = &caps[1];

        // Note: This doesn't handle ~ properly.

        env::set_current_dir(dir)?;
        return Ok(format!("Now in {}", env::current_dir()?.display()));
    }

    if let Some(caps) = re_set.captures(&input) {
        let action = &caps[1].to_lowercase();

        match action.as_ref() {
            "seq_view" => {
                state.ui.ui_vis.aa_seq = match caps.get(2) {
                    Some(action_2) => action_2.as_str() == "off",
                    None => true,
                };
                return Ok("Complete".to_owned());
            }
            "seq_view_format" => {
                // todo: Support 3-letter seq.
                return Ok("Complete".to_owned());
            }
            _ => return Err(new_invalid("Unable to find this residue")),
        }
    }

    Err(new_invalid("Can't find that command"))
}

fn get_files_curdir() -> io::Result<Vec<String>> {
    let entries = fs::read_dir(env::current_dir()?)?;
    Ok(entries
        .filter_map(|dir_entry| dir_entry.ok())
        .map(|dir_entry| dir_entry.file_name().to_string_lossy().into_owned())
        .collect())
}

/// Simple autocomplete.
pub fn autocomplete_cli(input: &mut String) {
    // todo: Try to guess arguments; not just the params.
    let trimmed = input.trim().to_string();
    for cmd in CLI_CMDS {
        if cmd.starts_with(&trimmed) {
            // Complete the command.
            // Start with just the command.
            *input = cmd.to_owned() + " ";
            // todo: Make it so it auto-positiosn the cursor at the end.
            // edit_resp.surrender_focus();
            // edit_resp.request_focus();
        } else if trimmed.starts_with(cmd) {
            // Complete the action.

            match cmd {
                "view" => {}
                "load" => {
                    // todo: Filter names by extension.
                    let fnames = get_files_curdir().unwrap_or_default();
                    for name in fnames {
                        if format!("{cmd} {name}").starts_with(&trimmed) {
                            *input = format!("{cmd} {name}");
                        }
                    }
                }
                "cd" => {
                    let fnames = get_files_curdir().unwrap_or_default();
                    for name in fnames {
                        // check if “name” is a directory on disk
                        if Path::new(&name).is_dir()
                            && format!("{cmd} {name}").starts_with(&trimmed)
                        {
                            *input = format!("{cmd} {name}");
                            break;
                        }
                    }
                }
                _ => (),
            }
        }
    }
}
