//! PyMOL-style commands shared by the console, command files, and macros.

pub mod scripts;

use std::{env, fs, io, path::Path};

use graphics::{EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC, arc_rotation};

use crate::{
    annotations::{LabelFormat, center_slab, label_selection},
    cam::{self, VIEW_DIR_FRONT, reset_camera},
    drawing::{MoleculeView, representations},
    file_io::download_mols,
    render::set_flashlight,
    selection::expression,
    state::State,
    util,
};

fn new_invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Split at an unquoted comma.
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

pub const CLI_CMDS: &[&str] = &[
    "help", "fetch", "save", "load", "show", "as", "show_as", "hide", "color", "label", "zoom",
    "clip", "align", "view", "remove", "orient", "turn", "move", "reset", "pwd", "ls", "cd",
    "select", "delete", "set", "run", "alias", "unalias", "macros",
];

pub fn handle_cmd(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
) -> io::Result<String> {
    let input = state.ui.cmd_line_input.trim().to_owned();
    state.volatile.cli_input_history.push(input.clone());
    state.volatile.cli_input_selected = state.volatile.cli_input_history.len();

    scripts::execute(state, scene, updates, redraw, &input)
}

pub fn parse_color(input: &str) -> io::Result<[f32; 3]> {
    let value = input.trim().to_ascii_lowercase();
    let color = match value.as_str() {
        "red" => [1., 0., 0.],
        "green" => [0., 1., 0.],
        "blue" => [0., 0., 1.],
        "yellow" => [1., 1., 0.],
        "cyan" | "aqua" => [0., 1., 1.],
        "magenta" => [1., 0., 1.],
        "white" => [1.; 3],
        "black" => [0.; 3],
        "gray" | "grey" => [0.5; 3],
        "orange" => [1., 0.5, 0.],
        "purple" => [0.5, 0., 0.75],
        "pink" => [1., 0.5, 0.75],
        "salmon" => [1., 0.5, 0.5],
        "marine" => [0., 0.5, 1.],
        "lime" => [0.5, 1., 0.],
        "teal" => [0., 0.5, 0.5],
        "wheat" => [0.99, 0.82, 0.65],
        _ => {
            let hex = value
                .strip_prefix('#')
                .or_else(|| value.strip_prefix("0x"))
                .filter(|h| h.len() == 6 && h.chars().all(|c| c.is_ascii_hexdigit()))
                .ok_or_else(|| new_invalid("Unknown color; use a color name or #RRGGBB"))?;
            let rgb = u32::from_str_radix(hex, 16).map_err(|_| new_invalid("Invalid color"))?;

            [
                ((rgb >> 16) & 255) as f32 / 255.,
                ((rgb >> 8) & 255) as f32 / 255.,
                (rgb & 255) as f32 / 255.,
            ]
        }
    };

    Ok(color)
}

fn finite_number(input: &str) -> io::Result<f32> {
    let value: f32 = input
        .trim()
        .parse()
        .map_err(|_| new_invalid("Expected a number"))?;
    if !value.is_finite() {
        return Err(new_invalid("Expected a finite number"));
    }

    Ok(value)
}

pub(super) fn execute_one(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    input: &str,
) -> io::Result<String> {
    let (command, args) = input.split_once(char::is_whitespace).unwrap_or((input, ""));
    let args = args.trim();
    match command.to_ascii_lowercase().as_str() {
        "help" => Ok("select name, expression\nshow representation[, selection] · as representation[, selection] · hide representation[, selection]\ncolor name|#RRGGBB[, selection] · label selection[, residue|chain_residue|name|off]\nzoom [selection[, buffer]] · clip slab, thickness[, selection] · clip move, distance · clip off\nrun \"commands.pml\" (or @commands.pml) · alias name, command; command · name [selection] · unalias name · macros\nMacros accept $selection. Command files use one command per line or semicolons; # starts comments.\nOther commands: fetch, load, save, align, view, orient, turn, move, reset, pwd, ls, cd, delete, set".into()),
        "select" | "sele" => {
            let count = expression::select(state, args).map_err(|e| new_invalid(&e))?;
            *redraw = true;

            Ok(format!("Selected {count} atoms"))
        }
        "delete" => {
            expression::delete_named(state, args.trim_start_matches('%')).map_err(|e| new_invalid(&e))?;

            Ok(format!("Deleted selection {args}"))
        }
        "show" | "as" | "show_as" => {
            let (view, selection) = selection_args(args).unwrap_or((args, "all"));
            let view: MoleculeView = view.trim().parse()?;
            let count = if command.eq_ignore_ascii_case("show") { representations::add(state, selection.trim(), view) }
                else { representations::replace(state, selection.trim(), view) }.map_err(|e| new_invalid(&e))?;
            *redraw = true;

            Ok(format!("{view}: {count} atoms"))
        }
        "hide" => {
            let (view, selection) = selection_args(args).unwrap_or((args, "all"));
            if view.trim().eq_ignore_ascii_case("labels") {
                let count = label_selection(state, selection.trim(), None).map_err(|e| new_invalid(&e))?;

                return Ok(format!("Cleared labels on {count} atoms"));
            }
            let shorthand = match view.trim().to_ascii_lowercase().as_str() {
                "solvents" | "solvent" => Some("solvent"), "hetatm" => Some("hetatm"),
                "hydro" => Some("hydro"), "resn hoh" => Some("resn HOH"), _ => None,
            };
            let view = if shorthand.is_some() || matches!(view.trim().to_ascii_lowercase().as_str(), "everything" | "all") { None } else { Some(view.trim().parse()?) };
            let count = representations::hide(state, shorthand.unwrap_or(selection.trim()), view).map_err(|e| new_invalid(&e))?;
            *redraw = true;

            Ok(format!("Hidden on {count} atoms"))
        }
        "color" => {
            let (color, selection) = selection_args(args).unwrap_or((args, "all"));
            let count = representations::recolor(state, selection.trim(), parse_color(color)?).map_err(|e| new_invalid(&e))?;
            *redraw = true;

            Ok(format!("Colored {count} atoms"))
        }
        "label" => {
            let (selection, format) = selection_args(args).unwrap_or((if args.is_empty() { "sele" } else { args }, "residue"));
            let format = if matches!(format.trim().to_ascii_lowercase().as_str(), "off" | "none" | "\"\"" | "''") { None } else { Some(LabelFormat::parse(format).map_err(|e| new_invalid(&e))?) };
            let count = label_selection(state, selection.trim(), format).map_err(|e| new_invalid(&e))?;

            Ok(format!("Updated {count} labels"))
        }
        "zoom" | "orient" => {
            let (selection, buffer) = selection_args(args).unwrap_or((if args.is_empty() { "all" } else { args }, "2"));
            let buffer = finite_number(buffer)?;
            if buffer < 0. { return Err(new_invalid("Zoom buffer must be nonnegative")); }

            cam::zoom_selection(state, scene, updates, selection.trim(), buffer).map_err(|e| new_invalid(&e))?;

            Ok("Framed selection".into())
        }
        "clip" => {
            let (mode, rest) = selection_args(args).unwrap_or((args, ""));
            match mode.trim().to_ascii_lowercase().as_str() {
                "off" | "disable" => {
                    if let Some(slab) = &mut state.to_save.inspection.slab { slab.enabled = false; }
                }
                "slab" => {
                    let (thickness, selection) = selection_args(rest).unwrap_or((rest, "sele"));
                    center_slab(state, selection.trim(), finite_number(thickness)?).map_err(|e| new_invalid(&e))?;
                }
                "move" => {
                    let offset = finite_number(rest)?;
                    let slab = state.to_save.inspection.slab.as_mut().filter(|s| s.enabled).ok_or_else(|| new_invalid("Create a slab first: clip slab, 10, sele"))?;
                    if !(slab.offset + offset).is_finite() { return Err(new_invalid("Slab offset is too large")); }

                    slab.offset += offset;
                }
                _ => return Err(new_invalid("Usage: clip slab, thickness, selection; clip move, distance; clip off")),
            }
            state.to_save.save_flag = true;
            cam::set_fog(state, &mut scene.camera);
            updates.camera = true;

            Ok("Clipping updated".into())
        }
        "fetch" => {
            if args.len() != 4 || !args.chars().all(|c| c.is_ascii_alphanumeric()) { return Err(new_invalid("Usage: fetch PDB_ID")); }

            download_mols::load_atom_coords_rcsb(args, state, scene, updates, redraw);

            Ok(format!("Loaded {args} from RCSB PDB"))
        }
        "load" | "save" => {
            let filename = scripts::unquote(args)?;
            if filename.is_empty() { return Err(new_invalid("Supply a file path")); }

            let path = Path::new(&filename);
            if command.eq_ignore_ascii_case("load") {
                state.open_file(path, scene, updates)?;
                set_flashlight(scene);
                updates.lighting = true;
                *redraw = true;
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("mcproj")) {
                crate::project::save(state, scene, path)?;
            } else {
                state.save(path)?;
            }

            Ok(format!("Complete: {filename}"))
        }
        "align" => {
            let (mobile, reference) = selection_args(args).ok_or_else(|| new_invalid("Usage: align mobile, reference"))?;
            crate::ui::popup::protein_comparison::start_expressions(state, mobile.trim(), reference.trim()).map_err(|e| new_invalid(&e))?;

            Ok("Protein comparison started; see Compare proteins".into())
        }
        "view" => {
            let (name, action) = selection_args(args).unwrap_or((args, "recall"));
            let name = name.trim();
            match action.trim().to_ascii_lowercase().as_str() {
                "store" => util::save_snap(state, scene, name),
                "recall" => {
                    state.ui.cam_snapshot = Some(state.cam_snapshots.iter().position(|s| s.name.eq_ignore_ascii_case(name)).ok_or_else(|| new_invalid("Unknown view"))?);
                    util::load_snap(state, scene, updates);
                    *redraw = true;
                }
                _ => return Err(new_invalid("Usage: view name, store|recall")),
            }

            Ok("View updated".into())
        }
        "turn" | "move" => {
            let (axis, amount) = selection_args(args).ok_or_else(|| new_invalid("Usage: turn|move x|y|z, amount"))?;
            let axis = match axis.trim().to_ascii_lowercase().as_str() { "x" => RIGHT_VEC, "y" => UP_VEC, "z" => FWD_VEC, _ => return Err(new_invalid("Expected axis x, y, or z")) };
            let amount = finite_number(amount)?;
            if command.eq_ignore_ascii_case("turn") {
                let center = util::orbit_center(state);
                arc_rotation(&mut scene.camera, axis, amount.to_radians(), center);
            } else {
                scene.camera.position += axis * amount;
            }
            cam::set_fog(state, &mut scene.camera);
            updates.camera = true;

            Ok("Camera updated".into())
        }
        "reset" => {
            reset_camera(state, scene, updates, VIEW_DIR_FRONT);

            Ok("View reset".into())
        }
        "pwd" => Ok(env::current_dir()?.display().to_string()),
        "ls" => Ok(get_files_curdir()?.join("   ")),
        "cd" => {
            env::set_current_dir(scripts::unquote(args)?)?;

            Ok(format!("Now in {}", env::current_dir()?.display()))
        }
        "remove" => Err(new_invalid("Remove is temporarily disabled")),
        "set" => {
            let (name, value) = selection_args(args).unwrap_or((args, "on"));
            match name.trim().to_ascii_lowercase().as_str() {
                "seq_view" => state.ui.ui_vis.aa_seq = match value.trim().to_ascii_lowercase().as_str() {
                    "on" | "1" => true, "off" | "0" => false, _ => return Err(new_invalid("Use on or off")),
                },
                _ => return Err(new_invalid("Unknown setting")),
            }

            Ok("Setting updated".into())
        }
        _ => Err(new_invalid("Unknown command; enter help")),
    }
}

fn get_files_curdir() -> io::Result<Vec<String>> {
    let mut names: Vec<_> = fs::read_dir(env::current_dir()?)?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();

    Ok(names)
}

pub fn autocomplete_cli(input: &mut String) {
    let trimmed = input.trim();
    if let Some(cmd) = CLI_CMDS.iter().find(|c| c.starts_with(trimmed)) {
        *input = format!("{cmd} ");

        return;
    }

    if let Some((cmd, prefix)) = trimmed.split_once(' ')
        && matches!(cmd, "load" | "run" | "cd")
        && let Ok(files) = get_files_curdir()
        && let Some(name) = files
            .iter()
            .find(|n| n.starts_with(prefix) && (cmd != "cd" || Path::new(n).is_dir()))
    {
        *input = format!("{cmd} {}", expression::quote(name));
    }
}
