//! Command files and saved command macros.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use graphics::{EngineUpdates, Scene};

use crate::{
    cli::{CLI_CMDS, new_invalid, selection_args},
    state::State,
};

#[derive(Clone, Debug, PartialEq, bincode::Encode, bincode::Decode)]
pub struct CommandMacro {
    pub name: String,
    pub body: String,
}

pub fn store_macro(state: &mut State, name: &str, body: &str) -> io::Result<()> {
    let name = name.trim().to_ascii_lowercase();
    if name.is_empty()
        || name.len() > 64
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.starts_with(|c: char| c.is_ascii_digit())
        || CLI_CMDS
            .iter()
            .any(|c| c.split_whitespace().next() == Some(&name))
        || matches!(name.as_str(), "alias" | "unalias" | "macros")
    {
        return Err(new_invalid(
            "Use a macro name with letters, digits, and underscores; command names are reserved",
        ));
    }
    if body.trim().is_empty() || body.len() > 1_000_000 {
        return Err(new_invalid(
            "Macro body must contain commands (maximum 1 MB)",
        ));
    }

    commands(body)?;
    state.to_save.command_macros.retain(|m| m.name != name);
    state.to_save.command_macros.push(CommandMacro {
        name,
        body: body.to_owned(),
    });
    state.to_save.save_flag = true;

    Ok(())
}

fn commands(text: &str) -> io::Result<Vec<(usize, String)>> {
    let text = text.trim_start_matches('\u{feff}');
    let mut result = Vec::new();
    for (line, input) in text.lines().enumerate() {
        let mut quote = None;
        let mut chars = input.char_indices().peekable();
        let mut start = 0;
        let alias = input
            .trim_start()
            .split_whitespace()
            .next()
            .is_some_and(|s| s.eq_ignore_ascii_case("alias"));
        let mut end = input.len();
        while let Some((i, c)) = chars.next() {
            if let Some(q) = quote {
                if c == '\\' && chars.peek().is_some_and(|&(_, c)| c == q || c == '\\') {
                    chars.next();
                } else if c == q {
                    quote = None;
                }
            } else if c == '"' || c == '\'' {
                quote = Some(c);
            } else if c == '#' && (i == 0 || input[..i].ends_with(char::is_whitespace)) {
                let token = input[i + 1..]
                    .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
                    .next()
                    .unwrap_or("");
                if token.len() == 6
                    && token.chars().all(|c| c.is_ascii_hexdigit())
                    && input[..i]
                        .split_whitespace()
                        .last()
                        .is_some_and(|s| s.eq_ignore_ascii_case("color"))
                {
                    continue;
                }

                end = i;
                break;
            } else if c == ';' && !alias {
                let command = input[start..i].trim();
                if !command.is_empty() {
                    result.push((line + 1, command.to_owned()));
                }
                start = i + 1;
            }
        }
        if quote.is_some() {
            return Err(new_invalid(&format!("Line {}: unclosed quote", line + 1)));
        }

        let command = input[start..end].trim();
        if !command.is_empty() {
            result.push((line + 1, command.to_owned()));
        }
    }

    Ok(result)
}

pub fn unquote(input: &str) -> io::Result<String> {
    let input = input.trim();
    if input.starts_with(['\'', '"']) {
        let quote = input.chars().next().unwrap();
        if input.len() < 2 || !input.ends_with(quote) {
            return Err(new_invalid("Unclosed path quote"));
        }

        let mut result = String::new();
        let mut chars = input[1..input.len() - 1].chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' && chars.peek().is_some_and(|&n| n == quote || n == '\\') {
                result.push(chars.next().unwrap());
            } else {
                result.push(c);
            }
        }

        return Ok(result);
    }

    Ok(input.to_owned())
}

pub fn execute(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    text: &str,
) -> io::Result<String> {
    let mut remaining = 4096;

    execute_inner(
        state,
        scene,
        updates,
        redraw,
        text,
        &std::env::current_dir()?,
        0,
        &mut remaining,
    )
}

fn execute_inner(
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
    redraw: &mut bool,
    text: &str,
    directory: &Path,
    depth: usize,
    remaining: &mut usize,
) -> io::Result<String> {
    if depth >= 16 {
        return Err(new_invalid(
            "Command-file or macro recursion limit (16) reached",
        ));
    }

    let mut output = Vec::new();
    for (line, input) in commands(text)? {
        let directory = if depth == 0 {
            std::env::current_dir()?
        } else {
            directory.to_path_buf()
        };
        if *remaining == 0 {
            return Err(new_invalid("Command limit (4096) reached"));
        }
        *remaining -= 1;
        let (cmd, args) = input
            .split_once(char::is_whitespace)
            .unwrap_or((&input, ""));
        let cmd = cmd.to_ascii_lowercase();
        let args = args.trim();
        let result = (|| {
            if cmd == "alias" {
                let (name, body) = selection_args(args)
                    .ok_or_else(|| new_invalid("Usage: alias name, command; command"))?;
                store_macro(state, name, body)?;

                return Ok(format!("Saved macro {}", name.trim()));
            }
            if cmd == "unalias" {
                let name = args.to_ascii_lowercase();
                let before = state.to_save.command_macros.len();
                state.to_save.command_macros.retain(|m| m.name != name);
                if before == state.to_save.command_macros.len() {
                    return Err(new_invalid("Unknown macro"));
                }

                state.to_save.save_flag = true;

                return Ok(format!("Deleted macro {name}"));
            }
            if cmd == "macros" {
                return Ok(state
                    .to_save
                    .command_macros
                    .iter()
                    .map(|m| m.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "));
            }
            if cmd == "run" || input.starts_with('@') {
                let path = unquote(if input.starts_with('@') {
                    &input[1..]
                } else {
                    args
                })?;
                let path = directory.join(PathBuf::from(path));
                if fs::metadata(&path)?.len() > 4_000_000 {
                    return Err(new_invalid("Command file exceeds 4 MB"));
                }

                let text = fs::read_to_string(&path)?;

                return execute_inner(
                    state,
                    scene,
                    updates,
                    redraw,
                    &text,
                    path.parent().unwrap_or(&directory),
                    depth + 1,
                    remaining,
                )
                .map_err(|e| new_invalid(&format!("{}: {e}", path.display())));
            }
            if let Some(mac) = state.to_save.command_macros.iter().find(|m| m.name == cmd) {
                let body = mac.body.clone();
                let selection = if args.is_empty() { "sele" } else { args };
                let body = body.replace("$selection", &format!("({selection})"));

                return execute_inner(
                    state,
                    scene,
                    updates,
                    redraw,
                    &body,
                    &directory,
                    depth + 1,
                    remaining,
                )
                .map_err(|e| new_invalid(&format!("Macro {cmd}: {e}")));
            }

            if matches!(cmd.as_str(), "load" | "save" | "cd") {
                let path = unquote(args)?;
                if path.is_empty() {
                    return Err(new_invalid("Supply a file path"));
                }

                let path = directory.join(path);
                let input = format!(
                    "{cmd} {}",
                    crate::selection::expression::quote(&path.to_string_lossy())
                );

                return super::execute_one(state, scene, updates, redraw, &input);
            }

            super::execute_one(state, scene, updates, redraw, &input)
        })()
        .map_err(|e: io::Error| new_invalid(&format!("Line {line}: {e}")))?;
        if !result.is_empty() {
            output.push(result);
        }
    }

    Ok(output.join("\n"))
}
