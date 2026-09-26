//! Opening and saving sequences: DNA, RNA, and protein. FASTA and GenBank can be opened and
//! saved; AB1 (Sanger traces) only opened, for their base calls. The formats themselves are
//! handled by `bio_files`.
//!
//! Unlike molecules, sequences aren't rendered; they're listed in the sidebar, and managed through
//! the open history like other files.

use std::{io, io::ErrorKind, path::Path};

use bio_files::{fasta::Fasta, genbank::GenBank, import_ab1};
use na_seq::Sequence;

use crate::{
    prefs::OpenType,
    state::{MetadataTarget, State},
    util::{handle_err, handle_success},
};

const FASTA_EXTS: [&str; 9] = [
    "fasta", "fa", "fas", "fna", "faa", "ffn", "frn", "fsa", "mpfa",
];
const GENBANK_EXTS: [&str; 5] = ["gb", "gbk", "gbff", "genbank", "gpff"];
/// Read-only.
const AB1_EXTS: [&str; 2] = ["ab1", "abi"];

/// All extensions we can open as sequences. For file dialog filters.
pub fn seq_exts_open() -> Vec<&'static str> {
    FASTA_EXTS
        .iter()
        .chain(GENBANK_EXTS.iter())
        .chain(AB1_EXTS.iter())
        .copied()
        .collect()
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

/// True if `ext` (lower case, without the dot) is a sequence format we can open.
pub fn is_seq_ext(ext: &str) -> bool {
    FASTA_EXTS.contains(&ext) || GENBANK_EXTS.contains(&ext) || AB1_EXTS.contains(&ext)
}

pub struct SeqsLoaded {
    pub records: Vec<Sequence>,
    /// Residue letters that couldn't be represented, e.g. ambiguity codes like N or X, and were
    /// left out.
    pub skipped_residues: usize,
}

/// Load every sequence record in a file. Self-contained, so session restoration can run it on its
/// worker thread.
pub fn load_sequences(path: &Path) -> io::Result<SeqsLoaded> {
    let ext = extension(path);

    let (records, skipped_residues) = if FASTA_EXTS.contains(&ext.as_str()) {
        let fasta = Fasta::load(path)?;
        (fasta.records, fasta.skipped_residues)
    } else if GENBANK_EXTS.contains(&ext.as_str()) {
        let gb = GenBank::load(path)?;
        (gb.records, gb.skipped_residues)
    } else if AB1_EXTS.contains(&ext.as_str()) {
        let records = import_ab1(path)?
            .iter()
            .map(|r| {
                let mut seq = r.to_sequence();
                seq.path = Some(path.to_owned());
                seq
            })
            .collect();
        (records, 0)
    } else {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "Unsupported sequence file extension",
        ));
    };

    if records.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "No sequences found in this file",
        ));
    }

    Ok(SeqsLoaded {
        records,
        skipped_residues,
    })
}

/// Save one sequence, in the format given by the path's extension: FASTA or GenBank.
pub fn save_sequence(seq: &Sequence, path: &Path) -> io::Result<()> {
    let ext = extension(path);
    let records = vec![seq.clone()];

    if FASTA_EXTS.contains(&ext.as_str()) {
        Fasta {
            records,
            ..Default::default()
        }
        .save(path)
    } else if GENBANK_EXTS.contains(&ext.as_str()) {
        GenBank {
            records,
            ..Default::default()
        }
        .save(path)
    } else {
        Err(io::Error::new(
            ErrorKind::InvalidData,
            "Sequences can be saved as FASTA or GenBank files",
        ))
    }
}

/// Prompt for where to save the sequence at index `i`. It's saved once a location is picked; see
/// `ui::util::update_file_dialogs`.
pub fn save_seq_dialog(state: &mut State, i: usize) {
    let Some(seq) = state.sequences.get(i) else {
        return;
    };

    let name: String = seq
        .display_name()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();

    let dialog = &mut state.volatile.dialogs.save_seq;
    dialog.config_mut().default_file_name = format!("{name}.fasta");
    dialog.save_file();

    state.volatile.dialogs.save_seq_i = Some(i);
}

/// The identifier we record in the open history for a file's sequences.
fn history_ident(records: &[Sequence]) -> Option<String> {
    let first = records.first()?.display_name();

    Some(match records.len() {
        1 => first.to_owned(),
        n => format!("{first} (+{})", n - 1),
    })
}

impl State {
    /// Open every sequence in a file, e.g. FASTA or GenBank, adding them to state and the open
    /// history. Reports the result to the user.
    pub fn open_sequences(&mut self, path: &Path) -> io::Result<()> {
        let loaded = load_sequences(path)?;

        let count = loaded.records.len();
        let ident = history_ident(&loaded.records);

        self.volatile.active_seq = Some(self.sequences.len());
        self.sequences.extend(loaded.records);

        self.update_history(path, OpenType::Sequence, ident);
        self.update_save_prefs();

        let fname = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("<unknown>");

        let plural = if count == 1 { "" } else { "s" };
        let msg = format!("Loaded {count} sequence{plural} from {fname}");

        if loaded.skipped_residues > 0 {
            handle_err(
                &mut self.ui,
                format!(
                    "{msg}. Left out {} residue letters that can't be represented, e.g. \
                     ambiguity codes like N or X.",
                    loaded.skipped_residues
                ),
            );
        } else {
            handle_success(&mut self.ui, msg);
        }

        Ok(())
    }

    /// Save the sequence at index `i`. It becomes associated with the new file, e.g. for the
    /// open history.
    pub fn save_sequence(&mut self, i: usize, path: &Path) -> io::Result<()> {
        let Some(seq) = self.sequences.get(i) else {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "No sequence to save",
            ));
        };

        save_sequence(seq, path)?;

        let prev_path = self.sequences[i].path.replace(path.to_owned());
        if let Some(prev) = prev_path
            && prev != path
        {
            self.mark_seq_file_closed_if_unused(&prev);
        }

        let ident = Some(self.sequences[i].display_name().to_owned());
        self.update_history(path, OpenType::Sequence, ident);
        self.update_save_prefs();

        handle_success(
            &mut self.ui,
            format!("Saved sequence to {}", path.display()),
        );

        Ok(())
    }

    /// Close the sequence at index `i`.
    pub fn close_sequence(&mut self, i: usize) {
        if i >= self.sequences.len() {
            eprintln!("Error: Out of bounds when closing a sequence");
            return;
        }

        let seq = self.sequences.remove(i);

        // Indices after this one shift down.
        let shift = |sel: &mut Option<usize>| {
            *sel = match *sel {
                Some(j) if j == i => None,
                Some(j) if j > i => Some(j - 1),
                v => v,
            };
        };

        shift(&mut self.volatile.active_seq);
        shift(&mut self.volatile.dialogs.save_seq_i);

        if let Some(MetadataTarget::Seq(j)) = self.ui.popup.metadata {
            let mut sel = Some(j);
            shift(&mut sel);
            self.ui.popup.metadata = sel.map(MetadataTarget::Seq);
        }
        // The edit rows are keyed to an index, which may now refer to a different sequence.
        if matches!(self.ui.metadata_edit.target, Some(MetadataTarget::Seq(_))) {
            self.ui.metadata_edit.target = None;
            self.ui.metadata_edit.rows.clear();
            self.ui.editing_metadata = false;
        }

        if let Some(path) = &seq.path {
            self.mark_seq_file_closed_if_unused(path);
        }

        self.update_save_prefs();
    }

    /// A file can hold several sequences. Once none from it remain open, don't reopen it next
    /// session.
    fn mark_seq_file_closed_if_unused(&mut self, path: &Path) {
        if self
            .sequences
            .iter()
            .any(|s| s.path.as_deref() == Some(path))
        {
            return;
        }

        for history in &mut self.to_save.open_history {
            if history.type_ == OpenType::Sequence && history.path == *path {
                history.last_session = false;
            }
        }
    }
}
