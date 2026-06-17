//! For saving and loading `ToSave` to disk. We use a binary format to save disk space and time. It's
//! a simple packet-based format; this makes existing save files robust to changes in the struct. Some parts
//! of it will break, but the parts broken are contained in the packet affected.
//!
//! Uses the `.mca` file extension.
///!
///! todo: Adjust the balance of code between this and prefs.rs as required. Or merge into one.
//! or make a new folder moledule containing both.

use std::{
    io, path::Path,
};

use crate::prefs::ToSave;

impl ToSave {
    pub fn save(&self, path: &Path) -> io::Result<()> {}

    pub fn load(path: &Path) -> io::Result<Self> {}
}