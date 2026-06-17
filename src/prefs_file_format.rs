//! For saving and loading `ToSave` to disk. We use a binary format to save disk space and time. It's
//! a simple packet-based format; this makes existing save files robust to changes in the struct. Some parts
//! of it will break, but the parts broken are contained in the packet affected.
//!
//! Packets can be in any order, and there can be any number of packets.
//!
//! Uses the `.mca` file extension.
///!
///! todo: Adjust the balance of code between this and prefs.rs as required. Or merge into one.
//! or make a new folder module containing both.

use std::{
    io, path::Path,
};

use crate::prefs::ToSave;

/// A sanity check: The start byte will always be this.
const MCA_START_BYTE: u8 = 0x69;

const MCA_HEADER_SIZE: usize = 20; // bytes. todo: A/R.
const PACKET_HEADER_SIZE: usize = 20; // bytes. todo: A/R.

#[derive(Clone, Copy PartialEq
)]
#[repr(
    u8
)]
/// The repr is the byte which uniquely identifies the packet type. These
/// loosely correspond to `ToSave` fields, but may diverse as required.
enum PacketType {
    PerMol = 0,
    OpenHistory = 1,
    ControlScheme = 2,
    // todo: etc.
}

/// One at the top of the file.
pub struct McaHeader {
    /// We don't expect to exceed 255 packets, so u8 is fine.
    pub num_packets: u8,
}

impl McaHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = vec![0; MCA_HEADER_SIZE];
        result[0] = MCA_START_BYTE;

        result
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self {}>
}

/// One per packet.
pub struct PacketHeader {
    packet_type: PacketType,
}

impl PacketHeader {
    pub fn to_bytes(&self) -> Vec<u8> {}

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {

    }
}

impl ToSave {
    pub fn save(&self, path: &Path) -> io::Result<()> {}

    pub fn load(path: &Path) -> io::Result<Self> {}
}