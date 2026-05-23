//! For playing molecular structure as audio.
//!
//! This is a sonification, not a normal-mode solver: each covalent bond is treated as a
//! diatomic oscillator, then transposed into the audible range. The ordering is chemically
//! meaningful: lighter atoms and stronger/shorter bonds produce higher tones.

use std::io;

use bio_files::BondType;
use na_seq::Element::Hydrogen;
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Source, source::SineWave};

use crate::molecules::common::MoleculeCommon;

const BASE_FREQ_HZ: f32 = 2_200.0;
const MIN_FREQ_HZ: f32 = 80.0;
const MAX_FREQ_HZ: f32 = 2_600.0;
const VOLUME: f32 = 0.16;

/// Playback handle for one molecule.
///
/// Audio stops when this is dropped. Call [`Self::stop`] if you want an explicit stop.
pub struct MoleculeSonification {
    stream: Option<MixerDeviceSink>,
    voice_count: usize,
}

impl MoleculeSonification {
    /// Start playing audio/harmonies from all covalent bonds in a molecule.
    pub fn start(mol: &MoleculeCommon, include_h: bool) -> io::Result<Self> {
        play(mol, include_h)
    }

    /// Stop playback. Calling this more than once is harmless.
    pub fn stop(&mut self) {
        self.stream.take();
    }

    pub fn is_playing(&self) -> bool {
        self.stream.is_some()
    }

    pub fn voice_count(&self) -> usize {
        self.voice_count
    }
}

/// Start playing audio/harmonies from all covalent bonds in a molecule.
///
/// Set `include_h` to false to omit bonds to hydrogen, which otherwise tend to sit at
/// the high end of the sonified pitch range.
pub fn play(mol: &MoleculeCommon, include_h: bool) -> io::Result<MoleculeSonification> {
    let freqs = bond_frequencies(mol, include_h)?;

    if freqs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "molecule has no covalent bonds to sonify",
        ));
    }

    let mut stream = DeviceSinkBuilder::open_default_sink().map_err(io::Error::other)?;
    stream.log_on_drop(false);

    let amplitude = VOLUME / (freqs.len() as f32).sqrt();
    for freq in &freqs {
        stream.mixer().add(SineWave::new(*freq).amplify(amplitude));
    }

    Ok(MoleculeSonification {
        stream: Some(stream),
        voice_count: freqs.len(),
    })
}

fn bond_frequencies(mol: &MoleculeCommon, include_h: bool) -> io::Result<Vec<f32>> {
    let mut result = Vec::with_capacity(mol.bonds.len());

    for bond in &mol.bonds {
        let atom_0 = mol
            .atoms
            .get(bond.atom_0)
            .ok_or_else(|| invalid_bond("atom_0"))?;
        let atom_1 = mol
            .atoms
            .get(bond.atom_1)
            .ok_or_else(|| invalid_bond("atom_1"))?;

        if !include_h && (atom_0.element == Hydrogen || atom_1.element == Hydrogen) {
            continue;
        }

        let Some(bond_order) = effective_bond_order(bond.bond_type) else {
            continue;
        };

        let Some(bond_len) = bond_length(mol, bond.atom_0, bond.atom_1) else {
            return Err(invalid_bond("position"));
        };

        result.push(bond_frequency_hz(
            atom_0.element.atomic_weight() as f32,
            atom_1.element.atomic_weight() as f32,
            bond_len,
            bond_order,
        ));
    }

    Ok(result)
}

fn bond_length(mol: &MoleculeCommon, atom_0: usize, atom_1: usize) -> Option<f32> {
    let posit_0 = mol
        .atom_posits
        .get(atom_0)
        .or_else(|| mol.atoms.get(atom_0).map(|a| &a.posit))?;
    let posit_1 = mol
        .atom_posits
        .get(atom_1)
        .or_else(|| mol.atoms.get(atom_1).map(|a| &a.posit))?;

    Some((*posit_0 - *posit_1).magnitude() as f32)
}

fn bond_frequency_hz(mass_0: f32, mass_1: f32, bond_len: f32, bond_order: f32) -> f32 {
    let mass_0 = mass_0.max(1.0);
    let mass_1 = mass_1.max(1.0);
    let reduced_mass = mass_0 * mass_1 / (mass_0 + mass_1);
    let bond_len = bond_len.max(0.6);

    let stiffness_proxy = bond_order / bond_len.powi(3);
    let freq = BASE_FREQ_HZ * (stiffness_proxy / reduced_mass).sqrt();

    freq.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ)
}

fn effective_bond_order(bond_type: BondType) -> Option<f32> {
    match bond_type {
        BondType::Single | BondType::PolymericLink | BondType::Unknown => Some(1.0),
        BondType::Amide => Some(1.3),
        BondType::Aromatic | BondType::Delocalized => Some(1.5),
        BondType::Double => Some(2.0),
        BondType::Triple => Some(3.0),
        BondType::Quadruple => Some(4.0),
        BondType::Dummy | BondType::NotConnected => None,
    }
}

fn invalid_bond(field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("molecule has a bond with an invalid {field}"),
    )
}
