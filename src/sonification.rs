//! For playing molecular structure as audio.
//!
//! This is a sonification, not a normal-mode solver: each covalent bond is treated as a
//! diatomic oscillator, then transposed into the audible range. The ordering is chemically
//! meaningful: lighter atoms and stronger/shorter bonds produce higher tones.

use std::{f32::consts::TAU, io};

use bio_files::{BondType, md_params::ForceFieldParams};
use na_seq::Element::Hydrogen;
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Source, source::SineWave};

use crate::molecules::common::MoleculeCommon;

const AMU_TO_KG: f32 = 1.660_539e-27;
const KCAL_PER_MOL_A2_TO_N_PER_M: f32 = 0.694_77;
const AUDIO_TRANSPOSITION: f32 = 2.0e-11;
const MIN_FREQ_HZ: f32 = 80.0;
// const MAX_FREQ_HZ: f32 = 2_600.0;
const MAX_FREQ_HZ: f32 = 5_000.0;
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
    pub fn start(
        mol: &MoleculeCommon,
        ff_params: &ForceFieldParams,
        include_h: bool,
    ) -> io::Result<Self> {
        play(mol, ff_params, include_h)
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
pub fn play(
    mol: &MoleculeCommon,
    ff_params: &ForceFieldParams,
    include_h: bool,
) -> io::Result<MoleculeSonification> {
    let freqs = bond_frequencies(mol, ff_params, include_h)?;

    if freqs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "molecule has no parameterized covalent bonds to sonify",
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

fn bond_frequencies(
    mol: &MoleculeCommon,
    ff_params: &ForceFieldParams,
    include_h: bool,
) -> io::Result<Vec<f32>> {
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

        if !is_parameterized_bond(bond.bond_type) {
            continue;
        }

        let ff_type_0 = force_field_type(mol, bond.atom_0)?;
        let ff_type_1 = force_field_type(mol, bond.atom_1)?;
        let bond_params = ff_params
            .get_bond(&(ff_type_0.to_owned(), ff_type_1.to_owned()), true)
            .ok_or_else(|| missing_bond_params(ff_type_0, ff_type_1))?;

        let mass_0 = ff_params
            .mass
            .get(ff_type_0)
            .map(|m| m.mass)
            .unwrap_or_else(|| atom_0.element.atomic_weight() as f32);
        let mass_1 = ff_params
            .mass
            .get(ff_type_1)
            .map(|m| m.mass)
            .unwrap_or_else(|| atom_1.element.atomic_weight() as f32);
        result.push(bond_frequency_hz(mass_0, mass_1, bond_params.k_b));
    }

    Ok(result)
}

fn bond_frequency_hz(mass_0: f32, mass_1: f32, k_b: f32) -> f32 {
    let mass_0 = mass_0.max(1.0);
    let mass_1 = mass_1.max(1.0);
    let reduced_mass_kg = (mass_0 * mass_1 / (mass_0 + mass_1)) * AMU_TO_KG;

    // Amber bond stretching uses U = k_b(r-r0)^2, so the harmonic curvature is 2*k_b.
    let spring_n_per_m = 2.0 * k_b * KCAL_PER_MOL_A2_TO_N_PER_M;
    let freq = (spring_n_per_m / reduced_mass_kg).sqrt() / TAU;

    (freq * AUDIO_TRANSPOSITION).clamp(MIN_FREQ_HZ, MAX_FREQ_HZ)
}

fn is_parameterized_bond(bond_type: BondType) -> bool {
    match bond_type {
        BondType::Dummy | BondType::NotConnected => false,
        _ => true,
    }
}

fn force_field_type(mol: &MoleculeCommon, atom_i: usize) -> io::Result<&str> {
    let atom = mol.atoms.get(atom_i).ok_or_else(|| invalid_bond("atom"))?;
    atom.force_field_type.as_deref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("atom #{} is missing a force-field type", atom.serial_number),
        )
    })
}

fn missing_bond_params(ff_type_0: &str, ff_type_1: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("missing bond-stretch params for {ff_type_0}-{ff_type_1}"),
    )
}

fn invalid_bond(field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("molecule has a bond with an invalid {field}"),
    )
}
