//! For playing molecular structure as audio.
//!
//! This is a sonification, not a normal-mode solver: each covalent bond is treated as a
//! diatomic oscillator, then transposed into the audible range. The ordering is chemically
//! meaningful: lighter atoms and stronger/shorter bonds produce higher tones.

use std::io;

use bio_files::{BondType, md_params::ForceFieldParams};
use mol_defs::molecules::{MolType, common::MoleculeCommon};
use na_seq::Element::Hydrogen;
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Source, source::SineWave};
use crate::state::State;
use crate::util;
use crate::util::handle_err;

const AUDIO_TRANSPOSITION_FROM_HZ: f64 = 2.0e-11;
const PS_INV_TO_HZ: f64 = 1.0e12;
const MIN_FREQ_HZ: f32 = 80.0;
const MAX_FREQ_HZ: f32 = 5_000.0;
const VOLUME: f32 = 0.08;

const SONIFICATION_INCLUDE_H: bool = true;

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
        println!(
            "sonification: played {:.2} Hz | bond {:.3} ps^-1 | atoms #{} {} - #{} {} | k_b {:.3} kcal/(mol A^2) | r_0 {:.3} A | masses {:.3}, {:.3} amu",
            freq.audio_freq_hz,
            freq.bond_freq_ps,
            freq.atom_serial_0,
            freq.ff_type_0,
            freq.atom_serial_1,
            freq.ff_type_1,
            freq.k_b,
            freq.r_0,
            freq.mass_0,
            freq.mass_1,
        );

        stream
            .mixer()
            .add(SineWave::new(freq.audio_freq_hz).amplify(amplitude));
    }

    Ok(MoleculeSonification {
        stream: Some(stream),
        voice_count: freqs.len(),
    })
}

struct BondFrequency {
    /// ps^-1
    bond_freq_ps: f64,
    /// Hz
    audio_freq_hz: f32,
    atom_serial_0: u32,
    atom_serial_1: u32,
    ff_type_0: String,
    ff_type_1: String,
    k_b: f32,
    r_0: f32,
    mass_0: f32,
    mass_1: f32,
}

fn bond_frequencies(
    mol: &MoleculeCommon,
    ff_params: &ForceFieldParams,
    include_h: bool,
) -> io::Result<Vec<BondFrequency>> {
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
            .unwrap_or_else(|| atom_0.element.atomic_weight());

        let mass_1 = ff_params
            .mass
            .get(ff_type_1)
            .map(|m| m.mass)
            .unwrap_or_else(|| atom_1.element.atomic_weight());

        let bond_freq_ps = util::bond_freq(bond_params.k_b, mass_0, mass_1);
        let audio_freq_hz = map_bond_freq_to_audio_freq(bond_freq_ps);

        result.push(BondFrequency {
            bond_freq_ps,
            audio_freq_hz,
            atom_serial_0: atom_0.serial_number,
            atom_serial_1: atom_1.serial_number,
            ff_type_0: ff_type_0.to_string(),
            ff_type_1: ff_type_1.to_string(),
            k_b: bond_params.k_b,
            r_0: bond_params.r_0,
            mass_0,
            mass_1,
        });
    }

    Ok(result)
}

/// Adjusts the output frequency of a bond to map suitably to human hearing. The input
/// bond frequency is in ps^-1. The output frequency is in Hz.
fn map_bond_freq_to_audio_freq(freq_ps: f64) -> f32 {
    (freq_ps * PS_INV_TO_HZ * AUDIO_TRANSPOSITION_FROM_HZ)
        .clamp(f64::from(MIN_FREQ_HZ), f64::from(MAX_FREQ_HZ)) as f32
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

pub struct PlayingAudio {
    pub mol_type: MolType,
    pub i_mol: usize,
    pub _handle: MoleculeSonification,
}

impl PlayingAudio {
    pub fn new(mol_type: MolType, i_mol: usize, handle: MoleculeSonification) -> Self {
        Self {
            mol_type,
            i_mol,
            _handle: handle,
        }
    }

    pub fn is_for(&self, mol_type: MolType, i_mol: usize) -> bool {
        self.mol_type == mol_type && self.i_mol == i_mol
    }
}

impl StateVolatile {
    pub fn is_playing_audio_for(&self, mol_type: MolType, i_mol: usize) -> bool {
        self.playing_audio
            .as_ref()
            .is_some_and(|audio| audio.is_for(mol_type, i_mol))
    }

    pub fn update_playing_audio_after_close(&mut self, mol_type: MolType, i_mol: usize) {
        let Some(audio) = &mut self.playing_audio else {
            return;
        };

        if audio.mol_type != mol_type {
            return;
        }

        if audio.i_mol == i_mol {
            self.playing_audio = None;
        } else if audio.i_mol > i_mol {
            audio.i_mol -= 1;
        }
    }
}

pub fn toggle_audio(state: &mut State, mol_type: MolType, i_mol: usize) {
    if state.volatile.is_playing_audio_for(mol_type, i_mol) {
        state.volatile.playing_audio = None;
        return;
    }

    let (mol, ff_params) = match crate::ui::sidebar::mol_picker::sonification_input(state, mol_type, i_mol) {
        Ok(input) => input,
        Err(e) => {
            handle_err(&mut state.ui, e);
            return;
        }
    };

    match sonification::play(&mol, &ff_params, crate::ui::sidebar::SONIFICATION_INCLUDE_H) {
        Ok(handle) => {
            state.volatile.playing_audio =
                Some(sonification::PlayingAudio::new(mol_type, i_mol, handle));
        }
        Err(e) => handle_err(&mut state.ui, format!("Unable to play molecule audio: {e}")),
    }
}
