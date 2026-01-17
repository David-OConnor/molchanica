// Prevents the Terminal displaying on Windows.
// #![cfg_attr(
//     all(not(debug_assertions), target_os = "windows"),
//     windows_subsystem = "windows"
// )]

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
// Note: To test if it compiles on ARM:
// `rustup target add aarch64-pc-windows-msvc`
// `cargo check --target aarch64-pc-windows-msvc`
// note: Currently getting Clang errors when I attempt htis.

// todo: Consider APBS: https://github.com/Electrostatics/apbs (For a type of surface visulaization?)P

//! [S3 Gemmi link](https://daedalus-mols.s3.us-east-1.amazonaws.com/gemmi.exe)
//! [S3 Geostd link](https://daedalus-mols.s3.us-east-1.amazonaws.com/amber_geostd)

mod bond_inference;
mod docking;
mod download_mols;
mod drawing;
mod file_io;
mod forces;
mod inputs;
mod prefs;
mod render;
mod ribbon_mesh;
mod sa_surface;
mod ui;
mod util;

mod cli;
mod reflection;

mod cam_misc;
mod drawing_wrappers;
mod drug_design;
mod md;
mod mol_alignment;
mod mol_characterization;
mod mol_editor;
mod mol_manip;
mod mol_screening;
mod molecules;
mod orca;
mod pharmacokinetics;
mod pharmacophore;
mod selection;
mod smiles;
mod state;
mod tautomers;
#[cfg(test)]
mod tests;
mod viridis_lut;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{fmt::Display, process::Command, time::Instant};

use bincode::{Decode, Encode};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaFunction;
use dynamics::{ComputationDevice, Integrator, SimBoxInit, params::FfParamSet};
use molecules::{MolType, lipid::load_lipid_templates, nucleic_acid::load_na_templates};
use state::State;

use crate::{render::render, util::handle_err};

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../molchanica.ptx");

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
// For now, we check for differences between to_save and to_save prev, and write to disk
// if they're not equal.
const PREFS_SAVE_INTERVAL: u64 = 20; // seconds

/// The MdModule is owned by `dynamics::ComputationDevice`.
#[cfg(feature = "cuda")]
struct CudaFunctions {
    /// For processing as part of loading electron density data
    pub reflections: Arc<CudaFunction>,
}

// /// This wraps `dyanmics::ComputationDevice`. It's a bit awkard, but for now
// /// allows Dynamics to own ComputationDev with the MD model. We add our additional
// /// models here. Note: The CudaStream is owned by the inner `dynamics::ComputationDevice`.
// enum ComputationDevOuter {
//     Cpu,
//     #[cfg(feature = "cuda")]
//     Gpu((ComputationDevice, Arc<CudaModule>>)),
// }

/// Flags to accomplish things that must be done somewhere with access to `Scene`.
#[derive(Default)]
struct SceneFlags {
    /// Secondary structure
    pub update_ss_mesh: bool,
    /// Solvent-accessible surface.
    pub update_sas_mesh: bool,
    pub update_sas_coloring: bool,
    pub ss_mesh_created: bool,
    pub sas_mesh_created: bool,
    pub make_density_iso_mesh: bool,
    pub clear_density_drawing: bool,
    pub new_density_loaded: bool,
    pub new_mol_loaded: bool,
}

fn main() {
    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;

    #[cfg(feature = "cuda")]
    let (dev, kernel_reflections) = util::get_computation_device();

    // let dev = ComputationDevice::Cpu;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("AVX-512 is available");
        } else if is_x86_feature_detected!("avx") {
            println!("AVX (256-bit) is available");
        } else {
            println!("AVX is not available.");
        }
    }

    println!("Using computing device: {:?}/n", dev);

    // todo: Consider a custom default impl. This is a substitute.
    let mut state = State {
        dev,
        ..Default::default()
    };

    #[cfg(feature = "cuda")]
    if let Some(k) = kernel_reflections {
        state.kernel_reflections = Some(k);
    }

    // todo: Consider if you want this here. Currently required when adding H to a molecule.
    // In release mode, takes 20ms on a fast CPU. (todo: Test on a slow CPU.)
    println!("Loading force field data from Amber lib/dat/frcmod...");
    let start = Instant::now();
    match FfParamSet::new_amber() {
        Ok(f) => {
            state.ff_param_set = f;
            let elapsed = start.elapsed().as_millis();
            println!("Loaded data in {elapsed}ms");
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load Amber force field data: {e:?}"),
            );
        }
    }

    state.load_prefs();

    {
        // Set these UI strings for numerical values up after loading prefs
        state.volatile.md_runtime = state.to_save.num_md_steps as f32 * state.to_save.md_dt;
        state.ui.ph_input = state.to_save.ph.to_string();
        state.ui.md.dt_input = state.to_save.md_dt.to_string();
        state.ui.md.pressure_input = (state.to_save.md_config.pressure_target as u16).to_string();
        state.ui.md.temp_input = (state.to_save.md_config.temp_target as u16).to_string();
        state.ui.md.simbox_pad_input = match state.to_save.md_config.sim_box {
            SimBoxInit::Pad(p) => (p as u16).to_string(),
            SimBoxInit::Fixed(_) => "0".to_string(), // We currently don't use this.
        };

        state.ui.md.langevin_γ = match state.to_save.md_config.integrator {
            Integrator::LangevinMiddle { gamma } => gamma.to_string(),
            _ => "0.".to_string(),
        };
    }

    // We must have loaded prefs prior to this, so we know which file to open.
    state.load_last_opened();

    // todo trouble: It's somewhere around here, saving the inited-from-load atom posits, overwriting
    // todo the previously-saved ones.

    // todo: Workaround to allow us to apply params to the ligand once it's loaded. Unfortunate we have
    // todo to double-load prefs.
    state.load_prefs();

    if let Some(mol) = &state.peptide {
        let posit = state.to_save.per_mol[&mol.common.ident]
            .docking_site
            .site_center;
        state.update_docking_site(posit);
    }

    // todo: Consider if you want this default, and if you also want to add default Lipids etc.
    if !state.ligands.is_empty() {
        state.volatile.active_mol = Some((MolType::Ligand, 0));
    }

    match load_lipid_templates() {
        Ok(t) => {
            state.templates.lipid = t;
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load lipid templates: {e}"),
            );
        }
    }

    match load_na_templates() {
        Ok((dna, rna)) => {
            state.templates.dna = dna;
            state.templates.rna = rna;
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load nucleic acid templates: {e}"),
            );
        }
    }

    // match load_aa_templates() {
    //     Ok(t) => {
    //         state.templates.amino_acid= t;
    //     }
    //     Err(e) => {
    //         handle_err(
    //             &mut state.ui,
    //             format!("Unable to load amino acid templates: {e}"),
    //         );
    //     }
    // }

    if let Ok(out) = Command::new("orca").output() {
        let out = String::from_utf8(out.stdout).unwrap();
        // No simpler way like version?
        if out.contains("This program requires") {
            state.volatile.orca_avail = true;
        }
    };

    render(state);
}
