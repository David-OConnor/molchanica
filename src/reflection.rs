#![allow(unused)]

//! For displaying electron density as measured by crytalography and Cryo-EM reflection data. From precomputed
//! data, or Miller indices.

// todo: This is currently a disorganized dumping ground of related data. Organize it,
// todo, move to bio_files as requried, and add cuFFT.

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{io, time::Instant};

use bio_files::{DensityMap, MapHeader, UnitCell, cif_sf::CifStructureFactors};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use dynamics::ComputationDevice;
use ewald::fft3d_c2r;
use graphics::{EngineUpdates, EntityUpdate, Mesh, Scene, Vertex};
#[cfg(feature = "cuda")]
use lin_alg::f32::Vec3 as Vec3F32;
#[cfg(feature = "cuda")]
use lin_alg::f32::{vec3s_from_dev, vec3s_to_dev};
use lin_alg::f64::Vec3;
use mcubes::{MarchingCubes, MeshSide};
use mol_defs::reflection::{DensityPt, DensityRect};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::{drawing::draw_density_surface, render::MESH_DENSITY_SURFACE, state::State, util};

const DIST_TO_ATOMS_SAMPLE_RATIO: usize = 5;

/// The parts of density-map construction that need a compute device (and, on the GPU path, our CUDA
/// kernel). `DensityRect` itself is defined in `mol_defs`, which has no reason to know about either,
/// so these hang off an extension trait here.
pub trait DensityRectExt {
    fn make_densities(
        &self,
        dev: &ComputationDevice,
        #[cfg(feature = "cuda")] kernel: &Option<CudaFunction>,
        atom_posits: &[Vec3],
        cell: &UnitCell,
        dist_thresh: f64,
    ) -> Vec<DensityPt>;
}

impl DensityRectExt for DensityRect {
    /// Convert a DensityRect into (coords, density) structs, which represent density over 3D space,
    /// and can be visualized. It is computationally intensive currently, due to our filter for only
    /// items near the atom coordinates. This may speed up rendering downstream, and declutter the view.
    ///
    /// We assume `atom_posits` has been filtered to not include Hydrogens, for performance reasons.
    fn make_densities(
        &self,
        dev: &ComputationDevice,
        #[cfg(feature = "cuda")] kernel: &Option<CudaFunction>,
        atom_posits: &[Vec3],
        cell: &UnitCell,
        dist_thresh: f64,
    ) -> Vec<DensityPt> {
        println!("Making electron densities...");
        let start = Instant::now();

        // Step vectors along a, b, c.
        let cols = cell.ortho.to_cols();

        // length of one voxel along the a-axis in Å = a / mx
        let step_vec_a = cols.0 * (self.step[0] / cell.a); //  = a_vec / mx
        let step_vec_b = cols.1 * (self.step[1] / cell.b); //  = b_vec / my
        let step_vec_c = cols.2 * (self.step[2] / cell.c); //  = c_vec / mz

        let step_vecs = (step_vec_a, step_vec_b, step_vec_c);

        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);

        let mut triplets = Vec::with_capacity(nx * ny * nz);
        for kz in 0..nz {
            for ky in 0..ny {
                for kx in 0..nx {
                    triplets.push((kx, ky, kz));
                }
            }
        }

        // Convert to f64, and don't use every atom. The latter reduces computation time, roughly
        // by a factor of `SAMPLE_RATIO`.
        let atom_posits_sample: Vec<Vec3> = atom_posits
            .iter()
            .enumerate()
            .filter(|(i, _)| i.is_multiple_of(DIST_TO_ATOMS_SAMPLE_RATIO))
            .map(|(_, a)| *a)
            .collect();

        // Note: We get a big speedup from both Rayon, and then GPU on top of that. e.g.:
        // CPU without rayon: 20,000ms
        // CPU, with rayon (No SIMD): 780ms
        // GPU: 54ms
        let out = match &dev {
            ComputationDevice::Cpu => make_densities_inner(
                self,
                triplets,
                &atom_posits_sample,
                step_vecs,
                dist_thresh,
                nx,
                ny,
            ),
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu(stream) => make_densities_inner_gpu(
                self,
                stream,
                // Assume Some if on Device::Gpu.
                kernel.as_ref().unwrap(),
                triplets,
                &atom_posits_sample,
                step_vecs,
                dist_thresh.powi(2),
                nx,
                ny,
            ),
        };

        let elapsed = start.elapsed().as_millis();
        println!("Complete, in {elapsed} ms");

        out
    }
}

#[cfg(feature = "cuda")]
/// Separate helper, to isolate from the CPU version.
fn make_densities_inner_gpu(
    rect: &DensityRect,
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    triplets: Vec<(usize, usize, usize)>,
    atom_posits: &[Vec3],
    step_vecs: (Vec3, Vec3, Vec3),
    dist_thresh: f64,
    nx: usize,
    ny: usize,
) -> Vec<DensityPt> {
    let n = triplets.len();
    let n_atom_posits = atom_posits.len();

    let mut coords_gpu = {
        let v = vec![Vec3F32::new_zero(); n];
        vec3s_to_dev(stream, &v)
    };

    let mut densities_gpu = stream.clone_htod(&vec![0.0f32; n]).unwrap();

    let triplets_gpu = {
        let mut trip_flat = Vec::with_capacity(n * 3);
        for (x, y, z) in triplets {
            trip_flat.push(x as u32);
            trip_flat.push(y as u32);
            trip_flat.push(z as u32);
        }
        stream.clone_htod(&trip_flat).unwrap()
    };

    let atom_posits_gpu = {
        let p_f32: Vec<Vec3F32> = atom_posits.iter().map(|a| (*a).into()).collect();
        vec3s_to_dev(stream, &p_f32)
    };

    // Convert vec3s to f32.
    let step_vecs_0: Vec3F32 = step_vecs.0.into();
    let step_vecs_1: Vec3F32 = step_vecs.1.into();
    let step_vecs_2: Vec3F32 = step_vecs.2.into();
    let origin_f32: Vec3F32 = rect.origin_cart.into();

    let data_gpu = stream.clone_htod(&rect.data).unwrap();

    let dist_thresh = dist_thresh as f32;

    let cfg = LaunchConfig::for_num_elems(n as u32);

    let mut launch_args = stream.launch_builder(&kernel);

    launch_args.arg(&mut coords_gpu);
    launch_args.arg(&mut densities_gpu);
    //
    launch_args.arg(&triplets_gpu);
    launch_args.arg(&atom_posits_gpu);
    launch_args.arg(&data_gpu);
    //
    launch_args.arg(&step_vecs_0);
    launch_args.arg(&step_vecs_1);
    launch_args.arg(&step_vecs_2);
    launch_args.arg(&origin_f32);
    //
    launch_args.arg(&dist_thresh);
    launch_args.arg(&nx);
    launch_args.arg(&ny);
    launch_args.arg(&n);
    launch_args.arg(&n_atom_posits);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
    let coords = vec3s_from_dev(stream, &coords_gpu);
    let densities = stream.clone_dtoh(&densities_gpu).unwrap();

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push(DensityPt {
            coords: coords[i].into(),
            density: densities[i] as f64,
        })
    }

    result
}

/// Separate helper, to isolate from the GPU version.
fn make_densities_inner(
    rect: &DensityRect,
    triplets: Vec<(usize, usize, usize)>,
    atom_posits: &[Vec3],
    step_vecs: (Vec3, Vec3, Vec3),
    dist_thresh: f64,
    nx: usize,
    ny: usize,
) -> Vec<DensityPt> {
    let dist_thresh_sq = dist_thresh * dist_thresh;

    // Note: We get a big speedup from using rayon here. For example, 200ms vs 5s, or 2.5s vs 70s
    // for a larger file. (9950x CPU)
    triplets
        .par_iter()
        .map(|&(kx, ky, kz)| {
            // Linear index in rect.data  (z → y → x fastest)
            let i_data = (kz * ny + ky) * nx + kx;
            let mut density = rect.data[i_data] as f64;

            // Cartesian center of this voxel
            let coords = rect.origin_cart
                + step_vecs.0 * kx as f64
                + step_vecs.1 * ky as f64
                + step_vecs.2 * kz as f64;

            // todo: Too slow, but can work for now.
            // We don't want to set up density points not near the atom coordinates.
            let mut nearest_dist_sq = f64::MAX;
            for p in atom_posits {
                let dist_sq = (*p - coords).magnitude_squared();
                if dist_sq < nearest_dist_sq {
                    nearest_dist_sq = dist_sq;
                }
            }

            if nearest_dist_sq > dist_thresh_sq {
                // We set density to 0, vice removing the coordinates; our marching cubes
                // algorithm requires a regular grid, with no absent values.
                density = 0.;
            }

            DensityPt { coords, density }
        })
        .collect()
}

/// Populate the electron-density mesh (isosurface). This assumes the density_rect is already set up.
pub fn make_density_mesh(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates) {
    let Some(mol) = state
        .peptide_for_tools_i()
        .and_then(|i| state.peptides.get(i))
    else {
        return;
    };
    let Some(rect) = &mol.density_rect else {
        return;
    };
    let Some(density_pts) = &mol.elec_density else {
        return;
    };
    let Some(map) = &mol.density_map else {
        return;
    };

    // Our marching cubes function requires Z to be the fast axis; convert to that instead of X fastest.
    let dims = (rect.dims[0], rect.dims[1], rect.dims[2]); // (nx, ny, nz)
    let step = [rect.step[0], rect.step[1], rect.step[2]]; // (nx, ny, nz)

    let size = (
        (step[0] * dims.0 as f64) as f32, // Δx * nx  (Å)
        (step[1] * dims.1 as f64) as f32,
        (step[2] * dims.2 as f64) as f32,
    );

    let sampling_interval = (dims.0 as f32, dims.1 as f32, dims.2 as f32);

    let density: Vec<_> = density_pts.iter().map(|p| p.density as f32).collect();
    match MarchingCubes::new(
        dims,
        size,
        sampling_interval,
        rect.origin_cart.into(),
        density,
        state.ui.density_iso_level,
    ) {
        Ok(mc) => {
            let mesh = mc.generate(MeshSide::OutsideOnly);

            // Convert from `mcubes::Mesh` to `graphics::Mesh`.
            let vertices = mesh
                .vertices
                .iter()
                .map(|v| Vertex::new(v.posit.to_arr(), v.normal))
                .collect();

            scene.meshes[MESH_DENSITY_SURFACE] = Mesh {
                vertices,
                indices: mesh.indices,
                material: 0,
            };
            updates.meshes = true;

            if !state.ui.visibility.hide_density_surface {
                draw_density_surface(&mut scene.entities, state, updates);
            }

            // _updates.entities.push_class(EntityClass::SaSurface as u32);
        }
        Err(e) => util::handle_err(&mut state.ui, e.to_string()),
    }
}

// todo: Code below represents a local implementation of creating a map from 2fo-fc
//----------------------------------------------

fn wrap_idx(i: i32, n: usize) -> usize {
    let n_i32 = n as i32;
    let m = i % n_i32;
    if m < 0 {
        (m + n_i32) as usize
    } else {
        m as usize
    }
}

fn inv_perm(p: [usize; 3]) -> [usize; 3] {
    let mut q = [0; 3];
    q[p[0]] = 0;
    q[p[1]] = 1;
    q[p[2]] = 2;

    q
}

fn lin3(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + nx * (j + ny * k)
}

fn idx_xfast(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize, nz: usize) -> usize {
    // crystal order (X,Y,Z) with X contiguous, Z slowest
    ix + nx * (iy + ny * iz)
}

fn idx_zfast(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize, nz: usize) -> usize {
    // crystal order (X,Y,Z) with Z contiguous, X slowest
    iz + nz * (iy + ny * ix)
}

fn idx_file(i_f: usize, j_f: usize, k_f: usize, nx: usize, ny: usize) -> usize {
    // file order (fast, medium, slow) with FAST contiguous (as before)
    i_f + nx * (j_f + ny * k_f)
}

/// Load electron density from Structure Factors (reflections) data, using a FFT. This is currently broken; we use
/// Gemmi instead.
// todo: This currently isn't working.
// todo: Use cuFFT if in GPU mode.
pub fn density_map_from_sf(
    sf: &CifStructureFactors,
    planner: &mut FftPlanner<f32>,
) -> io::Result<DensityMap> {
    println!("Computing electron density from mmCIF 2fo-fc data...");
    let start = Instant::now();

    let (nx, ny, nz) = (
        sf.header.mx as usize,
        sf.header.my as usize,
        sf.header.mz as usize,
    );

    let perm_f2c = [
        (sf.header.mapc - 1) as usize,
        (sf.header.mapr - 1) as usize,
        (sf.header.maps - 1) as usize,
    ];

    // Reciprocal grid in CRYSTAL order with X-fast layout ---
    let mut data_k = vec![Complex::<f32>::new(0.0, 0.0); nx * ny * nz];

    for r in &sf.miller_indices {
        let c = if let (Some(re), Some(im)) = (r.re, r.im) {
            Complex::new(re, im)
        } else {
            let Some(amp) = r.amp else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Miller index out of bounds",
                ));
            };
            let Some(phase) = r.phase else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Miller index out of bounds",
                ));
            };
            Complex::from_polar(amp, phase)
        };

        // Crystal grid indices
        let u = wrap_idx(r.h, nx);
        let v = wrap_idx(r.k, ny);
        let w = wrap_idx(r.l, nz);

        // place at (u,v,w) and its conjugate at (-u,-v,-w) in X-fast layout
        let i0 = idx_xfast(u, v, w, nx, ny, nz);
        data_k[i0] = c;

        let u2 = wrap_idx(-r.h, nx);
        let v2 = wrap_idx(-r.k, ny);
        let w2 = wrap_idx(-r.l, nz);

        let i1 = idx_xfast(u2, v2, w2, nx, ny, nz);

        if i1 != i0 && data_k[i1] == Complex::new(0.0, 0.0) {
            data_k[i1] = c.conj();
        }
    }

    // --- inverse FFT on X-fast layout; dims are crystal (mx,my,mz) ---
    let mut rho_crystal = fft3d_c2r(&mut data_k, (nx, ny, nz), planner);

    // If your FFT is unscaled, divide by N
    let nvox = (nx * ny * nz) as f32;
    for v in &mut rho_crystal {
        *v /= nvox;
    }

    // --- remap to FILE order buffer if you want DensityMap.data in file order ---
    // let mut rho_file = vec![0.; mx * my * mz];
    let mut density_data = vec![0.; nz * ny * nx];

    // file indices → crystal indices → pull from rho_crystal (Z-fast) → store in file buffer
    // todo: Experiment with order here too

    for i_f in 0..nx {
        for j_f in 0..ny {
            for k_f in 0..nz {
                let mut ic = [0usize; 3];
                ic[perm_f2c[0]] = i_f; // crystal X index
                ic[perm_f2c[1]] = j_f; // crystal Y index
                ic[perm_f2c[2]] = k_f; // crystal Z index

                let src_idx = idx_xfast(ic[0], ic[1], ic[2], nx, ny, nz);
                let dst_idx = idx_file(i_f, j_f, k_f, nx, ny);

                density_data[dst_idx] = rho_crystal[src_idx];
            }
        }
    }

    // stats on rho_file
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;

    for &x in &density_data {
        let xd = x as f64;
        sum += xd;
        sum2 += xd * xd;
        if x < min_v {
            min_v = x;
        }
        if x > max_v {
            max_v = x;
        }
    }
    let mean = (sum / (nvox as f64)) as f32;

    let hdr = MapHeader {
        inner: sf.header.clone(),
        nx: nx as i32,
        ny: ny as i32,
        nz: nz as i32,
        mode: 2, // f32

        dmin: min_v,
        dmax: max_v,
        dmean: mean,
    };

    let elapsed = start.elapsed().as_millis();
    println!("Complete in {elapsed} ms");

    // let density_data = xfast_to_zfast(&density_data, nx, ny, nz);

    DensityMap::new(hdr, density_data)
}

fn xfast_to_zfast<T: Copy>(src: &[T], nx: usize, ny: usize, nz: usize) -> Vec<T> {
    // src is X-fast: i + nx*(j + ny*k)
    // dst is Z-fast: k + nz*(j + ny*i)
    let mut dst = vec![src[0]; nx * ny * nz];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let sx = i + nx * (j + ny * k);
                let sz = k + nz * (j + ny * i);
                dst[sz] = src[sx];
            }
        }
    }
    dst
}

fn zfast_to_xfast<T: Copy>(src: &[T], nx: usize, ny: usize, nz: usize) -> Vec<T> {
    // inverse of the above
    let mut dst = vec![src[0]; nx * ny * nz];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let sx = k + nz * (j + ny * i);
                let sz = i + nx * (j + ny * k);
                dst[sz] = src[sx];
            }
        }
    }
    dst
}
