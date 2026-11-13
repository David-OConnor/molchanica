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
use graphics::{EngineUpdates, Mesh, Scene, Vertex};
#[cfg(feature = "cuda")]
use lin_alg::f32::Vec3 as Vec3F32;
use lin_alg::f64::{Vec3, Vec3x4, pack_vec3x4};
use mcubes::{MarchingCubes, MeshSide};
use mol_defs::reflection::{DensityPt, DensityRect};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::{
    drawing::peptide::draw_density_surface, render::MESH_DENSITY_SURFACE, state::State, util,
};

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

        // Note: We get a big speedup from both Rayon, and then GPU on top of that. e.g.:
        // CPU without rayon: 20,000ms
        // CPU, with rayon (No SIMD): 780ms
        // GPU: 54ms
        #[cfg(feature = "cuda")]
        if let ComputationDevice::Gpu(stream) = dev
            && let Some(kernel) = kernel
            && !triplets.is_empty()
            && !atom_posits.is_empty()
        {
            match make_densities_inner_gpu(
                self,
                stream,
                kernel,
                &triplets,
                atom_posits,
                step_vecs,
                dist_thresh.powi(2),
                nx,
                ny,
            ) {
                Ok(out) => return out,
                Err(e) => eprintln!("Density GPU computation failed; using CPU: {e}"),
            }
        }

        let out = make_densities_inner(self, triplets, atom_posits, step_vecs, dist_thresh, nx, ny);

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
    triplets: &[(usize, usize, usize)],
    atom_posits: &[Vec3],
    step_vecs: (Vec3, Vec3, Vec3),
    dist_thresh: f64,
    nx: usize,
    ny: usize,
) -> Result<Vec<DensityPt>, cudarc::driver::DriverError> {
    let n = triplets.len();
    let n_atom_posits = atom_posits.len();

    let mut coords_gpu = stream.alloc_zeros::<f32>(n * 3)?;
    let mut densities_gpu = stream.alloc_zeros::<f32>(n)?;

    let triplets_gpu = {
        let mut trip_flat = Vec::with_capacity(n * 3);
        for &(x, y, z) in triplets {
            trip_flat.push(x as u32);
            trip_flat.push(y as u32);
            trip_flat.push(z as u32);
        }
        stream.clone_htod(&trip_flat)?
    };

    let atom_posits_gpu = {
        let p_f32: Vec<f32> = atom_posits
            .iter()
            .flat_map(|a| [a.x as f32, a.y as f32, a.z as f32])
            .collect();
        stream.clone_htod(&p_f32)?
    };

    // Convert vec3s to f32.
    let step_vecs_0: Vec3F32 = step_vecs.0.into();
    let step_vecs_1: Vec3F32 = step_vecs.1.into();
    let step_vecs_2: Vec3F32 = step_vecs.2.into();
    let origin_f32: Vec3F32 = rect.origin_cart.into();

    let data_gpu = stream.clone_htod(&rect.data)?;

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

    unsafe { launch_args.launch(cfg) }?;

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
    let coords = stream.clone_dtoh(&coords_gpu)?;
    let densities = stream.clone_dtoh(&densities_gpu)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push(DensityPt {
            coords: Vec3::new(
                coords[3 * i] as f64,
                coords[3 * i + 1] as f64,
                coords[3 * i + 2] as f64,
            ),
            density: densities[i] as f64,
        })
    }

    Ok(result)
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
    let (packed_atoms, last_lanes) = pack_vec3x4(atom_posits);

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

            let near_atom = packed_atoms.iter().enumerate().any(|(i, p)| {
                let lanes = if i + 1 == packed_atoms.len() {
                    last_lanes
                } else {
                    4
                };
                let dist_sq = (*p - Vec3x4::splat(coords)).magnitude_squared().to_array();

                dist_sq[..lanes].iter().any(|&d| d <= dist_thresh_sq)
            });

            if !near_atom {
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

    // DensityRect and marching cubes both use X-fast storage.
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

/// P1 structure factors to a full-cell map. Other space groups require Gemmi expansion.
pub fn density_map_from_sf(
    sf: &CifStructureFactors,
    planner: &mut FftPlanner<f32>,
) -> io::Result<DensityMap> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    if sf.header.ispg != 1 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Use Gemmi to expand space-group symmetry",
        ));
    }
    let dimensions = [sf.header.mx, sf.header.my, sf.header.mz];
    if dimensions.iter().any(|&n| n <= 0) {
        return Err(invalid("Invalid reflection grid dimensions"));
    }
    let [nx, ny, nz] = dimensions.map(|n| n as usize);
    let nvox = nx
        .checked_mul(ny)
        .and_then(|n| n.checked_mul(nz))
        .ok_or_else(|| invalid("Reflection grid too large"))?;
    let axes = [sf.header.mapc, sf.header.mapr, sf.header.maps];
    let mut sorted_axes = axes;
    sorted_axes.sort();
    if sorted_axes != [1, 2, 3] {
        return Err(invalid("Invalid map axis permutation"));
    }
    let perm = axes.map(|a| (a - 1) as usize);
    let volume = sf.header.cell.ortho.determinant().abs();
    if !volume.is_finite() || volume <= 0. {
        return Err(invalid("Invalid unit-cell volume"));
    }

    // Ewald expects a Z-fast half spectrum; crystallographic phases have the opposite sign.
    let nzc = nz / 2 + 1;
    let mut data_k = vec![Complex::<f32>::new(0., 0.); nx * ny * nzc];
    for r in &sf.miller_indices {
        if [r.h, r.k, r.l]
            .into_iter()
            .zip([nx, ny, nz])
            .any(|(h, n)| h.unsigned_abs() as usize > n / 2)
        {
            return Err(invalid("Reflection lies outside the Fourier grid"));
        }
        let c = match (r.re, r.im, r.amp, r.phase) {
            (Some(re), Some(im), _, _) => Complex::new(re, im),
            (_, _, Some(amp), Some(phase)) => Complex::from_polar(amp, phase),
            _ => return Err(invalid("Missing reflection amplitude or phase")),
        };
        if !c.re.is_finite() || !c.im.is_finite() {
            return Err(invalid("Non-finite reflection coefficient"));
        }
        let (x, y, z) = (wrap_idx(r.h, nx), wrap_idx(r.k, ny), wrap_idx(r.l, nz));
        let (mx, my, mz) = ((nx - x) % nx, (ny - y) % ny, (nz - z) % nz);
        if z < nzc {
            data_k[(x * ny + y) * nzc + z] = c.conj();
        }
        if mz < nzc {
            data_k[(mx * ny + my) * nzc + mz] = c;
        }
    }
    let rho = fft3d_c2r(&mut data_k, (nx, ny, nz), planner);
    let crystal_dims = [nx, ny, nz];
    let file_dims = perm.map(|a| crystal_dims[a]);
    let density_data: Vec<f32> = (0..nvox)
        .into_par_iter()
        .map(|i| {
            let file = [
                i % file_dims[0],
                (i / file_dims[0]) % file_dims[1],
                i / (file_dims[0] * file_dims[1]),
            ];
            let mut crystal = [0; 3];
            for axis in 0..3 {
                crystal[perm[axis]] = file[axis];
            }

            (rho[(crystal[0] * ny + crystal[1]) * nz + crystal[2]] as f64 / volume) as f32
        })
        .collect();
    let mut inner = sf.header.clone();
    inner.nxstart = 0;
    inner.nystart = 0;
    inner.nzstart = 0;
    inner.xorigin = None;
    inner.yorigin = None;
    inner.zorigin = None;
    let hdr = MapHeader {
        inner,
        nx: file_dims[0] as i32,
        ny: file_dims[1] as i32,
        nz: file_dims[2] as i32,
        mode: 2,
        dmin: density_data.iter().copied().fold(f32::INFINITY, f32::min),
        dmax: density_data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        dmean: (density_data.iter().map(|&x| x as f64).sum::<f64>() / nvox as f64) as f32,
    };

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
