//! Wiring pocket surface meshes into the render engine.
//!
//! `Pocket` and its mesh live in `mol_defs`; getting that mesh into the engine's global mesh list,
//! and recolouring it, needs our scene layout and entity classes. Those parts live here, as an
//! extension trait so pocket code reads the same as it did before the split.

use graphics::{EngineUpdates, Mesh};
use mol_defs::{molecules::pocket::Pocket, sfc_mesh::MeshColoring};

use crate::{
    drawing::EntityClass,
    render::MESH_POCKET_START,
    sfc_mesh::{apply_mesh_colors, get_mesh_colors},
};

pub trait PocketRender {
    fn regen_mesh_vol(&mut self, scene_meshes: &mut Vec<Mesh>, updates: &mut EngineUpdates);
    fn reset_post_manip(
        &mut self,
        scene_meshes: &mut Vec<Mesh>,
        coloring: MeshColoring,
        updates: &mut EngineUpdates,
    );
}

impl PocketRender for Pocket {
    /// Run this, for example, after moving the molecule. Move the atoms in the same manner
    /// as with other molecule types, then run this to synchronize.
    ///
    /// Also rebuilds the mesh.
    fn regen_mesh_vol(&mut self, scene_meshes: &mut Vec<Mesh>, updates: &mut EngineUpdates) {
        self.rebuild_mesh_vol();

        let mesh_i = MESH_POCKET_START + self.mesh_i_rel;
        if mesh_i == scene_meshes.len() {
            scene_meshes.push(Mesh::default());
        } else if mesh_i > scene_meshes.len() {
            eprintln!(
                "Error: Unable to find the global mesh at {mesh_i} when assigning it for this pocket"
            );
            return;
        }

        scene_meshes[mesh_i] = self.surface_mesh.clone();

        updates.meshes = true;
        updates.entities.push_class(EntityClass::Pocket as u32);
    }

    /// Run this after a move. Resets local positions, and rebuilds everything else (volume, spheres,
    /// mesh etc, and updates the engine's meshes)
    fn reset_post_manip(
        &mut self,
        scene_meshes: &mut Vec<Mesh>,
        coloring: MeshColoring,
        updates: &mut EngineUpdates,
    ) {
        self.rebuild_spheres();
        self.regen_mesh_vol(scene_meshes, updates);

        let color = get_mesh_colors(&self.surface_mesh, &self.common, coloring, updates);
        apply_mesh_colors(&mut self.surface_mesh, &color);

        // We handle pushing this mesh in the regen method above.
        let mesh_i = MESH_POCKET_START + self.mesh_i_rel;
        if mesh_i >= scene_meshes.len() {
            eprintln!(
                "Error: Unable to find the global mesh at {mesh_i} when assigning it for this pocket"
            );
            return;
        }

        apply_mesh_colors(&mut scene_meshes[mesh_i], &color);
    }
}
