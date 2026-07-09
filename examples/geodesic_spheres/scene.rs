//! Geodesic-spheres demo scene. Self-contained; depends only on the
//! typed [`loam_scene::Scene`] / [`loam_scene::SceneNode`] layer and emits
//! WGSL against `EuclideanR3`.

use std::f32::consts::PI;

use glam::Vec3;
use loam_math::EuclideanR3;
use loam_scene::{Scene, SceneNode};

/// Geodesic-spheres demo scene parameters.
///
/// The default scene is seven geodesic spheres: one centre sphere and
/// six orbit spheres around it.
///
/// Optionally, a Euclidean-y slab (floor / ceiling planes) can be
/// enabled as a visual cage. The slab renders honestly via chart-coord
/// `dot(p, n) - d` in `EuclideanR3` (the Space this helper compiles
/// against); per `Primitive`'s `HalfSpace` arm it would sentinel in
/// H³ / S³ until geodesic-plane SDFs land.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicSpheresScene {
    pub sphere_radius: f32,
    pub orbit_radius: f32,
    pub orbit_height: f32,
    pub include_slab: bool,
    pub floor_y: f32,
    pub ceiling_y: f32,
}

impl Default for GeodesicSpheresScene {
    fn default() -> Self {
        Self {
            sphere_radius: 0.22,
            orbit_radius: 0.62,
            orbit_height: 0.18,
            include_slab: false,
            floor_y: -0.50,
            ceiling_y: 0.82,
        }
    }
}

impl GeodesicSpheresScene {
    /// Build the typed scene tree. Orbit centres are pre-computed in
    /// Rust and embedded as literals; `loam_distance` carries the Space
    /// metric.
    pub fn to_scene(self) -> Scene {
        let mut node = SceneNode::sphere(Vec3::new(0.0, 0.12, 0.0), self.sphere_radius);
        for i in 0..6 {
            let a = (i as f32) * PI / 3.0;
            let center = Vec3::new(
                a.cos() * self.orbit_radius,
                self.orbit_height,
                a.sin() * self.orbit_radius,
            );
            node = node.union(SceneNode::sphere(center, self.sphere_radius));
        }
        if self.include_slab {
            // floor: SDF positive above floor_y; ceiling: positive below
            // ceiling_y. union = min collapses to the closer wall.
            let floor = SceneNode::plane(Vec3::Y, self.floor_y);
            let ceiling = SceneNode::plane(Vec3::NEG_Y, -self.ceiling_y);
            node = node.union(floor.union(ceiling));
        }
        Scene::new(node)
    }

    pub fn to_wgsl(self) -> String {
        self.to_scene().to_wgsl(&EuclideanR3)
    }
}

/// Emit WGSL source defining `loam_scene_sdf` for the default scene.
pub fn geodesic_spheres_demo_wgsl() -> String {
    GeodesicSpheresScene::default().to_wgsl()
}
