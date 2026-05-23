//! Periodic geodesic lattice demo scene. Self-contained; depends only
//! on the typed [`rye_scene::Scene`] / [`rye_scene::SceneNode`] layer and
//! emits WGSL against any [`rye_math::WgslSpace`].

use glam::Vec3;
use rye_math::{Space, WgslSpace};
use rye_scene::{Scene, SceneNode};

/// Periodic geodesic lattice scene.
///
/// Emits a `rye_scene_sdf` that places a sphere at the origin and at
/// geodesic lattice positions along the ±X, ±Y, ±Z axes. The lattice
/// centres are computed in Rust by calling `space.exp` so they live at
/// evenly-spaced geodesic intervals in the given Space:
///
/// - E³: evenly-spaced Euclidean grid
/// - H³: tanh-compressed toward the Poincaré boundary
/// - S³: sin-based, wrapping past π/2
///
/// The visual difference between the three spaces emerges naturally
/// when the same shader is compiled against each Space prelude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeSphereScene {
    /// Number of shells along each axis direction (1..=N).
    pub steps: u32,
    /// Geodesic arc length between consecutive shells.
    pub geodesic_spacing: f32,
    /// Sphere radius in Space coordinates.
    pub sphere_radius: f32,
}

impl Default for LatticeSphereScene {
    fn default() -> Self {
        Self {
            steps: 2,
            geodesic_spacing: 0.45,
            sphere_radius: 0.12,
        }
    }
}

impl LatticeSphereScene {
    /// Build the typed scene tree for the given Space.
    ///
    /// Centres are computed via `space.exp` and stored as literal
    /// positions in `Sphere` primitives. The emitted `rye_scene_sdf`
    /// calls only `rye_distance`, so the spatial metric is fully
    /// Space-aware at runtime.
    pub fn to_scene<S>(self, space: &S) -> Scene
    where
        S: Space<Point = Vec3, Vector = Vec3> + WgslSpace,
    {
        let axes = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let mut node = SceneNode::sphere(Vec3::ZERO, self.sphere_radius);
        for axis in axes {
            for k in 1..=self.steps {
                let tangent = axis * (k as f32 * self.geodesic_spacing);
                node = node.union(SceneNode::sphere(
                    space.exp(Vec3::ZERO, tangent),
                    self.sphere_radius,
                ));
            }
        }
        Scene::new(node)
    }

    pub fn to_wgsl<S>(self, space: &S) -> String
    where
        S: Space<Point = Vec3, Vector = Vec3> + WgslSpace,
    {
        self.to_scene(space).to_wgsl(space)
    }
}
