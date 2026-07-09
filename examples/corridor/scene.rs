//! Corridor demo scene. Self-contained; depends only on the typed
//! [`loam_scene::Scene`] / [`loam_scene::SceneNode`] layer and emits WGSL
//! against `EuclideanR3`.

use glam::Vec3;
use loam_math::EuclideanR3;
use loam_scene::{Scene, SceneNode};

/// A rectangular corridor oriented along the Z axis, lined with rows
/// of geodesic spheres for depth cues.
///
/// Pillars use `loam_distance` so they are space-aware (perfect spheres
/// in every metric, with the curvature carried by ray bending). Floor,
/// ceiling, and side walls are chart-coordinate planes chosen
/// specifically to visualise the chart-vs-geodesic difference.
/// `corridor_demo_wgsl` compiles against `EuclideanR3` (flat), so
/// those wall planes emit honestly via `dot(p, n) - d`. A future
/// `corridor_demo_wgsl_<S>` for H³ / S³ would sentinel them via
/// `Primitive`'s `HalfSpace` arm until geodesic-plane SDFs land.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorridorScene {
    /// Half-width of the corridor along the Space X axis.
    pub half_width: f32,
    /// Half-height of the corridor along the Space Y axis.
    pub half_height: f32,
    /// Half-length along Z. End walls close the corridor so every ray
    /// hits geometry before reaching the Poincaré ball boundary in
    /// H³/S³. An open corridor (no end walls) lets rays drift into the
    /// saturation shell where Euclidean-coord wall SDFs produce noise.
    pub half_depth: f32,
    /// Geodesic radius of each pillar sphere.
    pub pillar_radius: f32,
    /// Distance from centerline to each pillar row (Space X).
    pub pillar_x_offset: f32,
    /// Y coordinate of pillar centres (typically `-half_height + pillar_radius`).
    pub pillar_y: f32,
    /// Geodesic spacing between consecutive pillars along Z.
    pub pillar_z_spacing: f32,
    /// Total pillars per row. Must be odd (symmetric about z = 0).
    pub pillars_per_row: u32,
}

impl Default for CorridorScene {
    fn default() -> Self {
        Self {
            half_width: 0.55,
            half_height: 0.40,
            // 0.70 keeps the end walls well inside the Poincaré ball
            // (|p| ≤ 0.70 has conformal factor ~3.9, still well-conditioned).
            half_depth: 0.70,
            pillar_radius: 0.07,
            pillar_x_offset: 0.40,
            pillar_y: -0.33,
            pillar_z_spacing: 0.24,
            pillars_per_row: 7,
        }
    }
}

impl CorridorScene {
    /// Build the typed scene tree for the corridor.
    ///
    /// Walls are `SceneNode::plane(...)` leaves; pillars are
    /// `SceneNode::sphere(...)` leaves wrapped in a union. The emitted
    /// SDF for the walls depends on the Space the scene is later
    /// compiled against: `dot(p, n) - d` in flat charts (E³),
    /// sentinel in curved charts (H³ / S³).
    pub fn to_scene(self) -> Scene {
        assert!(
            self.pillars_per_row % 2 == 1,
            "pillars_per_row must be odd so the row is symmetric about z=0"
        );
        // Six walls as Euclidean half-space planes.
        // Each plane's SDF is positive on the interior side of that wall.
        let walls = SceneNode::plane(Vec3::Y, -self.half_height) // floor: p.y + H
            .union(SceneNode::plane(Vec3::NEG_Y, -self.half_height)) // ceiling: H - p.y
            .union(SceneNode::plane(Vec3::X, -self.half_width)) // left: p.x + W
            .union(SceneNode::plane(Vec3::NEG_X, -self.half_width)) // right: W - p.x
            .union(SceneNode::plane(Vec3::Z, -self.half_depth)) // back: p.z + D
            .union(SceneNode::plane(Vec3::NEG_Z, -self.half_depth)); // front: D - p.z

        let half = (self.pillars_per_row - 1) / 2;
        let mut root = walls;
        for i in -(half as i32)..=(half as i32) {
            let z = (i as f32) * self.pillar_z_spacing;
            root = root.union(SceneNode::sphere(
                Vec3::new(-self.pillar_x_offset, self.pillar_y, z),
                self.pillar_radius,
            ));
            root = root.union(SceneNode::sphere(
                Vec3::new(self.pillar_x_offset, self.pillar_y, z),
                self.pillar_radius,
            ));
        }
        Scene::new(root)
    }

    pub fn to_wgsl(self) -> String {
        self.to_scene().to_wgsl(&EuclideanR3)
    }
}

/// Emit WGSL source defining `loam_scene_sdf` for the default
/// `CorridorScene`.
pub fn corridor_demo_wgsl() -> String {
    CorridorScene::default().to_wgsl()
}
