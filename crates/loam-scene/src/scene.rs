//! Typed scene tree that assembles SDF primitives into `loam_scene_sdf`.
//!
//! Build a [`Scene`] from [`SceneNode`] combinators, then call [`Scene::to_wgsl`]
//! with a Space to get the complete WGSL scene module.
//!
//! # Emission strategy
//!
//! Each leaf emits a named helper (`sdf_p{n}`); each combinator emits a `let`
//! binding in `loam_scene_sdf`. The depth-first walk emits children before their
//! parent so referenced variables are always in scope.
//!
//! # Example
//!
//! ```rust
//! use glam::Vec3;
//! use loam_scene::scene::{Scene, SceneNode};
//! use loam_math::EuclideanR3;
//!
//! let scene = Scene::new(
//!     SceneNode::sphere(Vec3::ZERO, 0.3)
//!         .union(SceneNode::plane(Vec3::Y, -0.5)),
//! );
//! let wgsl = scene.to_wgsl(&EuclideanR3);
//! assert!(wgsl.contains("fn loam_scene_sdf"));
//! ```

use std::boxed::Box;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::combinator::smooth_min_fn;
use crate::primitive::Primitive;
use loam_math::{Space, WgslSpace};
pub use loam_shape::Shape as PrimitiveKind;

/// A node in the typed SDF scene tree. Leaves hold a primitive; interior nodes
/// combine two children via a boolean or smooth operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneNode {
    Leaf(PrimitiveKind),
    Union(Box<SceneNode>, Box<SceneNode>),
    Intersection(Box<SceneNode>, Box<SceneNode>),
    /// Carve the right subtree from the left: `max(left, -right)`.
    Difference(Box<SceneNode>, Box<SceneNode>),
    /// Polynomial smooth-minimum union. `k` is the blend radius in Space units.
    SmoothUnion {
        k: f32,
        left: Box<SceneNode>,
        right: Box<SceneNode>,
    },
}

// ---- Constructors -----------------------------------------------------------

impl SceneNode {
    pub fn sphere(center: Vec3, radius: f32) -> Self {
        SceneNode::Leaf(PrimitiveKind::Sphere { center, radius })
    }

    /// Half-space leaf. Emission depends on the compile-time Space (see
    /// [`Primitive`]'s `HalfSpace` arm): chart-coord `dot(p, n) - d` in flat
    /// charts, [`crate::SENTINEL_DISTANCE`] in curved charts until geodesic-plane
    /// SDFs land.
    pub fn plane(normal: Vec3, offset: f32) -> Self {
        SceneNode::Leaf(PrimitiveKind::HalfSpace { normal, offset })
    }

    pub fn box_(half_extents: Vec3) -> Self {
        SceneNode::Leaf(PrimitiveKind::Box3 { half_extents })
    }

    pub fn cube(half_side: f32) -> Self {
        SceneNode::Leaf(PrimitiveKind::Box3 {
            half_extents: Vec3::splat(half_side),
        })
    }

    // ---- Combinators --------------------------------------------------------

    pub fn union(self, other: SceneNode) -> Self {
        SceneNode::Union(Box::new(self), Box::new(other))
    }

    pub fn intersect(self, other: SceneNode) -> Self {
        SceneNode::Intersection(Box::new(self), Box::new(other))
    }

    /// Carve `other` out of `self`.
    pub fn subtract(self, other: SceneNode) -> Self {
        SceneNode::Difference(Box::new(self), Box::new(other))
    }

    pub fn smooth_union(self, other: SceneNode, k: f32) -> Self {
        SceneNode::SmoothUnion {
            k,
            left: Box::new(self),
            right: Box::new(other),
        }
    }
}

// ---- Scene ------------------------------------------------------------------

/// A complete SDF scene: a single root [`SceneNode`] that emits
/// `fn loam_scene_sdf(p: vec3<f32>) -> f32` when compiled for a given Space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub root: SceneNode,
}

impl Scene {
    pub fn new(root: SceneNode) -> Self {
        Self { root }
    }

    /// Emit the complete WGSL scene module (helpers + `loam_scene_sdf` entry
    /// point) for the given Space. Prepend to the Space prelude and user shader.
    pub fn to_wgsl<S: WgslSpace>(&self, space: &S) -> String {
        let mut helpers = String::new();
        let mut body = String::new();
        let mut counter = 0u32;

        let result_var = emit_node(&self.root, space, &mut counter, &mut helpers, &mut body);

        format!(
            "// ---- loam-scene scene (typed) ----\n\
             {helpers}\
             fn loam_scene_sdf(p: vec3<f32>) -> f32 {{\n\
             {body}\
             \treturn {result_var};\n\
             }}\n"
        )
    }

    /// Signed distance from `p` to the scene, the CPU twin of the emitted
    /// `loam_scene_sdf`. Walks the same tree as [`emit_node`] with the scalar
    /// algebra inlined, so the two cannot diverge structurally; see
    /// [`Primitive::eval`] for the residual that remains.
    ///
    /// Allocation-free and recursion-only, so a grid bake pays no per-sample
    /// heap traffic.
    pub fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32 {
        eval_node(&self.root, space, p)
    }

    /// Deserialize a Scene from a RON string.
    pub fn from_ron(src: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(src)
    }

    /// Serialize this Scene to a RON string.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

// ---- Recursive emitter ------------------------------------------------------

/// Walk `node` depth-first, appending helper definitions to `helpers` and `let`
/// bindings to `body`. Returns the WGSL variable holding this node's distance.
fn emit_node<S: WgslSpace>(
    node: &SceneNode,
    space: &S,
    counter: &mut u32,
    helpers: &mut String,
    body: &mut String,
) -> String {
    let idx = *counter;
    *counter += 1;

    match node {
        SceneNode::Leaf(prim) => {
            let fn_name = format!("sdf_p{idx}");
            helpers.push_str(&prim.to_wgsl(space, &fn_name));
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = {fn_name}(p);\n"));
            var
        }

        SceneNode::Union(left, right) => {
            let lv = emit_node(left, space, counter, helpers, body);
            let rv = emit_node(right, space, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = min({lv}, {rv});\n"));
            var
        }

        SceneNode::Intersection(left, right) => {
            let lv = emit_node(left, space, counter, helpers, body);
            let rv = emit_node(right, space, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = max({lv}, {rv});\n"));
            var
        }

        SceneNode::Difference(left, right) => {
            let lv = emit_node(left, space, counter, helpers, body);
            let rv = emit_node(right, space, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = max({lv}, -({rv}));\n"));
            var
        }

        SceneNode::SmoothUnion { k, left, right } => {
            let fn_name = format!("sdf_smin{idx}");
            helpers.push_str(&smooth_min_fn(&fn_name, *k));
            let lv = emit_node(left, space, counter, helpers, body);
            let rv = emit_node(right, space, counter, helpers, body);
            let var = format!("d{idx}");
            body.push_str(&format!("\tlet {var} = {fn_name}({lv}, {rv});\n"));
            var
        }
    }
}

// ---- Recursive evaluator ----------------------------------------------------

/// Scalar twin of [`emit_node`], one arm per `SceneNode` variant in the same
/// order. The combinator expressions are transcribed from
/// [`crate::combinator`]'s emitted text operand for operand: reassociating them
/// is algebraically neutral but not bit-neutral, and this sits inside the
/// determinism boundary once a baked collider feeds the sim.
fn eval_node<S: Space<Point = Vec3, Vector = Vec3>>(node: &SceneNode, space: &S, p: Vec3) -> f32 {
    match node {
        SceneNode::Leaf(prim) => prim.eval(space, p),

        SceneNode::Union(left, right) => eval_node(left, space, p).min(eval_node(right, space, p)),

        SceneNode::Intersection(left, right) => {
            eval_node(left, space, p).max(eval_node(right, space, p))
        }

        SceneNode::Difference(left, right) => {
            eval_node(left, space, p).max(-eval_node(right, space, p))
        }

        SceneNode::SmoothUnion { k, left, right } => {
            // Quilez 2013, "smooth minimum", polynomial variant, as emitted by
            // `combinator::smooth_min_fn`. `mix(b, a, h)` is transcribed in the
            // form WGSL defines it, `b·(1 − h) + a·h` (WGSL spec, "mix"), not
            // the algebraically equal lerp form `b + (a − b)·h`: only the former
            // is exact at the clamped ends `h ∈ {0, 1}`, which is where the vast
            // majority of sample points sit.
            let a = eval_node(left, space, p);
            let b = eval_node(right, space, p);
            let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
            (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
        }
    }
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{EuclideanR3, HyperbolicH3};

    #[test]
    fn single_sphere_emits_scene_sdf() {
        let scene = Scene::new(SceneNode::sphere(Vec3::ZERO, 0.25));
        let wgsl = scene.to_wgsl(&EuclideanR3);
        assert!(wgsl.contains("fn loam_scene_sdf(p: vec3<f32>) -> f32"));
        assert!(wgsl.contains("loam_distance"));
        assert!(wgsl.contains("0.250000"));
    }

    #[test]
    fn union_of_two_spheres() {
        let scene = Scene::new(
            SceneNode::sphere(Vec3::ZERO, 0.2).union(SceneNode::sphere(Vec3::X * 0.5, 0.2)),
        );
        let wgsl = scene.to_wgsl(&EuclideanR3);
        assert!(wgsl.contains("fn loam_scene_sdf"));
        assert!(wgsl.contains("min("));
        assert!(wgsl.contains("sdf_p1"));
        assert!(wgsl.contains("sdf_p2"));
    }

    #[test]
    fn smooth_union_emits_smin_helper() {
        let scene = Scene::new(
            SceneNode::sphere(Vec3::ZERO, 0.2).smooth_union(SceneNode::cube(0.15), 0.05),
        );
        let wgsl = scene.to_wgsl(&EuclideanR3);
        assert!(wgsl.contains("sdf_smin0"));
        assert!(wgsl.contains("clamp"));
        assert!(wgsl.contains("mix"));
    }

    #[test]
    fn difference_uses_negation() {
        let scene = Scene::new(SceneNode::sphere(Vec3::ZERO, 0.3).subtract(SceneNode::cube(0.2)));
        let wgsl = scene.to_wgsl(&EuclideanR3);
        assert!(wgsl.contains("max("));
        assert!(wgsl.contains("-("));
    }

    #[test]
    fn scene_is_space_agnostic_for_spheres() {
        let scene = Scene::new(SceneNode::sphere(Vec3::ZERO, 0.25));
        let e3 = scene.to_wgsl(&EuclideanR3);
        let h3 = scene.to_wgsl(&HyperbolicH3);
        assert_eq!(e3, h3);
    }

    #[test]
    fn ron_round_trip() {
        let scene =
            Scene::new(SceneNode::sphere(Vec3::ZERO, 0.3).union(SceneNode::plane(Vec3::Y, -0.4)));
        let ron_str = scene.to_ron().expect("serialize");
        let recovered: Scene = Scene::from_ron(&ron_str).expect("deserialize");
        assert_eq!(scene.to_wgsl(&EuclideanR3), recovered.to_wgsl(&EuclideanR3),);
    }
}
