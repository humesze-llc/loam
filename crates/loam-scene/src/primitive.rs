//! WGSL emission and CPU evaluation for [`loam_shape::Shape`].
//!
//! The shape data model lives in `loam-shape` (shared with `loam-physics`); this
//! module is the rendering half, implementing the [`Primitive`] extension trait
//! on [`Shape`] and dispatching per-variant to the appropriate SDF formula.
//! [`Primitive::to_wgsl`] and [`Primitive::eval`] are the same `match`, arm for
//! arm; adding a `Shape` variant fails to compile in both.
//!
//! ## Variants and their SDFs
//!
//! - [`Shape::Sphere`]: `loam_distance(p, center) − radius` on the GPU,
//!   `space.distance(p, center) − radius` on the CPU. Center is part of the
//!   shape, so SDF scenes place spheres without a transform combinator.
//! - [`Shape::Box3`]: standard Euclidean box SDF. Honest in E³; chart-coord in
//!   H³/S³ (accepted; no closed-form geodesic-box SDF exists).
//! - [`Shape::HalfSpace`]: chart-coord `dot(p, n) − offset` only in flat Spaces
//!   (gated by `Space::is_chart_flat`). Curved Spaces draw the chart plane,
//!   not the geodesic plane, so they sentinel until a closed-form geodesic-plane
//!   SDF lands (artanh-of-Möbius in H³, chord-distance to a great hyperplane in
//!   S³).
//!
//! Variants that always return [`SENTINEL_DISTANCE`] today:
//!
//! - [`Shape::HalfSpace4D`], [`Shape::HyperSphere4D`]: 4D; live in
//!   [`Primitive4`](crate::Primitive4).
//! - [`Shape::Polygon2D`], [`Shape::ConvexPolytope3D`],
//!   [`Shape::ConvexPolytope4D`]: vertex-list shapes needing a baked mesh-SDF or
//!   a runtime convex-hull kernel.

use glam::Vec3;
use loam_math::{Space, WgslSpace};
use loam_shape::Shape;

use crate::SENTINEL_DISTANCE;

/// Extension trait on [`Shape`] exposing its signed-distance function as WGSL
/// text and as a Rust scalar.
///
/// Emits `fn {name}(p: vec3<f32>) -> f32`. Trait rule: SDFs use only `loam_*`
/// Space-prelude functions on the GPU and only [`Space`] methods on the CPU,
/// never raw chart-coord arithmetic, except when the Space self-reports flat via
/// [`Space::is_chart_flat`] (where chart-coord and Riemannian distances
/// coincide). See the module doc for per-variant status.
pub trait Primitive {
    /// Emit a WGSL function named `name` returning the signed distance from `p`
    /// to `self` in the given Space.
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String;

    /// Signed distance from `p` to `self` in the given Space, the CPU twin of
    /// [`Self::to_wgsl`]'s emitted body.
    ///
    /// Parity with the emitted function is bounded, not exact: the emitter
    /// prints baked constants at six decimals while this holds the `f32`, so a
    /// residual near 5e-7 is structural. Where a Space's WGSL prelude is a
    /// deliberate approximation of its Rust `distance` the residual is that
    /// Space's own divergence, which is larger and documented on the Space.
    fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32;
}

impl Primitive for Shape {
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String {
        match self {
            Shape::Sphere { center, radius } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn loam_distance(p, vec3<f32>({cx:.6}, {cy:.6}, {cz:.6})) - {r:.6};\n\
                 }}\n",
                name = name,
                cx = center.x,
                cy = center.y,
                cz = center.z,
                r = radius,
            ),
            Shape::Box3 { half_extents } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \tlet b = vec3<f32>({hx:.6}, {hy:.6}, {hz:.6});\n\
                 \tlet q = abs(p) - b;\n\
                 \treturn length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n\
                 }}\n",
                name = name,
                hx = half_extents.x,
                hy = half_extents.y,
                hz = half_extents.z,
            ),
            Shape::HalfSpace { normal, offset } if space.is_chart_flat() => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn dot(p, vec3<f32>({nx:.6}, {ny:.6}, {nz:.6})) - ({d:.6});\n\
                 }}\n",
                name = name,
                nx = normal.x,
                ny = normal.y,
                nz = normal.z,
                d = offset,
            ),
            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => {
                // Sentinel: `HalfSpace` in a curved Space, plus 4D and vertex-list
                // variants with no 3D closed form. Renders invisible so accidental
                // inclusion fails visibly instead of drawing wrong geometry.
                // `{:e}` is the WGSL float-literal spelling of the constant.
                format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn {SENTINEL_DISTANCE:e};\n}}\n",)
            }
        }
    }

    fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32 {
        match self {
            // `Space::distance` rather than a CPU port of the `loam_distance`
            // prelude: the Rust impl is the reference and the prelude the twin,
            // so building on the prelude would invert that.
            Shape::Sphere { center, radius } => space.distance(p, *center) - *radius,

            // Quilez 2013, "distance functions", exact box SDF. Operand order
            // matches the emitted `length(max(q, 0)) + min(max(q.x, max(q.y,
            // q.z)), 0)`.
            Shape::Box3 { half_extents } => {
                let q = p.abs() - *half_extents;
                q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
            }

            Shape::HalfSpace { normal, offset } if space.is_chart_flat() => {
                p.dot(*normal) - *offset
            }

            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => SENTINEL_DISTANCE,
        }
    }
}
