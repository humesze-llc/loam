//! WGSL emission for [`rye_shape::Shape`].
//!
//! The shape data model lives in `rye-shape` (shared with `rye-physics`); this
//! module is the rendering half, implementing the [`Primitive`] extension trait
//! on [`Shape`] and dispatching per-variant to the appropriate WGSL formula.
//!
//! ## Variants and their SDFs
//!
//! - [`Shape::Sphere`]: `rye_distance(p, center) − radius`. Center is part of
//!   the shape, so SDF scenes place spheres without a transform combinator.
//! - [`Shape::Box3`]: standard Euclidean box SDF. Honest in E³; chart-coord in
//!   H³/S³ (accepted; no closed-form geodesic-box SDF exists).
//! - [`Shape::HalfSpace`]: chart-coord `dot(p, n) − offset` only in flat Spaces
//!   (gated by `WgslSpace::is_chart_flat`). Curved Spaces draw the chart plane,
//!   not the geodesic plane, so they sentinel until a closed-form geodesic-plane
//!   SDF lands (artanh-of-Möbius in H³, chord-distance to a great hyperplane in
//!   S³).
//!
//! Variants that always emit a `+1e9` sentinel today:
//!
//! - [`Shape::HalfSpace4D`], [`Shape::HyperSphere4D`]: 4D; live in
//!   [`Primitive4`](crate::Primitive4).
//! - [`Shape::Polygon2D`], [`Shape::ConvexPolytope3D`],
//!   [`Shape::ConvexPolytope4D`]: vertex-list shapes needing a baked mesh-SDF or
//!   a runtime convex-hull kernel.

use rye_math::WgslSpace;
use rye_shape::Shape;

/// Extension trait on [`Shape`] that emits its signed-distance function as WGSL.
///
/// Emits `fn {name}(p: vec3<f32>) -> f32`. Trait rule: SDFs call only `rye_*`
/// Space-prelude functions, never raw chart-coord arithmetic, except when the
/// Space self-reports flat via `WgslSpace::is_chart_flat` (where chart-coord and
/// Riemannian distances coincide). See the module doc for per-variant status.
pub trait Primitive {
    /// Emit a WGSL function named `name` returning the signed distance from `p`
    /// to `self` in the given Space.
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String;
}

impl Primitive for Shape {
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String {
        match self {
            Shape::Sphere { center, radius } => format!(
                "fn {name}(p: vec3<f32>) -> f32 {{\n\
                 \treturn rye_distance(p, vec3<f32>({cx:.6}, {cy:.6}, {cz:.6})) - {r:.6};\n\
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
                // variants with no 3D closed form. 1e9 renders invisible so
                // accidental inclusion fails visibly instead of drawing wrong
                // geometry.
                format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn 1e9;\n}}\n",)
            }
        }
    }
}
