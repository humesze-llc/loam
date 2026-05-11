//! WGSL emission for [`rye_shape::Shape`].
//!
//! The shape data model lives in `rye-shape`, that's where `Sphere`, `HalfSpace`, `Box3`, and
//! the polytope variants are defined so that `rye-physics` and `rye-sdf` share one canonical
//! type. This module is the rendering half: it implements the [`Primitive`] extension trait on
//! [`Shape`], dispatching per-variant to the WGSL formula appropriate to each.
//!
//! ## Variants and their SDFs
//!
//! - [`Shape::Sphere`], `rye_distance(p, center) − radius`. The center is part of the shape
//!   (unlike physics, where the body's position is the center); SDF scenes use it to place
//!   spheres without a transform combinator.
//! - [`Shape::Box3`], standard Euclidean box SDF (`max(abs(p) − b, 0)` and the
//!   negative-interior correction). Axis-aligned, centered at the local frame's origin. Honest
//!   in E³; chart-coordinate in H³/S³ (the trait rule treats this as accepted because no
//!   geodesic-box SDF exists in closed form).
//! - [`Shape::HalfSpace`], chart-coord `dot(p, n) − offset` **only in flat Spaces** (gated by
//!   `WgslSpace::is_chart_flat`). In curved Spaces (H³, S³, `BlendedSpace`) it would draw the
//!   chart's straight plane, not the geodesic plane, so the emission falls back to the `+1e9`
//!   sentinel below until a closed-form geodesic-plane SDF lands (artanh-of-Möbius in H³,
//!   chord-distance to a great hyperplane in S³). The `Shape` variant stays in `rye-shape`
//!   because physics still uses it for collision walls regardless of the rendering side.
//!
//! Variants that always emit a `+1e9` invisible-far-away sentinel today:
//!
//! - [`Shape::HalfSpace4D`]: 4D variant; lives in [`Primitive4`](crate::Primitive4).
//! - [`Shape::Polygon2D`], [`Shape::ConvexPolytope3D`], [`Shape::ConvexPolytope4D`]:
//!   vertex-list shapes that need either a baked mesh-SDF or a runtime convex-hull kernel.
//! - [`Shape::HyperSphere4D`]: 4D variant; lives in [`Primitive4`](crate::Primitive4).

use rye_math::WgslSpace;
use rye_shape::Shape;

/// Extension trait on [`Shape`] that emits its signed-distance function as WGSL.
///
/// The emitted function has signature `fn {name}(p: vec3<f32>) -> f32`. The trait rule: emitted
/// SDFs must call only `rye_*` Space-prelude functions, never raw chart-coord arithmetic,
/// **except** when the Space self-reports flat via `WgslSpace::is_chart_flat`. In flat Spaces
/// the chart-coord and Riemannian distances coincide, so primitives like [`Shape::HalfSpace`]
/// are free to emit `dot(p, n) - offset` directly. In curved Spaces those primitives sentinel
/// until a real geodesic-plane SDF is written. See the module-level doc for the per-variant
/// status.
pub trait Primitive {
    /// Emit a WGSL function named `name` that returns the signed distance from `p` to `self` in
    /// the given Space.
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
                // Sentinel emission. `HalfSpace` falls here when the Space is curved
                // (chart-coord plane != geodesic plane); `HalfSpace4D` always sentinels
                // via this arm because `Primitive4` doesn't take a Space parameter today.
                // Vertex-list shapes (Polygon2D, ConvexPolytope*) need a baked mesh-SDF
                // or runtime convex-hull kernel. A `+1e9` SDF renders as an invisible
                // far-away surface so accidental inclusion fails visibly rather than
                // silently drawing wrong geometry.
                format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn 1e9;\n}}\n",)
            }
        }
    }
}
