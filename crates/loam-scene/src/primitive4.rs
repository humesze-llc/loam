//! `Primitive4`, 4D-SDF emit and CPU evaluation for the [`loam_shape::Shape`]
//! variants in ℝ⁴.
//!
//! Counterpart of [`crate::Primitive`] for 4D points; emits
//! `fn <name>(p: vec4<f32>) -> f32` returning the signed distance to the
//! shape's surface (positive outside, negative inside, zero on boundary), and
//! evaluates the same formulas on the CPU.
//!
//! ## Variant coverage
//!
//! | `Shape` variant | 4D SDF | Notes |
//! |---|---|---|
//! | [`Shape::HyperSphere4D`] | closed-form | `length(p − center) − radius` |
//! | [`Shape::HalfSpace4D`] | closed-form | `dot(p, normal) − offset`; honest in flat ℝ⁴ (the only 4D Space today) |
//! | [`Shape::ConvexPolytope4D`] | [`SENTINEL_DISTANCE`] | real path is `Hyperslice4DNode`'s per-frame uniforms |
//! | 3D-only variants | [`SENTINEL_DISTANCE`] | shouldn't appear in `Scene4` |

use glam::Vec4;
use loam_shape::Shape;

use crate::SENTINEL_DISTANCE;

/// Emit and evaluate a 4D signed-distance function. Counterpart of
/// [`crate::Primitive`] for 4D variants.
///
/// No `space: &S` parameter today: ℝ⁴ is the only 4D Space and it's flat, so
/// chart-coord SDFs are correct. A curved 4D Space would add that parameter and
/// gate `HalfSpace4D` on `Space::is_chart_flat`, like the 3D counterpart.
pub trait Primitive4 {
    /// Emit a WGSL function named `name` taking `vec4<f32>` and returning `f32`.
    fn to_wgsl_4d(&self, name: &str) -> String;

    /// Signed distance from `p` to `self`, the CPU twin of [`Self::to_wgsl_4d`]'s
    /// emitted body. Exact up to `f32` rounding: the 4D emitter prints constants
    /// in Rust's shortest round-tripping form, so no decimal truncation enters.
    fn eval_4d(&self, p: Vec4) -> f32;
}

impl Primitive4 for Shape {
    fn to_wgsl_4d(&self, name: &str) -> String {
        match self {
            Shape::HyperSphere4D { center, radius } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn length(p - vec4<f32>({cx}, {cy}, {cz}, {cw})) - ({radius});\n\
                }}\n",
                name = name,
                cx = center.x,
                cy = center.y,
                cz = center.z,
                cw = center.w,
                radius = radius,
            ),

            // Sentinel until the real path lands: face hyperplanes are pose-
            // dependent per frame, so `Hyperslice4DNode` ships them via a uniform
            // buffer rather than baking constants here. Renders invisible so
            // accidental scene inclusion fails visibly.
            Shape::ConvexPolytope4D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \t// ConvexPolytope4D: half-space emit lives in the\n\
                \t// render node's per-frame uniform path, not here.\n\
                \t// See Hyperslice4DNode.\n\
                \treturn {SENTINEL_DISTANCE:e};\n\
                }}\n",
            ),

            // Chart-coord hyperplane SDF; honest because ℝ⁴ is flat.
            Shape::HalfSpace4D { normal, offset } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn dot(p, vec4<f32>({nx}, {ny}, {nz}, {nw})) - ({offset});\n\
                }}\n",
                name = name,
                nx = normal.x,
                ny = normal.y,
                nz = normal.z,
                nw = normal.w,
                offset = offset,
            ),

            // 2D/3D variants don't belong in a 4D scene; sentinel keeps the trait
            // exhaustive without emitting type-mismatched WGSL.
            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \treturn {SENTINEL_DISTANCE:e};\n\
                }}\n",
            ),
        }
    }

    fn eval_4d(&self, p: Vec4) -> f32 {
        match self {
            Shape::HyperSphere4D { center, radius } => (p - *center).length() - *radius,

            Shape::ConvexPolytope4D { .. } => SENTINEL_DISTANCE,

            Shape::HalfSpace4D { normal, offset } => p.dot(*normal) - *offset,

            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => SENTINEL_DISTANCE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn hypersphere_4d_emit_has_expected_shape() {
        let s = Shape::HyperSphere4D {
            center: Vec4::new(1.0, 2.0, 3.0, 4.0),
            radius: 0.5,
        };
        let wgsl = s.to_wgsl_4d("ball");
        assert!(wgsl.contains("fn ball(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("length(p - vec4<f32>(1, 2, 3, 4))"));
        assert!(wgsl.contains("- (0.5)"));
    }

    #[test]
    fn halfspace_4d_emits_dot_in_flat_chart() {
        let s = Shape::HalfSpace4D {
            normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
            offset: 0.0,
        };
        let wgsl = s.to_wgsl_4d("floor4");
        assert!(wgsl.contains("fn floor4(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("dot(p, vec4<f32>(0, 1, 0, 0))"));
        assert!(wgsl.contains("- (0)"));
    }

    #[test]
    fn polytope_4d_emit_is_sentinel() {
        let s = Shape::ConvexPolytope4D {
            vertices: vec![Vec4::ZERO; 5],
        };
        let wgsl = s.to_wgsl_4d("pent");
        assert!(wgsl.contains("fn pent(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
    }

    #[test]
    fn three_d_variants_emit_sentinel_in_4d() {
        let s = Shape::sphere_at(glam::Vec3::ZERO, 1.0);
        let wgsl = s.to_wgsl_4d("oops");
        assert!(wgsl.contains("vec4<f32>"));
        assert!(wgsl.contains("return 1e9"));
    }
}
