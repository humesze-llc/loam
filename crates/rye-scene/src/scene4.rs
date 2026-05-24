//! Typed 4D scene tree, the 4D analogue of [`crate::scene::Scene`].
//!
//! Build a [`Scene4`] from [`SceneNode4`] combinators and emit WGSL for either:
//!
//! - **Native 4D**: `fn rye_scene_sdf_4d(p: vec4<f32>) -> f32`,
//!   useful for full 4D ray-march renderers (future).
//! - **Hyperslice**: `fn rye_scene_sdf(p: vec3<f32>) -> f32` that evaluates the 4D SDF at
//!   `vec4(p, w_slice)`, where `w_slice` is a uniform. This is the production path today,
//!   `Hyperslice4DNode` consumes it as the SDF for a 3D ray march.
//!
//! ## Why a parallel `Scene4`, not `Scene<S, const DIM>`
//!
//! The 3D and 4D paths share no shader code (different SDF signatures, different ray equations,
//! different uniforms), so dimensioning [`crate::scene::Scene`] generically saves no
//! implementation work and obscures the difference. Parallel hierarchies keep each clear.
//!
//! ## Example
//!
//! ```rust
//! use glam::Vec4;
//! use rye_scene::scene4::{Scene4, SceneNode4};
//!
//! let scene = Scene4::new(
//!     SceneNode4::hypersphere(Vec4::ZERO, 0.5)
//!         .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
//! );
//! // Native 4D: SDF takes vec4 directly.
//! let wgsl_4d = scene.to_wgsl_4d();
//! assert!(wgsl_4d.contains("fn rye_scene_sdf_4d(p: vec4<f32>) -> f32"));
//! // Hyperslice mode: SDF takes vec3, internally evaluates at
//! // vec4(p, u.w_slice). The `u.w_slice` uniform is supplied by
//! // the render node.
//! let wgsl_hs = scene.to_hyperslice_wgsl("u.w_slice");
//! assert!(wgsl_hs.contains("fn rye_scene_sdf(p3: vec3<f32>) -> f32"));
//! assert!(wgsl_hs.contains("u.w_slice"));
//! ```

use std::boxed::Box;

use glam::Vec4;
use serde::{Deserialize, Serialize};

use crate::primitive4::Primitive4;
pub use rye_shape::Shape;

/// A node in the 4D scene tree. Mirrors [`crate::scene::SceneNode`] but operates on 4D
/// primitives only.
///
/// Smooth-min isn't included today; the math is identical (use the same `smooth_min_fn` wrapper
/// on `f32`) but no demo currently needs it. Add when one does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneNode4 {
    Leaf(Shape),
    Union(Box<SceneNode4>, Box<SceneNode4>),
    Intersection(Box<SceneNode4>, Box<SceneNode4>),
    /// Carve `right` out of `left`: `max(left, −right)`.
    Difference(Box<SceneNode4>, Box<SceneNode4>),
}

impl SceneNode4 {
    // ---- Leaf constructors ------------------------------------------------

    pub fn hypersphere(center: Vec4, radius: f32) -> Self {
        SceneNode4::Leaf(Shape::HyperSphere4D { center, radius })
    }

    /// Half-space (hyperplane) leaf. ℝ⁴ is the only 4D Space rye ships and it's flat, so
    /// [`crate::Primitive4`] emits an honest `dot(p, n) - offset` hyperplane SDF here. When a
    /// curved 4D Space lands (`BlendedSpace4`, hyperbolic 4-space) `Primitive4` will grow a
    /// `space: &S` parameter and gate this emission on `WgslSpace::is_chart_flat` the same way
    /// the 3D path does today. The shape itself is canonical, also used by `rye-physics` for
    /// 4D collision walls.
    pub fn halfspace(normal: Vec4, offset: f32) -> Self {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset })
    }

    /// Convex 4D polytope leaf. Note: the static `Primitive4` emit returns a sentinel today;
    /// the production path is via `Hyperslice4DNode`'s per-frame uniform buffer (the body's
    /// world-space face hyperplanes are computed CPU-side and uploaded). Until that path lands,
    /// polytope leaves render invisible.
    pub fn polytope(vertices: Vec<Vec4>) -> Self {
        SceneNode4::Leaf(Shape::ConvexPolytope4D { vertices })
    }

    // ---- Combinators ------------------------------------------------------

    pub fn union(self, other: SceneNode4) -> Self {
        SceneNode4::Union(Box::new(self), Box::new(other))
    }

    pub fn intersect(self, other: SceneNode4) -> Self {
        SceneNode4::Intersection(Box::new(self), Box::new(other))
    }

    pub fn subtract(self, other: SceneNode4) -> Self {
        SceneNode4::Difference(Box::new(self), Box::new(other))
    }
}

/// A complete 4D SDF scene, a single root [`SceneNode4`] that emits either
/// `rye_scene_sdf_4d(p: vec4<f32>) -> f32` (full 4D) or `rye_scene_sdf(p: vec3<f32>) -> f32`
/// (hyperslice at the `w_slice` uniform).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene4 {
    pub root: SceneNode4,
}

impl Scene4 {
    pub fn new(root: SceneNode4) -> Self {
        Self { root }
    }

    /// Emit `fn rye_scene_sdf_4d(p: vec4<f32>) -> f32`. Used by future full-4D ray-march
    /// renderers; the hyperslice path uses [`Self::to_hyperslice_wgsl`]. Kind-tracking bindings
    /// are emitted but unused (WGSL accepts unused locals).
    pub fn to_wgsl_4d(&self) -> String {
        let mut helpers = String::new();
        let mut body = String::new();
        let mut counter = 0u32;
        let (d_root, _k_root) =
            emit_node_4d(&self.root, &mut counter, &mut helpers, &mut body, None);
        let kind_consts = SCENE_KIND_CONSTANTS;
        format!(
            "// ---- rye-scene scene4 (native 4D) ----\n\
             {kind_consts}\
             {helpers}\
             fn rye_scene_sdf_4d(p: vec4<f32>) -> f32 {{\n\
             {body}\
             \treturn {d_root};\n\
             }}\n"
        )
    }

    /// Emit the hyperslice SDF module: kind constants, `RyeSceneHit`,
    /// `rye_scene_at(p3) -> RyeSceneHit`, `rye_scene_sdf(p3) -> f32`
    /// (thin wrapper), and `rye_scene_max_t(ro, rd) -> f32` (analytical
    /// far-clip from HalfSpace4D leaves).
    ///
    /// `w_slice_expr` is the WGSL expression for the slicing w-coord; typically `"u.w_slice"`.
    ///
    /// Kind tracking: union picks the closer leaf, intersection picks the farther (boundary)
    /// leaf, difference returns `RYE_PRIM_OTHER`.
    pub fn to_hyperslice_wgsl(&self, w_slice_expr: &str) -> String {
        emit_hyperslice(self, w_slice_expr, None)
    }

    /// Like [`Self::to_hyperslice_wgsl`] but with a runtime gate on every
    /// [`Shape::HalfSpace4D`] leaf: when the WGSL expression
    /// `halfspace_gate_expr` evaluates to `< 0.5` at runtime, every
    /// halfspace SDF returns `1.0e9` (effectively absent) AND every
    /// halfspace's contribution to `rye_scene_max_t` is skipped. When the
    /// gate evaluates to `>= 0.5`, behavior matches the ungated emit.
    ///
    /// Use-case: per-frame toggle of floor / ceiling / cutaway planes via
    /// a single uniform read, without re-compiling the shader on every
    /// flip. Caller writes a `f32` into the uniform pointed at by
    /// `halfspace_gate_expr` (typically a slot of `u.params`).
    ///
    /// `halfspace_gate_expr` should be a scalar `f32` WGSL expression in
    /// scope at the call sites (`rye_scene_at`, `rye_scene_max_t`).
    /// Examples: `"u.params.x"`, `"u.floor_enabled"`, `"1.0"` (always on,
    /// equivalent to the ungated emit), `"0.0"` (always off; every
    /// halfspace silently disappears).
    pub fn to_hyperslice_wgsl_gated(
        &self,
        w_slice_expr: &str,
        halfspace_gate_expr: &str,
    ) -> String {
        emit_hyperslice(self, w_slice_expr, Some(halfspace_gate_expr))
    }

    pub fn from_ron(src: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(src)
    }

    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

/// WGSL constant block emitted at the top of every Scene4 module. Pinned so kind comparisons
/// in the kernel and tests reference the same numeric values.
const SCENE_KIND_CONSTANTS: &str = "\
const RYE_PRIM_HYPERSPHERE4D: u32 = 0u;\n\
const RYE_PRIM_HALFSPACE4D: u32 = 1u;\n\
const RYE_PRIM_OTHER: u32 = 255u;\n";

/// Shared emit driver for [`Scene4::to_hyperslice_wgsl`] and
/// [`Scene4::to_hyperslice_wgsl_gated`]. The optional `halfspace_gate_expr` wraps
/// every `HalfSpace4D` leaf's SDF call in `select(1.0e9, raw, <expr> >= 0.5)`
/// AND skips its `rye_scene_max_t` contribution under the same condition. When
/// `None`, the emit is identical to the original ungated form.
fn emit_hyperslice(
    scene: &Scene4,
    w_slice_expr: &str,
    halfspace_gate_expr: Option<&str>,
) -> String {
    let mut helpers = String::new();
    let mut body = String::new();
    let mut counter = 0u32;
    let (d_root, k_root) = emit_node_4d(
        &scene.root,
        &mut counter,
        &mut helpers,
        &mut body,
        halfspace_gate_expr,
    );
    let kind_consts = SCENE_KIND_CONSTANTS;
    let max_t_body = emit_max_t_body(&scene.root, halfspace_gate_expr);
    // Use a distinct parameter name `p3` and an inner `let p` for the 4D point. WGSL
    // forbids declaring a `let` with the same name as the function parameter (no shadowing);
    // naming the parameter `p3` keeps the helper-emit convention (which calls `sdfN_pK(p)`)
    // intact while sidestepping the collision.
    format!(
        "// ---- rye-scene scene4 (hyperslice at w = {w_slice_expr}) ----\n\
         {kind_consts}\
         struct RyeSceneHit {{ dist: f32, kind: u32 }}\n\
         {helpers}\
         fn rye_scene_at(p3: vec3<f32>) -> RyeSceneHit {{\n\
         \tlet p = vec4<f32>(p3, {w_slice_expr});\n\
         {body}\
         \treturn RyeSceneHit({d_root}, {k_root});\n\
         }}\n\
         fn rye_scene_sdf(p3: vec3<f32>) -> f32 {{\n\
         \treturn rye_scene_at(p3).dist;\n\
         }}\n\
         // Analytical upper bound on march distance: ray-plane intersection\n\
         // for every HalfSpace4D leaf in the scene whose 3D-projected\n\
         // normal points against the ray. Returns +infinity if no leaf\n\
         // contributes; the kernel uses this to terminate near-horizon\n\
         // rays that would otherwise exhaust the iteration budget.\n\
         fn rye_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {{\n\
         \tvar t_max: f32 = 1.0e9;\n\
         {max_t_body}\
         \treturn t_max;\n\
         }}\n"
    )
}

/// Emit the body of `rye_scene_max_t`: ray-plane intersection check for each `HalfSpace4D`
/// leaf. Only the 3D part of the normal is used (the slice fixes `p.w`). Combinator-agnostic:
/// visit every leaf, fold into `t_max` via `min` to get a conservative bound.
fn emit_max_t_body(node: &SceneNode4, halfspace_gate_expr: Option<&str>) -> String {
    let mut body = String::new();
    walk_max_t(node, &mut body, halfspace_gate_expr);
    body
}

fn walk_max_t(node: &SceneNode4, body: &mut String, halfspace_gate_expr: Option<&str>) {
    match node {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset }) => {
            // t = (offset - dot(ro, n)) / dot(rd, n), guarded by dot(rd, n) < 0 so we only
            // catch rays heading toward the plane's solid side. When a runtime gate is set,
            // wrap the entire t-contribution in `if (<gate> >= 0.5)` so a gated-off halfspace
            // contributes no early-termination bound (rays march to the global cap instead).
            let inner = format!(
                "\t\tlet n = vec3<f32>({nx:.6}, {ny:.6}, {nz:.6});\n\
                 \t\tlet dr = dot(rd, n);\n\
                 \t\tif (dr < -1.0e-4) {{\n\
                 \t\t\tlet t = ({offset:.6} - dot(ro, n)) / dr;\n\
                 \t\t\tif (t > 0.0 && t < t_max) {{ t_max = t; }}\n\
                 \t\t}}\n",
                nx = normal.x,
                ny = normal.y,
                nz = normal.z,
                offset = offset,
            );
            match halfspace_gate_expr {
                None => {
                    body.push_str("\t{\n");
                    body.push_str(&inner);
                    body.push_str("\t}\n");
                }
                Some(gate) => {
                    body.push_str(&format!("\tif ({gate} >= 0.5) {{\n"));
                    body.push_str(&inner);
                    body.push_str("\t}\n");
                }
            }
        }
        SceneNode4::Leaf(_) => {} // Other primitives: no closed-form bound.
        SceneNode4::Union(l, r) | SceneNode4::Intersection(l, r) | SceneNode4::Difference(l, r) => {
            walk_max_t(l, body, halfspace_gate_expr);
            walk_max_t(r, body, halfspace_gate_expr);
        }
    }
}

/// Map a Shape variant to its WGSL kind constant name.
fn primitive_kind_constant(shape: &Shape) -> &'static str {
    match shape {
        Shape::HyperSphere4D { .. } => "RYE_PRIM_HYPERSPHERE4D",
        Shape::HalfSpace4D { .. } => "RYE_PRIM_HALFSPACE4D",
        _ => "RYE_PRIM_OTHER",
    }
}

/// Walk the 4D scene tree, append helper functions to `helpers` and `let` bindings to `body`.
/// Returns `(dist_var, kind_var)`, the WGSL identifiers holding this node's signed distance and
/// closest-primitive kind. When `halfspace_gate_expr` is Some, every `HalfSpace4D` leaf's SDF
/// call is wrapped in `select(1.0e9, raw, <expr> >= 0.5)` so the leaf disappears from the
/// scene when the gate evaluates to off at runtime.
fn emit_node_4d(
    node: &SceneNode4,
    counter: &mut u32,
    helpers: &mut String,
    body: &mut String,
    halfspace_gate_expr: Option<&str>,
) -> (String, String) {
    let idx = *counter;
    *counter += 1;
    match node {
        SceneNode4::Leaf(prim) => {
            let fn_name = format!("sdf4_p{idx}");
            helpers.push_str(&prim.to_wgsl_4d(&fn_name));
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            let kind = primitive_kind_constant(prim);
            // Halfspace leaves with an active gate route through `select`; everything else
            // emits the raw SDF call unchanged. Keeping the gate confined to HalfSpace4D
            // avoids polluting hypersphere / polytope SDFs that have no toggle semantic.
            let gated = matches!(prim, Shape::HalfSpace4D { .. }) && halfspace_gate_expr.is_some();
            if gated {
                let gate = halfspace_gate_expr.expect("gated branch implies Some");
                body.push_str(&format!("\tlet {d_var}_raw = {fn_name}(p);\n"));
                body.push_str(&format!(
                    "\tlet {d_var} = select(1.0e9, {d_var}_raw, {gate} >= 0.5);\n"
                ));
            } else {
                body.push_str(&format!("\tlet {d_var} = {fn_name}(p);\n"));
            }
            body.push_str(&format!("\tlet {k_var}: u32 = {kind};\n"));
            (d_var, k_var)
        }
        SceneNode4::Union(left, right) => {
            let (ld, lk) = emit_node_4d(left, counter, helpers, body, halfspace_gate_expr);
            let (rd, rk) = emit_node_4d(right, counter, helpers, body, halfspace_gate_expr);
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            body.push_str(&format!("\tlet {d_var} = min({ld}, {rd});\n"));
            // Closer leaf wins.
            body.push_str(&format!(
                "\tlet {k_var}: u32 = select({rk}, {lk}, {ld} <= {rd});\n"
            ));
            (d_var, k_var)
        }
        SceneNode4::Intersection(left, right) => {
            let (ld, lk) = emit_node_4d(left, counter, helpers, body, halfspace_gate_expr);
            let (rd, rk) = emit_node_4d(right, counter, helpers, body, halfspace_gate_expr);
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            body.push_str(&format!("\tlet {d_var} = max({ld}, {rd});\n"));
            // Farther leaf is the active boundary.
            body.push_str(&format!(
                "\tlet {k_var}: u32 = select({rk}, {lk}, {ld} >= {rd});\n"
            ));
            (d_var, k_var)
        }
        SceneNode4::Difference(left, right) => {
            let (ld, _lk) = emit_node_4d(left, counter, helpers, body, halfspace_gate_expr);
            let (rd, _rk) = emit_node_4d(right, counter, helpers, body, halfspace_gate_expr);
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            body.push_str(&format!("\tlet {d_var} = max({ld}, -({rd}));\n"));
            // Per the to_hyperslice_wgsl docstring: difference's active surface alternates
            // between left's outside and right's inside, no clean per-region kind. Sentinel
            // until a caller needs it.
            body.push_str(&format!("\tlet {k_var}: u32 = RYE_PRIM_OTHER;\n"));
            (d_var, k_var)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_hypersphere_emits_4d_scene() {
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.25));
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("fn rye_scene_sdf_4d(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("length(p"));
        assert!(wgsl.contains("- (0.25)"));
    }

    #[test]
    fn hyperslice_wraps_4d_with_w_slice() {
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.5));
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        // Parameter is `p3` to avoid colliding with the inner `let p` 4D point; WGSL doesn't
        // allow declaring a let with the same name as a function parameter.
        assert!(wgsl.contains("fn rye_scene_sdf(p3: vec3<f32>) -> f32"));
        assert!(wgsl.contains("let p = vec4<f32>(p3, u.w_slice)"));
        // The hyperslice emit reuses the 4D SDF helpers, so the sphere's `length(p - ...)` body
        // is still present.
        assert!(wgsl.contains("length(p"));
    }

    #[test]
    fn union_of_two_hyperspheres() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.2)
                .union(SceneNode4::hypersphere(Vec4::X * 0.5, 0.2)),
        );
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("min("));
        assert!(wgsl.contains("sdf4_p1"));
        assert!(wgsl.contains("sdf4_p2"));
    }

    #[test]
    fn difference_uses_negation_on_4d() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.3).subtract(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("max("));
        assert!(wgsl.contains("-("));
    }

    #[test]
    fn intersection_emits_max() {
        let scene = Scene4::new(
            SceneNode4::halfspace(Vec4::Y, 0.0).intersect(SceneNode4::hypersphere(Vec4::ZERO, 0.4)),
        );
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("max("));
    }

    #[test]
    fn ron_round_trip_4d() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.3).union(SceneNode4::halfspace(Vec4::Y, -0.4)),
        );
        let ron_str = scene.to_ron().expect("serialize");
        let recovered: Scene4 = Scene4::from_ron(&ron_str).expect("deserialize");
        assert_eq!(scene.to_wgsl_4d(), recovered.to_wgsl_4d());
    }

    /// Polytope leaves still emit (their helper returns the sentinel today). The combinator
    /// path doesn't break on polytope leaves; it just produces a far-away surface.
    #[test]
    fn polytope_leaf_emits_sentinel_helper() {
        let scene = Scene4::new(SceneNode4::polytope(vec![Vec4::ZERO; 5]));
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("fn sdf4_p0(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
    }

    /// `to_hyperslice_wgsl` emits the per-primitive identity layer: kind constants, a
    /// `RyeSceneHit` struct, and `rye_scene_at` returning both `dist` and `kind`. The
    /// hyperslice marcher uses `kind` for floor classification (see the kernel's
    /// `kernel_has_expected_entry_points` test).
    #[test]
    fn hyperslice_emits_per_primitive_identity_layer() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        // Constants pinned by name and value; the kernel references them.
        assert!(wgsl.contains("const RYE_PRIM_HYPERSPHERE4D: u32 = 0u;"));
        assert!(wgsl.contains("const RYE_PRIM_HALFSPACE4D: u32 = 1u;"));
        assert!(wgsl.contains("const RYE_PRIM_OTHER: u32 = 255u;"));
        // Result struct + per-primitive entry point.
        assert!(wgsl.contains("struct RyeSceneHit { dist: f32, kind: u32 }"));
        assert!(wgsl.contains("fn rye_scene_at(p3: vec3<f32>) -> RyeSceneHit"));
        // Each leaf must tag its kind constant.
        assert!(wgsl.contains("RYE_PRIM_HYPERSPHERE4D"));
        assert!(wgsl.contains("RYE_PRIM_HALFSPACE4D"));
        // Union routes kind via `select(rhs, lhs, lhs <= rhs)`: closer leaf wins.
        assert!(wgsl.contains("select("));
        assert!(wgsl.contains("<="));
    }

    /// Difference's kind tracking is intentionally undefined (the active surface alternates
    /// between left's outside and right's inside). It emits `RYE_PRIM_OTHER` as a sentinel;
    /// pinning that here so the choice is explicit and a future tightening fails loudly.
    #[test]
    fn hyperslice_difference_emits_kind_sentinel() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).subtract(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        assert!(wgsl.contains(": u32 = RYE_PRIM_OTHER;"));
    }

    /// Gated emit wraps every HalfSpace4D leaf's SDF in a `select(1.0e9, raw, gate >= 0.5)`
    /// so a runtime uniform flip makes the halfspace effectively invisible. Hyperspheres
    /// (and any other non-halfspace primitive) are untouched by the gate.
    #[test]
    fn hyperslice_gated_wraps_halfspaces_only() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        // The halfspace leaf gets routed through select; the gate expression must appear
        // verbatim in the emit. Hypersphere leaf stays a plain `sdfN_pK(p)` call.
        assert!(
            wgsl.contains("select(1.0e9,"),
            "gated halfspace must emit select(1.0e9, ...)"
        );
        assert!(
            wgsl.contains("u.params.x >= 0.5"),
            "gate expression must appear in the select"
        );
        // The hypersphere leaf's helper is still called raw (no select wrapper around it).
        // The first leaf (sdf4_p1, since p0 is the union root in traversal order;
        // leaves are hypersphere=p1, halfspace=p2 in pre-order). Assert that at least
        // one leaf binds without `_raw`.
        assert!(wgsl.contains("let d1 = sdf4_p1(p);"));
    }

    /// Gated emit also wraps the `rye_scene_max_t` halfspace contribution in
    /// `if (gate >= 0.5) { ... }` so a gated-off halfspace doesn't terminate rays early.
    /// Without this, the marcher would still treat the floor as a far-clip even after
    /// the SDF goes to 1.0e9, capping near-horizon rays prematurely.
    #[test]
    fn hyperslice_gated_skips_max_t_when_off() {
        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        let wgsl = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        // The if-guard appears in the max_t body around the t_max update.
        assert!(
            wgsl.contains("if (u.params.x >= 0.5) {"),
            "gated max_t must guard the halfspace's t-contribution"
        );
        // The `t_max = t` update is still emitted inside the guarded block.
        assert!(wgsl.contains("t_max = t;"));
    }

    /// The ungated `to_hyperslice_wgsl` must produce identical output to the gated form
    /// with no halfspace gate active (i.e. when there are no halfspaces at all to gate).
    /// Catches regressions where the gated path diverges from the canonical emit on a
    /// halfspace-free scene.
    #[test]
    fn hyperslice_gated_matches_ungated_when_no_halfspaces() {
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.5));
        let ungated = scene.to_hyperslice_wgsl("u.w_slice");
        let gated = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        assert_eq!(
            ungated, gated,
            "scenes without halfspaces shouldn't diverge under gating"
        );
    }
}
