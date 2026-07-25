//! Typed 4D scene tree, the 4D analogue of [`crate::scene::Scene`].
//!
//! Build a [`Scene4`] from [`SceneNode4`] combinators and emit WGSL for either:
//!
//! - **Native 4D**: `fn loam_scene_sdf_4d(p: vec4<f32>) -> f32` (future full-4D
//!   ray march).
//! - **Hyperslice**: `fn loam_scene_sdf(p: vec3<f32>) -> f32` evaluating the 4D SDF
//!   at `vec4(p, w_slice)`. Production path today; `Hyperslice4DNode` consumes it.
//!
//! Parallel to [`crate::scene::Scene`] rather than `Scene<S, const DIM>`: the 3D
//! and 4D paths share no shader code, so genericizing saves nothing and obscures
//! the difference.
//!
//! ## Example
//!
//! ```rust
//! use glam::Vec4;
//! use loam_scene::scene4::{Scene4, SceneNode4};
//!
//! let scene = Scene4::new(
//!     SceneNode4::hypersphere(Vec4::ZERO, 0.5)
//!         .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
//! );
//! // Native 4D: SDF takes vec4 directly.
//! let wgsl_4d = scene.to_wgsl_4d();
//! assert!(wgsl_4d.contains("fn loam_scene_sdf_4d(p: vec4<f32>) -> f32"));
//! // Hyperslice mode: SDF takes vec3, internally evaluates at
//! // vec4(p, u.w_slice). The `u.w_slice` uniform is supplied by
//! // the render node.
//! let wgsl_hs = scene.to_hyperslice_wgsl("u.w_slice");
//! assert!(wgsl_hs.contains("fn loam_scene_sdf(p3: vec3<f32>) -> f32"));
//! assert!(wgsl_hs.contains("u.w_slice"));
//! ```

use std::boxed::Box;

use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::primitive4::Primitive4;
use crate::SENTINEL_DISTANCE;
pub use loam_shape::Shape;

/// A node in the 4D scene tree. Mirrors [`crate::scene::SceneNode`] over 4D
/// primitives. No smooth-min yet (math is identical; no demo needs it).
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

    /// Half-space (hyperplane) leaf. ℝ⁴ is flat and the only 4D Space loam ships, so
    /// [`crate::Primitive4`] emits a plain `dot(p, n) - offset` SDF; a curved 4D
    /// Space would gate this on `Space::is_chart_flat` like the 3D path. Also
    /// used by `loam-physics` for 4D collision walls.
    pub fn halfspace(normal: Vec4, offset: f32) -> Self {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset })
    }

    /// Convex 4D polytope leaf. The static `Primitive4` emit returns a sentinel;
    /// the live path is `Hyperslice4DNode`'s per-frame uniform buffer (face
    /// hyperplanes computed CPU-side). Until that lands, polytope leaves are
    /// invisible.
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

/// A complete 4D SDF scene: a single root [`SceneNode4`] emitting either the
/// native-4D or hyperslice SDF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene4 {
    pub root: SceneNode4,
}

impl Scene4 {
    pub fn new(root: SceneNode4) -> Self {
        Self { root }
    }

    /// Emit `fn loam_scene_sdf_4d(p: vec4<f32>) -> f32` for future full-4D ray
    /// marchers. Kind-tracking bindings are emitted but unused.
    pub fn to_wgsl_4d(&self) -> String {
        let mut helpers = String::new();
        let mut body = String::new();
        let mut counter = 0u32;
        let (d_root, _k_root) =
            emit_node_4d(&self.root, &mut counter, &mut helpers, &mut body, None);
        let kind_consts = scene_kind_constants();
        format!(
            "// ---- loam-scene scene4 (native 4D) ----\n\
             {kind_consts}\
             {helpers}\
             fn loam_scene_sdf_4d(p: vec4<f32>) -> f32 {{\n\
             {body}\
             \treturn {d_root};\n\
             }}\n"
        )
    }

    /// Emit the hyperslice SDF module: kind constants, `LoamSceneHit`,
    /// `loam_scene_at(p3) -> LoamSceneHit`, `loam_scene_sdf(p3) -> f32`, and
    /// `loam_scene_max_t(ro, rd) -> f32` (analytical far-clip from HalfSpace4D
    /// leaves). `w_slice_expr` is the slicing w-coord, typically `"u.w_slice"`.
    ///
    /// Kind tracking: union picks the closer leaf, intersection the farther
    /// (boundary) leaf, difference returns `LOAM_PRIM_OTHER`.
    pub fn to_hyperslice_wgsl(&self, w_slice_expr: &str) -> String {
        emit_hyperslice(self, w_slice_expr, None)
    }

    /// Like [`Self::to_hyperslice_wgsl`] but with a runtime gate on every
    /// [`Shape::HalfSpace4D`] leaf: when `halfspace_gate_expr` is `< 0.5` the
    /// halfspace SDF returns [`SENTINEL_DISTANCE`] and its `loam_scene_max_t`
    /// contribution is skipped; `>= 0.5` matches the ungated emit. Lets a uniform
    /// flip floor / ceiling / cutaway planes per frame without recompiling.
    ///
    /// `halfspace_gate_expr` is a scalar `f32` WGSL expression in scope at
    /// `loam_scene_at` / `loam_scene_max_t` (e.g. `"u.params.x"`, `"1.0"`, `"0.0"`).
    pub fn to_hyperslice_wgsl_gated(
        &self,
        w_slice_expr: &str,
        halfspace_gate_expr: &str,
    ) -> String {
        emit_hyperslice(self, w_slice_expr, Some(halfspace_gate_expr))
    }

    /// CPU twin of the emitted `loam_scene_at`: the signed distance at
    /// `vec4(p3, w_slice)` and the kind of the primitive that produced it.
    ///
    /// `halfspace_gate` is the value the gate uniform holds this frame; the CPU
    /// cannot evaluate [`Self::to_hyperslice_wgsl_gated`]'s WGSL expression
    /// string, so the caller supplies its truth value. Pass `true` to mirror the
    /// ungated [`Self::to_hyperslice_wgsl`], whose emit is identical to a gate
    /// that is always on.
    pub fn eval_at(&self, p3: Vec3, w_slice: f32, halfspace_gate: bool) -> (f32, u32) {
        eval_node_4d(&self.root, p3.extend(w_slice), halfspace_gate)
    }

    /// Signed distance alone, the `.dist` projection of [`Self::eval_at`],
    /// exactly as the emitter derives `loam_scene_sdf` from `loam_scene_at`.
    pub fn eval(&self, p3: Vec3, w_slice: f32, halfspace_gate: bool) -> f32 {
        self.eval_at(p3, w_slice, halfspace_gate).0
    }

    pub fn from_ron(src: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(src)
    }

    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }
}

// Kind of the primitive a `Scene4` hit belongs to, as returned by
// `Scene4::eval_at` and as emitted into WGSL by `scene_kind_constants`. The
// marcher routes floor classification on these.

/// A [`Shape::HyperSphere4D`] leaf.
pub const PRIM_KIND_HYPERSPHERE4D: u32 = 0;
/// A [`Shape::HalfSpace4D`] leaf.
pub const PRIM_KIND_HALFSPACE4D: u32 = 1;
/// A `Difference` node, which has no single owning primitive, or a leaf with no
/// 4D closed form.
pub const PRIM_KIND_OTHER: u32 = 255;

/// WGSL kind constants emitted atop every Scene4 module, formatted from the
/// `PRIM_KIND_*` values so the shader and [`Scene4::eval_at`] cannot drift.
fn scene_kind_constants() -> String {
    format!(
        "const LOAM_PRIM_HYPERSPHERE4D: u32 = {PRIM_KIND_HYPERSPHERE4D}u;\n\
         const LOAM_PRIM_HALFSPACE4D: u32 = {PRIM_KIND_HALFSPACE4D}u;\n\
         const LOAM_PRIM_OTHER: u32 = {PRIM_KIND_OTHER}u;\n"
    )
}

/// Shared emit driver for the two `to_hyperslice_wgsl*` methods. `None` produces
/// the ungated form; `Some` wraps every `HalfSpace4D` leaf's SDF in
/// `select(SENTINEL_DISTANCE, raw, <expr> >= 0.5)` and skips its
/// `loam_scene_max_t` term.
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
    let kind_consts = scene_kind_constants();
    let max_t_body = emit_max_t_body(&scene.root, halfspace_gate_expr);
    // Parameter `p3` with an inner `let p` for the 4D point: WGSL forbids a `let`
    // shadowing the parameter name, and helpers expect to call `sdfN_pK(p)`.
    format!(
        "// ---- loam-scene scene4 (hyperslice at w = {w_slice_expr}) ----\n\
         {kind_consts}\
         struct LoamSceneHit {{ dist: f32, kind: u32 }}\n\
         {helpers}\
         fn loam_scene_at(p3: vec3<f32>) -> LoamSceneHit {{\n\
         \tlet p = vec4<f32>(p3, {w_slice_expr});\n\
         {body}\
         \treturn LoamSceneHit({d_root}, {k_root});\n\
         }}\n\
         fn loam_scene_sdf(p3: vec3<f32>) -> f32 {{\n\
         \treturn loam_scene_at(p3).dist;\n\
         }}\n\
         // Analytical upper bound on march distance: ray-plane intersection\n\
         // for every HalfSpace4D leaf in the scene whose 3D-projected\n\
         // normal points against the ray. Returns +infinity if no leaf\n\
         // contributes; the kernel uses this to terminate near-horizon\n\
         // rays that would otherwise exhaust the iteration budget.\n\
         fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {{\n\
         \tvar t_max: f32 = {SENTINEL_DISTANCE:e};\n\
         {max_t_body}\
         \treturn t_max;\n\
         }}\n"
    )
}

/// Emit the body of `loam_scene_max_t`: a ray-plane intersection per `HalfSpace4D`
/// leaf, folded into `t_max` via `min`. Only the 3D normal is used (the slice
/// fixes `p.w`); combinator-agnostic, visits every leaf.
fn emit_max_t_body(node: &SceneNode4, halfspace_gate_expr: Option<&str>) -> String {
    let mut body = String::new();
    walk_max_t(node, &mut body, halfspace_gate_expr);
    body
}

fn walk_max_t(node: &SceneNode4, body: &mut String, halfspace_gate_expr: Option<&str>) {
    match node {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset }) => {
            // t = (offset - dot(ro, n)) / dot(rd, n), guarded by dot(rd, n) < 0 to
            // catch only rays heading toward the solid side. A set gate wraps the
            // whole t-contribution so a gated-off halfspace yields no bound.
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

/// Map a Shape variant to its WGSL kind-constant name and its numeric value.
/// One match, two consumers, so the emitted identifier and the value
/// [`Scene4::eval_at`] returns are the same decision.
fn primitive_kind(shape: &Shape) -> (&'static str, u32) {
    match shape {
        Shape::HyperSphere4D { .. } => ("LOAM_PRIM_HYPERSPHERE4D", PRIM_KIND_HYPERSPHERE4D),
        Shape::HalfSpace4D { .. } => ("LOAM_PRIM_HALFSPACE4D", PRIM_KIND_HALFSPACE4D),
        _ => ("LOAM_PRIM_OTHER", PRIM_KIND_OTHER),
    }
}

/// Walk the 4D scene tree, appending helpers to `helpers` and `let` bindings to
/// `body`. Returns `(dist_var, kind_var)`, the WGSL identifiers for this node's
/// distance and closest-primitive kind. `Some` gate wraps each `HalfSpace4D` SDF
/// in `select(SENTINEL_DISTANCE, raw, <expr> >= 0.5)`.
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
            let (kind, _kind_value) = primitive_kind(prim);
            // Only gated halfspaces route through `select`; everything else emits
            // the raw call (no toggle semantic on hypersphere / polytope SDFs).
            let gated = matches!(prim, Shape::HalfSpace4D { .. }) && halfspace_gate_expr.is_some();
            if gated {
                let gate = halfspace_gate_expr.expect("gated branch implies Some");
                body.push_str(&format!("\tlet {d_var}_raw = {fn_name}(p);\n"));
                body.push_str(&format!(
                    "\tlet {d_var} = select({SENTINEL_DISTANCE:e}, {d_var}_raw, {gate} >= 0.5);\n"
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
            // Difference has no clean per-region kind (the active surface
            // alternates between left's outside and right's inside); sentinel.
            body.push_str(&format!("\tlet {k_var}: u32 = LOAM_PRIM_OTHER;\n"));
            (d_var, k_var)
        }
    }
}

/// Scalar twin of [`emit_node_4d`], one arm per variant in the same order,
/// returning `(dist, kind)` where the emitter returns the names of the two
/// `let` bindings. Allocation-free.
fn eval_node_4d(node: &SceneNode4, p: Vec4, halfspace_gate: bool) -> (f32, u32) {
    match node {
        SceneNode4::Leaf(prim) => {
            let (_kind_name, kind) = primitive_kind(prim);
            let gated = matches!(prim, Shape::HalfSpace4D { .. }) && !halfspace_gate;
            let dist = if gated {
                SENTINEL_DISTANCE
            } else {
                prim.eval_4d(p)
            };
            (dist, kind)
        }
        SceneNode4::Union(left, right) => {
            let (ld, lk) = eval_node_4d(left, p, halfspace_gate);
            let (rd, rk) = eval_node_4d(right, p, halfspace_gate);
            (ld.min(rd), if ld <= rd { lk } else { rk })
        }
        SceneNode4::Intersection(left, right) => {
            let (ld, lk) = eval_node_4d(left, p, halfspace_gate);
            let (rd, rk) = eval_node_4d(right, p, halfspace_gate);
            (ld.max(rd), if ld >= rd { lk } else { rk })
        }
        SceneNode4::Difference(left, right) => {
            let (ld, _lk) = eval_node_4d(left, p, halfspace_gate);
            let (rd, _rk) = eval_node_4d(right, p, halfspace_gate);
            (ld.max(-rd), PRIM_KIND_OTHER)
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
        assert!(wgsl.contains("fn loam_scene_sdf_4d(p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("length(p"));
        assert!(wgsl.contains("- (0.25)"));
    }

    #[test]
    fn hyperslice_wraps_4d_with_w_slice() {
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.5));
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        assert!(wgsl.contains("fn loam_scene_sdf(p3: vec3<f32>) -> f32"));
        assert!(wgsl.contains("let p = vec4<f32>(p3, u.w_slice)"));
        // Reuses the 4D SDF helpers, so the sphere body is still present.
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

    /// Polytope leaves still emit (helper returns the sentinel); the combinator
    /// path produces a far-away surface, not a break.
    #[test]
    fn polytope_leaf_emits_sentinel_helper() {
        let scene = Scene4::new(SceneNode4::polytope(vec![Vec4::ZERO; 5]));
        let wgsl = scene.to_wgsl_4d();
        assert!(wgsl.contains("fn sdf4_p0(_p: vec4<f32>) -> f32"));
        assert!(wgsl.contains("return 1e9"));
    }

    /// `to_hyperslice_wgsl` emits the per-primitive identity layer: kind
    /// constants, `LoamSceneHit`, and `loam_scene_at` returning `dist` + `kind`
    /// (the marcher uses `kind` for floor classification).
    #[test]
    fn hyperslice_emits_per_primitive_identity_layer() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        assert!(wgsl.contains("const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;"));
        assert!(wgsl.contains("const LOAM_PRIM_HALFSPACE4D: u32 = 1u;"));
        assert!(wgsl.contains("const LOAM_PRIM_OTHER: u32 = 255u;"));
        assert!(wgsl.contains("struct LoamSceneHit { dist: f32, kind: u32 }"));
        assert!(wgsl.contains("fn loam_scene_at(p3: vec3<f32>) -> LoamSceneHit"));
        assert!(wgsl.contains("LOAM_PRIM_HYPERSPHERE4D"));
        assert!(wgsl.contains("LOAM_PRIM_HALFSPACE4D"));
        // Union routes kind via select: closer leaf wins.
        assert!(wgsl.contains("select("));
        assert!(wgsl.contains("<="));
    }

    /// Difference's kind is intentionally the `LOAM_PRIM_OTHER` sentinel; pin it so
    /// a future tightening fails loudly.
    #[test]
    fn hyperslice_difference_emits_kind_sentinel() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).subtract(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        assert!(wgsl.contains(": u32 = LOAM_PRIM_OTHER;"));
    }

    /// Gated emit wraps every HalfSpace4D SDF in `select(1e9, raw, gate >= 0.5)`;
    /// non-halfspace primitives are untouched.
    #[test]
    fn hyperslice_gated_wraps_halfspaces_only() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let wgsl = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        assert!(
            wgsl.contains("select(1e9,"),
            "gated halfspace must emit select(<sentinel>, ...)"
        );
        assert!(
            wgsl.contains("u.params.x >= 0.5"),
            "gate expression must appear in the select"
        );
        // Hypersphere leaf (p1 in pre-order) binds raw, no `_raw`/select wrapper.
        assert!(wgsl.contains("let d1 = sdf4_p1(p);"));
    }

    /// Gated emit also guards the `loam_scene_max_t` halfspace contribution in
    /// `if (gate >= 0.5)` so a gated-off halfspace doesn't terminate rays early.
    #[test]
    fn hyperslice_gated_skips_max_t_when_off() {
        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));
        let wgsl = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        assert!(
            wgsl.contains("if (u.params.x >= 0.5) {"),
            "gated max_t must guard the halfspace's t-contribution"
        );
        assert!(wgsl.contains("t_max = t;"));
    }

    /// On a halfspace-free scene the gated emit must match the ungated emit
    /// byte-for-byte; catches gated-path divergence.
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
