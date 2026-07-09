//! WGSL combinator helpers.
//!
//! Each combinator emits a named WGSL function that takes pre-evaluated distance values and
//! returns a combined distance. The Scene tree calls the SDF sub-functions first, stores results in
//! `let` bindings, then passes them to the combinator function.
//!
//! All combinators are Space-agnostic: they operate on scalar distances returned by
//! `loam_distance`-based SDF functions, so they are correct in E³, H³, and S³ without
//! modification.

/// Emit a WGSL expression for the union (minimum) of two distances.
///
/// `da` and `db` must be WGSL `f32` expressions (ideally simple variable names, not function
/// calls, to avoid double evaluation).
pub fn union_expr(da: &str, db: &str) -> String {
    format!("min({da}, {db})")
}

/// Emit a WGSL expression for the intersection (maximum) of two distances.
pub fn intersection_expr(da: &str, db: &str) -> String {
    format!("max({da}, {db})")
}

/// Emit a WGSL expression for the difference A − B (carve B from A).
pub fn difference_expr(da: &str, db: &str) -> String {
    format!("max({da}, -({db}))")
}

/// Emit a named WGSL helper function implementing smooth-minimum (Inigo Quilez polynomial
/// blend).
///
/// `k` controls the blend radius (in Space distance units). The function takes two pre-evaluated
/// distances `(a: f32, b: f32)` and returns the blended distance. Call it as `{name}(da, db)` in
/// the scene body.
pub fn smooth_min_fn(name: &str, k: f32) -> String {
    format!(
        "fn {name}(a: f32, b: f32) -> f32 {{\n\
         \tlet h = clamp(0.5 + 0.5 * (b - a) / {k:.6}, 0.0, 1.0);\n\
         \treturn mix(b, a, h) - {k:.6} * h * (1.0 - h);\n\
         }}\n",
    )
}
