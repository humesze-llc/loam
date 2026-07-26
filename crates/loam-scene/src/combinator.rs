//! WGSL combinator helpers.
//!
//! Each combinator emits a named WGSL function that takes pre-evaluated distance values and
//! returns a combined distance. The Scene tree calls the SDF sub-functions first, stores results in
//! `let` bindings, then passes them to the combinator function.
//!
//! All combinators are Space-agnostic: they operate on scalar distances returned by
//! `loam_distance`-based SDF functions, so they are correct in E³, H³, and S³ without
//! modification.

use crate::literal::wgsl_f32;

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
///
/// `k` is baked as a shortest-round-trip literal, never at fixed precision: it
/// is a divisor, and a six-decimal print collapses every `k` below 5e-7 to
/// `0.000000`, dividing by zero on the GPU while a CPU evaluation of the same
/// scene stays finite.
pub fn smooth_min_fn(name: &str, k: f32) -> String {
    let k = wgsl_f32(k);
    format!(
        "fn {name}(a: f32, b: f32) -> f32 {{\n\
         \tlet h = clamp(0.5 + 0.5 * (b - a) / ({k}), 0.0, 1.0);\n\
         \treturn mix(b, a, h) - ({k}) * h * (1.0 - h);\n\
         }}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extract the WGSL literal `smooth_min_fn` divides by.
    fn divisor_literal(src: &str) -> &str {
        let after = src.split("(b - a) / (").nth(1).expect("divide is emitted");
        after.split(')').next().expect("divisor is parenthesized")
    }

    /// The baked divisor must round-trip to the requested `k` for every `k` a
    /// caller can pass, including radii far below the `{:.6}` print floor that
    /// used to collapse to `0.000000`.
    #[test]
    fn smooth_min_divisor_round_trips_to_k() {
        for k in [0.08_f32, 4.9e-7, 1e-7, 1e-20, 1e-30] {
            let src = smooth_min_fn("smin", k);
            let literal = divisor_literal(&src);
            let parsed: f32 = literal.parse().expect("divisor parses as f32");
            assert_eq!(parsed, k, "divisor `{literal}` does not round-trip to {k}");
            assert_ne!(parsed, 0.0, "k = {k} emitted a zero divisor");
        }
    }

    /// Both occurrences of `k` (divisor and blend term) come from one value, so
    /// they must be the same literal; a divergence would make the emitted
    /// smooth-min discontinuous.
    #[test]
    fn smooth_min_bakes_one_literal_for_both_uses_of_k() {
        let src = smooth_min_fn("smin", 1e-7);
        let literal = divisor_literal(&src);
        assert_eq!(src.matches(literal).count(), 2, "emitted:\n{src}");
    }
}
