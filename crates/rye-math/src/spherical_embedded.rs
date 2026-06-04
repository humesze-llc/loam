//! Spherical 3-space (S³) in the **full ambient embedding**: points are unit
//! 4-vectors in R⁴, not a chart.
//!
//! ## Why a second S³ type
//!
//! [`crate::SphericalS3`] models S³ as the **upper hemisphere only**: its
//! `Point` is a `Vec3` with `|p| < 1`, lifted to `(p, √(1−|p|²))`. That keeps
//! the WGSL ABI at `vec3<f32>` for the ray-marched fractal demo, but it cannot
//! represent a point with `w ≤ 0`. A 4-polytope inscribed in S³ has vertices
//! all over the sphere (the tesseract reaches `w = ±0.5` after unit-circumradius
//! normalization), so the hemisphere chart collapses every `w < 0` vertex onto
//! the equator. Wrong geometry for wireframes.
//!
//! [`SphericalS3Embedded`] takes the opposite tradeoff: `Point = Vec4`
//! constrained to the unit sphere `|p| = 1`. Full coverage, no chart seam, exact
//! great-circle geodesics. The cost is that it has no `vec3` WGSL ABI (none is
//! implemented here; there is no shader consumer), so it serves the CPU-side
//! rasterizer wireframe path, not the SDF ray-marcher.
//!
//! ## Geometry
//!
//! Curvature `K = +1`. Geodesics are great circles. The exp / log / parallel
//! transport formulas are the standard unit-sphere maps (Absil, Mahony &
//! Sepulchre, *Optimization Algorithms on Matrix Manifolds*, 2008, §3.6 and
//! Example 8.1.1); the slerp tessellation is Shoemake's spherical linear
//! interpolation (*Animating Rotation with Quaternion Curves*, SIGGRAPH 1985).
//!
//! Isometries reuse [`crate::spherical::Iso4`] (an SO(4) matrix), shared with
//! the hemisphere model: an SO(4) matrix rotates the whole sphere the same way
//! regardless of which chart names its points.

use glam::Vec4;

use crate::rasterizable::{Projection, RasterizableSpace};
use crate::space::Space;
use crate::spherical::Iso4;
use crate::EuclideanR4;

/// Floor on the tangent-direction norm below which a geodesic has no
/// well-defined direction. Two cases hit it: near-coincident endpoints (the
/// perpendicular component `p1 − ⟨p0,p1⟩·p0` vanishes because `p1 ≈ p0`) and
/// near-antipodal endpoints (it vanishes because `p1 ≈ −p0`, and the connecting
/// great circle is non-unique). The same norm gates `exp` / `log`. Equal to the
/// tangent-norm floor the hemisphere model uses, so the two S³ impls agree on
/// "too close to have a direction."
const GEODESIC_EPSILON: f32 = 1e-7;

/// Spherical 3-space, full ambient embedding, curvature `K = +1`.
///
/// Stateless unit struct. Points are unit 4-vectors (`|p| = 1`); the trait
/// methods assume that precondition holds and clamp dot products into range
/// rather than re-normalizing on the hot path. [`RasterizableSpace::array_to_point`]
/// normalizes on the way in, so mesh-storage round-trips land back on the sphere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3Embedded;

impl Space for SphericalS3Embedded {
    type Point = Vec4;
    /// Ambient tangent vector: lives in R⁴, perpendicular to its base point.
    /// [`Self::exp`] projects out any radial component defensively, so a caller
    /// that passes a not-quite-tangent vector still gets an on-sphere result.
    type Vector = Vec4;
    type Iso = Iso4;

    fn distance(&self, a: Vec4, b: Vec4) -> f32 {
        // Chord half-angle form `d = 2·asin(|a − b| / 2)`, the same one
        // `SphericalS3::distance` uses. Better conditioned near `d = 0` than
        // `acos(dot)`, where `acos(1 − ε)` quantizes badly in f32.
        let half_chord = (a - b).length() * 0.5;
        2.0 * half_chord.clamp(0.0, 1.0).asin()
    }

    fn exp(&self, at: Vec4, v: Vec4) -> Vec4 {
        // Project `v` onto the tangent space at `at` (drop any radial part) so
        // the result is exactly on the sphere even if the caller's vector drifted
        // off-tangent. For a true tangent this subtracts zero.
        let v_tan = v - v.dot(at) * at;
        let theta = v_tan.length();
        if theta < GEODESIC_EPSILON {
            return at;
        }
        // Unit-sphere exponential: cos(θ)·at + sin(θ)·(v̂). Stays on S³ because
        // `at` and `v_tan/θ` are orthonormal.
        at * theta.cos() + v_tan * (theta.sin() / theta)
    }

    fn log(&self, from: Vec4, to: Vec4) -> Vec4 {
        let d = self.distance(from, to);
        // Component of `to` perpendicular to `from`: the initial geodesic
        // direction. `dot` clamped because a slightly-off-unit input could push
        // it past 1 and flip the sign of the perpendicular term.
        let dot = from.dot(to).clamp(-1.0, 1.0);
        let perp = to - dot * from;
        let n = perp.length();
        if n < GEODESIC_EPSILON {
            // Coincident (d ≈ 0) or antipodal (d ≈ π, direction undefined): no
            // well-defined geodesic tangent. Matches `SphericalS3::log`.
            return Vec4::ZERO;
        }
        perp * (d / n)
    }

    fn parallel_transport(&self, from: Vec4, to: Vec4, v: Vec4) -> Vec4 {
        // Unit-sphere parallel transport along the minimizing geodesic:
        //   v' = v − (⟨v, to⟩ / (1 + ⟨from, to⟩))·(from + to)
        // (do Carmo, *Riemannian Geometry*, ch. 2; the standard sphere
        // connection.) Acts on the full ambient 4-vector here, not a lifted
        // chart vector. Undefined at antipodes (`⟨from, to⟩ = −1`, the geodesic
        // is non-unique); the denominator floor keeps the result finite there.
        //
        // The denominator is `|from + to|² / 2`, NOT the literal `1 + ⟨from,to⟩`.
        // The two are equal for unit `from`, `to` (expand `|from + to|²
        // = 2 + 2⟨from,to⟩`), but `1 + ⟨from,to⟩` is a catastrophic
        // cancellation as `⟨from,to⟩ -> −1`: near the antipode the f32 sum loses
        // most of its significant digits and the transported vector's norm
        // drifts several percent off (the connection should preserve it
        // exactly). Forming the denominator from the squared length of the same
        // `from + to` the numerator already needs has no cancellation, so the
        // norm holds to f32 epsilon right up to the antipodal floor. This is the
        // `2·cos²(θ/2)` half-angle identity, the same conditioning principle as
        // the chord-form distance. Deliberate divergence from the sibling
        // `SphericalS3::parallel_transport`, which still carries the unhardened
        // `1 + c` form on its hemisphere chart (its near-equator clamp masks the
        // issue there); fold this form back into the sibling when it is hardened.
        let sum = from + to;
        let denom = (sum.length_squared() * 0.5).max(GEODESIC_EPSILON);
        v - (v.dot(to) / denom) * sum
    }

    fn iso_identity(&self) -> Iso4 {
        Iso4::IDENTITY
    }

    fn iso_compose(&self, a: Iso4, b: Iso4) -> Iso4 {
        Iso4 {
            matrix: a.matrix * b.matrix,
        }
    }

    fn iso_inverse(&self, a: Iso4) -> Iso4 {
        // SO(4) matrices are orthogonal: M⁻¹ = Mᵀ.
        Iso4 {
            matrix: a.matrix.transpose(),
        }
    }

    fn iso_apply(&self, iso: Iso4, p: Vec4) -> Vec4 {
        // Normalize after the matmul to shed accumulated f32 drift; repeated
        // SO(4) applications would otherwise creep off the unit sphere.
        (iso.matrix * p).normalize()
    }

    fn iso_transport(&self, iso: Iso4, _at: Vec4, v: Vec4) -> Vec4 {
        // An SO(4) matrix is a global linear isometry of R⁴, so its differential
        // at every point is the matrix itself. Applying it to the ambient tangent
        // vector is exact and base-point-independent (it preserves the tangency
        // ⟨M·at, M·v⟩ = ⟨at, v⟩ = 0). No geodesic round-trip needed, unlike the
        // chart-based hemisphere model.
        iso.matrix * v
    }
}

impl RasterizableSpace<4> for SphericalS3Embedded {
    fn point_to_array(p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 4]) -> Vec4 {
        // Mesh storage may carry slightly-off-sphere values; project back onto
        // S³. Inputs are polytope vertices (circumradius > 0), never the zero
        // vector, so `normalize` is well-defined.
        Vec4::from_array(arr).normalize()
    }

    fn project_point(point: Vec4, projection: &Projection<4>) -> glam::Vec3 {
        match projection {
            // Stereographic is the canonical conformal S³ to R³ map, so a true
            // spherical view computes it directly here rather than delegating to
            // the flat R⁴ projection (which has no notion of the sphere). Points
            // are already unit by this type's invariant, so no normalize is
            // needed; `EuclideanR4`'s arm normalizes because its inputs are
            // body-scaled, but here that would be a redundant op on a unit vector.
            Projection::Stereographic { pole } => {
                crate::rasterizable::stereographic_to_r3(point, *pole)
            }
            // The remaining variants project the ambient unit 4-vector exactly
            // as flat R4 does. The playground views S3 great-circle arcs through
            // the same Perspective4D pinhole or Schlegel diagram it uses for
            // flat wireframes. The flat edge and curved counterpart share one
            // screen embedding, so the lerp to slerp morph reads as the edge
            // bowing out, not as a projection change.
            Projection::Identity
            | Projection::Orthographic { .. }
            | Projection::Perspective4D { .. }
            | Projection::Schlegel { .. } => {
                <EuclideanR4 as RasterizableSpace<4>>::project_point(point, projection)
            }
        }
    }

    fn tessellate_segment(p0: Vec4, p1: Vec4, samples: usize, out: &mut Vec<Vec4>) {
        // Constant-speed walk along the great-circle arc from `p0` to `p1`, both
        // unit 4-vectors, in the exponential-map form (Absil/Mahony/Sepulchre
        // §3.6; equivalent to Shoemake's slerp, *Animating Rotation with
        // Quaternion Curves*, SIGGRAPH 1985):
        //   γ(t) = cos(t·ω)·p0 + sin(t·ω)·d̂,   d̂ = (p1 − ⟨p0,p1⟩·p0) / |·|
        // with `d̂` the unit tangent at `p0` pointing toward `p1`. This form
        // divides only by `n`, the length of the perpendicular component before
        // it is normalized (`n = sin(ω)`), never by `sin(ω)` formed separately
        // from a reconstructed angle, so it stays on S³ to machine epsilon as
        // ω -> π, where the classic `sin((1−t)ω)/sin(ω)` slerp catastrophically
        // loses the sphere (the f32 `acos(dot)` near `dot = −1` recovers an ω
        // whose `sin` mismatches the endpoints, and samples drift percent-level
        // off the sphere well before `sin(ω)` underflows). Deliberate divergence
        // from the spec's literal "guard the lerp normalize" fix: that only
        // rescues the exact-antipode midpoint NaN and leaves the near-antipode
        // off-sphere drift, which this well-conditioned form removes wholesale.
        //
        // `ω` uses the chord half-angle `2·asin(|p0−p1|/2)`, the same
        // well-conditioned form as `Self::distance`, rather than `acos(dot)`,
        // which quantizes badly near both 0 and π.
        //
        // Sampling convention matches `EuclideanR4::tessellate_segment`: push the
        // exact endpoint `p0`, then interior points at t = i/samples for
        // i in 1..samples, then the exact endpoint `p1`, never clearing `out`
        // (the upload loop reuses the buffer).
        let dot = p0.dot(p1).clamp(-1.0, 1.0);
        let half_chord = (p0 - p1).length() * 0.5;
        let omega = 2.0 * half_chord.clamp(0.0, 1.0).asin();
        // Tangent at `p0` toward `p1`: the component of `p1` perpendicular to
        // `p0`. Its pre-normalize length `n = sin(ω)` vanishes for both
        // coincident (ω ≈ 0) and antipodal (ω ≈ π) endpoints; both leave the
        // geodesic direction undefined, so they share the degenerate branch.
        let perp = p1 - dot * p0;
        let n = perp.length();
        out.push(p0);
        if n > GEODESIC_EPSILON {
            let dir = perp / n;
            for i in 1..samples {
                let t = i as f32 / samples as f32;
                let ang = t * omega;
                out.push(ang.cos() * p0 + ang.sin() * dir);
            }
        } else {
            // Degenerate: endpoints near-coincident OR near-antipodal, so the
            // toward-`p1` direction is undefined (at a true antipode, infinitely
            // many great circles connect the pair; for coincident points the arc
            // has zero length). Walk SOME deterministic great circle through `p0`
            // instead of dividing by the zero `perp`: pick a fixed perpendicular
            // `d̂` from `p0` and sample γ(t) = cos(t·ω)·p0 + sin(t·ω)·d̂. For
            // coincident points ω ≈ 0 so every sample collapses onto `p0`; for
            // antipodes ω ≈ π so the walk traverses a half great circle and the
            // final interior sample approaches −p0 = p1, all unit, no NaN. (The
            // old normalized-lerp fallback produced the zero vector at the
            // antipodal midpoint, whose `normalize()` is NaN.)
            let dir = deterministic_perp(p0);
            for i in 1..samples {
                let t = i as f32 / samples as f32;
                let ang = t * omega;
                out.push(ang.cos() * p0 + ang.sin() * dir);
            }
        }
        out.push(p1);
    }
}

/// A deterministic unit vector perpendicular to the unit `p0`, used only by the
/// degenerate (near-coincident / near-antipodal) branch of slerp where the
/// toward-`p1` direction is undefined. Picks the world axis least aligned with
/// `p0` (smallest `|component|`, so its residual after projecting out `p0` is the
/// longest and best-conditioned), Gram-Schmidt's it against `p0`, and normalizes
/// (do Carmo, *Differential Geometry of Curves and Surfaces*, §1.4). Ties resolve
/// toward the earliest axis (x, y, z, w), so the choice is a pure function of
/// `p0`: bit-reproducible, Tier 0. A unit `p0` cannot be aligned with all four
/// axes, so the chosen residual is always well clear of zero.
fn deterministic_perp(p0: Vec4) -> Vec4 {
    let a = p0.abs();
    // Index of the smallest-magnitude component (least-aligned world axis). The
    // `<` comparisons resolve ties toward the earlier index for determinism.
    let mut min_idx = 0usize;
    let mut min_v = a.x;
    if a.y < min_v {
        min_v = a.y;
        min_idx = 1;
    }
    if a.z < min_v {
        min_v = a.z;
        min_idx = 2;
    }
    if a.w < min_v {
        min_idx = 3;
    }
    let axis = match min_idx {
        0 => Vec4::X,
        1 => Vec4::Y,
        2 => Vec4::Z,
        _ => Vec4::W,
    };
    (axis - axis.dot(p0) * p0).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::PI;

    fn s3() -> SphericalS3Embedded {
        SphericalS3Embedded
    }

    /// Two basis 4-vectors are a quarter great circle apart: `acos(0) = π/2`.
    #[test]
    fn distance_orthonormal_is_quarter_circle() {
        let s = s3();
        assert_relative_eq!(s.distance(Vec4::X, Vec4::Y), PI / 2.0, epsilon = 1e-6);
        assert_relative_eq!(s.distance(Vec4::X, Vec4::W), PI / 2.0, epsilon = 1e-6);
    }

    /// Distance is symmetric, zero on the diagonal, and `π` between antipodes.
    #[test]
    fn distance_symmetric_zero_diag_pi_antipodal() {
        let s = s3();
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5); // already unit
        let b = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        assert_relative_eq!(s.distance(a, b), s.distance(b, a), epsilon = 1e-6);
        assert_relative_eq!(s.distance(a, a), 0.0, epsilon = 1e-7);
        assert_relative_eq!(s.distance(a, -a), PI, epsilon = 1e-5);
    }

    /// `exp(at, log(at, to)) == to` for non-antipodal points.
    #[test]
    fn exp_log_round_trip() {
        let s = s3();
        let from = Vec4::new(0.2, 0.1, -0.3, 0.9).normalize();
        let to = Vec4::new(-0.1, 0.4, 0.2, 0.8).normalize();
        let recovered = s.exp(from, s.log(from, to));
        assert_relative_eq!(recovered.x, to.x, epsilon = 1e-5);
        assert_relative_eq!(recovered.y, to.y, epsilon = 1e-5);
        assert_relative_eq!(recovered.z, to.z, epsilon = 1e-5);
        assert_relative_eq!(recovered.w, to.w, epsilon = 1e-5);
    }

    /// `log`'s magnitude equals the geodesic distance, and its direction is
    /// tangent to the sphere at `from` (perpendicular to `from`).
    #[test]
    fn log_magnitude_is_distance_and_is_tangent() {
        let s = s3();
        let from = Vec4::new(0.3, 0.2, 0.1, 0.9).normalize();
        let to = Vec4::new(-0.2, 0.5, 0.0, 0.8).normalize();
        let v = s.log(from, to);
        assert_relative_eq!(v.length(), s.distance(from, to), epsilon = 1e-5);
        // Tangent vector is perpendicular to the base point.
        assert_relative_eq!(v.dot(from), 0.0, epsilon = 1e-6);
    }

    /// `exp` lands exactly on the unit sphere even when the supplied vector has a
    /// radial component (the impl projects it out).
    #[test]
    fn exp_stays_on_sphere_with_non_tangent_input() {
        let s = s3();
        let at = Vec4::new(0.0, 0.0, 0.0, 1.0);
        // Mix a genuine tangent (xy-plane) with a radial part along `at`.
        let v = Vec4::new(0.4, 0.2, 0.0, 0.7);
        let moved = s.exp(at, v);
        assert_relative_eq!(moved.length(), 1.0, epsilon = 1e-6);
    }

    /// Parallel transport preserves the tangent vector's length and lands it in
    /// the tangent space at the destination. `v` lies in the xw-plane of motion
    /// so the transport actually rotates it (a y-direction vector would be left
    /// untouched, a weaker check).
    #[test]
    fn parallel_transport_preserves_norm_and_tangency() {
        let s = s3();
        let from = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let to = Vec4::new(0.6, 0.0, 0.0, 0.8); // unit, in the xw-plane
        let v = Vec4::new(0.5, 0.0, 0.0, 0.0); // tangent at `from`, in-plane
        let vt = s.parallel_transport(from, to, v);
        assert_relative_eq!(vt.length(), v.length(), epsilon = 1e-5);
        assert_relative_eq!(vt.dot(to), 0.0, epsilon = 1e-5);
        // Transport is non-trivial: the in-plane vector is rotated, not fixed.
        assert!((vt - v).length() > 1e-3, "in-plane vector should rotate");
    }

    /// Norm preservation must hold right up to the antipodal floor, not just at
    /// the well-conditioned quarter turn `parallel_transport_preserves_norm_and_tangency`
    /// checks. This pins the well-conditioned denominator: the literal
    /// `1 + ⟨from, to⟩` form cancels catastrophically as `⟨from, to⟩ -> −1` and
    /// drifts the transported norm several percent off across the band
    /// `ω ∈ [π − 3e-3, π − 1e-3]`; the `|from + to|² / 2` form holds it to f32
    /// epsilon. `to` is built as an exact unit vector at great-circle angle `ω`
    /// from `from` in the x-y plane, and `v` is a unit tangent in that plane (so
    /// transport genuinely rotates it), making the norm the discriminating
    /// invariant. `ω` is kept just outside the antipodal floor (`sin(ω) > 1e-3`)
    /// so the geodesic direction is still well defined and the exact transported
    /// norm is `|v| = 1`; the regime where transport is genuinely undefined
    /// (`sin(ω) < GEODESIC_EPSILON`) is the separate antipode-safety concern the
    /// slerp degenerate branch covers.
    #[test]
    fn parallel_transport_preserves_norm_near_antipode() {
        let s = s3();
        let from = Vec4::X;
        for delta in [3e-3_f32, 2e-3, 1e-3] {
            let omega = PI - delta;
            // Unit endpoint at exact great-circle angle `omega` from `from`.
            let to = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();
            // Unit tangent at `from` in the plane of motion (perpendicular to
            // `from`, so transport rotates it rather than leaving it fixed).
            let v = Vec4::Y;
            let vt = s.parallel_transport(from, to, v);
            assert_relative_eq!(vt.length(), v.length(), epsilon = 1e-3);
            assert_relative_eq!(vt.dot(to), 0.0, epsilon = 1e-3);
        }
    }

    /// SO(4) isometry preserves geodesic distance.
    #[test]
    fn iso_apply_preserves_distance() {
        let s = s3();
        // An honest SO(4) element: the hemisphere model's translation constructor
        // builds a Givens rotation in the plane spanned by the w-axis and the
        // target direction.
        let iso = Iso4::from_translation(glam::Vec3::new(0.3, 0.1, -0.2));
        let a = Vec4::new(0.2, 0.3, 0.1, 0.9).normalize();
        let b = Vec4::new(-0.1, 0.2, 0.4, 0.8).normalize();
        let before = s.distance(a, b);
        let after = s.distance(s.iso_apply(iso, a), s.iso_apply(iso, b));
        assert_relative_eq!(before, after, epsilon = 1e-5);
    }

    /// `iso_transport` sends a tangent at `at` to a tangent at `iso_apply(at)`,
    /// preserving length (SO(4) is an isometry of ambient R⁴).
    #[test]
    fn iso_transport_keeps_tangency_and_norm() {
        let s = s3();
        let iso = Iso4::from_translation(glam::Vec3::new(0.2, -0.1, 0.15));
        let at = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let v = Vec4::new(0.3, 0.2, 0.0, 0.0); // tangent at the north pole
        let moved_at = s.iso_apply(iso, at);
        let moved_v = s.iso_transport(iso, at, v);
        assert_relative_eq!(moved_v.length(), v.length(), epsilon = 1e-5);
        assert_relative_eq!(moved_v.dot(moved_at), 0.0, epsilon = 1e-5);
    }

    // ---- RasterizableSpace ----------------------------------------------

    /// `point_to_array` then `array_to_point` is identity for a unit input
    /// (`array_to_point` normalizes, so the round-trip only closes on S³).
    #[test]
    fn array_round_trip_on_unit_input() {
        let p = Vec4::new(0.5, -0.5, 0.5, 0.5); // unit by construction
        let arr = <SphericalS3Embedded as RasterizableSpace<4>>::point_to_array(p);
        let back = <SphericalS3Embedded as RasterizableSpace<4>>::array_to_point(arr);
        assert_relative_eq!(back.x, p.x, epsilon = 1e-6);
        assert_relative_eq!(back.y, p.y, epsilon = 1e-6);
        assert_relative_eq!(back.z, p.z, epsilon = 1e-6);
        assert_relative_eq!(back.w, p.w, epsilon = 1e-6);
    }

    /// Slerp endpoints are exact and the sample count matches the convention:
    /// `samples` subdivisions append `samples + 1` points.
    #[test]
    fn slerp_endpoints_exact_and_count() {
        let p0 = Vec4::X;
        let p1 = Vec4::Y;
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 4, &mut out);
        assert_eq!(out.len(), 5);
        assert_relative_eq!(out[0].x, p0.x, epsilon = 1e-6);
        assert_relative_eq!(out[4].y, p1.y, epsilon = 1e-6);
    }

    /// Every slerp sample is a unit vector (lies on S³), unlike the chord, whose
    /// interior dips inside the sphere.
    #[test]
    fn slerp_samples_stay_on_sphere() {
        let p0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 8, &mut out);
        for p in &out {
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
        }
    }

    /// The slerp midpoint of a quarter arc sits at 45°, bulging off the chord:
    /// it is equidistant from both endpoints and its components are `sin(π/4)`,
    /// strictly larger in magnitude than the chord midpoint `(0.5, 0, 0, 0.5)`.
    #[test]
    fn slerp_midpoint_is_on_great_circle_not_chord() {
        let s = s3();
        let p0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 2, &mut out);
        let mid = out[1];
        // Equidistant from both endpoints.
        assert_relative_eq!(s.distance(mid, p0), s.distance(mid, p1), epsilon = 1e-6);
        // On the sphere at 45°: x = w = cos(π/4) ≈ 0.7071, not the chord's 0.5.
        let c = (PI / 4.0).cos();
        assert_relative_eq!(mid.x, c, epsilon = 1e-5);
        assert_relative_eq!(mid.w, c, epsilon = 1e-5);
        assert!(mid.x > 0.5, "slerp midpoint must bulge off the chord");
    }

    /// Exact antipodes have no unique connecting great circle, but the public
    /// trait method must still return finite, on-sphere samples (no zero-vector
    /// `normalize` -> NaN). The degenerate branch walks a deterministic
    /// perpendicular great circle from `p0`; every sample is finite and unit.
    /// Unreachable from polytope edges (adjacent vertices subtend small angles),
    /// but the method is public and must be antipode-safe.
    #[test]
    fn slerp_antipode_produces_finite_unit_samples() {
        let p0 = Vec4::X;
        let p1 = -Vec4::X;
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 16, &mut out);
        assert_eq!(out.len(), 17);
        for p in &out {
            assert!(
                p.is_finite(),
                "antipodal slerp sample must be finite: {p:?}"
            );
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
        }
    }

    /// Near-antipodal endpoints are the case the old single-`sin(ω)` gate missed:
    /// for ω just below π the perpendicular norm passes the gate, but the classic
    /// `sin((1−t)ω)/sin(ω)` slerp drifts percent-level off the sphere because the
    /// f32 `acos(dot)` near `dot = −1` recovers an ω whose `sin` no longer matches
    /// the endpoints. The well-conditioned exp-form keeps every interior sample on
    /// S³. Tested across ω = π − {1e-3, 1e-5, 2e-7}: each endpoint is placed at the
    /// exact angle in the x-y plane so the test pins the impl, not the test setup.
    #[test]
    fn slerp_near_antipode_samples_stay_on_sphere() {
        let p0 = Vec4::X;
        for delta in [1e-3_f32, 1e-5, 2e-7] {
            let omega = PI - delta;
            // Unit endpoint at the exact great-circle angle `omega` from `p0`.
            let p1 = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();
            let mut out = Vec::new();
            <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 16, &mut out);
            // Interior samples (skip the exact endpoints, which are pushed verbatim).
            for p in &out[1..out.len() - 1] {
                assert!(p.is_finite(), "near-antipode sample must be finite: {p:?}");
                assert!(
                    (p.length() - 1.0).abs() < 1e-4,
                    "near-antipode (omega = PI - {delta:e}) sample off-sphere: |p| = {}",
                    p.length()
                );
            }
        }
    }

    /// Constant-speed parametrization: the geodesic distances between consecutive
    /// samples sum to the total endpoint distance, for any sample count. This is
    /// the correct formalization of "samples cover the great-circle arc" and pins
    /// that the walk does not bunch up or overshoot. A chord-lerp would undershoot
    /// (sum of chords < arc); slerp's sum of sub-arcs equals the whole arc.
    #[test]
    fn slerp_consecutive_arc_sum_equals_total() {
        let s = s3();
        let p0 = Vec4::new(0.2, 0.1, -0.3, 0.9).normalize();
        let p1 = Vec4::new(-0.1, 0.4, 0.2, 0.8).normalize();
        let total = s.distance(p0, p1);
        for samples in [2usize, 3, 8, 17] {
            let mut out = Vec::new();
            <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(
                p0, p1, samples, &mut out,
            );
            let arc_sum: f32 = out.windows(2).map(|w| s.distance(w[0], w[1])).sum();
            assert_relative_eq!(arc_sum, total, epsilon = 1e-6);
        }
    }

    /// Determinism is Tier 0: tessellating the same segment twice yields
    /// byte-identical samples. The existing on-sphere / midpoint tests only assert
    /// approximate equality; this asserts exact f32 bit-reproducibility, which the
    /// fixed op order (no reassociation, no FMA contraction) must guarantee.
    #[test]
    fn slerp_is_bit_reproducible() {
        let p0 = Vec4::new(0.3, -0.2, 0.5, 0.4).normalize();
        let p1 = Vec4::new(-0.4, 0.1, 0.2, 0.7).normalize();
        let mut a = Vec::new();
        let mut b = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 12, &mut a);
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 12, &mut b);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            // Bit-for-bit, not approx: same inputs must give the same f32 bits.
            assert_eq!(x.to_array(), y.to_array(), "slerp not bit-reproducible");
        }
        // The antipodal degenerate branch must be reproducible too (the
        // deterministic perpendicular is a pure function of `p0`).
        let mut c = Vec::new();
        let mut d = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(
            Vec4::Y,
            -Vec4::Y,
            9,
            &mut c,
        );
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(
            Vec4::Y,
            -Vec4::Y,
            9,
            &mut d,
        );
        for (x, y) in c.iter().zip(&d) {
            assert_eq!(
                x.to_array(),
                y.to_array(),
                "antipodal slerp not bit-reproducible"
            );
        }
    }

    /// Near-coincident endpoints (ω < GEODESIC_EPSILON) take the degenerate branch
    /// and must still emit unit samples, all clustered at `p0` (the arc has
    /// effectively zero length). Broadens the angle coverage of
    /// `slerp_samples_stay_on_sphere`, which only exercises a quarter arc.
    #[test]
    fn slerp_near_coincident_falls_back_on_sphere() {
        let p0 = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        // A point a hair off `p0`: omega well below GEODESIC_EPSILON = 1e-7.
        let p1 = (p0 + Vec4::new(1e-9, 0.0, -1e-9, 0.0)).normalize();
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 8, &mut out);
        for p in &out {
            assert!(p.is_finite(), "coincident sample must be finite: {p:?}");
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
            // Zero-length arc: every sample sits essentially at p0.
            assert!(
                s3().distance(*p, p0) < 1e-4,
                "coincident-arc sample should stay at p0, dist {}",
                s3().distance(*p, p0)
            );
        }
    }

    /// `project_point` agrees with the flat R⁴ projection it delegates to, on the
    /// canonical Perspective4D path.
    #[test]
    fn project_point_matches_flat_r4() {
        let p = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <SphericalS3Embedded as RasterizableSpace<4>>::project_point(p, &proj);
        let want = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        assert_eq!(got, want);
    }

    /// Schlegel delegates to the flat R4 projection too: central projection is
    /// on the ambient R4 embedding, so an S3 vertex and its flat counterpart
    /// share one screen point.
    /// Pins that the blanket delegation covers the Schlegel variant, not just Perspective4D.
    #[test]
    fn project_point_schlegel_matches_flat_r4() {
        let p = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let proj = Projection::schlegel(Vec4::W, 0.5, 0.75);
        let got = <SphericalS3Embedded as RasterizableSpace<4>>::project_point(p, &proj);
        let want = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        assert_eq!(got, want);
    }

    /// Stereographic does NOT blanket-delegate: the embedded S3 type computes
    /// the conformal map directly, with no normalize because the input is unit.
    /// It returns the stereographic image, not the flat R4 drop-w. Pins that a
    /// true spherical view uses stereographic projection.
    #[test]
    fn project_point_stereographic_is_conformal_map_not_drop_w() {
        let p = Vec4::new(0.5, 0.5, 0.5, 0.5); // unit by construction
        let proj = Projection::Stereographic { pole: Vec4::W };
        let got = <SphericalS3Embedded as RasterizableSpace<4>>::project_point(p, &proj);
        // Closed-form stereographic for the +w pole.
        let want = glam::Vec3::new(p.x, p.y, p.z) / (1.0 - p.w);
        assert_relative_eq!(got.x, want.x, epsilon = 1e-6);
        assert_relative_eq!(got.y, want.y, epsilon = 1e-6);
        assert_relative_eq!(got.z, want.z, epsilon = 1e-6);
        // It is genuinely the stereographic scaling, not the identity drop-w `(x, y, z)`.
        let drop_w = glam::Vec3::new(p.x, p.y, p.z);
        assert!(
            (got - drop_w).length() > 1e-3,
            "stereographic must scale by 1/(1-w), not pass through drop-w; got {got:?}"
        );
    }
}
