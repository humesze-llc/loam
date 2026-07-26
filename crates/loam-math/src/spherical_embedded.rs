//! Spherical 3-space (S³) in the **full ambient embedding**: points are unit
//! 4-vectors in R⁴, not a chart.
//!
//! [`crate::SphericalS3`] is the upper-hemisphere chart (`Vec3`, `vec3<f32>`
//! WGSL ABI for the fractal demo) and cannot represent `w ≤ 0`, so it collapses
//! every `w < 0` polytope vertex onto the equator. [`SphericalS3Embedded`] takes
//! `Point = Vec4` on the unit sphere: full coverage, no chart seam, exact
//! great-circle geodesics, but no WGSL ABI, so it serves the CPU rasterizer
//! wireframe path, not the SDF ray-marcher.
//!
//! Curvature `K = +1`; geodesics are great circles. The exp / log / transport
//! maps are the standard unit-sphere forms (Absil, Mahony & Sepulchre,
//! *Optimization Algorithms on Matrix Manifolds*, 2008, §3.6, Example 8.1.1);
//! slerp is Shoemake (*Animating Rotation with Quaternion Curves*, SIGGRAPH
//! 1985). Isometries reuse [`crate::spherical::Iso4`] (an SO(4) matrix), shared
//! with the hemisphere model.

use glam::Vec4;

use crate::rasterizable::{Projection, RasterizableSpace};
use crate::space::Space;
use crate::spherical::Iso4;
use crate::EuclideanR4;

/// Floor on the tangent-direction norm below which a geodesic has no defined
/// direction: near-coincident (`p1 ≈ p0`) or near-antipodal (`p1 ≈ −p0`, the
/// great circle is non-unique). Also gates `exp` / `log`. Matches the hemisphere
/// model's floor so the two S³ impls agree on "too close to have a direction."
const GEODESIC_EPSILON: f32 = 1e-7;

/// Spherical 3-space, full ambient embedding, curvature `K = +1`.
///
/// Stateless unit struct. Points are unit 4-vectors (`|p| = 1`); methods assume
/// that and clamp dot products rather than re-normalizing on the hot path.
/// [`RasterizableSpace::array_to_point`] normalizes on the way in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3Embedded;

impl Space for SphericalS3Embedded {
    type Point = Vec4;
    /// Ambient tangent vector in R⁴, perpendicular to its base point.
    /// [`Self::exp`] projects out any radial component.
    type Vector = Vec4;
    type Iso = Iso4;

    fn distance(&self, a: Vec4, b: Vec4) -> f32 {
        // Chord half-angle `d = 2·asin(|a − b| / 2)`: better conditioned near
        // `d = 0` than `acos(dot)`, where `acos(1 − ε)` quantizes in f32.
        let half_chord = (a - b).length() * 0.5;
        2.0 * half_chord.clamp(0.0, 1.0).asin()
    }

    fn exp(&self, at: Vec4, v: Vec4) -> Vec4 {
        // Drop any radial part so the result lands exactly on the sphere even if
        // the caller's vector drifted off-tangent.
        let v_tan = v - v.dot(at) * at;
        let theta = v_tan.length();
        if theta < GEODESIC_EPSILON {
            return at;
        }
        // Unit-sphere exponential cos(θ)·at + sin(θ)·v̂; `at` and `v_tan/θ` are
        // orthonormal.
        at * theta.cos() + v_tan * (theta.sin() / theta)
    }

    fn log(&self, from: Vec4, to: Vec4) -> Vec4 {
        let d = self.distance(from, to);
        // Component of `to` perpendicular to `from`, the initial geodesic
        // direction. `dot` clamped so a slightly-off-unit input cannot flip the
        // sign of the perpendicular term.
        let dot = from.dot(to).clamp(-1.0, 1.0);
        let perp = to - dot * from;
        let n = perp.length();
        if n < GEODESIC_EPSILON {
            // Coincident or antipodal: no defined tangent.
            return Vec4::ZERO;
        }
        perp * (d / n)
    }

    fn parallel_transport(&self, from: Vec4, to: Vec4, v: Vec4) -> Vec4 {
        // Unit-sphere transport `v − (⟨v, to⟩ / denom)·(from + to)` (do Carmo,
        // *Riemannian Geometry*, ch. 2). Undefined at antipodes; the floor keeps
        // it finite.
        //
        // The denominator is `|from + to|² / 2`, equal to `1 + ⟨from, to⟩` for
        // unit inputs but without its catastrophic cancellation as
        // `⟨from, to⟩ -> −1`; computed from the same `from + to` the numerator
        // needs, the transported norm holds to f32 epsilon up to the floor (the
        // `2·cos²(θ/2)` half-angle identity, same principle as the chord-form
        // distance).
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
        // Normalize to shed accumulated f32 drift across repeated applications.
        (iso.matrix * p).normalize()
    }

    fn iso_transport(&self, iso: Iso4, _at: Vec4, v: Vec4) -> Vec4 {
        // An SO(4) matrix is a global linear isometry, so its differential is the
        // matrix itself: exact and base-point-independent, no geodesic round-trip
        // unlike the chart-based hemisphere model.
        iso.matrix * v
    }
}

impl RasterizableSpace<4> for SphericalS3Embedded {
    fn point_to_array(p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 4]) -> Vec4 {
        // Mesh storage may drift off-sphere; project back. Inputs are polytope
        // vertices, never zero, so `normalize` is well-defined.
        Vec4::from_array(arr).normalize()
    }

    fn project_point(point: Vec4, projection: &Projection<4>) -> glam::Vec3 {
        match projection {
            // Compute the conformal map directly (a true spherical view) rather
            // than delegating to flat R⁴. No normalize: the input is already unit
            // by this type's invariant.
            Projection::Stereographic { pole } => {
                crate::rasterizable::stereographic_to_r3(point, *pole)
            }
            // The other variants project the ambient unit 4-vector exactly as
            // flat R⁴ does, so an S³ edge and its flat counterpart share one
            // screen embedding and the lerp-to-slerp morph reads as the edge
            // bowing out, not a projection change.
            Projection::Identity
            | Projection::Orthographic { .. }
            | Projection::Perspective4D { .. }
            | Projection::Schlegel { .. } => {
                <EuclideanR4 as RasterizableSpace<4>>::project_point(point, projection)
            }
        }
    }

    fn tessellate_segment(p0: Vec4, p1: Vec4, samples: usize, out: &mut Vec<Vec4>) {
        // Constant-speed walk along the great-circle arc in exponential-map form
        // (Absil/Mahony/Sepulchre §3.6; Shoemake slerp, *Animating Rotation with
        // Quaternion Curves*, SIGGRAPH 1985):
        //   γ(t) = cos(t·ω)·p0 + sin(t·ω)·d̂,  d̂ = (p1 − ⟨p0,p1⟩·p0) / n
        // This divides only by the pre-normalize perpendicular length `n = sin(ω)`,
        // never by a `sin(ω)` reconstructed from `acos(dot)`, so it stays on S³ to
        // machine epsilon as ω -> π, where the classic `sin((1−t)ω)/sin(ω)` slerp
        // drifts percent-level off the sphere. `ω` uses the chord half-angle
        // `2·asin(|p0−p1|/2)`, same well-conditioned form as `Self::distance`.
        //
        // Sampling convention matches `EuclideanR4::tessellate_segment`.
        let dot = p0.dot(p1).clamp(-1.0, 1.0);
        let half_chord = (p0 - p1).length() * 0.5;
        let omega = 2.0 * half_chord.clamp(0.0, 1.0).asin();
        // `n = sin(ω)` vanishes for both coincident and antipodal endpoints; both
        // leave the direction undefined and share the degenerate branch.
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
            // Direction undefined (coincident or antipodal): walk a deterministic
            // perpendicular great circle through `p0` rather than dividing by the
            // zero `perp`. Coincident: ω ≈ 0, samples collapse onto `p0`.
            // Antipodal: ω ≈ π, the final sample approaches −p0 = p1, all unit,
            // no NaN (the old normalized-lerp produced a zero-vector midpoint).
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

/// A deterministic unit vector perpendicular to unit `p0`, for the degenerate
/// slerp branch. Picks the world axis least aligned with `p0` (best-conditioned
/// residual), Gram-Schmidt's it against `p0`, normalizes (do Carmo, *Differential
/// Geometry of Curves and Surfaces*, §1.4). Ties resolve toward the earliest
/// axis, so it is a pure function of `p0` (Tier 0). A unit `p0` cannot align with
/// all four axes, so the residual is always clear of zero.
fn deterministic_perp(p0: Vec4) -> Vec4 {
    let a = p0.abs();
    // Smallest-magnitude component; `<` ties toward the earlier index.
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

    #[test]
    fn distance_orthonormal_is_quarter_circle() {
        let s = s3();
        assert_relative_eq!(s.distance(Vec4::X, Vec4::Y), PI / 2.0, epsilon = 1e-6);
        assert_relative_eq!(s.distance(Vec4::X, Vec4::W), PI / 2.0, epsilon = 1e-6);
    }

    /// The antipode is the cut locus, so it is the one distance the conformance
    /// suite's samples deliberately exclude.
    #[test]
    fn distance_at_antipode_is_pi() {
        let s = s3();
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5); // unit
        assert_relative_eq!(s.distance(a, -a), PI, epsilon = 1e-5);
    }

    #[test]
    fn log_magnitude_is_distance_and_is_tangent() {
        let s = s3();
        let from = Vec4::new(0.3, 0.2, 0.1, 0.9).normalize();
        let to = Vec4::new(-0.2, 0.5, 0.0, 0.8).normalize();
        let v = s.log(from, to);
        assert_relative_eq!(v.length(), s.distance(from, to), epsilon = 1e-5);
        assert_relative_eq!(v.dot(from), 0.0, epsilon = 1e-6);
    }

    /// `exp` lands on the sphere even with a radial component in the input.
    #[test]
    fn exp_stays_on_sphere_with_non_tangent_input() {
        let s = s3();
        let at = Vec4::new(0.0, 0.0, 0.0, 1.0);
        // Tangent plus a radial part along `at`.
        let v = Vec4::new(0.4, 0.2, 0.0, 0.7);
        let moved = s.exp(at, v);
        assert_relative_eq!(moved.length(), 1.0, epsilon = 1e-6);
    }

    /// Transport preserves norm and tangency. `v` is in the xw-plane of motion so
    /// it actually rotates (a y-vector would be left fixed, a weaker check).
    #[test]
    fn parallel_transport_preserves_norm_and_tangency() {
        let s = s3();
        let from = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let to = Vec4::new(0.6, 0.0, 0.0, 0.8); // unit, in the xw-plane
        let v = Vec4::new(0.5, 0.0, 0.0, 0.0); // tangent at `from`, in-plane
        let vt = s.parallel_transport(from, to, v);
        assert_relative_eq!(vt.length(), v.length(), epsilon = 1e-5);
        assert_relative_eq!(vt.dot(to), 0.0, epsilon = 1e-5);
        assert!((vt - v).length() > 1e-3, "in-plane vector should rotate");
    }

    /// Norm preservation up to the antipodal floor pins the well-conditioned
    /// denominator: the literal `1 + ⟨from, to⟩` drifts the norm percent-level as
    /// `⟨from, to⟩ -> −1` while `|from + to|² / 2` holds it to f32 epsilon. `ω` is
    /// kept just outside the floor (`sin(ω) > 1e-3`) so the exact norm is `|v| = 1`;
    /// the genuinely-undefined regime is the slerp degenerate branch's concern.
    #[test]
    fn parallel_transport_preserves_norm_near_antipode() {
        let s = s3();
        let from = Vec4::X;
        for delta in [3e-3_f32, 2e-3, 1e-3] {
            let omega = PI - delta;
            let to = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();
            // Unit tangent in the plane of motion, so transport rotates it.
            let v = Vec4::Y;
            let vt = s.parallel_transport(from, to, v);
            assert_relative_eq!(vt.length(), v.length(), epsilon = 1e-3);
            assert_relative_eq!(vt.dot(to), 0.0, epsilon = 1e-3);
        }
    }

    /// `iso_transport` sends a tangent at `at` to a tangent at `iso_apply(at)`,
    /// preserving length.
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

    /// Round-trip closes only on S³ since `array_to_point` normalizes.
    #[test]
    fn array_round_trip_on_unit_input() {
        let p = Vec4::new(0.5, -0.5, 0.5, 0.5); // unit
        let arr = <SphericalS3Embedded as RasterizableSpace<4>>::point_to_array(p);
        let back = <SphericalS3Embedded as RasterizableSpace<4>>::array_to_point(arr);
        assert_relative_eq!(back.x, p.x, epsilon = 1e-6);
        assert_relative_eq!(back.y, p.y, epsilon = 1e-6);
        assert_relative_eq!(back.z, p.z, epsilon = 1e-6);
        assert_relative_eq!(back.w, p.w, epsilon = 1e-6);
    }

    /// Slerp endpoints are exact; `samples` subdivisions append `samples + 1`.
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

    /// Every slerp sample is unit, unlike the chord whose interior dips inside.
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

    /// The quarter-arc midpoint sits at 45° on the sphere, bulging off the chord
    /// midpoint `(0.5, 0, 0, 0.5)`.
    #[test]
    fn slerp_midpoint_is_on_great_circle_not_chord() {
        let s = s3();
        let p0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 2, &mut out);
        let mid = out[1];
        assert_relative_eq!(s.distance(mid, p0), s.distance(mid, p1), epsilon = 1e-6);
        // At 45°: x = w = cos(π/4), not the chord's 0.5.
        let c = (PI / 4.0).cos();
        assert_relative_eq!(mid.x, c, epsilon = 1e-5);
        assert_relative_eq!(mid.w, c, epsilon = 1e-5);
        assert!(mid.x > 0.5, "slerp midpoint must bulge off the chord");
    }

    /// Exact antipodes must still yield finite, on-sphere samples (no zero-vector
    /// `normalize` -> NaN). Unreachable from polytope edges, but the method is
    /// public and must be antipode-safe.
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

    /// Near-antipodal endpoints (ω just below π) pass the perp-norm gate but break
    /// the classic `sin((1−t)ω)/sin(ω)` slerp; the exp-form keeps every interior
    /// sample on S³. Endpoints placed at exact angles so the test pins the impl.
    #[test]
    fn slerp_near_antipode_samples_stay_on_sphere() {
        let p0 = Vec4::X;
        for delta in [1e-3_f32, 1e-5, 2e-7] {
            let omega = PI - delta;
            let p1 = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();
            let mut out = Vec::new();
            <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 16, &mut out);
            // Skip the verbatim endpoints.
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

    /// Constant-speed: consecutive sub-arc distances sum to the total, for any
    /// sample count. A chord-lerp would undershoot.
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

    /// Tier 0: tessellating the same segment twice is byte-identical (exact f32
    /// bit-reproducibility, not just approximate as the other tests check).
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
            assert_eq!(x.to_array(), y.to_array(), "slerp not bit-reproducible");
        }
        // The degenerate branch must be reproducible too.
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

    /// Near-coincident endpoints take the degenerate branch and emit unit samples
    /// clustered at `p0`. Broadens the angle coverage of
    /// `slerp_samples_stay_on_sphere`.
    #[test]
    fn slerp_near_coincident_falls_back_on_sphere() {
        let p0 = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        // omega well below GEODESIC_EPSILON.
        let p1 = (p0 + Vec4::new(1e-9, 0.0, -1e-9, 0.0)).normalize();
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 8, &mut out);
        for p in &out {
            assert!(p.is_finite(), "coincident sample must be finite: {p:?}");
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
            assert!(
                s3().distance(*p, p0) < 1e-4,
                "coincident-arc sample should stay at p0, dist {}",
                s3().distance(*p, p0)
            );
        }
    }

    /// `project_point` delegates to flat R⁴ on the Perspective4D path.
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

    /// Schlegel delegates to flat R⁴ too, not just Perspective4D.
    #[test]
    fn project_point_schlegel_matches_flat_r4() {
        let p = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let proj = Projection::schlegel(Vec4::W, 0.5, 0.75);
        let got = <SphericalS3Embedded as RasterizableSpace<4>>::project_point(p, &proj);
        let want = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        assert_eq!(got, want);
    }

    /// Stereographic does NOT delegate: it returns the conformal image, not the
    /// flat R⁴ drop-w.
    #[test]
    fn project_point_stereographic_is_conformal_map_not_drop_w() {
        let p = Vec4::new(0.5, 0.5, 0.5, 0.5); // unit
        let proj = Projection::Stereographic { pole: Vec4::W };
        let got = <SphericalS3Embedded as RasterizableSpace<4>>::project_point(p, &proj);
        let want = glam::Vec3::new(p.x, p.y, p.z) / (1.0 - p.w);
        assert_relative_eq!(got.x, want.x, epsilon = 1e-6);
        assert_relative_eq!(got.y, want.y, epsilon = 1e-6);
        assert_relative_eq!(got.z, want.z, epsilon = 1e-6);
        let drop_w = glam::Vec3::new(p.x, p.y, p.z);
        assert!(
            (got - drop_w).length() > 1e-3,
            "stereographic must scale by 1/(1-w), not pass through drop-w; got {got:?}"
        );
    }
}
