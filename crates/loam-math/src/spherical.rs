//! Spherical 3-space (S³), the third constant-curvature `Space`.
//!
//! Upper-hemisphere model: `Point` is a `Vec3` with `|p| < 1`, lifted to the
//! unit 4-vector `(p, √(1−|p|²))` on S³ ⊂ R⁴; the origin is the north pole.
//! This keeps the WGSL ABI at `vec3<f32>` (the v0 `WgslSpace` contract) at the
//! cost of upper-hemisphere-only coverage: an isometry that pushes a point below
//! the equator returns out-of-domain (a debug warning fires). The fractal demo's
//! `ball_scale` keeps coordinates near the origin, well clear of the equator.
//!
//! Isometries are SO(4) matrices: composition is matmul, inverse is transpose.
//! Curvature `K = +1`; geodesic triangles have positive angle excess.
//!
//! Domain `|p|² < 1`. The boundary `|p| = 1` is the equator where `w = 0` and the
//! tangent formula `vw = −dot(v,p)/w` blows up; methods clamp to a saturation
//! shell at `SPHERE_R2_MAX`, so out-of-domain inputs degrade finitely, never NaN.

use std::borrow::Cow;

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::space::{Space, WgslSpace};

/// Closest `|p|²` allowed to 1.0; keeps `w ≥ ~1e-3` so the tangent formula does
/// not saturate at the equator.
const SPHERE_R2_MAX: f32 = 1.0 - 1e-6;

/// `exp` returns its base point below this `|v|²`. `|v| = 1e-7` is under one
/// f32 ulp of a chart coordinate of order 1, so the step is unrepresentable
/// rather than merely small. It is also the only floor `sin(mag)/mag` needs:
/// the lift appends a component, so `|v4| ≥ |v| ≥ 1e-7` past this return.
const EXP_TANGENT_MIN_SQ: f32 = 1e-14;

/// Floor on `|perp4|` in `log`, the sine of the geodesic angle. Below it the
/// two lifts agree to within their own rounding, so `perp4 / n` would report
/// the direction of that rounding rather than of the geodesic.
const LOG_PERP_MIN: f32 = 1e-7;

/// Below this chart radius `Iso4::from_translation` is the identity: the Givens
/// plane is spanned by `e_w` and the target direction, and the direction is
/// rounding once the target is under one f32 ulp of a coordinate of order 1.
const ISO_TRANSLATION_MIN_ARC: f32 = 1e-7;

fn clamp_to_hemisphere(p: Vec3) -> Vec3 {
    let r2 = p.length_squared();
    if r2 <= SPHERE_R2_MAX {
        p
    } else {
        #[cfg(debug_assertions)]
        tracing::warn!("SphericalS3: point outside upper hemisphere clamped (|p|²={r2:.4})");
        p * (SPHERE_R2_MAX.sqrt() / r2.sqrt())
    }
}

/// Lift a upper-hemisphere point `p` to its unit 4-vector on S³.
fn to_sphere(p: Vec3) -> Vec4 {
    let r2 = p.length_squared().min(SPHERE_R2_MAX);
    Vec4::new(p.x, p.y, p.z, (1.0 - r2).sqrt())
}

/// Project a 4D sphere point back to upper-hemisphere coords by discarding w.
/// Only correct when `q.w ≥ 0`; warns on lower-hemisphere points.
fn from_sphere(q: Vec4) -> Vec3 {
    #[cfg(debug_assertions)]
    if q.w < 0.0 {
        tracing::warn!(
            "SphericalS3: iso_apply moved point to lower hemisphere (w={:.4}); \
             result will be out-of-domain",
            q.w
        );
    }
    q.truncate()
}

/// An orientation-preserving isometry of S³, an SO(4) matrix. Composition is
/// matmul; inverse is transpose.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso4 {
    pub matrix: Mat4,
}

impl Iso4 {
    pub const IDENTITY: Self = Self {
        matrix: Mat4::IDENTITY,
    };

    /// Pure spatial rotation fixing the north pole: SO(3) in the upper-left 3×3
    /// block, identity w row and column.
    pub fn from_rotation(rotation: Quat) -> Self {
        let r = Mat3::from_quat(rotation);
        Self {
            matrix: Mat4::from_cols(
                r.col(0).extend(0.0),
                r.col(1).extend(0.0),
                r.col(2).extend(0.0),
                Vec4::W,
            ),
        }
    }

    /// Geodesic translation mapping the north pole to `target`: a Givens rotation
    /// in the `{e_w, xyz-direction-of-target}` plane by the geodesic distance.
    /// Out-of-domain targets clamp to the saturation shell.
    pub fn from_translation(target: Vec3) -> Self {
        let qt = to_sphere(clamp_to_hemisphere(target));
        let c = qt.w;
        let s = qt.truncate().length();
        if s < ISO_TRANSLATION_MIN_ARC {
            return Self::IDENTITY;
        }
        let n = qt.truncate() / s;
        let k = c - 1.0;

        // Same algebraic form as H³'s Lorentz boost with sinh->sin, cosh->cos,
        // and a sign flip on the (xyz, w) block (SO(4) vs SO⁺(3,1)).
        Self {
            matrix: Mat4::from_cols(
                Vec4::new(1.0 + k * n.x * n.x, k * n.x * n.y, k * n.x * n.z, -s * n.x),
                Vec4::new(k * n.y * n.x, 1.0 + k * n.y * n.y, k * n.y * n.z, -s * n.y),
                Vec4::new(k * n.z * n.x, k * n.z * n.y, 1.0 + k * n.z * n.z, -s * n.z),
                Vec4::new(s * n.x, s * n.y, s * n.z, c),
            ),
        }
    }
}

/// Spherical 3-space, upper hemisphere model, curvature `K = +1`.
///
/// Stateless unit struct. See the [module docs](self) for the representation and
/// domain constraint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3;

impl Space for SphericalS3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = Iso4;

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        let qa = to_sphere(clamp_to_hemisphere(a));
        let qb = to_sphere(clamp_to_hemisphere(b));
        // Chord half-angle `d = 2·asin(|qa − qb| / 2)`: better conditioned for
        // small d than `acos(dot)`, where `acos(1 − ε)` quantizes in f32.
        let half_chord = (qa - qb).length() * 0.5;
        2.0 * half_chord.clamp(0.0, 1.0).asin()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        let at = clamp_to_hemisphere(at);
        if v.length_squared() < EXP_TANGENT_MIN_SQ {
            return at;
        }
        let q = to_sphere(at);
        // Lift v to a 4D tangent perpendicular to q: vw = −dot(v,at)/q.w.
        let vw = -v.dot(at) / q.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        let mag = v4.length();
        let result4 = (q * mag.cos() + v4 * (mag.sin() / mag)).normalize();
        clamp_to_hemisphere(result4.truncate())
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        let qf = to_sphere(clamp_to_hemisphere(from));
        let qt = to_sphere(clamp_to_hemisphere(to));
        let d_dot = qf.dot(qt).clamp(-1.0, 1.0);
        // Component of qt perpendicular to qf, along the geodesic.
        let perp4 = qt - d_dot * qf;
        let n = perp4.length();
        if n < LOG_PERP_MIN {
            return Vec3::ZERO;
        }
        let half_chord = (qt - qf).length() * 0.5;
        let d = 2.0 * half_chord.clamp(0.0, 1.0).asin();
        // Return xyz; w is recovered in exp via the tangent constraint.
        perp4.truncate() * (d / n)
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        let from = clamp_to_hemisphere(from);
        let to = clamp_to_hemisphere(to);
        let qf = to_sphere(from);
        let qt = to_sphere(to);
        let vw = -v.dot(from) / qf.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        // Unit-sphere transport `v4 − (⟨v4, qt⟩ / denom)·(qf + qt)` (do Carmo,
        // *Riemannian Geometry*, ch. 2), with `denom = |qf + qt|² / 2`. The
        // literal `1 + ⟨qf, qt⟩` agrees only for exactly-unit lifts, and the
        // gap is the lifts' own rounding, which near antipodes is the whole
        // denominator. In this form the update is the Householder reflection in
        // the hyperplane normal to `qf + qt` (exact for a tangent `v4`, which
        // the `vw` lift above makes it), so it stays an isometry whatever `qf`
        // and `qt` round to. Unfloored: near-antipodal means near-equator in
        // this chart, and both lifts carry `w ≥ √(1 − SPHERE_R2_MAX)`, so
        // `denom ≥ 2·(1 − SPHERE_R2_MAX)`. `SphericalS3Embedded` takes unit
        // `Vec4` anywhere on S³, where `from = −to` is representable, and its
        // sibling denominator does need a floor.
        let sum = qf + qt;
        let denom = sum.length_squared() * 0.5;
        let v4_transported = v4 - v4.dot(qt) / denom * sum;
        v4_transported.truncate()
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

    fn iso_apply(&self, iso: Iso4, p: Vec3) -> Vec3 {
        from_sphere(iso.matrix * to_sphere(clamp_to_hemisphere(p)))
    }

    fn iso_transport(&self, iso: Iso4, at: Vec3, v: Vec3) -> Vec3 {
        // Exact via the geodesic round-trip identity.
        let target = self.exp(at, v);
        let m_at = self.iso_apply(iso, at);
        let m_target = self.iso_apply(iso, target);
        self.log(m_at, m_target)
    }
}

impl WgslSpace for SphericalS3 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
}

const WGSL_IMPL: &str = r#"
// loam-math :: SphericalS3 (v0 Space WGSL ABI)
// Upper hemisphere: points are vec3 with |p|² < 1, embedded in S³ as
// (p.x, p.y, p.z, sqrt(1 − |p|²)). Origin = north pole (0,0,0,1).
// Cap geodesic arcs well under π so rays cannot wrap past the S³ equator
// and hit the scene from behind. With ball_scale=0.15 the full t_scene=20
// budget only reaches t_arc≈3.0; cap at 1.5 to cut off wraparound while
// leaving the entire front hemisphere reachable (fractal fits in ~0.75).
const LOAM_MAX_ARC: f32 = 1.5;
const LOAM_S3_R2_MAX: f32 = 0.999999;

fn loam_s3_clamp(p: vec3<f32>) -> vec3<f32> {
    let r2 = dot(p, p);
    if (r2 <= LOAM_S3_R2_MAX) { return p; }
    return p * (sqrt(LOAM_S3_R2_MAX) / sqrt(r2));
}

fn loam_s3_lift(p: vec3<f32>) -> vec4<f32> {
    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return vec4<f32>(p.x, p.y, p.z, sqrt(1.0 - r2));
}

fn loam_origin_distance(p: vec3<f32>) -> f32 {
    // Arc from the north pole (0,0,0,1) to the lift (p, √(1−|p|²)) is
    // asin(|p|). The equivalent acos(√(1−|p|²)) loses the small-|p| regime
    // twice over in f32: it collapses to exactly 0 below |p| ≈ 1.73e-4
    // (1−|p|² rounds to 1.0 once |p|² ≤ 2⁻²⁵), and its smallest nonzero
    // output is 3.45e-4 = acos(1−2⁻²⁴), so every radius under that is
    // either zero or overstated (3.6% high at |p| = 1e-3).
    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return asin(sqrt(r2));
}

fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let qa = loam_s3_lift(loam_s3_clamp(a));
    let qb = loam_s3_lift(loam_s3_clamp(b));
    let half_chord = length(qa - qb) * 0.5;
    return 2.0 * asin(clamp(half_chord, 0.0, 1.0));
}

fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let p = loam_s3_clamp(at);
    let n2 = dot(v, v);
    if (n2 < 1e-14) { return p; }
    let q = loam_s3_lift(p);
    let vw = -dot(v, p) / q.w;
    let v4 = vec4<f32>(v.x, v.y, v.z, vw);
    let mag = length(v4);
    let result4 = normalize(q * cos(mag) + v4 * (sin(mag) / mag));
    return loam_s3_clamp(result4.xyz);
}

fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> {
    let qf = loam_s3_lift(loam_s3_clamp(p_from));
    let qt = loam_s3_lift(loam_s3_clamp(p_to));
    let d_dot = clamp(dot(qf, qt), -1.0, 1.0);
    let perp4 = qt - d_dot * qf;
    let n = length(perp4);
    if (n < 1e-7) { return vec3<f32>(0.0, 0.0, 0.0); }
    let half_chord = length(qt - qf) * 0.5;
    let d = 2.0 * asin(clamp(half_chord, 0.0, 1.0));
    return perp4.xyz * (d / n);
}

fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let pf = loam_s3_clamp(p_from);
    let pt = loam_s3_clamp(p_to);
    let qf = loam_s3_lift(pf);
    let qt = loam_s3_lift(pt);
    let vw = -dot(v, pf) / qf.w;
    let v4 = vec4<f32>(v.x, v.y, v.z, vw);
    // `|qf + qt|² / 2` is `1 + dot(qf, qt)` for exactly-unit lifts; in this
    // form the update is a Householder reflection, an isometry whatever the
    // lifts round to. Unfloored: near-antipodal pairs sit near the equator,
    // and both lifts carry w ≥ sqrt(1 − LOAM_S3_R2_MAX), so the denominator
    // cannot fall below 2·(1 − LOAM_S3_R2_MAX).
    let sum = qf + qt;
    let denom = dot(sum, sum) * 0.5;
    let v4t = v4 - (dot(v4, qt) / denom) * sum;
    return v4t.xyz;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn s3() -> SphericalS3 {
        SphericalS3
    }

    #[test]
    fn to_sphere_from_sphere_round_trip() {
        let p = Vec3::new(0.2, -0.3, 0.1);
        let q = to_sphere(p);
        assert_relative_eq!(q.length(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(q.w, (1.0 - p.length_squared()).sqrt(), epsilon = 1e-6);
        assert_relative_eq!(from_sphere(q).x, p.x, epsilon = 1e-6);
        assert_relative_eq!(from_sphere(q).y, p.y, epsilon = 1e-6);
        assert_relative_eq!(from_sphere(q).z, p.z, epsilon = 1e-6);
    }

    #[test]
    fn distance_is_symmetric_and_zero_on_diagonal() {
        let s = s3();
        let a = Vec3::new(0.1, 0.2, 0.3);
        let b = Vec3::new(-0.3, 0.05, 0.15);
        assert_relative_eq!(s.distance(a, b), s.distance(b, a), epsilon = 1e-6);
        assert_relative_eq!(s.distance(a, a), 0.0, epsilon = 1e-7);
    }

    #[test]
    fn distance_at_origin_matches_arc_length() {
        let s = s3();
        // Distance from the north pole to (r, 0, 0) is asin(r).
        let r = 0.4;
        let p = Vec3::new(r, 0.0, 0.0);
        assert_relative_eq!(s.distance(Vec3::ZERO, p), r.asin(), epsilon = 1e-5);
    }

    #[test]
    fn exp_log_round_trip() {
        let s = s3();
        let a = Vec3::new(0.1, -0.2, 0.05);
        let b = Vec3::new(0.25, 0.1, -0.1);
        let recovered = s.exp(a, s.log(a, b));
        assert_relative_eq!(recovered.x, b.x, epsilon = 1e-5);
        assert_relative_eq!(recovered.y, b.y, epsilon = 1e-5);
        assert_relative_eq!(recovered.z, b.z, epsilon = 1e-5);
    }

    #[test]
    fn exp_tiny_vector_clamps_out_of_domain_basepoint() {
        let s = s3();
        let at = Vec3::new(2.0, 0.0, 0.0);
        let tiny = Vec3::new(1e-8, 0.0, 0.0);
        let got = s.exp(at, tiny);
        let want = clamp_to_hemisphere(at);
        assert_relative_eq!(got.x, want.x, epsilon = 1e-6);
        assert_relative_eq!(got.y, want.y, epsilon = 1e-6);
        assert_relative_eq!(got.z, want.z, epsilon = 1e-6);
    }

    #[test]
    fn iso_identity_is_neutral() {
        let s = s3();
        let p = Vec3::new(0.2, -0.3, 0.1);
        let q = s.iso_apply(s.iso_identity(), p);
        assert_relative_eq!(q.x, p.x, epsilon = 1e-6);
        assert_relative_eq!(q.y, p.y, epsilon = 1e-6);
        assert_relative_eq!(q.z, p.z, epsilon = 1e-6);
    }

    #[test]
    fn iso_compose_with_inverse_is_identity() {
        let s = s3();
        let iso = Iso4::from_translation(Vec3::new(0.2, 0.1, -0.15));
        let id_a = s.iso_compose(iso, s.iso_inverse(iso));
        let id_b = s.iso_compose(s.iso_inverse(iso), iso);
        let p = Vec3::new(0.05, -0.1, 0.07);
        for id in [id_a, id_b] {
            let q = s.iso_apply(id, p);
            assert_relative_eq!(q.x, p.x, epsilon = 1e-5);
            assert_relative_eq!(q.y, p.y, epsilon = 1e-5);
            assert_relative_eq!(q.z, p.z, epsilon = 1e-5);
        }
    }

    #[test]
    fn iso_compose_matches_sequential_apply() {
        let s = s3();
        let a = Iso4::from_translation(Vec3::new(0.15, 0.0, 0.0));
        let b = Iso4::from_rotation(Quat::from_rotation_z(0.4));
        let p = Vec3::new(0.05, 0.05, 0.05);
        let composed = s.iso_apply(s.iso_compose(a, b), p);
        let sequential = s.iso_apply(a, s.iso_apply(b, p));
        assert_relative_eq!(composed.x, sequential.x, epsilon = 1e-5);
        assert_relative_eq!(composed.y, sequential.y, epsilon = 1e-5);
        assert_relative_eq!(composed.z, sequential.z, epsilon = 1e-5);
    }

    #[test]
    fn iso_translation_moves_origin_to_target() {
        let s = s3();
        let target = Vec3::new(0.2, -0.1, 0.15);
        let iso = Iso4::from_translation(target);
        let moved = s.iso_apply(iso, Vec3::ZERO);
        assert_relative_eq!(moved.x, target.x, epsilon = 1e-5);
        assert_relative_eq!(moved.y, target.y, epsilon = 1e-5);
        assert_relative_eq!(moved.z, target.z, epsilon = 1e-5);
    }

    #[test]
    fn distance_is_invariant_under_isometry() {
        let s = s3();
        let iso = Iso4::from_rotation(Quat::from_rotation_y(0.8));
        let a = Vec3::new(0.05, 0.0, 0.0);
        let b = Vec3::new(0.1, 0.1, 0.0);
        let d_before = s.distance(a, b);
        let d_after = s.distance(s.iso_apply(iso, a), s.iso_apply(iso, b));
        assert_relative_eq!(d_before, d_after, epsilon = 1e-5);
    }

    #[test]
    fn parallel_transport_preserves_spherical_norm() {
        let s = s3();
        let from = Vec3::ZERO;
        let to = Vec3::new(0.3, 0.0, 0.0);
        let v = Vec3::new(0.0, 0.05, 0.0); // tangent at origin, perpendicular to motion
        let v_to = s.parallel_transport(from, to, v);
        // Spherical norm |v4| (v4 = (v, vw)) must be preserved by transport.
        let norm_from = {
            let qf = to_sphere(from);
            let vw = -v.dot(from) / qf.w;
            Vec4::new(v.x, v.y, v.z, vw).length()
        };
        let norm_to = {
            let qt = to_sphere(to);
            let vw = -v_to.dot(to) / qt.w;
            Vec4::new(v_to.x, v_to.y, v_to.z, vw).length()
        };
        assert_relative_eq!(norm_from, norm_to, epsilon = 1e-5);
    }

    /// Norm preservation just off the antipodal singularity pins the
    /// well-conditioned denominator. `to` is `from` mirrored through the
    /// yz-plane, so the pair is near-antipodal and the two lifts share a
    /// bit-identical `w`; the exact denominator is then `2·(b² + w²)`, which
    /// `1 + ⟨qf, qt⟩` evaluates as `1 − a² + b² + w²` and loses to cancellation
    /// while `|qf + qt|² / 2` reads it off components that are already small.
    /// Only the near-equator band reaches the regime: `1 + ⟨qf, qt⟩ ≥ 2·w²`,
    /// and the saturation shell floors `w` at 1e-3. Measured error at the
    /// tightest case is 3.3e-3 for the literal form against 0 ulp here.
    #[test]
    fn parallel_transport_preserves_norm_near_antipode() {
        let s = s3();
        let lifted_norm = |p: Vec3, v: Vec3| {
            let vw = -v.dot(p) / to_sphere(p).w;
            Vec4::new(v.x, v.y, v.z, vw).length()
        };
        for w in [5e-3_f32, 2e-3, 1.2e-3] {
            let b = w;
            let a = (1.0 - w * w - b * b).sqrt();
            let from = Vec3::new(a, b, 0.0);
            let to = Vec3::new(-a, b, 0.0);
            // Along the plane of motion, so the transport actually rotates it.
            let v = Vec3::X;
            let vt = s.parallel_transport(from, to, v);
            let norm_from = lifted_norm(from, v);
            assert_relative_eq!(lifted_norm(to, vt), norm_from, max_relative = 1e-5);
            assert!(
                (vt - v).length() > 0.5 * norm_from,
                "transport should rotate an in-plane vector, got {vt:?}"
            );
        }
    }

    #[test]
    fn small_scale_distance_matches_euclidean() {
        // At the origin the metric factor is 1: ds_S³ = ds_R³.
        let s = s3();
        let eps = 1e-3;
        let p = Vec3::new(eps, 0.0, 0.0);
        assert_relative_eq!(s.distance(Vec3::ZERO, p), eps, epsilon = 1e-6);
    }

    #[test]
    fn angle_excess_in_small_triangle_scales_with_area() {
        // Gauss-Bonnet at K = +1: (α + β + γ) − π = area. A small equilateral
        // triangle of side L has area ≈ (√3/4)·L².
        let s = s3();
        let l = 0.05_f32;
        let a = Vec3::ZERO;
        let b = s.exp(a, Vec3::new(l, 0.0, 0.0));
        let c = s.exp(a, Vec3::new(l * 0.5, l * 3.0_f32.sqrt() * 0.5, 0.0));

        let angle_at = |p: Vec3, q: Vec3, r: Vec3| -> f32 {
            let u3 = s.log(p, q);
            let w3 = s.log(p, r);
            // The 3D metric is not Euclidean away from the origin, so take the
            // Riemannian angle from the lifted 4D tangents (vw = -dot(v3, p)/q.w).
            let qp = to_sphere(p);
            let u4 = Vec4::new(u3.x, u3.y, u3.z, -u3.dot(p) / qp.w);
            let w4 = Vec4::new(w3.x, w3.y, w3.z, -w3.dot(p) / qp.w);
            (u4.dot(w4) / (u4.length() * w4.length()))
                .clamp(-1.0, 1.0)
                .acos()
        };

        let alpha = angle_at(a, b, c);
        let beta = angle_at(b, a, c);
        let gamma = angle_at(c, a, b);
        let excess = (alpha + beta + gamma) - std::f32::consts::PI;
        let expected_area = 3.0_f32.sqrt() / 4.0 * l * l;

        assert!(
            excess > 0.0,
            "spherical triangle should have positive angle excess, got {excess}"
        );
        assert_relative_eq!(excess, expected_area, epsilon = 5e-4);
    }

    #[test]
    fn out_of_domain_does_not_panic() {
        let s = s3();
        let inside = Vec3::new(0.5, 0.0, 0.0);
        let on_boundary = Vec3::new(1.0, 0.0, 0.0);
        let outside = Vec3::new(2.0, 0.0, 0.0);
        let d1 = s.distance(inside, on_boundary);
        let d2 = s.distance(inside, outside);
        assert!(d1.is_finite() && d1 >= 0.0);
        assert!(d2.is_finite() && d2 >= 0.0);
    }

    /// Every shipped WGSL function with a CPU twin, pinned as one contiguous
    /// statement sequence covering the whole body, signature through closing
    /// brace. Comments are normalized out of the shipped source before the
    /// comparison, so prose edits do not read as drift while a statement
    /// added between two comments does.
    ///
    /// The pin fails when the shader form moves; the mirror parity tests below
    /// fail when the CPU form moves. Neither half of a twin can be edited
    /// alone.
    const WGSL_BODY_PINS: &[(&str, &str)] = &[
        (
            "loam_s3_clamp",
            r#"fn loam_s3_clamp(p: vec3<f32>) -> vec3<f32> {
    let r2 = dot(p, p);
    if (r2 <= LOAM_S3_R2_MAX) { return p; }
    return p * (sqrt(LOAM_S3_R2_MAX) / sqrt(r2));
}"#,
        ),
        (
            "loam_s3_lift",
            r#"fn loam_s3_lift(p: vec3<f32>) -> vec4<f32> {
    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return vec4<f32>(p.x, p.y, p.z, sqrt(1.0 - r2));
}"#,
        ),
        (
            "loam_origin_distance",
            r#"fn loam_origin_distance(p: vec3<f32>) -> f32 {
    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return asin(sqrt(r2));
}"#,
        ),
        (
            "loam_distance",
            r#"fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let qa = loam_s3_lift(loam_s3_clamp(a));
    let qb = loam_s3_lift(loam_s3_clamp(b));
    let half_chord = length(qa - qb) * 0.5;
    return 2.0 * asin(clamp(half_chord, 0.0, 1.0));
}"#,
        ),
        (
            "loam_exp",
            r#"fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let p = loam_s3_clamp(at);
    let n2 = dot(v, v);
    if (n2 < 1e-14) { return p; }
    let q = loam_s3_lift(p);
    let vw = -dot(v, p) / q.w;
    let v4 = vec4<f32>(v.x, v.y, v.z, vw);
    let mag = length(v4);
    let result4 = normalize(q * cos(mag) + v4 * (sin(mag) / mag));
    return loam_s3_clamp(result4.xyz);
}"#,
        ),
        (
            "loam_log",
            r#"fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> {
    let qf = loam_s3_lift(loam_s3_clamp(p_from));
    let qt = loam_s3_lift(loam_s3_clamp(p_to));
    let d_dot = clamp(dot(qf, qt), -1.0, 1.0);
    let perp4 = qt - d_dot * qf;
    let n = length(perp4);
    if (n < 1e-7) { return vec3<f32>(0.0, 0.0, 0.0); }
    let half_chord = length(qt - qf) * 0.5;
    let d = 2.0 * asin(clamp(half_chord, 0.0, 1.0));
    return perp4.xyz * (d / n);
}"#,
        ),
        (
            "loam_parallel_transport",
            r#"fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let pf = loam_s3_clamp(p_from);
    let pt = loam_s3_clamp(p_to);
    let qf = loam_s3_lift(pf);
    let qt = loam_s3_lift(pt);
    let vw = -dot(v, pf) / qf.w;
    let v4 = vec4<f32>(v.x, v.y, v.z, vw);
    let sum = qf + qt;
    let denom = dot(sum, sum) * 0.5;
    let v4t = v4 - (dot(v4, qt) / denom) * sum;
    return v4t.xyz;
}"#,
        ),
    ];

    /// `fn name` in `src`, signature through the closing brace in column 0,
    /// with comments stripped and blank lines dropped. WGSL has no string
    /// literals, so cutting each line at its first `//` cannot eat code.
    /// A missing or unterminated function panics: a pin that cannot find its
    /// target is drift, not a pass.
    fn wgsl_function_source(src: &str, name: &str) -> String {
        let start = src
            .find(&format!("\nfn {name}("))
            .unwrap_or_else(|| panic!("{name} is not in the shipped WGSL"))
            + 1;
        let end = src[start..]
            .find("\n}\n")
            .unwrap_or_else(|| panic!("{name} has no closing brace in column 0"));
        src[start..start + end + 2]
            .lines()
            .map(|line| line.split("//").next().unwrap().trim_end())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    // CPU ports of the shipped WGSL, expression for expression, so parity is
    // checkable without an adapter. They call each other rather than the
    // shipped helpers: a mirror that delegated to `to_sphere` would leave that
    // twin free to drift.
    fn wgsl_clamp_mirror(p: Vec3) -> Vec3 {
        let r2 = p.dot(p);
        if r2 <= SPHERE_R2_MAX {
            return p;
        }
        p * (SPHERE_R2_MAX.sqrt() / r2.sqrt())
    }

    fn wgsl_lift_mirror(p: Vec3) -> Vec4 {
        let r2 = p.dot(p).min(SPHERE_R2_MAX);
        Vec4::new(p.x, p.y, p.z, (1.0 - r2).sqrt())
    }

    fn wgsl_origin_distance_mirror(p: Vec3) -> f32 {
        let r2 = p.length_squared().min(SPHERE_R2_MAX);
        r2.sqrt().asin()
    }

    fn wgsl_distance_mirror(a: Vec3, b: Vec3) -> f32 {
        let qa = wgsl_lift_mirror(wgsl_clamp_mirror(a));
        let qb = wgsl_lift_mirror(wgsl_clamp_mirror(b));
        let half_chord = (qa - qb).length() * 0.5;
        2.0 * half_chord.clamp(0.0, 1.0).asin()
    }

    fn wgsl_exp_mirror(at: Vec3, v: Vec3) -> Vec3 {
        let p = wgsl_clamp_mirror(at);
        let n2 = v.dot(v);
        if n2 < 1e-14 {
            return p;
        }
        let q = wgsl_lift_mirror(p);
        let vw = -v.dot(p) / q.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        let mag = v4.length();
        let result4 = (q * mag.cos() + v4 * (mag.sin() / mag)).normalize();
        wgsl_clamp_mirror(result4.truncate())
    }

    fn wgsl_log_mirror(from: Vec3, to: Vec3) -> Vec3 {
        let qf = wgsl_lift_mirror(wgsl_clamp_mirror(from));
        let qt = wgsl_lift_mirror(wgsl_clamp_mirror(to));
        let d_dot = qf.dot(qt).clamp(-1.0, 1.0);
        let perp4 = qt - d_dot * qf;
        let n = perp4.length();
        if n < 1e-7 {
            return Vec3::ZERO;
        }
        let half_chord = (qt - qf).length() * 0.5;
        let d = 2.0 * half_chord.clamp(0.0, 1.0).asin();
        perp4.truncate() * (d / n)
    }

    fn wgsl_parallel_transport_mirror(from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        let pf = wgsl_clamp_mirror(from);
        let pt = wgsl_clamp_mirror(to);
        let qf = wgsl_lift_mirror(pf);
        let qt = wgsl_lift_mirror(pt);
        let vw = -v.dot(pf) / qf.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        let sum = qf + qt;
        let denom = sum.dot(sum) * 0.5;
        let v4t = v4 - (v4.dot(qt) / denom) * sum;
        v4t.truncate()
    }

    /// Shared direction for the point and tangent fixtures, unit length so a
    /// scaled entry has exactly the radius it names. Sharing it makes the
    /// band tangents radial at the shell points, which is what exercises the
    /// `vw` amplification below.
    const PARITY_DIR: Vec3 = Vec3::new(0.6, -0.48, 0.64);

    /// Chart fixtures, each labelled with the guard band it occupies, so a
    /// parity assertion crosses every branch a mirror has and lands on both
    /// sides of every threshold a pair of points can reach. Paired with the
    /// origin, `|perp4|` in `log` is the chart radius itself, which is what
    /// the two smallest radii straddle.
    ///
    /// The shell antipodes are the closest a pair can drive the unfloored
    /// transport denominator to zero, which is why
    /// `transport_denominator_is_bounded_below_by_the_saturation_shell`
    /// measures its bound over these points rather than asserting it alone.
    fn parity_points() -> [Vec3; 11] {
        let shell = SPHERE_R2_MAX.sqrt();
        [
            // Pole: |perp4| = 0, the zero side of every guard.
            Vec3::ZERO,
            // Under LOG_PERP_MIN.
            Vec3::new(5e-8, 0.0, 0.0),
            // Over LOG_PERP_MIN, by half again.
            Vec3::new(1.5e-7, 0.0, 0.0),
            // Two decades over LOG_PERP_MIN, still linear in the chart.
            Vec3::new(1e-5, 0.0, 0.0),
            // Interior, no guard active.
            Vec3::new(0.2, -0.3, 0.1),
            // Interior at |p| = 0.76, no guard active.
            Vec3::new(-0.45, 0.5, 0.35),
            // Inside the saturation shell, w ≈ 0.31.
            PARITY_DIR * 0.95,
            // On the shell, the smallest `w` the chart admits, 1e-3.
            PARITY_DIR * shell,
            // Chart-antipode of the shell point: the transport denominator's
            // reachable minimum, |qf + qt|²/2 = 2e-6.
            -PARITY_DIR * shell,
            // On the equator, |p|² = 1: the SPHERE_R2_MAX clamp branch.
            Vec3::new(1.0, 0.0, 0.0),
            // Outside the chart entirely, same clamp branch.
            Vec3::new(2.0, -1.0, 0.5),
        ]
    }

    /// Tangent fixtures bracketing the early return of `exp` and a
    /// displacement long enough to leave the chart.
    ///
    /// The two `PARITY_DIR` entries bracket [`EXP_TANGENT_MIN_SQ`] from either
    /// side, so shifting it changes what at least one fixture returns. They
    /// are radial because at a shell base point the lift `vw = −dot(v, p)/w`
    /// amplifies by `1/w ≈ 1e3`, which is the only regime where `|v4|` departs
    /// from `|v|` at all, and so the only one where
    /// `exp_lifted_magnitude_is_never_below_the_tangent_guard` is measuring
    /// rather than restating.
    fn parity_vectors() -> [Vec3; 6] {
        [
            // Under EXP_TANGENT_MIN_SQ at zero length.
            Vec3::ZERO,
            // |v|² = 1e-16, two decades under EXP_TANGENT_MIN_SQ.
            Vec3::new(1e-8, 0.0, 0.0),
            // |v|² = 9.8e-15, just under EXP_TANGENT_MIN_SQ.
            PARITY_DIR * 9.9e-8,
            // |v|² = 4e-14, over EXP_TANGENT_MIN_SQ; at the pole |v4| = 2e-7,
            // the smallest lifted magnitude any surviving fixture reaches.
            PARITY_DIR * 2e-7,
            // No guard active.
            Vec3::new(0.0, 0.05, 0.0),
            // |v| = 0.79, long enough to leave the chart from a shell point.
            Vec3::new(-0.3, 0.2, 0.7),
        ]
    }

    #[test]
    fn wgsl_bodies_match_the_cpu_mirrors() {
        let src = s3().wgsl_impl();
        for (name, pin) in WGSL_BODY_PINS {
            assert_eq!(
                wgsl_function_source(&src, name),
                *pin,
                "{name} drifted from its CPU mirror"
            );
        }
    }

    #[test]
    fn wgsl_saturation_shell_matches_the_cpu_constant() {
        let pin = format!("const LOAM_S3_R2_MAX: f32 = {SPHERE_R2_MAX};");
        assert!(
            s3().wgsl_impl().contains(&pin),
            "LOAM_S3_R2_MAX drifted from SPHERE_R2_MAX; expected `{pin}`"
        );
    }

    /// Guard thresholds live in four places per twin: the CPU function, its
    /// mirror, the shipped WGSL and the body pin. The mirrors carry literals,
    /// so a CPU retune inside a band a fixture straddles fails a parity test;
    /// this is what fails for a retune inside a band nothing straddles,
    /// because the expected text is formatted from the CPU constant itself.
    #[test]
    fn wgsl_guard_thresholds_match_the_cpu_constants() {
        let src = s3().wgsl_impl();
        let pins = [
            format!("if (n2 < {EXP_TANGENT_MIN_SQ:e})"),
            format!("if (n < {LOG_PERP_MIN:e})"),
        ];
        for pin in pins {
            assert!(src.contains(&pin), "shipped WGSL has no `{pin}`");
        }
    }

    /// `LOAM_MAX_ARC` caps the marcher's Riemannian arc length and has no Rust
    /// consumer, so nothing else pins its value: `march_geodesic_cpu` takes the
    /// cap as a parameter precisely because the kernel reads it as a prelude
    /// constant. The cap has to stay under the largest origin distance this
    /// chart can report, `asin(√SPHERE_R2_MAX)`, or the marcher's boundary
    /// escape can never fire and only the arc budget terminates a ray.
    #[test]
    fn wgsl_max_arc_stays_under_the_saturated_chart_radius() {
        const S3_MAX_ARC: f32 = 1.5;
        let pin = format!("const LOAM_MAX_ARC: f32 = {S3_MAX_ARC};");
        assert!(
            s3().wgsl_impl().contains(&pin),
            "LOAM_MAX_ARC drifted; expected `{pin}`"
        );
        let chart_radius = SPHERE_R2_MAX.sqrt().asin();
        assert!(
            S3_MAX_ARC < chart_radius,
            "arc cap {S3_MAX_ARC} is above the chart radius {chart_radius}"
        );
    }

    /// The deleted transport floor's reachability argument, as a bound the
    /// code has to keep satisfying: both lifts carry `w ≥ √(1 − SPHERE_R2_MAX)`
    /// whatever the input was, because `to_sphere` takes the `min` before the
    /// square root, so `|qf + qt|²/2 ≥ 2·(1 − SPHERE_R2_MAX)` with no
    /// assumption on the xyz parts. The upper bound keeps the shell antipodes
    /// in `parity_points`, which are what make this measured rather than
    /// merely asserted. Fails first if `SPHERE_R2_MAX` moves toward 1, which is
    /// where a floor would start to earn its place again.
    #[test]
    fn transport_denominator_is_bounded_below_by_the_saturation_shell() {
        let chart_min = 2.0 * (1.0 - SPHERE_R2_MAX);
        let mut worst = f32::INFINITY;
        for from in parity_points() {
            for to in parity_points() {
                let sum = to_sphere(clamp_to_hemisphere(from)) + to_sphere(clamp_to_hemisphere(to));
                worst = worst.min(sum.length_squared() * 0.5);
            }
        }
        assert!(
            worst >= chart_min,
            "denominator reached {worst:e}, under the shell bound {chart_min:e}"
        );
        // Upper bound so the assertion above cannot pass vacuously on a
        // fixture set that stopped containing the shell antipodes. The clamp's
        // rounding puts the measured minimum near the bound, not on it.
        assert!(
            worst <= chart_min * 1.5,
            "closest approach {worst:e} is not the chart minimum {chart_min:e}"
        );
    }

    /// The deleted `exp` lifted-magnitude floor's reachability argument, as a
    /// bound: past the surviving early return the lift only appends a
    /// component, so `mag ≥ |v| ≥ √EXP_TANGENT_MIN_SQ`, and at that floor
    /// `sin(mag)/mag` is exactly 1.0 in f32. Lowering `EXP_TANGENT_MIN_SQ` to
    /// where the quotient stops being exact is the edit that would make a
    /// second guard necessary again, and it fails here.
    #[test]
    fn exp_lifted_magnitude_is_never_below_the_tangent_guard() {
        let mut smallest = f32::INFINITY;
        for at in parity_points() {
            for v in parity_vectors() {
                if v.length_squared() < EXP_TANGENT_MIN_SQ {
                    continue;
                }
                let p = clamp_to_hemisphere(at);
                let q = to_sphere(p);
                let mag = Vec4::new(v.x, v.y, v.z, -v.dot(p) / q.w).length();
                assert!(
                    mag >= v.length(),
                    "lift shrank the tangent at {at:?} {v:?}: {mag:e} < {:e}",
                    v.length()
                );
                smallest = smallest.min(mag);
            }
        }
        let guard = EXP_TANGENT_MIN_SQ.sqrt();
        assert!(
            smallest >= guard,
            "smallest lifted magnitude {smallest:e} is under the guard {guard:e}"
        );
        assert_eq!(guard.sin() / guard, 1.0);
        assert_eq!(smallest.sin() / smallest, 1.0);
    }

    /// `Iso4::from_translation` has the only guard with no WGSL twin, so no
    /// mirror discriminates it; straddle it directly. Below the threshold the
    /// isometry is exactly the identity, above it the origin lands on the
    /// target.
    #[test]
    fn translation_guard_separates_degenerate_targets_from_representable_ones() {
        let s = s3();
        let below = Vec3::new(5e-8, 0.0, 0.0);
        assert_eq!(Iso4::from_translation(below).matrix, Mat4::IDENTITY);

        let above = Vec3::new(1.5e-7, 0.0, 0.0);
        let moved = s.iso_apply(Iso4::from_translation(above), Vec3::ZERO);
        assert_relative_eq!(moved.x, above.x, max_relative = 1e-5);
        assert_eq!(moved.y, 0.0);
        assert_eq!(moved.z, 0.0);
    }

    #[test]
    fn wgsl_clamp_mirror_is_bit_identical_to_cpu_clamp() {
        for p in parity_points() {
            assert_eq!(wgsl_clamp_mirror(p), clamp_to_hemisphere(p), "at {p:?}");
        }
    }

    #[test]
    fn wgsl_lift_mirror_is_bit_identical_to_cpu_lift() {
        for p in parity_points() {
            assert_eq!(wgsl_lift_mirror(p), to_sphere(p), "at {p:?}");
        }
    }

    #[test]
    fn wgsl_distance_mirror_is_bit_identical_to_cpu_distance() {
        let s = s3();
        for a in parity_points() {
            for b in parity_points() {
                assert_eq!(wgsl_distance_mirror(a, b), s.distance(a, b), "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn wgsl_exp_mirror_is_bit_identical_to_cpu_exp() {
        let s = s3();
        for at in parity_points() {
            for v in parity_vectors() {
                assert_eq!(wgsl_exp_mirror(at, v), s.exp(at, v), "{at:?} {v:?}");
            }
        }
    }

    #[test]
    fn wgsl_log_mirror_is_bit_identical_to_cpu_log() {
        let s = s3();
        for from in parity_points() {
            for to in parity_points() {
                assert_eq!(
                    wgsl_log_mirror(from, to),
                    s.log(from, to),
                    "{from:?} {to:?}"
                );
            }
        }
    }

    #[test]
    fn wgsl_parallel_transport_mirror_is_bit_identical_to_cpu_transport() {
        let s = s3();
        for from in parity_points() {
            for to in parity_points() {
                for v in parity_vectors() {
                    assert_eq!(
                        wgsl_parallel_transport_mirror(from, to, v),
                        s.parallel_transport(from, to, v),
                        "{from:?} {to:?} {v:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn wgsl_origin_distance_matches_cpu_distance_near_origin() {
        let s = s3();
        // Radii straddling both failure regimes of acos(√(1−|p|²)) in f32:
        // exact 0 below |p| ≈ 1.73e-4, and a quantized 3.45e-4 floor above it.
        let diagonal = Vec3::new(1.0, 1.0, 1.0).normalize();
        for r in [1e-3_f32, 3e-4, 1e-4, 1e-5, 1e-6] {
            for dir in [Vec3::X, diagonal] {
                let p = dir * r;
                assert_relative_eq!(
                    wgsl_origin_distance_mirror(p),
                    s.distance(Vec3::ZERO, p),
                    max_relative = 1e-6
                );
            }
        }
    }

    #[test]
    fn wgsl_origin_distance_matches_cpu_distance_across_the_hemisphere() {
        let s = s3();
        let dir = Vec3::new(0.6, -0.48, 0.64).normalize();
        for r in [0.01_f32, 0.1, 0.4, 0.7, 0.9] {
            let p = dir * r;
            assert_relative_eq!(
                wgsl_origin_distance_mirror(p),
                s.distance(Vec3::ZERO, p),
                max_relative = 1e-5
            );
        }
    }

    #[test]
    fn wgsl_impl_is_non_empty() {
        let src = s3().wgsl_impl();
        assert!(src.contains("fn loam_distance"));
        assert!(src.contains("fn loam_exp"));
        assert!(src.contains("fn loam_log"));
        assert!(src.contains("fn loam_parallel_transport"));
    }
}
