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
        if s < 1e-7 {
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
        if v.length_squared() < 1e-14 {
            return at;
        }
        let q = to_sphere(at);
        // Lift v to a 4D tangent perpendicular to q: vw = −dot(v,at)/q.w.
        let vw = -v.dot(at) / q.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        let mag = v4.length();
        if mag < 1e-7 {
            return at;
        }
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
        if n < 1e-7 {
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
        // v4' = v4 − (dot(v4,qt) / (1 + dot(qf,qt))) · (qf + qt); undefined at
        // antipodes (dot = −1), so clamp the denominator.
        let c = qf.dot(qt);
        let denom = (1.0 + c).max(1e-7);
        let v4_transported = v4 - v4.dot(qt) / denom * (qf + qt);
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
    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return acos(sqrt(1.0 - r2));
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
    if (mag < 1e-7) { return p; }
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
    let c = dot(qf, qt);
    let denom = max(1.0 + c, 1e-7);
    let v4t = v4 - (dot(v4, qt) / denom) * (qf + qt);
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

    #[test]
    fn wgsl_impl_is_non_empty() {
        let src = s3().wgsl_impl();
        assert!(src.contains("fn loam_distance"));
        assert!(src.contains("fn loam_exp"));
        assert!(src.contains("fn loam_log"));
        assert!(src.contains("fn loam_parallel_transport"));
    }
}
