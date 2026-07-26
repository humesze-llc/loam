//! Hyperbolic 3-space (H³), curvature `K = -1`.
//!
//! Dual representation: points in the **Poincaré ball** (`Vec3`, `|p| < 1`,
//! conformal and shader-compatible); isometries as 4×4 Lorentz matrices acting
//! on the **hyperboloid** model so composition is matmul (see [`Iso3H`]).
//! `iso_apply` round-trips Poincaré -> hyperboloid -> matmul -> Poincaré.
//!
//! Domain: Poincaré points must satisfy `|p| < 1`; the boundary sphere is the
//! point at infinity where distances diverge. Out-of-domain input is clamped
//! internally to a degraded-but-finite result rather than panicking; keeping
//! points interior is the caller's responsibility.

use std::borrow::Cow;

use glam::{Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::space::{Space, WgslSpace};

/// Max `|p|²` before the conformal factor `λ = 2/(1-|p|²)` saturates. `1 - 1e-7`
/// keeps `λ ≲ 2 × 10⁷`, well inside f32 dynamic range.
const POINCARE_R2_MAX: f32 = 1.0 - 1e-7;

/// Clamp an out-of-domain point onto the saturation shell. Never NaN or panic.
fn clamp_to_ball(p: Vec3) -> Vec3 {
    let r2 = p.length_squared();
    if r2 <= POINCARE_R2_MAX {
        p
    } else {
        #[cfg(debug_assertions)]
        tracing::warn!("HyperbolicH3: point outside Poincaré ball clamped (|p|²={r2:.4})");
        p * (POINCARE_R2_MAX.sqrt() / r2.sqrt())
    }
}

/// An orientation- and time-orientation-preserving isometry of H³: a 4×4 Lorentz
/// matrix in SO⁺(3,1) on hyperboloid coords `(x, y, z, w)` with `w` time-like.
/// Composition is matmul; inverse is `J Mᵀ J`, `J = diag(-1, -1, -1, +1)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso3H {
    pub matrix: Mat4,
}

impl Iso3H {
    pub const IDENTITY: Self = Self {
        matrix: Mat4::IDENTITY,
    };

    /// Pure spatial rotation about the ball origin: SO(3) embedded into SO⁺(3,1)
    /// as the block fixing the time axis.
    pub fn from_rotation(rotation: Quat) -> Self {
        let r = glam::Mat3::from_quat(rotation);
        let c0 = r.col(0);
        let c1 = r.col(1);
        let c2 = r.col(2);
        Self {
            matrix: Mat4::from_cols(
                Vec4::new(c0.x, c0.y, c0.z, 0.0),
                Vec4::new(c1.x, c1.y, c1.z, 0.0),
                Vec4::new(c2.x, c2.y, c2.z, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ),
        }
    }

    /// Hyperbolic translation (Lorentz boost) mapping the ball origin to `target`.
    /// Out-of-domain targets clamp to a finite rapidity rather than producing NaN.
    pub fn from_translation(target: Vec3) -> Self {
        let r2 = target.length_squared();
        if r2 < 1e-14 {
            return Self::IDENTITY;
        }
        let r = r2.sqrt();
        let dir = target / r;
        // Rapidity = hyperbolic distance origin -> radius r = 2·artanh(r).
        let rapidity = 2.0 * artanh(r.min(POINCARE_R2_MAX.sqrt()));
        let ch = rapidity.cosh();
        let sh = rapidity.sinh();
        let k = ch - 1.0;
        let (dx, dy, dz) = (dir.x, dir.y, dir.z);
        // Symmetric boost: spatial block I + k·dir⊗dir, coupling sh·dir, tt ch.
        Self {
            matrix: Mat4::from_cols(
                Vec4::new(1.0 + k * dx * dx, k * dx * dy, k * dx * dz, sh * dx),
                Vec4::new(k * dy * dx, 1.0 + k * dy * dy, k * dy * dz, sh * dy),
                Vec4::new(k * dz * dx, k * dz * dy, 1.0 + k * dz * dz, sh * dz),
                Vec4::new(sh * dx, sh * dy, sh * dz, ch),
            ),
        }
    }
}

/// Hyperbolic 3-space, Poincaré ball model, `K = -1`. Stateless unit struct; see
/// the [module docs](self) for the dual representation and domain constraint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HyperbolicH3;

impl Space for HyperbolicH3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = Iso3H;

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        // Möbius (artanh) form, not acosh: acosh(1 + δ) quantizes small distances
        // against f32's representable gap near 1.0. Möbius is well-conditioned.
        let a = clamp_to_ball(a);
        let b = clamp_to_ball(b);
        let n = mobius_add(-a, b).length();
        2.0 * artanh(n.min(POINCARE_R2_MAX.sqrt()))
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        let n = v.length();
        if n < 1e-7 {
            return at;
        }
        let at = clamp_to_ball(at);
        let lambda = 2.0 / (1.0 - at.length_squared());
        let dir = v / n;
        let scale = (lambda * n * 0.5).tanh();
        mobius_add(at, scale * dir)
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        let from = clamp_to_ball(from);
        let to = clamp_to_ball(to);
        let d = mobius_add(-from, to);
        let n = d.length();
        if n < 1e-7 {
            return Vec3::ZERO;
        }
        let lambda = 2.0 / (1.0 - from.length_squared());
        let mag = (2.0 / lambda) * artanh(n.min(POINCARE_R2_MAX.sqrt()));
        mag * d / n
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        // Geodesic PT: (λ_from / λ_to) · gyr[to, −from] v. Conformal factor
        // rescales for the metric at source vs destination.
        let from = clamp_to_ball(from);
        let to = clamp_to_ball(to);
        let conformal = (1.0 - to.length_squared()) / (1.0 - from.length_squared());
        conformal * gyr_apply(to, -from, v)
    }

    fn iso_identity(&self) -> Iso3H {
        Iso3H::IDENTITY
    }

    fn iso_compose(&self, a: Iso3H, b: Iso3H) -> Iso3H {
        Iso3H {
            matrix: a.matrix * b.matrix,
        }
    }

    fn iso_inverse(&self, a: Iso3H) -> Iso3H {
        // M⁻¹ = J·Mᵀ·J for J = diag(−1, −1, −1, +1): flips the off-diagonal
        // (spatial, time) blocks, leaves the diagonal blocks alone.
        let mt = a.matrix.transpose().to_cols_array_2d();
        let mut out = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                let sign = if (row == 3) ^ (col == 3) { -1.0 } else { 1.0 };
                out[col][row] = sign * mt[col][row];
            }
        }
        Iso3H {
            matrix: Mat4::from_cols_array_2d(&out),
        }
    }

    fn iso_apply(&self, iso: Iso3H, p: Vec3) -> Vec3 {
        let h = poincare_to_hyperboloid(p);
        let h2 = iso.matrix * h;
        hyperboloid_to_poincare(h2)
    }

    fn iso_transport(&self, iso: Iso3H, at: Vec3, v: Vec3) -> Vec3 {
        // M_*v = log(M·at, M·exp(at, v)). Exact since M is an isometry.
        let target = self.exp(at, v);
        let m_at = self.iso_apply(iso, at);
        let m_target = self.iso_apply(iso, target);
        self.log(m_at, m_target)
    }
}

impl WgslSpace for HyperbolicH3 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
}

// distance / exp / log / parallel_transport are the v0 WGSL ABI. Iso3H layout is
// absent: Lorentz matrices need a uniform-buffer binding decision before
// `iso_apply` can run in shaders.
const WGSL_IMPL: &str = r#"
// loam-math :: HyperbolicH3 (v0 Space WGSL ABI)
const LOAM_MAX_ARC: f32 = 1e9;
const LOAM_H3_R2_MAX: f32 = 0.9999999;

fn loam_artanh(x: f32) -> f32 {
    return 0.5 * log((1.0 + x) / (1.0 - x));
}

fn loam_clamp_to_ball(p: vec3<f32>) -> vec3<f32> {
    let r2 = dot(p, p);
    if (r2 <= LOAM_H3_R2_MAX) {
        return p;
    }
    return p * (sqrt(LOAM_H3_R2_MAX) / sqrt(r2));
}

fn loam_mobius_add(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = dot(a, b);
    let aa = dot(a, a);
    let bb = dot(b, b);
    let num = (1.0 + 2.0 * ab + bb) * a + (1.0 - aa) * b;
    let den = 1.0 + 2.0 * ab + aa * bb;
    if (abs(den) < 1e-12) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return num / den;
}

fn loam_gyr_apply(a: vec3<f32>, b: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let ab = loam_mobius_add(a, b);
    let bv = loam_mobius_add(b, v);
    let abv = loam_mobius_add(a, bv);
    return loam_mobius_add(-ab, abv);
}

fn loam_origin_distance(p: vec3<f32>) -> f32 {
    let r = min(length(p), sqrt(LOAM_H3_R2_MAX));
    return 2.0 * loam_artanh(r);
}

fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    // Möbius (artanh) form: stable near zero distance where the
    // equivalent acosh form quantizes. Saturates near the boundary.
    let aa = loam_clamp_to_ball(a);
    let bb = loam_clamp_to_ball(b);
    let d = loam_mobius_add(-aa, bb);
    let n = min(length(d), sqrt(LOAM_H3_R2_MAX));
    return 2.0 * loam_artanh(n);
}

fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let n = length(v);
    if (n < 1e-7) { return at; }
    let p_at = loam_clamp_to_ball(at);
    let aa = dot(p_at, p_at);
    let lambda = 2.0 / (1.0 - aa);
    let dir = v / n;
    let scale = tanh(lambda * n * 0.5);
    return loam_clamp_to_ball(loam_mobius_add(p_at, scale * dir));
}

fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> {
    let p_from_clamped = loam_clamp_to_ball(p_from);
    let p_to_clamped = loam_clamp_to_ball(p_to);
    let d = loam_mobius_add(-p_from_clamped, p_to_clamped);
    let n = length(d);
    if (n < 1e-7) { return vec3<f32>(0.0, 0.0, 0.0); }
    let aa = dot(p_from_clamped, p_from_clamped);
    let lambda = 2.0 / (1.0 - aa);
    let mag = (2.0 / lambda) * loam_artanh(min(n, sqrt(LOAM_H3_R2_MAX)));
    return mag * d / n;
}

fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let p_from_clamped = loam_clamp_to_ball(p_from);
    let p_to_clamped = loam_clamp_to_ball(p_to);
    let conformal = (1.0 - dot(p_to_clamped, p_to_clamped)) / (1.0 - dot(p_from_clamped, p_from_clamped));
    return conformal * loam_gyr_apply(p_to_clamped, -p_from_clamped, v);
}
"#;

// ---- helpers --------------------------------------------------------

/// `artanh(x)`. Caller ensures `|x| < 1`; boundary saturation is handled at the
/// call sites.
fn artanh(x: f32) -> f32 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

/// Möbius addition `a ⊕ b` in the Poincaré ball, K = -1. Non-associative; the
/// failure of associativity is the gyration. (Ungar, *From Möbius to
/// Gyrogroups*, Amer. Math. Monthly 115, 2008, §4, Def. 3.)
fn mobius_add(a: Vec3, b: Vec3) -> Vec3 {
    let ab = a.dot(b);
    let aa = a.length_squared();
    let bb = b.length_squared();
    let num = (1.0 + 2.0 * ab + bb) * a + (1.0 - aa) * b;
    let den = 1.0 + 2.0 * ab + aa * bb;
    if den.abs() < 1e-12 {
        Vec3::ZERO
    } else {
        num / den
    }
}

/// Möbius gyration `gyr[a, b] v`, the rotation from Möbius non-associativity
/// (Ungar, *From Möbius to Gyrogroups*, Amer. Math. Monthly 115, 2008, §4,
/// Def. 4).
fn gyr_apply(a: Vec3, b: Vec3, v: Vec3) -> Vec3 {
    let ab = mobius_add(a, b);
    let bv = mobius_add(b, v);
    let abv = mobius_add(a, bv);
    mobius_add(-ab, abv)
}

/// Lift a Poincaré point (`|p|² = r²`) to the hyperboloid:
/// `(2p / (1 − r²), (1 + r²) / (1 − r²))`.
fn poincare_to_hyperboloid(p: Vec3) -> Vec4 {
    let r2 = p.length_squared().min(POINCARE_R2_MAX);
    let den = 1.0 - r2;
    Vec4::new(
        2.0 * p.x / den,
        2.0 * p.y / den,
        2.0 * p.z / den,
        (1.0 + r2) / den,
    )
}

/// Project a hyperboloid point back to the Poincaré ball: `(x, y, z) / (1 + w)`.
/// The floor on `1 + w` keeps off-sheet inputs finite.
fn hyperboloid_to_poincare(h: Vec4) -> Vec3 {
    let den = (1.0 + h.w).max(1e-7);
    Vec3::new(h.x / den, h.y / den, h.z / den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tangent::Tangent;
    use approx::assert_relative_eq;

    fn h3() -> HyperbolicH3 {
        HyperbolicH3
    }

    fn lambda(p: Vec3) -> f32 {
        2.0 / (1.0 - p.length_squared())
    }

    #[test]
    fn distance_at_origin_is_twice_artanh() {
        let s = h3();
        let p = Vec3::new(0.4, 0.0, 0.0);
        assert_relative_eq!(s.distance(Vec3::ZERO, p), 2.0 * artanh(0.4), epsilon = 1e-5);
    }

    #[test]
    fn iso_translation_moves_origin_to_target() {
        let s = h3();
        let target = Vec3::new(0.3, -0.1, 0.2);
        let iso = Iso3H::from_translation(target);
        let moved = s.iso_apply(iso, Vec3::ZERO);
        assert_relative_eq!(moved.x, target.x, epsilon = 1e-5);
        assert_relative_eq!(moved.y, target.y, epsilon = 1e-5);
        assert_relative_eq!(moved.z, target.z, epsilon = 1e-5);
    }

    #[test]
    fn parallel_transport_preserves_hyperbolic_norm() {
        let s = h3();
        let from = Vec3::new(0.0, 0.0, 0.0);
        let to = Vec3::new(0.3, 0.0, 0.0);
        let v = Vec3::new(0.05, 0.05, 0.0);
        let v_at_to = s.parallel_transport(from, to, v);
        let n_from = lambda(from) * v.length();
        let n_to = lambda(to) * v_at_to.length();
        assert_relative_eq!(n_from, n_to, epsilon = 1e-4);
    }

    #[test]
    fn iso_transport_preserves_hyperbolic_norm() {
        let s = h3();
        let iso = Iso3H::from_translation(Vec3::new(0.2, 0.1, 0.0));
        let at = Vec3::new(0.05, 0.0, 0.0);
        let v = Vec3::new(0.02, 0.03, 0.0);
        let n_before = lambda(at) * v.length();
        let new_at = s.iso_apply(iso, at);
        let new_v = s.iso_transport(iso, at, v);
        let n_after = lambda(new_at) * new_v.length();
        assert_relative_eq!(n_before, n_after, epsilon = 1e-4);
    }

    #[test]
    fn small_scale_distance_matches_euclidean_via_metric_factor() {
        // At the origin ds_hyp = 2·ds_euc, so d_hyp(0, p) -> 2·|p| as p -> 0.
        let s = h3();
        let eps = 1e-3;
        let p = Vec3::new(eps, 0.0, 0.0);
        assert_relative_eq!(s.distance(Vec3::ZERO, p), 2.0 * eps, epsilon = 1e-6);
    }

    #[test]
    fn angle_defect_in_small_triangle_scales_with_area() {
        // Gauss-Bonnet, K = -1: π − (α + β + γ) = area. An equilateral hyperbolic
        // triangle of side L has area -> (√3/4) L² as L -> 0.
        let s = h3();
        let l = 0.05;
        let v_norm = l * 0.5; // exp from origin moves 2·|v|
        let a = Vec3::ZERO;
        let b = s.exp(a, Vec3::new(v_norm, 0.0, 0.0));
        let c = s.exp(
            a,
            Vec3::new(v_norm * 0.5, v_norm * (3.0_f32).sqrt() * 0.5, 0.0),
        );

        let angle_at = |p: Vec3, q: Vec3, r: Vec3| -> f32 {
            let u = s.log(p, q);
            let w = s.log(p, r);
            (u.dot(w) / (u.length() * w.length()))
                .clamp(-1.0, 1.0)
                .acos()
        };

        let alpha = angle_at(a, b, c);
        let beta = angle_at(b, a, c);
        let gamma = angle_at(c, a, b);
        let defect = std::f32::consts::PI - (alpha + beta + gamma);
        let expected_area = (3.0_f32.sqrt() / 4.0) * l * l;

        assert!(
            defect > 0.0,
            "hyperbolic triangle should have positive angle defect, got {defect}"
        );
        assert_relative_eq!(defect, expected_area, epsilon = 5e-4);
    }

    #[test]
    fn tangent_exp_matches_raw_exp() {
        let s = h3();
        let at = Vec3::new(0.1, 0.0, 0.0);
        let v = Vec3::new(0.05, 0.05, 0.0);
        let t = Tangent::<HyperbolicH3>::new(at, v);
        let via_tangent = t.exp(&s);
        let via_raw = s.exp(at, v);
        assert_relative_eq!(via_tangent.x, via_raw.x, epsilon = 1e-6);
        assert_relative_eq!(via_tangent.y, via_raw.y, epsilon = 1e-6);
        assert_relative_eq!(via_tangent.z, via_raw.z, epsilon = 1e-6);
    }

    #[test]
    fn out_of_domain_distance_does_not_panic() {
        let s = h3();
        let inside = Vec3::new(0.5, 0.0, 0.0);
        let on_boundary = Vec3::new(1.0, 0.0, 0.0);
        let outside = Vec3::new(2.0, 0.0, 0.0);
        let d1 = s.distance(inside, on_boundary);
        let d2 = s.distance(inside, outside);
        assert!(d1.is_finite() && d1 > 0.0);
        assert!(d2.is_finite() && d2 > 0.0);
    }

    #[test]
    fn wgsl_impl_is_non_empty() {
        assert!(!h3().wgsl_impl().is_empty());
        // The prelude must define the four `loam_*` ABI functions.
        let src = h3().wgsl_impl();
        assert!(src.contains("fn loam_distance"));
        assert!(src.contains("fn loam_exp"));
        assert!(src.contains("fn loam_log"));
        assert!(src.contains("fn loam_parallel_transport"));
    }

    #[test]
    fn poincare_hyperboloid_round_trip() {
        let p = Vec3::new(0.2, -0.3, 0.1);
        let h = poincare_to_hyperboloid(p);
        // On-sheet check: −x² − y² − z² + w² = 1
        let lorentz = -h.x * h.x - h.y * h.y - h.z * h.z + h.w * h.w;
        assert_relative_eq!(lorentz, 1.0, epsilon = 1e-5);
        let p2 = hyperboloid_to_poincare(h);
        assert_relative_eq!(p2.x, p.x, epsilon = 1e-6);
        assert_relative_eq!(p2.y, p.y, epsilon = 1e-6);
        assert_relative_eq!(p2.z, p.z, epsilon = 1e-6);
    }
}
