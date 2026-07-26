//! Euclidean R³, the sanity-check implementation of [`Space`].
//!
//! Exists primarily so the [`Space`] contract has a concrete witness with obviously-correct
//! behavior. Hyperbolic and spherical impls can compare their round-trips against this one in
//! tests.

use std::borrow::Cow;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::space::{Space, WgslSpace};

/// A rigid motion of R³: a rotation followed by a translation.
///
/// Pure isometry; scale and shear are excluded by construction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso3 {
    pub rotation: Quat,
    pub translation: Vec3,
}

impl Iso3 {
    pub const IDENTITY: Self = Self {
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            translation: Vec3::ZERO,
        }
    }

    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation,
        }
    }
}

/// Euclidean R³ with the standard metric.
///
/// Stateless: a unit struct that monomorphizes away. Construct via `EuclideanR3` directly;
/// there is only one R³.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR3;

impl Space for EuclideanR3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = Iso3;

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        at + v
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec3, _to: Vec3, v: Vec3) -> Vec3 {
        // R³ is flat; parallel transport is the identity.
        v
    }

    fn iso_identity(&self) -> Iso3 {
        Iso3::IDENTITY
    }

    fn iso_compose(&self, a: Iso3, b: Iso3) -> Iso3 {
        // (R_a, t_a) ∘ (R_b, t_b) applied to p:
        //   R_a (R_b p + t_b) + t_a = (R_a R_b) p + (R_a t_b + t_a)
        Iso3 {
            rotation: a.rotation * b.rotation,
            translation: a.rotation * b.translation + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso3) -> Iso3 {
        let inv_rot = a.rotation.inverse();
        Iso3 {
            rotation: inv_rot,
            translation: inv_rot * (-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso3, p: Vec3) -> Vec3 {
        iso.rotation * p + iso.translation
    }

    fn iso_transport(&self, iso: Iso3, _at: Vec3, v: Vec3) -> Vec3 {
        // Translation drops out for tangent vectors; rotation acts.
        iso.rotation * v
    }

    /// ℝ³ is globally flat: chart-coord SDFs (planes, axis-aligned boxes) are mathematically
    /// correct.
    fn is_chart_flat(&self) -> bool {
        true
    }
}

impl WgslSpace for EuclideanR3 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
}

// TODO(loam-shader): distance / exp / log / parallel_transport are the v0 WGSL ABI. Remaining
// questions:
//   - Struct layout for Point/Vector/Iso across shader stages. `vec3<f32>` is 16-byte-aligned
//     in uniform buffers; packed `vec4<f32>` may be preferable for cache / alignment reasons.
//   - Function-name mangling when multiple Space impls coexist in one shader module (e.g. a
//     portal between R3 and H3 in the fractal ray-marcher).
//   - How `iso_apply` / `iso_transport` are exposed to user WGSL, free functions, struct
//     methods, or a uniform-buffer-bound operator pattern.
//
// `from` is a WGSL reserved keyword (WGSL spec §3.2); use `p_from`/`p_to`.
const WGSL_IMPL: &str = r#"
// loam-math :: EuclideanR3 (v0 Space WGSL ABI)
const LOAM_MAX_ARC: f32 = 1e9;
fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 { return length(a - b); }
fn loam_origin_distance(p: vec3<f32>) -> f32 { return length(p); }
fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> { return at + v; }
fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> { return p_to - p_from; }
fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> { return v; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tangent::Tangent;
    use approx::assert_relative_eq;

    fn r3() -> EuclideanR3 {
        EuclideanR3
    }

    #[test]
    fn parallel_transport_preserves_distance_in_flat_space() {
        let s = r3();
        let from = Vec3::new(1.0, 0.0, 0.0);
        let to = Vec3::new(0.0, 1.0, 0.0);
        let v = Vec3::new(0.5, 0.0, 0.0);
        let v_at_to = s.parallel_transport(from, to, v);
        assert_relative_eq!(v.length(), v_at_to.length());
    }

    /// In flat space the path-aware primitive is the identity for any path. Default impl
    /// chains the segment-by-segment transport, which for E³ is the identity per segment.
    /// Pins: (1) empty / singleton paths return `v` unchanged, and (2) multi-segment paths
    /// agree with single-segment transport. Both invariants matter as the trait is consumed by
    /// camera and player controllers that construct polyline paths from per-frame motion.
    #[test]
    fn parallel_transport_along_default_impl_is_identity_in_flat_space() {
        let s = r3();
        let v = Vec3::new(1.7, -0.3, 0.5);
        // Empty path -> unchanged.
        assert_eq!(s.parallel_transport_along(&[], v), v);
        // Single point -> unchanged.
        assert_eq!(s.parallel_transport_along(&[Vec3::ZERO], v), v);
        // Multi-segment polyline -> still identity in E³.
        let path = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 3.0, -1.0),
        ];
        let transported = s.parallel_transport_along(&path, v);
        assert_relative_eq!(transported.x, v.x);
        assert_relative_eq!(transported.y, v.y);
        assert_relative_eq!(transported.z, v.z);
    }

    #[test]
    fn iso_transport_ignores_translation() {
        let s = r3();
        let iso = Iso3::from_translation(Vec3::new(100.0, -50.0, 7.0));
        let v = Vec3::new(1.0, 0.0, 0.0);
        let transported = s.iso_transport(iso, Vec3::ZERO, v);
        assert_relative_eq!(transported.x, v.x);
        assert_relative_eq!(transported.y, v.y);
        assert_relative_eq!(transported.z, v.z);
    }

    #[test]
    fn tangent_exp_matches_raw_exp() {
        let s = r3();
        let at = Vec3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(0.1, 0.2, 0.3);
        let t = Tangent::<EuclideanR3>::new(at, v);
        let via_tangent = t.exp(&s);
        let via_raw = s.exp(at, v);
        assert_relative_eq!(via_tangent.x, via_raw.x);
        assert_relative_eq!(via_tangent.y, via_raw.y);
        assert_relative_eq!(via_tangent.z, via_raw.z);
    }

    #[test]
    fn tangent_transport_and_scale() {
        let s = r3();
        let t = Tangent::<EuclideanR3>::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let moved = t.transport_to(&s, Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(moved.at, Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(moved.v, Vec3::new(1.0, 0.0, 0.0)); // flat space: v unchanged
        let scaled = t.scale(2.5);
        assert_eq!(scaled.at, Vec3::ZERO);
        assert_eq!(scaled.v, Vec3::new(2.5, 0.0, 0.0));
    }

    #[test]
    fn wgsl_impl_is_non_empty() {
        assert!(!r3().wgsl_impl().is_empty());
    }
}
