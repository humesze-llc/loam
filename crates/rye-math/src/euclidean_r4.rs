//! Euclidean R⁴, flat 4D space with a [`Rotor4`]-based isometry.
//!
//! Parallels [`crate::euclidean::EuclideanR3`] but in one higher dimension: [`Vec4`] points,
//! [`Vec4`] tangent vectors, and an `Iso4Flat` that carries a `Rotor4` rotation + `Vec4`
//! translation.
//!
//! Intentionally distinct from [`crate::spherical::Iso4`], that type is an SO(4) matrix used to
//! embed `S³` in 4D ambient space. The flat Iso here is for rigid motions of `R⁴` itself, the
//! setting in which 4D physics simulations live.

use std::borrow::Cow;

use glam::Vec4;
use serde::{Deserialize, Serialize};

use crate::bivector::{Rotor, Rotor4};
use crate::space::{Space, WgslSpace};

/// Rigid motion of R⁴: a rotor-rotation followed by a translation.
///
/// Pure isometry, scale and shear are excluded by construction. The rotor is normalized on
/// construction from `Space::iso_compose` / `iso_inverse` only when numerical drift warrants
/// it; per-call renormalization would regress determinism on the fast path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso4Flat {
    pub rotation: Rotor4,
    pub translation: Vec4,
}

impl Iso4Flat {
    pub const IDENTITY: Self = Self {
        rotation: Rotor4::IDENTITY,
        translation: Vec4::ZERO,
    };

    pub fn from_rotation(rotation: Rotor4) -> Self {
        Self {
            rotation,
            translation: Vec4::ZERO,
        }
    }

    pub fn from_translation(translation: Vec4) -> Self {
        Self {
            rotation: Rotor4::IDENTITY,
            translation,
        }
    }
}

impl Default for Iso4Flat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Euclidean R⁴ with the standard metric `‖x‖² = x₁² + x₂² + x₃² + x₄²`.
///
/// Stateless unit struct; there is only one R⁴. `Space` methods monomorphize to the bare
/// arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR4;

impl Space for EuclideanR4 {
    type Point = Vec4;
    type Vector = Vec4;
    type Iso = Iso4Flat;

    fn distance(&self, a: Vec4, b: Vec4) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec4, v: Vec4) -> Vec4 {
        at + v
    }

    fn log(&self, from: Vec4, to: Vec4) -> Vec4 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec4, _to: Vec4, v: Vec4) -> Vec4 {
        // R⁴ is flat; parallel transport along any path is the identity.
        v
    }

    fn iso_identity(&self) -> Iso4Flat {
        Iso4Flat::IDENTITY
    }

    fn iso_compose(&self, a: Iso4Flat, b: Iso4Flat) -> Iso4Flat {
        // `(a ∘ b)(p) = a.apply(b.apply(p))`. For Rotor4 the multiplication convention is
        // "left operand applied first" (verified by
        // `rotor4_composition_matches_sequential_apply`), so the composed rotor that applies
        // `b_rot` then `a_rot` is `b.rotation · a.rotation`, opposite to `Quat`'s convention,
        // which is why this differs from `Iso3::compose`.
        Iso4Flat {
            rotation: b.rotation * a.rotation,
            translation: a.rotation.apply(b.translation) + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso4Flat) -> Iso4Flat {
        let inv_rot = a.rotation.inverse();
        Iso4Flat {
            rotation: inv_rot,
            translation: inv_rot.apply(-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso4Flat, p: Vec4) -> Vec4 {
        iso.rotation.apply(p) + iso.translation
    }

    fn iso_transport(&self, iso: Iso4Flat, _at: Vec4, v: Vec4) -> Vec4 {
        // Tangent vectors are unaffected by translation; rotation acts.
        iso.rotation.apply(v)
    }
}

impl WgslSpace for EuclideanR4 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
    /// ℝ⁴ is globally flat: chart-coord 4D SDFs (hyperplanes, hyperboxes) are mathematically
    /// correct.
    fn is_chart_flat(&self) -> bool {
        true
    }
}

// EuclideanR4's WGSL is the honest, closed-form ABI for flat ℝ⁴: `exp(p, v) = p + v`,
// `log(a, b) = b - a`, transport is identity. Naga validation lives in `rye-shader/db.rs`'s
// `euclidean_r4_space_prelude_validates_against_abi_probe`. No render node consumes it today
// (4D rendering goes through the hyperslice path, not a native 4D geodesic march), but the
// prelude is correct content rather than a stub, ready for the first consumer.
const WGSL_IMPL: &str = r#"
// rye-math :: EuclideanR4 (v0 Space WGSL ABI)
const RYE_MAX_ARC: f32 = 1e9;
fn rye_distance(a: vec4<f32>, b: vec4<f32>) -> f32 { return length(a - b); }
fn rye_origin_distance(p: vec4<f32>) -> f32 { return length(p); }
fn rye_exp(at: vec4<f32>, v: vec4<f32>) -> vec4<f32> { return at + v; }
fn rye_log(p_from: vec4<f32>, p_to: vec4<f32>) -> vec4<f32> { return p_to - p_from; }
fn rye_parallel_transport(p_from: vec4<f32>, p_to: vec4<f32>, v: vec4<f32>) -> vec4<f32> { return v; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivector::Bivector;
    use crate::bivector::Bivector4;
    use approx::assert_relative_eq;

    fn r4() -> EuclideanR4 {
        EuclideanR4
    }

    #[test]
    fn distance_is_symmetric_and_zero_on_diagonal() {
        let s = r4();
        let a = Vec4::new(1.0, 2.0, 3.0, -0.5);
        let b = Vec4::new(-4.0, 0.5, 7.0, 1.2);
        assert_relative_eq!(s.distance(a, b), s.distance(b, a));
        assert_relative_eq!(s.distance(a, a), 0.0);
    }

    #[test]
    fn exp_log_round_trip() {
        let s = r4();
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, -1.0, 2.5, 0.7);
        let recovered = s.exp(a, s.log(a, b));
        assert_relative_eq!(recovered.x, b.x);
        assert_relative_eq!(recovered.y, b.y);
        assert_relative_eq!(recovered.z, b.z);
        assert_relative_eq!(recovered.w, b.w);
    }

    #[test]
    fn iso_identity_is_neutral() {
        let s = r4();
        let p = Vec4::new(2.0, -3.0, 4.0, 0.5);
        let back = s.iso_apply(s.iso_identity(), p);
        assert_relative_eq!(back.x, p.x);
        assert_relative_eq!(back.y, p.y);
        assert_relative_eq!(back.z, p.z);
        assert_relative_eq!(back.w, p.w);
    }

    #[test]
    fn iso_compose_with_inverse_is_identity() {
        let s = r4();
        // Compound 4D rotation (xy + zw) + nonzero translation: the non-trivial Iso
        // composition path.
        let rot = Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.2).exp();
        let iso = Iso4Flat {
            rotation: rot,
            translation: Vec4::new(1.0, 2.0, 3.0, -1.0),
        };
        let inv = s.iso_inverse(iso);
        let p = Vec4::new(10.0, -5.0, 3.0, 2.5);
        let via_a = s.iso_apply(s.iso_compose(iso, inv), p);
        let via_b = s.iso_apply(s.iso_compose(inv, iso), p);
        for (got, want) in [(via_a, p), (via_b, p)] {
            assert_relative_eq!(got.x, want.x, epsilon = 1e-4);
            assert_relative_eq!(got.y, want.y, epsilon = 1e-4);
            assert_relative_eq!(got.z, want.z, epsilon = 1e-4);
            assert_relative_eq!(got.w, want.w, epsilon = 1e-4);
        }
    }

    #[test]
    fn iso_compose_matches_sequential_apply() {
        let s = r4();
        let a = Iso4Flat {
            rotation: Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.0).exp(),
            translation: Vec4::new(1.0, 0.0, 0.0, 0.0),
        };
        let b = Iso4Flat {
            rotation: Bivector4::new(0.0, 0.0, 0.0, 0.9, 0.0, 0.0).exp(),
            translation: Vec4::new(0.0, 2.0, -1.0, 0.5),
        };
        let p = Vec4::new(3.0, 4.0, 5.0, 1.0);
        let composed = s.iso_apply(s.iso_compose(a, b), p);
        let sequential = s.iso_apply(a, s.iso_apply(b, p));
        assert_relative_eq!(composed.x, sequential.x, epsilon = 1e-4);
        assert_relative_eq!(composed.y, sequential.y, epsilon = 1e-4);
        assert_relative_eq!(composed.z, sequential.z, epsilon = 1e-4);
        assert_relative_eq!(composed.w, sequential.w, epsilon = 1e-4);
    }

    #[test]
    fn iso_apply_preserves_distances() {
        let s = r4();
        let iso = Iso4Flat {
            rotation: Bivector4::new(0.3, 0.1, -0.2, 0.4, 0.0, 0.15).exp(),
            translation: Vec4::new(5.0, -3.0, 1.0, 0.0),
        };
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let d_before = s.distance(a, b);
        let a2 = s.iso_apply(iso, a);
        let b2 = s.iso_apply(iso, b);
        assert_relative_eq!(s.distance(a2, b2), d_before, epsilon = 1e-4);
    }

    #[test]
    fn iso_transport_ignores_translation() {
        let s = r4();
        let iso = Iso4Flat::from_translation(Vec4::new(100.0, -50.0, 7.0, 12.0));
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let transported = s.iso_transport(iso, Vec4::ZERO, v);
        assert_relative_eq!(transported.x, v.x);
        assert_relative_eq!(transported.y, v.y);
        assert_relative_eq!(transported.z, v.z);
        assert_relative_eq!(transported.w, v.w);
    }

    #[test]
    fn parallel_transport_preserves_length_in_flat_space() {
        let s = r4();
        let from = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let to = Vec4::new(0.0, 1.0, 0.0, 2.0);
        let v = Vec4::new(0.5, 0.0, 0.0, 0.7);
        let v_at_to = s.parallel_transport(from, to, v);
        assert_relative_eq!(v.length(), v_at_to.length());
    }

    /// Pin the v0 ABI surface: every function the shader contract requires must appear in the
    /// emitted prelude. Naga validation of the assembled source lives in `rye-shader/db.rs`;
    /// this test is a fast local check that catches a name-rename regression without spinning
    /// up the WGSL parser.
    #[test]
    fn wgsl_impl_emits_v0_abi_surface() {
        let src = r4().wgsl_impl();
        for name in [
            "rye_distance",
            "rye_origin_distance",
            "rye_exp",
            "rye_log",
            "rye_parallel_transport",
            "RYE_MAX_ARC",
        ] {
            assert!(
                src.contains(name),
                "EuclideanR4 WGSL prelude missing `{name}`",
            );
        }
    }
}
