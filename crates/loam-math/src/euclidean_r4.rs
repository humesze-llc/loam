//! Euclidean R⁴, flat 4D space with a [`Rotor4`]-based isometry.
//!
//! Parallels [`crate::euclidean::EuclideanR3`] but in one higher dimension: [`Vec4`] points,
//! [`Vec4`] tangent vectors, and an `Iso4Flat` that carries a `Rotor4` rotation + `Vec4`
//! translation.
//!
//! Intentionally distinct from [`crate::spherical::Iso4`], that type is an SO(4) matrix used
//! to embed `S³` in 4D ambient space. The flat Iso here is for rigid motions of `R⁴` itself,
//! the setting in which 4D physics simulations live.

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

    /// ℝ⁴ is globally flat: chart-coord 4D SDFs (hyperplanes, hyperboxes) are mathematically
    /// correct.
    fn is_chart_flat(&self) -> bool {
        true
    }
}

impl WgslSpace for EuclideanR4 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
}

// EuclideanR4's WGSL is the honest, closed-form ABI for flat ℝ⁴: `exp(p, v) = p + v`,
// `log(a, b) = b - a`, transport is identity. Naga validation lives in `loam-shader/db.rs`'s
// `euclidean_r4_space_prelude_validates_against_abi_probe`. No render node consumes it today
// (4D rendering goes through the hyperslice path, not a native 4D geodesic march), but the
// prelude is correct content rather than a stub, ready for the first consumer.
const WGSL_IMPL: &str = r#"
// loam-math :: EuclideanR4 (v0 Space WGSL ABI)
const LOAM_MAX_ARC: f32 = 1e9;
fn loam_distance(a: vec4<f32>, b: vec4<f32>) -> f32 { return length(a - b); }
fn loam_origin_distance(p: vec4<f32>) -> f32 { return length(p); }
fn loam_exp(at: vec4<f32>, v: vec4<f32>) -> vec4<f32> { return at + v; }
fn loam_log(p_from: vec4<f32>, p_to: vec4<f32>) -> vec4<f32> { return p_to - p_from; }
fn loam_parallel_transport(p_from: vec4<f32>, p_to: vec4<f32>, v: vec4<f32>) -> vec4<f32> { return v; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn r4() -> EuclideanR4 {
        EuclideanR4
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
    /// emitted prelude. Naga validation of the assembled source lives in `loam-shader/db.rs`;
    /// this test is a fast local check that catches a name-rename regression without spinning
    /// up the WGSL parser.
    #[test]
    fn wgsl_impl_emits_v0_abi_surface() {
        let src = r4().wgsl_impl();
        for name in [
            "loam_distance",
            "loam_origin_distance",
            "loam_exp",
            "loam_log",
            "loam_parallel_transport",
            "LOAM_MAX_ARC",
        ] {
            assert!(
                src.contains(name),
                "EuclideanR4 WGSL prelude missing `{name}`",
            );
        }
    }
}
