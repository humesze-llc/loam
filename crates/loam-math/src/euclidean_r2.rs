//! Euclidean R², the 2D flat-space [`Space`] impl.
//!
//! Parallel to [`EuclideanR3`](crate::euclidean::EuclideanR3), but with `Point = Vec2` and
//! orientation represented by [`Rotor2`] (a unit complex number) rather than a quaternion.
//! Used by 2D physics demos and any drop-in 2D simulation.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::bivector::{Rotor, Rotor2};
use crate::space::Space;

/// A rigid motion of R²: a rotation followed by a translation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso2 {
    pub rotation: Rotor2,
    pub translation: Vec2,
}

impl Iso2 {
    pub const IDENTITY: Self = Self {
        rotation: Rotor2::IDENTITY,
        translation: Vec2::ZERO,
    };

    pub fn from_rotation(rotation: Rotor2) -> Self {
        Self {
            rotation,
            translation: Vec2::ZERO,
        }
    }

    pub fn from_translation(translation: Vec2) -> Self {
        Self {
            rotation: Rotor2::IDENTITY,
            translation,
        }
    }
}

impl Default for Iso2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// Rotor2 needs Serialize/Deserialize for Iso2's derive. Add a thin impl that piggybacks on
// serde via a plain pair.
impl Serialize for Rotor2 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.a, self.b).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Rotor2 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (a, b) = <(f32, f32)>::deserialize(d)?;
        Ok(Self { a, b })
    }
}

/// Euclidean R² with the standard metric. Stateless unit struct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR2;

impl Space for EuclideanR2 {
    type Point = Vec2;
    type Vector = Vec2;
    type Iso = Iso2;

    fn distance(&self, a: Vec2, b: Vec2) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec2, v: Vec2) -> Vec2 {
        at + v
    }

    fn log(&self, from: Vec2, to: Vec2) -> Vec2 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec2, _to: Vec2, v: Vec2) -> Vec2 {
        v
    }

    fn iso_identity(&self) -> Iso2 {
        Iso2::IDENTITY
    }

    fn iso_compose(&self, a: Iso2, b: Iso2) -> Iso2 {
        // (R_a, t_a) ∘ (R_b, t_b) applied to p:
        //   R_a (R_b p + t_b) + t_a = (R_a R_b) p + (R_a t_b + t_a)
        Iso2 {
            rotation: a.rotation * b.rotation,
            translation: a.rotation.apply(b.translation) + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso2) -> Iso2 {
        let inv = a.rotation.inverse();
        Iso2 {
            rotation: inv,
            translation: inv.apply(-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso2, p: Vec2) -> Vec2 {
        iso.rotation.apply(p) + iso.translation
    }

    fn iso_transport(&self, iso: Iso2, _at: Vec2, v: Vec2) -> Vec2 {
        // Translation drops out for tangent vectors; rotation acts.
        iso.rotation.apply(v)
    }

    /// ℝ² is globally flat: chart-coord SDFs (half-planes, axis-aligned boxes) are
    /// mathematically correct.
    fn is_chart_flat(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivector::{Bivector, Bivector2};
    use std::f32::consts::FRAC_PI_2;

    fn assert_vec2_close(a: Vec2, b: Vec2) {
        assert!((a - b).length() <= 1e-5, "expected {a:?} close to {b:?}");
    }

    #[test]
    fn iso_translation_moves_origin_to_target() {
        let s = EuclideanR2;
        let t = Vec2::new(2.0, -1.0);
        let iso = Iso2::from_translation(t);
        assert_vec2_close(s.iso_apply(iso, Vec2::ZERO), t);
    }

    #[test]
    fn iso_rotation_quarter_turn_sends_x_to_y() {
        let s = EuclideanR2;
        let r = Iso2::from_rotation(Bivector2(FRAC_PI_2).exp());
        assert_vec2_close(s.iso_apply(r, Vec2::X), Vec2::Y);
    }

    #[test]
    fn iso_transport_is_rotation_only() {
        let s = EuclideanR2;
        let iso = Iso2 {
            rotation: Bivector2(FRAC_PI_2).exp(),
            translation: Vec2::new(100.0, -50.0),
        };
        assert_vec2_close(s.iso_transport(iso, Vec2::ZERO, Vec2::X), Vec2::Y);
    }

    #[test]
    fn parallel_transport_preserves_distance_in_flat_space() {
        let s = EuclideanR2;
        let v = Vec2::new(1.0, 2.0);
        let from = Vec2::new(5.0, 5.0);
        let to = Vec2::new(-3.0, 7.0);
        assert_eq!(s.parallel_transport(from, to, v), v);
    }
}
