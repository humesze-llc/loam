//! The [`PhysicsSpace`] trait and the generic integration function.
//!
//! `PhysicsSpace` extends [`loam_math::Space`] with the rotation dynamics physics
//! needs: an angular-velocity type, an inertia type, and an orientation-
//! integration rule. Everything else is written against `Space` and works
//! unchanged across E², E³, H³, S³, etc.

use std::ops::Mul;

use loam_math::{Bivector, Space};

use crate::body::RigidBody;

/// A [`Space`] equipped with rotation dynamics: angular velocity, inertia, and
/// orientation integration.
///
/// New spaces opt into physics by implementing this. Sphere-sphere collision
/// works immediately via [`loam_math::Space::distance`] and
/// [`loam_math::Space::log`]; polygon/polyhedron collision needs per-space
/// narrowphase functions registered in [`crate::Narrowphase`].
pub trait PhysicsSpace: Space {
    /// Angular-velocity bivector (e.g. [`loam_math::Bivector2`],
    /// [`loam_math::Bivector3`]).
    type AngVel: Bivector;

    /// Inertia: scalar in 2D, 3×3 in 3D, 6×6 bivector map in 4D. Layout opaque.
    type Inertia: Copy;

    /// Integrate orientation by angular velocity over a timestep. Returns the new orientation.
    fn integrate_orientation(&self, iso: Self::Iso, omega: Self::AngVel, dt: f32) -> Self::Iso;

    /// Apply the inverse inertia to a torque-bivector. Used by the solver for `ω += I⁻¹τ dt`.
    fn apply_inv_inertia(&self, inertia: Self::Inertia, torque: Self::AngVel) -> Self::AngVel;

    /// Wedge product `a ∧ b` of two tangent vectors, as an angular-velocity
    /// bivector. The angular half of an impulse response is built from it:
    /// an impulse `J` at body offset `r` gives `ω += I⁻¹(r ∧ J)`.
    fn wedge(&self, a: Self::Vector, b: Self::Vector) -> Self::AngVel;

    /// World-space velocity of `body` at world point `p`: linear plus the
    /// angular contribution `ω × (p − body.position)`.
    fn velocity_at_point(&self, body: &RigidBody<Self>, p: Self::Point) -> Self::Vector
    where
        Self: Sized;

    /// Inverse effective mass for a unit-direction impulse at `contact_point`
    /// between `a` and `b`. The PGS solver divides by this to turn a velocity
    /// constraint into an impulse magnitude:
    ///
    /// ```text
    /// K = inv_m_a + inv_m_b
    ///      + ((r_a ∧ n) · I_a⁻¹ · (r_a ∧ n))
    ///      + ((r_b ∧ n) · I_b⁻¹ · (r_b ∧ n))
    /// ```
    ///
    /// Returns 0 only when both bodies are static.
    fn effective_mass_inv(
        &self,
        a: &RigidBody<Self>,
        b: &RigidBody<Self>,
        contact_point: Self::Point,
        direction: Self::Vector,
    ) -> f32
    where
        Self: Sized;

    /// Apply a linear+angular impulse of magnitude `magnitude` along `direction` at world point
    /// `contact_point`. Sign convention: subtracts from A, adds to B (matches `Contact::normal`
    /// pointing from A toward B as the *separating* direction).
    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<Self>,
        b: &mut RigidBody<Self>,
        contact_point: Self::Point,
        direction: Self::Vector,
        magnitude: f32,
    ) where
        Self: Sized;
}

/// Space-generic integration step: advance position along the geodesic,
/// parallel-transport velocity to the new tangent space, integrate orientation.
/// Calls only [`loam_math::Space::exp`], [`loam_math::Space::parallel_transport`],
/// and [`PhysicsSpace::integrate_orientation`].
pub fn integrate_body<S>(space: &S, body: &mut RigidBody<S>, dt: f32)
where
    S: PhysicsSpace,
    S::Vector: Mul<f32, Output = S::Vector>,
{
    if body.inv_mass == 0.0 {
        return; // static
    }

    let p_old = body.position;
    let v_dt = body.velocity * dt;
    let p_new = space.exp(p_old, v_dt);
    body.velocity = space.parallel_transport(p_old, p_new, body.velocity);
    body.position = p_new;
    body.orientation = space.integrate_orientation(body.orientation, body.angular_velocity, dt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::Collider;
    use glam::Vec3;
    use loam_math::EuclideanR3;

    /// Static (`inv_mass == 0`) bodies never advance, even with non-zero
    /// velocity.
    #[test]
    fn static_body_skips_integration() {
        let mut body = RigidBody::<EuclideanR3>::fixed(
            Vec3::ZERO,
            Collider::sphere_at_origin(0.5),
            1.0,
            &EuclideanR3,
        );
        body.velocity = Vec3::new(10.0, 0.0, 0.0);
        integrate_body(&EuclideanR3, &mut body, 1.0);
        assert_eq!(body.position, Vec3::ZERO);
    }

    /// In flat E³, position advances by `velocity * dt` and velocity is
    /// unchanged (parallel transport is the identity).
    #[test]
    fn dynamic_body_in_e3_moves_linearly() {
        let mut body = RigidBody::<EuclideanR3>::new(
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, -3.0),
            Collider::sphere_at_origin(0.1),
            1.0,
            0.1,
            &EuclideanR3,
        );
        integrate_body(&EuclideanR3, &mut body, 0.5);
        assert_eq!(body.position, Vec3::new(0.5, 1.0, -1.5));
        assert_eq!(body.velocity, Vec3::new(1.0, 2.0, -3.0));
    }

    /// Zero `dt` is a no-op; catches `space.exp` mishandling a zero tangent.
    #[test]
    fn zero_dt_does_not_advance_state() {
        let mut body = RigidBody::<EuclideanR3>::new(
            Vec3::new(2.0, 3.0, 5.0),
            Vec3::new(7.0, 11.0, 13.0),
            Collider::sphere_at_origin(0.1),
            1.0,
            0.1,
            &EuclideanR3,
        );
        let before = (body.position, body.velocity);
        integrate_body(&EuclideanR3, &mut body, 0.0);
        assert_eq!((body.position, body.velocity), before);
    }
}
