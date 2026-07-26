//! [`RigidBody<S>`], the physical object a [`crate::World`] simulates.

use std::ops::{Add, Mul};

use loam_math::Bivector;

use crate::collider::Collider;
use crate::integrator::PhysicsSpace;

/// A rigid body in some [`PhysicsSpace`]. Public-fields struct so the solver and user code can
/// read and write components directly, this crate doesn't hide state, it just provides the rules
/// for advancing it.
///
/// `inv_mass == 0.0` means a static body: gravity and impulses have no effect on its velocity,
/// and [`crate::integrate_body`] skips it.
pub struct RigidBody<S: PhysicsSpace> {
    pub position: S::Point,
    pub velocity: S::Vector,
    pub orientation: S::Iso,
    pub angular_velocity: S::AngVel,

    pub mass: f32,
    pub inv_mass: f32,
    pub inertia: S::Inertia,

    pub collider: Collider,

    /// Coefficient of restitution for elastic bounces. 0 = perfectly inelastic, 1 = perfectly
    /// elastic.
    pub restitution: f32,
}

impl<S: PhysicsSpace> RigidBody<S> {
    /// Build a dynamic body at `position` with the given mass and collider. `space` is passed so
    /// the caller can source an identity isometry without naming the space's [`crate::Collider`]
    /// types directly.
    pub fn new(
        position: S::Point,
        velocity: S::Vector,
        collider: Collider,
        mass: f32,
        inertia: S::Inertia,
        space: &S,
    ) -> Self {
        // Half-spaces are infinite planes; a finite mass with infinite extent breaks the
        // integrator's assumptions (no centre of mass, no bounded inertia). Static-only is the
        // only sensible mode; catch the misuse in debug builds before it produces silently wrong
        // physics in release.
        debug_assert!(
            !matches!(
                collider,
                Collider::HalfSpace { .. } | Collider::HalfSpace4D { .. }
            ) || mass <= 0.0,
            "half-space colliders must be static (mass <= 0); got mass = {mass}"
        );
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity,
            orientation: space.iso_identity(),
            angular_velocity: <S::AngVel as Bivector>::zero(),
            mass,
            inv_mass,
            inertia,
            collider,
            restitution: 0.2,
        }
    }

    /// Static body: infinite mass, zero velocity, immovable.
    pub fn fixed(position: S::Point, collider: Collider, inertia: S::Inertia, space: &S) -> Self
    where
        S::Vector: Default,
    {
        Self {
            position,
            velocity: S::Vector::default(),
            orientation: space.iso_identity(),
            angular_velocity: <S::AngVel as Bivector>::zero(),
            mass: 0.0,
            inv_mass: 0.0,
            inertia,
            collider,
            restitution: 0.2,
        }
    }

    /// Apply an impulse whose line of action passes through the centre of
    /// mass: `v += J/m`, no angular response. Static bodies ignore it.
    ///
    /// Impulse-momentum relation, Baraff 1997, "Physically Based Modeling:
    /// Rigid Body Simulation", colliding-contact section.
    pub fn apply_impulse(&mut self, impulse: S::Vector)
    where
        S::Vector: Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
        if self.inv_mass == 0.0 {
            return;
        }
        self.velocity = self.velocity + impulse * self.inv_mass;
    }

    /// Apply an impulse at world point `point`: `v += J/m` and
    /// `ω += I⁻¹(r ∧ J)` for `r` the offset from the centre of mass to
    /// `point`. Static bodies ignore it.
    ///
    /// Same reference as [`Self::apply_impulse`]; the angular half is the
    /// wedge form of `Δω = I⁻¹(r × J)`.
    pub fn apply_impulse_at_point(&mut self, space: &S, impulse: S::Vector, point: S::Point)
    where
        S::Vector: Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
        if self.inv_mass == 0.0 {
            return;
        }
        self.velocity = self.velocity + impulse * self.inv_mass;
        // The lever arm is a tangent vector at the body, so it is `log`, not a
        // chart-coordinate subtraction; the two agree in the flat spaces that
        // implement PhysicsSpace today.
        let lever = space.log(self.position, point);
        let torque = space.wedge(lever, impulse);
        self.angular_velocity =
            self.angular_velocity + space.apply_inv_inertia(self.inertia, torque);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};
    use loam_math::{Bivector3, Bivector4, EuclideanR3, EuclideanR4};

    #[test]
    fn dynamic_halfspace_3d_panics_in_debug() {
        let result = std::panic::catch_unwind(|| {
            RigidBody::<EuclideanR3>::new(
                Vec3::ZERO,
                Vec3::ZERO,
                Collider::HalfSpace {
                    normal: Vec3::Y,
                    offset: 0.0,
                },
                1.0,
                1.0,
                &EuclideanR3,
            )
        });
        assert!(
            result.is_err(),
            "expected debug_assert to fire on dynamic 3D half-space"
        );
    }

    #[test]
    fn dynamic_halfspace_4d_panics_in_debug() {
        let result = std::panic::catch_unwind(|| {
            RigidBody::<EuclideanR4>::new(
                Vec4::ZERO,
                Vec4::ZERO,
                Collider::HalfSpace4D {
                    normal: Vec4::Y,
                    offset: 0.0,
                },
                1.0,
                1.0,
                &EuclideanR4,
            )
        });
        assert!(
            result.is_err(),
            "expected debug_assert to fire on dynamic 4D half-space"
        );
    }

    // The impulse tests pass powers of two for mass and inertia so every
    // expected value below is exact in f32 and the asserts pin the formula
    // rather than a rounding tolerance.
    fn body_r3(position: Vec3, mass: f32, inertia: f32) -> RigidBody<EuclideanR3> {
        RigidBody::new(
            position,
            Vec3::ZERO,
            Collider::sphere_at_origin(0.5),
            mass,
            inertia,
            &EuclideanR3,
        )
    }

    fn body_r4(position: Vec4, mass: f32, inertia: f32) -> RigidBody<EuclideanR4> {
        RigidBody::new(
            position,
            Vec4::ZERO,
            Collider::sphere_at_origin(0.5),
            mass,
            inertia,
            &EuclideanR4,
        )
    }

    #[test]
    fn linear_impulse_changes_velocity_by_impulse_over_mass() {
        let mut body = body_r3(Vec3::ZERO, 4.0, 0.5);
        body.velocity = Vec3::new(1.0, 0.0, 0.0);
        body.apply_impulse(Vec3::new(8.0, -4.0, 2.0));
        assert_eq!(body.velocity, Vec3::new(3.0, -1.0, 0.5));
        assert_eq!(body.angular_velocity, Bivector3::ZERO);
    }

    /// An impulse through the centre of mass is pure translation, whichever
    /// entry point applies it.
    #[test]
    fn central_impulse_produces_no_spin_and_matches_linear_form() {
        let position = Vec3::new(2.0, -1.0, 3.0);
        let impulse = Vec3::new(8.0, -4.0, 2.0);

        let mut at_point = body_r3(position, 4.0, 0.5);
        at_point.apply_impulse_at_point(&EuclideanR3, impulse, position);

        let mut linear = body_r3(position, 4.0, 0.5);
        linear.apply_impulse(impulse);

        assert_eq!(at_point.velocity, linear.velocity);
        assert_eq!(at_point.angular_velocity, Bivector3::ZERO);
    }

    /// Off-centre impulse in R³: `Δω = I⁻¹(r ∧ J)`, sign included. A lever
    /// along +y with an impulse along +x spins negatively in the xy plane.
    #[test]
    fn off_center_impulse_spins_body_by_inverse_inertia_times_lever_wedge_impulse_r3() {
        let mut body = body_r3(Vec3::ZERO, 2.0, 0.5);
        body.apply_impulse_at_point(
            &EuclideanR3,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );

        assert_eq!(body.velocity, Vec3::new(1.5, 0.0, 0.0));
        // r ∧ J = (0,2,0) ∧ (3,0,0) = -6 e_xy; I⁻¹ = 2.
        assert_eq!(body.angular_velocity, Bivector3::new(-12.0, 0.0, 0.0));
    }

    /// The R⁴ twin, with the lever along +w so the response lands in the xw
    /// plane: a rotation plane with no R³ analogue.
    #[test]
    fn off_center_impulse_spins_body_by_inverse_inertia_times_lever_wedge_impulse_r4() {
        let mut body = body_r4(Vec4::ZERO, 2.0, 0.5);
        body.apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(3.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 2.0),
        );

        assert_eq!(body.velocity, Vec4::new(1.5, 0.0, 0.0, 0.0));
        // r ∧ J = (0,0,0,2) ∧ (3,0,0,0) = -6 e_xw; I⁻¹ = 2.
        let expected = Bivector4 {
            xw: -12.0,
            ..Bivector4::ZERO
        };
        assert_eq!(body.angular_velocity, expected);
    }

    /// Equal and opposite impulses at one shared world point are an internal
    /// interaction: total linear momentum and total angular momentum about a
    /// fixed origin are both unchanged. Pins the lever arm and the inverse
    /// inertia against the linear term; the absolute sign of the wedge is
    /// pinned by the two off-centre tests above.
    #[test]
    fn equal_and_opposite_impulses_conserve_linear_and_angular_momentum() {
        let space = EuclideanR3;

        let a_velocity_before = Vec3::new(0.5, -0.25, 1.0);
        let a_spin_before = Bivector3::new(0.125, -0.5, 0.25);
        let mut a = body_r3(Vec3::new(-1.0, 0.0, 0.0), 2.0, 0.5);
        a.velocity = a_velocity_before;
        a.angular_velocity = a_spin_before;

        let b_velocity_before = Vec3::new(-0.75, 0.5, 0.125);
        let b_spin_before = Bivector3::new(-0.25, 0.0, 0.5);
        let mut b = body_r3(Vec3::new(1.0, 0.5, 0.0), 4.0, 0.25);
        b.velocity = b_velocity_before;
        b.angular_velocity = b_spin_before;

        // L about the origin: Σ (x ∧ m·v) + I·ω, with scalar isotropic I.
        let momenta = |a: &RigidBody<EuclideanR3>, b: &RigidBody<EuclideanR3>| {
            let linear = a.velocity * a.mass + b.velocity * b.mass;
            let angular = space.wedge(a.position, a.velocity * a.mass)
                + a.angular_velocity * a.inertia
                + space.wedge(b.position, b.velocity * b.mass)
                + b.angular_velocity * b.inertia;
            (linear, angular)
        };

        let (linear_before, angular_before) = momenta(&a, &b);

        let point = Vec3::new(0.0, 0.25, 0.0);
        let impulse = Vec3::new(1.5, -0.5, 0.25);
        a.apply_impulse_at_point(&space, -impulse, point);
        b.apply_impulse_at_point(&space, impulse, point);

        let (linear_after, angular_after) = momenta(&a, &b);

        // Conservation is also satisfied by doing nothing, so pin that both
        // channels of both bodies actually responded.
        assert!((a.velocity - a_velocity_before).length() > 0.1);
        assert!((b.velocity - b_velocity_before).length() > 0.1);
        assert!((a.angular_velocity + a_spin_before * -1.0).magnitude() > 0.1);
        assert!((b.angular_velocity + b_spin_before * -1.0).magnitude() > 0.1);

        let linear_drift = (linear_after - linear_before).length();
        assert!(
            linear_drift < 1e-5,
            "linear momentum drifted by {linear_drift}"
        );
        let angular_drift = (angular_after + angular_before * -1.0).magnitude();
        assert!(
            angular_drift < 1e-5,
            "angular momentum drifted by {angular_drift}"
        );
    }

    /// `inv_mass == 0` is the documented static contract: impulses are inert
    /// on both channels, including the angular one that reads `inertia`
    /// directly.
    #[test]
    fn static_body_ignores_both_impulse_forms() {
        let mut body = RigidBody::<EuclideanR3>::fixed(
            Vec3::ZERO,
            Collider::sphere_at_origin(0.5),
            1.0,
            &EuclideanR3,
        );
        body.apply_impulse(Vec3::new(5.0, 5.0, 5.0));
        body.apply_impulse_at_point(
            &EuclideanR3,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert_eq!(body.velocity, Vec3::ZERO);
        assert_eq!(body.angular_velocity, Bivector3::ZERO);
    }

    #[test]
    fn static_halfspace_4d_is_allowed() {
        // Mass = 0 means static; the guard must not fire.
        let _ = RigidBody::<EuclideanR4>::new(
            Vec4::ZERO,
            Vec4::ZERO,
            Collider::HalfSpace4D {
                normal: Vec4::Y,
                offset: 0.0,
            },
            0.0,
            1.0,
            &EuclideanR4,
        );
    }
}
