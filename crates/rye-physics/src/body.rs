//! [`RigidBody<S>`], the physical object a [`crate::World`] simulates.

use rye_math::Bivector;

use crate::collider::Collider;
use crate::integrator::PhysicsSpace;

/// A rigid body in some [`PhysicsSpace`]. Public-fields struct so the solver and user code can
/// read and write components directly, this crate doesn't hide state, it just provides
/// the rules for advancing it.
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

    /// Coefficient of restitution for elastic bounces. 0 = perfectly
    /// inelastic, 1 = perfectly elastic.
    pub restitution: f32,
}

impl<S: PhysicsSpace> RigidBody<S> {
    /// Build a dynamic body at `position` with the given mass and collider. `space` is passed
    /// so the caller can source an identity isometry without naming the space's
    /// [`crate::Collider`] types directly.
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
        // only sensible mode; catch the misuse in debug builds before it produces silently
        // wrong physics in release.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};
    use rye_math::{EuclideanR3, EuclideanR4};

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
