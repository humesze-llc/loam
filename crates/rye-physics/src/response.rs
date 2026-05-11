//! Contacts and the shared friction coefficient used by the PGS solver.
//!
//! A [`Contact`] is the output of narrowphase collision detection. The PGS solver in
//! [`crate::world`] applies impulses to both bodies along the contact normal, scaled by
//! combined restitution and inverse mass.

use crate::integrator::PhysicsSpace;

/// Coulomb friction coefficient applied uniformly across spaces. 0.35 reads as "moderate grip":
/// shapes roll under gravity rather than slide indefinitely. A per-material-pair coefficient is
/// a future extension (track on `RigidBody` and combine via geometric mean).
pub const FRICTION_COEFF: f32 = 0.35;

/// Result of narrowphase collision detection between two bodies.
pub struct Contact<S: PhysicsSpace> {
    /// Unit vector from body A toward body B, in A's tangent space. Points outward from the
    /// contact; pushing along this normal separates the bodies.
    pub normal: S::Vector,

    /// The world-space point at which the contact is applied. Needed by the solver to compute
    /// angular response (torque = r x j).
    pub point: S::Point,

    /// How far the bodies overlap. Positive when they do.
    pub penetration: f32,

    /// Combined coefficient of restitution for this pair.
    pub restitution: f32,
}
