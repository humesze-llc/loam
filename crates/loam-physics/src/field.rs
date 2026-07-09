//! Force fields, plug-in trait for anything that produces a tangent vector at a body's
//! position.
//!
//! Register any number of force fields on a [`crate::World`]. Each integration tick,
//! [`crate::World::step`] samples every field at each body's position, accumulates the
//! forces, and applies `v += F*dt/m` before advancing bodies along geodesics.

use crate::body::RigidBody;
use crate::integrator::PhysicsSpace;

/// A field that produces a force tangent vector at a body's position. Implementations are
/// passed immutable references; forces are pure functions of body state and time.
pub trait ForceField<S: PhysicsSpace>: Send + Sync {
    fn force_at(&self, body: &RigidBody<S>, t: f32) -> S::Vector;
}

/// Constant downward (or arbitrary-direction) gravity. Force is independent of time and body
/// state, scales linearly with mass so all objects fall at the same rate regardless of mass.
pub struct Gravity<S: PhysicsSpace> {
    pub acceleration: S::Vector,
}

impl<S: PhysicsSpace> Gravity<S> {
    pub fn new(acceleration: S::Vector) -> Self {
        Self { acceleration }
    }
}

impl<S: PhysicsSpace> ForceField<S> for Gravity<S>
where
    S::Vector: Copy + std::ops::Mul<f32, Output = S::Vector>,
{
    fn force_at(&self, body: &RigidBody<S>, _t: f32) -> S::Vector {
        // F = m·a. The solver will divide by mass again to get acceleration, so all bodies fall
        // at the same rate.
        self.acceleration * body.mass
    }
}
