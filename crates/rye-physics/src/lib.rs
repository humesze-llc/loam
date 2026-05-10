//! `rye-physics`: Space-generic rigid-body physics for Rye.
//!
//! ## Shape of the crate
//!
//! - [`PhysicsSpace`] extends [`rye_math::Space`] with rotation dynamics
//!   (angular velocity, inertia, orientation integration).
//! - [`RigidBody<S>`] carries position, velocity, orientation, angular
//!   velocity, mass, and a [`Collider`].
//! - [`World<S>`] owns bodies, force fields, and a [`Narrowphase`]
//!   dispatch table. One `step(dt)` advances the simulation by one tick.
//! - [`Narrowphase<S>`] is a registry of `(ColliderKind, ColliderKind) -> NarrowphaseFn<S>`
//! entries. New collider types / new spaces / new
//!   collision algorithms are added by registering functions in this
//!   table; no existing code changes.
//! - [`ForceField<S>`] is a trait. Gravity is the first impl; users
//!   register their own for wind, radial fields, etc.
//!
//! ## Design commitment
//!
//! The integration loop is written against [`rye_math::Space`]
//! operations only: `exp`, `parallel_transport`, `distance`. It does not assume flat space. A
//! new `impl PhysicsSpace for ...` plugs in
//! any Space without modifying solver code.

pub mod body;
pub mod collider;
pub mod field;
pub mod integrator;
pub mod manifold;
pub mod narrowphase;
pub mod response;
pub mod world;

pub mod collision;
pub mod euclidean_r2;
pub mod euclidean_r3;
pub mod euclidean_r4;

pub use body::RigidBody;
pub use collider::{Collider, ColliderKind};
pub use field::{ForceField, Gravity};
pub use integrator::{integrate_body, PhysicsSpace};
pub use manifold::{ContactPoint, Manifold};
pub use narrowphase::{Narrowphase, NarrowphaseFn};
pub use response::{Contact, FRICTION_COEFF};
pub use world::World;
