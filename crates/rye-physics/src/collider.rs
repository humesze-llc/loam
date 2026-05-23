//! Collider types. Re-exports from [`rye_shape`]; `Collider` is just an alias for the canonical
//! [`rye_shape::Shape`] enum, and `ColliderKind` aliases [`rye_shape::ShapeKind`]. This is the
//! consolidation that resolves the "`rye-scene::PrimitiveKind` and `rye-physics::Collider` are
//! parallel hierarchies that must be kept in sync by hand" duplication.
//!
//! New collider types are added by:
//! 1. Adding a variant to [`rye_shape::Shape`] and [`rye_shape::ShapeKind`].
//! 2. Registering narrowphase functions in [`crate::Narrowphase::register`] for the new
//!    [`ColliderKind`].
//!
//! The `Collider::Sphere` variant carries an extra `center: Vec3` field that's unused by physics
//! (the body's position is the center). Construct physics spheres via
//! [`rye_shape::Shape::sphere_at_origin`] or pattern-match with `Shape::Sphere { radius, .. }`
//! to ignore it.

pub use rye_shape::{Shape as Collider, ShapeKind as ColliderKind};
