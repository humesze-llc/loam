//! Collider types. Re-exports from [`loam_shape`]; `Collider` is just an alias for the canonical
//! [`loam_shape::Shape`] enum, and `ColliderKind` aliases [`loam_shape::ShapeKind`]. This is the
//! consolidation that resolves the "`loam-scene::PrimitiveKind` and `loam-physics::Collider` are
//! parallel hierarchies that must be kept in sync by hand" duplication.
//!
//! New collider types are added by:
//! 1. Adding a variant to [`loam_shape::Shape`] and [`loam_shape::ShapeKind`].
//! 2. Registering narrowphase functions in [`crate::Narrowphase::register`] for the new
//!    [`ColliderKind`].
//!
//! The `Collider::Sphere` variant carries an extra `center: Vec3` field that's unused by physics
//! (the body's position is the center). Construct physics spheres via
//! [`loam_shape::Shape::sphere_at_origin`] or pattern-match with `Shape::Sphere { radius, .. }`
//! to ignore it.

pub use loam_shape::{Shape as Collider, ShapeKind as ColliderKind};
