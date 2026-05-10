//! Space-agnostic collision algorithms shared across Euclidean narrowphases.
//!
//! - [`vector_ops`]: the vector-algebra trait shared between 3D and 4D (and beyond),
//!   keeping GJK and most of EPA dimension-generic.
//! - [`gjk`] / [`gjk_r4`]: Gilbert-Johnson-Keerthi containment test for convex shapes
//!   via their Minkowski difference. Used for polytope-polytope narrowphase where
//!   analytical solutions (sphere-sphere, sphere-halfspace) don't apply.
//! - [`epa`][mod@self::epa] / [`epa_r4`][mod@self::epa_r4]: Expanding Polytope
//!   Algorithm for penetration depth and contact normal from a GJK-terminating simplex.
//! - [`simplex_r4`]: closest-point-on-simplex via Gram-matrix projection, the
//!   dimension-agnostic 4D simplex helper.
//!
//! The persistent contact cache lives in `crate::manifold`, not here, because it
//! depends on `PhysicsSpace` (S::Vector, etc.) and these algorithms are deliberately
//! Space-free.

pub mod epa;
pub mod epa_r4;
pub mod gjk;
pub mod gjk_r4;
pub mod simplex_r4;
pub mod vector_ops;

pub use epa::{epa, ContactInfo};
pub use epa_r4::{epa_r4, ContactInfo4};
pub use gjk::{gjk_intersect, ConvexHull, GjkResult, MinkowskiPoint, Sphere, SupportFn};
pub use gjk_r4::{gjk_intersect_r4, ConvexHull4, GjkResult4, MinkowskiPoint4, Sphere4, SupportFn4};
pub use simplex_r4::{closest_to_origin as closest_to_origin_r4, Closest as ClosestR4};
pub use vector_ops::VectorOps;
