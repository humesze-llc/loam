//! `rye-math`: geometric primitives for Rye.
//!
//! The central abstraction is the [`Space`] trait. Every other crate that cares about *where
//! things are* (scene graph, physics, render, shaders) routes through it, so that swapping
//! `EuclideanR3` for `HyperbolicH3` or `Spherical S3` is a type-level decision, not a fork.
//!
//! [`WgslSpace`] is a separate subtrait for the GPU half of the contract; CPU-only consumers
//! never need to implement it.
//!
//! [`Tangent`] bundles a tangent vector with its base point, the recommended holder outside
//! tight numerical kernels.
//!
//! ## Determinism
//!
//! Rye is built for lockstep multiplayer, so all math here must be bit-reproducible across
//! machines that agree on:
//! - target architecture's IEEE-754 f32 semantics (no fast-math, no FMA contraction unless
//!   globally enabled),
//! - this crate's exact version,
//! - call ordering (single-threaded sim).
//!
//! We commit to `f32` for v0. Hyperbolic distances grow as `acosh` and lose precision near the
//! horizon; if that becomes a problem we'll add a `Scalar` associated type rather than rewrite
//! call sites.

pub mod bivector;
pub mod blended;
pub mod euclidean;
pub mod euclidean_r2;
pub mod euclidean_r4;
pub mod hyperbolic;
pub mod rasterizable;
pub mod sectionable;
pub mod space;
pub mod spherical;
pub mod spherical_embedded;
pub mod tangent;

pub use bivector::{
    Bivector, Bivector2, Bivector3, Bivector4, Plane4, Rotor, Rotor2, Rotor3, Rotor4,
};
pub use blended::{BlendedSpace, BlendingField, ConformallyFlat, LinearBlendX};
pub use euclidean::{EuclideanR3, Iso3};
pub use euclidean_r2::{EuclideanR2, Iso2};
pub use euclidean_r4::{EuclideanR4, Iso4Flat};
pub use hyperbolic::{HyperbolicH3, Iso3H};
pub use rasterizable::{Projection, RasterizableSpace, STEREOGRAPHIC_POLE_EPSILON};
pub use sectionable::{
    SectionableSpace, WPlane, ZPlane, EDGE_PARALLEL_EPSILON, SLICE_PERTURBATION_EPSILON,
};
pub use space::{Space, WgslSpace};
pub use spherical::{Iso4, SphericalS3};
pub use spherical_embedded::SphericalS3Embedded;
pub use tangent::Tangent;
