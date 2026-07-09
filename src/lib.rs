//! `loam` aggregator crate.
//!
//! ## Facade scope
//!
//! Re-exports the foundational + rendering crates that an external
//! consumer is most likely to want by short name:
//!
//! - [`asset`] (filesystem watcher)
//! - [`math`] (Space trait + closed-form Spaces + bivectors)
//! - [`render`] (wgpu wrapper + ray-march nodes)
//! - [`shader`] (WGSL hot reload + Space-prelude injection)
//! - [`time`] (fixed-timestep accumulator)
//!
//! The remaining crates (`loam-app`, `loam-camera`, `loam-input`,
//! `loam-physics`, `loam-player`, `loam-scene`, `loam-shape`, `loam-text`)
//! are deliberately not re-exported here. They form the
//! application/runtime layer and are best depended on directly so
//! consumers see them in their own `Cargo.toml` rather than nested
//! under `loam::*`. Revisit if the surface stabilizes and a flat
//! `loam::*` import becomes the dominant ergonomic.
//!
//! Common types are gathered in [`prelude`] for `use loam::prelude::*;`.

pub use loam_asset as asset;
pub use loam_math as math;
pub use loam_render as render;
pub use loam_shader as shader;
pub use loam_time as time;

pub mod prelude;
