//! `rye` aggregator crate.
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
//! The remaining crates (`rye-app`, `rye-camera`, `rye-input`,
//! `rye-physics`, `rye-player`, `rye-scene`, `rye-shape`, `rye-text`)
//! are deliberately not re-exported here. They form the
//! application/runtime layer and are best depended on directly so
//! consumers see them in their own `Cargo.toml` rather than nested
//! under `rye::*`. Revisit if the surface stabilizes and a flat
//! `rye::*` import becomes the dominant ergonomic.
//!
//! Common types are gathered in [`prelude`] for `use rye::prelude::*;`.

pub use rye_asset as asset;
pub use rye_math as math;
pub use rye_render as render;
pub use rye_shader as shader;
pub use rye_time as time;

pub mod prelude;
