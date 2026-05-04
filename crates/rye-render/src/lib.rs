//! Thin wgpu wrapper plus a tiny render-graph harness.
//!
//! - [`device`]: window surface + adapter/device acquisition. One
//!   [`device::RenderDevice`] per app.
//! - [`graph`]: linear list of [`graph::RenderNode`]s executed in
//!   order against a [`wgpu::TextureView`].
//! - [`lattice`]: pixel-space layout primitive ([`Viewport`])
//!   for restricting a render node to a sub-region of the
//!   framebuffer (e.g. carving out an egui side-panel area).
//! - [`raymarch`]: ready-made fullscreen-triangle ray-march nodes
//!   (Euclidean, geodesic, hyperslice 4D); the engine's main render
//!   path until rasterised geometry shows up.
//!
//! The crate stays deliberately small: it hands wgpu primitives to
//! callers rather than abstracting them behind a higher-level engine
//! API.

pub mod device;
pub mod graph;
pub mod lattice;
pub mod raymarch;

pub use lattice::Viewport;
pub use raymarch::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
