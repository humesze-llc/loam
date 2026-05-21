//! Thin wgpu wrapper plus a tiny render-graph harness.
//!
//! - [`device`]: window surface + adapter/device acquisition. One [`device::RenderDevice`]
//!   per app.
//! - [`graph`]: linear list of [`graph::RenderNode`]s executed in order against a
//!   [`wgpu::TextureView`].
//! - [`lattice`]: pixel-space layout primitive ([`Viewport`]) for restricting a render
//!   node to a sub-region of the framebuffer (e.g. carving out an egui side-panel area).
//! - [`raymarch`]: ready-made fullscreen-triangle ray-march nodes (Euclidean, geodesic,
//!   hyperslice 4D); the engine's main render path until rasterised geometry shows up.
//!
//! The crate stays deliberately small: it hands wgpu primitives to callers rather than
//! abstracting them behind a higher-level engine API.

pub mod depth;
pub mod device;
pub mod graph;
pub mod lattice;
pub mod line_raster;
pub mod raymarch;
pub mod triangle_raster;

pub use depth::DepthBuffer;
pub use lattice::Viewport;
pub use line_raster::{LineRasterNode, LineRasterUniforms};
pub use raymarch::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
pub use triangle_raster::{TriangleRasterNode, TriangleRasterUniforms, TriangleVertex};

/// How a rasterizer pipeline interacts with depth. Three states only -- avoids the
/// invalid-combination problem an `Option<TextureFormat> + bool depth_write` API would
/// have. Shared by [`LineRasterNode`] and [`TriangleRasterNode`].
///
/// - [`DepthMode::Off`]: no depth attachment; passes draw on top in submission order.
///   Useful for HUD-style overlays where occlusion doesn't matter.
/// - [`DepthMode::ReadWrite`]: standard scene-geometry mode. Per-fragment `depth_compare:
///   Less` + depth-write enabled; this is how lines and filled triangles behave in
///   normal 3D rendering.
/// - [`DepthMode::ReadOnly`]: depth-test against the existing buffer but don't write to
///   it. Used for alpha-blended overlays that should be occluded by scene geometry
///   in front of them without burying subsequent draws behind them in depth.
#[derive(Copy, Clone, Debug)]
pub enum DepthMode {
    Off,
    ReadWrite { format: wgpu::TextureFormat },
    ReadOnly { format: wgpu::TextureFormat },
}

impl DepthMode {
    /// Format of the depth attachment, if any. Used by the pipeline-builder helpers to
    /// configure the [`wgpu::DepthStencilState`] uniformly.
    pub fn format(&self) -> Option<wgpu::TextureFormat> {
        match self {
            DepthMode::Off => None,
            DepthMode::ReadWrite { format } | DepthMode::ReadOnly { format } => Some(*format),
        }
    }

    /// `true` for any depth-aware mode (read-only or read-write). The pipeline needs a
    /// depth attachment when this is `true`; execute() validates this against the
    /// caller's `Option<&TextureView>`.
    pub fn is_active(&self) -> bool {
        !matches!(self, DepthMode::Off)
    }

    /// Whether the pipeline should write depth. `false` for `Off` and `ReadOnly`;
    /// `true` for `ReadWrite`.
    pub fn writes(&self) -> bool {
        matches!(self, DepthMode::ReadWrite { .. })
    }
}
