//! Swapchain-sized depth-attachment helper for examples that compose multiple raster
//! passes against a shared depth buffer.
//!
//! The crate doesn't take a position on which format to use; callers pass it explicitly so
//! a demo can switch between `Depth32Float` (highest precision, no stencil), `Depth24Plus`
//! (conventional), or any other depth-capable format. Sample count must match the color
//! attachment's MSAA configuration.
//!
//! Typical usage in an example's `render` function:
//!
//! ```ignore
//! DepthBuffer::ensure(
//!     &mut self.depth,
//!     &rd.device,
//!     wgpu::TextureFormat::Depth32Float,
//!     (cfg.width, cfg.height),
//!     rd.sample_count(),
//! );
//! let depth = self.depth.as_ref().expect("ensured above");
//! // clear pass then raster passes against `depth.view`
//! ```
//!
//! The framework doesn't surface a resize hook on `App`, so [`DepthBuffer::ensure`] checks
//! size + sample count each frame and recreates the texture only when they change. Holds
//! the [`wgpu::TextureView`] only; the underlying texture stays alive via wgpu's internal
//! Arc reference held by the view.

use wgpu::{
    Device, Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

/// Owns a depth texture view sized to the swapchain, recreated on resize.
pub struct DepthBuffer {
    /// Texture view bound to the `wgpu::RenderPassDepthStencilAttachment`.
    pub view: TextureView,
    /// Format the texture was created with. Stored so [`Self::ensure`] can recreate when
    /// the caller changes its mind (rare in practice).
    pub format: TextureFormat,
    /// Pixel dimensions of the underlying texture. Recreate when these change.
    size: (u32, u32),
    /// MSAA sample count. Recreate when this changes (e.g., the runtime negotiates a
    /// different MSAA level than was requested).
    sample_count: u32,
}

impl DepthBuffer {
    /// Allocate a new depth texture and return its view. Stable until the caller changes
    /// any of `format`, `size`, or `sample_count`.
    pub fn new(
        device: &Device,
        format: TextureFormat,
        size: (u32, u32),
        sample_count: u32,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("loam-render DepthBuffer"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            view,
            format,
            size,
            sample_count,
        }
    }

    /// Recreate the depth buffer in-place when its format / size / sample count don't
    /// match the requested values. No-op when everything already matches. Intended to be
    /// called once per frame at the top of the render function.
    pub fn ensure(
        slot: &mut Option<DepthBuffer>,
        device: &Device,
        format: TextureFormat,
        size: (u32, u32),
        sample_count: u32,
    ) {
        let needs_recreate = match slot {
            Some(b) => b.format != format || b.size != size || b.sample_count != sample_count,
            None => true,
        };
        if needs_recreate {
            *slot = Some(DepthBuffer::new(device, format, size, sample_count));
        }
    }
}
