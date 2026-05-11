//! [`GeodesicRayMarchNode`]: semantic wrapper for curved-space ray march.

use anyhow::Result;
use wgpu::*;

use crate::device::RenderDevice;
use crate::graph::RenderNode;
use crate::raymarch::{RayMarchNode, RayMarchUniforms};

/// A render node that draws a fullscreen triangle using a geodesic march shader assembled from
/// four layers: `[Space prelude] + [scene SDF] + [march kernel] + [user shading]`.
///
/// Build the compiled `ShaderModule` with `rye_shader::ShaderDb::load_geodesic_scene`, then
/// pass it to [`GeodesicRayMarchNode::from_module`].
pub struct GeodesicRayMarchNode(RayMarchNode);

impl GeodesicRayMarchNode {
    /// Construct from a pre-compiled geodesic shader module. `sample_count` must match the
    /// render target's sample count; use [`crate::device::RenderDevice::sample_count`] in app
    /// code.
    pub fn from_module(
        device: &Device,
        surface_format: TextureFormat,
        module: &ShaderModule,
        sample_count: u32,
    ) -> Self {
        Self(RayMarchNode::new(
            device,
            surface_format,
            module,
            sample_count,
        ))
    }

    pub fn uniforms(&self) -> &RayMarchUniforms {
        self.0.uniforms()
    }

    pub fn uniforms_mut(&mut self) -> &mut RayMarchUniforms {
        self.0.uniforms_mut()
    }

    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: RayMarchUniforms) {
        self.0.set_uniforms(queue, uniforms);
    }

    pub fn flush_uniforms(&self, queue: &Queue) {
        self.0.flush_uniforms(queue);
    }

    pub fn execute_panel(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        clear: bool,
        scissor: [u32; 4],
    ) -> Result<()> {
        self.0.execute_panel(rd, view, clear, scissor)
    }
}

impl RenderNode for GeodesicRayMarchNode {
    fn name(&self) -> &'static str {
        "geodesic_raymarch"
    }

    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        RenderNode::execute(&mut self.0, rd, view)
    }
}
