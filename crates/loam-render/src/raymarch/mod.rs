//! Ray-march render nodes: fullscreen-triangle + single UBO + user fragment shader.
//!
//! [`RayMarchNode`]: general Euclidean-fast path; takes a pre-compiled `ShaderModule`.
//! [`GeodesicRayMarchNode`]: curved-space path built for the 4-layer kernel assembly.
//!
//! The user shader must export two entry points:
//!
//! ```wgsl
//! @vertex   fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> { ... }
//! @fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> { ... }
//! ```
//!
//! See [`RayMarchUniforms`] for the layout of bind group 0 / binding 0.

mod geodesic;
mod hyperslice4d;
mod polytope_data;
pub use geodesic::GeodesicRayMarchNode;
pub use hyperslice4d::{
    BodyKind, BodyUniform, Hyperslice4DNode, Hyperslice4DUniforms, HYPERSLICE_KERNEL_WGSL,
    MAX_BODIES, SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_3SPHERE, SHAPE_600CELL,
    SHAPE_CLIFFORD_TORUS, SHAPE_DUOCYLINDER, SHAPE_PENTATOPE, SHAPE_SPHERINDER, SHAPE_TESSERACT,
};
pub use polytope_data::{polytope_extended_sdfs_wgsl, polytope_stub_sdfs_wgsl};

/// Bridge a raymarch shape ID (one of the `SHAPE_*` u32 constants re-exported above) to
/// the corresponding [`loam_shape::polytope::Polytope4`] variant for the six convex
/// regular polychora. Returns `None` for the smooth-surface SDFs (`SHAPE_3SPHERE`,
/// `SHAPE_DUOCYLINDER`, `SHAPE_CLIFFORD_TORUS`, `SHAPE_SPHERINDER`) which have no polytope
/// topology -- the cross-section algorithm and per-vertex coloring don't apply to them.
///
/// Low-level bridge for callers that already have a raw `u32`. Higher-level call sites
/// (catalogs, panel state) should prefer [`RaymarchShape`], which unifies the GPU shape
/// ID and the polytope topology on one type.
pub fn polytope4_from_shape_id(shape: u32) -> Option<loam_shape::polytope::Polytope4> {
    use loam_shape::polytope::Polytope4;
    Some(match shape {
        s if s == SHAPE_PENTATOPE => Polytope4::Pentatope,
        s if s == SHAPE_TESSERACT => Polytope4::Tesseract,
        s if s == SHAPE_16CELL => Polytope4::Cell16,
        s if s == SHAPE_24CELL => Polytope4::Cell24,
        s if s == SHAPE_120CELL => Polytope4::Cell120,
        s if s == SHAPE_600CELL => Polytope4::Cell600,
        _ => return None,
    })
}

/// All shapes the hyperslice raymarch kernel can render, unified across the polychoral
/// and smooth-surface families.
///
/// - The six convex regular polychora wrap a [`loam_shape::polytope::Polytope4`] so callers
///   can pull topology + vertex generators alongside the GPU shape ID via
///   [`RaymarchShape::polytope4`] / [`loam_shape::polytope::Polytope4::topology`].
/// - The four smooth-surface SDFs (3-sphere, duocylinder, Clifford torus, spherinder) are
///   their own variants. They have no polytope topology, so [`RaymarchShape::polytope4`]
///   returns `None` for them.
///
/// Use this in app catalogs instead of raw `u32` shape IDs so adding a new shape is
/// type-checked end-to-end. The raw `u32` for the GPU side comes from
/// [`RaymarchShape::shape_id`]; if needed, [`From<RaymarchShape> for u32`] supplies the
/// same conversion via `.into()`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RaymarchShape {
    /// One of the six convex regular 4-polytopes.
    Polytope(loam_shape::polytope::Polytope4),
    /// 4-ball boundary (a 3-sphere SDF; smooth, no topology).
    ThreeSphere,
    /// Cartesian product of two disks; smooth.
    Duocylinder,
    /// Flat 2-torus embedded in S³; smooth.
    CliffordTorus,
    /// Cylinder x line, extended to 4D; smooth.
    Spherinder,
}

impl RaymarchShape {
    /// GPU-side shape index (matches a `SHAPE_*` u32 constant). The hyperslice kernel
    /// branches on this when computing the SDF for a body.
    pub fn shape_id(&self) -> u32 {
        use loam_shape::polytope::Polytope4;
        match self {
            RaymarchShape::Polytope(p) => match p {
                Polytope4::Pentatope => SHAPE_PENTATOPE,
                Polytope4::Tesseract => SHAPE_TESSERACT,
                Polytope4::Cell16 => SHAPE_16CELL,
                Polytope4::Cell24 => SHAPE_24CELL,
                Polytope4::Cell120 => SHAPE_120CELL,
                Polytope4::Cell600 => SHAPE_600CELL,
            },
            RaymarchShape::ThreeSphere => SHAPE_3SPHERE,
            RaymarchShape::Duocylinder => SHAPE_DUOCYLINDER,
            RaymarchShape::CliffordTorus => SHAPE_CLIFFORD_TORUS,
            RaymarchShape::Spherinder => SHAPE_SPHERINDER,
        }
    }

    /// The polytope variant for polychoral shapes, or `None` for smooth-surface shapes
    /// that don't have a polytope topology.
    pub fn polytope4(&self) -> Option<loam_shape::polytope::Polytope4> {
        match self {
            RaymarchShape::Polytope(p) => Some(*p),
            _ => None,
        }
    }
}

impl From<loam_shape::polytope::Polytope4> for RaymarchShape {
    fn from(p: loam_shape::polytope::Polytope4) -> Self {
        RaymarchShape::Polytope(p)
    }
}

impl From<RaymarchShape> for u32 {
    fn from(s: RaymarchShape) -> u32 {
        s.shape_id()
    }
}

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

use crate::device::RenderDevice;
use crate::graph::RenderNode;

/// Uniform buffer for [`RayMarchNode`]. Bind group 0, binding 0.
///
/// Layout is `std140`-compatible (every `vec3` is padded to 16 bytes) so WGSL uniform access
/// matches without `@align` annotations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RayMarchUniforms {
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub camera_forward: [f32; 3],
    pub _pad1: f32,
    pub camera_right: [f32; 3],
    pub _pad2: f32,
    pub camera_up: [f32; 3],
    pub fov_y_tan: f32,
    /// Framebuffer size in pixels.
    pub resolution: [f32; 2],
    /// Seconds since app start.
    pub time: f32,
    /// Current sim tick as f32 (for shader-side animation).
    pub tick: f32,
    /// Four scalar knobs exposed to the shader; semantics are up to the user shader. Handy for
    /// live-tuning fractal parameters.
    pub params: [f32; 4],
}

impl Default for RayMarchUniforms {
    fn default() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 3.0],
            _pad0: 0.0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [1.0, 1.0],
            time: 0.0,
            tick: 0.0,
            params: [0.0; 4],
        }
    }
}

/// A render node that draws a fullscreen triangle using a user-provided fragment shader, with a
/// single UBO of [`RayMarchUniforms`].
pub struct RayMarchNode {
    pipeline: RenderPipeline,
    uniforms: RayMarchUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
    clear_color: Color,
}

impl RayMarchNode {
    /// Construct a fullscreen-triangle raymarch pipeline. `sample_count` must match the color
    /// attachment's sample count at draw time (use
    /// [`crate::device::RenderDevice::sample_count`] in app code; pass 1 in tests / headless
    /// contexts).
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        shader: &ShaderModule,
        sample_count: u32,
    ) -> Self {
        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("raymarch uniforms"),
            size: std::mem::size_of::<RayMarchUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raymarch bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("raymarch bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("raymarch pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("raymarch pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms: RayMarchUniforms::default(),
            uniform_buf,
            bind_group,
            clear_color: Color::BLACK,
        }
    }

    pub fn uniforms(&self) -> &RayMarchUniforms {
        &self.uniforms
    }

    pub fn uniforms_mut(&mut self) -> &mut RayMarchUniforms {
        &mut self.uniforms
    }

    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: RayMarchUniforms) {
        self.uniforms = uniforms;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    /// Flush current [`RayMarchUniforms`] to the GPU. Use after mutating via
    /// [`RayMarchNode::uniforms_mut`].
    ///
    /// Render loops must call this (or [`set_uniforms`](Self::set_uniforms)) before the first
    /// draw; the UBO is undefined at construction time.
    pub fn flush_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }
}

impl RayMarchNode {
    /// Execute into a sub-region of the view.
    ///
    /// `clear` selects `LoadOp::Clear` (first panel) or `LoadOp::Load` (subsequent panels).
    /// `scissor` is `[x, y, width, height]` in pixels; fragments outside this rect are
    /// discarded by the GPU.
    pub fn execute_panel(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        clear: bool,
        scissor: [u32; 4],
    ) -> Result<()> {
        self.execute_impl(rd, view, clear, Some(scissor))
    }

    fn execute_impl(
        &mut self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        clear: bool,
        scissor: Option<[u32; 4]>,
    ) -> Result<()> {
        let load = if clear {
            LoadOp::Clear(self.clear_color)
        } else {
            LoadOp::Load
        };
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("raymarch encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("raymarch pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            if let Some([x, y, w, h]) = scissor {
                rp.set_scissor_rect(x, y, w, h);
            }
            rp.draw(0..3, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

impl RenderNode for RayMarchNode {
    fn name(&self) -> &'static str {
        "raymarch"
    }

    fn execute(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.execute_impl(rd, view, true, None)
    }
}
