//! Point rasterizer pipeline. Antialiased screen-space discs composed on top of an existing
//! color attachment. Each input point is quad-expanded in the vertex shader into a billboarded
//! square; the fragment shader smoothsteps coverage from disc center to the AA edge.
//!
//! Lives next to [`crate::line_raster::LineRasterNode`] and [`crate::triangle_raster`] and is
//! constructed standalone in the same way.
//!
//! ## Pipeline shape
//!
//! - **Vertex buffer**: 4 sprite-corner indices (`0u32`, `1`, `2`, `3`). Static, shared across
//!   all points.
//! - **Index buffer**: `[0u32, 1, 2, 2, 1, 3]`. Static, two triangles per quad.
//! - **Instance buffer**: per-point `PointInstance` data (position, color, radius in pixels).
//!   `PointInstance` is module-private; consumers go through [`PointRasterNode::upload`]
//!   which converts a [`loam_shape::PointMesh`] into instance records internally.
//! - **Uniform buffer**: [`PointRasterUniforms`] (view-projection matrix + viewport size). Same
//!   shape as [`crate::line_raster::LineRasterUniforms`]; kept as a distinct type so the two
//!   pipelines can diverge later without binary churn.
//!
//! ## Depth
//!
//! Same [`crate::DepthMode`] options as the other rasterizers (`Off`, `ReadWrite`, `ReadOnly`).
//! `LessEqual` is the depth-compare convention, matching `LineRasterNode`: points draw on top
//! of co-located polygons (e.g., a vertex marker sitting on a triangle's corner) rather than
//! being z-fought into invisibility.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use loam_math::{Projection, RasterizableSpace};
use loam_shape::PointMesh;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, CompareFunction, DepthStencilState, Device, FragmentState, LoadOp,
    MultisampleState, Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StencilState, StoreOp, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexState, VertexStepMode,
};

use crate::device::RenderDevice;

/// Embedded WGSL source. Naga-validated in tests to catch ABI drift between the Rust-side
/// vertex layout and the shader's `@location` declarations.
const POINT_RASTER_WGSL: &str = include_str!("point_raster.wgsl");

/// Camera uniform handed to the point shader. Identical layout to
/// [`crate::line_raster::LineRasterUniforms`] (both pipelines need pixel-to-NDC conversion);
/// declared separately so a future divergence in either pipeline's uniform shape doesn't break
/// the other.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PointRasterUniforms {
    /// World-to-clip transform. Bit-identical to whatever the rest of the frame uses.
    pub view_projection: [[f32; 4]; 4],
    /// Render target size in pixels. Used by the vertex shader to convert pixel radii into NDC
    /// offsets.
    pub viewport_size: [f32; 2],
    /// Padding to round the struct to 16-byte alignment for `std140` uniform layout.
    pub _pad: [f32; 2],
}

impl Default for PointRasterUniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            viewport_size: [1.0, 1.0],
            _pad: [0.0; 2],
        }
    }
}

/// Per-instance point data uploaded to the GPU. One [`PointInstance`] per visible point; the
/// vertex shader expands each into a screen-space quad. Layout matches the `@location(1..=3)`
/// attribute slots in `point_raster.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct PointInstance {
    /// World-space (post-projection-to-R³) position. The disc center is rendered here.
    pos: [f32; 3],
    /// Screen-space pixel radius. AA falloff adds ~1 px beyond this; tuning the radius tunes
    /// the visible disc size in pixels.
    radius_px: f32,
    /// RGBA linear color, alpha-multiplied with the AA disc coverage.
    color: [f32; 4],
}

/// Antialiased point-disc rasterizer. Parallel to [`crate::line_raster::LineRasterNode`] and
/// [`crate::triangle_raster::TriangleRasterNode`]; construct once per `RenderDevice` and reuse
/// across frames.
pub struct PointRasterNode {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,

    /// Static corner-index buffer (always `[0u32, 1, 2, 3]`). Per-vertex input to the shader.
    corner_buf: Buffer,
    /// Static index buffer (`[0u32, 1, 2, 2, 1, 3]`, two triangles per quad).
    index_buf: Buffer,
    /// Per-instance buffer (one [`PointInstance`] per point). Grows on demand via
    /// [`Self::upload`].
    instance_buf: Buffer,
    /// Number of points currently uploaded; `0` means [`Self::execute`] is a no-op.
    instance_count: u32,
    /// Allocated capacity of `instance_buf` in instances. The buffer is re-created if a future
    /// upload exceeds this.
    instance_capacity: u32,
    /// Tracks whether the pipeline was created with a depth attachment so [`Self::execute`]
    /// can validate the caller's depth-view argument.
    has_depth: bool,
}

impl PointRasterNode {
    /// Construct the pipeline.
    ///
    /// - `surface_format` must match the color attachment at draw time.
    /// - `depth`: see [`crate::DepthMode`]. Determines whether the pipeline
    ///   reads depth, reads + writes depth, or skips it entirely. `LessEqual`
    ///   is the compare convention; same reasoning as in `LineRasterNode`
    ///   (points drawn on top of co-located polygons stay visible).
    /// - `sample_count` must match the attachment's MSAA sample count.
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("point_raster shader"),
            source: ShaderSource::Wgsl(POINT_RASTER_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("point_raster uniforms"),
            size: std::mem::size_of::<PointRasterUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("point_raster bgl"),
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
            label: Some("point_raster bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("point_raster pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // Per-vertex layout: one u32 per corner-index vertex.
        let corner_attrs = [VertexAttribute {
            format: VertexFormat::Uint32,
            offset: 0,
            shader_location: 0,
        }];
        let corner_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &corner_attrs,
        };
        // Per-instance layout: position (offset 0), radius_px (offset 12), color (offset 16).
        let instance_attrs = [
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 1,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            VertexAttribute {
                format: VertexFormat::Float32,
                offset: 12,
                shader_location: 3,
            },
        ];
        let instance_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &instance_attrs,
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("point_raster pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[corner_layout, instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: depth.format().map(|format| DepthStencilState {
                format,
                depth_write_enabled: depth.writes(),
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let corner_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_raster corner buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 3]),
            usage: BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_raster index buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 2, 1, 3]),
            usage: BufferUsages::INDEX,
        });

        // Instance buffer starts empty; grown on first upload.
        let instance_buf = device.create_buffer(&BufferDescriptor {
            label: Some("point_raster instance buffer"),
            size: 64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
            corner_buf,
            index_buf,
            instance_buf,
            instance_count: 0,
            instance_capacity: 0,
            has_depth: depth.is_active(),
        }
    }

    /// Update the camera uniform. Call once per frame before [`Self::execute`].
    pub fn set_camera(&self, queue: &Queue, view_projection: Mat4, viewport_size: Vec2) {
        let uniforms = PointRasterUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            viewport_size: viewport_size.to_array(),
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Upload a [`PointMesh`] for rendering. Projects each position from R^N to R³ via
    /// [`RasterizableSpace::project_point`]; copies the color and size attributes verbatim.
    ///
    /// Positions / colors / sizes must have the same length per the [`PointMesh`] invariant;
    /// empty meshes upload nothing and make the next [`Self::execute`] a no-op.
    pub fn upload<S, const N: usize>(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: &PointMesh<N>,
        projection: &Projection<N>,
    ) where
        S: RasterizableSpace<N>,
    {
        let n_points = mesh.positions.len();
        assert_eq!(
            mesh.colors.len(),
            n_points,
            "PointMesh invariant: colors.len() == positions.len()"
        );
        assert_eq!(
            mesh.sizes.len(),
            n_points,
            "PointMesh invariant: sizes.len() == positions.len()"
        );

        let mut instances: Vec<PointInstance> = Vec::with_capacity(n_points);
        for ((p, color), size) in mesh
            .positions
            .iter()
            .zip(mesh.colors.iter())
            .zip(mesh.sizes.iter())
        {
            let p_native = S::array_to_point(*p);
            let p3 = S::project_point(p_native, projection);
            // Same CPU-side backstop as `LineRasterNode::upload`: a central
            // projection (Schlegel, Stereographic, Perspective4D) can map a
            // vertex on the projection center / pole to a NaN or infinity, and
            // a single non-finite point poisons the GPU view-projection divide
            // into a full-screen garbage quad rather than a missing dot.
            // `is_finite` rejects every quiet-NaN bit pattern AND both
            // infinities; `continue` drops the offending point without
            // emitting it.
            if !p3.is_finite() {
                continue;
            }
            instances.push(PointInstance {
                pos: p3.to_array(),
                radius_px: *size,
                color: *color,
            });
        }

        // Grow on demand; round to next power of two to amortize re-allocations.
        if instances.len() as u32 > self.instance_capacity {
            let new_cap = (instances.len() as u32).next_power_of_two().max(16);
            self.instance_buf = device.create_buffer(&BufferDescriptor {
                label: Some("point_raster instance buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<PointInstance>() as u64),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }

        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&instances));
        }
        self.instance_count = instances.len() as u32;
    }

    /// Draw the uploaded points onto `view`. `LoadOp::Load` for both color and depth, matching
    /// the other rasterizer nodes so multiple passes share one cleared color + depth buffer
    /// within a frame.
    ///
    /// `depth_view` must be `Some` when the pipeline was constructed with a depth format and
    /// `None` otherwise. Mismatch panics with a descriptive message.
    pub fn execute(
        &self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: Option<&crate::Viewport>,
    ) -> anyhow::Result<()> {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => panic!(
                "PointRasterNode::execute: pipeline was created with a depth format but no \
                 depth view was provided"
            ),
            (false, true) => panic!(
                "PointRasterNode::execute: pipeline was created without a depth format but a \
                 depth view was provided"
            ),
            _ => {}
        }
        if self.instance_count == 0 {
            return Ok(());
        }
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("point_raster encoder"),
        });
        {
            let depth_attachment = depth_view.map(|dv| RenderPassDepthStencilAttachment {
                view: dv,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            });
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("point_raster pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(vp) = viewport {
                vp.apply(&mut rp);
            }
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.corner_buf.slice(..));
            rp.set_vertex_buffer(1, self.instance_buf.slice(..));
            rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..6, 0, 0..self.instance_count);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedded WGSL parses and validates against naga. Mirrors the line_raster validation
    /// test; catches drift between the Rust-side instance layout and the shader's `@location`
    /// declarations.
    #[test]
    fn point_raster_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(POINT_RASTER_WGSL)
            .unwrap_or_else(|e| panic!("point_raster WGSL parse failed:\n{e}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("point_raster WGSL must validate");
    }

    /// `PointRasterUniforms` is exactly 80 bytes (one mat4x4 + vec2 + 8 bytes pad), 16-byte-
    /// aligned. Drift here means the GPU reads the wrong bytes for the view-projection
    /// matrix or viewport size.
    #[test]
    fn uniforms_size_matches_wgsl() {
        assert_eq!(std::mem::size_of::<PointRasterUniforms>(), 80);
    }

    /// `PointInstance` is 32 bytes (12 position + 4 radius + 16 color). Layout offsets in the
    /// per-instance vertex layout descriptor must match this struct exactly.
    #[test]
    fn instance_size_matches_layout() {
        assert_eq!(std::mem::size_of::<PointInstance>(), 32);
        let zero = PointInstance::default();
        let base = &zero as *const _ as usize;
        let pos_off = &zero.pos as *const _ as usize - base;
        let radius_off = &zero.radius_px as *const _ as usize - base;
        let color_off = &zero.color as *const _ as usize - base;
        assert_eq!(pos_off, 0);
        assert_eq!(radius_off, 12);
        assert_eq!(color_off, 16);
    }
}
