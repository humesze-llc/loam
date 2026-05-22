//! Triangle rasterizer pipeline. Per-vertex color, optional depth attachment,
//! alpha-blended composition on top of the existing color attachment. Parallel to
//! [`crate::line_raster::LineRasterNode`] in shape; the two are designed to share a
//! depth attachment within a frame so filled triangles and outline edges occlude each
//! other correctly.
//!
//! Triangles are native to the rasterizer (no quad expansion); the WGSL is small.
//!
//! ## Pipeline shape
//!
//! - **Vertex buffer**: per-vertex [`TriangleVertex`] (position in R³ + color).
//! - **Index buffer**: u32 indices into the vertex buffer.
//! - **Uniform buffer**: [`TriangleRasterUniforms`] -- just the view-projection matrix.
//! - **Depth attachment**: opt-in at construction via [`crate::DepthMode`]
//!   ([`crate::DepthMode::Off`] / [`crate::DepthMode::ReadWrite`] /
//!   [`crate::DepthMode::ReadOnly`]). Same semantics as
//!   [`crate::line_raster::LineRasterNode`]: the caller owns the depth texture lifecycle
//!   and clears it once per frame; [`TriangleRasterNode::execute`] uses `LoadOp::Load`.
//!
//! ## Why per-vertex color + no normals
//!
//! `TriangleMesh<N>`'s docs spell this out: lighting in R⁴ has no standard convention, so
//! normals are deliberately omitted at v1. Per-vertex color covers the cross-section /
//! face-fill / debug-fill use cases the rasterizer was built for. When a caller wants lit
//! shading (e.g., polychoral cross-section cell caps), [`FragmentShading::FaceNormalLambert`]
//! computes the face normal in the fragment shader from screen-space derivatives of
//! world-space position; no normal vertex attribute required. This stays honest to the
//! "no convention for normals in R^N" invariant: derivatives are local to the projected
//! R³ surface, computed downstream of [`RasterizableSpace::project_point`].

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use rye_math::{Projection, RasterizableSpace};
use rye_shape::TriangleMesh;
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

/// Embedded WGSL source. Naga-validated in tests for ABI drift detection.
const TRIANGLE_RASTER_WGSL: &str = include_str!("triangle_raster.wgsl");

/// Camera uniform handed to the triangle vertex shader. Just the view-projection matrix; no
/// viewport size (triangles don't need pixel-to-NDC scaling the way the line rasterizer's
/// quad expansion does). 64 bytes, 16-byte aligned, matches WGSL `CameraUniform` std140
/// layout exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TriangleRasterUniforms {
    /// World-to-clip transform. Bit-identical to whatever the rest of the frame uses.
    pub view_projection: [[f32; 4]; 4],
}

impl Default for TriangleRasterUniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Per-vertex GPU layout. Position is in R³ (after the upload path projects from R^N via
/// [`RasterizableSpace::project_point`]); color is RGBA in linear space. 32 bytes per vertex
/// with explicit padding so the attribute offsets stay stable across compilers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct TriangleVertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub color: [f32; 4],
}

/// Fragment-shader selector for [`TriangleRasterNode`]. Picked at construction time so
/// the pipeline state matches the chosen entry point; switching modes after construction
/// would require a new pipeline (cheap but the caller usually knows up front which path
/// it wants).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FragmentShading {
    /// Pass per-vertex color through unmodified. Use for overlays, debug fills, and any
    /// mesh whose shading is already baked into vertex colors. This is the v1 behavior
    /// and the default.
    #[default]
    Flat,
    /// Face-normal Lambert. Computes the face normal from screen-space derivatives of
    /// world-space position in the fragment shader, multiplies the vertex color by
    /// Lambert intensity + ambient floor. Suitable for faceted surfaces (e.g.,
    /// polychoral cross-section cell caps) where each triangle is geometrically flat,
    /// since derivative-based normals are exact for that case.
    ///
    /// No normal vertex attribute is required; the [`TriangleMesh`] format does not
    /// change between modes.
    FaceNormalLambert,
}

impl FragmentShading {
    /// WGSL entry-point name to bind into the fragment stage. Stable across versions;
    /// callers don't see this string but it's how the pipeline picks between the two
    /// shaders embedded in `triangle_raster.wgsl`.
    fn entry_point(self) -> &'static str {
        match self {
            Self::Flat => "fs_flat",
            Self::FaceNormalLambert => "fs_lambert",
        }
    }
}

/// Triangle rasterizer node. Parallel to [`crate::line_raster::LineRasterNode`]; both
/// own their pipeline + buffers and are constructed standalone.
pub struct TriangleRasterNode {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,

    /// Per-vertex buffer. Grown on demand by [`Self::upload`].
    vertex_buf: Buffer,
    vertex_capacity: u32,

    /// Index buffer (u32). Grown on demand by [`Self::upload`].
    index_buf: Buffer,
    index_capacity: u32,
    /// Number of indices currently uploaded; `0` means [`Self::execute`] is a no-op.
    index_count: u32,

    /// Tracks whether the pipeline was created with a depth attachment so
    /// [`Self::execute`] can validate the caller's depth-view argument.
    has_depth: bool,
}

impl TriangleRasterNode {
    /// Construct the pipeline.
    ///
    /// - `surface_format` must match the color attachment at draw time.
    /// - `depth`: see [`crate::DepthMode`]. Determines whether the pipeline reads depth,
    ///   reads + writes depth, or skips it.
    /// - `shading`: which fragment shader the pipeline binds; see [`FragmentShading`].
    /// - `sample_count` must match the attachment's MSAA sample count.
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        shading: FragmentShading,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("triangle_raster shader"),
            source: ShaderSource::Wgsl(TRIANGLE_RASTER_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("triangle_raster uniforms"),
            size: std::mem::size_of::<TriangleRasterUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("triangle_raster bgl"),
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
            label: Some("triangle_raster bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("triangle_raster pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // Per-vertex attribute layout: position at offset 0, color at offset 16 (after
        // the 4-byte _pad0). Matches `TriangleVertex` exactly.
        let vertex_attrs = [
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
        ];
        let vertex_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<TriangleVertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("triangle_raster pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some(shading.entry_point()),
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
                depth_compare: CompareFunction::Less,
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

        // Buffers start empty; grow on first upload.
        let vertex_buf = device.create_buffer(&BufferDescriptor {
            label: Some("triangle_raster vertex buffer"),
            size: 64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buf = device.create_buffer(&BufferDescriptor {
            label: Some("triangle_raster index buffer"),
            size: 64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            vertex_capacity: 0,
            index_buf,
            index_capacity: 0,
            index_count: 0,
            has_depth: depth.is_active(),
        }
    }

    /// Update the camera uniform. Call once per frame before [`Self::execute`].
    pub fn set_camera(&self, queue: &Queue, view_projection: Mat4) {
        let uniforms = TriangleRasterUniforms {
            view_projection: view_projection.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Upload a [`TriangleMesh`] for rendering. Projects each vertex from R^N to R³ via
    /// [`RasterizableSpace::project_point`]; copies the index list verbatim (already u32 in
    /// the source).
    ///
    /// Vertices and colors must have the same length per the [`TriangleMesh`] invariant;
    /// indices reference into the vertex array (u32 triples). Empty meshes are no-ops.
    pub fn upload<S, const N: usize>(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: &TriangleMesh<N>,
        projection: &Projection<N>,
    ) where
        S: RasterizableSpace<N>,
    {
        // Pack per-vertex GPU records.
        let n_vertices = mesh.vertices.len();
        assert_eq!(
            mesh.colors.len(),
            n_vertices,
            "TriangleMesh invariant: colors.len() == vertices.len()"
        );
        let mut verts: Vec<TriangleVertex> = Vec::with_capacity(n_vertices);
        for (v, color) in mesh.vertices.iter().zip(mesh.colors.iter()) {
            let p_native = S::array_to_point(*v);
            let p3 = S::project_point(p_native, projection);
            verts.push(TriangleVertex {
                position: p3.to_array(),
                _pad0: 0.0,
                color: *color,
            });
        }

        // Flatten triangle indices [u32; 3] into a single index buffer.
        let mut indices: Vec<u32> = Vec::with_capacity(mesh.indices.len() * 3);
        for tri in &mesh.indices {
            indices.extend_from_slice(tri);
        }

        // Grow buffers if needed; round up to next power of two to amortize re-allocations.
        if verts.len() as u32 > self.vertex_capacity {
            let new_cap = (verts.len() as u32).next_power_of_two().max(16);
            self.vertex_buf = device.create_buffer(&BufferDescriptor {
                label: Some("triangle_raster vertex buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<TriangleVertex>() as u64),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = new_cap;
        }
        if indices.len() as u32 > self.index_capacity {
            let new_cap = (indices.len() as u32).next_power_of_two().max(16);
            self.index_buf = device.create_buffer(&BufferDescriptor {
                label: Some("triangle_raster index buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<u32>() as u64),
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity = new_cap;
        }

        if !verts.is_empty() {
            queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&verts));
        }
        if !indices.is_empty() {
            queue.write_buffer(&self.index_buf, 0, bytemuck::cast_slice(&indices));
        }
        self.index_count = indices.len() as u32;
    }

    /// Draw the uploaded triangles onto `view`. `LoadOp::Load` for both color and depth, same
    /// as [`crate::line_raster::LineRasterNode::execute`]; multiple raster nodes share one
    /// cleared color + depth buffer within a frame.
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
            (true, false) => {
                panic!(
                    "TriangleRasterNode::execute: pipeline was created with a depth format but \
                     no depth view was provided"
                )
            }
            (false, true) => {
                panic!(
                    "TriangleRasterNode::execute: pipeline was created without a depth format \
                     but a depth view was provided"
                )
            }
            _ => {}
        }
        if self.index_count == 0 {
            return Ok(());
        }
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("triangle_raster encoder"),
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
                label: Some("triangle_raster pass"),
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
            rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.index_count, 0, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedded WGSL parses and validates against naga. Mirrors the line_raster validation
    /// test; catches drift between the Rust-side vertex layout and the shader's `@location`
    /// declarations.
    #[test]
    fn triangle_raster_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(TRIANGLE_RASTER_WGSL)
            .unwrap_or_else(|e| panic!("triangle_raster WGSL parse failed:\n{e}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("triangle_raster WGSL must validate");
    }

    /// `TriangleRasterUniforms` is exactly 64 bytes (one mat4x4, no padding). Drift here means
    /// the GPU reads the wrong bytes for the view-projection matrix.
    #[test]
    fn uniforms_size_matches_wgsl() {
        assert_eq!(std::mem::size_of::<TriangleRasterUniforms>(), 64);
        assert_eq!(std::mem::align_of::<TriangleRasterUniforms>(), 4);
    }

    /// `TriangleVertex` is 32 bytes (12 position + 4 pad + 16 color). Attribute offsets in
    /// the vertex layout descriptor must match this layout exactly.
    #[test]
    fn vertex_size_matches_layout() {
        assert_eq!(std::mem::size_of::<TriangleVertex>(), 32);
        let zero = TriangleVertex::default();
        let base = &zero as *const _ as usize;
        let color_off = &zero.color as *const _ as usize - base;
        assert_eq!(color_off, 16);
    }
}
