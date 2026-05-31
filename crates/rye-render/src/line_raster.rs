//! Line rasterizer pipeline. Antialiased line-list rendering composed on top of an existing
//! color attachment. Lines are quad-expanded in the vertex shader to give them pixel width and
//! antialiased edges; the fragment shader smoothsteps coverage from line center to expanded
//! edge.
//!
//! Lives next to [`crate::raymarch`] modules and is constructed standalone, the same way the
//! existing `Hyperslice4DNode` is.
//!
//! ## Pipeline shape
//!
//! - **Vertex buffer**: 4 sprite-corner indices (`0u32`, `1`, `2`, `3`). Static, shared across
//!   all segments.
//! - **Index buffer**: `[0u32, 1, 2, 2, 1, 3]`. Static, two triangles per quad.
//! - **Instance buffer**: per-segment `LineInstance` data (start_pos, end_pos, start_color,
//!   end_color, width). Re-uploaded when the line mesh changes via the upload method.
//! - **Uniform buffer**: `LineRasterUniforms` (view-projection matrix + viewport size).
//!   Re-uploaded per frame via the camera method.
//!
//! ## Depth
//!
//! [`LineRasterNode::new`] takes a [`crate::DepthMode`]:
//!
//! - `Off`: no depth attachment, lines draw on top in submission order (HUD-style overlay).
//! - `ReadWrite { format }`: standard scene-geometry depth, `depth_compare: Less` +
//!   depth-write enabled. Use for opaque line wireframes that should occlude / be
//!   occluded by other depth-aware geometry.
//! - `ReadOnly { format }`: depth-test against the existing buffer but don't write. Use
//!   for alpha-blended overlays that should be occluded by scene geometry in front of
//!   them without burying subsequent draws behind them.
//!
//! When depth is active (`ReadWrite` or `ReadOnly`), [`LineRasterNode::execute`] requires
//! the matching depth attachment.
//!
//! The caller owns the depth texture lifecycle: examples create a swapchain-sized depth
//! texture, recreate it on resize, and clear it via a dedicated clear pass before the
//! rasterizer's draw (which uses `LoadOp::Load` for both color and depth).
//!
//! ## Current limitations
//!
//! - R³ + R⁴ impls only; other dimensions are additive on `RasterizableSpace<N>` and
//!   `Visualizable<N>`.
//! - [`Projection<N>`] variants beyond `Identity` + `Orthographic` (Perspective, Schlegel,
//!   Stereographic, Hyperslice) are open extensions.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use rye_math::{Projection, RasterizableSpace};
use rye_shape::LineMesh;
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

/// WGSL source for the line rasterizer pipeline. Embedded as `&'static str` so the build is
/// self-contained (no asset loading at startup). Naga-validated as part of the unit tests.
const LINE_RASTER_WGSL: &str = include_str!("line_raster.wgsl");

/// Camera uniform shared with the rasterizer's vertex shader. Matches the WGSL `CameraUniform`
/// struct exactly: 64 bytes for the matrix, 8 bytes for viewport, 8 bytes pad to 16-byte align.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineRasterUniforms {
    /// World-to-clip transform. Bit-identical to whatever the raymarcher uses for the same
    /// frame; both passes consume this to produce consistent screen-space projection.
    pub view_projection: [[f32; 4]; 4],
    /// Render target size in pixels. Used by the vertex shader to convert pixel widths into
    /// NDC offsets.
    pub viewport_size: [f32; 2],
    /// Padding to round the struct to 16-byte alignment for `std140` uniform layout.
    pub _pad: [f32; 2],
}

impl Default for LineRasterUniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            viewport_size: [1.0, 1.0],
            _pad: [0.0; 2],
        }
    }
}

/// Per-instance line data uploaded to the GPU. One [`LineInstance`] per visible segment; the
/// vertex shader expands each into a screen-space quad. Layout matches the `@location(1..=5)`
/// attribute slots in `line_raster.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct LineInstance {
    start_pos: [f32; 3],
    _pad0: f32,
    end_pos: [f32; 3],
    _pad1: f32,
    start_color: [f32; 4],
    end_color: [f32; 4],
    width_px: f32,
    _pad2: [f32; 3],
}

/// Antialiased line-list rasterizer. Construct once per `RenderDevice`; reuse across frames.
///
/// Upload mesh data via [`Self::upload`]; draw onto a color attachment via [`Self::execute`].
/// The pipeline owns its own vertex / index / instance / uniform buffers; the caller doesn't
/// manage GPU resources directly.
pub struct LineRasterNode {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,

    /// Static corner-index buffer (always `[0u32, 1, 2, 3]`). Per-vertex input to the shader's
    /// `corner` location.
    corner_buf: Buffer,
    /// Static index buffer (`[0u32, 1, 2, 2, 1, 3]`, two triangles per quad).
    index_buf: Buffer,
    /// Per-instance buffer (one [`LineInstance`] per segment). Grows on demand; re-uploaded via
    /// [`Self::upload`].
    instance_buf: Buffer,
    /// Number of segments currently uploaded. `0` means [`Self::execute`] is a no-op.
    instance_count: u32,
    /// Allocated capacity of `instance_buf` in instances. The buffer is re-created if a future
    /// upload exceeds this.
    instance_capacity: u32,
    /// `true` if the pipeline was created with a depth attachment; [`Self::execute`] then
    /// requires the matching depth view. Tracks the depth-or-not API contract so callers
    /// don't silently get mismatched render passes.
    has_depth: bool,
    /// Per-call scratch for the instances vector built inside [`Self::upload`].
    /// Persisted on the node so the Vec's heap capacity is reused across calls;
    /// `clear()` empties the logical length without freeing the buffer, and the
    /// next push series reuses the same allocation. Drops per-frame alloc count
    /// for dynamic-mesh users (polytope_playground) by 1.
    instances_scratch: Vec<LineInstance>,
}

impl LineRasterNode {
    /// Construct the pipeline.
    ///
    /// - `surface_format` must match the color attachment at draw time.
    /// - `depth`: see [`crate::DepthMode`]. Determines whether the pipeline reads depth,
    ///   reads + writes depth, or skips it entirely.
    /// - `sample_count` must match the attachment's MSAA sample count (color and depth
    ///   must share it when depth is enabled).
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("line_raster shader"),
            source: ShaderSource::Wgsl(LINE_RASTER_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("line_raster uniforms"),
            size: std::mem::size_of::<LineRasterUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("line_raster bgl"),
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
            label: Some("line_raster bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("line_raster pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // Vertex layout: one u32 per corner-index vertex, plus per-instance line data.
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
        let instance_attrs = [
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 1,
            }, // start_pos
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 16,
                shader_location: 2,
            }, // end_pos (after 4-byte pad)
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 32,
                shader_location: 3,
            }, // start_color
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 48,
                shader_location: 4,
            }, // end_color
            VertexAttribute {
                format: VertexFormat::Float32,
                offset: 64,
                shader_location: 5,
            }, // width_px
        ];
        let instance_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<LineInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &instance_attrs,
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("line_raster pipeline"),
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
                    // Alpha-blend overlay so partial-coverage AA edges fade into the
                    // underlying raymarched scene.
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
                // `LessEqual` rather than the usual `Less`: lines are typically drawn on top
                // of filled triangles as overlays / outlines / wireframes, and the polygon
                // outline use case puts the line at *exactly* the polygon's depth. Strict
                // `Less` would discard the line in that case (`line_depth < tri_depth` fails
                // when they're equal), making outlines invisible wherever they coincide with
                // their own polygon. `LessEqual` keeps them visible. Lines geometrically
                // behind a filled surface still fail the test and are correctly occluded.
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

        // Static corner buffer: 4 vertices, one per quad corner.
        let corner_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_raster corner buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 3]),
            usage: BufferUsages::VERTEX,
        });

        // Static index buffer: two triangles per quad.
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_raster index buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 2, 1, 3]),
            usage: BufferUsages::INDEX,
        });

        // Instance buffer starts empty; grown on first upload.
        let instance_capacity = 0u32;
        let instance_buf = device.create_buffer(&BufferDescriptor {
            label: Some("line_raster instance buffer"),
            size: 64, // placeholder, will be re-created on first upload
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
            instance_capacity,
            has_depth: depth.is_active(),
            instances_scratch: Vec::new(),
        }
    }

    /// Update the camera uniform. Call once per frame before [`Self::execute`].
    pub fn set_camera(&self, queue: &Queue, view_projection: Mat4, viewport_size: Vec2) {
        let uniforms = LineRasterUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            viewport_size: viewport_size.to_array(),
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Upload a [`LineMesh`] for rendering. Tessellates each segment via the space's
    /// [`RasterizableSpace::tessellate_segment`] (for flat spaces with `samples_per_segment ==
    /// 1` this is just the endpoints), projects each tessellated point through `projection` to
    /// R³, and packs the result into the instance buffer.
    ///
    /// `samples_per_segment` controls geodesic-space sampling density; for flat impls
    /// (`EuclideanR3`) any value >= 1 produces visually identical results since interior lerp
    /// points are still collinear with the endpoints. Use `1` for flat spaces, higher values
    /// for future curved-space impls.
    pub fn upload<S, const N: usize>(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: &LineMesh<N>,
        projection: &Projection<N>,
        samples_per_segment: usize,
    ) where
        S: RasterizableSpace<N>,
    {
        let samples = samples_per_segment.max(1);
        // Build into the node-owned scratch (capacity reused across frames),
        // then ship it to the GPU below. Factored out so the CPU-side
        // tessellation + projection + non-finite drop is unit-testable without
        // a GPU device (see `build_line_instances`).
        build_line_instances::<S, N>(&mut self.instances_scratch, mesh, projection, samples);

        let needed_capacity = self.instances_scratch.len() as u32;
        if needed_capacity > self.instance_capacity {
            // Grow buffer; round up to next power of two to amortize re-allocations.
            let new_cap = needed_capacity.next_power_of_two().max(16);
            self.instance_buf = device.create_buffer(&BufferDescriptor {
                label: Some("line_raster instance buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<LineInstance>() as u64),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }

        if !self.instances_scratch.is_empty() {
            queue.write_buffer(
                &self.instance_buf,
                0,
                bytemuck::cast_slice(&self.instances_scratch),
            );
        }
        self.instance_count = needed_capacity;
    }

    /// Record a render pass that draws the uploaded line mesh into `view`, using the
    /// caller-supplied `encoder`. **Does NOT call `encoder.finish()` or
    /// `queue.submit`**; those are the caller's responsibility, typically the
    /// runner batching multiple passes into one submit per frame (the
    /// `App::record` path).
    ///
    /// `LoadOp::Load` preserves the existing color attachment contents; the
    /// rasterizer composes with whatever ran before it.
    ///
    /// `depth_view` is required when the pipeline was created with
    /// `Some(depth_format)` and must be `None` otherwise. Mismatch panics with a
    /// descriptive message rather than surfacing a less-readable wgpu validation
    /// error.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: Option<&crate::Viewport>,
    ) {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => {
                panic!(
                    "LineRasterNode::record: pipeline was created with a depth format but \
                     no depth view was provided"
                )
            }
            (false, true) => {
                panic!(
                    "LineRasterNode::record: pipeline was created without a depth format but \
                     a depth view was provided"
                )
            }
            _ => {}
        }
        if self.instance_count == 0 {
            return;
        }
        let depth_attachment = depth_view.map(|dv| RenderPassDepthStencilAttachment {
            view: dv,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("line_raster pass"),
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

    /// Legacy wrapper: builds its own encoder + submits. Kept for backwards
    /// compatibility with demos still on the multi-submit `App::render` path.
    /// New code should prefer `Self::record` called from inside
    /// `App::record`, which lets the runner share one
    /// encoder across the whole frame.
    pub fn execute(
        &self,
        rd: &RenderDevice,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: Option<&crate::Viewport>,
    ) -> anyhow::Result<()> {
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("line_raster encoder"),
        });
        self.record(&mut encoder, view, depth_view, viewport);
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

/// Tessellate, project, and pack a [`LineMesh`] into `out` (one [`LineInstance`]
/// per rendered sub-segment). `out` is caller-owned scratch; it is `clear()`-ed
/// here (preserving heap capacity, so a reused buffer does zero allocations in
/// the steady state) and `reserve`-d for the worst-case segment count, so the
/// only allocation is the first-call growth or a growth past the prior high
/// watermark.
///
/// Each input segment is tessellated via [`RasterizableSpace::tessellate_segment`]
/// (for flat spaces with `samples == 1` this is just the two endpoints) and the
/// resulting consecutive points are paired into rendered sub-segments with a
/// linear color gradient along the original segment.
///
/// Sub-segments with a non-finite projected endpoint are dropped: a central
/// projection (Schlegel, Stereographic, Perspective4D) can map a vertex on the
/// projection center / pole to a NaN or infinity, and a single non-finite point
/// poisons the GPU view-projection divide into a full-screen garbage quad rather
/// than a missing line. `is_finite` rejects every quiet-NaN bit pattern AND both
/// infinities (an `== NAN` sentinel test would miss infinities and signaling
/// NaNs); the `continue` drops the offending sub-segment without pushing, so the
/// scratch capacity is untouched (no `filter().collect()` reallocation).
fn build_line_instances<S, const N: usize>(
    out: &mut Vec<LineInstance>,
    mesh: &LineMesh<N>,
    projection: &Projection<N>,
    samples: usize,
) where
    S: RasterizableSpace<N>,
{
    // `tess_buf` stays local because it's generic over `S::Point` and the
    // node-level scratch field can't pick a single type. For flat spaces
    // (the common case) `samples = 1` so this Vec stays at capacity 2; the
    // allocation cost is tiny (~64 bytes) and short-lived. Once a curved-
    // space user shows real cost here, lift it into a type-erased scratch
    // (`Vec<u8>` with `bytemuck::cast_slice_mut`) on the node.
    let mut tess_buf: Vec<S::Point> = Vec::with_capacity(samples + 1);

    out.clear();
    out.reserve(mesh.segments.len() * samples);

    for ((seg, (color_a, color_b)), &width) in mesh
        .segments
        .iter()
        .zip(mesh.colors.iter())
        .zip(mesh.widths.iter())
    {
        tess_buf.clear();
        let p0 = S::array_to_point(seg.0);
        let p1 = S::array_to_point(seg.1);
        S::tessellate_segment(p0, p1, samples, &mut tess_buf);

        // tess_buf now holds samples+1 points. Pair consecutive points into rendered
        // sub-segments. Per-endpoint color is linearly interpolated along the original
        // segment so multi-sample tessellation preserves the gradient.
        let n_sub = tess_buf.len().saturating_sub(1);
        for i in 0..n_sub {
            let t0 = i as f32 / samples as f32;
            let t1 = (i + 1) as f32 / samples as f32;
            let c0 = lerp_color(*color_a, *color_b, t0);
            let c1 = lerp_color(*color_a, *color_b, t1);
            let q0 = S::project_point(tess_buf[i], projection);
            let q1 = S::project_point(tess_buf[i + 1], projection);
            if !q0.is_finite() || !q1.is_finite() {
                continue;
            }
            out.push(LineInstance {
                start_pos: q0.to_array(),
                _pad0: 0.0,
                end_pos: q1.to_array(),
                _pad1: 0.0,
                start_color: c0,
                end_color: c1,
                width_px: width,
                _pad2: [0.0; 3],
            });
        }
    }
}

fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded WGSL parses and validates against naga, matching the kernel-validation
    /// pattern used by the raymarch nodes. Catches drift between Rust-side attribute layouts
    /// and the shader's `@location` declarations.
    #[test]
    fn line_raster_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(LINE_RASTER_WGSL)
            .unwrap_or_else(|e| panic!("line_raster WGSL parse failed:\n{e}"));
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("line_raster WGSL must validate");
    }

    /// `LineRasterUniforms` is 80 bytes (64 mat + 8 vec2 + 8 pad). Matches the WGSL
    /// `CameraUniform` std140 layout exactly. Drift here means the GPU reads garbage from
    /// the wrong offsets.
    #[test]
    fn uniforms_size_matches_wgsl() {
        assert_eq!(std::mem::size_of::<LineRasterUniforms>(), 80);
        assert_eq!(std::mem::align_of::<LineRasterUniforms>(), 4);
    }

    /// `LineInstance` is 80 bytes (12 + 4 + 12 + 4 + 16 + 16 + 4 + 12 pad). Each attribute
    /// offset in the vertex layout descriptor must match this layout exactly.
    #[test]
    fn instance_size_matches_layout() {
        assert_eq!(std::mem::size_of::<LineInstance>(), 80);
        // Spot-check the field offsets (`memoffset::offset_of!` would be cleaner but adds a
        // dep just for this).
        let zero = LineInstance::default();
        let base = &zero as *const _ as usize;
        let end_pos_off = &zero.end_pos as *const _ as usize - base;
        let start_color_off = &zero.start_color as *const _ as usize - base;
        let end_color_off = &zero.end_color as *const _ as usize - base;
        let width_off = &zero.width_px as *const _ as usize - base;
        assert_eq!(end_pos_off, 16);
        assert_eq!(start_color_off, 32);
        assert_eq!(end_color_off, 48);
        assert_eq!(width_off, 64);
    }

    // ---- non-finite drop backstop ---------------------------------------
    //
    // These exercise the CPU-side instance builder directly (no GPU device):
    // `build_line_instances` is the part of `upload` that tessellates,
    // projects, and packs, including the `is_finite` drop guard. `EuclideanR3`
    // + `Projection::Identity` is a bitwise pass-through, so a non-finite
    // coordinate in the input mesh flows straight to the projected endpoint and
    // exercises the guard without needing a projection that manufactures one.

    use rye_math::EuclideanR3;
    use rye_shape::LineMesh;

    /// Build a single-segment R³ mesh with the given endpoints, uniform white
    /// color, and unit width.
    fn one_segment_mesh(a: [f32; 3], b: [f32; 3]) -> LineMesh<3> {
        LineMesh {
            segments: vec![(a, b)],
            colors: vec![([1.0; 4], [1.0; 4])],
            widths: vec![1.0],
        }
    }

    /// A segment with all-finite endpoints survives; a segment carrying a NaN
    /// endpoint is dropped, so the built instance count excludes it. Pins the
    /// invariant that a non-finite endpoint contributes ZERO instances rather
    /// than a poisoned one.
    #[test]
    fn upload_drops_non_finite_segments() {
        let mut scratch: Vec<LineInstance> = Vec::new();

        let finite = one_segment_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        build_line_instances::<EuclideanR3, 3>(&mut scratch, &finite, &Projection::Identity, 1);
        assert_eq!(scratch.len(), 1, "finite segment must produce one instance");

        let nan = one_segment_mesh([0.0, 0.0, 0.0], [f32::NAN, 1.0, 1.0]);
        build_line_instances::<EuclideanR3, 3>(&mut scratch, &nan, &Projection::Identity, 1);
        assert_eq!(scratch.len(), 0, "NaN-endpoint segment must be dropped");
    }

    /// Re-running the builder over a NaN-bearing mesh leaves the scratch Vec's
    /// heap capacity unchanged from the prior (finite) build: the drop is a
    /// `continue`, not a `filter().collect()`, so the zero-per-frame-allocation
    /// contract holds even when segments are culled.
    #[test]
    fn upload_drops_non_finite_without_reallocating() {
        let mut scratch: Vec<LineInstance> = Vec::new();

        // Prime the scratch with several finite segments so it grows once.
        let mut many = LineMesh::<3>::default();
        for k in 0..8 {
            let f = k as f32;
            many.segments.push(([f, 0.0, 0.0], [f, 1.0, 0.0]));
            many.colors.push(([1.0; 4], [1.0; 4]));
            many.widths.push(1.0);
        }
        build_line_instances::<EuclideanR3, 3>(&mut scratch, &many, &Projection::Identity, 1);
        assert_eq!(scratch.len(), 8);
        let cap_after_growth = scratch.capacity();

        // Now feed a mesh whose only segment is non-finite. The builder
        // `clear()`s then `reserve()`s for one segment (<= current cap, so no
        // growth) and drops the segment, leaving capacity untouched.
        let nan = one_segment_mesh([f32::NAN, 0.0, 0.0], [0.0, 0.0, 0.0]);
        build_line_instances::<EuclideanR3, 3>(&mut scratch, &nan, &Projection::Identity, 1);
        assert_eq!(scratch.len(), 0);
        assert_eq!(
            scratch.capacity(),
            cap_after_growth,
            "dropping a segment must not change the scratch heap capacity"
        );
    }

    /// The drop predicate is `!is_finite()`, not an `== NAN` sentinel: a segment
    /// with a `+inf` (or `-inf`) endpoint is ALSO dropped. An equality-to-NaN
    /// test would let infinities through to poison the GPU divide.
    #[test]
    fn upload_drop_predicate_catches_infinity_not_just_nan() {
        let mut scratch: Vec<LineInstance> = Vec::new();

        let pos_inf = one_segment_mesh([0.0, 0.0, 0.0], [f32::INFINITY, 1.0, 1.0]);
        build_line_instances::<EuclideanR3, 3>(&mut scratch, &pos_inf, &Projection::Identity, 1);
        assert_eq!(scratch.len(), 0, "+inf endpoint must be dropped");

        let neg_inf = one_segment_mesh([f32::NEG_INFINITY, 0.0, 0.0], [1.0, 1.0, 1.0]);
        build_line_instances::<EuclideanR3, 3>(&mut scratch, &neg_inf, &Projection::Identity, 1);
        assert_eq!(scratch.len(), 0, "-inf endpoint must be dropped");
    }
}
