//! A custom-WGSL fullscreen (or viewport) shader pass: the reusable
//! generalization of [`crate::composite::CompositeNode`]. A caller supplies a
//! fragment shader plus a uniform blob; the node provides the fullscreen-triangle
//! vertex stage and one uniform bind group. Used for backdrops (gradient sky),
//! sweeps/washes, glows, and other screen-space effects.
//!
//! The fragment WGSL must define `@fragment fn fs_main(in: VsOut) -> @location(0)
//! vec4<f32>` and a uniform at `@group(0) @binding(0)`; [`FULLSCREEN_VERTEX_WGSL`]
//! (prepended automatically) supplies `VsOut` (with `uv` in `[0, 1]`, origin
//! top-left) and `vs_main`.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferBindingType, BufferDescriptor,
    BufferUsages, ColorTargetState, ColorWrites, Device, FragmentState, LoadOp, MultisampleState,
    Operations, PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StoreOp, TextureFormat, TextureView, VertexState,
};

use crate::device::RenderDevice;
use crate::Viewport;

/// Fullscreen-triangle vertex stage prepended to every effect's fragment source.
/// `uv` runs `[0, 1]` with the origin at the top-left.
pub const FULLSCREEN_VERTEX_WGSL: &str = r#"
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VsOut;
    out.clip_pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = uv[idx];
    return out;
}
"#;

pub struct ShaderEffect {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,
}

impl ShaderEffect {
    /// Build an effect that writes `target_format` with `blend` (use
    /// [`BlendState::REPLACE`] for an opaque backdrop, `ALPHA_BLENDING` for an
    /// overlay). `uniform_size` is the byte size of the uniform struct the
    /// fragment shader declares at `@group(0) @binding(0)`.
    pub fn new(
        device: &Device,
        target_format: TextureFormat,
        fragment_wgsl: &str,
        uniform_size: u64,
        blend: BlendState,
        sample_count: u32,
    ) -> Self {
        let source = format!("{FULLSCREEN_VERTEX_WGSL}\n{fragment_wgsl}");
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rye-render::shader_effect::module"),
            source: ShaderSource::Wgsl(source.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("rye-render::shader_effect::uniforms"),
            size: uniform_size.max(16),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rye-render::shader_effect::bgl"),
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
            label: Some("rye-render::shader_effect::bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rye-render::shader_effect::layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("rye-render::shader_effect::pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: Some(blend),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
            uniform_buf,
            bind_group,
        }
    }

    /// Overwrite the uniform buffer. `bytes` must match the declared struct size.
    pub fn set_uniforms(&self, queue: &Queue, bytes: &[u8]) {
        queue.write_buffer(&self.uniform_buf, 0, bytes);
    }

    /// Draw the effect into `view` (whole framebuffer, or `viewport` if given),
    /// preserving existing contents (`LoadOp::Load`) so it composites.
    pub fn execute(
        &self,
        rd: &RenderDevice,
        view: &TextureView,
        viewport: Option<&Viewport>,
    ) -> anyhow::Result<()> {
        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rye-render::shader_effect::encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("rye-render::shader_effect::pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(vp) = viewport {
                vp.apply(&mut pass);
            }
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}
