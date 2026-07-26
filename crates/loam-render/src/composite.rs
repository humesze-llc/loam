//! Final composite pass for browser-WebGPU: sample the offscreen scene texture
//! and write gamma-encoded values to a linear swapchain.
//!
//! Native (D3D/Vulkan/Metal) swapchains advertise sRGB formats, so the GPU
//! encodes linear output on write and no pass is needed; `RenderDevice::new`
//! skips the [`CompositeNode`] there. Browser WebGPU (Chrome 2026-05) only
//! advertises linear canvas formats, so direct writes display ~2.2x dark.
//! Instead the scene renders into an offscreen target carrying the canvas
//! format's sRGB sibling (`Bgra8Unorm` -> `Bgra8UnormSrgb`), or the canvas
//! format itself where it has none (`Rgba16Float`), and this pass samples it,
//! applies `linear_to_srgb`, and writes the sRGB-encoded bits the compositor
//! expects. Scene texels read back linear either way, by sampler decode or by
//! storage; egui-painted texels do not on the no-sibling arm, where egui-wgpu
//! writes encoded values that this pass then encodes twice.
//!
//! Cost: one fullscreen-triangle pass per frame, single-digit microseconds
//! at 1080p; negligible vs. the polytope SDF raymarch.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, ColorTargetState, ColorWrites,
    CommandEncoder, Device, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureFormat, TextureSampleType,
    TextureView, TextureViewDimension, VertexState,
};

/// Final-pass gamma encoder. The bind group is rebuilt on resize, since the
/// scene texture view changes when the surface resizes.
pub struct CompositeNode {
    pipeline: RenderPipeline,
    sampler: Sampler,
    bind_group_layout: BindGroupLayout,
    /// `None` until the first `set_scene_view`; rebuilt whenever the scene
    /// target is reallocated.
    bind_group: Option<BindGroup>,
}

impl CompositeNode {
    /// Build a composite node that writes to `target_format`. Reused for the
    /// device's lifetime.
    pub fn new(device: &Device, target_format: TextureFormat) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("loam-render::composite::shader"),
            source: ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("loam-render::composite::bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("loam-render::composite::sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("loam-render::composite::layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("loam-render::composite::pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            sampler,
            bind_group_layout,
            bind_group: None,
        }
    }

    /// Refresh the cached bind group when the scene-target view changes, e.g.
    /// after a resize reallocates the scene texture.
    pub fn set_scene_view(&mut self, device: &Device, scene_view: &TextureView) {
        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("loam-render::composite::bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(scene_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.bind_group = Some(bg);
    }

    /// Read the bound scene texture and write gamma-encoded RGBA to
    /// `target_view`. No-op before the first `set_scene_view`.
    pub fn run(&self, encoder: &mut CommandEncoder, target_view: &TextureView) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("loam-render::composite::pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
