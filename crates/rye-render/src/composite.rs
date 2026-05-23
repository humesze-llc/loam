//! Final composite pass for browser-WebGPU: sample an sRGB offscreen scene texture
//! and write gamma-encoded values to a linear swapchain so the canvas compositor
//! displays them correctly.
//!
//! ## When this runs (and when it doesn't)
//!
//! - **Native** (D3D/Vulkan/Metal): the swapchain advertises `Bgra8UnormSrgb` or
//!   similar; GPU hardware encodes linear shader output to sRGB on write to the
//!   swapchain, and the OS compositor displays it as expected. No composite pass
//!   needed. `RenderDevice::new` doesn't allocate a [`CompositeNode`] in this case
//!   and the runner's `composite_to_swap` is a no-op.
//! - **Browser WebGPU** (Chrome 2026-05): the canvas surface only advertises
//!   linear formats. Direct linear writes display ~2.2x darker than native. We
//!   allocate an offscreen `Bgra8UnormSrgb` texture, render the scene + UI into
//!   it (GPU auto-encodes linear -> sRGB on write because the storage is sRGB),
//!   then this composite pass samples it (auto-decodes back to linear), applies
//!   `linear_to_srgb` in the fragment shader, and writes to the linear swapchain.
//!   The bits in the swapchain are sRGB-encoded, which is what the canvas
//!   compositor expects.
//!
//! ## Cost
//!
//! One extra render pass per frame. Fullscreen-triangle vertex (3 vertices) plus
//! a single sample-and-curve-apply fragment per pixel. Single-digit microseconds
//! on integrated GPUs at 1080p. Negligible vs. the polytope SDF raymarch.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, ColorTargetState, ColorWrites,
    CommandEncoder, Device, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureFormat, TextureSampleType,
    TextureView, TextureViewDimension, VertexState,
};

/// Final-pass gamma encoder. Owns one pipeline + one sampler + the bind group
/// layout; the actual bind group is rebuilt on resize (because the scene texture
/// view changes when the surface resizes).
pub struct CompositeNode {
    pipeline: RenderPipeline,
    sampler: Sampler,
    bind_group_layout: BindGroupLayout,
    /// Cached bind group for the current scene-target view. `None` until the first
    /// `set_scene_view` call; rebuilt whenever the scene target is reallocated.
    bind_group: Option<BindGroup>,
}

impl CompositeNode {
    /// Build a composite node that writes to `target_format`. Typically called
    /// once at device-creation time and reused for the device's lifetime.
    pub fn new(device: &Device, target_format: TextureFormat) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rye-render::composite::shader"),
            source: ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rye-render::composite::bgl"),
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
            label: Some("rye-render::composite::sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rye-render::composite::layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("rye-render::composite::pipeline"),
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

    /// Refresh the cached bind group when the scene-target view changes (which
    /// happens whenever the surface resizes; the scene_target texture gets
    /// recreated at the new size, invalidating any view that points at the old
    /// texture).
    pub fn set_scene_view(&mut self, device: &Device, scene_view: &TextureView) {
        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("rye-render::composite::bg"),
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

    /// Run the composite: read whatever is in the scene texture (bound during the
    /// last `set_scene_view`) and write gamma-encoded RGBA to `target_view`. No-op
    /// if `set_scene_view` hasn't been called yet (defensive against bind-group
    /// invalidation between resize and the next frame).
    pub fn run(&self, encoder: &mut CommandEncoder, target_view: &TextureView) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("rye-render::composite::pass"),
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
