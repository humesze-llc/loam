//! Shader-driven UI: draw a UI element with a custom WGSL fragment shader, via an
//! `egui_wgpu` paint callback. The fragment receives `uv` in `[0, 1]` over the
//! widget rect (origin top-left) and a uniform blob at `@group(0) @binding(0)`.
//!
//! [`ShaderUi`] (a pipeline cache + the surface format/sample count) is registered
//! into the egui renderer's callback resources by `UiIntegration::new`; widgets
//! then reference shaders by their `&'static str` source.

use std::collections::HashMap;

use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};

/// Fullscreen-triangle vertex stage prepended to every shader-UI fragment.
/// `uv` runs `[0, 1]` with the origin at the top-left of the widget.
const FULLSCREEN_VS: &str = r#"
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
    var uv = array<vec2<f32>, 3>(vec2<f32>(0.0,1.0), vec2<f32>(2.0,1.0), vec2<f32>(0.0,-1.0));
    var out: VsOut;
    out.clip_pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = uv[idx];
    return out;
}
"#;

struct Cached {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Per-renderer shader-UI state: the target format/samples and a cache of
/// compiled pipelines keyed by their fragment source.
pub struct ShaderUi {
    format: wgpu::TextureFormat,
    samples: u32,
    cache: HashMap<&'static str, Cached>,
}

impl ShaderUi {
    pub fn new(format: wgpu::TextureFormat, samples: u32) -> Self {
        Self {
            format,
            samples,
            cache: HashMap::new(),
        }
    }
}

fn build(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
    fragment_wgsl: &str,
    uniform_size: u64,
) -> Cached {
    let source = format!("{FULLSCREEN_VS}\n{fragment_wgsl}");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rye-egui::shader_ui::module"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rye-egui::shader_ui::uniforms"),
        size: uniform_size.max(16),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rye-egui::shader_ui::bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rye-egui::shader_ui::bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rye-egui::shader_ui::layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rye-egui::shader_ui::pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    });
    Cached {
        pipeline,
        uniform_buf,
        bind_group,
    }
}

struct ShaderCallback {
    wgsl: &'static str,
    uniform_size: u64,
    uniforms: Vec<u8>,
}

impl CallbackTrait for ShaderCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(state) = resources.get_mut::<ShaderUi>() else {
            return Vec::new();
        };
        let (format, samples) = (state.format, state.samples);
        let cached = state
            .cache
            .entry(self.wgsl)
            .or_insert_with(|| build(device, format, samples, self.wgsl, self.uniform_size));
        queue.write_buffer(&cached.uniform_buf, 0, &self.uniforms);
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(state) = resources.get::<ShaderUi>() else {
            return;
        };
        let Some(cached) = state.cache.get(self.wgsl) else {
            return;
        };
        let vp = info.viewport_in_pixels();
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );
        render_pass.set_pipeline(&cached.pipeline);
        render_pass.set_bind_group(0, &cached.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Allocate a `size` rect and fill it with `fragment_wgsl` (which must define
/// `@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32>` and a uniform at
/// `@group(0) @binding(0)`). `uniforms` must match the declared struct size.
pub fn shader_widget(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    fragment_wgsl: &'static str,
    uniform_size: u64,
    uniforms: Vec<u8>,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ShaderCallback {
            wgsl: fragment_wgsl,
            uniform_size,
            uniforms,
        },
    ));
    response
}
