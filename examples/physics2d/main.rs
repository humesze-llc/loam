//! 2D physics demo, spawns a bucket and a handful of random shapes
//! (circles, triangles, squares, pentagons, hexagons) and lets them
//! fall under gravity, collide, and settle.
//!
//! Visualization: each body's SDF is evaluated per fragment by
//! `physics2d.wgsl`. No raymarching, this is flat 2D.
//!
//! Press Esc or close the window to exit. Press R to reset.

use std::borrow::Cow;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use loam_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx, TickCtx};
use loam_math::{EuclideanR2, EuclideanR3};
use loam_physics::{
    euclidean_r2::{polygon_body, register_default_narrowphase, sphere_body, static_wall},
    field::Gravity,
    Collider, World,
};
use loam_render::device::RenderDevice;
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

// World viewport: x ∈ [−5, 5], y ∈ [0, 8].
const WORLD_X_MIN: f32 = -5.0;
const WORLD_Y_MIN: f32 = 0.0;
const WORLD_WIDTH: f32 = 10.0;
const WORLD_HEIGHT: f32 = 8.0;

const MAX_BODIES: usize = 64;

const PHYSICS_HZ: u32 = 120;
const FIXED_DT: f32 = 1.0 / PHYSICS_HZ as f32;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneUniforms {
    view: [f32; 4],
    resolution: [f32; 2],
    body_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuBody {
    position: [f32; 2],
    rotation: [f32; 2],
    scale: f32,
    kind: u32,
    extent: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BodyBuffer([GpuBody; MAX_BODIES]);

/// Deterministic tiny RNG for reproducible scenes.
struct Rng(u32);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.0 >> 16) & 0xFFFF) as f32 / 65535.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

fn build_world() -> World<EuclideanR2> {
    let mut world = World::new(EuclideanR2);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec2::new(0.0, -9.8))));

    // Bucket: floor + two walls. The floor's top face is at y=0.5; its
    // center is at y=0.25 with half-height 0.25.
    world.push_body(static_wall(Vec2::new(0.0, 0.25), Vec2::new(4.8, 0.25)));
    world.push_body(static_wall(Vec2::new(-4.8, 4.0), Vec2::new(0.2, 3.75)));
    world.push_body(static_wall(Vec2::new(4.8, 4.0), Vec2::new(0.2, 3.75)));

    // Drop a random mix of shapes. Staggered y so they don't all spawn
    // at the same height (which would produce instant overlap).
    let mut rng = Rng(0xDEADBEEF);
    for i in 0..14 {
        let x = rng.range(-3.8, 3.8);
        let y = 4.0 + (i as f32) * 0.3;
        let kind = (rng.next_f32() * 5.0) as u32; // 0..=4
        let radius = rng.range(0.22, 0.40);
        let vel = Vec2::new(rng.range(-0.7, 0.7), 0.0);
        match kind {
            0 => {
                world.push_body(sphere_body(Vec2::new(x, y), vel, radius, 1.0));
            }
            k @ 1..=4 => {
                // kind 1->triangle(3), 2->square(4), 3->pentagon(5), 4->hexagon(6)
                world.push_body(polygon_body(Vec2::new(x, y), vel, k + 2, radius, 1.0));
            }
            _ => unreachable!(),
        }
    }

    world
}

fn collect_gpu_bodies(world: &World<EuclideanR2>) -> (BodyBuffer, u32) {
    let mut buf = BodyBuffer([GpuBody::default(); MAX_BODIES]);
    let count = world.bodies.len().min(MAX_BODIES);
    for (i, b) in world.bodies.iter().take(MAX_BODIES).enumerate() {
        // Half-angle rotor: full-angle (cos, sin) for the shader.
        let r = b.orientation.rotation;
        let c = r.a * r.a - r.b * r.b;
        let s = 2.0 * r.a * r.b;

        let (scale, kind, extent) = match &b.collider {
            Collider::Sphere { radius, .. } => (*radius, 0u32, [0.0, 0.0]),
            Collider::Polygon2D { vertices } => {
                // Static-wall convention: inv_mass == 0 with 4 axis-aligned
                // vertices renders as a rectangle (kind 100).
                if b.inv_mass == 0.0 && vertices.len() == 4 {
                    let hx = vertices.iter().map(|v| v.x.abs()).fold(0.0, f32::max);
                    let hy = vertices.iter().map(|v| v.y.abs()).fold(0.0, f32::max);
                    (0.0, 100u32, [hx, hy])
                } else {
                    let circumradius = vertices.iter().map(|v| v.length()).fold(0.0, f32::max);
                    (circumradius, vertices.len() as u32, [0.0, 0.0])
                }
            }
            _ => (0.5, 0u32, [0.0, 0.0]),
        };

        buf.0[i] = GpuBody {
            position: b.position.to_array(),
            rotation: [c, s],
            scale,
            kind,
            extent,
        };
    }
    (buf, count as u32)
}

struct Physics2DApp {
    /// Required by the `App` trait; the framework passes this to
    /// `ShaderDb::apply_events` on hot-reload. This demo doesn't go
    /// through `ShaderDb` (it builds a bespoke pipeline directly),
    /// so the field is dormant.
    space: EuclideanR3,
    pipeline: wgpu::RenderPipeline,
    scene_buffer: wgpu::Buffer,
    body_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    world: World<EuclideanR2>,
    sim_time: f32,
}

impl Physics2DApp {
    fn reset(&mut self) {
        self.world = build_world();
        self.sim_time = 0.0;
    }
}

impl App for Physics2DApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let rd = ctx.rd;
        let shader_src = include_str!("physics2d.wgsl");
        let module = rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("physics2d"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let scene_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics2d scene ub"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics2d bodies ub"),
            size: std::mem::size_of::<BodyBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = rd
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("physics2d bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = rd.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics2d bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body_buffer.as_entire_binding(),
                },
            ],
        });

        let layout = rd
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("physics2d pipeline layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = rd
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("physics2d pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: rd.surface_bundle.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Ok(Self {
            space: EuclideanR3,
            pipeline,
            scene_buffer,
            body_buffer,
            bind_group,
            world: build_world(),
            sim_time: 0.0,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn tick(&mut self, dt: f32, _ctx: &mut TickCtx) {
        self.world.step(dt);
        self.sim_time += dt;
    }

    fn on_event(&mut self, ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {
        if let WindowEvent::KeyboardInput { event, .. } = ev {
            if event.state == ElementState::Pressed
                && matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyR))
            {
                self.reset();
            }
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let (body_data, body_count) = collect_gpu_bodies(&self.world);

        let scene = SceneUniforms {
            view: [WORLD_X_MIN, WORLD_Y_MIN, WORLD_WIDTH, WORLD_HEIGHT],
            resolution: [
                rd.surface_bundle.config.width as f32,
                rd.surface_bundle.config.height as f32,
            ],
            body_count,
            _pad: 0,
        };
        rd.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&scene));
        rd.queue
            .write_buffer(&self.body_buffer, 0, bytemuck::bytes_of(&body_data));

        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics2d encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("physics2d pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.03,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        rd.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        let dynamic_count = self
            .world
            .bodies
            .iter()
            .filter(|b| b.inv_mass > 0.0)
            .count();
        Cow::Owned(format!(
            "Loam - 2D Physics | {fps:.0} fps | {dynamic_count} bodies | sim {:.1}s (R: reset)",
            self.sim_time
        ))
    }
}

fn main() -> Result<()> {
    let _ = FIXED_DT; // documentation: framework's fixed_hz drives tick dt
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Loam - 2D Physics")
            .with_inner_size(winit::dpi::LogicalSize::new(900.0, 720.0))
            .with_visible(false),
        fixed_hz: PHYSICS_HZ,
        ..RunConfig::default()
    };
    run_with_config::<Physics2DApp>(config)
}
