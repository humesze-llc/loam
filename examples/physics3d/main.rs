//! 3D physics demo, spheres fall onto an infinite floor under gravity,
//! bounce off each other off-center, and tumble via friction-induced
//! angular velocity.
//!
//! Visualization: raymarched per-body sphere SDFs + floor half-space,
//! simple sun-lambert shading. Orbit camera (left-drag to rotate,
//! scroll to zoom). R resets; Esc exits.

use std::borrow::Cow;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use loam_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx, TickCtx};
use loam_camera::OrbitCamera;
use loam_math::EuclideanR3;
use loam_physics::{
    euclidean_r3::{box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3},
    field::Gravity,
    Collider, World,
};
use loam_render::device::RenderDevice;
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

const MAX_BODIES: usize = 32;

const PHYSICS_HZ: u32 = 120;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_forward: [f32; 3],
    fov_y_tan: f32,
    camera_right: [f32; 3],
    _pad1: f32,
    camera_up: [f32; 3],
    _pad2: f32,
    resolution: [f32; 2],
    body_count: u32,
    _pad3: u32,
    floor_normal: [f32; 3],
    floor_offset: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuBody {
    position: [f32; 3],
    /// 0.0 = sphere, 1.0 = box. Packed as float for WGSL alignment.
    kind: f32,
    /// For sphere: (radius, 0, 0). For box: half_extents.
    extents: [f32; 3],
    _pad: f32,
    /// Unit quaternion (x, y, z, w). Identity for spheres.
    rotation: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BodyBuffer([GpuBody; MAX_BODIES]);

// Compile-time layout checks. GpuBody is 48 bytes: vec3+f32 (16) +
// vec3+f32 (16) + vec4 (16). Matches the WGSL Body struct.
const _: () = assert!(std::mem::size_of::<SceneUniforms>() == 96);
const _: () = assert!(std::mem::size_of::<GpuBody>() == 48);
const _: () = assert!(std::mem::size_of::<BodyBuffer>() == 48 * MAX_BODIES);

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

fn build_world() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec3::new(0.0, -9.8, 0.0))));

    // Infinite floor at y = 0.
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0));

    let mut rng = Rng(0x51A3);
    // Mix of spheres and boxes falling with mild horizontal velocity.
    // Boxes exercise the full GJK+EPA path; spheres stay on the
    // analytical sphere-sphere / sphere-halfspace fast paths.
    for i in 0..14 {
        let x = rng.range(-2.0, 2.0);
        let z = rng.range(-2.0, 2.0);
        let y = 2.5 + (i as f32) * 0.55;
        let vel = Vec3::new(rng.range(-0.4, 0.4), 0.0, rng.range(-0.4, 0.4));
        if i % 2 == 0 {
            let radius = rng.range(0.25, 0.40);
            world.push_body(sphere_body_r3(Vec3::new(x, y, z), vel, radius, 1.0));
        } else {
            let hx = rng.range(0.22, 0.36);
            let hy = rng.range(0.22, 0.36);
            let hz = rng.range(0.22, 0.36);
            world.push_body(box_body(
                Vec3::new(x, y, z),
                vel,
                Vec3::new(hx, hy, hz),
                1.0,
            ));
        }
    }

    world
}

fn collect_gpu_bodies(world: &World<EuclideanR3>) -> (BodyBuffer, u32) {
    let mut buf = BodyBuffer([GpuBody::default(); MAX_BODIES]);
    // Skip the floor; it's rendered via Scene.floor_normal/offset.
    let mut count = 0u32;
    for body in world.bodies.iter() {
        if count as usize >= MAX_BODIES {
            break;
        }
        // Skip bodies whose state went non-finite, feeding NaN
        // positions/quats to wgpu triggers validation panics that
        // look unrelated to the physics bug that produced them.
        if !body.position.is_finite() {
            continue;
        }
        let q = body.orientation.rotation;
        if !q.length_squared().is_finite() {
            continue;
        }
        let rotation = [q.x, q.y, q.z, q.w];

        match &body.collider {
            Collider::Sphere { radius, .. } => {
                buf.0[count as usize] = GpuBody {
                    position: body.position.to_array(),
                    kind: 0.0,
                    extents: [*radius, 0.0, 0.0],
                    _pad: 0.0,
                    rotation,
                };
                count += 1;
            }
            Collider::ConvexPolytope3D { vertices } => {
                // Assume the polytope is a box for visualization
                // purposes: recover half-extents as the AABB of the
                // local vertex set. Works for `box_body()` output; for
                // arbitrary polytopes this renders their OBB instead
                // of the true shape.
                let (mut mn, mut mx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for &v in vertices {
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let half = (mx - mn) * 0.5;
                buf.0[count as usize] = GpuBody {
                    position: body.position.to_array(),
                    kind: 1.0,
                    extents: half.to_array(),
                    _pad: 0.0,
                    rotation,
                };
                count += 1;
            }
            Collider::HalfSpace { .. }
            | Collider::HalfSpace4D { .. }
            | Collider::Polygon2D { .. }
            | Collider::Box3 { .. }
            | Collider::ConvexPolytope4D { .. }
            | Collider::HyperSphere4D { .. } => {
                // Halfspace is implicit (scene uniforms); 2D polygons
                // and 4D polytopes shouldn't appear in a 3D world;
                // Box3 isn't used by this demo (it uses
                // `ConvexPolytope3D`-backed box bodies instead). Skip
                // defensively.
            }
        }
    }
    (buf, count)
}

struct Physics3DApp {
    /// Required by the `App` trait; passed to `ShaderDb::apply_events`
    /// on hot-reload. The pipeline here is bespoke (not loaded through
    /// `ShaderDb`), so the field is dormant.
    space: EuclideanR3,
    pipeline: wgpu::RenderPipeline,
    scene_buffer: wgpu::Buffer,
    body_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    world: World<EuclideanR3>,
    camera: OrbitCamera,
    sim_time: f32,
}

impl Physics3DApp {
    fn reset(&mut self) {
        self.world = build_world();
        self.sim_time = 0.0;
    }
}

impl App for Physics3DApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let rd = ctx.rd;
        let shader_src = include_str!("physics3d.wgsl");
        let module = rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("physics3d"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let scene_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics3d scene ub"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_buffer = rd.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics3d bodies ub"),
            size: std::mem::size_of::<BodyBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = rd
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("physics3d bgl"),
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
            label: Some("physics3d bg"),
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
                label: Some("physics3d pipeline layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = rd
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("physics3d pipeline"),
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

        // Seed orbit camera to look at the bucket area from slightly above.
        let mut camera = OrbitCamera::default();
        camera.set_orbit(6.0, -0.45);

        Ok(Self {
            space: EuclideanR3,
            pipeline,
            scene_buffer,
            body_buffer,
            bind_group,
            world: build_world(),
            camera,
            sim_time: 0.0,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn tick(&mut self, dt: f32, _ctx: &mut TickCtx) {
        // Wrap the physics step in `catch_unwind` so a Rust panic in
        // narrowphase / EPA produces a tagged stderr line before the
        // app-level hook prints the full backtrace. Diagnostic for the
        // "no backtrace on crash" mystery; safe to keep, this path is
        // off the steady-state hot loop.
        let world = &mut self.world;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.step(dt);
        }));
        if let Err(payload) = result {
            let msg = if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else {
                "<non-string panic payload>".to_string()
            };
            let dynamic_count = self
                .world
                .bodies
                .iter()
                .filter(|b| b.inv_mass > 0.0)
                .count();
            eprintln!("PHYSICS STEP PANIC CAUGHT: {msg}");
            eprintln!(
                "  sim_time={:.3}s  dynamic_bodies={}",
                self.sim_time, dynamic_count
            );
            std::panic::resume_unwind(payload);
        }
        self.sim_time += dt;
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.camera.advance(ctx.input);
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
        let cam_view = self.camera.view();
        let (body_data, body_count) = collect_gpu_bodies(&self.world);

        let scene = SceneUniforms {
            camera_pos: cam_view.position.to_array(),
            _pad0: 0.0,
            camera_forward: cam_view.forward.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            camera_right: cam_view.right.to_array(),
            _pad1: 0.0,
            camera_up: cam_view.up.to_array(),
            _pad2: 0.0,
            resolution: [
                rd.surface_bundle.config.width as f32,
                rd.surface_bundle.config.height as f32,
            ],
            body_count,
            _pad3: 0,
            floor_normal: [0.0, 1.0, 0.0],
            floor_offset: 0.0,
        };
        rd.queue
            .write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&scene));
        rd.queue
            .write_buffer(&self.body_buffer, 0, bytemuck::bytes_of(&body_data));

        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics3d encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("physics3d pass"),
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
            "Loam - 3D Physics | {fps:.0} fps | {dynamic_count} bodies | sim {:.1}s (R: reset, drag: orbit, scroll: zoom)",
            self.sim_time
        ))
    }
}

fn main() -> Result<()> {
    // Force RUST_BACKTRACE=1 so a panic in the physics step produces
    // an actual stack trace even if the user didn't set the env var.
    // Also install a hook that flushes stderr before the process dies:
    // panics on the main/event-loop thread don't always get a chance
    // to print through the default hook when winit unwinds weirdly.
    std::env::set_var("RUST_BACKTRACE", "1");
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("\n=== PHYSICS3D PANIC ===");
        eprintln!("{info}");
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("backtrace:\n{bt}");
        eprintln!("=======================\n");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        default_hook(info);
    }));

    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Loam - 3D Physics")
            .with_inner_size(winit::dpi::LogicalSize::new(900.0, 720.0))
            .with_visible(false),
        fixed_hz: PHYSICS_HZ,
        ..RunConfig::default()
    };
    run_with_config::<Physics3DApp>(config)
}
