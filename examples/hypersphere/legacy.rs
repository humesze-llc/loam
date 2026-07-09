//! Hypersphere `w`-slice viewer, drop one or many 4D balls onto a
//! 4D floor and render their 3D cross-sections at user-controlled
//! `w₀`.
//!
//! The 4-ball `B = { x ∈ R⁴ : |x − c| ≤ r }` cross-sectioned at
//! `w = w₀` is a 3D ball of radius `sqrt(r² − (w₀ − c.w)²)` for
//! `|w₀ − c.w| < r`, and empty otherwise. As you scrub `w₀` past a
//! body's `w`-coordinate its rendered cross-section grows from a
//! point, peaks at radius `r` when `w₀ = c.w`, and shrinks back to
//! a point. That growth-and-shrink is the visible signature of "we
//! are slicing a 4D object through three dimensions."
//!
//! Visually this is simpler than the pentatope viewer (a 4-ball has
//! no rotation degrees of freedom worth distinguishing, its
//! cross-section is always a 3-ball) but it pins down the basic
//! 4D-physics-with-3D-rendering pipeline cleanly: drop bodies, watch
//! them collide and settle, scrub `w` to confirm they're really 4-
//! dimensional.
//!
//! ## CLI
//!
//! - `-n N` / `--count N`: spawn N hyperspheres (default 1, max 32).
//!   For `N > 1` the bodies are placed in a small lattice with
//!   staggered `y` and `w` so they fall, collide with each other, and
//!   settle as a 4D pile on the floor.
//!
//! ## Viewing modes
//!
//! - **Slice mode** (default), render the 3D cross-section of the
//!   bodies at the current `w₀`. Scrub `w₀` with ↑/↓ to peer at
//!   different cells; bodies outside the slice plane vanish.
//! - **Ghost mode** (toggle with **G**), render the bodies' full 4D
//!   extent simultaneously as a translucent volume.
//!
//! ## Controls
//!
//! - Mouse: orbit camera (left-drag), zoom (scroll).
//! - **Space**: pause / resume physics.
//! - **↑ / ↓**: hold to slide the cross-section along `w` (slice mode
//!   only; 0.6 u/s, range ±1.5 around the first body's `w`).
//! - **A**: toggle automatic offset sweep (slice mode only).
//! - **G**: toggle ghost / slice viewing mode.
//! - **R**: reset, re-spawn all bodies, offset = 0.
//! - **Esc**: exit.

use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use loam_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx, TickCtx};
use loam_camera::OrbitCamera;
use loam_math::{EuclideanR3, EuclideanR4};
use loam_physics::{
    euclidean_r4::{halfspace4_body_r4, register_default_narrowphase, sphere_body_r4},
    field::Gravity,
    World,
};
use loam_render::{device::RenderDevice, graph::RenderNode};
use loam_shader::ShaderId;
use wgpu::{util::DeviceExt, *};
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/hypersphere")
}
fn shader_path() -> PathBuf {
    shader_dir().join("hypersphere.wgsl")
}

const TITLE: &str = "Loam - hypersphere w-slice (live)";
const RADIUS_4D: f32 = 1.0;
/// Offset range: ±1.5 covers the full ball (radius 1) plus margin so
/// the user can scrub past the poles and watch the cross-section
/// vanish. For multi-body mode we also extend it once the body pile
/// spreads in `w`, but the default ±1.5 already covers a couple of
/// adjacent bodies' worth of `w`-spread.
const W_OFFSET_RANGE: f32 = 1.5;
const W_SWEEP_RATE: f32 = 0.6;
/// Maximum number of hyperspheres rendered simultaneously. Hard-capped
/// here to keep the uniform layout fixed-size (`array<vec4, MAX_BODIES>`
/// in WGSL) and avoid runtime resource recreation.
const MAX_BODIES: usize = 32;

#[derive(Debug)]
struct Args {
    count: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args { count: 1 };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-n" | "--count" => {
                    args.count = iter
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(args.count)
                        .clamp(1, MAX_BODIES);
                }
                "-h" | "--help" => {
                    println!("loam hypersphere w-slice viewer (legacy bespoke-shader path)");
                    println!();
                    println!("Usage: cargo run --example hypersphere -- --legacy [options]");
                    println!();
                    println!("Options:");
                    println!(
                        "  -n, --count N    spawn N hyperspheres (default 1, max {MAX_BODIES})"
                    );
                    println!("  -h, --help       show this help and exit");
                    std::process::exit(0);
                }
                _ => {
                    // Unknown flags are silently ignored: the parent
                    // `main` may pass framework flags this parser
                    // doesn't recognise (e.g. `--legacy` itself).
                }
            }
        }
        args
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SliceUniforms {
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_forward: [f32; 3],
    _pad1: f32,
    camera_right: [f32; 3],
    _pad2: f32,
    camera_up: [f32; 3],
    fov_y_tan: f32,
    resolution: [f32; 2],
    time: f32,
    tick: f32,
    /// `[w_slice, radius4, body_count_f, ghost_flag]`. `body_count_f`
    /// is the active entry count, encoded as `f32` for uniform-buffer
    /// simplicity (cast to `u32` in WGSL). `ghost_flag` is 0.0 for
    /// slice mode, 1.0 for ghost (volumetric) mode.
    params: [f32; 4],
    /// Up to `MAX_BODIES` body 4-positions (`vec4(x, y, z, w)` each).
    /// Slots `>= body_count` are unread by the shader.
    bodies: [[f32; 4]; MAX_BODIES],
}

impl Default for SliceUniforms {
    fn default() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 5.0],
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
            bodies: [[0.0; 4]; MAX_BODIES],
        }
    }
}

struct SliceNode {
    pipeline: RenderPipeline,
    uniforms: SliceUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
}

impl SliceNode {
    fn new(device: &Device, surface_format: TextureFormat, shader: &ShaderModule) -> Self {
        let uniforms = SliceUniforms::default();
        let uniform_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("hypersphere uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("hypersphere bgl"),
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
            label: Some("hypersphere bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hypersphere pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("hypersphere pipeline"),
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
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            uniforms,
            uniform_buf,
            bind_group,
        }
    }

    fn set_uniforms(&mut self, queue: &Queue, uniforms: SliceUniforms) {
        self.uniforms = uniforms;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    fn execute_frame(&self, rd: &RenderDevice, view: &TextureView) -> Result<()> {
        let mut encoder = rd.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hypersphere encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("hypersphere pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
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
}

impl RenderNode for SliceNode {
    fn name(&self) -> &'static str {
        "hypersphere"
    }
    fn execute(&mut self, rd: &RenderDevice, view: &TextureView) -> Result<()> {
        self.execute_frame(rd, view)
    }
}

/// Spawn-position generator for body `i` of `count`. See module doc.
fn spawn_position(i: usize, count: usize) -> Vec4 {
    if count <= 1 {
        return Vec4::new(0.0, 2.5, 0.0, 0.0);
    }
    let r_xz = 1.2 * RADIUS_4D;
    let phi = (i as f32 / count as f32) * std::f32::consts::TAU;
    let x = r_xz * phi.cos();
    let z = r_xz * phi.sin();
    let y = 2.5 + 2.5 * RADIUS_4D * i as f32;
    let w = 0.6 * RADIUS_4D * (i as f32 - 0.5 * (count - 1) as f32) / count.max(1) as f32;
    Vec4::new(x, y, z, w)
}

fn build_world(count: usize) -> (World<EuclideanR4>, Vec<usize>) {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
    let floor_id = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    world.bodies[floor_id].restitution = 0.0;
    let mut ball_ids = Vec::with_capacity(count);
    for i in 0..count {
        let id = world.push_body(sphere_body_r4(
            spawn_position(i, count),
            Vec4::ZERO,
            RADIUS_4D,
            1.0,
        ));
        world.bodies[id].restitution = 0.0;
        ball_ids.push(id);
    }
    (world, ball_ids)
}

struct LegacyHypersphereApp {
    /// `App::Space` requirement; the bespoke shader doesn't go
    /// through ShaderDb's prelude path, so the field is dormant.
    space: EuclideanR3,
    shader_id: ShaderId,
    shader_gen: u64,
    node: SliceNode,
    camera: OrbitCamera,

    world: World<EuclideanR4>,
    ball_ids: Vec<usize>,
    paused: bool,

    w_offset: f32,
    auto_sweep: bool,
    sweep_anchor: Instant,
    slider_up_held: bool,
    slider_down_held: bool,
    /// Ghost-mode flag: render the bodies' full 4D extent as a
    /// translucent volume rather than a single `w`-slice.
    ghost_mode: bool,

    initial_count: usize,
}

impl LegacyHypersphereApp {
    fn reset(&mut self) {
        let count = self.ball_ids.len();
        let (world, ball_ids) = build_world(count);
        self.world = world;
        self.ball_ids = ball_ids;
        self.w_offset = 0.0;
        self.auto_sweep = false;
        self.sweep_anchor = Instant::now();
        self.paused = false;
    }

    fn advance_auto_sweep(&mut self) {
        if !self.auto_sweep {
            return;
        }
        let phase = (self.sweep_anchor.elapsed().as_secs_f32() / 8.0) * std::f32::consts::TAU;
        self.w_offset = W_OFFSET_RANGE * phase.cos();
    }

    fn advance_slider(&mut self, dt: f32) {
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            self.w_offset =
                (self.w_offset + dir * W_SWEEP_RATE * dt).clamp(-W_OFFSET_RANGE, W_OFFSET_RANGE);
        }
    }

    fn handle_keyboard(&mut self, code: PhysicalKey, state: ElementState) {
        let PhysicalKey::Code(kc) = code else {
            return;
        };
        let pressed = state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => {
                if pressed && !self.slider_up_held {
                    self.auto_sweep = false;
                }
                self.slider_up_held = pressed;
                return;
            }
            KeyCode::ArrowDown => {
                if pressed && !self.slider_down_held {
                    self.auto_sweep = false;
                }
                self.slider_down_held = pressed;
                return;
            }
            _ => {}
        }
        if !pressed {
            return;
        }
        match kc {
            KeyCode::KeyA => {
                self.auto_sweep = !self.auto_sweep;
                if self.auto_sweep {
                    self.sweep_anchor = Instant::now();
                }
            }
            KeyCode::KeyR => self.reset(),
            KeyCode::KeyG => self.ghost_mode = !self.ghost_mode,
            KeyCode::Space => self.paused = !self.paused,
            _ => {}
        }
    }

    /// Reference body for the slice plane. Always the first
    /// hypersphere; multi-body mode picks the *first* spawned body so
    /// the ↑/↓ slider has a stable anchor (other bodies have varied
    /// `w` offsets, see `spawn_position`).
    fn anchor_body_w(&self) -> f32 {
        self.world.bodies[self.ball_ids[0]].position.w
    }

    fn effective_w_slice(&self) -> f32 {
        self.anchor_body_w() + self.w_offset
    }

    fn body_positions(&self) -> ([[f32; 4]; MAX_BODIES], usize) {
        let mut bodies = [[0.0_f32; 4]; MAX_BODIES];
        let count = self.ball_ids.len().min(MAX_BODIES);
        for (slot, &id) in bodies.iter_mut().zip(self.ball_ids.iter()).take(count) {
            *slot = self.world.bodies[id].position.to_array();
        }
        (bodies, count)
    }
}

impl App for LegacyHypersphereApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let rd = ctx.rd;
        let space = EuclideanR3;
        let shader_id = ctx.shader_db.load(shader_path(), &space)?;
        let shader_gen = ctx.shader_db.generation(shader_id);
        let node = SliceNode::new(
            &rd.device,
            rd.surface_bundle.config.format,
            ctx.shader_db.module(shader_id),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        let args = Args::parse();
        let (world, ball_ids) = build_world(args.count);

        // Pull the camera back proportionally to the body count so the
        // whole stack is in frame at the start.
        let cam_dist = (8.0 + 1.5 * args.count as f32).min(20.0);
        let mut camera = OrbitCamera::default();
        camera.set_orbit(cam_dist, -0.35);

        Ok(Self {
            space,
            shader_id,
            shader_gen,
            node,
            camera,
            world,
            ball_ids,
            paused: false,
            w_offset: 0.0,
            auto_sweep: false,
            sweep_anchor: Instant::now(),
            slider_up_held: false,
            slider_down_held: false,
            ghost_mode: false,
            initial_count: args.count,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn tick(&mut self, dt: f32, _ctx: &mut TickCtx) {
        self.advance_slider(dt);
        self.advance_auto_sweep();
        if !self.paused {
            self.world.step(dt);
        }
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.camera.advance(ctx.input);

        let view = self.camera.view();
        let cfg = &ctx.rd.surface_bundle.config;
        let (bodies, count) = self.body_positions();
        self.node.set_uniforms(
            &ctx.rd.queue,
            SliceUniforms {
                camera_pos: view.position.to_array(),
                _pad0: 0.0,
                camera_forward: view.forward.to_array(),
                _pad1: 0.0,
                camera_right: view.right.to_array(),
                _pad2: 0.0,
                camera_up: view.up.to_array(),
                fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
                resolution: [cfg.width as f32, cfg.height as f32],
                time: ctx.time,
                tick: ctx.tick as f32,
                params: [
                    self.effective_w_slice(),
                    RADIUS_4D,
                    count as f32,
                    if self.ghost_mode { 1.0 } else { 0.0 },
                ],
                bodies,
            },
        );
    }

    fn on_event(&mut self, ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {
        match ev {
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard(event.physical_key, event.state);
            }
            WindowEvent::Focused(false) => {
                self.slider_up_held = false;
                self.slider_down_held = false;
            }
            _ => {}
        }
    }

    fn on_shader_reload(&mut self, ctx: &mut SetupCtx<'_>) {
        let new_gen = ctx.shader_db.generation(self.shader_id);
        if new_gen != self.shader_gen {
            tracing::info!("rebuilding SliceNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.node = SliceNode::new(
                &ctx.rd.device,
                ctx.rd.surface_bundle.config.format,
                ctx.shader_db.module(self.shader_id),
            );
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.node.execute_frame(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        let n = self.initial_count;
        let anchor = &self.world.bodies[self.ball_ids[0]];
        let p = anchor.position;
        let pause = if self.paused { " [paused]" } else { "" };
        let view_mode = if self.ghost_mode { "ghost" } else { "slice" };
        if self.ghost_mode {
            Cow::Owned(format!(
                "{TITLE} | {fps:.0} fps | n={n} | mode={view_mode}{pause} | pos[0].y={:+.2} pos[0].w={:+.2}",
                p.y, p.w
            ))
        } else {
            let mode = if self.auto_sweep { "auto" } else { "manual" };
            let w_eff = p.w + self.w_offset;
            // Visible-radius indicator for the *anchor* body in slice
            // mode. Matches `slice_radius` in the WGSL.
            let dw = w_eff - p.w;
            let r3_sq = RADIUS_4D * RADIUS_4D - dw * dw;
            let r3 = if r3_sq > 0.0 { r3_sq.sqrt() } else { 0.0 };
            Cow::Owned(format!(
                "{TITLE} | {fps:.0} fps | n={n} | mode={view_mode} offset={:+.2} ({mode}) w₀={:+.2}{pause} | r₃[0]={:.3} | pos[0].y={:+.2} pos[0].w={:+.2}",
                self.w_offset, w_eff, r3, p.y, p.w
            ))
        }
    }
}

/// Entry point for the legacy bespoke-shader path. Invoked by the
/// outer `main` when `--legacy` is on the CLI.
///
/// Now runs through `loam-app`'s framework like every other example;
/// the "legacy" name persists because this kept the bespoke shader
/// (volumetric ghost mode) when the parent `hypersphere` migrated
/// to the slim `Hyperslice4DNode` path. Both are kept around so
/// either rendering style is one CLI flag away.
pub fn run() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title(TITLE)
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<LegacyHypersphereApp>(config)
}
