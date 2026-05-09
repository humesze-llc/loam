//! Pentatope `w`-slice viewer, live physics edition.
//!
//! A 4D pentatope falls under gravity onto a 4D `y = 0` floor; the
//! 3D viewport renders the cross-section of the whole 4D scene at
//! `w = w₀`. Sweep `w₀` to peer through the pentatope's interior;
//! the cross-section shape morphs as different cells of the 4D body
//! pass through the slicing hyperplane.
//!
//! The 4D floor is a half-space whose normal lives entirely in the
//! `xyz` directions (`w` component zero), so its cross-section at any
//! `w₀` is the same 3D plane `y = 0`. The pentatope's vertices, by
//! contrast, span all four dimensions, so its slice depends strongly
//! on `w₀`.
//!
//! ## w-slice convention
//!
//! `w_slice` (the slice hyperplane's `w` coordinate) **tracks the
//! body's current w-position** plus a user-controlled offset. The
//! pentatope's vertices span ≈ `[body.w − 0.25, body.w + 1.0]`, so
//! offset 0 always cuts through the pentatope; the user nudges the
//! offset to peer at different cross-sections without losing the
//! body off-screen as it drifts in 4D.
//!
//! ## Controls
//!
//! - Mouse: orbit camera (left-drag), zoom (scroll).
//! - **Space**: pause / resume physics.
//! - **↑ / ↓**: hold to slide the cross-section through `w`
//!   continuously (0.6 units/s, range ±1.5 around the body's
//!   `w`-position, slide far enough either way and the body
//!   leaves the slice plane entirely). Disables auto-sweep.
//! - **A**: toggle automatic offset sweep (cosine-paced, 8 s period,
//!   range ±1.5).
//! - **R**: reset, re-spawn the pentatope at `y = 2.5`, offset = 0.
//! - **0–4**: highlight that pentatope cell (its tinted faces glow
//!   brighter when visible). **5**: clear highlight.
//! - **Esc**: exit.

use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use rye_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx, TickCtx};
use rye_camera::OrbitCamera;
use rye_math::{EuclideanR3, EuclideanR4, Rotor};
use rye_physics::{
    euclidean_r4::{
        halfspace4_body_r4, pentatope_vertices, polytope_body_r4, register_default_narrowphase,
    },
    field::Gravity,
    World,
};
use rye_render::{device::RenderDevice, graph::RenderNode};
use rye_shader::ShaderId;
use wgpu::{util::DeviceExt, *};
use winit::{
    event::{ElementState, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/pentatope_slice")
}
fn shader_path() -> PathBuf {
    shader_dir().join("pentatope_slice.wgsl")
}

const TITLE: &str = "Rye - pentatope w-slice (live)";
const PENTATOPE_RADIUS: f32 = 1.0;
/// Offset range relative to the body's current `w`-position. The
/// pentatope's local vertices span ≈ `w ∈ [−0.25, +1.0]`, so a range
/// of ±1.5 lets the user slide the slice plane *entirely past* the
/// body in either direction (cross-section vanishes), then back.
const W_OFFSET_RANGE: f32 = 1.5;
/// Keyboard slider speed: held ↑/↓ moves `w_offset` at this rate
/// (units of w per second).
const W_SWEEP_RATE: f32 = 0.6;
const NO_HIGHLIGHT: f32 = 5.0;

// ---- Custom render uniform -----------------------------------------
//
// Bigger than `RayMarchUniforms` because we ship the pentatope's
// world-space vertices (5 × `vec4<f32>`) every frame. Layout is
// std140-compatible: `vec3` slots padded to 16 bytes; the trailing
// `vec4` array packs naturally.

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
    /// `[w_slice, highlight, _, _]`.
    params: [f32; 4],
    /// Pentatope vertices in world coordinates, after the body's
    /// orientation rotor and position translation.
    pentatope_v: [[f32; 4]; 5],
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
            pentatope_v: [[0.0; 4]; 5],
        }
    }
}

/// Custom render node: same fullscreen-triangle pipeline as
/// `RayMarchNode`, but with a wider uniform layout to carry the
/// pentatope vertex data alongside the camera.
struct SliceNode {
    pipeline: RenderPipeline,
    uniforms: SliceUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
}

impl SliceNode {
    fn new(
        device: &Device,
        surface_format: TextureFormat,
        shader: &ShaderModule,
        sample_count: u32,
    ) -> Self {
        let uniforms = SliceUniforms::default();
        let uniform_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("pentatope_slice uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pentatope_slice bgl"),
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
            label: Some("pentatope_slice bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pentatope_slice pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pentatope_slice pipeline"),
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
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
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
            label: Some("pentatope_slice encoder"),
        });
        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("pentatope_slice pass"),
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
        "pentatope_slice"
    }
    fn execute(&mut self, rd: &RenderDevice, view: &TextureView) -> Result<()> {
        self.execute_frame(rd, view)
    }
}

/// Build the demo world: a 4D `y ≥ 0` floor (id 0) and a pentatope
/// dropped from `y = 2.5` with zero restitution so it settles cleanly
/// instead of bouncing chaotically. Floor is id 0; pentatope is the
/// last (and only other) body, so callers should grab
/// `world.bodies.len() - 1`.
fn build_world() -> World<EuclideanR4> {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, -9.8, 0.0, 0.0))));
    let floor_id = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0));
    world.bodies[floor_id].restitution = 0.0;
    let pent_id = world.push_body(polytope_body_r4(
        Vec4::new(0.0, 2.5, 0.0, 0.0),
        Vec4::ZERO,
        pentatope_vertices(PENTATOPE_RADIUS),
        1.0,
    ));
    // Zero restitution so the body lands and rests instead of
    // bouncing chaotically and tumbling its `w`-coordinate out of
    // view.
    world.bodies[pent_id].restitution = 0.0;
    world
}

struct PentatopeApp {
    /// `App::Space` requirement. The shader is loaded as a non-Space
    /// scene (no Space prelude needed for this static 4D math), so
    /// the field is dormant.
    space: EuclideanR3,
    shader_id: ShaderId,
    shader_gen: u64,
    node: SliceNode,
    camera: OrbitCamera,

    // Physics.
    world: World<EuclideanR4>,
    pentatope_id: usize,
    paused: bool,

    // Slice. `w_slice` is computed each frame as
    // `body.position.w + w_offset` so the slice plane tracks the body
    // through 4D, without this the body drifts in `w` post-impact and
    // the cross-section disappears off-screen within a second.
    w_offset: f32,
    auto_sweep: bool,
    sweep_anchor: Instant,
    highlight: f32,
    /// Held-key state for the ↑ / ↓ continuous slider.
    slider_up_held: bool,
    slider_down_held: bool,
}

impl PentatopeApp {
    fn reset(&mut self) {
        self.world = build_world();
        self.pentatope_id = self.world.bodies.len() - 1;
        self.w_offset = 0.0;
        self.auto_sweep = false;
        self.sweep_anchor = Instant::now();
        self.paused = false;
    }

    fn advance_auto_sweep(&mut self) {
        if !self.auto_sweep {
            return;
        }
        // 8 s period, ±W_OFFSET_RANGE about 0 (cosine).
        let phase = (self.sweep_anchor.elapsed().as_secs_f32() / 8.0) * std::f32::consts::TAU;
        self.w_offset = W_OFFSET_RANGE * phase.cos();
    }

    /// Advance the held-key slider by `dt` seconds.
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
        // ↑ / ↓ behave as a held slider, track press/release so
        // `advance_slider` can move `w_offset` continuously per tick.
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
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::Digit0 => self.highlight = 0.0,
            KeyCode::Digit1 => self.highlight = 1.0,
            KeyCode::Digit2 => self.highlight = 2.0,
            KeyCode::Digit3 => self.highlight = 3.0,
            KeyCode::Digit4 => self.highlight = 4.0,
            KeyCode::Digit5 => self.highlight = NO_HIGHLIGHT,
            _ => {}
        }
    }

    /// World-space pentatope vertices: each body-local vertex
    /// rotated by the body's orientation rotor and translated by its
    /// position. Sent to the shader every frame.
    fn pentatope_world_vertices(&self) -> [[f32; 4]; 5] {
        let body = &self.world.bodies[self.pentatope_id];
        let local = pentatope_vertices(PENTATOPE_RADIUS);
        let mut out = [[0.0_f32; 4]; 5];
        for i in 0..5 {
            let v_world = body.orientation.rotation.apply(local[i]) + body.position;
            out[i] = v_world.to_array();
        }
        out
    }

    /// Effective slice plane: body's current `w` plus the user offset.
    fn effective_w_slice(&self) -> f32 {
        self.world.bodies[self.pentatope_id].position.w + self.w_offset
    }
}

impl App for PentatopeApp {
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
            rd.sample_count(),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        // Pull the camera up and back so the pentatope falling onto
        // `y = 0` is framed from above. `OrbitCamera` uses negative
        // pitch for "look down" (positive pitch tilts below the floor;
        // the SDF then reads the origin as inside the ground
        // half-space and the viewport floods with checker).
        let mut camera = OrbitCamera::default();
        camera.set_orbit(8.0, -0.35);

        let world = build_world();
        let pentatope_id = world.bodies.len() - 1;

        Ok(Self {
            space,
            shader_id,
            shader_gen,
            node,
            camera,
            world,
            pentatope_id,
            paused: false,
            w_offset: 0.0,
            auto_sweep: false,
            sweep_anchor: Instant::now(),
            highlight: NO_HIGHLIGHT,
            slider_up_held: false,
            slider_down_held: false,
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

        let cfg = &ctx.rd.surface_bundle.config;
        self.node.set_uniforms(
            &ctx.rd.queue,
            SliceUniforms {
                camera_pos: self.camera.view().position.to_array(),
                _pad0: 0.0,
                camera_forward: self.camera.view().forward.to_array(),
                _pad1: 0.0,
                camera_right: self.camera.view().right.to_array(),
                _pad2: 0.0,
                camera_up: self.camera.view().up.to_array(),
                fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
                resolution: [cfg.width as f32, cfg.height as f32],
                time: ctx.time,
                tick: ctx.tick as f32,
                params: [self.effective_w_slice(), self.highlight, 0.0, 0.0],
                pentatope_v: self.pentatope_world_vertices(),
            },
        );
    }

    fn on_event(&mut self, ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {
        match ev {
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard(event.physical_key, event.state);
            }
            WindowEvent::Focused(false) => {
                // Drop held-slider state so the offset doesn't keep
                // sweeping while the window is unfocused.
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
                ctx.rd.sample_count(),
            );
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.node.execute_frame(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        let body = &self.world.bodies[self.pentatope_id];
        let p = body.position;
        let pause = if self.paused { " [paused]" } else { "" };
        let mode = if self.auto_sweep { "auto" } else { "manual" };
        let w_eff = p.w + self.w_offset;
        Cow::Owned(format!(
            "{TITLE} | {fps:.0} fps | offset={:+.2} ({mode}) w₀={:+.2}{pause} | pos.y={:+.2} pos.w={:+.2}",
            self.w_offset, w_eff, p.y, p.w
        ))
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title(TITLE)
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<PentatopeApp>(config)
}
