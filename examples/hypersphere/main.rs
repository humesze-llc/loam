//! Hypersphere `w`-slice viewer, drop one or many 4D balls onto
//! a 4D floor, render their 3D cross-sections at user-controlled
//! `w₀`.
//!
//! ## Two implementations in one binary
//!
//! - **Default (no flag): `rye-app` + `Hyperslice4DNode`.** Uses
//!   the new framework: `App` trait, `Camera<S>`, dynamic-body
//!   uniform array. Slice-mode only.
//! - **`--legacy`: hand-rolled custom-shader path.** The
//!   pre-framework implementation in [`legacy`]. Has the full
//!   feature set including ghost mode (volumetric 4D-extent
//!   rendering), per-body color cycling, etc. Stays in the binary
//!   as the "use this when the framework's stock kernel isn't
//!   enough" reference.
//!
//! Both share the shader file at
//! `examples/hypersphere/hypersphere.wgsl` (the legacy path uses
//! it directly; the framework path embeds it indirectly through
//! the `Scene4` + `Hyperslice4DNode` kernel).
//!
//! ## CLI
//!
//! - `-n N` / `--count N`: N hyperspheres (default 1, max 32).
//! - `--legacy`: switch to the bespoke-shader implementation.
//!
//! ## Controls (framework path)
//!
//! - Mouse: orbit camera (left-drag), zoom (scroll).
//! - **Space**: pause / resume physics.
//! - **↑ / ↓**: hold to slide the cross-section along `w` (0.6 u/s).
//! - **A**: toggle automatic offset sweep (cosine-paced).
//! - **R**: reset.
//! - **Esc**: exit.
//!
//! Ghost mode lives only in `--legacy` (volumetric rendering needs
//! a different kernel than the framework's surface ray-march).

mod legacy;

use std::borrow::Cow;

use anyhow::Result;
use glam::{Vec3, Vec4};
use rye_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use rye_math::{EuclideanR3, EuclideanR4};
use rye_physics::{
    euclidean_r4::{halfspace4_body_r4, register_default_narrowphase, sphere_body_r4},
    field::Gravity,
    World,
};
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{
        polytope_stub_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL, MAX_BODIES,
    },
};
use rye_sdf::{Scene4, SceneNode4};
use winit::window::WindowAttributes;

const RADIUS_4D: f32 = 1.0;
/// Offset range relative to body[0]'s `w`, same as legacy.
const W_OFFSET_RANGE: f32 = 1.5;
const W_SWEEP_RATE: f32 = 0.6;

/// 8-color palette (matches legacy's `body_base_color`).
const PALETTE: [[f32; 3]; 8] = [
    [0.95, 0.55, 0.30], // warm orange
    [0.30, 0.55, 0.95], // cool blue
    [0.55, 0.95, 0.40], // green
    [0.95, 0.85, 0.30], // yellow
    [0.85, 0.45, 0.95], // magenta
    [0.30, 0.95, 0.85], // teal
    [0.95, 0.40, 0.55], // pink
    [0.70, 0.70, 0.78], // neutral
];

// ---------------------------------------------------------------------------
// World construction
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// HypersphereApp: the framework path
// ---------------------------------------------------------------------------

struct HypersphereApp {
    space: EuclideanR3,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    node: Hyperslice4DNode,

    world: World<EuclideanR4>,
    ball_ids: Vec<usize>,
    count: usize,
    paused: bool,

    w_offset: f32,
    auto_sweep: bool,
    sweep_anchor: std::time::Instant,
    slider_up_held: bool,
    slider_down_held: bool,
}

impl HypersphereApp {
    fn anchor_w(&self) -> f32 {
        self.world.bodies[self.ball_ids[0]].position.w
    }

    fn effective_w_slice(&self) -> f32 {
        self.anchor_w() + self.w_offset
    }

    fn write_bodies(&mut self) {
        let bodies: Vec<BodyUniform> = self
            .ball_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let p = self.world.bodies[id].position;
                BodyUniform::sphere([p.x, p.y, p.z, p.w], RADIUS_4D, PALETTE[i % PALETTE.len()])
            })
            .collect();
        self.node.set_bodies(&bodies);
    }
}

impl App for HypersphereApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let count = parse_count();

        // Build the static scene: just the 4D floor (`y >= 0`).
        // Hyperspheres are dynamic bodies, uploaded each frame
        // through `Hyperslice4DNode::set_bodies`.
        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, 0.0));

        // Hypersphere scene has no 120/600-cell bodies, so use the
        // cheap stubs that satisfy the kernel's symbol references
        // without paying the ~24 KB const-array compile cost.
        let shader_source = format!(
            "{kernel}\n{polytope}\n{scene}\n",
            kernel = HYPERSLICE_KERNEL_WGSL,
            polytope = polytope_stub_sdfs_wgsl(),
            scene = scene.to_hyperslice_wgsl("u.w_slice"),
        );
        let module = ctx
            .rd
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hypersphere shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let node = Hyperslice4DNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            &module,
            ctx.rd.sample_count(),
        );

        let (world, ball_ids) = build_world(count);
        Ok(Self {
            space: EuclideanR3,
            camera: {
                let mut c = Camera::<EuclideanR3>::at_origin();
                c.position = Vec3::new(0.0, 4.0, 8.0);
                c
            },
            orbit: {
                let mut o: OrbitController<EuclideanR3> = OrbitController::default();
                o.set_orbit((8.0 + 1.5 * count as f32).min(20.0), -0.35);
                o
            },
            node,
            world,
            ball_ids,
            count,
            paused: false,
            w_offset: 0.0,
            auto_sweep: false,
            sweep_anchor: std::time::Instant::now(),
            slider_up_held: false,
            slider_down_held: false,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn tick(&mut self, dt: f32, _ctx: &mut rye_app::TickCtx) {
        if !self.paused {
            self.world.step(dt);
        }
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Slider advance.
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        if dir != 0.0 {
            let dt_secs = ctx.n_ticks as f32 / 60.0;
            self.w_offset = (self.w_offset + dir * W_SWEEP_RATE * dt_secs)
                .clamp(-W_OFFSET_RANGE, W_OFFSET_RANGE);
        }
        if self.auto_sweep {
            let phase = (self.sweep_anchor.elapsed().as_secs_f32() / 8.0) * std::f32::consts::TAU;
            self.w_offset = W_OFFSET_RANGE * phase.cos();
        }

        // Camera.
        use rye_camera::CameraController;
        self.orbit
            .advance(ctx.input, &mut self.camera, &EuclideanR3, 0.0);
        let view = self.camera.view();

        // Body uniforms, every sphere's current 4D position.
        self.write_bodies();

        // Camera + slice-w + resolution into the node uniform.
        let cfg = &ctx.rd.surface_bundle.config;
        let w_slice = self.effective_w_slice();
        {
            let u = self.node.uniforms_mut();
            u.camera_pos = view.position.to_array();
            u.camera_forward = view.forward.to_array();
            u.camera_right = view.right.to_array();
            u.camera_up = view.up.to_array();
            u.fov_y_tan = (60.0_f32.to_radians() * 0.5).tan();
            u.resolution = [cfg.width as f32, cfg.height as f32];
            u.time = ctx.time;
            u.tick = ctx.tick as f32;
            u.w_slice = w_slice;
        }
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    fn on_event(&mut self, ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
        use winit::event::{ElementState, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        let WindowEvent::KeyboardInput { event, .. } = ev else {
            return;
        };
        let PhysicalKey::Code(kc) = event.physical_key else {
            return;
        };
        let pressed = event.state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => {
                if pressed {
                    self.auto_sweep = false;
                }
                self.slider_up_held = pressed;
            }
            KeyCode::ArrowDown => {
                if pressed {
                    self.auto_sweep = false;
                }
                self.slider_down_held = pressed;
            }
            _ if !pressed => {}
            KeyCode::KeyA => {
                self.auto_sweep = !self.auto_sweep;
                if self.auto_sweep {
                    self.sweep_anchor = std::time::Instant::now();
                }
            }
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::KeyR => {
                let (world, ball_ids) = build_world(self.count);
                self.world = world;
                self.ball_ids = ball_ids;
                self.w_offset = 0.0;
                self.auto_sweep = false;
                self.paused = false;
            }
            _ => {}
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.node.execute(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        let n = self.ball_ids.len();
        let p = self.world.bodies[self.ball_ids[0]].position;
        let pause = if self.paused { " [paused]" } else { "" };
        let mode = if self.auto_sweep { "auto" } else { "manual" };
        let w_eff = p.w + self.w_offset;
        Cow::Owned(format!(
            "Rye - hypersphere (rye-app) | {fps:.0} fps | n={n} | offset={:+.2} ({mode}) w₀={:+.2}{pause} | pos[0].y={:+.2} pos[0].w={:+.2}",
            self.w_offset, w_eff, p.y, p.w
        ))
    }
}

fn parse_count() -> usize {
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        if arg == "-n" || arg == "--count" {
            if let Some(n) = iter.next().and_then(|s| s.parse::<usize>().ok()) {
                return n.clamp(1, MAX_BODIES);
            }
        }
    }
    1
}

// ---------------------------------------------------------------------------
// Entry point, dispatch on --legacy
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let use_legacy = std::env::args().any(|a| a == "--legacy");
    if use_legacy {
        return legacy::run();
    }
    let count = parse_count();
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye - hypersphere (rye-app)")
            .with_visible(false),
        ..RunConfig::default()
    };
    let _ = count;
    run_with_config::<HypersphereApp>(config)
}
