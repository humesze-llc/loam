//! `blended`: BlendedSpace milestone demo.
//!
//! Renders a row of spheres above a checkerboard floor under
//! `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`. The X
//! axis carries a smooth metric transition: pure Euclidean for
//! `x ≤ −0.3`, pure Poincaré-H³ for `x ≥ +0.3`, smootherstep blend
//! between. The H³ chart only exists inside `|p| < 1`, so the
//! pure-H³ region of the scene fits inside that ball.
//!
//! ## What you should see
//!
//! - **Left third of view (x < −0.3):** flat E³ region. Spheres are
//!   round, floor checker is regular, light rays straight.
//! - **Middle third (−0.3 ≤ x ≤ +0.3):** transition zone. Sphere
//!   silhouettes distort smoothly; checker cells warp.
//! - **Right third (x > +0.3, inside `|p| < 1`):** pure H³. The
//!   floor at `y = 0` becomes a hyperbolic plane, appears as a
//!   spherical cap in Poincaré chart coordinates. Spheres visibly
//!   compress toward the ball boundary. **This is correct.**
//!
//! ## Controls
//!
//! - **WASD**: move forward/back/strafe along the camera basis.
//! - **Space / Shift**: rise / sink along world Y.
//! - **Left-mouse-drag**: look around (yaw + pitch).
//! - **`--floor-only`**: suppress the spheres so only the floor renders.
//! - **Esc**: exit.

use std::path::PathBuf;

use anyhow::Result;
use glam::Vec3;
use rye_app::{run_with_config, App, Camera, FirstPersonController, FrameCtx, RunConfig, SetupCtx};
use rye_camera::CameraController;
use rye_math::{BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX};
use rye_render::{
    device::RenderDevice,
    raymarch::{RayMarchNode, RayMarchUniforms},
};
use rye_shader::ShaderId;
use winit::window::WindowAttributes;

type DemoSpace = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/blended")
}
fn shader_path() -> PathBuf {
    shader_dir().join("blended.wgsl")
}

/// Scene SDF: floor at y=0 plus a row of small spheres at
/// y=0.12 spaced along x ∈ [−0.7, +0.7]. With `ball_scale = 1.0`
/// every coordinate here is also a shader coordinate, so the
/// blending zone in `LinearBlendX` ([−0.3, +0.3]) and the sphere
/// positions agree.
fn scene_module_wgsl() -> &'static str {
    r#"
const SPHERE_COUNT: i32 = 8;
fn sphere_x(i: i32) -> f32 { return -0.7 + 0.2 * f32(i); }
const SPHERE_Y: f32 = 0.12;
const SPHERE_R: f32 = 0.06;

fn rye_scene_sdf(p: vec3<f32>) -> f32 {
    // Floor as a chart-coordinate horizontal plane at y=0. On the
    // pure-H³ side this is a hyperbolic plane and renders as a
    // spherical cap in Poincaré coordinates, that's geometrically
    // correct, not an artifact.
    var d = p.y;

    if u.show_spheres > 0.5 {
        for (var i: i32 = 0; i < SPHERE_COUNT; i = i + 1) {
            let center = vec3<f32>(sphere_x(i), SPHERE_Y, 0.0);
            let s = length(p - center) - SPHERE_R;
            d = min(d, s);
        }
    }

    return d;
}
"#
}

struct BlendedApp {
    space: DemoSpace,
    camera: Camera<EuclideanR3>,
    look: FirstPersonController<EuclideanR3>,
    shader_id: ShaderId,
    shader_gen: u64,
    ray_march: RayMarchNode,
    show_spheres: f32,
}

impl BlendedApp {
    /// World-units per second of WASD/Space/Shift movement.
    /// Scene spans roughly ±0.7 along X; this lets the user fly
    /// the full width in ~3 seconds.
    const MOVE_SPEED: f32 = 0.5;

    /// Effective dt for the camera-position update. The framework
    /// runs `update` once per frame after all the frame's ticks;
    /// we use `n_ticks / fixed_hz` so movement is independent of
    /// frame rate.
    fn dt(ctx: &FrameCtx<'_>, fixed_hz: u32) -> f32 {
        ctx.n_ticks as f32 / fixed_hz as f32
    }
}

impl App for BlendedApp {
    type Space = DemoSpace;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        // Zone width 0.6 centred on origin. Pure E³ at x ≤ −0.3,
        // pure H³ at x ≥ +0.3 inside the Poincaré ball.
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.3, 0.3));

        let shader_id =
            ctx.shader_db
                .load_geodesic_scene(shader_path(), scene_module_wgsl(), &space)?;
        let shader_gen = ctx.shader_db.generation(shader_id);
        let ray_march = RayMarchNode::new(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.shader_db.module(shader_id),
            ctx.rd.sample_count(),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        // Start outside the unit ball, off to the E³ side so the
        // first frame shows clean flat geometry on the left and
        // the curved H³ region on the right simultaneously.
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(-1.2, 0.4, 1.5);
        let look = FirstPersonController::new(-0.50, -0.20);
        // Re-derive the basis from the initial yaw/pitch.
        let mut look = look;
        look.advance(
            rye_input::FrameInput::default(),
            &mut camera,
            &EuclideanR3,
            0.0,
        );

        let show_spheres = if std::env::args().any(|a| a == "--floor-only") {
            0.0
        } else {
            1.0
        };

        Ok(Self {
            space,
            camera,
            look,
            shader_id,
            shader_gen,
            ray_march,
            show_spheres,
        })
    }

    fn space(&self) -> &Self::Space {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Look only while left mouse is dragged, otherwise the
        // cursor escaping the window would flick the camera.
        let look_input = if ctx.input.left_mouse_down {
            ctx.input
        } else {
            rye_input::FrameInput {
                mouse_delta: glam::Vec2::ZERO,
                ..ctx.input
            }
        };
        self.look
            .advance(look_input, &mut self.camera, &EuclideanR3, 0.0);

        // Position update along the camera basis. World-Y for
        // Space/Shift so vertical motion stays intuitive even when
        // the camera is pitched.
        let dt = Self::dt(ctx, 60);
        let move_world = self.camera.right * ctx.input.move_right
            + Vec3::Y * ctx.input.move_up
            + self.camera.forward * ctx.input.move_forward;
        if move_world.length_squared() > 0.0 {
            self.camera.position += move_world.normalize() * Self::MOVE_SPEED * dt;
        }

        let view = self.camera.view();
        let cfg = &ctx.rd.surface_bundle.config;
        // ball_scale = 1.0 makes shader coords identical to world
        // coords. The blending zone, sphere positions, floor plane,
        // and camera all live in the same coordinate system, so
        // there's no place for shader-vs-world confusion.
        let ball_scale = 1.0;
        let fog_scale = 4.0;
        self.ray_march.set_uniforms(
            &ctx.rd.queue,
            RayMarchUniforms {
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
                params: [0.0, ball_scale, fog_scale, self.show_spheres],
            },
        );
    }

    fn on_shader_reload(&mut self, ctx: &mut SetupCtx<'_>) {
        let new_gen = ctx.shader_db.generation(self.shader_id);
        if new_gen != self.shader_gen {
            tracing::info!("rebuilding RayMarchNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.ray_march = RayMarchNode::new(
                &ctx.rd.device,
                ctx.rd.surface_bundle.config.format,
                ctx.shader_db.module(self.shader_id),
                ctx.rd.sample_count(),
            );
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        use rye_render::graph::RenderNode;
        self.ray_march.execute(rd, view)
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        let p = self.camera.view().position;
        std::borrow::Cow::Owned(format!(
            "Rye - BlendedSpace<E3,H3> | {fps:.0} fps | pos ({:.2}, {:.2}, {:.2})",
            p.x, p.y, p.z
        ))
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye - BlendedSpace<E3,H3>")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<BlendedApp>(config)
}
