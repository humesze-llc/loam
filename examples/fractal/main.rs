//! Loam's first graphics example: a live Mandelbulb raymarcher.
//!
//! Edit `examples/fractal/fractal.wgsl` while the example runs and the
//! scene recompiles on save.
//!
//! ## Flags
//!
//! `--hyperbolic` : swap Space prelude to HyperbolicH3 (geodesic fog)
//! `--spherical`  : swap Space prelude to SphericalS3
//! `--rotate`     : auto-rotate camera; 1 rev / 20 s

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use loam_app::{run_with_config, App, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use loam_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use loam_render::{
    device::RenderDevice,
    raymarch::{RayMarchNode, RayMarchUniforms},
};
use loam_shader::ShaderId;
use winit::window::WindowAttributes;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fractal")
}
fn shader_path() -> PathBuf {
    shader_dir().join("fractal.wgsl")
}

/// Per-Space tuning constants for the *shader prelude*. The Space
/// only affects the WGSL prelude (geodesic fog metric); the camera
/// is always Euclidean. See the crate-level docstring of
/// `examples/fractal_app` for the rationale (preserved here verbatim
/// after the merge).
trait FractalKnobs: WgslSpace + Default + 'static {
    const BALL_SCALE: f32;
    const FOG_SCALE: f32;
    const TITLE: &'static str;
}

impl FractalKnobs for EuclideanR3 {
    const BALL_SCALE: f32 = 1.0;
    const FOG_SCALE: f32 = 12.0;
    const TITLE: &'static str = "Loam - Mandelbulb";
}
impl FractalKnobs for HyperbolicH3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 4.0;
    const TITLE: &'static str = "Loam - Mandelbulb (H³ fog)";
}
impl FractalKnobs for SphericalS3 {
    const BALL_SCALE: f32 = 0.15;
    const FOG_SCALE: f32 = 2.5;
    const TITLE: &'static str = "Loam - Mandelbulb (S³ fog)";
}

struct FractalApp<S: FractalKnobs> {
    /// Drives the WGSL prelude (geodesic fog metric for H³ / S³;
    /// no-op for E³). Independent of the camera's own geometry.
    space: S,
    /// Camera lives in Euclidean 3-space regardless of `S`. The
    /// shader interprets `camera_pos` as a Poincaré-ball or
    /// 3-sphere point internally for the fog metric, but the
    /// orbit math itself is flat. See the crate-level note on
    /// `examples/fractal_app` for the H³-camera footgun.
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    shader_id: ShaderId,
    shader_gen: u64,
    ray_march: RayMarchNode,
    rotate: bool,
}

/// Yaw advance per tick when `rotate` is on. 1 revolution / 20 s
/// at the framework's 60 Hz fixed timestep.
const ROTATE_YAW_PER_TICK: f32 = std::f32::consts::TAU / (60.0 * 20.0);

impl<S: FractalKnobs> App for FractalApp<S> {
    type Space = S;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let space = S::default();

        let shader_id = ctx.shader_db.load(shader_path(), &space)?;
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

        let rotate = std::env::args().any(|a| a == "--rotate");

        Ok(Self {
            space,
            camera: Camera::<EuclideanR3>::at_origin(),
            orbit: OrbitController::default(),
            shader_id,
            shader_gen,
            ray_march,
            rotate,
        })
    }

    fn space(&self) -> &S {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.rotate {
            self.orbit
                .rotate_yaw(ROTATE_YAW_PER_TICK * ctx.n_ticks as f32);
        }
        // Camera always orbits in Euclidean space, even when the
        // shader prelude is H³ / S³.
        self.orbit
            .advance_with_input(ctx.input, &mut self.camera, &EuclideanR3);

        let view = self.camera.view();
        let cfg = &ctx.rd.surface_bundle.config;
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
                params: [0.0, S::BALL_SCALE, S::FOG_SCALE, 0.0],
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
        use loam_render::graph::RenderNode;
        self.ray_march.execute(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        let p = self.camera.view().position;
        Cow::Owned(format!(
            "{} | {fps:.0} fps | pos ({:.2}, {:.2}, {:.2})",
            S::TITLE,
            p.x,
            p.y,
            p.z
        ))
    }
}

/// Thin wrapper so the `update` call site reads cleanly without
/// dragging in `CameraController` and a meaningless `dt`.
trait OrbitInputExt {
    fn advance_with_input(
        &mut self,
        input: loam_input::FrameInput,
        camera: &mut Camera<EuclideanR3>,
        space: &EuclideanR3,
    );
}

impl OrbitInputExt for OrbitController<EuclideanR3> {
    fn advance_with_input(
        &mut self,
        input: loam_input::FrameInput,
        camera: &mut Camera<EuclideanR3>,
        space: &EuclideanR3,
    ) {
        use loam_camera::CameraController;
        self.advance(input, camera, space, 0.0);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let hyperbolic = args.iter().any(|a| a == "--hyperbolic");
    let spherical = args.iter().any(|a| a == "--spherical");

    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Loam - Mandelbulb")
            .with_visible(false),
        ..RunConfig::default()
    };

    if hyperbolic {
        run_with_config::<FractalApp<HyperbolicH3>>(config)
    } else if spherical {
        run_with_config::<FractalApp<SphericalS3>>(config)
    } else {
        run_with_config::<FractalApp<EuclideanR3>>(config)
    }
}
