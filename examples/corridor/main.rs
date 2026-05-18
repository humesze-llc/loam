//! Corridor raymarch demo, the README hero image.
//!
//! A rectangular hallway with pillars along both walls, rendered with
//! geodesic ray marching in E³, H³, or S³. Same scene, same shader,
//! different Space prelude. Because the march follows geodesics, the
//! walls read as flat in E³, bowing outward in H³ (parallel geodesics
//! diverge), and converging in S³.
//!
//! Note: `HalfSpace` SDF emission gates on `WgslSpace::is_chart_flat`
//! (rye-scene T1-1). E³ corridor walls render honestly via chart-coord
//! `dot(p, n) - d`; H³ / S³ corridor walls sentinel until closed-form
//! geodesic-plane SDFs land. The pillars (`rye_distance` spheres)
//! tell the curvature story in every Space.
//!
//! Assembles a shader from three layers:
//! - Space prelude from `rye-math` (`rye_distance`, `rye_exp`, ...)
//! - Scene module from the local `scene` submodule (`corridor_demo_wgsl`)
//! - User shader from `examples/corridor/corridor.wgsl`
//!
//! ## Flags
//!
//! `--hyperbolic` : swap Space prelude to HyperbolicH3
//! `--spherical`  : swap Space prelude to SphericalS3
//! `--rotate`     : auto-rotate camera; 1 rev / 20 s

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use rye_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx};
use rye_camera::OrbitCamera;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    graph::RenderNode,
    raymarch::{GeodesicRayMarchNode, RayMarchUniforms},
};
use rye_shader::ShaderId;
use winit::window::WindowAttributes;

mod scene;
use scene::corridor_demo_wgsl;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/corridor")
}

fn shader_path() -> PathBuf {
    shader_dir().join("corridor.wgsl")
}

trait CorridorKnobs: WgslSpace + Default + 'static {
    /// Camera-space-to-Space scale: corridor walls sit at Space
    /// x = ±0.55, y = ±0.40, so camera distance × ball_scale must
    /// also stay inside the chart's valid region (Poincaré ball or
    /// spherical hemisphere). 1.5 × 0.2 = 0.30, safely inside.
    const BALL_SCALE: f32;
    const FOG_SCALE: f32;
    const TITLE: &'static str;
}

impl CorridorKnobs for EuclideanR3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 2.8;
    const TITLE: &'static str = "Rye - Corridor (E³)";
}

impl CorridorKnobs for HyperbolicH3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 2.2;
    const TITLE: &'static str = "Rye - Corridor (H³ geodesics)";
}

impl CorridorKnobs for SphericalS3 {
    const BALL_SCALE: f32 = 0.2;
    const FOG_SCALE: f32 = 2.4;
    const TITLE: &'static str = "Rye - Corridor (S³ geodesics)";
}

/// Initial orbit pose so the camera starts inside the corridor; otherwise
/// the first interactive frame looks at it from outside.
const INITIAL_DISTANCE: f32 = 1.5;
const INITIAL_PITCH: f32 = -0.12;
/// 1 revolution / 20 s at the framework's 60 Hz fixed timestep.
const ROTATE_YAW_PER_TICK: f32 = std::f32::consts::TAU / (60.0 * 20.0);

struct CorridorApp<S: CorridorKnobs> {
    space: S,
    camera: OrbitCamera,
    shader_id: ShaderId,
    shader_gen: u64,
    ray_march: GeodesicRayMarchNode,
    rotate: bool,
}

impl<S: CorridorKnobs> App for CorridorApp<S> {
    type Space = S;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let space = S::default();
        let scene_module = corridor_demo_wgsl();
        let shader_id = ctx
            .shader_db
            .load_geodesic_scene(shader_path(), &scene_module, &space)?;
        let shader_gen = ctx.shader_db.generation(shader_id);
        let ray_march = GeodesicRayMarchNode::from_module(
            &ctx.rd.device,
            ctx.rd.surface_bundle.config.format,
            ctx.shader_db.module(shader_id),
            ctx.rd.sample_count(),
        );

        if let Some(watcher) = ctx.watcher.as_mut() {
            watcher.watch(shader_dir())?;
        }

        let rotate = std::env::args().any(|a| a == "--rotate");

        let mut camera = OrbitCamera::default();
        camera.set_orbit(INITIAL_DISTANCE, INITIAL_PITCH);

        Ok(Self {
            space,
            camera,
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
            self.camera
                .rotate_yaw(ROTATE_YAW_PER_TICK * ctx.n_ticks as f32);
        }
        self.camera.advance(ctx.input);

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
            tracing::info!("rebuilding GeodesicRayMarchNode for shader gen {new_gen}");
            self.shader_gen = new_gen;
            self.ray_march = GeodesicRayMarchNode::from_module(
                &ctx.rd.device,
                ctx.rd.surface_bundle.config.format,
                ctx.shader_db.module(self.shader_id),
                ctx.rd.sample_count(),
            );
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        self.ray_march.execute(rd, view)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        Cow::Owned(format!("{} | {fps:.0} fps", S::TITLE))
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let hyperbolic = args.iter().any(|a| a == "--hyperbolic");
    let spherical = args.iter().any(|a| a == "--spherical");

    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye - Corridor")
            .with_visible(false),
        ..RunConfig::default()
    };

    if hyperbolic {
        run_with_config::<CorridorApp<HyperbolicH3>>(config)
    } else if spherical {
        run_with_config::<CorridorApp<SphericalS3>>(config)
    } else {
        run_with_config::<CorridorApp<EuclideanR3>>(config)
    }
}
