//! Geodesic lattice, side-by-side E³ / H³ / S³ comparison demo.
//!
//! Renders three panels into a single window: left E³, centre H³, right S³.
//! The same camera orbits the same lattice of spheres; the only difference
//! is the Space prelude injected at shader compile time. The visual difference
//! (evenly-spaced grid vs. tanh-compressed vs. sin-wrapped) is the engine's
//! geometric thesis made visible.
//!
//! `lattice.wgsl` is compiled three times, once per Space, using
//! `WgslSpace::wgsl_impl()` assembled directly. The framework's
//! `ShaderDb` is single-path-keyed, so the three shader modules are
//! built outside it; hot-reload is disabled here.
//!
//! ## Flags
//!
//! `--rotate` : auto-rotate camera; 1 rev / 20 s

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::Result;
use rye_app::{run_with_config, App, FrameCtx, RunConfig, SetupCtx};
use rye_camera::OrbitCamera;
use rye_math::{EuclideanR3, HyperbolicH3, SphericalS3, WgslSpace};
use rye_render::{
    device::RenderDevice,
    raymarch::{GeodesicRayMarchNode, RayMarchUniforms},
};
use rye_shader::{validate_wgsl, GEODESIC_MARCH_KERNEL};
use winit::window::WindowAttributes;

mod scene;
use scene::LatticeSphereScene;

// BALL_SCALE is baked into lattice.wgsl as a constant; this comment documents
// the value so main.rs and the shader stay in sync. Camera orbit distance ×
// BALL_SCALE must stay < 1.0 for H³/S³ Poincaré ball validity (0.2 × 4.5 = 0.9).

// Fog scale per space (tuned for visual clarity).
const FOG_E3: f32 = 3.0;
const FOG_H3: f32 = 4.2;
const FOG_S3: f32 = 2.4;

/// 1 revolution / 20 s at the framework's 60 Hz fixed timestep.
const ROTATE_YAW_PER_TICK: f32 = std::f32::consts::TAU / (60.0 * 20.0);

fn shader_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/lattice/lattice.wgsl")
}

/// Assemble Space prelude + scene SDF + geodesic kernel + user shader.
fn assemble(prelude: &str, scene: &str, user: &str) -> String {
    format!(
        "// ---- rye-math Space prelude ----\n{prelude}\n\
         // ---- rye-sdf scene module ----\n{scene}\n\
         // ---- rye geodesic march kernel ----\n{GEODESIC_MARCH_KERNEL}\n\
         // ---- user shading ----\n{user}"
    )
}

struct LatticeApp {
    /// Required by the trait. ShaderDb hot-reload would re-emit
    /// preludes against this single Space, but the demo manually
    /// assembles three modules per Space, so the field is dormant.
    space: EuclideanR3,
    node_e3: GeodesicRayMarchNode,
    node_h3: GeodesicRayMarchNode,
    node_s3: GeodesicRayMarchNode,
    camera: OrbitCamera,
    rotate: bool,
}

struct PanelLayout {
    /// Window pixel width / height (the shader sees the full window
    /// resolution; the panel selects its own slice via `x_offset` /
    /// `panel_w`).
    window_w: u32,
    window_h: u32,
    x_offset: u32,
    panel_w: u32,
    /// Discriminator the user shader uses to pick this panel's
    /// per-Space tinting / labels (0.0 = E³, 1.0 = H³, 2.0 = S³).
    panel_idx: f32,
    fog_scale: f32,
}

impl LatticeApp {
    fn panel_uniforms(&self, time: f32, tick: u64, p: PanelLayout) -> RayMarchUniforms {
        let camera = self.camera.view();
        RayMarchUniforms {
            camera_pos: camera.position.to_array(),
            _pad0: 0.0,
            camera_forward: camera.forward.to_array(),
            _pad1: 0.0,
            camera_right: camera.right.to_array(),
            _pad2: 0.0,
            camera_up: camera.up.to_array(),
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [p.window_w as f32, p.window_h as f32],
            time,
            tick: tick as f32,
            params: [
                p.x_offset as f32,
                p.panel_idx,
                p.panel_w as f32,
                p.fog_scale,
            ],
        }
    }
}

impl App for LatticeApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let rd = ctx.rd;

        // Build lattice scene modules; centers are computed per-Space in Rust.
        let lattice = LatticeSphereScene::default();
        let scene_e3 = lattice.to_wgsl(&EuclideanR3);
        let scene_h3 = lattice.to_wgsl(&HyperbolicH3);
        let scene_s3 = lattice.to_wgsl(&SphericalS3);

        // Read the user WGSL once.
        let user_src = std::fs::read_to_string(shader_path())?;

        // Assemble and compile three distinct shader modules.
        let make_module = |prelude: &str, scene: &str| -> Result<wgpu::ShaderModule> {
            let full = assemble(prelude, scene, &user_src);
            validate_wgsl(&full)?;
            Ok(rd
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lattice"),
                    source: wgpu::ShaderSource::Wgsl(full.into()),
                }))
        };

        let fmt = rd.surface_bundle.config.format;
        let mod_e3 = make_module(&EuclideanR3.wgsl_impl(), &scene_e3)?;
        let mod_h3 = make_module(&HyperbolicH3.wgsl_impl(), &scene_h3)?;
        let mod_s3 = make_module(&SphericalS3.wgsl_impl(), &scene_s3)?;

        let samples = rd.sample_count();
        let node_e3 = GeodesicRayMarchNode::from_module(&rd.device, fmt, &mod_e3, samples);
        let node_h3 = GeodesicRayMarchNode::from_module(&rd.device, fmt, &mod_h3, samples);
        let node_s3 = GeodesicRayMarchNode::from_module(&rd.device, fmt, &mod_s3, samples);

        let rotate = std::env::args().any(|a| a == "--rotate");
        let camera = OrbitCamera::default();

        Ok(Self {
            space: EuclideanR3,
            node_e3,
            node_h3,
            node_s3,
            camera,
            rotate,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &self.space
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.rotate {
            self.camera
                .rotate_yaw(ROTATE_YAW_PER_TICK * ctx.n_ticks as f32);
        }
        self.camera.advance(ctx.input);

        let cfg = &ctx.rd.surface_bundle.config;
        let w = cfg.width;
        let h = cfg.height;
        let pw = w / 3;
        let pw2 = w - pw * 2;

        let panel = |x_offset, panel_w, panel_idx, fog_scale| PanelLayout {
            window_w: w,
            window_h: h,
            x_offset,
            panel_w,
            panel_idx,
            fog_scale,
        };
        let u_e3 = self.panel_uniforms(ctx.time, ctx.tick, panel(0, pw, 0.0, FOG_E3));
        let u_h3 = self.panel_uniforms(ctx.time, ctx.tick, panel(pw, pw, 1.0, FOG_H3));
        let u_s3 = self.panel_uniforms(ctx.time, ctx.tick, panel(pw * 2, pw2, 2.0, FOG_S3));
        self.node_e3.set_uniforms(&ctx.rd.queue, u_e3);
        self.node_h3.set_uniforms(&ctx.rd.queue, u_h3);
        self.node_s3.set_uniforms(&ctx.rd.queue, u_s3);
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let w = rd.surface_bundle.config.width;
        let h = rd.surface_bundle.config.height;
        let pw = w / 3;
        let pw2 = w - pw * 2;

        // E³: clear + draw left panel.
        self.node_e3.execute_panel(rd, view, true, [0, 0, pw, h])?;
        // H³: load + draw centre panel.
        self.node_h3
            .execute_panel(rd, view, false, [pw, 0, pw, h])?;
        // S³: load + draw right panel.
        self.node_s3
            .execute_panel(rd, view, false, [pw * 2, 0, pw2, h])?;
        Ok(())
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        Cow::Owned(format!(
            "Rye - Geodesic Lattice (E³ / H³ / S³) | {fps:.0} fps"
        ))
    }
}

fn main() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("Rye - Geodesic Lattice (E³ / H³ / S³)")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<LatticeApp>(config)
}
