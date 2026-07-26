//! Tesseract w-depth demo. An 8-cell wireframe spun in 4D, projected to R³ via
//! `Perspective4D` (the "cube within a cube" view), drawn as antialiased lines.
//! Camera is orbit by default, free-roam (WASD + mouse) on `F`.
//!
//! The minimal-footprint counterpart to polytope_playground: one render pipeline
//! (the line rasterizer, no SDF / triangle / point passes, no depth buffer) and
//! none of the rotor-composition UI, for a smaller wasm bundle.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3};
use loam_app::{
    egui, freecam::Freecam, App, Camera, CameraController, FrameCtx, OrbitController, RenderCtx,
    RunConfig, SetupCtx,
};

// Per-frame allocation telemetry surfaced in `frame_trace` + PerfOverlay; ~5-10ns
// per allocation, negligible next to the interop cost being chased.
#[global_allocator]
static GLOBAL: loam_time::alloc::CountingAllocator<std::alloc::System> =
    loam_time::alloc::CountingAllocator::new(std::alloc::System);
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector, Bivector4, EuclideanR3, Rotor4};
use loam_physics::polytope::Polytope4;
use loam_render::device::RenderDevice;
use loam_render::{DepthMode, LineRasterStaticR4Node, Viewport};
use loam_shape::LineMesh;
use winit::window::WindowAttributes;

/// `Perspective4D` focal distance along the w-axis. Vertices live at `w = ±0.5`
/// (unit circumradius); `2.0` keeps the viewer outside the polytope so `w = +0.5`
/// projects larger than `w = -0.5`, the cube-within-cube look.
const FOCAL_DISTANCE: f32 = 2.0;

/// Continuous-spin angular velocity (rad/s) in the xw plane. xw rotation sweeps
/// vertex w-coordinates, flipping the projection's inner-vs-outer assignment each
/// half turn; that swap is the visible signature of 4D rotation (an xy spin would
/// read as plain 3D rotation).
const SPIN_RATE: f32 = 0.4;

/// Whole-tesseract scale, for a comfortable mid-distance look-at range.
const POLYTOPE_SCALE: f32 = 1.5;

/// Per-line tint over the shader's w-depth palette (cool-back to warm-front). RGB
/// near white preserves the depth gradient's hue; alpha is per-line opacity.
const EDGE_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.95];
const EDGE_WIDTH_PX: f32 = 1.6;

/// Camera modes. Orbit is the default; `F` toggles FreeRoam (WASD + mouselook).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CameraMode {
    Orbit,
    FreeRoam,
}

struct TesseractApp {
    /// The demo's only render pipeline. `DepthMode::Off`: nothing else writes
    /// depth. `LineRasterStaticR4Node` keeps the mesh on the GPU between frames,
    /// so per-frame work is one 144-byte uniform write (rotor + view*proj +
    /// viewport + focal), no instance-buffer upload.
    lines: LineRasterStaticR4Node,
    /// Camera state. The 4D rotation is on the geometry side (the rotor), not the
    /// camera, which is plain `EuclideanR3`.
    camera: Camera<EuclideanR3>,
    /// Orbit controller (default mode); reused across toggles.
    orbit: OrbitController<EuclideanR3>,
    /// Freecam preset (mouse-look + WASD + cursor grab); owns its own pose/grab
    /// state. The demo only toggles `set_active` and calls `advance`.
    freecam: Freecam,
    /// Active controller selection.
    mode: CameraMode,
    /// Accumulated 4D rotor: the polytope's current orientation.
    rotor: Rotor4,
    /// Spin angular velocity, integrated into `rotor` each tick. Preserved across
    /// pause + `R` so resuming keeps the same spin.
    omega: Bivector4,
    /// Pause flag; `T`/`Space` toggle it. `omega` is preserved while paused.
    paused: bool,
    /// Dev console (backtick): `trace`, `log`, etc. `()` Ctx since no command
    /// needs demo state.
    console: Console<()>,
    /// F3-toggle perf overlay: live FPS + frame-gap stats + sparkline. The
    /// dominant gap on wasm is `between-frames` (browser RAF cadence), not render.
    perf: loam_app::trace::PerfOverlay,
}

impl TesseractApp {
    /// Force-compile the App's render pipelines by running one dummy `record`
    /// into a 1x1 throwaway texture (compilation is size-independent). egui's and
    /// the composite's pipelines are NOT warmed here. Lives in the demo, not the
    /// runner, because only the demo knows which pipelines and configs it touches.
    fn warm_pipelines(&mut self, rd: &RenderDevice) {
        let dummy_tex = rd.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tesseract_demo::warmup"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Must match the pipeline's target format (same value the pipeline was
            // built against), so the warmup pass is format-compatible.
            format: rd.target_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tesseract_demo::warmup-encoder"),
            });
        {
            let mut ctx = RenderCtx {
                rd,
                view: &dummy_view,
                encoder: &mut encoder,
            };
            // One run of the regular draw path drives the first compile; warming
            // isn't critical, so discard errors.
            let _ = self.record(&mut ctx);
        }
        rd.queue.submit(Some(encoder.finish()));

        tracing::info!("tesseract_demo: warmed render pipelines");
    }
}

impl App for TesseractApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let topo = Polytope4::Tesseract.topology();

        // `?spin_rate=N` (rad/s) and `?paused=true|1` override the defaults, for
        // blog embeds wanting a snapshot or slower animation. Native: `--spin_rate`.
        let args = loam_app::args::Args::current();
        let spin_rate = args.parse::<f32>("spin_rate").unwrap_or(SPIN_RATE);
        let paused = args
            .get("paused")
            .map(|v| matches!(v, "true" | "1" | "yes"))
            .unwrap_or(false);

        // One pipeline, no depth: `DepthMode::Off` skips the depth attachment.
        let mut lines = LineRasterStaticR4Node::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        // Build the canonical R⁴ edge mesh once (identity orientation; the rotor
        // is applied per frame in the vertex shader).
        let mut canonical = LineMesh::<4>::default();
        canonical.segments.reserve(topo.edges.len());
        canonical.colors.reserve(topo.edges.len());
        canonical.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let a = topo.vertices[i as usize] * POLYTOPE_SCALE;
            let b = topo.vertices[j as usize] * POLYTOPE_SCALE;
            canonical.segments.push((a.to_array(), b.to_array()));
            canonical.colors.push((EDGE_COLOR, EDGE_COLOR));
            canonical.widths.push(EDGE_WIDTH_PX);
        }
        lines.upload_mesh(&ctx.rd.device, &ctx.rd.queue, &canonical);

        // Orbit start: slightly above + behind the cube.
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 1.0, 5.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(5.0, -0.15);

        // Freecam preset, inactive at startup; `F` activates it.
        let freecam = Freecam::new().with_speed(2.5);

        let mut console = Console::<()>::new();
        loam_app::trace::register_command(&mut console);
        loam_app::fps::register_command(&mut console);
        loam_app::vsync::register_command(&mut console);
        loam_app::version::register_command(
            &mut console,
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_HASH"),
            env!("BUILD_DIRTY"),
        );
        loam_app::log::register_command(&mut console);
        let perf = loam_app::trace::PerfOverlay::new();

        let mut app = Self {
            lines,
            camera,
            orbit,
            freecam,
            mode: CameraMode::Orbit,
            rotor: Rotor4::IDENTITY,
            // basis(2) is the xw plane (Plane4 order: 0=xy,1=xz,2=xw,3=yz,4=yw,
            // 5=zw); the plane whose spin sweeps w through the projection.
            omega: Bivector4::basis(2) * spin_rate,
            paused,
            console,
            perf,
        };

        // Materialize the App's PSOs now, during setup, instead of stalling
        // ~100-500ms on first real draw. (egui + composite pipelines aren't
        // warmed; see `warm_pipelines`.)
        app.warm_pipelines(ctx.rd);

        Ok(app)
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Wall-clock dt so the spin tracks real cadence; clamped so a multi-second
        // stall doesn't catapult the rotor through a half-revolution on catch-up.
        let dt = ctx.dt.min(0.1);

        // 4D rotor integration: `(omega * dt).exp()` is the dt-step rotor,
        // composed onto the current orientation.
        if !self.paused {
            let step = (self.omega * dt).exp();
            self.rotor = (step * self.rotor).normalize();
        }

        // Advance whichever controller is active.
        match self.mode {
            CameraMode::Orbit => {
                self.orbit
                    .advance(ctx.input, &mut self.camera, &EuclideanR3, dt);
            }
            CameraMode::FreeRoam => {
                self.freecam.advance(ctx.input, &mut self.camera, dt);
            }
        }
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        // Don't fire app hotkeys while an egui widget (e.g. the console) has
        // keyboard focus, or typing `trace` would also toggle pause via `KeyT`.
        if ctx.ui_has_focus {
            return;
        }
        // Alt needs both edges (the freecam preset interprets press + release per
        // its cursor_mode); everything else acts on press only.
        if matches!(code, KeyCode::AltLeft | KeyCode::AltRight)
            && matches!(self.mode, CameraMode::FreeRoam)
        {
            self.freecam.on_alt(matches!(state, ElementState::Pressed));
            return;
        }
        if !matches!(state, ElementState::Pressed) {
            return;
        }
        match code {
            KeyCode::KeyF => {
                // Toggle camera mode; freecam seeds from the current pose so the
                // switch is continuous, not a teleport.
                self.mode = match self.mode {
                    CameraMode::Orbit => {
                        self.freecam.set_active(true, self.camera.position);
                        CameraMode::FreeRoam
                    }
                    CameraMode::FreeRoam => {
                        self.freecam.set_active(false, self.camera.position);
                        CameraMode::Orbit
                    }
                };
            }
            // T always toggles pause; Space too, but not in FreeRoam (where it's
            // the jump-up axis).
            KeyCode::KeyT => {
                self.paused = !self.paused;
            }
            KeyCode::Space if !matches!(self.mode, CameraMode::FreeRoam) => {
                self.paused = !self.paused;
            }
            KeyCode::KeyR => {
                // Reset orientation; omega is preserved.
                self.rotor = Rotor4::IDENTITY;
            }
            _ => {}
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        // Per-frame work is one uniform write; the vertex shader applies rotor ->
        // Perspective4D -> view*proj per vertex over the static mesh.
        let cfg = &ctx.rd.surface_bundle.config;
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.05, 100.0);
        let view_dir = self.camera.view();
        let view_m = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        self.lines.set_transform(
            &ctx.rd.queue,
            self.rotor,
            proj * view_m,
            Vec2::new(cfg.width as f32, cfg.height as f32),
            FOCAL_DISTANCE,
        );

        // Clear pass into the shared encoder. Could fuse into the raster pass via
        // a LoadOp::Clear variant; deferred until a second demo needs it.
        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tesseract_demo::clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.027,
                            g: 0.027,
                            b: 0.035,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // Line raster pass into the same encoder; `LoadOp::Load` preserves the
        // clear. The runner submits the encoder once at end of frame.
        let viewport = Viewport::full([cfg.width, cfg.height]);
        self.lines
            .record(ctx.encoder, ctx.view, None, Some(&viewport));
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        // Top-left HUD: mode + key legend, no interactive widgets.
        egui::Area::new(egui::Id::new("tesseract-hud"))
            .anchor(egui::Align2::LEFT_TOP, [12.0, 12.0])
            .show(ctx, |ui| {
                let mode = match self.mode {
                    CameraMode::Orbit => "orbit",
                    CameraMode::FreeRoam => "free-roam (WASD + mouse drag, space/shift = up/down)",
                };
                ui.colored_label(
                    egui::Color32::from_rgb(216, 216, 223),
                    format!(
                        "Tesseract \u{00b7} {mode} \u{00b7} F: toggle camera \u{00b7} \
                         T: pause \u{00b7} R: reset \u{00b7} \u{0060}: console \u{00b7} \
                         F3: perf"
                    ),
                );
                if self.paused {
                    ui.colored_label(egui::Color32::from_rgb(220, 180, 90), "[paused]");
                }
            });
        // Build id (git hash + dirty marker, baked via build.rs) in the faded
        // bottom-right, for "is this a fresh build?" verification across reloads.
        egui::Area::new(egui::Id::new("tesseract-build-id"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
            .show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(120, 120, 130),
                    format!("build {}{}", env!("BUILD_HASH"), env!("BUILD_DIRTY"),),
                );
            });
        // F3-toggle perf overlay; cheap when hidden.
        self.perf.show(ctx);
        // Mirror tracing events into the console (when `log on`), then draw it.
        loam_app::log::pump_into(&mut self.console);
        self.console.ui(ctx, &mut ());
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        format!("tesseract_demo  -  {fps:.0} fps").into()
    }
}

fn main() -> Result<()> {
    // `loam_app::run` handles native + wasm dispatch; the default `WasmConfig`
    // matches the standard `index.html` element IDs.
    loam_app::run::<TesseractApp>(RunConfig {
        window: WindowAttributes::default()
            .with_title("tesseract demo")
            .with_visible(false),
        ..RunConfig::default()
    })
}
