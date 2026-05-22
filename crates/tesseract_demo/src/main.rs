//! Tesseract w-depth demo. Tesseract (8-cell) wireframe rotated in 4D under a
//! continuously animated xy-bivector spin, projected to R³ via `Perspective4D`
//! (the canonical "cube within a cube" view), rendered as antialiased line
//! segments via `LineRasterNode`. Camera is free-roam: WASD + arrow-key
//! translation, mouse-drag look.
//!
//! ## Why this demo (vs. polytope_playground)
//!
//! Validates the small-demo hypothesis: ship the focused experience that a
//! blog post needs, with the minimum pipeline footprint to do it. This demo:
//!
//! - Uses exactly ONE render pipeline (the line rasterizer). No SDF raymarch,
//!   no triangle rasterizer, no point sprite pass, no shared depth buffer.
//! - Doesn't link a console or hot-reload-aware shader DB (still builds them
//!   in via `rye-app` but doesn't expose them).
//! - Skips the entire 4-D rotor composition UI (active set, composer,
//!   filmstrip) that polytope_playground bundles for power users.
//!
//! Resulting wasm bundle should be 1-2 MB compressed (vs polytope_playground's
//! ~3.6 MB) and have fewer compile stalls on first frame.

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3, Vec4};
use rye_app::{
    egui, run_with_config, App, Camera, CameraController, FirstPersonController, FrameCtx,
    OrbitController, RunConfig, SetupCtx,
};
use rye_egui::Console;
use rye_math::{Bivector, Bivector4, EuclideanR3, EuclideanR4, Projection, Rotor, Rotor4};
use rye_physics::polytope::Polytope4;
use rye_render::device::RenderDevice;
use rye_render::{line_raster::LineRasterNode, DepthMode, Viewport};
use rye_shape::LineMesh;
use winit::window::WindowAttributes;

/// Focal distance for the `Perspective4D` viewer along the w-axis. The
/// tesseract has unit circumradius, so vertices live at `w = ±0.5`. A focal
/// distance of `2.0` puts the viewer well outside the polytope; vertices at
/// `w = +0.5` (closest to viewer) project larger than those at `w = -0.5`
/// (farthest), producing the textbook cube-within-cube look.
const FOCAL_DISTANCE: f32 = 2.0;

/// Continuous-spin angular velocity (radians per second) around the xw plane.
/// xw rotation swaps vertex w-coordinates as it cycles, which means the
/// Perspective4D projection's inner-vs-outer assignment changes every half
/// rotation: vertices that were "near" (w = +0.5, projecting large) become
/// "far" (w = −0.5, projecting small) and vice versa. This is the visible
/// signature of 4D rotation under w-depth projection; an xy rotation keeps
/// the w-coordinate static and the projection looks like a plain 3D rotation.
const SPIN_RATE: f32 = 0.4;

/// Whole-tesseract scale at canonical positions. Multiplying every vertex by
/// this puts the polytope in a comfortable mid-distance "look-at" range
/// without the viewer needing to walk halfway across the page.
const POLYTOPE_SCALE: f32 = 1.5;

/// Wireframe line color: warm white, slightly translucent so overlap brightens
/// naturally. Matches polytope_playground's default style.
const EDGE_COLOR: [f32; 4] = [0.95, 0.94, 0.92, 0.9];
const EDGE_WIDTH_PX: f32 = 1.6;

/// Camera modes the demo supports. Orbit is the default (predictable for a
/// hands-off blog visitor); FreeRoam unlocks WASD + mouselook when the user
/// presses `F`. Toggle hint shows in the top-left HUD.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CameraMode {
    Orbit,
    FreeRoam,
}

struct TesseractApp {
    /// Single render pipeline for the whole demo. `DepthMode::Off` because
    /// nothing else writes depth; the wireframe sits on a clean
    /// `LoadOp::Clear` color attachment with no z-buffer fight to manage.
    lines: LineRasterNode,
    /// Camera state. Same `EuclideanR3` flavor polytope_playground uses; the
    /// 4D rotation is on the geometry side (via the rotor), not the camera.
    camera: Camera<EuclideanR3>,
    /// Orbit controller for the default mode. Reused across toggles.
    orbit: OrbitController<EuclideanR3>,
    /// Free-roam controller. Constructed lazily on the first toggle so it
    /// inherits the orbit camera's pose as its starting point.
    free_roam: FirstPersonController<EuclideanR3>,
    /// Active controller selection.
    mode: CameraMode,
    /// World-space position of the camera in free-roam mode. The translation
    /// piece is owned by the demo rather than the controller because the
    /// `FirstPersonController` advance only sets `right/up/forward`; we
    /// integrate position from input on top of it.
    free_roam_pos: Vec3,
    /// Accumulated 4D rotor describing the polytope's current orientation.
    rotor: Rotor4,
    /// Continuous-spin angular velocity. Multiplied by `dt` each tick and
    /// rotor-multiplied into `rotor`. xy-plane by default; `R` resets to
    /// identity (no spin) for a static snapshot.
    omega: Bivector4,
    /// Auto-pause flag. `Space` toggles. When true `omega` is zeroed for
    /// integration but `omega` itself stays so unpausing resumes the same
    /// spin direction and speed.
    paused: bool,
    /// Pre-allocated buffer for the rotated tesseract vertices. Re-used each
    /// frame so we don't allocate inside the hot path.
    rotated_verts: Vec<Vec4>,
    /// Pre-allocated buffer for the LineMesh segments. Same lifecycle.
    mesh: LineMesh<4>,
    /// Dev console: backtick to open, hosts `trace [summary|last|dump|clear|cap]`
    /// and `log [on|off]`. `()` as the Ctx because none of the registered
    /// commands need demo state. The trace command is the diagnostic path for
    /// "where is the stutter coming from?" questions; without the console
    /// there's no way to get its output of the wasm bundle.
    console: Console<()>,
    /// F3-toggle perf overlay. Live FPS + frame-gap stats + sparkline. Per the
    /// 2026-05-22 wasm diagnosis, the stutter source is `between-frames` (the
    /// gap the browser RAF cadence enforces), not our render code; the overlay
    /// makes that visible without re-running `trace dump`.
    perf: rye_app::trace::PerfOverlay,
}

impl App for TesseractApp {
    type Space = EuclideanR3;

    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let topo = Polytope4::Tesseract.topology();

        // One pipeline, no depth. The line rasterizer's `DepthMode::Off`
        // variant skips the depth attachment entirely; the pipeline doesn't
        // declare a depth-stencil state and the render pass omits the
        // attachment. Smallest possible footprint.
        let lines = LineRasterNode::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        // Orbit start: look from slightly above + behind the cube.
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 1.0, 5.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(5.0, -0.15);

        let free_roam = FirstPersonController::<EuclideanR3>::new(0.0, 0.0);
        let free_roam_pos = camera.position;

        // Pre-size buffers so the first-frame upload doesn't have to grow
        // them. Tesseract: 16 vertices, 32 edges.
        let mut rotated_verts = Vec::with_capacity(topo.vertices.len());
        rotated_verts.resize(topo.vertices.len(), Vec4::ZERO);
        let mut mesh = LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());

        let mut console = Console::<()>::new();
        rye_app::trace::register_command(&mut console);
        rye_app::log::register_command(&mut console);
        let perf = rye_app::trace::PerfOverlay::new();

        Ok(Self {
            lines,
            camera,
            orbit,
            free_roam,
            mode: CameraMode::Orbit,
            free_roam_pos,
            rotor: Rotor4::IDENTITY,
            // `Bivector4::basis(2)` is the xw plane in `Plane4`'s ordering
            // (0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw). Times SPIN_RATE = rad/sec
            // omega in the xw plane (sweeps vertex w-coordinates through the
            // projection's inner-vs-outer mapping, which is the actual visible
            // signature of 4D rotation under Perspective4D).
            omega: Bivector4::basis(2) * SPIN_RATE,
            paused: false,
            rotated_verts,
            mesh,
            console,
            perf,
        })
    }

    fn space(&self) -> &EuclideanR3 {
        &EuclideanR3
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        // Use the runner-supplied wall-clock dt (varies frame-to-frame, captures
        // stutter accurately). A hardcoded `1.0 / 60.0` would have the rotor
        // advance "60 fps worth" each frame regardless of actual cadence — at the
        // observed 50fps that's a ~17% slowdown of the intended SPIN_RATE, and on
        // stutter frames the rotor would visibly lag.
        //
        // Clamp the upper bound so a multi-second stall (tab backgrounded, GC)
        // doesn't catapult the rotor through a half-revolution on the
        // catch-up frame.
        let dt = ctx.dt.min(0.1);

        // 4D rotor integration. `(omega * dt).exp()` is the small-step rotor
        // that brings `rotor` to `rotor` after dt seconds of constant omega
        // rotation; multiplying composes it with the existing orientation.
        if !self.paused {
            let step = (self.omega * dt).exp();
            self.rotor = (step * self.rotor).normalize();
        }

        // Camera advance. Mode-aware; both controllers consume the drained
        // input but only one shapes the resulting frame.
        match self.mode {
            CameraMode::Orbit => {
                self.orbit
                    .advance(ctx.input.clone(), &mut self.camera, &EuclideanR3, dt);
            }
            CameraMode::FreeRoam => {
                self.free_roam
                    .advance(ctx.input.clone(), &mut self.camera, &EuclideanR3, dt);
                // `FrameInput` already aggregates WASD + Space/Shift into the
                // `move_forward`, `move_right`, `move_up` axes (+1 / 0 / -1
                // each frame). Combine with the camera's local basis to get
                // a world-space velocity vector.
                let speed = 2.5; // units/sec
                let mut delta = self.camera.forward * ctx.input.move_forward
                    + self.camera.right * ctx.input.move_right
                    + Vec3::Y * ctx.input.move_up;
                if delta.length_squared() > 1e-6 {
                    delta = delta.normalize();
                    self.free_roam_pos += delta * speed * dt;
                    self.camera.position = self.free_roam_pos;
                }
            }
        }
    }

    fn on_event(&mut self, ev: &winit::event::WindowEvent, _ctx: &mut FrameCtx<'_>) {
        use winit::event::{ElementState, KeyEvent, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(code),
                    ..
                },
            ..
        } = ev
        {
            match code {
                KeyCode::KeyF => {
                    // Toggle camera mode. Free-roam picks up where orbit left
                    // off so the view doesn't snap.
                    self.mode = match self.mode {
                        CameraMode::Orbit => CameraMode::FreeRoam,
                        CameraMode::FreeRoam => CameraMode::Orbit,
                    };
                    self.free_roam_pos = self.camera.position;
                }
                KeyCode::KeyT | KeyCode::Space => {
                    // T toggles pause; Space is the common gamer pause key.
                    // Both go through the same flag.
                    if !matches!(code, KeyCode::Space)
                        || !matches!(self.mode, CameraMode::FreeRoam)
                    {
                        self.paused = !self.paused;
                    }
                }
                KeyCode::KeyR => {
                    // Reset orientation to identity. omega is preserved so
                    // unpausing keeps the chosen spin direction.
                    self.rotor = Rotor4::IDENTITY;
                }
                _ => {}
            }
        }
    }

    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> Result<()> {
        let topo = Polytope4::Tesseract.topology();

        // 1. Apply the current 4D rotor to every vertex. Scale to the demo's
        // viewing size after rotation so the rotor stays unit-norm.
        for (i, &v) in topo.vertices.iter().enumerate() {
            self.rotated_verts[i] = self.rotor.apply(v) * POLYTOPE_SCALE;
        }

        // 2. Build the LineMesh from rotated vertex pairs + edge topology.
        // Reusing the same vectors avoids per-frame allocation.
        self.mesh.segments.clear();
        self.mesh.colors.clear();
        self.mesh.widths.clear();
        for &[i, j] in topo.edges {
            let a = self.rotated_verts[i as usize];
            let b = self.rotated_verts[j as usize];
            self.mesh
                .segments
                .push((a.to_array(), b.to_array()));
            self.mesh.colors.push((EDGE_COLOR, EDGE_COLOR));
            self.mesh.widths.push(EDGE_WIDTH_PX);
        }

        // 3. Upload + project. `Projection::Perspective4D` projects each 4D
        // point to R³ by scaling x/y/z by `focal_distance / (focal_distance -
        // w)`, the standard pinhole formula. The line rasterizer tessellates
        // each segment in R⁴ (where flat-Euclidean tessellation is just the
        // endpoints), then projects every tessellation sample to R³.
        let projection = Projection::<4>::Perspective4D {
            focal_distance: FOCAL_DISTANCE,
        };
        self.lines.upload::<EuclideanR4, 4>(
            &rd.device,
            &rd.queue,
            &self.mesh,
            &projection,
            1, // flat space; one sample per segment is exact.
        );

        // 4. Camera uniforms. Standard perspective projection from R³ to
        // clip; aspect from the surface size.
        let cfg = &rd.surface_bundle.config;
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.05, 100.0);
        let view_dir = self.camera.view();
        let view_m = Mat4::look_to_rh(view_dir.position, view_dir.forward, view_dir.up);
        self.lines.set_camera(
            &rd.queue,
            proj * view_m,
            Vec2::new(cfg.width as f32, cfg.height as f32),
        );

        // 5. Clear the color attachment + draw the lines in one pass. No
        // pre-pass exists in this demo so this is the first write to the
        // swapchain (or to the offscreen sRGB scene target on wasm; the
        // composite handles the gamma either way).
        let mut encoder = rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tesseract_demo::clear"),
            });
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tesseract_demo::clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
        rd.queue.submit(Some(encoder.finish()));

        // Line rasterizer: `LoadOp::Load`, so the clear above is the
        // backdrop. No depth, no resolve.
        let viewport = Viewport::full([cfg.width, cfg.height]);
        self.lines.execute(rd, view, None, Some(&viewport))?;
        Ok(())
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        // Minimal HUD: top-left overlay showing the mode + a one-line key
        // legend. No interactive widgets in v1; reduces egui's pipeline
        // count (just text + a background rect) and keeps the visual
        // uncluttered for the blog embed.
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
        // Perf overlay: F3-toggle, draws on top of the HUD. Reads
        // `frame_trace::history` for the live FPS / frame-time / between-frames
        // sparkline. Cheap when hidden (just a key-press check).
        self.perf.show(ctx);
        // Pump any tracing events into the console scrollback (only mirrors
        // when `log on` has been issued), then render the console panel.
        // Backtick toggles its visibility.
        rye_app::log::pump_into(&mut self.console);
        self.console.ui(ctx, &mut ());
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        format!("tesseract_demo  -  {fps:.0} fps").into()
    }
}

fn main() -> Result<()> {
    #[cfg(target_arch = "wasm32")]
    {
        const HOST_ID: &str = "rye-canvas-host";
        const BUTTON_ID: &str = "rye-launch";
        if rye_app::wasm::is_manual_mode(HOST_ID) {
            rye_app::wasm::wait_for_launch(BUTTON_ID, || {
                if let Err(e) = launch_app() {
                    tracing::error!("tesseract_demo launch failed: {e:#}");
                }
            })?;
            return Ok(());
        }
    }
    launch_app()
}

fn launch_app() -> Result<()> {
    let config = RunConfig {
        window: WindowAttributes::default()
            .with_title("tesseract demo")
            .with_visible(false),
        ..RunConfig::default()
    };
    run_with_config::<TesseractApp>(config)
}
