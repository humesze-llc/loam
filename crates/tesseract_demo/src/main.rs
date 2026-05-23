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
use glam::{Mat4, Vec2, Vec3};
use rye_app::{
    egui, run_with_config, App, Camera, CameraController, FirstPersonController, FrameCtx,
    OrbitController, RenderCtx, RunConfig, SetupCtx,
};

// Per-frame allocation telemetry. Wraps the system allocator with a counter
// pair that surfaces in `frame_trace` + PerfOverlay. ~5-10ns per allocation
// on native, ~10-20ns on wasm32 (atomic ops are cheaper there); negligible
// next to the per-frame interop cost we're hunting. Steady-state goal is
// "0 allocs/frame" in the overlay after the perf-hardening pass; without
// this telemetry we'd be flying blind on whether we're hitting it.
#[global_allocator]
static GLOBAL: rye_time::alloc::CountingAllocator<std::alloc::System> =
    rye_time::alloc::CountingAllocator::new(std::alloc::System);
use rye_render::device::RenderDevice;
use rye_egui::Console;
use rye_math::{Bivector, Bivector4, EuclideanR3, Rotor4};
use rye_physics::polytope::Polytope4;
use rye_render::{LineRasterStaticR4Node, DepthMode, Viewport};
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
    ///
    /// `LineRasterStaticR4Node` (vs the dynamic-upload `LineRasterNode`)
    /// keeps the mesh on the GPU between frames. Per-frame work is just a
    /// 144-byte uniform write carrying rotor + view*proj + viewport + focal.
    /// This was the perf-hardening N1 change (2026-05-22): eliminates one
    /// `queue.write_buffer` JS-interop call per frame on wasm32 (the
    /// instance-buffer upload that used to fire from `lines.upload`) AND
    /// removes the two `Vec::with_capacity` allocations the upload path
    /// did inside `LineRasterNode::upload`.
    lines: LineRasterStaticR4Node,
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
    // No per-frame vertex / mesh scratch state: the canonical R⁴ edge mesh is
    // uploaded ONCE at setup, then the GPU vertex shader applies the rotor +
    // Perspective4D projection every frame from a uniform.
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

impl TesseractApp {
    /// Force GPU compilation of every render pipeline this demo uses by
    /// running one dummy frame into a 1x1 throwaway texture. The pipelines
    /// touched are the App's own (currently just `LineRasterNode`); egui's
    /// and the composite's are intentionally NOT warmed here.
    ///
    /// Architectural note: warming lives in the demo (this `impl`), not in
    /// the trait, because only the demo knows which pipelines it'll touch
    /// and in what config (target_format / depth / sample_count). A generic
    /// runner-side warmup would either be too conservative (compile
    /// nothing) or too aggressive (compile every node-variant the demo
    /// links). Demos opt in by calling this from `setup`.
    ///
    /// Cost: one tiny texture alloc + one extra `queue.submit` at setup
    /// time. The size of the dummy target is 1x1; pipeline compilation
    /// doesn't depend on output size, just on the pipeline state
    /// configuration. The driver compiles for the format we built the
    /// pipeline against and caches the result for subsequent draws at any
    /// size.
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
            // Must match the pipeline's target format. `target_format()`
            // returns the sRGB sibling on the composite path (wasm) and the
            // direct swapchain format otherwise — same value the pipeline
            // was built with at construction time, so the warmup pass is
            // format-compatible.
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
            // `record` is the regular per-frame draw path; running it once
            // with the current state (identity rotor + initial camera) is
            // enough to drive the pipeline through its first compile.
            // Discard any error — warming isn't critical.
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

        // Args: `?spin_rate=N` overrides the default rotation speed (rad/s),
        // `?paused=true|1` starts paused. Both useful for blog embeds where
        // the page author wants a specific static snapshot or a slower
        // animation for screen-recording. Native users can pass the same as
        // `--spin_rate=0.2 --paused=true`.
        let args = rye_app::args::Args::current();
        let spin_rate = args.parse::<f32>("spin_rate").unwrap_or(SPIN_RATE);
        let paused = args
            .get("paused")
            .map(|v| matches!(v, "true" | "1" | "yes"))
            .unwrap_or(false);

        // One pipeline, no depth. The line rasterizer's `DepthMode::Off`
        // variant skips the depth attachment entirely; the pipeline doesn't
        // declare a depth-stencil state and the render pass omits the
        // attachment. Smallest possible footprint.
        let mut lines = LineRasterStaticR4Node::new(
            &ctx.rd.device,
            ctx.rd.target_format(),
            DepthMode::Off,
            ctx.rd.sample_count(),
        );

        // Build the canonical R⁴ edge mesh ONCE. Vertices are at the
        // tesseract's natural unit-circumradius positions, scaled by
        // POLYTOPE_SCALE for comfortable viewing distance. The rotor is
        // applied each frame inside the vertex shader, so the mesh uploaded
        // here is the identity-orientation reference.
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

        // Orbit start: look from slightly above + behind the cube.
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 1.0, 5.0);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(5.0, -0.15);

        let free_roam = FirstPersonController::<EuclideanR3>::new(0.0, 0.0);
        let free_roam_pos = camera.position;

        let mut console = Console::<()>::new();
        rye_app::trace::register_command(&mut console);
        rye_app::log::register_command(&mut console);
        let perf = rye_app::trace::PerfOverlay::new();

        let mut app = Self {
            lines,
            camera,
            orbit,
            free_roam,
            mode: CameraMode::Orbit,
            free_roam_pos,
            rotor: Rotor4::IDENTITY,
            // `Bivector4::basis(2)` is the xw plane in `Plane4`'s ordering
            // (0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw). Times the args-or-default
            // spin rate = rad/sec omega in the xw plane (sweeps vertex
            // w-coordinates through the projection's inner-vs-outer mapping,
            // which is the actual visible signature of 4D rotation under
            // Perspective4D).
            omega: Bivector4::basis(2) * spin_rate,
            paused,
            console,
            perf,
        };

        // Pipeline warmup: drive every render pipeline this demo will use
        // through one dummy `record` call into a 1x1 throwaway color
        // attachment. This forces the GPU driver to materialize the PSO
        // (pipeline state object) NOW, during setup, instead of stalling
        // for ~100-500ms the first time each pipeline is drawn against the
        // real swapchain.
        //
        // Architectural note: warming lives in the demo, not the runner,
        // for two reasons. (1) only the demo knows which pipelines it'll
        // touch and in what config (target_format / depth / sample_count);
        // a generic runner-side warmup would either be too conservative
        // (compile nothing) or too aggressive (compile every node-variant
        // the demo links). (2) Warming + click-to-start interact: when the
        // demo is manually-launched, warmup at App::setup runs after the
        // click, which is precisely the window where a brief loading delay
        // is acceptable — we don't want to spend that compile budget at
        // page-load before the user has expressed interest.
        //
        // Doesn't warm `ui.paint` (egui owns its pipelines, compiles them
        // lazily per glyph / shape variant) or `composite` (runner-owned).
        // If diagnostics still show big spikes after this, the cause isn't
        // pipeline compilation in the App-owned path.
        app.warm_pipelines(ctx.rd);

        Ok(app)
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

    fn on_event(&mut self, ev: &winit::event::WindowEvent, ctx: &mut FrameCtx<'_>) {
        use winit::event::{ElementState, KeyEvent, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};
        // Gate app-level hotkeys on egui NOT having keyboard focus. Without this,
        // typing `trace` in the console fires our `KeyT` handler and toggles
        // pause — silently freezing the animation while the user just wanted to
        // run a console command. `ui_has_focus` is the runner's flag for "an
        // egui widget (TextEdit, the console, etc.) is consuming keyboard."
        if ctx.ui_has_focus {
            return;
        }
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

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        // Per-frame work is now a single uniform write. The static mesh
        // uploaded at setup is reused unchanged; the GPU vertex shader
        // applies rotor -> Perspective4D -> view*proj per vertex.
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

        // Clear pass into the shared encoder. Could fuse into the line raster
        // pass by giving the rasterizer a LoadOp::Clear variant; saves one pass
        // per frame but adds API surface for one demo. Defer until justified by
        // another demo.
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

        // Line raster pass into the same encoder. `LoadOp::Load` preserves
        // the clear above. No depth, no resolve. The runner submits the
        // encoder once at end of frame (along with ui-paint + composite).
        let viewport = Viewport::full([cfg.width, cfg.height]);
        self.lines
            .record(ctx.encoder, ctx.view, None, Some(&viewport));
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
        // Build identifier: short git hash + dirty marker, baked at compile
        // time via build.rs. Bottom-right corner, faded so it doesn't compete
        // with the rotating tesseract for attention but is always visible
        // for "am I looking at a fresh build?" verification across reloads.
        egui::Area::new(egui::Id::new("tesseract-build-id"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
            .show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(120, 120, 130),
                    format!(
                        "build {}{}",
                        env!("BUILD_HASH"),
                        env!("BUILD_DIRTY"),
                    ),
                );
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
        // Worker side: the wasm binary was instantiated by the Blob-URL
        // bootstrap inside a fresh Worker. Run the App lifecycle via
        // `worker::run::<TesseractApp>()` — this constructs a
        // `WorkerRunner<TesseractApp>` after the canvas-transfer message
        // arrives, then drives App::setup + per-frame update/record on
        // its own RAF loop. Tracing + panic hook are installed inside
        // `worker::run` itself (worker has its own JS heap).
        if rye_app::wasm::is_worker_context() {
            return rye_app::wasm::worker::run::<TesseractApp>();
        }

        // Main thread: wire the launch button to spawn the worker on
        // click. The button + canvas + host element come from the demo's
        // index.html. The actual App lifecycle (Phase B onward) will run
        // inside the worker; for Phase A the worker only renders a
        // cycling clear-color.
        const HOST_ID: &str = "rye-canvas-host";
        const BUTTON_ID: &str = "rye-launch";
        const CANVAS_ID: &str = "rye-canvas";
        if rye_app::wasm::is_manual_mode(HOST_ID) {
            return rye_app::wasm::launch_on_click(HOST_ID, BUTTON_ID, CANVAS_ID);
        }
        // No `data-mode="manual"` on the host element: fall through to
        // the legacy main-thread auto-launch path. Useful for native or
        // for wasm demos that haven't migrated to worker mode yet.
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
