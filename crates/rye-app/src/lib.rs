//! `rye-app`: thin App trait + event-loop runner that extracts the winit boilerplate
//! every Rye example currently rewrites.
//!
//! ## What this crate is, and isn't
//!
//! This is a *small framework*. Apps implement [`App`] on a struct that owns their state,
//! and the runner [`run`] (or [`run_with_config`]) handles:
//!
//! - Window creation and the winit `ApplicationHandler` impl.
//! - [`RenderDevice`] construction and surface-error recovery.
//! - [`ShaderDb`] + [`AssetWatcher`] for shader hot-reload.
//! - [`InputState`] event routing -> drained `FrameInput` per redraw.
//! - [`FixedTimestep`] driving `App::tick` at the fixed-rate.
//! - FPS bookkeeping and rate-limited title updates.
//!
//! It is **explicitly not**:
//!
//! - An ECS or scene graph. Apps own their state directly.
//! - A render-graph orchestrator. Apps own their `RenderNode`s and compose them inside
//!   [`App::render`].
//! - A camera framework. The user owns [`Camera<S>`] and a [`CameraController<S>`] in
//!   their `App` struct, advanced from inside `App::update`. The framework only hands
//!   them the drained input.
//!
//! A frame-capture pipeline is included behind the `capture` feature (default-on); see
//! [`capture`] for the console commands, hotkeys, and two-tap (pre-egui / post-egui)
//! readback model. External screen recorders (OBS) remain the right tool for long
//! recording sessions that need codec choice + audio + multi-source mixing.
//!
//! Designed for a small ergonomic gain; explicitly not an ECS or scene graph.
//!
//! ## Lifecycle
//!
//! ```text
//! run::<MyApp>()
//!   * EventLoop::new
//!   * on `resumed`:
//!         create Window
//!         create RenderDevice
//!         create ShaderDb + AssetWatcher
//!         create UiIntegration (egui)
//!         A::setup(&mut SetupCtx) -> A
//!   * on each redraw:
//!         FixedTimestep::advance -> ticks
//!         for each tick: A::tick(dt, &mut TickCtx)
//!         input.take_frame()
//!         A::update(&mut FrameCtx)
//!         A::ui(&egui::Context, &mut FrameCtx)
//!         A::on_event(...) for each WindowEvent
//!         poll AssetWatcher -> if events:
//!             shader_db.apply_events(events, app.space())
//!             A::on_shader_reload(&mut SetupCtx)
//!         maybe update title (rate-limited to ~1 Hz)
//!         RenderDevice::begin_frame
//!         A::render(rd, view)
//!         UiIntegration::paint  (egui overlay, LoadOp::Load)
//!         frame.present
//!   * on `Esc` or `CloseRequested`: exit cleanly
//!     (Esc is suppressed when an egui TextEdit has keyboard focus)
//! ```

use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "capture")]
pub mod capture;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use rye_asset::AssetWatcher;
use rye_egui::UiIntegration;
use rye_input::{FrameInput, InputState};
use rye_math::WgslSpace;
use rye_render::device::RenderDevice;
use rye_shader::ShaderDb;
use rye_time::FixedTimestep;

// Convenience re-exports so apps don't have to depend on each crate
// individually for the most common types.
pub use rye_camera::{
    Camera, CameraController, CameraView, FirstPersonController, OrbitController,
};
pub use rye_input::FrameInput as Input;
// Re-export the egui surface so apps that override `App::ui` depend on
// `rye-app` only and the version pin lives in `rye-egui`.
pub use rye_egui::{egui, world_to_screen, BottomOverlay, LinearIndicator};

// ---------------------------------------------------------------------------
// App trait
// ---------------------------------------------------------------------------

/// The framework calls back into your App through this trait. All methods except
/// [`App::setup`], [`App::space`], and [`App::render`] have default impls; override
/// only what you need.
///
/// `Self::Space` is the ambient geometry. The user's app owns an instance of it
/// (typically as a struct field) so that hot-reload can re-emit shader preludes against
/// the same instance the renderer is using.
pub trait App: Sized + 'static {
    /// **Shader-prelude** geometry. The framework runs `ShaderDb::apply_events` against
    /// this instance during hot-reload, so `rye_distance` / `rye_log` / `rye_exp` etc.
    /// in WGSL evaluate under this metric. Apps that don't care about geometry use
    /// `EuclideanR3`.
    ///
    /// **This is not a commitment about the camera, the player, or the scene.** Those
    /// are user-owned types and may use a different Space, or no Space at all. Two
    /// valid patterns:
    ///
    /// - **All-in geometry**: scene, camera, player, and shader prelude all share one
    ///   Space. e.g. `App::Space = HyperbolicH3` + `Camera<HyperbolicH3>`. The camera
    ///   orbits along honest H³ geodesics, the player moves along honest H³ geodesics,
    ///   the shader applies H³ to distance / fog math.
    /// - **Hybrid** (fractal-demo-style): scene is Cartesian, camera orbits in flat
    ///   Euclidean space, but the shader prelude is non-Euclidean to apply a
    ///   geodesic-fog metric. e.g. `App::Space = HyperbolicH3` + `Camera<EuclideanR3>`.
    ///   The camera math is Cartesian; the shader applies H³ only to the fog distance.
    ///
    /// The conflation hazard: if you write `Camera<Self::Space>` without thinking, you
    /// commit your scene to live in that Space's coordinates. For H³ that means the
    /// Poincaré ball; orbit distances inherited from a Euclidean default
    /// (`OrbitController::default()` -> `distance ≈ 3.55`) will `exp_target` into a
    /// tangent vector that lands at `tanh(1.78) ≈ 0.94` of the way to the ideal
    /// boundary, where the metric explodes. If your scene's geometry isn't actually in
    /// H³, use `Camera<EuclideanR3>` and treat `App::Space` purely as the
    /// shader-prelude axis.
    type Space: WgslSpace + 'static;

    /// One-shot construction after `RenderDevice` and `ShaderDb` are ready. Build
    /// render nodes, load shaders, allocate gameplay state, and store everything
    /// (including `Self::Space` and any `Camera<S>` / `CameraController<S>`) inside
    /// the returned `Self`.
    fn setup(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self>;

    /// Borrow the user-owned `Self::Space` so the framework can pass it to
    /// `ShaderDb::apply_events` on hot-reload.
    fn space(&self) -> &Self::Space;

    /// Per-tick simulation step at the fixed-timestep rate (60 Hz by default;
    /// configurable via [`RunConfig::fixed_hz`]). `n` is usually 0 or 1 per frame;
    /// can spike up to [`RunConfig::max_ticks_per_frame`] if the renderer stalled.
    fn tick(&mut self, _dt: f32, _ctx: &mut TickCtx) {}

    /// Per-frame update: input drained, ready for the app to advance its camera
    /// controller, recompute uniforms, etc. Runs *after* all the frame's ticks.
    fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

    /// Custom `WindowEvent` handling beyond the input routing the framework runs first.
    /// Most apps don't need this; useful for keyboard-driven mode toggles,
    /// drag-and-drop, etc.
    fn on_event(&mut self, _ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {}

    /// Hot-reload notification: the framework polled `AssetWatcher`, applied events to
    /// `ShaderDb` against `self.space()`, and any consumer pipelines you built may be
    /// stale. Rebuild what you care about.
    fn on_shader_reload(&mut self, _ctx: &mut SetupCtx<'_>) {}

    /// Render this frame. The framework has begun a frame and hands you the surface
    /// view; do whatever rendering you like. The framework calls `frame.present` after
    /// this returns.
    fn render(&mut self, rd: &RenderDevice, view: &wgpu::TextureView) -> anyhow::Result<()>;

    /// Build this frame's egui UI. Called after [`App::update`] and before
    /// [`App::render`]; the framework paints the resulting widgets as a 2D overlay on
    /// the surface view.
    ///
    /// Default impl is a no-op; apps that want UI override this with immediate-mode
    /// egui code:
    ///
    /// ```ignore
    /// fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
    ///     egui::Window::new("Settings").show(ctx, |ui| {
    ///         ui.add(egui::Slider::new(&mut self.fov, 30.0..=120.0));
    ///         if ui.button("Reset").clicked() { self.reset(); }
    ///     });
    /// }
    /// ```
    ///
    /// Gameplay code that reads input should gate on [`FrameCtx::ui_has_focus`] so a
    /// player typing into a settings field doesn't also fire WASD movement.
    ///
    /// For "egui label that follows a 3D object," use [`world_to_screen`] to project
    /// the world point and place an `egui::Area` at the resulting pixel.
    fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

    /// Title bar text. Default returns the static name `"rye app"`. Override for live
    /// FPS / state readouts; the framework rate-limits the actual `set_title` call to
    /// roughly once a second.
    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("rye app")
    }
}

// ---------------------------------------------------------------------------
// Context structs
// ---------------------------------------------------------------------------

/// Setup-phase context. Available during [`App::setup`] and [`App::on_shader_reload`].
pub struct SetupCtx<'a> {
    pub rd: &'a RenderDevice,
    pub shader_db: &'a mut ShaderDb,
    /// `None` when filesystem watching failed to initialise (e.g. no inotify on the
    /// running system); apps can still load shaders, but won't get hot-reload.
    pub watcher: Option<&'a mut AssetWatcher>,
    /// Wall-clock seconds since `run` was called. Always 0 in `setup`, non-zero on
    /// subsequent `on_shader_reload` calls.
    pub time: f32,
}

/// Per-tick context. Visible to [`App::tick`]. Deliberately GPU-free so sim code stays
/// bit-deterministic.
pub struct TickCtx {
    pub time: f32,
    pub tick: u64,
}

/// Per-frame context. Visible to [`App::update`] and [`App::on_event`]. Carries the
/// drained input, FPS readout, and the count of ticks the framework just executed.
pub struct FrameCtx<'a> {
    pub rd: &'a RenderDevice,
    pub input: FrameInput,
    pub time: f32,
    pub fps: f32,
    pub n_ticks: usize,
    pub tick: u64,
    /// `true` if egui is consuming pointer or keyboard input this frame (a widget is
    /// hovered, focused, or accepting text). Gameplay code should gate movement /
    /// mouselook on `!ctx.ui_has_focus` so typing into a settings field doesn't also
    /// fire WASD or rotate the camera.
    pub ui_has_focus: bool,
    /// Phantom for forward-compat: future fields here mustn't silently break code that
    /// pattern-matches on the struct.
    _non_exhaustive: PhantomData<()>,
}

// ---------------------------------------------------------------------------
// RunConfig
// ---------------------------------------------------------------------------

/// Runtime knobs. New fields land with defaults so adding configuration is non-breaking.
pub struct RunConfig {
    pub window: WindowAttributes,
    pub fixed_hz: u32,
    pub max_ticks_per_frame: usize,
    /// `EnvFilter`-style log filter. `None` means keep whatever `tracing-subscriber`
    /// was already configured with (or the `RUST_LOG` env var); `Some` installs a new
    /// global default subscriber.
    pub log_filter: Option<String>,
    /// When true (default) the framework exits the event loop on `Esc`. Apps that bind
    /// `Esc` to a gameplay action (pause, menu, modal dismiss) set this to false and
    /// handle the key inside [`App::on_event`].
    pub esc_exits: bool,
    /// Bail out after this many consecutive [`App::render`] errors. The last error
    /// surfaces back through [`run_with_config`]'s `Result` instead of looping forever
    /// on a wedged GPU. Reset to zero on any successful frame. `0` disables the budget.
    pub render_error_budget: u32,
    /// MSAA sample count requested for the scene + UI render target. `1` disables
    /// MSAA. `4` is the conventional default (good quality / cost tradeoff, supported
    /// on every consumer GPU). Higher counts (8, 16) cost more and yield diminishing
    /// returns on edge antialiasing. The runtime negotiates with the adapter; if the
    /// requested count isn't supported on the chosen surface format, [`RenderDevice`]
    /// falls back to the highest supported lower count and logs a warning.
    pub msaa_samples: u32,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            window: WindowAttributes::default()
                .with_title("rye app")
                .with_visible(false),
            fixed_hz: 60,
            max_ticks_per_frame: 4,
            log_filter: None,
            esc_exits: true,
            render_error_budget: 8,
            msaa_samples: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run an app with default config.
pub fn run<A: App>() -> anyhow::Result<()> {
    run_with_config::<A>(RunConfig::default())
}

/// Run an app with custom config.
pub fn run_with_config<A: App>(config: RunConfig) -> anyhow::Result<()> {
    if let Some(filter) = &config.log_filter {
        // Best-effort init; ignore "already initialised" errors so
        // running tests / repeated `run` calls don't panic.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new(filter.clone()))
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .try_init();
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut runner = Runner::<A>::new(config);
    event_loop.run_app(&mut runner)?;
    runner.finish()
}

// ---------------------------------------------------------------------------
// Runner: internal `ApplicationHandler` impl
// ---------------------------------------------------------------------------

struct Runner<A: App> {
    config: RunConfig,

    timestep: FixedTimestep,
    input: InputState,
    start: Instant,

    // Lazy-init: created in `resumed`.
    window: Option<Arc<Window>>,
    rd: Option<RenderDevice>,
    shader_db: Option<ShaderDb>,
    watcher: Option<AssetWatcher>,
    ui: Option<UiIntegration>,
    app: Option<A>,

    minimized: bool,

    // FPS bookkeeping.
    last_fps_update: Instant,
    frame_count: u32,
    fps: f32,

    tick_index: u64,
    /// Consecutive `App::render` failures since the last successful frame. Compared
    /// against `RunConfig::render_error_budget`.
    render_error_streak: u32,
    /// Surfaced to the user via `finish()` if the runner exited because of a setup or
    /// render error, so callers can propagate it from `main`.
    deferred_error: Option<anyhow::Error>,

    #[cfg(feature = "capture")]
    capture: capture::Capture,
}

impl<A: App> Runner<A> {
    fn new(config: RunConfig) -> Self {
        let timestep = FixedTimestep::new(config.fixed_hz);
        Self {
            config,
            timestep,
            input: InputState::default(),
            start: Instant::now(),
            window: None,
            rd: None,
            shader_db: None,
            watcher: None,
            ui: None,
            app: None,
            minimized: false,
            last_fps_update: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            tick_index: 0,
            render_error_streak: 0,
            deferred_error: None,

            #[cfg(feature = "capture")]
            capture: capture::Capture::new(),
        }
    }

    /// Drain any error that the runner deferred during the event loop (setup or render
    /// failures cause `elwt.exit()` so the loop returns `Ok`; we surface the real error
    /// here).
    fn finish(self) -> anyhow::Result<()> {
        match self.deferred_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

/// Read back `texture` and hand the pixels to the capture state machine, which
/// dispatches to the active writer (one-shot PNG, sequence PNG, or GIF encoder).
/// Logs and swallows errors so a transient capture failure doesn't abort the render
/// loop. Free function (not a method on Runner) so the borrow checker can see that
/// `&mut capture` and `&rd` are disjoint borrows.
#[cfg(feature = "capture")]
fn capture_consume(
    capture: &mut capture::Capture,
    rd: &RenderDevice,
    texture: &wgpu::Texture,
    is_pre: bool,
) {
    let img = match capture::read_texture_rgba(
        &rd.device,
        &rd.queue,
        texture,
        rd.surface_bundle.size.width,
        rd.surface_bundle.size.height,
        rd.surface_bundle.config.format,
    ) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!("capture: readback failed: {e:#}");
            return;
        }
    };
    if let Err(e) = capture.consume_frame(is_pre, img.rgba, img.width, img.height) {
        tracing::error!("capture: write failed: {e:#}");
    }
}

impl<A: App> ApplicationHandler for Runner<A> {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = match elwt.create_window(self.config.window.clone()) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("create_window: {e}"));
                elwt.exit();
                return;
            }
        };

        let rd = match pollster::block_on(RenderDevice::new(win.clone(), self.config.msaa_samples))
        {
            Ok(r) => r,
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("RenderDevice::new: {e:#}"));
                elwt.exit();
                return;
            }
        };
        // for debugging negotiated MSAA count; the actual count is in `rd.sample_count()`
        // tracing::info!("scale_factor: {}, msaa: {}x", win.scale_factor(), rd.sample_count());

        let mut shader_db = ShaderDb::new(rd.device.clone());

        // AssetWatcher init failure isn't fatal: apps still work
        // without hot-reload. Log and proceed.
        let mut watcher = match AssetWatcher::new() {
            Ok(w) => Some(w),
            Err(e) => {
                tracing::warn!("AssetWatcher disabled: {e}");
                None
            }
        };

        let mut ctx = SetupCtx {
            rd: &rd,
            shader_db: &mut shader_db,
            watcher: watcher.as_mut(),
            time: 0.0,
        };
        let app = match A::setup(&mut ctx) {
            Ok(a) => a,
            Err(e) => {
                self.deferred_error = Some(e.context("App::setup"));
                elwt.exit();
                return;
            }
        };

        // egui pipelines must be built with the same sample count as
        // the multisampled scene attachment, since both passes write
        // into the same color attachment and the deferred MSAA
        // resolve happens at the end of the egui paint pass. See
        // [`UiIntegration::paint`]'s `resolve_target` parameter.
        let ui = UiIntegration::new(
            &rd.device,
            &win,
            rd.surface_bundle.config.format,
            rd.sample_count(),
        );

        self.window = Some(win.clone());
        self.rd = Some(rd);
        self.shader_db = Some(shader_db);
        self.watcher = watcher;
        self.ui = Some(ui);
        self.app = Some(app);
        self.minimized = false;
        self.start = Instant::now();
        self.last_fps_update = Instant::now();

        win.set_visible(true);
        win.request_redraw();
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _id: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        let Some(win) = self.window.clone() else {
            return;
        };

        // Forward to egui first so it can claim hover/focus/clicks
        // before Rye's own routing translates the event for gameplay.
        // egui consuming the event is informational; Rye still sees it.
        if let Some(ui) = self.ui.as_mut() {
            let _ = ui.handle_event(&win, &ev);
        }

        // Esc / close: exit cleanly. (When egui has keyboard focus,
        // e.g. a TextEdit is active, swallow Esc so it dismisses the
        // edit instead of exiting the app.)
        match &ev {
            WindowEvent::CloseRequested => {
                elwt.exit();
                return;
            }
            WindowEvent::KeyboardInput { event, .. }
                if self.config.esc_exits
                    && event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && !self.ui.as_ref().is_some_and(|u| u.ui_has_focus()) =>
            {
                elwt.exit();
                return;
            }
            _ => {}
        }

        // Always route input *first*, before user `on_event` sees
        // it. Means apps can read derived state (e.g. via
        // `FrameCtx::input`) without re-implementing routing.
        match &ev {
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.key_input(event.physical_key, event.state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y);
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor_invalidated(),
            WindowEvent::Focused(false) => {
                self.input.cursor_invalidated();
                self.input.release_buttons();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.mouse_input(*button, *state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.mouse_wheel(*delta);
            }
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if !self.minimized {
                    if let Some(rd) = &mut self.rd {
                        rd.resize(*size);
                    }
                }
            }
            _ => {}
        }

        // Notify user of the event *after* our routing has settled.
        if let WindowEvent::RedrawRequested = ev {
            self.redraw(elwt, &win);
            return;
        }

        let now = self.time();
        let fps = self.fps;
        let tick = self.tick_index;
        if let Some(app) = self.app.as_mut() {
            if let Some(rd) = self.rd.as_ref() {
                let ui_has_focus = self.ui.as_ref().is_some_and(|u| u.ui_has_focus());
                let mut ctx = FrameCtx {
                    rd,
                    input: FrameInput::default(),
                    time: now,
                    fps,
                    n_ticks: 0,
                    tick,
                    ui_has_focus,
                    _non_exhaustive: PhantomData,
                };
                app.on_event(&ev, &mut ctx);
            }
        }
    }
}

impl<A: App> Runner<A> {
    fn redraw(&mut self, elwt: &ActiveEventLoop, win: &Arc<Window>) {
        if self.minimized {
            return;
        }
        let Some(rd) = self.rd.as_ref() else { return };

        // 1. Fixed-timestep ticks.
        let ticks = self.timestep.advance(Instant::now());
        let n_ticks = ticks.count();
        let n_capped = n_ticks.min(self.config.max_ticks_per_frame);
        let dt = 1.0 / self.config.fixed_hz as f32;
        if let Some(app) = self.app.as_mut() {
            for _ in 0..n_capped {
                let mut tctx = TickCtx {
                    time: self.start.elapsed().as_secs_f32(),
                    tick: self.tick_index,
                };
                app.tick(dt, &mut tctx);
                self.tick_index = self.tick_index.wrapping_add(1);
            }
        }

        // 2. Per-frame update with drained input + UI build.
        // egui's focus reading reflects the *previous* frame's state
        // (egui hasn't run yet for this frame). That one-frame
        // staleness is fine: focus changes one frame at a time, and
        // `App::update` needs to know "should I gate gameplay input"
        // before this frame's UI runs.
        let ui_has_focus = self.ui.as_ref().is_some_and(|u| u.ui_has_focus());
        let input = self.input.take_frame();
        if let Some(app) = self.app.as_mut() {
            let mut fctx = FrameCtx {
                rd,
                input,
                time: self.start.elapsed().as_secs_f32(),
                fps: self.fps,
                n_ticks: n_capped,
                tick: self.tick_index,
                ui_has_focus,
                _non_exhaustive: PhantomData,
            };
            app.update(&mut fctx);

            // Build this frame's UI. egui captures the widgets;
            // `paint` later renders them after `App::render`.
            if let Some(ui) = self.ui.as_mut() {
                let egui_ctx = ui.begin_frame(win.as_ref()).clone();
                app.ui(&egui_ctx, &mut fctx);
            }
        }

        // 3. Hot-reload poll.
        let reload_events = self.watcher.as_mut().map(|w| w.poll()).unwrap_or_default();
        if !reload_events.is_empty() {
            if let (Some(app), Some(shader_db), Some(rd)) =
                (self.app.as_mut(), self.shader_db.as_mut(), self.rd.as_ref())
            {
                shader_db.apply_events(&reload_events, app.space());
                let mut ctx = SetupCtx {
                    rd,
                    shader_db,
                    watcher: self.watcher.as_mut(),
                    time: self.start.elapsed().as_secs_f32(),
                };
                app.on_shader_reload(&mut ctx);
            }
        }

        // 4. FPS + title (rate-limited to ~1 Hz).
        self.frame_count += 1;
        let elapsed = self.last_fps_update.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_fps_update = Instant::now();
            if let Some(app) = self.app.as_ref() {
                let title = app.title(self.fps);
                // Append capture status when active. 1 Hz refresh matches the title
                // update cadence and gives the user a visible recording counter without
                // wiring it through the demo's UI.
                #[cfg(feature = "capture")]
                let title = match self.capture.status() {
                    Some(status) => format!("{title} [{status}]").into(),
                    None => title,
                };
                win.set_title(&title);
            }
        }

        // 5. Drain any queued capture requests + update the state machine BEFORE the
        // render pass. Requests come from console commands and hotkey binds; they're
        // applied here so this frame can honor them.
        #[cfg(feature = "capture")]
        {
            let requests = capture::drain_requests();
            if !requests.is_empty() {
                let log = self.capture.apply_requests(requests);
                for line in log {
                    tracing::info!("{line}");
                }
            }
        }

        // 6. Render: scene (App::render) then UI overlay.
        //
        // When MSAA is enabled, both passes write into the
        // multisampled color attachment (`rd.msaa_view()`) and the
        // egui pass attaches the swapchain view as `resolve_target`
        // so the deferred MSAA resolve happens at the end of the
        // egui pass. When MSAA is disabled, both passes write
        // directly into the swapchain view and `resolve_target` is
        // `None`.
        //
        // Capture taps:
        //   - `pre`-egui:  after App::render, before ui.paint. MSAA must be off (the
        //     multisampled attachment isn't directly copyable). The pre tap reads the
        //     swapchain view, which at this point contains just the 3D pass output.
        //   - `post`-egui: after ui.paint, before frame.present. Reads the swapchain
        //     view, which contains the final composite (and the MSAA resolve target
        //     when MSAA is on). This is what DWM receives.
        // FPS-gate decides whether either tap fires this frame. Computed once before the
        // render pass so the same `now` is used to schedule the next capture interval.
        #[cfg(feature = "capture")]
        let capture_now = Instant::now();
        #[cfg(feature = "capture")]
        let do_capture = self.capture.should_capture(capture_now);

        match rd.begin_frame() {
            Ok((frame, swap_view)) => {
                let mut last_err: Option<anyhow::Error> = None;
                let render_view = rd.msaa_view().unwrap_or(&swap_view);
                if let Some(app) = self.app.as_mut() {
                    if let Err(e) = app.render(rd, render_view) {
                        tracing::error!("App::render error: {e:#}");
                        last_err = Some(e);
                    }
                }

                // Pre-egui capture tap. Only valid with MSAA off (see above).
                #[cfg(feature = "capture")]
                if do_capture && self.capture.wants_pre() {
                    if rd.sample_count() > 1 {
                        tracing::warn!(
                            "capture: `pre` stage skipped because MSAA is on; \
                             set RunConfig::msaa_samples = 1 for diagnostic capture"
                        );
                    } else {
                        capture_consume(&mut self.capture, rd, &frame.texture, true);
                    }
                }

                if let Some(ui) = self.ui.as_mut() {
                    let mut encoder =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rye-app::ui-paint"),
                            });
                    let viewport = (rd.surface_bundle.size.width, rd.surface_bundle.size.height);
                    let resolve_target = (rd.sample_count() > 1).then_some(&swap_view);
                    ui.paint(
                        &rd.device,
                        &rd.queue,
                        &mut encoder,
                        render_view,
                        resolve_target,
                        win.as_ref(),
                        viewport,
                    );
                    rd.queue.submit(Some(encoder.finish()));
                }

                // Post-egui capture tap. Always valid: swapchain has final composite.
                #[cfg(feature = "capture")]
                if do_capture && self.capture.wants_post() {
                    capture_consume(&mut self.capture, rd, &frame.texture, false);
                }
                #[cfg(feature = "capture")]
                if do_capture {
                    self.capture.advance_frame(capture_now);
                }
                // Publish status every frame so the panel + window title stay current
                // even when do_capture is false (FPS-gated idle frames between writes).
                #[cfg(feature = "capture")]
                capture::publish_status(self.capture.status());

                frame.present();
                if let Some(err) = last_err {
                    self.render_error_streak = self.render_error_streak.saturating_add(1);
                    let budget = self.config.render_error_budget;
                    if budget > 0 && self.render_error_streak >= budget {
                        self.deferred_error = Some(err.context(format!(
                            "App::render failed {budget} consecutive frames; aborting"
                        )));
                        elwt.exit();
                        return;
                    }
                } else {
                    self.render_error_streak = 0;
                }
                win.request_redraw();
            }
            Err(err) => match err {
                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                    if let Some(rd) = &mut self.rd {
                        let size = rd.surface_bundle.size;
                        rd.resize(size);
                    }
                    win.request_redraw();
                }
                wgpu::SurfaceError::Timeout => win.request_redraw(),
                wgpu::SurfaceError::OutOfMemory => {
                    self.deferred_error = Some(anyhow::anyhow!("wgpu surface out of memory"));
                    elwt.exit();
                }
                wgpu::SurfaceError::Other => {
                    tracing::error!("surface error: {err:?}");
                    win.request_redraw();
                }
            },
        }
    }
}
