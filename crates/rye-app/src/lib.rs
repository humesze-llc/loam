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
// `web_time::Instant` is a drop-in for `std::time::Instant` that works on both native
// (re-exports the std type verbatim) and wasm32 (backs it with `performance.now()`).
// `std::time::Instant::now` panics on wasm32, so the swap is mandatory for the browser
// runtime path.
use web_time::Instant;

// Capture module dispatch: real pipeline on native with `capture` feature on; stub
// API everywhere else (wasm, or `--no-default-features` lean native builds). The two
// files expose the same public surface so demos don't need `cfg` gates at their
// `rye_app::capture::*` call sites.
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
pub mod capture;

#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
#[path = "capture_stub.rs"]
pub mod capture;
pub mod args;
pub mod fps;
pub mod frame_pacing;
pub mod keymap;
pub mod log;
pub mod trace;
pub mod vsync;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

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

    /// **Legacy render path.** Implement either this OR `App::record`; the runner
    /// always calls `record`, whose default impl calls this. Each invocation of
    /// `render` typically creates its own command encoder + queue.submit (per pass
    /// or per node), so a demo with three nodes pays at least three submits per
    /// frame. On wasm32, each submit crosses the JS boundary and adds compositor
    /// latency; the `App::record` path lets the runner batch the demo's draws
    /// with ui-paint + composite into a single per-frame submit.
    ///
    /// New demos should override `record` instead. This method remains for
    /// backwards-compatibility with the broad set of examples that built against
    /// the original `App` trait; replacing them all in one sweep is bigger than
    /// the engine's blast-radius budget for now.
    fn render(&mut self, _rd: &RenderDevice, _view: &wgpu::TextureView) -> anyhow::Result<()> {
        Ok(())
    }

    /// **Preferred render path.** The runner owns one command encoder for the entire
    /// frame, shares it with the demo via [`RenderCtx::encoder`], then continues
    /// using the same encoder for ui-paint and the wasm-side gamma composite.
    /// Everything reaches the GPU in a single `queue.submit`.
    ///
    /// Default impl falls back to [`App::render`] (the legacy multi-submit path),
    /// so the migration is opt-in: override `record` to get the single-submit
    /// behaviour. The runner doesn't care which method you override.
    ///
    /// Implementation contract:
    ///
    /// - **Do NOT call `encoder.finish()` or `queue.submit(...)`**: the runner
    ///   does that exactly once at end-of-frame. Submitting prematurely splits
    ///   the frame's work and defeats the optimization.
    /// - **Multiple render passes per call are fine**: open a render pass on the
    ///   encoder, draw, drop the pass; open the next, etc. wgpu serializes them
    ///   correctly within the same encoder.
    /// - **Use `ctx.view` as the color target** for the bulk of your scene draws.
    ///   The runner has already selected the right view (MSAA / scene-target /
    ///   swapchain) based on platform + capture state.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> anyhow::Result<()> {
        // Legacy adapter: invoke `render` with the (rd, view) pair, ignoring the
        // shared encoder. Old demos that override `render` keep working; the
        // runner still does one extra submit at end-of-frame for ui-paint +
        // composite, which is a wash with the old code's separate encoders for
        // those passes (slight net win).
        self.render(ctx.rd, ctx.view)
    }

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

/// Render-time context. Handed to `App::record` each frame. Owns a shared command
/// encoder that the demo writes its scene passes into; the runner reuses the same
/// encoder for ui-paint and the wasm-side composite, then submits it exactly once
/// at end of frame.
///
/// `view` is the color target the demo's main scene passes should write into. It's
/// the runner's "best target right now": MSAA view on native+MSAA-on, the offscreen
/// sRGB scene texture on wasm, the swapchain view otherwise. The demo doesn't need
/// to know which case applies; pipelines built with [`RenderDevice::target_format`]
/// + [`RenderDevice::sample_count`] match this view automatically.
pub struct RenderCtx<'a> {
    pub rd: &'a RenderDevice,
    pub view: &'a wgpu::TextureView,
    /// Shared command encoder. Open render passes on it, draw, drop the pass; do
    /// NOT call `finish()` or `queue.submit`. The runner does that at end of frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
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
    /// Wall-clock seconds since the previous `App::update` call. Use this for
    /// variable-rate visual animation (camera smoothing, hover bobs, particles,
    /// continuous rotors driven by user-perceived time). For deterministic sim
    /// state that must be lockstep-reproducible, use [`App::tick`] instead;
    /// `tick`'s `dt` is the fixed-timestep interval regardless of frame rate.
    ///
    /// First call after setup gets `dt = 1.0 / RunConfig::fixed_hz` as a sensible
    /// fallback (no prior frame to measure from). Subsequent calls reflect actual
    /// elapsed time, so a 50fps frame gets dt ≈ 0.02 and a stutter-frame at
    /// 15fps gets dt ≈ 0.066.
    pub dt: f32,
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
    /// Wasm-specific knobs (DOM IDs the page exposes). Ignored on native. The
    /// defaults match the standard layout rye demos use (`rye-canvas-host`
    /// container, `rye-launch` button, `rye-canvas` canvas); demos that ship
    /// a different HTML layout override here.
    pub wasm: WasmConfig,
}

/// Wasm-only configuration knobs: the DOM element IDs the page uses to host
/// the demo. Defaults match the standard layout in the engine's example
/// `index.html` templates. Demos that need custom IDs override the fields
/// they care about and `..WasmConfig::default()` the rest.
#[derive(Clone)]
pub struct WasmConfig {
    /// Container element with `data-mode="manual"`. Determines whether the
    /// demo enters click-to-start mode (vs auto-launch on page load).
    pub host_id: String,
    /// Launch button. Click handler transfers the canvas to a worker.
    pub button_id: String,
    /// Canvas the worker renders into. The element is
    /// `transferControlToOffscreen()`-ed to the worker on click.
    pub canvas_id: String,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            host_id: "rye-canvas-host".into(),
            button_id: "rye-launch".into(),
            canvas_id: "rye-canvas".into(),
        }
    }
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
            wasm: WasmConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run a demo. The unified entry point that handles native + wasm
/// (both main-thread fallback and worker mode) in one call.
///
/// On native, this is equivalent to [`run_with_config`]: the function
/// blocks until the event loop exits and returns the deferred error
/// (or `Ok(())`).
///
/// On wasm32:
/// - When invoked inside a `DedicatedWorkerGlobalScope` (worker
///   context), routes to `wasm::worker::run`. Drives the App's
///   lifecycle on a worker-side RAF loop with the `wasm::worker_ui::WorkerUi`
///   egui integration.
/// - When invoked on main thread AND the page's `host_id` element has
///   `data-mode="manual"`, routes to `wasm::launch_on_click`. Wires
///   the launch button to spawn the worker on click.
/// - When invoked on main thread WITHOUT manual mode, falls back to
///   [`run_with_config`] (the legacy windowed-mode wasm path).
///
/// The single function call replaces ~8 lines of dispatch boilerplate
/// in each demo's `main()`. Demos that need finer control over the
/// dispatch (e.g. inspecting `wasm::is_worker_context()` for setup-time
/// side effects) can still call the lower-level entry points directly.
pub fn run<A: App + 'static>(config: RunConfig) -> anyhow::Result<()>
where
    A::Space: 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        if wasm::is_worker_context() {
            return wasm::worker::run::<A>();
        }
        if wasm::launch::is_manual_mode(&config.wasm.host_id) {
            return wasm::launch_on_click(
                &config.wasm.host_id,
                &config.wasm.button_id,
                &config.wasm.canvas_id,
            );
        }
        // Fall through to main-thread auto-launch (legacy wasm path).
    }
    run_with_config::<A>(config)
}

/// Run an app with custom config.
///
/// On native the function blocks until the event loop exits, then returns whatever
/// error the runner deferred (or `Ok(())`). On wasm32 there is no blocking event loop;
/// the function returns `Ok(())` synchronously after handing the runner off to
/// `EventLoopExtWebSys::spawn_app`, which keeps a reference alive on the JS heap and
/// drives the loop via `requestAnimationFrame`. The browser's JS runtime owns
/// lifecycle from that point; deferred errors are surfaced through `tracing::error!`
/// (and the console panic hook for unwinding) rather than a return value.
pub fn run_with_config<A: App>(config: RunConfig) -> anyhow::Result<()> {
    // Compose two tracing layers: the standard fmt layer (writes to stdout) and our
    // ConsoleLayer (pushes events into the in-process ring buffer for the dev
    // console). Both subscribe to the same EnvFilter so RUST_LOG / log_filter
    // controls both outputs uniformly. `try_init` is best-effort: if a subscriber is
    // already installed (tests, repeated calls) we silently no-op so the existing
    // sink keeps working.
    //
    // On wasm32 stdout doesn't exist and the env-filter has no `RUST_LOG` to read; we
    // route tracing events into the browser console via `tracing-wasm` instead, and
    // install the panic hook so a Rust panic surfaces a useful stack trace in
    // devtools rather than the default `unreachable executed` from `wasm-bindgen`.
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
        // Wire `performance.memory.usedJSHeapSize` (Chromium-only) into
        // frame_trace so each completed frame carries a signed heap delta and
        // spike-warns include `heap_delta=+24.5MB` style annotations. On
        // Firefox / Safari the sampler returns `None` and the field stays
        // empty (no misleading reads).
        rye_time::frame_trace::set_heap_sampler(wasm::js_heap_sampler);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let filter = match &config.log_filter {
            Some(s) => tracing_subscriber::EnvFilter::new(s.clone()),
            None => tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        };
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .with(log::ConsoleLayer)
            .try_init();
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let runner = Runner::<A>::new(config);

    #[cfg(target_arch = "wasm32")]
    {
        // `spawn_app` consumes the runner, parks it on the JS heap, and returns
        // immediately. Errors from the runner (deferred via `self.deferred_error` on
        // setup / render failure) are not visible to this call site because there's no
        // return path to bubble them up to JS; they surface through `tracing::error!`
        // -> browser console instead.
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(runner);
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut runner = runner;
        event_loop.run_app(&mut runner)?;
        runner.finish()
    }
}

// ---------------------------------------------------------------------------
// Runner: internal `ApplicationHandler` impl
// ---------------------------------------------------------------------------

/// Everything the runner needs after device acquisition has completed. Bundled so the
/// native (synchronous) and wasm32 (async via `spawn_local`) paths can both produce a
/// single value that the runner installs into its own fields atomically.
struct InitArtifacts<A: App> {
    rd: RenderDevice,
    shader_db: ShaderDb,
    watcher: Option<AssetWatcher>,
    ui: UiIntegration,
    app: A,
}

/// Sample count is part of the contract between `RenderDevice` (the multisampled scene
/// attachment) and `UiIntegration` (which builds egui pipelines against the same sample
/// count). Building `UiIntegration` here, in the same place A::setup runs, keeps that
/// pairing colocated so the two can't drift.
///
/// Free function (not a method on Runner) so it can be called from the wasm `spawn_local`
/// closure where `&mut Runner` isn't available across the await point.
fn setup_after_device<A: App>(
    win: &Arc<Window>,
    rd: RenderDevice,
) -> anyhow::Result<InitArtifacts<A>> {
    let mut shader_db = ShaderDb::new(rd.device.clone());

    // AssetWatcher init failure isn't fatal: apps still work without hot-reload. Log
    // and proceed. On wasm32 the watcher is a no-op stub (see `rye-asset`'s watcher.rs)
    // so this always succeeds and the `.warn` branch is dead code in the browser.
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
    let app = A::setup(&mut ctx).map_err(|e| e.context("App::setup"))?;

    // egui pipelines must be built with the same sample count as the multisampled
    // scene attachment, since both passes write into the same color attachment and the
    // deferred MSAA resolve happens at the end of the egui paint pass. See
    // [`UiIntegration::paint`]'s `resolve_target` parameter.
    let mut ui = UiIntegration::new(
        &rd.device,
        win,
        rd.target_format(),
        rd.sample_count(),
    );

    // Runner-side pipeline warming (N3). Forces lazy pipeline compilation for
    // egui-wgpu's shape variants and the browser-WebGPU composite pass during
    // setup, instead of stalling the user's first visible frame for ~50-200ms
    // per first-touched pipeline. App-owned pipelines are warmed inside the
    // app's `setup` (e.g. `tesseract_demo::warm_pipelines`); these two cover
    // the runner-owned ones every demo benefits from.
    //
    // Architectural note: lives here (in setup_after_device, after both ui +
    // rd exist but before the first redraw) so it's truly part of the setup
    // step, not a per-frame check + first-frame-flag pattern. A demo that
    // skips the runner (does its own event loop) would also skip this warm,
    // but no rye demo does that today.
    ui.warm_pipelines(
        &rd.device,
        &rd.queue,
        win,
        rd.target_format(),
        rd.sample_count(),
    );
    rd.warm_composite();

    Ok(InitArtifacts {
        rd,
        shader_db,
        watcher,
        ui,
        app,
    })
}

/// On wasm32 the device-acquisition future runs to completion in a JS microtask and
/// hands its result back to the runner through this slot. The runner polls the slot at
/// the top of every event-loop callback (window_event + redraw) and installs the
/// artifacts when they appear.
///
/// `Rc<RefCell<...>>` is fine: wasm32 is single-threaded, the future and the runner
/// never borrow the cell simultaneously (the future borrows it once, on completion, to
/// move the result in; the runner borrows it on each callback to try to take the
/// result out).
#[cfg(target_arch = "wasm32")]
type PendingInit<A> = std::rc::Rc<
    std::cell::RefCell<Option<anyhow::Result<InitArtifacts<A>>>>,
>;

/// Attach the canvas backing the winit window to the page's DOM. Without this the
/// canvas exists only as a JS object the surface can target but nothing the user can
/// see; appending it makes the render output visible and lets pointer / keyboard
/// events flow through.
///
/// Host element selection prefers `#rye-canvas-host` when the page provides one
/// (Trunk-generated pages typically have a dedicated container so CSS can layout
/// around the canvas); falls back to `<body>` so a minimal page without any host
/// element still works. Canvas style is set to fill its parent so a flex / grid /
/// percentage-sized container drives the surface size; the next resize observer
/// hookup (TODO) will then forward `ResizeObserver` fires to `winit::WindowEvent`.
#[cfg(target_arch = "wasm32")]
fn attach_canvas_to_dom(win: &winit::window::Window) -> anyhow::Result<()> {
    use winit::platform::web::WindowExtWebSys;

    let canvas = win
        .canvas()
        .ok_or_else(|| anyhow::anyhow!("winit window has no canvas (wasm32 only)"))?;

    let web_window =
        web_sys::window().ok_or_else(|| anyhow::anyhow!("no global `window` object"))?;
    let document = web_window
        .document()
        .ok_or_else(|| anyhow::anyhow!("no `document` on global window"))?;

    let host: web_sys::Element = match document.get_element_by_id("rye-canvas-host") {
        Some(el) => el,
        None => document
            .body()
            .map(Into::into)
            .ok_or_else(|| anyhow::anyhow!(
                "no canvas host: page is missing both `#rye-canvas-host` and `<body>`"
            ))?,
    };

    // Fill the host. Without these the canvas keeps winit's default intrinsic size
    // (typically 1024x768) which usually disagrees with the page layout.
    let style = canvas.style();
    let _ = style.set_property("width", "100%");
    let _ = style.set_property("height", "100%");
    let _ = style.set_property("display", "block");

    host.append_child(&canvas)
        .map_err(|e| anyhow::anyhow!("append canvas to host: {e:?}"))?;

    Ok(())
}

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

    /// Wasm32-only: present while the spawned device-acquisition future is in flight;
    /// taken on completion. `None` after the first successful poll (or before `resumed`
    /// has fired). See `PendingInit` for the design rationale.
    #[cfg(target_arch = "wasm32")]
    pending_init: Option<PendingInit<A>>,

    minimized: bool,

    // FPS bookkeeping.
    last_fps_update: Instant,
    frame_count: u32,
    fps: f32,

    /// Timestamp of the previous `App::update` call, used to compute `FrameCtx::dt`
    /// (wall-clock elapsed since the last update). `None` before the first frame so
    /// the first `dt` falls back to the fixed-timestep interval.
    last_update_at: Option<Instant>,

    tick_index: u64,
    /// Timestamp of the previous `redraw` entry. Used by the [`frame_pacing`]
    /// throttle at the top of each `redraw` to decide whether to sleep (native)
    /// or skip-and-rerequest (wasm) before doing this frame's work. `None`
    /// disables throttling until the first frame establishes a reference.
    last_redraw_at: Option<Instant>,
    /// Consecutive `App::render` failures since the last successful frame. Compared
    /// against `RunConfig::render_error_budget`.
    render_error_streak: u32,
    /// Surfaced to the user via `finish()` if the runner exited because of a setup or
    /// render error, so callers can propagate it from `main`.
    deferred_error: Option<anyhow::Error>,

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
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
            #[cfg(target_arch = "wasm32")]
            pending_init: None,
            minimized: false,
            last_fps_update: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            last_update_at: None,
            tick_index: 0,
            last_redraw_at: None,
            render_error_streak: 0,
            deferred_error: None,

            #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
            capture: capture::Capture::new(),
        }
    }

    /// Drain any error that the runner deferred during the event loop (setup or render
    /// failures cause `elwt.exit()` so the loop returns `Ok`; we surface the real error
    /// here).
    ///
    /// Native-only: on wasm32 the runner is consumed by
    /// `EventLoopExtWebSys::spawn_app` and its lifetime is owned by the JS heap, so
    /// there's no return path to surface deferred errors through. They bubble up via
    /// `tracing::error!` -> the browser console instead.
    #[cfg(not(target_arch = "wasm32"))]
    fn finish(self) -> anyhow::Result<()> {
        match self.deferred_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    /// Install artifacts produced by `setup_after_device`. Shared by both the native
    /// synchronous path (called from `resumed` directly) and the wasm32 async path
    /// (called from `poll_pending_init` when the future resolves).
    fn install_init(&mut self, win: Arc<Window>, artifacts: InitArtifacts<A>) {
        self.window = Some(win.clone());
        self.rd = Some(artifacts.rd);
        self.shader_db = Some(artifacts.shader_db);
        self.watcher = artifacts.watcher;
        self.ui = Some(artifacts.ui);
        self.app = Some(artifacts.app);
        self.minimized = false;
        self.start = Instant::now();
        self.last_fps_update = Instant::now();

        win.set_visible(true);
        win.request_redraw();
    }

    /// Wasm32-only: try to install the deferred init artifacts. Returns `true` if the
    /// state transitioned from Loading -> Ready (or Loading -> Failed) on this call.
    /// Called at the top of every event-loop callback so the runner can finish
    /// constructing its state as soon as `RenderDevice::new` resolves.
    #[cfg(target_arch = "wasm32")]
    fn poll_pending_init(&mut self, elwt: &ActiveEventLoop) -> bool {
        let Some(cell) = self.pending_init.as_ref() else {
            return false;
        };
        let Some(result) = cell.borrow_mut().take() else {
            return false;
        };
        // Drop the shared cell now that we've consumed its payload; the future itself
        // has already completed.
        self.pending_init = None;
        let Some(win) = self.window.clone() else {
            // Shouldn't happen: `resumed` always sets `self.window` before spawning the
            // future, but be defensive.
            self.deferred_error = Some(anyhow::anyhow!(
                "wasm init future resolved with no window present",
            ));
            elwt.exit();
            return true;
        };
        match result {
            Ok(artifacts) => {
                self.install_init(win, artifacts);
                true
            }
            Err(e) => {
                self.deferred_error = Some(e);
                elwt.exit();
                true
            }
        }
    }
}

/// Read back `texture` and hand the pixels to the capture state machine, which
/// dispatches to the active writer (one-shot PNG, sequence PNG, or GIF encoder).
/// Logs and swallows errors so a transient capture failure doesn't abort the render
/// loop. Free function (not a method on Runner) so the borrow checker can see that
/// `&mut capture` and `&rd` are disjoint borrows.
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
fn capture_consume(
    capture: &mut capture::Capture,
    rd: &RenderDevice,
    texture: &wgpu::Texture,
    is_pre: bool,
    captured_at: Instant,
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
    if let Err(e) = capture.consume_frame(is_pre, img.rgba, img.width, img.height, captured_at) {
        tracing::error!("capture: write failed: {e:#}");
    }
}

impl<A: App> ApplicationHandler for Runner<A> {
    #[cfg(target_arch = "wasm32")]
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        // Wasm32 init is a two-phase dance:
        //
        //   1. Synchronous prelude: create the winit `Window` (this is sync on every
        //      platform; on wasm it constructs an `HtmlCanvasElement` but does NOT
        //      attach it to the DOM, so we do that explicitly). Then we hand a clone
        //      of `Arc<Window>` to the async tail.
        //   2. Async tail: `RenderDevice::new` awaits `request_adapter` /
        //      `request_device`, both of which are JS-promise-backed on wasm. We hand
        //      that future to `wasm_bindgen_futures::spawn_local` and write the
        //      eventual `InitArtifacts` into a shared `Rc<RefCell<...>>` slot.
        //
        // `poll_pending_init` (called at the top of every event-loop callback) then
        // drains the slot and installs the artifacts into `self`.
        //
        // `with_prevent_default(false)` tells winit not to call
        // `event.preventDefault()` on every keyboard / mouse / wheel event the canvas
        // receives. Default-on capture made Ctrl+R / F12 / Ctrl+Shift+I unreachable
        // (the canvas swallowed them before browser chrome could see them); turning
        // it off means winit only consumes the events it actually translates into
        // `WindowEvent`s. App-relevant keys (Esc, arrows, etc.) still flow through
        // the input system because winit listens on those passively.
        use winit::platform::web::WindowAttributesExtWebSys;
        let attrs = self.config.window.clone().with_prevent_default(false);
        let win = match elwt.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("create_window: {e}"));
                elwt.exit();
                return;
            }
        };

        if let Err(e) = attach_canvas_to_dom(&win) {
            self.deferred_error = Some(e.context("attach canvas to DOM"));
            elwt.exit();
            return;
        }

        let msaa = self.config.msaa_samples;
        let win_for_future = win.clone();
        let cell: PendingInit<A> = std::rc::Rc::new(std::cell::RefCell::new(None));
        let cell_for_future = cell.clone();

        self.window = Some(win);
        self.pending_init = Some(cell);

        wasm_bindgen_futures::spawn_local(async move {
            let result = async {
                let rd = RenderDevice::new(win_for_future.clone(), msaa)
                    .await
                    .map_err(|e| anyhow::anyhow!("RenderDevice::new: {e:#}"))?;
                setup_after_device::<A>(&win_for_future, rd)
            }
            .await;
            *cell_for_future.borrow_mut() = Some(result);
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
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

        let artifacts = match setup_after_device::<A>(&win, rd) {
            Ok(a) => a,
            Err(e) => {
                self.deferred_error = Some(e);
                elwt.exit();
                return;
            }
        };

        self.install_init(win, artifacts);
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _id: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        // Wasm32: drain the deferred-init slot if the spawned device-acquisition future
        // has resolved. On native this is a no-op (the slot doesn't exist).
        #[cfg(target_arch = "wasm32")]
        let _installed = self.poll_pending_init(elwt);

        let Some(win) = self.window.clone() else {
            return;
        };

        // Event-correlation diagnostic. When `log events on` has been issued
        // (see rye_app::log), every meaningful WindowEvent emits a
        // tracing::info! so the spike investigation can cross-reference
        // browser events with the spike-warn timestamps. Cursor-moves are
        // filtered because they fire at 60Hz+ and drown out the signal;
        // everything else goes through. Lives BEFORE the egui forward so
        // RedrawRequested doesn't spam (RedrawRequested fires every frame
        // and would obscure the rare events we care about).
        if log::events_enabled() {
            match &ev {
                // Suppress per-frame noise.
                WindowEvent::CursorMoved { .. }
                | WindowEvent::RedrawRequested
                | WindowEvent::AxisMotion { .. } => {}
                other => {
                    tracing::info!("WindowEvent: {other:?}");
                }
            }
        }

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
                    // dt isn't meaningful for input events (they fire whenever the OS
                    // delivers, not on a frame cadence). Zero is the least-surprising
                    // value; apps that integrate continuous state should do that work
                    // in `update`, not `on_event`.
                    dt: 0.0,
                    ui_has_focus,
                    _non_exhaustive: PhantomData,
                };
                app.on_event(&ev, &mut ctx);
            }
        }
    }

    /// Fires after every event batch when `ControlFlow::Poll` is set. On wasm32 we use
    /// it to drain the deferred-init slot: the spawned device-acquisition future may
    /// have resolved between callbacks, and without a user-driven event the runner
    /// would otherwise sit idle. Once installed, the normal redraw cycle takes over.
    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.pending_init.is_some() {
                self.poll_pending_init(_elwt);
            }
        }
    }
}

impl<A: App> Runner<A> {
    fn redraw(&mut self, elwt: &ActiveEventLoop, win: &Arc<Window>) {
        if self.minimized {
            return;
        }
        // Pending vsync transitions from the `vsync` console command. Applied
        // here so the next `begin_frame` picks up the new present mode. The
        // off-side picks the best non-Fifo mode the adapter advertised:
        // `Mailbox` first (triple-buffered, no tearing), `Immediate` as
        // fallback (single-buffered, tearing allowed). If neither is offered
        // (typical browser surface), the request silently no-ops; surface
        // configuration is the wrong layer to surface an error in that case.
        if let (Some(want_on), Some(rd)) =
            (frame_pacing::take_pending_vsync(), self.rd.as_mut())
        {
            let target = if want_on {
                wgpu::PresentMode::Fifo
            } else {
                let modes = rd.supported_present_modes();
                if modes.contains(&wgpu::PresentMode::Mailbox) {
                    wgpu::PresentMode::Mailbox
                } else if modes.contains(&wgpu::PresentMode::Immediate) {
                    wgpu::PresentMode::Immediate
                } else {
                    rd.present_mode()
                }
            };
            let _ = rd.set_present_mode(target);
        }
        let Some(rd) = self.rd.as_ref() else { return };

        // Frame-rate cap. The `fps` console command pokes
        // [`frame_pacing::set_target_fps`]; we read it here. Cap is enforced
        // differently per target; native does a precise sleep up to the
        // deadline; wasm skips the RAF callback and re-requests, since we
        // can't block in the browser. With `target_fps = 0` the load returns
        // `None` and we fall through to the surface's native cadence (vsync
        // on native, RAF on wasm).
        //
        // We anchor the deadline on the previous frame's ideal start (not on
        // the actual wake-up) so the cadence stays locked to the period even
        // if individual frames overshoot. If we ran long (work + present took
        // longer than the period), we set `last_redraw_at = now` to "catch
        // up" instead of falling further behind on every subsequent frame.
        let mut frame_anchor = Instant::now();
        if let (Some(period), Some(last)) =
            (frame_pacing::target_period(), self.last_redraw_at)
        {
            let deadline = last + period;
            if frame_anchor < deadline {
                #[cfg(target_arch = "wasm32")]
                {
                    win.request_redraw();
                    return;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    frame_pacing::precise_sleep_until(deadline);
                    frame_anchor = deadline;
                }
            }
        }
        self.last_redraw_at = Some(frame_anchor);

        // Mark frame start for `idle` measurement. `end_frame` subtracts this from
        // the previous `end_frame` timestamp to get `idle` (browser/RAF gap, not
        // our work); separate from `between-frames` (total cadence, our work +
        // idle combined).
        rye_time::frame_trace::begin_frame();

        // Whole-redraw scope. Subsequent named sections sum to less than this; the
        // delta is the small bits in between (FPS bookkeeping, capture status
        // publish, etc.). Used by the `trace` console command to surface "total
        // CPU work this frame" vs. "what's the dominant section."
        let _frame_scope = rye_time::frame_trace::scope("frame");

        // 1. Fixed-timestep ticks.
        let n_capped;
        {
            let _scope = rye_time::frame_trace::scope("sim-ticks");
            let ticks = self.timestep.advance(Instant::now());
            let n_ticks = ticks.count();
            n_capped = n_ticks.min(self.config.max_ticks_per_frame);
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
        }

        // 2. Per-frame update with drained input + UI build.
        // egui's focus reading reflects the *previous* frame's state
        // (egui hasn't run yet for this frame). That one-frame
        // staleness is fine: focus changes one frame at a time, and
        // `App::update` needs to know "should I gate gameplay input"
        // before this frame's UI runs.
        let ui_has_focus = self.ui.as_ref().is_some_and(|u| u.ui_has_focus());
        let input = self.input.take_frame();

        // Compute dt (wall-clock seconds since previous update). First frame after
        // setup has no prior `last_update_at`, so we seed with the fixed-timestep
        // interval; better than 0.0, which would zero out any dt-driven animation
        // on its very first integration step.
        let now_inst = Instant::now();
        let dt = match self.last_update_at {
            Some(prev) => now_inst.saturating_duration_since(prev).as_secs_f32(),
            None => 1.0 / self.config.fixed_hz as f32,
        };
        self.last_update_at = Some(now_inst);

        if let Some(app) = self.app.as_mut() {
            let mut fctx = FrameCtx {
                rd,
                input,
                time: self.start.elapsed().as_secs_f32(),
                fps: self.fps,
                n_ticks: n_capped,
                tick: self.tick_index,
                dt,
                ui_has_focus,
                _non_exhaustive: PhantomData,
            };
            {
                let _scope = rye_time::frame_trace::scope("app-update");
                app.update(&mut fctx);
            }

            // Build this frame's UI. egui captures the widgets;
            // `paint` later renders them after `App::render`.
            if let Some(ui) = self.ui.as_mut() {
                let _scope = rye_time::frame_trace::scope("app-ui");
                let egui_ctx = ui.begin_frame(win.as_ref()).clone();
                app.ui(&egui_ctx, &mut fctx);
            }
        }

        // 3. Hot-reload poll.
        {
            let _scope = rye_time::frame_trace::scope("hot-reload");
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
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
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
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
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
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let capture_now = Instant::now();
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let do_capture = self.capture.should_capture(capture_now);

        match rd.begin_frame() {
            Ok((frame, swap_view)) => {
                let mut last_err: Option<anyhow::Error> = None;
                // Render-target priority chain:
                //   1. MSAA view (native, MSAA on): scene + UI render multisampled,
                //      resolve at egui paint pass's resolve_target = swap_view.
                //   2. Scene view (browser, non-sRGB swap): scene + UI render into
                //      an offscreen sRGB texture; a composite pass at end-of-frame
                //      samples it and gamma-encodes for write to swap_view.
                //   3. Swap view directly (native, MSAA off).
                let render_view = rd
                    .msaa_view()
                    .or(rd.scene_view())
                    .unwrap_or(&swap_view);

                // GPU timer start. Tiny dedicated encoder so the timestamp lands in
                // the queue before any of the frame's submitted work. Stays separate
                // from the main frame encoder so the start timestamp is on the GPU
                // BEFORE we begin recording scene passes; merging them would put the
                // start timestamp at the end of the same submit as the work, ruining
                // the measurement. Same logic for the end timer below. No-op when
                // the adapter didn't advertise TIMESTAMP_QUERY.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rye-app::gpu-timer-start"),
                            });
                    timer.write_start(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                }

                // THE frame encoder. App::record, ui.paint, and the composite pass
                // all write into this single encoder; the runner submits it once
                // before `frame.present`. The capture taps (native only) need
                // intermediate submits to make readback see the right pixels; those
                // paths split the encoder mid-frame to maintain correctness, at the
                // cost of the single-submit win in capture-active frames.
                let mut encoder =
                    rd.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("rye-app::frame"),
                        });

                if let Some(app) = self.app.as_mut() {
                    let _scope = rye_time::frame_trace::scope("app-record");
                    let mut ctx = RenderCtx {
                        rd,
                        view: render_view,
                        encoder: &mut encoder,
                    };
                    if let Err(e) = app.record(&mut ctx) {
                        tracing::error!("App::record error: {e:#}");
                        last_err = Some(e);
                    }
                }

                // Pre-egui capture tap. Only valid with MSAA off. Forces a mid-frame
                // submit so the GPU has actually drawn the scene before we read it
                // back; we restart the encoder afterwards for ui+composite.
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_pre() {
                    if rd.sample_count() > 1 {
                        tracing::warn!(
                            "capture: `pre` stage skipped because MSAA is on; \
                             set RunConfig::msaa_samples = 1 for diagnostic capture"
                        );
                    } else {
                        rd.queue.submit(Some(encoder.finish()));
                        encoder = rd.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("rye-app::frame-post-pre-capture"),
                            },
                        );
                        capture_consume(&mut self.capture, rd, &frame.texture, true, capture_now);
                    }
                }

                if let Some(ui) = self.ui.as_mut() {
                    let _scope = rye_time::frame_trace::scope("ui-paint");
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
                }

                // Post-egui capture tap. Like the pre-tap, forces a mid-frame submit
                // before the readback so the composite has actually happened on the
                // GPU. Restart the encoder for composite (if needed below).
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_post() {
                    rd.queue.submit(Some(encoder.finish()));
                    encoder = rd.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("rye-app::frame-post-post-capture"),
                        },
                    );
                    capture_consume(&mut self.capture, rd, &frame.texture, false, capture_now);
                }
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture {
                    self.capture.advance_frame(capture_now);
                }
                // Publish status every frame so the panel + window title stay current
                // even when do_capture is false (FPS-gated idle frames between writes).
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                capture::publish_status(self.capture.status());

                // Composite pass: sRGB scene texture -> linear swapchain with manual
                // gamma encoding. Writes into the same encoder as everything else
                // (no separate submit). No-op on native (where scene_view() is None
                // and rendering wrote directly into the swapchain).
                if rd.scene_view().is_some() {
                    let _scope = rye_time::frame_trace::scope("composite");
                    rd.composite_to_swap(&mut encoder, &swap_view);
                }

                // GPU timer end + resolve. Stays in a separate small encoder for the
                // same reason as the start timer: ordering vs the frame's main work.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    rd.queue.submit(Some(encoder.finish()));
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rye-app::gpu-timer-end"),
                            });
                    timer.write_end_and_resolve(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                } else {
                    // Single submit for the whole frame (the path we're optimizing
                    // for on wasm + native-without-timestamps).
                    rd.queue.submit(Some(encoder.finish()));
                }

                {
                    let _scope = rye_time::frame_trace::scope("present");
                    frame.present();
                }

                // Advance the GPU timer's frame index + drain any completed timings
                // into frame_trace. The slot whose end-timestamp was just resolved
                // gets its map_async scheduled here; the result lands in frame_trace
                // 1-2 frames later via the channel.
                if let Some(timer) = self.rd.as_mut().and_then(|rd| rd.gpu_timer.as_mut()) {
                    timer.tick();
                }
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

        // End the frame's trace. Must happen after _frame_scope drops (i.e. at the
        // very end of redraw); the surrounding block scope ensures that ordering.
        drop(_frame_scope);
        rye_time::frame_trace::end_frame();
    }
}
