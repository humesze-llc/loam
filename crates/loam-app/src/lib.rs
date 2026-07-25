//! `loam-app`: thin App trait + event-loop runner that extracts the winit
//! boilerplate every Loam example would otherwise rewrite.
//!
//! Apps implement [`App`] on a struct that owns their state; the runner [`run`]
//! (or [`run_with_config`]) handles window creation, [`RenderDevice`] +
//! surface-error recovery, [`ShaderDb`]/[`AssetWatcher`] hot-reload,
//! [`InputState`] routing, [`FixedTimestep`]-driven `App::tick`, and FPS/title
//! bookkeeping. It is not an ECS, scene graph, render-graph orchestrator, or
//! camera framework; apps own those directly.
//!
//! A frame-capture pipeline ships behind the `capture` feature (default-on); see
//! [`capture`]. External recorders (OBS) remain better for long sessions.
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
// `web_time::Instant` works on native (std type) and wasm32 (`performance.now()`);
// `std::time::Instant::now` panics on wasm32, so the swap is mandatory there.
use web_time::Instant;

// Real capture pipeline on native+`capture`; stub elsewhere. Both expose the
// same surface so demos need no `cfg` gates at `loam_app::capture::*` call sites.
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
pub mod capture;

pub mod args;
#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
#[path = "capture_stub.rs"]
pub mod capture;
pub mod cursor;
pub mod fps;
pub mod frame_pacing;
pub mod freecam;
pub mod keymap;
pub mod log;
pub mod trace;
pub mod version;
pub mod vsync;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use loam_asset::AssetWatcher;
use loam_egui::UiIntegration;
use loam_input::{FrameInput, InputState};
use loam_math::WgslSpace;
use loam_render::device::RenderDevice;
use loam_shader::ShaderDb;
use loam_time::FixedTimestep;

// Convenience re-exports so apps depend on `loam-app` alone for common types.
pub use loam_camera::{
    Camera, CameraController, CameraView, FirstPersonController, OrbitController,
};
pub use loam_egui::{egui, world_to_screen, BottomOverlay, LinearIndicator};
pub use loam_input::FrameInput as Input;

// ---------------------------------------------------------------------------
// App trait
// ---------------------------------------------------------------------------

/// The framework calls back into your App through this trait. All methods except
/// [`App::setup`], [`App::space`], and [`App::render`] have default impls.
pub trait App: Sized + 'static {
    /// Shader-prelude geometry. `ShaderDb::apply_events` runs against this instance
    /// on hot-reload, so `loam_distance` / `loam_log` / `loam_exp` in WGSL evaluate
    /// under this metric. Geometry-agnostic apps use `EuclideanR3`.
    ///
    /// This is not a commitment about the camera, player, or scene; those are
    /// user-owned and may use a different Space or none. Hazard: writing
    /// `Camera<Self::Space>` commits the scene to that Space's coordinates. For
    /// H³ (the Poincaré ball), a Euclidean-default orbit distance lands the camera
    /// near the ideal boundary where the metric explodes; use `Camera<EuclideanR3>`
    /// and treat `App::Space` purely as the shader-prelude axis when the scene
    /// isn't actually in H³.
    type Space: WgslSpace + 'static;

    /// One-shot construction after `RenderDevice` and `ShaderDb` are ready. Build
    /// render nodes, load shaders, allocate state, and store it all (including
    /// `Self::Space` and any cameras) in the returned `Self`.
    fn setup(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self>;

    /// Borrow the user-owned `Self::Space` for `ShaderDb::apply_events`.
    fn space(&self) -> &Self::Space;

    /// Per-tick simulation step at the fixed-timestep rate ([`RunConfig::fixed_hz`]).
    /// Usually 0 or 1 per frame; can spike to [`RunConfig::max_ticks_per_frame`].
    fn tick(&mut self, _dt: f32, _ctx: &mut TickCtx) {}

    /// Per-frame update with drained input, after all the frame's ticks. Advance
    /// the camera controller, recompute uniforms, etc.
    fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

    /// Custom `WindowEvent` handling beyond the framework's input routing.
    ///
    /// Not called for keyboard events in the wasm worker (it can't construct a
    /// `WindowEvent::KeyboardInput`); use [`App::on_key`] for hotkeys, which fires
    /// on both paths.
    fn on_event(&mut self, _ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {}

    /// Keyboard hotkey hook, fired for every press and release after input
    /// routing. Use for edge-triggered toggles (Space, Tab, digits, etc.);
    /// continuous WASD belongs in [`App::update`] reading held-key axes.
    ///
    /// Separate from [`App::on_event`] because the wasm worker can't construct a
    /// `winit::event::KeyEvent` but can produce `KeyCode + ElementState`, so
    /// `on_key` reaches both native and wasm.
    fn on_key(
        &mut self,
        _code: winit::keyboard::KeyCode,
        _state: ElementState,
        _ctx: &mut FrameCtx<'_>,
    ) {
    }

    /// Hot-reload notification: events were applied to `ShaderDb` against
    /// `self.space()`; rebuild any consumer pipelines that may be stale.
    fn on_shader_reload(&mut self, _ctx: &mut SetupCtx<'_>) {}

    /// Legacy render path. Implement either this or `App::record`; the runner
    /// always calls `record`, whose default impl calls this. Each `render` call
    /// typically does its own `queue.submit`, so an N-node demo pays N submits;
    /// `record` lets the runner batch into a single per-frame submit. New demos
    /// should override `record`; this remains for the existing examples.
    fn render(&mut self, _rd: &RenderDevice, _view: &wgpu::TextureView) -> anyhow::Result<()> {
        Ok(())
    }

    /// Preferred render path: the runner owns one frame-wide command encoder
    /// (shared via [`RenderCtx::encoder`]) for the demo's passes, ui-paint, and
    /// the wasm composite, reaching the GPU in a single `queue.submit`. Default
    /// impl falls back to [`App::render`], so migration is opt-in.
    ///
    /// Contract:
    /// - Do NOT call `encoder.finish()` or `queue.submit`; the runner does that
    ///   once at end-of-frame.
    /// - Multiple render passes per call are fine on the shared encoder.
    /// - Use `ctx.view` as the color target; the runner already selected the
    ///   right view (MSAA / scene-target / swapchain) for the platform.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> anyhow::Result<()> {
        self.render(ctx.rd, ctx.view)
    }

    /// Build this frame's egui UI, after [`App::update`]; painted as a 2D overlay.
    /// Gate gameplay input on [`FrameCtx::ui_has_focus`] so typing into a field
    /// doesn't also fire WASD. For a label following a 3D object, use
    /// [`world_to_screen`] to place an `egui::Area`.
    ///
    /// ```ignore
    /// fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
    ///     egui::Window::new("Settings").show(ctx, |ui| {
    ///         ui.add(egui::Slider::new(&mut self.fov, 30.0..=120.0));
    ///         if ui.button("Reset").clicked() { self.reset(); }
    ///     });
    /// }
    /// ```
    fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

    /// Title bar text. Override for live readouts; the runner rate-limits the
    /// `set_title` call to ~1 Hz.
    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("loam app")
    }
}

// ---------------------------------------------------------------------------
// Context structs
// ---------------------------------------------------------------------------

/// Run one frame's fixed-timestep ticks: advance the accumulator, cap the
/// catch-up at `max_ticks`, and call `App::tick` for each, bumping `tick_index`.
/// Shared by the native runner and the wasm worker so the determinism-critical
/// sim cadence has a single definition and cannot drift between platforms.
/// Returns the capped tick count (for `FrameCtx::n_ticks`).
///
/// `now` reaches the wall clock only through the accumulator, which decides
/// *how many* ticks run. What each tick sees is a pure function of its index.
pub(crate) fn drive_fixed_ticks<A: App>(
    app: &mut A,
    timestep: &mut FixedTimestep,
    tick_index: &mut u64,
    now: Instant,
    fixed_hz: u32,
    max_ticks: usize,
) -> usize {
    let _scope = loam_time::frame_trace::scope("sim-ticks");
    let n_capped = timestep.advance(now).count().min(max_ticks);
    let dt = 1.0 / fixed_hz as f32;
    for _ in 0..n_capped {
        let mut tctx = TickCtx {
            time: *tick_index as f32 * dt,
            tick: *tick_index,
        };
        app.tick(dt, &mut tctx);
        *tick_index = tick_index.wrapping_add(1);
    }
    n_capped
}

/// Setup-phase context. Available during [`App::setup`] and [`App::on_shader_reload`].
pub struct SetupCtx<'a> {
    pub rd: &'a RenderDevice,
    pub shader_db: &'a mut ShaderDb,
    /// `None` when filesystem watching failed to init; apps still load shaders
    /// but get no hot-reload.
    pub watcher: Option<&'a mut AssetWatcher>,
    /// Wall-clock seconds since `run`. Always 0 in `setup`.
    pub time: f32,
}

/// Per-tick context. Visible to [`App::tick`]. Deliberately GPU-free so sim code stays
/// bit-deterministic.
pub struct TickCtx {
    /// Sim time in seconds: `tick` scaled by the fixed timestep
    /// (`1.0 / RunConfig::fixed_hz`). Derived from the tick index rather than
    /// read from the clock, so replaying the same tick range yields the same
    /// bits however the frames were paced. Wall-clock time lives on
    /// [`FrameCtx::time`], outside the determinism boundary.
    pub time: f32,
    pub tick: u64,
}

/// Render-time context for `App::record`. Owns the shared frame encoder; the
/// runner reuses it for ui-paint and the wasm composite, then submits once.
///
/// `view` is the runner's best color target for the platform (MSAA view, wasm
/// sRGB scene texture, or swapchain). Pipelines built with
/// [`RenderDevice::target_format`] + [`RenderDevice::sample_count`] match it.
pub struct RenderCtx<'a> {
    pub rd: &'a RenderDevice,
    pub view: &'a wgpu::TextureView,
    /// Shared command encoder. Open passes, draw, drop; do NOT call `finish()`
    /// or `queue.submit`.
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
    /// Bail out after this many consecutive `wgpu::SurfaceError` results from
    /// `begin_frame`. Larger than [`Self::render_error_budget`] because a sleep /
    /// resume cycle on Windows / DX12 routinely takes several frames to settle
    /// (the surface returns `Outdated` or `Other` until the driver finishes
    /// rebuilding the swapchain). Counts all error variants except
    /// `OutOfMemory`, which exits immediately. Reset to zero on any successful
    /// `begin_frame`. `0` disables the budget.
    pub surface_error_budget: u32,
    /// MSAA sample count requested for the scene + UI render target. `1` disables
    /// MSAA. `4` is the conventional default (good quality / cost tradeoff, supported
    /// on every consumer GPU). Higher counts (8, 16) cost more and yield diminishing
    /// returns on edge antialiasing. The runtime negotiates with the adapter; if the
    /// requested count isn't supported on the chosen surface format, [`RenderDevice`]
    /// falls back to the highest supported lower count and logs a warning.
    pub msaa_samples: u32,
    /// Wasm-specific knobs (DOM IDs the page exposes). Ignored on native. The
    /// defaults match the standard layout loam demos use (`loam-canvas-host`
    /// container, `loam-launch` button, `loam-canvas` canvas); demos that ship
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
            host_id: "loam-canvas-host".into(),
            button_id: "loam-launch".into(),
            canvas_id: "loam-canvas".into(),
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            window: WindowAttributes::default()
                .with_title("loam app")
                .with_visible(false),
            fixed_hz: 60,
            max_ticks_per_frame: 4,
            log_filter: None,
            esc_exits: true,
            render_error_budget: 8,
            surface_error_budget: 32,
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
        loam_time::frame_trace::set_heap_sampler(wasm::js_heap_sampler);
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
    event_loop.set_control_flow(ControlFlow::Poll);

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
    // and proceed. On wasm32 the watcher is a no-op stub (see `loam-asset`'s watcher.rs)
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
    let mut ui = UiIntegration::new(&rd.device, win, rd.target_format(), rd.sample_count());

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
    // but no loam demo does that today.
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
type PendingInit<A> = std::rc::Rc<std::cell::RefCell<Option<anyhow::Result<InitArtifacts<A>>>>>;

/// Attach the canvas backing the winit window to the page's DOM. Without this the
/// canvas exists only as a JS object the surface can target but nothing the user can
/// see; appending it makes the render output visible and lets pointer / keyboard
/// events flow through.
///
/// Host element selection prefers `#loam-canvas-host` when the page provides one
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

    let host: web_sys::Element = match document.get_element_by_id("loam-canvas-host") {
        Some(el) => el,
        None => document.body().map(Into::into).ok_or_else(|| {
            anyhow::anyhow!("no canvas host: page is missing both `#loam-canvas-host` and `<body>`")
        })?,
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
    /// Consecutive `wgpu::SurfaceError` results from `begin_frame`, since the
    /// last successful frame. Compared against `RunConfig::surface_error_budget`.
    /// Sleep / resume on DX12 routinely produces several frames of `Outdated` or
    /// `Other` before the swapchain rebuilds; the streak only matters if it
    /// doesn't recover.
    surface_error_streak: u32,
    /// Wall-clock instant of the last surface-error log line. Used to rate-limit
    /// the `tracing::error!` for `SurfaceError::Other` to ~1 Hz so a stuck
    /// surface doesn't spew thousands of log lines (and the corresponding
    /// allocations) per second.
    last_surface_error_log: Option<Instant>,
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
            surface_error_streak: 0,
            last_surface_error_log: None,
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
        // (see loam_app::log), every meaningful WindowEvent emits a
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
        // before Loam's own routing translates the event for gameplay.
        // egui consuming the event is informational; Loam still sees it.
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
                let was_minimized = self.minimized;
                self.minimized = size.width == 0 || size.height == 0;
                match (was_minimized, self.minimized) {
                    // Just got minimized: park the event loop in `Wait` so we
                    // stop burning CPU on poll iterations that have no work to
                    // do. Without this the loop spins at 100% even though
                    // `redraw` early-returns, which is what makes laptop fans
                    // go nuts on a minimized demo window.
                    (false, true) => elwt.set_control_flow(ControlFlow::Wait),
                    // Just restored: switch back to `Poll` so the redraw loop
                    // re-engages at vsync rate, reconfigure the swapchain at
                    // the new size, and kick off the first redraw. The
                    // `FixedTimestep` spiral cap will discard any accumulated
                    // catch-up so we don't queue hours of ticks.
                    (true, false) => {
                        elwt.set_control_flow(ControlFlow::Poll);
                        if let Some(rd) = &mut self.rd {
                            rd.resize(*size);
                        }
                        win.request_redraw();
                    }
                    // Visible-to-visible resize: just reconfigure the
                    // swapchain. Minimized-to-minimized is impossible (winit
                    // wouldn't emit it) but harmless.
                    (false, false) => {
                        if let Some(rd) = &mut self.rd {
                            rd.resize(*size);
                        }
                    }
                    (true, true) => {}
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
                // Mirror the wasm worker's contract: route keyboard events
                // through `on_key` too, so demos with a single hotkey impl
                // (in `on_key`) work the same on both runners.
                if let WindowEvent::KeyboardInput { event, .. } = &ev {
                    if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                        app.on_key(code, event.state, &mut ctx);
                    }
                }
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

    /// Device-level events (mouse motion independent of cursor position,
    /// raw scroll wheel, etc.). The only one we currently consume is
    /// `MouseMotion`, which gives uncapped deltas that don't stop when the
    /// cursor hits the screen edge. Routed into `InputState::accumulate_raw_motion`
    /// so consumers (camera controllers, primarily) that grab the cursor can
    /// read it via `FrameInput::mouse_raw_delta`.
    fn device_event(
        &mut self,
        _elwt: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        ev: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = ev {
            self.input.accumulate_raw_motion(delta.0, delta.1);
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
        if let (Some(want_on), Some(rd)) = (frame_pacing::take_pending_vsync(), self.rd.as_mut()) {
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

        // Pending cursor grab + visibility transitions from
        // `loam_app::cursor`. Native only; browser Pointer Lock requires a
        // recent user gesture that console commands don't satisfy (the
        // cursor module emits a one-time tracing::warn on wasm).
        //
        // Grab + visibility are independent. The runner translates the
        // engine-level `GrabMode` to winit's `CursorGrabMode` and falls
        // back from `Locked` to `Confined` (or vice versa) if the platform
        // doesn't support the requested mode (macOS + Linux/X11 vary on
        // which they prefer; Windows accepts both). The fallback chain
        // preserves the user's intent (some form of grab) rather than
        // silently dropping to `None`.
        #[cfg(not(target_arch = "wasm32"))]
        {
            use winit::window::CursorGrabMode;
            let (pending_grab, pending_vis) = cursor::take_pending();
            let mut applied = cursor::current_state();
            if let Some(mode) = pending_grab {
                let primary = match mode {
                    cursor::GrabMode::None => CursorGrabMode::None,
                    cursor::GrabMode::Confined => CursorGrabMode::Confined,
                    cursor::GrabMode::Locked => CursorGrabMode::Locked,
                };
                if win.set_cursor_grab(primary).is_err() && mode != cursor::GrabMode::None {
                    let fallback = match mode {
                        cursor::GrabMode::Locked => CursorGrabMode::Confined,
                        cursor::GrabMode::Confined => CursorGrabMode::Locked,
                        cursor::GrabMode::None => CursorGrabMode::None,
                    };
                    let _ = win.set_cursor_grab(fallback);
                }
                applied.grab = mode;
            }
            if let Some(visible) = pending_vis {
                win.set_cursor_visible(visible);
                applied.visible = visible;
            }
            if pending_grab.is_some() || pending_vis.is_some() {
                cursor::mark_applied(applied.grab, applied.visible);
            }
            // Warp-to-center request. Applied AFTER the grab/visibility
            // transition so the new cursor state is in effect first; warping
            // a still-Locked cursor would be a no-op (winit pins it to the
            // center already), but pairing the warp with a release lands the
            // pointer where the user was aiming when they Alt-tabbed out.
            if cursor::take_pending_warp_center() {
                let size = win.inner_size();
                let center = winit::dpi::PhysicalPosition::new(
                    size.width as f64 / 2.0,
                    size.height as f64 / 2.0,
                );
                let _ = win.set_cursor_position(center);
            }
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
        let now = Instant::now();
        let frame_anchor = if let (Some(period), Some(last)) =
            (frame_pacing::target_period(), self.last_redraw_at)
        {
            let deadline = last + period;
            if now < deadline {
                #[cfg(target_arch = "wasm32")]
                {
                    win.request_redraw();
                    return;
                }
                // Native: sleep out the remainder and anchor on the
                // ideal deadline, not the wake-up time.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    frame_pacing::precise_sleep_until(deadline);
                    deadline
                }
            } else {
                now
            }
        } else {
            now
        };
        self.last_redraw_at = Some(frame_anchor);

        // Mark frame start for `idle` measurement. `end_frame` subtracts this from
        // the previous `end_frame` timestamp to get `idle` (browser/RAF gap, not
        // our work); separate from `between-frames` (total cadence, our work +
        // idle combined).
        loam_time::frame_trace::begin_frame();

        // Whole-redraw scope. Subsequent named sections sum to less than this; the
        // delta is the small bits in between (FPS bookkeeping, capture status
        // publish, etc.). Used by the `trace` console command to surface "total
        // CPU work this frame" vs. "what's the dominant section."
        let _frame_scope = loam_time::frame_trace::scope("frame");

        // 1. Fixed-timestep ticks (shared with the wasm worker via
        // `drive_fixed_ticks` so the sim cadence stays identical across platforms).
        let n_capped = if let Some(app) = self.app.as_mut() {
            drive_fixed_ticks(
                app,
                &mut self.timestep,
                &mut self.tick_index,
                Instant::now(),
                self.config.fixed_hz,
                self.config.max_ticks_per_frame,
            )
        } else {
            0
        };

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
                let _scope = loam_time::frame_trace::scope("app-update");
                app.update(&mut fctx);
            }

            // Build this frame's UI. egui captures the widgets;
            // `paint` later renders them after `App::render`.
            if let Some(ui) = self.ui.as_mut() {
                let _scope = loam_time::frame_trace::scope("app-ui");
                let egui_ctx = ui.begin_frame(win.as_ref()).clone();
                app.ui(&egui_ctx, &mut fctx);
            }
        }

        // 3. Hot-reload poll.
        {
            let _scope = loam_time::frame_trace::scope("hot-reload");
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

        let begin_result = rd.begin_frame();
        if begin_result.is_ok() {
            self.surface_error_streak = 0;
            self.last_surface_error_log = None;
        }
        match begin_result {
            Ok((frame, swap_view)) => {
                let mut last_err: Option<anyhow::Error> = None;
                // Render-target priority chain:
                //   1. MSAA view (native, MSAA on): scene + UI render multisampled,
                //      resolve at egui paint pass's resolve_target = swap_view.
                //   2. Scene view (browser, non-sRGB swap): scene + UI render into
                //      an offscreen sRGB texture; a composite pass at end-of-frame
                //      samples it and gamma-encodes for write to swap_view.
                //   3. Swap view directly (native, MSAA off).
                let render_view = rd.msaa_view().or(rd.scene_view()).unwrap_or(&swap_view);

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
                                label: Some("loam-app::gpu-timer-start"),
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
                            label: Some("loam-app::frame"),
                        });

                if let Some(app) = self.app.as_mut() {
                    let _scope = loam_time::frame_trace::scope("app-record");
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
                        encoder =
                            rd.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("loam-app::frame-post-pre-capture"),
                                });
                        capture_consume(&mut self.capture, rd, &frame.texture, true, capture_now);
                    }
                }

                if let Some(ui) = self.ui.as_mut() {
                    let _scope = loam_time::frame_trace::scope("ui-paint");
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
                    encoder = rd
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame-post-post-capture"),
                        });
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
                    let _scope = loam_time::frame_trace::scope("composite");
                    rd.composite_to_swap(&mut encoder, &swap_view);
                }

                // GPU timer end + resolve. Stays in a separate small encoder for the
                // same reason as the start timer: ordering vs the frame's main work.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    rd.queue.submit(Some(encoder.finish()));
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("loam-app::gpu-timer-end"),
                            });
                    timer.write_end_and_resolve(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                } else {
                    // Single submit for the whole frame (the path we're optimizing
                    // for on wasm + native-without-timestamps).
                    rd.queue.submit(Some(encoder.finish()));
                }

                {
                    let _scope = loam_time::frame_trace::scope("present");
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
            Err(err) => {
                // `OutOfMemory` is terminal; bail without budgeting (any retry
                // would just allocate again and fail again, faster).
                if matches!(err, wgpu::SurfaceError::OutOfMemory) {
                    self.deferred_error = Some(anyhow::anyhow!("wgpu surface out of memory"));
                    elwt.exit();
                    return;
                }

                // Recovery: re-configure the swapchain at the current size for
                // the `Lost`/`Outdated`/`Other` variants. `Other` is what DX12
                // returns after sleep/resume when the swapchain is wedged;
                // reconfiguring lets wgpu rebuild it. `Timeout` is transient
                // (driver took too long for one frame) so don't reconfigure.
                match err {
                    wgpu::SurfaceError::Lost
                    | wgpu::SurfaceError::Outdated
                    | wgpu::SurfaceError::Other => {
                        if let Some(rd) = &mut self.rd {
                            let size = rd.surface_bundle.size;
                            rd.resize(size);
                        }
                    }
                    _ => {}
                }

                // Rate-limit the `Other` log to ~1 Hz so a persistently wedged
                // surface doesn't spew thousands of lines/sec (and the
                // corresponding tracing allocations). Other variants are rare
                // enough that the unthrottled debug-level log is fine.
                if matches!(err, wgpu::SurfaceError::Other) {
                    let now = Instant::now();
                    let should_log = self
                        .last_surface_error_log
                        .map(|t| now.duration_since(t).as_secs_f32() >= 1.0)
                        .unwrap_or(true);
                    if should_log {
                        tracing::error!("surface error: {err:?}");
                        self.last_surface_error_log = Some(now);
                    }
                } else {
                    tracing::debug!("surface error: {err:?}");
                }

                self.surface_error_streak = self.surface_error_streak.saturating_add(1);
                let budget = self.config.surface_error_budget;
                if budget > 0 && self.surface_error_streak >= budget {
                    self.deferred_error = Some(anyhow::anyhow!(
                        "wgpu surface error persisted {budget} consecutive frames: {err:?}"
                    ));
                    elwt.exit();
                    return;
                }
                win.request_redraw();
            }
        }

        // End the frame's trace. Must happen after _frame_scope drops (i.e. at the
        // very end of redraw); the surrounding block scope ensures that ordering.
        drop(_frame_scope);
        loam_time::frame_trace::end_frame();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::EuclideanR3;
    use std::time::Duration;

    /// One tick of the 60 Hz accumulator, as `FixedTimestep` stores it
    /// (nanoseconds derived from the target Hz, truncated).
    const TICK: Duration = Duration::from_nanos(1_000_000_000 / 60);

    #[derive(Default)]
    struct TickRecorder {
        space: EuclideanR3,
        times: Vec<f32>,
    }

    impl App for TickRecorder {
        type Space = EuclideanR3;

        fn setup(_ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self> {
            Ok(Self::default())
        }

        fn space(&self) -> &Self::Space {
            &self.space
        }

        fn tick(&mut self, _dt: f32, ctx: &mut TickCtx) {
            self.times.push(ctx.time);
        }
    }

    /// Drive frames whose wall-clock instants are `base + offsets[i]` and return
    /// every `TickCtx::time` the app observed. The first offset only primes the
    /// accumulator.
    fn tick_times(base: Instant, offsets: &[Duration], max_ticks: usize) -> Vec<f32> {
        let mut app = TickRecorder::default();
        let mut timestep = FixedTimestep::new(60);
        let mut tick_index = 0u64;
        for offset in offsets {
            drive_fixed_ticks(
                &mut app,
                &mut timestep,
                &mut tick_index,
                base + *offset,
                60,
                max_ticks,
            );
        }
        app.times
    }

    #[test]
    fn tick_time_sequence_is_independent_of_wall_clock_offset() {
        let offsets: Vec<Duration> = (0..=20).map(|k| TICK * k).collect();
        let early = tick_times(Instant::now(), &offsets, 8);
        let late = tick_times(Instant::now() + Duration::from_secs(3_600), &offsets, 8);

        assert_eq!(
            early.len(),
            20,
            "one tick per frame after the priming frame"
        );
        assert_eq!(
            early, late,
            "tick time must not shift with the run's wall-clock origin"
        );
    }

    #[test]
    fn tick_time_sequence_is_independent_of_frame_pacing() {
        let base = Instant::now();
        let one_per_frame: Vec<Duration> = (0..=60).map(|k| TICK * k).collect();
        let ten_per_frame: Vec<Duration> = (0..=6).map(|k| TICK * (k * 10)).collect();

        let smooth = tick_times(base, &one_per_frame, 10);
        let stuttered = tick_times(base, &ten_per_frame, 10);

        let expected: Vec<f32> = (0..60).map(|i| i as f32 * (1.0 / 60.0)).collect();
        assert_eq!(smooth, expected, "tick time is tick_index * dt from zero");
        assert_eq!(
            stuttered, expected,
            "catching up ten ticks in one frame must yield the same time sequence"
        );
    }
}
