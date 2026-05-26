//! Web Worker mode for rye demos. Moves the render loop into a worker so V8's
//! GC pauses don't block the visible page.
//!
//! ## What this provides
//!
//! Worker receives an OffscreenCanvas via postMessage, creates a wgpu Surface
//! from it, and drives a rolled-own RAF loop that runs the full App lifecycle
//! (setup, update, ui, record) plus an egui overlay (via [`WorkerUi`], which
//! translates [`InputMessage`] directly into `egui::RawInput`, bypassing
//! egui-winit which doesn't run in worker context).
//!
//! ## Why a rolled-own event loop (no winit)
//!
//! winit 0.30 doesn't support `WorkerGlobalScope` (issue #1518 since 2020):
//! `web_sys::window()` panics in worker context, breaks scale-factor / event
//! pump. Until upstream lands worker support (Bevy tried and abandoned the
//! PR), we own this code path ourselves. Trade-off accepted because the
//! pieces we need (RAF, GPU surface creation, message passing) are all
//! available without winit.
//!
//! ## Two contexts, one binary
//!
//! Same wasm bundle runs on the main thread (the page) AND inside the
//! worker. Detection via [`crate::wasm::is_worker_context`] lets `main`
//! branch into [`run`] (worker entry) vs [`launch_on_click`] (main-thread
//! entry).

use anyhow::{anyhow, Context, Result};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, OffscreenCanvas};

use super::messages::{self, InputMessage};
use super::worker_ui::WorkerUi;
use crate::{App, FrameCtx, RenderCtx, SetupCtx, TickCtx};
use rye_asset::AssetWatcher;
use rye_input::InputState;
use rye_render::device::RenderDevice;
use rye_shader::ShaderDb;
use rye_time::FixedTimestep;
use winit::event::{ElementState, MouseScrollDelta};
use winit::keyboard::PhysicalKey;

/// Worker entry. Generic over the demo's `App` type so the same worker
/// scaffolding drives whatever lifecycle the demo defines.
///
/// Installs a `message` listener that waits for the canvas-transfer init,
/// then constructs `RenderDevice` + `WorkerRunner<A>` and starts the RAF
/// loop that drives `A`'s per-frame lifecycle (update + record).
///
/// Returns `Ok(())` synchronously after wiring the listener; the actual
/// work happens inside the message + RAF callbacks. The worker stays
/// alive as long as the wasm-bindgen heap holds the closures (the
/// `forget()` calls below ensure that).
///
pub fn run<A: App + 'static>() -> Result<()>
where
    A::Space: 'static,
{
    // Worker has its own JS heap + console; install panic hook + tracing
    // so unhandled errors + log lines surface in DevTools (under the
    // worker's context, selectable in DevTools' execution-context picker).
    install_logging_idempotent();

    tracing::debug!("rye_app::wasm::worker::run: entry");

    let scope = worker_scope()?;
    let scope_for_handler = scope.clone();

    // The message handler ingests the OffscreenCanvas + size from the
    // first postMessage and kicks off the wgpu init. Subsequent messages
    // (input events) reuse this same handler with a `kind` dispatch.
    //
    // Use `addEventListener("message", ...)` rather than `set_onmessage`
    // because `addEventListener` is reliably retroactive: messages
    // posted to the Worker before this listener is installed are queued
    // by the browser, then delivered when the listener registers. The
    // `set_onmessage` setter has had spec ambiguity around exactly when
    // queued messages are flushed; `addEventListener` is the safer
    // contract.
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        tracing::debug!("rye_app::wasm::worker: message handler firing");
        if let Err(e) = handle_message::<A>(&scope_for_handler, event) {
            tracing::error!("rye_app::wasm::worker: message handler failed: {e:#}");
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    scope
        .add_event_listener_with_callback("message", on_message.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("addEventListener('message'): {e:?}"))?;
    // Intentionally leak the closure: it must live for the worker's
    // entire lifetime, and the worker dies when the JS heap drops its
    // reference to the closure. `forget` is the correct primitive here.
    on_message.forget();

    // Signal main thread that we're ready to receive Init. Without this
    // handshake, main thread might post the Init message before this
    // listener is installed; the browser MAY queue messages for late-
    // installed listeners but the spec is ambiguous and Firefox empirically
    // drops them in some configurations. The handshake makes the protocol
    // explicit + robust across browsers.
    let ready_msg = js_sys::Object::new();
    js_sys::Reflect::set(
        &ready_msg,
        &JsValue::from_str("kind"),
        &JsValue::from_str("ready"),
    )
    .map_err(|e| anyhow!("Reflect::set ready.kind: {e:?}"))?;
    scope
        .post_message(&ready_msg)
        .map_err(|e| anyhow!("postMessage ready: {e:?}"))?;

    tracing::info!("rye_app::wasm::worker::run: message listener installed + ready posted");

    Ok(())
}

/// Dispatch a single inbound `postMessage`. The `init` kind is special-
/// cased here (extracts the OffscreenCanvas + spawns the async wgpu
/// setup); all other kinds go through [`messages::parse_non_init`] and
/// get pushed onto the per-worker queue for the frame-loop to drain.
fn handle_message<A: App + 'static>(
    scope: &DedicatedWorkerGlobalScope,
    event: MessageEvent,
) -> Result<()>
where
    A::Space: 'static,
{
    let data: JsValue = event.data();

    // `init` and `start` are both special-cased: `init` needs A (spawns
    // the async wgpu setup with the App type) and `start` needs to fire
    // BEFORE the RAF loop exists (so we can't route it through the
    // per-frame queue; that queue only drains inside the loop).
    let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|v| v.as_string());
    if kind.as_deref() == Some("start") {
        // Three orderings to handle:
        //  1. Init done, kickoff stashed: invoke it now.
        //  2. Init still running: stash the request; init_renderer will
        //     self-trigger once the kickoff is ready.
        //  3. Already started (kickoff consumed): no-op.
        if let Some(kickoff) = RAF_KICKOFF.with(|k| k.borrow_mut().take()) {
            tracing::info!("rye_app::wasm::worker: Start received, kicking off RAF loop");
            kickoff();
        } else {
            // Either case 2 (init still running) or case 3 (already
            // started). Setting the flag covers case 2; case 3 is
            // harmless (init_renderer doesn't re-read the flag after
            // first consume).
            START_REQUESTED.with(|s| s.set(true));
            tracing::info!("rye_app::wasm::worker: Start received before kickoff ready; queued");
        }
        return Ok(());
    }
    if kind.as_deref() == Some("init") {
        let canvas = js_sys::Reflect::get(&data, &JsValue::from_str("canvas"))
            .map_err(|e| anyhow!("init missing 'canvas' field: {e:?}"))?
            .dyn_into::<OffscreenCanvas>()
            .map_err(|e| anyhow!("init 'canvas' is not an OffscreenCanvas: {e:?}"))?;
        let width = js_sys::Reflect::get(&data, &JsValue::from_str("width"))
            .ok()
            .and_then(|v| v.as_f64())
            .map(|f| f as u32)
            .unwrap_or(800);
        let height = js_sys::Reflect::get(&data, &JsValue::from_str("height"))
            .ok()
            .and_then(|v| v.as_f64())
            .map(|f| f as u32)
            .unwrap_or(600);

        tracing::info!(
            "rye_app::wasm::worker: received init ({width}x{height}); spawning wgpu setup"
        );
        let scope_for_render = scope.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = init_renderer::<A>(scope_for_render, canvas, width, height).await {
                tracing::error!("rye_app::wasm::worker: init_renderer failed: {e:#}");
            }
        });
        return Ok(());
    }

    // Non-init kinds: parse + queue.
    match messages::parse_non_init(&data)? {
        Some(msg) => messages::enqueue(msg),
        None => {
            // Unknown kind. Don't error (defensive against future
            // additions on main side); the warn surfaces it for debugging.
            if let Some(k) = kind {
                tracing::warn!("rye_app::wasm::worker: unknown message kind '{k}'");
            }
        }
    }
    Ok(())
}

/// Build `RenderDevice` from the worker-owned OffscreenCanvas, run
/// `App::setup`, and start the RAF loop driving the App's per-frame lifecycle.
///
/// Uses `RenderDevice::from_surface` so the wgpu setup matches the
/// windowed-mode path (sRGB composite, MSAA negotiation, GPU timer
/// detection, etc.). The worker doesn't get MSAA because the OffscreenCanvas
/// surface format negotiation matches the browser-WebGPU non-sRGB swap case
/// (composite + sample_count=1).
async fn init_renderer<A: App + 'static>(
    scope: DedicatedWorkerGlobalScope,
    canvas: OffscreenCanvas,
    width: u32,
    height: u32,
) -> Result<()>
where
    A::Space: 'static,
{
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    // The OffscreenCanvas IS the surface target. wgpu 27 supports this via
    // `SurfaceTarget::OffscreenCanvas`; the resulting Surface behaves like
    // a regular swapchain-backed surface from then on.
    //
    // We clone the canvas reference (cheap: it's a JsValue ref-count bump,
    // not a pixel copy) so the WorkerRunner can keep its own handle for
    // resize calls. The clone has SHARED ownership of the same underlying
    // browser-owned OffscreenCanvas; mutations on one are visible on the
    // other.
    let canvas_for_runner = canvas.clone();
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas))
        .context("create_surface from OffscreenCanvas")?;

    // Hand the surface off to the shared `RenderDevice::from_surface`
    // setup. From here the worker path is feature-equivalent to the
    // windowed-mode path: same adapter selection, same MSAA negotiation,
    // same sRGB composite dance. The size passed is the OffscreenCanvas'
    // pixel dimensions at init; resize events are plumbed via InputMessage.
    let size = winit::dpi::PhysicalSize::new(width, height);
    let rd = RenderDevice::from_surface(
        instance, surface, size,
        // Worker mode: no MSAA for now. The composite pass for the
        // non-sRGB browser-WebGPU surface forces sample_count=1 anyway
        // (see RenderDevice::new's `effective_msaa` logic), so this is
        // a no-op even when set higher.
        1,
    )
    .await
    .context("RenderDevice::from_surface")?;
    tracing::info!(
        "rye_app::wasm::worker: RenderDevice ready (target_format={:?}, sample_count={})",
        rd.target_format(),
        rd.sample_count()
    );

    // Device pixel ratio for the worker's egui. We don't have direct
    // access to `window.devicePixelRatio` from inside the worker
    // (workers have no Window); the main thread sent us a value via
    // the init message (already applied to canvas dimensions). Recover
    // it from width / canvas.client_width if needed; for now, derive
    // it as width / css_logical_width assumption (1x equivalent) and
    // accept that worker mode shows egui at 1pt:1px scale. Plumbing
    // DPR explicitly through the init message is a future improvement.
    let pixels_per_point = 1.0; // see comment above; minor cosmetic issue.

    let mut runner =
        WorkerRunner::<A>::setup(rd, canvas_for_runner, width, height, pixels_per_point)
            .await
            .context("WorkerRunner::setup")?;

    // Render exactly ONE preview frame. The launch overlay on main side
    // uses `backdrop-filter: blur(...)` to blur whatever's rendered on
    // the canvas; this single frame becomes the blurred preview the
    // viewer sees BEFORE clicking. After this call, the canvas backing
    // store holds the demo's initial state; same content the worker
    // would render on the very first RAF tick if it had started, just
    // displayed via the placeholder canvas instead of through a live
    // RAF loop.
    runner.frame().context("preview frame")?;
    tracing::info!(
        "rye_app::wasm::worker: preview frame rendered; awaiting Start to begin RAF loop"
    );

    // Self-referential RAF closure. Captures the runner via Rc<RefCell>
    // and re-schedules itself each frame. Standard wasm-bindgen pattern.
    let runner = Rc::new(RefCell::new(runner));
    let raf_cb = Rc::new(RefCell::new(None::<Closure<dyn FnMut(f64)>>));
    let raf_cb_for_closure = raf_cb.clone();
    let scope_for_closure = scope.clone();
    let runner_for_closure = runner.clone();

    *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move |_timestamp: f64| {
        // Borrow the runner mutably for one frame. The closure is the
        // sole owner of mutable access on this single-threaded worker;
        // no contention possible.
        if let Err(e) = runner_for_closure.borrow_mut().frame() {
            tracing::error!("rye_app::wasm::worker: frame failed: {e:#}");
            // Stop the loop on error so the developer sees one log line,
            // not 60 per second.
            return;
        }
        let cb_ref = raf_cb_for_closure.borrow();
        if let Some(cb) = cb_ref.as_ref() {
            let _ = scope_for_closure.request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }) as Box<dyn FnMut(f64)>));

    // Stash the RAF kickoff in the thread_local so the `Start` message
    // handler (in `apply_message`) can invoke it when the user clicks.
    // We don't kick off RAF here; the launch overlay is still visible
    // and we want the canvas to keep showing the preview frame until
    // the user opts in.
    let scope_for_kickoff = scope.clone();
    let raf_cb_for_kickoff = raf_cb.clone();
    let kickoff: Box<dyn FnOnce()> = Box::new(move || {
        let cb_ref = raf_cb_for_kickoff.borrow();
        if let Some(cb) = cb_ref.as_ref() {
            if let Err(e) = scope_for_kickoff.request_animation_frame(cb.as_ref().unchecked_ref()) {
                tracing::error!("rye_app::wasm::worker: RAF kickoff failed: {e:?}");
            }
        }
    });
    RAF_KICKOFF.with(|k| *k.borrow_mut() = Some(kickoff));

    // If a Start landed during setup (user clicked the overlay before
    // wgpu init completed), self-trigger now. Without this, the click
    // would have dropped the Start (kickoff wasn't ready yet) and the
    // overlay-removed-but-demo-frozen state would be permanent.
    if START_REQUESTED.with(|s| s.replace(false)) {
        if let Some(kickoff) = RAF_KICKOFF.with(|k| k.borrow_mut().take()) {
            tracing::info!(
                "rye_app::wasm::worker: Start was requested during init; kicking off now"
            );
            kickoff();
        }
    }

    // Both `raf_cb` and `runner` need to outlive this function so the
    // closure + runner state survive across RAF callbacks (and the
    // wait-for-Start window before the first RAF tick).
    Box::leak(Box::new(raf_cb));
    Box::leak(Box::new(runner));

    Ok(())
}

thread_local! {
    /// Pending one-shot "kick off the RAF loop" closure. Populated by
    /// `init_renderer` after the preview frame; consumed by
    /// `handle_message` on the `Start` variant. `None` after
    /// consumption (subsequent Start messages no-op).
    ///
    /// Uses `Box<dyn FnOnce>` because the kickoff captures owned
    /// closure-state (the `Rc<RefCell<Closure>>` for the RAF callback
    /// + the scope clone) and is intentionally consumable once.
    static RAF_KICKOFF: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);

    /// Set to `true` if a Start message arrives BEFORE `init_renderer`
    /// has finished stashing the kickoff. Without this, an
    /// over-eager click during the ~200ms wgpu+egui setup window
    /// would drop the Start and freeze the demo on its preview
    /// frame (overlay already removed by the click handler, no RAF
    /// loop started). `init_renderer` checks this at the end of
    /// setup and self-triggers if it's been set.
    static START_REQUESTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Per-worker lifecycle state: owns the RenderDevice + the user's App
/// + the wall-clock / tick bookkeeping the existing main-thread Runner
/// owns. Sits inside the RAF closure via `Rc<RefCell>`.
///
/// Drives the full App lifecycle: `App::update` + `App::ui` (through
/// [`WorkerUi`]) + `App::record`, plus input fan-out via [`InputState`].
struct WorkerRunner<A: App + 'static>
where
    A::Space: 'static,
{
    rd: RenderDevice,
    /// Held so resize can update the canvas's pixel backing-store dimensions
    /// before reconfiguring the wgpu surface. Without setting `width`/`height`
    /// on the OffscreenCanvas, the underlying backing wouldn't track the new
    /// surface configuration and the render would silently stretch/scissor.
    canvas: OffscreenCanvas,
    #[allow(dead_code)] // held alive so any cached pipelines/shader handles stay valid
    shader_db: ShaderDb,
    #[allow(dead_code)] // wasm stub today; native parity in case the trait grows
    watcher: Option<AssetWatcher>,
    app: A,
    /// Input accumulator that converts our typed InputMessage stream into
    /// the FrameInput shape rye-camera + rye-input expect. Fed by
    /// `apply_message`, drained by `take_frame` once per frame.
    input: InputState,
    /// Worker-side egui integration (parallel to rye-egui's UiIntegration
    /// but without winit dependency). Receives raw events from
    /// `apply_message` and paints into the same encoder as App::record.
    ui: WorkerUi,
    /// Pixel dimensions stored separately so we don't have to round-trip
    /// through RenderDevice to give egui the right `size_in_pixels`.
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
    start: web_time::Instant,
    last_update_at: Option<web_time::Instant>,
    tick_index: u64,
    /// Fixed-timestep accumulator. Matches the windowed runner so demos
    /// that read `FrameCtx::n_ticks` (e.g. `dt_secs = n_ticks / 60.0`)
    /// see ticks fire on the same cadence in browser as on native.
    /// Hardcoded to 60Hz to match `RunConfig::default()`; if any demo
    /// ever wants a different rate, RunConfig will need to be wired
    /// through to the worker (currently set up via postMessage with no
    /// config payload).
    timestep: FixedTimestep,
    /// Mirror of the windowed runner's `max_ticks_per_frame` cap, kept
    /// at the windowed default (4) so a slow-rendering frame doesn't
    /// catch up by running 60 ticks in a row and spiraling further.
    max_ticks_per_frame: usize,
    _marker: PhantomData<A::Space>,
}

impl<A: App + 'static> WorkerRunner<A>
where
    A::Space: 'static,
{
    /// Construct the runner: build ShaderDb + AssetWatcher (wasm stub) +
    /// WorkerUi + invoke `A::setup`. Async because `A::setup` may itself
    /// await on asset loading or device-feature probes; in practice for
    /// the existing demos it's synchronous and returns immediately.
    async fn setup(
        rd: RenderDevice,
        canvas: OffscreenCanvas,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
    ) -> Result<Self> {
        let mut shader_db = ShaderDb::new(rd.device.clone());
        // AssetWatcher init failure isn't fatal: demos work without hot-
        // reload. On wasm32 the watcher is a no-op stub so Ok(_) always.
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

        // WorkerUi is constructed AFTER A::setup so the App's own
        // pipeline-warming (running inside setup) finishes first.
        // egui-wgpu's pipeline compilation happens lazily on first
        // paint; we accept that cost on the first real frame for now
        // (N3-style warming for the worker-side egui pipelines is a
        // follow-up).
        let ui = WorkerUi::new(
            &rd.device,
            rd.target_format(),
            rd.sample_count(),
            width_px,
            height_px,
            pixels_per_point,
        );

        Ok(Self {
            rd,
            canvas,
            shader_db,
            watcher,
            app,
            input: InputState::default(),
            ui,
            width_px,
            height_px,
            pixels_per_point,
            start: web_time::Instant::now(),
            last_update_at: None,
            tick_index: 0,
            timestep: FixedTimestep::new(60),
            max_ticks_per_frame: 4,
            _marker: PhantomData,
        })
    }

    /// Update the OffscreenCanvas pixel backing-store size + reconfigure
    /// the wgpu surface + update egui's screen rect. Called by the
    /// worker's message handler in response to an `InputMessage::Resize`
    /// from the main thread. Zero-sized resizes are no-ops (mirrors
    /// RenderDevice::resize's defensive check).
    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.rd.resize(winit::dpi::PhysicalSize::new(width, height));
        self.width_px = width;
        self.height_px = height;
        self.ui.resize(width, height, self.pixels_per_point);
        tracing::info!("rye_app::wasm::worker: resized to {width}x{height}");
    }

    /// Apply one `InputMessage`. Resize updates the surface; the other
    /// variants route into `InputState` for the App's per-frame `FrameInput`,
    /// and fan out to egui via `RawInput` so HUD widgets can respond.
    ///
    /// `App::on_event` (the path that handles winit `WindowEvent`s, e.g.
    /// tesseract_demo's KeyF / KeyT / Space / KeyR hotkeys) is deliberately
    /// NOT plumbed here; constructing winit's `KeyEvent` requires private
    /// platform-specific fields. Tesseract_demo's HOTKEYS won't work in
    /// worker mode yet; the camera + WASD axes will because those go
    /// through `FrameInput`.
    fn apply_message(&mut self, msg: InputMessage) {
        // Fan out to egui FIRST so egui sees the event even if it's
        // also consumed by InputState below. Egui filters based on
        // pointer position vs widget bounds; double-feeding is fine.
        self.ui.record_input(&msg);

        match msg {
            InputMessage::Resize { width, height } => {
                self.resize(width, height);
                // Render one frame at the new size so the preview
                // refreshes (no CSS-stretched display). When the RAF
                // loop is already running this is just one extra frame
                // before the next RAF tick; harmless. When we're in
                // pre-Start preview mode, this is the only frame that
                // happens, and it's what the launch overlay's
                // backdrop-filter blurs.
                if let Err(e) = self.frame() {
                    tracing::error!("rye_app::wasm::worker: post-resize frame failed: {e:#}");
                }
            }
            InputMessage::MouseMove { x, y, .. } => {
                self.input.cursor_moved(x as f64, y as f64);
            }
            InputMessage::MouseButton {
                button, pressed, ..
            } => {
                let button = crate::keymap::mouse_button_winit(button);
                let state = if pressed {
                    ElementState::Pressed
                } else {
                    ElementState::Released
                };
                self.input.mouse_input(button, state);
            }
            InputMessage::MouseWheel { dx, dy } => {
                self.input.mouse_wheel(MouseScrollDelta::LineDelta(dx, dy));
            }
            InputMessage::Key {
                ref code, pressed, ..
            } => {
                if let Some(code) = crate::keymap::keycode_winit(code) {
                    let state = if pressed {
                        ElementState::Pressed
                    } else {
                        ElementState::Released
                    };
                    self.input.key_input(PhysicalKey::Code(code), state);
                }
                // Hotkey routing via App::on_event deferred (see fn doc).
            }
            InputMessage::Focus(focused) => {
                if !focused {
                    // Mirror rye-input's focus-loss convention: drop held
                    // buttons + invalidate cursor delta so re-focus doesn't
                    // snap.
                    self.input.release_buttons();
                    self.input.cursor_invalidated();
                }
            }
            InputMessage::Visibility(_) => {
                // Visibility isn't load-bearing for tesseract_demo right
                // now; future work could pause continuous animation here.
            }
            InputMessage::Start => {
                // Handled directly by `handle_message` (special-cased
                // alongside `init`) because Start has to fire BEFORE
                // the RAF loop starts; the per-frame queue only drains
                // inside that loop, so routing through it would
                // deadlock.
            }
        }
    }

    /// One frame: drain input queue, dt, App::update, begin_frame,
    /// App::record, composite, submit, present. Wraps
    /// `rye_time::frame_trace::scope` so the same telemetry the windowed
    /// runner emits is available here.
    fn frame(&mut self) -> Result<()> {
        rye_time::frame_trace::begin_frame();
        let _frame_scope = rye_time::frame_trace::scope("frame");

        // Drain queued input messages. Resize updates the surface; mouse
        // and keyboard variants route into `InputState` and fan out to
        // egui via `RawInput` so HUD widgets see them too.
        for msg in messages::drain_messages() {
            self.apply_message(msg);
        }

        // dt: wall-clock since previous update. First frame falls back
        // to 1/60 so the App doesn't see a 0 dt that breaks integrators.
        let now = web_time::Instant::now();
        let dt = match self.last_update_at {
            Some(prev) => now.duration_since(prev).as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.last_update_at = Some(now);

        // Fixed-timestep ticks. Mirrors the windowed runner's structure
        // so demos that read `FrameCtx::n_ticks` (or compute `dt_secs =
        // n_ticks / 60.0` to integrate spin / freecam translation /
        // similar) behave the same in browser as on native.
        let n_capped;
        {
            let _scope = rye_time::frame_trace::scope("sim-ticks");
            let ticks = self.timestep.advance(now);
            let n_ticks = ticks.count();
            n_capped = n_ticks.min(self.max_ticks_per_frame);
            let tick_dt = 1.0 / 60.0;
            for _ in 0..n_capped {
                let mut tctx = TickCtx {
                    time: self.start.elapsed().as_secs_f32(),
                    tick: self.tick_index,
                };
                self.app.tick(tick_dt, &mut tctx);
                self.tick_index = self.tick_index.wrapping_add(1);
            }
        }

        // App::update with the FrameInput accumulated from this frame's
        // InputMessage events. `take_frame` returns the drained snapshot
        // and resets per-tick deltas (mouse motion + scroll).
        let input = self.input.take_frame();
        let ui_has_focus = self.ui.wants_input;
        {
            let _scope = rye_time::frame_trace::scope("app-update");
            let mut fctx = FrameCtx {
                rd: &self.rd,
                input,
                time: self.start.elapsed().as_secs_f32(),
                fps: 0.0, // worker-side FPS bookkeeping not yet wired; reads as 0.0.
                n_ticks: n_capped,
                tick: self.tick_index,
                dt,
                ui_has_focus,
                _non_exhaustive: PhantomData,
            };
            self.app.update(&mut fctx);

            // App::ui; egui frame begin, App builds widgets, paint happens
            // after the scene render below. Same lifecycle as the windowed
            // runner: begin_frame -> ctx clone -> App::ui -> paint at end.
            let egui_ctx = self.ui.begin_frame().clone();
            let _scope = rye_time::frame_trace::scope("app-ui");
            self.app.ui(&egui_ctx, &mut fctx);
        }

        // begin_frame -> record -> composite -> submit -> present.
        let (frame, swap_view) = self.rd.begin_frame().context("RenderDevice::begin_frame")?;
        let render_view = self
            .rd
            .msaa_view()
            .or(self.rd.scene_view())
            .unwrap_or(&swap_view);

        let mut encoder = self
            .rd
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rye_app::wasm::worker::frame"),
            });

        {
            let _scope = rye_time::frame_trace::scope("app-record");
            let mut ctx = RenderCtx {
                rd: &self.rd,
                view: render_view,
                encoder: &mut encoder,
            };
            self.app.record(&mut ctx).context("App::record")?;
        }

        // egui paint into the same encoder, overlaid on the scene. MSAA
        // resolve target is the swap view when MSAA is on (sample_count
        // is 1 for the current worker config so resolve_target is None).
        {
            let _scope = rye_time::frame_trace::scope("ui-paint");
            let resolve_target = (self.rd.sample_count() > 1).then_some(&swap_view);
            self.ui.paint(
                &self.rd.device,
                &self.rd.queue,
                &mut encoder,
                render_view,
                resolve_target,
            );
        }

        // Composite pass when the swap is non-sRGB (browser-WebGPU).
        if self.rd.scene_view().is_some() {
            let _scope = rye_time::frame_trace::scope("composite");
            self.rd.composite_to_swap(&mut encoder, &swap_view);
        }

        {
            let _scope = rye_time::frame_trace::scope("present");
            self.rd.queue.submit(Some(encoder.finish()));
            frame.present();
        }

        rye_time::frame_trace::end_frame();
        Ok(())
    }
}

fn worker_scope() -> Result<DedicatedWorkerGlobalScope> {
    js_sys::global()
        .dyn_into::<DedicatedWorkerGlobalScope>()
        .map_err(|_| anyhow!("not running in a DedicatedWorkerGlobalScope"))
}

/// Install console panic hook + tracing-to-DevTools routing exactly once
/// per JS context (main thread has its own instance, worker has its own
///; std::sync::Once is per-process). Both `run` (worker) and
/// `launch_on_click` (main) call this so a demo using worker mode gets
/// the same observability surface as the legacy `run_with_config` path
/// did.
///
/// `tracing_wasm::set_as_global_default` panics on second call within a
/// context; the `Once` guard makes the call site safe under any caller
/// pattern.
pub(super) fn install_logging_idempotent() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
    });
}

// Main-thread launcher + DOM forwarders moved to wasm/main_launcher.rs.
