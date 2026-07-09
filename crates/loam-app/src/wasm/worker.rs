//! Web Worker mode for loam demos. Moves the render loop into a worker so V8's
//! GC pauses don't block the visible page.
//!
//! The worker receives an OffscreenCanvas via postMessage, creates a wgpu
//! Surface from it, and drives a rolled-own RAF loop running the full App
//! lifecycle plus an egui overlay (via [`WorkerUi`], which translates
//! [`InputMessage`] directly into `egui::RawInput`).
//!
//! winit 0.30 has no `WorkerGlobalScope` support (issue #1518):
//! `web_sys::window()` panics in worker context, so this path is rolled
//! ourselves; the pieces we need (RAF, surface creation, message passing)
//! are all available without winit.
//!
//! The same wasm bundle runs on the main thread and inside the worker;
//! [`crate::wasm::is_worker_context`] lets `main` branch into [`run`] vs
//! [`launch_on_click`].

use anyhow::{anyhow, Context, Result};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, OffscreenCanvas};

use super::messages::{self, InputMessage};
use super::worker_ui::WorkerUi;
use crate::{App, FrameCtx, RenderCtx, SetupCtx};
use loam_asset::AssetWatcher;
use loam_input::InputState;
use loam_render::device::RenderDevice;
use loam_shader::ShaderDb;
use loam_time::FixedTimestep;
use winit::event::{ElementState, MouseScrollDelta};
use winit::keyboard::PhysicalKey;

/// Worker entry. Installs a `message` listener that waits for the
/// canvas-transfer init, then constructs `RenderDevice` + `WorkerRunner<A>`
/// and starts the RAF loop. Returns synchronously; the work happens in the
/// message + RAF callbacks, which the `forget()` calls keep alive.
pub fn run<A: App + 'static>() -> Result<()>
where
    A::Space: 'static,
{
    // Worker has its own JS heap + console.
    install_logging_idempotent();

    tracing::debug!("loam_app::wasm::worker::run: entry");

    let scope = worker_scope()?;
    let scope_for_handler = scope.clone();

    // `addEventListener` over `set_onmessage`: the former reliably delivers
    // messages queued before the listener installs; the latter has spec
    // ambiguity about when queued messages flush.
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        tracing::debug!("loam_app::wasm::worker: message handler firing");
        if let Err(e) = handle_message::<A>(&scope_for_handler, event) {
            tracing::error!("loam_app::wasm::worker: message handler failed: {e:#}");
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    scope
        .add_event_listener_with_callback("message", on_message.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("addEventListener('message'): {e:?}"))?;
    on_message.forget();

    // Signal readiness before main posts Init. Firefox empirically drops
    // messages posted to a worker before its listener installs; this
    // handshake makes the ordering explicit across browsers.
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

    tracing::info!("loam_app::wasm::worker::run: message listener installed + ready posted");

    Ok(())
}

/// Dispatch a single inbound `postMessage`. `init` and `start` are special-
/// cased here; other kinds go through [`messages::parse_non_init`] onto the
/// per-frame queue.
fn handle_message<A: App + 'static>(
    scope: &DedicatedWorkerGlobalScope,
    event: MessageEvent,
) -> Result<()>
where
    A::Space: 'static,
{
    let data: JsValue = event.data();

    let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|v| v.as_string());
    if kind.as_deref() == Some("start") {
        // If init stashed the kickoff, run it; otherwise flag the request
        // so init_renderer self-triggers when ready (no-op if already
        // started).
        if let Some(kickoff) = RAF_KICKOFF.with(|k| k.borrow_mut().take()) {
            tracing::info!("loam_app::wasm::worker: Start received, kicking off RAF loop");
            kickoff();
        } else {
            START_REQUESTED.with(|s| s.set(true));
            tracing::info!("loam_app::wasm::worker: Start received before kickoff ready; queued");
        }
        return Ok(());
    }
    if kind.as_deref() == Some("pause") {
        PAUSED.with(|p| p.set(true));
        // Synthetic focus-loss releases buttons held at pause time; drained
        // on the first resumed frame.
        messages::enqueue(InputMessage::Focus(false));
        tracing::info!("loam_app::wasm::worker: pause received; RAF chain will halt");
        return Ok(());
    }
    if kind.as_deref() == Some("resume") {
        let was_paused = PAUSED.with(|p| p.replace(false));
        // Restart only a halted chain: pause+resume within one frame gap
        // leaves the original RAF pending, and resume before Start must not
        // bypass the launch flow.
        if was_paused && LOOP_STARTED.with(|s| s.get()) && !RAF_PENDING.with(|p| p.get()) {
            RAF_RESTART.with(|r| {
                if let Some(restart) = r.borrow().as_ref() {
                    tracing::info!("loam_app::wasm::worker: resume received; restarting RAF");
                    restart();
                }
            });
        }
        return Ok(());
    }
    // No frame drains the queue while paused: drop pointer/key/wheel so it
    // stays bounded and stale input cannot replay. Resize still queues.
    if PAUSED.with(|p| p.get())
        && matches!(
            kind.as_deref(),
            Some("mouse_move" | "mouse_button" | "mouse_wheel" | "key")
        )
    {
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
        let read_str = |key: &str| {
            js_sys::Reflect::get(&data, &JsValue::from_str(key))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default()
        };
        crate::args::set_query_override(read_str("search"), read_str("hash"));

        tracing::info!(
            "loam_app::wasm::worker: received init ({width}x{height}); spawning wgpu setup"
        );
        let scope_for_render = scope.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = init_renderer::<A>(scope_for_render, canvas, width, height).await {
                tracing::error!("loam_app::wasm::worker: init_renderer failed: {e:#}");
            }
        });
        return Ok(());
    }

    match messages::parse_non_init(&data)? {
        Some(msg) => messages::enqueue(msg),
        None => {
            if let Some(k) = kind {
                tracing::warn!("loam_app::wasm::worker: unknown message kind '{k}'");
            }
        }
    }
    Ok(())
}

/// Build `RenderDevice` from the worker-owned OffscreenCanvas, run
/// `App::setup`, and start the RAF loop. Uses `RenderDevice::from_surface`
/// so the wgpu setup matches the windowed-mode path.
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

    // Clone is a JsValue ref-count bump (shared ownership, not a pixel
    // copy) so the WorkerRunner keeps its own handle for resize calls.
    let canvas_for_runner = canvas.clone();
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas))
        .context("create_surface from OffscreenCanvas")?;

    let size = winit::dpi::PhysicalSize::new(width, height);
    let rd = RenderDevice::from_surface(
        instance, surface, size,
        // No MSAA: the non-sRGB browser-WebGPU composite pass forces
        // sample_count=1 anyway (RenderDevice::new's `effective_msaa`).
        1,
    )
    .await
    .context("RenderDevice::from_surface")?;
    tracing::info!(
        "loam_app::wasm::worker: RenderDevice ready (target_format={:?}, sample_count={})",
        rd.target_format(),
        rd.sample_count()
    );

    // Workers have no Window, so no `window.devicePixelRatio`. Until DPR is
    // plumbed through the init message, egui renders at 1pt:1px.
    let pixels_per_point = 1.0;

    let mut runner =
        WorkerRunner::<A>::setup(rd, canvas_for_runner, width, height, pixels_per_point)
            .await
            .context("WorkerRunner::setup")?;

    // The preview frame becomes the backdrop-blurred thumbnail the viewer
    // sees before clicking. The warmup frames force wgpu's lazy pipeline
    // compilation (browser WebGPU defers `create_render_pipeline` until
    // first use) up front so the click -> running transition has no second
    // hitch. Safe because the default `rotate` state is false, so warmup
    // advances no simulation state.
    //
    // 11 frames (1 + 10) empirically covers polytope_playground's
    // first-frame compilation on Chrome and Firefox WebGPU. Each frame
    // posts `preview_progress` to advance the `#loam-page-loader` bar.
    const WARMUP_FRAMES: usize = 10;
    const TOTAL_FRAMES: usize = 1 + WARMUP_FRAMES;
    let post_progress = |step: usize| {
        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("kind"),
            &JsValue::from_str("preview_progress"),
        );
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("pct"),
            &JsValue::from_f64(step as f64 / TOTAL_FRAMES as f64),
        );
        let _ = scope.post_message(&msg);
    };
    runner.frame().context("preview frame")?;
    post_progress(1);
    for i in 0..WARMUP_FRAMES {
        runner.frame().context("warmup frame")?;
        post_progress(2 + i);
    }
    tracing::info!(
        "loam_app::wasm::worker: preview + {WARMUP_FRAMES} warmup frames rendered; \
         awaiting Start to begin RAF loop"
    );
    // Preview frame is on the canvas and pipelines are warm; main promotes
    // the launch overlay to `.ready`.
    {
        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("kind"),
            &JsValue::from_str("preview_ready"),
        );
        if let Err(e) = scope.post_message(&msg) {
            tracing::warn!("loam_app::wasm::worker: post preview_ready failed: {e:?}");
        }
    }

    // Self-referential RAF closure that re-schedules itself each frame
    // (standard wasm-bindgen pattern).
    let runner = Rc::new(RefCell::new(runner));
    let raf_cb = Rc::new(RefCell::new(None::<Closure<dyn FnMut(f64)>>));
    let raf_cb_for_closure = raf_cb.clone();
    let scope_for_closure = scope.clone();
    let runner_for_closure = runner.clone();

    *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move |_timestamp: f64| {
        RAF_PENDING.with(|p| p.set(false));
        // Halted chain keeps the last presented frame as the overlay backdrop.
        if PAUSED.with(|p| p.get()) {
            return;
        }
        if let Err(e) = runner_for_closure.borrow_mut().frame() {
            tracing::error!("loam_app::wasm::worker: frame failed: {e:#}");
            // Stop the loop on error: one log line, not 60 per second.
            return;
        }
        let cb_ref = raf_cb_for_closure.borrow();
        if let Some(cb) = cb_ref.as_ref() {
            if scope_for_closure
                .request_animation_frame(cb.as_ref().unchecked_ref())
                .is_ok()
            {
                RAF_PENDING.with(|p| p.set(true));
            }
        }
    }) as Box<dyn FnMut(f64)>));

    // Stash the kickoff for the `Start` handler instead of starting RAF
    // now: the canvas keeps showing the preview frame until the user
    // clicks.
    let scope_for_kickoff = scope.clone();
    let raf_cb_for_kickoff = raf_cb.clone();
    let kickoff: Box<dyn FnOnce()> = Box::new(move || {
        let cb_ref = raf_cb_for_kickoff.borrow();
        if let Some(cb) = cb_ref.as_ref() {
            match scope_for_kickoff.request_animation_frame(cb.as_ref().unchecked_ref()) {
                Ok(_) => {
                    LOOP_STARTED.with(|s| s.set(true));
                    RAF_PENDING.with(|p| p.set(true));
                }
                Err(e) => tracing::error!("loam_app::wasm::worker: RAF kickoff failed: {e:?}"),
            }
        }
    });
    RAF_KICKOFF.with(|k| *k.borrow_mut() = Some(kickoff));

    let scope_for_restart = scope.clone();
    let raf_cb_for_restart = raf_cb.clone();
    let runner_for_restart = runner.clone();
    let restart: Box<dyn Fn()> = Box::new(move || {
        runner_for_restart.borrow_mut().reset_frame_clock();
        let cb_ref = raf_cb_for_restart.borrow();
        if let Some(cb) = cb_ref.as_ref() {
            match scope_for_restart.request_animation_frame(cb.as_ref().unchecked_ref()) {
                Ok(_) => RAF_PENDING.with(|p| p.set(true)),
                Err(e) => tracing::error!("loam_app::wasm::worker: RAF restart failed: {e:?}"),
            }
        }
    });
    RAF_RESTART.with(|r| *r.borrow_mut() = Some(restart));

    // If a Start landed during setup (click before wgpu init finished),
    // self-trigger now; otherwise the demo would freeze on the preview
    // frame with the overlay already removed.
    if START_REQUESTED.with(|s| s.replace(false)) {
        if let Some(kickoff) = RAF_KICKOFF.with(|k| k.borrow_mut().take()) {
            tracing::info!(
                "loam_app::wasm::worker: Start was requested during init; kicking off now"
            );
            kickoff();
        }
    }

    // Both must outlive this function: the closure + runner state survive
    // across RAF callbacks and the wait-for-Start window.
    Box::leak(Box::new(raf_cb));
    Box::leak(Box::new(runner));

    Ok(())
}

thread_local! {
    /// One-shot RAF-loop kickoff. Populated by `init_renderer` after the
    /// preview frame, consumed by `handle_message` on `Start`; `None` after
    /// (subsequent Start messages no-op).
    static RAF_KICKOFF: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);

    /// Set when a Start arrives before `init_renderer` stashed the kickoff.
    /// Checked at the end of setup so an eager click during the wgpu+egui
    /// setup window still starts the loop instead of freezing the preview.
    static START_REQUESTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Embed deactivated: the RAF chain halts and input messages drop.
    static PAUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// A RAF callback is scheduled. Resume must not re-request while the
    /// original chain is alive (pause+resume within one frame gap), or two
    /// interleaved chains run forever.
    static RAF_PENDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Resume before Start must not bypass the launch flow.
    static LOOP_STARTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Restarts a halted chain on `resume`, re-anchoring the frame clock
    /// first. Unlike `RAF_KICKOFF`, reusable across pause cycles.
    static RAF_RESTART: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
}

/// Per-worker lifecycle state: owns the RenderDevice, the user's App, and
/// the wall-clock / tick bookkeeping. Drives the full App lifecycle
/// (`update` + `ui` via [`WorkerUi`] + `record`) plus input fan-out via
/// [`InputState`]. Lives inside the RAF closure via `Rc<RefCell>`.
struct WorkerRunner<A: App + 'static>
where
    A::Space: 'static,
{
    rd: RenderDevice,
    /// Held so resize can set the OffscreenCanvas backing-store dimensions
    /// before reconfiguring the surface; without it the render stretches.
    canvas: OffscreenCanvas,
    #[allow(dead_code)] // held alive so cached pipeline/shader handles stay valid
    shader_db: ShaderDb,
    #[allow(dead_code)] // wasm stub today; native parity in case the trait grows
    watcher: Option<AssetWatcher>,
    app: A,
    /// Converts the typed InputMessage stream into the FrameInput shape
    /// loam-camera + loam-input expect. Drained by `take_frame` per frame.
    input: InputState,
    /// Worker-side egui integration (parallel to loam-egui's UiIntegration
    /// but without the winit dependency).
    ui: WorkerUi,
    /// Pixel dimensions kept separately so egui gets `size_in_pixels`
    /// without a round-trip through RenderDevice.
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
    start: web_time::Instant,
    last_update_at: Option<web_time::Instant>,
    /// FPS-cap anchor: the previous frame's IDEAL deadline (not its wake-up
    /// time), so RAF jitter doesn't compound into alternating-skip at the
    /// target-period / refresh-rate boundary. Mirrors the native runner's
    /// `last_redraw_at`. `None` when uncapped.
    last_redraw_anchor: Option<web_time::Instant>,
    tick_index: u64,
    /// Fixed-timestep accumulator at 60Hz (matching `RunConfig::default()`)
    /// so demos reading `FrameCtx::n_ticks` see the native cadence. A
    /// different rate would need RunConfig plumbed through postMessage.
    timestep: FixedTimestep,
    /// Mirror of the windowed runner's `max_ticks_per_frame` (4) so a slow
    /// frame doesn't spiral by running a burst of catch-up ticks.
    max_ticks_per_frame: usize,
    _marker: PhantomData<A::Space>,
}

impl<A: App + 'static> WorkerRunner<A>
where
    A::Space: 'static,
{
    /// Build ShaderDb + AssetWatcher (wasm stub) + WorkerUi and invoke
    /// `A::setup`. Async because `A::setup` may await on asset loading.
    async fn setup(
        rd: RenderDevice,
        canvas: OffscreenCanvas,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
    ) -> Result<Self> {
        let mut shader_db = ShaderDb::new(rd.device.clone());
        // Watcher failure isn't fatal (demos work without hot-reload); the
        // wasm32 watcher is a no-op stub anyway.
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

        // Constructed after A::setup so the App's pipeline-warming runs
        // first. egui-wgpu compiles its pipelines lazily on first paint;
        // that cost lands on the first real frame for now.
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
            last_redraw_anchor: None,
            tick_index: 0,
            timestep: FixedTimestep::new(60),
            max_ticks_per_frame: 4,
            _marker: PhantomData,
        })
    }

    /// Re-anchor after a pause so the first resumed frame sees a normal dt.
    /// Anchors on `now`, not `None`: `last_update_at == None` doubles as the
    /// pre-Start preview flag in `apply_message`. `FixedTimestep` needs no
    /// reset; `advance` bounds catch-up and drains excess itself.
    fn reset_frame_clock(&mut self) {
        self.last_update_at = Some(web_time::Instant::now());
        self.last_redraw_anchor = None;
    }

    /// Resize the canvas backing store, reconfigure the surface, and update
    /// egui's screen rect. Zero-sized resizes are no-ops.
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
        tracing::info!("loam_app::wasm::worker: resized to {width}x{height}");
    }

    /// Apply one `InputMessage`. Resize updates the surface; other variants
    /// route into `InputState` and fan out to egui via `RawInput`.
    ///
    /// `App::on_event` (winit `WindowEvent`s) is deliberately not plumbed
    /// here: constructing winit's `KeyEvent` needs private platform fields.
    /// Camera + WASD axes work via `FrameInput`; winit-only hotkeys do not.
    fn apply_message(&mut self, msg: InputMessage) {
        // Fan out to egui first; it filters by pointer position vs widget
        // bounds, so double-feeding with InputState below is fine.
        self.ui.record_input(&msg);

        match msg {
            InputMessage::Resize { width, height } => {
                self.resize(width, height);
                // Render once only in pre-Start preview mode (to refresh the
                // backdrop-blur thumbnail). Once the RAF loop runs, the next
                // tick renders at the new size; calling frame() here would
                // re-drain queued Resize events recursively and overflow
                // wgpu's command queue under a sustained drag.
                if self.last_update_at.is_none() {
                    if let Err(e) = self.frame() {
                        tracing::error!(
                            "loam_app::wasm::worker: pre-Start resize frame failed: {e:#}"
                        );
                    }
                }
            }
            InputMessage::MouseMove { x, y, dx, dy, .. } => {
                // `dx`/`dy` are `movementX/Y` summed across coalesced events;
                // correct raw-motion source under Pointer Lock too, where
                // `offsetX/Y` pins to the locked center and would read zero.
                self.input.accumulate_raw_motion(dx as f64, dy as f64);
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
                    // Route to `App::on_key` so hotkeys work like native.
                    // `App::on_event` can't run here: winit's `KeyEvent` has
                    // a `pub(crate)` field we can't construct.
                    let ui_has_focus = self.ui.wants_input;
                    let mut fctx = FrameCtx {
                        rd: &self.rd,
                        input: loam_input::FrameInput::default(),
                        time: self.start.elapsed().as_secs_f32(),
                        fps: 0.0,
                        n_ticks: 0,
                        tick: self.tick_index,
                        dt: 0.0,
                        ui_has_focus,
                        _non_exhaustive: PhantomData,
                    };
                    self.app.on_key(code, state, &mut fctx);
                }
            }
            InputMessage::Focus(focused) => {
                if !focused {
                    // loam-input focus-loss convention: drop held buttons +
                    // invalidate cursor delta so re-focus doesn't snap.
                    self.input.release_buttons();
                    self.input.cursor_invalidated();
                }
            }
            InputMessage::Visibility(_) => {}
            InputMessage::Start => {
                // Handled in `handle_message`: Start must fire before the
                // RAF loop exists, but the per-frame queue only drains
                // inside that loop.
            }
            InputMessage::PointerLockChanged(locked) => {
                // Mirror the browser's actual lock state (set by the main
                // thread's `pointerlockchange` listener) into the cursor
                // module so `current_state()` reads the truth, including
                // releases the demo didn't request (Esc, tab switch).
                let grab = if locked {
                    crate::cursor::GrabMode::Locked
                } else {
                    crate::cursor::GrabMode::None
                };
                // On wasm, visibility tracks lock state: lock auto-hides,
                // release auto-shows.
                crate::cursor::mark_applied(grab, !locked);
            }
        }
    }

    /// One frame of the worker's RAF loop: FPS cap, input drain,
    /// fixed-timestep ticks, `App::update` + `App::ui`, render, then
    /// end-of-frame host actions. Wrapped in `loam_time::frame_trace::scope`
    /// for the same telemetry the windowed runner emits.
    fn frame(&mut self) -> Result<()> {
        // FPS cap: drop callbacks firing before the next ideal deadline
        // (previous anchor + target period). Anchoring on the ideal
        // deadline, not the wake-up time, absorbs RAF jitter; otherwise a
        // target matching the refresh rate suffers alternating-skip (jitter
        // skips a callback, the next fires a period late, half-rate).
        // Mirrors the native runner. Can only lower the rate below the
        // browser RAF cadence.
        let now_raf = web_time::Instant::now();
        match crate::frame_pacing::target_period() {
            Some(target) => {
                if let Some(last) = self.last_redraw_anchor {
                    let deadline = last + target;
                    if now_raf < deadline {
                        return Ok(());
                    }
                    // Clamp catch-up after a backgrounded tab: snap to at
                    // most one period behind `now_raf` instead of queueing
                    // every missed frame at once.
                    let catch_up_floor = now_raf.checked_sub(target).unwrap_or(now_raf);
                    self.last_redraw_anchor = Some(deadline.max(catch_up_floor));
                } else {
                    // First frame under an active cap: anchor on `now`.
                    self.last_redraw_anchor = Some(now_raf);
                }
            }
            None => {
                // Uncapped: clear so a later `fps <N>` starts fresh.
                self.last_redraw_anchor = None;
            }
        }

        loam_time::frame_trace::begin_frame();
        let _frame_scope = loam_time::frame_trace::scope("frame");

        for msg in messages::drain_messages() {
            self.apply_message(msg);
        }

        // dt: wall-clock since previous update; first frame falls back to
        // 1/60 so the App doesn't see a 0 dt that breaks integrators.
        let now = web_time::Instant::now();
        let dt = match self.last_update_at {
            Some(prev) => now.duration_since(prev).as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.last_update_at = Some(now);

        // Fixed-timestep ticks via the shared `drive_fixed_ticks`, so the sim
        // cadence is identical to the windowed runner (60Hz here).
        let n_capped = crate::drive_fixed_ticks(
            &mut self.app,
            &mut self.timestep,
            &mut self.tick_index,
            self.start,
            now,
            60,
            self.max_ticks_per_frame,
        );

        // `take_frame` drains the accumulated FrameInput and resets per-tick
        // deltas (mouse motion + scroll).
        let input = self.input.take_frame();
        let ui_has_focus = self.ui.wants_input;
        {
            let _scope = loam_time::frame_trace::scope("app-update");
            let mut fctx = FrameCtx {
                rd: &self.rd,
                input,
                time: self.start.elapsed().as_secs_f32(),
                fps: 0.0, // worker-side FPS bookkeeping not yet wired.
                n_ticks: n_capped,
                tick: self.tick_index,
                dt,
                ui_has_focus,
                _non_exhaustive: PhantomData,
            };
            self.app.update(&mut fctx);

            // egui begin_frame -> App::ui -> paint after the scene render
            // below, matching the windowed runner.
            let egui_ctx = self.ui.begin_frame().clone();
            let _scope = loam_time::frame_trace::scope("app-ui");
            self.app.ui(&egui_ctx, &mut fctx);
        }

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
                label: Some("loam_app::wasm::worker::frame"),
            });

        {
            let _scope = loam_time::frame_trace::scope("app-record");
            let mut ctx = RenderCtx {
                rd: &self.rd,
                view: render_view,
                encoder: &mut encoder,
            };
            self.app.record(&mut ctx).context("App::record")?;
        }

        // egui into the same encoder, overlaid on the scene. resolve_target
        // is None at sample_count 1 (the current worker config).
        {
            let _scope = loam_time::frame_trace::scope("ui-paint");
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
            let _scope = loam_time::frame_trace::scope("composite");
            self.rd.composite_to_swap(&mut encoder, &swap_view);
        }

        {
            let _scope = loam_time::frame_trace::scope("present");
            self.rd.queue.submit(Some(encoder.finish()));
            frame.present();
        }

        // Drain this frame's cursor / DOM-action requests and post them as
        // one batched `host_action`. After `present()` so postMessage
        // latency stays off the GPU submission path (matches native).
        let (pending_grab, _pending_visible) = crate::cursor::take_pending();
        if let Some(grab) = pending_grab {
            let action = match grab {
                crate::cursor::GrabMode::Locked | crate::cursor::GrabMode::Confined => {
                    super::host_action::HostAction::PointerLockRequest
                }
                crate::cursor::GrabMode::None => super::host_action::HostAction::PointerLockRelease,
            };
            super::host_action::queue(action);
        }
        // Drained to clear the flag: warp-to-center is a no-op on wasm
        // (no cursor-position write) but leaving it set would leak to a
        // later native build.
        let _ = crate::cursor::take_pending_warp_center();
        if let Ok(scope) = worker_scope() {
            if let Err(e) = super::host_action::post_pending_actions(&scope) {
                tracing::warn!("loam_app::wasm::worker: post host_action failed: {e:#}");
            }
        }

        loam_time::frame_trace::end_frame();
        Ok(())
    }
}

fn worker_scope() -> Result<DedicatedWorkerGlobalScope> {
    js_sys::global()
        .dyn_into::<DedicatedWorkerGlobalScope>()
        .map_err(|_| anyhow!("not running in a DedicatedWorkerGlobalScope"))
}

/// Install the console panic hook + tracing-to-DevTools routing once per JS
/// context (main and worker each have their own; `Once` is per-process).
/// `tracing_wasm::set_as_global_default` panics on a second call, so the
/// guard is required.
pub(super) fn install_logging_idempotent() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
    });
}
