//! Web Worker mode for rye demos. Moves the render loop into a worker so V8's
//! GC pauses don't block the visible page.
//!
//! See `docs/devlog/context/OFFSCREEN_CANVAS_WORKERS.md` for the full
//! architectural design + phasing plan.
//!
//! ## Status
//!
//! **Phase A** (active): bare-minimum POC. Worker receives an OffscreenCanvas
//! via postMessage, creates a wgpu Surface from it, runs a rolled-own RAF
//! loop that clears the canvas to a cycling color. No App-trait integration,
//! no input handling, no egui. Validates that wgpu + OffscreenCanvas + a
//! worker-side RAF loop actually compose on Chromium with our wgpu/web-sys
//! versions. If Phase A works, Phase B adds the App + RenderDevice wiring.
//!
//! ## Why a rolled-own event loop (no winit)
//!
//! winit 0.30 doesn't support `WorkerGlobalScope` (issue #1518 since 2020;
//! `web_sys::window()` panics in worker context, breaks scale-factor / event
//! pump). Until upstream lands worker support — Bevy tried and abandoned the
//! PR — we own this code path ourselves. Trade-off accepted because the
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
use crate::{App, FrameCtx, RenderCtx, SetupCtx};
use rye_asset::AssetWatcher;
use rye_input::InputState;
use rye_render::device::RenderDevice;
use rye_shader::ShaderDb;
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
/// **Phase B1 scope**: tesseract appears in worker. No input events,
/// no egui UI overlay (Phase B3 + B4 add those).
pub fn run<A: App + 'static>() -> Result<()>
where
    A::Space: 'static,
{
    // Worker has its own JS heap + console; install panic hook + tracing
    // so unhandled errors + log lines surface in DevTools (under the
    // worker's context, selectable in DevTools' execution-context picker).
    install_logging_idempotent();

    tracing::info!("rye_app::wasm::worker::run: Phase A entry");

    let scope = worker_scope()?;
    let scope_for_handler = scope.clone();

    // The message handler ingests the OffscreenCanvas + size from the
    // first postMessage and kicks off the wgpu init. Subsequent messages
    // (Phase B's input events) reuse this same handler with a `kind`
    // dispatch.
    //
    // Use `addEventListener("message", ...)` rather than `set_onmessage`
    // because `addEventListener` is reliably retroactive: messages
    // posted to the Worker before this listener is installed are queued
    // by the browser, then delivered when the listener registers. The
    // `set_onmessage` setter has had spec ambiguity around exactly when
    // queued messages are flushed; `addEventListener` is the safer
    // contract.
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        // `info!` (not debug) so the diagnostic is visible at default
        // tracing levels. Drop to debug once Phase B is stable.
        tracing::info!("rye_app::wasm::worker: message handler firing");
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

    // `init` is the only kind that touches A. Everything else lives in
    // the non-generic messages module.
    let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|v| v.as_string());
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

/// Phase B1: build `RenderDevice` from the worker-owned OffscreenCanvas,
/// run `App::setup`, and start the RAF loop driving the App's per-frame
/// lifecycle.
///
/// Uses `RenderDevice::from_surface` so the wgpu setup matches the
/// windowed-mode path (sRGB composite, MSAA negotiation, GPU timer
/// detection, etc.). The worker doesn't get MSAA in this phase
/// because the OffscreenCanvas surface format negotiation matches
/// the browser-WebGPU non-sRGB swap case (composite + sample_count=1).
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
    // pixel dimensions in this phase; Phase B2 will plumb resize events.
    let size = winit::dpi::PhysicalSize::new(width, height);
    let rd = RenderDevice::from_surface(
        instance,
        surface,
        size,
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
    // accept that worker mode shows egui at 1pt:1px scale. Phase B+
    // can plumb DPR explicitly through the init message.
    let pixels_per_point = 1.0; // see comment above; minor cosmetic issue.

    let runner = WorkerRunner::<A>::setup(
        rd,
        canvas_for_runner,
        width,
        height,
        pixels_per_point,
    )
        .await
        .context("WorkerRunner::setup")?;
    tracing::info!("rye_app::wasm::worker: WorkerRunner setup complete; starting RAF loop");

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
            let _ = scope_for_closure
                .request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }) as Box<dyn FnMut(f64)>));

    // Kick off the first frame (dropping the borrow before we leak the Rc).
    {
        let first_cb = raf_cb.borrow();
        let first = first_cb
            .as_ref()
            .expect("RAF closure populated above");
        scope
            .request_animation_frame(first.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("request_animation_frame: {e:?}"))?;
    }

    // Both `raf_cb` and `runner` need to outlive this function so the
    // closure stays valid for subsequent RAF callbacks.
    Box::leak(Box::new(raf_cb));
    Box::leak(Box::new(runner));

    Ok(())
}

/// Per-worker lifecycle state: owns the RenderDevice + the user's App
/// + the wall-clock / tick bookkeeping the existing main-thread Runner
/// owns. Sits inside the RAF closure via `Rc<RefCell>`.
///
/// Phase B1 scope: drives `App::update` + `App::record` only. No egui,
/// no input (Phase B3 + B4 add those).
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
    #[allow(dead_code)] // held alive for App's runtime; lookups via ShaderDb come in Phase B3+
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
        self.rd
            .resize(winit::dpi::PhysicalSize::new(width, height));
        self.width_px = width;
        self.height_px = height;
        self.ui.resize(width, height, self.pixels_per_point);
        tracing::info!("rye_app::wasm::worker: resized to {width}x{height}");
    }

    /// Apply one `InputMessage`. Resize updates the surface; the other
    /// variants route into `InputState` for the App's per-frame
    /// `FrameInput`. Phase B4 will also fan these out to egui via
    /// `RawInput` so HUD widgets can respond.
    ///
    /// `App::on_event` (the path that handles winit `WindowEvent`s, e.g.
    /// tesseract_demo's KeyF / KeyT / Space / KeyR hotkeys) is deliberately
    /// NOT plumbed here — constructing winit's `KeyEvent` requires private
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
                self.input
                    .mouse_wheel(MouseScrollDelta::LineDelta(dx, dy));
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
                // now; Phase B+ could pause continuous animation here.
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

        // Drain queued input messages. Phase B2 only handles `Resize`;
        // Phase B3 will add the mouse/keyboard variants here (routed
        // into `InputState`) and Phase B4 will fan them out to egui too.
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
        self.tick_index = self.tick_index.wrapping_add(1);

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
                fps: 0.0, // Phase B1: skip FPS bookkeeping; not used by tesseract_demo.
                n_ticks: 0,
                tick: self.tick_index,
                dt,
                ui_has_focus,
                _non_exhaustive: PhantomData,
            };
            self.app.update(&mut fctx);

            // App::ui — egui frame begin, App builds widgets, paint happens
            // after the scene render below. Same lifecycle as the windowed
            // runner: begin_frame -> ctx clone -> App::ui -> paint at end.
            let egui_ctx = self.ui.begin_frame().clone();
            let _scope = rye_time::frame_trace::scope("app-ui");
            self.app.ui(&egui_ctx, &mut fctx);
        }

        // begin_frame -> record -> composite -> submit -> present.
        let (frame, swap_view) = self
            .rd
            .begin_frame()
            .context("RenderDevice::begin_frame")?;
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
/// — std::sync::Once is per-process). Both `run` (worker) and
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
