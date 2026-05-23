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
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    DedicatedWorkerGlobalScope, HtmlButtonElement, HtmlCanvasElement, MessageEvent,
    OffscreenCanvas, Worker, WorkerOptions, WorkerType,
};

use crate::{App, FrameCtx, RenderCtx, SetupCtx};
use rye_asset::AssetWatcher;
use rye_egui::egui;
use rye_input::InputState;
use rye_render::device::RenderDevice;
use rye_shader::ShaderDb;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

// ---------------------------------------------------------------------------
// InputMessage: typed protocol between main thread and worker
// ---------------------------------------------------------------------------

/// Per-frame inputs the main thread forwards to the worker. Each variant
/// corresponds to a `kind` string in the postMessage JS object payload.
/// The worker-side `handle_message` parses these and pushes them onto
/// a thread_local queue; `WorkerRunner::frame` drains the queue at the
/// top of each frame and applies the events.
///
/// Why an enum (vs raw JS objects passed through to the App): keeps the
/// worker-side App code allocation-free for high-frequency events
/// (mouse-move at 60Hz; if we serialized/deserialized JSON per event,
/// every mouse motion would allocate strings).
#[derive(Debug)]
pub enum InputMessage {
    /// New canvas pixel dimensions. Sent by main thread on window resize
    /// (DPR-multiplied to physical pixels). Triggers a wgpu surface
    /// reconfigure on the next frame.
    Resize { width: u32, height: u32 },

    /// Pointer moved to (x, y) in canvas-local CSS pixels (the worker
    /// applies DPR if needed). `buttons` is the standard `MouseEvent.buttons`
    /// bitmask (1=primary, 2=secondary, 4=middle).
    MouseMove { x: f32, y: f32, buttons: u8 },

    /// Pointer button transitioned. `button` is the standard
    /// `MouseEvent.button` (0=primary, 1=middle, 2=secondary).
    MouseButton {
        x: f32,
        y: f32,
        button: u8,
        pressed: bool,
    },

    /// Wheel delta in lines (after the browser's pixel/line normalization).
    /// `dx`/`dy` follow DOM convention: positive = right/down.
    MouseWheel { dx: f32, dy: f32 },

    /// Keyboard key transitioned. `code` is the physical-key code (e.g.
    /// "KeyT", "Space"); `key` is the logical key (e.g. "t", " ").
    /// Phase B3 uses `code` for gameplay hotkeys + `key` for text input.
    Key {
        code: String,
        key: String,
        pressed: bool,
        repeat: bool,
        ctrl: bool,
        shift: bool,
        alt: bool,
        meta: bool,
    },

    /// Window focus state changed.
    Focus(bool),

    /// Page visibility (tab-in-foreground) state changed.
    Visibility(bool),
}

thread_local! {
    /// Per-worker queue of inbound input messages, drained at the top of
    /// every frame by `WorkerRunner::frame`. Using a thread_local because
    /// the message handler closure doesn't have direct access to the
    /// runner (the runner is constructed asynchronously, after the first
    /// init message arrives).
    static MESSAGE_QUEUE: RefCell<VecDeque<InputMessage>> = RefCell::new(VecDeque::new());
}

fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().push_back(msg));
}

/// Drain all queued messages. Returns them in arrival order so the
/// runner can apply them sequentially. Called once per frame.
fn drain_messages() -> Vec<InputMessage> {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().drain(..).collect())
}

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

/// Dispatch a single inbound `postMessage`. Phase B1 only handles the
/// `init` kind; Phase B3 adds input-event variants.
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

    match kind.as_deref() {
        Some("init") => {
            let canvas = js_sys::Reflect::get(&data, &JsValue::from_str("canvas"))
                .map_err(|e| anyhow!("init missing 'canvas' field: {e:?}"))?
                .dyn_into::<OffscreenCanvas>()
                .map_err(|e| anyhow!("init 'canvas' is not an OffscreenCanvas: {e:?}"))?;
            let width = read_u32_field(&data, "width").unwrap_or(800);
            let height = read_u32_field(&data, "height").unwrap_or(600);

            tracing::info!(
                "rye_app::wasm::worker: received init ({width}x{height}); spawning wgpu setup"
            );
            // wgpu setup is async; spawn it on the worker's task queue.
            // The closures inside take ownership of the canvas + RAF
            // scheduling state, so they survive after this handler returns.
            let scope_for_render = scope.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = init_renderer::<A>(scope_for_render, canvas, width, height).await {
                    tracing::error!("rye_app::wasm::worker: init_renderer failed: {e:#}");
                }
            });
            Ok(())
        }
        Some("resize") => {
            let width = read_u32_field(&data, "width").unwrap_or(0);
            let height = read_u32_field(&data, "height").unwrap_or(0);
            enqueue(InputMessage::Resize { width, height });
            Ok(())
        }
        Some("mouse_move") => {
            let x = read_f32_field(&data, "x").unwrap_or(0.0);
            let y = read_f32_field(&data, "y").unwrap_or(0.0);
            let buttons = read_u32_field(&data, "buttons").unwrap_or(0) as u8;
            enqueue(InputMessage::MouseMove { x, y, buttons });
            Ok(())
        }
        Some("mouse_button") => {
            let x = read_f32_field(&data, "x").unwrap_or(0.0);
            let y = read_f32_field(&data, "y").unwrap_or(0.0);
            let button = read_u32_field(&data, "button").unwrap_or(0) as u8;
            let pressed = read_bool_field(&data, "pressed").unwrap_or(false);
            enqueue(InputMessage::MouseButton {
                x,
                y,
                button,
                pressed,
            });
            Ok(())
        }
        Some("mouse_wheel") => {
            let dx = read_f32_field(&data, "dx").unwrap_or(0.0);
            let dy = read_f32_field(&data, "dy").unwrap_or(0.0);
            enqueue(InputMessage::MouseWheel { dx, dy });
            Ok(())
        }
        Some("key") => {
            let code = read_string_field(&data, "code").unwrap_or_default();
            let key = read_string_field(&data, "key").unwrap_or_default();
            let pressed = read_bool_field(&data, "pressed").unwrap_or(false);
            let repeat = read_bool_field(&data, "repeat").unwrap_or(false);
            let ctrl = read_bool_field(&data, "ctrl").unwrap_or(false);
            let shift = read_bool_field(&data, "shift").unwrap_or(false);
            let alt = read_bool_field(&data, "alt").unwrap_or(false);
            let meta = read_bool_field(&data, "meta").unwrap_or(false);
            enqueue(InputMessage::Key {
                code,
                key,
                pressed,
                repeat,
                ctrl,
                shift,
                alt,
                meta,
            });
            Ok(())
        }
        Some("focus") => {
            let focused = read_bool_field(&data, "focused").unwrap_or(false);
            enqueue(InputMessage::Focus(focused));
            Ok(())
        }
        Some("visibility") => {
            let visible = read_bool_field(&data, "visible").unwrap_or(false);
            enqueue(InputMessage::Visibility(visible));
            Ok(())
        }
        Some(other) => {
            tracing::warn!("rye_app::wasm::worker: unknown message kind '{other}'");
            Ok(())
        }
        None => Err(anyhow!("postMessage missing 'kind' field")),
    }
}

fn read_u32_field(obj: &JsValue, key: &str) -> Option<u32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|f| f as u32)
}

fn read_f32_field(obj: &JsValue, key: &str) -> Option<f32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
}

fn read_bool_field(obj: &JsValue, key: &str) -> Option<bool> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_bool())
}

fn read_string_field(obj: &JsValue, key: &str) -> Option<String> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
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
/// Worker-side egui integration: parallel to `rye_egui::UiIntegration`
/// but without `egui_winit` (which depends on `web_sys::Window` and
/// panics in `WorkerGlobalScope`). Translates `InputMessage` directly
/// into `egui::RawInput::events` instead.
///
/// Owns the egui-wgpu `Renderer`, the `egui::Context`, and a per-frame
/// `RawInput` accumulator. Mirror the `UiIntegration::begin_frame` +
/// `paint` lifecycle so the App's existing `ui` method works unchanged.
struct WorkerUi {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    /// Accumulates events between frames. Drained into `begin_pass` at
    /// the start of each frame; populated by `record_input` whenever
    /// an `InputMessage` arrives from main.
    raw_events: Vec<egui::Event>,
    /// Current modifier state. Updated on each key event so subsequent
    /// pointer/key events carry the right `Modifiers`.
    modifiers: egui::Modifiers,
    /// Canvas pixel dimensions + DPR. Egui works in "points" (CSS-pixel
    /// equivalents), wgpu in pixels — `pixels_per_point` is the conversion.
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
    /// `wants_pointer_input || wants_keyboard_input` from the last
    /// completed frame. Fed back to the App via `FrameCtx::ui_has_focus`
    /// so gameplay code (camera, hotkeys) can avoid double-handling
    /// events egui consumed.
    wants_input: bool,
}

impl WorkerUi {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
    ) -> Self {
        let ctx = egui::Context::default();
        let renderer = egui_wgpu::Renderer::new(
            device,
            target_format,
            egui_wgpu::RendererOptions {
                msaa_samples: sample_count,
                ..Default::default()
            },
        );
        Self {
            ctx,
            renderer,
            raw_events: Vec::new(),
            modifiers: egui::Modifiers::default(),
            width_px,
            height_px,
            pixels_per_point,
            wants_input: false,
        }
    }

    /// Translate one InputMessage into zero or more egui events.
    /// Updates `modifiers` as a side effect on key events.
    fn record_input(&mut self, msg: &InputMessage) {
        match msg {
            InputMessage::MouseMove { x, y, .. } => {
                let pos = self.point(*x, *y);
                self.raw_events.push(egui::Event::PointerMoved(pos));
            }
            InputMessage::MouseButton {
                x,
                y,
                button,
                pressed,
            } => {
                let pos = self.point(*x, *y);
                if let Some(b) = egui_button_from_dom(*button) {
                    self.raw_events.push(egui::Event::PointerButton {
                        pos,
                        button: b,
                        pressed: *pressed,
                        modifiers: self.modifiers,
                    });
                }
            }
            InputMessage::MouseWheel { dx, dy } => {
                // egui's MouseWheel unit is points (its "lines"-equivalent).
                // The DOM->lines conversion already happened on main thread.
                self.raw_events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: egui::vec2(-*dx, -*dy), // egui convention: up = +y
                    modifiers: self.modifiers,
                });
            }
            InputMessage::Key {
                code,
                key,
                pressed,
                repeat,
                ctrl,
                shift,
                alt,
                meta,
            } => {
                // Update modifier state. Egui's Modifiers carries each
                // boolean independently; we keep them current here.
                self.modifiers = egui::Modifiers {
                    alt: *alt,
                    ctrl: *ctrl,
                    shift: *shift,
                    mac_cmd: *meta,
                    command: *ctrl || *meta,
                };
                // Emit a Key event when the physical code maps to an
                // egui::Key. Unknown codes are silently dropped (the
                // App's hotkey routing covers them via InputState).
                if let Some(egui_key) = egui_key_from_code(code) {
                    self.raw_events.push(egui::Event::Key {
                        key: egui_key,
                        physical_key: Some(egui_key),
                        pressed: *pressed,
                        repeat: *repeat,
                        modifiers: self.modifiers,
                    });
                }
                // Emit a Text event for printable keys on press. Filters
                // modifier-only chords (Ctrl+C etc.) so they don't
                // accidentally insert text. `key.len() == 1` is a rough
                // "is this a single printable character" check that
                // works for ASCII + common Latin codepoints; multi-codepoint
                // logical keys ("ArrowUp", "Shift") are filtered out.
                if *pressed
                    && !*ctrl
                    && !*alt
                    && !*meta
                    && key.chars().count() == 1
                    && !key.starts_with(char::is_control)
                {
                    self.raw_events.push(egui::Event::Text(key.clone()));
                }
            }
            InputMessage::Focus(focused) => {
                self.raw_events.push(egui::Event::WindowFocused(*focused));
            }
            // Resize handled by the runner directly (not an egui event).
            InputMessage::Resize { .. } | InputMessage::Visibility(_) => {}
        }
    }

    /// Convert canvas-relative CSS pixels to egui points. Egui works in
    /// logical (DPI-independent) coordinates, so we divide by
    /// `pixels_per_point`. The InputMessage carries CSS pixels already
    /// (DOM `MouseEvent.offsetX/Y` are in CSS pixels), but we treat them
    /// as such even though the canvas backing-store is in physical
    /// pixels — egui's `screen_rect` is given in points to match.
    fn point(&self, x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(x, y)
    }

    /// Begin a frame: take the accumulated raw input, set screen rect +
    /// pixels-per-point, return the Context for the App's `ui` method.
    fn begin_frame(&mut self) -> &egui::Context {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(
                    self.width_px as f32 / self.pixels_per_point,
                    self.height_px as f32 / self.pixels_per_point,
                ),
            )),
            events: std::mem::take(&mut self.raw_events),
            modifiers: self.modifiers,
            viewport_id: egui::ViewportId::ROOT,
            time: None,
            ..egui::RawInput::default()
        };
        self.ctx.begin_pass(raw_input);
        &self.ctx
    }

    /// Finish the frame + paint into `view` via the supplied encoder.
    /// Mirrors `UiIntegration::paint` but without the winit
    /// `handle_platform_output` step (no cursor / clipboard routing yet).
    fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        resolve_target: Option<&wgpu::TextureView>,
    ) {
        let full_output = self.ctx.end_pass();
        self.wants_input =
            self.ctx.wants_pointer_input() || self.ctx.wants_keyboard_input();

        let primitives = self
            .ctx
            .tessellate(full_output.shapes, self.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.width_px, self.height_px],
            pixels_per_point: self.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &primitives, &screen);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rye_app::wasm::worker::egui-paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // egui-wgpu requires a `'static` lifetime on the pass; this
            // is the standard wasm-bindgen / egui-wgpu idiom.
            self.renderer
                .render(&mut pass.forget_lifetime(), &primitives, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        self.width_px = width;
        self.height_px = height;
        self.pixels_per_point = dpr;
    }
}

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
                let button = mouse_button_from_dom(button);
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
                if let Some(code) = winit_keycode_from_str(code) {
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
        for msg in drain_messages() {
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
fn install_logging_idempotent() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
    });
}

// ---------------------------------------------------------------------------
// Main-thread side
// ---------------------------------------------------------------------------

/// URL of the wasm bundle's JS entry, set by the demo's inline script via
/// `window.__rye_wasm_url = import.meta.url`. The worker is constructed
/// pointing at this URL so it runs the same wasm binary the main thread
/// loaded; the binary detects its context via [`crate::wasm::is_worker_context`].
///
/// Captured here as a function instead of a constant because `import.meta.url`
/// is per-document and we read it at launch time.
fn read_wasm_bundle_url() -> Result<String> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("__rye_wasm_url"))
        .map_err(|e| anyhow!("read __rye_wasm_url: {e:?}"))?;
    val.as_string()
        .ok_or_else(|| anyhow!("__rye_wasm_url is not a string; demo's index.html must set it"))
}

/// Main-thread entry point for worker mode. Wires the launch button so a
/// click transfers the page's canvas to a freshly-spawned worker, which
/// then runs the Phase A clear-loop.
///
/// Phase A signature: `host_id` (the container element with the launch
/// button), `button_id` (the click target), `canvas_id` (the canvas to
/// transfer to the worker). Phase B will add a `WorkerConfig` parameter
/// once the conductor exists.
///
/// Returns immediately after wiring the listener; the click might happen
/// seconds or minutes later.
pub fn launch_on_click(host_id: &str, button_id: &str, canvas_id: &str) -> Result<()> {
    // Main thread also wants tracing routed to DevTools so any setup-time
    // errors are visible. Worker side installs its own (separate JS heap).
    install_logging_idempotent();

    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let button = document
        .get_element_by_id(button_id)
        .ok_or_else(|| anyhow!("no element with id '{button_id}'"))?
        .dyn_into::<HtmlButtonElement>()
        .map_err(|_| anyhow!("element '{button_id}' is not a button"))?;
    let canvas_id_owned = canvas_id.to_string();
    let host_id_owned = host_id.to_string();
    let button_for_handler = button.clone();

    let on_click = Closure::once(Box::new(move || {
        // Remove the launch button so a frantic double-click can't fire
        // twice (canvas can only be transferred once).
        button_for_handler.remove();
        if let Err(e) = spawn_worker(&canvas_id_owned, &host_id_owned) {
            tracing::error!("rye_app::wasm::worker: spawn_worker failed: {e:#}");
        }
    }) as Box<dyn FnOnce()>);

    button
        .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("add_event_listener: {e:?}"))?;
    on_click.forget();

    Ok(())
}

/// Actually spawn the worker + transfer the canvas + post the init message.
/// Called from the launch button's click handler.
fn spawn_worker(canvas_id: &str, _host_id: &str) -> Result<()> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| anyhow!("no element with id '{canvas_id}'"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| anyhow!("element '{canvas_id}' is not a canvas"))?;

    // Size the canvas's pixel backing-store to match its DISPLAYED size
    // × device pixel ratio. The HTML's `width`/`height` attributes are a
    // pre-launch fallback; we compute the actual rendering size here
    // because CSS may have stretched the canvas to fill its container
    // (and Trunk serves a sized container with `width: 100%`). Without
    // this step the backing store stays at the HTML attribute size and
    // CSS stretches the rendered image, producing the squashed aspect
    // the original main-thread path didn't have (winit recomputed the
    // canvas size dynamically from the host element).
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let dpr = window.device_pixel_ratio() as f32;
    let css_w = canvas.client_width().max(1) as f32;
    let css_h = canvas.client_height().max(1) as f32;
    let width = (css_w * dpr).round() as u32;
    let height = (css_h * dpr).round() as u32;
    if width == 0 || height == 0 {
        return Err(anyhow!(
            "canvas '{canvas_id}' has zero displayed dimensions ({css_w}x{css_h}); \
             container must have layout dimensions before launch"
        ));
    }
    canvas.set_width(width);
    canvas.set_height(height);
    tracing::info!(
        "rye_app::wasm::worker: canvas sized to {width}x{height} (CSS {css_w}x{css_h} × DPR {dpr})"
    );

    let offscreen = canvas
        .transfer_control_to_offscreen()
        .map_err(|e| anyhow!("transfer_control_to_offscreen: {e:?}"))?;

    let js_url = read_wasm_bundle_url()?;
    // Derive the `_bg.wasm` URL from the JS URL. Trunk's convention:
    // `<name>-<hash>.js` and `<name>-<hash>_bg.wasm` live side by side.
    // We need to pass the wasm URL explicitly to `init()` because the
    // worker imports the JS via a Blob URL, which has no document base
    // for relative path resolution; without the explicit argument, the
    // generated init function falls back to fetching a relative path
    // that 404s back to the page HTML (MIME `text/html`, not `application/wasm`).
    let wasm_url = js_url.strip_suffix(".js").unwrap_or(&js_url).to_string() + "_bg.wasm";
    tracing::info!(
        "rye_app::wasm::worker: spawning worker (js={js_url}, wasm={wasm_url})"
    );

    // wasm-bindgen's generated `--target web` ESM exports `init` as the
    // default but doesn't auto-run at import time. The page's main script
    // calls `init({ module_or_path: ... })` explicitly; the worker needs
    // the same. Instead of shipping a separate `worker_bootstrap.js` per
    // demo, build the bootstrap inline as a Blob URL pointing at the same
    // wasm bundle.
    //
    // The bootstrap is a two-line ESM module that imports the trunk-
    // generated init function and awaits it WITH the explicit wasm URL.
    // After init resolves, the wasm's `main` runs in worker context,
    // detects worker-ness via `is_worker_context`, and calls
    // `worker::run`. From that point the wasm-side onmessage handler
    // waits for the init postMessage below.
    let bootstrap_js = format!(
        "import init from '{js_url}';\nawait init({{ module_or_path: '{wasm_url}' }});\n"
    );
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&JsValue::from_str(&bootstrap_js));
    let blob_options = web_sys::BlobPropertyBag::new();
    blob_options.set_type("application/javascript");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_options)
        .map_err(|e| anyhow!("Blob::new: {e:?}"))?;
    let blob_url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| anyhow!("createObjectURL: {e:?}"))?;

    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let worker = Worker::new_with_options(&blob_url, &opts)
        .map_err(|e| anyhow!("Worker::new: {e:?}"))?;

    // Wait for the worker to signal it's ready (handler installed) before
    // posting the Init message. The worker's `run()` posts `{kind: "ready"}`
    // back to main once its `message` listener is wired. Without this
    // handshake, the Init might arrive before the worker's listener exists
    // — Firefox empirically drops such messages.
    //
    // The on-ready closure captures everything needed for the Init post:
    // the offscreen canvas, dimensions, and a handle to the worker. Once
    // it fires successfully, it removes itself (one-shot) since subsequent
    // worker messages are application protocol, not lifecycle.
    let worker_for_ready = worker.clone();
    let offscreen_for_ready = offscreen.clone();
    let on_ready = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() != Some("ready") {
            // Ignore non-ready messages (none expected before init, but
            // be defensive). Once we've sent init, future incoming
            // messages will fall through to whatever Phase B installs.
            return;
        }
        tracing::info!("rye_app::wasm::worker: worker signalled ready, posting init");

        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("kind"),
            &JsValue::from_str("init"),
        );
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("canvas"),
            &offscreen_for_ready,
        );
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("width"),
            &JsValue::from_f64(width as f64),
        );
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("height"),
            &JsValue::from_f64(height as f64),
        );

        let transfer = js_sys::Array::new();
        transfer.push(&offscreen_for_ready);

        if let Err(e) = worker_for_ready.post_message_with_transfer(&msg, &transfer) {
            tracing::error!("rye_app::wasm::worker: postMessage init failed: {e:?}");
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", on_ready.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message'): {e:?}"))?;
    on_ready.forget();

    // Forward DOM events (resize / mouse / keyboard / focus / visibility)
    // from the page to the worker. Installed BEFORE the worker is fully
    // ready so events during the setup window get queued by the browser
    // (or via our handle_message queue) and applied on first frame.
    install_dom_input_forwarders(&worker, &canvas)
        .context("install_dom_input_forwarders")?;

    // Keep the Worker alive. Dropping it would terminate. Box::leak is the
    // wasm-bindgen idiom for "this lives forever in this page."
    Box::leak(Box::new(worker));

    Ok(())
}

/// Install all the DOM event listeners that forward main-thread events to
/// the worker via postMessage. Each listener constructs a typed JS object
/// (`{kind: "...", ...}`) the worker-side `handle_message` parses into an
/// `InputMessage` variant.
///
/// Listeners attached:
/// - `window`: resize, focus, blur, visibilitychange
/// - `canvas` (the placeholder post-transfer): mousemove, mousedown,
///   mouseup, wheel
/// - `document`: keydown, keyup (keyboard listeners on document so we
///   capture keys regardless of which element has focus, matching
///   game-style input expectations)
///
/// All listeners use `Closure::wrap` + `forget()` to leak themselves into
/// the JS heap, since they live for the page's lifetime. Demos that close
/// the worker would need explicit teardown — not a Phase B concern.
fn install_dom_input_forwarders(worker: &Worker, canvas: &HtmlCanvasElement) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;

    // Resize: compute the canvas's pixel dimensions (CSS size × DPR) and
    // post to worker. Resize fires on the WINDOW; we re-query the canvas
    // size each fire because CSS recalc may not have settled by event time.
    {
        let worker = worker.clone();
        let canvas = canvas.clone();
        let window_for_dpr = window.clone();
        let cb = Closure::wrap(Box::new(move || {
            let dpr = window_for_dpr.device_pixel_ratio() as f32;
            let w = (canvas.client_width() as f32 * dpr).max(1.0) as u32;
            let h = (canvas.client_height() as f32 * dpr).max(1.0) as u32;
            let msg = build_msg("resize");
            set_msg_u32(&msg, "width", w);
            set_msg_u32(&msg, "height", h);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("window resize listener: {e:?}"))?;
        cb.forget();
    }

    // Mouse move on the canvas. rAF-COALESCED: a DOM mouse-move can fire
    // hundreds of times per second; forwarding each one is a JS-object
    // allocation per event + a postMessage. At sustained drag speeds
    // that's enough to overwhelm the JS heap and crash the browser tab
    // (we hit this empirically during Phase B3 testing).
    //
    // Coalesce: the listener writes the latest mouse position into a
    // shared RefCell. A separate rAF loop checks for a pending mouse-
    // move once per browser-frame and posts ONE message with the
    // latest position. Result: at most 60 mouse-move postMessages per
    // second regardless of input device frequency.
    //
    // `MouseEvent.offsetX/Y` are canvas-relative; `buttons` is the
    // held-button bitmask. DPR scaling happens worker-side so the same
    // coords work whether the user has zoomed or not.
    {
        // Shared latest-position state. None when no pending move.
        let pending: Rc<RefCell<Option<(f32, f32, u32)>>> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            *pending_for_listener.borrow_mut() = Some((
                ev.offset_x() as f32,
                ev.offset_y() as f32,
                ev.buttons() as u32,
            ));
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("mousemove listener: {e:?}"))?;
        cb.forget();

        // rAF loop that drains `pending` once per frame.
        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let raf_cb: Rc<RefCell<Option<Closure<dyn FnMut()>>>> =
            Rc::new(RefCell::new(None));
        let raf_cb_for_closure = raf_cb.clone();
        *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            if let Some((x, y, buttons)) = pending_for_raf.borrow_mut().take() {
                let msg = build_msg("mouse_move");
                set_msg_f32(&msg, "x", x);
                set_msg_f32(&msg, "y", y);
                set_msg_u32(&msg, "buttons", buttons);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf
                    .request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            let first = first_cb.as_ref().expect("RAF cb populated above");
            window
                .request_animation_frame(first.as_ref().unchecked_ref())
                .map_err(|e| anyhow!("mousemove rAF init: {e:?}"))?;
        }
        Box::leak(Box::new(raf_cb));
    }

    // Mouse down / up: same shape, different `pressed` flag.
    for (event_name, pressed) in [("mousedown", true), ("mouseup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            let msg = build_msg("mouse_button");
            set_msg_f32(&msg, "x", ev.offset_x() as f32);
            set_msg_f32(&msg, "y", ev.offset_y() as f32);
            set_msg_u32(&msg, "button", ev.button() as u32);
            set_msg_bool(&msg, "pressed", pressed);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    // Wheel: WheelEvent's delta is per-mode (pixels / lines / pages).
    // Normalize to lines (the rye-input convention) by checking deltaMode.
    {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::WheelEvent| {
            // delta_mode: 0=pixels, 1=lines, 2=pages. Most browsers use
            // pixels; convert to lines using a 100px = 1 line heuristic
            // (matches winit's web backend convention).
            let (dx, dy) = match ev.delta_mode() {
                1 => (ev.delta_x() as f32, ev.delta_y() as f32),
                _ => (ev.delta_x() as f32 / 100.0, ev.delta_y() as f32 / 100.0),
            };
            ev.prevent_default(); // page-scroll suppression while interacting
            let msg = build_msg("mouse_wheel");
            set_msg_f32(&msg, "dx", dx);
            set_msg_f32(&msg, "dy", dy);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::WheelEvent)>);
        canvas
            .add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("wheel listener: {e:?}"))?;
        cb.forget();
    }

    // Keyboard: listen on document so keys reach us regardless of which
    // element has focus (game convention). The placeholder canvas can't
    // take keyboard focus by default; alternatives would require setting
    // tabindex + focus(), which adds page-author friction we'd rather
    // avoid for Phase B.
    for (event_name, pressed) in [("keydown", true), ("keyup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::KeyboardEvent| {
            let msg = build_msg("key");
            set_msg_string(&msg, "code", &ev.code());
            set_msg_string(&msg, "key", &ev.key());
            set_msg_bool(&msg, "pressed", pressed);
            set_msg_bool(&msg, "repeat", ev.repeat());
            set_msg_bool(&msg, "ctrl", ev.ctrl_key());
            set_msg_bool(&msg, "shift", ev.shift_key());
            set_msg_bool(&msg, "alt", ev.alt_key());
            set_msg_bool(&msg, "meta", ev.meta_key());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::KeyboardEvent)>);
        document
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    // Focus / blur on window for the camera-style "release held buttons
    // on focus loss" behaviour rye-input expects.
    for (event_name, focused) in [("focus", true), ("blur", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move || {
            let msg = build_msg("focus");
            set_msg_bool(&msg, "focused", focused);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    // Page-visibility on document: tabs being backgrounded should pause
    // continuous animation (camera spin, etc.) so we're not wasting CPU.
    {
        let worker = worker.clone();
        let document_for_query = document.clone();
        let cb = Closure::wrap(Box::new(move || {
            let visible = document_for_query.visibility_state()
                != web_sys::VisibilityState::Hidden;
            let msg = build_msg("visibility");
            set_msg_bool(&msg, "visible", visible);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback("visibilitychange", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("visibilitychange listener: {e:?}"))?;
        cb.forget();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Small helpers for constructing typed postMessage payloads on main thread.
// ---------------------------------------------------------------------------

fn build_msg(kind: &str) -> js_sys::Object {
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("kind"),
        &JsValue::from_str(kind),
    );
    obj
}

fn set_msg_u32(obj: &js_sys::Object, key: &str, v: u32) {
    let _ = js_sys::Reflect::set(
        obj,
        &JsValue::from_str(key),
        &JsValue::from_f64(v as f64),
    );
}

fn set_msg_f32(obj: &js_sys::Object, key: &str, v: f32) {
    let _ = js_sys::Reflect::set(
        obj,
        &JsValue::from_str(key),
        &JsValue::from_f64(v as f64),
    );
}

fn set_msg_bool(obj: &js_sys::Object, key: &str, v: bool) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_bool(v));
}

fn set_msg_string(obj: &js_sys::Object, key: &str, v: &str) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(v));
}

// ---------------------------------------------------------------------------
// Winit-type translation helpers
// ---------------------------------------------------------------------------

/// Map a DOM `MouseEvent.button` index to an `egui::PointerButton`.
/// `None` for buttons egui doesn't recognize (4=Back, 5=Forward —
/// egui's Extra1/Extra2 cover those but their semantics differ).
fn egui_button_from_dom(button: u8) -> Option<egui::PointerButton> {
    match button {
        0 => Some(egui::PointerButton::Primary),
        1 => Some(egui::PointerButton::Middle),
        2 => Some(egui::PointerButton::Secondary),
        3 => Some(egui::PointerButton::Extra1),
        4 => Some(egui::PointerButton::Extra2),
        _ => None,
    }
}

/// Map a DOM `KeyboardEvent.code` string to an `egui::Key`. Partial
/// mapping mirroring `winit_keycode_from_str` above; egui has fewer
/// key variants so unmapped codes are dropped silently. Coverage
/// focuses on what tesseract_demo + future demos plausibly need
/// inside egui widgets: arrows, text-editing keys, function keys,
/// alpha + digits.
fn egui_key_from_code(code: &str) -> Option<egui::Key> {
    let by_letter = match code {
        "KeyA" => Some(egui::Key::A), "KeyB" => Some(egui::Key::B),
        "KeyC" => Some(egui::Key::C), "KeyD" => Some(egui::Key::D),
        "KeyE" => Some(egui::Key::E), "KeyF" => Some(egui::Key::F),
        "KeyG" => Some(egui::Key::G), "KeyH" => Some(egui::Key::H),
        "KeyI" => Some(egui::Key::I), "KeyJ" => Some(egui::Key::J),
        "KeyK" => Some(egui::Key::K), "KeyL" => Some(egui::Key::L),
        "KeyM" => Some(egui::Key::M), "KeyN" => Some(egui::Key::N),
        "KeyO" => Some(egui::Key::O), "KeyP" => Some(egui::Key::P),
        "KeyQ" => Some(egui::Key::Q), "KeyR" => Some(egui::Key::R),
        "KeyS" => Some(egui::Key::S), "KeyT" => Some(egui::Key::T),
        "KeyU" => Some(egui::Key::U), "KeyV" => Some(egui::Key::V),
        "KeyW" => Some(egui::Key::W), "KeyX" => Some(egui::Key::X),
        "KeyY" => Some(egui::Key::Y), "KeyZ" => Some(egui::Key::Z),
        _ => None,
    };
    if by_letter.is_some() {
        return by_letter;
    }
    let by_digit = match code {
        "Digit0" => Some(egui::Key::Num0), "Digit1" => Some(egui::Key::Num1),
        "Digit2" => Some(egui::Key::Num2), "Digit3" => Some(egui::Key::Num3),
        "Digit4" => Some(egui::Key::Num4), "Digit5" => Some(egui::Key::Num5),
        "Digit6" => Some(egui::Key::Num6), "Digit7" => Some(egui::Key::Num7),
        "Digit8" => Some(egui::Key::Num8), "Digit9" => Some(egui::Key::Num9),
        _ => None,
    };
    if by_digit.is_some() {
        return by_digit;
    }
    let by_fn = match code {
        "F1" => Some(egui::Key::F1),  "F2" => Some(egui::Key::F2),
        "F3" => Some(egui::Key::F3),  "F4" => Some(egui::Key::F4),
        "F5" => Some(egui::Key::F5),  "F6" => Some(egui::Key::F6),
        "F7" => Some(egui::Key::F7),  "F8" => Some(egui::Key::F8),
        "F9" => Some(egui::Key::F9),  "F10" => Some(egui::Key::F10),
        "F11" => Some(egui::Key::F11), "F12" => Some(egui::Key::F12),
        _ => None,
    };
    if by_fn.is_some() {
        return by_fn;
    }
    match code {
        "Space" => Some(egui::Key::Space),
        "Enter" => Some(egui::Key::Enter),
        "Escape" => Some(egui::Key::Escape),
        "Tab" => Some(egui::Key::Tab),
        "Backspace" => Some(egui::Key::Backspace),
        "Delete" => Some(egui::Key::Delete),
        "ArrowUp" => Some(egui::Key::ArrowUp),
        "ArrowDown" => Some(egui::Key::ArrowDown),
        "ArrowLeft" => Some(egui::Key::ArrowLeft),
        "ArrowRight" => Some(egui::Key::ArrowRight),
        "Home" => Some(egui::Key::Home),
        "End" => Some(egui::Key::End),
        "PageUp" => Some(egui::Key::PageUp),
        "PageDown" => Some(egui::Key::PageDown),
        "Backquote" => Some(egui::Key::Backtick),
        "Minus" => Some(egui::Key::Minus),
        "Equal" => Some(egui::Key::Equals),
        _ => None,
    }
}

/// Map a DOM `MouseEvent.button` index to a `winit::event::MouseButton`.
/// The DOM convention is 0=primary, 1=middle, 2=secondary; winit uses
/// named variants.
fn mouse_button_from_dom(button: u8) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        3 => MouseButton::Back,
        4 => MouseButton::Forward,
        other => MouseButton::Other(other as u16),
    }
}

/// Map a DOM `KeyboardEvent.code` string to a `winit::keyboard::KeyCode`.
/// Partial mapping focused on the keys rye demos actually use today
/// (WASD + modifiers + the hotkey set + arrow keys + function keys 1-12
/// + digits). Returns `None` for unmapped codes; the caller drops the
/// event silently.
///
/// Growing this table: each new code we want to support adds one line.
/// Stays in worker.rs for now because it's the only consumer; if a
/// second crate ever needs the same translation, lift into rye-input.
fn winit_keycode_from_str(code: &str) -> Option<KeyCode> {
    // Letters A-Z. Done as a single match arm because the cases are
    // mechanical + writing them out gives the compiler the chance to
    // jump-table optimize.
    let by_letter = match code {
        "KeyA" => Some(KeyCode::KeyA), "KeyB" => Some(KeyCode::KeyB),
        "KeyC" => Some(KeyCode::KeyC), "KeyD" => Some(KeyCode::KeyD),
        "KeyE" => Some(KeyCode::KeyE), "KeyF" => Some(KeyCode::KeyF),
        "KeyG" => Some(KeyCode::KeyG), "KeyH" => Some(KeyCode::KeyH),
        "KeyI" => Some(KeyCode::KeyI), "KeyJ" => Some(KeyCode::KeyJ),
        "KeyK" => Some(KeyCode::KeyK), "KeyL" => Some(KeyCode::KeyL),
        "KeyM" => Some(KeyCode::KeyM), "KeyN" => Some(KeyCode::KeyN),
        "KeyO" => Some(KeyCode::KeyO), "KeyP" => Some(KeyCode::KeyP),
        "KeyQ" => Some(KeyCode::KeyQ), "KeyR" => Some(KeyCode::KeyR),
        "KeyS" => Some(KeyCode::KeyS), "KeyT" => Some(KeyCode::KeyT),
        "KeyU" => Some(KeyCode::KeyU), "KeyV" => Some(KeyCode::KeyV),
        "KeyW" => Some(KeyCode::KeyW), "KeyX" => Some(KeyCode::KeyX),
        "KeyY" => Some(KeyCode::KeyY), "KeyZ" => Some(KeyCode::KeyZ),
        _ => None,
    };
    if by_letter.is_some() {
        return by_letter;
    }

    let by_digit = match code {
        "Digit0" => Some(KeyCode::Digit0), "Digit1" => Some(KeyCode::Digit1),
        "Digit2" => Some(KeyCode::Digit2), "Digit3" => Some(KeyCode::Digit3),
        "Digit4" => Some(KeyCode::Digit4), "Digit5" => Some(KeyCode::Digit5),
        "Digit6" => Some(KeyCode::Digit6), "Digit7" => Some(KeyCode::Digit7),
        "Digit8" => Some(KeyCode::Digit8), "Digit9" => Some(KeyCode::Digit9),
        _ => None,
    };
    if by_digit.is_some() {
        return by_digit;
    }

    let by_fn = match code {
        "F1" => Some(KeyCode::F1),  "F2" => Some(KeyCode::F2),
        "F3" => Some(KeyCode::F3),  "F4" => Some(KeyCode::F4),
        "F5" => Some(KeyCode::F5),  "F6" => Some(KeyCode::F6),
        "F7" => Some(KeyCode::F7),  "F8" => Some(KeyCode::F8),
        "F9" => Some(KeyCode::F9),  "F10" => Some(KeyCode::F10),
        "F11" => Some(KeyCode::F11), "F12" => Some(KeyCode::F12),
        _ => None,
    };
    if by_fn.is_some() {
        return by_fn;
    }

    // Everything else: catch-all of the common control + arrow keys.
    match code {
        "Space" => Some(KeyCode::Space),
        "Enter" => Some(KeyCode::Enter),
        "Escape" => Some(KeyCode::Escape),
        "Tab" => Some(KeyCode::Tab),
        "Backspace" => Some(KeyCode::Backspace),
        "Delete" => Some(KeyCode::Delete),
        "Backquote" => Some(KeyCode::Backquote),
        "Minus" => Some(KeyCode::Minus),
        "Equal" => Some(KeyCode::Equal),
        "BracketLeft" => Some(KeyCode::BracketLeft),
        "BracketRight" => Some(KeyCode::BracketRight),
        "Semicolon" => Some(KeyCode::Semicolon),
        "Quote" => Some(KeyCode::Quote),
        "Comma" => Some(KeyCode::Comma),
        "Period" => Some(KeyCode::Period),
        "Slash" => Some(KeyCode::Slash),
        "Backslash" => Some(KeyCode::Backslash),
        "ShiftLeft" => Some(KeyCode::ShiftLeft),
        "ShiftRight" => Some(KeyCode::ShiftRight),
        "ControlLeft" => Some(KeyCode::ControlLeft),
        "ControlRight" => Some(KeyCode::ControlRight),
        "AltLeft" => Some(KeyCode::AltLeft),
        "AltRight" => Some(KeyCode::AltRight),
        "ArrowUp" => Some(KeyCode::ArrowUp),
        "ArrowDown" => Some(KeyCode::ArrowDown),
        "ArrowLeft" => Some(KeyCode::ArrowLeft),
        "ArrowRight" => Some(KeyCode::ArrowRight),
        _ => None,
    }
}
