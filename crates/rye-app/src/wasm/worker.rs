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
use web_sys::{
    DedicatedWorkerGlobalScope, HtmlButtonElement, HtmlCanvasElement, MessageEvent,
    OffscreenCanvas, Worker, WorkerOptions, WorkerType,
};

use crate::{App, FrameCtx, RenderCtx, SetupCtx};
use rye_asset::AssetWatcher;
use rye_input::FrameInput;
use rye_render::device::RenderDevice;
use rye_shader::ShaderDb;

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

    // Construct the App via the shared `SetupCtx` shape. AssetWatcher on
    // wasm32 is a no-op stub; ShaderDb is straightforward.
    let runner = WorkerRunner::<A>::setup(rd)
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
    #[allow(dead_code)] // held alive for App's runtime; lookups via ShaderDb come in Phase B3+
    shader_db: ShaderDb,
    #[allow(dead_code)] // wasm stub today; native parity in case the trait grows
    watcher: Option<AssetWatcher>,
    app: A,
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
    /// invoke `A::setup`. Async because `A::setup` may itself await on
    /// asset loading or device-feature probes; in practice for the
    /// existing demos it's synchronous and returns immediately.
    async fn setup(rd: RenderDevice) -> Result<Self> {
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
        Ok(Self {
            rd,
            shader_db,
            watcher,
            app,
            start: web_time::Instant::now(),
            last_update_at: None,
            tick_index: 0,
            _marker: PhantomData,
        })
    }

    /// One frame: dt, App::update, begin_frame, App::record, composite,
    /// submit, present. Wraps `rye_time::frame_trace::scope` so the same
    /// telemetry the windowed-mode runner emits is available here.
    fn frame(&mut self) -> Result<()> {
        rye_time::frame_trace::begin_frame();
        let _frame_scope = rye_time::frame_trace::scope("frame");

        // dt: wall-clock since previous update. First frame falls back
        // to 1/60 so the App doesn't see a 0 dt that breaks integrators.
        let now = web_time::Instant::now();
        let dt = match self.last_update_at {
            Some(prev) => now.duration_since(prev).as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.last_update_at = Some(now);
        self.tick_index = self.tick_index.wrapping_add(1);

        // App::update with an empty FrameInput (Phase B3 will plumb real
        // input from main-thread InputMessage events).
        {
            let _scope = rye_time::frame_trace::scope("app-update");
            let mut fctx = FrameCtx {
                rd: &self.rd,
                input: FrameInput::default(),
                time: self.start.elapsed().as_secs_f32(),
                fps: 0.0, // Phase B1: skip FPS bookkeeping; not used by tesseract_demo.
                n_ticks: 0,
                tick: self.tick_index,
                dt,
                ui_has_focus: false,
                _non_exhaustive: PhantomData,
            };
            self.app.update(&mut fctx);
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

    // Read the canvas's current pixel dimensions. CSS-sized canvases use
    // these attributes; if they're zero, the worker would create a 0×0
    // surface, which wgpu rejects. The page is responsible for sizing the
    // canvas before launch.
    let width = canvas.width();
    let height = canvas.height();
    if width == 0 || height == 0 {
        return Err(anyhow!(
            "canvas '{canvas_id}' has zero dimensions ({width}x{height}); set width/height \
             attributes or CSS before launch"
        ));
    }

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

    // Keep the Worker alive. Dropping it would terminate. Box::leak is the
    // wasm-bindgen idiom for "this lives forever in this page."
    Box::leak(Box::new(worker));

    Ok(())
}
