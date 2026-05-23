//! Main-thread side of worker mode. Handles the click-to-launch button,
//! the worker construction + `OffscreenCanvas` transfer, and the DOM
//! event listeners that forward input messages to the worker.
//!
//! The worker-side counterpart lives in [`super::worker`]. The protocol
//! between them is defined in [`super::messages`].

use anyhow::{anyhow, Context, Result};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlButtonElement, HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

/// Main-thread entry point for worker mode. Wires the launch button so a
/// click transfers the page's canvas to a freshly-spawned worker, which
/// then runs the App's lifecycle inside the worker.
///
/// Parameters:
/// - `host_id`: container element with the `data-mode="manual"` attribute.
/// - `button_id`: element to wire the click handler on.
/// - `canvas_id`: the `<canvas>` to transfer via
///   `transferControlToOffscreen` and hand to the worker as its render
///   target.
///
/// Returns immediately after wiring the listener; the click might happen
/// seconds or minutes later. Errors at this stage are configuration
/// mistakes (missing DOM elements, etc.) and surface via the returned
/// `Result`. Errors at click-time go to `tracing::error!` instead since
/// there's no caller to return to at that point.
pub fn launch_on_click(host_id: &str, button_id: &str, canvas_id: &str) -> Result<()> {
    // Main thread also wants tracing routed to DevTools so any setup-time
    // errors are visible. Worker side installs its own (separate JS heap).
    super::worker::install_logging_idempotent();

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

/// URL of the wasm bundle's JS entry, set by the demo's inline script via
/// `window.__rye_wasm_url = import.meta.url` (or by parsing trunk's
/// inline import script). The worker is constructed pointing at this URL
/// so it runs the same wasm binary the main thread loaded; the binary
/// detects its context via [`super::is_worker_context`].
///
/// Captured as a function rather than a constant because `import.meta.url`
/// is per-document and we read it at launch time.
fn read_wasm_bundle_url() -> Result<String> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("__rye_wasm_url"))
        .map_err(|e| anyhow!("read __rye_wasm_url: {e:?}"))?;
    val.as_string()
        .ok_or_else(|| anyhow!("__rye_wasm_url is not a string; demo's index.html must set it"))
}

/// Spawn the worker, transfer the canvas, post the init message.
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
    // CSS stretches the rendered image, producing a squashed aspect ratio
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
    tracing::info!("rye_app::wasm::worker: spawning worker (js={js_url}, wasm={wasm_url})");

    // wasm-bindgen's generated `--target web` ESM exports `init` as the
    // default but doesn't auto-run at import time. The page's main script
    // calls `init({ module_or_path: ... })` explicitly; the worker needs
    // the same. Instead of shipping a separate `worker_bootstrap.js` per
    // demo, build the bootstrap inline as a Blob URL pointing at the same
    // wasm bundle.
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
    // posting the Init message. Without this handshake, the Init might
    // arrive before the worker's listener exists — Firefox empirically
    // drops such messages despite the spec implying queue semantics.
    let worker_for_ready = worker.clone();
    let offscreen_for_ready = offscreen.clone();
    let on_ready = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() != Some("ready") {
            return;
        }
        tracing::info!("rye_app::wasm::worker: worker signalled ready, posting init");

        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("kind"), &JsValue::from_str("init"));
        let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("canvas"), &offscreen_for_ready);
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
    // (or via the worker's handle_message queue) and applied on first frame.
    install_dom_input_forwarders(&worker, &canvas)
        .context("install_dom_input_forwarders")?;

    // Keep the Worker alive. Dropping it would terminate. Box::leak is the
    // wasm-bindgen idiom for "this lives forever in this page."
    Box::leak(Box::new(worker));

    Ok(())
}

/// Install all the DOM event listeners that forward main-thread events to
/// the worker via postMessage. Each listener constructs a typed JS object
/// (`{kind: "...", ...}`) the worker-side [`super::messages::parse_non_init`]
/// parses into an `InputMessage` variant.
///
/// Listeners attached:
/// - `window`: resize, focus, blur, visibilitychange
/// - `canvas` (the placeholder post-transfer): mousemove (rAF-coalesced),
///   mousedown, mouseup, wheel
/// - `document`: keydown, keyup (keyboard listeners on document so we
///   capture keys regardless of which element has focus, matching
///   game-style input expectations)
///
/// All listeners use `Closure::wrap` + `forget()` to leak themselves into
/// the JS heap, since they live for the page's lifetime.
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
    {
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

        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let raf_cb: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
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
    // element has focus (game convention).
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

    // Focus / blur on window for the "release held buttons on focus loss"
    // behaviour rye-input expects.
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

    // Page-visibility on document: tabs being backgrounded should let the
    // app pause continuous animation so we're not wasting CPU.
    {
        let worker = worker.clone();
        let document_for_query = document.clone();
        let cb = Closure::wrap(Box::new(move || {
            let visible =
                document_for_query.visibility_state() != web_sys::VisibilityState::Hidden;
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
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("kind"), &JsValue::from_str(kind));
    obj
}

fn set_msg_u32(obj: &js_sys::Object, key: &str, v: u32) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(v as f64));
}

fn set_msg_f32(obj: &js_sys::Object, key: &str, v: f32) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(v as f64));
}

fn set_msg_bool(obj: &js_sys::Object, key: &str, v: bool) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_bool(v));
}

fn set_msg_string(obj: &js_sys::Object, key: &str, v: &str) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(v));
}
