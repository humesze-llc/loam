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
use web_sys::{HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

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

    // Spawn the worker IMMEDIATELY on page load (not on click). The
    // worker initializes wgpu + renders a single preview frame, then
    // idles. The launch button is now a full-cover overlay that
    // displays "click anywhere to launch" with a `backdrop-filter:
    // blur(...)` applied over the canvas underneath; so the viewer
    // sees a blurred preview of the demo's first frame, not a dark
    // placeholder. When the user clicks, the click handler posts a
    // `Start` message to the worker (kicks off its RAF loop) AND
    // removes the launch overlay (revealing the live canvas).
    //
    // Trade-off: the wasm bundle + a worker spawn happen at page-load
    // time, which costs bandwidth + memory before the viewer has shown
    // interest. For single-demo blog pages that's fine; for pages
    // embedding many demos at once, a future IntersectionObserver-based
    // lazy-spawn would be the right answer (NH6 in the perf plan).
    let _ = spawn_worker_for_preview(canvas_id, host_id, button_id)?;
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

/// Spawn the worker, transfer the canvas, post the init message, AND
/// wire the launch-overlay click handler that posts the `Start`
/// message + removes the overlay on click. Called once at page load
/// from `launch_on_click`.
fn spawn_worker_for_preview(canvas_id: &str, host_id: &str, button_id: &str) -> Result<()> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| anyhow!("no element with id '{canvas_id}'"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| anyhow!("element '{canvas_id}' is not a canvas"))?;
    // Engine owns the overlay markup + CSS. `inject_launch_overlay` is
    // idempotent: returns an existing button with `button_id` if the
    // demo's `index.html` shipped one (legacy demos), otherwise creates
    // it as a child of `host_id` and injects the default CSS into
    // `<head>` exactly once per page.
    let launch_overlay = super::launch::inject_launch_overlay(host_id, button_id)?;

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
    let bootstrap_js =
        format!("import init from '{js_url}';\nawait init({{ module_or_path: '{wasm_url}' }});\n");
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
    let worker =
        Worker::new_with_options(&blob_url, &opts).map_err(|e| anyhow!("Worker::new: {e:?}"))?;

    // Shared state for the ready handshake + the click-before-ready race:
    // - `worker_ready`: set to true once the worker has posted `ready`
    //   and main has posted `init` back. Once true, the click handler
    //   can post Start immediately.
    // - `pending_start`: set by the click handler if the user clicks
    //   BEFORE the worker is ready. The on_ready handler reads this
    //   and posts Start as part of its own sequence.
    //
    // Without this, an eager click during the wasm-bundle-download
    // window posts Start to a Worker that hasn't yet installed its
    // `message` listener; Firefox drops the message and the demo
    // never starts. Same root cause as the init handshake.
    let worker_ready: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));
    let pending_start: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));

    // Wait for the worker to signal it's ready (handler installed) before
    // posting the Init message. Without this handshake, the Init might
    // arrive before the worker's listener exists; Firefox empirically
    // drops such messages despite the spec implying queue semantics.
    let worker_for_ready = worker.clone();
    let offscreen_for_ready = offscreen.clone();
    let worker_ready_for_ready = worker_ready.clone();
    let pending_start_for_ready = pending_start.clone();
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
            return;
        }

        // The worker is now listening; subsequent click-Start posts are
        // safe. If the user already clicked while we were waiting for
        // ready, send the queued Start now.
        worker_ready_for_ready.set(true);
        if pending_start_for_ready.replace(false) {
            tracing::info!(
                "rye_app::wasm::worker: click occurred before ready; posting queued Start"
            );
            let start_msg = build_msg("start");
            if let Err(e) = worker_for_ready.post_message(&start_msg) {
                tracing::error!("rye_app::wasm::worker: postMessage queued Start failed: {e:?}");
            }
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
    install_dom_input_forwarders(&worker, &canvas).context("install_dom_input_forwarders")?;

    // Worker → main DOM-action channel + pointerlockchange forwarder +
    // canvas click-to-re-engage. This is the standard OffscreenCanvas-
    // in-Worker pattern for letting the worker drive DOM APIs that can
    // only be called from main thread.
    install_host_action_handler(&worker, &canvas).context("install_host_action_handler")?;

    // Listen for the worker's "preview_ready" signal -- sent once after
    // the worker has rendered the preview frame AND pre-warmed
    // pipelines. On receipt, promote the overlay from `.initializing`
    // to `.ready` so the user sees the click-to-launch affordance only
    // when the demo is genuinely ready. Click then removes the overlay
    // and starts the RAF loop without any further wait.
    install_preview_ready_handler(&worker, button_id)?;

    // Launch-overlay click handler. Spam-click defensive:
    // - FnMut closure (not `Closure::once`) so repeat invocations are
    //   guaranteed to be safe no-ops via the `fired` Cell guard. The
    //   `Closure::once` variant relies on internal `Option::take`
    //   semantics that aren't visibly diagnostic when they swallow
    //   repeat clicks; the explicit Cell is easier to reason about.
    // - Post Start FIRST, then disable + hide the overlay. If the post
    //   fails (worker terminated, etc.), the overlay stays visible so
    //   a retry click can fire again; better than "overlay gone and
    //   nothing happens".
    // - Set `pointer-events: none` on the overlay BEFORE removal so
    //   any pending click events queued by the browser don't fire on
    //   the not-yet-removed element.
    {
        let worker_for_click = worker.clone();
        let overlay_for_click = launch_overlay.clone();
        let worker_ready_for_click = worker_ready.clone();
        let pending_start_for_click = pending_start.clone();
        let fired: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));
        let on_click = Closure::wrap(Box::new(move || {
            if fired.get() {
                tracing::debug!("rye_app::wasm::worker: launch click ignored (already fired)");
                return;
            }
            // Gate on the overlay being in `.ready` state. Clicks during
            // `.initializing` (worker still spawning + preview frame not
            // rendered) are absorbed silently so the user can spam-click
            // without queueing a Start that fires before the worker is
            // actually ready to handle it.
            if !overlay_for_click.class_name().contains("ready") {
                tracing::debug!(
                    "rye_app::wasm::worker: launch click ignored (not yet ready, overlay state = {})",
                    overlay_for_click.class_name()
                );
                return;
            }
            fired.set(true);

            // If the worker isn't ready yet, QUEUE the Start intent.
            // The on_ready handler will send Start as soon as the
            // worker comes online. Firefox empirically drops messages
            // sent to a Worker before its listener is installed, so we
            // must not post Start during that window.
            if !worker_ready_for_click.get() {
                pending_start_for_click.set(true);
                tracing::info!(
                    "rye_app::wasm::worker: launch click before worker ready; \
                     queued Start for on_ready handler"
                );
            } else {
                tracing::info!("rye_app::wasm::worker: launch click; posting Start");
                let msg = build_msg("start");
                if let Err(e) = worker_for_click.post_message(&msg) {
                    // Keep the overlay visible so the user can retry. Reset
                    // `fired` so a subsequent click can try again.
                    fired.set(false);
                    tracing::error!(
                        "rye_app::wasm::worker: postMessage Start failed: {e:?}; \
                         overlay retained for retry"
                    );
                    return;
                }
            }

            // Worker pre-warmed pipelines before posting preview_ready,
            // so the demo is genuinely ready right now. Just remove the
            // overlay; click → demo is one transition, no second wait.
            overlay_for_click.remove();
        }) as Box<dyn FnMut()>);
        launch_overlay
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("launch overlay click listener: {e:?}"))?;
        on_click.forget();
    }

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

    // Resize: DEBOUNCED. Drag-to-resize fires the DOM event 30-60×/sec;
    // each event would trigger a wgpu `surface.configure()` on the worker
    // side, which recreates the swap chain + scene texture + composite
    // bind group (heavy + JS-allocating). Even at one reconfigure per
    // browser-frame the worker can't keep up; mid-drag the tab gets
    // genuinely stuck (we hit a hard freeze + crash during testing).
    //
    // The fix is debouncing, not just rate-limiting: wait until the user
    // has STOPPED resizing (~100ms of no events) before committing the
    // new size. During the drag the canvas backing-store stays at its
    // old size; CSS scales it visually so the demo stretches briefly
    // but no GPU work happens. Once the drag settles, one resize
    // message is sent and the worker reconfigures once.
    //
    // 100ms ≈ 6 frames at 60Hz: long enough to coalesce a sustained
    // drag, short enough that the final commit feels responsive.
    const RESIZE_DEBOUNCE_FRAMES: u32 = 6;
    {
        // Shared state: `Some((w, h, frames_since_last_event))` while
        // a resize is in flight; `None` when settled. Each new event
        // resets the frame counter to 0 so a continuous drag never
        // commits an intermediate size.
        let pending: Rc<RefCell<Option<(u32, u32, u32)>>> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let canvas_for_listener = canvas.clone();
        let window_for_listener = window.clone();
        let cb = Closure::wrap(Box::new(move || {
            let dpr = window_for_listener.device_pixel_ratio() as f32;
            let w = (canvas_for_listener.client_width() as f32 * dpr).max(1.0) as u32;
            let h = (canvas_for_listener.client_height() as f32 * dpr).max(1.0) as u32;
            *pending_for_listener.borrow_mut() = Some((w, h, 0));
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("window resize listener: {e:?}"))?;
        cb.forget();

        // rAF tick: advance the debounce countdown. When idle for the
        // threshold, commit the final size to the worker.
        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let raf_cb: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
        let raf_cb_for_closure = raf_cb.clone();
        *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            let commit = {
                let mut p = pending_for_raf.borrow_mut();
                match p.as_mut() {
                    Some((_, _, frames)) => {
                        *frames += 1;
                        if *frames >= RESIZE_DEBOUNCE_FRAMES {
                            // We just observed Some via `as_mut()`, so take()
                            // returns Some here. `.map` keeps this expression-
                            // shaped (no .expect, no panicking unwrap).
                            p.take().map(|(w, h, _)| (w, h))
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            };
            if let Some((w, h)) = commit {
                let msg = build_msg("resize");
                set_msg_u32(&msg, "width", w);
                set_msg_u32(&msg, "height", h);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            // `raf_cb` was just populated three lines up; the borrow is fresh.
            // Fall through silently if somehow None (browser would just never
            // animate; not a panicking condition).
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(|e| anyhow!("resize rAF init: {e:?}"))?;
            }
        }
        Box::leak(Box::new(raf_cb));
    }

    // Mouse move on the canvas. rAF-COALESCED: a DOM mouse-move can fire
    // hundreds of times per second; forwarding each one is a JS-object
    // allocation per event + a postMessage. At sustained drag speeds
    // that's enough to overwhelm the JS heap and crash the browser tab
    // (verified empirically during a sustained-drag stress test).
    //
    // Coalesce: the listener writes the latest mouse position into a
    // shared RefCell. A separate rAF loop checks for a pending mouse-
    // move once per browser-frame and posts ONE message with the
    // latest position. Result: at most 60 mouse-move postMessages per
    // second regardless of input device frequency.
    {
        // Pending state carries: latest (x, y, buttons) plus the *summed*
        // `movementX/Y` across all coalesced events since the last RAF
        // tick. Latest absolute position is what the worker uses for the
        // egui cursor; summed deltas are what the worker uses for FPS
        // mouse-look (`mouse_raw_delta`) so intermediate sub-frame motion
        // isn't lost when we coalesce.
        let pending: Rc<RefCell<Option<(f32, f32, u32, f32, f32)>>> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            let mut p = pending_for_listener.borrow_mut();
            let (sum_dx, sum_dy) = match *p {
                Some((_, _, _, dx, dy)) => (dx, dy),
                None => (0.0, 0.0),
            };
            *p = Some((
                ev.offset_x() as f32,
                ev.offset_y() as f32,
                ev.buttons() as u32,
                sum_dx + ev.movement_x() as f32,
                sum_dy + ev.movement_y() as f32,
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
            if let Some((x, y, buttons, dx, dy)) = pending_for_raf.borrow_mut().take() {
                let msg = build_msg("mouse_move");
                set_msg_f32(&msg, "x", x);
                set_msg_f32(&msg, "y", y);
                set_msg_u32(&msg, "buttons", buttons);
                set_msg_f32(&msg, "dx", dx);
                set_msg_f32(&msg, "dy", dy);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            // `raf_cb` was just populated four lines up; the borrow is fresh.
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(|e| anyhow!("mousemove rAF init: {e:?}"))?;
            }
        }
        Box::leak(Box::new(raf_cb));
    }

    // Mouse down / up: same shape, different `pressed` flag.
    for (event_name, pressed) in [("mousedown", true), ("mouseup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            // Suppress browser default for right (2) + middle (1) button
            // so right-click context menu and middle-click autoscroll
            // don't fire ahead of the app receiving the press. Left
            // button (0) is left alone; demos that pointer-lock will
            // request the lock on left-click themselves.
            if ev.button() != 0 {
                ev.prevent_default();
            }
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

    // Block the browser context menu so right-click reaches the app.
    // Separate from `mousedown`'s suppression because `contextmenu` also
    // fires for keyboard triggers (Shift+F10, the dedicated context-menu
    // key) which `mousedown` never sees.
    {
        let cb = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            ev.prevent_default();
        }) as Box<dyn FnMut(web_sys::Event)>);
        canvas
            .add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("contextmenu listener: {e:?}"))?;
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
    //
    // `preventDefault()` is selective. The browser owns reload (F5,
    // Ctrl+R), devtools (F12), tab/window management (Ctrl+T / Ctrl+W /
    // Ctrl+Tab), fullscreen (F11), and most modifier combos; we leave
    // those alone so the user retains an escape hatch. The demo owns
    // Tab (otherwise focus walks to off-canvas DOM elements and
    // subsequent keys never reach us), Space + arrows (otherwise the
    // browser scrolls or activates a focused button), slash + quote
    // (Firefox quick-find), and Alt (Chrome/Firefox treat lone-Alt taps
    // as menu-bar activation on the keyup edge; preventDefault on both
    // edges suppresses). Letter keys never trigger a browser default
    // action at the page level, so WASD / T / etc. need no handling.
    for (event_name, pressed) in [("keydown", true), ("keyup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::KeyboardEvent| {
            let code = ev.code();
            let no_modifier = !ev.ctrl_key() && !ev.alt_key() && !ev.meta_key();
            // Alt as the key itself: the `no_modifier` check excludes
            // this case (ev.alt_key is true while Alt is the down key),
            // so it gets its own check. Skip when Ctrl/Cmd is held so
            // Ctrl+Alt / Cmd+Alt combos still work as intended.
            let is_alt_self = matches!(code.as_str(), "AltLeft" | "AltRight");
            let suppress_alt = is_alt_self && !ev.ctrl_key() && !ev.meta_key();
            let owned_unmodified = matches!(
                code.as_str(),
                "Tab"
                    | "Space"
                    | "ArrowLeft"
                    | "ArrowRight"
                    | "ArrowUp"
                    | "ArrowDown"
                    | "Slash"
                    | "Quote",
            );
            if suppress_alt || (owned_unmodified && no_modifier) {
                ev.prevent_default();
            }
            let msg = build_msg("key");
            set_msg_string(&msg, "code", &code);
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
            let visible = document_for_query.visibility_state() != web_sys::VisibilityState::Hidden;
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

/// Listen for the worker's once-only `{kind: "demo_ready"}` message and
/// remove the launch overlay when it arrives. The worker sends this
/// after rendering a handful of warm-up frames (currently 10, per
/// `worker.rs::DEMO_READY_FRAMES`), which is enough for the heavy
/// per-pipeline shader compilations to settle and the first sim ticks
/// to land at the target rate. Until then the launch overlay's
/// `.loading` state absorbs spam clicks so they don't compound the
/// startup hitch.
/// Listen for the worker's once-only `{kind: "preview_ready"}` message
/// and promote the launch overlay from `.initializing` to `.ready` so
/// the "Click to launch" affordance becomes visible only after the
/// blurred preview frame has actually rendered. Before this, the
/// overlay shows "Initializing…" with a spinner and the click handler
/// short-circuits so spam clicks are absorbed.
fn install_preview_ready_handler(worker: &Worker, button_id: &str) -> Result<()> {
    let button_id_owned: String = button_id.to_string();
    let cb = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() != Some("preview_ready") {
            return;
        }
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        // Remove the static page-load spinner from `index.html`. The
        // wasm overlay (below) takes over the visual layer from here.
        if let Some(loader) = document.get_element_by_id("rye-page-loader") {
            loader.remove();
        }
        if let Some(overlay) = document.get_element_by_id(&button_id_owned) {
            overlay.set_class_name("rye-demo-launch ready");
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", cb.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message') for preview_ready: {e:?}"))?;
    cb.forget();
    Ok(())
}

/// Install the worker → main DOM-action handler, the
/// `pointerlockchange` → worker forwarder, and the canvas
/// click-to-re-engage Pointer Lock helper.
///
/// ## Architecture
///
/// The worker (in `wasm::worker`) posts
/// `{kind: "host_action", actions: [{kind: "pointer_lock_request"}, ...]}`
/// messages whenever an engine module (`rye_app::cursor`, future
/// `fullscreen` / `clipboard` / ...) queues a DOM-touching action. Every
/// browser API that needs DOM access has to round-trip through here
/// because the worker has no `document` / `window` / canvas-element
/// handle of its own; it owns only the `OffscreenCanvas` bitmap surface.
///
/// ## Pointer Lock specifically
///
/// `canvas.requestPointerLock()` succeeds when called inside the
/// transient-activation window (~5 sec) opened by the user's last
/// key/click. The worker can't see the activation token, but the main
/// thread can: a console-driven freecam toggle starts as an `Enter`
/// keydown on `document`, the worker processes it on the next frame,
/// queues a `PointerLockRequest`, and the round-trip from key-event to
/// `requestPointerLock` call is comfortably under that 5 sec window.
///
/// `document.exitPointerLock()` is always allowed (no activation
/// needed); the browser also auto-releases on Esc, on tab switch, on
/// visibility change. Each of those produces a `pointerlockchange`
/// event we forward back to the worker as
/// `InputMessage::PointerLockChanged(false)` so the engine cursor state
/// stays accurate.
///
/// ## Click-to-re-engage
///
/// When the browser auto-releases lock (e.g., Esc) but the demo still
/// wants it (the worker last requested grab; `want_locked = true`),
/// clicking the canvas re-requests the lock. This is the standard
/// WebGL-game pattern.
fn install_host_action_handler(worker: &Worker, canvas: &HtmlCanvasElement) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;

    // Demo's desired lock state, as last requested by the worker. Lives
    // here on the main thread (the worker's mirror via
    // `cursor::current_state` updates from `pointerlockchange` events
    // we forward back). Drives the canvas-click re-engagement decision:
    // if want_locked is true but the actual lock dropped (Esc, tab
    // switch), the next click re-requests. If want_locked is false (the
    // demo voluntarily released), the click does nothing.
    let want_locked: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));

    // Worker → main: drain a `host_action` message into DOM calls.
    {
        let canvas_for_dispatch = canvas.clone();
        let document_for_dispatch = document.clone();
        let want_locked_for_dispatch = want_locked.clone();
        let cb = Closure::wrap(Box::new(move |event: MessageEvent| {
            let data: JsValue = event.data();
            let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
                .ok()
                .and_then(|v| v.as_string());
            if kind.as_deref() != Some("host_action") {
                return;
            }
            let actions = match js_sys::Reflect::get(&data, &JsValue::from_str("actions")) {
                Ok(arr) => arr,
                Err(_) => return,
            };
            let arr = match actions.dyn_into::<js_sys::Array>() {
                Ok(a) => a,
                Err(_) => return,
            };
            for i in 0..arr.length() {
                let item = arr.get(i);
                let action_kind = js_sys::Reflect::get(&item, &JsValue::from_str("kind"))
                    .ok()
                    .and_then(|v| v.as_string());
                match action_kind.as_deref() {
                    Some("pointer_lock_request") => {
                        *want_locked_for_dispatch.borrow_mut() = true;
                        // `requestPointerLock` resolves async; the
                        // `pointerlockchange` listener below relays
                        // success/failure back to the worker. Failure
                        // can mean "no transient activation" or "another
                        // tab already has lock"; in either case the
                        // demo learns via the relay and can act on it
                        // (e.g., the user clicks again and we retry).
                        canvas_for_dispatch.request_pointer_lock();
                    }
                    Some("pointer_lock_release") => {
                        *want_locked_for_dispatch.borrow_mut() = false;
                        document_for_dispatch.exit_pointer_lock();
                    }
                    Some(other) => {
                        tracing::warn!(
                            "rye_app::wasm: unknown host_action kind '{other}'; dropping"
                        );
                    }
                    None => {
                        tracing::warn!("rye_app::wasm: host_action item missing 'kind'");
                    }
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        worker
            .add_event_listener_with_callback("message", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("worker.addEventListener('message') for host_action: {e:?}"))?;
        cb.forget();
    }

    // `pointerlockchange` on document: tells us the truth about the
    // browser's current lock state, whether we triggered it (requestPointerLock
    // / exitPointerLock) or the user did (Esc, tab switch). Forward to
    // the worker so `cursor::mark_applied` reflects reality.
    {
        let worker_for_change = worker.clone();
        let document_for_change = document.clone();
        let canvas_for_change = canvas.clone();
        let cb = Closure::wrap(Box::new(move || {
            let locked = document_for_change
                .pointer_lock_element()
                .as_ref()
                .map(|el| el == canvas_for_change.as_ref())
                .unwrap_or(false);
            let msg = build_msg("pointer_lock_changed");
            set_msg_bool(&msg, "locked", locked);
            let _ = worker_for_change.post_message(&msg);
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback("pointerlockchange", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("pointerlockchange listener: {e:?}"))?;
        cb.forget();
    }

    // Canvas click: re-engage Pointer Lock if the demo last requested
    // grab but the browser dropped it (Esc). No-op if want_locked is
    // false (release was voluntary) or if the browser already has lock
    // (the click landed during normal interaction, no work to do).
    {
        let canvas_for_click = canvas.clone();
        let document_for_click = document.clone();
        let want_locked_for_click = want_locked.clone();
        let cb = Closure::wrap(Box::new(move |_ev: web_sys::MouseEvent| {
            if !*want_locked_for_click.borrow() {
                return;
            }
            let already_locked = document_for_click
                .pointer_lock_element()
                .as_ref()
                .map(|el| el == canvas_for_click.as_ref())
                .unwrap_or(false);
            if already_locked {
                return;
            }
            canvas_for_click.request_pointer_lock();
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("canvas click listener: {e:?}"))?;
        cb.forget();
    }

    Ok(())
}
