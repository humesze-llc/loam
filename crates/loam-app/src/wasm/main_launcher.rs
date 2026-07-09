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

/// Main-thread entry for worker mode. Wires the launch overlay so a click
/// starts the worker (already spawned, see below); `host_id` is the
/// `data-mode="manual"` container, `button_id` the overlay, `canvas_id` the
/// `<canvas>` transferred via `transferControlToOffscreen`.
///
/// Returns after wiring; errors here are config mistakes (missing DOM
/// elements). Click-time errors go to `tracing::error!`, no caller to
/// return to.
pub fn launch_on_click(host_id: &str, button_id: &str, canvas_id: &str) -> Result<()> {
    super::worker::install_logging_idempotent();

    // Spawn the worker on page load, not on click: it inits wgpu, renders a
    // preview frame for the backdrop-blurred overlay, then idles until the
    // click posts `Start`. Trade-off is wasm + worker cost before the viewer
    // shows interest; fine for single-demo pages, a future
    // IntersectionObserver lazy-spawn fits many-demo pages.
    let _ = spawn_worker_for_preview(canvas_id, host_id, button_id)?;
    Ok(())
}

/// URL of the wasm bundle's JS entry, set by the demo's inline script via
/// `window.__loam_wasm_url = import.meta.url`. The worker points at this so
/// it runs the same binary, branching on [`super::is_worker_context`]. Read
/// at launch time because `import.meta.url` is per-document.
fn read_wasm_bundle_url() -> Result<String> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("__loam_wasm_url"))
        .map_err(|e| anyhow!("read __loam_wasm_url: {e:?}"))?;
    val.as_string()
        .ok_or_else(|| anyhow!("__loam_wasm_url is not a string; demo's index.html must set it"))
}

/// Spawn the worker, transfer the canvas, post init, and wire the
/// launch-overlay click handler (posts `Start`, removes the overlay).
fn spawn_worker_for_preview(canvas_id: &str, host_id: &str, button_id: &str) -> Result<()> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| anyhow!("no element with id '{canvas_id}'"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| anyhow!("element '{canvas_id}' is not a canvas"))?;
    // Idempotent: reuses an existing `button_id` element (legacy demos that
    // ship their own) or creates one and injects the CSS once per page.
    let launch_overlay = super::launch::inject_launch_overlay(host_id, button_id)?;

    // Size the backing store to displayed-size × DPR. The HTML width/height
    // are a pre-launch fallback; CSS may stretch the canvas to fill its
    // container, so without recomputing here the image renders squashed.
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
        "loam_app::wasm::worker: canvas sized to {width}x{height} (CSS {css_w}x{css_h} × DPR {dpr})"
    );

    let offscreen = canvas
        .transfer_control_to_offscreen()
        .map_err(|e| anyhow!("transfer_control_to_offscreen: {e:?}"))?;

    let js_url = read_wasm_bundle_url()?;
    // Trunk emits `<name>-<hash>.js` and `<name>-<hash>_bg.wasm` side by
    // side. The wasm URL must be passed to `init()` explicitly: the worker
    // imports the JS via a Blob URL, which has no document base, so the
    // generated relative-path fallback 404s to the page HTML.
    let wasm_url = js_url.strip_suffix(".js").unwrap_or(&js_url).to_string() + "_bg.wasm";
    tracing::info!("loam_app::wasm::worker: spawning worker (js={js_url}, wasm={wasm_url})");

    // wasm-bindgen `--target web` exports `init` but doesn't auto-run on
    // import. Build the bootstrap inline as a Blob URL instead of shipping
    // a per-demo `worker_bootstrap.js`.
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

    // Ready handshake + click-before-ready race. `worker_ready` flips once
    // the worker posted `ready` and main posted `init` back; `pending_start`
    // captures a click that lands before then. Firefox drops messages
    // posted to a worker before its listener installs, so Start must wait.
    let worker_ready: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));
    let pending_start: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));

    // Wait for the worker's `ready` before posting Init; same Firefox
    // drop-before-listener reason as above.
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
        tracing::info!("loam_app::wasm::worker: worker signalled ready, posting init");

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
        // Workers have no `window.location`; forward the page query so
        // `Args::current` works inside `App::setup`.
        let (search, hash) = web_sys::window()
            .map(|w| {
                let loc = w.location();
                (
                    loc.search().unwrap_or_default(),
                    loc.hash().unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        let _ = js_sys::Reflect::set(
            &msg,
            &JsValue::from_str("search"),
            &JsValue::from_str(&search),
        );
        let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("hash"), &JsValue::from_str(&hash));

        let transfer = js_sys::Array::new();
        transfer.push(&offscreen_for_ready);

        if let Err(e) = worker_for_ready.post_message_with_transfer(&msg, &transfer) {
            tracing::error!("loam_app::wasm::worker: postMessage init failed: {e:?}");
            return;
        }

        // Worker is listening now; flush a click that landed during the
        // wait.
        worker_ready_for_ready.set(true);
        if pending_start_for_ready.replace(false) {
            tracing::info!(
                "loam_app::wasm::worker: click occurred before ready; posting queued Start"
            );
            let start_msg = build_msg("start");
            if let Err(e) = worker_for_ready.post_message(&start_msg) {
                tracing::error!("loam_app::wasm::worker: postMessage queued Start failed: {e:?}");
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", on_ready.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message'): {e:?}"))?;
    on_ready.forget();

    // Forward DOM events to the worker. Installed before the worker is
    // ready so setup-window events queue and apply on first frame.
    install_dom_input_forwarders(&worker, &canvas).context("install_dom_input_forwarders")?;

    // Worker -> main DOM-action channel + pointerlockchange forwarder +
    // canvas click-to-re-engage (the standard OffscreenCanvas-in-Worker
    // pattern for DOM APIs only callable from the main thread).
    install_host_action_handler(&worker, &canvas).context("install_host_action_handler")?;

    // `preview_progress` drives the page-loader bar; `preview_ready` removes
    // it.
    install_preview_progress_handler(&worker)?;
    install_preview_ready_handler(&worker, button_id)?;

    install_embed_lifecycle(&worker, host_id, button_id).context("install_embed_lifecycle")?;

    // Launch-overlay click handler, spam-click defensive: FnMut + a `fired`
    // Cell make repeat clicks no-ops; post Start before removing the overlay
    // so a failed post leaves the overlay up for retry.
    {
        let worker_for_click = worker.clone();
        let overlay_for_click = launch_overlay.clone();
        let worker_ready_for_click = worker_ready.clone();
        let pending_start_for_click = pending_start.clone();
        let host_for_click = host_id.to_string();
        let fired: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));
        let on_click = Closure::wrap(Box::new(move || {
            if fired.get() {
                tracing::debug!("loam_app::wasm::worker: launch click ignored (already fired)");
                return;
            }
            // Gate on `.ready`: before `preview_ready` lands, clicks are
            // absorbed so a spam-click can't queue an early Start.
            if !overlay_for_click.class_name().contains("ready") {
                tracing::debug!(
                    "loam_app::wasm::worker: launch click ignored (not yet ready, overlay state = {})",
                    overlay_for_click.class_name()
                );
                return;
            }
            fired.set(true);

            // Worker not ready: queue the Start for the on_ready handler.
            // Posting now would hit the Firefox drop-before-listener window.
            if !worker_ready_for_click.get() {
                pending_start_for_click.set(true);
                tracing::info!(
                    "loam_app::wasm::worker: launch click before worker ready; \
                     queued Start for on_ready handler"
                );
            } else {
                tracing::info!("loam_app::wasm::worker: launch click; posting Start");
                let msg = build_msg("start");
                if let Err(e) = worker_for_click.post_message(&msg) {
                    // Keep the overlay up and allow a retry click.
                    fired.set(false);
                    tracing::error!(
                        "loam_app::wasm::worker: postMessage Start failed: {e:?}; \
                         overlay retained for retry"
                    );
                    return;
                }
            }

            // Pipelines were warmed before preview_ready, so removing the
            // overlay is the whole transition; no second wait.
            overlay_for_click.remove();
            dispatch_embed_activated(&host_for_click);
        }) as Box<dyn FnMut()>);
        launch_overlay
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("launch overlay click listener: {e:?}"))?;
        on_click.forget();
    }

    // Keep the Worker alive: dropping it terminates the worker.
    Box::leak(Box::new(worker));

    Ok(())
}

/// Dispatched on `document` when an embed activates; detail = host id.
/// Every embed listens: its own id flips it active, another id while active
/// deactivates it. One-active-demo-per-page with no shared JS state.
const EMBED_ACTIVATED_EVENT: &str = "loam-embed-activated";

fn dispatch_embed_activated(host_id: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let init = web_sys::CustomEventInit::new();
    init.set_detail(&JsValue::from_str(host_id));
    match web_sys::CustomEvent::new_with_event_init_dict(EMBED_ACTIVATED_EVENT, &init) {
        Ok(event) => {
            let _ = document.dispatch_event(&event);
        }
        Err(e) => tracing::error!("loam_app::wasm::worker: create activated event: {e:?}"),
    }
}

/// Post-launch lifecycle for one embed: pointerdown outside the host posts
/// `pause` and restores the blurred overlay; the overlay click posts
/// `resume` and re-announces; another embed's broadcast deactivates this
/// one. Starts inactive until the launch click's broadcast.
fn install_embed_lifecycle(worker: &Worker, host_id: &str, button_id: &str) -> Result<()> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let host_el = document
        .get_element_by_id(host_id)
        .ok_or_else(|| anyhow!("no host element with id '{host_id}'"))?;
    let active: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));

    // One closure shared across pause cycles; the button is recreated each
    // time, so it re-attaches rather than leaking a closure per cycle.
    let worker_for_resume = worker.clone();
    let host_for_resume = host_id.to_string();
    let button_for_resume = button_id.to_string();
    let on_resume_click: Rc<Closure<dyn FnMut()>> = Rc::new(Closure::wrap(Box::new(move || {
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(button) = doc.get_element_by_id(&button_for_resume) {
                button.remove();
            }
        }
        // Resume before announcing: a failed post leaves the demo paused
        // and the restored overlay retries.
        let msg = build_msg("resume");
        if let Err(e) = worker_for_resume.post_message(&msg) {
            tracing::error!("loam_app::wasm::worker: postMessage resume failed: {e:?}");
            let _ = super::launch::show_resume_overlay(&host_for_resume, &button_for_resume);
            return;
        }
        dispatch_embed_activated(&host_for_resume);
    })
        as Box<dyn FnMut()>));

    let worker_for_deact = worker.clone();
    let host_for_deact = host_id.to_string();
    let button_for_deact = button_id.to_string();
    let active_for_deact = active.clone();
    let resume_cb = on_resume_click.clone();
    let deactivate: Rc<dyn Fn()> = Rc::new(move || {
        if !active_for_deact.replace(false) {
            return;
        }
        let msg = build_msg("pause");
        if let Err(e) = worker_for_deact.post_message(&msg) {
            tracing::error!("loam_app::wasm::worker: postMessage pause failed: {e:?}");
        }
        match super::launch::show_resume_overlay(&host_for_deact, &button_for_deact) {
            Ok(button) => {
                if let Err(e) = button.add_event_listener_with_callback(
                    "click",
                    (*resume_cb).as_ref().unchecked_ref(),
                ) {
                    tracing::error!("loam_app::wasm::worker: resume click listener: {e:?}");
                }
            }
            Err(e) => tracing::error!("loam_app::wasm::worker: show_resume_overlay: {e:#}"),
        }
    });

    let active_for_evt = active.clone();
    let host_for_evt = host_id.to_string();
    let deactivate_for_evt = deactivate.clone();
    let on_activated = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
        if event.detail().as_string().as_deref() == Some(host_for_evt.as_str()) {
            active_for_evt.set(true);
        } else {
            deactivate_for_evt();
        }
    }) as Box<dyn FnMut(web_sys::CustomEvent)>);
    document
        .add_event_listener_with_callback(
            EMBED_ACTIVATED_EVENT,
            on_activated.as_ref().unchecked_ref(),
        )
        .map_err(|e| anyhow!("addEventListener('{EMBED_ACTIVATED_EVENT}'): {e:?}"))?;
    on_activated.forget();

    // Capture phase: page UI that stops propagation must not wedge the demo
    // active.
    let active_for_ptr = active.clone();
    let deactivate_for_ptr = deactivate.clone();
    let on_pointerdown = Closure::wrap(Box::new(move |event: web_sys::Event| {
        if !active_for_ptr.get() {
            return;
        }
        let inside = event
            .target()
            .and_then(|t| t.dyn_into::<web_sys::Node>().ok())
            .map(|node| host_el.contains(Some(&node)))
            .unwrap_or(false);
        if !inside {
            deactivate_for_ptr();
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    document
        .add_event_listener_with_callback_and_bool(
            "pointerdown",
            on_pointerdown.as_ref().unchecked_ref(),
            true,
        )
        .map_err(|e| anyhow!("addEventListener('pointerdown'): {e:?}"))?;
    on_pointerdown.forget();

    Ok(())
}

/// Install the DOM event listeners that forward main-thread events to the
/// worker. Each posts a typed `{kind: "...", ...}` object that
/// [`super::messages::parse_non_init`] turns into an `InputMessage`.
///
/// `window`: resize, focus, blur, visibilitychange. `canvas`: mousemove
/// (rAF-coalesced), mousedown, mouseup, wheel. `document`: keydown, keyup
/// (on document so keys arrive regardless of focus, game convention). All
/// leak via `forget()` for the page's lifetime.
fn install_dom_input_forwarders(worker: &Worker, canvas: &HtmlCanvasElement) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;

    // Resize is DEBOUNCED, not rate-limited: each event would trigger a
    // worker-side `surface.configure()` (recreates swap chain + scene
    // texture + composite bind group), which mid-drag freezes and crashes
    // the tab. Wait ~100ms of quiet before committing one resize; during
    // the drag CSS scales the old backing store, so no GPU work happens.
    // 6 frames at 60Hz coalesces a sustained drag yet still feels prompt.
    const RESIZE_DEBOUNCE_FRAMES: u32 = 6;
    {
        // `Some((w, h, frames_since_last_event))` while in flight, `None`
        // when settled; each new event resets the counter to 0.
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

        // rAF tick: advance the countdown, commit when idle past threshold.
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
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(|e| anyhow!("resize rAF init: {e:?}"))?;
            }
        }
        Box::leak(Box::new(raf_cb));
    }

    // Mouse move, rAF-COALESCED: a DOM mouse-move fires hundreds of times
    // per second, and one JS-object alloc + postMessage each overwhelms the
    // heap and crashes the tab under a sustained drag. The listener stores
    // the latest position; a rAF loop posts one message per browser-frame.
    {
        // Latest (x, y, buttons) plus the summed `movementX/Y` since the
        // last tick: absolute position drives the egui cursor, the summed
        // delta drives FPS mouse-look so sub-frame motion isn't lost.
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
            // Suppress context menu (right) and autoscroll (middle); leave
            // left alone for demos that pointer-lock on left-click.
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

    // Block the context menu so right-click reaches the app. Separate from
    // `mousedown` because `contextmenu` also fires for keyboard triggers
    // (Shift+F10, the menu key) that `mousedown` never sees.
    {
        let cb = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            ev.prevent_default();
        }) as Box<dyn FnMut(web_sys::Event)>);
        canvas
            .add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("contextmenu listener: {e:?}"))?;
        cb.forget();
    }

    // Wheel: normalize WheelEvent's per-mode delta (pixels/lines/pages) to
    // lines, the loam-input convention.
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

    // Keyboard on document so keys reach us regardless of focus (game
    // convention). `preventDefault()` is selective: the browser keeps
    // reload / devtools / tab management / fullscreen / modifier combos as
    // an escape hatch; the demo claims Tab (else focus walks off-canvas),
    // Space + arrows (scroll / button activation), slash + quote (Firefox
    // quick-find), and Alt (lone-Alt menu-bar activation, both edges).
    // Letter keys have no page-level default, so WASD / T need no handling.
    for (event_name, pressed) in [("keydown", true), ("keyup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::KeyboardEvent| {
            let code = ev.code();
            let no_modifier = !ev.ctrl_key() && !ev.alt_key() && !ev.meta_key();
            // Alt-as-key falls outside `no_modifier` (ev.alt_key is set
            // while Alt is down), so check it separately; skip under
            // Ctrl/Cmd so Ctrl+Alt / Cmd+Alt combos still work.
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

    // Focus / blur on window for loam-input's release-on-focus-loss.
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

    // Page-visibility on document so the app can pause when backgrounded.
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

// Helpers for constructing typed postMessage payloads on the main thread.

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

/// Listen for `{kind: "preview_progress", pct}` and drive the
/// `#loam-page-loader-fill` width plus the track's `aria-valuenow` (so a
/// screen reader announces progress). Bar markup lives in
/// `crates/loam-app/static/page_loader.html`.
fn install_preview_progress_handler(worker: &Worker) -> Result<()> {
    let cb = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() != Some("preview_progress") {
            return;
        }
        let pct = js_sys::Reflect::get(&data, &JsValue::from_str("pct"))
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let percent = pct * 100.0;
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        if let Some(fill) = document.get_element_by_id("loam-page-loader-fill") {
            let _ = fill.dyn_into::<web_sys::HtmlElement>().map(|el| {
                let _ = el.style().set_property("width", &format!("{percent}%"));
            });
        }
        if let Some(track) = document
            .query_selector("#loam-page-loader .loam-progress-track")
            .ok()
            .flatten()
        {
            let _ = track.set_attribute("aria-valuenow", &format!("{}", percent.round() as i32));
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", cb.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message') for preview_progress: {e:?}"))?;
    cb.forget();
    Ok(())
}

/// Listen for the worker's once-only `{kind: "preview_ready"}`, remove the
/// `#loam-page-loader` bar, and add `.ready` to the launch overlay so the
/// click affordance appears only after the preview frame has rendered.
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
        // Remove the static spinner; the wasm overlay takes over.
        if let Some(loader) = document.get_element_by_id("loam-page-loader") {
            loader.remove();
        }
        if let Some(overlay) = document.get_element_by_id(&button_id_owned) {
            overlay.set_class_name("loam-demo-launch ready");
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", cb.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message') for preview_ready: {e:?}"))?;
    cb.forget();
    Ok(())
}

/// Install the worker -> main DOM-action handler, the `pointerlockchange`
/// forwarder, and the canvas click-to-re-engage helper.
///
/// The worker posts `{kind: "host_action", actions: [...]}` whenever an
/// engine module queues a DOM-touching action; everything needing DOM
/// access round-trips here because the worker holds only the
/// `OffscreenCanvas`, no `document` / `window` / canvas element.
///
/// `requestPointerLock` needs transient activation (~5 sec from the user's
/// last key/click); the key-event -> worker -> request round-trip stays
/// inside that window. `exitPointerLock` is always allowed, and browser
/// auto-releases (Esc, tab switch) produce a `pointerlockchange` we forward
/// back so cursor state stays accurate. When the demo still wants lock
/// (`want_locked`) after an auto-release, a canvas click re-requests it.
fn install_host_action_handler(worker: &Worker, canvas: &HtmlCanvasElement) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;

    // Demo's desired lock state, as last requested by the worker. Drives
    // the canvas-click re-engagement: re-request only when want_locked but
    // the actual lock dropped.
    let want_locked: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));

    // Worker -> main: drain a `host_action` message into DOM calls.
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
                        // Resolves async; the `pointerlockchange` listener
                        // relays success or failure back to the worker.
                        canvas_for_dispatch.request_pointer_lock();
                    }
                    Some("pointer_lock_release") => {
                        *want_locked_for_dispatch.borrow_mut() = false;
                        document_for_dispatch.exit_pointer_lock();
                    }
                    Some(other) => {
                        tracing::warn!(
                            "loam_app::wasm: unknown host_action kind '{other}'; dropping"
                        );
                    }
                    None => {
                        tracing::warn!("loam_app::wasm: host_action item missing 'kind'");
                    }
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        worker
            .add_event_listener_with_callback("message", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("worker.addEventListener('message') for host_action: {e:?}"))?;
        cb.forget();
    }

    // `pointerlockchange`: the browser's true lock state, whoever triggered
    // it. Forward to the worker so `cursor::mark_applied` reflects reality.
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

    // Canvas click: re-engage Pointer Lock if want_locked but the browser
    // dropped it. No-op otherwise.
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
