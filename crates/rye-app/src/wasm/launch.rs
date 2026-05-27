//! Wasm32-only helpers for the browser embedding lifecycle. Demos opt into
//! click-to-start (so the page doesn't spin up a wgpu device + render loop until the
//! user actually wants the demo) by marking the host element in `index.html`:
//!
//! ```html
//! <div id="rye-canvas-host" data-mode="manual">
//!   <canvas id="rye-canvas" width="1280" height="800"></canvas>
//! </div>
//! ```
//!
//! [`inject_launch_overlay`] runs from the engine on startup and creates the
//! `#rye-launch` button as a sibling of the canvas, along with a `<style>`
//! element in `<head>` carrying [`LAUNCH_OVERLAY_CSS`]. Demos no longer ship
//! the button markup or CSS themselves; one line of HTML wires up the entire
//! click-to-start container.
//!
//! The wasm module still loads on page navigation (the bytes are downloaded and
//! `init()` runs), but `run_with_config` doesn't fire until the click event. The
//! browser pays the wasm download cost regardless of click-to-start; what it AVOIDS
//! is the per-frame wgpu work + the visible canvas eating GPU + RAF cycles.
//!
//! ## Multi-demo per page
//!
//! Each demo has its own host element id; the injected `<style>` element is
//! keyed on a fixed id so multiple demos on one page only insert the CSS once.
//! v1 ships single-demo embedding only (one wasm bundle per page); multi-demo
//! lands once the per-bundle JS surface is decided.
//!
//! ## Theming
//!
//! Demos that want a custom look can either ship a stylesheet that comes later
//! in the cascade and overrides the engine's `.rye-demo-launch` rules, or pre-
//! create a `<button id="...">` with their own classes inside the host element;
//! [`inject_launch_overlay`] reuses an existing element rather than duplicating
//! it when one is already in the DOM.
//!
//! ## Non-goals
//!
//! - **Pause / resume on scroll-out.** When a demo scrolls out of view in a
//!   blog embed, the ideal behavior is to pause its RAF loop and reclaim GPU
//!   resources. That belongs in the JS embed wrapper (an `IntersectionObserver`
//!   that posts a `pause` message to the worker), not in this engine path; the
//!   worker already has the lifecycle hooks a wrapper would drive.

use anyhow::{anyhow, Context, Result};
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlButtonElement, HtmlStyleElement};

/// Default CSS for the click-to-start overlay. Injected by
/// [`inject_launch_overlay`] into `<head>` once per page; demos that want
/// to theme the overlay can ship a stylesheet that comes later in the
/// cascade (or use higher-specificity selectors) to override. The blur
/// reads the canvas underneath, so the worker's preview frame appears as
/// a softened thumbnail until the viewer clicks.
const LAUNCH_OVERLAY_CSS: &str = r#"
/* Base: shared chrome (positioning, blur, font, transitions). The
   overlay is injected with no state class, so the chip is hidden and
   only the blurred backdrop shows. The worker's `preview_ready`
   message promotes it to `.ready` once the blurred preview frame is
   on the canvas AND pipelines are warm, which reveals the click
   affordance; clicking then removes the overlay entirely. The
   pre-`.ready` "something's happening" visual is the static
   `#rye-page-loader` progress bar, not this overlay. */
.rye-demo-launch {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: 14px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #d8d8df;
    background: rgba(14, 14, 18, 0.35);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: none;
    transition: background 200ms ease, opacity 200ms ease;
}

/* Default (no state class): chip hidden. The `.ready` class opts in. */
.rye-demo-launch::after {
    display: none;
}

/* Ready: preview frame is behind the blur AND warmup is complete,
   click affordance live. Clicking removes the overlay immediately
   and starts the RAF loop -- no second loading state because the
   worker pre-warmed pipelines before getting here. */
.rye-demo-launch.ready {
    cursor: pointer;
}
.rye-demo-launch.ready::after {
    display: inline-block;
    content: 'Click anywhere to launch';
    padding: 14px 28px;
    border: 1px solid rgba(200, 200, 220, 0.5);
    border-radius: 6px;
    background: rgba(20, 20, 28, 0.55);
}
.rye-demo-launch.ready:hover {
    background: rgba(14, 14, 18, 0.25);
}
.rye-demo-launch.ready:hover::after {
    background: rgba(28, 28, 38, 0.65);
    border-color: rgba(220, 220, 240, 0.7);
}
.rye-demo-launch.ready:active {
    background: rgba(14, 14, 18, 0.4);
}
"#;

const OVERLAY_STYLE_ID: &str = "rye-launch-overlay-styles";

/// Inject a launch-overlay `<button>` as a child of `host_id` plus a
/// `<style>` element carrying [`LAUNCH_OVERLAY_CSS`] into `<head>`. The
/// style element is idempotent (keyed on a fixed id), so calling this
/// for multiple demos on one page only inserts the CSS once.
///
/// Returns the freshly-created button so the caller can wire its click
/// handler. If an element with `button_id` already exists in the DOM
/// (because the demo's `index.html` ships a static button for legacy
/// reasons or for theming reasons), this function reuses it instead of
/// creating a duplicate.
pub fn inject_launch_overlay(host_id: &str, button_id: &str) -> Result<HtmlButtonElement> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;
    let host = document
        .get_element_by_id(host_id)
        .ok_or_else(|| anyhow!("no host element with id '{host_id}'"))?;

    if document.get_element_by_id(OVERLAY_STYLE_ID).is_none() {
        let head = document
            .head()
            .ok_or_else(|| anyhow!("no <head> element"))?;
        let style = document
            .create_element("style")
            .map_err(|e| anyhow!("create <style>: {e:?}"))?
            .dyn_into::<HtmlStyleElement>()
            .map_err(|_| anyhow!("created element is not HtmlStyleElement"))?;
        style.set_id(OVERLAY_STYLE_ID);
        style.set_text_content(Some(LAUNCH_OVERLAY_CSS));
        head.append_child(&style)
            .map_err(|e| anyhow!("append <style>: {e:?}"))?;
    }

    if let Some(existing) = document.get_element_by_id(button_id) {
        return existing
            .dyn_into::<HtmlButtonElement>()
            .map_err(|_| anyhow!("element '{button_id}' is not a button"));
    }

    let button = document
        .create_element("button")
        .map_err(|e| anyhow!("create <button>: {e:?}"))?
        .dyn_into::<HtmlButtonElement>()
        .map_err(|_| anyhow!("created element is not HtmlButtonElement"))?;
    button.set_id(button_id);
    // Starts with no state class -> CSS defaults hide the chip + spinner
    // (only the blurred background + base layout apply). The static
    // `#rye-page-loader` element in the demo's `index.html` carries the
    // visible spinner from page load until the worker posts
    // `preview_ready`; at that point the static loader is removed and
    // this overlay gets the `.ready` class to show the click affordance.
    // Click then removes the overlay entirely.
    button.set_class_name("rye-demo-launch");
    button.set_type("button");
    button
        .set_attribute("aria-label", "Launch demo")
        .map_err(|e| anyhow!("set aria-label: {e:?}"))?;
    host.append_child(&button)
        .map_err(|e| anyhow!("append button to host: {e:?}"))?;
    Ok(button)
}

/// Returns true if the page opted into click-to-start by setting
/// `data-mode="manual"` on the host element with the given id. Returns false on
/// missing element, missing attribute, or any other value (default = auto-launch).
pub fn is_manual_mode(host_id: &str) -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let Some(document) = window.document() else {
        return false;
    };
    let Some(el) = document.get_element_by_id(host_id) else {
        return false;
    };
    el.get_attribute("data-mode")
        .map(|m| m == "manual")
        .unwrap_or(false)
}

/// Attach a one-shot click handler to the button with the given id. When clicked,
/// the button removes itself from the DOM (so a frantic double-click can't fire the
/// closure twice) and then invokes `on_click`. Returns Ok(()) as soon as the
/// listener is wired; the actual click might happen seconds or minutes later.
///
/// The closure is a `FnOnce` because launching the app is a one-time operation. The
/// `Closure::once` wrapper ensures the JS side runs the callback exactly once.
pub fn wait_for_launch(button_id: &str, on_click: impl FnOnce() + 'static) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;
    let button = document
        .get_element_by_id(button_id)
        .ok_or_else(|| anyhow!("no element with id '{button_id}'"))?;
    let button_for_click = button.clone();

    let cb = Closure::once(Box::new(move || {
        // Remove the button before invoking the closure so any RAF / event loop the
        // closure spins up doesn't have to fight with the DOM node.
        button_for_click.remove();
        on_click();
    }) as Box<dyn FnOnce()>);

    button
        .add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("add_event_listener: {e:?}"))
        .context("wait_for_launch: attach click listener")?;
    // The closure must outlive its registration; `forget` leaks it intentionally,
    // which is fine because click-to-start happens at most once per page load.
    cb.forget();
    Ok(())
}

/// Sample the V8 JS heap size in bytes, or `None` if the runtime doesn't expose
/// `performance.memory.usedJSHeapSize`. Chromium browsers (Chrome / Edge) expose
/// it as a non-standard extension; Firefox + Safari do not, and there the
/// returned value is `None` so the caller can fall back to "no heap signal."
///
/// Intended to be registered into `rye_time::frame_trace::set_heap_sampler` at
/// startup so each frame gets a heap delta attached for spike correlation. The
/// underlying property is bucketed by V8 to a ~25-100ms resolution; this is
/// fine for catching multi-MB jumps that correlate with major GC pauses, less
/// fine for spotting single-object allocations. Don't read short-term changes
/// here as authoritative.
///
/// Architectural note: `js_sys::Reflect` is the right tool because `web-sys`
/// doesn't surface `Performance::memory` (it's not in the standard). Reflect
/// gracefully degrades to `None` on Firefox via the `is_undefined` check.
pub fn js_heap_sampler() -> Option<u64> {
    let window = web_sys::window()?;
    let performance = window.performance()?;
    let perf_val: &JsValue = performance.as_ref();
    let memory = js_sys::Reflect::get(perf_val, &JsValue::from_str("memory")).ok()?;
    if memory.is_undefined() || memory.is_null() {
        return None;
    }
    let used = js_sys::Reflect::get(&memory, &JsValue::from_str("usedJSHeapSize")).ok()?;
    let bytes = used.as_f64()?;
    if bytes.is_finite() && bytes >= 0.0 {
        Some(bytes as u64)
    } else {
        None
    }
}
