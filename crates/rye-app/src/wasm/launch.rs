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
//! ## Non-goals (v1)
//!
//! - **Pause / resume on IntersectionObserver.** Coming in v2 once the lifecycle
//!   handle returned by `run_with_config` exposes a way to stop the redraw loop
//!   without tearing down the wgpu device.
//! - **Loading-progress UI.** The current implementation hides the launch button on
//!   click and lets the canvas appear when ready (~the same time as the first
//!   `RenderDevice::new` resolve). Pipeline-warming progress UI is a v2 piece.

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
    cursor: pointer;
    transition: background 200ms ease, opacity 200ms ease;
}
.rye-demo-launch::after {
    content: 'Click anywhere to launch';
    display: inline-block;
    padding: 14px 28px;
    border: 1px solid rgba(200, 200, 220, 0.5);
    border-radius: 6px;
    background: rgba(20, 20, 28, 0.55);
}
.rye-demo-launch:hover {
    background: rgba(14, 14, 18, 0.25);
}
.rye-demo-launch:hover::after {
    background: rgba(28, 28, 38, 0.65);
    border-color: rgba(220, 220, 240, 0.7);
}
.rye-demo-launch:active {
    background: rgba(14, 14, 18, 0.4);
}
/* Loading state: applied to the overlay after the user clicks, before
   the worker has rendered enough frames to be smooth. Cursor stays
   default (not pointer), the click handler short-circuits subsequent
   clicks, and a small spinner appears next to the label. The overlay
   stays opaque (no pointer-events: none yet) so accidental click-spam
   doesn't cascade into the canvas underneath while the demo is still
   warming up. */
.rye-demo-launch.loading {
    cursor: wait;
}
.rye-demo-launch.loading::after {
    content: 'Loading\2026';
    padding-right: 56px;
    background-image: linear-gradient(
        from-left,
        transparent,
        transparent
    );
}
.rye-demo-launch.loading::before {
    content: '';
    position: absolute;
    width: 18px;
    height: 18px;
    /* Same coordinate system as the chip text so the spinner sits
       next to it. CSS pseudo-elements share the parent's flex layout
       only if they're sized; we absolute-position relative to the
       chip's rect via the parent's `display: flex; align-items:
       center` and a small negative offset. */
    right: calc(50% - 110px);
    border: 2px solid rgba(200, 200, 220, 0.25);
    border-top-color: rgba(220, 220, 240, 0.9);
    border-radius: 50%;
    animation: rye-demo-spinner 0.9s linear infinite;
}
@keyframes rye-demo-spinner {
    to { transform: rotate(360deg); }
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
