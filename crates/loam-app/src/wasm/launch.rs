//! Wasm32-only helpers for the browser embedding lifecycle. Demos opt into
//! click-to-start by marking the host element in `index.html`:
//!
//! ```html
//! <div id="loam-canvas-host" data-mode="manual">
//!   <canvas id="loam-canvas" width="1280" height="800"></canvas>
//! </div>
//! ```
//!
//! [`inject_launch_overlay`] creates the launch button as a sibling of the
//! canvas plus a `<style>` carrying [`LAUNCH_OVERLAY_CSS`], so demos ship one
//! line of HTML instead of the button markup + CSS. The wasm download happens
//! regardless; click-to-start only defers the per-frame wgpu work.
//!
//! The `<style>` is keyed on a fixed id so multiple demos insert the CSS once;
//! v1 ships single-demo embedding only. Demos theme by shipping a
//! later-cascade stylesheet or pre-creating their own `<button id="...">`,
//! which [`inject_launch_overlay`] reuses.
//!
//! After launch, the embed lifecycle in `main_launcher` enforces
//! one-active-demo-per-page: clicking outside the host pauses the worker
//! and restores this overlay via [`show_resume_overlay`]. Pause-on-
//! scroll-out remains a JS embed-wrapper concern (an `IntersectionObserver`
//! posting `pause`), which the worker's pause/resume messages support.

use anyhow::{anyhow, Context, Result};
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlButtonElement, HtmlStyleElement};

/// Default CSS for the click-to-start overlay, injected once per page by
/// [`inject_launch_overlay`]. The blur reads the canvas underneath, so the
/// worker's preview frame shows as a softened thumbnail until the click.
const LAUNCH_OVERLAY_CSS: &str = r#"
/* Base: shared chrome (positioning, blur, font, transitions). The
   overlay is injected with no state class, so the chip is hidden and
   only the blurred backdrop shows. The worker's `preview_ready`
   message promotes it to `.ready` once the blurred preview frame is
   on the canvas AND pipelines are warm, which reveals the click
   affordance; clicking then removes the overlay entirely. The
   pre-`.ready` "something's happening" visual is the static
   `#loam-page-loader` progress bar, not this overlay. */
.loam-demo-launch {
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
.loam-demo-launch::after {
    display: none;
}

/* Ready: preview frame is behind the blur AND warmup is complete,
   click affordance live. Clicking removes the overlay immediately
   and starts the RAF loop -- no second loading state because the
   worker pre-warmed pipelines before getting here. */
.loam-demo-launch.ready {
    cursor: pointer;
}
.loam-demo-launch.ready::after {
    display: inline-block;
    content: 'Click anywhere to launch';
    padding: 14px 28px;
    border: 1px solid rgba(200, 200, 220, 0.5);
    border-radius: 6px;
    background: rgba(20, 20, 28, 0.55);
}
.loam-demo-launch.ready:hover {
    background: rgba(14, 14, 18, 0.25);
}
.loam-demo-launch.ready:hover::after {
    background: rgba(28, 28, 38, 0.65);
    border-color: rgba(220, 220, 240, 0.7);
}
.loam-demo-launch.ready:active {
    background: rgba(14, 14, 18, 0.4);
}

/* Paused embed. `.resume` composes with `.ready`: a paused demo is warm. */
.loam-demo-launch.ready.resume::after {
    content: 'Click to start';
}
"#;

const OVERLAY_STYLE_ID: &str = "loam-launch-overlay-styles";

/// Inject a launch-overlay `<button>` under `host_id` plus a `<style>`
/// carrying [`LAUNCH_OVERLAY_CSS`] into `<head>` (the style is idempotent,
/// keyed on a fixed id). Returns the button for the caller to wire; reuses
/// an existing `button_id` element if the demo shipped its own.
///
/// No state class: CSS shows only the blurred backdrop until the worker's
/// `preview_ready` adds `.ready` and reveals the click affordance. The
/// `#loam-page-loader` element carries the spinner until then.
pub fn inject_launch_overlay(host_id: &str, button_id: &str) -> Result<HtmlButtonElement> {
    overlay_button(host_id, button_id, "loam-demo-launch", "Launch demo")
}

/// (Re)create the overlay as the paused-state affordance (`.ready.resume`:
/// immediately clickable, a paused demo is warm). The launch flow removed
/// the original button, so this usually creates a fresh element.
pub fn show_resume_overlay(host_id: &str, button_id: &str) -> Result<HtmlButtonElement> {
    overlay_button(
        host_id,
        button_id,
        "loam-demo-launch ready resume",
        "Resume demo",
    )
}

/// Idempotent style injection plus reuse-or-create of `button_id`. A reused
/// element gets its class list overwritten (launch -> resume transitions).
fn overlay_button(
    host_id: &str,
    button_id: &str,
    class_name: &str,
    aria_label: &str,
) -> Result<HtmlButtonElement> {
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
        let button = existing
            .dyn_into::<HtmlButtonElement>()
            .map_err(|_| anyhow!("element '{button_id}' is not a button"))?;
        button.set_class_name(class_name);
        return Ok(button);
    }

    let button = document
        .create_element("button")
        .map_err(|e| anyhow!("create <button>: {e:?}"))?
        .dyn_into::<HtmlButtonElement>()
        .map_err(|_| anyhow!("created element is not HtmlButtonElement"))?;
    button.set_id(button_id);
    button.set_class_name(class_name);
    button.set_type("button");
    button
        .set_attribute("aria-label", aria_label)
        .map_err(|e| anyhow!("set aria-label: {e:?}"))?;
    host.append_child(&button)
        .map_err(|e| anyhow!("append button to host: {e:?}"))?;
    Ok(button)
}

/// True if the host element has `data-mode="manual"`. False on missing
/// element / attribute / any other value (default = auto-launch).
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

/// Attach a one-shot click handler to the button. On click the button
/// removes itself (so a double-click can't fire twice) and invokes
/// `on_click`. `FnOnce` because launching is one-time.
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
        // Remove the button before the closure so any RAF / event loop it
        // spins up doesn't fight the DOM node.
        button_for_click.remove();
        on_click();
    }) as Box<dyn FnOnce()>);

    button
        .add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("add_event_listener: {e:?}"))
        .context("wait_for_launch: attach click listener")?;
    cb.forget();
    Ok(())
}

/// Sample the V8 JS heap in bytes, or `None` where
/// `performance.memory.usedJSHeapSize` is absent (Chromium exposes it as a
/// non-standard extension; Firefox + Safari do not).
///
/// Registered into `loam_time::frame_trace::set_heap_sampler` for per-frame
/// heap deltas. V8 buckets the value to ~25-100ms, fine for multi-MB GC
/// jumps, not for single allocations. Uses `js_sys::Reflect` because
/// `web-sys` doesn't surface the non-standard `Performance::memory`.
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
