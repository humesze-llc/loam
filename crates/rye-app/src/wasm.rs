//! Wasm32-only helpers for the browser embedding lifecycle. Demos opt into
//! click-to-start (so the page doesn't spin up a wgpu device + render loop until the
//! user actually wants the demo) by:
//!
//! 1. Marking the host element in `index.html`:
//!
//!    ```html
//!    <div id="rye-canvas-host" data-mode="manual">
//!      <button id="rye-launch" class="rye-demo-launch">
//!        Launch demo
//!      </button>
//!    </div>
//!    ```
//!
//! 2. Branching in `main`:
//!
//!    ```ignore
//!    fn main() -> anyhow::Result<()> {
//!        #[cfg(target_arch = "wasm32")]
//!        if rye_app::wasm::is_manual_mode("rye-canvas-host") {
//!            rye_app::wasm::wait_for_launch("rye-launch", || {
//!                let _ = launch_app();
//!            })?;
//!            return Ok(());
//!        }
//!        launch_app()
//!    }
//!    ```
//!
//! The wasm module still loads on page navigation (the bytes are downloaded and
//! `init()` runs), but `run_with_config` doesn't fire until the click event. The
//! browser pays the wasm download cost regardless of click-to-start; what it AVOIDS
//! is the per-frame wgpu work + the visible canvas eating GPU + RAF cycles.
//!
//! ## Multi-demo per page
//!
//! Each demo has its own host element id. Calling `wait_for_launch` for several
//! demos on the same page is supported as long as the button ids are unique. v1 ships
//! single-demo embedding only (one wasm bundle per page); multi-demo lands once the
//! per-bundle JS surface is decided.
//!
//! ## Non-goals (v1)
//!
//! - **Pause / resume on IntersectionObserver.** Coming in v2 once the lifecycle
//!   handle returned by `run_with_config` exposes a way to stop the redraw loop
//!   without tearing down the wgpu device.
//! - **Loading-progress UI.** The current implementation hides the launch button on
//!   click and lets the canvas appear when ready (~the same time as the first
//!   `RenderDevice::new` resolve). Pipeline-warming progress UI is a v2 piece.
//! - **Custom button styling from Rust.** All styling lives in the page's CSS so
//!   blog embedding can theme the button to match the surrounding content.

use anyhow::{anyhow, Context, Result};
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::JsCast;

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
