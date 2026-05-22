//! Web Worker mode for rye demos. Moves the render loop into a worker so V8's
//! GC pauses don't block the visible page.
//!
//! See `docs/devlog/context/OFFSCREEN_CANVAS_WORKERS.md` for the full
//! architectural design + phasing plan.
//!
//! ## Status
//!
//! Phase A in progress: tesseract render-only in a worker, no input
//! handling yet, validates that wgpu + OffscreenCanvas + a rolled-own RAF
//! loop actually compose. Bypasses winit on the worker path because winit
//! 0.30 doesn't support `WorkerGlobalScope` (issue #1518, open since 2020).
//!
//! ## Two contexts, one binary
//!
//! The same wasm bundle runs on the main thread (the page) AND inside the
//! worker. Detection via [`crate::wasm::is_worker_context`] lets `main`
//! branch into the right entry point:
//!
//! ```ignore
//! fn main() -> Result<()> {
//!     #[cfg(target_arch = "wasm32")]
//!     {
//!         if rye_app::wasm::is_worker_context() {
//!             rye_app::wasm::worker::run::<TesseractApp>()?;
//!             return Ok(());
//!         }
//!         // Main-thread entry: spawn worker on click, forward events.
//!         // (Implemented incrementally; Phase A has a stub.)
//!     }
//!     launch_app() // native or main-thread fallback
//! }
//! ```

use anyhow::Result;

/// Entry point for the worker context. Phase A: stub. Will block on
/// receiving the OffscreenCanvas transfer from main thread, then set up
/// the wgpu Surface + rolled-own RAF loop.
///
/// Generic over the user's `App` so the worker constructs + runs the
/// same app type the main-thread fallback would have constructed.
pub fn run<A>() -> Result<()>
where
    A: crate::App + 'static,
    A::Space: Send + 'static,
{
    // TODO Phase A: install message listener for the OffscreenCanvas
    // Init message, build wgpu Surface from the received canvas, run a
    // minimal RAF loop that drives the app's redraw.
    anyhow::bail!("rye_app::wasm::worker::run: Phase A not yet implemented")
}
