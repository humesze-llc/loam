//! Wasm32-only support code. Subdivided as the worker-mode plumbing grew
//! beyond what a single file could hold.
//!
//! - [`launch`]: click-to-start container + the original page-startup
//!   helpers (`is_manual_mode`, `wait_for_launch`).
//! - [`worker`]: OffscreenCanvas + Web Worker mode for GC-isolated rendering.
//!   The main thread owns the DOM and forwards input via postMessage; the
//!   worker owns wgpu + egui and drives RAF. Avoids main-thread GC pauses and
//!   sidesteps winit's incomplete worker-context support.
//!
//! Detection helpers ([`is_worker_context`], plus the heap sampler that
//! `rye-time::frame_trace` registers) live at the module root so the
//! `rye_app::wasm::*` import path stays flat for the common cases.

pub mod launch;
pub mod main_launcher;
pub mod messages;
pub mod worker;
pub mod worker_ui;

// Re-export the demo-facing entry so callers can write
// `rye_app::wasm::launch_on_click(...)` without descending into the
// submodule. The submodule path is still available for explicit access.
pub use main_launcher::launch_on_click;

// Re-export the click-to-start surface at the wasm module level so existing
// `rye_app::wasm::is_manual_mode(...)` and `rye_app::wasm::wait_for_launch(...)`
// call sites continue to work after the wasm.rs -> wasm/launch.rs move.
pub use launch::{is_manual_mode, js_heap_sampler, wait_for_launch};

/// Returns true when the wasm binary is executing inside a
/// `DedicatedWorkerGlobalScope` (i.e., a Web Worker), false on the main
/// page thread. The same wasm binary serves both contexts; this check
/// lets `main` branch into worker entry vs main-thread launcher.
///
/// Implementation: dyn-cast the global object to `DedicatedWorkerGlobalScope`.
/// In a worker the global is the worker scope; on the main thread it's the
/// Window. The cast succeeds in exactly one case and that's the answer.
pub fn is_worker_context() -> bool {
    use wasm_bindgen::JsCast;
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}
