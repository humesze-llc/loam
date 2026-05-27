//! Worker → main-thread channel for DOM-touching actions.
//!
//! ## Why this module exists
//!
//! The OffscreenCanvas-in-Worker architecture trades DOM access for GC
//! isolation. The worker can't touch `document`, the canvas DOM element,
//! `navigator`, or anything else that synchronously reads or mutates the
//! page; those calls panic with a `ReferenceError`. Every browser API
//! that requires DOM access -- Pointer Lock, Fullscreen, the Clipboard,
//! Wake Lock, Audio focus resumption, the File System Access pickers --
//! has to round-trip to the main thread.
//!
//! This module is the engine-side primitive for that round-trip. Demos
//! call high-level APIs (`rye_app::cursor::request_grab`,
//! `rye_app::fullscreen::request_enter`, etc.); those APIs queue a
//! [`HostAction`]; the worker's frame loop drains pending actions once
//! per frame and posts them to the main thread via
//! [`post_pending_actions`]. The main thread's listener in
//! `main_launcher::install_host_action_handler` dispatches each one to
//! the matching DOM call and (for stateful actions) listens for the
//! browser's state-change event to ping the worker back.
//!
//! ## Why not synchronous RPC
//!
//! `Atomics.wait` + `SharedArrayBuffer` could give the worker a blocking
//! call into the main thread, but it would stall the worker's RAF loop
//! while the main thread schedules the DOM call. The browser also
//! gates `SharedArrayBuffer` behind COOP/COEP headers that complicate
//! self-hosted demos. Async postMessage trades a millisecond of latency
//! (well within Pointer Lock's transient-activation window) for never
//! blocking either thread.
//!
//! ## Cadence
//!
//! Actions are batched per frame: the worker drains its pending list at
//! the end of each frame (after `App::update`/`App::record`), constructs
//! one `{kind: "host_action", actions: [...]}` message, and posts it.
//! Empty drains skip the post. Per-frame batching means a console
//! command that flips cursor grab + visibility + fullscreen all in one
//! tick produces one postMessage, not three.
//!
//! ## Extending
//!
//! Adding a new DOM-action capability is:
//! 1. New variant on [`HostAction`].
//! 2. Drain-and-encode case in [`encode_actions`].
//! 3. Main thread `host_action` dispatch arm in `main_launcher`.
//! 4. If the action has observable state (Pointer Lock did/didn't
//!    engage; Fullscreen did/didn't accept), wire the corresponding DOM
//!    event listener on main thread to ping back via the existing
//!    inbound `InputMessage` channel. Define a new `InputMessage`
//!    variant if needed.
//!
//! No coupling between actions; each one's plumbing is local.

use std::cell::RefCell;

use wasm_bindgen::JsValue;

/// One worker → main DOM action. Variants name *intent*; the main
/// thread translates each to the matching DOM call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAction {
    /// Engage Pointer Lock on the canvas. Locks the cursor to the
    /// canvas center and reports raw motion deltas via
    /// `MouseEvent.movementX/Y`. The main thread calls
    /// `canvas.requestPointerLock()`; the actual transition fires a
    /// `pointerlockchange` event the main thread relays back via
    /// `InputMessage::PointerLockChanged(true)`.
    PointerLockRequest,
    /// Release Pointer Lock. Main thread calls
    /// `document.exitPointerLock()`. Browser also auto-releases on Esc
    /// or tab switch; either path produces the same
    /// `pointerlockchange` round-trip.
    PointerLockRelease,
}

thread_local! {
    /// Per-worker pending action queue. Drained at the end of each
    /// frame by [`post_pending_actions`]. A `thread_local` because the
    /// engine APIs that produce these (`rye_app::cursor`,
    /// `rye_app::fullscreen` later) have no direct handle to the worker
    /// runner; they push into this static and the runner reads it.
    static PENDING: RefCell<Vec<HostAction>> = const { RefCell::new(Vec::new()) };
}

/// Queue a host action for the next frame's drain. Cheap (one push to a
/// per-worker `Vec`). Safe to call from anywhere on the worker thread;
/// engine modules wrap this in their own typed APIs.
pub fn queue(action: HostAction) {
    PENDING.with(|p| p.borrow_mut().push(action));
}

/// Drain pending actions and post them to the main thread as one
/// `{kind: "host_action", actions: [...]}` message. No-op if the queue
/// is empty. Called once per frame from the worker's frame loop.
///
/// Wrapped in a `Result` even though `post_message` on a worker's
/// global scope rarely fails -- if it does (Firefox during teardown,
/// rate-limited contexts), the caller logs it and we drop the actions
/// for that frame. Re-queuing on failure would risk spinning if the
/// transport is permanently down.
pub fn post_pending_actions(scope: &web_sys::DedicatedWorkerGlobalScope) -> anyhow::Result<()> {
    let actions = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
    if actions.is_empty() {
        return Ok(());
    }
    let msg = encode_actions(&actions)?;
    scope
        .post_message(&msg)
        .map_err(|e| anyhow::anyhow!("post host_action: {e:?}"))
}

/// Build the `{kind: "host_action", actions: [...]}` JS object for a
/// drained action list. Each action is encoded as `{kind: "<variant
/// lowercase>"}` with any per-variant fields tacked on. Kept separate
/// from `post_pending_actions` so unit tests can exercise the encoding
/// without needing a real `DedicatedWorkerGlobalScope`.
fn encode_actions(actions: &[HostAction]) -> anyhow::Result<JsValue> {
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(
        &msg,
        &JsValue::from_str("kind"),
        &JsValue::from_str("host_action"),
    )
    .map_err(|e| anyhow::anyhow!("set kind: {e:?}"))?;
    let arr = js_sys::Array::new();
    for action in actions {
        let item = js_sys::Object::new();
        match action {
            HostAction::PointerLockRequest => {
                js_sys::Reflect::set(
                    &item,
                    &JsValue::from_str("kind"),
                    &JsValue::from_str("pointer_lock_request"),
                )
                .map_err(|e| anyhow::anyhow!("set pointer_lock_request: {e:?}"))?;
            }
            HostAction::PointerLockRelease => {
                js_sys::Reflect::set(
                    &item,
                    &JsValue::from_str("kind"),
                    &JsValue::from_str("pointer_lock_release"),
                )
                .map_err(|e| anyhow::anyhow!("set pointer_lock_release: {e:?}"))?;
            }
        }
        arr.push(&item);
    }
    js_sys::Reflect::set(&msg, &JsValue::from_str("actions"), &arr)
        .map_err(|e| anyhow::anyhow!("set actions: {e:?}"))?;
    Ok(msg.into())
}
