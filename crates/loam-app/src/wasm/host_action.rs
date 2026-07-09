//! Worker -> main-thread channel for DOM-touching actions.
//!
//! The OffscreenCanvas-in-Worker architecture has no DOM access; browser
//! APIs that require it (Pointer Lock, Fullscreen, Clipboard, ...) must
//! round-trip to the main thread. Demos queue a [`HostAction`]; the
//! worker frame loop drains them via [`post_pending_actions`] and posts
//! one batched message that `main_launcher` dispatches to the DOM call.
//!
//! Async postMessage over `Atomics.wait` + `SharedArrayBuffer`: never
//! blocks either thread, and avoids the COOP/COEP header requirement.

use std::cell::RefCell;

use wasm_bindgen::JsValue;

/// One worker -> main DOM action. Variants name intent; the main thread
/// translates each to the matching DOM call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAction {
    /// Engage Pointer Lock on the canvas. The transition is confirmed
    /// asynchronously via `InputMessage::PointerLockChanged(true)`.
    PointerLockRequest,
    /// Release Pointer Lock. The browser also auto-releases on Esc or tab
    /// switch; both paths round-trip back via `PointerLockChanged`.
    PointerLockRelease,
}

impl HostAction {
    /// The `kind` string this action serializes to on the wire. Source of
    /// truth; `main_launcher`'s dispatch matches the same literals.
    const fn kind_str(self) -> &'static str {
        match self {
            HostAction::PointerLockRequest => "pointer_lock_request",
            HostAction::PointerLockRelease => "pointer_lock_release",
        }
    }
}

thread_local! {
    /// Per-worker pending action queue, drained each frame by
    /// [`post_pending_actions`]. `thread_local` because the producing
    /// engine APIs have no direct handle to the worker runner.
    static PENDING: RefCell<Vec<HostAction>> = const { RefCell::new(Vec::new()) };
}

/// Queue a host action for the next frame's drain. Engine modules wrap
/// this in their own typed APIs.
pub fn queue(action: HostAction) {
    PENDING.with(|p| p.borrow_mut().push(action));
}

/// Drain pending actions and post them to the main thread as one
/// `{kind: "host_action", actions: [...]}` message. No-op if empty.
///
/// On `post_message` failure the caller logs and drops the batch; we do
/// not re-queue, to avoid spinning if the transport is permanently down.
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
/// drained action list. Each action encodes as `{kind: <kind_str()>}`.
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
        js_sys::Reflect::set(
            &item,
            &JsValue::from_str("kind"),
            &JsValue::from_str(action.kind_str()),
        )
        .map_err(|e| anyhow::anyhow!("set action kind: {e:?}"))?;
        arr.push(&item);
    }
    js_sys::Reflect::set(&msg, &JsValue::from_str("actions"), &arr)
        .map_err(|e| anyhow::anyhow!("set actions: {e:?}"))?;
    Ok(msg.into())
}
