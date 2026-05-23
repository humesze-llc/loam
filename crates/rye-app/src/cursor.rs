//! Cursor grab + visibility request channel between the App and the Runner.
//!
//! Same shape as [`crate::frame_pacing::request_vsync_on`]: process-global
//! atomic, the App pokes it from anywhere it has access (console handler,
//! `App::update`, `App::on_event`), the Runner reads + applies it once per
//! redraw via `Window::set_cursor_grab` + `set_cursor_visible`.
//!
//! ## Why a request channel
//!
//! `winit::Window::set_cursor_grab` must run on the main thread (the only
//! thread holding the `Window`). The App, especially when a console command
//! drives the request, doesn't always have direct `Window` access. Funnelling
//! through this atomic keeps the call site simple and the application of the
//! request scoped to the Runner.
//!
//! ## What "grab" means here
//!
//! Encodes both the cursor-grab mode and visibility into a single boolean:
//!
//! - `true`: confine the cursor to the window AND hide it. Pairs with raw
//!   mouse delta consumption (`FrameInput::mouse_raw_delta`) for FPS-style
//!   mouse-look: the cursor stops appearing on screen, deltas keep arriving
//!   regardless of where the user's hand moves the mouse.
//! - `false`: release the cursor AND make it visible. Default UX.
//!
//! ## Wasm note
//!
//! Browser Pointer Lock API requires "transient activation" (a recent user
//! gesture) before `element.requestPointerLock()` succeeds. Console commands
//! issued via keystroke into the in-canvas console do NOT count as the
//! gesture from the browser's perspective; the keystroke is captured by the
//! worker and the lock request races without an activated context.
//!
//! Net effect on wasm: cursor-grab requests will silently fail unless the
//! request is triggered by a direct click handler. The runner's wasm path
//! does not currently call Pointer Lock at all for this reason; demos that
//! need freecam-on-wasm will need a click-to-engage UX layered on top. See
//! `FLATLAND_DEMO.md` and the runner's wasm module for the architectural
//! seam this would fit in.

use std::sync::atomic::{AtomicU8, Ordering};

// Encoded request state. `0` = no pending request, `1` = request grab=true,
// `2` = request grab=false. Runner swaps to 0 after reading so subsequent
// frames don't re-apply unchanged state every redraw.
const NONE: u8 = 0;
const REQ_GRAB: u8 = 1;
const REQ_RELEASE: u8 = 2;
static PENDING: AtomicU8 = AtomicU8::new(NONE);

/// Request the runner grab and hide the cursor on its next redraw.
pub fn request_grab() {
    PENDING.store(REQ_GRAB, Ordering::Release);
}

/// Request the runner release and show the cursor on its next redraw.
pub fn request_release() {
    PENDING.store(REQ_RELEASE, Ordering::Release);
}

/// Read + clear the pending cursor request. Runner calls this once per
/// redraw before applying any change to the window. `Some(true)` =
/// grab+hide; `Some(false)` = release+show; `None` = no change.
pub fn take_pending() -> Option<bool> {
    match PENDING.swap(NONE, Ordering::AcqRel) {
        REQ_GRAB => Some(true),
        REQ_RELEASE => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        // Clear any leftover state from sibling tests.
        let _ = take_pending();
        request_grab();
        assert_eq!(take_pending(), Some(true));
        assert_eq!(take_pending(), None, "second read clears");
        request_release();
        assert_eq!(take_pending(), Some(false));
    }
}
