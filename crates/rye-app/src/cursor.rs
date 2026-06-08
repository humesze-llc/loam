//! Cursor grab + visibility request channel between the App and the Runner.
//!
//! Same shape as [`crate::frame_pacing::request_vsync_on`]: process-global
//! atomics the App pokes from anywhere, applied once per redraw by the
//! Runner. `Window::set_cursor_grab` must run on the main thread, which the
//! App (especially a console command) doesn't always reach directly.
//!
//! Grab mode and visibility are independent. [`request_grab_mode`] picks
//! confinement ([`GrabMode`]); [`request_cursor_visible`] picks rendering.
//! Both default to released + visible. [`request_grab`] / [`request_release`]
//! wrap the common mouse-look pair; [`current_state`] reads what the runner
//! last applied.
//!
//! ## Wasm note
//!
//! The channel routes through the worker -> main DOM plumbing in
//! `wasm::host_action` (plain reference, not an intra-doc link: that module
//! is `#[cfg(target_arch = "wasm32")]`). The worker drains [`take_pending`],
//! posts a Pointer Lock request/release, and the main thread relays
//! `pointerlockchange` back as `PointerLockChanged`, which calls
//! [`mark_applied`].
//!
//! Pointer Lock requires transient activation; the console keystroke's
//! ~5-second window survives the sub-millisecond worker round-trip. Esc
//! release (browser-hardcoded) surfaces as `PointerLockChanged(false)`; a
//! canvas click re-engages. [`GrabMode::Confined`] has no browser
//! equivalent and maps to Pointer Lock. Visibility is implicit on wasm:
//! lock auto-hides, release auto-shows.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Cursor confinement modes. Mirrors `winit::window::CursorGrabMode` so
/// the engine API doesn't leak winit into demo code.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GrabMode {
    /// Cursor roams freely across the OS desktop. Default UX.
    #[default]
    None,
    /// Confined to the window client rect, otherwise normal. Pairs with
    /// `visible = true` for modal UIs.
    Confined,
    /// Pinned at window center, motion reported as raw device delta
    /// (`FrameInput::mouse_raw_delta`). Pairs with `visible = false` for
    /// FPS mouse-look.
    Locked,
}

/// Snapshot of the last cursor state the runner applied. Read via
/// [`current_state`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CursorState {
    pub grab: GrabMode,
    pub visible: bool,
}

impl CursorState {
    /// Runner default before any request lands: released + visible.
    pub const RELEASED: Self = Self {
        grab: GrabMode::None,
        visible: true,
    };
}

// Encoded pending requests; `0` = none. Runner swaps to 0 after reading so
// unchanged state isn't re-applied each redraw.
const NONE: u8 = 0;
const GRAB_NONE: u8 = 1;
const GRAB_CONFINED: u8 = 2;
const GRAB_LOCKED: u8 = 3;
static PENDING_GRAB: AtomicU8 = AtomicU8::new(NONE);

const VIS_NONE: u8 = 0;
const VIS_SHOW: u8 = 1;
const VIS_HIDE: u8 = 2;
static PENDING_VISIBLE: AtomicU8 = AtomicU8::new(VIS_NONE);

// Warp-to-center flag, paired with grab release so the OS-cached cursor
// doesn't reappear at a stale spot on un-grab.
static PENDING_WARP_CENTER: AtomicBool = AtomicBool::new(false);

// Last-applied state, committed by the runner; read by `current_state`.
static APPLIED_GRAB: AtomicU8 = AtomicU8::new(GRAB_NONE);
static APPLIED_VISIBLE: AtomicBool = AtomicBool::new(true);

fn grab_mode_to_code(mode: GrabMode) -> u8 {
    match mode {
        GrabMode::None => GRAB_NONE,
        GrabMode::Confined => GRAB_CONFINED,
        GrabMode::Locked => GRAB_LOCKED,
    }
}

fn code_to_grab_mode(code: u8) -> GrabMode {
    match code {
        GRAB_CONFINED => GrabMode::Confined,
        GRAB_LOCKED => GrabMode::Locked,
        _ => GrabMode::None,
    }
}

/// Request the runner change the grab mode on its next redraw.
pub fn request_grab_mode(mode: GrabMode) {
    PENDING_GRAB.store(grab_mode_to_code(mode), Ordering::Release);
}

/// Request the runner show or hide the cursor on its next redraw.
pub fn request_cursor_visible(visible: bool) {
    PENDING_VISIBLE.store(if visible { VIS_SHOW } else { VIS_HIDE }, Ordering::Release);
}

/// Request the runner warp the OS cursor to window center on its next
/// redraw. Pair with grab release so the cursor reappears where the user
/// was aiming, not at the OS-cached position. No-op on wasm (browser owns
/// cursor positioning).
pub fn request_warp_to_center() {
    PENDING_WARP_CENTER.store(true, Ordering::Release);
}

/// Convenience: request the FPS-mouse-look pair (Locked + hidden).
pub fn request_grab() {
    request_grab_mode(GrabMode::Locked);
    request_cursor_visible(false);
}

/// Convenience: request the conventional UI pair (None + visible).
pub fn request_release() {
    request_grab_mode(GrabMode::None);
    request_cursor_visible(true);
}

/// Read + clear the pending grab/visibility transition. Runner-internal;
/// each element is `Some(new_value)` if a request landed, else `None`.
/// Demos read [`current_state`] instead.
#[doc(hidden)]
pub fn take_pending() -> (Option<GrabMode>, Option<bool>) {
    let grab_code = PENDING_GRAB.swap(NONE, Ordering::AcqRel);
    let grab = (grab_code != NONE).then(|| code_to_grab_mode(grab_code));
    let vis_code = PENDING_VISIBLE.swap(VIS_NONE, Ordering::AcqRel);
    let visible = match vis_code {
        VIS_SHOW => Some(true),
        VIS_HIDE => Some(false),
        _ => None,
    };
    (grab, visible)
}

/// Read + clear the pending warp-to-center flag. Runner-internal.
#[doc(hidden)]
pub fn take_pending_warp_center() -> bool {
    PENDING_WARP_CENTER.swap(false, Ordering::AcqRel)
}

/// Record what the runner just applied, so [`current_state`] reads it.
/// Runner-internal.
#[doc(hidden)]
pub fn mark_applied(grab: GrabMode, visible: bool) {
    APPLIED_GRAB.store(grab_mode_to_code(grab), Ordering::Release);
    APPLIED_VISIBLE.store(visible, Ordering::Release);
}

/// Last-applied cursor state. Read from anywhere instead of mirroring a
/// copy of the grab flag.
pub fn current_state() -> CursorState {
    CursorState {
        grab: code_to_grab_mode(APPLIED_GRAB.load(Ordering::Acquire)),
        visible: APPLIED_VISIBLE.load(Ordering::Acquire),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_grab_mode() {
        let _ = take_pending();
        request_grab_mode(GrabMode::Locked);
        let (grab, _) = take_pending();
        assert_eq!(grab, Some(GrabMode::Locked));
        let (grab, _) = take_pending();
        assert_eq!(grab, None, "second read clears");
        request_grab_mode(GrabMode::Confined);
        let (grab, _) = take_pending();
        assert_eq!(grab, Some(GrabMode::Confined));
    }

    #[test]
    fn round_trip_visibility() {
        let _ = take_pending();
        request_cursor_visible(false);
        let (_, vis) = take_pending();
        assert_eq!(vis, Some(false));
        request_cursor_visible(true);
        let (_, vis) = take_pending();
        assert_eq!(vis, Some(true));
    }

    #[test]
    fn convenience_pairs_set_both() {
        let _ = take_pending();
        request_grab();
        let (g, v) = take_pending();
        assert_eq!(g, Some(GrabMode::Locked));
        assert_eq!(v, Some(false));

        request_release();
        let (g, v) = take_pending();
        assert_eq!(g, Some(GrabMode::None));
        assert_eq!(v, Some(true));
    }

    #[test]
    fn current_state_reads_applied_value() {
        mark_applied(GrabMode::Confined, false);
        let s = current_state();
        assert_eq!(s.grab, GrabMode::Confined);
        assert!(!s.visible);
        // Restore default for sibling tests.
        mark_applied(GrabMode::None, true);
    }

    // Warp-to-center is swap-on-read: set once, consumed once, then clears,
    // so the runner doesn't re-warp every frame.
    #[test]
    fn warp_to_center_is_one_shot() {
        let _ = take_pending_warp_center();

        assert!(!take_pending_warp_center());

        request_warp_to_center();
        assert!(take_pending_warp_center(), "first read after request");
        assert!(
            !take_pending_warp_center(),
            "second read should clear the flag"
        );

        // Two requests before a read coalesce to one event.
        request_warp_to_center();
        request_warp_to_center();
        assert!(take_pending_warp_center());
        assert!(!take_pending_warp_center());
    }
}
