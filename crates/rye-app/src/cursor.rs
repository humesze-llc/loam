//! Cursor grab + visibility request channel between the App and the Runner.
//!
//! Same shape as [`crate::frame_pacing::request_vsync_on`]: process-global
//! atomics; the App pokes them from anywhere it has access (console handler,
//! `App::update`, `App::on_event`); the Runner reads + applies them once per
//! redraw via `Window::set_cursor_grab` + `set_cursor_visible`.
//!
//! ## Why a request channel
//!
//! `winit::Window::set_cursor_grab` must run on the main thread (the only
//! thread holding the `Window`). The App, especially when a console command
//! drives the request, doesn't always have direct `Window` access.
//! Funnelling through these atomics keeps the call site simple and the
//! application of the request scoped to the Runner.
//!
//! ## Surface
//!
//! Grab mode and visibility are independent:
//!
//! - [`request_grab_mode`] picks confinement: `None` lets the cursor roam
//!   freely across the OS desktop; `Confined` keeps it inside the window
//!   but visible; `Locked` pins it to the window center and reports raw
//!   motion (the FPS-mouse-look mode).
//! - [`request_cursor_visible`] picks whether the cursor renders.
//!
//! Both default to "released and visible" (the conventional UI state).
//! Mouse-look apps typically pair `Locked` with `visible = false`; an
//! adventure-game pointer might pair `Confined` with `visible = true`; a
//! cinematic mode might pair `None` with `visible = false`.
//!
//! [`request_grab`] / [`request_release`] are convenience wrappers for the
//! common mouse-look toggle.
//!
//! [`current_state`] returns what the runner last applied so demos don't
//! need to mirror their own copy.
//!
//! ## Wasm note
//!
//! On wasm32 the request channel routes through the worker -> main DOM-
//! action plumbing in `wasm::host_action` (plain reference, not an
//! intra-doc link: that module is `#[cfg(target_arch = "wasm32")]` so
//! it doesn't exist when docs build for the native target). The worker
//! drains [`take_pending`] at the end of each frame, translates the
//! grab mode to a `HostAction::PointerLockRequest` or
//! `HostAction::PointerLockRelease`, and posts to the main thread. The
//! main thread calls `canvas.requestPointerLock()` /
//! `document.exitPointerLock()` and relays the resulting
//! `pointerlockchange` event back to the worker as
//! `InputMessage::PointerLockChanged`, which calls [`mark_applied`] so
//! [`current_state`] stays accurate.
//!
//! Browser Pointer Lock requires "transient activation" (a recent user
//! gesture). The keystroke that produced the console command opens a
//! ~5-second window; the worker -> main round-trip is sub-millisecond,
//! so the activation token is still valid when `requestPointerLock`
//! runs. If the user releases the lock with Esc (a browser-hardcoded
//! shortcut we can't suppress), the demo learns via
//! `PointerLockChanged(false)` and a canvas click re-engages it if the
//! demo last requested grab.
//!
//! The [`GrabMode::Confined`] variant has no direct browser equivalent
//! and is treated as a Pointer Lock request (the closest behavior we
//! can deliver). Visibility requests are implicit on wasm: Pointer Lock
//! auto-hides, release auto-shows.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Cursor confinement modes. Mirrors `winit::window::CursorGrabMode` so
/// the engine API doesn't leak winit into demo code.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GrabMode {
    /// Cursor moves freely across the OS desktop, including outside the
    /// window. Default UX.
    #[default]
    None,
    /// Cursor confined inside the window's client rect but otherwise
    /// moves normally. Pairs with `visible = true` for click discipline
    /// in modal UIs.
    Confined,
    /// Cursor pinned at the window center; movement reported as raw
    /// device delta (`FrameInput::mouse_raw_delta`). Pairs with
    /// `visible = false` for FPS-style mouse-look.
    Locked,
}

/// Snapshot of the last cursor state the runner applied. Read via
/// [`current_state`] so demos don't need to mirror this themselves.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CursorState {
    pub grab: GrabMode,
    pub visible: bool,
}

impl CursorState {
    /// Default the runner uses before any request lands: released +
    /// visible. The conventional UI cursor.
    pub const RELEASED: Self = Self {
        grab: GrabMode::None,
        visible: true,
    };
}

// Encoded pending requests. `0` = no request; nonzero values map to
// specific transitions. Runner swaps to 0 after reading so subsequent
// frames don't re-apply unchanged state on every redraw.
const NONE: u8 = 0;
const GRAB_NONE: u8 = 1;
const GRAB_CONFINED: u8 = 2;
const GRAB_LOCKED: u8 = 3;
static PENDING_GRAB: AtomicU8 = AtomicU8::new(NONE);

const VIS_NONE: u8 = 0;
const VIS_SHOW: u8 = 1;
const VIS_HIDE: u8 = 2;
static PENDING_VISIBLE: AtomicU8 = AtomicU8::new(VIS_NONE);

// "Warp cursor to window center" pending flag. Demos pair this with a grab
// release so the OS-cached cursor position doesn't pop up at a stale spot
// when the user temporarily un-grabs (e.g., Alt in MMO-style cursor mode).
static PENDING_WARP_CENTER: AtomicBool = AtomicBool::new(false);

// Last-applied state. Updated by the runner once it commits a transition
// to the window. Read by [`current_state`].
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

/// Request the runner warp the OS cursor to the window's center on its
/// next redraw. Use this when releasing the grab so the cursor reappears
/// in a predictable spot (typically the same screen position the user
/// was aiming at) instead of the OS-cached random position. No-op on wasm
/// (the browser owns cursor positioning; the warn from `request_grab` covers
/// this).
pub fn request_warp_to_center() {
    PENDING_WARP_CENTER.store(true, Ordering::Release);
}

/// Convenience: request the FPS-mouse-look pair (Locked + hidden). Common
/// enough to be its own call.
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
/// returns `(grab_change, visible_change)` where each is `Some(new_value)`
/// if a request landed since the last call, `None` otherwise.
///
/// `#[doc(hidden)]` because the engine is the only consumer; demos that
/// want the applied state read [`current_state`] instead.
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

/// Last-applied cursor state. Read this from anywhere (demo, console
/// handler, render code) instead of mirroring a copy of the grab flag.
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
        // Restore default so sibling tests aren't surprised.
        mark_applied(GrabMode::None, true);
    }

    /// Warp-to-center is a one-shot flag: `request_warp_to_center` sets it,
    /// `take_pending_warp_center` returns true once and then false until the next
    /// request. Mirrors the grab/visibility channel's "swap-on-read" semantic so the
    /// runner doesn't re-warp every frame if no one re-requests.
    #[test]
    fn warp_to_center_is_one_shot() {
        // Drain anything left over from a sibling test.
        let _ = take_pending_warp_center();

        // No request yet -> false.
        assert!(!take_pending_warp_center());

        // After requesting -> true once, then false on the second read.
        request_warp_to_center();
        assert!(take_pending_warp_center(), "first read after request");
        assert!(
            !take_pending_warp_center(),
            "second read should clear the flag"
        );

        // Two requests before a read coalesce: still one consumed event.
        request_warp_to_center();
        request_warp_to_center();
        assert!(take_pending_warp_center());
        assert!(!take_pending_warp_center());
    }
}
