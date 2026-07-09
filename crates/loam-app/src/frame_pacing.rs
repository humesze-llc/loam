//! Process-global frame pacing state (target framerate, vsync toggle,
//! precise sleep), read by `Runner::redraw` to gate each frame's work.
//!
//! Global atomics rather than `Runner` fields because the `fps` / `vsync`
//! console commands get `&mut Ctx`, not the `Runner`; same pattern as
//! [`loam_time::frame_trace::set_capacity`].
//!
//! ## Native vs. wasm semantics
//!
//! - **Native**: the runner [`precise_sleep_until`]s each redraw's deadline.
//!   `target_fps = 0` removes the cap and the surface `PresentMode` drives
//!   cadence (`Fifo` blocks at vsync); `vsync off` swaps to `Mailbox` /
//!   `Immediate` so the cap can exceed the refresh rate.
//! - **Wasm**: `requestAnimationFrame` is the upper bound; the runner can
//!   only cap *lower* by skipping early RAF callbacks. `vsync` is a no-op
//!   (browser surfaces advertise only `Fifo`).

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Duration;
// `web_time::Instant`: cross-target wall clock (native `std`, wasm
// `performance.now()`). `std::time::Instant::now` panics on wasm32.
use web_time::Instant;

/// Initial period. `0` = uncapped (native uses the surface PresentMode,
/// wasm uses RAF cadence). Defaulting uncapped avoids the prior 60 fps
/// default suppressing the native rate on 120/144/240 Hz displays and
/// introducing RAF alternating-skip jitter; `fps <N>` is the only way in.
const DEFAULT_PERIOD_NS: u64 = 0;

/// Target frame period in ns. `0` = uncapped; the runner neither sleeps nor
/// skips and lets the refresh rate or RAF pace.
static TARGET_PERIOD_NS: AtomicU64 = AtomicU64::new(DEFAULT_PERIOD_NS);

/// Set the target frame period from a desired fps. `fps <= 0.0` removes the cap
/// (uncapped: frames as fast as the surface and browser allow).
pub fn set_target_fps(fps: f32) {
    if fps <= 0.0 {
        TARGET_PERIOD_NS.store(0, Ordering::Release);
        return;
    }
    let period_ns = (1_000_000_000.0 / fps as f64) as u64;
    TARGET_PERIOD_NS.store(period_ns.max(1), Ordering::Release);
}

/// Current target frame period, or `None` if uncapped.
pub fn target_period() -> Option<Duration> {
    let ns = TARGET_PERIOD_NS.load(Ordering::Acquire);
    if ns == 0 {
        None
    } else {
        Some(Duration::from_nanos(ns))
    }
}

/// Current target fps. `0.0` = uncapped.
pub fn target_fps() -> f32 {
    let ns = TARGET_PERIOD_NS.load(Ordering::Acquire);
    if ns == 0 {
        0.0
    } else {
        1_000_000_000.0 / ns as f32
    }
}

// ---------------------------------------------------------------------------
// Vsync request channel
// ---------------------------------------------------------------------------

// Pending vsync request; `0` = none. Runner swaps back to 0 after applying
// so the surface isn't reconfigured every tick.
const VSYNC_NONE: u8 = 0;
const VSYNC_REQ_ON: u8 = 1;
const VSYNC_REQ_OFF: u8 = 2;
static PENDING_VSYNC: AtomicU8 = AtomicU8::new(VSYNC_NONE);

/// Request that the runner switch the surface to vsync-on (`PresentMode::Fifo`)
/// on its next redraw.
pub fn request_vsync_on() {
    PENDING_VSYNC.store(VSYNC_REQ_ON, Ordering::Release);
}

/// Request the runner switch the surface to vsync-off on its next redraw.
/// The runner picks `Mailbox`, else `Immediate`, else leaves `Fifo` (the
/// typical browser case).
pub fn request_vsync_off() {
    PENDING_VSYNC.store(VSYNC_REQ_OFF, Ordering::Release);
}

/// Read + clear the pending vsync transition. `Some(true)` = on,
/// `Some(false)` = off, `None` = no change.
pub fn take_pending_vsync() -> Option<bool> {
    match PENDING_VSYNC.swap(VSYNC_NONE, Ordering::AcqRel) {
        VSYNC_REQ_ON => Some(true),
        VSYNC_REQ_OFF => Some(false),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Precise sleep
// ---------------------------------------------------------------------------

/// Busy-wait tail after the coarse sleep. 2 ms covers the worst-case
/// `std::thread::sleep` overshoot seen on Win11's 15.625 ms timer tick;
/// native-only since wasm skips-and-rerequests instead of sleeping.
#[cfg(not(target_arch = "wasm32"))]
const SPIN_TAIL: Duration = Duration::from_millis(2);

/// Sleep until `deadline` with sub-millisecond precision: coarse-sleep
/// until `SPIN_TAIL` before, then spin. Plain `std::thread::sleep` rounds
/// to the timer granularity (~15.6 ms on Windows), missing sub-vsync caps.
/// The spin saturates one core for <=2 ms/frame (<13% at a 60 fps cap).
#[cfg(not(target_arch = "wasm32"))]
pub fn precise_sleep_until(deadline: Instant) {
    let now = Instant::now();
    if deadline <= now {
        return;
    }
    let total = deadline - now;
    if total > SPIN_TAIL {
        std::thread::sleep(total - SPIN_TAIL);
    }
    // `Instant::now()` is monotonic, so the spin terminates for any valid
    // future deadline.
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

/// Stub on wasm32 (the runner skips-and-rerequests instead); exists so the
/// call site needs no `cfg`.
#[cfg(target_arch = "wasm32")]
pub fn precise_sleep_until(_deadline: Instant) {}

// Shared lock serializing tests that touch the global pacing atomics;
// without it cargo's parallel runner interleaves reads/writes and flakes.
// `unwrap_or_else(|e| e.into_inner())` ignores poison (the data is unit).
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_60fps() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Reset in case a sibling test left the atomic elsewhere.
        set_target_fps(60.0);
        assert!((target_fps() - 60.0).abs() < 0.01);
        assert!(target_period().is_some());
    }

    #[test]
    fn unlimited_round_trip() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_target_fps(0.0);
        assert_eq!(target_fps(), 0.0);
        assert_eq!(target_period(), None);
        set_target_fps(60.0);
    }

    #[test]
    fn set_then_read() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_target_fps(144.0);
        let f = target_fps();
        assert!((f - 144.0).abs() < 0.5);
        set_target_fps(60.0);
    }

    #[test]
    fn vsync_request_round_trip() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_pending_vsync();
        request_vsync_on();
        assert_eq!(take_pending_vsync(), Some(true));
        assert_eq!(take_pending_vsync(), None, "should clear after read");
        request_vsync_off();
        assert_eq!(take_pending_vsync(), Some(false));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn precise_sleep_lands_close_to_deadline() {
        // 50 ms is well above the ~15.625 ms Windows timer tick; the hybrid
        // should wake within ~SPIN_TAIL of the deadline.
        let start = Instant::now();
        let deadline = start + Duration::from_millis(50);
        precise_sleep_until(deadline);
        let actual = Instant::now() - start;
        assert!(
            actual >= Duration::from_millis(50),
            "woke up early: {actual:?} (deadline was 50 ms)"
        );
        assert!(
            actual < Duration::from_millis(55),
            "overshot too much: {actual:?} (deadline was 50 ms)"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn precise_sleep_steady_cadence() {
        // Five chained 20-ms periods should land within +/-5 ms of 100 ms;
        // catches per-call timer-rounding drift.
        let period = Duration::from_millis(20);
        let start = Instant::now();
        let mut deadline = start;
        for _ in 0..5 {
            deadline += period;
            precise_sleep_until(deadline);
        }
        let actual = Instant::now() - start;
        let expected = period * 5;
        let diff = actual.abs_diff(expected);
        assert!(
            diff < Duration::from_millis(5),
            "cadence drifted: actual={actual:?} expected={expected:?}",
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn precise_sleep_past_deadline_returns_immediately() {
        // A past deadline must not deadlock the spin loop.
        let start = Instant::now();
        let deadline = start - Duration::from_millis(10);
        precise_sleep_until(deadline);
        let actual = Instant::now() - start;
        assert!(
            actual < Duration::from_millis(2),
            "took too long: {actual:?}"
        );
    }
}
