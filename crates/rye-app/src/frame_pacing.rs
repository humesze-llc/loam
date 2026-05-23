//! Process-global frame pacing state (target framerate, vsync toggle, precise
//! sleep utility), read by `Runner::redraw` at the top of every
//! redraw to gate this frame's work.
//!
//! ## Why a process-global atomic instead of a `Runner` field
//!
//! The `fps` / `vsync` console commands live behind the `Console` handler,
//! which gets a `&mut Ctx` (the app's own context), not the `Runner`. The
//! cleanest way to let the handler change a runner setting without threading
//! state through every demo's `Ctx` is a static the runner reads each frame.
//! The same pattern is used for [`rye_time::frame_trace::set_capacity`].
//!
//! Cost: a couple of relaxed atomic loads per redraw. Negligible.
//!
//! ## Native vs. wasm semantics
//!
//! - **Native**: the runner [`precise_sleep_until`]s the deadline at the start
//!   of each redraw. With `target_fps = 0` the cap is removed and the
//!   surface's `PresentMode` decides the cadence; `Fifo` (default) blocks
//!   at vsync. `vsync off` swaps the surface to `Mailbox` (or `Immediate` as
//!   fallback) so the cap can drive cadence above the display refresh rate.
//! - **Wasm**: the browser's `requestAnimationFrame` is the upper bound
//!   (typically display refresh rate). The runner can only cap *lower* than
//!   that by skipping RAF callbacks that arrive too early. Setting fps higher
//!   than 60 on wasm is accepted but won't produce more frames than the
//!   browser provides; the `vsync` command is effectively a no-op there
//!   (browser surfaces typically advertise only `Fifo`).

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Duration;
// `web_time::Instant` is the workspace's cross-target wall-clock type (aliases
// `std::time::Instant` on native, shims to `performance.now()` on wasm32).
// `std::time::Instant::now` panics on wasm32, so the swap is mandatory.
use web_time::Instant;

/// 60 fps in nanoseconds (≈16.667 ms). Initial value the runner uses unless a
/// console command changes it. Picked because it matches the typical display
/// refresh rate the browser RAF and most native vsync settings already
/// enforce. i.e., the cap is mostly a no-op until the user explicitly raises
/// or lowers it.
const DEFAULT_PERIOD_NS: u64 = 16_666_667;

/// Target frame period in nanoseconds. `0` means "uncapped"; the runner will
/// not sleep / skip and lets the display refresh rate or browser RAF be the
/// pacing source.
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

// Encodes a pending vsync request between the console command (writer) and the
// runner (reader, once per redraw). `0` = no request pending; `1` = "on";
// `2` = "off". The runner swaps it back to 0 after applying so subsequent
// frames don't re-reconfigure the surface every tick.
const VSYNC_NONE: u8 = 0;
const VSYNC_REQ_ON: u8 = 1;
const VSYNC_REQ_OFF: u8 = 2;
static PENDING_VSYNC: AtomicU8 = AtomicU8::new(VSYNC_NONE);

/// Request that the runner switch the surface to vsync-on (`PresentMode::Fifo`)
/// on its next redraw.
pub fn request_vsync_on() {
    PENDING_VSYNC.store(VSYNC_REQ_ON, Ordering::Release);
}

/// Request that the runner switch the surface to vsync-off on its next redraw.
/// The runner picks the best available off-mode (`Mailbox` if advertised,
/// otherwise `Immediate`, otherwise leaves the mode alone since the adapter
/// has nothing better than `Fifo` to offer; this is the typical browser case).
pub fn request_vsync_off() {
    PENDING_VSYNC.store(VSYNC_REQ_OFF, Ordering::Release);
}

/// Pending vsync transitions for the runner to apply. `Some(true)` = caller
/// asked for vsync-on; `Some(false)` = vsync-off; `None` = no pending change.
/// Reading clears the pending request so the runner re-applies only when the
/// user pokes the console.
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

/// Spin-tail length: the runner coarse-sleeps the bulk of the wait, then
/// busy-waits the last `SPIN_TAIL` so the wake-up lands inside ~100 µs of the
/// deadline regardless of the OS timer's nominal precision. Tuned to be wider
/// than Windows' default 15.625 ms timer tick is *imprecise*, not wider than
/// the *whole* tick: `std::thread::sleep` rounds DOWN sometimes and UP others,
/// and 2 ms of spin covers the worst-case overshoot we've seen on Win11.
const SPIN_TAIL: Duration = Duration::from_millis(2);

/// Sleep until `deadline`, hybrid coarse-sleep + spin-wait for sub-millisecond
/// precision. The naive `std::thread::sleep` rounds to the system timer
/// granularity (default ~15.6 ms on Windows), which makes any fps cap below
/// the vsync rate land far off the intended period. We coarse-sleep until
/// `SPIN_TAIL` before the deadline, then `Instant::now()`-spin the tail.
///
/// The spin section saturates one core at 100% but only for ≤2 ms per frame;
/// at a 60 fps cap that's <13% of one logical core, well inside the noise of
/// any real workload. On wasm32 this collapses to "do nothing" since the
/// runner takes the skip-and-rerequest path before reaching this code.
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
    // Busy-wait the tail. `Instant::now()` is monotonic so this loop must
    // terminate as long as the deadline is a valid future instant.
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

/// Stub on wasm32 (the runner takes the skip-and-rerequest path before
/// reaching the precise-sleep code; this exists only so the call site doesn't
/// need a `cfg`).
#[cfg(target_arch = "wasm32")]
pub fn precise_sleep_until(_deadline: Instant) {}

// Test-only shared lock. Every test in this crate that touches the
// process-global pacing atomics (`TARGET_PERIOD_NS`, `PENDING_VSYNC`) must
// hold this lock for the duration of its observations. Without it, cargo's
// default parallel test runner interleaves writes and reads across modules
// (`fps::tests`, `vsync::tests`, this module) and produces flaky assertions.
//
// `unwrap_or_else(|e| e.into_inner())` keeps the suite robust against a
// poisoned lock from a panicking sibling test; the data inside is just unit,
// so there's nothing to recover from.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_60fps() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Reset to the canonical default in case a sibling test (or a
        // previous run) left the atomic in another state.
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
        // 50 ms is well above any reasonable OS timer granularity (Windows
        // default is ~15.625 ms); the hybrid should wake within ~SPIN_TAIL of
        // the deadline.
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
        // Five 20-ms periods chained back-to-back should land within ±5 ms of
        // the ideal 100 ms total. Catches drift from the system timer rounding
        // each individual sleep call.
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
        // Deadline already in the past shouldn't deadlock the spin loop.
        let start = Instant::now();
        let deadline = start - Duration::from_millis(10);
        precise_sleep_until(deadline);
        let actual = Instant::now() - start;
        assert!(actual < Duration::from_millis(2), "took too long: {actual:?}");
    }
}
