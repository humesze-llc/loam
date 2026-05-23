//! Process-global target framerate. Read by [`crate::Runner::redraw`] at the top
//! of every redraw call to decide whether to sleep (native) or skip-and-rerequest
//! (wasm) before doing this frame's work.
//!
//! ## Why a process-global atomic instead of a `Runner` field
//!
//! The `fps` console command lives behind the [`Console`] handler, which gets a
//! `&mut Ctx` (the app's own context) — not the [`Runner`]. The cleanest way to
//! let the handler change a runner setting without threading state through every
//! demo's `Ctx` is a static the runner reads each frame. The same pattern is
//! used for [`rye_time::frame_trace::set_capacity`].
//!
//! Cost: an `AtomicU64::load(Acquire)` per redraw. Negligible.
//!
//! ## Native vs. wasm semantics
//!
//! - **Native**: the runner sleeps the remainder of the target period at the
//!   start of each redraw. With `target_fps = 0` the cap is removed and the
//!   surface's `PresentMode` (typically vsync/`Fifo`) decides the cadence.
//!   Setting `target_fps` higher than the display refresh rate has no effect
//!   beyond vsync — `Fifo` will still block at present.
//! - **Wasm**: the browser's `requestAnimationFrame` is the upper bound
//!   (typically display refresh rate). The runner can only cap *lower* than
//!   that by skipping RAF callbacks that arrive too early. Setting a value
//!   higher than 60 on wasm is allowed; it just won't produce more frames than
//!   the browser already provides.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// 60 fps in nanoseconds (≈16.667 ms). Initial value the runner uses unless a
/// console command changes it. Picked because it matches the typical display
/// refresh rate the browser RAF and most native vsync settings already enforce
/// — i.e. the cap is mostly a no-op until the user explicitly raises or lowers
/// it.
const DEFAULT_PERIOD_NS: u64 = 16_666_667;

/// Target frame period in nanoseconds. `0` means "uncapped" — the runner will
/// not sleep / skip and lets the display refresh rate or browser RAF be the
/// pacing source.
static TARGET_PERIOD_NS: AtomicU64 = AtomicU64::new(DEFAULT_PERIOD_NS);

/// Set the target frame period from a desired fps. `fps <= 0.0` removes the cap
/// (uncapped — frames as fast as the surface and browser allow).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_60fps() {
        assert!((target_fps() - 60.0).abs() < 0.01);
        assert!(target_period().is_some());
    }

    #[test]
    fn unlimited_round_trip() {
        set_target_fps(0.0);
        assert_eq!(target_fps(), 0.0);
        assert_eq!(target_period(), None);
        // Restore default for other tests sharing this process.
        set_target_fps(60.0);
    }

    #[test]
    fn set_then_read() {
        set_target_fps(144.0);
        let f = target_fps();
        assert!((f - 144.0).abs() < 0.5);
        set_target_fps(60.0);
    }
}
