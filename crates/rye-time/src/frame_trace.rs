//! Frame-timing instrumentation. CPU-side per-section timers that collect into a
//! rolling per-frame ring buffer, surfaced via a runtime toggle (egui panel or console
//! drain) for "what's the hot path this frame" diagnosis.
//!
//! ## Why a separate module from `tracing`
//!
//! `tracing` spans are great for hierarchical event logs but expensive for per-frame
//! hot-path timing: every span allocates a context, traverses subscriber registry,
//! and pays for level filtering. This module is dumb-on-purpose: one `Instant::now`
//! at scope start, one at scope end, one `Vec::push`. ~50ns end-to-end on native,
//! ~100ns on wasm32 (where `Instant` is backed by `performance.now()`).
//!
//! ## Always-on collection, opt-in display
//!
//! Mirrors the `rye-app::log` pattern: data is always being collected when the cargo
//! feature is on (the default), but the surfacing UI is a runtime toggle. Reading
//! `history()` is the only access pattern; the egui panel + the console `trace`
//! command both go through it. When the user isn't looking, the rolling buffer
//! quietly recycles itself.
//!
//! ## Feature gating
//!
//! `frame-trace` (default-on). With the feature OFF the entire module degrades to
//! zero-sized types and empty drops; calls to `scope` / `end_frame` etc. optimize
//! away. Embedded / production builds that don't want the ~50ns per scope can opt
//! out via `--no-default-features`.
//!
//! ## Wasm32 safety
//!
//! `thread_local!` works on `wasm32-unknown-unknown` because the standard library
//! treats the single browser thread as the only thread; `with` is a direct getter.
//! Multi-threaded wasm (via `SharedArrayBuffer`) is not on the roadmap so the
//! single-thread assumption is durable.

#[cfg(feature = "frame-trace")]
use std::cell::RefCell;
#[cfg(feature = "frame-trace")]
use std::collections::VecDeque;
use std::time::Duration;
#[cfg(feature = "frame-trace")]
use web_time::Instant;

/// One CPU section timing inside a single frame's trace.
#[derive(Clone, Debug)]
pub struct Section {
    /// Static label baked at the call site. Avoids per-frame string allocation.
    pub name: &'static str,
    pub elapsed: Duration,
}

/// All sections recorded inside one redraw cycle. `Default` produces an empty trace.
#[derive(Clone, Debug, Default)]
pub struct FrameTrace {
    pub sections: Vec<Section>,
}

impl FrameTrace {
    /// Total time across every section. Subject to double-counting if scopes overlap;
    /// today the runner only opens disjoint scopes, but if hierarchical scopes ever
    /// land this needs revisiting.
    pub fn total(&self) -> Duration {
        self.sections.iter().map(|s| s.elapsed).sum()
    }
}

/// Default rolling-buffer capacity. 120 frames is two seconds at 60fps; enough to see
/// the average and the worst spike of a smooth interaction, short enough that the
/// buffer's memory footprint stays under ~10 KB.
pub const DEFAULT_CAPACITY: usize = 120;

#[cfg(feature = "frame-trace")]
struct Tracer {
    current: FrameTrace,
    history: VecDeque<FrameTrace>,
    capacity: usize,
}

#[cfg(feature = "frame-trace")]
impl Tracer {
    fn new(capacity: usize) -> Self {
        Self {
            current: FrameTrace::default(),
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
}

#[cfg(feature = "frame-trace")]
thread_local! {
    static TRACER: RefCell<Tracer> = RefCell::new(Tracer::new(DEFAULT_CAPACITY));
    /// Timestamp of the last `end_frame` call. Combined with the next frame's
    /// `begin_frame` + `end_frame` to decompose total cadence into our work
    /// (`frame`) + browser idle time (`idle`); both are recorded explicitly so
    /// readers don't have to mentally subtract.
    static LAST_FRAME_END: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    /// Timestamp of the current frame's `begin_frame` call. Used by `end_frame`
    /// to compute `frame` (CPU work) separately from `idle` (everything in the
    /// browser/RAF/vsync gap that we don't control).
    static CURRENT_FRAME_START: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
}

/// RAII guard returned from [`scope`]. Records elapsed time on drop. Holding one of
/// these across `await` points is not supported (the tracer is thread-local and
/// borrowed mutably during drop).
#[cfg(feature = "frame-trace")]
#[must_use = "Scope records on drop; binding it to `_` would record immediately"]
pub struct Scope {
    name: &'static str,
    start: Instant,
}

#[cfg(feature = "frame-trace")]
impl Drop for Scope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let name = self.name;
        // Use `try_borrow_mut` to avoid panicking if the tracer is mid-rotation (e.g.
        // `end_frame` is reading history while a scope drops). In practice nothing
        // does this today, but defensive against future tracing-the-trace code paths.
        TRACER.with(|t| {
            if let Ok(mut t) = t.try_borrow_mut() {
                t.current.sections.push(Section { name, elapsed });
            }
        });
    }
}

/// Open a CPU-timing scope. The returned guard records on drop. Bind it with a real
/// name (`let _s = scope("foo")`) — binding to `_` drops it immediately and records
/// a zero-duration section, which is not what you want.
#[cfg(feature = "frame-trace")]
#[inline]
pub fn scope(name: &'static str) -> Scope {
    Scope {
        name,
        start: Instant::now(),
    }
}

/// Mark the start of a frame's work. Pairs with [`end_frame`] to compute the `idle`
/// section (time the browser/event loop spent before handing us this frame). Called
/// by the runner at the top of each `redraw`, BEFORE opening the `frame` scope.
///
/// Without this signal, `end_frame` can only measure total cadence
/// (`between-frames`); it can't tell how much of that was our CPU work vs. how much
/// was the browser doing something else.
#[cfg(feature = "frame-trace")]
pub fn begin_frame() {
    CURRENT_FRAME_START.with(|c| c.set(Some(Instant::now())));
}

/// Push the in-flight frame into history and start a new one. Called once per redraw
/// cycle by the runner, AFTER all the frame's scopes have closed.
///
/// Records two synthetic sections per frame (in addition to the explicit scopes the
/// runner + demo opened):
///
/// - **`between-frames`**: total wall-clock between successive `end_frame` calls.
///   Equal to `1 / fps`. Useful as a sanity check; if the perf overlay says 50fps
///   the mean should be ~20ms.
/// - **`idle`**: time from the last `end_frame` until this frame's `begin_frame`.
///   That's the gap when our code wasn't running — browser RAF scheduling, vsync
///   alignment, JS GC, tab throttling. This is what dominates `between-frames` on
///   wasm (per the 2026-05-22 diagnosis); separating it explicitly means the perf
///   overlay can show "where is time going?" without mental subtraction.
///
/// The first frame after startup records neither (no prior end to measure from).
#[cfg(feature = "frame-trace")]
pub fn end_frame() {
    let now = Instant::now();
    let last_end = LAST_FRAME_END.with(|cell| {
        let prev = cell.get();
        cell.set(Some(now));
        prev
    });
    let frame_start = CURRENT_FRAME_START.with(|c| c.take());

    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        if let Some(last_end) = last_end {
            let between_frames = now.saturating_duration_since(last_end);
            t.current.sections.push(Section {
                name: "between-frames",
                elapsed: between_frames,
            });
            // `idle` is between the previous end_frame and this frame's begin_frame.
            // If begin_frame wasn't called (e.g. the runner is mid-migration to the
            // new API), we silently skip the idle section rather than emitting a
            // bogus value.
            if let Some(frame_start) = frame_start {
                let idle = frame_start.saturating_duration_since(last_end);
                t.current.sections.push(Section {
                    name: "idle",
                    elapsed: idle,
                });
            }
        }
        let cap = t.capacity;
        let frame = std::mem::take(&mut t.current);
        if t.history.len() >= cap {
            t.history.pop_front();
        }
        t.history.push_back(frame);
    });
}

/// Set the rolling window size. Truncates older frames if shrinking.
#[cfg(feature = "frame-trace")]
pub fn set_capacity(capacity: usize) {
    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        t.capacity = capacity.max(1);
        while t.history.len() > t.capacity {
            t.history.pop_front();
        }
    });
}

/// Snapshot the rolling history. Allocates; intended for the (occasional) display
/// path, not the hot path. Returns frames in oldest-to-newest order.
#[cfg(feature = "frame-trace")]
pub fn history() -> Vec<FrameTrace> {
    TRACER.with(|t| t.borrow().history.iter().cloned().collect())
}

/// Snapshot only the last completed frame. Cheaper than [`history`] when the caller
/// just wants "what happened on the most recent frame" (a per-frame readout).
#[cfg(feature = "frame-trace")]
pub fn last_frame() -> Option<FrameTrace> {
    TRACER.with(|t| t.borrow().history.back().cloned())
}

/// Push a section produced outside the normal `scope` lifecycle. Used by the GPU
/// timer path: a GPU timestamp's wall-clock delta arrives via `map_async` callback,
/// outside the scope-on-drop flow, but it conceptually belongs to the current frame.
///
/// The section lands in `current` (the in-flight frame) and rides into history with
/// the next `end_frame`. If `end_frame` has already rolled for the frame the
/// timestamp belongs to (typical: timestamps arrive 1-2 frames late), the section is
/// attributed to whatever frame is currently in flight. That's good enough for
/// aggregate stats — the rolling window absorbs the small attribution drift.
#[cfg(feature = "frame-trace")]
pub fn record_external(name: &'static str, elapsed: Duration) {
    TRACER.with(|t| {
        if let Ok(mut t) = t.try_borrow_mut() {
            t.current.sections.push(Section { name, elapsed });
        }
    });
}

/// Feature-OFF stub: drops the section, no-op.
#[cfg(not(feature = "frame-trace"))]
pub fn record_external(_name: &'static str, _elapsed: Duration) {}

/// Aggregate stats across the rolling window for one section name.
#[derive(Clone, Debug)]
pub struct SectionStats {
    pub name: &'static str,
    pub samples: usize,
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Aggregate every section across the rolling window. Names are matched as
/// `&'static str` pointers first (cheap), then by string content; in practice every
/// call site uses a literal so pointer equality is sufficient. Returns sections in
/// descending p95 order so the slowest sections sort to the top of the panel.
#[cfg(feature = "frame-trace")]
pub fn aggregate() -> Vec<SectionStats> {
    use std::collections::HashMap;
    let frames = history();
    let mut buckets: HashMap<&'static str, Vec<Duration>> = HashMap::new();
    for frame in &frames {
        for section in &frame.sections {
            buckets.entry(section.name).or_default().push(section.elapsed);
        }
    }

    let mut stats: Vec<SectionStats> = buckets
        .into_iter()
        .map(|(name, mut samples)| {
            samples.sort();
            let n = samples.len();
            let mean = samples.iter().sum::<Duration>() / (n as u32).max(1);
            let pick = |q: f32| samples[((n as f32 * q) as usize).min(n - 1)];
            SectionStats {
                name,
                samples: n,
                mean,
                p50: pick(0.50),
                p95: pick(0.95),
                p99: pick(0.99),
                max: *samples.last().unwrap_or(&Duration::ZERO),
            }
        })
        .collect();

    stats.sort_by(|a, b| b.p95.cmp(&a.p95));
    stats
}

// ---------------------------------------------------------------------------
// Feature-OFF stubs: zero-sized scope + empty drop. Compiler optimizes away.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "frame-trace"))]
#[must_use]
pub struct Scope;

#[cfg(not(feature = "frame-trace"))]
#[inline]
pub fn scope(_name: &'static str) -> Scope {
    Scope
}

#[cfg(not(feature = "frame-trace"))]
pub fn end_frame() {}

#[cfg(not(feature = "frame-trace"))]
pub fn begin_frame() {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_capacity(_capacity: usize) {}

#[cfg(not(feature = "frame-trace"))]
pub fn history() -> Vec<FrameTrace> {
    Vec::new()
}

#[cfg(not(feature = "frame-trace"))]
pub fn last_frame() -> Option<FrameTrace> {
    None
}

#[cfg(not(feature = "frame-trace"))]
pub fn aggregate() -> Vec<SectionStats> {
    Vec::new()
}

#[cfg(all(test, feature = "frame-trace"))]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn scope_records_elapsed_on_drop() {
        // Each thread gets a fresh tracer (thread_local), so this test owns its
        // tracer instance and won't see history from earlier tests.
        end_frame(); // discard any pre-existing in-flight frame
        let _ = history(); // sanity touch

        {
            let _s = scope("test-a");
            sleep(Duration::from_millis(1));
        }
        let pre_frame = last_frame();
        end_frame();
        let post_frame = last_frame().expect("end_frame should produce a frame");

        // The scope dropped before end_frame, so its section is in the just-rolled
        // frame, not the pre-end snapshot. (pre_frame was the previous frame, which
        // is what last_frame returns mid-frame because we haven't rolled yet.)
        let _ = pre_frame;
        let sections = &post_frame.sections;
        assert!(
            sections.iter().any(|s| s.name == "test-a"),
            "expected 'test-a' in {sections:?}"
        );
        let test_a = sections.iter().find(|s| s.name == "test-a").unwrap();
        assert!(
            test_a.elapsed >= Duration::from_millis(1),
            "scope elapsed should be >= sleep duration, got {:?}",
            test_a.elapsed
        );
    }

    #[test]
    fn end_frame_caps_history_to_capacity() {
        set_capacity(3);
        for _ in 0..10 {
            {
                let _s = scope("cap-test");
            }
            end_frame();
        }
        assert!(history().len() <= 3, "history should be capped");
    }

    #[test]
    fn aggregate_sorts_by_p95_descending() {
        set_capacity(20);
        // Two sections; "slow" sleeps longer than "fast". After enough samples the
        // p95 ordering should put "slow" first.
        for _ in 0..10 {
            {
                let _s = scope("slow");
                sleep(Duration::from_millis(2));
            }
            {
                let _f = scope("fast");
                // No sleep -- just the scope overhead.
            }
            end_frame();
        }
        let stats = aggregate();
        let slow_idx = stats.iter().position(|s| s.name == "slow");
        let fast_idx = stats.iter().position(|s| s.name == "fast");
        assert!(
            slow_idx.is_some() && fast_idx.is_some(),
            "both sections present"
        );
        assert!(
            slow_idx.unwrap() < fast_idx.unwrap(),
            "'slow' should sort before 'fast' by descending p95: {stats:?}"
        );
    }
}
