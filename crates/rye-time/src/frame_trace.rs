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
///
/// `heap_delta_bytes` is the JS heap growth observed inside this frame, sampled at
/// `begin_frame` and again at `end_frame`. Only populated when a host has
/// registered a [`HeapSampler`] via [`set_heap_sampler`] AND the runtime exposes
/// the underlying API (Chrome / Edge expose `performance.memory.usedJSHeapSize`;
/// Firefox + native return `None`). Positive values indicate growth, negative
/// values indicate that a GC happened inside the frame and reclaimed some heap.
///
/// `allocs` is the per-frame allocation delta observed by the
/// [`crate::alloc::CountingAllocator`] wrapper. Only populated when a demo
/// installs that allocator as its `#[global_allocator]`. The fields cover net
/// bytes (signed), bytes allocated (unsigned), and the alloc/dealloc call
/// counts, which together let the PerfOverlay show "we're allocating 1.2 MB
/// across 3,400 calls per frame"; both numbers matter when chasing the
/// per-frame-interop-leak pattern characterized 2026-05-22.
#[derive(Clone, Debug, Default)]
pub struct FrameTrace {
    pub sections: Vec<Section>,
    pub heap_delta_bytes: Option<i64>,
    pub allocs: Option<crate::alloc::AllocDelta>,
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

/// Function pointer registered by the host to sample the JS heap in bytes. On
/// wasm32 + Chromium, the host wires this to read
/// `performance.memory.usedJSHeapSize` via wasm-bindgen + Reflect; on Firefox +
/// native, no sampler is set (the field stays `None`) and `heap_delta_bytes`
/// on every `FrameTrace` is `None`.
///
/// Architectural note: keeping the sampler as a function-pointer slot
/// registered from outside means `rye-time` doesn't depend on `js-sys` /
/// `web-sys` for this functionality; the host crate (`rye-app`) owns the
/// platform-specific access and registers a callback. Keeps the leaf crate's
/// dep graph small.
pub type HeapSampler = fn() -> Option<u64>;

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
    /// JS heap snapshot at `begin_frame`. Paired with the `end_frame` snapshot
    /// to produce `heap_delta_bytes` on the completed `FrameTrace`. `None`
    /// until the host registers a [`HeapSampler`] via [`set_heap_sampler`].
    static CURRENT_FRAME_HEAP_START: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    /// CountingAllocator snapshot at `begin_frame`. Paired with the
    /// `end_frame` snapshot to produce `allocs` on the completed
    /// `FrameTrace`. `None` until the demo installs the wrapper.
    static CURRENT_FRAME_ALLOC_START: std::cell::Cell<Option<crate::alloc::AllocSnapshot>> = const { std::cell::Cell::new(None) };
    /// Optional host-registered JS heap sampler. Set once at startup by the
    /// host; called from `begin_frame` + `end_frame`. Cell because function
    /// pointers are Copy.
    static HEAP_SAMPLER: std::cell::Cell<Option<HeapSampler>> = const { std::cell::Cell::new(None) };
    /// Session-lifetime maxima per section name. Distinct lifecycle from
    /// [`TRACER::history`]: the rolling window is the recent-distribution
    /// signal (drops samples as frames age out), while `MAX_EVER` is the
    /// outlier-visibility signal that survives indefinitely.
    ///
    /// Architectural note: a 1-second freeze inserts ONE entry into the rolling
    /// window. With a 120-frame window at 50fps (= 2.4s of history), spikes that
    /// happen sparser than ~once-per-second fall out of the window before a
    /// human notices the demo stuttered. `MAX_EVER` is the answer to "what's
    /// the worst this has ever been?"; independent of when the user opened
    /// the perf overlay. Cleared only by [`clear_max_ever`].
    static MAX_EVER: RefCell<std::collections::HashMap<&'static str, Duration>> =
        RefCell::new(std::collections::HashMap::new());
    /// Threshold above which `end_frame` emits a `tracing::warn!` naming the
    /// offending section + its elapsed time + the frame index. Default chosen
    /// to catch user-visible stalls (>= 50ms ≈ 3 vsync intervals at 60Hz)
    /// without flooding on common 25ms variance.
    ///
    /// Architectural note: tracing::warn! goes to console.warn on wasm
    /// (selectable + copyable in DevTools) and to stderr on native; both are
    /// the right surfaces for "something just went pathological for one
    /// frame." If this becomes too chatty under load, raise the threshold or
    /// add a "first N per session" gate rather than turning it off.
    static SPIKE_THRESHOLD: std::cell::Cell<Duration> =
        const { std::cell::Cell::new(Duration::from_millis(50)) };
    /// Strictly-increasing frame counter for the spike log message, so a
    /// human reading three "spike at frame 1247" messages can tell they're
    /// genuinely separate events vs. one redraw firing the warn multiple
    /// times.
    static FRAME_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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
/// name (`let _s = scope("foo")`); binding to `_` drops it immediately and records
/// a zero-duration section, which is not what you want.
#[cfg(feature = "frame-trace")]
#[inline]
pub fn scope(name: &'static str) -> Scope {
    Scope {
        name,
        start: Instant::now(),
    }
}

/// Register the host's JS heap sampler. Called once at startup; subsequent calls
/// overwrite. On wasm32 + Chromium the host wires this to
/// `performance.memory.usedJSHeapSize` via wasm-bindgen Reflect; on Firefox +
/// native the host either doesn't call this or registers a sampler that returns
/// `None`, leaving every `FrameTrace::heap_delta_bytes` as `None`.
///
/// Once registered, [`begin_frame`] + [`end_frame`] snapshot the heap and the
/// signed delta is attached to each completed `FrameTrace`. The spike-warn log
/// message also includes the delta when present, so a 500ms freeze with a
/// matching 30 MB heap jump reads as "GC pressure" at a glance.
#[cfg(feature = "frame-trace")]
pub fn set_heap_sampler(sampler: HeapSampler) {
    HEAP_SAMPLER.with(|s| s.set(Some(sampler)));
}

/// Mark the start of a frame's work. Pairs with [`end_frame`] to compute the `idle`
/// section (time the browser/event loop spent before handing us this frame). Called
/// by the runner at the top of each `redraw`, BEFORE opening the `frame` scope.
///
/// Without this signal, `end_frame` can only measure total cadence
/// (`between-frames`); it can't tell how much of that was our CPU work vs. how much
/// was the browser doing something else.
///
/// Also snapshots the JS heap (via the registered [`HeapSampler`], if any) so
/// [`end_frame`] can compute a per-frame heap delta.
#[cfg(feature = "frame-trace")]
pub fn begin_frame() {
    CURRENT_FRAME_START.with(|c| c.set(Some(Instant::now())));
    let sampler = HEAP_SAMPLER.with(|s| s.get());
    CURRENT_FRAME_HEAP_START.with(|c| c.set(sampler.and_then(|f| f())));
    // Alloc counters are global atomics in `crate::alloc`; a `current_snapshot`
    // call is four `Relaxed` loads + a bool check, so ~ns. Returns None when
    // the demo hasn't installed the CountingAllocator wrapper.
    CURRENT_FRAME_ALLOC_START.with(|c| c.set(crate::alloc::current_snapshot()));
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
///   That's the gap when our code wasn't running; browser RAF scheduling, vsync
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

    // Pair the begin_frame heap snapshot with a fresh sample here. Either may be
    // None (sampler not registered, or platform doesn't expose the API); we only
    // compute a delta when both samples exist. Stored as signed i64 so a GC
    // mid-frame can show as negative.
    let heap_start = CURRENT_FRAME_HEAP_START.with(|c| c.take());
    let sampler = HEAP_SAMPLER.with(|s| s.get());
    let heap_end = sampler.and_then(|f| f());
    let heap_delta_bytes: Option<i64> = match (heap_start, heap_end) {
        (Some(a), Some(b)) => Some((b as i64).saturating_sub(a as i64)),
        _ => None,
    };

    // Same pattern for alloc counters. Both endpoints come from the global
    // atomic counters; if the wrapper isn't installed, both are None and we
    // leave `allocs` unset on the frame.
    let alloc_start = CURRENT_FRAME_ALLOC_START.with(|c| c.take());
    let alloc_end = crate::alloc::current_snapshot();
    let alloc_delta: Option<crate::alloc::AllocDelta> = match (alloc_start, alloc_end) {
        (Some(a), Some(b)) => Some(crate::alloc::delta(a, b)),
        _ => None,
    };

    // Advance the strictly-increasing frame counter once per end_frame. Cheap
    // (single Cell write) and gives the spike-log a way to name the frame.
    let frame_index = FRAME_COUNTER.with(|c| {
        let n = c.get();
        c.set(n.wrapping_add(1));
        n
    });

    // Collect the frame's sections for max_ever + spike-log AFTER the borrow_mut
    // below drops; doing this inside the same borrow would force a clone path
    // through `current.sections`. We snapshot the section names + durations
    // before rotating history.
    let threshold = SPIKE_THRESHOLD.with(|c| c.get());
    let mut new_max: Vec<(&'static str, Duration)> = Vec::new();
    let mut over_threshold: Vec<(&'static str, Duration)> = Vec::new();

    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        // Attach the per-frame heap delta before the frame rolls into history.
        // None when no sampler is registered or the platform doesn't expose
        // the API; otherwise signed bytes (negative on GC mid-frame).
        t.current.heap_delta_bytes = heap_delta_bytes;
        t.current.allocs = alloc_delta;
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

        // Pre-scan the just-completed frame so the post-borrow code can update
        // MAX_EVER + emit warnings without holding the TRACER borrow.
        // Architectural note: doing this inside the borrow would force MAX_EVER
        // updates to also be under that borrow's lifetime, and a future
        // `tracing::warn!` subscriber that re-enters `frame_trace` would
        // deadlock. Keeping the two RefCells independent costs an extra Vec
        // walk per frame (cheap; same length as the frame's section count).
        for section in &t.current.sections {
            new_max.push((section.name, section.elapsed));
            if section.elapsed > threshold {
                over_threshold.push((section.name, section.elapsed));
            }
        }

        let cap = t.capacity;
        let frame = std::mem::take(&mut t.current);
        if t.history.len() >= cap {
            t.history.pop_front();
        }
        t.history.push_back(frame);
    });

    // Now update MAX_EVER (a separate RefCell, so no nested-borrow risk).
    MAX_EVER.with(|m| {
        let mut m = m.borrow_mut();
        for (name, elapsed) in new_max {
            let entry = m.entry(name).or_insert(Duration::ZERO);
            if elapsed > *entry {
                *entry = elapsed;
            }
        }
    });

    // Emit spike warnings outside the trace borrows so a tracing subscriber
    // that re-enters frame_trace (e.g. for its own scope) doesn't conflict
    // with our state. Common case is zero entries per frame; only allocates
    // strings on the actual spike path. The heap-delta and allocs suffixes
    // are appended only when their respective signals are wired AND produced
    // valid endpoints; on platforms without them the fields stay absent so
    // the log doesn't show misleading "heap=0" / "allocs=0" reads.
    for (name, elapsed) in over_threshold {
        let heap_suffix = heap_delta_bytes
            .map(|d| format!(" heap_delta={:+.2}MB", d as f64 / (1024.0 * 1024.0)))
            .unwrap_or_default();
        let alloc_suffix = alloc_delta
            .map(|d| {
                format!(
                    " allocs={} ({:+.2}MB net)",
                    d.alloc_count,
                    d.net_bytes as f64 / (1024.0 * 1024.0),
                )
            })
            .unwrap_or_default();
        tracing::warn!(
            "frame_trace spike: section='{name}' elapsed={:.1}ms frame={frame_index}{heap_suffix}{alloc_suffix}",
            elapsed.as_secs_f32() * 1000.0,
        );
    }
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
///
/// Prefer [`with_history`] in per-frame callers (e.g. PerfOverlay): cloning the
/// whole history once a frame at 60fps is ~120 allocations/frame on its own,
/// which is enough to swamp the actual demo's allocation rate when reading
/// the alloc telemetry.
#[cfg(feature = "frame-trace")]
pub fn history() -> Vec<FrameTrace> {
    TRACER.with(|t| t.borrow().history.iter().cloned().collect())
}

/// Run `f` with a borrow of the rolling history. Zero-allocation read path:
/// per-frame callers (PerfOverlay) iterate refs and accumulate scalars on the
/// stack without cloning the VecDeque or its `Vec<Section>` payloads.
///
/// `f` must NOT call back into `frame_trace` mutators (`scope`, `end_frame`,
/// `record_external`) while the borrow is held; that would deadlock the
/// `RefCell`. Reading via `last_frame`, `max_ever`, etc. is fine because
/// those use separate cells / re-entry-safe paths.
///
/// Returns whatever `f` returns so callers can pipe through stack-only
/// reductions (sums, means, max) without intermediate Vecs.
#[cfg(feature = "frame-trace")]
pub fn with_history<R>(f: impl FnOnce(&std::collections::VecDeque<FrameTrace>) -> R) -> R {
    TRACER.with(|t| f(&t.borrow().history))
}

/// Snapshot only the last completed frame. Cheaper than [`history`] when the caller
/// just wants "what happened on the most recent frame" (a per-frame readout).
#[cfg(feature = "frame-trace")]
pub fn last_frame() -> Option<FrameTrace> {
    TRACER.with(|t| t.borrow().history.back().cloned())
}

/// Session-lifetime maximum elapsed time recorded for `name` across every
/// frame since program start (or last [`clear_max_ever`]).
///
/// Use this when the rolling window's `max` may have already aged out a spike.
/// A page that runs for an hour with a single 500ms freeze will still report
/// the 500ms via `max_ever("between-frames")` long after the 120-frame window
/// has rotated past it. Returns `Duration::ZERO` for never-seen sections.
#[cfg(feature = "frame-trace")]
pub fn max_ever(name: &'static str) -> Duration {
    MAX_EVER.with(|m| m.borrow().get(name).copied().unwrap_or(Duration::ZERO))
}

/// All session-lifetime maxima, name -> duration. Sorted by descending
/// duration so the worst offenders sort to the top of caller-rendered tables.
#[cfg(feature = "frame-trace")]
pub fn all_max_ever() -> Vec<(&'static str, Duration)> {
    MAX_EVER.with(|m| {
        let mut out: Vec<(&'static str, Duration)> =
            m.borrow().iter().map(|(k, v)| (*k, *v)).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    })
}

/// Reset session-lifetime maxima. Doesn't touch the rolling window. Useful
/// after the user navigates into a new mode + wants to start measuring fresh.
#[cfg(feature = "frame-trace")]
pub fn clear_max_ever() {
    MAX_EVER.with(|m| m.borrow_mut().clear());
}

/// Set the threshold above which `end_frame` logs a `tracing::warn!`. Default
/// is 50ms. Pass `Duration::MAX` to disable spike logging entirely.
///
/// Architectural note: the threshold is process-global because spikes are
/// a single concept regardless of which scope produced them. Per-section
/// thresholds would clutter the API for a use case (silencing chatty
/// sections) that hasn't materialized.
#[cfg(feature = "frame-trace")]
pub fn set_spike_threshold(threshold: Duration) {
    SPIKE_THRESHOLD.with(|c| c.set(threshold));
}

/// Push a section produced outside the normal `scope` lifecycle. Used by the GPU
/// timer path: a GPU timestamp's wall-clock delta arrives via `map_async` callback,
/// outside the scope-on-drop flow, but it conceptually belongs to the current frame.
///
/// The section lands in `current` (the in-flight frame) and rides into history with
/// the next `end_frame`. If `end_frame` has already rolled for the frame the
/// timestamp belongs to (typical: timestamps arrive 1-2 frames late), the section is
/// attributed to whatever frame is currently in flight. That's good enough for
/// aggregate stats; the rolling window absorbs the small attribution drift.
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

#[cfg(not(feature = "frame-trace"))]
pub fn max_ever(_name: &'static str) -> Duration {
    Duration::ZERO
}

#[cfg(not(feature = "frame-trace"))]
pub fn all_max_ever() -> Vec<(&'static str, Duration)> {
    Vec::new()
}

#[cfg(not(feature = "frame-trace"))]
pub fn clear_max_ever() {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_spike_threshold(_threshold: Duration) {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_heap_sampler(_sampler: HeapSampler) {}

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
pub fn with_history<R>(f: impl FnOnce(&std::collections::VecDeque<FrameTrace>) -> R) -> R {
    let empty = std::collections::VecDeque::new();
    f(&empty)
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
    fn heap_sampler_populates_delta_on_completed_frame() {
        use std::sync::atomic::{AtomicU64, Ordering};
        // Thread-local atomic counter so the synthetic sampler returns
        // strictly-increasing values without needing real perf.memory. Two
        // calls inside one frame: one in begin_frame, one in end_frame; the
        // delta should equal the per-call increment.
        static FAKE_HEAP: AtomicU64 = AtomicU64::new(1_000_000);
        fn fake_sampler() -> Option<u64> {
            Some(FAKE_HEAP.fetch_add(4096, Ordering::SeqCst) + 4096)
        }
        FAKE_HEAP.store(1_000_000, Ordering::SeqCst);
        set_heap_sampler(fake_sampler);
        // Drain any pre-existing in-flight frame from prior tests so the
        // begin/end pair below produces the heap-delta-bearing frame.
        end_frame();
        begin_frame();
        end_frame();
        let frame = last_frame().expect("end_frame should produce a frame");
        let delta = frame
            .heap_delta_bytes
            .expect("sampler is registered; delta should be Some");
        // begin captured first, end captured second; both incremented the
        // counter by 4096. Delta = end - begin = 4096.
        assert_eq!(delta, 4096, "expected one-increment delta, got {delta}");
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
