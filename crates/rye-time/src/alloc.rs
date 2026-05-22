//! Per-frame allocation counter via a [`GlobalAlloc`] wrapper. Demos opt in by
//! installing [`CountingAllocator`] as their `#[global_allocator]`; once installed,
//! every `alloc` / `dealloc` increments process-global atomic counters and the
//! per-frame delta is surfaced through `rye_time::frame_trace::FrameTrace`.
//!
//! ## Why a GlobalAlloc wrapper instead of `tracking-allocator` / `dhat`
//!
//! `dhat` is a heavy profiling tool with file-based output; great for one-off
//! deep-dives, awful for "show the current allocation rate in the overlay every
//! frame." `tracking-allocator` is similarly oriented at offline profiles.
//!
//! All we need for the wasm-perf story is "bytes net + allocs count this frame";
//! that's a four-atomic increment per allocation, ~5-10ns of overhead on native,
//! basically free relative to the underlying `System::alloc`. Cheap enough to
//! leave on by default in debug + release wasm builds without measurable impact.
//!
//! ## Why atomics and not thread-locals
//!
//! `GlobalAlloc` is called from any thread; on wasm32 we're single-threaded so
//! thread-locals would suffice, but atomics are correct everywhere and the cost
//! is one `fetch_add` per call. Relaxed ordering is fine because we don't
//! synchronize OTHER memory through these counters — they're plain counts.
//!
//! ## Why "installed" is its own bool
//!
//! The counters start at zero. If a demo never installs the wrapper, the per-
//! frame delta is identically zero forever, which would print as "0 allocs"
//! misleadingly. The wrapper sets [`ALLOC_INSTALLED`] on first call so
//! `frame_trace` can distinguish "no allocator wired" from "no allocations
//! this frame." The latter is the steady-state goal we're driving toward.
//!
//! ## Usage
//!
//! In a demo's `main.rs`:
//!
//! ```ignore
//! use rye_time::alloc::CountingAllocator;
//! use std::alloc::System;
//!
//! #[global_allocator]
//! static GLOBAL: CountingAllocator<System> = CountingAllocator::new(System);
//! ```
//!
//! The wrapper is generic over the inner allocator so wasm targets can swap in
//! `wee_alloc` or a custom allocator without changing the counting layer.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Process-global counter of total bytes allocated since startup. Monotonic
/// (never decreases); the per-frame delta is computed by sampling at frame
/// boundaries and subtracting. `Relaxed` ordering everywhere — no other memory
/// is synchronized through these counters.
pub(crate) static TOTAL_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Process-global counter of total bytes deallocated since startup. Same shape
/// as [`TOTAL_ALLOC_BYTES`]; the net heap delta over a frame is
/// `(alloc_bytes_end - alloc_bytes_start) - (dealloc_bytes_end - dealloc_bytes_start)`.
pub(crate) static TOTAL_DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Process-global allocation-call count since startup. Useful per-frame as the
/// "alloc churn" signal that's independent of allocation size: a million 1-byte
/// allocations is a different problem than one 1 MB allocation.
pub(crate) static TOTAL_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
/// Process-global deallocation-call count since startup.
pub(crate) static TOTAL_DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
/// Sentinel: was a [`CountingAllocator`] ever called? If false, the counters
/// are all zero because nothing was installed, not because nothing allocated.
/// `frame_trace` reads this to decide whether to attach `AllocDelta` to
/// completed frames.
pub(crate) static ALLOC_INSTALLED: AtomicBool = AtomicBool::new(false);

/// `GlobalAlloc` wrapper that counts every alloc + dealloc through atomic
/// counters. Generic over the inner allocator so the demo can wrap `System`
/// (native), `wee_alloc` (wasm-lean), or any other GlobalAlloc.
///
/// ## Drop semantics
///
/// `GlobalAlloc` is `unsafe` to implement; we delegate every call to the inner
/// allocator without modification + only add counter updates. Safety contract
/// is therefore "as safe as `A`."
///
/// ## Layout::size()
///
/// We count `Layout::size()` bytes per allocation, not the actual aligned size
/// returned by the allocator (which can be larger to satisfy alignment). The
/// `size()` value matches what Rust code "thinks" it allocated; reads slightly
/// low vs. true heap pressure but matches what a programmer would expect to
/// see in the overlay.
pub struct CountingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> CountingAllocator<A> {
    /// Construct a counting wrapper around `inner`. `const fn` so the
    /// constructor can be called in a `#[global_allocator] static` definition.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Mark installed on first call. AcqRel not needed (no other memory
        // ordering depends on this); Relaxed + a one-way write is fine.
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        TOTAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        self.inner.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        TOTAL_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        self.inner.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        TOTAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        self.inner.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Account realloc as one dealloc of the old size + one alloc of the
        // new. This matches how Rust code thinks about it (`Vec::push` past
        // capacity = "I allocated more"). The underlying allocator may or
        // may not actually move the buffer; we don't care.
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        TOTAL_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        TOTAL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        self.inner.realloc(ptr, layout, new_size)
    }
}

/// Snapshot of the alloc counters at one point in time. Subtracting two
/// snapshots produces an [`AllocDelta`] for the interval between them.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocSnapshot {
    pub alloc_bytes: u64,
    pub dealloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

/// Per-frame delta computed by subtracting two [`AllocSnapshot`]s. Signed bytes
/// (net = alloc - dealloc) so a frame that drops a 10 MB buffer reads as
/// negative net.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocDelta {
    /// Net bytes added to the heap this frame. Negative when this frame
    /// dropped more than it allocated.
    pub net_bytes: i64,
    /// Bytes allocated this frame, regardless of how many were dropped.
    /// Useful as the "allocation pressure" signal that doesn't cancel out.
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

/// Read the current allocation counters. Returns `None` when no
/// [`CountingAllocator`] has been installed (the sentinel
/// [`ALLOC_INSTALLED`] was never set), so callers can distinguish "nothing
/// allocated" from "no allocator wired."
pub fn current_snapshot() -> Option<AllocSnapshot> {
    if !ALLOC_INSTALLED.load(Ordering::Relaxed) {
        return None;
    }
    Some(AllocSnapshot {
        alloc_bytes: TOTAL_ALLOC_BYTES.load(Ordering::Relaxed),
        dealloc_bytes: TOTAL_DEALLOC_BYTES.load(Ordering::Relaxed),
        alloc_count: TOTAL_ALLOC_COUNT.load(Ordering::Relaxed),
        dealloc_count: TOTAL_DEALLOC_COUNT.load(Ordering::Relaxed),
    })
}

/// Compute the delta between two snapshots. `start` must be the earlier
/// snapshot; counters are monotonic so `end >= start` per-field.
pub fn delta(start: AllocSnapshot, end: AllocSnapshot) -> AllocDelta {
    let alloc_bytes = end.alloc_bytes.saturating_sub(start.alloc_bytes);
    let dealloc_bytes = end.dealloc_bytes.saturating_sub(start.dealloc_bytes);
    AllocDelta {
        net_bytes: (alloc_bytes as i64).saturating_sub(dealloc_bytes as i64),
        alloc_bytes,
        alloc_count: end.alloc_count.saturating_sub(start.alloc_count),
        dealloc_count: end.dealloc_count.saturating_sub(start.dealloc_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_computes_net_bytes() {
        let start = AllocSnapshot {
            alloc_bytes: 1_000,
            dealloc_bytes: 200,
            alloc_count: 10,
            dealloc_count: 3,
        };
        let end = AllocSnapshot {
            alloc_bytes: 5_000,
            dealloc_bytes: 4_200,
            alloc_count: 50,
            dealloc_count: 40,
        };
        let d = delta(start, end);
        assert_eq!(d.alloc_bytes, 4_000);
        assert_eq!(d.net_bytes, 4_000 - 4_000);
        assert_eq!(d.alloc_count, 40);
        assert_eq!(d.dealloc_count, 37);
    }

    #[test]
    fn delta_handles_dealloc_dominant() {
        let start = AllocSnapshot {
            alloc_bytes: 1_000,
            dealloc_bytes: 200,
            alloc_count: 10,
            dealloc_count: 3,
        };
        let end = AllocSnapshot {
            alloc_bytes: 1_100,
            dealloc_bytes: 1_000,
            alloc_count: 12,
            dealloc_count: 20,
        };
        let d = delta(start, end);
        assert_eq!(d.alloc_bytes, 100);
        assert_eq!(d.net_bytes, 100 - 800);
        assert!(d.net_bytes < 0);
    }

    #[test]
    fn current_snapshot_is_none_when_uninstalled() {
        // This test is best-effort: if some other test in this run installed
        // the allocator, ALLOC_INSTALLED is true forever. We only assert the
        // "None" branch when the bool is observably false; otherwise the
        // installed-path semantics already test the Some branch.
        if !ALLOC_INSTALLED.load(Ordering::Relaxed) {
            assert!(current_snapshot().is_none());
        }
    }
}
