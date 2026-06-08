//! Per-frame allocation counter via a [`GlobalAlloc`] wrapper. Demos opt in by
//! installing [`CountingAllocator`] as their `#[global_allocator]`; every
//! alloc/dealloc bumps process-global atomic counters and the per-frame delta is
//! surfaced through `rye_time::frame_trace::FrameTrace`.
//!
//! Lighter than `dhat`/`tracking-allocator` (offline, file-based): all we need is
//! net bytes + alloc count this frame, a four-atomic increment per call (~5-10ns),
//! cheap enough to leave on in release wasm. Atomics not thread-locals so the
//! wrapper is correct off the single wasm thread too; `Relaxed` since no other
//! memory is synchronized through these counts. `ALLOC_INSTALLED` lets
//! `frame_trace` distinguish "no allocator wired" from "no allocations this frame."
//!
//! ```ignore
//! use rye_time::alloc::CountingAllocator;
//! use std::alloc::System;
//!
//! #[global_allocator]
//! static GLOBAL: CountingAllocator<System> = CountingAllocator::new(System);
//! ```
//!
//! Generic over the inner allocator so wasm can swap in `wee_alloc`.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Total bytes allocated since startup. Monotonic; per-frame delta is sampled at
/// frame boundaries and subtracted.
pub(crate) static TOTAL_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Total bytes deallocated since startup; net heap delta is alloc-delta minus
/// dealloc-delta.
pub(crate) static TOTAL_DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Allocation-call count since startup; the size-independent "churn" signal.
pub(crate) static TOTAL_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
/// Deallocation-call count since startup.
pub(crate) static TOTAL_DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
/// Sentinel: was a [`CountingAllocator`] ever called? Distinguishes "no allocator
/// wired" (counters all zero) from "no allocations." Read by `frame_trace`.
pub(crate) static ALLOC_INSTALLED: AtomicBool = AtomicBool::new(false);

/// `GlobalAlloc` wrapper that counts every alloc/dealloc through atomic counters.
/// Generic over the inner allocator. Calls delegate unmodified, so it is "as safe
/// as `A`." Counts `Layout::size()` (what Rust code thinks it allocated), not the
/// aligned size; reads slightly low vs true heap pressure but matches expectation.
pub struct CountingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> CountingAllocator<A> {
    /// Wrap `inner`. `const fn` so it can be called in a `#[global_allocator]`
    /// static.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
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
        // One dealloc of the old size + one alloc of the new, matching how Rust
        // code thinks of it; whether the buffer actually moves is irrelevant.
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

/// Per-frame delta from subtracting two [`AllocSnapshot`]s. `net_bytes` is signed
/// (alloc - dealloc) so a frame that drops more than it allocates reads negative.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocDelta {
    /// Net bytes added this frame; negative when this frame dropped more.
    pub net_bytes: i64,
    /// Bytes allocated this frame; the pressure signal that doesn't cancel out.
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

/// Read the current allocation counters, or `None` when no [`CountingAllocator`]
/// has been installed.
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

/// Delta between two snapshots; `start` must be the earlier one (counters are
/// monotonic, so `end >= start` per-field).
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
        // Best-effort: ALLOC_INSTALLED is sticky, so only assert the None branch
        // when the bool is observably false.
        if !ALLOC_INSTALLED.load(Ordering::Relaxed) {
            assert!(current_snapshot().is_none());
        }
    }
}
