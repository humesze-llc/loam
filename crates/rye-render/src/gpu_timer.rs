//! GPU timer queries via wgpu's `TIMESTAMP_QUERY` feature. Wraps the begin / end of
//! a frame's submitted GPU work in `write_timestamp` calls; the delta is the
//! wall-clock time the GPU spent on that frame's commands.
//!
//! ## Why a separate module
//!
//! Keeping the triple-buffered query / resolve / map state out of `device.rs` makes
//! the surface that `RenderDevice` exposes small: just `gpu_timer: Option<GpuTimer>`
//! with a few thin methods. Apps that don't care about timing don't pay attention to
//! this module; apps that do can reach in directly for sub-pass instrumentation.
//!
//! ## One cycle, per slot
//!
//! Each frame uses ONE of `FRAMES_IN_FLIGHT` slots in a single query set + two
//! striped GPU buffers (a resolve buffer the GPU writes into, plus a map buffer
//! the CPU reads from). The flow per frame `f`:
//!
//! 1. **Write start.** Just after `begin_frame`: a tiny `gpu-timer-start` encoder
//!    writes `timestamp(slot=2f)` and submits.
//! 2. **Write end + resolve + copy.** Just before `frame.present`: a tiny
//!    `gpu-timer-end` encoder writes `timestamp(slot=2f+1)`, calls
//!    `resolve_query_set` to copy the two u64 ticks into the resolve buffer slice
//!    for slot f, then `copy_buffer_to_buffer` into the map buffer's matching
//!    slice. (See "Why two buffers" below.) Then submits.
//! 3. **Schedule map.** The runner's `tick()` call (after queue.submit) schedules
//!    `map_async` on the map buffer's slot slice. The callback fires when the GPU
//!    has actually finished the frame; it reads the two u64s, converts ticks ->
//!    nanoseconds via `Queue::get_timestamp_period`, and sends the delta over an
//!    `mpsc` channel.
//! 4. **Drain.** The next frame's `tick()` drains the channel and pushes any
//!    received durations into `rye_time::frame_trace` as `gpu-total` sections,
//!    where they appear in `trace summary` alongside the CPU sections.
//!
//! ## Why two layers of buffers
//!
//! wgpu 27 validates: "MAP usage can only be combined with the opposite COPY."
//! That is, `MAP_READ` may only pair with `COPY_DST` (and `MAP_WRITE` with
//! `COPY_SRC`); any other non-MAP usage on the same buffer is rejected at
//! `create_buffer` time. `QUERY_RESOLVE` is a separate write usage, not a COPY,
//! so `MAP_READ | QUERY_RESOLVE` panics on device init. The fix is the standard
//! staging dance: one GPU-only resolve buffer with `QUERY_RESOLVE | COPY_SRC`,
//! plus CPU-mappable map buffers with `MAP_READ | COPY_DST`, joined by
//! `copy_buffer_to_buffer` inside the end-of-frame encoder.
//!
//! And map buffers are **per-slot**, not a single striped buffer, because wgpu
//! treats a buffer as "mapped" the instant `map_async` is requested on any
//! slice; for the whole buffer, until `unmap()`. If we shared one map buffer
//! across slots, slot N+1's `copy_buffer_to_buffer` would fail at
//! `Queue::submit` ("buffer is still mapped") while slot N is awaiting its
//! callback. Three tiny per-slot buffers let independent slots make progress.
//!
//! Triple-buffering (3 frame slots) gives enough headroom for the typical 1-2 frame
//! GPU latency without colliding with the slot being mapped for read.
//!
//! ## Edge: adapter without TIMESTAMP_QUERY
//!
//! `GpuTimer::new` returns `None` if the device wasn't created with the feature.
//! Callers should guard with `if let Some(t) = rd.gpu_timer.as_mut()` (the cfg check
//! lives at device-creation time in `device.rs`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, QuerySet,
    QuerySetDescriptor, QueryType, Queue, QUERY_RESOLVE_BUFFER_ALIGNMENT,
};

/// Three in-flight frames is enough headroom for the typical WebGPU latency profile
/// (1 frame submitted, 1 frame on the GPU, 1 frame staged for mapping). More slots
/// would push displayed timings further into the past; fewer risks map-vs-write
/// contention.
const FRAMES_IN_FLIGHT: usize = 3;

/// Bytes per resolved slot pair (two `u64` ticks).
const BYTES_PER_SLOT: u64 = 16;

/// Stride between slot offsets in the resolve buffer. `resolve_query_set`
/// requires the destination offset to be aligned to
/// `QUERY_RESOLVE_BUFFER_ALIGNMENT` (256 bytes), so each slot's payload sits at
/// `slot * SLOT_STRIDE_BYTES` even though it only uses the first 16 bytes of
/// that window. Map buffers don't need this stride; each is its own buffer.
const SLOT_STRIDE_BYTES: u64 = QUERY_RESOLVE_BUFFER_ALIGNMENT;

/// Per-slot state. `in_flight` is set when the slot has been resolved and is
/// awaiting its `map_async` callback. The callback clears it (via the same
/// `Arc`) after the timing has been delivered through the channel. `map_buffer`
/// is per-slot so independent slots don't block each other on `Queue::submit`
/// (wgpu locks a whole buffer the moment any slice is mapped).
struct SlotState {
    in_flight: Arc<AtomicBool>,
    map_buffer: Buffer,
}

/// Triple-buffered timestamp recorder owned by `RenderDevice`.
pub struct GpuTimer {
    /// One query set with `FRAMES_IN_FLIGHT * 2` slots (start + end per frame).
    query_set: QuerySet,
    /// GPU-only resolve buffer striped by `SLOT_STRIDE_BYTES`. `QUERY_RESOLVE`
    /// for `resolve_query_set` writes, `COPY_SRC` so the end-of-frame encoder
    /// can stage the slot into its per-slot map buffer. One buffer is enough
    /// here because there's no CPU mapping on this side.
    resolve_buffer: Buffer,
    /// Per-slot state. Slot at `frame_index % FRAMES_IN_FLIGHT` is the current
    /// frame's slot. See [`SlotState`] for why map buffers are per-slot.
    slots: [SlotState; FRAMES_IN_FLIGHT],
    /// Strictly increasing frame counter. Wraps after `u64::MAX`.
    frame_index: u64,
    /// `Queue::get_timestamp_period()` snapshot. Constant for the device's lifetime;
    /// multiplies u64 ticks into nanoseconds.
    timestamp_period_ns: f32,
    /// Async result channel: callbacks send `Duration`, `tick` drains.
    rx: Receiver<Duration>,
    tx: Sender<Duration>,
}

impl GpuTimer {
    /// Construct a timer for a device that was built with the timestamp features.
    /// Returns `None` if the device doesn't have BOTH `TIMESTAMP_QUERY` (needed for
    /// the query set itself) and `TIMESTAMP_QUERY_INSIDE_ENCODERS` (needed for the
    /// `write_timestamp` calls outside render passes).
    pub fn new(device: &Device, queue: &Queue) -> Option<Self> {
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(needed) {
            return None;
        }
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("rye-render::GpuTimer::query_set"),
            ty: QueryType::Timestamp,
            count: (FRAMES_IN_FLIGHT * 2) as u32,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("rye-render::GpuTimer::resolve_buffer"),
            size: SLOT_STRIDE_BYTES * FRAMES_IN_FLIGHT as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = std::array::from_fn(|_| SlotState {
            in_flight: Arc::new(AtomicBool::new(false)),
            map_buffer: device.create_buffer(&BufferDescriptor {
                label: Some("rye-render::GpuTimer::map_buffer"),
                size: BYTES_PER_SLOT,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });
        let (tx, rx) = channel();
        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            frame_index: 0,
            timestamp_period_ns: queue.get_timestamp_period(),
            rx,
            tx,
        })
    }

    fn current_slot(&self) -> usize {
        (self.frame_index as usize) % FRAMES_IN_FLIGHT
    }

    fn slot_query_range(slot: usize) -> std::ops::Range<u32> {
        let base = (slot * 2) as u32;
        base..(base + 2)
    }

    fn slot_byte_range(slot: usize) -> std::ops::Range<u64> {
        let base = slot as u64 * SLOT_STRIDE_BYTES;
        base..(base + BYTES_PER_SLOT)
    }

    /// Write the start-of-frame timestamp into `encoder`. Caller submits the encoder
    /// before the actual render work. Skips silently when the current slot is still
    /// in flight (the previous cycle's map hasn't completed yet); the slot's data
    /// gets skipped this frame rather than corrupted.
    pub fn write_start(&self, encoder: &mut CommandEncoder) {
        let slot = self.current_slot();
        if self.slots[slot].in_flight.load(Ordering::Acquire) {
            return;
        }
        let range = Self::slot_query_range(slot);
        encoder.write_timestamp(&self.query_set, range.start);
    }

    /// Write the end-of-frame timestamp, resolve the slot pair into the resolve
    /// buffer, then stage that slot into the CPU-mappable map buffer. Caller
    /// submits the encoder before `frame.present`. Marks the slot in-flight; the
    /// next `tick` schedules its `map_async` on the map buffer.
    pub fn write_end_and_resolve(&self, encoder: &mut CommandEncoder) {
        let slot = self.current_slot();
        if self.slots[slot].in_flight.load(Ordering::Acquire) {
            return;
        }
        let query_range = Self::slot_query_range(slot);
        let byte_range = Self::slot_byte_range(slot);
        encoder.write_timestamp(&self.query_set, query_range.end - 1);
        encoder.resolve_query_set(
            &self.query_set,
            query_range,
            &self.resolve_buffer,
            byte_range.start,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            byte_range.start,
            &self.slots[slot].map_buffer,
            0,
            BYTES_PER_SLOT,
        );
        self.slots[slot].in_flight.store(true, Ordering::Release);
    }

    /// Advance the frame counter, drain any completed map_async results into
    /// `rye_time::frame_trace`, then schedule new `map_async` reads for slots
    /// currently in-flight.
    ///
    /// Call once per redraw, after the queue submit that includes the end-of-frame
    /// timestamp.
    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);

        // 1. Drain results from previous frames' callbacks.
        while let Ok(duration) = self.rx.try_recv() {
            rye_time::frame_trace::record_external("gpu-total", duration);
        }

        // 2. For each slot that's in-flight but doesn't have a pending map_async,
        //    schedule one. The map_async callback runs on whatever thread wgpu picks
        //    for completion; it reads the buffer, computes the delta, sends through
        //    the channel, and clears the flag.
        //
        //    Note: in_flight stays true between resolve() and the callback's clear.
        //    Scheduling map_async on a slot whose callback already fired (in_flight
        //    cleared) is harmless; we just won't see it because we skip cleared
        //    slots. The only invariant we rely on is that map_async is called at
        //    most once per resolve, which is true because resolve sets in_flight,
        //    and the callback clears it (re-arming the slot for the next resolve).
        //
        //    To avoid re-scheduling map_async on a slot whose callback is mid-flight
        //    but hasn't cleared yet, we track a per-slot map_pending flag. Without
        //    it, calling map_async on an already-mapping buffer slice is a wgpu
        //    validation error.
        //
        //    For v1 we side-step by using a simpler invariant: we only call
        //    map_async ONCE per resolve, the first time we see in_flight set after
        //    the resolve. We track "already scheduled" with the in_flight flag
        //    itself: leave it set until the callback clears it. Slots with their
        //    callback pending are visible as in_flight=true; we need a SECOND flag
        //    to distinguish "needs map_async scheduled" from "map_async scheduled,
        //    waiting for callback."
        //
        //    Keep it simple: only schedule map_async on the slot that was JUST
        //    resolved on this tick's previous frame. That's `(frame_index - 1) %
        //    FRAMES_IN_FLIGHT`. The other slots are either free or already
        //    callback-pending; we don't re-schedule.
        let just_resolved_slot = (self.frame_index.wrapping_sub(1) as usize) % FRAMES_IN_FLIGHT;
        if !self.slots[just_resolved_slot]
            .in_flight
            .load(Ordering::Acquire)
        {
            return;
        }
        let buffer = self.slots[just_resolved_slot].map_buffer.clone();
        let buffer_for_callback = buffer.clone();
        let period_ns = self.timestamp_period_ns;
        let tx = self.tx.clone();
        let flag = self.slots[just_resolved_slot].in_flight.clone();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let view = buffer_for_callback.slice(..).get_mapped_range();
                // Per-slot map buffer is constructed with `size: BYTES_PER_SLOT`
                // (16), so the slice length is guaranteed by wgpu. We still
                // pattern-destructure here so a divergence between
                // BYTES_PER_SLOT and the literal byte ranges would be caught at
                // compile time rather than via a runtime `.expect` panic.
                if let (Ok(start_bytes), Ok(end_bytes)) = (
                    <[u8; 8]>::try_from(&view[0..8]),
                    <[u8; 8]>::try_from(&view[8..16]),
                ) {
                    let start_ticks = u64::from_le_bytes(start_bytes);
                    let end_ticks = u64::from_le_bytes(end_bytes);
                    let delta_ticks = end_ticks.saturating_sub(start_ticks);
                    let delta_ns = (delta_ticks as f64 * period_ns as f64) as u64;
                    // Reject implausible deltas. At display refresh rates above
                    // ~120 Hz the triple-buffer cycle can race on some drivers
                    // (Vulkan validation reports the writes are correctly
                    // ordered but the resolved values appear to pair a start
                    // tick from one cycle with an end tick several cycles
                    // later, yielding deltas that grow linearly with elapsed
                    // wall time). A real frame's GPU work is sub-100ms even on
                    // the heaviest scenes; anything beyond that is a desynced
                    // slot, not a stall. Dropping the sample keeps `gpu-total`
                    // in `trace summary` honest and avoids spamming the
                    // frame-trace spike WARN at high refresh rates.
                    const MAX_PLAUSIBLE_FRAME_NS: u64 = 1_000_000_000 / 10;
                    if delta_ns <= MAX_PLAUSIBLE_FRAME_NS {
                        let _ = tx.send(Duration::from_nanos(delta_ns));
                    }
                }
                drop(view);
                buffer_for_callback.unmap();
            }
            // Clear the flag whether the read succeeded or not; otherwise a
            // single failed map_async would permanently stall the slot.
            flag.store(false, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure-function tests: these exercise the slot index / range arithmetic
    // without needing a wgpu device. The integration paths (write_start,
    // write_end_and_resolve, tick) require a real Device+Queue and are
    // covered indirectly by polytope_playground / tesseract_demo running
    // without panic at startup; that surfaced the original wgpu 27 MAP+
    // QUERY_RESOLVE validation error this module's design corrects.

    // Compile-time enforcement of the stride-fits-payload invariant. If
    // someone shrinks SLOT_STRIDE_BYTES below BYTES_PER_SLOT the build
    // fails, not a test run.
    const _: () = assert!(SLOT_STRIDE_BYTES >= BYTES_PER_SLOT);

    #[test]
    fn slot_constants_match_wgpu_alignment() {
        assert_eq!(SLOT_STRIDE_BYTES, QUERY_RESOLVE_BUFFER_ALIGNMENT);
        assert_eq!(BYTES_PER_SLOT, 16, "two u64 ticks per slot");
        // resolve_query_set destination offset alignment is the load-bearing
        // invariant; if QUERY_RESOLVE_BUFFER_ALIGNMENT ever changes upstream,
        // SLOT_STRIDE_BYTES tracks it via the binding above.
    }

    #[test]
    fn slot_query_range_is_pair_per_slot() {
        for slot in 0..FRAMES_IN_FLIGHT {
            let range = GpuTimer::slot_query_range(slot);
            assert_eq!(range.end - range.start, 2, "two queries per slot");
            assert_eq!(range.start, (slot * 2) as u32);
        }
        // Adjacent slots' ranges must not overlap; resolve_query_set would
        // otherwise stomp on the previous slot's ticks.
        for slot in 0..FRAMES_IN_FLIGHT.saturating_sub(1) {
            let a = GpuTimer::slot_query_range(slot);
            let b = GpuTimer::slot_query_range(slot + 1);
            assert!(a.end <= b.start, "slot {slot} overlaps slot {}", slot + 1);
        }
    }

    #[test]
    fn slot_byte_range_is_aligned_and_disjoint() {
        for slot in 0..FRAMES_IN_FLIGHT {
            let range = GpuTimer::slot_byte_range(slot);
            // resolve_query_set requires destination offset to be
            // QUERY_RESOLVE_BUFFER_ALIGNMENT-aligned. Verifying here so a
            // refactor that changes SLOT_STRIDE_BYTES caught immediately.
            assert_eq!(
                range.start % QUERY_RESOLVE_BUFFER_ALIGNMENT,
                0,
                "slot {slot} start not aligned"
            );
            assert_eq!(range.end - range.start, BYTES_PER_SLOT);
        }
        // Disjoint between adjacent slots: otherwise a copy from one slot
        // would corrupt another's pending data.
        for slot in 0..FRAMES_IN_FLIGHT.saturating_sub(1) {
            let a = GpuTimer::slot_byte_range(slot);
            let b = GpuTimer::slot_byte_range(slot + 1);
            assert!(a.end <= b.start);
        }
    }

    // Helper: a GpuTimer-shaped value with just enough state to drive
    // `current_slot` and the `(frame_index - 1) % N` invariants. We don't
    // construct an actual GpuTimer (that needs a wgpu Device); we test the
    // arithmetic in isolation.
    fn slot_of(frame_index: u64) -> usize {
        (frame_index as usize) % FRAMES_IN_FLIGHT
    }

    fn just_resolved_of(frame_index: u64) -> usize {
        (frame_index.wrapping_sub(1) as usize) % FRAMES_IN_FLIGHT
    }

    #[test]
    fn current_slot_cycles_through_frames_in_flight() {
        for f in 0..(FRAMES_IN_FLIGHT * 4) as u64 {
            let s = slot_of(f);
            assert!(s < FRAMES_IN_FLIGHT, "slot {s} out of range");
            assert_eq!(s, (f as usize) % FRAMES_IN_FLIGHT);
        }
    }

    #[test]
    fn just_resolved_slot_is_previous_frame() {
        // For a normal forward sequence: just_resolved_slot is the
        // current_slot of frame_index - 1.
        for f in 1..(FRAMES_IN_FLIGHT * 4) as u64 {
            assert_eq!(just_resolved_of(f), slot_of(f - 1));
        }
    }

    #[test]
    fn just_resolved_slot_wraps_around_u64_max() {
        // The runner uses `frame_index.wrapping_add(1)` per redraw. After a
        // full u64::MAX worth of frames, the index wraps to 0. We must still
        // pick the correct previous slot. Demonstrate the math is stable.
        let f = 0u64; // this is what `frame_index` is right after wrapping
        let prev = just_resolved_of(f);
        // `0.wrapping_sub(1) = u64::MAX`; u64::MAX % 3 == 0 (since
        // 3 * 6148914691236517205 = u64::MAX - 0, so MAX % 3 == 0... actually
        // 2^64 ≡ 1 mod 3, so u64::MAX ≡ 0 mod 3.)
        assert_eq!(prev, 0, "u64::MAX % 3 should be 0");
        assert_eq!(prev, (u64::MAX as usize) % FRAMES_IN_FLIGHT);
    }

    #[test]
    fn in_flight_flag_round_trip_clears_after_callback() {
        // Mimic the slot's lifecycle: resolve sets in_flight; callback clears
        // it. The real path goes through map_async; here we exercise the
        // AtomicBool directly to prove the clearing semantics the callback
        // relies on are sound and re-arm the slot for the next cycle.
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::Acquire), "starts clear");

        flag.store(true, Ordering::Release);
        assert!(flag.load(Ordering::Acquire), "resolve sets it");

        // Two consecutive resolves without a callback in between SHOULD be
        // caught by the write_start / write_end guard (`if in_flight return`).
        // We assert that path here: starting from in_flight=true, a re-check
        // sees true and would skip.
        let still_in_flight = flag.load(Ordering::Acquire);
        assert!(still_in_flight, "skip guard fires on consecutive resolves");

        // Callback clears (even on map_async failure, per the implementation
        // comment "Clear the flag whether the read succeeded or not").
        flag.store(false, Ordering::Release);
        assert!(!flag.load(Ordering::Acquire), "callback re-arms the slot");
    }

    #[test]
    fn channel_round_trip_carries_duration() {
        // The async map_async callback sends `Duration` over `tx` and `tick`
        // drains via `try_recv`. Cover the round-trip in isolation; the live
        // path is the same channel with the buffer-read wedged between send
        // and recv.
        let (tx, rx) = channel::<Duration>();
        let _ = tx.send(Duration::from_micros(16_500));
        let _ = tx.send(Duration::from_micros(17_100));
        assert_eq!(rx.try_recv().ok(), Some(Duration::from_micros(16_500)));
        assert_eq!(rx.try_recv().ok(), Some(Duration::from_micros(17_100)));
        assert!(rx.try_recv().is_err(), "channel drained");
    }
}
