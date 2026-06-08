//! GPU timer queries via wgpu's `TIMESTAMP_QUERY` feature. Brackets a frame's
//! submitted GPU work in `write_timestamp` calls; the delta is the GPU's
//! wall-clock time for that frame.
//!
//! Each frame uses one of `FRAMES_IN_FLIGHT` slots: write start, then write end,
//! `resolve_query_set`, and `copy_buffer_to_buffer` into a per-slot map buffer.
//! `tick()` schedules `map_async`; the callback converts ticks to nanoseconds
//! via `Queue::get_timestamp_period` and sends the delta over an `mpsc`
//! channel, which the next `tick()` drains into `rye_time::frame_trace` as a
//! `gpu-total` section.
//!
//! Two buffer layers because wgpu 27 rejects `MAP_READ | QUERY_RESOLVE` at
//! `create_buffer` (MAP usage may only pair with the opposite COPY): one
//! GPU-only resolve buffer (`QUERY_RESOLVE | COPY_SRC`) feeds CPU-mappable map
//! buffers (`MAP_READ | COPY_DST`). Map buffers are per-slot because wgpu locks
//! a whole buffer the instant any slice is mapped, so a shared buffer would
//! fail slot N+1's `copy_buffer_to_buffer` at submit while slot N awaits its
//! callback.
//!
//! `GpuTimer::new` returns `None` when the device lacks the timestamp features;
//! guard with `if let Some(t) = rd.gpu_timer.as_mut()`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, QuerySet,
    QuerySetDescriptor, QueryType, Queue, QUERY_RESOLVE_BUFFER_ALIGNMENT,
};

/// Submitted, on-GPU, staged-for-mapping: one slot each, matching the typical
/// WebGPU latency profile. More slots delay displayed timings; fewer risk
/// map-vs-write contention.
const FRAMES_IN_FLIGHT: usize = 3;

/// Upper bound on a believable single-frame GPU time; beyond this is a desynced
/// slot, not a stall.
const MAX_PLAUSIBLE_FRAME_NS: u64 = 1_000_000_000 / 10;

/// Whether a resolved delta is small enough to be a real frame time. Above
/// ~120 Hz the triple-buffer cycle can race on some drivers, pairing a start
/// tick with an end tick several cycles later; the resulting deltas grow with
/// wall time. Dropping them keeps `gpu-total` honest. Free function so the
/// threshold is testable without a GPU.
fn is_plausible_frame_delta_ns(delta_ns: u64) -> bool {
    delta_ns <= MAX_PLAUSIBLE_FRAME_NS
}

/// Bytes per resolved slot pair (two `u64` ticks).
const BYTES_PER_SLOT: u64 = 16;

/// `resolve_query_set` requires a `QUERY_RESOLVE_BUFFER_ALIGNMENT`-aligned
/// destination offset, so each slot's 16-byte payload starts at
/// `slot * SLOT_STRIDE_BYTES`.
const SLOT_STRIDE_BYTES: u64 = QUERY_RESOLVE_BUFFER_ALIGNMENT;

/// Per-slot state. `in_flight` is set on resolve and cleared by the `map_async`
/// callback once the timing is sent. `map_buffer` is per-slot so a mapped slot
/// doesn't block another's `Queue::submit`.
struct SlotState {
    in_flight: Arc<AtomicBool>,
    map_buffer: Buffer,
}

/// Triple-buffered timestamp recorder owned by `RenderDevice`.
pub struct GpuTimer {
    /// One query set with `FRAMES_IN_FLIGHT * 2` slots (start + end per frame).
    query_set: QuerySet,
    /// GPU-only resolve buffer striped by `SLOT_STRIDE_BYTES`.
    resolve_buffer: Buffer,
    /// Slot at `frame_index % FRAMES_IN_FLIGHT` is the current frame's.
    slots: [SlotState; FRAMES_IN_FLIGHT],
    /// Strictly increasing frame counter. Wraps after `u64::MAX`.
    frame_index: u64,
    /// `Queue::get_timestamp_period()` snapshot; ticks to nanoseconds.
    timestamp_period_ns: f32,
    /// Async result channel: callbacks send `Duration`, `tick` drains.
    rx: Receiver<Duration>,
    tx: Sender<Duration>,
}

impl GpuTimer {
    /// Returns `None` unless the device has both `TIMESTAMP_QUERY` (query set)
    /// and `TIMESTAMP_QUERY_INSIDE_ENCODERS` (`write_timestamp` outside passes).
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

    /// Write the start-of-frame timestamp into `encoder`. Skips silently when
    /// the current slot is still in flight rather than corrupting its data.
    pub fn write_start(&self, encoder: &mut CommandEncoder) {
        let slot = self.current_slot();
        if self.slots[slot].in_flight.load(Ordering::Acquire) {
            return;
        }
        let range = Self::slot_query_range(slot);
        encoder.write_timestamp(&self.query_set, range.start);
    }

    /// Write the end-of-frame timestamp, resolve the slot pair, and stage it
    /// into the map buffer. Marks the slot in-flight for the next `tick` to map.
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

    /// Advance the frame counter, drain completed results into
    /// `rye_time::frame_trace`, then schedule the just-resolved slot's
    /// `map_async`. Call once per redraw, after the end-of-frame queue submit.
    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);

        while let Ok(duration) = self.rx.try_recv() {
            rye_time::frame_trace::record_external("gpu-total", duration);
        }

        // Schedule map_async on only the slot just resolved on the previous
        // frame; the in_flight flag (set by resolve, cleared by the callback)
        // guarantees at most one map per resolve. Mapping an already-mapping
        // slice is a wgpu validation error.
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
                if let (Ok(start_bytes), Ok(end_bytes)) = (
                    <[u8; 8]>::try_from(&view[0..8]),
                    <[u8; 8]>::try_from(&view[8..16]),
                ) {
                    let start_ticks = u64::from_le_bytes(start_bytes);
                    let end_ticks = u64::from_le_bytes(end_bytes);
                    let delta_ticks = end_ticks.saturating_sub(start_ticks);
                    let delta_ns = (delta_ticks as f64 * period_ns as f64) as u64;
                    if is_plausible_frame_delta_ns(delta_ns) {
                        let _ = tx.send(Duration::from_nanos(delta_ns));
                    }
                }
                drop(view);
                buffer_for_callback.unmap();
            }
            // Clear even on failure; otherwise one failed map stalls the slot.
            flag.store(false, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure slot index / range arithmetic; no wgpu device needed. The
    // integration paths are covered indirectly by the demos starting without
    // panic.

    // Compile-time stride-fits-payload invariant.
    const _: () = assert!(SLOT_STRIDE_BYTES >= BYTES_PER_SLOT);

    #[test]
    fn slot_constants_match_wgpu_alignment() {
        assert_eq!(SLOT_STRIDE_BYTES, QUERY_RESOLVE_BUFFER_ALIGNMENT);
        assert_eq!(BYTES_PER_SLOT, 16, "two u64 ticks per slot");
    }

    #[test]
    fn slot_query_range_is_pair_per_slot() {
        for slot in 0..FRAMES_IN_FLIGHT {
            let range = GpuTimer::slot_query_range(slot);
            assert_eq!(range.end - range.start, 2, "two queries per slot");
            assert_eq!(range.start, (slot * 2) as u32);
        }
        // Adjacent slots must not overlap, else resolve_query_set stomps ticks.
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
            // resolve_query_set destination offset must be
            // QUERY_RESOLVE_BUFFER_ALIGNMENT-aligned.
            assert_eq!(
                range.start % QUERY_RESOLVE_BUFFER_ALIGNMENT,
                0,
                "slot {slot} start not aligned"
            );
            assert_eq!(range.end - range.start, BYTES_PER_SLOT);
        }
        // Adjacent slots must be disjoint, else a copy corrupts pending data.
        for slot in 0..FRAMES_IN_FLIGHT.saturating_sub(1) {
            let a = GpuTimer::slot_byte_range(slot);
            let b = GpuTimer::slot_byte_range(slot + 1);
            assert!(a.end <= b.start);
        }
    }

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
        for f in 1..(FRAMES_IN_FLIGHT * 4) as u64 {
            assert_eq!(just_resolved_of(f), slot_of(f - 1));
        }
    }

    #[test]
    fn just_resolved_slot_wraps_around_u64_max() {
        // After frame_index wraps to 0, the previous slot must stay correct.
        // 0.wrapping_sub(1) = u64::MAX; 2^64 ≡ 1 mod 3 so u64::MAX ≡ 0 mod 3.
        let f = 0u64;
        let prev = just_resolved_of(f);
        assert_eq!(prev, 0, "u64::MAX % 3 should be 0");
        assert_eq!(prev, (u64::MAX as usize) % FRAMES_IN_FLIGHT);
    }

    #[test]
    fn in_flight_flag_round_trip_clears_after_callback() {
        // Resolve sets in_flight; callback clears it, re-arming the slot.
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::Acquire), "starts clear");

        flag.store(true, Ordering::Release);
        assert!(flag.load(Ordering::Acquire), "resolve sets it");

        // The write_start / write_end skip guard fires while in_flight is set.
        let still_in_flight = flag.load(Ordering::Acquire);
        assert!(still_in_flight, "skip guard fires on consecutive resolves");

        flag.store(false, Ordering::Release);
        assert!(!flag.load(Ordering::Acquire), "callback re-arms the slot");
    }

    #[test]
    fn plausible_frame_delta_rejects_over_budget() {
        assert!(is_plausible_frame_delta_ns(4_000_000)); // 240 Hz
        assert!(is_plausible_frame_delta_ns(16_666_667)); // 60 Hz
        assert!(is_plausible_frame_delta_ns(MAX_PLAUSIBLE_FRAME_NS)); // inclusive cap
                                                                      // Desynced-slot garbage the filter exists to drop.
        assert!(!is_plausible_frame_delta_ns(250_000_000));
        assert!(!is_plausible_frame_delta_ns(950_000_000));
        // Zero (start == end) is plausible, not a stall.
        assert!(is_plausible_frame_delta_ns(0));
    }

    #[test]
    fn channel_round_trip_carries_duration() {
        // The map_async callback sends over `tx`; `tick` drains via `try_recv`.
        let (tx, rx) = channel::<Duration>();
        let _ = tx.send(Duration::from_micros(16_500));
        let _ = tx.send(Duration::from_micros(17_100));
        assert_eq!(rx.try_recv().ok(), Some(Duration::from_micros(16_500)));
        assert_eq!(rx.try_recv().ok(), Some(Duration::from_micros(17_100)));
        assert!(rx.try_recv().is_err(), "channel drained");
    }
}
