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
//! Each frame uses ONE of `FRAMES_IN_FLIGHT` slots in a single query set + resolve
//! buffer (striped). The flow per frame `f`:
//!
//! 1. **Write start.** Just after `begin_frame`: a tiny `gpu-timer-start` encoder
//!    writes `timestamp(slot=2f)` and submits.
//! 2. **Write end + resolve.** Just before `frame.present`: a tiny `gpu-timer-end`
//!    encoder writes `timestamp(slot=2f+1)`, calls `resolve_query_set` to copy the
//!    two u64 ticks into the resolve buffer slice for slot f, and submits.
//! 3. **Schedule map.** The runner's `tick()` call (after queue.submit) schedules
//!    `map_async` on the slot's buffer slice. The callback fires when the GPU has
//!    actually finished the frame; it reads the two u64s, converts ticks ->
//!    nanoseconds via `Queue::get_timestamp_period`, and sends the delta over an
//!    `mpsc` channel.
//! 4. **Drain.** The next frame's `tick()` drains the channel and pushes any
//!    received durations into `rye_time::frame_trace` as `gpu-total` sections,
//!    where they appear in `trace summary` alongside the CPU sections.
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
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, Queue,
    QuerySet, QuerySetDescriptor, QueryType,
};

/// Three in-flight frames is enough headroom for the typical WebGPU latency profile
/// (1 frame submitted, 1 frame on the GPU, 1 frame staged for mapping). More slots
/// would push displayed timings further into the past; fewer risks map-vs-write
/// contention.
const FRAMES_IN_FLIGHT: usize = 3;

/// Bytes per resolved slot pair (two `u64` ticks).
const BYTES_PER_SLOT: u64 = 16;

/// Per-slot state. `in_flight` is set when the slot has been resolved and is awaiting
/// its `map_async` callback. The callback clears it (via the same `Arc`) after the
/// timing has been delivered through the channel.
struct SlotState {
    in_flight: Arc<AtomicBool>,
}

/// Triple-buffered timestamp recorder owned by `RenderDevice`.
pub struct GpuTimer {
    /// One query set with `FRAMES_IN_FLIGHT * 2` slots (start + end per frame).
    query_set: QuerySet,
    /// Single resolve buffer striped into per-slot 16-byte slices. `QUERY_RESOLVE`
    /// for `resolve_query_set` writes, `MAP_READ` for the async callback's reads.
    resolve_buffer: Buffer,
    /// Per-slot in-flight flags. Slot at `frame_index % FRAMES_IN_FLIGHT` is the
    /// current frame's slot.
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
            size: BYTES_PER_SLOT * FRAMES_IN_FLIGHT as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let slots = std::array::from_fn(|_| SlotState {
            in_flight: Arc::new(AtomicBool::new(false)),
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
        let base = slot as u64 * BYTES_PER_SLOT;
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

    /// Write the end-of-frame timestamp + resolve the slot pair into the resolve
    /// buffer. Caller submits the encoder before `frame.present`. Marks the slot
    /// in-flight; the next `tick` schedules its `map_async`.
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
        //    cleared) is harmless — we just won't see it because we skip cleared
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
        let just_resolved_slot =
            (self.frame_index.wrapping_sub(1) as usize) % FRAMES_IN_FLIGHT;
        if !self.slots[just_resolved_slot]
            .in_flight
            .load(Ordering::Acquire)
        {
            return;
        }
        let byte_range = Self::slot_byte_range(just_resolved_slot);
        let buffer = self.resolve_buffer.clone();
        let buffer_for_callback = buffer.clone();
        let period_ns = self.timestamp_period_ns;
        let tx = self.tx.clone();
        let flag = self.slots[just_resolved_slot].in_flight.clone();
        let cb_range = byte_range.clone();
        buffer
            .slice(byte_range)
            .map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    let view = buffer_for_callback.slice(cb_range).get_mapped_range();
                    if view.len() == BYTES_PER_SLOT as usize {
                        let start_ticks = u64::from_le_bytes(
                            view[0..8].try_into().expect("8-byte u64"),
                        );
                        let end_ticks = u64::from_le_bytes(
                            view[8..16].try_into().expect("8-byte u64"),
                        );
                        let delta_ticks = end_ticks.saturating_sub(start_ticks);
                        let delta_ns = (delta_ticks as f64 * period_ns as f64) as u64;
                        let _ = tx.send(Duration::from_nanos(delta_ns));
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
