//! The worker's inbound message queue: the typed input events
//! [`super::messages::parse_non_init`] produces, and the bounded ring the
//! message handler pushes them onto for `WorkerRunner::frame` to drain.
//!
//! Free of `web-sys` and `js-sys` so the queue policy is exercisable off
//! target; see `tests/wasm_input_queue.rs`.

use std::cell::RefCell;
use std::collections::VecDeque;

/// Per-frame inputs the main thread forwards to the worker. Each variant
/// corresponds to a `kind` string in the postMessage payload.
#[derive(Debug)]
pub enum InputMessage {
    /// New canvas pixel dimensions (DPR-multiplied to physical pixels).
    /// Triggers a wgpu surface reconfigure on the next frame.
    Resize { width: u32, height: u32 },

    /// Pointer moved to (x, y) in canvas-local CSS pixels. `buttons` is
    /// the `MouseEvent.buttons` bitmask. `dx`/`dy` are raw
    /// `movementX/Y` deltas for FPS mouse-look, valid both before and
    /// after Pointer Lock engages; coalesced moves sum the dropped
    /// intermediate deltas.
    MouseMove {
        x: f32,
        y: f32,
        buttons: u8,
        dx: f32,
        dy: f32,
    },

    /// Pointer button transitioned. `button` is `MouseEvent.button`
    /// (0=primary, 1=middle, 2=secondary).
    MouseButton {
        x: f32,
        y: f32,
        button: u8,
        pressed: bool,
    },

    /// Wheel delta in lines (normalized on main thread). DOM convention:
    /// positive = right/down.
    MouseWheel { dx: f32, dy: f32 },

    /// Keyboard key transitioned. `code` is the physical-key code (for
    /// hotkey routing via `keymap::keycode_*`); `key` is the logical key
    /// (for text-input fan-out to egui).
    Key {
        code: String,
        key: String,
        pressed: bool,
        repeat: bool,
        ctrl: bool,
        shift: bool,
        alt: bool,
        meta: bool,
    },

    /// Window focus state changed.
    Focus(bool),

    /// Page visibility (tab-in-foreground) state changed.
    Visibility(bool),

    /// Begin the continuous RAF loop. Sent after the user clicks the
    /// launch overlay; before it arrives the worker has rendered one
    /// preview frame for the overlay to blur.
    Start,

    /// Pointer Lock state mirror from the main thread's
    /// `pointerlockchange` event. The worker calls
    /// [`crate::cursor::mark_applied`] so `current_state()` tracks what
    /// the browser actually has.
    PointerLockChanged(bool),
}

/// Hard cap on queued messages. A running frame loop drains every frame
/// and the forwarders coalesce to a few messages per browser frame, so
/// this is dozens of frames of headroom; it is reachable only when
/// nothing is draining (paused embed, halted RAF chain), which is
/// unbounded in wall-clock time.
pub const MESSAGE_QUEUE_CAPACITY: usize = 256;

thread_local! {
    /// Per-worker inbound queue, drained each frame by
    /// `WorkerRunner::frame`. `thread_local` because the message handler
    /// closure has no handle to the asynchronously-constructed runner.
    static MESSAGE_QUEUE: RefCell<VecDeque<InputMessage>> = const { RefCell::new(VecDeque::new()) };
}

/// Push a parsed message onto the per-worker queue, evicting the oldest
/// when full.
///
/// Oldest-first eviction because a full queue means no frame has drained
/// for a long time, so the newest messages carry the state that will
/// still be true on resume. Dropping the newest instead can strand a key
/// release behind its press and leave the key stuck down.
pub fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        if q.len() >= MESSAGE_QUEUE_CAPACITY {
            let dropped = q.pop_front();
            tracing::warn!("loam_app::wasm::worker: input queue full, dropped {dropped:?}");
        }
        q.push_back(msg);
    });
}

/// Drain all queued messages in arrival order. Called once per frame.
pub fn drain_messages() -> Vec<InputMessage> {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().drain(..).collect())
}
