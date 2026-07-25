//! Worker inbound-queue policy. `crate::wasm` is wasm32-gated, so the
//! module is included by path: the bound is plain data-structure policy
//! and this pins the shipping source rather than a copy of it.

#[path = "../src/wasm/input_queue.rs"]
// Variants this test has no reason to construct.
#[allow(dead_code)]
mod input_queue;

use input_queue::{drain_messages, enqueue, InputMessage, MESSAGE_QUEUE_CAPACITY};

/// A queue nothing drains (paused embed, halted RAF chain) must not grow
/// without bound, and the survivors must be the newest arrivals in order:
/// evicting the newest instead would strand a key release behind its
/// press and leave the key stuck down on resume.
#[test]
fn queue_caps_at_capacity_keeping_newest_in_arrival_order() {
    const OVERFLOW: u32 = 37;
    for width in 0..(MESSAGE_QUEUE_CAPACITY as u32 + OVERFLOW) {
        enqueue(InputMessage::Resize { width, height: 0 });
    }

    let drained = drain_messages();
    assert_eq!(drained.len(), MESSAGE_QUEUE_CAPACITY);

    let widths: Vec<u32> = drained
        .iter()
        .map(|msg| match msg {
            InputMessage::Resize { width, .. } => *width,
            other => panic!("unexpected message {other:?}"),
        })
        .collect();
    let expected: Vec<u32> = (OVERFLOW..OVERFLOW + MESSAGE_QUEUE_CAPACITY as u32).collect();
    assert_eq!(widths, expected);

    assert!(drain_messages().is_empty());
}
