//! Typed protocol for postMessage traffic between main thread and worker.
//!
//! Main thread builds `{kind: "...", ...}` JS objects; the worker parses
//! non-init messages via [`parse_non_init`] into [`InputMessage`] and
//! queues them for `WorkerRunner::frame`. The `init` kind is handled by
//! the worker entry (it carries an `OffscreenCanvas` transferable and
//! triggers the one-time async wgpu+App setup).

use anyhow::Result;
use std::cell::RefCell;
use std::collections::VecDeque;
use wasm_bindgen::JsValue;

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

thread_local! {
    /// Per-worker inbound queue, drained each frame by
    /// `WorkerRunner::frame`. `thread_local` because the message handler
    /// closure has no handle to the asynchronously-constructed runner.
    static MESSAGE_QUEUE: RefCell<VecDeque<InputMessage>> = RefCell::new(VecDeque::new());
}

/// Push a parsed message onto the per-worker queue.
pub fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().push_back(msg));
}

/// Drain all queued messages in arrival order. Called once per frame.
pub fn drain_messages() -> Vec<InputMessage> {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().drain(..).collect())
}

/// Parse a non-init postMessage payload into an [`InputMessage`].
///
/// `Ok(None)` covers both the "init" kind (caller handles) and unknown
/// kinds (logged and dropped). `Err` means a malformed payload (no
/// `kind` field). "init" is excluded here because its parse depends on
/// the App type parameter this non-generic module lacks.
pub fn parse_non_init(data: &JsValue) -> Result<Option<InputMessage>> {
    let kind = js_sys::Reflect::get(data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|v| v.as_string());

    let kind = kind.ok_or_else(|| anyhow::anyhow!("postMessage missing 'kind' field"))?;

    let msg = match kind.as_str() {
        "init" => return Ok(None),
        "resize" => InputMessage::Resize {
            width: read_u32_field(data, "width").unwrap_or(0),
            height: read_u32_field(data, "height").unwrap_or(0),
        },
        "mouse_move" => InputMessage::MouseMove {
            x: read_f32_field(data, "x").unwrap_or(0.0),
            y: read_f32_field(data, "y").unwrap_or(0.0),
            buttons: read_u32_field(data, "buttons").unwrap_or(0) as u8,
            dx: read_f32_field(data, "dx").unwrap_or(0.0),
            dy: read_f32_field(data, "dy").unwrap_or(0.0),
        },
        "mouse_button" => InputMessage::MouseButton {
            x: read_f32_field(data, "x").unwrap_or(0.0),
            y: read_f32_field(data, "y").unwrap_or(0.0),
            button: read_u32_field(data, "button").unwrap_or(0) as u8,
            pressed: read_bool_field(data, "pressed").unwrap_or(false),
        },
        "mouse_wheel" => InputMessage::MouseWheel {
            dx: read_f32_field(data, "dx").unwrap_or(0.0),
            dy: read_f32_field(data, "dy").unwrap_or(0.0),
        },
        "key" => InputMessage::Key {
            code: read_string_field(data, "code").unwrap_or_default(),
            key: read_string_field(data, "key").unwrap_or_default(),
            pressed: read_bool_field(data, "pressed").unwrap_or(false),
            repeat: read_bool_field(data, "repeat").unwrap_or(false),
            ctrl: read_bool_field(data, "ctrl").unwrap_or(false),
            shift: read_bool_field(data, "shift").unwrap_or(false),
            alt: read_bool_field(data, "alt").unwrap_or(false),
            meta: read_bool_field(data, "meta").unwrap_or(false),
        },
        "focus" => InputMessage::Focus(read_bool_field(data, "focused").unwrap_or(false)),
        "visibility" => InputMessage::Visibility(read_bool_field(data, "visible").unwrap_or(false)),
        "start" => InputMessage::Start,
        "pointer_lock_changed" => {
            InputMessage::PointerLockChanged(read_bool_field(data, "locked").unwrap_or(false))
        }
        _ => return Ok(None), // unknown kind; logged and dropped by caller
    };

    Ok(Some(msg))
}

// ---------------------------------------------------------------------------
// JS-object field readers
// ---------------------------------------------------------------------------

fn read_u32_field(obj: &JsValue, key: &str) -> Option<u32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|f| f as u32)
}

fn read_f32_field(obj: &JsValue, key: &str) -> Option<f32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
}

fn read_bool_field(obj: &JsValue, key: &str) -> Option<bool> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_bool())
}

fn read_string_field(obj: &JsValue, key: &str) -> Option<String> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}
