//! Typed protocol for postMessage traffic between main thread and worker.
//!
//! Main thread builds `{kind: "...", ...}` JS objects in
//! `wasm/main_launcher.rs::install_dom_input_forwarders`; the worker
//! receives them in its `message` event listener and dispatches by
//! `kind` string. Non-init messages are parsed via
//! [`parse_non_init`] into typed `InputMessage` variants and pushed
//! onto a per-worker thread_local queue, drained at the top of each
//! frame by `WorkerRunner::frame`.
//!
//! The `init` kind is special-cased by the worker entry: it carries
//! an `OffscreenCanvas` transferable + initial dimensions and triggers
//! the one-time async wgpu+App setup. The other kinds are pure data.
//!
//! Allocation note: the `Key` variant holds two `String`s (`code` and
//! `key`). Keyboard events are bursty but rare relative to mouse-move
//! events, so the string allocation cost is acceptable. Mouse-move
//! events take a fixed-size payload (three numbers) so they're
//! allocation-free.

use anyhow::Result;
use std::cell::RefCell;
use std::collections::VecDeque;
use wasm_bindgen::JsValue;

/// Per-frame inputs the main thread forwards to the worker. Each variant
/// corresponds to a `kind` string in the postMessage JS object payload.
/// The worker-side dispatcher parses these and pushes them onto a
/// thread_local queue; `WorkerRunner::frame` drains the queue at the
/// top of each frame and applies the events.
#[derive(Debug)]
pub enum InputMessage {
    /// New canvas pixel dimensions. Sent by main thread on window resize
    /// (DPR-multiplied to physical pixels). Triggers a wgpu surface
    /// reconfigure on the next frame.
    Resize { width: u32, height: u32 },

    /// Pointer moved to (x, y) in canvas-local CSS pixels. `buttons` is
    /// the standard `MouseEvent.buttons` bitmask (1=primary, 2=secondary,
    /// 4=middle).
    MouseMove { x: f32, y: f32, buttons: u8 },

    /// Pointer button transitioned. `button` is the standard
    /// `MouseEvent.button` (0=primary, 1=middle, 2=secondary).
    MouseButton {
        x: f32,
        y: f32,
        button: u8,
        pressed: bool,
    },

    /// Wheel delta in lines (after the browser's pixel/line normalization
    /// on main thread). `dx`/`dy` follow DOM convention: positive =
    /// right/down.
    MouseWheel { dx: f32, dy: f32 },

    /// Keyboard key transitioned. `code` is the physical-key code (e.g.
    /// "KeyT", "Space"); `key` is the logical key (e.g. "t", " ").
    /// `code` is used for hotkey routing via `keymap::keycode_*`; `key`
    /// is used for text-input fan-out to egui.
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

    /// Begin the continuous RAF loop. Sent by main thread after the
    /// user clicks the launch overlay. Before this arrives, the worker
    /// has already initialized + rendered ONE preview frame (so the
    /// launch overlay can blur the demo's actual first frame instead
    /// of a gradient placeholder); the RAF loop only starts once
    /// `Start` lands.
    Start,
}

thread_local! {
    /// Per-worker queue of inbound input messages. Drained at the top
    /// of every frame by `WorkerRunner::frame`. A `thread_local` because
    /// the message handler closure doesn't have direct access to the
    /// runner (the runner is constructed asynchronously, after the
    /// first init message arrives).
    static MESSAGE_QUEUE: RefCell<VecDeque<InputMessage>> = RefCell::new(VecDeque::new());
}

/// Push a message onto the per-worker queue. Called from the worker's
/// `message` event listener after parsing.
pub fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().push_back(msg));
}

/// Drain all queued messages. Returns them in arrival order so the
/// runner can apply them sequentially. Called once per frame.
pub fn drain_messages() -> Vec<InputMessage> {
    MESSAGE_QUEUE.with(|q| q.borrow_mut().drain(..).collect())
}

/// Parse a non-init postMessage payload into an `InputMessage`.
///
/// Returns:
/// - `Ok(Some(msg))`: parsed successfully
/// - `Ok(None)`: message kind is "init" (caller handles specially) OR
///   the kind is unrecognized (we don't error on unknown kinds; the
///   worker just logs and drops them)
/// - `Err(_)`: the payload itself is malformed (missing `kind` field)
///
/// The `init` kind is intentionally NOT handled here because parsing it
/// requires extracting an `OffscreenCanvas` transferable + triggering
/// the async wgpu setup, both of which depend on type-parameter `A`
/// (the App type) that this non-generic module doesn't have.
pub fn parse_non_init(data: &JsValue) -> Result<Option<InputMessage>> {
    let kind = js_sys::Reflect::get(data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|v| v.as_string());

    let kind = kind.ok_or_else(|| anyhow::anyhow!("postMessage missing 'kind' field"))?;

    let msg = match kind.as_str() {
        "init" => return Ok(None), // caller (worker.rs) handles
        "resize" => InputMessage::Resize {
            width: read_u32_field(data, "width").unwrap_or(0),
            height: read_u32_field(data, "height").unwrap_or(0),
        },
        "mouse_move" => InputMessage::MouseMove {
            x: read_f32_field(data, "x").unwrap_or(0.0),
            y: read_f32_field(data, "y").unwrap_or(0.0),
            buttons: read_u32_field(data, "buttons").unwrap_or(0) as u8,
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
        "visibility" => {
            InputMessage::Visibility(read_bool_field(data, "visible").unwrap_or(false))
        }
        "start" => InputMessage::Start,
        _ => return Ok(None), // unknown kind; caller decides to warn or ignore
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
