//! Typed protocol for postMessage traffic between main thread and worker.
//!
//! Main thread builds `{kind: "...", ...}` JS objects; the worker parses
//! non-init messages via [`parse_non_init`] into [`InputMessage`] and
//! queues them for `WorkerRunner::frame`. The `init` kind is handled by
//! the worker entry (it carries an `OffscreenCanvas` transferable and
//! triggers the one-time async wgpu+App setup).

use anyhow::Result;
use wasm_bindgen::JsValue;

use super::input_queue::InputMessage;

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
