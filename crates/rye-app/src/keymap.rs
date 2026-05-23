//! Translation tables between DOM event identifiers and the typed
//! event enums used by `winit` (for rye-input's `InputState`) and `egui`
//! (for `WorkerUi`'s `RawInput`).
//!
//! Why two tables: the receiver libraries each have their own key enum
//! (`winit::keyboard::KeyCode`, `egui::Key`). They overlap in coverage
//! but diverge in naming conventions (`winit::KeyCode::KeyA` vs
//! `egui::Key::A`), so a single shared table would lose information.
//!
//! Coverage is **partial**: every key tesseract_demo + polytope_playground
//! plausibly touch is here, plus the standard set (alpha + digits +
//! function keys + arrows + common control keys). Unmapped codes return
//! `None`; the caller silently drops the event. Adding a key is one
//! line per table.
//!
//! All functions are `const`-fn-friendly in spirit (one giant `match`
//! lookup; no allocation). Pure utilities with no dependency on the
//! rest of the worker module surface.

use rye_egui::egui;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

/// Map a DOM `MouseEvent.button` index to a `winit::event::MouseButton`.
///
/// DOM convention: 0=primary, 1=middle, 2=secondary, 3=back, 4=forward.
/// Anything else lands in `MouseButton::Other(button as u16)`.
pub fn mouse_button_winit(button: u8) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        3 => MouseButton::Back,
        4 => MouseButton::Forward,
        other => MouseButton::Other(other as u16),
    }
}

/// Map a DOM `MouseEvent.button` index to an `egui::PointerButton`.
///
/// `None` for buttons egui doesn't model. Egui's `Extra1` / `Extra2`
/// cover the browser's back/forward buttons.
pub fn mouse_button_egui(button: u8) -> Option<egui::PointerButton> {
    match button {
        0 => Some(egui::PointerButton::Primary),
        1 => Some(egui::PointerButton::Middle),
        2 => Some(egui::PointerButton::Secondary),
        3 => Some(egui::PointerButton::Extra1),
        4 => Some(egui::PointerButton::Extra2),
        _ => None,
    }
}

/// Map a DOM `KeyboardEvent.code` string to a `winit::keyboard::KeyCode`.
/// Returns `None` for unmapped codes; the caller is expected to drop
/// the event silently. Coverage is the alphabet + digits + function
/// keys + arrows + the common control / modifier keys.
pub fn keycode_winit(code: &str) -> Option<KeyCode> {
    if let Some(k) = letter_winit(code) {
        return Some(k);
    }
    if let Some(k) = digit_winit(code) {
        return Some(k);
    }
    if let Some(k) = function_winit(code) {
        return Some(k);
    }
    match code {
        "Space" => Some(KeyCode::Space),
        "Enter" => Some(KeyCode::Enter),
        "Escape" => Some(KeyCode::Escape),
        "Tab" => Some(KeyCode::Tab),
        "Backspace" => Some(KeyCode::Backspace),
        "Delete" => Some(KeyCode::Delete),
        "Backquote" => Some(KeyCode::Backquote),
        "Minus" => Some(KeyCode::Minus),
        "Equal" => Some(KeyCode::Equal),
        "BracketLeft" => Some(KeyCode::BracketLeft),
        "BracketRight" => Some(KeyCode::BracketRight),
        "Semicolon" => Some(KeyCode::Semicolon),
        "Quote" => Some(KeyCode::Quote),
        "Comma" => Some(KeyCode::Comma),
        "Period" => Some(KeyCode::Period),
        "Slash" => Some(KeyCode::Slash),
        "Backslash" => Some(KeyCode::Backslash),
        "ShiftLeft" => Some(KeyCode::ShiftLeft),
        "ShiftRight" => Some(KeyCode::ShiftRight),
        "ControlLeft" => Some(KeyCode::ControlLeft),
        "ControlRight" => Some(KeyCode::ControlRight),
        "AltLeft" => Some(KeyCode::AltLeft),
        "AltRight" => Some(KeyCode::AltRight),
        "ArrowUp" => Some(KeyCode::ArrowUp),
        "ArrowDown" => Some(KeyCode::ArrowDown),
        "ArrowLeft" => Some(KeyCode::ArrowLeft),
        "ArrowRight" => Some(KeyCode::ArrowRight),
        _ => None,
    }
}

/// Map a DOM `KeyboardEvent.code` to an `egui::Key`. Same coverage
/// principle as [`keycode_winit`].
pub fn keycode_egui(code: &str) -> Option<egui::Key> {
    if let Some(k) = letter_egui(code) {
        return Some(k);
    }
    if let Some(k) = digit_egui(code) {
        return Some(k);
    }
    if let Some(k) = function_egui(code) {
        return Some(k);
    }
    match code {
        "Space" => Some(egui::Key::Space),
        "Enter" => Some(egui::Key::Enter),
        "Escape" => Some(egui::Key::Escape),
        "Tab" => Some(egui::Key::Tab),
        "Backspace" => Some(egui::Key::Backspace),
        "Delete" => Some(egui::Key::Delete),
        "ArrowUp" => Some(egui::Key::ArrowUp),
        "ArrowDown" => Some(egui::Key::ArrowDown),
        "ArrowLeft" => Some(egui::Key::ArrowLeft),
        "ArrowRight" => Some(egui::Key::ArrowRight),
        "Home" => Some(egui::Key::Home),
        "End" => Some(egui::Key::End),
        "PageUp" => Some(egui::Key::PageUp),
        "PageDown" => Some(egui::Key::PageDown),
        "Backquote" => Some(egui::Key::Backtick),
        "Minus" => Some(egui::Key::Minus),
        "Equal" => Some(egui::Key::Equals),
        _ => None,
    }
}

fn letter_winit(code: &str) -> Option<KeyCode> {
    match code {
        "KeyA" => Some(KeyCode::KeyA),
        "KeyB" => Some(KeyCode::KeyB),
        "KeyC" => Some(KeyCode::KeyC),
        "KeyD" => Some(KeyCode::KeyD),
        "KeyE" => Some(KeyCode::KeyE),
        "KeyF" => Some(KeyCode::KeyF),
        "KeyG" => Some(KeyCode::KeyG),
        "KeyH" => Some(KeyCode::KeyH),
        "KeyI" => Some(KeyCode::KeyI),
        "KeyJ" => Some(KeyCode::KeyJ),
        "KeyK" => Some(KeyCode::KeyK),
        "KeyL" => Some(KeyCode::KeyL),
        "KeyM" => Some(KeyCode::KeyM),
        "KeyN" => Some(KeyCode::KeyN),
        "KeyO" => Some(KeyCode::KeyO),
        "KeyP" => Some(KeyCode::KeyP),
        "KeyQ" => Some(KeyCode::KeyQ),
        "KeyR" => Some(KeyCode::KeyR),
        "KeyS" => Some(KeyCode::KeyS),
        "KeyT" => Some(KeyCode::KeyT),
        "KeyU" => Some(KeyCode::KeyU),
        "KeyV" => Some(KeyCode::KeyV),
        "KeyW" => Some(KeyCode::KeyW),
        "KeyX" => Some(KeyCode::KeyX),
        "KeyY" => Some(KeyCode::KeyY),
        "KeyZ" => Some(KeyCode::KeyZ),
        _ => None,
    }
}

fn digit_winit(code: &str) -> Option<KeyCode> {
    match code {
        "Digit0" => Some(KeyCode::Digit0),
        "Digit1" => Some(KeyCode::Digit1),
        "Digit2" => Some(KeyCode::Digit2),
        "Digit3" => Some(KeyCode::Digit3),
        "Digit4" => Some(KeyCode::Digit4),
        "Digit5" => Some(KeyCode::Digit5),
        "Digit6" => Some(KeyCode::Digit6),
        "Digit7" => Some(KeyCode::Digit7),
        "Digit8" => Some(KeyCode::Digit8),
        "Digit9" => Some(KeyCode::Digit9),
        _ => None,
    }
}

fn function_winit(code: &str) -> Option<KeyCode> {
    match code {
        "F1" => Some(KeyCode::F1),
        "F2" => Some(KeyCode::F2),
        "F3" => Some(KeyCode::F3),
        "F4" => Some(KeyCode::F4),
        "F5" => Some(KeyCode::F5),
        "F6" => Some(KeyCode::F6),
        "F7" => Some(KeyCode::F7),
        "F8" => Some(KeyCode::F8),
        "F9" => Some(KeyCode::F9),
        "F10" => Some(KeyCode::F10),
        "F11" => Some(KeyCode::F11),
        "F12" => Some(KeyCode::F12),
        _ => None,
    }
}

fn letter_egui(code: &str) -> Option<egui::Key> {
    match code {
        "KeyA" => Some(egui::Key::A),
        "KeyB" => Some(egui::Key::B),
        "KeyC" => Some(egui::Key::C),
        "KeyD" => Some(egui::Key::D),
        "KeyE" => Some(egui::Key::E),
        "KeyF" => Some(egui::Key::F),
        "KeyG" => Some(egui::Key::G),
        "KeyH" => Some(egui::Key::H),
        "KeyI" => Some(egui::Key::I),
        "KeyJ" => Some(egui::Key::J),
        "KeyK" => Some(egui::Key::K),
        "KeyL" => Some(egui::Key::L),
        "KeyM" => Some(egui::Key::M),
        "KeyN" => Some(egui::Key::N),
        "KeyO" => Some(egui::Key::O),
        "KeyP" => Some(egui::Key::P),
        "KeyQ" => Some(egui::Key::Q),
        "KeyR" => Some(egui::Key::R),
        "KeyS" => Some(egui::Key::S),
        "KeyT" => Some(egui::Key::T),
        "KeyU" => Some(egui::Key::U),
        "KeyV" => Some(egui::Key::V),
        "KeyW" => Some(egui::Key::W),
        "KeyX" => Some(egui::Key::X),
        "KeyY" => Some(egui::Key::Y),
        "KeyZ" => Some(egui::Key::Z),
        _ => None,
    }
}

fn digit_egui(code: &str) -> Option<egui::Key> {
    match code {
        "Digit0" => Some(egui::Key::Num0),
        "Digit1" => Some(egui::Key::Num1),
        "Digit2" => Some(egui::Key::Num2),
        "Digit3" => Some(egui::Key::Num3),
        "Digit4" => Some(egui::Key::Num4),
        "Digit5" => Some(egui::Key::Num5),
        "Digit6" => Some(egui::Key::Num6),
        "Digit7" => Some(egui::Key::Num7),
        "Digit8" => Some(egui::Key::Num8),
        "Digit9" => Some(egui::Key::Num9),
        _ => None,
    }
}

fn function_egui(code: &str) -> Option<egui::Key> {
    match code {
        "F1" => Some(egui::Key::F1),
        "F2" => Some(egui::Key::F2),
        "F3" => Some(egui::Key::F3),
        "F4" => Some(egui::Key::F4),
        "F5" => Some(egui::Key::F5),
        "F6" => Some(egui::Key::F6),
        "F7" => Some(egui::Key::F7),
        "F8" => Some(egui::Key::F8),
        "F9" => Some(egui::Key::F9),
        "F10" => Some(egui::Key::F10),
        "F11" => Some(egui::Key::F11),
        "F12" => Some(egui::Key::F12),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mouse_button_winit_known_dom_indices() {
        assert!(matches!(mouse_button_winit(0), MouseButton::Left));
        assert!(matches!(mouse_button_winit(1), MouseButton::Middle));
        assert!(matches!(mouse_button_winit(2), MouseButton::Right));
        // Unknown index lands in Other.
        match mouse_button_winit(99) {
            MouseButton::Other(n) => assert_eq!(n, 99),
            _ => panic!("expected MouseButton::Other(99)"),
        }
    }

    #[test]
    fn mouse_button_egui_known_dom_indices() {
        assert_eq!(mouse_button_egui(0), Some(egui::PointerButton::Primary));
        assert_eq!(mouse_button_egui(2), Some(egui::PointerButton::Secondary));
        assert_eq!(mouse_button_egui(99), None);
    }

    #[test]
    fn keycode_winit_spot_check_letter_digit_function_control() {
        // The codepaths we lean on: alpha, digit, function, control.
        assert_eq!(keycode_winit("KeyA"), Some(KeyCode::KeyA));
        assert_eq!(keycode_winit("KeyZ"), Some(KeyCode::KeyZ));
        assert_eq!(keycode_winit("Digit5"), Some(KeyCode::Digit5));
        assert_eq!(keycode_winit("F3"), Some(KeyCode::F3));
        assert_eq!(keycode_winit("Space"), Some(KeyCode::Space));
        assert_eq!(keycode_winit("ArrowUp"), Some(KeyCode::ArrowUp));
        assert_eq!(keycode_winit("Backquote"), Some(KeyCode::Backquote));
        // Unknown -> None.
        assert_eq!(keycode_winit("NotARealKey"), None);
        assert_eq!(keycode_winit(""), None);
    }

    #[test]
    fn keycode_egui_spot_check() {
        assert_eq!(keycode_egui("KeyA"), Some(egui::Key::A));
        assert_eq!(keycode_egui("Digit0"), Some(egui::Key::Num0));
        assert_eq!(keycode_egui("F3"), Some(egui::Key::F3));
        assert_eq!(keycode_egui("Backquote"), Some(egui::Key::Backtick));
        assert_eq!(keycode_egui("Equal"), Some(egui::Key::Equals));
        assert_eq!(keycode_egui("NotARealKey"), None);
    }

    /// Tesseract_demo's known hotkeys + axes must all resolve. Catches
    /// regressions if someone tightens the mapping by accident.
    #[test]
    fn tesseract_demo_keys_all_map_winit() {
        for code in [
            "KeyW",
            "KeyA",
            "KeyS",
            "KeyD", // WASD
            "KeyF", // toggle camera mode
            "KeyT",
            "Space",     // pause
            "KeyR",      // reset
            "F3",        // perf overlay
            "Backquote", // console
            "ShiftLeft",
            "ShiftRight", // free-roam descend
        ] {
            assert!(
                keycode_winit(code).is_some(),
                "tesseract_demo key '{code}' must map to a KeyCode"
            );
        }
    }
}
