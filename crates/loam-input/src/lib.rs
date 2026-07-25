//! Per-frame input accumulator. Routes raw winit events into a
//! [`FrameInput`] snapshot the rest of the engine consumes.
//!
//! [`InputState::take_frame`] resets per-tick deltas (mouse delta, scroll)
//! while preserving held state (buttons and their press anchors, cursor
//! position, WASD axes, modifiers) across drains.
//! [`InputState::cursor_invalidated`] / [`InputState::release_buttons`]
//! are called on focus loss / cursor exit to drop cursor-relative and held
//! state cleanly.

use glam::{Vec2, Vec3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Approximate pixels-per-notch used to normalize trackpad `PixelDelta`
/// scroll events into the same `scroll_lines` units as `LineDelta`.
pub const SCROLL_PIXELS_PER_LINE: f32 = 50.0;

/// One mouse button: held state plus the cursor position at its press
/// edge. Drag chords need that anchor to measure a later cursor position
/// against; a delta sum cannot recover it once the button has been held
/// across several drains.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ButtonState {
    pub down: bool,
    /// `None` while the button is up, and also for a press that arrived
    /// while the cursor position was unknown (before the first
    /// `CursorMoved`, or after [`InputState::cursor_invalidated`]).
    pub press_pos: Option<Vec2>,
}

/// The three positional mouse buttons. `Back` / `Forward` / `Other` are
/// dropped: no input path in the engine binds them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MouseButtons {
    pub left: ButtonState,
    pub right: ButtonState,
    pub middle: ButtonState,
}

impl MouseButtons {
    fn slot(&mut self, button: MouseButton) -> Option<&mut ButtonState> {
        match button {
            MouseButton::Left => Some(&mut self.left),
            MouseButton::Right => Some(&mut self.right),
            MouseButton::Middle => Some(&mut self.middle),
            _ => None,
        }
    }
}

/// Chord modifiers. Derived from the held physical keys rather than
/// winit's `ModifiersChanged` so one key path feeds both the move axes
/// and the modifier set, and left/right variants cannot disagree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    /// Windows / Command key (winit `SuperLeft` / `SuperRight`).
    pub super_key: bool,
}

/// Accumulated input for one simulation tick, consumed by
/// [`InputState::take_frame`].
///
/// `mouse_delta` is the OS-clamped cursor delta (`CursorMoved`); stops at the
/// screen edge. `mouse_raw_delta` is the raw device delta (`MouseMotion`);
/// accumulates past the edge for FPS-style infinite-yaw mouse-look.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInput {
    pub mouse_delta: Vec2,
    pub mouse_raw_delta: Vec2,
    pub scroll_lines: f32,
    /// Shorthand for `buttons.left.down`; [`InputState`] keeps the two in
    /// step.
    pub left_mouse_down: bool,
    pub buttons: MouseButtons,
    /// Window-relative cursor position in physical pixels. `None` before
    /// the first `CursorMoved` and after the cursor leaves the window, so
    /// picking cannot silently reuse a stale position.
    pub cursor_pos: Option<Vec2>,
    pub modifiers: Modifiers,
    /// WASD forward/back: W = +1, S = −1.
    pub move_forward: f32,
    /// WASD strafe: D = +1, A = −1.
    pub move_right: f32,
    /// Vertical: Space = +1, Left/Right Shift = −1.
    pub move_up: f32,
}

impl FrameInput {
    /// WASD axes as a (possibly zero) direction vector; callers scale and
    /// apply to their coordinate system.
    pub fn move_dir(&self) -> Vec3 {
        Vec3::new(self.move_right, self.move_up, -self.move_forward)
    }
}

/// Per-window input accumulator. Feed winit events in; call
/// [`InputState::take_frame`] once per tick to drain them.
#[derive(Debug, Default)]
pub struct InputState {
    frame: FrameInput,
    held_keys: std::collections::HashSet<KeyCode>,
}

impl InputState {
    pub fn cursor_moved(&mut self, x: f64, y: f64) {
        let pos = Vec2::new(x as f32, y as f32);
        if let Some(last) = self.frame.cursor_pos {
            self.frame.mouse_delta += pos - last;
        }
        self.frame.cursor_pos = Some(pos);
    }

    /// Re-anchor delta accumulation and mark the cursor position unknown.
    /// Call on `CursorLeft` and focus loss.
    pub fn cursor_invalidated(&mut self) {
        self.frame.cursor_pos = None;
    }

    /// Release held mouse buttons and their press anchors. Call on focus
    /// loss.
    pub fn release_buttons(&mut self) {
        self.frame.buttons = MouseButtons::default();
        self.frame.left_mouse_down = false;
    }

    pub fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        let cursor = self.frame.cursor_pos;
        let pressed = state == ElementState::Pressed;
        if let Some(slot) = self.frame.buttons.slot(button) {
            slot.down = pressed;
            // Anchor on the press edge and drop it on release so a stale
            // anchor cannot outlive the drag that set it.
            slot.press_pos = if pressed { cursor } else { None };
        }
        self.frame.left_mouse_down = self.frame.buttons.left.down;
    }

    pub fn mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.frame.scroll_lines += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / SCROLL_PIXELS_PER_LINE,
        };
    }

    /// Update held-key set from a `WindowEvent::KeyboardInput` physical key.
    pub fn key_input(&mut self, physical_key: PhysicalKey, state: ElementState) {
        if let PhysicalKey::Code(code) = physical_key {
            match state {
                ElementState::Pressed => {
                    self.held_keys.insert(code);
                }
                ElementState::Released => {
                    self.held_keys.remove(&code);
                }
            }
        }
    }

    /// Drain accumulated input for one tick: resets deltas, persists held
    /// state (buttons, press anchors, cursor position), recomputes move
    /// axes and modifiers from held keys.
    pub fn take_frame(&mut self) -> FrameInput {
        let held = &self.held_keys;
        self.frame.move_forward = axis(held, KeyCode::KeyW, KeyCode::KeyS);
        self.frame.move_right = axis(held, KeyCode::KeyD, KeyCode::KeyA);
        self.frame.move_up = axis(
            held,
            KeyCode::Space,
            // treat either shift as down
            if held.contains(&KeyCode::ShiftRight) {
                KeyCode::ShiftRight
            } else {
                KeyCode::ShiftLeft
            },
        );
        self.frame.modifiers = Modifiers {
            shift: either(held, KeyCode::ShiftLeft, KeyCode::ShiftRight),
            control: either(held, KeyCode::ControlLeft, KeyCode::ControlRight),
            alt: either(held, KeyCode::AltLeft, KeyCode::AltRight),
            super_key: either(held, KeyCode::SuperLeft, KeyCode::SuperRight),
        };

        let frame = self.frame;
        self.frame.mouse_delta = Vec2::ZERO;
        self.frame.mouse_raw_delta = Vec2::ZERO;
        self.frame.scroll_lines = 0.0;
        frame
    }

    /// Accumulate raw device motion (`DeviceEvent::MouseMotion`) into
    /// `FrameInput::mouse_raw_delta`. `pub` only because the runner lives in
    /// a separate crate; `#[doc(hidden)]` keeps it out of demo-facing docs.
    #[doc(hidden)]
    pub fn accumulate_raw_motion(&mut self, dx: f64, dy: f64) {
        self.frame.mouse_raw_delta += Vec2::new(dx as f32, dy as f32);
    }
}

fn axis(held: &std::collections::HashSet<KeyCode>, pos: KeyCode, neg: KeyCode) -> f32 {
    let p = held.contains(&pos) as u8 as f32;
    let n = held.contains(&neg) as u8 as f32;
    p - n
}

fn either(held: &std::collections::HashSet<KeyCode>, left: KeyCode, right: KeyCode) -> bool {
    held.contains(&left) || held.contains(&right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} to be within 1e-5 of {b}"
        );
    }

    #[test]
    fn cursor_invalidated_prevents_jump_delta() {
        let mut input = InputState::default();
        input.cursor_moved(100.0, 100.0);
        input.cursor_invalidated();
        input.cursor_moved(500.0, 500.0);
        let frame = input.take_frame();
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
    }

    #[test]
    fn release_buttons_clears_left_drag() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.release_buttons();
        let frame = input.take_frame();
        assert!(!frame.left_mouse_down);
    }

    #[test]
    fn pixel_delta_scroll_uses_pixels_per_line() {
        let mut input = InputState::default();
        input.mouse_wheel(MouseScrollDelta::PixelDelta(
            winit::dpi::PhysicalPosition::new(0.0, SCROLL_PIXELS_PER_LINE as f64),
        ));
        let frame = input.take_frame();
        assert_close(frame.scroll_lines, 1.0);
    }

    #[test]
    fn take_frame_keeps_button_state_and_clears_deltas() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.cursor_moved(10.0, 20.0);
        input.cursor_moved(15.0, 18.0);
        input.mouse_wheel(MouseScrollDelta::LineDelta(0.0, 2.0));

        let frame = input.take_frame();
        assert!(frame.left_mouse_down);
        assert_eq!(frame.mouse_delta, Vec2::new(5.0, -2.0));
        assert_eq!(frame.scroll_lines, 2.0);

        let frame = input.take_frame();
        assert!(frame.left_mouse_down);
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
        assert_eq!(frame.scroll_lines, 0.0);
    }

    #[test]
    fn wasd_keys_produce_move_axes() {
        let mut input = InputState::default();
        input.key_input(PhysicalKey::Code(KeyCode::KeyW), ElementState::Pressed);
        input.key_input(PhysicalKey::Code(KeyCode::KeyD), ElementState::Pressed);
        let frame = input.take_frame();
        assert_close(frame.move_forward, 1.0);
        assert_close(frame.move_right, 1.0);
        assert_close(frame.move_up, 0.0);
    }

    #[test]
    fn key_release_clears_axis() {
        let mut input = InputState::default();
        input.key_input(PhysicalKey::Code(KeyCode::KeyW), ElementState::Pressed);
        input.key_input(PhysicalKey::Code(KeyCode::KeyW), ElementState::Released);
        let frame = input.take_frame();
        assert_close(frame.move_forward, 0.0);
    }

    #[test]
    fn cursor_moved_reports_absolute_position_and_invalidation_clears_it() {
        let mut input = InputState::default();
        assert_eq!(input.take_frame().cursor_pos, None);
        input.cursor_moved(10.0, 20.0);
        assert_eq!(input.take_frame().cursor_pos, Some(Vec2::new(10.0, 20.0)));
        input.cursor_invalidated();
        assert_eq!(input.take_frame().cursor_pos, None);
    }

    #[test]
    fn button_press_anchors_at_cursor_and_release_drops_the_anchor() {
        let mut input = InputState::default();
        input.cursor_moved(30.0, 40.0);
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        // Anchor must survive later motion: it records where the drag
        // began, not where the cursor is now.
        input.cursor_moved(90.0, 40.0);

        let frame = input.take_frame();
        assert!(frame.buttons.left.down);
        assert_eq!(frame.buttons.left.press_pos, Some(Vec2::new(30.0, 40.0)));
        assert_eq!(frame.cursor_pos, Some(Vec2::new(90.0, 40.0)));

        input.mouse_input(MouseButton::Left, ElementState::Released);
        let frame = input.take_frame();
        assert!(!frame.buttons.left.down);
        assert_eq!(frame.buttons.left.press_pos, None);
    }

    #[test]
    fn press_with_unknown_cursor_has_no_anchor() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        assert_eq!(input.take_frame().buttons.right.press_pos, None);

        input.mouse_input(MouseButton::Right, ElementState::Released);
        input.cursor_moved(5.0, 5.0);
        input.cursor_invalidated();
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        let frame = input.take_frame();
        assert!(frame.buttons.right.down);
        assert_eq!(frame.buttons.right.press_pos, None);
    }

    #[test]
    fn each_button_carries_its_own_state_and_anchor() {
        let mut input = InputState::default();
        input.cursor_moved(1.0, 2.0);
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        input.cursor_moved(3.0, 4.0);
        input.mouse_input(MouseButton::Middle, ElementState::Pressed);

        let frame = input.take_frame();
        assert!(!frame.buttons.left.down);
        assert_eq!(frame.buttons.right.press_pos, Some(Vec2::new(1.0, 2.0)));
        assert_eq!(frame.buttons.middle.press_pos, Some(Vec2::new(3.0, 4.0)));

        input.mouse_input(MouseButton::Right, ElementState::Released);
        let frame = input.take_frame();
        assert!(!frame.buttons.right.down);
        assert!(frame.buttons.middle.down);
    }

    #[test]
    fn unbound_buttons_leave_tracked_state_untouched() {
        let mut input = InputState::default();
        input.cursor_moved(7.0, 8.0);
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.mouse_input(MouseButton::Back, ElementState::Pressed);
        input.mouse_input(MouseButton::Other(9), ElementState::Pressed);

        let frame = input.take_frame();
        assert_eq!(frame.buttons.left.press_pos, Some(Vec2::new(7.0, 8.0)));
        assert!(!frame.buttons.right.down);
        assert!(!frame.buttons.middle.down);
    }

    #[test]
    fn left_mouse_down_mirrors_left_button_state() {
        let mut input = InputState::default();
        for (button, state) in [
            (MouseButton::Right, ElementState::Pressed),
            (MouseButton::Left, ElementState::Pressed),
            (MouseButton::Middle, ElementState::Released),
            (MouseButton::Left, ElementState::Released),
        ] {
            input.mouse_input(button, state);
            let frame = input.take_frame();
            assert_eq!(frame.left_mouse_down, frame.buttons.left.down);
        }
    }

    #[test]
    fn release_buttons_clears_every_button_and_anchor() {
        let mut input = InputState::default();
        input.cursor_moved(11.0, 12.0);
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        input.mouse_input(MouseButton::Middle, ElementState::Pressed);
        input.release_buttons();

        let frame = input.take_frame();
        assert_eq!(frame.buttons, MouseButtons::default());
        assert!(!frame.left_mouse_down);
    }

    #[test]
    fn held_buttons_and_cursor_persist_across_drains() {
        let mut input = InputState::default();
        input.cursor_moved(50.0, 60.0);
        input.mouse_input(MouseButton::Right, ElementState::Pressed);

        let first = input.take_frame();
        let second = input.take_frame();
        assert_eq!(second.buttons, first.buttons);
        assert_eq!(second.cursor_pos, first.cursor_pos);
        assert_eq!(second.mouse_delta, Vec2::ZERO);
    }

    #[test]
    fn modifiers_persist_across_drains_until_the_key_releases() {
        let mut input = InputState::default();
        input.key_input(
            PhysicalKey::Code(KeyCode::ControlLeft),
            ElementState::Pressed,
        );
        input.key_input(PhysicalKey::Code(KeyCode::AltLeft), ElementState::Pressed);

        let first = input.take_frame();
        assert!(first.modifiers.control);
        assert!(first.modifiers.alt);
        assert!(!first.modifiers.shift);
        assert_eq!(input.take_frame().modifiers, first.modifiers);

        input.key_input(PhysicalKey::Code(KeyCode::AltLeft), ElementState::Released);
        let frame = input.take_frame();
        assert!(frame.modifiers.control);
        assert!(!frame.modifiers.alt);
    }

    #[test]
    fn either_side_of_a_modifier_pair_sets_exactly_its_own_flag() {
        let shift = Modifiers {
            shift: true,
            ..Modifiers::default()
        };
        let control = Modifiers {
            control: true,
            ..Modifiers::default()
        };
        let alt = Modifiers {
            alt: true,
            ..Modifiers::default()
        };
        let super_key = Modifiers {
            super_key: true,
            ..Modifiers::default()
        };
        let pairs = [
            (KeyCode::ShiftLeft, KeyCode::ShiftRight, shift),
            (KeyCode::ControlLeft, KeyCode::ControlRight, control),
            (KeyCode::AltLeft, KeyCode::AltRight, alt),
            (KeyCode::SuperLeft, KeyCode::SuperRight, super_key),
        ];
        for (left, right, expected) in pairs {
            for key in [left, right] {
                let mut input = InputState::default();
                input.key_input(PhysicalKey::Code(key), ElementState::Pressed);
                assert_eq!(
                    input.take_frame().modifiers,
                    expected,
                    "{key:?} must set exactly {expected:?}"
                );
            }
        }
    }
}
