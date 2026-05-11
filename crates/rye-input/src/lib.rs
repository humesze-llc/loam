//! Per-frame input accumulator. Routes raw winit events into a [`FrameInput`] snapshot
//! the rest of the engine consumes.
//!
//! ## Drain semantic
//!
//! [`InputState`] accumulates mouse motion / scroll / key state as winit events arrive.
//! [`InputState::take_frame`] returns the current snapshot **and resets the per-tick deltas**
//! (mouse delta, scroll lines): callers see exactly the motion that happened since the last
//! drain. Held state (button-down, WASD axes) is preserved across drains so a user holding
//! a key continues to produce non-zero `move_*` until they release.
//!
//! ## Focus / cursor-loss contract
//!
//! Window focus loss and cursor exit are handled by [`InputState::cursor_invalidated`]
//! (called from the framework's `WindowEvent::CursorLeft` / `Focused(false)` branches). It
//! zeroes the cursor-relative state to avoid a snap when the cursor re-enters at a different
//! position. [`InputState::release_buttons`] is similarly called on focus loss to drop held
//! buttons cleanly. Apps that bind their own keyboard handling should call
//! [`InputState::release_buttons`] when their gameplay needs the same clean-on-defocus
//! semantic.

use glam::{Vec2, Vec3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Approximate pixels-per-notch used to normalize trackpad `PixelDelta`
/// scroll events into the same `scroll_lines` units as `LineDelta`.
pub const SCROLL_PIXELS_PER_LINE: f32 = 50.0;

/// Accumulated input for one simulation tick, consumed by [`InputState::take_frame`].
///
/// `mouse_delta` and `scroll_lines` reset to zero each frame. `left_mouse_down` persists
/// until the button is released. `move_*` axes are recomputed from held keys each frame:
/// +1 / 0 / -1, not accumulated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInput {
    pub mouse_delta: Vec2,
    pub scroll_lines: f32,
    pub left_mouse_down: bool,
    /// WASD forward/back: W = +1, S = −1.
    pub move_forward: f32,
    /// WASD strafe: D = +1, A = −1.
    pub move_right: f32,
    /// Vertical: Space = +1, Left/Right Shift = −1.
    pub move_up: f32,
}

impl FrameInput {
    /// Returns the WASD axes as a (possibly zero) direction vector.
    /// Callers must scale and apply to their coordinate system.
    pub fn move_dir(&self) -> Vec3 {
        Vec3::new(self.move_right, self.move_up, -self.move_forward)
    }
}

/// Per-window input accumulator. Feed winit events in; call
/// [`InputState::take_frame`] once per tick to drain them.
#[derive(Debug, Default)]
pub struct InputState {
    frame: FrameInput,
    last_cursor: Option<Vec2>,
    held_keys: std::collections::HashSet<KeyCode>,
}

impl InputState {
    pub fn cursor_moved(&mut self, x: f64, y: f64) {
        let pos = Vec2::new(x as f32, y as f32);
        if let Some(last) = self.last_cursor {
            self.frame.mouse_delta += pos - last;
        }
        self.last_cursor = Some(pos);
    }

    /// Re-anchor delta accumulation. Call on `CursorLeft` and focus loss.
    pub fn cursor_invalidated(&mut self) {
        self.last_cursor = None;
    }

    /// Release held mouse buttons. Call on focus loss.
    pub fn release_buttons(&mut self) {
        self.frame.left_mouse_down = false;
    }

    pub fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.frame.left_mouse_down = state == ElementState::Pressed;
        }
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

    /// Drain accumulated input for one tick.
    ///
    /// Resets `mouse_delta` and `scroll_lines` to zero. `left_mouse_down`
    /// persists. Move axes are recomputed from held keys.
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

        let frame = self.frame;
        self.frame.mouse_delta = Vec2::ZERO;
        self.frame.scroll_lines = 0.0;
        frame
    }
}

fn axis(held: &std::collections::HashSet<KeyCode>, pos: KeyCode, neg: KeyCode) -> f32 {
    let p = held.contains(&pos) as u8 as f32;
    let n = held.contains(&neg) as u8 as f32;
    p - n
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
}
