//! Freecam preset for FPS-style mouse-look + WASD translation.
//!
//! Composes [`rye_camera::FirstPersonController`] + a world-space position +
//! [`crate::cursor`] grab requests into a single struct. The pattern was
//! growing in two demos (tesseract_demo + polytope_playground) and shared
//! enough to lift here per the engine-first guideline.
//!
//! ## What a Freecam does each frame
//!
//! When [`Freecam::active`] is `true` AND [`Freecam::cursor_grabbed`] is
//! `true`:
//!
//! 1. The internal [`FirstPersonController`] consumes
//!    `FrameInput::mouse_raw_delta` to update yaw + pitch.
//! 2. WASD + Space/Shift on `FrameInput` integrate the world-space
//!    `position` field at `speed` units/sec.
//! 3. The owning `Camera<EuclideanR3>` has its `position` + basis updated.
//!
//! When `active = false` OR `cursor_grabbed = false`: [`Freecam::advance`]
//! is a no-op. The cursor-grab toggle lets the user release the mouse for
//! UI access without leaving freecam mode entirely (the position + look
//! direction freeze in place; resuming the grab continues from there).
//!
//! ## Activation contract
//!
//! [`Freecam::set_active`] seeds the freecam's `position` from the
//! current camera location AND requests a cursor grab via
//! `rye_app::cursor`. Deactivating requests a cursor release. The cursor
//! grab is paired with hidden visibility for FPS mouse-look; demos that
//! want a different cursor regime override via `rye_app::cursor` directly
//! after `set_active`.
//!
//! [`Freecam::toggle_cursor_grab`] flips JUST the grab (and the controller's
//! `use_raw_delta` flag) without changing `active`. Bind this to a key
//! (Alt is the convention) for in-freecam UI interaction.
//!
//! ## Wasm caveat
//!
//! `rye_app::cursor::request_grab` is a documented no-op on wasm
//! (browser Pointer Lock requires a user gesture). Freecam on wasm will
//! work but the mouse will escape the screen; demos shipping freecam to
//! browsers need a click-to-engage layer on the main-thread side. See the
//! cursor module doc.

use glam::Vec3;
use rye_camera::{CameraController, FirstPersonController};
use rye_input::FrameInput;
use rye_math::EuclideanR3;

use crate::Camera;

/// Default WASD translation speed for new Freecam instances.
const DEFAULT_SPEED: f32 = 4.5;

/// FPS-style camera preset: mouse-look + WASD/Space/Shift translation +
/// cursor grab management.
///
/// Construct with [`Freecam::new`], optionally tune with [`Freecam::with_speed`],
/// then call [`Freecam::set_active`] to enter freecam mode and
/// [`Freecam::advance`] each frame.
#[derive(Clone, Copy, Debug)]
pub struct Freecam {
    /// Underlying look-direction controller. Public so demos can read
    /// yaw/pitch for HUD display or tweak `use_raw_delta` directly.
    /// `Freecam::advance` overwrites `use_raw_delta` based on the grab
    /// state each frame, so manual changes don't persist past the next
    /// advance call.
    pub controller: FirstPersonController<EuclideanR3>,
    /// World-space position. Updated each `advance` while active +
    /// grabbed. Public so demos can teleport or restore from save.
    pub position: Vec3,
    /// Translation speed in units per second. Affects WASD + Space/Shift
    /// only; mouse-look sensitivity is set on `controller` (see
    /// `rye_camera::FIRST_PERSON_MOUSE_SENSITIVITY`).
    pub speed: f32,
    active: bool,
    cursor_grabbed: bool,
}

impl Default for Freecam {
    fn default() -> Self {
        Self::new()
    }
}

impl Freecam {
    /// Construct an inactive freecam at the origin with default speed.
    /// Call [`set_active(true, current_pos)`](Self::set_active) when the
    /// demo enters freecam mode.
    pub fn new() -> Self {
        Self {
            controller: FirstPersonController::<EuclideanR3>::new(0.0, 0.0),
            position: Vec3::ZERO,
            speed: DEFAULT_SPEED,
            active: false,
            cursor_grabbed: false,
        }
    }

    /// Builder: set translation speed at construction. Default 4.5 u/sec.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Is freecam currently the active camera mode for the demo?
    pub fn active(&self) -> bool {
        self.active
    }

    /// Is the cursor currently grabbed (mouse-look engaged)? Always false
    /// when inactive; toggleable via [`toggle_cursor_grab`](Self::toggle_cursor_grab)
    /// while active.
    pub fn cursor_grabbed(&self) -> bool {
        self.cursor_grabbed
    }

    /// Enter or leave freecam mode.
    ///
    /// On entry (`active = true`): seeds the internal position from
    /// `current_camera_pos` (so the toggle feels continuous rather than
    /// teleporting), grabs the cursor + hides it, and primes the
    /// controller to read raw mouse deltas. The caller should switch
    /// their camera-mode state machine to "freecam" and stop running the
    /// orbit controller's advance.
    ///
    /// On exit (`active = false`): releases the cursor + shows it,
    /// resets the controller's raw-delta flag, leaves position + yaw +
    /// pitch where they were so a re-entry returns to the same pose.
    pub fn set_active(&mut self, active: bool, current_camera_pos: Vec3) {
        if active == self.active {
            return;
        }
        self.active = active;
        if active {
            self.position = current_camera_pos;
            self.cursor_grabbed = true;
            self.controller.use_raw_delta = true;
            crate::cursor::request_grab();
        } else {
            self.cursor_grabbed = false;
            self.controller.use_raw_delta = false;
            crate::cursor::request_release();
        }
    }

    /// Flip the cursor grab without changing the active state. Bind to
    /// Alt (or similar) in the demo for "release the cursor so I can
    /// click a UI widget, then re-grab to resume mouse-look." No-op if
    /// the freecam isn't active.
    pub fn toggle_cursor_grab(&mut self) {
        if !self.active {
            return;
        }
        self.cursor_grabbed = !self.cursor_grabbed;
        self.controller.use_raw_delta = self.cursor_grabbed;
        if self.cursor_grabbed {
            crate::cursor::request_grab();
        } else {
            crate::cursor::request_release();
        }
    }

    /// Per-frame update. Drives the camera's basis from yaw/pitch and
    /// the camera's position from `position` (which itself is integrated
    /// from WASD + Space/Shift). No-op when inactive or when the cursor
    /// is released (Alt-toggled in-freecam UI-access mode).
    ///
    /// Caller still owns the higher-level mode switch (orbit vs freecam);
    /// this method assumes `Freecam` IS the active controller this frame.
    pub fn advance(
        &mut self,
        input: FrameInput,
        camera: &mut Camera<EuclideanR3>,
        dt: f32,
    ) {
        if !self.active || !self.cursor_grabbed {
            return;
        }
        // Look direction: controller reads raw delta, updates yaw + pitch,
        // writes the camera's right/up/forward.
        self.controller
            .advance(input, camera, &EuclideanR3, dt);
        // Position: integrate the WASD + Space/Shift axes in the camera's
        // local frame.
        let mut delta = camera.forward * input.move_forward
            + camera.right * input.move_right
            + Vec3::Y * input.move_up;
        if delta.length_squared() > 1e-6 {
            delta = delta.normalize();
            self.position += delta * self.speed * dt;
            camera.position = self.position;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor;

    fn make_camera() -> Camera<EuclideanR3> {
        let mut c = Camera::<EuclideanR3>::at_origin();
        c.position = Vec3::new(0.0, 1.0, 5.0);
        c.forward = -Vec3::Z;
        c.right = Vec3::X;
        c.up = Vec3::Y;
        c
    }

    #[test]
    fn new_is_inactive() {
        let f = Freecam::new();
        assert!(!f.active());
        assert!(!f.cursor_grabbed());
        assert_eq!(f.speed, DEFAULT_SPEED);
    }

    #[test]
    fn with_speed_overrides() {
        let f = Freecam::new().with_speed(12.0);
        assert!((f.speed - 12.0).abs() < 1e-6);
    }

    #[test]
    fn set_active_true_seeds_position_and_grabs() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        let cam_pos = Vec3::new(3.0, 2.0, 7.0);
        f.set_active(true, cam_pos);
        assert!(f.active());
        assert!(f.cursor_grabbed());
        assert!(f.controller.use_raw_delta);
        assert_eq!(f.position, cam_pos);
        // Verify a grab request was queued.
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::Locked));
        assert_eq!(vis, Some(false));
    }

    #[test]
    fn set_active_false_releases() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();
        f.set_active(false, Vec3::ZERO);
        assert!(!f.active());
        assert!(!f.cursor_grabbed());
        assert!(!f.controller.use_raw_delta);
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::None));
        assert_eq!(vis, Some(true));
    }

    #[test]
    fn set_active_idempotent() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();
        // Second call with same value: no new request queued.
        f.set_active(true, Vec3::new(99.0, 99.0, 99.0));
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None, "idempotent set_active shouldn't re-queue");
        assert_eq!(vis, None);
        // Position should NOT be overwritten by an idempotent call.
        assert_eq!(f.position, Vec3::ZERO);
    }

    #[test]
    fn toggle_cursor_grab_flips_within_active() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();

        f.toggle_cursor_grab();
        assert!(f.active(), "toggle doesn't change active");
        assert!(!f.cursor_grabbed());
        assert!(!f.controller.use_raw_delta);
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::None));
        assert_eq!(vis, Some(true));

        f.toggle_cursor_grab();
        assert!(f.cursor_grabbed());
        assert!(f.controller.use_raw_delta);
    }

    #[test]
    fn toggle_cursor_grab_noop_when_inactive() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.toggle_cursor_grab();
        assert!(!f.active());
        assert!(!f.cursor_grabbed());
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);
    }

    #[test]
    fn advance_noop_when_inactive() {
        let mut f = Freecam::new();
        let mut cam = make_camera();
        let cam_before = cam;
        let pos_before = f.position;
        let input = FrameInput {
            mouse_raw_delta: glam::Vec2::new(100.0, 50.0),
            move_forward: 1.0,
            ..Default::default()
        };
        f.advance(input, &mut cam, 0.016);
        assert_eq!(cam.position, cam_before.position);
        assert_eq!(cam.forward, cam_before.forward);
        assert_eq!(f.position, pos_before);
    }

    #[test]
    fn advance_noop_when_cursor_released() {
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        f.toggle_cursor_grab(); // release
        let mut cam = make_camera();
        let cam_before = cam;
        let input = FrameInput {
            mouse_raw_delta: glam::Vec2::new(100.0, 50.0),
            move_forward: 1.0,
            ..Default::default()
        };
        f.advance(input, &mut cam, 0.016);
        assert_eq!(cam.forward, cam_before.forward, "look frozen when released");
    }

    #[test]
    fn advance_integrates_forward_motion() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new().with_speed(2.0);
        f.set_active(true, Vec3::ZERO);
        let mut cam = make_camera();
        cam.position = Vec3::ZERO;
        cam.forward = -Vec3::Z; // forward = -Z
        let input = FrameInput {
            move_forward: 1.0,
            ..Default::default()
        };
        f.advance(input, &mut cam, 0.5);
        // speed * dt = 2.0 * 0.5 = 1.0 unit, in -Z direction.
        let expected = Vec3::new(0.0, 0.0, -1.0);
        assert!(
            (f.position - expected).length() < 1e-5,
            "expected {expected:?}, got {:?}",
            f.position,
        );
        assert!((cam.position - expected).length() < 1e-5);
    }
}
