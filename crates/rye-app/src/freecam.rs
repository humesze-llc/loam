//! Freecam preset for FPS-style mouse-look + WASD translation.
//!
//! Composes [`rye_camera::FirstPersonController`] + a world-space position +
//! [`crate::cursor`] grab requests into a single struct. The pattern was
//! growing in two demos (tesseract_demo + polytope_playground) and shared
//! enough to lift here per the engine-first guideline.
//!
//! ## What a Freecam does each frame
//!
//! When [`Freecam::active`] is `true`:
//!
//! 1. WASD + Space/Shift on `FrameInput` integrate the world-space
//!    `position` field at `speed` units/sec. Runs whenever active,
//!    independent of cursor grab.
//! 2. **When [`Freecam::cursor_grabbed`] is `true` as well**, the
//!    internal [`FirstPersonController`] consumes
//!    `FrameInput::mouse_raw_delta` to update yaw + pitch.
//! 3. The owning `Camera<EuclideanR3>` has its `position` + basis updated.
//!
//! When `active = false`: [`Freecam::advance`] is a no-op. When
//! `cursor_grabbed = false`: look-direction freezes (so releasing the
//! cursor for UI access doesn't drag the camera), but WASD translation
//! continues. This split is what lets the browser flavor work at all --
//! on wasm, Pointer Lock requires a user gesture the engine doesn't
//! deliver, so `cursor_grabbed` is permanently false; gating WASD on it
//! would freeze freecam translation entirely there.
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
//! Prefer [`Freecam::on_alt`] for the standard FPS/MMO modifier-key contract:
//! demos forward Alt press AND release into `on_alt(pressed)` and the
//! cursor-mode field decides whether to toggle (default, FPS sticky) or
//! hold-to-release (MMO). The mode is runtime-configurable via
//! [`Freecam::set_cursor_mode`].
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

/// How the Alt modifier influences the cursor grab in [`Freecam`].
///
/// Both modes still flip the controller's `use_raw_delta` flag so a
/// released cursor never accumulates yaw/pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CursorMode {
    /// MMO-style: cursor released while Alt is held, re-grabbed when Alt
    /// is released. Best when the cursor-release windows are short
    /// (click a button, then resume).
    Hold,
    /// FPS sticky-modifier: press Alt to flip the grab, press again to
    /// flip back. Release does nothing. Default. Best when the
    /// cursor-release windows are long (read a panel, type into a text
    /// field).
    #[default]
    Toggle,
}

impl CursorMode {
    /// Parse a console token (e.g. from `camera freecam cursor_mode hold`).
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "hold" => Some(Self::Hold),
            "toggle" => Some(Self::Toggle),
            _ => None,
        }
    }

    /// Lowercase token mirroring [`Self::from_token`].
    pub fn token(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Toggle => "toggle",
        }
    }
}

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
    cursor_mode: CursorMode,
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
            cursor_mode: CursorMode::default(),
        }
    }

    /// Builder: set translation speed at construction. Default 4.5 u/sec.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Builder: override the Alt-cursor mode. Default
    /// [`CursorMode::Toggle`].
    pub fn with_cursor_mode(mut self, mode: CursorMode) -> Self {
        self.cursor_mode = mode;
        self
    }

    /// Current Alt-cursor mode.
    pub fn cursor_mode(&self) -> CursorMode {
        self.cursor_mode
    }

    /// Switch between [`CursorMode::Hold`] and [`CursorMode::Toggle`] at
    /// runtime. Does NOT change the current grab state; only how future
    /// Alt events are interpreted by [`on_alt`](Self::on_alt). Call after
    /// any user-facing setting flips, e.g. a console command.
    pub fn set_cursor_mode(&mut self, mode: CursorMode) {
        self.cursor_mode = mode;
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

    /// Apply an Alt key event according to the current [`cursor_mode`](Self::cursor_mode).
    ///
    /// - [`CursorMode::Hold`]: cursor released while `pressed=true`;
    ///   re-grabbed when `pressed=false`. Demos must forward BOTH press
    ///   and release events.
    /// - [`CursorMode::Toggle`]: `pressed=true` flips the grab once;
    ///   `pressed=false` is ignored. Demos can forward release safely.
    ///
    /// No-op when freecam isn't active.
    pub fn on_alt(&mut self, pressed: bool) {
        if !self.active {
            return;
        }
        let target_grabbed = match self.cursor_mode {
            CursorMode::Hold => !pressed,
            CursorMode::Toggle => {
                if !pressed {
                    return;
                }
                !self.cursor_grabbed
            }
        };
        if target_grabbed == self.cursor_grabbed {
            return;
        }
        self.cursor_grabbed = target_grabbed;
        self.controller.use_raw_delta = target_grabbed;
        if target_grabbed {
            crate::cursor::request_grab();
        } else {
            // Warp to window center on release so the cursor reappears in a
            // predictable spot (rather than wherever the OS cached it before
            // the grab); the user was aiming at the screen center the whole
            // time freecam was engaged.
            crate::cursor::request_release();
            crate::cursor::request_warp_to_center();
        }
    }

    /// Flip the cursor grab without changing the active state. Bind to
    /// Alt (or similar) in the demo for "release the cursor so I can
    /// click a UI widget, then re-grab to resume mouse-look." No-op if
    /// the freecam isn't active.
    ///
    /// Prefer [`on_alt`](Self::on_alt) for the standard modifier-key
    /// contract; this method ignores [`cursor_mode`](Self::cursor_mode)
    /// and always toggles.
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
            crate::cursor::request_warp_to_center();
        }
    }

    /// Per-frame update. Drives the camera's basis from yaw/pitch and
    /// the camera's position from `position` (which itself is integrated
    /// from WASD + Space/Shift). No-op when inactive or when the cursor
    /// is released (Alt-toggled in-freecam UI-access mode).
    ///
    /// Caller still owns the higher-level mode switch (orbit vs freecam);
    /// this method assumes `Freecam` IS the active controller this frame.
    pub fn advance(&mut self, input: FrameInput, camera: &mut Camera<EuclideanR3>, dt: f32) {
        if !self.active {
            return;
        }
        // Look direction: needs `cursor_grabbed` because the controller
        // reads raw mouse deltas, which only make sense while the OS has
        // the pointer trapped inside the window (or, on wasm, while
        // Pointer Lock is engaged). When the cursor is voluntarily
        // released via Alt for UI access, look-direction freezes so the
        // mouse can move over UI without dragging the camera with it.
        if self.cursor_grabbed {
            self.controller.advance(input, camera, &EuclideanR3, dt);
        }
        // Position: integrates WASD + Space/Shift in the camera's local
        // frame whenever freecam is active. Keyboard input doesn't depend
        // on cursor grab, and on wasm `cursor_grabbed` is always false
        // because Pointer Lock requires a user gesture that the current
        // engine plumbing doesn't deliver -- gating WASD on it would
        // freeze translation in the browser permanently. On native this
        // also keeps WASD moving while the user has Alt-released the
        // cursor; in practice the demo's `if !ctx.ui_has_focus` gate
        // upstream blocks the call once the user actually clicks into
        // the UI, so the change is mostly transparent.
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
    fn advance_with_cursor_released_freezes_look_but_keeps_wasd() {
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
        assert_eq!(
            cam.forward, cam_before.forward,
            "look frozen when cursor released"
        );
        assert_ne!(
            cam.position, cam_before.position,
            "WASD continues to translate when cursor released (matches wasm where Pointer Lock never engages)"
        );
    }

    #[test]
    fn default_cursor_mode_is_toggle() {
        let f = Freecam::new();
        assert_eq!(f.cursor_mode(), CursorMode::Toggle);
    }

    #[test]
    fn cursor_mode_from_token_roundtrip() {
        for m in [CursorMode::Hold, CursorMode::Toggle] {
            assert_eq!(CursorMode::from_token(m.token()), Some(m));
        }
        assert_eq!(CursorMode::from_token("nope"), None);
    }

    #[test]
    fn on_alt_hold_releases_and_regrabs() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new().with_cursor_mode(CursorMode::Hold);
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();

        f.on_alt(true);
        assert!(f.active());
        assert!(!f.cursor_grabbed());
        let (grab, _vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::None));

        f.on_alt(false);
        assert!(f.cursor_grabbed());
        let (grab, _vis) = cursor::take_pending();
        assert_eq!(grab, Some(cursor::GrabMode::Locked));
    }

    #[test]
    fn on_alt_toggle_ignores_release() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new().with_cursor_mode(CursorMode::Toggle);
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();

        // First press: flip to released.
        f.on_alt(true);
        assert!(!f.cursor_grabbed());
        let _ = cursor::take_pending();

        // Release: ignored.
        f.on_alt(false);
        assert!(!f.cursor_grabbed());
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);

        // Second press: flip back to grabbed.
        f.on_alt(true);
        assert!(f.cursor_grabbed());
    }

    #[test]
    fn on_alt_noop_when_inactive() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.on_alt(true);
        assert!(!f.active());
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);
    }

    #[test]
    fn set_cursor_mode_preserves_grab_state() {
        let _ = cursor::take_pending();
        let mut f = Freecam::new();
        f.set_active(true, Vec3::ZERO);
        let _ = cursor::take_pending();
        let grabbed_before = f.cursor_grabbed();

        f.set_cursor_mode(CursorMode::Toggle);
        assert_eq!(f.cursor_mode(), CursorMode::Toggle);
        assert_eq!(
            f.cursor_grabbed(),
            grabbed_before,
            "mode flip leaves grab alone"
        );
        let (grab, vis) = cursor::take_pending();
        assert_eq!(grab, None);
        assert_eq!(vis, None);
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
