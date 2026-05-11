//! Camera abstractions for Rye.
//!
//! Two layers, used together:
//!
//! 1. [`Camera<S>`]: Space-generic position + tangent frame. Pure data; storage
//!    agnostic to which controller is driving it. Works for any `Space` whose `Point`
//!    and `Vector` are `glam::Vec3` (i.e. all the closed-form 3D Spaces today).
//! 2. [`CameraController`]: input-driven logic that mutates a `Camera<S>` each frame.
//!    Concrete impls: [`OrbitController`], [`FirstPersonController`].
//!
//! ## Legacy Euclidean-only types
//!
//! [`OrbitCamera`] and [`FirstPersonCamera`] predate the Space-generic refactor. They remain
//! exported and unchanged so existing examples build during incremental migration; new code
//! should prefer `OrbitController<EuclideanR3>` / `FirstPersonController<EuclideanR3>`. They
//! will be removed once every example has migrated.

mod camera;
mod controller;

pub use camera::Camera;
pub use controller::{CameraController, FirstPersonController, OrbitController};

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};
use rye_input::FrameInput;
use rye_math::Iso3;

const ORBIT_RADIANS_PER_PIXEL: f32 = 0.006;
const ZOOM_LOG_STEP: f32 = 0.12;
const MIN_DISTANCE: f32 = 1.5;
const MAX_DISTANCE: f32 = 8.0;
const INITIAL_HEIGHT: f32 = 0.6;
const INITIAL_RADIUS: f32 = 3.5;
const MIN_PITCH: f32 = -1.45;
const MAX_PITCH: f32 = 1.45;

const FIRST_PERSON_MOUSE_SENSITIVITY: f32 = 0.002;
const FIRST_PERSON_MIN_PITCH: f32 = -FRAC_PI_2 + 0.02;
const FIRST_PERSON_MAX_PITCH: f32 = FRAC_PI_2 - 0.02;

/// Camera basis vectors produced each frame; feed directly into shader uniforms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

// ---------------------------------------------------------------------------
// Orbit camera (extracted from examples/fractal/camera.rs)
// ---------------------------------------------------------------------------

/// Spherical-coordinate orbit camera that circles a target point.
/// Left-drag to orbit, scroll to zoom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    orientation: Iso3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        let distance = (INITIAL_RADIUS * INITIAL_RADIUS + INITIAL_HEIGHT * INITIAL_HEIGHT).sqrt();
        let pitch = -(INITIAL_HEIGHT / distance).asin();
        let mut cam = Self {
            target: Vec3::ZERO,
            yaw: FRAC_PI_2,
            pitch,
            distance,
            orientation: Iso3::IDENTITY,
        };
        cam.rebuild_orientation();
        cam
    }
}

impl OrbitCamera {
    pub fn advance(&mut self, input: FrameInput) {
        if input.left_mouse_down {
            self.yaw -= input.mouse_delta.x * ORBIT_RADIANS_PER_PIXEL;
            self.pitch = (self.pitch - input.mouse_delta.y * ORBIT_RADIANS_PER_PIXEL)
                .clamp(MIN_PITCH, MAX_PITCH);
        }
        if input.scroll_lines != 0.0 {
            self.distance = (self.distance * (-input.scroll_lines * ZOOM_LOG_STEP).exp())
                .clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
        self.rebuild_orientation();
    }

    pub fn view(&self) -> CameraView {
        let right = self.orientation.rotation * Vec3::X;
        let up = self.orientation.rotation * Vec3::Y;
        let back = self.orientation.rotation * Vec3::Z;
        CameraView {
            position: self.target + back * self.distance,
            forward: -back,
            right,
            up,
        }
    }

    /// Advance yaw by `delta` radians; used by auto-rotate mode.
    pub fn rotate_yaw(&mut self, delta: f32) {
        self.yaw += delta;
        self.rebuild_orientation();
    }

    /// Snap to a fixed orbit position; used by capture / movie mode.
    pub fn set_orbit(&mut self, distance: f32, pitch: f32) {
        self.distance = distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.pitch = pitch.clamp(MIN_PITCH, MAX_PITCH);
        self.yaw = 0.0;
        self.rebuild_orientation();
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    fn rebuild_orientation(&mut self) {
        let yaw = Quat::from_rotation_y(self.yaw);
        let pitch = Quat::from_rotation_x(self.pitch);
        self.orientation = Iso3::from_rotation(yaw * pitch);
    }
}

// ---------------------------------------------------------------------------
// First-person camera
// ---------------------------------------------------------------------------

/// Free-look first-person camera. The caller owns the position (e.g. inside a
/// `rye_player::PlayerState`); this type tracks only yaw and pitch.
///
/// Call [`FirstPersonCamera::advance_look`] with the frame's mouse delta, then
/// [`FirstPersonCamera::view`] with the current world position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirstPersonCamera {
    yaw: f32,
    pitch: f32,
}

impl Default for FirstPersonCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

impl FirstPersonCamera {
    pub fn new(yaw: f32, pitch: f32) -> Self {
        Self {
            yaw,
            pitch: pitch.clamp(FIRST_PERSON_MIN_PITCH, FIRST_PERSON_MAX_PITCH),
        }
    }

    /// Rotate look direction from mouse delta. Always integrates the delta, so
    /// pointer-locked windows just call it every frame. Callers that want a
    /// hold-to-look UX wrap the call in their own `if input.left_mouse_down { ... }`
    /// (or right-button gate); doing it inside the controller would force the always-on
    /// case to fight against gating, which is the more common mode in SDF-render demos.
    pub fn advance_look(&mut self, input: FrameInput) {
        self.yaw -= input.mouse_delta.x * FIRST_PERSON_MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - input.mouse_delta.y * FIRST_PERSON_MOUSE_SENSITIVITY)
            .clamp(FIRST_PERSON_MIN_PITCH, FIRST_PERSON_MAX_PITCH);
    }

    pub fn view(&self, position: Vec3) -> CameraView {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        let rot = yaw_q * pitch_q;
        CameraView {
            position,
            forward: rot * -Vec3::Z,
            right: rot * Vec3::X,
            up: rot * Vec3::Y,
        }
    }

    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} to be within 1e-5 of {b}"
        );
    }

    #[test]
    fn starts_at_previous_static_orbit_view() {
        let view = OrbitCamera::default().view();
        assert_close(view.position.x, 3.5);
        assert_close(view.position.y, 0.6);
        assert_close(view.position.z, 0.0);
        assert_close(view.right.x, 0.0);
        assert_close(view.right.y, 0.0);
        assert_close(view.right.z, -1.0);
    }

    #[test]
    fn left_drag_orbits_camera() {
        let mut camera = OrbitCamera::default();
        let before = camera.view();
        camera.advance(FrameInput {
            mouse_delta: Vec2::new(50.0, -20.0),
            left_mouse_down: true,
            ..FrameInput::default()
        });
        let after = camera.view();
        assert_ne!(before.position, after.position);
        assert_close(after.forward.length(), 1.0);
        assert_close(after.right.length(), 1.0);
        assert_close(after.up.length(), 1.0);
    }

    #[test]
    fn wheel_zoom_clamps_distance() {
        let mut camera = OrbitCamera::default();
        camera.advance(FrameInput {
            scroll_lines: 100.0,
            ..FrameInput::default()
        });
        assert_close(camera.distance(), MIN_DISTANCE);
        camera.advance(FrameInput {
            scroll_lines: -100.0,
            ..FrameInput::default()
        });
        assert_close(camera.distance(), MAX_DISTANCE);
    }

    #[test]
    fn first_person_view_is_normalized() {
        let cam = FirstPersonCamera::new(0.3, 0.2);
        let view = cam.view(Vec3::ZERO);
        assert_close(view.forward.length(), 1.0);
        assert_close(view.right.length(), 1.0);
        assert_close(view.up.length(), 1.0);
    }

    #[test]
    fn first_person_pitch_clamps() {
        let mut cam = FirstPersonCamera::default();
        cam.advance_look(FrameInput {
            mouse_delta: Vec2::new(0.0, 1e9),
            ..FrameInput::default()
        });
        assert!(cam.pitch() >= FIRST_PERSON_MIN_PITCH);
        assert!(cam.pitch() <= FIRST_PERSON_MAX_PITCH);
    }
}
