//! [`CameraController`] is the input-driven logic that mutates a [`Camera`] each frame.
//! Camera is data, controller is logic; they live in separate types so a game can swap
//! controllers without reconstructing the camera state.
//!
//! ## Concrete controllers shipped here
//!
//! - [`OrbitController`]: spherical-coordinate orbit around a target. The current
//!   default for SDF-rendering demos.
//! - [`FirstPersonController`]: yaw/pitch free-look from a user-controlled position.
//!   Pairs naturally with `rye_player::PlayerState` for WASD-style movement.

use std::f32::consts::FRAC_PI_2;
use std::marker::PhantomData;

use glam::{Quat, Vec3};
use rye_input::FrameInput;
use rye_math::Space;

use crate::Camera;

const ORBIT_RADIANS_PER_PIXEL: f32 = 0.006;
const ZOOM_LOG_STEP: f32 = 0.12;
const MIN_DISTANCE: f32 = 1.5;
const MAX_DISTANCE: f32 = 8.0;
const INITIAL_HEIGHT: f32 = 0.6;
const INITIAL_RADIUS: f32 = 3.5;
const MIN_ORBIT_PITCH: f32 = -1.45;
const MAX_ORBIT_PITCH: f32 = 1.45;

const FIRST_PERSON_MOUSE_SENSITIVITY: f32 = 0.002;
const FIRST_PERSON_MIN_PITCH: f32 = -FRAC_PI_2 + 0.02;
const FIRST_PERSON_MAX_PITCH: f32 = FRAC_PI_2 - 0.02;

/// Drives camera state from per-frame input. Implementations own their own
/// controller-specific state (orbit angles, etc.) and write the resulting pose into
/// the [`Camera`] each call.
pub trait CameraController<S: Space> {
    /// Read the frame's input, update internal controller state, and write the
    /// resulting position + tangent frame into `camera`. `dt` is the wall-clock
    /// seconds since the last `advance` call (frame-rate-independent controllers can
    /// use it; orbit ignores it).
    fn advance(&mut self, input: FrameInput, camera: &mut Camera<S>, space: &S, dt: f32);
}

// ---------------------------------------------------------------------------
// OrbitController
// ---------------------------------------------------------------------------

/// Spherical-coordinate orbit camera that circles a target point. Left-drag orbits;
/// scroll zooms.
///
/// In flat space this is the same math as the legacy [`crate::OrbitCamera`]. In H³ /
/// S³ the camera position is computed by `Space::exp` from the target along the
/// orbit-direction tangent vector, and the camera basis parallel-transports from the
/// target to the camera position so it arrives orthonormal in any geometry.
#[derive(Clone, Copy, Debug)]
pub struct OrbitController<S: Space> {
    /// Orbit centre in the manifold's coordinates.
    pub target: S::Point,
    /// Yaw around `world_up` (tangent direction at `target`).
    pub yaw: f32,
    /// Pitch about the right axis. Clamped to `[-1.45, 1.45]` so the camera doesn't
    /// flip through poles.
    pub pitch: f32,
    /// Geodesic distance from `target` to the camera position.
    pub distance: f32,
    _marker: PhantomData<S>,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> Default for OrbitController<S> {
    fn default() -> Self {
        let distance = (INITIAL_RADIUS * INITIAL_RADIUS + INITIAL_HEIGHT * INITIAL_HEIGHT).sqrt();
        let pitch = -(INITIAL_HEIGHT / distance).asin();
        Self {
            target: Vec3::ZERO,
            yaw: FRAC_PI_2,
            pitch,
            distance,
            _marker: PhantomData,
        }
    }
}

impl<S: Space<Point = Vec3, Vector = Vec3>> OrbitController<S> {
    /// Build an orbit controller around `target` at the default pose. `target` is in
    /// the Space's own coordinates.
    pub fn around(target: Vec3) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }

    /// Snap to a fixed orbit position. Used by capture / movie mode (mirror of the
    /// legacy [`crate::OrbitCamera::set_orbit`]).
    pub fn set_orbit(&mut self, distance: f32, pitch: f32) {
        self.distance = distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.pitch = pitch.clamp(MIN_ORBIT_PITCH, MAX_ORBIT_PITCH);
        self.yaw = 0.0;
    }

    /// Advance yaw by `delta` radians; used by auto-rotate mode.
    pub fn rotate_yaw(&mut self, delta: f32) {
        self.yaw += delta;
    }

    /// Local "back" tangent vector at `target`: the direction pointing from `target`
    /// toward the camera. Encodes the orbit angles in canonical xyz so the same
    /// yaw/pitch convention works in any Space.
    fn back_at_target(&self) -> Vec3 {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        (yaw_q * pitch_q) * Vec3::Z
    }

    fn right_at_target(&self) -> Vec3 {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        (yaw_q * pitch_q) * Vec3::X
    }

    fn up_at_target(&self) -> Vec3 {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        (yaw_q * pitch_q) * Vec3::Y
    }
}

impl<S: Space<Point = Vec3, Vector = Vec3>> CameraController<S> for OrbitController<S> {
    fn advance(&mut self, input: FrameInput, camera: &mut Camera<S>, space: &S, _dt: f32) {
        if input.left_mouse_down {
            self.yaw -= input.mouse_delta.x * ORBIT_RADIANS_PER_PIXEL;
            self.pitch = (self.pitch - input.mouse_delta.y * ORBIT_RADIANS_PER_PIXEL)
                .clamp(MIN_ORBIT_PITCH, MAX_ORBIT_PITCH);
        }
        if input.scroll_lines != 0.0 {
            self.distance = (self.distance * (-input.scroll_lines * ZOOM_LOG_STEP).exp())
                .clamp(MIN_DISTANCE, MAX_DISTANCE);
        }

        // Orbit-direction tangent vector at `target`.
        let back_at_target = self.back_at_target();
        let right_at_target = self.right_at_target();
        let up_at_target = self.up_at_target();

        // Camera position is `target` exp'd along that direction
        // by `distance` units.
        let cam_pos = space.exp(self.target, back_at_target * self.distance);

        // Parallel-transport the basis from target to camera, then normalise. In flat
        // space transport is the identity and the inputs were already unit; in H³ / S³
        // the transport preserves *Riemannian* length, but the Poincaré-ball /
        // S³-embedding scales Euclidean length by a position-dependent factor,
        // re-normalising restores the Euclidean-unit convention the renderer expects
        // (the WGSL prelude handles the metric on those Euclidean-unit ray directions).
        let path = [self.target, cam_pos];
        let cam_right = space
            .parallel_transport_along(&path, right_at_target)
            .try_normalize()
            .unwrap_or(Vec3::X);
        let cam_up = space
            .parallel_transport_along(&path, up_at_target)
            .try_normalize()
            .unwrap_or(Vec3::Y);
        let cam_back = space
            .parallel_transport_along(&path, back_at_target)
            .try_normalize()
            .unwrap_or(Vec3::Z);

        camera.position = cam_pos;
        camera.right = cam_right;
        camera.up = cam_up;
        camera.forward = -cam_back;
    }
}

// ---------------------------------------------------------------------------
// FirstPersonController
// ---------------------------------------------------------------------------

/// Free-look first-person controller. The caller owns the position (typically
/// `rye_player::PlayerState`'s `position`); this controller only manages the look
/// direction.
///
/// Use by setting `camera.position` directly each frame (or via player physics), then
/// calling `advance` to update the basis from yaw/pitch.
///
/// `advance` always integrates the mouse delta; pointer-locked windows just call it
/// every frame. Apps that want hold-to-look gate the call themselves
/// (`if input.left_mouse_down { ctrl.advance(...) }`).
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstPersonController<S: Space> {
    pub yaw: f32,
    pub pitch: f32,
    /// When `true`, integrate [`FrameInput::mouse_raw_delta`] instead of
    /// `mouse_delta`. Pair with a grabbed cursor (see `rye_app::cursor`):
    /// the OS reports raw device motion that doesn't cap at the screen
    /// edge, so a fast horizontal flick can yaw the camera arbitrarily far
    /// without the cursor "running out of room." Default off (uses the
    /// clamped delta), which is what the orbit-mode camera consumers want.
    pub use_raw_delta: bool,
    _marker: PhantomData<S>,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> FirstPersonController<S> {
    pub fn new(yaw: f32, pitch: f32) -> Self {
        Self {
            yaw,
            pitch: pitch.clamp(FIRST_PERSON_MIN_PITCH, FIRST_PERSON_MAX_PITCH),
            use_raw_delta: false,
            _marker: PhantomData,
        }
    }
}

impl<S: Space<Point = Vec3, Vector = Vec3>> CameraController<S> for FirstPersonController<S> {
    fn advance(&mut self, input: FrameInput, camera: &mut Camera<S>, _space: &S, _dt: f32) {
        let delta = if self.use_raw_delta {
            input.mouse_raw_delta
        } else {
            input.mouse_delta
        };
        self.yaw -= delta.x * FIRST_PERSON_MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - delta.y * FIRST_PERSON_MOUSE_SENSITIVITY)
            .clamp(FIRST_PERSON_MIN_PITCH, FIRST_PERSON_MAX_PITCH);

        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);
        let rot = yaw_q * pitch_q;
        // The basis is in T_position M; for the closed-form Spaces currently supported,
        // we treat the canonical xyz as the local frame. Honest geodesic transport
        // between frames (parallel-transport along the polyline the camera traversed)
        // belongs in a dedicated controller; not implemented yet.
        camera.right = rot * Vec3::X;
        camera.up = rot * Vec3::Y;
        camera.forward = rot * -Vec3::Z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use rye_math::EuclideanR3;

    fn close(a: Vec3, b: Vec3, tol: f32) {
        assert!((a - b).length() < tol, "expected {a:?} ≈ {b:?}");
    }

    #[test]
    fn orbit_default_in_e3_matches_legacy_pose() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: OrbitController<EuclideanR3> = OrbitController::default();
        ctrl.advance(FrameInput::default(), &mut camera, &EuclideanR3, 0.0);
        // Same expected pose as the legacy `OrbitCamera::default()`
        // test in lib.rs: position (3.5, 0.6, 0).
        close(camera.position, Vec3::new(3.5, 0.6, 0.0), 1e-5);
        // Frame is orthonormal.
        assert!((camera.right.length() - 1.0).abs() < 1e-5);
        assert!((camera.up.length() - 1.0).abs() < 1e-5);
        assert!((camera.forward.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orbit_left_drag_moves_camera() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: OrbitController<EuclideanR3> = OrbitController::default();
        ctrl.advance(FrameInput::default(), &mut camera, &EuclideanR3, 0.0);
        let before = camera.position;
        ctrl.advance(
            FrameInput {
                mouse_delta: Vec2::new(50.0, -20.0),
                left_mouse_down: true,
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert_ne!(before, camera.position);
        // Frame stays unit-length after orbit.
        assert!((camera.forward.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orbit_scroll_clamps_distance() {
        let mut ctrl: OrbitController<EuclideanR3> = OrbitController::default();
        let mut camera = Camera::<EuclideanR3>::at_origin();
        ctrl.advance(
            FrameInput {
                scroll_lines: 100.0,
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert!((ctrl.distance - MIN_DISTANCE).abs() < 1e-5);
        ctrl.advance(
            FrameInput {
                scroll_lines: -100.0,
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert!((ctrl.distance - MAX_DISTANCE).abs() < 1e-5);
    }

    #[test]
    fn first_person_view_is_normalised() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: FirstPersonController<EuclideanR3> = FirstPersonController::new(0.3, 0.2);
        ctrl.advance(FrameInput::default(), &mut camera, &EuclideanR3, 0.0);
        assert!((camera.forward.length() - 1.0).abs() < 1e-5);
        assert!((camera.right.length() - 1.0).abs() < 1e-5);
        assert!((camera.up.length() - 1.0).abs() < 1e-5);
    }

    /// `OrbitController<S>` should produce valid, finite, unit-length frames in any 3D
    /// Space. This pins that the generic impl actually works against `HyperbolicH3` and
    /// `SphericalS3`, not just the closed-form-flat path. Catches regressions in
    /// `Space::exp` / `parallel_transport_along` that would surface as NaN frames or
    /// out-of-domain points.
    #[test]
    fn orbit_in_hyperbolic_h3_produces_valid_frame() {
        use rye_math::HyperbolicH3;
        let mut camera = Camera::<HyperbolicH3>::at_origin();
        let mut ctrl: OrbitController<HyperbolicH3> = OrbitController::around(Vec3::ZERO);
        // Smaller distance because the Poincaré ball is bounded by `|p| < 1`; the
        // default 3.5 would push the camera outside the model.
        ctrl.distance = 0.4;
        ctrl.advance(FrameInput::default(), &mut camera, &HyperbolicH3, 0.0);
        // Position is inside the Poincaré ball.
        assert!(
            camera.position.length() < 1.0,
            "camera escaped the Poincaré ball: {:?}",
            camera.position
        );
        // Frame is finite and unit-ish (some f32 wobble OK; we just rule out NaN /
        // unbounded drift).
        assert!(camera.right.is_finite() && camera.up.is_finite() && camera.forward.is_finite());
        assert!((camera.right.length() - 1.0).abs() < 1e-3);
        assert!((camera.up.length() - 1.0).abs() < 1e-3);
        assert!((camera.forward.length() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn orbit_in_spherical_s3_produces_valid_frame() {
        use rye_math::SphericalS3;
        let mut camera = Camera::<SphericalS3>::at_origin();
        let mut ctrl: OrbitController<SphericalS3> = OrbitController::around(Vec3::ZERO);
        ctrl.distance = 0.5;
        ctrl.advance(FrameInput::default(), &mut camera, &SphericalS3, 0.0);
        // S³ embeds the upper hemisphere with `|p| < 1`; same domain check as H³.
        assert!(camera.position.length() < 1.0);
        assert!(camera.right.is_finite() && camera.up.is_finite() && camera.forward.is_finite());
        assert!((camera.right.length() - 1.0).abs() < 1e-3);
        assert!((camera.up.length() - 1.0).abs() < 1e-3);
        assert!((camera.forward.length() - 1.0).abs() < 1e-3);
    }

    /// `OrbitController` against `BlendedSpace<E3, H3, LinearBlendX>` exercises the
    /// variable-metric `parallel_transport_along` path the closed-form Spaces never hit.
    /// Pin: targeting the H³-side of a transition zone, the orbit produces a finite,
    /// ≈-orthonormal frame inside the Poincaré ball.
    #[test]
    fn orbit_in_blended_e3_h3_produces_valid_frame() {
        use rye_math::{BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX};
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-2.0, 2.0));
        // Orbit around a point firmly inside the H³ region (x > 2).
        let mut camera =
            Camera::<BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>>::at_origin();
        camera.position = Vec3::new(2.5, 0.0, 0.0);
        let mut ctrl = OrbitController::around(Vec3::new(2.5, 0.0, 0.0));
        ctrl.distance = 0.4;
        ctrl.advance(FrameInput::default(), &mut camera, &space, 0.0);
        assert!(camera.position.is_finite());
        assert!(camera.right.is_finite() && camera.up.is_finite() && camera.forward.is_finite());
        // Tolerances are looser than closed-form because the variable-metric integrator
        // has finite step error.
        assert!((camera.right.length() - 1.0).abs() < 1e-2);
        assert!((camera.up.length() - 1.0).abs() < 1e-2);
        assert!((camera.forward.length() - 1.0).abs() < 1e-2);
    }

    /// Per-Space `OrbitController::advance` timing. `#[ignore]` by default, run on
    /// demand via
    ///
    /// ```text
    /// cargo test --release --package rye-camera \
    ///     -- --ignored --nocapture orbit_advance_perf
    /// ```
    ///
    /// to print median timings for E³ / H³ / S³. Smoke-test for the design-doc claim
    /// that the controller's hot path is sub-microsecond per call (the framework calls
    /// it once per frame, so anything close to a microsecond is "free" relative to a
    /// 16 ms frame budget). Catches accidental quadratic-blowups in `Space::exp` /
    /// `parallel_transport_along` impls.
    ///
    /// Not a CI gate: `cargo test --release` is heavy and the timing numbers are
    /// machine-dependent. Run when changing camera or Space hot-path code; eyeball the
    /// output.
    #[test]
    #[ignore]
    fn orbit_advance_perf() {
        use rye_math::{HyperbolicH3, SphericalS3};
        use std::time::Instant;

        const ITERATIONS: u32 = 100_000;

        fn bench<S>(label: &str, space: &S, distance: f32)
        where
            S: rye_math::Space<Point = Vec3, Vector = Vec3>,
        {
            let mut camera = Camera::<S>::at_origin();
            let mut ctrl: OrbitController<S> = OrbitController::around(Vec3::ZERO);
            ctrl.distance = distance;
            // Warm-up.
            for _ in 0..1_000 {
                ctrl.advance(FrameInput::default(), &mut camera, space, 0.0);
            }
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                ctrl.advance(FrameInput::default(), &mut camera, space, 0.0);
                std::hint::black_box(camera.view());
            }
            let elapsed = start.elapsed();
            let per_call_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
            println!("[orbit_advance_perf] {label:>14}: {per_call_ns:>7.0} ns / call");
        }

        bench("EuclideanR3", &EuclideanR3, 3.5);
        bench("HyperbolicH3", &HyperbolicH3, 0.4);
        bench("SphericalS3", &SphericalS3, 0.5);
    }

    #[test]
    fn first_person_pitch_clamps() {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        let mut ctrl: FirstPersonController<EuclideanR3> = FirstPersonController::default();
        ctrl.advance(
            FrameInput {
                mouse_delta: Vec2::new(0.0, 1e9),
                ..FrameInput::default()
            },
            &mut camera,
            &EuclideanR3,
            0.0,
        );
        assert!(ctrl.pitch >= FIRST_PERSON_MIN_PITCH);
        assert!(ctrl.pitch <= FIRST_PERSON_MAX_PITCH);
    }
}
