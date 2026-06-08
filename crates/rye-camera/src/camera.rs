//! Space-generic camera state: a position on the manifold plus an
//! orthonormal frame in its tangent space.
//!
//! [`Camera<S>`] stores a `position: S::Point` and three tangent basis
//! vectors (`right`, `up`, `forward`); the frame is the orientation,
//! there is no separate rotation field. This orthonormal-frame-bundle
//! form avoids the per-Space convention questions of the `Iso` types:
//! [`rye_math::Space::parallel_transport_along`] moves the frame
//! correctly for any Space.
//!
//! [`Camera::view`] yields a [`CameraView`] for direct shader upload,
//! available where `S::Point` and `S::Vector` are both `glam::Vec3`.

use glam::Vec3;
use rye_math::Space;
use std::ops::Mul;

use crate::CameraView;

/// Position + orthonormal tangent frame at that position. Generic over
/// any [`Space`]; `view` and `translate` require `S::Point = S::Vector =
/// Vec3`.
///
/// ## Invariants (caller-maintained, not type-enforced)
///
/// - `right`, `up`, `forward` are pairwise-orthogonal Euclidean-unit
///   vectors in the Space's embedding; the WGSL prelude applies the
///   metric, so storing Riemannian-unit vectors would leak embedding
///   scale factors into the renderer.
/// - Right-handed: `forward` is the look direction, so `right × up =
///   -forward`. Matches [`crate::OrbitCamera`] and the WGSL prelude.
/// - Construct via [`Camera::looking_at`] or a
///   [`crate::CameraController`]; mutating the basis by hand drifts off
///   orthonormal under `translate`.
#[derive(Clone, Copy, Debug)]
pub struct Camera<S: Space> {
    pub position: S::Point,
    pub right: S::Vector,
    pub up: S::Vector,
    pub forward: S::Vector,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> Camera<S> {
    /// Origin camera looking down −Z, 60° vertical FOV, near/far for
    /// unit-scale scenes. A `Default` stand-in for `S` instances that
    /// can't derive it; controllers usually overwrite the pose at once.
    pub fn at_origin() -> Self {
        Self {
            position: Vec3::ZERO,
            right: Vec3::X,
            up: Vec3::Y,
            forward: -Vec3::Z,
            fov_y: 60.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.05,
            far: 100.0,
        }
    }

    /// Place the camera at `position` looking toward `target`, with
    /// `world_up` as the up hint. Forward is the normalised
    /// `Space::log(position, target)`: the initial geodesic velocity from
    /// `position` to `target`.
    pub fn looking_at(position: Vec3, target: Vec3, world_up: Vec3, space: &S) -> Self {
        let log = space.log(position, target);
        let forward = log.try_normalize().unwrap_or(-Vec3::Z);
        // Right-handed: right = forward × world_up, up = right × forward.
        let right = forward.cross(world_up).try_normalize().unwrap_or(Vec3::X);
        let up = right.cross(forward);
        Self {
            position,
            right,
            up,
            forward,
            fov_y: 60.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.05,
            far: 100.0,
        }
    }

    /// Renderer-facing view basis, in the Space's embedding coordinates
    /// the shader expects (the WGSL prelude applies the metric).
    pub fn view(&self) -> CameraView {
        CameraView {
            position: self.position,
            forward: self.forward,
            right: self.right,
            up: self.up,
        }
    }

    /// Move along the geodesic from `v * dt`, parallel-transporting the
    /// frame so it stays orthonormal at the new point. Identity on the
    /// basis in flat space; a holonomy rotation in H³ / S³.
    pub fn translate(&mut self, v: S::Vector, dt: f32, space: &S)
    where
        S::Vector: Mul<f32, Output = S::Vector>,
    {
        let new_pos = space.exp(self.position, v * dt);
        // Re-normalise to Euclidean-unit: transport preserves Riemannian
        // length, but the Poincaré-ball / S³ embedding scales Euclidean
        // length by a position-dependent factor, and the renderer expects
        // Euclidean-unit directions.
        let path = [self.position, new_pos];
        self.right = space
            .parallel_transport_along(&path, self.right)
            .try_normalize()
            .unwrap_or(self.right);
        self.up = space
            .parallel_transport_along(&path, self.up)
            .try_normalize()
            .unwrap_or(self.up);
        self.forward = space
            .parallel_transport_along(&path, self.forward)
            .try_normalize()
            .unwrap_or(self.forward);
        self.position = new_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rye_math::EuclideanR3;

    fn close(a: Vec3, b: Vec3, tol: f32) {
        assert!((a - b).length() < tol, "expected {a:?} ≈ {b:?}");
    }

    #[test]
    fn at_origin_is_orthonormal() {
        let cam = Camera::<EuclideanR3>::at_origin();
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
        assert!((cam.up.length() - 1.0).abs() < 1e-6);
        assert!((cam.forward.length() - 1.0).abs() < 1e-6);
        // Right-handed: right × up = -forward.
        close(cam.right.cross(cam.up), -cam.forward, 1e-6);
        assert!(cam.right.dot(cam.up).abs() < 1e-6);
        assert!(cam.right.dot(cam.forward).abs() < 1e-6);
        assert!(cam.up.dot(cam.forward).abs() < 1e-6);
    }

    #[test]
    fn looking_at_target_points_toward_it() {
        let space = EuclideanR3;
        let cam = Camera::<EuclideanR3>::looking_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            &space,
        );
        close(cam.forward, -Vec3::Z, 1e-6);
        close(cam.up, Vec3::Y, 1e-6);
    }

    #[test]
    fn translate_in_flat_space_preserves_frame() {
        let space = EuclideanR3;
        let mut cam = Camera::<EuclideanR3>::at_origin();
        let original = cam.view();
        cam.translate(Vec3::new(1.0, 2.0, -3.0), 1.0, &space);
        close(cam.position, Vec3::new(1.0, 2.0, -3.0), 1e-6);
        // Transport is the identity in E³.
        close(cam.right, original.right, 1e-6);
        close(cam.up, original.up, 1e-6);
        close(cam.forward, original.forward, 1e-6);
    }

    #[test]
    fn translate_view_position_matches_exp() {
        let space = EuclideanR3;
        let mut cam = Camera::<EuclideanR3>::at_origin();
        cam.translate(Vec3::X, 2.5, &space);
        close(cam.view().position, Vec3::new(2.5, 0.0, 0.0), 1e-6);
    }

    /// `translate` in H³ keeps the position in the Poincaré ball and the
    /// basis finite + ≈orthonormal. Catches NaN drift in `Space::exp` /
    /// `parallel_transport_along` for `HyperbolicH3`.
    #[test]
    fn translate_in_hyperbolic_h3_stays_in_ball_and_orthonormal() {
        use rye_math::HyperbolicH3;
        let space = HyperbolicH3;
        let mut cam = Camera::<HyperbolicH3>::at_origin();
        cam.translate(Vec3::X, 0.5, &space);
        assert!(
            cam.position.length() < 1.0,
            "camera escaped Poincaré ball: {:?}",
            cam.position
        );
        assert!(cam.right.is_finite());
        assert!(cam.up.is_finite());
        assert!(cam.forward.is_finite());
        assert!((cam.right.length() - 1.0).abs() < 1e-3);
        assert!((cam.up.length() - 1.0).abs() < 1e-3);
        assert!((cam.forward.length() - 1.0).abs() < 1e-3);
        assert!(cam.right.dot(cam.up).abs() < 1e-3);
        assert!(cam.right.dot(cam.forward).abs() < 1e-3);
        assert!(cam.up.dot(cam.forward).abs() < 1e-3);
    }

    /// `looking_at` with `position == target` has `log = 0`, no defined
    /// forward; the fallback must stay finite and orthonormal.
    #[test]
    fn looking_at_collapsed_target_falls_back_to_finite_frame() {
        let cam = Camera::<EuclideanR3>::looking_at(Vec3::ZERO, Vec3::ZERO, Vec3::Y, &EuclideanR3);
        assert!(cam.forward.is_finite() && cam.right.is_finite() && cam.up.is_finite());
        // Direction is unspecified under fallback; only length is pinned.
        assert!((cam.forward.length() - 1.0).abs() < 1e-6);
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
        assert!((cam.up.length() - 1.0).abs() < 1e-6);
    }

    /// `looking_at` with `forward` parallel to `world_up` makes
    /// `forward × world_up = 0`; the fallback `right` keeps the frame
    /// finite.
    #[test]
    fn looking_at_world_up_parallel_to_forward_falls_back() {
        let cam = Camera::<EuclideanR3>::looking_at(
            Vec3::new(0.0, 0.0, 0.0),
            // Look straight up, parallel to world_up +Y.
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            &EuclideanR3,
        );
        assert!(cam.forward.is_finite() && cam.right.is_finite() && cam.up.is_finite());
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
    }
}
