//! Space-generic camera state: a position on the manifold plus an orthonormal frame in
//! its tangent space.
//!
//! ## Representation
//!
//! [`Camera<S>`] stores a `position: S::Point` plus three tangent basis vectors at
//! that point: `right`, `up`, `forward`. This is the orthonormal-frame-bundle
//! representation of a camera, the same way differential geometry treats observers in
//! any Riemannian manifold. There is no separate "rotation" field; the frame *is* the
//! orientation.
//!
//! Why three vectors instead of `S::Iso`? `Iso` types across the engine (Iso3, Iso3H,
//! Iso4) conflate translation and rotation with conventions that vary per Space.
//! Storing the tangent frame directly sidesteps the convention question entirely:
//! parallel-transport via [`rye_math::Space::parallel_transport_along`] handles motion
//! correctly for any Space.
//!
//! ## Renderer integration
//!
//! [`Camera::view`] produces a [`CameraView`] with `Vec3` basis vectors suitable for
//! direct shader-uniform upload. Available today for any Space whose `Point` and
//! `Vector` are both `glam::Vec3`; the 4D camera will get its own type when the 4D
//! render path lands.

use glam::Vec3;
use rye_math::Space;
use std::ops::Mul;

use crate::CameraView;

/// Position + orthonormal tangent frame at that position. Generic over any [`Space`];
/// the `view` and `translate` methods below require `S::Point = Vec3` and
/// `S::Vector = Vec3` (the convention for every closed-form 3D Space currently in
/// `rye-math`).
///
/// ## Invariants (caller-maintained, not type-enforced)
///
/// - `right`, `up`, `forward` are pairwise-orthogonal **Euclidean-unit** vectors in
///   the Space's embedding (Cartesian for E³, Poincaré-ball for H³, unit-3-sphere for
///   S³). The WGSL prelude handles the actual metric on those Euclidean-unit
///   directions; storing Riemannian-unit vectors instead would force the renderer to
///   know about embedding scale factors, which is not the prelude split's intent.
/// - Right-handed-camera convention: `forward` points where the camera looks, so
///   `right × up = -forward` (the "back" direction). Matches the legacy
///   [`crate::OrbitCamera`] and the WGSL prelude expectations.
/// - Construct via [`Camera::looking_at`] or via a [`crate::CameraController`] to
///   keep the frame orthonormal. Manually mutating the basis without re-orthonormalising
///   will drift over time under `translate`.
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
    /// Default-ish camera: at the origin, looking down −Z, with 60° vertical FOV and a
    /// sensible near/far for unit-scale scenes. Most callers want a controller to
    /// overwrite the pose immediately; this exists so a "default" camera is a thing for
    /// `S` instances that can't auto-derive `Default`.
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

    /// Place the camera at `position` looking toward `target`, with `world_up` as the
    /// up-direction hint. Resolves into an orthonormal frame in the tangent space at
    /// `position`. For curved Spaces, "look toward `target`" means: the forward
    /// direction is `Space::log(position, target)` normalised, the initial geodesic
    /// velocity that would carry you from `position` to `target`.
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

    /// Compute the renderer-facing view basis. All four components land in the same
    /// coordinate system the shader expects (the Space's chosen embedding, Cartesian
    /// for E³, Poincaré-ball for H³, unit-3-sphere for S³, with the WGSL prelude
    /// handling the metric).
    pub fn view(&self) -> CameraView {
        CameraView {
            position: self.position,
            forward: self.forward,
            right: self.right,
            up: self.up,
        }
    }

    /// Move the camera by tangent vector `v` for time `dt`. The position advances along
    /// the geodesic, and the frame parallel-transports along that same geodesic so the
    /// camera arrives with a still-orthonormal tangent frame at the new point.
    ///
    /// In flat space this is `position += v * dt` and the basis is unchanged. In H³ /
    /// S³ the basis rotates by the holonomy of the geodesic step. In a future
    /// variable-metric Space (`BlendedSpace`) the geodesic is a single integration step
    /// and the basis transports honestly through the changing metric.
    pub fn translate(&mut self, v: S::Vector, dt: f32, space: &S)
    where
        S::Vector: Mul<f32, Output = S::Vector>,
    {
        let new_pos = space.exp(self.position, v * dt);
        // Use the path-aware primitive over the 2-point polyline so future Spaces with
        // non-trivial geodesic-construction cost can override `parallel_transport_along`
        // and skip the BVP. Re-normalise the transported basis to Euclidean-unit length:
        // parallel transport preserves Riemannian length, but the Poincaré-ball /
        // S³-embedding scales Euclidean length by a position-dependent factor. The
        // renderer expects Euclidean-unit directions in the embedding; the WGSL prelude
        // handles the metric.
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
        // Right-handed camera convention: forward is `-Z`, so
        // `right × up = +Z = -forward` (the "back" direction).
        close(cam.right.cross(cam.up), -cam.forward, 1e-6);
        // Pairwise orthogonal.
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
        // From (0,0,5) looking at origin, forward = -Z.
        close(cam.forward, -Vec3::Z, 1e-6);
        // World up Y should keep `up` ≈ Y.
        close(cam.up, Vec3::Y, 1e-6);
    }

    #[test]
    fn translate_in_flat_space_preserves_frame() {
        let space = EuclideanR3;
        let mut cam = Camera::<EuclideanR3>::at_origin();
        let original = cam.view();
        cam.translate(Vec3::new(1.0, 2.0, -3.0), 1.0, &space);
        // Position moved by the translation vector.
        close(cam.position, Vec3::new(1.0, 2.0, -3.0), 1e-6);
        // Frame is unchanged in flat space (parallel transport is
        // the identity in E³).
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

    /// `translate` in H³ must keep the position inside the Poincaré ball and the basis
    /// finite + ≈orthonormal after parallel transport. Catches NaN drift in `Space::exp`
    /// / `parallel_transport_along` for `HyperbolicH3`.
    #[test]
    fn translate_in_hyperbolic_h3_stays_in_ball_and_orthonormal() {
        use rye_math::HyperbolicH3;
        let space = HyperbolicH3;
        let mut cam = Camera::<HyperbolicH3>::at_origin();
        // Step in +X for half a second at unit Riemannian speed.
        cam.translate(Vec3::X, 0.5, &space);
        assert!(
            cam.position.length() < 1.0,
            "camera escaped Poincaré ball: {:?}",
            cam.position
        );
        assert!(cam.right.is_finite());
        assert!(cam.up.is_finite());
        assert!(cam.forward.is_finite());
        // Frame remains ≈orthonormal (transport + renormalise above).
        assert!((cam.right.length() - 1.0).abs() < 1e-3);
        assert!((cam.up.length() - 1.0).abs() < 1e-3);
        assert!((cam.forward.length() - 1.0).abs() < 1e-3);
        assert!(cam.right.dot(cam.up).abs() < 1e-3);
        assert!(cam.right.dot(cam.forward).abs() < 1e-3);
        assert!(cam.up.dot(cam.forward).abs() < 1e-3);
    }

    /// `looking_at` with `position == target` has no defined forward direction
    /// (`log = 0`). The fallback should produce a finite, orthonormal frame rather than
    /// NaN/inf.
    #[test]
    fn looking_at_collapsed_target_falls_back_to_finite_frame() {
        let cam = Camera::<EuclideanR3>::looking_at(Vec3::ZERO, Vec3::ZERO, Vec3::Y, &EuclideanR3);
        assert!(cam.forward.is_finite() && cam.right.is_finite() && cam.up.is_finite());
        // Frame is unit-length; the exact direction is unspecified
        // (fallback), only finiteness is guaranteed.
        assert!((cam.forward.length() - 1.0).abs() < 1e-6);
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
        assert!((cam.up.length() - 1.0).abs() < 1e-6);
    }

    /// `looking_at` where `forward` is parallel to `world_up` makes `forward × world_up
    /// = 0`. The fallback `right` keeps the frame finite even though it can't be derived
    /// from the cross product.
    #[test]
    fn looking_at_world_up_parallel_to_forward_falls_back() {
        let cam = Camera::<EuclideanR3>::looking_at(
            Vec3::new(0.0, 0.0, 0.0),
            // Look straight up; world_up is also +Y, so cross is zero.
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            &EuclideanR3,
        );
        assert!(cam.forward.is_finite() && cam.right.is_finite() && cam.up.is_finite());
        assert!((cam.right.length() - 1.0).abs() < 1e-6);
    }
}
