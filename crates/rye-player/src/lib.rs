//! Space-generic first-person player controller. Reads
//! [`rye_input::FrameInput`] and advances a position along the ambient
//! [`rye_math::Space`]'s geodesics, with an `f32` yaw for facing.
//!
//! WASD drives forward/back/strafe relative to yaw; Space/Shift drive
//! world-Y. The yaw tangent is integrated by `space.exp`, so curved Spaces
//! bend the path without extra work in the controller. Yaw is an `f32`
//! (not a rotor) since facing only rotates about Y.

use glam::Vec3;
use rye_input::FrameInput;
use rye_math::Space;

/// Space-generic player controller. [`PlayerState::advance`] moves along
/// geodesics from WASD; [`PlayerState::advance_look`] updates yaw from mouse.
///
/// `S` must map `Vec3 -> Vec3` (point and tangent both in R³ from the Space's
/// ambient embedding, e.g. Poincaré ball for H³).
pub struct PlayerState<S: Space<Point = Vec3, Vector = Vec3>> {
    pub position: Vec3,
    /// Camera/facing yaw in radians; 0 = −Z (into screen), positive = left.
    pub yaw: f32,
    _space: std::marker::PhantomData<fn() -> S>,
}

impl<S: Space<Point = Vec3, Vector = Vec3>> PlayerState<S> {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            _space: std::marker::PhantomData,
        }
    }

    pub fn with_yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// Move along a geodesic for one tick. `speed` is in Space-distance units;
    /// the tangent is the WASD axes rotated by `self.yaw`.
    pub fn advance(&mut self, input: &FrameInput, space: &S, speed: f32) {
        let sin_y = self.yaw.sin();
        let cos_y = self.yaw.cos();

        // Local basis from yaw only (pitch-independent movement plane).
        let fwd = Vec3::new(-sin_y, 0.0, -cos_y);
        let right = Vec3::new(cos_y, 0.0, -sin_y);
        let up = Vec3::Y;

        let tangent = fwd * input.move_forward + right * input.move_right + up * input.move_up;
        let len2 = tangent.length_squared();
        if len2 < 1e-8 {
            return;
        }
        // Normalize so diagonal movement isn't faster, then scale by speed.
        let t = tangent * (speed / len2.sqrt());
        self.position = space.exp(self.position, t);
    }

    /// Update yaw from mouse delta (mouse sensitivity in radians per pixel).
    pub fn advance_look(&mut self, input: &FrameInput, sensitivity: f32) {
        self.yaw -= input.mouse_delta.x * sensitivity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use rye_math::EuclideanR3;

    fn assert_close(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-4, "expected {a} close to {b}");
    }

    #[test]
    fn advance_forward_moves_in_minus_z() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        // yaw=0 -> forward = −Z
        player.advance(
            &FrameInput {
                move_forward: 1.0,
                ..FrameInput::default()
            },
            &EuclideanR3,
            1.0,
        );
        assert_close(player.position.x, 0.0);
        assert_close(player.position.y, 0.0);
        assert_close(player.position.z, -1.0);
    }

    #[test]
    fn advance_with_zero_input_does_not_move() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        player.advance(&FrameInput::default(), &EuclideanR3, 1.0);
        assert_eq!(player.position, Vec3::ZERO);
    }

    #[test]
    fn advance_look_updates_yaw() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        player.advance_look(
            &FrameInput {
                mouse_delta: Vec2::new(100.0, 0.0),
                ..FrameInput::default()
            },
            0.002,
        );
        assert_close(player.yaw, -0.2);
    }

    /// W+D travels `speed`, not `sqrt(2) * speed`: the diagonal tangent must
    /// be normalized so diagonal motion isn't faster than axis-aligned.
    #[test]
    fn advance_diagonal_input_speed_matches_axis_aligned_speed() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        player.advance(
            &FrameInput {
                move_forward: 1.0,
                move_right: 1.0,
                ..FrameInput::default()
            },
            &EuclideanR3,
            1.0,
        );
        assert_close(player.position.length(), 1.0);
    }

    #[test]
    fn advance_move_up_lifts_player_along_world_y() {
        let mut player: PlayerState<EuclideanR3> = PlayerState::new(Vec3::ZERO);
        player.advance(
            &FrameInput {
                move_up: 1.0,
                ..FrameInput::default()
            },
            &EuclideanR3,
            1.0,
        );
        assert_close(player.position.x, 0.0);
        assert_close(player.position.y, 1.0);
        assert_close(player.position.z, 0.0);
    }

    #[test]
    fn with_yaw_rotates_forward_direction() {
        // yaw = +π/2 rotates forward from −Z to −X.
        let mut player: PlayerState<EuclideanR3> =
            PlayerState::new(Vec3::ZERO).with_yaw(std::f32::consts::FRAC_PI_2);
        player.advance(
            &FrameInput {
                move_forward: 1.0,
                ..FrameInput::default()
            },
            &EuclideanR3,
            1.0,
        );
        assert_close(player.position.x, -1.0);
        assert_close(player.position.y, 0.0);
        assert_close(player.position.z, 0.0);
    }

    /// W in H³ from the origin must stay strictly inside the Poincaré ball
    /// (`|p| < 1`); the geodesic step is shorter than the Euclidean one.
    #[test]
    fn advance_in_hyperbolic_h3_stays_inside_ball() {
        use rye_math::HyperbolicH3;
        let mut player: PlayerState<HyperbolicH3> = PlayerState::new(Vec3::ZERO);
        player.advance(
            &FrameInput {
                move_forward: 1.0,
                ..FrameInput::default()
            },
            &HyperbolicH3,
            0.5,
        );
        assert!(player.position.is_finite());
        assert!(
            player.position.length() < 1.0,
            "player escaped Poincaré ball: {:?}",
            player.position
        );
        // Forward at yaw=0 is −Z, so motion is purely in −Z.
        assert_close(player.position.x, 0.0);
        assert_close(player.position.y, 0.0);
        assert!(player.position.z < 0.0);
    }

    /// Same shape in S³: position must stay inside the unit-3-sphere embedding (`|p| < 1`)
    /// and motion direction is consistent with the −Z forward convention.
    #[test]
    fn advance_in_spherical_s3_stays_inside_embedding() {
        use rye_math::SphericalS3;
        let mut player: PlayerState<SphericalS3> = PlayerState::new(Vec3::ZERO);
        player.advance(
            &FrameInput {
                move_forward: 1.0,
                ..FrameInput::default()
            },
            &SphericalS3,
            0.5,
        );
        assert!(player.position.is_finite());
        assert!(player.position.length() < 1.0);
        assert_close(player.position.x, 0.0);
        assert_close(player.position.y, 0.0);
        assert!(player.position.z < 0.0);
    }
}
