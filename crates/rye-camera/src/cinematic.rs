use glam::{Mat4, Vec3};

/// A y-up orbit camera pose for scripted/cinematic shots: a pure function of its
/// parameters, so it can be sampled at any timeline `t` (deterministic playback)
/// and interpolated between key poses. Unlike the input-driven
/// [`crate::OrbitController`], nothing here is stateful.
#[derive(Clone, Copy, Debug)]
pub struct OrbitPose {
    pub target: Vec3,
    /// Around +y (up), radians.
    pub azimuth: f32,
    /// Above the horizon, radians.
    pub elevation: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl OrbitPose {
    pub fn eye(&self) -> Vec3 {
        let (se, ce) = self.elevation.sin_cos();
        let (sa, ca) = self.azimuth.sin_cos();
        self.target + Vec3::new(sa * ce, se, ca * ce) * self.distance
    }

    pub fn view_proj(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, near, far)
            * Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let f = |x: f32, y: f32| x + (y - x) * t;
        Self {
            target: a.target.lerp(b.target, t),
            azimuth: f(a.azimuth, b.azimuth),
            elevation: f(a.elevation, b.elevation),
            distance: f(a.distance, b.distance),
            fov_y: f(a.fov_y, b.fov_y),
        }
    }

    pub fn with_distance(mut self, distance: f32) -> Self {
        self.distance = distance;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_on_pose_sits_on_plus_z() {
        let pose = OrbitPose {
            target: Vec3::ZERO,
            azimuth: 0.0,
            elevation: 0.0,
            distance: 10.0,
            fov_y: 1.0,
        };
        let eye = pose.eye();
        assert!((eye - Vec3::new(0.0, 0.0, 10.0)).length() < 1e-5);
    }

    #[test]
    fn lerp_is_componentwise_and_endpoints_exact() {
        let a = OrbitPose {
            target: Vec3::ZERO,
            azimuth: 0.0,
            elevation: 0.0,
            distance: 10.0,
            fov_y: 0.5,
        };
        let b = OrbitPose {
            target: Vec3::new(1.0, 2.0, 0.0),
            azimuth: 1.0,
            elevation: 0.5,
            distance: 4.0,
            fov_y: 1.0,
        };
        let mid = OrbitPose::lerp(&a, &b, 0.5);
        assert!((mid.distance - 7.0).abs() < 1e-6);
        assert!((mid.azimuth - 0.5).abs() < 1e-6);
        let end = OrbitPose::lerp(&a, &b, 1.0);
        assert!((end.target - b.target).length() < 1e-6);
        assert!((end.fov_y - b.fov_y).abs() < 1e-6);
    }
}
