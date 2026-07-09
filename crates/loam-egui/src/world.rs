//! World-space to screen-space projection for anchoring egui widgets to 3D points.
//!
//! References:
//! - Akenine-Möller, Haines, Hoffman, *Real-Time Rendering* (4th ed,
//!   2018), §4.7 (view + projection transforms).

use glam::{Mat4, Vec3, Vec4Swizzles};
use loam_camera::CameraView;

/// Project a world-space point to screen pixel coordinates via the camera's view +
/// perspective projection.
///
/// Returns `None` if the point is behind the camera or outside the canonical view volume after
/// projection (clipped).
///
/// `viewport` is `(width_px, height_px)`. `fov_y_radians` matches the `Camera::fov_y` field;
/// `aspect` should be `width_px / height_px`. `near` and `far` should match the renderer's
/// depth setup; for screen-anchoring purposes `near = 0.05`, `far = 100.0` is fine since we
/// only care about NDC.
///
/// The returned [`egui::Pos2`] is in egui's pixel coordinates (top-left origin, y-down),
/// ready to pass to `egui::Area::fixed_pos`.
pub fn world_to_screen(
    world: Vec3,
    camera: &CameraView,
    fov_y_radians: f32,
    viewport: (u32, u32),
    near: f32,
    far: f32,
) -> Option<egui::Pos2> {
    let (vw, vh) = (viewport.0 as f32, viewport.1 as f32);
    if vw <= 0.0 || vh <= 0.0 {
        return None;
    }
    let aspect = vw / vh;

    // Build a right-handed view matrix from the camera's orthonormal frame: camera looks along
    // `forward`, with `right` and `up` as the screen axes. `Mat4::look_to_rh` is exactly this.
    let view = Mat4::look_to_rh(camera.position, camera.forward, camera.up);
    let proj = Mat4::perspective_rh(fov_y_radians, aspect, near, far);
    let clip = proj * view * world.extend(1.0);

    // Behind the camera (or on the near plane).
    if clip.w <= 0.0 {
        return None;
    }

    let ndc = clip.xyz() / clip.w;
    // Outside the canonical NDC cube; the point isn't visible.
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }

    // NDC has y-up; egui screen coords are y-down. Flip y here.
    let screen_x = (ndc.x * 0.5 + 0.5) * vw;
    let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * vh;
    Some(egui::Pos2::new(screen_x, screen_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera_at(position: Vec3, forward: Vec3, up: Vec3) -> CameraView {
        let right = forward.cross(up).normalize();
        CameraView {
            position,
            forward: forward.normalize(),
            right,
            up: up.normalize(),
        }
    }

    /// A point at the camera's gaze direction projects to the centre of the viewport.
    #[test]
    fn point_in_front_projects_to_centre() {
        let camera = camera_at(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let world = Vec3::new(0.0, 0.0, -5.0);
        let pos = world_to_screen(world, &camera, 60_f32.to_radians(), (800, 600), 0.05, 100.0)
            .expect("point in front of camera should project");
        let cx = 800.0 * 0.5;
        let cy = 600.0 * 0.5;
        assert!((pos.x - cx).abs() < 1e-3, "x off centre: {}", pos.x);
        assert!((pos.y - cy).abs() < 1e-3, "y off centre: {}", pos.y);
    }

    /// A point behind the camera returns None.
    #[test]
    fn point_behind_camera_returns_none() {
        let camera = camera_at(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let world = Vec3::new(0.0, 0.0, 5.0); // behind a -Z-looking camera
        assert!(
            world_to_screen(world, &camera, 60_f32.to_radians(), (800, 600), 0.05, 100.0).is_none()
        );
    }

    /// Off-axis world points map to the correct screen quadrant. A point above-and-right of
    /// the gaze direction lands in the upper-right of the screen (small x > centre, small y <
    /// centre because egui is y-down).
    #[test]
    fn upper_right_world_lands_upper_right_screen() {
        let camera = camera_at(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let world = Vec3::new(0.5, 0.5, -5.0);
        let pos = world_to_screen(world, &camera, 60_f32.to_radians(), (800, 600), 0.05, 100.0)
            .expect("visible point should project");
        assert!(pos.x > 400.0, "expected right half: x={}", pos.x);
        assert!(pos.y < 300.0, "expected upper half (y-down): y={}", pos.y);
    }

    /// A point far outside the frustum's lateral extent is clipped to None rather than
    /// producing nonsense pixel coordinates.
    #[test]
    fn point_outside_frustum_returns_none() {
        let camera = camera_at(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        // 100 units off-axis at near distance = far past the FOV cone.
        let world = Vec3::new(100.0, 0.0, -1.0);
        assert!(
            world_to_screen(world, &camera, 60_f32.to_radians(), (800, 600), 0.05, 100.0).is_none()
        );
    }

    /// Zero-area viewport returns None rather than panicking on a divide.
    #[test]
    fn zero_viewport_returns_none() {
        let camera = camera_at(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let world = Vec3::new(0.0, 0.0, -5.0);
        assert!(
            world_to_screen(world, &camera, 60_f32.to_radians(), (0, 600), 0.05, 100.0).is_none()
        );
        assert!(
            world_to_screen(world, &camera, 60_f32.to_radians(), (800, 0), 0.05, 100.0).is_none()
        );
    }
}
