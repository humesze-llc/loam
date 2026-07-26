//! Cross-crate pin: `loam_camera::Camera::ray_from_ndc` must invert
//! `loam_egui::world_to_screen`. Both shipping functions are called here;
//! loam-camera cannot host this test because it sits below loam-egui in the
//! dependency DAG.

use glam::{Vec2, Vec3};
use loam_camera::Camera;
use loam_egui::world_to_screen;
use loam_math::EuclideanR3;

/// The screen convention `world_to_screen` is contracted to produce: NDC x
/// in [-1, 1] runs left to right, NDC y runs up while pixel y runs down.
fn expected_pixel(ndc: Vec2, viewport: Vec2) -> Vec2 {
    Vec2::new(
        (ndc.x * 0.5 + 0.5) * viewport.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.y,
    )
}

/// Unprojection inverts projection: every point along the ray through an NDC
/// coordinate projects back to that coordinate's pixel, at any depth. An
/// off-axis pose plus one landscape and one portrait viewport leave no room
/// for a swapped `right`/`up`, a transposed `aspect`, or a flipped y.
#[test]
fn ray_from_ndc_round_trips_through_world_to_screen() {
    for viewport in [(1600_u32, 900_u32), (720, 1280)] {
        let viewport_px = Vec2::new(viewport.0 as f32, viewport.1 as f32);
        let mut camera = Camera::<EuclideanR3>::looking_at(
            Vec3::new(2.0, 1.0, 4.0),
            Vec3::new(-1.0, 0.5, -2.0),
            Vec3::Y,
            &EuclideanR3,
        );
        camera.fov_y = 47.0_f32.to_radians();
        camera.aspect = viewport_px.x / viewport_px.y;

        // Sampling stops short of ±1: a ray through the exact frustum edge
        // round-trips to |ndc| = 1 plus float error, which `world_to_screen`
        // rejects as off-screen.
        for x_step in -3..=3 {
            for y_step in -3..=3 {
                let ndc = Vec2::new(x_step as f32 * 0.3, y_step as f32 * 0.3);
                let ray = camera.ray_from_ndc(ndc);
                assert!((ray.direction.length() - 1.0).abs() < 1e-6);
                let expected = expected_pixel(ndc, viewport_px);

                for depth in [0.5_f32, 3.0, 25.0] {
                    let world = ray.origin + ray.direction * depth;
                    let screen = world_to_screen(
                        world,
                        &camera.view(),
                        camera.fov_y,
                        viewport,
                        camera.near,
                        camera.far,
                    )
                    .expect("a point on an in-frustum ray must be visible");
                    let pixel = Vec2::new(screen.x, screen.y);
                    assert!(
                        (pixel - expected).length() < 1e-2,
                        "{viewport:?}: ndc {ndc:?} at depth {depth} projected to \
                         {pixel:?}, expected {expected:?}"
                    );
                }
            }
        }
    }
}

/// The round trip holds at every azimuth. One fixed pose can leave a frame
/// error invisible whenever the erroneous and correct bases happen to agree
/// on the sampled axis; orbiting through eight yaws puts every sign
/// combination of the world axes into `right` and `forward`.
#[test]
fn round_trip_holds_across_camera_orientations() {
    let viewport = (1280_u32, 720_u32);
    let viewport_px = Vec2::new(viewport.0 as f32, viewport.1 as f32);
    let ndc_samples = [
        Vec2::ZERO,
        Vec2::new(0.7, 0.4),
        Vec2::new(-0.7, 0.4),
        Vec2::new(0.7, -0.4),
        Vec2::new(-0.7, -0.4),
    ];

    for yaw_step in 0..8 {
        let yaw = yaw_step as f32 * std::f32::consts::FRAC_PI_4;
        let position = Vec3::new(3.0 * yaw.cos(), 1.25, 3.0 * yaw.sin());
        let mut camera =
            Camera::<EuclideanR3>::looking_at(position, Vec3::ZERO, Vec3::Y, &EuclideanR3);
        camera.fov_y = 70.0_f32.to_radians();
        camera.aspect = viewport_px.x / viewport_px.y;

        for ndc in ndc_samples {
            let ray = camera.ray_from_ndc(ndc);
            let world = ray.origin + ray.direction * 2.0;
            let screen = world_to_screen(
                world,
                &camera.view(),
                camera.fov_y,
                viewport,
                camera.near,
                camera.far,
            )
            .expect("a point on an in-frustum ray must be visible");
            let pixel = Vec2::new(screen.x, screen.y);
            let expected = expected_pixel(ndc, viewport_px);
            assert!(
                (pixel - expected).length() < 1e-2,
                "yaw {yaw}: ndc {ndc:?} projected to {pixel:?}, expected {expected:?}"
            );
        }
    }
}
