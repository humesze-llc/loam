//! `loam-scene`: signed-distance field primitives and scene builders for Loam.
//!
//! [`Primitive`] is the typed abstraction for geometric objects; each emits a
//! WGSL `fn {name}(p: vec3<f32>) -> f32` using only `loam_*` Space-prelude
//! functions, so SDFs stay correct across E³, H³, and S³.
//!
//! [`combinator`] provides Space-agnostic combinators (union, intersection,
//! smooth-min) over the scalar distances returned by primitive SDFs.

pub mod combinator;
pub mod primitive;
pub mod primitive4;
pub mod scene;
pub mod scene4;

pub use loam_shape::Shape;
pub use primitive::Primitive;
pub use primitive4::Primitive4;
pub use scene::{PrimitiveKind, Scene, SceneNode};
pub use scene4::{Scene4, SceneNode4};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // ---- Primitive trait tests -------------------------------------------

    #[test]
    fn sphere_emits_loam_distance_call() {
        use loam_math::EuclideanR3;
        let s = Shape::sphere_at(Vec3::ZERO, 0.25);
        let src = s.to_wgsl(&EuclideanR3, "sdf_0");
        assert!(src.contains("fn sdf_0(p: vec3<f32>) -> f32"));
        assert!(src.contains("loam_distance"));
        assert!(src.contains("0.250000"));
    }

    #[test]
    fn sphere_wgsl_is_space_agnostic() {
        use loam_math::{EuclideanR3, HyperbolicH3, SphericalS3};
        let s = Shape::sphere_at_origin(0.3);
        let e3 = s.to_wgsl(&EuclideanR3, "sdf_0");
        let h3 = s.to_wgsl(&HyperbolicH3, "sdf_0");
        let s3 = s.to_wgsl(&SphericalS3, "sdf_0");
        // The emitted body must be identical across spaces; only loam_distance differs at
        // prelude link time, not in the emitted text.
        assert_eq!(e3, h3);
        assert_eq!(h3, s3);
    }

    /// `HalfSpace` emits the honest chart-coord `dot(p, n) - d` in flat E³.
    #[test]
    fn halfspace_emits_dot_in_flat_chart() {
        use loam_math::EuclideanR3;
        let p = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let src = p.to_wgsl(&EuclideanR3, "sdf_floor");
        assert!(src.contains("fn sdf_floor(p: vec3<f32>) -> f32"));
        assert!(src.contains("dot(p,"));
        assert!(src.contains("-0.500000"));
    }

    /// `HalfSpace` sentinels in a curved Space; pinned so a regression that
    /// re-enables raw chart-coord `dot()` there fails loud.
    #[test]
    fn halfspace_sentinels_in_curved_chart() {
        use loam_math::HyperbolicH3;
        let p = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let src = p.to_wgsl(&HyperbolicH3, "sdf_floor");
        assert!(src.contains("fn sdf_floor(_p: vec3<f32>) -> f32"));
        assert!(src.contains("return 1e9"));
        assert!(
            !src.contains("dot(p,"),
            "HalfSpace must not emit raw chart-coord dot product in curved Spaces",
        );
    }

    #[test]
    fn box_emits_euclidean_box_sdf() {
        use loam_math::EuclideanR3;
        let b = Shape::Box3 {
            half_extents: Vec3::splat(0.4),
        };
        let src = b.to_wgsl(&EuclideanR3, "sdf_box");
        assert!(src.contains("fn sdf_box(p: vec3<f32>) -> f32"));
        assert!(src.contains("abs(p)"));
        assert!(src.contains("0.400000"));
    }

    #[test]
    fn combinator_union_expr() {
        use combinator::union_expr;
        let expr = union_expr("da", "db");
        assert_eq!(expr, "min(da, db)");
    }

    #[test]
    fn combinator_smooth_min_fn_compiles() {
        use combinator::smooth_min_fn;
        let src = smooth_min_fn("smin", 0.08);
        assert!(src.contains("fn smin(a: f32, b: f32) -> f32"));
        assert!(src.contains("0.080000"));
        assert!(src.contains("clamp"));
        assert!(src.contains("mix"));
    }

    // ---- Scene-tree integration tests ------------------------------------

    /// Sphere + plane in E³ emits both `loam_distance` and `dot(p,` inside one
    /// `loam_scene_sdf`.
    #[test]
    fn scene_with_sphere_and_plane_emits_both_paths_in_e3() {
        use loam_math::EuclideanR3;
        let scene =
            Scene::new(SceneNode::sphere(Vec3::ZERO, 0.22).union(SceneNode::plane(Vec3::Y, -0.5)));
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("fn loam_scene_sdf"));
        assert!(src.contains("loam_distance"));
        assert!(src.contains("dot(p,"));
        assert!(src.contains("-0.500000"));
    }

    /// A sphere-only scene emits no `dot()` calls.
    #[test]
    fn sphere_only_scene_emits_no_chart_coord_dot() {
        use loam_math::EuclideanR3;
        let scene = Scene::new(SceneNode::sphere(Vec3::ZERO, 0.3));
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("fn loam_scene_sdf"));
        assert!(src.contains("loam_distance"));
        assert!(!src.contains("dot(p,"));
    }

    /// Baked sphere centres match the input point literally.
    #[test]
    fn sphere_center_appears_as_wgsl_literal() {
        use loam_math::EuclideanR3;
        let scene = Scene::new(SceneNode::sphere(Vec3::new(0.5, 0.0, 0.0), 0.1));
        let src = scene.to_wgsl(&EuclideanR3);
        assert!(src.contains("0.500000, 0.000000, 0.000000"));
    }

    /// A tangent vector exped through H³ compresses below its E³ coordinate, so
    /// the H³ scene does not emit the E³-style literal.
    #[test]
    fn lattice_centres_compress_under_hyperbolic_exp() {
        use loam_math::{HyperbolicH3, Space};
        let p = HyperbolicH3.exp(Vec3::ZERO, Vec3::X * 0.5);
        let scene = Scene::new(SceneNode::sphere(p, 0.1));
        let src = scene.to_wgsl(&HyperbolicH3);
        // tanh(0.25) ≈ 0.2449, well under 0.5.
        assert!(p.x < 0.5);
        assert!(!src.contains("0.500000, 0.000000, 0.000000"));
    }

    // ---- Semantic-SDF correctness + Lipschitz-bound tests ------------------
    // These verify the math each primitive represents: sign, surface zero,
    // Lipschitz-1; the string-emit tests above cover the WGSL text.

    fn sphere_sdf_cpu(p: Vec3, center: Vec3, radius: f32) -> f32 {
        (p - center).length() - radius
    }

    fn box3_sdf_cpu(p: Vec3, half_extents: Vec3) -> f32 {
        let q = p.abs() - half_extents;
        q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
    }

    fn halfspace_sdf_cpu(p: Vec3, normal: Vec3, offset: f32) -> f32 {
        p.dot(normal) - offset
    }

    fn hypersphere_sdf_cpu(p: glam::Vec4, center: glam::Vec4, radius: f32) -> f32 {
        (p - center).length() - radius
    }

    fn halfspace4d_sdf_cpu(p: glam::Vec4, normal: glam::Vec4, offset: f32) -> f32 {
        p.dot(normal) - offset
    }

    /// Deterministic xorshift32 point-pair sampler for Lipschitz checks.
    fn deterministic_pair_samples(seed: u32, count: usize, extent: f32) -> Vec<(Vec3, Vec3)> {
        let mut state = seed;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0 // [-1, 1]
        };
        (0..count)
            .map(|_| {
                let a = Vec3::new(
                    next_f32() * extent,
                    next_f32() * extent,
                    next_f32() * extent,
                );
                let b = Vec3::new(
                    next_f32() * extent,
                    next_f32() * extent,
                    next_f32() * extent,
                );
                (a, b)
            })
            .collect()
    }

    /// Assert `|sdf(a) - sdf(b)| <= |a - b| * (1 + 1e-5)` across all pairs.
    fn assert_lipschitz_1<F: Fn(Vec3) -> f32>(label: &str, sdf: F, samples: &[(Vec3, Vec3)]) {
        for &(a, b) in samples {
            let dist_ab = (a - b).length();
            if dist_ab < 1e-6 {
                continue;
            }
            let lhs = (sdf(a) - sdf(b)).abs();
            let rhs = dist_ab * (1.0 + 1e-5);
            assert!(
                lhs <= rhs,
                "{label}: |sdf({a:?}) - sdf({b:?})| = {lhs} exceeds |a-b| = {dist_ab} \
                 (Lipschitz-1 violated)"
            );
        }
    }

    /// Sphere SDF: sign + surface zero + centre = -radius.
    #[test]
    fn sphere_sdf_distance_and_signs() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let radius = 0.5_f32;
        assert!((sphere_sdf_cpu(center, center, radius) + radius).abs() < 1e-6);
        let on_surface = center + Vec3::X * radius;
        assert!(sphere_sdf_cpu(on_surface, center, radius).abs() < 1e-6);
        let far = center + Vec3::X * 10.0;
        assert!((sphere_sdf_cpu(far, center, radius) - (10.0 - radius)).abs() < 1e-5);
        assert!(sphere_sdf_cpu(center + Vec3::X * 0.1, center, radius) < 0.0);
        assert!(sphere_sdf_cpu(center + Vec3::X * 1.0, center, radius) > 0.0);
    }

    #[test]
    fn sphere_sdf_is_lipschitz_1() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let radius = 0.5_f32;
        let samples = deterministic_pair_samples(0xABCD_1234, 256, 5.0);
        assert_lipschitz_1("sphere", |p| sphere_sdf_cpu(p, center, radius), &samples);
    }

    /// Box SDF: sign + surface zero + corner Euclidean distance.
    #[test]
    fn box3_sdf_distance_and_signs() {
        let h = Vec3::splat(0.4);
        assert!((box3_sdf_cpu(Vec3::ZERO, h) + 0.4).abs() < 1e-6);
        assert!(box3_sdf_cpu(Vec3::new(0.4, 0.0, 0.0), h).abs() < 1e-6);
        assert!((box3_sdf_cpu(Vec3::new(1.4, 0.0, 0.0), h) - 1.0).abs() < 1e-5);
        let corner = Vec3::splat(0.4);
        let outside_corner = corner + Vec3::splat(1.0);
        let expected = (outside_corner - corner).length();
        assert!((box3_sdf_cpu(outside_corner, h) - expected).abs() < 1e-5);
    }

    #[test]
    fn box3_sdf_is_lipschitz_1() {
        let h = Vec3::splat(0.4);
        let samples = deterministic_pair_samples(0xBEEF_5678, 256, 5.0);
        assert_lipschitz_1("box3", |p| box3_sdf_cpu(p, h), &samples);
    }

    /// HalfSpace SDF: sign + plane zero (gradient = unit normal).
    #[test]
    fn halfspace_sdf_distance_and_signs() {
        let normal = Vec3::Y;
        let offset = -0.5_f32;
        assert!(halfspace_sdf_cpu(Vec3::new(0.0, -0.5, 0.0), normal, offset).abs() < 1e-6);
        assert!((halfspace_sdf_cpu(Vec3::new(0.0, 1.0, 0.0), normal, offset) - 1.5).abs() < 1e-5);
        assert!(halfspace_sdf_cpu(Vec3::new(0.0, -1.0, 0.0), normal, offset) < 0.0);
    }

    #[test]
    fn halfspace_sdf_is_lipschitz_1() {
        let normal = Vec3::new(0.6, 0.8, 0.0); // unit
        let offset = 0.0_f32;
        let samples = deterministic_pair_samples(0xCAFE_F00D, 256, 5.0);
        assert_lipschitz_1(
            "halfspace",
            |p| halfspace_sdf_cpu(p, normal, offset),
            &samples,
        );
    }

    /// `min(a, b)` is Lipschitz-1 when both `a` and `b` are.
    #[test]
    fn union_min_preserves_lipschitz_1() {
        let center_a = Vec3::new(-1.0, 0.0, 0.0);
        let center_b = Vec3::new(1.0, 0.0, 0.0);
        let radius = 0.5_f32;
        let union_sdf =
            |p: Vec3| sphere_sdf_cpu(p, center_a, radius).min(sphere_sdf_cpu(p, center_b, radius));
        let samples = deterministic_pair_samples(0x1357_9BDF, 256, 4.0);
        assert_lipschitz_1("union(sphere_a, sphere_b)", union_sdf, &samples);
    }

    /// HyperSphere4D: sign + surface zero + Lipschitz-1 spot check.
    #[test]
    fn hypersphere4d_sdf_distance_and_lipschitz() {
        use glam::Vec4;
        let center = Vec4::new(0.5, 1.0, -0.5, 0.25);
        let radius = 0.7_f32;
        let on_surface = center + Vec4::X * radius;
        assert!(hypersphere_sdf_cpu(on_surface, center, radius).abs() < 1e-5);
        let far = center + Vec4::W * 5.0;
        assert!((hypersphere_sdf_cpu(far, center, radius) - (5.0 - radius)).abs() < 1e-5);
        let mut state: u32 = 0x9999_AAAA;
        let mut nf32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for _ in 0..256 {
            let a = Vec4::new(nf32() * 5.0, nf32() * 5.0, nf32() * 5.0, nf32() * 5.0);
            let b = Vec4::new(nf32() * 5.0, nf32() * 5.0, nf32() * 5.0, nf32() * 5.0);
            let dist_ab = (a - b).length();
            if dist_ab < 1e-6 {
                continue;
            }
            let lhs = (hypersphere_sdf_cpu(a, center, radius)
                - hypersphere_sdf_cpu(b, center, radius))
            .abs();
            assert!(lhs <= dist_ab * (1.0 + 1e-5));
        }
    }

    #[test]
    fn halfspace4d_sdf_distance_and_lipschitz() {
        use glam::Vec4;
        let normal = Vec4::Y;
        let offset = 0.0_f32;
        assert!(halfspace4d_sdf_cpu(Vec4::new(1.0, 0.0, 2.0, 3.0), normal, offset).abs() < 1e-6);
        assert!(halfspace4d_sdf_cpu(Vec4::new(0.0, 2.0, 0.0, 0.0), normal, offset) > 0.0);
        let mut state: u32 = 0x5555_3333;
        let mut nf32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for _ in 0..256 {
            let a = Vec4::new(nf32() * 5.0, nf32() * 5.0, nf32() * 5.0, nf32() * 5.0);
            let b = Vec4::new(nf32() * 5.0, nf32() * 5.0, nf32() * 5.0, nf32() * 5.0);
            let dist_ab = (a - b).length();
            if dist_ab < 1e-6 {
                continue;
            }
            let lhs = (halfspace4d_sdf_cpu(a, normal, offset)
                - halfspace4d_sdf_cpu(b, normal, offset))
            .abs();
            assert!(lhs <= dist_ab * (1.0 + 1e-5));
        }
    }
}
