//! `loam-scene`: signed-distance field primitives and scene builders for Loam.
//!
//! [`Primitive`] is the typed abstraction for geometric objects; each emits a
//! WGSL `fn {name}(p: vec3<f32>) -> f32` using only `loam_*` Space-prelude
//! functions, so SDFs stay correct across E³, H³, and S³.
//!
//! [`combinator`] provides Space-agnostic combinators (union, intersection,
//! smooth-min) over the scalar distances returned by primitive SDFs.
//!
//! Every emitter has a CPU twin ([`Primitive::eval`], [`Primitive4::eval_4d`],
//! [`Scene::eval`], [`Scene4::eval_at`]) written as the same `match`, arm for
//! arm, so a new [`Shape`] variant fails to compile on both halves. The twin
//! delegates all curved geometry to [`loam_math::Space::distance`], the
//! reference implementation, rather than transliterating the WGSL prelude; for
//! Spaces whose prelude is a deliberate approximation the two halves are
//! different scalar fields and parity is a measured bound, not an identity.

pub mod combinator;
pub mod primitive;
pub mod primitive4;
pub mod scene;
pub mod scene4;

pub use loam_shape::Shape;
pub use primitive::Primitive;
pub use primitive4::Primitive4;
pub use scene::{PrimitiveKind, Scene, SceneNode};
pub use scene4::{
    Scene4, SceneNode4, PRIM_KIND_HALFSPACE4D, PRIM_KIND_HYPERSPHERE4D, PRIM_KIND_OTHER,
};

/// Distance returned by shapes with no closed-form SDF in the emitted dimension.
///
/// Large enough that the marcher's `t_scene > 40` bail fires before the surface
/// is ever reached, so an accidentally included shape renders as nothing rather
/// than as wrong geometry. Both halves read this constant: the emitters format
/// it into WGSL and the evaluators return it, so the sentinel-parity test can
/// compare against a name instead of a repeated literal.
pub const SENTINEL_DISTANCE: f32 = 1e9;

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

    // ---- Sentinel parity --------------------------------------------------

    /// One `Shape` per [`loam_shape::ShapeKind`], in declaration order. The
    /// sentinel tables sweep this, and `shape_table_covers_every_kind_exactly_once`
    /// keeps it honest when a variant is added.
    fn one_shape_per_kind() -> Vec<Shape> {
        use glam::{Vec2, Vec4};
        vec![
            Shape::sphere_at(Vec3::new(0.05, -0.02, 0.03), 0.25),
            Shape::HalfSpace {
                normal: Vec3::Y,
                offset: -0.5,
            },
            Shape::HalfSpace4D {
                normal: Vec4::Y,
                offset: -0.25,
            },
            Shape::Box3 {
                half_extents: Vec3::new(0.4, 0.3, 0.2),
            },
            Shape::Polygon2D {
                vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
            },
            Shape::ConvexPolytope3D {
                vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            },
            Shape::ConvexPolytope4D {
                vertices: vec![Vec4::ZERO, Vec4::X, Vec4::Y, Vec4::Z, Vec4::W],
            },
            Shape::HyperSphere4D {
                center: Vec4::new(0.1, 0.0, -0.1, 0.2),
                radius: 0.3,
            },
        ]
    }

    #[test]
    fn shape_table_covers_every_kind_exactly_once() {
        use loam_shape::ShapeKind;
        let kinds: Vec<ShapeKind> = one_shape_per_kind().iter().map(Shape::kind).collect();
        assert_eq!(
            kinds,
            vec![
                ShapeKind::Sphere,
                ShapeKind::HalfSpace,
                ShapeKind::HalfSpace4D,
                ShapeKind::Box3,
                ShapeKind::Polygon2D,
                ShapeKind::ConvexPolytope3D,
                ShapeKind::ConvexPolytope4D,
                ShapeKind::HyperSphere4D,
            ],
        );
    }

    /// The highest-consequence parity failure in the whole evaluator: returning
    /// a finite distance where the shader emits the sentinel would bake a
    /// collider for geometry the renderer never draws, and nothing downstream
    /// would catch it. Asserted as an iff over every `ShapeKind` in every
    /// shipped 3D Space; the emitted body is string-sniffed because "the
    /// emitter chose the sentinel arm" has no other observable.
    #[test]
    fn eval_sentinels_exactly_where_emit_sentinels() {
        use loam_math::{
            BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX, Space, SphericalS3, WgslSpace,
        };

        fn check<S: WgslSpace + Space<Point = Vec3, Vector = Vec3>>(space: &S, label: &str) {
            // In-chart for H³ (|p| < 1) and S³ (|p|² < 1) alike.
            let probe = Vec3::new(0.11, -0.07, 0.13);
            for shape in one_shape_per_kind() {
                let emitted_sentinel = shape
                    .to_wgsl(space, "sdf_probe")
                    .contains(&format!("return {SENTINEL_DISTANCE:e};"));
                let eval_sentinel = shape.eval(space, probe) == SENTINEL_DISTANCE;
                assert_eq!(
                    eval_sentinel,
                    emitted_sentinel,
                    "{label}/{:?}: eval sentinel = {eval_sentinel}, emit sentinel = \
                     {emitted_sentinel}",
                    shape.kind(),
                );
            }
        }

        check(&EuclideanR3, "EuclideanR3");
        check(&HyperbolicH3, "HyperbolicH3");
        check(&SphericalS3, "SphericalS3");
        check(
            &BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.5, 0.5)),
            "BlendedSpace<E3,H3>",
        );
    }

    /// Same iff for the 4D half. ℝ⁴ is the only 4D Space, so there is no Space
    /// parameter to sweep.
    #[test]
    fn eval_4d_sentinels_exactly_where_emit_4d_sentinels() {
        use glam::Vec4;
        let probe = Vec4::new(0.11, -0.07, 0.13, 0.05);
        for shape in one_shape_per_kind() {
            let emitted_sentinel = shape
                .to_wgsl_4d("sdf_probe")
                .contains(&format!("return {SENTINEL_DISTANCE:e};"));
            let eval_sentinel = shape.eval_4d(probe) == SENTINEL_DISTANCE;
            assert_eq!(
                eval_sentinel,
                emitted_sentinel,
                "{:?}: eval sentinel = {eval_sentinel}, emit sentinel = {emitted_sentinel}",
                shape.kind(),
            );
        }
    }

    /// The 3D `HalfSpace` arm is gated on flatness, so one shape must be finite
    /// in E³ and sentinel in H³. Pins that `eval` reads the gate at all, which
    /// the deleted shadow helper never did.
    #[test]
    fn halfspace_eval_follows_the_chart_flatness_gate() {
        use loam_math::{EuclideanR3, HyperbolicH3};
        let plane = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let p = Vec3::new(0.0, 0.25, 0.0);
        assert!((plane.eval(&EuclideanR3, p) - 0.75).abs() < 1e-6);
        assert_eq!(plane.eval(&HyperbolicH3, p), SENTINEL_DISTANCE);
    }

    // ---- Analytic invariants of the shipped evaluator ---------------------

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

    /// Deterministic xorshift32 single-point sampler.
    fn deterministic_samples(seed: u32, count: usize, extent: f32) -> Vec<Vec3> {
        let mut state = seed;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..count)
            .map(|_| {
                Vec3::new(
                    next_f32() * extent,
                    next_f32() * extent,
                    next_f32() * extent,
                )
            })
            .collect()
    }

    /// A signed-distance function is 1-Lipschitz with respect to the metric of
    /// the Space it lives in, not with respect to chart coordinates. In E³ the
    /// two coincide; in H³ and S³ the chart metric is strictly smaller than the
    /// Riemannian one, so the geodesic-distance leaves satisfy only this form.
    fn assert_lipschitz_1_under_space_metric<S, F>(label: &str, space: &S, sdf: F, extent: f32)
    where
        S: loam_math::Space<Point = Vec3, Vector = Vec3>,
        F: Fn(Vec3) -> f32,
    {
        for (a, b) in deterministic_pair_samples(0xABCD_1234, 256, extent) {
            let separation = space.distance(a, b);
            if separation < 1e-6 {
                continue;
            }
            let delta = (sdf(a) - sdf(b)).abs();
            assert!(
                delta <= separation * (1.0 + 1e-5),
                "{label}: |sdf({a:?}) - sdf({b:?})| = {delta} exceeds d(a, b) = {separation}",
            );
        }
    }

    /// Every combinator and every 3D primitive with a closed form, in one tree,
    /// checked against the Riemannian Lipschitz bound in all three shipped
    /// curvature regimes.
    #[test]
    fn scene_eval_is_lipschitz_1_under_the_space_metric() {
        use loam_math::{EuclideanR3, HyperbolicH3, SphericalS3};
        let scene = Scene::new(
            SceneNode::sphere(Vec3::new(0.1, 0.0, 0.0), 0.2)
                .smooth_union(SceneNode::cube(0.15), 0.06)
                .union(SceneNode::sphere(Vec3::new(-0.2, 0.1, 0.0), 0.12))
                .subtract(SceneNode::sphere(Vec3::new(0.0, 0.2, 0.0), 0.08))
                .intersect(SceneNode::plane(Vec3::Y, -0.6)),
        );
        // H³ and S³ charts saturate near their boundary shells, so sample well
        // inside; E³ has no boundary and gets the wider box.
        assert_lipschitz_1_under_space_metric(
            "E3",
            &EuclideanR3,
            |p| scene.eval(&EuclideanR3, p),
            1.0,
        );
        assert_lipschitz_1_under_space_metric(
            "H3",
            &HyperbolicH3,
            |p| scene.eval(&HyperbolicH3, p),
            0.3,
        );
        assert_lipschitz_1_under_space_metric(
            "S3",
            &SphericalS3,
            |p| scene.eval(&SphericalS3, p),
            0.3,
        );
    }

    /// Sphere leaves vanish exactly on the geodesic sphere of radius `r` in
    /// every Space, which is the point of routing them through `Space::distance`
    /// rather than a chart-coord length.
    #[test]
    fn sphere_eval_is_zero_on_the_geodesic_surface_in_every_space() {
        use loam_math::{EuclideanR3, HyperbolicH3, Space, SphericalS3};

        fn check<S: Space<Point = Vec3, Vector = Vec3>>(space: &S, label: &str) {
            let center = Vec3::new(0.05, -0.03, 0.02);
            let radius = 0.2_f32;
            let shape = Shape::sphere_at(center, radius);
            assert!(
                (shape.eval(space, center) + radius).abs() < 1e-6,
                "{label}: centre must read -radius",
            );
            for direction in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, Vec3::ONE.normalize()] {
                // `exp`'s tangent argument is in chart coordinates, so its
                // Riemannian length is the conformal factor times its chart
                // length. Geodesic arc length is linear in |v|, so one probe
                // step calibrates the scale that lands on the sphere of radius
                // `radius` in any Space.
                let probe = direction * 0.1;
                let probe_arc = space.distance(center, space.exp(center, probe));
                let at_arc = |arc: f32| space.exp(center, probe * (arc / probe_arc));

                assert!(
                    shape.eval(space, at_arc(radius)).abs() < 1e-5,
                    "{label}: the geodesic sphere of radius r must be the zero set",
                );
                assert!(
                    shape.eval(space, at_arc(radius * 2.0)) > 0.0,
                    "{label}: sign outside",
                );
                assert!(
                    shape.eval(space, at_arc(radius * 0.5)) < 0.0,
                    "{label}: sign inside",
                );
            }
        }

        check(&EuclideanR3, "E3");
        check(&HyperbolicH3, "H3");
        check(&SphericalS3, "S3");
    }

    /// Box and half-space leaves are chart-coord formulas; pin their zero set
    /// and sign in the flat chart where they are honest.
    #[test]
    fn box_and_halfspace_eval_zero_sets_and_signs_in_e3() {
        use loam_math::EuclideanR3;
        let half_extents = Vec3::splat(0.4);
        let box3 = Shape::Box3 { half_extents };
        assert!((box3.eval(&EuclideanR3, Vec3::ZERO) + 0.4).abs() < 1e-6);
        assert!(box3.eval(&EuclideanR3, Vec3::new(0.4, 0.0, 0.0)).abs() < 1e-6);
        assert!((box3.eval(&EuclideanR3, Vec3::new(1.4, 0.0, 0.0)) - 1.0).abs() < 1e-5);
        // Outside the corner the exact box SDF is the Euclidean distance to it.
        let corner_offset = Vec3::splat(1.0);
        assert!(
            (box3.eval(&EuclideanR3, half_extents + corner_offset) - corner_offset.length()).abs()
                < 1e-5
        );

        let plane = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        assert!(plane.eval(&EuclideanR3, Vec3::new(0.0, -0.5, 0.0)).abs() < 1e-6);
        assert!((plane.eval(&EuclideanR3, Vec3::Y) - 1.5).abs() < 1e-5);
        assert!(plane.eval(&EuclideanR3, Vec3::new(0.0, -1.0, 0.0)) < 0.0);
    }

    // ---- Combinator algebra ----------------------------------------------

    /// Union commutes and difference does not: `max(l, -r)` picks a side, so
    /// swapping the operands must change the field where the shapes overlap.
    /// Catches a swapped-operand transcription of the `Difference` arm.
    #[test]
    fn union_commutes_and_difference_does_not() {
        use loam_math::EuclideanR3;
        let left = SceneNode::sphere(Vec3::new(-0.1, 0.0, 0.0), 0.3);
        let right = SceneNode::sphere(Vec3::new(0.1, 0.0, 0.0), 0.3);
        let union_lr = Scene::new(left.clone().union(right.clone()));
        let union_rl = Scene::new(right.clone().union(left.clone()));
        let diff_lr = Scene::new(left.clone().subtract(right.clone()));
        let diff_rl = Scene::new(right.subtract(left));

        let mut asymmetric = 0usize;
        for p in deterministic_samples(0x0BAD_F00D, 256, 0.6) {
            assert_eq!(
                union_lr.eval(&EuclideanR3, p),
                union_rl.eval(&EuclideanR3, p)
            );
            if diff_lr.eval(&EuclideanR3, p) != diff_rl.eval(&EuclideanR3, p) {
                asymmetric += 1;
            }
        }
        assert!(
            asymmetric > 0,
            "A minus B and B minus A must differ somewhere in the sampled volume",
        );
    }

    /// Quilez's polynomial smooth-min is a lower bound on `min` and converges to
    /// it as the blend radius vanishes, with worst-case gap exactly `k/4` at
    /// `a == b`. Pins the transcription of the polynomial rather than merely the
    /// presence of the word `clamp` in the emitted text.
    #[test]
    fn smooth_union_underestimates_min_and_converges_to_it() {
        use loam_math::EuclideanR3;
        let left = SceneNode::sphere(Vec3::new(-0.2, 0.0, 0.0), 0.25);
        let right = SceneNode::sphere(Vec3::new(0.2, 0.0, 0.0), 0.25);
        let hard = Scene::new(left.clone().union(right.clone()));
        let samples = deterministic_samples(0x51DF_00D5, 512, 0.7);

        for k in [0.25_f32, 0.05, 1e-3] {
            let soft = Scene::new(left.clone().smooth_union(right.clone(), k));
            for &p in &samples {
                let smooth = soft.eval(&EuclideanR3, p);
                let sharp = hard.eval(&EuclideanR3, p);
                assert!(
                    smooth <= sharp + 1e-6,
                    "k={k}: smooth_union {smooth} must not exceed min {sharp} at {p:?}",
                );
                assert!(
                    sharp - smooth <= k * 0.25 + 1e-6,
                    "k={k}: gap {} exceeds the k/4 worst case at {p:?}",
                    sharp - smooth,
                );
            }
        }
    }

    /// Far outside the blend band `|a - b| < k` the polynomial saturates and the
    /// smooth union must equal `min` bit-for-bit, not merely approximately.
    #[test]
    fn smooth_union_matches_min_outside_the_blend_band() {
        use loam_math::EuclideanR3;
        let left = SceneNode::sphere(Vec3::new(-0.5, 0.0, 0.0), 0.2);
        let right = SceneNode::sphere(Vec3::new(0.5, 0.0, 0.0), 0.2);
        let soft = Scene::new(left.clone().smooth_union(right.clone(), 0.02));
        let hard = Scene::new(left.union(right));
        // At the left ball's centre the right leaf is ~1.0 away, far outside the
        // band, so `h` clamps to 1 and the `k·h·(1 − h)` term is exactly zero.
        let p = Vec3::new(-0.5, 0.0, 0.0);
        assert_eq!(soft.eval(&EuclideanR3, p), hard.eval(&EuclideanR3, p));
    }

    // ---- 4D evaluator -----------------------------------------------------

    /// The hyperslice `dist` is the 4D field restricted to `w = w_slice`, and
    /// `kind` follows the emitted `select`: closer leaf under union, farther
    /// under intersection, sentinel under difference.
    #[test]
    fn hyperslice_eval_tracks_distance_and_kind_through_combinators() {
        use glam::Vec4;
        use scene4::{PRIM_KIND_HALFSPACE4D, PRIM_KIND_HYPERSPHERE4D, PRIM_KIND_OTHER};

        let ball = SceneNode4::hypersphere(Vec4::ZERO, 0.5);
        let floor = SceneNode4::halfspace(Vec4::Y, -0.4);

        let union = Scene4::new(ball.clone().union(floor.clone()));
        // Above the ball's north pole and far from the floor plane.
        let (dist, kind) = union.eval_at(Vec3::new(0.0, 0.6, 0.0), 0.0, true);
        assert!((dist - 0.1).abs() < 1e-6);
        assert_eq!(kind, PRIM_KIND_HYPERSPHERE4D);
        // Beside the ball and just above the floor: the floor is closer.
        let (_, kind) = union.eval_at(Vec3::new(2.0, -0.3, 0.0), 0.0, true);
        assert_eq!(kind, PRIM_KIND_HALFSPACE4D);

        // Intersection reports the farther (active boundary) leaf.
        let intersection = Scene4::new(ball.clone().intersect(floor.clone()));
        let (dist, kind) = intersection.eval_at(Vec3::new(0.0, 0.6, 0.0), 0.0, true);
        assert!((dist - 1.0).abs() < 1e-6);
        assert_eq!(kind, PRIM_KIND_HALFSPACE4D);

        let difference = Scene4::new(ball.subtract(floor));
        let (_, kind) = difference.eval_at(Vec3::ZERO, 0.0, true);
        assert_eq!(kind, PRIM_KIND_OTHER);
    }

    /// The slice coordinate is a real degree of freedom: a hypersphere centred
    /// at `w = 0` presents radius `sqrt(r² − w²)` in the slice and vanishes past
    /// its pole.
    #[test]
    fn hyperslice_radius_shrinks_with_the_slice_coordinate() {
        use glam::Vec4;
        let radius = 0.5_f32;
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, radius));
        for w in [0.0_f32, 0.2, 0.4] {
            let sliced_radius = (radius * radius - w * w).sqrt();
            assert!(
                scene
                    .eval(Vec3::new(sliced_radius, 0.0, 0.0), w, true)
                    .abs()
                    < 1e-6,
                "w={w}: sliced surface point must read zero",
            );
        }
        assert!(scene.eval(Vec3::ZERO, 0.6, true) > 0.0);
    }

    /// A gated-off halfspace reads exactly the sentinel, matching the emitted
    /// `select`; every other leaf is untouched by the gate.
    #[test]
    fn hyperslice_gate_off_returns_the_sentinel_for_halfspaces_only() {
        use glam::Vec4;
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).union(SceneNode4::halfspace(Vec4::Y, -0.4)),
        );
        let just_above_floor = Vec3::new(3.0, -0.39, 0.0);
        assert!(scene.eval(just_above_floor, 0.0, true) < 0.02);
        let ball_only = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.5));
        assert_eq!(
            scene.eval(just_above_floor, 0.0, false),
            ball_only.eval(just_above_floor, 0.0, true),
        );
    }

    /// Both 4D leaves with a closed form are Lipschitz-1 in flat ℝ⁴, where the
    /// chart metric is the Riemannian one.
    #[test]
    fn scene4_eval_is_lipschitz_1_in_flat_r4() {
        use glam::Vec4;
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(0.1, 0.0, -0.1, 0.05), 0.4)
                .union(SceneNode4::halfspace(Vec4::Y, -0.5))
                .subtract(SceneNode4::hypersphere(Vec4::new(0.3, 0.0, 0.0, 0.0), 0.15)),
        );
        let mut state: u32 = 0x5555_3333;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for _ in 0..256 {
            let a = Vec4::new(next_f32(), next_f32(), next_f32(), next_f32()) * 2.0;
            let b = Vec4::new(next_f32(), next_f32(), next_f32(), next_f32()) * 2.0;
            let separation = (a - b).length();
            if separation < 1e-6 {
                continue;
            }
            let delta =
                (scene.eval(a.truncate(), a.w, true) - scene.eval(b.truncate(), b.w, true)).abs();
            assert!(
                delta <= separation * (1.0 + 1e-5),
                "|sdf({a:?}) - sdf({b:?})| = {delta} exceeds |a - b| = {separation}",
            );
        }
    }

    // ---- Determinism ------------------------------------------------------

    /// `Scene::eval` sits inside the Tier-0 boundary the moment a baked collider
    /// feeds the sim, so the field is pinned bit-exactly over a fixed lattice.
    /// This is the test that fails when a combinator is reassociated or FMA
    /// contraction is enabled. `EuclideanR3` only: its `distance` is the square
    /// root of a dot product, IEEE-exact and therefore portable, whereas H³ and
    /// S³ route through `artanh` / `asin`, whose last bit is a libm decision.
    #[test]
    fn scene_eval_bit_pattern_is_pinned_over_a_fixed_lattice() {
        use loam_math::EuclideanR3;
        let scene = Scene::new(
            SceneNode::sphere(Vec3::new(0.1, -0.05, 0.2), 0.3)
                .smooth_union(SceneNode::box_(Vec3::new(0.4, 0.2, 0.3)), 0.07)
                .union(SceneNode::plane(Vec3::Y, -0.5))
                .subtract(SceneNode::sphere(Vec3::new(-0.2, 0.1, 0.0), 0.15)),
        );

        // FNV-1a 64 (Fowler / Noll / Vo, 1991); chosen for being three lines
        // with no dependency, not for any statistical property.
        const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        const STEPS: i32 = 12;
        let mut hash = FNV_OFFSET_BASIS;
        for ix in 0..STEPS {
            for iy in 0..STEPS {
                for iz in 0..STEPS {
                    let p = Vec3::new(
                        ix as f32 / STEPS as f32 - 0.5,
                        iy as f32 / STEPS as f32 - 0.5,
                        iz as f32 / STEPS as f32 - 0.5,
                    );
                    for byte in scene.eval(&EuclideanR3, p).to_bits().to_le_bytes() {
                        hash ^= byte as u64;
                        hash = hash.wrapping_mul(FNV_PRIME);
                    }
                }
            }
        }
        assert_eq!(hash, 0x052d_c4d2_ffeb_2586, "golden hash");
    }
}
