//! One invariant harness over every [`Space`] impl in the crate.
//!
//! A [`SpaceFixture`] supplies a Space's domain samples, its metric, its
//! isometries and its tolerances; [`conformance_suite!`] expands to one named
//! `#[test]` per invariant per fixture, so a failure names the property it
//! broke rather than the harness.
//!
//! The single hook that separates flat from curved is the metric inner product
//! [`SpaceFixture::inner`]. Every length in this file is measured with it, so
//! "preserves length" means the lambda-weighted length in the Poincaré ball,
//! the lifted 4D length in the S³ hemisphere chart, and the plain dot product
//! in the flat impls, without the assertions branching on which is which.
//! Vector residuals are always `|u - v|_metric`, never componentwise, and point
//! residuals are always `distance`, because a geodesic residual is a physical
//! quantity and a chart residual is not.
//!
//! Group laws are stated as actions on points: `Space::Iso` carries no
//! `PartialEq` bound and `Rotor4` double-covers SO(4), so comparing composed
//! `Iso` values would be both uncompilable in the generic and wrong if it
//! compiled.
//!
//! An integration test sees only `pub` items, so the suite doubles as a
//! public-contract pin: every fixture is written against the same surface an
//! out-of-crate implementor has.

use glam::{Quat, Vec2, Vec3, Vec4};
use loam_math::{
    Bivector, Bivector2, Bivector4, BlendedSpace, ConformallyFlat, EuclideanR2, EuclideanR3,
    EuclideanR4, HyperbolicH3, Iso2, Iso3, Iso3H, Iso4, Iso4Flat, LinearBlendX, Space, SphericalS3,
    SphericalS3Embedded,
};

// ---------------------------------------------------------------------------
// The fixture surface
// ---------------------------------------------------------------------------

/// Per-impl error budget, keyed by what is being measured rather than by which
/// invariant does the measuring: three numbers stay honest, eleven would be
/// eleven places to launder a failure.
///
/// A tolerance is meaningless without the sampling extent it was derived at;
/// each fixture states its extent next to its numbers, and a fixture that
/// widens its domain must re-derive rather than inherit.
#[derive(Clone, Copy)]
struct Tol {
    /// Geodesic distance between two points that should coincide.
    point: f32,
    /// Metric norm of a tangent residual that should vanish.
    vector: f32,
    /// Difference of two scalars that should agree (a distance or a norm).
    scalar: f32,
    /// Same as `scalar`, but for inputs the fixture declares degenerate: the
    /// cut locus and the saturation shells, where the conditioning class is
    /// different and pretending otherwise would mean loosening `scalar` for
    /// everyone.
    degenerate: f32,
}

/// Everything the harness needs from a `Space`, and nothing else.
///
/// `combine` exists because `Space::Vector` is bounded only by
/// `Copy + Send + Sync + 'static`: the trait gives the harness no way to add or
/// scale a tangent, and the residual, linearity and triangle-construction items
/// all need one. `point_components` exists for the same reason one dimension
/// down: NaN and bit-identity are properties of the coordinates, and `Point`
/// exposes none.
trait SpaceFixture {
    type Point: Copy;
    type Vector: Copy;
    type Iso: Copy;
    type S: Space<Point = Self::Point, Vector = Self::Vector, Iso = Self::Iso>;

    /// Declared by the one fixture whose Space has no non-trivial isometry
    /// group. Not defaulted silently: a vacuously-passing invariant is the
    /// failure class this suite exists to remove, so the exemption is written
    /// down and the group-law tests assert the triviality instead of assuming
    /// it.
    const ISOMETRY_GROUP_IS_TRIVIAL: bool = false;

    /// Whether `parallel_transport` follows the geodesic from `a` to `b`.
    /// `Space` leaves the path implementation-defined, so a fixture whose impl
    /// walks some other path declares `false`, and both items it gates assert
    /// the exemption is still earned instead of skipping.
    const TRANSPORT_FOLLOWS_THE_GEODESIC: bool = true;

    fn space(&self) -> Self::S;

    /// In-domain samples. The fixture owns the count and the seed; the harness
    /// has no sample-budget knob, because the impls differ by four orders of
    /// magnitude in cost per `log` and a shared budget would be sized for the
    /// slowest.
    fn points(&self) -> Vec<Self::Point>;

    /// Tangents at `at`, short enough that `exp` stays inside the injectivity
    /// radius so the reverse round trip is well posed.
    fn tangents(&self, at: Self::Point) -> Vec<Self::Vector>;

    /// At least three isometries, pairwise non-commuting, chosen so the sampled
    /// points stay in the chart under every one of them.
    fn isos(&self) -> Vec<Self::Iso>;

    /// Riemannian inner product at `at`. The one hook that separates flat from
    /// curved; flat fixtures return the dot product and are thereby not a
    /// special case, just the one where the hook is constant.
    fn inner(&self, at: Self::Point, u: Self::Vector, v: Self::Vector) -> f32;

    /// `s*u + t*v` in the tangent space. Every impl's `Vector` is a linear
    /// space; a future Space whose `Vector` is not would break this harness
    /// rather than fail it.
    fn combine(&self, u: Self::Vector, s: f32, v: Self::Vector, t: f32) -> Self::Vector;

    /// Pairs the fixture declares out of the well-conditioned regime: the cut
    /// locus, the chart boundary, the saturation shells. This is where
    /// per-geometry knowledge lives; the harness only asserts that nothing
    /// here returns NaN, infinity, or a norm the transport failed to preserve.
    fn degenerate_pairs(&self) -> Vec<(Self::Point, Self::Point)>;

    /// Sectional curvature, when it is a constant. `None` means the curvature
    /// varies with position and Gauss-Bonnet with a single `K` does not apply.
    fn curvature(&self) -> Option<f32>;

    fn tol(&self) -> Tol;

    /// Chart coordinates of `p`, trailing lanes zero for charts under four
    /// dimensions.
    fn point_components(&self, p: Self::Point) -> [f32; 4];

    /// Chart coordinates of `v`, trailing lanes zero.
    fn vector_components(&self, v: Self::Vector) -> [f32; 4];
}

// ---------------------------------------------------------------------------
// Harness primitives
// ---------------------------------------------------------------------------

/// Seeded xorshift32 (Marsaglia, *Xorshift RNGs*, J. Stat. Soft. 8(14), 2003,
/// §3). Every fixture's sample set is a pure function of a hardcoded seed, per
/// the Tier 0 determinism contract.
struct Xorshift32(u32);

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// Uniform in `[-1, 1]`.
    fn signed_unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Seeded points in a ball of radius `max_radius`, radius bounded away from
/// zero so no sample is accidentally coincident with the chart origin. Shared
/// by the two `Vec3` chart fixtures, whose domains are both a ball.
fn ball_samples(seed: u32, count: usize, max_radius: f32) -> Vec<Vec3> {
    let mut rng = Xorshift32::new(seed);
    (0..count)
        .map(|_| {
            let dir =
                Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()).normalize();
            dir * (0.1 + 0.9 * rng.signed_unit().abs()) * max_radius
        })
        .collect()
}

/// Absolute below unit magnitude, relative above. Without the switch a single
/// fixture number cannot cover both a residual at the origin and one at the
/// Poincaré saturation shell, where lambda is ~2e7 and every norm carries it.
fn agrees(got: f32, want: f32, tol: f32) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

/// Largest absolute chart coordinate of `p`. Point residuals are compared
/// absolutely below unit magnitude and relatively above: an f32 coordinate's
/// ulp is proportional to its magnitude and the flat charts are unbounded, so a
/// fixed absolute budget would buy a different number of ulps at every sample.
/// The bounded charts (both S³ models, the Poincaré ball) never leave the
/// absolute regime.
fn chart_magnitude<F: SpaceFixture>(f: &F, p: F::Point) -> f32 {
    f.point_components(p)
        .iter()
        .fold(0.0f32, |m, c| m.max(c.abs()))
}

fn metric_norm<F: SpaceFixture>(f: &F, at: F::Point, v: F::Vector) -> f32 {
    // Clamp before the root: a fixture `inner` can round a vanishing norm
    // negative, and a NaN here would read as a tolerance failure rather than
    // as the finiteness failure it is.
    f.inner(at, v, v).max(0.0).sqrt()
}

/// Metric length of `u - v`, both tangents at `at`.
fn metric_residual<F: SpaceFixture>(f: &F, at: F::Point, u: F::Vector, v: F::Vector) -> f32 {
    metric_norm(f, at, f.combine(u, 1.0, v, -1.0))
}

/// Riemannian angle between two tangents at `at`, from the fixture's metric.
fn metric_angle<F: SpaceFixture>(f: &F, at: F::Point, u: F::Vector, v: F::Vector) -> f32 {
    let denom = metric_norm(f, at, u) * metric_norm(f, at, v);
    (f.inner(at, u, v) / denom).clamp(-1.0, 1.0).acos()
}

/// Side of the geodesic triangle in `geodesic_triangle_angle_excess_matches_
/// gauss_bonnet`, in metric units. Gauss-Bonnet is exact for a geodesic
/// triangle, but the harness compares against the *flat* equilateral area
/// `(sqrt(3)/4)L^2`, whose relative error is O(K L^2); at L = 0.1 that is ~1e-3,
/// well under the 5% bound below, while the excess itself (4.3e-3 at |K| = 1)
/// stays far above the f32 noise in the six `log` calls that build it.
const TRIANGLE_SIDE: f32 = 0.1;

/// Fraction of the triangle's area by which the measured angle excess may miss
/// `K * area`. Sized by the flat-area truncation above, not by what makes the
/// suite green; a sign flip in `K` misses by `2 * area`, forty times this.
const GAUSS_BONNET_RELATIVE_TOL: f32 = 0.05;

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

mod invariants {
    use super::*;

    /// `d(a, a) = 0` and `d(a, b) = d(b, a)`.
    pub fn distance_is_symmetric_and_zero_on_the_diagonal<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            let self_distance = s.distance(a, a);
            assert!(
                self_distance.abs() <= tol.point,
                "d(a, a) = {self_distance} at {:?}",
                f.point_components(a)
            );
            for &b in &points {
                let ab = s.distance(a, b);
                let ba = s.distance(b, a);
                assert!(
                    agrees(ab, ba, tol.scalar),
                    "d(a, b) = {ab} but d(b, a) = {ba} at {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b)
                );
            }
        }
    }

    /// Distance is finite and non-negative on every sampled pair.
    pub fn distance_is_finite_and_nonnegative<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                let d = s.distance(a, b);
                assert!(
                    d.is_finite() && d >= 0.0,
                    "d = {d} at {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b)
                );
            }
        }
    }

    /// `d(a, c) <= d(a, b) + d(b, c)` over every sampled triple.
    pub fn distance_satisfies_the_triangle_inequality<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        // The matrix is built once: `distance` is a Gauss-Newton shooting solve
        // for `BlendedSpace`, and the naive triple loop would call it n^3 times
        // for n^2 distinct values.
        let n = points.len();
        let mut d = Vec::with_capacity(n * n);
        for &a in &points {
            for &b in &points {
                d.push(s.distance(a, b));
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let direct = d[i * n + k];
                    let via = d[i * n + j] + d[j * n + k];
                    assert!(
                        direct <= via + tol.scalar * via.max(1.0),
                        "d(a, c) = {direct} exceeds d(a, b) + d(b, c) = {via} at \
                         {:?} {:?} {:?}",
                        f.point_components(points[i]),
                        f.point_components(points[j]),
                        f.point_components(points[k])
                    );
                }
            }
        }
    }

    /// `exp(a, log(a, b)) = b`, the direction every impl already tests.
    pub fn exp_inverts_log_on_sampled_pairs<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                let recovered = s.exp(a, s.log(a, b));
                assert_point_agrees(f, &s, recovered, b, tol.point, "exp(a, log(a, b))");
            }
        }
    }

    /// `log(a, exp(a, v)) = v`. The direction no impl tested: a pair of
    /// compensating errors in `exp` and `log` survives the forward round trip
    /// and dies here.
    pub fn log_inverts_exp_on_sampled_tangents<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        for a in f.points() {
            for v in f.tangents(a) {
                let recovered = s.log(a, s.exp(a, v));
                let residual = metric_residual(f, a, recovered, v);
                let scale = metric_norm(f, a, v).max(1.0);
                assert!(
                    residual <= tol.vector * scale,
                    "log(a, exp(a, v)) missed v by {residual} at {:?} {:?}",
                    f.point_components(a),
                    f.vector_components(v)
                );
            }
        }
    }

    /// `|log(a, b)|_metric = d(a, b)`. The item that fails loudly when a
    /// conformal factor is dropped, and the reason `inner` and not `norm` is
    /// the fixture hook.
    pub fn log_magnitude_equals_geodesic_distance<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                let magnitude = metric_norm(f, a, s.log(a, b));
                let distance = s.distance(a, b);
                assert!(
                    agrees(magnitude, distance, tol.scalar),
                    "|log(a, b)| = {magnitude} but d(a, b) = {distance} at {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b)
                );
            }
        }
    }

    /// `d(a, exp(a, v)) = |v|_metric`, the calibration in the other direction.
    pub fn exp_advances_by_the_metric_norm_of_its_tangent<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        for a in f.points() {
            for v in f.tangents(a) {
                let travelled = s.distance(a, s.exp(a, v));
                let norm = metric_norm(f, a, v);
                assert!(
                    agrees(travelled, norm, tol.scalar),
                    "exp travelled {travelled} for a tangent of norm {norm} at {:?} {:?}",
                    f.point_components(a),
                    f.vector_components(v)
                );
            }
        }
    }

    /// `d(g.a, g.b) = d(a, b)`. The one item that ties the isometry surface to
    /// the metric: without it an `Iso` can be any invertible map at all.
    pub fn distance_is_invariant_under_isometry<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for g in f.isos() {
            for &a in &points {
                for &b in &points {
                    let before = s.distance(a, b);
                    let after = s.distance(s.iso_apply(g, a), s.iso_apply(g, b));
                    assert!(
                        agrees(after, before, tol.scalar),
                        "isometry changed d from {before} to {after} at {:?} {:?}",
                        f.point_components(a),
                        f.point_components(b)
                    );
                }
            }
        }
    }

    /// `id.p = p`, and `id . g` and `g . id` act as `g`.
    pub fn iso_identity_is_neutral<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        let id = s.iso_identity();
        let isos = f.isos();
        for p in f.points() {
            assert_point_agrees(f, &s, s.iso_apply(id, p), p, tol.point, "id.p");
            for &g in &isos {
                let want = s.iso_apply(g, p);
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(id, g), p),
                    want,
                    tol.point,
                    "(id . g).p",
                );
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(g, id), p),
                    want,
                    tol.point,
                    "(g . id).p",
                );
            }
        }
    }

    /// `g . g^-1` and `g^-1 . g` both act as the identity.
    pub fn iso_inverse_is_two_sided<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        for g in f.isos() {
            let inv = s.iso_inverse(g);
            for p in f.points() {
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(g, inv), p),
                    p,
                    tol.point,
                    "(g . g^-1).p",
                );
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(inv, g), p),
                    p,
                    tol.point,
                    "(g^-1 . g).p",
                );
            }
        }
    }

    /// `(a . b) . c` and `a . (b . c)` act alike. Asserted for no impl in the
    /// workspace before this suite.
    pub fn iso_compose_is_associative<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        let isos = f.isos();
        let points = f.points();
        for &a in &isos {
            for &b in &isos {
                for &c in &isos {
                    let left = s.iso_compose(s.iso_compose(a, b), c);
                    let right = s.iso_compose(a, s.iso_compose(b, c));
                    for &p in &points {
                        assert_point_agrees(
                            f,
                            &s,
                            s.iso_apply(left, p),
                            s.iso_apply(right, p),
                            tol.point,
                            "((a . b) . c).p vs (a . (b . c)).p",
                        );
                    }
                }
            }
        }
    }

    /// `(a . b).p = a.(b.p)`, the item that pins the `Quat`-versus-`Rotor4`
    /// composition-order divergence between the R³ and R⁴ isometries.
    pub fn iso_compose_matches_sequential_apply<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        let isos = f.isos();
        for &a in &isos {
            for &b in &isos {
                for p in f.points() {
                    assert_point_agrees(
                        f,
                        &s,
                        s.iso_apply(s.iso_compose(a, b), p),
                        s.iso_apply(a, s.iso_apply(b, p)),
                        tol.point,
                        "(a . b).p vs a.(b.p)",
                    );
                }
            }
        }
    }

    /// `g.exp(p, v) = exp(g.p, dg(v))`: the isometry's differential is the one
    /// that intertwines with the exponential map.
    ///
    /// Degenerates into an `exp`/`log` consistency check for the two impls that
    /// *define* `iso_transport` as that round trip; the flat impls and the
    /// ambient S³ use closed forms, and so will every future impl that can.
    pub fn iso_transport_is_the_differential_of_iso_apply<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        for g in f.isos() {
            for p in f.points() {
                for v in f.tangents(p) {
                    let moved = s.iso_apply(g, s.exp(p, v));
                    let transported = s.exp(s.iso_apply(g, p), s.iso_transport(g, p, v));
                    assert_point_agrees(
                        f,
                        &s,
                        moved,
                        transported,
                        tol.point,
                        "g.exp(p, v) vs exp(g.p, dg(v))",
                    );
                }
            }
        }
    }

    /// `|dg(v)|_{at g.p} = |v|_{at p}`.
    pub fn iso_transport_preserves_the_metric_norm<F: SpaceFixture>(f: &F) {
        if declared_trivial_group(f) {
            return;
        }
        let s = f.space();
        let tol = f.tol();
        for g in f.isos() {
            for p in f.points() {
                let moved = s.iso_apply(g, p);
                for v in f.tangents(p) {
                    let before = metric_norm(f, p, v);
                    let after = metric_norm(f, moved, s.iso_transport(g, p, v));
                    assert!(
                        agrees(after, before, tol.scalar),
                        "iso_transport changed the norm from {before} to {after} at \
                         {:?} {:?}",
                        f.point_components(p),
                        f.vector_components(v)
                    );
                }
            }
        }
    }

    /// `|PT(a, b, v)|_{at b} = |v|_{at a}`.
    ///
    /// Nothing here asserts which path the transport follows: `Space` makes the
    /// path implementation-defined and `BlendedSpace` follows the chart
    /// straight line rather than the geodesic, so an invariant that assumed the
    /// geodesic would fail the oracle for being the oracle.
    pub fn parallel_transport_preserves_the_metric_norm<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                for v in f.tangents(a) {
                    let before = metric_norm(f, a, v);
                    let after = metric_norm(f, b, s.parallel_transport(a, b, v));
                    assert!(
                        agrees(after, before, tol.scalar),
                        "transport changed the norm from {before} to {after} at \
                         {:?} {:?} {:?}",
                        f.point_components(a),
                        f.point_components(b),
                        f.vector_components(v)
                    );
                }
            }
        }
    }

    /// `PT(a, b, s*u + t*w) = s*PT(a, b, u) + t*PT(a, b, w)`: transport is a
    /// linear map between tangent spaces, not merely a norm-preserving one.
    pub fn parallel_transport_is_linear_in_the_transported_vector<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        // Coefficients of opposite sign and unequal magnitude: equal ones
        // cannot separate a linear map from one that only respects sums.
        let (cu, cw) = (1.5, -0.75);
        for &a in &points {
            let tangents = f.tangents(a);
            for &b in &points {
                for (i, &u) in tangents.iter().enumerate() {
                    for &w in &tangents[i..] {
                        let combined = s.parallel_transport(a, b, f.combine(u, cu, w, cw));
                        let separate = f.combine(
                            s.parallel_transport(a, b, u),
                            cu,
                            s.parallel_transport(a, b, w),
                            cw,
                        );
                        let residual = metric_residual(f, b, combined, separate);
                        let scale = metric_norm(f, b, separate).max(1.0);
                        assert!(
                            residual <= tol.vector * scale,
                            "transport is not linear by {residual} at {:?} {:?} {:?} {:?}",
                            f.point_components(a),
                            f.point_components(b),
                            f.vector_components(u),
                            f.vector_components(w)
                        );
                    }
                }
            }
        }
    }

    /// The polyline form agrees with the single-segment form on one segment,
    /// and is the identity on paths too short to have one. Pins
    /// `BlendedSpace`'s override against its own segment method.
    pub fn parallel_transport_along_one_segment_matches_parallel_transport<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for v in f.tangents(a) {
                let residual = metric_residual(f, a, s.parallel_transport_along(&[], v), v);
                assert!(residual <= tol.vector, "empty path moved v by {residual}");
                let residual = metric_residual(f, a, s.parallel_transport_along(&[a], v), v);
                assert!(
                    residual <= tol.vector,
                    "one-point path moved v by {residual}"
                );
                for &b in &points {
                    let along = s.parallel_transport_along(&[a, b], v);
                    let direct = s.parallel_transport(a, b, v);
                    let residual = metric_residual(f, b, along, direct);
                    let scale = metric_norm(f, b, direct).max(1.0);
                    assert!(
                        residual <= tol.vector * scale,
                        "polyline transport differs from segment transport by {residual} \
                         at {:?} {:?}",
                        f.point_components(a),
                        f.point_components(b)
                    );
                }
            }
        }
    }

    /// `PT(a, b, log(a, b))` is the forward tangent at `b`, which points along
    /// `-log(b, a)`: a geodesic's own velocity field is parallel along it. This
    /// is the item that sees the rotation. Norm, linearity and polyline
    /// consistency are the only other things the suite asks of transport, and
    /// the identity map satisfies all three wherever the metric agrees at the
    /// two endpoints, which for the ambient S³ fixture is every pair it
    /// samples. In a flat chart both sides reduce to the same expression, so
    /// there the item is a tautology and correctly so: the identity is the
    /// transport.
    ///
    /// Coincident and near-coincident pairs need no guard: both sides vanish
    /// with the separation, and the budget below is absolute for separations
    /// under one metric unit. Worst measured residual over the curved fixtures
    /// is 1.7e-2 of the budget; the identity map lands at 1.8e3 (H³) to 4.9e4
    /// (ambient S³) and swapping the gyration's operands at 3.1e3, so the item
    /// separates by five decades rather than by a margin.
    pub fn parallel_transport_carries_a_geodesic_tangent_along_its_own_geodesic<F: SpaceFixture>(
        f: &F,
    ) {
        let (ratio, a, b) = worst_geodesic_tangent_ratio(f);
        if !F::TRANSPORT_FOLLOWS_THE_GEODESIC {
            assert!(
                ratio > 1.0,
                "the fixture declares a non-geodesic transport path but the \
                 worst sampled pair misses the geodesic tangent by only \
                 {ratio} of the vector budget: the exemption is stale",
            );
            return;
        }
        assert!(
            ratio <= 1.0,
            "transport of log(a, b) misses the forward tangent at b by \
             {ratio} of the vector budget at {:?} {:?}",
            f.point_components(a),
            f.point_components(b)
        );
    }

    /// The geodesics determine the transport: the pole ladder rebuilds
    /// `PT(a, b, .)` from `exp` and `log` alone (Pennec, *Parallel Transport
    /// with Pole Ladder: a Third Order Scheme in Affine Connection Spaces which
    /// is Exact in Affine Symmetric Spaces*, arXiv:1805.11436, 2018, §3), so an
    /// impl whose `exp` and `log` are already pinned has no freedom
    /// left in its transport. This is the item that sees a twist about the
    /// direction of travel: the geodesic-tangent item above pins the image of
    /// that direction alone, and any rotation about it fixes that image, while
    /// norm and linearity are blind to a rotation outright.
    ///
    /// Exact here, not asymptotic. Write `s_m` for the geodesic symmetry at the
    /// midpoint `m` of `a` and `b`, the map `x -> exp(m, -log(m, x))`. In a
    /// symmetric space `s_m` is an isometry with `s_m(a) = b`, and the
    /// transvection `s_m . s_a` is the transport along the geodesic (Helgason,
    /// *Differential Geometry, Lie Groups, and Symmetric Spaces*, 1978, Ch. IV,
    /// §3), so `s_m(exp(a, v)) = exp(b, -PT(a, b, v))`: the construction below,
    /// read backwards. Every fixture here that declares
    /// `TRANSPORT_FOLLOWS_THE_GEODESIC` is a symmetric space, flat R^n or one
    /// of the two constant-curvature models; the ladder's leading error term
    /// carries `∇R`, so a future impl with non-parallel curvature would fail
    /// this item rather than pass it vacuously, which is the direction a stale
    /// precondition should fail in.
    ///
    /// The pairs are unfiltered. Coincident `a` and `b` collapse the ladder to
    /// the `exp`/`log` round trip, which is the identity the transport also
    /// returns; near-antipodal pairs are the fixtures' declared degenerate set
    /// and are not sampled here.
    ///
    /// Worst measured residual is 2.5e-1 of the budget on the flat trio, where
    /// the ladder is the identity map and the number is the six `exp`/`log`
    /// calls' own rounding at chart coordinates of order 1, and 1.5e-2 on the
    /// curved fixtures. A 0.05 rad twist about the direction of travel lands at
    /// 1.1e3 (ambient S³) and 7.0e1 (H³) while leaving every other item in this
    /// suite green, so the separation is four decades and not a margin.
    pub fn parallel_transport_matches_the_one_its_own_geodesics_imply<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        let mut worst = (0.0f32, points[0], points[0], f.tangents(points[0])[0]);
        for &a in &points {
            for &b in &points {
                for v in f.tangents(a) {
                    let ladder = pole_ladder(f, &s, a, b, v);
                    let residual = metric_residual(f, b, s.parallel_transport(a, b, v), ladder);
                    let ratio = residual / (tol.vector * metric_norm(f, b, ladder).max(1.0));
                    // `!(ratio <= worst.0)` rather than `ratio > worst.0`: a NaN
                    // ratio compares false against everything, so the ordinary
                    // form would leave `worst` at its seed and report the item
                    // green on a transport that returned NaN.
                    if !(ratio <= worst.0) {
                        worst = (ratio, a, b, v);
                    }
                }
            }
        }
        let (ratio, a, b, v) = worst;
        if !F::TRANSPORT_FOLLOWS_THE_GEODESIC {
            assert!(
                ratio > 1.0,
                "the fixture declares a non-geodesic transport path but its \
                 transport agrees with the geodesic one to {ratio} of the \
                 vector budget everywhere sampled: the exemption is stale",
            );
            return;
        }
        assert!(
            ratio <= 1.0,
            "transport misses the map its own geodesics imply by {ratio} of \
             the vector budget at {:?} {:?} {:?}",
            f.point_components(a),
            f.point_components(b),
            f.vector_components(v)
        );
    }

    /// Gauss-Bonnet on a small geodesic triangle: the angle excess is
    /// `K * area` (do Carmo, *Differential Geometry of Curves and Surfaces*,
    /// 1976, §4.5). This is what independently cross-checks a fixture's
    /// `inner`: a wrong metric produces the wrong angle sum.
    ///
    /// A fixture with no constant curvature still runs the construction and
    /// pins that the angles are finite and sum near `pi` for a small triangle;
    /// its curvature is pinned where it varies, not here.
    pub fn geodesic_triangle_angle_excess_matches_gauss_bonnet<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let l = TRIANGLE_SIDE;
        let apex = f.points()[0];
        let (e1, e2) = metric_orthonormal_pair(f, apex);

        // Equilateral in the tangent plane at `apex`: the third vertex sits at
        // 60 degrees, so all three sides are `l` to O(K l^3).
        let b = s.exp(apex, f.combine(e1, l, e2, 0.0));
        let c = s.exp(apex, f.combine(e1, l * 0.5, e2, l * 3.0_f32.sqrt() * 0.5));

        let angle = |at, u, w| metric_angle(f, at, s.log(at, u), s.log(at, w));
        let sum = angle(apex, b, c) + angle(b, apex, c) + angle(c, apex, b);
        assert!(sum.is_finite(), "triangle angle sum is {sum}");

        let area = 3.0_f32.sqrt() / 4.0 * l * l;
        let excess = sum - std::f32::consts::PI;
        let Some(k) = f.curvature() else {
            assert!(
                excess.abs() <= 1.0,
                "a triangle of side {l} has angle excess {excess}, far past any \
                 small-triangle regime"
            );
            return;
        };
        assert!(
            (excess - k * area).abs() <= GAUSS_BONNET_RELATIVE_TOL * area,
            "angle excess {excess} misses K * area = {} at K = {k}",
            k * area
        );
    }

    /// Nothing the fixture declares degenerate produces NaN, infinity or a
    /// panic, and transport still preserves the metric norm there, at the
    /// looser tolerance the conditioning class earns.
    pub fn degenerate_inputs_stay_finite<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        for (a, b) in f.degenerate_pairs() {
            let d = s.distance(a, b);
            assert!(
                d.is_finite() && d >= 0.0,
                "d = {d} at {:?} {:?}",
                f.point_components(a),
                f.point_components(b)
            );
            let v = s.log(a, b);
            assert_vector_is_finite(f, v, "log");
            assert_point_is_finite(f, s.exp(a, v), "exp(a, log(a, b))");
            for w in f.tangents(a) {
                let transported = s.parallel_transport(a, b, w);
                assert_vector_is_finite(f, transported, "parallel_transport");
                // Norm preservation is a statement about a Riemannian metric,
                // so it has no content where there is none. A degenerate set is
                // built from the chart boundary and the cut locus, and the
                // metric diverges at the Poincaré ideal boundary and at the S³
                // equator; the transported vector's own finiteness is asserted
                // above, so skipping here cannot hide a blow-up.
                if !metric_is_defined(f, a, w) || !metric_is_defined(f, b, transported) {
                    continue;
                }
                let before = metric_norm(f, a, w);
                let after = metric_norm(f, b, transported);
                assert!(
                    agrees(after, before, tol.degenerate),
                    "transport changed the norm from {before} to {after} at {:?} {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b),
                    f.vector_components(w)
                );
            }
            for g in f.isos() {
                assert_point_is_finite(f, s.iso_apply(g, a), "iso_apply");
                for w in f.tangents(a) {
                    assert_vector_is_finite(f, s.iso_transport(g, a, w), "iso_transport");
                }
            }
        }
    }

    /// Same binary, same inputs, same bits. Compared with `to_bits`, not with a
    /// tolerance; no golden constant, because every curved impl here is built
    /// on libm transcendentals that are not bit-portable across targets, and
    /// cross-target bit-identity is not part of the determinism contract.
    pub fn sampled_calls_are_bit_reproducible<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let points = f.points();
        // Consecutive pairs, not the full product: repeatability is a property
        // of each call, so coverage of the call graph is what matters and a
        // quadratic sweep buys nothing.
        for w in points.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert_eq!(
                s.distance(a, b).to_bits(),
                s.distance(a, b).to_bits(),
                "distance is not bit-reproducible at {:?} {:?}",
                f.point_components(a),
                f.point_components(b)
            );
            assert_eq!(
                vector_bits(f, s.log(a, b)),
                vector_bits(f, s.log(a, b)),
                "log is not bit-reproducible at {:?} {:?}",
                f.point_components(a),
                f.point_components(b)
            );
            for v in f.tangents(a) {
                assert_eq!(
                    point_bits(f, s.exp(a, v)),
                    point_bits(f, s.exp(a, v)),
                    "exp is not bit-reproducible at {:?} {:?}",
                    f.point_components(a),
                    f.vector_components(v)
                );
                assert_eq!(
                    vector_bits(f, s.parallel_transport(a, b, v)),
                    vector_bits(f, s.parallel_transport(a, b, v)),
                    "parallel_transport is not bit-reproducible at {:?} {:?}",
                    f.point_components(a),
                    f.vector_components(v)
                );
                for g in f.isos() {
                    assert_eq!(
                        point_bits(f, s.iso_apply(g, a)),
                        point_bits(f, s.iso_apply(g, a)),
                        "iso_apply is not bit-reproducible at {:?}",
                        f.point_components(a)
                    );
                    assert_eq!(
                        vector_bits(f, s.iso_transport(g, a, v)),
                        vector_bits(f, s.iso_transport(g, a, v)),
                        "iso_transport is not bit-reproducible at {:?} {:?}",
                        f.point_components(a),
                        f.vector_components(v)
                    );
                }
            }
        }
    }

    // ---- shared machinery ------------------------------------------------

    /// Returns whether the fixture claims a trivial isometry group, asserting
    /// the claim on the way out so the group-law items report a real hole
    /// instead of passing vacuously. Also enforces the floor of three
    /// isometries: two operands cannot distinguish a composition convention
    /// from its transpose in a group that is nearly abelian on the sample.
    fn declared_trivial_group<F: SpaceFixture>(f: &F) -> bool {
        let isos = f.isos();
        if !F::ISOMETRY_GROUP_IS_TRIVIAL {
            assert!(
                isos.len() >= 3,
                "a non-trivial isometry group needs at least three sampled \
                 isometries, got {}",
                isos.len()
            );
            return false;
        }
        let s = f.space();
        assert_eq!(
            isos.len(),
            1,
            "a trivial isometry group has exactly one element"
        );
        for p in f.points() {
            assert_eq!(
                point_bits(f, s.iso_apply(isos[0], p)),
                point_bits(f, p),
                "a trivial isometry group acts as the identity"
            );
        }
        true
    }

    /// Worst `|PT(a, b, log(a, b)) - (-log(b, a))|_b` over the sampled pairs,
    /// as a multiple of the fixture's vector budget, with the pair that
    /// attained it. Shared so the geodesic-tangent item and its staleness check
    /// measure the same quantity.
    ///
    /// The residual carries the magnitudes rather than comparing unit
    /// directions: `|log|` is pinned against `distance` by its own item, and
    /// normalizing here would divide the transport's error by a separation
    /// that goes to zero on the diagonal.
    fn worst_geodesic_tangent_ratio<F: SpaceFixture>(f: &F) -> (f32, F::Point, F::Point) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        let mut worst = (0.0, points[0], points[0]);
        for &a in &points {
            for &b in &points {
                let reverse = s.log(b, a);
                let forward = scaled(f, reverse, -1.0);
                let transported = s.parallel_transport(a, b, s.log(a, b));
                let residual = metric_residual(f, b, transported, forward);
                let ratio = residual / (tol.vector * metric_norm(f, b, forward).max(1.0));
                // NaN-safe, for the reason given at the ladder item above.
                if !(ratio <= worst.0) {
                    worst = (ratio, a, b);
                }
            }
        }
        worst
    }

    /// One rung of the pole ladder: parallel transport of `v` from `a` to `b`
    /// along the geodesic, built from `exp` and `log` and nothing else, so it is
    /// independent of the impl's own transport. Mirror `exp(a, v)` through the
    /// geodesic midpoint and read the result off at `b`, negated. Both signs,
    /// the exactness claim and the citations are at the item that uses this.
    fn pole_ladder<F: SpaceFixture>(
        f: &F,
        s: &F::S,
        a: F::Point,
        b: F::Point,
        v: F::Vector,
    ) -> F::Vector {
        let midpoint = s.exp(a, scaled(f, s.log(a, b), 0.5));
        let mirrored = s.exp(midpoint, scaled(f, s.log(midpoint, s.exp(a, v)), -1.0));
        scaled(f, s.log(b, mirrored), -1.0)
    }

    /// `s*v`. `combine` takes two operands and these constructions have one.
    fn scaled<F: SpaceFixture>(f: &F, v: F::Vector, s: f32) -> F::Vector {
        f.combine(v, s, v, 0.0)
    }

    /// Two metric-orthonormal tangents at `at`, Gram-Schmidt over the fixture's
    /// `inner` (do Carmo, *Differential Geometry of Curves and Surfaces*, 1976,
    /// §1.4). The harness can build them generically only because it has the
    /// metric.
    fn metric_orthonormal_pair<F: SpaceFixture>(f: &F, at: F::Point) -> (F::Vector, F::Vector) {
        let tangents = f.tangents(at);
        assert!(
            tangents.len() >= 2,
            "the angle-excess item needs two independent tangents"
        );
        let n1 = metric_norm(f, at, tangents[0]);
        assert!(n1 > 0.0, "the first sampled tangent has zero metric norm");
        let e1 = f.combine(tangents[0], 1.0 / n1, tangents[0], 0.0);

        let raw = tangents[1];
        let projection = f.inner(at, raw, e1);
        let orthogonal = f.combine(raw, 1.0, e1, -projection);
        let n2 = metric_norm(f, at, orthogonal);
        assert!(
            n2 > 0.0,
            "the first two sampled tangents are metrically parallel"
        );
        (e1, f.combine(orthogonal, 1.0 / n2, orthogonal, 0.0))
    }

    fn assert_point_agrees<F: SpaceFixture>(
        f: &F,
        s: &F::S,
        got: F::Point,
        want: F::Point,
        tol: f32,
        what: &str,
    ) {
        let residual = s.distance(got, want);
        assert!(
            residual <= tol * chart_magnitude(f, want).max(1.0),
            "{what} is off by {residual}: {:?} vs {:?}",
            f.point_components(got),
            f.point_components(want)
        );
    }

    /// Raw bits of a point's coordinates. `f32`'s `PartialEq` is the wrong
    /// comparison for a reproducibility claim: it calls two NaNs different and
    /// the two signed zeros the same, and both are outcomes a repeatability
    /// regression can produce.
    fn point_bits<F: SpaceFixture>(f: &F, p: F::Point) -> [u32; 4] {
        f.point_components(p).map(f32::to_bits)
    }

    fn vector_bits<F: SpaceFixture>(f: &F, v: F::Vector) -> [u32; 4] {
        f.vector_components(v).map(f32::to_bits)
    }

    /// Whether the fixture reports a finite metric for `v` at `at`.
    fn metric_is_defined<F: SpaceFixture>(f: &F, at: F::Point, v: F::Vector) -> bool {
        f.inner(at, v, v).is_finite()
    }

    fn assert_point_is_finite<F: SpaceFixture>(f: &F, p: F::Point, what: &str) {
        let c = f.point_components(p);
        assert!(c.iter().all(|x| x.is_finite()), "{what} returned {c:?}");
    }

    fn assert_vector_is_finite<F: SpaceFixture>(f: &F, v: F::Vector, what: &str) {
        let c = f.vector_components(v);
        assert!(c.iter().all(|x| x.is_finite()), "{what} returned {c:?}");
    }
}

// ---------------------------------------------------------------------------
// The macro
// ---------------------------------------------------------------------------

/// One `#[test]` per invariant, named after the property, so a failure names
/// the property rather than "conformance".
macro_rules! conformance_tests {
    ($fixture:expr; $($invariant:ident),+ $(,)?) => {
        $(
            #[test]
            fn $invariant() {
                invariants::$invariant(&$fixture);
            }
        )+
    };
}

macro_rules! conformance_suite {
    ($suite:ident, $fixture:expr) => {
        mod $suite {
            use super::*;

            conformance_tests!($fixture;
                distance_is_symmetric_and_zero_on_the_diagonal,
                distance_is_finite_and_nonnegative,
                distance_satisfies_the_triangle_inequality,
                exp_inverts_log_on_sampled_pairs,
                log_inverts_exp_on_sampled_tangents,
                log_magnitude_equals_geodesic_distance,
                exp_advances_by_the_metric_norm_of_its_tangent,
                distance_is_invariant_under_isometry,
                iso_identity_is_neutral,
                iso_inverse_is_two_sided,
                iso_compose_is_associative,
                iso_compose_matches_sequential_apply,
                iso_transport_is_the_differential_of_iso_apply,
                iso_transport_preserves_the_metric_norm,
                parallel_transport_preserves_the_metric_norm,
                parallel_transport_is_linear_in_the_transported_vector,
                parallel_transport_along_one_segment_matches_parallel_transport,
                parallel_transport_carries_a_geodesic_tangent_along_its_own_geodesic,
                parallel_transport_matches_the_one_its_own_geodesics_imply,
                geodesic_triangle_angle_excess_matches_gauss_bonnet,
                degenerate_inputs_stay_finite,
                sampled_calls_are_bit_reproducible,
            );
        }
    };
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Euclidean R². Extent: coordinates of order 1. Tolerances are the closed-form
/// ones the inline tests already meet at a wider extent.
struct EuclideanR2Fixture;

impl SpaceFixture for EuclideanR2Fixture {
    type Point = Vec2;
    type Vector = Vec2;
    type Iso = Iso2;
    type S = EuclideanR2;

    fn space(&self) -> EuclideanR2 {
        EuclideanR2
    }

    fn points(&self) -> Vec<Vec2> {
        let mut rng = Xorshift32::new(0x00E2_0F1A);
        (0..6)
            .map(|_| Vec2::new(rng.signed_unit(), rng.signed_unit()) * 1.5)
            .collect()
    }

    fn tangents(&self, _at: Vec2) -> Vec<Vec2> {
        let mut rng = Xorshift32::new(0x00E2_7A46);
        (0..3)
            .map(|_| Vec2::new(rng.signed_unit(), rng.signed_unit()) * 0.4)
            .collect()
    }

    fn isos(&self) -> Vec<Iso2> {
        vec![
            Iso2 {
                rotation: Bivector2(0.5).exp(),
                translation: Vec2::new(1.0, 0.0),
            },
            Iso2 {
                rotation: Bivector2(-0.9).exp(),
                translation: Vec2::new(0.0, 2.0),
            },
            Iso2::from_translation(Vec2::new(-0.7, 0.3)),
        ]
    }

    fn inner(&self, _at: Vec2, u: Vec2, v: Vec2) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec2, s: f32, v: Vec2, t: f32) -> Vec2 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec2, Vec2)> {
        let far = Vec2::new(1.0e7, -3.0e6);
        vec![
            (Vec2::ZERO, Vec2::ZERO),
            (far, far),
            (Vec2::ZERO, far),
            (Vec2::new(1.0e-30, 0.0), Vec2::ZERO),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(0.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-6,
            vector: 1e-6,
            scalar: 1e-6,
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec2) -> [f32; 4] {
        [p.x, p.y, 0.0, 0.0]
    }

    fn vector_components(&self, v: Vec2) -> [f32; 4] {
        [v.x, v.y, 0.0, 0.0]
    }
}

/// Euclidean R³. Extent: coordinates of order 1.
struct EuclideanR3Fixture;

impl SpaceFixture for EuclideanR3Fixture {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = Iso3;
    type S = EuclideanR3;

    fn space(&self) -> EuclideanR3 {
        EuclideanR3
    }

    fn points(&self) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00E3_0F1A);
        (0..6)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 1.5)
            .collect()
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00E3_7A46);
        (0..3)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.4)
            .collect()
    }

    fn isos(&self) -> Vec<Iso3> {
        vec![
            Iso3 {
                rotation: Quat::from_rotation_z(0.4),
                translation: Vec3::new(1.0, 0.0, 0.0),
            },
            Iso3 {
                rotation: Quat::from_rotation_x(0.9),
                translation: Vec3::new(0.0, 2.0, -1.0),
            },
            Iso3 {
                rotation: Quat::from_rotation_y(-0.6),
                translation: Vec3::new(-0.5, 0.25, 3.0),
            },
        ]
    }

    fn inner(&self, _at: Vec3, u: Vec3, v: Vec3) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let far = Vec3::new(1.0e7, -3.0e6, 5.0e6);
        vec![
            (Vec3::ZERO, Vec3::ZERO),
            (far, far),
            (Vec3::ZERO, far),
            (Vec3::new(1.0e-30, 0.0, 0.0), Vec3::ZERO),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(0.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-6,
            vector: 1e-6,
            scalar: 1e-6,
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

/// Euclidean R⁴. Extent: coordinates of order 1. The isometries carry compound
/// (two-plane) rotors, the case where a `Rotor4` composition order error does
/// not cancel.
struct EuclideanR4Fixture;

impl SpaceFixture for EuclideanR4Fixture {
    type Point = Vec4;
    type Vector = Vec4;
    type Iso = Iso4Flat;
    type S = EuclideanR4;

    fn space(&self) -> EuclideanR4 {
        EuclideanR4
    }

    fn points(&self) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x00E4_0F1A);
        (0..6)
            .map(|_| {
                Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                ) * 1.5
            })
            .collect()
    }

    fn tangents(&self, _at: Vec4) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x00E4_7A46);
        (0..3)
            .map(|_| {
                Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                ) * 0.4
            })
            .collect()
    }

    fn isos(&self) -> Vec<Iso4Flat> {
        vec![
            Iso4Flat {
                rotation: Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.2).exp(),
                translation: Vec4::new(1.0, 0.0, 0.0, 0.0),
            },
            Iso4Flat {
                rotation: Bivector4::new(0.0, 0.0, 0.0, 0.9, 0.0, 0.0).exp(),
                translation: Vec4::new(0.0, 2.0, -1.0, 0.5),
            },
            Iso4Flat {
                rotation: Bivector4::new(0.3, 0.1, -0.2, 0.4, 0.0, 0.15).exp(),
                translation: Vec4::new(-0.5, 0.25, 3.0, -2.0),
            },
        ]
    }

    fn inner(&self, _at: Vec4, u: Vec4, v: Vec4) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec4, s: f32, v: Vec4, t: f32) -> Vec4 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec4, Vec4)> {
        let far = Vec4::new(1.0e7, -3.0e6, 5.0e6, 2.0e6);
        vec![
            (Vec4::ZERO, Vec4::ZERO),
            (far, far),
            (Vec4::ZERO, far),
            (Vec4::new(1.0e-30, 0.0, 0.0, 0.0), Vec4::ZERO),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(0.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-6,
            vector: 1e-6,
            scalar: 1e-6,
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn vector_components(&self, v: Vec4) -> [f32; 4] {
        v.to_array()
    }
}

/// Hyperbolic H³, Poincaré ball. Extent: `|p| <= 0.4`, where the inline tests'
/// 1e-4 holds; the Möbius chains compound, so it is a decade looser than the
/// flat trio at the same extent.
///
/// `inner` is `lambda(p)^2 * dot(u, v)` with `lambda = 2/(1 - |p|^2)`, which is
/// exactly `ConformallyFlat::conformal_factor`: the ball chart really is
/// conformal, so sourcing it there is the metric and not a coincidence.
struct HyperbolicH3Fixture;

impl SpaceFixture for HyperbolicH3Fixture {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = Iso3H;
    type S = HyperbolicH3;

    fn space(&self) -> HyperbolicH3 {
        HyperbolicH3
    }

    fn points(&self) -> Vec<Vec3> {
        ball_samples(0x0083_0F1A, 5, 0.4)
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x0083_7A46);
        (0..3)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.06)
            .collect()
    }

    fn isos(&self) -> Vec<Iso3H> {
        vec![
            Iso3H::from_translation(Vec3::new(0.15, 0.0, 0.0)),
            Iso3H::from_rotation(Quat::from_rotation_z(0.4)),
            Iso3H::from_translation(Vec3::new(-0.05, 0.2, 0.1)),
        ]
    }

    fn inner(&self, at: Vec3, u: Vec3, v: Vec3) -> f32 {
        HyperbolicH3.conformal_factor(at) * u.dot(v)
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let interior = Vec3::new(0.3, 0.0, 0.0);
        // H³ has no cut locus; its conditioning hazard is the ideal boundary.
        // The mirrored pair at |p| = 0.85 spans 5.0 hyperbolic units and is the
        // one place transport can be checked against an exactly known answer:
        // the gyration's axis `to × -from` vanishes identically for antipodal
        // arguments and the conformal ratio is a value over itself, so the
        // transported vector is bit-identical to the input on any IEEE target.
        let near_boundary = Vec3::new(0.85, 0.0, 0.0);
        // Off-axis and outside, chosen because it is one of the ~34% of
        // out-of-ball directions whose naive clamp overshoots: the scale factor
        // is two square roots, a divide and three products, and this direction
        // rounds back to `|q|² == 1.0` exactly. `gyr_apply(q, -q, v)` then has
        // `1 + a·b == 0` and a zero axis, so an unenforced clamp postcondition
        // returns NaN here. Every other out-of-domain sample above is
        // axis-aligned, which is the benign case.
        let off_axis_outside = Vec3::new(1.2, 0.9, 1.4);
        vec![
            (interior, interior),
            (interior, Vec3::new(1.0, 0.0, 0.0)),
            (interior, Vec3::new(2.0, 0.0, 0.0)),
            (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
            (near_boundary, -near_boundary),
            (off_axis_outside, off_axis_outside),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(-1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-4,
            vector: 1e-4,
            scalar: 1e-4,
            // Every declared pair on which the metric is defined transports
            // exactly, per the note on `degenerate_pairs`; 1e-6 matches the
            // flat fixtures rather than reserving room the impl does not use.
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

/// Spherical S³, upper-hemisphere chart. Extent: `|p| <= 0.4`, matching the
/// inline tests' 1e-5.
///
/// `inner` is the lifted ambient one the impl uses internally: with
/// `w = sqrt(1 - |p|^2)` and `u4 = (u, -dot(u, p)/w)`, `inner = dot(u4, v4)`.
/// Written out here rather than routed through `ConformallyFlat`, unlike H³:
/// this chart's metric is `delta_ij + p_i p_j/(1 - |p|^2)`, which is not a
/// scalar multiple of the identity, so no conformal factor can express it and
/// any that appeared would be a stereographic factor belonging to a different
/// chart. The distinction is recorded once here instead of being rediscovered
/// per test; the angle-excess item is what would catch getting it wrong.
struct SphericalS3Fixture;

impl SphericalS3Fixture {
    fn lift(p: Vec3, v: Vec3) -> Vec4 {
        let w = (1.0 - p.length_squared()).sqrt();
        Vec4::new(v.x, v.y, v.z, -v.dot(p) / w)
    }
}

impl SpaceFixture for SphericalS3Fixture {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = Iso4;
    type S = SphericalS3;

    fn space(&self) -> SphericalS3 {
        SphericalS3
    }

    fn points(&self) -> Vec<Vec3> {
        ball_samples(0x0053_0F1A, 5, 0.4)
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x0053_7A46);
        (0..3)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.15)
            .collect()
    }

    fn isos(&self) -> Vec<Iso4> {
        // Small enough that no sampled point crosses the equator, the chart's
        // documented failure mode.
        vec![
            Iso4::from_translation(Vec3::new(0.15, 0.0, 0.0)),
            Iso4::from_rotation(Quat::from_rotation_z(0.4)),
            Iso4::from_translation(Vec3::new(-0.05, 0.2, 0.1)),
        ]
    }

    fn inner(&self, at: Vec3, u: Vec3, v: Vec3) -> f32 {
        Self::lift(at, u).dot(Self::lift(at, v))
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let interior = Vec3::new(0.3, 0.0, 0.0);
        // The near-equator mirror pair is the closest the chart can drive the
        // transport denominator to zero, which is the conditioning class the
        // `degenerate` tolerance names.
        let near_equator = Vec3::new((1.0 - 2.0e-6_f32).sqrt(), 1e-3, 0.0);
        vec![
            (interior, interior),
            (interior, Vec3::new(1.0, 0.0, 0.0)),
            (interior, Vec3::new(2.0, 0.0, 0.0)),
            (
                near_equator,
                Vec3::new(-near_equator.x, near_equator.y, 0.0),
            ),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-5,
            vector: 1e-5,
            scalar: 1e-5,
            degenerate: 1e-3,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

/// Spherical S³, full ambient embedding. Extent: a cap around `+w` of angular
/// radius ~0.7 rad, so every sampled pair is well inside the injectivity
/// radius `pi`.
struct SphericalS3EmbeddedFixture;

impl SpaceFixture for SphericalS3EmbeddedFixture {
    type Point = Vec4;
    type Vector = Vec4;
    type Iso = Iso4;
    type S = SphericalS3Embedded;

    fn space(&self) -> SphericalS3Embedded {
        SphericalS3Embedded
    }

    fn points(&self) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x0054_0F1A);
        (0..5)
            .map(|_| {
                (Vec4::W
                    + Vec4::new(
                        rng.signed_unit(),
                        rng.signed_unit(),
                        rng.signed_unit(),
                        rng.signed_unit(),
                    ) * 0.4)
                    .normalize()
            })
            .collect()
    }

    fn tangents(&self, at: Vec4) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x0054_7A46);
        (0..3)
            .map(|_| {
                let raw = Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                );
                // Ambient tangents are perpendicular to their base point by
                // this impl's contract; a radial part would be projected out
                // by `exp` and the reverse round trip would be testing the
                // projection instead of the geodesic.
                (raw - raw.dot(at) * at) * 0.2
            })
            .collect()
    }

    fn isos(&self) -> Vec<Iso4> {
        vec![
            Iso4::from_translation(Vec3::new(0.3, 0.1, -0.2)),
            Iso4::from_rotation(Quat::from_rotation_z(0.4)),
            Iso4::from_translation(Vec3::new(-0.05, 0.2, 0.1)),
        ]
    }

    fn inner(&self, _at: Vec4, u: Vec4, v: Vec4) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec4, s: f32, v: Vec4, t: f32) -> Vec4 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec4, Vec4)> {
        let p = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let omega = std::f32::consts::PI - 1e-3;
        vec![
            (p, p),
            (Vec4::X, -Vec4::X),
            (
                Vec4::X,
                Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize(),
            ),
            (p, (p + Vec4::new(1e-9, 0.0, -1e-9, 0.0)).normalize()),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-5,
            vector: 1e-5,
            scalar: 1e-5,
            degenerate: 1e-3,
        }
    }

    fn point_components(&self, p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn vector_components(&self, v: Vec4) -> [f32; 4] {
        v.to_array()
    }
}

/// `BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>`, the variable-metric
/// case. Tolerances are the ones the inline integrator tests already meet; they
/// are two to three decades looser than the closed-form impls because `exp` is
/// RK4-of-32 and `log` is Gauss-Newton shooting on top of it.
///
/// Extent: `|p| <= 0.34`, chart separation `<= 0.66`, with `x` covering both
/// zone extremes twice and the blend interior twice, so the pure-A and pure-B
/// short circuits in `distance` are both exercised by a same-extreme pair. The
/// bound is measured, not guessed, and it is the tightest constraint any
/// fixture here carries: `log` is a shooting solve seeded with the chart
/// displacement (Press et al., *Numerical Recipes*, 3rd ed., 2007, §18.1), and
/// past roughly this separation the seed is far enough off in the H³-dominated
/// half that Newton walks the integrator out of the Poincaré ball instead of
/// converging. Inside the bound the worst measured `log` residual is 6.6e-6 and
/// the worst distance asymmetry 1.5e-4; just outside it, both diverge without
/// bound.
struct BlendedSpaceFixture;

impl SpaceFixture for BlendedSpaceFixture {
    type Point = Vec3;
    type Vector = Vec3;
    type Iso = ();
    type S = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

    /// `Iso = ()`: the blending field breaks translation and rotation symmetry,
    /// so this Space ships no isometries at all. Declared rather than left to
    /// pass vacuously, because a silently-vacuous invariant is exactly the hole
    /// this suite exists to close, and this one blocks face-pairing
    /// identifications on composed spaces.
    const ISOMETRY_GROUP_IS_TRIVIAL: bool = true;

    /// `parallel_transport` integrates the transport ODE along the
    /// chart-coordinate straight line, which is not this Space's geodesic; the
    /// impl names the geodesic path's cost (~7x) at that callsite. So
    /// `PT(a, b, log(a, b))` is not the forward tangent at `b` here, and the
    /// items that pin the geodesic transport everywhere else would be pinning
    /// the wrong map. The pole ladder is doubly inapplicable: a blended metric
    /// has non-parallel curvature, which is the ladder's own precondition.
    /// Declared rather than omitted: the exemption is one fixture's and not the
    /// suite's, and both gated items assert the miss is real, 4.9 and 2.0e1
    /// budgets at their worst sampled pairs.
    ///
    /// What covers this transport instead. Only one item in the suite sees a
    /// rotation about the direction of travel,
    /// `parallel_transport_matches_the_one_its_own_geodesics_imply`, and it
    /// reads that rotation off a geodesic oracle, so the exemption would
    /// otherwise leave one unpinned. Two tests in `blended.rs` cover the RK4
    /// kernel this fixture's `Space` methods call, from the other side:
    ///
    /// - `transport_is_invariant_to_how_its_own_path_is_subdivided` pins that
    ///   the kernel is a convergent discretization and that the answer does not
    ///   depend on how the path is chopped. It does not pin which connection is
    ///   integrated, and a rotation proportional to arc length passes it
    ///   unchanged: every discretization of a wrong RHS converges to the same
    ///   wrong flow.
    /// - `h3_transport_agrees_with_the_closed_form_by_a_vanishing_coefficient`
    ///   is what pins the connection, against the one source metric that has a
    ///   closed form. It divides the disagreement by the step and shrinks the
    ///   step: following the chord instead of the geodesic costs O(h³), so its
    ///   coefficient vanishes, while a wrong connection contributes a term
    ///   linear in h whose coefficient does not. That separation is what an
    ///   absolute budget on one short segment cannot make.
    const TRANSPORT_FOLLOWS_THE_GEODESIC: bool = false;

    fn space(&self) -> Self::S {
        BlendedSpace::new(EuclideanR3, HyperbolicH3, LinearBlendX::new(-0.15, 0.15))
    }

    fn points(&self) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00B1_0F1A);
        (0..6)
            .map(|i| {
                // x is swept deterministically rather than sampled, so which
                // branch of the blend each point lands in is a property of the
                // fixture and not of the seed.
                let x = -0.32 + 0.128 * i as f32;
                Vec3::new(x, rng.signed_unit() * 0.08, rng.signed_unit() * 0.08)
            })
            .collect()
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00B1_7A46);
        (0..2)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.05)
            .collect()
    }

    fn isos(&self) -> Vec<()> {
        vec![()]
    }

    fn inner(&self, at: Vec3, u: Vec3, v: Vec3) -> f32 {
        self.space().conformal_factor(at) * u.dot(v)
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        // Both endpoints stay inside the Poincaré ball: outside it the H³
        // source's conformal factor is infinite, which `BlendedSpace` treats as
        // a caller error (debug assertion), not as a degenerate input to
        // survive.
        let interior = Vec3::new(0.1, 0.0, 0.0);
        vec![
            (interior, interior),
            (Vec3::new(-0.32, 0.0, 0.0), Vec3::new(0.32, 0.0, 0.0)),
            (Vec3::new(0.3, 0.08, 0.08), Vec3::new(0.3, 0.08, 0.08)),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        None
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 5e-4,
            vector: 5e-3,
            scalar: 5e-3,
            degenerate: 5e-3,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

conformance_suite!(euclidean_r2, EuclideanR2Fixture);
conformance_suite!(euclidean_r3, EuclideanR3Fixture);
conformance_suite!(euclidean_r4, EuclideanR4Fixture);
conformance_suite!(hyperbolic_h3, HyperbolicH3Fixture);
conformance_suite!(spherical_s3, SphericalS3Fixture);
conformance_suite!(spherical_s3_embedded, SphericalS3EmbeddedFixture);
conformance_suite!(blended_space, BlendedSpaceFixture);
