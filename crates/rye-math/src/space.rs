//! The [`Space`] trait, Rye's interface to geometry.
//!
//! A `Space` is a Riemannian manifold equipped with an isometry group. Every concrete geometry
//! (Euclidean R³, hyperbolic H³, spherical S³, Euclidean R⁴, …) implements this trait, and
//! every other system in the engine that needs to know *what space means* takes `S: Space` as a
//! generic.
//!
//! The GPU-facing half of the contract lives on [`WgslSpace`], a separate subtrait; see its docs
//! for the rationale.
//!
//! ## Why methods take `&self`
//!
//! Stateless geometries like [`crate::EuclideanR3`] carry no data and monomorphize to direct
//! calls. Parametric geometries (hyperbolic with a curvature scalar, spherical with a radius)
//! need somewhere to store that parameter. `&self` gives them that without forcing stateless
//! impls to pay a runtime cost.
//!
//! ## Tangent vectors carry an implicit base point
//!
//! `Self::Vector` is a tangent vector at *some* point on the manifold; the trait does not enforce
//! *which* point. Callers tracking tangent vectors outside tight numerical kernels should prefer
//! [`crate::Tangent`], which bundles the base point with the vector and exposes safe transport /
//! exponential / scale helpers.

use std::borrow::Cow;

/// A Riemannian manifold with a transitive isometry group.
///
/// Implementors describe a geometry by exposing its point set, tangent vectors, and isometries.
/// Shader integration lives on [`WgslSpace`].
///
/// All methods must be deterministic and side-effect-free.
pub trait Space {
    /// A point on the manifold.
    type Point: Copy + Send + Sync + 'static;
    /// A tangent vector at *some* point. The base point is tracked by the caller, not the
    /// type; use [`crate::Tangent`] when that tracking should be enforced.
    type Vector: Copy + Send + Sync + 'static;
    /// An orientation-preserving isometry of the manifold.
    type Iso: Copy + Send + Sync + 'static;

    // ---- Riemannian primitives ----------------------------------------

    /// Geodesic distance between two points.
    fn distance(&self, a: Self::Point, b: Self::Point) -> f32;

    /// Exponential map: travel from `at` along the geodesic with initial velocity `v` for
    /// unit time. Inverse of [`Self::log`].
    fn exp(&self, at: Self::Point, v: Self::Vector) -> Self::Point;

    /// Logarithm map: the tangent vector at `from` whose [`Self::exp`] reaches `to`. Inverse
    /// of [`Self::exp`]. Undefined if `to` is in the cut locus of `from` (e.g. antipode on a
    /// sphere); impls should document their handling.
    fn log(&self, from: Self::Point, to: Self::Point) -> Self::Vector;

    /// Parallel-transport `v` (a tangent vector at `from`) to `to`, returning the
    /// corresponding tangent vector at `to`.
    ///
    /// **Implementations choose the path.** Parallel transport is path-dependent in any non-flat
    /// geometry, and this signature does not name a path. Closed-form constant-curvature Spaces
    /// (E³, H³, S³ away from antipodes, E⁴) transport along the unique minimizing geodesic.
    /// Variable-metric Spaces (`BlendedSpace`, future numerical-only Spaces) may transport along
    /// a cheaper path, typically the chart-coordinate straight line, when the cost of solving
    /// the geodesic boundary-value problem outweighs the accuracy gain. Each implementation
    /// documents its choice.
    ///
    /// Callers that need transport along a *specific* path (a camera's polyline over the last few
    /// frames, a player's integrated trajectory) should call [`Self::parallel_transport_along`]
    /// with the polyline explicitly. That avoids both the path-uniqueness ambiguity and any
    /// geodesic boundary-value problem.
    fn parallel_transport(
        &self,
        from: Self::Point,
        to: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;

    /// Parallel-transport `v` along the piecewise path through the listed points, segment by
    /// segment. Returns the transported vector at the final point.
    ///
    /// **Why this is the path-aware primitive.** Parallel transport is path-dependent in any
    /// non-flat geometry; "transport from `a` to `b`" is only well-defined once a path is chosen.
    /// This method lets the caller pin down the *macro* path (the polyline a camera or player
    /// traversed over the last few frames), so the integrated result no longer depends on
    /// whether the caller batched the journey into one [`Self::parallel_transport`] call or
    /// many. Per-segment path choice still belongs to the implementation (see the default note
    /// below); the contract is that finer subdivision converges to the true parallel transport
    /// along the polyline as segment lengths shrink.
    ///
    /// Edge cases:
    /// - `path.len() < 2`: returns `v` unchanged.
    /// - Consecutive duplicate points contribute identity transports.
    ///
    /// The default implementation chains [`Self::parallel_transport`] over consecutive pairs,
    /// which inherits each Space's per-segment path choice. Variable-metric Spaces with expensive
    /// geodesic construction may override this to integrate the polyline directly without
    /// rebuilding per-segment geodesics.
    fn parallel_transport_along(&self, path: &[Self::Point], v: Self::Vector) -> Self::Vector {
        let mut current = v;
        for w in path.windows(2) {
            current = self.parallel_transport(w[0], w[1], current);
        }
        current
    }

    // ---- Isometry group -----------------------------------------------

    /// The identity isometry.
    fn iso_identity(&self) -> Self::Iso;

    /// `a ∘ b`, apply `b` first, then `a`.
    fn iso_compose(&self, a: Self::Iso, b: Self::Iso) -> Self::Iso;

    /// Inverse isometry: `iso_compose(a, iso_inverse(a)) == iso_identity()`.
    fn iso_inverse(&self, a: Self::Iso) -> Self::Iso;

    /// Apply an isometry to a point.
    fn iso_apply(&self, iso: Self::Iso, p: Self::Point) -> Self::Point;

    /// Apply an isometry's differential to a tangent vector at `at`. The result is a tangent
    /// vector at `iso_apply(iso, at)`.
    fn iso_transport(&self, iso: Self::Iso, at: Self::Point, v: Self::Vector) -> Self::Vector;
}

/// A [`Space`] that additionally exposes its primitives as WGSL, for inlining into shaders by
/// `rye-shader`.
///
/// Split from [`Space`] deliberately. The math trait is the most stable surface in the engine;
/// the shader ABI is the least, it will evolve with every render-graph revision. Coupling them
/// would marry their release cadences and break Rye's stability-discipline goal.
///
/// Splitting also lets CPU-only consumers (determinism validators, dedicated servers, offline
/// tools) depend on `rye-math` without being forced to know anything about WGSL.
pub trait WgslSpace: Space {
    /// WGSL source providing this space's primitives, for inlining into shaders by
    /// `rye-shader`. The v0 shader ABI is deliberately tiny and single-space:
    ///
    /// ```wgsl
    /// fn rye_distance(a: vec3<f32>, b: vec3<f32>) -> f32
    /// fn rye_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// fn rye_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32>
    /// fn rye_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32>
    /// ```
    ///
    /// This ABI intentionally covers only spaces whose point/vector shader representation is
    /// `vec3<f32>`. `Iso` layout and multi-space name mangling are left out until a render path
    /// actually needs them.
    ///
    /// Stateless geometries return `Cow::Borrowed`; parametric ones may `format!` constants in and
    /// return `Cow::Owned`.
    fn wgsl_impl(&self) -> Cow<'static, str>;

    /// Whether this Space's chart is globally flat: chart-coord arithmetic (raw
    /// `dot(p, n) - offset`, axis-aligned `abs(p) - b`, etc.) computes the correct geometry
    /// without going through the Riemannian `rye_*` machinery. False for curved Spaces (Poincaré
    /// ball H³, stereographic S³, `BlendedSpace`) where chart arithmetic and geodesic distance
    /// diverge visibly.
    ///
    /// Defaults to `false` so a new Space implementor must explicitly opt in to chart-coord SDF
    /// fast paths. Consumers of this flag (e.g. `rye_sdf::Primitive::HalfSpace::to_wgsl`) emit
    /// the chart-coord formula when `true`, sentinel otherwise.
    fn is_chart_flat(&self) -> bool {
        false
    }
}
